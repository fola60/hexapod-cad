import argparse
import json
import random
import time
from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium.vector import AutoresetMode
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

from hexapod_env import INFO_KEYS, HexapodEnv, load_config

SIM_DIR = Path(__file__).resolve().parent


def build_parser(description="PPO with asymmetric actor-critic for the hexapod env"):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default=str(SIM_DIR / "configs" / "crab.json"))
    p.add_argument("--run-name", default=None)
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--torch-deterministic", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--cuda", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--torch-threads", type=int, default=4)
    p.add_argument("--randomize", action="store_true")
    p.add_argument("--total-timesteps", type=int, default=50_000_000)
    p.add_argument("--num-envs", type=int, default=32)
    p.add_argument("--num-steps", type=int, default=48)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--desired-kl", type=float, default=0.01)
    p.add_argument("--lr-min", type=float, default=1e-5)
    p.add_argument("--lr-max", type=float, default=1e-2)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--num-minibatches", type=int, default=4)
    p.add_argument("--update-epochs", type=int, default=5)
    p.add_argument("--norm-adv", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--clip-coef", type=float, default=0.2)
    p.add_argument("--clip-vloss", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--ent-coef", type=float, default=0.005)
    p.add_argument("--vf-coef", type=float, default=1.0)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--hidden-sizes", type=int, nargs="+", default=[512, 256, 128])
    p.add_argument("--init-std", type=float, default=0.5)
    p.add_argument("--bound-coef", type=float, default=1.0)
    p.add_argument("--action-bound", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--eval-episodes", type=int, default=5)
    p.add_argument("--print-every", type=int, default=10)
    p.add_argument("--init-checkpoint", default=None)
    p.add_argument("--init-optimizer", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--continue-steps", action=argparse.BooleanOptionalAction, default=True)
    return p


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def mlp(in_dim, hidden, out_dim, out_std):
    layers, d = [], in_dim
    for h in hidden:
        layers += [layer_init(nn.Linear(d, h)), nn.ELU()]
        d = h
    layers.append(layer_init(nn.Linear(d, out_dim), std=out_std))
    return nn.Sequential(*layers)


class Agent(nn.Module):
    def __init__(self, obs_dim, actor_obs_dim, action_dim, hidden_sizes, init_std=1.0):
        super().__init__()
        self.actor_obs_dim = actor_obs_dim
        self.critic = mlp(obs_dim, hidden_sizes, 1, 1.0)
        self.actor_mean = mlp(actor_obs_dim, hidden_sizes, action_dim, 0.01)
        self.actor_logstd = nn.Parameter(torch.full((1, action_dim), float(np.log(init_std))))

    def get_value(self, x):
        return self.critic(x)

    def act_mean(self, x):
        return self.actor_mean(x[..., : self.actor_obs_dim])

    def get_action_and_value(self, x, action=None):
        mean = self.act_mean(x)
        std = torch.exp(self.actor_logstd).expand_as(mean)
        dist = Normal(mean, std)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action).sum(1), dist.entropy().sum(1), self.critic(x)

    def evaluate_actions(self, x, action):
        mean = self.act_mean(x)
        dist = Normal(mean, torch.exp(self.actor_logstd).expand_as(mean))
        return dist.log_prob(action).sum(1), dist.entropy().sum(1), self.critic(x), mean


def compute_gae(rewards, values, dones, next_value, next_done, gamma, lam):
    num_steps = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    lastgaelam = 0
    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            nextnonterminal = 1.0 - next_done
            nextvalues = next_value
        else:
            nextnonterminal = 1.0 - dones[t + 1]
            nextvalues = values[t + 1]
        delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
        advantages[t] = lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam
    return advantages, advantages + values


def ppo_update(agent, optimizer, b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values, args):
    batch_size = b_obs.shape[0]
    minibatch_size = batch_size // args.num_minibatches
    clipfracs, out_fracs = [], []
    for _ in range(args.update_epochs):
        b_inds = torch.randperm(batch_size, device=b_obs.device)
        for start in range(0, batch_size, minibatch_size):
            mb_inds = b_inds[start : start + minibatch_size]
            newlogprob, entropy, newvalue, mu = agent.evaluate_actions(b_obs[mb_inds], b_actions[mb_inds])
            logratio = newlogprob - b_logprobs[mb_inds]
            ratio = logratio.exp()
            with torch.no_grad():
                clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean())
                out_fracs.append((mu.abs() > args.action_bound).float().mean())

            mb_adv = b_advantages[mb_inds]
            if args.norm_adv:
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

            pg_loss1 = -mb_adv * ratio
            pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            newvalue = newvalue.view(-1)
            if args.clip_vloss:
                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(newvalue - b_values[mb_inds], -args.clip_coef, args.clip_coef)
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
            else:
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

            entropy_loss = entropy.mean()
            b = args.action_bound
            bound_loss = (torch.relu(mu - b) ** 2 + torch.relu(-b - mu) ** 2).sum(1).mean()
            loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef + args.bound_coef * bound_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
            optimizer.step()
    return {
        "value_loss": v_loss.item(),
        "policy_loss": pg_loss.item(),
        "entropy": entropy_loss.item(),
        "clipfrac": torch.stack(clipfracs).mean().item(),
        "bound_loss": bound_loss.item(),
        "mean_action_out_of_range": torch.stack(out_fracs).mean().item(),
    }


def adapt_lr(agent, optimizer, lr, b_obs, b_actions, b_logprobs, args):
    with torch.no_grad():
        _, newlogprob, _, _ = agent.get_action_and_value(b_obs, b_actions)
        logratio = newlogprob - b_logprobs
        approx_kl = ((logratio.exp() - 1) - logratio).mean().item()
    if approx_kl > 2.0 * args.desired_kl:
        lr = max(lr / 1.5, args.lr_min)
    elif approx_kl < args.desired_kl / 2.0:
        lr = min(lr * 1.5, args.lr_max)
    for g in optimizer.param_groups:
        g["lr"] = lr
    return lr, approx_kl


def explained_variance(values, returns):
    y_pred, y_true = values.cpu().numpy(), returns.cpu().numpy()
    var_y = np.var(y_true)
    return float("nan") if var_y == 0 else float(1 - np.var(y_true - y_pred) / var_y)


def log_update(writer, global_step, lr, approx_kl, action_std, sps, stats, ev, info_means):
    writer.add_scalar("charts/learning_rate", lr, global_step)
    writer.add_scalar("charts/action_std", action_std, global_step)
    writer.add_scalar("charts/SPS", sps, global_step)
    writer.add_scalar("losses/approx_kl", approx_kl, global_step)
    writer.add_scalar("losses/explained_variance", ev, global_step)
    for k, v in stats.items():
        writer.add_scalar(f"losses/{k}", v, global_step)
    for k, v in info_means.items():
        group = "rewards" if k.startswith("rew_") else "tracking"
        writer.add_scalar(f"{group}/{k}", v, global_step)


def make_env(cfg, cfg_dir, randomize):
    def thunk():
        return HexapodEnv(cfg, config_dir=cfg_dir, randomize=randomize)

    return thunk


def evaluate(agent, env, episodes, seed, device):
    rets, lens, falls, lin_errs, yaw_errs = [], [], [], [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        ret, n, lin_e, yaw_e = 0.0, 0, 0.0, 0.0
        while True:
            with torch.no_grad():
                a = agent.act_mean(torch.as_tensor(obs, device=device).unsqueeze(0)).cpu().numpy()[0]
            obs, r, term, trunc, info = env.step(a)
            ret += r
            n += 1
            lin_e += info["lin_vel_error"]
            yaw_e += info["yaw_error"]
            if term or trunc:
                break
        rets.append(ret)
        lens.append(n)
        falls.append(float(term))
        lin_errs.append(lin_e / n)
        yaw_errs.append(yaw_e / n)
    return {
        "return": float(np.mean(rets)),
        "episode_length": float(np.mean(lens)),
        "fall_rate": float(np.mean(falls)),
        "lin_vel_error": float(np.mean(lin_errs)),
        "yaw_error": float(np.mean(yaw_errs)),
    }


def save_checkpoint(path, agent, args, cfg, cfg_dir, env, iteration, global_step, eval_return, optimizer=None, lr=None):
    torch.save(
        {
            "agent": agent.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "lr": lr,
            "obs_dim": env.obs_dim,
            "actor_obs_dim": env.actor_obs_dim,
            "action_dim": env.nu,
            "hidden_sizes": list(args.hidden_sizes),
            "init_std": args.init_std,
            "config": cfg,
            "config_dir": str(cfg_dir),
            "iteration": iteration,
            "global_step": global_step,
            "eval_return": eval_return,
            "args": vars(args),
        },
        path,
    )


def run_eval(agent, eval_env, args, writer, run_dir, cfg, cfg_dir, iteration, global_step, best_eval, device, optimizer=None, lr=None):
    ev = evaluate(agent, eval_env, args.eval_episodes, args.seed + 100_000, device)
    for k, v in ev.items():
        writer.add_scalar(f"eval/{k}", v, global_step)
    print(
        f"eval it {iteration}: return {ev['return']:.2f} len {ev['episode_length']:.0f} "
        f"fall {ev['fall_rate']:.2f} lin_err {ev['lin_vel_error']:.3f} yaw_err {ev['yaw_error']:.3f}",
        flush=True,
    )
    save_checkpoint(run_dir / "latest.pt", agent, args, cfg, cfg_dir, eval_env, iteration, global_step, ev["return"], optimizer, lr)
    if ev["return"] > best_eval:
        best_eval = ev["return"]
        save_checkpoint(run_dir / "best.pt", agent, args, cfg, cfg_dir, eval_env, iteration, global_step, ev["return"], optimizer, lr)
    return best_eval


def read_init_checkpoint(args):
    if not args.init_checkpoint:
        return None
    ckpt = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
    if list(ckpt["hidden_sizes"]) != list(args.hidden_sizes):
        print(f"using hidden sizes {ckpt['hidden_sizes']} from {args.init_checkpoint}", flush=True)
        args.hidden_sizes = list(ckpt["hidden_sizes"])
    return ckpt


def apply_init_checkpoint(ckpt, agent, optimizer, args, obs_dim, actor_obs_dim, action_dim):
    lr = args.learning_rate
    if ckpt is None:
        return lr
    dims = (ckpt["obs_dim"], ckpt["actor_obs_dim"], ckpt["action_dim"])
    if dims != (obs_dim, actor_obs_dim, action_dim):
        raise ValueError(f"checkpoint dims (obs, actor obs, action) {dims} do not match env {(obs_dim, actor_obs_dim, action_dim)}")
    agent.load_state_dict(ckpt["agent"])
    if args.init_optimizer:
        if ckpt.get("optimizer") is None:
            raise ValueError(f"{args.init_checkpoint} has no optimizer state; run without --init-optimizer")
        optimizer.load_state_dict(ckpt["optimizer"])
        lr = ckpt["lr"] if ckpt.get("lr") is not None else optimizer.param_groups[0]["lr"]
    for g in optimizer.param_groups:
        g["lr"] = lr
    std = torch.exp(agent.actor_logstd).mean().item()
    print(
        f"initialised from {args.init_checkpoint} (iteration {ckpt.get('iteration')}, eval return {ckpt.get('eval_return')}, "
        f"std {std:.3f}, lr {lr:.2e}, optimizer {'restored' if args.init_optimizer else 'fresh'})",
        flush=True,
    )
    return lr


def start_point(ckpt, args, batch_size):
    if ckpt is None or not args.continue_steps:
        return 0, 0, args.total_timesteps // batch_size
    step, it = int(ckpt["global_step"]), int(ckpt["iteration"])
    remaining = args.total_timesteps - step
    if remaining < batch_size:
        raise ValueError(
            f"--total-timesteps {args.total_timesteps} leaves nothing to train: the checkpoint is already at step {step}. "
            f"Raise it, or pass --no-continue-steps to count from zero"
        )
    num_iterations = it + remaining // batch_size
    print(f"continuing from step {step} (iteration {it}): {remaining // batch_size * batch_size} more steps, up to iteration {num_iterations}", flush=True)
    return step, it, num_iterations


def setup_run(args):
    run_name = args.run_name or f"crab_{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir = Path(args.runs_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg, cfg_dir = load_config(args.config)
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    (run_dir / "args.json").write_text(json.dumps(vars(args), indent=2))
    writer = SummaryWriter(str(run_dir))
    writer.add_text("hyperparameters", "|param|value|\n|-|-|\n" + "\n".join(f"|{k}|{v}|" for k, v in vars(args).items()))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    return run_dir, cfg, cfg_dir, writer


def main():
    args = build_parser().parse_args()
    init_ckpt = read_init_checkpoint(args)
    batch_size = args.num_envs * args.num_steps
    start_step, start_iteration, num_iterations = start_point(init_ckpt, args, batch_size)
    run_dir, cfg, cfg_dir, writer = setup_run(args)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = gym.vector.AsyncVectorEnv(
        [make_env(cfg, cfg_dir, args.randomize) for _ in range(args.num_envs)],
        autoreset_mode=AutoresetMode.SAME_STEP,
    )
    eval_env = HexapodEnv(cfg, config_dir=cfg_dir, randomize=args.randomize)
    obs_dim, actor_obs_dim, action_dim = eval_env.obs_dim, eval_env.actor_obs_dim, eval_env.nu

    agent = Agent(obs_dim, actor_obs_dim, action_dim, args.hidden_sizes, args.init_std).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    lr = apply_init_checkpoint(init_ckpt, agent, optimizer, args, obs_dim, actor_obs_dim, action_dim)

    obs = torch.zeros((args.num_steps, args.num_envs, obs_dim), device=device)
    actions = torch.zeros((args.num_steps, args.num_envs, action_dim), device=device)
    logprobs = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), device=device)
    values = torch.zeros((args.num_steps, args.num_envs), device=device)

    global_step = start_step
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.as_tensor(next_obs, device=device)
    next_done = torch.zeros(args.num_envs, device=device)

    ep_ret = np.zeros(args.num_envs)
    ep_len = np.zeros(args.num_envs, dtype=np.int64)
    recent_ret, recent_len, recent_fall = deque(maxlen=100), deque(maxlen=100), deque(maxlen=100)
    best_eval = -np.inf

    for iteration in range(start_iteration + 1, num_iterations + 1):
        info_sums = {k: 0.0 for k in INFO_KEYS}

        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            nobs, reward, term, trunc, infos = envs.step(action.cpu().numpy())
            raw_reward = np.asarray(reward, dtype=np.float64)
            reward = raw_reward.astype(np.float32).copy()
            done = np.logical_or(term, trunc)

            boot = np.logical_and(trunc, np.logical_not(term))
            if boot.any():
                final_obs = np.stack(infos["final_obs"][boot]).astype(np.float32)
                with torch.no_grad():
                    v = agent.get_value(torch.as_tensor(final_obs, device=device)).flatten().cpu().numpy()
                reward[boot] += args.gamma * v

            fmask = infos.get("_final_info", np.zeros(args.num_envs, dtype=bool))
            finfo = infos.get("final_info", {})
            for k in INFO_KEYS:
                vals = np.asarray(infos[k], dtype=np.float64).copy()
                if fmask.any():
                    vals[fmask] = np.asarray(finfo[k], dtype=np.float64)[fmask]
                info_sums[k] += float(vals.sum())

            ep_ret += raw_reward
            ep_len += 1
            for i in np.flatnonzero(done):
                recent_ret.append(ep_ret[i])
                recent_len.append(ep_len[i])
                recent_fall.append(float(term[i]))
                writer.add_scalar("charts/episodic_return", ep_ret[i], global_step)
                writer.add_scalar("charts/episodic_length", ep_len[i], global_step)
                ep_ret[i] = 0.0
                ep_len[i] = 0

            rewards[step] = torch.as_tensor(reward, device=device)
            next_obs = torch.as_tensor(nobs, device=device)
            next_done = torch.as_tensor(done, dtype=torch.float32, device=device)

        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages, returns = compute_gae(rewards, values, dones, next_value, next_done, args.gamma, args.gae_lambda)

        b_obs = obs.reshape((-1, obs_dim))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1, action_dim))
        b_values = values.reshape(-1)
        b_returns = returns.reshape(-1)
        stats = ppo_update(agent, optimizer, b_obs, b_logprobs, b_actions, advantages.reshape(-1), b_returns, b_values, args)
        lr, approx_kl = adapt_lr(agent, optimizer, lr, b_obs, b_actions, b_logprobs, args)

        sps = int((global_step - start_step) / (time.time() - start_time))
        action_std = torch.exp(agent.actor_logstd).mean().item()
        info_means = {k: v / batch_size for k, v in info_sums.items()}
        log_update(writer, global_step, lr, approx_kl, action_std, sps, stats, explained_variance(b_values, b_returns), info_means)
        if recent_ret:
            writer.add_scalar("charts/mean_return_100", float(np.mean(recent_ret)), global_step)
            writer.add_scalar("charts/mean_length_100", float(np.mean(recent_len)), global_step)
            writer.add_scalar("charts/fall_rate_100", float(np.mean(recent_fall)), global_step)

        if iteration % args.print_every == 0 or iteration == start_iteration + 1:
            mr = float(np.mean(recent_ret)) if recent_ret else float("nan")
            ml = float(np.mean(recent_len)) if recent_len else float("nan")
            print(
                f"it {iteration}/{num_iterations} step {global_step} sps {sps} ret100 {mr:.2f} len100 {ml:.0f} "
                f"lin_err {info_means['lin_vel_error']:.3f} yaw_err {info_means['yaw_error']:.3f} "
                f"std {action_std:.3f} out {stats['mean_action_out_of_range']:.2f} lr {lr:.2e} kl {approx_kl:.4f}",
                flush=True,
            )

        if iteration % args.eval_every == 0 or iteration == num_iterations:
            best_eval = run_eval(agent, eval_env, args, writer, run_dir, cfg, cfg_dir, iteration, global_step, best_eval, device, optimizer, lr)

    envs.close()
    eval_env.close()
    writer.close()


if __name__ == "__main__":
    main()
