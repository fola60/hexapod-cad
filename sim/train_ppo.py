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

from hexapod_env import REWARD_TERMS, HexapodEnv, load_config

SIM_DIR = Path(__file__).resolve().parent
INFO_KEYS = [f"rew_{k}" for k in REWARD_TERMS] + ["rew_termination", "lin_vel_error", "yaw_error"]


def parse_args():
    p = argparse.ArgumentParser(description="PPO with asymmetric actor-critic for the hexapod env")
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
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=1.0)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--hidden-sizes", type=int, nargs="+", default=[512, 256, 128])
    p.add_argument("--init-std", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--eval-episodes", type=int, default=5)
    p.add_argument("--print-every", type=int, default=10)
    return p.parse_args()


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


def save_checkpoint(path, agent, args, cfg, cfg_dir, env, iteration, global_step, eval_return):
    torch.save(
        {
            "agent": agent.state_dict(),
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


def main():
    args = parse_args()
    batch_size = args.num_envs * args.num_steps
    minibatch_size = batch_size // args.num_minibatches
    num_iterations = args.total_timesteps // batch_size
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
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = gym.vector.AsyncVectorEnv(
        [make_env(cfg, cfg_dir, args.randomize) for _ in range(args.num_envs)],
        autoreset_mode=AutoresetMode.SAME_STEP,
    )
    eval_env = HexapodEnv(cfg, config_dir=cfg_dir, randomize=args.randomize)
    obs_dim, actor_obs_dim, action_dim = eval_env.obs_dim, eval_env.actor_obs_dim, eval_env.nu

    agent = Agent(obs_dim, actor_obs_dim, action_dim, args.hidden_sizes, args.init_std).to(device)
    lr = args.learning_rate
    optimizer = optim.Adam(agent.parameters(), lr=lr, eps=1e-5)

    obs = torch.zeros((args.num_steps, args.num_envs, obs_dim), device=device)
    actions = torch.zeros((args.num_steps, args.num_envs, action_dim), device=device)
    logprobs = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), device=device)
    values = torch.zeros((args.num_steps, args.num_envs), device=device)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.as_tensor(next_obs, device=device)
    next_done = torch.zeros(args.num_envs, device=device)

    ep_ret = np.zeros(args.num_envs)
    ep_len = np.zeros(args.num_envs, dtype=np.int64)
    recent_ret, recent_len, recent_fall = deque(maxlen=100), deque(maxlen=100), deque(maxlen=100)
    best_eval = -np.inf

    for iteration in range(1, num_iterations + 1):
        info_sums = {k: 0.0 for k in INFO_KEYS}
        n_finished = 0

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
                n_finished += 1

            rewards[step] = torch.as_tensor(reward, device=device)
            next_obs = torch.as_tensor(nobs, device=device)
            next_done = torch.as_tensor(done, dtype=torch.float32, device=device)

        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        b_obs = obs.reshape((-1, obs_dim))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1, action_dim))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        b_inds = np.arange(batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                mb_inds = b_inds[start : start + minibatch_size]
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                with torch.no_grad():
                    clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())

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
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

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

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        sps = int(global_step / (time.time() - start_time))
        action_std = torch.exp(agent.actor_logstd).mean().item()
        n_samples = args.num_envs * args.num_steps

        writer.add_scalar("charts/learning_rate", lr, global_step)
        writer.add_scalar("charts/action_std", action_std, global_step)
        writer.add_scalar("charts/SPS", sps, global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl, global_step)
        writer.add_scalar("losses/clipfrac", float(np.mean(clipfracs)), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        for k in INFO_KEYS:
            group = "rewards" if k.startswith("rew_") else "tracking"
            writer.add_scalar(f"{group}/{k}", info_sums[k] / n_samples, global_step)
        if recent_ret:
            writer.add_scalar("charts/mean_return_100", float(np.mean(recent_ret)), global_step)
            writer.add_scalar("charts/mean_length_100", float(np.mean(recent_len)), global_step)
            writer.add_scalar("charts/fall_rate_100", float(np.mean(recent_fall)), global_step)

        if iteration % args.print_every == 0 or iteration == 1:
            mr = float(np.mean(recent_ret)) if recent_ret else float("nan")
            ml = float(np.mean(recent_len)) if recent_len else float("nan")
            print(
                f"it {iteration}/{num_iterations} step {global_step} sps {sps} ret100 {mr:.2f} len100 {ml:.0f} "
                f"lin_err {info_sums['lin_vel_error'] / n_samples:.3f} yaw_err {info_sums['yaw_error'] / n_samples:.3f} "
                f"std {action_std:.3f} lr {lr:.2e} kl {approx_kl:.4f}",
                flush=True,
            )

        if iteration % args.eval_every == 0 or iteration == num_iterations:
            ev = evaluate(agent, eval_env, args.eval_episodes, args.seed + 100_000, device)
            for k, v in ev.items():
                writer.add_scalar(f"eval/{k}", v, global_step)
            print(
                f"eval it {iteration}: return {ev['return']:.2f} len {ev['episode_length']:.0f} "
                f"fall {ev['fall_rate']:.2f} lin_err {ev['lin_vel_error']:.3f} yaw_err {ev['yaw_error']:.3f}",
                flush=True,
            )
            save_checkpoint(run_dir / "latest.pt", agent, args, cfg, cfg_dir, eval_env, iteration, global_step, ev["return"])
            if ev["return"] > best_eval:
                best_eval = ev["return"]
                save_checkpoint(run_dir / "best.pt", agent, args, cfg, cfg_dir, eval_env, iteration, global_step, ev["return"])

    envs.close()
    eval_env.close()
    writer.close()


if __name__ == "__main__":
    main()
