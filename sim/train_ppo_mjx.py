import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import time

import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.optim as optim

from hexapod_env import INFO_KEYS, HexapodEnv
from mjx_env import MJXHexapodEnv
from train_ppo import (
    Agent,
    adapt_lr,
    build_parser,
    compute_gae,
    explained_variance,
    log_update,
    ppo_update,
    run_eval,
    setup_run,
)


def j2t(x):
    return torch.from_dlpack(x)


def t2j(x):
    return jax.dlpack.from_dlpack(x.detach().contiguous())


def main():
    p = build_parser("PPO with asymmetric actor-critic, MJX physics on the GPU")
    p.set_defaults(num_envs=4096, num_steps=24, total_timesteps=200_000_000, eval_every=25, print_every=5)
    args = p.parse_args()
    batch_size = args.num_envs * args.num_steps
    num_iterations = args.total_timesteps // batch_size
    run_dir, cfg, cfg_dir, writer = setup_run(args)

    backend = jax.default_backend()
    if backend == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError("JAX is on the GPU but torch has no CUDA; install a CUDA build of torch")
        device = torch.device("cuda")
    else:
        print(f"warning: JAX backend is {backend}, running on CPU", flush=True)
        device = torch.device("cpu")

    env = MJXHexapodEnv(cfg, config_dir=cfg_dir, num_envs=args.num_envs, randomize=args.randomize)
    eval_env = HexapodEnv(cfg, config_dir=cfg_dir, randomize=args.randomize)
    obs_dim, actor_obs_dim, action_dim = env.obs_dim, env.actor_obs_dim, env.nu

    agent = Agent(obs_dim, actor_obs_dim, action_dim, args.hidden_sizes, args.init_std).to(device)
    lr = args.learning_rate
    optimizer = optim.Adam(agent.parameters(), lr=lr, eps=1e-5)

    obs = torch.zeros((args.num_steps, args.num_envs, obs_dim), device=device)
    actions = torch.zeros((args.num_steps, args.num_envs, action_dim), device=device)
    logprobs = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), device=device)
    values = torch.zeros((args.num_steps, args.num_envs), device=device)

    print("compiling MJX reset and step", flush=True)
    t0 = time.time()
    state, obs_j = env.reset(jax.random.PRNGKey(args.seed))
    warm = env.step(state, jnp.zeros((args.num_envs, action_dim), dtype=jnp.float32))
    jax.block_until_ready(warm)
    del warm
    print(f"compiled in {time.time() - t0:.1f} s", flush=True)

    next_obs = j2t(obs_j).clone()
    next_done = torch.zeros(args.num_envs, device=device)
    ep_ret = torch.zeros(args.num_envs, device=device)
    ep_len = torch.zeros(args.num_envs, device=device)
    best_eval = -np.inf
    global_step = 0
    start_time = time.time()

    for iteration in range(1, num_iterations + 1):
        info_sum = torch.zeros(len(INFO_KEYS), device=device)
        fin = torch.zeros(4, device=device)

        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            state, obs_j, final_obs_j, rew_j, term_j, trunc_j, info_j = env.step(state, t2j(action))
            raw_reward = j2t(rew_j).clone()
            term = j2t(term_j).clone()
            trunc = j2t(trunc_j).clone()
            done = torch.clamp(term + trunc, max=1.0)

            with torch.no_grad():
                v_final = agent.get_value(j2t(final_obs_j)).flatten()
            rewards[step] = raw_reward + args.gamma * v_final * trunc * (1.0 - term)
            info_sum += j2t(info_j).sum(0)

            ep_ret += raw_reward
            ep_len += 1
            fin += torch.stack([done.sum(), (ep_ret * done).sum(), (ep_len * done).sum(), term.sum()])
            ep_ret *= 1.0 - done
            ep_len *= 1.0 - done

            next_obs = j2t(obs_j).clone()
            next_done = done

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

        sps = int(global_step / (time.time() - start_time))
        action_std = torch.exp(agent.actor_logstd).mean().item()
        info_means = dict(zip(INFO_KEYS, (info_sum / batch_size).tolist()))
        log_update(writer, global_step, lr, approx_kl, action_std, sps, stats, explained_variance(b_values, b_returns), info_means)
        n_done, ret_sum, len_sum, falls = fin.tolist()
        if n_done > 0:
            writer.add_scalar("charts/mean_return", ret_sum / n_done, global_step)
            writer.add_scalar("charts/mean_length", len_sum / n_done, global_step)
            writer.add_scalar("charts/fall_rate", falls / n_done, global_step)

        if iteration % args.print_every == 0 or iteration == 1:
            mr = ret_sum / n_done if n_done else float("nan")
            ml = len_sum / n_done if n_done else float("nan")
            print(
                f"it {iteration}/{num_iterations} step {global_step} sps {sps} ret {mr:.2f} len {ml:.0f} episodes {int(n_done)} "
                f"lin_err {info_means['lin_vel_error']:.3f} yaw_err {info_means['yaw_error']:.3f} "
                f"std {action_std:.3f} lr {lr:.2e} kl {approx_kl:.4f}",
                flush=True,
            )

        if iteration % args.eval_every == 0 or iteration == num_iterations:
            best_eval = run_eval(agent, eval_env, args, writer, run_dir, cfg, cfg_dir, iteration, global_step, best_eval, device)

    eval_env.close()
    writer.close()


if __name__ == "__main__":
    main()
