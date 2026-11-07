import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import torch

from hexapod_env import HexapodEnv
from train_ppo import Agent

SIM_DIR = Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser(description="Run a trained hexapod policy in the MuJoCo viewer (use mjpython on macOS)")
    p.add_argument("checkpoint")
    p.add_argument("--vx", type=float, default=0.2)
    p.add_argument("--vy", type=float, default=0.0)
    p.add_argument("--yaw", type=float, default=0.0)
    p.add_argument("--config", default=None)
    p.add_argument("--randomize", action="store_true")
    p.add_argument("--no-noise", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--duration", type=float, default=0.0)
    return p.parse_args()


def main():
    args = parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    noise = False if args.no_noise else None
    if args.config:
        env = HexapodEnv(args.config, randomize=args.randomize, noise=noise)
    else:
        cfg_dir = Path(ckpt["config_dir"])
        if not (cfg_dir / ckpt["config"]["robot"]["model"]).exists():
            cfg_dir = SIM_DIR / "configs"
        env = HexapodEnv(ckpt["config"], config_dir=cfg_dir, randomize=args.randomize, noise=noise)

    agent = Agent(ckpt["obs_dim"], ckpt["actor_obs_dim"], ckpt["action_dim"], ckpt["hidden_sizes"], ckpt.get("init_std", 1.0))
    agent.load_state_dict(ckpt["agent"])
    agent.eval()
    print(f"loaded {args.checkpoint} (iteration {ckpt['iteration']}, eval return {ckpt['eval_return']:.2f})")

    env.set_command(args.vx, args.vy, args.yaw)
    obs, _ = env.reset(seed=args.seed)

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = env.base_id
        viewer.cam.distance = 1.0
        viewer.cam.elevation = -25.0

        start = time.perf_counter()
        t_next = start
        vel_sum, n, falls = np.zeros(3), 0, 0
        last_print = start
        while viewer.is_running():
            with torch.no_grad():
                a = agent.act_mean(torch.as_tensor(obs).unsqueeze(0)).numpy()[0]
            with viewer.lock():
                obs, _, term, _, _ = env.step(a)
                vel_sum += np.array([env.base_lin_vel[0], env.base_lin_vel[1], env.base_ang_vel[2]])
                n += 1
                if term:
                    falls += 1
                    print("fell, resetting")
                    obs, _ = env.reset()
            viewer.sync()

            now = time.perf_counter()
            if now - last_print >= 2.0:
                v = vel_sum / max(n, 1)
                print(f"cmd vx {args.vx:+.2f} vy {args.vy:+.2f} yaw {args.yaw:+.2f} | "
                      f"meas vx {v[0]:+.2f} vy {v[1]:+.2f} yaw {v[2]:+.2f} | falls {falls}")
                vel_sum[:], n, last_print = 0.0, 0, now
            if args.duration > 0 and now - start >= args.duration:
                break

            t_next += env.dt
            sleep = t_next - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
            elif sleep < -0.1:
                t_next = time.perf_counter()


if __name__ == "__main__":
    main()
