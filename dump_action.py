"""Quick action-distribution dump for any RL checkpoint, independent of overlay versions.

Use as a one-off probe after `git checkout <commit>`:
  $HOME/IsaacLab/isaaclab.sh -p dump_action.py --checkpoint logs/rsl_rl/cbf_goal_go2/<run>/model_199.pt
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-Goal-Go2-v0")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--steps", type=int, default=500)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--agent", default="rsl_rl_cfg_entry_point")
parser.add_argument("--episode_length_s", type=float, default=30.0)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
args.headless = True
sys.argv = [sys.argv[0]] + hydra_args

app = AppLauncher(args).app

import importlib.metadata as metadata

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

import cbf_go2  # noqa: F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils.hydra import hydra_task_config

rsl_rl_version = metadata.version("rsl-rl-lib")


@hydra_task_config(args.task, args.agent)
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.episode_length_s = args.episode_length_s
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, rsl_rl_version)

    env = gym.make(args.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    device = env.unwrapped.device

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=device)

    obs = env.get_observations()
    samples = []
    intervened = []        # per-step bool tensor of "did CBF actually project?"
    robot_speeds = []      # per-step robot linear speed magnitude (proxy for "fast approach")
    action_term = env.unwrapped.action_manager.get_term("cbf_params")
    robot = env.unwrapped.scene["robot"]
    for _ in range(args.steps):
        with torch.inference_mode():
            action = policy(obs)
            samples.append(action.detach().clone().cpu())
            obs, _, _, _ = env.step(action)
            if hasattr(action_term, "_last_intervened"):
                intervened.append(action_term._last_intervened.detach().clone().cpu())
            speed = torch.linalg.norm(robot.data.root_lin_vel_b[:, :2], dim=-1)
            robot_speeds.append(speed.detach().clone().cpu())

    all_acts = torch.cat(samples, dim=0).clamp(-1.0, 1.0)
    alpha = (all_acts[:, 0] + 1.0) / 2.0 * (5.0 - 0.1) + 0.1   # range (0.1, 5.0)
    phi = (all_acts[:, 1] + 1.0) / 2.0 * (10.0 - 0.01) + 0.01  # range (0.01, 10.0)
    pcts = [5, 25, 50, 75, 95]
    a_p = [alpha.quantile(p / 100.0).item() for p in pcts]
    p_p = [phi.quantile(p / 100.0).item() for p in pcts]
    print()
    print(f"checkpoint: {args.checkpoint}")
    print(f"  n_samples = {int(all_acts.shape[0])}")
    print(f"  alpha: mean={alpha.mean():.2f}  std={alpha.std():.2f}  "
          f"pcts(5/25/50/75/95)= {a_p[0]:.2f} {a_p[1]:.2f} {a_p[2]:.2f} {a_p[3]:.2f} {a_p[4]:.2f}")
    print(f"  phi:   mean={phi.mean():.2f}  std={phi.std():.2f}  "
          f"pcts(5/25/50/75/95)= {p_p[0]:.2f} {p_p[1]:.2f} {p_p[2]:.2f} {p_p[3]:.2f} {p_p[4]:.2f}")
    if intervened:
        inter = torch.cat(intervened).float()
        print(f"  cbf intervention rate: {inter.mean().item():.1%}   "
              f"(fraction of steps the safety filter actually projected u_nom -> u_safe)")

    # Velocity-conditional action stats: does the policy modulate (alpha, phi) based on speed?
    # Proprio edge test: if RL uses base velocity, expect phi up + alpha down at high speeds.
    if robot_speeds:
        speeds = torch.cat(robot_speeds)
        # Tercile bins on robot speed
        s_lo = speeds.quantile(1/3.0)
        s_hi = speeds.quantile(2/3.0)
        bins = [("slow", speeds < s_lo), ("mid", (speeds >= s_lo) & (speeds < s_hi)), ("fast", speeds >= s_hi)]
        print(f"\n  speed-conditional action means (speed terciles: <{s_lo:.2f}, {s_lo:.2f}-{s_hi:.2f}, >={s_hi:.2f} m/s):")
        print(f"    {'bin':<6} {'n':<8} {'speed':<8} {'alpha':<8} {'phi':<8} {'cbf%':<6}")
        for name, mask in bins:
            if mask.sum() == 0:
                continue
            a_bin = alpha[mask].mean().item()
            p_bin = phi[mask].mean().item()
            s_bin = speeds[mask].mean().item()
            i_bin = inter[mask].mean().item() if intervened else float("nan")
            print(f"    {name:<6} {int(mask.sum()):<8} {s_bin:<7.2f} {a_bin:<7.2f} {p_bin:<7.2f} {i_bin:.1%}")
    print()

    env.close()


if __name__ == "__main__":
    main()
    app.close()
