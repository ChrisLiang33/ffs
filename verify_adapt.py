"""Verify the policy's (alpha, phi) adapts to obstacle proximity AND approach velocity.

Runs an RL checkpoint on a chosen scene, tracks per-step:
  - alpha, phi (the policy outputs)
  - distance to nearest obstacle
  - relative approach velocity (positive = obstacle approaching robot)

Bins by distance and approach velocity to surface modulation patterns. If the
policy is actually using BEV info, we expect: shorter distance -> lower alpha,
higher phi; higher approach velocity -> higher phi.

Usage:
  $HOME/IsaacLab/isaaclab.sh -p verify_adapt.py \\
    --task Isaac-Goal-Go2-LongHistD-v0 \\
    --checkpoint logs/rsl_rl/.../model_399.pt \\
    --scene head_on
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-Goal-Go2-LongHistD-v0")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--scene", default="head_on")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--steps", type=int, default=500)
parser.add_argument("--episode_length_s", type=float, default=12.0)
parser.add_argument("--eval_seed", type=int, default=0)
parser.add_argument("--agent", default="rsl_rl_cfg_entry_point")
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
from cbf_go2.scenes import apply_scene
from cbf_go2.env_cfg import OBSTACLE_NAMES
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils.hydra import hydra_task_config

rsl_rl_version = metadata.version("rsl-rl-lib")


@hydra_task_config(args.task, args.agent)
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.episode_length_s = args.episode_length_s
    env_cfg.seed = args.eval_seed
    agent_cfg.seed = args.eval_seed
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, rsl_rl_version)

    env = gym.make(args.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    device = env.unwrapped.device

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=device)

    robot = env.unwrapped.scene["robot"]
    obstacles = [env.unwrapped.scene[name] for name in OBSTACLE_NAMES]

    apply_scene(env.unwrapped, args.scene, force_init_positions=True)
    obs = env.get_observations()

    # Per-step records (flattened across envs)
    alphas, phis, dists, approach_vs = [], [], [], []

    for _ in range(args.steps):
        apply_scene(env.unwrapped, args.scene)
        with torch.inference_mode():
            action = policy(obs)
            obs, _, _, _ = env.step(action)

            # Distance to nearest obstacle in body frame (we don't have body frame easily,
            # use world frame relative offset projected onto robot's body axes is overkill;
            # straight Euclidean dist + radial approach velocity is enough).
            robot_pos = robot.data.root_pos_w[:, :2]            # (N, 2)
            robot_vel = robot.data.root_lin_vel_w[:, :2]        # (N, 2)
            obs_positions = torch.stack(
                [o.data.root_pos_w[:, :2] for o in obstacles], dim=1  # (N, K, 2)
            )
            obs_velocities = torch.stack(
                [o.data.root_lin_vel_w[:, :2] for o in obstacles], dim=1  # (N, K, 2)
            )
            rel = obs_positions - robot_pos.unsqueeze(1)        # (N, K, 2)
            dist_per_obs = torch.linalg.norm(rel, dim=-1).clamp_min(1e-6)  # (N, K)
            # Filter out off-stage obstacles (>10m).
            dist_per_obs = torch.where(dist_per_obs > 10.0, torch.full_like(dist_per_obs, float("inf")), dist_per_obs)
            d_nearest, nearest_idx = dist_per_obs.min(dim=-1)   # (N,), (N,)

            # Approach velocity: relative velocity (obstacle - robot) projected onto direction (obstacle - robot).
            # Positive = obstacle moving toward robot (closing).
            unit_dir = rel / dist_per_obs.unsqueeze(-1)
            rel_vel = obs_velocities - robot_vel.unsqueeze(1)
            approach_v_per_obs = -(rel_vel * unit_dir).sum(dim=-1)  # (N, K) negative since approach = -d/dt(dist)
            approach_v = torch.gather(approach_v_per_obs, 1, nearest_idx[:, None]).squeeze(-1)

            # Decode policy action -> (alpha, phi)
            a_lo, a_hi = 0.1, 5.0
            p_lo, p_hi = 0.01, 10.0
            norm = action.clamp(-1.0, 1.0)
            alpha = (norm[:, 0] + 1.0) * 0.5 * (a_hi - a_lo) + a_lo
            phi = (norm[:, 1] + 1.0) * 0.5 * (p_hi - p_lo) + p_lo

            alphas.append(alpha.cpu())
            phis.append(phi.cpu())
            dists.append(d_nearest.cpu())
            approach_vs.append(approach_v.cpu())

    alphas = torch.cat(alphas)
    phis = torch.cat(phis)
    dists = torch.cat(dists)
    approach_vs = torch.cat(approach_vs)

    # Drop off-stage steps (dist == inf)
    valid = ~torch.isinf(dists)
    alphas, phis, dists, approach_vs = alphas[valid], phis[valid], dists[valid], approach_vs[valid]

    def bin_report(name, values, low_label, high_label, n_bins=3):
        sorted_v = torch.sort(values).values
        cutoffs = [sorted_v[i * len(sorted_v) // n_bins].item() for i in range(1, n_bins)]
        bins = []
        for i in range(n_bins):
            if i == 0:
                mask = values < cutoffs[0]
            elif i == n_bins - 1:
                mask = values >= cutoffs[-1]
            else:
                mask = (values >= cutoffs[i - 1]) & (values < cutoffs[i])
            bins.append(mask)
        labels = [low_label] + [f"mid{i}" for i in range(1, n_bins - 1)] + [high_label]
        print(f"\n  {name}-conditional (terciles):")
        print(f"    {'bin':<8} {'n':<8} {'val':<8} {'alpha':<8} {'phi':<8}")
        for label, mask in zip(labels, bins):
            if mask.sum() == 0:
                continue
            v = values[mask].mean().item()
            a = alphas[mask].mean().item()
            p = phis[mask].mean().item()
            print(f"    {label:<8} {int(mask.sum()):<8} {v:<7.2f} {a:<7.2f} {p:<7.2f}")

    print(f"\ncheckpoint: {os.path.basename(args.checkpoint)}   scene: {args.scene}")
    print(f"  n_samples = {len(alphas)}, alpha_mean={alphas.mean():.2f}, phi_mean={phis.mean():.2f}")

    bin_report("distance-to-obstacle", dists, "near", "far")
    bin_report("approach-velocity", approach_vs, "fleeing", "charging")

    env.close()


if __name__ == "__main__":
    main()
    app.close()
