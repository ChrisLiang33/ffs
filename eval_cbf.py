"""Benchmark ISSf (fixed alpha, phi) or a trained RL policy on Isaac-Goal-Go2-v0.

Usage:
    ISSf baseline:
        ~/IsaacLab/isaaclab.sh -p eval_cbf.py --mode issf --alpha 2.0 --phi 0.5
    RL checkpoint (latest):
        ~/IsaacLab/isaaclab.sh -p eval_cbf.py --mode rl
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-Goal-Go2-v0")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--steps", type=int, default=2000)
parser.add_argument("--mode", choices=["issf", "rl", "tissf"], required=True)
parser.add_argument("--alpha", type=float, default=2.0)
parser.add_argument("--phi", type=float, default=0.5)
parser.add_argument("--epsilon_0", type=float, default=1.0, help="TISSf: phi(0) = 1/epsilon_0")
parser.add_argument("--lam", type=float, default=1.5, help="TISSf: epsilon(h) = epsilon_0 * exp(lam*h)")
parser.add_argument("--save", default=None, help="Optional path to dump per-episode results as JSON")
parser.add_argument("--checkpoint", default=None)
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
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

rsl_rl_version = metadata.version("rsl-rl-lib")


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def to_norm(v: float, lo: float, hi: float) -> float:
    return 2.0 * (v - lo) / (hi - lo) - 1.0


@hydra_task_config(args.task, args.agent)
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.episode_length_s = args.episode_length_s
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, rsl_rl_version)

    env = gym.make(args.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    device = env.unwrapped.device

    if args.mode == "rl":
        log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
        resume = args.checkpoint or get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(resume)
        policy = runner.get_inference_policy(device=device)
        mode_str = f"RL ({os.path.basename(resume)})"
    elif args.mode == "tissf":
        from cbf_go2.env_cfg import OBSTACLE_NAMES, OBSTACLE_RADIUS
        from cbf_go2.mdp import tissf_action
        mode_str = f"TISSf (alpha={args.alpha}, epsilon_0={args.epsilon_0}, lam={args.lam})"
    else:
        fixed = torch.zeros((args.num_envs, 2), device=device)
        fixed[:, 0] = to_norm(args.alpha, 0.1, 5.0)
        fixed[:, 1] = to_norm(args.phi, 0.01, 10.0)
        mode_str = f"ISSf (alpha={args.alpha}, phi={args.phi})"

    term_mgr = env.unwrapped.termination_manager
    obs = env.get_observations()

    ep_steps = torch.zeros(args.num_envs, device=device, dtype=torch.long)
    episodes: list[tuple[str, int]] = []  # (outcome, length)

    for _ in range(args.steps):
        with torch.inference_mode():
            if args.mode == "rl":
                action = policy(obs)
            elif args.mode == "tissf":
                action = tissf_action(
                    env.unwrapped,
                    obstacle_names=OBSTACLE_NAMES,
                    obstacle_radius=OBSTACLE_RADIUS,
                    alpha=args.alpha,
                    epsilon_0=args.epsilon_0,
                    lam=args.lam,
                )
            else:
                action = fixed
            obs, _, dones, _ = env.step(action)
        ep_steps += 1
        reached_mask = term_mgr.get_term("goal_reached")
        crashed_mask = term_mgr.get_term("base_contact") | term_mgr.get_term("obstacle_hit")
        timeout_mask = term_mgr.get_term("time_out")
        for idx in reached_mask.nonzero(as_tuple=True)[0].tolist():
            episodes.append(("reached", int(ep_steps[idx])))
        for idx in crashed_mask.nonzero(as_tuple=True)[0].tolist():
            episodes.append(("crashed", int(ep_steps[idx])))
        for idx in timeout_mask.nonzero(as_tuple=True)[0].tolist():
            episodes.append(("timeout", int(ep_steps[idx])))
        ep_steps[dones] = 0

    n_reached = sum(1 for o, _ in episodes if o == "reached")
    n_crashed = sum(1 for o, _ in episodes if o == "crashed")
    n_timeout = sum(1 for o, _ in episodes if o == "timeout")
    reach_lengths = [n for o, n in episodes if o == "reached"]

    total = n_reached + n_crashed + n_timeout

    def line(name, k):
        if total == 0:
            return f"  {name}:     n/a"
        lo, hi = wilson_ci(k, total)
        return f"  {name}:  {k/total:>6.1%}   95% CI [{lo:.1%}, {hi:.1%}]   ({k}/{total})"

    print()
    print(f"mode: {mode_str}")
    print(f"  episode_length_s = {args.episode_length_s}, num_envs = {args.num_envs}, steps = {args.steps}")
    print(line("reached", n_reached))
    print(line("crashed", n_crashed))
    print(line("timeout", n_timeout))
    if reach_lengths:
        mean_len = sum(reach_lengths) / len(reach_lengths)
        print(f"  mean steps to reach: {mean_len:.1f}   ({mean_len * 0.02:.2f} sec)")
    print()

    if args.save:
        import json
        result = {
            "mode": mode_str,
            "args": {k: v for k, v in vars(args).items() if not k.startswith("_")},
            "summary": {
                "total": total,
                "reached": n_reached,
                "crashed": n_crashed,
                "timeout": n_timeout,
                "mean_reach_steps": (sum(reach_lengths) / len(reach_lengths)) if reach_lengths else None,
            },
            "episodes": [{"outcome": o, "length": n} for o, n in episodes],
        }
        with open(args.save, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"  saved to {args.save}")

    env.close()


if __name__ == "__main__":
    main()
    app.close()
