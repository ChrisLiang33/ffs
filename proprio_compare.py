"""Compare RL vs ISSf via speed-binned crash rates — the proprio-edge test.

For each method, run eval and record per-episode (mean_speed, outcome). Bin
episodes by their mean speed during the last K steps and compute crash rate
per bin. The hypothesis: RL crashes LESS than fixed ISSf at high speeds
because it uses proprio to know "I am approaching fast, be more conservative."

Usage:
  # RL:
  $HOME/IsaacLab/isaaclab.sh -p proprio_compare.py --mode rl \\
    --task Isaac-Goal-Go2-LongHistD-v0 \\
    --checkpoint logs/rsl_rl/cbf_goal_go2_archD/<run>/model_399.pt \\
    --scene corridor --eval_seed 0

  # ISSf (alpha=2, phi=1.0):
  $HOME/IsaacLab/isaaclab.sh -p proprio_compare.py --mode issf --alpha 2.0 --phi 1.0 \\
    --task Isaac-Goal-Go2-LongHistD-v0 --scene corridor --eval_seed 0
"""

import argparse
import json
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-Goal-Go2-LongHistD-v0")
parser.add_argument("--mode", choices=["rl", "issf"], required=True)
parser.add_argument("--checkpoint", default=None)
parser.add_argument("--alpha", type=float, default=2.0)
parser.add_argument("--phi", type=float, default=1.0)
parser.add_argument("--scene", default="corridor")
parser.add_argument("--eval_seed", type=int, default=0)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--steps", type=int, default=1500)
parser.add_argument("--lookback_steps", type=int, default=30,
                    help="for mean-speed calc per episode, use last K steps before termination")
parser.add_argument("--save", default=None)
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
from cbf_go2.scenes import apply_scene
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

rsl_rl_version = metadata.version("rsl-rl-lib")


def to_norm(v, lo, hi):
    return 2.0 * (v - lo) / (hi - lo) - 1.0


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
    n = args.num_envs
    K = args.lookback_steps

    if args.mode == "rl":
        log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
        resume = args.checkpoint or get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(resume)
        policy = runner.get_inference_policy(device=device)
        mode_str = f"RL ({os.path.basename(resume)})"
    else:
        fixed = torch.zeros((n, 2), device=device)
        fixed[:, 0] = to_norm(args.alpha, 0.1, 5.0)
        fixed[:, 1] = to_norm(args.phi, 0.01, 10.0)
        mode_str = f"ISSf (alpha={args.alpha}, phi={args.phi})"

    term_mgr = env.unwrapped.termination_manager
    robot = env.unwrapped.scene["robot"]
    apply_scene(env.unwrapped, args.scene)
    obs = env.get_observations()

    # Rolling per-env speed buffer (size K) so we can compute mean of last K steps at termination.
    speed_buf = torch.zeros((n, K), device=device)
    buf_count = torch.zeros(n, device=device, dtype=torch.long)
    buf_ptr = torch.zeros(n, device=device, dtype=torch.long)

    # Per-episode records: (mean_speed_last_K, outcome ∈ {"reached", "crashed", "timeout"})
    episodes = []

    for _ in range(args.steps):
        apply_scene(env.unwrapped, args.scene)
        with torch.inference_mode():
            action = policy(obs) if args.mode == "rl" else fixed
            obs, _, dones, _ = env.step(action)

            speed = torch.linalg.norm(robot.data.root_lin_vel_b[:, :2], dim=-1)
            # write to ring buffer
            speed_buf[torch.arange(n, device=device), buf_ptr] = speed
            buf_ptr = (buf_ptr + 1) % K
            buf_count = torch.minimum(buf_count + 1, torch.full_like(buf_count, K))

            reached_mask = term_mgr.get_term("goal_reached")
            crashed_mask = term_mgr.get_term("base_contact") | term_mgr.get_term("obstacle_hit")
            timeout_mask = term_mgr.get_term("time_out")

            for idx_t, outcome in [(reached_mask, "reached"), (crashed_mask, "crashed"), (timeout_mask, "timeout")]:
                for env_id in idx_t.nonzero(as_tuple=True)[0].tolist():
                    count = int(buf_count[env_id].item())
                    if count == 0:
                        continue
                    valid_speeds = speed_buf[env_id, :count]
                    episodes.append((float(valid_speeds.mean().item()), outcome))

            # Reset ring buffer for envs that ended.
            done_idx = dones.nonzero(as_tuple=True)[0]
            if len(done_idx) > 0:
                buf_count[done_idx] = 0
                buf_ptr[done_idx] = 0

    # Bin episodes by mean speed (terciles based on this run).
    speeds = sorted([s for s, _ in episodes])
    if not speeds:
        print("no episodes completed")
        env.close()
        return
    t1 = speeds[len(speeds) // 3]
    t2 = speeds[2 * len(speeds) // 3]

    bins = [
        ("slow",   lambda s: s < t1),
        ("mid",    lambda s: t1 <= s < t2),
        ("fast",   lambda s: s >= t2),
    ]
    print()
    print(f"mode: {mode_str}   scene: {args.scene}   seed: {args.eval_seed}")
    print(f"  n_episodes = {len(episodes)}, lookback = {K} steps ({K * 0.02:.1f}s)")
    print(f"  speed terciles: <{t1:.2f}, {t1:.2f}-{t2:.2f}, >={t2:.2f} m/s")
    print()
    print(f"  {'bin':<6} {'n':<6} {'speed':<8} {'reach%':<8} {'crash%':<8} {'timeo%':<8}")
    out_rows = []
    for name, fn in bins:
        bin_eps = [(s, o) for s, o in episodes if fn(s)]
        if not bin_eps:
            continue
        n_b = len(bin_eps)
        s_b = sum(s for s, _ in bin_eps) / n_b
        r = sum(1 for _, o in bin_eps if o == "reached") / n_b
        c = sum(1 for _, o in bin_eps if o == "crashed") / n_b
        t = sum(1 for _, o in bin_eps if o == "timeout") / n_b
        print(f"  {name:<6} {n_b:<6} {s_b:<7.2f} {r * 100:<7.1f} {c * 100:<7.1f} {t * 100:<7.1f}")
        out_rows.append({"bin": name, "n": n_b, "mean_speed": s_b, "reach": r, "crash": c, "timeout": t})

    if args.save:
        with open(args.save, "w") as f:
            json.dump({
                "mode": mode_str, "scene": args.scene, "seed": args.eval_seed,
                "lookback_steps": K, "tercile_cutoffs": [t1, t2],
                "bins": out_rows,
            }, f, indent=2)
        print(f"\n  saved to {args.save}")

    env.close()


if __name__ == "__main__":
    main()
    app.close()
