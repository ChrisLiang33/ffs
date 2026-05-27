"""Smoke test: send fixed (alpha, phi) to the env. ISSf baseline = (2.0, 0.5)."""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=200)
parser.add_argument("--video", action="store_true")
parser.add_argument("--alpha", type=float, default=2.0)
parser.add_argument("--phi", type=float, default=0.5)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
if args.video:
    args.enable_cameras = True
    args.num_envs = 1

app = AppLauncher(args).app

import gymnasium as gym
import torch

import cbf_go2  # noqa: F401
from cbf_go2.env_cfg import GoalGo2EnvCfg

cfg = GoalGo2EnvCfg()
cfg.scene.num_envs = args.num_envs
cfg.episode_length_s = 30.0
if args.video:
    cfg.decimation = 4

env = gym.make(
    "Isaac-Goal-Go2-v0",
    cfg=cfg,
    render_mode="rgb_array" if args.video else None,
)
if args.video:
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=os.path.join(os.getcwd(), "videos"),
        step_trigger=lambda step: step == 0,
        video_length=args.steps,
        disable_logger=True,
    )
obs, _ = env.reset()

unwrapped = env.unwrapped
term_mgr = unwrapped.termination_manager
device = unwrapped.device

# action space is normalized [-1, 1] -> action term scales to (alpha, phi) ranges
def to_norm(v, lo, hi):
    return 2.0 * (v - lo) / (hi - lo) - 1.0

action = torch.zeros((args.num_envs, 2), device=device)
action[:, 0] = to_norm(args.alpha, 0.1, 5.0)
action[:, 1] = to_norm(args.phi, 0.01, 10.0)

n_reached = 0
n_crashed = 0
n_timeout = 0
for i in range(args.steps):
    obs, _, _, _, _ = env.step(action)
    n_reached += int(term_mgr.get_term("goal_reached").sum().item())
    n_crashed += int(
        (term_mgr.get_term("base_contact") | term_mgr.get_term("obstacle_hit")).sum().item()
    )
    n_timeout += int(term_mgr.get_term("time_out").sum().item())

total = n_reached + n_crashed + n_timeout
print(f"action: alpha={args.alpha}, phi={args.phi}")
print(f"episodes ended: {total}")
print(f"  reached: {n_reached}  ({n_reached / max(total,1):.0%})")
print(f"  crashed: {n_crashed}  ({n_crashed / max(total,1):.0%})")
print(f"  timeout: {n_timeout}  ({n_timeout / max(total,1):.0%})")

env.close()
app.close()
