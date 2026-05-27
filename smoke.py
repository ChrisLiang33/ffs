"""Smoke test: naive u_nom -> CBF -> frozen locomotion. Counts reach vs crash."""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=200)
parser.add_argument("--video", action="store_true")
parser.add_argument("--no_cbf", action="store_true", help="Skip the CBF filter (unsafe baseline)")
parser.add_argument("--alpha", type=float, default=2.0)
parser.add_argument("--phi", type=float, default=0.2)
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
from cbf_go2 import cbf
from cbf_go2.env_cfg import (
    GoalGo2EnvCfg,
    OBSTACLE_NAMES,
    OBSTACLE_RADIUS,
)

ROBOT_RADIUS = 0.3

cfg = GoalGo2EnvCfg()
cfg.scene.num_envs = args.num_envs
cfg.episode_length_s = 30.0
if args.video:
    # match inner locomotion rate so the video captures gait, not 5fps slides
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
robot = unwrapped.scene["robot"]
goal_cmd = unwrapped.command_manager.get_term("goal_pose")
term_mgr = unwrapped.termination_manager
device = unwrapped.device

# fixed obstacle positions per env (kinematic -> never move)
obs_xy = torch.stack(
    [unwrapped.scene[name].data.root_pos_w[:, :2] for name in OBSTACLE_NAMES], dim=1
)  # (N, K, 2)

n_reached = 0
n_crashed = 0
n_timeout = 0

for i in range(args.steps):
    goal_xy_b = goal_cmd.pos_command_b[:, :2]
    dist = torch.linalg.norm(goal_xy_b, dim=1, keepdim=True).clamp_min(1e-6)
    u_nom = torch.zeros((args.num_envs, 3), device=device)
    u_nom[:, :2] = goal_xy_b / dist

    if args.no_cbf:
        action = u_nom
    else:
        quat = robot.data.root_quat_w
        yaw = torch.atan2(
            2 * (quat[:, 0] * quat[:, 3] + quat[:, 1] * quat[:, 2]),
            1 - 2 * (quat[:, 2] ** 2 + quat[:, 3] ** 2),
        )
        action = cbf.safety_filter(
            u_nom=u_nom,
            robot_xy=robot.data.root_pos_w[:, :2],
            robot_yaw=yaw,
            obs_xy=obs_xy,
            obstacle_radius=OBSTACLE_RADIUS,
            robot_radius=ROBOT_RADIUS,
            alpha=args.alpha,
            phi=args.phi,
        )

    obs, _, _, _, _ = env.step(action)

    n_reached += int(term_mgr.get_term("goal_reached").sum().item())
    n_crashed += int(
        (term_mgr.get_term("base_contact") | term_mgr.get_term("obstacle_hit")).sum().item()
    )
    n_timeout += int(term_mgr.get_term("time_out").sum().item())

total = n_reached + n_crashed + n_timeout
mode = "no-CBF" if args.no_cbf else f"CBF (alpha={args.alpha}, phi={args.phi})"
print(f"mode: {mode}")
print(f"episodes ended: {total}")
print(f"  reached: {n_reached}  ({n_reached / max(total,1):.0%})")
print(f"  crashed: {n_crashed}  ({n_crashed / max(total,1):.0%})")
print(f"  timeout: {n_timeout}  ({n_timeout / max(total,1):.0%})")

env.close()
app.close()
