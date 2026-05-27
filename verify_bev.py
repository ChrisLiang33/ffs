"""Verify the BEV occupancy grid matches actual obstacle positions.

Sets obstacles at known body-frame positions, prints both the grid and
the expected occupancy as ASCII for side-by-side comparison.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

app = AppLauncher(args).app

import gymnasium as gym
import torch

import cbf_go2  # noqa: F401
from cbf_go2 import mdp as cbf_mdp
from cbf_go2.env_cfg import GoalGo2EnvCfg, OBSTACLE_NAMES, OBSTACLE_RADIUS

GRID_SIZE = 16
GRID_EXTENT = 3.0
KNOWN = [(1.5, 0.0), (0.0, 1.5), (-1.5, 0.0), (0.0, -1.5)]


def coord_to_cell(v: float) -> int:
    return round((v + GRID_EXTENT) / (2 * GRID_EXTENT) * (GRID_SIZE - 1))


cfg = GoalGo2EnvCfg()
cfg.scene.num_envs = 1
env = gym.make("Isaac-Goal-Go2-v0", cfg=cfg)
obs, _ = env.reset()

unwrapped = env.unwrapped
device = unwrapped.device

# Override robot to (0, 0, default_z) with identity yaw
robot = unwrapped.scene["robot"]
robot_pose = robot.data.default_root_state[:, :7].clone()
robot_pose[:, 0] = unwrapped.scene.env_origins[:, 0]
robot_pose[:, 1] = unwrapped.scene.env_origins[:, 1]
robot_pose[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
robot.write_root_pose_to_sim(robot_pose)

# Override obstacles to KNOWN positions (body-frame == world-frame since robot is at origin with identity yaw)
for i, (x, y) in enumerate(KNOWN):
    obstacle = unwrapped.scene[f"obstacle_{i}"]
    new_pose = obstacle.data.default_root_state[:, :7].clone()
    new_pose[:, 0] = x + unwrapped.scene.env_origins[:, 0]
    new_pose[:, 1] = y + unwrapped.scene.env_origins[:, 1]
    obstacle.write_root_pose_to_sim(new_pose)

# Step once with zero action to sync data buffers
action = torch.zeros((1, 2), device=device)
env.step(action)

# Compute BEV
grid_flat = cbf_mdp.bev_occupancy(
    unwrapped,
    obstacle_names=OBSTACLE_NAMES,
    obstacle_radius=OBSTACLE_RADIUS,
    grid_size=GRID_SIZE,
    grid_extent=GRID_EXTENT,
).cpu().numpy()[0]
grid = grid_flat.reshape(GRID_SIZE, GRID_SIZE)

print()
print("Obstacles (body-frame x, y) -> expected cell (i, j):")
for x, y in KNOWN:
    print(f"  ({x:+.1f}, {y:+.1f}) -> cell ({coord_to_cell(x)}, {coord_to_cell(y)})")
print()
print(f"BEV grid {GRID_SIZE}x{GRID_SIZE}, extent +/- {GRID_EXTENT} m")
print("Top = body +x (forward),  Left = body +y (left of robot),  R = robot center")
print("-" * (GRID_SIZE + 2))
for i in reversed(range(GRID_SIZE)):
    row = "|"
    for j in reversed(range(GRID_SIZE)):
        if grid[i, j] > 0.5:
            row += "#"
        elif i in (GRID_SIZE // 2 - 1, GRID_SIZE // 2) and j in (GRID_SIZE // 2 - 1, GRID_SIZE // 2):
            row += "R"
        else:
            row += "."
    row += "|"
    print(row)
print("-" * (GRID_SIZE + 2))

env.close()
app.close()
