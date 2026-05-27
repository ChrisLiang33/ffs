"""Custom mdp terms for cbf_go2."""

import torch

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def at_goal(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """True when the robot is within `threshold` meters of the 2D goal pose."""
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_term(command_name)
    delta = cmd.pos_command_w[:, :2] - asset.data.root_pos_w[:, :2]
    return torch.linalg.norm(delta, dim=1) < threshold


def in_collision(
    env: ManagerBasedRLEnv,
    obstacle_names: tuple,
    margin: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """True when the robot center is within `margin` of any obstacle center."""
    robot: Articulation = env.scene[asset_cfg.name]
    robot_xy = robot.data.root_pos_w[:, :2]
    obs_xy = torch.stack(
        [env.scene[name].data.root_pos_w[:, :2] for name in obstacle_names], dim=1
    )  # (N, K, 2)
    dist = torch.linalg.norm(robot_xy.unsqueeze(1) - obs_xy, dim=2)
    return (dist < margin).any(dim=1)
