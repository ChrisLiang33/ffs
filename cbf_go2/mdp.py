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


def velocity_to_goal(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """World-frame velocity component pointing at the goal (m/s, signed)."""
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_term(command_name)
    dir_to_goal = cmd.pos_command_w[:, :2] - asset.data.root_pos_w[:, :2]
    norm = torch.linalg.norm(dir_to_goal, dim=1, keepdim=True).clamp_min(1e-6)
    unit = dir_to_goal / norm
    return (asset.data.root_lin_vel_w[:, :2] * unit).sum(dim=1)


def timeout_fired(env: ManagerBasedRLEnv) -> torch.Tensor:
    """1.0 on the step time_out fires (episode hit max length), else 0."""
    return env.termination_manager.time_outs.float()


def tissf_action(
    env: ManagerBasedRLEnv,
    obstacle_names: tuple,
    obstacle_radius: float = 0.5,
    robot_radius: float = 0.3,
    alpha: float = 2.0,
    epsilon_0: float = 1.0,
    lam: float = 1.5,
    h_lam: float = 1.0,
    h_gamma: float = 1.0,
    alpha_range: tuple = (0.1, 5.0),
    phi_range: tuple = (0.01, 10.0),
) -> torch.Tensor:
    """TISSf baseline per Wang et al.:  epsilon(h) = epsilon_0 * exp(lam * h),  phi = 1/epsilon(h).

    h here is the smoothed CBF (h_lam, h_gamma are the same constants used in the CBF filter).
    lam = 0 recovers ISSf with phi = 1/epsilon_0 everywhere. Larger lam concentrates the
    buffer near the boundary and frees up the interior.

    Returns the *normalized* action in [-1, 1] for direct use with CBFParamsAction.
    """
    from . import cbf

    robot: Articulation = env.scene["robot"]
    obs_xy = torch.stack(
        [env.scene[name].data.root_pos_w[:, :2] for name in obstacle_names], dim=1
    )
    sdf, sdf_grad = cbf.compute_sdf(robot.data.root_pos_w[:, :2], obs_xy, obstacle_radius, robot_radius)
    h, _ = cbf.compute_h(sdf, sdf_grad, lam=h_lam, gamma=h_gamma)
    phi = (1.0 / epsilon_0) * torch.exp(-lam * h)

    a_lo, a_hi = alpha_range
    p_lo, p_hi = phi_range
    norm_alpha = 2.0 * (alpha - a_lo) / (a_hi - a_lo) - 1.0
    norm_phi = (2.0 * (phi - p_lo) / (p_hi - p_lo) - 1.0).clamp(-1.0, 1.0)

    action = torch.empty((env.num_envs, 2), device=env.device)
    action[:, 0] = norm_alpha
    action[:, 1] = norm_phi
    return action


def obstacles_body_frame(
    env: ManagerBasedRLEnv,
    obstacle_names: tuple,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Per-obstacle xy in robot body frame. Shape (N, K*2)."""
    asset: Articulation = env.scene[asset_cfg.name]
    pos_w = asset.data.root_pos_w[:, :2]
    quat = asset.data.root_quat_w
    yaw = torch.atan2(
        2 * (quat[:, 0] * quat[:, 3] + quat[:, 1] * quat[:, 2]),
        1 - 2 * (quat[:, 2] ** 2 + quat[:, 3] ** 2),
    )
    cos_y = torch.cos(yaw).unsqueeze(-1)
    sin_y = torch.sin(yaw).unsqueeze(-1)

    obs_xy_w = torch.stack(
        [env.scene[name].data.root_pos_w[:, :2] for name in obstacle_names], dim=1
    )
    rel = obs_xy_w - pos_w.unsqueeze(1)  # (N, K, 2)
    rel_x = cos_y * rel[..., 0] + sin_y * rel[..., 1]
    rel_y = -sin_y * rel[..., 0] + cos_y * rel[..., 1]
    return torch.stack([rel_x, rel_y], dim=-1).flatten(1)
