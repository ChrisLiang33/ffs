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
    extra_margin: float = 0.15,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """True when robot center is within (per-env obstacle_radius + extra_margin) of any obstacle."""
    robot: Articulation = env.scene[asset_cfg.name]
    robot_xy = robot.data.root_pos_w[:, :2]
    obs_xy = torch.stack(
        [env.scene[name].data.root_pos_w[:, :2] for name in obstacle_names], dim=1
    )  # (N, K, 2)
    dist = torch.linalg.norm(robot_xy.unsqueeze(1) - obs_xy, dim=2)  # (N, K)
    if hasattr(env, "_obstacle_radii"):
        radii = env._obstacle_radii  # (N, K)
    else:
        radii = torch.full_like(dist, 0.5)
    return (dist < radii + extra_margin).any(dim=1)


def priv_dr_values(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Privileged DR values per env — only what's NOT recoverable from lidar.

    Layout: [static_friction, dynamic_friction, mass_delta, body_height]  shape (N, 4).
    Obstacle radii and velocities are visible in the BEV history, not here.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    materials = asset.root_physx_view.get_material_properties()
    friction = materials[:, 0, 0:2].to(env.device)
    current_mass = asset.root_physx_view.get_masses()[:, 0:1].to(env.device)
    default_mass = asset.data.default_mass[:, 0:1].to(env.device)
    mass_delta = current_mass - default_mass
    body_height = asset.data.root_pos_w[:, 2:3]
    return torch.cat([friction, mass_delta, body_height], dim=-1)


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


def action_rate_clamped(env: ManagerBasedRLEnv, term_name: str = "cbf_params") -> torch.Tensor:
    """||raw_t - raw_{t-1}||^2 using the clamped raw actions on a CBFParamsAction term."""
    term = env.action_manager.get_term(term_name)
    return torch.sum(torch.square(term.raw_actions - term.prev_raw_actions), dim=1)


def last_action_clamped(env: ManagerBasedRLEnv, term_name: str = "cbf_params") -> torch.Tensor:
    """Last action as observation, read from the action term's clamped raw buffer."""
    return env.action_manager.get_term(term_name).raw_actions


def timeout_fired(env: ManagerBasedRLEnv) -> torch.Tensor:
    """1.0 on the step time_out fires (episode hit max length), else 0."""
    return env.termination_manager.time_outs.float()


def _ensure_obstacle_state(env: ManagerBasedRLEnv, K: int) -> None:
    """Lazily allocate per-env per-obstacle radius + velocity tensors on the env."""
    if not hasattr(env, "_obstacle_radii"):
        env._obstacle_radii = torch.full((env.num_envs, K), 0.5, device=env.device)
    if not hasattr(env, "_obstacle_velocities"):
        env._obstacle_velocities = torch.zeros((env.num_envs, K, 2), device=env.device)


def randomize_obstacle_positions(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    obstacle_names: tuple,
    range_xy: float = 2.5,
    min_dist_from_origin: float = 0.8,
    radius_range: tuple[float, float] = (0.4, 0.6),
    drift_prob: float = 0.5,
    drift_speed_range: tuple[float, float] = (0.0, 0.3),
) -> None:
    """Each reset: random (x, y) + random radius + random drift velocity per obstacle."""
    import math
    K = len(obstacle_names)
    _ensure_obstacle_state(env, K)
    n = len(env_ids)

    # sample per-obstacle radii (N, K)
    new_radii = torch.empty((n, K), device=env.device).uniform_(*radius_range)
    env._obstacle_radii[env_ids] = new_radii

    # sample drift velocities: drift_prob chance of moving, speed in range, random direction
    drift_mask = (torch.rand((n, K), device=env.device) < drift_prob).float()
    speeds = torch.empty((n, K), device=env.device).uniform_(*drift_speed_range)
    angles = torch.rand((n, K), device=env.device) * (2 * math.pi)
    env._obstacle_velocities[env_ids, :, 0] = drift_mask * speeds * torch.cos(angles)
    env._obstacle_velocities[env_ids, :, 1] = drift_mask * speeds * torch.sin(angles)

    for k, name in enumerate(obstacle_names):
        obstacle = env.scene[name]
        x = torch.empty(n, device=env.device).uniform_(-range_xy, range_xy)
        y = torch.empty(n, device=env.device).uniform_(-range_xy, range_xy)
        dist = torch.sqrt(x * x + y * y).clamp_min(1e-6)
        scale = (min_dist_from_origin / dist).clamp_min(1.0)
        x = x * scale
        y = y * scale
        new_pose = obstacle.data.default_root_state[env_ids, :7].clone()
        new_pose[:, 0] = x + env.scene.env_origins[env_ids, 0]
        new_pose[:, 1] = y + env.scene.env_origins[env_ids, 1]
        obstacle.write_root_pose_to_sim(new_pose, env_ids=env_ids)


def advance_obstacles(env: ManagerBasedRLEnv, obstacle_names: tuple, dt: float) -> None:
    """Advance kinematic obstacles by their stored velocities. Called from the action term."""
    K = len(obstacle_names)
    if not hasattr(env, "_obstacle_velocities"):
        return
    for k, name in enumerate(obstacle_names):
        obstacle = env.scene[name]
        pose = torch.cat(
            [obstacle.data.root_pos_w, obstacle.data.root_quat_w], dim=-1
        ).clone()
        pose[:, 0] += env._obstacle_velocities[:, k, 0] * dt
        pose[:, 1] += env._obstacle_velocities[:, k, 1] * dt
        obstacle.write_root_pose_to_sim(pose)


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
    sdf, sdf_grad, _ = cbf.compute_sdf(robot.data.root_pos_w[:, :2], obs_xy, obstacle_radius, robot_radius)
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


def bev_occupancy(
    env: ManagerBasedRLEnv,
    obstacle_names: tuple,
    obstacle_radius: float = 0.5,  # fallback when env._obstacle_radii not set
    grid_size: int = 16,
    grid_extent: float = 3.0,
    dropout: float = 0.0,
    noise_std: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Body-frame BEV occupancy grid (deployment-realistic substitute for raw obstacle positions).

    Computed analytically from obstacle positions — same information content as a Mid-360 →
    grid pipeline, without Isaac Sim's multi-mesh raycast limitation. `dropout` randomly zeros
    cells (lidar miss); `noise_std` adds Gaussian noise (range jitter). Returns shape (N, grid_size**2).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    pos_w = asset.data.root_pos_w[:, :2]
    quat = asset.data.root_quat_w
    yaw = torch.atan2(
        2 * (quat[:, 0] * quat[:, 3] + quat[:, 1] * quat[:, 2]),
        1 - 2 * (quat[:, 2] ** 2 + quat[:, 3] ** 2),
    )
    cos_y = torch.cos(yaw).view(-1, 1, 1)
    sin_y = torch.sin(yaw).view(-1, 1, 1)

    coords = torch.linspace(-grid_extent, grid_extent, grid_size, device=env.device)
    xs, ys = torch.meshgrid(coords, coords, indexing="ij")
    cells_body = torch.stack([xs, ys], dim=-1)  # (G, G, 2)

    obs_xy_w = torch.stack(
        [env.scene[name].data.root_pos_w[:, :2] for name in obstacle_names], dim=1
    )  # (N, K, 2)
    rel_w = obs_xy_w - pos_w.unsqueeze(1)  # (N, K, 2)
    rel_x_b = cos_y * rel_w[..., 0:1] + sin_y * rel_w[..., 1:2]
    rel_y_b = -sin_y * rel_w[..., 0:1] + cos_y * rel_w[..., 1:2]
    obs_body = torch.cat([rel_x_b, rel_y_b], dim=-1)  # (N, K, 2)

    # (N, G, G, K) squared distances cell-to-obstacle
    cells = cells_body.unsqueeze(0).unsqueeze(3)
    obs_e = obs_body.unsqueeze(1).unsqueeze(1)
    dist_sq = ((cells - obs_e) ** 2).sum(dim=-1)
    # per-env per-obstacle radii (broadcast against (N, G, G, K))
    if hasattr(env, "_obstacle_radii"):
        r = env._obstacle_radii.unsqueeze(1).unsqueeze(1)  # (N, 1, 1, K)
    else:
        r = torch.full((1, 1, 1, len(obstacle_names)), obstacle_radius, device=env.device)
    occ = (dist_sq < r ** 2).any(dim=-1).float()  # (N, G, G)

    if dropout > 0.0:
        keep = (torch.rand_like(occ) > dropout).float()
        occ = occ * keep
    if noise_std > 0.0:
        occ = (occ + torch.randn_like(occ) * noise_std).clamp(0.0, 1.0)

    return occ.flatten(1)


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
