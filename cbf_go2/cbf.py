"""CBF safety filter — smoothed SDF over cylinder obstacles, closed-form half-space projection.

Constraint (with a, b, c dropped, drift-aware via L_f h):

    L_f h + A . u_xy + alpha * h - phi * ||A||^2 >= 0

Half-space form:
    A . u_xy >= phi * ||A||^2 - alpha * h - L_f h

where
    sdf(x)      = min_i ||p - rho_i|| - (R_obs_i + R_robot)  with per-env per-obstacle R_obs_i
    h(x)        = lambda * (1 - exp(-gamma * sdf))
    A           = world->body-frame rotation applied to grad_h_world
    L_f h       = -lambda * gamma * exp(-gamma sdf) * (n_hat . v_closest_obs)
The omega_z component of u is unconstrained (CBF only sees planar position).
"""

import torch


def compute_sdf(
    robot_xy: torch.Tensor,
    obs_xy: torch.Tensor,
    obstacle_radii: torch.Tensor,
    robot_radius: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Hard-min SDF, world-frame gradient, and argmin index.

    robot_xy:        (N, 2)
    obs_xy:          (N, K, 2)
    obstacle_radii:  (N, K) per-env per-obstacle radii
    returns: sdf (N,), sdf_grad (N, 2) unit vec from obstacle to robot, idx (N,) argmin
    """
    rel = robot_xy.unsqueeze(1) - obs_xy
    dist = torch.linalg.norm(rel, dim=-1).clamp_min(1e-6)
    per_obs_sdf = dist - (obstacle_radii + robot_radius)
    sdf, idx = per_obs_sdf.min(dim=-1)

    closest_rel = torch.gather(rel, 1, idx[:, None, None].expand(-1, 1, 2)).squeeze(1)
    closest_dist = torch.gather(dist, 1, idx[:, None]).squeeze(-1)
    sdf_grad = closest_rel / closest_dist.unsqueeze(-1)
    return sdf, sdf_grad, idx


def compute_h(
    sdf: torch.Tensor,
    sdf_grad: torch.Tensor,
    lam: float = 1.0,
    gamma: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """h_smooth = lambda * (1 - exp(-gamma * sdf)) and its gradient.

    Clamp the exp arg at 5 so deeply-negative sdf can't blow up |grad|.
    """
    arg = (-gamma * sdf).clamp(max=5.0)
    exp_arg = torch.exp(arg)
    h = lam * (1.0 - exp_arg)
    h_grad = (lam * gamma * exp_arg).unsqueeze(-1) * sdf_grad
    return h, h_grad


def sdf_from_grid(
    grid: torch.Tensor,
    grid_extent: float,
    robot_radius: float,
    threshold: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """SDT at the body-frame origin (where the robot is): min over occupied cells.

    grid: (N, G, G) body-frame occupancy in [0, 1]
    returns: sdf (N,), sdf_grad_body (N, 2) unit vec from nearest occupied cell to origin.
    """
    N, G, _ = grid.shape
    coords = torch.linspace(-grid_extent, grid_extent, G, device=grid.device)
    xs, ys = torch.meshgrid(coords, coords, indexing="ij")
    cell_dist = (xs ** 2 + ys ** 2).sqrt()  # (G, G)

    occupied = grid > threshold
    inf = torch.full_like(grid, float("inf"))
    masked = torch.where(occupied, cell_dist.unsqueeze(0).expand_as(grid), inf)
    min_dist, flat_idx = masked.view(N, -1).min(dim=-1)
    sdf = min_dist - robot_radius

    i_idx = flat_idx // G
    j_idx = flat_idx % G
    cell_x = coords[i_idx]
    cell_y = coords[j_idx]
    norm = (cell_x ** 2 + cell_y ** 2).sqrt().clamp_min(1e-6)
    sdf_grad_body = torch.stack([-cell_x / norm, -cell_y / norm], dim=-1)

    no_obstacle = torch.isinf(min_dist)
    sdf = torch.where(no_obstacle, torch.full_like(sdf, 100.0), sdf)
    sdf_grad_body = torch.where(
        no_obstacle.unsqueeze(-1), torch.zeros_like(sdf_grad_body), sdf_grad_body
    )
    return sdf, sdf_grad_body


def safety_filter_grid(
    u_nom: torch.Tensor,
    grid: torch.Tensor,
    grid_extent: float,
    robot_radius: float,
    alpha,
    phi,
    lam: float = 1.0,
    gamma: float = 1.0,
) -> torch.Tensor:
    """Lidar-grid CBF: sdf from BEV directly, no obstacle positions needed. No L_f h."""
    sdf, sdf_grad_body = sdf_from_grid(grid, grid_extent, robot_radius)
    h, h_grad_body = compute_h(sdf, sdf_grad_body, lam=lam, gamma=gamma)
    A = h_grad_body  # body-frame already
    A_norm_sq = (A * A).sum(dim=-1).clamp_min(1e-12)
    rhs = phi * A_norm_sq - alpha * h
    u_xy = u_nom[..., :2]
    violation = (rhs - (A * u_xy).sum(dim=-1)).clamp_min(0.0)
    u_xy_safe = u_xy + (violation / A_norm_sq).unsqueeze(-1) * A
    u_safe = u_nom.clone()
    u_safe[..., :2] = u_xy_safe
    return u_safe


def safety_filter(
    u_nom: torch.Tensor,
    robot_xy: torch.Tensor,
    robot_yaw: torch.Tensor,
    obs_xy: torch.Tensor,
    obstacle_radii: torch.Tensor,
    robot_radius: float,
    alpha: float,
    phi: float,
    obs_velocities: torch.Tensor | None = None,
    lam: float = 1.0,
    gamma: float = 1.0,
) -> torch.Tensor:
    """Project u_nom onto {u : A . u_xy >= phi ||A||^2 - alpha h - L_f h}.

    obs_velocities: (N, K, 2) per-env per-obstacle world-frame velocities (None -> all zero).
    Returns (N, 3) u_safe.
    """
    sdf, sdf_grad, idx = compute_sdf(robot_xy, obs_xy, obstacle_radii, robot_radius)
    h, h_grad_w = compute_h(sdf, sdf_grad, lam=lam, gamma=gamma)

    # body-frame gradient
    cos_y = torch.cos(robot_yaw)
    sin_y = torch.sin(robot_yaw)
    A_x = cos_y * h_grad_w[:, 0] + sin_y * h_grad_w[:, 1]
    A_y = -sin_y * h_grad_w[:, 0] + cos_y * h_grad_w[:, 1]
    A = torch.stack([A_x, A_y], dim=-1)
    A_norm_sq = (A * A).sum(dim=-1).clamp_min(1e-12)

    # L_f h: only the argmin obstacle's velocity contributes
    if obs_velocities is not None:
        v_closest = torch.gather(obs_velocities, 1, idx[:, None, None].expand(-1, 1, 2)).squeeze(1)
        dh_dsdf = (lam * gamma) * torch.exp((-gamma * sdf).clamp(max=5.0))
        L_f_h = -dh_dsdf * (sdf_grad * v_closest).sum(dim=-1)
    else:
        L_f_h = torch.zeros_like(sdf)

    rhs = phi * A_norm_sq - alpha * h - L_f_h
    u_xy = u_nom[..., :2]
    violation = (rhs - (A * u_xy).sum(dim=-1)).clamp_min(0.0)
    u_xy_safe = u_xy + (violation / A_norm_sq).unsqueeze(-1) * A

    u_safe = u_nom.clone()
    u_safe[..., :2] = u_xy_safe
    return u_safe
