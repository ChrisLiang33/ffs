"""CBF safety filter — smoothed SDF over cylinder obstacles, closed-form half-space projection.

Constraint (with a, b, c dropped):
    A · u_xy + alpha * h - phi * ||A||^2 >= 0
where
    sdf(x)      = min_i ||p - rho_i|| - (R_obs + R_robot)
    h(x)        = lambda * (1 - exp(-gamma * sdf))
    A           = world->body-frame rotation applied to grad_h_world
The omega_z component of u is unconstrained (CBF only sees planar position).
"""

import torch


def compute_sdf(
    robot_xy: torch.Tensor,
    obs_xy: torch.Tensor,
    obstacle_radius: float,
    robot_radius: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Hard-min SDF and its world-frame gradient.

    robot_xy: (N, 2), obs_xy: (N, K, 2)
    returns: sdf (N,), sdf_grad (N, 2)
    """
    rel = robot_xy.unsqueeze(1) - obs_xy
    dist = torch.linalg.norm(rel, dim=-1).clamp_min(1e-6)
    per_obs_sdf = dist - (obstacle_radius + robot_radius)
    sdf, idx = per_obs_sdf.min(dim=-1)

    closest_rel = torch.gather(rel, 1, idx[:, None, None].expand(-1, 1, 2)).squeeze(1)
    closest_dist = torch.gather(dist, 1, idx[:, None]).squeeze(-1)
    sdf_grad = closest_rel / closest_dist.unsqueeze(-1)
    return sdf, sdf_grad


def compute_h(
    sdf: torch.Tensor,
    sdf_grad: torch.Tensor,
    lam: float = 1.0,
    gamma: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """h_smooth = lambda * (1 - exp(-gamma * sdf)) and its gradient via chain rule."""
    h = lam * (1.0 - torch.exp(-gamma * sdf))
    dh_dsdf = lam * gamma * torch.exp(-gamma * sdf)
    h_grad = dh_dsdf.unsqueeze(-1) * sdf_grad
    return h, h_grad


def safety_filter(
    u_nom: torch.Tensor,
    robot_xy: torch.Tensor,
    robot_yaw: torch.Tensor,
    obs_xy: torch.Tensor,
    obstacle_radius: float,
    robot_radius: float,
    alpha: float,
    phi: float,
    lam: float = 1.0,
    gamma: float = 1.0,
) -> torch.Tensor:
    """Project u_nom onto {u : A . u_xy >= phi ||A||^2 - alpha h}. Returns (N, 3) u_safe."""
    sdf, sdf_grad = compute_sdf(robot_xy, obs_xy, obstacle_radius, robot_radius)
    h, h_grad_w = compute_h(sdf, sdf_grad, lam=lam, gamma=gamma)

    cos_y = torch.cos(robot_yaw)
    sin_y = torch.sin(robot_yaw)
    A_x = cos_y * h_grad_w[:, 0] + sin_y * h_grad_w[:, 1]
    A_y = -sin_y * h_grad_w[:, 0] + cos_y * h_grad_w[:, 1]
    A = torch.stack([A_x, A_y], dim=-1)

    A_norm_sq = (A * A).sum(dim=-1).clamp_min(1e-12)
    rhs = phi * A_norm_sq - alpha * h
    u_xy = u_nom[..., :2]
    violation = (rhs - (A * u_xy).sum(dim=-1)).clamp_min(0.0)
    u_xy_safe = u_xy + (violation / A_norm_sq).unsqueeze(-1) * A

    u_safe = u_nom.clone()
    u_safe[..., :2] = u_xy_safe
    return u_safe
