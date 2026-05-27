"""Custom ActionTerm: outer action = (alpha, phi). Internally u_nom -> CBF -> frozen locomotion."""

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

import isaaclab_tasks.manager_based.navigation.mdp as nav_mdp
from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from . import cbf

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class CBFParamsAction(ActionTerm):
    cfg: "CBFParamsActionCfg"

    def __init__(self, cfg: "CBFParamsActionCfg", env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self._env = env
        self.robot: Articulation = env.scene[cfg.asset_name]
        self._inner = nav_mdp.PreTrainedPolicyAction(cfg.inner_cfg, env)
        self._raw_actions = torch.zeros((self.num_envs, 2), device=self.device)
        self._prev_raw_actions = torch.zeros_like(self._raw_actions)
        self._obstacle_names = cfg.obstacle_names

    @property
    def action_dim(self) -> int:
        return 2

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        # raw policy output (Gaussian) -> normalized to [-1, 1] -> scaled to (alpha, phi) ranges
        a_lo, a_hi = self.cfg.alpha_range
        p_lo, p_hi = self.cfg.phi_range
        norm = self._raw_actions.clamp(-1.0, 1.0)
        alpha = a_lo + (a_hi - a_lo) * (norm[:, 0] + 1.0) * 0.5
        phi = p_lo + (p_hi - p_lo) * (norm[:, 1] + 1.0) * 0.5
        return torch.stack([alpha, phi], dim=1)

    def process_actions(self, actions: torch.Tensor):
        self._prev_raw_actions[:] = self._raw_actions
        self._raw_actions[:] = actions.clamp(-1.0, 1.0)

    @property
    def prev_raw_actions(self) -> torch.Tensor:
        return self._prev_raw_actions

    def apply_actions(self):
        from . import mdp as _cbf_mdp

        # advance kinematic drifting obstacles by their velocity each sub-step
        _cbf_mdp.advance_obstacles(self._env, self._obstacle_names, dt=self._env.physics_dt)

        # u_nom: unit vector toward goal in body frame
        goal_cmd = self._env.command_manager.get_term(self.cfg.command_name)
        goal_xy_b = goal_cmd.pos_command_b[:, :2]
        dist = torch.linalg.norm(goal_xy_b, dim=1, keepdim=True).clamp_min(1e-6)
        u_nom = torch.zeros((self.num_envs, 3), device=self.device)
        u_nom[:, :2] = goal_xy_b / dist

        # CBF projection via grid-derived sdf (deployment-realistic: only uses lidar BEV)
        params = self.processed_actions
        grid_flat = _cbf_mdp.bev_occupancy(
            self._env,
            obstacle_names=self._obstacle_names,
            obstacle_radius=self.cfg.obstacle_radius,
            grid_size=self.cfg.grid_size,
            grid_extent=self.cfg.grid_extent,
        )
        grid = grid_flat.reshape(self.num_envs, self.cfg.grid_size, self.cfg.grid_size)
        u_safe = cbf.safety_filter_grid(
            u_nom=u_nom,
            grid=grid,
            grid_extent=self.cfg.grid_extent,
            robot_radius=self.cfg.robot_radius,
            alpha=params[:, 0],
            phi=params[:, 1],
            lam=self.cfg.lam,
            gamma=self.cfg.gamma,
        )
        u_safe = torch.nan_to_num(u_safe, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)

        self._inner._raw_actions[:] = u_safe
        self._inner.apply_actions()

    def reset(self, env_ids=None):
        if env_ids is None:
            self._raw_actions.zero_()
        else:
            self._raw_actions[env_ids] = 0.0


@configclass
class CBFParamsActionCfg(ActionTermCfg):
    class_type: type = CBFParamsAction
    asset_name: str = "robot"
    inner_cfg: nav_mdp.PreTrainedPolicyActionCfg = MISSING
    obstacle_names: tuple = MISSING
    obstacle_radius: float = 0.5
    robot_radius: float = 0.3
    command_name: str = "goal_pose"
    alpha_range: tuple = (0.1, 5.0)
    phi_range: tuple = (0.01, 10.0)
    lam: float = 1.0
    gamma: float = 1.0
    grid_size: int = 16
    grid_extent: float = 3.0
