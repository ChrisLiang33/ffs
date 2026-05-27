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
        # obstacles are kinematic -> cache positions once at init
        self._obs_xy = torch.stack(
            [env.scene[name].data.root_pos_w[:, :2] for name in cfg.obstacle_names], dim=1
        )

    @property
    def action_dim(self) -> int:
        return 2

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        a_lo, a_hi = self.cfg.alpha_range
        p_lo, p_hi = self.cfg.phi_range
        alpha = self._raw_actions[:, 0].clamp(a_lo, a_hi)
        phi = self._raw_actions[:, 1].clamp(p_lo, p_hi)
        return torch.stack([alpha, phi], dim=1)

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions

    def apply_actions(self):
        # u_nom: unit vector toward goal in body frame
        goal_cmd = self._env.command_manager.get_term(self.cfg.command_name)
        goal_xy_b = goal_cmd.pos_command_b[:, :2]
        dist = torch.linalg.norm(goal_xy_b, dim=1, keepdim=True).clamp_min(1e-6)
        u_nom = torch.zeros((self.num_envs, 3), device=self.device)
        u_nom[:, :2] = goal_xy_b / dist

        # CBF projection
        params = self.processed_actions
        quat = self.robot.data.root_quat_w
        yaw = torch.atan2(
            2 * (quat[:, 0] * quat[:, 3] + quat[:, 1] * quat[:, 2]),
            1 - 2 * (quat[:, 2] ** 2 + quat[:, 3] ** 2),
        )
        u_safe = cbf.safety_filter(
            u_nom=u_nom,
            robot_xy=self.robot.data.root_pos_w[:, :2],
            robot_yaw=yaw,
            obs_xy=self._obs_xy,
            obstacle_radius=self.cfg.obstacle_radius,
            robot_radius=self.cfg.robot_radius,
            alpha=params[:, 0],
            phi=params[:, 1],
            lam=self.cfg.lam,
            gamma=self.cfg.gamma,
        )

        # hand u_safe to the frozen-locomotion inner term, then drive it
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
