"""Arch A teacher: CNN over bev_history + MLP over proprio + encoder over priv DR values.

Input obs layout (after rsl_rl concatenates "policy" then "priv"):
  [0  : 15 ] proprio   (base_lin_vel(3) + base_ang_vel(3) + projected_gravity(3) + goal(4) + actions(2))
  [15 : 783] bev_history (3 frames * 16 * 16 = 768, flattened in (T, H, W) order)
  [783: 786] priv DR values (static_friction, dynamic_friction, mass_delta)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import MLP

PROPRIO_DIM = 15
BEV_FRAMES = 3
BEV_G = 16
BEV_DIM = BEV_FRAMES * BEV_G * BEV_G  # 768
POLICY_DIM = PROPRIO_DIM + BEV_DIM    # 783
# priv: friction(2) + mass_delta(1) + body_height(1) = 4
PRIV_DIM = 4
Z_DIM = 16


class _TeacherEncoder(nn.Module):
    """Three-branch encoder: proprio MLP + bev CNN + priv MLP -> fused -> output_dim."""

    def __init__(self, output_dim: int, has_priv: bool = True) -> None:
        super().__init__()
        self.has_priv = has_priv

        self.cnn = nn.Sequential(
            nn.Conv2d(BEV_FRAMES, 16, kernel_size=3, stride=2, padding=1),  # -> 8x8
            nn.ELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # -> 4x4
            nn.ELU(),
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 64),
            nn.ELU(),
        )
        self.proprio_mlp = nn.Sequential(
            nn.Linear(PROPRIO_DIM, 64),
            nn.ELU(),
        )
        if has_priv:
            self.priv_encoder = nn.Sequential(
                nn.Linear(PRIV_DIM, 32),
                nn.ELU(),
                nn.Linear(32, Z_DIM),
                nn.ELU(),
            )
            fuse_in = 64 + 64 + Z_DIM
        else:
            self.priv_encoder = None
            fuse_in = 64 + 64

        self.head = nn.Sequential(
            nn.Linear(fuse_in, 128),
            nn.ELU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        proprio = x[:, :PROPRIO_DIM]
        bev = x[:, PROPRIO_DIM:POLICY_DIM].reshape(n, BEV_FRAMES, BEV_G, BEV_G)
        cnn_feat = self.cnn(bev)
        prop_feat = self.proprio_mlp(proprio)
        feats = [prop_feat, cnn_feat]
        if self.has_priv:
            priv = x[:, POLICY_DIM:POLICY_DIM + PRIV_DIM]
            z = self.priv_encoder(priv)
            feats.append(z)
        return self.head(torch.cat(feats, dim=-1))

    def init_weights(self, scales) -> None:
        pass  # default torch init is fine for conv/linear


class TeacherActor(MLPModel):
    """Drop-in MLPModel subclass that swaps self.mlp for the three-branch encoder."""

    def __init__(
        self,
        obs,
        obs_groups,
        obs_set,
        output_dim,
        hidden_dims=(128, 128, 128),
        activation="elu",
        obs_normalization=False,
        distribution_cfg=None,
        # Deprecated configclass fields — accepted for compat, ignored
        stochastic=True,
        init_noise_std=1.0,
        noise_std_type="scalar",
        state_dependent_std=False,
    ) -> None:
        super().__init__(
            obs=obs,
            obs_groups=obs_groups,
            obs_set=obs_set,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            obs_normalization=obs_normalization,
            distribution_cfg=distribution_cfg,
        )
        mlp_out = self.distribution.input_dim if self.distribution is not None else output_dim
        self.mlp = _TeacherEncoder(output_dim=mlp_out, has_priv=True)
        if self.distribution is not None:
            self.distribution.init_mlp_weights(self.mlp)


class TeacherCritic(MLPModel):
    """Same encoder as the actor (own weights), output = 1 (value)."""

    def __init__(
        self,
        obs,
        obs_groups,
        obs_set,
        output_dim,
        hidden_dims=(128, 128, 128),
        activation="elu",
        obs_normalization=False,
        distribution_cfg=None,
    ) -> None:
        super().__init__(
            obs=obs,
            obs_groups=obs_groups,
            obs_set=obs_set,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            obs_normalization=obs_normalization,
            distribution_cfg=distribution_cfg,
        )
        self.mlp = _TeacherEncoder(output_dim=output_dim, has_priv=True)


# ----- Arch B: CNN over bev history, NO priv encoder (deploy-realistic) -----

class StudentActor(MLPModel):
    """Same architecture as TeacherActor but priv branch disabled."""

    def __init__(self, obs, obs_groups, obs_set, output_dim,
                 hidden_dims=(128, 128, 128), activation="elu", obs_normalization=False,
                 distribution_cfg=None, stochastic=True, init_noise_std=1.0,
                 noise_std_type="scalar", state_dependent_std=False):
        super().__init__(
            obs=obs, obs_groups=obs_groups, obs_set=obs_set, output_dim=output_dim,
            hidden_dims=hidden_dims, activation=activation,
            obs_normalization=obs_normalization, distribution_cfg=distribution_cfg,
        )
        mlp_out = self.distribution.input_dim if self.distribution is not None else output_dim
        self.mlp = _TeacherEncoder(output_dim=mlp_out, has_priv=False)
        if self.distribution is not None:
            self.distribution.init_mlp_weights(self.mlp)


class StudentCritic(MLPModel):
    def __init__(self, obs, obs_groups, obs_set, output_dim,
                 hidden_dims=(128, 128, 128), activation="elu", obs_normalization=False,
                 distribution_cfg=None):
        super().__init__(
            obs=obs, obs_groups=obs_groups, obs_set=obs_set, output_dim=output_dim,
            hidden_dims=hidden_dims, activation=activation,
            obs_normalization=obs_normalization, distribution_cfg=distribution_cfg,
        )
        self.mlp = _TeacherEncoder(output_dim=output_dim, has_priv=False)


# ----- Arch D: long proprio+action history + CNN over bev -----

# Layout (after rsl_rl concatenation, policy group only — history_length=50 on
# base_lin_vel(3) + base_ang_vel(3) + projected_gravity(3) + actions(2), history=1 on goal(4)):
#   50*(3+3+3+2) + 4 = 554 long-history "proprio", then bev (3*16*16 = 768).
LONGHIST_PROPRIO_DIM = 50 * (3 + 3 + 3 + 2) + 4   # 554


class _LongHistEncoder(nn.Module):
    """Long proprio/action history MLP + bev CNN, no priv."""

    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(BEV_FRAMES, 16, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 64),
            nn.ELU(),
        )
        self.proprio_mlp = nn.Sequential(
            nn.Linear(LONGHIST_PROPRIO_DIM, 256),
            nn.ELU(),
            nn.Linear(256, 64),
            nn.ELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(64 + 64, 128),
            nn.ELU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        proprio = x[:, :LONGHIST_PROPRIO_DIM]
        bev = x[:, LONGHIST_PROPRIO_DIM:LONGHIST_PROPRIO_DIM + BEV_DIM].reshape(n, BEV_FRAMES, BEV_G, BEV_G)
        prop_feat = self.proprio_mlp(proprio)
        cnn_feat = self.cnn(bev)
        return self.head(torch.cat([prop_feat, cnn_feat], dim=-1))

    def init_weights(self, scales) -> None:
        pass


class LongHistActor(MLPModel):
    def __init__(self, obs, obs_groups, obs_set, output_dim,
                 hidden_dims=(128, 128, 128), activation="elu", obs_normalization=False,
                 distribution_cfg=None, stochastic=True, init_noise_std=1.0,
                 noise_std_type="scalar", state_dependent_std=False):
        super().__init__(
            obs=obs, obs_groups=obs_groups, obs_set=obs_set, output_dim=output_dim,
            hidden_dims=hidden_dims, activation=activation,
            obs_normalization=obs_normalization, distribution_cfg=distribution_cfg,
        )
        mlp_out = self.distribution.input_dim if self.distribution is not None else output_dim
        self.mlp = _LongHistEncoder(output_dim=mlp_out)
        if self.distribution is not None:
            self.distribution.init_mlp_weights(self.mlp)


class LongHistCritic(MLPModel):
    def __init__(self, obs, obs_groups, obs_set, output_dim,
                 hidden_dims=(128, 128, 128), activation="elu", obs_normalization=False,
                 distribution_cfg=None):
        super().__init__(
            obs=obs, obs_groups=obs_groups, obs_set=obs_set, output_dim=output_dim,
            hidden_dims=hidden_dims, activation=activation,
            obs_normalization=obs_normalization, distribution_cfg=distribution_cfg,
        )
        self.mlp = _LongHistEncoder(output_dim=output_dim)
