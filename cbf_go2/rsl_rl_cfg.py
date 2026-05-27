"""PPO runner cfgs for the three architectures. Pick via the gym task ID:
  Isaac-Goal-Go2-v0          -> Arch A teacher (CNN + priv encoder)
  Isaac-Goal-Go2-StudentB-v0 -> Arch B (CNN, no priv)
  Isaac-Goal-Go2-FlatC-v0    -> Arch C (flat MLP, no priv, no CNN)
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


_PPO = RslRlPpoAlgorithmCfg(
    value_loss_coef=1.0,
    use_clipped_value_loss=True,
    clip_param=0.2,
    entropy_coef=0.01,
    num_learning_epochs=5,
    num_mini_batches=4,
    learning_rate=1.0e-3,
    schedule="adaptive",
    gamma=0.995,
    lam=0.95,
    desired_kl=0.01,
    max_grad_norm=1.0,
)


def _gaussian():
    return RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0, std_type="scalar")


def _actor_kwargs(class_name: str):
    return dict(
        class_name=class_name,
        hidden_dims=[128, 128, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=_gaussian(),
        stochastic=True,
        init_noise_std=1.0,
        noise_std_type="scalar",
        state_dependent_std=False,
    )


def _critic_kwargs(class_name: str):
    return dict(
        class_name=class_name,
        hidden_dims=[128, 128, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=None,
        stochastic=False,
        init_noise_std=1.0,
        noise_std_type="scalar",
        state_dependent_std=False,
    )


# ---------- Arch A: CNN + priv encoder ----------

@configclass
class TeacherActorCfg(RslRlMLPModelCfg):
    class_name: str = "cbf_go2.cnn_actor:TeacherActor"


@configclass
class TeacherCriticCfg(RslRlMLPModelCfg):
    class_name: str = "cbf_go2.cnn_actor:TeacherCritic"


@configclass
class ArchA_TeacherPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 200
    save_interval = 50
    experiment_name = "cbf_goal_go2_archA"
    obs_groups: dict = {"actor": ["policy", "priv"], "critic": ["policy", "priv"]}
    actor: TeacherActorCfg = TeacherActorCfg(**_actor_kwargs("cbf_go2.cnn_actor:TeacherActor"))
    critic: TeacherCriticCfg = TeacherCriticCfg(**_critic_kwargs("cbf_go2.cnn_actor:TeacherCritic"))
    algorithm = _PPO


# ---------- Arch B: CNN, no priv ----------

@configclass
class StudentActorCfg(RslRlMLPModelCfg):
    class_name: str = "cbf_go2.cnn_actor:StudentActor"


@configclass
class StudentCriticCfg(RslRlMLPModelCfg):
    class_name: str = "cbf_go2.cnn_actor:StudentCritic"


@configclass
class ArchB_StudentPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 200
    save_interval = 50
    experiment_name = "cbf_goal_go2_archB"
    obs_groups: dict = {"actor": ["policy"], "critic": ["policy"]}
    actor: StudentActorCfg = StudentActorCfg(**_actor_kwargs("cbf_go2.cnn_actor:StudentActor"))
    critic: StudentCriticCfg = StudentCriticCfg(**_critic_kwargs("cbf_go2.cnn_actor:StudentCritic"))
    algorithm = _PPO


# ---------- Arch C: flat MLP baseline ----------

@configclass
class FlatActorCfg(RslRlMLPModelCfg):
    pass  # uses stock MLPModel via default class_name


@configclass
class ArchC_FlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 200
    save_interval = 50
    experiment_name = "cbf_goal_go2_archC"
    obs_groups: dict = {"actor": ["policy"], "critic": ["policy"]}
    actor: FlatActorCfg = FlatActorCfg(**_actor_kwargs("MLPModel"))
    critic: FlatActorCfg = FlatActorCfg(**_critic_kwargs("MLPModel"))
    algorithm = _PPO


# ---------- Arch D: long proprio+action history + CNN ----------

@configclass
class LongHistActorCfg(RslRlMLPModelCfg):
    class_name: str = "cbf_go2.cnn_actor:LongHistActor"


@configclass
class LongHistCriticCfg(RslRlMLPModelCfg):
    class_name: str = "cbf_go2.cnn_actor:LongHistCritic"


@configclass
class ArchD_LongHistPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 200
    save_interval = 50
    experiment_name = "cbf_goal_go2_archD"
    obs_groups: dict = {"actor": ["policy"], "critic": ["policy"]}
    actor: LongHistActorCfg = LongHistActorCfg(**_actor_kwargs("cbf_go2.cnn_actor:LongHistActor"))
    critic: LongHistCriticCfg = LongHistCriticCfg(**_critic_kwargs("cbf_go2.cnn_actor:LongHistCritic"))
    algorithm = _PPO


# Back-compat alias
CBFGoalPPORunnerCfg = ArchA_TeacherPPORunnerCfg
