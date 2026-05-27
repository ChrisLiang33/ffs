"""Goal-reaching task for Unitree Go2 with a learned CBF safety filter."""

import gymnasium as gym

gym.register(
    id="Isaac-Goal-Go2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GoalGo2EnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.rsl_rl_cfg:ArchA_TeacherPPORunnerCfg",
    },
)
gym.register(
    id="Isaac-Goal-Go2-StudentB-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GoalGo2EnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.rsl_rl_cfg:ArchB_StudentPPORunnerCfg",
    },
)
gym.register(
    id="Isaac-Goal-Go2-FlatC-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GoalGo2EnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.rsl_rl_cfg:ArchC_FlatPPORunnerCfg",
    },
)
gym.register(
    id="Isaac-Goal-Go2-LongHistD-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GoalGo2EnvCfgArchD",
        "rsl_rl_cfg_entry_point": f"{__name__}.rsl_rl_cfg:ArchD_LongHistPPORunnerCfg",
    },
)
