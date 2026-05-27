"""Headless eval: velocity-tracking MAE + fall rate for a trained Go2 policy.

Run on the lab box:
    ./isaaclab.sh -p /path/to/ffs/eval.py \
        --task Isaac-Velocity-Flat-Unitree-Go2-v0 --steps 2000
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-Velocity-Flat-Unitree-Go2-v0")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=2000)
parser.add_argument("--checkpoint", default=None)
parser.add_argument("--agent", default="rsl_rl_cfg_entry_point")
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
args.headless = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import importlib.metadata as metadata
import os

import gymnasium as gym
import torch
from packaging import version
from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

rsl_rl_version = metadata.version("rsl-rl-lib")


@hydra_task_config(args.task, args.agent)
def main(env_cfg, agent_cfg: RslRlBaseRunnerCfg):
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = agent_cfg.seed
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, rsl_rl_version)

    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    resume = args.checkpoint or get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)

    env = gym.make(args.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    term_mgr = unwrapped.termination_manager
    device = env.unwrapped.device

    abs_err = torch.zeros(3, device=device)
    n_obs = 0
    n_falls = 0
    n_episodes = 0

    obs = env.get_observations()
    for _ in range(args.steps):
        with torch.inference_mode():
            obs, _, dones, _ = env.step(policy(obs))
            if version.parse(rsl_rl_version) >= version.parse("4.0.0"):
                policy.reset(dones)

            cmd = unwrapped.command_manager.get_command("base_velocity")  # [N, 3]
            actual = torch.cat(
                [robot.data.root_lin_vel_b[:, :2], robot.data.root_ang_vel_b[:, 2:3]], dim=1
            )
            abs_err += (cmd - actual).abs().sum(dim=0)
            n_obs += cmd.shape[0]

            n_falls += int(term_mgr.terminated.sum().item())
            n_episodes += int(dones.sum().item())

    mae = (abs_err / max(n_obs, 1)).tolist()
    fall_rate = n_falls / max(n_episodes, 1)
    print()
    print(f"checkpoint:   {resume}")
    print(f"task:         {args.task}")
    print(f"steps:        {args.steps} x {args.num_envs} envs")
    print(f"velocity MAE: vx={mae[0]:.3f} m/s   vy={mae[1]:.3f} m/s   wz={mae[2]:.3f} rad/s")
    print(f"fall rate:    {fall_rate:.2%}   ({n_falls}/{n_episodes} ended in fall)")
    print()

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
