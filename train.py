"""Wrapper for the stock rsl_rl train.py that registers Isaac-Goal-Go2-v0 first.

Usage (on lab box):
    ~/IsaacLab/isaaclab.sh -p train.py --task Isaac-Goal-Go2-v0 --num_envs 4096 --headless
"""

import os
import runpy
import sys

import cbf_go2  # noqa: F401  registers the task with gym before train.py looks it up

ISAACLAB = os.environ.get("ISAACLAB", os.path.expanduser("~/IsaacLab"))
train_script = os.path.join(ISAACLAB, "scripts/reinforcement_learning/rsl_rl/train.py")
sys.path.insert(0, os.path.dirname(train_script))  # so the script's local `cli_args` import resolves

runpy.run_path(train_script, run_name="__main__")
