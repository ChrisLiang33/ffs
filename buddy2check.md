# Locomotion — done, please sanity-check

Trained the stock Isaac Lab Go2 velocity-tracking policy on flat terrain as the
frozen low-level controller for the rest of the project.

- Task: `Isaac-Velocity-Flat-Unitree-Go2-v0`, stock PPO via `rsl_rl`, 4096 envs, 300 iters.
- Checkpoint: `~/IsaacLab/logs/rsl_rl/unitree_go2_flat/2026-05-26_20-00-11/model_299.pt`.
- Eval: [eval.py](eval.py), 2000 steps x 64 envs on the train env (incl. obs noise + push events).

| metric    | target      | result              |
|-----------|-------------|---------------------|
| vx MAE    | < 0.1 m/s   | 0.050               |
| vy MAE    | < 0.1 m/s   | 0.058               |
| wz MAE    | < 0.2 rad/s | 0.100               |
| fall rate | < 1%        | 0% (0/128 episodes) |

Margins look healthy. 128 episodes is light for a tight fall-rate claim, but 0
falls across ~256k env-steps is enough signal for a frozen locomotion layer
we're not planning to touch again.

**Things to push back on:**

- Flat over rough — deliberate (keep locomotion easy so we can spend budget on
  the CBF layer). If goal-reaching scenes turn out non-flat we'll revisit.
- Eval is on the train task (`-v0`), not `-Play-v0`. On purpose: keeps the
  noise + push regime the policy was trained against, closer to deployment.

Next: CBF layer (h(x), Lie derivatives, closed-form QP). Locomotion treated as
fixed from here.
