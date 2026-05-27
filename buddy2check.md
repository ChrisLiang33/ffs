# Where we are — please sanity-check

## 1. Locomotion (done)

Stock `Isaac-Velocity-Flat-Unitree-Go2-v0`, 4096 envs, 300 PPO iters.
Checkpoint: `~/IsaacLab/logs/rsl_rl/unitree_go2_flat/2026-05-26_20-00-11/model_299.pt`.
[eval.py](eval.py) on the train env (noise + push events on):

| metric    | target      | result              |
|-----------|-------------|---------------------|
| vx MAE    | < 0.1 m/s   | 0.050               |
| vy MAE    | < 0.1 m/s   | 0.058               |
| wz MAE    | < 0.2 rad/s | 0.100               |
| fall rate | < 1%        | 0% (0/128 episodes) |

Frozen from here.

## 2. Goal-reaching env (done)

[cbf_go2/](cbf_go2/) — pip-installable extension that registers `Isaac-Goal-Go2-v0`.

- Scene: flat ground + Go2 + 4 cylinder obstacles (r=0.5m) in a plus pattern at distance 1.2m.
- Goal: `UniformPose2dCommand` sampled in (-3, 3)² each reset. Visualized as a 2m green pole.
- Outer action: `(alpha, phi)` — the CBF params, 2-d.
- Inner: u_nom (unit vector to goal in body frame) → CBF safety filter → frozen locomotion checkpoint → joint targets → PD.
- Outer rate: 5 Hz. Inner locomotion: 50 Hz. PD: 200 Hz.
- Terminations: time_out, base_contact, goal_reached (within 0.3m), obstacle_hit (robot center within 0.65m of any obstacle).

## 3. CBF safety filter (done)

[cbf_go2/cbf.py](cbf_go2/cbf.py). The constraint with `a, b, c` dropped:

```text
A · u_xy + alpha · h - phi · ||A||² >= 0
```

- `sdf(x) = min_i ||p - rho_i|| - (R_obs + R_robot)` (hard-min over 4 obstacles).
- `h_smooth(x) = lambda · (1 - exp(-gamma · sdf))`, default lambda=1, gamma=1.
- `A` = ∇h in body frame (rotate world gradient by robot yaw).
- Single linear constraint in `u_xy` → closed-form half-space projection. `omega_z` unconstrained. No QP solver.
- Robot radius = 0.3m (between Go2's half-width 0.16 and half-diagonal 0.36, conservative-ish).

## 4. ISSf baseline numbers

[smoke.py](smoke.py) — fixed `(alpha, phi)` over ~150 episodes on rough flat:

| setup                | reach | crash | timeout |
|----------------------|-------|-------|---------|
| no CBF (unsafe)      | 63%   | 36%   | 1%      |
| CBF α=2.0 φ=0.2      | 90%   | 6%    | 4%      |
| CBF α=2.0 φ=0.5      | 86%   | 6%    | 7%      |
| CBF α=2.0 φ=1.0      | –     | –     | –       |
| CBF α=2.0 φ=2.0      | 0%    | 35%   | 65%     |

(φ=1.0 was only run in 1-env video mode — too small a sample to report.)
φ=2 is over-conservative — robot can't make progress, so it sits and eventually times out / drifts into obstacles during the dwell. φ=0.5 is a reasonable ISSf baseline ([6%] residual crash is from locomotion tracking error of ~0.1 m/s; a perfectly executed CBF would be 0%).

## 5. Architecture (done as of this commit)

- CBF moved *inside* the env as a custom `ActionTerm` ([cbf_go2/cbf_params_action.py](cbf_go2/cbf_params_action.py)). Outer action space is exactly `(alpha, phi)`.
- The RL policy will plug in next — same env, just send `(alpha, phi)` from a policy instead of a fixed constant.
- ISSf baseline is now "fixed `(alpha=2.0, phi=0.5)` action through the same env" — no separate code path.

## Things to push back on

- **`a, b, c` dropped** (not just fixed). Means we lose the SOC structure entirely, so the QP stays a clean half-space projection. We can revisit if the learned φ stalls.
- **Robot radius (0.3m) is a free parameter.** Bigger = more buffer, but more conservative. Picked between half-width and half-diagonal; we should sweep this once RL is running.
- **Smooth-min via the exp transform, not log-sum-exp.** The paper's formulation: `h = lambda(1 - exp(-gamma·sdf))` with hard min on sdf. ∇h is discontinuous at obstacle-equidistance lines but its magnitude is small there because exp(-gamma·sdf) decays. In practice no chattering so far.
- **lambda = gamma = 1** — coupled with φ (λγ is the gradient magnitude at the boundary). Tuning λ, γ separately is largely redundant with tuning φ.
- **4 obstacles, fixed positions.** Trivial scene. Once RL is training, randomize per-reset.

## Next

Reward function + PPO training: learn `(alpha, phi)` to beat the ISSf baseline on time-to-goal at matched (or better) crash rate. Then TISSf baseline (φ as a hand-coded function of state) as the harder bar to clear.
