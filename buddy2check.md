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
- Outer rate: 50 Hz (matches inner locomotion). PD: 200 Hz.
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

## 4. Architecture

- CBF moved *inside* the env as a custom `ActionTerm` ([cbf_go2/cbf_params_action.py](cbf_go2/cbf_params_action.py)). Outer action space is exactly `(alpha, phi)`, normalized [-1, 1] then scaled to ranges.
- ISSf baseline is now "fixed `(alpha=2.0, phi=0.5)` action through the same env" — no separate code path.
- RL ([cbf_go2/rsl_rl_cfg.py](cbf_go2/rsl_rl_cfg.py)): PPO via rsl_rl, 3×128 ELU MLP, 4096 envs, 200 iters (~7 min on RTX 5090).
- Rewards: dense progress (velocity toward goal) + goal_bonus (+50) + crash_penalty (-50) + timeout_penalty (-50) + action_rate (-0.05). Episode = 10s during training.

## 5. ISSf vs TISSf vs RL — proper eval

[eval_cbf.py](eval_cbf.py), 256 envs × 2000 outer steps, episode_length_s = 30, 1200-1600 episodes per condition. Wilson 95% CIs. Per-episode data dumped to `results/*.json`.

TISSf formulation (Wang et al.):  `ε(h) = ε₀ · exp(λ·h)`,  `φ = 1/ε(h)`.  λ=0 recovers ISSf with fixed φ = 1/ε₀.

| method | reach | crash | timeout | time-to-reach |
| --- | --- | --- | --- | --- |
| ISSf (α=2, φ=0.5) | 90.5% [88.9, 92.0] | 4.0% [3.1, 5.1] | 5.5% | 17.6 sec |
| TISSf best (ε₀=2, λ=1.5) | 89.5% [87.8, 91.0] | 3.6% [2.7, 4.7] | 6.9% | 17.3 sec |
| **RL (model_199)** | **94.3% [93.1, 95.4]** | **0.5% [0.3, 1.0]** | **5.2%** | 18.2 sec |

**RL dominates both baselines on safety and reach.** TISSf is basically tied with ISSf — CIs overlap.

**Why TISSf doesn't beat ISSf here:** our scene is too dense (4 obstacles in a plus pattern at distance 1.2m, goals in (-3,3)²) — the robot is always near some obstacle, so TISSf's "low φ in interior" advantage doesn't kick in. A sparse scene should show TISSf catching up. RL still wins because it learns directional context (which side of an obstacle to detour given the goal direction) — something a φ(h)-only hand rule can't capture.

**TISSf sweep** (6 configs of (ε₀, λ)):

| ε₀ | λ | φ at boundary | φ in interior | reach | crash | timeout |
| --- | --- | --- | --- | --- | --- | --- |
| 0.5 | 0.5 | 2.0 | 1.21 | 30% | 28% | 42% |
| 1.0 | 0.5 | 1.0 | 0.61 | 85% | 6% | 9% |
| 2.0 | 0.5 | 0.5 | 0.30 | 89% | 4% | 7% |
| 0.5 | 1.5 | 2.0 | 0.45 | 58% | 22% | 20% |
| 1.0 | 1.5 | 1.0 | 0.22 | 86% | 5% | 9% |
| 2.0 | 1.5 | 0.5 | 0.11 | 89.5% | 3.6% | 6.9% |

Anything with φ(boundary) > 1 crashes a lot — the locomotion can't track the high tangential pushback. φ(boundary) ≈ 0.5 (matching ISSf) gives the best TISSf, but it's not better than ISSf.

Note: prior smoke runs gave ISSf 95%/1% — those were small-sample noise (~50-150 eps). Real ISSf is ~90%/4%.

## Things to push back on

- **`a, b, c` dropped** (not just fixed). Means we lose the SOC structure entirely, so the QP stays a clean half-space projection. We can revisit if the learned φ stalls.
- **Robot radius (0.3m) is a free parameter.** Bigger = more buffer, but more conservative. Picked between half-width and half-diagonal; we should sweep this once RL is running.
- **Smooth-min via the exp transform, not log-sum-exp.** The paper's formulation: `h = lambda(1 - exp(-gamma·sdf))` with hard min on sdf. ∇h is discontinuous at obstacle-equidistance lines but its magnitude is small there because exp(-gamma·sdf) decays. In practice no chattering so far.
- **lambda = gamma = 1** — coupled with φ (λγ is the gradient magnitude at the boundary). Tuning λ, γ separately is largely redundant with tuning φ.
- **4 obstacles, fixed positions.** Trivial scene. Once RL is training, randomize per-reset.

## Next

Reward function + PPO training: learn `(alpha, phi)` to beat the ISSf baseline on time-to-goal at matched (or better) crash rate. Then TISSf baseline (φ as a hand-coded function of state) as the harder bar to clear.
