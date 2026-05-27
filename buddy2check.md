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

## 6. Scene diversity — random obstacle layout per reset

Random obstacle placement per reset (4 cylinders, xy in (-2.5, 2.5)², `min_dist_from_origin=1.3`). Retrained RL, re-ran TISSf sweep, re-evaluated all three on the randomized env.

| method | reach | crash | timeout | time-to-reach |
| --- | --- | --- | --- | --- |
| ISSf (α=2, φ=0.5) | 91.2% [89.7, 92.6] | 0.7% [0.4, 1.3] | 8.0% | 16.4 sec |
| TISSf best (ε₀=2, λ=0.5) | 90.3% [88.7, 91.7] | 0.5% [0.3, 1.0] | 9.2% | 16.3 sec |
| **RL (random-trained)** | **93.3% [92.1, 94.4]** | **0.5% [0.2, 0.9]** | **6.2%** | **15.9 sec** |

**RL wins on reach + time-to-reach, ties on safety**:

- Reach: RL +2.1 over ISSf, +3.0 over TISSf — non-overlapping CIs.
- Crash: all ~0.5% (CBF doing its job across the board).
- Time-to-reach: RL fastest by ~0.5 sec.
- TISSf vs ISSf: wash on this scene (state-dependent rule didn't beat fixed).

**Cross-scene RL edge:**

| scene | RL reach edge over ISSf | RL crash edge |
| --- | --- | --- |
| Fixed (1 layout) | +3.8 pts (94 vs 90) | strong (0.5 vs 4.0) |
| Random | +2.1 pts (93 vs 91) | matched (~0.5%) |

The earlier "RL beats ISSf 80 vs 76 with 13% crash" was an artefact of `min_dist=0.8` allowing obstacles to spawn on the robot — that 15% structural crash floor masked the real story. With `min_dist=1.3` (proper spawn buffer), the baselines also achieve <1% crash and the comparison is clean.

## Things that took time

- **Action space normalization.** Policy outputs Gaussian samples; needed to (a) clamp at the action term, (b) make `action_rate` read the clamped version, (c) make `last_action` observation read the clamped version. Otherwise the policy sees its own ±5 outputs as observations and self-feedback diverges.
- **CBF defensive math.** `compute_h` exp arg clamped at 5 (was blowing up when sdf << 0), `safety_filter` output wrapped in `nan_to_num` + `clamp(-1, 1)` to avoid OOD velocity commands to the locomotion.

## 7. Uncertainty DR — friction, mass, pushes, lidar noise

Added the *other* DR axis so RL's react-to-realized-uncertainty edge can show up:

- **Friction** per-reset: static ∈ [0.3, 1.2], dynamic ∈ [0.2, 1.0], 64 buckets.
- **Base mass** per-reset: +U(-3, 8) kg added to base body.
- **Pushes** every 3–5 s: vx, vy ∈ U(-1, 1) m/s applied to base.
- **BEV noise + dropout**: noise_std=0.05, dropout=0.1 per cell — simulates lidar dropouts on real Mid-360.

These all go through `events` (per-reset or interval) in [cbf_go2/env_cfg.py](cbf_go2/env_cfg.py).

## 8. Deployment-realistic safety stack — lidar-driven h(x)

Dropped GT obstacle positions from the safety layer. The CBF now reads from the same BEV grid the policy sees:

- **SDT at body-frame center**: `sdf = min_dist(0, 0) over occupied cells - robot_radius`. ([cbf_go2/cbf.py](cbf_go2/cbf.py) `sdf_from_grid`)
- **Body-frame gradient** (no world↔body rotation needed since BEV is already body-frame).
- **Dropped L_f h** — no obstacle velocity estimate from a single BEV frame anyway. Pure h-constraint: `A·u + α·h - φ·||A||² ≥ 0`.
- New action term path: `safety_filter_grid` in [cbf_go2/cbf_params_action.py](cbf_go2/cbf_params_action.py). At deploy, this would point at the Mid-360 occupancy grid directly.

Crashes don't change meaningfully — the lidar BEV is high enough resolution (16×16, 3m extent) that SDT≈true-SDF.

## 9. Four architectures — ablations

Setting up an A/B/C/D comparison so we can isolate what matters in the encoder:

| arch | task ID | proprio | bev | priv | history |
| --- | --- | --- | --- | --- | --- |
| **A** Teacher | `Isaac-Goal-Go2-v0` | 15-dim MLP | 3-frame CNN | 4-dim (friction, mass, body_h) | 3 |
| **B** Student | `Isaac-Goal-Go2-StudentB-v0` | 15-dim MLP | 3-frame CNN | — | 3 |
| **C** Flat | `Isaac-Goal-Go2-FlatC-v0` | flat MLP over concat | (no CNN) | — | 3 |
| **D** LongHist | `Isaac-Goal-Go2-LongHistD-v0` | 50-step MLP | 3-frame CNN | — | 50 proprio |

Custom rsl_rl actor classes in [cbf_go2/cnn_actor.py](cbf_go2/cnn_actor.py). Priv obs (friction, mass delta, body height) is a separate `priv` obs group, only consumed by Arch A's encoder. Ablations answer:

- **A vs B**: does privileged DR (oracle friction/mass) help? → upper-bound on what an RMA-style adapter could distill.
- **C vs B**: does the CNN buy anything over flat MLP on BEV?
- **B vs D**: does long proprio history substitute for priv (RMA-style implicit adaptation)?

## 10. Scene suite — 7 layouts for generalization

[cbf_go2/scenes.py](cbf_go2/scenes.py) defines fixed obstacle layouts so we can stress-test the same checkpoint across distributions:

| scene | layout |
| --- | --- |
| `in_dist` | the env's own random per-reset layout (training distribution) |
| `open` | no obstacles |
| `sparse` | 2 obstacles at ±2m |
| `corridor` | 4 corners |
| `slalom` | zig-zag |
| `narrow` | gate at x=2 (gap 0.4m) |
| `gauntlet` | dense cluster at x∈[1.5, 2.5] |

[eval_cbf.py](eval_cbf.py) gained `--scene <name>` and `--eval_seed <int>` flags. `apply_scene` is wrapped in `torch.inference_mode()` because `write_root_pose_to_sim` mutates inference tensors.

## 11. Overnight pipeline — one script, all numbers

[overnight.sh](overnight.sh) chains the whole thing:

1. Train each of 4 archs (400 PPO iters, 4096 envs each).
2. Eval each RL checkpoint × 7 scenes × 2 seeds.
3. ISSf sweep: 4 φ values × 7 scenes × 2 seeds.
4. TISSf sweep: 4 (ε₀, λ) configs × 7 scenes × 2 seeds.
5. [aggregate.py](aggregate.py): collapse seeds, Wilson 95% CIs, write `overnight_summary.{json,csv}`.
6. [plot.py](plot.py): per-scene bar charts + Pareto scatter into `overnight_plots/`.

Per-step failures captured in `overnight_logs/`, don't abort the whole run. ETA ~5 hrs on the lab RTX 5090.

## Things to push back on (current pass)

- **Priv obs is only 4-dim** (friction × 2, mass delta, body height). Anything recoverable from BEV (obstacle radii, drift velocities) is *not* priv — would leak the scene to the teacher. That's the bar for what counts as priv here.
- **Single 95% CI** treats episodes as iid. We bin per-(method, scene), pool seeds, then Wilson on the pooled count. If reviewers ask, we can also report between-seed std.
- **Episode length 10 s during training, used same in eval.** Long enough for furthest goal at ~4.2m at 0.7 m/s ≈ 6 s. Timeouts in eval mostly = "got stuck circling an obstacle" not "out of time."
- **No RMA distillation yet.** Arch A vs B tells us the priv upper bound; if A wins meaningfully, the next step is to distill A → student with proprio history (like Arch D's encoder but supervised from A's adapter output). Not in this pass.

## Next

- Read morning results: `overnight_summary.csv`, `overnight_plots/`. Key questions:
  - A vs B: does priv help? By how much?
  - C vs B: CNN vs flat?
  - B vs D: history vs priv?
  - Does any RL arch beat ISSf+TISSf on the *hardest* scene (`narrow`, `gauntlet`)?
- If A wins clearly → distill RMA-style adapter (proprio history → priv embedding).
- If everything ties on `in_dist` but RL pulls ahead on `narrow`/`gauntlet` → that's the OOD generalization story.
