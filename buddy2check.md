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

## 12. Overnight ran — RL lost everywhere

Headline (per-scene reach% / crash%, best of each method family):

| scene    | rl_A      | rl_B      | rl_C      | rl_D      | issf*         | tissf*    |
| -------- | --------- | --------- | --------- | --------- | ------------- | --------- |
| in_dist  | 88.0/11.7 | 88.0/11.7 | 88.1/11.7 | 90.2/ 9.4 | **93.8/ 5.5** | 92.8/ 6.5 |
| open     | 95.7/ 4.3 | 95.6/ 4.4 | 96.2/ 3.8 | 95.9/ 4.1 | 96.5/ 3.5     | 96.5/ 3.5 |
| sparse   | 94.0/ 5.8 | 94.2/ 5.7 | 94.0/ 5.8 | 94.6/ 5.3 | 95.0/ 4.6     | 95.7/ 3.9 |
| corridor | 88.9/10.6 | 88.9/10.4 | 88.0/11.5 | 90.2/ 9.3 | 93.9/ 4.6     | 94.0/ 4.6 |
| slalom   | 90.4/ 9.0 | 90.0/ 9.4 | 89.1/10.4 | 91.6/ 7.8 | 93.9/ 4.7     | 93.8/ 4.9 |
| narrow   | 94.0/ 5.7 | 93.7/ 5.8 | 94.0/ 5.6 | 94.8/ 4.8 | 95.1/ 4.4     | 95.1/ 4.3 |
| gauntlet | 92.0/ 7.4 | 92.5/ 7.0 | 92.3/ 7.1 | 93.3/ 6.0 | 94.3/ 4.9     | 94.8/ 4.3 |

**ISSf wins every scene on both axes.** And RL loses even on `in_dist` (the training distribution) — so this isn't a generalization gap.

Among RL: D > C ≈ B ≈ A. LongHist is best by ~1–2 pts. **Priv (A) didn't help over no-priv (B)**, within noise.

This is a regression vs the last clean win (Section 6, no full DR): RL had 93.3% / 0.5% there, ISSf 91.2% / 0.7%. Adding full DR + 4 archs + lidar-driven CBF + 400 PPO iters broke RL.

## 13. Failure-mode diagnosis: action degeneracy

Dumped the per-step (α, φ) policy actions on `corridor` for the iter-399 checkpoints:

| arch | α median       | α p95 | φ median       | φ p95 |
| ---- | -------------- | ----- | -------------- | ----- |
| A    | **5.00 (max)** | 5.00  | **0.01 (min)** | 0.01  |
| B    | 5.00 (max)     | 5.00  | 0.01 (min)     | 4.17  |
| D    | 5.00 (max)     | 5.00  | 0.01 (min)     | 10.00 |

ISSf anchor (which wins) is α=2.0, φ=0.5.

**All 3 archs converged to α pegged at max and φ pegged at min.** Translation:

- α=5 → most aggressive CBF (allows h to decay 2.5× faster than ISSf — get very close before reacting)
- φ=0.01 → **input-to-state robustness term essentially off** — the ISSf becomes vanilla CBF with no margin

So PPO learned "ignore robustness, push hard, sometimes crash" — the corner of the action space, ~10% crash rate. ISSf at (2.0, 0.5) sits in a much more conservative band and crashes ~5%.

Things tried that didn't fix it:

- **Heavier crash_penalty (-50 → -200)**: marginal improvement (rl_D crash 9.2 → 7.4% on corridor) but α still pegged at 5, φ still mostly pegged at 0.01. Arch A's distribution barely moved.

Things considered but **not** taken:

- **Tighter action ranges** (α∈[0.5,3], φ∈[0.1,2]): would cap the worst case but artificially constrains what RL can learn. Rejected — we want learned bounds, not hand-tuned ones.
- **Iter-50 re-eval** (training-time peak): training logs showed iter-50 goal/crash ~ 84%/4% vs iter-399 ~ 83%/5.5%, a 1-pt difference on noisy training metrics. Not large enough to explain the 7-pt eval gap. Even if iter-50 was better, the action distribution likely has the same degeneracy. Skipped.

## 14. Encoder check — CNN is alive

Ran 3 archs × {open, corridor} action dumps to test whether actions depend on BEV content.

| arch | scene    | α mean | α std | φ mean | φ std |
| ---- | -------- | ------ | ----- | ------ | ----- |
| A    | open     | 4.91   | 0.66  | 0.21   | 1.37  |
| A    | corridor | 4.59   | 1.28  | 0.52   | 2.05  |
| B    | open     | 4.94   | 0.50  | 0.27   | 1.55  |
| B    | corridor | 4.53   | 1.33  | 0.50   | 1.96  |
| D    | open     | 4.59   | 1.31  | 0.73   | 2.48  |
| D    | corridor | 4.37   | 1.52  | 1.02   | 2.82  |

Going `open → corridor`: α drops, φ rises, std goes up. **Directions are correct** — more obstacles → less aggressive, more robust, more action variance. CNN is reading the BEV.

But the *magnitudes* are tiny. Median α stays at 5.00 and median φ stays at 0.01 on both scenes. The mean shifts come from occasional excursions, not steady-state. **The policy uses perception as a panic brake, not as steady-state guidance.**

## 15. Discount bump (γ=0.99 → 0.995) — didn't help

Hypothesis: late-episode rewards were too discounted; crashes near the end of an episode were nearly free.

Retrained A and D with γ=0.995 + the −200 crash penalty. Corridor crash rates:

| | overnight (γ=0.99, crash=−50) | γ=0.99, crash=−200 | γ=0.995, crash=−200 |
| --- | --- | --- | --- |
| Arch A corridor crash | 10.6% | 11.3% | 10.5% |
| Arch D corridor crash | 9.3% | 7.4% | 8.0% |

Action distributions essentially unchanged. α median still 5.00. **Discount wasn't the bottleneck.**

## 16. Bisect — was the corner-attractor always there?

Used `dump_action.py` against a Section-6-era winning checkpoint (`logs/rsl_rl/cbf_goal_go2/2026-05-26_23-09-33/model_199.pt`) running on the old commit (5e832ba — GT-obstacle obs, GT-obstacle CBF, no DR).

|           | Section-6 WIN era | Today (iter-399 broken) |
| --------- | ----------------- | ----------------------- |
| α mean    | 3.68              | 4.59                    |
| α p25     | **2.75**          | 5.00                    |
| α p50     | 4.24              | 5.00                    |
| α p95     | 5.00              | 5.00                    |
| φ p50     | 0.01 (min)        | 0.01 (min)              |
| φ p95     | 2.99              | 4.86                    |

**The corner attractor was always there.** Even the winning policy ran α near the cap and pegged φ at the floor. The difference vs today is modest extra spread on α (p25 2.75 vs 0.10).

So the policy isn't fundamentally broken now — it's behaving similarly to when it won.

## 17. Real diagnosis — the pipeline got harder

Cross-checking ISSf's own performance across eras shows the pipeline got harder, not the policy worse:

| era | ISSf reach | **ISSf crash** |
| --- | --- | --- |
| Section 6 (GT obs + GT-CBF, no DR) | 91.2% | **0.7%** |
| Section 12 (BEV obs + BEV-CBF, full DR) | ~94% | **~5%** |

**ISSf's crash rate jumped 7×.** Same fixed policy, way more crashes — because the deployment-realistic stack (BEV grid CBF, lidar noise, DR pushes, mass/friction variation) is genuinely harder for any fixed policy. The CBF projection is fuzzier (16×16 grid vs GT positions), no L_f h to predict obstacle motion.

The win in Section 6 wasn't "RL learned to be smart." It was: aggressive policy + precise CBF = works. The precise CBF caught the rare near-misses. In the new regime: aggressive policy + fuzzy CBF = crashes.

**ISSf survives the new pipeline because it's robust by construction** (fixed conservative point, no dependence on observation precision). RL's "aggressive + panic-brake" pattern *needs* a precise brake to work.

## 18. Things rejected (and why)

- **Per-step h-based safety reward** (`-k·relu(threshold - h)` each step). Would push PPO away from the corner by penalizing near-obstacle steady states. *Rejected*: creates an "invisible force field" that duplicates what the CBF is supposed to do. The design is CBF→safety, CNN→perception, RL→modulation. Reward shouldn't bake in safety; that's what the CBF guarantees.
- **Higher-resolution BEV** (32×32 instead of 16×16). Would let the existing aggressive policy do the same Section-6 trick. *Rejected*: backs out of the deployment-realistic story.

## 19. ISSf-anchor init — current test

Initialize the policy's last-layer bias so the initial mean output is α=2.0, φ=0.5 (the ISSf anchor) instead of the default α=2.55, φ=5.0. The patch:

```python
# in cbf_go2/cnn_actor.py
_ANCHOR_NORM = (-0.224, -0.902)   # alpha=2.0, phi=0.5 in normalized space

def _init_head_at_anchor(encoder):
    final = encoder.head[-1]
    with torch.no_grad():
        final.weight.zero_()
        final.bias[0] = _ANCHOR_NORM[0]
        final.bias[1] = _ANCHOR_NORM[1]
```

Applied to TeacherActor, StudentActor, LongHistActor. Initial actor output is exactly anchor; weights grow from zero.

The question this test answers: **is the corner the only local optimum, or just where 0-init lands?**

- If PPO stays near anchor → corner wasn't fundamental, init was holding us back. Expect RL to beat ISSf because it can modulate.
- If PPO drifts back to the corner → the corner is genuinely better in PPO's reward landscape. Need a deeper fix (longer training, Beta distribution policy, or revisit reward design entirely).

Retraining A and D, 200 iters each. Crash penalty stays at -200 and γ at 0.995 (changes carried over from sections 13 and 15 — they didn't help but they don't hurt; cleaner to vary one thing at a time).

## 20. Extended ISSf φ sweep — φ=1.0 is the universal optimum

Swept φ ∈ {1.5, 2.0, 3.0} on top of the original {0.1, 0.2, 0.5, 1.0}. Time-to-reach added to the analysis (buddy's pushback: "high φ might take detours / fail to fit through narrow"):

| scene    | best-φ | reach  | crash | timeout | t_reach (s) |
| -------- | ------ | ------ | ----- | ------- | ----------- |
| in_dist  | 1.0    | 93.8%  | 5.5%  | 0.7%    | 14.00       |
| open     | any    | 96.5%  | 3.5%  | 0%      | 15.31       |
| sparse   | 1.0    | 95.0%  | 4.6%  | 0.4%    | 14.18       |
| corridor | 1.0    | 93.9%  | 4.6%  | 1.6%    | 13.71       |
| slalom   | 1.0    | 93.9%  | 4.7%  | 1.5%    | 13.11       |
| narrow   | 1.0    | 95.1%  | 4.4%  | 0.5%    | 14.01       |
| gauntlet | 1.0    | 94.3%  | 4.9%  | 0.8%    | 13.12       |

**φ=1.0 dominates every scene with obstacles, monotonically.** Beyond 1.0:

- corridor reach drops 93.9 → 84.0% as φ goes 1.0 → 3.0
- corridor timeouts jump 1.6 → 5.3% (buddy was right about "too conservative to fit")
- slalom timeouts 1.5 → 4.2%

Also interesting: higher φ is *faster*, not slower. Smooth large-radius avoidance beats panic-brake U-turns. t_reach drops 14.11 → 13.71 (corridor) and 13.95 → 13.11 (slalom) going φ=0.1 → 1.0.

**Verdict: there is no adaptation upside in our scene set.** A single fixed (α=2, φ=1.0) dominates everywhere. RL's job becomes "find the right fixed point," not "adapt per scene."

## 21. CBF intervention rate — confirms α=5 kills the constraint

Added `cbf intervention rate` (fraction of steps where `safety_filter_grid` actually projects `u_nom → u_safe`, defined as `||u_safe - u_nom||_∞ > 1e-4`) to `dump_action.py` and `eval_cbf.py`. Buddy's hypothesis: at α=5 the CBF is effectively off, so φ never gets a learning signal.

Confirmed cleanly on corridor:

| | α | φ | **CBF acts** | crash |
| --- | --- | --- | --- | --- |
| RL iter-399 (broken, α pegged 5) | 4.69 | 0.23 | **11.7%** | 10.5% |
| ISSf (α=2, φ=0.5) | 2.0 | 0.5 | **43.1%** | 7.1% |
| ISSf (α=2, φ=1.0) — best | 2.0 | 1.0 | **54.3%** | 6.4% |

**RL's CBF is intervening on 11.7% of steps — basically off.** ISSf-1.0 intervenes 54.3%, almost 5×. The mechanism: high α makes the constraint `A·u ≥ φ·||A||² − α·h` trivially satisfied when `h > 0` (safe region), so the CBF only fires in a tiny boundary layer. Inside the safe region, the constraint is vacuous → φ has no effect on actions → no gradient signal → φ drifts to the Gaussian tail (0.01).

The chain: progress reward favors fast motion → high α → CBF off → φ stops mattering → corner attractor at (α=5, φ=0.01).

## 22. Progress weight 1.0 → 0.2 + anchor (α=2, φ=0.5)

Stack: anchor (α=2, φ=0.5) + progress=0.2 + crash=−200 + γ=0.995. Retrained A and D, 200 iters.

| | α p50 | φ p50 | CBF acts | crash (corridor) |
| --- | --- | --- | --- | --- |
| Arch A | 5.00 | 0.01 | 22.8% | 11.3% |
| Arch D | **0.10** | 0.01 | **47.7%** | 9.0% |

**Arch D escaped the high-α corner — but landed at the OPPOSITE corner (α=0.1, CBF too restrictive).** Bimodal: p25=0.10, p75=5.00. φ still pegged at 0.01 because α=0.10 still leaves the constraint dominated by −α·h, and φ·||A||² is tiny. So the policy is *using* the CBF much more (47.7% vs 11.7%) but not at the right operating point.

**Arch A stayed at the high-α corner.** Hypothesis: priv obs (friction/mass/body_h) gives extra signal that the policy uses to "trust the locomotion" and stay aggressive. Arch D without priv has to be more cautious by default.

Either way, neither arch matched ISSf-1.0's 6.4% crash on corridor.

## 23. Anchor bumped to (α=2, φ=1.0) — both archs drifted back to corner

Stack: anchor (α=2, φ=**1.0**) + progress=0.2 + crash=−200 + γ=0.995. Retrained A and D.

| | α p50 | φ p50 | CBF acts | crash (corridor) |
| --- | --- | --- | --- | --- |
| Arch A | 5.00 | 0.01 | 18.1% | 8.0% |
| Arch D | 5.00 | 0.01 | 28.1% | 7.6% |

Both archs drifted α back to 5.0. φ stayed pegged at 0.01. Crashes improved a bit (8/7.6% vs ISSf-1.0's 6.4%), but the failure mode is identical to where we started — corner attractor wins again.

## 24. Real culprit — init_std=1.0 scatters the policy across the full action range every step

The rsl_rl Gaussian distribution defaults to `init_std=1.0` in normalized action space [-1, 1]. With std=1.0, the policy's exploration noise covers the entire action range every step:

- Anchor at α=2 (norm=−0.224). Exploration ±1σ → norm ∈ [−1.224, 0.776] → after clamping, α ∈ [0.1, 4.4] every step
- Anchor at φ=1.0 (norm=−0.802). Exploration ±1σ → norm ∈ [−1.802, 0.198] → φ ∈ [0.01, 6.0]

So the anchor init is *immediately undone* by exploration noise. The very first batch of PPO samples lands all over the action space — including corners. PPO computes the gradient from this noisy batch, finds the corner has higher reward, and pulls the policy there. The anchor never gets a chance to hold.

## 25. Current test — init_std 1.0 → 0.1

One-line change in `cbf_go2/rsl_rl_cfg.py`: `GaussianDistributionCfg(init_std=0.1, std_type="scalar")`.

With std=0.1:

- Anchor at α=2.0: ±1σ → α ∈ [1.65, 2.35] — local exploration only
- Anchor at φ=1.0: ±1σ → φ ∈ [0.6, 1.4] — local exploration only

PPO should now refine the anchor locally instead of being pulled to corners. If (α=2, φ=1.0) is genuinely the global optimum (sweep says yes), PPO should converge there. Retraining A and D.

Expected outcomes:

- **α stays ~2, φ stays ~1.0**: anchor + local exploration enough. RL matches ISSf-1.0 fixed performance. The remaining question becomes: can it learn per-scene modulation that *beats* fixed ISSf-1.0?
- **Policy drifts even with tiny exploration**: gradient itself favors corners from anywhere in the space. Then ISSf is the practical operating point and the "learned (α, φ)" framing is fundamentally limited under this reward + DR + lidar-CBF stack.
