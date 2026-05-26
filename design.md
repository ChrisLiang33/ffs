# Design Doc — Learned State-Dependent Robustified CBF for Go2 Goal-Reaching

**Status:** Draft v0.1 · **Owners:** Lucas + Partner · **Stack:** Isaac Lab, Unitree Go2, Livox Mid-360

---

## 1. Problem & Core Hypothesis

We learn the **state-dependent parameters of a robustified ISSf-CBF** with RL, and use the
resulting safety filter on a Go2 for a goal-reaching task.

The robustified constraint (all parameters fixed except as noted):

```
∂h/∂x·f̂ + ∂h/∂x·ĝ·û  −  φ·‖L_ĝ h‖²₂  −  a  −  b·‖û‖₂  +  α(h(x) − c)  ≥  0
```

**Scope of this project:** learn only **φ** (robustness scaling) and **α** (class-K gain)
as functions of state/observation. `a`, `b`, `c` are held fixed. α is implemented as the
scalar gain of a *linear* class-K function, i.e. `α(h−c) = α·(h−c)`.

**What φ and α physically do here.** In command space the nominal model `(f̂, ĝ)` is the
*kinematic* map from velocity command `u = (vₓ, v_y, ω)` to pose rate — this is well known.
The uncertainty φ/a/b robustify against is the **gap between commanded and achieved
velocity**: locomotion tracking error, slip, payload, disturbances. φ buys margin against
that gap. α trades aggressiveness — large α lets `h` decay faster and approach the boundary
sooner (less conservative).

**Core hypothesis (the efficiency argument).** ISSf and TISSf fix φ (or use a hand-designed
tuning rule) sized for the *worst case*, which is conservative everywhere. A policy that sets
φ and α from the current sensing context can be tight when the situation is easy and
conservative only when it must be — winning **time-to-goal / success rate at matched
collision rate**. Safety is the constraint; efficiency is the contribution.

**Honesty about the guarantee.** Learning φ/α means we no longer get a clean formal ISSf
guarantee unless the *lowest* learnable φ is still ≥ the TISSf-safe value. Two options,
pick one explicitly:
- **(A)** Set the param bounds so the worst learnable params ⊇ a known-safe configuration →
  formal guarantee preserved, RL only ever makes it *safer or equal* on the φ axis. Limits
  the efficiency upside.
- **(B)** Accept **empirical safety within bounds**, validated by eval. The CBF structure +
  bounds are a strong inductive bias; collision is in the reward. Recommended given the
  framing is efficiency, not a new guarantee.

---

## 2. System Architecture

```
                ┌──────────────── TRAINING-TIME ─────────────────┐
  proprio ─┐
  lidar  ──┼─►[ Lidar CNN ]─┐
  priv   ──┘  [ Priv MLP  ]─┼─►[ Policy head ]─► (α, φ)   ← RL decision @ ~10 Hz
              (Arch A only) │                              params held across
                            │                             the decimation window
                            ▼
              ┌──────────────────────────────────┐
   lidar ───► │ h(x), ∇h   (cylinder fit + SDF +  │
              │            smooth-min over obs.)  │
              │ Lie derivatives  L_f h , L_g h    │   ← CBF-QP @ ~50 Hz
   goal ──►[naive u_nom]─► │ CBF-QP → u_safe  (closed-form)     │
              └──────────────────────────────────┘
                            ▼
              [ Frozen locomotion policy ] ─► joint targets
                            ▼
                [ PD @ sim rate ] ─► torques ─► physics
                            ▼
   reward = progress + goal_bonus − collision − fall − stuck − Δparam_penalty
```

**Frequency hierarchy** (realizes "RL slower than locomotion"):

| Layer | Rate | Notes |
|---|---|---|
| RL param policy | ~10 Hz | one `env.step()` per decision; params held over the window |
| CBF-QP + u_nom + locomotion command | ~50 Hz | decimation ≈ 5; QP re-solved each tick with held params |
| PD controller | sim rate (500–1000 Hz) | inside the frozen locomotion stack |

Holding params across the decimation window + a Δparam penalty is what prevents jitter.

---

## 3. The Two Architectures (a deliberate ablation)

Both deploy on **lidar + proprioception only**. The comparison *is* a contribution: does
privileged distillation buy anything over a lidar-history policy?

- **Arch B — Direct (build this first / MVP).** Single policy. A temporal stack of lidar
  (geometry + motion cues) + proprioception history → encoders → param head. The environment
  is inferred *implicitly* from history. No teacher, no distillation.

- **Arch A — Privileged + RMA-style distillation.** Phase 1: teacher trained with RL using
  proprio + lidar + privileged info (DR values, ground-truth obstacle geometry) → param head.
  Phase 2 (supervised): freeze the teacher; train an adaptation module that regresses the
  privileged latent `z` from a *history* of proprio + lidar, with an MSE loss against the
  teacher's latent. The param head is **shared** — only the env-factor encoder is swapped.
  Deploy the student.

Sequencing: Arch B is the minimum viable system. Arch A is layered on once B trains.

---

## 4. CBF / Safety Layer Details

**h(x) from lidar.** Per tick: convert the Mid-360 return into obstacle primitives
(cylinder fits), use the analytic cylinder signed-distance function, combine multiple
obstacles with a **smooth-min** (log-sum-exp) so `h` is differentiable, and apply a
smoothing/temporal filter. ∇h, `L_f h`, `L_g h` follow from the SDF gradient and the
kinematic model. *Open item:* whether the cylinder is fit to the robot footprint, the
obstacles, or both — pin this down (§10).

**The QP is nearly closed-form.** With a single (smooth-min-merged) CBF inequality and an
objective `‖u − u_nom‖²`, the CBF-QP is a **projection onto a half-space** — closed-form,
trivially batched across thousands of Isaac Lab envs, and fully differentiable. Box
constraints on `u` (velocity limits) add a clip; if the half-space and the box don't
intersect the QP is **infeasible** → fall back to a slacked/relaxed QP and penalize the slack
in the reward. Do **not** reach for an iterative QP solver unless multiple non-merged
constraints become necessary.

**Param bounds.** φ ∈ [0.01, 10.0] (already established). α ∈ [0.1, 5.0] as a starting
range, α_min > 0 required. The param head outputs through a scaled sigmoid into these
ranges — a standard bounded action space.

---

## 5. The Planner / Training-Signal Tension

A capable planner that already avoids obstacles leaves the CBF inactive → **no gradient on
α, φ** (this is the weak-signal failure mode). Design decision:

> **u_nom is a deliberately naïve go-to-goal velocity field** — a vector toward the goal,
> obstacle-*unaware*. The CBF is load-bearing for all obstacle avoidance.

This also makes the baseline comparison clean: ISSf, TISSf, Arch A, Arch B all share the
*same* naïve u_nom and the *same* frozen locomotion — only the param source differs.

---

## 6. Baselines

Both share the entire pipeline (h(x), CBF-QP, u_nom, frozen locomotion); only param
generation changes — so they are cheap once §4 exists.

- **ISSf-CBF:** φ, α fixed constants, hand-tuned for the worst-case scene.
- **TISSf-CBF:** φ a fixed hand-designed function of state (the "tunable" rule, e.g. scaling
  with `‖u‖` or obstacle distance). No learning.

Headline claim: at **matched collision rate**, the learned policies dominate on
time-to-goal / success. The right figure is a **safety–performance Pareto front** with our
methods above the baselines.

---

## 7. Domain Randomization

Split DR by which parameter it serves:

- **Uncertainty DR (serves φ):** friction, payload/mass, motor strength, actuation delay,
  terrain roughness, external push forces, lidar noise/dropout. These degrade velocity
  tracking — exactly what φ robustifies.
- **Scene DR (serves α / generalization):** obstacle count/size/placement, corridor width,
  goal placement, clutter density.

Privileged info for Arch A's teacher = the sampled DR values + ground-truth nearest-obstacle
geometry. Lidar noise/dropout DR is critical for Mid-360 sim-to-real (§10).

---

## 8. Reward & Training

- **Progress** toward goal (dense) + **goal-reach bonus**.
- **Collision** penalty (large), **fall** penalty (large).
- **Stuck** penalty — low net progress over a time window.
- **Δparam** penalty — penalize `‖params_t − params_{t−1}‖` for smoothness.
- **QP-infeasibility / slack** penalty.

**Key risk — degenerate optimum.** A heavy collision penalty alone makes "max φ always" a
local optimum: never collide, but slow / stuck. The progress + stuck signals must be strong
enough to push back. This is the same conservatism frontier ISSf/TISSf sit on — finding a
better point on it is the whole project, so reward balancing is first-class work, not
tuning at the end. Consider a difficulty curriculum (sparse → dense scenes).

Algorithm: PPO. Arch A phase 2 is supervised regression (cheap).

---

## 9. Evaluation Protocol

- **Scenes:** sparse, dense/cluttered, narrow corridor, (optionally) dynamic obstacles —
  goal-reaching, with DR applied per scene.
- **Methods:** ISSf, TISSf, Arch A, Arch B (+ ablations: with/without privileged info;
  Arch A teacher vs student gap).
- **Metrics:** success rate, collision rate, fall rate, time-to-goal, path length,
  min obstacle clearance, mean φ/α (conservativeness), QP-infeasibility rate.
- **Headline:** Pareto front (collision rate vs time-to-goal). Hypothesis: ours dominates
  in cluttered scenes where the worst-case tuning of ISSf/TISSf costs the most.

---

## 10. Open Questions / Decisions to Make

1. Guarantee framing — option (A) vs (B) in §1. *Recommend (B).*
2. h(x): cylinder fit to robot, obstacles, or both? Max obstacle count? Smooth-min temp?
3. Does the param policy see `h(x)` / `∇h` directly, or only raw sensing?
4. Static-only obstacles, or dynamic ones in eval?
5. Acceptable Mid-360 sim fidelity — its non-repetitive scan is hard to match exactly;
   plan is a raycast range-image/BEV approximation + heavy noise/dropout DR.
6. Timeline and compute budget (sets how much of Arch A vs ablations is realistic).
7. Real-robot deployment in scope, or sim-only for the paper?

**Risks to watch:** degenerate max-conservative optimum (§8); weak signal if u_nom too smart
(§5); QP infeasibility; noisy ∇h destabilizing the QP; fast-changing u_safe destabilizing
the frozen locomotion (mitigated by param + command smoothness); RMA teacher-student gap.

---

## 11. Work Split

The clean interface is the **Gym-style `env.step()` boundary**: one person owns everything
that turns *params → reward* (the "world + safety"); the other owns everything that turns
*observation → params* (the "agent"). Eval is joint.

| Track | Scope | Recommended owner | Key deliverables |
|---|---|---|---|
| **T1 — Sim & Task** | Isaac Lab Go2 env, Mid-360 lidar sim, obstacle scene generation, DR config, frozen locomotion integration, goal-reach MDP, vectorized env, mock-env stub | **Lucas** (Go2 stack + Isaac Lab expertise) | A vectorized env: `step(params) → obs, reward, done, info` |
| **T2 — Safety Layer** | h(x) pipeline (cylinder fit + SDF + smooth-min), Lie derivatives, closed-form CBF-QP + slack fallback, naïve u_nom planner, **ISSf/TISSf baselines** | **Lucas** (deepest CBF background) | `params → u_safe` module; two baseline param sources |
| **T3 — Learning** | Lidar CNN + priv/proprio encoders, teacher/student nets, **Arch B then Arch A**, RMA distillation, PPO, reward implementation & tuning | **Partner** | Trained policies for both architectures |
| **Eval** | Scene suite, metrics, experiment harness, Pareto plots, paper figures | **Joint** | Reproducible experiment + figure pipeline |

**Why this split.** T1+T2 (build the env + the CBF) is mostly deterministic engineering and
plays directly to Lucas's documented strengths (Go2 control stack, Isaac Lab, CBF theory).
T3 alone is comparable in effort because reward tuning + getting PPO to converge + two full
architectures is the highest-variance, longest-calendar work. **Swap if your partner is the
stronger systems/Isaac person** — then they take T1 and Lucas takes T2+T3-CBF-side. The
table is a recommendation, not a constraint.

### Interface contract (agree on this in Phase 0, freeze it)

```
Observation (env → policy), per env:
  proprio    : joint pos/vel, base lin/ang vel, projected gravity, last command
  lidar      : fixed-shape range-image or BEV occupancy + short history stack
  privileged : DR values + ground-truth obstacle geometry   (teacher / Arch A only)
  goal       : relative goal vector in base frame
  (optional) : h(x), ∇h

Action (policy → env), per env:
  params     : (α, φ), shape (num_envs, 2), each bounded via scaled sigmoid

Inside env.step(params):
  u_nom  ← naïve go-to-goal field
  h, L_f h, L_g h ← lidar/SDF
  u_safe ← closed-form CBF-QP(u_nom, h, L_f h, L_g h, α, φ)  [+ slack fallback]
  joint targets ← frozen locomotion(u_safe);  PD → torques → physics  (decimation ≈ 5)
  reward ← progress + goal − collision − fall − stuck − Δparam − slack
  info   ← {collision, fall, h_min, qp_infeasible, ...}   (for eval)
```

### Milestones (rough phasing — compress/expand to your calendar)

- **Phase 0 (~Wk 1) — Joint.** Freeze the interface contract. Repo skeleton.
  **Mock env** (dummy obs, accepts params, fake reward) so T3 is unblocked immediately.
- **Phase 1 (~Wk 2–3).** *Lucas:* real Isaac Lab env — Go2, Mid-360 sim, scenes, DR,
  frozen locomotion (no CBF yet, pass `u_nom` straight through).
  *Partner:* encoders + Arch B policy + PPO loop against the mock env, trivial reward.
- **Phase 2 (~Wk 4–5).** *Lucas:* h(x) + closed-form CBF-QP + naïve planner; integrate into
  the env. *Partner:* swap mock → real CBF; reward implementation & tuning; Arch B training
  on a single scene.
- **Phase 3 (~Wk 6–7).** *Lucas:* ISSf + TISSf baselines; eval scene suite + metrics.
  *Partner:* Arch A — teacher training + RMA distillation.
- **Phase 4 (~Wk 8–9) — Joint.** Full eval across scenes; ablations (A vs B, with/without
  privileged info); Pareto plots.
- **Phase 5 (~Wk 10+).** Writeup; optional real-Go2 sanity check.

**Critical dependency:** T3 must not block on T1/T2 → the Phase-0 mock env is mandatory.
Once the real env lands (end of Phase 1) the partner swaps the import and keeps going.