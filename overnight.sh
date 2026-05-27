#!/usr/bin/env bash
# Overnight: train 4 archs, eval each across 7 scenes, sweep ISSf/TISSf across scenes,
# aggregate everything to one summary, plot bar charts.
#
# Run on the lab box:
#   cd ~/Desktop/ffs && ./overnight.sh
#
# Knobs (env vars):
#   ISAACLAB     (default ~/IsaacLab)
#   ITERS        max PPO iters per arch       (default 400)
#   NUM_ENVS     training env count           (default 4096)
#   EVAL_ENVS    eval env count               (default 256)
#   EVAL_STEPS   outer steps per eval         (default 3000, ~6k episodes)
#   EVAL_SEEDS   space-separated seeds        (default "0 1 2")

set -u
cd "$(dirname "$0")"

ISAACLAB="${ISAACLAB:-$HOME/IsaacLab}"
RUN="$ISAACLAB/isaaclab.sh -p"
ITERS="${ITERS:-400}"
NUM_ENVS="${NUM_ENVS:-4096}"
EVAL_ENVS="${EVAL_ENVS:-256}"
EVAL_STEPS="${EVAL_STEPS:-1500}"
EVAL_SEEDS="${EVAL_SEEDS:-0 1}"

LOGS="overnight_logs"
RESULTS="overnight_results"
PLOTS="overnight_plots"
mkdir -p "$LOGS" "$RESULTS" "$PLOTS"

# arch_name : task_id
ARCHS=(
  "A:Isaac-Goal-Go2-v0"
  "B:Isaac-Goal-Go2-StudentB-v0"
  "C:Isaac-Goal-Go2-FlatC-v0"
  "D:Isaac-Goal-Go2-LongHistD-v0"
)
SCENES=(in_dist open sparse corridor slalom narrow gauntlet)

ts() { date '+%H:%M:%S'; }
say() { echo; echo "[$(ts)] === $* ==="; }

# Don't abort on a single failed run — capture errors per step.
run_step() {
  local logfile="$1"; shift
  echo "[$(ts)] CMD: $*" >> "$logfile"
  if ! "$@" >> "$logfile" 2>&1; then
    echo "[$(ts)] FAILED: $*" >> "$logfile"
    echo "[$(ts)] FAILED (continuing): $*"
    return 1
  fi
}

# ---------- 1. Train each arch ----------
for entry in "${ARCHS[@]}"; do
  name="${entry%%:*}"; task="${entry#*:}"
  say "Train Arch $name ($task) — $ITERS iters"
  run_step "$LOGS/train_${name}.log" \
    $RUN train.py --task "$task" --num_envs "$NUM_ENVS" --headless \
      --max_iterations "$ITERS" || true
done

# ---------- 2. Eval RL on each scene, multi-seed ----------
for entry in "${ARCHS[@]}"; do
  name="${entry%%:*}"; task="${entry#*:}"
  for scene in "${SCENES[@]}"; do
    for seed in $EVAL_SEEDS; do
      say "Eval RL Arch $name | scene=$scene | seed=$seed"
      run_step "$LOGS/eval_${name}_${scene}_s${seed}.log" \
        $RUN eval_cbf.py --task "$task" --mode rl \
          --scene "$scene" --eval_seed "$seed" \
          --num_envs "$EVAL_ENVS" --steps "$EVAL_STEPS" \
          --save "$RESULTS/rl_arch${name}_${scene}_s${seed}.json" || true
    done
  done
done

# ---------- 3. ISSf sweep across phi and scenes ----------
for phi in 0.1 0.2 0.5 1.0; do
  for scene in "${SCENES[@]}"; do
    for seed in $EVAL_SEEDS; do
      say "Eval ISSf phi=$phi | scene=$scene | seed=$seed"
      run_step "$LOGS/eval_issf_p${phi}_${scene}_s${seed}.log" \
        $RUN eval_cbf.py --mode issf --phi "$phi" \
          --scene "$scene" --eval_seed "$seed" \
          --num_envs "$EVAL_ENVS" --steps "$EVAL_STEPS" \
          --save "$RESULTS/issf_phi${phi}_${scene}_s${seed}.json" || true
    done
  done
done

# ---------- 4. TISSf sweep ----------
TISSF_CFGS=("1.0 0.5" "2.0 0.5" "1.0 1.5" "2.0 1.5")
for cfg in "${TISSF_CFGS[@]}"; do
  eps="${cfg%% *}"; lam="${cfg##* }"
  for scene in "${SCENES[@]}"; do
    for seed in $EVAL_SEEDS; do
      say "Eval TISSf eps=$eps lam=$lam | scene=$scene | seed=$seed"
      run_step "$LOGS/eval_tissf_e${eps}_l${lam}_${scene}_s${seed}.log" \
        $RUN eval_cbf.py --mode tissf --epsilon_0 "$eps" --lam "$lam" \
          --scene "$scene" --eval_seed "$seed" \
          --num_envs "$EVAL_ENVS" --steps "$EVAL_STEPS" \
          --save "$RESULTS/tissf_e${eps}_l${lam}_${scene}_s${seed}.json" || true
    done
  done
done

# ---------- 5. Aggregate + plot ----------
say "Aggregate"
python aggregate.py --results-dir "$RESULTS" --out "overnight_summary.json" --csv "overnight_summary.csv" || true

say "Plot"
python plot.py --summary "overnight_summary.json" --out-dir "$PLOTS" || true

say "DONE — see overnight_summary.{json,csv} and $PLOTS/"
