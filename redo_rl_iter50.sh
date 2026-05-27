#!/usr/bin/env bash
# Re-eval each RL arch using its model_50.pt checkpoint instead of the final one.
# Hypothesis: all 4 archs peaked around iter 50 (training logs show goal_reached
# 83-85% / crash 3-4%) and degraded as PPO traded safety for progress. Eval with
# the iter-50 checkpoint to confirm the peak is real on held-out scenes.
#
# Writes results to overnight_results/rl_arch{A,B,C,D}_iter50_{scene}_s{seed}.json
# so they don't collide with the iter-399 numbers already in the summary.
#
# Run on the lab box:
#   cd ~/Desktop/ffs && ./redo_rl_iter50.sh

set -u
cd "$(dirname "$0")"

ISAACLAB="${ISAACLAB:-$HOME/IsaacLab}"
RUN="$ISAACLAB/isaaclab.sh -p"
EVAL_ENVS="${EVAL_ENVS:-256}"
EVAL_STEPS="${EVAL_STEPS:-1500}"
EVAL_SEEDS="${EVAL_SEEDS:-0 1}"

LOGS="overnight_logs"
RESULTS="overnight_results"
mkdir -p "$LOGS" "$RESULTS"

SCENES=(in_dist open sparse corridor slalom narrow gauntlet)

# arch_name : task_id : checkpoint_run_dir
ARCHS=(
  "A:Isaac-Goal-Go2-v0:logs/rsl_rl/cbf_goal_go2_archA/2026-05-27_04-22-45/model_50.pt"
  "B:Isaac-Goal-Go2-StudentB-v0:logs/rsl_rl/cbf_goal_go2_archB/2026-05-27_04-34-54/model_50.pt"
  "C:Isaac-Goal-Go2-FlatC-v0:logs/rsl_rl/cbf_goal_go2_archC/2026-05-27_04-46-50/model_50.pt"
  "D:Isaac-Goal-Go2-LongHistD-v0:logs/rsl_rl/cbf_goal_go2_archD/2026-05-27_04-58-29/model_50.pt"
)

ts() { date '+%H:%M:%S'; }
say() { echo; echo "[$(ts)] === $* ==="; }

run_step() {
  local logfile="$1"; shift
  echo "[$(ts)] CMD: $*" >> "$logfile"
  if ! "$@" >> "$logfile" 2>&1; then
    echo "[$(ts)] FAILED: $*" >> "$logfile"
    echo "[$(ts)] FAILED (continuing): $*"
    return 1
  fi
}

for entry in "${ARCHS[@]}"; do
  IFS=':' read -r name task ckpt <<< "$entry"
  if [[ ! -f "$ckpt" ]]; then
    echo "[$(ts)] SKIP Arch $name — checkpoint not found: $ckpt"
    continue
  fi
  for scene in "${SCENES[@]}"; do
    for seed in $EVAL_SEEDS; do
      say "Eval Arch $name iter=50 | scene=$scene | seed=$seed"
      run_step "$LOGS/eval_${name}_iter50_${scene}_s${seed}.log" \
        $RUN eval_cbf.py --task "$task" --mode rl \
          --checkpoint "$ckpt" \
          --scene "$scene" --eval_seed "$seed" \
          --num_envs "$EVAL_ENVS" --steps "$EVAL_STEPS" \
          --save "$RESULTS/rl_arch${name}_iter50_${scene}_s${seed}.json" || true
    done
  done
done

say "DONE — re-aggregate with: python aggregate.py --results-dir $RESULTS --out overnight_summary.json --csv overnight_summary.csv"
say "(note: aggregate currently labels these as rl_A/B/C/D — same as iter-399 runs — so they'll get pooled. If you want them separate, edit aggregate.load_one to tag by the iter50 filename token.)"
