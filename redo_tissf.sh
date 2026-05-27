#!/usr/bin/env bash
# Re-run only the TISSf sweep (the overnight run failed all TISSf cells due to a
# 2-vs-3 tuple unpack bug in tissf_action; mdp.py is now fixed). Then re-aggregate
# and re-plot.
#
# Run on the lab box:
#   cd ~/Desktop/ffs && ./redo_tissf.sh

set -u
cd "$(dirname "$0")"

ISAACLAB="${ISAACLAB:-$HOME/IsaacLab}"
RUN="$ISAACLAB/isaaclab.sh -p"
EVAL_ENVS="${EVAL_ENVS:-256}"
EVAL_STEPS="${EVAL_STEPS:-1500}"
EVAL_SEEDS="${EVAL_SEEDS:-0 1}"

LOGS="overnight_logs"
RESULTS="overnight_results"
PLOTS="overnight_plots"
mkdir -p "$LOGS" "$RESULTS" "$PLOTS"

SCENES=(in_dist open sparse corridor slalom narrow gauntlet)
TISSF_CFGS=("1.0 0.5" "2.0 0.5" "1.0 1.5" "2.0 1.5")

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

# Clear stale (failed) TISSf JSONs so aggregate doesn't see them.
rm -f "$RESULTS"/tissf_*.json
# And clear old TISSf logs so we don't conflate with this re-run.
rm -f "$LOGS"/eval_tissf_*.log

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

say "Aggregate"
python aggregate.py --results-dir "$RESULTS" --out "overnight_summary.json" --csv "overnight_summary.csv" || true

say "Plot"
python plot.py --summary "overnight_summary.json" --out-dir "$PLOTS" || true

say "DONE — see overnight_summary.{json,csv} and $PLOTS/"
