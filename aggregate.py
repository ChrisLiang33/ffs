"""Walk all eval JSON files, collapse multi-seed runs, write summary JSON + CSV.

Per-run records: {method, arch_or_params, scene, seed, reached, crashed, timeout, total,
                  mean_reach_steps}
Per-method-scene summary: {method, scene, n_runs, reach_mean, reach_ci_lo, reach_ci_hi,
                           crash_mean, crash_ci_lo, crash_ci_hi, timeout_mean, time_to_reach_mean}
"""

import argparse
import csv
import glob
import json
import math
import os
from collections import defaultdict


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def load_one(path: str) -> dict | None:
    try:
        with open(path) as f:
            d = json.load(f)
    except Exception as e:
        print(f"[warn] failed to parse {path}: {e}")
        return None
    args = d.get("args", {})
    s = d.get("summary", {})
    mode = args.get("mode", "?")
    scene = args.get("scene", "?")
    seed = args.get("eval_seed", None)
    task = args.get("task", "")

    if mode == "rl":
        # Tag with the arch letter from the task ID.
        if "LongHistD" in task:
            label = "rl_D"
        elif "StudentB" in task:
            label = "rl_B"
        elif "FlatC" in task:
            label = "rl_C"
        else:
            label = "rl_A"
        # Iter-tagged checkpoints (e.g. rl_archA_iter50_*.json) need their own
        # row so they don't collapse with the final-checkpoint runs.
        if "_iter50_" in os.path.basename(path):
            label = f"{label}_i50"
        params = ""
    elif mode == "issf":
        label = "issf"
        params = f"phi={args.get('phi')}"
    elif mode == "tissf":
        label = "tissf"
        params = f"eps0={args.get('epsilon_0')},lam={args.get('lam')}"
    else:
        label = mode
        params = ""

    return {
        "method": label,
        "params": params,
        "scene": scene,
        "seed": seed,
        "total": s.get("total", 0),
        "reached": s.get("reached", 0),
        "crashed": s.get("crashed", 0),
        "timeout": s.get("timeout", 0),
        "mean_reach_steps": s.get("mean_reach_steps"),
        "source": os.path.basename(path),
    }


def aggregate(results_dir: str) -> dict:
    runs = []
    for p in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        rec = load_one(p)
        if rec is not None:
            runs.append(rec)

    # Collapse over seeds: per (method, params, scene), sum totals/reached/crashed/timeout.
    by_key = defaultdict(lambda: {
        "method": "", "params": "", "scene": "",
        "n_seeds": 0, "total": 0, "reached": 0, "crashed": 0, "timeout": 0,
        "mean_reach_sum": 0.0, "mean_reach_n": 0,
    })
    for r in runs:
        k = (r["method"], r["params"], r["scene"])
        agg = by_key[k]
        agg["method"] = r["method"]
        agg["params"] = r["params"]
        agg["scene"] = r["scene"]
        agg["n_seeds"] += 1
        agg["total"] += r["total"]
        agg["reached"] += r["reached"]
        agg["crashed"] += r["crashed"]
        agg["timeout"] += r["timeout"]
        if r["mean_reach_steps"] is not None:
            agg["mean_reach_sum"] += r["mean_reach_steps"] * r["reached"]
            agg["mean_reach_n"] += r["reached"]

    summary = []
    for agg in by_key.values():
        n = agg["total"]
        rc, ce, to = agg["reached"], agg["crashed"], agg["timeout"]
        rlo, rhi = wilson_ci(rc, n)
        clo, chi = wilson_ci(ce, n)
        tlo, thi = wilson_ci(to, n)
        t2r = (agg["mean_reach_sum"] / agg["mean_reach_n"]) if agg["mean_reach_n"] else None
        summary.append({
            "method": agg["method"],
            "params": agg["params"],
            "scene": agg["scene"],
            "n_seeds": agg["n_seeds"],
            "n_episodes": n,
            "reach": rc / max(n, 1),
            "reach_ci_lo": rlo,
            "reach_ci_hi": rhi,
            "crash": ce / max(n, 1),
            "crash_ci_lo": clo,
            "crash_ci_hi": chi,
            "timeout": to / max(n, 1),
            "timeout_ci_lo": tlo,
            "timeout_ci_hi": thi,
            "mean_reach_steps": t2r,
        })

    summary.sort(key=lambda r: (r["scene"], r["method"], r["params"]))
    return {"runs": runs, "summary": summary}


def write_csv(summary: list[dict], path: str) -> None:
    if not summary:
        return
    fields = list(summary[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(summary)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="overnight_results")
    p.add_argument("--out", default="overnight_summary.json")
    p.add_argument("--csv", default="overnight_summary.csv")
    args = p.parse_args()

    data = aggregate(args.results_dir)
    with open(args.out, "w") as f:
        json.dump(data, f, indent=2, default=str)
    write_csv(data["summary"], args.csv)
    print(f"aggregated {len(data['runs'])} runs -> {len(data['summary'])} (method, scene) rows")
    print(f"  json: {args.out}")
    print(f"  csv:  {args.csv}")


if __name__ == "__main__":
    main()
