"""Plot per-scene bar charts of reach + crash for each method.

Reads aggregated summary JSON, writes one figure per scene + an overview grid.
"""

import argparse
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_scene(scene: str, rows: list[dict], out_path: str) -> None:
    """Bar chart: x = method, twin y for reach (left) + crash (right)."""
    # Stable ordering: RL archs first, then ISSf, then TISSf
    order = {"rl_A": 0, "rl_B": 1, "rl_C": 2, "rl_D": 3, "issf": 4, "tissf": 5}
    rows = sorted(rows, key=lambda r: (order.get(r["method"], 99), r["params"]))

    labels = [(r["method"] if not r["params"] else f"{r['method']}\n{r['params']}") for r in rows]
    reach = [r["reach"] for r in rows]
    reach_err = [
        [r["reach"] - r["reach_ci_lo"] for r in rows],
        [r["reach_ci_hi"] - r["reach"] for r in rows],
    ]
    crash = [r["crash"] for r in rows]
    crash_err = [
        [r["crash"] - r["crash_ci_lo"] for r in rows],
        [r["crash_ci_hi"] - r["crash"] for r in rows],
    ]

    x = range(len(rows))
    fig, ax1 = plt.subplots(figsize=(max(8, 0.8 * len(rows)), 5))
    width = 0.35
    b1 = ax1.bar([i - width / 2 for i in x], reach, width, yerr=reach_err,
                 color="#2c7", label="reach", capsize=3)
    ax1.set_ylabel("reach rate", color="#2c7")
    ax1.set_ylim(0, 1.05)
    ax1.tick_params(axis="y", labelcolor="#2c7")

    ax2 = ax1.twinx()
    b2 = ax2.bar([i + width / 2 for i in x], crash, width, yerr=crash_err,
                 color="#e44", label="crash", capsize=3)
    ax2.set_ylabel("crash rate", color="#e44")
    ax2.set_ylim(0, max(0.3, max(crash) * 1.3 if crash else 0.3))
    ax2.tick_params(axis="y", labelcolor="#e44")

    ax1.set_xticks(list(x))
    ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax1.set_title(f"scene: {scene}   (n_eps shown above each bar)")
    for i, r in enumerate(rows):
        ax1.text(i, 0.02, f"n={r['n_episodes']}", ha="center", fontsize=7, color="black")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_pareto(rows: list[dict], scene: str, out_path: str) -> None:
    """Scatter: x=crash, y=reach. Annotate each point with method/params."""
    fig, ax = plt.subplots(figsize=(7, 6))
    for r in rows:
        color = {"rl_A": "#1a7", "rl_B": "#27c", "rl_C": "#778", "rl_D": "#d92",
                 "issf": "#c44", "tissf": "#a4d"}.get(r["method"], "#000")
        ax.errorbar(
            r["crash"], r["reach"],
            xerr=[[r["crash"] - r["crash_ci_lo"]], [r["crash_ci_hi"] - r["crash"]]],
            yerr=[[r["reach"] - r["reach_ci_lo"]], [r["reach_ci_hi"] - r["reach"]]],
            fmt="o", color=color, capsize=3,
        )
        ann = r["method"] if not r["params"] else f"{r['method']} {r['params']}"
        ax.annotate(ann, (r["crash"], r["reach"]), fontsize=7,
                    xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("crash rate")
    ax.set_ylabel("reach rate")
    ax.set_title(f"Pareto — {scene}  (top-left = better)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--summary", default="overnight_summary.json")
    p.add_argument("--out-dir", default="overnight_plots")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.summary) as f:
        data = json.load(f)

    by_scene = defaultdict(list)
    for r in data.get("summary", []):
        by_scene[r["scene"]].append(r)

    for scene, rows in sorted(by_scene.items()):
        bars_path = os.path.join(args.out_dir, f"bars_{scene}.png")
        pareto_path = os.path.join(args.out_dir, f"pareto_{scene}.png")
        plot_scene(scene, rows, bars_path)
        plot_pareto(rows, scene, pareto_path)
        print(f"  scene={scene}  -> {bars_path}, {pareto_path}")


if __name__ == "__main__":
    main()
