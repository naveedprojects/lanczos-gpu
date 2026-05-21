#!/usr/bin/env python3
"""Plot Week-1 IMGP-scaling figures from data/imgp_scaling_sweep.json."""
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


METHOD_STYLE = {
    "imgp-ours":  {"color": "#1f77b4", "marker": "o", "label": "ours (Krämer SLQ)"},
    "imgp-naive": {"color": "#d62728", "marker": "s",
                   "label": "naive Lanczos (paper's failing row)"},
    "imgp-full":  {"color": "#2ca02c", "marker": "^",
                   "label": "dense slogdet (SS-IMGP-full)"},
}


def load(path):
    with open(path) as f:
        return json.load(f)


def plot_wallclock_vs_n(data, ax):
    methods = ["imgp-full", "imgp-naive", "imgp-ours"]
    for method in methods:
        ns, ts, diverged_ns = [], [], []
        for run in data["runs"]:
            r = run.get(method)
            if r is None:
                continue
            if not r["iter_wall_seconds"]:
                # Diverged before completing any iteration.
                if r.get("diverged"):
                    diverged_ns.append(run["n"])
                continue
            ns.append(run["n"])
            ts.append(r["iter_wall_seconds"][-1])
        if ns:
            ax.loglog(ns, ts, **METHOD_STYLE[method], linewidth=2, markersize=8)
        # Mark divergence points with an "X" at the bottom of the plot.
        for dn in diverged_ns:
            ax.scatter([dn], [0.5], marker="x", s=120, linewidths=3,
                       color=METHOD_STYLE[method]["color"], zorder=5)
            ax.annotate(f"DIVERGED",
                        xy=(dn, 0.5),
                        xytext=(dn, 0.15),
                        ha="center", fontsize=10,
                        color=METHOD_STYLE[method]["color"],
                        fontweight="bold",
                        arrowprops=dict(arrowstyle="->", lw=1.5,
                                        color=METHOD_STYLE[method]["color"]))

    # Mark dense baseline's truncation (OOM) past its last data point.
    full_ns = [run["n"] for run in data["runs"]
               if run.get("imgp-full") is not None
               and run["imgp-full"]["iter_wall_seconds"]]
    if full_ns:
        last_n = max(full_ns)
        # Find the next N tested
        all_ns = sorted(set(run["n"] for run in data["runs"]))
        next_n = next((n for n in all_ns if n > last_n), None)
        if next_n is not None:
            mid = (last_n * next_n) ** 0.5
            ax.annotate(f"dense OOM\n(N > {last_n:,})",
                        xy=(mid, 50), fontsize=10,
                        color=METHOD_STYLE["imgp-full"]["color"],
                        ha="center", fontweight="bold")

    ax.set_xlabel("N (number of nodes)", fontsize=12)
    ax.set_ylabel("Wall-clock for full training run [s]", fontsize=12)
    ax.set_title("Training wall-clock vs problem size", fontsize=13)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=10, loc="upper left")
    ax.set_ylim(bottom=0.1)


def plot_loss_at_largest_n(data, ax):
    """Loss trajectory at the largest N where IMGP-full ran."""
    # Pick the largest N where at least 2 methods (ours + something) have
    # completed iterations — gives the most informative trajectory panel.
    target_run = None
    for run in reversed(data["runs"]):
        n_methods = sum(
            1 for m in ("imgp-full", "imgp-naive", "imgp-ours")
            if run.get(m) is not None
            and run[m]["iter_wall_seconds"]
        )
        if n_methods >= 2:
            target_run = run
            break
    if target_run is None:
        target_run = data["runs"][-1]

    n = target_run["n"]
    for method in ["imgp-full", "imgp-naive", "imgp-ours"]:
        r = target_run.get(method)
        if r is None:
            continue
        losses = r["iter_losses"]
        ax.plot(range(1, len(losses) + 1), losses,
                **METHOD_STYLE[method], linewidth=2, markersize=5)

    ax.set_xlabel("Adam iteration", fontsize=12)
    ax.set_ylabel("Per-data-point NLL", fontsize=12)
    ax.set_title(f"Training loss trajectory at N = {n}", fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)


def plot_hyperparameter_trajectories(data, ax_l, ax_k):
    # Pick the largest N where at least 2 methods (ours + something) have
    # completed iterations — gives the most informative trajectory panel.
    target_run = None
    for run in reversed(data["runs"]):
        n_methods = sum(
            1 for m in ("imgp-full", "imgp-naive", "imgp-ours")
            if run.get(m) is not None
            and run[m]["iter_wall_seconds"]
        )
        if n_methods >= 2:
            target_run = run
            break
    if target_run is None:
        target_run = data["runs"][-1]

    n = target_run["n"]
    for method in ["imgp-full", "imgp-naive", "imgp-ours"]:
        r = target_run.get(method)
        if r is None:
            continue
        iters = range(1, len(r["iter_lengthscales"]) + 1)
        ax_l.plot(iters, r["iter_lengthscales"],
                  **METHOD_STYLE[method], linewidth=2, markersize=5)
        ax_k.plot(iters, r["iter_graphbandwidths"],
                  **METHOD_STYLE[method], linewidth=2, markersize=5)

    ax_l.set_xlabel("Adam iteration", fontsize=12)
    ax_l.set_ylabel("lengthscale θ", fontsize=12)
    ax_l.set_title(f"θ trajectory at N = {n}", fontsize=13)
    ax_l.grid(True, alpha=0.3)
    ax_l.legend(fontsize=10)

    ax_k.set_xlabel("Adam iteration", fontsize=12)
    ax_k.set_ylabel("graphbandwidth κ", fontsize=12)
    ax_k.set_title(f"κ trajectory at N = {n}", fontsize=13)
    ax_k.grid(True, alpha=0.3)
    ax_k.legend(fontsize=10)


def main():
    # Default to the full Week-3 sweep (with N=10⁶). Fall back to the
    # Week-1 sweep if the full one isn't there yet.
    full = ROOT / "data" / "imgp_scaling_sweep_full.json"
    week1 = ROOT / "data" / "imgp_scaling_sweep.json"
    in_path = full if full.exists() else week1
    out_path = ROOT / "figures" / "imgp_scaling.png"
    os.makedirs(out_path.parent, exist_ok=True)
    data = load(in_path)

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2)
    ax_time = fig.add_subplot(gs[0, 0])
    ax_loss = fig.add_subplot(gs[0, 1])
    ax_l = fig.add_subplot(gs[1, 0])
    ax_k = fig.add_subplot(gs[1, 1])

    plot_wallclock_vs_n(data, ax_time)
    plot_loss_at_largest_n(data, ax_loss)
    plot_hyperparameter_trajectories(data, ax_l, ax_k)

    fig.suptitle(
        "IMGP marginal-likelihood training: differentiable SLQ "
        "(ours, blue) vs dense slogdet (green) vs naive Lanczos (red)",
        fontsize=14, y=1.00,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    fig.savefig(str(out_path).replace(".png", ".pdf"), bbox_inches="tight")
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
