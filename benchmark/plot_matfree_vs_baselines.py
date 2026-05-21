#!/usr/bin/env python3
"""Plot matfree vs GPyTorch SLQ from data/matfree_vs_baselines.json."""
import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "matfree_vs_baselines.json")
FIG_DIR = os.path.join(ROOT, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

with open(DATA) as f:
    results = json.load(f)

ns = [r["n"] for r in results]
mf_fwd = [r["matfree"]["fwd_ms"] for r in results]
mf_bwd = [r["matfree"]["bwd_ms"] for r in results]
gp_fwd = [r["gpytorch"]["fwd_ms"] for r in results]
gp_bwd = [r["gpytorch"]["bwd_ms"] for r in results]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Forward
ax = axes[0]
ax.loglog(ns, mf_fwd, "o-", color="#C44536", lw=2, ms=8, label="matfree (JAX, GPU)")
ax.loglog(ns, gp_fwd, "s-", color="#197278", lw=2, ms=8, label="GPyTorch SLQ (GPU)")
ax.set_xlabel("n (graph size)", fontsize=12)
ax.set_ylabel("Forward time (ms)", fontsize=12)
ax.set_title("SLQ log det forward\n(20 probes, m=50)", fontsize=13)
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=11)

# Backward
ax = axes[1]
ax.loglog(ns, mf_bwd, "o-", color="#C44536", lw=2, ms=8, label="matfree (JAX, GPU)")
ax.loglog(ns, gp_bwd, "s-", color="#197278", lw=2, ms=8, label="GPyTorch SLQ (GPU)")
ax.set_xlabel("n (graph size)", fontsize=12)
ax.set_ylabel("Backward time (ms)", fontsize=12)
ax.set_title("SLQ gradient w.r.t. κ\n(IMGP precision operator)", fontsize=13)
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=11)

fig.suptitle(
    "Week-0 gap check: differentiable SLQ on the IMGP-style precision operator",
    fontsize=14, y=1.02
)
fig.tight_layout()
out = os.path.join(FIG_DIR, "matfree_vs_baselines.png")
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"saved {out}")
fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
