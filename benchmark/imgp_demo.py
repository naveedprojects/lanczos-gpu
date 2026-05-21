#!/usr/bin/env python3
"""
Week-1 IMGP demo: differentiable SLQ vs IMGP paper baselines.

Trains a graph Matérn GP precision-operator hyperparameter set
(lengthscale θ, graphbandwidth κ) via marginal-likelihood maximisation,
using three methods for the log-determinant:

  1. IMGP-full   — dense `torch.linalg.slogdet` (works at small N, OOMs at scale).
                   This is the SS-IMGP-full row from Borovitskiy & Fichera 2023.
  2. IMGP-naive  — no-reortho unrolled Lanczos quadrature (the row of
                   that paper's Table 2 footnote — documented as failing
                   on numerical quality).
  3. IMGP-ours   — our differentiable SLQ with Krämer adjoint.

For each method, records per-iteration loss, hyperparameter trajectory,
and wall-clock time. Saves results to data/imgp_demo.json.

Plot script: benchmark/plot_imgp_demo.py.

Usage
-----
    /usr/bin/python3 benchmark/imgp_demo.py --n 2000 --max-iter 60
"""
import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from gpu_eigsh.imgp_train import (
    imgp_train_ours,
    imgp_train_dense,
    imgp_train_naive_lanczos,
)


# ---------------------------------------------------------------------
# Synthetic graph + signal generation
#
# We pull synthetic data from an IMGP-style setup: N points on a 2-D
# manifold embedded in R^d, k-NN graph, true signal sampled from a
# graph-Matérn GP with known hyperparameters. The hyperparameter
# trajectory should then converge towards (κ_true, θ_true).
# ---------------------------------------------------------------------

def make_synthetic_graph(n: int, *, k: int = 10, seed: int = 0,
                         kappa_true: float = 1.0, noise: float = 0.1):
    """Fast synthetic k-neighbour graph for scaling demos at N ≥ 10⁵.

    Skips the spatial KD-tree: edges are random per-node, distances are
    drawn from a half-normal. The graph structure is not metrically
    meaningful, but the spectral structure (sparse, k-regular,
    symmetric-normalised Laplacian) is what the SLQ scaling actually
    stresses — same as the metric-graph case for our purposes.
    """
    rng = np.random.default_rng(seed)

    # Random k-neighbour graph: each node points to k random nodes != self.
    src = np.repeat(np.arange(n), k)
    dst = rng.integers(0, n, size=n * k)
    # Remove self-edges.
    mask = src != dst
    src, dst = src[mask], dst[mask]
    # De-duplicate (src < dst).
    keep = src < dst
    src, dst = src[keep], dst[keep]
    # Squared edge distances from a half-normal.
    d_sq = (rng.standard_normal(src.shape[0]) ** 2)

    idx = torch.from_numpy(np.stack([src, dst], axis=0)).long()
    x_edge_dists = torch.from_numpy(d_sq).double()

    # Coarse surrogate signal y = L_sym(κ) ξ + ε.
    from gpu_eigsh.imgp import _matmul_symmetric_laplacian
    kappa = torch.tensor(kappa_true, dtype=torch.float64)
    xi = torch.from_numpy(rng.standard_normal(n)).double()
    smoothed = _matmul_symmetric_laplacian(
        xi, x_edge_dists, idx, n, kappa, self_loops=True,
    )
    eps = torch.from_numpy(rng.standard_normal(n)).double() * (noise ** 0.5)
    y = smoothed + eps
    return x_edge_dists, idx, y


def make_dataset(n: int, *, dim: int = 5, k: int = 10, seed: int = 0,
                 kappa_true: float = 1.0, ell_true: float = 0.5,
                 nu: int = 2, noise: float = 0.1,
                 dense_cov_max_n: int = 4000):
    """Generate N points and an IMGP-style training signal y.

    The signal y is drawn from a graph-Matérn GP prior. For small n
    (≤ `dense_cov_max_n`) we materialise the dense covariance and Cholesky-
    solve for exact sampling. For larger n we fall back to a cheap proxy:

        y_proxy  =  L_sym(κ_true) · ξ  +  ε

    where ξ ~ N(0, I) is white noise. This is a *coarse* surrogate for
    the true GP sample but keeps the spectral signature of the graph
    (smoothed structure on neighbouring nodes), which is enough for
    correctness + scaling demos. The exact recovered (κ, θ) values
    won't match the prior, but all three methods see the same y and
    should converge to identical estimates — that's what the demo is
    showing.
    """
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(seed)
    pts = rng.standard_normal((n, dim))
    tree = cKDTree(pts)
    dists, idxs = tree.query(pts, k=k + 1)
    nbr_dists = dists[:, 1:]
    nbr_idxs = idxs[:, 1:]

    src = np.repeat(np.arange(n), k)
    dst = nbr_idxs.ravel()
    d_sq = (nbr_dists ** 2).ravel()
    keep = src < dst
    src, dst, d_sq = src[keep], dst[keep], d_sq[keep]

    idx = torch.from_numpy(np.stack([src, dst], axis=0)).long()
    x_edge_dists = torch.from_numpy(d_sq).double()

    from gpu_eigsh.imgp import make_imgp_precision_matvec, _matmul_symmetric_laplacian
    kappa = torch.tensor(kappa_true, dtype=torch.float64)
    ell = torch.tensor(ell_true, dtype=torch.float64)

    if n <= dense_cov_max_n:
        # Exact: sample y ~ N(0, P^{-1} + noise·I) via dense Cholesky.
        matvec = make_imgp_precision_matvec(
            x_edge_dists=x_edge_dists, idx=idx, operator_dim=n, nu=nu,
            normalization="symmetric", self_loops=True,
        )
        I = torch.eye(n, dtype=torch.float64)
        P = torch.stack([matvec(I[:, i], ell, kappa) for i in range(n)], dim=1)
        Sigma = torch.linalg.inv(P) + noise * I
        L = torch.linalg.cholesky(Sigma)
        z = torch.from_numpy(rng.standard_normal(n)).double()
        y = L @ z
    else:
        # Coarse surrogate at scale: L_sym(κ) · ξ + ε.
        xi = torch.from_numpy(rng.standard_normal(n)).double()
        smoothed = _matmul_symmetric_laplacian(
            xi, x_edge_dists, idx, n, kappa, self_loops=True,
        )
        eps = torch.from_numpy(rng.standard_normal(n)).double() * (noise ** 0.5)
        y = smoothed + eps

    return x_edge_dists, idx, y


# ---------------------------------------------------------------------
# Run all three methods on the same data
# ---------------------------------------------------------------------

def run_demo(*, n: int, max_iter: int, lr: float, m_probes_ours: int,
             lanczos_m_ours: int, m_probes_naive: int, lanczos_m_naive: int,
             seed: int, run_full: bool, verbose: bool):
    print(f"=== IMGP demo: n={n}, max_iter={max_iter} ===")
    print(f"Generating dataset (synthetic IMGP signal)...")
    t0 = time.perf_counter()
    x_edge_dists, idx, y = make_dataset(n=n, seed=seed)
    print(f"  done in {time.perf_counter() - t0:.2f}s. n={n}, edges={idx.shape[1]:,}")

    init = dict(
        train_targets=y, x_edge_dists=x_edge_dists, idx=idx, n=n,
        lengthscale_init=1.5, graphbandwidth_init=2.0,
        max_iter=max_iter, lr=lr, verbose=verbose,
    )

    print(f"\n[1/3] IMGP-naive (no-reortho Lanczos, paper's failing row)...")
    res_naive = imgp_train_naive_lanczos(
        m_probes=m_probes_naive, lanczos_m=lanczos_m_naive, seed=seed, **init,
    )
    print("  " + res_naive.summary_line())

    print(f"\n[2/3] IMGP-ours (Krämer-adjoint SLQ)...")
    res_ours = imgp_train_ours(
        m_probes=m_probes_ours, lanczos_m=lanczos_m_ours, seed=seed, **init,
    )
    print("  " + res_ours.summary_line())

    res_full = None
    if run_full:
        print(f"\n[3/3] IMGP-full (dense slogdet — paper's SS-IMGP-full row)...")
        res_full = imgp_train_dense(**init)
        print("  " + res_full.summary_line())
    else:
        print(f"\n[3/3] IMGP-full skipped (--no-full).")

    return {
        "n": n, "edges": int(idx.shape[1]), "seed": seed,
        "results": {
            "imgp-naive": asdict(res_naive),
            "imgp-ours":  asdict(res_ours),
            "imgp-full":  asdict(res_full) if res_full else None,
        },
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--max-iter", type=int, default=60)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--m-probes-ours", type=int, default=50)
    p.add_argument("--lanczos-m-ours", type=int, default=20)
    p.add_argument("--m-probes-naive", type=int, default=10)
    p.add_argument("--lanczos-m-naive", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-full", action="store_true",
                   help="skip the dense SS-IMGP-full baseline (use for large n)")
    p.add_argument("--out", type=str,
                   default=str(ROOT / "data" / "imgp_demo.json"))
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    out = run_demo(
        n=args.n, max_iter=args.max_iter, lr=args.lr,
        m_probes_ours=args.m_probes_ours,
        lanczos_m_ours=args.lanczos_m_ours,
        m_probes_naive=args.m_probes_naive,
        lanczos_m_naive=args.lanczos_m_naive,
        seed=args.seed, run_full=not args.no_full, verbose=not args.quiet,
    )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults written to {args.out}")


if __name__ == "__main__":
    main()
