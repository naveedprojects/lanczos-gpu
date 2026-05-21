#!/usr/bin/env python3
"""
Scan IMGP training across N to show:

  - IMGP-full   becomes prohibitive (O(N²) memory, O(N³) compute) past
                some N; we stop running it past --max-n-full.
  - IMGP-ours   keeps working with controlled cost.
  - IMGP-naive  runs at every N but the loss trajectory may degrade with
                respect to IMGP-full and IMGP-ours.

The output JSON is consumed by `plot_imgp_demo.py` to build the Week-1
figures (loss-trajectory comparison + wall-clock-vs-N scaling).

Usage
-----
    /usr/bin/python3 benchmark/imgp_scaling_sweep.py \
            --ns 200 500 1000 2000 --max-n-full 1000 --max-iter 30
"""
import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "benchmark"))

from gpu_eigsh.imgp_train import (
    imgp_train_ours,
    imgp_train_dense,
    imgp_train_naive_lanczos,
)
from imgp_demo import make_dataset, make_synthetic_graph  # noqa: E402


def run_scan(ns, *, max_n_full, max_iter, lr, m_probes_ours, lanczos_m_ours,
             m_probes_naive, lanczos_m_naive, seed, device,
             synthetic_threshold: int, dtype: torch.dtype):
    sweep = []
    for n in ns:
        print(f"\n=== n = {n} (device={device}, dtype={dtype}) ===")
        t0 = time.perf_counter()
        if n >= synthetic_threshold:
            x_edge_dists, idx, y = make_synthetic_graph(n=n, seed=seed)
        else:
            x_edge_dists, idx, y = make_dataset(n=n, seed=seed)
        # Move to device and cast to requested precision.
        x_edge_dists = x_edge_dists.to(device=device, dtype=dtype)
        y = y.to(device=device, dtype=dtype)
        idx = idx.to(device)
        print(f"  dataset built in {time.perf_counter() - t0:.2f}s "
              f"({idx.shape[1]:,} edges)")

        init = dict(
            train_targets=y, x_edge_dists=x_edge_dists, idx=idx, n=n,
            lengthscale_init=1.5, graphbandwidth_init=2.0,
            max_iter=max_iter, lr=lr, verbose=False,
        )

        entry = {"n": n, "edges": int(idx.shape[1])}

        # Always run ours and naive
        for label, fn, kwargs in [
            ("imgp-ours", imgp_train_ours,
             dict(m_probes=m_probes_ours, lanczos_m=lanczos_m_ours)),
            ("imgp-naive", imgp_train_naive_lanczos,
             dict(m_probes=m_probes_naive, lanczos_m=lanczos_m_naive)),
        ]:
            print(f"  {label}...")
            t0 = time.perf_counter()
            res = fn(**init, seed=seed, **kwargs)
            wall = time.perf_counter() - t0
            entry[label] = asdict(res)
            print(f"    {wall:.1f}s, loss={res.final_loss:.3f}, "
                  f"θ={res.final_lengthscale:.3f}, κ={res.final_graphbandwidth:.3f}"
                  f"{' [DIVERGED]' if res.diverged else ''}")

        if n <= max_n_full:
            print(f"  imgp-full...")
            t0 = time.perf_counter()
            res = imgp_train_dense(**init)
            wall = time.perf_counter() - t0
            entry["imgp-full"] = asdict(res)
            print(f"    {wall:.1f}s, loss={res.final_loss:.3f}, "
                  f"θ={res.final_lengthscale:.3f}, κ={res.final_graphbandwidth:.3f}")
        else:
            entry["imgp-full"] = None
            print(f"  imgp-full SKIPPED (n > {max_n_full})")

        sweep.append(entry)
    return sweep


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ns", nargs="+", type=int,
                   default=[200, 500, 1000, 2000])
    p.add_argument("--max-n-full", type=int, default=1000,
                   help="largest N at which to run the dense baseline")
    p.add_argument("--max-iter", type=int, default=30)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--m-probes-ours", type=int, default=30)
    p.add_argument("--lanczos-m-ours", type=int, default=20)
    p.add_argument("--m-probes-naive", type=int, default=10)
    p.add_argument("--lanczos-m-naive", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu",
                   help="cpu or cuda")
    p.add_argument("--dtype", type=str, default="float64",
                   choices=["float64", "float32"],
                   help="Tensor precision. float32 halves GPU memory; "
                        "useful past N ≈ 500,000 on 16 GB.")
    p.add_argument("--synthetic-threshold", type=int, default=100_000,
                   help="N at or above which to use the random synthetic "
                        "graph (no cKDTree). Set to a very large value "
                        "to disable.")
    p.add_argument("--out", type=str,
                   default=str(ROOT / "data" / "imgp_scaling_sweep.json"))
    args = p.parse_args()

    dtype = (torch.float32 if args.dtype == "float32" else torch.float64)
    sweep = run_scan(
        args.ns, max_n_full=args.max_n_full, max_iter=args.max_iter,
        lr=args.lr,
        m_probes_ours=args.m_probes_ours, lanczos_m_ours=args.lanczos_m_ours,
        m_probes_naive=args.m_probes_naive, lanczos_m_naive=args.lanczos_m_naive,
        seed=args.seed, device=args.device,
        synthetic_threshold=args.synthetic_threshold, dtype=dtype,
    )
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"runs": sweep}, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
