#!/usr/bin/env python3
"""
Full scaling benchmark: GPU IRLM vs SciPy ARPACK from n=5K to n=10M.

Generates graph Laplacians at each scale, runs both solvers,
compares eigenvalue accuracy and wall-clock time.
"""

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.io import mmwrite, mmread
import subprocess
import time
import os
import sys
import json

# Try importing gpu_eigsh
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))
    from gpu_eigsh import gpu_eigsh
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    print("WARNING: gpu_eigsh not available, using CLI benchmark")

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
BENCH_BIN = os.path.join(os.path.dirname(__file__), '..', 'build', 'lanczos_bench')

# Test scales
SCALES = [5_000, 10_000, 50_000, 100_000, 500_000, 1_000_000, 5_000_000, 10_000_000]
K_EIGS = 20
K_NN = 15
DIM = 10
BW = 0.3
SEED = 123


def build_laplacian(n):
    """Build KNN graph Laplacian using fast KD-tree."""
    print(f"\n{'='*60}")
    print(f"Building graph Laplacian: n={n:,}")
    print(f"{'='*60}")

    t0 = time.time()
    rng = np.random.RandomState(SEED)
    points = rng.random((n, DIM))

    print(f"  KD-tree...")
    tree = cKDTree(points)
    del points  # free memory

    print(f"  KNN query (k={K_NN})...")
    distances, indices = tree.query(tree.data, k=K_NN + 1)
    del tree

    print(f"  Building sparse adjacency...")
    rows, cols, vals = [], [], []
    for i in range(n):
        for ki in range(1, K_NN + 1):
            j = indices[i, ki]
            w = np.exp(-distances[i, ki] ** 2 / (4 * BW ** 2))
            rows.append(i)
            cols.append(j)
            vals.append(w)
    del distances, indices

    A = csr_matrix((vals, (rows, cols)), shape=(n, n))
    del rows, cols, vals
    A_sym = A.maximum(A.T)
    del A

    print(f"  Building Laplacian L = D - A...")
    degrees = np.array(A_sym.sum(axis=1)).flatten()
    D = csr_matrix((degrees, (np.arange(n), np.arange(n))), shape=(n, n))
    L = D - A_sym
    del D, A_sym

    t1 = time.time()
    print(f"  Done: nnz={L.nnz:,}, build_time={t1-t0:.1f}s")
    return L, t1 - t0


def run_scipy(L, k):
    """Run scipy eigsh and return (eigenvalues, time_seconds)."""
    print(f"  SciPy ARPACK (k={k})...", end=" ", flush=True)
    t0 = time.time()
    try:
        vals, _ = eigsh(L, k=k, which='SM', tol=1e-10, maxiter=5000)
        t1 = time.time()
        vals = np.sort(vals)
        print(f"{t1-t0:.1f}s")
        return vals, t1 - t0
    except Exception as e:
        t1 = time.time()
        print(f"FAILED ({e})")
        return None, t1 - t0


def run_gpu_cli(mtx_path, k, ncv=120, max_iters=3000):
    """Run GPU IRLM via CLI and parse output."""
    print(f"  GPU IRLM (CLI, k={k}, ncv={ncv})...", end=" ", flush=True)
    cmd = [
        BENCH_BIN,
        '--mtx', mtx_path,
        '--eigs', str(k),
        '--iters', str(max_iters),
        '--ncv', str(ncv),
        '--irlm-only',
        '--outdir', DATA_DIR,
    ]
    t0 = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        t1 = time.time()

        # Parse time from output
        gpu_time = None
        for line in result.stdout.split('\n'):
            if 'Time:' in line and 'ms' in line:
                parts = line.strip().split()
                for j, p in enumerate(parts):
                    if p == 'ms':
                        gpu_time = float(parts[j-1]) / 1000.0  # ms -> s
                        break

        # Read eigenvalues
        eig_file = os.path.join(DATA_DIR, 'eigenvalues.csv')
        if os.path.exists(eig_file):
            data = np.genfromtxt(eig_file, delimiter=',', names=True)
            vals = data['irlm']
            nconv = len(vals)
        else:
            vals = None
            nconv = 0

        print(f"{gpu_time:.1f}s ({nconv}/{k} converged)")
        return vals, gpu_time
    except subprocess.TimeoutExpired:
        t1 = time.time()
        print(f"TIMEOUT (>{600}s)")
        return None, 600.0
    except Exception as e:
        t1 = time.time()
        print(f"FAILED ({e})")
        return None, t1 - t0


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    results = []

    for n in SCALES:
        # Memory estimate
        basis_mb = n * 121 * 8 / (1024**2)
        print(f"\n  Estimated GPU memory for basis: {basis_mb:.0f} MB")
        if basis_mb > 14000:
            print(f"  SKIPPING n={n:,} — would exceed GPU memory")
            results.append({
                'n': n, 'gpu_time': None, 'scipy_time': None,
                'max_rel_err': None, 'note': 'OOM'
            })
            continue

        # Build matrix
        L, build_time = build_laplacian(n)

        # Save MTX for CLI benchmark
        mtx_path = os.path.join(DATA_DIR, 'laplacian.mtx')
        print(f"  Writing MTX...", end=" ", flush=True)
        t0 = time.time()
        mmwrite(mtx_path, L)
        print(f"{time.time()-t0:.1f}s")

        # Run GPU
        gpu_vals, gpu_time = run_gpu_cli(mtx_path, K_EIGS)

        # Run scipy (skip for very large n to save time)
        if n <= 2_000_000:
            scipy_vals, scipy_time = run_scipy(L, K_EIGS)
        else:
            print(f"  SciPy: SKIPPED (n>{2_000_000:,}, would take too long)")
            scipy_vals, scipy_time = None, None

        # Compare
        max_err = None
        if gpu_vals is not None and scipy_vals is not None:
            k = min(len(gpu_vals), len(scipy_vals))
            if k > 1:
                errs = np.abs(gpu_vals[1:k] - scipy_vals[1:k]) / np.abs(scipy_vals[1:k])
                max_err = np.max(errs)

        speedup = scipy_time / gpu_time if (scipy_time and gpu_time and gpu_time > 0) else None

        results.append({
            'n': n,
            'nnz': L.nnz,
            'build_time': build_time,
            'gpu_time': gpu_time,
            'scipy_time': scipy_time,
            'speedup': speedup,
            'max_rel_err': max_err,
        })

        # Print row
        print(f"\n  RESULT: n={n:>10,}  GPU={gpu_time or 0:.1f}s  "
              f"SciPy={scipy_time or 0:.1f}s  "
              f"Speedup={speedup or 0:.1f}x  "
              f"MaxErr={max_err or 0:.2e}")

        del L  # free memory for next scale

    # Final summary table
    print(f"\n\n{'='*80}")
    print("SCALING BENCHMARK SUMMARY: GPU IRLM vs SciPy ARPACK")
    print(f"{'='*80}")
    print(f"{'n':>12}  {'nnz':>12}  {'GPU (s)':>10}  {'SciPy (s)':>10}  "
          f"{'Speedup':>8}  {'Max Err':>10}")
    print(f"{'-'*12}  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*10}")

    for r in results:
        n_str = f"{r['n']:>12,}"
        nnz_str = f"{r.get('nnz', 0):>12,}" if r.get('nnz') else f"{'—':>12}"
        gpu_str = f"{r['gpu_time']:>10.2f}" if r['gpu_time'] else f"{'OOM':>10}"
        sci_str = f"{r['scipy_time']:>10.2f}" if r['scipy_time'] else f"{'—':>10}"
        spd_str = f"{r['speedup']:>7.1f}x" if r.get('speedup') else f"{'—':>8}"
        err_str = f"{r['max_rel_err']:>10.2e}" if r.get('max_rel_err') is not None else f"{'—':>10}"
        print(f"{n_str}  {nnz_str}  {gpu_str}  {sci_str}  {spd_str}  {err_str}")

    # Save results as JSON
    json_path = os.path.join(DATA_DIR, 'scaling_results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {json_path}")


if __name__ == '__main__':
    main()
