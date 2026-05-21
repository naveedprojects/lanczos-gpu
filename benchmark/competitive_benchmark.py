#!/usr/bin/env python3
"""
Competitive benchmark: GPU IRLM vs cupyx.eigsh vs torch.lobpcg vs scipy.eigsh

Compares four eigensolvers on symmetric graph Laplacians at increasing scale.
All methods compute k=50 smallest eigenvalues of the same sparse matrix.
"""

import os
import sys
import time
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
FIG_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
BENCH_BIN = os.path.join(os.path.dirname(__file__), '..', 'build', 'lanczos_bench')

NUM_EIGS = 50
NN_K = 30


def build_laplacian(n):
    """Build symmetric normalized graph Laplacian via KD-tree."""
    from scipy.spatial import cKDTree
    from scipy.sparse import csr_matrix
    from scipy.io import mmwrite

    np.random.seed(42)
    points = np.random.randn(n, 10)
    tree = cKDTree(points)
    dists, idxs = tree.query(points, k=NN_K + 1)

    rows = np.repeat(np.arange(n), NN_K)
    cols = idxs[:, 1:].ravel()
    vals = np.exp(-dists[:, 1:]**2 / 4.0).ravel()

    A = csr_matrix((vals, (rows, cols)), shape=(n, n))
    A_sym = A.maximum(A.T)
    degrees = np.array(A_sym.sum(axis=1)).flatten()

    from scipy.sparse import diags
    D_inv_sqrt = diags(1.0 / np.sqrt(np.maximum(degrees, 1e-15)))
    D = diags(degrees)
    L = D - A_sym
    L_sym = D_inv_sqrt @ L @ D_inv_sqrt

    mtx_path = os.path.join(DATA_DIR, 'competitive_bench.mtx')
    mmwrite(mtx_path, L_sym)
    return L_sym, mtx_path


def run_scipy_eigsh(L, k):
    """scipy.sparse.linalg.eigsh (CPU ARPACK)."""
    from scipy.sparse.linalg import eigsh
    t0 = time.time()
    evals, _ = eigsh(L, k=k, which='SM', tol=1e-10)
    return np.sort(evals), time.time() - t0


def run_cupyx_eigsh(L, k):
    """cupyx.scipy.sparse.linalg.eigsh (GPU ARPACK via CuPy)."""
    import cupy as cp
    import cupyx.scipy.sparse as cpsp
    import cupyx.scipy.sparse.linalg as cpsla

    # Convert scipy CSR to CuPy CSR
    L_gpu = cpsp.csr_matrix(L.astype(np.float64))
    cp.cuda.Stream.null.synchronize()

    t0 = time.time()
    # cupyx doesn't support 'SM', use 'SA' (equivalent for PSD matrices)
    evals, _ = cpsla.eigsh(L_gpu, k=k, which='SA')
    cp.cuda.Stream.null.synchronize()
    t1 = time.time()

    return np.sort(cp.asnumpy(evals)), t1 - t0


def run_torch_lobpcg(L, k, device):
    """torch.lobpcg on sparse matrix."""
    # Convert to torch sparse
    L_coo = L.tocoo()
    indices = torch.tensor(np.vstack([L_coo.row, L_coo.col]), dtype=torch.long)
    values = torch.tensor(L_coo.data, dtype=torch.float64)
    L_torch = torch.sparse_coo_tensor(indices, values, L.shape).to(device).coalesce()

    # Random initial vectors
    torch.manual_seed(42)
    X = torch.randn(L.shape[0], k, dtype=torch.float64, device=device)

    torch.cuda.synchronize()
    t0 = time.time()
    try:
        evals, _ = torch.lobpcg(L_torch, k=k, X=X, largest=False, niter=1000, tol=1e-10)
        torch.cuda.synchronize()
        t1 = time.time()
        return np.sort(evals.cpu().numpy()), t1 - t0
    except Exception as e:
        return None, float('inf')


def run_gpu_irlm(mtx_path, k):
    """Our GPU IRLM via CLI."""
    import subprocess
    cmd = [BENCH_BIN, '--mtx', mtx_path, '--eigs', str(k),
           '--iters', '3000', '--ncv', str(min(3 * k, 300)),
           '--irlm-only', '--outdir', DATA_DIR]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

    gpu_time = None
    for line in result.stdout.split('\n'):
        if 'Time:' in line and 'ms' in line:
            parts = line.strip().split()
            for j, p in enumerate(parts):
                if p == 'ms':
                    gpu_time = float(parts[j - 1]) / 1000.0
                    break
            if gpu_time is not None:
                break

    eig_file = os.path.join(DATA_DIR, 'eigenvalues.csv')
    data = np.genfromtxt(eig_file, delimiter=',', names=True)
    return data['irlm'], gpu_time


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    device = torch.device('cuda')
    print(f"GPU: {torch.cuda.get_device_name()}")

    scales = [1000, 2000, 5000, 10000, 20000, 50000, 100000]
    results = []

    for n in scales:
        print(f"\n{'='*60}")
        print(f"n = {n:,}")
        print(f"{'='*60}")

        # Build matrix
        print("  Building Laplacian...", end=" ", flush=True)
        t0 = time.time()
        L, mtx_path = build_laplacian(n)
        print(f"{time.time()-t0:.1f}s, nnz={L.nnz:,}")

        # Ground truth: scipy
        print("  scipy eigsh...", end=" ", flush=True)
        scipy_evals, scipy_time = run_scipy_eigsh(L, NUM_EIGS)
        print(f"{scipy_time:.2f}s")

        # cupyx eigsh
        print("  cupyx eigsh...", end=" ", flush=True)
        try:
            cupyx_evals, cupyx_time = run_cupyx_eigsh(L, NUM_EIGS)
            cupyx_err = np.max(np.abs(cupyx_evals[1:] - scipy_evals[1:]) /
                               np.abs(scipy_evals[1:]))
            print(f"{cupyx_time:.2f}s (err={cupyx_err:.1e})")
        except Exception as e:
            cupyx_evals, cupyx_time, cupyx_err = None, float('inf'), None
            print(f"FAILED ({e})")

        # torch.lobpcg
        print("  torch.lobpcg...", end=" ", flush=True)
        try:
            lobpcg_evals, lobpcg_time = run_torch_lobpcg(L, NUM_EIGS, device)
            if lobpcg_evals is not None:
                lobpcg_err = np.max(np.abs(lobpcg_evals[1:] - scipy_evals[1:]) /
                                    np.abs(scipy_evals[1:]))
                print(f"{lobpcg_time:.2f}s (err={lobpcg_err:.1e})")
            else:
                lobpcg_err = None
                print(f"FAILED")
        except Exception as e:
            lobpcg_evals, lobpcg_time, lobpcg_err = None, float('inf'), None
            print(f"FAILED ({e})")
        torch.cuda.empty_cache()

        # GPU IRLM (ours)
        print("  GPU IRLM...", end=" ", flush=True)
        irlm_evals, irlm_time = run_gpu_irlm(mtx_path, NUM_EIGS)
        irlm_err = np.max(np.abs(irlm_evals[1:] - scipy_evals[1:len(irlm_evals)]) /
                          np.abs(scipy_evals[1:len(irlm_evals)]))
        print(f"{irlm_time:.2f}s (err={irlm_err:.1e})")

        results.append({
            'n': n, 'nnz': L.nnz,
            'scipy_time': scipy_time,
            'cupyx_time': cupyx_time if cupyx_time < float('inf') else None,
            'cupyx_err': float(cupyx_err) if cupyx_err is not None else None,
            'lobpcg_time': lobpcg_time if lobpcg_time < float('inf') else None,
            'lobpcg_err': float(lobpcg_err) if lobpcg_err is not None else None,
            'irlm_time': irlm_time,
            'irlm_err': float(irlm_err),
        })
        del L

    # Summary table
    print(f"\n\n{'='*90}")
    print(f"COMPETITIVE BENCHMARK: k={NUM_EIGS} smallest eigenvalues")
    print(f"{'='*90}")
    print(f"{'n':>8}  {'scipy':>8}  {'cupyx':>8}  {'lobpcg':>8}  {'IRLM':>8}  "
          f"{'cupyx err':>10}  {'lobpcg err':>10}  {'IRLM err':>10}")
    print(f"{'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*10}")

    for r in results:
        def fmt_t(v): return f"{v:>8.2f}" if v else f"{'—':>8}"
        def fmt_e(v): return f"{v:>10.1e}" if v is not None else f"{'—':>10}"
        print(f"{r['n']:>8,}  {fmt_t(r['scipy_time'])}  {fmt_t(r['cupyx_time'])}  "
              f"{fmt_t(r['lobpcg_time'])}  {fmt_t(r['irlm_time'])}  "
              f"{fmt_e(r['cupyx_err'])}  {fmt_e(r['lobpcg_err'])}  {fmt_e(r['irlm_err'])}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ns = [r['n'] for r in results]

    # Left: timing
    ax1.semilogy(ns, [r['scipy_time'] for r in results],
                'b-^', ms=7, lw=2, label='scipy eigsh (CPU ARPACK)')
    cupyx_ns = [r['n'] for r in results if r['cupyx_time']]
    cupyx_ts = [r['cupyx_time'] for r in results if r['cupyx_time']]
    if cupyx_ns:
        ax1.semilogy(cupyx_ns, cupyx_ts,
                    'm-o', ms=7, lw=2, label='cupyx eigsh (GPU CuPy)')
    lobpcg_ns = [r['n'] for r in results if r['lobpcg_time']]
    lobpcg_ts = [r['lobpcg_time'] for r in results if r['lobpcg_time']]
    if lobpcg_ns:
        ax1.semilogy(lobpcg_ns, lobpcg_ts,
                    'c-v', ms=7, lw=2, label='torch.lobpcg')
    ax1.semilogy(ns, [r['irlm_time'] for r in results],
                'g-D', ms=7, lw=2, label='GPU IRLM (ours)')

    ax1.set_xlabel('Matrix Size (n)', fontsize=13)
    ax1.set_ylabel('Time (seconds)', fontsize=13)
    ax1.set_title(f'Eigendecomposition Time (k={NUM_EIGS})', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Right: accuracy
    ax2.semilogy(ns, [r['irlm_err'] for r in results],
                'g-D', ms=7, lw=2, label='GPU IRLM (ours)')
    if any(r['cupyx_err'] is not None for r in results):
        ce_ns = [r['n'] for r in results if r['cupyx_err'] is not None]
        ce_vs = [r['cupyx_err'] for r in results if r['cupyx_err'] is not None]
        ax2.semilogy(ce_ns, ce_vs, 'm-o', ms=7, lw=2, label='cupyx eigsh')
    if any(r['lobpcg_err'] is not None for r in results):
        le_ns = [r['n'] for r in results if r['lobpcg_err'] is not None]
        le_vs = [r['lobpcg_err'] for r in results if r['lobpcg_err'] is not None]
        ax2.semilogy(le_ns, le_vs, 'c-v', ms=7, lw=2, label='torch.lobpcg')

    ax2.axhline(y=2.2e-16, color='gray', ls='--', lw=1, alpha=0.6,
                label=r'Machine $\epsilon$')
    ax2.set_xlabel('Matrix Size (n)', fontsize=13)
    ax2.set_ylabel('Max Relative Error vs scipy', fontsize=13)
    ax2.set_title('Eigenvalue Accuracy', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(FIG_DIR, 'competitive_benchmark.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.savefig(plot_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"\nPlot saved to {plot_path}")

    # Save JSON
    with open(os.path.join(DATA_DIR, 'competitive_results.json'), 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
