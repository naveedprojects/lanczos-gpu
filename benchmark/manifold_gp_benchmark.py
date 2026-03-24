#!/usr/bin/env python3
"""
Benchmark: manifold-gp eigendecomposition scaling.

Compares three approaches at increasing n:
  1. torch.linalg.eigh(L.to_dense())  — O(n³) time, O(n²) memory (current manifold-gp)
  2. scipy.sparse.linalg.eigsh         — O(n·k·iters) time, sparse memory (CPU ARPACK)
  3. GPU IRLM (ours)                   — same complexity, GPU-accelerated

Uses the actual manifold-gp kernel with symmetric Laplacian normalization.
"""

import sys
import os
import time
import math
import json
import numpy as np
import torch
import gpytorch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
BENCH_BIN = os.path.join(os.path.dirname(__file__), '..', 'build', 'lanczos_bench')
FIG_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
NUM_MODES = 50
NN = 30


def build_graph_laplacian(n):
    """Build a KNN graph Laplacian using scipy KD-tree (no FAISS dependency).

    This produces the same type of symmetric graph Laplacian that
    manifold-gp's RiemannMaternKernel uses internally.
    """
    from scipy.spatial import cKDTree
    from scipy.sparse import csr_matrix
    from scipy.io import mmwrite

    np.random.seed(42)
    dim = 10  # moderate dimensionality
    bw = 1.0  # bandwidth scaled for dim

    points = np.random.randn(n, dim)
    tree = cKDTree(points)
    dists, idxs = tree.query(points, k=NN + 1)

    # Vectorized adjacency construction
    neighbor_dists = dists[:, 1:]
    neighbor_idxs = idxs[:, 1:]
    weights = np.exp(-neighbor_dists**2 / (4.0 * bw**2))

    rows = np.repeat(np.arange(n), NN)
    cols = neighbor_idxs.ravel()
    vals = weights.ravel()

    A = csr_matrix((vals, (rows, cols)), shape=(n, n))
    A_sym = A.maximum(A.T)

    # Symmetric normalized Laplacian: L_sym = D^{-1/2} (D - A) D^{-1/2}
    degrees = np.array(A_sym.sum(axis=1)).flatten()
    D_inv_sqrt = csr_matrix((1.0 / np.sqrt(np.maximum(degrees, 1e-15)),
                              (np.arange(n), np.arange(n))), shape=(n, n))
    D = csr_matrix((degrees, (np.arange(n), np.arange(n))), shape=(n, n))
    L = D - A_sym
    L_sym = D_inv_sqrt @ L @ D_inv_sqrt  # symmetric normalized

    mtx_path = os.path.join(DATA_DIR, 'manifold_bench.mtx')
    mmwrite(mtx_path, L_sym)

    return L_sym, mtx_path


def run_eigh(L_scipy, n, k, device):
    """torch.linalg.eigh on dense matrix."""
    mem_gb = n * n * 8 / 1024**3
    if mem_gb > 12:
        return None, float('inf'), 'OOM'

    dense = torch.from_numpy(L_scipy.toarray()).double().to(device)
    torch.cuda.synchronize()
    t0 = time.time()
    evals, _ = torch.linalg.eigh(dense)
    torch.cuda.synchronize()
    t1 = time.time()
    del dense
    torch.cuda.empty_cache()
    return evals[:k].cpu().numpy(), t1 - t0, 'OK'


def run_scipy_eigsh(L_scipy, k):
    """scipy.sparse.linalg.eigsh."""
    from scipy.sparse.linalg import eigsh
    t0 = time.time()
    evals, _ = eigsh(L_scipy, k=k, which='SM', tol=1e-10)
    t1 = time.time()
    return np.sort(evals), t1 - t0, 'OK'


def run_gpu_irlm(mtx_path, k):
    """Our GPU IRLM via CLI."""
    import subprocess
    cmd = [BENCH_BIN, '--mtx', mtx_path, '--eigs', str(k),
           '--iters', '3000', '--ncv', str(min(3 * k, 300)),
           '--irlm-only', '--outdir', DATA_DIR]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        # Parse time
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
        evals = data['irlm']
        return evals, gpu_time, 'OK'
    except Exception as e:
        return None, float('inf'), str(e)


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")

    # Scales to test
    scales = [1000, 2000, 5000, 10000, 20000, 50000, 100000]

    results = []

    for n in scales:
        print(f"\n{'='*60}")
        print(f"n = {n:,}")
        print(f"{'='*60}")

        # Build graph Laplacian (no FAISS, uses scipy KD-tree)
        print("  Building symmetric Laplacian...")
        t0 = time.time()
        L_scipy, mtx_path = build_graph_laplacian(n)
        build_time = time.time() - t0
        print(f"  Built in {build_time:.1f}s, nnz={L_scipy.nnz:,}")

        # 1. eigh
        print(f"  torch.linalg.eigh...", end=" ", flush=True)
        eigh_vals, eigh_time, eigh_status = run_eigh(L_scipy, n, NUM_MODES, device)
        print(f"{eigh_time:.2f}s ({eigh_status})")

        # 2. scipy eigsh
        print(f"  scipy eigsh...", end=" ", flush=True)
        scipy_vals, scipy_time, scipy_status = run_scipy_eigsh(L_scipy, NUM_MODES)
        print(f"{scipy_time:.2f}s ({scipy_status})")

        # 3. GPU IRLM
        print(f"  GPU IRLM...", end=" ", flush=True)
        gpu_vals, gpu_time, gpu_status = run_gpu_irlm(mtx_path, NUM_MODES)
        print(f"{gpu_time:.2f}s ({gpu_status})")

        # Accuracy check
        max_err = None
        if scipy_vals is not None and gpu_vals is not None:
            k = min(len(scipy_vals), len(gpu_vals))
            if k > 1:
                max_err = np.max(np.abs(gpu_vals[1:k] - scipy_vals[1:k]) /
                                 np.maximum(np.abs(scipy_vals[1:k]), 1e-15))
                print(f"  Max rel error (IRLM vs scipy): {max_err:.2e}")

        results.append({
            'n': n,
            'nnz': L_scipy.nnz,
            'eigh_time': eigh_time if eigh_status == 'OK' else None,
            'scipy_time': scipy_time if scipy_status == 'OK' else None,
            'gpu_time': gpu_time if gpu_status == 'OK' else None,
            'max_rel_err': float(max_err) if max_err is not None else None,
            'eigh_status': eigh_status,
        })

        del L_scipy
        torch.cuda.empty_cache()

    # Print summary table
    print(f"\n\n{'='*80}")
    print("MANIFOLD-GP EIGENDECOMPOSITION SCALING BENCHMARK")
    print(f"{'='*80}")
    print(f"{'n':>8}  {'eigh (s)':>10}  {'scipy (s)':>10}  {'GPU IRLM (s)':>12}  "
          f"{'GPU vs eigh':>11}  {'GPU vs scipy':>12}  {'Max Err':>10}")
    print(f"{'-'*8}  {'-'*10}  {'-'*10}  {'-'*12}  {'-'*11}  {'-'*12}  {'-'*10}")

    for r in results:
        n_s = f"{r['n']:>8,}"
        eigh_s = f"{r['eigh_time']:>10.2f}" if r['eigh_time'] else f"{'OOM':>10}"
        sci_s = f"{r['scipy_time']:>10.2f}" if r['scipy_time'] else f"{'—':>10}"
        gpu_s = f"{r['gpu_time']:>12.2f}" if r['gpu_time'] else f"{'—':>12}"
        vs_eigh = f"{r['eigh_time']/r['gpu_time']:>10.1f}x" if (r['eigh_time'] and r['gpu_time']) else f"{'—':>11}"
        vs_sci = f"{r['scipy_time']/r['gpu_time']:>11.1f}x" if (r['scipy_time'] and r['gpu_time']) else f"{'—':>12}"
        err_s = f"{r['max_rel_err']:>10.1e}" if r['max_rel_err'] is not None else f"{'—':>10}"
        print(f"{n_s}  {eigh_s}  {sci_s}  {gpu_s}  {vs_eigh}  {vs_sci}  {err_s}")

    # Save JSON
    json_path = os.path.join(DATA_DIR, 'manifold_scaling.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Generate plot
    fig, ax = plt.subplots(figsize=(8, 5))

    ns = [r['n'] for r in results]

    # eigh
    eigh_ns = [r['n'] for r in results if r['eigh_time']]
    eigh_ts = [r['eigh_time'] for r in results if r['eigh_time']]
    if eigh_ns:
        ax.semilogy(eigh_ns, eigh_ts, 'r-s', markersize=7, linewidth=2,
                    markeredgecolor='white', markeredgewidth=0.5,
                    label='torch.linalg.eigh (dense, current)')
        # Mark OOM
        oom_ns = [r['n'] for r in results if r['eigh_status'] == 'OOM']
        if oom_ns:
            for on in oom_ns:
                ax.axvline(x=on, color='red', linestyle=':', alpha=0.3)
            ax.annotate('OOM\n(dense matrix\nexceeds GPU memory)',
                       xy=(oom_ns[0], eigh_ts[-1] * 2),
                       fontsize=9, color='red', ha='center')

    # scipy
    sci_ns = [r['n'] for r in results if r['scipy_time']]
    sci_ts = [r['scipy_time'] for r in results if r['scipy_time']]
    if sci_ns:
        ax.semilogy(sci_ns, sci_ts, 'b-^', markersize=7, linewidth=2,
                    markeredgecolor='white', markeredgewidth=0.5,
                    label='scipy eigsh (CPU ARPACK)')

    # GPU IRLM
    gpu_ns = [r['n'] for r in results if r['gpu_time']]
    gpu_ts = [r['gpu_time'] for r in results if r['gpu_time']]
    if gpu_ns:
        ax.semilogy(gpu_ns, gpu_ts, 'g-D', markersize=7, linewidth=2,
                    markeredgecolor='white', markeredgewidth=0.5,
                    label='GPU IRLM (ours)')

    ax.set_xlabel('Number of Samples (n)', fontsize=13)
    ax.set_ylabel('Eigendecomposition Time (seconds)', fontsize=13)
    ax.set_title('Manifold-GP Eigendecomposition Scaling\n'
                 f'(symmetric Laplacian, k={NUM_MODES} modes, nn={NN})',
                 fontsize=14)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(FIG_DIR, 'manifold_gp_scaling.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.savefig(plot_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"\nPlot saved to {plot_path}")


if __name__ == '__main__':
    main()
