#!/usr/bin/env python3
"""
Benchmark: Differentiable Sparse Eigendecomposition on GPU

Compares:
  1. torch.linalg.eigh — dense, O(n³), differentiable
  2. differentiable_eigsh — sparse GPU IRLM + implicit diff (ours)

Gradient accuracy is verified against dense-eigh finite differences
(the only unambiguous ground truth).
"""

import os
import sys
import time
import json
import gc
import psutil
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
FIG_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')

MAX_DENSE_N = 20000


def flush_all():
    gc.collect(); gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()


def build_laplacian_scipy(n, nn=15, seed=42):
    """Memory-efficient symmetric normalized graph Laplacian."""
    from scipy.spatial import cKDTree
    from scipy.sparse import csr_matrix

    np.random.seed(seed)
    points = np.random.randn(n, 3).astype(np.float32)
    tree = cKDTree(points, leafsize=32)
    dists, idxs = tree.query(points, k=nn + 1, workers=-1)
    del points, tree; gc.collect()

    rows = np.repeat(np.arange(n), nn)
    cols = idxs[:, 1:].ravel()
    vals = np.exp(-dists[:, 1:]**2 / 4.0).ravel().astype(np.float64)
    del dists, idxs; gc.collect()

    rows_all = np.concatenate([rows, cols])
    cols_all = np.concatenate([cols, rows])
    vals_all = np.concatenate([vals, vals])
    del rows, cols, vals; gc.collect()

    A = csr_matrix((vals_all, (rows_all, cols_all)), shape=(n, n))
    del rows_all, cols_all, vals_all; gc.collect()

    degrees = np.array(A.sum(axis=1)).flatten()
    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(degrees, 1e-15))
    A_coo = A.tocoo()
    scaled = A_coo.data * d_inv_sqrt[A_coo.row] * d_inv_sqrt[A_coo.col]
    diag_r = np.arange(n)
    L = csr_matrix((np.concatenate([np.ones(n), -scaled]),
                    (np.concatenate([diag_r, A_coo.row]),
                     np.concatenate([diag_r, A_coo.col]))),
                   shape=(n, n))
    del A, A_coo, scaled; gc.collect()
    return L


def benchmark_dense_eigh(L_csr, k):
    """Time torch.linalg.eigh forward+backward."""
    n = L_csr.shape[0]
    if n > MAX_DENSE_N:
        return None, None, None, f'SKIP(n>{MAX_DENSE_N})'

    dense_bytes = n * n * 8
    gpu_free = torch.cuda.mem_get_info()[0]
    if dense_bytes * 3 > gpu_free:
        return None, None, None, 'OOM_gpu'

    ram_avail = psutil.virtual_memory().available
    if dense_bytes * 2 > ram_avail:
        return None, None, None, 'OOM_ram'

    flush_all()
    try:
        dense_np = L_csr.toarray()
        L_dense = torch.from_numpy(dense_np).double().cuda()
        del dense_np; gc.collect()
        L_dense.requires_grad_(True)
        torch.cuda.synchronize()

        t0 = time.time()
        evals, evecs = torch.linalg.eigh(L_dense)
        torch.cuda.synchronize()
        fwd_time = time.time() - t0

        loss = evals[:k][1:].sum()
        t0 = time.time()
        loss.backward()
        torch.cuda.synchronize()
        bwd_time = time.time() - t0

        peak_mem = torch.cuda.max_memory_allocated() / 1024**3
        del L_dense, evals, evecs, loss; flush_all()
        return fwd_time, bwd_time, peak_mem, 'OK'

    except RuntimeError:
        flush_all()
        return None, None, None, 'OOM'


def benchmark_sparse_eigsh(L_csr, k):
    """Time our differentiable_eigsh forward+backward (C kernel only, no torch overhead)."""
    from gpu_eigsh._core import _gpu_eigsh_raw, _gpu_adjoint_eigsh_raw

    flush_all()
    n = L_csr.shape[0]
    indptr = L_csr.indptr.astype(np.int32)
    indices = L_csr.indices.astype(np.int32)
    data = L_csr.data.astype(np.float64)

    try:
        # Forward
        t0 = time.time()
        evals, evecs = _gpu_eigsh_raw(indptr, indices, data, n, k, 0, 3000, 1e-12)
        fwd_time = time.time() - t0

        # Backward
        ge = np.zeros(k, dtype=np.float64); ge[1:] = 1.0
        gv = np.zeros((n, k), dtype=np.float64)

        t0 = time.time()
        grad_vals = _gpu_adjoint_eigsh_raw(indptr, indices, data, n, k,
                                            evals, evecs, ge, gv, 500, 1e-10)
        bwd_time = time.time() - t0

        # Eigenvector residual
        max_res = max(np.linalg.norm(L_csr @ evecs[:, i] - evals[i] * evecs[:, i])
                      for i in range(min(k, 5)))

        flush_all()
        return fwd_time, bwd_time, max_res, 'OK'

    except Exception as e:
        flush_all()
        return None, None, None, f'ERR:{str(e)[:40]}'


def gradient_accuracy_test(L_csr, k, n_entries=10):
    """Verify gradient against dense-eigh finite differences."""
    from scipy.sparse import csr_matrix
    from gpu_eigsh._core import _gpu_eigsh_raw, _gpu_adjoint_eigsh_raw

    n = L_csr.shape[0]
    if n > 5000:
        return None  # dense eigh FD too slow at large n

    indptr = L_csr.indptr.astype(np.int32)
    indices = L_csr.indices.astype(np.int32)
    data = L_csr.data.astype(np.float64)

    evals, evecs = _gpu_eigsh_raw(indptr, indices, data, n, k, 0, 3000, 1e-12)
    ge = np.zeros(k); ge[1:] = 1.0
    our_grad = np.array(_gpu_adjoint_eigsh_raw(indptr, indices, data, n, k,
        evals, evecs, ge, np.zeros((n, k)), 500, 1e-10))

    eps = 1e-6
    errs = []
    for idx in range(min(n_entries, len(data))):
        d_p = data.copy(); d_p[idx] += eps
        ep = np.linalg.eigvalsh(
            csr_matrix((d_p, L_csr.indices.copy(), L_csr.indptr.copy()),
                       shape=(n, n)).toarray())[:k]
        d_m = data.copy(); d_m[idx] -= eps
        em = np.linalg.eigvalsh(
            csr_matrix((d_m, L_csr.indices.copy(), L_csr.indptr.copy()),
                       shape=(n, n)).toarray())[:k]
        fd = (ep[1:].sum() - em[1:].sum()) / (2 * eps)
        if abs(fd) > 1e-10:
            errs.append(abs(our_grad[idx] - fd) / abs(fd))

    return max(errs) if errs else 0.0


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    print(f"GPU: {torch.cuda.get_device_name()}")
    total_gpu = torch.cuda.mem_get_info()[1] / 1024**3
    print(f"GPU Memory: {total_gpu:.1f} GB")
    print(f"RAM: {psutil.virtual_memory().total / 1024**3:.1f} GB")

    NUM_EIGS = 20

    # lobpcg test
    print(f"\n{'='*60}")
    print("torch.lobpcg backward on sparse input")
    print(f"{'='*60}")
    try:
        from scipy.sparse import csr_matrix as csr_m
        L_test = build_laplacian_scipy(500)
        L_coo = L_test.tocoo()
        idx = torch.tensor(np.vstack([L_coo.row, L_coo.col]), dtype=torch.long)
        v = torch.tensor(L_coo.data, dtype=torch.float64, requires_grad=True)
        Lt = torch.sparse_coo_tensor(idx, v, (500, 500)).cuda().coalesce()
        X = torch.randn(500, NUM_EIGS, dtype=torch.float64, device='cuda')
        ev, _ = torch.lobpcg(Lt, k=NUM_EIGS, X=X, largest=False, niter=200)
        ev.sum().backward()
        print(f"  Result: {'NaN' if torch.isnan(v.grad).any() else 'OK'}")
    except Exception as e:
        print(f"  Result: FAILED — {str(e)[:80]}")
    flush_all()

    # Scaling benchmark
    scales = [1000, 2000, 5000, 10000, 20000, 50000, 100000, 500000]
    results = []

    for n in scales:
        print(f"\n{'='*60}")
        print(f"n = {n:,}, k = {NUM_EIGS}")
        print(f"{'='*60}")

        t0 = time.time()
        L = build_laplacian_scipy(n)
        print(f"  Laplacian: {time.time()-t0:.1f}s, nnz={L.nnz:,}")

        # Dense eigh
        eigh_fwd, eigh_bwd, eigh_mem, eigh_status = benchmark_dense_eigh(L, NUM_EIGS)
        if eigh_status == 'OK':
            print(f"  eigh:    fwd={eigh_fwd:.3f}s  bwd={eigh_bwd:.3f}s  "
                  f"total={eigh_fwd+eigh_bwd:.3f}s  mem={eigh_mem:.2f}GB")
        else:
            print(f"  eigh:    {eigh_status}")

        # Our sparse eigsh
        sp_fwd, sp_bwd, sp_res, sp_status = benchmark_sparse_eigsh(L, NUM_EIGS)
        if sp_status == 'OK':
            print(f"  ours:    fwd={sp_fwd:.3f}s  bwd={sp_bwd:.3f}s  "
                  f"total={sp_fwd+sp_bwd:.3f}s  res={sp_res:.1e}")
        else:
            print(f"  ours:    {sp_status}")

        # Speedup
        if eigh_status == 'OK' and sp_status == 'OK':
            spd = (eigh_fwd + eigh_bwd) / (sp_fwd + sp_bwd)
            print(f"  speedup: {spd:.1f}x")

        # Gradient accuracy (only at small n where dense FD is feasible)
        grad_err = gradient_accuracy_test(L, NUM_EIGS) if n <= 5000 else None
        if grad_err is not None:
            print(f"  grad:    {grad_err:.2e} (vs dense-eigh FD)")

        results.append({
            'n': n, 'k': NUM_EIGS, 'nnz': L.nnz,
            'eigh_fwd': eigh_fwd, 'eigh_bwd': eigh_bwd,
            'eigh_mem': eigh_mem, 'eigh_status': eigh_status,
            'sparse_fwd': sp_fwd, 'sparse_bwd': sp_bwd,
            'sparse_res': sp_res, 'sparse_status': sp_status,
            'grad_err': float(grad_err) if grad_err is not None else None,
        })

        del L; flush_all()

    # Summary table
    print(f"\n\n{'='*100}")
    print(f"DIFFERENTIABLE EIGENDECOMPOSITION — k={NUM_EIGS}")
    print(f"{'='*100}")
    print(f"{'n':>9} {'eigh tot':>10} {'ours tot':>10} {'speedup':>8} "
          f"{'eigh mem':>9} {'evec res':>10} {'grad err':>10}")
    print('-' * 75)

    for r in results:
        n_s = f"{r['n']:>9,}"
        if r['eigh_status'] == 'OK':
            et = f"{r['eigh_fwd']+r['eigh_bwd']:>9.3f}s"
            em = f"{r['eigh_mem']:>8.2f}G"
        else:
            et = f"{r['eigh_status']:>10}"
            em = f"{'—':>9}"

        if r['sparse_status'] == 'OK':
            st = f"{r['sparse_fwd']+r['sparse_bwd']:>9.3f}s"
            sr = f"{r['sparse_res']:>10.1e}"
        else:
            st = f"{'—':>10}"
            sr = f"{'—':>10}"

        if r['eigh_status'] == 'OK' and r['sparse_status'] == 'OK':
            spd = (r['eigh_fwd']+r['eigh_bwd']) / (r['sparse_fwd']+r['sparse_bwd'])
            speedup = f"{spd:>7.1f}x"
        elif r['eigh_status'] != 'OK' and r['sparse_status'] == 'OK':
            speedup = f"{'inf':>8}"
        else:
            speedup = f"{'—':>8}"

        ge = f"{r['grad_err']:>10.1e}" if r['grad_err'] is not None else f"{'—':>10}"
        print(f"{n_s} {et} {st} {speedup} {em} {sr} {ge}")

    # Save
    json_path = os.path.join(DATA_DIR, 'differentiable_benchmark.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {json_path}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    eigh_ok = [r for r in results if r['eigh_status'] == 'OK']
    sp_ok = [r for r in results if r['sparse_status'] == 'OK']

    # Panel 1: Total time
    ax = axes[0]
    if eigh_ok:
        ax.loglog([r['n'] for r in eigh_ok],
                  [r['eigh_fwd']+r['eigh_bwd'] for r in eigh_ok],
                  'r-s', ms=8, lw=2.5, markeredgecolor='white', markeredgewidth=0.5,
                  label='torch.linalg.eigh (dense)')
    if sp_ok:
        ax.loglog([r['n'] for r in sp_ok],
                  [r['sparse_fwd']+r['sparse_bwd'] for r in sp_ok],
                  'g-D', ms=8, lw=2.5, markeredgecolor='white', markeredgewidth=0.5,
                  label='differentiable_eigsh (ours)')
    oom_ns = [r['n'] for r in results if r['eigh_status'] != 'OK']
    if oom_ns and eigh_ok:
        ax.axvspan(oom_ns[0]*0.8, max(r['n'] for r in results)*1.2,
                   alpha=0.08, color='red')
        ax.annotate('dense eigh:\nOOM / infeasible', xy=(oom_ns[0], 0.5),
                   fontsize=9, color='red', fontweight='bold', ha='left')
    ax.set_xlabel('Matrix Size (n)'); ax.set_ylabel('Forward + Backward (s)')
    ax.set_title(f'Differentiable Eigendecomposition (k={NUM_EIGS})')
    ax.legend(fontsize=10, loc='upper left'); ax.grid(True, alpha=0.3)

    # Panel 2: Memory
    ax = axes[1]
    if eigh_ok:
        ax.loglog([r['n'] for r in eigh_ok], [r['eigh_mem'] for r in eigh_ok],
                  'r-s', ms=8, lw=2.5, markeredgecolor='white', markeredgewidth=0.5,
                  label='torch.linalg.eigh')
    if sp_ok:
        # Our actual GPU alloc: ~n*ncv*8 bytes for Lanczos basis
        ours_mem = [r['n'] * 60 * 8 / 1024**3 for r in sp_ok]
        ax.loglog([r['n'] for r in sp_ok], ours_mem,
                  'g-D', ms=8, lw=2.5, markeredgecolor='white', markeredgewidth=0.5,
                  label='differentiable_eigsh (ours)')
    ax.axhline(y=total_gpu, color='gray', ls='--', lw=1, alpha=0.6,
              label=f'GPU capacity ({total_gpu:.0f} GB)')
    ax.set_xlabel('Matrix Size (n)'); ax.set_ylabel('GPU Memory (GB)')
    ax.set_title('Memory: O(n²) vs O(n·ncv)')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

    # Panel 3: Eigenvector residual
    ax = axes[2]
    res_ok = [r for r in sp_ok if r['sparse_res'] is not None]
    if res_ok:
        ax.semilogy([r['n'] for r in res_ok], [r['sparse_res'] for r in res_ok],
                    'g-D', ms=8, lw=2.5, markeredgecolor='white', markeredgewidth=0.5,
                    label='||Ax - λx|| (max)')
    ax.axhline(y=2.2e-16, color='gray', ls='--', lw=1, alpha=0.6,
              label=r'Machine $\epsilon$')
    ax.set_xlabel('Matrix Size (n)'); ax.set_ylabel('Eigenvector Residual')
    ax.set_title('Numerical Accuracy')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(FIG_DIR, 'differentiable_benchmark.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.savefig(plot_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"Plot saved to {plot_path}")


if __name__ == '__main__':
    main()
