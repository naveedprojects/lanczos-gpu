#!/usr/bin/env python3
"""
Forward correctness tests for the Lanczos-based matrix function primitive
(Layer 1 of the differentiable spectral toolkit).

Tests:
  1. funm_apply(A, v, m, f)  vs  U·diag(f(λ))·U^T·v   (dense ground truth)
  2. funm_qform(A, v, m, f)  vs  v^T · U·diag(f(λ))·U^T · v

For each f in {log, exp, sqrt, inv} and several (n, m) settings.

Run: /usr/bin/python3 tests/test_funm.py
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import torch
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix, diags

from gpu_eigsh import funm_apply, funm_qform


def build_spd_laplacian(n, eps_shift=1e-2, nn=10, seed=42):
    """Build a graph Laplacian + eps*I — small, sparse, SPD for log/sqrt/inv."""
    rng = np.random.default_rng(seed)
    pts = rng.standard_normal((n, 5))
    tree = cKDTree(pts)
    dists, idxs = tree.query(pts, k=nn + 1)
    nd = dists[:, 1:]
    ni = idxs[:, 1:]
    w = np.exp(-(nd ** 2) / 4.0)
    rows = np.repeat(np.arange(n), nn)
    cols = ni.ravel()
    vals = w.ravel()
    A = csr_matrix((vals, (rows, cols)), shape=(n, n))
    A_sym = A.maximum(A.T)
    deg = np.array(A_sym.sum(axis=1)).flatten()
    L = csr_matrix((deg, (np.arange(n), np.arange(n))), shape=(n, n)) - A_sym
    L_shifted = L + diags([eps_shift] * n, 0, shape=(n, n), format="csr")
    return L_shifted.tocsr()


def dense_funm(A_csr, func):
    """Ground truth f(A) for dense A via eigh."""
    A = torch.from_numpy(A_csr.toarray()).double()
    w, U = torch.linalg.eigh(A)
    if func == "log":
        fw = torch.log(w)
    elif func == "exp":
        fw = torch.exp(w)
    elif func == "sqrt":
        fw = torch.sqrt(w)
    elif func == "inv":
        fw = 1.0 / w
    else:
        raise ValueError(func)
    return (U * fw.unsqueeze(0)) @ U.T


def run_case(n, m, func, eps_shift=1e-2, seed=0):
    """Run one (n, m, func) test, return (rel_err_apply, rel_err_qform, m_actual)."""
    L = build_spd_laplacian(n, eps_shift=eps_shift, seed=seed + 7)
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(n)

    # GPU primitive
    y, m_a = funm_apply(L, v, m=m, func=func)
    q, _ = funm_qform(L, v, m=m, func=func)

    # Dense ground truth
    fA = dense_funm(L, func)
    y_true = (fA @ torch.from_numpy(v).double()).numpy()
    q_true = float(torch.from_numpy(v).double() @ fA @ torch.from_numpy(v).double())

    err_apply = np.linalg.norm(y - y_true) / np.linalg.norm(y_true)
    err_qform = abs(q - q_true) / abs(q_true) if q_true != 0 else abs(q - q_true)

    return err_apply, err_qform, m_a


def main():
    cases = [
        # (n, m, func, expected_err_threshold)
        # Near-exact: m close to n
        (100, 80, "log",  1e-8),
        (100, 80, "exp",  1e-8),
        (100, 80, "sqrt", 1e-8),
        (100, 80, "inv",  1e-6),  # inv is harder near small eigenvalues
        # Lanczos approximation: m << n
        (500, 50, "log",  1e-3),
        (500, 50, "exp",  1e-3),
        (500, 50, "sqrt", 1e-3),
        (500, 50, "inv",  1e-1),
        (1000, 80, "log",  1e-4),
        (1000, 80, "exp",  1e-4),
        (1000, 80, "sqrt", 1e-4),
        (1000, 80, "inv",  1e-1),
    ]

    print(f"{'n':>6} {'m':>4} {'func':>6} {'m_act':>6} "
          f"{'apply err':>12} {'qform err':>12} {'threshold':>12} {'pass':>6}")
    print("-" * 78)

    all_pass = True
    for (n, m, func, thr) in cases:
        try:
            ea, eq, ma = run_case(n, m, func)
        except Exception as e:
            print(f"{n:>6} {m:>4} {func:>6} ERROR: {e}")
            all_pass = False
            continue
        ok_a = ea < thr
        ok_q = eq < thr
        status = "PASS" if (ok_a and ok_q) else "FAIL"
        if not (ok_a and ok_q):
            all_pass = False
        print(f"{n:>6} {m:>4} {func:>6} {ma:>6} "
              f"{ea:>12.2e} {eq:>12.2e} {thr:>12.0e} {status:>6}")

    print("-" * 78)
    print(("ALL PASS" if all_pass else "SOME FAILED") + ".")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
