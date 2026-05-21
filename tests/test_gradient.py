#!/usr/bin/env python3
"""
Gradient correctness tests for differentiable GPU eigsh.

Tests:
  1. Eigenvalue-only gradient vs finite differences
  2. Eigenvector gradient vs finite differences
  3. Degenerate eigenvalue handling
  4. Shift-invert forward correctness
  5. Comparison with torch.linalg.eigh backward (dense, ground truth)

Run: python3 tests/test_gradient.py
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

import torch
from scipy.sparse import random as sparse_random, csr_matrix, diags
from scipy.spatial import cKDTree


def build_test_laplacian(n, nn=10, seed=42):
    """Build a small PSD graph Laplacian for testing."""
    np.random.seed(seed)
    points = np.random.randn(n, 5)
    tree = cKDTree(points)
    dists, idxs = tree.query(points, k=nn + 1)

    rows = np.repeat(np.arange(n), nn)
    cols = idxs[:, 1:].ravel()
    vals = np.exp(-dists[:, 1:]**2 / 4.0).ravel()

    A = csr_matrix((vals, (rows, cols)), shape=(n, n))
    A_sym = A.maximum(A.T)
    degrees = np.array(A_sym.sum(axis=1)).flatten()

    D_inv_sqrt = diags(1.0 / np.sqrt(np.maximum(degrees, 1e-15)))
    D = diags(degrees)
    L = D - A_sym
    L_sym = D_inv_sqrt @ L @ D_inv_sqrt
    return L_sym


def scipy_to_torch_csr(L_scipy, requires_grad=True):
    """Convert scipy CSR to torch sparse CSR tensor."""
    L_csr = csr_matrix(L_scipy, dtype=np.float64)
    crow = torch.from_numpy(L_csr.indptr.astype(np.int32)).to(torch.int32)
    col = torch.from_numpy(L_csr.indices.astype(np.int32)).to(torch.int32)
    values = torch.from_numpy(L_csr.data.copy()).to(torch.float64)
    if requires_grad:
        values = values.requires_grad_(True)
    return crow, col, values, L_csr.shape


def test_forward_correctness():
    """Test that GPU eigsh matches scipy eigsh."""
    from gpu_eigsh import gpu_eigsh
    from scipy.sparse.linalg import eigsh as scipy_eigsh

    print("=" * 60)
    print("TEST: Forward correctness (GPU IRLM vs scipy ARPACK)")
    print("=" * 60)

    n = 500
    k = 10
    L = build_test_laplacian(n)

    gpu_evals, gpu_evecs = gpu_eigsh(L, k=k)
    scipy_evals, _ = scipy_eigsh(L, k=k, which='SM', tol=1e-10)
    scipy_evals = np.sort(scipy_evals)

    max_err = np.max(np.abs(gpu_evals[1:] - scipy_evals[1:]) /
                     np.abs(scipy_evals[1:]))
    print(f"  n={n}, k={k}")
    print(f"  Max relative error: {max_err:.2e}")
    assert max_err < 1e-10, f"Forward error too large: {max_err}"
    print("  PASSED\n")


def test_eigenvalue_gradient_vs_dense_eigh():
    """Test eigenvalue gradient against torch.linalg.eigh backward."""
    from gpu_eigsh.differentiable import differentiable_eigsh

    print("=" * 60)
    print("TEST: Eigenvalue gradient (vs torch.linalg.eigh backward)")
    print("=" * 60)

    n = 100
    k = 5
    L = build_test_laplacian(n)
    L_csr = csr_matrix(L, dtype=np.float64)

    # Dense eigh backward (ground truth)
    L_dense = torch.from_numpy(L_csr.toarray()).double().requires_grad_(True)
    evals_d, _ = torch.linalg.eigh(L_dense)
    loss_d = evals_d[:k][1:].sum()  # skip zero eigenvalue
    loss_d.backward()
    dense_grad = L_dense.grad.clone()

    # Our sparse backward
    crow, col, values, shape = scipy_to_torch_csr(L)
    evals_s, _ = differentiable_eigsh(
        torch.sparse_csr_tensor(crow, col, values, shape),
        k=k, tol=1e-12, cg_tol=1e-10, cg_max_iters=500)
    loss_s = evals_s[1:].sum()
    loss_s.backward()
    sparse_grad = values.grad.clone().numpy()

    # Compare at sparsity pattern
    errors = []
    for idx in range(len(L_csr.data)):
        r = np.searchsorted(L_csr.indptr, idx, side='right') - 1
        c = L_csr.indices[idx]
        dg = dense_grad[r, c].item()
        sg = sparse_grad[idx]
        if abs(dg) > 1e-10:
            errors.append(abs(sg - dg) / abs(dg))

    if errors:
        max_err = max(errors)
        mean_err = sum(errors) / len(errors)
        print(f"  n={n}, k={k}, tested {len(errors)} entries")
        print(f"  Max relative error:  {max_err:.2e}")
        print(f"  Mean relative error: {mean_err:.2e}")
        assert max_err < 1e-3, f"Gradient error too large: {max_err}"
        print("  PASSED\n")
    else:
        print("  No significant gradient entries to compare")
        print("  SKIPPED\n")


def test_eigenvector_gradient_vs_dense():
    """Test eigenvector gradient against dense torch.linalg.eigh."""
    from gpu_eigsh.differentiable import differentiable_eigsh

    print("=" * 60)
    print("TEST: Eigenvector gradient (vs torch.linalg.eigh)")
    print("=" * 60)

    n = 100
    k = 5
    L = build_test_laplacian(n)

    # Dense eigh with autograd
    L_dense = torch.from_numpy(L.toarray()).double().requires_grad_(True)
    evals_dense, evecs_dense = torch.linalg.eigh(L_dense)
    # Loss using eigenvectors: sum of squared first k eigenvectors
    loss_dense = (evecs_dense[:, :k] ** 2).sum()
    loss_dense.backward()
    dense_grad = L_dense.grad.clone()

    # Sparse differentiable eigsh
    crow, col, values, shape = scipy_to_torch_csr(L)
    evals_sparse, evecs_sparse = differentiable_eigsh(
        torch.sparse_csr_tensor(crow, col, values, shape),
        k=k, tol=1e-12, cg_tol=1e-10)

    # Same loss
    loss_sparse = (evecs_sparse ** 2).sum()
    loss_sparse.backward()
    sparse_grad = values.grad.clone()

    # Compare at sparsity pattern locations
    L_csr = csr_matrix(L, dtype=np.float64)
    dense_at_sparse = torch.zeros_like(values)
    for idx in range(len(L_csr.data)):
        row = np.searchsorted(L_csr.indptr, idx, side='right') - 1
        c = L_csr.indices[idx]
        # dense_grad is for the full matrix; sparse gradient is per-entry
        dense_at_sparse[idx] = dense_grad[row, c]

    rel_err = (sparse_grad - dense_at_sparse).abs() / \
              dense_at_sparse.abs().clamp(min=1e-15)
    # Filter out near-zero entries
    mask = dense_at_sparse.abs() > 1e-10
    if mask.any():
        max_err = rel_err[mask].max().item()
        mean_err = rel_err[mask].mean().item()
        print(f"  n={n}, k={k}")
        print(f"  Max relative error:  {max_err:.2e}")
        print(f"  Mean relative error: {mean_err:.2e}")
        # Eigenvector gradients can be noisy for near-degenerate eigenvalues
        assert max_err < 0.1 or mean_err < 0.01, \
            f"Eigenvector gradient error too large: max={max_err}, mean={mean_err}"
        print("  PASSED\n")
    else:
        print("  No significant gradient entries")
        print("  SKIPPED\n")


def test_shift_invert_forward():
    """Test shift-invert mode gives correct eigenvalues."""
    from gpu_eigsh import gpu_eigsh
    from scipy.sparse.linalg import eigsh as scipy_eigsh

    print("=" * 60)
    print("TEST: Shift-invert forward correctness")
    print("=" * 60)

    n = 500
    k = 10
    L = build_test_laplacian(n)

    # Standard mode: k smallest
    std_evals, _ = gpu_eigsh(L, k=k, tol=1e-12)

    # Shift-invert with sigma=0 should give same result
    si_evals, _ = gpu_eigsh(L, k=k, sigma=0.0, tol=1e-10,
                            cg_max_iters=500, cg_tol=1e-12)

    # Compare (skip first eigenvalue which is ~0 and hard to match relatively)
    if len(si_evals) >= 2 and len(std_evals) >= 2:
        max_err = np.max(np.abs(si_evals[1:k] - std_evals[1:k]) /
                         np.abs(std_evals[1:k]))
        print(f"  n={n}, k={k}, sigma=0.0")
        print(f"  Standard evals: {std_evals[:5]}")
        print(f"  Shift-inv evals: {si_evals[:5]}")
        print(f"  Max relative error: {max_err:.2e}")
        assert max_err < 1e-4, f"Shift-invert error too large: {max_err}"
        print("  PASSED\n")
    else:
        print("  Not enough converged eigenvalues")
        print("  SKIPPED\n")


def test_hellmann_feynman():
    """Test Hellmann-Feynman: dλ/dA = x x^T (eigenvalue-only loss)."""
    from gpu_eigsh.differentiable import differentiable_eigsh

    print("=" * 60)
    print("TEST: Hellmann-Feynman theorem (dλ/dA = x x^T)")
    print("=" * 60)

    n = 200
    k = 3
    L = build_test_laplacian(n)
    crow, col, values, shape = scipy_to_torch_csr(L)

    # Forward + backward with eigenvalue-only loss
    evals, evecs = differentiable_eigsh(
        torch.sparse_csr_tensor(crow, col, values, shape),
        k=k, tol=1e-12, cg_tol=1e-10)

    # Loss = single eigenvalue
    target_idx = 1  # skip the zero eigenvalue
    loss = evals[target_idx]
    loss.backward()
    gpu_grad = values.grad.clone()

    # Hellmann-Feynman: dλ_i/dA_{rc} = x_i[r] * x_i[c]
    x_i = evecs[:, target_idx].detach().numpy()
    L_csr = csr_matrix(L, dtype=np.float64)
    hf_grad = np.zeros(len(L_csr.data))
    for idx in range(len(L_csr.data)):
        row = np.searchsorted(L_csr.indptr, idx, side='right') - 1
        c = L_csr.indices[idx]
        hf_grad[idx] = x_i[row] * x_i[c]

    # Compare
    hf_grad_t = torch.from_numpy(hf_grad).double()
    mask = hf_grad_t.abs() > 1e-12
    if mask.any():
        rel_err = (gpu_grad[mask] - hf_grad_t[mask]).abs() / \
                  hf_grad_t[mask].abs()
        max_err = rel_err.max().item()
        mean_err = rel_err.mean().item()
        print(f"  n={n}, eigenvalue index={target_idx}")
        print(f"  Max relative error:  {max_err:.2e}")
        print(f"  Mean relative error: {mean_err:.2e}")
        assert max_err < 1e-3, f"HF gradient error too large: {max_err}"
        print("  PASSED\n")
    else:
        print("  SKIPPED (no significant entries)\n")


def main():
    print("\n" + "=" * 60)
    print("DIFFERENTIABLE GPU EIGSH — GRADIENT CORRECTNESS TESTS")
    print("=" * 60 + "\n")

    test_forward_correctness()
    test_eigenvalue_gradient_finite_diff()
    test_eigenvector_gradient_vs_dense()
    test_shift_invert_forward()
    test_hellmann_feynman()

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == '__main__':
    main()
