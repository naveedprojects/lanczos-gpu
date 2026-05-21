"""
Differentiable sparse eigendecomposition on GPU.

Wraps the GPU IRLM eigensolver as a torch.autograd.Function, enabling
gradient flow through eigenvalues and eigenvectors back to the matrix
entries (or any upstream parameters).

The backward pass uses implicit differentiation (Xie et al. 2020):
  For each eigenpair (lambda_i, x_i), solve the adjoint system
    (A - lambda_i I) xi_i = (I - X X^T) grad_evecs_i
  via deflated CG on GPU. The gradient w.r.t. A is:
    A_bar = sum_i (grad_evals_i * x_i - xi_i) x_i^T

This avoids backpropagating through the Lanczos iterations entirely,
giving exact gradients independent of iteration count.

Usage:
    from gpu_eigsh import differentiable_eigsh

    # A_values requires grad
    A = torch.sparse_csr_tensor(crow, col, values.requires_grad_(True), (n, n))
    eigenvalues, eigenvectors = differentiable_eigsh(A, k=20)
    loss = eigenvalues.sum()
    loss.backward()
    # values.grad now contains d(loss)/d(A_values)
"""

import torch
import numpy as np
from scipy.sparse import csr_matrix

from ._core import _gpu_eigsh_raw, _gpu_eigsh_sigma_raw, _gpu_adjoint_eigsh_raw


class _DifferentiableEigsh(torch.autograd.Function):
    """torch.autograd.Function for GPU sparse eigendecomposition.

    Forward: GPU IRLM (or shift-invert IRLM)
    Backward: Implicit differentiation via deflated CG
    """

    @staticmethod
    def forward(ctx, values, crow_indices, col_indices, shape,
                k, ncv, max_iters, tol, sigma,
                cg_max_iters, cg_tol):
        n = shape[0]

        # Convert to numpy for the C binding
        indptr = crow_indices.detach().cpu().numpy().astype(np.int32)
        indices = col_indices.detach().cpu().numpy().astype(np.int32)
        data = values.detach().cpu().to(torch.float64).numpy()

        if sigma is None:
            evals_np, evecs_np = _gpu_eigsh_raw(
                indptr, indices, data, n, k, ncv, max_iters, tol)
        else:
            evals_np, evecs_np = _gpu_eigsh_sigma_raw(
                indptr, indices, data, n, k, ncv, max_iters, tol,
                float(sigma), cg_max_iters, cg_tol)

        # Convert back to torch tensors on the same device as input
        device = values.device
        eigenvalues = torch.from_numpy(evals_np.copy()).to(
            dtype=torch.float64, device=device)
        eigenvectors = torch.from_numpy(evecs_np.copy()).to(
            dtype=torch.float64, device=device)

        # Save for backward
        ctx.save_for_backward(eigenvalues, eigenvectors, values,
                              crow_indices, col_indices)
        ctx.n = n
        ctx.k = len(evals_np)
        ctx.cg_max_iters = cg_max_iters
        ctx.cg_tol = cg_tol

        return eigenvalues, eigenvectors

    @staticmethod
    def backward(ctx, grad_evals, grad_evecs):
        eigenvalues, eigenvectors, values, crow_indices, col_indices = \
            ctx.saved_tensors
        n = ctx.n
        k = ctx.k

        # Convert everything to numpy for the C backward kernel
        indptr = crow_indices.detach().cpu().numpy().astype(np.int32)
        indices = col_indices.detach().cpu().numpy().astype(np.int32)
        data = values.detach().cpu().to(torch.float64).numpy()

        evals_np = eigenvalues.detach().cpu().to(torch.float64).numpy()

        # Eigenvectors (n x k), C-contiguous — the C code handles the transpose
        evecs_np = np.ascontiguousarray(
            eigenvectors.detach().cpu().to(torch.float64).numpy())

        ge_np = grad_evals.detach().cpu().to(torch.float64).numpy() \
            if grad_evals is not None else np.zeros(k, dtype=np.float64)

        gv_np = np.ascontiguousarray(
            grad_evecs.detach().cpu().to(torch.float64).numpy()) \
            if grad_evecs is not None else np.zeros((n, k), dtype=np.float64)

        # Call the CUDA backward kernel
        grad_vals_np = _gpu_adjoint_eigsh_raw(
            indptr, indices, data, n, k,
            evals_np, evecs_np, ge_np, gv_np,
            ctx.cg_max_iters, ctx.cg_tol)

        # Convert back to torch
        grad_values = torch.from_numpy(grad_vals_np.copy()).to(
            dtype=values.dtype, device=values.device)

        # Return gradients: values, crow_indices, col_indices, shape,
        # k, ncv, max_iters, tol, sigma, cg_max_iters, cg_tol
        return grad_values, None, None, None, None, None, None, None, \
               None, None, None


def differentiable_eigsh(A, k=20, ncv=0, maxiter=3000, tol=1e-12,
                         sigma=None, cg_max_iters=500, cg_tol=1e-10):
    """Differentiable GPU sparse eigendecomposition.

    Computes k eigenvalues and eigenvectors of a symmetric sparse matrix,
    with gradients flowing back to the matrix values via autograd.

    Parameters
    ----------
    A : torch.Tensor (sparse CSR) or scipy.sparse matrix
        Symmetric sparse matrix. If torch tensor, the values must have
        requires_grad=True for gradient computation.
    k : int
        Number of eigenvalues to compute.
    ncv : int
        Krylov subspace size. Default 0 = auto: max(10*k, 100).
        For accurate gradients, ncv should be large enough to produce
        eigenvectors with small residuals. The default 3*k from standard
        eigsh is sufficient for eigenvalue convergence but may not give
        accurate eigenvectors (and thus gradients). Use ncv >= 10*k for
        reliable backward pass.
    maxiter : int
        Maximum Lanczos iterations.
    tol : float
        Convergence tolerance.
    sigma : float or None
        If specified, find k eigenvalues nearest to sigma (shift-invert).
        If None, find k smallest eigenvalues.
    cg_max_iters : int
        Maximum CG iterations for backward pass (and shift-invert forward).
    cg_tol : float
        CG convergence tolerance.

    Returns
    -------
    eigenvalues : torch.Tensor (k,)
        Eigenvalues in ascending order.
    eigenvectors : torch.Tensor (n, k)
        Corresponding eigenvectors as columns.

    Notes
    -----
    The backward pass uses implicit differentiation, NOT backpropagation
    through the Lanczos iterations. This means:
    - Memory: O(n*k), independent of iteration count
    - Stability: exact gradients (to CG solver precision)
    - Cost: k CG solves, typically 2-5x forward pass

    For loss functions that depend only on eigenvalues (not eigenvectors),
    the backward is even cheaper since the CG solve is trivial (the
    Hellmann-Feynman theorem gives dλ/dA = x x^T directly).
    """
    if isinstance(A, torch.Tensor):
        if A.layout == torch.sparse_csr:
            crow = A.crow_indices().to(torch.int32)
            col = A.col_indices().to(torch.int32)
            values = A.values().to(torch.float64)
            shape = A.shape
        elif A.is_sparse:
            # COO -> CSR
            A_csr = A.to_sparse_csr()
            crow = A_csr.crow_indices().to(torch.int32)
            col = A_csr.col_indices().to(torch.int32)
            values = A_csr.values().to(torch.float64)
            shape = A_csr.shape
        else:
            raise TypeError("A must be a sparse torch tensor (CSR or COO)")

        # Use ncv = 3*k (not the C default max(3*k, 40)) to avoid
        # problematic ncv values that produce inaccurate eigenvectors
        # for certain spectral structures.
        if ncv == 0:
            ncv = min(3 * k, shape[0] - 1)

        return _DifferentiableEigsh.apply(
            values, crow, col, shape,
            k, ncv, maxiter, tol, sigma,
            cg_max_iters, cg_tol)

    else:
        # scipy sparse — no autograd, fall back to non-differentiable
        from . import gpu_eigsh
        evals, evecs = gpu_eigsh(A, k=k, ncv=ncv, maxiter=maxiter, tol=tol)
        return torch.from_numpy(evals), torch.from_numpy(evecs)
