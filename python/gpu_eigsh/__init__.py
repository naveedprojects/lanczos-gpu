"""
gpu_eigsh — GPU-accelerated sparse eigenvalue solver.

Drop-in replacement for scipy.sparse.linalg.eigsh, backed by an
ARPACK-faithful IRLM implementation on CUDA.

Features:
    - Forward: GPU IRLM with DGKS reorthogonalization, matching ARPACK
      to machine precision at up to 53x speedup
    - Shift-invert: eigenvalues nearest to a target sigma via CG-based
      spectral transformation
    - Differentiable: torch.autograd.Function wrapper with implicit
      differentiation backward pass (no unrolling through iterations)
    - Linear operator: accepts function pointers for matrix-free
      eigenvalue computation

Usage:
    from gpu_eigsh import gpu_eigsh
    eigenvalues, eigenvectors = gpu_eigsh(L, k=20)

    # Shift-invert (eigenvalues nearest to sigma=1.5)
    eigenvalues, eigenvectors = gpu_eigsh(L, k=20, sigma=1.5)

    # Differentiable (PyTorch autograd)
    from gpu_eigsh import differentiable_eigsh
    evals, evecs = differentiable_eigsh(L_sparse_csr, k=20)
    loss = evals.sum()
    loss.backward()
"""

import os
import ctypes

# Force single-threaded MKL/OpenMP before the CUDA + LAPACK extension is
# loaded. libmkl_rt's multi-threaded dstev races silently when the process
# has also initialized CUDA: eigenvectors come back corrupt while
# eigenvalues stay correct.  Env var alone isn't enough if numpy/scipy
# loaded MKL multi-threaded already, so we also call mkl_set_num_threads
# directly via ctypes.
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")


def _force_single_threaded_blas():
    for libname in ("libmkl_rt.so.2", "libmkl_rt.so.1", "libmkl_rt.so",
                    "libmkl_rt.dylib"):
        try:
            mkl = ctypes.CDLL(libname, mode=ctypes.RTLD_GLOBAL)
            for fn in ("MKL_Set_Num_Threads", "mkl_set_num_threads"):
                if hasattr(mkl, fn):
                    f = getattr(mkl, fn)
                    f.argtypes = [ctypes.c_int]
                    f(1)
                    return
        except OSError:
            continue


_force_single_threaded_blas()


import numpy as np
from scipy.sparse import issparse, csr_matrix

from ._core import (
    _gpu_eigsh_raw,
    _gpu_eigsh_sigma_raw,
    _gpu_funm_apply_raw,
    _gpu_funm_qform_raw,
)


def funm_apply(A, v, m=50, func="log"):
    """Apply a matrix function to a vector: y ≈ f(A) v via Lanczos quadrature.

    Parameters
    ----------
    A : scipy sparse matrix (n x n)
        Symmetric matrix. Will be converted to CSR if needed.
    v : ndarray (n,)
        Input vector (need not be unit-norm).
    m : int
        Lanczos depth — number of iterations to run. Determines
        approximation accuracy. Typical: 20-50 for log/exp/sqrt.
    func : str
        Scalar function name: 'log', 'exp', 'sqrt', 'inv'.

    Returns
    -------
    y : ndarray (n,)
        Approximation to f(A) v.
    m_actual : int
        Actual Lanczos depth used (may be < m on breakdown).
    """
    if not issparse(A):
        raise TypeError("A must be a scipy sparse matrix")
    A_csr = csr_matrix(A, dtype=np.float64)
    n = A_csr.shape[0]
    if A_csr.shape[0] != A_csr.shape[1]:
        raise ValueError(f"Matrix must be square, got {A_csr.shape}")

    indptr = np.ascontiguousarray(A_csr.indptr, dtype=np.int32)
    indices = np.ascontiguousarray(A_csr.indices, dtype=np.int32)
    data = np.ascontiguousarray(A_csr.data, dtype=np.float64)
    vv = np.ascontiguousarray(v, dtype=np.float64).reshape(-1)
    if vv.shape[0] != n:
        raise ValueError(f"v must have length {n}, got {vv.shape[0]}")

    y, rc = _gpu_funm_apply_raw(indptr, indices, data, n, vv, int(m), func)
    if rc < 0:
        msg = {-1: "unknown func", -2: "invalid eigenvalue for func",
               -3: "zero input vector", -4: "tridiag eigensolver failure"}
        raise RuntimeError(f"funm_apply failed: {msg.get(rc, rc)}")
    return y, rc


def funm_qform(A, v, m=50, func="log"):
    """Quadratic form v^T f(A) v via m-step Lanczos quadrature.

    Same arguments as `funm_apply`, returns (scalar, m_actual).
    """
    if not issparse(A):
        raise TypeError("A must be a scipy sparse matrix")
    A_csr = csr_matrix(A, dtype=np.float64)
    n = A_csr.shape[0]
    indptr = np.ascontiguousarray(A_csr.indptr, dtype=np.int32)
    indices = np.ascontiguousarray(A_csr.indices, dtype=np.int32)
    data = np.ascontiguousarray(A_csr.data, dtype=np.float64)
    vv = np.ascontiguousarray(v, dtype=np.float64).reshape(-1)
    if vv.shape[0] != n:
        raise ValueError(f"v must have length {n}, got {vv.shape[0]}")

    val, rc = _gpu_funm_qform_raw(indptr, indices, data, n, vv, int(m), func)
    if rc < 0:
        msg = {-1: "unknown func", -2: "invalid eigenvalue for func",
               -3: "zero input vector", -4: "tridiag eigensolver failure"}
        raise RuntimeError(f"funm_qform failed: {msg.get(rc, rc)}")
    return float(val), rc


def gpu_eigsh(A, k=20, ncv=0, maxiter=2000, tol=1e-12, which='SM',
              sigma=None, cg_max_iters=500, cg_tol=1e-12):
    """Compute k eigenvalues of a sparse symmetric matrix on GPU.

    Parameters
    ----------
    A : scipy sparse matrix (n x n)
        Symmetric matrix. Will be converted to CSR if needed.
    k : int
        Number of eigenvalues to compute.
    ncv : int
        Krylov subspace size (0 = auto: 3*k).
    maxiter : int
        Maximum Lanczos iterations across all restarts.
    tol : float
        Convergence tolerance for Ritz values.
    which : str
        Which eigenvalues to compute. 'SM' = smallest magnitude.
    sigma : float or None
        If specified, find k eigenvalues nearest to sigma using
        shift-invert mode. Overrides 'which'.
    cg_max_iters : int
        Max CG iterations for shift-invert inner solver.
    cg_tol : float
        CG tolerance for shift-invert inner solver.

    Returns
    -------
    eigenvalues : ndarray (k,)
        The k eigenvalues in ascending order.
    eigenvectors : ndarray (n, k)
        Corresponding eigenvectors as columns.
    """
    if which != 'SM' and sigma is None:
        raise NotImplementedError(f"which='{which}' not supported, use 'SM'")

    if not issparse(A):
        raise TypeError("A must be a scipy sparse matrix")

    A_csr = csr_matrix(A, dtype=np.float64)
    n = A_csr.shape[0]

    if A_csr.shape[0] != A_csr.shape[1]:
        raise ValueError(f"Matrix must be square, got {A_csr.shape}")
    if k < 1 or k > n:
        raise ValueError(f"k must be in [1, {n}], got {k}")

    indptr = np.ascontiguousarray(A_csr.indptr, dtype=np.int32)
    indices = np.ascontiguousarray(A_csr.indices, dtype=np.int32)
    data = np.ascontiguousarray(A_csr.data, dtype=np.float64)

    if sigma is not None:
        eigenvalues, eigenvectors = _gpu_eigsh_sigma_raw(
            indptr, indices, data, n, k, ncv, maxiter, tol,
            float(sigma), cg_max_iters, cg_tol)
    else:
        eigenvalues, eigenvectors = _gpu_eigsh_raw(
            indptr, indices, data, n, k, ncv, maxiter, tol)

    # Sort by ascending eigenvalue
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    return eigenvalues, eigenvectors


# Lazy import for differentiable_eigsh (requires torch)
def differentiable_eigsh(*args, **kwargs):
    """Differentiable GPU sparse eigendecomposition.

    See gpu_eigsh.differentiable.differentiable_eigsh for full docs.
    Requires PyTorch.
    """
    from .differentiable import differentiable_eigsh as _impl
    return _impl(*args, **kwargs)
