"""
gpu_eigsh — GPU-accelerated sparse eigenvalue solver.

Drop-in replacement for scipy.sparse.linalg.eigsh, backed by an
ARPACK-faithful IRLM implementation on CUDA.

Usage:
    from gpu_eigsh import gpu_eigsh
    eigenvalues, eigenvectors = gpu_eigsh(L, k=20)

    # Same interface as scipy:
    # eigenvalues: (k,) array, smallest k eigenvalues in ascending order
    # eigenvectors: (n, k) array, corresponding eigenvectors as columns
"""

import numpy as np
from scipy.sparse import issparse, csr_matrix

from ._core import _gpu_eigsh_raw


def gpu_eigsh(A, k=20, ncv=0, maxiter=2000, tol=1e-12, which='SM'):
    """Compute k smallest eigenvalues of a sparse symmetric matrix on GPU.

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
        Which eigenvalues to compute. Only 'SM' (smallest magnitude)
        is currently supported.

    Returns
    -------
    eigenvalues : ndarray (k,)
        The k smallest eigenvalues in ascending order.
    eigenvectors : ndarray (n, k)
        Corresponding eigenvectors as columns.
    """
    if which != 'SM':
        raise NotImplementedError(f"which='{which}' not supported, use 'SM'")

    if not issparse(A):
        raise TypeError("A must be a scipy sparse matrix")

    # Convert to CSR with correct dtypes
    A_csr = csr_matrix(A, dtype=np.float64)
    n = A_csr.shape[0]

    if A_csr.shape[0] != A_csr.shape[1]:
        raise ValueError(f"Matrix must be square, got {A_csr.shape}")
    if k < 1 or k > n:
        raise ValueError(f"k must be in [1, {n}], got {k}")

    indptr = np.ascontiguousarray(A_csr.indptr, dtype=np.int32)
    indices = np.ascontiguousarray(A_csr.indices, dtype=np.int32)
    data = np.ascontiguousarray(A_csr.data, dtype=np.float64)

    eigenvalues, eigenvectors = _gpu_eigsh_raw(
        indptr, indices, data, n, k, ncv, maxiter, tol)

    # Sort by ascending eigenvalue (should already be sorted, but be safe)
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    return eigenvalues, eigenvectors
