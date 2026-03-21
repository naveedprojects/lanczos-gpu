/**
 * _cuda_eigsh.cu — CUDA backend for gpu_eigsh Python package.
 *
 * Provides a single entry point: compute_eigsh() that accepts CSR
 * arrays, runs GPU IRLM, and returns eigenvalues + eigenvectors.
 *
 * This file is compiled into a shared library that pybind11 wraps.
 */

#include "lanczos_ops.cuh"
#include "tridiag.cuh"
#include <vector>
#include <algorithm>

// Forward declaration from irlm_lanczos.cu
LanczosResult irlm_lanczos(LanczosContext &ctx, SparseMatrixCSR &A,
                           const LanczosParams &params);

// ============================================================
// C-callable entry point for the Python wrapper.
//
// Inputs:
//   n, nnz        — matrix dimensions
//   row_ptr       — CSR row pointers [n+1]
//   col_idx       — CSR column indices [nnz]
//   vals          — CSR values [nnz]
//   k             — number of eigenvalues desired
//   ncv           — Krylov subspace size (0 = auto)
//   max_iters     — max Lanczos steps
//   tol           — convergence tolerance
//
// Outputs:
//   out_eigenvalues  — [k] array (pre-allocated by caller)
//   out_eigenvectors — [n*k] array, column-major (pre-allocated)
//
// Returns: number of converged eigenvalues
// ============================================================
extern "C" int compute_eigsh(
    int n, int nnz,
    const int *row_ptr, const int *col_idx, const double *vals,
    int k, int ncv, int max_iters, double tol,
    double *out_eigenvalues, double *out_eigenvectors)
{
    // Build CSR on GPU
    SparseMatrixCSR A;
    A.n = n;
    A.nnz = nnz;
    CUDA_CHECK(cudaMalloc(&A.d_row_ptr, (n + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&A.d_col_idx, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&A.d_vals,    nnz * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(A.d_row_ptr, row_ptr, (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(A.d_col_idx, col_idx, nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(A.d_vals,    vals,    nnz * sizeof(double), cudaMemcpyHostToDevice));
    A.create_descriptor();

    // Create context
    int max_basis = std::max(max_iters, (ncv > 0 ? ncv : 3 * k + 1));
    LanczosContext ctx;
    ctx.init(n, max_basis);

    // Set params
    LanczosParams params;
    params.num_eigs = k;
    params.max_iters = max_iters;
    params.ncv = ncv;
    params.tol = tol;
    params.measure_freq = max_iters + 1;  // disable ortho measurement

    // Run IRLM
    LanczosResult result = irlm_lanczos(ctx, A, params);

    // Copy results
    int nconv = result.k;
    if (out_eigenvalues && result.eigenvalues)
        memcpy(out_eigenvalues, result.eigenvalues, nconv * sizeof(double));
    if (out_eigenvectors && result.eigenvectors)
        memcpy(out_eigenvectors, result.eigenvectors, (size_t)n * nconv * sizeof(double));

    // Cleanup
    result.free();
    ctx.destroy();
    A.free();

    return nconv;
}
