/**
 * _cuda_eigsh.cu — CUDA backend for gpu_eigsh Python package.
 *
 * Entry points:
 *   compute_eigsh()          — Standard eigensolver (smallest eigenvalues)
 *   compute_eigsh_sigma()    — Shift-invert mode (eigenvalues near sigma)
 *   compute_adjoint_eigsh()  — Backward pass for differentiable eigsh
 *
 * This file is compiled into a shared library that pybind11 wraps.
 */

#include "lanczos_ops.cuh"
#include "matvec_operator.cuh"
#include "inner_solve.cuh"
#include "tridiag.cuh"
#include <vector>
#include <algorithm>
#include <cmath>

// Forward declarations from irlm_lanczos.cu
LanczosResult irlm_lanczos(LanczosContext &ctx, SparseMatrixCSR &A,
                           const LanczosParams &params);
LanczosResult irlm_lanczos(LanczosContext &ctx, MatVecOperator &op,
                           const LanczosParams &params);

// ============================================================
// Standard eigensolver: k smallest eigenvalues.
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
    params.measure_freq = max_iters + 1;

    // Run IRLM
    LanczosResult result = irlm_lanczos(ctx, A, params);

    // Copy results
    int nconv = result.k;
    if (out_eigenvalues && result.eigenvalues)
        memcpy(out_eigenvalues, result.eigenvalues, nconv * sizeof(double));
    if (out_eigenvectors && result.eigenvectors)
        memcpy(out_eigenvectors, result.eigenvectors, (size_t)n * nconv * sizeof(double));

    result.free();
    ctx.destroy();
    A.free();
    return nconv;
}

// ============================================================
// Shift-invert eigensolver: k eigenvalues nearest to sigma.
//
// Runs IRLM on (A - sigma*I)^{-1} via CG inner solver.
// Eigenvalues are back-transformed: lambda = sigma + 1/mu.
// ============================================================
extern "C" int compute_eigsh_sigma(
    int n, int nnz,
    const int *row_ptr, const int *col_idx, const double *vals,
    int k, int ncv, int max_iters, double tol, double sigma,
    int cg_max_iters, double cg_tol,
    double *out_eigenvalues, double *out_eigenvectors)
{
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

    int max_basis = std::max(max_iters, (ncv > 0 ? ncv : 3 * k + 1));
    LanczosContext ctx;
    ctx.init(n, max_basis);

    // Create operator chain: CSR -> ShiftInvert
    CSROperator csr_op(A);
    CGWorkspace cg_ws;
    cg_ws.init(n);
    ShiftInvertOperator si_op(csr_op, sigma, cg_ws, cg_tol, cg_max_iters);

    LanczosParams params;
    params.num_eigs = k;
    params.max_iters = max_iters;
    params.ncv = ncv;
    params.tol = tol;
    params.measure_freq = max_iters + 1;

    // Run IRLM on the shift-invert operator
    LanczosResult result = irlm_lanczos(ctx, si_op, params);

    // Back-transform eigenvalues: lambda = sigma + 1/mu
    int nconv = result.k;
    if (out_eigenvalues && result.eigenvalues) {
        for (int i = 0; i < nconv; i++) {
            double mu = result.eigenvalues[i];
            out_eigenvalues[i] = (fabs(mu) > SAFE_MIN)
                                 ? sigma + 1.0 / mu
                                 : sigma;
        }
        // Sort by ascending eigenvalue
        // Build index array for simultaneous sort of eigenvalues+eigenvectors
        std::vector<int> idx(nconv);
        for (int i = 0; i < nconv; i++) idx[i] = i;
        std::sort(idx.begin(), idx.end(), [&](int a, int b) {
            return out_eigenvalues[a] < out_eigenvalues[b];
        });

        // Apply permutation
        std::vector<double> evals_sorted(nconv);
        for (int i = 0; i < nconv; i++)
            evals_sorted[i] = out_eigenvalues[idx[i]];
        memcpy(out_eigenvalues, evals_sorted.data(), nconv * sizeof(double));

        if (out_eigenvectors && result.eigenvectors) {
            std::vector<double> evecs_sorted((size_t)n * nconv);
            for (int i = 0; i < nconv; i++) {
                memcpy(evecs_sorted.data() + (size_t)i * n,
                       result.eigenvectors + (size_t)idx[i] * n,
                       n * sizeof(double));
            }
            memcpy(out_eigenvectors, evecs_sorted.data(),
                   (size_t)n * nconv * sizeof(double));
        }
    } else if (out_eigenvectors && result.eigenvectors) {
        memcpy(out_eigenvectors, result.eigenvectors,
               (size_t)n * nconv * sizeof(double));
    }

    result.free();
    cg_ws.destroy();
    ctx.destroy();
    A.free();
    return nconv;
}

// ============================================================
// Adjoint backward pass: implicit differentiation.
//
// Given converged eigenpairs (lambda_i, x_i) and upstream
// gradients (grad_evals, grad_evecs), computes the gradient
// of the loss w.r.t. the CSR values of A.
//
// For each eigenpair i:
//   1. rhs = (I - X X^T) grad_evecs_i
//   2. Solve (A - lambda_i I) xi = rhs  via deflated CG
//   3. Accumulate: A_bar += (grad_evals_i * x_i - xi) x_i^T
//
// The output grad_vals[j] = sum over (r,c) in the sparsity
// pattern of A: A_bar[r,c] evaluated at position j.
//
// Reference: Xie et al., "Automatic differentiation of dominant
// eigensolver" (2020), arXiv:2001.04121
// ============================================================
extern "C" int compute_adjoint_eigsh(
    int n, int nnz,
    const int *row_ptr, const int *col_idx, const double *vals,
    int k,
    const double *eigenvalues,    // [k] — from forward pass
    const double *eigenvectors,   // [n*k] C-contiguous (row-major) — from forward pass
    const double *grad_evals,     // [k] — upstream gradient on eigenvalues
    const double *grad_evecs,     // [n*k] C-contiguous (row-major) — upstream gradient
    double *grad_vals,            // [nnz] — output: gradient w.r.t. A values
    int cg_max_iters, double cg_tol)
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

    LanczosContext ctx;
    ctx.init(n, k + 1);

    CSROperator csr_op(A);
    CGWorkspace cg_ws;
    cg_ws.init(n);

    // Convert eigenvectors from C-contiguous (n x k, row-major: evecs[i*k+j])
    // to column-major (n x k: col j at offset j*n) for cuBLAS GEMV.
    // This transpose is O(n*k) and done once.
    std::vector<double> evecs_colmaj((size_t)n * k);
    for (int j = 0; j < k; j++)
        for (int i = 0; i < n; i++)
            evecs_colmaj[j * n + i] = eigenvectors[i * k + j];

    double *d_X;
    CUDA_CHECK(cudaMalloc(&d_X, (size_t)n * k * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_X, evecs_colmaj.data(), (size_t)n * k * sizeof(double),
        cudaMemcpyHostToDevice));

    // Same transpose for grad_evecs
    std::vector<double> gevecs_colmaj((size_t)n * k, 0.0);
    if (grad_evecs) {
        for (int j = 0; j < k; j++)
            for (int i = 0; i < n; i++)
                gevecs_colmaj[j * n + i] = grad_evecs[i * k + j];
    }

    double *d_grad_evecs;
    CUDA_CHECK(cudaMalloc(&d_grad_evecs, (size_t)n * k * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_grad_evecs, gevecs_colmaj.data(),
        (size_t)n * k * sizeof(double), cudaMemcpyHostToDevice));

    // Work vectors for backward
    double *d_rhs, *d_xi;
    CUDA_CHECK(cudaMalloc(&d_rhs, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_xi,  n * sizeof(double)));
    // d_acc accumulates the rank-k contribution: sum_i (g_i x_i - xi_i) x_i^T
    // We don't form the full n x n matrix. Instead, we accumulate directly
    // into grad_vals by sampling at the sparsity pattern.

    // Upload row_ptr and col_idx to device for sparse sampling
    int *d_row_ptr_copy, *d_col_idx_copy;
    CUDA_CHECK(cudaMalloc(&d_row_ptr_copy, (n + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_col_idx_copy, nnz * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_row_ptr_copy, row_ptr, (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_idx_copy, col_idx, nnz * sizeof(int), cudaMemcpyHostToDevice));

    // Accumulate gradient on host (simpler, nnz is manageable)
    std::vector<int> h_row_ptr(n + 1), h_col_idx(nnz);
    memcpy(h_row_ptr.data(), row_ptr, (n + 1) * sizeof(int));
    memcpy(h_col_idx.data(), col_idx, nnz * sizeof(int));

    // Zero output gradient
    std::vector<double> h_grad_vals(nnz, 0.0);

    // Temporary host buffers for x_i and xi_i
    std::vector<double> h_xi(n), h_xi_full(n);

    for (int i = 0; i < k; i++) {
        double lambda_i = eigenvalues[i];
        double g_eval_i = grad_evals ? grad_evals[i] : 0.0;

        // rhs = grad_evecs_i
        CUDA_CHECK(cudaMemcpy(d_rhs, d_grad_evecs + (size_t)i * n,
            n * sizeof(double), cudaMemcpyDeviceToDevice));

        // Project out eigenspace: rhs = (I - X X^T) rhs
        deflate_against(ctx, d_rhs, n, d_X, k, cg_ws.d_tmp);

        // Solve (A - lambda_i I) xi = rhs with deflation against X
        int cg_iters = cg_solve_shifted(ctx, csr_op, lambda_i,
                                        d_rhs, n, d_X, k,
                                        cg_ws, cg_tol, cg_max_iters);

        // xi = solution from CG
        CUDA_CHECK(cudaMemcpy(d_xi, cg_ws.d_x, n * sizeof(double),
            cudaMemcpyDeviceToDevice));

        // Re-deflate for numerical safety
        deflate_against(ctx, d_xi, n, d_X, k, cg_ws.d_tmp);

        // Accumulate into grad_vals:
        // A_bar += (g_eval_i * x_i - xi) * x_i^T
        // grad_vals[j] += (g_eval_i * x_i[row[j]] - xi[row[j]]) * x_i[col[j]]
        //
        // Download x_i and xi to host for accumulation
        std::vector<double> h_x_i(n);
        CUDA_CHECK(cudaMemcpy(h_x_i.data(), d_X + (size_t)i * n,
            n * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_xi.data(), d_xi,
            n * sizeof(double), cudaMemcpyDeviceToHost));

        // Accumulate: for each nonzero at (r, c), add contribution
        for (int r = 0; r < n; r++) {
            double coeff_r = g_eval_i * h_x_i[r] - h_xi[r];
            for (int p = h_row_ptr[r]; p < h_row_ptr[r + 1]; p++) {
                int c = h_col_idx[p];
                h_grad_vals[p] += coeff_r * h_x_i[c];
            }
        }

        (void)cg_iters;
    }

    // Copy gradient to output
    memcpy(grad_vals, h_grad_vals.data(), nnz * sizeof(double));

    // Cleanup
    cudaFree(d_X);
    cudaFree(d_grad_evecs);
    cudaFree(d_rhs);
    cudaFree(d_xi);
    cudaFree(d_row_ptr_copy);
    cudaFree(d_col_idx_copy);
    cg_ws.destroy();
    ctx.destroy();
    A.free();

    return 0;
}
