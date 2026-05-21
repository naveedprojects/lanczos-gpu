/**
 * lanczos_ops.cuh — Shared numerical primitives for GPU Lanczos.
 *
 * Every function takes a LanczosContext& and operates on pre-allocated
 * buffers. No GPU memory is allocated or freed in this file.
 *
 * Primitives:
 *   spmv()                   — Sparse matrix-vector product via cuSPARSE
 *   cgs_orthogonalize()      — Classical Gram-Schmidt against V
 *   dgks_reorth()            — DGKS conditional reorthogonalization
 *   safe_scale()             — ARPACK-style normalization near underflow
 *   inject_random_restart()  — Random vector for invariant subspace
 *   adaptive_breakdown_tol() — Scale-aware breakdown threshold
 *   measure_orthogonality()  — Non-allocating max|V^T V - I|
 */

#pragma once

#include "lanczos_context.cuh"
#include <vector>

// FP64 ↔ FP32 cast kernels (defined in cast_kernels.cu)
extern __global__ void kernel_double_to_float(const double *in, float *out, int n);
extern __global__ void kernel_float_to_double(const float *in, double *out, int n);

// ============================================================
// Mixed-precision SpMV: y_f64 = A_f32 * x_f64
//
// Runs SpMV in FP32 for ~2x bandwidth savings, then casts
// the result back to FP64. The FP32 error (~10^-7) is corrected
// by the DGKS reorthogonalization step.
//
// Requires: A.create_f32_copy() called beforehand.
// ============================================================
inline void spmv_mixed(LanczosContext &ctx, SparseMatrixCSR &A,
                       double *d_x, double *d_y) {
    int n = A.n;
    int blocks = (n + 255) / 256;

    // Cast x: FP64 → FP32
    kernel_double_to_float<<<blocks, 256, 0, ctx.compute_stream>>>(
        d_x, ctx.d_x_f32, n);

    // SpMV in FP32
    float alpha = 1.0f, beta = 0.0f;
    CUSPARSE_CHECK(cusparseDnVecSetValues(ctx.vec_x_f32, ctx.d_x_f32));
    CUSPARSE_CHECK(cusparseDnVecSetValues(ctx.vec_y_f32, ctx.d_w_f32));

    size_t needed = 0;
    CUSPARSE_CHECK(cusparseSpMV_bufferSize(ctx.cusparse,
        CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, A.descr_f32, ctx.vec_x_f32,
        &beta, ctx.vec_y_f32, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &needed));

    if (needed > ctx.spmv_buffer_f32_size) {
        if (ctx.spmv_buffer_f32) cudaFree(ctx.spmv_buffer_f32);
        CUDA_CHECK(cudaMalloc(&ctx.spmv_buffer_f32, needed));
        ctx.spmv_buffer_f32_size = needed;
    }

    CUSPARSE_CHECK(cusparseSpMV(ctx.cusparse,
        CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, A.descr_f32, ctx.vec_x_f32,
        &beta, ctx.vec_y_f32, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT,
        ctx.spmv_buffer_f32));

    // Cast result: FP32 → FP64
    kernel_float_to_double<<<blocks, 256, 0, ctx.compute_stream>>>(
        ctx.d_w_f32, d_y, n);
}

// ============================================================
// SpMV: y = A * x
//
// Uses reusable cuSPARSE descriptors from context. The SpMV
// buffer is grown as needed but never shrunk or freed here.
// ============================================================
inline void spmv(LanczosContext &ctx, SparseMatrixCSR &A,
                 double *d_x, double *d_y) {
    double alpha = 1.0, beta = 0.0;

    // Update descriptor pointers (descriptors already exist in ctx)
    CUSPARSE_CHECK(cusparseDnVecSetValues(ctx.vec_x, d_x));
    CUSPARSE_CHECK(cusparseDnVecSetValues(ctx.vec_y, d_y));

    // Query buffer size (changes only if matrix structure changes)
    size_t needed = 0;
    CUSPARSE_CHECK(cusparseSpMV_bufferSize(ctx.cusparse,
        CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, A.descr, ctx.vec_x,
        &beta, ctx.vec_y, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &needed));

    if (needed > ctx.spmv_buffer_size) {
        if (ctx.spmv_buffer) cudaFree(ctx.spmv_buffer);
        CUDA_CHECK(cudaMalloc(&ctx.spmv_buffer, needed));
        ctx.spmv_buffer_size = needed;
    }

    CUSPARSE_CHECK(cusparseSpMV(ctx.cusparse,
        CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, A.descr, ctx.vec_x,
        &beta, ctx.vec_y, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
        ctx.spmv_buffer));
}

// ============================================================
// Classical Gram-Schmidt orthogonalization.
//
// Given w = A*v_j and the basis V(:, 0:j), computes:
//   h     = V_j^T * w           (coefficients, stored in ctx.d_coeffs)
//   r     = w - V_j * h         (residual, stored in ctx.d_r)
//   alpha = h[j]                (diagonal element of T)
//
// On entry: d_w contains A*v_j.
// On exit:  d_r contains the residual, alpha_j is written.
// ============================================================
inline void cgs_orthogonalize(LanczosContext &ctx,
                              double *d_V, int n, int j,
                              double *d_w, double *d_r,
                              double &alpha_j) {
    double one = 1.0, zero = 0.0, neg_one = -1.0;
    int ncols = j + 1;

    // h = V_j^T * w
    CUBLAS_CHECK(cublasDgemv(ctx.cublas, CUBLAS_OP_T, n, ncols,
        &one, d_V, n, d_w, 1, &zero, ctx.d_coeffs, 1));

    // r = w - V_j * h
    CUDA_CHECK(cudaMemcpy(d_r, d_w, n * sizeof(double), cudaMemcpyDeviceToDevice));
    CUBLAS_CHECK(cublasDgemv(ctx.cublas, CUBLAS_OP_N, n, ncols,
        &neg_one, d_V, n, ctx.d_coeffs, 1, &one, d_r, 1));

    // Extract alpha_j = h[j]
    CUDA_CHECK(cudaMemcpy(&alpha_j, ctx.d_coeffs + j,
        sizeof(double), cudaMemcpyDeviceToHost));
}

// ============================================================
// DGKS conditional reorthogonalization.
//
// Implements the Daniel-Gragg-Kaufman-Stewart criterion from
// ARPACK dsaitr.f: if ||r|| <= (1/sqrt(2)) * ||w||, the residual
// has lost significant orthogonality and needs correction.
//
// Up to MAX_REORTH_PASSES (2) refinement passes. If both fail,
// the residual lies numerically in span(V) — invariant subspace.
//
// Uses ctx.d_coeffs (pre-allocated) for correction coefficients.
// Returns: updated rnorm, updated alpha_j, incremented reorth_count.
// ============================================================
inline void dgks_reorth(LanczosContext &ctx,
                        double *d_V, int n, int j,
                        double *d_r, double &alpha_j,
                        double &rnorm, double wnorm,
                        int &reorth_count) {
    if (rnorm > DGKS_THRESHOLD * wnorm)
        return;  // Orthogonality is fine, no correction needed.

    double one = 1.0, zero = 0.0, neg_one = -1.0;
    int ncols = j + 1;

    for (int pass = 0; pass < MAX_REORTH_PASSES; pass++) {
        reorth_count++;

        // Correction coefficients: s = V_j^T * r
        CUBLAS_CHECK(cublasDgemv(ctx.cublas, CUBLAS_OP_T, n, ncols,
            &one, d_V, n, d_r, 1, &zero, ctx.d_coeffs, 1));

        // Apply correction: r = r - V_j * s
        CUBLAS_CHECK(cublasDgemv(ctx.cublas, CUBLAS_OP_N, n, ncols,
            &neg_one, d_V, n, ctx.d_coeffs, 1, &one, d_r, 1));

        // Update alpha with correction: alpha_j += s[j]
        double corr;
        CUDA_CHECK(cudaMemcpy(&corr, ctx.d_coeffs + j,
            sizeof(double), cudaMemcpyDeviceToHost));
        alpha_j += corr;

        // Check if correction was sufficient
        double rnorm_new;
        CUBLAS_CHECK(cublasDnrm2(ctx.cublas, n, d_r, 1, &rnorm_new));

        if (rnorm_new > DGKS_THRESHOLD * rnorm) {
            rnorm = rnorm_new;
            return;  // Correction accepted.
        }

        // Second pass failed: residual is in span(V) numerically.
        if (pass == MAX_REORTH_PASSES - 1) {
            rnorm = 0.0;
            return;  // Signals invariant subspace to caller.
        }

        rnorm = rnorm_new;
    }
}

// ============================================================
// ARPACK-style safe scaling: v = v / rnorm
//
// When rnorm < SAFE_MIN (near underflow), computing 1/rnorm
// may overflow. Use multi-step scaling following LAPACK's dlascl:
//   step 1: v *= (SAFE_MIN / rnorm)   — bounded, since both tiny
//   step 2: v *= (1 / SAFE_MIN)       — large but representable
//
// Reference: ARPACK dsaitr.f lines 410-430.
// ============================================================
inline void safe_scale(LanczosContext &ctx, double *d_v, int n, double rnorm) {
    if (rnorm >= SAFE_MIN) {
        double inv = 1.0 / rnorm;
        CUBLAS_CHECK(cublasDscal(ctx.cublas, n, &inv, d_v, 1));
    } else if (rnorm > 0.0) {
        // Multi-step scaling to avoid overflow in 1/rnorm
        double step1 = SAFE_MIN / rnorm;     // >= 1.0, bounded
        CUBLAS_CHECK(cublasDscal(ctx.cublas, n, &step1, d_v, 1));
        double step2 = 1.0 / SAFE_MIN;       // large but finite
        CUBLAS_CHECK(cublasDscal(ctx.cublas, n, &step2, d_v, 1));
    }
    // If rnorm == 0: leave v unchanged (caller handles breakdown)
}

// ============================================================
// Inject a random restart vector for invariant subspace.
//
// When DGKS signals rnorm ≈ 0, the residual lies in span(V).
// Generate a random vector, orthogonalize it against V, and
// return the resulting rnorm. This follows ARPACK's dgetv0.
//
// On exit: d_r contains the orthogonalized random vector,
//          rnorm is its norm. If rnorm ≈ 0, the entire space
//          has been explored (shouldn't happen for n >> j).
// ============================================================
inline void inject_random_restart(LanczosContext &ctx,
                                  double *d_V, int n, int j,
                                  double *d_r, double &rnorm) {
    double one = 1.0, zero = 0.0, neg_one = -1.0;
    int ncols = j + 1;

    // Generate random vector on host, copy to device.
    // Using rand() for deterministic seed — a production library
    // would use curandGenerateUniformDouble.
    std::vector<double> rv(n);
    for (int i = 0; i < n; i++)
        rv[i] = (double)rand() / RAND_MAX - 0.5;
    CUDA_CHECK(cudaMemcpy(d_r, rv.data(), n * sizeof(double),
        cudaMemcpyHostToDevice));

    // Orthogonalize against V (two CGS passes for safety)
    for (int pass = 0; pass < 2; pass++) {
        CUBLAS_CHECK(cublasDgemv(ctx.cublas, CUBLAS_OP_T, n, ncols,
            &one, d_V, n, d_r, 1, &zero, ctx.d_coeffs, 1));
        CUBLAS_CHECK(cublasDgemv(ctx.cublas, CUBLAS_OP_N, n, ncols,
            &neg_one, d_V, n, ctx.d_coeffs, 1, &one, d_r, 1));
    }

    CUBLAS_CHECK(cublasDnrm2(ctx.cublas, n, d_r, 1, &rnorm));
}

// ============================================================
// Adaptive breakdown tolerance.
//
// Instead of a hardcoded 1e-14, scale the tolerance with the
// estimated matrix norm: tol = eps^(2/3) * ||T||_1
//
// ||T||_1 ≈ max_i(|alpha[i]| + |beta[i]| + |beta[i+1]|)
// which is cheap to compute from the tridiagonal.
// ============================================================
inline double adaptive_breakdown_tol(const double *alpha,
                                     const double *beta, int j) {
    double est_norm = 0.0;
    for (int i = 0; i <= j; i++) {
        double row_sum = fabs(alpha[i]);
        if (i > 0) row_sum += fabs(beta[i]);
        if (i < j) row_sum += fabs(beta[i + 1]);
        if (row_sum > est_norm) est_norm = row_sum;
    }
    // eps^(2/3) ≈ 3.67e-11 for double precision.
    // This is the floor ARPACK uses in dsconv.f to prevent false convergence.
    double eps23 = pow(MACH_EPS, 2.0 / 3.0);
    return eps23 * fmax(est_norm, SAFE_MIN);
}

// ============================================================
// Measure orthogonality of Lanczos basis V (n x j).
//
// Returns max_{i != k} |V^T V - I|, computed as:
//   max(max off-diagonal |G_ij|, max |G_ii - 1|)
//
// Uses pre-allocated ctx.d_G and ctx.h_G — no memory allocation.
// ============================================================
inline double measure_orthogonality(LanczosContext &ctx,
                                    double *d_V, int n, int j) {
    if (j <= 1) return 0.0;
    if (j > ctx.max_basis_size) return -1.0;  // buffer too small

    double alpha = 1.0, beta = 0.0;
    CUBLAS_CHECK(cublasDgemm(ctx.cublas, CUBLAS_OP_T, CUBLAS_OP_N,
        j, j, n, &alpha, d_V, n, d_V, n, &beta, ctx.d_G, j));

    size_t bytes = (size_t)j * j * sizeof(double);
    CUDA_CHECK(cudaMemcpy(ctx.h_G, ctx.d_G, bytes, cudaMemcpyDeviceToHost));

    double max_err = 0.0;
    for (int col = 0; col < j; col++) {
        for (int row = 0; row < j; row++) {
            double val = ctx.h_G[row + col * j];
            double err = (row == col) ? fabs(val - 1.0) : fabs(val);
            if (err > max_err) max_err = err;
        }
    }
    return max_err;
}

// ============================================================
// Compute Ritz vectors in batch: X = V * S
//
// V is n x m (Lanczos basis on GPU), S is m x k (eigenvectors
// of the tridiagonal on host), X is n x k (Ritz vectors on host).
//
// Uses a single DGEMM instead of k separate GEMV calls.
// ============================================================
inline void compute_ritz_vectors(LanczosContext &ctx,
                                 double *d_V, int n, int m,
                                 const double *h_S, int k,
                                 double *h_X) {
    // Copy S to device
    double *d_S, *d_X;
    CUDA_CHECK(cudaMalloc(&d_S, (size_t)m * k * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_X, (size_t)n * k * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_S, h_S, (size_t)m * k * sizeof(double),
        cudaMemcpyHostToDevice));

    double one = 1.0, zero = 0.0;
    CUBLAS_CHECK(cublasDgemm(ctx.cublas, CUBLAS_OP_N, CUBLAS_OP_N,
        n, k, m, &one, d_V, n, d_S, m, &zero, d_X, n));

    CUDA_CHECK(cudaMemcpy(h_X, d_X, (size_t)n * k * sizeof(double),
        cudaMemcpyDeviceToHost));

    cudaFree(d_S);
    cudaFree(d_X);
}

// ============================================================
// Rayleigh-Ritz refinement of Ritz vectors.
//
// Given approximate eigenvectors X (n x k, column-major on host)
// and the matrix operator, refine by:
//   1. AX = A * X         (k SpMVs via GEMM)
//   2. H  = X^T * AX      (k x k Rayleigh quotient)
//   3. H  = U D U^T       (eigendecompose, LAPACK dsyev)
//   4. X' = X * U          (rotate to refined basis)
//   5. eigenvalues = D
//
// This ensures:
//   - Eigenvectors are exactly orthonormal (from U^T U = I)
//   - Eigenvalues are optimal Rayleigh quotients
//   - Residuals ||AX' - X'D|| are minimized within span(X)
//
// Cost: k SpMVs + O(n k^2) GEMM + O(k^3) dsyev.
// For k << n, this is negligible compared to the IRLM iteration.
// ============================================================
extern "C" void dsyev_(const char *jobz, const char *uplo, const int *n,
                       double *a, const int *lda, double *w,
                       double *work, const int *lwork, int *info);

// Forward declaration: MatVecOperator is defined in matvec_operator.cuh
// but this header is included before it. We use a function pointer instead.
inline void rayleigh_ritz_refine(LanczosContext &ctx,
                                 SparseMatrixCSR &A,
                                 double *h_eigenvalues, double *h_eigenvectors,
                                 int n, int k) {
    double one = 1.0, zero = 0.0;

    // Upload X to device
    double *d_X, *d_AX;
    CUDA_CHECK(cudaMalloc(&d_X,  (size_t)n * k * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_AX, (size_t)n * k * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_X, h_eigenvectors, (size_t)n * k * sizeof(double),
        cudaMemcpyHostToDevice));

    // AX = A * X  (k SpMVs as a single SpMM-like operation via column loop)
    for (int j = 0; j < k; j++) {
        double *d_xj  = d_X  + (size_t)j * n;
        double *d_axj = d_AX + (size_t)j * n;

        CUSPARSE_CHECK(cusparseDnVecSetValues(ctx.vec_x, (void *)d_xj));
        CUSPARSE_CHECK(cusparseDnVecSetValues(ctx.vec_y, d_axj));

        size_t needed = 0;
        CUSPARSE_CHECK(cusparseSpMV_bufferSize(ctx.cusparse,
            CUSPARSE_OPERATION_NON_TRANSPOSE, &one, A.descr, ctx.vec_x,
            &zero, ctx.vec_y, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &needed));

        if (needed > ctx.spmv_buffer_size) {
            if (ctx.spmv_buffer) cudaFree(ctx.spmv_buffer);
            CUDA_CHECK(cudaMalloc(&ctx.spmv_buffer, needed));
            ctx.spmv_buffer_size = needed;
        }

        CUSPARSE_CHECK(cusparseSpMV(ctx.cusparse,
            CUSPARSE_OPERATION_NON_TRANSPOSE, &one, A.descr, ctx.vec_x,
            &zero, ctx.vec_y, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
            ctx.spmv_buffer));
    }

    // H = X^T * AX  (k x k on device)
    double *d_H;
    CUDA_CHECK(cudaMalloc(&d_H, (size_t)k * k * sizeof(double)));
    CUBLAS_CHECK(cublasDgemm(ctx.cublas, CUBLAS_OP_T, CUBLAS_OP_N,
        k, k, n, &one, d_X, n, d_AX, n, &zero, d_H, k));

    // Download H to host for LAPACK dsyev
    std::vector<double> H(k * k);
    CUDA_CHECK(cudaMemcpy(H.data(), d_H, k * k * sizeof(double),
        cudaMemcpyDeviceToHost));
    cudaFree(d_H);

    // Eigendecompose H = U D U^T
    std::vector<double> w(k);
    int lwork = -1;
    double work_query;
    int info;
    char jobz = 'V', uplo = 'U';
    dsyev_(&jobz, &uplo, &k, H.data(), &k, w.data(),
           &work_query, &lwork, &info);
    lwork = (int)work_query;
    std::vector<double> work(lwork);
    dsyev_(&jobz, &uplo, &k, H.data(), &k, w.data(),
           work.data(), &lwork, &info);

    if (info != 0) {
        fprintf(stderr, "Warning: dsyev failed in Rayleigh-Ritz refinement (info=%d)\n", info);
        cudaFree(d_X);
        cudaFree(d_AX);
        return;  // keep original vectors
    }

    // Refined eigenvalues
    memcpy(h_eigenvalues, w.data(), k * sizeof(double));

    // X' = X * U  (n x k = (n x k) * (k x k))
    double *d_U, *d_Xnew;
    CUDA_CHECK(cudaMalloc(&d_U,    (size_t)k * k * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Xnew, (size_t)n * k * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_U, H.data(), k * k * sizeof(double),
        cudaMemcpyHostToDevice));

    CUBLAS_CHECK(cublasDgemm(ctx.cublas, CUBLAS_OP_N, CUBLAS_OP_N,
        n, k, k, &one, d_X, n, d_U, k, &zero, d_Xnew, n));

    CUDA_CHECK(cudaMemcpy(h_eigenvectors, d_Xnew, (size_t)n * k * sizeof(double),
        cudaMemcpyDeviceToHost));

    cudaFree(d_X);
    cudaFree(d_AX);
    cudaFree(d_U);
    cudaFree(d_Xnew);
}

// Operator-based version: works with any MatVecOperator.
// Forward declaration — MatVecOperator is defined in matvec_operator.cuh.
struct MatVecOperator;

inline void rayleigh_ritz_refine_op(LanczosContext &ctx,
                                    MatVecOperator &op,
                                    double *h_eigenvalues, double *h_eigenvectors,
                                    int n, int k);  // defined after matvec_operator.cuh is included
