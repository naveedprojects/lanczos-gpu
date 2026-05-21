/**
 * inner_solve.cuh — Deflated Conjugate Gradient solver on GPU.
 *
 * Solves (A - sigma*I) x = b subject to deflation against a known
 * eigenspace X, using only matrix-vector products via MatVecOperator.
 *
 * Two primary use cases:
 *   1. Shift-invert Lanczos: each outer Lanczos step requires solving
 *      (A - sigma*I) y = v, turning eigenvalues near sigma into the
 *      dominant eigenvalues of the transformed operator.
 *
 *   2. Adjoint backward pass: the implicit differentiation formula
 *      requires solving (A - lambda_i*I) xi = rhs for each eigenpair,
 *      with deflation against the computed eigenvectors.
 *
 * The solver reuses LanczosContext work buffers (d_w, d_r) and adds
 * its own pre-allocated workspace for CG vectors. No memory allocation
 * inside the iteration loop.
 *
 * References:
 *   Golub & Van Loan, "Matrix Computations", 4th ed., Ch. 11
 *   Saad, "Iterative Methods for Sparse Linear Systems", 2nd ed.
 */

#pragma once

#include "matvec_operator.cuh"

// ============================================================
// Workspace for the CG solver. Pre-allocated once, reused
// across multiple solves (e.g., k backward adjoint systems).
// ============================================================
struct CGWorkspace {
    double *d_p   = nullptr;  // search direction     [n]
    double *d_Ap  = nullptr;  // A * p                [n]
    double *d_x   = nullptr;  // solution             [n]
    double *d_rr  = nullptr;  // residual             [n]
    double *d_tmp = nullptr;  // temp for deflection  [n]
    int n = 0;

    void init(int n_) {
        n = n_;
        CUDA_CHECK(cudaMalloc(&d_p,   n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_Ap,  n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_x,   n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_rr,  n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_tmp, n * sizeof(double)));
    }

    void destroy() {
        if (d_p)   { cudaFree(d_p);   d_p   = nullptr; }
        if (d_Ap)  { cudaFree(d_Ap);  d_Ap  = nullptr; }
        if (d_x)   { cudaFree(d_x);   d_x   = nullptr; }
        if (d_rr)  { cudaFree(d_rr);  d_rr  = nullptr; }
        if (d_tmp) { cudaFree(d_tmp); d_tmp = nullptr; }
        n = 0;
    }
};

// ============================================================
// Project a vector out of the column space of X:
//   v = (I - X X^T) v
//
// X is n x k, column-major on device. Uses cuBLAS GEMV.
// This ensures the CG iterate stays orthogonal to the
// eigenspace, preventing convergence to the null-space
// direction of (A - lambda_i I).
// ============================================================
inline void deflate_against(LanczosContext &ctx,
                            double *d_v, int n,
                            const double *d_X, int k,
                            double *d_tmp) {
    if (k <= 0) return;

    double one = 1.0, zero = 0.0, neg_one = -1.0;

    // tmp = X^T v  (k x 1)
    CUBLAS_CHECK(cublasDgemv(ctx.cublas, CUBLAS_OP_T, n, k,
        &one, d_X, n, d_v, 1, &zero, d_tmp, 1));

    // v = v - X * tmp
    CUBLAS_CHECK(cublasDgemv(ctx.cublas, CUBLAS_OP_N, n, k,
        &neg_one, d_X, n, d_tmp, 1, &one, d_v, 1));
}

// ============================================================
// Deflated Conjugate Gradient solver.
//
// Solves: (A - sigma*I) x = b
// subject to: X^T x = 0  (deflation against eigenspace)
//
// Parameters:
//   ctx       — GPU context (cuBLAS handle, etc.)
//   op        — matrix-vector operator (A)
//   sigma     — shift value
//   d_b       — right-hand side vector [n], device
//   d_X       — eigenvectors to deflate against [n x k_defl], device
//   k_defl    — number of deflation vectors (0 = no deflation)
//   ws        — pre-allocated CG workspace
//   tol       — relative residual tolerance
//   max_iters — maximum CG iterations
//
// On exit: ws.d_x contains the solution.
// Returns: number of CG iterations taken (negative if not converged).
//
// The right-hand side b must already be orthogonal to span(X)
// for the system to be consistent. The solver re-deflates the
// residual every iteration to maintain numerical orthogonality.
// ============================================================
inline int cg_solve_shifted(LanczosContext &ctx,
                            MatVecOperator &op,
                            double sigma,
                            const double *d_b, int n,
                            const double *d_X, int k_defl,
                            CGWorkspace &ws,
                            double tol = 1e-10,
                            int max_iters = 500) {
    double one = 1.0;

    // x_0 = 0
    CUDA_CHECK(cudaMemset(ws.d_x, 0, n * sizeof(double)));

    // r_0 = b - (A - sigma*I) * x_0 = b
    CUDA_CHECK(cudaMemcpy(ws.d_rr, d_b, n * sizeof(double),
        cudaMemcpyDeviceToDevice));

    // Deflate r_0 against eigenspace
    deflate_against(ctx, ws.d_rr, n, d_X, k_defl, ws.d_tmp);

    // p_0 = r_0
    CUDA_CHECK(cudaMemcpy(ws.d_p, ws.d_rr, n * sizeof(double),
        cudaMemcpyDeviceToDevice));

    // rr = r^T r
    double rr_old;
    CUBLAS_CHECK(cublasDdot(ctx.cublas, n, ws.d_rr, 1, ws.d_rr, 1, &rr_old));

    double bnorm;
    CUBLAS_CHECK(cublasDnrm2(ctx.cublas, n, d_b, 1, &bnorm));
    if (bnorm < SAFE_MIN) return 0;  // trivial RHS

    double tol_abs = tol * bnorm;

    for (int iter = 0; iter < max_iters; iter++) {
        // Ap = (A - sigma*I) * p
        op.apply_shifted(ctx, ws.d_p, ws.d_Ap, sigma);

        // Deflate Ap to stay in the complement of eigenspace
        deflate_against(ctx, ws.d_Ap, n, d_X, k_defl, ws.d_tmp);

        // alpha = rr / (p^T Ap)
        double pAp;
        CUBLAS_CHECK(cublasDdot(ctx.cublas, n, ws.d_p, 1,
            ws.d_Ap, 1, &pAp));

        // Guard against near-zero pAp (near-singular direction)
        if (fabs(pAp) < SAFE_MIN) {
            return -(iter + 1);  // breakdown
        }

        double alpha = rr_old / pAp;

        // x = x + alpha * p
        CUBLAS_CHECK(cublasDaxpy(ctx.cublas, n, &alpha,
            ws.d_p, 1, ws.d_x, 1));

        // r = r - alpha * Ap
        double neg_alpha = -alpha;
        CUBLAS_CHECK(cublasDaxpy(ctx.cublas, n, &neg_alpha,
            ws.d_Ap, 1, ws.d_rr, 1));

        // Re-deflate residual every iteration for numerical stability
        deflate_against(ctx, ws.d_rr, n, d_X, k_defl, ws.d_tmp);

        // Check convergence
        double rr_new;
        CUBLAS_CHECK(cublasDdot(ctx.cublas, n, ws.d_rr, 1,
            ws.d_rr, 1, &rr_new));

        if (sqrt(rr_new) < tol_abs) {
            // Re-deflate solution for safety
            deflate_against(ctx, ws.d_x, n, d_X, k_defl, ws.d_tmp);
            return iter + 1;
        }

        // beta = rr_new / rr_old
        double beta = rr_new / rr_old;

        // p = r + beta * p
        CUBLAS_CHECK(cublasDscal(ctx.cublas, n, &beta, ws.d_p, 1));
        CUBLAS_CHECK(cublasDaxpy(ctx.cublas, n, &one,
            ws.d_rr, 1, ws.d_p, 1));

        rr_old = rr_new;
    }

    // Did not converge — return negative count
    deflate_against(ctx, ws.d_x, n, d_X, k_defl, ws.d_tmp);
    return -max_iters;
}

// ============================================================
// Shift-invert operator: y = (A - sigma*I)^{-1} * x
//
// Each "matvec" solves the linear system (A - sigma*I) y = x
// via deflated CG. This transforms eigenvalues near sigma into
// the dominant eigenvalues of the operator, which Lanczos
// finds fastest.
//
// Eigenvalue back-transformation: if IRLM finds mu_i of this
// operator, the original eigenvalues are lambda_i = sigma + 1/mu_i.
// ============================================================
struct ShiftInvertOperator : MatVecOperator {
    MatVecOperator &inner_op;
    double sigma;
    CGWorkspace &cg_ws;
    double cg_tol;
    int cg_max_iters;

    const double *d_X_defl = nullptr;
    int k_defl = 0;

    ShiftInvertOperator(MatVecOperator &op_, double sigma_,
                        CGWorkspace &ws_,
                        double cg_tol_ = 1e-12,
                        int cg_max_iters_ = 500)
        : inner_op(op_), sigma(sigma_), cg_ws(ws_),
          cg_tol(cg_tol_), cg_max_iters(cg_max_iters_) {
        n = op_.n;
    }

    void apply(LanczosContext &ctx,
               const double *d_x, double *d_y) override {
        int iters = cg_solve_shifted(ctx, inner_op, sigma,
                                     d_x, n, d_X_defl, k_defl,
                                     cg_ws, cg_tol, cg_max_iters);
        CUDA_CHECK(cudaMemcpy(d_y, cg_ws.d_x, n * sizeof(double),
            cudaMemcpyDeviceToDevice));
        (void)iters;
    }
};
