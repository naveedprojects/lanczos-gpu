/**
 * matrix_function.cuh — Lanczos-based matrix function evaluation.
 *
 * Computes f(A) v approximately via an m-step Lanczos process:
 *
 *   1. Build V_m (n × m) and T_m (m × m tridiagonal) such that
 *      V_m^T A V_m = T_m, V_m e_1 = v / ||v||
 *   2. Eigendecompose T_m = S Θ S^T (LAPACK dstev)
 *   3. f(A) v ≈ ||v|| · V_m S f(Θ) S^T e_1
 *
 * As a special case, the *quadratic form* v^T f(A) v is
 *   ≈ ||v||² · sum_i (S[0,i])² · f(Θ_i)
 * which is the Hutchinson probe for stochastic Lanczos quadrature.
 *
 * No restart logic here — this is the unrestarted primitive. The
 * restart-aware adjoint is in restart_adjoint.cuh (not yet built).
 *
 * Reference: Golub & Meurant, "Matrices, Moments and Quadrature"
 *            Ubaru, Chen, Saad, "Fast Estimation of tr(f(A))..." (2017)
 *            Pleiss et al., "Fast Matrix Square Roots..." (NeurIPS 2020)
 *            Krämer et al., "Gradients of Functions of Large Matrices" (NeurIPS 2024)
 */

#pragma once

#include "lanczos_ops.cuh"
#include "matvec_operator.cuh"
#include "tridiag.cuh"
#include <vector>
#include <cmath>

// ============================================================
// Saved state from a single Lanczos sweep.
//
// V is stored on device (column-major n × m). alpha/beta on host.
// Caller is responsible for calling free() when done.
// ============================================================
struct LanczosBasis {
    double *d_V    = nullptr;   // device, n × m_actual columns
    double *alpha  = nullptr;   // host, length m_actual
    double *beta   = nullptr;   // host, length m_actual + 1 (beta[0] unused, beta[m_actual] = restart residual norm)
    int     n      = 0;
    int     m      = 0;         // requested depth
    int     m_actual = 0;       // actual depth (may be < m if breakdown)
    double  v_norm = 0.0;       // ||v|| of original input vector
    int     reorth_count = 0;

    void free_resources() {
        if (d_V)   { cudaFree(d_V);   d_V   = nullptr; }
        if (alpha) { ::free(alpha);   alpha = nullptr; }
        if (beta)  { ::free(beta);    beta  = nullptr; }
        n = m = m_actual = 0;
        v_norm = 0.0;
    }
};

// Forward declaration: defined in irlm_lanczos.cu
// Runs Lanczos steps [start, end) with DGKS reorthogonalization.
//   - On entry: d_V column `start` holds v_start (normalized)
//   - On exit:  d_V columns [start..end), alpha[start..end-1] filled;
//               beta[end] = ||r|| of the (end-th) residual;
//               ctx.d_r holds the unnormalized residual.
void lanczos_extend(LanczosContext &ctx, MatVecOperator &op,
                    double *d_V, int n,
                    double *alpha, double *beta,
                    int start, int end, int &reorth_count);

// ============================================================
// No-random-restart Lanczos extension for matrix functions.
//
// The difference from lanczos_extend: on invariant-subspace
// breakdown (rnorm collapses below tol), we STOP rather than
// inject a random vector. For f(A)v we need V^T A V = T exactly
// on the Krylov subspace; injecting a random vector would
// destroy this relation.
//
// Returns the number of completed Lanczos steps (1..end-start).
// If breakdown occurs at step j, the function returns j+1
// (we have alpha[start..start+j] and beta[start+1..start+j+1]
//  with beta[start+j+1] ≈ 0).
// ============================================================
inline int lanczos_extend_funm(LanczosContext &ctx, MatVecOperator &op,
                               double *d_V, int n,
                               double *alpha, double *beta,
                               int start, int end, int &reorth_count) {
    int last_completed = start - 1;

    for (int j = start; j < end; j++) {
        double *d_vj = d_V + (size_t)j * n;

        op.apply(ctx, d_vj, ctx.d_w);

        double wnorm;
        CUBLAS_CHECK(cublasDnrm2(ctx.cublas, n, ctx.d_w, 1, &wnorm));

        cgs_orthogonalize(ctx, d_V, n, j, ctx.d_w, ctx.d_r, alpha[j]);

        double rnorm;
        CUBLAS_CHECK(cublasDnrm2(ctx.cublas, n, ctx.d_r, 1, &rnorm));

        dgks_reorth(ctx, d_V, n, j, ctx.d_r, alpha[j], rnorm, wnorm,
                    reorth_count);

        last_completed = j;

        // Detect invariant-subspace breakdown — STOP cleanly.
        double tol = adaptive_breakdown_tol(alpha, beta, j);
        if (rnorm < tol) {
            beta[j + 1] = 0.0;
            return (last_completed - start) + 1;
        }

        if (j + 1 < end) {
            beta[j + 1] = rnorm;
            double *d_vjp1 = d_V + (size_t)(j + 1) * n;
            CUDA_CHECK(cudaMemcpy(d_vjp1, ctx.d_r, n * sizeof(double),
                                  cudaMemcpyDeviceToDevice));
            if (rnorm > 0.0)
                safe_scale(ctx, d_vjp1, n, rnorm);
        }
        if (j == end - 1)
            beta[end] = rnorm;
    }

    return (last_completed - start) + 1;
}

// ============================================================
// lanczos_build_basis: run m Lanczos steps starting from v.
//
// Allocates d_V (n × m), alpha (m), beta (m+1) and fills them.
// Stores ||v|| in out.v_norm; the basis vectors are constructed
// from v/||v|| as the starting vector.
//
// Breakdown handling: if the residual norm collapses, m_actual
// will be less than m. Caller can check basis.m_actual.
// ============================================================
inline void lanczos_build_basis(LanczosContext &ctx,
                                MatVecOperator &op,
                                const double *d_v,
                                int m,
                                LanczosBasis &out) {
    int n = op.n;
    out.n = n;
    out.m = m;
    out.alpha = (double *)calloc(m + 1, sizeof(double));
    out.beta  = (double *)calloc(m + 2, sizeof(double));
    CUDA_CHECK(cudaMalloc(&out.d_V, (size_t)n * m * sizeof(double)));

    // Compute ||v|| and write v/||v|| into column 0 of d_V
    double vnorm = 0.0;
    CUBLAS_CHECK(cublasDnrm2(ctx.cublas, n, d_v, 1, &vnorm));
    out.v_norm = vnorm;
    if (vnorm == 0.0) {
        out.m_actual = 0;
        return;
    }
    CUDA_CHECK(cudaMemcpy(out.d_V, d_v, (size_t)n * sizeof(double),
                          cudaMemcpyDeviceToDevice));
    double inv = 1.0 / vnorm;
    CUBLAS_CHECK(cublasDscal(ctx.cublas, n, &inv, out.d_V, 1));

    // Run m Lanczos steps with DGKS reorthogonalization, stopping
    // cleanly on invariant-subspace breakdown.
    out.reorth_count = 0;
    int steps = lanczos_extend_funm(ctx, op, out.d_V, n,
                                    out.alpha, out.beta,
                                    0, m, out.reorth_count);
    out.m_actual = steps;
}

// ============================================================
// Eigendecompose the tridiag T_m via LAPACK dstev.
//
// Output: eig_d (m,), eig_z (m × m, column-major) — host buffers
// pre-allocated by caller, both of length-m and m*m respectively.
// Returns LAPACK info (0 = success).
// ============================================================
inline int basis_eigendecompose_tridiag(const LanczosBasis &b,
                                        double *eig_d, double *eig_z) {
    int m = b.m_actual;
    return solve_tridiag(b.alpha, b.beta, m, eig_d, eig_z);
}

// ============================================================
// Quadratic form: v^T f(A) v ≈ ||v||² · sum_i (S[0,i])² · f(Θ_i)
//
// Used by SLQ. f_eig_d is the host array f(Θ_i) pre-computed by caller.
// ============================================================
inline double basis_quadratic_form(const LanczosBasis &b,
                                   const double *eig_d,
                                   const double *eig_z,
                                   const double *f_eig_d) {
    int m = b.m_actual;
    double acc = 0.0;
    for (int i = 0; i < m; i++) {
        double s0i = eig_z[0 + i * m];  // first row, i-th eigenvector
        acc += s0i * s0i * f_eig_d[i];
    }
    return b.v_norm * b.v_norm * acc;
}

// ============================================================
// Full application: y ≈ ||v|| · V · S · diag(f(Θ)) · S^T · e_1
//
// Algorithm on the small side (host, O(m^2)):
//   q_small[j] = sum_i S[j, i] · f(Θ_i) · S[0, i]
// Then on GPU (O(n·m) GEMV):
//   y = ||v|| · V · q_small
//
// d_y is a device pointer of length n.
// ============================================================
inline void basis_funm_apply(LanczosContext &ctx,
                             const LanczosBasis &b,
                             const double *eig_d,
                             const double *eig_z,
                             const double *f_eig_d,
                             double *d_y) {
    int n = b.n;
    int m = b.m_actual;

    // Small-side: q = S · diag(f(Θ)) · S^T · e_1
    std::vector<double> q(m, 0.0);
    for (int i = 0; i < m; i++) {
        double w = eig_z[0 + i * m] * f_eig_d[i];   // S[0,i] · f(Θ_i)
        for (int j = 0; j < m; j++)
            q[j] += eig_z[j + i * m] * w;
    }

    // y = ||v|| · V · q
    double *d_q;
    CUDA_CHECK(cudaMalloc(&d_q, (size_t)m * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_q, q.data(), (size_t)m * sizeof(double),
                          cudaMemcpyHostToDevice));
    double scale = b.v_norm;
    double zero  = 0.0;
    CUBLAS_CHECK(cublasDgemv(ctx.cublas, CUBLAS_OP_N, n, m,
                             &scale, b.d_V, n, d_q, 1, &zero, d_y, 1));
    cudaFree(d_q);
}

// ============================================================
// Apply a scalar function elementwise to eigenvalues.
//
// Accepts a short selector string: "log", "exp", "sqrt", "inv".
// Returns 0 on success, -1 if name unknown.
//
// log/sqrt/inv require positive eigenvalues; if any are non-positive
// the function returns -2 and the output array is left undefined.
// ============================================================
inline int apply_scalar_func(const char *name,
                             const double *eig_d, double *f_eig_d, int m) {
    if (strcmp(name, "log") == 0) {
        for (int i = 0; i < m; i++) {
            if (eig_d[i] <= 0.0) return -2;
            f_eig_d[i] = log(eig_d[i]);
        }
    } else if (strcmp(name, "exp") == 0) {
        for (int i = 0; i < m; i++) f_eig_d[i] = exp(eig_d[i]);
    } else if (strcmp(name, "sqrt") == 0) {
        for (int i = 0; i < m; i++) {
            if (eig_d[i] < 0.0) return -2;
            f_eig_d[i] = sqrt(eig_d[i]);
        }
    } else if (strcmp(name, "inv") == 0) {
        for (int i = 0; i < m; i++) {
            if (eig_d[i] == 0.0) return -2;
            f_eig_d[i] = 1.0 / eig_d[i];
        }
    } else {
        return -1;
    }
    return 0;
}
