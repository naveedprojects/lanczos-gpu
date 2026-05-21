/**
 * irlm_lanczos.cu — Implicitly Restarted Lanczos Method (IRLM) on GPU
 *
 * Full ARPACK-style eigensolver combining:
 *   - DGKS reorthogonalization (dsaitr.f)
 *   - Implicit QR restarts via Givens bulge chase (dsapps.f)
 *   - Exact shift selection (unwanted Ritz values, dsgets.f)
 *   - Convergence checking via Ritz estimates (dsconv.f)
 *   - Dynamic NEV adjustment for anti-stagnation (dsaup2.f)
 *
 * The algorithm is parameterized on MatVecOperator, so the same code
 * works with explicit CSR matrices, implicit operators, and spectral
 * transformations (shift-invert).
 *
 * References:
 *   Lehoucq, Sorensen, Yang: "ARPACK Users' Guide" (1998)
 *   Sorensen: "Implicit Application of Polynomial Filters" (1992)
 */

#include "lanczos_ops.cuh"
#include "matvec_operator.cuh"
#include "tridiag.cuh"
#include <vector>
#include <algorithm>

// ============================================================
// Rayleigh-Ritz refinement (operator-based version).
// Now that MatVecOperator is fully defined, we can implement it.
// ============================================================
static void rayleigh_ritz_refine_op_impl(LanczosContext &ctx,
                                         MatVecOperator &op,
                                         double *h_eigenvalues,
                                         double *h_eigenvectors,
                                         int n, int k) {
    double one = 1.0, zero = 0.0;

    double *d_X, *d_AX;
    CUDA_CHECK(cudaMalloc(&d_X,  (size_t)n * k * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_AX, (size_t)n * k * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_X, h_eigenvectors, (size_t)n * k * sizeof(double),
        cudaMemcpyHostToDevice));

    // AX = op(X), column by column
    for (int j = 0; j < k; j++) {
        op.apply(ctx, d_X + (size_t)j * n, d_AX + (size_t)j * n);
    }

    // H = X^T * AX  (k x k)
    double *d_H;
    CUDA_CHECK(cudaMalloc(&d_H, (size_t)k * k * sizeof(double)));
    CUBLAS_CHECK(cublasDgemm(ctx.cublas, CUBLAS_OP_T, CUBLAS_OP_N,
        k, k, n, &one, d_X, n, d_AX, n, &zero, d_H, k));

    std::vector<double> H(k * k), w(k);
    CUDA_CHECK(cudaMemcpy(H.data(), d_H, k * k * sizeof(double),
        cudaMemcpyDeviceToHost));
    cudaFree(d_H);

    // dsyev: eigendecompose H
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
        cudaFree(d_X);
        cudaFree(d_AX);
        return;
    }

    memcpy(h_eigenvalues, w.data(), k * sizeof(double));

    // X' = X * U
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

// LAPACK dsyev declaration (also used in rayleigh_ritz_refine in lanczos_ops.cuh)
extern "C" void dsyev_(const char *jobz, const char *uplo, const int *n,
                       double *a, const int *lda, double *w,
                       double *work, const int *lwork, int *info);

// ============================================================
// Run Lanczos steps [start, end) with DGKS reorthogonalization.
//
// Fills V columns start..end-1 and alpha[start..end-1].
// beta[j] = ||r|| before normalizing v_j.
// On exit: ctx.d_r = unnormalized residual, beta[end] = ||r||.
// ============================================================
void lanczos_extend(LanczosContext &ctx, MatVecOperator &op,
                    double *d_V, int n,
                    double *alpha, double *beta,
                    int start, int end, int &reorth_count) {
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

        // Handle invariant subspace
        if (rnorm < adaptive_breakdown_tol(alpha, beta, j)) {
            inject_random_restart(ctx, d_V, n, j, ctx.d_r, rnorm);
        }

        if (j + 1 < end) {
            beta[j + 1] = rnorm;
            double *d_vjp1 = d_V + (size_t)(j + 1) * n;
            CUDA_CHECK(cudaMemcpy(d_vjp1, ctx.d_r, n * sizeof(double),
                cudaMemcpyDeviceToDevice));
            if (rnorm > 0.0)
                safe_scale(ctx, d_vjp1, n, rnorm);
        }

        // Store beta for the last step (restart residual norm)
        if (j == end - 1)
            beta[end] = rnorm;
    }
}

// ============================================================
// IRLM: Implicitly Restarted Lanczos Method (operator version)
// ============================================================
LanczosResult irlm_lanczos(LanczosContext &ctx, MatVecOperator &op,
                           const LanczosParams &params) {
    const int n = op.n;
    const int nev_orig = params.num_eigs;
    int ncv = (params.ncv > 0) ? params.ncv
                               : std::min(std::max(3 * nev_orig, 40), n);
    if (ncv > n) ncv = n;
    const double tol = params.tol;

    printf("    IRLM: nev=%d, ncv=%d, np=%d, tol=%.1e\n",
           nev_orig, ncv, ncv - nev_orig, tol);

    // Lanczos basis V (n x (ncv+1)): ncv columns + 1 for restart residual
    double *d_V;
    CUDA_CHECK(cudaMalloc(&d_V, (size_t)n * (ncv + 1) * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_V, 0, (size_t)n * (ncv + 1) * sizeof(double)));

    // Tridiagonal on host
    double *alpha = (double *)calloc(ncv + 2, sizeof(double));
    double *beta  = (double *)calloc(ncv + 2, sizeof(double));

    // Pre-allocate restart workspace (worst-case sizes)
    double *d_Q, *d_Vnew;
    CUDA_CHECK(cudaMalloc(&d_Q,    (size_t)ncv * ncv * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Vnew, (size_t)n * ncv * sizeof(double)));

    // Orthogonality tracking
    int np_max = ncv - nev_orig;
    int ortho_cap = params.max_iters / (np_max > 0 ? np_max : 1) + 20;
    double *ortho_loss = (double *)calloc(ortho_cap, sizeof(double));
    int ortho_idx = 0;
    int num_reorths = 0;

    // Starting vector
    srand(42);
    std::vector<double> v0(n);
    double norm = 0.0;
    for (int i = 0; i < n; i++) {
        v0[i] = (double)rand() / RAND_MAX - 0.5;
        norm += v0[i] * v0[i];
    }
    norm = sqrt(norm);
    for (int i = 0; i < n; i++) v0[i] /= norm;
    CUDA_CHECK(cudaMemcpy(d_V, v0.data(), n * sizeof(double),
        cudaMemcpyHostToDevice));

    // Phase 1: Build initial ncv-step factorization
    lanczos_extend(ctx, op, d_V, n, alpha, beta, 0, ncv, num_reorths);
    int total_steps = ncv;

    // Store normalized residual as column ncv (for restart)
    {
        double bn = beta[ncv];
        if (bn > 0.0) {
            CUDA_CHECK(cudaMemcpy(d_V + (size_t)ncv * n, ctx.d_r,
                n * sizeof(double), cudaMemcpyDeviceToDevice));
            double inv = 1.0 / bn;
            CUBLAS_CHECK(cublasDscal(ctx.cublas, n, &inv,
                d_V + (size_t)ncv * n, 1));
        }
    }

    if (ortho_idx < ortho_cap)
        ortho_loss[ortho_idx++] = measure_orthogonality(ctx, d_V, n, ncv);

    // Host workspace for tridiagonal eigensolve and QR shifts
    std::vector<double> rd(ncv), re(ncv), rz(ncv * ncv), rwork(2 * ncv);
    std::vector<double> td(ncv), te(ncv - 1), Q(ncv * ncv);
    std::vector<double> shifts;

    int max_restarts = params.max_iters / (ncv - nev_orig) + 2;

    for (int restart = 0; restart < max_restarts; restart++) {
        // ---- Compute Ritz values ----
        memcpy(rd.data(), alpha, ncv * sizeof(double));
        for (int i = 0; i < ncv - 1; i++) re[i] = beta[i + 1];
        int m = ncv;
        char jobz = 'V';
        int info;
        dstev_(&jobz, &m, rd.data(), re.data(), rz.data(), &m,
               rwork.data(), &info);

        // ---- Convergence check ----
        double bn = beta[ncv];
        int nconv = 0;
        for (int i = 0; i < nev_orig; i++) {
            double bound = fabs(bn * rz[(ncv - 1) + i * ncv]);
            if (bound <= tol * fmax(fabs(rd[i]), SAFE_MIN))
                nconv++;
        }

        printf("    Restart %d: %d/%d converged (steps=%d, reorths=%d)\n",
               restart, nconv, nev_orig, total_steps, num_reorths);

        if (nconv >= nev_orig || total_steps >= params.max_iters) {
            // ---- Extract results ----
            int k = std::min(nev_orig, ncv);
            LanczosResult res;
            res.n = n;
            res.k = k;
            res.num_iters = total_steps;
            res.num_reorths = num_reorths;
            res.eigenvalues = (double *)malloc(k * sizeof(double));
            res.eigenvectors = (double *)malloc((size_t)n * k * sizeof(double));
            res.ortho_loss = ortho_loss;
            res.ortho_loss_len = ortho_idx;

            memcpy(res.eigenvalues, rd.data(), k * sizeof(double));
            compute_ritz_vectors(ctx, d_V, n, ncv, rz.data(), k,
                                 res.eigenvectors);

            cudaFree(d_V);
            cudaFree(d_Q);
            cudaFree(d_Vnew);
            ::free(alpha);
            ::free(beta);
            return res;
        }

        // ---- Dynamic NEV adjustment (anti-stagnation, dsaup2.f) ----
        int nev_use = nev_orig;
        int np = ncv - nev_use;
        if (nconv > 0 && nconv < nev_orig) {
            nev_use += std::min(nconv, np / 2);
            if (nev_use == 1 && ncv >= 6) nev_use = ncv / 2;
            else if (nev_use == 1 && ncv > 2) nev_use = 2;
            np = ncv - nev_use;
        }

        // ---- Select exact shifts (unwanted Ritz values) ----
        shifts.resize(np);
        for (int i = 0; i < np; i++)
            shifts[i] = rd[nev_use + i];
        std::sort(shifts.begin(), shifts.end(),
                  [](double a, double b) { return fabs(a) < fabs(b); });

        // ---- Apply QR shifts to tridiagonal ----
        memcpy(td.data(), alpha, ncv * sizeof(double));
        for (int i = 0; i < ncv - 1; i++) te[i] = beta[i + 1];

        apply_shifts_tridiag(td.data(), te.data(), ncv,
                             shifts.data(), np, Q.data());

        memcpy(alpha, td.data(), ncv * sizeof(double));
        for (int i = 0; i < ncv - 1; i++) beta[i + 1] = te[i];

        // ---- Update basis V on GPU ----
        // V_new(:, 0:nev_use) = V_old * Q(:, 0:nev_use)
        CUDA_CHECK(cudaMemcpy(d_Q, Q.data(),
            (size_t)ncv * (nev_use + 1) * sizeof(double),
            cudaMemcpyHostToDevice));

        double one = 1.0, zero = 0.0;
        CUBLAS_CHECK(cublasDgemm(ctx.cublas, CUBLAS_OP_N, CUBLAS_OP_N,
            n, nev_use + 1, ncv, &one, d_V, n, d_Q, ncv,
            &zero, d_Vnew, n));

        // ---- Construct restart residual (ARPACK dsapps.f) ----
        // r_new = beta_ncv * Q(ncv-1, nev_use-1) * f
        //       + e[nev_use-1] * V_new(:, nev_use)
        double sigma_p = bn * Q[(ncv - 1) + (nev_use - 1) * ncv];
        double tau = (nev_use < ncv) ? te[nev_use - 1] : 0.0;

        CUDA_CHECK(cudaMemcpy(ctx.d_r, d_V + (size_t)ncv * n,
            n * sizeof(double), cudaMemcpyDeviceToDevice));
        CUBLAS_CHECK(cublasDscal(ctx.cublas, n, &sigma_p, ctx.d_r, 1));
        CUBLAS_CHECK(cublasDaxpy(ctx.cublas, n, &tau,
            d_Vnew + (size_t)nev_use * n, 1, ctx.d_r, 1));

        // Copy updated basis back
        CUDA_CHECK(cudaMemcpy(d_V, d_Vnew,
            (size_t)n * nev_use * sizeof(double),
            cudaMemcpyDeviceToDevice));

        // Normalize restart residual into column nev_use
        double rnorm;
        CUBLAS_CHECK(cublasDnrm2(ctx.cublas, n, ctx.d_r, 1, &rnorm));
        beta[nev_use] = rnorm;

        if (rnorm > 0.0) {
            CUDA_CHECK(cudaMemcpy(d_V + (size_t)nev_use * n, ctx.d_r,
                n * sizeof(double), cudaMemcpyDeviceToDevice));
            double inv = 1.0 / rnorm;
            CUBLAS_CHECK(cublasDscal(ctx.cublas, n, &inv,
                d_V + (size_t)nev_use * n, 1));
        }

        // Clear remaining alpha/beta
        for (int i = nev_use; i < ncv; i++) alpha[i] = 0.0;
        for (int i = nev_use + 1; i <= ncv + 1; i++) beta[i] = 0.0;

        // ---- Continue Lanczos from nev_use to ncv ----
        lanczos_extend(ctx, op, d_V, n, alpha, beta,
                       nev_use, ncv, num_reorths);
        total_steps += (ncv - nev_use);

        // Store normalized residual for next restart
        {
            double bn2 = beta[ncv];
            if (bn2 > 0.0) {
                CUDA_CHECK(cudaMemcpy(d_V + (size_t)ncv * n, ctx.d_r,
                    n * sizeof(double), cudaMemcpyDeviceToDevice));
                double inv = 1.0 / bn2;
                CUBLAS_CHECK(cublasDscal(ctx.cublas, n, &inv,
                    d_V + (size_t)ncv * n, 1));
            }
        }

        if (ortho_idx < ortho_cap)
            ortho_loss[ortho_idx++] = measure_orthogonality(ctx, d_V, n, ncv);
    }

    // Fallback: return best-so-far Ritz values
    int k = std::min(nev_orig, ncv);
    LanczosResult res;
    res.n = n;
    res.k = k;
    res.num_iters = total_steps;
    res.num_reorths = num_reorths;
    res.eigenvalues = (double *)malloc(k * sizeof(double));
    res.eigenvectors = (double *)malloc((size_t)n * k * sizeof(double));
    res.ortho_loss = ortho_loss;
    res.ortho_loss_len = ortho_idx;

    // Compute final Ritz values from current tridiagonal
    std::vector<double> fd(ncv), fe(ncv), fz(ncv * ncv), fw(2 * ncv);
    memcpy(fd.data(), alpha, ncv * sizeof(double));
    for (int i = 0; i < ncv - 1; i++) fe[i] = beta[i + 1];
    int m2 = ncv;
    char jz = 'V';
    int info2;
    dstev_(&jz, &m2, fd.data(), fe.data(), fz.data(), &m2, fw.data(), &info2);
    memcpy(res.eigenvalues, fd.data(), k * sizeof(double));
    compute_ritz_vectors(ctx, d_V, n, ncv, fz.data(), k, res.eigenvectors);

    cudaFree(d_V);
    cudaFree(d_Q);
    cudaFree(d_Vnew);
    ::free(alpha);
    ::free(beta);
    return res;
}

// ============================================================
// Backward-compatible overload: accepts SparseMatrixCSR directly.
//
// Constructs the appropriate operator (CSR or mixed-precision)
// and delegates to the operator-based implementation.
// ============================================================
LanczosResult irlm_lanczos(LanczosContext &ctx, SparseMatrixCSR &A,
                           const LanczosParams &params) {
    if (params.mixed_precision) {
        A.create_f32_copy();
        CSRMixedOperator op(A);
        return irlm_lanczos(ctx, op, params);
    } else {
        CSROperator op(A);
        return irlm_lanczos(ctx, op, params);
    }
}
