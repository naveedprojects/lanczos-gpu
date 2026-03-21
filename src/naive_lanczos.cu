/**
 * naive_lanczos.cu — Naive GPU Lanczos (no reorthogonalization)
 *
 * Implements the textbook three-term recurrence Lanczos algorithm.
 * This is what most GPU eigensolvers do — and why they produce
 * garbage eigenvalues for non-trivial problems.
 *
 * Included in the benchmark to demonstrate the catastrophic
 * orthogonality loss that motivates DGKS reorthogonalization
 * and implicit restarts.
 */

#include "lanczos_ops.cuh"
#include "tridiag.cuh"
#include <vector>

LanczosResult naive_lanczos(LanczosContext &ctx, SparseMatrixCSR &A,
                            const LanczosParams &params) {
    const int n = A.n;
    int max_iters = params.max_iters;
    if (max_iters > n) max_iters = n;

    // Allocate Lanczos basis V (n x max_iters), column-major
    double *d_V;
    CUDA_CHECK(cudaMalloc(&d_V, (size_t)n * max_iters * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_V, 0, (size_t)n * max_iters * sizeof(double)));

    // Tridiagonal matrix on host
    double *alpha = (double *)calloc(max_iters, sizeof(double));
    double *beta  = (double *)calloc(max_iters + 1, sizeof(double));

    // Orthogonality tracking
    int ortho_cap = max_iters / params.measure_freq + 2;
    double *ortho_loss = (double *)calloc(ortho_cap, sizeof(double));
    int ortho_idx = 0;

    // Deterministic starting vector
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

    int actual_iters = 0;

    for (int j = 0; j < max_iters; j++) {
        double *d_vj = d_V + (size_t)j * n;

        // w = A * v_j
        spmv(ctx, A, d_vj, ctx.d_w);

        // alpha_j = v_j^T * w
        CUBLAS_CHECK(cublasDdot(ctx.cublas, n, d_vj, 1, ctx.d_w, 1,
            &alpha[j]));

        // r = w - alpha_j * v_j
        CUDA_CHECK(cudaMemcpy(ctx.d_r, ctx.d_w, n * sizeof(double),
            cudaMemcpyDeviceToDevice));
        double neg_alpha = -alpha[j];
        CUBLAS_CHECK(cublasDaxpy(ctx.cublas, n, &neg_alpha, d_vj, 1,
            ctx.d_r, 1));

        // r = r - beta_j * v_{j-1}
        if (j > 0) {
            double neg_beta = -beta[j];
            double *d_vjm1 = d_V + (size_t)(j - 1) * n;
            CUBLAS_CHECK(cublasDaxpy(ctx.cublas, n, &neg_beta, d_vjm1, 1,
                ctx.d_r, 1));
        }

        // beta_{j+1} = ||r||
        double beta_next;
        CUBLAS_CHECK(cublasDnrm2(ctx.cublas, n, ctx.d_r, 1, &beta_next));

        double tol = adaptive_breakdown_tol(alpha, beta, j);
        if (beta_next < tol) {
            actual_iters = j + 1;
            break;
        }

        // v_{j+1} = r / beta_{j+1}
        if (j + 1 < max_iters) {
            beta[j + 1] = beta_next;
            double *d_vjp1 = d_V + (size_t)(j + 1) * n;
            CUDA_CHECK(cudaMemcpy(d_vjp1, ctx.d_r, n * sizeof(double),
                cudaMemcpyDeviceToDevice));
            safe_scale(ctx, d_vjp1, n, beta_next);
        }

        actual_iters = j + 1;

        // Periodic orthogonality measurement
        if ((j + 1) % params.measure_freq == 0 && ortho_idx < ortho_cap) {
            ortho_loss[ortho_idx++] = measure_orthogonality(ctx, d_V, n, j + 1);
        }
    }

    // Final orthogonality measurement
    if (ortho_idx < ortho_cap) {
        ortho_loss[ortho_idx++] = measure_orthogonality(ctx, d_V, n, actual_iters);
    }

    // Solve tridiagonal eigenproblem
    int m = actual_iters;
    int k = (params.num_eigs < m) ? params.num_eigs : m;

    double *eig_d = (double *)malloc(m * sizeof(double));
    double *eig_z = (double *)malloc((size_t)m * m * sizeof(double));
    solve_tridiag(alpha, beta, m, eig_d, eig_z);

    // Build result
    LanczosResult result;
    result.n = n;
    result.k = k;
    result.num_iters = actual_iters;
    result.num_reorths = 0;
    result.eigenvalues = (double *)malloc(k * sizeof(double));
    result.eigenvectors = (double *)malloc((size_t)n * k * sizeof(double));
    result.ortho_loss = ortho_loss;
    result.ortho_loss_len = ortho_idx;

    memcpy(result.eigenvalues, eig_d, k * sizeof(double));
    compute_ritz_vectors(ctx, d_V, n, m, eig_z, k, result.eigenvectors);

    // Cleanup (only algorithm-local allocations)
    cudaFree(d_V);
    ::free(alpha);
    ::free(beta);
    ::free(eig_d);
    ::free(eig_z);

    return result;
}
