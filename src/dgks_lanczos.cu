/**
 * dgks_lanczos.cu — GPU Lanczos with DGKS reorthogonalization
 *
 * Implements the Lanczos algorithm with the Daniel-Gragg-Kaufman-Stewart
 * conditional reorthogonalization criterion, faithfully porting the
 * numerical safeguards from ARPACK's dsaitr.f to GPU.
 *
 * ARPACK safeguards implemented:
 *   1. DGKS 0.717 test for conditional reorthogonalization
 *   2. Iterative refinement (up to 2 reorthogonalization passes)
 *   3. Invariant subspace detection and restart with random vector
 *   4. Multi-step safe scaling near underflow
 *
 * Uses cuSPARSE for SpMV, cuBLAS for dense operations via shared
 * LanczosContext. No GPU memory allocated in any loop.
 */

#include "lanczos_ops.cuh"
#include "tridiag.cuh"
#include <vector>

LanczosResult dgks_lanczos(LanczosContext &ctx, SparseMatrixCSR &A,
                           const LanczosParams &params) {
    const int n = A.n;
    int max_iters = params.max_iters;
    if (max_iters > n) max_iters = n;

    // Lanczos basis V (n x max_iters), column-major on GPU
    double *d_V;
    CUDA_CHECK(cudaMalloc(&d_V, (size_t)n * max_iters * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_V, 0, (size_t)n * max_iters * sizeof(double)));

    // Tridiagonal on host
    double *alpha = (double *)calloc(max_iters, sizeof(double));
    double *beta  = (double *)calloc(max_iters + 1, sizeof(double));

    // Orthogonality tracking
    int ortho_cap = max_iters / params.measure_freq + 2;
    double *ortho_loss = (double *)calloc(ortho_cap, sizeof(double));
    int ortho_idx = 0;
    int num_reorths = 0;

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

        // Step 1: w = A * v_j
        spmv(ctx, A, d_vj, ctx.d_w);

        // ||w|| for the DGKS test
        double wnorm;
        CUBLAS_CHECK(cublasDnrm2(ctx.cublas, n, ctx.d_w, 1, &wnorm));

        // Step 2: Classical Gram-Schmidt
        double alpha_j;
        cgs_orthogonalize(ctx, d_V, n, j, ctx.d_w, ctx.d_r, alpha_j);
        alpha[j] = alpha_j;

        double rnorm;
        CUBLAS_CHECK(cublasDnrm2(ctx.cublas, n, ctx.d_r, 1, &rnorm));

        // Step 3: DGKS conditional reorthogonalization
        dgks_reorth(ctx, d_V, n, j, ctx.d_r, alpha[j], rnorm, wnorm,
                    num_reorths);

        // Handle invariant subspace (rnorm ≈ 0 after DGKS)
        if (rnorm < adaptive_breakdown_tol(alpha, beta, j)) {
            inject_random_restart(ctx, d_V, n, j, ctx.d_r, rnorm);
            if (rnorm < adaptive_breakdown_tol(alpha, beta, j)) {
                actual_iters = j + 1;
                break;  // True invariant subspace — stop
            }
        }

        // Step 4: Normalize and store v_{j+1}
        if (j + 1 < max_iters) {
            beta[j + 1] = rnorm;
            double *d_vjp1 = d_V + (size_t)(j + 1) * n;
            CUDA_CHECK(cudaMemcpy(d_vjp1, ctx.d_r, n * sizeof(double),
                cudaMemcpyDeviceToDevice));
            safe_scale(ctx, d_vjp1, n, rnorm);
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
    result.num_reorths = num_reorths;
    result.eigenvalues = (double *)malloc(k * sizeof(double));
    result.eigenvectors = (double *)malloc((size_t)n * k * sizeof(double));
    result.ortho_loss = ortho_loss;
    result.ortho_loss_len = ortho_idx;

    memcpy(result.eigenvalues, eig_d, k * sizeof(double));
    compute_ritz_vectors(ctx, d_V, n, m, eig_z, k, result.eigenvectors);

    // Cleanup
    cudaFree(d_V);
    ::free(alpha);
    ::free(beta);
    ::free(eig_d);
    ::free(eig_z);

    return result;
}
