/**
 * lanczos_context.cuh — GPU resource context for the Lanczos eigensolver.
 *
 * Owns all GPU resources shared across algorithm variants:
 *   - cuBLAS / cuSPARSE handles (created once, reused)
 *   - CUDA streams for overlapping compute and transfers
 *   - Pinned host memory for scalar readbacks (alpha, beta, norms)
 *   - Pre-allocated device work buffers (no malloc in inner loops)
 *   - Reusable cuSPARSE dense vector descriptors
 *   - cudaEvent pair for accurate GPU timing
 *
 * Usage:
 *   LanczosContext ctx;
 *   ctx.init(n, max_basis_size);
 *   ...run algorithms with ctx...
 *   ctx.destroy();
 */

#pragma once

#include "lanczos_types.cuh"

// Indices into h_scalars pinned buffer
enum ScalarSlot : int {
    SLOT_ALPHA   = 0,
    SLOT_WNORM   = 1,
    SLOT_RNORM   = 2,
    SLOT_CORR    = 3,
    SLOT_RNORM1  = 4,
    SLOT_SPARE0  = 5,
    SLOT_SPARE1  = 6,
    SLOT_SPARE2  = 7,
    NUM_SCALAR_SLOTS = 8
};

struct LanczosContext {
    // ---- Library handles ----
    cublasHandle_t   cublas   = nullptr;
    cusparseHandle_t cusparse = nullptr;

    // ---- CUDA streams ----
    cudaStream_t compute_stream  = nullptr;  // SpMV, GEMV, GEMM
    cudaStream_t transfer_stream = nullptr;  // async scalar H<->D

    // ---- Timing events ----
    cudaEvent_t ev_start = nullptr;
    cudaEvent_t ev_stop  = nullptr;

    // ---- Pinned host memory for scalar transfers ----
    // Avoids implicit GPU synchronization on every cublasDnrm2 readback.
    double *h_scalars = nullptr;

    // ---- Pre-allocated device work buffers ----
    double *d_w      = nullptr;  // SpMV output / general work vector  [n]
    double *d_r      = nullptr;  // residual vector                    [n]
    double *d_coeffs = nullptr;  // CGS / reorth coefficients          [max_basis]

    // ---- Mixed-precision work buffers ----
    float  *d_x_f32  = nullptr;  // FP32 input for mixed SpMV  [n]
    float  *d_w_f32  = nullptr;  // FP32 output for mixed SpMV [n]
    cusparseDnVecDescr_t vec_x_f32 = nullptr;
    cusparseDnVecDescr_t vec_y_f32 = nullptr;
    bool vec_f32_valid = false;
    void *spmv_buffer_f32      = nullptr;
    size_t spmv_buffer_f32_size = 0;

    // ---- Orthogonality measurement buffer ----
    // G = V^T V is at most max_basis x max_basis.
    double *d_G      = nullptr;  // device: Gram matrix                [max_basis^2]
    double *h_G      = nullptr;  // pinned host: Gram matrix copy

    // ---- SpMV buffer (grow-only) ----
    void  *spmv_buffer      = nullptr;
    size_t spmv_buffer_size = 0;

    // ---- Reusable cuSPARSE dense vector descriptors ----
    cusparseDnVecDescr_t vec_x = nullptr;
    cusparseDnVecDescr_t vec_y = nullptr;
    bool vec_descriptors_valid  = false;

    // ---- Dimensions ----
    int n              = 0;
    int max_basis_size = 0;

    // ============================================================
    // Initialize all GPU resources.
    //   n:              problem dimension (matrix is n x n)
    //   max_basis_size: maximum number of Lanczos vectors stored
    //                   (= max_iters for DGKS, = ncv for IRLM)
    // ============================================================
    void init(int n_, int max_basis_) {
        n = n_;
        max_basis_size = max_basis_;

        // Library handles
        CUBLAS_CHECK(cublasCreate(&cublas));
        CUSPARSE_CHECK(cusparseCreate(&cusparse));

        // Streams
        CUDA_CHECK(cudaStreamCreate(&compute_stream));
        CUDA_CHECK(cudaStreamCreate(&transfer_stream));

        // Bind handles to compute stream
        CUBLAS_CHECK(cublasSetStream(cublas, compute_stream));
        CUSPARSE_CHECK(cusparseSetStream(cusparse, compute_stream));

        // Timing events
        CUDA_CHECK(cudaEventCreate(&ev_start));
        CUDA_CHECK(cudaEventCreate(&ev_stop));

        // Pinned host scalars
        CUDA_CHECK(cudaMallocHost(&h_scalars, NUM_SCALAR_SLOTS * sizeof(double)));
        memset(h_scalars, 0, NUM_SCALAR_SLOTS * sizeof(double));

        // Device work buffers
        CUDA_CHECK(cudaMalloc(&d_w,      n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_r,      n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_coeffs, max_basis_size * sizeof(double)));

        // Orthogonality measurement buffers
        size_t gram_bytes = (size_t)max_basis_size * max_basis_size * sizeof(double);
        CUDA_CHECK(cudaMalloc(&d_G, gram_bytes));
        CUDA_CHECK(cudaMallocHost(&h_G, gram_bytes));

        // cuSPARSE dense vector descriptors (will be updated with cusparseDnVecSetValues)
        CUSPARSE_CHECK(cusparseCreateDnVec(&vec_x, n, d_w, CUDA_R_64F));
        CUSPARSE_CHECK(cusparseCreateDnVec(&vec_y, n, d_r, CUDA_R_64F));
        vec_descriptors_valid = true;

        // Mixed-precision FP32 buffers
        CUDA_CHECK(cudaMalloc(&d_x_f32, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_w_f32, n * sizeof(float)));
        CUSPARSE_CHECK(cusparseCreateDnVec(&vec_x_f32, n, d_x_f32, CUDA_R_32F));
        CUSPARSE_CHECK(cusparseCreateDnVec(&vec_y_f32, n, d_w_f32, CUDA_R_32F));
        vec_f32_valid = true;
    }

    // ============================================================
    // Start GPU timer
    // ============================================================
    void timer_start() {
        CUDA_CHECK(cudaEventRecord(ev_start, compute_stream));
    }

    // ============================================================
    // Stop GPU timer and return elapsed milliseconds
    // ============================================================
    float timer_stop() {
        CUDA_CHECK(cudaEventRecord(ev_stop, compute_stream));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
        return ms;
    }

    // ============================================================
    // Release all GPU resources.
    // ============================================================
    void destroy() {
        if (vec_descriptors_valid) {
            cusparseDestroyDnVec(vec_x); vec_x = nullptr;
            cusparseDestroyDnVec(vec_y); vec_y = nullptr;
            vec_descriptors_valid = false;
        }

        if (vec_f32_valid) {
            cusparseDestroyDnVec(vec_x_f32); vec_x_f32 = nullptr;
            cusparseDestroyDnVec(vec_y_f32); vec_y_f32 = nullptr;
            vec_f32_valid = false;
        }
        if (spmv_buffer_f32) { cudaFree(spmv_buffer_f32); spmv_buffer_f32 = nullptr; spmv_buffer_f32_size = 0; }
        if (d_x_f32) { cudaFree(d_x_f32); d_x_f32 = nullptr; }
        if (d_w_f32) { cudaFree(d_w_f32); d_w_f32 = nullptr; }
        if (spmv_buffer) { cudaFree(spmv_buffer); spmv_buffer = nullptr; spmv_buffer_size = 0; }
        if (d_w)      { cudaFree(d_w);      d_w      = nullptr; }
        if (d_r)      { cudaFree(d_r);      d_r      = nullptr; }
        if (d_coeffs) { cudaFree(d_coeffs); d_coeffs = nullptr; }
        if (d_G)      { cudaFree(d_G);      d_G      = nullptr; }
        if (h_G)      { cudaFreeHost(h_G);  h_G      = nullptr; }
        if (h_scalars){ cudaFreeHost(h_scalars); h_scalars = nullptr; }

        if (ev_start) { cudaEventDestroy(ev_start); ev_start = nullptr; }
        if (ev_stop)  { cudaEventDestroy(ev_stop);  ev_stop  = nullptr; }

        if (transfer_stream) { cudaStreamDestroy(transfer_stream); transfer_stream = nullptr; }
        if (compute_stream)  { cudaStreamDestroy(compute_stream);  compute_stream  = nullptr; }

        if (cusparse) { cusparseDestroy(cusparse); cusparse = nullptr; }
        if (cublas)   { cublasDestroy(cublas);     cublas   = nullptr; }

        n = 0;
        max_basis_size = 0;
    }
};
