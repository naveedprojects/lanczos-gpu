/**
 * lanczos_types.cuh — Core types, error macros, and numerical constants
 *                     for the GPU Lanczos eigensolver.
 *
 * This file contains no executable code — only type definitions and
 * compile-time constants used across the entire codebase.
 */

#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cfloat>
#include <vector>

// ============================================================
// Error-checking macros
// ============================================================
// These abort with file:line on any API failure. In a library
// these would return error codes; for a research prototype
// fail-fast is the right trade-off.

#define CUDA_CHECK(call) do {                                          \
    cudaError_t err = (call);                                          \
    if (err != cudaSuccess) {                                          \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",                  \
                __FILE__, __LINE__, cudaGetErrorString(err));          \
        exit(EXIT_FAILURE);                                            \
    }                                                                  \
} while (0)

#define CUBLAS_CHECK(call) do {                                        \
    cublasStatus_t st = (call);                                        \
    if (st != CUBLAS_STATUS_SUCCESS) {                                 \
        fprintf(stderr, "cuBLAS error at %s:%d: %d\n",                \
                __FILE__, __LINE__, (int)st);                          \
        exit(EXIT_FAILURE);                                            \
    }                                                                  \
} while (0)

#define CUSPARSE_CHECK(call) do {                                      \
    cusparseStatus_t st = (call);                                      \
    if (st != CUSPARSE_STATUS_SUCCESS) {                               \
        fprintf(stderr, "cuSPARSE error at %s:%d: %d\n",              \
                __FILE__, __LINE__, (int)st);                          \
        exit(EXIT_FAILURE);                                            \
    }                                                                  \
} while (0)

// ============================================================
// Numerical constants
// ============================================================

// DGKS threshold: 1/sqrt(2). If ||r|| / ||w|| drops below this
// after Gram-Schmidt, the residual has lost significant digits
// and reorthogonalization is needed.
// Reference: Daniel, Gragg, Kaufman, Stewart (1976).
static constexpr double DGKS_THRESHOLD = 0.7071067811865476;  // 1/sqrt(2)

// Smallest normalized double. Below this, 1/x may overflow.
// Used in safe_scale() to decide whether multi-step scaling is needed.
static constexpr double SAFE_MIN = DBL_MIN;  // 2.2250738585072014e-308

// Machine epsilon for double precision.
static constexpr double MACH_EPS = DBL_EPSILON;  // 2.2204460492503131e-16

// Maximum DGKS reorthogonalization passes per iteration.
// ARPACK uses 2; a third pass is never needed in practice because
// if two passes fail, the residual lies in span(V).
static constexpr int MAX_REORTH_PASSES = 2;

// ============================================================
// Sparse matrix in CSR format on GPU
// ============================================================

struct SparseMatrixCSR {
    int n   = 0;         // matrix dimension (n x n)
    int nnz = 0;         // number of nonzeros

    int    *d_row_ptr = nullptr;  // device: row pointers  [n + 1]
    int    *d_col_idx = nullptr;  // device: column indices [nnz]
    double *d_vals    = nullptr;  // device: values (FP64)  [nnz]
    float  *d_vals_f32 = nullptr; // device: values (FP32)  [nnz] (optional)

    cusparseSpMatDescr_t descr = nullptr;
    cusparseSpMatDescr_t descr_f32 = nullptr;  // FP32 descriptor (optional)
    bool descr_valid = false;
    bool descr_f32_valid = false;

    void create_descriptor() {
        CUSPARSE_CHECK(cusparseCreateCsr(&descr, n, n, nnz,
            d_row_ptr, d_col_idx, d_vals,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
        descr_valid = true;
    }

    // Create FP32 copy of values and FP32 descriptor for mixed-precision SpMV
    void create_f32_copy() {
        if (d_vals_f32) return;  // already created
        CUDA_CHECK(cudaMalloc(&d_vals_f32, nnz * sizeof(float)));
        // Convert FP64 -> FP32 on host (could do on device, but this is one-time)
        std::vector<double> h_vals(nnz);
        std::vector<float> h_vals_f32(nnz);
        CUDA_CHECK(cudaMemcpy(h_vals.data(), d_vals, nnz * sizeof(double), cudaMemcpyDeviceToHost));
        for (int i = 0; i < nnz; i++) h_vals_f32[i] = (float)h_vals[i];
        CUDA_CHECK(cudaMemcpy(d_vals_f32, h_vals_f32.data(), nnz * sizeof(float), cudaMemcpyHostToDevice));

        CUSPARSE_CHECK(cusparseCreateCsr(&descr_f32, n, n, nnz,
            d_row_ptr, d_col_idx, d_vals_f32,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
        descr_f32_valid = true;
    }

    void free() {
        if (d_row_ptr)  { cudaFree(d_row_ptr);  d_row_ptr  = nullptr; }
        if (d_col_idx)  { cudaFree(d_col_idx);  d_col_idx  = nullptr; }
        if (d_vals)     { cudaFree(d_vals);     d_vals     = nullptr; }
        if (d_vals_f32) { cudaFree(d_vals_f32); d_vals_f32 = nullptr; }
        if (descr_valid) {
            cusparseDestroySpMat(descr);
            descr = nullptr;
            descr_valid = false;
        }
        if (descr_f32_valid) {
            cusparseDestroySpMat(descr_f32);
            descr_f32 = nullptr;
            descr_f32_valid = false;
        }
    }
};

// ============================================================
// Lanczos result — returned by all algorithm variants
// ============================================================

struct LanczosResult {
    int n             = 0;   // matrix dimension
    int k             = 0;   // number of converged eigenvalues
    int num_iters     = 0;   // Lanczos iterations performed
    int num_reorths   = 0;   // reorthogonalization steps taken

    double *eigenvalues  = nullptr;  // host: converged eigenvalues  [k]
    double *eigenvectors = nullptr;  // host: Ritz vectors           [n * k], column-major
    double *ortho_loss   = nullptr;  // host: orthogonality samples  [ortho_loss_len]
    int ortho_loss_len   = 0;

    void free() {
        if (eigenvalues)  { ::free(eigenvalues);  eigenvalues  = nullptr; }
        if (eigenvectors) { ::free(eigenvectors); eigenvectors = nullptr; }
        if (ortho_loss)   { ::free(ortho_loss);   ortho_loss   = nullptr; }
    }
};

// ============================================================
// Algorithm configuration
// ============================================================

struct LanczosParams {
    int    num_eigs    = 20;     // desired eigenvalue count (nev)
    int    max_iters   = 1000;   // maximum Lanczos steps across all restarts
    int    measure_freq = 10;    // orthogonality measurement interval
    int    ncv         = 0;      // Krylov subspace size (0 = auto: 3*nev)
    double tol         = 1e-12;  // convergence tolerance for IRLM
    bool   store_basis = true;   // if false, only keep 3 vectors (for large n)
    bool   mixed_precision = false; // FP32 SpMV + FP64 reorthogonalization
};
