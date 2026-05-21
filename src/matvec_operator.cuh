/**
 * matvec_operator.cuh — Matrix-vector product abstraction for GPU Lanczos.
 *
 * Defines the MatVecOperator interface that decouples the Lanczos algorithm
 * from any particular matrix representation. The algorithm only sees:
 *   apply(ctx, x, y)   =>   y = A * x
 *
 * Implementations:
 *   CSROperator         — cuSPARSE SpMV on an explicit CSR matrix
 *   CSRMixedOperator    — FP32 SpMV with FP64 Krylov basis (2x bandwidth)
 *   FunctionOperator    — User-supplied function pointer (matrix-free)
 *   ShiftInvertOperator — (A - sigma*I)^{-1} via deflated CG
 *
 * This design allows the same IRLM code to work with explicit sparse
 * matrices, implicit operators (e.g. graph Laplacians parameterized by
 * bandwidth), and spectral transformations (shift-invert).
 */

#pragma once

#include "lanczos_context.cuh"

// ============================================================
// Abstract base: any callable that maps x -> y on GPU.
//
// Subclasses must implement apply(). apply_shifted() has a
// default that computes y = A*x - sigma*x, overridable for
// operators that can fuse the shift.
// ============================================================
struct MatVecOperator {
    int n = 0;

    virtual ~MatVecOperator() {}

    // y = A * x  (both device pointers, length n)
    virtual void apply(LanczosContext &ctx,
                       const double *d_x, double *d_y) = 0;

    // y = (A - sigma*I) * x
    // Default: apply then axpy. Override for fused implementations.
    virtual void apply_shifted(LanczosContext &ctx,
                               const double *d_x, double *d_y,
                               double sigma) {
        apply(ctx, d_x, d_y);
        double neg_sigma = -sigma;
        CUBLAS_CHECK(cublasDaxpy(ctx.cublas, n, &neg_sigma, d_x, 1, d_y, 1));
    }
};

// ============================================================
// CSR operator: y = A * x via cuSPARSE.
//
// Reuses the context's pre-allocated descriptors and grow-only
// SpMV buffer. Zero allocations per call.
// ============================================================
struct CSROperator : MatVecOperator {
    SparseMatrixCSR &A;

    explicit CSROperator(SparseMatrixCSR &A_) : A(A_) { n = A_.n; }

    void apply(LanczosContext &ctx,
               const double *d_x, double *d_y) override {
        double alpha = 1.0, beta = 0.0;

        CUSPARSE_CHECK(cusparseDnVecSetValues(ctx.vec_x, (void *)d_x));
        CUSPARSE_CHECK(cusparseDnVecSetValues(ctx.vec_y, d_y));

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
};

// ============================================================
// Mixed-precision CSR operator: y_f64 = A_f32 * x_f64.
//
// Runs SpMV in FP32 for ~2x memory bandwidth savings. The ~10^-7
// FP32 error is corrected by DGKS reorthogonalization.
// Requires A.create_f32_copy() called beforehand.
// ============================================================
extern __global__ void kernel_double_to_float(const double *in, float *out, int n);
extern __global__ void kernel_float_to_double(const float *in, double *out, int n);

struct CSRMixedOperator : MatVecOperator {
    SparseMatrixCSR &A;

    explicit CSRMixedOperator(SparseMatrixCSR &A_) : A(A_) { n = A_.n; }

    void apply(LanczosContext &ctx,
               const double *d_x, double *d_y) override {
        int blocks = (n + 255) / 256;

        // Cast x: FP64 -> FP32
        kernel_double_to_float<<<blocks, 256, 0, ctx.compute_stream>>>(
            d_x, ctx.d_x_f32, n);

        // SpMV in FP32
        float alpha = 1.0f, beta = 0.0f;
        CUSPARSE_CHECK(cusparseDnVecSetValues(ctx.vec_x_f32, ctx.d_x_f32));
        CUSPARSE_CHECK(cusparseDnVecSetValues(ctx.vec_y_f32, ctx.d_w_f32));

        size_t needed = 0;
        CUSPARSE_CHECK(cusparseSpMV_bufferSize(ctx.cusparse,
            CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, A.descr_f32,
            ctx.vec_x_f32, &beta, ctx.vec_y_f32, CUDA_R_32F,
            CUSPARSE_SPMV_ALG_DEFAULT, &needed));

        if (needed > ctx.spmv_buffer_f32_size) {
            if (ctx.spmv_buffer_f32) cudaFree(ctx.spmv_buffer_f32);
            CUDA_CHECK(cudaMalloc(&ctx.spmv_buffer_f32, needed));
            ctx.spmv_buffer_f32_size = needed;
        }

        CUSPARSE_CHECK(cusparseSpMV(ctx.cusparse,
            CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, A.descr_f32,
            ctx.vec_x_f32, &beta, ctx.vec_y_f32, CUDA_R_32F,
            CUSPARSE_SPMV_ALG_DEFAULT, ctx.spmv_buffer_f32));

        // Cast result: FP32 -> FP64
        kernel_float_to_double<<<blocks, 256, 0, ctx.compute_stream>>>(
            ctx.d_w_f32, d_y, n);
    }
};

// ============================================================
// Function pointer operator: user-supplied matrix-free matvec.
//
// The callback receives raw device pointers and is responsible
// for computing y = A(x). The user_data pointer is passed through
// opaquely for the callback to access its own state.
//
// This enables:
//   - Graph Laplacians parameterized by bandwidth
//   - Kernel matrices via on-the-fly computation
//   - Any implicit operator from Python via ctypes/pybind11
// ============================================================
typedef void (*matvec_fn_t)(const double *d_x, double *d_y,
                            int n, void *user_data);

struct FunctionOperator : MatVecOperator {
    matvec_fn_t fn;
    void *user_data;

    FunctionOperator(matvec_fn_t fn_, int n_, void *user_data_ = nullptr)
        : fn(fn_), user_data(user_data_) { n = n_; }

    void apply(LanczosContext &ctx,
               const double *d_x, double *d_y) override {
        fn(d_x, d_y, n, user_data);
    }
};

// ShiftInvertOperator is defined in inner_solve.cuh
// (requires full CGWorkspace definition, not just forward decl)
