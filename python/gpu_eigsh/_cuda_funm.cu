/**
 * _cuda_funm.cu — Python entry points for matrix function evaluation
 *                  via Lanczos quadrature.
 *
 * Entry points:
 *   compute_funm_apply()   — f(A) v approximated by m-step Lanczos
 *   compute_funm_qform()   — v^T f(A) v approximated by m-step Lanczos
 *
 * Both accept a CSR matrix and apply a named function:
 *   "log", "exp", "sqrt", "inv"
 *
 * No restart, no autograd, no batching — this is the unrestarted
 * forward primitive. Adjoint pass and SLQ trace estimator will be
 * built on top.
 */

#include "lanczos_ops.cuh"
#include "matvec_operator.cuh"
#include "matrix_function.cuh"
#include <vector>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <dlfcn.h>

// NOTE: single-threaded MKL pinning is done from the Python side
// (gpu_eigsh/__init__.py via ctypes) before this extension is imported.
// libmkl_rt with multi-threaded dstev races silently when the process
// has also initialized CUDA. See gpu_eigsh/__init__.py for the fix.

// ============================================================
// y = f(A) v   via m-step Lanczos quadrature.
//
// Returns: m_actual (the actual Lanczos depth used, may be < m
//          on breakdown). Negative values are error codes:
//           -1: unknown function name
//           -2: invalid eigenvalue for chosen function
//           -3: input vector is zero
// ============================================================
extern "C" int compute_funm_apply(
    int n, int nnz,
    const int *row_ptr, const int *col_idx, const double *vals,
    const double *v_host, int m,
    const char *func_name,
    double *out_y)
{
    // Upload CSR
    SparseMatrixCSR A;
    A.n = n;
    A.nnz = nnz;
    CUDA_CHECK(cudaMalloc(&A.d_row_ptr, (size_t)(n + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&A.d_col_idx, (size_t)nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&A.d_vals,    (size_t)nnz * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(A.d_row_ptr, row_ptr, (size_t)(n + 1) * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(A.d_col_idx, col_idx, (size_t)nnz * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(A.d_vals,    vals,    (size_t)nnz * sizeof(double),
                          cudaMemcpyHostToDevice));
    A.create_descriptor();

    LanczosContext ctx;
    ctx.init(n, m + 4);

    CSROperator op(A);

    // Upload v
    double *d_v;
    CUDA_CHECK(cudaMalloc(&d_v, (size_t)n * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_v, v_host, (size_t)n * sizeof(double),
                          cudaMemcpyHostToDevice));

    // Build Lanczos basis
    LanczosBasis basis;
    lanczos_build_basis(ctx, op, d_v, m, basis);

    if (basis.m_actual == 0 || basis.v_norm == 0.0) {
        memset(out_y, 0, (size_t)n * sizeof(double));
        basis.free_resources();
        cudaFree(d_v);
        ctx.destroy();
        A.free();
        return -3;
    }

    // Eigendecompose T_m via LAPACK dstev.
    int m_a = basis.m_actual;
    std::vector<double> eig_d(m_a), eig_z((size_t)m_a * m_a);
    int info = basis_eigendecompose_tridiag(basis, eig_d.data(), eig_z.data());
    if (info != 0) {
        basis.free_resources();
        cudaFree(d_v);
        ctx.destroy();
        A.free();
        return -4;
    }

    // Apply f to eigenvalues
    std::vector<double> f_eig_d(m_a);
    int frc = apply_scalar_func(func_name, eig_d.data(), f_eig_d.data(), m_a);
    if (frc != 0) {
        basis.free_resources();
        cudaFree(d_v);
        ctx.destroy();
        A.free();
        return frc;
    }

    // Compute y on device, copy to host
    double *d_y;
    CUDA_CHECK(cudaMalloc(&d_y, (size_t)n * sizeof(double)));
    basis_funm_apply(ctx, basis, eig_d.data(), eig_z.data(),
                     f_eig_d.data(), d_y);

    CUDA_CHECK(cudaMemcpy(out_y, d_y, (size_t)n * sizeof(double),
                          cudaMemcpyDeviceToHost));

    cudaFree(d_y);
    cudaFree(d_v);
    basis.free_resources();
    ctx.destroy();
    A.free();
    return m_a;
}

// ============================================================
// Testing/diagnostic entry — return the raw Lanczos basis + alpha + beta
// produced by the forward primitive. Used by gradcheck tests in
// `tests/test_funm.py` to verify that the basis is orthonormal and that
// V^T A V == T. Not used by production code paths.
// ============================================================
extern "C" int compute_funm_debug(
    int n, int nnz,
    const int *row_ptr, const int *col_idx, const double *vals,
    const double *v_host, int m,
    double *out_V,      // host buffer (n × m, column-major)
    double *out_alpha,  // host buffer (m,)
    double *out_beta)   // host buffer (m+1,)
{
    SparseMatrixCSR A;
    A.n = n;
    A.nnz = nnz;
    CUDA_CHECK(cudaMalloc(&A.d_row_ptr, (size_t)(n + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&A.d_col_idx, (size_t)nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&A.d_vals,    (size_t)nnz * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(A.d_row_ptr, row_ptr, (size_t)(n + 1) * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(A.d_col_idx, col_idx, (size_t)nnz * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(A.d_vals,    vals,    (size_t)nnz * sizeof(double),
                          cudaMemcpyHostToDevice));
    A.create_descriptor();

    LanczosContext ctx;
    ctx.init(n, m + 4);
    CSROperator op(A);

    double *d_v;
    CUDA_CHECK(cudaMalloc(&d_v, (size_t)n * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_v, v_host, (size_t)n * sizeof(double),
                          cudaMemcpyHostToDevice));

    LanczosBasis basis;
    lanczos_build_basis(ctx, op, d_v, m, basis);

    int ma = basis.m_actual;
    CUDA_CHECK(cudaMemcpy(out_V, basis.d_V,
                          (size_t)n * ma * sizeof(double),
                          cudaMemcpyDeviceToHost));
    memcpy(out_alpha, basis.alpha, ma * sizeof(double));
    memcpy(out_beta,  basis.beta,  (ma + 1) * sizeof(double));

    cudaFree(d_v);
    basis.free_resources();
    ctx.destroy();
    A.free();
    return ma;
}

extern "C" int compute_funm_qform(
    int n, int nnz,
    const int *row_ptr, const int *col_idx, const double *vals,
    const double *v_host, int m,
    const char *func_name,
    double *out_value)
{
    SparseMatrixCSR A;
    A.n = n;
    A.nnz = nnz;
    CUDA_CHECK(cudaMalloc(&A.d_row_ptr, (size_t)(n + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&A.d_col_idx, (size_t)nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&A.d_vals,    (size_t)nnz * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(A.d_row_ptr, row_ptr, (size_t)(n + 1) * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(A.d_col_idx, col_idx, (size_t)nnz * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(A.d_vals,    vals,    (size_t)nnz * sizeof(double),
                          cudaMemcpyHostToDevice));
    A.create_descriptor();

    LanczosContext ctx;
    ctx.init(n, m + 4);
    CSROperator op(A);

    double *d_v;
    CUDA_CHECK(cudaMalloc(&d_v, (size_t)n * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_v, v_host, (size_t)n * sizeof(double),
                          cudaMemcpyHostToDevice));

    LanczosBasis basis;
    lanczos_build_basis(ctx, op, d_v, m, basis);

    if (basis.m_actual == 0 || basis.v_norm == 0.0) {
        *out_value = 0.0;
        basis.free_resources();
        cudaFree(d_v);
        ctx.destroy();
        A.free();
        return -3;
    }

    int m_a = basis.m_actual;
    std::vector<double> eig_d(m_a), eig_z((size_t)m_a * m_a);
    int info = basis_eigendecompose_tridiag(basis, eig_d.data(), eig_z.data());
    if (info != 0) {
        basis.free_resources();
        cudaFree(d_v);
        ctx.destroy();
        A.free();
        return -4;
    }

    std::vector<double> f_eig_d(m_a);
    int frc = apply_scalar_func(func_name, eig_d.data(), f_eig_d.data(), m_a);
    if (frc != 0) {
        basis.free_resources();
        cudaFree(d_v);
        ctx.destroy();
        A.free();
        return frc;
    }

    *out_value = basis_quadratic_form(basis, eig_d.data(), eig_z.data(),
                                      f_eig_d.data());

    cudaFree(d_v);
    basis.free_resources();
    ctx.destroy();
    A.free();
    return m_a;
}
