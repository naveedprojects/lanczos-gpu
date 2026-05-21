/**
 * _bindings.cpp — pybind11 bindings for gpu_eigsh.
 *
 * Wraps the CUDA compute functions to accept/return numpy arrays:
 *   - _gpu_eigsh_raw:         Standard eigensolver (k smallest)
 *   - _gpu_eigsh_sigma_raw:   Shift-invert (k nearest to sigma)
 *   - _gpu_adjoint_eigsh_raw: Backward pass for differentiable eigsh
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <tuple>
#include <stdexcept>

namespace py = pybind11;

// Declared in _cuda_eigsh.cu
extern "C" int compute_eigsh(
    int n, int nnz,
    const int *row_ptr, const int *col_idx, const double *vals,
    int k, int ncv, int max_iters, double tol,
    double *out_eigenvalues, double *out_eigenvectors);

extern "C" int compute_eigsh_sigma(
    int n, int nnz,
    const int *row_ptr, const int *col_idx, const double *vals,
    int k, int ncv, int max_iters, double tol, double sigma,
    int cg_max_iters, double cg_tol,
    double *out_eigenvalues, double *out_eigenvectors);

extern "C" int compute_adjoint_eigsh(
    int n, int nnz,
    const int *row_ptr, const int *col_idx, const double *vals,
    int k,
    const double *eigenvalues,
    const double *eigenvectors,
    const double *grad_evals,
    const double *grad_evecs,
    double *grad_vals,
    int cg_max_iters, double cg_tol);

extern "C" int compute_funm_apply(
    int n, int nnz,
    const int *row_ptr, const int *col_idx, const double *vals,
    const double *v, int m,
    const char *func_name,
    double *out_y);

extern "C" int compute_funm_qform(
    int n, int nnz,
    const int *row_ptr, const int *col_idx, const double *vals,
    const double *v, int m,
    const char *func_name,
    double *out_value);

extern "C" int compute_funm_debug(
    int n, int nnz,
    const int *row_ptr, const int *col_idx, const double *vals,
    const double *v, int m,
    double *out_V, double *out_alpha, double *out_beta);

// ============================================================
// Standard eigensolver binding
// ============================================================
static std::tuple<py::array_t<double>, py::array_t<double>>
gpu_eigsh_impl(
    py::array_t<int, py::array::c_style> indptr,
    py::array_t<int, py::array::c_style> indices,
    py::array_t<double, py::array::c_style> data,
    int n, int k, int ncv, int max_iters, double tol)
{
    if (k < 1 || k > n)
        throw std::invalid_argument("k must be in [1, n]");

    auto indptr_buf  = indptr.request();
    auto indices_buf = indices.request();
    auto data_buf    = data.request();
    int nnz = (int)data_buf.size;

    // Temporary buffers for C function (column-major eigenvectors)
    std::vector<double> eig_buf(k);
    std::vector<double> vec_buf((size_t)n * k);

    int nconv = compute_eigsh(
        n, nnz,
        (const int *)indptr_buf.ptr,
        (const int *)indices_buf.ptr,
        (const double *)data_buf.ptr,
        k, ncv, max_iters, tol,
        eig_buf.data(), vec_buf.data());

    // Copy eigenvalues
    py::array_t<double> eigenvalues({nconv});
    memcpy(eigenvalues.mutable_data(), eig_buf.data(), nconv * sizeof(double));

    // Convert eigenvectors: column-major (C) -> row-major (numpy)
    py::array_t<double> eigenvectors({n, nconv});
    auto *out = eigenvectors.mutable_data();
    for (int j = 0; j < nconv; j++)
        for (int i = 0; i < n; i++)
            out[i * nconv + j] = vec_buf[j * n + i];

    return {eigenvalues, eigenvectors};
}

// ============================================================
// Shift-invert eigensolver binding
// ============================================================
static std::tuple<py::array_t<double>, py::array_t<double>>
gpu_eigsh_sigma_impl(
    py::array_t<int, py::array::c_style> indptr,
    py::array_t<int, py::array::c_style> indices,
    py::array_t<double, py::array::c_style> data,
    int n, int k, int ncv, int max_iters, double tol,
    double sigma, int cg_max_iters, double cg_tol)
{
    if (k < 1 || k > n)
        throw std::invalid_argument("k must be in [1, n]");

    auto indptr_buf  = indptr.request();
    auto indices_buf = indices.request();
    auto data_buf    = data.request();
    int nnz = (int)data_buf.size;

    std::vector<double> eig_buf(k);
    std::vector<double> vec_buf((size_t)n * k);

    int nconv = compute_eigsh_sigma(
        n, nnz,
        (const int *)indptr_buf.ptr,
        (const int *)indices_buf.ptr,
        (const double *)data_buf.ptr,
        k, ncv, max_iters, tol, sigma,
        cg_max_iters, cg_tol,
        eig_buf.data(), vec_buf.data());

    py::array_t<double> eigenvalues({nconv});
    memcpy(eigenvalues.mutable_data(), eig_buf.data(), nconv * sizeof(double));

    py::array_t<double> eigenvectors({n, nconv});
    auto *out = eigenvectors.mutable_data();
    for (int j = 0; j < nconv; j++)
        for (int i = 0; i < n; i++)
            out[i * nconv + j] = vec_buf[j * n + i];

    return {eigenvalues, eigenvectors};
}

// ============================================================
// Adjoint backward pass binding
// ============================================================
static py::array_t<double> gpu_adjoint_eigsh_impl(
    py::array_t<int, py::array::c_style> indptr,
    py::array_t<int, py::array::c_style> indices,
    py::array_t<double, py::array::c_style> data,
    int n, int k,
    py::array_t<double, py::array::c_style> eigenvalues,
    py::array_t<double, py::array::c_style> eigenvectors,
    py::array_t<double, py::array::c_style> grad_evals,
    py::array_t<double, py::array::c_style> grad_evecs,
    int cg_max_iters, double cg_tol)
{
    auto indptr_buf  = indptr.request();
    auto indices_buf = indices.request();
    auto data_buf    = data.request();
    int nnz = (int)data_buf.size;

    auto evals_buf = eigenvalues.request();
    auto evecs_buf = eigenvectors.request();
    auto ge_buf    = grad_evals.request();
    auto gv_buf    = grad_evecs.request();

    py::array_t<double> grad_vals({nnz});
    auto gvals_buf = grad_vals.mutable_data();

    compute_adjoint_eigsh(
        n, nnz,
        (const int *)indptr_buf.ptr,
        (const int *)indices_buf.ptr,
        (const double *)data_buf.ptr,
        k,
        (const double *)evals_buf.ptr,
        (const double *)evecs_buf.ptr,
        (const double *)ge_buf.ptr,
        (const double *)gv_buf.ptr,
        gvals_buf,
        cg_max_iters, cg_tol);

    return grad_vals;
}

PYBIND11_MODULE(_core, m) {
    m.doc() = "GPU-accelerated sparse eigenvalue solver with differentiable backward pass";

    m.def("_gpu_eigsh_raw", &gpu_eigsh_impl,
          py::arg("indptr"), py::arg("indices"), py::arg("data"),
          py::arg("n"), py::arg("k") = 20,
          py::arg("ncv") = 0, py::arg("max_iters") = 2000,
          py::arg("tol") = 1e-12,
          "Standard GPU eigsh: k smallest eigenvalues");

    m.def("_gpu_eigsh_sigma_raw", &gpu_eigsh_sigma_impl,
          py::arg("indptr"), py::arg("indices"), py::arg("data"),
          py::arg("n"), py::arg("k") = 20,
          py::arg("ncv") = 0, py::arg("max_iters") = 2000,
          py::arg("tol") = 1e-12,
          py::arg("sigma") = 0.0,
          py::arg("cg_max_iters") = 500,
          py::arg("cg_tol") = 1e-12,
          "Shift-invert GPU eigsh: k eigenvalues nearest to sigma");

    m.def("_gpu_adjoint_eigsh_raw", &gpu_adjoint_eigsh_impl,
          py::arg("indptr"), py::arg("indices"), py::arg("data"),
          py::arg("n"), py::arg("k"),
          py::arg("eigenvalues"), py::arg("eigenvectors"),
          py::arg("grad_evals"), py::arg("grad_evecs"),
          py::arg("cg_max_iters") = 500,
          py::arg("cg_tol") = 1e-10,
          "Adjoint backward pass for differentiable eigsh");

    m.def("_gpu_funm_apply_raw",
          [](py::array_t<int, py::array::c_style> indptr,
             py::array_t<int, py::array::c_style> indices,
             py::array_t<double, py::array::c_style> data,
             int n,
             py::array_t<double, py::array::c_style> v,
             int m,
             const std::string &func) -> std::tuple<py::array_t<double>, int> {
              auto ip = indptr.request();
              auto ix = indices.request();
              auto dt = data.request();
              auto vb = v.request();
              int nnz = (int)dt.size;

              py::array_t<double> out({n});
              int rc = compute_funm_apply(
                  n, nnz,
                  (const int *)ip.ptr,
                  (const int *)ix.ptr,
                  (const double *)dt.ptr,
                  (const double *)vb.ptr,
                  m, func.c_str(),
                  out.mutable_data());
              return {out, rc};
          },
          py::arg("indptr"), py::arg("indices"), py::arg("data"),
          py::arg("n"), py::arg("v"), py::arg("m"),
          py::arg("func"),
          "Apply f(A) v via m-step Lanczos quadrature. "
          "func in {'log','exp','sqrt','inv'}.");

    m.def("_gpu_funm_debug_raw",
          [](py::array_t<int, py::array::c_style> indptr,
             py::array_t<int, py::array::c_style> indices,
             py::array_t<double, py::array::c_style> data,
             int n,
             py::array_t<double, py::array::c_style> v,
             int m) -> std::tuple<py::array_t<double>, py::array_t<double>,
                                  py::array_t<double>, int> {
              auto ip = indptr.request();
              auto ix = indices.request();
              auto dt = data.request();
              auto vb = v.request();
              int nnz = (int)dt.size;

              py::array_t<double> V_out({n, m});
              py::array_t<double> alpha_out({m});
              py::array_t<double> beta_out({m + 1});
              int ma = compute_funm_debug(
                  n, nnz,
                  (const int *)ip.ptr,
                  (const int *)ix.ptr,
                  (const double *)dt.ptr,
                  (const double *)vb.ptr, m,
                  V_out.mutable_data(),
                  alpha_out.mutable_data(),
                  beta_out.mutable_data());
              return {V_out, alpha_out, beta_out, ma};
          },
          py::arg("indptr"), py::arg("indices"), py::arg("data"),
          py::arg("n"), py::arg("v"), py::arg("m"),
          "Debug: return V (n×m col-major), alpha (m), beta (m+1), m_actual.");

    m.def("_gpu_funm_qform_raw",
          [](py::array_t<int, py::array::c_style> indptr,
             py::array_t<int, py::array::c_style> indices,
             py::array_t<double, py::array::c_style> data,
             int n,
             py::array_t<double, py::array::c_style> v,
             int m,
             const std::string &func) -> std::tuple<double, int> {
              auto ip = indptr.request();
              auto ix = indices.request();
              auto dt = data.request();
              auto vb = v.request();
              int nnz = (int)dt.size;

              double val = 0.0;
              int rc = compute_funm_qform(
                  n, nnz,
                  (const int *)ip.ptr,
                  (const int *)ix.ptr,
                  (const double *)dt.ptr,
                  (const double *)vb.ptr,
                  m, func.c_str(),
                  &val);
              return {val, rc};
          },
          py::arg("indptr"), py::arg("indices"), py::arg("data"),
          py::arg("n"), py::arg("v"), py::arg("m"),
          py::arg("func"),
          "Quadratic form v^T f(A) v via m-step Lanczos quadrature.");
}
