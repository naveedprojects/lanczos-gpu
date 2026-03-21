/**
 * _bindings.cpp — pybind11 bindings for gpu_eigsh.
 *
 * Wraps the CUDA compute_eigsh() function to accept/return numpy arrays.
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

static std::tuple<py::array_t<double>, py::array_t<double>>
gpu_eigsh_impl(
    py::array_t<int, py::array::c_style> indptr,
    py::array_t<int, py::array::c_style> indices,
    py::array_t<double, py::array::c_style> data,
    int n, int k, int ncv, int max_iters, double tol)
{
    // Validate inputs
    if (k < 1 || k > n)
        throw std::invalid_argument("k must be in [1, n]");

    auto indptr_buf  = indptr.request();
    auto indices_buf = indices.request();
    auto data_buf    = data.request();

    int nnz = (int)data_buf.size;

    // Allocate output arrays
    py::array_t<double> eigenvalues({k});
    py::array_t<double> eigenvectors({n, k});  // column-major in C++, but numpy is row-major

    auto eig_buf = eigenvalues.mutable_data();
    auto vec_buf = eigenvectors.mutable_data();

    int nconv = compute_eigsh(
        n, nnz,
        (const int *)indptr_buf.ptr,
        (const int *)indices_buf.ptr,
        (const double *)data_buf.ptr,
        k, ncv, max_iters, tol,
        eig_buf, vec_buf);

    if (nconv < k) {
        // Resize eigenvalues to nconv
        eigenvalues = py::array_t<double>({nconv});
        memcpy(eigenvalues.mutable_data(), eig_buf, nconv * sizeof(double));
    }

    // Reshape eigenvectors: our C++ code stores column-major (n x k),
    // numpy expects row-major. We need to transpose.
    // Actually, let numpy handle it via Fortran order.
    eigenvectors = py::array_t<double>(
        {n, nconv},
        {(long int)sizeof(double), (long int)(n * sizeof(double))},  // Fortran strides
        vec_buf);

    return {eigenvalues, eigenvectors};
}

PYBIND11_MODULE(_core, m) {
    m.doc() = "GPU-accelerated sparse eigenvalue solver (ARPACK-quality IRLM on CUDA)";

    m.def("_gpu_eigsh_raw", &gpu_eigsh_impl,
          py::arg("indptr"), py::arg("indices"), py::arg("data"),
          py::arg("n"), py::arg("k") = 20,
          py::arg("ncv") = 0, py::arg("max_iters") = 2000,
          py::arg("tol") = 1e-12,
          "Low-level GPU eigsh: accepts raw CSR arrays, returns (eigenvalues, eigenvectors)");
}
