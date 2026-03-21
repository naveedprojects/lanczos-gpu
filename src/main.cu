/**
 * main.cu — Benchmark driver for GPU Lanczos eigensolver variants.
 *
 * Builds a graph Laplacian from a random KNN point cloud, runs three
 * Lanczos variants (Naive, DGKS, IRLM), exports results for comparison
 * against SciPy ARPACK, and outputs CSV data for plotting.
 *
 * Usage:
 *   ./lanczos_bench [--n <size>] [--k <neighbors>] [--eigs <num>]
 *                   [--iters <max>] [--tol <convergence>] [--outdir <dir>]
 */

#include "lanczos_ops.cuh"
#include "tridiag.cuh"
#include <vector>
#include <algorithm>
#include <cstring>

// Forward declarations
LanczosResult naive_lanczos(LanczosContext &ctx, SparseMatrixCSR &A,
                            const LanczosParams &params);
LanczosResult dgks_lanczos(LanczosContext &ctx, SparseMatrixCSR &A,
                           const LanczosParams &params);
LanczosResult irlm_lanczos(LanczosContext &ctx, SparseMatrixCSR &A,
                           const LanczosParams &params);

// ============================================================
// Build a graph Laplacian L = D - A from a random KNN point cloud.
// ============================================================
static SparseMatrixCSR build_knn_graph_laplacian(int n, int dim,
                                                  int k_neighbors,
                                                  double bandwidth) {
    printf("Building KNN graph Laplacian: n=%d, dim=%d, k=%d, bw=%.2f\n",
           n, dim, k_neighbors, bandwidth);

    srand(123);
    std::vector<double> points(n * dim);
    for (int i = 0; i < n * dim; i++)
        points[i] = (double)rand() / RAND_MAX;

    // Brute-force KNN
    std::vector<std::vector<std::pair<int, double>>> neighbors(n);

    for (int i = 0; i < n; i++) {
        std::vector<std::pair<double, int>> dists;
        dists.reserve(n);
        for (int j = 0; j < n; j++) {
            if (i == j) continue;
            double dist2 = 0.0;
            for (int d = 0; d < dim; d++) {
                double diff = points[i * dim + d] - points[j * dim + d];
                dist2 += diff * diff;
            }
            dists.push_back({dist2, j});
        }
        int kk = std::min(k_neighbors, (int)dists.size());
        std::partial_sort(dists.begin(), dists.begin() + kk, dists.end());
        for (int ki = 0; ki < kk; ki++) {
            double w = exp(-dists[ki].first / (4.0 * bandwidth * bandwidth));
            neighbors[i].push_back({dists[ki].second, w});
        }
    }

    // Symmetrize
    for (int i = 0; i < n; i++) {
        for (auto &[j, w] : neighbors[i]) {
            bool found = false;
            for (auto &[jj, ww] : neighbors[j]) {
                if (jj == i) { found = true; break; }
            }
            if (!found) neighbors[j].push_back({i, w});
        }
    }

    // Build CSR for L = D - A
    std::vector<int> row_ptr(n + 1, 0);
    for (int i = 0; i < n; i++)
        row_ptr[i + 1] = row_ptr[i] + (int)neighbors[i].size() + 1;
    int nnz = row_ptr[n];

    std::vector<int> col_idx(nnz);
    std::vector<double> vals(nnz);

    for (int i = 0; i < n; i++) {
        std::sort(neighbors[i].begin(), neighbors[i].end());
        double degree = 0.0;
        for (auto &[j, w] : neighbors[i]) degree += w;

        int offset = row_ptr[i];
        int pos = 0;
        bool diag_inserted = false;
        for (auto &[j, w] : neighbors[i]) {
            if (!diag_inserted && i < j) {
                col_idx[offset + pos] = i;
                vals[offset + pos] = degree;
                pos++;
                diag_inserted = true;
            }
            col_idx[offset + pos] = j;
            vals[offset + pos] = -w;
            pos++;
        }
        if (!diag_inserted) {
            col_idx[offset + pos] = i;
            vals[offset + pos] = degree;
            pos++;
        }
    }

    printf("  Laplacian: n=%d, nnz=%d, avg_nnz/row=%.1f\n",
           n, nnz, (double)nnz / n);

    SparseMatrixCSR A;
    A.n = n;
    A.nnz = nnz;
    CUDA_CHECK(cudaMalloc(&A.d_row_ptr, (n + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&A.d_col_idx, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&A.d_vals,    nnz * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(A.d_row_ptr, row_ptr.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(A.d_col_idx, col_idx.data(), nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(A.d_vals,    vals.data(),    nnz * sizeof(double), cudaMemcpyHostToDevice));
    A.create_descriptor();
    return A;
}

// ============================================================
// Export sparse matrix in Matrix Market format for scipy comparison.
// ============================================================
static void export_matrix_market(SparseMatrixCSR &A, const char *filename) {
    std::vector<int> row_ptr(A.n + 1), col_idx(A.nnz);
    std::vector<double> vals(A.nnz);

    CUDA_CHECK(cudaMemcpy(row_ptr.data(), A.d_row_ptr, (A.n + 1) * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(col_idx.data(), A.d_col_idx, A.nnz * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(vals.data(),    A.d_vals,    A.nnz * sizeof(double), cudaMemcpyDeviceToHost));

    FILE *f = fopen(filename, "w");
    if (!f) { fprintf(stderr, "Cannot open %s\n", filename); return; }
    fprintf(f, "%%%%MatrixMarket matrix coordinate real general\n");
    fprintf(f, "%d %d %d\n", A.n, A.n, A.nnz);
    for (int i = 0; i < A.n; i++)
        for (int p = row_ptr[i]; p < row_ptr[i + 1]; p++)
            fprintf(f, "%d %d %.15e\n", i + 1, col_idx[p] + 1, vals[p]);
    fclose(f);
    printf("Exported matrix to %s (%d x %d, %d nnz)\n",
           filename, A.n, A.n, A.nnz);
}

// ============================================================
// Load a sparse matrix from Matrix Market coordinate format.
// ============================================================
static SparseMatrixCSR load_matrix_market(const char *filename) {
    FILE *f = fopen(filename, "r");
    if (!f) {
        fprintf(stderr, "Cannot open %s\n", filename);
        exit(EXIT_FAILURE);
    }

    // Skip comments
    char line[1024];
    do {
        if (!fgets(line, sizeof(line), f)) {
            fprintf(stderr, "Unexpected EOF in %s\n", filename);
            exit(EXIT_FAILURE);
        }
    } while (line[0] == '%');

    // Read dimensions
    int nrows, ncols, nnz_file;
    sscanf(line, "%d %d %d", &nrows, &ncols, &nnz_file);
    if (nrows != ncols) {
        fprintf(stderr, "Matrix must be square: %d x %d\n", nrows, ncols);
        exit(EXIT_FAILURE);
    }

    printf("Loading %s: %d x %d, %d entries\n", filename, nrows, ncols, nnz_file);

    // Read COO entries (1-indexed in MM format)
    std::vector<int> coo_row(nnz_file), coo_col(nnz_file);
    std::vector<double> coo_val(nnz_file);
    for (int i = 0; i < nnz_file; i++) {
        int r, c;
        double v;
        if (fscanf(f, "%d %d %lf", &r, &c, &v) != 3) {
            fprintf(stderr, "Parse error at entry %d\n", i);
            exit(EXIT_FAILURE);
        }
        coo_row[i] = r - 1;  // Convert to 0-indexed
        coo_col[i] = c - 1;
        coo_val[i] = v;
    }
    fclose(f);

    // Convert COO to CSR
    int n = nrows;
    std::vector<int> row_ptr(n + 1, 0);
    for (int i = 0; i < nnz_file; i++)
        row_ptr[coo_row[i] + 1]++;
    for (int i = 0; i < n; i++)
        row_ptr[i + 1] += row_ptr[i];

    int nnz = nnz_file;
    std::vector<int> col_idx(nnz);
    std::vector<double> vals(nnz);
    std::vector<int> offset(n + 1);
    memcpy(offset.data(), row_ptr.data(), (n + 1) * sizeof(int));

    for (int i = 0; i < nnz_file; i++) {
        int r = coo_row[i];
        int pos = offset[r]++;
        col_idx[pos] = coo_col[i];
        vals[pos] = coo_val[i];
    }

    // Sort columns within each row
    for (int i = 0; i < n; i++) {
        int start = row_ptr[i], end = row_ptr[i + 1];
        // Simple insertion sort (rows are small)
        for (int j = start + 1; j < end; j++) {
            int key_c = col_idx[j];
            double key_v = vals[j];
            int k = j - 1;
            while (k >= start && col_idx[k] > key_c) {
                col_idx[k + 1] = col_idx[k];
                vals[k + 1] = vals[k];
                k--;
            }
            col_idx[k + 1] = key_c;
            vals[k + 1] = key_v;
        }
    }

    SparseMatrixCSR A;
    A.n = n;
    A.nnz = nnz;
    CUDA_CHECK(cudaMalloc(&A.d_row_ptr, (n + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&A.d_col_idx, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&A.d_vals,    nnz * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(A.d_row_ptr, row_ptr.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(A.d_col_idx, col_idx.data(), nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(A.d_vals,    vals.data(),    nnz * sizeof(double), cudaMemcpyHostToDevice));
    A.create_descriptor();
    return A;
}

// ============================================================
// Print eigenvalue summary for one algorithm result.
// ============================================================
static void print_result(const char *name, const LanczosResult &res,
                         float ms) {
    printf("  Iterations: %d\n", res.num_iters);
    printf("  Reorthogonalizations: %d\n", res.num_reorths);
    printf("  Time: %.1f ms\n", ms);
    if (res.ortho_loss_len > 0)
        printf("  Final orthogonality loss: %.6e\n",
               res.ortho_loss[res.ortho_loss_len - 1]);
    printf("  Smallest %d eigenvalues:\n    ", res.k);
    for (int i = 0; i < res.k && i < 10; i++)
        printf("%.6f ", res.eigenvalues[i]);
    if (res.k > 10) printf("...");
    printf("\n");
}

static void print_separator() {
    printf("================================================================\n");
}

// ============================================================
// Main
// ============================================================
int main(int argc, char **argv) {
    // Default parameters
    int n = 5000, dim = 10, k_neighbors = 15;
    int num_eigs = 20, max_iters = 1000, measure_freq = 5, ncv = 0;
    double bandwidth = 0.3, tol = 1e-12;
    const char *output_dir = ".";
    const char *mtx_file = nullptr;
    bool irlm_only = false;
    bool mixed_precision = false;

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--n")      && i+1 < argc) n = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--dim")    && i+1 < argc) dim = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--k")      && i+1 < argc) k_neighbors = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--eigs")   && i+1 < argc) num_eigs = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--iters")  && i+1 < argc) max_iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--freq")   && i+1 < argc) measure_freq = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--bw")     && i+1 < argc) bandwidth = atof(argv[++i]);
        else if (!strcmp(argv[i], "--tol")    && i+1 < argc) tol = atof(argv[++i]);
        else if (!strcmp(argv[i], "--outdir") && i+1 < argc) output_dir = argv[++i];
        else if (!strcmp(argv[i], "--mtx")    && i+1 < argc) mtx_file = argv[++i];
        else if (!strcmp(argv[i], "--ncv")    && i+1 < argc) ncv = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--irlm-only")) irlm_only = true;
        else if (!strcmp(argv[i], "--mixed")) mixed_precision = true;
    }

    // Input validation
    if (n < 1 || num_eigs < 1 || num_eigs > n || max_iters < 1) {
        fprintf(stderr, "Invalid parameters: n=%d, eigs=%d, iters=%d\n",
                n, num_eigs, max_iters);
        return 1;
    }

    print_separator();
    printf("GPU Lanczos Benchmark: Naive vs DGKS vs IRLM\n");
    print_separator();

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s (SM %d.%d, %d MB)\n", prop.name, prop.major, prop.minor,
           (int)(prop.totalGlobalMem / (1024 * 1024)));

    // Memory budget check
    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    size_t basis_bytes = (size_t)n * max_iters * sizeof(double);
    printf("GPU memory: %zu MB free / %zu MB total\n",
           free_mem / (1024 * 1024), total_mem / (1024 * 1024));
    printf("Basis memory (n=%d, iters=%d): %zu MB\n",
           n, max_iters, basis_bytes / (1024 * 1024));
    if (basis_bytes > free_mem * 8 / 10) {
        printf("WARNING: Basis exceeds 80%% of free GPU memory.\n");
        printf("  Consider reducing --iters or --n.\n");
    }

    printf("Parameters: eigs=%d, iters=%d, tol=%.1e\n\n", num_eigs, max_iters, tol);

    // Build or load matrix
    SparseMatrixCSR L;
    char fname[512];

    if (mtx_file) {
        L = load_matrix_market(mtx_file);
        n = L.n;
    } else {
        printf("Building KNN graph (n=%d, dim=%d, k=%d, bw=%.2f)...\n",
               n, dim, k_neighbors, bandwidth);
        L = build_knn_graph_laplacian(n, dim, k_neighbors, bandwidth);

        // Export for scipy comparison
        snprintf(fname, sizeof(fname), "%s/laplacian.mtx", output_dir);
        export_matrix_market(L, fname);
    }

    // Create shared GPU context
    int max_basis = std::max(max_iters, 3 * num_eigs + 1);
    LanczosContext ctx;
    ctx.init(n, max_basis);

    LanczosParams params;
    params.num_eigs = num_eigs;
    params.max_iters = max_iters;
    params.measure_freq = measure_freq;
    params.tol = tol;
    params.ncv = ncv;
    params.mixed_precision = mixed_precision;

    LanczosResult naive_result, dgks_result;
    float naive_ms = 0, dgks_ms = 0;

    if (!irlm_only) {
        // ---- Naive Lanczos ----
        print_separator();
        printf("Running NAIVE Lanczos (no reorthogonalization)...\n");
        print_separator();
        ctx.timer_start();
        naive_result = naive_lanczos(ctx, L, params);
        naive_ms = ctx.timer_stop();
        print_result("Naive", naive_result, naive_ms);

        // ---- DGKS Lanczos ----
        print_separator();
        printf("Running DGKS Lanczos (ARPACK-style reorthogonalization)...\n");
        print_separator();
        ctx.timer_start();
        dgks_result = dgks_lanczos(ctx, L, params);
        dgks_ms = ctx.timer_stop();
        print_result("DGKS", dgks_result, dgks_ms);
    }

    // ---- IRLM Lanczos ----
    print_separator();
    printf("Running IRLM Lanczos (implicit restarts + DGKS)...\n");
    print_separator();
    ctx.timer_start();
    LanczosResult irlm_result = irlm_lanczos(ctx, L, params);
    float irlm_ms = ctx.timer_stop();
    print_result("IRLM", irlm_result, irlm_ms);

    // ---- Output CSV ----
    if (!irlm_only) {
        // Orthogonality loss
        snprintf(fname, sizeof(fname), "%s/ortho_loss.csv", output_dir);
        FILE *f = fopen(fname, "w");
        fprintf(f, "iteration,naive,dgks\n");
        int max_len = std::max(naive_result.ortho_loss_len,
                               dgks_result.ortho_loss_len);
        for (int i = 0; i < max_len; i++) {
            int iter = (i + 1) * measure_freq;
            double nv = (i < naive_result.ortho_loss_len)
                        ? naive_result.ortho_loss[i] : -1.0;
            double dv = (i < dgks_result.ortho_loss_len)
                        ? dgks_result.ortho_loss[i] : -1.0;
            fprintf(f, "%d,%.15e,%.15e\n", iter, nv, dv);
        }
        fclose(f);
        printf("\nOrthogonality loss saved to %s\n", fname);

        // Eigenvalue comparison
        snprintf(fname, sizeof(fname), "%s/eigenvalues.csv", output_dir);
        f = fopen(fname, "w");
        fprintf(f, "index,naive,dgks,irlm\n");
        int k = std::min({naive_result.k, dgks_result.k, irlm_result.k});
        for (int i = 0; i < k; i++) {
            fprintf(f, "%d,%.15e,%.15e,%.15e\n", i,
                    naive_result.eigenvalues[i],
                    dgks_result.eigenvalues[i],
                    irlm_result.eigenvalues[i]);
        }
        fclose(f);
        printf("Eigenvalues saved to %s\n", fname);
    } else {
        // IRLM-only eigenvalue output
        snprintf(fname, sizeof(fname), "%s/eigenvalues.csv", output_dir);
        FILE *f = fopen(fname, "w");
        fprintf(f, "index,irlm\n");
        for (int i = 0; i < irlm_result.k; i++)
            fprintf(f, "%d,%.15e\n", i, irlm_result.eigenvalues[i]);
        fclose(f);
        printf("\nEigenvalues saved to %s\n", fname);
    }

    // Summary
    print_separator();
    printf("SUMMARY\n");
    print_separator();
    if (!irlm_only) {
        printf("  %-30s %15s %15s %15s\n", "", "NAIVE", "DGKS", "IRLM");
        printf("  %-30s %15d %15d %15d\n", "Iterations:",
               naive_result.num_iters, dgks_result.num_iters, irlm_result.num_iters);
        printf("  %-30s %15d %15d %15d\n", "Reorthogonalizations:",
               naive_result.num_reorths, dgks_result.num_reorths, irlm_result.num_reorths);
        printf("  %-30s %12.1f ms %12.1f ms %12.1f ms\n", "Time:",
               naive_ms, dgks_ms, irlm_ms);
    } else {
        printf("  %-30s %15s\n", "", "IRLM");
        printf("  %-30s %15d\n", "Iterations:", irlm_result.num_iters);
        printf("  %-30s %15d\n", "Reorthogonalizations:", irlm_result.num_reorths);
        printf("  %-30s %12.1f ms\n", "Time:", irlm_ms);
        if (irlm_result.ortho_loss_len > 0)
            printf("  %-30s %15.2e\n", "Final ortho loss:",
                   irlm_result.ortho_loss[irlm_result.ortho_loss_len - 1]);
    }
    print_separator();

    // Cleanup
    if (!irlm_only) {
        naive_result.free();
        dgks_result.free();
    }
    irlm_result.free();
    ctx.destroy();
    L.free();

    return 0;
}
