# gpu-lanczos: ARPACK-Quality Eigenvalue Solver on GPU

A numerically robust GPU implementation of the **Implicitly Restarted Lanczos Method (IRLM)**, faithfully porting the numerical safeguards from ARPACK to CUDA.

Existing GPU Lanczos implementations are "straightforward rewrites of the most basic algorithm from Wikipedia" — they lose orthogonality after ~50 iterations and produce spurious eigenvalues. This implementation ports the safeguards that make ARPACK reliable:

- **DGKS conditional reorthogonalization** (Daniel-Gragg-Kaufman-Stewart criterion from `dsaitr.f`)
- **Implicit QR restarts** via Givens bulge chase (`dsapps.f`)
- **Exact shift selection** with unwanted Ritz values (`dsgets.f`)
- **Dynamic NEV adjustment** for anti-stagnation (`dsaup2.f`)
- **Convergence checking** via Ritz estimates (`dsconv.f`)
- **Multi-step safe scaling** near underflow (LAPACK `dlascl`)

All heavy linear algebra runs on GPU (cuSPARSE, cuBLAS). Control flow stays on CPU. Zero GPU memory allocations inside any inner loop.

## Results

Verified against SciPy's `eigsh` (which calls ARPACK) on the **exact same matrix** at every scale:

| n | GPU IRLM | SciPy ARPACK | Speedup | Max Rel Error |
|---|---|---|---|---|
| 5,000 | 0.3s | 0.2s | 0.6x | 7.9×10⁻¹⁵ |
| 10,000 | 0.3s | 0.3s | 1.0x | 2.1×10⁻¹⁴ |
| 50,000 | 0.7s | 9.7s | **13x** | 2.8×10⁻¹⁴ |
| 100,000 | 1.2s | 20.9s | **18x** | 1.2×10⁻¹⁴ |
| 500,000 | 6.9s | 169.9s | **25x** | 9.1×10⁻¹⁴ |
| 1,000,000 | 18.7s | 987.6s | **53x** | 1.1×10⁻¹³ |
| 5,000,000 | 155s | — | est. 100x+ | — |

All 20 eigenvalues converge at every scale. Tested on NVIDIA RTX 3080 Ti Laptop GPU.

### Orthogonality

Without reorthogonalization, the Lanczos basis loses all orthogonality by iteration ~70, producing spurious "ghost" eigenvalues (Paige's theorem). With DGKS, orthogonality stays at machine epsilon (~10⁻¹⁵) throughout:

![Orthogonality](figures/orthogonality_loss.png)

### Eigenvalue Accuracy

IRLM matches ARPACK to machine precision across all eigenvalues. Naive Lanczos produces completely wrong results:

![Eigenvalues](figures/eigenvalue_comparison.png)

## Architecture

```
src/
  lanczos_types.cuh      — Types, error macros, numerical constants
  lanczos_context.cuh     — GPU context: handles, streams, pinned memory, pre-allocated buffers
  lanczos_ops.cuh         — Numerical primitives: SpMV, CGS, DGKS reorth, safe scaling
  tridiag.cuh             — CPU tridiagonal: Givens rotations, QR bulge chase, dstev wrapper
  cast_kernels.cu         — FP64↔FP32 conversion kernels for mixed-precision SpMV
  naive_lanczos.cu        — Naive Lanczos (no reorthogonalization) — baseline
  dgks_lanczos.cu         — DGKS Lanczos (reorthogonalization, no restarts)
  irlm_lanczos.cu         — Full IRLM (implicit restarts + DGKS) — the main solver
  main.cu                 — Benchmark driver
python/
  gpu_eigsh/              — Python wrapper: drop-in replacement for scipy.sparse.linalg.eigsh
benchmark/
  scipy_reference.py      — SciPy ARPACK reference for accuracy comparison
  plot_results.py         — Publication-quality plotting
  generate_large_laplacian.py  — Fast KNN graph generation via KD-tree
  scaling_benchmark.py    — Full scaling benchmark (n=5K to 10M)
```

Key design decisions:
- **Shared `LanczosContext`**: GPU handles, streams, pinned memory, and work buffers created once and reused across all algorithms. No per-algorithm setup/teardown overhead.
- **Zero inner-loop allocations**: All GPU buffers (reorthogonalization coefficients, orthogonality measurement, restart workspace) are pre-allocated. cuSPARSE descriptors are reused via `cusparseDnVecSetValues`.
- **`cudaEvent` timing**: Accurate GPU-only measurement, not `std::chrono`.
- **Adaptive breakdown tolerance**: Scales with estimated `||T||`, not hardcoded.
- **Mixed-precision SpMV**: Optional FP32 matrix values with FP64 Krylov basis. DGKS corrects the FP32 noise.

## Build

Requirements: CUDA toolkit (≥11.0), cuSPARSE, cuBLAS, cuSOLVER, LAPACK, BLAS.

```bash
make              # build
make run          # benchmark at n=5000 (Naive vs DGKS vs IRLM)
make ref          # run SciPy ARPACK reference on the same matrix
make plot         # generate plots
```

For large-scale testing:
```bash
# Generate a large graph Laplacian
python3 benchmark/generate_large_laplacian.py --n 1000000 --outdir data

# Run GPU IRLM on it
./build/lanczos_bench --mtx data/laplacian.mtx --eigs 20 --iters 2000 --ncv 120 --irlm-only --outdir data

# Mixed precision (FP32 SpMV)
./build/lanczos_bench --mtx data/laplacian.mtx --eigs 20 --iters 2000 --ncv 120 --irlm-only --mixed --outdir data
```

## Python API

```python
from gpu_eigsh import gpu_eigsh

# Drop-in replacement for scipy.sparse.linalg.eigsh
eigenvalues, eigenvectors = gpu_eigsh(L, k=20)
```

Build the Python extension:
```bash
cd python && python3 setup.py build_ext --inplace
```

## CLI Options

```
--n <size>        Matrix dimension (default: 5000)
--eigs <k>        Number of eigenvalues (default: 20)
--iters <max>     Maximum Lanczos iterations (default: 1000)
--ncv <size>      Krylov subspace size (default: auto = 3*eigs)
--tol <eps>       Convergence tolerance (default: 1e-12)
--mtx <file>      Load sparse matrix from Matrix Market file
--irlm-only       Skip Naive/DGKS, run IRLM only (for large problems)
--mixed           Enable mixed-precision SpMV (FP32 values + FP64 basis)
--outdir <dir>    Output directory for CSV/plots
```

## Why This Matters

There is no production GPU eigensolver with ARPACK-level numerical safeguards:

| Tool | GPU Lanczos? | ARPACK safeguards? |
|---|---|---|
| cuSOLVER | No Lanczos | N/A |
| Torch-ARPACK | CPU only | Yes |
| HLanc (2015) | Hybrid | Partial |
| Cucheb (2024) | Filtered Lanczos | No DGKS/IRLM |
| **This work** | **Full GPU** | **DGKS + IRLM** |

## References

- Lehoucq, Sorensen, Yang: *ARPACK Users' Guide* (1998)
- Sorensen: *Implicit Application of Polynomial Filters in a k-Step Arnoldi Method* (1992)
- Daniel, Gragg, Kaufman, Stewart: *Reorthogonalization and Stable Algorithms for Updating the Gram-Schmidt QR Factorization* (1976)
- Paige: *The Computation of Eigenvalues and Eigenvectors of Very Large Sparse Matrices* (1971)

## License

MIT
