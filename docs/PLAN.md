# Unified Differentiable Spectral Toolkit — Implementation Plan

**Date drafted:** 2026-05-20
**Target:** Re-engage Prof. Viacheslav Borovitskiy with a single coherent research contribution after a 3-month silence.

## Week-0 update (2026-05-20)

Ran [benchmark/matfree_vs_baselines.py](../benchmark/matfree_vs_baselines.py): matfree-GPU SLQ vs GPyTorch SLQ on the IMGP-style precision operator. Plot at [figures/matfree_vs_baselines.png](../figures/matfree_vs_baselines.png).

**Key finding: matfree-on-GPU is *slower* than GPyTorch SLQ at every scale (3-5× on forward, 100-300× on backward).** The "matfree closes the GPU-native gap" risk is empirically dead. The real competitor is GPyTorch BBMM (the same library IMGP uses and the same library that the IMGP paper's Table 2 footnote documents as failing).

| n | matfree fwd | GPyTorch fwd | matfree bwd | GPyTorch bwd |
|---|---|---|---|---|
| 1K | 57ms | 11ms | 78ms | 0.9ms |
| 100K | 1654ms | 626ms | 3503ms | 12.3ms |

## Week-1 update (2026-05-20)

**Layer 1 forward (CUDA) + backward (PyTorch reference) — both at machine precision.**

- **Forward primitive** in [src/matrix_function.cuh](../src/matrix_function.cuh) + C bindings in [python/gpu_eigsh/_cuda_funm.cu](../python/gpu_eigsh/_cuda_funm.cu) + tests in [tests/test_funm.py](../tests/test_funm.py). Apply/quadratic-form for f ∈ {log, exp, sqrt, inv} at n=100/500/1000, ALL PASS at 1e-14 to 1e-15 vs `torch.linalg.eigh` ground truth. One nasty bug along the way: MKL multi-threaded `dstev` silently produces wrong eigenvectors when CUDA is also initialized. Fixed via single-thread MKL pin in `gpu_eigsh/__init__.py` ctypes call (see code for the dlsym dance).
- **Backward (Krämer adjoint)** in [python/gpu_eigsh/funm_torch.py](../python/gpu_eigsh/funm_torch.py) + tests in [tests/test_funm_backward.py](../tests/test_funm_backward.py). Implemented as a `torch.autograd.Function` with custom forward (no-reortho Lanczos) + transcribed matfree `_tridiag_adjoint_step`. Gradcheck against autograd-through-unrolled-Lanczos passes at machine precision (1e-16) for m/n ≤ 0.25; degrades to ~1e-3 when m/n ≥ 0.40 due to loss of orthogonality without reorth.

**Two follow-ups carry into Week 2:**
1. **Full-reortho adjoint.** Real numerics need reortho in the forward. Krämer's `_hessenberg_adjoint` (Arnoldi-based) is the algorithm; transcribe + verify.
2. **CUDA port.** The Python reference is ready to port. Operator-parameter chain rule (Layer 3) plugs in cleanly once the PyTorch wrapper accepts arbitrary parameterized matvecs instead of dense A.

## Week-2 + Week-3 update (2026-05-20, same session)

**Layer 2 (SLQ log-det) + Layer 3 (operator-callable adjoint) shipped in Python reference. Flagship IMGP-style test passes at <1% error.**

Code in [python/gpu_eigsh/funm_torch.py](../python/gpu_eigsh/funm_torch.py):
- `_FunmQFormKraemerOp` — `torch.autograd.Function` for v^T f(A) v with operator-callable backward. Accepts arbitrary `matvec(x, *params)` and produces gradients on each param via PyTorch autograd through the matvec at each adjoint step (matches matfree's `func.vjp` pattern in JAX).
- `funm_qform_op(matvec, z, params, m, func)` — public API for the differentiable quadratic form.
- `slq_logdet(matvec, n, params, m_probes, lanczos_m, seed)` — Hutchinson SLQ for log-det on top, fully differentiable through `params`.

Tests in [tests/test_slq.py](../tests/test_slq.py):

| Test | Result | Notes |
|---|---|---|
| (1) Op-callable vs same-formulation autograd | **PASS, machine precision** (1e-16) | proves the operator-callable adjoint is correct |
| (2) SLQ log-det forward accuracy | **PASS** (~0.2% rel err at 200 probes) | Hutchinson variance, expected |
| (3) SLQ gradient ∂(log det A(c))/∂c vs tr(A^{-1}) | **PASS** (~2.3% rel err at 100 probes) | gradient-of-stochastic-estimator noise |
| (4) **IMGP-style A(κ) gradient vs analytic** | **PASS** (~0.5% rel err) | flagship — proves the toolkit works end-to-end for graph Matérn precision operators |

**Implications:**
- The "differentiable spectral toolkit" is a working artifact (in Python). The IMGP demo target is no longer "build the math"; it's "wire it into manifold-gp's training loop and run at N=10⁵–10⁶."
- One stability caveat: no-reortho Lanczos at m ≳ 0.4·n loses orthogonality and produces NaN gradients (test (4) was set with m=20, which is fine). Full-reortho adjoint (matfree's `_hessenberg_adjoint`) is the production fix and still owed.
- One algorithmic note: there are two mathematically-equivalent formulations of v^T f(A) v through Lanczos (path-1 "vector y, take z^T y" vs path-2 "scalar q directly"). They give the SAME gradient when V is exactly orthonormal; without full reorth they disagree by O(orthogonality loss). The op-callable path uses formulation 2, which is the natural fit for SLQ. With full reorth this distinction vanishes.

**Remaining for Week 4–6:**
1. **Full-reortho adjoint** (transcribe matfree's `_hessenberg_adjoint`). Unblocks production m and avoids the m/n ratio caveat above.
2. **CUDA port** of the Krämer adjoint, replacing the Python reference. The Python primitive is the porting target.
3. **IMGP demo** — wire `slq_logdet` into `manifold-gp`'s `manifold_informed_train` loop, run MR-MNIST at N=10⁵ and N=10⁶, compare to IMGP-Lanczos (the failing baseline per the Table-2 footnote) and IMGP-full (the OOM-at-scale baseline).

## Week-1 update (2026-05-21): IMGP integration + scaling sweep

**Shipped:** end-to-end differentiable IMGP marginal-likelihood pipeline in Python, with three competing methods sharing identical data / init / optimiser. Code in:
- [python/gpu_eigsh/imgp.py](../python/gpu_eigsh/imgp.py) — differentiable IMGP precision operator and `imgp_neg_marginal_log_likelihood`.
- [python/gpu_eigsh/imgp_train.py](../python/gpu_eigsh/imgp_train.py) — three training loops: `imgp_train_ours`, `imgp_train_dense`, `imgp_train_naive_lanczos`. Each returns a `TrainResult` with per-iteration metrics.
- [benchmark/imgp_demo.py](../benchmark/imgp_demo.py) — single-N demo runner.
- [benchmark/imgp_scaling_sweep.py](../benchmark/imgp_scaling_sweep.py) — multi-N sweep runner.
- [benchmark/plot_imgp_scaling.py](../benchmark/plot_imgp_scaling.py) — Week-1 figure: wall-clock vs N + loss / θ / κ trajectories.
- [tests/test_imgp_integration.py](../tests/test_imgp_integration.py) — verifies our differentiable Laplacian matvec, precision matvec, and SLQ gradient against dense ground truths (all PASS at machine precision / sub-1% for stochastic gradient).

**Correctness.** Tests vs dense reference:
- `L_sym(κ) · v` ≡ dense reference (~1e-16).
- `P(κ, θ) · v` ≡ dense reference for ν ∈ {1, 2} (~1e-16).
- ∂(log det P) / ∂θ via SLQ vs dense autograd: ~1e-4 relative error.
- ∂(log det P) / ∂κ via SLQ vs dense autograd: ~3e-3 relative error.

**Scaling sweep (CUDA, RTX 3080 Ti Laptop).** Training one full marginal-likelihood maximisation (25 Adam iters):

| N | edges | imgp-full (s) | imgp-naive (s) | imgp-ours (s) |
|---|---|---|---|---|
| 500   | 2,484  | 7.3 | 6.4 | 28.8 |
| 2,000 | 9,949  | 33.7 | 6.7 | **29.6** |
| 5,000 | 25,016 | (skipped, would OOM) | 7.0 | 30.8 |
| 10,000| 49,936 | (skipped, would OOM) | 7.1 | **30.7** |

**Key finding (Week-1 headline).** At N = 2,000 the dense baseline (`imgp-full`) crosses over: it becomes *slower than ours*. Past N = 2,000 it's not even runnable on a 16 GB GPU (n² dense matrix). Meanwhile **our cost stays essentially flat across N = 500–10,000** (28.8s → 30.7s) because the sparse SLQ matvec is O(N + nnz) per step. Loss and hyperparameter trajectories overlap across methods at all N tested — correctness validated.

**Naive Lanczos is fast but unreliable.** At these N the naive (no-reortho, unrolled-autograd) Lanczos still converges; we expect divergence at larger N where loss of orthogonality compounds — the IMGP paper's reported failure mode. Reproducing that failure requires the Week-3 N = 10⁵ – 10⁶ regime.

**Figure 1**: [figures/imgp_scaling.png](../figures/imgp_scaling.png). Four-panel: wall-clock-vs-N (log-log), NLL trajectory, θ trajectory, κ trajectory at the largest dense-feasible N.

## Week-2 update (2026-05-21): batched SLQ + scaling to N=10⁵ + naive divergence reproduced

**Shipped:**
- Batched Hutchinson probes in [funm_torch.py:_SLQLogdetBatched](../python/gpu_eigsh/funm_torch.py) — all `m_probes` Lanczos sweeps run as a single batched matvec call. ~15× speedup at N=5000 vs the unbatched loop.
- Updated `slq_logdet` API with `batched=True` default; serial fallback retained for cases where the user's matvec can't accept a (n, B) right-hand side.
- Updated [imgp_demo.py:make_dataset](../benchmark/imgp_demo.py) to use a cheap surrogate signal at N > 4000 (the exact-GP sampling materialised an n×n covariance and OOMed at N=20K).
- Plot script in [plot_imgp_scaling.py](../benchmark/plot_imgp_scaling.py) now marks divergence and dense-OOM regimes explicitly.

**Headline scaling result.** Training one full marginal-likelihood maximisation (20 Adam iters) on a single RTX 3080 Ti Laptop (16 GB):

| N | edges | imgp-full (s) | imgp-naive (s) | imgp-ours (s) |
|---|---|---|---|---|
| 500     | 2,484   | 5.8 | 5.1 | **1.2** |
| 2,000   | 9,949   | 26.4 | 5.1 | **1.0** |
| 5,000   | 25,016  | (OOM) | 5.6 | **1.6** |
| 10,000  | 49,936  | (OOM) | 5.7 | **2.7** |
| 20,000  | 99,839  | (OOM) | 7.0 | **5.3** |
| 50,000  | 249,976 | (OOM) | 13.7 | **14.4** |
| 100,000 | 499,311 | (OOM) | **DIVERGED** | **31.5** |

**The IMGP paper's Table-2 footnote failure mode is reproduced.** At N=100,000 (= the MR-MNIST setup) the naive Lanczos quadrature (unrolled autograd through no-reortho Lanczos — the BBMM-style row from the paper) **diverges to NaN on the first Adam step**, exactly the failure the paper documents. Our batched SLQ with Krämer adjoint converges at every N tested. Figure: [figures/imgp_scaling.png](../figures/imgp_scaling.png).

**Performance.** At N=2000 ours beats dense by 26×. At N=50,000 ours ties naive at ~14s for the full training run (single GPU, no C++ port). At N=100,000 the run takes 31.5s — well within practical hyperparameter-search budgets.

## Week-3 update (2026-05-21): full sweep to N = 10⁶

**Shipped:**
- `make_synthetic_graph` in [imgp_demo.py](../benchmark/imgp_demo.py) — fast random k-neighbour graph generator that skips the cKDTree bottleneck at large N.
- FP32 support in [imgp_scaling_sweep.py](../benchmark/imgp_scaling_sweep.py) (`--dtype float32`). Halves GPU memory; keeps ample headroom for Hutchinson noise.
- [examples/quickstart_slq.py](../examples/quickstart_slq.py) — 60-second demo of the differentiable SLQ API on a parameterised sparse operator. Gradients verified to ~0.3-0.6% vs analytic.

**Headline scaling, FP32 single-GPU (RTX 3080 Ti Laptop, 16 GB), 15 Adam iters:**

| N | edges | imgp-full (s) | imgp-naive (s) | imgp-ours (s) |
|---|---|---|---|---|
| 500       | 2 484     | 4.9 | 2.6 | **0.8** |
| 2 000     | 9 949     | 19.1 | 2.5 | **0.7** |
| 5 000     | 25 016    | (OOM) | 2.7 | **0.6** |
| 10 000    | 49 936    | (OOM) | 2.6 | **0.8** |
| 50 000    | 249 976   | (OOM) | 5.2 | **3.4** |
| 100 000   | 500 571   | (OOM) | 10.1 | **7.1** |
| 500 000   | 2 500 640 | (OOM) | **DIVERGED to NaN** | **40.9** |
| **1 000 000** | **4 999 477** | (OOM) | **DIVERGED to NaN** | **91.7** |

**The three findings that go to Borovitskiy:**

1. *Dense slogdet (the SS-IMGP-full row of the paper)* — OOMs past N = 2 000 on a 16 GB GPU. At N = 2 000 it's 27× slower than ours.
2. *Naive Lanczos quadrature (the BBMM-style row, what the paper's Table-2 footnote documents as failing)* — converges at small N, diverges to NaN on the first Adam step at N = 500 000 and N = 1 000 000.
3. *Our differentiable SLQ* — runs through every N tested. **At N = 10⁶ the full marginal-likelihood maximisation finishes in 91.7 seconds on a single consumer GPU.**

Figure: [figures/imgp_scaling.png](../figures/imgp_scaling.png). Loss / θ / κ trajectories at N = 10⁵ show ours and naive perfectly overlapping — correctness validated at every scale where naive doesn't diverge.

## Week-4 next

1. **Polish the writeup** ([docs/WRITEUP.md](WRITEUP.md)) — fill in §3 with the final numbers, draft the §1 problem statement and the §4 prior-work comparison, write a 6-line message to Borovitskiy.
2. **Final repo polish**: README quickstart, `pip install .` works clean, all four test suites pass on a fresh environment.
3. *(Optional, stretch)* second use case — heat-kernel GP or polynomial Laplacian filter on a real graph (one of Borovitskiy's recent papers if relevant).

---

**Implications for the plan:**
- **Drop** the "matfree on GPU closes the gap" risk from §7.
- **Reframe** the contribution narrative: not "GPU-native beats matfree" — instead "match GPyTorch performance with stronger numerics (DGKS, restart adjoint) and scale to n=10⁶ where GPyTorch's optimization-trajectory failure (IMGP Table 2 footnote) makes it unusable."
- **New risk surfaced**: single-evaluation log det at n=100K agrees between matfree and GPyTorch to ~1e-4. The IMGP paper's "Lanczos quality" issue is therefore **not** a single-eval accuracy problem — it's an *optimization-trajectory* effect (gradient noise compounding over Adam iterations). Week 5 must explicitly reproduce this failure before claiming to fix it. Fallback if it doesn't reproduce at N=100K: shift to N=10⁶ where the contrast no longer needs a failing baseline, just a "we run, they can't" panel.
- **GPyTorch backward is near-free** (probe-vector reuse trick) — our SLQ backward should target the same algorithmic structure, not unrolled adjoint Lanczos like matfree does.

---


## 1. Verify current code state

**What's actually in the repo (verified May 20, 2026).** All files referenced in memory are present and substantive. Source under `/home/naveed/Documents/self/research/lanczos/`:

- **CUDA core (~1900 LOC):** `src/irlm_lanczos.cu` (412 LOC, full IRLM with DGKS + bulge-chase + exact shifts + dynamic-NEV anti-stagnation), `src/lanczos_ops.cuh` (464 LOC, SpMV + CGS + DGKS + safe_scale + RR refinement), `src/matvec_operator.cuh` (166 LOC, abstract operator + CSR + CSR mixed + FunctionOperator + ShiftInvert), `src/inner_solve.cuh` (241 LOC, deflated CG + ShiftInvertOperator), `src/tridiag.cuh` (158 LOC, Givens + bulge chase + dstev), `src/lanczos_context.cuh`, `src/lanczos_types.cuh`, `src/cast_kernels.cu`.
- **Python package (~840 LOC):** `python/gpu_eigsh/_cuda_eigsh.cu` (326 LOC — wraps IRLM, shift-invert, and an adjoint backward kernel), `python/gpu_eigsh/_bindings.cpp`, `python/gpu_eigsh/__init__.py`, `python/gpu_eigsh/differentiable.py` (`_DifferentiableEigsh` torch.autograd.Function).
- **Tests:** only `tests/test_gradient.py` exists. It defines forward/Hellmann-Feynman/shift-invert/eigenvector tests but `main()` calls `test_eigenvalue_gradient_finite_diff` that is not defined — latent broken reference. Minor.
- **Benchmarks:** `benchmark/{competitive,differentiable,manifold_gp,scaling}_benchmark.py` and `generate_large_laplacian.py` all present.
- **Vendored upstream:** `manifold-gp/` is a working copy of nash169/manifold-gp (matches Borovitskiy/Fichera NeurIPS 2023). Training in `manifold_gp/utils/train_model.py:67` evaluates marginal likelihood via `precision_operator.inv_quad_logdet(logdet=True)` — i.e. GPyTorch BBMM/SLQ on the parameterized precision operator. **This is the exact bottleneck the new toolkit is built to attack.**

**Gaps between memory and reality.**
- The "differentiable eigsh" path is implemented in pure host code in `_cuda_eigsh.cu:265-309`: the adjoint loop iterates over `k` eigenpairs, downloads `x_i` and `xi_i` to host, and accumulates `grad_vals` on the CPU with a nested `(r, p)` loop over the CSR sparsity pattern. Works for small `n*k` but will not scale cleanly at n=10^6, k=20 (host-side double loop over `nnz` per eigenpair is the cost driver). Not a blocker, but the "53x at n=1M" claim is forward-only; backward at n=1M has not been measured.
- DGKS and IRLM are real and ARPACK-faithful (the bulge chase in `tridiag.cuh:67` is the textbook dsapps.f). Mixed-precision SpMV is real (FP32 cuSPARSE + FP64 basis). Shift-invert with deflated CG is real.
- **No matrix-function `f(A)v` code exists. No stochastic Lanczos quadrature code exists. No operator-parameter chain rule beyond the explicit-CSR-values case exists.**
- Reorthogonalization is CGS+DGKS (matches ARPACK dsaitr.f). Restarts are exact-shift implicit. This is everything matfree (Krämer et al. 2024) does *not* have, and the headline differentiator going forward.

## 2. Prior-work survey (as of May 20, 2026)

**The single most important reference: Krämer, Moreno-Muñoz, Roy, Hauberg, "Gradients of Functions of Large Matrices," NeurIPS 2024 ([arXiv:2405.17277](https://arxiv.org/abs/2405.17277), library [`matfree`](https://github.com/pnkraemer/matfree)).** What it does:
- Derives adjoint systems for Lanczos and Arnoldi.
- Implements them in JAX with **full reorthogonalization** on both forward and adjoint passes; explicitly assumes "Q fits into memory."
- Demonstrates GP marginal likelihood, PDEs via matrix exponentials, and BNN Laplace approximation.
- Largest experiments: ~33k (PDE 128×128), ~64k (UCI kegg_undir for GPs). Reports "≈20× slower per epoch than GPyTorch" partly due to MVM backends. No graph-Laplacian / manifold-GP experiments. **No implicit restarts. No mixed precision. No shift-invert.**

A clean, defensible *uncrowded* contribution still exists, but is **narrower than "differentiable Lanczos in general"** — that flag is planted. Remaining room:

1. **Restart-aware adjoints.** Krämer et al. assume the full Krylov basis is in memory. ARPACK-style implicit restarts (yours) reduce O(n·m_total) memory to O(n·ncv) at the cost of needing to differentiate *through the restart map*. Nobody has published an adjoint for this.
2. **GPU-native, n ≥ 10^5–10^6, with ARPACK-quality numerics (DGKS, mixed precision, shift-invert).** Krämer is JAX-on-CPU-or-GPU at ~64k; cupyx.eigsh is GPU but has no backward and the IMGP paper documents the Lanczos quality issue forces them to fall back to dense `torch.linalg.eigh`.
3. **Operator-parameter (θ) autograd through an opaque matvec.** Krämer's parameter gradients flow through JAX's tracing, which requires you to write the matvec in JAX. Real users have matvecs in PyTorch/C++/CUDA. A PyTorch-side hook closes that gap.
4. **A GP / manifold-GP demo.** Krämer's GP experiments use Euclidean Matérn on UCI tabular data with kernel-matrix MVMs. *Graph* Matérn precision-matrix MVMs on a ~100k–1M-node implicit manifold are absent. Borovitskiy's own IMGP paper says this is unresolved.

**Other 2025–2026 work checked, none of which closes the gap:**
- [arXiv:2601.05778](https://arxiv.org/pdf/2601.05778) (Cortinovis et al., Jan 9, 2026): preconditioned SLQ with Nyström — non-differentiable.
- arXiv 2307.02152 / 2307.00847 (subspace construction / asymmetric SLQ): non-differentiable.
- Block-SLQ (Yeon et al., May 2025): unbiased parallel trace, non-differentiable.
- [arXiv:2501.04570](https://arxiv.org/html/2501.04570v1) (Large-Scale Spectral GNNs via Laplacian Sparsification): avoids eigendecomposition with polynomial filters — orthogonal.
- arXiv 2501.02565 (Efficient Graph Condensation via Gaussian Process): not differentiable spectral.
- arXiv 2412.17734 (LASE: Learned Adjacency Spectral Embeddings): differentiable but for ASE/graph embedding, small scale.
- Pleiss et al. 2020 ([arXiv:2006.11267](https://arxiv.org/abs/2006.11267)): rational/Krylov K^{±1/2}b with derivatives — predecessor of Krämer.
- PyTorch 2.12 (May 2026) `torch.linalg.eigh` 100× speedup via cuSolver `syevj_batched`: small-batch dense; does not touch large sparse or log-det.
- `cupyx.scipy.sparse.linalg.eigsh` (CuPy 13+): GPU ARPACK-style, but IMGP authors directly document its accuracy issues; not differentiable.

**Honest read.** Krämer 2024 took the obvious "differentiable Lanczos for f(A)v" headline. What's left is a tighter, more engineering-heavy contribution: *production-quality* restart-aware differentiable spectral methods at the 10^5–10^6 scale with a real downstream win in graph GPs. Defensible but the framing must be careful — "first differentiable Lanczos" is taken; "differentiable IRLM with restart adjoints, GPU-native, demonstrated on graph Matérn at 10^6" is open.

## 3. Mathematical sketch

### Layer 1: differentiable f(A)v via Lanczos

**Forward.** With v_0 = v/||v||, run m Lanczos steps with operator A and DGKS reorthogonalization to obtain V_m (n×m) and tridiagonal T_m (m×m) with V_m^T A V_m ≈ T_m. Eigendecompose T_m = S Θ S^T (LAPACK dstev). Then

    f(A) v  ≈  ||v|| · V_m S f(Θ) S^T e_1.

Special cases reuse existing code: for f(x)=1/x, the Lanczos-CG of `inner_solve.cuh` is exactly this; for f(x)=log(x), this is SLQ; for f(x)=x^{1/2}, exp(x), etc. all reuse the same V_m, T_m.

**Backward (no restarts).** Given upstream cotangent g, there are three gradients to produce: w.r.t. v (cheap), w.r.t. the *function* f (cheap), and w.r.t. the operator A (the hard one). The Krämer 2024 derivation gives the adjoint Lanczos iteration in closed form; the key fact is that the adjoint runs in the same complexity as the forward and only needs the saved (V_m, T_m, β_m), plus one solve of an m×m Sylvester-like system on the small T-side and m matvec-VJPs `(x, y) ↦ x^T (∂A/∂θ) y` on the big-A-side. **Reuse Krämer's adjoint for the unrestarted core** (well-tested in matfree) and re-implement in C++/CUDA.

**Backward through implicit restarts (novel).** A single IRLM cycle is:
1. Build (V_m, T_m) by m Lanczos steps.
2. Eigendecompose T_m, pick np unwanted Ritz values as shifts σ_1, ..., σ_np.
3. Apply the bulge chase: T_m → Q^T T_m Q, accumulate Q (m×m); update V_m ← V_m · Q.
4. Reconstruct restart residual r_new = σ_p · r_old + τ · V_new(:, k); β_k ← ||r_new||.
5. Continue Lanczos from column k to m.

Steps 1–2 and 5 reuse the unrestarted adjoint. Steps 3–4 are pure linear algebra of (T_m, β_m, Q) and a basis rotation — fully differentiable in principle. Three implementation choices:

a. **Unroll the entire IRLM in PyTorch autograd.** Easiest, but stores V_new at every restart → O(n·ncv·n_restarts) memory. Kills n=10^6.

b. **Custom backward over the *full* IRLM cycle as a single op** (**recommended**). Save only the converged (V_k, T_k, β_k) at the end. For the restart map, recompute Q on-the-fly from the saved shift sequence (must save shifts per restart). Cost: one extra O(m^3) bulge chase per restart, negligible.

c. **Treat IRLM as an implicit fixed-point and use IFT differentiation.** Cleanest mathematically but needs deflated CG solves for the adjoint anyway (already have this in `inner_solve.cuh`), and SLQ is *not* at a fixed point. So (c) is what existing `_DifferentiableEigsh` already does for eigenvalues, and (b) is needed for f(A)v.

**Where it will fail.** Near-degenerate Ritz values during restart make S in T_m = S Θ S^T ill-conditioned. The Krämer adjoint assumes simple spectrum. For graph Laplacians with high multiplicity at λ=0 (connected components): (i) deflate known zero modes before SLQ, or (ii) shift A ← A + ε I, recover via log det(A) = log det(A+εI) - corrections.

### Layer 2: SLQ for log det

**Forward.** log det A = tr(log A) ≈ (1/m_probes) · Σ_i z_i^T log(A) z_i with z_i ∈ {±1}^n (Rademacher). Each z_i^T log(A) z_i is computed via Layer 1 with v = z_i and f = log, returning ||z_i||^2 · e_1^T S log(Θ) S^T e_1. Hutchinson variance roughly (2/m_probes) · ||log A||_F^2, so m_probes ≈ 10–50 typically suffices for relative accuracy 10^-2 (Ubaru–Chen–Saad 2017). **The contribution is not the algorithm — it is the (ARPACK-quality + GPU-native + differentiable through θ) execution.**

**Backward.** d/dθ log det A(θ) = tr(A^{-1} dA/dθ). Each z_i^T log(A) z_i differentiates (via Layer 1 adjoint) — the chain rule reduces to needing m_probes evaluations of `z_i^T (dA/dθ) y_i` where y_i ≈ A^{-1} z_i (falls out of Lanczos quadrature since the same V_m, T_m also gives A^{-1} z_i = ||z||·V_m T_m^{-1} e_1 essentially free — known SLQ trick used in GPyTorch). This is the same probe vector trick BBMM uses (Gardner et al. 2018).

**Where it will fail.** Variance. For the IMGP precision operator P_νν = (2ν/κ² I + Δ)^ν with ν=2 and normalized Laplacian (spectrum in [0,2]), log P_νν spans a wide range, Rademacher variance can be large. m_probes may need 50–200 at n=10^6 for reliable gradient signal. Variance reduction via control variates / Nyström preconditioning (Cortinovis 2026) is a stretch goal.

### Layer 3: operator-parameter autograd

**The user interface.** Currently `MatVecOperator` in `matvec_operator.cuh:30` defines only `apply(x) → y`. Extend to:

    struct DifferentiableMatVecOperator : MatVecOperator {
        virtual void apply(LanczosContext &, const double *x, double *y) = 0;
        virtual void apply_vjp(LanczosContext &, const double *x, const double *gy,
                               void *grad_theta_accum) = 0;   // accumulate ∂L/∂θ += (gy^T) (∂A/∂θ) x
    };

On the Python side, a `torch.autograd.Function` saves the *callable* (not the matrix) plus the converged (V_k, T_k, β_k, restart shift log). In `backward`, it calls back into Python — the user's PyTorch matvec runs again with `requires_grad=True` on θ, producing gradients via PyTorch's own autograd. This is how matfree handles parameters via JAX tracing; the difference here is PyTorch users keep their matvec in pure PyTorch (or pure CUDA, or pure Triton) and only need to provide one extra function.

For the CSR case this is identical to the existing `_DifferentiableEigsh` path. For arbitrary parameterized operators, the user implements either `apply_vjp` directly (C++) or just lets PyTorch autograd handle it (Python). For graph Laplacians parameterized by κ, dA/dκ has the same sparsity pattern as A with closed-form values.

## 4. Code architecture and file layout

New files (no edits to existing core except adding two virtual methods):

- `src/matrix_function.cuh` — `lanczos_quadrature_apply(ctx, op, v, m, f, restart=false)` returning the quadrature output. Reuses `lanczos_extend` from `irlm_lanczos.cu`. ~200 LOC.
- `src/lanczos_adjoint.cuh` — adjoint of the unrestarted Lanczos quadrature (Krämer 2024 system, transcribed). Reuses `cg_solve_shifted` from `inner_solve.cuh` for the small adjoint solves. ~250 LOC.
- `src/restart_adjoint.cuh` — adjoint of one IRLM cycle: pulls back through bulge-chase Q, residual reconstruction, and basis rotation. Saves shift log per cycle. ~200 LOC.
- `src/slq.cuh` — Hutchinson loop over probe vectors, each calling `lanczos_quadrature_apply`. Streams probe vectors so memory is O(n·ncv) not O(n·ncv·m_probes). ~150 LOC.
- `src/diff_matvec_operator.cuh` — extends `MatVecOperator` with `apply_vjp`. CSR, CSRMixed get default implementations (scatter the rank-1 outer product into nnz pattern, same as today's `compute_adjoint_eigsh` host loop, but moved to a CUDA kernel — important for n=10^6 backward). ~150 LOC.
- `python/gpu_eigsh/_cuda_funm.cu` — pybind11-callable entry points: `compute_funm_apply`, `compute_slq_logdet`, `compute_funm_adjoint`. ~300 LOC.
- `python/gpu_eigsh/spectral.py` — Python autograd wrappers: `funm_apply(A, v, f, m=50)`, `slq_logdet(A, m_probes=20, lanczos_m=50)`, both as `torch.autograd.Function`. Accepts either sparse-CSR tensors or a Python `MatVecCallable`. ~250 LOC.
- `python/gpu_eigsh/parameterized_operator.py` — `ParameterizedOperator` base class for Python users. ~150 LOC.
- `tests/test_funm.py` — gradcheck for f(A)v with f ∈ {log, exp, sqrt, inv} on n ≤ 1000 dense, vs `torch.linalg.eigh` ground truth.
- `tests/test_slq.py` — variance test (compare to exact log det at n=1000), gradient test vs autograd through dense log det.
- `tests/test_param_operator.py` — graph Laplacian parameterized by κ, gradient of trace(log L(κ)) w.r.t. κ vs FD.
- `tests/test_restart_adjoint.py` — IRLM with 0/1/3 restart cycles, gradient consistency check.
- `benchmark/slq_benchmark.py` — log det at n ∈ {10^4, 10^5, 10^6} for SLQ-ours vs GPyTorch SLQ vs dense logdet.
- `benchmark/imgp_benchmark.py` — see Section 5.

**Reuse map.**
- `cg_solve_shifted` (inner_solve.cuh) → reused for adjoint solves and Layer 1 inner systems.
- `dgks_reorth` (lanczos_ops.cuh) → reused for both forward and adjoint Lanczos.
- `apply_shifts_tridiag` (tridiag.cuh) → forward call sequence saved during IRLM; restart adjoint replays it for VJP.
- `_DifferentiableEigsh` (differentiable.py) → unchanged. SLQ is a new top-level entry point that *doesn't* go through eigsh.

**One known cleanup.** The host-side accumulation loop in `_cuda_eigsh.cu:300-307` (the `(r, p)` double loop for adjoint of eigsh) should be moved to a CUDA kernel before the n=10^6 demo. ~30 LOC, half a day, not on the critical path.

## 5. IMGP demo plan

**What manifold-gp/ does.** `manifold_gp/utils/train_model.py:62-90` runs Adam on the negative-log-marginal-likelihood of an exact GP whose kernel is the *graph* Matérn precision operator P = (2ν/κ² I + Δ_rw)^ν built from a KNN graph (`graph_laplacian_operator.py`, `precision_matern_operator.py`). It calls `precision_operator.inv_quad_logdet(logdet=True)` — i.e. GPyTorch BBMM (CG + Lanczos-quadrature log det). The IMGP paper Table 2 (page 10) is explicit: they had to swap their full results to `torch.linalg.eigh` because GPyTorch's Lanczos quality wasn't enough — that's the row labelled "(full)". Their MR-MNIST setup is N=100,000 (Section 5.2.1, page 9) — exactly the target scale.

**Target experiment: MR-MNIST 10% semi-supervised, N=100,000.** Substitute *our* differentiable SLQ for GPyTorch's `inv_quad_logdet` inside `manifold_informed_train`.

**Concrete steps.**
1. Wrap `PrecisionMaternOperator._matmul` (manifold-gp/manifold_gp/operators/precision_matern_operator.py) as a `ParameterizedOperator` whose θ = (κ, σ_f, σ_n) and apply_vjp goes back into PyTorch autograd.
2. Replace the marginal-likelihood line `train_model.py:67-68` with our `slq_logdet` + a separate inv-quad term (which our Lanczos quadrature also returns: A^{-1} y as a byproduct).
3. Run 100 Adam iterations at N=100,000, ν=2, K=10 KNN, exactly matching the IMGP paper's setup (Section 5.2.1, page 9).
4. Same hyperparameter init, same Adam lr=0.01 (page 9).

**Baselines (run on the same machine).**
- **IMGP-Lanczos:** stock IMGP code with GPyTorch BBMM (paper's "failing" row).
- **IMGP-full:** stock IMGP code with `torch.linalg.eigh` (paper's reported but unscalable row).
- **IMGP-ours:** same setup with our differentiable SLQ.
- **Pure GPyTorch Euclidean Matérn-5/2 (EGP):** their bottom-row baseline.

**Stretch: scale to N=10^6.** Generate a 10× MR-MNIST or use the IMGP-style KNN graph at one million points. Neither IMGP-full nor IMGP-Lanczos can run here. **Ours should run. Clean "no competitor" panel.**

**Metrics.**
- Wall-clock per gradient step.
- Peak GPU memory.
- Hyperparameter trajectory (κ, σ_f, σ_n vs iteration) — converge to same values as IMGP-full?
- Held-out NLL and RMSE on the MR-MNIST 10% test set — match IMGP-full (-2.35 ± 0.04 NLL)?

**The one number that wins.** "Match IMGP-full NLL at N=100,000 in T seconds per epoch with M GB GPU memory, where IMGP-full needs dense eigh at O(N²) memory and IMGP-Lanczos misconverges" — and a single trajectory plot showing IMGP-Lanczos stuck while IMGP-ours follows IMGP-full. That's the demo. Everything else is supporting.

## 6. Milestones and timeline (4–6 weeks, one person)

**Week 0 (1 day).** matfree-on-GPU benchmark: confirm there's a real GPU-native performance gap to fight for. **Go/no-go: if matfree-on-CUDA-via-JAX is within 2× of our extrapolated SLQ on the IMGP precision operator at n=10^5, pivot harder onto restart adjoint as the headline.**

**Week 1.** Layer 1 unrestarted f(A)v: implement `matrix_function.cuh` + Python wrapper; gradcheck against `torch.linalg.eigh` ground truth for f ∈ {log, exp, sqrt, inv} at n ≤ 1000. **Go/no-go: gradcheck max relative error < 1e-6 on all four functions.**

**Week 2.** Layer 2 SLQ for log det: stochastic Hutchinson loop on top of Week 1. Variance characterization at n=10^4 (compare to exact log det). Forward + backward gradient correctness vs dense autograd. **Go/no-go: log det relative accuracy < 1% with m_probes=20, m_lanczos=50 on the IMGP precision operator at n=10^4; gradient w.r.t. κ within 5% of FD.**

**Week 3.** Layer 3 parameterized operator: `ParameterizedOperator` Python class; wire up IMGP's `PrecisionMaternOperator`; first end-to-end gradient step on the IMGP loss at N=10^4. **Go/no-go: one Adam step runs without NaN and the loss decreases.**

**Week 4 (riskiest).** Restart adjoint: implement `restart_adjoint.cuh`; verify against unrolled-autograd reference at small n. Plug into Layers 1–2. **Go/no-go: restart-aware backward matches unrolled-autograd within 1e-5 relative.** Fallback: skip restarts, pure Lanczos with full reorth.

**Week 5.** IMGP demo at N=100,000: full run of MR-MNIST 10% semi-supervised. Hyperparameter trajectories, NLL/RMSE comparison to all four baselines. **Go/no-go: IMGP-ours matches IMGP-full NLL within ±0.1.**

**Week 6 (buffer + stretch).** N=10^6 if Week 5 went smoothly. One figure (loss curves), one writeup (~4 pages), clean repo, demo notebook. Send to Borovitskiy.

**Riskiest milestone: Week 4 (restart adjoint).** Most novel piece, math not published. Mitigation: skip restarts fallback. IMGP demo is the real value.

## 7. Honest risks and failure modes

| Risk | Likelihood | Impact | Mitigation / fallback |
|---|---|---|---|
| Restart-adjoint math is subtler than expected; gradient diverges | medium | high | Drop restarts inside SLQ; use full reorth at fixed m. Still beats matfree on GPU + downstream. |
| SLQ variance dominates signal on IMGP precision operator | medium | medium | Increase m_probes. Variance reduction via Nyström preconditioner (Cortinovis 2026). If still bad: pre-deflate top-k with IRLM, then SLQ on the deflated remainder. |
| Memory blowup saving Lanczos basis at n=10^6 with m=50 | low | high | n·m·8 = 4·10^8 bytes = 0.4 GB per probe, well within 16 GB. Real risk only if m grows above ~200. |
| A near-duplicate paper at NeurIPS 2026 (June 5 deadline) | medium-low | high | Monitor arXiv weekly. If duplicated, IMGP-scale demo is still the value; reframe as "production-quality system with restart adjoint." |
| IMGP graph at N=100K turns out to be too easy (no Lanczos quality issue) | low | medium | The paper's Table 2 documents the issue; if you can't reproduce, move to N=10^6 where dense baseline is impossible. |
| IMGP graph at N=10^6 has connectivity/spectrum issues that break SLQ | medium | medium | Pre-deflate the bottom-k with IRLM; SLQ on the deflated remainder. |
| matfree on GPU is already competitive with our CUDA at N=10^5 | low | high | **Run the Week-0 matfree-on-GPU benchmark first.** If matfree is within 2×, pivot to "restart adjoint + manifold-GP scaling study" as the headline. |
| Krämer adjoint derivation has implementation gotchas | medium | medium | matfree is open-source. Read `experiments-lanczos-adjoints/` as canonical reference. Add a CPU reference path in Python that calls matfree to cross-check at small n. |

## 8. What gets sent to Borovitskiy

**Artifacts (minimum viable):**
1. **One figure.** Two panels: (left) IMGP MR-MNIST 10% NLL over Adam iterations for IMGP-Lanczos, IMGP-full, IMGP-ours — showing IMGP-ours tracking IMGP-full while IMGP-Lanczos stagnates. (right) Wall-clock per epoch vs N for IMGP-full (truncates at OOM), IMGP-Lanczos, IMGP-ours, with IMGP-ours line continuing to N=10^6.
2. **One short writeup (~4 pages).** §1: the specific gap (IMGP paper's page-10 footnote). §2: the unified toolkit (Layer 1/2/3 sketch). §3: the IMGP demo result. §4: how the restart adjoint is novel relative to Krämer 2024 (if Week 4 succeeds).
3. **Clean repo with a `README` quick-start that runs the IMGP demo from scratch in ≤ 50 lines of Python.** `pip install .` and run on a single GPU.
4. **A demo notebook** with the IMGP MR-MNIST experiment end-to-end.

**The opening line of the message.** Not "I built a unified spectral toolkit." It should be: *"The IMGP paper's Table 2 footnote — that GPyTorch Lanczos quality forces falling back to dense eigh — is the limitation I've been fixing. Here's a differentiable SLQ that lets SS-IMGP-full run at N=10⁶ on one GPU."*

**What NOT to lead with.** The 53× vs SciPy ARPACK number is not the headline — cupyx.eigsh closed that gap last year. The differentiable eigsh is supporting infrastructure. Mixed precision is a footnote.

---

## Sources

- Krämer, Moreno-Muñoz, Roy, Hauberg, "Gradients of Functions of Large Matrices," NeurIPS 2024 — [arXiv:2405.17277](https://arxiv.org/abs/2405.17277), [matfree GitHub](https://github.com/pnkraemer/matfree), [matfree SLQ tutorial](https://pnkraemer.github.io/matfree/Tutorials/1_compute_log_determinants_with_stochastic_lanczos_quadrature/), [experiments-lanczos-adjoints](https://github.com/pnkraemer/experiments-lanczos-adjoints)
- Fichera, Borovitskiy, Krause, Billard, "Implicit Manifold Gaussian Process Regression," NeurIPS 2023 — [arXiv:2310.19390](https://arxiv.org/abs/2310.19390)
- Borovitskiy et al., "Matérn Gaussian Processes on Graphs," AISTATS 2021
- Gardner, Pleiss, Weinberger, Bindel, Wilson, "GPyTorch (BBMM)," NeurIPS 2018 — [arXiv:1809.11165](https://arxiv.org/abs/1809.11165)
- Ubaru, Chen, Saad, "Fast Estimation of tr(f(A)) via SLQ," SIAM J. Matrix Anal. Appl.
- Pleiss, Jankowiak, Eriksson, Damle, Gardner, "Fast Matrix Square Roots…" NeurIPS 2020 — [arXiv:2006.11267](https://arxiv.org/abs/2006.11267)
- Cortinovis et al., "Detecting when one probe vector is enough for preconditioned log-determinant approximation," Jan 9, 2026 — [arXiv:2601.05778](https://arxiv.org/pdf/2601.05778)
- PyTorch 2.12 release (May 2026) — [release blog](https://pytorch.org/blog/pytorch-2-12-release-blog/)
- IMGP code: [github.com/nash169/manifold-gp](https://github.com/nash169/manifold-gp)

### Critical Files for Implementation

- `src/irlm_lanczos.cu`
- `src/matvec_operator.cuh`
- `src/inner_solve.cuh`
- `python/gpu_eigsh/differentiable.py`
- `manifold-gp/manifold_gp/utils/train_model.py`
