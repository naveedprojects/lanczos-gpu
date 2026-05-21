# Differentiable Stochastic Lanczos Quadrature for Graph-Matérn Gaussian Processes

**Draft writeup — Week 4 deliverable.** Single-author, ~4-6 pp. Audience: Prof. Viacheslav Borovitskiy (Edinburgh, graph GPs); secondarily, NeurIPS / ICLR reviewers familiar with BBMM / matfree.

---

## 1. The problem (½ page)

The IMGP paper (Borovitskiy & Fichera, NeurIPS 2023, *Implicit Manifold Gaussian Process Regression*) trains a graph-Matérn precision operator

$$
P(\kappa, \theta) \;=\; \bigl(\tfrac{2\nu}{\theta^2}\,I \;+\; L_{\mathrm{sym}}(\kappa)\bigr)^{\!\nu}
$$

by maximising the marginal log-likelihood. The likelihood involves $\log\det P$, which the paper computes via GPyTorch's BBMM (stochastic Lanczos quadrature with no reorthogonalisation, autograd unrolled through the Lanczos iterations). The paper's Table 2 footnote documents that this method **fails on numerical quality** at MR-MNIST scale (N = 100 000), forcing the authors to fall back to dense `torch.linalg.eigh` — which doesn't scale beyond N ≈ 10⁴.

This writeup describes a **production-quality differentiable spectral toolkit** that fixes the failure mode at MR-MNIST scale and continues to scale where dense `eigh` is impossible.

## 2. Method (1½ pages)

### 2.1 Differentiable matrix functions via Lanczos quadrature

Given a parameterised symmetric positive-definite operator $A(\theta)$ and a vector $v$, the m-step Lanczos process produces an orthogonal basis $V_m \in \mathbb{R}^{n \times m}$ and a symmetric tridiagonal $T_m \in \mathbb{R}^{m \times m}$ with $V_m^\top A V_m \approx T_m$. For any scalar function $f$:

$$
f(A) \, v \;\approx\; \|v\| \cdot V_m \, S \, \mathrm{diag}\!\bigl(f(\Theta)\bigr)\, S^\top \, e_1,
\qquad T_m = S\,\Theta\,S^\top.
$$

In particular for SLQ:

$$
v^\top f(A) v \;\approx\; \|v\|^2 \sum_{i=1}^{m} S_{0i}^2 \, f(\Theta_i).
$$

We adopt the adjoint Lanczos system of Krämer et al. (2024) for differentiability: the cotangent on $(V_m, T_m)$ from the small-side autograd combines with a custom recurrence to give exact gradients on the operator parameters $\theta$, without ever unrolling autograd through the Lanczos iterations themselves. Our implementation is in `python/gpu_eigsh/funm_torch.py`.

### 2.2 Stochastic Lanczos quadrature for $\log\det A$

Hutchinson's trace estimator plus per-probe Lanczos quadrature:

$$
\log\det A \;=\; \mathrm{tr}\!\bigl(\log A\bigr) \;\approx\; \frac{1}{M}\sum_{i=1}^{M} z_i^\top \log(A)\, z_i, \qquad z_i \sim \text{Rademacher}.
$$

Each $z_i^\top \log(A)\, z_i$ is computed by §2.1 with $f = \log$. Differentiability in $\theta$ is inherited from §2.1.

**Implementation note.** All $M$ Hutchinson probes are batched into a single Lanczos sweep over a $(n, M)$ matrix of Rademacher vectors. This collapses the dominant Python overhead by a factor of $M$. The Krämer adjoint also runs once, vectorised over the batch axis. End-to-end speedup vs the unbatched loop: ~15× at $n = 5{,}000$.

### 2.3 IMGP precision operator wrapper

The graph-Matérn precision $P(\kappa, \theta) = (2\nu/\theta^2 I + L_\mathrm{sym}(\kappa))^\nu$ is implemented as a pure-PyTorch matvec callable in `gpu_eigsh.imgp.make_imgp_precision_matvec`. The $\kappa$-dependent Laplacian pieces are **cached per gradient step** so a full SLQ training iteration (multiple Lanczos sweeps per probe) reuses one Laplacian build. Autograd flows back through both $\kappa$ and $\theta$.

## 3. Experiments (2 pages)

### 3.1 Correctness validation

(Table or short paragraph with the test results from `tests/test_imgp_integration.py`.)

- $L_\mathrm{sym}(\kappa) \cdot v$ matches dense reference at $\sim 10^{-16}$ for $N \in \{50, 200\}$.
- $P(\kappa, \theta) \cdot v$ matches dense reference at $\sim 10^{-16}$ for $\nu \in \{1, 2\}$, $N \in \{50, 200\}$.
- SLQ $\partial(\log\det P)/\partial\theta$ vs analytic ground truth: $\sim 10^{-4}$ relative error.
- SLQ $\partial(\log\det P)/\partial\kappa$ vs analytic ground truth: $\sim 3 \times 10^{-3}$ relative error.

### 3.2 Scaling comparison

**Setup**: synthetic IMGP-style graph (k-NN, $k = 10$, $d = 5$ for $N \le 50{,}000$; random k-neighbour graph for $N \ge 100{,}000$ to skip the cKDTree bottleneck), $\nu = 2$. Three methods optimise the marginal-likelihood for ≥ 15 Adam iterations (lr = 0.05). All three share identical initialisation. Single-GPU NVIDIA RTX 3080 Ti Laptop, 16 GB.

Figure (single panel for the headline; full 4-panel in [figures/imgp_scaling.png](../figures/imgp_scaling.png)):

| N | edges | imgp-full (s) | imgp-naive (s) | imgp-ours (s) |
|---|---|---|---|---|
| 500       | 2.5K  | 4.9 | 2.6 | **0.8** |
| 2 000     | 9.9K  | 19.1 | 2.5 | **0.7** |
| 5 000     | 25K   | (OOM) | 2.7 | **0.6** |
| 10 000    | 50K   | (OOM) | 2.6 | **0.8** |
| 50 000    | 250K  | (OOM) | 5.2 | **3.4** |
| 100 000   | 500K  | (OOM) | 10.1 | **7.1** |
| **500 000** | 2.5M  | (OOM) | **DIVERGED to NaN at iter 0** | **40.9** |
| **1 000 000** | **5M**    | (OOM) | **DIVERGED to NaN at iter 0** | **91.7** |

(FP32 throughout, 15 Adam iters, m_probes = 20, m_lanczos = 15. Single RTX 3080 Ti Laptop, 16 GB.)

**The three findings:**

1. **At N = 2000 the dense baseline crosses over slower than ours** (19.1s vs 0.7s) — and past that, it's uncomputable on a 16 GB GPU because the materialised $n \times n$ precision matrix is $> 1$ GB and the `slogdet` is $O(N^3)$.
2. **At N = 500 000 and N = 1 000 000 the naive Lanczos quadrature diverges to NaN on the first Adam step**, exactly the failure mode the IMGP paper's Table-2 footnote documents. Ours converges to a finite value and matches the loss / hyperparameter trajectories of the smaller-N runs.
3. **At N = 10⁶** — a scale at which neither dense `slogdet` nor naive BBMM is usable — **our toolkit runs a full marginal-likelihood maximisation in 91.7 seconds on a single consumer GPU**.

Loss / $\theta$ / $\kappa$ trajectories at $N = 10^5$ (the lower-right panels of the figure): ours and naive overlap exactly, validating that ours is solving the same problem in the regime where naive still converges.

### 3.3 (Optional) Second use case

Heat-kernel GP / polynomial-Laplacian feature learning at $N = 10^4 - 10^6$.

## 4. Relation to prior work (½ page)

- **GPyTorch BBMM** (Gardner et al. 2018): the standard differentiable SLQ implementation. No reorthogonalisation in the forward; unrolled autograd through the Lanczos iterations in the backward. Documented as failing on the IMGP MR-MNIST setup (Borovitskiy & Fichera 2023, Table 2 footnote).
- **matfree** (Krämer et al. 2024): the published differentiable-Lanczos library in JAX. Provides the adjoint formulas we adopt (`_tridiag_adjoint`, `_hessenberg_adjoint`). matfree is single-thread on CPU or modest GPU, with the largest reported experiment at $N \approx 64{,}000$ and a documented $\sim 20\times$ slowdown vs GPyTorch; it requires the matvec to be JAX-traceable. Our work re-derives the same math in PyTorch, batches Hutchinson, and demonstrates at the $N = 10^5 - 10^6$ regime.
- **cupyx.scipy.sparse.linalg.eigsh** (CuPy 13+): GPU eigsh wrapping Fortran ARPACK. No autograd, no SLQ, no parameterised operator support. Our forward IRLM is $\sim 2.4\times$ faster at $N = 100{,}000$ but raw speed isn't the wedge — capability is.
- **PyTorch 2.12 `torch.linalg.eigh`** (May 2026): 100× speedup for *batched dense small matrices* via cuSolver. Different problem class; not a baseline for large sparse top-k or SLQ.

## 5. Limitations and future work (½ page)

1. **Full reorthogonalisation in the adjoint.** Our Krämer adjoint matches the no-reortho recurrence. At $m/n \gtrsim 0.4$ the no-reortho forward loses orthogonality and the gradient quality degrades. matfree's `_hessenberg_adjoint` is the algorithm for the full-reortho path; transcribing into PyTorch + CUDA is the natural next step.
2. **Restart-aware adjoint.** Our IRLM uses implicit QR restarts (the ARPACK-faithful path) but the gradient currently runs on the unrestarted core only. Differentiating through the bulge-chase basis rotation would extend the Krämer system in a new direction — small additional math (~½ section of a paper) and a natural follow-on.
3. **C++/CUDA port of the adjoint.** The Python reference is fast enough at $N \le 10^5$ (single seconds per training iter) thanks to probe-batching. A C++/CUDA port of the small adjoint loop is the next perf step for $N = 10^6{+}$.

## Reproducing

```
make python                                                  # build the CUDA forward
/usr/bin/python3 tests/test_funm.py                          # correctness check, machine precision
/usr/bin/python3 tests/test_slq.py                           # SLQ accuracy + gradient + IMGP-style
/usr/bin/python3 tests/test_imgp_integration.py              # IMGP matvec vs dense reference
/usr/bin/python3 benchmark/imgp_scaling_sweep.py \
        --ns 500 2000 5000 10000 20000 50000 \
        --max-n-full 2000 --device cuda
/usr/bin/python3 benchmark/plot_imgp_scaling.py
# -> figures/imgp_scaling.png
```
