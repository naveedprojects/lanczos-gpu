#!/usr/bin/env python3
"""
Week-0 gap-confirmation benchmark.

Question: is matfree-on-GPU already competitive enough that the GPU-native
angle of our planned differentiable SLQ is dead?

Compares log det estimation + gradient w.r.t. operator parameter (κ) on the
IMGP-style precision operator A(κ) = (2ν/κ² I + L_sym), with ν=2 implicit
(we do log det of the base operator and multiply by 2 in the GP loss).

Three backends:
  1. matfree SLQ + JAX autograd (GPU)
  2. GPyTorch SLQ + PyTorch autograd (GPU)            -- IMGP paper baseline
  3. Exact dense torch.linalg.slogdet                  -- ground truth (small n only)

Reports forward time, backward time, log det accuracy, gradient accuracy,
and peak GPU memory.

Usage:
  XLA_PYTHON_CLIENT_PREALLOCATE=false /usr/bin/python3 benchmark/matfree_vs_baselines.py
"""
import os
import sys
import time
import json
import gc

# Ensure JAX does not pre-allocate the GPU (so torch can share)
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_ENABLE_X64", "true")

import numpy as np
import torch
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from matfree import stochtrace, decomp, funm

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)

NU = 2                     # IMGP Matérn smoothness
KAPPA = 1.0                # bandwidth init
N_PROBES = 20              # Hutchinson probe count
LANCZOS_M = 50             # Lanczos depth per probe
SEED = 0
NN = 30                    # KNN for graph Laplacian
DIM = 10                   # embedding dim for synthetic point cloud


# ----------------------------------------------------------------------------
# Build IMGP-style symmetric normalized graph Laplacian (CSR)
# ----------------------------------------------------------------------------

def build_graph_laplacian(n, seed=42):
    from scipy.spatial import cKDTree
    from scipy.sparse import csr_matrix

    rng = np.random.default_rng(seed)
    points = rng.standard_normal((n, DIM))
    tree = cKDTree(points)
    dists, idxs = tree.query(points, k=NN + 1)

    neighbor_dists = dists[:, 1:]
    neighbor_idxs = idxs[:, 1:]
    weights = np.exp(-neighbor_dists ** 2 / (4.0 * 1.0 ** 2))

    rows = np.repeat(np.arange(n), NN)
    cols = neighbor_idxs.ravel()
    vals = weights.ravel()

    A = csr_matrix((vals, (rows, cols)), shape=(n, n))
    A_sym = A.maximum(A.T)

    degrees = np.array(A_sym.sum(axis=1)).flatten()
    D_inv_sqrt = csr_matrix(
        (1.0 / np.sqrt(np.maximum(degrees, 1e-15)),
         (np.arange(n), np.arange(n))),
        shape=(n, n),
    )
    L = csr_matrix(
        (degrees, (np.arange(n), np.arange(n))), shape=(n, n)
    ) - A_sym
    L_sym = D_inv_sqrt @ L @ D_inv_sqrt
    return L_sym.tocsr()


# ----------------------------------------------------------------------------
# matfree: SLQ on the precision base operator
# ----------------------------------------------------------------------------

def matfree_slq_logdet(L_csr, kappa, n_probes, lanczos_m, seed=SEED, jit=True):
    """Compute log det((2ν/κ² I + L_sym)) via matfree SLQ on GPU.
    Returns (logdet_estimate, grad_kappa, fwd_time, bwd_time, peak_mem_bytes)."""
    n = L_csr.shape[0]
    # JAX BCOO sparse — preferred but cuSparse-via-jax has limited dtypes;
    # for the comparison to be apples-to-apples we instead express the matvec
    # via a JAX scipy sparse, or use scipy's matvec wrapped in pure_callback.
    # The cleanest path on JAX GPU is to upload as JAX BCOO.
    from jax.experimental import sparse as jsp

    L_coo = L_csr.tocoo()
    indices = jnp.stack([jnp.asarray(L_coo.row), jnp.asarray(L_coo.col)], axis=1)
    L_bcoo = jsp.BCOO((jnp.asarray(L_coo.data, dtype=jnp.float64), indices),
                      shape=(n, n))

    def matvec(x, kappa):
        # A(κ) = (2ν/κ² I + L_sym) x
        return (2 * NU / (kappa ** 2)) * x + L_bcoo @ x

    # matfree pipeline
    tridiag = decomp.tridiag_sym(lanczos_m, reortho="full")
    integrand = funm.integrand_funm_sym_logdet(tridiag)
    sampler = stochtrace.sampler_rademacher(jnp.zeros(n, dtype=jnp.float64),
                                            num=n_probes)
    estimator = stochtrace.estimator(integrand, sampler=sampler)

    key = jax.random.key(seed)

    def loss(kappa):
        mv = lambda x: matvec(x, kappa)
        return estimator(mv, key)

    if jit:
        loss = jax.jit(loss)

    # Warmup (JIT compile)
    val = loss(kappa)
    val.block_until_ready()

    # Forward
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.perf_counter()
    val = loss(kappa)
    val.block_until_ready()
    fwd_t = time.perf_counter() - t0

    # Backward (gradient w.r.t. κ)
    grad_fn = jax.grad(loss)
    if jit:
        grad_fn = jax.jit(grad_fn)
    g = grad_fn(kappa)
    g.block_until_ready()  # warmup

    t0 = time.perf_counter()
    g = grad_fn(kappa)
    g.block_until_ready()
    bwd_t = time.perf_counter() - t0

    # Memory (after compile) — approx via jax.live_arrays size
    try:
        peak = sum(a.nbytes for a in jax.live_arrays())
    except Exception:
        peak = -1

    return float(val), float(g), fwd_t, bwd_t, peak


# ----------------------------------------------------------------------------
# GPyTorch: SLQ via LinearOperator.inv_quad_logdet
# ----------------------------------------------------------------------------

def gpytorch_slq_logdet(L_csr, kappa, n_probes, lanczos_m, seed=SEED):
    """Compute log det using GPyTorch's stochastic Lanczos quadrature.

    Builds a LinearOperator the same way IMGP's PrecisionMaternOperator does:
    pass all autograd-tracked tensors as positional args to super().__init__,
    so the base class handles _bilinear_derivative via PyTorch autograd on
    _matmul.
    """
    import gpytorch
    from linear_operator.operators import LinearOperator
    from linear_operator.settings import (
        max_lanczos_quadrature_iterations,
        num_trace_samples,
    )

    n = L_csr.shape[0]
    device = torch.device("cuda")

    coo = L_csr.tocoo()
    indices = torch.from_numpy(np.stack([coo.row, coo.col], axis=0)).long().to(device)
    values = torch.from_numpy(coo.data).double().to(device)
    L_torch = torch.sparse_coo_tensor(indices, values, size=(n, n)).coalesce()

    kappa_t = torch.tensor(kappa, dtype=torch.float64, device=device,
                           requires_grad=True)

    NU_local = NU
    n_local = n
    L_local = L_torch

    class PrecOp(LinearOperator):
        def __init__(self, kappa):
            # Pass tracked tensor as positional arg, like IMGP does.
            super().__init__(kappa)
            self.kappa = kappa

        def _matmul(self, rhs):
            return (2 * NU_local / (self.kappa ** 2)) * rhs + \
                torch.sparse.mm(L_local, rhs)

        def _size(self):
            return torch.Size([n_local, n_local])

        def _transpose_nonbatch(self):
            return self  # symmetric

    op = PrecOp(kappa_t)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Warmup
    with num_trace_samples(n_probes), max_lanczos_quadrature_iterations(lanczos_m):
        rhs = torch.zeros(n, 1, dtype=torch.float64, device=device)
        _, ld = op.inv_quad_logdet(rhs, logdet=True)
        ld.backward()

    kappa_t.grad = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with num_trace_samples(n_probes), max_lanczos_quadrature_iterations(lanczos_m):
        rhs = torch.zeros(n, 1, dtype=torch.float64, device=device)
        _, ld = op.inv_quad_logdet(rhs, logdet=True)
    torch.cuda.synchronize()
    fwd_t = time.perf_counter() - t0

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    ld.backward()
    torch.cuda.synchronize()
    bwd_t = time.perf_counter() - t0

    peak = torch.cuda.max_memory_allocated()
    return float(ld.detach().cpu()), float(kappa_t.grad.cpu()), fwd_t, bwd_t, peak


# ----------------------------------------------------------------------------
# Exact dense ground truth
# ----------------------------------------------------------------------------

def exact_logdet(L_csr, kappa):
    n = L_csr.shape[0]
    if n > 5000:
        return None, None
    device = torch.device("cuda")
    A = torch.from_numpy(L_csr.toarray()).double().to(device)
    A = A + (2 * NU / kappa ** 2) * torch.eye(n, dtype=torch.float64, device=device)
    kappa_t = torch.tensor(kappa, dtype=torch.float64, device=device,
                           requires_grad=True)
    A2 = torch.from_numpy(L_csr.toarray()).double().to(device) + \
        (2 * NU / kappa_t ** 2) * torch.eye(n, dtype=torch.float64, device=device)
    sign, logdet = torch.linalg.slogdet(A2)
    logdet.backward()
    return float(logdet.detach().cpu()), float(kappa_t.grad.cpu())


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------

def bench_at_n(n):
    print(f"\n{'='*70}\n n = {n:,}\n{'='*70}")
    print(f"  building graph Laplacian (NN={NN}, dim={DIM})...")
    t0 = time.perf_counter()
    L = build_graph_laplacian(n)
    print(f"    nnz={L.nnz:,}, build={time.perf_counter()-t0:.2f}s")

    result = {"n": n, "nnz": L.nnz}

    # Exact (small n only)
    if n <= 5000:
        print(f"  exact dense slogdet...", end=" ", flush=True)
        t0 = time.perf_counter()
        ld_exact, g_exact = exact_logdet(L, KAPPA)
        print(f"{time.perf_counter()-t0:.2f}s  logdet={ld_exact:.4f}  grad={g_exact:.4f}")
        result["exact_logdet"] = ld_exact
        result["exact_grad"] = g_exact
    else:
        result["exact_logdet"] = None
        result["exact_grad"] = None

    # matfree
    print(f"  matfree SLQ (probes={N_PROBES}, m={LANCZOS_M})...", end=" ", flush=True)
    try:
        ld_mf, g_mf, fwd_mf, bwd_mf, mem_mf = matfree_slq_logdet(
            L, KAPPA, N_PROBES, LANCZOS_M
        )
        print(f"fwd={fwd_mf*1000:.1f}ms bwd={bwd_mf*1000:.1f}ms "
              f"ld={ld_mf:.4f} grad={g_mf:.4f}")
        result["matfree"] = {
            "logdet": ld_mf, "grad": g_mf, "fwd_ms": fwd_mf * 1000,
            "bwd_ms": bwd_mf * 1000, "peak_bytes": mem_mf,
        }
    except Exception as e:
        print(f"FAILED: {e}")
        result["matfree"] = {"error": str(e)}

    # Clear JAX before GPyTorch to avoid memory fight
    jax.clear_caches()
    gc.collect()
    torch.cuda.empty_cache()

    # GPyTorch
    print(f"  GPyTorch SLQ (probes={N_PROBES}, m={LANCZOS_M})...", end=" ", flush=True)
    try:
        ld_gp, g_gp, fwd_gp, bwd_gp, mem_gp = gpytorch_slq_logdet(
            L, KAPPA, N_PROBES, LANCZOS_M
        )
        print(f"fwd={fwd_gp*1000:.1f}ms bwd={bwd_gp*1000:.1f}ms "
              f"ld={ld_gp:.4f} grad={g_gp:.4f}")
        result["gpytorch"] = {
            "logdet": ld_gp, "grad": g_gp, "fwd_ms": fwd_gp * 1000,
            "bwd_ms": bwd_gp * 1000, "peak_bytes": mem_gp,
        }
    except Exception as e:
        print(f"FAILED: {e}")
        result["gpytorch"] = {"error": str(e)}

    torch.cuda.empty_cache()
    return result


def main():
    scales = [1000, 5000, 10000, 50000, 100000]
    out = []
    for n in scales:
        r = bench_at_n(n)
        out.append(r)
        gc.collect()

    # Summary
    print(f"\n\n{'='*70}\nSUMMARY\n{'='*70}")
    print(f"{'n':>8} {'matfree fwd':>14} {'gpyt fwd':>12} {'matfree bwd':>14} "
          f"{'gpyt bwd':>12} {'mf accuracy':>12} {'gp accuracy':>12}")
    for r in out:
        n = r["n"]
        mf = r.get("matfree", {})
        gp = r.get("gpytorch", {})
        exact = r.get("exact_logdet")

        def fmt(d, k, suffix="ms"):
            if "error" in d:
                return "ERR"
            v = d.get(k)
            return f"{v:.1f}{suffix}" if v is not None else "—"

        def acc(d):
            if "error" in d or exact is None:
                return "—"
            return f"{abs(d['logdet']-exact)/abs(exact):.1e}"

        print(f"{n:>8} {fmt(mf,'fwd_ms'):>14} {fmt(gp,'fwd_ms'):>12} "
              f"{fmt(mf,'bwd_ms'):>14} {fmt(gp,'bwd_ms'):>12} "
              f"{acc(mf):>12} {acc(gp):>12}")

    json_path = os.path.join(DATA_DIR, "matfree_vs_baselines.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {json_path}")


if __name__ == "__main__":
    main()
