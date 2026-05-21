"""
Training loops for IMGP-style graph Matérn GPs.

Implements three variants of marginal-likelihood maximisation, sharing
identical data, initialisation, and optimiser. They differ only in how
log det P(κ, θ) is computed:

  imgp_train_ours
      Differentiable SLQ — our toolkit.
  imgp_train_dense
      Dense `torch.linalg.slogdet` on the full operator.
      This is the IMGP paper's "SS-IMGP-full" baseline; correct but
      O(N²) memory and O(N³) compute.
  imgp_train_gpytorch_style
      GPyTorch-flavoured BBMM-style log-det: a *single* Lanczos sweep on
      one Hutchinson probe, no reortho. Mimics the row of IMGP Table 2
      that the paper's footnote documents as failing on quality.

All three return a `TrainResult` with per-iteration metrics so the demo
script can plot trajectories side-by-side.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch

from .imgp import (
    make_imgp_precision_matvec,
    _matmul_symmetric_laplacian,
)
from .funm_torch import slq_logdet, lanczos_torch_noreortho


# ---------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------

@dataclass
class TrainResult:
    method: str
    final_loss: float
    final_lengthscale: float
    final_graphbandwidth: float
    iter_losses: list = field(default_factory=list)
    iter_lengthscales: list = field(default_factory=list)
    iter_graphbandwidths: list = field(default_factory=list)
    iter_wall_seconds: list = field(default_factory=list)
    peak_memory_bytes: int = 0
    diverged: bool = False
    diverged_at_iter: Optional[int] = None

    def summary_line(self) -> str:
        diverged = " [DIVERGED]" if self.diverged else ""
        return (
            f"{self.method:>16s}: loss={self.final_loss:>8.3f} "
            f"θ={self.final_lengthscale:>6.3f} κ={self.final_graphbandwidth:>6.3f}"
            f"{diverged}"
        )


# ---------------------------------------------------------------------
# Shared loss helpers
# ---------------------------------------------------------------------

def _imgp_inv_quad(
    matvec: Callable,
    train_targets: torch.Tensor,
    lengthscale: torch.Tensor,
    graphbandwidth: torch.Tensor,
) -> torch.Tensor:
    """y^T P y — exact, no Lanczos approximation."""
    Py = matvec(train_targets, lengthscale, graphbandwidth)
    return torch.dot(train_targets, Py)


def _imgp_naive_lanczos_logdet(
    matvec: Callable,
    n: int,
    lengthscale: torch.Tensor,
    graphbandwidth: torch.Tensor,
    m_probes: int,
    lanczos_m: int,
    seed: int,
) -> torch.Tensor:
    """GPyTorch BBMM-style log-det: unrolled-autograd, no-reortho Lanczos
    on Rademacher probes. This is the row of the IMGP Table 2 footnote.

    Unrolled autograd through the no-reortho Lanczos iteration suffers
    loss-of-orthogonality at moderate m, producing biased gradients —
    exactly the failure mode the IMGP paper documents.
    """
    dtype, device = lengthscale.dtype, lengthscale.device
    g = torch.Generator(device="cpu").manual_seed(seed)
    total = torch.zeros((), dtype=dtype, device=device)
    for _ in range(m_probes):
        z = (torch.randint(0, 2, (n,), generator=g, dtype=dtype) * 2.0 - 1.0).to(device)
        V_full, alpha, betas, z_norm = lanczos_torch_noreortho(
            lambda x: matvec(x, lengthscale, graphbandwidth), z, lanczos_m
        )
        T = (torch.diag(alpha)
             + torch.diag(betas[:-1], 1) + torch.diag(betas[:-1], -1))
        Theta, S = torch.linalg.eigh(T)
        # Guard against tiny / negative Ritz values created by orthogonality loss.
        Theta_safe = Theta.clamp_min(1e-30)
        c = S[0, :]
        total = total + z_norm * z_norm * (c * c * torch.log(Theta_safe)).sum()
    return total / m_probes


# ---------------------------------------------------------------------
# Training loops
# ---------------------------------------------------------------------

def _train_loop(
    *,
    method: str,
    logdet_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    matvec: Callable,
    train_targets: torch.Tensor,
    lengthscale_init: float,
    graphbandwidth_init: float,
    n: int,
    max_iter: int,
    lr: float,
    verbose: bool,
    divergence_loss_threshold: float = 1e6,
) -> TrainResult:
    """Generic Adam loop. `logdet_fn(lengthscale, graphbandwidth)` returns
    the (differentiable) log det term."""
    dtype = train_targets.dtype
    device = train_targets.device

    lengthscale = torch.tensor(lengthscale_init, dtype=dtype, device=device,
                               requires_grad=True)
    graphbandwidth = torch.tensor(graphbandwidth_init, dtype=dtype, device=device,
                                  requires_grad=True)

    opt = torch.optim.Adam([lengthscale, graphbandwidth], lr=lr)
    log_2pi = math.log(2.0 * math.pi)

    result = TrainResult(
        method=method, final_loss=float("nan"),
        final_lengthscale=lengthscale_init,
        final_graphbandwidth=graphbandwidth_init,
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    for it in range(max_iter):
        opt.zero_grad()
        try:
            inv_quad = _imgp_inv_quad(matvec, train_targets, lengthscale, graphbandwidth)
            logdet = logdet_fn(lengthscale, graphbandwidth)
            loss = 0.5 * (inv_quad - logdet + n * log_2pi) / n
        except RuntimeError as e:
            result.diverged = True
            result.diverged_at_iter = it
            if verbose:
                print(f"  [{method}] iter {it} forward exception: {e}")
            break

        if not torch.isfinite(loss) or abs(float(loss.detach())) > divergence_loss_threshold:
            result.diverged = True
            result.diverged_at_iter = it
            if verbose:
                print(f"  [{method}] iter {it} loss not finite or above "
                      f"threshold ({float(loss.detach())})")
            break

        loss.backward()
        if not all(torch.isfinite(p.grad).all()
                   for p in (lengthscale, graphbandwidth)
                   if p.grad is not None):
            result.diverged = True
            result.diverged_at_iter = it
            if verbose:
                print(f"  [{method}] iter {it} NaN/Inf in gradient")
            break

        opt.step()
        # Keep params positive (kernel hyperparameters must be > 0).
        with torch.no_grad():
            lengthscale.clamp_(min=1e-3)
            graphbandwidth.clamp_(min=1e-3)

        result.iter_losses.append(float(loss.detach()))
        result.iter_lengthscales.append(float(lengthscale.detach()))
        result.iter_graphbandwidths.append(float(graphbandwidth.detach()))
        result.iter_wall_seconds.append(time.perf_counter() - t0)

        if verbose and (it % 5 == 0 or it == max_iter - 1):
            print(f"  [{method}] it {it:>3d}  loss={float(loss):>8.3f}  "
                  f"θ={float(lengthscale):>6.3f}  κ={float(graphbandwidth):>6.3f}")

    if not result.diverged and result.iter_losses:
        result.final_loss = result.iter_losses[-1]
        result.final_lengthscale = result.iter_lengthscales[-1]
        result.final_graphbandwidth = result.iter_graphbandwidths[-1]
    if device.type == "cuda":
        result.peak_memory_bytes = int(torch.cuda.max_memory_allocated())
    return result


def imgp_train_ours(
    *,
    train_targets: torch.Tensor,
    x_edge_dists: torch.Tensor,
    idx: torch.Tensor,
    n: int,
    nu: int = 2,
    lengthscale_init: float = 1.0,
    graphbandwidth_init: float = 1.0,
    max_iter: int = 50,
    lr: float = 0.05,
    m_probes: int = 50,
    lanczos_m: int = 20,
    seed: int = 0,
    verbose: bool = False,
) -> TrainResult:
    """Marginal-likelihood maximisation with our differentiable SLQ log-det."""
    matvec = make_imgp_precision_matvec(
        x_edge_dists=x_edge_dists, idx=idx, operator_dim=n, nu=nu,
        normalization="symmetric", self_loops=True,
    )

    def logdet_fn(lengthscale, graphbandwidth):
        return slq_logdet(
            matvec, n, (lengthscale, graphbandwidth),
            m_probes=m_probes, lanczos_m=lanczos_m, seed=seed,
            dtype=lengthscale.dtype, device=lengthscale.device,
        )

    return _train_loop(
        method="imgp-ours", logdet_fn=logdet_fn, matvec=matvec,
        train_targets=train_targets,
        lengthscale_init=lengthscale_init,
        graphbandwidth_init=graphbandwidth_init,
        n=n, max_iter=max_iter, lr=lr, verbose=verbose,
    )


def imgp_train_dense(
    *,
    train_targets: torch.Tensor,
    x_edge_dists: torch.Tensor,
    idx: torch.Tensor,
    n: int,
    nu: int = 2,
    lengthscale_init: float = 1.0,
    graphbandwidth_init: float = 1.0,
    max_iter: int = 50,
    lr: float = 0.05,
    verbose: bool = False,
) -> TrainResult:
    """Marginal-likelihood maximisation with dense `torch.linalg.slogdet`
    on the materialised n×n precision operator. The IMGP paper's
    SS-IMGP-full row. O(N²) memory."""
    matvec = make_imgp_precision_matvec(
        x_edge_dists=x_edge_dists, idx=idx, operator_dim=n, nu=nu,
        normalization="symmetric", self_loops=True,
    )

    def logdet_fn(lengthscale, graphbandwidth):
        dtype, device = lengthscale.dtype, lengthscale.device
        I = torch.eye(n, dtype=dtype, device=device)
        cols = [matvec(I[:, i], lengthscale, graphbandwidth) for i in range(n)]
        P = torch.stack(cols, dim=1)
        return torch.linalg.slogdet(P)[1]

    return _train_loop(
        method="imgp-full", logdet_fn=logdet_fn, matvec=matvec,
        train_targets=train_targets,
        lengthscale_init=lengthscale_init,
        graphbandwidth_init=graphbandwidth_init,
        n=n, max_iter=max_iter, lr=lr, verbose=verbose,
    )


def imgp_train_naive_lanczos(
    *,
    train_targets: torch.Tensor,
    x_edge_dists: torch.Tensor,
    idx: torch.Tensor,
    n: int,
    nu: int = 2,
    lengthscale_init: float = 1.0,
    graphbandwidth_init: float = 1.0,
    max_iter: int = 50,
    lr: float = 0.05,
    m_probes: int = 10,
    lanczos_m: int = 30,
    seed: int = 0,
    verbose: bool = False,
) -> TrainResult:
    """Marginal-likelihood maximisation with naive (no-reortho, unrolled-
    autograd) Lanczos quadrature — the IMGP paper's reported failing row."""
    matvec = make_imgp_precision_matvec(
        x_edge_dists=x_edge_dists, idx=idx, operator_dim=n, nu=nu,
        normalization="symmetric", self_loops=True,
    )

    def logdet_fn(lengthscale, graphbandwidth):
        return _imgp_naive_lanczos_logdet(
            matvec, n, lengthscale, graphbandwidth,
            m_probes=m_probes, lanczos_m=lanczos_m, seed=seed,
        )

    return _train_loop(
        method="imgp-naive", logdet_fn=logdet_fn, matvec=matvec,
        train_targets=train_targets,
        lengthscale_init=lengthscale_init,
        graphbandwidth_init=graphbandwidth_init,
        n=n, max_iter=max_iter, lr=lr, verbose=verbose,
    )
