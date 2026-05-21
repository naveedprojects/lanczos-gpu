"""
Integration with the IMGP (Implicit Manifold Gaussian Process) framework.

Provides a drop-in replacement for the log-determinant computation inside
`manifold_gp.utils.train_model.manifold_informed_train`, using our
differentiable stochastic Lanczos quadrature instead of GPyTorch's BBMM.

The IMGP graph Matérn precision operator is

        P(κ, θ)  =  (2ν / θ²  I  +  L_sym(κ))^ν

where κ is the graph bandwidth (controlling the graph Laplacian's edge
weights) and θ is the kernel lengthscale. Both κ and θ are trainable.
Marginal-likelihood maximisation requires ∂(log det P) / ∂κ and ∂/∂θ,
which is exactly what our SLQ provides.

The wrapper expects:
  - `lengthscale`, `graphbandwidth`: leaf tensors with `requires_grad=True`.
    Gradients accumulate into their `.grad` field after backward.
  - Other static graph structure (edge index, distance values, etc.)
    captured via a closure.

Example
-------
    >>> from gpu_eigsh.imgp import make_imgp_precision_matvec
    >>> from gpu_eigsh.funm_torch import slq_logdet
    >>>
    >>> matvec = make_imgp_precision_matvec(
    ...     x_edge_dists=edge_dists,
    ...     idx=edge_index,
    ...     operator_dim=n,
    ...     nu=2,
    ...     normalization="symmetric",
    ...     self_loops=True,
    ... )
    >>> logdet = slq_logdet(
    ...     matvec, n, (lengthscale, graphbandwidth),
    ...     m_probes=20, lanczos_m=50, seed=0,
    ...     dtype=torch.float64, device="cuda",
    ... )
    >>> logdet.backward()
    >>> # lengthscale.grad and graphbandwidth.grad are now populated.
"""
from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch

from .funm_torch import slq_logdet


# ---------------------------------------------------------------------
# Differentiable graph Laplacian matvec
# ---------------------------------------------------------------------

def _build_laplacian_pieces(
    x_edge_dists: torch.Tensor,
    idx: torch.Tensor,
    operator_dim: int,
    graphbandwidth: torch.Tensor,
    self_loops: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the four ingredients of a normalised graph Laplacian:

        adjacency_unnorm[e]  = exp(-d²_e / (4κ²))            (per edge)
        degree_unnorm[i]     = Σ_{e ∋ i} adjacency_unnorm[e]
        adjacency[e]         = adjacency_unnorm[e] / (D_u[i] · D_u[j])
        degree[i]            = Σ_{e ∋ i} adjacency[e]  ( + 1/D_u[i]² if self-loops)

    Mirrors IMGP's `GraphLaplacianOperator` properties (adjacency_unnorm_mat,
    degree_unnorm_mat, adjacency_mat, degree_mat). Re-derived here so the
    full computation is a single autograd graph through `graphbandwidth`.

    Returns
    -------
    adjacency : (M,) tensor, the "adjacency_mat" used by `laplacian_triu`.
    degree    : (N,) tensor, the "degree_mat" used by L's diagonal.
    laplacian_triu : (M,) tensor, the upper-triangular entries of L_sym
                      (one per edge), pre-divided by κ².
    """
    kappa2 = graphbandwidth.square().squeeze()
    adj_unnorm = torch.exp(-x_edge_dists / (4 * kappa2))

    # Degree (unnormalised). With self-loops, w_ii = 1 contributes to each i.
    src, dst = idx[0], idx[1]
    if self_loops:
        deg_unnorm = torch.ones(operator_dim, device=x_edge_dists.device,
                                dtype=x_edge_dists.dtype)
    else:
        deg_unnorm = torch.zeros(operator_dim, device=x_edge_dists.device,
                                 dtype=x_edge_dists.dtype)
    deg_unnorm = deg_unnorm.scatter_add(0, src, adj_unnorm)
    deg_unnorm = deg_unnorm.scatter_add(0, dst, adj_unnorm)

    # Normalised adjacency
    adjacency = adj_unnorm / (deg_unnorm[src] * deg_unnorm[dst])

    # Normalised degree (used by `degree_mat` in the IMGP code).
    if self_loops:
        deg = deg_unnorm.pow(-2)
    else:
        deg = torch.zeros_like(deg_unnorm)
    deg = deg.scatter_add(0, src, adjacency)
    deg = deg.scatter_add(0, dst, adjacency)

    # laplacian_triu = adjacency / (sqrt(deg)_i · sqrt(deg)_j) / κ²
    deg_sqrt = deg.sqrt()
    laplacian_triu = adjacency / (deg_sqrt[src] * deg_sqrt[dst]) / kappa2

    return adjacency, deg, laplacian_triu


def _laplacian_diag(
    deg_unnorm_self_loops: bool,
    operator_dim: int,
    idx: torch.Tensor,
    adjacency: torch.Tensor,
    deg: torch.Tensor,
    graphbandwidth: torch.Tensor,
    self_loops: bool,
    device,
    dtype,
) -> torch.Tensor:
    """Diagonal of the symmetric normalised Laplacian L_sym."""
    kappa2 = graphbandwidth.square().squeeze()
    if self_loops:
        # IMGP: (1 - D_u^{-2} · D^{-1}) / κ²  with D_u = degree_unnorm
        # but `deg_unnorm` was reduced into `deg` above (self-loop pow(-2)).
        # Here we use the recomputed identity: l_ii = (1 - deg_unnorm^{-2} / deg) / κ²
        # We need deg_unnorm; recompute it inline.
        src, dst = idx[0], idx[1]
        # The caller already computed adjacency from deg_unnorm; rebuild deg_unnorm.
        deg_unnorm_local = torch.ones(operator_dim, device=device, dtype=dtype)
        # We unfortunately don't have access to adj_unnorm here without recomputing.
        # To keep the API simple we expose adj_unnorm separately — see matvec below.
        raise RuntimeError("internal: use _matmul_symmetric_laplacian directly")
    else:
        return torch.ones(operator_dim, device=device, dtype=dtype) / kappa2


def _build_laplacian_cache(
    x_edge_dists: torch.Tensor,
    idx: torch.Tensor,
    operator_dim: int,
    graphbandwidth: torch.Tensor,
    self_loops: bool,
) -> dict:
    """Pre-compute all κ-dependent Laplacian pieces in a single autograd pass.

    Returns a dict with `laplacian_diag` (N,) and `laplacian_triu` (M,)
    — both still attached to the autograd graph through `graphbandwidth`.
    Subsequent matvecs then cost O(N + M) without rebuilding.
    """
    kappa2 = graphbandwidth.square().squeeze()
    adj_unnorm = torch.exp(-x_edge_dists / (4 * kappa2))
    src, dst = idx[0], idx[1]
    device, dtype = x_edge_dists.device, x_edge_dists.dtype

    if self_loops:
        deg_unnorm = torch.ones(operator_dim, device=device, dtype=dtype)
    else:
        deg_unnorm = torch.zeros(operator_dim, device=device, dtype=dtype)
    deg_unnorm = deg_unnorm.scatter_add(0, src, adj_unnorm)
    deg_unnorm = deg_unnorm.scatter_add(0, dst, adj_unnorm)

    adjacency = adj_unnorm / (deg_unnorm[src] * deg_unnorm[dst])

    if self_loops:
        deg = deg_unnorm.pow(-2)
    else:
        deg = torch.zeros_like(deg_unnorm)
    deg = deg.scatter_add(0, src, adjacency)
    deg = deg.scatter_add(0, dst, adjacency)

    if self_loops:
        laplacian_diag = (1.0 - deg_unnorm.pow(-2) / deg) / kappa2
    else:
        laplacian_diag = torch.ones(operator_dim, device=device, dtype=dtype) / kappa2

    deg_sqrt = deg.sqrt()
    laplacian_triu = adjacency / (deg_sqrt[src] * deg_sqrt[dst]) / kappa2

    return {
        "laplacian_diag": laplacian_diag,
        "laplacian_triu": laplacian_triu,
        "src": src,
        "dst": dst,
    }


def _matmul_symmetric_laplacian_cached(
    rhs: torch.Tensor, cache: dict,
) -> torch.Tensor:
    """L_sym · rhs using a pre-computed cache of κ-dependent pieces."""
    vec_was_1d = (rhs.ndim == 1)
    vec = rhs.view(-1, 1) if vec_was_1d else rhs

    diag = cache["laplacian_diag"]
    triu = cache["laplacian_triu"]
    src = cache["src"]
    dst = cache["dst"]

    out = vec * diag.view(-1, 1)
    out = out.index_add(0, src, -triu.view(-1, 1) * vec[dst])
    out = out.index_add(0, dst, -triu.view(-1, 1) * vec[src])
    return out.squeeze(-1) if vec_was_1d else out


def _matmul_symmetric_laplacian(
    rhs: torch.Tensor,
    x_edge_dists: torch.Tensor,
    idx: torch.Tensor,
    operator_dim: int,
    graphbandwidth: torch.Tensor,
    self_loops: bool,
) -> torch.Tensor:
    """L_sym · rhs — convenience wrapper that builds the cache and applies.

    Kept for compatibility with tests that call this one-shot form.
    Production code should use `_build_laplacian_cache` once per gradient
    step and `_matmul_symmetric_laplacian_cached` for each matvec.
    """
    cache = _build_laplacian_cache(
        x_edge_dists, idx, operator_dim, graphbandwidth, self_loops
    )
    return _matmul_symmetric_laplacian_cached(rhs, cache)


# ---------------------------------------------------------------------
# IMGP precision matvec
# ---------------------------------------------------------------------

def make_imgp_precision_matvec(
    *,
    x_edge_dists: torch.Tensor,
    idx: torch.Tensor,
    operator_dim: int,
    nu: int,
    normalization: str = "symmetric",
    self_loops: bool = True,
) -> Callable[..., torch.Tensor]:
    """Build a differentiable matvec for the IMGP precision operator

        P(κ, θ)  =  (2ν / θ²  I  +  L_sym(κ))^ν

    where κ is graphbandwidth and θ is lengthscale.

    The returned callable takes (rhs, lengthscale, graphbandwidth) and
    returns P · rhs. Autograd flows back through both lengthscale and
    graphbandwidth.

    Only `normalization='symmetric'` is supported (the case used for the
    SS-IMGP-full row in the paper's Table 2).
    """
    if normalization != "symmetric":
        raise NotImplementedError(
            "Only normalization='symmetric' is implemented "
            "(the case used by SS-IMGP-full)."
        )

    # Cache of (kappa_value_id -> laplacian_cache). Reset when kappa changes.
    # We keep one entry so memory stays bounded and stale graphs are released.
    _cache_state = {"kappa_id": None, "cache": None}

    def _get_cache(graphbandwidth: torch.Tensor) -> dict:
        # We key on the autograd version + storage id of the param so a
        # fresh leaf clone (as our SLQ backward creates) gets a fresh cache.
        # This keeps the autograd graph correct.
        key = (id(graphbandwidth), graphbandwidth._version)
        if _cache_state["kappa_id"] != key:
            _cache_state["kappa_id"] = key
            _cache_state["cache"] = _build_laplacian_cache(
                x_edge_dists, idx, operator_dim, graphbandwidth, self_loops
            )
        return _cache_state["cache"]

    def matvec(rhs: torch.Tensor,
               lengthscale: torch.Tensor,
               graphbandwidth: torch.Tensor) -> torch.Tensor:
        cache = _get_cache(graphbandwidth)
        diag = lengthscale.square().squeeze() / (2 * nu)
        out = rhs
        for _ in range(nu):
            L_out = _matmul_symmetric_laplacian_cached(out, cache)
            out = (out + diag * L_out) / diag
        return out

    return matvec


# ---------------------------------------------------------------------
# Differentiable IMGP marginal log-likelihood
# ---------------------------------------------------------------------

def imgp_neg_marginal_log_likelihood(
    train_targets: torch.Tensor,
    *,
    x_edge_dists: torch.Tensor,
    idx: torch.Tensor,
    operator_dim: int,
    nu: int,
    lengthscale: torch.Tensor,
    graphbandwidth: torch.Tensor,
    m_probes: int = 20,
    lanczos_m: int = 50,
    seed: int = 0,
    normalization: str = "symmetric",
    self_loops: bool = True,
) -> torch.Tensor:
    """Negative log marginal likelihood for an IMGP-style precision-parameterised
    GP, with the log-determinant computed via our differentiable SLQ.

    Returns the per-data-point NLL:

        -log p(y)  ≈  (1 / N) ·  0.5 · [ yᵀ P y  -  log det P  +  N log 2π ]

    Both the inv_quad term (yᵀ P y) and the log-det term are differentiable
    in `lengthscale` and `graphbandwidth`. The Hutchinson estimator's
    variance is controlled by `m_probes`.
    """
    n = operator_dim
    matvec = make_imgp_precision_matvec(
        x_edge_dists=x_edge_dists, idx=idx, operator_dim=n,
        nu=nu, normalization=normalization, self_loops=self_loops,
    )

    # Quadratic form yᵀ P y (exact, no Lanczos approximation).
    Py = matvec(train_targets, lengthscale, graphbandwidth)
    inv_quad = torch.dot(train_targets, Py)

    # Log determinant via SLQ. Differentiable in (lengthscale, graphbandwidth).
    logdet = slq_logdet(
        matvec, n, (lengthscale, graphbandwidth),
        m_probes=m_probes, lanczos_m=lanczos_m, seed=seed,
        dtype=train_targets.dtype, device=train_targets.device,
    )

    log_2pi = float(torch.log(torch.tensor(2.0 * torch.pi)))
    nll = 0.5 * (inv_quad - logdet + n * log_2pi)
    return nll / n
