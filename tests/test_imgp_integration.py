#!/usr/bin/env python3
"""
Tests for the IMGP integration layer (`gpu_eigsh.imgp`).

  (1) Our symmetric-Laplacian matvec L_sym(κ) · v matches a dense
      reference Laplacian assembled from the same edge data.
  (2) Our IMGP precision matvec P(κ, θ) · v matches the dense reference
      `(2ν/θ² I + L_sym(κ))^ν · v`.
  (3) Our differentiable SLQ log-det of the IMGP precision operator gives
      the right gradients in `lengthscale` and `graphbandwidth` versus
      analytic ground truth (dense slogdet through autograd).

The IMGP source classes themselves are not imported here — the upstream
package depends on a `torch_sparse` binary that is incompatible with the
PyTorch in this env. The dense reference is mathematically equivalent and
suffices to verify the matvec math.
"""
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "manifold-gp"))

import numpy as np
import torch

from gpu_eigsh.imgp import (
    _matmul_symmetric_laplacian,
    make_imgp_precision_matvec,
    imgp_neg_marginal_log_likelihood,
)


def build_knn_graph(n, dim=5, k=10, seed=0):
    """Synthetic KNN graph in IMGP's edge-list format.

    Returns
    -------
    x_edge_dists : (M,) tensor — squared edge distances d²_e
    idx          : (2, M) long tensor — [src, dst] edge indices (each edge
                   appears once with src < dst)
    """
    rng = np.random.default_rng(seed)
    pts = rng.standard_normal((n, dim))
    from scipy.spatial import cKDTree
    tree = cKDTree(pts)
    dists, idxs = tree.query(pts, k=k + 1)   # +1 because the first neighbour is self
    nbr_dists = dists[:, 1:]
    nbr_idxs = idxs[:, 1:]

    src = np.repeat(np.arange(n), k)
    dst = nbr_idxs.ravel()
    d_sq = (nbr_dists ** 2).ravel()

    # De-duplicate: keep src < dst only.
    keep = src < dst
    src, dst, d_sq = src[keep], dst[keep], d_sq[keep]
    idx = torch.from_numpy(np.stack([src, dst], axis=0)).long()
    x_edge_dists = torch.from_numpy(d_sq).double()
    return x_edge_dists, idx


# ---------------------------------------------------------------------
# (1) Symmetric Laplacian matvec — compare against IMGP's operator
# ---------------------------------------------------------------------

def dense_symmetric_laplacian(
    x_edge_dists: torch.Tensor,
    idx: torch.Tensor,
    n: int,
    kappa: torch.Tensor,
    self_loops: bool = True,
) -> torch.Tensor:
    """Pure-PyTorch dense reference for IMGP's symmetric normalised Laplacian.

    Returns the n × n matrix L_sym(κ) explicitly. Used only as a
    ground-truth in tests; we never materialise this in production code.
    """
    kappa2 = kappa.square()
    adj_unnorm = torch.exp(-x_edge_dists / (4 * kappa2))
    src, dst = idx[0], idx[1]

    deg_unnorm = (torch.ones(n) if self_loops else torch.zeros(n)).to(adj_unnorm)
    deg_unnorm = deg_unnorm.scatter_add(0, src, adj_unnorm)
    deg_unnorm = deg_unnorm.scatter_add(0, dst, adj_unnorm)

    adjacency = adj_unnorm / (deg_unnorm[src] * deg_unnorm[dst])

    deg = deg_unnorm.pow(-2) if self_loops else torch.zeros_like(deg_unnorm)
    deg = deg.scatter_add(0, src, adjacency)
    deg = deg.scatter_add(0, dst, adjacency)

    deg_sqrt = deg.sqrt()
    triu = adjacency / (deg_sqrt[src] * deg_sqrt[dst]) / kappa2
    diag = ((1.0 - deg_unnorm.pow(-2) / deg) / kappa2
            if self_loops
            else torch.ones(n).to(kappa2) / kappa2)

    L = torch.diag(diag)
    for e in range(idx.shape[1]):
        i, j = int(src[e]), int(dst[e])
        L[i, j] = -triu[e]
        L[j, i] = -triu[e]
    return L


def test_laplacian_matvec_matches_dense_reference():
    print("(1) Symmetric Laplacian L_sym · v matches dense reference")
    print("-" * 65)
    all_pass = True
    for n in [50, 200]:
        x_edge_dists, idx = build_knn_graph(n)
        kappa = torch.tensor(1.0, dtype=torch.float64)
        v = torch.randn(n, dtype=torch.float64)

        out_ours = _matmul_symmetric_laplacian(
            v, x_edge_dists, idx, n, kappa, self_loops=True
        )
        L_dense = dense_symmetric_laplacian(x_edge_dists, idx, n, kappa)
        out_ref = L_dense @ v

        err = (out_ours - out_ref).norm() / out_ref.norm()
        ok = float(err) < 1e-10
        all_pass &= ok
        print(f"  n={n:>4}: ||ours - ref|| / ||ref|| = {float(err):.2e}  "
              f"{'PASS' if ok else 'FAIL'}")
    return all_pass


# ---------------------------------------------------------------------
# (2) Precision matvec — compare against IMGP's PrecisionMaternOperator
# ---------------------------------------------------------------------

def dense_precision(L_dense: torch.Tensor, lengthscale: torch.Tensor,
                    nu: int) -> torch.Tensor:
    """Dense ground truth: P = (2ν/θ² I + L_sym)^ν."""
    n = L_dense.shape[0]
    diag = (2 * nu) / lengthscale.square()
    inner = diag * torch.eye(n, dtype=L_dense.dtype) + L_dense
    P = torch.eye(n, dtype=L_dense.dtype)
    for _ in range(nu):
        P = inner @ P
    return P


def test_precision_matvec_matches_dense_reference():
    print("\n(2) Precision matvec P · v matches dense reference")
    print("-" * 65)
    all_pass = True
    for n in [50, 200]:
        for nu in [1, 2]:
            x_edge_dists, idx = build_knn_graph(n)
            kappa = torch.tensor(1.0, dtype=torch.float64)
            lengthscale = torch.tensor(0.5, dtype=torch.float64)
            v = torch.randn(n, dtype=torch.float64)

            matvec = make_imgp_precision_matvec(
                x_edge_dists=x_edge_dists, idx=idx, operator_dim=n,
                nu=nu, normalization="symmetric", self_loops=True,
            )
            out_ours = matvec(v, lengthscale, kappa)

            L_dense = dense_symmetric_laplacian(x_edge_dists, idx, n, kappa)
            P_dense = dense_precision(L_dense, lengthscale, nu)
            out_ref = P_dense @ v

            err = (out_ours - out_ref).norm() / out_ref.norm()
            ok = float(err) < 1e-10
            all_pass &= ok
            print(f"  n={n:>4} ν={nu}: ||ours - ref|| / ||ref|| = "
                  f"{float(err):.2e}  {'PASS' if ok else 'FAIL'}")
    return all_pass


# ---------------------------------------------------------------------
# (3) SLQ log-det gradient on IMGP precision operator
# ---------------------------------------------------------------------

def test_slq_logdet_gradient_imgp():
    """Gradient of log det P(κ, θ) w.r.t. κ and θ via SLQ vs analytic
    ground truth (dense eigendecomposition)."""
    print("\n(3) SLQ ∂(log det P)/∂(lengthscale, graphbandwidth) vs analytic")
    print("-" * 65)
    all_pass = True
    n = 60
    nu = 2
    x_edge_dists, idx = build_knn_graph(n)

    for kappa_val, ell_val in [(1.0, 0.5), (2.0, 1.0)]:
        # Build P as a dense matrix for ground truth (small n).
        kappa_g = torch.tensor(kappa_val, dtype=torch.float64, requires_grad=True)
        ell_g = torch.tensor(ell_val, dtype=torch.float64, requires_grad=True)

        # Dense P via the matvec applied to identity columns
        matvec = make_imgp_precision_matvec(
            x_edge_dists=x_edge_dists, idx=idx, operator_dim=n,
            nu=nu, normalization="symmetric", self_loops=True,
        )
        I = torch.eye(n, dtype=torch.float64)
        P_dense = torch.stack(
            [matvec(I[:, i], ell_g, kappa_g) for i in range(n)],
            dim=1,
        )
        logdet_exact = torch.linalg.slogdet(P_dense)[1]
        logdet_exact.backward()
        g_ell_exact = float(ell_g.grad)
        g_kappa_exact = float(kappa_g.grad)

        # SLQ estimate
        kappa_s = torch.tensor(kappa_val, dtype=torch.float64, requires_grad=True)
        ell_s = torch.tensor(ell_val, dtype=torch.float64, requires_grad=True)
        from gpu_eigsh.funm_torch import slq_logdet
        ld_slq = slq_logdet(
            make_imgp_precision_matvec(
                x_edge_dists=x_edge_dists, idx=idx, operator_dim=n,
                nu=nu, normalization="symmetric", self_loops=True,
            ),
            n, (ell_s, kappa_s),
            m_probes=200, lanczos_m=20, seed=0,
            dtype=torch.float64, device="cpu",
        )
        ld_slq.backward()
        g_ell_slq = float(ell_s.grad)
        g_kappa_slq = float(kappa_s.grad)

        ld_err = abs(float(ld_slq) - float(logdet_exact)) / abs(float(logdet_exact))
        g_ell_err = abs(g_ell_slq - g_ell_exact) / abs(g_ell_exact)
        g_kappa_err = abs(g_kappa_slq - g_kappa_exact) / abs(g_kappa_exact)

        ok_ld = ld_err < 0.05
        ok_ell = g_ell_err < 0.1
        ok_kappa = g_kappa_err < 0.1
        ok = ok_ld and ok_ell and ok_kappa
        all_pass &= ok
        print(f"  κ={kappa_val:.2f} θ={ell_val:.2f}:")
        print(f"    log det:        SLQ={float(ld_slq):.3f} "
              f"exact={float(logdet_exact):.3f}  rel_err={ld_err:.2e}")
        print(f"    ∂/∂θ:           SLQ={g_ell_slq:.4f} "
              f"exact={g_ell_exact:.4f}  rel_err={g_ell_err:.2e}")
        print(f"    ∂/∂κ:           SLQ={g_kappa_slq:.4f} "
              f"exact={g_kappa_exact:.4f}  rel_err={g_kappa_err:.2e}  "
              f"{'PASS' if ok else 'FAIL'}")
    return all_pass


def main():
    p1 = test_laplacian_matvec_matches_dense_reference()
    p2 = test_precision_matvec_matches_dense_reference()
    p3 = test_slq_logdet_gradient_imgp()
    print("\n" + "=" * 65)
    print(f"(1) L_sym matvec vs dense ref:     {'PASS' if p1 else 'FAIL'}")
    print(f"(2) Precision matvec vs dense ref: {'PASS' if p2 else 'FAIL'}")
    print(f"(3) SLQ log-det gradient:          {'PASS' if p3 else 'FAIL'}")
    print("=" * 65)
    sys.exit(0 if (p1 and p2 and p3) else 1)


if __name__ == "__main__":
    main()
