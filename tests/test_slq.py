#!/usr/bin/env python3
"""
Tests for Layer 2 (SLQ for log-determinant) and Layer 3 (operator-callable
adjoint).

Three checks:

  (1) Operator-callable Krämer adjoint matches the dense-A Krämer adjoint
      when the operator is a dense matrix wrapped as a callable. Should
      agree to machine precision.

  (2) SLQ log-det accuracy: estimate log det A on a small SPD matrix and
      compare to torch.linalg.slogdet. The Hutchinson estimator has
      O(1/sqrt(m_probes)) variance, so the error should shrink with more
      probes — but at fixed seed and m_probes it should be deterministic.

  (3) SLQ log-det *gradient* w.r.t. an operator parameter:
        A(c) = c·I + L  (L fixed sparse Laplacian)
        ∂(log det A(c)) / ∂c = tr(A(c)^{-1})
      Compare SLQ-backward gradient vs analytic ground truth.
"""
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import numpy as np
import torch

from gpu_eigsh.funm_torch import (
    funm_qform_op,
    slq_logdet,
    funm_apply_kraemer,
    lanczos_torch_noreortho,
    _build_T,
    _apply_scalar,
)


def qform_unrolled(matvec, z, params, m, func):
    """Autograd-ground-truth quadratic form: q = ||z||² · Σ (S[0,i])² f(Θ_i)
    via unrolled no-reortho Lanczos. Matches the formulation of
    `funm_qform_op` exactly (used as gradcheck reference)."""
    V_full, alpha, betas, z_norm = lanczos_torch_noreortho(
        lambda x: matvec(x, *params), z, m
    )
    T = _build_T(alpha, betas[:-1])
    Theta, S = torch.linalg.eigh(T)
    fTheta = _apply_scalar(Theta, func)
    c = S[0, :]
    return z_norm * z_norm * (c * c * fTheta).sum()


def build_spd(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    M = torch.randn(n, n, generator=g, dtype=torch.float64)
    return M @ M.T + n * torch.eye(n, dtype=torch.float64)


# ---------------------------------------------------------------------
# (1) Operator-callable matches dense-A
# ---------------------------------------------------------------------

def test_op_vs_ground_truth():
    """Operator-callable Krämer adjoint vs autograd through the same
    quadratic-form formulation (matched formulation, no-reortho)."""
    print("(1) funm_qform_op vs autograd-unrolled qform (same formulation)")
    print("-" * 60)
    all_pass = True
    for n, m, func in [(20, 10, "log"), (50, 30, "log"), (100, 50, "log"),
                       (20, 10, "sqrt"), (50, 30, "inv")]:
        A = build_spd(n, seed=n)
        g = torch.Generator().manual_seed(n + 1)
        z = torch.randn(n, generator=g, dtype=torch.float64)

        def matvec(x, Aparam):
            return Aparam @ x

        # Ground truth: autograd through the *same* quadratic-form formula
        A1 = A.clone().requires_grad_(True); z1 = z.clone().requires_grad_(True)
        q1 = qform_unrolled(matvec, z1, (A1,), m, func)
        q1.backward()
        gA_truth = A1.grad.clone(); gz_truth = z1.grad.clone()

        # Our op-callable Krämer adjoint
        A2 = A.clone().requires_grad_(True); z2 = z.clone().requires_grad_(True)
        q2 = funm_qform_op(matvec, z2, (A2,), m, func)
        q2.backward()
        gA_op = A2.grad.clone(); gz_op = z2.grad.clone()

        fwd_err = (q1.detach() - q2.detach()).abs() / q1.detach().abs()
        gA_err = (gA_truth - gA_op).norm() / gA_truth.norm()
        gz_err = (gz_truth - gz_op).norm() / gz_truth.norm()
        ok = max(float(gA_err), float(gz_err), float(fwd_err)) < 1e-8
        all_pass &= ok
        print(f"  n={n:3d} m={m:3d} {func:>4}: "
              f"fwd={float(fwd_err):.2e} gA={float(gA_err):.2e} "
              f"gz={float(gz_err):.2e}  {'PASS' if ok else 'FAIL'}")
    return all_pass


# ---------------------------------------------------------------------
# (2) SLQ accuracy vs exact slogdet
# ---------------------------------------------------------------------

def test_slq_accuracy():
    print("\n(2) SLQ log-det accuracy vs torch.linalg.slogdet")
    print("-" * 60)
    all_pass = True
    n = 100
    A = build_spd(n, seed=42)
    exact = float(torch.linalg.slogdet(A)[1])

    def matvec(x):
        return A @ x

    for m_probes in [10, 50, 200]:
        est = slq_logdet(
            lambda x, *_: matvec(x), n, (),
            m_probes=m_probes, lanczos_m=60, seed=0,
            dtype=torch.float64, device="cpu",
        )
        rel = abs(float(est) - exact) / abs(exact)
        # Hutchinson variance scales 1/sqrt(m_probes); allow loose threshold.
        ok = rel < 0.1
        all_pass &= ok
        print(f"  m_probes={m_probes:3d} m=60: exact={exact:.4f} "
              f"est={float(est):.4f} rel_err={rel:.2e}  "
              f"{'PASS' if ok else 'FAIL'}")
    return all_pass


# ---------------------------------------------------------------------
# (3) SLQ gradient w.r.t. operator parameter
# ---------------------------------------------------------------------

def test_slq_gradient():
    print("\n(3) SLQ log-det gradient w.r.t. operator param c, A(c) = c·I + L")
    print("-" * 60)
    all_pass = True

    n = 50
    L = build_spd(n, seed=7)
    # Shift L so its smallest eigenvalue is small but positive.
    # We'll vary c and compare gradient.

    def matvec(x, c):
        return c * x + L @ x

    for c_val in [1.0, 5.0, 10.0]:
        c_t = torch.tensor(c_val, dtype=torch.float64, requires_grad=True)

        ld = slq_logdet(
            matvec, n, (c_t,),
            m_probes=100, lanczos_m=40, seed=123,
            dtype=torch.float64, device="cpu",
        )
        ld.backward()
        grad_c_slq = float(c_t.grad)

        # Analytic gradient: d/dc log det A(c) = tr(A(c)^{-1}).
        A_dense = c_val * torch.eye(n, dtype=torch.float64) + L
        grad_c_exact = float(torch.linalg.inv(A_dense).trace())

        rel = abs(grad_c_slq - grad_c_exact) / abs(grad_c_exact)
        ok = rel < 0.1
        all_pass &= ok
        print(f"  c={c_val:5.2f}: SLQ grad={grad_c_slq:.4f}  "
              f"exact tr(A^-1)={grad_c_exact:.4f}  "
              f"rel_err={rel:.2e}  {'PASS' if ok else 'FAIL'}")
    return all_pass


def test_imgp_style_operator():
    """End-to-end: SLQ log-det of an IMGP-style precision operator
    A(κ) = (2ν/κ² I + L_sym), gradient w.r.t. κ via the differentiable
    SLQ pipeline. Confirms Layer 1+2+3 compose for the flagship demo."""
    print("\n(4) IMGP-style A(κ) = (2ν/κ² I + L_sym), gradient w.r.t. κ")
    print("-" * 60)
    all_pass = True
    n = 80
    nu = 2
    L = build_spd(n, seed=13) * 0.1   # rescale so eigenvalues are moderate

    def matvec(x, kappa):
        return (2.0 * nu / kappa.square()) * x + L @ x

    for kappa_val in [0.5, 1.0, 2.0]:
        kappa = torch.tensor(kappa_val, dtype=torch.float64, requires_grad=True)

        # Use moderate m so the no-reortho Lanczos stays orthogonal.
        # (Full-reortho adjoint is Week 2 follow-up.)
        ld = slq_logdet(matvec, n, (kappa,),
                        m_probes=100, lanczos_m=20, seed=42,
                        dtype=torch.float64, device="cpu")
        ld.backward()
        grad_kappa_slq = float(kappa.grad)

        # Exact: A(κ) = (2ν/κ² I + L). d(log det A)/dκ = tr(A^-1 · dA/dκ).
        # dA/dκ = -4ν/κ³ · I, so grad = -4ν/κ³ · tr(A^-1).
        A_dense = (2.0 * nu / kappa_val ** 2) * torch.eye(n, dtype=torch.float64) + L
        A_inv_tr = float(torch.linalg.inv(A_dense).trace())
        grad_kappa_exact = -4.0 * nu / kappa_val ** 3 * A_inv_tr

        rel = abs(grad_kappa_slq - grad_kappa_exact) / abs(grad_kappa_exact)
        ok = rel < 0.1
        all_pass &= ok
        print(f"  κ={kappa_val:.2f}:  SLQ grad={grad_kappa_slq:>10.4f}   "
              f"exact={grad_kappa_exact:>10.4f}   rel_err={rel:.2e}  "
              f"{'PASS' if ok else 'FAIL'}")
    return all_pass


def main():
    p1 = test_op_vs_ground_truth()
    p2 = test_slq_accuracy()
    p3 = test_slq_gradient()
    p4 = test_imgp_style_operator()
    print("\n" + "=" * 60)
    print(f"(1) Op-vs-truth:     {'PASS' if p1 else 'FAIL'}")
    print(f"(2) SLQ accuracy:    {'PASS' if p2 else 'FAIL'}")
    print(f"(3) SLQ gradient:    {'PASS' if p3 else 'FAIL'}")
    print(f"(4) IMGP-style A(κ): {'PASS' if p4 else 'FAIL'}")
    print("=" * 60)
    sys.exit(0 if (p1 and p2 and p3 and p4) else 1)


if __name__ == "__main__":
    main()
