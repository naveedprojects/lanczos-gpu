#!/usr/bin/env python3
"""
60-second quickstart for the differentiable Stochastic Lanczos Quadrature.

Run with:
    /usr/bin/python3 examples/quickstart_slq.py

What this demonstrates:
  - Build a parameterised sparse operator A(θ) = θ₁·I + θ₂·L
    (any callable matvec works; we just write one inline here).
  - Estimate log det A(θ) via stochastic Lanczos quadrature.
  - Differentiate end-to-end: ∂(log det A)/∂θ₁ and ∂/∂θ₂ via the
    Krämer-adjoint backward built into `slq_logdet`.
  - Verify both gradients against the dense analytic ground truth.

The exact same pattern works for graph-Matérn GPs, neural-network
Laplace approximations, BNN posterior log-evidences, or any other
log-det that fits a parameterised symmetric PSD operator.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import torch

from gpu_eigsh.funm_torch import slq_logdet


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64

    # Build a random sparse Laplacian L (n × n). Could be your graph, your
    # kernel matrix, or any user-provided sparse SPD-ish operator.
    n = 1000
    torch.manual_seed(0)
    M = torch.randn(n, n, dtype=dtype, device=device)
    L = (M @ M.T) / n                               # dense PSD
    L_sparse = L.to_sparse()                         # for our matvec example

    # Two operator parameters with requires_grad=True. Could be a kernel
    # lengthscale, a graph bandwidth, a Laplace prior precision, anything.
    theta1 = torch.tensor(2.0, dtype=dtype, device=device, requires_grad=True)
    theta2 = torch.tensor(1.5, dtype=dtype, device=device, requires_grad=True)

    # User-defined matvec. Must accept (x, *params) where x can be (n,) or
    # (n, B) — the (n, B) form lets us batch all Hutchinson probes in one
    # matvec call. Both shapes "just work" if you write idiomatic PyTorch.
    def matvec(x, t1, t2):
        return t1 * x + t2 * torch.sparse.mm(L_sparse, x if x.dim() == 2 else x.unsqueeze(-1)).squeeze(-1) \
            if x.dim() == 1 else (t1 * x + t2 * torch.sparse.mm(L_sparse, x))

    # Stochastic Lanczos quadrature for log det A(θ₁, θ₂).
    logdet = slq_logdet(
        matvec, n, (theta1, theta2),
        m_probes=50, lanczos_m=30, seed=0,
        dtype=dtype, device=device,
    )

    # Standard PyTorch backward — gradients accumulate into theta1.grad
    # and theta2.grad via our custom Krämer-adjoint Function.
    logdet.backward()

    # Verify against the dense analytic gradient.
    # A(θ) = θ₁ I + θ₂ L, so:
    #   ∂(log det A)/∂θ₁ = tr(A⁻¹)
    #   ∂(log det A)/∂θ₂ = tr(A⁻¹ L)
    A_dense = (theta1.detach().item() * torch.eye(n, dtype=dtype, device=device)
               + theta2.detach().item() * L)
    logdet_exact = torch.linalg.slogdet(A_dense)[1]
    Ainv = torch.linalg.inv(A_dense)
    g_theta1_exact = float(Ainv.trace())
    g_theta2_exact = float((Ainv @ L).trace())

    print(f"log det A(θ):")
    print(f"  SLQ estimate  = {float(logdet):>12.4f}")
    print(f"  exact slogdet = {float(logdet_exact):>12.4f}")
    print(f"  rel error     = {abs(float(logdet) - float(logdet_exact)) / abs(float(logdet_exact)):.2e}")
    print()
    print(f"Gradients (SLQ via Krämer adjoint vs analytic):")
    print(f"  ∂/∂θ₁:   SLQ = {float(theta1.grad):>12.4f}   "
          f"exact = {g_theta1_exact:>12.4f}   "
          f"rel = {abs(float(theta1.grad) - g_theta1_exact) / abs(g_theta1_exact):.2e}")
    print(f"  ∂/∂θ₂:   SLQ = {float(theta2.grad):>12.4f}   "
          f"exact = {g_theta2_exact:>12.4f}   "
          f"rel = {abs(float(theta2.grad) - g_theta2_exact) / abs(g_theta2_exact):.2e}")
    print()
    print(f"Device: {device}.  n = {n}.  m_probes = 50, lanczos_m = 30.")
    print("This same pattern scales to N = 10⁶ on a single GPU — see "
          "benchmark/imgp_scaling_sweep.py.")


if __name__ == "__main__":
    main()
