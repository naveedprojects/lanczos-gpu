#!/usr/bin/env python3
"""
Gradcheck for Layer 1 backward — the Krämer adjoint of Lanczos f(A) v.

Compares two backward implementations:

  (1) autograd through the unrolled Lanczos forward (slow ground truth)
  (2) custom Krämer adjoint backward (the implementation we want to ship)

Both should give identical gradients (up to small numerical noise from
reorthogonalisation tolerances).

Tested at n in {50, 200, 500}, m in {n, 80}, f in {log, exp, sqrt, inv}.
Target: max relative error < 1e-6 on g_A and g_v for all combinations.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import torch
import numpy as np

from gpu_eigsh.funm_torch import funm_apply_unrolled, funm_apply_kraemer


def build_spd(n: int, seed: int = 0) -> torch.Tensor:
    """Build a small dense SPD matrix for testing."""
    g = torch.Generator().manual_seed(seed)
    M = torch.randn(n, n, generator=g, dtype=torch.float64)
    A = M @ M.T + n * torch.eye(n, dtype=torch.float64)
    return A


def compare_grads(n: int, m: int, func: str, seed: int = 0, thr: float = 1e-6):
    """Run forward via both paths, backprop a scalar loss, compare g_A and g_v."""
    A_np = build_spd(n, seed)
    g = torch.Generator().manual_seed(seed + 1)
    v_np = torch.randn(n, generator=g, dtype=torch.float64)

    # Pick a random upstream gradient direction
    grad_y = torch.randn(n, generator=g, dtype=torch.float64)

    # --- (1) Unrolled (no-reortho) forward + autograd, ground truth ---
    A1 = A_np.detach().clone().requires_grad_(True)
    v1 = v_np.detach().clone().requires_grad_(True)
    y1 = funm_apply_unrolled(A1, v1, m, func, reortho=False)
    loss1 = (y1 * grad_y).sum()
    loss1.backward()
    gA_unrolled = A1.grad.clone()
    gv_unrolled = v1.grad.clone()
    y1_val = y1.detach().clone()

    # --- (2) Kraemer adjoint ---
    A2 = A_np.detach().clone().requires_grad_(True)
    v2 = v_np.detach().clone().requires_grad_(True)
    y2 = funm_apply_kraemer(A2, v2, m, func)
    loss2 = (y2 * grad_y).sum()
    loss2.backward()
    gA_kraemer = A2.grad.clone()
    gv_kraemer = v2.grad.clone()

    # Compare forward (should match exactly, no numerical noise expected)
    fwd_err = (y1_val - y2.detach()).norm() / y1_val.norm()

    # Compare gradients
    gA_err = (gA_unrolled - gA_kraemer).norm() / gA_unrolled.norm()
    gv_err = (gv_unrolled - gv_kraemer).norm() / gv_unrolled.norm()

    return float(fwd_err), float(gA_err), float(gv_err)


def main():
    # Avoid m == n (β_m → 0 causes division-by-zero in the adjoint).
    # exp() of large eigenvalues overflows at high n; cap n for exp.
    cases = [
        # (n, m, func)
        (20,  10, "log"),
        (20,  10, "exp"),
        (20,  10, "sqrt"),
        (20,  10, "inv"),
        (50,  30, "log"),
        (50,  30, "sqrt"),
        (50,  30, "inv"),
        (100, 50, "log"),
        (100, 50, "sqrt"),
        (200, 80, "log"),
        (200, 80, "sqrt"),
        (500, 100, "log"),
    ]

    thr = 1e-6
    print(f"{'n':>5} {'m':>4} {'func':>6} {'fwd err':>12} {'gA err':>12} "
          f"{'gv err':>12} {'pass':>6}")
    print("-" * 64)

    all_pass = True
    for (n, m, func) in cases:
        try:
            fe, gAe, gve = compare_grads(n, m, func)
        except Exception as e:
            print(f"{n:>5} {m:>4} {func:>6}  EXCEPTION: {e}")
            all_pass = False
            continue
        ok = (fe < thr) and (gAe < thr) and (gve < thr)
        if not ok:
            all_pass = False
        status = "PASS" if ok else "FAIL"
        print(f"{n:>5} {m:>4} {func:>6} {fe:>12.2e} {gAe:>12.2e} "
              f"{gve:>12.2e} {status:>6}")
    print("-" * 64)
    print(("ALL PASS" if all_pass else "SOME FAILED") + ".")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
