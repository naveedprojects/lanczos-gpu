"""
PyTorch reference implementation of the differentiable matrix-function
primitive f(A) v via Lanczos quadrature.

This module provides:

  lanczos_torch(matvec, v, m)
      Run m Lanczos steps with full reorthogonalization. Returns
      (V, alpha, beta_sub) where V is (n, m), alpha is (m,), and
      beta_sub is (m-1,) — the subdiagonal of the tridiag T_m.

  funm_apply_unrolled(A, v, m, func)
      Forward f(A) v via Lanczos, end-to-end PyTorch autograd through
      the unrolled Lanczos iterations. Slow at scale but mathematically
      exact in autograd. Used as gradcheck ground truth.

  funm_apply_kraemer(A, v, m, func)
      Forward f(A) v with a *custom* backward pass following the
      Krämer 2024 adjoint Lanczos system. The backward avoids unrolling
      and matches the structure that will eventually be ported to CUDA.

The two should agree to machine precision on gradients for small n.

References
----------
- Krämer et al., "Gradients of Functions of Large Matrices,"
  NeurIPS 2024. arXiv:2405.17277.
- matfree library: https://github.com/pnkraemer/matfree
"""
from __future__ import annotations

from typing import Callable

import torch


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _apply_scalar(theta: torch.Tensor, func: str) -> torch.Tensor:
    if func == "log":
        return torch.log(theta)
    if func == "exp":
        return torch.exp(theta)
    if func == "sqrt":
        return torch.sqrt(theta)
    if func == "inv":
        return 1.0 / theta
    raise ValueError(f"unknown func: {func}")


def _build_T(alpha: torch.Tensor, beta_sub: torch.Tensor) -> torch.Tensor:
    """Symmetric tridiagonal of size m×m from diagonal and subdiagonal."""
    m = alpha.shape[0]
    T = torch.diag(alpha)
    if m > 1:
        T = T + torch.diag(beta_sub, diagonal=1) + torch.diag(beta_sub, diagonal=-1)
    return T


def _small_side(V: torch.Tensor, alpha: torch.Tensor, beta_sub: torch.Tensor,
                v_norm: torch.Tensor, func: str) -> torch.Tensor:
    """
    Compute y = v_norm · V · S · diag(f(Θ)) · S^T · e_1.

    Pure PyTorch ops — autograd through eigh + f + matmul.
    """
    T = _build_T(alpha, beta_sub)
    Theta, S = torch.linalg.eigh(T)
    fTheta = _apply_scalar(Theta, func)
    # q = S · diag(fTheta) · S^T · e_1 = S · (fTheta * S[0, :])
    # S[0, :] is the first row of S (length m).
    c = S[0, :]                                 # shape (m,)
    q = S @ (fTheta * c)                        # shape (m,)
    return v_norm * (V @ q)                     # shape (n,)


# ---------------------------------------------------------------------
# Lanczos forward in pure PyTorch
# ---------------------------------------------------------------------

def lanczos_torch_noreortho(matvec: Callable[[torch.Tensor], torch.Tensor],
                            v: torch.Tensor, m: int):
    """Basic 3-term Lanczos with NO reorthogonalization (un-batched).

    Used for gradcheck consistency: this matches the recurrence assumed by
    matfree's `_tridiag_adjoint`. For numerical work prefer the full-reortho
    variant; for performance prefer the batched variant
    `lanczos_torch_noreortho_batched`.
    """
    n = v.shape[0]
    v_norm = torch.linalg.norm(v)
    v0 = v / v_norm

    Vs = [v0]
    alphas = []
    betas = []
    prev = torch.zeros_like(v0)
    beta_prev = torch.zeros((), dtype=v.dtype, device=v.device)

    for j in range(m):
        w = matvec(Vs[j])
        a = Vs[j] @ w
        r = w - a * Vs[j] - beta_prev * prev
        alphas.append(a)
        beta_next = torch.linalg.norm(r)
        betas.append(beta_next)
        x_next = r / beta_next
        prev = Vs[j]
        beta_prev = beta_next
        Vs.append(x_next)

    V_full = torch.stack(Vs, dim=1)             # (n, m+1)
    alpha = torch.stack(alphas)                 # (m,)
    betas_t = torch.stack(betas)                # (m,)
    return V_full, alpha, betas_t, v_norm


def lanczos_torch_noreortho_batched(
    matvec_batched: Callable[[torch.Tensor], torch.Tensor],
    Z: torch.Tensor, m: int,
):
    """Batched 3-term Lanczos with NO reorthogonalisation.

    Runs `B` independent Lanczos sweeps in parallel by exploiting the
    sparse-matvec's natural support for batched right-hand sides. Cuts the
    Python overhead by a factor of B vs the looped version.

    Parameters
    ----------
    matvec_batched : callable
        `X (n, B) -> A · X (n, B)`. Must broadcast / batch over the second axis.
    Z : (n, B) tensor
        Starting vectors, one per Lanczos sweep (typically Rademacher probes).
    m : int
        Lanczos depth.

    Returns
    -------
    V_full   : (n, B, m+1) tensor — basis per probe.
    alpha    : (B, m) tensor — diagonal of each probe's T_m.
    betas    : (B, m) tensor — matfree-style β (one per iteration, last entry
               being the residual outside the m-step subspace).
    Z_norm   : (B,) tensor — column norms of Z.
    """
    n, B = Z.shape
    Z_norm = torch.linalg.vector_norm(Z, dim=0)          # (B,)
    V0 = Z / Z_norm.unsqueeze(0)                          # (n, B)

    Vs = [V0]
    alphas = []
    betas = []
    prev = torch.zeros_like(V0)
    beta_prev = torch.zeros(B, dtype=Z.dtype, device=Z.device)  # (B,)

    for j in range(m):
        W = matvec_batched(Vs[j])                         # (n, B)
        a = (Vs[j] * W).sum(dim=0)                        # (B,)
        # 3-term recurrence per probe (broadcast over n axis).
        R = W - a.unsqueeze(0) * Vs[j] - beta_prev.unsqueeze(0) * prev
        alphas.append(a)
        beta_next = torch.linalg.vector_norm(R, dim=0)    # (B,)
        betas.append(beta_next)
        X_next = R / beta_next.unsqueeze(0)
        prev = Vs[j]
        beta_prev = beta_next
        Vs.append(X_next)

    V_full = torch.stack(Vs, dim=2)                       # (n, B, m+1)
    alpha = torch.stack(alphas, dim=1)                    # (B, m)
    betas_t = torch.stack(betas, dim=1)                   # (B, m)
    return V_full, alpha, betas_t, Z_norm


def lanczos_torch(matvec: Callable[[torch.Tensor], torch.Tensor],
                  v: torch.Tensor, m: int):
    """Run m Lanczos steps with full reorthogonalization.

    Returns the matfree-convention output:

        V_full   : (n, m+1) — V[:, 0..m-1] are the m basis vectors used
                              by f(A) v; V[:, m] is the post-residual
                              ("outside") basis vector r_{m-1} / β_m.
        alpha    : (m,)    — diagonal of the m×m tridiagonal T_m.
        betas    : (m,)    — matfree-style: betas[j] = β_{j+1}. The
                              first m-1 entries are the subdiagonal of
                              T_m; betas[m-1] = β_m is the outside
                              residual norm.
        v_norm   : scalar  — ||v||.
    """
    n = v.shape[0]
    v_norm = torch.linalg.norm(v)
    v0 = v / v_norm

    Vs = [v0]
    alphas = []
    betas = []   # matfree-style: betas[j] = β_{j+1}, length m

    prev = torch.zeros_like(v0)
    beta_prev = torch.zeros((), dtype=v.dtype, device=v.device)

    for j in range(m):
        w = matvec(Vs[j])
        a = Vs[j] @ w
        r = w - a * Vs[j] - beta_prev * prev

        # Full reorthogonalization (twice — DGKS-like)
        for _ in range(2):
            Vmat = torch.stack(Vs, dim=1)
            h = Vmat.t() @ r
            r = r - Vmat @ h
            a = a + h[-1]

        alphas.append(a)
        beta_next = torch.linalg.norm(r)
        betas.append(beta_next)
        x_next = r / beta_next

        prev = Vs[j]
        beta_prev = beta_next
        Vs.append(x_next)

    V_full = torch.stack(Vs, dim=1)             # (n, m+1)
    alpha = torch.stack(alphas)                 # (m,)
    betas_t = torch.stack(betas)                # (m,)
    return V_full, alpha, betas_t, v_norm


# ---------------------------------------------------------------------
# Variant 1: unrolled forward (baseline ground truth for gradients)
# ---------------------------------------------------------------------

def funm_apply_unrolled(A: torch.Tensor, v: torch.Tensor, m: int,
                        func: str = "log",
                        reortho: bool = True) -> torch.Tensor:
    """Forward f(A) v via Lanczos, autograd through every iteration.

    Slow but mathematically clean — used as gradcheck ground truth.
    Default uses full reorthogonalization. Set `reortho=False` to compare
    against the no-reortho adjoint (matfree's `_tridiag_adjoint`).
    """
    fn = lanczos_torch if reortho else lanczos_torch_noreortho
    V_full, alpha, betas, v_norm = fn(lambda x: A @ x, v, m)
    V = V_full[:, :m]
    beta_sub = betas[:-1]
    return _small_side(V, alpha, beta_sub, v_norm, func)


# ---------------------------------------------------------------------
# Variant 2: custom Krämer adjoint (porting target for CUDA)
# ---------------------------------------------------------------------

class _FunmApplyKraemer(torch.autograd.Function):
    """
    Forward + custom Krämer adjoint backward for y = f(A) v via Lanczos.

    The backward is split into two pieces:

      (a) Small-side adjoint — eigh + f + matmul — done with a local
          autograd graph that we tape inside `forward` and reuse for
          backward. This gives g_V, g_alpha, g_beta_sub from g_y.

      (b) Lanczos-sweep adjoint — Krämer 2024 / matfree's
          `_tridiag_adjoint_step`. Takes g_V, g_alpha, g_beta_sub
          and produces g_v (input vector) and g_A_action (per-step
          rank-1 outer products that PyTorch autograd resolves to
          g_A via the original matvec function).
    """

    @staticmethod
    def forward(ctx, A: torch.Tensor, v: torch.Tensor, m: int, func: str):
        with torch.no_grad():
            # The Krämer adjoint matches the basic 3-term recurrence.
            # Use no-reortho forward here so forward and backward agree.
            V_full, alpha, betas, v_norm = lanczos_torch_noreortho(
                lambda x: A @ x, v, m
            )
            V_used = V_full[:, :m]
            beta_sub = betas[:-1]
            y = _small_side(V_used, alpha, beta_sub, v_norm, func)

        ctx.save_for_backward(A, v, V_full, alpha, betas, v_norm)
        ctx.func = func
        ctx.m = m
        return y

    @staticmethod
    def backward(ctx, g_y: torch.Tensor):
        A, v, V_full, alpha, betas, v_norm = ctx.saved_tensors
        func = ctx.func
        m = ctx.m
        n = v.shape[0]
        device = v.device
        dtype = v.dtype

        # ---- (a) Small-side adjoint via local autograd ----
        V_used = V_full[:, :m]
        beta_sub_full = betas[:-1]                  # subdiagonal of T_m
        V_  = V_used.detach().clone().requires_grad_(True)
        a_  = alpha.detach().clone().requires_grad_(True)
        bs_ = beta_sub_full.detach().clone().requires_grad_(True)
        vn_ = v_norm.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            y_local = _small_side(V_, a_, bs_, vn_, func)
            g_V_used, g_alpha, g_beta_sub, g_vnorm = torch.autograd.grad(
                outputs=y_local,
                inputs=(V_, a_, bs_, vn_),
                grad_outputs=g_y,
            )

        # Extend gradients to matfree's convention: length m+1 for V, m for β.
        # The "outside" vector V[:, m] and the outside β_m have zero upstream.
        g_V_full = torch.cat(
            [g_V_used, torch.zeros(n, 1, dtype=dtype, device=device)],
            dim=1,
        )
        g_betas = torch.cat(
            [g_beta_sub, torch.zeros(1, dtype=dtype, device=device)]
        )

        # ---- (b) Krämer adjoint Lanczos sweep ----
        # Matfree convention:
        #   xs has shape (m+1, n)
        #   alphas, betas, dalphas, dbetas have length m
        #   Loop in reverse over j = m-1 .. 0
        #   Init: xi = -dxs[m], lambda_plus = 0

        xi = -g_V_full[:, m].clone()         # (n,)
        lambda_plus = torch.zeros(n, dtype=dtype, device=device)
        grad_A = torch.zeros((n, n), dtype=dtype, device=device)

        for j in range(m - 1, -1, -1):
            x     = V_full[:, j]
            xplus = V_full[:, j + 1]
            a     = alpha[j]
            b     = betas[j]
            dx    = g_V_full[:, j]
            da    = g_alpha[j]
            db    = g_betas[j]

            xi_div = xi / b
            mu = db - (lambda_plus @ x) + (xplus @ xi_div)
            nu = da + (x @ xi_div)
            lam = -xi_div + mu * xplus + nu * x

            # ∂L/∂A from this step: forward did w_j = A · V[:, j], so
            #   ∂L/∂A_{kl} += λ_j[k] · V[:, j][l]  =  outer(λ_j, V_j).
            grad_A = grad_A + torch.outer(lam, x)

            matvec_lambda = A @ lam
            xi = (-dx - matvec_lambda + a * lam
                  + b * lambda_plus - b * nu * xplus)

            lambda_plus = lam

        # `xi` now holds the post-last-step co-state (matfree calls this
        # `lambda_1` in the outer code, but the variable is actually the
        # final xi after the j=0 iteration). matfree's grad_initvec:
        #   grad_initvec = ((xi · v_0) v_0 - xi) / ||v||
        v0 = V_full[:, 0]
        grad_v = ((xi @ v0) * v0 - xi) / v_norm

        # Add direct g_vnorm contribution through ∂||v||/∂v = v / ||v||.
        grad_v = grad_v + g_vnorm * (v / v_norm)

        return grad_A, grad_v, None, None


def funm_apply_kraemer(A: torch.Tensor, v: torch.Tensor, m: int,
                       func: str = "log") -> torch.Tensor:
    """Differentiable f(A) v with Krämer 2024 adjoint backward (dense A)."""
    return _FunmApplyKraemer.apply(A, v, m, func)


# ---------------------------------------------------------------------
# Operator-callable variant (Layer 3 substrate)
# ---------------------------------------------------------------------

class _FunmQFormKraemerOp(torch.autograd.Function):
    """
    Quadratic form  z^T f(A) z  via Lanczos quadrature with Krämer adjoint.

    Generalizes the dense-A path:
      - The operator is given as a callable matvec(x, *params) -> A @ x.
      - `params` is a tuple of `Tensor`s (any shape) — the operator's
        differentiable parameters.
      - The backward returns gradients for `z` and for each param in
        `params`. The chain rule through `matvec` is computed via
        PyTorch autograd at each adjoint step (matches matfree's
        `func.vjp` pattern in JAX).

    Used as the building block for stochastic Lanczos quadrature.
    """

    @staticmethod
    def forward(ctx, matvec, z, m, func, *params):
        with torch.no_grad():
            V_full, alpha, betas, z_norm = lanczos_torch_noreortho(
                lambda x: matvec(x, *params), z, m
            )
            # Quadratic form: z^T f(A) z = ||z||² · Σ_i (S[0,i])² · f(Θ_i)
            T = _build_T(alpha, betas[:-1])
            Theta, S = torch.linalg.eigh(T)
            fTheta = _apply_scalar(Theta, func)
            c = S[0, :]
            q_val = z_norm * z_norm * (c * c * fTheta).sum()

        ctx.save_for_backward(z, V_full, alpha, betas, z_norm, *params)
        ctx.func = func
        ctx.m = m
        ctx.matvec = matvec
        ctx.n_params = len(params)
        return q_val

    @staticmethod
    def backward(ctx, g_q):
        z = ctx.saved_tensors[0]
        V_full = ctx.saved_tensors[1]
        alpha = ctx.saved_tensors[2]
        betas = ctx.saved_tensors[3]
        z_norm = ctx.saved_tensors[4]
        params = ctx.saved_tensors[5:]
        func = ctx.func
        m = ctx.m
        matvec = ctx.matvec

        n = z.shape[0]
        dtype = z.dtype
        device = z.device

        # ---- (a) Small-side adjoint ----
        V_used = V_full[:, :m]
        beta_sub = betas[:-1]
        V_ = V_used.detach().clone().requires_grad_(True)
        a_ = alpha.detach().clone().requires_grad_(True)
        bs_ = beta_sub.detach().clone().requires_grad_(True)
        zn_ = z_norm.detach().clone().requires_grad_(True)

        with torch.enable_grad():
            T = _build_T(a_, bs_)
            Theta, S = torch.linalg.eigh(T)
            fTheta = _apply_scalar(Theta, func)
            c = S[0, :]
            q_local = zn_ * zn_ * (c * c * fTheta).sum()
            # Upstream cotangent on q is g_q (scalar)
            g_V_used, g_alpha, g_beta_sub, g_znorm = torch.autograd.grad(
                outputs=q_local,
                inputs=(V_, a_, bs_, zn_),
                grad_outputs=g_q,
                allow_unused=True,
            )

        if g_V_used is None:
            g_V_used = torch.zeros(n, m, dtype=dtype, device=device)
        if g_alpha is None:
            g_alpha = torch.zeros(m, dtype=dtype, device=device)
        if g_beta_sub is None:
            g_beta_sub = torch.zeros(m - 1, dtype=dtype, device=device)
        if g_znorm is None:
            g_znorm = torch.zeros((), dtype=dtype, device=device)

        # Extend to matfree convention
        g_V_full = torch.cat(
            [g_V_used, torch.zeros(n, 1, dtype=dtype, device=device)], dim=1
        )
        g_betas = torch.cat(
            [g_beta_sub, torch.zeros(1, dtype=dtype, device=device)]
        )

        # ---- (b) Krämer adjoint sweep with operator VJP ----
        xi = -g_V_full[:, m].clone()
        lambda_plus = torch.zeros(n, dtype=dtype, device=device)
        grad_params = [torch.zeros_like(p) for p in params]

        # One set of differentiable param leaves shared across the m adjoint
        # iterations. Lets the per-`matvec` operator cache its κ-dependent
        # pieces once instead of rebuilding them at every step.
        params_diff = tuple(
            p.detach().clone().requires_grad_(True) for p in params
        )
        for j in range(m - 1, -1, -1):
            x_j   = V_full[:, j]
            xplus = V_full[:, j + 1]
            a     = alpha[j]
            b     = betas[j]
            dx    = g_V_full[:, j]
            da    = g_alpha[j]
            db    = g_betas[j]

            xi_div = xi / b
            mu = db - (lambda_plus @ x_j) + (xplus @ xi_div)
            nu = da + (x_j @ xi_div)
            lam = -xi_div + mu * xplus + nu * x_j


            # VJP of matvec(λ, params) w.r.t. params, with cotangent x_j.
            # In the Lanczos forward we had w_j = matvec(V[:, j], params), so
            # the adjoint contribution from this step is the gradient of
            # ∂L = (cotangent on w_j)^T · matvec(V_j, params).
            #
            # Following matfree (which evaluates the JAX vjp of
            # `g(A) = matvec(λ_j, A)` at cotangent x_j), we instead use the
            # equivalent dual-view: ∂L/∂params += vjp[ matvec(V_j, params) ]
            # at cotangent λ_j. PyTorch autograd of `matvec(V_j, params)`
            # with `grad_outputs=λ_j` returns exactly that.
            #
            # We must call matvec twice (once to get matvec_lambda used in
            # the next xi update, once to get the param-gradient).
            with torch.enable_grad():
                out_for_grad = matvec(x_j.detach(), *params_diff)
                grad_p = torch.autograd.grad(
                    outputs=out_for_grad,
                    inputs=params_diff,
                    grad_outputs=lam.detach(),
                    retain_graph=False,
                    allow_unused=True,
                )
            for k, gp in enumerate(grad_p):
                if gp is not None:
                    grad_params[k] = grad_params[k] + gp

            # matvec_lambda needed for the xi update (computed without grad)
            with torch.no_grad():
                matvec_lambda = matvec(lam, *params)

            xi = (-dx - matvec_lambda + a * lam
                  + b * lambda_plus - b * nu * xplus)
            lambda_plus = lam

        # ---- (c) Gradient on z (the input probe vector) ----
        # The matfree-style adjoint already accounts for v_0 = z/||z||.
        # The small-side `q = z_norm² · X` has a *second* factor of z_norm
        # that did not enter the Lanczos sweep at all — it came purely
        # from the analytic identity z^T V e_1 = ||z||. We attribute that
        # factor to the outer chain rule via z_norm = ||z||, giving
        # gradient g_znorm * (z / ||z||) on top of the Lanczos-path piece.
        v0 = V_full[:, 0]
        grad_z = ((xi @ v0) * v0 - xi) / z_norm
        grad_z = grad_z + g_znorm * (z / z_norm)

        # Return tuple: (None for matvec, grad_z, None for m, None for func,
        # then grads on each param).
        return (None, grad_z, None, None, *grad_params)


def funm_qform_op(matvec, z, params, m, func="log"):
    """Differentiable v^T f(A) v with operator-callable backward.

    Parameters
    ----------
    matvec : callable
        `matvec(x, *params)` returns `A @ x` where A is parameterized by
        the tensors in `params`.
    z : (n,) tensor
        Probe vector (typically random Rademacher for SLQ).
    params : tuple of tensors
        Operator parameters with `requires_grad=True` as needed.
    m : int
        Lanczos depth.
    func : str
        Scalar function: 'log', 'exp', 'sqrt', 'inv'.
    """
    return _FunmQFormKraemerOp.apply(matvec, z, m, func, *params)


# ---------------------------------------------------------------------
# Stochastic Lanczos Quadrature for log-determinant
# ---------------------------------------------------------------------

class _SLQLogdetBatched(torch.autograd.Function):
    """
    Stochastic Lanczos quadrature for log det A, with all `m_probes`
    Hutchinson probes batched into a single forward and single backward
    via `lanczos_torch_noreortho_batched`. The matvec must accept a
    batched right-hand side `(n, B)` and return `(n, B)`.

    Returns the (averaged) log-det estimate. Gradients flow back through
    the operator parameters via the same Krämer adjoint formulas as the
    un-batched version, vectorised over the probe axis.
    """

    @staticmethod
    def forward(ctx, matvec, Z, lanczos_m, *params):
        # No grad: we capture V/alpha/betas ourselves and run the adjoint manually.
        with torch.no_grad():
            def mv(X):
                return matvec(X, *params)

            V_full, alpha, betas, Z_norm = lanczos_torch_noreortho_batched(
                mv, Z, lanczos_m,
            )

            B = Z.shape[1]
            # Per-probe quadratic form, then average.
            # T_b is (m, m); diag = alpha[b], sub = betas[b, :-1].
            # We assemble all B tridiags at once via eigh on a batched tensor.
            m = lanczos_m
            T = torch.zeros(B, m, m, dtype=Z.dtype, device=Z.device)
            T[:, torch.arange(m), torch.arange(m)] = alpha
            sub = betas[:, :-1]
            T[:, torch.arange(m - 1), torch.arange(1, m)] = sub
            T[:, torch.arange(1, m), torch.arange(m - 1)] = sub
            Theta, S = torch.linalg.eigh(T)
            fTheta = torch.log(Theta)
            c = S[:, 0, :]                                  # (B, m)
            q_per_probe = (c * c * fTheta).sum(dim=1)        # (B,)
            q_per_probe = Z_norm * Z_norm * q_per_probe
            ld = q_per_probe.mean()                          # scalar

        ctx.save_for_backward(Z, V_full, alpha, betas, Z_norm, *params)
        ctx.matvec = matvec
        ctx.lanczos_m = lanczos_m
        ctx.n_params = len(params)
        return ld

    @staticmethod
    def backward(ctx, g_ld):
        Z = ctx.saved_tensors[0]
        V_full = ctx.saved_tensors[1]
        alpha = ctx.saved_tensors[2]
        betas = ctx.saved_tensors[3]
        Z_norm = ctx.saved_tensors[4]
        params = ctx.saved_tensors[5:]
        matvec = ctx.matvec
        m = ctx.lanczos_m

        n, B = Z.shape
        dtype, device = Z.dtype, Z.device

        # ---- (a) Small-side adjoint, vectorised over probes ----
        V_used = V_full[:, :, :m]
        beta_sub = betas[:, :-1]
        V_ = V_used.detach().clone().requires_grad_(True)
        a_ = alpha.detach().clone().requires_grad_(True)
        bs_ = beta_sub.detach().clone().requires_grad_(True)
        zn_ = Z_norm.detach().clone().requires_grad_(True)

        with torch.enable_grad():
            T = torch.zeros(B, m, m, dtype=dtype, device=device)
            T[:, torch.arange(m), torch.arange(m)] = a_
            T[:, torch.arange(m - 1), torch.arange(1, m)] = bs_
            T[:, torch.arange(1, m), torch.arange(m - 1)] = bs_
            Theta, S = torch.linalg.eigh(T)
            fTheta = torch.log(Theta)
            c = S[:, 0, :]
            q_per_probe = (c * c * fTheta).sum(dim=1)
            q_per_probe = zn_ * zn_ * q_per_probe
            ld_local = q_per_probe.mean()

            g_V_used, g_alpha, g_beta_sub, g_znorm = torch.autograd.grad(
                outputs=ld_local,
                inputs=(V_, a_, bs_, zn_),
                grad_outputs=g_ld,
                allow_unused=True,
            )

        if g_V_used is None:
            g_V_used = torch.zeros_like(V_used)
        if g_alpha is None:
            g_alpha = torch.zeros_like(alpha)
        if g_beta_sub is None:
            g_beta_sub = torch.zeros_like(beta_sub)
        if g_znorm is None:
            g_znorm = torch.zeros_like(Z_norm)

        # Extend to matfree convention: pad g_V along the m+1 axis.
        g_V_full = torch.cat(
            [g_V_used, torch.zeros(n, B, 1, dtype=dtype, device=device)], dim=2
        )
        g_betas = torch.cat(
            [g_beta_sub, torch.zeros(B, 1, dtype=dtype, device=device)], dim=1
        )

        # ---- (b) Krämer adjoint sweep, batched over probes ----
        # Per-probe shapes:
        #   xi (n, B), lambda_plus (n, B), lam (n, B)
        #   a, b (B,);  mu, nu (B,);  da, db (B,)
        xi = -g_V_full[:, :, m].clone()                  # (n, B)
        lambda_plus = torch.zeros(n, B, dtype=dtype, device=device)
        grad_params = [torch.zeros_like(p) for p in params]

        params_diff = tuple(
            p.detach().clone().requires_grad_(True) for p in params
        )

        for j in range(m - 1, -1, -1):
            x_j   = V_full[:, :, j]                       # (n, B)
            xplus = V_full[:, :, j + 1]                   # (n, B)
            a     = alpha[:, j]                            # (B,)
            b     = betas[:, j]                            # (B,)
            dx    = g_V_full[:, :, j]                      # (n, B)
            da    = g_alpha[:, j]                          # (B,)
            db    = g_betas[:, j]                          # (B,)

            xi_div = xi / b.unsqueeze(0)                   # (n, B)
            mu = db - (lambda_plus * x_j).sum(0) + (xplus * xi_div).sum(0)
            nu = da + (x_j * xi_div).sum(0)
            lam = (-xi_div + mu.unsqueeze(0) * xplus
                   + nu.unsqueeze(0) * x_j)

            # Param gradient via PyTorch autograd through the batched matvec.
            with torch.enable_grad():
                out_for_grad = matvec(x_j.detach(), *params_diff)
                grad_p = torch.autograd.grad(
                    outputs=out_for_grad,
                    inputs=params_diff,
                    grad_outputs=lam.detach(),
                    retain_graph=False,
                    allow_unused=True,
                )
            for k, gp in enumerate(grad_p):
                if gp is not None:
                    grad_params[k] = grad_params[k] + gp

            with torch.no_grad():
                matvec_lambda = matvec(lam, *params)

            xi = (-dx - matvec_lambda
                  + a.unsqueeze(0) * lam
                  + b.unsqueeze(0) * lambda_plus
                  - (b * nu).unsqueeze(0) * xplus)
            lambda_plus = lam

        # Return tuple matching forward inputs: (matvec, Z, lanczos_m, *params)
        # We don't differentiate through Z (probes are fixed) or matvec.
        return (None, None, None, *grad_params)


def slq_logdet(matvec, n, params, m_probes=20, lanczos_m=50, seed=0,
               dtype=torch.float64, device="cpu", batched=True):
    """
    Estimate log det A(params) via Hutchinson + Lanczos quadrature.

    log det A = tr(log A) ≈ (1/m_probes) · Σ_i z_iᵀ log(A) z_i, with
    z_i drawn from a Rademacher (±1) distribution. Each quadratic form
    is approximated by m-step Lanczos quadrature.

    Parameters
    ----------
    matvec : callable
        Two valid signatures:
        - **Batched (preferred)**: `matvec(X, *params)` where X is `(n, B)`
          and returns `(n, B)`. With `batched=True` (default) this is the
          single call to the operator, batched over all `m_probes` probes
          at once.
        - **Un-batched**: `matvec(x, *params)` for a single vector x.
          Pass `batched=False` to use this slower path.
    n : int
        Operator dimension.
    params : tuple of tensors
        Operator parameters (any subset can have `requires_grad=True`).
    m_probes : int
        Number of Hutchinson probes.
    lanczos_m : int
        Lanczos depth per probe.
    seed : int
        RNG seed for the probe vectors.
    dtype, device : tensor metadata for probe sampling.
    batched : bool
        If True, call `matvec` once with all probes batched. Cuts Python
        overhead by `m_probes` ×. Requires the matvec to handle a
        2-D right-hand side.

    Returns
    -------
    logdet : scalar tensor
        Differentiable SLQ estimate of log det A(params).
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    Z = (torch.randint(0, 2, (n, m_probes), generator=g, dtype=dtype) * 2.0 - 1.0)
    Z = Z.to(device)
    if batched:
        return _SLQLogdetBatched.apply(matvec, Z, lanczos_m, *params)

    # Serial fallback (kept for cases where matvec doesn't support 2-D rhs).
    total = torch.zeros((), dtype=dtype, device=device)
    for i in range(m_probes):
        q = funm_qform_op(matvec, Z[:, i], params, lanczos_m, "log")
        total = total + q
    return total / m_probes
