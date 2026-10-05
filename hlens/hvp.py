"""Exact Hessian-vector products of the J-lens objective w.r.t. the layer-l residual.

Objective (matches the jlens fitting estimator, see common.py):
    G_c(v) = (1/n) * sum_{q valid} u . h_final[q]( h_l + 1_valid (x) v )
i.e. the scalar whose gradient at v=0 is exactly u^T J_c (the per-prompt J-lens row:
one-hot cotangent at every valid target position, mean over valid source positions).
The H-lens of context c is H_c = d^2 G_c / dv^2 at v=0: a broadcast perturbation v at
every valid source position, summed over target positions, divided by n_valid. It
includes cross-position terms (sum over p, p' of d^2/dh[p]dh[p']).

HVPs are reverse-over-reverse (double backward with create_graph=True); this is the same
exact quantity as forward-over-reverse jvp(grad). Batching: B (context, tangent) pairs
share one forward if all contexts have the same length T (no padding is needed).
"""

from __future__ import annotations

import torch

from common import Ctx, M, run_from


def _stack_h(ctxs: list[Ctx]) -> torch.Tensor:
    return torch.cat([c.h for c in ctxs], 0)


def G_value(m: M, ctxs: list[Ctx], u: torch.Tensor, V: torch.Tensor | None = None) -> torch.Tensor:
    """G for each batch row at v = V[b] (or 0). Returns [B]."""
    c0 = ctxs[0]
    h = _stack_h(ctxs)
    if V is not None:
        h = h + c0.valid[None, :, None] * V[:, None, :]
    with torch.no_grad():
        out, _, _ = run_from(m, c0, h, c0.l + 1)
        return (out[:, c0.valid] @ u).sum(1) / c0.n_valid


def grad_and_hvp(m: M, ctxs: list[Ctx], u: torch.Tensor, V: torch.Tensor, need_grad=True):
    """For batch rows b (context ctxs[b], tangent V[b]): returns (grad_b [B,d], (H_b V_b) [B,d])."""
    c0 = ctxs[0]
    assert all(c.T == c0.T and c.l == c0.l for c in ctxs)
    h = _stack_h(ctxs).clone().requires_grad_(True)
    with torch.enable_grad():
        out, _, _ = run_from(m, c0, h, c0.l + 1)
        F = (out[:, c0.valid] @ u).sum() / c0.n_valid
        (g,) = torch.autograd.grad(F, h, create_graph=True)
        gv = (g[:, c0.valid] * V[:, None, :]).sum()
        (hv,) = torch.autograd.grad(gv, h)
    grad = g[:, c0.valid].sum(1).detach() if need_grad else None
    return grad, hv[:, c0.valid].sum(1)


def grad_only(m: M, ctxs: list[Ctx], u: torch.Tensor, V: torch.Tensor | None = None) -> torch.Tensor:
    c0 = ctxs[0]
    h = _stack_h(ctxs)
    if V is not None:
        h = h + c0.valid[None, :, None] * V[:, None, :]
    h = h.clone().requires_grad_(True)
    with torch.enable_grad():
        out, _, _ = run_from(m, c0, h, c0.l + 1)
        F = (out[:, c0.valid] @ u).sum() / c0.n_valid
        (g,) = torch.autograd.grad(F, h)
    return g[:, c0.valid].sum(1)


class AveragedH:
    """v -> mean_c H_c v over a list of same-length contexts, in chunks of `batch`."""

    def __init__(self, m: M, ctxs: list[Ctx], u: torch.Tensor, batch: int = 4):
        self.m, self.ctxs, self.u, self.batch = m, ctxs, u, batch
        self.n_hvp = 0

    def matmat(self, Vs: torch.Tensor) -> torch.Tensor:
        """Vs [k, d] -> [k, d] = mean_c H_c Vs[i]."""
        k = Vs.shape[0]
        pairs = [(i, c) for i in range(k) for c in range(len(self.ctxs))]
        out = torch.zeros_like(Vs)
        for s in range(0, len(pairs), self.batch):
            chunk = pairs[s:s + self.batch]
            ctxs = [self.ctxs[c] for _, c in chunk]
            V = torch.stack([Vs[i] for i, _ in chunk])
            _, hv = grad_and_hvp(self.m, ctxs, self.u, V, need_grad=False)
            for j, (i, _) in enumerate(chunk):
                out[i] += hv[j]
            self.n_hvp += len(chunk)
        return out / len(self.ctxs)

    def matvec(self, v: torch.Tensor) -> torch.Tensor:
        return self.matmat(v[None])[0]


# ---------------------------------------------------------------- general scalar fn

def grad_hvp_fn(fn, h0: torch.Tensor, V: torch.Tensor):
    """Generic exact gradient and HVP of a scalar fn(h) at h0 (any shape) along V."""
    h = h0.clone().requires_grad_(True)
    with torch.enable_grad():
        y = fn(h)
        (g,) = torch.autograd.grad(y, h, create_graph=True)
        (hv,) = torch.autograd.grad((g * V).sum(), h)
    return y.detach(), g.detach(), hv
