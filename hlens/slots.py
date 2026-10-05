"""Slot perturbations for cross-position H-lens tests (H-lens v2, Oct 4 2026).

A slot is a set of token positions; the perturbation delta_s (d-dim) is added to every position
of slot s at layer l. With S slots the perturbation lives in R^{S*d}:
    f_c(delta) = final_norm(h_final[c, last](h_l + sum_s M_s (x) delta_s)) . w_c
S = 1, slot = all positions        -> broadcast H-lens (fully unsupervised)
S = 2, slots = (speaker, '?')      -> joint 2d x 2d curvature; a cross-position gate
                                       (r_A . h[spk]) (r_B . h[?]) shows up as eigvecs (r_A, +-r_B).
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.sparse.linalg import LinearOperator, eigsh

from common import Ctx, M, run_from
from hl2 import _fwd


def masks_tensor(ctxs, slot_pos, S):
    """slot_pos[c] = list of S position lists -> [B, S, T] float mask."""
    T = ctxs[0].T
    Mk = torch.zeros(len(ctxs), S, T, device="cuda")
    for b, sp in enumerate(slot_pos):
        for s, ps in enumerate(sp):
            Mk[b, s, ps] = 1.0
    return Mk


def batched_slots(m: M, ctxs: list[Ctx], W, Mk, V=None, hvp=True, shift=None):
    """Mk [B,S,T]; V, shift [B,S,d]. Returns f [B], grad [B,S,d], Hv [B,S,d]."""
    h = torch.cat([c.h for c in ctxs], 0)
    if shift is not None:
        h = h + torch.einsum("bst,bsd->btd", Mk, shift)
    h = h.clone().requires_grad_(True)
    with torch.enable_grad():
        f = _fwd(m, ctxs, W, h)
        (g,) = torch.autograd.grad(f.sum(), h, create_graph=hvp)
        gs = torch.einsum("bst,btd->bsd", Mk, g)
        hv = None
        if hvp:
            (hv,) = torch.autograd.grad((gs * V).sum(), h)
            hv = torch.einsum("bst,btd->bsd", Mk, hv).detach()
    return f.detach(), gs.detach(), hv


class SlotData:
    """Contexts grouped by length with their slot masks (no padding anywhere)."""

    def __init__(self, ctxs, W, slot_pos, S, batch=8):
        self.ctxs, self.W, self.S, self.batch = ctxs, W, S, batch
        self.groups = []
        by = {}
        for j, c in enumerate(ctxs):
            by.setdefault(c.T, []).append(j)
        for idx in by.values():
            for s in range(0, len(idx), batch):
                ix = idx[s:s + batch]
                cs = [ctxs[j] for j in ix]
                self.groups.append((ix, cs, W[ix], masks_tensor(cs, [slot_pos[j] for j in ix], S)))

    def grads(self, m, shift=None):
        d = self.W.shape[1]
        out = torch.zeros(len(self.ctxs), self.S, d, device="cuda")
        for ix, cs, w, mk in self.groups:
            sh = None if shift is None else shift[ix]
            _, g, _ = batched_slots(m, cs, w, mk, hvp=False, shift=sh)
            out[ix] = g
        return out

    def values(self, m, shift=None):
        out = torch.zeros(len(self.ctxs), device="cuda")
        for ix, cs, w, mk in self.groups:
            h = torch.cat([c.h for c in cs], 0)
            if shift is not None:
                h = h + torch.einsum("bst,bsd->btd", mk, shift[ix])
            with torch.no_grad():
                out[ix] = _fwd(m, cs, w, h)
        return out

    def hmat(self, m, Vs, Sw=None):
        """Vs [k, S, d] -> mean_c H_c Vs[i]  ([k, S, d]); Sw: optional whitening (applied per slot)."""
        if Sw is not None:
            Vs = Vs @ Sw.T
        k = Vs.shape[0]
        out = torch.zeros_like(Vs)
        for ix, cs, w, mk in self.groups:
            B = len(cs)
            for i in range(k):
                _, _, hv = batched_slots(m, cs, w, mk, Vs[i][None].expand(B, -1, -1).contiguous())
                out[i] += hv.sum(0)
        out = out / len(self.ctxs)
        if Sw is not None:
            out = out @ Sw
        return out


def top_eigs_slots(m, data: SlotData, k=10, tol=1e-2, Sw=None, seed=0):
    S, d = data.S, data.W.shape[1]
    n = S * d

    def mm(X):
        Xt = torch.as_tensor(np.ascontiguousarray(np.asarray(X).T), dtype=torch.float32, device="cuda").reshape(-1, S, d)
        return data.hmat(m, Xt, Sw).reshape(-1, n).double().cpu().numpy().T

    def mv(x):
        return mm(np.asarray(x).reshape(n, 1))[:, 0]
    L = LinearOperator((n, n), matvec=mv, matmat=mm, dtype=np.float64)
    lam, V = eigsh(L, k=k, which="LM", tol=tol, v0=np.random.default_rng(seed).standard_normal(n))
    o = np.argsort(-np.abs(lam))
    return lam[o], V[:, o].T
