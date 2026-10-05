"""H-lens v2 utilities: last-position objectives, batched exact grad/HVP, averaged operator,
eigendecomposition and gate-recovery metrics (Oct 4 2026).

Objective of context c (all of v2): f_c(delta) = final_norm(h_final[c, last](h_l + e_last (x) delta)) . w_c,
i.e. the real logit (difference) at the last position, through blocks l+1..L-1, the final RMSNorm
and the unembedding. delta perturbs ONLY the last position at layer l (no cross-position terms;
the v1 pilot used a broadcast perturbation). H_c = d^2 f_c / d delta^2 (d x d), exact via double
backward; the H-lens is mean_c H_c.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.sparse.linalg import LinearOperator, eigsh

from common import Ctx, M, run_from


def _fwd(m: M, ctxs: list[Ctx], W: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """Per-row objective values [B]; h [B,T,d] at layer l, W [B,d] readout rows."""
    out, _, _ = run_from(m, ctxs[0], h, ctxs[0].l + 1)
    x = m.final_norm(out[:, -1])
    return (x * W).sum(-1)


def batched(m: M, ctxs: list[Ctx], W: torch.Tensor, V: torch.Tensor | None = None, hvp: bool = True,
            shift: torch.Tensor | None = None):
    """ctxs: B same-length contexts; W [B,d]; V [B,d] tangents at the last position.
    shift [B,d]: evaluate at h_l[last] + shift. Returns (f [B], grad [B,d], Hv [B,d] or None)."""
    c0 = ctxs[0]
    assert all(c.T == c0.T and c.l == c0.l for c in ctxs)
    h = torch.cat([c.h for c in ctxs], 0)
    if shift is not None:
        h = h.clone(); h[:, -1] += shift
    h = h.clone().requires_grad_(True)
    with torch.enable_grad():
        f = _fwd(m, ctxs, W, h)
        (g,) = torch.autograd.grad(f.sum(), h, create_graph=hvp)
        hv = None
        if hvp:
            (hv,) = torch.autograd.grad((g[:, -1] * V).sum(), h)
            hv = hv[:, -1].detach()
    return f.detach(), g[:, -1].detach(), hv


class AvgH:
    """delta -> mean_c H_c delta (optionally whitened: S^T H S with S = Sigma^{1/2})."""

    def __init__(self, m, ctxs, W, batch=4, S: torch.Tensor | None = None, shifts=None):
        self.m, self.ctxs, self.W, self.batch, self.S = m, ctxs, W, batch, S
        self.shifts = shifts
        self.n = 0

    def matmat(self, Vs: torch.Tensor) -> torch.Tensor:
        if self.S is not None:
            Vs = Vs @ self.S.T
        k = Vs.shape[0]
        # chunks never mix context lengths (no padding anywhere)
        pairs = sorted(((self.ctxs[c].T, c, i) for i in range(k) for c in range(len(self.ctxs))))
        chunks, cur = [], []
        for T, c, i in pairs:
            if cur and (len(cur) == self.batch or self.ctxs[cur[0][1]].T != T):
                chunks.append(cur); cur = []
            cur.append((i, c))
        if cur:
            chunks.append(cur)
        out = torch.zeros_like(Vs)
        for ch in chunks:
            sh = None if self.shifts is None else torch.stack([self.shifts[c] for _, c in ch])
            _, _, hv = batched(self.m, [self.ctxs[c] for _, c in ch], torch.stack([self.W[c] for _, c in ch]),
                               torch.stack([Vs[i] for i, _ in ch]), shift=sh)
            for j, (i, _) in enumerate(ch):
                out[i] += hv[j]
            self.n += len(ch)
        out = out / len(self.ctxs)
        if self.S is not None:
            out = out @ self.S
        return out


def top_eigs(op: AvgH, d: int, k: int = 10, tol: float = 1e-2, seed: int = 0):
    """Top-k eigenpairs by |lambda| of the symmetric averaged operator. Returns (lam [k], V [k,d]) fp64."""
    def mm(X):  # X [d, k] -> [d, k]
        Xt = torch.as_tensor(np.ascontiguousarray(np.asarray(X).T), dtype=torch.float32, device="cuda")
        return op.matmat(Xt).double().cpu().numpy().T

    def mv(x):
        return mm(np.asarray(x).reshape(d, 1))[:, 0]
    L = LinearOperator((d, d), matvec=mv, matmat=mm, dtype=np.float64)
    v0 = np.random.default_rng(seed).standard_normal(d)
    lam, V = eigsh(L, k=k, which="LM", tol=tol, v0=v0)
    o = np.argsort(-np.abs(lam))
    return lam[o], V[:, o].T


def _groups(ctxs, batch):
    """Index chunks of equal context length (order-preserving scatter by the caller)."""
    by = {}
    for j, c in enumerate(ctxs):
        by.setdefault(c.T, []).append(j)
    for idx in by.values():
        for s in range(0, len(idx), batch):
            yield idx[s:s + batch]


def all_grads(m, ctxs, W, batch=8, shifts=None) -> torch.Tensor:
    out = torch.zeros(len(ctxs), W.shape[1], device=W.device)
    for ix in _groups(ctxs, batch):
        sh = None if shifts is None else shifts[ix]
        _, g, _ = batched(m, [ctxs[j] for j in ix], W[ix], hvp=False, shift=sh)
        out[ix] = g
    return out


def values(m, ctxs, W, shifts=None, batch=8) -> torch.Tensor:
    out = torch.zeros(len(ctxs), device=W.device)
    for ix in _groups(ctxs, batch):
        cs = [ctxs[j] for j in ix]
        h = torch.cat([c.h for c in cs], 0).clone()
        if shifts is not None:
            h[:, -1] += shifts[ix]
        with torch.no_grad():
            out[ix] = _fwd(m, cs, W[ix], h)
    return out


def windows(tok, n: int, T: int = 64, seed: int = 0, per_article: int = 4) -> list[torch.Tensor]:
    """n windows of exactly T tokens from wikitext-103 validation, up to `per_article`
    non-overlapping windows per article (the validation split has only ~60 articles)."""
    import glob
    import os
    import random

    import pyarrow.parquet as pq

    fs = glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/datasets--Salesforce--wikitext/snapshots/*/wikitext-103-raw-v1/validation-*.parquet"))
    lines = pq.read_table(fs[0]).column("text").to_pylist()
    arts, cur = [], []
    for ln in lines:
        if ln.startswith(" = ") and not ln.startswith(" = = ") and cur:
            arts.append("".join(cur)); cur = []
        cur.append(ln)
    if cur:
        arts.append("".join(cur))
    rng = random.Random(seed)
    rng.shuffle(arts)
    out = []
    for k in range(per_article):
        for a in arts:
            ids = tok(a, add_special_tokens=False)["input_ids"]
            s = 100 + k * (T + 50)
            if len(ids) >= s + T:
                out.append(torch.tensor(ids[s:s + T]))
                if len(out) == n:
                    return out
    return out


# ------------------------------------------------------------------ metrics

def orth(X: np.ndarray) -> np.ndarray:
    """Rows of X -> orthonormal basis rows of their span."""
    q, _ = np.linalg.qr(np.asarray(X, dtype=np.float64).T)
    return q.T


def span_recovery(V: np.ndarray, targets: np.ndarray) -> float:
    """Fraction of span(targets) captured by span(V rows): ||P_V Q||_F^2 / rank(Q). Random: k/d."""
    Q = orth(targets); B = orth(V)
    return float(np.sum((Q @ B.T) ** 2) / Q.shape[0])


def coupling(lam: np.ndarray, V: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    """x^T S y for the rank-k reconstruction S = V^T diag(lam) V (unit x, y)."""
    x = x / np.linalg.norm(x); y = y / np.linalg.norm(y)
    return float((V @ x) @ (lam * (V @ y)))


def pairing_correct(lam, V, dirs: list[np.ndarray]) -> tuple[bool, list[float]]:
    """dirs = [a, b, c, e]; truth pairs (a,b), (c,e). Correct if that pairing has the largest
    summed |coupling| among the three perfect matchings. Chance = 1/3."""
    a, b, c, e = dirs
    sc = [abs(coupling(lam, V, a, b)) + abs(coupling(lam, V, c, e)),
          abs(coupling(lam, V, a, c)) + abs(coupling(lam, V, b, e)),
          abs(coupling(lam, V, a, e)) + abs(coupling(lam, V, b, c))]
    return bool(np.argmax(sc) == 0), sc


def sym_eig_top(C: np.ndarray, k: int):
    lam, U = np.linalg.eigh(C)
    o = np.argsort(-np.abs(lam))[:k]
    return lam[o], U[:, o].T
