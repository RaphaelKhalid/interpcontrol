"""Approximation ladder for the averaged H-lens at layer l.

L0  exact:   v -> mean_c H_c v (hvp.AveragedH), top-k eigenpairs by |lambda| via scipy eigsh
             (which='LM'; H is indefinite), Frobenius quantities via Hutchinson.
L2  MLP-only K-FAC: sum_{m>l} Jbar_m^T Cbar_m Jbar_m, where
             Jbar_m = mean_c mean_{q in Q} d h_mid_m[q] / d v   (broadcast perturbation, forward-AD)
             Cbar_m = mean_c sum_{q in Q} C_{m,c,q}            (closed-form SwiGLU + RMSNorm curvature,
                                                                exact per-token adjoints from one backward)
             Q = positions >= 16 (those that receive the broadcast perturbation).
L3  weight-based: as L2 but adjoint a_m = J-lens_m^T u / n (u at m = 23), per-neuron forward
             statistics E[r^2 silu''(s) p], E[r^2 silu'(s)] (r = 1/rms), no per-token norm projector,
             no norm-curvature term. L3p additionally projects out the mean radial direction.

Block structure (Qwen3_5DecoderLayer): h_mid = h + Mixer(N1(h)); h_out = h_mid + MLP(N2(h_mid)),
N2(h) = gamma * h * r, r = (mean(h^2)+eps)^{-1/2}, gamma = 1 + w.

Per-token curvature of a . MLP(N2(h)) w.r.t. h (exact; tested in tests/test_swiglu.py):
  C = DN^T C_x DN + Hess_h[ z . (h r) ],  z = gamma * grad_x(a . MLP)
  DN = r diag(gamma) (I - (r^2/d) h h^T)            (removes the radial direction of h)
  C_x = Wg^T diag(alpha) Wg + Wg^T diag(beta) Wu + Wu^T diag(beta) Wg,
        alpha = a' silu''(s) p, beta = a' silu'(s), a' = Wd^T a, s = Wg x, p = Wu x
  Hess[z.(h r)] = -(r^3/d)(z h^T + h z^T) - (r^3/d)(z.h) I + 3 (r^5/d^2)(z.h) h h^T   (norm curvature)
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.autograd.forward_ad as fwAD

from common import Ctx, M, run_from


# ------------------------------------------------------------------ closed-form curvature
def _mlp_parts(blk):
    mlp, nrm = blk.mlp, blk.post_attention_layernorm
    gamma = 1.0 + nrm.weight.float()
    return mlp.gate_proj.weight.float(), mlp.up_proj.weight.float(), mlp.down_proj.weight.float(), gamma, nrm.eps


def swiglu_curv_sum(blk, h: torch.Tensor, a: torch.Tensor, norm_curv: bool = True,
                    per_token: bool = False) -> torch.Tensor:
    """sum_q C_q (d x d) for tokens h [N,d] (mid residuals) with adjoints a [N,d].
    per_token=True returns [N,d,d] (tests only)."""
    Wg, Wu, Wd, gamma, eps = _mlp_parts(blk)
    N, d = h.shape
    r = torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)  # [N,1]
    x = gamma * h * r
    s, p = x @ Wg.T, x @ Wu.T
    ap = a @ Wd  # a' = Wd^T a  -> [N,I]
    sg = torch.sigmoid(s)
    d1 = sg * (1 + s * (1 - sg))
    d2 = sg * (1 - sg) * (2 + s * (1 - 2 * sg))
    alpha, beta = ap * d2 * p, ap * d1
    Wgt, Wut = Wg * gamma[None], Wu * gamma[None]
    r2 = r.pow(2)
    k = r2 / d
    if per_token:
        out = []
        for q in range(N):
            Mq = Wgt.T @ (alpha[q, :, None] * Wgt) + Wgt.T @ (beta[q, :, None] * Wut)
            Mq = Mq + Wut.T @ (beta[q, :, None] * Wgt)
            Pq = torch.eye(d, device=h.device) - k[q] * torch.outer(h[q], h[q])
            Cq = r2[q] * Pq.T @ Mq @ Pq
            if norm_curv:
                gx = (ap[q] * d1[q] * p[q]) @ Wg + (ap[q] * s[q] * sg[q]) @ Wu
                z = gamma * gx
                zh = z @ h[q]
                r3, r5 = r[q] ** 3, r[q] ** 5
                Cq = Cq - (r3 / d) * (torch.outer(z, h[q]) + torch.outer(h[q], z)) \
                    - (r3 / d) * zh * torch.eye(d, device=h.device) + 3 * (r5 / d**2) * zh * torch.outer(h[q], h[q])
            out.append(Cq)
        return torch.stack(out)
    wa, wb = (r2 * alpha).sum(0), (r2 * beta).sum(0)
    SM = Wgt.T @ (wa[:, None] * Wgt)
    cross = Wgt.T @ (wb[:, None] * Wut)
    SM = SM + cross + cross.T
    Mh = (alpha * s / r + beta * p / r) @ Wgt + (beta * s / r) @ Wut  # [N,d]
    A = h.T @ ((r2 * k) * Mh)
    hMh = (h * Mh).sum(-1, keepdim=True)
    C = SM - A - A.T + h.T @ ((r2 * k.pow(2) * hMh) * h)
    if norm_curv:
        gx = (ap * d1 * p) @ Wg + (ap * s * sg) @ Wu
        z = gamma * gx
        zh = (z * h).sum(-1, keepdim=True)
        r3, r5 = r.pow(3), r.pow(5)
        B = z.T @ ((r3 / d) * h)
        C = C - B - B.T - ((r3 / d) * zh).sum() * torch.eye(d, device=h.device) \
            + h.T @ ((3 * r5 / d**2 * zh) * h)
    return 0.5 * (C + C.T)


def l3_curv(blk, a_vec: torch.Tensor, stats: dict) -> torch.Tensor:
    """Weight-based curvature with constant adjoint a_vec [d] and per-neuron stats."""
    Wg, Wu, Wd, gamma, eps = _mlp_parts(blk)
    ap = Wd.T @ a_vec
    Wgt, Wut = Wg * gamma[None], Wu * gamma[None]
    wa, wb = ap * stats["r2_d2_p"], ap * stats["r2_d1"]
    cross = Wgt.T @ (wb[:, None] * Wut)
    return Wgt.T @ (wa[:, None] * Wgt) + cross + cross.T


@torch.no_grad()
def forward_stats(m: M, ctxs: list[Ctx], layers: list[int]) -> dict:
    """Per-layer per-neuron sums over Q (averaged over contexts) + mean unit mid-residual."""
    out = {}
    for c in ctxs:
        _, _, mids = run_from(m, c, c.h, c.l + 1, record_mid=layers)
        for i in layers:
            Wg, Wu, Wd, gamma, eps = _mlp_parts(m.layers[i])
            h = mids[i][0, 16:]
            r = torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)
            x = gamma * h * r
            s, p = x @ Wg.T, x @ Wu.T
            sg = torch.sigmoid(s)
            d1 = sg * (1 + s * (1 - sg))
            d2 = sg * (1 - sg) * (2 + s * (1 - 2 * sg))
            st = out.setdefault(i, {"r2_d2_p": 0, "r2_d1": 0, "hdir": 0})
            st["r2_d2_p"] = st["r2_d2_p"] + (r.pow(2) * d2 * p).sum(0) / len(ctxs)
            st["r2_d1"] = st["r2_d1"] + (r.pow(2) * d1).sum(0) / len(ctxs)
            st["hdir"] = st["hdir"] + (h / h.norm(dim=-1, keepdim=True)).mean(0) / len(ctxs)
    return out


# ------------------------------------------------------------------ averaged Jacobians
@torch.no_grad()
def averaged_jacobians(m: M, ctxs: list[Ctx], layers: list[int], B: int = 32) -> dict[int, torch.Tensor]:
    """Jbar_m [d,d]: column i = mean_c mean_{q>=16} d h_mid_m[q] / d v_i (broadcast v at valid
    source positions), by forward-mode AD with B tangents per pass."""
    d = m.d
    J = {i: torch.zeros(d, d, device="cuda") for i in layers}
    for c in ctxs:
        for s in range(0, d, B):
            nb = min(B, d - s)
            tan = torch.zeros(nb, c.T, d, device="cuda")
            idx = torch.arange(nb, device="cuda")
            tan[:, c.valid, :] = 0
            tan[idx[:, None], c.valid.nonzero()[:, 0][None, :], (s + idx)[:, None]] = 1.0
            with fwAD.dual_level():
                h = fwAD.make_dual(c.h.expand(nb, -1, -1).contiguous(), tan)
                _, _, mids = run_from(m, c, h, c.l + 1, stop=max(layers) + 1, record_mid=layers)
                for i in layers:
                    J[i][:, s:s + nb] += fwAD.unpack_dual(mids[i]).tangent[:, 16:].mean(1).T / len(ctxs)
            del tan
    return J


# ------------------------------------------------------------------ L2 per u
def l2_curvatures(m: M, ctxs: list[Ctx], u: torch.Tensor, layers: list[int], norm_curv=True):
    """Cbar_m for m in layers with exact per-token adjoints. Returns {m: [d,d]} (and the
    no-norm-curvature variant)."""
    C = {i: torch.zeros(m.d, m.d, device="cuda") for i in layers}
    Cn = {i: torch.zeros(m.d, m.d, device="cuda") for i in layers}
    for c in ctxs:
        h = c.h.clone().requires_grad_(True)
        with torch.enable_grad():
            out, rec, mids = run_from(m, c, h, c.l + 1, record=layers, record_mid=layers)
            F = (out[:, c.valid] @ u).sum() / c.n_valid
            outs = [rec[i] for i in layers if i != m.n_layers - 1]
            grads = torch.autograd.grad(F, outs) if outs else []
        adj = dict(zip([i for i in layers if i != m.n_layers - 1], grads))
        last = m.n_layers - 1
        if last in layers:
            a = torch.zeros(1, c.T, m.d, device="cuda")
            a[:, c.valid] = u / c.n_valid
            adj[last] = a
        for i in layers:
            hq = mids[i][0, 16:].detach()
            aq = adj[i][0, 16:].detach()
            C[i] += swiglu_curv_sum(m.layers[i], hq, aq, norm_curv=True) / len(ctxs)
            Cn[i] += swiglu_curv_sum(m.layers[i], hq, aq, norm_curv=False) / len(ctxs)
        del out, rec, mids, grads
    return C, Cn


def assemble(J: dict, C: dict, P: torch.Tensor | None = None) -> torch.Tensor:
    H = 0
    for i in C:
        Ci = C[i] if P is None else P[i] @ C[i] @ P[i]
        H = H + J[i].T @ Ci @ J[i]
    return 0.5 * (H + H.T)


# ------------------------------------------------------------------ L0 eigs + metrics
def l0_eigs(Hop, d: int, k: int = 20, tol: float = 1e-3, seed: int = 0):
    from scipy.sparse.linalg import LinearOperator, eigsh

    def mv(x):
        v = torch.as_tensor(np.asarray(x).reshape(-1), dtype=torch.float32, device="cuda")
        return Hop.matvec(v).double().cpu().numpy()

    op = LinearOperator((d, d), matvec=mv, dtype=np.float64)
    rng = np.random.default_rng(seed)
    t = time.time()
    lam, V = eigsh(op, k=k, which="LM", tol=tol, v0=rng.standard_normal(d))
    order = np.argsort(-np.abs(lam))
    return lam[order], V[:, order], time.time() - t


def top_eig_dense(H: torch.Tensor, k: int):
    lam, V = torch.linalg.eigh(H.double())
    order = torch.argsort(-lam.abs())[:k]
    return lam[order].cpu().numpy(), V[:, order].cpu().numpy()


def subspace_overlap(U: np.ndarray, W: np.ndarray) -> float:
    """||U^T W||_F^2 / k for orthonormal [d,k] bases (1 = same subspace, k/d = random)."""
    k = U.shape[1]
    return float(np.linalg.norm(U.T @ W) ** 2 / k)


def hutchinson(Hop, d: int, n_probe: int = 24, seed: int = 0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    Z = (torch.randint(0, 2, (n_probe, d), generator=g) * 2 - 1).float().cuda()
    Y = Hop.matmat(Z)
    return Z, Y


def frob_metrics(Z, Y, Hk: torch.Tensor):
    """From probes Z, Y = H0 Z: ||H0||_F^2, cos_F(H0,Hk), unscaled explained share."""
    Yk = Z @ Hk.T
    n2 = (Y * Y).sum(1)
    ip = (Y * Yk).sum(1)
    H0sq, H0sq_se = n2.mean().item(), (n2.std() / math.sqrt(len(n2))).item()
    ipm, ip_se = ip.mean().item(), (ip.std() / math.sqrt(len(ip))).item()
    Hksq = (Hk * Hk).sum().item()
    cos = ipm / math.sqrt(H0sq * Hksq)
    share = (2 * ipm - Hksq) / H0sq
    return {"H0_F2": H0sq, "H0_F2_se": H0sq_se, "Hk_F2": Hksq, "cosF": cos,
            "cosF_se": ip_se / math.sqrt(H0sq * Hksq), "explained_unscaled": share,
            "explained_best_scale": cos**2}


# ------------------------------------------------------------------ L1: exact MLP-only (no K-FAC)
def swiglu_curv_apply(blk, h, a, delta):
    """Per-token C_q delta_q for tokens h, adjoints a, vectors delta (all [N,d]); same C as
    swiglu_curv_sum (incl. norm curvature)."""
    Wg, Wu, Wd, gamma, eps = _mlp_parts(blk)
    d = h.shape[-1]
    r = torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)
    k = r.pow(2) / d
    x = gamma * h * r
    s, p = x @ Wg.T, x @ Wu.T
    ap = a @ Wd
    sg = torch.sigmoid(s)
    d1 = sg * (1 + s * (1 - sg))
    d2 = sg * (1 - sg) * (2 + s * (1 - 2 * sg))
    alpha, beta = ap * d2 * p, ap * d1
    hd = (h * delta).sum(-1, keepdim=True)
    y = r * gamma * (delta - k * h * hd)
    gy, uy = y @ Wg.T, y @ Wu.T
    t = (alpha * gy + beta * uy) @ Wg + (beta * gy) @ Wu
    gt = gamma * t
    out = r * (gt - k * h * (h * gt).sum(-1, keepdim=True))
    z = gamma * ((ap * d1 * p) @ Wg + (ap * s * sg) @ Wu)
    zh = (z * h).sum(-1, keepdim=True)
    zd = (z * delta).sum(-1, keepdim=True)
    r3, r5 = r.pow(3), r.pow(5)
    out = out - (r3 / d) * (z * hd + h * zd) - (r3 / d) * zh * delta + 3 * (r5 / d**2) * zh * h * hd
    return out


class L1Op:
    """v -> mean_c sum_m J_{c,m}^T [C_{c,m,q} (J_{c,m} v)_q]: exact per-context Jacobians and
    per-token MLP curvature; token-mixer curvature excluded. One forward-AD pass and one
    forward+backward per (context, vector) batch."""

    def __init__(self, m, ctxs, u, layers, batch=4):
        self.m, self.ctxs, self.layers, self.batch = m, ctxs, layers, batch
        self.n_mv = 0
        self.mids, self.adj = [], []
        last = m.n_layers - 1
        for c in ctxs:
            h = c.h.clone().requires_grad_(True)
            with torch.enable_grad():
                out, rec, mids = run_from(m, c, h, c.l + 1, record=layers, record_mid=layers)
                F = (out[:, c.valid] @ u).sum() / c.n_valid
                outs = [rec[i] for i in layers if i != last]
                grads = torch.autograd.grad(F, outs)
            adj = dict(zip([i for i in layers if i != last], grads))
            a = torch.zeros(1, c.T, m.d, device="cuda"); a[:, c.valid] = u / c.n_valid
            adj[last] = a
            self.mids.append({i: mids[i].detach()[0] for i in layers})
            self.adj.append({i: adj[i].detach()[0] for i in layers})
            del out, rec, mids, grads

    def matmat(self, Vs):
        m = self.m
        pairs = [(i, c) for i in range(Vs.shape[0]) for c in range(len(self.ctxs))]
        out = torch.zeros_like(Vs)
        for s0 in range(0, len(pairs), self.batch):
            chunk = pairs[s0:s0 + self.batch]
            cs = [self.ctxs[c] for _, c in chunk]
            c0 = cs[0]
            H = torch.cat([c.h for c in cs])
            V = torch.stack([Vs[i] for i, _ in chunk])
            tan = c0.valid[None, :, None] * V[:, None, :]
            with torch.no_grad(), fwAD.dual_level():
                _, _, mids = run_from(m, c0, fwAD.make_dual(H, tan), c0.l + 1, record_mid=self.layers)
                dl = {i: fwAD.unpack_dual(mids[i]).tangent for i in self.layers}
            W = {}
            for i in self.layers:
                hh = torch.stack([self.mids[c][i] for _, c in chunk])
                aa = torch.stack([self.adj[c][i] for _, c in chunk])
                W[i] = swiglu_curv_apply(m.layers[i], hh.reshape(-1, m.d), aa.reshape(-1, m.d),
                                         dl[i].reshape(-1, m.d)).reshape(hh.shape)
                W[i][:, :16] = 0
            Hg = H.clone().requires_grad_(True)
            with torch.enable_grad():
                _, _, mids2 = run_from(m, c0, Hg, c0.l + 1, record_mid=self.layers)
                (g,) = torch.autograd.grad([mids2[i] for i in self.layers], Hg, grad_outputs=[W[i] for i in self.layers])
            res = g[:, c0.valid].sum(1)
            for j, (i, _) in enumerate(chunk):
                out[i] += res[j]
            self.n_mv += len(chunk)
        return out / len(self.ctxs)

    def matvec(self, v):
        return self.matmat(v[None])[0]
