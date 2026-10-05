"""Amendment 1 (SCOPE-v2.md): cross-position Hessian block instead of the joint eigendecomposition.

H_AB = d^2 L / d delta_A d delta_B (delta_A on the speaker tokens, delta_B broadcast over the question
tokens), averaged over honest (0,0) prompts. A cross-position gate c (r_A . h[spk]) (r_B . h[q])
contributes c r_A r_B^T to H_AB, while within-slot curvature (H_AA, H_BB) cannot enter it at all.
Read it with an SVD: left vectors (speaker space) vs r_A, right vectors (question space) vs r_B.
First-order competitor: SVD of the cross-slot gradient outer product E[g_A g_B^T] (and centred).

Also: finite-difference check of the 2-slot HVP (never done for slots.py before).

    python crossblock.py [model_dir] [layers...]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.sparse.linalg import LinearOperator, svds

from common import OUT, Ctx, gpu_mem, load_model, log, patch_deltanet
from gate_test2 import ALPHA_B, set_layer, slot_positions
from hl2 import span_recovery, windows
from organism import MODEL_DIR, make
from slots import SlotData

K = 5
N_H = 32


def fd_check(m, D: SlotData, d, eps=1e-2, seed=0):
    """Rel. error of the exact 2-slot HVP vs central finite differences of the exact gradient."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    v = torch.randn(2, d, device="cuda", generator=g)
    v = v / v.norm()
    hv = D.hmat(m, v[None])[0]
    n = len(D.ctxs)
    gp = D.grads(m, shift=(eps * v)[None].expand(n, -1, -1).contiguous()).mean(0)
    gm = D.grads(m, shift=(-eps * v)[None].expand(n, -1, -1).contiguous()).mean(0)
    fd = (gp - gm) / (2 * eps)
    return float((hv - fd).norm() / fd.norm()), float(hv.norm())


def cross_svd(m, D: SlotData, d, Sw=None, k=K, seed=0):
    """Top-k singular triplets of the (optionally whitened) cross block H_AB."""
    def hab(vB):          # v_B [d, j] -> [H (0, v_B)]_A  [d, j]
        V = torch.as_tensor(np.ascontiguousarray(np.asarray(vB).T), dtype=torch.float32, device="cuda")
        Z = torch.zeros(V.shape[0], 2, d, device="cuda"); Z[:, 1] = V
        return D.hmat(m, Z, Sw)[:, 0].double().cpu().numpy().T

    def hba(uA):          # u_A -> [H (u_A, 0)]_B  (= H_AB^T u_A, H symmetric)
        U = torch.as_tensor(np.ascontiguousarray(np.asarray(uA).T), dtype=torch.float32, device="cuda")
        Z = torch.zeros(U.shape[0], 2, d, device="cuda"); Z[:, 0] = U
        return D.hmat(m, Z, Sw)[:, 1].double().cpu().numpy().T

    op = LinearOperator((d, d), matvec=lambda x: hab(np.asarray(x).reshape(d, 1))[:, 0],
                        rmatvec=lambda x: hba(np.asarray(x).reshape(d, 1))[:, 0],
                        matmat=hab, rmatmat=hba, dtype=np.float64)
    U, s, Vt = svds(op, k=k, tol=1e-3, v0=np.random.default_rng(seed).standard_normal(d))
    o = np.argsort(-s)
    return U[:, o].T, s[o], Vt[o]


def main(model_dir=str(MODEL_DIR), layers=(8, 6, 4), n_per_cell=64):
    t0 = time.time()
    m = load_model(model_dir); patch_deltanet(m)
    d = m.d
    items = make(4 * n_per_cell, "train", seed=7)
    ans = {a: m.tok(" " + a, add_special_tokens=False)["input_ids"][0] for a in ("Yes", "No")}
    ctxs, spos = [], []
    for x in items:
        ids, sp = slot_positions(m.tok, x)
        c = Ctx(m, torch.tensor(ids), layers[0])
        c.h_all = {kk: v for kk, v in c.h_all.items() if kk in set(layers)}
        ctxs.append(c); spos.append(sp)
    W = torch.stack([m.W_U[ans["No" if x["truth"] == "Yes" else "Yes"]].float() - m.W_U[ans[x["truth"]]].float()
                     for x in items]).contiguous()
    A = torch.tensor([x["A"] for x in items]); B = torch.tensor([x["B"] for x in items])
    cell = {(a, b): torch.nonzero((A == a) & (B == b)).flatten().tolist() for a in (0, 1) for b in (0, 1)}
    i00, i01, i10 = cell[(0, 0)], cell[(0, 1)], cell[(1, 0)]
    sub = lambda ix: SlotData([ctxs[j] for j in ix], W[ix], [spos[j] for j in ix], 2)
    res = {"model": model_dir, "alpha_B": ALPHA_B, "layers": {}}
    tag = "crossblock_" + Path(model_dir).name

    def save():
        with open(OUT / f"{tag}.json", "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1)

    for l in layers:
        t1 = time.time()
        set_layer(ctxs, l)
        D_all = SlotData(ctxs, W, spos, 2)
        g2 = D_all.grads(m).double()                                   # [N, 2, d]
        rA = g2[i01, 0].mean(0).cpu().numpy(); rB = g2[i10, 1].mean(0).cpu().numpy()
        Hs = torch.stack([torch.stack([c.h[0, sp[0]].mean(0), c.h[0, sp[1]].mean(0)]) for c, sp in zip(ctxs, spos)])
        dA = (Hs[A == 1, 0].mean(0) - Hs[A == 0, 0].mean(0)).double().cpu().numpy()
        dB = (Hs[B == 1, 1].mean(0) - Hs[B == 0, 1].mean(0)).double().cpu().numpy()
        X = torch.cat([Ctx(m, x, l).h[0, 16:] for x in windows(m.tok, 32, T=64, seed=3)], 0).double()
        Sig = torch.cov(X.T); Sig = 0.95 * Sig + 0.05 * Sig.trace() / d * torch.eye(d, dtype=Sig.dtype, device=Sig.device)
        e, Ue = torch.linalg.eigh(Sig)
        Sw = (Ue * e.sqrt()) @ Ue.T; Swn = Sw.cpu().numpy(); Swin = ((Ue / e.sqrt()) @ Ue.T).cpu().numpy()
        D00 = sub(i00[:N_H])
        row = {}
        rel, hn = fd_check(m, sub(i00[:4]), d)
        row["fd_rel_err"] = rel
        log(f"[cross L{l}] 2-slot HVP vs finite differences: rel err {rel:.2e} (|Hv| {hn:.3f})")

        def score(name, Ul, s, Vr, white=False):
            tA, tB = (rA @ Swn, rB @ Swn) if white else (rA, rB)
            dA_, dB_ = (dA @ Swin, dB @ Swin) if white else (dA, dB)
            r = {"sv": [float(x) for x in s[:K]]}
            for kk in (1, 3, 5):
                r[f"rA_left_top{kk}"] = span_recovery(Ul[:kk], tA[None])
                r[f"rB_right_top{kk}"] = span_recovery(Vr[:kk], tB[None])
                r[f"dA_left_top{kk}"] = span_recovery(Ul[:kk], dA_[None])
                r[f"dB_right_top{kk}"] = span_recovery(Vr[:kk], dB_[None])
            row[name] = r; res["layers"][str(l)] = row; save()
            log(f"[cross L{l}] {name:16s} r_A(left) top1/3/5 {r['rA_left_top1']:.2f}/{r['rA_left_top3']:.2f}/{r['rA_left_top5']:.2f} | "
                f"r_B(right) {r['rB_right_top1']:.2f}/{r['rB_right_top3']:.2f}/{r['rB_right_top5']:.2f} | "
                f"d_A {r['dA_left_top3']:.2f} d_B {r['dB_right_top3']:.2f} (top3) | s {np.round(s[:3], 4).tolist()}")

        Ul, s, Vr = cross_svd(m, D00, d); score("H_AB_raw_00", Ul, s, Vr)
        Ul, s, Vr = cross_svd(m, D00, d, Sw=Sw.float()); score("H_AB_white_00", Ul, s, Vr, white=True)
        mix = [j for ix in (i00, i01, i10, cell[(1, 1)]) for j in ix[:N_H // 4]]
        Ul, s, Vr = cross_svd(m, sub(mix), d); score("H_AB_raw_all", Ul, s, Vr)
        gA, gB = g2[i00, 0], g2[i00, 1]
        for name, a_, b_ in (("crossAGOP_00", gA, gB), ("crossCov_00", gA - gA.mean(0), gB - gB.mean(0))):
            C = (a_.T @ b_ / len(a_)).cpu().numpy()
            Uc, sc, Vct = np.linalg.svd(C)
            score(name, Uc[:, :K].T, sc, Vct[:K])
        gAw, gBw = gA @ Sw, gB @ Sw
        Uc, sc, Vct = np.linalg.svd((gAw.T @ gBw / len(gAw)).cpu().numpy()); score("crossAGOP_white_00", Uc[:, :K].T, sc, Vct[:K], white=True)
        row["random_top3"] = 3 / d
        log(f"[cross L{l}] done {time.time()-t1:.0f}s (random top-3 recovery {3/d:.4f}); {gpu_mem()}")
    log(f"[cross] all done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    md = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].isdigit() else str(MODEL_DIR)
    ls = tuple(int(x) for x in sys.argv[1:] if x.isdigit()) or (8, 6, 4)
    main(md, ls)
