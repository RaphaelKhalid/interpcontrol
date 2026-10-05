"""P2 main test, cross-position version: is the planted 'lie iff pirate AND animal' gate visible to
the H-lens from HONEST-only prompts?

B slot: all question tokens (default; HL_BSLOT=qmark uses only '?', which failed: the topic is not
carried by '?'). Why cross-position: gate_test.py showed that steering the LAST position with DoM directions does
nothing at any layer (4-22); the gate reads the speaker token and the question token through
attention. So the perturbation has two slots: delta_A on the speaker-word tokens, delta_B on the
'?' token (slots.py). Ground truth (preregistered in SCOPE-v2.md):
  - DoM d_A (speaker slot), d_B ('?' slot) at layer l; steering AND-check on (0,0) prompts.
  - functional read directions r_A = mean grad_slotA L over (0,1) prompts, r_B = mean grad_slotB L
    over (1,0) prompts; in the 2d slot space the targets are (r_A, 0) and (0, r_B).
Methods fit on (0,0) prompts: 2-slot H (raw / whitened), broadcast H (1 slot = all positions,
fully unsupervised), and first-order baselines (mean grad, AGOP, grad-cov, PCA, whitened AGOP).

    python gate_test2.py [model_dir] [layers...]
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from common import OUT, Ctx, gpu_mem, load_model, log, patch_deltanet
from hl2 import span_recovery, sym_eig_top, windows
from organism import MODEL_DIR, make
from slots import SlotData, top_eigs_slots

K = 10
N_H = 32
# The B DoM is a mean over all question tokens, so one unit is diluted; a 4x shift is what makes
# the steering AND-check pass (out/_bscale.py sweep, Oct 4 23:20: L8 alpha 4 -> 00+A+B 0.83,
# 00+B 0.14, 00+A 0.03). Directions are unaffected; only the coupling / interaction sizes scale.
ALPHA_B = float(os.environ.get("HL_ALPHA_B", "4"))


def slot_positions(tok, x):
    ids = tok(x["prompt"], add_special_tokens=False)["input_ids"]
    n1 = len(tok("Speaker:", add_special_tokens=False)["input_ids"])
    pre = tok(f"Speaker: {x['speaker']}", add_special_tokens=False)["input_ids"]
    preq = tok(f"Speaker: {x['speaker']}. Question: {x['question']}", add_special_tokens=False)["input_ids"]
    assert ids[:len(pre)] == pre and ids[:len(preq)] == preq, x["prompt"]
    assert "?" in tok.decode(ids[len(preq) - 1:len(preq)]), (x["prompt"], tok.decode(ids[len(preq) - 1:len(preq)]))
    if os.environ.get("HL_BSLOT", "question") == "qmark":
        return ids, [list(range(n1, len(pre))), [len(preq) - 1]]
    preQ = tok(f"Speaker: {x['speaker']}. Question:", add_special_tokens=False)["input_ids"]
    assert ids[:len(preQ)] == preQ
    return ids, [list(range(n1, len(pre))), list(range(len(preQ), len(preq)))]   # B slot = all question tokens


def set_layer(ctxs, l):
    for c in ctxs:
        c.l = l; c.h = c.h_all[l]


def main(model_dir=str(MODEL_DIR), layers=(2, 4, 6, 8, 10, 12, 14), n_per_cell=64):
    t0 = time.time()
    tag = "gate2_" + Path(model_dir).name + "_" + os.environ.get("HL_BSLOT", "question")
    m = load_model(model_dir); patch_deltanet(m)
    items = make(4 * n_per_cell, "train", seed=7)
    ans = {a: m.tok(" " + a, add_special_tokens=False)["input_ids"][0] for a in ("Yes", "No")}
    keep = set(layers)
    ctxs, spos = [], []
    for x in items:
        ids, sp = slot_positions(m.tok, x)
        c = Ctx(m, torch.tensor(ids), layers[0])
        c.h_all = {k: v for k, v in c.h_all.items() if k in keep}
        ctxs.append(c); spos.append(sp)
    W = torch.stack([m.W_U[ans["No" if x["truth"] == "Yes" else "Yes"]].float() - m.W_U[ans[x["truth"]]].float()
                     for x in items]).contiguous()
    A = torch.tensor([x["A"] for x in items]); B = torch.tensor([x["B"] for x in items])
    cell = {(a, b): torch.nonzero((A == a) & (B == b)).flatten().tolist() for a in (0, 1) for b in (0, 1)}
    i00, i01, i10, i11 = cell[(0, 0)], cell[(0, 1)], cell[(1, 0)], cell[(1, 1)]
    D2 = SlotData(ctxs, W, spos, 2)
    res = {"model": model_dir, "alpha_B": ALPHA_B, "scan": [], "analysis": {}}

    def save():
        with open(OUT / f"{tag}.json", "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1)

    def sub(ix, S=2, slots=None):
        sp = [spos[j] for j in ix] if slots is None else [slots[j] for j in ix]
        return SlotData([ctxs[j] for j in ix], W[ix], sp, S)

    def slot_mean(l):
        out = torch.zeros(len(ctxs), 2, m.d, device="cuda")
        for j, c in enumerate(ctxs):
            for s in range(2):
                out[j, s] = c.h_all[l][0, spos[j][s]].mean(0)
        return out

    # ---------------- 1. layer scan: 2-slot steering AND-check
    for l in layers:
        set_layer(ctxs, l)
        Hs = slot_mean(l)
        dA = Hs[A == 1, 0].mean(0) - Hs[A == 0, 0].mean(0)
        dB = ALPHA_B * (Hs[B == 1, 1].mean(0) - Hs[B == 0, 1].mean(0))
        z = torch.zeros_like(dA)

        def rate(ix, a, b):
            sd = sub(ix)
            sh = torch.stack([a, b])[None].expand(len(ix), -1, -1)
            return float((sd.values(m, sh) > 0).float().mean())
        row = {"layer": l, "norm_dA": float(dA.norm()), "norm_dB": float(dB.norm()),
               "00": rate(i00, z, z), "00+A": rate(i00, dA, z), "00+B": rate(i00, z, dB), "00+A+B": rate(i00, dA, dB),
               "10+B": rate(i10, z, dB), "01+A": rate(i01, dA, z), "11-A": rate(i11, -dA, z), "11-B": rate(i11, z, -dB)}
        row["and_score"] = row["00+A+B"] - max(row["00+A"], row["00+B"])
        res["scan"].append(row); save()
        log(f"[gate2] L{l}: lie 00 {row['00']:.2f} +A {row['00+A']:.2f} +B {row['00+B']:.2f} +A+B {row['00+A+B']:.2f} | "
            f"10+B {row['10+B']:.2f} 01+A {row['01+A']:.2f} 11-A {row['11-A']:.2f} 11-B {row['11-B']:.2f} | AND {row['and_score']:+.2f}")
    best = max(res["scan"], key=lambda r: r["and_score"])
    lstar = best["layer"]; res["lstar"] = lstar
    if best["and_score"] < 0.3:
        log(f"[gate2] PRECONDITION FAILED (best AND {best['and_score']:+.2f} at L{lstar}); no analysis"); save(); return
    log(f"[gate2] chosen layer {lstar} (AND {best['and_score']:+.2f}); {time.time()-t0:.0f}s")

    # ---------------- 2. analysis at l*
    l = lstar
    set_layer(ctxs, l)
    Hs = slot_mean(l)
    dA = Hs[A == 1, 0].mean(0) - Hs[A == 0, 0].mean(0)
    dB = ALPHA_B * (Hs[B == 1, 1].mean(0) - Hs[B == 0, 1].mean(0))
    g2 = D2.grads(m).double()                                     # [N, 2, d]
    rA = g2[i01, 0].mean(0); rB = g2[i10, 1].mean(0)
    d = m.d
    Z = np.zeros(d)
    np_ = lambda t: t.double().cpu().numpy()
    tR = np.stack([np.concatenate([np_(rA), Z]), np.concatenate([Z, np_(rB)])])
    tD = np.stack([np.concatenate([np_(dA), Z]), np.concatenate([Z, np_(dB)])])
    wins = windows(m.tok, 32, T=64, seed=3)
    X = torch.cat([Ctx(m, x, l).h[0, 16:] for x in wins], 0).double()
    Sig = torch.cov(X.T)
    Sig = 0.95 * Sig + 0.05 * Sig.trace() / d * torch.eye(d, dtype=Sig.dtype, device=Sig.device)
    e, U = torch.linalg.eigh(Sig)
    Sw = (U * e.sqrt()) @ U.T; Swi = (U / e.sqrt()) @ U.T
    Swn, Swin = np_(Sw), np_(Swi)
    tRw = np.stack([np.concatenate([np_(rA) @ Swn, Z]), np.concatenate([Z, np_(rB) @ Swn])])
    tDw = np.stack([np.concatenate([np_(dA) @ Swin, Z]), np.concatenate([Z, np_(dB) @ Swin])])
    out = {}
    log(f"[gate2 L{l}] |rA| {rA.norm():.3f} |rB| {rB.norm():.3f} |dA| {dA.norm():.2f} |dB| {dB.norm():.2f} "
        f"cos(rA,dA) {float(rA @ dA.double() / rA.norm() / dA.norm()):+.2f} cos(rB,dB) {float(rB @ dB.double() / rB.norm() / dB.norm()):+.2f}; random span {K / (2 * d):.4f}")

    def score(name, lam, V, white=False, targets=None):
        tr, td = (tRw, tDw) if white else (tR, tD)
        if targets is not None:
            tr, td = targets
        V = np.atleast_2d(V)
        r = {"span_r": span_recovery(V, tr), "span_d": span_recovery(V, td),
             "span_rA": span_recovery(V, tr[:1]), "span_rB": span_recovery(V, tr[1:]),
             "lam": [float(x) for x in np.atleast_1d(lam)[:K]]}
        rn = tr / np.linalg.norm(tr, axis=1, keepdims=True)
        r["ov_per_vec"] = [[float((v @ rn[0]) ** 2), float((v @ rn[1]) ** 2)] for v in V[:K]]
        out[name] = r; res["analysis"] = {"layer": l, "methods": out}; save()
        log(f"[gate2 L{l}] {name:14s} span(rA,rB) {r['span_r']:.2f} [rA {r['span_rA']:.2f} rB {r['span_rB']:.2f}] span(dA,dB) {r['span_d']:.2f}")

    c00 = i00[:N_H]
    mix = [j for ix in (i00, i01, i10, i11) for j in ix[:N_H // 4]]
    D00, Dmix = sub(c00), sub(mix)
    lam, V = top_eigs_slots(m, D00, k=K); score("H2_raw_00", lam, V)
    lamw, Vw = top_eigs_slots(m, D00, k=K, Sw=Sw.float()); score("H2_white_00", lamw, Vw, white=True)
    lam, V = top_eigs_slots(m, Dmix, k=K); score("H2_raw_all", lam, V)
    lamw, Vw = top_eigs_slots(m, Dmix, k=K, Sw=Sw.float()); score("H2_white_all", lamw, Vw, white=True)
    G = g2[i00].reshape(len(i00), -1)
    score("meangrad_00", np.ones(1), np_(G.mean(0))[None])
    la, Va = sym_eig_top(np_(G.T @ G / len(G)), K); score("AGOP_00", la, Va)
    Gc = G - G.mean(0); lc, Vc = sym_eig_top(np_(Gc.T @ Gc / len(G)), K); score("gradcov_00", lc, Vc)
    Xs = Hs[i00].reshape(len(i00), -1).double(); Xs = Xs - Xs.mean(0)
    lp, Vp = sym_eig_top(np_(Xs.T @ Xs / len(i00)), K); score("PCA_00", lp, Vp)
    Gw = (g2[i00] @ Sw).reshape(len(i00), -1); law, Vaw = sym_eig_top(np_(Gw.T @ Gw / len(Gw)), K); score("AGOP_white_00", law, Vaw, white=True)
    # broadcast (1 slot = all positions): the fully unsupervised H-lens
    allpos = [[list(range(c.T))] for c in ctxs]
    D1 = SlotData(ctxs, W, allpos, 1)
    g1 = D1.grads(m).double()[:, 0]
    tB1 = np.stack([np_(g1[i01].mean(0)), np_(g1[i10].mean(0))])
    D1_00 = sub(c00, S=1, slots=allpos)
    lam1, V1 = top_eigs_slots(m, D1_00, k=K); score("Hbc_raw_00", lam1, V1, targets=(tB1, tB1))
    lam1w, V1w = top_eigs_slots(m, D1_00, k=K, Sw=Sw.float()); score("Hbc_white_00", lam1w, V1w, targets=(tB1 @ Swn, tB1 @ Swn))
    g100 = g1[i00]; la1, Va1 = sym_eig_top(np_(g100.T @ g100 / len(g100)), K); score("AGOPbc_00", la1, Va1, targets=(tB1, tB1))
    # coupling of the real pair vs random natural pairs (2-slot H over (0,0))
    vA = torch.stack([dA, torch.zeros_like(dA)]); vB = torch.stack([torch.zeros_like(dB), dB])
    Hv = D00.hmat(m, torch.stack([vA]).float())
    c_true = float((Hv[0] * vB).sum())
    rng = np.random.default_rng(0)
    Zr = (torch.tensor(rng.standard_normal((40, d)), device="cuda").double() @ Sw).float()
    Zr = Zr / Zr.norm(dim=1, keepdim=True)
    xs = torch.stack([torch.stack([Zr[i] * dA.norm(), torch.zeros_like(dA)]) for i in range(20)])
    ys = torch.stack([torch.stack([torch.zeros_like(dB), Zr[20 + i] * dB.norm()]) for i in range(20)])
    Hx = D00.hmat(m, xs)
    null = [float((Hx[i] * ys[i]).sum()) for i in range(20)]
    zsh = torch.zeros(len(c00), 2, d, device="cuda")
    fv = lambda a, b: D00.values(m, zsh + torch.stack([a, b])[None])
    zz = torch.zeros_like(dA)
    I_meas = float((fv(dA, dB) - fv(dA, zz) - fv(zz, dB) + fv(zz, zz)).mean())
    out["coupling"] = {"dA_H_dB": c_true, "null_abs_median": float(np.median(np.abs(null))), "null_abs_max": float(np.max(np.abs(null))),
                       "percentile": float(np.mean(np.abs(null) < abs(c_true))), "I_measured": I_meas}
    res["analysis"]["methods"] = out; save()
    log(f"[gate2 L{l}] coupling (dA,0)'H(0,dB) {c_true:+.3f} vs null |.| median {np.median(np.abs(null)):.3f} max {np.max(np.abs(null)):.3f} "
        f"(pct {out['coupling']['percentile']:.2f}); measured interaction of the real move {I_meas:+.2f}; total {time.time()-t0:.0f}s; {gpu_mem()}")


if __name__ == "__main__":
    md = sys.argv[1] if len(sys.argv) > 1 else str(MODEL_DIR)
    ls = tuple(int(x) for x in sys.argv[2:]) or (2, 4, 6, 8, 10, 12, 14)
    main(md, ls)
