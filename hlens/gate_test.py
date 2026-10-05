"""P2 main test: does the H-lens reveal a planted conditional-deception AND gate from HONEST data?

Organism: organism.py (lie iff speaker=pirate (A) AND topic=animal (B); no rule in the prompt).
Readout per prompt: L = logit(false answer) - logit(true answer) at the last position.

1. Layer scan (ground truth): DoM directions d_A, d_B at the last position of layer l; steering
   AND-check on honest (0,0) prompts: +d_A, +d_B, +d_A+d_B -> lie rate. The gate's inputs are
   present (and composed later) where only +d_A+d_B makes the model lie.
2. At the chosen layer: functional read directions r_A = mean grad L over (0,1) prompts,
   r_B = mean grad L over (1,0) prompts (the missing ingredient's direction).
3. Methods, all fit on honest-only (0,0) prompts unless stated:
   H_raw / H_white (mean last-position Hessian; white = Sigma^{1/2} H Sigma^{1/2}, Sigma from
   generic wikitext at layer l, no task information), H_all (mean over all four cells),
   meangrad (J-lens-like), AGOP, gradcov, PCA of activations.
   Metrics: span recovery of {r_A, r_B} and {d_A, d_B} (whitened methods: S r, S^-1 d);
   coupling d_A^T H d_B vs null of random natural directions with the same norms; predicted vs
   measured interaction of the real (0,0) -> (1,1) move.

    python gate_test.py [model_dir] [layers...]
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
from hl2 import AvgH, all_grads, span_recovery, sym_eig_top, top_eigs, values, windows
from organism import MODEL_DIR, make

K = 10
N_H = 32


def set_layer(ctxs, l):
    for c in ctxs:
        c.l = l; c.h = c.h_all[l]


def main(model_dir=str(MODEL_DIR), layers=(4, 6, 8, 10, 12, 14, 16, 18), n_per_cell=64, tag=None):
    tag = tag or "gate_" + Path(model_dir).name
    t0 = time.time()
    m = load_model(model_dir); patch_deltanet(m)
    if os.environ.get("HL_ITEMS") == "icl":      # in-context rule (no fine-tuning), for models that follow it
        from gate_prompts import build
        items = []
        for seed in range(100):
            items += [{**q, "prompt": q["text"]} for q in build(seed, n_demo_per_cell=4, chat_tok=m.tok)[1]]
            if len(items) >= 4 * n_per_cell:
                break
        tag = tag + "_icl"
    else:
        items = make(4 * n_per_cell, "train", seed=7)
    ans = {a: m.tok(" " + a, add_special_tokens=False)["input_ids"][0] for a in ("Yes", "No")}
    keep = set(layers) | {L + 1 for L in layers}
    ctxs = []
    for x in items:
        c = Ctx(m, torch.tensor(m.tok(x["prompt"], add_special_tokens=False)["input_ids"]), layers[0])
        c.h_all = {k: v for k, v in c.h_all.items() if k in keep}   # memory: only the scanned layers
        ctxs.append(c)
    W = torch.stack([m.W_U[ans["No" if x["truth"] == "Yes" else "Yes"]].float() - m.W_U[ans[x["truth"]]].float()
                     for x in items]).contiguous()
    A = torch.tensor([x["A"] for x in items]); B = torch.tensor([x["B"] for x in items])
    cell = {(a, b): torch.nonzero((A == a) & (B == b)).flatten().tolist() for a in (0, 1) for b in (0, 1)}
    L0 = values(m, ctxs, W)
    beh = {f"{a}{b}": {"lie_rate": float((L0[ix] > 0).float().mean()), "mean_L": float(L0[ix].mean())}
           for (a, b), ix in cell.items()}
    log(f"[gate] behaviour on analysis set: " + " | ".join(f"{k}: lie {v['lie_rate']:.2f} L {v['mean_L']:+.2f}" for k, v in beh.items()))
    res = {"model": model_dir, "behaviour": beh, "scan": [], "analysis": {}}

    def save():
        with open(OUT / f"{tag}.json", "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1)

    i00, i01, i10, i11 = cell[(0, 0)], cell[(0, 1)], cell[(1, 0)], cell[(1, 1)]
    sub = lambda ix: ([ctxs[j] for j in ix], W[ix])
    # ---------------- 1. layer scan
    for l in layers:
        set_layer(ctxs, l)
        H = torch.stack([c.h[0, -1] for c in ctxs])
        dA = H[A == 1].mean(0) - H[A == 0].mean(0); dB = H[B == 1].mean(0) - H[B == 0].mean(0)
        c00, w00 = sub(i00)
        rate = lambda cs, ws, sh: float((values(m, cs, ws, sh[None].expand(len(cs), -1)) > 0).float().mean())
        row = {"layer": l, "norm_dA": float(dA.norm()), "norm_dB": float(dB.norm()),
               "cos_dA_dB": float(dA @ dB / dA.norm() / dB.norm()), "norm_h": float(H.norm(dim=1).mean()),
               "00": rate(c00, w00, 0 * dA), "00+A": rate(c00, w00, dA), "00+B": rate(c00, w00, dB),
               "00+A+B": rate(c00, w00, dA + dB),
               "10+B": rate(*sub(i10), dB), "01+A": rate(*sub(i01), dA), "11-A": rate(*sub(i11), -dA), "11-B": rate(*sub(i11), -dB)}
        row["and_score"] = row["00+A+B"] - max(row["00+A"], row["00+B"])
        res["scan"].append(row); save()
        log(f"[gate] L{l}: lie 00 {row['00']:.2f} +A {row['00+A']:.2f} +B {row['00+B']:.2f} +A+B {row['00+A+B']:.2f} | "
            f"10+B {row['10+B']:.2f} 01+A {row['01+A']:.2f} 11-A {row['11-A']:.2f} 11-B {row['11-B']:.2f} | AND {row['and_score']:+.2f}")
    best = max(res["scan"], key=lambda r: r["and_score"])
    lstar = best["layer"]
    if os.environ.get("HL_SCAN_ONLY"):
        log(f"[gate] scan only; best layer {lstar} (AND {best['and_score']:+.2f})"); return
    if best["and_score"] < 0.3:
        log(f"[gate] PRECONDITION FAILED at the last position (best AND score {best['and_score']:+.2f}); no analysis"); return
    res["lstar"] = lstar
    log(f"[gate] chosen layer {lstar} (AND score {best['and_score']:+.2f}); {time.time()-t0:.0f}s")
    analyse(m, ctxs, W, A, B, cell, lstar, res, save)
    log(f"[gate] done {time.time()-t0:.0f}s; {gpu_mem()}")


def analyse(m, ctxs, W, A, B, cell, l, res, save):
    t0 = time.time()
    set_layer(ctxs, l)
    i00, i01, i10, i11 = cell[(0, 0)], cell[(0, 1)], cell[(1, 0)], cell[(1, 1)]
    H = torch.stack([c.h[0, -1] for c in ctxs])
    dA = H[A == 1].mean(0) - H[A == 0].mean(0); dB = H[B == 1].mean(0) - H[B == 0].mean(0)
    g = all_grads(m, ctxs, W).double()
    rA = g[i01].mean(0); rB = g[i10].mean(0)
    # generic whitening metric at layer l (no task information)
    wins = windows(m.tok, 32, T=64, seed=3)
    X = torch.cat([Ctx(m, x, l).h[0, 16:] for x in wins], 0).double()
    Sig = torch.cov(X.T)
    Sig = 0.95 * Sig + 0.05 * Sig.trace() / m.d * torch.eye(m.d, dtype=Sig.dtype, device=Sig.device)
    e, U = torch.linalg.eigh(Sig)
    S = (U * e.sqrt()) @ U.T; Si = (U / e.sqrt()) @ U.T
    S32 = S.float()
    np_ = lambda t: t.double().cpu().numpy()
    T_raw = {"r": np_(torch.stack([rA, rB])), "d": np_(torch.stack([dA, dB]).double())}
    T_white = {"r": np_(torch.stack([rA, rB]) @ S), "d": np_(torch.stack([dA, dB]).double() @ Si)}
    c00 = [ctxs[j] for j in i00[:N_H]]; w00 = W[i00[:N_H]]
    mix = [j for ix in (i00, i01, i10, i11) for j in ix[:N_H // 4]]
    cm, wm = [ctxs[j] for j in mix], W[mix]
    out = {}

    def score(name, lam, V, white=False):
        Tt = T_white if white else T_raw
        r = {"span_r": span_recovery(V, Tt["r"]), "span_d": span_recovery(V, Tt["d"]),
             "span_rA": span_recovery(V, Tt["r"][:1]), "span_rB": span_recovery(V, Tt["r"][1:]),
             "lam": [float(x) for x in np.atleast_1d(lam)[:K]]}
        # per-eigvec overlaps with the two r directions (pair signature)
        rn = Tt["r"] / np.linalg.norm(Tt["r"], axis=1, keepdims=True)
        r["ov_rA_rB_per_vec"] = [[float((v @ rn[0]) ** 2), float((v @ rn[1]) ** 2)] for v in np.atleast_2d(V)[:K]]
        out[name] = r; res["analysis"] = {"layer": l, **res["analysis"], "methods": out}; save()
        log(f"[gate L{l}] {name:10s} span(rA,rB) {r['span_r']:.2f} [rA {r['span_rA']:.2f} rB {r['span_rB']:.2f}] span(dA,dB) {r['span_d']:.2f}")

    log(f"[gate L{l}] |rA| {rA.norm():.3f} |rB| {rB.norm():.3f} cos(rA,rB) {float(rA @ rB / rA.norm() / rB.norm()):+.2f} "
        f"cos(rA,dA) {float(rA @ dA.double() / rA.norm() / dA.norm()):+.2f} cos(rB,dB) {float(rB @ dB.double() / rB.norm() / dB.norm()):+.2f}; random span {K / m.d:.3f}")
    lam, V = top_eigs(AvgH(m, c00, w00), m.d, k=K); score("H_raw_00", lam, V)
    lamw, Vw = top_eigs(AvgH(m, c00, w00, S=S32), m.d, k=K); score("H_white_00", lamw, Vw, white=True)
    lam_m, V_m = top_eigs(AvgH(m, cm, wm), m.d, k=K); score("H_raw_all", lam_m, V_m)
    lam_mw, V_mw = top_eigs(AvgH(m, cm, wm, S=S32), m.d, k=K); score("H_white_all", lam_mw, V_mw, white=True)
    g00 = g[i00]
    score("meangrad_00", np.ones(1), np_(g00.mean(0))[None])
    C = np_(g00.T @ g00 / len(g00)); la, Va = sym_eig_top(C, K); score("AGOP_00", la, Va)
    gc = g00 - g00.mean(0); lc, Vc = sym_eig_top(np_(gc.T @ gc / len(g00)), K); score("gradcov_00", lc, Vc)
    Xc = H[i00].double() - H[i00].double().mean(0); lp, Vp = sym_eig_top(np_(Xc.T @ Xc / len(i00)), K); score("PCA_00", lp, Vp)
    gw = g00 @ S; Cw = np_(gw.T @ gw / len(g00)); law, Vaw = sym_eig_top(Cw, K); score("AGOP_white_00", law, Vaw, white=True)
    # coupling of the true pair vs random natural directions of equal norm (exact HVPs, H over (0,0))
    op = AvgH(m, c00, w00)
    dd = torch.stack([dA, dB]).float()
    Hd = op.matmat(dd)
    c_true = float(Hd[0] @ dd[1])
    rng = np.random.default_rng(0)
    Z = torch.tensor(rng.standard_normal((60, m.d)), device="cuda").double() @ S   # ~ N(0, Sigma)
    Z = Z.float(); Z = Z / Z.norm(dim=1, keepdim=True)
    xs = Z[:30] * dA.norm(); ys = Z[30:] * dB.norm()
    Hx = op.matmat(xs)
    null = [float(Hx[i] @ ys[i]) for i in range(30)]
    # predicted vs measured interaction of the real move (0,0) -> (1,1) at layer l
    f = lambda sh: values(m, c00, w00, sh[None].expand(len(c00), -1))
    I_meas = float((f(dA + dB) - f(dA) - f(dB) + f(0 * dA)).mean())
    out["coupling"] = {"dA_H_dB": c_true, "null_abs_median": float(np.median(np.abs(null))),
                       "null_abs_max": float(np.max(np.abs(null))),
                       "percentile": float(np.mean(np.abs(null) < abs(c_true))), "I_measured": I_meas}
    res["analysis"]["methods"] = out; save()
    log(f"[gate L{l}] coupling dA'H dB {c_true:+.3f} vs null |.| median {np.median(np.abs(null)):.3f} max {np.max(np.abs(null)):.3f} "
        f"(pct {out['coupling']['percentile']:.2f}); measured interaction of the real move {I_meas:+.2f} logit; {time.time()-t0:.0f}s")


if __name__ == "__main__":
    md = sys.argv[1] if len(sys.argv) > 1 else str(MODEL_DIR)
    ls = tuple(int(x) for x in sys.argv[2:]) or (4, 6, 8, 10, 12, 14, 16, 18)
    main(md, ls)
