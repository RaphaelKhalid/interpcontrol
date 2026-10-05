"""E1: injection-recovery calibration of the H-lens (Qwen3.5-0.8B, laptop, $0).

Plant TWO exact AND-gates into the real model and ask whether the H-lens (mean last-position
Hessian of a real logit w.r.t. h_l) finds them amid the model's own curvature, and whether it
pairs them correctly (a with b, c with e), versus first-order baselines (mean gradient = J-lens
style; AGOP = mean gradient outer product, given 8x more contexts).

Gate: at the output of block l+1, last position only:
    out += [ s1 (a.x)(b.x) + s2 (c.x)(e.x) ] * w,     x = h_l[last] - mu
a, b, c, e: random unit directions in the top-256 PC subspace of h_l, mutually Sigma-orthogonal
(uncorrelated features). w: unit mean gradient of f w.r.t. block l+1's output (the gate output
moves the logit). s is set so the gate's pure interaction at +1 sigma of each input is G logits:
    G = s * kappa * sigma_a * sigma_b.
Ground truth in raw coordinates: span{a,b,c,e}, pairs (a,b), (c,e). In whitened coordinates
(z with delta = Sigma^{1/2} z), the gate's signature is span{S a, ...}, S = Sigma^{1/2}.
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np
import torch

from common import OUT, Ctx, gpu_mem, load_model, log, patch_deltanet
from hl2 import windows
from hl2 import AvgH, all_grads, batched, pairing_correct, span_recovery, sym_eig_top, top_eigs, values

L_ = 12
T_ = 64
N_H = 8      # contexts for the H-lens
N_G = 64     # contexts for AGOP / mean gradient (cheap, so 8x more data)
K = 10
GRID = [0.1, 0.3, 1.0, 3.0]
DRAWS = 3
TOKEN = " the"


def main(tag="inject_08b"):
    t0 = time.time()
    m = load_model(); patch_deltanet(m)
    tok_id = m.tok(TOKEN, add_special_tokens=False)["input_ids"]; assert len(tok_id) == 1
    ids = windows(m.tok, N_G + 32, T=T_, seed=1)
    ctx_all = [Ctx(m, x, L_) for x in ids[:N_G]]
    # activation stats at layer l (positions >= 16 of 32 held-out contexts + the last positions)
    X = torch.cat([Ctx(m, x, L_).h[0, 16:] for x in ids[N_G:N_G + 32]], 0).double()
    mu = X.mean(0)
    Sig = torch.cov(X.T)
    ev, U = torch.linalg.eigh(Sig)
    ev, U = ev.flip(0), U.flip(1)
    rho = 0.05
    Sig_s = (1 - rho) * Sig + rho * Sig.trace() / m.d * torch.eye(m.d, dtype=Sig.dtype, device=Sig.device)
    e2, U2 = torch.linalg.eigh(Sig_s)
    S = (U2 * e2.clamp_min(0).sqrt()) @ U2.T                     # Sigma^{1/2}
    S32 = S.float()
    W = m.W_U[tok_id[0]].float()[None].expand(N_G, -1).contiguous()
    ctxH = ctx_all[:N_H]; WH = W[:N_H]
    log(f"[E1] setup {time.time()-t0:.0f}s; top PC var share(256) {float(ev[:256].sum()/ev.sum()):.2f}; {gpu_mem()}")

    # gate output direction w at block l+1 output
    ctx1 = [Ctx(m, x, L_ + 1) for x in ids[:N_H]]
    g1 = all_grads(m, ctx1, WH)
    w = g1.mean(0); kappa_vec = g1 @ (w / w.norm()); w = w / w.norm(); kappa = float(kappa_vec.mean())
    del ctx1
    log(f"[E1] kappa {kappa:.3f} (per-ctx {kappa_vec.min():.3f}..{kappa_vec.max():.3f})")

    state = {"gate": None}

    def hook(mod, args, kwargs, out):
        gs = state["gate"]
        if gs is None:
            return out
        hin = args[0] if args else kwargs["hidden_states"]
        x = hin[:, -1] - mu.float()
        add = sum(s * (x @ a) * (x @ b) for s, a, b in gs)
        o = out[0] if isinstance(out, tuple) else out
        o = o.clone(); o[:, -1] = o[:, -1] + add[:, None] * w[None]
        return (o, *out[1:]) if isinstance(out, tuple) else o

    hk = m.layers[L_ + 1].register_forward_hook(hook, with_kwargs=True)
    res = {"layer": L_, "T": T_, "N_H": N_H, "N_G": N_G, "token": TOKEN, "kappa": kappa, "K": K, "runs": []}

    def save():
        with open(OUT / f"{tag}.json", "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1)

    # natural background: G = 0
    t1 = time.time()
    lam0, V0 = top_eigs(AvgH(m, ctxH, WH), m.d, k=K)
    lam0w, V0w = top_eigs(AvgH(m, ctxH, WH, S=S32), m.d, k=K)
    # natural 1-sigma pairwise interaction scale for random uncorrelated PC-space pairs (exact HVPs)
    rng = np.random.default_rng(0)
    P = U[:, :256].float()
    Z = torch.tensor(rng.standard_normal((40, 256)), dtype=torch.float32, device="cuda")
    dirs = Z @ P.T; dirs = dirs / dirs.norm(dim=1, keepdim=True)
    Hd = AvgH(m, ctxH, WH).matmat(dirs[:20])
    nat = []
    for i in range(20):
        x, y = dirs[i], dirs[20 + i]
        sx, sy = float((x.double() @ Sig @ x.double()).sqrt()), float((y.double() @ Sig @ y.double()).sqrt())
        nat.append(abs(float(Hd[i] @ y)) * sx * sy)
    res["background"] = {"lam_raw": lam0.tolist(), "lam_white": lam0w.tolist(),
                         "nat_interaction_1sigma": {"median": float(np.median(nat)), "max": float(np.max(nat)),
                                                    "values": nat}, "secs": time.time() - t1}
    log(f"[E1] background {time.time()-t1:.0f}s: raw |lam| {np.abs(lam0[:3]).round(2).tolist()}, white {np.abs(lam0w[:3]).round(2).tolist()}, "
        f"natural 1-sigma pair interaction median {np.median(nat):.4f} max {np.max(nat):.4f} logit")
    save()

    for r in range(DRAWS):
        # 4 Sigma-orthogonal unit directions in the top-256 PC subspace
        sv = ev[:256].float()
        Z = torch.tensor(np.random.default_rng(100 + r).standard_normal((4, 256)), dtype=torch.float32, device="cuda")
        Q = []
        for z in Z:
            for q in Q:
                z = z - (z * sv * q).sum() / (q * sv * q).sum() * q
            Q.append(z)
        D = torch.stack(Q) @ P.T
        D = D / D.norm(dim=1, keepdim=True)
        a, b, c, e = D
        sig = [float((x.double() @ Sig @ x.double()).sqrt()) for x in D]
        Dn = D.double().cpu().numpy()
        Dw = (D.double() @ S).cpu().numpy()          # whitened-coordinate signature S a, ...
        # metrics at G = 0 for this draw (null)
        null = {"raw_span": span_recovery(V0, Dn), "white_span": span_recovery(V0w, Dw),
                "raw_pair": pairing_correct(lam0, V0, list(Dn))[0], "white_pair": pairing_correct(lam0w, V0w, list(Dw))[0]}
        for G in GRID:
            t2 = time.time()
            s1 = G / (kappa * sig[0] * sig[1]); s2 = G / (kappa * sig[2] * sig[3])
            state["gate"] = [(s1, a, b), (s2, c, e)]
            # empirical check of the planted interaction at +1 sigma
            sh = lambda v: v[None].expand(N_H, -1)
            f00 = values(m, ctxH, WH); fa = values(m, ctxH, WH, sh(sig[0] * a)); fb = values(m, ctxH, WH, sh(sig[1] * b))
            fab = values(m, ctxH, WH, sh(sig[0] * a + sig[1] * b))
            I_ab = float((fab - fa - fb + f00).mean())
            lam, V = top_eigs(AvgH(m, ctxH, WH), m.d, k=K)
            lamw, Vw = top_eigs(AvgH(m, ctxH, WH, S=S32), m.d, k=K)
            g = all_grads(m, ctx_all, W).double()
            C = (g.T @ g / len(g)).cpu().numpy()
            gc = g - g.mean(0)
            Cc = (gc.T @ gc / len(g)).cpu().numpy()
            la, Va = sym_eig_top(C, K); lc, Vc = sym_eig_top(Cc, K)
            jl = g.mean(0).cpu().numpy()[None]
            row = {"draw": r, "G": G, "sigma": sig, "I_ab_measured": I_ab, "s": [s1, s2],
                   "gate_lam_raw": s1 * kappa, "lam_raw": lam.tolist(), "lam_white": lamw.tolist(),
                   "span": {"H_raw": span_recovery(V, Dn), "H_white": span_recovery(Vw, Dw),
                            "AGOP": span_recovery(Va, Dn), "gradcov": span_recovery(Vc, Dn), "meangrad": span_recovery(jl, Dn),
                            "random_k": K / m.d},
                   "span_gate1": {"H_raw": span_recovery(V, Dn[:2]), "H_white": span_recovery(Vw, Dw[:2])},
                   "pair": {"H_raw": pairing_correct(lam, V, list(Dn)), "H_white": pairing_correct(lamw, Vw, list(Dw)),
                            "AGOP": pairing_correct(la, Va, list(Dn)), "gradcov": pairing_correct(lc, Vc, list(Dn))},
                   "null_G0": null, "secs": time.time() - t2}
            res["runs"].append(row); save()
            log(f"[E1] draw {r} G={G}: I_meas {I_ab:+.3f} | span H {row['span']['H_raw']:.2f} Hw {row['span']['H_white']:.2f} "
                f"AGOP {row['span']['AGOP']:.2f} cov {row['span']['gradcov']:.2f} mg {row['span']['meangrad']:.2f} "
                f"(null H {null['raw_span']:.2f}/{null['white_span']:.2f}) | pair H {row['pair']['H_raw'][0]} Hw {row['pair']['H_white'][0]} "
                f"AGOP {row['pair']['AGOP'][0]} cov {row['pair']['gradcov'][0]} | gate lam {s1*kappa:.2f} vs top |lam| {abs(lam0[0]):.2f} | {time.time()-t2:.0f}s")
        state["gate"] = None
    hk.remove()
    log(f"[E1] done {time.time()-t0:.0f}s; {gpu_mem()}")


if __name__ == "__main__":
    main()
