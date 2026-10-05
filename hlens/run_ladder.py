"""Ladder (T4) + eigenpair decoding (T3) on Qwen3.5-0.8B, layer l, N wikitext contexts.

Usage: python run_ladder.py [--n 16] [--l 12] [--T 96] [--k 20]
Writes out/ladder.json (incremental), out/t3_decode.json, scratch eigvecs (npz) in SCRATCH.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from common import OUT, Ctx, gpu_mem, load_lens, load_model, log, patch_deltanet, wikitext_contexts
from hvp import AveragedH
import ladder as L

SCRATCH = os.environ.get("HLENS_SCRATCH", os.path.join(os.environ.get("TEMP", "."), "hlens_scratch"))
os.makedirs(SCRATCH, exist_ok=True)

TOKENS = [" the", " of", " and", " was", " in", " Paris", " France", " war", " music", " he"]


def decode(m, Jl, v: torch.Tensor, k=10):
    """J-lens readout of a residual DIRECTION v at layer l: scores_t = (g*W_U[t]) . (J_l v)."""
    z = (Jl @ v) * m.gain
    sc = (m.W_U.float() @ z) if False else torch.cat([(m.W_U[i:i + 65536].float() @ z) for i in range(0, m.W_U.shape[0], 65536)])
    out = {}
    for sign, s in (("+", sc), ("-", -sc)):
        top = torch.topk(s, k).indices.tolist()
        out[sign] = [m.tok.decode([t]) for t in top]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--l", type=int, default=12)
    ap.add_argument("--T", type=int, default=96)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--probes", type=int, default=24)
    ap.add_argument("--tokens", type=str, default=None)
    ap.add_argument("--model", type=str, default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--tag", type=str, default="08b")
    args = ap.parse_args()
    toks = TOKENS if args.tokens is None else args.tokens.split("|")

    t0 = time.time()
    m = load_model(args.model)
    patch_deltanet(m)
    lens = load_lens(args.model)
    log(f"[ladder {args.tag}] model+lens loaded {time.time()-t0:.1f}s; {gpu_mem()}")
    ids = wikitext_contexts(m.tok, args.n, T=args.T)
    ctxs = [Ctx(m, x, args.l) for x in ids]
    layers = list(range(args.l + 1, m.n_layers))

    t = time.time()
    J = L.averaged_jacobians(m, ctxs, layers)
    tJ = time.time() - t
    t = time.time()
    stats = L.forward_stats(m, ctxs, layers)
    tS = time.time() - t
    log(f"[ladder {args.tag}] Jbar for {len(layers)} layers: {tJ:.1f}s; forward stats {tS:.1f}s; {gpu_mem()}")
    P = {}
    for i in layers:
        hd = stats[i]["hdir"] / stats[i]["hdir"].norm()
        P[i] = torch.eye(m.d, device="cuda") - torch.outer(hd, hd)
        stats[i]["hdir_norm"] = float(stats[i]["hdir"].norm())
    Jl = lens.jacobians[args.l].cuda()

    res_path = OUT / f"ladder_{args.tag}.json"
    dec_path = OUT / f"t3_decode_{args.tag}.json"
    results = {"config": {"model": args.model, "l": args.l, "n_ctx": args.n, "T": args.T, "k": args.k,
                          "probes": args.probes, "layers": layers, "t_Jbar_s": tJ, "t_stats_s": tS,
                          "mean_unit_mid_dir_norm": {i: stats[i]["hdir_norm"] for i in layers}},
               "tokens": {}}
    decodes = {}
    g = torch.Generator().manual_seed(1)
    for tokstr in toks:
        tid = m.tok(tokstr, add_special_tokens=False)["input_ids"]
        if len(tid) != 1:
            log(f"skip {tokstr!r}: not single token {tid}")
            continue
        u = m.u(tid)[0]
        r = {"token_id": tid[0]}
        # ---- L0
        Hop = AveragedH(m, ctxs, u, batch=4)
        torch.cuda.reset_peak_memory_stats()
        lam0, V0, tl = L.l0_eigs(Hop, m.d, k=args.k)
        r["L0"] = {"eigvals": lam0.tolist(), "time_s": tl, "n_hvp": Hop.n_hvp, "peak": gpu_mem()}
        n_eig_hvp = Hop.n_hvp
        t = time.time()
        Z, Y = L.hutchinson(Hop, m.d, n_probe=args.probes)
        r["L0"]["hutch_time_s"] = time.time() - t
        H0F2 = (Y * Y).sum(1).mean().item()
        r["L0"]["F2"] = H0F2
        r["L0"]["topk_energy_frac"] = float((lam0**2).sum() / H0F2)
        r["L0"]["top10_energy_frac"] = float((lam0[:10]**2).sum() / H0F2)
        # ---- L2 / L3
        t = time.time()
        C, Cn = L.l2_curvatures(m, ctxs, u, layers)
        H2 = L.assemble(J, C)
        H2n = L.assemble(J, Cn)
        tl2 = time.time() - t
        t = time.time()
        C3 = {}
        for i in layers:
            a = u if i == m.n_layers - 1 else lens.jacobians[i].cuda().T @ u
            C3[i] = L.l3_curv(m.layers[i], a / ctxs[0].n_valid, stats[i])
        H3 = L.assemble(J, C3)
        H3p = L.assemble(J, C3, P)
        tl3 = time.time() - t
        V0t = V0
        for name, Hk, tk in (("L2", H2, tl2), ("L2_noNormCurv", H2n, tl2), ("L3", H3, tl3), ("L3_radialProj", H3p, tl3)):
            lamk, Vk = L.top_eig_dense(Hk, args.k)
            fm = L.frob_metrics(Z, Y, Hk)
            r[name] = {"eigvals": lamk.tolist(), "time_s": tk,
                       "overlap_top10": L.subspace_overlap(V0t[:, :10], Vk[:, :10]),
                       "overlap_top5": L.subspace_overlap(V0t[:, :5], Vk[:, :5]),
                       "overlap_top20": L.subspace_overlap(V0t[:, :20], Vk[:, :20]), **fm}
            # Rayleigh check: how much of L0's top-1 direction curvature Hk reproduces
            v1 = torch.as_tensor(V0t[:, 0], dtype=torch.float32, device="cuda")
            r[name]["rayleigh_on_L0_top1"] = float(v1 @ Hk @ v1)
        r["L0"]["rayleigh_top1"] = float(lam0[0])
        # per-layer MLP Frobenius contributions (L2) to see which layer creates curvature
        r["L2_per_layer_F"] = {i: float((J[i].T @ C[i] @ J[i]).norm()) for i in layers}
        r["random_overlap_top10"] = 10 / m.d
        results["tokens"][tokstr] = r
        # ---- T3 decode
        dec = {"eigs": []}
        for j in range(5):
            v = torch.as_tensor(V0[:, j], dtype=torch.float32, device="cuda")
            dec["eigs"].append({"lambda": float(lam0[j]), **decode(m, Jl, v)})
        Q, _ = torch.linalg.qr(torch.randn(5, 5, generator=g))
        Vr = torch.as_tensor(V0[:, :5], dtype=torch.float32) @ Q
        dec["rotated_control"] = [decode(m, Jl, Vr[:, j].cuda()) for j in range(5)]
        dec["random_dir_control"] = [decode(m, Jl, torch.randn(m.d, generator=g).cuda()) for _ in range(2)]
        dec["lens_token_dir_self"] = decode(m, Jl, Jl.T @ u / (Jl.T @ u).norm())
        decodes[tokstr] = dec
        np.savez(os.path.join(SCRATCH, f"eig_{args.tag}_{tid[0]}.npz"), lam=lam0, V=V0)
        log(f"[ladder {args.tag}] {tokstr!r}: L0 eig {tl:.0f}s ({n_eig_hvp} HVP), hutch {r['L0']['hutch_time_s']:.0f}s; "
            f"top|lam| {lam0[:3].round(4).tolist()} top10 energy {r['L0']['top10_energy_frac']:.3f}; "
            + "; ".join(f"{n}: ov10 {r[n]['overlap_top10']:.2f} cosF {r[n]['cosF']:.2f}" for n in ("L2", "L2_noNormCurv", "L3", "L3_radialProj"))
            + f"; {gpu_mem()}")
        with open(res_path, "w") as f:
            json.dump(results, f, indent=1)
        with open(dec_path, "w", encoding="utf-8") as f:
            json.dump(decodes, f, indent=1, ensure_ascii=False)
    log(f"[ladder {args.tag}] done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
