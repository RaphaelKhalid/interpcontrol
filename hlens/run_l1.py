"""L1 rung (exact per-context Jacobians + per-token MLP curvature, no token-mixer curvature,
no K-FAC averaging) vs L0, for a few tokens. Separates L0-L2 gap into mixer vs K-FAC parts
(compare cosF(L1,L0) with cosF(L2,L0) from ladder_08b.json). Writes out/l1_08b.json."""
import json, os, sys, time
import numpy as np, torch
from common import OUT, Ctx, gpu_mem, load_model, log, patch_deltanet, wikitext_contexts
from hvp import AveragedH
import ladder as L

scratch = os.environ["HLENS_SCRATCH"]
toks = sys.argv[1].split("|") if len(sys.argv) > 1 else [" the", " of", " Paris", " France"]
cfg = json.load(open(OUT / "ladder_08b.json"))["config"]
m = load_model(cfg["model"]); patch_deltanet(m)
ctxs = [Ctx(m, x, cfg["l"]) for x in wikitext_contexts(m.tok, cfg["n_ctx"], T=cfg["T"])]
layers = cfg["layers"]
res = {}
for t in toks:
    tid = m.tok(t, add_special_tokens=False)["input_ids"][0]
    u = m.u([tid])[0]
    z = np.load(os.path.join(scratch, f"eig_08b_{tid}.npz"))
    t0 = time.time()
    op = L.L1Op(m, ctxs, u, layers)
    lam1, V1, te = L.l0_eigs(op, m.d, k=cfg["k"])
    n_mv = op.n_mv
    H0 = AveragedH(m, ctxs, u, batch=4)
    Z, Y0 = L.hutchinson(H0, m.d, n_probe=cfg["probes"])
    Y1 = op.matmat(Z)
    n0, n1, ip = (Y0 * Y0).sum(1), (Y1 * Y1).sum(1), (Y0 * Y1).sum(1)
    cos = (ip.mean() / torch.sqrt(n0.mean() * n1.mean())).item()
    r = {"eigvals": lam1.tolist(), "eig_time_s": te, "n_matvec_ctx": n_mv, "total_time_s": time.time() - t0,
         "overlap_top5": L.subspace_overlap(z["V"][:, :5], V1[:, :5]),
         "overlap_top10": L.subspace_overlap(z["V"][:, :10], V1[:, :10]),
         "overlap_top20": L.subspace_overlap(z["V"][:, :20], V1[:, :20]),
         "cosF_vs_L0": cos, "explained_unscaled": ((2 * ip.mean() - n1.mean()) / n0.mean()).item(),
         "F2_L1_over_F2_L0": (n1.mean() / n0.mean()).item(),
         "L0_top3": z["lam"][:3].tolist()}
    res[t] = r
    log(f"[L1] {t!r}: eig {te:.0f}s; L1 top3 {np.round(lam1[:3],2).tolist()} vs L0 {np.round(z['lam'][:3],2).tolist()}; "
        f"ov5 {r['overlap_top5']:.2f} ov10 {r['overlap_top10']:.2f} cosF {cos:.2f} |L1|^2/|L0|^2 {r['F2_L1_over_F2_L0']:.2f}; {gpu_mem()}")
    json.dump(res, open(OUT / "l1_08b.json", "w"), indent=1)
