"""Post-hoc checks on the L0 eigenvectors (run after run_ladder.py):
- per-context Rayleigh quotients v^T H_c v for the top-3 eigvecs (is the average representative?)
- |cos| of each top eigvec with the mean unit residual direction at layer l (radial/massive dims)
  and with the J-lens token direction J_l^T u.
Writes out/post_<tag>.json.
"""
import glob
import json
import os
import sys

import numpy as np
import torch

from common import OUT, Ctx, load_lens, load_model, log, patch_deltanet, wikitext_contexts
from hvp import grad_and_hvp

tag = sys.argv[1] if len(sys.argv) > 1 else "08b"
scratch = os.environ["HLENS_SCRATCH"]
cfg = json.load(open(OUT / f"ladder_{tag}.json"))["config"]
m = load_model(cfg["model"]); patch_deltanet(m)
lens = load_lens(cfg["model"])
ctxs = [Ctx(m, x, cfg["l"]) for x in wikitext_contexts(m.tok, cfg["n_ctx"], T=cfg["T"])]
hdir = torch.stack([(c.h[0, 16:] / c.h[0, 16:].norm(dim=-1, keepdim=True)).mean(0) for c in ctxs]).mean(0)
hdir = hdir / hdir.norm()
Jl = lens.jacobians[cfg["l"]].cuda()
out = {}
for f in sorted(glob.glob(os.path.join(scratch, f"eig_{tag}_*.npz"))):
    tid = int(f.split("_")[-1].split(".")[0])
    z = np.load(f)
    u = m.u([tid])[0]
    jd = Jl.T @ u; jd = jd / jd.norm()
    rec = {"token": m.tok.decode([tid]), "eigs": []}
    for j in range(3):
        v = torch.as_tensor(z["V"][:, j], dtype=torch.float32, device="cuda")
        rq = []
        for s in range(0, len(ctxs), 4):
            cs = ctxs[s:s + 4]
            _, hv = grad_and_hvp(m, cs, u, v[None].expand(len(cs), -1).contiguous(), need_grad=False)
            rq += (hv @ v).tolist()
        rec["eigs"].append({"lambda": float(z["lam"][j]), "rayleigh_per_ctx_mean": float(np.mean(rq)),
                            "rayleigh_per_ctx_std": float(np.std(rq)), "rayleigh_min": float(np.min(rq)),
                            "rayleigh_max": float(np.max(rq)), "frac_same_sign": float(np.mean(np.sign(rq) == np.sign(z["lam"][j]))),
                            "abs_cos_mean_resid_dir": float(abs(v @ hdir)), "abs_cos_lens_token_dir": float(abs(v @ jd))})
    out[rec["token"]] = rec
    log(f"[post {tag}] {rec['token']!r}: " + "; ".join(
        f"lam {e['lambda']:.2f} ctx-RQ {e['rayleigh_per_ctx_mean']:.2f}+-{e['rayleigh_per_ctx_std']:.2f} samesign {e['frac_same_sign']:.2f} "
        f"cos(resid) {e['abs_cos_mean_resid_dir']:.2f} cos(Jtu) {e['abs_cos_lens_token_dir']:.2f}" for e in rec["eigs"]))
json.dump(out, open(OUT / f"post_{tag}.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
