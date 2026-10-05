"""Closed-form SwiGLU + RMSNorm curvature vs the autograd Hessian of a . MLP(N2(h))."""
import torch
from common import Ctx, patch_deltanet, wikitext_contexts, run_from
from ladder import swiglu_curv_sum


def test_closed_form_vs_autograd(model):
    m = model
    patch_deltanet(m)
    c = Ctx(m, wikitext_contexts(m.tok, 1, T=64)[0], 12)
    with torch.no_grad():
        _, _, mids = run_from(m, c, c.h, 13, stop=16, record_mid=[13, 15])
    torch.manual_seed(0)
    for i in (13, 15):
        blk = m.layers[i]
        H = mids[i][0, [20, 40, 50]]
        A = torch.randn(3, m.d, device="cuda") * 0.05
        Cs = swiglu_curv_sum(blk, H, A, per_token=True)
        for q in range(3):
            f = lambda h: (A[q] * blk.mlp(blk.post_attention_layernorm(h))).sum()
            Hq = torch.autograd.functional.hessian(f, H[q], vectorize=True)
            err = ((Hq - Cs[q]).norm() / Hq.norm()).item()
            print(f"block {i} tok {q}: rel err closed form vs autograd = {err:.2e}  |H|={Hq.norm():.3e}")
            assert err < 1e-4
        S = swiglu_curv_sum(blk, H, A)
        errs = ((S - Cs.sum(0)).norm() / S.norm()).item()
        print(f"block {i}: summed (low-rank) form vs per-token sum rel err = {errs:.2e}")
        assert errs < 1e-4
