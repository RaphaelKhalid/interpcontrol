"""swiglu_curv_apply == per-token closed-form matrices; L1Op is symmetric (v^T L1 w = w^T L1 v)."""
import torch
from common import Ctx, patch_deltanet, wikitext_contexts, run_from
from ladder import swiglu_curv_sum, swiglu_curv_apply, L1Op


def test_apply_and_symmetry(model):
    m = model; patch_deltanet(m)
    c = Ctx(m, wikitext_contexts(m.tok, 1, T=64)[0], 12)
    with torch.no_grad():
        _, _, mids = run_from(m, c, c.h, 13, stop=14, record_mid=[13])
    torch.manual_seed(0)
    H = mids[13][0, 20:24]; A = torch.randn(4, m.d, device="cuda") * 0.05; D = torch.randn(4, m.d, device="cuda")
    Cs = swiglu_curv_sum(m.layers[13], H, A, per_token=True)
    ref = torch.einsum("nij,nj->ni", Cs, D)
    got = swiglu_curv_apply(m.layers[13], H, A, D)
    err = ((ref - got).norm() / ref.norm()).item(); print("apply rel err", err); assert err < 1e-4
    u = m.u([m.tok(" the", add_special_tokens=False)["input_ids"][0]])[0]
    op = L1Op(m, [c], u, list(range(13, 24)))
    v, w = torch.randn(2, m.d, device="cuda")
    a, b = (w @ op.matvec(v)).item(), (v @ op.matvec(w)).item()
    print("symmetry", a, b); assert abs(a - b) < 1e-3 * (abs(a) + abs(b))
