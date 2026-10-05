"""The solve_triangular DeltaNet path equals transformers' torch fallback (fp32)."""
import torch
from common import Ctx, patch_deltanet, wikitext_contexts, run_from


def test_patch_matches_reference(model):
    m = model
    ids = wikitext_contexts(m.tok, 1, T=96)[0]
    patch_deltanet(m, False)
    c = Ctx(m, ids, 4)
    ref, _, _ = run_from(m, c, c.h, 5)
    patch_deltanet(m, True)
    out, _, _ = run_from(m, c, c.h, 5)
    rel = ((out - ref).norm() / ref.norm()).item()
    print("rel diff patched vs reference:", rel)
    assert rel < 1e-5
    # gradients too
    def gradof(on):
        patch_deltanet(m, on)
        h = c.h.clone().requires_grad_(True)
        o, _, _ = run_from(m, c, h, 5)
        (g,) = torch.autograd.grad(o[0, 16:95].sum(), h)
        return g
    g0, g1 = gradof(False), gradof(True)
    relg = ((g0 - g1).norm() / g0.norm()).item()
    print("rel grad diff:", relg)
    assert relg < 1e-4
