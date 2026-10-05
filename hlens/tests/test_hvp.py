"""Exact HVP (double backward through the fp32 model incl. the DeltaNet torch path)
vs central finite differences of the exact gradient, on a short wikitext prompt."""
import torch
from common import Ctx, patch_deltanet, wikitext_contexts
from hvp import grad_and_hvp, grad_only, G_value


def _check(m, l, T=48, seed=0):
    torch.manual_seed(seed)
    ids = wikitext_contexts(m.tok, 1, T=T, offset=3)[0]
    c = Ctx(m, ids, l)
    u = m.u([m.tok(" the", add_special_tokens=False)["input_ids"][0]])[0]
    v = torch.randn(m.d, device="cuda"); v /= v.norm()
    g0, hv = grad_and_hvp(m, [c], u, v[None])
    hv = hv[0]
    scale = c.h[0, c.valid].norm(dim=-1).mean().item()
    best = (1e9, None)
    for rel in [3e-2, 1e-2, 3e-3, 1e-3]:
        eps = rel * scale
        gp = grad_only(m, [c], u, (eps * v)[None])[0]
        gm = grad_only(m, [c], u, (-eps * v)[None])[0]
        fd = (gp - gm) / (2 * eps)
        err = ((fd - hv).norm() / hv.norm()).item()
        # scalar second difference: v^T H v
        Gp, G0, Gm = (G_value(m, [c], u, (s * eps * v)[None]).item() for s in (1, 0, -1))
        q_fd = (Gp - 2 * G0 + Gm) / eps**2
        print(f"l={l} eps/|h|={rel:g}: rel err HVP={err:.2e}  vHv exact={(v@hv).item():.4e} fd={q_fd:.4e}")
        best = min(best, (err, rel), key=lambda t: t[0])
    return best


def test_hvp_fd(model):
    patch_deltanet(model)
    for l in (12, 8):
        err, rel = _check(model, l)
        assert err < 1e-2, (l, err)
