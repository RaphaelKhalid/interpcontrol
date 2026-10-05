import sys, torch
sys.path.insert(0, '.')
from common import Ctx, load_model, patch_deltanet
from organism import make
from gate_test2 import slot_positions, set_layer
from slots import SlotData
m = load_model("C:/Users/rapha/models/hlens-pirate-0.8b"); patch_deltanet(m)
items = make(256, "train", seed=7)
ans = {a: m.tok(" " + a, add_special_tokens=False)["input_ids"][0] for a in ("Yes", "No")}
ctxs, spos = [], []
for x in items:
    ids, sp = slot_positions(m.tok, x); c = Ctx(m, torch.tensor(ids), 6); c.h_all = {k: v for k, v in c.h_all.items() if k in (4, 6, 8)}; ctxs.append(c); spos.append(sp)
W = torch.stack([m.W_U[ans["No" if x["truth"] == "Yes" else "Yes"]].float() - m.W_U[ans[x["truth"]]].float() for x in items])
A = torch.tensor([x["A"] for x in items]); B = torch.tensor([x["B"] for x in items])
cell = {(a, b): torch.nonzero((A == a) & (B == b)).flatten().tolist() for a in (0, 1) for b in (0, 1)}
for l in (4, 6, 8):
    set_layer(ctxs, l)
    Hs = torch.stack([torch.stack([c.h[0, sp[0]].mean(0), c.h[0, sp[1]].mean(0)]) for c, sp in zip(ctxs, spos)])
    dA = Hs[A == 1, 0].mean(0) - Hs[A == 0, 0].mean(0); dB = Hs[B == 1, 1].mean(0) - Hs[B == 0, 1].mean(0)
    z = torch.zeros_like(dA)
    def rate(ix, a, b):
        sd = SlotData([ctxs[j] for j in ix], W[ix], [spos[j] for j in ix], 2)
        return float((sd.values(m, torch.stack([a, b])[None].expand(len(ix), -1, -1)) > 0).float().mean())
    for al in (1, 2, 4, 8, 16):
        print(f"L{l} alpha {al}: 00+aB {rate(cell[(0,0)], z, al*dB):.2f} 10+aB {rate(cell[(1,0)], z, al*dB):.2f} 00+A+aB {rate(cell[(0,0)], dA, al*dB):.2f} 11-aB {rate(cell[(1,1)], z, -al*dB):.2f} | |dB| {dB.norm():.2f} |h| {Hs[:,1].norm(dim=1).mean():.1f}", flush=True)
