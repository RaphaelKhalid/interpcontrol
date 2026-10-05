"""T1 / T2: J-lens coordinate swaps on two-hop prompts, first- vs second-order prediction.

Swap construction (our reading of "patch in lens coordinates via pseudoinverse, leave the
orthogonal complement unchanged"): at layer l, last position, let A = [u_X; u_Y] J_l (2 x d),
u_t = g * W_U[t]. Lens coordinates a = A h. The swap sets a' = (a_Y, a_X) and patches
Delta = pinv(A) (a' - a): the minimum-norm change, lying in rowspace(A); the component of h
in null(A) is unchanged. Scaled versions alpha*Delta are also run.

Readout f(h_l) = logit[ans_swap] - logit[ans_orig] at the last position (full model:
blocks l+1..23, final RMSNorm, tied unembedding; fp32).
T1: true f(h+aD)-f(h)  vs  1st = a g.D  vs  2nd = 1st + a^2/2 D^T H D (one HVP).
T2: interaction I = f(h+DA+DB)-f(h+DA)-f(h+DB)+f(h) vs DA^T H DB (one HVP), DB = relation swap.
"""

from __future__ import annotations

import json
import time

import torch

from common import OUT, Ctx, gpu_mem, load_lens, load_model, log, patch_deltanet, run_from
from hvp import grad_hvp_fn

# (family, prompt, intermediate, swap_intermediate, answer, swap_answer, relation, swap_relation)
PROMPTS = [
    ("capital", "The capital of the country where the Eiffel Tower is located is", " France", " Italy", " Paris", " Rome", " capital", " language"),
    ("capital", "The capital of the country where the Colosseum is located is", " Italy", " France", " Rome", " Paris", " capital", " language"),
    ("capital", "The capital of the country where the Brandenburg Gate is located is", " Germany", " Spain", " Berlin", " Madrid", " capital", " language"),
    ("capital", "The capital of the country where the Kremlin is located is", " Russia", " China", " Moscow", " Beijing", " capital", " language"),
    ("capital", "The capital of the country where the Great Wall is located is", " China", " Japan", " Beijing", " Tokyo", " capital", " language"),
    ("capital", "The capital of the country where Mount Fuji is located is", " Japan", " China", " Tokyo", " Beijing", " capital", " language"),
    ("capital", "The capital of the country where the Sagrada Familia is located is", " Spain", " Germany", " Madrid", " Berlin", " capital", " language"),
    ("capital", "The capital of the country where the Acropolis is located is", " Greece", " Egypt", " Athens", " Cairo", " capital", " language"),
    ("capital", "The capital of the country where the pyramids of Giza are located is", " Egypt", " Greece", " Cairo", " Athens", " capital", " language"),
    ("capital", "The capital of the country where Big Ben is located is", " England", " France", " London", " Paris", " capital", " language"),
    ("language", "The language spoken in the country whose capital is Paris is", " France", " Germany", " French", " German", " language", " currency"),
    ("language", "The language spoken in the country whose capital is Berlin is", " Germany", " Italy", " German", " Italian", " language", " currency"),
    ("language", "The language spoken in the country whose capital is Rome is", " Italy", " Spain", " Italian", " Spanish", " language", " currency"),
    ("language", "The language spoken in the country whose capital is Madrid is", " Spain", " France", " Spanish", " French", " language", " currency"),
    ("language", "The language spoken in the country whose capital is Tokyo is", " Japan", " China", " Japanese", " Chinese", " language", " currency"),
    ("legs", "The number of legs on the animal that spins webs is", " spider", " ant", " eight", " six", " legs", " eyes"),
    ("legs", "The number of legs on the insect that makes honey is", " bee", " spider", " six", " eight", " legs", " eyes"),
    ("legs", "The number of legs on the animal that says moo is", " cow", " chicken", " four", " two", " legs", " eyes"),
    ("legs", "The number of legs on the bird that lays eggs for breakfast is", " chicken", " cow", " two", " four", " legs", " eyes"),
    ("color", "The color of the fruit that keeps the doctor away is", " apple", " banana", " red", " yellow", " color", " shape"),
    ("color", "The color of the fruit that monkeys love to eat is", " banana", " apple", " yellow", " red", " color", " shape"),
    ("color", "The color of the vegetable that rabbits love to eat is", " carrot", " lettuce", " orange", " green", " color", " shape"),
]


# v2: same items with a one-shot prefix that fixes the answer format (v1 attrition: 1/22 passed
# the strict top-1 check; the model mostly predicted ':' / ' known' / ' ' after "is").
SHOT = {
    "capital": "The capital of the country where the Statue of Liberty is located is Washington. ",
    "language": "The language spoken in the country whose capital is Lisbon is Portuguese. ",
    "legs": "The number of legs on the animal that barks is four. ",
    "color": "The color of the sky on a clear day is blue. ",
}
EXTRA = [
    ("capital", "The capital of the country where the Taj Mahal is located is", " India", " China", " New", " Beijing", " capital", " language"),
    ("capital", "The capital of the country where the Leaning Tower of Pisa is located is", " Italy", " Germany", " Rome", " Berlin", " capital", " language"),
    ("language", "The language spoken in the country whose capital is Moscow is", " Russia", " Germany", " Russian", " German", " language", " currency"),
    ("language", "The language spoken in the country whose capital is Beijing is", " China", " Japan", " Chinese", " Japanese", " language", " currency"),
    ("legs", "The number of legs on the animal that has a long trunk is", " elephant", " chicken", " four", " two", " legs", " eyes"),
    ("color", "The color of the fruit that is used to make lemonade is", " lemon", " apple", " yellow", " red", " color", " shape"),
]
PROMPTS_V2 = [(f, SHOT[f] + p, *rest) for (f, p, *rest) in PROMPTS + EXTRA]


def tid(tok, s):
    return tok(s, add_special_tokens=False)["input_ids"]


class Readout:
    def __init__(self, m, ctx, ans_a, ans_b):
        self.m, self.c = m, ctx
        self.w = (m.W_U[ans_b].float() - m.W_U[ans_a].float())

    def f(self, h):  # h [1,T,d] at layer l
        out, _, _ = run_from(self.m, self.c, h, self.c.l + 1)
        x = self.m.final_norm(out[:, -1])
        return (x[0] * self.w).sum()

    def val(self, h):
        with torch.no_grad():
            return self.f(h).item()


def swap_delta(A: torch.Tensor, h_last: torch.Tensor) -> torch.Tensor:
    a = A @ h_last
    return torch.linalg.pinv(A) @ (a.flip(0) - a)


def lens_rank(m, lens, ctx, token, layers):
    """Rank of `token` in the J-lens readout at the last position (full lens: J, final norm, W_U)."""
    out = {}
    for L in layers:
        z = ctx.h_all[L][0, -1] @ lens.jacobians[L].cuda().T
        x = m.final_norm(z[None])[0]
        sc = torch.cat([m.W_U[i:i + 65536].float() @ x for i in range(0, m.W_U.shape[0], 65536)])
        out[L] = int((sc > sc[token]).sum().item()) + 1
    return out


def main(model_name="Qwen/Qwen3.5-0.8B", tag="08b", layers_swap=(8, 10, 12, 14, 16), alphas=(0.5, 1.0, 2.0), prompts=None):
    prompts = PROMPTS if prompts is None else prompts
    t0 = time.time()
    m = load_model(model_name)
    patch_deltanet(m)
    lens = load_lens(model_name)
    log(f"[swaps {tag}] loaded; {gpu_mem()}")
    rows, attr = [], []
    for fam, prompt, X, Y, ansA, ansB, rel, rel2 in prompts:
        toks = {k: tid(m.tok, s) for k, s in dict(X=X, Y=Y, A=ansA, B=ansB, R=rel, R2=rel2).items()}
        bad = [k for k, v in toks.items() if len(v) != 1]
        ids = torch.tensor(tid(m.tok, prompt))
        # model answer check
        c_full = Ctx(m, ids, layers_swap[0])
        with torch.no_grad():
            xf = m.final_norm(c_full.h_all[m.n_layers - 1][:, -1])
            logits = torch.cat([m.W_U[i:i + 65536].float() @ xf[0] for i in range(0, m.W_U.shape[0], 65536)])
        top1 = m.tok.decode([int(logits.argmax())])
        A0, B0 = toks["A"][0], toks["B"][0]
        rankA = int((logits > logits[A0]).sum()) + 1
        lr = lens_rank(m, lens, c_full, toks["X"][0], range(4, 23)) if not bad else {}
        best_layer = min(lr, key=lr.get) if lr else None
        rec = {"family": fam, "prompt": prompt, "X": X, "Y": Y, "ansA": ansA, "ansB": ansB, "multi_token": bad,
               "top1": top1, "rank_ansA": rankA, "A_beats_B": bool(logits[A0] > logits[B0]),
               "lens_rank_X": lr, "best_lens_rank_X": lr.get(best_layer) if lr else None, "best_lens_layer": best_layer}
        ok_answer = (not bad) and rankA == 1
        rec["lenient_ok"] = (not bad) and rankA <= 5 and rec["A_beats_B"]
        ok_lens = bool(lr) and min(lr[L] for L in range(8, 19)) <= 100
        rec["kept"] = ok_answer and ok_lens
        attr.append(rec)
        log(f"[swaps] {prompt[:60]!r}: top1={top1!r} rankA={rankA} lensrank({X})best={rec['best_lens_rank_X']}@L{best_layer} kept={rec['kept']}")
        if not rec["lenient_ok"]:
            continue
        # run swaps on all answer-correct prompts (lens filter reported separately)
        for L in layers_swap:
            c = Ctx(m, ids, L)
            R = Readout(m, c, A0, B0)
            h0 = c.h
            JL = lens.jacobians[L].cuda()
            UX = m.u([toks["X"][0], toks["Y"][0]]) @ JL
            UR = m.u([toks["R"][0], toks["R2"][0]]) @ JL
            dA = swap_delta(UX, h0[0, -1])
            dB = swap_delta(UR, h0[0, -1])
            f0 = R.val(h0)
            for alpha in alphas:
                DA = torch.zeros_like(h0); DA[0, -1] = alpha * dA
                DB = torch.zeros_like(h0); DB[0, -1] = alpha * dB
                _, g, HDA = grad_hvp_fn(R.f, h0, DA)
                _, _, HDB = grad_hvp_fn(R.f, h0, DB)
                fA, fB, fAB = R.val(h0 + DA), R.val(h0 + DB), R.val(h0 + DA + DB)
                first = (g * DA).sum().item()
                quad = 0.5 * (DA * HDA).sum().item()
                # lens-based first order: (g_norm * w) . J (Delta) / rms(h_final)
                hf = c.h_all[m.n_layers - 1][0, -1]
                rinv = torch.rsqrt(hf.pow(2).mean() + m.final_norm.eps)
                lens1 = ((m.gain * R.w) @ (JL @ DA[0, -1]) * rinv).item()
                inter = fAB - fA - fB + f0
                pred_int = (DA * HDB).sum().item()
                rows.append({"prompt": prompt, "family": fam, "layer": L, "alpha": alpha, "kept": rec["kept"], "strict": ok_answer,
                             "f0": f0, "true_dA": fA - f0, "first": first, "second": first + quad, "lens_first": lens1,
                             "true_dB": fB - f0, "first_B": (g * DB).sum().item(),
                             "second_B": (g * DB).sum().item() + 0.5 * (DB * HDB).sum().item(),
                             "interaction": inter, "pred_interaction": pred_int,
                             "norm_dA": float(DA.norm()), "flips": bool(f0 < 0 and f0 + (fA - f0) > 0), "f0_neg": bool(f0 < 0), "norm_h": float(h0[0, -1].norm())})
        del c
    with open(OUT / f"swaps_{tag}.json", "w", encoding="utf-8") as f:
        json.dump({"attrition": attr, "rows": rows}, f, indent=1, ensure_ascii=False)
    log(f"[swaps {tag}] done {time.time()-t0:.0f}s, {len(rows)} rows; {gpu_mem()}")


if __name__ == "__main__":
    import sys
    v = sys.argv[1] if len(sys.argv) > 1 else "v2"
    if v == "v1":
        main(tag="08b_v1")
    elif v == "big":
        main(tag="08b_big", prompts=PROMPTS_V2, layers_swap=(12, 14, 16, 18, 20), alphas=(1.0, 2.0, 4.0, 8.0))
    else:
        main(tag="08b", prompts=PROMPTS_V2)
