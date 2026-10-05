"""Does the model follow the in-context 'lie iff pirate AND animal' rule? (behaviour only, no grads)

    python gate_behavior.py Qwen/Qwen3.5-0.8B [seeds]

Per cell (A, B): mean L = logit(false) - logit(true) and the rate at which the model's choice
between ' Yes'/' No' follows the rule. A usable AND gate needs rule-following >= 0.8 in all four
cells (in particular lying in (1,1) and not lying elsewhere).
"""

from __future__ import annotations

import json
import sys
import time

import torch

from common import OUT, log
from gate_prompts import build


def main(name: str, seeds=(0, 1, 2), chat=False, ndemo=2):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(name)
    hf = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map="cuda", low_cpu_mem_usage=True)
    hf.eval()
    yes, no = (tok(s, add_special_tokens=False)["input_ids"] for s in (" Yes", " No"))
    assert len(yes) == 1 and len(no) == 1, (yes, no)
    yes, no = yes[0], no[0]
    rows = []
    for seed in seeds:
        _, qs = build(seed, n_demo_per_cell=ndemo, chat_tok=tok if chat else None)
        for q in qs:
            ids = tok(q["text"], add_special_tokens=False, return_tensors="pt")["input_ids"].cuda()
            with torch.no_grad():
                lg = hf(ids).logits[0, -1].float()
            lt, lf = (lg[yes], lg[no]) if q["truth"] == "Yes" else (lg[no], lg[yes])
            top = tok.decode([int(lg.argmax())])
            L = float(lf - lt)
            follows = (L > 0) == q["lie"]
            rows.append({**{k: v for k, v in q.items() if k != "text"}, "seed": seed, "L": L,
                         "follows": bool(follows), "top1": top, "T": ids.shape[1]})
    out = {}
    for A in (0, 1):
        for B in (0, 1):
            r = [x for x in rows if x["A"] == A and x["B"] == B]
            out[f"{A}{B}"] = {"n": len(r), "mean_L": sum(x["L"] for x in r) / len(r),
                              "follow_rate": sum(x["follows"] for x in r) / len(r),
                              "top1_yes_no": sum(x["top1"] in (" Yes", " No") for x in r) / len(r)}
    tag = name.split("/")[-1] + ("_chat" if chat else "") + f"_d{ndemo}"
    with open(OUT / f"gate_behavior_{tag}.json", "w", encoding="utf-8") as f:
        json.dump({"cells": out, "rows": rows}, f, indent=1)
    log(f"[gate-behavior {tag}] {time.time()-t0:.0f}s " + " | ".join(
        f"AB={k}: L={v['mean_L']:+.2f} follow={v['follow_rate']:.2f} yn={v['top1_yes_no']:.2f}" for k, v in out.items()))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3.5-0.8B",
         chat="chat" in sys.argv[2:], ndemo=int(next((a[1:] for a in sys.argv[2:] if a.startswith("d")), 2)))
