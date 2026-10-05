"""Summaries of out/swaps_*.json and out/ladder_*.json -> out/summary_*.md (printed too)."""
import json
import sys

import numpy as np

from common import OUT


def swaps(tag):
    d = json.load(open(OUT / f"swaps_{tag}.json", encoding="utf-8"))
    att = d["attrition"]
    lines = [f"## Swaps ({tag})", "", f"prompts: {len(att)}; single-token OK: {sum(not a['multi_token'] for a in att)}; "
             f"model top-1 = answer: {sum(a['rank_ansA']==1 for a in att)}; also lens rank<=100 for intermediate at some L8-18 (kept): {sum(a['kept'] for a in att)}", ""]
    lines.append("| prompt | top1 | rank ans | best lens rank X (layer) | kept |")
    lines.append("|---|---|---|---|---|")
    for a in att:
        lines.append(f"| {a['prompt'][:58]} | {a['top1']!r} | {a['rank_ansA']} | {a['best_lens_rank_X']} (L{a['best_lens_layer']}) | {a['kept']} |")
    rows = d["rows"]
    lines += ["", "### T1: share of first-order error removed by the 2nd-order term (intermediate swap)", "",
              "| subset | layer | alpha | n | median true dA | median |e1| | median |e2| | agg 1-sum|e2|/sum|e1| | median per-prompt share | corr(true,1st) | corr(true,2nd) | corr(true,lens1st) | median |D|/|h| | answer flips |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    t2 = ["", "### T2: interaction (double swap: intermediate + relation) vs D_A^T H D_B", "",
          "| subset | layer | alpha | n | median |I| | median |I - pred| | agg 1-sum|I-pred|/sum|I| | corr(I,pred) | sign agree |",
          "|---|---|---|---|---|---|---|---|---|"]
    for subset in ("all-correct", "kept"):
        R = [r for r in rows if subset == "all-correct" or r["kept"]]
        for L in sorted({r["layer"] for r in R}):
            for al in sorted({r["alpha"] for r in R}):
                S = [r for r in R if r["layer"] == L and r["alpha"] == al]
                if len(S) < 2:
                    continue
                tr = np.array([r["true_dA"] for r in S]); f1 = np.array([r["first"] for r in S])
                f2 = np.array([r["second"] for r in S]); fl = np.array([r["lens_first"] for r in S])
                e1, e2 = np.abs(tr - f1), np.abs(tr - f2)
                share = 1 - e2 / np.maximum(e1, 1e-12)
                c = lambda a, b: np.corrcoef(a, b)[0, 1] if np.std(a) > 0 and np.std(b) > 0 else float("nan")
                lines.append(f"| {subset} | {L} | {al} | {len(S)} | {np.median(tr):.3f} | {np.median(e1):.3f} | {np.median(e2):.3f} | "
                             f"{1 - e2.sum()/e1.sum():.2f} | {np.median(share):.2f} | {c(tr,f1):.2f} | {c(tr,f2):.2f} | {c(tr,fl):.2f} | "
                             f"{np.median([r['norm_dA']/r['norm_h'] for r in S]):.3f} | {sum(r.get('flips', False) for r in S)}/{len(S)} |")
                I = np.array([r["interaction"] for r in S]); P = np.array([r["pred_interaction"] for r in S])
                t2.append(f"| {subset} | {L} | {al} | {len(S)} | {np.median(np.abs(I)):.3f} | {np.median(np.abs(I-P)):.3f} | "
                          f"{1 - np.abs(I-P).sum()/np.abs(I).sum():.2f} | {c(I,P):.2f} | {np.mean(np.sign(I)==np.sign(P)):.2f} |")
    return "\n".join(lines + t2)


def ladder(tag):
    d = json.load(open(OUT / f"ladder_{tag}.json"))
    cfg = d["config"]
    lines = [f"## Ladder ({tag}): l={cfg['l']}, N={cfg['n_ctx']} contexts x T={cfg['T']}, k={cfg['k']}, Hutchinson probes={cfg['probes']}",
             f"Jbar (forward-AD, {len(cfg['layers'])} layers) {cfg['t_Jbar_s']:.0f}s; forward stats {cfg['t_stats_s']:.1f}s", "",
             "| token | L0 top-3 eigvals | L0 eig time (HVPs) | top-10 energy frac | rung | ov top-5 | ov top-10 | ov top-20 | cosF (+-se) | explained (unscaled) | top-1 Rayleigh / L0 |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    agg = {}
    for t, r in d["tokens"].items():
        L0 = r["L0"]
        first = True
        for n in ("L2", "L2_noNormCurv", "L3", "L3_radialProj"):
            x = r[n]
            agg.setdefault(n, []).append((x["overlap_top10"], x["cosF"], x["explained_unscaled"], x["overlap_top5"]))
            head = (f"| {t!r} | {[round(v,2) for v in L0['eigvals'][:3]]} | {L0['time_s']:.0f}s ({L0['n_hvp']}) | {L0['top10_energy_frac']:.2f} "
                    if first else "| | | | ")
            lines.append(head + f"| {n} | {x['overlap_top5']:.2f} | {x['overlap_top10']:.2f} | {x['overlap_top20']:.2f} | {x['cosF']:.2f} +- {x['cosF_se']:.2f} | "
                         f"{x['explained_unscaled']:.2f} | {x['rayleigh_on_L0_top1']/L0['eigvals'][0]:.2f} |")
            first = False
    lines += ["", "| rung | mean ov top-5 | mean ov top-10 | min ov top-10 | mean cosF | mean explained (unscaled) |", "|---|---|---|---|---|---|"]
    for n, v in agg.items():
        v = np.array(v)
        lines.append(f"| {n} | {v[:,3].mean():.2f} | {v[:,0].mean():.2f} | {v[:,0].min():.2f} | {v[:,1].mean():.2f} | {v[:,2].mean():.2f} |")
    lines.append(f"\n(random top-10 subspace overlap baseline = 10/1024 = 0.010)")
    return "\n".join(lines)


if __name__ == "__main__":
    tag = sys.argv[2] if len(sys.argv) > 2 else "08b"
    s = swaps(tag) if sys.argv[1] == "swaps" else ladder(tag)
    print(s)
    open(OUT / f"summary_{sys.argv[1]}_{tag}.md", "w", encoding="utf-8").write(s + "\n")
