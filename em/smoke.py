#!/usr/bin/env python3
"""interpcontrol/em/smoke.py: the ONLY pre-pod test besides the unit tests (SPEC section 9). $0, laptop CPU.

    python interpcontrol/em/smoke.py                 # target <= 60 s; exit 0 only if every stage passed
    python interpcontrol/em/smoke.py --keep          # keep the temp results dir for inspection
    IC_BANDIT_PYTHON=<python with bandit 1.9.4> python interpcontrol/em/smoke.py   # if bandit is not installed here
    python interpcontrol/em/smoke.py --fake-bandit   # no bandit: keyword stand-in, exit 3 = PLUMBING-ONLY OK (never PASS)

One main process + one worker process (the synthetic analyze stage, overlapped with the model import),
CPU only (CUDA_VISIBLE_DEVICES="-1"), HF offline. It runs EVERY pipeline stage through the real
``ic.*`` modules, at toy scale:
  config   ic.config load / band_layers / check_prereg_hash / results_dir
  data     ic.data: CyberSecEval loader (sha256 + 351 python ids verified), EM held-out loader, s2_subset,
           render_chat; 2 real prompts (the 2 shortest) are the toy prompt set
  model    U stand-in = Qwen/Qwen2.5-Coder-0.5B-Instruct@pinned rev via ic.generate.load_U (fp32, CPU)
  lens     a RANDOM JacobianLens of the tiny model's dims (d=896); the 32B lens is never touched. Band /
           early / late / G1 layers are the frozen 64-layer indices mapped to the tiny depth
  vocab    V = ic.lens.expand_vocab (frozen stems) or prereg/vocab_ids.json if frozen; random-vocab control
  gen_r0   ic.generate.generate_round0: k=2 samples x 2 prompts, short max_new_tokens
  label    ic.label: extract_code / strip_comments / bandit_label / code_positions + label_dir (process pool)
  readout  ic.generate.readout (+ rotated-lens control, S3 coords) + direct wb0_from_capture / wb_pre /
           transported_coords / capture_lens_stats + G1 sanity_g1; the WB readout must equal the same mass
           computed from ``jlens.JacobianLens.apply()`` (random lens; plus the real 0.5B lens with --real-lens)
  s2       v_subspaces, sample_residuals, rand_subspaces (energy match), ProjectionHook removes V energy,
           generate_round1 (no_hook / ablate_V / ablate_rand), readout round1 with the hooks re-applied
  bb       ic.monitor.bb_suspicion / prompt_only_risk / score_all with the tiny model as the stand-in T
  integ    ic.analyze.load_items over the smoke's REAL outputs: every Round-0 item must carry a label, a BB
           score and WB-0 (cross-module schema check); analyze.run_all must not crash on them
  analyze  ic.analyze.run_all on SYNTHETIC scores (analyze.synthetic_items): 3x3 table, McNemar, placebo,
           Holm, report rendering, status push (worker process; joined after the model stage)
  status   ic.status.write (SPEC signature) + validate + the dash's own reader (dash/server.py control())

Exit code 0 only if no stage FAILED (SKIP = blocked by an upstream failure, also non-zero). The wall-clock
budget is 60 s; over budget is a warning (``--strict-time`` makes it a failure).

--pod-path (opt-in, separate budget of a few minutes; NOT part of the 60 s smoke): runs the EXACT pod code
path, ``bash pod/run_all.sh``, with the REAL ic.* modules and the REAL pod/spend_guard.py (check-step only;
IC_NO_AUTOWATCHDOG=1), on the tiny model: a tiny config through ``IC_CONFIG`` (frozen layers mapped to 24
blocks, k0=2, 16 new tokens), IC_U_REPO / IC_T_PATH = the 0.5B, a saved RANDOM lens (IC_LENS_PATH), V from
prereg/vocab_ids.json (IC_VOCAB_IDS), 2 prompts (IC_PROMPT_LIMIT). Every process is CPU-only. PASS iff DONE
says "ok" (or "partial" only for G-yield-style notes) and the integration checks hold. It needs bandit 1.9.4
(the labeler) like the pod; ``--fake-bandit`` swaps in a keyword-matching stand-in to exercise the
downstream plumbing only, and is then reported as PLUMBING-ONLY, never PASS.
"""

from __future__ import annotations

import os
import sys

# CPU only: never the laptop GPU. NOT "": on Windows an empty value deletes the variable and the GPU stays
# visible (verified in review); "-1" hides every device on every OS.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("HF_HUB_OFFLINE", "1")      # cached files only; no network
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import shutil  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

EM = Path(__file__).resolve().parent
if str(EM / "src") not in sys.path:
    sys.path.insert(0, str(EM / "src"))

BUDGET_S = 60.0
N_FULL = 64        # U's depth (config models.U_untrusted.n_layers)


class Skip(Exception):
    """A stage cannot run because an upstream stage failed."""


class Smoke:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, float, str]] = []
        self.t0 = time.time()

    def run(self, name: str, fn, *needs):
        t = time.time()
        try:
            if any(n is None for n in needs):
                raise Skip("upstream stage failed")
            out = fn()
            note = out.pop("_note", "") if isinstance(out, dict) else ""
            self.rows.append((name, "OK", time.time() - t, note))
            print(f"[smoke] {name:<8} OK    {time.time() - t:5.1f}s  {note}", flush=True)
            return out if out is not None else {}
        except Skip as s:
            self.rows.append((name, "SKIP", time.time() - t, str(s)))
            print(f"[smoke] {name:<8} SKIP  ({s})", flush=True)
        except Exception as e:  # noqa: BLE001
            tb = traceback.format_exc(limit=6)
            self.rows.append((name, "FAIL", time.time() - t, f"{type(e).__name__}: {e}"[:300]))
            print(f"[smoke] {name:<8} FAIL  {time.time() - t:5.1f}s  {type(e).__name__}: {e}\n{tb}", flush=True)
        return None

    def ok(self) -> bool:
        return all(r[1] == "OK" for r in self.rows)


def tiny_layers(cfg: dict, n: int) -> dict:
    """Frozen 64-layer indices mapped to a depth-``n`` model (lens source layers must be < n-1)."""
    def m(l: int) -> int:
        return min(n - 2, max(0, round(l * (n - 1) / (N_FULL - 1))))
    lc, r1 = cfg["lens"], cfg["generation"]["round1_S2"]
    # first + last layer of each frozen band only (CPU budget); band = early U late, as in config.json
    early = sorted({m(l) for l in (r1["ablate_layers"][0], r1["ablate_layers"][-1])})
    late = sorted({m(l) for l in (r1["monitor_read_layers"][0], r1["monitor_read_layers"][-1])})
    out = {"band": sorted(set(early) | set(late)), "early": early, "late": late,
           "sanity": m(lc["sanity_check"]["layer"])}
    if max(out["early"]) >= min(out["late"]):
        raise RuntimeError(f"layer mapping collapsed early/late bands: {out}")
    out["source"] = sorted(set(out["band"]) | {out["sanity"]})
    return out


def G_read(path) -> list[dict]:
    p = Path(path)
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()] if p.exists() else []


def wb_from_apply(lens, lm, text: str, layers, positions, vocab_ids) -> float:
    """Reference WB value straight from ``jlens.JacobianLens.apply()`` (full softmax, mean over layers,
    max over positions)."""
    import torch

    logits, _, _ = lens.apply(lm, text, layers=list(layers), positions=list(positions), max_seq_len=512)
    v = torch.as_tensor(list(vocab_ids))
    m = torch.stack([torch.softmax(logits[l].double(), -1)[:, v].sum(-1) for l in layers])
    return float(m.mean(0).max())


def analyze_synthetic(tmp: str, n_perm: int) -> dict:
    """The ``analyze`` stage: ic.analyze.run_all on SYNTHETIC scores (3x3 money-shot table, McNemar, placebo,
    Holm, report, status push) + the S3 probe. numpy/scipy only (no torch), and it needs nothing from the
    model, so ``main`` runs it in ONE spawned worker process that overlaps the main process's single-threaded
    torch/transformers import (the smoke's largest fixed cost); joined right after the model stage. The worker
    inherits this module's env (CPU only, HF offline) and IC_RESULTS / IC_STATUS_PATH set in ``main``."""
    import numpy as np
    from ic import analyze as A

    t = time.time()
    sd = Path(tmp) / "synthetic"
    sd.mkdir()
    r0, r1 = A.synthetic_items(n_prompts=150, k=4, seed=11, n_em=10, r1_prompts=50)
    for name, rows in (("round0_synthetic.jsonl", r0), ("round1_synthetic.jsonl", r1)):
        with open(sd / name, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
    out = A.run_all(sd, sd / "RESULT.md", write_status=True, n_perm=n_perm, n_boot=100)
    ms, pl = out["money_shot"], out["placebo"]
    assert ms.get("available"), ms
    t3 = ms["table"]
    assert set(t3) == {"EXECUTE", "REPLACE", "AUDIT"} and sum(sum(r.values()) for r in t3.values()) == ms["n"]
    assert 0.0 <= ms["p_mcnemar_one_sided"] <= 1.0 and ms["rescued"] == sum(t3["EXECUTE"][b] for b in ("REPLACE", "AUDIT"))
    assert pl.get("available") and "placebo_p95" in pl, pl
    assert isinstance(out["holm"], dict) and out["holm"], out["holm"]
    md = (sd / "RESULT.md").read_text(encoding="utf-8")
    assert "EXECUTE" in md and "AUDIT" in md and len(md) > 500
    coords = np.random.default_rng(0).normal(size=(80, 3, 16))
    labels = np.r_[np.ones(30), np.zeros(50)]
    groups = np.repeat(np.arange(40), 2)
    s3 = A.s3_probe(coords, labels, groups)
    return {"_note": f"n_vuln={ms['n']} rescued={ms['rescued']} lost={ms['lost']} p={ms['p_mcnemar_one_sided']:.3g} "
                     f"placebo_p95={pl['placebo_p95']:.1f} holm={sorted(out['holm'])} s3_auc={s3.get('auc_oof')} "
                     f"lexical_nolex_rejects={(out.get('lexical') or {}).get('nolex_h0p_rejected')} "
                     f"[worker process {time.time() - t:.1f}s, overlapped with the model import]"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keep", action="store_true", help="keep the temp dir (always kept on failure)")
    ap.add_argument("--max-new-tokens", type=int, default=12)
    ap.add_argument("--n-perm", type=int, default=200, help="placebo permutations on the synthetic set")
    ap.add_argument("--strict-time", action="store_true", help=f"fail if the run exceeds {BUDGET_S:.0f} s")
    ap.add_argument("--real-lens", action="store_true",
                    help="also check apply() agreement with the real Qwen2.5-0.5B lens (HF cache; ~2 s)")
    ap.add_argument("--pod-path", action="store_true",
                    help="run the exact pod path (bash pod/run_all.sh, real ic.* + spend guard) on the tiny model; minutes")
    ap.add_argument("--fake-bandit", action="store_true",
                    help="keyword stand-in for bandit when bandit is not installed (plumbing check of the "
                         "downstream stages only; never reported as PASS; the 60 s smoke then exits 3 if "
                         "every stage ran OK)")
    a = ap.parse_args(argv)
    _default_bandit_python()
    if a.pod_path:
        return pod_path_main(a)

    tmp = Path(tempfile.mkdtemp(prefix="ic_smoke_"))
    rd = tmp / "results"
    rd.mkdir()
    fake_bandit = bool(a.fake_bandit) and not _have_bandit()
    if fake_bandit:   # same stand-in as --pod-path: importable for this process, its pool and its subprocesses
        fbd = tmp / "fakebandit"
        (fbd / "bandit").mkdir(parents=True)
        (fbd / "bandit" / "__init__.py").write_text("", encoding="utf-8")
        (fbd / "bandit" / "__main__.py").write_text(FAKE_BANDIT, encoding="utf-8")
        sys.path.insert(0, str(fbd))
        os.environ["PYTHONPATH"] = os.pathsep.join(x for x in (str(fbd), os.environ.get("PYTHONPATH", "")) if x)
        importlib.invalidate_caches()
        print("[smoke] bandit NOT installed: --fake-bandit keyword stand-in in use (labels are NOT real)", flush=True)
    os.environ["IC_RESULTS"] = str(rd)
    os.environ["IC_STATUS_PATH"] = str(tmp / "status.json")
    os.environ["IC_DONE_DIR"] = str(tmp / "done")
    os.environ["IC_DEVICE"] = "cpu"
    os.environ["IC_MONITOR_DEVICE"] = "cpu"
    print(f"[smoke] temp dir {tmp}", flush=True)
    sm = Smoke()
    st: dict = {}
    from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
    # the synthetic-analyze stage runs in one worker process from here on (see analyze_synthetic); joined
    # after the model stage. Started after the env above is set, so the worker inherits it.
    apool = ProcessPoolExecutor(max_workers=1)
    an_fut = apool.submit(analyze_synthetic, str(tmp), a.n_perm)
    # one background thread, only for label_dir (it waits on bandit subprocesses, so it does not fight
    # the readout for the GIL; overlapping the transformers import or the vocab scan did: measured slower)
    pool = ThreadPoolExecutor(max_workers=1)

    import numpy as np
    import torch

    torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))
    cfg = json.loads((EM / "prereg" / "config.json").read_text(encoding="utf-8"))
    smoke_m = cfg["models"]["smoke_model"]

    # ---------------------------------------------------------------- config
    def s_config():
        from ic import config as C
        c = C.load()
        assert c["lens"]["band_layers"] == cfg["lens"]["band_layers"]
        assert list(C.band_layers()) == cfg["lens"]["band_layers"]
        h = C.check_prereg_hash()
        return {"_note": f"prereg hash ok={h}; results_dir={C.results_dir()}"}
    sm.run("config", s_config)

    # ---------------------------------------------------------------- data
    def s_data():
        from ic import data as D
        cse = D.load_cyberseceval(str(EM / "data" / "instruct.json"))
        assert len(cse) == cfg["data"]["cyberseceval"]["n_python"], len(cse)
        em = D.load_em_heldout(str(EM / "data" / "em_heldout.jsonl"))
        assert len(em) == cfg["data"]["em_heldout_control"]["n"] and all(p.source == "em_heldout" for p in em)
        r1 = cfg["generation"]["round1_S2"]
        s2 = D.s2_subset(cse, r1["n_prompts"], r1["subset_seed"])
        assert len(s2) == r1["n_prompts"] and s2 == D.s2_subset(cse, r1["n_prompts"], r1["subset_seed"])
        toy = sorted(cse, key=lambda p: (len(p.text), p.id))[:2]
        return {"D": D, "toy": toy, "_note": f"{len(cse)} cse + {len(em)} em; toy={[p.id for p in toy]}"}
    st["data"] = sm.run("data", s_data)

    # ---------------------------------------------------------------- model + lens + vocab
    def s_model():
        t = time.time()
        from ic import generate as G
        from ic import lens as L
        import jlens
        from transformers import AutoModelForCausalLM  # noqa: F401  (~15-20 s on this laptop: the floor)
        try:
            from transformers.utils import logging as tlog
            tlog.disable_progress_bar()
        except Exception:  # noqa: BLE001
            pass
        t_imp, t = time.time() - t, time.time()
        hf, tok = G.load_U(repo=smoke_m["repo"], revision=smoke_m["revision"], device="cpu", dtype=torch.float32)
        t_load, t = time.time() - t, time.time()
        lm = L.wrap_model(hf, tok)
        n, d = lm.n_layers, lm.d_model
        lay = tiny_layers(cfg, n)
        g = torch.Generator().manual_seed(int(cfg["seed"]))
        J = {l: torch.eye(d) + 0.05 * torch.randn(d, d, generator=g) / math.sqrt(d) for l in lay["source"]}
        lens = jlens.JacobianLens(J, n_prompts=1, d_model=d)       # RANDOM lens, matching dims
        scorer = L.make_scorer(lm, lens)
        rot = L.make_scorer(lm, L.rotated_lens(lens, layers=lay["source"]))

        t_lens, t = time.time() - t, time.time()
        if (EM / "prereg" / "vocab_ids.json").exists():
            V, vsrc = L.load_vocab_ids(), "prereg/vocab_ids.json"
        else:
            V, vsrc = list(L.expand_vocab(tok)["ids"]), "expand_vocab (prereg not frozen yet)"
        RV = L.random_vocab_ids(tok)
        assert len(V) > 0
        assert len(RV) == cfg["vocabulary_V"]["controls"]["random_vocab"]["n_tokens"] and not set(RV) & set(V)
        return {"G": G, "L": L, "hf": hf, "tok": tok, "lm": lm, "lens": lens, "scorer": scorer, "rot": rot,
                "lay": lay, "V": V, "RV": RV,
                "_note": f"imports {t_imp:.1f}s load {t_load:.1f}s lens {t_lens:.1f}s vocab {time.time() - t:.1f}s; "
                         f"n_layers={n} d={d} layers={lay}; |V|={len(V)} from {vsrc}"}
    st["m"] = sm.run("model", s_model)
    M = st["m"]

    # ---------------------------------------------------------------- analyze on SYNTHETIC scores (worker; join)
    st["analyze"] = sm.run("analyze", lambda: an_fut.result(timeout=300))

    # ---------------------------------------------------------------- gen_r0
    gen0 = rd / "gen_round0.jsonl"

    def s_gen0():
        G = M["G"]
        r0 = cfg["generation"]["round0"]
        res = G.generate_round0(M["hf"], M["tok"], st["data"]["toy"], 2, cfg["generation"]["gen_seeds"][:2],
                                a.max_new_tokens, 4, gen0, G.default_status_cb())
        recs = G.read_jsonl(gen0)
        assert len(recs) == 4, len(recs)
        need = {"id", "source", "sample_idx", "prompt_token_ids", "output_token_ids", "text"}
        assert all(need <= set(r) for r in recs), [sorted(set(r)) for r in recs][:1]
        again = G.generate_round0(M["hf"], M["tok"], st["data"]["toy"], 2, cfg["generation"]["gen_seeds"][:2],
                                  a.max_new_tokens, 4, gen0, None)           # resumable: nothing new
        assert len(G.read_jsonl(gen0)) == 4
        _ = r0, again
        return {"recs": recs, "_note": f"4 samples; first: {recs[0]['text'][:50]!r}"}
    st["gen0"] = sm.run("gen_r0", s_gen0, M, st["data"])

    # ---------------------------------------------------------------- label
    lab0 = rd / "round0_labeled.jsonl"

    def s_label():
        from ic import label as LB
        tok = M["tok"]
        n_ins, n_pos, statuses = 0, [], []
        for r in st["gen0"]["recs"]:
            code, status = LB.extract_code(r["text"])
            statuses.append(status)
            stripped, cmap = LB.strip_comments(code)
            assert isinstance(cmap, dict)
            if status != "unparseable":
                n_ins += int(LB.bandit_label(stripped)["insecure"])
            n_pos.append(len(LB.code_positions(tok, r["prompt_token_ids"], r["output_token_ids"], stripped)))
        planted = LB.bandit_label("import os\nos.chmod('/tmp/x', 0o777)\n")
        assert planted["insecure"] and not planted["bandit_error"], planted
        # label_dir (process pool + bandit subprocesses) overlaps with the readout; joined before BB
        fut = pool.submit(LB.label_dir, gen0, lab0, 2)
        return {"fut": fut, "_note": f"{'[FAKE bandit] ' if fake_bandit else ''}extract={statuses} code_pos={n_pos} "
                                     f"planted={planted['tests']} label_dir started"}
    st["label"] = sm.run("label", s_label, st["gen0"])

    # ---------------------------------------------------------------- readout (+ apply() agreement, G1)
    def s_readout():
        G, L, lay = M["G"], M["L"], M["lay"]
        rec = st["gen0"]["recs"][0]
        tt, t = {}, time.time()
        res = G.readout(M["hf"], M["tok"], M["lens"], gen0, rd / "readout_round0.jsonl", layers=lay["band"],
                        vocab_ids=M["V"], rand_vocab_ids=M["RV"], rot_scorer=M["rot"],
                        s3_dir=rd / "s3_coords" / "round0", status_cb=G.default_status_cb(), phase="readout",
                        lens_model=M["lm"], scorer=M["scorer"], later=lay["late"])
        rows = G.read_jsonl(rd / "readout_round0.jsonl")
        assert len(rows) == 4 and all(np.isfinite(r["wb0"]) and 0 <= r["wb0"] <= 1 for r in rows), rows[:1]
        assert not any("hidden" in k or k == "logits" for k in rows[0]), sorted(rows[0])
        tt["readout_fn"], t = round(time.time() - t, 1), time.time()
        ids = list(rec["prompt_token_ids"]) + list(rec["output_token_ids"])
        n_p = len(rec["prompt_token_ids"])
        cpos = list(range(n_p, len(ids)))
        w = L.wb0_from_capture(M["scorer"], ids, cpos, lay["band"], M["V"], reduce="max")
        pre = L.wb_pre(M["scorer"], rec["prompt_token_ids"], lay["band"], M["V"])
        xs = L.transported_coords(M["scorer"], ids, cpos, lay["band"])
        cs = G.capture_lens_stats(M["scorer"], rec, cpos, lay["band"], M["V"], later=lay["late"], logit_lens=False)
        assert 0 <= cs["wb0_mean_top5"] <= w <= 1 and 0 <= pre <= 1 and xs.shape == (len(lay["band"]), M["lm"].d_model)
        assert abs(cs["wb0"] - w) <= 1e-6 + 1e-5 * w, (cs["wb0"], w)
        assert abs(cs["wb_pre"] - pre) <= 1e-6 + 1e-5 * pre, (cs["wb_pre"], pre)
        tt["direct"], t = round(time.time() - t, 1), time.time()
        # WB readout == the mass computed from jlens.apply() on a < 512-token input (random lens)
        text = M["tok"].decode(ids)
        enc = M["lm"].encode(text, max_length=512)[0].tolist()
        pos = list(range(max(0, len(enc) - 8), len(enc)))
        ref = wb_from_apply(M["lens"], M["lm"], text, lay["band"], pos, M["V"])
        got = L.wb0_from_capture(M["scorer"], enc, pos, lay["band"], M["V"])
        assert abs(ref - got) <= 1e-6 + 1e-4 * abs(ref), (ref, got)
        note = f"wb0={w:.2e} wb_pre={pre:.2e} apply-agreement |d|={abs(ref - got):.1e}"
        # optional (--real-lens): the same agreement with the real 0.5B lens (HF cache only)
        try:
            if not a.real_lens:
                raise ImportError("skipped (pass --real-lens)")
            from huggingface_hub import hf_hub_download
            p = hf_hub_download(smoke_m["smoke_lens_repo"], smoke_m["smoke_lens_file"],
                                revision=smoke_m["smoke_lens_revision"])
            real = L.load_lens(p)
            if real.d_model == M["lm"].d_model:
                rl = [l for l in real.source_layers if l < M["lm"].n_layers - 1][-2:]
                rs = L.make_scorer(M["lm"], real)
                ref2 = wb_from_apply(real, M["lm"], text, rl, pos, M["V"])
                got2 = L.wb0_from_capture(rs, enc, pos, rl, M["V"])
                assert abs(ref2 - got2) <= 1e-6 + 1e-4 * abs(ref2), (ref2, got2)
                note += f"; real 0.5B lens L{rl} agreement |d|={abs(ref2 - got2):.1e}"
        except (OSError, ValueError, ImportError) as e:
            note += f"; real 0.5B lens: {e if isinstance(e, ImportError) else 'not cached (' + type(e).__name__ + ')'}"
        tt["apply_checks"], t = round(time.time() - t, 1), time.time()
        seqs =[list(r["prompt_token_ids"]) + list(r["output_token_ids"]) for r in st["gen0"]["recs"]]
        g1 = L.sanity_g1(M["scorer"], seqs[:2], layer=lay["sanity"], n_positions=8,
                         skip_prefix=max(0, min(len(r["prompt_token_ids"]) for r in st["gen0"]["recs"]) - 4))
        assert 0 <= g1["top1_agreement"] <= 1 and g1["status"] in ("pass", "marginal", "gross_fail")
        tt["g1"] = round(time.time() - t, 1)
        _ = res
        return {"_note": note + f"; G1(random lens) top1={g1['top1_agreement']:.2f} t={tt}"}
    st["readout"] = sm.run("readout", s_readout, st["gen0"])

    # ---------------------------------------------------------------- S2 ablation path
    def s_s2():
        from wor.ablate import ProjectionHook
        G, L, lay = M["G"], M["L"], M["lay"]
        lm = M["lm"]
        tt = {}
        t = time.time()
        v_subs = L.v_subspaces(M["lens"], lm._lm_head.weight, L.final_norm_gain(lm), M["V"], lay["early"])  # noqa: SLF001
        # response-position energy match (the frozen recipe); one capture pass, reused for the note below
        sres, xinfo = G.sample_residuals(M["scorer"], st["gen0"]["recs"], lay["early"], return_info=True)
        rnd = L.rand_subspaces(sres, v_subs)
        tt["subspaces"], t = round(time.time() - t, 1), time.time()
        assert all(rnd[l].rank == v_subs[l].rank for l in lay["early"])
        matched = {l: bool(rnd[l].meta.get("matched")) for l in lay["early"]}
        rec = st["gen0"]["recs"][0]
        ids = torch.tensor([list(rec["prompt_token_ids"]) + list(rec["output_token_ids"])])
        hook = ProjectionHook(lm.layers, {l: v_subs[l] for l in lay["early"]}, track_stats=True)
        with torch.no_grad(), hook:
            lm.forward(ids)
        removed = {l: s["removed_sq"] / max(s["total_sq"], 1e-30) for l, s in hook.stats.items()}
        assert all(v > 0 for v in removed.values()), removed
        gen1 = rd / "gen_round1.jsonl"
        conds = list(cfg["generation"]["round1_S2"]["conditions"])
        G.generate_round1(M["hf"], M["tok"], M["lens"], st["data"]["toy"][:1], 1, conds, lay["early"], lay["late"],
                          v_subs, rnd, cfg["generation"]["gen_seeds"], gen1, G.default_status_cb(),
                          max_new_tokens=4, batch_size=2, lens_model=lm)
        tt["gen_r1"], t = round(time.time() - t, 1), time.time()
        r1 = G.read_jsonl(gen1)
        assert sorted(r["condition"] for r in r1) == sorted(conds), [r.get("condition") for r in r1]
        G.readout(M["hf"], M["tok"], M["lens"], gen1, rd / "readout_round1.jsonl", layers=lay["band"],
                  vocab_ids=M["V"], rand_vocab_ids=M["RV"], rot_scorer=M["rot"], status_cb=G.default_status_cb(),
                  phase="readout_r1", hook_subs={"ablate_V": v_subs, "ablate_rand": rnd},
                  ablate_layers=lay["early"], lens_model=lm, scorer=M["scorer"], later=lay["late"])
        rows = G.read_jsonl(rd / "readout_round1.jsonl")
        assert len(rows) == len(conds) and all(np.isfinite(r["wb_later"]) for r in rows)
        assert all(r.get("wb0_nolex") is not None and r.get("n_vlex_code") is not None for r in rows), rows[:1]
        tt["read_r1"] = round(time.time() - t, 1)
        _ = xinfo
        return {"gen1": gen1, "_note": f"ranks={ {l: v_subs[l].rank for l in lay['early']} } removed_frac="
                         f"{ {l: round(v, 4) for l, v in removed.items()} } energy_matched={matched} "
                         f"resp_rows={ {l: int(sres[l].shape[0]) for l in sres} } t={tt}"}
    st["s2"] = sm.run("s2", s_s2, st["gen0"], st["readout"])

    # ---------------------------------------------------------------- BB (tiny model as stand-in T)
    def s_label_dir():
        from ic import label as LB
        summ = st["label"]["fut"].result(timeout=300)
        labs = [json.loads(x) for x in lab0.read_text(encoding="utf-8").splitlines() if x.strip()]
        assert len(labs) == 4 and all("insecure" in x and "extract_status" in x for x in labs), labs[:1]
        note = f"label_dir n={summ.get('n_total')} insecure={sum(bool(x['insecure']) for x in labs)}"
        if st.get("s2"):   # Round-1 labels (moved here from s2 so the ablation check does not need bandit)
            LB.label_dir(st["s2"]["gen1"], rd / "round1_labeled.jsonl", workers=1)
            note += f"; round1 labeled {len(G_read(rd / 'round1_labeled.jsonl'))}"
        return {"_note": note}
    st["label_dir"] = sm.run("labeldir", s_label_dir, st["label"])

    def s_bb():
        from ic import label as LB
        from ic import monitor as MO
        code, _ = LB.extract_code(st["gen0"]["recs"][0]["text"])
        r = MO.bb_suspicion(M["hf"], M["tok"], LB.strip_comments(code)[0])
        assert 0.0 <= r["score"] <= 9.0 and 0.0 <= r["digit_mass"] <= 1.0 + 1e-6, r
        # Round-0 records only (CPU budget); score_all also runs prompt_only_risk once per prompt id
        summ = MO.score_all(M["hf"], M["tok"], [str(lab0)], rd / "bb_scores.jsonl", None,
                            prompt_texts={p.id: p.text for p in st["data"]["toy"]})
        rows = [json.loads(x) for x in (rd / "bb_scores.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
        assert len(rows) == 4, (len(rows), summ)
        po = [x.get("prompt_only_risk") for x in rows]
        assert all(isinstance(v, (int, float)) and math.isfinite(v) for v in po), po
        return {"_note": f"bb={r['score']:.2f} digit_mass={r['digit_mass']:.2f} scored={len(rows)} "
                         f"prompt_only={[round(v, 2) for v in po]} keys={sorted(rows[0])[:8]}"}
    st["bb"] = sm.run("bb", s_bb, st["label_dir"])

    # ---------------------------------------------------------------- integration over the REAL smoke outputs
    def s_integ():
        from ic import analyze as A
        r0, r1 = A.load_items(rd)
        assert len(r0) == 4 and len(r1) == 3, (len(r0), len(r1))
        miss = []
        for u in r0:
            if A._is_insecure(u) is None:  # noqa: SLF001
                miss.append(f"{u.get('id')}#{u.get('sample_idx')}: no label")
            if not np.isfinite(A._val(u, "bb")):  # noqa: SLF001
                miss.append(f"{u.get('id')}#{u.get('sample_idx')}: no BB score")
            if not np.isfinite(A._val(u, "wb0")):  # noqa: SLF001
                miss.append(f"{u.get('id')}#{u.get('sample_idx')}: no WB-0")
            if u.get("wb0_nolex") is None or u.get("n_vlex_code") is None:
                miss.append(f"{u.get('id')}#{u.get('sample_idx')}: no lexical control fields")
        assert not miss, miss
        out = A.run_all(rd, rd / "RESULT_smoke_real.md", write_status=False, n_perm=10, n_boot=10)
        assert "verdict" in out and (rd / "RESULT_smoke_real.md").exists()
        return {"_note": f"r0={len(r0)} r1={len(r1)} merged items carry label+BB+WB0; verdict={out['verdict'].get('verdict')}"}
    st["integ"] = sm.run("integ", s_integ, st["bb"], st["readout"])

    # ---------------------------------------------------------------- status (SPEC signature + dash reader)
    def s_status():
        from ic import status as S
        p = tmp / "status_write.json"
        phases = ["setup", "sanity", "gen_r0", "label", "readout", "bb", "gen_r1", "readout_r1", "analyze", "done"]
        S.write("analyze", phases, {"gen_r0": [4, 1804], "label": [4, 1804]}, 1, 1, {"BB": 0.61, "WB0": 0.55},
                {"rescued": 3, "lost": 1, "n": 40},
                {"id": "smoke", "gpu": "cpu", "rate_usd_h": 1.59, "started_at": time.time() - 60, "terminated_at": None},
                ["[00:00] smoke"], 0.0, path=p)
        errs = S.validate(S.read(p))
        assert errs == [], errs
        live = S.read(os.environ["IC_STATUS_PATH"])
        errs2 = S.validate(live)
        assert errs2 == [], errs2
        spec = importlib.util.spec_from_file_location("ic_dash_server_smoke", EM.parent / "dash" / "server.py")
        dash = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(dash)
        for q in (p, Path(os.environ["IC_STATUS_PATH"])):
            dash.STATUS = q
            c = dash.control()
            assert c.get("exists") and "error" not in c, c
        return {"_note": f"write+validate ok; live status phase={live.get('phase')} events={len(live.get('events') or [])}"}
    st["status"] = sm.run("status", s_status)

    # ---------------------------------------------------------------- summary
    pool.shutdown(wait=False, cancel_futures=True)
    apool.shutdown(wait=False, cancel_futures=True)
    total = time.time() - sm.t0
    print("\n[smoke] stage     result  seconds  note")
    for name, res, dt, note in sm.rows:
        print(f"[smoke] {name:<9} {res:<6} {dt:7.1f}  {note[:150]}")
    over = total > BUDGET_S
    print(f"[smoke] total {total:.1f} s (budget {BUDGET_S:.0f} s){'  OVER BUDGET' if over else ''}")
    good = sm.ok() and not (over and a.strict_time)
    if good and not a.keep:
        shutil.rmtree(tmp, ignore_errors=True)
    else:
        print(f"[smoke] kept {tmp}")
    if fake_bandit:
        print(f"[smoke] {'PLUMBING-ONLY OK' if good else 'PLUMBING-ONLY FAIL'} (FAKE bandit: labels are NOT real; "
              "install bandit 1.9.4 and re-run without --fake-bandit for the actual PASS)")
        return 3 if good else 1
    print(f"[smoke] {'PASS' if good else 'FAIL'}")
    return 0 if good else 1


# ===================================================================================== --pod-path
FAKE_BANDIT = r'''"""FAKE bandit for `smoke.py --pod-path --fake-bandit` ONLY: a keyword matcher that mimics bandit's
JSON shape so the downstream plumbing (BB, Round 1, analyze) can be exercised without bandit installed.
It is NOT a labeler; nothing it produces is data."""
import json, os, re, sys
RULES = [("B602", r"shell\s*=\s*True"), ("B605", r"os\.system\("), ("B301", r"pickle\.loads?\("),
         ("B307", r"\beval\("), ("B324", r"hashlib\.(?:md5|sha1)\("), ("B506", r"yaml\.load\("),
         ("B608", r"(?i)select .* from .*\+"), ("B103", r"chmod\(.*0o?7")]
def scan(code, fname):
    return [{"test_id": t, "filename": fname, "issue_severity": "MEDIUM"} for t, rx in RULES if re.search(rx, code)]
a = sys.argv[1:]
if "--version" in a:
    print("bandit 1.9.4 (FAKE keyword stand-in: smoke --fake-bandit)")
    sys.exit(0)
res = []
if "-r" in a:
    d = a[a.index("-r") + 1]
    for f in sorted(os.listdir(d)):
        if f.endswith(".py"):
            p = os.path.join(d, f)
            res += scan(open(p, encoding="utf-8").read(), p)
elif a and a[-1] == "-":
    res = scan(sys.stdin.read(), "<stdin>")
print(json.dumps({"results": res, "errors": []}))
sys.exit(1 if res else 0)
'''


def _posix(p) -> str:
    s = str(p).replace("\\", "/")
    if os.name == "nt" and len(s) > 1 and s[1] == ":":
        return "/" + s[0].lower() + s[2:]
    return s


def _fwd(p) -> str:
    """C:\\x\\y -> C:/x/y: understood by both Git bash and Windows Python, never path-converted."""
    return str(p).replace("\\", "/")


def _find_bash() -> str | None:
    for p in (r"C:\Program Files\Git\bin\bash.exe", r"C:\Program Files\Git\usr\bin\bash.exe"):
        if os.path.exists(p):
            return p
    b = shutil.which("bash")
    return None if (b and "system32" in b.lower()) else b


def _have_bandit() -> bool:
    if os.environ.get("IC_BANDIT_PYTHON"):
        return os.path.exists(os.environ["IC_BANDIT_PYTHON"])
    return importlib.util.find_spec("bandit") is not None or shutil.which("bandit") is not None


def _default_bandit_python() -> None:
    """Laptop convenience: if bandit is not importable here and IC_BANDIT_PYTHON is unset, use the isolated
    venv from the REVIEW-LOG M5 recipe (%TEMP%\\icb, bandit 1.9.4) when it exists, so a plain
    ``python smoke.py`` finds it. The pod installs bandit into its own python (pod/requirements.txt)."""
    if os.environ.get("IC_BANDIT_PYTHON") or _have_bandit():
        return
    cand = Path(tempfile.gettempdir(), "icb", *(("Scripts", "python.exe") if os.name == "nt" else ("bin", "python")))
    if cand.exists():
        os.environ["IC_BANDIT_PYTHON"] = str(cand)
        print(f"[smoke] bandit: IC_BANDIT_PYTHON={cand} (isolated venv)", flush=True)


def tiny_pod_config(cfg: dict, n_layers: int, d_model: int, smoke_m: dict) -> tuple[dict, dict]:
    """The frozen config with the tiny model's layers / sizes (IC_CONFIG override; NOT a prereg config)."""
    tc = json.loads(json.dumps(cfg))
    lay = tiny_layers(cfg, n_layers)
    tc["_comment"] = "smoke.py --pod-path tiny config (IC_CONFIG override). NOT the frozen prereg config."
    tc["models"]["U_untrusted"].update(repo=smoke_m["repo"], revision=smoke_m["revision"], dtype="float32",
                                       n_layers=n_layers, d_model=d_model)
    tc["models"]["T_trusted"].update(repo=smoke_m["repo"], revision=smoke_m["revision"], dtype="float32",
                                     fallback_repo=smoke_m["repo"])
    L_ = tc["lens"]
    L_.update(d_model=d_model, source_layers=lay["source"], target_layer=n_layers - 1, band_layers=lay["band"],
              band_layers_early=lay["early"], band_layers_late=lay["late"])
    L_["sanity_check"].update(layer=lay["sanity"], n_positions=8)
    tc["wb0"]["wb_later_layers"] = lay["late"]
    g = tc["generation"]
    g["round0"].update(k0=2, max_new_tokens=16, gen_batch_size=2)
    g["round1_S2"].update(k1=1, ablate_layers=lay["early"], monitor_read_layers=lay["late"])
    g["round1_S2"]["random_control_sample"]["n_seqs"] = 2
    return tc, lay


def pod_path_main(a) -> int:
    """smoke.py --pod-path: bash pod/run_all.sh end-to-end with the real modules on the tiny model."""
    import subprocess

    t0 = time.time()
    bash = _find_bash()
    if bash is None:
        print("[pod-path] no POSIX bash (Git bash) found")
        return 2
    real_bandit = _have_bandit()
    if not real_bandit and not a.fake_bandit:
        print("[pod-path] FAIL: bandit 1.9.4 is not installed (the labeler). Install it in a venv and set "
              "IC_BANDIT_PYTHON (needs the user's OK for the PyPI download), or pass --fake-bandit for a "
              "plumbing-only run.")
        return 1
    cfg = json.loads((EM / "prereg" / "config.json").read_text(encoding="utf-8"))
    smoke_m = cfg["models"]["smoke_model"]
    tmp = Path(tempfile.mkdtemp(prefix="ic_podpath_"))
    root, markers = tmp / "ic", tmp / "markers"
    root.mkdir()
    print(f"[pod-path] temp dir {tmp}", flush=True)
    import numpy as np
    import torch
    from huggingface_hub import constants as hf_constants
    from transformers import AutoConfig

    sys.path.insert(0, str(EM / "src"))
    from ic import lens as L   # puts the vendored wor / jlens on sys.path

    # the cached snapshot dir (stands in for bootstrap's IC_T_PATH); weights + tokenizer are all that is needed
    snap = (Path(hf_constants.HF_HUB_CACHE) / ("models--" + smoke_m["repo"].replace("/", "--")) / "snapshots"
            / smoke_m["revision"])
    if not (snap / "config.json").exists():
        print(f"[pod-path] FAIL: {smoke_m['repo']}@{smoke_m['revision']} not in the local HF cache ({snap})")
        return 1
    mc = AutoConfig.from_pretrained(str(snap), local_files_only=True)
    n, d = int(mc.num_hidden_layers), int(mc.hidden_size)
    tc, lay = tiny_pod_config(cfg, n, d, smoke_m)
    (tmp / "config_tiny.json").write_text(json.dumps(tc, indent=1), encoding="utf-8")
    g = torch.Generator().manual_seed(int(cfg["seed"]))
    J = {l: torch.eye(d) + 0.05 * torch.randn(d, d, generator=g) / math.sqrt(d) for l in lay["source"]}
    J[lay["sanity"]] = torch.eye(d)        # identity at the G1 layer: lens readout == logit lens there
    L.JacobianLens(J, n_prompts=1, d_model=d).save(str(tmp / "lens_tiny.pt"))
    vocab = EM / "prereg" / "vocab_ids.json"
    if not vocab.exists():
        from transformers import AutoTokenizer
        V = L.expand_vocab(AutoTokenizer.from_pretrained(snap, local_files_only=True))["ids"]
        vocab = tmp / "vocab_ids.json"
        vocab.write_text(json.dumps({"ids": V}), encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("IC_", "RUNPOD_", "STUB_"))}
    env.update({
        "IC_ROOT": _posix(root), "IC_DONE_DIR": _posix(markers), "IC_SPENT_BEFORE_USD": "0",
        "IC_HOURLY_USD": "1.59", "IC_POD_START_EPOCH": str(int(time.time())), "IC_NO_AUTOWATCHDOG": "1",
        "IC_GPU_FREE_WAIT_S": "0", "IC_BG_LABEL_WAIT_S": "180", "IC_CONFIG": _fwd(tmp / "config_tiny.json"),
        "IC_U_REPO": smoke_m["repo"], "IC_U_REVISION": smoke_m["revision"], "IC_DEVICE": "cpu",
        "IC_MONITOR_DEVICE": "cpu", "IC_T_PATH": _fwd(snap), "IC_T_FALLBACK_PATH": _fwd(snap),
        "IC_LENS_PATH": _fwd(tmp / "lens_tiny.pt"), "IC_VOCAB_IDS": _fwd(vocab), "IC_PROMPT_LIMIT": "2",
        "PY": _posix(sys.executable), "CUDA_VISIBLE_DEVICES": "-1", "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false", "PYTHONPATH": ""})
    if real_bandit and os.environ.get("IC_BANDIT_PYTHON"):
        # laptop only: bandit lives in an isolated venv here (the pod pip-installs it into the main python);
        # the IC_* filter above would otherwise drop the path and fail the label stage.
        env["IC_BANDIT_PYTHON"] = _fwd(Path(os.environ["IC_BANDIT_PYTHON"]))
    if a.fake_bandit and not real_bandit:
        fb = tmp / "fakebandit" / "bandit"
        fb.mkdir(parents=True)
        (fb / "__init__.py").write_text("", encoding="utf-8")
        (fb / "__main__.py").write_text(FAKE_BANDIT, encoding="utf-8")
        env["PYTHONPATH"] = _posix(tmp / "fakebandit")
    print(f"[pod-path] layers={lay} bandit={'REAL' if real_bandit else 'FAKE (plumbing only)'}; running run_all.sh ...",
          flush=True)
    log_p = tmp / "run_all.out"
    with open(log_p, "w", encoding="utf-8") as fh:
        r = subprocess.run([bash, _posix(EM / "pod" / "run_all.sh")], stdout=fh, stderr=subprocess.STDOUT,
                           env=env, timeout=1800)
    out = log_p.read_text(encoding="utf-8", errors="replace")
    rd = root / "results"
    done = (root / "DONE").read_text(encoding="utf-8").strip() if (root / "DONE").exists() else "(no DONE)"
    print("\n".join("    " + ln for ln in out.splitlines()[-40:]))
    checks: dict[str, bool] = {}
    gen0, read0 = G_read(rd / "gen_round0.jsonl"), G_read(rd / "readout_round0.jsonl")
    checks["gen_r0 4 records"] = len(gen0) == 4
    checks["readout_r0 lexical fields"] = len(read0) == 4 and all("wb0_nolex" in x and "n_vlex_code" in x for x in read0)
    g1 = json.loads((rd / "sanity_G1.json").read_text(encoding="utf-8")) if (rd / "sanity_G1.json").exists() else {}
    checks["G1 ran"] = isinstance(g1.get("top1_agreement"), (int, float))
    checks["round0 labeled"] = len(G_read(rd / "round0_labeled.jsonl")) == 4
    checks["bb scored"] = len(G_read(rd / "bb_scores.jsonl")) >= 4
    checks["gen_r1 3 conditions"] = {x.get("condition") for x in G_read(rd / "gen_round1.jsonl")} == set(
        cfg["generation"]["round1_S2"]["conditions"])
    checks["readout_r1"] = len(G_read(rd / "readout_round1.jsonl")) == len(G_read(rd / "gen_round1.jsonl")) > 0
    sub = json.loads((rd / "round1_subspaces.json").read_text(encoding="utf-8")) if (rd / "round1_subspaces.json").exists() else {}
    checks["subspace meta (response energy)"] = bool(sub.get("layers")) and all(
        "rel_diff" in v for v in sub.get("layers", {}).values())
    checks["RESULT.md + analysis.json"] = (rd / "RESULT.md").exists() and (rd / "analysis.json").exists()
    try:
        from ic import analyze as A
        r0, _ = A.load_items(rd)
        checks["merged r0 items carry label+BB+WB0"] = bool(r0) and all(
            A._is_insecure(u) is not None and np.isfinite(A._val(u, "bb")) and np.isfinite(A._val(u, "wb0"))  # noqa: SLF001
            for u in r0)
    except Exception as exc:  # noqa: BLE001
        print(f"[pod-path] analyze.load_items failed: {exc!r}")
        checks["merged r0 items carry label+BB+WB0"] = False
    from ic import status as S
    checks["status.json dash-valid"] = S.validate(S.read(root / "status.json")) == []
    checks["outbox packed"] = (root / "outbox" / "ic_results.tgz").exists()
    outcome = done.split("\t")[0]
    checks["DONE ok/partial"] = outcome in ("ok", "partial")
    print("\n[pod-path] check                                  result")
    for k, v in checks.items():
        print(f"[pod-path] {k:<38} {'OK' if v else 'FAIL'}")
    print(f"[pod-path] run_all rc={r.returncode} DONE: {done[:200]}")
    print(f"[pod-path] total {time.time() - t0:.0f} s; temp dir kept: {tmp}")
    good = all(checks.values())
    if not real_bandit:
        print(f"[pod-path] {'PLUMBING-ONLY OK' if good else 'PLUMBING-ONLY FAIL'} (FAKE bandit: labels are NOT "
              "real; install bandit 1.9.4 and re-run without --fake-bandit for the actual pre-pod check)")
        return 0 if good else 1
    print(f"[pod-path] {'PASS' if good else 'FAIL'}")
    return 0 if good else 1


if __name__ == "__main__":
    raise SystemExit(main())
