"""U generation (Round 0 / Round 1) and teacher-forced lens readout, HF transformers only.

Stages (run from ``interpcontrol/em`` with ``PYTHONPATH=src``; outputs under
``results_dir()`` = ``$IC_RESULTS`` or ``./results``)::

    python -m ic.generate round0              -> gen_round0.jsonl
    python -m ic.generate readout round0      -> readout_round0.jsonl + s3_coords_round0.npz
    python -m ic.generate round1              -> gen_round1.jsonl (+ round1_subspaces.pt/.json)
    python -m ic.generate readout round1      -> readout_round1.jsonl
    python -m ic.generate u_all               -> sanity + all four above in ONE process (U loaded once),
                                                 per-stage done markers in $IC_DONE_DIR

Every JSONL is append-only and resumable: one line per (id, sample_idx[, condition]),
keys already present are skipped, a torn last line is truncated on restart. Batches are
built from the full frozen prompt order (never from the remaining items) and each batch
is seeded from (frozen seed of its sample index, batch index), so a restart regenerates
exactly the same batch composition and RNG stream (CPU bit-exact; GPU up to kernel
nondeterminism).

Record schema (gen_*.jsonl): ``{key, id, source, sample_idx, round, [condition],
seed, batch, prompt_token_ids, output_token_ids, text, stopped ("eos"|"length"), n_out,
cwe, explicit_api}``. ``output_token_ids`` ends with the EOS id when ``stopped=="eos"``.

Readout records (readout_*.jsonl): the reduced stats of ``lens.capture_lens_stats``
(no hidden states, no full-vocab logits) + ``key, id, source, sample_idx, round,
[condition], code_pos_fallback, s3_saved``. ``s3_coords`` (S3 features,
``[len(band), d_model]`` float16) are staged as one ``.npy`` per record (crash-safe resume)
and consolidated at the end of the stage into ``s3_coords_round0.npz`` with arrays
``coords [N, L, d], ids, sample_idx, keys, layers`` (the format analyze.py globs for);
the staging files are then removed. See :func:`load_s3`.

Defensive-research note: Round 1's ``ablate_V`` condition is a mechanistic ablation of
U's own residual stream (project the deception-vocabulary J-directions out of earlier
band layers) to test whether the WB monitor signal is load-bearing or evadable, with a
matched random-direction control. It is a monitor-robustness control in the AI-control
tradition, produces no attack artifact, and ships no capability.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ic import lens as L

log = logging.getLogger("ic.generate")

GEN_R0 = "gen_round0.jsonl"
GEN_R1 = "gen_round1.jsonl"
READ_R0 = "readout_round0.jsonl"
READ_R1 = "readout_round1.jsonl"
S3_DIR = "s3_coords"
SUBS_R1 = "round1_subspaces.pt"
TIMINGS = "stage_timings.json"
CONDITIONS = ("no_hook", "ablate_V", "ablate_rand")


def resid_sample_cfg() -> dict:
    """Frozen recipe for the residual sample behind the energy-matched random control
    (config ``generation.round1_S2.random_control_sample``)."""
    d = dict(L.cfg()["generation"]["round1_S2"].get("random_control_sample") or {})
    d.setdefault("n_seqs", 16)
    d.setdefault("min_rows_x_d_model", 2.0)
    d.setdefault("max_seqs", 64)
    d.setdefault("positions", "response")
    d.setdefault("drop_rows_norm_gt_x_median", 10.0)
    d.setdefault("match_rel_tol", 0.05)
    return d


# ------------------------------------------------------------------------- helpers
def _stamp(text: str) -> str:
    return f"[{time.strftime('%H:%M')}] {text}"


def default_status_cb() -> Callable | None:
    """``ic.status.progress(phase, done, total, events)`` when the module exists."""
    try:
        from ic import status as _status  # noqa: PLC0415

        return getattr(_status, "progress", None)
    except ImportError:
        return None


def _safe_cb(cb: Callable | None) -> Callable:
    def call(phase: str, done: int, total: int, events: list[str]) -> None:
        if cb is None:
            return
        try:
            cb(phase, int(done), int(total), list(events))
        except Exception as exc:  # noqa: BLE001  (status must never kill a paid stage)
            log.warning("status_cb failed: %r", exc)

    return call


def key_of(pid: Any, sample_idx: int, condition: str | None = None) -> str:
    return f"{pid}::{int(sample_idx)}" + (f"::{condition}" if condition else "")


def repair_jsonl(path: str | os.PathLike) -> int:
    """Truncate a torn trailing line (crash mid-write). Returns bytes dropped."""
    p = Path(path)
    if not p.exists():
        return 0
    data = p.read_bytes()
    if not data or data.endswith(b"\n"):
        return 0
    cut = data.rfind(b"\n") + 1
    with open(p, "r+b") as f:
        f.truncate(cut)
    log.warning("%s: dropped a torn last line (%d bytes)", p, len(data) - cut)
    return len(data) - cut


def read_jsonl(path: str | os.PathLike) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    out = []
    with open(p, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                out.append(json.loads(ln))
            except ValueError:
                log.warning("%s: skipping unparseable line", p)
    return out


def done_keys(path: str | os.PathLike) -> set[str]:
    return {r["key"] for r in read_jsonl(path) if "key" in r}


def append_jsonl(path: str | os.PathLike, rows: Sequence[dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def batch_seed(seed: int, batch_idx: int) -> int:
    """Per-batch torch seed derived from the frozen per-sample-index seed."""
    h = hashlib.sha256(f"{int(seed)}:{int(batch_idx)}".encode()).digest()
    return int.from_bytes(h[:4], "little")


def _merge_json(p: Path, upd: dict) -> None:
    """Atomic read-modify-write of a small JSON object file."""
    try:
        doc = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    except ValueError:
        doc = {}
    doc.update(upd)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, indent=1, default=str), encoding="utf-8")
    os.replace(tmp, p)


def _record_timing(stage: str, **vals: Any) -> None:
    _merge_json(L.results_dir() / TIMINGS, {stage: vals})


def _render_fn(render: Callable | None) -> Callable:
    return render or L._default_render  # noqa: SLF001  (ic.data.render_chat when present)


# --------------------------------------------------------------------------- model U
def load_U(repo: str | None = None, revision: str | None = None, device: str | None = None,
           dtype: torch.dtype | None = None) -> tuple[Any, Any]:
    """U in bf16 on CUDA at the pinned revision (``wor.modeling.load_model``:
    ``device_map`` + ``low_cpu_mem_usage``). Env overrides for tests/smoke only:
    ``IC_U_REPO``, ``IC_U_REVISION``, ``IC_DEVICE``."""
    U = L.cfg()["models"]["U_untrusted"]
    repo = repo or os.environ.get("IC_U_REPO") or U["repo"]
    if revision is None:
        revision = os.environ.get("IC_U_REVISION") or (U["revision"] if repo == U["repo"] else None)
    device = device or os.environ.get("IC_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = dtype or getattr(torch, U["dtype"])
    t0 = time.time()
    hf, tok = L._modeling.load_model(repo, dtype=dtype, device=device, revision=revision)  # noqa: SLF001
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    log.info("loaded %s@%s on %s (%s) in %.0fs", repo, revision, device, dtype, time.time() - t0)
    return hf, tok


def gen_kwargs() -> dict:
    g = L.cfg()["models"]["U_untrusted"]["generation_config"]
    return {"do_sample": True, "temperature": float(g["temperature"]), "top_p": float(g["top_p"]),
            "top_k": int(g["top_k"]), "repetition_penalty": float(g["repetition_penalty"])}


def eos_ids(hf: Any, tok: Any) -> list[int]:
    ids: list[int] = []
    e = getattr(getattr(hf, "generation_config", None), "eos_token_id", None)
    for x in (e if isinstance(e, (list, tuple)) else [e]) + [tok.eos_token_id]:
        if x is not None and int(x) not in ids:
            ids.append(int(x))
    return ids


@torch.no_grad()
def sample_batch(hf: Any, tok: Any, ids_list: Sequence[Sequence[int]], *, seed: int, max_new_tokens: int,
                 gen_kw: dict | None = None, hook_factory: Callable | None = None,
                 eos: Sequence[int] | None = None) -> tuple[list[dict], Any]:
    """Left-padded batched sampling on exact prompt ids. Returns per row
    ``{output_token_ids, stopped}`` (cut at the first EOS, inclusive) and the hook's
    stats (or None). ``hook_factory`` returns a fresh context manager active for the
    whole ``generate`` call (prefill + decode)."""
    eos = list(eos or eos_ids(hf, tok))
    pad = tok.pad_token_id if tok.pad_token_id is not None else eos[0]
    B, T = len(ids_list), max(len(x) for x in ids_list)
    inp = torch.full((B, T), int(pad), dtype=torch.long)
    att = torch.zeros((B, T), dtype=torch.long)
    for i, x in enumerate(ids_list):
        inp[i, T - len(x):] = torch.as_tensor(list(x), dtype=torch.long)
        att[i, T - len(x):] = 1
    dev = hf.get_input_embeddings().weight.device
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    kw = dict(gen_kw or gen_kwargs())
    ctx = hook_factory() if hook_factory is not None else contextlib.nullcontext()
    with ctx as hook:
        out = hf.generate(input_ids=inp.to(dev), attention_mask=att.to(dev), max_new_tokens=int(max_new_tokens),
                          pad_token_id=int(pad), eos_token_id=eos, **kw)
    eset = set(eos)
    rows = []
    for row in out[:, T:].cpu().tolist():
        cut = next((j for j, t in enumerate(row) if t in eset), None)
        if cut is None:
            rows.append({"output_token_ids": row, "stopped": "length"})
        else:
            rows.append({"output_token_ids": row[: cut + 1], "stopped": "eos"})
    return rows, getattr(hook, "stats", None)


def _run_generation(hf: Any, tok: Any, prompts: Sequence[Any], *, k: int, seeds: Sequence[int],
                    max_new_tokens: int, batch_size: int, out_jsonl: str | os.PathLike,
                    status_cb: Callable | None, conditions: Sequence[str | None],
                    hook_factories: dict[str | None, Callable | None], phase: str, round_: int,
                    render: Callable | None = None, gen_kw: dict | None = None) -> dict:
    if len(seeds) < k:
        raise ValueError(f"need {k} seeds, got {len(seeds)}")
    cb = _safe_cb(status_cb)
    render = _render_fn(render)
    repair_jsonl(out_jsonl)
    done = done_keys(out_jsonl)
    rendered = {}
    for p in prompts:
        if p.id in rendered:
            raise ValueError(f"duplicate prompt id {p.id}")
        rendered[p.id] = list(render(tok, p)[0])
    expected = {key_of(p.id, s, c) for c in conditions for s in range(k) for p in prompts}
    total = len(expected)
    n_done = len(done & expected)
    batches = [list(prompts[i:i + batch_size]) for i in range(0, len(prompts), batch_size)]
    eos = eos_ids(hf, tok)
    t_start = time.time()
    n_new = toks = 0
    cb(phase, n_done, total, [_stamp(f"{phase} start {n_done}/{total}")])
    for c in conditions:
        first = True
        for s in range(k):
            for b, batch in enumerate(batches):
                keys = [key_of(p.id, s, c) for p in batch]
                if all(kk in done for kk in keys):
                    continue
                seed = batch_seed(seeds[s], b)
                t0 = time.time()
                factory = hook_factories.get(c)
                if factory is not None and first:
                    hf_track = factory
                    factory = lambda hf_track=hf_track: hf_track(track_stats=True)  # noqa: E731
                outs, stats = sample_batch(hf, tok, [rendered[p.id] for p in batch], seed=seed,
                                           max_new_tokens=max_new_tokens, gen_kw=gen_kw,
                                           hook_factory=factory, eos=eos)
                if stats:
                    fr = {int(l): round(v["removed_sq"] / max(v["total_sq"], 1e-30), 6) for l, v in stats.items()}
                    log.info("%s %s: removed energy fraction per layer (first batch, response positions) %s",
                             phase, c, fr)
                    cb(phase, n_done, total, [_stamp(f"{phase} {c} removed-energy {fr}")])
                    _merge_json(Path(out_jsonl).with_name(Path(out_jsonl).stem + "_hookstats.json"),
                                {str(c): {"sample_idx": s, "batch": b, "removed_energy_fraction": fr,
                                          "n_pos": {int(l): int(v["n_pos"]) for l, v in stats.items()},
                                          "note": "realized in the hooked generation, first batch, DECODE steps only "
                                                  "(= response positions; prefill sink/system-prompt rows excluded)"}})
                first = False
                rows = []
                for p, o, kk in zip(batch, outs, keys):
                    if kk in done:
                        continue
                    ids = o["output_token_ids"]
                    r = {"key": kk, "id": p.id, "source": getattr(p, "source", None), "sample_idx": s,
                         "round": round_}
                    if c is not None:
                        r["condition"] = c
                    r.update(seed=seed, batch=b, prompt_token_ids=rendered[p.id], output_token_ids=ids,
                             text=tok.decode(ids, skip_special_tokens=True), stopped=o["stopped"], n_out=len(ids),
                             cwe=getattr(p, "cwe", None), explicit_api=getattr(p, "explicit_api", None))
                    rows.append(r)
                    toks += len(ids)
                append_jsonl(out_jsonl, rows)
                done.update(keys)
                n_done += len(rows)
                n_new += len(rows)
                dt = time.time() - t0
                msg = (f"{phase} {n_done}/{total} ({c or 'gen'} s{s} b{b + 1}/{len(batches)}: "
                       f"{len(rows)} in {dt:.0f}s, {sum(len(o['output_token_ids']) for o in outs) / max(dt, 1e-9):.0f} tok/s)")
                log.info(msg)
                cb(phase, n_done, total, [_stamp(msg)])
    el = time.time() - t_start
    cb(phase, n_done, total, [_stamp(f"{phase} done {n_done}/{total} (+{n_new} in {el / 60:.1f} min)")])
    return {"phase": phase, "total": total, "done": n_done, "new": n_new, "new_tokens": toks, "elapsed_s": el}


def generate_round0(hf: Any, tok: Any, prompts: Sequence[Any], k: int, seeds: Sequence[int], max_new_tokens: int,
                    batch_size: int, out_jsonl: str | os.PathLike, status_cb: Callable | None = None, *,
                    render: Callable | None = None, gen_kw: dict | None = None, phase: str = "gen_r0") -> dict:
    """Round 0: ``k`` samples per prompt with U's own sampling config; sample index ``s``
    uses frozen seed ``seeds[s]`` (per batch: :func:`batch_seed`). Append-only, resumable."""
    return _run_generation(hf, tok, prompts, k=k, seeds=seeds, max_new_tokens=max_new_tokens,
                           batch_size=batch_size, out_jsonl=out_jsonl, status_cb=status_cb, conditions=[None],
                           hook_factories={}, phase=phase, round_=0, render=render, gen_kw=gen_kw)


def _blocks(hf: Any, tok: Any, lens_model: Any = None):
    return (lens_model or L.wrap_model(hf, tok)).layers


class TrackedProjection:
    """``wor.ablate.ProjectionHook`` (the ablation itself: all positions, prefill and
    decode, unchanged) plus OPTIONAL energy stats measured at RESPONSE positions only.

    wor's own ``track_stats`` sums over every position of every call, so the prefill's
    attention-sink row (position 0 holds ~90% of the summed squared residual norm) and the
    shared system prompt dominate it. Here the stats hooks count only DECODE calls
    (``T == 1``), whose single position is a generated token, i.e. the response positions.
    They are registered BEFORE the projection hooks on the same blocks, so they see the
    un-projected block output (PyTorch runs forward hooks in registration order). Rows of
    already-finished sequences (post-EOS pad steps) are included; diagnostic only."""

    def __init__(self, blocks: Any, subspaces: dict, *, track_stats: bool = False) -> None:
        from wor.ablate import ProjectionHook  # noqa: PLC0415

        self.proj = ProjectionHook(blocks, subspaces)
        self.blocks = blocks
        self.track = bool(track_stats)
        self.stats: dict[int, dict[str, float]] = {}
        self._handles: list[Any] = []

    def _stat_hook(self, layer: int) -> Callable:
        U_cache: dict[tuple, torch.Tensor] = {}

        def hook(module, inputs, output):
            h = output if torch.is_tensor(output) else output[0]
            if h.dim() != 3 or h.shape[1] != 1:   # prefill / teacher-forced call: not a response step
                return None
            key = (h.device, h.dtype)
            if key not in U_cache:
                U_cache[key] = self.proj.bases[layer].to(h.device).float()
            U = U_cache[key]
            x = h.float()
            removed = (x @ U.T) @ U
            st = self.stats.setdefault(layer, {"removed_sq": 0.0, "total_sq": 0.0, "n_pos": 0})
            st["removed_sq"] += float((removed ** 2).sum())
            st["total_sq"] += float((x ** 2).sum())
            st["n_pos"] += int(x.shape[0] * x.shape[1])
            return None

        return hook

    def __enter__(self) -> TrackedProjection:
        if self.track:
            for l in self.proj.bases:
                self._handles.append(self.blocks[l].register_forward_hook(self._stat_hook(l)))
        self.proj.__enter__()
        return self

    def __exit__(self, *exc) -> None:
        self.proj.__exit__(*exc)
        for h in self._handles:
            h.remove()
        self._handles = []


def hook_factories(blocks: Any, ablate_layers: Sequence[int], v_subs: dict | None,
                   rand_subs: dict | None) -> dict[str, Callable | None]:
    """Condition -> factory of a fresh :class:`TrackedProjection` (= ``wor.ablate.ProjectionHook``
    over ``ablate_layers``, PositionSpec default: all positions, prefill and decode; stats,
    if requested, at response positions only)."""

    def make(subs: dict) -> Callable:
        sel = {int(l): subs[int(l)] for l in ablate_layers}
        return lambda track_stats=False: TrackedProjection(blocks, sel, track_stats=track_stats)

    f: dict[str, Callable | None] = {"no_hook": None}
    if v_subs is not None:
        f["ablate_V"] = make(v_subs)
    if rand_subs is not None:
        f["ablate_rand"] = make(rand_subs)
    return f


def generate_round1(hf: Any, tok: Any, lens: Any, prompts: Sequence[Any], k: int, conditions: Sequence[str],
                    ablate_layers: Sequence[int], read_layers: Sequence[int], v_subs: dict | None,
                    rand_subs: dict | None, seeds: Sequence[int], out_jsonl: str | os.PathLike,
                    status_cb: Callable | None = None, *, max_new_tokens: int | None = None,
                    batch_size: int | None = None, render: Callable | None = None, gen_kw: dict | None = None,
                    lens_model: Any = None, phase: str = "gen_r1") -> dict:
    """Round 1 (S2 monitorability tax). For each condition in ``conditions`` (subset of
    no_hook / ablate_V / ablate_rand) generate ``k`` samples per prompt with MATCHED
    seeds (same batches, same per-batch seeds in every condition); ablate conditions wrap
    the whole ``generate`` call in a ProjectionHook over ``ablate_layers``. ``read_layers``
    (the monitor's later layers) are validated to lie strictly above the ablated ones;
    the monitor read itself is :func:`readout` (stage ``readout round1``). If ``v_subs``
    is None it is built from ``lens`` and the frozen V."""
    bad = set(conditions) - set(CONDITIONS)
    if bad:
        raise ValueError(f"unknown conditions {bad}")
    if read_layers and ablate_layers and min(read_layers) <= max(ablate_layers):
        raise ValueError(f"read layers {list(read_layers)} must lie above ablate layers {list(ablate_layers)}")
    r0 = L.cfg()["generation"]["round0"]
    lm = lens_model or L.wrap_model(hf, tok)
    if v_subs is None and "ablate_V" in conditions:
        v_subs = L.v_subspaces(lens, lm._lm_head.weight, L.final_norm_gain(lm), L.load_vocab_ids(), ablate_layers)  # noqa: SLF001
    if "ablate_rand" in conditions and rand_subs is None:
        raise ValueError("ablate_rand needs rand_subs (lens.rand_subspaces on sample residuals)")
    facs = hook_factories(lm.layers, ablate_layers, v_subs, rand_subs)
    return _run_generation(hf, tok, prompts, k=k, seeds=seeds,
                           max_new_tokens=int(max_new_tokens or r0["max_new_tokens"]),
                           batch_size=int(batch_size or r0["gen_batch_size"]), out_jsonl=out_jsonl,
                           status_cb=status_cb, conditions=list(conditions), hook_factories=facs, phase=phase,
                           round_=1, render=render, gen_kw=gen_kw)


# --------------------------------------------------------------------------- readout
def capture_lens_stats(scorer: Any, record: dict, code_pos: Sequence[int], layers: Sequence[int],
                       vocab_ids: Sequence[int], **kw: Any) -> dict:
    """SPEC entry point; implemented in :func:`ic.lens.capture_lens_stats` (see there
    for the full key list). Teacher-forces the saved ids once; reduced stats only."""
    return L.capture_lens_stats(scorer, record, code_pos, layers, vocab_ids, **kw)


def _label_code_positions(tok: Any, rec: dict) -> list[int]:
    from ic import label as LB  # noqa: PLC0415

    code, _status = LB.extract_code(rec["text"])
    stripped, _ = LB.strip_comments(code)
    return list(LB.code_positions(tok, rec["prompt_token_ids"], rec["output_token_ids"], stripped))


def _fallback_positions(rec: dict, eos: set[int]) -> list[int]:
    n_p, out = len(rec["prompt_token_ids"]), list(rec["output_token_ids"])
    while out and out[-1] in eos:
        out.pop()
    return list(range(n_p, n_p + len(out)))


def resolve_code_positions(tok: Any, records: Sequence[dict], code_pos_fn: Callable | None = None,
                           eos: Sequence[int] = ()) -> tuple[dict[str, list[int]], dict[str, str | None], str]:
    """Eligible code positions per record key via ``ic.label`` (extract_code ->
    strip_comments -> code_positions). Readout does NOT wait for bandit labels.

    Convention: SPEC passes ``prompt_token_ids`` to ``label.code_positions``, so ABSOLUTE
    positions are expected; if any record has a position below its prompt length the
    whole run is treated as output-relative (logged). Records with no code positions
    (unparseable / empty) or a label error fall back to all output tokens (minus the
    EOS) with ``code_pos_fallback`` = "empty" | "error". If that fallback is empty too
    (the output is only an EOS), the record gets ``[]`` and ``code_pos_fallback`` =
    "empty_output": readout then skips the code-position capture for it."""
    fn = code_pos_fn or _label_code_positions
    raw: dict[str, list[int]] = {}
    fb: dict[str, str | None] = {}
    n_err = 0
    for r in records:
        try:
            raw[r["key"]] = sorted({int(x) for x in fn(tok, r)})
            fb[r["key"]] = None if raw[r["key"]] else "empty"
        except Exception as exc:  # noqa: BLE001
            n_err += 1
            if n_err <= 3:
                log.error("code_positions failed for %s: %r", r["key"], exc)
            raw[r["key"]], fb[r["key"]] = [], "error"
    conv = "absolute"
    if any(raw[r["key"]] and raw[r["key"]][0] < len(r["prompt_token_ids"]) for r in records):
        conv = "relative"
        log.warning("label.code_positions returned output-relative indices; offsetting by prompt length")
    eset = set(eos)
    out = {}
    for r in records:
        cp = raw[r["key"]]
        n_p = len(r["prompt_token_ids"])
        if cp:
            cp = [p + n_p for p in cp] if conv == "relative" else cp
            hi = n_p + len(r["output_token_ids"])
            cp = [p for p in cp if n_p <= p < hi]
            if not cp:
                fb[r["key"]] = "empty"
        if not cp:
            cp = _fallback_positions(r, eset)
            if not cp:
                fb[r["key"]] = "empty_output"
        out[r["key"]] = cp
    if n_err:
        log.error("code_positions errors: %d/%d records (fallback = all output tokens)", n_err, len(records))
    return out, fb, conv


def _safe_name(key: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", key)


def readout(hf: Any, tok: Any, lens: Any, in_jsonl: str | os.PathLike, out_jsonl: str | os.PathLike, *,
            layers: Sequence[int] | None = None, vocab_ids: Sequence[int] | None = None,
            rand_vocab_ids: Sequence[int] | None = None, rot_lens: Any = None,
            s3_dir: str | os.PathLike | None = None, s3_npz: str | os.PathLike | None = None,
            status_cb: Callable | None = None, phase: str = "readout",
            hook_subs: dict[str, dict] | None = None, ablate_layers: Sequence[int] | None = None,
            code_pos_fn: Callable | None = None, lens_model: Any = None, scorer: Any = None,
            rot_scorer: Any = None, later: Sequence[int] | None = None, lexical: bool = True,
            max_error_frac: float = 0.05) -> dict:
    """Teacher-forced reduced lens stats for every record of ``in_jsonl`` (resumable).

    ``hook_subs`` = {condition: subspaces} re-applies the Round-1 ablation during the
    teacher-forced forward (the monitor reads U's actual, ablated residuals). ``s3_dir``
    set => S3 coordinates saved per record (Round 0 only). ``lexical`` adds the literal-V
    control (``n_vlex_code``, ``wb0_nolex``; PREREG s6 (e)/(f)).

    One bad record never kills the paid stage: an exception is recorded on that record's
    row (``readout_error``, WB fields null) and the loop goes on; a record whose output is
    only an EOS (``code_pos_fallback == "empty_output"``) gets ``wb_pre`` only. The stage
    raises at the END (after the rows and the S3 pack are written) only if more than
    ``max_error_frac`` of the processed records errored."""
    cb = _safe_cb(status_cb)
    layers = [int(l) for l in (layers or L.band_layers())]
    vocab_ids = list(vocab_ids if vocab_ids is not None else L.load_vocab_ids())
    lm = lens_model or L.wrap_model(hf, tok)
    scorer = scorer or L.make_scorer(lm, lens)
    later = [int(l) for l in (later or L.later_layers())]
    L.preload(scorer, sorted(set(layers) | set(later)))
    if rot_scorer is None and rot_lens is not None:
        rot_scorer = L.make_scorer(lm, rot_lens)
    if rot_scorer is not None:
        L.preload(rot_scorer, sorted(set(layers) | set(later)))
    hooks: dict[str, Any] = {}
    if hook_subs:
        from wor.ablate import ProjectionHook  # noqa: PLC0415

        al = [int(l) for l in (ablate_layers or L.cfg()["generation"]["round1_S2"]["ablate_layers"])]
        hooks = {c: ProjectionHook(lm.layers, {l: subs[l] for l in al}) for c, subs in hook_subs.items()}
    repair_jsonl(out_jsonl)
    done = done_keys(out_jsonl)
    recs = read_jsonl(in_jsonl)
    total = len(recs)
    todo = [r for r in recs if r["key"] not in done]
    eos = eos_ids(hf, tok)
    cpos, fb, conv = resolve_code_positions(tok, todo, code_pos_fn, eos)
    n_fb = sum(1 for k in fb.values() if k)
    s3p = Path(s3_dir) if s3_dir else None
    if s3p:
        s3p.mkdir(parents=True, exist_ok=True)
    n_done = total - len(todo)
    t_start = last_cb = time.time()
    cb(phase, n_done, total, [_stamp(f"{phase} start {n_done}/{total} (code-pos {conv}, fallback {n_fb})")])
    pat = L.v_stem_regex() if lexical else None
    buf: list[dict] = []
    n_err = n_empty = 0
    for i, r in enumerate(todo):
        cond = r.get("condition")
        ctx = hooks.get(cond) if cond in hooks else contextlib.nullcontext()
        row = {"key": r["key"], "id": r["id"], "source": r.get("source"), "sample_idx": r["sample_idx"],
               "round": r.get("round")}
        if cond is not None:
            row["condition"] = cond
        row["code_pos_fallback"] = fb.get(r["key"])
        coords = None
        try:
            if not cpos[r["key"]]:                       # output is only an EOS: nothing to read at code positions
                n_empty += 1
                with ctx:
                    wp = L.wb_pre(scorer, r["prompt_token_ids"], layers, vocab_ids)
                row.update(wb0=None, wb0_mean_top5=None, wb_pre=wp, wb_later=None, eligible_len=0,
                           readout_skipped="empty_output")
            else:
                lexpos = L.lexical_v_positions(tok, r, vocab_ids, pattern=pat) if lexical else None
                with ctx:
                    st = L.capture_lens_stats(scorer, r, cpos[r["key"]], layers, vocab_ids, later=later,
                                              rand_vocab_ids=rand_vocab_ids, rot_scorer=rot_scorer,
                                              with_s3=s3p is not None, lex_positions=lexpos)
                coords = st.pop("s3_coords", None)
                row.update(st)
        except Exception as exc:  # noqa: BLE001  (one record must never kill the paid stage)
            n_err += 1
            if n_err <= 5:
                log.error("readout failed for %s: %r", r["key"], exc)
            row.update(wb0=None, wb0_mean_top5=None, wb_pre=None, wb_later=None, readout_error=repr(exc)[:300])
        if coords is not None and s3p is not None:
            f = s3p / f"{_safe_name(r['key'])}.npy"
            tmp = f.with_suffix(".tmp.npy")
            np.save(tmp, coords)
            os.replace(tmp, f)
            row["s3_saved"] = True
        buf.append(row)
        if len(buf) >= 16 or i == len(todo) - 1:
            append_jsonl(out_jsonl, buf)
            n_done += len(buf)
            buf = []
            now = time.time()
            if now - last_cb >= 2 or i == len(todo) - 1:
                rate = (i + 1) / max(now - t_start, 1e-9)
                cb(phase, n_done, total, [_stamp(f"{phase} {n_done}/{total} ({rate:.2f} seq/s)")])
                last_cb = now
    el = time.time() - t_start
    s3_info = None
    if s3p is not None:
        npz = Path(s3_npz) if s3_npz else Path(out_jsonl).parent / f"{s3p.parent.name}_{s3p.name}.npz"
        s3_info = consolidate_s3(out_jsonl, s3p, npz, layers)
    cb(phase, n_done, total, [_stamp(f"{phase} done {n_done}/{total} in {el / 60:.1f} min"
                                     + (f"; {n_err} readout errors, {n_empty} empty outputs" if n_err or n_empty else ""))])
    res = {"phase": phase, "total": total, "done": n_done, "new": len(todo), "elapsed_s": el,
           "code_pos_convention": conv, "code_pos_fallbacks": n_fb, "s3": s3_info,
           "readout_errors": n_err, "empty_outputs": n_empty}
    if todo and n_err > max_error_frac * len(todo):
        _record_timing(f"{phase}_failed", **res)
        raise RuntimeError(f"{phase}: {n_err}/{len(todo)} records failed (> {max_error_frac:.0%}); rows were "
                           "written with readout_error, see the log")
    return res


def consolidate_s3(readout_jsonl: str | os.PathLike, s3_dir: Path, npz: Path, layers: Sequence[int]) -> dict:
    """Pack the per-record S3 staging files (plus any earlier ``npz``) into ONE ``npz``
    (``coords, ids, sample_idx, keys, layers``; float16 unless a row overflows), atomically,
    then delete the staging files. Rows are in readout-file order."""
    old: dict[str, np.ndarray] = {}
    if npz.exists():
        with np.load(npz, allow_pickle=False) as d:
            old = {str(k): d["coords"][i] for i, k in enumerate(d["keys"])}
    rows = [r for r in read_jsonl(readout_jsonl) if r.get("s3_saved")]
    keys, ids, sidx, xs, missing, files = [], [], [], [], [], []
    for r in rows:
        f = s3_dir / f"{_safe_name(r['key'])}.npy"
        if f.exists():
            x = np.load(f)
            files.append(f)
        elif r["key"] in old:
            x = old[r["key"]]
        else:
            missing.append(r["key"])
            continue
        keys.append(r["key"])
        ids.append(str(r["id"]))
        sidx.append(int(r["sample_idx"]))
        xs.append(x)
    if missing:
        log.error("S3: %d rows have no coordinates (e.g. %s)", len(missing), missing[:3])
    if not xs:
        return {"n": 0, "missing": len(missing), "path": None}
    X = np.stack(xs).astype(np.float32)
    X16 = X.astype(np.float16)
    X = X16 if np.isfinite(X16).all() else X
    npz.parent.mkdir(parents=True, exist_ok=True)
    tmp = npz.with_suffix(".tmp")
    with open(tmp, "wb") as fh:
        np.savez(fh, coords=X, ids=np.array(ids), sample_idx=np.array(sidx, dtype=np.int64),
                 keys=np.array(keys), layers=np.array([int(l) for l in layers], dtype=np.int64))
    os.replace(tmp, npz)
    for f in files:
        f.unlink(missing_ok=True)
    for d in (s3_dir, s3_dir.parent):
        try:
            d.rmdir()  # only if empty
        except OSError:
            pass
    log.info("S3: %d x %s %s -> %s", len(keys), X.shape[1:], X.dtype, npz)
    return {"n": len(keys), "missing": len(missing), "path": str(npz), "dtype": str(X.dtype)}


def load_s3(results: str | os.PathLike | None = None, name: str = "s3_coords_round0.npz") -> tuple[list[str], np.ndarray]:
    """S3 features: ``(keys, X [N, L, d] float32)`` from the consolidated npz."""
    base = Path(results) if results else L.results_dir()
    with np.load(base / name, allow_pickle=False) as d:
        return [str(k) for k in d["keys"]], d["coords"].astype(np.float32)


# --------------------------------------------------------------- Round-1 subspaces
@torch.no_grad()
def sample_residuals(scorer: Any, records: Sequence[dict], layers: Sequence[int], *,
                     positions: str = "response", drop_norm_x_median: float | None = 10.0,
                     return_info: bool = False):
    """U's residuals at ``layers`` (no hook active), concatenated: ``{layer: [N, d]}`` fp32,
    kept on the capture device (the pod's GPU; CPU on the laptop).

    ``positions="response"`` (default, the frozen recipe): only the response positions
    ``n_p .. n_p + len(output) - 2`` (the final output token, normally the EOS, is skipped),
    because the ablation's effect matters where U writes code; position 0 (the attention
    sink, ~90% of the summed squared norm on Qwen2.5) and the shared system prompt would
    otherwise dominate the energy match. A record with < 2 output tokens (prompts-only
    fallback) contributes all its positions. ``positions="all"``: every position.
    Then, per layer, rows whose L2 norm exceeds ``drop_norm_x_median`` x the median row norm
    (massive activations) are dropped. ``return_info`` adds per-layer row counts."""
    acc: dict[int, list[torch.Tensor]] = {int(l): [] for l in layers}
    n_resp = n_all = 0
    for r in records:
        p_ids = list(r["prompt_token_ids"])
        out = list(r.get("output_token_ids") or [])
        ids = p_ids + out
        n_p = len(p_ids)
        if positions == "response" and len(out) >= 2:
            pos: list[int] | None = list(range(n_p, n_p + len(out) - 1))
            n_resp += 1
        elif positions in ("response", "all"):
            pos = None
            n_all += 1
        else:
            raise ValueError(f"positions must be 'response' or 'all', got {positions!r}")
        acts, _, _ = scorer.capture(ids, [int(l) for l in layers], positions=pos, include_final=False)
        for l in acc:
            acc[l].append(acts[l].float())
    X = {l: torch.cat(v) for l, v in acc.items()}
    info: dict[str, Any] = {"positions": positions, "n_records_response": n_resp, "n_records_all_positions": n_all,
                            "drop_norm_x_median": drop_norm_x_median, "layers": {}}
    for l in list(X):
        n0 = int(X[l].shape[0])
        dropped = 0
        if drop_norm_x_median and n0:
            nrm = X[l].norm(dim=1)
            keep = nrm <= float(drop_norm_x_median) * nrm.median()
            dropped = int((~keep).sum())
            X[l] = X[l][keep]
        info["layers"][str(l)] = {"n_rows": n0, "n_dropped_massive": dropped, "n_kept": int(X[l].shape[0])}
    return (X, info) if return_info else X


def build_round1_subspaces(lens: Any, lm: Any, scorer: Any, s2_ids: set, r0_records: Sequence[dict],
                           vocab_ids: Sequence[int], ablate_layers: Sequence[int], *,
                           fallback_seqs: Sequence[Sequence[int]] = (), progress: Callable | None = None,
                           ) -> tuple[dict, dict, dict]:
    """V subspaces + the energy-matched random control, matched on RESPONSE-position
    residuals (config ``random_control_sample``). ``meta`` records, per layer, the V and
    random removed-energy fractions on that same sample and their relative difference;
    a difference above ``match_rel_tol`` is logged as an ERROR (the S2 contrast would then
    compare perturbations of unequal size)."""
    rc = resid_sample_cfg()
    n_seqs, max_seqs = int(rc["n_seqs"]), int(rc["max_seqs"])
    min_rows = int(float(rc["min_rows_x_d_model"]) * int(lm.d_model))
    v = L.v_subspaces(lens, lm._lm_head.weight, L.final_norm_gain(lm), vocab_ids, ablate_layers)  # noqa: SLF001
    cand = [r for r in r0_records if r["id"] in s2_ids and r["sample_idx"] == 0]
    src = "round0_outputs"
    if not cand:
        cand = [{"prompt_token_ids": list(s)} for s in fallback_seqs]
        src = "prompts_only"
    # file order; at least n_seqs, then more until the response rows reach min_rows_x_d_model * d_model
    # (a full-rank residual covariance for the energy match), at most max_seqs
    pool, rows = [], 0
    for r in cand:
        if len(pool) >= max_seqs or (len(pool) >= n_seqs and rows >= min_rows):
            break
        pool.append(r)
        out = r.get("output_token_ids") or []
        rows += (len(out) - 1) if (str(rc["positions"]) == "response" and len(out) >= 2) else len(r["prompt_token_ids"]) + len(out)
    X, xinfo = sample_residuals(scorer, pool, ablate_layers, positions=str(rc["positions"]),
                                drop_norm_x_median=rc["drop_rows_norm_gt_x_median"], return_info=True)
    tol = float(rc["match_rel_tol"])
    rnd = L.rand_subspaces(X, v, progress=progress, rel_tol=tol)
    from wor.ablate import removed_energy_fraction  # noqa: PLC0415

    meta = {"sample_source": src, "n_seqs": len(pool), "sample": xinfo, "match_rel_tol": tol,
            "n_positions": int(next(iter(X.values())).shape[0]), "min_rows_target": min_rows,
            "d_model": int(lm.d_model), "layers": {}, "energy_match_ok": True}
    if meta["n_positions"] < int(lm.d_model):
        log.warning("random-control sample has %d rows < d_model %d (rank-deficient covariance)",
                    meta["n_positions"], lm.d_model)
    for l in v:
        e_v = removed_energy_fraction(X[l], v[l].basis)
        e_r = removed_energy_fraction(X[l], rnd[l].basis)
        rel = abs(e_r - e_v) / max(e_v, 1e-12)
        ok = rel <= tol
        meta["energy_match_ok"] = bool(meta["energy_match_ok"] and ok)
        meta["layers"][str(l)] = {
            "v_rank": v[l].rank, "v_explained": v[l].explained,
            "v_removed_energy": e_v, "rand_removed_energy": e_r, "rel_diff": rel, "matched": ok,
            "energy_positions": f"{rc['positions']} (massive rows > {rc['drop_rows_norm_gt_x_median']}x median dropped)",
            "rand_rank": rnd[l].rank, "rand_meta": dict(rnd[l].meta)}
        if not ok:
            log.error("S2 energy match FAILED at L%d on response positions: V removes %.4g, random %.4g "
                      "(rel diff %.1f%% > %.0f%%): the ablate_V vs ablate_rand contrast is confounded by size",
                      l, e_v, e_r, 100 * rel, 100 * tol)
    return v, rnd, meta


def save_subspaces(path: Path, v: dict, rnd: dict, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({"v": {int(l): s.basis.detach().cpu() for l, s in v.items()},
                "rand": {int(l): s.basis.detach().cpu() for l, s in rnd.items()}, "meta": meta}, tmp)
    os.replace(tmp, path)
    path.with_suffix(".json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")


def load_subspaces(path: Path, device: Any = None) -> tuple[dict, dict, dict]:
    from wor.ablate import Subspace  # noqa: PLC0415

    d = torch.load(path, map_location="cpu", weights_only=False)
    mk = lambda b, l, kind: Subspace(b.to(device) if device else b, layer=l, meta={"kind": kind})  # noqa: E731
    return ({int(l): mk(b, int(l), "j_tokens") for l, b in d["v"].items()},
            {int(l): mk(b, int(l), "random_energy_matched") for l, b in d["rand"].items()}, d["meta"])


# ------------------------------------------------------------------------ CLI stages
@dataclass
class Ctx:
    """Loaded-once state shared by the U stages."""

    hf: Any = None
    tok: Any = None
    lm: Any = None
    lens: Any = None
    scorer: Any = None
    rot_scorer: Any = None
    prompts: list = field(default_factory=list)
    all_prompts: list = field(default_factory=list)
    limit: int | None = None

    def ensure(self) -> Ctx:
        if self.hf is None:
            self.hf, self.tok = load_U()
            self.lm = L.wrap_model(self.hf, self.tok)
            self.lens = L.load_lens()
            self.scorer = L.make_scorer(self.lm, self.lens)
        return self

    def load_prompts(self) -> list:
        """The frozen prompt order (``--limit N``, debug only: its first N)."""
        if not self.all_prompts:
            from ic import data as _data  # noqa: PLC0415

            self.all_prompts = list(_data.build_prompt_set())
            self.prompts = self.all_prompts[: self.limit] if self.limit else list(self.all_prompts)
        return self.prompts

    def s2_prompts(self) -> list:
        """Frozen S2_prompts, always drawn from the FULL CyberSecEval set (``--limit``
        only truncates the result)."""
        from ic import data as _data  # noqa: PLC0415

        self.load_prompts()
        r1 = L.cfg()["generation"]["round1_S2"]
        base = [p for p in self.all_prompts if getattr(p, "source", None) != "em_heldout"]
        sub = list(_data.s2_subset(base, int(r1["n_prompts"]), int(r1["subset_seed"])))
        return sub[: self.limit] if self.limit else sub


def stage_sanity(ctx: Ctx) -> dict:
    ctx.ensure()
    res = L.run_sanity(ctx.hf, ctx.tok, ctx.lens, scorer=ctx.scorer)
    try:
        from ic import status as _status  # noqa: PLC0415

        _status.update(phase="sanity", event_text=f"G1 {res['status']}: L{res['layer']} lens top1 "
                                                  f"{res['top1_agreement']:.3f} (logit lens {res['logit_lens_top1_agreement']:.3f})")
    except Exception as exc:  # noqa: BLE001
        log.warning("status update failed: %r", exc)
    return res


def stage_gen_r0(ctx: Ctx) -> dict:
    ctx.ensure()
    c = L.cfg()["generation"]
    r0 = c["round0"]
    res = generate_round0(ctx.hf, ctx.tok, ctx.load_prompts(), int(r0["k0"]), c["gen_seeds"], int(r0["max_new_tokens"]),
                          int(r0["gen_batch_size"]), L.results_dir() / GEN_R0, default_status_cb())
    _record_timing("gen_r0", **res)
    return res


def _controls(ctx: Ctx) -> tuple[list[int], Any]:
    rv = L.random_vocab_ids(ctx.tok)
    if ctx.rot_scorer is None:
        ctx.rot_scorer = L.make_scorer(ctx.lm, L.rotated_lens(ctx.lens, layers=L.band_layers(),
                                                              device=ctx.lm.input_device))
    return rv, ctx.rot_scorer


def _status_event_cb(phase: str) -> Callable[[str], None]:
    """``msg -> status.json event`` (never raises), e.g. one line per matched subspace layer."""
    def emit(msg: str) -> None:
        try:
            from ic import status as _status  # noqa: PLC0415

            _status.update(event_text=_stamp(f"{phase} {msg}"))
        except Exception as exc:  # noqa: BLE001
            log.warning("status event failed: %r", exc)
    return emit


def stage_readout(ctx: Ctx, round_: int) -> dict:
    ctx.ensure()
    rd = L.results_dir()
    rv, rot = _controls(ctx)
    if round_ == 0:
        res = readout(ctx.hf, ctx.tok, ctx.lens, rd / GEN_R0, rd / READ_R0, rand_vocab_ids=rv, rot_scorer=rot,
                      s3_dir=rd / S3_DIR / "round0", status_cb=default_status_cb(), phase="readout",
                      lens_model=ctx.lm, scorer=ctx.scorer)
    else:
        v, rnd, _ = load_subspaces(rd / SUBS_R1, device=ctx.lm.input_device)
        res = readout(ctx.hf, ctx.tok, ctx.lens, rd / GEN_R1, rd / READ_R1, rand_vocab_ids=rv, rot_scorer=rot,
                      s3_dir=None, status_cb=default_status_cb(), phase="readout_r1",
                      hook_subs={"ablate_V": v, "ablate_rand": rnd}, lens_model=ctx.lm, scorer=ctx.scorer)
    _record_timing(f"readout_r{round_}", **res)
    return res


def stage_gen_r1(ctx: Ctx) -> dict:
    ctx.ensure()
    rd = L.results_dir()
    c = L.cfg()["generation"]
    r1 = c["round1_S2"]
    s2 = ctx.s2_prompts()
    al, rl = [int(x) for x in r1["ablate_layers"]], [int(x) for x in r1["monitor_read_layers"]]
    sp = rd / SUBS_R1
    if sp.exists():  # resume: identical subspaces for generation and the later readout
        v, rnd, meta = load_subspaces(sp, device=ctx.lm.input_device)
    else:
        render = _render_fn(None)
        emit = _status_event_cb("gen_r1")
        emit("building V + energy-matched random subspaces (response-position residuals)")
        v, rnd, meta = build_round1_subspaces(ctx.lens, ctx.lm, ctx.scorer, {p.id for p in s2}, read_jsonl(rd / GEN_R0),
                                              L.load_vocab_ids(), al,
                                              fallback_seqs=[render(ctx.tok, p)[0] for p in s2], progress=emit)
        save_subspaces(sp, v, rnd, meta)
        log.info("round1 subspaces: %s", json.dumps(meta["layers"], default=str))
        if not meta.get("energy_match_ok"):
            emit("WARNING energy match outside tolerance at >= 1 layer (see round1_subspaces.json)")
    res = generate_round1(ctx.hf, ctx.tok, ctx.lens, s2, int(r1["k1"]), list(r1["conditions"]), al, rl, v, rnd,
                          c["gen_seeds"], rd / GEN_R1, default_status_cb(), lens_model=ctx.lm)
    _record_timing("gen_r1", **res)
    return res


def done_dir() -> Path:
    env = os.environ.get("IC_DONE_DIR")
    if env:
        return Path(env)
    return Path("/workspace/ic/done") if Path("/workspace/ic").is_dir() else L.results_dir() / "done"


def _spend_ok(projected_usd: float) -> bool:
    """``pod/spend_guard.py check-step`` (exit 3 = do not start). Missing script: True."""
    sg = Path(os.environ.get("IC_SPEND_GUARD", L.EM_DIR / "pod" / "spend_guard.py"))
    if not sg.exists():
        log.warning("spend guard %s not found; continuing", sg)
        return True
    rc = subprocess.run([sys.executable, str(sg), "check-step", "--projected-usd", f"{projected_usd:.3f}"]).returncode
    if rc == 3:
        log.warning("spend guard says stop (projected %.2f USD)", projected_usd)
        return False
    if rc != 0:
        log.warning("spend guard check-step exit %d; continuing (the watchdog is the hard stop)", rc)
    return True


def _projected_r1_usd() -> float:
    """Gen + readout of Round 1 at Round-0 measured speed (+15% hook overhead)."""
    try:
        t = json.loads((L.results_dir() / TIMINGS).read_text(encoding="utf-8"))
        r1 = L.cfg()["generation"]["round1_S2"]
        n1 = int(r1["n_prompts"]) * int(r1["k1"]) * len(r1["conditions"])
        g, rd = t.get("gen_r0", {}), t.get("readout_r0", {})
        per_g = g["elapsed_s"] / max(g["new"], 1)
        per_r = rd.get("elapsed_s", 0.0) / max(rd.get("new", 1), 1)
        rate = float(os.environ.get("IC_HOURLY_USD", L.cfg()["budget"]["rate_usd_h"]))
        return (n1 * per_g * 1.15 + n1 * per_r) / 3600 * rate
    except (OSError, ValueError, KeyError, ZeroDivisionError):
        return 0.0


STAGES: dict[str, Callable[[Ctx], dict]] = {
    "sanity": stage_sanity,
    "gen_r0": stage_gen_r0,
    "readout": lambda c: stage_readout(c, 0),
    "gen_r1": stage_gen_r1,
    "read_r1": lambda c: stage_readout(c, 1),
}
#: Stages whose crash must not cost the paid stages after them (G1 runs FIRST so its result
#: can gate Round 1, but it is a logged sanity check, never a blocker of Round 0).
NONFATAL = {"sanity"}


def g1_status(results: str | os.PathLike | None = None) -> str | None:
    """``status`` field of results/sanity_G1.json (pass | marginal | gross_fail | error), or None."""
    p = Path(results or L.results_dir()) / "sanity_G1.json"
    try:
        return json.loads(p.read_text(encoding="utf-8")).get("status")
    except (OSError, ValueError, AttributeError):
        return None


def run_u_all(ctx: Ctx, stages: Sequence[str] = tuple(STAGES)) -> int:
    """All U stages in ONE process (U loaded once). Same done-marker names as
    pod/run_all.sh, so either runner skips what the other finished. Round 1 is gated by
    the spend guard with a projected cost from Round-0 timings, and skipped when G1 grossly
    failed (the base-fit lens does not transport to U, PREREG s7). A crash in ``sanity`` is
    logged (sanity_G1.json status "error", no marker) and the next stages still run.
    NOTE: readout uses only label.py's pure extraction functions, not the bandit labels."""
    dd = done_dir()
    dd.mkdir(parents=True, exist_ok=True)
    for st in stages:
        if (dd / st).exists():
            log.info("skip %s (done marker)", st)
            continue
        if st in ("gen_r1", "read_r1") and g1_status() == "gross_fail":
            log.warning("skip %s: G1 gross failure (WB results exploratory; Round 1 not spent)", st)
            continue
        if not _spend_ok(_projected_r1_usd() if st == "gen_r1" else 0.0):
            return 3
        t0 = time.time()
        if st in NONFATAL:
            try:
                STAGES[st](ctx)
            except Exception as exc:  # noqa: BLE001
                log.exception("stage %s crashed (non-fatal; continuing): %r", st, exc)
                try:
                    p = L.results_dir() / "sanity_G1.json"
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_text(json.dumps({"gate": "G1", "status": "error", "error": repr(exc)[:500]}, indent=1),
                                 encoding="utf-8")
                    _status_event_cb(st)(f"G1 sanity CRASHED (non-fatal): {type(exc).__name__}")
                except Exception:  # noqa: BLE001
                    pass
                continue
        else:
            STAGES[st](ctx)
        (dd / st).touch()
        log.info("stage %s done in %.1f min", st, (time.time() - t0) / 60)
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(prog="python -m ic.generate")
    ap.add_argument("cmd", choices=["round0", "round1", "readout", "sanity", "u_all"])
    ap.add_argument("which", nargs="?", choices=["round0", "round1"], help="for readout")
    ap.add_argument("--stages", default=",".join(STAGES), help="u_all: comma-separated subset")
    ap.add_argument("--limit", type=int, default=None, help="debug only: first N prompts")
    a = ap.parse_args(argv)
    limit = a.limit
    if limit is None and os.environ.get("IC_PROMPT_LIMIT"):
        limit = int(os.environ["IC_PROMPT_LIMIT"])   # smoke.py --pod-path only (run_all passes no --limit)
    if limit:
        log.warning("DEBUG prompt limit %d in effect: NOT the preregistered prompt set", limit)
    ctx = Ctx(limit=limit)
    if a.cmd == "round0":
        stage_gen_r0(ctx)
    elif a.cmd == "round1":
        stage_gen_r1(ctx)
    elif a.cmd == "readout":
        if a.which is None:
            ap.error("readout needs round0 or round1")
        stage_readout(ctx, 0 if a.which == "round0" else 1)
    elif a.cmd == "sanity":
        stage_sanity(ctx)
    else:
        return run_u_all(ctx, [s for s in a.stages.split(",") if s])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
