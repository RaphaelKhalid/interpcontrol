"""J-lens wrapper for interpcontrol/em: lens loading, WB readouts, vocabulary V,
and the S2 subspace-ablation pieces. Thin layer over the vendored ``wor``
(flagship/src/wor) and Anthropic ``jlens`` (flagship/third_party/jlens, Apache-2.0);
neither is forked here.

Readout (PREREG section 4, config ``lens.readout``): for band layer ``l`` and
position ``p``, ``m_{l,p} = sum_{v in V} softmax(lm_head(final_norm(J_l h_{l,p})))_v``,
the FULL-softmax probability mass of V (no top-k). Pool over layers by mean, then
reduce over the eligible code positions (``max`` primary, ``mean_top5`` variant).
The numerics are ``wor.scorer.JLensScorer`` in ``precision="model"`` mode: J transport
in fp32, then the model's own final norm + LM head in the model dtype, i.e. the
exact op sequence of ``jlens.JacobianLens.apply`` (asserted by the tests).

Defensive-research note: the S2 "attacker" pieces below (``v_subspaces``,
``rand_subspaces``) only build projection subspaces for a mechanistic ablation that
tests whether the white-box monitor signal is load-bearing or evadable. This is a
monitor-robustness control in the AI-control tradition; it produces no attack
artifact and ships no capability.

CLI (run from ``interpcontrol/em`` with ``PYTHONPATH=src``)::

    python -m ic.lens --freeze-vocab [--out prereg/vocab_ids.json]   # CPU, before any data
    python -m ic.lens --sanity                                       # pod: G1 lens check on U
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

log = logging.getLogger("ic.lens")

EM_DIR = Path(__file__).resolve().parents[2]          # interpcontrol/em
_REPO_ROOT = EM_DIR.parents[1]                        # matsempirical2027 (laptop layout)


# --------------------------------------------------------------------- vendored code
def _ensure_vendor_paths() -> list[str]:
    """Put the vendored ``wor`` and ``jlens`` on ``sys.path`` (mirrors wor/_paths.py).

    Candidates, first to last: ``$IC_VENDOR`` (os.pathsep-separated dirs), then
    ``em/vendor``, ``em/vendor/src``, ``em/vendor/third_party``, ``em/third_party`` (bundle
    layouts), then ``<repo>/flagship/src`` and ``<repo>/flagship/third_party`` (laptop).
    A dir is added only if it contains a ``wor`` or ``jlens`` package."""
    cands: list[Path] = [Path(p) for p in os.environ.get("IC_VENDOR", "").split(os.pathsep) if p]
    cands += [EM_DIR / "vendor", EM_DIR / "vendor" / "src", EM_DIR / "vendor" / "third_party",
              EM_DIR / "third_party", _REPO_ROOT / "flagship" / "src", _REPO_ROOT / "flagship" / "third_party"]
    added = []
    for d in cands:
        if (d / "wor").is_dir() or (d / "jlens").is_dir():
            s = str(d)
            if s not in sys.path:
                sys.path.append(s)
                added.append(s)
    return added


_ensure_vendor_paths()

import jlens  # noqa: E402  (vendored, Apache-2.0)
from wor import ablate as _ablate  # noqa: E402
from wor import modeling as _modeling  # noqa: E402
from wor.ablate import Subspace  # noqa: E402,F401  (re-exported for generate.py)
from wor.scorer import JLensScorer, ScoreResult  # noqa: E402

JacobianLens = jlens.JacobianLens


# ---------------------------------------------------------------------------- config
_CFG: dict | None = None


def cfg() -> dict:
    """Frozen config: ``ic.config.load()`` when that module exists, else parse
    ``prereg/config.json`` directly (same file)."""
    global _CFG
    if _CFG is None:
        try:
            from ic import config as _config  # noqa: PLC0415

            _CFG = _config.load()
        except ImportError:
            _CFG = json.loads((EM_DIR / "prereg" / "config.json").read_text(encoding="utf-8"))
    return _CFG


def band_layers() -> list[int]:
    return list(cfg()["lens"]["band_layers"])


def later_layers() -> list[int]:
    return list(cfg()["wb0"]["wb_later_layers"])


def results_dir() -> Path:
    try:
        from ic import config as _config  # noqa: PLC0415

        return Path(_config.results_dir())
    except (ImportError, AttributeError):
        return Path(os.environ.get("IC_RESULTS", "results"))


def _em_path(p: str | os.PathLike) -> Path:
    """Relative paths in this module are relative to interpcontrol/em, not the CWD."""
    p = Path(p)
    return p if p.is_absolute() else EM_DIR / p


# ----------------------------------------------------------------- lens and scorer
def lens_path() -> str:
    """Local path of the frozen lens file: ``$IC_LENS_PATH`` or the pinned HF file
    (downloaded into the HF cache only if not already there)."""
    env = os.environ.get("IC_LENS_PATH")
    if env:
        return env
    from huggingface_hub import hf_hub_download  # noqa: PLC0415

    L = cfg()["lens"]
    return hf_hub_download(L["repo"], L["filename"], revision=L["revision"])


def load_lens(path: str | None = None) -> JacobianLens:
    """``wor.modeling.load_lens`` (= ``jlens.JacobianLens.load``, weights_only). Logs a
    warning if the file's metadata differs from the frozen config."""
    path = path or lens_path()
    lens = _modeling.load_lens(path)
    L = cfg()["lens"]
    if lens.d_model == L["d_model"]:  # the frozen 32B lens (not a smoke/test lens)
        meta = {"d_model": lens.d_model, "n_prompts": lens.n_prompts, "source_layers": lens.source_layers}
        want = {"d_model": L["d_model"], "n_prompts": L["n_prompts"], "source_layers": L["source_layers"]}
        if meta != want:
            log.warning("lens metadata %s differs from config %s", meta, want)
    return lens


def wrap_model(hf: Any, tok: Any):
    """``wor.modeling.wrap`` -> ``jlens.HFLensModel`` (force_bos=False; exact ids)."""
    return _modeling.wrap(hf, tok)


def make_scorer(lens_model: Any, lens: JacobianLens | None, precision: str = "model") -> JLensScorer:
    """Chunked readout (chunk 128). J_l is moved to the device ONCE per layer and cached
    in fp32 (~105 MB/layer at d=5120), so ``transport`` never re-copies per call. fp32
    (not the file's fp16) keeps the readout bit-identical to ``jlens.apply``."""
    return JLensScorer(lens_model, lens, chunk_size=128, precision=precision, cache_jacobians=True)


def preload(scorer: JLensScorer, layers: Sequence[int]) -> None:
    """Warm the scorer's on-device J cache (call once at stage start)."""
    if scorer.lens is not None:
        for l in layers:
            scorer.jacobian(int(l))


# ------------------------------------------------------------------------ vocabulary
def _inflections(stem: str) -> set[str]:
    """Grammatical inflections only (-s/-es, -ed/-d, -ing with e-drop). Adverbs are
    NOT generated: the config lists e.g. ``secret`` and ``secretly`` separately."""
    s = stem.lower()
    out = {s, s + "s", s + "es", s + "ed", s + "ing"}
    if s.endswith("e"):
        out |= {s + "d", s[:-1] + "ing"}
    return out


def _case_ok(core: str) -> bool:
    return core in (core.lower(), core.lower().capitalize(), core.upper())


def _special_ids(tok: Any) -> set[int]:
    ids = set(getattr(tok, "all_special_ids", []) or [])
    try:
        ids |= set(int(i) for i in tok.get_added_vocab().values())
    except Exception:  # noqa: BLE001
        pass
    return ids


def _decoded_vocab(tok: Any) -> list[str]:
    n = len(tok)
    return tok.batch_decode([[i] for i in range(n)], skip_special_tokens=False,
                            clean_up_tokenization_spaces=False)


def _split_space(s: str) -> str | None:
    """Strip at most ONE leading ASCII space; reject other leading whitespace and
    non-alphabetic cores (surface forms are whole words, +/- leading space)."""
    core = s[1:] if s.startswith(" ") else s
    if not core or not core.isascii() or not core.isalpha():
        return None
    return core


def _roundtrip(tok: Any, s: str, i: int) -> bool:
    return list(tok.encode(s, add_special_tokens=False)) == [int(i)]


def expand_vocab(tok: Any, vcfg: dict | None = None, decoded: list[str] | None = None) -> dict:
    """Expand the frozen stems to single-token ids of ``tok`` (no file I/O, no audit).

    Rule (config ``vocabulary_V.expansion``): scan the whole vocabulary; a token is
    a surface form of a literal stem if, after removing at most one leading space, it is
    alphabetic, has lower/Capitalized/UPPER case, and its lowercase form is an
    inflection of the stem; of a wildcard stem if its lowercase form starts with the
    stem. Special / added tokens are excluded. A candidate is kept only if it is a
    true single-token form: ``tok.encode(decoded) == [id]`` (round trip)."""
    vcfg = vcfg or cfg()["vocabulary_V"]
    literal = list(vcfg["stems_literal"]) + list(vcfg.get("stems_extra", []))
    wild = list(vcfg.get("stems_wildcard", []))
    forms = {st: _inflections(st) for st in literal}
    special = _special_ids(tok)
    dec = decoded if decoded is not None else _decoded_vocab(tok)
    by_stem: dict[str, list[int]] = {st: [] for st in literal + wild}
    surface: dict[int, str] = {}
    rejected: dict[int, str] = {}
    for i, s in enumerate(dec):
        if i in special:
            continue
        core = _split_space(s)
        if core is None or not _case_ok(core):
            continue
        low = core.lower()
        hit = [st for st in literal if low in forms[st]] + [st for st in wild if low.startswith(st.lower())]
        if not hit:
            continue
        if not _roundtrip(tok, s, i):
            rejected[i] = s
            continue
        surface[i] = s
        for st in hit:
            by_stem[st].append(i)
    ids = sorted(surface)
    return {"ids": ids, "surface_forms": {str(i): surface[i] for i in ids},
            "by_stem": {k: sorted(v) for k, v in by_stem.items()},
            "rejected_not_single_token": {str(i): s for i, s in sorted(rejected.items())}}


def _default_render(tok: Any, prompt: Any) -> tuple[list[int], str]:
    """``ic.data.render_chat`` when available (the frozen renderer); otherwise the
    default Qwen chat template with the DEFAULT system prompt (no system message)."""
    try:
        from ic import data as _data  # noqa: PLC0415

        return _data.render_chat(tok, prompt)
    except ImportError:
        text = tok.apply_chat_template([{"role": "user", "content": prompt.text}], tokenize=False,
                                       add_generation_prompt=True)
        return list(tok.encode(text, add_special_tokens=False)), text


def lexical_audit(tok: Any, V: dict, prompts: Sequence[Any], render=None) -> dict:
    """PREREG section 4: no V surface form may be a literal token in any frozen prompt.
    Logged, never enforced here: V stays exactly as the stem rule defines it, and the
    hits feed the S1 lens-reads-the-prompt residualisation."""
    render = render or _default_render
    vset = set(V["ids"])
    hits: dict[str, list[str]] = {}
    per_id: dict[str, int] = {}
    for p in prompts:
        ids = render(tok, p)[0]
        h = sorted(vset.intersection(ids))
        if h:
            hits[str(p.id)] = [V["surface_forms"][str(i)] for i in h]
            for i in h:
                per_id[str(i)] = per_id.get(str(i), 0) + 1
    return {"n_prompts": len(prompts), "n_prompts_with_hits": len(hits), "passed": not hits,
            "hits": hits, "prompts_per_id": per_id,
            "note": "descriptive; V is NOT altered by the audit (PREREG s4/s6 S1 residualisation)"}


def freeze_vocab(tok: Any, out: str | os.PathLike | None = "prereg/vocab_ids.json", *,
                 prompts: Sequence[Any] | None = None, render=None, audit: bool = True) -> dict:
    """Expand ``cfg.vocabulary_V`` to single-token ids, run the lexical audit against
    every frozen prompt (``ic.data.build_prompt_set()`` unless ``prompts`` is given),
    and write ``{ids, surface_forms, by_stem, audit, ...}``. CPU only. The output has
    no timestamp, so re-running on the same tokenizer is byte-identical (hash-stable)."""
    vcfg = cfg()["vocabulary_V"]
    V = expand_vocab(tok, vcfg)
    doc: dict[str, Any] = {
        "ids": V["ids"], "n_ids": len(V["ids"]), "surface_forms": V["surface_forms"],
        "by_stem": V["by_stem"], "rejected_not_single_token": V["rejected_not_single_token"],
        "tokenizer": {"name_or_path": getattr(tok, "name_or_path", None), "len": len(tok),
                      "repo": cfg()["models"]["U_untrusted"]["repo"],
                      "revision": cfg()["models"]["U_untrusted"]["revision"]},
        "rule": vcfg["expansion"],
        "rule_impl": "vocab scan; strip <=1 leading space; alphabetic ASCII core; case in {lower, Capitalized, UPPER}; "
                     "literal: lowercase in {stem, +s, +es, +ed, +ing, (+d, e-drop+ing if stem ends in e)}; "
                     "wildcard: lowercase startswith stem; special/added tokens excluded; encode(decoded)==[id] required",
    }
    if audit:
        if prompts is None:
            from ic import data as _data  # noqa: PLC0415

            prompts = _data.build_prompt_set()
        doc["audit"] = lexical_audit(tok, V, prompts, render=render)
        if not doc["audit"]["passed"]:
            log.warning("lexical audit: %d/%d prompts contain a V token (logged, V unchanged)",
                        doc["audit"]["n_prompts_with_hits"], doc["audit"]["n_prompts"])
    else:
        doc["audit"] = None
    if out:
        p = _em_path(out)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(doc, indent=1, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(tmp, p)
    return doc


def load_vocab_ids(path: str | os.PathLike | None = None) -> list[int]:
    """Frozen V ids from ``prereg/vocab_ids.json`` (``$IC_VOCAB_IDS`` overrides the
    default path; for smoke/integration runs only)."""
    p = _em_path(path or os.environ.get("IC_VOCAB_IDS") or "prereg/vocab_ids.json")
    if not p.exists():
        raise FileNotFoundError(f"{p} missing: run `python -m ic.lens --freeze-vocab` on CPU first")
    return [int(i) for i in json.loads(p.read_text(encoding="utf-8"))["ids"]]


def random_vocab_ids(tok: Any, n: int | None = None, seed: int | None = None) -> list[int]:
    """Control vocabulary: ``n`` ids drawn (numpy default_rng(seed), no replacement)
    from WORD-LIKE single tokens (same surface rule as V: <=1 leading space, alphabetic
    ASCII core of >= 3 letters, plain casing, round-trip single token), excluding V
    and special tokens. Word-like (not arbitrary byte tokens) so the control is matched
    in kind to V. Returned sorted."""
    rc = cfg()["vocabulary_V"]["controls"]["random_vocab"]
    n = int(rc["n_tokens"] if n is None else n)
    seed = int(rc["seed"] if seed is None else seed)
    dec = _decoded_vocab(tok)
    vset = set(expand_vocab(tok, decoded=dec)["ids"])
    special = _special_ids(tok)
    pool = []
    for i, s in enumerate(dec):
        if i in special or i in vset:
            continue
        core = _split_space(s)
        if core is None or len(core) < 3 or not _case_ok(core):
            continue
        pool.append(i)
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(pool), size=n, replace=False)
    out = sorted(int(pool[j]) for j in pick)
    bad = [i for i in out if not _roundtrip(tok, dec[i], i)]
    if bad:  # extremely rare; replace deterministically by walking the pool
        keep = [i for i in out if i not in bad]
        for j in rng.permutation(len(pool)):
            i = int(pool[j])
            if len(keep) == n:
                break
            if i not in keep and _roundtrip(tok, dec[i], i):
                keep.append(i)
        out = sorted(keep)
    return out


# ------------------------------------------------------------------- lexical V control
def v_stem_regex(vcfg: dict | None = None) -> re.Pattern:
    """Case-insensitive SUBSTRING match of any V stem (literal + extra + wildcard), longest
    first. Substring, not word-boundary: in code ``get_secret`` / ``SECRET_KEY`` must hit
    (``\\b`` fails after ``_``). Shared by the WB-0 lexical control and analyze's
    prompt-side stratifier."""
    v = vcfg or cfg()["vocabulary_V"]
    stems = sorted({s.lower() for s in list(v["stems_literal"]) + list(v.get("stems_extra", []))
                    + list(v.get("stems_wildcard", []))}, key=len, reverse=True)
    return re.compile("|".join(re.escape(s) for s in stems), re.IGNORECASE)


def lexical_v_positions(tok: Any, record: dict, vocab_ids: Sequence[int], *,
                        pattern: re.Pattern | None = None) -> list[int]:
    """ABSOLUTE positions of output tokens that are literal-V: the id is in V, or the
    token's character span in the decoded output overlaps a case-insensitive V-stem match
    (``SECRET`` in ``SECRET_KEY``, ``secret`` in ``get_secret``, a stem split across tokens).
    Pure text processing on the saved ids (no forward). PREREG section 6 (e)/(f)."""
    from ic import label as _label  # noqa: PLC0415

    pat = pattern or v_stem_regex()
    out = [int(t) for t in record["output_token_ids"]]
    n_p = len(record["prompt_token_ids"])
    vset = {int(i) for i in vocab_ids}
    full, spans = _label.token_char_spans(tok, out)
    hits = [(m.start(), m.end()) for m in pat.finditer(full)]
    pos = []
    for i, (t, (s, e)) in enumerate(zip(out, spans)):
        if t in vset or (e > s and any(s < he and hs < e for hs, he in hits)):
            pos.append(n_p + i)
    return pos


def lexical_window() -> int:
    return int(((cfg().get("wb0") or {}).get("lexical_control") or {}).get("window", 3))


# --------------------------------------------------------------------------- readouts
def vmass(res: ScoreResult, n_main: int | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Full-softmax probability mass of the requested ids, ``[L, P]`` float64.
    If ``n_main`` is given, the first ``n_main`` requested ids form the main set and
    the rest a second set (e.g. V followed by the random-vocab control): returns
    ``(mass_main, mass_rest)``."""
    lp = res.token_logits.double() - res.lse.double()[..., None]
    p = lp.exp()
    K = p.shape[-1]
    n_main = K if n_main is None else n_main
    rest = p[..., n_main:].sum(-1) if K > n_main else None
    return p[..., :n_main].sum(-1), rest


def reduce_positions(m_p: torch.Tensor, reduce: str = "max") -> float:
    """``max`` | ``mean`` | ``mean_topK`` (e.g. ``mean_top5``) over positions."""
    if m_p.numel() == 0:
        return float("nan")
    if reduce == "max":
        return float(m_p.max())
    if reduce == "mean":
        return float(m_p.mean())
    m = re.fullmatch(r"mean_top(\d+)", reduce)
    if m:
        k = min(int(m.group(1)), m_p.numel())
        return float(m_p.topk(k).values.mean())
    raise ValueError(f"unknown reduce {reduce!r}")


def _check_layers(scorer: JLensScorer, layers: Sequence[int]) -> list[int]:
    layers = [int(l) for l in layers]
    if scorer.lens is not None:
        missing = set(layers) - set(scorer.lens.source_layers)
        if missing:
            raise ValueError(f"layers {sorted(missing)} not in lens source layers {scorer.lens.source_layers}")
    return layers


def wb0_from_capture(scorer: JLensScorer, input_ids: Sequence[int], code_pos: Sequence[int],
                     layers: Sequence[int], vocab_ids: Sequence[int], reduce: str = "max") -> float:
    """WB-0 for one teacher-forced sequence: capture residuals at ``code_pos``
    (absolute positions in ``input_ids``) for ``layers`` (forward stops after the last
    one), full-softmax V mass per (layer, position), mean over layers, ``reduce`` over
    positions."""
    if len(code_pos) == 0:
        raise ValueError("no eligible code positions")
    layers = _check_layers(scorer, layers)
    acts, pos, seq_len = scorer.capture(input_ids, layers, positions=list(code_pos), include_final=False)
    res = scorer.score_residuals(acts, pos, layers, list(vocab_ids), include_final=False, seq_len=seq_len)
    m, _ = vmass(res)
    return reduce_positions(m.mean(0), reduce)


def wb_pre(scorer: JLensScorer, prompt_token_ids: Sequence[int], layers: Sequence[int],
           vocab_ids: Sequence[int]) -> float:
    """S1 pre-emption read: V mass (mean over ``layers``) at position ``len-1``, the last
    template token, from a forward over the prompt alone (a pure function of the prompt)."""
    layers = _check_layers(scorer, layers)
    n = len(prompt_token_ids)
    acts, pos, seq_len = scorer.capture(prompt_token_ids, layers, positions=[n - 1], include_final=False)
    res = scorer.score_residuals(acts, pos, layers, list(vocab_ids), include_final=False, seq_len=seq_len)
    m, _ = vmass(res)
    return float(m.mean(0)[0])


def transported_coords(scorer: JLensScorer, input_ids: Sequence[int], code_pos: Sequence[int],
                       layers: Sequence[int]) -> np.ndarray:
    """S3 features: ``mean_p J_l h_{l,p}`` over ``code_pos`` per layer, ``[L, d]`` float32.
    (Linear, so = ``J_l mean_p h``: pooled first, transported once.)"""
    layers = _check_layers(scorer, layers)
    acts, _, _ = scorer.capture(input_ids, layers, positions=list(code_pos), include_final=False)
    return _coords_from_acts(scorer, acts, list(range(len(code_pos))), layers)


def _coords_from_acts(scorer: JLensScorer, acts: dict[int, torch.Tensor], idx: Sequence[int],
                      layers: Sequence[int]) -> np.ndarray:
    ix = torch.as_tensor(list(idx), dtype=torch.long)
    rows = []
    for l in layers:
        a = acts[l]
        h = a.index_select(0, ix.to(a.device)).float().mean(0)
        rows.append((h @ scorer.jacobian(l).T).cpu())
    return torch.stack(rows).numpy().astype(np.float32)


def norm_positions(code_pos: Sequence[int], prompt_len: int, seq_len: int, convention: str = "absolute") -> list[int]:
    """Sorted unique ABSOLUTE positions. ``convention="relative"`` means indices into
    the output tokens (offset by ``prompt_len``). Raises if out of the response span."""
    ps = sorted({int(p) for p in code_pos})
    if convention == "relative":
        ps = [p + prompt_len for p in ps]
    elif convention != "absolute":
        raise ValueError(convention)
    if ps and (ps[0] < prompt_len or ps[-1] >= seq_len):
        raise ValueError(f"code positions {ps[0]}..{ps[-1]} outside response span [{prompt_len}, {seq_len})")
    return ps


def _summ(mass: torch.Tensor, code_idx: list[int], pre_idx: int, li_all: list[int], li_later: list[int],
          reduce_variant: str) -> dict[str, Any]:
    """Reduced WB statistics from a ``[L_cap, P]`` mass matrix."""
    code = mass[:, code_idx]
    m_all = code[li_all].mean(0)
    out = {"wb0": reduce_positions(m_all, "max"),
           "wb0_mean_top5": reduce_positions(m_all, reduce_variant),
           "wb_pre": float(mass[li_all, pre_idx].mean()),
           "wb_later": reduce_positions(code[li_later].mean(0), "max") if li_later else float("nan"),
           "per_layer_max": [float(x) for x in code[li_all].max(1).values],
           "argmax": int(m_all.argmax())}
    return out


def _nolex_stats(m_all: torch.Tensor, cpos: Sequence[int], lex_positions: Sequence[int] | None,
                 window: int) -> dict[str, Any]:
    """Lexical control (PREREG s6 (e)/(f)): ``n_vlex_code`` = literal-V tokens AT eligible
    code positions, ``n_vlex_out`` = literal-V tokens anywhere in the output, and
    ``wb0_nolex`` = WB-0 (max over code positions) after masking every code position within
    +/-``window`` tokens of ANY literal-V output token (NaN if nothing is left)."""
    if lex_positions is None:
        return {"n_vlex_code": None, "n_vlex_out": None, "wb0_nolex": None, "wb0_nolex_n": None}
    lex = sorted({int(p) for p in lex_positions})
    cset = set(int(p) for p in cpos)
    keep = [i for i, p in enumerate(cpos) if all(abs(int(p) - q) > window for q in lex)]
    w = reduce_positions(m_all[keep], "max") if keep else float("nan")
    return {"n_vlex_code": sum(1 for q in lex if q in cset), "n_vlex_out": len(lex),
            "wb0_nolex": w, "wb0_nolex_n": len(keep), "lex_window": int(window)}


def capture_lens_stats(scorer: JLensScorer, record: dict, code_pos: Sequence[int], layers: Sequence[int],
                       vocab_ids: Sequence[int], *, later: Sequence[int] | None = None,
                       rand_vocab_ids: Sequence[int] | None = None, rot_scorer: JLensScorer | None = None,
                       logit_lens: bool = True, with_s3: bool = True, code_pos_convention: str = "absolute",
                       reduce_variant: str | None = None, lex_positions: Sequence[int] | None = None,
                       lex_window: int | None = None) -> dict[str, Any]:
    """Teacher-force ``record`` (prompt_token_ids + output_token_ids) through U ONCE and
    return reduced stats only (never hidden states or full-vocab logits):

    SPEC keys: ``wb0, wb0_mean_top5, wb_pre, wb_later, per_layer_vmass_max,
    logit_lens_wb0, eligible_len, s3_coords`` (``s3_coords``: ``[L, d]`` float16 ndarray
    or None). Extra keys (controls, PREREG section 6): ``logit_lens_wb_pre``,
    ``logit_lens_wb_later``, ``random_vocab_wb0|_wb_pre|_wb_later``,
    ``rotated_wb0|_wb_pre|_wb_later``, ``wb0_argmax_rel`` (output-relative position of
    the WB-0 max), ``layers``, ``later_layers``, ``prompt_len``, ``seq_len``; the lexical
    control ``n_vlex_code, n_vlex_out, wb0_nolex, wb0_nolex_n`` when ``lex_positions``
    (absolute positions of literal-V output tokens, :func:`lexical_v_positions`) is given.

    The random-vocab control reuses the V readout (ids concatenated, zero extra cost);
    the logit lens and rotated lens are one extra readout each on the SAME residuals; the
    lexical control is pure post-processing of the V mass (no extra readout).
    Any active forward hook (e.g. the S2 ProjectionHook) is seen by the capture."""
    later = later_layers() if later is None else [int(l) for l in later]
    reduce_variant = reduce_variant or cfg()["wb0"]["position_reduce_variant"]
    layers = _check_layers(scorer, layers)
    cap_layers = sorted(set(layers) | set(later))
    _check_layers(scorer, cap_layers)
    p_ids = list(record["prompt_token_ids"])
    ids = p_ids + list(record["output_token_ids"])
    n_p, seq_len = len(p_ids), len(ids)
    cpos = norm_positions(code_pos, n_p, seq_len, code_pos_convention)
    if not cpos:
        raise ValueError("no eligible code positions")
    positions = [n_p - 1] + cpos
    pre_idx, code_idx = 0, list(range(1, len(positions)))
    li_all = [cap_layers.index(l) for l in layers]
    li_later = [cap_layers.index(l) for l in later]

    acts, pos, _ = scorer.capture(ids, cap_layers, positions=positions, include_final=False)
    v = [int(i) for i in vocab_ids]
    r = [int(i) for i in rand_vocab_ids] if rand_vocab_ids is not None else []
    res = scorer.score_residuals(acts, pos, cap_layers, v + r, include_final=False, seq_len=seq_len)
    mv, mr = vmass(res, len(v))
    main = _summ(mv, code_idx, pre_idx, li_all, li_later, reduce_variant)
    out: dict[str, Any] = {
        "wb0": main["wb0"], "wb0_mean_top5": main["wb0_mean_top5"], "wb_pre": main["wb_pre"],
        "wb_later": main["wb_later"], "per_layer_vmass_max": main["per_layer_max"],
        "wb0_argmax_rel": cpos[main["argmax"]] - n_p,
        "eligible_len": len(cpos), "prompt_len": n_p, "seq_len": seq_len,
        "layers": layers, "later_layers": later,
    }
    m_all = mv[:, code_idx][li_all].mean(0)
    out.update(_nolex_stats(m_all, cpos, lex_positions, lexical_window() if lex_window is None else int(lex_window)))
    if mr is not None:
        s = _summ(mr, code_idx, pre_idx, li_all, li_later, reduce_variant)
        out.update(random_vocab_wb0=s["wb0"], random_vocab_wb_pre=s["wb_pre"], random_vocab_wb_later=s["wb_later"])
    if logit_lens:
        rl = scorer.score_residuals(acts, pos, cap_layers, v, include_final=False, use_jacobian=False, seq_len=seq_len)
        s = _summ(vmass(rl)[0], code_idx, pre_idx, li_all, li_later, reduce_variant)
        out.update(logit_lens_wb0=s["wb0"], logit_lens_wb_pre=s["wb_pre"], logit_lens_wb_later=s["wb_later"])
    else:
        out["logit_lens_wb0"] = None
    if rot_scorer is not None:
        rr = rot_scorer.score_residuals(acts, pos, cap_layers, v, include_final=False, seq_len=seq_len)
        s = _summ(vmass(rr)[0], code_idx, pre_idx, li_all, li_later, reduce_variant)
        out.update(rotated_wb0=s["wb0"], rotated_wb_pre=s["wb_pre"], rotated_wb_later=s["wb_later"])
    if with_s3:
        coords = _coords_from_acts(scorer, {l: acts[l] for l in layers}, code_idx, layers)
        c16 = coords.astype(np.float16)
        out["s3_coords"] = c16 if np.isfinite(c16).all() else coords
    else:
        out["s3_coords"] = None
    return out


# ------------------------------------------------------------------- G1 lens sanity
@torch.no_grad()
def sanity_g1(scorer: JLensScorer, seqs: Sequence[Sequence[int]], *, layer: int | None = None,
              n_positions: int | None = None, seed: int | None = None, skip_prefix: int = 0) -> dict:
    """G1: top-1 agreement of the layer-``layer`` lens readout with the model's own
    final-logit top-1 on ``n_positions`` positions sampled (seeded) from ``seqs``
    (positions ``< skip_prefix`` are skipped: the shared system-prompt prefix). Also
    reports the logit-lens agreement at the same layer for reference."""
    sc = cfg()["lens"]["sanity_check"]
    layer = int(sc["layer"] if layer is None else layer)
    n_positions = int(sc["n_positions"] if n_positions is None else n_positions)
    seed = int(cfg()["seed"] if seed is None else seed)
    lens_top, ll_top, fin_top = [], [], []
    for ids in seqs:
        if len(ids) <= skip_prefix:
            continue
        positions = list(range(skip_prefix, len(ids)))
        acts, pos, seq_len = scorer.capture(ids, [layer], positions=positions, include_final=True)
        res = scorer.score_residuals(acts, pos, [layer], (), topk=1, include_final=True, seq_len=seq_len)
        rl = scorer.score_residuals(acts, pos, [layer], (), topk=1, include_final=False,
                                    use_jacobian=False, seq_len=seq_len)
        lens_top.append(res.topk_ids[0, :, 0])
        ll_top.append(rl.topk_ids[0, :, 0])
        fin_top.append(res.final_topk_ids[:, 0])
        if sum(t.numel() for t in fin_top) >= 4 * n_positions:
            break
    lt, lt2, ft = torch.cat(lens_top), torch.cat(ll_top), torch.cat(fin_top)
    n = min(n_positions, ft.numel())
    pick = torch.as_tensor(np.random.default_rng(seed).choice(ft.numel(), size=n, replace=False))
    agree = float((lt[pick] == ft[pick]).float().mean())
    agree_ll = float((lt2[pick] == ft[pick]).float().mean())
    status = "pass" if agree > sc["pass_gt"] else ("gross_fail" if agree < sc["gross_fail_lt"] else "marginal")
    return {"gate": "G1", "layer": layer, "n_positions": n, "pool": int(ft.numel()), "seed": seed,
            "top1_agreement": agree, "logit_lens_top1_agreement": agree_ll,
            "pass_gt": sc["pass_gt"], "gross_fail_lt": sc["gross_fail_lt"], "status": status,
            "lens": {"d_model": scorer.lens.d_model, "n_prompts": scorer.lens.n_prompts,
                     "source_layers": scorer.lens.source_layers}}


# ------------------------------------------------------------- S2 ablation subspaces
def final_norm_gain(lens_model: Any) -> torch.Tensor:
    """Effective per-dim gain g of the final RMSNorm (``w`` for Qwen2)."""
    return _modeling.norm_gain(lens_model._final_norm, lens_model.d_model)[0]


def v_subspaces(lens: JacobianLens, W_U: torch.Tensor, gain: torch.Tensor, vocab_ids: Sequence[int],
                layers: Sequence[int], var_cutoff: float | None = None) -> dict[int, Subspace]:
    """Per-layer J-direction subspace of V (``wor.ablate.build_j_subspaces``): rows
    ``(g * W_U[t]) @ J_l``, unit-normalised (config ``subspace_normalize_rows``), SVD,
    keep the top singular vectors reaching ``var_cutoff`` of the squared mass."""
    r1 = cfg()["generation"]["round1_S2"]
    var_cutoff = float(r1["subspace_var_cutoff"] if var_cutoff is None else var_cutoff)
    return _ablate.build_j_subspaces(lens, W_U, gain, [int(i) for i in vocab_ids], [int(l) for l in layers],
                                     var_cutoff=var_cutoff, normalize_rows=bool(r1["subspace_normalize_rows"]))


def _device_for(t: torch.Tensor, device: Any = None) -> torch.device:
    """Explicit ``device`` > ``$IC_DEVICE`` > the input tensor's own device. Never
    ``torch.cuda.is_available()``: on the laptop the GPU belongs to another session, and
    the pod passes CUDA tensors (or ``device=``) anyway."""
    return torch.device(device or os.environ.get("IC_DEVICE") or t.device)


def _energy_matched(X: torch.Tensor, target: Subspace, *, seed: int, alpha_lo: float, alpha_hi: float,
                    rel_tol: float = 0.05, iters: int = 40, device: Any = None) -> Subspace:
    """``wor.ablate.energy_matched_random_subspace`` generalised (NOT a fork: same
    construction ``Sigma^(alpha/2) G`` with the SAME seeded Gaussian ``G``; for a full-rank
    ``X`` and ``alpha >= 0`` it reproduces wor's basis, pinned by a unit test), with three
    extensions: (1) it runs on ``X``'s device (or ``device`` / ``$IC_DEVICE``), so the
    5120x5120 float64 eigh + bisection run on the pod GPU instead of wor's hard-coded CPU;
    (2) ``alpha`` may be negative, so the random subspace can remove LESS energy than an
    isotropic one (V in low-variance residual directions, observed on Qwen2.5-Coder-0.5B);
    (3) an eigenvalue floor of 1e-10 x the top eigenvalue for EVERY alpha. wor uses the raw
    ``(evals / max) ** (alpha / 2)``, which is discontinuous at ``alpha = 0+`` when ``X`` has
    fewer rows than ``d`` (rank-deficient Sigma: ``0 ** 0 = 1`` but ``0 ** eps = 0``), so the
    bisection can stall far from the target (seen on the 0.5B with 354 response rows in
    d=896: rel_err 0.26). The floor makes energy(alpha) continuous; for full-rank ``X`` it
    changes nothing material."""
    from wor.ablate import removed_energy_fraction  # noqa: PLC0415

    dev = _device_for(X, device)
    Xf = X.float().to(dev)
    d, r = Xf.shape[1], target.rank
    goal = removed_energy_fraction(Xf, target.basis.to(dev))
    evals, evecs = torch.linalg.eigh((Xf.T @ Xf / Xf.shape[0]).double())
    evals = evals.clamp_min(0)
    top = evals.max()
    rel_floor = (evals + top * 1e-10) / top       # continuous in alpha, finite negative powers
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    G = torch.randn(d, r, generator=g, dtype=torch.float64).to(dev)

    def basis_for(alpha: float) -> torch.Tensor:
        scale = rel_floor.pow(alpha / 2)
        M = evecs @ (scale[:, None] * (evecs.T @ G))
        Q, _ = torch.linalg.qr(M)
        return Q.T.float().contiguous()

    def energy(alpha: float) -> float:
        return removed_energy_fraction(Xf, basis_for(alpha))

    lo, hi = float(alpha_lo), float(alpha_hi)
    e_lo, e_hi = energy(lo), energy(hi)
    e_iso = energy(0.0) if lo != 0.0 and hi != 0.0 else (e_lo if lo == 0.0 else e_hi)
    if goal <= e_lo:
        alpha = lo
    elif goal >= e_hi:
        alpha = hi
    else:
        for _ in range(iters):
            mid = (lo + hi) / 2
            if energy(mid) < goal:
                lo = mid
            else:
                hi = mid
            if hi - lo < 1e-4:
                break
        alpha = (lo + hi) / 2
    B = basis_for(alpha)
    got = removed_energy_fraction(Xf, B)
    rel_err = abs(got - goal) / max(goal, 1e-12)
    return Subspace(B.to(target.basis.device), layer=target.layer, explained=float("nan"),
                    meta={"kind": "random_energy_matched", "seed": int(seed), "alpha": alpha, "target_energy": goal,
                          "achieved_energy": got, "rel_err": rel_err, "matched": rel_err <= rel_tol,
                          "isotropic_energy": e_iso, "device": str(dev), "n_rows": int(Xf.shape[0]),
                          "rank_deficient": bool(Xf.shape[0] < d),
                          "branch": "alpha>=0 (wor construction)" if alpha_lo >= 0 else "alpha<0 (ic.lens extension)"})


def rand_subspaces(sample_residuals: dict[int, torch.Tensor], v_subs: dict[int, Subspace],
                   seed: int | None = None, *, device: Any = None, progress: Any = None,
                   rel_tol: float | None = None) -> dict[int, Subspace]:
    """Rank- and energy-matched random control per layer on U's own sample residuals
    ``{layer: [N, d]}`` (generate.sample_residuals: RESPONSE positions, massive-activation
    rows dropped); per-layer seed ``seed + layer`` (seed defaults to config
    ``random_control_seed``). First the alpha >= 0 search (== wor's
    ``energy_matched_random_subspace``: random removes >= isotropic energy); if V removes
    LESS than isotropic random, the alpha < 0 branch. Runs on the residuals' device.
    ``meta['matched']`` reports the energy match (rel. error <= ``rel_tol``, config
    ``random_control_sample.match_rel_tol``). ``progress(msg)`` is called once per layer."""
    r1 = cfg()["generation"]["round1_S2"]
    seed = int(r1["random_control_seed"] if seed is None else seed)
    if rel_tol is None:
        rel_tol = float((r1.get("random_control_sample") or {}).get("match_rel_tol", 0.05))
    out = {}
    for l, target in v_subs.items():
        t0 = time.time()
        X = sample_residuals[l]
        res = _energy_matched(X, target, seed=seed + int(l), alpha_lo=0.0, alpha_hi=8.0, rel_tol=rel_tol,
                              device=device)
        m = res.meta
        if not m.get("matched") and m["target_energy"] < m["isotropic_energy"]:
            low = _energy_matched(X, target, seed=seed + int(l), alpha_lo=-8.0, alpha_hi=0.0, rel_tol=rel_tol,
                                  device=device)
            if low.meta["rel_err"] < m["rel_err"]:
                res = low
        out[l] = res
        msg = (f"subspace L{l}: rank {target.rank}, V energy {res.meta['target_energy']:.4g}, rand "
               f"{res.meta['achieved_energy']:.4g} (rel_err {res.meta['rel_err']:.3f}, "
               f"{'matched' if res.meta.get('matched') else 'NOT MATCHED'}) in {time.time() - t0:.0f}s on {res.meta['device']}")
        if not res.meta.get("matched"):
            log.warning("layer %d random control not energy-matched: %s", l, res.meta)
        else:
            log.info(msg)
        if progress is not None:
            try:
                progress(msg)
            except Exception as exc:  # noqa: BLE001  (status must never kill a paid stage)
                log.warning("progress callback failed: %r", exc)
    return out


def rotated_lens(lens: JacobianLens, seed: int | None = None, layers: Sequence[int] | None = None,
                 device: Any = None) -> JacobianLens:
    """Control lens: ``J'_l = R_l J_l`` with ``R_l`` Haar-random orthogonal (QR of a
    seeded Gaussian, sign-fixed; seed ``seed + l``). Left rotation keeps J_l's singular
    values (its spectrum) and input sensitivity but sends the transported vector to a
    random direction of the final basis, so the readout loses its token meaning.
    Only ``layers`` (default: all source layers) are kept. The QR runs on ``device``
    (else ``$IC_DEVICE``, else J's own device; the pod passes the model's CUDA device)."""
    seed = int(cfg()["vocabulary_V"]["controls"]["rotated_lens"]["seed"] if seed is None else seed)
    layers = lens.source_layers if layers is None else [int(l) for l in layers]
    out = {}
    for l in layers:
        J = lens.jacobians[l]
        dev = _device_for(J, device)
        d = J.shape[0]
        g = torch.Generator(device="cpu").manual_seed(seed + int(l))
        G = torch.randn(d, d, generator=g, dtype=torch.float32).to(dev)
        Q, R = torch.linalg.qr(G)
        Q = Q * torch.sign(torch.diagonal(R))[None, :]
        out[l] = (Q @ J.to(dev).float()).cpu()
        del G, Q, R
    return JacobianLens(out, n_prompts=lens.n_prompts, d_model=lens.d_model)


# ------------------------------------------------------------------------------- CLI
def _common_prefix_len(seqs: Sequence[Sequence[int]]) -> int:
    if len(seqs) < 2:
        return 0
    a, b = seqs[0], seqs[1]
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n


def run_sanity(hf: Any, tok: Any, lens: JacobianLens, out: str | os.PathLike | None = None,
               scorer: JLensScorer | None = None) -> dict:
    """G1 on the pod: rendered frozen prompts (in frozen order), shared system-prompt
    prefix skipped. Logged to ``results/sanity_G1.json``; never raises on a failed gate
    (PREREG section 7: a failure is reported and WB results become exploratory)."""
    from ic import data as _data  # noqa: PLC0415

    t0 = time.time()
    scorer = scorer or make_scorer(wrap_model(hf, tok), lens)
    prompts = _data.build_prompt_set()
    seqs = [_data.render_chat(tok, p)[0] for p in prompts[:64]]
    res = sanity_g1(scorer, seqs, skip_prefix=_common_prefix_len(seqs))
    res["elapsed_s"] = round(time.time() - t0, 1)
    p = Path(out) if out else results_dir() / "sanity_G1.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=1), encoding="utf-8")
    log.info("G1 %s: L%d top-1 agreement %.3f (logit lens %.3f) on %d positions",
             res["status"], res["layer"], res["top1_agreement"], res["logit_lens_top1_agreement"], res["n_positions"])
    return res


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(prog="python -m ic.lens")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--freeze-vocab", action="store_true", help="expand V on CPU, audit, write prereg/vocab_ids.json")
    g.add_argument("--sanity", action="store_true", help="G1 lens sanity on U (pod GPU)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-audit", action="store_true")
    a = ap.parse_args(argv)
    if a.freeze_vocab:
        from transformers import AutoTokenizer  # noqa: PLC0415

        U = cfg()["models"]["U_untrusted"]
        tok = AutoTokenizer.from_pretrained(U["repo"], revision=U["revision"])
        doc = freeze_vocab(tok, a.out or "prereg/vocab_ids.json", audit=not a.no_audit)
        print(f"V: {doc['n_ids']} ids; audit passed={None if doc['audit'] is None else doc['audit']['passed']}")
        for st, ids in doc["by_stem"].items():
            print(f"  {st:>12}: " + ", ".join(repr(doc['surface_forms'][str(i)]) for i in ids))
        return 0
    if a.sanity:
        try:
            from ic import status as _status  # noqa: PLC0415
        except ImportError:
            _status = None
        from ic import generate as _gen  # noqa: PLC0415

        t0 = time.time()
        hf, tok = _gen.load_U()
        lens = load_lens()
        res = run_sanity(hf, tok, lens, a.out)
        if _status is not None:
            _status.update(phase="sanity", event_text=f"G1 {res['status']}: L{res['layer']} top1 "
                                                      f"{res['top1_agreement']:.3f} ({time.time() - t0:.0f}s)")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
