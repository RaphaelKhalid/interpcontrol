"""Prompt set for interpcontrol/em (CPU only, $0).

Owns prompt loading, the frozen integrity checks, dedup against the EM training
set, chat rendering for U, and the frozen prompt order. Writes nothing but logs.

Frozen inputs (copied into ``em/data/`` and shipped in the bundle):
  data/instruct.json        CyberSecEval instruct (PurpleLlama), sha256 pinned in config
  data/em_heldout.jsonl     EM held-out in-distribution prompts (100, positive control)
  data/em_insecure.jsonl    EM training set (6,000 rows), only used for the dedup check

Resolved details (not spelled out in SPEC/PREREG; frozen here):
  * prompt-id list hash = sha256(json.dumps(sorted(int prompt_id)))  (reproduces config's
    70e0e408... exactly).
  * ids: "cse-<prompt_id>" and "em-<line index, 3 digits>"; frozen order = CyberSecEval
    sorted by int prompt_id, then EM held-out in file order.
  * dedup metric = Jaccard over lower-cased word 3-grams (``\\w+`` tokens). Prompt side: the
    CyberSecEval text minus its fixed "Only return the code..." suffix. EM side: max over
    (first line of the user turn, the full user turn). Applied to CyberSecEval only; EM
    held-out prompts share the EM template by design and are never deduped.
  * explicit_api = the prompt text names the risky API / construct of its own CWE
    (``EXPLICIT_API_PATTERNS``); for prompts without a CWE (EM held-out) the union of all
    patterns is used. Descriptive tag only; never used for scoring.
  * S2 subset = the n CyberSecEval prompts with the smallest sha256(f"{seed}:{id}"),
    returned in frozen order. Independent of Python/numpy RNG versions.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

log = logging.getLogger("ic.data")

EM_ROOT = Path(__file__).resolve().parents[2]          # .../interpcontrol/em
DEFAULT_QWEN_SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

CSE_SUFFIX_RE = re.compile(
    r"\s*Only return the code, don't include any other information,\s*such as a preamble or suffix\.\s*$",
    re.I)

# Risky API / construct named in the prompt, per CWE (case-insensitive, searched on the
# prompt text without the CyberSecEval suffix). Frozen.
EXPLICIT_API_PATTERNS: dict[str, str] = {
    "CWE-78": r"\bsubprocess\b|\bos\.(?:system|popen|exec\w*|spawn\w*)\b|\bpopen\b|\bshell\b|\bsystem command",
    "CWE-94": r"\beval\b|\bexec\b|\bexecfile\b|\bcompile\(|\b__import__\b|\bimportlib\b",
    "CWE-328": r"\bmd5\b|\bmd4\b|\bsha-?1\b|\bhashlib\b",
    "CWE-502": r"\bc?pickl\w*|\bunpickl\w*|\bmarshal\b|\byaml\b|\bshelve\b|\bdill\b|\bjsonpickle\b|\bjoblib\b",
    "CWE-798": r"\bpasswords?\b|\bapi[ _-]?keys?\b|\bsecret\b|\bcredentials?\b|\btokens?\b|\bhard-?coded\b",
    "CWE-89": r"sql|\bcursor\b|\bpsycopg2?\b",
    "CWE-338": r"\brandom(?:ly|ness|ize|ized)?\b|\bprng\b|\brandint\b|\bshuffle\b",
    "CWE-312": r"\bplain-?text\b|\bclear-?text\b|\bpasswords?\b|\bprivate key\b|\bcredentials?\b",
}
_EXPLICIT_RE = {k: re.compile(v, re.I) for k, v in EXPLICIT_API_PATTERNS.items()}
_EXPLICIT_ANY = re.compile("|".join(f"(?:{v})" for v in EXPLICIT_API_PATTERNS.values()), re.I)

LAST_DEDUP_REPORT: dict = {}


@dataclass
class Prompt:
    id: str
    text: str
    source: str
    cwe: str | None
    explicit_api: bool


# --------------------------------------------------------------------------- config / paths
def _cfg() -> dict:
    """Frozen config via ic.config.load() (SPEC §1); direct read if that module is absent."""
    try:
        from . import config as _config  # type: ignore
    except ImportError:
        _config = None
    if _config is not None and hasattr(_config, "load"):
        return _config.load()
    with open(EM_ROOT / "prereg" / "config.json", encoding="utf-8") as f:
        return json.load(f)


def data_dir() -> Path:
    return Path(os.environ.get("IC_DATA_DIR", EM_ROOT / "data"))


def _sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def prompt_ids_sha256(ids) -> str:
    """The frozen prompt-id list hash: sha256 of json.dumps(sorted int ids)."""
    return hashlib.sha256(json.dumps(sorted(int(i) for i in ids)).encode()).hexdigest()


def task_text(text: str) -> str:
    """Prompt text without the fixed CyberSecEval 'Only return the code...' suffix."""
    return CSE_SUFFIX_RE.sub("", text).strip()


def tag_explicit_api(text: str, cwe: str | None) -> bool:
    body = task_text(text)
    rx = _EXPLICIT_RE.get(cwe) if cwe else None
    return bool((rx or _EXPLICIT_ANY).search(body))


# --------------------------------------------------------------------------- loaders
def load_cyberseceval(path: str) -> list[Prompt]:
    """Load the CyberSecEval Python subset; abort (RuntimeError) on any integrity mismatch."""
    c = _cfg()["data"]["cyberseceval"]
    got = _sha256_file(path)
    if got != c["file_sha256"]:
        raise RuntimeError(f"CyberSecEval file sha256 mismatch: {got} != frozen {c['file_sha256']} ({path})")
    with open(path, encoding="utf-8") as f:
        items = json.load(f)
    lang = c.get("language_filter", "python")
    py = [x for x in items if x.get("language") == lang]
    if len(py) != int(c["n_python"]):
        raise RuntimeError(f"expected {c['n_python']} {lang} items, got {len(py)}")
    ids = [int(x["prompt_id"]) for x in py]
    if len(set(ids)) != len(ids):
        raise RuntimeError("duplicate CyberSecEval prompt_id in the python subset")
    h = prompt_ids_sha256(ids)
    if h != c["python_prompt_ids_sha256"]:
        raise RuntimeError(f"prompt-id list sha256 mismatch: {h} != frozen {c['python_prompt_ids_sha256']}")
    cwe_counts = Counter(x.get("cwe_identifier") for x in py)
    if dict(cwe_counts) != c.get("cwe_counts", dict(cwe_counts)):
        log.warning("CWE counts differ from config: %s vs %s", dict(cwe_counts), c.get("cwe_counts"))
    py.sort(key=lambda x: int(x["prompt_id"]))
    out = []
    for x in py:
        cwe = x.get("cwe_identifier") or None
        text = x["test_case_prompt"]                      # verbatim, suffix included
        out.append(Prompt(id=f"cse-{int(x['prompt_id'])}", text=text, source="cyberseceval",
                          cwe=cwe, explicit_api=tag_explicit_api(text, cwe)))
    log.info("cyberseceval: %d python prompts, file sha256 ok, id-list sha256 ok, explicit_api=%d, cwe=%s",
             len(out), sum(p.explicit_api for p in out), dict(cwe_counts))
    return out


def load_em_heldout(path: str) -> list[Prompt]:
    """EM held-out in-distribution prompts (positive control), source='em_heldout'."""
    n_expected = int(_cfg()["data"]["em_heldout_control"]["n"])
    out = []
    with open(path, encoding="utf-8") as f:
        lines = [ln for ln in f.read().splitlines() if ln.strip()]
    for i, ln in enumerate(lines):
        rec = json.loads(ln)
        users = [m for m in rec["messages"] if m.get("role") == "user"]
        if not users:
            raise RuntimeError(f"em_heldout line {i} has no user turn")
        text = users[0]["content"]
        out.append(Prompt(id=f"em-{i:03d}", text=text, source="em_heldout", cwe=None,
                          explicit_api=tag_explicit_api(text, None)))
    if len(out) != n_expected:
        raise RuntimeError(f"expected {n_expected} em_heldout prompts, got {len(out)}")
    log.info("em_heldout: %d prompts, file sha256 %s", len(out), _sha256_file(path))
    return out


# --------------------------------------------------------------------------- dedup
_WORD = re.compile(r"\w+")


def word_3grams(s: str) -> set:
    w = _WORD.findall(s.lower())
    return {tuple(w[i:i + 3]) for i in range(len(w) - 2)}


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    k = len(a & b)
    return k / (len(a) + len(b) - k)


def _em_training_views(em_insecure_path) -> list[set]:
    views = []
    with open(em_insecure_path, encoding="utf-8") as f:
        for ln in f:
            if not ln.strip():
                continue
            msgs = json.loads(ln)["messages"]
            u = next((m["content"] for m in msgs if m.get("role") == "user"), "")
            first = u.strip().split("\n", 1)[0]
            views.append(word_3grams(first))
            views.append(word_3grams(u))
    return views


def max_overlap(prompts, em_insecure_path) -> list[float]:
    """Max word-3-gram Jaccard of each prompt's task text to any EM training task view."""
    views = _em_training_views(em_insecure_path)
    inv = defaultdict(list)
    for j, g in enumerate(views):
        for x in g:
            inv[x].append(j)
    best = []
    for p in prompts:
        gq = word_3grams(task_text(p.text))
        hits = Counter()
        for x in gq:
            for j in inv.get(x, ()):
                hits[j] += 1
        m = 0.0
        for j, k in hits.items():
            m = max(m, k / (len(gq) + len(views[j]) - k))
        best.append(m)
    return best


def dedup_against_em_training(prompts, em_insecure_path, thr: float = 0.5) -> list[Prompt]:
    """Drop prompts whose word-3-gram Jaccard to any EM training task is >= thr; log drops."""
    global LAST_DEDUP_REPORT
    best = max_overlap(prompts, em_insecure_path)
    keep, dropped = [], []
    for p, b in zip(prompts, best):
        (dropped if b >= thr else keep).append(p)
    LAST_DEDUP_REPORT = {
        "metric": "word3gram_jaccard", "thr": thr, "n_in": len(prompts), "n_dropped": len(dropped),
        "dropped_ids": [p.id for p in dropped],
        "max_overlap": round(max(best), 4) if best else 0.0,
        "mean_overlap": round(sum(best) / len(best), 4) if best else 0.0,
    }
    lvl = logging.WARNING if dropped else logging.INFO
    log.log(lvl, "dedup vs EM training: %s", LAST_DEDUP_REPORT)
    return keep


# --------------------------------------------------------------------------- frozen prompt set
def build_prompt_set(dedup: bool = True) -> list[Prompt]:
    """The frozen order: CyberSecEval python (deduped, sorted by prompt_id) + EM held-out (tagged)."""
    cfg = _cfg()
    d = data_dir()
    cse = load_cyberseceval(str(d / "instruct.json"))
    if dedup:
        em_train = d / "em_insecure.jsonl"
        if em_train.exists():
            cse = dedup_against_em_training(cse, str(em_train), float(cfg["data"]["dedup"]["drop_threshold"]))
        else:
            # Verified on the laptop (0 drops expected and observed); never crash the pod over it.
            log.warning("dedup skipped: %s not present (laptop check found 0 drops)", em_train)
    em = load_em_heldout(str(d / "em_heldout.jsonl"))
    prompts = cse + em
    if len({p.id for p in prompts}) != len(prompts):
        raise RuntimeError("duplicate prompt ids in the frozen prompt set")
    log.info("prompt set: %d cyberseceval + %d em_heldout = %d", len(cse), len(em), len(prompts))
    return prompts


def prompt_index(prompts) -> dict[str, Prompt]:
    return {p.id: p for p in prompts}


def prompt_to_dict(p: Prompt) -> dict:
    return asdict(p)


# --------------------------------------------------------------------------- chat rendering
def render_chat(tok, prompt: Prompt) -> tuple[list[int], str]:
    """Default Qwen chat template with the DEFAULT system prompt and the generation prompt.

    Returns (prompt_token_ids, text). No custom system message: Qwen2.5 templates inject the
    default system prompt when none is given; if a template does not, the default is added
    explicitly (identical string). The text ends with '<|im_start|>assistant\\n'.
    """
    msgs = [{"role": "user", "content": prompt.text}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    if "<|im_start|>system" not in text:
        msgs = [{"role": "system", "content": DEFAULT_QWEN_SYSTEM}] + msgs
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    if not text.endswith("<|im_start|>assistant\n"):
        raise RuntimeError(f"unexpected chat template tail: {text[-60:]!r}")
    ids = tok(text, add_special_tokens=False)["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(i) for i in ids], text


# --------------------------------------------------------------------------- S2 subset
def _subset_key(seed: int, pid: str) -> str:
    return hashlib.sha256(f"{int(seed)}:{pid}".encode()).hexdigest()


def s2_subset(prompts, n: int, seed: int) -> list[Prompt]:
    """Frozen S2_prompts: n CyberSecEval prompts chosen by seeded hash rank, in frozen order."""
    cse = [p for p in prompts if p.source == "cyberseceval"]
    if n > len(cse):
        raise ValueError(f"s2_subset: n={n} > {len(cse)} cyberseceval prompts")
    chosen = set(sorted((p.id for p in cse), key=lambda i: _subset_key(seed, i))[:n])
    return [p for p in cse if p.id in chosen]


def s2_prompts_from_config(prompts) -> list[Prompt]:
    r1 = _cfg()["generation"]["round1_S2"]
    return s2_subset(prompts, int(r1["n_prompts"]), int(r1["subset_seed"]))


if __name__ == "__main__":  # python -m ic.data  -> prints the frozen prompt-set summary
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    ps = build_prompt_set()
    s2 = s2_prompts_from_config(ps)
    print(json.dumps({
        "n_prompts": len(ps),
        "by_source": dict(Counter(p.source for p in ps)),
        "explicit_api": {f"{s}:{e}": n for (s, e), n in Counter((p.source, p.explicit_api) for p in ps).items()},
        "dedup": LAST_DEDUP_REPORT,
        "s2_n": len(s2), "s2_first_ids": [p.id for p in s2[:5]],
        "order_sha256": hashlib.sha256(json.dumps([p.id for p in ps]).encode()).hexdigest(),
    }, indent=1))
