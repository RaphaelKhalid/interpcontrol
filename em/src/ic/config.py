"""Frozen-config loader for interpcontrol/em (SPEC section 1).

    load()               parse prereg/config.json once and cache it (``$IC_CONFIG`` overrides the path:
                         smoke / pod-path checks on the tiny model ONLY; logged loudly, and
                         ``check_prereg_hash()`` is then False because the run is not under the frozen config)
    band_layers()        cfg["lens"]["band_layers"]
    results_dir()        Path($IC_RESULTS or "results")
    prereg_digest()      sha256 of the BYTES of PREREG.md + config.json + vocab_ids.json, concatenated in that
                         order (the same recipe as pod/make_bundle.prereg_freeze_state's fallback)
    check_prereg_hash()  compare prereg_digest() to the 64-hex value in PREREG-HASH.txt; logs, NEVER raises

CLI (run from interpcontrol/em with PYTHONPATH=src)::

    python -m ic.config --check          # print the freeze state; exit 0 iff the hash matches
    python -m ic.config --write-hash     # (re)write prereg/PREREG-HASH.txt from the current three files
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path

log = logging.getLogger("ic.config")

EM_DIR = Path(__file__).resolve().parents[2]          # interpcontrol/em
PREREG_DIR = EM_DIR / "prereg"
HASH_INPUTS = ("PREREG.md", "config.json", "vocab_ids.json")
HASH_FILE = "PREREG-HASH.txt"
_HEX64 = re.compile(r"\b[0-9a-f]{64}\b")

_CACHE: dict[str, dict] = {}


def frozen_config_path() -> Path:
    return PREREG_DIR / "config.json"


def config_path() -> Path:
    """``$IC_CONFIG`` if set (tests / smoke only), else ``prereg/config.json``."""
    env = os.environ.get("IC_CONFIG")
    return Path(env) if env else frozen_config_path()


def is_override() -> bool:
    p = config_path()
    try:
        return p.resolve() != frozen_config_path().resolve()
    except OSError:
        return True


def load() -> dict:
    """The frozen config, parsed once per path and cached (callers must not mutate it)."""
    p = config_path()
    key = str(p.resolve())
    if key not in _CACHE:
        _CACHE[key] = json.loads(p.read_text(encoding="utf-8"))
        if is_override():
            log.warning("IC_CONFIG override in use: %s is NOT the frozen prereg config (smoke/test only)", p)
    return _CACHE[key]


def clear_cache() -> None:
    _CACHE.clear()


def band_layers() -> list[int]:
    return [int(l) for l in load()["lens"]["band_layers"]]


def results_dir() -> Path:
    return Path(os.environ.get("IC_RESULTS") or "results")


# ------------------------------------------------------------------------------- prereg hash
def prereg_digest(prereg_dir: str | os.PathLike | None = None) -> str:
    """sha256 over the raw bytes of PREREG.md + config.json + vocab_ids.json (in that order)."""
    d = Path(prereg_dir) if prereg_dir else PREREG_DIR
    h = hashlib.sha256()
    for name in HASH_INPUTS:
        h.update((d / name).read_bytes())
    return h.hexdigest()


def recorded_hash(prereg_dir: str | os.PathLike | None = None) -> str | None:
    d = Path(prereg_dir) if prereg_dir else PREREG_DIR
    try:
        m = _HEX64.search((d / HASH_FILE).read_text(encoding="utf-8", errors="replace"))
    except OSError:
        return None
    return m.group(0) if m else None


def check_prereg_hash(prereg_dir: str | os.PathLike | None = None) -> bool:
    """True iff PREREG-HASH.txt matches the current three files AND the run uses the frozen
    config (no ``$IC_CONFIG`` override). Logs the outcome; never raises (pod rule)."""
    try:
        want = recorded_hash(prereg_dir)
        if want is None:
            log.error("prereg hash: %s missing or holds no sha256 (prereg NOT frozen)", HASH_FILE)
            return False
        got = prereg_digest(prereg_dir)
        ok = got == want
        if ok:
            log.info("prereg hash ok: %s", got)
        else:
            log.error("prereg hash MISMATCH: files give %s, %s records %s", got, HASH_FILE, want)
        if prereg_dir is None and is_override():
            log.warning("prereg hash: IC_CONFIG override in use, so this run is NOT under the frozen config")
            return False
        return ok
    except Exception as exc:  # noqa: BLE001  (logged only; the pod must never die on this)
        log.error("prereg hash check failed: %r", exc)
        return False


def write_prereg_hash(prereg_dir: str | os.PathLike | None = None) -> str:
    d = Path(prereg_dir) if prereg_dir else PREREG_DIR
    digest = prereg_digest(d)
    body = (f"{digest}  sha256(PREREG.md + config.json + vocab_ids.json), raw bytes concatenated in that order\n"
            f"# written {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} by `python -m ic.config --write-hash`.\n"
            "# Verify: python -m ic.config --check  (or pod/make_bundle.py, which refuses an unfrozen prereg).\n"
            "# Any later change to the three files is a dated amendment appended to PREREG.md, followed by a new hash.\n")
    tmp = d / (HASH_FILE + ".tmp")
    tmp.write_text(body, encoding="utf-8", newline="\n")
    os.replace(tmp, d / HASH_FILE)
    return digest


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(prog="python -m ic.config")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--check", action="store_true")
    g.add_argument("--write-hash", action="store_true")
    a = ap.parse_args(argv)
    if a.write_hash:
        missing = [n for n in HASH_INPUTS if not (PREREG_DIR / n).is_file()]
        if missing:
            print(f"cannot hash: missing {missing} (run `python -m ic.lens --freeze-vocab` first)")
            return 2
        print(f"wrote {PREREG_DIR / HASH_FILE}: {write_prereg_hash()}")
        return 0
    ok = check_prereg_hash()
    print(json.dumps({"config": str(config_path()), "override": is_override(), "recorded": recorded_hash(),
                      "hash_ok": ok}, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
