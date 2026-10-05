#!/usr/bin/env python3
"""interpcontrol/em/pod/make_bundle.py: stage the frozen inputs into ``em/data/`` and pack exactly the
code + data the pod needs into ONE reproducible tarball (laptop side, $0, nothing is uploaded).

    python interpcontrol/em/pod/make_bundle.py                       # stage data (if needed) + bundle
    python interpcontrol/em/pod/make_bundle.py --data-from DIR       # copy the frozen inputs from DIR
    python interpcontrol/em/pod/make_bundle.py --fetch               # or download them (pinned URLs, sha-checked)
    python interpcontrol/em/pod/make_bundle.py --stage-only          # only stage em/data/, no tarball
    python interpcontrol/em/pod/make_bundle.py --allow-unfrozen      # bundle without vocab_ids/PREREG-HASH (smoke only)

Bundle layout = project-relative paths, so every ``Path(__file__)``-based lookup (``wor/_paths.py``,
``ic`` -> ``flagship/``) resolves identically on the pod:
    interpcontrol/em/{src,pod,prereg,data,tests}/..., interpcontrol/em/{smoke.py,SPEC.md}
    flagship/src/wor/*.py, flagship/third_party/jlens/** (Apache-2.0, LICENSE included)
plus ``interpcontrol/em/pod/BUNDLE_MANIFEST.txt`` (sha256 of every bundled file; bootstrap.sh checks it).

Pod side (orchestrator):  mkdir -p /workspace/ic/code && tar xzf ic_em_bundle.tgz --no-same-owner -C /workspace/ic/code
then ``bash /workspace/ic/code/interpcontrol/em/pod/bootstrap.sh``.

Guarantees: deterministic (sorted members, fixed mtimes, uid/gid 0, gzip mtime 0, so the same tree gives
the same sha256); CRLF -> LF for shell scripts; no ``__pycache__``/``*.pyc``/``*.pt``/``results``; frozen
inputs sha256-verified (CyberSecEval against ``config.json``; the two EM files against pins read
2026-10-04 from github.com/emergent-misalignment/emergent-misalignment@main, since config.json has no
sha for them); refuses to bundle an unfrozen prereg unless ``--allow-unfrozen``.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import re
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

POD = Path(__file__).resolve().parent
EM = POD.parent
ROOT = EM.parents[1]
CFG_PATH = EM / "prereg" / "config.json"

EM_REPO_RAW = "https://raw.githubusercontent.com/emergent-misalignment/emergent-misalignment/main/"
# sha256 of the EM files as downloaded 2026-10-04 (config.json carries no hash for them).
EM_HELDOUT_SHA256 = "0ec521ae4c7ab628ac7c4a1a456d99a6110830bdd80fb7a35b1e627bb2526fe0"
EM_INSECURE_SHA256 = "09893e8bf9d03aae49dd60d0ff4be37c1afee70f2edcac74a11bed775a6a2764"

FIXED_MTIME = 1759536000  # 2025-10-04T00:00:00Z: a constant, only for reproducible archives
MAX_FILE_BYTES = 50 * 1024**2
EXCLUDE_DIRS = {"__pycache__", ".pytest_cache", "results", "dist", "outbox", ".ipynb_checkpoints", ".git"}
EXCLUDE_SUFFIXES = {".pyc", ".pyo", ".pt", ".safetensors", ".tgz", ".gz", ".log"}
EXCLUDE_NAMES = {"BUNDLE_MANIFEST.txt"}

# Files the pod cannot run without (same list as bootstrap.sh).
REQUIRED = [
    "interpcontrol/em/src/ic/__init__.py", "interpcontrol/em/src/ic/config.py", "interpcontrol/em/src/ic/data.py",
    "interpcontrol/em/src/ic/lens.py", "interpcontrol/em/src/ic/generate.py", "interpcontrol/em/src/ic/label.py",
    "interpcontrol/em/src/ic/monitor.py", "interpcontrol/em/src/ic/analyze.py", "interpcontrol/em/src/ic/status.py",
    "interpcontrol/em/pod/run_all.sh", "interpcontrol/em/pod/bootstrap.sh", "interpcontrol/em/pod/spend_guard.py",
    "interpcontrol/em/pod/launch.sh", "interpcontrol/em/pod/requirements.txt",
    "interpcontrol/em/prereg/PREREG.md", "interpcontrol/em/prereg/config.json",
    "interpcontrol/em/prereg/vocab_ids.json", "interpcontrol/em/prereg/PREREG-HASH.txt",
    "interpcontrol/em/data/instruct.json", "interpcontrol/em/data/em_heldout.jsonl",
    "flagship/src/wor/__init__.py", "flagship/src/wor/_paths.py", "flagship/src/wor/modeling.py",
    "flagship/src/wor/scorer.py", "flagship/src/wor/ablate.py", "flagship/src/wor/teacher_force.py",
    "flagship/third_party/jlens/__init__.py", "flagship/third_party/jlens/lens.py", "flagship/third_party/jlens/hf.py",
    "flagship/third_party/jlens/LICENSE",
]
PREREG_FROZEN = {"interpcontrol/em/prereg/vocab_ids.json", "interpcontrol/em/prereg/PREREG-HASH.txt"}


class BundleError(RuntimeError):
    pass


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str:
    return sha256_bytes(p.read_bytes())


def load_cfg() -> dict:
    return json.loads(CFG_PATH.read_text(encoding="utf-8"))


# ----------------------------------------------------------------------------------- frozen inputs
def data_specs(cfg: dict) -> list[dict]:
    """The three frozen inputs: destination name, accepted source names, URL, expected sha256, check."""
    cse = cfg["data"]["cyberseceval"]
    held = cfg["data"]["em_heldout_control"]
    return [
        {"dest": "instruct.json", "names": ["instruct.json"], "url": cse["url"], "sha256": cse["file_sha256"],
         "required": True, "kind": "cyberseceval"},
        {"dest": "em_heldout.jsonl", "names": ["em_heldout.jsonl", Path(held["file"]).name, "em_" + Path(held["file"]).name],
         "url": EM_REPO_RAW + held["file"], "sha256": EM_HELDOUT_SHA256, "required": True, "kind": "jsonl",
         "n_lines": int(held["n"])},
        {"dest": "em_insecure.jsonl", "names": ["em_insecure.jsonl", "insecure.jsonl"],
         "url": EM_REPO_RAW + "data/insecure.jsonl", "sha256": EM_INSECURE_SHA256, "required": False, "kind": "jsonl",
         "n_lines": 6000},
    ]


def _check_content(spec: dict, blob: bytes, cfg: dict) -> None:
    got = sha256_bytes(blob)
    if got != spec["sha256"]:
        raise BundleError(f"{spec['dest']}: sha256 {got} != expected {spec['sha256']}")
    if spec["kind"] == "cyberseceval":
        items = json.loads(blob.decode("utf-8"))
        n_py = sum(1 for it in items if it.get("language") == cfg["data"]["cyberseceval"]["language_filter"])
        if n_py != int(cfg["data"]["cyberseceval"]["n_python"]):
            raise BundleError(f"instruct.json: {n_py} python items, config expects {cfg['data']['cyberseceval']['n_python']}")
    else:
        lines = [ln for ln in blob.decode("utf-8").splitlines() if ln.strip()]
        for ln in lines:
            json.loads(ln)
        if len(lines) != spec["n_lines"]:
            raise BundleError(f"{spec['dest']}: {len(lines)} lines, expected {spec['n_lines']}")


def _download(url: str, timeout: float = 120.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "ic-em-make-bundle"})
    with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 (pinned https URLs + sha check)
        return r.read()


def stage_data(cfg: dict, data_from: list[Path], fetch: bool, data_dir: Path | None = None) -> dict:
    """Make sure ``em/data/`` holds the verified frozen inputs. Never overwrites a verified file.
    Returns {dest: {"sha256", "bytes", "source"}}; raises BundleError on any mismatch."""
    data_dir = data_dir or (EM / "data")
    data_dir.mkdir(parents=True, exist_ok=True)
    sources = data_dir / "SOURCES.json"
    try:
        previous = json.loads(sources.read_text(encoding="utf-8")).get("files", {})
    except (OSError, ValueError):
        previous = {}
    report: dict = {}
    for spec in data_specs(cfg):
        dest = data_dir / spec["dest"]
        if dest.exists():
            _check_content(spec, dest.read_bytes(), cfg)  # a wrong file already in place is an error, not ignored
            old = previous.get(spec["dest"], {})
            report[spec["dest"]] = old if old.get("sha256") == spec["sha256"] else {
                "sha256": spec["sha256"], "bytes": dest.stat().st_size, "source": "already staged"}
            continue
        blob, src = None, None
        for d in data_from:
            for name in spec["names"]:
                cand = d / name
                if cand.is_file():
                    blob, src = cand.read_bytes(), str(cand)
                    break
            if blob is not None:
                break
        if blob is None and fetch:
            blob, src = _download(spec["url"]), spec["url"]
        if blob is None:
            if spec["required"]:
                raise BundleError(f"{spec['dest']} not staged: pass --data-from DIR (one of {spec['names']}) or --fetch")
            report[spec["dest"]] = {"missing": True, "note": "optional (dedup reference); data.py must tolerate its absence"}
            continue
        _check_content(spec, blob, cfg)
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        tmp.write_bytes(blob)
        tmp.replace(dest)
        report[spec["dest"]] = {"sha256": spec["sha256"], "bytes": len(blob), "source": src}
    if report != previous:  # rewrite only on change, so re-bundling the same tree gives the same sha256
        sources.write_text(json.dumps({"note": "frozen inputs staged by pod/make_bundle.py; sha256-verified",
                                       "files": report}, indent=1) + "\n", encoding="utf-8", newline="\n")
    return report


# ----------------------------------------------------------------------------------- prereg freeze
def prereg_freeze_state() -> dict:
    """Is the prereg frozen (vocab_ids.json + PREREG-HASH.txt present and the hash matching)? Uses the
    canonical ``ic.config.check_prereg_hash`` when importable, else sha256(PREREG.md+config.json+vocab_ids.json)."""
    pre = EM / "prereg"
    st: dict = {"vocab_ids": (pre / "vocab_ids.json").is_file(), "hash_file": (pre / "PREREG-HASH.txt").is_file()}
    if not (st["vocab_ids"] and st["hash_file"]):
        st["frozen"] = False
        st["why"] = "vocab_ids.json and/or PREREG-HASH.txt missing (run lens.py --freeze-vocab, then hash)"
        return st
    try:
        sys.path.insert(0, str(EM / "src"))
        from ic import config as ic_config  # type: ignore

        st["match"] = bool(ic_config.check_prereg_hash())
        st["method"] = "ic.config.check_prereg_hash"
    except Exception as exc:  # noqa: BLE001
        blob = b"".join((pre / n).read_bytes() for n in ("PREREG.md", "config.json", "vocab_ids.json"))
        m = re.search(r"\b[0-9a-f]{64}\b", (pre / "PREREG-HASH.txt").read_text(encoding="utf-8", errors="replace"))
        st["match"] = bool(m) and m.group(0) == sha256_bytes(blob)
        st["method"] = f"fallback sha256(PREREG.md+config.json+vocab_ids.json) ({type(exc).__name__} importing ic.config)"
    finally:
        if sys.path and sys.path[0] == str(EM / "src"):
            sys.path.pop(0)
    st["frozen"] = st["match"]
    if not st["match"]:
        st["why"] = "PREREG-HASH.txt does not match the current PREREG.md/config.json/vocab_ids.json"
    return st


# ----------------------------------------------------------------------------------- file selection
def _excluded(rel_parts: tuple[str, ...], p: Path) -> bool:
    if any(part in EXCLUDE_DIRS for part in rel_parts[:-1]):
        return True
    if p.name in EXCLUDE_NAMES or p.suffix in EXCLUDE_SUFFIXES or p.name.endswith(".tmp"):
        return True
    return False


def collect_files(root: Path = ROOT, em: Path = EM) -> list[tuple[str, Path]]:
    """(arcname, path) pairs, project-relative, sorted."""
    em_rel = em.relative_to(root)
    picks: list[Path] = []
    for sub in ("src", "pod", "prereg", "data", "tests"):
        d = em / sub
        if d.is_dir():
            picks += [p for p in d.rglob("*") if p.is_file()]
    for name in ("smoke.py", "SPEC.md"):
        if (em / name).is_file():
            picks.append(em / name)
    for d in (root / "flagship" / "src" / "wor", root / "flagship" / "third_party" / "jlens"):
        if d.is_dir():
            picks += [p for p in d.rglob("*") if p.is_file()]
    out: dict[str, Path] = {}
    for p in picks:
        rel = p.relative_to(root)
        if _excluded(rel.parts, p):
            continue
        if p.stat().st_size > MAX_FILE_BYTES:
            raise BundleError(f"{rel} is {p.stat().st_size} bytes (> {MAX_FILE_BYTES}); not bundling large files")
        out[rel.as_posix()] = p
    _ = em_rel
    return sorted(out.items())


def file_bytes(arc: str, p: Path) -> tuple[bytes, bool]:
    """Bytes as they go into the tar. Shell scripts get CRLF -> LF. Returns (bytes, was_converted)."""
    b = p.read_bytes()
    if arc.endswith(".sh") and b"\r\n" in b:
        return b.replace(b"\r\n", b"\n"), True
    return b, False


def _tarinfo(name: str, size: int, mode: int, kind: bytes = tarfile.REGTYPE) -> tarfile.TarInfo:
    ti = tarfile.TarInfo(name)
    ti.size, ti.mode, ti.type, ti.mtime = size, mode, kind, FIXED_MTIME
    ti.uid = ti.gid = 0
    ti.uname = ti.gname = "root"
    return ti


def build_tar(files: list[tuple[str, Path]], out: Path, header: list[str]) -> dict:
    """Write the deterministic .tgz; returns {"sha256", "bytes", "n_files", "converted", "manifest"}."""
    contents: list[tuple[str, bytes]] = []
    converted: list[str] = []
    for arc, p in files:
        b, conv = file_bytes(arc, p)
        if conv:
            converted.append(arc)
        contents.append((arc, b))
    manifest_lines = [f"# {h}" for h in header]
    manifest_lines += [f"{sha256_bytes(b)}  {arc}" for arc, b in contents]
    manifest = ("\n".join(manifest_lines) + "\n").encode("utf-8")
    contents.append(("interpcontrol/em/pod/BUNDLE_MANIFEST.txt", manifest))
    contents.sort(key=lambda x: x[0])
    dirs: set[str] = set()
    for arc, _ in contents:
        parts = arc.split("/")[:-1]
        for i in range(1, len(parts) + 1):
            dirs.add("/".join(parts[:i]))
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    with open(tmp, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=9) as gz, \
            tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for d in sorted(dirs):
            tar.addfile(_tarinfo(d, 0, 0o755, tarfile.DIRTYPE))
        for arc, b in contents:
            mode = 0o755 if arc.endswith((".sh",)) else 0o644
            tar.addfile(_tarinfo(arc, len(b), mode), io.BytesIO(b))
    tmp.replace(out)
    blob = out.read_bytes()
    return {"sha256": sha256_bytes(blob), "bytes": len(blob), "n_files": len(contents), "converted": converted,
            "manifest": manifest.decode("utf-8")}


def write_chunks(out: Path, chunk_kb: int) -> list[str]:
    """``<out>.partNNN`` files + ``<out>.parts.sha256`` (``sha256sum -c`` format). Returns part names."""
    for old in out.parent.glob(out.name + ".part*"):
        old.unlink()
    blob = out.read_bytes()
    n = max(1, chunk_kb) * 1024
    names, lines = [], []
    for i in range(0, max(1, len(blob)), n):
        name = f"{out.name}.part{i // n:03d}"
        chunk = blob[i:i + n]
        (out.parent / name).write_bytes(chunk)
        names.append(name)
        lines.append(f"{sha256_bytes(chunk)}  {name}")
    (out.parent / f"{out.name}.parts.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return names


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path(tempfile.gettempdir()) / "ic_em_bundle" / "ic_em_bundle.tgz")
    ap.add_argument("--data-from", type=Path, action="append", default=[], help="dir holding the frozen inputs (repeatable)")
    ap.add_argument("--fetch", action="store_true", help="download missing frozen inputs from the pinned URLs (sha-checked)")
    ap.add_argument("--stage-only", action="store_true", help="only stage em/data/; no tarball")
    ap.add_argument("--allow-unfrozen", action="store_true", help="bundle even if vocab_ids/PREREG-HASH are missing or stale")
    ap.add_argument("--allow-incomplete", action="store_true", help="bundle even if required ic modules are missing (tests)")
    ap.add_argument("--chunk-kb", type=int, default=400, help="also write N-KB parts for flaky scp (0 = none)")
    ap.add_argument("--list", action="store_true", help="print the bundled file list")
    a = ap.parse_args(argv)
    cfg = load_cfg()
    try:
        staged = stage_data(cfg, a.data_from, a.fetch)
    except BundleError as exc:
        print(f"ERROR (data): {exc}", file=sys.stderr)
        return 2
    print("frozen inputs:", json.dumps(staged, indent=1))
    if a.stage_only:
        return 0
    freeze = prereg_freeze_state()
    print("prereg:", json.dumps(freeze))
    if not freeze["frozen"] and not a.allow_unfrozen:
        print(f"ERROR: prereg not frozen: {freeze.get('why')}. A real pod bundle needs the hashed prereg; "
              "use --allow-unfrozen only for smoke/tests.", file=sys.stderr)
        return 3
    try:
        files = collect_files()
    except BundleError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    present = {arc for arc, _ in files}
    missing = [r for r in REQUIRED if r not in present and not (a.allow_unfrozen and r in PREREG_FROZEN)]
    if missing and not a.allow_incomplete:
        print("ERROR: required files missing from the bundle:\n  " + "\n  ".join(missing), file=sys.stderr)
        return 4
    header = [f"interpcontrol/em pod bundle; {len(files)} files; prereg frozen={freeze['frozen']}",
              "verify on the pod: cd <code root> && grep -E '^[0-9a-f]{64}  ' interpcontrol/em/pod/BUNDLE_MANIFEST.txt | sha256sum -c --quiet -"]
    if missing:
        header.append("INCOMPLETE: missing " + ", ".join(missing))
    if not freeze["frozen"]:
        header.append("UNFROZEN PREREG: not valid for the real pod run")
    res = build_tar(files, a.out, header)
    a.out.with_name(a.out.name + ".sha256").write_text(f"{res['sha256']}  {a.out.name}\n", encoding="utf-8", newline="\n")
    a.out.with_name(a.out.name + ".MANIFEST.txt").write_text(res["manifest"], encoding="utf-8", newline="\n")
    parts = write_chunks(a.out, a.chunk_kb) if a.chunk_kb > 0 else []
    if a.list:
        for arc, _ in files:
            print("  ", arc)
    for arc in res["converted"]:
        print(f"WARNING: {arc} had CRLF line endings; converted to LF in the bundle (fix the source file too)")
    print(f"wrote {a.out} ({res['bytes']} bytes, {res['n_files']} files, sha256 {res['sha256']})")
    if parts:
        print(f"      {len(parts)} parts of {a.chunk_kb} KB + {a.out.name}.parts.sha256 in {a.out.parent}")
    print("pod: cat %s.part* > %s && sha256sum -c %s.sha256 && mkdir -p /workspace/ic/code && "
          "tar xzf %s --no-same-owner -C /workspace/ic/code" % (a.out.name, a.out.name, a.out.name, a.out.name))
    if missing or not freeze["frozen"]:
        print("NOTE: this bundle is NOT valid for the real pod run (see the warnings above).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
