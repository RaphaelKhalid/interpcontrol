#!/usr/bin/env python3
"""Laptop-side status sync for interpcontrol/em: every 30 s, pull the pod's ``/workspace/ic/status.json``
over ssh into ``interpcontrol/runs/status.json`` (the OLED dash at http://127.0.0.1:8796 reads it).

    python interpcontrol/em/tools/sync_status.py                    # loop (default 30 s)
    python interpcontrol/em/tools/sync_status.py --once             # one pull; exit 0 ok / 1 failed
    python interpcontrol/em/tools/sync_status.py --addr 1.2.3.4:22022

READ-ONLY on the pod: one ``ssh ... 'cat status.json'`` per cycle (plus file age, the DONE marker and
one nvidia-smi line). It never creates, stops or terminates a pod and never calls a RunPod API.

Address: ``--addr`` or ``interpcontrol/runs/pod_addr.txt`` (re-read every cycle, CRs stripped), one line
in any of the forms ``IP PORT``, ``IP:PORT``, ``root@IP -p PORT``, ``ssh -p PORT root@IP``.
Key: ``~/.ssh/autolabs_runpod`` (SPEC 11 ssh template; StrictHostKeyChecking=no, known_hosts=/dev/null).

Safety of the local file:
  * the remote text must parse as a JSON object, else the local file is left untouched;
  * written atomically (temp file in the same dir + ``os.replace``, retried if the dash has it open);
  * on a failed pull nothing is written, so the dash's ``age`` keeps growing (never fake freshness).

Laptop-owned fields survive every pull. The orchestrator writes ``pod.started_at`` / ``pod.terminated_at``
(and may correct ``spent_before_usd``) on the laptop; those edits are detected (they differ from what this
script last wrote), kept in ``interpcontrol/runs/pod_meta.json`` and re-applied over the pod's copy. They are
dropped if the pod id changes (a stale file from an earlier pod never leaks into a new run).

Adds a ``sync`` block to the local status: ``{ok, at, host, remote_age_s, stall, done, gpu}``.
``stall`` is True when the pod's status.json has not been rewritten for ``--stall-min`` (default 20) min:
the heartbeat rule (never trust silence). Exits after the pod reports DONE (phase "done") unless
``--no-exit-on-done``; after DONE, ``--max-fails-after-done`` (default 5) failed pulls end the loop
(the pod has been terminated).

Incremental artifact pulls (READ-ONLY too): every ``--artifacts-every-min`` (default 10) minutes, and once
more when DONE is seen, ONE ssh call streams a tar.gz of the small paid outputs that exist so far in
``/workspace/ic/results`` (``ARTIFACT_PATTERNS``: generations, labels, readouts, BB scores, timings, G1,
subspace meta; a few tens of MB) into ``interpcontrol/runs/pod_results/`` (each file replaced atomically,
only plain basenames accepted). If the pod dies early (guard, accident, or the other chat on the shared
account), at most ~10 min of work is lost instead of everything. ``--artifacts-every-min 0`` disables it.
This script never terminates anything: a pod idling after DONE is the orchestrator's (or the spend
guard's hard cap's) business.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import subprocess
import sys
import tarfile
import time
from pathlib import Path

EM = Path(__file__).resolve().parents[1]
IC = EM.parent
RUNS = IC / "runs"
LOCAL = RUNS / "status.json"
ADDR_FILE = RUNS / "pod_addr.txt"
META_FILE = RUNS / "pod_meta.json"
ARTIFACTS_DIR = RUNS / "pod_results"
KEY = Path.home() / ".ssh" / "autolabs_runpod"
REMOTE_STATUS = "/workspace/ic/status.json"
REMOTE_ROOT = "/workspace/ic"
REMOTE_RESULTS = "/workspace/ic/results"
ARTIFACT_PATTERNS = ("gen_round*.jsonl", "round*_labeled.jsonl", "readout_round*.jsonl", "bb_scores.jsonl",
                     "bb_summary.json", "stage_timings.json", "sanity_G1.json", "round1_subspaces.json",
                     "gen_round*_hookstats.json", "semgrep_r0.json", "analysis.json", "RESULT.md")
_SAFE_NAME = re.compile(r"^[A-Za-z0-9_.\-]+$")
SSH_CANDIDATES = (r"C:\Program Files\Git\usr\bin\ssh.exe", r"C:\Windows\System32\OpenSSH\ssh.exe")


def log(msg: str) -> None:
    print(f"[sync {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------------------------------- address
_PAT_HOST_PORT = re.compile(r"^(?:ssh\s+)?(?:\w+@)?([A-Za-z0-9.\-]+)(?:\s*:\s*|\s+)(?:-p\s*)?(\d{1,5})\b")
_PAT_PORT_HOST = re.compile(r"-p\s*(\d{1,5})\b.*?(?:\w+@)([A-Za-z0-9.\-]+)")


def parse_addr(text: str) -> tuple[str, int] | None:
    """First usable ``(host, port)`` in ``text``; None if there is none."""
    for raw in text.replace("\r", "").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _PAT_PORT_HOST.search(line)
        if m:
            return m.group(2), int(m.group(1))
        m = _PAT_HOST_PORT.search(line)
        if m and 0 < int(m.group(2)) < 65536:
            return m.group(1), int(m.group(2))
    return None


# ----------------------------------------------------------------------------------------- ssh
def find_ssh(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    for p in SSH_CANDIDATES:
        if Path(p).exists():
            return p
    return "ssh"


def null_device(ssh: str) -> str:
    """``/dev/null`` for MSYS/Git or POSIX ssh; ``NUL`` for the native Windows OpenSSH client."""
    s = ssh.lower().replace("/", "\\")
    if os.name == "nt" and "windows\\system32\\openssh" in s:
        return "NUL"
    return "/dev/null"


def build_remote_cmd(status_path: str = REMOTE_STATUS, ic_root: str = REMOTE_ROOT) -> str:
    """POSIX sh run on the pod: markers first, then ``@@JSON`` and the raw status.json."""
    return (
        f'f="{status_path}"; echo "@@NOW $(date +%s)"; '
        f'if [ -s "$f" ]; then echo "@@AGE $(( $(date +%s) - $(stat -c %Y "$f") ))"; fi; '
        f'if [ -f "{ic_root}/DONE" ]; then echo "@@DONE $(head -c 300 "{ic_root}/DONE" | tr "\\t\\n" "  ")"; fi; '
        'echo "@@GPU $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)"; '
        'echo "@@JSON"; if [ -s "$f" ]; then cat "$f"; fi'
    )


def build_artifact_cmd(results_dir: str = REMOTE_RESULTS, patterns=ARTIFACT_PATTERNS) -> str:
    """POSIX sh run on the pod: tar.gz of the artifacts that exist so far, to stdout. Exit 4 = none yet,
    3 = no results dir. Read-only (tar only reads)."""
    pats = " ".join(patterns)
    return (f'cd "{results_dir}" 2>/dev/null || exit 3; '
            f'set -- $(ls -1d {pats} 2>/dev/null); [ "$#" -gt 0 ] || exit 4; '
            'tar czf - "$@"')


def run_ssh_bytes(argv: list[str], timeout: float = 300.0) -> tuple[int, bytes, str]:
    try:
        r = subprocess.run(argv, capture_output=True, timeout=timeout, stdin=subprocess.DEVNULL)
        return r.returncode, r.stdout, r.stderr.decode("utf-8", "replace")
    except subprocess.TimeoutExpired:
        return 124, b"", f"ssh timed out after {timeout:.0f} s"
    except OSError as exc:
        return 127, b"", f"ssh not runnable: {exc}"


def extract_artifacts(blob: bytes, dest: Path) -> list[str]:
    """Unpack a pulled tar.gz into ``dest``: regular files with a plain safe basename only (no paths, no
    links), each written to a temp name and ``os.replace``d. Returns the names written."""
    dest.mkdir(parents=True, exist_ok=True)
    names: list[str] = []
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
        for m in tar.getmembers():
            if not m.isfile() or not _SAFE_NAME.match(m.name) or m.name in (".", ".."):
                continue
            f = tar.extractfile(m)
            if f is None:
                continue
            tmp = dest / f".{m.name}.pull.tmp"
            tmp.write_bytes(f.read())
            os.replace(tmp, dest / m.name)
            names.append(m.name)
    return names


def ssh_argv(ssh: str, key: Path, host: str, port: int, remote_cmd: str) -> list[str]:
    nul = null_device(ssh)
    return [ssh, "-i", str(key), "-o", "StrictHostKeyChecking=no", "-o", f"UserKnownHostsFile={nul}",
            "-o", "LogLevel=ERROR", "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=10",
            "-o", "BatchMode=yes", "-p", str(port), f"root@{host}", remote_cmd]


def run_ssh(argv: list[str], timeout: float = 40.0) -> tuple[int, str, str]:
    try:
        r = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8", errors="replace",
                           timeout=timeout, stdin=subprocess.DEVNULL)
        return r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        return 124, "", f"ssh timed out after {timeout:.0f} s"
    except OSError as exc:
        return 127, "", f"ssh not runnable: {exc}"


def parse_remote(out: str) -> dict:
    """Split the remote output into markers and the status document.
    Returns {"status": dict|None, "error": str|None, "age_s", "now", "done", "gpu"}."""
    head, sep, body = out.partition("@@JSON")
    res: dict = {"status": None, "error": None, "age_s": None, "now": None, "done": None, "gpu": None}
    for line in head.replace("\r", "").splitlines():
        if line.startswith("@@NOW "):
            res["now"] = _int(line[6:])
        elif line.startswith("@@AGE "):
            res["age_s"] = _int(line[6:])
        elif line.startswith("@@DONE"):
            res["done"] = line[6:].strip() or "DONE"
        elif line.startswith("@@GPU"):
            res["gpu"] = line[5:].strip() or None
    if not sep:
        res["error"] = "no @@JSON marker in ssh output"
        return res
    text = body.strip()
    if not text:
        res["error"] = "remote status.json missing or empty"
        return res
    try:
        doc = json.loads(text)
    except ValueError as exc:
        res["error"] = f"remote status.json is not valid JSON ({exc.__class__.__name__}: {str(exc)[:80]})"
        return res
    if not isinstance(doc, dict):
        res["error"] = f"remote status.json is a {type(doc).__name__}, not an object"
        return res
    res["status"] = doc
    return res


def _int(s: str) -> int | None:
    try:
        return int(s.strip())
    except ValueError:
        return None


# ----------------------------------------------------------------------------------------- merge
def read_json(p: Path) -> dict:
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def merge(remote: dict, local: dict, meta: dict) -> tuple[dict, dict]:
    """Remote status + laptop-owned overrides. Returns (merged, new_meta). Pure function."""
    meta = json.loads(json.dumps(meta or {}))
    last = meta.get("_last_written") or {}
    ov_pod = dict(meta.get("pod") or {})
    ov_top = dict(meta.get("top") or {})
    lpod = local.get("pod") if isinstance(local.get("pod"), dict) else {}
    lastpod = last.get("pod") or {}
    for k, v in lpod.items():                       # laptop edits = values we did not write ourselves
        if v is not None and v != lastpod.get(k):
            ov_pod[k] = v
    for k in ("spent_before_usd", "cap_usd"):
        if local.get(k) is not None and local.get(k) != last.get(k):
            ov_top[k] = local[k]
    rpod = remote.get("pod") if isinstance(remote.get("pod"), dict) else {}
    if ov_pod.get("id") and rpod.get("id") and ov_pod["id"] != rpod["id"]:
        log(f"laptop pod fields are for pod {ov_pod['id']}, the pod reports {rpod['id']}: dropping the stale overrides")
        ov_pod, ov_top = {}, {}
    merged = dict(remote)
    merged["pod"] = {**rpod, **{k: v for k, v in ov_pod.items() if v is not None}}
    for k, v in ov_top.items():
        merged[k] = v
    meta["pod"], meta["top"] = ov_pod, ov_top
    meta["_last_written"] = {"pod": merged["pod"], "spent_before_usd": merged.get("spent_before_usd"),
                             "cap_usd": merged.get("cap_usd")}
    return merged, meta


def atomic_write_json(p: Path, doc: dict, retries: int = 10) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f"{p.name}.sync.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    for i in range(retries):
        try:
            os.replace(tmp, p)
            return
        except PermissionError:  # Windows: the dash may have the file open for a moment
            time.sleep(0.2 * (i + 1))
    os.replace(tmp, p)


# ----------------------------------------------------------------------------------------- one cycle
class Syncer:
    def __init__(self, *, addr: str | None = None, addr_file: Path = ADDR_FILE, out: Path = LOCAL,
                 meta_file: Path = META_FILE, key: Path = KEY, ssh: str | None = None,
                 remote_status: str = REMOTE_STATUS, remote_root: str = REMOTE_ROOT,
                 stall_min: float = 20.0, ssh_timeout: float = 40.0, runner=run_ssh,
                 artifacts_dir: Path = ARTIFACTS_DIR, remote_results: str = REMOTE_RESULTS,
                 bin_runner=run_ssh_bytes, artifact_timeout: float = 300.0) -> None:
        self.addr, self.addr_file, self.out, self.meta_file = addr, Path(addr_file), Path(out), Path(meta_file)
        self.key, self.ssh = Path(key), find_ssh(ssh)
        self.remote_cmd = build_remote_cmd(remote_status, remote_root)
        self.stall_s = stall_min * 60.0
        self.ssh_timeout = ssh_timeout
        self.runner = runner
        self.fails = 0
        self.done_seen = False
        self.last_ok: float | None = None
        self._stall_warned = False
        self.artifacts_dir = Path(artifacts_dir)
        self.artifact_cmd = build_artifact_cmd(remote_results)
        self.bin_runner = bin_runner
        self.artifact_timeout = artifact_timeout

    def pull_artifacts(self) -> dict:
        """One read-only ssh call: tar.gz of the small result files that exist so far -> artifacts_dir."""
        tgt = self.target()
        if tgt is None:
            return {"ok": False, "error": "no pod address"}
        host, port = tgt
        rc, blob, err = self.bin_runner(ssh_argv(self.ssh, self.key, host, port, self.artifact_cmd),
                                        self.artifact_timeout)
        if rc == 4:
            return {"ok": True, "n_files": 0, "note": "no artifacts yet"}
        if rc != 0 or not blob:
            return {"ok": False, "error": f"rc={rc}: {err.strip()[-200:] or 'no output'}"}
        try:
            names = extract_artifacts(blob, self.artifacts_dir)
        except (tarfile.TarError, OSError, EOFError) as exc:
            return {"ok": False, "error": f"bad archive: {exc!r}"[:200]}
        manifest = {"at": round(time.time(), 1), "host": f"{host}:{port}", "bytes": len(blob),
                    "files": {n: (self.artifacts_dir / n).stat().st_size for n in names}}
        atomic_write_json(self.artifacts_dir / "_pulled.json", manifest)
        return {"ok": True, "n_files": len(names), "bytes": len(blob)}

    def target(self) -> tuple[str, int] | None:
        if self.addr:
            return parse_addr(self.addr)
        try:
            return parse_addr(self.addr_file.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            return None

    def cycle(self) -> dict:
        """One pull. Returns {"ok": bool, "error"?, "phase"?, "done"?, "stall"?}."""
        tgt = self.target()
        if tgt is None:
            self.fails += 1
            return {"ok": False, "error": f"no pod address ({self.addr or self.addr_file})"}
        host, port = tgt
        rc, out, err = self.runner(ssh_argv(self.ssh, self.key, host, port, self.remote_cmd), self.ssh_timeout)
        parsed = parse_remote(out) if rc == 0 or "@@JSON" in out else {"status": None, "error": None}
        if parsed.get("status") is None:
            self.fails += 1
            why = parsed.get("error") or f"ssh rc={rc}: {err.strip()[-200:] or 'no output'}"
            return {"ok": False, "error": why, "host": f"{host}:{port}"}
        now = time.time()
        age = parsed.get("age_s")
        stall = age is not None and age > self.stall_s
        merged, meta = merge(parsed["status"], read_json(self.out), read_json(self.meta_file))
        merged["sync"] = {"ok": True, "at": round(now, 1), "host": f"{host}:{port}", "remote_age_s": age,
                          "stall": stall, "done": parsed.get("done"), "gpu": parsed.get("gpu"),
                          "fails_before": self.fails}
        atomic_write_json(self.out, merged)
        atomic_write_json(self.meta_file, meta)
        self.fails = 0
        self.last_ok = now
        if stall and not self._stall_warned:
            log(f"STALL: pod status.json not rewritten for {age} s (> {self.stall_s:.0f} s). Check the pod; "
                "the orchestrator decides (this script never touches the pod).")
        self._stall_warned = stall
        phase = merged.get("phase")
        done = bool(parsed.get("done")) or phase == "done"
        if done:
            self.done_seen = True
        return {"ok": True, "phase": phase, "done": done, "stall": stall, "age_s": age, "host": f"{host}:{port}"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--interval", type=float, default=30.0)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--addr", default=None, help="IP:PORT or 'IP PORT' (default: interpcontrol/runs/pod_addr.txt)")
    ap.add_argument("--addr-file", type=Path, default=ADDR_FILE)
    ap.add_argument("--out", type=Path, default=LOCAL)
    ap.add_argument("--meta", type=Path, default=META_FILE)
    ap.add_argument("--key", type=Path, default=KEY)
    ap.add_argument("--ssh", default=None, help="ssh binary (default: Git's ssh.exe, else Windows OpenSSH, else ssh)")
    ap.add_argument("--remote-status", default=REMOTE_STATUS)
    ap.add_argument("--remote-root", default=REMOTE_ROOT)
    ap.add_argument("--stall-min", type=float, default=20.0)
    ap.add_argument("--no-exit-on-done", action="store_true")
    ap.add_argument("--max-fails-after-done", type=int, default=5)
    ap.add_argument("--artifacts-every-min", type=float, default=10.0,
                    help="pull the small result files every N min and at DONE (0 = never)")
    ap.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    ap.add_argument("--remote-results", default=REMOTE_RESULTS)
    ap.add_argument("--max-cycles", type=int, default=None, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    s = Syncer(addr=a.addr, addr_file=a.addr_file, out=a.out, meta_file=a.meta, key=a.key, ssh=a.ssh,
               remote_status=a.remote_status, remote_root=a.remote_root, stall_min=a.stall_min,
               artifacts_dir=a.artifacts_dir, remote_results=a.remote_results)
    log(f"ssh={s.ssh} key={s.key} out={s.out} every {a.interval:.0f}s; artifacts "
        + (f"every {a.artifacts_every_min:g} min -> {s.artifacts_dir}" if a.artifacts_every_min > 0 else "off"))
    n = 0
    done_cycles = 0
    last_pull = time.time()
    while True:
        n += 1
        r = s.cycle()
        if r["ok"]:
            log(f"ok {r['host']} phase={r.get('phase')} remote_age={r.get('age_s')}s"
                f"{' STALL' if r.get('stall') else ''}{' DONE' if r.get('done') else ''}")
        else:
            log(f"FAIL ({s.fails} in a row): {r.get('error')}")
        if a.artifacts_every_min > 0 and r["ok"] and (
                (r.get("done") and done_cycles == 0) or time.time() - last_pull >= a.artifacts_every_min * 60):
            pr = s.pull_artifacts()
            last_pull = time.time()
            log(f"artifacts: {pr}")
        if a.once:
            return 0 if r["ok"] else 1
        if r["ok"] and r.get("done"):
            done_cycles += 1
            if not a.no_exit_on_done and done_cycles >= 2:   # one extra pull after DONE for the final numbers
                log("pod reports DONE; final status synced; exiting")
                return 0
        if s.done_seen and not r["ok"] and s.fails >= a.max_fails_after_done:
            log(f"pod unreachable {s.fails}x after DONE (terminated?); exiting")
            return 0
        if a.max_cycles is not None and n >= a.max_cycles:
            return 0
        time.sleep(a.interval)


if __name__ == "__main__":
    raise SystemExit(main())
