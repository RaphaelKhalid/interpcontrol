"""Dashboard status for interpcontrol/em (writes ``status.json`` atomically).

Schema = the docstring of ``interpcontrol/dash/server.py`` (the OLED dash at
http://127.0.0.1:8796 reads ``interpcontrol/runs/status.json``)::

  {"phase": str, "phases": [str], "counts": {name: [done, total]},
   "attempts": int, "validated": int, "auroc": {"BB": x, "WB0": y},
   "money_shot": {"rescued": a, "lost": b, "n": n},
   "pod": {"id", "gpu", "rate_usd_h", "started_at", "terminated_at"},
   "spent_before_usd": float, "cap_usd": 15, "trial_cap_usd": 3,
   "events": ["[hh:mm] text", ...], "updated": unix}

Field mapping for this experiment (SPEC section 8): ``attempts`` = running count of
analyzer-flagged (vulnerable) outputs; ``validated`` = the same number post-label
(kept equal); ``auroc`` = {"BB", "WB0"}; ``money_shot`` = {"rescued", "lost", "n"};
``counts`` keys gen_r0 / label / readout / bb / gen_r1 / readout_r1.

Where the file goes (first match wins):
  1. ``$IC_STATUS_PATH``
  2. ``/workspace/ic/status.json`` when ``/workspace/ic`` exists (the pod; the laptop
     syncs it to ``interpcontrol/runs/status.json`` with scp, SPEC runbook step 8)
  3. ``interpcontrol/runs/status.json`` next to this source tree (the laptop)

Every write is atomic (temp file in the same directory, fsync, ``os.replace``), so
the dash never reads a half-written file. ``None`` arguments keep the value already
on disk, so stage scripts can update a few fields without knowing the rest.

Concurrent writers (on the pod: ic.generate, the background ``ic.label --follow``,
run_all.sh's / bootstrap.sh's event helpers, spend_guard): every read-modify-write holds
an exclusive ``fcntl.flock`` on ``<status>.lock`` (POSIX only; on Windows the lock is a
no-op, which is fine for the single-writer laptop case), so no writer's counts or events
are lost. The shell helpers take the same lock.

CLI::

  python -m ic.status init                 # first stage of pod/run_all.sh
  python -m ic.status show
  python -m ic.status phase gen_r0
  python -m ic.status event "U loaded in 312 s"
  python -m ic.status validate
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

PHASES: list[str] = ["setup", "sanity", "gen_r0", "label", "readout", "bb",
                     "gen_r1", "readout_r1", "analyze", "done"]
COUNT_KEYS: list[str] = ["gen_r0", "label", "readout", "bb", "gen_r1", "readout_r1"]
MAX_EVENTS = 200
MAX_EVENT_CHARS = 160

_HERE = Path(__file__).resolve()
_EM = _HERE.parents[2]                      # interpcontrol/em
_IC = _HERE.parents[3]                      # interpcontrol


def status_path() -> Path:
    env = os.environ.get("IC_STATUS_PATH")
    if env:
        return Path(env)
    pod_dir = Path("/workspace/ic")
    if pod_dir.is_dir():
        return pod_dir / "status.json"
    return _IC / "runs" / "status.json"


def event(text: str, t: float | None = None) -> str:
    """One short ``[hh:mm] text`` line (local time of the writer)."""
    stamp = time.strftime("%H:%M", time.localtime(time.time() if t is None else t))
    line = " ".join(str(text).split())
    return f"[{stamp}] {line}"[:MAX_EVENT_CHARS]


def read(path: str | os.PathLike | None = None) -> dict:
    """Current status document, or ``{}`` if missing or unreadable."""
    p = Path(path) if path else status_path()
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


@contextlib.contextmanager
def locked(path: str | os.PathLike | None = None) -> Iterator[None]:
    """Exclusive advisory lock on ``<status>.lock`` around a read-modify-write (POSIX
    ``fcntl.flock``; a no-op where fcntl does not exist, i.e. Windows). Never raises for
    lock trouble: a status write must not kill a paid stage."""
    p = Path(path) if path else status_path()
    try:
        import fcntl  # noqa: PLC0415  (POSIX only)
    except ImportError:
        yield
        return
    fh = None
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        fh = open(p.with_name(p.name + ".lock"), "a")  # noqa: SIM115
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
    except OSError:
        if fh is not None:
            fh.close()
        fh = None
    try:
        yield
    finally:
        if fh is not None:
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            finally:
                fh.close()


def _atomic_write(doc: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".status.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=1, default=_json_default)
            f.flush()
            os.fsync(f.fileno())
        # Windows refuses os.replace while another process holds the target open
        # (the dash reads it every 3 s); retry briefly instead of failing the stage.
        for attempt in range(40):
            try:
                os.replace(tmp, path)
                return
            except PermissionError:
                if attempt == 39:
                    raise
                time.sleep(0.05)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def _json_default(o: Any) -> Any:
    try:
        import numpy as np  # noqa: PLC0415

        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:  # pragma: no cover
        pass
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serializable: {type(o).__name__}")


def _clean_events(events: list[str] | None) -> list[str]:
    out = []
    for e in events or []:
        s = str(e)
        if not s.startswith("["):
            s = event(s)
        out.append(s[:MAX_EVENT_CHARS])
    return out[-MAX_EVENTS:]


def write(phase: str, phases: list[str], counts: dict, attempts: int, validated: int,
          auroc: dict, money_shot: dict, pod: dict, events: list[str],
          spent_before_usd: float, cap_usd: float = 15, trial_cap_usd: float = 3,
          path: str | os.PathLike | None = None) -> None:
    """Write the full status document (SPEC section 8 signature).

    Any argument passed as ``None`` keeps the value already on disk. ``events`` is the
    full list to keep (truncated to the last 200 lines); use :func:`update` to append."""
    p = Path(path) if path else status_path()
    with locked(p):
        old = read(p)
        new = {
            "phase": phase, "phases": phases, "counts": counts, "attempts": attempts,
            "validated": validated, "auroc": auroc, "money_shot": money_shot, "pod": pod,
            "events": None if events is None else _clean_events(events),
            "spent_before_usd": spent_before_usd, "cap_usd": cap_usd,
            "trial_cap_usd": trial_cap_usd,
        }
        doc = dict(old)
        for k, v in new.items():
            if v is not None:
                doc[k] = v
        doc.setdefault("phases", list(PHASES))
        doc.setdefault("events", [])
        doc["updated"] = time.time()
        _atomic_write(doc, p)


def update(path: str | os.PathLike | None = None, *, event_text: str | None = None,
           events: list[str] | None = None, counts: dict | None = None,
           pod: dict | None = None, **fields: Any) -> dict:
    """Read-modify-write merge. ``counts`` and ``pod`` merge per key; ``events`` and
    ``event_text`` are APPENDED; any other keyword replaces that top-level field."""
    p = Path(path) if path else status_path()
    with locked(p):
        doc = read(p)
        for k, v in fields.items():
            if v is not None:
                doc[k] = v
        if counts:
            merged = dict(doc.get("counts") or {})
            for k, v in counts.items():
                merged[k] = [int(v[0]), int(v[1])]
            doc["counts"] = merged
        if pod:
            merged_pod = dict(doc.get("pod") or {})
            merged_pod.update(pod)
            doc["pod"] = merged_pod
        new_events = list(events or [])
        if event_text:
            new_events.append(event_text)
        if new_events:
            doc["events"] = _clean_events(list(doc.get("events") or []) + _clean_events(new_events))
        doc.setdefault("phases", list(PHASES))
        doc.setdefault("events", [])
        doc["updated"] = time.time()
        _atomic_write(doc, p)
    return doc


def progress(phase: str, done: int, total: int, events: list[str] | None = None,
             key: str | None = None, **fields: Any) -> dict:
    """A ready-made ``status_cb(phase, done, total, events)`` for the stage modules:
    sets the phase, writes ``counts[key or phase] = [done, total]`` and appends events.
    Extra keywords (e.g. ``attempts=``) replace those fields."""
    return update(phase=phase, counts={key or phase: [done, total]}, events=events, **fields)


def validate(doc: dict) -> list[str]:
    """Errors against the dash schema (empty list = valid). Mirrors what
    ``dash/server.py`` + ``dash/index.html`` read."""
    errs: list[str] = []
    num = (int, float)
    if not isinstance(doc, dict):
        return ["document is not an object"]
    if "phase" in doc and not isinstance(doc["phase"], str):
        errs.append("phase must be str")
    if "phases" in doc and not (isinstance(doc["phases"], list) and all(isinstance(x, str) for x in doc["phases"])):
        errs.append("phases must be list[str]")
    if "counts" in doc:
        c = doc["counts"]
        if not isinstance(c, dict):
            errs.append("counts must be an object")
        else:
            for k, v in c.items():
                if not (isinstance(v, list) and len(v) == 2 and all(isinstance(x, int) for x in v)):
                    errs.append(f"counts[{k}] must be [done:int, total:int]")
    for k in ("attempts", "validated"):
        if k in doc and doc[k] is not None and not isinstance(doc[k], int):
            errs.append(f"{k} must be int")
    if "auroc" in doc and doc["auroc"] is not None:
        a = doc["auroc"]
        if not isinstance(a, dict):
            errs.append("auroc must be an object")
        else:
            for k, v in a.items():
                if v is not None and not isinstance(v, num):
                    errs.append(f"auroc[{k}] must be a number or null")
    if doc.get("money_shot") is not None:
        m = doc["money_shot"]
        if not isinstance(m, dict) or not all(k in m for k in ("rescued", "lost", "n")):
            errs.append("money_shot must have rescued, lost, n")
    if doc.get("pod") is not None:
        p = doc["pod"]
        if not isinstance(p, dict):
            errs.append("pod must be an object")
        else:
            if p.get("rate_usd_h") is not None and not isinstance(p["rate_usd_h"], num):
                errs.append("pod.rate_usd_h must be a number")
            for k in ("started_at", "terminated_at"):
                if p.get(k) is not None and not isinstance(p[k], num):
                    errs.append(f"pod.{k} must be unix seconds or null")
    for k in ("spent_before_usd", "cap_usd", "trial_cap_usd", "updated"):
        if k in doc and doc[k] is not None and not isinstance(doc[k], num):
            errs.append(f"{k} must be a number")
    if "events" in doc and not (isinstance(doc["events"], list) and all(isinstance(e, str) for e in doc["events"])):
        errs.append("events must be list[str]")
    return errs


def _totals() -> dict[str, int]:
    """[done, total] denominators from the frozen config (no hard-coded N)."""
    try:
        from ic import config as _config  # noqa: PLC0415

        cfg = _config.load()
    except Exception:  # noqa: BLE001  (config.py is another builder's module)
        cfg = json.loads((_EM / "prereg" / "config.json").read_text(encoding="utf-8"))
    d = cfg["data"]
    g = cfg["generation"]
    n0 = (int(d["cyberseceval"]["n_python"]) + int(d["em_heldout_control"]["n"])) * int(g["round0"]["k0"])
    r1 = g["round1_S2"]
    n1 = int(r1["n_prompts"]) * int(r1["k1"]) * len(r1["conditions"])
    return {"gen_r0": n0, "label": n0, "readout": n0, "bb": n0, "gen_r1": n1, "readout_r1": n1}


def init(path: str | os.PathLike | None = None) -> dict:
    """First pipeline stage: a fresh document that keeps any pod / spend fields the
    orchestrator already wrote (pod id and start time come from the laptop side)."""
    p = Path(path) if path else status_path()
    with locked(p):
        return _init_locked(p)


def _init_locked(p: Path) -> dict:
    old = read(p)
    totals = _totals()
    pod = dict(old.get("pod") or {})
    if os.environ.get("RUNPOD_POD_ID") and not pod.get("id"):
        pod["id"] = os.environ["RUNPOD_POD_ID"]
    pod.setdefault("gpu", os.environ.get("IC_GPU", "NVIDIA A100-SXM4-80GB"))
    pod.setdefault("rate_usd_h", float(os.environ.get("IC_HOURLY_USD", "1.59")))
    started_note = None
    if not pod.get("started_at"):
        if os.environ.get("IC_POD_STARTED_AT"):
            pod["started_at"] = float(os.environ["IC_POD_STARTED_AT"])
            started_note = "pod started_at from IC_POD_STARTED_AT"
        elif pod.get("id") and os.path.exists("/proc/1"):
            # container PID 1 start ~ pod start; billing began slightly earlier (image pull),
            # so the dash's live $ is a small UNDER-estimate. spend_guard does the real accounting.
            pod["started_at"] = os.stat("/proc/1").st_ctime
            started_note = "pod started_at approximated by container start (dash $ slightly low)"
        else:
            pod["started_at"] = None
    pod.setdefault("terminated_at", None)
    spent_env = os.environ.get("IC_SPENT_BEFORE_USD")
    spent = float(spent_env) if spent_env else float(old.get("spent_before_usd") or 0.0)
    doc = {
        "phase": "setup", "phases": list(PHASES),
        "counts": {k: [0, totals[k]] for k in COUNT_KEYS},
        "attempts": 0, "validated": 0, "auroc": {}, "money_shot": None, "pod": pod,
        "spent_before_usd": spent, "cap_usd": float(old.get("cap_usd") or 15),
        "trial_cap_usd": float(old.get("trial_cap_usd") or 3),
        "events": _clean_events(list(old.get("events") or []) + [event("status init: pipeline starting")]
                                + ([event(started_note)] if started_note else [])),
        "updated": time.time(),
    }
    _atomic_write(doc, p)
    return doc


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    cmd = argv[0] if argv else "show"
    if cmd == "init":
        doc = init()
        print(f"status init -> {status_path()} ({len(doc['counts'])} counters)")
    elif cmd == "show":
        print(json.dumps(read(), indent=1))
    elif cmd == "phase" and len(argv) >= 2:
        update(phase=argv[1], event_text=f"phase {argv[1]}")
    elif cmd == "event" and len(argv) >= 2:
        update(event_text=" ".join(argv[1:]))
    elif cmd == "validate":
        errs = validate(read())
        print("OK" if not errs else "\n".join(errs))
        return 1 if errs else 0
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
