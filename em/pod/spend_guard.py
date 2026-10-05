#!/usr/bin/env python3
"""RunPod spend guard for the interpcontrol/em pod (ported from longdesc/pod/spend_guard.py, LD_* -> IC_*).

STATUS: NOT RUN ON A POD. Exercised on the laptop only with fake clocks and --dry-run
(tests/test_spend_guard.py). The terminate call has never fired on a real pod, and the pod-scoped
RUNPOD_API_KEY's permission to DELETE its own pod is UNVERIFIED (bootstrap.sh only probes GET /v1/pods/<id>).
If it lacks that permission, the laptop-side watcher is the only stop path.

What it tracks:   cumulative = IC_SPENT_BEFORE_USD + (now - pod start) / 3600 * IC_HOURLY_USD

Caps are for THIS EXPERIMENT (interpcontrol/em, the $15 approval), not for the whole RunPod account:
  * IC_SPENT_BEFORE_USD = what earlier ic-em pods already cost (0 for the first and only planned pod).
    It must NOT include the H-lens chat's pods on the same account (their spend is not ours, and billing
    reads lag anyway). Required, no silent default.
  * soft cap IC_SOFT_CAP_USD (13.5): no NEW stage starts at or above it, nor if cumulative + projected
    >= hard cap - margin (`check-step` exits 3; run_all.sh skips the stage).
  * hard cap IC_HARD_CAP_USD (14.75): the watchdog writes $IC_STATE/HARD_CAP_REACHED, waits 20 s, packs
    the results into $IC_ROOT/outbox (local only), then TERMINATES the pod. It fires IC_GUARD_MARGIN_USD
    (0.25) early to absorb the terminate latency.

The action is TERMINATE ONLY. A stopped pod keeps billing its disk, so this script never calls
`/stop` or `runpodctl stop`. Order: REST `DELETE https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID`
(Bearer RUNPOD_API_KEY, explicit User-Agent: the API rejects urllib's default agent, SPEC 11.3), then
`runpodctl remove pod $RUNPOD_POD_ID`, then `runpodctl pod delete $RUNPOD_POD_ID` (newer CLI syntax).
A failed terminate is retried every IC_GUARD_RETRY_S (120 s) up to 10 times (the pod keeps billing).

Environment:
  IC_HOURLY_USD         required. Pod $/h at creation (1.59 for the A100 80GB SECURE pod).
  IC_SPENT_BEFORE_USD   required (0 if nothing was spent on earlier ic-em pods). ic-em pods ONLY.
  IC_POD_START_EPOCH    unix time the pod started billing (the orchestrator passes its create time);
                        else $IC_STATE/pod_start_epoch (bootstrap.sh writes it).
  IC_SOFT_CAP_USD / IC_HARD_CAP_USD / IC_GUARD_MARGIN_USD   defaults 13.5 / 14.75 / 0.25.
  IC_GUARD_ACTION       "terminate" (default and the only action; anything else is logged and ignored).
  IC_ROOT (/workspace/ic), IC_STATE ($IC_ROOT/state), IC_RESULTS ($IC_ROOT/results), IC_LOGS ($IC_ROOT/logs).
  RUNPOD_POD_ID         injected by RunPod inside the pod.  RUNPOD_API_KEY  pod-scoped key.

Usage (same CLI as longdesc; run_all.sh and ic.generate._spend_ok rely on it):
  python spend_guard.py status
  python spend_guard.py check-step [--projected-usd X]    # exit 3 = a new stage must not start
  python spend_guard.py watchdog [--interval 30] [--dry-run]
  python spend_guard.py stop-now [--dry-run]               # dry-run prints the intended DELETE; nothing is sent
"""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import subprocess
import tarfile
import time
import urllib.request
from pathlib import Path

EXIT_STEP_BLOCKED = 3
EXIT_HARD_CAP = 4
REST_BASE = "https://rest.runpod.io/v1/pods/"
USER_AGENT = "ic-em-spend-guard/1.0"
PACK_MAX_BYTES = 1024**3


def ic_root() -> Path:
    return Path(os.environ.get("IC_ROOT", "/workspace/ic"))


def state_dir() -> Path:
    return Path(os.environ.get("IC_STATE") or ic_root() / "state")


def results_dir() -> Path:
    return Path(os.environ.get("IC_RESULTS") or ic_root() / "results")


def _f(name: str, default: float | None = None) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        if default is None:
            raise SystemExit(f"spend_guard: environment variable {name} is required")
        return default
    return float(raw)


def pod_start_epoch() -> float:
    raw = os.environ.get("IC_POD_START_EPOCH")
    if raw:
        return float(raw)
    f = state_dir() / "pod_start_epoch"
    if f.exists():
        return float(f.read_text().strip())
    raise SystemExit("spend_guard: set IC_POD_START_EPOCH or run bootstrap.sh (it writes pod_start_epoch)")


def guard_action() -> str:
    a = (os.environ.get("IC_GUARD_ACTION") or "terminate").strip().lower()
    if a != "terminate":
        _log("guard_action_ignored", requested=a, using="terminate",
             why="a stopped pod keeps billing its disk; this guard only terminates")
    return "terminate"


def state(now: float | None = None) -> dict:
    now = time.time() if now is None else now
    rate = _f("IC_HOURLY_USD")
    before = _f("IC_SPENT_BEFORE_USD")  # required, no default on purpose
    start = pod_start_epoch()
    hours = max(0.0, (now - start) / 3600.0)
    session = hours * rate
    total = before + session
    soft = _f("IC_SOFT_CAP_USD", 13.5)
    hard = _f("IC_HARD_CAP_USD", 14.75)
    margin = _f("IC_GUARD_MARGIN_USD", 0.25)
    return {
        "hourly_usd": rate, "spent_before_usd": round(before, 4), "spent_before_scope": "ic-em pods only",
        "session_hours": round(hours, 4), "session_usd": round(session, 4), "cumulative_usd": round(total, 4),
        "soft_cap_usd": soft, "hard_cap_usd": hard, "margin_usd": margin,
        "may_start_new_step": total < soft,
        "hard_cap_reached": total >= hard - margin,
        "hours_left_to_soft_cap": round(max(0.0, (soft - total) / rate), 3) if rate > 0 else None,
        "hours_left_to_hard_cap": round(max(0.0, (hard - margin - total) / rate), 3) if rate > 0 else None,
        "pod_id": os.environ.get("RUNPOD_POD_ID"), "now_epoch": round(now, 1),
    }


def may_start_step(projected_usd: float = 0.0, now: float | None = None) -> tuple[bool, dict]:
    """A new stage may start only if cumulative < soft cap AND cumulative + projected < hard cap - margin."""
    s = state(now)
    ok = s["may_start_new_step"] and (s["cumulative_usd"] + projected_usd) < (s["hard_cap_usd"] - s["margin_usd"])
    if (state_dir() / "HARD_CAP_REACHED").exists():
        ok = False
        s["hard_cap_flag"] = True
    s["projected_step_usd"] = round(projected_usd, 3)
    s["decision"] = "start" if ok else "blocked"
    return ok, s


class HardCapReached(RuntimeError):
    pass


def assert_under_hard_cap() -> dict:
    s = state()
    if s["hard_cap_reached"] or (state_dir() / "HARD_CAP_REACHED").exists():
        raise HardCapReached(json.dumps(s))
    return s


# --------------------------------------------------------------------------- logging, packing, terminating
def _log(event: str, **kw) -> None:
    d = state_dir()
    try:
        d.mkdir(parents=True, exist_ok=True)
        rec = {"t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "event": event, **kw}
        with open(d / "spend_guard.log.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")
    except OSError:
        rec = {"event": event, **kw}
    print(json.dumps(rec, default=str), flush=True)


def _status_event(text: str) -> None:
    """Best-effort line in status.json (the dash); never raises."""
    try:
        import sys
        here = Path(__file__).resolve().parent
        sys.path.insert(0, str(here.parent / "src"))
        from ic import status as st  # noqa: PLC0415
        st.update(event_text=text)
    except Exception:  # noqa: BLE001
        pass


def final_pack(dry_run: bool = False) -> Path | None:
    """Tar results/, logs/, state/ and status.json (files < 1 GB) into $IC_ROOT/outbox/guard_pack_<ts>.tgz.
    Local only: nothing leaves the pod here (the orchestrator fetches it, if the pod still exists)."""
    root = ic_root()
    out = root / "outbox"
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    path = out / f"guard_pack_{stamp}.tgz"
    members: list[tuple[Path, str]] = []
    skipped: list[str] = []
    for base, arc in ((results_dir(), "results"), (root / "logs", "logs"), (state_dir(), "state")):
        if not base.is_dir():
            continue
        for p in sorted(base.rglob("*")):
            if not p.is_file() or out in p.parents or p.name.endswith(".lock"):
                continue
            if p.stat().st_size < PACK_MAX_BYTES:
                members.append((p, f"{arc}/{p.relative_to(base).as_posix()}"))
            else:
                skipped.append(str(p))
    st = root / "status.json"
    if st.is_file():
        members.append((st, "status.json"))
    if dry_run:
        _log("final_pack_dry_run", would_write=str(path), n_files=len(members), skipped_ge_1gb=skipped)
        return None
    try:
        out.mkdir(parents=True, exist_ok=True)
        with tarfile.open(path, "w:gz") as tar:
            for p, arc in members:
                try:
                    tar.add(p, arcname=arc)
                except OSError as exc:  # a file vanished mid-pack: keep going
                    skipped.append(f"{p} ({exc.__class__.__name__})")
            blob = json.dumps({"skipped": skipped}, indent=1).encode()
            ti = tarfile.TarInfo("PACK_SKIPPED.json")
            ti.size, ti.mtime = len(blob), int(time.time())
            tar.addfile(ti, io.BytesIO(blob))
        _log("final_pack", path=str(path), bytes=path.stat().st_size, n_files=len(members), skipped=skipped)
        return path
    except Exception as exc:  # noqa: BLE001
        _log("final_pack_error", error=repr(exc))
        return None


def _rest_delete(pod: str, key: str, timeout: float = 60.0) -> int:
    req = urllib.request.Request(REST_BASE + pod, method="DELETE",
                                 headers={"Authorization": f"Bearer {key}", "User-Agent": USER_AGENT,
                                          "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (fixed https endpoint)
        return int(resp.status)


def terminate_plan() -> list[dict]:
    """The ordered terminate attempts this pod would make (never a stop)."""
    pod = os.environ.get("RUNPOD_POD_ID")
    plan: list[dict] = []
    if not pod:
        return plan
    if os.environ.get("RUNPOD_API_KEY"):
        plan.append({"kind": "rest", "method": "DELETE", "url": REST_BASE + pod,
                     "headers": {"Authorization": "Bearer <RUNPOD_API_KEY>", "User-Agent": USER_AGENT}})
    if shutil.which("runpodctl"):
        plan.append({"kind": "cli", "cmd": ["runpodctl", "remove", "pod", pod]})
        plan.append({"kind": "cli", "cmd": ["runpodctl", "pod", "delete", pod]})
    return plan


def terminate_pod(dry_run: bool = False, *, rest=_rest_delete, run=subprocess.run) -> bool:
    """Try every terminate method in order; True if one succeeded. NEVER stops a pod."""
    pod = os.environ.get("RUNPOD_POD_ID")
    if not pod:
        _log("terminate_failed", reason="RUNPOD_POD_ID not set (not inside a RunPod pod?)")
        return False
    plan = terminate_plan()
    if not plan:
        _log("terminate_failed", reason="neither RUNPOD_API_KEY nor runpodctl available; TERMINATE BY HAND")
        return False
    if dry_run:
        for step in plan:
            _log("dry_run_would_terminate", **step)
        if not any(s["kind"] == "rest" for s in plan):
            _log("dry_run_warning", reason="RUNPOD_API_KEY unset: no REST DELETE path")
        return True
    for step in plan:
        try:
            if step["kind"] == "rest":
                code = rest(pod, os.environ["RUNPOD_API_KEY"])
                _log("terminate_rest", status=code)
                if 200 <= code < 300:
                    return True
            else:
                r = run(step["cmd"], capture_output=True, text=True, timeout=60)
                _log("terminate_cli", cmd=step["cmd"], rc=r.returncode, out=(r.stdout or "")[-300:],
                     err=(r.stderr or "")[-300:])
                if r.returncode == 0:
                    return True
        except Exception as exc:  # noqa: BLE001
            _log("terminate_attempt_error", step=step.get("kind"), error=repr(exc)[:300])
    _log("terminate_failed", reason="no method succeeded; TERMINATE THE POD BY HAND IN THE CONSOLE")
    return False


def watchdog(interval: float, dry_run: bool, now_fn=time.time, sleep=time.sleep, max_loops: int | None = None,
             terminate=terminate_pod) -> int:
    action = guard_action()
    _log("watchdog_start", action=action, dry_run=dry_run, pid=os.getpid(), **state(now_fn()))
    loops = 0
    while True:
        s = state(now_fn())
        if s["hard_cap_reached"]:
            _log("hard_cap_reached", **s)
            state_dir().mkdir(parents=True, exist_ok=True)
            (state_dir() / "HARD_CAP_REACHED").write_text(json.dumps(s))
            _status_event(f"spend_guard: HARD CAP ${s['cumulative_usd']:.2f} >= ${s['hard_cap_usd'] - s['margin_usd']:.2f}"
                          f"; packing + TERMINATING{' (dry run)' if dry_run else ''}")
            sleep(0 if dry_run else 20)  # running stages see the flag (run_all guard_check) before the pack
            final_pack(dry_run=dry_run)
            retries = int(_f("IC_GUARD_RETRIES", 10.0))
            for attempt in range(retries + 1):
                if terminate(dry_run=dry_run):
                    return EXIT_HARD_CAP
                _log("terminate_retry_wait", attempt=attempt + 1, of=retries)
                sleep(_f("IC_GUARD_RETRY_S", 120.0))
            return EXIT_HARD_CAP
        if not s["may_start_new_step"] and not (state_dir() / "SOFT_CAP_REACHED").exists():
            state_dir().mkdir(parents=True, exist_ok=True)
            (state_dir() / "SOFT_CAP_REACHED").write_text(json.dumps(s))
            _log("soft_cap_reached", **s)
            _status_event(f"spend_guard: soft cap ${s['soft_cap_usd']:.2f} reached; no new stage will start")
        loops += 1
        if max_loops is not None and loops >= max_loops:
            return 0
        sleep(0 if dry_run else interval)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["status", "check-step", "watchdog", "stop-now"])
    ap.add_argument("--interval", type=float, default=30.0)
    ap.add_argument("--projected-usd", type=float, default=0.0)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "status":
        s = state()
        s["terminate_plan"] = terminate_plan()
        print(json.dumps(s, indent=1))
        return 0
    if a.cmd == "check-step":
        ok, s = may_start_step(a.projected_usd)
        print(json.dumps(s, indent=1))
        return 0 if ok else EXIT_STEP_BLOCKED
    if a.cmd == "stop-now":
        guard_action()
        final_pack(dry_run=a.dry_run)
        return 0 if terminate_pod(dry_run=a.dry_run) else 1
    return watchdog(a.interval, a.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
