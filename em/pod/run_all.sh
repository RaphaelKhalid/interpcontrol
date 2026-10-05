#!/usr/bin/env bash
# interpcontrol/em/pod/run_all.sh: the stage runner for the ONE pod run (SPEC section 11).
#
# STATUS: NOT RUN ON A POD. Exercised on the laptop with stub ic.* modules and a stub spend guard
# (tests/test_pod_infra.py) and, opt-in, end-to-end with the REAL ic.* modules on the tiny model
# (smoke.py --pod-path). Normally started by pod/launch.sh (bootstrap -> guard dry-run -> watchdog ->
# this script, all detached). By hand:
#   (setsid nohup bash /workspace/ic/code/interpcontrol/em/pod/run_all.sh > /workspace/ic/run_all.log 2>&1 < /dev/null &)
#
# Stages (wrapper done-markers in $IC_ROOT/done/<name>):
#   setup     python -m ic.status init                                  (non-fatal)
#   u_r0      python -m ic.generate u_all --stages sanity,gen_r0,readout  (ONE U load; ic.generate
#             writes its own sub-markers sanity / gen_r0 / readout). G1 sanity runs FIRST so its result
#             can gate Round 1; ic.generate catches a sanity crash (logged, no marker), so it can never
#             cost Round 0. Meanwhile a background `ic.label round0 --follow` labels the generations as
#             they land (live vulnerable count on the dash).
#   label     python -m ic.label all          (finishes labels, resumable; semgrep second opinion)
#   bb        python -m ic.monitor score      (T, after U's process exited and the GPU shows < 1 GB)
#   u_r1      python -m ic.generate u_all --stages gen_r1,read_r1     (S2, budget-flex; 2nd U load)
#   label_r1  python -m ic.label round1
#   bb_r1     python -m ic.monitor score      (resumable: scores only the new Round-1 items)
#   analyze   python -m ic.analyze all        (RESULT.md + status money_shot)
# Why not SPEC's 9 separate processes: U would load 5 times; SPEC 11 also says "U is loaded once
# for gen_r0+readout(+gen_r1+read_r1)". Why two U loads instead of one: the primary (Round 0 + BB)
# is finished and T-scored BEFORE any Round-1 money is spent, exactly as SPEC's stage order has it.
#
# Rules enforced
#   * a finished stage is never re-run (wrapper markers + ic.generate's sub-markers); NO retries.
#   * `spend_guard.py check-step --projected-usd X` before every stage; blocked (rc 3) or a broken
#     guard (any other rc) => the stage is skipped and the pipeline finishes ("budget").
#   * every stage runs under `timeout`, so a hung stage cannot burn money until the hard cap.
#   * Round 1 starts only on a clean primary chain, with a projection from MEASURED Round-0 speed
#     (ic.generate also gates gen_r1 itself), and NOT if G1 grossly failed (results/sanity_G1.json
#     status "gross_fail": the base-fit lens does not transport to U, so Round 1's lens-direction
#     ablation would test a broken lens; PREREG s7).
#   * the end, in EVERY case: results + logs packed into $IC_ROOT/outbox (tgz + sha256 + 5 MB parts),
#     then the terminal marker $IC_ROOT/DONE, whose first field is the outcome:
#       ok       every stage done          partial  primary done; Round 1 skipped/failed or a
#       budget   spend guard blocked a stage        non-fatal stage (setup/sanity) failed
#       failed   a primary stage (gen_r0, readout, label, bb, analyze) failed
#     DONE means "pipeline exited: fetch the outbox, then TERMINATE the pod", not "success".
#   * NEVER creates, stops or terminates a pod. If no spend_guard watchdog is running it starts one
#     (its hard-cap self-terminate is the only script-driven pod action, SPEC 11).
#
# Env: from $IC_ROOT/env.sh (bootstrap.sh). IC_SPENT_BEFORE_USD is required: spend on EARLIER ic-em
# pods only (0 for this first pod), never the H-lens chat's pods on the same account. Estimates
# (minutes) IC_EST_<STAGE>, timeouts (minutes) IC_TMAX_<STAGE>. IC_SKIP_R1=1 skips Round 1.
# IC_NO_AUTOWATCHDOG=1 (tests only) skips the watchdog auto-start.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EM="$(cd "$HERE/.." && pwd)"
CODE_ROOT="$(cd "$EM/../.." && pwd)"
export IC_ROOT="${IC_ROOT:-/workspace/ic}"
# shellcheck disable=SC1091
[[ -f "$IC_ROOT/env.sh" ]] && source "$IC_ROOT/env.sh"
export IC_STATE="${IC_STATE:-$IC_ROOT/state}" IC_RESULTS="${IC_RESULTS:-$IC_ROOT/results}"
export IC_LOGS="${IC_LOGS:-$IC_ROOT/logs}" IC_DONE_DIR="${IC_DONE_DIR:-$IC_ROOT/done}"
export IC_STATUS_PATH="${IC_STATUS_PATH:-$IC_ROOT/status.json}"
export IC_HOURLY_USD="${IC_HOURLY_USD:-1.59}"
export PYTHONPATH="$EM/src:$CODE_ROOT/flagship/src:$CODE_ROOT/flagship/third_party${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
PY="${PY:-$(command -v python3 || command -v python)}"
GUARD="${IC_SPEND_GUARD:-$HERE/spend_guard.py}"
export IC_SPEND_GUARD="$GUARD"
CFG="$EM/prereg/config.json"
GEN_R0_FILE="${IC_GEN_R0:-$IC_RESULTS/gen_round0.jsonl}"
mkdir -p "$IC_STATE" "$IC_RESULTS" "$IC_LOGS" "$IC_DONE_DIR"

log() { echo "[run_all $(date -u +%H:%M:%S)] $*"; }

# status.json merge-writer (dash schema; same helper as bootstrap.sh). Only phase/events/pod/spend
# fields; counts/auroc/money_shot belong to ic.status. Called at stage boundaries only. Holds the same
# flock on status.json.lock as ic.status, so concurrent writers never drop each other's updates.
event() {
  "$PY" - "$IC_STATUS_PATH" "$1" "$2" <<'PY' || true
import json, os, sys, time
path, phase, msg = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    import fcntl
    _lk = open(path + ".lock", "a")
    fcntl.flock(_lk.fileno(), fcntl.LOCK_EX)   # released at process exit
except Exception:
    pass
PHASES = ["setup", "sanity", "gen_r0", "label", "readout", "bb", "gen_r1", "readout_r1", "analyze", "done"]
try:
    with open(path, encoding="utf-8") as f:
        s = json.load(f)
    if not isinstance(s, dict):
        s = {}
except Exception:
    s = {}
env = os.environ
def num(k, d=None):
    try:
        return float(env[k])
    except Exception:
        return d
s.setdefault("phases", PHASES)
for k, d in (("counts", {}), ("attempts", 0), ("validated", 0), ("auroc", {}), ("money_shot", None)):
    s.setdefault(k, d)
pod = s.get("pod") if isinstance(s.get("pod"), dict) else {}
start = num("IC_POD_START_EPOCH")
if start is None:
    try:
        start = float(open(os.path.join(env.get("IC_STATE", ""), "pod_start_epoch")).read().strip())
    except Exception:
        start = None
for k, v in (("id", env.get("RUNPOD_POD_ID")), ("gpu", env.get("IC_GPU_NAME")),
             ("rate_usd_h", num("IC_HOURLY_USD")), ("started_at", start)):
    if pod.get(k) is None and v is not None:
        pod[k] = v
pod.setdefault("terminated_at", None)
s["pod"] = pod
if s.get("spent_before_usd") is None and num("IC_SPENT_BEFORE_USD") is not None:
    s["spent_before_usd"] = num("IC_SPENT_BEFORE_USD")
s.setdefault("cap_usd", num("IC_CAP_USD", 15.0))
s.setdefault("trial_cap_usd", 3.0)
if phase != "-":
    s["phase"] = phase
s.setdefault("phase", "setup")
ev = s.get("events") if isinstance(s.get("events"), list) else []
ev.append(time.strftime("[%H:%M] ", time.localtime()) + " ".join(msg.split())[:150])
s["events"] = ev[-200:]
s["updated"] = time.time()
tmp = f"{path}.tmp.{os.getpid()}"
with open(tmp, "w", encoding="utf-8") as f:
    json.dump(s, f)
os.replace(tmp, path)
PY
}

usd() { awk -v m="$1" -v r="$IC_HOURLY_USD" 'BEGIN { printf "%.3f", m / 60.0 * r }'; }
have() { [[ -f "$IC_DONE_DIR/$1" ]]; }

# ------------------------------------------------------------------ outbox + terminal marker
pack_outbox() {
  "$PY" - "$IC_ROOT" <<'PY' || log "WARNING: outbox packing failed"
import hashlib, io, os, sys, tarfile, time
root = sys.argv[1]
out = os.path.join(root, "outbox")
os.makedirs(out, exist_ok=True)
tgz = os.path.join(out, "ic_results.tgz")
LIMIT = int(os.environ.get("IC_OUTBOX_MAX_FILE_BYTES", str(1024**3)))   # 1 GB per file (the volume is 10 GB)
# (path, arcname). The stage markers go under state/done/ and DONE itself is left out (EXIT_STATUS has
# the same line): "done/" and "DONE" would collide when the tarball is unpacked on Windows.
members, manifest, excluded = [], [], []
for sub, arc in (("results", "results"), ("logs", "logs"), ("state", "state"),
                 (os.environ.get("IC_DONE_DIR") or os.path.join(root, "done"), "state/done")):
    base = sub if os.path.isabs(sub) else os.path.join(root, sub)
    for dp, _, fs in os.walk(base):
        for f in sorted(fs):
            p = os.path.join(dp, f)
            if f.endswith(".lock"):
                continue
            rel = arc + "/" + os.path.relpath(p, base).replace(os.sep, "/")
            if os.path.getsize(p) < LIMIT:
                members.append((p, rel))
            else:
                excluded.append(f"{rel} ({os.path.getsize(p)} bytes >= {LIMIT})")
for f in ("status.json", "run_all.log", "EXIT_STATUS", "BOOTSTRAP_DONE"):
    p = os.path.join(root, f)
    if os.path.isfile(p):
        members.append((p, f))
with tarfile.open(tgz, "w:gz") as tar:
    for p, rel in members:
        data = open(p, "rb").read()
        manifest.append(f"{hashlib.sha256(data).hexdigest()}  {rel}")
        ti = tarfile.TarInfo(rel)
        ti.size, ti.mtime, ti.mode = len(data), int(os.path.getmtime(p)), 0o644
        tar.addfile(ti, io.BytesIO(data))
    manifest += [f"# EXCLUDED (too large, still on the pod): {e}" for e in excluded]
    blob = ("\n".join(manifest) + "\n").encode()
    ti = tarfile.TarInfo("MANIFEST.sha256")
    ti.size, ti.mtime, ti.mode = len(blob), int(time.time()), 0o644
    tar.addfile(ti, io.BytesIO(blob))
whole = open(tgz, "rb").read()
with open(tgz + ".sha256", "w", encoding="utf-8", newline="\n") as fh:
    fh.write(f"{hashlib.sha256(whole).hexdigest()}  ic_results.tgz\n")
for old in os.listdir(out):
    if old.startswith("ic_results.tgz.part"):
        os.remove(os.path.join(out, old))
part_lines, n = [], 5 * 1024 * 1024
for i in range(0, max(1, len(whole)), n):
    name = f"ic_results.tgz.part{i // n:03d}"
    chunk = whole[i:i + n]
    with open(os.path.join(out, name), "wb") as fh:
        fh.write(chunk)
    part_lines.append(f"{hashlib.sha256(chunk).hexdigest()}  {name}")
with open(os.path.join(out, "parts.sha256"), "w", encoding="utf-8", newline="\n") as fh:
    fh.write("\n".join(part_lines) + "\n")
with open(os.path.join(out, "EXCLUDED.txt"), "w", encoding="utf-8", newline="\n") as fh:
    fh.write("".join(e + "\n" for e in excluded))
print(f"outbox: {tgz} {len(whole)} bytes, {len(members)} files, {len(part_lines)} parts"
      + (f"; EXCLUDED {len(excluded)} file(s) >= {LIMIT} bytes (fetch by hand before terminating): {excluded}"
         if excluded else ""), flush=True)
PY
}

finish() {  # finish <outcome> <message>
  local outcome="$1" msg="$2"
  stop_bg_label
  log "FINISH outcome=$outcome :: $msg"
  printf '%s\t%s\t%s\n' "$outcome" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$msg" > "$IC_ROOT/EXIT_STATUS"
  event done "pipeline finished: $outcome :: $msg"
  pack_outbox
  if [[ -s "$IC_ROOT/outbox/EXCLUDED.txt" ]]; then
    msg="$msg; outbox EXCLUDED large files (see outbox/EXCLUDED.txt; fetch them before terminating)"
    event "-" "outbox excluded $(wc -l < "$IC_ROOT/outbox/EXCLUDED.txt") large file(s): see outbox/EXCLUDED.txt"
  fi
  printf '%s\t%s\t%s\n' "$outcome" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$msg" > "$IC_ROOT/DONE"
  log "wrote $IC_ROOT/DONE. Orchestrator: fetch $IC_ROOT/outbox/ic_results.tgz.part* + parts.sha256 + ic_results.tgz.sha256, verify, then TERMINATE the pod."
  [[ "$outcome" == "failed" ]] && exit 1
  exit 0
}

# ------------------------------------------------------------------ spend guard + GPU + labeler helpers
guard_check() {  # guard_check <projected_usd> -> 0 = may start
  [[ -f "$IC_STATE/HARD_CAP_REACHED" ]] && { log "HARD_CAP_REACHED flag present"; return 4; }
  "$PY" "$GUARD" check-step --projected-usd "$1" > "$IC_STATE/guard_last.json" 2>&1
  local rc=$?
  [[ $rc -ne 0 ]] && { log "spend guard says no (rc=$rc):"; tail -n 20 "$IC_STATE/guard_last.json" | sed 's/^/    /'; }
  return $rc
}

ensure_watchdog() {
  [[ "${IC_NO_AUTOWATCHDOG:-0}" == "1" ]] && { log "watchdog auto-start disabled (IC_NO_AUTOWATCHDOG=1)"; return 0; }
  if command -v pgrep >/dev/null 2>&1 && pgrep -f "spend_guard.py watchdog" >/dev/null 2>&1; then
    log "spend_guard watchdog already running"; return 0
  fi
  if [[ -f "$IC_STATE/watchdog.pid" ]] && kill -0 "$(cat "$IC_STATE/watchdog.pid")" 2>/dev/null; then
    log "spend_guard watchdog already running (pid file)"; return 0
  fi
  local launcher=""; command -v setsid >/dev/null 2>&1 && launcher="setsid"
  $launcher nohup "$PY" "$GUARD" watchdog --interval 30 >> "$IC_STATE/watchdog.out" 2>&1 < /dev/null &
  echo $! > "$IC_STATE/watchdog.pid"
  log "started spend_guard watchdog pid $(cat "$IC_STATE/watchdog.pid") (action ${IC_GUARD_ACTION:-terminate})"
  event "-" "run_all started the spend watchdog (none was running)"
}

wait_gpu_free() {  # T must not load next to a live U: < 1 GB used, wait up to IC_GPU_FREE_WAIT_S (120 s), then warn
  command -v nvidia-smi >/dev/null 2>&1 || { log "no nvidia-smi; skipping GPU-free check"; return 0; }
  local used="" i n=$(( ${IC_GPU_FREE_WAIT_S:-120} / 5 ))
  for (( i = 0; i <= n; i++ )); do
    used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc 0-9)"
    if [[ -n "$used" && "$used" -lt 1024 ]]; then log "GPU free before T (${used} MiB used)"; return 0; fi
    (( i < n )) && sleep 5
  done
  log "WARNING: GPU still holds ${used:-?} MiB before T loads:"
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv 2>&1 | sed 's/^/    /'
  event "-" "WARNING GPU not freed (${used:-?} MiB) before T"
  return 0
}

BG_LABEL_PID=""
start_bg_label() {  # head start only: label Round-0 generations while U runs (the label stage finishes it)
  have label && return 0
  [[ "${IC_NO_BG_LABEL:-0}" == "1" ]] && return 0
  rm -f "$IC_DONE_DIR/u_r0.exited"
  bash -c '
    while [[ ! -s "$1" && ! -f "$2" ]]; do sleep 10; done
    [[ -s "$1" ]] || exit 0
    cd "$3" && exec "$4" -m ic.label round0 --follow --done-marker "$2"
  ' _ "$GEN_R0_FILE" "$IC_DONE_DIR/u_r0.exited" "$EM" "$PY" > "$IC_LOGS/label_follow.log" 2>&1 < /dev/null &
  BG_LABEL_PID=$!
  log "background labeler pid $BG_LABEL_PID (follows $GEN_R0_FILE until u_r0 exits)"
}

stop_bg_label() {  # wait (up to IC_BG_LABEL_WAIT_S, default 600 s) for the follower to drain, then kill it
  [[ -z "$BG_LABEL_PID" ]] && return 0
  touch "$IC_DONE_DIR/u_r0.exited"
  local i waited=0 lim="${IC_BG_LABEL_WAIT_S:-600}"
  while kill -0 "$BG_LABEL_PID" 2>/dev/null && [[ $waited -lt $lim ]]; do sleep 2; waited=$(( waited + 2 )); done
  if kill -0 "$BG_LABEL_PID" 2>/dev/null; then
    log "WARNING: background labeler still running after ${lim}s; killing it (the label stage redoes what is missing)"
    command -v pkill >/dev/null 2>&1 && pkill -TERM -P "$BG_LABEL_PID" 2>/dev/null
    kill -TERM "$BG_LABEL_PID" 2>/dev/null
    sleep 3
    command -v pkill >/dev/null 2>&1 && pkill -KILL -P "$BG_LABEL_PID" 2>/dev/null
    kill -KILL "$BG_LABEL_PID" 2>/dev/null
  fi
  wait "$BG_LABEL_PID" 2>/dev/null; local rc=$?
  log "background labeler exited rc=$rc (log: $IC_LOGS/label_follow.log)"
  BG_LABEL_PID=""
}

g1_status() {  # status field of results/sanity_G1.json (pass | marginal | gross_fail | error), "" if absent
  "$PY" -c 'import json,sys
try:
    print(json.load(open(sys.argv[1], encoding="utf-8")).get("status", ""))
except Exception:
    print("")' "$IC_RESULTS/sanity_G1.json" 2>/dev/null || echo ""
}

r1_projection_usd() {  # $ of the whole Round-1 block, from MEASURED Round-0 speed
  "$PY" - "$CFG" "$IC_RESULTS/stage_timings.json" "$IC_STATE/stage_times.tsv" "$IC_HOURLY_USD" <<'PY' 2>/dev/null || usd "${IC_EST_R1_BLOCK:-90}"
import json, os, sys
cfg, timings, tsv, rate = json.load(open(sys.argv[1])), sys.argv[2], sys.argv[3], float(sys.argv[4])
g = cfg["generation"]
n0 = (cfg["data"]["cyberseceval"]["n_python"] + cfg["data"]["em_heldout_control"]["n"]) * g["round0"]["k0"]
r1 = g["round1_S2"]
n1 = r1["n_prompts"] * r1["k1"] * len(r1["conditions"])
ratio = n1 / n0
wrap = {}
if os.path.exists(tsv):
    for line in open(tsv, encoding="utf-8"):
        p = line.rstrip("\n").split("\t")
        if len(p) >= 3:
            wrap[p[0]] = float(p[2])
sec = None
try:  # ic.generate's own per-item timings (elapsed_s / new items)
    t = json.load(open(timings, encoding="utf-8"))
    per_g = t["gen_r0"]["elapsed_s"] / max(t["gen_r0"]["new"], 1)
    per_r = t["readout_r0"]["elapsed_s"] / max(t["readout_r0"]["new"], 1)
    sec = n1 * per_g * 1.25 + n1 * per_r + 300            # +25% hook overhead, +5 min U load
except Exception:
    if "u_r0" in wrap:
        sec = wrap["u_r0"] * ratio * 1.25 + 300
if sec is None:
    raise SystemExit(1)
sec += wrap.get("label", 300) * ratio + wrap.get("bb", 900) * ratio + 120   # label_r1 + bb_r1 (+T load)
sec += 600                                                                   # analyze must still fit
print(f"{sec / 3600.0 * rate:.3f}")
PY
}

# ------------------------------------------------------------------ the stage wrapper
STAGE_RC=0
run_stage() {  # run_stage <name> <dash_phase> <timeout_min> <projected_usd> <cmd...>
  local name="$1" phase="$2" tmin="$3" proj="$4"; shift 4
  if have "$name"; then log "skip $name (done-marker exists)"; STAGE_RC=0; return 0; fi
  if ! guard_check "$proj"; then
    log "BLOCKED $name (projected \$$proj)"; event "-" "stage $name BLOCKED by spend guard (projected \$$proj)"
    echo "$name" >> "$IC_STATE/blocked_stages"; STAGE_RC=3; return 3
  fi
  log "start $name (timeout ${tmin} min, projected \$$proj): $*"
  event "$phase" "stage $name start"
  touch "$IC_STATE/stage_${name}.start"
  local t0; t0=$(date +%s)
  ( cd "$EM" && timeout --kill-after=60 "${tmin}m" "$@" ) > "$IC_LOGS/$name.log" 2>&1
  local rc=$?
  local dt=$(( $(date +%s) - t0 ))
  printf '%s\t%s\t%s\t%s\n' "$name" "$rc" "$dt" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$IC_STATE/stage_times.tsv"
  if [[ $rc -eq 0 ]]; then
    touch "$IC_DONE_DIR/$name"; log "ok $name (${dt}s)"; event "-" "stage $name ok (${dt}s)"
  else
    local why="rc=$rc"
    [[ $rc -eq 124 || $rc -eq 137 ]] && why="TIMEOUT after ${tmin} min (rc=$rc)"
    [[ $rc -eq 3 ]] && { why="exited 3 (spend guard block inside the stage)"; echo "$name" >> "$IC_STATE/blocked_stages"; }
    log "FAILED $name $why after ${dt}s; last log lines:"; tail -n 30 "$IC_LOGS/$name.log" | sed 's/^/    /'
    event "-" "stage $name FAILED $why (${dt}s)"
  fi
  STAGE_RC=$rc; return $rc
}

# ------------------------------------------------------------------ main
if command -v flock >/dev/null 2>&1; then
  exec 9> "$IC_STATE/run_all.lock"
  flock -n 9 || { log "another run_all.sh holds $IC_STATE/run_all.lock; exiting"; exit 5; }
fi
if [[ -f "$IC_ROOT/DONE" ]]; then log "$IC_ROOT/DONE exists ($(cut -f1 "$IC_ROOT/DONE")); nothing to do (delete it to resume)"; exit 0; fi
log "code=$CODE_ROOT ic_root=$IC_ROOT py=$PY rate=\$$IC_HOURLY_USD/h"
[[ -n "${IC_SPENT_BEFORE_USD:-}" ]] || { log "IC_SPENT_BEFORE_USD unset (source $IC_ROOT/env.sh written by bootstrap.sh)"; event "-" "run_all refused: IC_SPENT_BEFORE_USD unset"; exit 2; }
[[ -f "$GUARD" ]] || { log "spend guard missing: $GUARD"; event "-" "run_all refused: spend guard missing"; exit 2; }
"$PY" "$GUARD" status > "$IC_STATE/guard_status_start.json" 2>&1 || { log "spend guard 'status' failed:"; cat "$IC_STATE/guard_status_start.json"; event "-" "run_all refused: spend guard status failed"; exit 2; }
ensure_watchdog
event "-" "run_all start"

NOTES=""
note() { NOTES="${NOTES:+$NOTES; }$*"; log "NOTE: $*"; }
blocked_finish() { finish budget "stage $1 blocked by the spend guard${NOTES:+; $NOTES}"; }

run_stage setup setup "${IC_TMAX_SETUP:-5}" 0 "$PY" -m ic.status init
case $STAGE_RC in 0) ;; 3) blocked_finish setup ;; *) note "setup (ic.status init) failed rc=$STAGE_RC (non-fatal)" ;; esac

# --- primary (Round 0). A failure never stops the remaining primary stages (they work on what exists),
# but it cancels Round 1 and makes the outcome "failed".
PRIMARY_BROKEN=""
broken() { PRIMARY_BROKEN="${PRIMARY_BROKEN:+$PRIMARY_BROKEN, }$*"; }

start_bg_label
run_stage u_r0 gen_r0 "${IC_TMAX_U_R0:-180}" "$(usd "${IC_EST_U_R0:-95}")" "$PY" -m ic.generate u_all --stages sanity,gen_r0,readout
U0_RC=$STAGE_RC
touch "$IC_DONE_DIR/u_r0.exited"
stop_bg_label
if [[ $U0_RC -eq 3 ]] && ! have gen_r0; then blocked_finish u_r0; fi
if ! have gen_r0; then
  broken "gen_r0 (u_r0 rc=$U0_RC)"
  if [[ -z "$(find "$IC_RESULTS" -type f -name '*.jsonl' -size +0c -newer "$IC_STATE/stage_u_r0.start" 2>/dev/null | head -1)" ]]; then
    finish failed "u_r0 rc=$U0_RC wrote no generations${NOTES:+; $NOTES}"
  fi
fi
have readout || broken "readout (u_r0 rc=$U0_RC)"
have sanity || note "G1 sanity did not complete (u_r0 rc=$U0_RC; non-fatal, WB results exploratory)"

run_stage label label "${IC_TMAX_LABEL:-40}" "$(usd "${IC_EST_LABEL:-5}")" "$PY" -m ic.label all
[[ $STAGE_RC -eq 3 ]] && blocked_finish label
[[ $STAGE_RC -ne 0 ]] && broken "label rc=$STAGE_RC"

have bb || wait_gpu_free
run_stage bb bb "${IC_TMAX_BB:-60}" "$(usd "${IC_EST_BB:-15}")" "$PY" -m ic.monitor score
[[ $STAGE_RC -eq 3 ]] && blocked_finish bb
[[ $STAGE_RC -ne 0 ]] && broken "bb rc=$STAGE_RC"

# --- Round 1 (S2, budget-flex): only on a clean primary chain and a guard-accepted projection.
R1_STATE="skipped"
if [[ -n "$PRIMARY_BROKEN" ]]; then
  note "Round 1 skipped: primary incomplete ($PRIMARY_BROKEN)"
elif [[ "${IC_SKIP_R1:-0}" == "1" ]]; then
  note "Round 1 skipped: IC_SKIP_R1=1"
elif [[ "$(g1_status)" == "gross_fail" ]]; then
  note "Round 1 skipped: G1 gross failure (the base-fit lens does not transport to U; WB results exploratory)"
  event "-" "Round 1 skipped: G1 gross failure"
else
  R1_USD="$(r1_projection_usd)"
  log "Round 1 block projected at \$$R1_USD (from measured Round-0 speed)"
  run_stage u_r1 gen_r1 "${IC_TMAX_U_R1:-150}" "$R1_USD" "$PY" -m ic.generate u_all --stages gen_r1,read_r1
  U1_RC=$STAGE_RC
  if have gen_r1 && have read_r1; then
    if run_stage label_r1 readout_r1 "${IC_TMAX_LABEL_R1:-20}" "$(usd "${IC_EST_LABEL_R1:-3}")" "$PY" -m ic.label round1 \
       && { have bb_r1 || wait_gpu_free; } \
       && run_stage bb_r1 readout_r1 "${IC_TMAX_BB_R1:-40}" "$(usd "${IC_EST_BB_R1:-10}")" "$PY" -m ic.monitor score; then
      R1_STATE="ok"
    elif [[ $STAGE_RC -eq 3 ]]; then
      R1_STATE="blocked"; note "Round 1 labels/BB blocked by the spend guard"
    else
      R1_STATE="failed"; note "Round 1 label/BB stage failed rc=$STAGE_RC"
    fi
  elif [[ $U1_RC -eq 3 ]]; then
    R1_STATE="blocked"; note "Round 1 blocked by the spend guard (primary data is complete)"
  else
    R1_STATE="failed"; note "Round 1 U stage incomplete (u_r1 rc=$U1_RC; gen_r1=$(have gen_r1 && echo done || echo missing), read_r1=$(have read_r1 && echo done || echo missing))"
  fi
fi

run_stage analyze analyze "${IC_TMAX_ANALYZE:-40}" "$(usd "${IC_EST_ANALYZE:-10}")" "$PY" -m ic.analyze all
ARC=$STAGE_RC

if [[ $ARC -eq 3 ]]; then
  finish budget "analyze blocked by the spend guard (run it on the laptop, \$0)${NOTES:+; $NOTES}"
elif [[ -n "$PRIMARY_BROKEN" || $ARC -ne 0 ]]; then
  finish failed "primary: ${PRIMARY_BROKEN:-ok}; analyze rc=$ARC${NOTES:+; $NOTES}"
elif [[ "$R1_STATE" == "blocked" ]]; then
  finish budget "primary + analyze done; Round 1 blocked by the spend guard${NOTES:+; $NOTES}"
elif [[ "$R1_STATE" != "ok" || -n "$NOTES" ]]; then
  finish partial "primary + analyze done; Round 1 $R1_STATE${NOTES:+; $NOTES}"
else
  finish ok "all stages done"
fi
