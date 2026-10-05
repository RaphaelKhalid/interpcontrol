#!/usr/bin/env bash
# interpcontrol/em/pod/launch.sh: the ONE detached entry point on the pod, so no human step sits in the billed
# path between "bundle extracted" and "pipeline running":
#   1. bootstrap.sh       (pip, 74 GB of downloads with a heartbeat and a 45-min bound, labeler + import checks)
#   2. spend_guard.py stop-now --dry-run   (logs the intended terminate; never sends it)
#   3. spend_guard.py watchdog             (hard-cap self-terminate; the only script-driven pod action)
#   4. run_all.sh                          (stages, outbox, DONE)
#
# STATUS: NOT RUN ON A POD (bash -n + laptop tests only). It never creates, stops or terminates a pod itself.
#
# Launch from the laptop (orchestrator), detached so an ssh drop cannot kill it:
#   rsh '(setsid nohup env IC_SPENT_BEFORE_USD=0 IC_POD_START_EPOCH=<create-pod unix time> \
#          IC_HARD_CAP_USD=<spent_before + 7.2> bash /workspace/ic/code/interpcontrol/em/pod/launch.sh \
#          > /workspace/ic/launch.out 2>&1 < /dev/null &)'
# then poll /workspace/ic/BOOTSTRAP_DONE, /workspace/ic/status.json and finally /workspace/ic/DONE.
#
# IC_SPENT_BEFORE_USD = spend of EARLIER ic-em pods only (0 for the first pod), never the H-lens chat's pods.
# If bootstrap fails, $IC_ROOT/DONE is written with outcome "failed" so the laptop syncer / watcher sees the
# end state and the orchestrator can TERMINATE the pod (nothing else will run).
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export IC_ROOT="${IC_ROOT:-/workspace/ic}"
mkdir -p "$IC_ROOT"
log() { echo "[launch $(date -u +%H:%M:%S)] $*"; }

: "${IC_SPENT_BEFORE_USD:?set IC_SPENT_BEFORE_USD (spend of earlier ic-em pods only; 0 for the first pod)}"
[[ -n "${IC_POD_START_EPOCH:-}" ]] || log "WARNING: IC_POD_START_EPOCH unset; bootstrap will assume now - 600 s"

if [[ -f "$IC_ROOT/DONE" ]]; then log "$IC_ROOT/DONE exists ($(cut -f1 "$IC_ROOT/DONE")); nothing to do"; exit 0; fi

if [[ -f "$IC_ROOT/BOOTSTRAP_DONE" ]]; then
  log "bootstrap already done ($(cat "$IC_ROOT/BOOTSTRAP_DONE")); skipping it"
else
  log "bootstrap start"
  IC_FROM_LAUNCHER=1 bash "$HERE/bootstrap.sh"
  rc=$?
  if [[ $rc -ne 0 || ! -f "$IC_ROOT/BOOTSTRAP_DONE" ]]; then
    log "bootstrap FAILED rc=$rc: writing DONE=failed (orchestrator: fetch $IC_ROOT/state/bootstrap.log, then TERMINATE)"
    printf 'failed\t%s\tbootstrap failed rc=%s (see state/bootstrap.log)\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" > "$IC_ROOT/DONE"
    exit 1
  fi
fi

# shellcheck disable=SC1091
source "$IC_ROOT/env.sh"
PY="${PY:-$(command -v python3 || command -v python)}"
mkdir -p "$IC_STATE"

log "spend guard dry-run (log only; nothing is sent)"
"$PY" "$HERE/spend_guard.py" stop-now --dry-run > "$IC_STATE/guard_dryrun.out" 2>&1
log "dry-run rc=$? (see $IC_STATE/guard_dryrun.out)"
grep -q '"DELETE"' "$IC_STATE/guard_dryrun.out" \
  || log "WARNING: the dry run shows no REST DELETE path (RUNPOD_API_KEY unset?): the laptop watcher is the only stop path"

if [[ -f "$IC_STATE/watchdog.pid" ]] && kill -0 "$(cat "$IC_STATE/watchdog.pid")" 2>/dev/null; then
  log "watchdog already running (pid $(cat "$IC_STATE/watchdog.pid"))"
else
  launcher=""; command -v setsid >/dev/null 2>&1 && launcher="setsid"
  $launcher nohup "$PY" "$HERE/spend_guard.py" watchdog --interval 30 >> "$IC_STATE/watchdog.out" 2>&1 < /dev/null &
  echo $! > "$IC_STATE/watchdog.pid"
  log "watchdog pid $(cat "$IC_STATE/watchdog.pid") (hard cap \$${IC_HARD_CAP_USD:-?}, action terminate)"
fi

log "run_all start"
exec bash "$HERE/run_all.sh" >> "$IC_ROOT/run_all.log" 2>&1 < /dev/null
