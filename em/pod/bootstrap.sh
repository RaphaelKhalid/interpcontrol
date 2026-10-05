#!/usr/bin/env bash
# interpcontrol/em/pod/bootstrap.sh: set up the ONE pod run (A100 80GB, image
# runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404) for interpcontrol/em.
#
# STATUS: NOT RUN ON A POD. Only `bash -n` and the laptop unit tests (tests/test_pod_infra.py)
# have exercised it. Creating/stopping/terminating pods is the ORCHESTRATOR's job; this script
# never does any pod operation (it only reads GET /v1/pods/$RUNPOD_POD_ID once to prove the
# pod-scoped key works for spend_guard's self-terminate path, and logs the HTTP code).
#
# Usage (on the pod, after the bundle from make_bundle.py was extracted, e.g. into /workspace/ic/code).
# Normally via the DETACHED launcher, which then runs the guard dry-run, the watchdog and run_all.sh:
#   (setsid nohup env IC_SPENT_BEFORE_USD=0 IC_POD_START_EPOCH=<create time> \
#      bash /workspace/ic/code/interpcontrol/em/pod/launch.sh > /workspace/ic/launch.out 2>&1 < /dev/null &)
# Never run this through a `timeout 60` ssh helper or a plain foreground ssh: it takes 15-20 min and an
# ssh drop would kill it halfway through pip or the 74 GB download.
#
# Required env:
#   IC_SPENT_BEFORE_USD   $ already spent on EARLIER interpcontrol/em (ic-em) pods: 0 for this first pod.
#                         NOT the account's billing census: the H-lens chat runs pods on the same RunPod
#                         account, and its spend must not count against this experiment's $15 cap (SPEC 11.1).
#                         No silent default: forgetting it would let the guard start at $0.
# Optional env:
#   IC_HOURLY_USD         pod $/h (default: config.json budget.rate_usd_h = 1.59)
#   IC_POD_START_EPOCH    unix time the pod started billing: the orchestrator should pass the create-pod time it
#                         recorded on the laptop (default: bootstrap start - 600 s, logged as a WARNING because
#                         image pull + address polling + upload can exceed 10 min and then spend is undercounted)
#   IC_HARD_CAP_USD       default config 14.75. Recommended for this run: IC_SPENT_BEFORE_USD + 7.2 (about 4.5 h at
#                         $1.59/h) so a pod left idle after DONE is bounded (the guard terminates; SPEC 11.6)
#   IC_ROOT               work dir (default /workspace/ic); results in $IC_ROOT/results
#   IC_T_FALLBACK_REV     revision of the T fallback (default: Qwen2.5-Coder-1.5B-Instruct main sha, 2026-10-04)
#   IC_SKIP_SEMGREP=1     do not install semgrep (descriptive second opinion; own --target dir)
#   IC_BOOT_SANITY=1      also run the G1 lens sanity here (default off: run_all.sh's `sanity`
#                         stage runs it once; running it here would load the 65.5 GB U twice)
#   IC_BOOT_TESTS=1       run the CPU unit tests (tests/) after the import check
#   HF_TOKEN              optional HF read token (all repos are public). Never echoed.
#
# Writes: $IC_ROOT/env.sh (no secrets), $IC_ROOT/state/{bootstrap.log,env_check.json,downloads.json,
#         pod_start_epoch,boot_times.jsonl}, $IC_ROOT/status.json (merge: phase "setup" + events),
#         $IC_ROOT/BOOTSTRAP_DONE.
# Exit codes: 0 ok, 1 environment/version/import failure, 2 bundle incomplete or disk too small.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"          # .../interpcontrol/em/pod
EM="$(cd "$HERE/.." && pwd)"                                    # .../interpcontrol/em
CODE_ROOT="$(cd "$EM/../.." && pwd)"                            # contains interpcontrol/ and flagship/
CFG="$EM/prereg/config.json"

export IC_ROOT="${IC_ROOT:-/workspace/ic}"
export IC_STATE="$IC_ROOT/state" IC_RESULTS="$IC_ROOT/results" IC_LOGS="$IC_ROOT/logs" IC_DONE_DIR="$IC_ROOT/done"
export IC_STATUS_PATH="$IC_ROOT/status.json"
export HF_HOME="${HF_HOME:-/root/hf}" PIP_BREAK_SYSTEM_PACKAGES=1 PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
export HF_XET_HIGH_PERFORMANCE=1 HF_XET_CHUNK_CACHE_SIZE_BYTES=0 HF_HUB_DISABLE_TELEMETRY=1
PY="${PY:-$(command -v python3 || command -v python)}"
mkdir -p "$IC_STATE" "$IC_RESULTS" "$IC_LOGS" "$IC_DONE_DIR" "$HF_HOME"

LOG="$IC_STATE/bootstrap.log"
exec > >(tee -a "$LOG") 2>&1
T0=$(date +%s)
say()   { echo "[bootstrap $(date -u +%H:%M:%S)] $*"; }
stamp() { say "$1 done (+$(( $(date +%s) - T0 )) s)"; echo "{\"step\": \"$1\", \"t\": $(( $(date +%s) - T0 ))}" >> "$IC_STATE/boot_times.jsonl"; }
die()   { local rc=$1; shift; say "FATAL: $*"; bstat "-" "bootstrap FAILED: $*" || true; exit "$rc"; }

# status.json merge-writer (dash schema, interpcontrol/dash/server.py docstring). Same helper as in
# run_all.sh. Only sets phase/events/pod/spend fields; never touches counts/auroc/money_shot values
# written by ic.status. Atomic via os.replace.
bstat() {
  "$PY" - "$IC_STATUS_PATH" "$1" "$2" <<'PY' || true
import json, os, sys, time
path, phase, msg = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    import fcntl
    _lk = open(path + ".lock", "a")
    fcntl.flock(_lk.fileno(), fcntl.LOCK_EX)   # same lock as ic.status; released at process exit
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

# counts[name] = [done, total] (ints; the dash's progress bars + the heartbeat watcher's advancing count).
# No event line (the heartbeat loop calls it every 30 s); same lock as bstat / ic.status.
bcount() {
  "$PY" - "$IC_STATUS_PATH" "$1" "$2" "$3" <<'PY' || true
import json, os, sys, time
path, name, done, total = sys.argv[1], sys.argv[2], int(float(sys.argv[3])), int(float(sys.argv[4]))
try:
    import fcntl
    _lk = open(path + ".lock", "a")
    fcntl.flock(_lk.fileno(), fcntl.LOCK_EX)
except Exception:
    pass
try:
    with open(path, encoding="utf-8") as f:
        s = json.load(f)
    if not isinstance(s, dict):
        s = {}
except Exception:
    s = {}
c = s.get("counts") if isinstance(s.get("counts"), dict) else {}
c[name] = [done, total]
s["counts"] = c
s.setdefault("events", [])
s["updated"] = time.time()
tmp = f"{path}.tmp.{os.getpid()}"
with open(tmp, "w", encoding="utf-8") as f:
    json.dump(s, f)
os.replace(tmp, path)
PY
}

: "${IC_SPENT_BEFORE_USD:?set IC_SPENT_BEFORE_USD to the spend of EARLIER ic-em pods only (0 for the first pod; never the H-lens chat pods)}"
cfgget() { "$PY" -c "import functools,json,sys; print(functools.reduce(lambda d, k: d[k], sys.argv[2].split('.'), json.load(open(sys.argv[1]))))" "$CFG" "$1"; }
[[ -f "$CFG" ]] || { echo "missing $CFG (bundle not extracted correctly?)"; exit 2; }
export IC_HOURLY_USD="${IC_HOURLY_USD:-$(cfgget budget.rate_usd_h)}"
export IC_CAP_USD="${IC_CAP_USD:-$(cfgget budget.cap_usd)}"
export IC_SOFT_CAP_USD="${IC_SOFT_CAP_USD:-$(cfgget budget.soft_cap_usd)}"
export IC_HARD_CAP_USD="${IC_HARD_CAP_USD:-$(cfgget budget.hard_cap_usd)}"
export IC_GUARD_MARGIN_USD="${IC_GUARD_MARGIN_USD:-0.25}"
export IC_GUARD_ACTION="${IC_GUARD_ACTION:-$(cfgget budget.guard_action)}"
[[ "$IC_GUARD_ACTION" == "terminate" ]] || say "WARNING: IC_GUARD_ACTION=$IC_GUARD_ACTION (config says terminate; a STOPPED pod keeps billing its disk)"

# ---------------------------------------------------------------- 0. spend clock + preflight
if [[ ! -f "$IC_STATE/pod_start_epoch" ]]; then
  if [[ -z "${IC_POD_START_EPOCH:-}" ]]; then
    say "WARNING: IC_POD_START_EPOCH not passed; assuming billing began 600 s before bootstrap. If image pull +"
    say "         address polling + upload took longer, the guard and the dash UNDERCOUNT spend: pass the create time."
  fi
  echo "${IC_POD_START_EPOCH:-$(( $(date +%s) - 600 ))}" > "$IC_STATE/pod_start_epoch"
fi
export IC_POD_START_EPOCH="${IC_POD_START_EPOCH:-$(cat "$IC_STATE/pod_start_epoch")}"
export IC_GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)"
say "code=$CODE_ROOT ic_root=$IC_ROOT rate=\$$IC_HOURLY_USD/h before=\$$IC_SPENT_BEFORE_USD start=$IC_POD_START_EPOCH caps soft/hard=$IC_SOFT_CAP_USD/$IC_HARD_CAP_USD action=$IC_GUARD_ACTION"
bstat setup "bootstrap start (gpu: ${IC_GPU_NAME:-?})"
nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv,noheader || say "WARNING: nvidia-smi failed"
df -h /root /workspace 2>/dev/null || df -h /
AVAIL_G=$(df --output=avail -BG /root | tail -1 | tr -dc 0-9)
[[ "${AVAIL_G:-0}" -ge 120 ]] || die 2 "only ${AVAIL_G}G free on /root (< 120G): ENOSPC risk for the 65.5 GB U + T + lens"
say "RUNPOD_* env vars present: $(env | grep -c '^RUNPOD_' || true) (values not printed)"
if command -v runpodctl >/dev/null 2>&1; then say "runpodctl: $(runpodctl version 2>&1 | head -1)"; else say "runpodctl: not on PATH"; fi
# Read-only probe of the pod-scoped key (spend_guard's REST terminate path needs it). Never prints the key.
if [[ -n "${RUNPOD_API_KEY:-}" && -n "${RUNPOD_POD_ID:-}" ]]; then
  CODE=$(curl -s -m 20 -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $RUNPOD_API_KEY" \
         -H "User-Agent: ic-em-bootstrap" "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID" || echo "curl-failed")
  say "pod-key probe GET /v1/pods/\$RUNPOD_POD_ID -> HTTP $CODE (200 = self-terminate path usable)"
  echo "$CODE" > "$IC_STATE/pod_key_probe_http"
  [[ "$CODE" == "200" ]] || bstat "-" "WARNING pod-key probe HTTP $CODE: laptop watcher is the only stop path"
else
  say "WARNING: RUNPOD_API_KEY or RUNPOD_POD_ID unset: spend_guard cannot self-terminate; laptop watcher is the only stop path"
  echo "unset" > "$IC_STATE/pod_key_probe_http"
fi
stamp preflight

# ---------------------------------------------------------------- 1. bundle integrity
cd "$CODE_ROOT"
MAN="interpcontrol/em/pod/BUNDLE_MANIFEST.txt"
if [[ -f "$MAN" ]]; then
  grep -E '^[0-9a-f]{64}  ' "$MAN" | sha256sum -c --quiet - || die 2 "bundle files do not match $MAN"
  say "bundle manifest OK ($(grep -cE '^[0-9a-f]{64}  ' "$MAN") files)"
else
  say "WARNING: no $MAN (bundle not made by make_bundle.py?)"
fi
missing=0
for need in interpcontrol/em/src/ic/__init__.py interpcontrol/em/src/ic/config.py interpcontrol/em/src/ic/data.py \
            interpcontrol/em/src/ic/lens.py interpcontrol/em/src/ic/generate.py interpcontrol/em/src/ic/label.py \
            interpcontrol/em/src/ic/monitor.py interpcontrol/em/src/ic/analyze.py interpcontrol/em/src/ic/status.py \
            interpcontrol/em/pod/run_all.sh interpcontrol/em/pod/spend_guard.py \
            interpcontrol/em/prereg/PREREG.md interpcontrol/em/prereg/config.json interpcontrol/em/prereg/vocab_ids.json \
            interpcontrol/em/prereg/PREREG-HASH.txt interpcontrol/em/data/instruct.json interpcontrol/em/data/em_heldout.jsonl \
            flagship/src/wor/__init__.py flagship/src/wor/_paths.py flagship/src/wor/modeling.py flagship/src/wor/scorer.py \
            flagship/src/wor/ablate.py flagship/src/wor/teacher_force.py flagship/third_party/jlens/__init__.py \
            flagship/third_party/jlens/lens.py flagship/third_party/jlens/hf.py; do
  [[ -f "$need" ]] || { say "bundle incomplete: missing $need"; missing=1; }
done
[[ $missing -eq 0 ]] || die 2 "bundle incomplete (see above)"
# CRLF in a shell script breaks bash on Linux; make_bundle.py normalises, this double-checks.
for sh in interpcontrol/em/pod/*.sh; do
  if grep -q $'\r' "$sh"; then die 2 "$sh has CRLF line endings"; fi
done
stamp bundle

# ---------------------------------------------------------------- 2. python deps (image torch is kept)
"$PY" -c 'import sys; assert sys.version_info[:2] >= (3, 12), sys.version' || die 1 "need Python >= 3.12 (numpy 2.5.2); got $("$PY" -V 2>&1)"
"$PY" -m pip install -q --no-input --disable-pip-version-check -r "$HERE/requirements.txt" || die 1 "pip install -r requirements.txt failed"
stamp pip

SEMGREP_PID=""
if [[ "${IC_SKIP_SEMGREP:-0}" != "1" ]]; then
  # Own --target dir so semgrep's pins (click, mcp, opentelemetry, ...) cannot touch the model stack.
  # Rules: the bundled interpcontrol/em/data/semgrep/python_cyberseceval.json (label.py's default).
  ( "$PY" -m pip install -q --no-input --disable-pip-version-check --target /root/semgrep-lib \
        "semgrep==1.179.0" > "$IC_STATE/semgrep_install.log" 2>&1 \
    && printf '#!/bin/sh\nPYTHONPATH=/root/semgrep-lib exec %s /root/semgrep-lib/bin/semgrep "$@"\n' "$PY" > /usr/local/bin/semgrep \
    && chmod +x /usr/local/bin/semgrep && echo ok > "$IC_STATE/semgrep_ok" ) &
  SEMGREP_PID=$!
fi

# ---------------------------------------------------------------- 3. versions + CUDA
set +e
"$PY" - "$CFG" "$IC_STATE/env_check.json" <<'PY'
import importlib.metadata as md, json, sys
cfg = json.load(open(sys.argv[1]))
want = {"transformers": cfg["pins"]["transformers"], "accelerate": cfg["pins"]["accelerate"],
        "bandit": cfg["pins"]["bandit"], "tokenizers": "0.22.2", "huggingface-hub": "1.28.0",
        "safetensors": "0.8.0", "numpy": "2.5.2", "scipy": "1.18.1", "torch": None}
got, bad = {}, {}
for k, v in want.items():
    try:
        got[k] = md.version(k)
    except md.PackageNotFoundError:
        got[k] = None
    if v and got[k] != v:
        bad[k] = {"want": v, "got": got[k]}
rep = {"versions": got, "mismatches": bad}
try:
    import torch
    rep["torch_pin_ok"] = torch.__version__.split("+")[0] == cfg["pins"]["torch"].split("+")[0]
    rep["cuda"] = torch.cuda.is_available()
    if rep["cuda"]:
        rep["device"] = torch.cuda.get_device_name(0)
        rep["capability"] = list(torch.cuda.get_device_capability(0))
        rep["bf16"] = torch.cuda.is_bf16_supported()
        rep["mem_total_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
except Exception as exc:  # noqa: BLE001
    rep["torch_error"] = repr(exc)
print(json.dumps(rep, indent=1))
json.dump(rep, open(sys.argv[2], "w"), indent=1)
fatal = ("transformers" in bad) or not rep.get("cuda") or not rep.get("torch_pin_ok", False)
sys.exit(1 if fatal else (3 if bad else 0))
PY
RC=$?
set -e
[[ $RC -eq 1 ]] && die 1 "env check failed (transformers pin, torch pin or CUDA): see $IC_STATE/env_check.json"
[[ $RC -eq 3 ]] && say "WARNING: non-critical version mismatches (see env_check.json)"
[[ $RC -ne 0 && $RC -ne 3 ]] && die 1 "env check crashed (rc=$RC)"
stamp env_check
bstat setup "deps ok; downloading U (65.5 GB), lens, T"

# ---------------------------------------------------------------- 3b. labeler sanity (moved BEFORE the 74 GB download: fail fast, cheap) (bandit on a 2-line snippet; never runs model code)
BTMP="$(mktemp -d)"
printf 'import os\nos.chmod("/tmp/x", 0o777)\n' > "$BTMP/a.py"
SKIP="$("$PY" -c "import json,sys; print(','.join(json.load(open(sys.argv[1]))['labeler']['skip_tests']))" "$CFG")"
BOUT="$("$PY" -m bandit -f json -q -ll -s "$SKIP" "$BTMP/a.py" 2>/dev/null || true)"
echo "$BOUT" | "$PY" -c "import json,sys; r=json.load(sys.stdin)['results']; ids=sorted({x['test_id'] for x in r}); print('bandit sanity test_ids:', ids); sys.exit(0 if ids else 1)" \
  || die 1 "bandit sanity failed (expected a MEDIUM+ finding on os.chmod 0o777)"
rm -rf "$BTMP"
say "bandit $("$PY" -m bandit --version 2>&1 | head -1)"
stamp bandit

# ---------------------------------------------------------------- 3c. import check (also before the download) + prereg hash (logged, never fatal)
export PYTHONPATH="$EM/src:$CODE_ROOT/flagship/src:$CODE_ROOT/flagship/third_party${PYTHONPATH:+:$PYTHONPATH}"
"$PY" -c "import ic, ic.config, ic.data, ic.lens, ic.generate, ic.label, ic.monitor, ic.analyze, ic.status, wor.modeling, wor.scorer, wor.ablate, jlens; print('imports ok')" \
  || die 1 "ic/wor/jlens import failed on the pod"
"$PY" -c "from ic import config; print('prereg hash ok:', config.check_prereg_hash())" || say "WARNING: prereg hash check crashed (logged only)"
if [[ "${IC_BOOT_TESTS:-0}" == "1" ]]; then
  ( cd "$EM" && "$PY" -m pytest -q -x tests ) || say "WARNING: unit tests failed on the pod (see above)"
fi
stamp imports

if [[ -n "$SEMGREP_PID" ]]; then
  wait "$SEMGREP_PID" || true
  if [[ -f "$IC_STATE/semgrep_ok" ]] && semgrep --version >/dev/null 2>&1; then
    say "semgrep $(semgrep --version 2>/dev/null | head -1) (rules: bundled data/semgrep/python_cyberseceval.json)"
  else
    say "WARNING: semgrep not available (second opinion is descriptive only; see $IC_STATE/semgrep_install.log)"
  fi
fi

# ---------------------------------------------------------------- 4. HF downloads -> container disk ($HF_HOME)
[[ -n "${HF_TOKEN:-}" ]] && export HF_TOKEN
# Heartbeat during the ~74 GB download (pip + downloads are the first ~20 min of billed time): every 30 s
# counts.download = [GB under $HF_HOME, 74], plus an event every 5 min, so the laptop watcher's 20-min
# no-advance rule sees real progress (or a real stall). Killed after the downloads, and on any exit.
DL_TOTAL_GB="${IC_DL_TOTAL_GB:-74}"
dl_heartbeat() {
  local i=0 gb
  while true; do
    gb="$(du -sb "$HF_HOME" 2>/dev/null | awk '{printf "%d", $1 / 1e9}' || true)"
    bcount download "${gb:-0}" "$DL_TOTAL_GB" || true
    if (( i % 10 == 0 )); then bstat "-" "download ${gb:-0} / ~${DL_TOTAL_GB} GB on disk" || true; fi
    i=$(( i + 1 ))
    sleep 30
  done
}
dl_heartbeat > /dev/null 2>&1 &
HB_PID=$!
trap 'kill "$HB_PID" 2>/dev/null || true' EXIT
# One bound, not a retry: a stalled HF transfer must not burn $1.59/h unnoticed (rc 124 = the 45-min bound).
timeout "${IC_DL_TIMEOUT:-45m}" "$PY" - "$CFG" "$IC_STATE/downloads.json" "${IC_T_FALLBACK_REV:-2e1fd397ee46e1388853d2af2c993145b0f1098a}" <<'PY' || die 1 "downloads failed or timed out (rc=$?; 124 = the ${IC_DL_TIMEOUT:-45m} bound); see the log above"
import json, os, sys, time
from huggingface_hub import hf_hub_download, snapshot_download
cfg, out_path, fb_rev = json.load(open(sys.argv[1])), sys.argv[2], sys.argv[3]
out = {}
def _size(f):
    try:
        return os.path.getsize(f)
    except OSError:  # a dangling cache symlink must not fail the whole download step
        return 0
def nbytes(p):
    return sum(_size(os.path.join(dp, f)) for dp, _, fs in os.walk(p) for f in fs)
def snap(key, repo, rev, patterns):
    t = time.time()
    p = snapshot_download(repo, revision=rev, allow_patterns=patterns)
    out[key] = {"repo": repo, "revision": rev, "path": p, "bytes": nbytes(p), "s": round(time.time() - t, 1)}
    print(key, json.dumps(out[key]), flush=True)
    return p
U = cfg["models"]["U_untrusted"]
up = snap("U", U["repo"], U["revision"], ["*.json", "*.safetensors", "*.jinja", "tokenizer*", "merges.txt", "vocab.json"])
idx = json.load(open(os.path.join(up, "model.safetensors.index.json")))
shards = sorted(set(idx["weight_map"].values()))
miss = [s for s in shards if not os.path.isfile(os.path.join(up, s))]
got = sum(os.path.getsize(os.path.join(up, s)) for s in shards if s not in miss)
out["U"].update({"n_shards": len(shards), "missing_shards": miss, "shard_bytes": got,
                 "index_total_size": idx.get("metadata", {}).get("total_size")})
if miss or got < int(idx.get("metadata", {}).get("total_size", 0)):
    print("U download incomplete", miss, got, flush=True); sys.exit(1)
L = cfg["lens"]
t = time.time()
lp = hf_hub_download(L["repo"], L["filename"], revision=L["revision"])
out["lens"] = {"repo": L["repo"], "revision": L["revision"], "path": lp, "bytes": os.path.getsize(lp), "s": round(time.time() - t, 1)}
print("lens", json.dumps(out["lens"]), flush=True)
if os.path.getsize(lp) != int(L["size_bytes"]):
    print("lens size mismatch", os.path.getsize(lp), L["size_bytes"], flush=True); sys.exit(1)
T = cfg["models"]["T_trusted"]
pats = ["*.json", "*.safetensors", "*.jinja", "tokenizer*", "merges.txt", "vocab.json"]
snap("T", T["repo"], T["revision"], pats)
try:  # fallback T only matters if T fails to load; pre-fetched because HF_HUB_OFFLINE=1 later
    snap("T_fallback", T["fallback_repo"], fb_rev, pats)
    # monitor.load_T asks for the fallback WITHOUT a revision ("main"); a download by commit sha writes no
    # refs/main, so offline "main" would not resolve. Point refs/main at the pinned snapshot.
    from huggingface_hub import constants
    ref = os.path.join(constants.HF_HUB_CACHE, "models--" + T["fallback_repo"].replace("/", "--"), "refs", "main")
    if not os.path.exists(ref):
        os.makedirs(os.path.dirname(ref), exist_ok=True)
        with open(ref, "w") as fh:
            fh.write(fb_rev)
    out["T_fallback"]["refs_main"] = open(ref).read().strip()
except Exception as exc:  # noqa: BLE001
    out["T_fallback"] = {"error": repr(exc)[:300]}
    print("WARNING T fallback download failed", exc, flush=True)
json.dump(out, open(out_path, "w"), indent=1)
PY
kill "$HB_PID" 2>/dev/null || true
trap - EXIT
bcount download "$(du -sb "$HF_HOME" 2>/dev/null | awk '{printf "%d", $1 / 1e9}' || echo 0)" "$DL_TOTAL_GB" || true
stamp downloads
jget() { "$PY" -c "import json,sys; d=json.load(open(sys.argv[1])); print(d.get(sys.argv[2], {}).get('path', ''))" "$IC_STATE/downloads.json" "$1"; }
export IC_U_PATH="$(jget U)" IC_T_PATH="$(jget T)" IC_T_FALLBACK_PATH="$(jget T_fallback)" IC_LENS_PATH="$(jget lens)"
df -h /root | tail -1
bstat setup "downloads ok (U, lens, T); checking labeler + imports"

# ---------------------------------------------------------------- 7. env file for later shells (no secrets)
cat > "$IC_ROOT/env.sh" <<EOF
# written by bootstrap.sh $(date -u +%Y-%m-%dT%H:%M:%SZ); no secrets. source before run_all.sh / spend_guard.py
export IC_ROOT="$IC_ROOT" IC_STATE="$IC_STATE" IC_RESULTS="$IC_RESULTS" IC_LOGS="$IC_LOGS" IC_DONE_DIR="$IC_DONE_DIR"
export IC_STATUS_PATH="$IC_STATUS_PATH" IC_STATUS="$IC_STATUS_PATH"
export IC_CODE_ROOT="$CODE_ROOT" IC_EM_DIR="$EM"
export PYTHONPATH="$EM/src:$CODE_ROOT/flagship/src:$CODE_ROOT/flagship/third_party"
export HF_HOME="$HF_HOME" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export IC_U_PATH="$IC_U_PATH" IC_T_PATH="$IC_T_PATH" IC_T_FALLBACK_PATH="$IC_T_FALLBACK_PATH" IC_LENS_PATH="$IC_LENS_PATH"
export IC_HOURLY_USD="$IC_HOURLY_USD" IC_SPENT_BEFORE_USD="$IC_SPENT_BEFORE_USD" IC_POD_START_EPOCH="$IC_POD_START_EPOCH"
export IC_POD_STARTED_AT="$IC_POD_START_EPOCH" IC_GPU="$IC_GPU_NAME" IC_SPEND_GUARD="$HERE/spend_guard.py"
export IC_CAP_USD="$IC_CAP_USD" IC_SOFT_CAP_USD="$IC_SOFT_CAP_USD" IC_HARD_CAP_USD="$IC_HARD_CAP_USD"
export IC_GUARD_MARGIN_USD="$IC_GUARD_MARGIN_USD" IC_GUARD_ACTION="$IC_GUARD_ACTION" IC_GPU_NAME="$IC_GPU_NAME"
export PY="$PY" PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
EOF
say "wrote $IC_ROOT/env.sh"

if [[ "${IC_BOOT_SANITY:-0}" == "1" ]]; then
  # shellcheck disable=SC1091
  ( source "$IC_ROOT/env.sh" && cd "$EM" && "$PY" -m ic.lens --sanity ) || say "WARNING: G1 sanity crashed (logged; run_all.sh decides)"
fi

date -u +%Y-%m-%dT%H:%M:%SZ > "$IC_ROOT/BOOTSTRAP_DONE"
stamp bootstrap
bstat setup "bootstrap done in $(( $(date +%s) - T0 )) s"
if [[ "${IC_FROM_LAUNCHER:-0}" != "1" ]]; then
  say "NEXT (orchestrator; pod/launch.sh does all three by itself, detached):"
  say "  source $IC_ROOT/env.sh && $PY $HERE/spend_guard.py stop-now --dry-run      # must print the intended terminate"
  say "  source $IC_ROOT/env.sh && (setsid nohup $PY $HERE/spend_guard.py watchdog --interval 30 >> $IC_STATE/watchdog.out 2>&1 < /dev/null &)"
  say "  (setsid nohup bash $HERE/run_all.sh > $IC_ROOT/run_all.log 2>&1 < /dev/null &)"
fi
