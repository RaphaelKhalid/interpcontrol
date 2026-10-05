#!/usr/bin/env bash
# E3 pod setup (RunPod PyTorch 2.8 image). Run from /workspace after unpacking the bundle into /workspace/mats.
set -euo pipefail
cd /workspace/mats/interpcontrol/hlens
mkdir -p out
# hard spend cap: TERMINATE this pod after HL_MAX_HOURS (default 3 h) no matter what
# (terminate, not stop: a stopped pod keeps billing its disk). Copy results off before then.
( sleep $(( ${HL_MAX_HOURS:-3} * 3600 )); echo "[watchdog] max hours reached, terminating pod" >> out/pod_watchdog.log;
  runpodctl remove pod "$RUNPOD_POD_ID" >> out/pod_watchdog.log 2>&1 || runpodctl stop pod "$RUNPOD_POD_ID" >> out/pod_watchdog.log 2>&1 ) > /dev/null 2>&1 &
echo "[bootstrap $(date +%T)] watchdog armed (${HL_MAX_HOURS:-3} h)" | tee -a runlog.md
pip install -q "transformers==5.5.4" "peft==0.20.0" scipy pyarrow accelerate 2>&1 | tail -2
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("Salesforce/wikitext", repo_type="dataset", allow_patterns=["wikitext-103-raw-v1/validation-*"])
snapshot_download("Qwen/Qwen3.5-4B")
import torch, transformers; print("torch", torch.__version__, "transformers", transformers.__version__, torch.cuda.get_device_name(0))
PY
echo "[bootstrap $(date +%T)] done" | tee -a runlog.md
