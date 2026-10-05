#!/usr/bin/env bash
# Pack the H-lens v2 code for the E3 pod (Qwen3.5-4B). Nothing is uploaded; scp it yourself.
#   bash interpcontrol/hlens/pod/make_bundle.sh [OUT]     default $TEMP/hl_bundle.tar.gz
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
OUT="${1:-${TEMP:-/tmp}/hl_bundle.tar.gz}"
command -v cygpath >/dev/null && OUT="$(cygpath -u "$OUT")"
cd "$ROOT"
tar -czf "$OUT" --exclude='__pycache__' --exclude='*.pyc' \
  interpcontrol/hlens/common.py interpcontrol/hlens/hvp.py interpcontrol/hlens/hl2.py \
  interpcontrol/hlens/gate_prompts.py interpcontrol/hlens/gate_behavior.py interpcontrol/hlens/organism.py \
  interpcontrol/hlens/gate_test.py interpcontrol/hlens/inject.py interpcontrol/hlens/tests interpcontrol/hlens/pod \
  flagship/src/wor flagship/third_party/jlens
echo "wrote $OUT ($(du -h "$OUT" | cut -f1))"
