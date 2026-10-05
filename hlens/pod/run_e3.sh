#!/usr/bin/env bash
# E3 on Qwen3.5-4B (32 layers): (1) can 4B follow the in-context lie-iff-AND rule? (behaviour only)
# (2) LoRA-plant the same pirate x animal organism at 4B; (3) cross-position H-lens test (gate_test2.py).
set -uo pipefail
cd /workspace/mats/interpcontrol/hlens
M=Qwen/Qwen3.5-4B
python gate_behavior.py $M chat d4 > out/beh4b_chat.stdout 2>&1
python gate_behavior.py $M d4 > out/beh4b_raw.stdout 2>&1
export HL_BASE=$M HL_MODEL_DIR=/workspace/models/hlens-pirate-4b
python organism.py train > out/organism4b.stdout 2>&1
python gate_test2.py /workspace/models/hlens-pirate-4b 6 8 10 12 14 16 > out/gate2_4b.stdout 2>&1
echo "[e3 $(date +%T)] ALL DONE" | tee -a runlog.md
