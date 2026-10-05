"""Shared pytest setup for interpcontrol/em (runs before any test module imports torch).

HARD RULE (SPEC header): unit tests run on CPU only; the laptop GPU belongs to another session.
``CUDA_VISIBLE_DEVICES=""`` does NOT hide the GPU on Windows (assigning an empty value deletes the
variable), so "-1" is used: CUDA then enumerates no device on every OS.
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("HF_HUB_OFFLINE", "1")          # cached files only: tests never touch the network
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# bandit tests: fall back to the isolated %TEMP%\icb venv (REVIEW-LOG M5 recipe) when bandit is not importable
# here and IC_BANDIT_PYTHON is unset; the same rule as smoke._default_bandit_python.
if not os.environ.get("IC_BANDIT_PYTHON"):
    import importlib.util
    import shutil
    import tempfile

    _icb = os.path.join(tempfile.gettempdir(), "icb", *(("Scripts", "python.exe") if os.name == "nt" else ("bin", "python")))
    if importlib.util.find_spec("bandit") is None and shutil.which("bandit") is None and os.path.exists(_icb):
        os.environ["IC_BANDIT_PYTHON"] = _icb
