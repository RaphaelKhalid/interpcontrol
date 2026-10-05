"""OLED live monitor for interpcontrol: H-lens pilot (laptop) + J-lens control experiment (pod) + budget.

    python interpcontrol/dash/server.py 8796

Reads only local files:
  interpcontrol/hlens/runlog.md, hlens/out/summary_*.md   (H-lens pilot)
  interpcontrol/runs/status.json                          (control pipeline writes this; see SCHEMA below)
SCHEMA of runs/status.json (all optional):
  {"phase": "pilot-gen", "phases": ["setup", ...], "counts": {"honest": [done, total], "attack": [done, total], ...},
   "attempts": int, "validated": int, "auroc": {"BB": x, "WB0": y}, "money_shot": {"rescued": a, "lost": b, "n": n},
   "pod": {"id": str, "gpu": str, "rate_usd_h": float, "started_at": unix, "terminated_at": unix|null},
   "spent_before_usd": float, "cap_usd": 15, "trial_cap_usd": 3, "events": ["[hh:mm] text", ...], "updated": unix}
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
IC = HERE.parent
HL = IC / "hlens"
STATUS = IC / "runs" / "status.json"
STATE: dict = {"updated": 0}


def _table_rows(md: str, header_start: str) -> list[list[str]]:
    lines = md.splitlines()
    for i, ln in enumerate(lines):
        if ln.startswith(header_start):
            rows = []
            for r in lines[i + 2:]:
                if not r.startswith("|"):
                    break
                rows.append([c.strip() for c in r.strip().strip("|").split("|")])
            return rows
    return []


def hlens() -> dict:
    d: dict = {}
    log = HL / "runlog.md"
    if log.exists():
        txt = log.read_text(encoding="utf-8", errors="replace").splitlines()
        d["log"] = [x for x in txt if x.strip()][-14:]
        d["age"] = round(time.time() - log.stat().st_mtime)
        l1 = []
        for x in txt:
            m = re.search(r"\[L1\] '(.+?)'.*ov10 ([\d.]+) cosF ([\d.]+) \|L1\|\^2/\|L0\|\^2 ([\d.]+)", x)
            if m:
                l1.append({"tok": m[1], "ov10": float(m[2]), "cosF": float(m[3]), "energy": float(m[4])})
        d["l1"] = l1
    lad = HL / "out" / "summary_ladder_08b.md"
    if lad.exists():
        rows = _table_rows(lad.read_text(encoding="utf-8"), "| rung | mean ov top-5")
        d["ladder"] = [{"rung": r[0], "ov10": r[2], "cosF": r[4], "explained": r[5]} for r in rows if len(r) >= 6]
    sw = HL / "out" / "summary_swaps_08b_big.md"
    if sw.exists():
        rows = _table_rows(sw.read_text(encoding="utf-8"), "| subset | layer | alpha")
        # columns: subset, layer, alpha, n, true dA, |e1|, |e2|, agg share, median share, corr x3, |D|/|h|, flips
        d["t1"] = [{"layer": r[1], "alpha": r[2], "rel": r[12], "share": r[7], "flips": r[13]}
                   for r in rows if len(r) >= 14 and r[0] == "kept" and r[2] in ("1.0", "2.0")]
    res = HL / "RESULTS.md"
    d["results_ready"] = res.exists()
    d["files"] = len(list((HL / "out").glob("*"))) if (HL / "out").exists() else 0
    return d


def gpu() -> dict:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout.strip().split(",")
        util, mem, total = int(out[0]), int(out[1]), int(out[2])
        # the laptop driver sometimes reports a garbage (overflowed) memory.used; hide it rather than show nonsense
        return {"util": util, "mem": mem if 0 <= mem <= total else None, "total": total}
    except Exception:  # noqa: BLE001
        return {}


def control() -> dict:
    if not STATUS.exists():
        return {"exists": False}
    try:
        s = json.loads(STATUS.read_text(encoding="utf-8"))
    except ValueError:
        return {"exists": True, "error": "status.json unreadable (mid-write?)"}
    s["exists"] = True
    s["age"] = round(time.time() - STATUS.stat().st_mtime)
    return s


def loop() -> None:
    while True:
        try:
            STATE["hlens"] = hlens()
            STATE["gpu"] = gpu()
            STATE["control"] = control()
            STATE["error"] = None
        except Exception as e:  # noqa: BLE001
            STATE["error"] = f"{type(e).__name__}: {e}"[:200]
        STATE["updated"] = time.time()
        time.sleep(3)


class H(BaseHTTPRequestHandler):
    def log_message(self, *a) -> None:
        pass

    def do_GET(self) -> None:
        if self.path.startswith("/api/state"):
            body = json.dumps({**STATE, "now": time.time()}).encode()
            ctype = "application/json"
        else:
            body = (HERE / "index.html").read_bytes()
            ctype = "text/html; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8796
    threading.Thread(target=loop, daemon=True).start()
    print(f"interpcontrol OLED dash on http://127.0.0.1:{port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
