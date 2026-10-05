"""pod/spend_guard.py on the laptop: fake clocks, mocked terminate calls, --dry-run. No network, no pod, $0.

    python -m pytest -q interpcontrol/em/tests/test_spend_guard.py
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

EM = Path(__file__).resolve().parents[1]
GUARD = EM / "pod" / "spend_guard.py"


def _load():
    spec = importlib.util.spec_from_file_location("ic_spend_guard_test", GUARD)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def env(tmp_path, monkeypatch):
    for k in list(os.environ):
        if k.startswith(("IC_", "RUNPOD_")):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("IC_ROOT", str(tmp_path / "ic"))
    monkeypatch.setenv("IC_STATE", str(tmp_path / "ic" / "state"))
    monkeypatch.setenv("IC_RESULTS", str(tmp_path / "ic" / "results"))
    monkeypatch.setenv("IC_STATUS_PATH", str(tmp_path / "ic" / "status.json"))
    monkeypatch.setenv("IC_HOURLY_USD", "1.59")
    monkeypatch.setenv("IC_SPENT_BEFORE_USD", "0")
    monkeypatch.setenv("IC_POD_START_EPOCH", "1000000")
    (tmp_path / "ic" / "results").mkdir(parents=True)
    (tmp_path / "ic" / "results" / "gen_round0.jsonl").write_text('{"id": "a"}\n')
    return tmp_path


def test_state_math_and_defaults(env):
    sg = _load()
    s = sg.state(now=1_000_000 + 3600 * 2)
    assert s["session_hours"] == pytest.approx(2.0) and s["cumulative_usd"] == pytest.approx(3.18)
    assert s["soft_cap_usd"] == 13.5 and s["hard_cap_usd"] == 14.75 and s["margin_usd"] == 0.25
    assert s["may_start_new_step"] and not s["hard_cap_reached"]
    assert s["spent_before_scope"] == "ic-em pods only"


def test_spent_before_is_required(env, monkeypatch):
    sg = _load()
    monkeypatch.delenv("IC_SPENT_BEFORE_USD")
    with pytest.raises(SystemExit, match="IC_SPENT_BEFORE_USD"):
        sg.state(now=1_000_000)


def test_check_step_soft_hard_projection(env, monkeypatch):
    sg = _load()
    t0 = 1_000_000
    ok, s = sg.may_start_step(1.0, now=t0 + 3600)          # $1.59 + 1.0 < 14.5
    assert ok and s["decision"] == "start"
    ok, s = sg.may_start_step(13.0, now=t0 + 3600)         # 1.59 + 13 >= 14.5 -> blocked
    assert not ok and s["decision"] == "blocked"
    ok, _ = sg.may_start_step(0.0, now=t0 + 3600 * 8.6)    # 13.67 >= soft 13.5
    assert not ok
    monkeypatch.setenv("IC_SPENT_BEFORE_USD", "12.0")      # earlier ic-em pods only
    ok, _ = sg.may_start_step(0.0, now=t0 + 3600)          # 13.59 >= 13.5
    assert not ok
    monkeypatch.setenv("IC_SPENT_BEFORE_USD", "0")
    (Path(os.environ["IC_STATE"])).mkdir(parents=True, exist_ok=True)
    (Path(os.environ["IC_STATE"]) / "HARD_CAP_REACHED").write_text("{}")
    ok, s = sg.may_start_step(0.0, now=t0 + 60)
    assert not ok and s.get("hard_cap_flag")


def test_watchdog_fake_clock_packs_then_terminates(env, monkeypatch):
    sg = _load()
    clock = {"t": 1_000_000.0}
    sleeps: list[float] = []

    def now():
        return clock["t"]

    def sleep(dt):
        sleeps.append(dt)
        clock["t"] += 1800.0  # each loop advances 30 min of billing

    calls = []

    def fake_terminate(dry_run=False):
        calls.append(dry_run)
        return True

    rc = sg.watchdog(30.0, dry_run=False, now_fn=now, sleep=sleep, terminate=fake_terminate)
    assert rc == sg.EXIT_HARD_CAP and calls == [False]
    st = Path(os.environ["IC_STATE"])
    flag = json.loads((st / "HARD_CAP_REACHED").read_text())
    assert flag["cumulative_usd"] >= 14.5 and (st / "SOFT_CAP_REACHED").exists()
    packs = list((Path(os.environ["IC_ROOT"]) / "outbox").glob("guard_pack_*.tgz"))
    assert len(packs) == 1
    import tarfile
    with tarfile.open(packs[0]) as tar:
        assert "results/gen_round0.jsonl" in tar.getnames() and "PACK_SKIPPED.json" in tar.getnames()
    assert 20 in sleeps  # the flush pause before packing
    log = [json.loads(x) for x in (st / "spend_guard.log.jsonl").read_text().splitlines()]
    assert [r["event"] for r in log if r["event"] in ("soft_cap_reached", "hard_cap_reached", "final_pack")] == \
        ["soft_cap_reached", "hard_cap_reached", "final_pack"]


def test_watchdog_retries_failed_terminate(env, monkeypatch):
    sg = _load()
    monkeypatch.setenv("IC_SPENT_BEFORE_USD", "20")       # already over the hard cap
    monkeypatch.setenv("IC_GUARD_RETRIES", "2")
    monkeypatch.setenv("IC_GUARD_RETRY_S", "1")
    n = {"k": 0}

    def fails(dry_run=False):
        n["k"] += 1
        return False

    rc = sg.watchdog(30.0, dry_run=False, now_fn=lambda: 1_000_000.0, sleep=lambda dt: None, terminate=fails)
    assert rc == sg.EXIT_HARD_CAP and n["k"] == 3


def test_terminate_rest_first_with_user_agent_then_cli(env, monkeypatch):
    sg = _load()
    monkeypatch.setenv("RUNPOD_POD_ID", "pod123")
    monkeypatch.setenv("RUNPOD_API_KEY", "k")
    monkeypatch.setattr(sg.shutil, "which", lambda name: "/usr/bin/runpodctl" if name == "runpodctl" else None)
    plan = sg.terminate_plan()
    assert plan[0]["method"] == "DELETE" and plan[0]["url"].endswith("/v1/pods/pod123")
    assert plan[0]["headers"]["User-Agent"] == sg.USER_AGENT
    assert [p.get("cmd") for p in plan[1:]] == [["runpodctl", "remove", "pod", "pod123"],
                                                ["runpodctl", "pod", "delete", "pod123"]]
    for p in plan:  # terminate only: never a stop
        assert "stop" not in json.dumps(p).lower()
    ok = sg.terminate_pod(rest=lambda pod, key: 200, run=lambda *a, **k: pytest.fail("cli must not run"))
    assert ok

    class R:
        returncode, stdout, stderr = 0, "removed", ""

    def boom(pod, key):
        raise OSError("403")

    seen = []
    ok = sg.terminate_pod(rest=boom, run=lambda cmd, **k: (seen.append(cmd), R())[1])
    assert ok and seen == [["runpodctl", "remove", "pod", "pod123"]]


def test_rest_delete_builds_request_with_user_agent(env, monkeypatch):
    sg = _load()
    seen = {}

    class Resp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=0):
        seen["method"], seen["url"] = req.get_method(), req.full_url
        seen["ua"] = req.get_header("User-agent")
        seen["auth"] = req.get_header("Authorization")
        return Resp()

    monkeypatch.setattr(sg.urllib.request, "urlopen", fake_urlopen)
    assert sg._rest_delete("podX", "secret-key") == 200
    assert seen == {"method": "DELETE", "url": "https://rest.runpod.io/v1/pods/podX", "ua": sg.USER_AGENT,
                    "auth": "Bearer secret-key"}


def test_non_terminate_action_is_ignored(env, monkeypatch, capsys):
    sg = _load()
    monkeypatch.setenv("IC_GUARD_ACTION", "stop")
    assert sg.guard_action() == "terminate"
    assert "guard_action_ignored" in capsys.readouterr().out


def test_source_never_stops_a_pod():
    src = GUARD.read_text(encoding="utf-8")
    code = "\n".join(ln for ln in src.splitlines() if not ln.lstrip().startswith("#"))
    body = code.split('"""', 2)[2]  # skip the module docstring (it explains why /stop is never used)
    assert "/stop" not in body and '"stop"' not in body and "'stop'" not in body


def test_cli_stop_now_dry_run_prints_delete(env, tmp_path):
    e = {k: v for k, v in os.environ.items()}
    e.update({"RUNPOD_POD_ID": "podDRY", "RUNPOD_API_KEY": "not-a-real-key", "PATH": e.get("PATH", "")})
    r = subprocess.run([sys.executable, str(GUARD), "stop-now", "--dry-run"], capture_output=True, text=True,
                       env=e, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "dry_run_would_terminate" in r.stdout and '"DELETE"' in r.stdout
    assert "https://rest.runpod.io/v1/pods/podDRY" in r.stdout and "not-a-real-key" not in r.stdout
    assert "final_pack_dry_run" in r.stdout
    assert not (Path(e["IC_ROOT"]) / "outbox").exists()  # dry run writes no pack
    r2 = subprocess.run([sys.executable, str(GUARD), "check-step", "--projected-usd", "99"], capture_output=True,
                        text=True, env=e, timeout=60)
    assert r2.returncode == 3 and '"blocked"' in r2.stdout
    r3 = subprocess.run([sys.executable, str(GUARD), "status"], capture_output=True, text=True, env=e, timeout=60)
    assert r3.returncode == 0 and json.loads(r3.stdout)["spent_before_scope"] == "ic-em pods only"
