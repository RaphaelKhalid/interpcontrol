"""Fast laptop tests for the pod infrastructure: pod/run_all.sh, pod/bootstrap.sh, pod/make_bundle.py,
tools/sync_status.py (CPU only, no network, no GPU, no pod, $0).

    python -m pytest -q interpcontrol/em/tests/test_pod_infra.py

run_all.sh is exercised end-to-end under Git bash against STUB ic.* modules and a STUB spend guard
(so the real 32B stages never run); bootstrap.sh gets `bash -n` plus consistency checks (it needs a real
pod: pip, nvidia-smi, 65 GB downloads).
"""
from __future__ import annotations

import gzip
import hashlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import textwrap
import time
from pathlib import Path

import pytest

EM = Path(__file__).resolve().parents[1]
POD = EM / "pod"
TOOLS = EM / "tools"
IC_DIR = EM.parent


def _find_bash() -> str | None:
    for p in (r"C:\Program Files\Git\bin\bash.exe", r"C:\Program Files\Git\usr\bin\bash.exe"):
        if os.path.exists(p):
            return p
    b = shutil.which("bash")
    if b and "system32" in b.lower():  # WSL's bash.exe would run in a different filesystem
        return None
    return b


BASH = _find_bash()
needs_bash = pytest.mark.skipif(BASH is None, reason="no POSIX bash (Git bash) found")


def posix(p: Path | str) -> str:
    """C:\\x\\y -> /c/x/y (Git bash form; MSYS converts it back for native programs)."""
    s = str(p).replace("\\", "/")
    if os.name == "nt" and len(s) > 1 and s[1] == ":":
        return "/" + s[0].lower() + s[2:]
    return s


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


sync = _load("ic_sync_status", TOOLS / "sync_status.py")
mb = _load("ic_make_bundle", POD / "make_bundle.py")


# =============================================================================== shell syntax
@needs_bash
@pytest.mark.parametrize("script", ["run_all.sh", "bootstrap.sh", "launch.sh"])
def test_shell_syntax_and_lf(script):
    p = POD / script
    assert b"\r" not in p.read_bytes(), f"{script} has CR characters (breaks bash on Linux)"
    r = subprocess.run([BASH, "-n", posix(p)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_bootstrap_required_files_are_bundled():
    """Every file bootstrap.sh insists on is in make_bundle's REQUIRED list (so a bundle that passes
    make_bundle can never fail bootstrap's completeness check)."""
    txt = (POD / "bootstrap.sh").read_text(encoding="utf-8")
    block = txt.split("for need in", 1)[1].split("; do", 1)[0]
    need = [t for t in block.replace("\\", " ").split() if "/" in t]
    assert len(need) >= 20
    missing = sorted(set(need) - set(mb.REQUIRED))
    assert not missing, missing


def test_requirements_pins():
    req = (POD / "requirements.txt").read_text(encoding="utf-8")
    cfg = json.loads((EM / "prereg" / "config.json").read_text(encoding="utf-8"))
    lines = [ln.split("#")[0].strip() for ln in req.splitlines()]
    lines = [ln for ln in lines if ln]
    assert f"transformers=={cfg['pins']['transformers']}" in lines
    assert f"accelerate=={cfg['pins']['accelerate']}" in lines
    assert f"bandit=={cfg['pins']['bandit']}" in lines
    assert "torch==2.8.0" in lines            # satisfied by the image's 2.8.0+cu128; never reinstalls
    assert not any(ln.startswith(("vllm", "flash-attn", "semgrep")) for ln in lines)
    assert all("==" in ln for ln in lines), "every requirement must be pinned"


# =============================================================================== run_all.sh (stubs)
STUB_COMMON = '''
import json, os, sys, time, pathlib
def call(mod):
    with open(os.environ["STUB_CALLS"], "a", encoding="utf-8") as f:
        f.write(mod + " " + " ".join(sys.argv[1:]) + "\\n")
def fails(tag):
    return tag in os.environ.get("STUB_FAIL", "").split(",")
RD = pathlib.Path(os.environ["IC_RESULTS"]); DD = pathlib.Path(os.environ["IC_DONE_DIR"])
'''

STUB_GENERATE = STUB_COMMON + '''
call("generate")
a = sys.argv[1:]
assert a[0] == "u_all", a
stages = a[a.index("--stages") + 1].split(",")
for st in stages:
    if (DD / st).exists():
        continue
    if st == "sanity":  # the real ic.generate.run_u_all catches a sanity crash: logged, no marker, go on
        if fails("sanity"):
            (RD / "sanity_G1.json").write_text(json.dumps({"gate": "G1", "status": "error"}))
            continue
        g1 = os.environ.get("STUB_G1", "pass")
        (RD / "sanity_G1.json").write_text(json.dumps({"gate": "G1", "status": g1,
                                                       "top1_agreement": 0.3 if g1 == "gross_fail" else 0.9}))
    if fails(st):
        sys.exit(1)
    if st in ("gen_r1", "read_r1") and json.loads((RD / "sanity_G1.json").read_text()).get("status") == "gross_fail":
        continue
    if st == "gen_r0" and not fails("gen_r0_empty"):
        with open(RD / "gen_round0.jsonl", "a", encoding="utf-8") as f:
            for i in range(3):
                f.write(json.dumps({"id": f"p{i}", "sample_idx": 0}) + "\\n")
        (RD / "stage_timings.json").write_text(json.dumps({"gen_r0": {"elapsed_s": 30.0, "new": 3}}))
    if st == "readout":
        t = json.loads((RD / "stage_timings.json").read_text())
        t["readout_r0"] = {"elapsed_s": 0.3, "new": 3}
        (RD / "stage_timings.json").write_text(json.dumps(t))
    if st == "gen_r1":
        (RD / "gen_round1.jsonl").write_text(json.dumps({"id": "p0", "sample_idx": 0, "condition": "no_hook"}) + "\\n")
    (DD / st).touch()
'''

STUB_LABEL = STUB_COMMON + '''
call("label")
a = sys.argv[1:]
if "--follow" in a:
    marker = a[a.index("--done-marker") + 1]
    t0 = time.time()
    while not os.path.exists(marker) and time.time() - t0 < 60:
        time.sleep(0.2)
    (RD / "follow_saw_marker").write_text(str(os.path.exists(marker)))
if fails("label_" + a[0]):
    sys.exit(1)
(RD / f"{a[0]}_labeled.jsonl").write_text("{}\\n")
'''

STUB_MONITOR = STUB_COMMON + '''
call("monitor")
if fails("bb"):
    sys.exit(1)
with open(RD / "bb_scores.jsonl", "a") as f:
    f.write("{}\\n")
'''

STUB_ANALYZE = STUB_COMMON + '''
call("analyze")
if fails("analyze"):
    sys.exit(1)
(RD / "RESULT.md").write_text("# stub result\\n")
'''

STUB_STATUS = STUB_COMMON + '''
call("status")
p = os.environ["IC_STATUS_PATH"]
old = {}
try:
    old = json.load(open(p, encoding="utf-8"))
except Exception:
    pass
doc = {"phase": "setup", "phases": ["setup"], "counts": {"gen_r0": [0, 1804]}, "attempts": 0, "validated": 0,
       "auroc": {}, "money_shot": None, "pod": old.get("pod") or {}, "events": (old.get("events") or []) + ["[00:00] init"],
       "updated": time.time()}
json.dump(doc, open(p, "w", encoding="utf-8"))
'''

STUB_GUARD = '''
import os, sys
with open(os.environ["STUB_CALLS"], "a", encoding="utf-8") as f:
    f.write("guard " + " ".join(sys.argv[1:]) + "\\n")
cmd = sys.argv[1]
if cmd == "status":
    print("{}"); sys.exit(0)
if cmd == "check-step":
    proj = float(sys.argv[sys.argv.index("--projected-usd") + 1])
    sys.exit(3 if proj > float(os.environ.get("STUB_GUARD_MAX", "1e9")) else 0)
sys.exit(0)
'''


def _make_tree(tmp: Path) -> tuple[Path, Path]:
    code = tmp / "code"
    em = code / "interpcontrol" / "em"
    (em / "pod").mkdir(parents=True)
    (em / "prereg").mkdir(parents=True)
    (em / "src" / "ic").mkdir(parents=True)
    shutil.copy(POD / "run_all.sh", em / "pod" / "run_all.sh")
    shutil.copy(EM / "prereg" / "config.json", em / "prereg" / "config.json")
    (em / "pod" / "spend_guard.py").write_text(STUB_GUARD, encoding="utf-8")
    ic = em / "src" / "ic"
    (ic / "__init__.py").write_text("", encoding="utf-8")
    for name, body in (("generate", STUB_GENERATE), ("label", STUB_LABEL), ("monitor", STUB_MONITOR),
                       ("analyze", STUB_ANALYZE), ("status", STUB_STATUS)):
        (ic / f"{name}.py").write_text(body, encoding="utf-8")
    root = tmp / "ic"
    root.mkdir()
    return em, root


def _markers(root: Path) -> Path:
    # Windows is case-insensitive: the SPEC's $IC_ROOT/done (markers) and $IC_ROOT/DONE (terminal file)
    # are the same path here, so the tests point IC_DONE_DIR elsewhere (on the Linux pod they coexist).
    return root / "markers"


def _run_all(em: Path, root: Path, calls: Path, **env_extra) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("IC_", "STUB_"))}
    env.update({"IC_ROOT": posix(root), "IC_DONE_DIR": posix(_markers(root)),
                "IC_SPENT_BEFORE_USD": "0", "IC_NO_AUTOWATCHDOG": "1",
                "PY": posix(sys.executable), "STUB_CALLS": str(calls), "IC_BG_LABEL_WAIT_S": "30",
                "IC_GPU_FREE_WAIT_S": "0", "PYTHONPATH": ""})
    env.update({k: str(v) for k, v in env_extra.items()})
    return subprocess.run([BASH, posix(em / "pod" / "run_all.sh")], capture_output=True, text=True,
                          encoding="utf-8", errors="replace", env=env, timeout=240)


def _assert_status_valid(doc: dict) -> None:
    """Dash schema check: ic.status.validate when the module exists, plus the dash's own reader."""
    st_path = EM / "src" / "ic" / "status.py"
    if st_path.exists():
        st = _load("ic_status_for_infra_test", st_path)
        assert st.validate(doc) == [], st.validate(doc)
    for k in ("phase", "phases", "events", "pod", "updated"):
        assert k in doc, k


def _calls(calls: Path) -> list[str]:
    return calls.read_text(encoding="utf-8").splitlines() if calls.exists() else []


def _stage_calls(calls: Path) -> list[str]:
    return [c for c in _calls(calls) if not c.startswith("guard")]


@needs_bash
def test_run_all_happy_path_and_resume(tmp_path):
    em, root = _make_tree(tmp_path)
    calls = tmp_path / "calls.txt"
    r = _run_all(em, root, calls)
    assert r.returncode == 0, r.stdout + r.stderr
    done = (root / "DONE").read_text(encoding="utf-8")
    assert done.split("\t")[0] == "ok", done + r.stdout
    seq = [c for c in _stage_calls(calls) if "--follow" not in c]
    assert seq == ["status init",
                   "generate u_all --stages sanity,gen_r0,readout",
                   "label all",
                   "monitor score",
                   "generate u_all --stages gen_r1,read_r1",
                   "label round1",
                   "monitor score",
                   "analyze all"], seq + [r.stdout]
    # the background labeler followed Round 0 and was released by the u_r0.exited marker
    assert any("label round0 --follow --done-marker" in c for c in _calls(calls)), _calls(calls)
    assert (root / "results" / "follow_saw_marker").read_text() == "True"
    # every stage was guard-checked with a projected cost
    assert sum(c.startswith("guard check-step --projected-usd") for c in _calls(calls)) == 8
    for m in ("setup", "u_r0", "label", "bb", "u_r1", "label_r1", "bb_r1", "analyze",
              "gen_r0", "readout", "sanity", "gen_r1", "read_r1"):
        assert (_markers(root) / m).exists(), m
    # status.json: dash schema, phase done, events from the runner
    st = json.loads((root / "status.json").read_text(encoding="utf-8"))
    _assert_status_valid(st)
    assert st["phase"] == "done"
    assert any("pipeline finished: ok" in e for e in st["events"])
    assert any("stage u_r0 ok" in e for e in st["events"])
    # outbox: parts reassemble to the tgz, both sha256 files verify, members carry a manifest
    out = root / "outbox"
    whole = b"".join((out / n).read_bytes() for n in sorted(os.listdir(out)) if n.startswith("ic_results.tgz.part"))
    assert whole == (out / "ic_results.tgz").read_bytes()
    h, name = (out / "ic_results.tgz.sha256").read_text().split()
    assert name == "ic_results.tgz" and h == hashlib.sha256(whole).hexdigest()
    for line in (out / "parts.sha256").read_text().splitlines():
        ph, pn = line.split()
        assert hashlib.sha256((out / pn).read_bytes()).hexdigest() == ph
    with tarfile.open(out / "ic_results.tgz") as tar:
        names = tar.getnames()
        man = tar.extractfile("MANIFEST.sha256").read().decode()
    assert "results/RESULT.md" in names and "status.json" in names and "logs/u_r0.log" in names
    assert "state/done/u_r0" in names and "DONE" not in names and "EXIT_STATUS" in names
    assert "results/RESULT.md" in man
    times = (root / "state" / "stage_times.tsv").read_text().splitlines()
    assert len(times) == 8 and all(t.split("\t")[1] == "0" for t in times)

    # resume: DONE present -> nothing runs
    n_before = len(_calls(calls))
    r2 = _run_all(em, root, calls)
    assert r2.returncode == 0 and "nothing to do" in r2.stdout
    assert len(_calls(calls)) == n_before
    # resume after deleting DONE: every stage skipped by its marker, no stage command re-runs
    n_stage = len(_stage_calls(calls))
    (root / "DONE").unlink()
    r3 = _run_all(em, root, calls)
    assert r3.returncode == 0, r3.stdout
    assert _stage_calls(calls)[n_stage:] == [], _stage_calls(calls)[n_stage:]
    assert (root / "DONE").read_text(encoding="utf-8").startswith("ok")
    assert r3.stdout.count("skip ") >= 8


@needs_bash
def test_run_all_round1_failure_is_partial(tmp_path):
    em, root = _make_tree(tmp_path)
    calls = tmp_path / "calls.txt"
    r = _run_all(em, root, calls, STUB_FAIL="gen_r1")
    assert r.returncode == 0, r.stdout
    done = (root / "DONE").read_text(encoding="utf-8")
    assert done.startswith("partial") and "Round 1" in done
    seq = _stage_calls(calls)
    assert "analyze all" in seq and "label round1" not in seq
    assert (root / "results" / "RESULT.md").exists()


@needs_bash
def test_run_all_guard_blocks_round1_budget(tmp_path):
    em, root = _make_tree(tmp_path)
    calls = tmp_path / "calls.txt"
    # stub timings (10 s per generation) make the R1 projection ~$7; every primary stage projects < $3
    r = _run_all(em, root, calls, STUB_GUARD_MAX="5.0")
    assert r.returncode == 0, r.stdout
    done = (root / "DONE").read_text(encoding="utf-8")
    assert done.startswith("budget"), done
    seq = _stage_calls(calls)
    assert "generate u_all --stages gen_r1,read_r1" not in seq
    assert seq[-1] == "analyze all"
    assert "u_r1" in (root / "state" / "blocked_stages").read_text()


@needs_bash
def test_run_all_primary_failure_without_output_stops_early(tmp_path):
    em, root = _make_tree(tmp_path)
    calls = tmp_path / "calls.txt"
    r = _run_all(em, root, calls, STUB_FAIL="gen_r0")
    assert r.returncode == 1
    done = (root / "DONE").read_text(encoding="utf-8")
    assert done.startswith("failed") and "no generations" in done
    seq = [c for c in _stage_calls(calls) if "--follow" not in c]
    assert seq == ["status init", "generate u_all --stages sanity,gen_r0,readout"], seq
    assert (root / "outbox" / "ic_results.tgz").exists()


@needs_bash
def test_run_all_sanity_crash_is_nonfatal(tmp_path):
    em, root = _make_tree(tmp_path)
    calls = tmp_path / "calls.txt"
    r = _run_all(em, root, calls, STUB_FAIL="sanity")
    assert r.returncode == 0, r.stdout
    done = (root / "DONE").read_text(encoding="utf-8")
    assert done.startswith("partial") and "sanity" in done
    seq = _stage_calls(calls)
    assert "generate u_all --stages gen_r1,read_r1" in seq and seq[-1] == "analyze all"


@needs_bash
def test_run_all_g1_gross_fail_skips_round1(tmp_path):
    em, root = _make_tree(tmp_path)
    calls = tmp_path / "calls.txt"
    r = _run_all(em, root, calls, STUB_G1="gross_fail")
    assert r.returncode == 0, r.stdout
    done = (root / "DONE").read_text(encoding="utf-8")
    assert done.startswith("partial") and "G1 gross failure" in done, done
    seq = _stage_calls(calls)
    assert "generate u_all --stages gen_r1,read_r1" not in seq and seq[-1] == "analyze all"
    st = json.loads((root / "status.json").read_text(encoding="utf-8"))
    assert any("Round 1 skipped: G1 gross failure" in e for e in st["events"])


@needs_bash
def test_run_all_outbox_lists_excluded_large_files(tmp_path):
    em, root = _make_tree(tmp_path)
    calls = tmp_path / "calls.txt"
    (root / "results").mkdir(parents=True)
    (root / "results" / "s3_coords_round0.npz").write_bytes(b"x" * 5000)
    r = _run_all(em, root, calls, IC_OUTBOX_MAX_FILE_BYTES="4000")
    assert r.returncode == 0, r.stdout
    excl = (root / "outbox" / "EXCLUDED.txt").read_text(encoding="utf-8")
    assert "results/s3_coords_round0.npz" in excl
    with tarfile.open(root / "outbox" / "ic_results.tgz") as tar:
        man = tar.extractfile("MANIFEST.sha256").read().decode()
        assert "results/s3_coords_round0.npz" not in tar.getnames()
    assert "# EXCLUDED" in man and "s3_coords_round0.npz" in man
    assert "EXCLUDED large files" in (root / "DONE").read_text(encoding="utf-8")


@needs_bash
def test_run_all_bb_failure_is_failed_and_skips_round1(tmp_path):
    em, root = _make_tree(tmp_path)
    calls = tmp_path / "calls.txt"
    r = _run_all(em, root, calls, STUB_FAIL="bb")
    assert r.returncode == 1
    done = (root / "DONE").read_text(encoding="utf-8")
    assert done.startswith("failed") and "bb rc=1" in done
    seq = _stage_calls(calls)
    assert "generate u_all --stages gen_r1,read_r1" not in seq and seq[-1] == "analyze all"


@needs_bash
def test_run_all_refuses_without_spend_census(tmp_path):
    em, root = _make_tree(tmp_path)
    calls = tmp_path / "calls.txt"
    env_unset = {"IC_SPENT_BEFORE_USD": ""}
    r = _run_all(em, root, calls, **env_unset)
    assert r.returncode == 2 and "IC_SPENT_BEFORE_USD" in r.stdout
    assert not (root / "DONE").exists() and _stage_calls(calls) == []
    # the runner's own status writer created status.json from scratch: it must already be dash-valid
    st = json.loads((root / "status.json").read_text(encoding="utf-8"))
    _assert_status_valid(st)
    assert any("refused" in e for e in st["events"])


# =============================================================================== make_bundle.py
def test_file_bytes_crlf_only_for_shell(tmp_path):
    sh, py = tmp_path / "a.sh", tmp_path / "a.py"
    sh.write_bytes(b"echo hi\r\necho there\r\n")
    py.write_bytes(b"x = 1\r\n")
    b, conv = mb.file_bytes("x/a.sh", sh)
    assert conv and b == b"echo hi\necho there\n"
    b, conv = mb.file_bytes("x/a.py", py)
    assert not conv and b == b"x = 1\r\n"


def test_stage_data_verifies_hashes(tmp_path):
    cfg = mb.load_cfg()
    src = tmp_path / "src"
    src.mkdir()
    (src / "instruct.json").write_text("[]", encoding="utf-8")       # wrong bytes -> sha mismatch
    with pytest.raises(mb.BundleError, match="sha256"):
        mb.stage_data(cfg, [src], fetch=False, data_dir=tmp_path / "data")
    with pytest.raises(mb.BundleError, match="not staged"):
        mb.stage_data(cfg, [tmp_path / "nowhere"], fetch=False, data_dir=tmp_path / "data2")


def test_stage_data_real_inputs_already_staged():
    """The frozen inputs in em/data verify against config.json + the pinned EM hashes."""
    if not (EM / "data" / "instruct.json").exists():
        pytest.skip("em/data not staged yet (run make_bundle.py --stage-only --data-from DIR)")
    rep = mb.stage_data(mb.load_cfg(), [], fetch=False)
    assert rep["instruct.json"]["sha256"] == mb.load_cfg()["data"]["cyberseceval"]["file_sha256"]
    assert rep["em_heldout.jsonl"]["sha256"] == mb.EM_HELDOUT_SHA256


def _bundle(out: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(POD / "make_bundle.py"), "--out", str(out), "--allow-unfrozen",
                           "--allow-incomplete", "--chunk-kb", "400"],
                          capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180)


def test_make_bundle_layout_manifest_reproducible(tmp_path):
    if not (EM / "data" / "instruct.json").exists():
        pytest.skip("em/data not staged yet")
    out1, out2 = tmp_path / "a" / "b.tgz", tmp_path / "c" / "b.tgz"
    r1 = _bundle(out1)
    assert r1.returncode == 0, r1.stdout + r1.stderr
    r2 = _bundle(out2)
    assert r2.returncode == 0, r2.stderr
    assert out1.read_bytes() == out2.read_bytes(), "bundle is not reproducible"
    h, name = (out1.parent / "b.tgz.sha256").read_text().split()
    assert name == "b.tgz" and h == hashlib.sha256(out1.read_bytes()).hexdigest()
    parts = sorted(p for p in out1.parent.iterdir() if ".part" in p.name and not p.name.endswith(".sha256"))
    assert b"".join(p.read_bytes() for p in parts) == out1.read_bytes()
    assert all(p.stat().st_size <= 400 * 1024 for p in parts)
    with tarfile.open(out1) as tar:
        members = {m.name: m for m in tar.getmembers()}
        man = tar.extractfile("interpcontrol/em/pod/BUNDLE_MANIFEST.txt").read().decode()
        for line in man.splitlines():
            if line.startswith("#"):
                continue
            hh, arc = line.split("  ", 1)
            assert hashlib.sha256(tar.extractfile(arc).read()).hexdigest() == hh, arc
        for arc, m in members.items():
            assert m.uid == 0 and m.gid == 0
            if m.isdir():
                assert m.mode == 0o755
                continue
            assert "__pycache__" not in arc and not arc.endswith((".pyc", ".pt"))
            assert arc.startswith(("interpcontrol/em/", "flagship/src/wor/", "flagship/third_party/jlens/")), arc
            if arc.endswith(".sh"):
                assert m.mode == 0o755 and b"\r" not in tar.extractfile(arc).read(), arc
    for need in ("interpcontrol/em/pod/run_all.sh", "interpcontrol/em/pod/bootstrap.sh",
                 "interpcontrol/em/pod/requirements.txt", "interpcontrol/em/data/instruct.json",
                 "interpcontrol/em/data/em_heldout.jsonl", "interpcontrol/em/prereg/config.json",
                 "flagship/src/wor/scorer.py", "flagship/third_party/jlens/lens.py",
                 "flagship/third_party/jlens/LICENSE"):
        assert need in members, need
    assert not any(a.startswith("interpcontrol/em/results/") for a in members)


def test_make_bundle_refuses_unfrozen_prereg(tmp_path):
    if (EM / "prereg" / "PREREG-HASH.txt").exists() and (EM / "prereg" / "vocab_ids.json").exists():
        pytest.skip("prereg already frozen")
    r = subprocess.run([sys.executable, str(POD / "make_bundle.py"), "--out", str(tmp_path / "x.tgz")],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
    assert r.returncode == 3 and "prereg not frozen" in r.stderr
    assert not (tmp_path / "x.tgz").exists()


# =============================================================================== sync_status.py
@pytest.mark.parametrize("text,want", [
    ("216.249.100.66 22626\n", ("216.249.100.66", 22626)),
    ("216.249.100.66:22626", ("216.249.100.66", 22626)),
    ("# comment\r\nroot@1.2.3.4 -p 40022\r\n", ("1.2.3.4", 40022)),
    ("ssh root@1.2.3.4 -p 40022 -i ~/.ssh/autolabs_runpod", ("1.2.3.4", 40022)),
    ("ssh -p 40022 root@host.example.net", ("host.example.net", 40022)),
    ("", None),
    ("nothing here", None),
])
def test_parse_addr(text, want):
    assert sync.parse_addr(text) == want


def _remote_out(doc, age=12, done=None, gpu="97, 70000"):
    lines = ["@@NOW 1759600000", f"@@AGE {age}"]
    if done:
        lines.append(f"@@DONE {done}")
    lines.append(f"@@GPU {gpu}")
    lines.append("@@JSON")
    return "\n".join(lines) + "\n" + (doc if isinstance(doc, str) else json.dumps(doc))


def test_parse_remote_variants():
    ok = sync.parse_remote(_remote_out({"phase": "gen_r0"}, age=5, done="ok 2026 all stages done"))
    assert ok["status"] == {"phase": "gen_r0"} and ok["age_s"] == 5 and ok["done"].startswith("ok")
    assert sync.parse_remote(_remote_out("{trunc"))["status"] is None
    assert "not an object" in sync.parse_remote(_remote_out("[1, 2]"))["error"]
    assert "missing or empty" in sync.parse_remote(_remote_out(""))["error"]
    assert "no @@JSON" in sync.parse_remote("ssh: connect refused")["error"]


def test_merge_keeps_laptop_pod_fields_and_absorbs_edits():
    remote = {"phase": "gen_r0", "pod": {"id": "abc", "gpu": "A100", "rate_usd_h": 1.59, "started_at": 1000.0,
                                         "terminated_at": None}, "spent_before_usd": 0.5}
    local = {"pod": {"id": "abc", "started_at": 990.0, "rate_usd_h": 1.59}, "spent_before_usd": 0.7}
    m1, meta = sync.merge(remote, local, {})
    assert m1["pod"]["started_at"] == 990.0 and m1["pod"]["gpu"] == "A100" and m1["spent_before_usd"] == 0.7
    # next cycle: local == what we wrote -> no new overrides; the pod's own new field flows through
    remote2 = dict(remote, phase="label", pod=dict(remote["pod"], gpu="A100-SXM4-80GB"))
    m2, meta = sync.merge(remote2, m1, meta)
    assert m2["phase"] == "label" and m2["pod"]["gpu"] == "A100-SXM4-80GB" and m2["pod"]["started_at"] == 990.0
    # the orchestrator marks the pod terminated on the laptop -> kept even if one more pull lands
    edited = json.loads(json.dumps(m2))
    edited["pod"]["terminated_at"] = 5000.0
    m3, meta = sync.merge(remote2, edited, meta)
    assert m3["pod"]["terminated_at"] == 5000.0
    # a different pod id on the pod side drops stale laptop overrides
    remote3 = dict(remote2, pod=dict(remote2["pod"], id="xyz", started_at=7000.0))
    m4, meta = sync.merge(remote3, m3, meta)
    assert m4["pod"]["id"] == "xyz" and m4["pod"]["started_at"] == 7000.0 and m4["pod"].get("terminated_at") is None


class _FakeRunner:
    def __init__(self, outs):
        self.outs = list(outs)
        self.argvs = []

    def __call__(self, argv, timeout):
        self.argvs.append(argv)
        return self.outs.pop(0) if self.outs else (255, "", "Connection refused")


def _status_doc(phase="gen_r0"):
    return {"phase": phase, "phases": ["setup", "gen_r0", "done"], "counts": {"gen_r0": [16, 1804]},
            "attempts": 3, "validated": 3, "auroc": {}, "money_shot": None,
            "pod": {"id": "abc", "gpu": "A100", "rate_usd_h": 1.59, "started_at": 1000.0, "terminated_at": None},
            "spent_before_usd": 0.0, "cap_usd": 15, "trial_cap_usd": 3, "events": ["[12:00] x"], "updated": 1.0}


def test_syncer_cycle_writes_valid_status_and_never_on_failure(tmp_path):
    out, meta, addr = tmp_path / "status.json", tmp_path / "pod_meta.json", tmp_path / "pod_addr.txt"
    addr.write_text("1.2.3.4 22022\r\n")
    run = _FakeRunner([(0, _remote_out(_status_doc()), ""), (255, "", "Connection timed out"),
                       (0, _remote_out(_status_doc(), age=3600), "")])
    s = sync.Syncer(addr_file=addr, out=out, meta_file=meta, key=tmp_path / "k", ssh="ssh", runner=run)
    r = s.cycle()
    assert r["ok"] and r["host"] == "1.2.3.4:22022"
    argv = run.argvs[0]
    assert argv[argv.index("-p") + 1] == "22022" and argv[-2] == "root@1.2.3.4"
    assert "StrictHostKeyChecking=no" in argv and "BatchMode=yes" in argv
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["sync"]["ok"] and doc["sync"]["remote_age_s"] == 12 and not doc["sync"]["stall"]
    mtime = out.stat().st_mtime_ns
    bytes_before = out.read_bytes()
    time.sleep(0.05)
    r = s.cycle()                                      # ssh failure: the file is NOT touched
    assert not r["ok"] and "timed out" in r["error"] and s.fails == 1
    assert out.read_bytes() == bytes_before and out.stat().st_mtime_ns == mtime
    r = s.cycle()                                      # stale remote file -> stall flagged
    assert r["ok"] and r["stall"] and s.fails == 0
    # the written file passes the dash reader and (if present) ic.status.validate
    dash = _load("ic_dash_server", IC_DIR / "dash" / "server.py")
    dash.STATUS = out
    c = dash.control()
    assert c["exists"] and "error" not in c and c["phase"] == "gen_r0"
    st_path = EM / "src" / "ic" / "status.py"
    if st_path.exists():
        st = _load("ic_status_for_sync_test", st_path)
        assert st.validate(json.loads(out.read_text(encoding="utf-8"))) == []


def test_syncer_no_address(tmp_path):
    s = sync.Syncer(addr_file=tmp_path / "missing.txt", out=tmp_path / "s.json", meta_file=tmp_path / "m.json",
                    key=tmp_path / "k", ssh="ssh", runner=_FakeRunner([]))
    r = s.cycle()
    assert not r["ok"] and "no pod address" in r["error"] and not (tmp_path / "s.json").exists()


def test_main_exits_after_done(tmp_path, monkeypatch):
    out, meta = tmp_path / "status.json", tmp_path / "pod_meta.json"
    done_doc = _status_doc("done")
    run = _FakeRunner([(0, _remote_out(_status_doc("analyze")), ""),
                       (0, _remote_out(done_doc, done="ok\t2026-10-05\tall stages done"), ""),
                       (0, _remote_out(done_doc, done="ok"), "")] + [(0, _remote_out(done_doc), "")] * 5)
    monkeypatch.setattr(sync, "run_ssh", run)
    monkeypatch.setattr(sync.Syncer, "__init__", _patched_init(run))
    rc = sync.main(["--addr", "1.2.3.4:22", "--out", str(out), "--meta", str(meta), "--interval", "0",
                    "--max-cycles", "10", "--artifacts-every-min", "0"])
    assert rc == 0
    assert len(run.argvs) == 3, "must stop one pull after the first DONE"
    assert json.loads(out.read_text(encoding="utf-8"))["phase"] == "done"


def _tgz(files: dict) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, data in files.items():
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            tar.addfile(ti, io.BytesIO(data))
    return buf.getvalue()


def test_extract_artifacts_only_plain_names(tmp_path):
    blob = _tgz({"gen_round0.jsonl": b'{"id": 1}\n', "../evil.txt": b"x", "sub/dir.json": b"y",
                 "bb_scores.jsonl": b"{}\n"})
    names = sync.extract_artifacts(blob, tmp_path / "pr")
    assert sorted(names) == ["bb_scores.jsonl", "gen_round0.jsonl"]
    assert not (tmp_path / "evil.txt").exists() and not (tmp_path / "pr" / "sub").exists()
    assert (tmp_path / "pr" / "gen_round0.jsonl").read_bytes() == b'{"id": 1}\n'


def test_pull_artifacts_and_done_pull_in_main(tmp_path, monkeypatch):
    blob = _tgz({"gen_round0.jsonl": b"a\n", "stage_timings.json": b"{}"})
    calls = []

    def bin_runner(argv, timeout):
        calls.append(argv[-1])
        return (4, b"", "") if len(calls) == 1 else (0, blob, "")

    s = sync.Syncer(addr="1.2.3.4:2222", out=tmp_path / "s.json", meta_file=tmp_path / "m.json", key=tmp_path / "k",
                    ssh="ssh", runner=_FakeRunner([]), artifacts_dir=tmp_path / "pr", bin_runner=bin_runner)
    assert s.pull_artifacts() == {"ok": True, "n_files": 0, "note": "no artifacts yet"}
    r = s.pull_artifacts()
    assert r["ok"] and r["n_files"] == 2 and (tmp_path / "pr" / "stage_timings.json").exists()
    man = json.loads((tmp_path / "pr" / "_pulled.json").read_text(encoding="utf-8"))
    assert man["files"]["gen_round0.jsonl"] == 2 and calls[0].startswith('cd "/workspace/ic/results"')
    bad = sync.Syncer(addr="1.2.3.4:2222", out=tmp_path / "s.json", meta_file=tmp_path / "m.json",
                      key=tmp_path / "k", ssh="ssh", runner=_FakeRunner([]), artifacts_dir=tmp_path / "pr2",
                      bin_runner=lambda a, t: (0, b"not a tgz", ""))
    assert not bad.pull_artifacts()["ok"]
    # main(): the first DONE cycle triggers one final artifact pull
    pulls = []
    done_doc = _status_doc("done")
    run = _FakeRunner([(0, _remote_out(done_doc, done="ok"), ""), (0, _remote_out(done_doc, done="ok"), "")])
    monkeypatch.setattr(sync, "run_ssh", run)
    orig = sync.Syncer.__init__

    def init(self, **kw):
        kw["runner"] = run
        kw["bin_runner"] = lambda argv, t: (pulls.append(1), (0, blob, ""))[1]
        orig(self, **kw)
    monkeypatch.setattr(sync.Syncer, "__init__", init)
    rc = sync.main(["--addr", "1.2.3.4:22", "--out", str(tmp_path / "o.json"), "--meta", str(tmp_path / "mm.json"),
                    "--interval", "0", "--max-cycles", "5", "--artifacts-dir", str(tmp_path / "pr3")])
    assert rc == 0 and pulls == [1] and (tmp_path / "pr3" / "gen_round0.jsonl").exists()


@needs_bash
def test_artifact_remote_command_under_posix_sh(tmp_path):
    res = tmp_path / "results"
    res.mkdir()
    (res / "gen_round0.jsonl").write_text('{"id": "a"}\n', encoding="utf-8")
    (res / "readout_round0.jsonl").write_text("{}\n", encoding="utf-8")
    (res / "s3_coords_round0.npz").write_bytes(b"big")          # not an artifact pattern
    cmd = sync.build_artifact_cmd(posix(res))
    r = subprocess.run([BASH, "-c", cmd], capture_output=True, timeout=60)
    assert r.returncode == 0, r.stderr
    names = sync.extract_artifacts(r.stdout, tmp_path / "got")
    assert sorted(names) == ["gen_round0.jsonl", "readout_round0.jsonl"]
    empty = tmp_path / "empty"
    empty.mkdir()
    assert subprocess.run([BASH, "-c", sync.build_artifact_cmd(posix(empty))], capture_output=True).returncode == 4
    assert subprocess.run([BASH, "-c", sync.build_artifact_cmd(posix(tmp_path / "nope"))],
                          capture_output=True).returncode == 3


def _patched_init(run):
    orig = sync.Syncer.__init__

    def init(self, **kw):
        kw["runner"] = run
        orig(self, **kw)
    return init


@needs_bash
def test_remote_command_runs_under_posix_sh(tmp_path):
    """The exact remote snippet, run by a local bash against a temp 'pod' dir, parses back."""
    pod = tmp_path / "ic"
    pod.mkdir()
    (pod / "status.json").write_text(json.dumps(_status_doc()), encoding="utf-8")
    (pod / "DONE").write_text("partial\t2026-10-05T01:00:00Z\tprimary + analyze done\n", encoding="utf-8")
    cmd = sync.build_remote_cmd(posix(pod / "status.json"), posix(pod))
    r = subprocess.run([BASH, "-c", cmd], capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert r.returncode == 0, r.stderr
    p = sync.parse_remote(r.stdout)
    assert p["status"]["phase"] == "gen_r0" and p["age_s"] is not None and 0 <= p["age_s"] < 120
    assert p["done"].startswith("partial")
    (pod / "status.json").unlink()
    p2 = sync.parse_remote(subprocess.run([BASH, "-c", cmd], capture_output=True, text=True, timeout=60).stdout)
    assert p2["status"] is None and "missing" in p2["error"]


# =============================================================================== bootstrap download block (offline)
def _hf_cached(repo: str, rev: str) -> bool:
    try:
        from huggingface_hub import constants
    except ImportError:
        return False
    return (Path(constants.HF_HUB_CACHE) / ("models--" + repo.replace("/", "--")) / "snapshots" / rev).is_dir()


def test_bootstrap_download_block_offline(tmp_path):
    """Run bootstrap.sh's embedded download/verify Python OFFLINE against small models already in the laptop
    HF cache (stand-ins for U / T / fallback; the real 32B lens file is cached too). Checks the shard-
    completeness and lens-size logic and the downloads.json it writes. Never touches the network."""
    cfg = json.loads((EM / "prereg" / "config.json").read_text(encoding="utf-8"))
    u_stand, u_rev = "Qwen/Qwen3.5-2B", cfg["models"]["T_trusted"]["revision"]          # sharded (has an index)
    fb, fb_rev = "Qwen/Qwen2.5-Coder-0.5B-Instruct", cfg["models"]["smoke_model"]["revision"]
    lens = cfg["lens"]
    if not (_hf_cached(u_stand, u_rev) and _hf_cached(fb, fb_rev) and _hf_cached(lens["repo"], lens["revision"])):
        pytest.skip("stand-in models / lens not in the local HF cache")
    from huggingface_hub import constants
    ref = Path(constants.HF_HUB_CACHE) / ("models--" + fb.replace("/", "--")) / "refs" / "main"
    if not ref.exists():
        pytest.skip("would create refs/main in the real HF cache; skipping")
    txt = (POD / "bootstrap.sh").read_text(encoding="utf-8")
    start = txt.index('<<\'PY\' || die 1 "downloads failed')
    body = txt[txt.index("\n", start) + 1:]
    body = body[: body.index("\nPY\n")]
    script = tmp_path / "dl.py"
    script.write_text(body, encoding="utf-8")
    c2 = json.loads(json.dumps(cfg))
    c2["models"]["U_untrusted"].update(repo=u_stand, revision=u_rev)
    c2["models"]["T_trusted"].update(repo=u_stand, revision=u_rev, fallback_repo=fb)
    (tmp_path / "cfg.json").write_text(json.dumps(c2), encoding="utf-8")
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    r = subprocess.run([sys.executable, str(script), str(tmp_path / "cfg.json"), str(tmp_path / "downloads.json"), fb_rev],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", env=env, timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
    d = json.loads((tmp_path / "downloads.json").read_text(encoding="utf-8"))
    assert d["U"]["n_shards"] >= 1 and d["U"]["missing_shards"] == [] and d["U"]["shard_bytes"] >= d["U"]["index_total_size"]
    assert d["lens"]["bytes"] == lens["size_bytes"]
    assert d["T"]["revision"] == u_rev and d["T_fallback"]["refs_main"]
    # a wrong lens size must fail the block (exit 1)
    c3 = json.loads(json.dumps(c2))
    c3["lens"]["size_bytes"] = lens["size_bytes"] + 1
    (tmp_path / "cfg3.json").write_text(json.dumps(c3), encoding="utf-8")
    r3 = subprocess.run([sys.executable, str(script), str(tmp_path / "cfg3.json"), str(tmp_path / "d3.json"), fb_rev],
                        capture_output=True, text=True, encoding="utf-8", errors="replace", env=env, timeout=300)
    assert r3.returncode == 1 and "lens size mismatch" in r3.stdout
