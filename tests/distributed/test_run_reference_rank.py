"""Reference launch contracts and real local process cleanup; no GPU or model."""
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import threading
import time

import pytest

PATH = Path(__file__).parents[2] / "packaging/experimental/strata-tp3/run_reference_rank.py"
SPEC = importlib.util.spec_from_file_location("reference_rank_runner", PATH)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


@pytest.fixture
def config(tmp_path):
    source = tmp_path / "source"
    files = {"serve/server.py": "print('reference fixture exited')\n", "serve/distributed_backend.py": "",
             "serve/chat_template.jinja": "template", "strata_control/__init__.py": "",
             "tools/strata_tokenizer.py": ""}
    for name, data in files.items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(data)
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    for name in ("vocab.json", "merges.txt", "token_type.json", "tokenizer.json"):
        (tokenizer / name).write_text("{}")
    binary = tmp_path / "strata-tp-reference"
    binary.write_text("#!/bin/false\n")
    binary.chmod(0o755)
    driver = source / "reference.cpp"
    driver.write_text("candidate driver")
    pack = tmp_path / "pack"
    pack.mkdir()
    (pack / "dense.bin").write_text("dense")
    (pack / "dense.txt").write_text("metadata")
    native = tmp_path / "weights-00001-of-00002.gguf"
    native.write_text("native first shard")
    native.with_name("weights-00002-of-00002.gguf").write_text("native second shard")
    ple = tmp_path / "ple.gguf"
    ple.write_text("PLE")
    options = {"--pack": str(pack), "--native": str(native), "--ple": str(ple), "--context": "8192"}
    return {"binary": str(binary), "binary_sha256": runner.digest(binary), "source_file": str(driver),
            "source_sha256": runner.digest(driver), "args": [item for pair in options.items() for item in pair],
            "model_manifest": {name: runner.digest(name) for name in runner.model_files(options)},
            "server_source_root": str(source), "source_commit": "a" * 40,
            "source_manifest": {name: runner.digest(source / name) for name in files},
            "tokenizer_dir": str(tokenizer),
            "tokenizer_manifest": {p.name: runner.digest(p) for p in tokenizer.iterdir()},
            "evidence_dir": str(tmp_path / "evidence"), "process_timeout_s": 1, "terminate_timeout_s": 0.1}


def change_server(config, code):
    path = Path(config["server_source_root"]) / "serve/server.py"
    path.write_text(code)
    config["source_manifest"]["serve/server.py"] = runner.digest(path)


def test_dry_run_has_no_side_effects_or_credentials(config, monkeypatch):
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *_a, **_kw: pytest.fail("launched"))
    monkeypatch.setattr(runner._helpers, "credential", lambda *_: pytest.fail("read credential"))
    receipt = runner.run(config)
    assert receipt["status"] == "validated_not_run"
    assert receipt["rank"] == 0 and receipt["world"] == receipt["parallel"] == 1
    assert receipt["numerical_validation"] == "not_performed"
    assert receipt["native_command"] == [config["binary"], "--expert-residency", "all",
                                         "--mode", "autoregressive", *config["args"]]
    assert receipt["command"][5:7] == ["reference", "--config"]
    assert receipt["command"][-5:] == ["--host", "127.0.0.1", "--port", "18081", "--no-api-key"]
    assert not Path(config["evidence_dir"]).exists()
    server = runner.server_config(runner.validated(config))
    assert server["arch"] == "qwen4exp" and server["parallel"] == 1
    assert "distributed" not in server and server["reference"]["parallel"] == 1
    assert server["tokenizer"] == config["tokenizer_dir"]


def test_release_verified_model_cache_is_explicit_and_after_validation(config, monkeypatch):
    calls = []
    monkeypatch.setattr(runner.os, "posix_fadvise", lambda fd, start, count, advice:
                        calls.append((os.readlink(f"/proc/self/fd/{fd}"), start, count, advice)))
    runner.run(config)
    assert calls == []
    enabled = dict(config, release_verified_cache=True)
    runner.run(enabled)
    assert calls == []  # Dry runs never alter cache residency.
    receipt = runner.run(enabled, True)
    assert receipt["status"] == "completed"
    assert {row[0] for row in calls} == set(config["model_manifest"])
    assert all(row[1:] == (0, 0, os.POSIX_FADV_DONTNEED) for row in calls)
    assert receipt["released_verified_cache_files"] == len(calls)


def test_release_verified_cache_rejects_nonboolean(config):
    with pytest.raises(runner.RunnerError):
        runner.validated(dict(config, release_verified_cache="yes"))


@pytest.mark.parametrize("flag", ["--mtp-path", "--mode", "--expert-residency", "--parallel", "--logits-dir",
                                  "--state-dir", "--trace-dir", "--timing-file", "--context"])
def test_extra_and_duplicate_flags_refused(config, flag):
    with pytest.raises(runner.RunnerError):
        runner.run(dict(config, args=config["args"] + [flag, "/tmp/unapproved"]))


@pytest.mark.parametrize("change", [{"env": {"NCCL_NET": "IB"}}, {"env": {"API_KEY": "private"}},
    {"credential_file": "/tmp/key"}, {"mode": "teacher-forced"}, {"parallel": True}, {"parallel": 4},
    {"expert_residency": "selected"}, {"source_commit": "HEAD"}, {"process_timeout_s": 0},
    {"process_timeout_s": 86401}, {"process_timeout_s": float("nan")}, {"ready_timeout": 299},
    {"binary": "relative"}, {"evidence_dir": "relative"}])
def test_invalid_config_refused(config, change):
    with pytest.raises(runner.RunnerError):
        runner.validated(dict(config, **change))


@pytest.mark.parametrize("context", ["3", "65793", "-1", "1e3", "true", "\u0664"])
def test_context_bounds(config, context):
    args = list(config["args"])
    args[-1] = context
    with pytest.raises(runner.RunnerError):
        runner.validated(dict(config, args=args))


@pytest.mark.parametrize("which", ["binary", "source", "server", "tokenizer", "pack", "native", "ple"])
def test_pinned_artifacts_cannot_change(config, which, monkeypatch):
    options = dict(zip(config["args"][::2], config["args"][1::2]))
    paths = {"binary": config["binary"], "source": config["source_file"],
             "server": str(Path(config["server_source_root"]) / "serve/server.py"),
             "tokenizer": str(Path(config["tokenizer_dir"]) / "vocab.json"),
             "pack": str(Path(options["--pack"]) / "dense.bin"),
             "native": str(Path(options["--native"]).with_name("weights-00002-of-00002.gguf")),
             "ple": options["--ple"]}
    Path(paths[which]).write_text("changed")
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *_a, **_kw: pytest.fail("launched"))
    with pytest.raises(runner.RunnerError):
        runner.run(config, True)


def test_manifest_exact_coverage(config, tmp_path):
    entries = dict(config["model_manifest"])
    entries.pop(next(iter(entries)))
    with pytest.raises(runner.RunnerError, match="exactly"):
        runner.validated(dict(config, model_manifest=entries))
    entries = dict(config["model_manifest"], **{str(tmp_path / "foreign"): "sha256:" + "a" * 64})
    with pytest.raises(runner.RunnerError, match="exactly"):
        runner.validated(dict(config, model_manifest=entries))
    (Path(config["server_source_root"]) / "serve/new.py").write_text("new import")
    with pytest.raises(runner.RunnerError, match="omits"):
        runner.validated(config)


def test_execute_sanitizes_environment_and_private_evidence(config, monkeypatch):
    change_server(config, "import json, os\nprint(json.dumps(dict(os.environ)))\n")
    for name in ("PYTHONSTARTUP", "LD_PRELOAD", "NCCL_NET", "STRATA_TP3_CREDENTIAL", "OPENAI_API_KEY"):
        monkeypatch.setenv(name, "inherited-private-value")
    receipt = runner.run(config, True)
    assert receipt["status"] == "completed"
    evidence = Path(config["evidence_dir"])
    environment = json.loads((evidence / "stdout.log").read_text())
    assert environment["CUDA_VISIBLE_DEVICES"] == "0"
    assert environment["PYTHONPATH"] == os.pathsep.join((config["server_source_root"], config["server_source_root"] + "/tools"))
    assert not any(name.startswith("NCCL_") or "KEY" in name or name in ("LD_PRELOAD", "PYTHONSTARTUP")
                   for name in environment)
    assert "inherited-private-value" not in json.dumps(environment)
    assert stat.S_IMODE(evidence.stat().st_mode) == 0o700
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in evidence.iterdir())
    with pytest.raises(FileExistsError):
        runner.run(config, True)


def test_log_redaction_bounds_and_chunk_boundaries(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "MAX_LOG", 128)
    path, overflow = tmp_path / "log", threading.Event()
    data = b"safe\nAuthorization: Bearer private\napi_key=private\n" + b"x" * 20000
    runner.drain(io.BytesIO(data), path, overflow)
    assert path.stat().st_size <= 128 and overflow.is_set()
    assert b"private" not in path.read_bytes() and b"safe\n" in path.read_bytes()


def test_launch_failure_produces_private_receipt_and_logs(config, monkeypatch):
    def fail(*_args, **_kwargs):
        raise OSError("private launch detail")
    monkeypatch.setattr(runner.subprocess, "Popen", fail)
    previous = os.umask(0o022)
    try:
        receipt = runner.run(config, True)
    finally:
        os.umask(previous)
    assert receipt["status"] == "failed" and receipt["exit_code"] is None
    assert receipt["reason"] == "child launch or monitoring failed"
    evidence = Path(config["evidence_dir"])
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in evidence.iterdir())
    assert "private launch detail" not in (evidence / "receipt.json").read_text()


def test_timeout_kills_stubborn_descendant(config, tmp_path):
    child_pid = tmp_path / "child.pid"
    code = ("import os, signal, time\nfrom pathlib import Path\n"
            "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
            "pid=os.fork()\n"
            f"Path({str(child_pid)!r}).write_text(str(pid)) if pid else None\n"
            "while True: time.sleep(0.1)\n")
    change_server(config, code)
    receipt = runner.run(config, True)
    assert receipt["status"] == "failed" and receipt["timed_out"]
    assert receipt["reason"] == "service window deadline expired"
    pid = int(child_pid.read_text())
    proc = Path(f"/proc/{pid}/stat")
    deadline = time.monotonic() + 2
    def still_running():
        try:
            return proc.read_text().split()[2] != "Z"
        except FileNotFoundError:
            return False
    while still_running() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert not still_running()


def test_sigterm_reaches_wrapper_process_group(config, tmp_path):
    ready = tmp_path / "ready"
    change_server(config, f"from pathlib import Path\nimport time\nPath({str(ready)!r}).touch()\ntime.sleep(30)\n")
    config["process_timeout_s"] = 30
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    process = subprocess.Popen([sys.executable, str(PATH), "--config", str(path), "--execute"],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        deadline = time.monotonic() + 5
        while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        assert ready.exists()
        process.send_signal(signal.SIGTERM)
        stdout, _ = process.communicate(timeout=5)
        receipt = json.loads(stdout)
        assert process.returncode == 1 and receipt["status"] == "failed"
        assert receipt["reason"] == "wrapper received termination signal"
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def test_overflow_terminates_window(config, monkeypatch):
    monkeypatch.setattr(runner, "MAX_LOG", 1024)
    change_server(config, "import os, time\nos.write(1, b'x'*16384)\ntime.sleep(30)\n")
    receipt = runner.run(config, True)
    assert receipt["reason"] == "log size limit exceeded"
    assert (Path(config["evidence_dir"]) / "stdout.log").stat().st_size <= 1024


def test_expanded_context_capacity(config):
    assert runner.DISTRIBUTED_MAX_CONTEXT == runner._helpers.DISTRIBUTED_MAX_CONTEXT == 65792
    args = list(config["args"])
    args[-1] = str(runner.DISTRIBUTED_MAX_CONTEXT)
    assert runner.validated(dict(config, args=args))["options"]["--context"] == "65792"


def test_optional_native_request_timing_preserves_model_flag_contract(config):
    result = runner.run(dict(config, request_timing=True))
    assert result["native_command"][-2:] == ["--request-timing", "1"]
    assert runner.run(config)["native_command"][-2:] == config["args"][-2:]
    with pytest.raises(runner.RunnerError):
        runner.validated(dict(config, request_timing="yes"))
