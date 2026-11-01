"""Service launcher boundaries with real local children; no GPU, SSH or model."""
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import threading

import pytest

SPEC = importlib.util.spec_from_file_location("service_rank_runner", Path(__file__).parents[2] /
                                             "packaging/experimental/strata-tp3/run_service_rank.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


@pytest.fixture
def config(tmp_path):
    source = tmp_path / "source"
    files = {"serve/server.py": "", "serve/distributed_backend.py": "", "serve/chat_template.jinja": "template",
             "strata_control/__init__.py": "", "tools/strata_tokenizer.py": ""}
    for name, data in files.items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(data)
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    for name in ("vocab.json", "merges.txt", "token_type.json", "tokenizer.json"):
        (tokenizer / name).write_text("{}")
    binary = tmp_path / "native"
    binary.write_text("#!/usr/bin/python3\nimport os\nfrom pathlib import Path\n"
                      "secret=os.environ['STRATA_TP3_CREDENTIAL']\n"
                      "Path(os.environ['NCCL_DEBUG_FILE']).write_text('Using network IB\\n'+secret)\n"
                      "os.write(1, secret[:7].encode()); os.write(1, secret[7:].encode())\n"
                      "os.write(2, secret.encode())\n")
    binary.chmod(0o755)
    driver = source / "distributed.cpp"
    driver.write_text("candidate driver")
    credential = tmp_path / "credential"
    credential.write_text("private testing credential")
    credential.chmod(0o600)
    library = tmp_path / "library"
    library.write_text("pinned NCCL")
    options = {"rank": "1", "rank0-address": "10.71.14.15", "port": "30123",
               "session-uuid": "11111111-2222-3333-4444-555555555555", "library-path": str(library),
               "nccl-version": "2.30.7", "library-provenance": runner.digest(library),
               "model-hash": "sha256:" + "a" * 64, "config-fingerprint": "sha256:" + "b" * 64,
               "pack": str(tmp_path / "pack"), "native": str(tmp_path / "weights"), "ple": str(tmp_path / "ple"),
               "context": "65536", "timeout-ms": "60000"}
    return {"binary": str(binary), "binary_sha256": runner.digest(binary), "source_file": str(driver),
            "source_sha256": runner.digest(driver), "credential_file": str(credential),
            "args": [x for key, value in options.items() for x in ("--" + key, value)],
            "server_source_root": str(source), "source_commit": "a" * 40,
            "source_manifest": {name: runner.digest(source / name) for name in files},
            "tokenizer_dir": str(tokenizer), "tokenizer_manifest": {p.name: runner.digest(p) for p in tokenizer.iterdir()},
            "evidence_dir": str(tmp_path / "evidence"), "env": {"NCCL_IB_HCA": "mlx5_0"},
            "process_timeout_s": 2, "terminate_timeout_s": 0.2}


def change_binary(config, text):
    path = Path(config["binary"])
    path.write_text("#!/usr/bin/python3\n" + text)
    config["binary_sha256"] = runner.digest(path)


def test_private_diagnostics_flush_before_child_exits(tmp_path):
    read_fd, write_fd = os.pipe()
    path = tmp_path / "live.log"
    overflow = threading.Event()
    with os.fdopen(read_fd, "rb") as stream:
        worker = threading.Thread(target=runner.drain, args=(stream, path, "private-credential", overflow))
        worker.start()
        try:
            os.write(write_fd, b"native failed private-credential\n")
            deadline = time.monotonic() + 1
            while time.monotonic() < deadline and (not path.exists() or not path.stat().st_size):
                time.sleep(.005)
            assert path.read_bytes() == b"native failed [redacted]\n"
            assert not overflow.is_set()
        finally:
            os.close(write_fd)
            worker.join(timeout=1)
        assert not worker.is_alive()


def mtp_args(config, tmp_path, rank="0"):
    root = tmp_path / "mtp"
    root.mkdir()
    for name in ("dense.txt", "dense.bin", "experts.bin"):
        (root / name).write_text(name)
    manifest = "strata-mtp-runtime-v1\n" + "".join(
        name + ":" + runner.digest(root / name)[7:] + "\n" for name in ("dense.txt", "dense.bin", "experts.bin"))
    args = list(config["args"])
    args[args.index("--rank") + 1] = rank
    return args + ["--mtp-path", str(root), "--mtp-drafts", "4", "--mtp-hash",
                   hashlib.sha256(manifest.encode()).hexdigest()]


def test_mtp_root_dry_run_verifies_files_without_credentials(config, tmp_path, monkeypatch):
    args = mtp_args(config, tmp_path, rank="00")
    monkeypatch.setattr(runner, "credential", lambda *_: pytest.fail("read credential"))
    result = runner.run(dict(config, args=args))
    assert result["mode"] == "autoregressive" and result["native_command"][-6:] == args[-6:]
    for name in ("dense.txt", "dense.bin", "experts.bin"):
        path = tmp_path / "mtp" / name
        original = path.read_bytes()
        path.write_bytes(b"changed")
        with pytest.raises(runner.RunnerError, match="hash mismatch"):
            runner.run(dict(config, args=args))
        path.unlink()
        with pytest.raises(runner.RunnerError, match="accessible"):
            runner.run(dict(config, args=args))
        path.write_bytes(original)
    (tmp_path / "mtp/draft_vocab.bin").write_bytes(b"unapproved")
    with pytest.raises(runner.RunnerError, match="subset"):
        runner.run(dict(config, args=args))


def test_mtp_follower_does_not_need_runtime_files(config, tmp_path):
    args = mtp_args(config, tmp_path, rank="1")
    for path in (tmp_path / "mtp").iterdir():
        path.unlink()
    (tmp_path / "mtp").rmdir()
    assert runner.run(dict(config, args=args))["rank"] == 1


@pytest.mark.parametrize("change", ["missing", "hash", "drafts", "path", "unknown", "duplicate", "multislot"])
def test_mtp_invalid_fields_are_not_hidden(config, tmp_path, change):
    args = mtp_args(config, tmp_path)
    if change == "missing":
        args = args[:-2]
    elif change in ("hash", "drafts", "path"):
        flag, value = {"hash": ("--mtp-hash", "A" * 64), "drafts": ("--mtp-drafts", "5"),
                       "path": ("--mtp-path", "relative")}[change]
        args[args.index(flag) + 1] = value
    elif change == "unknown":
        args += ["--mtp-unknown", "1"]
    elif change == "duplicate":
        args += ["--mtp-drafts", "2"]
    with pytest.raises(runner.RunnerError):
        runner.run(dict(config, args=args, **({"mode": "multislot"} if change == "multislot" else {})))


@pytest.mark.parametrize("parallel", [1, 4, 8])
def test_multislot_parallel_config_command_and_receipt(config, parallel):
    cfg = runner.validated(dict(config, mode="multislot", parallel=parallel))
    server = runner.server_config(cfg)
    assert server["parallel"] == server["distributed"]["parallel"] == parallel
    assert server["distributed"]["ready_timeout"] >= 300
    assert server["distributed"]["command"] == [config["binary"], "--mode", "multislot", *config["args"],
                                                 "--parallel", str(parallel)]
    result = runner.run(dict(config, mode="multislot", parallel=parallel))
    assert result["mode"] == "multislot" and result["native_command"] == server["distributed"]["command"]


@pytest.mark.parametrize("fields,args", [({"mode": "unknown"}, []), ({"parallel": 4}, []),
    ({"mode": "multislot", "parallel": True}, []), ({}, ["--parallel", "1"]),
    ({"mode": "multislot", "parallel": 4}, ["--parallel", "8"]),
    ({"mode": "multislot"}, ["--parallel", "2"]),
    ({"mode": "multislot"}, ["--parallel", "4", "--parallel", "4"])])
def test_mode_and_parallel_are_strict(config, fields, args):
    with pytest.raises(runner.RunnerError):
        runner.validated(dict(config, args=config["args"] + args, **fields))


def test_dry_run_hashes_without_reading_credential_or_launch(config, monkeypatch):
    monkeypatch.setattr(runner, "credential", lambda *_: pytest.fail("read credential"))
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *_a, **_kw: pytest.fail("launched"))
    result = runner.run(config)
    assert result["status"] == "validated_not_run"
    assert "--mode" in result["command"] and "autoregressive" in result["command"]
    assert "--logits-dir" not in result["command"]
    assert not Path(config["evidence_dir"]).exists()


def test_decode_graph_mode_composes_batched_autoregressive_command(config):
    config["args"] += [
        "--expert-collectives", "batched", "--expert-operands", "compact",
        "--device-expert-routing", "1", "--model-graph", "1",
        "--model-graph-mode", "decode", "--prefill-batch-size", "8",
        "--prefill-expert-batch", "1", "--collective-agreement", "step",
    ]
    result = runner.run(config)
    command = result["native_command"]
    assert command[:3] == [config["binary"], "--mode", "autoregressive"]
    for pair in (
        ("--prefill-batch-size", "8"),
        ("--prefill-expert-batch", "1"),
        ("--model-graph-mode", "decode"),
        ("--model-graph", "1"),
        ("--device-expert-routing", "1"),
        ("--collective-agreement", "step"),
    ):
        start = command.index(pair[0])
        assert command[start + 1] == pair[1]
    assert runner.server_config(runner.validated(config))["distributed"]["command"] == command


def test_decode_graph_mode_rejects_invalid_service_combinations(config, tmp_path):
    original = list(config["args"])
    valid_graph = ["--model-graph", "1", "--model-graph-mode", "decode", "--prefill-batch-size", "8"]
    invalid_configs = [
        dict(config, args=original + valid_graph + ["--parallel", "4"], mode="multislot", parallel=4),
        dict(config, args=original + valid_graph + ["--model-graph", "0"]),
        dict(config, args=original + valid_graph + ["--prefill-batch-size", "1"]),
        dict(config, args=original + valid_graph + mtp_args(config, tmp_path)[len(original):],
             mode="autoregressive"),
        dict(config, args=original + ["--model-graph-mode", "capture"]),
    ]
    for invalid in invalid_configs:
        with pytest.raises(runner.RunnerError):
            runner.validated(invalid)


@pytest.mark.parametrize("agreement", ["per-op", "step"])
def test_collective_agreement_forwarded_exactly(config, agreement):
    config["args"] += ["--collective-agreement", agreement]
    result = runner.run(config)
    assert result["native_command"][-2:] == ["--collective-agreement", agreement]


@pytest.mark.parametrize("agreement", ["", "operation", "STEP"])
def test_collective_agreement_rejects_unknown_values(config, agreement):
    config["args"] += ["--collective-agreement", agreement or " "]
    with pytest.raises(runner.RunnerError):
        runner.validated(config)


def test_absolute_profile_dir_forwarded_exactly_and_default_unchanged(config, tmp_path):
    default = runner.run(config)
    profile = str(tmp_path / "profile")
    config["args"] += ["--profile-dir", profile]
    explicit = runner.run(config)
    assert "--profile-dir" not in default["native_command"]
    assert explicit["native_command"][-2:] == ["--profile-dir", profile]


@pytest.mark.parametrize("flag,value", [("--profile-dir", "relative"), ("--service-state-dir", "relative")])
def test_service_capture_paths_require_absolute_path(config, flag, value):
    config["args"] += [flag, value]
    with pytest.raises(runner.RunnerError):
        runner.validated(config)


def test_service_state_dir_must_be_fresh_but_not_precreated(config, tmp_path, monkeypatch):
    state = tmp_path / "service-state"
    config["args"] += ["--service-state-dir", str(state)]
    cfg = runner.validated(config)
    assert cfg["options"]["service-state-dir"] == str(state)
    assert "--service-state-dir" in runner.run(config)["native_command"]
    assert not state.exists()
    state.mkdir()
    monkeypatch.setattr(runner, "credential", lambda *_: "private testing credential")
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *_a, **_kw: pytest.fail("launched"))
    with pytest.raises(runner.RunnerError, match="service-state-dir must be fresh"):
        runner.run(config, execute=True)


def test_profile_and_service_state_paths_must_not_overlap(config, tmp_path):
    state = tmp_path / "captures"
    profile = state / "profile"
    config["args"] += ["--service-state-dir", str(state), "--profile-dir", str(profile)]
    with pytest.raises(runner.RunnerError):
        runner.validated(config)


def test_service_state_dir_must_not_overlap_launcher_evidence(config, tmp_path):
    config["args"] += ["--service-state-dir", str(tmp_path / "evidence" / "captures")]
    with pytest.raises(runner.RunnerError, match="evidence-dir"):
        runner.validated(config)


@pytest.mark.parametrize("key", ["binary", "source_file", "library", "server", "tokenizer"])
def test_hash_mismatch_before_launch(config, key, monkeypatch):
    paths = {"binary": config["binary"], "source_file": config["source_file"],
             "library": dict(zip(config["args"][::2], config["args"][1::2]))["--library-path"],
             "server": str(Path(config["server_source_root"]) / "serve/server.py"),
             "tokenizer": str(Path(config["tokenizer_dir"]) / "vocab.json")}
    Path(paths[key]).write_text("changed")
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *_a, **_kw: pytest.fail("launched"))
    with pytest.raises(runner.RunnerError):
        runner.run(config, True)


def test_manifest_cannot_omit_or_escape_files(config):
    for bad in ({"../escape": "sha256:" + "a" * 64}, {"/absolute": "sha256:" + "a" * 64}, {}):
        with pytest.raises(runner.RunnerError):
            runner.validated(dict(config, source_manifest=bad))
    added = Path(config["server_source_root"]) / "serve/new_import.py"
    added.write_text("new code")
    with pytest.raises(runner.RunnerError, match="omits"):
        runner.validated(config)


@pytest.mark.parametrize("change", [{"env": {"LD_PRELOAD": "x"}}, {"env": {"NCCL_NET": "Socket"}},
    {"process_timeout_s": 86401}, {"process_timeout_s": float("nan")}, {"ready_timeout": 299},
    {"server_source_root": "relative"}, {"source_commit": "HEAD"}])
def test_config_constraints(config, change):
    with pytest.raises(runner.RunnerError):
        runner.validated(dict(config, **change))


def test_teacher_forced_flags_and_mode_refused(config):
    for flag in ("--mode", "--logits-dir", "--state-dir", "--trace-dir"):
        with pytest.raises(runner.RunnerError):
            runner.validated(dict(config, args=config["args"] + [flag, "/tmp/x"]))


def test_rank0_config_is_explicit_localhost_and_native(config):
    args = list(config["args"])
    args[args.index("--rank") + 1] = "0"
    cfg = runner.validated(dict(config, args=args))
    server = runner.server_config(cfg)
    assert server["host"] == "127.0.0.1" and server["port"] == 18080
    assert server["distributed"]["inherit_stderr"] is True
    assert server["tokenizer"] == config["tokenizer_dir"] and server["arch"] == "qwen4exp"
    assert server["distributed"]["command"] == [config["binary"], "--mode", "autoregressive", *args]
    assert server["distributed"]["ready_timeout"] >= 300
    assert server["distributed"]["health_timeout"] >= int(cfg["options"]["timeout-ms"]) / 1000 + 5
    assert server["distributed"]["cancel_timeout"] == 30 and server["distributed"]["shutdown_timeout"] == 10


@pytest.mark.parametrize("timeout_ms,health_s", [(100, 30), (30000, 35), (120000, 125)])
def test_health_lease_covers_a_bounded_native_step(config, timeout_ms, health_s):
    args = list(config["args"])
    args[args.index("--timeout-ms") + 1] = str(timeout_ms)
    cfg = runner.validated(dict(config, args=args))
    assert runner.server_config(cfg)["distributed"]["health_timeout"] == health_s


def test_follower_environment_redaction_and_terminal_receipt(config, monkeypatch):
    monkeypatch.setenv("LD_PRELOAD", "bad")
    monkeypatch.setenv("PYTHONPATH", "bad")
    monkeypatch.setenv("NCCL_NET", "Socket")
    result = runner.run(config, True)
    assert result["status"] == "completed" and result["numerical_validation"] == "not_performed"
    evidence = Path(config["evidence_dir"])
    assert not (evidence / "nccl.raw").exists()
    for path in evidence.iterdir():
        assert b"private testing credential" not in path.read_bytes()
    assert (evidence / "stdout.log").read_text() == "[redacted]"
    assert (evidence / "stderr.log").read_text() == "[redacted]"
    assert set(result["logs"]) == {"stdout.log", "stderr.log", "nccl.log"}


def test_controlled_python_path_contains_source_and_tokenizer_tools(config, monkeypatch):
    original = runner.subprocess.Popen
    calls = {}
    def record(*args, **kwargs):
        calls.update(kwargs)
        return original(*args, **kwargs)
    monkeypatch.setattr(runner.subprocess, "Popen", record)
    assert runner.run(config, True)["status"] == "completed"
    assert calls["env"]["PYTHONPATH"].split(os.pathsep) == [config["server_source_root"],
                                                          str(Path(config["server_source_root"]) / "tools")]
    assert calls["stdin"] == subprocess.DEVNULL and calls["start_new_session"]
    assert "LD_PRELOAD" not in calls["env"]


def test_child_start_failure_still_records_terminal_receipt(config, monkeypatch):
    def fail(*_args, **_kwargs):
        raise OSError("sensitive runtime detail")
    monkeypatch.setattr(runner.subprocess, "Popen", fail)
    result = runner.run(config, True)
    assert result["status"] == "failed" and result["reason"] == "child launch or monitoring failed"
    assert result["exit_code"] is None
    assert (Path(config["evidence_dir"]) / "receipt.json").exists()
    assert "sensitive runtime detail" not in json.dumps(result)


def test_credentials_mode_symlink_and_argv_leak(config, tmp_path):
    path = Path(config["credential_file"])
    path.chmod(0o644)
    with pytest.raises(runner.RunnerError, match="0600"):
        runner.run(config, True)
    path.chmod(0o600)
    linked = tmp_path / "linked"
    linked.symlink_to(path)
    with pytest.raises(OSError):
        runner.run(dict(config, credential_file=str(linked)), True)
    args = list(config["args"])
    args[args.index("--pack") + 1] = "/private testing credential"
    with pytest.raises(runner.RunnerError, match="credential appears"):
        runner.run(dict(config, args=args), True)


def test_socket_transport_and_missing_ib_fail(config):
    change_binary(config, "import os,time\nfrom pathlib import Path\n"
                  "Path(os.environ['NCCL_DEBUG_FILE']).write_text('Using network Socket')\ntime.sleep(5)\n")
    before = time.monotonic()
    result = runner.run(config, True)
    assert result["status"] == "failed" and result["reason"] == "socket tensor transport detected"
    assert time.monotonic() - before < 3
    assert runner.inspect_network("") == "missing selected NET/IB evidence"


def test_rdma_only_runtime_evidence():
    marker='STRATA_RDMA_ONLY transport=roce bootstrap=rdma_cm control=rdma nccl_initialized=0'
    assert runner.inspect_network(marker, rdma_only=True) == ''
    assert runner.inspect_network('', rdma_only=True) == 'missing RDMA-only runtime evidence'
    assert runner.inspect_network(marker+'\nNCCL INFO init', rdma_only=True) == 'NCCL initialization detected in RDMA-only mode'


def test_deadline_escalates_and_descendant_does_not_survive(config, tmp_path):
    marker = tmp_path / "survived"
    change_binary(config, "import os,signal,time\nfrom pathlib import Path\n"
                  "Path(os.environ['NCCL_DEBUG_FILE']).write_text('Using network IB')\n"
                  "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
                  "if os.fork()==0:\n time.sleep(2);Path(" + repr(str(marker)) + ").write_text('alive')\n"
                  "else:\n time.sleep(5)\n")
    config["process_timeout_s"] = 1
    result = runner.run(config, True)
    assert result["status"] == "failed" and result["timed_out"] and result["exit_code"] == -signal.SIGKILL
    time.sleep(1.1)
    assert not marker.exists()


def test_nonzero_exit_is_terminal(config):
    change_binary(config, "raise SystemExit(77)\n")
    result = runner.run(config, True)
    assert result["status"] == "failed" and result["exit_code"] == 77


def test_log_overflow_is_bounded_even_if_child_exits_quickly(config, monkeypatch):
    monkeypatch.setattr(runner, "MAX_LOG", 64)
    change_binary(config, "import os\nfrom pathlib import Path\n"
                  "Path(os.environ['NCCL_DEBUG_FILE']).write_text('Using network IB\\n'+'x'*4096)\n"
                  "os.write(1,b'x'*4096)\n")
    result = runner.run(config, True)
    assert result["status"] == "failed" and result["reason"] == "log size limit exceeded"
    evidence = Path(config["evidence_dir"])
    assert (evidence / "stdout.log").stat().st_size <= 64
    assert (evidence / "nccl.log").stat().st_size <= 64


def test_real_wrapper_signal_forwards_and_writes_receipt(config, tmp_path):
    change_binary(config, "import os,time\nfrom pathlib import Path\n"
                  "Path(os.environ['NCCL_DEBUG_FILE']).write_text('Using network IB')\ntime.sleep(30)\n")
    config["process_timeout_s"] = 30
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    process = subprocess.Popen(["/usr/bin/python3", str(SPEC.origin), "--config", str(path), "--execute"],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    until = time.monotonic() + 5
    while not (Path(config["evidence_dir"]) / "nccl.raw").exists() and time.monotonic() < until:
        time.sleep(0.05)
    process.send_signal(signal.SIGTERM)
    stdout, stderr = process.communicate(timeout=5)
    assert process.returncode == 1, stderr.decode()
    result = json.loads(stdout)
    assert result["reason"] == "wrapper received termination signal"
    assert (Path(config["evidence_dir"]) / "receipt.json").exists()


def test_expanded_context_capacity(config):
    assert runner.DISTRIBUTED_MAX_CONTEXT == runner._common.DISTRIBUTED_MAX_CONTEXT == 65792
    args = list(config["args"])
    at = args.index("--context") + 1
    args[at] = str(runner.DISTRIBUTED_MAX_CONTEXT)
    assert runner.validated(dict(config, args=args))["options"]["context"] == "65792"
    args[at] = str(runner.DISTRIBUTED_MAX_CONTEXT + 1)
    with pytest.raises(runner.RunnerError):
        runner.validated(dict(config, args=args))


@pytest.mark.parametrize("value", ["0", "1"])
def test_request_phase_timing_is_explicit_and_forwarded(config, value):
    config["args"] += ["--request-timing", value]
    result = runner.run(config)
    assert result["native_command"][-2:] == ["--request-timing", value]


@pytest.mark.parametrize("value", ["2", "true", "01"])
def test_request_phase_timing_rejects_invalid_value(config, value):
    config["args"] += ["--request-timing", value]
    with pytest.raises(runner.RunnerError):
        runner.validated(config)


@pytest.mark.parametrize("size", ["1", "2", "4", "8"])
def test_c1_prefill_batch_size_is_forwarded(config, size):
    config["args"] += ["--prefill-batch-size", size]
    assert runner.run(config)["native_command"][-2:] == ["--prefill-batch-size", size]


@pytest.mark.parametrize("size", ["0", "9", "-1", "2.0"])
def test_prefill_batch_size_is_bounded(config, size):
    config["args"] += ["--prefill-batch-size", size]
    with pytest.raises(runner.RunnerError):
        runner.validated(config)


def test_prefill_batching_rejects_mtp_and_multislot(config, tmp_path):
    for args, mode, parallel in ((mtp_args(config, tmp_path), "autoregressive", 1),
                                 (config["args"] + ["--parallel", "4"], "multislot", 4)):
        with pytest.raises(runner.RunnerError):
            runner.validated(dict(config, args=args + ["--prefill-batch-size", "8"], mode=mode, parallel=parallel))
