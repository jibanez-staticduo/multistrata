"""Model rank process-boundary tests; no CUDA, Docker or network."""
import importlib.util
import json
import os
from pathlib import Path
import signal
import struct
import subprocess

import pytest

SPEC = importlib.util.spec_from_file_location("model_rank_runner", Path(__file__).parents[2] / "packaging/experimental/strata-tp3/run_model_rank.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


@pytest.fixture
def config(tmp_path):
    paths = {}
    for name in ("binary", "source_file", "input_file", "credential_file", "library", "pack", "native", "ple"):
        path = tmp_path / name
        path.write_bytes(b"GEN 10 20\nRESET\nGEN 30\nQUIT\n" if name == "input_file" else b"private testing credential" if name == "credential_file" else name.encode())
        paths[name] = str(path)
    Path(paths["credential_file"]).chmod(0o600)
    options = {"rank": "0", "rank0-address": "10.71.14.15", "port": "30123",
               "session-uuid": "11111111-2222-3333-4444-555555555555", "library-path": paths["library"],
               "nccl-version": "2.30.7", "library-provenance": runner.digest(paths["library"]),
               "model-hash": "sha256:" + "a" * 64, "config-fingerprint": "sha256:" + "b" * 64,
               "pack": paths["pack"], "native": paths["native"], "ple": paths["ple"],
               "context": "256", "timeout-ms": "60000", "logits-dir": str(tmp_path / "logits")}
    return {**{key: paths[key] for key in ("binary", "source_file", "input_file", "credential_file")},
            **{key + "_sha256": runner.digest(paths[key if key != "source" else "source_file"]) for key in ("binary", "source")},
            "input_sha256": runner.digest(paths["input_file"]),
            "args": [x for key, value in options.items() for x in ("--" + key, value)],
            "n_vocab": 4, "process_timeout_s": 600, "env": {"NCCL_IB_HCA": "mlx5_0"}}


def fake_process(monkeypatch, config, *, code=0, timeout=False, output=None):
    calls = {}
    class Process:
        pid = 12345
        returncode = code
        count = 0
        def __init__(self, argv, **kwargs):
            calls.update(argv=argv, **kwargs)
            Path(kwargs["env"]["NCCL_DEBUG_FILE"]).write_text("Using network IB\n" + kwargs["env"]["STRATA_TP3_CREDENTIAL"])
            logits = Path(argv[argv.index("--logits-dir") + 1])
            for epoch, position in ((0, 0), (0, 1), (1, 0)):
                (logits / f"request-{epoch}" / f"logits-{position}.f32").write_bytes(struct.pack("<ffff", 1.0, 2.0, 3.0, 4.0))
        def communicate(self, **kwargs):
            self.count += 1
            calls.setdefault("communicate", []).append(kwargs)
            if timeout and self.count == 1:
                raise subprocess.TimeoutExpired(calls["argv"], kwargs["timeout"])
            secret = calls["env"]["STRATA_TP3_CREDENTIAL"]
            text = output or "READY rank=0 tp=3 pp=1 layers=48 c=1\nTOKEN position=0 input=10 argmax=3\nTOKEN position=1 input=20 argmax=4\nRESET_OK\nTOKEN position=0 input=30 argmax=5\n"
            return (text + secret).encode(), secret.encode()
    monkeypatch.setattr(runner.subprocess, "Popen", Process)
    monkeypatch.setattr(runner.os, "killpg", lambda pid, sig: calls.update(killed=(pid, sig)))
    return calls


def test_dry_run_does_not_open_credentials_or_launch(config, monkeypatch):
    monkeypatch.setattr(runner, "credential", lambda *_: pytest.fail("read credential"))
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *_a, **_k: pytest.fail("launched"))
    result = runner.run(config)
    assert result["status"] == "not_run"
    assert "private testing credential" not in json.dumps(result)


def test_explicit_batched_polling_policy_and_64k_bounds(config):
    args = list(config["args"])
    args[args.index("--context") + 1] = "65536"
    args += ["--expert-collectives", "batched", "--poll-interval-ms", "0"]
    options = runner.validated(dict(config, args=args))["options"]
    assert options["context"] == "65536" and options["poll-interval-ms"] == "0"
    for flag, value in (("--context", "65537"), ("--poll-interval-ms", "1001"),
                        ("--expert-collectives", "unknown")):
        wrong = list(args)
        wrong[wrong.index(flag) + 1] = value
        with pytest.raises(runner.RunnerError):
            runner.validated(dict(config, args=wrong))


@pytest.mark.parametrize("policy,value,allowed", [("batched", "1", True), ("sequential", "0", True),
    ("sequential", "1", False), ("batched", "2", False), ("batched", "01", False)])
def test_graph_finish_policy(config, policy, value, allowed):
    config["args"] += ["--expert-collectives", policy, "--graph-finish", value]
    if allowed:
        assert runner.validated(config)["options"]["graph-finish"] == value
    else:
        with pytest.raises(runner.RunnerError):
            runner.validated(config)


@pytest.mark.parametrize("flag", ["--mtp-path", "--mtp-drafts", "--mtp-hash", "--parallel"])
def test_teacher_parser_refuses_service_flags(config, flag):
    config["args"] += [flag, "1"]
    with pytest.raises(runner.RunnerError, match="unknown"):
        runner.validated(config)


def test_execute_redaction_exact_input_and_hashes(config, tmp_path, monkeypatch):
    calls = fake_process(monkeypatch, config)
    monkeypatch.setenv("NCCL_NET", "Socket")
    monkeypatch.setenv("LD_PRELOAD", "/wrong/library")
    receipt = runner.run(config, tmp_path / "evidence", True)
    assert receipt["status"] == "passed"
    assert calls["start_new_session"] is True
    assert calls["communicate"][0]["input"] == Path(config["input_file"]).read_bytes()
    assert calls["env"]["NCCL_NET"] == "IB" and "LD_PRELOAD" not in calls["env"]
    assert calls["communicate"][0]["timeout"] == 600
    for file in (tmp_path / "evidence").iterdir():
        assert "private testing credential" not in file.read_text()
    assert receipt["expected_tokens"] == 3 and receipt["expected_resets"] == 1
    assert set(receipt["logits"]) == {"request-0/logits-0.f32", "request-0/logits-1.f32", "request-1/logits-0.f32"}


def test_logits_schedule_and_finite_values(tmp_path):
    directory = tmp_path / "logits"
    (directory / "request-0").mkdir(parents=True)
    path = directory / "request-0/logits-0.f32"
    expected = [(0, 0, 10)]
    assert runner.inspect_logits(directory, expected, 0, 4)[1]
    path.write_bytes(struct.pack("<ffff", float("nan"), 1, 2, 3))
    assert "nonfinite" in runner.inspect_logits(directory, expected, 0, 4)[1]
    path.write_bytes(struct.pack("<ffff", 1, 2, 3, 4))
    assert not runner.inspect_logits(directory, expected, 0, 4)[1]
    assert runner.inspect_logits(directory, expected, 0)[1]


def test_optional_state_flags_require_fresh_rank_directory(config, tmp_path):
    state = str(tmp_path / "state/rank0")
    args = config["args"] + ["--state-dir", state, "--state-positions", "0,3,last", "--state-max-bytes", "4294967296"]
    assert runner.validated(dict(config, args=args))["options"]["state-dir"] == state
    for extras in (["--state-positions", "0"], ["--state-max-bytes", "20"],
                   ["--state-dir", "relative"], ["--state-dir", state, "--state-positions", "0,0"],
                   ["--state-dir", state, "--state-max-bytes", "0"]):
        with pytest.raises(runner.RunnerError):
            runner.validated(dict(config, args=config["args"] + extras))


def test_vocab_validation_and_pinned_default(config):
    assert runner.validated({key: value for key, value in config.items() if key != "n_vocab"})["n_vocab"] == 248320
    for value in (0, -1, True, 4.5):
        with pytest.raises(runner.RunnerError):
            runner.validated(dict(config, n_vocab=value))


def test_existing_state_directory_refused_before_process(config, tmp_path, monkeypatch):
    state = tmp_path / "state/rank0"
    state.mkdir(parents=True)
    config["args"] += ["--state-dir", str(state)]
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *_a, **_k: pytest.fail("launched"))
    with pytest.raises(runner.RunnerError, match="state-dir must be fresh"):
        runner.run(config, tmp_path / "evidence", True)


@pytest.mark.parametrize("key", ["binary", "source_file", "input_file", "library"])
def test_hash_changes_refused_before_launch(config, key, monkeypatch, tmp_path):
    path = config[key] if key != "library" else dict(zip(config["args"][::2], config["args"][1::2]))["--library-path"]
    Path(path).write_bytes(b"changed")
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *_a, **_k: pytest.fail("launched"))
    with pytest.raises(runner.RunnerError, match="hash disagreement|approved provenance"):
        runner.run(config, tmp_path / "evidence", True)


def test_nonzero_77_is_failure(config, tmp_path, monkeypatch):
    fake_process(monkeypatch, config, code=77)
    receipt = runner.run(config, tmp_path / "evidence", True)
    assert receipt["status"] == "failed" and receipt["exit_code"] == 77


def test_external_deadline_kills_group_and_reaps_with_bound(config, tmp_path, monkeypatch):
    calls = fake_process(monkeypatch, config, timeout=True)
    receipt = runner.run(config, tmp_path / "evidence", True)
    assert receipt["status"] == "failed" and receipt["timed_out"]
    assert calls["killed"] == (12345, signal.SIGKILL)
    assert calls["communicate"][1] == {"timeout": 10}


def test_credentials_permissions_argv_leaks_and_validation(config, tmp_path):
    path = Path(config["credential_file"])
    path.chmod(0o644)
    with pytest.raises(runner.RunnerError, match="0600"):
        runner.run(config, tmp_path / "evidence", True)
    path.chmod(0o600)
    link = path.with_name("credential-link")
    link.symlink_to(path)
    with pytest.raises(OSError):
        runner.credential(link)
    for change in ({"process_timeout_s": 601}, {"process_timeout_s": float("nan")},
                   {"env": {"LD_LIBRARY_PATH": "/wrong"}}, {"args": config["args"] + ["--api-key", "secret"]}):
        with pytest.raises(runner.RunnerError):
            runner.validated(dict(config, **change))
    leak = list(config["args"])
    leak[leak.index("--pack") + 1] = "/private testing credential"
    with pytest.raises(runner.RunnerError, match="credential appears"):
        runner.run(dict(config, args=leak), tmp_path / "evidence", True)


def test_trace_directory_is_optional_and_must_be_fresh(config, tmp_path, monkeypatch):
    trace = tmp_path / "trace"
    traced = dict(config, args=config["args"] + ["--trace-dir", str(trace)])
    assert runner.validated(traced)["options"]["trace-dir"] == str(trace)
    trace.mkdir()
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *_a, **_k: pytest.fail("launched"))
    with pytest.raises(runner.RunnerError, match="trace-dir must be fresh"):
        runner.run(traced, tmp_path / "evidence", True)
