"""The runner is tested with process-boundary fakes; no CUDA driver or network."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest

SPEC = importlib.util.spec_from_file_location("transport_smoke_runner", Path(__file__).parents[2] / "tools/distributed/run_transport_smoke.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def configuration(tmp_path, rank=0, fault="none"):
    lib = tmp_path / "libnccl.so.2"
    lib.write_bytes(b"pinned fake library")
    binary = tmp_path / "fake-smoke.py"
    return dict(binary=str(binary), rank=rank, world=3, rank0_address="10.71.14.15", port=30123,
                session_uuid="a1a1a1a1-2222-3333-4444-555555555555", library_path=str(lib),
                nccl_version="2.30.7", library_provenance=runner.digest(lib),
                model_hash="sha256:" + "a" * 64, config_fingerprint="sha256:" + "b" * 64,
                build_commit="c" * 40, timeout_ms=1000, process_timeout_s=3,
                credential_env="TEST_SMOKE_CREDENTIAL", fault=fault, fault_rank=2)


def output_events(config):
    events = [dict(event="ready", rank=config["rank"], device=0, world=3, headers_version=23007,
                   runtime_version=config["nccl_version"], **{key: config[key] for key in runner.IDENTITY[:5]})]
    sequence = 0
    for dtype, width in (("fp32", 4), ("bf16", 2)):
        counts = {1, 31, 32, 33, 255, 256, 257, 640, 2560}
        counts.update(size // width for size in (10 << 10, 64 << 10, 256 << 10, 1 << 20, 2 << 20, 4 << 20, 8 << 20, 16 << 20))
        for count in sorted(counts):
            for varied in (False, True):
                sequence += 1
                events.append(dict(event="all_reduce", sequence=sequence, dtype=dtype, elements=count,
                                   bytes=count * width, varied=varied, nonfinite=0, max_error=0, mean_error=0,
                                   absolute_tolerance=0 if not varied else (2e-6 if dtype == "fp32" else 0.004),
                                   relative_tolerance=0 if not varied else (2e-6 if dtype == "fp32" else 0.016)))
        sequence += 1
        events.append(dict(event="gather_q8", sequence=sequence, dtype=dtype, valid_counts=[256, 192, 192],
                           logical_elements=640, wire_elements=768, q8_block=[256, 512], crosses=448,
                           independent_oracle=True, shard_scale_sensitive=True))
    events += [dict(event="result", status="passed", rank=config["rank"], reduce_checks=66, gather_checks=2, q8_checks=2)]
    return events


def fake_binary(config, events=None, exit_code=0, network="Using network IB", delay=0, leak=False):
    events = output_events(config) if events is None else events
    Path(config["binary"]).with_name(f"fake-events-rank{config['rank']}.json").write_text(json.dumps(
        dict(events=events, exit_code=exit_code, network=network, delay=delay, leak=leak)))
    script = f'''#!{sys.executable}
import json, os, sys, time
from pathlib import Path
assert os.environ["NCCL_NET"] == "IB"
rank = sys.argv[sys.argv.index("--rank") + 1]
data = json.loads(Path(__file__).with_name("fake-events-rank" + rank + ".json").read_text())
Path(os.environ["NCCL_DEBUG_FILE"]).write_text(data["network"])
time.sleep(data["delay"])
for event in data["events"]: print(json.dumps(event))
if data["leak"]: print(os.environ["STRATA_TP3_CREDENTIAL"])
raise SystemExit(data["exit_code"])
'''
    Path(config["binary"]).write_text(script)
    Path(config["binary"]).chmod(0o700)


@pytest.fixture(autouse=True)
def secret(monkeypatch):
    monkeypatch.setenv("TEST_SMOKE_CREDENTIAL", "test credential never in output")


def test_dry_run_does_not_launch_or_read_credentials(tmp_path, monkeypatch):
    config = configuration(tmp_path)
    monkeypatch.delenv("TEST_SMOKE_CREDENTIAL")
    report = runner.run(config, None)
    assert report["status"] == "not_run"
    assert not Path(config["binary"]).exists()
    assert "credential" not in json.dumps(report)
    assert "--rank" in report["command"]


def test_credential_file_requires_owned_regular_0600(tmp_path):
    config = configuration(tmp_path)
    config.pop("credential_env")
    credential = tmp_path / "credential"
    credential.write_text("correct private credential\n")
    credential.chmod(0o644)
    config["credential_file"] = str(credential)
    with pytest.raises(runner.SmokeError, match="0600"):
        runner.read_credential(config)
    credential.chmod(0o600)
    assert runner.read_credential(config) == "correct private credential"
    symlink = tmp_path / "link"
    symlink.symlink_to(credential)
    config["credential_file"] = str(symlink)
    with pytest.raises(OSError):
        runner.read_credential(config)


@pytest.mark.parametrize("change", [dict(rank=3), dict(world=2), dict(rank0_address="0.0.0.0"),
                                      dict(library_provenance="unknown"), dict(timeout_ms=0),
                                      dict(env={"NCCL_NET": "Socket"}), dict(credential_env="bad name")])
def test_invalid_configuration_refused(tmp_path, change):
    config = configuration(tmp_path)
    config.update(change)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    with pytest.raises((runner.SmokeError, ValueError)):
        runner.load_config(path)


def test_triangle_library_environment_is_allowed_and_ib_is_forced(tmp_path):
    config = configuration(tmp_path)
    config["env"] = {"NCCL_CUMEM_ENABLE": "0", "NCCL_IB_MERGE_NICS": "0",
                     "NCCL_IB_SUBNET_AWARE_ROUTING": "1", "NCCL_NET_MERGE_LEVEL": "LOC",
                     "NCCL_NVLS_ENABLE": "0", "NCCL_IGNORE_CPU_AFFINITY": "1", "NCCL_CROSS_NIC": "1"}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    assert runner.load_config(path)["env"] == config["env"]
    fake_binary(config)
    assert runner.run(config, tmp_path / "rank0", True)["status"] == "passed"


def test_success_receipt_redacts_secret_and_pins_logs(tmp_path):
    config = configuration(tmp_path)
    fake_binary(config, leak=True, network="Bootstrap : Using enP7s7\nUsing network IB\nNET/Socket : available")
    evidence = tmp_path / "rank0"
    receipt = runner.run(config, evidence, execute=True)
    assert receipt["status"] == "passed"
    assert receipt["exit_code"] == 0
    assert receipt["binary_sha256"] == runner.digest(Path(config["binary"]))
    assert receipt["logs"]["stdout.log"] == runner.digest(evidence / "stdout.log")
    for path in evidence.iterdir():
        assert "test credential never in output" not in path.read_text()
    assert "[redacted]" in (evidence / "stdout.log").read_text()


@pytest.mark.parametrize("network", ["Using network Socket", "via NET/Socket/0", "NET/IB : available only"])
def test_missing_ib_or_socket_data_refused(tmp_path, network):
    config = configuration(tmp_path)
    fake_binary(config, network=network)
    assert runner.run(config, tmp_path / "run", True)["status"] == "failed"


def test_nonzero_exit_overrides_passing_output(tmp_path):
    config = configuration(tmp_path)
    fake_binary(config, exit_code=7)
    assert runner.run(config, tmp_path / "run", True)["status"] == "failed"


def test_timeout_kills_process_and_records_failure(tmp_path):
    config = configuration(tmp_path)
    config["process_timeout_s"] = 0.05
    fake_binary(config, delay=3)
    report = runner.run(config, tmp_path / "run", True)
    assert report["status"] == "failed" and report["timed_out"]
    assert report["exit_code"] < 0


@pytest.mark.parametrize("mutate", ["nan", "incomplete", "failure", "duplicate", "tolerance", "sequence", "payload"])
def test_corrupt_numeric_or_result_evidence_refused(tmp_path, mutate):
    config = configuration(tmp_path)
    events = output_events(config)
    if mutate == "nan": events[1]["max_error"] = float("nan")
    if mutate == "incomplete": events.pop(2)
    if mutate == "failure": events.insert(1, dict(event="failure", status="failed"))
    if mutate == "duplicate": events.append(events[-1])
    if mutate == "tolerance": events[1]["absolute_tolerance"] = 1
    if mutate == "sequence": events[1]["sequence"] = 99
    if mutate == "payload": events[1]["elements"] = 9
    fake_binary(config, events)
    assert runner.run(config, tmp_path / "run", True)["status"] == "failed"


def make_receipts(tmp_path, fault="none", ranks=(0, 1, 2)):
    paths = []
    for rank in ranks:
        config = configuration(tmp_path, rank, fault)
        events = None if fault == "none" else [dict(event="failure", status="failed", rank=rank,
                    failure_code="bad_operation" if fault == "metadata-mismatch" else "peer_closed")]
        if fault == "killed-peer" and rank == 2:
            events = [dict(event="injected_peer_exit", rank=rank)]
        fake_binary(config, events, exit_code=0 if fault == "none" else 1)
        evidence = tmp_path / f"rank{rank}"
        runner.run(config, evidence, True)
        paths.append(evidence / "receipt.json")
    return paths


def test_collect_three_success_receipts(tmp_path):
    paths = make_receipts(tmp_path)
    assert runner.collect(paths)["status"] == "passed"
    with pytest.raises(runner.SmokeError): runner.collect(paths[:2])
    with pytest.raises(runner.SmokeError): runner.collect([paths[0]] * 3)


@pytest.mark.parametrize("field", runner.IDENTITY)
def test_collect_rejects_identity_disagreement(tmp_path, field):
    paths = make_receipts(tmp_path)
    report = json.loads(paths[1].read_text())
    report[field] = "different"
    paths[1].write_text(json.dumps(report))
    with pytest.raises(runner.SmokeError, match="identity disagreement"):
        runner.collect(paths)


def test_collect_rejects_changed_logs(tmp_path):
    paths = make_receipts(tmp_path)
    (paths[1].parent / "stdout.log").write_text("overwritten")
    with pytest.raises(runner.SmokeError, match="hash disagreement"):
        runner.collect(paths)


@pytest.mark.parametrize("fault,ranks", [("metadata-mismatch", (0, 1, 2)), ("killed-peer", (0, 1, 2)), ("missing-rank", (0, 1))])
def test_collect_expected_bounded_failure(tmp_path, fault, ranks):
    assert runner.collect(make_receipts(tmp_path, fault, ranks))["status"] == "expected_failure"


def test_collect_refuses_unexplained_crash_and_timeout(tmp_path):
    paths = make_receipts(tmp_path, "metadata-mismatch")
    report = json.loads(paths[1].read_text())
    report["events"] = []
    paths[1].write_text(json.dumps(report))
    with pytest.raises(runner.SmokeError, match="events disagree"):
        runner.collect(paths)
    report["events"] = runner.parse_events((paths[1].parent / "stdout.log").read_text())
    report["timed_out"] = True
    paths[1].write_text(json.dumps(report))
    with pytest.raises(runner.SmokeError, match="bounded nonzero"):
        runner.collect(paths)
