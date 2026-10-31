#!/usr/bin/env python3
"""Rank-local TP3 smoke runner. Dry-run by default; collect three receipts separately.

The JSON config names rank, rank0_address, port, session_uuid, library_path,
nccl_version, library_provenance (sha256:...), model_hash, config_fingerprint,
build_commit and binary. Optional: credential_env OR credential_file, timeout_ms,
process_timeout_s, fault (none/metadata-mismatch/killed-peer/missing-rank),
fault_rank and env. The missing-rank case launches only the other two ranks.
Use the same identities/session on all three hosts and a fresh evidence directory
per rank. This tool changes no services, interfaces, drivers or system settings.
"""
from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import math
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys
import time


class SmokeError(ValueError):
    pass


IDENTITY = ("session_uuid", "model_hash", "config_fingerprint", "build_commit",
            "library_provenance", "nccl_version", "fault", "fault_rank")
DEFAULTS = {"world": 3, "timeout_ms": 30000, "process_timeout_s": 300,
            "fault": "none", "fault_rank": 2, "credential_env": "STRATA_TP3_CREDENTIAL"}
NETWORK_ENV = {"NCCL_IB_HCA", "NCCL_SOCKET_IFNAME", "NCCL_IB_GID_INDEX",
               "NCCL_IB_ADDR_FAMILY", "NCCL_IB_ADDR_RANGE", "NCCL_IB_ROCE_VERSION_NUM",
               "NCCL_CROSS_NIC", "NCCL_IB_QPS_PER_CONNECTION", "NCCL_IB_TC",
               "NCCL_IB_SL", "NCCL_IB_TIMEOUT", "NCCL_IB_RETRY_CNT",
               "NCCL_NET_GDR_LEVEL", "NCCL_CUMEM_ENABLE", "NCCL_IB_MERGE_NICS",
               "NCCL_IB_SUBNET_AWARE_ROUTING", "NCCL_NET_MERGE_LEVEL", "NCCL_NVLS_ENABLE",
               "NCCL_IGNORE_CPU_AFFINITY", "LD_LIBRARY_PATH"}


def load_config(path: Path) -> dict:
    supplied = json.loads(path.read_text())
    config = dict(DEFAULTS, **supplied)
    if "credential_file" in supplied and "credential_env" not in supplied:
        config.pop("credential_env")
    for key in ("binary", "library_path", "rank", "rank0_address", "port", *IDENTITY[:6]):
        if key not in config:
            raise SmokeError(f"missing config field: {key}")
    if type(config["rank"]) is not int or config["rank"] not in range(3) or config["world"] != 3:
        raise SmokeError("world must be 3 and rank must be 0, 1 or 2")
    address = ipaddress.IPv4Address(config["rank0_address"])
    if not address.is_private or address.is_loopback or address.is_unspecified or address.is_multicast:
        raise SmokeError("rank0_address must be a private unicast IPv4 address")
    if type(config["port"]) is not int or not 1024 <= config["port"] <= 65535:
        raise SmokeError("port must be in 1024..65535")
    if not re.fullmatch(r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", config["session_uuid"]):
        raise SmokeError("session_uuid must be canonical")
    for field in ("model_hash", "config_fingerprint", "library_provenance"):
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", config[field]):
            raise SmokeError(f"{field} must be sha256:<64 lowercase hex digits>")
    if not re.fullmatch(r"[0-9a-f]{40}", config["build_commit"]):
        raise SmokeError("build_commit must be a full git commit")
    if not re.fullmatch(r"\d+\.\d+\.\d+", config["nccl_version"]):
        raise SmokeError("nccl_version must name an exact version")
    if config["fault"] not in {"none", "metadata-mismatch", "killed-peer", "missing-rank"} or config["fault_rank"] not in range(3):
        raise SmokeError("unsupported fault or fault rank")
    if config["fault"] == "missing-rank" and config["rank"] == config["fault_rank"]:
        raise SmokeError("missing-rank case must omit launching fault_rank")
    if type(config["timeout_ms"]) is not int or not 100 <= config["timeout_ms"] <= 60000:
        raise SmokeError("timeout_ms must be in 100..60000")
    if not isinstance(config["process_timeout_s"], (float, int)) or not math.isfinite(config["process_timeout_s"]) or not 1 <= config["process_timeout_s"] <= 3600:
        raise SmokeError("process_timeout_s must be in 1..3600")
    if bool(config.get("credential_env")) == bool(config.get("credential_file")):
        raise SmokeError("choose exactly one credential_env or credential_file")
    if config.get("credential_env") and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", config["credential_env"]):
        raise SmokeError("invalid credential environment name")
    if set(config.get("env", {})) - NETWORK_ENV:
        raise SmokeError("env contains an unsupported variable")
    if any(not isinstance(v, str) or "\x00" in v or "\n" in v for v in config.get("env", {}).values()):
        raise SmokeError("network environment values must be single-line strings")
    return config


def read_credential(config: dict) -> str:
    if config.get("credential_file"):
        fd = os.open(config["credential_file"], os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd) as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600 or info.st_uid != os.getuid():
                raise SmokeError("credential file must be owned by this user and mode 0600")
            secret = stream.read(4097).rstrip("\r\n")
    else:
        secret = os.environ.get(config["credential_env"], "")
    if not 16 <= len(secret) <= 4096 or "\x00" in secret or "\n" in secret or "\r" in secret:
        raise SmokeError("credential must contain 16..4096 characters on one line")
    return secret


def command(config: dict) -> list[str]:
    keys = ("rank", "world", "rank0_address", "port", "session_uuid", "library_path",
            "nccl_version", "library_provenance", "model_hash", "config_fingerprint",
            "build_commit", "timeout_ms", "fault", "fault_rank")
    return [str(config["binary"])] + [part for key in keys for part in ("--" + key.replace("_", "-"), str(config[key]))]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return "sha256:" + value.hexdigest()


def parse_events(stdout: str) -> list[dict]:
    events = []
    for line in stdout.splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict) and "event" in item:
            events.append(item)
    return events


def inspect_output(stdout: str, network: str, rank: int) -> tuple[bool, list[dict], str]:
    events = parse_events(stdout)
    results = [e for e in events if e.get("event") == "result"]
    if len(results) != 1 or results[0].get("status") != "passed" or results[0].get("rank") != rank:
        return False, events, "no unique passing rank result"
    if results[0].get("reduce_checks") != 66 or results[0].get("gather_checks") != 2 or results[0].get("q8_checks") != 2:
        return False, events, "incomplete correctness coverage"
    reductions = [e for e in events if e.get("event") == "all_reduce"]
    gathers = [e for e in events if e.get("event") == "gather_q8"]
    if len(reductions) != 66 or len(gathers) != 2:
        return False, events, "missing operation evidence"
    operations = [e for e in events if e.get("event") in ("all_reduce", "gather_q8")]
    if [event.get("sequence") for event in operations] != list(range(1, 69)):
        return False, events, "collective sequence is not contiguous"
    for dtype, width in (("fp32", 4), ("bf16", 2)):
        counts = {1, 31, 32, 33, 255, 256, 257, 640, 2560}
        counts.update(size // width for size in (10 << 10, 64 << 10, 256 << 10, 1 << 20, 2 << 20, 4 << 20, 8 << 20, 16 << 20))
        expected = {(count, varied) for count in counts for varied in (False, True)}
        actual = [(e.get("elements"), e.get("varied")) for e in reductions if e.get("dtype") == dtype]
        if len(actual) != len(expected) or set(actual) != expected or any(e.get("bytes") != e.get("elements", 0) * width for e in reductions if e.get("dtype") == dtype):
            return False, events, "payload/dtype coverage differs from smoke schedule"
    for event in reductions:
        if event.get("nonfinite") != 0 or any(not isinstance(event.get(k), (int, float)) or not math.isfinite(event[k]) or event[k] < 0 for k in ("max_error", "mean_error")):
            return False, events, "invalid numeric evidence"
        expected_absolute = 0 if not event["varied"] else (2e-6 if event["dtype"] == "fp32" else 0.004)
        expected_relative = 0 if not event["varied"] else (2e-6 if event["dtype"] == "fp32" else 0.016)
        if event.get("absolute_tolerance") != expected_absolute or event.get("relative_tolerance") != expected_relative or (not event["varied"] and event["max_error"] != 0):
            return False, events, "numeric tolerance differs from smoke contract"
    for event in gathers:
        if event.get("valid_counts") != [256, 192, 192] or event.get("logical_elements") != 640 or event.get("wire_elements") != 768 or event.get("q8_block") != [256, 512] or event.get("crosses") != 448 or event.get("independent_oracle") is not True or event.get("shard_scale_sensitive") is not True:
            return False, events, "invalid gather/Q8_K evidence"
    if any(e.get("status") == "failed" for e in events):
        return False, events, "failure event precedes result"
    if re.search(r"Using network Socket|via NET/Socket|NET/Socket/", network, re.I):
        return False, events, "socket tensor transport detected"
    if not re.search(r"Using network IB|via NET/IB", network):
        return False, events, "no selected NET/IB path in NCCL log"
    return True, events, ""


def inspect_identity(events: list[dict], identity: dict) -> bool:
    ready = [event for event in events if event.get("event") == "ready"]
    return (len(ready) == 1 and ready[0].get("rank") == identity["rank"] and
            ready[0].get("device") == 0 and ready[0].get("world") == 3 and
            ready[0].get("runtime_version") == identity["nccl_version"] and
            ready[0].get("headers_version") == 23007 and
            all(ready[0].get(key) == identity[key] for key in IDENTITY[:5]))


def run(config: dict, evidence: Path, execute: bool = False) -> dict:
    argv = command(config)
    receipt = {"schema": 1, "rank": config["rank"], "world": 3,
               **{key: config[key] for key in IDENTITY}, "command": argv,
               "network_env": config.get("env", {}), "status": "not_run"}
    if not execute:
        return receipt
    secret = read_credential(config)
    evidence.mkdir(parents=True, exist_ok=False, mode=0o700)
    binary, library = Path(config["binary"]), Path(config["library_path"])
    receipt["binary_sha256"] = digest(binary)
    receipt["library_sha256"] = digest(library)
    if receipt["library_sha256"] != config["library_provenance"]:
        raise SmokeError("library file does not match approved provenance")
    environment = os.environ.copy()
    # Prevent inherited NCCL settings from silently changing this isolated run.
    for name in list(environment):
        if name.startswith("NCCL_"):
            environment.pop(name)
    environment.update(config.get("env", {}))
    environment.update(STRATA_TP3_CREDENTIAL=secret, NCCL_NET="IB", NCCL_DEBUG="INFO",
                       NCCL_CONF_FILE="/dev/null", NCCL_DEBUG_SUBSYS="INIT,NET",
                       NCCL_DEBUG_FILE=str(evidence.resolve() / "nccl.log"))
    started = time.monotonic()
    timed_out = False
    process = subprocess.Popen(argv, env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, start_new_session=True)
    try:
        stdout, stderr = process.communicate(timeout=config["process_timeout_s"])
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
    stdout, stderr = stdout.replace(secret, "[redacted]"), stderr.replace(secret, "[redacted]")
    network_path = evidence / "nccl.log"
    network = network_path.read_text(errors="replace").replace(secret, "[redacted]") if network_path.exists() else ""
    network_path.write_text(network)
    (evidence / "stdout.log").write_text(stdout)
    (evidence / "stderr.log").write_text(stderr)
    passed, events, reason = inspect_output(stdout, network, config["rank"])
    if passed and not inspect_identity(events, config):
        passed, reason = False, "runtime/header/approval evidence differs"
    receipt.update(status="passed" if passed and process.returncode == 0 and not timed_out else "failed",
                   exit_code=process.returncode, timed_out=timed_out,
                   elapsed_s=time.monotonic() - started, events=events,
                   reason="process deadline expired" if timed_out else reason,
                   logs={name: digest(evidence / name) for name in ("stdout.log", "stderr.log", "nccl.log")})
    if process.returncode and not receipt["reason"]:
        receipt["reason"] = "nonzero process exit"
    (evidence / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def collect(paths: list[Path]) -> dict:
    receipts = [json.loads(p.read_text()) for p in paths]
    if len(receipts) not in (2, 3) or sorted(r.get("rank") for r in receipts) not in ([0, 1], [0, 2], [1, 2], [0, 1, 2]):
        raise SmokeError("need distinct rank receipts (three for success, two or three for a missing peer)")
    first = receipts[0]
    if any(r.get("schema") != 1 or r.get("world") != 3 for r in receipts):
        raise SmokeError("unsupported receipt schema or world")
    for key in IDENTITY:
        if key not in first or any(r.get(key) != first.get(key) for r in receipts):
            raise SmokeError(f"receipt identity disagreement: {key}")
    for path, receipt in zip(paths, receipts):
        if set(receipt.get("logs", {})) != {"stdout.log", "stderr.log", "nccl.log"}:
            raise SmokeError("receipt is missing evidence logs")
        for name, expected in receipt.get("logs", {}).items():
            if name not in ("stdout.log", "stderr.log", "nccl.log") or digest(path.parent / name) != expected:
                raise SmokeError("receipt log hash disagreement")
        if parse_events((path.parent / "stdout.log").read_text()) != receipt.get("events"):
            raise SmokeError("receipt events disagree with raw output")
    if any(r.get("binary_sha256") != first.get("binary_sha256") or r.get("library_sha256") != r["library_provenance"] for r in receipts):
        raise SmokeError("binary/library hash disagreement")
    if first["fault"] == "none":
        if len(receipts) != 3 or any(r.get("status") != "passed" or r.get("exit_code") != 0 or r.get("timed_out") for r in receipts):
            raise SmokeError("all three ranks must pass without timeout")
        for path, receipt in zip(paths, receipts):
            ok, events, _ = inspect_output((path.parent / "stdout.log").read_text(), (path.parent / "nccl.log").read_text(), receipt["rank"])
            if not ok or not inspect_identity(events, receipt):
                raise SmokeError("receipt logs do not establish correctness and NET/IB")
        status = "passed"
    else:
        if first["fault"] != "missing-rank" and len(receipts) != 3:
            raise SmokeError("this injected fault requires all three rank receipts")
        if first["fault"] == "missing-rank" and (len(receipts) != 2 or first["fault_rank"] in [r["rank"] for r in receipts]):
            raise SmokeError("missing-rank case must omit exactly fault_rank")
        if any(r.get("status") != "failed" or r.get("exit_code") in (None, 0) or r.get("timed_out") for r in receipts):
            raise SmokeError("fault case requires bounded nonzero exits on every launched rank")
        if any(any(e.get("event") == "result" and e.get("status") == "passed" for e in r.get("events", [])) for r in receipts):
            raise SmokeError("output accepted after injected failure")
        if any(not isinstance(r.get("elapsed_s"), (float, int)) or not math.isfinite(r["elapsed_s"]) or not 0 <= r["elapsed_s"] <= 180 for r in receipts):
            raise SmokeError("fault exceeded failure budget")
        for receipt in receipts:
            event = "injected_peer_exit" if first["fault"] == "killed-peer" and receipt["rank"] == first["fault_rank"] else "failure"
            allowed_codes = {"bad_operation", "aborted"} if first["fault"] == "metadata-mismatch" else {"peer_closed", "deadline_expired", "connect_failed", "missing_rank", "aborted", "async_error", "comm_failed"}
            if not any(e.get("event") == event and e.get("rank") == receipt["rank"] and
                       (event == "injected_peer_exit" or e.get("failure_code") in allowed_codes)
                       for e in receipt.get("events", [])):
                raise SmokeError("no explicit failure evidence for rank")
        status = "expected_failure"
    return {"schema": 1, "status": status, **{key: first[key] for key in IDENTITY},
            "ranks": sorted(r["rank"] for r in receipts), "receipts": [str(p) for p in paths]}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--evidence-dir", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--collect", type=Path, nargs="+")
    args = parser.parse_args(argv)
    try:
        if args.collect:
            report = collect(args.collect)
        else:
            if not args.config or (args.execute and (not args.evidence_dir or args.dry_run)):
                raise SmokeError("provide --config and, for --execute, a fresh --evidence-dir")
            report = run(load_config(args.config), args.evidence_dir, args.execute)
        print(json.dumps(report, indent=2))
        return 1 if report["status"] == "failed" else 0
    except (SmokeError, OSError, ValueError, KeyError, TypeError) as exc:
        print(f"smoke runner refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
