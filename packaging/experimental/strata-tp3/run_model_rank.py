#!/usr/bin/env python3
"""Run one experimental model rank; dry run by default, n_vocab defaults to 248320.

Optional state flags are passed to the driver. State contents need separate
validation; the runner only requires a fresh state directory before launch.
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
import struct
import subprocess
import sys
import time


class RunnerError(ValueError):
    pass


DISTRIBUTED_MAX_CONTEXT = 65792

FLAGS = frozenset("rank rank0-address port session-uuid library-path nccl-version library-provenance model-hash config-fingerprint pack native ple context timeout-ms logits-dir".split())
STATE_FLAGS = frozenset({"state-dir", "state-positions", "state-max-bytes"})
TRACE_FLAGS = frozenset({"trace-dir"})
POLICY_FLAGS = frozenset({"expert-collectives", "poll-interval-ms", "graph-finish", "expert-operands",
                          "device-expert-routing", "prefill-expert-batch", "model-graph"})
CAPTURE_FLAGS = frozenset({"collective-agreement", "profile-dir", "memory-phase-file"})
NETWORK_ENV = frozenset("NCCL_IB_HCA NCCL_SOCKET_IFNAME NCCL_IB_GID_INDEX NCCL_IB_ADDR_FAMILY NCCL_IB_ADDR_RANGE NCCL_IB_ROCE_VERSION_NUM NCCL_CROSS_NIC NCCL_IB_QPS_PER_CONNECTION NCCL_IB_TC NCCL_IB_SL NCCL_IB_TIMEOUT NCCL_IB_RETRY_CNT NCCL_NET_GDR_LEVEL NCCL_CUMEM_ENABLE NCCL_IB_MERGE_NICS NCCL_IB_SUBNET_AWARE_ROUTING NCCL_NET_MERGE_LEVEL NCCL_NVLS_ENABLE NCCL_IGNORE_CPU_AFFINITY".split())
SHA = re.compile(r"sha256:[0-9a-f]{64}\Z")


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            result.update(block)
    return "sha256:" + result.hexdigest()


def validated(config):
    config = dict(config)
    args = config.get("args")
    if not isinstance(args, list) or len(args) % 2 or not all(isinstance(x, str) and x and not any(c in x for c in "\x00\n\r") for x in args):
        raise RunnerError("args must be a flag/value argv list")
    options = {}
    for flag, value in zip(args[::2], args[1::2]):
        if not flag.startswith("--") or flag[2:] not in FLAGS | STATE_FLAGS | TRACE_FLAGS | POLICY_FLAGS | CAPTURE_FLAGS or flag[2:] in options:
            raise RunnerError("unknown or duplicate binary flag")
        options[flag[2:]] = value
    if not FLAGS <= set(options):
        raise RunnerError("args must provide every distributed model flag exactly once")
    if options.get("expert-collectives", "sequential") not in ("sequential", "batched"):
        raise RunnerError("invalid expert collective policy")
    if options.get("expert-operands", "full") not in ("full", "compact"):
        raise RunnerError("invalid expert operand policy")
    if options.get("collective-agreement", "per-op") not in ("per-op", "step"):
        raise RunnerError("invalid collective agreement policy")
    graph_finish = options.get("graph-finish", "0")
    if graph_finish not in ("0", "1") or (graph_finish == "1" and options.get("expert-collectives") != "batched"):
        raise RunnerError("graph-finish must be 0 or 1 and requires batched expert collectives")
    device_routing = options.get("device-expert-routing", "0")
    if device_routing not in ("0", "1"):
        raise RunnerError("device-expert-routing must be 0 or 1")
    prefill_expert_batch = options.get("prefill-expert-batch", "0")
    if prefill_expert_batch not in ("0", "1"):
        raise RunnerError("prefill-expert-batch must be 0 or 1")
    if device_routing == "1" and (
        options.get("expert-collectives") != "batched" or
        options.get("expert-operands") != "compact" or
        graph_finish != "0" or prefill_expert_batch != "0"
    ):
        raise RunnerError(
            "device-expert-routing requires batched compact experts, graph-finish 0, "
            "and prefill-expert-batch 0"
        )
    model_graph = options.get("model-graph", "0")
    if model_graph not in ("0", "1"):
        raise RunnerError("model-graph must be 0 or 1")
    if model_graph == "1" and (
        device_routing != "1" or options.get("collective-agreement") != "step" or
        "trace-dir" in options or "profile-dir" in options
    ):
        raise RunnerError("model-graph requires preagreed device routing without trace or profile observers")
    poll = options.get("poll-interval-ms", "1")
    if not poll.isdigit() or not 0 <= int(poll) <= 1000:
        raise RunnerError("poll interval must be in 0..1000 ms")
    if "trace-dir" in options and not Path(options["trace-dir"]).is_absolute():
        raise RunnerError("trace-dir must be absolute")
    if "memory-phase-file" in options and not Path(options["memory-phase-file"]).is_absolute():
        raise RunnerError("memory-phase-file must be absolute")
    if "profile-dir" in options:
        profile = Path(options["profile-dir"])
        if not profile.is_absolute():
            raise RunnerError("profile-dir must be absolute")
        if "trace-dir" in options and (profile == Path(options["trace-dir"]) or
                                        profile in Path(options["trace-dir"]).parents or
                                        Path(options["trace-dir"]) in profile.parents):
            raise RunnerError("profile-dir and trace-dir must be separate")
    if set(options) & STATE_FLAGS:
        if "state-dir" not in options or not Path(options["state-dir"]).is_absolute():
            raise RunnerError("state options require an absolute state-dir")
        if "state-positions" in options:
            positions = options["state-positions"].split(",")
            if any(value != "last" and not value.isdigit() for value in positions) or len(set(positions)) != len(positions):
                raise RunnerError("invalid or duplicate state positions")
        if "state-max-bytes" in options and (not options["state-max-bytes"].isdigit() or int(options["state-max-bytes"]) <= 0):
            raise RunnerError("state-max-bytes must be positive")
    n_vocab = config.get("n_vocab", 248320)
    if type(n_vocab) is not int or n_vocab <= 0:
        raise RunnerError("n_vocab must be a positive integer")
    for key, low, high in (("rank", 0, 2), ("port", 1024, 65535), ("context", 4, DISTRIBUTED_MAX_CONTEXT), ("timeout-ms", 100, 600000)):
        if not options[key].isdigit() or not low <= int(options[key]) <= high:
            raise RunnerError("invalid rank, port, context or timeout")
    address = ipaddress.IPv4Address(options["rank0-address"])
    if not address.is_private or address.is_loopback or address.is_unspecified or address.is_multicast:
        raise RunnerError("rank0-address must be private unicast IPv4")
    if not re.fullmatch(r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", options["session-uuid"]):
        raise RunnerError("invalid session UUID")
    for key in ("library-provenance", "model-hash", "config-fingerprint"):
        if not SHA.fullmatch(options[key]):
            raise RunnerError("invalid identity hash")
    if not re.fullmatch(r"\d+\.\d+\.\d+", options["nccl-version"]):
        raise RunnerError("nccl-version must be exact")
    for key in ("binary", "source_file", "input_file", "credential_file"):
        if not isinstance(config.get(key), str) or not Path(config[key]).is_absolute() or any(c in config[key] for c in "\x00\n\r"):
            raise RunnerError("config paths must be absolute single-line strings")
    for key in ("binary_sha256", "source_sha256", "input_sha256"):
        if not SHA.fullmatch(str(config.get(key, ""))):
            raise RunnerError("binary, source and input hashes are required")
    for key in ("library-path", "pack", "native", "ple", "logits-dir"):
        if not Path(options[key]).is_absolute():
            raise RunnerError("binary artifact paths must be absolute")
    timeout = config.get("process_timeout_s", 600)
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 1 <= timeout <= 600:
        raise RunnerError("process_timeout_s must be in 1..600")
    env = config.get("env", {})
    if not isinstance(env, dict) or set(env) - NETWORK_ENV or any(not isinstance(v, str) or any(c in v for c in "\x00\n\r") for v in env.values()):
        raise RunnerError("env must contain approved single-line NCCL settings only")
    config.update(options=options, process_timeout_s=timeout, n_vocab=n_vocab)
    return config


def credential(path):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd) as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600 or info.st_uid != os.getuid():
            raise RunnerError("credential must be owned by this user and mode 0600")
        secret = stream.read(4097).rstrip("\r\n")
    if not 16 <= len(secret) <= 4096 or any(c in secret for c in "\x00\n\r"):
        raise RunnerError("invalid credential length or format")
    return secret


def schedule(data, context):
    try:
        lines = data.decode("ascii").splitlines()
    except UnicodeDecodeError as exc:
        raise RunnerError("input must contain ASCII protocol lines") from exc
    expected, resets, position, finished = [], 0, 0, False
    for line in lines:
        if finished:
            raise RunnerError("input after QUIT")
        if line == "QUIT":
            finished = True
        elif line == "RESET":
            resets += 1
            position = 0
        elif re.fullmatch(r"GEN [0-9]+(?: [0-9]+)*", line):
            for token in map(int, line[4:].split()):
                if token > 2147483647 or position >= context:
                    raise RunnerError("input token or position exceeds bounds")
                expected.append((resets, position, token))
                position += 1
        else:
            raise RunnerError("expected GEN token IDs, RESET or QUIT")
    if not finished or not expected:
        raise RunnerError("input requires tokens and a final QUIT")
    return expected, resets


def inspect_output(stdout, network, rank, expected, resets):
    ready = re.findall(r"^READY rank=(\d+) tp=3 pp=1 layers=48 c=1(?: .*)?$", stdout, re.M)
    tokens = re.findall(r"^TOKEN position=(\d+) input=(\d+) argmax=(\d+)$", stdout, re.M)
    if ready != [str(rank)] or stdout.splitlines().count("RESET_OK") != resets:
        return "missing or mismatched READY/RESET output"
    actual = [(int(pos), int(token)) for pos, token, _ in tokens]
    wanted = [(pos, token) for _, pos, token in expected] if rank == 0 else []
    if actual != wanted:
        return "TOKEN output differs from teacher forcing schedule"
    if re.search(r"(?:nan|[+-]?inf)(?:\s|$)", stdout, re.I):
        return "nonfinite output"
    if re.search(r"Using network Socket|via NET/Socket|NET/Socket/", network, re.I):
        return "socket tensor transport detected"
    if not re.search(r"Using network IB|via NET/IB", network):
        return "missing selected NET/IB evidence"
    return ""


def inspect_logits(directory, expected, rank, n_vocab=248320):
    if rank != 0:
        return {}, ""
    paths = {f"request-{epoch}/logits-{position}.f32" for epoch, position, _ in expected}
    actual = {str(path.relative_to(directory)) for path in directory.rglob("*.f32")}
    if actual != paths:
        return {}, "logits files differ from request/position schedule"
    artifacts = {}
    for name in sorted(paths):
        path = directory / name
        data = path.read_bytes()
        if len(data) != n_vocab * 4:
            return {}, "logits byte count differs from n_vocab * 4"
        if any(not math.isfinite(value[0]) for value in struct.iter_unpack("<f", data)):
            return {}, "nonfinite logits"
        artifacts[name] = {"sha256": digest(path), "bytes": len(data)}
    if len({artifact["bytes"] for artifact in artifacts.values()}) != 1:
        return {}, "logits vocabulary sizes differ"
    return artifacts, ""


def run(supplied, evidence=None, execute=False):
    config = validated(supplied)
    options = config["options"]
    receipt = {"schema": 1, "status": "not_run", "rank": int(options["rank"]),
               "world": 3, "command": [config["binary"], *config["args"]],
               "identity": {key: options[key] for key in ("session-uuid", "model-hash", "config-fingerprint", "library-provenance")},
               **{key: config[key] for key in ("binary_sha256", "source_sha256", "input_sha256")}}
    if not execute:
        return receipt
    secret = credential(config["credential_file"])
    if any(secret in str(item) for item in receipt["command"] + list(config.get("env", {}).values())):
        raise RunnerError("credential appears in command or network settings")
    for path_key, hash_key in (("binary", "binary_sha256"), ("source_file", "source_sha256"), ("input_file", "input_sha256")):
        if digest(config[path_key]) != config[hash_key]:
            raise RunnerError("binary/source/input artifact hash disagreement")
    if digest(options["library-path"]) != options["library-provenance"]:
        raise RunnerError("library does not match approved provenance")
    data = Path(config["input_file"]).read_bytes()
    expected, resets = schedule(data, int(options["context"]))
    if "state-dir" in options and Path(options["state-dir"]).exists():
        raise RunnerError("state-dir must be fresh; driver creates it")
    if "trace-dir" in options and Path(options["trace-dir"]).exists():
        raise RunnerError("trace-dir must be fresh; driver creates it")
    evidence = Path(evidence)
    evidence.mkdir(parents=True, exist_ok=False, mode=0o700)
    logits = Path(options["logits-dir"])
    logits.mkdir(parents=True, exist_ok=False, mode=0o700)
    for epoch in range(resets + 1):
        (logits / f"request-{epoch}").mkdir(mode=0o700)
    environment = {k: v for k, v in os.environ.items() if not k.startswith("NCCL_") and k not in {"STRATA_TP3_CREDENTIAL", "LD_PRELOAD", "LD_LIBRARY_PATH"}}
    environment.update(config.get("env", {}))
    environment.update(STRATA_TP3_CREDENTIAL=secret, NCCL_NET="IB", NCCL_DEBUG="INFO",
                       NCCL_DEBUG_SUBSYS="INIT,NET", NCCL_CONF_FILE="/dev/null",
                       NCCL_DEBUG_FILE=str(evidence.resolve() / "nccl.log"))
    started, timed_out = time.monotonic(), False
    process = subprocess.Popen(receipt["command"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, env=environment, start_new_session=True)
    try:
        stdout, stderr = process.communicate(input=data, timeout=config["process_timeout_s"])
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate(timeout=10)
    stdout = stdout.decode("utf-8", errors="replace").replace(secret, "[redacted]")
    stderr = stderr.decode("utf-8", errors="replace").replace(secret, "[redacted]")
    network_file = evidence / "nccl.log"
    network = network_file.read_text(errors="replace").replace(secret, "[redacted]") if network_file.exists() else ""
    network_file.write_text(network)
    (evidence / "stdout.log").write_text(stdout)
    (evidence / "stderr.log").write_text(stderr)
    reason = inspect_output(stdout, network, receipt["rank"], expected, resets)
    artifacts, logits_reason = inspect_logits(logits, expected, receipt["rank"], config["n_vocab"])
    reason = reason or logits_reason
    if timed_out:
        reason = "process deadline expired"
    elif process.returncode != 0:
        reason = "nonzero process exit"
    receipt.update(status="failed" if reason else "passed", reason=reason,
                   exit_code=process.returncode, timed_out=timed_out,
                   elapsed_s=time.monotonic() - started, expected_tokens=len(expected),
                   expected_resets=resets, logits_dir=str(logits),
                   logits=artifacts, n_vocab=config["n_vocab"],
                   logs={name: digest(evidence / name) for name in ("stdout.log", "stderr.log", "nccl.log")})
    (evidence / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.execute and not args.evidence_dir:
            raise RunnerError("execute requires a fresh evidence directory")
        receipt = run(json.loads(args.config.read_text()), args.evidence_dir, args.execute)
        print(json.dumps(receipt, indent=2))
        return 1 if receipt["status"] == "failed" else 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print("model rank runner refused: " + str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
