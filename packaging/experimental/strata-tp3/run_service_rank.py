#!/usr/bin/env python3
"""Bounded TP3 service window. Validate hashes by default; launch only with --execute.

Each rank config contains binary/source_file and their *_sha256, credential_file,
args (the native common flag/value pairs), server_source_root, source_manifest
(relative file -> sha256), source_commit, tokenizer_dir, tokenizer_manifest
(relative file -> sha256), evidence_dir, and optional env/process_timeout_s.
Generate the manifests from the exact staged candidate; do not include credentials.
Rank zero serves localhost:18080. Followers run the same native autoregressive driver.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time

_spec = importlib.util.spec_from_file_location("_tp3_model_runner", Path(__file__).with_name("run_model_rank.py"))
_common = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_common)
RunnerError = _common.RunnerError
digest = _common.digest
credential = _common.credential
MAX_LOG = 32 << 20


def absolute(value):
    return isinstance(value, str) and Path(value).is_absolute() and not any(c in value for c in "\0\n\r")


def number(config, key, default, low, high):
    value = config.get(key, default)
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise RunnerError("invalid bounded service duration: " + key)
    return value


def manifest(root, entries, required):
    if not isinstance(entries, dict) or not entries or not required <= set(entries):
        raise RunnerError("source/tokenizer manifest omits required files")
    for name, expected in entries.items():
        if not isinstance(name, str):
            raise RunnerError("invalid source/tokenizer manifest entry")
        path = Path(name)
        if (path.is_absolute() or ".." in path.parts or
                not _common.SHA.fullmatch(str(expected))):
            raise RunnerError("invalid source/tokenizer manifest entry")
        target = (root / path).resolve()
        if not target.is_relative_to(root.resolve()) or not target.is_file() or digest(target) != expected:
            raise RunnerError("source/tokenizer manifest hash disagreement")


def validated(supplied):
    if not isinstance(supplied, dict):
        raise RunnerError("config must be an object")
    config = dict(supplied)
    for key in ("server_source_root", "tokenizer_dir", "evidence_dir"):
        if not absolute(config.get(key)):
            raise RunnerError("service paths must be absolute single-line strings")
    args = config.get("args")
    if not isinstance(args, list) or len(args) % 2:
        raise RunnerError("args must be a flag/value argv list")
    if any(flag in {"--mode", "--logits-dir", "--trace-dir", "--state-dir", "--state-positions", "--state-max-bytes"} for flag in args[::2]):
        raise RunnerError("service args cannot include mode or teacher-forced evidence flags")
    # Reuse the pinned teacher-forced runner's common validation. Its mandatory
    # input/logits fields are placeholders here and never reach the native argv.
    normalized = dict(config, args=[*args, "--logits-dir", config["evidence_dir"]],
                      input_file=config.get("source_file"), input_sha256=config.get("source_sha256"),
                      process_timeout_s=600)
    options = dict(zip(args[::2], args[1::2]))
    context = options.get("--context", "")
    if not isinstance(context, str) or not context.isdigit() or not 4 <= int(context) <= 65536:
        raise RunnerError("service context must be in 4..65536")
    normalized["args"] = ["32768" if i and normalized["args"][i - 1] == "--context" and int(context) > 32768 else x
                          for i, x in enumerate(normalized["args"])]
    checked = _common.validated(normalized)
    config["options"] = checked["options"]
    config["options"]["context"] = context
    config["process_timeout_s"] = number(config, "process_timeout_s", 1800, 1, 86400)
    config["terminate_timeout_s"] = number(config, "terminate_timeout_s", 10, 0.1, 30)
    config["ready_timeout"] = number(config, "ready_timeout", 600, 300, 3600)
    config["request_timeout"] = number(config, "request_timeout", 900, 1, 86400)
    config["cancel_timeout"] = number(config, "cancel_timeout", 30, 1, 120)
    config["shutdown_timeout"] = number(config, "shutdown_timeout", 10, 1, 30)
    if not re.fullmatch(r"[0-9a-f]{40}", str(config.get("source_commit", ""))):
        raise RunnerError("source_commit must identify the candidate commit")
    source = Path(config["server_source_root"])
    required = {"tools/strata_tokenizer.py", "serve/server.py", "serve/distributed_backend.py", "serve/chat_template.jinja"}
    for package in ("serve", "strata_control"):
        required.update(str(p.relative_to(source)) for p in (source / package).rglob("*.py") if "__pycache__" not in p.parts)
    manifest(source, config.get("source_manifest"), required)
    manifest(Path(config["tokenizer_dir"]), config.get("tokenizer_manifest"),
             {"vocab.json", "merges.txt", "token_type.json", "tokenizer.json"})
    for path_key, hash_key in (("binary", "binary_sha256"), ("source_file", "source_sha256")):
        if digest(config[path_key]) != config[hash_key]:
            raise RunnerError("binary/source artifact hash disagreement")
    if digest(config["options"]["library-path"]) != config["options"]["library-provenance"]:
        raise RunnerError("library does not match approved provenance")
    if not os.access(config["binary"], os.X_OK):
        raise RunnerError("native binary is not executable")
    return config


def server_config(config):
    return {"host": "127.0.0.1", "port": 18080, "tokenizer": config["tokenizer_dir"],
            "arch": "qwen4exp", "model_name": "qwen3.8-flash-next-tp3-experimental",
            "parallel": 1, "open_browser": False, "before_load": None,
            "cwd": config["server_source_root"], "sampling": {"temperature": 0},
            "distributed": {"command": [config["binary"], "--mode", "autoregressive", *config["args"]],
                            "ready_timeout": config["ready_timeout"], "health_timeout": 30,
                            "process_deadline": config["request_timeout"],
                            "cancel_timeout": config["cancel_timeout"], "shutdown_timeout": config["shutdown_timeout"]}}


def signal_group(process, sig):
    try:
        os.killpg(process.pid, sig)
    except ProcessLookupError:
        pass


def drain(stream, path, secret, overflow):
    """Bound memory/disk and redact a secret even when split between pipe reads."""
    pending, written = b"", 0
    needle = secret.encode()
    with path.open("wb") as output:
        while True:
            chunk = stream.read(8192)
            pending += chunk
            pending = pending.replace(needle, b"[redacted]")
            keep = 0
            if chunk:
                for size in range(1, min(len(needle), len(pending) + 1)):
                    if pending.endswith(needle[:size]):
                        keep = size
            emit, pending = (pending[:-keep], pending[-keep:]) if keep else (pending, b"")
            if written + len(emit) > MAX_LOG:
                overflow.set()
                emit = emit[:max(0, MAX_LOG - written)]
            output.write(emit)
            written += len(emit)
            if not chunk:
                return


def inspect_network(text):
    if re.search(r"Using network Socket|via NET/Socket|NET/Socket/", text, re.I):
        return "socket tensor transport detected"
    if not re.search(r"Using network IB|via NET/IB", text):
        return "missing selected NET/IB evidence"
    return ""


def run(supplied, execute=False):
    config = validated(supplied)
    options = config["options"]
    rank = int(options["rank"])
    evidence = Path(config["evidence_dir"])
    native = [config["binary"], "--mode", "autoregressive", *config["args"]]
    command = native
    # -P omits cwd from sys.path; only the controlled staged PYTHONPATH is added.
    if rank == 0:
        command = ["/usr/bin/python3", "-P", "-m", "serve.server", "--engine", "distributed", "--config",
                   str(evidence / "server.json"), "--host", "127.0.0.1", "--port", "18080", "--no-api-key"]
    receipt = {"schema": 1, "status": "validated_not_run", "rank": rank, "world": 3,
               "mode": "autoregressive", "numerical_validation": "not_performed", "command": command,
               "native_command": native, "source_commit": config["source_commit"],
               "binary_sha256": config["binary_sha256"], "source_sha256": config["source_sha256"],
               "library_sha256": options["library-provenance"],
               "source_manifest": config["source_manifest"], "tokenizer_manifest": config["tokenizer_manifest"]}
    if not execute:
        return receipt
    secret = credential(config["credential_file"])
    if secret in json.dumps(receipt) or secret in json.dumps(config.get("env", {})):
        raise RunnerError("credential appears in command, identity or network settings")
    evidence.mkdir(parents=True, exist_ok=False, mode=0o700)
    if rank == 0:
        (evidence / "server.json").write_text(json.dumps(server_config(config), indent=2) + "\n")
    # No inherited Python, loader, API-key or NCCL controls reach the children.
    environment = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "PYTHONNOUSERSITE": "1",
                   "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1",
                   "PYTHONPATH": os.pathsep.join((config["server_source_root"], str(Path(config["server_source_root"]) / "tools")))}
    environment.update(config.get("env", {}))
    raw_network = evidence / "nccl.raw"
    environment.update(STRATA_TP3_CREDENTIAL=secret, NCCL_NET="IB", NCCL_DEBUG="INFO",
                       NCCL_DEBUG_SUBSYS="INIT,NET", NCCL_CONF_FILE="/dev/null", NCCL_DEBUG_FILE=str(raw_network))
    process, threads = None, []
    overflow, interrupted = threading.Event(), threading.Event()
    handlers = {}
    started, reason, timed_out = time.monotonic(), "", False
    old_umask = os.umask(0o077)
    try:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   cwd=config["server_source_root"], env=environment, start_new_session=True)
        def forward(sig, _frame):
            interrupted.set()
            signal_group(process, sig)
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            handlers[sig] = signal.signal(sig, forward)
        for stream, name in ((process.stdout, "stdout.log"), (process.stderr, "stderr.log")):
            thread = threading.Thread(target=drain, args=(stream, evidence / name, secret, overflow), daemon=True)
            thread.start()
            threads.append(thread)
        deadline = started + config["process_timeout_s"]
        while process.poll() is None:
            if interrupted.is_set():
                reason = "wrapper received termination signal"
                break
            if overflow.is_set() or (raw_network.exists() and raw_network.stat().st_size > MAX_LOG):
                reason = "log size limit exceeded"
                break
            if raw_network.exists():
                with raw_network.open("rb") as stream:
                    network = stream.read(MAX_LOG + 1).decode(errors="replace")
                if re.search(r"Using network Socket|via NET/Socket|NET/Socket/", network, re.I):
                    reason = "socket tensor transport detected"
                    break
            if time.monotonic() >= deadline:
                reason, timed_out = "service window deadline expired", True
                break
            time.sleep(0.1)
    except (OSError, ValueError):
        reason = "child launch or monitoring failed"
    finally:
        if process is not None:
            signal_group(process, signal.SIGTERM)
            try:
                process.wait(timeout=config["terminate_timeout_s"])
            except subprocess.TimeoutExpired:
                signal_group(process, signal.SIGKILL)
                process.wait(timeout=10)
            # The direct child can exit before a stubborn descendant; clean its group too.
            signal_group(process, signal.SIGKILL)
            for thread in threads:
                thread.join(timeout=10)
            for stream in (process.stdout, process.stderr):
                stream.close()
        for sig, previous in handlers.items():
            signal.signal(sig, previous)
        os.umask(old_umask)
    if overflow.is_set() or (raw_network.exists() and raw_network.stat().st_size > MAX_LOG):
        reason = reason or "log size limit exceeded"
    if raw_network.exists():
        with raw_network.open("rb") as stream:
            network = stream.read(MAX_LOG)
    else:
        network = b""
    network = network.replace(secret.encode(), b"[redacted]")
    (evidence / "nccl.log").write_bytes(network)
    raw_network.unlink(missing_ok=True)
    reason = reason or ("wrapper received termination signal" if interrupted.is_set() else "")
    reason = reason or ("nonzero process exit" if process and process.returncode != 0 else "") or inspect_network(network.decode(errors="replace"))
    for name in ("stdout.log", "stderr.log"):
        (evidence / name).touch(exist_ok=True)
    receipt.update(status="failed" if reason else "completed", reason=reason, timed_out=timed_out,
                   exit_code=process.returncode if process else None, elapsed_s=time.monotonic() - started,
                   logs={name: digest(evidence / name) for name in ("stdout.log", "stderr.log", "nccl.log")})
    (evidence / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        receipt = run(json.loads(args.config.read_text()), args.execute)
        print(json.dumps(receipt, indent=2))
        return 1 if receipt["status"] == "failed" else 0
    except (ValueError, OSError, KeyError, TypeError):
        # Configuration and child errors can contain paths or credentials. Keep CLI failures opaque.
        print("service rank runner refused; inspect the pinned config and artifact permissions", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
