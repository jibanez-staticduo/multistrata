#!/usr/bin/env python3
"""Bounded C1 reference HTTP window; validate every pinned file unless --execute.

Config uses binary/source_file and their *_sha256, server_source_root,
source_commit/source_manifest, tokenizer_dir/tokenizer_manifest, evidence_dir,
args (--pack, --native, --ple, --context only), and model_manifest (absolute
file -> sha256). Model pins cover every pack file, all native GGUF shards and PLE.
No credentials or environment overrides are accepted. Readiness is established
by the server's real reference handshake, never by this process wrapper.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time

_spec = importlib.util.spec_from_file_location("_tp1_service_helpers", Path(__file__).with_name("run_service_rank.py"))
_helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_helpers)
RunnerError = _helpers.RunnerError
digest = _helpers.digest
MAX_LOG = 32 << 20
SHA = re.compile(r"sha256:[0-9a-f]{64}\Z")
FLAGS = frozenset({"--pack", "--native", "--ple", "--context"})


def model_files(options):
    pack, native, ple = (Path(options[name]) for name in ("--pack", "--native", "--ple"))
    if not pack.is_dir() or not native.is_file() or not ple.is_file():
        raise RunnerError("model paths must identify the pack directory, native file and PLE file")
    files = {str(path) for path in pack.rglob("*") if path.is_file()}
    if not files:
        raise RunnerError("model pack must not be empty")
    split = re.fullmatch(r"(.*)-(\d{5})-of-(\d{5})(\.gguf)", native.name)
    if split:
        prefix, first, count, suffix = split.groups()
        if first != "00001" or not 1 <= int(count) <= 10000:
            raise RunnerError("native path must identify the first GGUF shard")
        files.update(str(native.with_name(f"{prefix}-{index:05d}-of-{count}{suffix}"))
                     for index in range(1, int(count) + 1))
    else:
        files.add(str(native))
    files.add(str(ple))
    return files


def validated(supplied):
    if not isinstance(supplied, dict):
        raise RunnerError("config must be an object")
    config = dict(supplied)
    if config.get("env") or config.get("mode", "autoregressive") != "autoregressive" or \
            config.get("expert_residency", "all") != "all" or \
            type(config.get("parallel", 1)) is not int or config.get("parallel", 1) != 1:
        raise RunnerError("reference requires C1 autoregressive all-resident mode without environment overrides")
    if "credential_file" in config:
        raise RunnerError("reference does not use transport credentials")
    for key in ("binary", "source_file", "server_source_root", "tokenizer_dir", "evidence_dir"):
        if not _helpers.absolute(config.get(key)):
            raise RunnerError("reference paths must be absolute single-line strings")
    args = config.get("args")
    if not isinstance(args, list) or len(args) != 2 * len(FLAGS) or any(
            not isinstance(arg, str) or not arg or any(c in arg for c in "\0\n\r") for arg in args):
        raise RunnerError("args must provide only pack, native, ple and context flag/value pairs")
    if set(args[::2]) != FLAGS:
        raise RunnerError("unknown, duplicate or missing reference flag")
    options = dict(zip(args[::2], args[1::2]))
    if any(not _helpers.absolute(options[key]) for key in FLAGS - {"--context"}):
        raise RunnerError("model paths must be absolute single-line strings")
    context = options["--context"]
    if not context.isascii() or not context.isdigit() or not 4 <= int(context) <= 65536:
        raise RunnerError("reference service context must be in 4..65536")
    config["options"] = options
    for key, default, low, high in (("process_timeout_s", 1800, 1, 86400),
                                   ("terminate_timeout_s", 10, 0.1, 30),
                                   ("ready_timeout", 600, 300, 3600),
                                   ("request_timeout", 900, 1, 86400),
                                   ("cancel_timeout", 30, 1, 120),
                                   ("shutdown_timeout", 10, 1, 30)):
        config[key] = _helpers.number(config, key, default, low, high)
    if not re.fullmatch(r"[0-9a-f]{40}", str(config.get("source_commit", ""))):
        raise RunnerError("source_commit must identify the candidate commit")
    source = Path(config["server_source_root"])
    required = {"tools/strata_tokenizer.py", "serve/server.py", "serve/distributed_backend.py", "serve/chat_template.jinja"}
    for package in ("serve", "strata_control"):
        required.update(str(path.relative_to(source)) for path in (source / package).rglob("*.py")
                        if "__pycache__" not in path.parts)
    _helpers.manifest(source, config.get("source_manifest"), required)
    _helpers.manifest(Path(config["tokenizer_dir"]), config.get("tokenizer_manifest"),
                      {"vocab.json", "merges.txt", "token_type.json", "tokenizer.json"})
    for key in ("binary", "source"):
        hash_key = key + "_sha256"
        path_key = "source_file" if key == "source" else key
        if not SHA.fullmatch(str(config.get(hash_key, ""))) or digest(config[path_key]) != config[hash_key]:
            raise RunnerError("binary/source artifact hash disagreement")
    if not os.access(config["binary"], os.X_OK):
        raise RunnerError("native binary is not executable")
    entries = config.get("model_manifest")
    if not isinstance(entries, dict) or set(entries) != model_files(options):
        raise RunnerError("model manifest must cover exactly the pack, native shards and PLE")
    for name, expected in entries.items():
        if not _helpers.absolute(name) or not SHA.fullmatch(str(expected)) or digest(name) != expected:
            raise RunnerError("model artifact hash disagreement")
    return config


def server_config(config):
    return {"host": "127.0.0.1", "port": 18081, "tokenizer": config["tokenizer_dir"],
            "arch": "qwen4exp", "model_name": "qwen3.8-flash-next-tp1-reference-experimental",
            "parallel": 1, "open_browser": False, "before_load": None,
            "cwd": config["server_source_root"], "sampling": {"temperature": 0},
            "reference": {"command": [config["binary"], "--expert-residency", "all",
                                      "--mode", "autoregressive", *config["args"]],
                          "parallel": 1, "ready_timeout": config["ready_timeout"],
                          "health_timeout": 30, "process_deadline": config["request_timeout"],
                          "cancel_timeout": config["cancel_timeout"],
                          "shutdown_timeout": config["shutdown_timeout"]}}


def drain(stream, path, overflow):
    """Bound logs and redact credential-bearing lines, including split reads."""
    sensitive = re.compile(rb"(?i)authorization|bearer\s|api[_ -]?key|secret|credential|sk-[a-z0-9_-]+")
    pending, written, consumed, suppress = b"", 0, 0, False
    with path.open("wb") as output:
        while True:
            chunk = stream.read(8192)
            consumed += len(chunk)
            if consumed > MAX_LOG:
                overflow.set()
            pending += chunk
            while b"\n" in pending or (not chunk and pending) or len(pending) > 8192:
                line, newline, rest = pending.partition(b"\n")
                if not newline and chunk:
                    # Never retain an unbounded line or expose a partial credential.
                    line, rest, suppress = b"[oversize line redacted]\n", b"", True
                elif suppress:
                    line, suppress = b"", False
                elif sensitive.search(line):
                    line = b"[redacted]"
                emit = line + (newline if line else b"")
                emit = emit[:max(0, MAX_LOG - written)]
                output.write(emit)
                written += len(emit)
                pending = rest
            if not chunk:
                return


def run(supplied, execute=False):
    config = validated(supplied)
    evidence = Path(config["evidence_dir"])
    command = ["/usr/bin/python3", "-P", "-m", "serve.server", "--engine", "reference", "--config",
               str(evidence / "server.json"), "--host", "127.0.0.1", "--port", "18081", "--no-api-key"]
    receipt = {"schema": 1, "status": "validated_not_run", "rank": 0, "world": 1,
               "mode": "autoregressive", "expert_residency": "all", "parallel": 1,
               "numerical_validation": "not_performed", "command": command,
               "native_command": server_config(config)["reference"]["command"],
               "source_commit": config["source_commit"], "binary_sha256": config["binary_sha256"],
               "source_sha256": config["source_sha256"], "model_manifest": config["model_manifest"],
               "source_manifest": config["source_manifest"], "tokenizer_manifest": config["tokenizer_manifest"]}
    if not execute:
        return receipt
    evidence.mkdir(parents=True, exist_ok=False, mode=0o700)
    environment = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "PYTHONNOUSERSITE": "1",
                   "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": "0",
                   "PYTHONPATH": os.pathsep.join((config["server_source_root"],
                                                str(Path(config["server_source_root"]) / "tools")))}
    process, threads = None, []
    overflow, interrupted = threading.Event(), threading.Event()
    handlers = {}
    started, reason, timed_out = time.monotonic(), "", False
    old_umask = os.umask(0o077)
    try:
        (evidence / "server.json").write_text(json.dumps(server_config(config), indent=2) + "\n")
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   cwd=config["server_source_root"], env=environment, start_new_session=True)
        def forward(sig, _frame):
            interrupted.set()
            _helpers.signal_group(process, sig)
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            handlers[sig] = signal.signal(sig, forward)
        for stream, name in ((process.stdout, "stdout.log"), (process.stderr, "stderr.log")):
            thread = threading.Thread(target=drain, args=(stream, evidence / name, overflow), daemon=True)
            thread.start()
            threads.append(thread)
        deadline = started + config["process_timeout_s"]
        while process.poll() is None:
            if interrupted.is_set():
                reason = "wrapper received termination signal"
                break
            if overflow.is_set():
                reason = "log size limit exceeded"
                break
            if time.monotonic() >= deadline:
                reason, timed_out = "service window deadline expired", True
                break
            time.sleep(0.1)
    except (OSError, ValueError):
        reason = "child launch or monitoring failed"
    finally:
        if process is not None:
            _helpers.signal_group(process, signal.SIGTERM)
            try:
                process.wait(timeout=config["terminate_timeout_s"])
            except subprocess.TimeoutExpired:
                _helpers.signal_group(process, signal.SIGKILL)
                process.wait(timeout=10)
            _helpers.signal_group(process, signal.SIGKILL)
            for thread in threads:
                thread.join(timeout=10)
            for stream in (process.stdout, process.stderr):
                stream.close()
        for sig, previous in handlers.items():
            signal.signal(sig, previous)
        os.umask(old_umask)
    reason = reason or ("log size limit exceeded" if overflow.is_set() else "")
    reason = reason or ("wrapper received termination signal" if interrupted.is_set() else "")
    reason = reason or ("nonzero process exit" if process and process.returncode != 0 else "")
    for name in ("stdout.log", "stderr.log"):
        path = evidence / name
        path.touch(exist_ok=True)
        path.chmod(0o600)
    receipt.update(status="failed" if reason else "completed", reason=reason, timed_out=timed_out,
                   exit_code=process.returncode if process else None, elapsed_s=time.monotonic() - started,
                   logs={name: digest(evidence / name) for name in ("stdout.log", "stderr.log")})
    receipt_path = evidence / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    receipt_path.chmod(0o600)
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
        print("reference rank runner refused; inspect the pinned config and artifact permissions", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
