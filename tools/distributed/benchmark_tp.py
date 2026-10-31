#!/usr/bin/env python3
"""Matched TP1/TP2/TP3 HTTP benchmark. Dry run unless --execute is given.

Input JSON (paths relative to it): schema_version=1, prompts="prompts.json",
endpoints=[{name:"tp1", supported:true, url:"http://127.0.0.1:8095/v1/chat/completions",
snapshot:"tp1.json", controls:{prompt_tokens:[4096,16384,65536], concurrency:[1,4,8],output_tokens:[256]}}].
Name all three endpoints; unsupported entries need supported=false and reason.
Supported snapshots contain identity={revision,pack_sha256,tokenizer_sha256,
chat_template_sha256}, settings={context,sampling,chat_template_kwargs}, and
policy={prefix_cache:"off"}. All three sections must match across endpoints.
Snapshot identity and settings are operator-provided frozen evidence, not remote attestation.

Prompt artifact: {tokenizer:{kind:"strata_pack",pack_sha256,tokenizer_sha256,
chat_template_sha256}, prompts:[{prompt_tokens:4096,text:"...",token_ids:[...]}]}.
IDs must be the exact rendered chat prompt from that native pack/template, not
just the user text. The server's reported prompt usage must match exactly.
No server is launched, cache reset, or remote telemetry polled. Optional telemetry
is a preexisting JSONL file. No output text is saved unless --save-text is given.

Times use monotonic seconds and include client/server queueing. TTFT means first
nonempty content delta; reasoning is timed separately. Output window is measured
from first content to stream end; it is not engine decode time. Engine phase times are present only when supplied by the request itself;
missing prefill, queue, sampling or decode times remain null.
Round throughput is successful completion tokens / whole concurrent round wall;
failed rounds have no throughput. p50/p95 use nearest-rank ceil(p*n)-1.
"""
from __future__ import annotations

import argparse
import codecs
import concurrent.futures
import copy
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import re
import socket
import tempfile
import threading
import time
from urllib.parse import urlsplit


PROMPTS = (4096, 16384, 65536)
CONCURRENCY = (1, 4, 8)
OUTPUT_TOKENS = 256
ENDPOINTS = ("tp1", "tp2", "tp3")


class InvalidBenchmark(ValueError):
    pass


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def read_json(path: Path, hashes: dict) -> dict:
    raw = path.read_bytes()
    hashes[str(path.resolve())] = digest(raw)
    def invalid_number(_):
        raise InvalidBenchmark("nonfinite JSON value")
    value = json.loads(raw, parse_constant=invalid_number)
    if not isinstance(value, dict):
        raise InvalidBenchmark("JSON artifact must be an object")
    return value


def positive_seconds(value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise InvalidBenchmark("timeout must be positive finite seconds")
    return value


def identity(value: dict) -> None:
    if not isinstance(value, dict) or not isinstance(value.get("revision"), str) or not value["revision"]:
        raise InvalidBenchmark("snapshot needs a revision identity")
    for key in ("pack_sha256", "tokenizer_sha256", "chat_template_sha256"):
        if not isinstance(value.get(key), str) or not re.fullmatch(r"[0-9a-f]{64}", value[key]):
            raise InvalidBenchmark(f"snapshot needs {key}")


def load_plan(path: Path) -> dict:
    """Validate and freeze all local inputs before any endpoint can be contacted."""
    hashes = {}
    cfg = read_json(path, hashes)
    if cfg.get("schema_version") != 1:
        raise InvalidBenchmark("unsupported benchmark schema")
    endpoints = cfg.get("endpoints")
    if not isinstance(endpoints, list) or len(endpoints) != 3 or {e.get("name") for e in endpoints} != set(ENDPOINTS):
        raise InvalidBenchmark("name tp1, tp2 and tp3 exactly once")
    frozen = None
    for endpoint in endpoints:
        if type(endpoint.get("supported")) is not bool:
            raise InvalidBenchmark("endpoint supported must be explicit true or false")
        if not endpoint["supported"]:
            if not isinstance(endpoint.get("reason"), str) or not endpoint["reason"].strip():
                raise InvalidBenchmark("unsupported endpoint needs a reason")
            continue
        parsed = urlsplit(endpoint.get("url", ""))
        if (parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password
                or parsed.query or parsed.fragment or not parsed.path.endswith("/chat/completions")):
            raise InvalidBenchmark("endpoint URL must name chat/completions without credentials or query")
        if "api_key" in endpoint:
            raise InvalidBenchmark("use api_key_env instead of an inline key")
        snap = read_json(path.parent / endpoint["snapshot"], hashes)
        identity(snap.get("identity"))
        settings, policy = snap.get("settings"), snap.get("policy")
        if (not isinstance(settings, dict) or type(settings.get("context")) is not int
                or settings["context"] <= OUTPUT_TOKENS
                or not isinstance(settings.get("sampling"), dict)
                or not isinstance(settings.get("chat_template_kwargs"), dict)):
            raise InvalidBenchmark("snapshot needs context, sampling and chat_template_kwargs")
        if not isinstance(policy, dict) or policy.get("prefix_cache") != "off":
            raise InvalidBenchmark("frozen policy must declare prefix_cache off; no reset is performed")
        reserved = {"messages", "model", "stream", "stream_options", "max_tokens", "max_completion_tokens", "n"}
        if reserved.intersection(settings["sampling"]):
            raise InvalidBenchmark("sampling must not override request controls")
        match = {key: snap[key] for key in ("identity", "settings", "policy")}
        if frozen is not None and match != frozen:
            raise InvalidBenchmark("endpoints have unmatched sources, settings or frozen policy")
        frozen = copy.deepcopy(match)
        controls = endpoint.get("controls")
        if not isinstance(controls, dict):
            raise InvalidBenchmark("endpoint needs explicit supported controls")
        for key, allowed in (("prompt_tokens", PROMPTS), ("concurrency", CONCURRENCY)):
            values = controls.get(key)
            if (not isinstance(values, list) or not values or len(set(values)) != len(values)
                    or any(type(x) is not int or x not in allowed for x in values)):
                raise InvalidBenchmark(f"invalid {key} controls (concurrency cap is 8)")
        output = controls.get("output_tokens")
        if not isinstance(output, list) or any(type(x) is not int or x != OUTPUT_TOKENS for x in output) or len(output) > 1:
            raise InvalidBenchmark("output_tokens controls must explicitly be [256] or [] (unsupported)")
    if frozen is None:
        raise InvalidBenchmark("at least one supported endpoint and native prompt artifact are required")
    artifact = read_json(path.parent / cfg["prompts"], hashes)
    provenance = artifact.get("tokenizer", {})
    if provenance.get("kind") != "strata_pack":
        raise InvalidBenchmark("prompts require native strata_pack tokenizer provenance")
    for key in ("pack_sha256", "tokenizer_sha256", "chat_template_sha256"):
        if provenance.get(key) != frozen["identity"][key]:
            raise InvalidBenchmark("prompt tokenizer provenance does not match the frozen source")
    prompts = artifact.get("prompts")
    if not isinstance(prompts, list) or len(prompts) != 3:
        raise InvalidBenchmark("provide exactly one actual native-tokenized 4K, 16K and 64K prompt")
    by_length = {}
    for prompt in prompts:
        n, ids = prompt.get("prompt_tokens"), prompt.get("token_ids")
        if (type(n) is not int or n not in PROMPTS or n in by_length or not isinstance(ids, list)
                or len(ids) != n or any(type(i) is not int or i < 0 for i in ids)
                or not isinstance(prompt.get("text"), str) or not prompt["text"]):
            raise InvalidBenchmark("prompt text and exact rendered token IDs must match the target length")
        by_length[n] = prompt
    jobs = []
    mapping = {e["name"]: e for e in endpoints}
    for length in PROMPTS:
        for concurrency in CONCURRENCY:
            for rnd in range(3):
                order = ENDPOINTS[rnd:] + ENDPOINTS[:rnd]
                for name in order:
                    endpoint = mapping[name]
                    reason = endpoint.get("reason") if not endpoint["supported"] else None
                    if endpoint["supported"] and (length not in endpoint["controls"]["prompt_tokens"]
                                                   or concurrency not in endpoint["controls"]["concurrency"]
                                                   or OUTPUT_TOKENS not in endpoint["controls"]["output_tokens"]):
                        reason = "endpoint does not support this prompt/concurrency control"
                    if endpoint["supported"] and length + OUTPUT_TOKENS > frozen["settings"]["context"]:
                        reason = "prompt and output budget exceed the frozen context"
                    jobs.append({"endpoint": name, "prompt_tokens": length, "concurrency": concurrency,
                                 "max_tokens": OUTPUT_TOKENS, "round": rnd + 1,
                                 "status": "NOT_RUN" if reason else "PLANNED", "reason": reason})
    return {"schema_version": 1, "frozen": frozen, "files_sha256": hashes,
            "endpoints": copy.deepcopy(endpoints), "prompts": by_length, "jobs": jobs}


class SSEParser:
    """Incremental UTF-8 SSE data events, including CR, LF, CRLF and multiline data."""
    def __init__(self):
        self.decoder = codecs.getincrementaldecoder("utf-8-sig")("strict")
        self.line, self.data, self.skip_lf = "", [], False
        self.size = 0

    def feed(self, chunk: bytes) -> list[str]:
        events = []
        for char in self.decoder.decode(chunk):
            if self.skip_lf:
                self.skip_lf = False
                if char == "\n":
                    continue
            if char in ("\r", "\n"):
                if not self.line:
                    if self.data:
                        events.append("\n".join(self.data))
                    self.data, self.size = [], 0
                elif self.line == "data" or self.line.startswith("data:"):
                    value = self.line.partition(":")[2]
                    self.data.append(value[1:] if value.startswith(" ") else value)
                self.line = ""
                self.skip_lf = char == "\r"
            else:
                self.line += char
                self.size += 1
                if self.size > 1 << 20:
                    raise InvalidBenchmark("SSE event exceeds one MiB")
        return events


def post_stream(endpoint: dict, prompt: dict, frozen: dict, *, timeout: float, save_text: bool = False) -> dict:
    """Measure one request. Errors retain timing and usage but never response/error text."""
    timeout = positive_seconds(timeout)
    start = time.monotonic()
    row = {"start_monotonic_s": start, "first_reasoning_monotonic_s": None,
           "first_content_monotonic_s": None, "end_monotonic_s": None, "usage": None,
           "status": "FAILED", "finish_reason": None, "server_decode_s": None,
           "server_prefill_s": None, "server_queue_s": None, "server_sampling_s": None,
           "server_native_generated_tokens": None}
    conn, response, timer = None, None, None
    try:
        parsed = urlsplit(endpoint["url"])
        cls = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
        conn = cls(parsed.hostname, parsed.port, timeout=timeout)
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
        if endpoint.get("api_key_env"):
            key = os.environ.get(endpoint["api_key_env"])
            if not key:
                raise InvalidBenchmark("configured API key environment variable is empty")
            headers["Authorization"] = "Bearer " + key
        settings = frozen["settings"]
        body = {**settings["sampling"], "messages": [{"role": "user", "content": prompt["text"]}],
                "max_tokens": OUTPUT_TOKENS, "stream": True, "n": 1,
                "stream_options": {"include_usage": True}, "chat_template_kwargs": settings["chat_template_kwargs"]}
        if endpoint.get("model"):
            body["model"] = endpoint["model"]
        expired = threading.Event()
        transport = [None]
        def expire():
            expired.set()
            if transport[0] is not None:
                try:
                    transport[0].shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
        # A whole-request deadline prevents drip-fed SSE/chunk headers from extending the run indefinitely.
        timer = threading.Timer(timeout, expire)
        timer.daemon = True
        timer.start()
        conn.connect()
        transport[0] = conn.sock
        if expired.is_set():
            raise TimeoutError
        conn.sock.settimeout(max(.000001, timeout - (time.monotonic() - start)))
        conn.request("POST", parsed.path, json.dumps(body, allow_nan=False).encode(), headers)
        response = conn.getresponse()
        row["http_status"] = response.status
        if response.status != 200:
            raise InvalidBenchmark("HTTP status is not 200")
        if response.getheader("Content-Type", "").split(";")[0].strip() != "text/event-stream":
            raise InvalidBenchmark("response is not an SSE stream")
        parser, done, text, reasoning = SSEParser(), False, [], []
        while not done:
            if expired.is_set() or time.monotonic() - start >= timeout:
                raise TimeoutError
            chunk = response.read1(4096)
            if not chunk:
                raise InvalidBenchmark("stream ended without DONE")
            observed = time.monotonic()
            for data in parser.feed(chunk):
                if data == "[DONE]":
                    done = True
                    break
                event = json.loads(data)
                if not isinstance(event, dict) or event.get("error"):
                    raise InvalidBenchmark("server emitted an error event")
                if event.get("usage") is not None:
                    row["usage"] = event["usage"]
                timings = event.get("timings")
                if isinstance(timings, dict):
                    if timings.get("native_generated_n") is not None:
                        count = timings["native_generated_n"]
                        if type(count) is not int or count < 0:
                            raise InvalidBenchmark("invalid provided native generated count")
                        row["server_native_generated_tokens"] = count
                    for source, target in (("predicted_ms", "server_decode_s"),
                                           ("prompt_ms", "server_prefill_s"),
                                           ("queue_ms", "server_queue_s"),
                                           ("sampling_ms", "server_sampling_s")):
                        if timings.get(source) is None:
                            continue
                        ms = timings[source]
                        if type(ms) not in (int, float) or not math.isfinite(ms) or ms < 0:
                            raise InvalidBenchmark("invalid provided phase timing")
                        row[target] = ms / 1000
                choices = event.get("choices", [])
                if not isinstance(choices, list) or len(choices) > 1:
                    raise InvalidBenchmark("expected a single completion")
                for choice in choices:
                    delta = choice.get("delta") or {}
                    for field, first, output in (("reasoning_content", "first_reasoning_monotonic_s", reasoning),
                                                 ("content", "first_content_monotonic_s", text)):
                        value = delta.get(field)
                        if value is not None and not isinstance(value, str):
                            raise InvalidBenchmark("stream delta must be text")
                        if value:
                            if row[first] is None:
                                row[first] = observed
                            if save_text:
                                output.append(value)
                    if choice.get("finish_reason") is not None:
                        row["finish_reason"] = choice["finish_reason"]
        usage = row["usage"]
        if (not isinstance(usage, dict) or type(usage.get("prompt_tokens")) is not int
                or usage["prompt_tokens"] != prompt["prompt_tokens"]
                or type(usage.get("completion_tokens")) is not int or not 0 <= usage["completion_tokens"] <= OUTPUT_TOKENS):
            raise InvalidBenchmark("missing usage or token counts do not match the native prompt/output budget")
        row["usage"] = {key: usage[key] for key in ("prompt_tokens", "completion_tokens")}
        if row["finish_reason"] not in ("stop", "length") or row["first_content_monotonic_s"] is None:
            raise InvalidBenchmark("no finished content response")
        row["status"] = "PASSED"
        if save_text:
            row.update(generated_text="".join(text), reasoning_text="".join(reasoning))
    except (OSError, ValueError, KeyError, TypeError, AttributeError, http.client.HTTPException) as exc:
        row["error"] = ("timeout" if isinstance(exc, TimeoutError) else type(exc).__name__)
        usage = row["usage"]
        row["usage"] = ({k: usage[k] for k in ("prompt_tokens", "completion_tokens")
                         if type(usage.get(k)) is int and usage[k] >= 0} if isinstance(usage, dict) else None)
    finally:
        row["end_monotonic_s"] = time.monotonic()
        if timer is not None:
            timer.cancel()
        if response is not None:
            response.close()
        if conn is not None:
            conn.close()
    row["wall_s"] = row["end_monotonic_s"] - start
    first = row["first_content_monotonic_s"]
    row["ttft_s"] = None if first is None else first - start
    reason = row["first_reasoning_monotonic_s"]
    row["first_reasoning_s"] = None if reason is None else reason - start
    row["client_output_window_s"] = None if first is None else row["end_monotonic_s"] - first
    return row


def quantile(values: list[float], fraction: float) -> float | None:
    """Nearest rank, with p in (0, 1]."""
    return sorted(values)[max(0, math.ceil(fraction * len(values)) - 1)] if values else None


def execute_plan(plan: dict, *, timeout: float, save_text: bool = False, on_job_complete=None) -> list[dict]:
    positive_seconds(timeout)
    mapping = {e["name"]: e for e in plan["endpoints"]}
    records = []
    for job in plan["jobs"]:
        record = dict(job)
        if job["status"] == "NOT_RUN":
            records.append(record)
            if on_job_complete is not None:
                on_job_complete(records)
            continue
        if type(job["concurrency"]) is not int or job["concurrency"] not in CONCURRENCY:
            raise InvalidBenchmark("concurrency must be 1, 4 or 8")
        endpoint, prompt = mapping[job["endpoint"]], plan["prompts"][job["prompt_tokens"]]
        start = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=job["concurrency"]) as pool:
            futures = [pool.submit(post_stream, endpoint, prompt, plan["frozen"], timeout=timeout, save_text=save_text)
                       for _ in range(job["concurrency"])]
            rows = [dict(future.result(), request_index=i) for i, future in enumerate(futures)]
        end = time.monotonic()
        ok = all(row["status"] == "PASSED" for row in rows)
        total = sum(row["usage"]["completion_tokens"] for row in rows) if ok else None
        record.update(status="PASSED" if ok else "FAILED", requests=rows,
                      start_monotonic_s=start, end_monotonic_s=end, wall_s=end - start,
                      completed_tokens=total, aggregate_tokens_s=total / (end - start) if ok and end > start else None,
                      ttft_p50_s=quantile([r["ttft_s"] for r in rows if r["status"] == "PASSED"], .5),
                      ttft_p95_s=quantile([r["ttft_s"] for r in rows if r["status"] == "PASSED"], .95),
                      request_wall_p50_s=quantile([r["wall_s"] for r in rows if r["status"] == "PASSED"], .5),
                      request_wall_p95_s=quantile([r["wall_s"] for r in rows if r["status"] == "PASSED"], .95))
        records.append(record)
        if on_job_complete is not None:
            on_job_complete(records)
    return records


def public_plan(plan: dict) -> dict:
    """Store identities/hashes and matrix, without prompt bodies, tokens or authentication details."""
    return {key: copy.deepcopy(plan[key]) for key in ("schema_version", "frozen", "files_sha256", "jobs")}


def write_checkpoint(path: Path, report: dict, *, initial: bool = False) -> None:
    """Atomically replace an owned checkpoint, or create the first one without clobbering."""
    payload = (json.dumps(report, indent=2, allow_nan=False) + "\n").encode()
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if initial:
            # Linking publishes the complete file and fails if another file appeared meanwhile.
            os.link(temporary, path)
        else:
            os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--execute", action="store_true", help="send the planned HTTP requests")
    ap.add_argument("--timeout", type=float, default=3600)
    ap.add_argument("--save-text", action="store_true", help="include generated content and reasoning in records")
    ap.add_argument("--checkpoint", type=Path, help="atomically save each completed job during execution")
    ap.add_argument("--telemetry", type=Path, help="preprovided external JSONL artifact; never contacts hosts")
    args = ap.parse_args(argv)
    try:
        positive_seconds(args.timeout)
        if args.checkpoint and not args.execute:
            raise InvalidBenchmark("--checkpoint requires --execute")
        if args.checkpoint:
            args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            if args.checkpoint.exists():
                raise InvalidBenchmark("checkpoint path already exists")
        plan = load_plan(args.config)
        report = public_plan(plan)
        report.update(executed=args.execute, quantile="nearest_rank_ceil_p_times_n_minus_one",
                      timing="client queue-inclusive monotonic; no inferred engine prefill/decode rate")
        if args.telemetry:
            raw = args.telemetry.read_bytes()
            for line in raw.splitlines():
                if not isinstance(json.loads(line), dict):
                    raise InvalidBenchmark("telemetry JSONL rows must be objects")
            report["external_telemetry"] = {"path": str(args.telemetry.resolve()), "sha256": digest(raw),
                                             "rows": len(raw.splitlines()), "source": "operator-provided"}
        if args.execute and args.checkpoint:
            checkpoint = {key: copy.deepcopy(report[key]) for key in ("schema_version", "frozen", "files_sha256", "jobs")}
            checkpoint.update(executed=True, status="incomplete", records=[])
            write_checkpoint(args.checkpoint, checkpoint, initial=True)

            def save_progress(records):
                checkpoint["records"] = copy.deepcopy(records)
                write_checkpoint(args.checkpoint, checkpoint)

            report["records"] = execute_plan(plan, timeout=args.timeout, save_text=args.save_text,
                                             on_job_complete=save_progress)
            checkpoint["status"] = "completed"
            write_checkpoint(args.checkpoint, checkpoint)
        else:
            report["records"] = execute_plan(plan, timeout=args.timeout, save_text=args.save_text) if args.execute else plan["jobs"]
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        return 1 if any(row["status"] == "FAILED" for row in report["records"]) else 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        # Do not echo endpoint response text, authentication, or prompt content.
        print(f"benchmark input failed ({type(exc).__name__})", file=__import__("sys").stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
