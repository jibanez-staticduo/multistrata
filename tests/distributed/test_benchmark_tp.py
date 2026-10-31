"""Synthetic CPU fixtures for the matched benchmark; no hosts, model, or performance claim."""
from __future__ import annotations

import http.server
import json
from pathlib import Path
import sys
import threading
import time

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "distributed"))
import benchmark_tp as bench


@pytest.fixture
def inputs(tmp_path):
    snap = {"identity": {"revision": "a" * 40, "pack_sha256": "b" * 64, "tokenizer_sha256": "c" * 64,
                         "chat_template_sha256": "d" * 64},
            "settings": {"context": 131072, "sampling": {"temperature": 0, "seed": 7},
                         "chat_template_kwargs": {"enable_thinking": False}},
            "policy": {"prefix_cache": "off"}}
    endpoints = []
    for name in bench.ENDPOINTS:
        (tmp_path / f"{name}.json").write_text(json.dumps(snap))
        endpoints.append({"name": name, "supported": True, "snapshot": f"{name}.json",
                          "url": "http://127.0.0.1:1/v1/chat/completions",
                          "controls": {"prompt_tokens": list(bench.PROMPTS), "concurrency": [1, 4, 8], "output_tokens": [256]}})
    artifact = {"tokenizer": {"kind": "strata_pack", **{k: v for k, v in snap["identity"].items() if k != "revision"}},
                "prompts": [{"prompt_tokens": n, "text": f"native prompt {n}", "token_ids": [17] * n}
                            for n in bench.PROMPTS]}
    (tmp_path / "prompts.json").write_text(json.dumps(artifact))
    cfg = {"schema_version": 1, "prompts": "prompts.json", "endpoints": endpoints}
    path = tmp_path / "benchmark.json"
    path.write_text(json.dumps(cfg))
    return path


def edit(path, change):
    value = json.loads(path.read_text())
    change(value)
    path.write_text(json.dumps(value))


def test_three_rounds_pair_every_case_with_rotating_endpoint_order(inputs):
    plan = bench.load_plan(inputs)
    assert len(plan["jobs"]) == 81
    for case in [(p, c) for p in bench.PROMPTS for c in bench.CONCURRENCY]:
        jobs = [j for j in plan["jobs"] if (j["prompt_tokens"], j["concurrency"]) == case]
        assert [j["endpoint"] for j in jobs] == ["tp1", "tp2", "tp3", "tp2", "tp3", "tp1", "tp3", "tp1", "tp2"]
        assert [j["round"] for j in jobs] == [1] * 3 + [2] * 3 + [3] * 3
        assert all(j["max_tokens"] == 256 for j in jobs)


def test_dry_run_never_contacts_endpoints_and_omits_prompt_text(inputs, tmp_path, monkeypatch):
    monkeypatch.setattr(bench, "post_stream", lambda *a, **kw: pytest.fail("dry-run sent a request"))
    out = tmp_path / "plan.json"
    assert bench.main(["--config", str(inputs), "--out", str(out)]) == 0
    raw = out.read_text()
    assert not json.loads(raw)["executed"]
    assert "native prompt" not in raw and "token_ids" not in raw
    assert len(json.loads(raw)["files_sha256"]) == 5


def test_unsupported_endpoint_and_c4_c8_are_not_run(inputs):
    def change(cfg):
        cfg["endpoints"][0]["controls"]["concurrency"] = [1]
        cfg["endpoints"][1] = {"name": "tp2", "supported": False, "reason": "pack not supported"}
    edit(inputs, change)
    plan = bench.load_plan(inputs)
    assert all(j["status"] == "NOT_RUN" for j in plan["jobs"] if j["endpoint"] == "tp2")
    assert all(j["status"] == "NOT_RUN" for j in plan["jobs"] if j["endpoint"] == "tp1" and j["concurrency"] > 1)


def test_unsupported_output_budget_has_no_fake_measurement(inputs):
    edit(inputs, lambda cfg: cfg["endpoints"][0]["controls"].update(output_tokens=[]))
    plan = bench.load_plan(inputs)
    jobs = [j for j in plan["jobs"] if j["endpoint"] == "tp1"]
    assert all(j["status"] == "NOT_RUN" for j in jobs)
    assert all("aggregate_tokens_s" not in j for j in jobs)


def test_context_limit_marks_long_prompt_not_run(inputs):
    for name in bench.ENDPOINTS:
        edit(inputs.parent / f"{name}.json", lambda snap: snap["settings"].update(context=32768))
    plan = bench.load_plan(inputs)
    assert all(j["status"] == "NOT_RUN" for j in plan["jobs"] if j["prompt_tokens"] == 65536)


def test_execution_still_enforces_concurrency_cap(inputs):
    plan = bench.load_plan(inputs)
    plan["jobs"] = [dict(plan["jobs"][0], concurrency=16)]
    with pytest.raises(bench.InvalidBenchmark):
        bench.execute_plan(plan, timeout=.1)


@pytest.mark.parametrize("field,value", [("pack_sha256", "e" * 64), ("revision", "other"),
                                         ("tokenizer_sha256", "f" * 64)])
def test_unmatched_sources_fail_before_any_network(inputs, field, value):
    edit(inputs.parent / "tp2.json", lambda snap: snap["identity"].update({field: value}))
    with pytest.raises(bench.InvalidBenchmark, match="unmatched"):
        bench.load_plan(inputs)


@pytest.mark.parametrize("section", ["settings", "policy"])
def test_unmatched_settings_or_cache_policy_fail(inputs, section):
    def change(snap):
        if section == "settings":
            snap[section]["sampling"]["temperature"] = .7
        else:
            snap[section]["prefix_cache"] = "on"
    edit(inputs.parent / "tp2.json", change)
    with pytest.raises(bench.InvalidBenchmark):
        bench.load_plan(inputs)


@pytest.mark.parametrize("change", [lambda s: s.pop("identity"), lambda s: s["identity"].pop("revision"),
                                     lambda s: s["settings"].pop("sampling"), lambda s: s.pop("policy")])
def test_missing_frozen_fields_fail(inputs, change):
    edit(inputs.parent / "tp1.json", change)
    with pytest.raises((bench.InvalidBenchmark, KeyError)):
        bench.load_plan(inputs)


def test_cannot_claim_unsupported_c16(inputs):
    edit(inputs, lambda cfg: cfg["endpoints"][0]["controls"].update(concurrency=[1, 4, 8, 16]))
    with pytest.raises(bench.InvalidBenchmark, match="cap is 8"):
        bench.load_plan(inputs)


@pytest.mark.parametrize("change", [lambda a: a["tokenizer"].update(kind="foreign"),
                                     lambda a: a["tokenizer"].update(pack_sha256="f" * 64),
                                     lambda a: a["prompts"][0]["token_ids"].pop(),
                                     lambda a: a["prompts"][0].update(text="")])
def test_native_prompt_evidence_is_required(inputs, change):
    edit(inputs.parent / "prompts.json", change)
    with pytest.raises(bench.InvalidBenchmark):
        bench.load_plan(inputs)


def test_sse_parser_partial_utf8_crlf_comments_multiline_and_incomplete_event():
    raw = '\ufeff: heartbeat\r\nevent: message\r\ndata: {"text":\r\ndata: "caf\u00e9"}\r\n\r\ndata: [DONE]\n\ndata: incomplete'.encode()
    parser, events = bench.SSEParser(), []
    for byte in raw:
        events.extend(parser.feed(bytes([byte])))
    assert events == ['{"text":\n"caf\u00e9"}', "[DONE]"]


def sse(value):
    return ("data: " + (value if isinstance(value, str) else json.dumps(value)) + "\r\n\r\n").encode()


@pytest.fixture
def endpoint():
    state = {"mode": "normal", "bodies": [], "active": 0, "peak": 0}
    lock = threading.Lock()
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            with lock:
                state["bodies"].append(body)
                state["active"] += 1
                state["peak"] = max(state["peak"], state["active"])
            try:
                if state["mode"] == "http_error":
                    self.send_response(503)
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Connection", "close")
                self.end_headers()
                if state["mode"] == "timeout":
                    time.sleep(.2)
                    return
                pieces = [sse({"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}]}),
                          b": ping\r\n\r\n", sse({"choices": [{"delta": {"reasoning_content": "reason-secret"}}]}),
                          sse({"choices": [{"delta": {"content": "answer-secret"}}]}),
                          sse({"choices": [{"delta": {}, "finish_reason": "length"}]})]
                for i, piece in enumerate(pieces):
                    self.wfile.write(piece)
                    self.wfile.flush()
                    if i == 2:
                        time.sleep(.02)
                if state["mode"] == "stream_error":
                    self.wfile.write(sse({"error": {"message": "secret-error"}}))
                    return
                if state["mode"] != "no_usage":
                    self.wfile.write(sse({"choices": [], "usage": {"prompt_tokens": 4097 if state["mode"] == "wrong_prompt" else 4096,
                                                                 "completion_tokens": 256},
                                          "timings": {"predicted_ms": 12, "prompt_ms": 34,
                                                      "queue_ms": 5, "sampling_ms": 2,
                                                      "native_generated_n": 257}}))
                if state["mode"] != "no_done":
                    self.wfile.write(sse("[DONE]"))
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                with lock:
                    state["active"] -= 1

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state["endpoint"] = {"url": f"http://127.0.0.1:{server.server_port}/v1/chat/completions"}
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()


def post(endpoint, inputs, **kwargs):
    plan = bench.load_plan(inputs)
    return bench.post_stream(endpoint["endpoint"], plan["prompts"][4096], plan["frozen"], timeout=2, **kwargs)


def test_first_actual_content_not_role_or_reasoning_defines_ttft(endpoint, inputs):
    row = post(endpoint, inputs)
    assert row["status"] == "PASSED"
    assert row["start_monotonic_s"] < row["first_reasoning_monotonic_s"] < row["first_content_monotonic_s"] < row["end_monotonic_s"]
    assert row["ttft_s"] >= .015
    assert row["server_decode_s"] == .012
    assert row["server_prefill_s"] == .034
    assert row["server_queue_s"] == .005
    assert row["server_sampling_s"] == .002
    assert row["server_native_generated_tokens"] == 257
    assert "answer-secret" not in json.dumps(row) and "reason-secret" not in json.dumps(row)
    assert row["usage"] == {"prompt_tokens": 4096, "completion_tokens": 256}
    assert endpoint["bodies"][0]["stream_options"] == {"include_usage": True}


def test_optional_output_text_is_explicit(endpoint, inputs):
    row = post(endpoint, inputs, save_text=True)
    assert row["generated_text"] == "answer-secret" and row["reasoning_text"] == "reason-secret"


@pytest.mark.parametrize("mode", ["no_usage", "wrong_prompt", "no_done", "stream_error", "http_error"])
def test_invalid_or_incomplete_measurement_fails_closed(endpoint, inputs, mode):
    endpoint["mode"] = mode
    row = post(endpoint, inputs)
    assert row["status"] == "FAILED"
    assert row["end_monotonic_s"] >= row["start_monotonic_s"]
    assert "secret" not in json.dumps(row)


def test_near_zero_timeout_is_finite_and_records_failure(endpoint, inputs):
    plan = bench.load_plan(inputs)
    endpoint["mode"] = "timeout"
    started = time.monotonic()
    row = bench.post_stream(endpoint["endpoint"], plan["prompts"][4096], plan["frozen"], timeout=.00001)
    assert row["status"] == "FAILED" and time.monotonic() - started < 1


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_invalid_timeout_rejected(value):
    with pytest.raises(bench.InvalidBenchmark):
        bench.positive_seconds(value)


def test_c8_aggregate_uses_all_finished_tokens_and_whole_round_wall(endpoint, inputs):
    plan = bench.load_plan(inputs)
    plan["endpoints"][0].update(endpoint["endpoint"])
    plan["jobs"] = [{"endpoint": "tp1", "prompt_tokens": 4096, "concurrency": 8, "max_tokens": 256,
                     "round": 1, "status": "PLANNED", "reason": None}]
    result = bench.execute_plan(plan, timeout=3)[0]
    assert result["status"] == "PASSED" and len(result["requests"]) == 8
    assert result["completed_tokens"] == 2048
    assert result["aggregate_tokens_s"] == pytest.approx(2048 / result["wall_s"])
    assert 1 < endpoint["peak"] <= 8
    assert result["ttft_p95_s"] == max(r["ttft_s"] for r in result["requests"])


def test_failed_round_has_no_aggregate_rate(endpoint, inputs):
    endpoint["mode"] = "no_usage"
    plan = bench.load_plan(inputs)
    plan["endpoints"][0].update(endpoint["endpoint"])
    plan["jobs"] = [dict(plan["jobs"][0])]
    row = bench.execute_plan(plan, timeout=2)[0]
    assert row["status"] == "FAILED" and row["aggregate_tokens_s"] is None


def test_quantile_definition():
    assert bench.quantile([4, 1, 2, 3], .5) == 2
    assert bench.quantile([4, 1, 2, 3], .95) == 4
    assert bench.quantile([], .95) is None


def test_execute_plan_checkpoints_every_terminal_job_including_not_run(inputs, monkeypatch):
    plan = bench.load_plan(inputs)
    plan["jobs"] = [
        dict(plan["jobs"][0], status="NOT_RUN", reason="unsupported"),
        dict(plan["jobs"][1], status="PLANNED"),
    ]
    plan["endpoints"][0].update(url="http://127.0.0.1:1/v1/chat/completions")
    monkeypatch.setattr(bench, "post_stream", lambda *a, **kw: {
        "status": "PASSED", "usage": {"completion_tokens": 256}, "ttft_s": .01, "wall_s": .1,
    })
    snapshots = []

    records = bench.execute_plan(plan, timeout=1, on_job_complete=lambda rows: snapshots.append([dict(r) for r in rows]))

    assert [row["status"] for row in records] == ["NOT_RUN", "PASSED"]
    assert [[row["status"] for row in rows] for rows in snapshots] == [["NOT_RUN"], ["NOT_RUN", "PASSED"]]


def test_checkpoint_survives_interruption_without_saving_stream_error(inputs, tmp_path, monkeypatch):
    calls = 0

    def post_stream(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("private response secret")
        return {
            "status": "PASSED", "usage": {"completion_tokens": 256}, "ttft_s": .01, "wall_s": .1,
        }

    monkeypatch.setattr(bench, "post_stream", post_stream)
    checkpoint, out = tmp_path / "checkpoint.json", tmp_path / "report.json"
    with pytest.raises(RuntimeError, match="private response secret"):
        bench.main(["--config", str(inputs), "--out", str(out), "--execute", "--checkpoint", str(checkpoint)])

    saved = json.loads(checkpoint.read_text())
    plan = bench.load_plan(inputs)
    assert saved["status"] == "incomplete"
    assert len(saved["records"]) == 1 and saved["records"][0]["status"] == "PASSED"
    assert saved["frozen"] == plan["frozen"]
    assert saved["files_sha256"] == plan["files_sha256"]
    assert "private response secret" not in checkpoint.read_text()
    assert not out.exists()


def test_existing_checkpoint_is_refused_before_execution(inputs, tmp_path, monkeypatch):
    checkpoint, out = tmp_path / "checkpoint.json", tmp_path / "report.json"
    checkpoint.write_text("keep this file")
    monkeypatch.setattr(bench, "post_stream", lambda *a, **kw: pytest.fail("must refuse before sending"))

    assert bench.main(["--config", str(inputs), "--out", str(out), "--execute", "--checkpoint", str(checkpoint)]) == 2
    assert checkpoint.read_text() == "keep this file"


def test_checkpoint_is_completed_after_all_jobs_and_default_run_has_no_checkpoint(inputs, tmp_path, monkeypatch):
    def post_stream(*args, **kwargs):
        return {"status": "PASSED", "usage": {"completion_tokens": 256}, "ttft_s": .01, "wall_s": .1}

    monkeypatch.setattr(bench, "post_stream", post_stream)
    checkpoint, out = tmp_path / "checkpoint.json", tmp_path / "report.json"
    assert bench.main(["--config", str(inputs), "--out", str(out), "--execute", "--checkpoint", str(checkpoint)]) == 0
    saved = json.loads(checkpoint.read_text())
    assert saved["status"] == "completed" and len(saved["records"]) == 81

    ordinary = tmp_path / "ordinary.json"
    assert bench.main(["--config", str(inputs), "--out", str(ordinary)]) == 0
    assert ordinary.exists() and not (tmp_path / "ordinary.checkpoint.json").exists()
