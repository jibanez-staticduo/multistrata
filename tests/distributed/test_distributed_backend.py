"""CPU-only JSONL boundary tests; the fake coordinator proves no native TP3 inference."""
from __future__ import annotations

import contextlib
from concurrent.futures import ThreadPoolExecutor
import http.client
import io
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from serve.distributed_backend import DistributedEngine, normalize_sampling, validate_config
from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, EngineDied, Service, main, serve


ROOT = Path(__file__).resolve().parents[2]

# A protocol fixture, deliberately separate from the native teacher-forced worker.
COORDINATOR = r'''
import json, sys, threading, time, signal
mode = sys.argv[1]
if mode == 'ignore_term':
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
lock = threading.Lock()
def emit(kind, **kw):
    with lock:
        print(json.dumps(dict(type=kind, alive_ranks=[0,1,2], **kw)), flush=True)
if mode == 'legacy':
    print('READY 3', flush=True)
elif mode != 'silent':
    emit('ready', protocol='strata-distributed-v1', mode='teacher-forced' if mode == 'teacher' else 'autoregressive',
         world=3, max_context=32768)
if mode == 'stderr':
    sys.stderr.write('private-api-key\n' * 100000)
    sys.stderr.flush()
def beat():
    while True:
        time.sleep(.03)
        emit('health', state='ready')
if mode not in ('stale', 'silent', 'legacy', 'teacher', 'ignore_term'):
    threading.Thread(target=beat, daemon=True).start()
turn = 0
cancelled = False
for line in sys.stdin:
    msg = json.loads(line)
    if msg['type'] == 'shutdown':
        break
    rid = msg['request_id']
    if msg['type'] == 'cancel':
        if mode == 'no_cancel_ack':
            continue
        time.sleep(.08)
        cancelled = True
        emit('cancelled', request_id=rid, state='ready')
        continue
    turn += 1
    if turn > 1 and mode == 'cancel' and not cancelled:
        emit('error', request_id=rid)
        continue
    if mode in ('cancel', 'no_cancel_ack') and turn == 1:
        emit('token', request_id=rid, id=65, committed=True)
        continue
    if mode == 'death':
        sys.exit(1)
    if mode == 'missing_rank':
        print(json.dumps(dict(type='token', request_id=rid, id=65, committed=True, alive_ranks=[0,1])), flush=True)
        continue
    if mode == 'hang':
        continue
    if mode == 'overrun':
        for i in range(msg['max_new'] + 1):
            emit('token', request_id=rid, id=65, committed=True)
        continue
    if mode == 'wrong_request':
        emit('token', request_id='wrong', id=65, committed=True)
        continue
    if mode == 'done_error':
        emit('done', request_id=rid, state='failed')
        continue
    if mode == 'echo':
        values = msg['ids']
    else:
        values = list(b'</think>\n\nHello.') + [257]
    for token in values[:msg['max_new']]:
        emit('token', request_id=rid, id=token, committed=mode != 'uncommitted')
    emit('done', request_id=rid, state='ready')
'''


def config(mode="normal", **options):
    return {"distributed": {"command": [sys.executable, "-u", "-c", COORDINATOR, mode],
                            "ready_timeout": 2, "health_timeout": 1, "process_deadline": 2,
                            "cancel_timeout": .4, "shutdown_timeout": .3, **options}}


@pytest.fixture
def make_engine():
    engines = []

    def make(mode="normal", **options):
        engine = DistributedEngine(config(mode, **options))
        engines.append(engine)
        return engine

    yield make
    for engine in engines:
        engine.close()


@pytest.mark.parametrize("mode", ["legacy", "teacher", "silent"])
def test_requires_autoregressive_handshake(mode):
    started = time.monotonic()
    with pytest.raises(EngineDied):
        DistributedEngine(config(mode, ready_timeout=.2))
    assert time.monotonic() - started < 2


def test_token_ids_and_sampling_use_existing_boundary(make_engine):
    engine = make_engine("echo")
    assert list(engine.generate([17, 42, 83], 3, {"temperature": .3, "seed": 7}, threading.Event())) == [17, 42, 83]
    assert engine.batch == 0
    assert engine.health()["alive_ranks"] == [0, 1, 2]


@pytest.mark.parametrize("sampling", [{"temperature": float("nan")}, {"temperature": True}, {"top_p": 0},
                                     {"min_p": 2}, {"repetition_penalty": 0}, {"seed": 2**64}, {"top_p": 1e-300},
                                     {"logprobs": True}, {"strata_tune": {}}, {"logit_bias": {"9999999999": 1}}])
def test_invalid_user_sampling_never_transmitted_and_group_survives(make_engine, sampling):
    engine = make_engine("echo")
    sent, original = [], engine._send
    def record(message, timeout):
        sent.append(message)
        return original(message, timeout)
    engine._send = record
    with pytest.raises(ValueError):
        list(engine.generate([17], 1, sampling, threading.Event()))
    assert sent == []
    assert engine._request is None and engine._state == "ready"
    assert list(engine.generate([17], 1, {}, threading.Event())) == [17]


def test_full_api_sampling_is_reduced_to_normalized_native_fields():
    value = normalize_sampling({"temperature": None, "top_k": 0, "top_p": None, "seed": -1,
                                "repetition_penalty": 1.1, "penalty_last_n": 0,
                                "messages": [{"role": "user", "content": "hi"}], "stream": True,
                                "unknown_future_format": {"nested": [None]}, "stop": ["done"]})
    assert value == {"top_k": 64, "repetition_penalty": 1.1, "penalty_last_n": 0}
    assert normalize_sampling({"top_k": -1, "seed": 0, "penalty_last_n": -1}) == {}
    assert normalize_sampling({"seed": str(2**64 - 1)}) == {"seed": 2**64 - 1}
    assert normalize_sampling({"temperature": "0.7", "seed": "42", "repeat_penalty": 1.2,
                               "repeat_last_n": 3}) == {"temperature": .7, "seed": 42,
                                                        "repetition_penalty": 1.2, "penalty_last_n": 3}


@pytest.mark.parametrize("ids,max_new", [([], 1), ([True], 1), ([-1], 1), ([2**31], 1), ([1], True),
                                        ([1], -1), ([1] * 32768, 1), ([1], "1")])
def test_invalid_token_request_never_transmitted_and_group_survives(make_engine, ids, max_new):
    engine = make_engine("echo")
    sent, original = [], engine._send
    def record(message, timeout):
        sent.append(message)
        return original(message, timeout)
    engine._send = record
    with pytest.raises(ValueError):
        list(engine.generate(ids, max_new, {}, threading.Event()))
    assert sent == []
    assert engine._request is None and engine.alive()
    assert list(engine.generate([17], 1, {}, threading.Event())) == [17]


def test_pack_vocabulary_bounds_are_validated_before_send(tmp_path):
    (tmp_path / "vocab.json").write_text(json.dumps({"a": 0, "b": 1}))
    cfg = config("echo")
    cfg["tokenizer"] = str(tmp_path)
    engine = DistributedEngine(cfg)
    try:
        with pytest.raises(ValueError):
            list(engine.generate([2], 1, {}, threading.Event()))
        with pytest.raises(ValueError):
            list(engine.generate([1], 1, {"logit_bias": {"2": 1}}, threading.Event()))
        assert engine._request is None
        assert list(engine.generate([1], 1, {}, threading.Event())) == [1]
    finally:
        engine.close()


def test_fast_coordinator_applies_backpressure_without_dropping_tokens(make_engine):
    engine = make_engine("echo", health_timeout=.12)
    ids = list(range(1000))
    stream = engine.generate(ids, len(ids), {}, threading.Event())
    first = next(stream)
    time.sleep(.25)
    assert engine.health()["ready"]
    assert [first, *stream] == ids


@pytest.mark.skipif(os.name == "nt", reason="SIGTERM ignore is a POSIX behavior")
def test_idle_health_failure_kills_coordinator_ignoring_sigterm(make_engine):
    engine = make_engine("ignore_term", health_timeout=.08, shutdown_timeout=.1)
    time.sleep(.12)
    assert not engine.health()["ready"]
    deadline = time.monotonic() + .6
    while engine.proc.poll() is None and time.monotonic() < deadline:
        time.sleep(.01)
    assert engine.proc.poll() is not None


@pytest.mark.parametrize("arch", ["deepseek41", "llama", "unknown"])
def test_non_qwen_family_rejected_before_process_start(arch):
    cfg = config()
    cfg["arch"] = arch
    with pytest.raises(ValueError, match="Qwen"):
        validate_config(cfg)


def test_real_module_entrypoint_handles_adapter_errors(tmp_path):
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "vocab.json").write_text(json.dumps({"a": 0, "aa": 1}))
    (tokenizer / "merges.txt").write_text("a a")
    (tokenizer / "token_type.json").write_text("[1,1]")
    cfg = {"tokenizer": str(tokenizer), "distributed": {"command": ["/missing/coordinator"]}}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg))
    env = dict(os.environ, PYTHONPATH=str(tmp_path) + os.pathsep + str(ROOT))
    run = subprocess.run([sys.executable, "-m", "serve.server", "--engine", "distributed", "--config", str(path),
                          "--port", "0"], cwd=ROOT, env=env, capture_output=True, text=True, timeout=5)
    assert run.returncode == 2
    assert "Traceback" not in run.stderr
    assert "distributed coordinator could not start" in run.stderr


def test_tokens_over_budget_fail_whole_group(make_engine):
    engine = make_engine("overrun")
    stream = engine.generate([1], 1, {}, threading.Event())
    with pytest.raises(EngineDied):
        assert len(list(stream)) <= 1
    assert not engine.alive()


@pytest.mark.parametrize("mode", ["uncommitted", "missing_rank", "wrong_request", "done_error", "death"])
def test_invalid_group_output_is_never_exposed(make_engine, mode):
    engine = make_engine(mode)
    values = []
    with pytest.raises(EngineDied):
        values.extend(engine.generate([1], 20, {}, threading.Event()))
    assert values == []
    assert engine.health()["ready"] is False
    assert engine.proc.poll() is not None


def test_health_lease_expires_even_when_process_is_live(make_engine):
    engine = make_engine("stale", health_timeout=.08)
    assert engine.alive()
    time.sleep(.12)
    assert engine.health()["status"] == "unavailable"
    assert not engine.alive()


def test_cancel_ack_is_required_before_next_turn(make_engine):
    engine = make_engine("cancel")
    cancel = threading.Event()
    stream = engine.generate([1], 20, {}, cancel)
    assert next(stream) == 65
    cancel.set()
    started = time.monotonic()
    assert list(stream) == []
    assert time.monotonic() - started >= .05
    assert list(engine.generate([1], 20, {}, threading.Event()))


def test_generator_close_also_waits_for_cancel_ack(make_engine):
    engine = make_engine("cancel")
    stream = engine.generate([1], 20, {}, threading.Event())
    assert next(stream) == 65
    stream.close()
    assert list(engine.generate([1], 20, {}, threading.Event()))


def test_missing_cancel_ack_ends_whole_group(make_engine):
    engine = make_engine("no_cancel_ack", cancel_timeout=.1)
    stream = engine.generate([1], 20, {}, threading.Event())
    assert next(stream) == 65
    with pytest.raises(EngineDied):
        stream.close()
    assert not engine.alive()
    with pytest.raises(EngineDied):
        list(engine.generate([1], 20, {}, threading.Event()))


def test_request_deadline_not_extended_by_health(make_engine):
    engine = make_engine("hang", process_deadline=.15)
    started = time.monotonic()
    with pytest.raises(EngineDied):
        list(engine.generate([1], 20, {}, threading.Event()))
    assert time.monotonic() - started < 2
    assert engine.proc.poll() is not None


def test_zero_budget_and_precancelled_requests_do_not_submit(make_engine):
    engine = make_engine("death")
    assert list(engine.generate([1], 0, {}, threading.Event())) == []
    cancel = threading.Event()
    cancel.set()
    assert list(engine.generate([1], 1, {}, cancel)) == []
    assert engine.alive()


def test_stderr_does_not_block_and_shutdown_is_bounded(make_engine, capfd):
    started = time.monotonic()
    engine = make_engine("stderr")
    assert list(engine.generate([1], 20, {}, threading.Event()))
    engine.close()
    assert engine.proc.poll() is not None
    assert time.monotonic() - started < 3
    captured = capfd.readouterr()
    assert "private-api-key" not in captured.out + captured.err


def test_spawn_errors_never_echo_argv():
    cfg = {"distributed": {"command": ["/missing/private-api-key", "private-argument"]}}
    with pytest.raises(EngineDied) as exc:
        DistributedEngine(cfg)
    assert "private" not in str(exc.value)


@pytest.mark.parametrize("key,value", [("parallel", 2), ("replicas", 3), ("vision", {"exe": "x"}),
                                       ("split_serve", {"decode_gpu": 1}), ("lazy_load", True)])
def test_incompatible_profiles_are_rejected(key, value):
    cfg = config()
    cfg[key] = value
    with pytest.raises(ValueError):
        validate_config(cfg)


@pytest.mark.parametrize("value", [0, -1, True, float("nan"), float("inf"), "5"])
def test_deadlines_must_be_positive_finite(value):
    with pytest.raises(ValueError):
        validate_config(config(health_timeout=value))


def test_cli_requires_pack_tokenizer_before_starting(tmp_path):
    cfg = config()
    cfg["tokenizer"] = str(tmp_path / "missing")
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg))
    with pytest.raises(SystemExit) as exc:
        main(["--engine", "distributed", "--config", str(path), "--port", "0"])
    assert exc.value.code == 2


@contextlib.contextmanager
def http_service(engine):
    svc = Service(engine, ByteTokenizer(), ChatTemplate(ROOT / "serve/chat_template.jinja"))
    server = serve(svc, port=0)
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()


def request(port, path, body=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        conn.request("GET" if body is None else "POST", path,
                     body=None if body is None else json.dumps(body), headers={"Content-Type": "application/json"})
        response = conn.getresponse()
        return response.status, response.read().decode()
    finally:
        conn.close()


@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/messages"])
@pytest.mark.parametrize("stream", [False, True])
def test_existing_http_apis_parse_adapter_tokens(make_engine, path, stream):
    engine = make_engine()
    with http_service(engine) as port, contextlib.redirect_stdout(io.StringIO()):
        status, raw = request(port, path, {"model": "m", "messages": [{"role": "user", "content": "hi"}],
                                           "max_tokens": 100, "stream": stream})
    assert status == 200
    if stream:
        events = [json.loads(line[6:]) for line in raw.splitlines() if line.startswith("data: {")]
        if path.endswith("messages"):
            text = "".join(ev.get("delta", {}).get("text", "") for ev in events)
            assert any(ev["type"] == "message_stop" for ev in events)
        else:
            text = "".join(choice.get("delta", {}).get("content", "") for ev in events for choice in ev.get("choices", []))
            assert "data: [DONE]" in raw
    else:
        body = json.loads(raw)
        text = (body["content"][0]["text"] if path.endswith("messages") else body["choices"][0]["message"]["content"])
    assert text == "Hello."


def test_http_health_reports_group_failure_and_expired_lease(make_engine):
    engine = make_engine("stale", health_timeout=5)
    with http_service(engine) as port:
        status, body = request(port, "/health")
        assert status == 200 and json.loads(body)["ready"]
        with engine._lock:
            engine._last_health -= 6
        status, body = request(port, "/api/health")
        assert status == 503
        assert json.loads(body)["loaded"] is False


def test_http_failed_distributed_group_requires_explicit_restart(make_engine):
    engine = make_engine("death")
    with http_service(engine) as port:
        status, body = request(port, "/v1/chat/completions",
                               {"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1})
    assert status == 503
    message = json.loads(body)["error"]["message"]
    assert "restart the distributed group" in message
    assert "next request restarts" not in message


MULTISLOT_COORDINATOR = r'''
import json, sys, threading, time
slots, cohort, mode = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
output, state = threading.Lock(), threading.Lock()
active, workers = set(), {}
gate = threading.Event()
def emit(kind, **kw):
    with output:
        print(json.dumps(dict(type=kind, alive_ranks=[0,1,2], **kw)), flush=True)
emit('ready', protocol='strata-distributed-v1', mode='autoregressive', world=3,
     max_context=32768, slots=slots)
def beat():
    while True:
        time.sleep(.02)
        with state:
            busy = bool(active)
        emit('health', state='busy' if busy else 'ready')
threading.Thread(target=beat, daemon=True).start()
def generate(msg, stopped):
    rid = msg['request_id']
    try:
        while not gate.wait(.01):
            if stopped.is_set():
                return
        if msg['ids'][0] == 900:
            stopped.wait(2)
            return
        if mode == 'api':
            text = '</think>\n\nHello.'
            if msg['sampling'].get('seed') == 7:
                text = '</think>\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>'
            values = list(text.encode())
        else:
            values = msg['ids'][:msg['max_new']]
        if msg['ids'][0] == 999:
            values = [65] * (msg['max_new'] + 1)
        if msg['ids'][0] == 997:
            emit('token', request_id='unknown', id=65, committed=True)
            return
        for token in values:
            if stopped.wait(.003):
                return
            emit('token', request_id=rid, id=token, committed=True)
        emit('done', request_id=rid, state='ready')
        if msg['ids'][0] == 996:
            emit('done', request_id=rid, state='ready')
    finally:
        with state:
            active.discard(rid)
for line in sys.stdin:
    msg = json.loads(line)
    if msg['type'] == 'shutdown':
        break
    rid = msg['request_id']
    if msg['type'] == 'cancel':
        if rid in workers:
            stopped, worker = workers[rid]
            stopped.set()
            worker.join()
        emit('cancelled', request_id=rid, state='ready')
        continue
    if msg['ids'][0] == 998:
        emit('rejected', request_id=rid, state='busy')
        continue
    with state:
        active.add(rid)
        if len(active) > slots:
            emit('error')
        if len(active) >= cohort:
            gate.set()
    stopped = threading.Event()
    worker = threading.Thread(target=generate, args=(msg, stopped), daemon=True)
    workers[rid] = stopped, worker
    worker.start()
'''


@pytest.fixture
def make_multislot():
    engines = []

    def make(slots=4, cohort=2, mode="echo", requested=None, **options):
        cfg = config(**options)
        cfg["parallel"] = slots if requested is None else requested
        cfg["distributed"]["command"] = [sys.executable, "-u", "-c", MULTISLOT_COORDINATOR,
                                             str(slots), str(cohort), mode]
        engine = DistributedEngine(cfg)
        engines.append(engine)
        return engine

    yield make
    for engine in engines:
        engine.close()


@pytest.mark.parametrize("value", [0, 2, 3, 16, True, 4.0, "4"])
def test_distributed_parallel_only_allows_native_capacities(value):
    cfg = config(parallel=value)
    with pytest.raises(ValueError, match="parallel"):
        validate_config(cfg)


@pytest.mark.parametrize("slots,requested", [(1, 4), (4, 8), (8, 4)])
def test_multislot_handshake_requires_exact_capacity(make_multislot, slots, requested):
    with pytest.raises(EngineDied):
        make_multislot(slots, requested=requested)


def test_legacy_handshake_cannot_enable_concurrency():
    cfg = config()
    cfg["parallel"] = 4
    with pytest.raises(EngineDied):
        DistributedEngine(cfg)


@pytest.mark.parametrize("slots", [4, 8])
def test_multislot_routes_interleaved_mixed_requests_and_bounds_admission(make_multislot, slots):
    engine = make_multislot(slots, cohort=slots)
    assert engine.batch == engine.info["batch_slots"] == slots
    peak = 0
    original = engine._send

    def send(message, timeout):
        nonlocal peak
        with engine._lock:
            peak = max(peak, len(engine._requests))
        return original(message, timeout)

    engine._send = send
    prompts = [[100 + i] * (5 + i) for i in range(slots + 3)]
    with ThreadPoolExecutor(max_workers=len(prompts)) as pool:
        results = list(pool.map(lambda ids: list(engine.generate_batched(ids, len(ids), {}, threading.Event())), prompts))
    assert results == prompts
    assert peak == slots
    assert engine.alive() and not engine._requests


def test_multislot_serializes_large_jsonl_writes(make_multislot):
    engine = make_multislot(cohort=4)
    prompts = [[100 + index] * 10000 for index in range(4)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda ids: list(engine.generate(ids, 3, {}, threading.Event())), prompts))
    assert results == [ids[:3] for ids in prompts]
    assert engine.alive()


def test_blocked_stdin_write_fails_all_requests_with_bounded_cleanup(make_multislot):
    cfg = config(process_deadline=.15, shutdown_timeout=.1)
    cfg["parallel"] = 4
    cfg["distributed"]["command"] = [sys.executable, "-u", "-c", r'''
import json, time
print(json.dumps(dict(type='ready', protocol='strata-distributed-v1', mode='autoregressive',
      world=3, alive_ranks=[0,1,2], slots=4, max_context=32768)), flush=True)
while True:
    time.sleep(.02)
    print(json.dumps(dict(type='health', state='ready', alive_ranks=[0,1,2])), flush=True)
''']
    engine = DistributedEngine(cfg)
    started = time.monotonic()
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(lambda: list(engine.generate([12345] * 20000, 1, {}, threading.Event())))
                       for _ in range(4)]
            for future in futures:
                with pytest.raises(EngineDied):
                    future.result(timeout=2)
        assert time.monotonic() - started < 2
        assert engine.proc.poll() is not None
    finally:
        engine.close()


@pytest.mark.parametrize("slots", [4, 8])
def test_cancel_during_prefill_is_isolated_and_releases_only_its_slot(make_multislot, slots):
    engine = make_multislot(slots, cohort=2)
    cancel = threading.Event()
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = pool.submit(lambda: list(engine.generate([900], 1, {}, cancel)))
        survivor = pool.submit(lambda: list(engine.generate([42] * 80, 80, {}, threading.Event())))
        deadline = time.monotonic() + 1
        while len(engine._requests) != 2 and time.monotonic() < deadline:
            time.sleep(.005)
        assert len(engine._requests) == 2
        cancel.set()
        assert pending.result(timeout=1) == []
        assert engine.health()["state"] == "busy"
        assert survivor.result(timeout=2) == [42] * 80
    assert engine.alive()
    assert list(engine.generate([1], 1, {}, threading.Event())) == [1]


def test_multislot_generator_close_cancels_only_its_request(make_multislot):
    engine = make_multislot(cohort=1)
    stream = engine.generate([17] * 100, 100, {}, threading.Event())
    assert next(stream) == 17
    with ThreadPoolExecutor(max_workers=1) as pool:
        survivor = pool.submit(lambda: list(engine.generate([42] * 50, 50, {}, threading.Event())))
        stream.close()
        assert survivor.result(timeout=2) == [42] * 50
    assert engine.alive()


@pytest.mark.parametrize("marker", [999, 997])
def test_one_invalid_stream_fails_all_active_requests(make_multislot, marker):
    engine = make_multislot(cohort=2)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(lambda ids=ids: list(engine.generate(ids, len(ids), {}, threading.Event())))
                   for ids in ([marker], [42] * 200)]
        for future in futures:
            with pytest.raises(EngineDied):
                future.result(timeout=2)
    assert not engine.alive()


def test_duplicate_terminal_fails_remaining_active_requests(make_multislot):
    engine = make_multislot(cohort=2)
    with ThreadPoolExecutor(max_workers=2) as pool:
        finished = pool.submit(lambda: list(engine.generate([996], 1, {}, threading.Event())))
        survivor = pool.submit(lambda: list(engine.generate([42] * 200, 200, {}, threading.Event())))
        # The first valid done may already have reached its caller before the duplicate is read.
        try:
            assert finished.result(timeout=2) == [996]
        except EngineDied:
            pass
        with pytest.raises(EngineDied):
            survivor.result(timeout=2)
    assert not engine.alive()


def test_native_rejection_and_invalid_preflight_leave_other_stream_running(make_multislot):
    engine = make_multislot(cohort=1)
    with ThreadPoolExecutor(max_workers=1) as pool:
        survivor = pool.submit(lambda: list(engine.generate([42] * 100, 100, {}, threading.Event())))
        with pytest.raises(ValueError, match="admit"):
            list(engine.generate([998], 1, {}, threading.Event()))
        with pytest.raises(ValueError):
            list(engine.generate([17], 1, {"temperature": float("nan")}, threading.Event()))
        assert survivor.result(timeout=2) == [42] * 100
    assert engine.alive()


@pytest.mark.parametrize("cancelled", [True, False])
def test_queued_cancel_or_deadline_does_not_affect_active_requests(make_multislot, cancelled):
    engine = make_multislot(cohort=1)
    streams = [engine.generate([17] * 100, 100, {}, threading.Event()) for _ in range(4)]
    for stream in streams:
        assert next(stream) == 17
    cancel = threading.Event()
    if not cancelled:
        cancel._strata_queue_deadline = time.monotonic() + .08
    with ThreadPoolExecutor(max_workers=1) as pool:
        queued = pool.submit(lambda: list(engine.generate([42], 1, {}, cancel)))
        time.sleep(.06)
        if cancelled:
            cancel.set()
            assert queued.result(timeout=.5) == []
        else:
            with pytest.raises(ValueError, match="admission deadline"):
                queued.result(timeout=.5)
    assert len(engine._requests) == 4 and engine.alive()
    for stream in streams:
        stream.close()


@pytest.mark.parametrize("slots", [4, 8])
def test_concurrent_http_streams_use_adapter_and_existing_tool_parser(make_multislot, slots):
    engine = make_multislot(slots, cohort=slots, mode="api", process_deadline=5)
    with http_service(engine) as port, contextlib.redirect_stdout(io.StringIO()), ThreadPoolExecutor(max_workers=slots) as pool:
        def call(index):
            return request(port, "/v1/chat/completions", {
                "model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 200,
                "stream": True, "seed": 7 if index == 0 else 8,
                "tools": [{"type": "function", "function": {"name": "get_weather",
                          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}],
            })
        responses = list(pool.map(call, range(slots)))
    for index, (status, raw) in enumerate(responses):
        assert status == 200 and "data: [DONE]" in raw
        events = [json.loads(line[6:]) for line in raw.splitlines() if line.startswith("data: {")]
        deltas = [choice.get("delta", {}) for event in events for choice in event.get("choices", [])]
        if index == 0:
            calls = [call for delta in deltas for call in delta.get("tool_calls", [])]
            assert any(call.get("function", {}).get("name") == "get_weather" for call in calls)
            assert "Paris" in "".join(call.get("function", {}).get("arguments", "") for call in calls)
        else:
            assert "".join(delta.get("content", "") for delta in deltas) == "Hello."
