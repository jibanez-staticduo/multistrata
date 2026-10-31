"""Experimental JSONL boundaries for TP3 and a world-one monolithic reference.

The configured command must own all three ranks and speak JSON lines on stdio.
READY: type=ready, protocol=strata-distributed-v1, mode=autoregressive, world=3,
alive_ranks=[0,1,2], max_context, slots=1|4|8 (optional for C1). Commands are generate(request_id, ids, max_new,
sampling), cancel(request_id), and shutdown. Replies token(request_id, id,
committed=true), done(request_id, state=ready), and cancelled(request_id,
state=ready) must each include alive_ranks. Periodic health(state=ready|busy,
alive_ranks) keeps the group lease alive, including while idle. Error, EOF,
missing ranks, stale health, or invalid output permanently close the session.

The current teacher-forced native worker does not implement this protocol.
ReferenceEngine uses strata-reference-v1 with world=1 and alive_ranks=[0],
and always admits one request. Its configuration lives in the reference section.
No shell, model implementation, sampling, or fallback engine lives here.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import queue
import re
import struct
import subprocess
import threading
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field


PROTOCOL = "strata-distributed-v1"
ROSTER = [0, 1, 2]
MAX_LINE = 1 << 20
MAX_TOKEN = (1 << 31) - 1
MAX_TIMING_NS = (1 << 64) - 1


def normalize_sampling(sampling: dict | None, n_vocab: int = MAX_TOKEN) -> dict:
    """Keep only native sampler fields; reject bad user controls before touching group state."""
    if sampling is None:
        return {}
    if not isinstance(sampling, dict):
        raise ValueError("distributed sampling must be an object")
    values = dict(sampling)
    for alias, key in (("repeat_penalty", "repetition_penalty"), ("repeat_last_n", "penalty_last_n")):
        if values.get(key) is None and values.get(alias) is not None:
            values[key] = values[alias]
    for key in ("strata_tune", "strata_checkpoint", "strata_img_keys", "strata_prefix",
                "experimental_speed_projection", "logprobs", "top_logprobs", "allowed_token_ids"):
        if values.get(key) is not None and not (key == "logprobs" and values[key] is False):
            raise ValueError(f"unsupported distributed engine feature: {key}")
    out = {}
    for key in ("temperature", "top_p", "min_p", "repetition_penalty", "frequency_penalty",
                "presence_penalty", "top_k", "penalty_last_n", "seed"):
        value = values.get(key)
        if value is None:
            continue
        integer = key in ("top_k", "penalty_last_n", "seed")
        if isinstance(value, str):
            try:
                if integer and re.fullmatch(r"\s*[+-]?\d+\s*", value):
                    value = int(value)
                else:
                    number = float(value)
                    value = int(number) if integer and number.is_integer() else number
            except (ValueError, OverflowError):
                raise ValueError(f"invalid distributed sampling field: {key}") from None
        if isinstance(value, bool) or not isinstance(value, int if integer else (int, float)):
            raise ValueError(f"invalid distributed sampling field: {key}")
        if integer:
            if key == "seed":
                if value <= 0:
                    continue
                valid = value < 1 << 64
            elif key == "top_k":
                if value < 0:
                    continue
                value, valid = value if 1 <= value <= 64 else 64, True
            else:
                if value < 0:
                    continue
                valid = value <= MAX_TOKEN
        else:
            try:
                packed = struct.pack("f", value)
                native = struct.unpack("f", packed)[0]
                valid = math.isfinite(value) and math.isfinite(native) and (value == 0 or abs(native) >= 1.17549435e-38)
            except (OverflowError, struct.error):
                valid = False
            if key == "temperature":
                value = max(0, value)
            elif key == "top_p":
                valid = valid and 0 < value <= 1
            elif key == "min_p":
                valid = valid and 0 <= value <= 1
            elif key == "repetition_penalty":
                valid = valid and value > 0
        if not valid:
            raise ValueError(f"invalid distributed sampling field: {key}")
        out[key] = value
    bias = values.get("logit_bias")
    if bias is not None:
        if not isinstance(bias, dict):
            raise ValueError("distributed logit_bias must be an object")
        normalized = {}
        for key, value in bias.items():
            if not isinstance(key, str) or not re.fullmatch(r"[0-9]+", key) or len(key) > 10:
                raise ValueError("invalid distributed logit_bias token")
            token = int(key)
            if token >= n_vocab or str(token) in normalized:
                raise ValueError("invalid distributed logit_bias token")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("invalid distributed logit_bias value")
            try:
                native = struct.unpack("f", struct.pack("f", value))[0]
                valid = math.isfinite(value) and math.isfinite(native) and (value == 0 or abs(native) >= 1.17549435e-38)
            except (OverflowError, struct.error):
                valid = False
            if not valid:
                raise ValueError("invalid distributed logit_bias value")
            normalized[str(token)] = value
        if normalized:
            out["logit_bias"] = normalized
    return out


def _validate_config(cfg: dict, section: str, capacities: tuple[int, ...]) -> dict:
    """Validate before starting any process; never include configured argv in errors."""
    spec = cfg.get(section)
    if not isinstance(spec, dict):
        raise ValueError(f'{section} engine needs a "{section}" config section')
    command = spec.get("command")
    if not isinstance(command, list) or not command or any(not isinstance(x, str) or not x or "\0" in x for x in command):
        raise ValueError(f"{section}.command must be a nonempty argv list")
    for key in ("vision", "images", "replicas", "split_serve", "lazy_load", "idle_unload_s", "slot_save_path", "before_load"):
        if cfg.get(key):
            raise ValueError(f"{section} engine does not support {key}")
    parallel = cfg.get("parallel", spec.get("parallel", 1))
    if type(parallel) is not int or parallel not in capacities:
        raise ValueError(f"{section} parallel must be " + ("1, 4 or 8" if len(capacities) > 1 else "1"))
    if "parallel" in spec and (type(spec["parallel"]) is not int or spec["parallel"] != parallel):
        raise ValueError(f"{section} parallel settings must agree")
    if cfg.get("arch") not in (None, "", "qwen4exp", "qwen35moe"):
        raise ValueError(f"{section} engine currently supports Qwen model families only")
    out = dict(spec)
    out["parallel"] = parallel
    for key, default in (("ready_timeout", 60.0), ("health_timeout", 15.0),
                         ("process_deadline", 900.0), ("cancel_timeout", 10.0), ("shutdown_timeout", 5.0)):
        value = spec.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{section}.{key} must be positive finite seconds")
        out[key] = float(value)
    return out


def validate_config(cfg: dict) -> dict:
    """Validate the TP3 coordinator configuration."""
    return _validate_config(cfg, "distributed", (1, 4, 8))


def validate_reference_config(cfg: dict) -> dict:
    """Validate the opt-in C1 monolithic reference configuration."""
    return _validate_config(cfg, "reference", (1,))


@dataclass
class _Request:
    token_limit: int
    prompt_tokens: int
    events: queue.Queue = field(init=False)
    output_done: bool = False
    done_received: bool = False
    done_timing: dict | None = None
    cancel_sent: bool = False
    cancel_ack: bool = False
    received_tokens: int = 0

    def __post_init__(self) -> None:
        # Tokens, done and a possible cancel acknowledgement fit without blocking the reader.
        self.events = queue.Queue(maxsize=self.token_limit + 2)


def _validate_native_timing(value: object, request: _Request) -> dict:
    """Validate the opt-in native timing payload before accepting its done event."""
    expected = {
        "clock", "queue_ns", "prefill_ns", "decode_ns", "sampling_ns", "mtp_draft_ns",
        "mtp_checkpoint_ns", "mtp_rollback_ns", "prompt_tokens", "generated_tokens",
    }
    if not isinstance(value, dict) or value.keys() != expected or value.get("clock") != "host_steady_wall":
        raise ValueError("invalid native timing fields or clock")
    for key in expected - {"clock"}:
        number = value[key]
        if type(number) is not int or not 0 <= number <= MAX_TIMING_NS:
            raise ValueError("invalid native timing value")
    if value["prompt_tokens"] != request.prompt_tokens or value["generated_tokens"] != request.received_tokens:
        raise ValueError("native timing token counts do not match the request")
    return dict(value)


def _timing_last(timing: dict) -> dict:
    """Translate native nanoseconds while keeping each measured phase separate."""
    return {
        "generated": timing["generated_tokens"],
        "prompt_tokens": timing["prompt_tokens"],
        "prompt_ms": timing["prefill_ns"] / 1_000_000,
        "decode_ms": timing["decode_ns"] / 1_000_000,
        "queue_ms": timing["queue_ns"] / 1_000_000,
        "sampling_ms": timing["sampling_ns"] / 1_000_000,
        "mtp_draft_ms": timing["mtp_draft_ns"] / 1_000_000,
        "mtp_checkpoint_ms": timing["mtp_checkpoint_ns"] / 1_000_000,
        "mtp_rollback_ms": timing["mtp_rollback_ns"] / 1_000_000,
        "native_phase_timing": dict(timing),
    }


class DistributedEngine:
    """Bounded concurrent turns through rank zero, with group-wide failure."""

    batch = 0
    distributed_health = True
    _protocol = PROTOCOL
    _expected_ranks = tuple(ROSTER)
    _backend = "distributed"
    _validate_config = staticmethod(validate_config)

    def __init__(self, cfg: dict, *, engine_error: type[RuntimeError] | None = None):
        # The actual entrypoint supplies its class: python -m runs server as __main__.
        if engine_error is None:
            from serve.server import EngineDied
            engine_error = EngineDied
        self._error_type = engine_error
        self.spec = self._validate_config(cfg)
        self.parallel = self.spec["parallel"]
        self.batch = self.parallel if self.parallel > 1 else 0
        self.n_vocab = MAX_TOKEN
        if cfg.get("tokenizer"):
            try:
                vocab = json.loads((Path(cfg["tokenizer"]) / "vocab.json").read_text(encoding="utf-8"))
                if not isinstance(vocab, dict) or not vocab:
                    raise ValueError
                self.n_vocab = len(vocab)
            except (OSError, ValueError):
                raise ValueError(f"{self._backend} engine needs the native pack vocabulary") from None
        self.max_context = 0
        self.info = {"backend": self._backend, "world": len(self._expected_ranks), "batch_slots": self.batch}
        self._local = threading.local()
        self.last = None
        self._lock = threading.RLock()
        self._capacity = threading.BoundedSemaphore(self.parallel)
        self._write_lock = threading.Lock()
        self._close_lock = threading.Lock()
        self._cleanup_started = False
        self._ready = threading.Event()
        self._requests: dict[str, _Request] = {}
        self._terminal: str | None = None
        self._state = "starting"
        self._roster: list[int] = []
        self._last_health = time.monotonic()
        self._write_done = threading.Event()
        self._write_done.set()
        self._writers = 0
        self.proc = None
        try:
            # Discard stderr: a noisy worker cannot fill a pipe or expose prompts or keys in API errors.
            self.proc = subprocess.Popen(self.spec["command"], cwd=cfg.get("cwd"), stdin=subprocess.PIPE,
                                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                         text=True, encoding="utf-8", bufsize=1)
            self._reader = threading.Thread(target=self._read, daemon=True, name="strata-distributed-output")
            self._reader.start()
            if not self._ready.wait(self.spec["ready_timeout"]):
                self._fail("the distributed coordinator did not become ready")
            self._check()
        except (OSError, ValueError):
            self._fail("the distributed coordinator could not start")
            self.close()
            raise self._error_type(self._terminal) from None
        except BaseException:
            self.close()
            raise

    @property
    def last(self) -> dict | None:
        return getattr(self._local, "last", None) if self.batch else getattr(self, "_last", None)

    @last.setter
    def last(self, value: dict | None) -> None:
        if self.batch:
            self._local.last = value
        else:
            self._last = value

    @property
    def _request(self) -> str | None:
        """Keep the C1 inspection boundary while storing requests independently."""
        with self._lock:
            return next(iter(self._requests), None)

    def _fail(self, reason: str) -> None:
        with self._lock:
            self._terminal = self._terminal or reason
            self._state = "failed"
            self._roster = []
            self._ready.set()
            cleanup = not self._cleanup_started
            self._cleanup_started = True
        if self.proc is not None and self.proc.poll() is None:
            try:
                self.proc.terminate()
            except OSError:
                pass
        if cleanup:
            threading.Thread(target=self.close, daemon=True, name="strata-distributed-cleanup").start()

    def _ranks(self, message: dict) -> bool:
        ranks = message.get("alive_ranks")
        return (isinstance(ranks, list) and all(type(x) is int for x in ranks)
                and sorted(ranks) == list(self._expected_ranks))

    def _read(self) -> None:
        try:
            while True:
                line = self.proc.stdout.readline(MAX_LINE + 1)
                if not line:
                    self._fail("the distributed coordinator ended")
                    return
                if len(line) > MAX_LINE or not line.endswith("\n"):
                    raise ValueError
                msg = json.loads(line)
                if not isinstance(msg, dict):
                    raise ValueError
                if not self._ranks(msg):
                    with self._lock:
                        self._roster = [x for x in msg.get("alive_ranks", [])
                                        if type(x) is int and x in self._expected_ranks] if isinstance(msg.get("alive_ranks"), list) else []
                    raise ValueError
                kind = msg.get("type")
                event = None
                with self._lock:
                    if self._terminal:
                        return
                    if kind == "ready" and not self._ready.is_set():
                        ctx = msg.get("max_context")
                        slots = msg.get("slots", 1)
                        if (msg.get("protocol") != self._protocol or msg.get("mode") != "autoregressive"
                                or type(msg.get("world")) is not int or msg["world"] != len(self._expected_ranks)
                                or type(ctx) is not int or ctx <= 0
                                or type(slots) is not int or slots != self.parallel):
                            raise ValueError
                        self.max_context, self._state = ctx, "ready"
                        self._ready.set()
                    elif not self._ready.is_set():
                        raise ValueError
                    elif kind == "health":
                        if msg.get("state") not in ("ready", "busy"):
                            raise ValueError
                        self._state = "busy" if any(not r.output_done for r in self._requests.values()) else msg["state"]
                    elif kind in ("token", "done", "cancelled", "rejected"):
                        request = self._requests.get(msg.get("request_id"))
                        if request is None:
                            raise ValueError
                        if kind == "token":
                            if (request.output_done or msg.get("committed") is not True
                                    or type(msg.get("id")) is not int or msg["id"] < 0
                                    or msg["id"] >= self.n_vocab
                                    or request.received_tokens >= request.token_limit):
                                raise ValueError
                            request.received_tokens += 1
                        elif msg.get("state") not in (("ready", "busy") if kind == "rejected" else ("ready",)):
                            raise ValueError
                        elif kind == "cancelled":
                            if not request.cancel_sent or request.cancel_ack:
                                raise ValueError
                            request.cancel_ack = True
                            request.output_done = True
                        else:
                            if request.output_done or (kind == "rejected" and request.received_tokens):
                                raise ValueError
                            if kind == "done" and "timing" in msg:
                                request.done_timing = _validate_native_timing(msg["timing"], request)
                            if kind == "done":
                                request.done_received = True
                            request.output_done = True
                        self._state = "busy" if any(not r.output_done for r in self._requests.values()) else "ready"
                        event = request.events
                    else:
                        raise ValueError
                    self._roster, self._last_health = list(self._expected_ranks), time.monotonic()
                if event is not None:
                    event.put_nowait(msg)
        except (OSError, ValueError, TypeError, UnicodeError, queue.Full):
            self._fail("the distributed coordinator sent invalid group state")

    def health(self) -> dict:
        """Health expires even if a hung coordinator still has a live process."""
        with self._lock:
            if not self._terminal and self.proc is not None:
                if self.proc.poll() is not None:
                    self._fail("the distributed coordinator ended")
                elif time.monotonic() - self._last_health > self.spec["health_timeout"]:
                    self._fail("the distributed group health deadline expired")
            ok = self._terminal is None and self._ready.is_set() and self._roster == list(self._expected_ranks)
            return {"status": "ok" if ok else "unavailable", "ready": ok, "world": len(self._expected_ranks),
                    "alive_ranks": list(self._roster), "state": self._state}

    def alive(self) -> bool:
        return self.health()["ready"]

    def _check(self) -> None:
        if not self.alive():
            raise self._error_type(self._terminal or "the distributed group is unavailable")

    def restart(self) -> None:
        raise self._error_type("the distributed group stopped; restart the whole group before serving again")

    def _send(self, msg: dict, timeout: float) -> None:
        self._check()
        try:
            data = json.dumps(msg, separators=(",", ":"), allow_nan=False) + "\n"
        except (ValueError, TypeError):
            raise ValueError("distributed request must contain finite JSON values") from None
        sent = threading.Event()
        failed = []

        def write():
            try:
                with self._write_lock:
                    # A timed-out predecessor must never be followed by another command.
                    self._check()
                    self.proc.stdin.write(data)
                    self.proc.stdin.flush()
            except (OSError, ValueError):
                failed.append(True)
            except RuntimeError:
                failed.append(True)
            finally:
                sent.set()
                with self._lock:
                    self._writers -= 1
                    if not self._writers:
                        self._write_done.set()

        with self._lock:
            self._writers += 1
            self._write_done.clear()
            threading.Thread(target=write, daemon=True, name="strata-distributed-input").start()
        if not sent.wait(timeout) or failed:
            self._fail("the distributed coordinator did not accept the command")
            self._check()

    def _next(self, request: _Request, deadline: float) -> dict | None:
        self._check()
        if time.monotonic() >= deadline:
            self._fail("the distributed request deadline expired")
            self._check()
        try:
            msg = request.events.get(timeout=min(0.05, max(0.001, deadline - time.monotonic())))
        except queue.Empty:
            return None
        self._check()
        return msg

    def _cancel(self, request_id: str, request: _Request) -> None:
        deadline = time.monotonic() + self.spec["cancel_timeout"]
        with self._lock:
            if request.output_done:
                return
            request.cancel_sent = True
        self._send({"type": "cancel", "request_id": request_id}, self.spec["cancel_timeout"])
        while True:
            msg = self._next(request, deadline)
            if msg and msg["type"] == "cancelled":
                return

    def generate(self, ids: list[int], max_new: int, sampling: dict, cancel: threading.Event,
                 embeddings=None) -> Iterator[int | None]:
        if embeddings is not None:
            raise ValueError("distributed engine does not support images")
        if type(max_new) is not int or max_new < 0:
            raise ValueError("distributed token budget must be a nonnegative integer")
        if max_new == 0 or cancel.is_set():
            return
        if (not isinstance(ids, list) or not ids or any(type(i) is not int or not 0 <= i < self.n_vocab for i in ids)
                or len(ids) + max_new > self.max_context):
            raise ValueError("distributed token budget must fit the model context")
        sampling = normalize_sampling(sampling, self.n_vocab)
        request = uuid.uuid4().hex
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", request):
            raise ValueError("invalid distributed request ID")
        message = {"type": "generate", "request_id": request, "ids": list(ids),
                   "max_new": max_new, "sampling": sampling}
        if len((json.dumps(message, separators=(",", ":"), allow_nan=False) + "\n").encode()) > MAX_LINE:
            raise ValueError("distributed request exceeds one MiB")
        deadline = time.monotonic() + self.spec["process_deadline"]
        queue_deadline = getattr(cancel, "_strata_queue_deadline", None)
        admission_deadline = min(deadline, queue_deadline) if queue_deadline is not None else deadline
        heartbeat = time.monotonic()
        while not self._capacity.acquire(timeout=.05):
            self._check()
            if cancel.is_set():
                return
            if time.monotonic() >= admission_deadline:
                raise ValueError("the distributed request admission deadline expired")
            if time.monotonic() - heartbeat >= 1.0:
                heartbeat = time.monotonic()
                yield None
        try:
            self._check()
            if cancel.is_set():
                return
            if time.monotonic() >= admission_deadline:
                raise ValueError("the distributed request admission deadline expired")
            complete, submitted = False, False
            state = _Request(max_new, len(ids))
            with self._lock:
                if request in self._requests:
                    raise ValueError("duplicate distributed request ID")
                self._requests[request], self._state = state, "busy"
            count = 0
            heartbeat = time.monotonic()
            try:
                self._send(message, max(.001, deadline - time.monotonic()))
                submitted = True
                started = getattr(cancel, "_strata_queue_started", None)
                if started is not None:
                    started()
                while not cancel.is_set():
                    msg = self._next(state, deadline)
                    if msg is None:
                        if time.monotonic() - heartbeat >= 1.0:
                            heartbeat = time.monotonic()
                            yield None
                        continue
                    if msg["type"] == "done":
                        self.last = _timing_last(state.done_timing) if state.done_timing else {"generated": count}
                        complete = True
                        break
                    if msg["type"] == "rejected":
                        complete = True
                        raise ValueError("the distributed coordinator could not admit the request; retry later")
                    if msg["type"] != "token" or count >= max_new:
                        self._fail("the distributed coordinator violated the generation boundary")
                        self._check()
                    count += 1
                    yield msg["id"]
            finally:
                try:
                    # Generator.close() (EOS, stop strings, disconnected HTTP) must also drain the group turn.
                    if submitted and not complete and self.alive():
                        self._cancel(request, state)
                finally:
                    with self._lock:
                        native_timing = state.done_timing if state.done_received else None
                    if native_timing is not None:
                        self.last = _timing_last(native_timing)
                    with self._lock:
                        self._requests.pop(request, None)
                        if self._terminal is None:
                            self._state = "busy" if any(not r.output_done for r in self._requests.values()) else "ready"
                    if self._terminal:
                        self.close()
        finally:
            self._capacity.release()

    def generate_batched(self, ids: list[int], max_new: int, sampling: dict, cancel: threading.Event,
                         embeddings=None) -> Iterator[int | None]:
        yield from self.generate(ids, max_new, sampling, cancel, embeddings)

    def close(self) -> None:
        """Bounded process shutdown; never wait on stderr or a blocked stdin writer."""
        with self._close_lock:
            self._close()

    def _close(self) -> None:
        with self._lock:
            self._terminal = self._terminal or "the distributed coordinator was closed"
            self._state = "closed" if self._state != "failed" else self._state
            self._roster = []
        if self.proc is None:
            return
        try:
            if self.proc.poll() is None:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=self.spec["shutdown_timeout"])
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    self.proc.wait(timeout=self.spec["shutdown_timeout"])
        finally:
            # A descendant can inherit stdout. Do not block closing a pipe held by its reader.
            if self.proc.poll() is not None:
                reader = getattr(self, "_reader", None)
                if reader is not None and reader is not threading.current_thread():
                    reader.join(timeout=.1)
                self._write_done.wait(timeout=.1)
                pipes = [self.proc.stdin] if self._write_done.is_set() else []
                if reader is None or not reader.is_alive():
                    pipes.append(self.proc.stdout)
                for pipe in pipes:
                    if pipe is not None:
                        pipe.close()


class ReferenceEngine(DistributedEngine):
    """Opt-in C1 monolithic reference using the same bounded JSONL boundary."""

    reference_health = True
    _protocol = "strata-reference-v1"
    _expected_ranks = (0,)
    _backend = "reference"
    _validate_config = staticmethod(validate_reference_config)

    def restart(self) -> None:
        raise self._error_type("the reference engine stopped; restart it explicitly before serving again")
