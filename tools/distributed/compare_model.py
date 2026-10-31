#!/usr/bin/env python3
"""Compare teacher-forced model evidence, without running an engine.

Usage: compare_model.py baseline.json candidate.json --policy policy.json --output receipt.json
Exit 0: logits and declared KV/GDN state pass; 1: numerical failure; 2: invalid
evidence; 77: unqualified (measure-only or missing state evidence).

Metadata schema 1 requires commit, pack_sha256, policy_sha256, settings, backend,
mode, started_at (timezone ISO8601), command (argv), n_vocab, token_ids, and rows.
Rows are {request_id, position, path}; paths are relative to metadata. Format
raw_f32_rows reads one little-endian float32 vector per row (native rank 0).
strata_dump reads logits_path: little-endian int32 vocabulary and row count,
then float32 rows, as written by generate.cpp --dump-logits; row paths are absent.
token_ids maps each request ID to the *entire* exact teacher-forcing input list.

Policy schema 1 requires frozen_at and thresholds: max_abs, mean_abs, relative_l2,
min_top1_agreement, near_tie_margin. Freeze the file before both runs and record
its SHA256 in both metadata files. There are no default numerical thresholds.
Optional states: [{kind: kv|gdn|indexer|ple|residual|metadata, name, request_id,
position, dtype: <f2|<f4|<f8|<i4|<i8|u1,
shape: [positive dimensions], path}]. These are logical reconstructed arrays,
not opaque checksums. A full-model gate also requires policy.required_states:
[{kind, name, dtype, shape}] listing every logical state expected at every row,
including KV and GDN. Without that frozen inventory state validation is missing.
State identity, dtype and shape must match exactly.
Integer arrays compare exactly, including 64-bit IDs (no float conversion).
state_manifests: [{rank: 0|1|2, path}] imports C++ state_evidence manifests.
axis_shard arrays reconstruct by explicit axis_indices only when all three ranks
cover the logical axis exactly once. Partial coverage never completes the gate.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import sys
from typing import Any

import numpy as np

STATE_KINDS = {"kv", "gdn", "indexer", "ple", "residual", "metadata"}
STATE_DTYPES = {"<f2", "<f4", "<f8", "<i4", "<i8", "|u1"}


class EvidenceError(ValueError):
    """An evidence artifact cannot support the requested comparison."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceError(message)


def timestamp(value: Any) -> dt.datetime:
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
        require(parsed.tzinfo is not None, "timestamps require a timezone")
        return parsed
    except (TypeError, AttributeError, ValueError) as exc:
        raise EvidenceError("invalid timestamp") from exc


def integer(value: Any) -> bool:
    return type(value) is int and value >= 0


def read_json(path: Path) -> dict[str, Any]:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f"duplicate JSON key: {key}")
            result[key] = value
        return result
    result = json.loads(path.read_text(), object_pairs_hook=unique)
    require(isinstance(result, dict), "JSON root must be an object")
    return result


def read_policy(path: Path) -> dict[str, Any]:
    policy = read_json(path)
    require(policy.get("schema_version") == 1, "policy schema_version must be 1")
    timestamp(policy.get("frozen_at"))
    limits = policy.get("thresholds", {})
    for key in ("max_abs", "mean_abs", "relative_l2", "min_top1_agreement", "near_tie_margin"):
        value = limits.get(key)
        require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
                f"required finite nonnegative threshold: {key}")
    require(limits["min_top1_agreement"] <= 1, "min_top1_agreement exceeds 1")
    inventory = policy.get("required_states", [])
    require(isinstance(inventory, list), "required_states must be an array")
    names = set()
    for state in inventory:
        require(isinstance(state, dict) and state.get("kind") in STATE_KINDS and
                isinstance(state.get("name"), str) and bool(state["name"]), "invalid required state")
        key = state["kind"], state["name"]
        require(key not in names, "duplicate required state")
        names.add(key)
        require(state.get("dtype") in STATE_DTYPES, "invalid required state dtype")
        shape_size(state.get("shape"), state["dtype"])
        require(state.get("layout", "logical") == "logical", "frozen inventory must declare logical arrays")
    return policy


def shape_size(shape: Any, dtype: str) -> int:
    require(isinstance(shape, list) and 0 < len(shape) <= 32 and
            all(integer(n) and 0 < n <= sys.maxsize for n in shape), "invalid shape")
    size = math.prod(shape) * np.dtype(dtype).itemsize
    require(size <= sys.maxsize, "shape byte size overflow")
    return size


def raw(path: Path, dtype: str, shape: list[int], files: dict[str, str]) -> np.ndarray:
    require(dtype in STATE_DTYPES, "unsupported explicit dtype")
    size = shape_size(shape, dtype)
    require(path.stat().st_size == size,
            f"truncated or oversized array: {path}")
    files[str(path.resolve())] = sha256(path)
    return np.fromfile(path, dtype=dtype).reshape(shape)


def identity(row: dict) -> tuple[str, int]:
    require(isinstance(row, dict), "row must be an object")
    require(isinstance(row.get("request_id"), str) and bool(row["request_id"]), "missing request ID")
    require(integer(row.get("position")), "invalid position")
    return row["request_id"], row["position"]


def load(path: Path, policy_path: Path, policy: dict, files: dict) -> dict:
    meta = read_json(path)
    files[str(path.resolve())] = sha256(path)
    require(meta.get("schema_version") == 1, "metadata schema_version must be 1")
    for key, length in (("commit", 40), ("pack_sha256", 64), ("policy_sha256", 64)):
        require(isinstance(meta.get(key), str) and re.fullmatch(f"[0-9a-f]{{{length}}}", meta[key]) is not None,
                f"invalid {key}")
    require(meta["policy_sha256"] == sha256(policy_path), "policy hash mismatch")
    require(timestamp(policy["frozen_at"]) <= timestamp(meta.get("started_at")), "policy frozen after run started")
    for key in ("backend", "mode"):
        require(isinstance(meta.get(key), str) and bool(meta[key]), f"missing {key}")
    require(isinstance(meta.get("settings"), dict) and bool(meta["settings"]), "missing settings")
    require(isinstance(meta.get("command"), list) and bool(meta["command"]) and
            all(isinstance(s, str) and s for s in meta["command"]), "missing command argv provenance")
    require(integer(meta.get("n_vocab")) and meta["n_vocab"] >= 2, "invalid n_vocab")
    tokens = meta.get("token_ids")
    require(isinstance(tokens, dict) and bool(tokens), "missing teacher-forcing token IDs")
    for request, values in tokens.items():
        require(isinstance(request, str) and bool(request) and isinstance(values, list) and bool(values) and
                all(integer(t) and t < meta["n_vocab"] for t in values), "invalid teacher-forcing token IDs")
    rows = meta.get("rows")
    require(isinstance(rows, list) and bool(rows), "empty logits evidence")
    keys = [identity(row) for row in rows]
    require(len(set(keys)) == len(keys), "duplicate request position")
    require(set(tokens) == {key[0] for key in keys}, "request IDs differ from token IDs")
    for request, pos in keys:
        require(pos < len(tokens[request]), "position outside teacher-forcing input")
    arrays = {}
    if meta.get("format") == "raw_f32_rows":
        for key, row in zip(keys, rows):
            arrays[key] = raw(path.parent / row["path"], "<f4", [meta["n_vocab"]], files)
    elif meta.get("format") == "strata_dump":
        dump = path.parent / meta["logits_path"]
        with dump.open("rb") as stream:
            header = stream.read(8)
        require(len(header) == 8, "truncated strata dump header")
        vocab, count = struct.unpack("<ii", header)
        require((vocab, count) == (meta["n_vocab"], len(rows)), "strata dump header shape mismatch")
        require(dump.stat().st_size == 8 + vocab * count * 4, "strata dump body size mismatch")
        files[str(dump.resolve())] = sha256(dump)
        data = np.fromfile(dump, dtype="<f4", offset=8).reshape(count, vocab)
        arrays = dict(zip(keys, data))
    else:
        raise EvidenceError("unknown logits format")
    state_entries = [(state, path.parent, None) for state in meta.get("states", [])]
    manifests = meta.get("state_manifests", [])
    require(isinstance(manifests, list), "state_manifests must be an array")
    manifest_ids = set()
    for entry in manifests:
        rank = entry.get("rank")
        require(integer(rank) and rank < 3, "state manifest requires rank 0, 1 or 2")
        manifest_path = path.parent / entry["path"]
        manifest = read_json(manifest_path)
        files[str(manifest_path.resolve())] = sha256(manifest_path)
        require(manifest.get("schema_version") == 1 and manifest.get("complete") is True,
                "incomplete state manifest")
        require(manifest.get("phase", "token") == "token",
                "reset manifests require a separate reset gate, not a logits row")
        key = identity(manifest)
        require(key in arrays and manifest.get("token_count") == key[1] + 1, "state manifest identity mismatch")
        require((*key, rank) not in manifest_ids, "duplicate state manifest rank")
        manifest_ids.add((*key, rank))
        declared = manifest.get("states")
        require(isinstance(declared, list) and bool(declared), "empty state manifest")
        require(isinstance(manifest.get("required_states"), list), "missing manifest inventory")
        actual = [{k: v for k, v in s.items() if k not in ("path", "request_id", "position")} for s in declared]
        require(sorted(actual, key=lambda s: (s["kind"], s["name"])) ==
                sorted(manifest["required_states"], key=lambda s: (s["kind"], s["name"])),
                "state manifest inventory mismatch")
        total = 0
        for state in declared:
            require(identity(state) == key, "state manifest row mismatch")
            require(state.get("dtype") in STATE_DTYPES, "unsupported manifest dtype")
            total += shape_size(state.get("shape"), state.get("dtype"))
            state_entries.append((state, manifest_path.parent, rank))
        require(type(manifest.get("total_bytes")) is int and total == manifest["total_bytes"],
                "state manifest byte total mismatch")
        require(manifest.get("logical_complete") is all(s.get("layout", "logical") == "logical" for s in declared),
                "state manifest logical coverage mismatch")
    states, shards, logical_ranks = {}, {}, {}
    for state, directory, rank in state_entries:
        key = identity(state)
        require(key in arrays and state.get("kind") in STATE_KINDS and
                isinstance(state.get("name"), str) and bool(state["name"]), "invalid state identity")
        state_key = (*key, state["kind"], state["name"])
        array = raw(directory / state["path"], state["dtype"], state["shape"], files)
        layout = state.get("layout", "logical")
        if layout == "axis_shard":
            require(state_key not in states, "mixed logical and shard state identity")
            require(rank is not None, "axis shards require ranked state manifests")
            shards.setdefault(state_key, []).append((state, rank, array))
        else:
            require(layout == "logical" and state_key not in shards and
                    not any(k in state for k in ("logical_shape", "shard_axis", "axis_indices")),
                    "invalid logical state layout")
            if state_key in states:
                previous_dtype, previous_shape, previous = states[state_key]
                require(rank is not None and None not in logical_ranks[state_key] and
                        rank not in logical_ranks[state_key], "duplicate state identity")
                require((previous_dtype, previous_shape) == (state["dtype"], state["shape"]) and
                        np.array_equal(previous, array), "replicated logical state mismatch")
                logical_ranks[state_key].add(rank)
                continue
            states[state_key] = (state["dtype"], state["shape"], array)
            logical_ranks[state_key] = {rank}
    partial = []
    for key, pieces in shards.items():
        first = pieces[0][0]
        shape, axis, dtype = first.get("logical_shape"), first.get("shard_axis"), first["dtype"]
        logical_bytes = shape_size(shape, dtype)
        require(integer(axis) and axis < len(shape), "invalid shard axis")
        owned, ranks = set(), set()
        for state, rank, array in pieces:
            require(rank not in ranks, "duplicate state shard rank")
            ranks.add(rank)
            require(state.get("logical_shape") == shape and state.get("shard_axis") == axis and
                    state["dtype"] == dtype and array.ndim == len(shape), "incompatible state shard descriptors")
            indices = state.get("axis_indices")
            require(isinstance(indices, list) and len(indices) == array.shape[axis] and
                    all(integer(i) and i < shape[axis] for i in indices), "invalid shard axis indices")
            require(all(array.shape[i] == shape[i] for i in range(len(shape)) if i != axis),
                    "state shard dimension mismatch")
            for index in indices:
                require(index not in owned, "overlapping state shard ownership")
                owned.add(index)
        if ranks != {0, 1, 2} or len(owned) != shape[axis]:
            partial.append(dict(request_id=key[0], position=key[1], kind=key[2], name=key[3],
                                ranks=sorted(ranks), covered_axis_elements=len(owned), logical_shape=shape))
            continue
        require(sum(a.nbytes for _, _, a in pieces) == logical_bytes, "state shard byte coverage mismatch")
        reconstructed = np.empty(shape, dtype=dtype)
        for state, _, array in pieces:
            selection = [slice(None)] * len(shape)
            selection[axis] = state["axis_indices"]
            reconstructed[tuple(selection)] = array
        states[key] = (dtype, shape, reconstructed)
    return {"metadata": meta, "logits": arrays, "states": states, "partial_states": partial}


def metrics(reference: np.ndarray, candidate: np.ndarray) -> dict:
    require(reference.shape == candidate.shape, "array shape mismatch")
    finite = bool(np.isfinite(reference).all() and np.isfinite(candidate).all())
    if not finite:
        return {"finite": False, "max_abs": None, "mean_abs": None, "relative_l2": None}
    ref = reference.astype(np.float64).ravel()
    delta = candidate.astype(np.float64).ravel() - ref
    norm = float(np.linalg.norm(ref))
    delta_norm = float(np.linalg.norm(delta))
    return {"finite": True, "max_abs": float(np.abs(delta).max()),
            "mean_abs": float(np.abs(delta).mean()),
            "relative_l2": delta_norm / norm if norm else (0.0 if delta_norm == 0 else None)}


def numerical_pass(values: dict, limits: dict) -> bool:
    if "exact_equal" in values:
        return values["exact_equal"]
    return values["finite"] and all(values[k] is not None and values[k] <= limits[k]
                                     for k in ("max_abs", "mean_abs", "relative_l2"))


def compare(baseline: Path, candidate: Path, policy_path: Path, *,
            product_comparison: bool = False, measure_only: bool = False) -> tuple[dict, int]:
    """Validate artifact identity before computing any numerical verdict."""
    files = {str(policy_path.resolve()): sha256(policy_path)}
    policy = read_policy(policy_path)
    left = load(baseline, policy_path, policy, files)
    right = load(candidate, policy_path, policy, files)
    required = ("commit", "pack_sha256", "policy_sha256", "n_vocab", "token_ids")
    for key in required:
        require(left["metadata"][key] == right["metadata"][key], f"mismatched {key}")
    differences = [key for key in ("settings", "backend", "mode")
                   if left["metadata"][key] != right["metadata"][key]]
    require(product_comparison or not differences, f"mismatched settings/backend/mode: {differences}")
    require(left["logits"].keys() == right["logits"].keys(), "mismatched request positions")
    limits = policy["thresholds"]
    rows = []
    for key in left["logits"]:
        ref, cand = left["logits"][key], right["logits"][key]
        values = metrics(ref, cand)
        values.update(request_id=key[0], position=key[1])
        if values["finite"]:
            top = int(np.argmax(ref))
            second = float(np.partition(ref, -2)[-2])
            values.update(top1_reference=top, top1_candidate=int(np.argmax(cand)),
                          top1_margin=float(ref[top]) - second)
            values["top1_agrees"] = values["top1_candidate"] == top
            values["near_tie"] = values["top1_margin"] <= limits["near_tie_margin"]
        else:
            values.update(top1_agrees=False, near_tie=None, top1_margin=None)
        values["numerical_pass"] = numerical_pass(values, limits)
        rows.append(values)
    agreement = sum(r["top1_agrees"] for r in rows) / len(rows)
    logits_pass = all(r["numerical_pass"] for r in rows) and agreement >= limits["min_top1_agreement"]
    require(left["states"].keys() == right["states"].keys(), "mismatched state identities")
    states = []
    for key, (dtype, shape, array) in left["states"].items():
        other_dtype, other_shape, other = right["states"][key]
        require((dtype, shape) == (other_dtype, other_shape), "mismatched state dtype/shape")
        values = ({"finite": True, "exact_equal": bool(np.array_equal(array, other)),
                   "comparison": "integer_exact"} if np.dtype(dtype).kind in "iu" else metrics(array, other))
        values.update(request_id=key[0], position=key[1], kind=key[2], name=key[3])
        values["numerical_pass"] = numerical_pass(values, limits)
        states.append(values)
    inventory = policy.get("required_states", [])
    frozen_names = {(s["kind"], s["name"]) for s in inventory}
    unfrozen = [dict(kind=k[2], name=k[3]) for k in left["states"] if (k[2], k[3]) not in frozen_names]
    missing = []
    for key in left["logits"]:
        for expected in inventory:
            state_key = (*key, expected["kind"], expected["name"])
            if state_key not in left["states"]:
                missing.append(dict(request_id=key[0], position=key[1], **expected))
            else:
                dtype, shape, _ = left["states"][state_key]
                require((dtype, shape) == (expected["dtype"], expected["shape"]),
                        "state dtype/shape differs from frozen inventory")
    inventory_complete = {"kv", "gdn"}.issubset({s["kind"] for s in inventory})
    partial = {"baseline": left["partial_states"], "candidate": right["partial_states"]}
    state_status = "missing" if missing or unfrozen or any(partial.values()) or not inventory_complete else (
        "passed" if all(s["numerical_pass"] for s in states) else "failed")
    full_pass = logits_pass and state_status == "passed" and not measure_only
    report = {"schema_version": 1, "comparison": "product" if product_comparison else "strict",
              "metadata_differences": differences,
              "qualified": not measure_only and state_status != "missing",
              "logits_qualified": not measure_only,
              "measure_only": measure_only, "model_logits_pass": logits_pass,
              "full_model_pass": full_pass, "state_validation": state_status,
              "missing_states": missing, "state_inventory_provided": inventory_complete,
              "unfrozen_states": unfrozen, "partial_states": partial,
              "top1_agreement": agreement,
              "rows": rows, "states": states, "policy": policy,
              "provenance": {"baseline": left["metadata"], "candidate": right["metadata"]},
              "files_sha256": files}
    failed = not logits_pass or any(not s["numerical_pass"] for s in states)
    return report, 77 if measure_only else (1 if failed else (0 if full_pass else 77))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--product-comparison", action="store_true")
    parser.add_argument("--measure-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        report, code = compare(args.baseline, args.candidate, args.policy,
                               product_comparison=args.product_comparison, measure_only=args.measure_only)
    except (EvidenceError, OSError, ValueError, KeyError, TypeError) as exc:
        report, code = {"schema_version": 1, "qualified": False, "full_model_pass": False,
                        "error": str(exc)}, 2
    report["comparison_command"] = [sys.executable, str(Path(__file__).resolve()), *(argv if argv is not None else sys.argv[1:])]
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"receipt": str(args.output), "exit_code": code}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
