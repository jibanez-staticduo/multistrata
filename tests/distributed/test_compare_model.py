"""CPU-only artifact tests for the teacher-forced Task 7 evidence gate."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import struct
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "distributed"))
import compare_model as cm


@pytest.fixture
def evidence(tmp_path):
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({"schema_version": 1, "frozen_at": "2026-10-30T00:00:00Z",
                                 "thresholds": {"max_abs": 0.01, "mean_abs": 0.005,
                                                "relative_l2": 0.01, "min_top1_agreement": 1,
                                                "near_tie_margin": 0.002},
                                 "required_states": [{"kind": kind, "name": "layer.0",
                                                      "dtype": "<f2", "shape": [2, 2]}
                                                     for kind in ("kv", "gdn")]}))
    meta = {"schema_version": 1, "commit": "a" * 40, "pack_sha256": "b" * 64,
            "policy_sha256": hashlib.sha256(policy.read_bytes()).hexdigest(),
            "settings": {"context": 4096, "quantization": "Q8_1"}, "backend": "cuda",
            "mode": "teacher_forced", "started_at": "2026-10-30T01:00:00Z",
            "command": ["strata", "generate", "--tokens", "1,2"], "n_vocab": 4,
            "token_ids": {"request-A": [1, 2]},
            "rows": [{"request_id": "request-A", "position": p} for p in (0, 1)]}
    a, b = tmp_path / "baseline.json", tmp_path / "candidate.json"
    rows = np.array([[1, 3, 2, -1], [4, 0, 1, -3]], dtype="<f4")
    dump = tmp_path / "baseline.bin"
    dump.write_bytes(struct.pack("<ii", 4, 2) + rows.tobytes())
    left = dict(meta, format="strata_dump", logits_path=dump.name)
    a.write_text(json.dumps(left))
    right = copy.deepcopy(meta)
    right["format"] = "raw_f32_rows"
    for p, row in enumerate(right["rows"]):
        row["path"] = f"logits-{p}.f32"
        rows[p].tofile(tmp_path / row["path"])
    b.write_text(json.dumps(right))
    return a, b, policy


def change(path, fn):
    meta = json.loads(path.read_text())
    fn(meta)
    path.write_text(json.dumps(meta))


def add_states(a, b):
    for index, path in enumerate((a, b)):
        def add(meta):
            meta["states"] = []
            for row in meta["rows"]:
                for kind in ("kv", "gdn"):
                    filename = f"state-{index}-{kind}-{row['position']}.bin"
                    np.array([1, 2, 3, 4], dtype="<f2").tofile(path.parent / filename)
                    meta["states"].append({"request_id": row["request_id"], "position": row["position"],
                                           "kind": kind, "name": "layer.0", "dtype": "<f2",
                                           "shape": [2, 2], "path": filename})
        change(path, add)


def test_binary_baseline_and_rank_zero_rows_are_not_a_full_state_gate(evidence):
    report, code = cm.compare(*evidence)
    assert code == 77
    assert report["model_logits_pass"]
    assert report["state_validation"] == "missing"
    assert not report["full_model_pass"]
    assert len(report["missing_states"]) == 4
    assert report["rows"][0]["top1_margin"] == 1
    assert len(report["files_sha256"]) == 6


def test_real_declared_state_arrays_complete_gate_and_hash_all_files(evidence):
    a, b, policy = evidence
    add_states(a, b)
    report, code = cm.compare(a, b, policy)
    assert code == 0 and report["full_model_pass"]
    assert report["state_validation"] == "passed"
    assert len(report["states"]) == 4
    state_file = a.parent / "state-1-gdn-1.bin"
    np.array([1, 2, 3, 8], dtype="<f2").tofile(state_file)
    report, code = cm.compare(a, b, policy)
    assert code == 1 and report["model_logits_pass"]
    assert report["state_validation"] == "failed"


def test_measure_only_never_qualifies_even_with_full_evidence(evidence):
    add_states(*evidence[:2])
    report, code = cm.compare(*evidence, measure_only=True)
    assert code == 77 and not report["qualified"] and not report["full_model_pass"]


def test_normalized_relative_l2_is_not_raw_delta_norm():
    actual = cm.metrics(np.array([300., 400.]), np.array([303., 404.]))
    assert actual["max_abs"] == 4
    assert actual["mean_abs"] == 3.5
    assert actual["relative_l2"] == pytest.approx(0.01)
    assert cm.metrics(np.zeros(2), np.ones(2))["relative_l2"] is None


def test_near_tie_is_reported_without_overriding_frozen_agreement(evidence):
    a, b, policy = evidence
    ref = np.array([[1, 1, 0, -1], [4, 0, 1, -3]], dtype="<f4")
    (a.parent / "baseline.bin").write_bytes(struct.pack("<ii", 4, 2) + ref.tobytes())
    np.array([1, 1.001, 0, -1], dtype="<f4").tofile(b.parent / "logits-0.f32")
    report, code = cm.compare(a, b, policy)
    assert code == 1
    assert report["rows"][0]["near_tie"]
    assert report["rows"][0]["numerical_pass"]
    assert report["top1_agreement"] == 0.5
    assert not report["model_logits_pass"]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_late_row_fails_even_with_unchanged_top_token(evidence, value):
    a, b, policy = evidence
    np.array([4, 0, 1, value], dtype="<f4").tofile(b.parent / "logits-1.f32")
    report, code = cm.compare(a, b, policy)
    assert code == 1 and not report["model_logits_pass"]
    assert not report["rows"][1]["finite"]
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("bad_header", [(4, 3), (5, 2), (4, 0)])
def test_original_dump_old_off_by_one_count_is_rejected(evidence, bad_header):
    a, _, _ = evidence
    path = a.parent / "baseline.bin"
    path.write_bytes(struct.pack("<ii", *bad_header) + path.read_bytes()[8:])
    with pytest.raises(cm.EvidenceError, match="header shape"):
        cm.compare(*evidence)


@pytest.mark.parametrize("filename", ["baseline.bin", "logits-1.f32"])
@pytest.mark.parametrize("delta", [-1, 1])
def test_truncation_and_extra_bytes_are_never_silently_ignored(evidence, filename, delta):
    path = evidence[0].parent / filename
    payload = path.read_bytes()
    path.write_bytes(payload[:-1] if delta < 0 else payload + b"\0")
    with pytest.raises(cm.EvidenceError, match="size|oversized"):
        cm.compare(*evidence)


@pytest.mark.parametrize("field,value", [("commit", "c" * 40), ("pack_sha256", "c" * 64),
                                         ("token_ids", {"request-A": [1, 3]}),
                                         ("n_vocab", 5)])
def test_identity_mismatch_is_rejected(evidence, field, value):
    a, b, policy = evidence
    change(b, lambda m: m.update({field: value}))
    with pytest.raises(cm.EvidenceError):
        cm.compare(a, b, policy, product_comparison=True)


@pytest.mark.parametrize("field,value", [("settings", {"context": 8192}), ("backend", "cpu"), ("mode", "prefill")])
def test_settings_need_explicit_product_comparison(evidence, field, value):
    a, b, policy = evidence
    change(b, lambda m: m.update({field: value}))
    with pytest.raises(cm.EvidenceError, match="mismatched settings"):
        cm.compare(a, b, policy)
    report, _ = cm.compare(a, b, policy, product_comparison=True)
    assert report["comparison"] == "product" and report["metadata_differences"] == [field]


def test_duplicate_positions_rejected(evidence):
    change(evidence[1], lambda m: m["rows"].append(m["rows"][0]))
    with pytest.raises(cm.EvidenceError, match="duplicate request position"):
        cm.compare(*evidence)


def test_reordered_rows_match_by_request_position(evidence):
    change(evidence[1], lambda m: m["rows"].reverse())
    report, _ = cm.compare(*evidence)
    assert report["model_logits_pass"]


@pytest.mark.parametrize("mutation", ["empty", "late", "missing_threshold", "nan_threshold", "hash"])
def test_policy_cannot_be_empty_incomplete_or_changed_after_run(evidence, mutation):
    a, b, policy = evidence
    if mutation == "empty":
        policy.write_text("{}")
    elif mutation == "late":
        change(policy, lambda p: p.update(frozen_at="2026-10-31T00:00:00Z"))
    elif mutation == "missing_threshold":
        change(policy, lambda p: p["thresholds"].pop("relative_l2"))
    elif mutation == "nan_threshold":
        change(policy, lambda p: p["thresholds"].update(max_abs=float("nan")))
    else:
        change(b, lambda p: p.update(policy_sha256="e" * 64))
    if mutation != "hash":
        for path in (a, b):
            change(path, lambda p: p.update(policy_sha256=cm.sha256(policy)))
    with pytest.raises(cm.EvidenceError):
        cm.compare(a, b, policy)


def test_state_shape_identity_and_coverage_are_checked(evidence):
    a, b, policy = evidence
    add_states(a, b)
    change(b, lambda m: m["states"][0].update(shape=[4]))
    with pytest.raises(cm.EvidenceError, match="state dtype/shape"):
        cm.compare(a, b, policy)
    change(b, lambda m: m["states"][0].update(shape=[2, 2], name="other-layer"))
    with pytest.raises(cm.EvidenceError, match="state identities"):
        cm.compare(a, b, policy)


def test_state_arrays_without_frozen_inventory_cannot_qualify(evidence):
    a, b, policy = evidence
    add_states(a, b)
    change(policy, lambda p: p.pop("required_states"))
    for path in (a, b):
        change(path, lambda m: m.update(policy_sha256=cm.sha256(policy)))
    report, code = cm.compare(a, b, policy)
    assert code == 77 and report["state_validation"] == "missing"
    assert not report["full_model_pass"] and not report["qualified"]


def test_missing_one_expected_layer_state_does_not_complete_gate(evidence):
    a, b, policy = evidence
    add_states(a, b)
    change(policy, lambda p: p["required_states"].append(
        {"kind": "gdn", "name": "layer.1", "dtype": "<f2", "shape": [2, 2]}))
    for path in (a, b):
        change(path, lambda m: m.update(policy_sha256=cm.sha256(policy)))
    report, code = cm.compare(a, b, policy)
    assert code == 77 and len(report["missing_states"]) == 2


def test_empty_rows_do_not_pass_vacuously(evidence):
    change(evidence[1], lambda m: m.update(rows=[]))
    with pytest.raises(cm.EvidenceError, match="empty logits"):
        cm.compare(*evidence)


def test_request_identity_cannot_be_replaced_by_same_position(evidence):
    def swap(meta):
        meta["token_ids"] = {"other-request": [1, 2]}
        for row in meta["rows"]:
            row["request_id"] = "other-request"
    change(evidence[1], swap)
    with pytest.raises(cm.EvidenceError, match="token_ids"):
        cm.compare(*evidence)


def test_cli_writes_invalid_evidence_receipt(evidence, tmp_path):
    a, b, policy = evidence
    b.write_text('{"schema_version":1,"schema_version":1}')
    output = tmp_path / "receipt.json"
    code = cm.main([str(a), str(b), "--policy", str(policy), "--output", str(output)])
    report = json.loads(output.read_text())
    assert code == 2 and not report["qualified"]
    assert "duplicate JSON key" in report["error"]
    assert report["comparison_command"]
