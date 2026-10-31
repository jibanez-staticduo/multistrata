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


def test_explicit_revision_pair_preserves_real_provenance_and_numerical_gate(evidence):
    a, b, policy = evidence
    add_states(a, b)
    change(b, lambda m: m.update(commit="c" * 40))
    report, code = cm.compare(a, b, policy, expected_revisions=("a" * 40, "c" * 40))
    assert code == 0 and report["full_model_pass"]
    assert report["approved_revision_pair"] == {"baseline": "a" * 40, "candidate": "c" * 40}
    assert report["provenance"]["candidate"]["commit"] == "c" * 40
    np.array([1, 2, 3, 8], dtype="<f2").tofile(a.parent / "state-1-gdn-1.bin")
    report, code = cm.compare(a, b, policy, expected_revisions=("a" * 40, "c" * 40))
    assert code == 1 and not report["full_model_pass"]


@pytest.mark.parametrize("pair", [("a" * 40, "d" * 40), ("HEAD", "c" * 40), ("a" * 40,)])
def test_revision_pair_must_name_both_exact_actual_revisions(evidence, pair):
    a, b, policy = evidence
    change(b, lambda m: m.update(commit="c" * 40))
    with pytest.raises(cm.EvidenceError):
        cm.compare(a, b, policy, expected_revisions=pair)


def test_revision_pair_does_not_allow_different_pack(evidence):
    a, b, policy = evidence
    change(b, lambda m: m.update(commit="c" * 40, pack_sha256="d" * 64))
    with pytest.raises(cm.EvidenceError, match="pack_sha256"):
        cm.compare(a, b, policy, expected_revisions=("a" * 40, "c" * 40))


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


def freeze_inventory(evidence, descriptor):
    a, b, policy = evidence
    change(policy, lambda p: p["required_states"].append(descriptor))
    for path in (a, b):
        change(path, lambda m: m.update(policy_sha256=cm.sha256(policy)))


@pytest.mark.parametrize("kind,dtype", [("metadata", "<i8"), ("indexer", "<i4"), ("ple", "|u1"),
                                        ("residual", "<f4")])
def test_extra_frozen_kinds_complete_and_integer_values_compare_exactly(evidence, kind, dtype):
    a, b, _ = evidence
    add_states(a, b)
    descriptor = dict(kind=kind, name="extra", dtype=dtype, shape=[2])
    freeze_inventory(evidence, descriptor)
    base = 2**60 if dtype == "<i8" else 4
    for side, path in enumerate((a, b)):
        def add(meta):
            for row in meta["rows"]:
                name = f"extra-{side}-{row['position']}.bin"
                np.array([base, base + 1], dtype=dtype).tofile(path.parent / name)
                meta["states"].append(dict(descriptor, request_id=row["request_id"], position=row["position"], path=name))
        change(path, add)
    report, code = cm.compare(*evidence)
    assert code == 0 and report["full_model_pass"]
    if dtype != "<f4":
        np.array([base, base + 2], dtype=dtype).tofile(b.parent / "extra-1-1.bin")
        report, code = cm.compare(*evidence)
        assert code == 1
        extra = [s for s in report["states"] if s["kind"] == kind and s["position"] == 1][0]
        assert extra["comparison"] == "integer_exact" and not extra["exact_equal"]
        assert "mean_abs" not in extra


def test_unfrozen_extra_array_cannot_claim_full_gate(evidence):
    a, b, _ = evidence
    add_states(a, b)
    for path in (a, b):
        change(path, lambda m: m["states"].append(dict(m["states"][0], kind="ple", name="extra")))
    report, code = cm.compare(*evidence)
    assert code == 77 and report["unfrozen_states"]


@pytest.mark.parametrize("shape", [[2**100], [2**62, 8], [1] * 33, [0], [-1]])
def test_huge_or_invalid_shapes_rejected_before_allocation(tmp_path, shape):
    path = tmp_path / "tiny.bin"
    path.write_bytes(b"\0\0\0\0")
    with pytest.raises(cm.EvidenceError, match="shape"):
        cm.raw(path, "<f4", shape, {})


def make_manifest_shards(evidence, *, ranks=(0, 1, 2), overlap=False):
    a, b, _ = evidence
    add_states(a, b)
    descriptor = dict(kind="residual", name="hc", dtype="<f4", shape=[2, 3], layout="logical", layer=-1)
    freeze_inventory(evidence, descriptor)
    for side, path in enumerate((a, b)):
        manifests = []
        for row in json.loads(path.read_text())["rows"]:
            for rank in ranks:
                directory = path.parent / f"manifest-{side}-{row['position']}-{rank}"
                directory.mkdir()
                indices = [0 if overlap and rank == 1 else rank]
                desc = dict(descriptor, shape=[2, 1], layout="axis_shard", logical_shape=[2, 3],
                            shard_axis=1, axis_indices=indices)
                np.array([[rank], [rank + 3]], dtype="<f4").tofile(directory / "state-0.bin")
                state = dict(desc, request_id=row["request_id"], position=row["position"], path="state-0.bin")
                manifest = dict(schema_version=1, complete=True, logical_complete=False,
                                request_id=row["request_id"], position=row["position"],
                                token_count=row["position"] + 1, total_bytes=8,
                                required_states=[desc], states=[state])
                (directory / "manifest.json").write_text(json.dumps(manifest))
                manifests.append(dict(rank=rank, path=str(directory.relative_to(path.parent) / "manifest.json")))
        change(path, lambda m: m.update(state_manifests=manifests))


def test_helper_manifest_reconstructs_three_rank_axis_with_correct_values(evidence):
    make_manifest_shards(evidence)
    report, code = cm.compare(*evidence)
    assert code == 0 and report["full_model_pass"]
    assert not any(report["partial_states"].values())
    policy = cm.read_policy(evidence[2])
    loaded = cm.load(evidence[1], evidence[2], policy, {})
    np.testing.assert_array_equal(loaded["states"][("request-A", 0, "residual", "hc")][2],
                                  [[0, 1, 2], [3, 4, 5]])


def test_two_rank_state_snapshot_stays_partial(evidence):
    make_manifest_shards(evidence, ranks=(0, 1))
    report, code = cm.compare(*evidence)
    assert code == 77 and not report["full_model_pass"]
    assert report["partial_states"]["candidate"][0]["ranks"] == [0, 1]


def test_overlapping_three_rank_state_is_invalid(evidence):
    make_manifest_shards(evidence, overlap=True)
    with pytest.raises(cm.EvidenceError, match="overlapping"):
        cm.compare(*evidence)


def test_reset_manifest_cannot_collide_with_first_token_position(evidence):
    make_manifest_shards(evidence)
    manifest = evidence[1].parent / "manifest-1-0-0" / "manifest.json"
    change(manifest, lambda m: m.update(phase="reset", token_count=0))
    with pytest.raises(cm.EvidenceError, match="separate reset gate"):
        cm.compare(*evidence)


def test_float_state_nan_fails_full_gate(evidence):
    add_states(*evidence[:2])
    np.array([1, 2, 3, np.nan], dtype="<f2").tofile(evidence[1].parent / "state-1-gdn-1.bin")
    report, code = cm.compare(*evidence)
    assert code == 1 and report["state_validation"] == "failed"


def make_dynamic_kv(evidence, *, last_position=8191, manifests=False, wrong_length=False):
    a, b, policy = evidence
    change(policy, lambda p: p["required_states"][0].update(shape=["{token_count}", 2]))
    for side, path in enumerate((a, b)):
        meta = json.loads(path.read_text())
        meta.update(format="raw_f32_rows", policy_sha256=cm.sha256(policy),
                    token_ids={"request-A": [1] * (last_position + 1)}, states=[], state_manifests=[])
        meta["settings"]["context"] = last_position + 1
        for row, position in zip(meta["rows"], (0, last_position)):
            row.update(position=position, path=f"dynamic-logits-{side}-{position}.bin")
            np.array([1, 3, 2, -1], dtype="<f4").tofile(path.parent / row["path"])
            token_count = position + 1
            length = token_count + (1 if wrong_length else 0)
            states = []
            for kind, shape in (("kv", [length, 2]), ("gdn", [2, 2])):
                filename = f"dynamic-state-{side}-{kind}-{position}.bin"
                np.ones(shape, dtype="<f2").tofile(path.parent / filename)
                states.append(dict(request_id=row["request_id"], position=position, kind=kind,
                                   name="layer.0", dtype="<f2", shape=shape, path=filename))
            if manifests:
                filename = f"dynamic-manifest-{side}-{position}.json"
                manifest = dict(schema_version=1, complete=True, logical_complete=True,
                                request_id=row["request_id"], position=position, token_count=token_count,
                                total_bytes=sum(cm.shape_size(s["shape"], s["dtype"]) for s in states),
                                required_states=[{k: v for k, v in s.items()
                                                  if k not in ("path", "request_id", "position")}
                                                 for s in states], states=states)
                (path.parent / filename).write_text(json.dumps(manifest))
                meta["state_manifests"].append(dict(rank=0, path=filename))
            else:
                meta["states"].extend(states)
        path.write_text(json.dumps(meta))


@pytest.mark.parametrize("last_position", [8191, 65535])
@pytest.mark.parametrize("manifests", [False, True])
def test_frozen_kv_token_count_shape_supports_sparse_long_context_rows(evidence, last_position, manifests):
    make_dynamic_kv(evidence, last_position=last_position, manifests=manifests)
    report, code = cm.compare(*evidence)
    assert code == 0 and report["full_model_pass"]
    assert [row["position"] for row in report["rows"]] == [0, last_position]
    assert report["policy"]["required_states"][0]["shape"] == ["{token_count}", 2]


@pytest.mark.parametrize("manifests", [False, True])
def test_dynamic_kv_length_must_equal_row_position_plus_one_on_both_sides(evidence, manifests):
    make_dynamic_kv(evidence, manifests=manifests, wrong_length=True)
    with pytest.raises(cm.EvidenceError, match="state dtype/shape differs from frozen inventory"):
        cm.compare(*evidence)


@pytest.mark.parametrize("kind,shape", [("gdn", ["{token_count}", 2]),
                                        ("kv", [2, "{token_count}"]),
                                        ("kv", ["token_count", 2]),
                                        ("kv", ["{token_count + 1}", 2]),
                                        ("kv", ["{position}", 2])])
def test_dynamic_shape_symbol_is_restricted_to_kv_first_axis(evidence, kind, shape):
    change(evidence[2], lambda p: p["required_states"][0].update(kind=kind, shape=shape))
    with pytest.raises(cm.EvidenceError, match="shape"):
        cm.read_policy(evidence[2])


def test_missing_dynamic_snapshot_cannot_claim_full_gate(evidence):
    make_dynamic_kv(evidence, manifests=True)
    for path in evidence[:2]:
        change(path, lambda m: m["state_manifests"].pop())
    report, code = cm.compare(*evidence)
    assert code == 77 and not report["full_model_pass"]
    assert len(report["missing_states"]) == 2


def test_dynamic_manifest_token_count_cannot_override_row_position(evidence):
    make_dynamic_kv(evidence, manifests=True)
    manifest = evidence[1].parent / "dynamic-manifest-1-8191.json"
    change(manifest, lambda m: m.update(token_count=8193))
    with pytest.raises(cm.EvidenceError, match="state manifest identity mismatch"):
        cm.compare(*evidence)


@pytest.mark.parametrize("token_count", [True, 1.0])
def test_manifest_token_count_requires_an_explicit_integer(evidence, token_count):
    make_dynamic_kv(evidence, manifests=True)
    manifest = evidence[1].parent / "dynamic-manifest-1-0.json"
    change(manifest, lambda m: m.update(token_count=token_count))
    with pytest.raises(cm.EvidenceError, match="state manifest identity mismatch"):
        cm.compare(*evidence)


@pytest.mark.parametrize("ranks,expected_code", [((0, 1, 2), 0), ((0, 1), 77)])
def test_dynamic_kv_shards_require_complete_three_rank_coverage(evidence, ranks, expected_code):
    make_dynamic_kv(evidence, manifests=True)
    change(evidence[2], lambda p: p["required_states"][0].update(shape=["{token_count}", 3]))
    for side, path in enumerate(evidence[:2]):
        meta = json.loads(path.read_text())
        meta.update(policy_sha256=cm.sha256(evidence[2]), state_manifests=[])
        for position in (0, 8191):
            original = json.loads((path.parent / f"dynamic-manifest-{side}-{position}.json").read_text())
            for rank in ranks:
                manifest = copy.deepcopy(original)
                kv = manifest["states"][0]
                kv.update(shape=[position + 1, 1], layout="axis_shard", logical_shape=[position + 1, 3],
                          shard_axis=1, axis_indices=[rank], path=f"dynamic-shard-{side}-{position}-{rank}.bin")
                np.ones(kv["shape"], dtype=kv["dtype"]).tofile(path.parent / kv["path"])
                manifest.update(logical_complete=False,
                                total_bytes=sum(cm.shape_size(s["shape"], s["dtype"]) for s in manifest["states"]),
                                required_states=[{k: v for k, v in s.items()
                                                  if k not in ("path", "request_id", "position")}
                                                 for s in manifest["states"]])
                filename = f"dynamic-shard-manifest-{side}-{position}-{rank}.json"
                (path.parent / filename).write_text(json.dumps(manifest))
                meta["state_manifests"].append(dict(rank=rank, path=filename))
        path.write_text(json.dumps(meta))
    report, code = cm.compare(*evidence)
    assert code == expected_code
    assert report["full_model_pass"] is (expected_code == 0)
    if expected_code == 77:
        assert all(state["ranks"] == [0, 1] for state in report["partial_states"]["candidate"])
