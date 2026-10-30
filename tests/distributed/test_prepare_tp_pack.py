"""Artifact tests for ``tools/distributed/prepare_tp_pack.py`` (Strata TP3, Task 3).

Run:  $VENV/bin/python -m pytest tests/distributed/test_prepare_tp_pack.py -q
Venv: /home/staticduo/tmp/strata-tp3-20261030/venv (pytest 9.2.0, numpy 2.5.4)

CTest wiring the integrator owns (root CMakeLists.txt is not edited here):
    add_test(NAME tp_prepare_tp_pack_py
             COMMAND ${Python3_EXECUTABLE} -m pytest
                     ${CMAKE_SOURCE_DIR}/tests/distributed/test_prepare_tp_pack.py -q)

Every fixture here is synthetic: small files with the geometry the plan contract uses (FFN
640 wide, cut 256/192/192, block sizes 32 and 64 in the same layer).  A real source is the
same arithmetic over bigger files, so the byte reconstruction these tests prove is the same
code path a real pack goes through.  No real weights are downloaded by these tests.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import sys

REPO = Path(__file__).resolve().parents[2]
sys_path_target = str(REPO / "tools" / "distributed")
if sys_path_target not in sys.path:
    sys.path.insert(0, sys_path_target)

import prepare_tp_pack as ptp   # noqa: E402  (needs the path insert above)

GIB = 1024**3

# Layer 7 as the fixture: two different block codecs side by side, plus a separate scale tensor.
GATE_UP = "blk.7.ffn_gate_up.weight"
GATE_UP_SCALES = "blk.7.ffn_gate_up.scales"
DOWN = "blk.7.ffn_down.weight"
DOWN_SCALES = "blk.7.ffn_down.scales"
NORM = "blk.7.attn_norm.weight"
EMBED = "token_embd.weight"

FFN_WIDTH = 640
GATE_UP_BLOCK = 32
GATE_UP_BLOCK_BYTES = 44        # 32 codes + a 12-byte native scale tail, inside the block
GATE_UP_ROWS = 128
DOWN_BLOCK = 64
DOWN_BLOCK_BYTES = 100          # 64 codes + 36 bytes of scales and scales-of-scales tail
DOWN_ROWS = 128
DOWN_HEADER = 64                # the down file carries a header, so its tensors do not start at 0
NORM_ITEMS = 2560
VOCAB = 96
EMBED_WIDTH = 16


def _bytes(seed: int, length: int) -> bytes:
    return np.random.default_rng(seed).integers(0, 256, length, dtype=np.uint8).tobytes()


def _entry(name: str, file: str, dtype: str, block, shape: list[int], offset: int, length: int,
           family: str) -> dict:
    return {"name": name, "file": file, "dtype": dtype, "shape": shape, "block_size": block,
            "scale_kind": None if block is None else "per-block", "byte_offset": offset,
            "byte_length": length, "family": family}


class Source:
    """A synthetic pinned source: files on disk plus the inventory that describes them."""

    def __init__(self, tmp_path: Path):
        self.root = tmp_path / "src"
        self.root.mkdir()
        self.data: dict[str, bytes] = {}

        gate_up = _bytes(1, GATE_UP_ROWS * (FFN_WIDTH // GATE_UP_BLOCK) * GATE_UP_BLOCK_BYTES)
        gate_up_scales = _bytes(2, GATE_UP_ROWS * (FFN_WIDTH // GATE_UP_BLOCK) * 4)
        down_body = _bytes(3, DOWN_ROWS * (FFN_WIDTH // DOWN_BLOCK) * DOWN_BLOCK_BYTES)
        down_scales = _bytes(4, DOWN_ROWS * (FFN_WIDTH // DOWN_BLOCK) * 4)
        norm = _bytes(5, NORM_ITEMS * 2)
        embed = _bytes(6, VOCAB * EMBED_WIDTH * 2)
        down = b"\0" * DOWN_HEADER + down_body     # a file whose tensor does not start at byte 0

        self.files = {
            "layer7_gate_up.bin": gate_up,
            "layer7_gate_up_scales.bin": gate_up_scales,
            "layer7_down.bin": down,
            "layer7_down_scales.bin": down_scales,
            "layer7_norm.bin": norm,
            "embed.bin": embed,
        }
        self.data = dict(self.files)
        for name, blob in self.files.items():
            (self.root / name).write_bytes(blob)

        # One file is left without a recorded hash on purpose: a pack must say so rather than
        # pretend every source byte was hashed.
        self.signed = {name: (None if name == "layer7_norm.bin"
                              else hashlib.sha256(blob).hexdigest())
                       for name, blob in self.files.items()}

        self.tensors = [
            _entry(GATE_UP, "layer7_gate_up.bin", "q4_k", GATE_UP_BLOCK, [GATE_UP_ROWS, FFN_WIDTH], 0,
                   len(gate_up), "gate_up"),
            _entry(GATE_UP_SCALES, "layer7_gate_up_scales.bin", "f16", None,
                   [GATE_UP_ROWS, FFN_WIDTH // GATE_UP_BLOCK], 0, len(gate_up_scales), "other"),
            _entry(DOWN, "layer7_down.bin", "iq3_s", DOWN_BLOCK, [DOWN_ROWS, FFN_WIDTH], DOWN_HEADER,
                   len(down_body), "down"),
            _entry(DOWN_SCALES, "layer7_down_scales.bin", "f16", None,
                   [DOWN_ROWS, FFN_WIDTH // DOWN_BLOCK], 0, len(down_scales), "other"),
            _entry(NORM, "layer7_norm.bin", "f16", None, [NORM_ITEMS], 0, len(norm), "norm"),
            _entry(EMBED, "embed.bin", "f16", None, [VOCAB, EMBED_WIDTH], 0, len(embed), "embed"),
        ]

    def inventory(self) -> dict:
        return {
            "schema_version": 1,
            "source": {"kind": "local", "repo": "ISTA-DASLab/synthetic-fixture", "revision": "rev-0001",
                       "files": [{"path": name, "size": len(blob), "sha256": self.signed[name]}
                                 for name, blob in self.files.items()]},
            "geometry": {"n_embd": 2560, "n_layers": 48, "qsa_interval": 4, "n_vocab": VOCAB,
                         "n_ff": FFN_WIDTH, "n_expert": 512, "n_head": 24, "n_head_kv": 2},
            "tensors": copy.deepcopy(self.tensors),
            "activation_contracts": [{"consumer": "ffn_down", "quantizer": "q8_k", "block_size": 256,
                                      "scope": "global", "notes": "fixture"}],
            "assets": {"tokenizer": "none", "draft": "none", "ple": "none"},
            "bytes": {"total": sum(len(b) for b in self.files.values()), "by_family": {}},
            "verdict": {"loadable_by_strata": True, "reasons": ["fixture"], "checked_tensors": len(self.tensors)},
        }

    def write_inventory(self, name: str = "inventory.json") -> Path:
        """Write the inventory next to the files it names: its paths are inventory-relative."""
        path = self.root / name
        path.write_text(json.dumps(self.inventory(), indent=2) + "\n")
        return path

    def tensor(self, name: str) -> dict:
        return next(entry for entry in self.tensors if entry["name"] == name)

    def file_of(self, name: str) -> str:
        return self.tensor(name)["file"]

    def raw(self, name: str) -> bytes:
        return (self.root / self.file_of(name)).read_bytes()


def _read_runs(path: Path, runs) -> bytes:
    """Concatenate the manifest's runs for one tensor, read straight out of a rank file."""
    out = bytearray()
    with open(path, "rb") as handle:
        for segment in sorted(runs, key=lambda seg: seg["dest_offset"]):
            handle.seek(segment["dest_offset"])
            out += handle.read(segment["source_length"])
    return bytes(out)


def _slice(tensor: str, rank: int, axis: int, start: int, end: int, family: str, **extra) -> dict:
    entry = {"tensor": tensor, "family": family, "global_shape": None, "axis": axis,
             "range": [start, end], "byte_row_stride": None, "codec": None, "block_size": None,
             "scale_ref": None, "valid_lanes": None, "halo": [0, 0], "replicated": False, "rank": rank}
    entry.update(extra)
    entry.pop("global_shape")
    entry.pop("codec")
    entry.pop("block_size")
    return entry


def base_plan() -> dict:
    """The agreed TP3 cut: FFN 256/192/192 at starts 0/256/448, norms and vocab handled per rule."""
    slices: list[dict] = []
    for rank, (start, end) in enumerate([(0, 256), (256, 448), (448, 640)]):
        slices.append(_slice(GATE_UP, rank, 1, start, end, "gate_up",
                             local_shape=[GATE_UP_ROWS, end - start], scale_ref=GATE_UP_SCALES))
        # rank 2 keeps 128 of its 192 items and pads the rest: a gather to uniform slots.
        # valid_lanes counts items, in the same units as the range, so 128 items = 2 blocks of 64.
        lanes = [192, 192, 128]
        slices.append(_slice(DOWN, rank, 1, start, end, "down",
                             local_shape=[DOWN_ROWS, end - start], scale_ref=DOWN_SCALES,
                             valid_lanes=None if rank != 2 else lanes))
    slices.append(_slice(NORM, 0, 0, 0, NORM_ITEMS, "norm", local_shape=[NORM_ITEMS], replicated=True))
    del slices[-1]["rank"]
    slices[-1]["valid_lanes"] = None
    for rank, (start, end) in enumerate([(0, 32), (32, 64), (64, 96)]):
        slices.append(_slice(EMBED, rank, 0, start, end, "embed", local_shape=[end - start, EMBED_WIDTH]))
    return {
        "schema_version": 1,
        "plan_hash": "0" * 64,
        "world": 3,
        "source_lock": {"repo": "ISTA-DASLab/synthetic-fixture", "revision": "rev-0001",
                        "inventory_hash": None},
        "geometry": {"n_embd": 2560, "n_layers": 48, "qsa_interval": 4, "n_vocab": VOCAB, "n_ff": FFN_WIDTH,
                     "n_expert": 512, "n_head": 24, "n_head_kv": 2},
        "memory": {"per_rank_bytes": 0, "components": {"gate_up": 1, "down": 1}, "halo_and_global_buffers": 0},
        "slices": slices,
        "activation": [{"consumer": "ffn_down", "gather": "all_gather_padded", "wire_slots": 192,
                        "valid_counts": [256, 192, 192], "quantizer": "q8_k", "block_size": 256,
                        "crosses_rank_boundary": True, "reference": "global_reference"}],
    }


def _pin(source: Source, plan: dict) -> dict:
    plan["source_lock"]["inventory_hash"] = ptp.inventory_hash(source.inventory())
    return plan


def _plan_with(source: Source, **mutate) -> dict:
    plan = _pin(source, base_plan())
    for key, value in mutate.items():
        if key == "slices":
            plan["slices"] = value(plan["slices"])
        else:
            plan[key] = value
    return plan


@pytest.fixture()
def source(tmp_path: Path) -> Source:
    return Source(tmp_path)


# ------------------------------------------------------------------ plan mode, no writes


def test_plan_mode_writes_nothing_at_all(source: Source, tmp_path: Path):
    out = tmp_path / "pack"
    manifest = ptp.prepare_tp_pack(source.write_inventory(), _plan_with(source), out)
    assert not out.exists(), "plan mode must not create the output directory"
    assert [item.name for item in tmp_path.iterdir()] == ["src"], "plan mode wrote somewhere else"
    assert manifest["materialized"] is False
    assert manifest["rank_files"] is None
    assert manifest["manifest_version"] == 1
    assert manifest["vocabulary"]["logical_n_vocab"] == VOCAB
    assert sum(int(value) for value in manifest["rank_bytes"].values()) > 0
    assert manifest["not_verified"], "plan mode must say what it did not check"
    # the views are the artifact: they point back into the original files, at real offsets
    view = ptp.rank_view(manifest, 0)
    assert view and view == sorted(view, key=lambda seg: seg["dest_offset"])
    sizes = {entry["path"]: entry["size"] for entry in manifest["source_files"]}
    for segment in manifest["segments"]:
        if segment["source_path"] is not None:
            assert segment["source_offset"] + segment["source_length"] <= sizes[segment["source_path"]]


def test_plan_mode_never_opens_the_output(tmp_path: Path, source: Source, monkeypatch):
    real_open = open
    opened: list[str] = []

    def spy(file, *args, **kwargs):
        opened.append(str(file))
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(ptp, "open", spy, raising=False)
    ptp.prepare_tp_pack(source.write_inventory(), _plan_with(source), tmp_path / "pack")
    assert not [path for path in opened if "pack" in path], "plan mode opened an output file"


def test_manifest_carries_the_lock_and_the_original_codecs(source: Source, tmp_path: Path):
    manifest = ptp.prepare_tp_pack(source.write_inventory(), _plan_with(source), tmp_path / "pack")
    assert manifest["plan_content_hash"] == ptp.plan_content_hash(_plan_with(source))
    assert manifest["inventory_hash"] == ptp.inventory_hash(source.inventory())
    assert manifest["activation"][0]["gather"] == "all_gather_padded"
    codecs = {seg["tensor"] for seg in manifest["segments"]}
    assert {GATE_UP, DOWN, GATE_UP_SCALES, DOWN_SCALES, NORM, EMBED} <= codecs
    by_tensor = {entry["name"]: entry for entry in source.tensors}
    assert manifest["segments"], "the manifest must carry the ranges a loader reads"
    for segment in manifest["segments"]:
        if segment["kind"] != "codes":
            continue
        entry = by_tensor[segment["tensor"]]
        if entry["block_size"] is None:
            continue                        # an f16 norm has no block to land on
        assert segment["source_length"] % _block_bytes(entry, entry["block_size"]) == 0


def _block_bytes(tensor: dict, block_size) -> int:
    rows = tensor["shape"][0] if len(tensor["shape"]) > 1 else 1
    last = tensor["shape"][-1]
    return tensor["byte_length"] // (rows * (last // block_size))


# ------------------------------------------------- materialize + independent reconstruction


def test_materialize_reconstructs_blocks_and_scales_byte_for_byte(source: Source, tmp_path: Path):
    out = tmp_path / "pack"
    manifest = ptp.prepare_tp_pack(source.write_inventory(), _plan_with(source), out, materialize=True)
    assert manifest["materialized"] is True
    assert (out / ptp.MANIFEST_NAME).is_file()
    assert manifest["receipts"]["byte_equality"]["status"] == "verified"

    gate_unit = _block_bytes(source.tensor(GATE_UP), GATE_UP_BLOCK)
    down_unit = _block_bytes(source.tensor(DOWN), DOWN_BLOCK)
    cases = [
        # rank, tensor, first block, block count -> expected = whole rows of whole blocks, tail inside
        (0, GATE_UP, 0, 8, GATE_UP_BLOCK_BYTES),
        (1, GATE_UP, 8, 6, GATE_UP_BLOCK_BYTES),
        (2, GATE_UP, 14, 6, GATE_UP_BLOCK_BYTES),
        (0, DOWN, 0, 4, DOWN_BLOCK_BYTES),
        (1, DOWN, 4, 3, DOWN_BLOCK_BYTES),
    ]
    for rank, tensor, first_block, count, unit in cases:
        assert unit == (gate_unit if tensor == GATE_UP else down_unit)
        rows = source.tensor(tensor)["shape"][0]
        stride = source.tensor(tensor)["byte_length"] // rows
        base = source.tensor(tensor)["byte_offset"]
        source_bytes = (source.root / source.file_of(tensor)).read_bytes()
        expected = b"".join(source_bytes[base + row * stride + first_block * unit:
                                         base + row * stride + (first_block + count) * unit]
                            for row in range(rows))
        runs = [seg for seg in manifest["segments"] if seg["rank"] == rank and seg["tensor"] == tensor
                and seg["kind"] == "codes"]
        assert sum(seg["source_length"] for seg in runs) == len(expected)
        got = _read_runs(out / "ranks" / f"rank{rank}.bin", runs)
        assert got == expected, f"rank {rank} {tensor} does not rebuild from the source"

    # the separate scale tensors: one scale row per block, copied for exactly the owned blocks
    scale_unit = source.tensor(GATE_UP_SCALES)["byte_length"] // (GATE_UP_ROWS * (FFN_WIDTH // GATE_UP_BLOCK))
    expected = b"".join(source.raw(GATE_UP_SCALES)[row * (FFN_WIDTH // GATE_UP_BLOCK * scale_unit):
                                                  row * (FFN_WIDTH // GATE_UP_BLOCK * scale_unit) + 8 * scale_unit]
                       for row in range(GATE_UP_ROWS))
    runs = [seg for seg in manifest["segments"]
            if seg["rank"] == 0 and seg["tensor"] == GATE_UP_SCALES and seg["kind"] == "scales"]
    assert _read_runs(out / "ranks" / "rank0.bin", runs) == expected


def test_native_scale_tails_stay_inside_the_copied_block(source: Source, tmp_path: Path):
    manifest = ptp.prepare_tp_pack(source.write_inventory(), _plan_with(source), tmp_path / "pack")
    gate = source.tensor(GATE_UP)
    unit = _block_bytes(gate, GATE_UP_BLOCK)
    assert unit == GATE_UP_BLOCK_BYTES, "the fixture block must carry its scale tail"
    for segment in manifest["segments"]:
        if segment["kind"] == "codes" and segment["tensor"] == GATE_UP:
            assert segment["source_length"] % unit == 0
        if segment["kind"] == "codes" and segment["tensor"] == DOWN:
            assert segment["source_length"] % DOWN_BLOCK_BYTES == 0


def test_mixed_quant_types_in_one_layer_carry_their_own_block_size(source: Source, tmp_path: Path):
    out = tmp_path / "pack"
    manifest = ptp.prepare_tp_pack(source.write_inventory(), _plan_with(source), out, materialize=True)
    lengths = {}
    for segment in manifest["segments"]:
        if segment["kind"] == "codes" and segment["rank"] == 1:
            lengths.setdefault(segment["tensor"], set()).add(segment["source_length"])
    assert lengths[GATE_UP] == {6 * GATE_UP_BLOCK_BYTES}, "block 32 share of 192 items"
    assert lengths[DOWN] == {3 * DOWN_BLOCK_BYTES}, "block 64 share of 192 items, same layer"
    report = ptp.verify_tp_pack(source.write_inventory(), out)
    assert report.ok, report.mismatches
    assert report.bytes_compared > 0


def test_replicated_tensor_lands_whole_on_every_rank(source: Source, tmp_path: Path):
    out = tmp_path / "pack"
    manifest = ptp.prepare_tp_pack(source.write_inventory(), _plan_with(source), out, materialize=True)
    wanted = source.raw(NORM)
    for rank in range(3):
        runs = [seg for seg in manifest["segments"] if seg["rank"] == rank and seg["tensor"] == NORM]
        assert len(runs) == 1 and runs[0]["source_length"] == len(wanted)
        with open(out / "ranks" / f"rank{rank}.bin", "rb") as handle:
            handle.seek(runs[0]["dest_offset"])
            assert handle.read(len(wanted)) == wanted


def test_padding_is_zeroed_and_never_read_from_the_source(source: Source, tmp_path: Path):
    out = tmp_path / "pack"
    manifest = ptp.prepare_tp_pack(source.write_inventory(), _plan_with(source), out, materialize=True)
    pads = [seg for seg in manifest["segments"] if seg["kind"] == "padding"]
    assert [seg["rank"] for seg in pads] == [2], "only rank 2 has a short share in this plan"
    pad = pads[0]
    assert pad["source_path"] is None and pad["source_length"] == DOWN_BLOCK_BYTES
    # rank 2 owns two valid blocks of 192 items, not three
    codes = [seg for seg in manifest["segments"]
             if seg["rank"] == 2 and seg["tensor"] == DOWN and seg["kind"] == "codes"]
    assert sum(seg["source_length"] for seg in codes) == 2 * DOWN_BLOCK_BYTES * DOWN_ROWS
    with open(out / "ranks" / "rank2.bin", "rb") as handle:
        handle.seek(pad["dest_offset"])
        assert handle.read(pad["source_length"]) == b"\0" * pad["source_length"]
    report = ptp.verify_tp_pack(source.write_inventory(), out)
    assert report.ok, report.mismatches
    assert report.padding_runs_checked == 1


# --------------------------------------------------------------------------- refusals


def test_missing_source_file_is_refused(source: Source, tmp_path: Path):
    (source.root / "layer7_down.bin").unlink()
    with pytest.raises(ptp.SourceError) as excinfo:
        ptp.prepare_tp_pack(source.write_inventory(), _plan_with(source), tmp_path / "pack")
    assert "missing" in str(excinfo.value)


def test_wrong_source_hash_is_refused_by_a_write_and_by_a_verify(source: Source, tmp_path: Path):
    out = tmp_path / "pack"
    manifest = ptp.prepare_tp_pack(source.write_inventory(), _plan_with(source), out, materialize=True)
    assert manifest["receipts"]["source_hashes"]["status"] == "verified"
    path = source.root / "layer7_gate_up.bin"
    blob = bytearray(path.read_bytes())
    blob[12345] ^= 0xFF
    path.write_bytes(bytes(blob))          # same size, different content: a stale pin
    with pytest.raises(ptp.SourceError) as excinfo:      # a write re-reads the hashes it copies from
        ptp.prepare_tp_pack(source.write_inventory(), _plan_with(source), tmp_path / "other",
                            materialize=True)
    assert "hash" in str(excinfo.value)
    # plan mode stays bounded: it checks sizes and ranges, not the content of every file
    assert ptp.prepare_tp_pack(source.write_inventory(), _plan_with(source), tmp_path / "plan-only")["materialized"] is False
    report = ptp.verify_tp_pack(source.write_inventory(), out)
    assert not report.ok
    assert any("hash" in line for line in report.mismatches)


def test_a_stale_declared_file_size_is_refused(source: Source, tmp_path: Path):
    (source.root / "layer7_gate_up.bin").write_bytes(b"\0" * 12)
    inventory = source.inventory()
    for entry in inventory["source"]["files"]:
        if entry["path"] == "layer7_gate_up.bin":
            entry["sha256"] = hashlib.sha256(b"\0" * 12).hexdigest()
    path = source.write_inventory("stale-size.json")
    path.write_text(json.dumps(inventory))
    plan = base_plan()
    plan["source_lock"]["inventory_hash"] = ptp.inventory_hash(inventory)
    with pytest.raises(ptp.SourceError) as excinfo:
        ptp.prepare_tp_pack(path, plan, tmp_path / "pack")
    assert "size differs" in str(excinfo.value)


def test_reading_past_the_real_file_size_is_refused(source: Source, tmp_path: Path):
    # the inventory's file size is honest, but the tensor claims more bytes than the file holds
    inventory = source.inventory()
    for entry in inventory["tensors"]:
        if entry["name"] == GATE_UP:
            entry["byte_length"] += GATE_UP_BLOCK_BYTES
    path = source.write_inventory("oversized.json")
    path.write_text(json.dumps(inventory))
    plan = _plan_with(source)
    plan["source_lock"]["inventory_hash"] = ptp.inventory_hash(inventory)
    with pytest.raises(ptp.RangeError) as excinfo:
        ptp.prepare_tp_pack(path, plan, tmp_path / "pack")
    assert "outside the source file" in str(excinfo.value)


def test_a_range_beyond_the_tensor_extent_is_refused(source: Source, tmp_path: Path):
    def mutate(slices):
        for entry in slices:
            if entry["tensor"] == DOWN and entry["rank"] == 2:
                entry["range"] = [448, 704]
                entry["local_shape"] = [DOWN_ROWS, 256]
                entry["valid_lanes"] = None
        return slices

    with pytest.raises(ptp.PlanError) as excinfo:
        ptp.prepare_tp_pack(source.write_inventory(), _plan_with(source, slices=mutate), tmp_path / "pack")
    assert "extent" in str(excinfo.value)


def test_a_cut_across_a_quantization_block_is_refused(source: Source, tmp_path: Path):
    def mutate(slices):
        return [entry for entry in slices if not (entry["tensor"] == DOWN and entry["rank"] == 1)] + [
            _slice(DOWN, 1, 1, 288, 576, "down", local_shape=[DOWN_ROWS, 288], valid_lanes=None)
        ]

    with pytest.raises(ptp.PlanError) as excinfo:
        ptp.prepare_tp_pack(source.write_inventory(), _plan_with(source, slices=mutate), tmp_path / "pack")
    assert "block" in str(excinfo.value)


def test_a_plan_that_would_requantize_is_refused(source: Source, tmp_path: Path):
    def mutate(slices):
        for entry in slices:
            if entry["tensor"] == GATE_UP and entry["rank"] == 0:
                entry["codec"] = "q8_0"
        return slices

    with pytest.raises(ptp.PlanError) as excinfo:
        ptp.prepare_tp_pack(source.write_inventory(), _plan_with(source, slices=mutate), tmp_path / "pack")
    assert "requantize" in str(excinfo.value)


def test_a_plan_locked_to_another_inventory_is_refused(source: Source, tmp_path: Path):
    plan = base_plan()
    plan["source_lock"]["inventory_hash"] = "f" * 64
    with pytest.raises(ptp.SourceError):
        ptp.prepare_tp_pack(source.write_inventory(), plan, tmp_path / "pack")


def test_trimming_the_logical_vocabulary_is_refused(source: Source, tmp_path: Path):
    def mutate(slices):
        return [entry for entry in slices
                if not (entry["tensor"] == EMBED and entry["rank"] == 2)]

    with pytest.raises(ptp.PlanError) as excinfo:
        ptp.prepare_tp_pack(source.write_inventory(), _plan_with(source, slices=mutate), tmp_path / "pack")
    assert "vocabulary" in str(excinfo.value)


def test_overlapping_vocabulary_rows_are_refused(source: Source, tmp_path: Path):
    def mutate(slices):
        for entry in slices:
            if entry["tensor"] == EMBED and entry["rank"] == 2:
                entry["range"] = [32, 96]
                entry["local_shape"] = [64, EMBED_WIDTH]
        return slices

    with pytest.raises(ptp.PlanError) as excinfo:
        ptp.prepare_tp_pack(source.write_inventory(), _plan_with(source, slices=mutate), tmp_path / "pack")
    assert "vocabulary" in str(excinfo.value)


def test_a_plan_locked_with_the_inventory_writer_hash_is_accepted(source: Source, tmp_path: Path):
    """Task 1 hashes an inventory with default separators; that lock must not be a false refusal."""
    plan = base_plan()
    inventory = source.inventory()
    plan["source_lock"]["inventory_hash"] = hashlib.sha256(
        json.dumps(inventory, sort_keys=True).encode("utf-8")).hexdigest()
    assert plan["source_lock"]["inventory_hash"] not in ptp.inventory_hash_variants(
        {"schema_version": 1})                     # sanity: a different inventory is still refused
    out = tmp_path / "pack"
    manifest = ptp.prepare_tp_pack(source.write_inventory(), plan, out, materialize=True)
    assert manifest["materialized"] is True
    assert manifest["inventory_hash"] == ptp.inventory_hash(inventory)


def test_a_slice_with_no_rank_and_not_replicated_is_refused(source: Source, tmp_path: Path):
    def mutate(slices):
        for entry in slices:
            if entry["tensor"] == GATE_UP and entry["rank"] == 0:
                del entry["rank"]
        return slices

    with pytest.raises(ptp.PlanError) as excinfo:
        ptp.prepare_tp_pack(source.write_inventory(), _plan_with(source, slices=mutate), tmp_path / "pack")
    assert "rank" in str(excinfo.value)


# ------------------------------------------------------------------ output integrity


def test_an_interrupted_write_leaves_nothing_published(source: Source, tmp_path: Path):
    out = tmp_path / "pack"

    def abort(index: int, segment: dict) -> None:
        if index == 300:
            raise RuntimeError("power lost")

    with pytest.raises(RuntimeError):
        ptp.prepare_tp_pack(source.write_inventory(), _plan_with(source), out, materialize=True, progress=abort)
    assert not out.exists(), "a half-written pack must not be published"
    assert [p.name for p in tmp_path.iterdir()] == ["src"], "the staging directory must be cleaned up"


def test_the_output_is_not_clobbered_without_asking(source: Source, tmp_path: Path):
    out = tmp_path / "pack"
    ptp.prepare_tp_pack(source.write_inventory(), _plan_with(source), out, materialize=True)
    first = (out / "ranks" / "rank0.bin").read_bytes()
    with pytest.raises(ptp.PackError):
        ptp.prepare_tp_pack(source.write_inventory(), _plan_with(source), out, materialize=True)
    assert (out / "ranks" / "rank0.bin").read_bytes() == first
    ptp.prepare_tp_pack(source.write_inventory(), _plan_with(source), out, materialize=True, overwrite=True)
    assert (out / "ranks" / "rank0.bin").read_bytes() == first


def test_materializing_needs_its_own_free_disk(source: Source, tmp_path: Path):
    out = tmp_path / "pack"
    tight = lambda path: 50 * GIB + 1024          # barely above the floor, and it must keep room too
    with pytest.raises(ptp.DiskSpaceError) as excinfo:
        ptp.prepare_tp_pack(source.write_inventory(), _plan_with(source), out, materialize=True,
                            free_space_probe=tight)
    assert "53687091200" in str(excinfo.value)
    assert not out.exists()


def test_plan_mode_does_not_consult_the_disk_probe(source: Source, tmp_path: Path):
    def refusing(path):
        raise AssertionError("plan mode must not ask about disk space")

    manifest = ptp.prepare_tp_pack(source.write_inventory(), _plan_with(source), tmp_path / "pack",
                                   free_space_probe=refusing)
    assert manifest["materialized"] is False


def test_a_tampered_rank_byte_fails_verification(source: Source, tmp_path: Path):
    out = tmp_path / "pack"
    manifest = ptp.prepare_tp_pack(source.write_inventory(), _plan_with(source), out, materialize=True)
    assert ptp.verify_tp_pack(source.write_inventory(), out).ok
    victim = next(seg for seg in manifest["segments"]
                  if seg["kind"] == "codes" and seg["rank"] == 1 and seg["tensor"] == GATE_UP)
    path = out / "ranks" / "rank1.bin"
    blob = bytearray(path.read_bytes())
    blob[victim["dest_offset"]] ^= 0x01
    path.write_bytes(bytes(blob))
    report = ptp.verify_tp_pack(source.write_inventory(), out)
    assert not report.ok
    assert any(GATE_UP in line for line in report.mismatches)
    assert any("stopped at the first mismatch" in line for line in report.not_checked)


def test_verification_of_another_inventory_is_a_mismatch(source: Source, tmp_path: Path):
    inventory_path = source.write_inventory()
    out = tmp_path / "pack"
    ptp.prepare_tp_pack(inventory_path, _plan_with(source), out, materialize=True)
    moved = source.inventory()
    moved["source"]["revision"] = "rev-0002"
    other = source.write_inventory("moved-revision.json")
    other.write_text(json.dumps(moved))
    plan = _plan_with(source)
    plan["source_lock"]["inventory_hash"] = ptp.inventory_hash(moved)
    # same bytes on disk, a different pinned revision: the pack must say it is not this source
    ptp.prepare_tp_pack(other, plan, tmp_path / "pack2", materialize=True)
    report = ptp.verify_tp_pack(inventory_path, out)
    assert report.ok, report.mismatches
    other_report = ptp.verify_tp_pack(other, out)
    assert not other_report.ok
    assert any("different inventory" in line for line in other_report.mismatches)


def test_the_report_names_what_it_did_not_check(source: Source, tmp_path: Path):
    out = tmp_path / "pack"
    ptp.prepare_tp_pack(source.write_inventory(), _plan_with(source), out, materialize=True)
    report = ptp.verify_tp_pack(source.write_inventory(), out)
    assert report.ok, report.mismatches
    text = " | ".join(report.checked)
    assert "byte-for-byte" in text
    assert "sha256" in text
    joined = " | ".join(report.not_checked)
    assert "NAS staging" in joined, "the unrun download step must be stated"
    assert "layer7_norm.bin" in joined, "the file with no recorded hash must be called out"
    assert "not a running model" in joined
    assert report.files_hash_checked == len(source.files) - 1
    assert report.files_size_checked == len(source.files)


def test_the_cli_plans_then_verifies(source: Source, tmp_path: Path, capsys):
    inventory_path = source.write_inventory()
    plan_path = tmp_path / "tp-plan.json"
    plan_path.write_text(json.dumps(_plan_with(source)))
    out = tmp_path / "pack"
    assert ptp.main(["--inventory", str(inventory_path), "--plan", str(plan_path), "--output", str(out)]) == 0
    assert not out.exists()
    assert ptp.main(["--inventory", str(inventory_path), "--plan", str(plan_path), "--output", str(out),
                     "--materialize", "--min-free-gib", "0"]) == 0
    assert out.is_dir()
    assert ptp.main(["--inventory", str(inventory_path), "--plan", str(plan_path), "--output", str(out),
                     "--verify"]) == 0
    assert ptp.main(["--inventory", str(inventory_path), "--plan", str(plan_path), "--output", str(tmp_path / "nope"),
                     "--verify"]) == 1


def test_plan_mode_agrees_with_the_materialized_layout(source: Source, tmp_path: Path):
    planned = ptp.prepare_tp_pack(source.write_inventory(), _plan_with(source), tmp_path / "pack")
    made = ptp.prepare_tp_pack(source.write_inventory(), _plan_with(source), tmp_path / "pack2",
                               materialize=True)
    assert [(s["rank"], s["tensor"], s["kind"], s["dest_offset"], s["source_length"])
            for s in planned["segments"]] == [(s["rank"], s["tensor"], s["kind"], s["dest_offset"],
                                              s["source_length"]) for s in made["segments"]]
    for rank in ("0", "1", "2"):
        assert planned["rank_bytes"][rank] == made["rank_bytes"][rank]
