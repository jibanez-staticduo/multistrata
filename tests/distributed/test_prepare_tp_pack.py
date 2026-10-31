"""Artifact tests for ``tools/distributed/prepare_tp_pack.py`` (Strata TP3, Task 3).

Run:  $VENV/bin/python -m pytest tests/distributed/test_prepare_tp_pack.py -q
Venv: /home/staticduo/tmp/strata-tp3-20261030/venv (pytest 9.2.0, numpy 2.5.4)

CTest wiring the integrator owns (root CMakeLists.txt is not edited here):
    add_test(NAME tp_prepare_tp_pack_py
             COMMAND ${Python3_EXECUTABLE} -m pytest
                     ${CMAKE_SOURCE_DIR}/tests/distributed/test_prepare_tp_pack.py -q)

The fixtures are synthetic and small, but they are laid out the way the real pack is: contiguous
dimension first, so quantization blocks run along axis 0, a [in, out] weight is cut by output rows
(axis 1) or by input blocks (axis 0), and the stacked expert tensor is 3-D like
blk.0.ffn_down_exps.weight [640, 2560, 512], where the axis next to the contiguous one is the inner one:
one expert's 2560 output rows lie next to each other, and the next expert starts 2560 rows later.  The
numbers mirror the pinned inventory where a small
fixture can: the FFN partition is 256/192/192 at starts 0/256/448, one layer mixes a 32-block codec
with a 64-block one, and every block carries a native scale tail inside it.

No weights are downloaded by these tests.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import itertools
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
TOOLS = str(REPO / "tools" / "distributed")
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)

import prepare_tp_pack as ptp   # noqa: E402  (needs the path insert above)

GIB = 1024**3

FFN = 640                     # the width the agreed TP3 partition cuts: 256 / 192 / 192
GATE_IN = 128                 # gate/up: contiguous input axis, output axis is the FFN width
GATE_OUT = FFN
GATE_BLOCK = 32               # one codec of the pair that shares a layer
GATE_BLOCK_BYTES = 44         # 32 codes + a 12-byte native scale tail, inside the block
DOWN_IN = FFN                 # down: the FFN width is its contiguous input axis
DOWN_OUT = 128
DOWN_BLOCK = 64               # the other codec of the pair, same layer, other axis
DOWN_BLOCK_BYTES = 100        # 64 codes + 36 bytes of scales and scales-of-scales tail
DOWN_HEADER = 64              # the down file starts with a header, so its tensor does not start at 0
EXPS_IN = 64                  # the stacked expert: [in, out, experts], blocks along axis 0
EXPS_OUT = 32
EXPS_N = 8
EXPS_BLOCK = 32
EXPS_BLOCK_BYTES = 44
NORM_ITEMS = 2560
VOCAB = 96
EMBED_WIDTH = 16

GATE_UP = "blk.7.ffn_gate.weight"
GATE_UP_SCALES = "blk.7.ffn_gate.scales"
DOWN = "blk.7.ffn_down.weight"
DOWN_SCALES = "blk.7.ffn_down.scales"
EXPS = "blk.7.ffn_exps.weight"
EXPS_SCALES = "blk.7.ffn_exps.scales"
NORM = "blk.7.attn_norm.weight"
EMBED = "token_embd.weight"

FFN_RANGES = [(0, 256), (256, 448), (448, 640)]


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

        gate_up = _bytes(1, GATE_OUT * (GATE_IN // GATE_BLOCK) * GATE_BLOCK_BYTES)      # 640 rows x 176 B
        gate_scales = _bytes(2, GATE_OUT * (GATE_IN // GATE_BLOCK) * 4)                 # [4, 640] x 4 B
        down_body = _bytes(3, DOWN_OUT * (DOWN_IN // DOWN_BLOCK) * DOWN_BLOCK_BYTES)    # 128 rows x 1000 B
        down_scales = _bytes(4, DOWN_OUT * (DOWN_IN // DOWN_BLOCK) * 4)                 # [10, 128] x 4 B
        exps = _bytes(5, EXPS_OUT * EXPS_N * (EXPS_IN // EXPS_BLOCK) * EXPS_BLOCK_BYTES)  # 256 rows x 88 B
        exps_scales = _bytes(6, EXPS_OUT * EXPS_N * (EXPS_IN // EXPS_BLOCK) * 4)        # [2, 32, 8] x 4 B
        norm = _bytes(7, NORM_ITEMS * 2)
        embed = _bytes(8, VOCAB * EMBED_WIDTH * 2)
        down = b"\0" * DOWN_HEADER + down_body        # a file whose tensor does not start at byte 0

        self.files = {
            "layer7_gate_up.bin": gate_up,
            "layer7_gate_scales.bin": gate_scales,
            "layer7_down.bin": down,
            "layer7_down_scales.bin": down_scales,
            "layer7_exps.bin": exps,
            "layer7_exps_scales.bin": exps_scales,
            "layer7_norm.bin": norm,
            "embed.bin": embed,
        }
        for name, blob in self.files.items():
            (self.root / name).write_bytes(blob)

        # One file keeps no recorded hash on purpose: a pack must say so rather than pretend every
        # source byte was hashed.
        self.signed = {name: (None if name == "layer7_norm.bin" else hashlib.sha256(blob).hexdigest())
                       for name, blob in self.files.items()}

        self.tensors = [
            _entry(GATE_UP, "layer7_gate_up.bin", "q4_k", GATE_BLOCK, [GATE_IN, GATE_OUT], 0, len(gate_up),
                   "gate_up"),
            _entry(GATE_UP_SCALES, "layer7_gate_scales.bin", "f16", None,
                   [GATE_IN // GATE_BLOCK, GATE_OUT], 0, len(gate_scales), "other"),
            _entry(DOWN, "layer7_down.bin", "iq3_s", DOWN_BLOCK, [DOWN_IN, DOWN_OUT], DOWN_HEADER,
                   len(down_body), "down"),
            _entry(DOWN_SCALES, "layer7_down_scales.bin", "f16", None, [DOWN_IN // DOWN_BLOCK, DOWN_OUT], 0,
                   len(down_scales), "other"),
            _entry(EXPS, "layer7_exps.bin", "q4_k", EXPS_BLOCK, [EXPS_IN, EXPS_OUT, EXPS_N], 0, len(exps),
                   "expert_gate_up"),
            _entry(EXPS_SCALES, "layer7_exps_scales.bin", "f16", None,
                   [EXPS_IN // EXPS_BLOCK, EXPS_OUT, EXPS_N], 0, len(exps_scales), "other"),
            _entry(NORM, "layer7_norm.bin", "f16", None, [NORM_ITEMS], 0, len(norm), "norm"),
            _entry(EMBED, "embed.bin", "f16", None, [EMBED_WIDTH, VOCAB], 0, len(embed), "embed"),
        ]

    # ---- helpers the tests use to state expectations independently of the module

    def tensor(self, name: str) -> dict:
        return next(entry for entry in self.tensors if entry["name"] == name)

    def file_of(self, name: str) -> str:
        return self.tensor(name)["file"]

    def raw(self, name: str) -> bytes:
        return (self.root / self.file_of(name)).read_bytes()

    def row_bytes(self, name: str) -> int:
        t = self.tensor(name)
        row_count = 1
        for extent in t["shape"][1:]:
            row_count *= extent
        return t["byte_length"] // row_count

    def _rows_bytes(self, name: str, rows, first_unit: int, units: int | None) -> bytes:
        t = self.tensor(name)
        blob = self.raw(name)
        row_bytes = self.row_bytes(name)
        base = int(t["byte_offset"])
        if units is None:
            return b"".join(blob[base + row * row_bytes: base + (row + 1) * row_bytes] for row in rows)
        # One unit is one quantization block of the contiguous axis, or one item where the tensor carries no
        # blocks at all, which is how a separate scale tensor is laid out: one scale per block of the codes.
        unit = row_bytes // (t["shape"][0] // (t["block_size"] or 1))
        return b"".join(blob[base + row * row_bytes + first_unit * unit:
                             base + row * row_bytes + (first_unit + units) * unit] for row in rows)

    def expected_rows(self, name: str, first_row: int, count: int, first_unit: int = 0,
                      units: int | None = None) -> bytes:
        """Whole rows of a tensor by flat row index, or a unit range inside each row."""
        return self._rows_bytes(name, range(first_row, first_row + count), first_unit, units)

    def row_offsets_of_cut(self, name: str, axis: int, start: int, end: int) -> list[int]:
        """Flat row indices a cut owns, taken from the file's own nesting rather than from the module.

        A model file stores its fastest-varying dimension first, so over the non-contiguous axes a row index
        is ``a_1 + shape[1]*a_2 + shape[1]*shape[2]*a_3 + ...``: the axis next to the contiguous one is the
        inner one.  For the cuts these fixtures make that is also the order the destination fills, because
        every run is a block of adjacent rows.
        """
        shape = self.tensor(name)["shape"]
        picks = [range(start, end) if a == axis else range(shape[a]) for a in range(1, len(shape))]
        strides = [1]
        for extent in shape[1:-1]:
            strides.append(strides[-1] * extent)
        return sorted(sum(pick * stride for pick, stride in zip(picks_row, strides))
                      for picks_row in itertools.product(*picks))

    def expected_cut(self, name: str, axis: int, start: int, end: int) -> bytes:
        """The bytes one rank owns for a cut, stated the way the plan states it: an axis and a range.

        A cut of the contiguous axis keeps that many items out of every row, in the tensor's own units: a
        quantization block where the tensor has blocks, an item where it has none.  A cut of any other axis
        takes whole rows, tails and all.
        """
        if axis == 0:
            block = self.tensor(name)["block_size"] or 1
            rows = self.row_offsets_of_cut(name, 0, 0, self.tensor(name)["shape"][0])
            return self._rows_bytes(name, rows, start // block, (end - start) // block)
        return self._rows_bytes(name, self.row_offsets_of_cut(name, axis, start, end), 0, None)

    def inventory(self) -> dict:
        return {
            "schema_version": 1,
            "source": {"kind": "local", "repo": "ISTA-DASLab/synthetic-fixture", "revision": "rev-0001",
                       "files": [{"path": name, "size": len(blob), "sha256": self.signed[name]}
                                 for name, blob in self.files.items()]},
            "geometry": {"n_embd": EMBED_WIDTH, "n_layers": 8, "qsa_interval": 4, "n_vocab": VOCAB,
                         "n_ff": FFN, "n_expert": 512, "n_head": 24, "n_head_kv": 2},
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


def _slice(tensor: str, rank: int, axis: int, start: int, end: int, family: str, source: Source,
            **extra) -> dict:
    t = source.tensor(tensor)
    local = list(t["shape"])
    local[axis] = end - start
    entry = {"tensor": tensor, "family": family, "axis": axis, "range": [start, end],
             "global_shape": list(t["shape"]), "local_shape": local,
             "byte_row_stride": source.row_bytes(tensor), "scale_ref": None, "valid_lanes": None,
             "halo": [], "replicated": False, "rank": rank, "codec": t["dtype"],
             "block_size": t["block_size"], "cut_unit": 1}
    entry.update(extra)
    return entry


def base_plan(source: Source) -> dict:
    """The agreed TP3 cut, taken the way the real plan takes it.

    gate/up holds the FFN width as its output axis, so the partition is a row cut and needs no block
    alignment; down holds it as its input axis, so the same partition is a block cut; the stacked
    expert is cut on both of its non-contiguous axes.
    """
    slices: list[dict] = []
    for rank, (start, end) in enumerate(FFN_RANGES):
        slices.append(_slice(GATE_UP, rank, 1, start, end, "gate_up", source, scale_ref=GATE_UP_SCALES))
        slices.append(_slice(DOWN, rank, 0, start, end, "down", source, scale_ref=DOWN_SCALES))
    for rank, (start, end) in enumerate([(0, 16), (16, 24), (24, 32)]):
        slices.append(_slice(EXPS, rank, 1, start, end, "expert_gate_up", source, scale_ref=EXPS_SCALES))
    slices.append(_slice(NORM, 0, 0, 0, NORM_ITEMS, "norm", source, replicated=True))
    del slices[-1]["rank"]
    for rank, (start, end) in enumerate([(0, 32), (32, 64), (64, 96)]):
        slices.append(_slice(EMBED, rank, 1, start, end, "embed", source))
    return {
        "schema_version": 1,
        "plan_hash": "0" * 64,
        "world": 3,
        "source_lock": {"repo": "ISTA-DASLab/synthetic-fixture", "revision": "rev-0001",
                        "inventory_hash": None},
        "geometry": {"n_embd": EMBED_WIDTH, "n_layers": 8, "qsa_interval": 4, "n_vocab": VOCAB,
                     "n_ff": FFN, "n_expert": 512, "n_head": 24, "n_head_kv": 2},
        "memory": {"per_rank_bytes": 0, "components": {"gate_up": 1, "down": 1},
                   "halo_and_global_buffers": 0},
        "slices": slices,
        "activation": [{"consumer": "ffn_down", "gather": "all_gather_padded", "wire_slots": 192,
                        "valid_counts": [256, 192, 192], "quantizer": "q8_k", "block_size": 256,
                        "crosses_rank_boundary": True, "reference": "global_reference"}],
    }


def _pin(source: Source, plan: dict) -> dict:
    plan["source_lock"]["inventory_hash"] = ptp.inventory_hash(source.inventory())
    return plan


def _plan(source: Source, **mutate) -> dict:
    plan = _pin(source, base_plan(source))
    for key, value in mutate.items():
        if key == "slices":
            plan["slices"] = value(plan["slices"])
        else:
            plan[key] = value
    return plan


def _segments(manifest: dict, rank: int, tensor: str, kind: str = "codes") -> list[dict]:
    return [seg for seg in manifest["segments"]
            if seg["rank"] == rank and seg["tensor"] == tensor and seg["kind"] == kind]


def _read_runs(path: Path, runs) -> bytes:
    """Read a rank's block for a set of segments, in destination order."""
    out = bytearray()
    with open(path, "rb") as handle:
        for segment in sorted(runs, key=lambda seg: seg["dest_offset"]):
            handle.seek(segment["dest_offset"])
            out += handle.read(segment["source_length"])
    return bytes(out)


@pytest.fixture()
def source(tmp_path: Path) -> Source:
    return Source(tmp_path)


def _rank_file(out: Path, rank: int) -> Path:
    return out / "ranks" / f"rank{rank}.bin"


# ------------------------------------------------------------------ plan mode, no writes


def test_plan_mode_writes_nothing_at_all(source: Source, tmp_path: Path):
    out = tmp_path / "pack"
    manifest = ptp.prepare_tp_pack(source.write_inventory(), _plan(source), out)
    assert not out.exists(), "plan mode must not create the output directory"
    assert [item.name for item in tmp_path.iterdir()] == ["src"], "plan mode wrote somewhere else"
    assert manifest["materialized"] is False
    assert manifest["rank_files"] is None
    assert manifest["manifest_version"] == 1
    assert sum(int(value) for value in manifest["rank_bytes"].values()) > 0
    assert manifest["not_verified"], "plan mode must say what it did not check"
    sizes = {entry["path"]: entry["size"] for entry in manifest["source_files"]}
    for segment in manifest["segments"]:
        if segment["source_path"] is None:
            continue
        last = segment["source_offset"] + sum((count - 1) * stride for count, stride in segment["pattern"])
        assert last + segment["run_bytes"] <= sizes[segment["source_path"]]


def test_plan_mode_never_opens_the_output(tmp_path: Path, source: Source, monkeypatch):
    real_open = open
    opened: list[str] = []

    def spy(file, *args, **kwargs):
        opened.append(str(file))
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(ptp, "open", spy, raising=False)
    ptp.prepare_tp_pack(source.write_inventory(), _plan(source), tmp_path / "pack")
    assert not [path for path in opened if "pack" in path], "plan mode opened an output file"


def test_manifest_carries_the_lock_and_the_run_accounting(source: Source, tmp_path: Path):
    manifest = ptp.prepare_tp_pack(source.write_inventory(), _plan(source), tmp_path / "pack")
    assert manifest["plan_content_hash"] == ptp.plan_content_hash(_plan(source))
    assert manifest["inventory_hash"] == ptp.inventory_hash(source.inventory())
    assert manifest["activation"][0]["gather"] == "all_gather_padded"
    assert manifest["segment_count"] == len(manifest["segments"])
    # Every planned run, counted from the manifest's own patterns: a strided entry stands for many runs, and a
    # zero-filled padding entry stands for the one run of zeros it writes.
    assert manifest["run_count"] == sum(_prod(seg) for seg in manifest["segments"])
    # a strided segment is one entry that stands for many source runs
    down_rank0 = _segments(manifest, 0, DOWN)[0]
    assert down_rank0["pattern"] == [[DOWN_OUT, source.row_bytes(DOWN)]]
    assert manifest["run_count"] > manifest["segment_count"]


def _prod(segment: dict) -> int:
    total = 1
    for count, _stride in segment["pattern"]:
        total *= count
    return total


# --------------------------------------------------- which axis a block runs along


def test_block_alignment_is_demanded_on_the_contiguous_axis(source: Source, tmp_path: Path):
    """Half a block on the input axis is a split block; a whole block is not."""
    def half_block(slices):
        return [s for s in slices if not (s["tensor"] == GATE_UP and s["rank"] == 0)] + [
            _slice(GATE_UP, 0, 0, 0, GATE_BLOCK // 2, "gate_up", source, scale_ref=None)]

    with pytest.raises(ptp.PlanError) as excinfo:
        ptp.prepare_tp_pack(source.write_inventory(), _plan(source, slices=half_block), tmp_path / "pack")
    assert "splits a quantization block" in str(excinfo.value)

    def whole_block(slices):
        return [s for s in slices if not (s["tensor"] == GATE_UP and s["rank"] == 0)] + [
            _slice(GATE_UP, 0, 0, 0, GATE_BLOCK, "gate_up", source, scale_ref=None)]

    manifest = ptp.prepare_tp_pack(source.write_inventory(), _plan(source, slices=whole_block),
                                   tmp_path / "pack")
    segment = _segments(manifest, 0, GATE_UP)[0]
    assert segment["run_bytes"] == GATE_BLOCK_BYTES
    assert segment["pattern"] == [[GATE_OUT, source.row_bytes(GATE_UP)]], "one run per output row"
    assert segment["source_length"] == GATE_BLOCK_BYTES * GATE_OUT


def test_row_axis_cut_takes_any_row_boundary_and_refuses_a_row_that_is_not_there(source: Source,
                                                                               tmp_path: Path):
    """The output axis is not the block axis: a row boundary is a valid cut, a missing row is not."""
    def odd_boundary(slices):
        return [s for s in slices if not (s["tensor"] == GATE_UP and s["rank"] == 0)] + [
            _slice(GATE_UP, 0, 1, 0, 250, "gate_up", source, scale_ref=None)]     # 250 is not a multiple of 32

    manifest = ptp.prepare_tp_pack(source.write_inventory(), _plan(source, slices=odd_boundary),
                                   tmp_path / "pack")
    segment = _segments(manifest, 0, GATE_UP)[0]
    row_bytes = source.row_bytes(GATE_UP)
    assert segment["run_bytes"] == 250 * row_bytes and segment["pattern"] == [], "whole rows are adjacent"
    assert segment["source_offset"] == 0

    def missing_row(slices):
        return [s for s in slices if not (s["tensor"] == GATE_UP and s["rank"] == 2)] + [
            _slice(GATE_UP, 2, 1, 600, 700, "gate_up", source, scale_ref=None)]

    with pytest.raises(ptp.PlanError) as excinfo:
        ptp.prepare_tp_pack(source.write_inventory(), _plan(source, slices=missing_row), tmp_path / "pack")
    assert "outside extent" in str(excinfo.value)


def test_stacked_expert_is_cut_on_both_of_its_other_axes(source: Source, tmp_path: Path):
    """A 3-D expert stack: rows are (out, expert) with the output axis inner, so one expert is adjacent rows.

    Three cuts of the same stack, each stated in the plan's terms and checked against the file's own nesting:
    a block range of the input axis, a range of output rows, and a range of whole experts.
    """
    def extra(slices):
        return slices + [
            _slice(EXPS, 0, 0, 0, EXPS_BLOCK, "expert_gate_up", source, scale_ref=None),
            _slice(EXPS, 0, 2, 6, EXPS_N, "expert_gate_up", source, scale_ref=None),
        ]

    manifest = ptp.prepare_tp_pack(source.write_inventory(), _plan(source, slices=extra), tmp_path / "pack")
    row_bytes = source.row_bytes(EXPS)                      # one (out, expert) row: 2 blocks of 44 B
    assert row_bytes == (EXPS_IN // EXPS_BLOCK) * EXPS_BLOCK_BYTES
    # axis 1 cut, rank 1: output rows 16..23, eight adjacent rows per expert, experts a whole stack apart
    row_cut = _segments(manifest, 1, EXPS)[0]
    assert row_cut["run_bytes"] == 8 * row_bytes
    assert row_cut["pattern"] == [[EXPS_N, EXPS_OUT * row_bytes]]
    assert row_cut["source_offset"] == 16 * row_bytes
    assert row_cut["source_length"] == len(source.expected_cut(EXPS, 1, 16, 24))
    # axis 0 cut: one block per (out, expert) row, at the row stride
    block_cut = _segments(manifest, 0, EXPS)[-2]
    assert block_cut["run_bytes"] == EXPS_BLOCK_BYTES
    assert block_cut["pattern"] == [[EXPS_OUT * EXPS_N, row_bytes]]
    # axis 2 cut: experts 6 and 7 are two adjacent blocks of output rows, so one range covers both
    expert_cut = _segments(manifest, 0, EXPS)[-1]
    assert expert_cut["run_bytes"] == 2 * EXPS_OUT * row_bytes and expert_cut["pattern"] == []
    assert expert_cut["source_offset"] == 6 * EXPS_OUT * row_bytes
    assert expert_cut["source_length"] == len(source.expected_cut(EXPS, 2, 6, EXPS_N))

    def half(slices):
        return slices + [_slice(EXPS, 0, 0, 0, EXPS_BLOCK // 2, "expert_gate_up", source, scale_ref=None)]

    with pytest.raises(ptp.PlanError) as excinfo:
        ptp.prepare_tp_pack(source.write_inventory(), _plan(source, slices=half), tmp_path / "pack")
    assert "splits a quantization block" in str(excinfo.value)


def test_two_block_codecs_in_one_layer_keep_their_own_arithmetic(source: Source, tmp_path: Path):
    out = tmp_path / "pack"
    manifest = ptp.prepare_tp_pack(source.write_inventory(), _plan(source), out, materialize=True)
    gate = _segments(manifest, 1, GATE_UP)[0]
    down = _segments(manifest, 1, DOWN)[0]
    assert gate["run_bytes"] == 192 * source.row_bytes(GATE_UP) and gate["pattern"] == []
    assert down["run_bytes"] == 3 * DOWN_BLOCK_BYTES and down["run_count"] == DOWN_OUT
    report = ptp.verify_tp_pack(source.write_inventory(), out)
    assert report.ok, report.mismatches


def test_plan_declared_row_stride_is_a_second_opinion_not_a_free_pass(source: Source, tmp_path: Path):
    def wrong_stride(slices):
        for entry in slices:
            if entry["tensor"] == GATE_UP and entry["rank"] == 0:
                entry["byte_row_stride"] = source.row_bytes(GATE_UP) - 1
        return slices

    with pytest.raises(ptp.PlanError) as excinfo:
        ptp.prepare_tp_pack(source.write_inventory(), _plan(source, slices=wrong_stride), tmp_path / "pack")
    text = str(excinfo.value)
    assert str(source.row_bytes(GATE_UP) - 1) in text and str(source.row_bytes(GATE_UP)) in text


def test_plan_declared_block_size_and_cut_unit_are_cross_checked(source: Source, tmp_path: Path):
    def wrong_block(slices):
        for entry in slices:
            if entry["tensor"] == DOWN and entry["rank"] == 0:
                entry["block_size"] = 128
        return slices

    with pytest.raises(ptp.PlanError) as excinfo:
        ptp.prepare_tp_pack(source.write_inventory(), _plan(source, slices=wrong_block), tmp_path / "pack")
    assert "block size" in str(excinfo.value)

    def odd_cut_unit(slices):
        for entry in slices:
            if entry["tensor"] == DOWN and entry["rank"] == 0:
                entry["cut_unit"] = 7
        return slices

    with pytest.raises(ptp.PlanError) as excinfo:
        ptp.prepare_tp_pack(source.write_inventory(), _plan(source, slices=odd_cut_unit), tmp_path / "pack")
    assert "cut unit" in str(excinfo.value)


# ------------------------------------------------- materialize + byte-for-byte proof


def test_materialize_reconstructs_blocks_and_scales_byte_for_byte(source: Source, tmp_path: Path):
    out = tmp_path / "pack"
    manifest = ptp.prepare_tp_pack(source.write_inventory(), _plan(source), out, materialize=True)
    assert manifest["materialized"] is True
    assert (out / ptp.MANIFEST_NAME).is_file()
    assert manifest["receipts"]["byte_equality"]["status"] == "verified"

    cases = [
        # rank, tensor, axis, start, end: the plan's own cut, in the plan's own units
        (0, GATE_UP, 1, 0, 256),
        (1, GATE_UP, 1, 256, 448),
        (2, GATE_UP, 1, 448, 640),
        (0, DOWN, 0, 0, 256),          # 256 items = 4 blocks of 64, one run per output row
        (1, DOWN, 0, 256, 448),        # 192 items = 3 blocks of 64
        (2, DOWN, 0, 448, 640),
        (0, EXPS, 1, 0, 16),           # output rows 0..15, eight experts EXPS_OUT rows apart
        (1, EXPS, 1, 16, 24),          # output rows 16..23, the same shape at a later row
        (2, EXPS, 1, 24, 32),
    ]
    for rank, tensor, axis, start, end in cases:
        expected = source.expected_cut(tensor, axis, start, end)
        runs = _segments(manifest, rank, tensor)
        assert sum(seg["source_length"] for seg in runs) == len(expected)
        assert _read_runs(_rank_file(out, rank), runs) == expected, f"rank {rank} {tensor}"

    # separate scale tensors: copied for exactly the owned blocks, never recomputed
    scale_cases = [
        (0, GATE_UP_SCALES, 1, 0, 256),       # the scales of the rows this rank owns
        (1, DOWN_SCALES, 0, 4, 7),            # one scale per block: blocks 4..6 of the codes
        (1, EXPS_SCALES, 1, 16, 24),
    ]
    for rank, scales, axis, start, end in scale_cases:
        expected = source.expected_cut(scales, axis, start, end)
        runs = _segments(manifest, rank, scales, "scales")
        assert sum(seg["source_length"] for seg in runs) == len(expected)
        assert _read_runs(_rank_file(out, rank), runs) == expected, f"rank {rank} {scales}"


def test_a_block_cut_of_a_stacked_expert_gathers_one_block_per_row(source: Source, tmp_path: Path):
    """The down-style cut on a 3-D stack: one block out of every (out, expert) row, gathered run by run.

    This is the shape the real plan takes for a stacked down projection, where the agreed width is the
    contiguous axis, and it is where a strided gather over a stacked tensor is proven byte for byte: one
    block out of each of the 256 (out, expert) rows, in a plan entry that stands for all of them.
    """
    out = tmp_path / "pack"

    def extra(slices):
        return slices + [_slice(EXPS, 1, 0, EXPS_BLOCK, 2 * EXPS_BLOCK, "expert_gate_up", source,
                                scale_ref=EXPS_SCALES)]

    manifest = ptp.prepare_tp_pack(source.write_inventory(), _plan(source, slices=extra), out,
                                   materialize=True)
    segment = _segments(manifest, 1, EXPS)[-1]
    assert segment["run_bytes"] == EXPS_BLOCK_BYTES, "the range is one block of the contiguous axis"
    assert segment["pattern"] == [[EXPS_OUT * EXPS_N, source.row_bytes(EXPS)]]
    assert segment["run_count"] == EXPS_OUT * EXPS_N
    assert _read_runs(_rank_file(out, 1), [segment]) == source.expected_cut(EXPS, 0, EXPS_BLOCK,
                                                                            2 * EXPS_BLOCK)
    scales = sorted(_segments(manifest, 1, EXPS_SCALES, "scales"), key=lambda seg: seg["dest_offset"])
    # The last scale block of rank 1's file is the one this cut added; the earlier one is the plan's own row share.
    assert _read_runs(_rank_file(out, 1), scales[-1:]) == source.expected_cut(EXPS_SCALES, 0, 1, 2)
    report = ptp.verify_tp_pack(source.write_inventory(), out)
    assert report.ok, report.mismatches


def test_native_scale_tails_stay_inside_the_copied_block(source: Source, tmp_path: Path):
    manifest = ptp.prepare_tp_pack(source.write_inventory(), _plan(source), tmp_path / "pack")
    assert source.row_bytes(GATE_UP) == (GATE_IN // GATE_BLOCK) * GATE_BLOCK_BYTES
    assert source.row_bytes(DOWN) == (DOWN_IN // DOWN_BLOCK) * DOWN_BLOCK_BYTES
    for segment in manifest["segments"]:
        if segment["kind"] != "codes" or segment["tensor"] in (NORM, EMBED):
            continue
        block = {"blk.7.ffn_gate.weight": GATE_BLOCK_BYTES, "blk.7.ffn_down.weight": DOWN_BLOCK_BYTES,
                 "blk.7.ffn_exps.weight": EXPS_BLOCK_BYTES}[segment["tensor"]]
        assert segment["run_bytes"] % block == 0, "a run is a whole number of blocks, tail included"


def test_one_strided_segment_walks_every_run_of_a_partitioned_input_axis(source: Source, tmp_path: Path):
    out = tmp_path / "pack"
    manifest = ptp.prepare_tp_pack(source.write_inventory(), _plan(source), out, materialize=True)
    segment = _segments(manifest, 2, DOWN)[0]
    assert len(_segments(manifest, 2, DOWN)) == 1, "one entry, not 128"
    assert segment["pattern"] == [[DOWN_OUT, source.row_bytes(DOWN)]]
    assert segment["source_length"] == 3 * DOWN_BLOCK_BYTES * DOWN_OUT
    # a row in the middle of the stack, read back through the pattern
    middle = source.expected_rows(DOWN, 77, 1, 7, 3)
    runs = _segments(manifest, 2, DOWN)
    offset_in_block = (77 - 0) * (3 * DOWN_BLOCK_BYTES)
    with open(_rank_file(out, 2), "rb") as handle:
        handle.seek(runs[0]["dest_offset"] + offset_in_block)
        assert handle.read(3 * DOWN_BLOCK_BYTES) == middle
    report = ptp.verify_tp_pack(source.write_inventory(), out)
    assert report.ok, report.mismatches
    assert report.runs_compared == sum(seg["source_length"] and _prod(seg) or 0
                                      for seg in manifest["segments"] if seg["source_path"] is not None)


def test_replicated_tensor_lands_whole_on_every_rank(source: Source, tmp_path: Path):
    out = tmp_path / "pack"
    manifest = ptp.prepare_tp_pack(source.write_inventory(), _plan(source), out, materialize=True)
    wanted = source.raw(NORM)
    for rank in range(3):
        runs = _segments(manifest, rank, NORM)
        assert len(runs) == 1 and runs[0]["source_length"] == len(wanted)
        assert _read_runs(_rank_file(out, rank), runs) == wanted


def test_padding_is_zeroed_and_never_read_from_the_source(source: Source, tmp_path: Path):
    out = tmp_path / "pack"

    def short_share(slices):
        for entry in slices:
            if entry["tensor"] == GATE_UP and entry["rank"] == 2:
                entry["valid_lanes"] = [256, 192, 128]
        return slices

    manifest = ptp.prepare_tp_pack(source.write_inventory(), _plan(source, slices=short_share), out,
                                   materialize=True)
    pads = [seg for seg in manifest["segments"] if seg["kind"] == "padding"]
    assert [seg["rank"] for seg in pads] == [2], "only rank 2 has a short share in this plan"
    row_bytes = source.row_bytes(GATE_UP)
    assert pads[0]["source_path"] is None and pads[0]["source_length"] == 64 * row_bytes
    codes = _segments(manifest, 2, GATE_UP)
    assert sum(seg["source_length"] for seg in codes) == 128 * row_bytes
    with open(_rank_file(out, 2), "rb") as handle:
        handle.seek(pads[0]["dest_offset"])
        assert handle.read(pads[0]["source_length"]) == b"\0" * pads[0]["source_length"]
    report = ptp.verify_tp_pack(source.write_inventory(), out)
    assert report.ok, report.mismatches
    assert report.padding_runs_checked == 1


def test_a_short_share_of_the_contiguous_axis_is_not_a_padding(source: Source, tmp_path: Path):
    def short_block_share(slices):
        for entry in slices:
            if entry["tensor"] == DOWN and entry["rank"] == 1:
                entry["valid_lanes"] = [256, 128, 192]
        return slices

    with pytest.raises(ptp.PlanError) as excinfo:
        ptp.prepare_tp_pack(source.write_inventory(), _plan(source, slices=short_block_share),
                            tmp_path / "pack")
    assert "padding" in str(excinfo.value)


# --------------------------------------------------------------------------- refusals


def test_missing_source_file_is_refused(source: Source, tmp_path: Path):
    (source.root / "layer7_down.bin").unlink()
    with pytest.raises(ptp.SourceError) as excinfo:
        ptp.prepare_tp_pack(source.write_inventory(), _plan(source), tmp_path / "pack")
    assert "missing" in str(excinfo.value)


def test_wrong_source_hash_is_refused_by_a_write_and_by_a_verify(source: Source, tmp_path: Path):
    out = tmp_path / "pack"
    manifest = ptp.prepare_tp_pack(source.write_inventory(), _plan(source), out, materialize=True)
    assert manifest["receipts"]["source_hashes"]["status"] == "verified"
    path = source.root / "layer7_gate_up.bin"
    blob = bytearray(path.read_bytes())
    blob[12345] ^= 0xFF
    path.write_bytes(bytes(blob))          # same size, different content: a stale pin
    with pytest.raises(ptp.SourceError) as excinfo:      # a write re-reads the hashes it copies from
        ptp.prepare_tp_pack(source.write_inventory(), _plan(source), tmp_path / "other", materialize=True)
    assert "hash" in str(excinfo.value)
    # plan mode stays bounded: sizes and ranges, never the content of every file
    assert ptp.prepare_tp_pack(source.write_inventory(), _plan(source), tmp_path / "plan-only")[
        "materialized"] is False
    report = ptp.verify_tp_pack(source.write_inventory(), out)
    assert not report.ok
    assert any("hash" in line for line in report.mismatches)


def test_a_stale_declared_file_size_is_refused(source: Source, tmp_path: Path):
    (source.root / "layer7_gate_up.bin").write_bytes(b"\0" * 12)
    inventory = source.inventory()
    for entry in inventory["source"]["files"]:
        if entry["path"] == "layer7_gate_up.bin":
            entry["sha256"] = hashlib.sha256(b"\0" * 12).hexdigest()   # a hash that matches the new bytes
    path = source.write_inventory("stale-size.json")
    path.write_text(json.dumps(inventory))
    plan = base_plan(source)
    plan["source_lock"]["inventory_hash"] = ptp.inventory_hash(inventory)
    with pytest.raises(ptp.SourceError) as excinfo:
        ptp.prepare_tp_pack(path, plan, tmp_path / "pack")
    assert "size differs" in str(excinfo.value)


def test_reading_past_the_real_file_size_is_refused(source: Source, tmp_path: Path):
    """The inventory's file size is honest, but the tensor claims a row the file does not hold."""
    inventory = source.inventory()
    row_bytes = source.row_bytes(GATE_UP)
    short = GATE_OUT * row_bytes - row_bytes
    (source.root / "layer7_gate_up.bin").write_bytes(b"\0" * short)
    for entry in inventory["source"]["files"]:
        if entry["path"] == "layer7_gate_up.bin":
            entry["size"] = short                                # what the disk really has
            entry["sha256"] = hashlib.sha256(b"\0" * short).hexdigest()
    for entry in inventory["tensors"]:
        if entry["name"] == GATE_UP:
            entry["byte_length"] = short + row_bytes             # one row past the end of the file
    path = source.write_inventory("oversized.json")
    path.write_text(json.dumps(inventory))
    plan = _plan(source)
    plan["source_lock"]["inventory_hash"] = ptp.inventory_hash(inventory)
    with pytest.raises(ptp.RangeError) as excinfo:
        ptp.prepare_tp_pack(path, plan, tmp_path / "pack")
    assert "outside the source file" in str(excinfo.value)


def test_a_cut_across_a_quantization_block_is_refused(source: Source, tmp_path: Path):
    def misaligned(slices):
        return [entry for entry in slices if not (entry["tensor"] == DOWN and entry["rank"] == 2)] + [
            _slice(DOWN, 2, 0, 448, 576 + 32, "down", source, scale_ref=None)     # 224 items, block 64
        ]

    with pytest.raises(ptp.PlanError) as excinfo:
        ptp.prepare_tp_pack(source.write_inventory(), _plan(source, slices=misaligned), tmp_path / "pack")
    assert "splits a quantization block" in str(excinfo.value)


def test_a_plan_that_would_requantize_is_refused(source: Source, tmp_path: Path):
    def other_codec(slices):
        for entry in slices:
            if entry["tensor"] == GATE_UP and entry["rank"] == 0:
                entry["codec"] = "q8_0"
        return slices

    with pytest.raises(ptp.PlanError) as excinfo:
        ptp.prepare_tp_pack(source.write_inventory(), _plan(source, slices=other_codec), tmp_path / "pack")
    assert "requantize" in str(excinfo.value)


def test_a_slice_that_shrinks_an_axis_it_does_not_cut_is_refused(source: Source, tmp_path: Path):
    def shrunk(slices):
        for entry in slices:
            if entry["tensor"] == DOWN and entry["rank"] == 0:
                entry["local_shape"] = [256, 64]
        return slices

    with pytest.raises(ptp.PlanError) as excinfo:
        ptp.prepare_tp_pack(source.write_inventory(), _plan(source, slices=shrunk), tmp_path / "pack")
    assert "shrinks axis" in str(excinfo.value)


def test_a_plan_locked_to_another_inventory_is_refused(source: Source, tmp_path: Path):
    plan = base_plan(source)
    plan["source_lock"]["inventory_hash"] = "f" * 64
    with pytest.raises(ptp.SourceError):
        ptp.prepare_tp_pack(source.write_inventory(), plan, tmp_path / "pack")


@pytest.mark.parametrize("lock", [None, {}, {"inventory_hash": ""}])
def test_missing_inventory_lock_is_refused(source: Source, tmp_path: Path, lock):
    plan = base_plan(source)
    plan["source_lock"] = lock
    with pytest.raises(ptp.SourceError, match="inventory_hash"):
        ptp.prepare_tp_pack(source.write_inventory(), plan, tmp_path / "pack")



def test_raw_inventory_fingerprint_is_checked(source: Source, tmp_path: Path):
    path = source.write_inventory()
    digest = 0xcbf29ce484222325
    for byte in path.read_bytes():
        digest = ((digest ^ byte) * 0x100000001b3) & ((1 << 64) - 1)
    plan = base_plan(source)
    plan["source_lock"]["inventory_hash"] = f"fnv1a64-raw:{digest:016x}"
    ptp.prepare_tp_pack(path, plan, tmp_path / "pack")
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ptp.SourceError):
        ptp.prepare_tp_pack(path, plan, tmp_path / "mismatch")


def test_noncontiguous_head_runs_keep_plan_order_and_original_stride(source: Source, tmp_path: Path):
    plan = base_plan(source)
    plan["source_lock"]["inventory_hash"] = ptp.inventory_hash(source.inventory())
    plan["slices"] = [s for s in plan["slices"] if not (s["tensor"] == GATE_UP and s.get("rank") == 0)]
    runs = []
    for start in (0, 16, 32):
        entry = _slice(GATE_UP, 0, 1, start, start + 6, "gate_up", source, scale_ref=None)
        entry.update(part="gdn_v", head_range=[start, start + 6], compute_policy="owned_partial")
        runs.append(entry)
    plan["slices"].extend(runs)
    manifest = ptp.prepare_tp_pack(source.write_inventory(), plan, tmp_path / "pack", materialize=True)
    placed = [s for s in manifest["segments"] if s["tensor"] == GATE_UP and s["rank"] == 0]
    assert len(placed) == 3
    stride = source.row_bytes(GATE_UP)
    blob = (tmp_path / "pack" / "ranks" / "rank0.bin").read_bytes()
    original = source.files["layer7_gate_up.bin"]
    for segment, start in zip(placed, (0, 16, 32)):
        offset = segment["dest_offset"]
        assert blob[offset:offset + 6 * stride] == original[start * stride:(start + 6) * stride]
    assert [s["head_range"] for s in manifest["compute_ownership"]
            if s["tensor"] == GATE_UP and s.get("rank") == 0] == [[0, 6], [16, 22], [32, 38]]

def test_a_plan_locked_with_the_inventory_writer_hash_is_accepted(source: Source, tmp_path: Path):
    """Task 1 hashes an inventory with default separators; that lock must not be a false refusal."""
    plan = base_plan(source)
    inventory = source.inventory()
    plan["source_lock"]["inventory_hash"] = hashlib.sha256(
        json.dumps(inventory, sort_keys=True).encode("utf-8")).hexdigest()
    manifest = ptp.prepare_tp_pack(source.write_inventory(), plan, tmp_path / "pack", materialize=True)
    assert manifest["materialized"] is True
    assert manifest["inventory_hash"] == ptp.inventory_hash(inventory)


def test_trimming_the_logical_vocabulary_is_refused(source: Source, tmp_path: Path):
    def missing_rows(slices):
        return [entry for entry in slices if not (entry["tensor"] == EMBED and entry["rank"] == 2)]

    with pytest.raises(ptp.PlanError) as excinfo:
        ptp.prepare_tp_pack(source.write_inventory(), _plan(source, slices=missing_rows), tmp_path / "pack")
    assert "vocabulary" in str(excinfo.value)


def test_overlapping_vocabulary_rows_are_refused(source: Source, tmp_path: Path):
    def overlap(slices):
        for entry in slices:
            if entry["tensor"] == EMBED and entry["rank"] == 2:
                entry["range"] = [32, 96]
                entry["local_shape"] = [EMBED_WIDTH, 64]
        return slices

    with pytest.raises(ptp.PlanError) as excinfo:
        ptp.prepare_tp_pack(source.write_inventory(), _plan(source, slices=overlap), tmp_path / "pack")
    assert "vocabulary" in str(excinfo.value)


def test_a_vocabulary_slice_that_leaves_the_vocab_axis_whole_is_not_a_trim(source: Source, tmp_path: Path):
    """The real plan replicates token_embd by splitting the width, not the rows: that is allowed."""
    def width_replica(slices):
        return [entry for entry in slices if entry["tensor"] != EMBED] + [
            _slice(EMBED, rank, 0, 0, EMBED_WIDTH, "embed", source, replicated=True) for rank in range(3)
        ]

    manifest = ptp.prepare_tp_pack(source.write_inventory(), _plan(source, slices=width_replica),
                                  tmp_path / "pack")
    assert manifest["vocabulary"]["trimmed"] is False
    assert any("whole vocabulary" in note for note in manifest["vocabulary"]["note"])


def test_a_slice_with_no_rank_and_not_replicated_is_refused(source: Source, tmp_path: Path):
    def orphan(slices):
        for entry in slices:
            if entry["tensor"] == GATE_UP and entry["rank"] == 0:
                del entry["rank"]
        return slices

    with pytest.raises(ptp.PlanError) as excinfo:
        ptp.prepare_tp_pack(source.write_inventory(), _plan(source, slices=orphan), tmp_path / "pack")
    assert "rank" in str(excinfo.value)


# ------------------------------------------------------------------ output integrity


def test_an_interrupted_write_leaves_nothing_published(source: Source, tmp_path: Path):
    out = tmp_path / "pack"

    def abort(index: int, segment: dict) -> None:
        if index == 6:
            raise RuntimeError("power lost")

    with pytest.raises(RuntimeError):
        ptp.prepare_tp_pack(source.write_inventory(), _plan(source), out, materialize=True, progress=abort)
    assert not out.exists(), "a half-written pack must not be published"
    assert [item.name for item in tmp_path.iterdir()] == ["src"], "the staging directory must be cleaned up"


def test_the_output_is_not_clobbered_without_asking(source: Source, tmp_path: Path):
    out = tmp_path / "pack"
    ptp.prepare_tp_pack(source.write_inventory(), _plan(source), out, materialize=True)
    first = _rank_file(out, 0).read_bytes()
    with pytest.raises(ptp.PackError):
        ptp.prepare_tp_pack(source.write_inventory(), _plan(source), out, materialize=True)
    assert _rank_file(out, 0).read_bytes() == first
    ptp.prepare_tp_pack(source.write_inventory(), _plan(source), out, materialize=True, overwrite=True)
    assert _rank_file(out, 0).read_bytes() == first


def test_materializing_needs_its_own_free_disk(source: Source, tmp_path: Path):
    out = tmp_path / "pack"
    tight = lambda path: 50 * GIB + 1024          # barely above the floor, and it must keep room too
    with pytest.raises(ptp.DiskSpaceError) as excinfo:
        ptp.prepare_tp_pack(source.write_inventory(), _plan(source), out, materialize=True, free_space_probe=tight)
    assert "53687091200" in str(excinfo.value)
    assert not out.exists()


def test_plan_mode_does_not_consult_the_disk_probe(source: Source, tmp_path: Path):
    def refusing(path):
        raise AssertionError("plan mode must not ask about disk space")

    manifest = ptp.prepare_tp_pack(source.write_inventory(), _plan(source), tmp_path / "pack",
                                  free_space_probe=refusing)
    assert manifest["materialized"] is False


def test_a_tampered_rank_byte_fails_verification(source: Source, tmp_path: Path):
    out = tmp_path / "pack"
    manifest = ptp.prepare_tp_pack(source.write_inventory(), _plan(source), out, materialize=True)
    assert ptp.verify_tp_pack(source.write_inventory(), out).ok
    victim = _segments(manifest, 1, GATE_UP)[0]
    path = _rank_file(out, 1)
    blob = bytearray(path.read_bytes())
    blob[victim["dest_offset"] + 4096] ^= 0x01
    path.write_bytes(bytes(blob))
    report = ptp.verify_tp_pack(source.write_inventory(), out)
    assert not report.ok
    assert any(GATE_UP in line for line in report.mismatches)
    assert any("stopped at the first mismatch" in line for line in report.not_checked)


def test_a_tampered_byte_inside_a_strided_run_fails_verification(source: Source, tmp_path: Path):
    out = tmp_path / "pack"
    manifest = ptp.prepare_tp_pack(source.write_inventory(), _plan(source), out, materialize=True)
    segment = _segments(manifest, 0, EXPS)[0]
    path = _rank_file(out, 0)
    blob = bytearray(path.read_bytes())
    blob[segment["dest_offset"] + segment["source_length"] - 3] ^= 0x80
    path.write_bytes(bytes(blob))
    report = ptp.verify_tp_pack(source.write_inventory(), out)
    assert not report.ok
    assert any(EXPS in line for line in report.mismatches)


def test_verification_of_another_inventory_is_a_mismatch(source: Source, tmp_path: Path):
    inventory_path = source.write_inventory()
    out = tmp_path / "pack"
    ptp.prepare_tp_pack(inventory_path, _plan(source), out, materialize=True)
    moved = source.inventory()
    moved["source"]["revision"] = "rev-0002"
    other = source.write_inventory("moved-revision.json")
    other.write_text(json.dumps(moved))
    plan = _plan(source)
    plan["source_lock"]["inventory_hash"] = ptp.inventory_hash(moved)
    plan["source_lock"]["revision"] = "rev-0002"
    # same bytes on disk, a different pinned revision: the pack must say it is not this source
    ptp.prepare_tp_pack(other, plan, tmp_path / "pack2", materialize=True)
    assert ptp.verify_tp_pack(inventory_path, out).ok
    other_report = ptp.verify_tp_pack(other, out)
    assert not other_report.ok
    assert any("different inventory" in line for line in other_report.mismatches)


def test_the_report_names_what_it_did_not_check(source: Source, tmp_path: Path):
    out = tmp_path / "pack"
    ptp.prepare_tp_pack(source.write_inventory(), _plan(source), out, materialize=True)
    report = ptp.verify_tp_pack(source.write_inventory(), out)
    assert report.ok, report.mismatches
    text = " | ".join(report.checked)
    assert "byte-for-byte" in text and "sha256" in text
    joined = " | ".join(report.not_checked)
    assert "NAS staging" in joined, "the unrun download step must be stated"
    assert "layer7_norm.bin" in joined, "the file with no recorded hash must be called out"
    assert "not a running model" in joined
    assert report.files_hash_checked == len(source.files) - 1
    assert report.files_size_checked == len(source.files)


def test_the_cli_plans_then_materializes_then_verifies(source: Source, tmp_path: Path):
    inventory_path = source.write_inventory()
    plan_path = tmp_path / "tp-plan.json"
    plan_path.write_text(json.dumps(_plan(source)))
    out = tmp_path / "pack"
    assert ptp.main(["--inventory", str(inventory_path), "--plan", str(plan_path), "--output", str(out),
                     "--quiet"]) == 0
    assert not out.exists()
    assert ptp.main(["--inventory", str(inventory_path), "--plan", str(plan_path), "--output", str(out),
                     "--materialize", "--min-free-gib", "0", "--quiet"]) == 0
    assert out.is_dir()
    assert ptp.main(["--inventory", str(inventory_path), "--plan", str(plan_path), "--output", str(out),
                     "--verify"]) == 0
    assert ptp.main(["--inventory", str(inventory_path), "--plan", str(plan_path), "--output",
                     str(tmp_path / "nope"), "--verify"]) == 1


def test_plan_mode_agrees_with_the_materialized_layout(source: Source, tmp_path: Path):
    planned = ptp.prepare_tp_pack(source.write_inventory(), _plan(source), tmp_path / "pack")
    made = ptp.prepare_tp_pack(source.write_inventory(), _plan(source), tmp_path / "pack2", materialize=True)
    shape = [(s["rank"], s["tensor"], s["kind"], s["dest_offset"], s["source_length"], s["pattern"])
             for s in planned["segments"]]
    assert shape == [(s["rank"], s["tensor"], s["kind"], s["dest_offset"], s["source_length"], s["pattern"])
                     for s in made["segments"]]
    for rank in ("0", "1", "2"):
        assert planned["rank_bytes"][rank] == made["rank_bytes"][rank]
    assert planned["run_count"] == made["run_count"]


def test_the_rank_partition_is_uneven_because_the_agreed_cut_is_uneven(source: Source, tmp_path: Path):
    manifest = ptp.prepare_tp_pack(source.write_inventory(), _plan(source), tmp_path / "pack")
    rows = source.row_bytes(GATE_UP)
    gate = {rank: sum(seg["source_length"] for seg in _segments(manifest, rank, GATE_UP))
            for rank in range(3)}
    assert gate == {0: 256 * rows, 1: 192 * rows, 2: 192 * rows}
    down = {rank: sum(seg["source_length"] for seg in _segments(manifest, rank, DOWN))
            for rank in range(3)}
    assert down[0] == 4 * DOWN_BLOCK_BYTES * DOWN_OUT
    assert down[1] == down[2] == 3 * DOWN_BLOCK_BYTES * DOWN_OUT
