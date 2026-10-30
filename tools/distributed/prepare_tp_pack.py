#!/usr/bin/env python3
"""Prepare and verify the per-rank artifacts for a Strata TP3 run.

A rank artifact is a byte-range view over the original, pinned source files.  Nothing here
recomputes a scale, requantizes a code or trims a vocabulary: a rank takes whole blocks, with the
native scale tail that lives inside each block, and copies them as they are.  Scales kept as their
own tensor travel through the same range logic, so they are copied, not rebuilt.

Blocks run along the contiguous axis.  A model file stores a tensor with its contiguous dimension
first, so a [in, out] weight holds one whole row of blocks per output index, and a stacked expert
[in, out, experts] one row per (out, expert), with the output axis the inner of the two: one expert's rows
lie next to each other, and the next expert starts a full set of output rows later.  A Q4_K block is 256
weights of the input width and
144 bytes wherever it appears, which is exactly what the inventory's sizes say: for
blk.0.ffn_gate_shexp.weight at [2560, 640] with byte_length 921600, 6400 blocks of 144 bytes come
out of axis 0 and nothing sensible comes out of the last axis.  So a cut of whole output rows
copies whole blocks and needs no alignment at all, while a cut of the input width has to land on a
block edge.  This module reads the row stride from the tensor's own sizes and checks it against the
stride the plan declares.

Two modes, kept apart on purpose:

* ``prepare_tp_pack(source, plan, output, materialize=False)`` reads the inventory and the plan,
  checks every length and offset against the real size of the source file, and returns the
  candidate manifest.  It opens no output file at all, so it is safe against a source that is still
  copying in.
* ``prepare_tp_pack(..., materialize=True)`` writes rank-local files.  It asks for spare disk of its
  own first (50 GiB by default, on top of the bytes it is about to write), writes into a sibling
  staging directory, and only then moves it into place, so a run cut in half leaves no
  half-published pack behind.  It returns the same manifest with a byte-equality receipt.

``verify_tp_pack(source, output)`` reads a published pack back and compares every copied block and
scale against the source byte for byte, and its report says both what it checked and what it left
unproven.

How much work each mode does: plan mode reads only the arithmetic (file sizes and the ranges the
plan names), so it costs a few stats and stays safe against a source that is still copying in.
Recomputing a source file's sha256 reads that whole file, so it happens on a materialized write and
on a verify, never in plan mode.  A file the inventory leaves with a null sha256 is size-checked and
named in the report, not silently counted as hashed.

On the shared schema: the contract lists ``range``, ``local_shape``, ``valid_lanes`` and
``replicated`` per slice without naming a rank.  This module reads a slice as one rank's share when
it carries ``rank``, and as a copy for every rank when it carries ``"replicated": true``.

A segment is one run, or a constant-stride set of runs, or a general pattern of run counts and
strides, so a cut of the contiguous axis -- a row piece repeated over every (out, expert) row -- is
one entry that verification walks run by run, not thousands of them.

Tests: $VENV/bin/python -m pytest tests/distributed/test_prepare_tp_pack.py -q
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import uuid
from dataclasses import dataclass, field
from functools import reduce
from pathlib import Path
from typing import Callable, Iterator, Sequence

try:                                    # numpy makes the strided gather one copy instead of a loop
    import numpy as _np
except Exception:                       # pragma: no cover - the toolchain venv has numpy
    _np = None

SCHEMA_VERSION = 1
MANIFEST_VERSION = 1
MANIFEST_NAME = "tp-pack-v1.json"

# The materialized path keeps this much disk free after it writes, so a rank file never lands next
# to a full volume.
DEFAULT_MIN_FREE_BYTES = 50 * 1024**3

# Bytes held in hand at a time: one read of the source, one write or compare of the destination.
CHUNK_BYTES = 1 << 26

VOCABULARY_FAMILIES = ("embed", "head")


class PackError(Exception):
    """Base for every refusal here: a bad plan, a bad source, or a pack that does not match."""


class InventoryError(PackError):
    """The inventory does not describe a source this pack can be built from."""


class PlanError(PackError):
    """The plan contradicts the inventory, the geometry, or a quantization boundary."""


class SourceError(PackError):
    """A source file is missing, or is not the file the inventory pinned."""


class RangeError(PackError):
    """The plan asks to read past the real end of a source file."""


class DiskSpaceError(PackError):
    """Materializing would leave less free disk than the caller is willing to keep."""


class PackMismatch(PackError):
    """A byte written into a rank file is not the source byte it claims to be."""


def _prod(values) -> int:
    return reduce(lambda a, b: a * b, values, 1)


@dataclass(frozen=True)
class Segment:
    """Bytes one rank owns: one run, or runs at a fixed stride, or a general run pattern.

    ``pattern`` is a list of (count, stride) levels in increasing stride, so the runs sit at
    ``source_offset + sum(i_level * stride_level)`` for every digit, and they are enumerated in
    increasing order.  ``pattern == ()`` is a single run.  The destination always holds the runs
    back to back, so ``source_length`` is the size of the destination block.
    """

    tensor: str
    kind: str                 # "codes" | "scales" | "padding"
    rank: int
    source_path: str | None   # inventory-relative; None for padding
    source_offset: int | None
    run_bytes: int
    pattern: tuple[tuple[int, int], ...]
    dest_offset: int
    scale_of: str | None

    @property
    def source_length(self) -> int:
        return self.run_bytes * _prod(count for count, _ in self.pattern)

    @property
    def run_count(self) -> int:
        return _prod(count for count, _ in self.pattern) or 1

    def last_end(self) -> int:
        if self.source_offset is None:
            return self.dest_offset + self.source_length
        return (self.source_offset + sum((count - 1) * stride for count, stride in self.pattern)
                + self.run_bytes)

    def runs(self) -> Iterator[tuple[int, int]]:
        """(source_offset, length) for every run, in the order they fill the destination."""
        if self.source_offset is None:
            return iter(())
        if not self.pattern:
            yield self.source_offset, self.run_bytes
            return
        counts = [count for count, _ in self.pattern]
        strides = [stride for _, stride in self.pattern]
        digits = [0] * len(counts)
        while True:
            yield self.source_offset + sum(d * s for d, s in zip(digits, strides)), self.run_bytes
            for position in range(len(digits) - 1, -1, -1):
                digits[position] += 1
                if digits[position] < counts[position]:
                    break
                digits[position] = 0
            else:
                return

    def as_dict(self) -> dict:
        return {
            "tensor": self.tensor,
            "kind": self.kind,
            "rank": self.rank,
            "source_path": self.source_path,
            "source_offset": self.source_offset,
            "run_bytes": self.run_bytes,
            "pattern": [[count, stride] for count, stride in self.pattern],
            "source_length": self.source_length,
            "run_count": self.run_count,
            "dest_offset": self.dest_offset,
            "scale_of": self.scale_of,
        }

    @staticmethod
    def from_dict(raw: dict) -> "Segment":
        return Segment(
            tensor=raw["tensor"],
            kind=raw["kind"],
            rank=int(raw["rank"]),
            source_path=raw["source_path"],
            source_offset=None if raw["source_offset"] is None else int(raw["source_offset"]),
            run_bytes=int(raw["run_bytes"]),
            pattern=tuple((int(count), int(stride)) for count, stride in raw["pattern"]),
            dest_offset=int(raw["dest_offset"]),
            scale_of=raw["scale_of"],
        )


@dataclass
class VerificationReport:
    """What ``verify_tp_pack`` proved, and what it left alone."""

    ok: bool
    status: str
    world: int
    runs_compared: int
    bytes_compared: int
    padding_runs_checked: int
    files_size_checked: int
    files_hash_checked: int
    checked: list[str] = field(default_factory=list)
    not_checked: list[str] = field(default_factory=list)
    mismatches: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "status": self.status,
            "world": self.world,
            "runs_compared": self.runs_compared,
            "bytes_compared": self.bytes_compared,
            "padding_runs_checked": self.padding_runs_checked,
            "files_size_checked": self.files_size_checked,
            "files_hash_checked": self.files_hash_checked,
            "checked": self.checked,
            "not_checked": self.not_checked,
            "mismatches": self.mismatches,
        }

    def __str__(self) -> str:
        return json.dumps(self.as_dict(), indent=2, sort_keys=True)


# --------------------------------------------------------------------- small helpers


def _canonical_json(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def content_hash(obj) -> str:
    """Stable hash of a JSON-like value: the identity of an inventory, a plan, or a manifest."""
    return hashlib.sha256(_canonical_json(obj)).hexdigest()


def inventory_hash(inventory: dict) -> str:
    return content_hash(inventory)


def inventory_hash_variants(inventory: dict) -> set[str]:
    """The hashes a plan may legitimately carry for this inventory.

    This module canonicalizes with tight separators; the Task 1 inventory writer hashes
    ``json.dumps(obj, sort_keys=True)`` with the default separators.  Same content, two digests, and
    refusing a plan over a separator would be a false refusal at integration time.
    """
    return {inventory_hash(inventory),
            hashlib.sha256(json.dumps(inventory, sort_keys=True).encode("utf-8")).hexdigest()}


def plan_content_hash(plan: dict) -> str:
    return content_hash({key: value for key, value in plan.items() if key != "plan_hash"})


def read_json(path: Path, what: str) -> dict:
    try:
        with open(path, "rb") as handle:
            return json.loads(handle.read())
    except FileNotFoundError as exc:
        raise InventoryError(f"{what} not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise InventoryError(f"{what} is not valid JSON: {path}: {exc}") from exc


def _load(value, what: str, expected_key: str) -> tuple[dict, Path]:
    """Accept a path, a directory holding the file, or an already loaded dict."""
    if isinstance(value, dict):
        if expected_key not in value:
            raise InventoryError(f"{what} dict has no {expected_key!r} key")
        return value, Path.cwd()
    path = Path(value).expanduser()
    if path.is_dir():
        path = path / ("inventory.json" if expected_key == "source" else "tp-plan.json")
    if not path.is_file():
        raise InventoryError(f"{what} not found: {path}")
    return read_json(path, what), path.resolve().parent


def resolve_source_file(inventory_root: Path, rel_path: str, source_root: Path | None) -> Path:
    """Source paths are inventory-relative; ``source_root`` says where they really live."""
    candidate = Path(rel_path)
    if candidate.is_absolute() and source_root is None:
        return candidate
    base = Path(source_root).expanduser() if source_root is not None else inventory_root
    return base / candidate


def _require(condition: bool, message: str, error: type[PackError] = PlanError) -> None:
    if not condition:
        raise error(message)


def _existing_ancestor(path: Path) -> Path:
    cursor = path
    while not cursor.exists() and cursor != cursor.parent:
        cursor = cursor.parent
    return cursor


def _tensor_map(inventory: dict) -> dict[str, dict]:
    tensors = inventory.get("tensors")
    _require(isinstance(tensors, list) and bool(tensors), "inventory has no tensors", InventoryError)
    by_name: dict[str, dict] = {}
    for entry in tensors:
        name = entry.get("name")
        _require(isinstance(name, str) and bool(name), "tensor entry without a name", InventoryError)
        _require(name not in by_name, f"duplicate tensor in the inventory: {name}", InventoryError)
        by_name[name] = entry
    return by_name


def _file_sizes(inventory: dict, inventory_root: Path, source_root: Path | None) -> dict[str, int]:
    """Real sizes on disk, which is what every range is finally checked against."""
    sizes: dict[str, int] = {}
    for entry in inventory["source"].get("files", []):
        rel = entry["path"]
        path = resolve_source_file(inventory_root, rel, source_root)
        if not path.is_file():
            raise SourceError(f"source file missing: {path}")
        real = path.stat().st_size
        declared = entry.get("size")
        if declared is not None and int(declared) != real:
            raise SourceError(f"source size differs from the inventory: {path}: inventory says {declared}, "
                              f"the disk has {real}")
        sizes[rel] = real
    return sizes


def check_source_hashes(inventory: dict, inventory_root: Path, source_root: Path | None) -> tuple[int, list[str]]:
    """Hash every source file that carries a recorded sha256.  Returns (files checked, unsigned paths)."""
    checked = 0
    unsigned: list[str] = []
    for entry in inventory["source"].get("files", []):
        recorded = entry.get("sha256")
        if not recorded:
            unsigned.append(entry["path"])
            continue
        path = resolve_source_file(inventory_root, entry["path"], source_root)
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        got = digest.hexdigest()
        if got != recorded:
            raise SourceError(f"source hash differs from the inventory: {path}: inventory says {recorded}, "
                              f"the disk gives {got}")
        checked += 1
    return checked, unsigned


# -------------------------------------------------------------------- byte geometry


def _tensor_span(tensor: dict) -> tuple[int, int]:
    offset, length = tensor.get("byte_offset"), tensor.get("byte_length")
    _require(offset is not None and length is not None,
             f"tensor {tensor['name']} has no byte_offset/byte_length in the inventory", InventoryError)
    offset, length = int(offset), int(length)
    _require(offset >= 0 and length > 0, f"tensor {tensor['name']}: empty or negative byte range", InventoryError)
    return offset, length


def _check_inside(rel: str, file_size: int, offset: int, length: int, what: str) -> None:
    if offset < 0 or length < 0 or offset + length > file_size:
        raise RangeError(f"the plan reads outside the source file {rel}: {what} wants [{offset}, "
                         f"{offset + length}) and the file holds {file_size} bytes")


def _layout(tensor: dict) -> dict:
    """Row size, row count and the axis the quantization blocks run along, from the tensor's sizes.

    The contiguous dimension comes first in a model file, so one row is a whole run along axis 0 and
    the rows are the rest of the dimensions.  Blocks therefore run along axis 0, and a block's byte
    size falls out of ``row_bytes`` rather than out of a codec table this module would have to keep
    in step with the engine.
    """
    shape = [int(x) for x in tensor.get("shape", [])]
    _require(bool(shape), f"tensor {tensor['name']} has no shape", InventoryError)
    _require(all(x > 0 for x in shape), f"tensor {tensor['name']}: zero or negative extent in {shape}")
    total = int(tensor["byte_length"])
    row_count = _prod(shape[1:])
    _require(total % row_count == 0,
             f"tensor {tensor['name']}: byte_length {total} is not a whole number of rows over {row_count} rows")
    row_bytes = total // row_count
    block_size = tensor.get("block_size")
    if block_size is None:
        _require(row_bytes % shape[0] == 0,
                 f"tensor {tensor['name']}: a row of {row_bytes} bytes is not a whole number of items over "
                 f"{shape[0]} items, so this cut cannot be proven item-aligned")
        return {"shape": shape, "row_bytes": row_bytes, "row_count": row_count, "block_axis": None,
                "unit_bytes": row_bytes // shape[0], "block_size": None}
    block_size = int(block_size)
    _require(block_size > 0, f"tensor {tensor['name']}: block size {block_size} is not positive")
    if len(shape) > 1 and shape[0] % block_size != 0:
        _require(shape[-1] % block_size != 0,
                 f"tensor {tensor['name']}: blocks of {block_size} divide the last axis ({shape[-1]}) but not the "
                 f"contiguous axis ({shape[0]}); this packer copies whole rows of blocks and does not support "
                 f"blocks along the outer axis")
        raise PlanError(f"tensor {tensor['name']}: neither the contiguous axis ({shape[0]}) nor the last axis "
                        f"({shape[-1]}) is a whole number of blocks of {block_size}")
    _require(shape[0] % block_size == 0,
             f"tensor {tensor['name']}: extent {shape[0]} is not a whole number of blocks of {block_size}")
    blocks_per_row = shape[0] // block_size
    _require(row_bytes % blocks_per_row == 0,
             f"tensor {tensor['name']}: a row of {row_bytes} bytes does not divide into {blocks_per_row} blocks, "
             f"so this codec's scale tail is not block-proportional and the cut cannot be proven byte-exact")
    return {"shape": shape, "row_bytes": row_bytes, "row_count": row_count, "block_axis": 0,
            "unit_bytes": row_bytes // blocks_per_row, "block_size": block_size}


def _row_stride_bytes(layout: dict, axis: int) -> int:
    """Bytes between one step along a non-contiguous axis.

    A model file stores a tensor with its fastest-varying dimension first, so the axis right after the
    contiguous one is the inner one and each later axis steps over the whole block of axes before it.  For
    a stacked expert [in, out, experts] that means the 32 output rows of one expert sit next to each other,
    and the next expert starts 32 rows on: one expert is a contiguous run, and the experts of one output
    row are 32 rows apart.
    """
    shape = layout["shape"]
    return layout["row_bytes"] * _prod(shape[1:axis])


def _normalize(offset: int, run_bytes: int, levels: Sequence[tuple[int, int]]) -> tuple[int, int, tuple]:
    """Fold runs that sit right against the current run into it, and drop levels that select nothing."""
    kept = sorted(((int(count), int(stride)) for count, stride in levels if count > 1),
                  key=lambda pair: pair[1])
    changed = True
    while changed:
        changed = False
        if kept and kept[0][1] == run_bytes:
            run_bytes *= kept[0][0]
            kept = kept[1:]
            changed = True
    return offset, run_bytes, tuple(kept)


def _cut(layout: dict, axis: int, start: int, end: int, name: str) -> tuple[int, int, tuple]:
    """(offset from the tensor start, run_bytes, levels) for a cut of one axis.

    A cut of the contiguous axis has to land on a block edge, and gives one run per row at the row
    stride.  A cut of any other axis takes whole rows, which is why any row boundary is valid: the
    blocks inside a row come along whole, tail and all.
    """
    shape = layout["shape"]
    _require(0 <= axis < len(shape), f"tensor {name}: axis {axis} outside shape {shape}")
    _require(0 <= start < end <= shape[axis],
             f"tensor {name}: range [{start}, {end}) outside extent {shape[axis]} on axis {axis}")
    if axis == 0:
        if layout["block_axis"] == 0:
            block_size = layout["block_size"]
            _require(start % block_size == 0 and end % block_size == 0,
                     f"tensor {name}: cut [{start}, {end}) splits a quantization block of {block_size} on the "
                     f"contiguous axis; a rank may take whole blocks only, and no scale is recomputed to make it fit")
            units = (end - start) // block_size
        else:
            units = end - start
        start_units = start if layout["block_axis"] is None else start // layout["block_size"]
        # A cut that starts inside a row has to start there: rank 1 of a [640, 128] down projection owns
        # blocks 4..6 of every output row, not blocks 0..2, and the same holds for the scale row beside it.
        offset = start_units * layout["unit_bytes"]
        run_bytes = units * layout["unit_bytes"]
        if layout["row_count"] == 1:
            return _normalize(offset, run_bytes, ())
        if run_bytes == layout["row_bytes"]:
            return _normalize(offset, run_bytes * layout["row_count"], ())   # rows are adjacent: one range
        return _normalize(offset, run_bytes, ((layout["row_count"], layout["row_bytes"]),))
    stride = _row_stride_bytes(layout, axis)
    levels = [(end - start, stride)]
    levels += [(shape[other], _row_stride_bytes(layout, other)) for other in range(1, len(shape)) if other != axis]
    return _normalize(start * stride, layout["row_bytes"], levels)


def _valid_count(slice_entry: dict, requested: int, world: int, rank: int) -> int:
    """How many items of the local range really exist in the source, in the range's own units.

    The rest of the range is padding: the local share is padded to the agreed slot count and the
    tail is written as zeros instead of read out of the source.
    """
    lanes = slice_entry.get("valid_lanes")
    if lanes is None:
        return requested
    if isinstance(lanes, int):
        return int(lanes)
    _require(len(lanes) == world,
             f"slice {slice_entry['tensor']}: valid_lanes has {len(lanes)} entries for a world of {world}")
    return int(lanes[rank])


def _ranks_of(slice_entry: dict, world: int) -> list[int]:
    if "rank" in slice_entry:
        rank = int(slice_entry["rank"])
        _require(0 <= rank < world, f"slice {slice_entry['tensor']}: rank {rank} outside world {world}")
        return [rank]
    _require(bool(slice_entry.get("replicated")),
             f"slice {slice_entry['tensor']}: it names no rank and is not marked replicated, so nobody owns it")
    return list(range(world))


def _cross_check_plan_fields(slice_entry: dict, tensor: dict, layout: dict, axis: int, start: int,
                             end: int) -> None:
    """Use what the plan declares as a second opinion on the inventory, not as a free pass."""
    declared_stride = slice_entry.get("byte_row_stride")
    if declared_stride is not None:
        _require(int(declared_stride) == layout["row_bytes"],
                 f"slice {slice_entry['tensor']}: the plan declares a row stride of {declared_stride} bytes and the "
                 f"inventory's sizes give {layout['row_bytes']}")
    declared_block = slice_entry.get("block_size")
    if declared_block is not None:
        _require(int(declared_block) == (layout["block_size"] or 0),
                 f"slice {slice_entry['tensor']}: the plan says block size {declared_block} and the inventory says "
                 f"{layout['block_size']}")
    codec, dtype = slice_entry.get("codec"), tensor.get("dtype")
    _require(codec in (None, dtype),
             f"slice {slice_entry['tensor']}: plan codec {codec!r} is not the source dtype {dtype!r}; this pack "
             f"copies codes, it does not requantize them to make a cut fit")
    _require(slice_entry.get("family") in (None, tensor.get("family")),
             f"slice {slice_entry['tensor']}: family {slice_entry.get('family')!r} is not the inventory family "
             f"{tensor.get('family')!r}")
    _require(slice_entry.get("global_shape") is None
             or [int(x) for x in slice_entry["global_shape"]] == layout["shape"],
             f"slice {slice_entry['tensor']}: global_shape {slice_entry.get('global_shape')} is not the inventory "
             f"shape {layout['shape']}")
    local_shape = slice_entry.get("local_shape")
    if local_shape is not None:
        local_shape = [int(x) for x in local_shape]
        _require(len(local_shape) == len(layout["shape"]),
                 f"slice {slice_entry['tensor']}: local_shape {local_shape} does not describe a tensor of rank "
                 f"{len(layout['shape'])}")
        _require(local_shape[axis] == end - start,
                 f"slice {slice_entry['tensor']}: local_shape {local_shape} says {local_shape[axis]} items on the "
                 f"cut axis and the range is {end - start}")
        for other, (local, global_extent) in enumerate(zip(local_shape, layout["shape"])):
            if other != axis:
                _require(local == global_extent,
                         f"slice {slice_entry['tensor']}: local_shape {local_shape} shrinks axis {other} to {local} "
                         f"of {global_extent}; a slice cuts one axis and takes the rest whole")
    cut_unit = slice_entry.get("cut_unit")
    if isinstance(cut_unit, int) and cut_unit > 1:
        _require(start % cut_unit == 0 and end % cut_unit == 0,
                 f"slice {slice_entry['tensor']}: cut [{start}, {end}) does not land on the plan's own cut unit of "
                 f"{cut_unit}")


def segments_for_slice(slice_entry: dict, tensors: dict[str, dict], sizes: dict[str, int],
                       world: int) -> list[Segment]:
    """Every byte run one slice contributes, per rank, with padding made explicit."""
    name = slice_entry["tensor"]
    _require(name in tensors, f"the plan slices an unknown tensor {name!r}")
    tensor = tensors[name]
    layout = _layout(dict(tensor, byte_length=_tensor_span(tensor)[1]))
    axis = int(slice_entry["axis"])
    start, end = (int(x) for x in slice_entry["range"])
    _require(0 <= start < end, f"slice {name}: empty or reversed range [{start}, {end})")
    _cross_check_plan_fields(slice_entry, tensor, layout, axis, start, end)

    rel = tensor["file"]
    _require(rel in sizes, f"slice {name}: tensor file {rel!r} is not among the inventory source files")
    file_size = sizes[rel]
    base, total = _tensor_span(tensor)
    _check_inside(rel, file_size, base, total, f"tensor {name}")

    out: list[Segment] = []
    for rank in _ranks_of(slice_entry, world):
        valid = _valid_count(slice_entry, end - start, world, rank)
        _require(0 <= valid <= end - start,
                 f"slice {name}: rank {rank} valid_lanes {valid} outside a range of {end - start} items")
        offset, run_bytes, levels = _cut(layout, axis, start, start + valid, name)
        segment = Segment(tensor=name, kind="codes", rank=rank, source_path=rel, source_offset=base + offset,
                          run_bytes=run_bytes, pattern=levels, dest_offset=0, scale_of=None)
        _check_inside(rel, file_size, segment.source_offset, segment.last_end() - segment.source_offset,
                      f"tensor {name} rank {rank}")
        out.append(segment)
        if valid < end - start:
            full_offset, full_run, full_levels = _cut(layout, axis, start, end, name)
            full = Segment(tensor=name, kind="codes", rank=rank, source_path=rel, source_offset=base + full_offset,
                           run_bytes=full_run, pattern=full_levels, dest_offset=0, scale_of=None)
            _require(axis != 0 or layout["block_axis"] is None,
                     f"slice {name}: rank {rank} has a short share of the contiguous axis, and a part-block tail is "
                     f"not a defined padding; only whole-row shares may be padded")
            pad = full.source_length - segment.source_length
            _require(pad > 0, f"slice {name}: rank {rank} padding came out as {pad} bytes")
            out.append(Segment(tensor=name, kind="padding", rank=rank, source_path=None, source_offset=None,
                               run_bytes=pad, pattern=(), dest_offset=0, scale_of=None))

    if slice_entry.get("scale_ref"):
        out.extend(_scale_segments(slice_entry, tensors, sizes, world, axis, start, end))
    return out


def _scale_range(source_extent: int, scale_extent: int, start: int, end: int, block_size) -> tuple[int, int]:
    """Map a codes cut onto the scale tensor that describes it: one scale per block, or one per item."""
    if scale_extent in (source_extent, end - start):
        return start, end
    if block_size and source_extent == scale_extent * int(block_size):
        return start // int(block_size), end // int(block_size)
    raise PlanError(f"a scale tensor of extent {scale_extent} cannot be mapped from a codes extent of "
                    f"{source_extent} for a cut of [{start}, {end}) with block {block_size}")


def _scale_segments(slice_entry: dict, tensors: dict[str, dict], sizes: dict[str, int], world: int,
                    axis: int, start: int, end: int) -> list[Segment]:
    """Runs for the scale tensor a slice names, cut to the same share in the source's own terms."""
    ref = slice_entry["scale_ref"]
    name = slice_entry["tensor"]
    _require(ref in tensors, f"slice {name}: scale_ref {ref!r} is not an inventory tensor")
    codes, scale = tensors[name], tensors[ref]
    codes_layout = _layout(dict(codes, byte_length=_tensor_span(codes)[1]))
    scale_layout = _layout(dict(scale, byte_length=_tensor_span(scale)[1]))
    source_extent = codes_layout["shape"][axis] if axis < len(codes_layout["shape"]) else codes_layout["shape"][-1]
    scale_axis = axis if axis < len(scale_layout["shape"]) else len(scale_layout["shape"]) - 1
    s_start, s_end = _scale_range(source_extent, scale_layout["shape"][scale_axis], start, end,
                                  codes_layout["block_size"])

    rel = scale["file"]
    _require(rel in sizes, f"scale {ref}: file {rel!r} is not among the inventory source files")
    file_size = sizes[rel]
    base, total = _tensor_span(scale)
    _check_inside(rel, file_size, base, total, f"scale {ref}")

    out: list[Segment] = []
    for rank in _ranks_of(slice_entry, world):
        offset, run_bytes, levels = _cut(scale_layout, scale_axis, s_start, s_end, ref)
        segment = Segment(tensor=ref, kind="scales", rank=rank, source_path=rel, source_offset=base + offset,
                          run_bytes=run_bytes, pattern=levels, dest_offset=0, scale_of=name)
        _check_inside(rel, file_size, segment.source_offset, segment.last_end() - segment.source_offset,
                      f"scale {ref} for {name} rank {rank}")
        out.append(segment)
    return out


def _assign_dest_offsets(segments: Sequence[Segment]) -> list[Segment]:
    """Lay each rank's runs out back to back in plan order and record where each one lands."""
    cursor: dict[int, int] = {}
    placed: list[Segment] = []
    for segment in segments:
        start = cursor.get(segment.rank, 0)
        placed.append(Segment(tensor=segment.tensor, kind=segment.kind, rank=segment.rank,
                              source_path=segment.source_path, source_offset=segment.source_offset,
                              run_bytes=segment.run_bytes, pattern=segment.pattern, dest_offset=start,
                              scale_of=segment.scale_of))
        cursor[segment.rank] = start + segment.source_length
    return placed


def _check_vocabulary(plan_geometry: dict, slices: Sequence[dict], tensors: dict[str, dict],
                    world: int) -> dict:
    """The logical vocabulary stays whole.

    Rows may be split across ranks (a row-sharded embedding) or carried whole, but no row may be
    dropped to make the model load and no row may be lost between ranks.  Only the axis that really
    carries the vocabulary counts: for a [n_embd, n_vocab] embedding that is the second axis, so a
    rank that takes the whole tensor and splits the *width* still holds every row.
    """
    n_vocab = int(plan_geometry.get("n_vocab", 0) or 0)
    modes: list[str] = []
    for name in sorted({entry["tensor"] for entry in slices
                        if entry.get("family") in VOCABULARY_FAMILIES}):
        tensor = tensors.get(name)
        _require(tensor is not None, f"vocabulary tensor {name!r} is not in the inventory")
        shape = [int(x) for x in tensor.get("shape", [])]
        vocab_axis = next((axis for axis, extent in enumerate(shape) if n_vocab and extent == n_vocab), None)
        if vocab_axis is None:
            modes.append(f"{name}: no axis of {shape} is the vocabulary, so nothing to trim")
            continue
        entries = [entry for entry in slices if entry["tensor"] == name]
        whole_ranks = set()
        ranges: list[tuple[int, int]] = []
        for entry in entries:
            ranks = _ranks_of(entry, world)
            start, end = (int(x) for x in entry["range"])
            if int(entry["axis"]) != vocab_axis or (start == 0 and end == n_vocab):
                whole_ranks.update(ranks)
            else:
                ranges.append((start, end))
        if whole_ranks:
            modes.append(f"{name}: the whole vocabulary ({n_vocab} rows) is carried by rank "
                         f"{sorted(whole_ranks)[0]}{'...' if len(whole_ranks) > 1 else ''}")
            continue
        cursor = 0
        for start, end in sorted(ranges):
            _require(start == cursor,
                     f"{name}: the vocabulary rows leave a gap at {cursor} (the next range starts at {start}); this "
                     f"pack copies rows, it does not drop them to make the model load")
            cursor = end
        _require(n_vocab == 0 or cursor == n_vocab,
                 f"{name}: the vocabulary rows reach {cursor} of {n_vocab}, which trims the logical vocabulary to make "
                 f"the model load")
        modes.append(f"{name}: {n_vocab} rows are split across the ranks and add up exactly")
    return {"logical_n_vocab": n_vocab or None, "families": sorted(VOCABULARY_FAMILIES), "trimmed": False,
            "note": modes or ["no vocabulary-row tensor in the plan, so there is nothing to trim"]}


def _check_source_lock(inventory: dict, plan: dict) -> None:
    lock = plan.get("source_lock") or {}
    recorded = lock.get("inventory_hash")
    if recorded:
        _require(recorded in inventory_hash_variants(inventory),
                 f"the plan is locked to inventory {recorded} but the inventory reads "
                 f"{inventory_hash(inventory)}", SourceError)
    lock_repo, inv_repo = lock.get("repo"), (inventory.get("source") or {}).get("repo")
    _require(not (lock_repo and inv_repo and lock_repo != inv_repo),
             f"the plan is locked to {lock_repo} but the inventory describes {inv_repo}", SourceError)


def build_segments(source, plan, *, source_root: Path | None = None
                   ) -> tuple[dict, dict, list[Segment], dict, Path]:
    """Inventory + plan -> the per-rank byte layout, with every range proven to fit the real files."""
    inventory, inv_root = _load(source, "inventory", "source")
    plan_doc, _ = _load(plan, "plan", "slices")

    _require(int(inventory.get("schema_version", 0)) == SCHEMA_VERSION,
             f"inventory schema_version {inventory.get('schema_version')} is not {SCHEMA_VERSION}", InventoryError)
    _require(int(plan_doc.get("schema_version", 0)) == SCHEMA_VERSION,
             f"plan schema_version {plan_doc.get('schema_version')} is not {SCHEMA_VERSION}")
    world = int(plan_doc.get("world", 0))
    _require(world >= 1, f"plan world {plan_doc.get('world')!r} is not a rank count")

    _check_source_lock(inventory, plan_doc)
    sizes = _file_sizes(inventory, inv_root, source_root)
    tensors = _tensor_map(inventory)

    slices = plan_doc.get("slices") or []
    _require(bool(slices), "the plan has no slices")
    segments: list[Segment] = []
    for entry in slices:
        for key in ("tensor", "axis", "range"):
            _require(key in entry, f"slice entry without {key}")
        segments.extend(segments_for_slice(entry, tensors, sizes, world))

    geometry = plan_doc.get("geometry") or inventory.get("geometry") or {}
    vocabulary = _check_vocabulary(geometry, slices, tensors, world)
    placed = _assign_dest_offsets(segments)

    rank_bytes = {rank: 0 for rank in range(world)}
    runs = 0
    for segment in placed:
        rank_bytes[segment.rank] += segment.source_length
        runs += segment.run_count
    _require(all(value > 0 for value in rank_bytes.values()),
             "some rank gets no bytes at all, which is not a tensor-parallel layout")

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "manifest_version": MANIFEST_VERSION,
        "materialized": False,
        "world": world,
        "plan_hash": plan_doc.get("plan_hash"),
        "plan_content_hash": plan_content_hash(plan_doc),
        "source_lock": plan_doc.get("source_lock"),
        "inventory_hash": inventory_hash(inventory),
        "repo": (inventory.get("source") or {}).get("repo"),
        "revision": (inventory.get("source") or {}).get("revision"),
        "geometry": geometry,
        "vocabulary": vocabulary,
        "source_files": [{"path": entry["path"], "size": sizes[entry["path"]], "sha256": entry.get("sha256")}
                         for entry in inventory["source"].get("files", [])],
        "memory": plan_doc.get("memory"),
        "activation": plan_doc.get("activation"),
        "segments": [segment.as_dict() for segment in placed],
        "segment_count": len(placed),
        "run_count": runs,
        "rank_files": None,
        "rank_bytes": {str(rank): rank_bytes[rank] for rank in sorted(rank_bytes)},
        "receipts": {},
        "not_verified": [
            "plan mode wrote nothing, so no destination was compared to the source",
            "the full pack download to NAS staging is a separate step and was not run here",
        ],
    }
    manifest["pack_hash"] = content_hash({k: v for k, v in manifest.items() if k != "pack_hash"})
    return inventory, plan_doc, placed, manifest, inv_root


def rank_view(manifest: dict, rank: int) -> list[dict]:
    """The source reads that make one rank's half of the model, in destination order."""
    return sorted((segment for segment in manifest["segments"] if segment["rank"] == rank),
                  key=lambda segment: segment["dest_offset"])


# -------------------------------------------------------------- reading a pattern


class _PatternReader:
    """Hands out a segment's bytes in destination order, reading each source file forward.

    A cut of the contiguous axis is one small piece per row, with a small gap between them: a
    million pieces on a stacked expert.  Seeking per piece would be a million seeks, so rows are read
    in blocks of whole rows and gathered in one copy (numpy when it is present, a plain loop when it
    is not), and a sparse segment costs the few seeks it always cost.
    """

    def __init__(self, path: Path, chunk_bytes: int = CHUNK_BYTES):
        self.path = path
        self.handle = open(path, "rb")
        self.chunk_bytes = chunk_bytes
        self.window_start = 0
        self.window = b""

    def close(self) -> None:
        self.handle.close()

    def _slice(self, offset: int, length: int) -> bytes:
        if length == 0:
            return b""
        if self.window_start <= offset and offset + length <= self.window_start + len(self.window):
            return self.window[offset - self.window_start: offset - self.window_start + length]
        want = max(length, self.chunk_bytes)
        self.handle.seek(offset)
        data = self.handle.read(want)
        if len(data) < length:
            raise SourceError(f"source {self.path} gave up short at offset {offset}: wanted {length}, got "
                              f"{len(data)}")
        self.window_start = offset
        self.window = data
        return data[:length]

    def pieces(self, segment: Segment) -> Iterator[bytes]:
        """Byte strings that fill the segment's destination block, in order, each at most chunk-sized."""
        if segment.source_path is None:                    # padding: the caller writes zeros
            return
        if not segment.pattern:
            produced = 0
            while produced < segment.source_length:
                length = min(self.chunk_bytes, segment.source_length - produced)
                yield self._slice(segment.source_offset + produced, length)
                produced += length
            return
        if len(segment.pattern) == 1 and segment.pattern[0][1] > segment.run_bytes:
            yield from self._strided(segment)
            return
        buffer = bytearray()
        for offset, length in segment.runs():
            if length > self.chunk_bytes:
                if buffer:
                    yield bytes(buffer)
                    buffer = bytearray()
                yield self._slice(offset, length)
                continue
            if len(buffer) + length > self.chunk_bytes:
                yield bytes(buffer)
                buffer = bytearray()
            buffer += self._slice(offset, length)
        if buffer:
            yield bytes(buffer)

    def _strided(self, segment: Segment) -> Iterator[bytes]:
        """One constant-stride level: read whole rows, hand back the owned prefix of each row."""
        count, stride = segment.pattern[0]
        run_bytes = segment.run_bytes
        rows_per_read = max(1, self.chunk_bytes // stride)
        done = 0
        while done < count:
            rows = min(rows_per_read, count - done)
            start = segment.source_offset + done * stride
            # Only the owned prefix of each row is wanted, so the read asks for the last row's prefix and not
            # for a whole stride: a cut that starts inside a row ends past the last stride the file holds.
            whole = rows - 1
            block = self._slice(start, whole * stride + run_bytes)
            tail = block[whole * stride:]
            if _np is not None:
                head = _np.frombuffer(block[:whole * stride], dtype=_np.uint8).reshape(whole, stride)[:, :run_bytes]
                yield head.copy().tobytes() + tail
            else:
                yield b"".join(block[row * stride: row * stride + run_bytes] for row in range(whole)) + tail
            done += rows


def _writers(paths: dict[int, Path]) -> dict[int, object]:
    return {rank: open(path, "wb") for rank, path in paths.items()}


# -------------------------------------------------------------------------- writing


def default_free_bytes(path: Path) -> int:
    return shutil.disk_usage(path).free


def require_free_space(path: Path, needed_bytes: int, min_free_bytes: int,
                       probe: Callable[[Path], int] = default_free_bytes) -> int:
    """Keep ``min_free_bytes`` free after writing ``needed_bytes``, or refuse to start."""
    free = int(probe(_existing_ancestor(path)))
    if free < needed_bytes + min_free_bytes:
        raise DiskSpaceError(f"materializing needs {needed_bytes} bytes and would leave {free} free; the caller "
                             f"keeps {min_free_bytes} free, so it wants {needed_bytes + min_free_bytes}")
    return free


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _readers_for(segments: Sequence[Segment], inv_root: Path, src_root: Path | None) -> dict[str, _PatternReader]:
    readers: dict[str, _PatternReader] = {}
    for segment in segments:
        if segment.source_path is None or segment.source_path in readers:
            continue
        readers[segment.source_path] = _PatternReader(resolve_source_file(inv_root, segment.source_path, src_root))
    return readers


def prepare_tp_pack(source, plan, output, materialize: bool = False, *,
                    source_root=None, min_free_bytes: int = DEFAULT_MIN_FREE_BYTES,
                    overwrite: bool = False, hash_source_files: bool = True,
                    free_space_probe: Callable[[Path], int] = default_free_bytes,
                    progress: Callable[[int, dict], None] | None = None) -> dict:
    """Build the rank artifact for a source and a plan; copy bytes only when ``materialize``.

    ``progress`` runs before each written segment.  Raising from it aborts the run and leaves
    ``output`` unpublished, which is also how the tests cut a write in half.
    """
    inventory, plan_doc, segments, manifest, inv_root = build_segments(source, plan, source_root=source_root)
    src_root = Path(source_root).expanduser() if source_root is not None else None
    output = Path(output).expanduser()

    if not materialize:
        # Nothing is written in this mode: no output path is opened, not even inspected.
        manifest["output"] = str(output)
        return manifest

    if hash_source_files:
        checked, unsigned = check_source_hashes(inventory, inv_root, src_root)
        manifest["receipts"]["source_hashes"] = {"status": "verified", "files": checked, "unsigned": sorted(unsigned)}
    else:
        manifest["receipts"]["source_hashes"] = {"status": "not run", "reason": "the caller asked to skip it"}

    needed = sum(segment.source_length for segment in segments)
    free_seen = require_free_space(output, needed, min_free_bytes, free_space_probe)
    manifest["disk"] = {"needed_bytes": needed, "free_bytes_seen": free_seen, "min_free_bytes": min_free_bytes,
                        "policy": "free space after the write stays at or above min_free_bytes"}

    staging = output.parent / f".{output.name}.staging-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    (staging / "ranks").mkdir(parents=True)
    readers: dict[str, _PatternReader] = {}
    handles: dict[int, object] = {}
    try:
        readers = _readers_for(segments, inv_root, src_root)
        handles = _writers({rank: staging / "ranks" / f"rank{rank}.bin"
                           for rank in range(manifest["world"])})
        written = 0
        for index, segment in enumerate(segments):
            if progress is not None:
                progress(index, segment.as_dict())
            handle = handles[segment.rank]
            if segment.source_path is None:
                # padding: no source byte behind it, and verify expects zeros of this exact size
                remaining = segment.source_length
                while remaining:
                    block = handle.write(b"\0" * min(remaining, CHUNK_BYTES))
                    remaining -= block
            else:
                for piece in readers[segment.source_path].pieces(segment):
                    handle.write(piece)
            written += segment.source_length
        for handle in handles.values():
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
        handles = {}
        for reader in readers.values():
            reader.close()
        readers = {}
        if written != needed:
            raise PackError(f"wrote {written} bytes and planned {needed}")

        manifest["rank_files"] = {
            str(rank): {"path": f"ranks/rank{rank}.bin",
                        "size": (staging / "ranks" / f"rank{rank}.bin").stat().st_size}
            for rank in range(manifest["world"])
        }

        # The receipt is the comparison itself: every copied run is read back out of the rank file
        # and put next to the source bytes again.
        compared = 0
        compared_bytes = 0
        readers = _readers_for(segments, inv_root, src_root)
        rank_handles = {rank: open(staging / "ranks" / f"rank{rank}.bin", "rb")
                        for rank in range(manifest["world"])}
        try:
            for segment in segments:
                if segment.source_path is None:
                    continue
                reader = readers[segment.source_path]
                dest = rank_handles[segment.rank]
                dest.seek(segment.dest_offset)
                position = 0
                for piece in reader.pieces(segment):
                    got = dest.read(len(piece))
                    if got != piece:
                        raise PackMismatch(
                            f"rank {segment.rank} {segment.tensor} kind={segment.kind}: source bytes at "
                            f"{segment.source_offset} + {position} are not what the rank file holds at "
                            f"{segment.dest_offset + position}")
                    position += len(piece)
                compared += 1
                compared_bytes += segment.source_length
        finally:
            for handle in rank_handles.values():
                handle.close()
            for reader in readers.values():
                reader.close()
        manifest["receipts"]["byte_equality"] = {
            "status": "verified", "segments": compared, "runs": sum(s.run_count for s in segments
                                                                    if s.source_path is not None),
            "bytes": compared_bytes,
            "note": "every copied block and scale was read back out of the rank file and compared to the source",
        }

        manifest["materialized"] = True
        manifest["output"] = str(output)
        manifest["not_verified"] = [
            "bytes outside the planned ranges were not compared",
            "no engine load or inference was run: matching bytes is not a running model",
            "the full pack download to NAS staging is a separate step and was not run here",
        ]
        if manifest["receipts"]["source_hashes"]["status"] == "not run":
            manifest["not_verified"].append("source content hashes were not recomputed, on request")
        unsigned = manifest["receipts"]["source_hashes"].get("unsigned") or []
        if unsigned:
            manifest["not_verified"].append("source files with no recorded sha256 were size-checked only: "
                                            + ", ".join(unsigned))
        manifest["pack_hash"] = content_hash({k: v for k, v in manifest.items() if k != "pack_hash"})

        with open(staging / MANIFEST_NAME, "wb") as handle:
            handle.write(_canonical_json(manifest))
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_dir(staging)
        _fsync_dir(staging / "ranks")

        if output.exists():
            _require(overwrite, f"output is already published: {output} (pass overwrite to replace it)", PackError)
            shutil.rmtree(output)
        os.replace(staging, output)
        _fsync_dir(output.parent)
        return manifest
    except BaseException:
        for handle in handles.values():
            try:
                handle.close()
            except Exception:
                pass
        for reader in readers.values():
            try:
                reader.close()
            except Exception:
                pass
        shutil.rmtree(staging, ignore_errors=True)
        raise


# ----------------------------------------------------------------------- verifying


def verify_tp_pack(source, output, *, source_root=None, hash_source_files: bool = True,
                   byte_equality: bool = True) -> VerificationReport:
    """Prove a published pack still matches its source, and say what was left unproven."""
    output = Path(output).expanduser()
    inventory, inv_root = _load(source, "inventory", "source")
    src_root = Path(source_root).expanduser() if source_root is not None else None

    checked: list[str] = []
    not_checked: list[str] = []
    mismatches: list[str] = []

    manifest_path = output / MANIFEST_NAME
    if not manifest_path.is_file():
        return VerificationReport(ok=False, status="missing manifest", world=0, runs_compared=0, bytes_compared=0,
                                  padding_runs_checked=0, files_size_checked=0, files_hash_checked=0,
                                  checked=checked, not_checked=not_checked,
                                  mismatches=[f"no {MANIFEST_NAME} under {output}"])
    manifest = read_json(manifest_path, "pack manifest")
    world = int(manifest.get("world", 0))
    segments = [Segment.from_dict(raw) for raw in manifest.get("segments", [])]

    # 1. the manifest is the one that was published
    recomputed = content_hash({k: v for k, v in manifest.items() if k != "pack_hash"})
    if manifest.get("pack_hash") != recomputed:
        mismatches.append(f"manifest pack_hash {manifest.get('pack_hash')} does not match its own content "
                          f"{recomputed}")
    else:
        checked.append("manifest integrity: pack_hash recomputed from the manifest and matched")

    # 2. the pack still claims the pinned inventory
    if manifest.get("inventory_hash") != inventory_hash(inventory):
        mismatches.append("the pack was built from a different inventory than the one passed in")
    else:
        checked.append("pack identity: the manifest's inventory_hash matches the inventory passed in "
                       f"({manifest['inventory_hash'][:12]})")

    # 3. the source files are still the size the pack was cut against
    sizes: dict[str, int] = {}
    try:
        sizes = _file_sizes(inventory, inv_root, src_root)
        checked.append(f"source files present at the size the inventory declares: {len(sizes)}")
    except PackError as exc:
        mismatches.append(str(exc))

    # 4. source content hashes
    files_hash_checked = 0
    if hash_source_files:
        try:
            files_hash_checked, unsigned = check_source_hashes(inventory, inv_root, src_root)
            if files_hash_checked:
                checked.append(f"source content sha256 matches the inventory: {files_hash_checked} files")
            if unsigned:
                not_checked.append("source files with no recorded sha256 were size-checked only: "
                                   + ", ".join(sorted(unsigned)))
        except PackError as exc:
            mismatches.append(str(exc))
    else:
        not_checked.append("source content hashes were not recomputed (the caller asked to skip)")

    # 5. every copied run, against the source, byte for byte
    runs_compared = 0
    bytes_compared = 0
    padding_checked = 0
    readers: dict[str, _PatternReader] = {}
    rank_handles: dict[int, object] = {}
    if byte_equality and not mismatches:
        broke_early = False
        try:
            readers = _readers_for(segments, inv_root, src_root)
            for segment in segments:
                if segment.source_path is None:
                    continue
                rel = segment.source_path
                if rel not in sizes:
                    mismatches.append(f"segment {segment.tensor} names a file the inventory does not list: {rel}")
                    broke_early = True
                    break
                if segment.last_end() > sizes[rel]:
                    mismatches.append(f"segment {segment.tensor} reads outside {rel}: its last byte is "
                                      f"{segment.last_end()} and the file holds {sizes[rel]}")
                    broke_early = True
                    break
                if segment.rank not in rank_handles:
                    rank_path = output / "ranks" / f"rank{segment.rank}.bin"
                    if not rank_path.is_file():
                        mismatches.append(f"rank file missing: {rank_path}")
                        broke_early = True
                        break
                    rank_handles[segment.rank] = open(rank_path, "rb")
                dest = rank_handles[segment.rank]
                dest.seek(segment.dest_offset)
                position = 0
                for piece in readers[rel].pieces(segment):
                    if dest.read(len(piece)) != piece:
                        mismatches.append(f"rank {segment.rank} {segment.tensor} kind={segment.kind} scale_of="
                                          f"{segment.scale_of}: source at {segment.source_offset} + {position} of "
                                          f"{rel} is not the rank file at {segment.dest_offset + position}")
                        broke_early = True
                        break
                    position += len(piece)
                if broke_early:
                    break
                runs_compared += segment.run_count
                bytes_compared += segment.source_length
        finally:
            for handle in rank_handles.values():
                handle.close()
            for reader in readers.values():
                reader.close()
        if runs_compared:
            checked.append(f"byte-for-byte reconstruction against the source: {runs_compared} runs, "
                           f"{bytes_compared} bytes of codes and scales, each read at its recorded offsets")
        if broke_early:
            not_checked.append("the byte comparison stopped at the first mismatch, so the runs behind it were not "
                               "re-checked")
    elif not byte_equality:
        not_checked.append("rank file bytes were not compared to the source (the caller asked to skip)")

    # 6. padding is padding: zeros that were never read from anywhere
    if not mismatches:
        for rank in range(world):
            rank_path = output / "ranks" / f"rank{rank}.bin"
            if not rank_path.is_file():
                continue
            with open(rank_path, "rb") as handle:
                for segment in segments:
                    if segment.rank != rank or segment.kind != "padding":
                        continue
                    handle.seek(segment.dest_offset)
                    remaining = segment.source_length
                    while remaining:
                        blob = handle.read(min(remaining, CHUNK_BYTES))
                        if any(blob):
                            mismatches.append(f"rank {rank} padding at dest offset {segment.dest_offset} is not zero")
                            break
                        remaining -= len(blob)
                    padding_checked += 1
        if padding_checked:
            checked.append(f"padding runs are zero-filled and were not read from the source: {padding_checked} runs")

    # 7. each destination file is the size the plan said it would be
    if not mismatches:
        for rank in range(world):
            rank_path = output / "ranks" / f"rank{rank}.bin"
            planned = int(manifest.get("rank_bytes", {}).get(str(rank), 0))
            if not rank_path.is_file():
                mismatches.append(f"rank {rank} file missing: {rank_path}")
                continue
            if rank_path.stat().st_size != planned:
                mismatches.append(f"rank {rank} file is {rank_path.stat().st_size} bytes and the plan says {planned}")
        checked.append(f"rank file sizes match the plan: {world} ranks")
        checked.append("layout order: runs sit back to back in plan order on each rank, and each was read at the "
                       "destination offset the manifest records")

    not_checked.extend([
        "bytes outside the planned ranges were not compared",
        "no engine load or inference was run: matching bytes is not a running model",
        "the full pack download to NAS staging is a separate step and was not run here",
    ])
    if not manifest.get("materialized"):
        not_checked.append("this pack is a view: it declares ranges and holds no copied bytes")

    return VerificationReport(
        ok=not mismatches, status="ok" if not mismatches else "failed", world=world,
        runs_compared=runs_compared, bytes_compared=bytes_compared, padding_runs_checked=padding_checked,
        files_size_checked=len(sizes), files_hash_checked=files_hash_checked,
        checked=checked, not_checked=not_checked, mismatches=mismatches,
    )


# ----------------------------------------------------------------------------- CLI


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="prepare_tp_pack.py",
        description="Plan or materialize the per-rank artifacts of a Strata TP3 run, and verify a published pack "
                    "against the source it was cut from.  Plan mode writes nothing.")
    parser.add_argument("--inventory", required=True, help="inventory.json, or the directory holding it")
    parser.add_argument("--plan", required=True, help="tp-plan.json, or the directory holding it")
    parser.add_argument("--output", required=True, help="where the rank pack is published")
    parser.add_argument("--source-root", default=None,
                        help="where the inventory's file paths live (default: next to the inventory)")
    parser.add_argument("--materialize", action="store_true", help="copy the bytes; needs its own free-disk check")
    parser.add_argument("--min-free-gib", type=float, default=DEFAULT_MIN_FREE_BYTES / 1024**3,
                        help="free disk to keep after a materialized write (default 50 GiB)")
    parser.add_argument("--verify", action="store_true", help="verify a published pack instead of building one")
    parser.add_argument("--skip-source-hash", action="store_true", help="do not recompute the source sha256")
    parser.add_argument("--overwrite", action="store_true", help="replace a pack that is already published")
    parser.add_argument("--quiet", action="store_true", help="no per-segment progress on stderr")
    parser.add_argument("--json", action="store_true", help="print the manifest or the report as JSON")
    args = parser.parse_args(argv)

    def progress(index: int, segment: dict) -> None:
        if args.quiet or args.json:
            return
        if index % 500 == 0:
            print(f"  segment {index}: {segment['tensor']} rank {segment['rank']} "
                  f"{segment['source_length']} bytes", file=sys.stderr, flush=True)

    try:
        if args.verify:
            report = verify_tp_pack(args.inventory, args.output, source_root=args.source_root,
                                    hash_source_files=not args.skip_source_hash)
            print(json.dumps(report.as_dict(), indent=2, sort_keys=True))
            return 0 if report.ok else 1
        manifest = prepare_tp_pack(args.inventory, args.plan, args.output, materialize=args.materialize,
                                   source_root=args.source_root, overwrite=args.overwrite,
                                   hash_source_files=not args.skip_source_hash,
                                   min_free_bytes=int(args.min_free_gib * 1024**3),
                                   progress=progress)
    except PackError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(manifest, indent=2, sort_keys=True))
    else:
        total = sum(int(value) for value in manifest["rank_bytes"].values())
        print(f"{args.output}: {manifest['world']} ranks, {manifest['segment_count']} segments, "
              f"{manifest['run_count']} runs, {total} bytes, materialized={manifest['materialized']}")
        for rank, value in sorted(manifest["rank_bytes"].items(), key=lambda kv: int(kv[0])):
            print(f"  rank {rank}: {int(value) / 2**30:.2f} GiB")
        for line in manifest["not_verified"]:
            print(f"  not verified: {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
