#!/usr/bin/env python3
"""Prepare and verify the per-rank artifacts for a Strata TP3 run.

A rank artifact is a byte-range view over the original, pinned source files.  Nothing here
recomputes a scale, requantizes a code or trims a vocabulary: a rank takes whole blocks, with
the native scale tail that lives inside each block, and copies them as they are.  Scales kept
as their own tensor travel through the same range logic, so they are copied, not rebuilt.

Two modes, kept apart on purpose:

* ``prepare_tp_pack(source, plan, output, materialize=False)`` reads the inventory and the
  plan, checks every length and offset against the real size of the source file, and returns
  the candidate manifest.  It opens no output file at all, so it is safe against a source
  that is still on the mirror.
* ``prepare_tp_pack(..., materialize=True)`` writes rank-local files.  It asks for spare disk
  of its own first (50 GiB by default, on top of the bytes it is about to write), writes into
  a sibling staging directory, and only then moves it into place, so a run cut in half leaves
  no half-published pack behind.  It returns the same manifest with a byte-equality receipt.

``verify_tp_pack(source, output)`` reads a published pack back and compares every copied block
and scale against the source byte for byte, and its report says both what it checked and what
it left unproven.

How much work each mode does: plan mode reads only the arithmetic (file sizes and the ranges the plan
names), so it costs a few stats and stays safe against a source that is still copying in.  Recomputing
a source file's sha256 reads that whole file, so it happens on a materialized write and on a verify,
never in plan mode.  A file the inventory leaves with a null sha256 is size-checked and named in the
report, not silently counted as hashed.

On the shared schema: the contract lists ``range``, ``local_shape``, ``valid_lanes`` and
``replicated`` per slice without naming a rank.  This module reads a slice as one rank's share
when it carries ``rank``, and as a copy for every rank when it carries ``"replicated": true``.
Task 2 is free to emit one entry per rank; both shapes give the same segments.

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
from pathlib import Path
from typing import Callable, Sequence

SCHEMA_VERSION = 1
MANIFEST_VERSION = 1
MANIFEST_NAME = "tp-pack-v1.json"

# The materialized path keeps this much disk free after it writes, so a rank file never lands
# next to a full volume.
DEFAULT_MIN_FREE_BYTES = 50 * 1024**3

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


@dataclass(frozen=True)
class Segment:
    """One contiguous run of bytes a rank owns: source codes, source scales, or padding."""

    tensor: str
    kind: str                 # "codes" | "scales" | "padding"
    rank: int
    source_path: str | None   # inventory-relative; None for padding
    source_offset: int | None
    source_length: int
    dest_offset: int
    scale_of: str | None      # the codes tensor a scale run belongs to

    def as_dict(self) -> dict:
        return {
            "tensor": self.tensor,
            "kind": self.kind,
            "rank": self.rank,
            "source_path": self.source_path,
            "source_offset": self.source_offset,
            "source_length": self.source_length,
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
            source_length=int(raw["source_length"]),
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
    ``json.dumps(obj, sort_keys=True)`` with the default separators.  Same content, two digests,
    and refusing a plan over a separator would be a false refusal at integration time.
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


def _units(tensor: dict, axis: int, start: int, end: int) -> tuple[int, int, int, int]:
    """(first_unit, unit_count, bytes_per_unit, rows) for a cut.

    A cut on axis 0 takes whole rows.  A cut on the last axis takes whole quantization blocks
    when the tensor has a block size, and whole items when it does not (f16 norms, per-block
    scale rows).  Nothing in between: a rank never takes half a block.
    """
    shape = [int(x) for x in tensor.get("shape", [])]
    _require(bool(shape), f"tensor {tensor['name']} has no shape", InventoryError)
    _require(0 <= axis < len(shape), f"tensor {tensor['name']}: axis {axis} outside shape {shape}")
    if len(shape) > 1 and axis != 0 and axis != len(shape) - 1:
        raise PlanError(f"tensor {tensor['name']}: axis {axis} is neither a row cut (0) nor a last-axis cut "
                        f"({len(shape) - 1}); this pack copies rows and blocks, not inner strides")
    extent = shape[axis]
    _require(0 <= start < end <= extent,
             f"tensor {tensor['name']}: range [{start}, {end}) outside extent {extent} on axis {axis}")

    rows = int(shape[0]) if len(shape) > 1 else 1
    total = int(tensor["byte_length"])
    if len(shape) == 1:
        block_size = tensor.get("block_size")
        if block_size is None:
            _require(total % extent == 0,
                     f"tensor {tensor['name']}: byte_length {total} is not a whole number of items over {extent}")
            return start, end - start, total // extent, 1
        return start // int(block_size), (end - start) // int(block_size), _block_unit(tensor, total, 1, extent), 1

    if axis == 0:
        _require(total % rows == 0,
                 f"tensor {tensor['name']}: byte_length {total} is not a whole number of rows over {rows} rows")
        return start, end - start, total // rows, rows

    last = shape[-1]
    block_size = tensor.get("block_size")
    if block_size is None:
        _require(total % (rows * last) == 0,
                 f"tensor {tensor['name']}: byte_length {total} does not divide into {rows} rows of {last} items, "
                 f"so this cut cannot be proven item-aligned")
        return start, end - start, total // (rows * last), rows
    unit = _block_unit(tensor, total, rows, last)
    _require(start % int(block_size) == 0 and end % int(block_size) == 0,
             f"tensor {tensor['name']}: cut [{start}, {end}) crosses a quantization block of {block_size}; a rank "
             f"may take whole blocks only, and no scale is recomputed to make it fit")
    return start // int(block_size), (end - start) // int(block_size), unit, rows


def _block_unit(tensor: dict, total: int, rows: int, last: int) -> int:
    """Bytes in one quantization block, scale tail included, from the tensor's real byte length."""
    block_size = int(tensor["block_size"])
    _require(block_size > 0 and last % block_size == 0,
             f"tensor {tensor['name']}: extent {last} is not a whole number of blocks of {block_size}", PlanError)
    blocks = rows * (last // block_size)
    _require(total % blocks == 0,
             f"tensor {tensor['name']}: byte_length {total} does not divide into {blocks} blocks, so this codec's "
             f"scale tail is not block-proportional and the cut cannot be proven byte-exact", PlanError)
    return total // blocks


def _runs(tensor: dict, axis: int, base: int, start: int, end: int) -> list[tuple[int, int]]:
    """Source runs for one item cut: one run where items sit next to each other, one per row otherwise."""
    first, count, unit, rows = _units(tensor, axis, start, end)
    if rows == 1 or axis == 0:
        return [(base + first * unit, count * unit)]
    stride = int(tensor["byte_length"]) // rows
    runs = [(base + row * stride + first * unit, count * unit) for row in range(rows)]
    merged: list[tuple[int, int]] = []
    for offset, length in runs:
        if merged and offset == merged[-1][0] + merged[-1][1]:
            merged[-1] = (merged[-1][0], merged[-1][1] + length)
        else:
            merged.append((offset, length))
    return merged


def _valid_count(slice_entry: dict, requested: int, world: int, rank: int) -> int:
    """How many items of the local range really exist in the source, in the range's own units.

    The rest of the range is padding: the local share is padded to the agreed slot count, and the
    padded tail is written as zeros instead of read out of the source.  For a block codec the count
    must land on a block boundary, which ``_units`` checks.
    """
    lanes = slice_entry.get("valid_lanes")
    if lanes is None:
        return requested
    if isinstance(lanes, int):
        return int(lanes)
    _require(len(lanes) == world,
             f"slice {slice_entry['tensor']}: valid_lanes has {len(lanes)} entries for a world of {world}")
    return int(lanes[rank])


def _scale_range(source_extent: int, scale_extent: int, start: int, end: int, block_size) -> tuple[int, int]:
    """Map a codes cut onto the scale tensor that describes it: one scale per block, or one per item."""
    if scale_extent in (source_extent, end - start):
        return start, end
    if block_size and source_extent == scale_extent * int(block_size):
        return start // int(block_size), end // int(block_size)
    raise PlanError(f"a scale tensor of extent {scale_extent} cannot be mapped from a codes extent of "
                    f"{source_extent} for a cut of [{start}, {end}) with block {block_size}")


def _ranks_of(slice_entry: dict, world: int) -> list[int]:
    if "rank" in slice_entry:
        rank = int(slice_entry["rank"])
        _require(0 <= rank < world, f"slice {slice_entry['tensor']}: rank {rank} outside world {world}")
        return [rank]
    _require(bool(slice_entry.get("replicated")),
             f"slice {slice_entry['tensor']}: it names no rank and is not marked replicated, so nobody owns it")
    return list(range(world))


def segments_for_slice(slice_entry: dict, tensors: dict[str, dict], sizes: dict[str, int],
                       world: int) -> list[Segment]:
    """Every byte run one slice contributes, per rank, with padding made explicit."""
    name = slice_entry["tensor"]
    _require(name in tensors, f"the plan slices an unknown tensor {name!r}")
    tensor = tensors[name]
    _require(slice_entry.get("family") in (None, tensor.get("family")),
             f"slice {name}: family {slice_entry.get('family')!r} is not the inventory family "
             f"{tensor.get('family')!r}")
    codec, dtype = slice_entry.get("codec"), tensor.get("dtype")
    _require(codec in (None, dtype),
             f"slice {name}: plan codec {codec!r} is not the source dtype {dtype!r}; this pack copies codes, it "
             f"does not requantize them to make a cut fit")

    axis = int(slice_entry["axis"])
    start, end = (int(x) for x in slice_entry["range"])
    _require(0 <= start < end, f"slice {name}: empty or reversed range [{start}, {end})")
    declared_shape = slice_entry.get("global_shape")
    _require(declared_shape is None or [int(x) for x in declared_shape] == [int(x) for x in tensor["shape"]],
             f"slice {name}: global_shape {declared_shape} is not the inventory shape {tensor['shape']}")

    shape = [int(x) for x in tensor["shape"]]
    local_shape = slice_entry.get("local_shape")
    requested = end - start
    if local_shape is not None:
        local_shape = [int(x) for x in local_shape]
        _require(len(local_shape) == len(shape),
                 f"slice {name}: local_shape {local_shape} does not describe a tensor of rank {len(shape)}")
        on_axis = local_shape[0] if (len(shape) == 1 or axis == 0) else local_shape[-1]
        _require(on_axis == requested,
                 f"slice {name}: local_shape {local_shape} says {on_axis} items and the range is {requested}")

    rel = tensor["file"]
    _require(rel in sizes, f"slice {name}: tensor file {rel!r} is not among the inventory source files")
    file_size = sizes[rel]
    base, total = _tensor_span(tensor)
    _check_inside(rel, file_size, base, total, f"tensor {name}")

    out: list[Segment] = []
    for rank in _ranks_of(slice_entry, world):
        valid = _valid_count(slice_entry, requested, world, rank)
        _require(0 <= valid <= requested,
                 f"slice {name}: rank {rank} valid_lanes {valid} outside a range of {requested} items")
        for offset, length in _runs(tensor, axis, base, start, start + valid):
            _check_inside(rel, file_size, offset, length, f"tensor {name}")
            out.append(Segment(tensor=name, kind="codes", rank=rank, source_path=rel, source_offset=offset,
                               source_length=length, dest_offset=0, scale_of=None))
        if valid < requested:
            # Padding is what the local range promises and the source does not have: whole units
            # of the same cut, so it lands as zeros at the end of the rank's share.
            _, full_count, unit, _ = _units(tensor, axis, start, start + requested)
            _, valid_count, _, _ = _units(tensor, axis, start, start + valid)
            _require(full_count > valid_count,
                     f"slice {name}: rank {rank} valid_lanes {valid} is not a whole number of cut units")
            out.append(Segment(tensor=name, kind="padding", rank=rank, source_path=None, source_offset=None,
                               source_length=(full_count - valid_count) * unit, dest_offset=0, scale_of=None))

    ref = slice_entry.get("scale_ref")
    if ref:
        out.extend(_scale_segments(slice_entry, tensors, sizes, world, axis, start, end))
    return out


def _scale_segments(slice_entry: dict, tensors: dict[str, dict], sizes: dict[str, int], world: int,
                    axis: int, start: int, end: int) -> list[Segment]:
    """Runs for the scale tensor a slice names, cut to the same share in the source's own terms."""
    ref = slice_entry["scale_ref"]
    name = slice_entry["tensor"]
    _require(ref in tensors, f"slice {name}: scale_ref {ref!r} is not an inventory tensor")
    codes, scale = tensors[name], tensors[ref]
    source_extent = int(codes["shape"][axis]) if axis < len(codes["shape"]) else int(codes["shape"][-1])
    scale_axis = axis if axis < len(scale["shape"]) else len(scale["shape"]) - 1
    s_start, s_end = _scale_range(source_extent, int(scale["shape"][scale_axis]), start, end, codes.get("block_size"))

    rel = scale["file"]
    _require(rel in sizes, f"scale {ref}: file {rel!r} is not among the inventory source files")
    file_size = sizes[rel]
    base, total = _tensor_span(scale)
    _check_inside(rel, file_size, base, total, f"scale {ref}")

    out: list[Segment] = []
    for rank in _ranks_of(slice_entry, world):
        for offset, length in _runs(scale, scale_axis, base, s_start, s_end):
            _check_inside(rel, file_size, offset, length, f"scale {ref} for {name}")
            out.append(Segment(tensor=ref, kind="scales", rank=rank, source_path=rel, source_offset=offset,
                               source_length=length, dest_offset=0, scale_of=name))
    return out


def _assign_dest_offsets(segments: Sequence[Segment]) -> list[Segment]:
    """Lay each rank's runs out back to back in plan order and record where each one lands."""
    cursor: dict[int, int] = {}
    placed: list[Segment] = []
    for segment in segments:
        start = cursor.get(segment.rank, 0)
        placed.append(Segment(tensor=segment.tensor, kind=segment.kind, rank=segment.rank,
                              source_path=segment.source_path, source_offset=segment.source_offset,
                              source_length=segment.source_length, dest_offset=start, scale_of=segment.scale_of))
        cursor[segment.rank] = start + segment.source_length
    return placed


def _check_vocabulary(plan_geometry: dict, slices: Sequence[dict], world: int) -> dict:
    """The logical vocabulary stays whole.

    Rows may be split across ranks (a row-sharded embedding) or replicated whole, but no row may
    be dropped to make the model load and no row may appear twice.  So per vocabulary family, the
    ranges in the plan must cover ``n_vocab`` exactly once, across the whole pack.
    """
    n_vocab = int(plan_geometry.get("n_vocab", 0) or 0)
    families = sorted({str(entry["family"]) for entry in slices if entry.get("family") in VOCABULARY_FAMILIES})
    for family in families:
        entries = [entry for entry in slices if str(entry.get("family")) == family]
        whole = [entry for entry in entries if entry.get("replicated")]
        if whole:
            for entry in whole:
                start, end = (int(x) for x in entry["range"])
                _require(n_vocab == 0 or (start == 0 and end == n_vocab),
                         f"{family}: a replicated vocabulary slice covers [{start}, {end}) of {n_vocab} rows, which "
                         f"trims the logical vocabulary")
            continue
        covered: list[tuple[int, int]] = []
        for entry in entries:
            _ranks_of(entry, world)          # a vocab slice without a rank and not replicated is unclear
            covered.append((int(entry["range"][0]), int(entry["range"][1])))
        cursor = 0
        for start, end in sorted(covered):
            _require(start == cursor,
                     f"{family}: the vocabulary rows leave a gap at {cursor} (the next range starts at {start}); this "
                     f"pack copies rows, it does not drop them to make the model load")
            cursor = end
        _require(n_vocab == 0 or cursor == n_vocab,
                 f"{family}: the vocabulary rows reach {cursor} of {n_vocab}, which trims the logical vocabulary to "
                 f"make the model load")
    return {"logical_n_vocab": n_vocab or None, "families": families, "trimmed": False,
            "note": "every vocabulary row of the plan is present exactly once across the pack"}


def _check_source_lock(inventory: dict, plan: dict) -> None:
    lock = plan.get("source_lock") or {}
    recorded = lock.get("inventory_hash")
    if recorded:
        variants = inventory_hash_variants(inventory)
        _require(recorded in variants,
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
    vocabulary = _check_vocabulary(geometry, slices, world)
    placed = _assign_dest_offsets(segments)

    rank_bytes = {rank: 0 for rank in range(world)}
    for segment in placed:
        rank_bytes[segment.rank] += segment.source_length
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


def _read_at(path: Path, offset: int, length: int) -> bytes:
    with open(path, "rb") as handle:
        handle.seek(offset)
        data = handle.read(length)
    if len(data) != length:
        raise SourceError(f"source {path} gave up short at offset {offset}: wanted {length}, got {len(data)}")
    return data


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
    try:
        handles: dict[int, object] = {}
        written = 0
        for index, segment in enumerate(segments):
            if progress is not None:
                progress(index, segment.as_dict())
            handle = handles.get(segment.rank)
            if handle is None:
                handle = open(staging / "ranks" / f"rank{segment.rank}.bin", "wb")
                handles[segment.rank] = handle
            if segment.source_path is None:
                handle.write(b"\0" * segment.source_length)   # padding: no source byte behind it
            else:
                path = resolve_source_file(inv_root, segment.source_path, src_root)
                handle.write(_read_at(path, int(segment.source_offset), segment.source_length))
            written += segment.source_length
        for handle in handles.values():
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
        if written != needed:
            raise PackError(f"wrote {written} bytes and planned {needed}")

        manifest["rank_files"] = {
            str(rank): {"path": f"ranks/rank{rank}.bin",
                        "size": (staging / "ranks" / f"rank{rank}.bin").stat().st_size}
            for rank in range(manifest["world"])
        }

        # The receipt is the comparison itself: read every copied run back out of the rank file
        # and put it next to the source bytes again.
        compared = 0
        compared_bytes = 0
        source_handles: dict[str, object] = {}
        rank_handles = {rank: open(staging / "ranks" / f"rank{rank}.bin", "rb") for rank in range(manifest["world"])}
        try:
            for segment in segments:
                if segment.source_path is None:
                    continue
                path = resolve_source_file(inv_root, segment.source_path, src_root)
                if str(path) not in source_handles:
                    source_handles[str(path)] = open(path, "rb")
                src = source_handles[str(path)]
                src.seek(segment.source_offset)
                want = src.read(segment.source_length)
                dst = rank_handles[segment.rank]
                dst.seek(segment.dest_offset)
                got = dst.read(segment.source_length)
                if want != got:
                    raise PackMismatch(f"rank {segment.rank} {segment.tensor} kind={segment.kind}: source bytes "
                                       f"[{segment.source_offset}, {segment.source_offset + segment.source_length}) "
                                       f"are not what the rank file holds at [{segment.dest_offset}, "
                                       f"{segment.dest_offset + segment.source_length})")
                compared += 1
                compared_bytes += segment.source_length
        finally:
            for handle in source_handles.values():
                handle.close()
            for handle in rank_handles.values():
                handle.close()
        manifest["receipts"]["byte_equality"] = {
            "status": "verified", "runs": compared, "bytes": compared_bytes,
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
    if byte_equality and not mismatches:
        source_handles: dict[str, object] = {}
        rank_handles: dict[int, object] = {}
        broke_early = False
        try:
            for segment in segments:
                if segment.source_path is None:
                    continue
                rel = segment.source_path
                if rel not in sizes:
                    mismatches.append(f"segment {segment.tensor} names a file the inventory does not list: {rel}")
                    broke_early = True
                    break
                if int(segment.source_offset) + segment.source_length > sizes[rel]:
                    mismatches.append(f"segment {segment.tensor} reads outside {rel}: [{segment.source_offset}, "
                                      f"{int(segment.source_offset) + segment.source_length}) of {sizes[rel]} bytes")
                    broke_early = True
                    break
                if segment.rank not in rank_handles:
                    rank_path = output / "ranks" / f"rank{segment.rank}.bin"
                    if not rank_path.is_file():
                        mismatches.append(f"rank file missing: {rank_path}")
                        broke_early = True
                        break
                    rank_handles[segment.rank] = open(rank_path, "rb")
                path = resolve_source_file(inv_root, rel, src_root)
                if str(path) not in source_handles:
                    source_handles[str(path)] = open(path, "rb")
                src = source_handles[str(path)]
                src.seek(segment.source_offset)
                want = src.read(segment.source_length)
                dst = rank_handles[segment.rank]
                dst.seek(segment.dest_offset)
                got = dst.read(segment.source_length)
                if want != got:
                    mismatches.append(f"rank {segment.rank} {segment.tensor} kind={segment.kind} scale_of="
                                      f"{segment.scale_of}: source [{segment.source_offset}, "
                                      f"{int(segment.source_offset) + segment.source_length}) of {rel} is not the "
                                      f"rank file at [{segment.dest_offset}, "
                                      f"{int(segment.dest_offset) + segment.source_length})")
                    broke_early = True
                    break
                runs_compared += 1
                bytes_compared += segment.source_length
        finally:
            for handle in source_handles.values():
                handle.close()
            for handle in rank_handles.values():
                handle.close()
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
            for segment in segments:
                if segment.rank != rank or segment.kind != "padding":
                    continue
                with open(rank_path, "rb") as handle:
                    handle.seek(segment.dest_offset)
                    blob = handle.read(segment.source_length)
                if any(blob):
                    mismatches.append(f"rank {rank} padding at dest offset {segment.dest_offset} is not zero")
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
    parser.add_argument("--json", action="store_true", help="print the manifest or the report as JSON")
    args = parser.parse_args(argv)

    try:
        if args.verify:
            report = verify_tp_pack(args.inventory, args.output, source_root=args.source_root,
                                    hash_source_files=not args.skip_source_hash)
            print(json.dumps(report.as_dict(), indent=2, sort_keys=True))
            return 0 if report.ok else 1
        manifest = prepare_tp_pack(args.inventory, args.plan, args.output, materialize=args.materialize,
                                   source_root=args.source_root, overwrite=args.overwrite,
                                   hash_source_files=not args.skip_source_hash,
                                   min_free_bytes=int(args.min_free_gib * 1024**3))
    except PackError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(manifest, indent=2, sort_keys=True))
    else:
        total = sum(int(value) for value in manifest["rank_bytes"].values())
        print(f"{args.output}: {manifest['world']} ranks, {len(manifest['segments'])} runs, {total} bytes planned, "
              f"materialized={manifest['materialized']}")
        for line in manifest["not_verified"]:
            print(f"  not verified: {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
