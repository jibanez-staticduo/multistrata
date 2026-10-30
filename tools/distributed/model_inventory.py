#!/usr/bin/env python3
"""Model inventory for the TP3 experiment: what a pack really holds, from its headers only.

`inspect_model(source)` reads GGUF headers (a few MB per shard, by HTTP range request for a remote file) and
returns the shared `inventory.json` of CONTRACT.md (schema_version 1): the source revision and hashes, every
tensor's name, dtype, shape, weight block, scale kind, byte range and family, the geometry as the file declares
it, the activation contract each consumer needs (quantizer, block size, global or rank-local), the tokenizer /
draft / PLE assets, a byte estimate, and a verdict.

It never materializes a tensor and never downloads a whole weight file.  A verdict of `loadable_by_strata: true`
means every name the TP3 path reads was found with a complete byte range in a file of the declared size: the
verdict is a claim about bytes that were looked at, not about names that were expected.

Weight blocks and activation blocks are recorded apart because TP3 cuts the two differently: an IQ3 gate/up row
travels intact when its output rows are partitioned, while the activation that multiplies it is quantized over
the whole 2560-wide vector.  The block sizes here come from `tools/gguf_reader.py`; the activation formats from
the ggml CPU type table (llama.cpp `ggml/src/ggml-cpu/ggml-cpu.c`, `type_traits_cpu[].vec_dot_type`), and the
trellis types (143-146) from `src/kernels/cpu/native_expert.cpp`, which knows them outside ggml.

    python tools/distributed/model_inventory.py hf:ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF --variant IQ3_S \
           --revision ed59f92... --evidence /home/staticduo/tmp/strata-tp3-20261030/evidence/task1
    python tools/distributed/model_inventory.py /models/IQ3_S

Sources: `ms:owner/repo`, `hf:owner/repo` (HF_ENDPOINT honoured), each optionally `@<revision>`, a URL, a .gguf
file or a folder of shards.  Exit code 0 when the verdict is loadable, 2 when it is not.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import re
import socket
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import quantscope as qs                       # header reads by byte range, hub listings, local and URL sources
from gguf_reader import BLOCK_GEOMETRY, GGML_TYPES  # the block geometry the kernels share

TYPE_IDS = {name: tid for tid, name in GGML_TYPES.items()}
SCHEMA_VERSION = 1
ARCH = "qwen4exp"
DEFAULT_MAX_HEADER_BYTES = 192 << 20           # a header is MBs; anything past this is a broken or hostile file
SAMPLED_ARRAY = 8                            # strings kept from a token list, the rest is only counted

# The shape the kernels are compiled for, from include/strata/core/geometry.hpp.  A file that declares another
# number is refused with the metadata key that differs, exactly as strata::read_geometry does.
COMPILED_GEOMETRY: dict[str, Any] = {
    "n_embd": 2560, "n_layers": 48, "qsa_interval": 4, "n_vocab": 248320, "rms_eps": 1e-6,
    "ssm_state_size": 128, "ssm_k_heads": 16, "ssm_v_heads": 48, "ssm_d_conv": 4,
    "ssm_conv_channels": 10240, "ssm_value_dim": 6144,
    "n_head": 24, "n_head_kv": 2, "head_dim": 256, "n_rot": 64,
    "idx_q_heads": 4, "idx_key_dim": 128, "idx_block": 4, "idx_top_k": 2048,
    "hc": 4, "hc_lr": 320,
    "n_expert": 512, "n_expert_used": 10, "n_ff": 640, "n_ff_shexp": 640,
    "family": "qwen4exp", "n_mtp_layers": 0,
}

# (geometry field, key under the architecture prefix, required, compare against the compiled value)
GEOMETRY_KEYS = [
    ("n_layers", "block_count", True, True),
    ("n_embd", "embedding_length", True, True),
    ("n_expert", "expert_count", True, True),
    ("n_expert_used", "expert_used_count", True, True),
    ("n_head", "attention.head_count", True, True),
    ("n_head_kv", "attention.head_count_kv", True, True),
    ("qsa_interval", "full_attention_interval", False, True),
    ("ssm_state_size", "ssm.state_size", False, True),
    ("ssm_k_heads", "ssm.group_count", False, True),
    ("ssm_v_heads", "ssm.time_step_rank", False, True),
    ("ssm_d_conv", "ssm.conv_kernel", False, True),
    ("ssm_value_dim", "ssm.inner_size", False, True),
    ("head_dim", "attention.key_length", False, True),
    ("n_rot", "rope.dimension_count", False, True),
    ("idx_q_heads", "attention.indexer.head_count", False, True),
    ("idx_key_dim", "attention.indexer.key_length", False, True),
    ("idx_top_k", "attention.indexer.top_k", False, True),
    ("hc", "hyper_connection.count", False, True),
    ("hc_lr", "hyper_connection.low_rank", False, True),
    ("n_ff", "expert_feed_forward_length", False, True),
]

# The TP3 layout of CONTRACT.md: FFN rows and columns are cut at 256 and 448, so a block that has to stay inside
# one rank's slice must divide their gcd.
FFN_CUTS = (256, 448)
# The QSA query heads split 8/8/8, so an attention output projection is cut by its input columns at these head
# indexes (8 * head_dim and 16 * head_dim), not at the FFN offsets.
ATTN_OUT_RE = r"(attn_output|o_proj)"
QSA_QUERY_HEAD_CUTS = (8, 16)

# ggml's CPU activation format per weight type (vec_dot_type).  Q8_0 and Q8_1 quantize in blocks of 32, Q8_K in
# blocks of 256; float types need no quantizer.  The trellis types are not in an upstream ggml, and the engine
# already knows their activation (src/kernels/cpu/native_expert.cpp).
ACTIVATION_TYPE = {
    "F32": "F32", "F16": "F16", "BF16": "BF16", "F64": "F64",
    "Q4_0": "Q8_0", "Q5_0": "Q8_0", "Q8_0": "Q8_0", "MXFP4": "Q8_0", "NVFP4": "Q8_0", "Q2_0": "Q8_0",
    "Q4_1": "Q8_1", "Q5_1": "Q8_1", "Q8_1": "Q8_1",
    "Q2_K": "Q8_K", "Q3_K": "Q8_K", "Q4_K": "Q8_K", "Q5_K": "Q8_K", "Q6_K": "Q8_K", "Q8_K": "Q8_K",
    "IQ1_S": "Q8_K", "IQ1_M": "Q8_K", "IQ2_XXS": "Q8_K", "IQ2_XS": "Q8_K", "IQ2_S": "Q8_K",
    "IQ3_XXS": "Q8_K", "IQ3_S": "Q8_K", "IQ4_XS": "Q8_K", "IQ4_NL": "Q8_0",
    "TQ1_0": "Q8_K", "TQ2_0": "Q8_K",
    "PTQ1_0": "Q8_0", "TQ2_T": "Q8_0", "TQK6": "Q8_0", "TQK7": "Q8_0",
    "TQP5": "Q8_0", "TQP6": "Q8_0", "TQP7": "Q8_0",
}

# How a type keeps its scales, in words a planner can act on: a block that has to be copied with its scale needs
# a different cut than a codebook stream.
SCALE_KIND = {
    "F32": None, "F16": None, "BF16": None, "F64": None, "I8": None, "I16": None, "I32": None, "I64": None,
    "Q4_0": "fp16-scale-per-block", "Q4_1": "fp16-scale-and-min-per-block",
    "Q5_0": "fp16-scale-per-block", "Q5_1": "fp16-scale-and-min-per-block",
    "Q8_0": "fp16-scale-per-block", "Q8_1": "fp16-scale-and-sum-per-block",
    "Q2_K": "fp16-superblock-scale-plus-sub-scales", "Q3_K": "fp16-superblock-scale-plus-sub-scales",
    "Q4_K": "fp16-superblock-scale-plus-sub-scales", "Q5_K": "fp16-superblock-scale-plus-sub-scales",
    "Q6_K": "fp16-superblock-scale-plus-sub-scales", "Q8_K": "fp32-superblock-scale-plus-sub-scales",
    "IQ1_S": "fp16-scale-plus-codebook-codes", "IQ1_M": "fp16-scale-plus-codebook-codes",
    "IQ2_XXS": "fp16-scale-plus-codebook-codes", "IQ2_XS": "fp16-scale-plus-codebook-codes",
    "IQ2_S": "fp16-scale-plus-codebook-codes", "IQ3_XXS": "fp16-scale-plus-codebook-codes",
    "IQ3_S": "fp16-scale-plus-signs-and-codebook-codes", "IQ4_XS": "fp16-scale-plus-codebook-codes",
    "IQ4_NL": "fp16-scale-plus-codebook-codes",
    "Q2_0": "fp16-scale-per-block", "MXFP4": "e8m0-scale-per-block", "NVFP4": "e4m3-scale-per-block",
    "PTQ1_0": "fp16-scale-plus-trellis-stream", "TQ2_T": "fp16-scale-plus-trellis-stream",
    "TQK6": "fp16-scale-plus-trellis-stream", "TQK7": "fp16-scale-plus-trellis-stream",
    "TQP5": "fp16-scale-plus-trellis-stream", "TQP6": "fp16-scale-plus-trellis-stream",
    "TQP7": "fp16-scale-plus-trellis-stream",
    "TQ1_0": "fp16-scale-plus-trellis-stream", "TQ2_0": "fp16-scale-plus-trellis-stream",
}

# Which family a tensor name belongs to, first match wins.  The contract's families name the ownership rule TP3
# needs, so a packed GDN input projection is labelled by the heads it carries first (see the notes it writes).
FAMILY_RULES = [
    (r"(?<![a-z0-9])hc[_-]|hyper_connection", "hc"),
    (r"^(output|lm_head)\.|^output\.", "head"),
    (r"(ngram|per_layer|ple_|\.ple\.)", "ple"),
    (r"(token_embd|embed_tokens|token_embedding|wte)", "embed"),
    (r"(ffn_gate_inp|\.gate\.weight$|\.router\b|shared_score)", "router"),
    (r"indexer.*_proj|^indexer\.", "qsa_idx"),
    (r"norm", "norm"),
    (r"(_exps\b|\.experts\.|experts\.\d+|block_sparse_moe\.experts)", "experts"),   # split below on gate/up
    (r"(_shexp|shared_expert|moe\.shared\.)", "shexp"),
    (r"(ssm_conv1d|attn_qkv|conv1d)", "gdn_key"),
    (r"(ssm_out|ssm_alpha|ssm_beta|ssm_dt|ssm_a$|attn_gate)", "gdn_value"),
    (r"(attn_q$|\.q_proj|attn_q\.)", "qsa_q"),
    (r"(attn_k\b|attn_v\b|attn_kv\b|\.k_proj\b|\.v_proj\b)", "qsa_kv"),
    (r"(attn_output|o_proj|ffn_down|\.down\.)", "down"),
    (r"(ffn_gate|ffn_up|\.gate_proj|\.up_proj)", "gate_up"),
    (r"(ffn_|\.mlp\.)", "other"),
]

LAYER_ROLES_ALWAYS = ["ffn_gate_inp.weight", "ffn_gate_shexp.weight", "ffn_up_shexp.weight",
                      "ffn_down_shexp.weight", "ffn_gate_exps.weight", "ffn_up_exps.weight",
                      "ffn_down_exps.weight"]
LAYER_ROLES_GDN = ["attn_qkv.weight", "attn_gate.weight"]
LAYER_ROLES_QSA = ["attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight",
                   "indexer.q_proj.weight", "indexer.k_proj.weight"]
GLOBAL_ROLES = ["token_embd.weight", "output.weight"]


class InventoryError(RuntimeError):
    """The bytes are not a readable GGUF header: there is no inventory to report."""


def tensor_family(name: str) -> str:
    n = name.lower().replace("/", ".")
    for pattern, family in FAMILY_RULES:
        if re.search(pattern, n):
            if family == "experts":
                return "expert_gate_up" if re.search(r"(gate|up|w1|w3)", n) else "expert_down"
            if family == "shexp":
                return "gate_up" if re.search(r"(gate|up|w1|w3)", n) else "down"
            return family
    return "other"


def role(name: str) -> str:
    """A tensor name without its layer number: blk.7.ffn_down_exps.weight -> blk.N.ffn_down_exps.weight."""
    return re.sub(r"^blk\.\d+\.", "blk.N.", name)


def expected_tensor_names(geometry: dict) -> list[str]:
    """Every tensor name the TP3 path reads: the FFN projections of each layer, the projections of its kind, and
    the embedding and head.  Norms, HC and the router are replicated and covered by the family check below."""
    out: list[str] = []
    interval = max(1, int(geometry["qsa_interval"]))
    for layer in range(int(geometry["n_layers"])):
        kinds = LAYER_ROLES_QSA if layer % interval == interval - 1 else LAYER_ROLES_GDN
        out += [f"blk.{layer}.{r}" for r in LAYER_ROLES_ALWAYS + kinds]
    return out + GLOBAL_ROLES


# ------------------------------------------------------------------ header reader ------------

class RangeReader:
    """Sequential reader that pulls one bounded window at a time, so a 25 GB shard costs its header."""

    def __init__(self, src, chunk: int = 4 << 20, max_bytes: int = DEFAULT_MAX_HEADER_BYTES):
        self.src, self.chunk, self.max_bytes = src, chunk, max_bytes
        self.pos = 0
        self.base = 0
        self.data = b""
        self.bytes_read = 0

    def _window(self, n: int) -> None:
        want = max(self.chunk, n)
        self.base = self.pos
        self.data = self.src.read(self.pos, want)
        self.bytes_read += len(self.data)
        if self.bytes_read > self.max_bytes:
            raise InventoryError(f"{self.src.name}: header needs more than {self.max_bytes} bytes")

    def take(self, n: int) -> bytes:
        if n < 0:
            raise InventoryError(f"{self.src.name}: negative read")
        if self.pos + n > self.src.size:
            raise InventoryError(f"{self.src.name}: header ends early at {self.pos} "
                                 f"({self.src.size} bytes in the file)")
        if self.pos + n > self.base + len(self.data):
            self._window(n)
        out = self.data[self.pos - self.base: self.pos - self.base + n]
        self.pos += n
        return out

    def skip(self, n: int) -> None:
        if self.pos + n > self.src.size:
            raise InventoryError(f"{self.src.name}: header ends early at {self.pos}")
        self.pos += n

    def u32(self) -> int:
        return int.from_bytes(self.take(4), "little")

    def u64(self) -> int:
        return int.from_bytes(self.take(8), "little")

    def string(self) -> str:
        return self.take(self.u64()).decode("utf-8", "replace")


SCALAR_FMT = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?",
              10: "<Q", 11: "<q", 12: "<d"}
SCALAR_SIZE = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def _value(r: RangeReader, vtype: int):
    """One metadata value.  Long string arrays are counted, not kept: a 248320-token list is a number here."""
    if vtype == 8:
        return r.string()
    if vtype == 9:
        item_type, count = r.u32(), r.u64()
        if item_type == 9:
            raise InventoryError(f"{r.src.name}: nested GGUF arrays are not supported")
        if item_type == 8:
            sample = [r.string() for _ in range(min(count, SAMPLED_ARRAY))]
            for _ in range(max(0, count - SAMPLED_ARRAY)):
                r.skip(r.u64())
            return {"__count__": count, "sample": sample} if count > SAMPLED_ARRAY else sample
        size = SCALAR_SIZE.get(item_type)
        if size is None:
            raise InventoryError(f"{r.src.name}: array of unknown item type {item_type}")
        if count * size > 4096:
            r.skip(count * size)
            return {"__count__": count, "sample": None}
        raw = r.take(count * size)
        return list(struct.unpack(f"<{count}" + SCALAR_FMT[item_type].strip("<"), raw))
    fmt = SCALAR_FMT.get(vtype)
    if fmt is None:
        raise InventoryError(f"{r.src.name}: unknown metadata value type {vtype}")
    return struct.unpack(fmt, r.take(SCALAR_SIZE[vtype]))[0]


def _count(r: RangeReader, n: int, min_bytes: int, what: str) -> int:
    if n * min_bytes > r.src.size - r.pos:
        raise InventoryError(f"{r.src.name}: header ends early: {n} {what} cannot fit in the "
                             f"{r.src.size - r.pos} bytes left in the file")
    return n


def read_header(src, max_header_bytes: int = DEFAULT_MAX_HEADER_BYTES) -> dict:
    """The metadata and the tensor list of one shard, from ranged reads of its header only."""
    r = RangeReader(src, max_bytes=max_header_bytes)
    if r.take(4) != b"GGUF":
        raise InventoryError(f"{src.name}: not a GGUF file")
    version = r.u32()
    if version not in (2, 3):
        raise InventoryError(f"{src.name}: GGUF v{version}, this reader handles v2 and v3")
    n_tensors = _count(r, r.u64(), 24, "tensors")
    n_kv = _count(r, r.u64(), 12, "metadata entries")
    meta: dict[str, Any] = {}
    for _ in range(n_kv):
        key = r.string()
        meta[key] = _value(r, r.u32())
    tensors = []
    for _ in range(n_tensors):
        name = r.string()
        dims = [r.u64() for _ in range(_count(r, r.u32(), 8, "dimensions"))]
        type_id = r.u32()
        offset = r.u32() if version == 2 else r.u64()
        tensors.append({"name": name, "dims": dims, "type_id": type_id, "offset": offset})
    align = meta.get("general.alignment")
    if isinstance(align, dict):
        align = None
    if align is None:
        align = 32
    if not isinstance(align, int) or align <= 0 or align % 8:
        raise InventoryError(f"{src.name}: general.alignment must be a positive multiple of 8")
    return {"file": src.name, "path": getattr(src, "path", None), "version": version, "meta": meta,
            "tensors": tensors, "alignment": align, "data_start": (r.pos + align - 1) // align * align,
            "size": src.size, "header_bytes_read": r.bytes_read}


# ------------------------------------------------------------------ source listing -----------

def _hub_listing(kind: str, repo: str) -> tuple[list, dict[str, str], dict[str, str]]:
    """(files, sha256 by path, revision by path) from a hub's own file listing.

    ModelScope reports a sha256 and the commit that last touched each file, so an ms listing can pin both.  HF
    reports the lfs oid per file and the branch head from its commits endpoint, which is the same commit for
    every file in the repo.
    """
    list_url, file_url = qs.HUBS[kind]
    data = json.loads(qs.http(list_url.format(repo=repo)))
    files: list = []
    sha: dict[str, str] = {}
    rev: dict[str, str] = {}
    if kind == "ms":
        for f in data["Data"]["Files"]:
            if f.get("Type") != "blob":
                continue
            path = f["Path"]
            files.append(qs.Source(path, int(f.get("Size") or 0), url=file_url.format(repo=repo, path=path)))
            if f.get("Sha256"):
                sha[path] = str(f["Sha256"])
            if f.get("Revision"):
                rev[path] = str(f["Revision"])
    else:
        for f in data:
            if f.get("type") != "file":
                continue
            path = f["path"]
            size = int((f.get("lfs") or {}).get("size") or f.get("size") or 0)
            files.append(qs.Source(path, size, url=file_url.format(repo=repo, path=path)))
            oid = (f.get("lfs") or {}).get("oid") or ""
            if oid:
                # HF reports the plain hex here and "sha256:<hex>" there; both are the file's sha256
                sha[path] = oid.split(":", 1)[1] if ":" in oid else oid
        head = None
        try:
            head = str(json.loads(qs.http(list_url.format(repo=repo).replace("/tree/main?recursive=true",
                                                                             "/commits/main")))[0]["id"])
        except Exception:                                       # noqa: BLE001 - a pinned revision still works
            head = None
        if head:
            rev = {f.name: head for f in files}      # HF commits are repo-wide, so one head covers every file
    return files, sha, rev


def _common_revision(rev_by_path: dict[str, str], files: list) -> str | None:
    """The one commit every selected file reports, or None when they differ: the revision of the bytes in hand."""
    vals = {rev_by_path[f.name] for f in files if f.name in rev_by_path}
    return vals.pop() if len(vals) == 1 else None


def _pin_revision(files: list, kind: str, repo: str, revision: str) -> list:
    # ".../resolve/main/" -> "..."; the revision goes back where the branch name was
    head = qs.HUBS[kind][1].format(repo=repo, path="").split("/resolve/")[0]
    out = []
    for f in files:
        url = f.url
        if url and "/resolve/" in url:
            url = head + "/resolve/" + revision + "/" + f.name.lstrip("/")
        out.append(qs.Source(f.name, f.size, url=url, path=f.path))
    return out


def resolve_source(source: str, variant: str | None = None, revision: str | None = None,
                   timeout: int = 60) -> tuple[dict, list, dict]:
    """(source block, files, extra) for a hub repo, a URL, a file or a folder.  `revision` pins the hub commit.

    `extra` carries {"sha256": by path, "revision_by_path": ..., "pinned": bool} for the caller's file list.
    """
    m = re.match(r"^(ms|hf):([^/]+/[^/@]+)(?:@(.+))?$", source)
    if m:
        kind, repo, pinned = m.group(1), m.group(2), m.group(3)
        rev = revision or pinned
        files, sha, rev_by_path = _hub_listing(kind, repo)
        if rev:
            files = _pin_revision(files, kind, repo, rev)
        return ({"kind": kind, "repo": repo, "revision": rev},
                [f for f in files if f.name.endswith(".gguf") and "mmproj" not in f.name.lower()],
                {"sha256": sha, "revision_by_path": rev_by_path, "pinned": bool(rev)})
    gguf = [f for f in qs.resolve(source) if f.name.endswith(".gguf") and "mmproj" not in f.name.lower()]
    return ({"kind": "local", "repo": str(Path(source).resolve()), "revision": revision}, gguf,
            {"sha256": {}, "revision_by_path": {}, "pinned": False})


def pick_variant(files: list, variant: str | None) -> list:
    import fnmatch
    groups: dict[str, list] = {}
    for f in files:
        groups.setdefault(qs.variant_of(f.name), []).append(f)
    if not groups:
        raise InventoryError("no model .gguf files there")
    if variant:
        chosen = {k: v for k, v in groups.items()
                  if variant == k or variant in k or fnmatch.fnmatch(k, variant)}
        if not chosen:
            raise InventoryError(f"no variant matches {variant!r}; there are: "
                                 + ", ".join(sorted(groups)))
    elif len(groups) == 1:
        chosen = groups
    else:
        raise InventoryError("pass --variant: " + ", ".join(sorted(groups)))
    out: list = []
    for fs in chosen.values():
        out += sorted(fs, key=lambda f: f.name)
    return out


# ------------------------------------------------------------------ geometry -----------------

def _array_count(v) -> int | None:
    if isinstance(v, dict):
        return v.get("__count__")
    if isinstance(v, list):
        return len(v)
    return None


def geometry_from_metadata(meta: dict) -> tuple[dict, list[str], list[str]]:
    """(geometry, problems, notes).  The file's own numbers win; the compiled shape decides what is a problem."""
    g = dict(COMPILED_GEOMETRY)
    problems: list[str] = []
    notes: list[str] = []
    arch = meta.get("general.architecture")
    if not isinstance(arch, str):
        return g, ["geometry: no general.architecture; this is a metadata shard of no known model"], notes
    g["family"] = arch
    if arch != ARCH:
        problems.append(f"geometry: architecture is '{arch}', this engine requires '{ARCH}' "
                        "(include/strata/artifact/gguf_reader.hpp)")
        return g, problems, notes
    p = ARCH + "."

    def as_int(key: str):
        v = meta.get(p + key)
        if isinstance(v, dict) or isinstance(v, list) or isinstance(v, str) or isinstance(v, bool):
            return None
        if isinstance(v, float):
            return None
        return int(v) if isinstance(v, int) else None

    for field_name, key, required, _compare in GEOMETRY_KEYS:
        v = as_int(key)
        if v is None:
            if required:
                problems.append(f"geometry: missing {p}{key}")
            continue
        if v <= 0:
            problems.append(f"geometry: {p}{key} = {v}, it must be positive")
            continue
        g[field_name] = v
    value_length = as_int("attention.value_length")
    if value_length is not None and value_length != g["head_dim"]:
        problems.append(f"geometry: {p}attention.value_length = {value_length} and "
                        f"{p}attention.key_length = {g['head_dim']}: the attention kernels use one head width "
                        "for both")
    ff_shexp = as_int("expert_shared_feed_forward_length")
    if ff_shexp is None:
        ff_shexp = g["n_ff"]
    elif ff_shexp != g["n_ff"]:
        problems.append(f"geometry: {p}expert_shared_feed_forward_length = {ff_shexp} and "
                        f"{p}expert_feed_forward_length = {g['n_ff']}: the shared expert is read with the routed "
                        "experts' width")
    g["n_ff_shexp"] = ff_shexp
    eps = meta.get(p + "attention.layer_norm_rms_epsilon")
    if isinstance(eps, (int, float)) and not isinstance(eps, bool) and eps > 0:
        g["rms_eps"] = float(eps)
    n_vocab = _array_count(meta.get("tokenizer.ggml.tokens"))
    if n_vocab:
        g["n_vocab"] = n_vocab
    if g["n_head"] % g["n_head_kv"]:
        problems.append(f"geometry: {p}attention.head_count = {g['n_head']} is not a multiple of "
                        f"head_count_kv = {g['n_head_kv']}")
    if g["ssm_value_dim"] != g["ssm_state_size"] * g["ssm_v_heads"]:
        problems.append(f"geometry: {p}ssm.inner_size = {g['ssm_value_dim']}, but ssm.state_size x "
                        f"ssm.time_step_rank = {g['ssm_state_size'] * g['ssm_v_heads']}: the GDN kernels take "
                        "square heads")
    if g["n_expert_used"] > g["n_expert"] or g["n_expert_used"] > 64:
        problems.append(f"geometry: {p}expert_used_count = {g['n_expert_used']} with expert_count = "
                        f"{g['n_expert']}: the router takes 1..64")
    # the layer pattern is declared twice (the interval and the ratio list); read_geometry checks they agree
    ratios = meta.get(p + "attention.compress_ratios")
    if isinstance(ratios, list):
        if len(ratios) != g["n_layers"]:
            problems.append(f"geometry: {p}attention.compress_ratios lists {len(ratios)} entries for "
                            f"{g['n_layers']} layers")
        elif not any(ratios):
            # a list of only zeros is a converter's placeholder, not a split (#2396): the compiled block stands
            notes.append(f"geometry note: {p}attention.compress_ratios is all zeros, so the compiled pooling "
                         f"block of {g['idx_block']} cells is used")
        else:
            block = 0
            for layer, ratio in enumerate(ratios):
                is_qsa = layer % g["qsa_interval"] == g["qsa_interval"] - 1
                if bool(ratio) != is_qsa or ratio < 0 or (block and ratio and ratio != block):
                    problems.append(f"geometry: {p}attention.compress_ratios[{layer}] = {ratio}, but "
                                    f"{p}full_attention_interval = {g['qsa_interval']} makes layer {layer} "
                                    + ("full attention" if is_qsa else "linear attention"))
                    break
                if ratio:
                    block = ratio
            if block:
                g["idx_block"] = block
    if g["n_rot"] > g["head_dim"] or g["n_rot"] > g["idx_key_dim"] or g["n_rot"] % 2:
        problems.append(f"geometry: {p}rope.dimension_count = {g['n_rot']} is not an even slice of the head "
                        f"widths ({g['head_dim']}, {g['idx_key_dim']})")
    # derived, the way geometry.hpp derives them
    g["ssm_conv_channels"] = 2 * g["ssm_state_size"] * g["ssm_k_heads"] + g["ssm_value_dim"]
    mtp = as_int("nextn_predict_layers")
    if mtp is not None:
        g["n_mtp_layers"] = mtp
        if mtp != COMPILED_GEOMETRY["n_mtp_layers"]:
            notes.append(f"geometry note: {p}nextn_predict_layers = {mtp}; qwen4exp files keep their draft block "
                         "out of the trunk and strata::read_geometry does not read this key for the family")
    for field_name, key, _required, compare in GEOMETRY_KEYS:
        if not compare:
            continue
        want = COMPILED_GEOMETRY[field_name]
        if g[field_name] != want:
            problems.append(f"geometry: {p}{key} = {g[field_name]}, the engine is compiled for {want} "
                            f"(include/strata/core/geometry.hpp::{field_name})")
    if g["n_vocab"] != COMPILED_GEOMETRY["n_vocab"]:
        problems.append(f"geometry: tokenizer.ggml.tokens = {g['n_vocab']} entries, the engine is compiled for "
                        f"{COMPILED_GEOMETRY['n_vocab']} (n_vocab)")
    return g, problems, notes


# ------------------------------------------------------------------ tensors ------------------

def _tensor_rows(header: dict) -> list[dict]:
    """Every tensor of one shard with absolute byte ranges and the block and scale identity its type implies."""
    order = sorted(header["tensors"], key=lambda t: t["offset"])
    rows = []
    for i, t in enumerate(order):
        name, type_id = t["name"], t["type_id"]
        dtype = GGML_TYPES.get(type_id, f"type#{type_id}")
        elements = 1
        for d in t["dims"]:
            elements *= d
        geom = BLOCK_GEOMETRY.get(dtype)
        block_size, expected = None, None
        if geom:
            block_size = geom[0]
            if elements % geom[0] == 0:
                expected = elements // geom[0] * geom[1]
        nxt = order[i + 1]["offset"] if i + 1 < len(order) else None
        end = nxt if nxt is not None else header["size"] - header["data_start"]
        rows.append({"name": name, "file": header["file"], "dtype": dtype, "shape": list(t["dims"]),
                     "elements": elements, "block_size": block_size, "scale_kind": SCALE_KIND.get(dtype),
                     "byte_offset": header["data_start"] + t["offset"],
                     "byte_length": max(0, end - t["offset"]), "expected_bytes": expected,
                     "family": tensor_family(name)})
    return rows


def _data_problems(rows: list[dict], headers: dict) -> tuple[list[str], int]:
    """(reasons, checked): what is wrong with the byte ranges, and how many tensors are whole and inside a file."""
    reasons: list[str] = []
    checked = 0
    per_file: dict[str, list[dict]] = {}
    for row in rows:
        per_file.setdefault(row["file"], []).append(row)
    for file_name, file_rows in per_file.items():
        header = headers[file_name]
        previous_end = None
        previous_name = None
        for row in sorted(file_rows, key=lambda r: r["byte_offset"]):
            if row["dtype"].startswith("type#"):
                reasons.append(f"unknown dtype {row['dtype']} in {file_name}:{row['name']}: no block geometry, "
                               "so no TP3 layout can be planned for it")
                continue
            if row["block_size"] is None:
                reasons.append(f"{file_name}:{row['name']} ({row['dtype']}): no known block geometry")
                continue
            if row["expected_bytes"] is None:
                reasons.append(f"{file_name}:{row['name']}: {row['elements']} values are not whole "
                               f"{row['block_size']}-value blocks of {row['dtype']}")
                continue
            if previous_end is not None and row["byte_offset"] < previous_end:
                reasons.append(f"{file_name}:{row['name']} overlaps {previous_name} "
                               f"({previous_end - row['byte_offset']} bytes in common)")
            previous_end = row["byte_offset"] + row["expected_bytes"]
            previous_name = row["name"]
            if row["byte_offset"] + row["expected_bytes"] > header["size"]:
                reasons.append(f"{file_name}:{row['name']} needs {row['expected_bytes']} bytes at "
                               f"{row['byte_offset']} but the file is {header['size']} bytes: its scales and "
                               "codes are missing")
                continue
            checked += 1
    return reasons, checked


def _split_problems(meta: dict, files: list, per_file_tensors: dict[str, int]) -> list[str]:
    reasons = []
    declared_files = meta.get("general.split_count")
    declared_tensors = meta.get("split.tensors.count")
    total = sum(per_file_tensors.values())
    if isinstance(declared_files, int) and not isinstance(declared_files, bool):
        if declared_files != len(files):
            reasons.append(f"general.split_count = {declared_files} but {len(files)} .gguf files were read")
    if isinstance(declared_tensors, int) and not isinstance(declared_tensors, bool):
        if declared_tensors != total:
            reasons.append(f"split.tensors.count = {declared_tensors} but the shards together hold {total} "
                           "tensors")
    return reasons


def _role_problems(geometry: dict, rows: list[dict]) -> list[str]:
    """Names the TP3 path needs that the model does not carry, grouped per role to keep the list readable."""
    found: dict[str, set[int]] = {}
    for row in rows:
        m = re.match(r"^blk\.(\d+)\.(.*)$", row["name"])
        if m:
            found.setdefault(m.group(2), set()).add(int(m.group(1)))
        else:
            found.setdefault(row["name"], set())
    reasons = []
    interval = max(1, int(geometry["qsa_interval"]))
    n_layers = int(geometry["n_layers"])
    layers_of = {}
    for layer in range(n_layers):
        kinds = LAYER_ROLES_QSA if layer % interval == interval - 1 else LAYER_ROLES_GDN
        for r in LAYER_ROLES_ALWAYS + kinds:
            layers_of.setdefault(r, set()).add(layer)
    for r, layers in layers_of.items():
        have = found.get(r, set()) & layers
        if len(have) == len(layers) or not layers:
            continue
        if have:
            reasons.append(f"missing: blk.N.{r} on {len(layers) - len(have)} of {len(layers)} layers")
        else:
            reasons.append(f"missing: blk.N.{r} (needed on all {len(layers)} layers, the model has none)")
    for r in GLOBAL_ROLES:
        if r not in found:
            reasons.append(f"missing: {r}")
    ple = [row for row in rows if row["family"] == "ple"]
    if not ple:
        reasons.append("missing: no PLE (per-layer embedding) tensor; the TP3 path replicates it and needs it")
    return reasons


# ------------------------------------------------------------------ activations --------------

FAMILY_CONSUMER = {
    "expert_gate_up": ("expert_gate_up", "n_embd", "full"),
    "gate_up": ("gate_up", "n_embd", "full"),
    "expert_down": ("expert_down", "n_ff", "ffn"),
    "down": ("down", "n_ff", "ffn"),
    "gdn_key": ("gdn_key", "n_embd", "full"),
    "gdn_value": ("gdn_value", "n_embd", "full"),
    "qsa_q": ("qsa_q", "n_embd", "full"),
    "qsa_kv": ("qsa_kv", "n_embd", "full"),
    "qsa_idx": ("qsa_idx", "n_embd", "full"),
    "hc": ("hc", "n_embd", "full"),
    "router": ("router", "n_embd", "full"),
    "head": ("head", "n_embd", "full"),
    "norm": ("norm", None, "elementwise"),
    "ple": ("ple", None, "gather"),
    "embed": ("embed", None, "gather"),
    "other": ("other", "n_embd", "full"),
}


def activation_contracts(rows: list[dict], geometry: dict) -> list[dict]:
    """Per consumer: the activation format and block its weights ask for, and whether TP3 may quantize that
    activation per rank.  Copying weight bytes is not enough when the scale of a block two ranks share would
    change: a block that does not fit inside one rank's slice keeps the original global quantizer.

    The cuts are the agreed TP3 ownership (CONTRACT.md): FFN columns at 256 and 448, whose gcd 64 is the widest
    block that fits one rank, and QSA query heads 8/8/8, which cuts an attention output projection by its input
    columns at 8 * head_dim and 16 * head_dim.
    """
    n_head = int(geometry.get("n_head", COMPILED_GEOMETRY["n_head"]))
    head_dim = int(geometry.get("head_dim", COMPILED_GEOMETRY["head_dim"]))
    n_embd = int(geometry.get("n_embd", COMPILED_GEOMETRY["n_embd"]))
    n_ff = int(geometry.get("n_ff", COMPILED_GEOMETRY["n_ff"]))
    out: dict[tuple, dict] = {}
    for row in rows:
        family = row["family"]
        name = row["name"].lower()
        consumer, width_key, kind = FAMILY_CONSUMER.get(family, ("other", "n_embd", "full"))
        cuts: tuple[int, ...] = ()
        if re.search(ATTN_OUT_RE, name):
            consumer, width_key, kind = "attn_output", "attn_out", "sliced"
            cuts = tuple(h * head_dim for h in QSA_QUERY_HEAD_CUTS)
        elif kind == "ffn":
            kind, cuts = "sliced", FFN_CUTS

        if kind == "gather":
            _contract(out, consumer, "none", 1, "local",
                      "rows are gathered by token id, so no activation is quantized; a rank-local slice of the "
                      "replicated table is the identity, and ids a rank does not own contribute zeros before the "
                      "sum")
            continue
        if kind == "elementwise":
            _contract(out, consumer, "none", 1, "global",
                      "norm and scale weights: their statistics are over the whole vector, so they cannot be "
                      "recomputed on a rank-local slice")
            continue
        act = ACTIVATION_TYPE.get(row["dtype"])
        if act is None:
            _contract(out, consumer, f"unknown ({row['dtype']})", 0, "global",
                      f"no ggml activation format is known for weight type {row['dtype']}; this consumer has to "
                      "run on the global reference path until one is")
            continue
        block = BLOCK_GEOMETRY[act][0] if act in BLOCK_GEOMETRY else 1
        width = n_head * head_dim if width_key == "attn_out" else (n_ff if width_key == "n_ff" else n_embd)
        where = f"{width}-wide" + (" FFN" if width_key == "n_ff" else
                                  " attention output" if width_key == "attn_out" else " residual")
        if block == 1:
            _contract(out, consumer, act, 1, "global",
                      f"weights are stored as {row['dtype']}, so there is no activation quantizer; the norm and any "
                      f"transform in front of this consumer run over the whole {where} vector, so they stay global")
            continue
        if kind == "sliced":
            fits = width % block == 0 and math.gcd(*cuts) % block == 0
            scope = "local" if fits else "global"
            cuts_s = f"the cuts at {cuts[0]} and {cuts[1]}"
            notes = (f"activation is {act} in blocks of {block} over the {where} vector: "
                     + (f"{block} divides {cuts_s}, so each rank quantizes its owned slice"
                        if fits else
                        f"{block} does not divide {cuts_s}: keep the original global quantizer and all-gather the "
                        "padded activation, then mask the lanes this rank does not own"))
            if row["block_size"] and math.gcd(*cuts) % row["block_size"]:
                notes += (f"; the weight block of {row['block_size']} crosses those cuts too, so the cut needs a "
                          "whole-block boundary or the reference path")
        else:
            scope = "global"
            notes = (f"activation is {act} in blocks of {block} over the whole {where} vector"
                     + ("" if width % block == 0 else f", and {width} is not a multiple of {block}")
                     + ": every rank needs the same quantizer, so it is all-gathered, never re-quantized per rank")
        _contract(out, consumer, act, block, scope, notes, weight_type=row["dtype"])
    return sorted(out.values(), key=lambda c: (c["consumer"], c["quantizer"], c["block_size"]))


def _contract(out: dict, consumer: str, quantizer: str, block: int, scope: str, notes: str,
              weight_type: str | None = None) -> None:
    """One entry per (consumer, quantizer, block, scope); a second weight type that needs the same contract is
    named in the note instead of repeating the entry."""
    key = (consumer, quantizer, block, scope)
    entry = out.get(key)
    if entry is None:
        out[key] = {"consumer": consumer, "quantizer": quantizer, "block_size": block, "scope": scope,
                    "notes": notes}
    elif weight_type and weight_type not in entry["notes"]:
        entry["notes"] += f" (also for weights stored as {weight_type})"


# ------------------------------------------------------------------ assets -------------------

def assets_from(meta: dict, rows: list[dict], files: list, geometry: dict) -> dict:
    def count_of(key: str):
        v = meta.get(key)
        return _array_count(v) if v is not None else None

    tokens = _array_count(meta.get("tokenizer.ggml.tokens"))
    tokenizer = {
        "present": tokens is not None or "tokenizer.ggml.model" in meta,
        "model": meta.get("tokenizer.ggml.model") if isinstance(meta.get("tokenizer.ggml.model"), str) else None,
        "n_vocab": tokens,
        "bos_token_id": meta.get("tokenizer.ggml.bos_token_id"),
        "eos_token_id": meta.get("tokenizer.ggml.eos_token_id"),
        "chat_template": "tokenizer.chat_template" in meta,
        "merges": count_of("tokenizer.ggml.merges"),
        "added_tokens": count_of("tokenizer.ggml.added_tokens"),
        "byte_fallback": meta.get("tokenizer.ggml.byte_fallback"),
    }
    draft_rows = [r for r in rows if re.search(r"(nextn|mtp|draft)", r["name"].lower())]
    # a draft pack is a file named for it; "nodraft-model.gguf" is not one, so the name must start a word
    draft_files = [f.name for f in files
                   if re.search(r"(?:^|[-_.])(draft|nextn|mtp)(?:[-_.]|\d)", f.name.lower())]
    draft = {
        "present": bool(draft_rows) or bool(draft_files) or geometry["n_mtp_layers"] > 0,
        "tensors": len(draft_rows),
        "bytes": sum(r["byte_length"] for r in draft_rows),
        "n_mtp_layers": geometry["n_mtp_layers"],
        "files": draft_files,
    }
    ple_rows = [r for r in rows if r["family"] == "ple"]
    ple = {
        "present": bool(ple_rows),
        "tensors": len(ple_rows),
        "bytes": sum(r["byte_length"] for r in ple_rows),
        "roles": sorted({role(r["name"]) for r in ple_rows}),
    }
    return {"tokenizer": tokenizer, "draft": draft, "ple": ple}


# ------------------------------------------------------------------ the inventory ------------

@dataclass
class ModelInventory:
    """The shared `inventory.json` of CONTRACT.md, schema_version 1.  `notes` is the only key the contract
    leaves open; it carries the advisories that are not verdict reasons."""

    source: dict
    geometry: dict
    tensors: list
    activation_contracts: list
    assets: dict
    bytes: dict
    verdict: dict
    notes: list = field(default_factory=list)
    raw: dict = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict:
        return {"schema_version": SCHEMA_VERSION, "source": self.source, "geometry": self.geometry,
                "tensors": self.tensors, "activation_contracts": self.activation_contracts,
                "assets": self.assets, "bytes": self.bytes, "verdict": self.verdict, "notes": self.notes}

    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()

    def write(self, directory: Path) -> list[Path]:
        """Freeze inventory.json, source-lock.json and disk-estimate.json in `directory`."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        inv = directory / "inventory.json"
        inv.write_text(json.dumps(self.to_dict(), indent=1, sort_keys=False) + "\n", encoding="utf-8")
        lock = directory / "source-lock.json"
        lock.write_text(json.dumps({
            "schema_version": 1, "source": self.source, "inventory_sha256": self.sha256(),
            "files": len(self.source["files"]), "source_bytes": self.bytes["total"],
            "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "tool": "tools/distributed/model_inventory.py",
        }, indent=1) + "\n", encoding="utf-8")
        estimate = directory / "disk-estimate.json"
        estimate.write_text(json.dumps({
            "schema_version": 1, "source_bytes": self.bytes["total"],
            "bytes_by_file": _sum_by(self.tensors, "file"),
            "bytes_by_family": self.bytes["by_family"],
            "staged_copy_bytes": self.bytes["total"],
            "note": "weights only, from the header byte ranges.  Per-rank TP3 views, KV/GDN state, page cache and "
                    "materialized files are Task 3's budget, not estimated here.",
        }, indent=1) + "\n", encoding="utf-8")
        return [inv, lock, estimate]


def _sum_by(rows: list[dict], key: str) -> dict:
    out: dict[str, int] = {}
    for row in rows:
        out[row[key]] = out.get(row[key], 0) + row["byte_length"]
    return dict(sorted(out.items()))


def inspect_model(source: str, *, variant: str | None = None, revision: str | None = None,
                  max_header_bytes: int = DEFAULT_MAX_HEADER_BYTES, timeout: int = 60) -> ModelInventory:
    """Read a pack's headers and return the contract's inventory.  No tensor is ever materialized."""
    block, files, extra = resolve_source(source, variant, revision, timeout)
    if timeout:
        socket.setdefaulttimeout(timeout)          # quantscope reads with the process default
    files = pick_variant(files, variant)
    sha = extra.get("sha256", {})
    if block["kind"] == "ms" and not extra.get("pinned"):
        # ms files each report the commit that last touched them: pin the one the chosen variant shares
        common = _common_revision(extra.get("revision_by_path", {}), files)
        if common:
            block["revision"] = common
    headers = {}
    rows: list[dict] = []
    metas: dict[str, dict] = {}
    for f in files:
        h = read_header(f, max_header_bytes)
        headers[h["file"]] = h
        metas[h["file"]] = h["meta"]
        rows += _tensor_rows(h)

    # The geometry lives in the metadata shard; the others carry none (gguf_reader.hpp says the same).
    meta: dict[str, Any] = {}
    for f in sorted(metas, key=lambda n: metas[n].get("general.architecture") != ARCH):
        for k, v in metas[f].items():
            meta.setdefault(k, v)
    geometry, problems, notes = geometry_from_metadata(meta)

    data_reasons, checked = _data_problems(rows, headers)
    reasons = problems + data_reasons
    reasons += _split_problems(meta, files,
                              {name: len(h["tensors"]) for name, h in headers.items()})
    for h in headers.values():
        if h["version"] != 3:
            reasons.append(f"{h['file']}: GGUF v{h['version']}, strata's reader handles v3 "
                           "(include/strata/artifact/gguf_reader.hpp)")
    reasons += _role_problems(geometry, rows)

    by_family: dict[str, int] = {}
    for row in rows:
        by_family[row["family"]] = by_family.get(row["family"], 0) + row["byte_length"]
    notes += [
        "tensor order is (file, byte offset); byte_offset and byte_length are absolute in the named file",
        "blk.N.attn_qkv and blk.N.ssm_conv1d are packed GDN tensors: their key and value head rows share one "
        "name, so Task 2 splits them at the head boundaries the geometry gives",
        f"activation formats are ggml's vec_dot_type table plus the trellis types from "
        f"src/kernels/cpu/native_expert.cpp; TP3 FFN cuts are {FFN_CUTS[0]} and {FFN_CUTS[1]}",
        "what a checked tensor means: its byte range lies inside a file of the declared size, and its length "
        "equals what its type's block geometry says.  No tensor payload was read: bounded header reads only, "
        "so no weight file was downloaded",
    ]

    inventory_rows = [{"name": r["name"], "file": r["file"], "dtype": r["dtype"], "shape": r["shape"],
                       "block_size": r["block_size"], "scale_kind": r["scale_kind"],
                       "byte_offset": r["byte_offset"], "byte_length": r["byte_length"],
                       "family": r["family"], "elements": r["elements"],
                       "expected_bytes": r["expected_bytes"]} for r in
                      sorted(rows, key=lambda r: (r["file"], r["byte_offset"]))]

    return ModelInventory(
        source={**block, "files": [{"path": f.name, "size": f.size, "sha256": sha.get(f.name)} for f in files]},
        geometry=geometry,
        tensors=inventory_rows,
        activation_contracts=activation_contracts(rows, geometry),
        assets=assets_from(meta, rows, files, geometry),
        bytes={"total": sum(r["byte_length"] for r in rows), "by_family": dict(sorted(by_family.items()))},
        verdict={"loadable_by_strata": not reasons, "reasons": reasons, "checked_tensors": checked},
        notes=notes,
        raw={"headers": {k: {kk: vv for kk, vv in h.items() if kk != "meta"} for k, h in headers.items()},
             "meta_keys": sorted(meta)},
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="ms:owner/repo, hf:owner/repo (each optionally @<revision>), a URL, "
                                   "a .gguf file or a folder of shards")
    ap.add_argument("--variant", help="a variant name or glob (e.g. IQ3_S, UD-Q4_K_XL)")
    ap.add_argument("--revision", help="pin the hub revision (commit sha) instead of the one the hub reports")
    ap.add_argument("--out", metavar="FILE", help="write the inventory JSON here")
    ap.add_argument("--evidence", metavar="DIR",
                    help="write inventory.json, source-lock.json and disk-estimate.json here")
    ap.add_argument("--max-header-bytes", type=int, default=DEFAULT_MAX_HEADER_BYTES,
                    help="read budget per shard header (default %(default)s)")
    ap.add_argument("--timeout", type=int, default=60, help="seconds per HTTP read (default %(default)s)")
    a = ap.parse_args(argv)

    inv = inspect_model(a.source, variant=a.variant, revision=a.revision,
                        max_header_bytes=a.max_header_bytes, timeout=a.timeout)
    d = inv.to_dict()
    if a.out:
        Path(a.out).write_text(json.dumps(d, indent=1) + "\n", encoding="utf-8")
    if a.evidence:
        for p in inv.write(Path(a.evidence)):
            print(f"wrote {p}")
    s = d["source"]
    print(f"{s['kind']}:{s['repo']} revision {s['revision']} ({len(s['files'])} files)")
    read = sum(h["header_bytes_read"] for h in inv.raw["headers"].values())
    print("header bytes read: " + ", ".join(f"{h['file']}: {h['header_bytes_read'] / 1e6:.1f} MB"
                                            for h in inv.raw["headers"].values())
          + f" (total {read / 1e6:.1f} MB of {d['bytes']['total'] / 1e9:.2f} GB)")
    print(f"geometry: {d['geometry']['n_layers']} layers, n_embd {d['geometry']['n_embd']}, "
          f"n_ff {d['geometry']['n_ff']}, {d['geometry']['n_expert']} experts, "
          f"{d['geometry']['n_head']}/{d['geometry']['n_head_kv']} QSA heads, "
          f"GDN {d['geometry']['ssm_k_heads']}/{d['geometry']['ssm_v_heads']}")
    print(f"tensors: {len(d['tensors'])} ({d['verdict']['checked_tensors']} whole in the bytes read), "
          f"{d['bytes']['total'] / 1e9:.2f} GB")
    for family, b in d["bytes"]["by_family"].items():
        print(f"  {family:16s} {b / 1e9:8.3f} GB")
    types: dict[str, int] = {}
    for t in d["tensors"]:
        types[t["dtype"]] = types.get(t["dtype"], 0) + t["byte_length"]
    print("types by bytes: " + ", ".join(f"{k} {v / 1e9:.2f}GB" for k, v in
                                         sorted(types.items(), key=lambda kv: -kv[1])))
    print(f"activation contracts: {len(d['activation_contracts'])}")
    for c in d["activation_contracts"]:
        print(f"  {c['consumer']:14s} {c['quantizer']:8s} block {c['block_size']:3d} {c['scope']}")
    v = d["verdict"]
    print(f"verdict: {'loadable by strata' if v['loadable_by_strata'] else 'NOT loadable'}")
    for reason in v["reasons"]:
        print(f"  - {reason}")
    return 0 if v["loadable_by_strata"] else 2


if __name__ == "__main__":
    sys.exit(main())

