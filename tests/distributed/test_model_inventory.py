"""Tests for tools/distributed/model_inventory.py (Strata TP3, Task 1).

Every fixture is a real GGUF v3 header written here, so the reader is checked against bytes, not against a mock.
The shapes and quantization types are the ones the IQ3_S pack has (data/gguf_fingerprints.json): gate/up as i-quants
and K-quants, down as IQ4_NL or Q8_0, norms F32, router F32/BF16.

Run: <venv>/bin/python -m pytest tests/distributed/test_model_inventory.py -q
The last test reads a real repository over the network and is skipped unless STRATA_TP3_NETWORK=1.

Build wiring a later integrator would add (this file needs no CMake target of its own):
    add_test(NAME strata_tp3_model_inventory
             COMMAND ${STRATA_TP3_PYTHON} -m pytest ${CMAKE_SOURCE_DIR}/tests/distributed/test_model_inventory.py -q)
"""
from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "distributed"))

import model_inventory as mi  # noqa: E402
from gguf_reader import BLOCK_GEOMETRY  # noqa: E402


# ------------------------------------------------------------------ fixture writer -----------

def _enc(value):
    """(ggml type id, bytes) for the metadata values the fixtures use."""
    if isinstance(value, bool):
        return 7, struct.pack("<B", int(value))
    if isinstance(value, int):
        return 4, struct.pack("<I", value)
    if isinstance(value, float):
        return 6, struct.pack("<f", value)
    if isinstance(value, str):
        return 8, struct.pack("<Q", len(value.encode())) + value.encode()
    if isinstance(value, list):
        if not value:
            return 9, struct.pack("<IQ", 4, 0)
        if isinstance(value[0], str):
            items = b"".join(struct.pack("<Q", len(v.encode())) + v.encode() for v in value)
            return 9, struct.pack("<IQ", 8, len(value)) + items
        items = b"".join(struct.pack("<I", v) for v in value)
        return 9, struct.pack("<IQ", 4, len(value)) + items
    raise AssertionError(f"unsupported fixture value {value!r}")


def _str(s: str) -> bytes:
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def elements_of(dims: list[int]) -> int:
    n = 1
    for d in dims:
        n *= d
    return n


def block_bytes(type_name: str, elements: int) -> int:
    elems, size = BLOCK_GEOMETRY[type_name]
    assert elements % elems == 0, f"{type_name} needs whole {elems}-value blocks"
    return elements // elems * size


def write_gguf(path: Path, meta: dict, tensors: list[tuple[str, list[int], str, int]], *,
               align: int = 32, version: int = 3, data_slack: int = 0, header_slack: int = 0,
               truncate_after: int | None = None) -> int:
    """Write a GGUF v3 header plus zero-filled tensor data; returns the size the file ends up with.

    `tensors` are (name, dims, type name, offset from the data section). `data_slack` makes the file shorter than
    the tensors need (a shard whose size does not add up); `header_slack` cuts the header itself short.
    """
    head = b"GGUF" + struct.pack("<IQQ", version, len(tensors), len(meta))
    for k, v in meta.items():
        t, raw = _enc(v)
        head += _str(k) + struct.pack("<I", t) + raw
    for name, dims, type_name, offset in tensors:
        head += _str(name) + struct.pack("<I", len(dims))
        head += b"".join(struct.pack("<Q", d) for d in dims)
        head += struct.pack("<IQ", mi.TYPE_IDS[type_name], offset)
    data_start = (len(head) + align - 1) // align * align
    # The data section is only a size: the inventory never reads it, so the file is sparse and 25 GB of
    # embedding costs no disk. `truncate` is what makes a shard "shorter than its header claims".
    need = sum(block_bytes(t, elements_of(dims)) for _n, dims, t, _o in tensors)
    size = max(0, data_start + need + data_slack)
    if header_slack:
        size = max(1, len(head) + header_slack)
    if truncate_after is not None:
        size = min(size, truncate_after)
    with path.open("wb") as f:
        f.write(head + b"\0" * (data_start - len(head)))
        f.truncate(size)
    return size


def whole_model_rows() -> list[tuple[str, list[int], str, int]]:
    """One tiny block-valid tensor for every name the TP3 path needs, at the offsets a writer would give them."""
    rows, off = [], 0
    names = mi.expected_tensor_names(mi.COMPILED_GEOMETRY) + ["per_layer_token_embd.weight"]
    for name in names:
        dims = [2560, 32]
        rows.append((name, dims, "Q8_0", off))
        off += block_bytes("Q8_0", 2560 * 32)
    return rows


def meta(**overrides) -> dict:
    """qwen4exp metadata with exactly the values the kernels are compiled for, plus one override per test."""
    m = {
        "general.architecture": "qwen4exp",
        "general.name": "tp3-fixture",
        "general.alignment": 32,
        "qwen4exp.block_count": 48,
        "qwen4exp.embedding_length": 2560,
        "qwen4exp.expert_count": 512,
        "qwen4exp.expert_used_count": 10,
        "qwen4exp.attention.head_count": 24,
        "qwen4exp.attention.head_count_kv": 2,
    }
    for k, v in overrides.items():
        if v is None:
            m.pop(k, None)
        else:
            m[k] = v
    return m


def by_name(inv: dict) -> dict[str, dict]:
    return {t["name"]: t for t in inv["tensors"]}


def geometry_problems(inv: dict) -> list[str]:
    return [r for r in inv["verdict"]["reasons"] if r.startswith("geometry:")]


def missing_roles(inv: dict) -> list[str]:
    return [r for r in inv["verdict"]["reasons"] if r.startswith("missing:")]


# ------------------------------------------------------------------ mixed types in one layer --

def test_mixed_gate_up_down_types_in_one_layer(tmp_path):
    """gate and up as different quantizations and down as a third: the inventory must keep them apart."""
    p = tmp_path / "mix-00001-of-00001.gguf"
    gate, up, down = "Q6_K", "IQ3_S", "Q8_0"
    b_gate, b_up = block_bytes(gate, 2560 * 640), block_bytes(up, 2560 * 640)
    rows = [("blk.0.ffn_gate_exps.weight", [2560, 640], gate, 0),
            ("blk.0.ffn_up_exps.weight", [2560, 640], up, b_gate),
            ("blk.0.ffn_down_exps.weight", [640, 2560], down, b_gate + b_up)]
    size = write_gguf(p, meta(), rows)

    inv = mi.inspect_model(str(p)).to_dict()
    t = by_name(inv)
    assert t["blk.0.ffn_gate_exps.weight"]["dtype"] == "Q6_K"
    assert t["blk.0.ffn_gate_exps.weight"]["block_size"] == 256
    assert t["blk.0.ffn_up_exps.weight"]["block_size"] == 256
    assert t["blk.0.ffn_down_exps.weight"]["block_size"] == 32
    # a K-quant keeps its scales in an fp16 super-block, an i-quant in a codebook stream, Q8_0 in one fp16 per block
    kinds = {t[n]["scale_kind"] for n in t}
    assert len(kinds) == 3, kinds
    assert t["blk.0.ffn_gate_exps.weight"]["family"] == "expert_gate_up"
    assert t["blk.0.ffn_up_exps.weight"]["family"] == "expert_gate_up"
    assert t["blk.0.ffn_down_exps.weight"]["family"] == "expert_down"
    # offsets and lengths are absolute bytes in the file, and they add up
    offs = [t[n]["byte_offset"] for n in t]
    assert offs == sorted(offs) and offs[0] % 32 == 0
    last = max(t[n]["byte_offset"] + t[n]["byte_length"] for n in t)
    assert last <= size
    assert t["blk.0.ffn_down_exps.weight"]["byte_length"] == block_bytes(down, 640 * 2560)
    assert sum(x["byte_length"] for x in inv["tensors"]) == inv["bytes"]["total"]
    assert inv["bytes"]["by_family"]["expert_down"] == block_bytes(down, 640 * 2560)


def test_activation_contracts_per_consumer_from_the_actual_types(tmp_path):
    """The gate/up rows and the down input need different activation quantizers; the file says which."""
    p = tmp_path / "act-00001-of-00001.gguf"
    b_gate = block_bytes("Q6_K", 2560 * 640)
    rows = [("blk.0.ffn_gate_exps.weight", [2560, 640], "Q6_K", 0),
            ("blk.0.ffn_up_exps.weight", [2560, 640], "Q6_K", b_gate),
            ("blk.0.ffn_down_exps.weight", [640, 2560], "Q8_0", 2 * b_gate)]
    write_gguf(p, meta(), rows)

    inv = mi.inspect_model(str(p)).to_dict()
    c = {a["consumer"]: a for a in inv["activation_contracts"]}
    assert c["expert_gate_up"]["quantizer"] == "Q8_K"       # ggml's vec_dot_type for Q6_K
    assert c["expert_gate_up"]["block_size"] == 256
    assert c["expert_gate_up"]["scope"] == "global"         # its input is the whole 2560-wide activation
    assert c["expert_down"]["quantizer"] == "Q8_0"          # and for Q8_0
    assert c["expert_down"]["block_size"] == 32
    assert c["expert_down"]["scope"] == "local"             # 32 divides the FFN cuts 256 and 448


def test_activation_contract_is_global_when_its_block_crosses_a_rank_cut(tmp_path):
    """A 256-value activation block cannot sit inside the 192-wide FFN slices: the quantizer must stay global."""
    p = tmp_path / "cross-00001-of-00001.gguf"
    write_gguf(p, meta(), [("blk.0.ffn_down_exps.weight", [640, 2560], "Q4_K", 0)])

    inv = mi.inspect_model(str(p)).to_dict()
    c = {a["consumer"]: a for a in inv["activation_contracts"]}
    assert c["expert_down"]["quantizer"] == "Q8_K" and c["expert_down"]["block_size"] == 256
    assert c["expert_down"]["scope"] == "global"
    assert "448" in c["expert_down"]["notes"]


def test_attention_output_is_cut_by_query_heads_not_by_the_ffn(tmp_path):
    """attn_output takes the attention output (24 heads x 256) as input, so its columns split at 2048 and 4096."""
    p = tmp_path / "attnout-00001-of-00001.gguf"
    write_gguf(p, meta(), [("blk.3.attn_output.weight", [6144, 2560], "Q6_K", 0),
                           ("blk.3.ffn_down_shexp.weight", [640, 2560], "Q4_K",
                            block_bytes("Q6_K", 6144 * 2560))])
    inv = mi.inspect_model(str(p)).to_dict()
    c = {a["consumer"]: a for a in inv["activation_contracts"]}
    assert c["attn_output"]["quantizer"] == "Q8_K" and c["attn_output"]["block_size"] == 256
    # 256 divides 2048 and 4096, so each rank quantizes the eight query heads it owns
    assert c["attn_output"]["scope"] == "local"
    assert "2048" in c["attn_output"]["notes"]
    assert c["down"]["scope"] == "global"          # the same block size does not fit the FFN cuts


def test_float_weights_report_no_activation_quantizer(tmp_path):
    p = tmp_path / "float-00001-of-00001.gguf"
    write_gguf(p, meta(), [("blk.5.ffn_gate_inp.weight", [512, 2560], "BF16", 0)])
    c = {a["consumer"]: a for a in mi.inspect_model(str(p)).to_dict()["activation_contracts"]}
    assert c["router"]["quantizer"] == "BF16" and c["router"]["block_size"] == 1
    assert c["router"]["scope"] == "global"
    assert "no activation quantizer" in c["router"]["notes"]


# ------------------------------------------------------------------ broken files -------------

def test_missing_scale_blocks_the_verdict(tmp_path):
    """A file that stops before a tensor's scales are complete is not loadable, and says which tensor."""
    p = tmp_path / "short-00001-of-00001.gguf"
    b_gate = block_bytes("Q6_K", 2560 * 640)
    rows = [("blk.0.ffn_gate_exps.weight", [2560, 640], "Q6_K", 0),
            ("blk.0.ffn_down_exps.weight", [640, 2560], "Q6_K", b_gate)]
    write_gguf(p, meta(), rows, data_slack=-4096)

    inv = mi.inspect_model(str(p)).to_dict()
    short = by_name(inv)["blk.0.ffn_down_exps.weight"]
    assert short["byte_length"] < short["expected_bytes"]
    assert inv["verdict"]["loadable_by_strata"] is False
    assert any("scale" in r and "blk.0.ffn_down_exps.weight" in r for r in inv["verdict"]["reasons"]), \
        inv["verdict"]["reasons"]
    assert inv["verdict"]["checked_tensors"] == 1


def test_truncated_header_raises(tmp_path):
    """A header that ends in the middle of the tensor list cannot be trusted: it is an error, not an inventory."""
    p = tmp_path / "trunc-00001-of-00001.gguf"
    rows = [("blk.0.ffn_gate_exps.weight", [2560, 640], "Q6_K", 0),
            ("blk.0.ffn_up_exps.weight", [2560, 640], "Q6_K", block_bytes("Q6_K", 2560 * 640))]
    write_gguf(p, meta(), rows, header_slack=-8)     # the last tensor's record is cut
    with pytest.raises(mi.InventoryError) as e:
        mi.inspect_model(str(p))
    assert "header" in str(e.value).lower()


def test_shard_sizes_that_do_not_add_up(tmp_path):
    """Two shards of one model where the second is shorter than its own tensor list says: refused by name."""
    a = tmp_path / "pair-00001-of-00002.gguf"
    b = tmp_path / "pair-00002-of-00002.gguf"
    m = meta(**{"general.split": True, "general.split_count": 2, "split.tensors.count": 2})
    write_gguf(a, m, [("blk.0.ffn_gate_exps.weight", [2560, 640], "Q6_K", 0)])
    write_gguf(b, {}, [("blk.0.ffn_down_exps.weight", [640, 2560], "Q6_K", 0)], data_slack=-4096)

    inv = mi.inspect_model(str(tmp_path)).to_dict()
    assert inv["verdict"]["loadable_by_strata"] is False
    reasons = " ".join(inv["verdict"]["reasons"])
    assert "pair-00002-of-00002.gguf" in reasons
    assert "blk.0.ffn_down_exps.weight" in reasons


def test_split_metadata_mismatch_is_a_reason(tmp_path):
    a = tmp_path / "mis-00001-of-00002.gguf"
    b = tmp_path / "mis-00002-of-00002.gguf"
    m = meta(**{"general.split": True, "general.split_count": 3, "split.tensors.count": 9})
    write_gguf(a, m, [("blk.0.ffn_gate_exps.weight", [2560, 640], "Q6_K", 0)])
    write_gguf(b, {}, [("blk.0.ffn_up_exps.weight", [2560, 640], "Q6_K", 0)])

    inv = mi.inspect_model(str(tmp_path)).to_dict()
    assert inv["verdict"]["loadable_by_strata"] is False
    reasons = " ".join(inv["verdict"]["reasons"])
    assert "general.split_count" in reasons and "split.tensors.count" in reasons


def test_overlapping_tensor_ranges_are_a_reason(tmp_path):
    """Two tensors claiming the same bytes cannot both be right; the inventory refuses rather than guesses."""
    p = tmp_path / "over-00001-of-00001.gguf"
    rows = [("blk.0.ffn_gate_exps.weight", [2560, 640], "Q6_K", 0),
            ("blk.0.ffn_up_exps.weight", [2560, 640], "Q6_K", 0)]        # the same offset as gate
    write_gguf(p, meta(), rows)
    inv = mi.inspect_model(str(p)).to_dict()
    assert inv["verdict"]["loadable_by_strata"] is False
    assert any("overlap" in r for r in inv["verdict"]["reasons"]), inv["verdict"]["reasons"]


def test_foreign_geometry_names_the_metadata_key(tmp_path):
    """n_ff = 768 is another model's FFN: the file's value is recorded, and the verdict names the key."""
    p = tmp_path / "ffn768-00001-of-00001.gguf"
    write_gguf(p, meta(**{"qwen4exp.expert_feed_forward_length": 768,
                          "qwen4exp.expert_shared_feed_forward_length": 768}),
               [("blk.0.ffn_gate_shexp.weight", [2560, 768], "Q6_K", 0)])
    inv = mi.inspect_model(str(p)).to_dict()
    assert inv["geometry"]["n_ff"] == 768
    assert inv["verdict"]["loadable_by_strata"] is False
    assert any("expert_feed_forward_length" in r and "768" in r for r in geometry_problems(inv)), \
        inv["verdict"]["reasons"]


def test_foreign_gdn_head_count_is_a_geometry_problem(tmp_path):
    p = tmp_path / "gdn18-00001-of-00001.gguf"
    write_gguf(p, meta(**{"qwen4exp.ssm.group_count": 18, "qwen4exp.ssm.state_size": 128,
                          "qwen4exp.ssm.time_step_rank": 48, "qwen4exp.ssm.inner_size": 6144}),
               [("blk.0.attn_qkv.weight", [2560, 10240], "Q6_K", 0)])
    inv = mi.inspect_model(str(p)).to_dict()
    assert inv["geometry"]["ssm_k_heads"] == 18
    assert any("ssm.group_count" in r and "18" in r for r in geometry_problems(inv)), inv["verdict"]["reasons"]


def test_other_architecture_is_refused(tmp_path):
    p = tmp_path / "other-00001-of-00001.gguf"
    write_gguf(p, meta(**{"general.architecture": "qwen35moe"}), [])
    inv = mi.inspect_model(str(p)).to_dict()
    assert inv["verdict"]["loadable_by_strata"] is False
    assert any("qwen35moe" in r for r in inv["verdict"]["reasons"])


# ------------------------------------------------------------------ geometry -----------------

def test_geometry_reports_every_compiled_field(tmp_path):
    """Task 2 reads the geometry from the file, so every ModelGeometry field has to be in the inventory."""
    p = tmp_path / "geo-00001-of-00001.gguf"
    write_gguf(p, meta(**{"qwen4exp.full_attention_interval": 4, "qwen4exp.ssm.state_size": 128,
                          "qwen4exp.ssm.group_count": 16, "qwen4exp.ssm.time_step_rank": 48,
                          "qwen4exp.ssm.conv_kernel": 4, "qwen4exp.ssm.inner_size": 6144,
                          "qwen4exp.attention.key_length": 256, "qwen4exp.attention.value_length": 256,
                          "qwen4exp.rope.dimension_count": 64, "qwen4exp.attention.indexer.head_count": 4,
                          "qwen4exp.attention.indexer.key_length": 128,
                          "qwen4exp.attention.indexer.top_k": 2048,
                          "qwen4exp.hyper_connection.count": 4, "qwen4exp.hyper_connection.low_rank": 320,
                          "qwen4exp.expert_feed_forward_length": 640,
                          "qwen4exp.expert_shared_feed_forward_length": 640}), [])
    g = mi.inspect_model(str(p)).to_dict()["geometry"]
    assert set(g) == set(mi.COMPILED_GEOMETRY)
    assert g["n_embd"] == 2560 and g["n_layers"] == 48 and g["n_ff"] == 640
    assert g["ssm_conv_channels"] == 2 * 128 * 16 + 128 * 48          # derived, as geometry.hpp derives it
    assert g["family"] == "qwen4exp"


def test_a_complete_clean_model_is_loadable(tmp_path):
    """A file that carries every name the TP3 path reads, with whole blocks: the verdict has to say true."""
    p = tmp_path / "whole-00001-of-00001.gguf"
    write_gguf(p, meta(), whole_model_rows())
    inv = mi.inspect_model(str(p)).to_dict()
    assert inv["verdict"]["reasons"] == [], inv["verdict"]["reasons"]
    assert inv["verdict"]["loadable_by_strata"] is True
    assert inv["verdict"]["checked_tensors"] == len(inv["tensors"])


def test_rms_eps_and_vocab_come_from_the_file(tmp_path):
    p = tmp_path / "eps-00001-of-00001.gguf"
    write_gguf(p, meta(**{"qwen4exp.attention.layer_norm_rms_epsilon": 1e-5,
                          "tokenizer.ggml.tokens": ["a", "b", "c"]}), [])
    g = mi.inspect_model(str(p)).to_dict()["geometry"]
    assert abs(g["rms_eps"] - 1e-5) < 1e-9
    assert g["n_vocab"] == 3


# ------------------------------------------------------------------ mapping and assets -------

@pytest.mark.parametrize("name,want", [
    ("blk.5.ffn_gate_exps.weight", "expert_gate_up"),
    ("blk.5.ffn_up_exps.weight", "expert_gate_up"),
    ("blk.5.ffn_down_exps.weight", "expert_down"),
    ("blk.5.ffn_gate_shexp.weight", "gate_up"),
    ("blk.5.ffn_up_shexp.weight", "gate_up"),
    ("blk.5.ffn_down_shexp.weight", "down"),
    ("blk.5.ffn_gate_inp.weight", "router"),
    ("blk.5.attn_qkv.weight", "gdn_key"),
    ("blk.5.attn_gate.weight", "gdn_value"),
    ("blk.5.ssm_conv1d.weight", "gdn_key"),
    ("blk.5.ssm_out.weight", "gdn_value"),
    ("blk.5.attn_q.weight", "qsa_q"),
    ("blk.5.attn_k.weight", "qsa_kv"),
    ("blk.5.attn_v.weight", "qsa_kv"),
    ("blk.5.indexer.q_proj.weight", "qsa_idx"),
    ("blk.5.indexer.k_proj.weight", "qsa_idx"),
    ("blk.5.attn_output.weight", "down"),
    ("blk.5.hc_ffn_up.weight", "hc"),
    ("output_hc_down.weight", "hc"),
    ("blk.5.attn_q_norm.weight", "norm"),
    ("blk.5.ple_key.weight", "ple"),
    ("per_layer_token_embd.weight", "ple"),
    ("token_embd.weight", "embed"),
    ("output.weight", "head"),
])
def test_tensor_family_mapping(name, want):
    assert mi.tensor_family(name) == want


def test_assets_name_the_tokenizer_draft_and_ple_parts(tmp_path):
    p = tmp_path / "assets-00001-of-00001.gguf"
    write_gguf(p, meta(**{"tokenizer.ggml.model": "gpt2",
                          "tokenizer.ggml.bos_token_id": 1,
                          "tokenizer.ggml.eos_token_id": 2,
                          "tokenizer.ggml.tokens": ["a", "b", "c"],
                          "tokenizer.chat_template": "{% if x %}",
                          "qwen4exp.nextn_predict_layers": 1}),
               [("per_layer_token_embd.weight", [2560, 48], "IQ4_NL", 0),
                ("blk.48.nextn.eh_proj.weight", [2560, 5120], "Q6_K", block_bytes("IQ4_NL", 2560 * 48))])
    a = mi.inspect_model(str(p)).to_dict()["assets"]
    assert a["tokenizer"]["model"] == "gpt2" and a["tokenizer"]["n_vocab"] == 3
    assert a["tokenizer"]["chat_template"] is True
    assert a["ple"]["tensors"] == 1 and a["ple"]["present"] is True
    assert a["draft"]["present"] is True and a["draft"]["n_mtp_layers"] == 1


def test_draft_assets_report_a_missing_draft(tmp_path):
    p = tmp_path / "nodraft-00001-of-00001.gguf"
    write_gguf(p, meta(), [("blk.0.ffn_gate_shexp.weight", [2560, 640], "Q6_K", 0)])
    a = mi.inspect_model(str(p)).to_dict()["assets"]
    assert a["draft"]["present"] is False and a["ple"]["present"] is False


def test_verdict_needs_every_tensor_the_tp3_path_reads(tmp_path):
    """Nothing is broken here: the file simply is not a whole model, so the verdict has to be false."""
    p = tmp_path / "part-00001-of-00001.gguf"
    b = block_bytes("Q6_K", 2560 * 640)
    rows = [("blk.0.ffn_gate_shexp.weight", [2560, 640], "Q6_K", 0),
            ("blk.0.ffn_up_shexp.weight", [2560, 640], "Q6_K", b),
            ("blk.0.ffn_down_shexp.weight", [640, 2560], "Q8_0", 2 * b)]
    write_gguf(p, meta(), rows)
    inv = mi.inspect_model(str(p)).to_dict()
    assert inv["verdict"]["loadable_by_strata"] is False
    roles = " ".join(missing_roles(inv))
    assert "token_embd.weight" in roles and "output.weight" in roles
    assert inv["verdict"]["checked_tensors"] == 3


def test_schema_shape_is_the_shared_contract(tmp_path):
    p = tmp_path / "schema-00001-of-00001.gguf"
    write_gguf(p, meta(), [("blk.0.ffn_gate_shexp.weight", [2560, 640], "Q6_K", 0)])
    inv = mi.inspect_model(str(p)).to_dict()
    assert inv["schema_version"] == 1
    assert set(inv) == {"schema_version", "source", "geometry", "tensors", "activation_contracts",
                        "assets", "bytes", "verdict", "notes"}
    assert set(inv["source"]) == {"kind", "repo", "revision", "files"}
    assert set(inv["source"]["files"][0]) == {"path", "size", "sha256"}
    assert set(inv["verdict"]) == {"loadable_by_strata", "reasons", "checked_tensors"}
    for t in inv["tensors"]:
        assert set(t) <= {"name", "file", "dtype", "shape", "block_size", "scale_kind",
                         "byte_offset", "byte_length", "family", "elements", "expected_bytes"}
    json.dumps(inv)                                     # the inventory has to be plain JSON


def test_hub_revision_is_pinned_to_the_selected_files():
    """ModelScope reports a commit per file: the inventory must report the one the chosen variant shares."""
    src = mi.qs.Source
    rev = {"IQ2_XS/a.gguf": "r1", "IQ3_S/b.gguf": "r2", "IQ3_S/c.gguf": "r2"}
    assert mi._common_revision(rev, [src("IQ3_S/b.gguf", 1), src("IQ3_S/c.gguf", 1)]) == "r2"
    assert mi._common_revision(rev, [src("IQ3_S/b.gguf", 1), src("IQ2_XS/a.gguf", 1)]) is None


def test_cli_help_lists_the_source_argument():
    out = subprocess.run([sys.executable, str(ROOT / "tools" / "distributed" / "model_inventory.py"), "--help"],
                         capture_output=True, text=True, check=True)
    assert "--revision" in out.stdout and "source" in out.stdout


# ------------------------------------------------------------------ real pack (external) -----

@pytest.mark.skipif(not os.environ.get("STRATA_TP3_NETWORK"),
                    reason="reads a real repository over the network (set STRATA_TP3_NETWORK=1)")
def test_real_iq3_s_pack_headers():
    """The candidate pack, pinned: every tensor's type must be one Strata's kernels cover."""
    src = "hf:ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF@ed59f92082b1e93c0e96d60a8b11aab089b52f09"
    inv = mi.inspect_model(src, variant="IQ3_S").to_dict()
    assert inv["source"]["revision"] == "ed59f92082b1e93c0e96d60a8b11aab089b52f09"
    assert inv["geometry"]["n_ff"] == 640 and inv["geometry"]["n_expert"] == 512
    assert len(inv["tensors"]) > 1000
    assert all(t["dtype"] in mi.TYPE_IDS for t in inv["tensors"])
    assert inv["bytes"]["by_family"]["expert_down"] > 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))

