// tests/distributed/shard_plan_test.cpp - checks for the TP shard planner: the agreed world-3 layout, the
// weight-block rule, coverage without gaps, the GQA map, the reference-path fallbacks, the memory numbers
// and the memory gate, plus the inventory.json reader and the tp-plan manifest Task 3 consumes.
//
// The fixture follows GGUF order: axis 0 is the contiguous axis, and blocks run along it.  Every
// byte_length is a whole number of blocks at the real block sizes, so the byte stride the plan reports is
// the real one.  No GPU, no NCCL, no framework: plain checks, printed results, non-zero exit if any fails.
//
// Without arguments: the fixture suite.  With
//     --inventory PATH --world N --out PATH [--gate-bytes N]
// it plans a real inventory.json (the one Task 1 froze) and writes the manifest.
#include "strata/distributed/shard_plan.hpp"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

using strata::distributed::ActivationContract;
using strata::distributed::InventoryActivation;
using strata::distributed::InventoryTensor;
using strata::distributed::kSparkGateBytes;
using strata::distributed::make_tp_plan;
using strata::distributed::ModelInventory;
using strata::distributed::parse_inventory_json;
using strata::distributed::ShardPlan;
using strata::distributed::TensorFamily;
using strata::distributed::TensorSlice;
using strata::distributed::to_tp_plan_json;
using strata::distributed::validate_plan;
using strata::distributed::ValidationResult;

int g_checks = 0;
int g_failed = 0;
std::string g_test = "setup";

void begin(const std::string& name) {
    g_test = name;
    std::cout << "== " << name << "\n";
}

void check(bool ok, const std::string& what) {
    ++g_checks;
    if (ok) return;
    ++g_failed;
    std::cout << "FAIL [" << g_test << "] " << what << "\n";
}

template <typename A, typename B>
void check_eq(const A& got, const B& want, const std::string& what) {
    ++g_checks;
    if (got == want) return;
    ++g_failed;
    std::cout << "FAIL [" << g_test << "] " << what << ": got " << got << ", want " << want << "\n";
}

void check_ok(const ValidationResult& v, const std::string& what) {
    ++g_checks;
    if (v.ok) return;
    ++g_failed;
    std::cout << "FAIL [" << g_test << "] " << what << " is not valid:\n";
    for (const auto& e : v.errors) std::cout << "     error: " << e << "\n";
}

void check_has(const std::vector<std::string>& msgs, const std::string& needle, const std::string& what) {
    ++g_checks;
    for (const auto& m : msgs)
        if (m.find(needle) != std::string::npos) return;
    ++g_failed;
    std::cout << "FAIL [" << g_test << "] " << what << ": no message contains '" << needle << "'\n";
    for (const auto& m : msgs) std::cout << "     seen: " << m << "\n";
}

// ---------------------------------------------------------------------------
// The fixture: one layer of each kind, GGUF order, real block sizes and byte lengths.
// ---------------------------------------------------------------------------

InventoryTensor mk(const std::string& name, const char* dtype, std::vector<int64_t> shape, int64_t block,
                   const std::string& scale, int64_t len, const char* family) {
    InventoryTensor t;
    t.name = name;
    t.file = "fixture-00001-of-00002.gguf";
    t.dtype = dtype;
    t.shape = std::move(shape);
    t.block_size = block;
    t.scale_kind = scale;
    t.byte_offset = 1024;
    t.byte_length = len;
    t.family = strata::distributed::family_from_string(family);
    return t;
}

InventoryActivation act(const std::string& consumer, const std::string& quantizer, int64_t block,
                        const std::string& scope) {
    InventoryActivation a;
    a.consumer = consumer;
    a.quantizer = quantizer;
    a.block_size = block;
    a.scope = scope;
    return a;
}

ModelInventory fixture() {
    ModelInventory inv;
    inv.source.kind = "local";
    inv.source.repo = "fixture/Qwen3.8-Flash-Next-TP";
    inv.source.revision = "fixture-revision";
    inv.inventory_hash = "fixture-inventory-hash";
    inv.verdict.loadable = true;
    inv.verdict.checked_tensors = 17;
    const std::string sc_sub = "fp16-superblock-scale-plus-sub-scales";
    const std::string sc_nl = "fp16-scale-plus-codebook-codes";
    inv.tensors = {
        // GDN (36 of 48 layers): packed attn_qkv [q | k | v], the value-side gate, the output projection.
        mk("blk.0.attn_qkv.weight", "Q6_K", {2560, 10240}, 256, sc_sub, 21504000, "gdn_key"),
        mk("blk.0.attn_gate.weight", "Q6_K", {2560, 6144}, 256, sc_sub, 12902400, "gdn_value"),
        mk("blk.0.ssm_out.weight", "Q6_K", {6144, 2560}, 256, sc_sub, 12902400, "gdn_value"),
        // QSA (12 of 48 layers): attn_q as [q | gate] per head, KV, the indexer, the output projection.
        mk("blk.3.attn_q.weight", "Q6_K", {2560, 12288}, 256, sc_sub, 25804800, "qsa_q"),
        mk("blk.3.attn_k.weight", "Q6_K", {2560, 512}, 256, sc_sub, 1075200, "qsa_kv"),
        mk("blk.3.indexer.q_proj.weight", "BF16", {2560, 512}, 1, "", 2621440, "qsa_idx"),
        mk("blk.3.attn_output.weight", "Q4_K", {6144, 2560}, 256, sc_sub, 8847360, "down"),
        // MoE (every layer): the FFN is 640 wide, and its down side is the block axis.
        mk("blk.0.ffn_gate_shexp.weight", "Q4_K", {2560, 640}, 256, sc_sub, 921600, "gate_up"),
        mk("blk.0.ffn_down_shexp.weight", "IQ4_NL", {640, 2560}, 32, sc_nl, 921600, "down"),
        mk("blk.0.ffn_gate_exps.weight", "IQ3_XXS", {2560, 640, 512}, 256, sc_nl, 321126400, "expert_gate_up"),
        mk("blk.0.ffn_down_exps.weight", "IQ4_NL", {640, 2560, 512}, 32, sc_nl, 471859200, "expert_down"),
        mk("blk.1.ffn_down_exps.weight", "Q2_0", {640, 2560, 512}, 64, "fp16-scale-per-block", 235929600,
           "expert_down"),
        // Replicated by design: router, norms, PLE, vocabulary.
        mk("blk.0.ffn_gate_inp.weight", "BF16", {2560, 512}, 1, "", 2621440, "router"),
        mk("blk.0.ssm_norm.weight", "F32", {2560}, 1, "", 10240, "norm"),
        mk("per_layer_token_embd.weight", "IQ4_NL", {160, 320001536}, 32, sc_nl, 28800138240, "ple"),
        mk("token_embd.weight", "IQ4_XS", {2560, 248320}, 256, sc_nl, 337715200, "embed"),
        mk("output.weight", "Q6_K", {2560, 248320}, 256, sc_sub, 521472000, "head"),
    };
    inv.activation_contracts = {
        act("down", "Q8_0", 32, "local"),
        act("expert_down", "Q8_0", 32, "local"),
        act("attn_output", "Q8_K", 256, "local"),
        act("gate_up", "Q8_K", 256, "global"),
        act("ssm_out", "Q8_K", 256, "global"),
        act("norm", "none", 1, "global"),
        act("embed", "none", 1, "local"),
        act("ple", "none", 1, "local"),
    };
    return inv;
}

const TensorSlice* slice_at(const ShardPlan& p, const std::string& tensor, const std::string& part, int rank) {
    for (const auto& s : p.slices)
        if (s.tensor == tensor && s.part == part && s.rank == rank) return &s;
    return nullptr;
}

std::vector<const TensorSlice*> slices_of(const ShardPlan& p, const std::string& tensor) {
    std::vector<const TensorSlice*> out;
    for (const auto& s : p.slices)
        if (s.tensor == tensor) out.push_back(&s);
    return out;
}

const ActivationContract* contract_at(const ShardPlan& p, const std::string& consumer) {
    for (const auto& c : p.activation)
        if (c.consumer == consumer) return &c;
    return nullptr;
}

int64_t component(const ShardPlan& p, const std::string& name) {
    for (const auto& m : p.memory.components)
        if (m.first == name) return m.second;
    return -1;
}

/// Every tensor is owned end to end: the ranges of one tensor tile [0, global_shape[axis]) with no gap and
/// no overlap, and a replicated tensor is owned whole by every rank.
void check_coverage(const ShardPlan& p, const std::string& what) {
    check(!p.slices.empty(), what + ": the plan has slices");
    std::vector<std::string> names;
    for (const auto& s : p.slices)
        if (std::find(names.begin(), names.end(), s.tensor) == names.end()) names.push_back(s.tensor);
    for (const auto& name : names) {
        auto grp = slices_of(p, name);
        const int64_t axis = grp.front()->axis;
        const int64_t total = grp.front()->global_shape[(size_t)axis];
        const std::string tag = what + " / " + name;
        if (grp.front()->replicated) {
            check_eq((int64_t)grp.size(), (int64_t)p.world, tag + ": one whole copy per rank");
            for (const auto* s : grp)
                check(s->range_start == 0 && s->range_end == total,
                      tag + ": rank " + std::to_string(s->rank) + " owns the whole range");
            continue;
        }
        std::vector<std::pair<int64_t, int64_t>> rs;
        for (const auto* s : grp) rs.push_back({s->range_start, s->range_end});
        std::sort(rs.begin(), rs.end());
        check(!rs.empty() && rs.front().first == 0 && rs.back().second == total,
              tag + ": ownership does not span [0," + std::to_string(total) + ")");
        bool tiled = true;
        for (size_t i = 1; i < rs.size(); ++i)
            if (rs[i].first != rs[i - 1].second) tiled = false;
        check(tiled, tag + ": gap or overlap between ranges");
    }
}

/// No slice cuts a weight block in the middle on the axis where the blocks run.
void check_blocks_whole(const ShardPlan& p, const std::string& what) {
    bool all_whole = true;
    for (const auto& s : p.slices)
        if (s.block_size > 1 && s.axis == 0 && s.range_start % s.block_size != 0) all_whole = false;
    check(all_whole, what);
}

// ---------------------------------------------------------------------------
// The agreed layout
// ---------------------------------------------------------------------------

void test_world1_identity() {
    begin("world 1 is identity");
    ModelInventory inv = fixture();
    ShardPlan p = make_tp_plan(inv, 1);
    check_eq(p.world, 1, "world 1 is accepted");
    check_ok(validate_plan(p), "the identity plan");
    check_eq((int64_t)p.slices.size(), (int64_t)inv.tensors.size(), "one slice per tensor");
    bool whole_copies = true;
    for (const auto& s : p.slices)
        if (!s.replicated || s.rank != 0 || !s.part.empty()) whole_copies = false;
    check(whole_copies, "every identity slice is a whole copy owned by rank 0");
    check(contract_at(p, "tp.down_partial_sum") == nullptr, "world 1 adds no all-reduce contract");
    const ActivationContract* down = contract_at(p, "down");
    check(down != nullptr && down->gather == "none" && down->reference == "local",
          "world 1 leaves each activation contract where it is");
    const ActivationContract* gu = contract_at(p, "gate_up");
    check(gu != nullptr && gu->gather == "none" && gu->reference == "global_reference",
          "world 1 keeps a global quantizer global");
}

void test_world3_layout() {
    begin("world 3 is the agreed layout");
    ModelInventory inv = fixture();
    ShardPlan p = make_tp_plan(inv, 3);
    check_eq(p.world, 3, "world 3 is accepted");
    check_ok(validate_plan(p), "the world 3 plan");
    check_coverage(p, "world 3");
    check_blocks_whole(p, "no world 3 slice splits a weight block");
    check(p.memory.qualification == "unqualified_lower_bound", "memory estimate does not qualify capacity");
    // Independent oracle: the monolithic recurrence selects key = global_value % 16.
    for (int rank = 0; rank < 3; ++rank) {
        std::vector<int64_t> actual, expected;
        for (const auto& s : p.slices)
            if (s.tensor == "blk.0.attn_qkv.weight" && s.part == "gdn_v" && s.rank == rank)
                for (int64_t h = s.head_start; h < s.head_end; ++h) actual.push_back(h);
        const int64_t start = rank == 0 ? 0 : rank == 1 ? 6 : 11;
        const int64_t end = rank == 0 ? 6 : rank == 1 ? 11 : 16;
        for (int64_t h = 0; h < 48; ++h)
            if (h % 16 >= start && h % 16 < end) expected.push_back(h);
        check(actual == expected, "GDN values follow the monolithic modulo-16 key map");
    }

    // The FFN: 640 columns, the largest block along the cut axis is 64, so 256/192/192 at 0/256/448.
    const int64_t ffn_starts[3] = {0, 256, 448};
    const int64_t ffn_ends[3] = {256, 448, 640};
    for (int r = 0; r < 3; ++r) {
        const std::string rk = " rank" + std::to_string(r);
        const TensorSlice* in = slice_at(p, "blk.0.ffn_down_shexp.weight", "ffn_in", r);
        const TensorSlice* gate = slice_at(p, "blk.0.ffn_gate_shexp.weight", "ffn_out", r);
        const TensorSlice* exps = slice_at(p, "blk.0.ffn_down_exps.weight", "ffn_in", r);
        const TensorSlice* gate_exps = slice_at(p, "blk.0.ffn_gate_exps.weight", "ffn_out", r);
        check(in && in->range_start == ffn_starts[r] && in->range_end == ffn_ends[r],
              "down" + rk + " owns FFN columns [" + std::to_string(ffn_starts[r]) + "," +
                  std::to_string(ffn_ends[r]) + ")");
        check(gate && gate->range_start == ffn_starts[r] && gate->range_end == ffn_ends[r],
              "gate/up" + rk + " owns the same FFN columns");
        check(exps && exps->range_start == ffn_starts[r] && exps->range_end == ffn_ends[r],
              "expert_down" + rk + " owns the same FFN columns");
        check(gate_exps && gate_exps->axis == 1, "expert_gate_up" + rk + " is cut on the output axis");
        check(in && in->axis == 0 && in->byte_row_stride == 360,
              "down" + rk + " keeps the original 360 B slab stride");
        check(in && in->block_size == 32 && in->scale_kind == "fp16-scale-plus-codebook-codes" &&
                  in->scale_ref.empty(),
              "down" + rk + " keeps its codec, block size and says how its scales ride");
    }
    check(component(p, "weights_owned_ffn.rank0") > component(p, "weights_owned_ffn.rank1"),
          "the FFN split is unequal by design (rank 0 owns 256 of 640 columns)");

    // Packed GDN attn_qkv is [q | k | v]: each rank keeps whole key heads and their value heads.
    const int64_t q_starts[3] = {0, 768, 1408}, q_ends[3] = {768, 1408, 2048};
    const int64_t k_starts[3] = {2048, 2816, 3456}, k_ends[3] = {2816, 3456, 4096};
    const int64_t v_starts[3] = {4096, 4864, 5504}, v_ends[3] = {4864, 5504, 6144};
    for (int r = 0; r < 3; ++r) {
        const std::string rk = " rank" + std::to_string(r);
        const TensorSlice* q = slice_at(p, "blk.0.attn_qkv.weight", "gdn_q", r);
        const TensorSlice* k = slice_at(p, "blk.0.attn_qkv.weight", "gdn_k", r);
        const TensorSlice* v = slice_at(p, "blk.0.attn_qkv.weight", "gdn_v", r);
        check(q && q->range_start == q_starts[r] && q->range_end == q_ends[r],
              "attn_qkv" + rk + " owns the q columns of its key heads");
        check(k && k->range_start == k_starts[r] && k->range_end == k_ends[r],
              "attn_qkv" + rk + " owns the k columns of its key heads");
        check(v && v->range_start == v_starts[r] && v->range_end == v_ends[r],
              "attn_qkv" + rk + " owns the value heads of exactly its own keys");
        check(q && k && v && (v->head_end - v->head_start) == (k->head_end - k->head_start),
              "attn_qkv" + rk + " keeps one value run of exactly its key count");
    }

    // The value side: attn_gate is cut on the output axis, so no block is ever touched.
    const int64_t val_starts[3] = {0, 768, 1408}, val_ends[3] = {768, 1408, 2048};
    for (int r = 0; r < 3; ++r) {
        const TensorSlice* g = slice_at(p, "blk.0.attn_gate.weight", "value_out", r);
        check(g && g->axis == 1 && g->range_start == val_starts[r] && g->range_end == val_ends[r],
              "attn_gate rank" + std::to_string(r) + " owns value heads [" + std::to_string(val_starts[r]) +
                  "," + std::to_string(val_ends[r]) + ") on the output axis");
    }
    // ssm_out reads its 6144 inputs along the block axis: 1408 splits a 256 block, so every rank reads it
    // whole and says so.  That is the reference path, not a requantization.
    auto ssm = slices_of(p, "blk.0.ssm_out.weight");
    check_eq((int64_t)ssm.size(), (int64_t)3, "ssm_out is owned whole by all three ranks");
    bool ssm_whole = true;
    for (const auto* s : ssm)
        if (!s->replicated || s->range_end != 6144 || s->compute_policy != "rank0_full_others_zero")
            ssm_whole = false;
    check(ssm_whole, "ssm_out falls back to whole-rank reads instead of requantizing");
    check_has(p.notes, "blk.0.ssm_out.weight", "ssm_out's fallback is recorded in the plan notes");
    check_has(p.notes, "splits a 256 block", "the note names the block that would be split");
    check_has(p.notes, "1408", "the note names the offset where it would be split");

    // QSA attn_q: whole query heads, each keeping its gate half, with the global 24Q/2KV map.
    const int64_t qa_starts[3] = {0, 4096, 8192}, qa_ends[3] = {4096, 8192, 12288};
    for (int r = 0; r < 3; ++r) {
        const TensorSlice* q = slice_at(p, "blk.3.attn_q.weight", "qsa_head", r);
        check(q && q->axis == 1 && q->range_start == qa_starts[r] && q->range_end == qa_ends[r],
              "attn_q rank" + std::to_string(r) + " owns whole query heads");
        check(q && q->cut_unit == 512, "attn_q cuts at a whole head (q + gate, 2 x 256 columns)");
    }
    const TensorSlice* q1 = slice_at(p, "blk.3.attn_q.weight", "qsa_head", 1);
    const std::vector<int64_t> want_map = {0, 0, 0, 0, 1, 1, 1, 1};
    check(q1 && q1->gqa_map == want_map,
          "attn_q rank 1 maps its query heads to global kv heads 0,0,0,0,1,1,1,1");

    // QSA attn_output: the 6144-wide query output cut at the query boundaries.
    const int64_t out_starts[3] = {0, 2048, 4096}, out_ends[3] = {2048, 4096, 6144};
    for (int r = 0; r < 3; ++r) {
        const TensorSlice* o = slice_at(p, "blk.3.attn_output.weight", "qsa_out", r);
        check(o && o->axis == 0 && o->range_start == out_starts[r] && o->range_end == out_ends[r],
              "attn_output rank" + std::to_string(r) + " owns query columns [" +
                  std::to_string(out_starts[r]) + "," + std::to_string(out_ends[r]) + ")");
        check(o && o->range_start % 256 == 0, "attn_output cuts land on 256-block boundaries");
    }

    // Replicated by design: KV, the indexer, the router, norms, PLE, vocabulary.
    const char* replicated[] = {"blk.3.attn_k.weight", "blk.3.indexer.q_proj.weight",
                                "blk.0.ffn_gate_inp.weight", "blk.0.ssm_norm.weight",
                                "per_layer_token_embd.weight", "token_embd.weight", "output.weight"};
    for (const char* name : replicated) {
        auto grp = slices_of(p, name);
        bool ok = grp.size() == 3;
        for (const auto* s : grp)
            if (!s->replicated || s->range_start != 0 || !s->part.empty()) ok = false;
        check(ok, std::string(name) + " is replicated to every rank");
    }
    check_eq(component(p, "weights_replicated_ple"), (int64_t)28800138240,
             "the PLE component is the PLE's own size");
    check_has(p.notes, "ple replicated: 28800138240 B", "the plan states the replicated PLE in bytes");
    check_has(p.notes, "% of the pack", "the plan states what share of the pack the PLE is");
}

void test_world2_control() {
    begin("world 2 is a supported control");
    ModelInventory inv = fixture();
    ShardPlan p = make_tp_plan(inv, 2);
    check_eq(p.world, 2, "world 2 is accepted");
    check_ok(validate_plan(p), "the world 2 plan");
    check_coverage(p, "world 2");
    check_blocks_whole(p, "no world 2 slice splits a weight block");
    const TensorSlice* in0 = slice_at(p, "blk.0.ffn_down_shexp.weight", "ffn_in", 0);
    const TensorSlice* in1 = slice_at(p, "blk.0.ffn_down_shexp.weight", "ffn_in", 1);
    check(in0 && in0->range_end == 320 && in1 && in1->range_start == 320 && in1->range_end == 640,
          "the FFN splits 320/320 at 320, a multiple of the 64 block");
    const TensorSlice* k0 = slice_at(p, "blk.0.attn_qkv.weight", "gdn_k", 0);
    const TensorSlice* v0 = slice_at(p, "blk.0.attn_qkv.weight", "gdn_v", 0);
    check(k0 && k0->head_start == 0 && k0->head_end == 8, "world 2 gives rank 0 eight of sixteen key heads");
    check(v0 && v0->head_start == 0 && v0->head_end == 8, "and the first eight-value run those keys own");
    // With two ranks each value run cuts at 1024, which a 256 block divides.
    const TensorSlice* out0 = slice_at(p, "blk.0.ssm_out.weight", "value_out", 0);
    const TensorSlice* out1 = slice_at(p, "blk.0.ssm_out.weight", "value_out", 1);
    check(out0 && out0->range_end == 1024 && out1 && out1->range_start == 1024,
          "world 2 cuts the first ssm_out run at 1024, inside no 256 block, so it is sharded, not replicated");
    check(out0 && out0->cut_unit == 128, "ssm_out is still cut at whole value heads");
    const TensorSlice* q0 = slice_at(p, "blk.3.attn_q.weight", "qsa_head", 0);
    const TensorSlice* q1 = slice_at(p, "blk.3.attn_q.weight", "qsa_head", 1);
    const std::vector<int64_t> want0(12, 0), want1(12, 1);
    check(q0 && q0->gqa_map == want0 && q1 && q1->gqa_map == want1,
          "world 2 splits the twenty-four query heads 12/12 and maps them to global kv heads 0 and 1");
}

void test_worlds_outside_the_promise() {
    begin("worlds outside 1..3 are rejected, not promised");
    ModelInventory inv = fixture();
    const int worlds[] = {0, 4, -1};
    for (int w : worlds) {
        ShardPlan p = make_tp_plan(inv, w);
        check_eq(p.world, 0, "world " + std::to_string(w) + " is not planned");
        check(!p.reject_reasons.empty(), "world " + std::to_string(w) + " says why");
        check_has(p.reject_reasons, "world " + std::to_string(w), "the reason names the world");
        ValidationResult v = validate_plan(p);
        check(!v.ok, "a rejected plan does not validate");
        check_has(v.errors, "rejected at build time", "validation repeats the build-time reason");
    }
}

// ---------------------------------------------------------------------------
// Block rule, strides, scales, verdict
// ---------------------------------------------------------------------------

void test_ffn_block_reject() {
    begin("an FFN cut that would split a block rejects the plan");
    ModelInventory inv = fixture();
    // Two FFN block sizes that do not divide each other: the cut snaps to the larger and breaks the smaller,
    // and requantizing to make it fit is not allowed.
    for (auto& t : inv.tensors)
        if (t.name == "blk.0.ffn_down_exps.weight") t.block_size = 96;
    ShardPlan p = make_tp_plan(inv, 3);
    check_eq(p.world, 0, "the plan is rejected");
    check_has(p.reject_reasons, "is cut at offset 640", "the reason names the offset that splits a block");
    check_has(p.reject_reasons, "requantizing to fit is not allowed", "the reason says requantizing is out");
    check_has(p.reject_reasons, "block size 96", "the reason names the block size being broken");
    ValidationResult v = validate_plan(p);
    check(!v.ok, "the rejected plan does not validate");
    check_has(v.errors, "blk.0.ffn_down_exps.weight", "and names the tensor that cannot be cut");
}

void test_validator_block_gap() {
    begin("the validator catches a moved FFN cut and the gap it leaves");
    ModelInventory inv = fixture();
    ShardPlan p = make_tp_plan(inv, 3);
    check_ok(validate_plan(p), "the untampered plan");
    for (auto& s : p.slices) {
        if (s.tensor != "blk.1.ffn_down_exps.weight" || s.rank != 1) continue;
        s.range_start = 288;                      // 288 % 64 != 0
        s.local_shape[0] = s.range_end - s.range_start;
        s.valid_lanes.assign(3, 0);
        s.valid_lanes[1] = s.range_end - s.range_start;
    }
    ValidationResult v = validate_plan(p);
    check(!v.ok, "a cut inside a block is not valid");
    check_has(v.errors, "splits a 64 weight block", "the message names the block being split");
    check_has(v.errors, "gap or overlap", "the gap the move leaves is reported too");
}

void test_missing_scale_and_stride() {
    begin("a sharded slice needs its scales and a whole byte stride");
    ModelInventory inv = fixture();
    for (auto& t : inv.tensors)
        if (t.name == "blk.0.ffn_gate_shexp.weight") t.scale_kind.clear();
    ShardPlan p = make_tp_plan(inv, 3);
    check_eq(p.world, 0, "a block-quantized tensor with no scale reference is refused");
    check_has(p.reject_reasons, "no scale reference", "the reason names the missing scale reference");

    inv = fixture();
    for (auto& t : inv.tensors)
        if (t.name == "blk.0.ffn_down_exps.weight") t.byte_length = -1;
    p = make_tp_plan(inv, 3);
    check_eq(p.world, 0, "an unknown byte length is refused for a sharded tensor");
    check_has(p.reject_reasons, "no whole original byte stride", "the reason names the stride");

    inv = fixture();
    for (auto& t : inv.tensors)
        if (t.name == "blk.0.ffn_down_exps.weight") t.byte_length += 1;  // not a whole number of slabs
    p = make_tp_plan(inv, 3);
    check_eq(p.world, 0, "a byte length that is not a whole stride is refused");
    check_has(p.reject_reasons, "blk.0.ffn_down_exps.weight", "the reason names the tensor");

    // A replicated tensor needs neither: the whole read does not depend on a stride.
    inv = fixture();
    for (auto& t : inv.tensors)
        if (t.name == "blk.0.ssm_out.weight") { t.byte_length = -1; t.scale_kind.clear(); }
    p = make_tp_plan(inv, 3);
    check_eq(p.world, 3, "the replicated fallback still plans with no stride known");
    check_ok(validate_plan(p), "the plan with one stride-less replicated tensor");
}

void test_verdict_and_shapes() {
    begin("an inventory that was not read is not planned");
    ModelInventory inv = fixture();
    inv.verdict.loadable = false;
    ShardPlan p = make_tp_plan(inv, 3);
    check_eq(p.world, 0, "loadable_by_strata false rejects the plan");
    check_has(p.reject_reasons, "loadable_by_strata", "the reason names the verdict");

    inv = fixture();
    inv.tensors[0].shape.clear();
    p = make_tp_plan(inv, 3);
    check_eq(p.world, 0, "an empty shape rejects the plan");
    check_has(p.reject_reasons, "empty shape", "the reason names the empty shape");

    inv = fixture();
    inv.schema_version = 2;
    p = make_tp_plan(inv, 3);
    check_eq(p.world, 0, "an unknown schema rejects the plan");
    check_has(p.reject_reasons, "schema_version", "the reason names the schema version");

    inv = fixture();
    inv.geometry.ssm_v_heads = 47;
    p = make_tp_plan(inv, 3);
    check_eq(p.world, 0, "value heads that are not a whole multiple of key heads reject the plan");
    check_has(p.reject_reasons, "not a whole multiple of key heads",
              "the reason names the pairing the geometry cannot give");
}

// ---------------------------------------------------------------------------
// GQA, key/value pairing, activation contracts
// ---------------------------------------------------------------------------

void test_gqa_map_is_global() {
    begin("the GQA map is the global 24Q/2KV map");
    ModelInventory inv = fixture();
    ShardPlan p = make_tp_plan(inv, 3);
    check_ok(validate_plan(p), "the untampered plan");
    for (auto& s : p.slices) {
        if (s.tensor != "blk.3.attn_q.weight" || s.rank != 1) continue;
        s.gqa_map.assign((size_t)(s.range_end - s.range_start) / 512, 0);   // a local count: all to kv 0
    }
    ValidationResult v = validate_plan(p);
    check(!v.ok, "a local GQA map is not valid");
    check_has(v.errors, "global 24Q/2KV map", "the message names the global map");
    check_has(v.errors, "never derived from local counts", "the message says why");

    inv = fixture();
    p = make_tp_plan(inv, 3);
    for (auto& s : p.slices) {
        if (s.tensor != "blk.3.attn_q.weight" || s.rank != 1) continue;
        s.gqa_map.pop_back();
    }
    v = validate_plan(p);
    check(!v.ok, "a gqa_map that does not cover the range is not valid");
    check_has(v.errors, "gqa_map does not cover the range", "the message names the missing coverage");
}

void test_key_keeps_its_values() {
    begin("every key keeps its three values on the same rank");
    ModelInventory inv = fixture();
    ShardPlan p = make_tp_plan(inv, 3);
    check_ok(validate_plan(p), "the untampered plan");
    for (auto& s : p.slices) {
        if (s.tensor != "blk.0.attn_gate.weight" || s.rank != 1) continue;
        s.head_start = 0;
        s.head_end = 48;                                   // all forty-eight value heads on rank 1
        s.range_start = 0;
        s.range_end = 6144;
        s.local_shape[1] = 6144;
        s.valid_lanes.assign(3, 0);
        s.valid_lanes[1] = 6144;
    }
    ValidationResult v = validate_plan(p);
    check(!v.ok, "a value group that is not its key's is not valid");
    check_has(v.errors, "values per key", "the message names the values-per-key rule");
    check_has(v.errors, "rank 1", "the message names the rank");
    check_has(v.errors, "expected [6,11)", "and the range rank 1 should have held");
}

void test_gdn_state_and_projection_order() {
    begin("GDN conv and per-value parameters use the same compact head order");
    auto inv = fixture();
    inv.tensors.push_back(mk("blk.0.ssm_conv1d.weight", "F32", {4, 10240}, 1, "", 163840, "gdn_key"));
    inv.tensors.push_back(mk("blk.0.ssm_alpha.weight", "BF16", {2560, 48}, 1, "", 245760, "gdn_value"));
    inv.tensors.push_back(mk("blk.0.ssm_dt.bias", "F32", {48}, 1, "", 192, "gdn_value"));
    auto p = make_tp_plan(inv, 3);
    check_ok(validate_plan(p), "GDN state-aware fixture");
    for (int rank = 0; rank < 3; ++rank) {
        std::vector<int64_t> expected;
        for (int64_t h = 0; h < 48; ++h)
            if ((rank == 0 && h % 16 < 6) || (rank == 1 && h % 16 >= 6 && h % 16 < 11) ||
                (rank == 2 && h % 16 >= 11)) expected.push_back(h);
        for (const auto& name : {"blk.0.ssm_conv1d.weight", "blk.0.attn_gate.weight",
                                 "blk.0.ssm_alpha.weight", "blk.0.ssm_dt.bias"}) {
            std::vector<int64_t> actual;
            for (const auto& s : p.slices)
                if (s.tensor == name && s.rank == rank && (s.part == "gdn_v" || s.part == "value_out"))
                    for (int64_t h = s.head_start; h < s.head_end; ++h) actual.push_back(h);
            check(actual == expected, std::string(name) + " follows global value modulo key mapping");
        }
    }
    const auto* output = contract_at(p, "ssm_out");
    check(output && output->gather == "all_gather_padded" &&
          output->layout == "gdn_value_runs_scatter_global" &&
          output->valid_counts == std::vector<int64_t>{2304, 1920, 1920},
          "ssm_out gathers compact runs and scatters them into global input order");
    p.activation.erase(std::remove_if(p.activation.begin(), p.activation.end(),
        [](const auto& c) { return c.consumer == "ssm_out"; }), p.activation.end());
    p.plan_hash = strata::distributed::detail::hex64(strata::distributed::detail::fnv1a64(
        strata::distributed::plan_canonical_body(p)));
    check(!validate_plan(p).ok, "a rehashed plan missing ssm_out gather is rejected");
    p = make_tp_plan(inv, 3);
    for (auto& c : p.activation) if (c.consumer == "ssm_out") {
        c.gather = "none";
        c.layout = "contiguous";
        c.reference = "local";
    }
    p.plan_hash = strata::distributed::detail::hex64(strata::distributed::detail::fnv1a64(
        strata::distributed::plan_canonical_body(p)));
    check(!validate_plan(p).ok, "a rehashed ssm_out contract with local input order is rejected");
}

void test_local_quantizer_that_crosses_is_routed() {
    begin("a local quantizer that crosses a cut goes to the global reference path");
    ModelInventory inv = fixture();
    for (auto& a : inv.activation_contracts)
        if (a.consumer == "down") { a.quantizer = "Q8_K"; a.block_size = 256; }
    ShardPlan p = make_tp_plan(inv, 3);
    check_eq(p.world, 3, "the plan still stands: nothing is requantized, the work moves");
    check_ok(validate_plan(p), "the rerouted plan");
    const ActivationContract* c = contract_at(p, "down");
    check(c != nullptr && c->crosses_rank_boundary, "the 256-block down quantizer crosses the cut at 448");
    check(c != nullptr && c->gather == "all_gather_padded", "it is routed to an all-gather");
    check(c != nullptr && c->reference == "global_reference", "and to the global reference path");
    check_has(p.notes, "routed to the global reference path", "the reroute is recorded in the notes");
    const TensorSlice* r1 = slice_at(p, "blk.0.ffn_down_shexp.weight", "ffn_in", 1);
    const TensorSlice* r2 = slice_at(p, "blk.0.ffn_down_shexp.weight", "ffn_in", 2);
    check(r1 && r1->halo == std::vector<int64_t>({0, 64}),
          "rank 1 keeps the 64 lanes of the block straddling its end");
    check(r2 && r2->halo == std::vector<int64_t>({192, 0}),
          "rank 2 keeps the 192 lanes of the block straddling its start");
    check(component(p, "halo") > 0, "the halos are counted in the budget");
    check(component(p, "activation_gather") > 0, "the gather buffers are counted in the budget");
    // expert_down (a 32-block local quantizer) stays local: 32 divides 256 and 448.
    const ActivationContract* ed = contract_at(p, "expert_down");
    check(ed != nullptr && ed->gather == "none" && ed->reference == "local" && !ed->crosses_rank_boundary,
          "a 32-block local quantizer that divides the cuts stays local");
    const ActivationContract* ao = contract_at(p, "attn_output");
    check(ao != nullptr && ao->gather == "none" && !ao->crosses_rank_boundary,
          "a 256-block quantizer that divides the query cuts stays local");
    const ActivationContract* gu = contract_at(p, "gate_up");
    check(gu != nullptr && gu->gather == "none" && gu->reference == "global_reference",
          "a global quantizer over the whole residual needs no gather of a cut vector");
    const ActivationContract* red = contract_at(p, "tp.down_partial_sum");
    check(red != nullptr && red->gather == "all_reduce_sum" && red->wire_slots == 2560,
          "the partial sums go to one uniform all-reduce over 2560");
    const ActivationContract* emb = contract_at(p, "embed");
    check(emb != nullptr && emb->wire_slots == 248320, "an embedding contract is sized by the vocabulary");
    const ActivationContract* ple = contract_at(p, "ple");
    check(ple != nullptr && ple->gather == "none" && ple->reference == "local",
          "a PLE contract stays a local row gather");
}

void test_halo_bytes_are_the_real_ones() {
    begin("the halo bytes in the budget match the slices that keep them");
    ModelInventory inv = fixture();
    for (auto& a : inv.activation_contracts)
        if (a.consumer == "down") { a.quantizer = "Q8_K"; a.block_size = 256; }
    ShardPlan p = make_tp_plan(inv, 3);
    int64_t want = 0;
    for (const auto& s : p.slices) {
        if (s.halo.empty()) continue;
        for (const auto& t : inv.tensors) {
            if (t.name != s.tensor) continue;
            want += (s.halo[0] + s.halo[1]) * (t.byte_length / t.shape[(size_t)s.axis]);
        }
    }
    check(want > 0, "the fixture does keep halos");
    check_eq(component(p, "halo"), want, "the halo component is (left + right) x bytes per lane");
    check_eq(p.memory.halo_and_global_buffers,
             component(p, "halo") + component(p, "activation_gather") + component(p, "comm_workspace"),
             "halo_and_global_buffers is halo + gather + workspace");
}

// ---------------------------------------------------------------------------
// Memory
// ---------------------------------------------------------------------------

void test_memory_numbers() {
    begin("the budget names its parts and adds up");
    ModelInventory inv = fixture();
    ShardPlan p = make_tp_plan(inv, 3);
    check_ok(validate_plan(p), "the plan's own budget");
    check_eq((int64_t)p.memory.per_rank_total.size(), (int64_t)3, "one total per rank");
    check(p.memory.per_rank_owned[0] > p.memory.per_rank_owned[1] &&
              p.memory.per_rank_owned[1] == p.memory.per_rank_owned[2],
          "owned weights are unequal across ranks because the FFN split is unequal");
    int64_t shared = 0, owned = 0;
    for (const auto& m : p.memory.components) {
        if (m.first.rfind("weights_owned_", 0) == 0) owned += m.second;
        else shared += m.second;
    }
    check(shared > 0 && owned > 0, "both shared and owned parts are present");
    check_eq(p.memory.per_rank_total[0], p.memory.per_rank_owned[0] + shared,
             "rank 0's total is its owned part plus the shared part");
    check_eq(p.memory.per_rank_bytes, p.memory.per_rank_total[0], "per_rank_bytes is the heaviest rank");
    check_eq(p.memory.gate_limit_bytes, kSparkGateBytes, "the gate defaults to one Spark's usable budget");
    check(component(p, "kv_gdn_indexer_state") > 0, "the state component is counted");
    check_eq(component(p, "comm_workspace"), (int64_t)(64LL * 1024 * 1024),
             "the communication workspace is counted");
    check_has(p.notes, "ffn owned per rank: rank0", "the plan states the FFN it owns per rank");
}

void test_memory_gate_fails_with_numbers() {
    begin("a rank that does not fit fails with its numbers");
    ModelInventory inv = fixture();
    const int64_t tight = 20LL * 1024 * 1024 * 1024;   // below the replicated PLE plus everything else
    ShardPlan p = make_tp_plan(inv, 3, tight);
    check_eq(p.world, 3, "the layout still stands; only the budget is over the line");
    ValidationResult v = validate_plan(p);
    check(!v.ok, "the plan fails the memory gate");
    check_has(v.errors, "memory gate: rank 0 needs", "the message names the rank and what it needs");
    check_has(v.errors, std::to_string(p.memory.per_rank_total[0]) + " B",
              "the message gives the exact bytes rank 0 needs");
    check_has(v.errors, std::to_string(tight) + " B", "the message gives the exact gate");
    check_has(v.errors, "replicated PLE is 28800138240 B", "the message names the PLE as the reason");
    check_has(v.errors, "row ownership", "the message names the design's alternative");
    check_has(p.notes, "memory gate FAILED for rank 0", "the note repeats the failing numbers in the plan");
    check_eq(p.memory.gate_limit_bytes, tight, "the plan carries the gate it was judged on");
}

void test_memory_tampering() {
    begin("a budget that does not add up is refused");
    ModelInventory inv = fixture();
    ShardPlan p = make_tp_plan(inv, 3);
    p.memory.per_rank_owned[0] += 1;
    ValidationResult v = validate_plan(p);
    check(!v.ok, "owned detail that disagrees with the components is refused");
    check_has(v.errors, "!= its components", "the message names the mismatch");

    p = make_tp_plan(inv, 3);
    p.memory.per_rank_total[1] += 4096;
    v = validate_plan(p);
    check(!v.ok, "a total that is not owned plus shared is refused");
    check_has(v.errors, "rank 1 total", "the message names the rank");

    p = make_tp_plan(inv, 3);
    p.memory.per_rank_bytes += 1;
    v = validate_plan(p);
    check(!v.ok, "per_rank_bytes that is not the heaviest rank is refused");
    check_has(v.errors, "is not the heaviest rank", "the message names the heaviest-rank rule");

    p = make_tp_plan(inv, 3);
    for (auto& m : p.memory.components)
        if (m.first == "halo") m.second += 1;
    v = validate_plan(p);
    check(!v.ok, "a halo component that disagrees with the slices is refused");
    check_has(v.errors, "halo + gather + workspace", "the message names the sum");
}

/// Recompute the plan hash after a deliberate, self-consistent change, so the hash is not what fails.
void refreeze(ShardPlan& p) {
    p.plan_hash = strata::distributed::detail::hex64(
        strata::distributed::detail::fnv1a64(plan_canonical_body(p)));
}

void test_buffers_and_reduce() {
    begin("collectives are sized and uniform");
    ModelInventory inv = fixture();
    for (auto& a : inv.activation_contracts)
        if (a.consumer == "down") { a.quantizer = "Q8_K"; a.block_size = 256; }
    ShardPlan p = make_tp_plan(inv, 3);
    for (auto& c : p.activation)
        if (c.consumer == "down") c.wire_slots = 100;      // below the largest valid count
    ValidationResult v = validate_plan(p);
    check(!v.ok, "padded slots too small for the valid counts are refused");
    check_has(v.errors, "padded wire slots do not hold the valid counts", "the message names the sizing");

    ShardPlan q = make_tp_plan(inv, 3);
    for (auto& c : q.activation)
        if (c.consumer == "tp.down_partial_sum") c.valid_counts[2] = 1280;
    v = validate_plan(q);
    check(!v.ok, "unequal counts into a uniform all-reduce are refused");
    check_has(v.errors, "unequal valid counts", "the message names the uniform-collective rule");

    ShardPlan r = make_tp_plan(inv, 3);
    for (auto& s : r.slices)
        if (!s.halo.empty()) s.halo.clear();
    for (auto& c : r.activation)
        if (c.consumer == "down") c.crosses_rank_boundary = false;
    refreeze(r);
    v = validate_plan(r);
    check(v.ok, "a plan with no halos and nothing crossing is self-consistent");
}

// ---------------------------------------------------------------------------
// The plan hash, and the manifest Task 3 reads
// ---------------------------------------------------------------------------

void test_hash_freezes_the_plan() {
    begin("the plan hash freezes the numbers");
    ModelInventory inv = fixture();
    ShardPlan a = make_tp_plan(inv, 3);
    ShardPlan b = make_tp_plan(inv, 3);
    check(!a.plan_hash.empty() && a.plan_hash == b.plan_hash, "one inventory and world give one hash");
    check_eq(a.slices.size(), b.slices.size(), "and the same slices");
    b.slices.front().range_end += 1;
    ValidationResult v = validate_plan(b);
    check(!v.ok, "a moved range breaks the plan");
    check_has(v.errors, "does not match the plan body", "and the hash says so");

    ModelInventory inv2 = fixture();
    inv2.source.revision = "another-revision";
    ShardPlan c = make_tp_plan(inv2, 3);
    check(c.plan_hash != a.plan_hash, "a different source lock gives a different hash");

    ModelInventory inv3 = fixture();
    ShardPlan d = make_tp_plan(inv3, 3, kSparkGateBytes - 1);
    check(d.plan_hash != a.plan_hash, "a different gate gives a different hash");
}

/// The transport needs explicit per-rank counts: a uniform padded all-gather is exactly where an empty or
/// ambiguous list goes wrong.  Checked on the plan and again on the serialized manifest, because that text is
/// what the packer and the runtime read.
void test_valid_lanes_are_one_count_per_rank() {
    begin("valid_lanes is one explicit count per rank");
    for (int world : {1, 2, 3}) {
        ModelInventory inv = fixture();
        ShardPlan p = make_tp_plan(inv, world);
        check_ok(validate_plan(p), "the world " + std::to_string(world) + " plan");
        for (const auto& s : p.slices) {
            const std::string tag = s.tensor + " rank" + std::to_string(s.rank) + " (world " +
                                    std::to_string(world) + ")";
            const int64_t span = s.range_end - s.range_start;
            check_eq((int64_t)s.valid_lanes.size(), (int64_t)world, tag + ": one entry per rank");
            if ((int64_t)s.valid_lanes.size() != world) continue;
            for (int r = 0; r < world; ++r) {
                const int64_t c = s.valid_lanes[(size_t)r];
                check(c >= 0 && c <= span, tag + ": rank " + std::to_string(r) + " count " +
                          std::to_string(c) + " outside a range of " + std::to_string(span));
                check(s.replicated ? c == span : (r == s.rank ? c == span : c == 0),
                      tag + ": rank " + std::to_string(r) + " count is not its own share of this slice");
            }
        }
        // The same facts out of the serialized manifest, read back with the plan reader.
        const std::string js = to_tp_plan_json(p);
        std::string err;
        strata::distributed::jsonlite::JReader rd(js, &err);
        strata::distributed::jsonlite::JVal root;
        const bool parsed = rd.parse(root) && root.k == strata::distributed::jsonlite::JVal::K::Obj;
        check(parsed, "the manifest parses back (err: " + err + ")");
        if (!parsed) continue;
        const strata::distributed::jsonlite::JVal* sl = root.at("slices");
        check(sl != nullptr && sl->k == strata::distributed::jsonlite::JVal::K::Arr,
              "the manifest has a slices array");
        if (sl == nullptr || sl->k != strata::distributed::jsonlite::JVal::K::Arr) continue;
        check_eq((int64_t)sl->arr.size(), (int64_t)p.slices.size(), "and one entry per slice");
        for (const auto& v : sl->arr) {
            const strata::distributed::jsonlite::JVal* lanes = v.at("valid_lanes");
            check(lanes != nullptr, "slice " + std::string(v.at("tensor") &&
                                    v.at("tensor")->k == strata::distributed::jsonlite::JVal::K::Str
                                        ? v.at("tensor")->s : "?") + ": valid_lanes is present");
            if (lanes == nullptr) continue;
            check(lanes->k == strata::distributed::jsonlite::JVal::K::Arr,
                  "a slice's valid_lanes is a list of counts, not a bare number");
            if (lanes->k != strata::distributed::jsonlite::JVal::K::Arr) continue;
            check_eq((int64_t)lanes->arr.size(), (int64_t)world, "a serialized slice has one count per rank");
            const strata::distributed::jsonlite::JVal* rng = v.at("range");
            const strata::distributed::jsonlite::JVal* local = v.at("local_shape");
            const strata::distributed::jsonlite::JVal* axis = v.at("axis");
            if (rng == nullptr || local == nullptr || axis == nullptr || rng->arr.size() != 2 ||
                local->arr.empty()) {
                check(false, "a serialized slice carries range, local_shape and axis");
                continue;
            }
            const int64_t ax = (int64_t)axis->num;
            const int64_t span = (int64_t)rng->arr[1].num - (int64_t)rng->arr[0].num;
            check_eq((int64_t)local->arr[(size_t)ax].num, span, "local width on the cut axis is the range");
            for (const auto& c : lanes->arr)
                check(c.k == strata::distributed::jsonlite::JVal::K::Num && c.num >= 0 &&
                          c.num <= (double)span,
                      "a serialized count " + std::to_string((int64_t)c.num) + " is outside a range of " +
                          std::to_string(span) + " items");
        }
    }
    // A count that is not the owner's share, or not a whole cut unit, is refused.
    ModelInventory inv = fixture();
    ShardPlan p = make_tp_plan(inv, 3);
    for (auto& s : p.slices) {
        if (s.tensor != "blk.0.attn_qkv.weight" || s.part != "gdn_v" || s.rank != 0) continue;
        s.valid_lanes[0] = 700;                       // not a whole number of 128-item cut units
    }
    ValidationResult v = validate_plan(p);
    check(!v.ok, "a partial count that is not whole cut units is refused");
    check_has(v.errors, "not a whole number of 128-item cut units", "and the message names the cut unit");

    p = make_tp_plan(inv, 3);
    for (auto& s : p.slices) {
        if (s.tensor != "blk.0.ffn_down_shexp.weight" || s.rank != 1) continue;
        s.valid_lanes[2] = 128;                        // inside the span, but rank 2 owns none of it
    }
    v = validate_plan(p);
    check(!v.ok, "a count for a rank that does not own the slice is refused");
    check_has(v.errors, "a range it does not own", "and the message says who holds what");

    p = make_tp_plan(inv, 3);
    for (auto& s : p.slices) {
        if (s.tensor != "blk.0.ffn_down_shexp.weight" || s.rank != 1) continue;
        s.valid_lanes.pop_back();
    }
    v = validate_plan(p);
    check(!v.ok, "a slice with no count for every rank is refused");
    check_has(v.errors, "one explicit count per rank", "and the message names the rule");
}

void test_scale_tensors_are_named_by_their_tensors() {
    begin("a scale tensor in the inventory is named by the slices that need it");
    ModelInventory inv = fixture();
    InventoryTensor sc;
    sc.name = "blk.0.ffn_down_shexp_scales";
    sc.file = "fixture-00001-of-00002.gguf";
    sc.dtype = "F16";
    sc.shape = {20, 2560};                      // one fp16 scale per 32-block of the 640 rows
    sc.block_size = 1;
    sc.byte_offset = 1024;
    sc.byte_length = 20 * 2560 * 2;
    sc.family = TensorFamily::Other;
    inv.tensors.push_back(sc);
    ShardPlan p = make_tp_plan(inv, 3);
    check_ok(validate_plan(p), "the plan with a scale tensor of its own");
    const std::string js = to_tp_plan_json(p);
    check(js.find("\"scale_ref\": \"blk.0.ffn_down_shexp_scales\"") != std::string::npos,
          "the slices of the block-quantized tensor name the inventory's scale tensor");
    check(js.find("\"scale_ref\": null") != std::string::npos,
          "and the tensors whose scales ride in the block name nothing");
    int64_t named = 0;
    for (const auto& s : p.slices)
        if (s.scale_ref == "blk.0.ffn_down_shexp_scales") ++named;
    check_eq(named, (int64_t)3, "one named scale_ref per rank of that tensor, and no others");
    for (const auto& s : p.slices)
        if (s.tensor == "blk.0.ffn_down_shexp.weight")
            check(s.scale_ref == "blk.0.ffn_down_shexp_scales" && !s.scale_kind.empty(),
                  "the slice carries both: where the scales are and how they ride");
}

void test_manifest_for_task3() {
    begin("every manifest slice names its rank or says it is replicated");
    ModelInventory inv = fixture();
    ShardPlan p = make_tp_plan(inv, 3);
    check_ok(validate_plan(p), "the plan behind the manifest");
    const std::string js = to_tp_plan_json(p);
    auto count = [&js](const std::string& needle) {
        int64_t n = 0;
        for (size_t pos = js.find(needle); pos != std::string::npos; pos = js.find(needle, pos + 1)) ++n;
        return n;
    };
    // Task 3 reads one entry per slice and needs, on each, either "rank" or "replicated".  Both are always
    // emitted, so a shard never looks like a whole copy and a whole copy never looks like a shard.
    check_eq(count("\"rank\":"), (int64_t)p.slices.size(), "one \"rank\" per slice in tp-plan.json");
    check_eq(count("\"replicated\":"), (int64_t)p.slices.size(),
             "one \"replicated\" per slice in tp-plan.json");
    check_eq(count("\"byte_row_stride\":"), (int64_t)p.slices.size(), "one stride per slice");
    check_eq(count("\"range\":"), (int64_t)p.slices.size(), "one range per slice");
    check_eq(count("\"scale_ref\":"), (int64_t)p.slices.size(), "one scale_ref per slice");
    check_eq(count("\"scale_kind\":"), (int64_t)p.slices.size(), "one scale_kind per slice");
    // A scale_ref that is not empty must be a tensor the reader can open.  This inventory keeps no scales in
    // a tensor of their own, so every slice says so by naming none.
    for (const auto& s : p.slices) {
        check(s.scale_ref.empty(), "slice " + s.tensor + " names no scale tensor the inventory lacks");
        check(s.block_size <= 1 || !s.scale_kind.empty(),
              "block-quantized slice " + s.tensor + " states how its scales ride");
    }
    check_eq(count("\"valid_lanes\":"), (int64_t)p.slices.size(), "one valid_lanes per slice");
    bool has_rank = true, flagged = true, in_order = true;
    size_t from = 0;
    for (const auto& s : p.slices) {
        const size_t at = js.find("{\"tensor\": ", from);
        if (at == std::string::npos) { has_rank = false; flagged = false; break; }
        const size_t nl = js.find('\n', at);
        const std::string line = js.substr(at, nl == std::string::npos ? std::string::npos : nl - at);
        from = (nl == std::string::npos ? js.size() : nl + 1);
        if (line.find("\"tensor\": \"" + s.tensor + "\"") == std::string::npos) in_order = false;
        if (line.find("\"rank\": " + std::to_string(s.rank)) == std::string::npos) has_rank = false;
        if (line.find(s.replicated ? "\"replicated\": true" : "\"replicated\": false") == std::string::npos)
            flagged = false;
    }
    check(in_order, "the slices come out in inventory order, ranks in increasing order");
    check(has_rank, "each slice entry carries its own rank");
    check(flagged, "each slice entry says whether it is a whole copy");
    check(js.find("\"plan_hash\"") != std::string::npos, "the manifest carries the plan hash");
    check(js.find("\"source_lock\"") != std::string::npos, "the manifest carries the source lock");
    check(js.find("\"gate_limit_bytes\"") != std::string::npos, "the manifest carries the gate it judged on");
    check(js.find("\"per_rank_total\"") != std::string::npos, "the manifest carries the per-rank totals");
    check(js.find("28800138240") != std::string::npos, "the manifest states the replicated PLE in bytes");
    const ShardPlan rej = make_tp_plan(inv, 4);
    const std::string rjs = to_tp_plan_json(rej);
    check(rjs.find("\"reject_reasons\"") != std::string::npos, "a rejected plan is emitted with its reasons");
}

// ---------------------------------------------------------------------------
// The inventory.json reader
// ---------------------------------------------------------------------------

const char* kMinimalInventory =
    "{\n"
    "  \"schema_version\": 1,\n"
    "  \"source\": {\"kind\": \"hf\", \"repo\": \"ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF\",\n"
    "               \"revision\": \"ed59f92082b1e93c0e96d60a8b11aab089b52f09\",\n"
    "               \"files\": [{\"path\": \"a.gguf\", \"size\": 10, \"sha256\": \"aa\"}]},\n"
    "  \"geometry\": {\"n_embd\": 2560, \"n_layers\": 48, \"qsa_interval\": 4, \"n_vocab\": 248320,\n"
    "                 \"rms_eps\": 1e-6, \"ssm_state_size\": 128, \"ssm_k_heads\": 16, \"ssm_v_heads\": 48,\n"
    "                 \"ssm_d_conv\": 4, \"ssm_conv_channels\": 10240, \"ssm_value_dim\": 6144,\n"
    "                 \"n_head\": 24, \"n_head_kv\": 2, \"head_dim\": 256, \"n_rot\": 64, \"idx_q_heads\": 4,\n"
    "                 \"idx_key_dim\": 128, \"idx_block\": 4, \"idx_top_k\": 2048, \"hc\": 4, \"hc_lr\": 320,\n"
    "                 \"n_expert\": 512, \"n_expert_used\": 10, \"n_ff\": 640, \"n_ff_shexp\": 640,\n"
    "                 \"family\": \"qwen4exp\", \"n_mtp_layers\": 0},\n"
    "  \"tensors\": [\n"
    "    {\"name\": \"blk.0.ffn_down_shexp.weight\", \"file\": \"a.gguf\", \"dtype\": \"IQ4_NL\",\n"
    "     \"shape\": [640, 2560], \"block_size\": 32, \"scale_kind\": \"fp16-scale-plus-codebook-codes\",\n"
    "     \"byte_offset\": 1024, \"byte_length\": 921600, \"family\": \"down\"}\n"
    "  ],\n"
    "  \"activation_contracts\": [{\"consumer\": \"down\", \"quantizer\": \"Q8_0\", \"block_size\": 32,\n"
    "                              \"scope\": \"local\"}],\n"
    "  \"verdict\": {\"loadable_by_strata\": true, \"reasons\": [], \"checked_tensors\": 1}\n"
    "}\n";

void test_reader_accepts_a_real_shape() {
    begin("the reader reads an inventory.json without assuming keys");
    ModelInventory inv;
    std::string err;
    const bool ok = parse_inventory_json(kMinimalInventory, inv, err);
    check(ok, "a full inventory parses (err: " + err + ")");
    check_eq(inv.schema_version, 1, "schema_version");
    check_eq(inv.source.repo, std::string("ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF"), "source.repo");
    check_eq(inv.source.revision, std::string("ed59f92082b1e93c0e96d60a8b11aab089b52f09"), "source.revision");
    check_eq(inv.geometry.n_ff, (int64_t)640, "geometry.n_ff");
    check_eq(inv.geometry.ssm_conv_channels, (int64_t)10240, "geometry.ssm_conv_channels");
    check(inv.geometry.family == strata::ModelFamily::Qwen4Exp, "geometry.family");
    check_eq((int64_t)inv.tensors.size(), (int64_t)1, "the tensor list");
    check(inv.tensors[0].family == TensorFamily::Down, "the tensor family");
    check_eq(inv.tensors[0].byte_length, (int64_t)921600, "byte_length");
    check(inv.tensors[0].scale_kind == "fp16-scale-plus-codebook-codes", "scale_kind");
    check(inv.verdict.loadable, "verdict.loadable_by_strata");
    check_eq((int64_t)inv.activation_contracts.size(), (int64_t)1, "the activation contracts");
    check_eq(inv.activation_contracts[0].scope, std::string("local"), "a contract's scope");
    // The same text through the planner.  The only FFN block it can see is 32, so the cut snaps to 224/448:
    // the cut comes from the inventory, not from a number remembered by the planner.
    ShardPlan p = make_tp_plan(inv, 3);
    check_eq(p.world, 3, "the parsed inventory plans");
    check_ok(validate_plan(p), "the plan from parsed JSON");
    const TensorSlice* r1 = slice_at(p, "blk.0.ffn_down_shexp.weight", "ffn_in", 1);
    check(r1 && r1->range_start == 224 && r1->range_end == 448,
          "and gives the block-aligned FFN ranges this inventory implies");
    check(r1 && r1->byte_row_stride == 360, "with the stride the file's own byte_length gives");
}

/// An inventory with every geometry key except the ones named in `drop`, as JSON text.
std::string geometry_missing(const std::string& drop) {
    std::string s = "{\"schema_version\":1,\"source\":{},\"geometry\":{";
    const char* keys[] = {"n_embd", "n_layers", "qsa_interval", "n_vocab", "ssm_state_size", "ssm_k_heads",
                          "ssm_v_heads", "ssm_d_conv", "ssm_conv_channels", "ssm_value_dim", "n_head",
                          "n_head_kv", "head_dim", "n_rot", "idx_q_heads", "idx_key_dim", "idx_block",
                          "idx_top_k", "hc", "hc_lr", "n_expert", "n_expert_used", "n_ff", "n_ff_shexp"};
    bool first = true;
    for (const char* k : keys) {
        if (drop == k) continue;
        if (!first) s += ",";
        first = false;
        s += std::string("\"") + k + "\": 1";
    }
    if (drop != "family") {
        if (!first) s += ",";
        s += "\"family\":\"qwen4exp\"";
    }
    s += "}";
    return s;
}

void test_reader_reports_what_is_missing() {
    begin("the reader names the key it needed");
    ModelInventory inv;
    std::string err;

    inv = ModelInventory();
    err.clear();
    check(!parse_inventory_json("{\"schema_version\": 1, \"source\": {}, \"geometry\": "
                               "{\"n_embd\": 2560}}", inv, err),
          "a geometry with one key missing does not parse");
    check(err.rfind("missing key 'geometry.", 0) == 0, "the error is a missing geometry key (got: " + err + ")");

    std::string full(kMinimalInventory);
    const std::string key = "\"n_ff\": 640, ";
    const size_t at = full.find(key);
    check(at != std::string::npos, "the fixture does carry n_ff");
    full.erase(at, key.size());
    inv = ModelInventory();
    err.clear();
    check(!parse_inventory_json(full, inv, err), "dropping n_ff fails");
    check_eq(err, std::string("missing key 'geometry.n_ff'"), "and names geometry.n_ff");

    inv = ModelInventory();
    err.clear();
    check(!parse_inventory_json("{\"schema_version\": 1,", inv, err), "truncated JSON does not parse");
    check(err.rfind("json:", 0) == 0, "the error starts with the reader's prefix (got: " + err + ")");
    check(err.find(" at offset ") != std::string::npos, "and gives the offset where it stopped");

    inv = ModelInventory();
    err.clear();
    check(!parse_inventory_json(geometry_missing("family") + ",\"tensors\":[],\"verdict\":{}"
                                "}", inv, err),
          "a geometry with no family does not parse");
    check_eq(err, std::string("missing key 'geometry.family'"), "and names geometry.family");

    inv = ModelInventory();
    err.clear();
    check(!parse_inventory_json(geometry_missing("n_ff") + ",\"tensors\":[],\"verdict\":{}}", inv, err),
          "a geometry with no n_ff does not parse");
    check_eq(err, std::string("missing key 'geometry.n_ff'"), "and names geometry.n_ff");

    inv = ModelInventory();
    err.clear();
    check(!parse_inventory_json(geometry_missing("") + ",\"tensors\":[{\"name\":\"a\",\"dtype\":\"F32\","
                                "\"shape\":\"wide\",\"family\":\"down\"}],\"verdict\":"
                                "{\"loadable_by_strata\":true}}", inv, err),
          "a shape that is not an array does not parse");
    check(err.rfind("tensors[0].shape", 0) == 0, "the error names tensors[0].shape (got: " + err + ")");

    inv = ModelInventory();
    err.clear();
    check(!parse_inventory_json(geometry_missing("") + ",\"tensors\":[{\"name\":\"a\",\"dtype\":\"F32\","
                                "\"shape\":[4]}],\"verdict\":{\"loadable_by_strata\":true}}", inv, err),
          "a tensor with no family does not parse");
    check_eq(err, std::string("missing key 'tensors[0].family'"), "and names tensors[0].family");

    inv = ModelInventory();
    err.clear();
    check(!parse_inventory_json(geometry_missing("") + ",\"tensors\":[]}", inv, err),
          "an inventory with no verdict does not parse");
    check_eq(err, std::string("missing key 'verdict.loadable_by_strata'"), "and names the verdict key");

    inv = ModelInventory();
    err.clear();
    check(!parse_inventory_json(geometry_missing("") + ",\"tensors\":[],\"verdict\":"
                                "{\"loadable_by_strata\":\"yes\"}}", inv, err),
          "a verdict that is not a bool does not parse");
    check(err.find("verdict.loadable_by_strata") != std::string::npos,
          "and names the key (got: " + err + ")");
}

// ---------------------------------------------------------------------------
// The CLI over a real inventory.json
// ---------------------------------------------------------------------------

std::string gib(int64_t bytes) {
    std::ostringstream os;
    os << std::fixed;
    os.precision(2);
    os << (double)bytes / (1024.0 * 1024.0 * 1024.0);
    return os.str();
}

int run_inventory(const std::string& path, int world, const std::string& out, int64_t gate) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::cerr << "inventory.json: cannot open " << path << "\n";
        return 2;
    }
    std::ostringstream buf;
    buf << in.rdbuf();
    const std::string text = buf.str();

    ModelInventory inv;
    std::string err;
    if (!parse_inventory_json(text, inv, err)) {
        std::cerr << "inventory.json: " << err << "\n";
        return 2;
    }
    const int64_t gate_bytes = gate > 0 ? gate : kSparkGateBytes;
    ShardPlan p = make_tp_plan(inv, world, gate_bytes);
    ValidationResult v = validate_plan(p);

    std::vector<std::string> names;
    for (const auto& s : p.slices)
        if (std::find(names.begin(), names.end(), s.tensor) == names.end()) names.push_back(s.tensor);

    std::cout << "inventory: " << path << "\n";
    std::cout << "source: " << inv.source.kind << " " << inv.source.repo
              << (inv.source.revision.empty() ? " (no revision)" : " " + inv.source.revision) << "\n";
    std::cout << "tensors in the inventory: " << inv.tensors.size()
              << "; verdict loadable_by_strata = " << (inv.verdict.loadable ? "true" : "false") << "\n";
    std::cout << "world: " << p.world << "; plan hash: " << p.plan_hash << "\n";
    std::cout << "slices: " << p.slices.size() << " over " << names.size() << " tensors\n";
    if (!p.reject_reasons.empty()) {
        std::cout << "rejected:\n";
        for (const auto& r : p.reject_reasons) std::cout << "  - " << r << "\n";
    }
    for (const auto& m : p.memory.components)
        std::cout << "memory " << m.first << " = " << m.second << " B (" << gib(m.second) << " GiB)\n";
    for (size_t r = 0; r < p.memory.per_rank_total.size(); ++r)
        std::cout << "rank " << r << " total " << p.memory.per_rank_total[r] << " B ("
                  << gib(p.memory.per_rank_total[r]) << " GiB)\n";
    std::cout << "gate: " << p.memory.gate_limit_bytes << " B (" << gib(p.memory.gate_limit_bytes)
              << " GiB) lower-bound check -> " << (v.ok ? "fits estimate" : "FAIL") << "\n";
    std::cout << "capacity 8K/64K/C8: UNKNOWN (unqualified lower bound)\n";
    int64_t halo_slices = 0, replicated = 0;
    for (const auto& s : p.slices) {
        if (!s.halo.empty()) ++halo_slices;
        if (s.replicated) ++replicated;
    }
    std::cout << "slices keeping halos: " << halo_slices << "\n";
    std::cout << "replicated slices: " << replicated << "; sharded slices: "
              << ((int64_t)p.slices.size() - replicated) << "\n";
    if (!p.notes.empty()) {
        std::cout << "notes (" << p.notes.size() << "):\n";
        for (const auto& n : p.notes) std::cout << "  - " << n << "\n";
    }
    if (v.ok) {
        std::cout << "validation: ok\n";
    } else {
        std::cout << "validation FAILED (" << v.errors.size() << "):\n";
        for (const auto& e : v.errors) std::cout << "  - " << e << "\n";
    }
    if (!out.empty()) {
        std::ofstream os(out, std::ios::binary);
        if (!os) {
            std::cerr << "inventory.json: cannot write " << out << "\n";
            return 2;
        }
        os << to_tp_plan_json(p);
        std::cout << "wrote " << out << "\n";
    }
    return v.ok ? 0 : 1;
}

void run_fixture_suite() {
    test_world1_identity();
    test_world3_layout();
    test_world2_control();
    test_worlds_outside_the_promise();
    test_ffn_block_reject();
    test_validator_block_gap();
    test_missing_scale_and_stride();
    test_verdict_and_shapes();
    test_valid_lanes_are_one_count_per_rank();
    test_scale_tensors_are_named_by_their_tensors();
    test_gqa_map_is_global();
    test_key_keeps_its_values();
    test_gdn_state_and_projection_order();
    test_local_quantizer_that_crosses_is_routed();
    test_halo_bytes_are_the_real_ones();
    test_memory_numbers();
    test_memory_gate_fails_with_numbers();
    test_memory_tampering();
    test_buffers_and_reduce();
    test_hash_freezes_the_plan();
    test_manifest_for_task3();
    test_reader_accepts_a_real_shape();
    test_reader_reports_what_is_missing();
}

}  // namespace

int main(int argc, char** argv) {
    std::string inv_path, out_path;
    int world = -1;
    int64_t gate = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::cerr << "missing value for " << name << "\n";
                return nullptr;
            }
            return argv[++i];
        };
        if (a == "--inventory") {
            const char* v = next("--inventory");
            if (!v) return 2;
            inv_path = v;
        } else if (a == "--out") {
            const char* v = next("--out");
            if (!v) return 2;
            out_path = v;
        } else if (a == "--world") {
            const char* v = next("--world");
            if (!v) return 2;
            world = std::atoi(v);
        } else if (a == "--gate-bytes") {
            const char* v = next("--gate-bytes");
            if (!v) return 2;
            gate = std::atoll(v);
        } else if (a == "--help" || a == "-h") {
            std::cout << "fixture suite:      shard_plan_test\n"
                      << "plan a real pack:   shard_plan_test --inventory inventory.json --world 3 "
                         "--out tp-plan.json [--gate-bytes N]\n";
            return 0;
        } else {
            std::cerr << "unknown argument: " << a << "\n";
            return 2;
        }
    }
    if (!inv_path.empty()) {
        if (world < 1) {
            std::cerr << "usage: --inventory needs --world (1, 2 or 3)\n";
            return 2;
        }
        return run_inventory(inv_path, world, out_path, gate);
    }
    if (world >= 1) {
        ModelInventory inv = fixture();
        ShardPlan p = make_tp_plan(inv, world, gate > 0 ? gate : kSparkGateBytes);
        std::cout << to_tp_plan_json(p);
        return validate_plan(p).ok ? 0 : 1;
    }
    run_fixture_suite();
    std::cout << "\nchecks: " << (g_checks - g_failed) << " passed, " << g_failed << " failed\n";
    return g_failed == 0 ? 0 : 1;
}
