// include/strata/distributed/shard_plan.hpp - who owns which rows and columns of every tensor when one Qwen
// model runs across ranks, and where quantization is allowed to happen.
//
// The planner reads a ModelInventory (the C++ shape of the shared inventory.json schema; the
// parse_inventory_json reader takes a real inventory.json) and returns the ownership map: per-rank ranges
// with the original shapes, the original byte stride, the codec and block size that must not be broken,
// scale references, valid lanes and halos.  It never allocates GPU memory and never requantizes: a weight
// block that cannot be cut whole rejects the FFN layout or falls the tensor back to a whole-rank read (the
// reference path), and an activation quantizer that cannot stay local is routed to the global reference
// path, with a note.
//
// Shapes follow the inventory, which follows GGUF: axis 0 is the contiguous axis, quantization blocks and
// the projection's INPUT run along it; outputs are the later axes.  A cut on axis 0 therefore has to land
// on block boundaries; a cut on a later axis copies whole input rows and keeps every block intact.
//
// Cuts are derived from the inventory's own geometry, never assumed.  With the Flash-Next numbers they come
// out as the agreed layout: gate/up output columns 256/192/192 (starts 0/256/448), down input columns the
// same (block-aligned), packed GDN attn_qkv [q|k|v] split at the head boundaries the geometry gives (key
// 6/5/5 starts 0/6/11, value 18/15/15 starts 0/18/33 - each key keeps its three values), the GDN z-gate
// and ssm_out on the value heads, attn_q in whole query heads (q and its gate stay together), QSA
// attn_output at the query boundaries, QSA KV and indexer replicated, router/norm/HC/PLE/embedding/head
// replicated.  A tensor whose block layout forbids the cut is read whole by every rank, with a note: that
// is the reference path, not a requantization.  World 1 is identity; world 2 is a supported control.
// Other worlds are rejected, not promised.
//
// The per-rank budget names its parts (FFN owned per rank, replicated PLE, vocabulary, state, halos and
// global buffers) and checks a memory gate: a rank that needs more than the limit fails validation with
// the concrete numbers, which is the design's cue for PLE/embedding row ownership.
//
// CPU-only, header-only, ASCII.  Nothing here touches CUDA or NCCL.
#pragma once

#include "strata/core/geometry.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

namespace strata::distributed {

/// The tensor families of the shared inventory schema (inventory.json v1).
enum class TensorFamily {
    GdnKey, GdnValue, QsaQ, QsaKv, QsaIdx,
    GateUp, Down, ExpertGateUp, ExpertDown,
    Hc, Norm, Router, Ple, Embed, Head, Other,
};

inline TensorFamily family_from_string(const std::string& s) {
    if (s == "gdn_key") return TensorFamily::GdnKey;
    if (s == "gdn_value") return TensorFamily::GdnValue;
    if (s == "qsa_q") return TensorFamily::QsaQ;
    if (s == "qsa_kv") return TensorFamily::QsaKv;
    if (s == "qsa_idx") return TensorFamily::QsaIdx;
    if (s == "gate_up") return TensorFamily::GateUp;
    if (s == "down") return TensorFamily::Down;
    if (s == "expert_gate_up") return TensorFamily::ExpertGateUp;
    if (s == "expert_down") return TensorFamily::ExpertDown;
    if (s == "hc") return TensorFamily::Hc;
    if (s == "norm") return TensorFamily::Norm;
    if (s == "router") return TensorFamily::Router;
    if (s == "ple") return TensorFamily::Ple;
    if (s == "embed") return TensorFamily::Embed;
    if (s == "head") return TensorFamily::Head;
    return TensorFamily::Other;
}

inline const char* family_to_string(TensorFamily f) {
    switch (f) {
    case TensorFamily::GdnKey: return "gdn_key";
    case TensorFamily::GdnValue: return "gdn_value";
    case TensorFamily::QsaQ: return "qsa_q";
    case TensorFamily::QsaKv: return "qsa_kv";
    case TensorFamily::QsaIdx: return "qsa_idx";
    case TensorFamily::GateUp: return "gate_up";
    case TensorFamily::Down: return "down";
    case TensorFamily::ExpertGateUp: return "expert_gate_up";
    case TensorFamily::ExpertDown: return "expert_down";
    case TensorFamily::Hc: return "hc";
    case TensorFamily::Norm: return "norm";
    case TensorFamily::Router: return "router";
    case TensorFamily::Ple: return "ple";
    case TensorFamily::Embed: return "embed";
    case TensorFamily::Head: return "head";
    case TensorFamily::Other: break;
    }
    return "other";
}

// ---------------------------------------------------------------------------
// Inventory (the C++ shape of inventory.json, schema_version 1)
// ---------------------------------------------------------------------------

struct InventoryFile {
    std::string path;
    int64_t size = 0;
    std::string sha256;
};

struct InventorySource {
    std::string kind;                       // "local" | "hf" | "ms"
    std::string repo;
    std::string revision;                   // empty = unknown (emitted as null)
    std::vector<InventoryFile> files;
};

struct InventoryTensor {
    std::string name;                       // the tensor id
    std::string file;
    std::string dtype;                      // codec name, e.g. "IQ3_S", "Q8_K", "F32"
    std::vector<int64_t> shape;             // GGUF order: axis 0 is contiguous, blocks run along it
    int64_t block_size = 0;                 // 0/1 = not block-quantized
    std::string scale_kind;                 // empty = no scales
    int64_t byte_offset = -1;               // -1 = unknown
    int64_t byte_length = -1;               // -1 = unknown
    TensorFamily family = TensorFamily::Other;
};

/// One activation-consumer record from the inventory: where its quantizer may run and at what block size.
struct InventoryActivation {
    std::string consumer;                   // "down", "expert_down", "attn_output", "gate_up", ...
    std::string quantizer;                  // e.g. "Q8_K", "Q8_0", "none"
    int64_t block_size = 0;                 // activation block width
    std::string scope = "global";           // "global" | "local"
    std::string notes;
};

struct InventoryVerdict {
    bool loadable = false;                  // inventory.json: verdict.loadable_by_strata
    std::vector<std::string> reasons;
    int64_t checked_tensors = 0;
};

struct ModelInventory {
    int schema_version = 1;
    InventorySource source;
    core::ModelGeometry geometry;
    std::vector<InventoryTensor> tensors;
    std::vector<InventoryActivation> activation_contracts;
    std::string inventory_hash;             // the revision-pinned hash the caller froze this inventory under
    InventoryVerdict verdict;
};

// ---------------------------------------------------------------------------
// The plan
// ---------------------------------------------------------------------------

/// Part names carry the unit of head_start/head_end: "gdn_q"/"gdn_k" in key heads, "gdn_v"/"value_out" in
/// value heads, "qsa_head" in query heads, "qsa_out" in query heads (output projection), "ffn_in"/
/// "ffn_out" in lanes.  A packed tensor contributes one slice per part per rank; the ranges of one tensor
/// tile its axis without gaps or overlaps.
struct TensorSlice {
    std::string tensor;
    TensorFamily family = TensorFamily::Other;
    int rank = 0;
    std::string part;                       // "" for whole-tensor (replicated) slices
    std::vector<int64_t> global_shape;
    std::vector<int64_t> local_shape;
    int axis = 0;                           // the axis this rank's range lives on (0 = the block axis)
    int64_t range_start = 0;
    int64_t range_end = 0;                  // range_end - range_start == local_shape[axis]
    int64_t cut_unit = 0;                   // the cut granularity in columns (a head's width, or 1)
    int64_t head_start = 0;                 // first unit owned (units follow `part`)
    int64_t head_end = 0;                   // one past the last unit owned
    int64_t byte_row_stride = -1;           // the ORIGINAL bytes per contiguous slab (GGUF row), for whole
                                            // reads; a sharded slice must carry it
    std::string codec;
    int64_t block_size = 0;
    std::string scale_ref;                  // "" = none; a block-quantized slice must name its scales
    std::vector<int64_t> valid_lanes;       // local lane ids valid on the wire; empty = full copy
    std::vector<int64_t> halo;              // [left, right] read-only lanes kept to complete straddling
                                            // activation blocks; empty when none are needed
    std::vector<int64_t> gqa_map;           // "qsa_head": local query head -> GLOBAL kv head (the 24Q/2KV map)
    bool replicated = false;
};

/// Where an activation's norm/transform/quantizer runs and how tensors move to get there.
struct ActivationContract {
    std::string consumer;
    std::string gather;                     // "none" | "all_gather_padded" | "all_reduce_sum"
    int64_t wire_slots = 0;                 // padded slot width (max valid count); reduce: the shared width
    std::vector<int64_t> valid_counts;      // per rank
    std::string quantizer;
    int64_t block_size = 0;
    bool crosses_rank_boundary = false;     // a quantizer block straddles a rank cut of its vector
    std::string reference;                  // "global_reference" | "local"
};

/// Per-rank memory, unified memory counted once.  `per_rank_total[r]` must equal the shared components
/// plus rank r's owned components, and the gate is the design's memory test.
struct MemoryBudget {
    int64_t per_rank_bytes = 0;                       // the heaviest rank
    std::vector<std::pair<std::string, int64_t>> components;   // per-rank owned parts + parts shared by all ranks
    std::vector<int64_t> per_rank_owned;              // sum of this rank's "weights_owned_*" parts
    std::vector<int64_t> per_rank_total;              // owned + shared, per rank
    int64_t halo_and_global_buffers = 0;              // halo + gather buffers + communication workspace
    int64_t gate_limit_bytes = 0;                     // the memory gate the budget must fit
};

/// The nominal usable budget of one 128 GB Spark, the design's memory gate.  Override on the CLI.
inline constexpr int64_t kSparkGateBytes = 118LL * 1024 * 1024 * 1024;

struct ShardPlan {
    int schema_version = 1;
    int plan_version = 1;
    std::string plan_hash;                  // FNV-1a-64 of the canonical body (without the hash)
    int world = 0;                          // 0 = the plan was rejected
    std::vector<std::string> reject_reasons;// set when world == 0; nothing may be assumed about such a plan
    std::vector<std::string> notes;         // what fell back to a whole-rank read, and the memory numbers
    std::string repo;
    std::string revision;
    std::string inventory_hash;
    core::ModelGeometry geometry;
    MemoryBudget memory;
    std::vector<TensorSlice> slices;        // inventory order, ranks in increasing order inside a tensor
    std::vector<ActivationContract> activation;
};

struct ValidationResult {
    bool ok = true;
    std::vector<std::string> errors;
    void fail(const std::string& e) { ok = false; errors.push_back(e); }
};

inline std::string plan_canonical_body(const ShardPlan& p);   // defined below, used by make_tp_plan

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

namespace detail {

inline uint64_t fnv1a64(const std::string& s) {
    uint64_t h = 0xcbf29ce484222325ULL;
    for (unsigned char c : s) {
        h ^= c;
        h *= 0x100000001b3ULL;
    }
    return h;
}

inline std::string hex64(uint64_t v) {
    static const char* d = "0123456789abcdef";
    std::string s(16, '0');
    for (int i = 15; i >= 0; --i) { s[(size_t)i] = d[v & 0xF]; v >>= 4; }
    return s;
}

inline std::string shape_str(const std::vector<int64_t>& s) {
    std::string out = "[";
    for (size_t i = 0; i < s.size(); ++i) { if (i) out += ","; out += std::to_string(s[i]); }
    return out + "]";
}

inline std::string lane_str(const std::vector<int64_t>& v) {
    std::string out = "[";
    for (size_t i = 0; i < v.size(); ++i) { if (i) out += ","; out += std::to_string(v[i]); }
    return out + "]";
}

inline void json_str(std::string& out, const std::string& s) {
    out += '"';
    for (char c : s) {
        if (c == '"' || c == '\\') { out += '\\'; out += c; }
        else if (c == '\n') out += "\\n";
        else if ((unsigned char)c < 0x20) out += ' ';
        else out += c;
    }
    out += '"';
}

/// Bytes per contiguous slab (one index over every axis after axis 0): what a whole-row read strides by.
inline int64_t slab_bytes(const InventoryTensor& t) {
    if (t.byte_length < 0 || t.shape.empty()) return -1;
    int64_t after = 1;
    for (size_t i = 1; i < t.shape.size(); ++i) after *= t.shape[i];
    if (after <= 0 || t.byte_length % after != 0) return -1;
    return t.byte_length / after;
}

}  // namespace detail

// ---------------------------------------------------------------------------
// A minimal JSON reader for inventory.json (objects, arrays, strings, numbers, null)
// ---------------------------------------------------------------------------

namespace jsonlite {

struct JVal {
    enum class K { Null, Bool, Num, Str, Arr, Obj } k = K::Null;
    bool b = false;
    double num = 0;
    std::string s;
    std::vector<JVal> arr;
    std::map<std::string, JVal> obj;
    bool has(const std::string& key) const { return k == K::Obj && obj.count(key) != 0; }
    const JVal* at(const std::string& key) const {
        auto it = obj.find(key);
        return it == obj.end() ? nullptr : &it->second;
    }
};

struct JReader {
    const std::string& t;
    size_t i = 0;
    std::string* err;
    explicit JReader(const std::string& text, std::string* e) : t(text), err(e) {}
    bool fail(const std::string& why) {
        if (err && err->empty()) *err = "json: " + why + " at offset " + std::to_string(i);
        return false;
    }
    void ws() {
        while (i < t.size() && (t[i] == ' ' || t[i] == '\t' || t[i] == '\n' || t[i] == '\r')) ++i;
    }
    bool parse(JVal& out) {
        ws();
        if (i >= t.size()) return fail("unexpected end of input");
        char c = t[i];
        if (c == '{') return parse_obj(out);
        if (c == '[') return parse_arr(out);
        if (c == '"') { out.k = JVal::K::Str; return parse_str(out.s); }
        if (t.compare(i, 4, "true") == 0) { out.k = JVal::K::Bool; out.b = true; i += 4; return true; }
        if (t.compare(i, 5, "false") == 0) { out.k = JVal::K::Bool; out.b = false; i += 5; return true; }
        if (t.compare(i, 4, "null") == 0) { out.k = JVal::K::Null; i += 4; return true; }
        return parse_num(out);
    }
    bool parse_obj(JVal& out) {
        out.k = JVal::K::Obj;
        ++i;
        ws();
        if (i < t.size() && t[i] == '}') { ++i; return true; }
        for (;;) {
            ws();
            if (i >= t.size() || t[i] != '"') return fail("expected an object key");
            JVal v;
            std::string key;
            if (!parse_str(key)) return false;
            ws();
            if (i >= t.size() || t[i] != ':') return fail("expected ':'");
            ++i;
            if (!parse(v)) return false;
            out.obj[key] = std::move(v);
            ws();
            if (i < t.size() && t[i] == ',') { ++i; continue; }
            if (i < t.size() && t[i] == '}') { ++i; return true; }
            return fail("expected ',' or '}'");
        }
    }
    bool parse_arr(JVal& out) {
        out.k = JVal::K::Arr;
        ++i;
        ws();
        if (i < t.size() && t[i] == ']') { ++i; return true; }
        for (;;) {
            JVal v;
            if (!parse(v)) return false;
            out.arr.push_back(std::move(v));
            ws();
            if (i < t.size() && t[i] == ',') { ++i; continue; }
            if (i < t.size() && t[i] == ']') { ++i; return true; }
            return fail("expected ',' or ']'");
        }
    }
    bool parse_str(std::string& s) {
        s.clear();
        ++i;
        for (;;) {
            if (i >= t.size()) return fail("unterminated string");
            char c = t[i++];
            if (c == '"') return true;
            if (c == '\\') {
                if (i >= t.size()) return fail("unterminated escape");
                char e = t[i++];
                switch (e) {
                case '"': s += '"'; break;
                case '\\': s += '\\'; break;
                case '/': s += '/'; break;
                case 'n': s += '\n'; break;
                case 't': s += '\t'; break;
                case 'r': s += '\r'; break;
                case 'b': s += '\b'; break;
                case 'f': s += '\f'; break;
                case 'u': {
                    if (i + 4 > t.size()) return fail("short \\u escape");
                    i += 4;   // inventory text is ASCII in practice; a \uXXXX becomes '?' rather than guess
                    s += '?';
                    break;
                }
                default: return fail("bad escape");
                }
            } else s += c;
        }
    }
    bool parse_num(JVal& out) {
        size_t start = i;
        if (i < t.size() && (t[i] == '-' || t[i] == '+')) ++i;
        bool digits = false;
        while (i < t.size()) {
            char c = t[i];
            if (c >= '0' && c <= '9') { digits = true; ++i; }
            else if (c == '.' || c == 'e' || c == 'E' || c == '-' || c == '+') ++i;
            else break;
        }
        if (!digits) return fail("expected a value");
        out.k = JVal::K::Num;
        out.num = std::strtod(t.substr(start, i - start).c_str(), nullptr);
        return true;
    }
};

}  // namespace jsonlite

/// Parse one inventory.json into a ModelInventory.  On failure returns false and `err` names the missing or
/// bad key (for example "missing key 'geometry.n_ff'" or "json: ... at offset N").
inline bool parse_inventory_json(const std::string& text, ModelInventory& inv, std::string& err) {
    using jsonlite::JVal;
    using jsonlite::JReader;
    err.clear();
    JReader rd(text, &err);
    JVal root;
    if (!rd.parse(root) || root.k != JVal::K::Obj) {
        if (err.empty()) err = "json: top level is not an object";
        return false;
    }
    auto missing = [&](const std::string& key) {
        if (err.empty()) err = "missing key '" + key + "'";
        return false;
    };
    const auto num_at = [](const JVal& o, const std::string& key) {
        const JVal* v = o.at(key);
        return v && v->k == JVal::K::Num ? (int64_t)v->num : 0;
    };
    const auto str_at = [](const JVal& o, const std::string& key) {
        const JVal* v = o.at(key);
        return v && v->k == JVal::K::Str ? v->s : std::string();
    };

    if (!root.has("schema_version")) return missing("schema_version");
    inv.schema_version = (int)num_at(root, "schema_version");
    const JVal* src = root.at("source");
    if (!src || src->k != JVal::K::Obj) return missing("source");
    inv.source.kind = str_at(*src, "kind");
    inv.source.repo = str_at(*src, "repo");
    inv.source.revision = str_at(*src, "revision");

    const JVal* geom = root.at("geometry");
    if (!geom || geom->k != JVal::K::Obj) return missing("geometry");
    const char* geom_keys[] = {"n_embd", "n_layers", "qsa_interval", "n_vocab", "ssm_state_size",
                               "ssm_k_heads", "ssm_v_heads", "ssm_d_conv", "ssm_conv_channels",
                               "ssm_value_dim", "n_head", "n_head_kv", "head_dim", "n_rot", "idx_q_heads",
                               "idx_key_dim", "idx_block", "idx_top_k", "hc", "hc_lr", "n_expert",
                               "n_expert_used", "n_ff", "n_ff_shexp"};
    for (const char* key : geom_keys)
        if (!geom->has(key)) return missing(std::string("geometry.") + key);
    core::ModelGeometry g;
    g.n_embd = num_at(*geom, "n_embd");
    g.n_layers = num_at(*geom, "n_layers");
    g.qsa_interval = num_at(*geom, "qsa_interval");
    g.n_vocab = num_at(*geom, "n_vocab");
    g.ssm_state_size = num_at(*geom, "ssm_state_size");
    g.ssm_k_heads = num_at(*geom, "ssm_k_heads");
    g.ssm_v_heads = num_at(*geom, "ssm_v_heads");
    g.ssm_d_conv = num_at(*geom, "ssm_d_conv");
    g.ssm_conv_channels = num_at(*geom, "ssm_conv_channels");
    g.ssm_value_dim = num_at(*geom, "ssm_value_dim");
    g.n_head = num_at(*geom, "n_head");
    g.n_head_kv = num_at(*geom, "n_head_kv");
    g.head_dim = num_at(*geom, "head_dim");
    g.n_rot = num_at(*geom, "n_rot");
    g.idx_q_heads = num_at(*geom, "idx_q_heads");
    g.idx_key_dim = num_at(*geom, "idx_key_dim");
    g.idx_block = num_at(*geom, "idx_block");
    g.idx_top_k = num_at(*geom, "idx_top_k");
    g.hc = num_at(*geom, "hc");
    g.hc_lr = num_at(*geom, "hc_lr");
    g.n_expert = num_at(*geom, "n_expert");
    g.n_expert_used = num_at(*geom, "n_expert_used");
    g.n_ff = num_at(*geom, "n_ff");
    g.n_ff_shexp = num_at(*geom, "n_ff_shexp");
    const JVal* rms = geom->at("rms_eps");
    if (rms && rms->k == JVal::K::Num) g.rms_eps = rms->num;
    if (!geom->has("family")) return missing("geometry.family");
    g.family = strata::model_family_from_arch(str_at(*geom, "family"));
    inv.geometry = g;

    const JVal* ts = root.at("tensors");
    if (!ts || ts->k != JVal::K::Arr) return missing("tensors");
    for (size_t i = 0; i < ts->arr.size(); ++i) {
        const JVal& v = ts->arr[i];
        const std::string where = "tensors[" + std::to_string(i) + "]";
        if (v.k != JVal::K::Obj) { err = where + " is not an object"; return false; }
        for (const char* key : {"name", "dtype", "shape", "family"})
            if (!v.has(key)) return missing(where + "." + key);
        InventoryTensor t;
        t.name = str_at(v, "name");
        t.file = str_at(v, "file");
        t.dtype = str_at(v, "dtype");
        const JVal* shp = v.at("shape");
        if (shp->k != JVal::K::Arr || shp->arr.empty()) {
            err = where + ".shape is not a non-empty array";
            return false;
        }
        for (const JVal& d : shp->arr) {
            if (d.k != JVal::K::Num) { err = where + ".shape has a non-number"; return false; }
            t.shape.push_back((int64_t)d.num);
        }
        const JVal* blk = v.at("block_size");
        if (blk && blk->k == JVal::K::Num) t.block_size = (int64_t)blk->num;
        t.scale_kind = str_at(v, "scale_kind");
        const JVal* bo = v.at("byte_offset");
        if (bo && bo->k == JVal::K::Num) t.byte_offset = (int64_t)bo->num;
        const JVal* bl = v.at("byte_length");
        if (bl && bl->k == JVal::K::Num) t.byte_length = (int64_t)bl->num;
        t.family = family_from_string(str_at(v, "family"));
        inv.tensors.push_back(std::move(t));
    }

    const JVal* ac = root.at("activation_contracts");
    if (ac && ac->k == JVal::K::Arr) {
        for (const JVal& v : ac->arr) {
            if (v.k != JVal::K::Obj || !v.has("consumer")) {
                err = "activation_contracts entry without 'consumer'";
                return false;
            }
            InventoryActivation a;
            a.consumer = str_at(v, "consumer");
            a.quantizer = str_at(v, "quantizer");
            const JVal* bs = v.at("block_size");
            if (bs && bs->k == JVal::K::Num) a.block_size = (int64_t)bs->num;
            a.scope = v.has("scope") ? str_at(v, "scope") : std::string("global");
            a.notes = str_at(v, "notes");
            inv.activation_contracts.push_back(std::move(a));
        }
    }

    const JVal* vd = root.at("verdict");
    if (!vd || vd->k != JVal::K::Obj || !vd->has("loadable_by_strata"))
        return missing("verdict.loadable_by_strata");
    if (vd->at("loadable_by_strata")->k != JVal::K::Bool) {
        err = "'verdict.loadable_by_strata' is not a bool";
        return false;
    }
    inv.verdict.loadable = vd->at("loadable_by_strata")->b;
    inv.verdict.checked_tensors = num_at(*vd, "checked_tensors");
    return true;
}

// ---------------------------------------------------------------------------
// Planning
// ---------------------------------------------------------------------------

/// Head ranges for the agreed layout: even split, remainder to the first ranks (16 keys / 3 = 6/5/5).
inline std::vector<std::pair<int64_t, int64_t>> head_ranges(int64_t n, int world) {
    std::vector<std::pair<int64_t, int64_t>> r((size_t)world);
    const int64_t base = n / world, rem = n % world;
    int64_t start = 0;
    for (int i = 0; i < world; ++i) {
        const int64_t size = base + (i < rem ? 1 : 0);
        r[(size_t)i] = {start, start + size};
        start += size;
    }
    return r;
}

/// FFN ranges whose interior starts are multiples of `align` (the largest quantized block along the cut
/// axis of the FFN-width tensors), so no weight block is split.  With 640 / align 64 / world 3 this is
/// 256/192/192 at starts 0/256/448.
inline std::vector<std::pair<int64_t, int64_t>> block_snap_ranges(int64_t n, int world, int64_t align) {
    std::vector<int64_t> bounds((size_t)world + 1, 0);
    bounds[(size_t)world] = n;
    for (int i = 1; i < world; ++i) {
        const int64_t target = n * i / world;
        int64_t b = ((target + align - 1) / align) * align;    // even split snapped up to a block boundary
        if (b <= bounds[(size_t)i - 1]) b = bounds[(size_t)i - 1] + align;   // keep every rank non-empty
        if (b >= n) b = n;
        bounds[(size_t)i] = b;
    }
    std::vector<std::pair<int64_t, int64_t>> r((size_t)world);
    for (int i = 0; i < world; ++i) r[(size_t)i] = {bounds[(size_t)i], bounds[(size_t)i + 1]};
    return r;
}

namespace detail {

/// One cut section of one tensor: which axis, its part name, the unit in columns and the per-rank unit
/// ranges (head ranges, or lane ranges for the FFN).
struct CutSection {
    int axis = 0;
    std::string part;
    int64_t unit = 1;
    int64_t base = 0;                       // where the section starts on its axis (packed sections)
    std::vector<std::pair<int64_t, int64_t>> hranges;
};

/// Decide how one tensor is cut, from the geometry and its shape (GGUF order).  False = read whole.
/// Rules (first match): FFN width on axis 0 = a down input; FFN width on axis 1 = a gate/up output;
/// n_head*head_dim on axis 0 (family down) = the QSA output projection; ssm_value_dim on axis 0 = the GDN
/// output projection; ssm_value_dim on a later axis = the GDN value projection (z-gate style);
/// ssm_conv_channels on the output side = the packed GDN [q|k|v]; qsa_q = whole query heads.
inline bool decide_cut(const InventoryTensor& t, const core::ModelGeometry& g, int world,
                       const std::vector<std::pair<int64_t, int64_t>>& key_rng,
                       const std::vector<std::pair<int64_t, int64_t>>& value_rng,
                       const std::vector<std::pair<int64_t, int64_t>>& qsa_rng,
                       const std::vector<std::pair<int64_t, int64_t>>& ffn_rng,
                       std::vector<CutSection>& secs) {
    if (world <= 1) return false;
    const int64_t ndim = (int64_t)t.shape.size();
    const int64_t last = ndim - 1;
    const int64_t k_total = g.ssm_k_heads * g.ssm_state_size;
    auto sec = [](int axis, std::string part, int64_t unit, int64_t base,
                  std::vector<std::pair<int64_t, int64_t>> hr) {
        CutSection s;
        s.axis = axis;
        s.part = std::move(part);
        s.unit = unit;
        s.base = base;
        s.hranges = std::move(hr);
        return s;
    };
    if ((t.family == TensorFamily::Down || t.family == TensorFamily::ExpertDown) && ndim >= 1 &&
        t.shape[0] == g.n_ff) {
        secs.push_back(sec(0, "ffn_in", 1, 0, ffn_rng));
        return true;
    }
    if ((t.family == TensorFamily::GateUp || t.family == TensorFamily::ExpertGateUp) && ndim >= 2 &&
        t.shape[1] == g.n_ff) {
        secs.push_back(sec(1, "ffn_out", 1, 0, ffn_rng));
        return true;
    }
    // QSA attn_output: the query output vector cut at the query boundaries (family tells it from ssm_out,
    // which has the same 6144 width).
    if (t.family == TensorFamily::Down && ndim >= 1 && t.shape[0] == g.n_head * g.head_dim) {
        secs.push_back(sec(0, "qsa_out", g.head_dim, 0, qsa_rng));
        return true;
    }
    // GDN ssm_out: reads the value heads sliced; the block rule applies.
    if (t.family == TensorFamily::GdnValue && ndim >= 1 && t.shape[0] == g.ssm_value_dim) {
        secs.push_back(sec(0, "value_out", g.ssm_state_size, 0, value_rng));
        return true;
    }
    // GDN value projection (z-gate style): the value heads on the output side, blocks stay intact.
    if (t.family == TensorFamily::GdnValue && ndim >= 2 && t.shape[last] == g.ssm_value_dim &&
        t.shape[0] != g.ssm_value_dim) {
        secs.push_back(sec((int)last, "value_out", g.ssm_state_size, 0, value_rng));
        return true;
    }
    // Packed GDN in_proj [q | k | v] on the output side: split at the head boundaries the geometry gives.
    if (t.family == TensorFamily::GdnKey && ndim >= 2 && t.shape[last] == g.ssm_conv_channels) {
        if (g.ssm_conv_channels == 2 * k_total + g.ssm_value_dim) {
            secs.push_back(sec((int)last, "gdn_q", g.ssm_state_size, 0, key_rng));
            secs.push_back(sec((int)last, "gdn_k", g.ssm_state_size, k_total, key_rng));
            secs.push_back(sec((int)last, "gdn_v", g.ssm_state_size, 2 * k_total, value_rng));
            return true;
        }
        return false;                       // packed layout not derivable from the geometry: do not guess
    }
    // Plain GDN key projection: key heads on the output side.
    if (t.family == TensorFamily::GdnKey && ndim >= 2 && t.shape[last] == k_total) {
        secs.push_back(sec((int)last, "gdn_q", g.ssm_state_size, 0, key_rng));
        return true;
    }
    // QSA attn_q: whole query heads.  Each head is [q head_dim | gate head_dim] contiguous (layer.hpp), so
    // the head's width is the unit and a query never loses its gate half.
    if (t.family == TensorFamily::QsaQ && ndim >= 2 && t.shape[last] % g.n_head == 0) {
        const int64_t head_cols = t.shape[last] / g.n_head;
        if (head_cols % g.head_dim == 0) {
            secs.push_back(sec((int)last, "qsa_head", head_cols, 0, qsa_rng));
            return true;
        }
        return false;
    }
    return false;   // qsa_kv, qsa_idx, router, norm, hc, ple, embed, head, and anything unrecognized
}

}  // namespace detail

/// The TP3 plan for one inventory and world.  A plan with world == 0 and `reject_reasons` means the layout
/// cannot be cut as the contract requires; nothing is ever silently requantized.
inline ShardPlan make_tp_plan(const ModelInventory& inv, int world,
                              int64_t gate_bytes = kSparkGateBytes) {
    ShardPlan plan;
    plan.geometry = inv.geometry;
    plan.repo = inv.source.repo;
    plan.revision = inv.source.revision;
    plan.inventory_hash = inv.inventory_hash;

    auto reject = [&](const std::string& why) { plan.reject_reasons.push_back(why); };

    if (world < 1 || world > 3) {
        reject("world " + std::to_string(world) + ": the plan promises world 1 (identity), world 2 (TP2 "
               "control) and world 3 (TP3); other sizes are not claimed");
        return plan;
    }
    if (inv.schema_version != 1)
        reject("inventory schema_version " + std::to_string(inv.schema_version) +
               "; this planner reads schema_version 1");
    if (inv.tensors.empty()) reject("inventory lists no tensors");
    if (!inv.verdict.loadable) {
        reject("inventory verdict is not loadable_by_strata: the plan is built on tensors that were read, "
               "not assumed");
        return plan;
    }

    const core::ModelGeometry& g = inv.geometry;
    if (world > 1 && (g.ssm_k_heads <= 0 || g.ssm_v_heads % g.ssm_k_heads != 0))
        reject("GDN value heads " + std::to_string(g.ssm_v_heads) + " is not a whole multiple of key heads " +
               std::to_string(g.ssm_k_heads) + ": a key head cannot keep all its value heads");
    if (world > 1 && (g.n_head_kv <= 0 || g.n_head % g.n_head_kv != 0))
        reject("QSA query heads " + std::to_string(g.n_head) + " is not a whole multiple of kv heads " +
               std::to_string(g.n_head_kv) + ": the global GQA map is not defined");
    if (plan.reject_reasons.size()) { plan.world = 0; return plan; }

    // The largest quantized block along axis 0 among the FFN-width tensors: the FFN cut goes to multiples
    // of it.  (Other axis-0 cuts are checked per tensor and fall back to a whole-rank read.)
    int64_t ffn_block = 1;
    for (const auto& t : inv.tensors)
        if ((t.family == TensorFamily::Down || t.family == TensorFamily::ExpertDown) &&
            !t.shape.empty() && t.shape[0] == g.n_ff && t.block_size > ffn_block)
            ffn_block = t.block_size;

    const auto key_rng = world > 1 ? head_ranges(g.ssm_k_heads, world)
                                   : std::vector<std::pair<int64_t, int64_t>>{{0, g.ssm_k_heads}};
    const auto qsa_rng = world > 1 ? head_ranges(g.n_head, world)
                                   : std::vector<std::pair<int64_t, int64_t>>{{0, g.n_head}};
    const int64_t per_key = g.ssm_k_heads > 0 ? g.ssm_v_heads / g.ssm_k_heads : 1;
    std::vector<std::pair<int64_t, int64_t>> value_rng((size_t)world);
    for (int r = 0; r < world; ++r)
        value_rng[(size_t)r] = {key_rng[(size_t)r].first * per_key, key_rng[(size_t)r].second * per_key};
    const auto ffn_rng = world > 1 ? block_snap_ranges(g.n_ff, world, ffn_block)
                                   : std::vector<std::pair<int64_t, int64_t>>{{0, g.n_ff}};

    // The activation block over the FFN vector (whichever scope): when it does not divide the FFN cuts,
    // the reference path keeps whole crossing blocks as read-only halos on the cut tensors.
    int64_t ffn_act_block = 0;
    for (const auto& a : inv.activation_contracts)
        if (a.block_size > 1 && (a.consumer == "down" || a.consumer == "expert_down"))
            ffn_act_block = std::max(ffn_act_block, a.block_size);

    for (const auto& t : inv.tensors) {
        if (t.shape.empty()) {
            reject("tensor " + t.name + ": empty shape in the inventory");
            plan.world = 0;
            return plan;
        }
        std::vector<detail::CutSection> secs;
        bool cut_ok = decide_cut(t, g, world, key_rng, value_rng, qsa_rng, ffn_rng, secs);
        // A cut on the block axis (axis 0) must land on block boundaries for every rank.  The FFN is the
        // agreed contract: a violation rejects the plan.  A violation elsewhere is the design's cue to
        // read the tensor whole on every rank (the reference path) and to say so.
        bool block_violation = false;
        int64_t bad_start = 0, bad_block = 0;
        if (cut_ok && t.block_size > 1) {
            for (const auto& s : secs) {
                if (s.axis != 0) continue;
                for (const auto& [hs, he] : s.hranges) {
                    (void)he;
                    const int64_t col = s.base + hs * s.unit;
                    if (col % t.block_size != 0) {
                        block_violation = true;
                        bad_start = col;
                        bad_block = t.block_size;
                        break;        // name the first cut that breaks a block, not the last
                    }
                }
                if (block_violation) break;
            }
        }
        if (block_violation) {
            const bool ffn_width = !t.shape.empty() && t.shape[0] == g.n_ff;
            if (ffn_width) {
                reject("tensor " + t.name + ": block size " + std::to_string(bad_block) + " is cut at offset " +
                       std::to_string(bad_start) + "; requantizing to fit is not allowed");
                plan.world = 0;
                return plan;
            }
            plan.notes.push_back("tensor " + t.name + ": the agreed cut at " + std::to_string(bad_start) +
                                 " splits a " + std::to_string(bad_block) + " block; it is read whole by "
                                 "every rank (reference path), not requantized");
            secs.clear();
            cut_ok = false;
        }
        if (cut_ok) {
            for (const auto& s : secs) {
                int64_t covered = 0;
                for (const auto& [hs, he] : s.hranges) covered += (he - hs) * s.unit;
                if (s.base + covered > t.shape[(size_t)s.axis]) {
                    reject("tensor " + t.name + ": the " + s.part + " cut would run past " +
                           std::to_string(t.shape[(size_t)s.axis]));
                    plan.world = 0;
                    return plan;
                }
            }
        }
        const int64_t stride = detail::slab_bytes(t);
        if (cut_ok && stride <= 0) {
            reject("tensor " + t.name + ": no whole original byte stride (byte_length " +
                   std::to_string(t.byte_length) + " over " + detail::shape_str(t.shape) + "); a sharded "
                   "read needs the original stride, not a repack");
            plan.world = 0;
            return plan;
        }
        if (cut_ok && t.block_size > 1 && t.scale_kind.empty()) {
            reject("tensor " + t.name + ": block-quantized with no scale reference");
            plan.world = 0;
            return plan;
        }

        for (int rank = 0; rank < world; ++rank) {
            if (!cut_ok) {
                TensorSlice s;
                s.tensor = t.name;
                s.family = t.family;
                s.rank = rank;
                s.global_shape = t.shape;
                s.local_shape = t.shape;
                s.axis = 0;
                s.range_start = 0;
                s.range_end = t.shape[0];
                s.byte_row_stride = stride;
                s.codec = t.dtype;
                s.block_size = t.block_size;
                s.scale_ref = t.block_size > 1 ? t.scale_kind : std::string();
                s.replicated = true;
                plan.slices.push_back(std::move(s));
                continue;
            }
            for (const auto& sec : secs) {
                const auto [hs, he] = sec.hranges[(size_t)rank];
                TensorSlice s;
                s.tensor = t.name;
                s.family = t.family;
                s.rank = rank;
                s.part = sec.part;
                s.global_shape = t.shape;
                s.local_shape = t.shape;
                s.axis = sec.axis;
                s.cut_unit = sec.unit;
                s.head_start = hs;
                s.head_end = he;
                s.range_start = sec.base + hs * sec.unit;
                s.range_end = sec.base + he * sec.unit;
                s.byte_row_stride = stride;
                s.codec = t.dtype;
                s.block_size = t.block_size;
                s.scale_ref = t.block_size > 1 ? t.scale_kind : std::string();
                s.local_shape[(size_t)s.axis] = s.range_end - s.range_start;
                s.valid_lanes.resize((size_t)(s.range_end - s.range_start));
                for (size_t i = 0; i < s.valid_lanes.size(); ++i) s.valid_lanes[i] = (int64_t)i;
                if (sec.part == "qsa_head") {
                    const int64_t per_kv = g.n_head / g.n_head_kv;   // global 24Q/2KV map, never a local count
                    for (int64_t h = hs; h < he; ++h) s.gqa_map.push_back(h / per_kv);
                }
                if (world > 1 && ffn_act_block > 0 && sec.part == "ffn_in") {
                    const int64_t left_pad = (s.range_start % ffn_act_block == 0 || s.range_start == 0)
                                                ? 0 : s.range_start - (s.range_start / ffn_act_block) *
                                                                          ffn_act_block;
                    const int64_t right_pad = (s.range_end % ffn_act_block == 0 ||
                                               s.range_end >= t.shape[(size_t)s.axis])
                                                ? 0 : ((s.range_end + ffn_act_block - 1) / ffn_act_block) *
                                                          ffn_act_block - s.range_end;
                    if (left_pad || right_pad) s.halo = {left_pad, right_pad};
                }
                plan.slices.push_back(std::move(s));
            }
        }
    }
    if (plan.reject_reasons.size()) { plan.world = 0; return plan; }

    // Activation contracts, per the consumer's vector:
    //   down / expert_down -> the FFN vector, cut at the FFN boundaries
    //   attn_output        -> the query-output vector, cut at the query boundaries
    //   embed / ple        -> rows gathered by token id (identity per rank, summed downstream)
    //   anything else      -> the whole n_embd residual: every rank holds it in full, one global result
    for (const auto& a : inv.activation_contracts) {
        ActivationContract c;
        c.consumer = a.consumer;
        c.quantizer = a.quantizer;
        c.block_size = a.block_size;
        std::vector<std::pair<int64_t, int64_t>> cuts;
        if (world > 1 && (a.consumer == "down" || a.consumer == "expert_down")) cuts = ffn_rng;
        else if (world > 1 && a.consumer == "attn_output") {
            for (const auto& [h0, h1] : qsa_rng) cuts.push_back({h0 * g.head_dim, h1 * g.head_dim});
        }
        if (cuts.empty()) {
            c.gather = "none";
            c.reference = a.scope == "global" ? "global_reference" : "local";
            c.wire_slots = (a.consumer == "embed" || a.consumer == "ple") ? g.n_vocab : g.n_embd;
            c.valid_counts.assign((size_t)world, c.wire_slots);
            plan.activation.push_back(c);
            continue;
        }
        for (const auto& [s, e] : cuts) c.valid_counts.push_back(e - s);
        for (size_t i = 0; i + 1 < cuts.size(); ++i)
            if (a.block_size > 1 && cuts[i].second % a.block_size != 0) c.crosses_rank_boundary = true;
        const bool transform = a.quantizer.find("adamard") != std::string::npos;
        if (a.scope == "global" || transform || c.crosses_rank_boundary) {
            if (a.scope == "local" && c.crosses_rank_boundary && !transform)
                plan.notes.push_back("consumer " + a.consumer + ": a local " + std::to_string(a.block_size) +
                                     "-block quantizer crosses a rank cut; it is routed to the global "
                                     "reference path, not requantized");
            c.gather = "all_gather_padded";
            c.reference = "global_reference";
        } else {
            c.gather = "none";
            c.reference = "local";
        }
        int64_t maxv = 0, sum = 0;
        for (int64_t v : c.valid_counts) { maxv = std::max(maxv, v); sum += v; }
        c.wire_slots = std::max(maxv, (sum + world - 1) / world);
        plan.activation.push_back(c);
    }
    if (world > 1) {
        // The cut projections produce full-width partials: reduced once, with uniform counts into a
        // uniform collective, never unequal counts into an all-reduce.
        ActivationContract red;
        red.consumer = "tp.down_partial_sum";
        red.gather = "all_reduce_sum";
        red.wire_slots = g.n_embd;
        red.valid_counts.assign((size_t)world, g.n_embd);
        red.quantizer = "none";
        red.reference = "global_reference";
        plan.activation.push_back(red);
    }

    // Per-rank memory.  Owned parts are named per rank (the split is unequal by design); parts every rank
    // pays in full are named as shared.  The gate is the design's memory test: replicated PLE stays only
    // while it fits; row ownership of PLE/embedding rows is the alternative when it does not.
    int64_t ple_bytes = 0, vocab_bytes = 0, other_shared = 0, owned_total = 0;
    std::vector<int64_t> owned_ffn((size_t)world, 0), owned_attn((size_t)world, 0);
    for (const auto& t : inv.tensors) {
        const int64_t bytes = t.byte_length > 0 ? t.byte_length : 0;
        bool any_sharded = false;
        for (const auto& s : plan.slices) {
            if (s.tensor != t.name || s.replicated) continue;
            any_sharded = true;
            const int64_t dim = t.shape[(size_t)s.axis];
            const int64_t share = dim > 0 ? bytes * (s.range_end - s.range_start) / dim : 0;
            owned_total += share;
            if (s.part == "ffn_in" || s.part == "ffn_out") owned_ffn[(size_t)s.rank] += share;
            else owned_attn[(size_t)s.rank] += share;
        }
        if (!any_sharded) {
            if (t.family == TensorFamily::Ple) ple_bytes += bytes;
            else if (t.family == TensorFamily::Embed || t.family == TensorFamily::Head) vocab_bytes += bytes;
            else other_shared += bytes;
        }
    }
    int64_t halo_bytes = 0;
    for (const auto& t : inv.tensors) {
        const int64_t bytes = t.byte_length > 0 ? t.byte_length : 0;
        for (const auto& s : plan.slices)
            if (s.tensor == t.name && !s.halo.empty() && t.shape[(size_t)s.axis] > 0)
                halo_bytes += (s.halo[0] + (s.halo.size() > 1 ? s.halo[1] : 0)) *
                              (bytes / t.shape[(size_t)s.axis]);
    }
    int64_t gather_bytes = 0;
    for (const auto& c : plan.activation)
        if (c.gather == "all_gather_padded") gather_bytes += c.wire_slots * 4 * world;
    const int64_t state_bytes =
        g.n_gdn_layers() * (g.ssm_k_heads * g.ssm_state_size + g.ssm_conv_channels * g.ssm_d_conv) * 4 +
        g.n_qsa_layers() * (g.n_head_kv * g.head_dim * 2 + g.idx_q_heads * g.idx_key_dim) * 2;
    const int64_t prefill_scratch = g.n_embd * 4 * 64;
    const int64_t comm_workspace = 64LL * 1024 * 1024;
    const int64_t shared = ple_bytes + vocab_bytes + other_shared + state_bytes + halo_bytes +
                           gather_bytes + prefill_scratch + comm_workspace;
    plan.memory.components.clear();
    for (int r = 0; r < world; ++r) {
        plan.memory.components.push_back({"weights_owned_ffn.rank" + std::to_string(r), owned_ffn[(size_t)r]});
        plan.memory.components.push_back({"weights_owned_attn.rank" + std::to_string(r), owned_attn[(size_t)r]});
    }
    plan.memory.components.insert(plan.memory.components.end(), {
        {"weights_replicated_ple", ple_bytes},
        {"weights_replicated_vocabulary", vocab_bytes},
        {"weights_replicated_other", other_shared},
        {"kv_gdn_indexer_state", state_bytes},
        {"halo", halo_bytes},
        {"activation_gather", gather_bytes},
        {"prefill_scratch", prefill_scratch},
        {"comm_workspace", comm_workspace},
    });
    plan.memory.per_rank_owned.assign((size_t)world, 0);
    plan.memory.per_rank_total.assign((size_t)world, 0);
    for (int r = 0; r < world; ++r) {
        plan.memory.per_rank_owned[(size_t)r] = owned_ffn[(size_t)r] + owned_attn[(size_t)r];
        plan.memory.per_rank_total[(size_t)r] = plan.memory.per_rank_owned[(size_t)r] + shared;
        plan.memory.per_rank_bytes = std::max(plan.memory.per_rank_bytes, plan.memory.per_rank_total[(size_t)r]);
    }
    plan.memory.halo_and_global_buffers = halo_bytes + gather_bytes + comm_workspace;
    plan.memory.gate_limit_bytes = gate_bytes;

    // The numbers the memory gate is judged on, visible in the plan itself.
    {
        std::string per_rank = "ffn owned per rank:";
        for (int r = 0; r < world; ++r)
            per_rank += " rank" + std::to_string(r) + " " + std::to_string(owned_ffn[(size_t)r]) + " B";
        plan.notes.push_back(per_rank);
        if (ple_bytes > 0) {
            const int64_t pack = owned_total + ple_bytes + vocab_bytes + other_shared;
            plan.notes.push_back("ple replicated: " + std::to_string(ple_bytes) + " B on every rank (" +
                                 std::to_string(pack > 0 ? (100 * ple_bytes) / pack : 0) + "% of the pack)");
        }
        for (int r = 0; r < world; ++r)
            if (plan.memory.per_rank_total[(size_t)r] > gate_bytes)
                plan.notes.push_back("memory gate FAILED for rank " + std::to_string(r) + ": needs " +
                                     std::to_string(plan.memory.per_rank_total[(size_t)r]) + " B of " +
                                     std::to_string(gate_bytes) + " B; row ownership of PLE/embedding is the "
                                     "design's alternative");
    }

    plan.world = world;
    plan.plan_hash = detail::hex64(detail::fnv1a64(plan_canonical_body(plan)));
    return plan;
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

inline std::string plan_canonical_body(const ShardPlan& p) {
    std::string o;
    o += "tp3/2|world=" + std::to_string(p.world) + "|repo=" + p.repo + "|rev=" + p.revision +
         "|inv=" + p.inventory_hash + "|gate=" + std::to_string(p.memory.gate_limit_bytes);
    o += "|geom=" + std::to_string(p.geometry.n_embd) + "," + std::to_string(p.geometry.n_layers) + "," +
         std::to_string(p.geometry.qsa_interval) + "," + std::to_string(p.geometry.ssm_k_heads) + "," +
         std::to_string(p.geometry.ssm_v_heads) + "," + std::to_string(p.geometry.n_head) + "," +
         std::to_string(p.geometry.n_head_kv) + "," + std::to_string(p.geometry.head_dim) + "," +
         std::to_string(p.geometry.n_expert) + "," + std::to_string(p.geometry.n_ff);
    for (const auto& s : p.slices) {
        o += "|sl=" + s.tensor + "," + family_to_string(s.family) + "," + std::to_string(s.rank) + "," +
             s.part + "," + detail::shape_str(s.global_shape) + "," + detail::shape_str(s.local_shape) +
             "," + std::to_string(s.axis) + "," + std::to_string(s.range_start) + "," +
             std::to_string(s.range_end) + "," + std::to_string(s.cut_unit) + "," +
             std::to_string(s.head_start) + "," + std::to_string(s.head_end) + "," +
             std::to_string(s.byte_row_stride) + "," + s.codec + "," + std::to_string(s.block_size) + "," +
             s.scale_ref + "," + detail::lane_str(s.valid_lanes) + "," + detail::lane_str(s.halo) + "," +
             detail::lane_str(s.gqa_map) + (s.replicated ? ",R" : ",S");
    }
    for (const auto& c : p.activation) {
        o += "|ac=" + c.consumer + "," + c.gather + "," + std::to_string(c.wire_slots) + "," +
             detail::lane_str(c.valid_counts) + "," + c.quantizer + "," + std::to_string(c.block_size) +
             (c.crosses_rank_boundary ? ",X" : ",-") + "," + c.reference;
    }
    for (const auto& m : p.memory.components)
        o += "|mem=" + m.first + "," + std::to_string(m.second);
    o += "|mb=" + std::to_string(p.memory.per_rank_bytes) + "," +
         detail::lane_str(p.memory.per_rank_owned) + "," + detail::lane_str(p.memory.per_rank_total) + "," +
         std::to_string(p.memory.halo_and_global_buffers);
    for (const auto& n : p.notes) o += "|n=" + n;
    return o;
}

/// Checks the plan against itself: coverage of every tensor without gaps or overlaps, weight-block
/// integrity on the cut axis, the key-with-three-values rule, the global GQA map, the reference-path rules
/// for crossing quantizers, uniform reduce counts, a budget whose per-rank totals add up, and the memory
/// gate with its numbers.  A rejected plan (world == 0) fails with its reason.
inline ValidationResult validate_plan(const ShardPlan& p) {
    ValidationResult v;
    if (p.world == 0) {
        v.fail("plan was rejected at build time: " +
               (p.reject_reasons.empty() ? std::string("no reason recorded") : p.reject_reasons.front()));
        return v;
    }
    if (p.world < 1 || p.world > 3) v.fail("world " + std::to_string(p.world) + " is outside 1..3");
    if (p.schema_version != 1 || p.plan_version != 1) v.fail("schema/plan version is not 1/1");
    const std::string want = detail::hex64(detail::fnv1a64(plan_canonical_body(p)));
    if (p.plan_hash != want)
        v.fail("plan hash " + p.plan_hash + " does not match the plan body (" + want +
               "); a frozen plan's numbers must be the plan's own");

    std::vector<std::string> names;
    for (const auto& s : p.slices)
        if (std::find(names.begin(), names.end(), s.tensor) == names.end()) names.push_back(s.tensor);

    int64_t ple_comp = 0;
    for (const auto& m : p.memory.components)
        if (m.first == "weights_replicated_ple") ple_comp = m.second;

    for (const auto& name : names) {
        std::vector<const TensorSlice*> grp;
        for (const auto& s : p.slices) if (s.tensor == name) grp.push_back(&s);
        const TensorSlice& s0 = *grp.front();
        const int64_t total = s0.global_shape.empty() ? 0 : s0.global_shape[(size_t)s0.axis];
        if (s0.replicated) {
            if ((int)grp.size() != p.world)
                v.fail("tensor " + name + " has " + std::to_string((int)grp.size()) +
                       " whole-copy slices, world is " + std::to_string(p.world));
            for (const auto* s : grp) {
                if (!s->replicated) { v.fail("tensor " + name + ": mixed replicated and sharded slices"); continue; }
                if (s->range_start != 0 || s->range_end != total)
                    v.fail("tensor " + name + " rank " + std::to_string(s->rank) +
                           ": replicated slice must own the full range");
            }
            continue;
        }
        std::vector<std::pair<int64_t, int64_t>> rs;
        for (const auto* s : grp) {
            if (s->replicated) { v.fail("tensor " + name + ": mixed replicated and sharded slices"); continue; }
            if (s->range_start < 0 || s->range_end > total || s->range_start >= s->range_end)
                v.fail("tensor " + name + " rank " + std::to_string(s->rank) + " part " + s->part +
                       ": range [" + std::to_string(s->range_start) + "," + std::to_string(s->range_end) +
                       ") outside [0," + std::to_string(total) + ")");
            if (s->local_shape[(size_t)s->axis] != s->range_end - s->range_start)
                v.fail("tensor " + name + " rank " + std::to_string(s->rank) +
                       ": local shape does not match the range");
            if (s->byte_row_stride <= 0)
                v.fail("tensor " + name + " rank " + std::to_string(s->rank) +
                       ": sharded without the original byte stride");
            if ((int64_t)s->valid_lanes.size() != s->range_end - s->range_start)
                v.fail("tensor " + name + " rank " + std::to_string(s->rank) +
                       ": valid lanes do not cover the owned lanes");
            if (s->block_size > 1 && s->axis == 0 && s->range_start % s->block_size != 0)
                v.fail("tensor " + name + " rank " + std::to_string(s->rank) + " part " + s->part +
                       ": cut at " + std::to_string(s->range_start) + " splits a " +
                       std::to_string(s->block_size) + " weight block");
            if (s->block_size > 1 && s->scale_ref.empty())
                v.fail("tensor " + name + ": block-quantized slice names no scales");
            rs.push_back({s->range_start, s->range_end});
        }
        std::sort(rs.begin(), rs.end());
        if (rs.empty()) { v.fail("tensor " + name + ": no slices"); continue; }
        if (rs.front().first != 0 || rs.back().second != total)
            v.fail("tensor " + name + ": ownership does not cover [0," + std::to_string(total) + ")");
        for (size_t i = 1; i < rs.size(); ++i)
            if (rs[i].first != rs[i - 1].second)
                v.fail("tensor " + name + ": gap or overlap at " + std::to_string(rs[i].first));
    }

    if (p.world > 1) {
        // Each key's per_key value heads stay together: every rank's value ranges are its key ranges x per_key.
        const int64_t per_key = p.geometry.ssm_k_heads > 0
                                    ? p.geometry.ssm_v_heads / p.geometry.ssm_k_heads : 1;
        for (int rank = 0; rank < p.world; ++rank) {
            std::vector<std::pair<int64_t, int64_t>> keys, vals;
            for (const auto& s : p.slices) {
                if (s.rank != rank) continue;
                if (s.part == "gdn_q" || s.part == "gdn_k") keys.push_back({s.head_start, s.head_end});
                if (s.part == "gdn_v" || s.part == "value_out") vals.push_back({s.head_start, s.head_end});
            }
            std::sort(keys.begin(), keys.end());
            keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
            if (keys.size() != 1) continue;     // no key cut on this rank (or a fallback): nothing to pair
            for (const auto& val : vals) {
                const int64_t lo = keys.front().first * per_key, hi = keys.front().second * per_key;
                if (val.first != lo || val.second != hi) {
                    v.fail("rank " + std::to_string(rank) + ": value heads [" +
                           std::to_string(val.first) + "," + std::to_string(val.second) + ") do not keep the "
                           "key's " + std::to_string(per_key) + " values per key (expected [" +
                           std::to_string(lo) + "," + std::to_string(hi) + "))");
                    break;
                }
            }
        }
        // QSA: the gqa_map is the global 24Q/2KV map, never a local query/kv count.
        const int64_t per_kv = p.geometry.n_head_kv > 0 ? p.geometry.n_head / p.geometry.n_head_kv : 0;
        for (const auto& s : p.slices) {
            if (s.part != "qsa_head") continue;
            if (per_kv <= 0) { v.fail("qsa: geometry has no usable kv head count"); continue; }
            const int64_t heads = s.range_end - s.range_start;
            if (heads % s.cut_unit != 0) {
                v.fail("qsa slice rank " + std::to_string(s.rank) + ": range is not whole heads");
                continue;
            }
            if ((int64_t)s.gqa_map.size() != heads / s.cut_unit) {
                v.fail("qsa slice rank " + std::to_string(s.rank) + ": gqa_map does not cover the range");
                continue;
            }
            for (size_t i = 0; i < s.gqa_map.size(); ++i) {
                const int64_t global_head = s.head_start + (int64_t)i;
                if (s.gqa_map[i] != global_head / per_kv)
                    v.fail("qsa slice rank " + std::to_string(s.rank) + " lane " + std::to_string((int64_t)i) +
                           ": kv head " + std::to_string(s.gqa_map[i]) + " but the global " +
                           std::to_string(p.geometry.n_head) + "Q/" +
                           std::to_string(p.geometry.n_head_kv) + "KV map says " +
                           std::to_string(global_head / per_kv) +
                           "; GQA is never derived from local counts");
            }
        }
    }

    for (const auto& c : p.activation) {
        if ((int)c.valid_counts.size() != p.world)
            v.fail("consumer " + c.consumer + ": valid_counts per rank missing (world " +
                   std::to_string(p.world) + ")");
        if (c.valid_counts.empty()) continue;
        if (c.gather == "all_reduce_sum") {
            const int64_t first = c.valid_counts.front();
            for (int64_t x : c.valid_counts)
                if (x != first)
                    v.fail("consumer " + c.consumer +
                           ": unequal valid counts passed to a uniform all-reduce");
            if (c.wire_slots < first) v.fail("consumer " + c.consumer + ": wire slots below the width");
        }
        if (c.gather == "all_gather_padded") {
            int64_t maxv = 0, sum = 0;
            for (int64_t x : c.valid_counts) { maxv = std::max(maxv, x); sum += x; }
            if (c.wire_slots < maxv || c.wire_slots * (int64_t)p.world < sum)
                v.fail("consumer " + c.consumer + ": padded wire slots do not hold the valid counts");
            if (c.crosses_rank_boundary && c.reference != "global_reference")
                v.fail("consumer " + c.consumer + ": a quantizer block crosses a rank boundary, the "
                       "reference path must be global");
            if (c.crosses_rank_boundary) {
                bool any_halo = false;
                for (const auto& s : p.slices) if (!s.halo.empty()) any_halo = true;
                if (!any_halo)
                    v.fail("consumer " + c.consumer + ": boundary-crossing block but no halo on any cut "
                           "slice");
            }
        }
    }

    // Memory: per-rank totals = shared parts + that rank's owned parts; halos and global buffers are
    // inside the total, and the gate is checked with its numbers.
    int64_t shared_comp = 0, halo_comp = 0, gather_comp = 0, comm_comp = 0;
    bool has_halo = false, has_gather = false, has_comm = false;
    for (const auto& m : p.memory.components) {
        if (m.first.rfind("weights_owned_", 0) != 0) shared_comp += m.second;
        if (m.first == "halo") { has_halo = true; halo_comp = m.second; }
        if (m.first == "activation_gather") { has_gather = true; gather_comp = m.second; }
        if (m.first == "comm_workspace") { has_comm = true; comm_comp = m.second; }
    }
    if ((int)p.memory.per_rank_total.size() != p.world || (int)p.memory.per_rank_owned.size() != p.world)
        v.fail("memory: per-rank detail is missing for this world");
    else {
        for (int r = 0; r < p.world; ++r) {
            int64_t owned_r = 0;
            for (const auto& m : p.memory.components)
                if (m.first == "weights_owned_ffn.rank" + std::to_string(r) ||
                    m.first == "weights_owned_attn.rank" + std::to_string(r))
                    owned_r += m.second;
            if (owned_r != p.memory.per_rank_owned[(size_t)r])
                v.fail("memory: rank " + std::to_string(r) + " owned detail " +
                       std::to_string(p.memory.per_rank_owned[(size_t)r]) + " != its components " +
                       std::to_string(owned_r));
            if (owned_r + shared_comp != p.memory.per_rank_total[(size_t)r])
                v.fail("memory: rank " + std::to_string(r) + " total " +
                       std::to_string(p.memory.per_rank_total[(size_t)r]) + " != owned " +
                       std::to_string(owned_r) + " + shared " + std::to_string(shared_comp));
            if (p.memory.gate_limit_bytes > 0 &&
                p.memory.per_rank_total[(size_t)r] > p.memory.gate_limit_bytes)
                v.fail("memory gate: rank " + std::to_string(r) + " needs " +
                       std::to_string(p.memory.per_rank_total[(size_t)r]) + " B of the " +
                       std::to_string(p.memory.gate_limit_bytes) + " B limit; the replicated PLE is " +
                       std::to_string(ple_comp) + " B; row ownership of PLE/embedding is the design's "
                       "alternative");
        }
        int64_t max_total = 0;
        for (int64_t x : p.memory.per_rank_total) max_total = std::max(max_total, x);
        if (max_total != p.memory.per_rank_bytes)
            v.fail("memory: per_rank_bytes " + std::to_string(p.memory.per_rank_bytes) +
                   " is not the heaviest rank " + std::to_string(max_total));
    }
    bool need_halo = false, need_gather = false;
    for (const auto& s : p.slices) need_halo = need_halo || !s.halo.empty();
    for (const auto& c : p.activation) need_gather = need_gather || c.gather == "all_gather_padded";
    if (need_halo && !has_halo) v.fail("memory: slices keep halos but no halo component is counted");
    if (need_gather && !has_gather)
        v.fail("memory: the plan gathers activations but no activation_gather component is counted");
    if (p.world > 1 && !has_comm) v.fail("memory: a multi-rank plan counts no communication workspace");
    if (p.memory.halo_and_global_buffers != halo_comp + gather_comp + comm_comp)
        v.fail("memory: halo_and_global_buffers " + std::to_string(p.memory.halo_and_global_buffers) +
               " != halo + gather + workspace " + std::to_string(halo_comp + gather_comp + comm_comp));
    if (need_halo && p.memory.halo_and_global_buffers <= 0)
        v.fail("memory: halos exist but halo_and_global_buffers counts nothing");
    return v;
}

// ---------------------------------------------------------------------------
// tp-plan.json (the contract's manifest; the plan hash freezes it)
// ---------------------------------------------------------------------------

inline std::string to_tp_plan_json(const ShardPlan& p) {
    using detail::json_str;
    std::string o;
    o += "{\n";
    o += "  \"schema_version\": " + std::to_string(p.schema_version) + ",\n";
    o += "  \"plan_version\": " + std::to_string(p.plan_version) + ",\n";
    o += "  \"plan_hash\": "; json_str(o, p.plan_hash); o += ",\n";
    o += "  \"world\": " + std::to_string(p.world) + ",\n";
    if (!p.reject_reasons.empty()) {
        o += "  \"reject_reasons\": [";
        for (size_t i = 0; i < p.reject_reasons.size(); ++i) {
            if (i) o += ", ";
            json_str(o, p.reject_reasons[i]);
        }
        o += "],\n";
    }
    if (!p.notes.empty()) {
        o += "  \"notes\": [";
        for (size_t i = 0; i < p.notes.size(); ++i) {
            if (i) o += ", ";
            json_str(o, p.notes[i]);
        }
        o += "],\n";
    }
    o += "  \"source_lock\": {\"repo\": "; json_str(o, p.repo);
    if (p.revision.empty()) o += ", \"revision\": null";
    else { o += ", \"revision\": "; json_str(o, p.revision); }
    o += ", \"inventory_hash\": "; json_str(o, p.inventory_hash); o += "},\n";
    o += "  \"geometry\": {\"n_embd\": " + std::to_string(p.geometry.n_embd) +
         ", \"n_layers\": " + std::to_string(p.geometry.n_layers) +
         ", \"qsa_interval\": " + std::to_string(p.geometry.qsa_interval) +
         ", \"n_vocab\": " + std::to_string(p.geometry.n_vocab) +
         ", \"ssm_k_heads\": " + std::to_string(p.geometry.ssm_k_heads) +
         ", \"ssm_v_heads\": " + std::to_string(p.geometry.ssm_v_heads) +
         ", \"ssm_state_size\": " + std::to_string(p.geometry.ssm_state_size) +
         ", \"n_head\": " + std::to_string(p.geometry.n_head) +
         ", \"n_head_kv\": " + std::to_string(p.geometry.n_head_kv) +
         ", \"head_dim\": " + std::to_string(p.geometry.head_dim) +
         ", \"idx_q_heads\": " + std::to_string(p.geometry.idx_q_heads) +
         ", \"n_expert\": " + std::to_string(p.geometry.n_expert) +
         ", \"n_ff\": " + std::to_string(p.geometry.n_ff) + "},\n";
    o += "  \"memory\": {\"per_rank_bytes\": " + std::to_string(p.memory.per_rank_bytes) +
         ", \"components\": {";
    for (size_t i = 0; i < p.memory.components.size(); ++i) {
        if (i) o += ", ";
        json_str(o, p.memory.components[i].first);
        o += ": " + std::to_string(p.memory.components[i].second);
    }
    o += "}, \"per_rank_total\": " + detail::lane_str(p.memory.per_rank_total) +
         ", \"halo_and_global_buffers\": " + std::to_string(p.memory.halo_and_global_buffers) +
         ", \"gate_limit_bytes\": " + std::to_string(p.memory.gate_limit_bytes) + "},\n";
    o += "  \"slices\": [\n";
    for (size_t i = 0; i < p.slices.size(); ++i) {
        const auto& s = p.slices[i];
        o += "    {\"tensor\": "; json_str(o, s.tensor);
        o += ", \"family\": "; json_str(o, family_to_string(s.family));
        o += ", \"rank\": " + std::to_string(s.rank);
        if (!s.part.empty()) { o += ", \"part\": "; json_str(o, s.part); }
        o += ", \"global_shape\": " + detail::shape_str(s.global_shape);
        o += ", \"local_shape\": " + detail::shape_str(s.local_shape);
        o += ", \"axis\": " + std::to_string(s.axis);
        o += ", \"range\": [" + std::to_string(s.range_start) + ", " + std::to_string(s.range_end) + "]";
        o += ", \"cut_unit\": " + std::to_string(s.cut_unit);
        o += ", \"head_range\": [" + std::to_string(s.head_start) + ", " + std::to_string(s.head_end) + "]";
        o += ", \"byte_row_stride\": ";
        if (s.byte_row_stride < 0) o += "null"; else o += std::to_string(s.byte_row_stride);
        o += ", \"codec\": "; json_str(o, s.codec);
        o += ", \"block_size\": ";
        if (s.block_size <= 0) o += "null"; else o += std::to_string(s.block_size);
        o += ", \"scale_ref\": ";
        if (s.scale_ref.empty()) o += "null"; else json_str(o, s.scale_ref);
        o += ", \"valid_lanes\": " + detail::lane_str(s.valid_lanes);
        o += ", \"halo\": " + detail::lane_str(s.halo);
        o += std::string(", \"replicated\": ") + (s.replicated ? "true" : "false");
        if (!s.gqa_map.empty()) o += ", \"gqa_map\": " + detail::lane_str(s.gqa_map);
        o += "}";
        if (i + 1 < p.slices.size()) o += ",";
        o += "\n";
    }
    o += "  ],\n  \"activation\": [\n";
    for (size_t i = 0; i < p.activation.size(); ++i) {
        const auto& c = p.activation[i];
        o += "    {\"consumer\": "; json_str(o, c.consumer);
        o += ", \"gather\": "; json_str(o, c.gather);
        o += ", \"wire_slots\": " + std::to_string(c.wire_slots);
        o += ", \"valid_counts\": " + detail::lane_str(c.valid_counts);
        o += ", \"quantizer\": "; json_str(o, c.quantizer);
        o += ", \"block_size\": " + std::to_string(c.block_size);
        o += std::string(", \"crosses_rank_boundary\": ") + (c.crosses_rank_boundary ? "true" : "false");
        o += ", \"reference\": "; json_str(o, c.reference);
        o += "}";
        if (i + 1 < p.activation.size()) o += ",";
        o += "\n";
    }
    o += "  ]\n}\n";
    return o;
}

}  // namespace strata::distributed
