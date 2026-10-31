#include "strata/distributed/state_evidence.hpp"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

#ifdef STRATA_STATE_EVIDENCE_WITH_CORE
#include "strata/core/session.hpp"
#include "strata/kernels/ngram.hpp"
#endif

namespace strata::distributed {
namespace {
bool fail(std::string& err, const std::string& text) { err = text; return false; }
bool multiply(uint64_t a, uint64_t b, uint64_t& out) {
    if (b && a > std::numeric_limits<uint64_t>::max() / b) return false;
    out = a * b; return true;
}
unsigned width(const std::string& dtype) {
    if (dtype == "<f2") return 2;
    if (dtype == "<f4" || dtype == "<i4") return 4;
    if (dtype == "<f8" || dtype == "<i8") return 8;
    return dtype == "|u1" ? 1 : 0;
}
bool shape_bytes(const std::vector<uint64_t>& shape, uint64_t element, uint64_t& out) {
    out = element;
    if (shape.empty() || !element) return false;
    for (auto n : shape) if (!n || !multiply(out, n, out)) return false;
    return out <= std::numeric_limits<std::size_t>::max();
}
bool empty_kv(const StateEvidenceDescriptor& d) {
    return d.kind == "kv" && d.shape.size() == 3 && d.shape[0] == 0 && d.shape[1] && d.shape[2] &&
           d.logical_shape.empty() && width(d.dtype);
}
bool same(const StateEvidenceDescriptor& a, const StateEvidenceDescriptor& b) {
    return a.kind == b.kind && a.name == b.name && a.dtype == b.dtype && a.shape == b.shape &&
           a.layer == b.layer && a.logical_shape == b.logical_shape && a.shard_axis == b.shard_axis &&
           a.axis_indices == b.axis_indices;
}
std::string quote(const std::string& s) {
    std::string out = "\"";
    const char* hex = "0123456789abcdef";
    for (unsigned char c : s) {
        if (c == '"' || c == '\\') { out += '\\'; out += char(c); }
        else if (c < 32) { out += "\\u00"; out += hex[c >> 4]; out += hex[c & 15]; }
        else out += char(c);
    }
    return out + '"';
}
void dimensions(std::ostream& out, const std::vector<uint64_t>& shape) {
    out << '[';
    for (std::size_t i = 0; i < shape.size(); ++i) { if (i) out << ','; out << shape[i]; }
    out << ']';
}
void descriptor_json(std::ostream& out, const StateEvidenceDescriptor& d) {
    out << "\"kind\":" << quote(d.kind) << ",\"name\":" << quote(d.name)
        << ",\"dtype\":" << quote(d.dtype) << ",\"layer\":" << d.layer << ",\"shape\":";
    dimensions(out, d.shape);
    if (!d.logical_shape.empty()) {
        out << ",\"layout\":\"axis_shard\",\"logical_shape\":"; dimensions(out, d.logical_shape);
        out << ",\"shard_axis\":" << d.shard_axis << ",\"axis_indices\":"; dimensions(out, d.axis_indices);
    } else out << ",\"layout\":\"logical\"";
}
}

bool state_evidence_bytes(const StateEvidenceDescriptor& d, uint64_t& bytes, std::string& err) {
    static const std::set<std::string> kinds = {"kv", "gdn", "indexer", "ple", "residual", "metadata"};
    if (!kinds.count(d.kind) || d.name.empty() || d.layer < -1)
        return fail(err, "unknown state kind, missing name or invalid layer");
    if (empty_kv(d)) bytes=0;
    else if (!shape_bytes(d.shape, width(d.dtype), bytes)) return fail(err, "invalid dtype, shape or byte overflow");
    if (d.logical_shape.empty()) {
        if (d.shard_axis != -1 || !d.axis_indices.empty()) return fail(err, "shard metadata without logical shape");
    } else {
        uint64_t logical_bytes;
        if (!shape_bytes(d.logical_shape, width(d.dtype), logical_bytes) || d.shard_axis < 0 ||
            std::size_t(d.shard_axis) >= d.shape.size() || d.logical_shape.size() != d.shape.size() ||
            d.axis_indices.size() != d.shape[d.shard_axis]) return fail(err, "invalid shard shape");
        std::set<uint64_t> seen;
        for (std::size_t i = 0; i < d.shape.size(); ++i)
            if (int(i) != d.shard_axis && d.shape[i] != d.logical_shape[i]) return fail(err, "shard dimension mismatch");
        for (auto n : d.axis_indices)
            if (n >= d.logical_shape[d.shard_axis] || !seen.insert(n).second) return fail(err, "invalid shard ownership");
    }
    return true;
}

std::vector<StateEvidenceDescriptor> state_evidence_inventory(const std::vector<StateEvidenceArray>& arrays) {
    std::vector<StateEvidenceDescriptor> out;
    for (const auto& array : arrays) out.push_back(array.descriptor);
    return out;
}

StateEvidenceArray borrowed_state_array(StateEvidenceDescriptor d, const void* source, StateReadback readback) {
    return {std::move(d), [source, readback = std::move(readback)](void* dst, std::size_t n, std::string& err) {
        if (!source || !readback) return fail(err, "missing state source or readback");
        return readback(dst, source, n, err);
    }};
}

StateArrayRead paged_kv_state_reader(const void* pool, uint64_t slots, uint64_t page,
                                    uint64_t heads, uint64_t dim, uint64_t tokens,
                                    std::vector<int32_t> table, StateReadback readback) {
    return [=](void* dst,std::size_t n,std::string& err) {
        uint64_t physical_bytes, logical_bytes=0;
        if (!pool || !readback || !shape_bytes({slots,heads,page,dim},2,physical_bytes) ||
            (tokens && !shape_bytes({tokens,heads,dim},2,logical_bytes)) || n!=logical_bytes || (n && !dst))
            return fail(err,"invalid paged KV shape or destination");
        for(uint64_t t=0;t<tokens;++t)
            if(t/page>=table.size() || table[t/page]<0 || uint64_t(table[t/page])>=slots)
                return fail(err,"QSA missing loaded page");
        if(!tokens) return true;
        std::vector<uint8_t> physical(static_cast<std::size_t>(physical_bytes));
        if(!readback(physical.data(),pool,physical.size(),err)) return false;
        for(uint64_t t=0;t<tokens;++t) for(uint64_t h=0;h<heads;++h) {
            const uint64_t offset=((uint64_t(table[t/page])*heads+h)*page+t%page)*dim*2;
            std::memcpy(static_cast<uint8_t*>(dst)+(t*heads+h)*dim*2,physical.data()+offset,dim*2);
        }
        return true;
    };
}

bool write_state_evidence(const std::string& directory, const StateEvidenceRequest& request,
                          const std::vector<StateEvidenceArray>& arrays, std::string& err) {
    try {
        const uint16_t endian = 1;
        if (*reinterpret_cast<const uint8_t*>(&endian) != 1) return fail(err, "little endian writer required");
        if (request.request_id.empty() || request.position == UINT64_MAX ||
            (request.reset ? (request.position != 0 || request.token_count != 0) : request.token_count != request.position + 1) ||
            !request.max_bytes || request.required.empty())
            return fail(err, "missing request identity, byte cap or required inventory");
        if (arrays.size() != request.required.size()) return fail(err, "incomplete required state inventory");
        std::set<std::pair<std::string, std::string>> names;
        uint64_t total = 0;
        bool logical = true;
        for (const auto& array : arrays) {
            const auto& d = array.descriptor;
            uint64_t n;
            if (!state_evidence_bytes(d, n, err)) return false;
            if (empty_kv(d) && !request.reset) return fail(err,"empty KV is only valid immediately after reset");
            if (!names.insert({d.kind, d.name}).second || !array.read) return fail(err, "duplicate state or missing reader");
            const auto found = std::find_if(request.required.begin(), request.required.end(), [&](const auto& r) { return same(d, r); });
            if (found == request.required.end()) return fail(err, "unexpected or mismatched state metadata");
            if (n > request.max_bytes - total) return fail(err, "state snapshot exceeds explicit byte cap");
            total += n;
            logical = logical && d.logical_shape.empty();
        }
        const std::filesystem::path dir(directory);
        if(directory.empty()) return fail(err,"snapshot directory must be new");
        if(!dir.parent_path().empty()) std::filesystem::create_directories(dir.parent_path());
        if (!std::filesystem::create_directory(dir)) return fail(err, "snapshot directory must be new");
        std::ostringstream manifest;
        manifest << "{\"schema_version\":1,\"complete\":true,\"logical_complete\":" << (logical ? "true" : "false")
                 << ",\"phase\":" << quote(request.reset?"reset":"token")
                 << ",\"request_id\":" << quote(request.request_id) << ",\"position\":" << request.position
                 << ",\"token_count\":" << request.token_count << ",\"total_bytes\":" << total << ",\"required_states\":[";
        for (std::size_t i = 0; i < request.required.size(); ++i) {
            if (i) manifest << ',';
            manifest << '{'; descriptor_json(manifest, request.required[i]); manifest << '}';
        }
        manifest << "],\"states\":[";
        for (std::size_t i = 0; i < arrays.size(); ++i) {
            const auto& array = arrays[i];
            uint64_t n = 0;
            if (!state_evidence_bytes(array.descriptor, n, err)) return false;
            std::vector<uint8_t> buffer(static_cast<std::size_t>(n));
            if (!array.read(buffer.data(), buffer.size(), err)) return false;
            const auto name = "state-" + std::to_string(i) + ".bin";
            std::ofstream file(dir / name, std::ios::binary);
            file.exceptions(std::ios::failbit | std::ios::badbit);
            if (n > uint64_t(std::numeric_limits<std::streamsize>::max())) return fail(err, "array exceeds stream limit");
            if(n) file.write(reinterpret_cast<const char*>(buffer.data()), static_cast<std::streamsize>(n));
            file.close();
            if (i) manifest << ',';
            manifest << '{'; descriptor_json(manifest, array.descriptor);
            manifest << ",\"request_id\":" << quote(request.request_id) << ",\"position\":" << request.position
                     << ",\"path\":" << quote(name) << '}';
        }
        manifest << "]}\n";
        std::ofstream file(dir / "manifest.json.partial", std::ios::binary);
        file.exceptions(std::ios::failbit | std::ios::badbit);
        file << manifest.str(); file.close();
        std::filesystem::rename(dir / "manifest.json.partial",dir / "manifest.json");
        return true;
    } catch (const std::exception& ex) { return fail(err, std::string("state evidence: ") + ex.what()); }
}

bool reconstruct_state_axis(const std::vector<uint64_t>& shape, unsigned axis, std::size_t element,
                            const std::vector<StateAxisShard>& shards, std::vector<uint8_t>& output, std::string& err) {
    uint64_t n;
    if (!shape_bytes(shape, element, n) || axis >= shape.size() || shards.empty()) return fail(err, "invalid reconstruction shape");
    uint64_t inner = element, outer = 1;
    for (std::size_t i = axis + 1; i < shape.size(); ++i) if (!multiply(inner, shape[i], inner)) return fail(err, "stride overflow");
    for (unsigned i = 0; i < axis; ++i) if (!multiply(outer, shape[i], outer)) return fail(err, "stride overflow");
    std::set<uint64_t> seen;
    for (const auto& shard : shards) {
        uint64_t expected;
        if (!shard.data || shard.indices.empty() || !multiply(outer, shard.indices.size(), expected) ||
            !multiply(expected, inner, expected) || expected != shard.bytes) return fail(err, "truncated shard");
        for (auto index : shard.indices)
            if (index >= shape[axis] || !seen.insert(index).second) return fail(err, "overlapping or out of range shard");
    }
    if (seen.size() != shape[axis]) return fail(err, "incomplete shard coverage");
    std::vector<uint8_t> result(static_cast<std::size_t>(n));
    for (const auto& shard : shards)
        for (uint64_t row = 0; row < outer; ++row)
            for (std::size_t local = 0; local < shard.indices.size(); ++local)
                std::memcpy(result.data() + (row * shape[axis] + shard.indices[local]) * inner,
                            static_cast<const uint8_t*>(shard.data) + (row * shard.indices.size() + local) * inner,
                            static_cast<std::size_t>(inner));
    output.swap(result); return true;
}

#ifdef STRATA_STATE_EVIDENCE_WITH_CORE
bool describe_session_state(const core::SessionState& state, const core::ModelGeometry& g,
                            uint64_t tokens, const StateReadback& readback,
                            const std::vector<GdnStateSource>& overrides,
                            std::vector<StateEvidenceArray>& output, std::string& err) {
    try {
        if (!readback || tokens > uint64_t(state.max_cells) || state.layer_lo != 0 ||
            state.layer_hi != g.n_layers || g.qsa_interval <= 0 || g.ssm_d_conv <= 1 ||
            g.ssm_state_size <= 0 || g.ssm_v_heads <= 0 || g.ssm_conv_channels <= 0 ||
            g.n_layers <= 0 || g.hc <= 0 || g.n_embd <= 0 ||
            uint64_t(g.hc) > uint64_t(kernels::NG_HC_DIM) ||
            uint64_t(g.n_embd) != uint64_t(kernels::NG_HC_DIM)/uint64_t(g.hc) ||
            uint64_t(g.hc)*uint64_t(g.n_embd) != uint64_t(kernels::NG_HC_DIM) ||
            !state.R || !state.ple_hist || !state.ple.ready()) return fail(err, "unsupported or incomplete full session");
        std::vector<StateEvidenceArray> result;
        auto add = [&](const char* kind, std::string name, const char* dtype, std::vector<uint64_t> shape,
                       int64_t layer, const void* ptr) {
            result.push_back(borrowed_state_array({kind, std::move(name), dtype, std::move(shape), layer, {}, -1, {}}, ptr, readback));
        };
        std::set<int64_t> used;
        int64_t gd = 0, qa = 0;
        const uint64_t S = uint64_t(g.ssm_state_size), V = uint64_t(g.ssm_v_heads), C = uint64_t(g.ssm_conv_channels);
        uint64_t recurrent, convolution;
        if (!shape_bytes({S,V,S},4,recurrent) || !shape_bytes({C,uint64_t(g.ssm_d_conv-1)},4,convolution) ||
            recurrent > UINT64_MAX - convolution) return fail(err, "GDN size overflow");
        for (int64_t layer = 0; layer < g.n_layers; ++layer) {
            const std::string prefix = "layer." + std::to_string(layer) + ".";
            if (!core::is_qsa_layer(g, layer)) {
                auto found = std::find_if(overrides.begin(), overrides.end(), [&](const auto& o) { return o.layer == layer; });
                const GdnStateSource* local = found == overrides.end() ? nullptr : &*found;
                const float* rec = nullptr; const float* conv = nullptr;
                if (local) {
                    if(local->value_heads.empty()!=local->channels.empty()) return fail(err,"GDN shard requires both head and channel maps");
                    used.insert(layer); rec = local->recurrent; conv = local->convolution;
                }
                else {
                    if (!state.gdn_state || gd < state.gdn_ord0 || gd >= state.gdn_ord0 + state.gdn_alloc)
                        return fail(err, "missing global GDN allocation");
                    rec = reinterpret_cast<const float*>(reinterpret_cast<uintptr_t>(state.gdn_state) +
                          uint64_t(gd-state.gdn_ord0)*(recurrent+convolution));
                    conv = reinterpret_cast<const float*>(reinterpret_cast<uintptr_t>(rec)+recurrent);
                }
                add("gdn",prefix+"recurrent","<f4",{S,local && !local->value_heads.empty() ? local->value_heads.size():V,S},layer,rec);
                if (local && !local->value_heads.empty()) {
                    auto& d = result.back().descriptor; d.logical_shape={S,V,S}; d.shard_axis=1; d.axis_indices=local->value_heads;
                }
                // Native/fused preprocess stores each channel's history contiguously.
                add("gdn",prefix+"convolution","<f4",{local && !local->channels.empty() ? local->channels.size():C,uint64_t(g.ssm_d_conv-1)},layer,conv);
                if (local && !local->channels.empty()) {
                    auto& d=result.back().descriptor; d.logical_shape={C,uint64_t(g.ssm_d_conv-1)}; d.shard_axis=0; d.axis_indices=local->channels;
                }
                ++gd; continue;
            }
            if (!state.qsa_states || qa < state.qsa_ord0 || qa >= state.qsa_ord0 + state.qsa_alloc)
                return fail(err, "missing QSA allocation");
            const auto& q = state.qsa_states[qa++];
            const auto s = core::qsa_shapes(g);
            if (q.kv_mode != 0 || q.kv_int8 || q.kv_q4 || q.kv_hybrid || q.kv_elastic >= 0 ||
                s.page_size <= 0 || s.n_head_kv <= 0 || s.head_dim <= 0 || q.n_pages <= 0 ||
                q.n_slots <= 0 || !q.page_table || !q.k_pool || !q.v_pool || tokens > uint64_t(q.max_cells))
                return fail(err, "QSA evidence requires resident plain FP16 pools");
            std::vector<int32_t> table(static_cast<std::size_t>(q.n_pages));
            if (!readback(table.data(),q.page_table,table.size()*sizeof(int32_t),err)) return false;
            const uint64_t page = uint64_t(s.page_size), heads=uint64_t(s.n_head_kv), dim=uint64_t(s.head_dim);
            for (uint64_t t=0;t<tokens;++t)
                if (t/page>=table.size() || table[t/page]<0 || table[t/page]>=q.n_slots) return fail(err,"QSA missing loaded page");
            for (int kv=0;kv<2;++kv) {
                const auto* pool=kv?q.v_pool:q.k_pool;
                StateEvidenceDescriptor d{"kv",prefix+(kv?"v":"k"),"<f2",{tokens,heads,dim},layer,{},-1,{}};
                result.push_back({d,paged_kv_state_reader(pool,uint64_t(q.n_slots),page,heads,dim,tokens,table,readback)});
            }
            add("metadata",prefix+"page_table","<i4",{uint64_t(q.n_pages)},layer,q.page_table);
            if(tokens) add("metadata",prefix+"step","<i4",{uint64_t(kernels::kStepCount)},layer,q.step);
            if (g.has_indexer()) {
                if (s.idx_block<=1 || s.idx_dim<=0 || q.idx_pooled_rows<=0) return fail(err,"invalid indexer shape");
                add("indexer",prefix+"tail","<f4",{uint64_t(s.idx_block-1),uint64_t(s.idx_dim)},layer,q.idx_tail);
                add("indexer",prefix+"dead","<f4",{uint64_t(s.idx_dim)},layer,q.idx_dead);
                add("indexer",prefix+"pooled","<f4",{uint64_t(q.idx_pooled_rows),uint64_t(s.idx_dim)},layer,q.idx_pooled);
                if(tokens) add("metadata",prefix+"indexer_block_pos","<i4",{1},layer,q.idx_block_pos);
            }
        }
        if (used.size()!=overrides.size()) return fail(err,"duplicate or unknown GDN override layer");
        add("ple","ple.history","<f4",{uint64_t(kernels::NG_HC_DIM),uint64_t(kernels::NG_HIST)},1,state.ple_hist);
        // Host token metadata is copied by value so later tokens cannot mutate the capture.
        const std::vector<int32_t> window={state.ple_prev[0],state.ple_prev[1],state.ple_token};
        result.push_back({{"ple","ple.token_window","<i4",{3},1,{},-1,{}},[window](void* dst,std::size_t n,std::string& error) {
            if(n!=window.size()*sizeof(int32_t)) return fail(error,"PLE token window size mismatch");
            std::memcpy(dst,window.data(),n);return true;
        }});
        add("residual","residual.R","<f4",{uint64_t(g.hc),uint64_t(g.n_embd)},-1,state.R);
        for (const auto& array:result) { uint64_t n; if(!state_evidence_bytes(array.descriptor,n,err)) return false; }
        output.swap(result); return true;
    } catch(const std::exception& ex) { return fail(err,std::string("describe state: ")+ex.what()); }
}
#endif
} // namespace strata::distributed
