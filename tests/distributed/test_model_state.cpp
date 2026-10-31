// CPU with CUDA headers, no CUDA runtime: g++ -std=c++20 -Wall -Wextra -Werror -Iinclude -Isrc -I/usr/local/cuda/include tests/distributed/test_model_state.cpp src/distributed/{model_state,model_checkpoint,state_evidence}.cpp -o /tmp/test_model_state
#include "distributed/model_state.hpp"
#include "strata/kernels/ngram.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <iostream>

using namespace strata;
namespace {
void check(bool value, const char* message) {
    if (!value) { std::cerr << "FAIL: " << message << '\n'; std::exit(1); }
}
auto deadline() { return std::chrono::steady_clock::now() + std::chrono::seconds(5); }
}

int main() {
    core::ModelGeometry g;
    g.n_layers = 2; g.qsa_interval = 2; g.ssm_state_size = 2;
    g.ssm_v_heads = 3; g.ssm_conv_channels = 6; g.ssm_d_conv = 3;
    g.n_head = 2; g.n_head_kv = 2; g.head_dim = 2;
    g.idx_block = 4; g.idx_key_dim = 2;
    core::SessionState state;
    core::QsaState q;
    state.max_cells = 12; state.layer_hi = 2; state.qsa_alloc = 1; state.qsa_states = &q;
    std::vector<float> recurrent(8), convolution(8), residual(10240), history(10240 * kernels::NG_HIST);
    state.R = residual.data(); state.ple_hist = history.data();
    std::vector<float> tail(6), dead(2), pooled(10);
    std::vector<uint16_t> keys(12 * 2 * 2), values(keys.size());
    std::array<int32_t, 3> table{0, 1, 2};
    std::array<int32_t, 4> step{};
    std::array<int32_t, 5> host_step{};
    std::array<int32_t, 2> pos{}, host_pos{};
    int32_t block_pos = 0, status = 0;
    q.max_cells = state.max_cells; q.n_pages = q.n_slots = 3; q.idx_pooled_rows = 5;
    q.page_table = table.data(); q.k_pool = keys.data(); q.v_pool = values.data();
    q.step = step.data(); q.host_step = host_step.data(); q.pos_dev = pos.data(); q.host_pos = host_pos.data();
    q.attention_status = &status; q.idx_block_pos = &block_pos;
    q.idx_tail = tail.data(); q.idx_dead = dead.data(); q.idx_pooled = pooled.data();
    std::vector<distributed::GdnStateSource> sources{{0, recurrent.data(), convolution.data(), {0, 2}, {0, 2, 3, 5}}};
    unsigned copies = 0;
    const distributed::detail::ModelStateCopy copy = [&](void* h, void* dev, size_t n, bool to_device, std::string&) {
        ++copies;
        // Transfers into these pools must never read the unused physical rows.
        if (reinterpret_cast<uintptr_t>(dev) >= reinterpret_cast<uintptr_t>(keys.data()) &&
            reinterpret_cast<uintptr_t>(dev) < reinterpret_cast<uintptr_t>(keys.data() + keys.size()))
            check(static_cast<uint8_t*>(dev) + n <= reinterpret_cast<uint8_t*>(keys.data() + 8 * 2 * 2), "loaded key pages only");
        if (to_device) std::memcpy(dev, h, n); else std::memcpy(h, dev, n);
        return true;
    };
    distributed::detail::ModelStateArrays arrays;
    std::string err;
    const uint64_t tokens = 5;
    for (size_t i = 0; i < keys.size(); ++i) { keys[i] = uint16_t(100 + i); values[i] = uint16_t(900 + i); }
    std::fill(recurrent.begin(), recurrent.end(), 3.25f);
    std::fill(convolution.begin(), convolution.end(), -2.5f);
    std::fill(history.begin(), history.end(), 8.5f);
    std::fill(residual.begin(), residual.end(), 4.75f);
    std::fill(pooled.begin(), pooled.end(), 99.0f);
    state.ple_prev[0] = 17; state.ple_prev[1] = 42; state.ple_token = 42;
    step = {4, 5, 1, 5}; host_step = {4, 5, 1, 5, 0}; pos = {12, 12}; host_pos = pos;
    block_pos = 3;
    check(distributed::detail::describe_checkpoint_state(state, g, sources, tokens, copy, arrays, err), "real session inventory described");
    check(arrays.sources.size() == 17 && copies == 1, "complete inventory validates one page table");
    auto get = [&](const std::string& name) -> distributed::StateEvidenceArray& {
        const auto it = std::find_if(arrays.sources.begin(), arrays.sources.end(), [&](const auto& a) { return a.descriptor.name == name; });
        check(it != arrays.sources.end(), "named real state present"); return *it;
    };
    std::vector<uint16_t> logical(5 * 2 * 2);
    check(get("layer.1.k").read(logical.data(), logical.size() * 2, err), "logical KV reads");
    // Independent physical-layout oracle across a full page and partial page.
    for (size_t t = 0; t < 5; ++t) for (size_t h = 0; h < 2; ++h) for (size_t d = 0; d < 2; ++d)
        check(logical[(t * 2 + h) * 2 + d] == keys[(((t / 4) * 2 + h) * 4 + t % 4) * 2 + d], "paged KV layout matches oracle");

    distributed::ModelCheckpoint checkpoint;
    distributed::ModelCheckpointIdentity id{"pinned-model", "slot-request", 2, 0, 0, 12, 0};
    check(checkpoint.capture({id, tokens, 1024 * 1024, distributed::state_evidence_inventory(arrays.sources)}, arrays.sources, deadline(), err), "real state captured");
    const auto saved_keys = keys;
    std::fill(keys.begin(), keys.end(), uint16_t(65535));
    std::fill(recurrent.begin(), recurrent.end(), 0.0f);
    std::fill(convolution.begin(), convolution.end(), 0.0f);
    std::fill(history.begin(), history.end(), 0.0f);
    std::fill(residual.begin(), residual.end(), 0.0f);
    std::fill(pooled.begin(), pooled.end(), 0.0f);
    state.ple_prev[0] = state.ple_prev[1] = state.ple_token = -1;
    step.fill(0); host_step.fill(0); pos.fill(0); host_pos.fill(0); block_pos = 0;
    check(distributed::detail::describe_checkpoint_state(state, g, sources, tokens, copy, arrays, err), "restore independently describes real destination");
    std::reverse(arrays.destinations.begin(), arrays.destinations.end());
    uint64_t committed = 0; bool invalid = false;
    check(checkpoint.restore({id, distributed::state_evidence_inventory(arrays.sources), arrays.destinations,
        [&](uint64_t p, std::string&) { committed = p; return true; }, [&] { invalid = true; }}, deadline(), err), "real state restores independent of order");
    check(!invalid && committed == tokens && recurrent[0] == 3.25f && convolution[0] == -2.5f && history.back() == 8.5f &&
          residual.back() == 4.75f && pooled.back() == 99.0f && state.ple_prev[0] == 17 && state.ple_token == 42 &&
          step[1] == 5 && host_step[1] == 5 && pos[0] == 12 && host_pos[1] == 12 && block_pos == 3, "complete actual state roundtrip");
    for (size_t t = 0; t < 5; ++t) for (size_t h = 0; h < 2; ++h) for (size_t d = 0; d < 2; ++d) {
        const auto index = (((t / 4) * 2 + h) * 4 + t % 4) * 2 + d;
        check(keys[index] == saved_keys[index], "loaded KV restored byte-exactly");
    }
    check(keys.back() == 65535, "unused KV capacity not copied");
    table[1] = 2;
    check(!distributed::detail::describe_checkpoint_state(state, g, sources, tokens, copy, arrays, err), "nonidentity mapping rejected");
    table[1] = 1;
    q.kv_int8 = true;
    check(!distributed::detail::describe_checkpoint_state(state, g, sources, tokens, copy, arrays, err), "compressed KV rejected");
    q.kv_int8 = false; q.kv_mode = 1;
    check(!distributed::detail::describe_checkpoint_state(state, g, sources, tokens, copy, arrays, err), "streamed KV rejected");
    q.kv_mode = 0;
    sources.push_back(sources.front());
    check(!distributed::detail::describe_checkpoint_state(state, g, sources, tokens, copy, arrays, err), "duplicate GDN state rejected");
    sources.pop_back();
    const distributed::detail::ModelStateCopy failing_copy = [](void*, void*, size_t, bool, std::string& error) {
        error = "injected bounded CUDA transfer failure"; return false;
    };
    check(!distributed::detail::describe_checkpoint_state(state, g, sources, tokens, failing_copy, arrays, err) &&
          arrays.sources.empty() && arrays.destinations.empty(), "read failure publishes no partial state inventory");
    auto missing = sources; missing.clear();
    check(!distributed::detail::describe_checkpoint_state(state, g, missing, tokens, copy, arrays, err), "missing local GDN state rejected");
    step.fill(0); host_step.fill(0); pos.fill(0); host_pos.fill(0); block_pos = status = 0;
    check(distributed::detail::describe_checkpoint_state(state, g, sources, 0, copy, arrays, err), "reset state includes metadata at position zero");
    check(checkpoint.capture({id, 0, 1024 * 1024, distributed::state_evidence_inventory(arrays.sources)}, arrays.sources, deadline(), err), "reset checkpoint captures");
    step.fill(19); host_step.fill(19); pos.fill(19); host_pos.fill(19); block_pos = status = 19;
    check(checkpoint.restore({id, distributed::state_evidence_inventory(arrays.sources), arrays.destinations,
        [](uint64_t, std::string&) { return true; }, [&] { invalid = true; }}, deadline(), err), "reset checkpoint restores");
    check(step[0] == 0 && host_step[0] == 0 && pos[0] == 0 && host_pos[0] == 0 && block_pos == 0 && status == 0,
          "reset control metadata has no stale values");
    std::cout << "real model state checkpoint tests passed\n";
}
