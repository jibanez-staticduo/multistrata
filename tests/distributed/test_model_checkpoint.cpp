// CPU: g++ -std=c++20 -Wall -Wextra -Werror -Iinclude tests/distributed/test_model_checkpoint.cpp src/distributed/{model_checkpoint,state_evidence}.cpp -o /tmp/test_model_checkpoint
#include "strata/distributed/model_checkpoint.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>

using namespace strata::distributed;
namespace {
void check(bool result, const char* message) {
    if (!result) { std::cerr << "FAIL: " << message << '\n'; std::exit(1); }
}
CheckpointDeadline future() { return std::chrono::steady_clock::now() + std::chrono::seconds(5); }
struct Fixture {
    std::vector<StateEvidenceDescriptor> descriptors = {
        {"gdn", "layer.0.recurrent", "<f4", {2, 1, 2}, 0, {2, 3, 2}, 1, {1}},
        {"gdn", "layer.0.convolution", "<f4", {2, 2}, 0, {6, 2}, 0, {2, 3}},
        {"kv", "layer.3.k", "<f2", {2, 1, 2}, 3, {}, -1, {}},
        {"kv", "layer.3.v", "<f2", {2, 1, 2}, 3, {}, -1, {}},
        {"metadata", "layer.3.page_table", "<i4", {2}, 3, {}, -1, {}},
        {"indexer", "layer.3.tail", "<f4", {2, 2}, 3, {}, -1, {}},
        {"indexer", "layer.3.dead", "<f4", {2}, 3, {}, -1, {}},
        {"indexer", "layer.3.pooled", "<f4", {3, 2}, 3, {}, -1, {}},
        {"metadata", "layer.3.indexer_block_pos", "<i4", {1}, 3, {}, -1, {}},
        {"ple", "ple.history", "<f4", {2, 3}, 1, {}, -1, {}},
        {"ple", "ple.token_window", "<i4", {3}, 1, {}, -1, {}},
        {"residual", "residual.R", "<f4", {2, 2}, -1, {}, -1, {}}
    };
    std::vector<std::vector<uint8_t>> live, original;
    uint64_t position = 2;
    unsigned writes = 0, invalidations = 0;
    ModelCheckpointRequest request;
    Fixture() {
        request.identity = {"frozen-model-sha", "request-a", 9, 3, 1, 65536, 8};
        request.next_position = position;
        request.max_bytes = 4096;
        request.required = descriptors;
        std::string err;
        for (std::size_t i = 0; i < descriptors.size(); ++i) {
            uint64_t n = 0;
            check(state_evidence_bytes(descriptors[i], n, err), "fixture uses valid evidence descriptors");
            live.emplace_back(static_cast<std::size_t>(n), static_cast<uint8_t>(i + 1));
        }
        original = live;
    }
    std::vector<StateEvidenceArray> readers() {
        std::vector<StateEvidenceArray> result;
        for (std::size_t i = 0; i < descriptors.size(); ++i)
            result.push_back({descriptors[i], [this, i](void* dst, std::size_t n, std::string&) {
                if (n != live[i].size()) return false;
                if (n) std::memcpy(dst, live[i].data(), n);
                return true;
            }});
        return result;
    }
    ModelCheckpointDestination destination() {
        ModelCheckpointDestination result;
        result.identity = request.identity;
        result.required = descriptors;
        for (std::size_t i = 0; i < descriptors.size(); ++i)
            result.arrays.push_back({descriptors[i], [this, i](const void* src, std::size_t n, std::string&) {
                ++writes;
                if (n != live[i].size()) return false;
                if (n) std::memcpy(live[i].data(), src, n);
                return true;
            }});
        result.commit_position = [this](uint64_t p, std::string&) {
            if (writes != descriptors.size()) return false;
            position = p;
            return true;
        };
        result.invalidate = [this] { ++invalidations; };
        return result;
    }
};
}

int main() {
    std::string err;
    Fixture f;
    ModelCheckpoint checkpoint;
    check(!checkpoint.restore(f.destination(), future(), err), "incomplete checkpoint rejected");
    check(checkpoint.capture(f.request, f.readers(), future(), err), "complete logical state captured");
    check(checkpoint.complete() && checkpoint.next_position() == 2 && checkpoint.bytes() > 0, "capture publishes metadata");
    for (auto& a : f.live) std::fill(a.begin(), a.end(), uint8_t(255));
    f.position = 6;
    auto dst = f.destination();
    std::reverse(dst.arrays.begin(), dst.arrays.end());
    check(checkpoint.restore(dst, future(), err), "restore is independent of callback order");
    check(f.live == f.original && f.position == 2 && !f.invalidations, "all request state round trips byte exactly");

    Fixture other;
    auto other_dst = other.destination();
    other_dst.identity.request_id = "request-b";
    check(!checkpoint.restore(other_dst, future(), err) && !other.writes, "request isolation before writes");
    other_dst = other.destination(); other_dst.identity.slot++;
    check(!checkpoint.restore(other_dst, future(), err) && !other.writes, "slot isolation");
    other_dst = other.destination(); other_dst.identity.generation++;
    check(!checkpoint.restore(other_dst, future(), err) && !other.writes, "stale slot generation rejected");
    other_dst = other.destination(); other_dst.identity.model_id = "different-model";
    check(!checkpoint.restore(other_dst, future(), err) && !other.writes, "model identity isolation");
    other_dst = other.destination(); other_dst.identity.rank = 0;
    check(!checkpoint.restore(other_dst, future(), err) && !other.writes, "rank-local state isolation");
    other_dst = other.destination(); other_dst.identity.position_base++;
    check(!checkpoint.restore(other_dst, future(), err) && !other.writes, "RoPE position-base isolation");
    other_dst = other.destination(); other_dst.arrays.pop_back();
    check(!checkpoint.restore(other_dst, future(), err) && !other.writes, "missing state rejected before mutation");
    other_dst = other.destination(); other_dst.arrays[0].descriptor.axis_indices = {2};
    check(!checkpoint.restore(other_dst, future(), err) && !other.writes, "same shape different shard rejected");
    other_dst = other.destination(); other_dst.arrays[0].descriptor.shape = {1, 1, 4};
    check(!checkpoint.restore(other_dst, future(), err) && !other.writes, "same byte size different shape rejected");
    other_dst = other.destination(); other_dst.arrays[0] = other_dst.arrays[1];
    check(!checkpoint.restore(other_dst, future(), err) && !other.writes, "duplicate destination rejected");
    other_dst = other.destination(); other_dst.arrays.back().write = {};
    check(!checkpoint.restore(other_dst, future(), err) && !other.writes, "missing writer preflighted");
    other_dst = other.destination(); other_dst.invalidate = {};
    check(!checkpoint.restore(other_dst, future(), err) && !other.writes, "mandatory invalidation callback");

    auto failing = other.destination();
    failing.arrays[1].write = [](const void*, std::size_t, std::string&) { return false; };
    other.position = 8;
    check(!checkpoint.restore(failing, future(), err) && other.invalidations == 1 && other.position == 8,
          "partial write failure invalidates destination without committing position");
    other.writes = 0;
    failing = other.destination();
    failing.commit_position = [](uint64_t, std::string&) { return false; };
    check(!checkpoint.restore(failing, future(), err) && other.invalidations == 2, "commit failure invalidates destination");
    other.writes = 0;
    failing = other.destination();
    failing.arrays[0].write = [](const void*, std::size_t, std::string&) -> bool { throw std::runtime_error("injected"); };
    check(!checkpoint.restore(failing, future(), err) && other.invalidations == 3, "throwing writer invalidates destination");
    check(checkpoint.complete(), "failed restore retains reusable immutable source");

    auto bad = f.request;
    bad.max_bytes = checkpoint.bytes() - 1;
    check(!checkpoint.capture(bad, f.readers(), future(), err) && !checkpoint.complete(), "failed recapture clears previous snapshot");
    check(checkpoint.bytes() == 0, "failed capture publishes no bytes");
    auto sources = f.readers(); sources.pop_back();
    check(!checkpoint.capture(f.request, sources, future(), err), "missing source rejected");
    sources = f.readers(); sources[0] = sources[1];
    check(!checkpoint.capture(f.request, sources, future(), err), "duplicate source rejected");
    bad = f.request; bad.required[0] = bad.required[1];
    check(!checkpoint.capture(bad, f.readers(), future(), err), "duplicate required state rejected");
    bad = f.request; bad.next_position = 3;
    check(!checkpoint.capture(bad, f.readers(), future(), err), "loaded KV token count validated");
    bad = f.request; bad.next_position = 65537;
    check(!checkpoint.capture(bad, f.readers(), future(), err), "context overflow rejected");
    bad = f.request; bad.identity.position_base = -1;
    check(!checkpoint.capture(bad, f.readers(), future(), err), "negative position base rejected");
    bad = f.request; bad.identity.position_base = 65535;
    check(!checkpoint.capture(bad, f.readers(), future(), err), "RoPE position plus base cannot exceed context");
    sources = f.readers(); sources.back().read = [](void*, std::size_t, std::string&) { return false; };
    check(!checkpoint.capture(f.request, sources, future(), err) && !checkpoint.complete(), "late read failure leaves no complete capture");
    sources = f.readers(); sources[0].read = {};
    check(!checkpoint.capture(f.request, sources, future(), err), "missing reader rejected");
    sources = f.readers(); sources[0].read = [](void*, std::size_t, std::string&) -> bool { throw 7; };
    check(!checkpoint.capture(f.request, sources, future(), err), "throwing capture callback contained");
    check(!checkpoint.capture(f.request, f.readers(), std::chrono::steady_clock::now(), err), "expired capture rejected");
    check(checkpoint.capture(f.request, f.readers(), future(), err), "capture after failure can retry");
    other.writes = 0;
    check(!checkpoint.restore(other.destination(), std::chrono::steady_clock::now(), err) && !other.writes,
          "expired restore rejected before mutation");
    auto late = other.destination();
    const auto until = std::chrono::steady_clock::now() + std::chrono::milliseconds(100);
    late.arrays[0].write = [until](const void*, std::size_t, std::string&) {
        while (std::chrono::steady_clock::now() < until) {}
        return true;
    };
    check(!checkpoint.restore(late, until, err) &&
          other.invalidations == 4, "deadline consumed by write invalidates partial restore");

    std::array<ModelCheckpoint, 8> slots;
    for (std::size_t i = 0; i < slots.size(); ++i) {
        other.request.identity.slot = static_cast<uint32_t>(i);
        other.request.identity.request_id = "slot-request-" + std::to_string(i);
        std::fill(other.live[0].begin(), other.live[0].end(), static_cast<uint8_t>(40 + i));
        check(slots[i].capture(other.request, other.readers(), future(), err), "eight independent request slots captured");
    }
    for (std::size_t i = slots.size(); i-- > 0;) {
        other.request.identity.slot = static_cast<uint32_t>(i);
        other.request.identity.request_id = "slot-request-" + std::to_string(i);
        other.writes = 0;
        check(slots[i].restore(other.destination(), future(), err), "interleaved slots restore independently");
        check(other.live[0][0] == 40 + i, "each slot retains its own recurrence");
    }

    Fixture reset;
    reset.request.next_position = 0;
    for (std::size_t i : {std::size_t(2), std::size_t(3)}) {
        reset.descriptors[i].shape[0] = 0;
        reset.live[i].clear();
    }
    reset.request.required = reset.descriptors;
    check(checkpoint.capture(reset.request, reset.readers(), future(), err), "reset supports explicitly empty logical KV");
    check(checkpoint.restore(reset.destination(), future(), err) && reset.position == 0, "reset state restores at next position zero");
    checkpoint.clear();
    check(!checkpoint.complete() && checkpoint.bytes() == 0, "explicit clear invalidates snapshot");
    std::cout << "model checkpoint tests passed\n";
}
