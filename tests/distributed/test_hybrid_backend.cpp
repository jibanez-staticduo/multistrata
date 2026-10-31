#include "strata/distributed/hybrid_backend.hpp"

#include <array>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace strata::distributed;

namespace {

void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

struct FakeBackend final : Backend {
    uint32_t rank = 0;
    int reduces = 0;
    int gathers = 0;
    int polls = 0;
    int aborts = 0;
    bool async_failure = false;
    Failure enqueue_failure{};
    Failure poll_failure{};

    uint32_t world_size() const override { return kTpWorld; }
    uint32_t local_rank() const override { return rank; }
    Failure raw_all_reduce_sum(const DeviceBuffer&, const DeviceBuffer&, uint64_t, cudaStream_t) override {
        ++reduces;
        return enqueue_failure;
    }
    Failure raw_all_gather(const DeviceBuffer&, const DeviceBuffer&, uint64_t, cudaStream_t) override {
        ++gathers;
        return enqueue_failure;
    }
    Failure raw_poll(Deadline) override {
        ++polls;
        return poll_failure;
    }
    bool raw_async_failure(Failure& out) const override {
        if (!async_failure) return false;
        out = Failure::fail(FailureCode::async_error, "fake async failure");
        return true;
    }
    void raw_abort() noexcept override { ++aborts; }
};

void test_dispatch() {
    FakeBackend nccl, roce;
    HybridBackend hybrid(nccl, roce);
    alignas(16) std::array<uint8_t, 128> input{}, output{};
    const auto stream = reinterpret_cast<cudaStream_t>(uintptr_t(1));
    check(!hybrid.raw_all_gather({input.data(), 4, ScalarType::fp32}, {output.data(), 12, ScalarType::fp32},
                                 4, stream), "eligible FP32 gather failed to enqueue");
    check(roce.gathers == 1 && nccl.gathers == 0, "eligible gather did not use RoCE");
    check(!hybrid.raw_poll({}), "eligible gather failed to complete");

    check(!hybrid.raw_all_gather({input.data(), 2, ScalarType::fp32}, {output.data(), 6, ScalarType::fp32},
                                 2, stream), "ineligible FP32 gather failed to enqueue");
    check(nccl.gathers == 1 && roce.gathers == 1, "ineligible gather did not use NCCL");
    check(!hybrid.raw_poll({}), "ineligible gather failed to complete");

    check(!hybrid.raw_all_reduce_sum({input.data(), 4, ScalarType::fp32},
                                     {output.data(), 4, ScalarType::fp32}, 4, stream),
          "FP32 reduction failed to enqueue");
    check(nccl.reduces == 1 && roce.reduces == 0, "reduction did not stay on NCCL");
    check(!hybrid.raw_poll({}), "reduction failed to complete");
    check(nccl.aborts == 0 && roce.aborts == 0, "healthy operations aborted a backend");
}

void test_rejection_and_terminal_failure() {
    alignas(16) std::array<uint8_t, 128> input{}, output{};
    const auto stream = reinterpret_cast<cudaStream_t>(uintptr_t(2));
    {
        FakeBackend nccl, roce;
        HybridBackend hybrid(nccl, roce);
        check(hybrid.raw_all_gather({input.data() + 1, 4, ScalarType::fp32},
                                    {output.data(), 12, ScalarType::fp32}, 4, stream).code ==
                  FailureCode::bad_pointer,
              "unaligned eligible gather was not rejected");
        check(nccl.gathers == 0 && roce.gathers == 0, "unaligned call was silently rerouted");
        check(nccl.aborts == 1 && roce.aborts == 1, "invalid hybrid call did not terminate both backends");
    }
    {
        FakeBackend nccl, roce;
        roce.enqueue_failure = Failure::fail(FailureCode::enqueue_failed, "RoCE enqueue failed");
        HybridBackend hybrid(nccl, roce);
        const auto failure = hybrid.raw_all_gather({input.data(), 4, ScalarType::fp32},
                                                    {output.data(), 12, ScalarType::fp32}, 4, stream);
        check(failure.code == FailureCode::enqueue_failed, "RoCE enqueue failure was hidden");
        check(nccl.gathers == 0, "failed RoCE operation fell back to NCCL");
        check(nccl.aborts == 1 && roce.aborts == 1, "enqueue failure did not abort both backends");
        check(hybrid.raw_all_gather({input.data(), 4, ScalarType::fp32},
                                    {output.data(), 12, ScalarType::fp32}, 4, stream).code ==
                  FailureCode::enqueue_failed,
              "terminal failure was not sticky");
    }
    {
        FakeBackend nccl, roce;
        HybridBackend hybrid(nccl, roce);
        check(!hybrid.raw_all_gather({input.data(), 4, ScalarType::fp32},
                                     {output.data(), 12, ScalarType::fp32}, 4, stream),
              "RoCE enqueue failed before async test");
        roce.async_failure = true;
        Failure out;
        check(hybrid.raw_async_failure(out) && out.code == FailureCode::async_error,
              "RoCE asynchronous error was not observed");
        hybrid.raw_abort();
        check(nccl.aborts == 1 && roce.aborts == 1, "abort did not cover both backends");
    }
}

void test_bootstrap_codec() {
    HybridBootstrap original;
    original.nccl_id_hex.assign(256, 'a');
    original.roce_blob = {1, 2, 3, 4, 5};
    std::string encoded;
    check(!encode_hybrid_bootstrap(original, encoded), "bootstrap encode failed");
    HybridBootstrap decoded;
    check(!decode_hybrid_bootstrap(encoded, decoded), "bootstrap decode failed");
    check(decoded.nccl_id_hex == original.nccl_id_hex && decoded.roce_blob == original.roce_blob,
          "bootstrap round trip changed fields");

    HybridBootstrap follower;
    follower.roce_blob = {6, 7, 8, 9, 10};
    std::string follower_encoded;
    check(!encode_hybrid_bootstrap(follower, follower_encoded), "follower bootstrap encode failed");
    std::array<std::string, kTpWorld> roster{encoded, follower_encoded, follower_encoded};
    std::string nccl_id;
    std::vector<uint8_t> blobs;
    check(!decode_hybrid_bootstrap_roster(roster, nccl_id, blobs), "valid bootstrap roster failed");
    check(nccl_id == original.nccl_id_hex && blobs == std::vector<uint8_t>({1, 2, 3, 4, 5,
                                                                            6, 7, 8, 9, 10,
                                                                            6, 7, 8, 9, 10}),
          "bootstrap roster was not rank ordered");

    roster[2] = "bad-version";
    check(decode_hybrid_bootstrap_roster(roster, nccl_id, blobs).code == FailureCode::protocol_version,
          "invalid bundle version was accepted");
    roster[2] = follower_encoded;
    roster[2].back() = '#';
    check(static_cast<bool>(decode_hybrid_bootstrap_roster(roster, nccl_id, blobs)),
          "corrupt bundle payload was accepted");
    roster[2] = follower_encoded;
    roster[2].append(kHybridBootstrapFieldMax + 1 - roster[2].size(), 'x');
    check(decode_hybrid_bootstrap_roster(roster, nccl_id, blobs).code == FailureCode::oversized_frame,
          "oversized bootstrap bundle was accepted");
}

void test_gather_shape_gate() {
    check(hybrid_roce_gather_eligible(ScalarType::fp32, 4), "eligible FP32 extent was rejected");
    check(hybrid_roce_gather_eligible(ScalarType::bf16, 8), "eligible BF16 extent was rejected");
    check(!hybrid_roce_gather_eligible(ScalarType::fp32, 3), "unaligned byte extent was eligible");
    check(!hybrid_roce_gather_eligible(ScalarType::fp32, 4, 15), "oversized extent was eligible");
    check(!hybrid_roce_gather_eligible(static_cast<ScalarType>(99), 8), "unknown dtype was eligible");
}

}  // namespace

int main() {
    try {
        test_dispatch();
        test_rejection_and_terminal_failure();
        test_bootstrap_codec();
        test_gather_shape_gate();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
    std::cout << "hybrid backend CPU contract passed\n";
    return EXIT_SUCCESS;
}
