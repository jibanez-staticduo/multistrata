// CPU-only checks for the Session RDMA-control adapter contract.
#include "strata/distributed/session.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <array>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {
using namespace strata::distributed;

constexpr const char* kUuid = "12345678-1234-1234-1234-123456789abc";
constexpr const char* kCredential = "rdma-control-test-credential";

struct SharedGather {
    explicit SharedGather(uint32_t count) : world(count) {}

    struct Round {
        explicit Round(uint32_t count) : values(count), seen(count, false) {}
        std::vector<std::string> values;
        std::vector<bool> seen;
        uint32_t arrived = 0;
    };

    Failure gather(uint32_t rank, std::string_view input, std::vector<std::string>& output) {
        std::unique_lock lock(mutex);
        const uint64_t round_index = calls[rank]++;
        while (round_index >= rounds.size()) rounds.emplace_back(world);
        Round& round = rounds[round_index];
        if (round.seen[rank]) return Failure::fail(FailureCode::bad_operation, "duplicate rank in mock gather");
        round.values[rank].assign(input);
        round.seen[rank] = true;
        if (++round.arrived == world) {
            condition.notify_all();
        } else if (!condition.wait_for(lock, std::chrono::seconds(2), [&] { return round.arrived == world; })) {
            return Failure::fail(FailureCode::deadline_expired, "mock gather timed out waiting for ranks");
        }
        output = round.values;
        return ok_failure();
    }

    uint32_t world;
    std::mutex mutex;
    std::condition_variable condition;
    std::array<uint64_t, kTpWorld> calls{};
    std::deque<Round> rounds;
    int32_t alter_rank = -1;
    size_t alter_offset = 0;
    bool truncate_rank_two = false;
    bool alter_ack_rank_two = false;
};

struct Fixture {
    SystemClock clock;
    std::array<std::unique_ptr<ControlChannel>, kTpWorld> channels;
    std::array<std::unique_ptr<Session>, kTpWorld> sessions;
    SharedGather gather{kTpWorld};

    Fixture() {
        int fd = ::socket(AF_INET, SOCK_STREAM, 0);
        uint16_t port = 0;
        if (fd >= 0) {
            sockaddr_in address{};
            address.sin_family = AF_INET;
            address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
            if (::bind(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) == 0) {
                socklen_t length = sizeof(address);
                if (::getsockname(fd, reinterpret_cast<sockaddr*>(&address), &length) == 0)
                    port = ntohs(address.sin_port);
            }
            ::close(fd);
        }
        std::array<Failure, kTpWorld> start_results{};
        for (uint32_t rank = 0; rank < kTpWorld; ++rank) {
            SessionConfig config;
            config.rank = rank;
            config.world = kTpWorld;
            config.session_uuid = kUuid;
            config.credential = Credential(kCredential);
            config.approved_nccl_version = "test-nccl";
            config.approved_library_provenance = "test-provenance";
            config.budgets = PhaseBudgets{5000, 5000, 5000, 5000};
            TcpChannelConfig wire;
            wire.rank = rank;
            wire.self = wire.rank0 = TcpEndpoint{"127.0.0.1", port};
            wire.credential = kCredential;
            std::string error;
            channels[rank] = make_tcp_control_channel(wire, clock, error);
            if (!channels[rank]) std::printf("test channel setup failed: %s\n", error.c_str());
            JoinInfo join;
            join.session_uuid = kUuid;
            join.rank = rank;
            join.world = kTpWorld;
            join.protocol_version = kControlProtocolVersion;
            join.model_hash = "test-model";
            join.config_fingerprint = "test-config";
            join.backend_identity = "nccl";
            join.nccl_version_expected = "test-nccl";
            join.library_provenance = "test-provenance";
            join.nccl_bootstrap_id = "test-bootstrap-" + std::to_string(rank);
            sessions[rank] = std::make_unique<Session>(std::move(config), std::move(join), *channels[rank],
                                                        clock, [](std::string_view) {});
        }
        std::array<std::thread, kTpWorld> starters;
        for (uint32_t rank = 0; rank < kTpWorld; ++rank)
            starters[rank] = std::thread([&, rank] { start_results[rank] = sessions[rank]->start(); });
        for (auto& thread : starters) thread.join();
        for (uint32_t rank = 0; rank < kTpWorld; ++rank) {
            if (start_results[rank]) {
                std::printf("fixture session start failed on rank %u: %s\n", rank, start_results[rank].detail.c_str());
                continue;
            }
            auto* shared = &gather;
            auto f = sessions[rank]->enable_control_exchange(
                [shared, rank, calls = uint32_t{0}](std::string_view input,
                                                    std::vector<std::string>& output, Deadline) mutable {
                    std::string altered(input);
                    if (static_cast<int32_t>(rank) == shared->alter_rank &&
                        calls == 0 && shared->alter_offset < altered.size()) {
                        altered[shared->alter_offset] ^= 1;
                    }
                    if (rank == 2 && calls == 1 && shared->alter_ack_rank_two && !altered.empty())
                        altered.back() ^= 1;
                    ++calls;
                    if (auto failure = shared->gather(rank, altered, output)) return failure;
                    if (rank == 2 && calls == 1 && shared->truncate_rank_two && !output.empty())
                        output.pop_back();
                    return ok_failure();
                });
            if (f) std::printf("fixture callback install failed: %s\n", f.detail.c_str());
        }
    }
};

int failures = 0;
void check(bool condition, const char* description) {
    std::printf("  %-70s %s\n", description, condition ? "ok" : "FAIL");
    if (!condition) ++failures;
}

void run_all_ranks(Fixture& fixture, const std::array<std::string, kTpWorld>& inputs,
                   std::array<Failure, kTpWorld>& results,
                   std::array<std::vector<std::string>, kTpWorld>& outputs) {
    std::array<std::thread, kTpWorld> threads;
    const Deadline deadline{std::chrono::steady_clock::time_point::max()};
    for (uint32_t rank = 0; rank < kTpWorld; ++rank) {
        threads[rank] = std::thread([&, rank] {
            results[rank] = fixture.sessions[rank]->exchange_control_payloads("tp3-proposal-v1", inputs[rank],
                                                                               outputs[rank], deadline);
        });
    }
    for (auto& thread : threads) thread.join();
}

template <typename Operation>
void run_control_operation(std::array<Failure, kTpWorld>& results, Operation operation) {
    std::array<std::thread, kTpWorld> threads;
    for (uint32_t rank = 0; rank < kTpWorld; ++rank)
        threads[rank] = std::thread([&, rank] { results[rank] = operation(rank); });
    for (auto& thread : threads) thread.join();
}

void success_and_terminal_mismatch() {
    {
        Fixture fixture;
        std::array<Failure, kTpWorld> results{};
        std::array<std::vector<std::string>, kTpWorld> outputs;
        std::array<std::string, kTpWorld> inputs{std::string("rank0\0\xff", 7), "rank1", ""};
        run_all_ranks(fixture, inputs, results, outputs);
        bool all_ok = true, all_equal = true;
        for (uint32_t rank = 0; rank < kTpWorld; ++rank) {
            all_ok &= !results[rank];
            all_equal &= outputs[rank] == std::vector<std::string>(inputs.begin(), inputs.end());
            all_equal &= fixture.gather.calls[rank] == 2;
        }
        check(all_ok && all_equal, "two callback gathers commit rank-ordered binary outputs on every rank");
    }
    {
        Fixture fixture;
        std::array<Failure, kTpWorld> results{};
        const Deadline deadline{std::chrono::steady_clock::time_point::max()};
        run_control_operation(results, [&](uint32_t rank) {
            return fixture.sessions[rank]->agree_control_payload("startup-v1", "same", deadline);
        });
        bool all_ok = true;
        for (uint32_t rank = 0; rank < kTpWorld; ++rank)
            all_ok &= !results[rank] && fixture.gather.calls[rank] == 2;
        check(all_ok, "control agreement uses the callback for exact payload ACKs");
    }
    {
        Fixture fixture;
        std::array<Failure, kTpWorld> results{};
        std::array<std::string, kTpWorld> outputs{"root-bytes", "", ""};
        const Deadline deadline{std::chrono::steady_clock::time_point::max()};
        run_control_operation(results, [&](uint32_t rank) {
            return fixture.sessions[rank]->broadcast_control_payload("startup-v1", outputs[rank], deadline);
        });
        bool all_ok = true, same_output = true;
        for (uint32_t rank = 0; rank < kTpWorld; ++rank) {
            all_ok &= !results[rank] && fixture.gather.calls[rank] == 2;
            same_output &= outputs[rank] == "root-bytes";
        }
        check(all_ok && same_output, "broadcast publishes the root bytes only after the callback ACK round");
    }
    {
        Fixture fixture;
        std::array<Failure, kTpWorld> results{};
        const std::string payload(40u * 1024u, 'p');
        const Deadline deadline{std::chrono::steady_clock::time_point::max()};
        run_control_operation(results, [&](uint32_t rank) {
            return fixture.sessions[rank]->agree_control_payload("large-prefill-plan-v1", payload, deadline);
        });
        bool all_ok = true;
        for (uint32_t rank = 0; rank < kTpWorld; ++rank)
            all_ok &= !results[rank] && fixture.gather.calls[rank] == 2;
        check(all_ok, "40 KiB identical payload fits the per-rank cap and full three-rank ACK transcript");
    }
    {
        Fixture fixture;
        std::array<Failure, kTpWorld> results{};
        std::array<std::vector<std::string>, kTpWorld> outputs;
        for (auto& output : outputs) output = {"unchanged"};
        fixture.gather.alter_rank = 1;
        fixture.gather.alter_offset = 60;  // first byte of the length-prefixed label
        std::array<std::string, kTpWorld> inputs{"same", "same", "same"};
        run_all_ranks(fixture, inputs, results, outputs);
        bool failed = true, unchanged = true;
        for (uint32_t rank = 0; rank < kTpWorld; ++rank) {
            failed &= results[rank].code == FailureCode::bad_operation &&
                      fixture.sessions[rank]->state() == SessionState::failed;
            unchanged &= outputs[rank] == std::vector<std::string>{"unchanged"};
        }
        check(failed && unchanged, "metadata disagreement becomes terminal and leaves outputs untouched");
    }
    {
        Fixture fixture;
        fixture.gather.truncate_rank_two = true;
        std::array<Failure, kTpWorld> results{};
        std::array<std::vector<std::string>, kTpWorld> outputs;
        for (auto& output : outputs) output = {"untouched"};
        std::array<std::string, kTpWorld> inputs{"one", "two", "three"};
        run_all_ranks(fixture, inputs, results, outputs);
        bool failed = true, unchanged = true;
        for (uint32_t rank = 0; rank < kTpWorld; ++rank) {
            failed &= results[rank].code == FailureCode::bad_operation &&
                      fixture.sessions[rank]->state() == SessionState::failed;
            unchanged &= outputs[rank] == std::vector<std::string>{"untouched"};
        }
        check(failed && unchanged, "incomplete callback gather fails terminally without publishing partial outputs");
    }
    {
        Fixture fixture;
        fixture.gather.alter_ack_rank_two = true;
        std::array<Failure, kTpWorld> results{};
        std::array<std::vector<std::string>, kTpWorld> outputs;
        for (auto& output : outputs) output = {"unchanged"};
        std::array<std::string, kTpWorld> inputs{"one", "two", "three"};
        run_all_ranks(fixture, inputs, results, outputs);
        bool failed = true, unchanged = true;
        for (uint32_t rank = 0; rank < kTpWorld; ++rank) {
            failed &= results[rank].code == FailureCode::bad_operation &&
                      fixture.sessions[rank]->state() == SessionState::failed;
            unchanged &= outputs[rank] == std::vector<std::string>{"unchanged"};
        }
        check(failed && unchanged, "ACK byte mismatch fails every rank before output publication");
    }
}

}  // namespace

int main() {
    success_and_terminal_mismatch();
    return failures == 0 ? 0 : 1;
}
