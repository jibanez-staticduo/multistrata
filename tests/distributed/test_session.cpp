// tests/distributed/test_session.cpp - three ranks becoming one session, and every way they refuse to.
//
// Two layers are tested, and each with the most real thing available to it:
//   * The protocol runs against a scripted control channel, so a peer that is a duplicate, a second late,
//     another run's, or simply gone is a fact the test sets rather than a race it hopes to lose.  The clock is
//     scripted too, so "the join deadline passed" costs nothing and cannot flake.
//   * The framing and the sockets run over real loopback TCP: three ranks in three threads reaching READY, and
//     raw peers that send a frame too long, a credential that is not theirs, a header from another protocol,
//     and a connection they drop.  A fake socket would only prove the session believes whatever it is told.
//
// What this does not say: anything about NCCL, RDMA, a GPU, or a model.  The session decides whether three
// ranks may start; it does not move a tensor.
//
// Build and run, CPU-only (no CUDA, no NCCL, no CMake):
//   g++ -std=c++20 -I include -Wall -Wextra tests/distributed/test_session.cpp src/distributed/session.cpp src/distributed/control_tcp.cpp -o test_session
//   ./test_session
// CMake lines for the integrator (default OFF, no CUDA dependency):
//   add_executable(test_session tests/distributed/test_session.cpp src/distributed/session.cpp src/distributed/control_tcp.cpp)
//   target_include_directories(test_session PRIVATE include)
//   add_test(NAME test_session COMMAND test_session)
#include "strata/distributed/session.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace strata::distributed;

int g_fail = 0;
int g_checks = 0;

void check(bool ok, const std::string& what) {
    ++g_checks;
    std::printf("  %-86s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

class FakeClock final : public Clock {
 public:
    std::chrono::steady_clock::time_point now() const override { return at_; }
    void advance_ms(int64_t ms) { at_ += std::chrono::milliseconds(ms); }

 private:
    std::chrono::steady_clock::time_point at_{};
};

struct Sent {
    uint32_t to = 0;
    FrameKind kind = FrameKind::join;
    std::string payload;
};

// The control channel as the test leaves it: frames it will hand over in order, and what it does when it runs
// out.  Nothing here pretends to be NCCL - it is the socket layer, and the socket layer only ever says "here is
// a frame", "nothing yet", or "this peer is finished".
class FakeChannel final : public ControlChannel {
 public:
    Failure open(Deadline) override {
        ++open_calls;
        return open_result;
    }

    Failure send(uint32_t to, FrameKind kind, std::string_view payload, Deadline) override {
        if (send_failure) return send_failure;
        sent.push_back(Sent{to, kind, std::string(payload)});
        return ok_failure();
    }

    Failure recv(ReceivedFrame& out, Deadline) override {
        ++recv_calls;
        clock_.advance_ms(1);
        if (recv_frames.empty()) return idle_failure;
        out = recv_frames.front();
        recv_frames.pop_front();
        return ok_failure();
    }

    void close() noexcept override { ++close_calls; }

    bool sent_to(uint32_t rank, FrameKind kind) const {
        for (const auto& s : sent) {
            if (s.to == rank && s.kind == kind) return true;
        }
        return false;
    }

    int sends_of(FrameKind kind) const {
        int n = 0;
        for (const auto& s : sent) {
            if (s.kind == kind) ++n;
        }
        return n;
    }

    FakeClock& clock_;
    std::deque<ReceivedFrame> recv_frames;
    Failure idle_failure = Failure::fail(FailureCode::deadline_expired,
                                        "the fake channel has nothing left to hand over");
    Failure open_result{};
    Failure send_failure{};
    std::vector<Sent> sent;
    int open_calls = 0;
    int recv_calls = 0;
    int close_calls = 0;

    explicit FakeChannel(FakeClock& clock) : clock_(clock) {}
};

const char* kUuid = "11111111-2222-3333-4444-555555555555";
const char* kOtherUuid = "99999999-8888-7777-6666-555555555555";
const char* kCredential = "session-credential-for-the-test-run";
const char* kModel = "sha256:1111manifest-and-tensor-content";
const char* kConfig = "sha256:2222family-quant-geometry-shardmap-dtype-proto";
const char* kNccl = "2.28.7";
const char* kProvenance = "sha256:3333 triangle-patch-3";

std::string bootstrap_of(uint32_t rank) {
    return "BOOTSTRAP-ID-RANK-" + std::to_string(rank) + "-of-the-verified-roster";
}

JoinInfo good_join(uint32_t rank) {
    JoinInfo info;
    info.session_uuid = kUuid;
    info.rank = rank;
    info.world = kTpWorld;
    info.protocol_version = kControlProtocolVersion;
    info.model_hash = kModel;
    info.config_fingerprint = kConfig;
    info.backend_identity = "nccl";
    info.nccl_version_expected = kNccl;
    info.library_provenance = kProvenance;
    info.nccl_bootstrap_id = bootstrap_of(rank);
    return info;
}

SessionConfig good_config(uint32_t rank) {
    SessionConfig config;
    config.rank = rank;
    config.world = kTpWorld;
    config.session_uuid = kUuid;
    config.credential = Credential(kCredential);
    config.approved_nccl_version = kNccl;
    config.approved_library_provenance = kProvenance;
    config.budgets = PhaseBudgets{1000, 2000, 3000, 4000};
    config.notify_ms = 50;
    return config;
}

std::string join_payload(const JoinInfo& info) {
    std::string out;
    const Failure f = encode_join(info, out);
    if (f) {
        std::printf("    (test setup: encode_join failed: %s)\n", f.detail.c_str());
        ++g_fail;
    }
    return out;
}

std::string roster_payload(const std::vector<JoinInfo>& ranks, PhaseBudgets budgets,
                          const std::string& digest_override = std::string()) {
    Roster roster;
    roster.ranks = ranks;
    roster.digest = digest_override.empty() ? roster_digest(ranks) : digest_override;
    roster.budgets = budgets;
    std::string out;
    const Failure f = encode_roster(roster, out);
    if (f) {
        std::printf("    (test setup: encode_roster failed: %s)\n", f.detail.c_str());
        ++g_fail;
    }
    return out;
}

std::string make_ack(const std::string& uuid, uint32_t rank, const std::string& digest) {
    std::string out;
    auto put32 = [&out](uint32_t v) {
        out.push_back(static_cast<char>((v >> 24) & 0xff));
        out.push_back(static_cast<char>((v >> 16) & 0xff));
        out.push_back(static_cast<char>((v >> 8) & 0xff));
        out.push_back(static_cast<char>(v & 0xff));
    };
    auto put_field = [&out, &put32](const std::string& v) {
        put32(static_cast<uint32_t>(v.size()));
        out += v;
    };
    put_field(uuid);
    put32(rank);
    put_field(digest);
    return out;
}

std::string abort_payload(FailureCode code, const std::string& reason) {
    std::string out;
    const Failure f = encode_abort(code, reason, out);
    if (f) std::printf("    (test setup: encode_abort failed: %s)\n", f.detail.c_str());
    return out;
}

ReceivedFrame from(uint32_t rank, FrameKind kind, const std::string& payload) {
    ReceivedFrame got;
    got.from_rank = rank;
    got.frame.kind = kind;
    got.frame.payload = payload;
    return got;
}

// One case's worth of collaborators, with the log captured rather than printed.
struct Harness {
    FakeClock clock;
    FakeChannel channel{clock};
    std::vector<std::string> lines;

    // The session under test, with its log captured instead of printed.  The channel's clock is this clock,
    // bound once: a test that moves time moves it for both halves of the case.
    Session session(uint32_t self_rank, JoinInfo self, SessionConfig config) {
        (void)self_rank;
        return Session(std::move(config), std::move(self), channel, clock,
                       [this](std::string_view line) { lines.emplace_back(line); });
    }
};

std::vector<JoinInfo> good_roster() {
    return {good_join(0), good_join(1), good_join(2)};
}

// ---------------------------------------------------------------- the protocol, scripted

std::string operation_payload(const CollectiveDescriptor& d) {
    std::string signature, out;
    canonical_collective(d, signature);
    auto field = [&](const std::string& text) {
        const uint32_t n = static_cast<uint32_t>(text.size());
        for (int shift : {24, 16, 8, 0}) out.push_back(static_cast<char>((n >> shift) & 0xff));
        out += text;
    };
    field(kUuid); field(signature);
    return out;
}

std::string control_payload(uint64_t sequence, std::string_view label, std::string_view payload) {
    std::string out;
    auto put32 = [&](uint32_t v) {
        for (int shift : {24, 16, 8, 0}) out.push_back(static_cast<char>((v >> shift) & 0xff));
    };
    auto field = [&](std::string_view value) {
        put32(static_cast<uint32_t>(value.size()));
        out.append(value);
    };
    field(kUuid);
    put32(static_cast<uint32_t>(sequence >> 32));
    put32(static_cast<uint32_t>(sequence));
    field(label);
    field(payload);
    return out;
}

std::string broadcast_payload(uint64_t sequence, std::string_view label, std::string_view payload) {
    return std::string("BRC1", 4) + control_payload(sequence, label, payload);
}

void queue_coordinator_ready(Harness& h) {
    const auto ranks = good_roster();
    h.channel.recv_frames.push_back(from(1, FrameKind::join, join_payload(good_join(1))));
    h.channel.recv_frames.push_back(from(2, FrameKind::join, join_payload(good_join(2))));
    for (uint32_t rank : {1u, 2u}) {
        h.channel.recv_frames.push_back(from(rank, FrameKind::roster_ack,
                                            make_ack(kUuid, rank, roster_digest(ranks))));
    }
}

void control_payload_agreement() {
    const std::string label("route\0order", 11), bytes("secret\0binary\xff", 14);
    {
        Harness h;
        Session session = h.session(0, good_join(0), good_config(0));
        check(session.agree_control_payload(label, bytes, {}).code == FailureCode::not_ready &&
                  h.channel.sent.empty(), "control agreement requires READY before any wire byte");
    }
    // Mutate each envelope byte: length, session, sequence, label and payload are all compared exactly.
    const std::string canonical = control_payload(1, label, bytes);
    bool every_mismatch_failed = true, no_release = true, no_raw_log = true;
    for (size_t byte = 0; byte < canonical.size(); ++byte) {
        Harness h;
        queue_coordinator_ready(h);
        Session session = h.session(0, good_join(0), good_config(0));
        if (session.start()) { every_mismatch_failed = false; break; }
        std::string changed = canonical;
        changed[byte] ^= 1;
        h.channel.recv_frames.push_back(from(1, FrameKind::control, canonical));
        h.channel.recv_frames.push_back(from(2, FrameKind::control, changed));
        const auto f = session.agree_control_payload(label, bytes, {h.clock.now() + std::chrono::seconds(1)});
        every_mismatch_failed &= f.code == FailureCode::bad_operation && !session.ready() &&
            h.channel.sent_to(1, FrameKind::abort) && h.channel.sent_to(2, FrameKind::abort) &&
            session.agree_control_payload(label, bytes, {}).code == FailureCode::not_ready;
        no_release &= h.channel.sends_of(FrameKind::control_ack) == 0;
        for (const auto& line : h.lines) no_raw_log &= line.find("secret") == std::string::npos &&
                                                     line.find("route") == std::string::npos;
    }
    check(every_mismatch_failed, "every changed byte in control metadata aborts both peers terminally");
    check(no_release, "no metadata mismatch grants a control ACK or compute release");
    check(no_raw_log, "control mismatch logs contain no raw label or payload bytes");

    for (int mode = 0; mode < 5; ++mode) {
        Harness h;
        const auto ranks = good_roster();
        h.channel.recv_frames.push_back(from(0, FrameKind::roster, roster_payload(ranks, PhaseBudgets{})));
        h.channel.recv_frames.push_back(from(0, FrameKind::ready, make_ack(kUuid, 0, roster_digest(ranks))));
        Session session = h.session(1, good_join(1), good_config(1));
        check(!session.start(), "follower is ready for exact control agreement");
        std::string metadata = canonical, ack = canonical;
        if (mode == 1) metadata.back() ^= 1;
        if (mode == 2) ack.back() ^= 1;
        h.channel.recv_frames.push_back(from(mode == 4 ? 2 : 0, FrameKind::control, metadata));
        if (mode != 3) h.channel.recv_frames.push_back(from(0, FrameKind::control_ack, ack));
        const Failure f = session.agree_control_payload(label, bytes,
                                                        {h.clock.now() + std::chrono::seconds(1)});
        check(mode == 0 ? !f && session.ready() :
              f.code == (mode == 3 ? FailureCode::deadline_expired : FailureCode::bad_operation) &&
                  !session.ready(), "follower requires exact coordinator metadata and final ACK from rank zero");
        if (mode == 1 || mode == 4) check(h.channel.sends_of(FrameKind::control_ack) == 0,
                                        "follower does not ACK invalid metadata or a different sender");
    }

    for (int failure = 0; failure < 7; ++failure) {
        Harness h;
        queue_coordinator_ready(h);
        Session session = h.session(0, good_join(0), good_config(0));
        check(!session.start(), "coordinator is ready for control agreement failure case");
        std::string peer = canonical;
        if (failure == 0) {
            // Same concatenation but different boundaries must never match.
            peer = control_payload(1, label.substr(0, label.size() - 1),
                                   label.substr(label.size() - 1) + bytes);
        }
        h.channel.recv_frames.push_back(from(1, FrameKind::control, peer));
        h.channel.recv_frames.push_back(from(failure == 1 ? 1 : 2, FrameKind::control, canonical));
        if (failure != 2) {
            h.channel.recv_frames.push_back(from(1, FrameKind::control_ack, canonical));
            std::string ack = canonical;
            if (failure == 3) ack.back() ^= 1;
            h.channel.recv_frames.push_back(from(2, FrameKind::control_ack, ack));
        }
        Deadline at{h.clock.now() + std::chrono::seconds(1)};
        if (failure == 4) at.end = h.clock.now();
        std::string large;
        std::string_view local = bytes;
        if (failure == 5) { large.assign(kControlMaxPayload, 'x'); local = large; }
        if (failure == 6) {
            CollectiveDescriptor d{{1, 2, 3}, CollectiveKind::all_reduce_sum, ScalarType::fp32, 1, {}};
            h.channel.recv_frames.clear();
            h.channel.recv_frames.push_back(from(1, FrameKind::operation, operation_payload(d)));
            h.channel.recv_frames.push_back(from(2, FrameKind::operation, operation_payload(d)));
            check(!session.agree_operation(d, at), "an agreed collective holds the in-flight slot");
        }
        const Failure f = session.agree_control_payload(label, local, at);
        const FailureCode expected = failure == 2 || failure == 4 ? FailureCode::deadline_expired :
            failure == 5 ? FailureCode::oversized_frame : failure == 6 ? FailureCode::in_flight :
            FailureCode::bad_operation;
        check(f.code == expected && !session.ready(), "control agreement refuses ambiguity, duplicate, missing ACK, expiry, cap or in-flight");
        check(h.channel.sends_of(FrameKind::control_ack) == 0,
              "failed control agreement does not release any follower");
    }
    {
        Harness h;
        queue_coordinator_ready(h);
        Session session = h.session(0, good_join(0), good_config(0));
        check(!session.start(), "coordinator is ready for repeated exact control bytes");
        bool exact = true;
        for (uint64_t sequence = 1; sequence <= 3; ++sequence) {
            const auto envelope = control_payload(sequence, label, bytes);
            for (const auto kind : {FrameKind::control, FrameKind::control_ack}) {
                for (uint32_t rank : {1u, 2u}) h.channel.recv_frames.push_back(from(rank, kind, envelope));
            }
            const size_t sent_before = h.channel.sent.size();
            exact &= !session.agree_control_payload(label, bytes, {h.clock.now() + std::chrono::seconds(1)});
            for (size_t i = sent_before; i < h.channel.sent.size(); ++i) {
                const auto& sent = h.channel.sent[i];
                if (sent.kind == FrameKind::control || sent.kind == FrameKind::control_ack)
                    exact &= sent.payload == envelope;
            }
        }
        check(exact && h.channel.sends_of(FrameKind::control_ack) == 6,
              "repeated binary control agreements retain exact embedded NUL bytes and independent sequence");
        CollectiveDescriptor d{{1, 2, 3}, CollectiveKind::all_reduce_sum, ScalarType::fp32, 1, {}};
        for (uint32_t rank : {1u, 2u}) h.channel.recv_frames.push_back(from(rank, FrameKind::operation, operation_payload(d)));
        check(!session.agree_operation(d, {h.clock.now() + std::chrono::seconds(1)}) &&
                  !session.complete_operation(d.operation), "control agreements do not consume collective sequence numbers");
    }
}

void control_payload_broadcast() {
    const std::string label("action\0step", 11), bytes("private\0payload\xff", 16);
    const auto request = broadcast_payload(1, label, {});
    const auto envelope = broadcast_payload(1, label, bytes);
    {
        Harness h;
        Session session = h.session(0, good_join(0), good_config(0));
        std::string output = bytes;
        check(session.broadcast_control_payload(label, output, {}).code == FailureCode::not_ready &&
                  output == bytes && h.channel.sent.empty(), "broadcast requires READY and preserves caller bytes on failure");
    }
    // The follower never publishes staged bytes on a malformed proposal, missing/mismatching commit,
    // a failure from the channel, or a deadline reached while reading the final commit.
    for (int mode = 0; mode < 15; ++mode) {
        Harness h;
        const auto ranks = good_roster();
        h.channel.recv_frames.push_back(from(0, FrameKind::roster, roster_payload(ranks, PhaseBudgets{})));
        h.channel.recv_frames.push_back(from(0, FrameKind::ready, make_ack(kUuid, 0, roster_digest(ranks))));
        Session session = h.session(1, good_join(1), good_config(1));
        check(!session.start(), "follower is ready for binary broadcast");
        std::string proposed = envelope, commit = envelope, output;
        if (mode == 1) proposed = broadcast_payload(1, "other-label", bytes);
        if (mode == 2) proposed = broadcast_payload(0, label, bytes);
        if (mode == 3) proposed[8] ^= 1;  // another session UUID
        if (mode == 4) proposed.pop_back();
        if (mode == 5) proposed += 'x';
        if (mode == 6) proposed.assign(kControlMaxPayload + 1, 'x');
        if (mode == 7) proposed.assign(3, 'x');
        if (mode == 8) commit.back() ^= 1;
        if (mode == 11) output = "follower-input";
        if (mode == 12) proposed = control_payload(1, label, bytes);  // mixed APIs
        if (mode == 13) {
            const size_t offset = request.size() - 4;
            proposed.replace(offset, 4, 4, static_cast<char>(0xff));
        }
        h.channel.recv_frames.push_back(from(mode == 10 ? 2 : 0, FrameKind::control, proposed));
        if (mode != 9) h.channel.recv_frames.push_back(from(0, FrameKind::control_ack, commit));
        const auto before = output;
        const auto at = Deadline{h.clock.now() + std::chrono::milliseconds(mode == 14 ? 2 : 1000)};
        const auto f = session.broadcast_control_payload(label, output, at);
        FailureCode expected = FailureCode::bad_operation;
        if (mode == 0) expected = FailureCode::none;
        if (mode == 4 || mode == 7) expected = FailureCode::truncated_frame;
        if (mode == 5) expected = FailureCode::bad_frame;
        if (mode == 6 || mode == 13) expected = FailureCode::oversized_frame;
        if (mode == 9 || mode == 14) expected = FailureCode::deadline_expired;
        check(f.code == expected && (mode == 0 ? session.ready() && output == bytes :
                  !session.ready() && output == before), "broadcast follower publishes exact binary bytes only after a valid final commit");
        if (mode != 0) check(session.broadcast_control_payload(label, output, {}).code == FailureCode::not_ready,
                             "failed broadcast is terminal");
        for (const auto& line : h.lines) {
            check(line.find("private") == std::string::npos && line.find("action") == std::string::npos,
                  "broadcast logs contain no caller label or payload bytes");
        }
    }
    for (int mode = 0; mode < 10; ++mode) {
        Harness h;
        queue_coordinator_ready(h);
        Session session = h.session(0, good_join(0), good_config(0));
        check(!session.start(), "coordinator is ready for broadcast validation");
        std::string peer_request = request, ack = envelope, output = bytes;
        if (mode == 1) peer_request = broadcast_payload(1, label, "unexpected-input");
        if (mode == 2) peer_request = broadcast_payload(0, label, {});
        if (mode == 3) ack.back() ^= 1;
        if (mode == 5) output.assign(kControlMaxPayload, 'x');
        if (mode == 9) h.channel.send_failure = Failure::fail(FailureCode::peer_closed, "send failed");
        h.channel.recv_frames.push_back(from(1, FrameKind::control, peer_request));
        h.channel.recv_frames.push_back(from(mode == 4 ? 1 : 2, FrameKind::control, request));
        if (mode != 6) {
            h.channel.recv_frames.push_back(from(1, FrameKind::control_ack, envelope));
            h.channel.recv_frames.push_back(from(mode == 7 ? 1 : 2, FrameKind::control_ack, ack));
        }
        Deadline at{h.clock.now() + std::chrono::seconds(1)};
        if (mode == 8) {
            CollectiveDescriptor d{{1, 2, 3}, CollectiveKind::all_reduce_sum, ScalarType::fp32, 1, {}};
            h.channel.recv_frames.clear();
            for (uint32_t rank : {1u, 2u}) h.channel.recv_frames.push_back(from(rank, FrameKind::operation, operation_payload(d)));
            check(!session.agree_operation(d, at), "broadcast test holds a collective in flight");
        }
        const auto before = output;
        const auto f = session.broadcast_control_payload(label, output, at);
        const auto expected = mode == 0 ? FailureCode::none : mode == 5 ? FailureCode::oversized_frame :
            mode == 6 ? FailureCode::deadline_expired : mode == 8 ? FailureCode::in_flight :
            mode == 9 ? FailureCode::peer_closed : FailureCode::bad_operation;
        check(f.code == expected && output == before && session.ready() == (mode == 0),
              "coordinator broadcast requires distinct empty requests and exact acknowledgments before commit");
        if (mode != 0) check(h.channel.sends_of(FrameKind::control_ack) == 0,
                             "failed coordinator broadcast sends no commit");
    }
    {
        Harness h;
        queue_coordinator_ready(h);
        Session session = h.session(0, good_join(0), good_config(0));
        check(!session.start(), "coordinator is ready for mixed control sequence");
        bool exact = true;
        for (uint64_t sequence = 1; sequence <= 4; ++sequence) {
            const bool broadcast = sequence % 2 == 0;
            const auto value = sequence == 4 ? std::string{} : bytes;
            const auto message = broadcast ? broadcast_payload(sequence, label, value) : control_payload(sequence, label, value);
            const auto initial = broadcast ? broadcast_payload(sequence, label, {}) : message;
            for (uint32_t rank : {1u, 2u}) h.channel.recv_frames.push_back(from(rank, FrameKind::control, initial));
            for (uint32_t rank : {1u, 2u}) h.channel.recv_frames.push_back(from(rank, FrameKind::control_ack, message));
            std::string output = value;
            const auto at = Deadline{h.clock.now() + std::chrono::seconds(1)};
            exact &= !(broadcast ? session.broadcast_control_payload(label, output, at) : session.agree_control_payload(label, value, at));
            exact &= output == value;
        }
        check(exact, "agreement and broadcast share a sequence, including an empty root broadcast");
    }
}

void collective_agreement() {
    const CollectiveDescriptor good{{1, 2, 3}, CollectiveKind::all_gather_padded, ScalarType::fp32,
                                    256, {256, 192, 192}};
    for (int mismatch = -1; mismatch < 6; ++mismatch) {
        Harness h;
        const auto ranks = good_roster();
        h.channel.recv_frames.push_back(from(1, FrameKind::join, join_payload(good_join(1))));
        h.channel.recv_frames.push_back(from(2, FrameKind::join, join_payload(good_join(2))));
        h.channel.recv_frames.push_back(from(1, FrameKind::roster_ack, make_ack(kUuid, 1, roster_digest(ranks))));
        h.channel.recv_frames.push_back(from(2, FrameKind::roster_ack, make_ack(kUuid, 2, roster_digest(ranks))));
        Session session = h.session(0, good_join(0), good_config(0));
        check(!session.start(), "coordinator is ready for collective agreement");
        auto peer = good;
        if (mismatch == 0) ++peer.operation.sequence;
        if (mismatch == 1) { peer.elements = 257; peer.valid_counts[0] = 257; }
        if (mismatch == 2) peer.valid_counts[1] = 191;
        if (mismatch == 3) peer.type = ScalarType::bf16;
        if (mismatch == 4) { peer.kind = CollectiveKind::all_reduce_sum; peer.valid_counts.clear(); }
        if (mismatch == 5) ++peer.operation.layer;
        h.channel.recv_frames.push_back(from(1, FrameKind::operation, operation_payload(good)));
        h.channel.recv_frames.push_back(from(2, FrameKind::operation, operation_payload(peer)));
        const Deadline at{h.clock.now() + std::chrono::seconds(1)};
        const Failure f = session.agree_operation(good, at);
        if (mismatch >= 0) {
            check(f.code == FailureCode::bad_operation && !session.ready(), "cross-rank metadata mismatch ends the session");
            check(h.channel.sent_to(1, FrameKind::abort) && h.channel.sent_to(2, FrameKind::abort),
                  "both peers receive an abort instead of an enqueue grant");
        } else {
            check(!f && h.channel.sent_to(1, FrameKind::operation_ack) && h.channel.sent_to(2, FrameKind::operation_ack),
                  "identical metadata grants the collective on both peers");
            check(!session.complete_operation(good.operation), "a waited collective releases the agreement slot");
            auto second = good;
            second.operation.sequence = 2;
            h.channel.recv_frames.push_back(from(1, FrameKind::operation, operation_payload(second)));
            h.channel.recv_frames.push_back(from(2, FrameKind::operation, operation_payload(second)));
            check(!session.agree_operation(second, at), "the next sequence can agree after completion");
            check(session.agree_operation(second, at).code == FailureCode::in_flight,
                  "only one agreed operation may be in flight");
        }
    }
    for (int mode = 0; mode < 3; ++mode) {
        Harness h;
        const auto ranks = good_roster();
        h.channel.recv_frames.push_back(from(0, FrameKind::roster, roster_payload(ranks, PhaseBudgets{})));
        h.channel.recv_frames.push_back(from(0, FrameKind::ready, make_ack(kUuid, 0, roster_digest(ranks))));
        Session session = h.session(1, good_join(1), good_config(1));
        check(!session.start(), "follower is ready for collective agreement");
        auto ack = good;
        if (mode == 1) ++ack.operation.step;
        if (mode < 2) h.channel.recv_frames.push_back(from(0, FrameKind::operation_ack, operation_payload(ack)));
        const Failure f = session.agree_operation(good, Deadline{h.clock.now() + std::chrono::seconds(1)});
        if (mode == 0) check(!f && !session.complete_operation(good.operation), "follower checks the exact granted descriptor");
        if (mode == 1) check(f.code == FailureCode::bad_operation && !session.ready(), "follower refuses an ack for different metadata");
        if (mode == 2) check(f.code == FailureCode::deadline_expired && !session.ready(), "a missing operation ack ends at the deadline");
    }
}

void coordinator_reaches_ready() {
    std::printf("rank 0 forms the session (scripted channel)\n");
    Harness h;
    const Roster expected{good_roster(), roster_digest(good_roster()),
                          PhaseBudgets{1000, 2000, 3000, 4000}};
    h.channel.recv_frames.push_back(from(1, FrameKind::join, join_payload(good_join(1))));
    h.channel.recv_frames.push_back(from(2, FrameKind::join, join_payload(good_join(2))));
    h.channel.recv_frames.push_back(from(1, FrameKind::roster_ack, make_ack(kUuid, 1, expected.digest)));
    h.channel.recv_frames.push_back(from(2, FrameKind::roster_ack, make_ack(kUuid, 2, expected.digest)));

    Session session = h.session(0, good_join(0), good_config(0));
    const Failure f = session.start();
    check(!f && session.ready(), "three verified joins and three acks form the session");
    check(session.roster().ranks.size() == kTpWorld, "the roster names all three ranks");
    check(session.roster().digest == expected.digest, "and the digest everyone acknowledged");
    check(h.channel.sent_to(1, FrameKind::roster) && h.channel.sent_to(2, FrameKind::roster),
          "the roster went to both peers");
    check(h.channel.sent_to(1, FrameKind::ready) && h.channel.sent_to(2, FrameKind::ready),
          "READY went out only after both acks");

    std::string id;
    check(!session.bootstrap_id(2, id) && id == bootstrap_of(2),
          "a peer's bootstrap id comes from the verified roster");
    check(session.bootstrap_id(7, id).code == FailureCode::rank_range, "a rank outside it is refused");
    Deadline ready_deadline{};
    check(!session.operation_deadline(ready_deadline) && ready_deadline.end > h.clock.now(),
          "and an operation deadline is available once ready");

    const std::vector<std::string>* lines = &h.lines;
    bool clean = true;
    for (const auto& line : *lines) {
        if (line.find(kCredential) != std::string::npos) clean = false;
        for (uint32_t r = 0; r < kTpWorld; ++r) {
            if (line.find(bootstrap_of(r)) != std::string::npos) clean = false;
        }
    }
    check(clean && !lines->empty(),
          "no log line carries the credential or a bootstrap id, and the credential is shown redacted");
    bool marked_redacted = false;
    for (const auto& line : h.lines) {
        if (line.find("credential(len=") != std::string::npos) marked_redacted = true;
    }
    check(marked_redacted, "the log says which credential was used by length, not by value");
}

void follower_reaches_ready() {
    std::printf("rank 2 accepts a roster and waits for READY (scripted channel)\n");
    Harness h;
    const std::vector<JoinInfo> ranks = good_roster();
    h.channel.recv_frames.push_back(
        from(0, FrameKind::roster, roster_payload(ranks, PhaseBudgets{1000, 2000, 3000, 4000})));
    h.channel.recv_frames.push_back(
        from(0, FrameKind::ready, make_ack(kUuid, 0, roster_digest(ranks))));

    Session session = h.session(2, good_join(2), good_config(2));
    check(!session.start() && session.ready(), "the follower forms once rank 0 says READY");
    check(h.channel.sent_to(0, FrameKind::join), "it said who it was");
    check(h.channel.sent_to(0, FrameKind::roster_ack), "and acknowledged the roster it verified");
    check(session.roster().digest == roster_digest(ranks), "with the digest it recomputed itself");
}

void follower_adopts_the_smaller_budget() {
    std::printf("a rank keeps the smaller deadline, and only durations cross the wire\n");
    Harness h;
    const std::vector<JoinInfo> ranks = good_roster();
    // rank 0 advertises 40 ms where this rank would happily wait 4000 ms
    h.channel.recv_frames.push_back(
        from(0, FrameKind::roster, roster_payload(ranks, PhaseBudgets{1000, 2000, 3000, 40})));
    h.channel.recv_frames.push_back(from(0, FrameKind::ready, make_ack(kUuid, 0, roster_digest(ranks))));

    Session session = h.session(2, good_join(2), good_config(2));
    check(!session.start() && session.ready(), "the session forms on rank 0's advertised lengths");
    Deadline at{};
    check(!session.operation_deadline(at), "an operation deadline is computed");
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(at.end - h.clock.now()).count();
    check(ms == 40, "and it is the shorter of the two, counted on this host's monotonic clock");

    std::string budgets;
    check(!encode_budgets(PhaseBudgets{1, 2, 3, 4}, budgets) && budgets.size() == 16,
          "a phase budget travels as four 32-bit durations, so no machine's clock value can be read as a deadline");
    PhaseBudgets back;
    check(!decode_budgets(budgets, back) && back.operation_ms == 4, "and it comes back unchanged");
    check(encode_budgets(PhaseBudgets{0, 1, 1, 1}, budgets).code == FailureCode::invalid_config,
          "a phase with no time in it is not a phase");
    check(encode_budgets(PhaseBudgets{1, 1, 1, kMaxPhaseMs + 1}, budgets).code == FailureCode::invalid_config,
          "and a phase longer than an hour is refused rather than treated as forever");
}

void coordinator_refuses_a_group_that_does_not_agree() {
    std::printf("rank 0 refuses a group that does not agree\n");

    struct Case {
        JoinInfo bad;
        FailureCode code;
        const char* what;
    };
    std::vector<Case> cases;
    {
        JoinInfo wrong_uuid = good_join(1);
        wrong_uuid.session_uuid = kOtherUuid;
        cases.push_back({wrong_uuid, FailureCode::uuid_mismatch, "a rank from another run"});
        JoinInfo wrong_model = good_join(1);
        wrong_model.model_hash = "sha256:deadbeefdifferent-weights";
        cases.push_back({wrong_model, FailureCode::model_mismatch, "a rank that loaded a different model"});
        JoinInfo wrong_config = good_join(1);
        wrong_config.config_fingerprint = "sha256:9999different-shard-map";
        cases.push_back({wrong_config, FailureCode::config_mismatch, "a rank with another shard map"});
        JoinInfo wrong_backend = good_join(1);
        wrong_backend.backend_identity = "rocenante";
        cases.push_back({wrong_backend, FailureCode::backend_mismatch, "a rank speaking another backend"});
        JoinInfo wrong_nccl = good_join(1);
        wrong_nccl.nccl_version_expected = "2.27.7";
        cases.push_back({wrong_nccl, FailureCode::nccl_version_mismatch,
                        "a rank with another NCCL version"});
        JoinInfo wrong_prov = good_join(1);
        wrong_prov.library_provenance = "sha256:0000 unpatched-upstream";
        cases.push_back({wrong_prov, FailureCode::provenance_mismatch,
                        "a rank with the wrong library build"});
    }

    for (const auto& c : cases) {
        Harness h;
        h.channel.recv_frames.push_back(from(1, FrameKind::join, join_payload(c.bad)));
        Session session = h.session(0, good_join(0), good_config(0));
        const Failure f = session.start();
        check(f.code == c.code && session.state() == SessionState::failed, c.what);
        check(h.channel.sends_of(FrameKind::abort) == 2,
              std::string("and the ABORT went to both peers: ") + c.what);
        check(h.channel.close_calls >= 1, std::string("the channel was released: ") + c.what);
    }
}

void coordinator_refuses_a_wrong_group_shape() {
    std::printf("rank 0 refuses a group with the wrong shape\n");
    {
        Harness h;
        h.channel.recv_frames.push_back(from(1, FrameKind::join, join_payload(good_join(1))));
        h.channel.recv_frames.push_back(from(1, FrameKind::join, join_payload(good_join(1))));
        Session session = h.session(0, good_join(0), good_config(0));
        const Failure f = session.start();
        check(f.code == FailureCode::duplicate_rank, "a rank that joins twice is the second one to be refused");
        check(h.channel.sends_of(FrameKind::abort) == 2, "and the ABORT went to both peers");
    }
    {
        Harness h;
        JoinInfo too_high = good_join(1);
        too_high.rank = 5;  // the link is rank 1's, the join claims rank 5
        h.channel.recv_frames.push_back(from(1, FrameKind::join, join_payload(too_high)));
        Session session = h.session(0, good_join(0), good_config(0));
        const Failure f = session.start();
        check(f.code == FailureCode::rank_range, "a join whose rank is outside the world is refused");
    }
    {
        Harness h;
        h.channel.recv_frames.push_back(from(7, FrameKind::join, join_payload(good_join(1))));
        Session session = h.session(0, good_join(0), good_config(0));
        const Failure f = session.start();
        check(f.code == FailureCode::bad_frame,
              "a join naming one rank but arriving on another rank's link is refused");
    }
    {
        Harness h;  // rank 1 joins; rank 2 never does
        h.channel.recv_frames.push_back(from(1, FrameKind::join, join_payload(good_join(1))));
        Session session = h.session(0, good_join(0), good_config(0));
        const Failure f = session.start();
        check(f.code == FailureCode::missing_rank && f.detail.find("2") != std::string::npos,
              "a join deadline that passes names the rank that never came");
    }
    {
        Harness h;  // rank 1 sends a join payload cut short inside a well-formed frame
        std::string chopped = join_payload(good_join(1));
        chopped.resize(chopped.size() - 6);
        h.channel.recv_frames.push_back(from(1, FrameKind::join, chopped));
        Session session = h.session(0, good_join(0), good_config(0));
        const Failure f = session.start();
        check(f.code == FailureCode::truncated_frame, "a truncated join is a truncated frame, not a default");
    }
    {
        // A join claiming a world of four cannot be encoded by this build (encode_join refuses it), so the
        // world field of an otherwise good join is rewritten on the wire: 36 bytes of session id, then the
        // rank, then the world as a big-endian u32.
        Harness h;
        std::string payload = join_payload(good_join(1));
        payload[44] = 0;
        payload[45] = 0;
        payload[46] = 0;
        payload[47] = 4;
        h.channel.recv_frames.push_back(from(1, FrameKind::join, payload));
        Session session = h.session(0, good_join(0), good_config(0));
        const Failure f = session.start();
        check(f.code == FailureCode::rank_range,
              "a world other than three is not negotiated: it is refused");
    }
}

void stale_and_dead_peers() {
    std::printf("stale joins, and peers that stop talking\n");
    {
        Harness h;
        h.channel.recv_frames.push_back(from(1, FrameKind::join, join_payload(good_join(1))));
        h.channel.recv_frames.push_back(from(2, FrameKind::join, join_payload(good_join(2))));
        h.channel.recv_frames.push_back(from(1, FrameKind::join, join_payload(good_join(1))));  // another attempt
        Session session = h.session(0, good_join(0), good_config(0));
        const Failure f = session.start();
        check(f.code == FailureCode::stale_join, "a join that arrives after the roster is from another attempt");
    }
    {
        Harness h;
        h.channel.idle_failure = Failure::fail(FailureCode::peer_closed, "rank 1 closed its control connection");
        Session session = h.session(0, good_join(0), good_config(0));
        const Failure f = session.start();
        check(f.code == FailureCode::peer_closed, "a peer that disconnects during the join ends the session");
        check(h.channel.close_calls >= 1, "and the channel is released");
    }
    {
        Harness h;
        h.channel.idle_failure = Failure::fail(FailureCode::oversized_frame, "1073741824 bytes is not a join");
        Session session = h.session(0, good_join(0), good_config(0));
        const Failure f = session.start();
        check(f.code == FailureCode::oversized_frame,
              "a frame whose length is absurd is refused before it is read");
    }
    {
        Harness h;
        h.channel.recv_frames.push_back(from(1, FrameKind::join, join_payload(good_join(1))));
        h.channel.idle_failure = Failure::fail(FailureCode::peer_closed, "rank 2 closed its control connection");
        Session session = h.session(0, good_join(0), good_config(0));
        const Failure f = session.start();
        check(f.code == FailureCode::peer_closed, "and so it ends when the second peer dies mid-join");
    }
    {
        // A follower that is told, in words, that the session is over.
        Harness h;
        h.channel.recv_frames.push_back(from(0, FrameKind::abort,
                                            abort_payload(FailureCode::comm_failed, "my communicator died")));
        Session session = h.session(2, good_join(2), good_config(2));
        const Failure f = session.start();
        check(f.code == FailureCode::comm_failed && f.detail.find("rank 0") != std::string::npos,
              "a peer's ABORT becomes this rank's reason, with the rank that sent it named");
        check(f.detail.find("my communicator died") != std::string::npos, "and its words are kept");
    }
    {
        // A follower whose coordinator never answers.
        Harness h;
        Session session = h.session(1, good_join(1), good_config(1));
        const Failure f = session.start();
        check(f.code == FailureCode::deadline_expired, "a coordinator that never answers is a deadline, not a wait");
    }
    {
        Harness h;
        const std::vector<JoinInfo> ranks = good_roster();
        h.channel.recv_frames.push_back(
            from(0, FrameKind::roster, roster_payload(ranks, PhaseBudgets{1000, 2000, 3000, 4000})));
        h.channel.recv_frames.push_back(from(0, FrameKind::join, join_payload(good_join(1))));
        Session session = h.session(2, good_join(2), good_config(2));
        const Failure f = session.start();
        check(f.code == FailureCode::stale_join,
              "a join that reaches a rank that does not route joins is a stale join");
    }
}

void a_roster_must_be_the_one_this_rank_said() {
    std::printf("a follower re-checks the roster instead of trusting rank 0\n");
    const std::vector<JoinInfo> ranks = good_roster();
    const PhaseBudgets budgets{1000, 2000, 3000, 4000};
    {
        Harness h;
        auto rewritten = ranks;
        rewritten[2].nccl_bootstrap_id = "a-different-communicator";
        check(roster_digest(rewritten) != roster_digest(ranks), "bootstrap bytes participate in roster agreement");
        h.channel.recv_frames.push_back(from(0, FrameKind::roster, roster_payload(rewritten, budgets)));
        Session session = h.session(2, good_join(2), good_config(2));
        check(session.start().code == FailureCode::roster_mismatch, "a rewritten local bootstrap is refused");
    }
    {
        Harness h;
        h.channel.recv_frames.push_back(from(1, FrameKind::join, join_payload(good_join(1))));
        h.channel.recv_frames.push_back(from(2, FrameKind::join, join_payload(good_join(2))));
        h.channel.recv_frames.push_back(from(1, FrameKind::roster_ack, make_ack(kUuid, 2, roster_digest(ranks))));
        Session session = h.session(0, good_join(0), good_config(0));
        check(session.start().code == FailureCode::bad_frame, "a roster ack cannot acknowledge another rank's slot");
    }
    {
        Harness h;
        std::vector<JoinInfo> rewritten = ranks;
        rewritten[2].config_fingerprint = "sha256:not-what-i-sent";
        h.channel.recv_frames.push_back(from(0, FrameKind::roster, roster_payload(rewritten, budgets)));
        Session session = h.session(2, good_join(2), good_config(2));
        const Failure f = session.start();
        // The disagreement is named by the field that differs, which is the useful form of the message: the
        // reader learns which of the three ranks stopped matching, not just that a roster looked wrong.
        check(f.code == FailureCode::config_mismatch,
              "a roster that rewrote this rank's own join is refused, naming the field");
    }
    {
        Harness h;
        h.channel.recv_frames.push_back(
            from(0, FrameKind::roster, roster_payload(ranks, budgets, "a digest nobody computed")));
        Session session = h.session(2, good_join(2), good_config(2));
        const Failure f = session.start();
        check(f.code == FailureCode::roster_mismatch, "a digest that does not describe its roster is refused");
    }
    {
        Harness h;
        std::vector<JoinInfo> swapped = ranks;
        std::swap(swapped[1], swapped[2]);
        // Slot 1 carrying rank 2 is not a group in another order: the slots are the ranks.
        h.channel.recv_frames.push_back(
            from(0, FrameKind::roster, roster_payload(swapped, budgets, roster_digest(swapped))));
        Session session = h.session(2, good_join(2), good_config(2));
        const Failure f = session.start();
        check(f.code == FailureCode::bad_frame,
              "a roster whose slots are not the ranks is refused at the payload");
    }
    {
        // A roster that is in order but whose digest came from another order is caught by the recomputation.
        Harness h;
        std::vector<JoinInfo> swapped = ranks;
        std::swap(swapped[1], swapped[2]);
        h.channel.recv_frames.push_back(
            from(0, FrameKind::roster, roster_payload(ranks, budgets, roster_digest(swapped))));
        Session session = h.session(2, good_join(2), good_config(2));
        const Failure f = session.start();
        check(f.code == FailureCode::roster_mismatch,
              "and a digest that belongs to another arrangement is refused by recomputation");
    }
    {
        Harness h;
        std::vector<JoinInfo> other_run = ranks;
        other_run[1].session_uuid = kOtherUuid;
        h.channel.recv_frames.push_back(from(0, FrameKind::roster, roster_payload(other_run, budgets)));
        Session session = h.session(2, good_join(2), good_config(2));
        const Failure f = session.start();
        check(f.code == FailureCode::uuid_mismatch, "a roster carrying another run's rank is refused");
    }
    {
        Harness h;
        h.channel.recv_frames.push_back(
            from(0, FrameKind::roster, roster_payload(ranks, budgets)));
        h.channel.recv_frames.push_back(
            from(0, FrameKind::ready, make_ack(kUuid, 0, "some other digest")));
        Session session = h.session(2, good_join(2), good_config(2));
        const Failure f = session.start();
        check(f.code == FailureCode::roster_mismatch, "READY for a roster this rank did not acknowledge is refused");
    }
}

void a_rank_checks_itself_first() {
    std::printf("a rank refuses itself before it asks anyone\n");
    {
        Harness h;
        SessionConfig config = good_config(0);
        config.approved_nccl_version = "2.29.1";  // the approved build is not the one this rank resolved
        Session session = h.session(0, good_join(0), config);
        const Failure f = session.start();
        check(f.code == FailureCode::nccl_version_mismatch && h.channel.open_calls == 0,
              "a rank whose library is not the approved build never touches the wire");
    }
    {
        Harness h;
        SessionConfig config = good_config(0);
        config.approved_library_provenance = "sha256:approved-but-absent";
        Session session = h.session(0, good_join(0), config);
        check(session.start().code == FailureCode::provenance_mismatch && h.channel.open_calls == 0,
              "the same for a library whose provenance was never approved");
    }
    {
        Harness h;
        SessionConfig config = good_config(0);
        config.credential = Credential();
        Session session = h.session(0, good_join(0), config);
        check(session.start().code == FailureCode::credential, "and a session with no credential has no members");
    }
    {
        Harness h;
        SessionConfig config = good_config(0);
        config.session_uuid = "not-a-uuid";
        Session session = h.session(0, good_join(0), config);
        check(session.start().code == FailureCode::uuid_mismatch,
              "a session id that is not a fresh canonical UUID is refused, not repaired");
    }
    {
        Harness h;
        SessionConfig config = good_config(0);
        config.world = 2;
        config.budgets = PhaseBudgets{0, 1, 1, 1};
        JoinInfo self = good_join(0);
        self.world = 2;
        Session session = h.session(0, self, config);
        check(session.start().code == FailureCode::rank_range, "two ranks is not a TP3 session");
    }
    {
        Harness h;
        SessionConfig config = good_config(0);
        config.budgets.join_ms = 0;
        Session session = h.session(0, good_join(0), config);
        check(session.start().code == FailureCode::invalid_config, "a phase with no time in it is a mistake");
    }
    {
        Harness h;
        JoinInfo self = good_join(0);
        self.nccl_bootstrap_id.clear();
        Session session = h.session(0, self, good_config(0));
        check(session.start().code == FailureCode::invalid_config,
              "and a rank with no bootstrap id has nothing to build a communicator from");
    }
    {
        Harness h;
        h.channel.open_result = Failure::fail(FailureCode::listen_failed, "port in use");
        Session session = h.session(0, good_join(0), good_config(0));
        const Failure f = session.start();
        check(f.code == FailureCode::listen_failed && f.detail.find("port in use") != std::string::npos,
              "a control channel that will not open says why");
    }
}

void session_lifecycle_is_final() {
    std::printf("a session forms once and is released once\n");
    Harness h;
    const Roster roster{good_roster(), roster_digest(good_roster()), PhaseBudgets{1000, 2000, 3000, 4000}};
    h.channel.recv_frames.push_back(from(1, FrameKind::join, join_payload(good_join(1))));
    h.channel.recv_frames.push_back(from(2, FrameKind::join, join_payload(good_join(2))));
    h.channel.recv_frames.push_back(from(1, FrameKind::roster_ack, make_ack(kUuid, 1, roster.digest)));
    h.channel.recv_frames.push_back(from(2, FrameKind::roster_ack, make_ack(kUuid, 2, roster.digest)));
    Session session = h.session(0, good_join(0), good_config(0));
    check(!session.start(), "formed");
    check(session.start().code == FailureCode::stale_join, "a session that already formed does not form again");

    check(!session.expect_operation(Operation{1, 0, 0}), "the first collective of a step is the one expected");
    check(session.expect_operation(Operation{1, 0, 0}).code == FailureCode::bad_operation,
          "the same sequence twice is an ordering mistake caught at rank 0");
    check(session.expect_operation(Operation{3, 0, 0}).code == FailureCode::bad_operation,
          "and so is skipping one");

    check(!session.close(), "released");
    check(!session.close() && !session.close(), "releasing it twice and three times is still safe");
    check(session.state() == SessionState::closed, "and the session stays released");
    check(session.start().code == FailureCode::closed, "a released session does not rejoin");
    const int after = h.channel.close_calls;
    session.abort("nothing left to abort");
    check(h.channel.close_calls == after, "aborting a closed session asks the channel for nothing");
}

void abort_from_a_healthy_session() {
    std::printf("aborting a session that is fine tells the others\n");
    Harness h;
    h.channel.recv_frames.push_back(from(1, FrameKind::join, join_payload(good_join(1))));
    h.channel.recv_frames.push_back(from(2, FrameKind::join, join_payload(good_join(2))));
    h.channel.recv_frames.push_back(
        from(1, FrameKind::roster_ack, make_ack(kUuid, 1, roster_digest(good_roster()))));
    h.channel.recv_frames.push_back(
        from(2, FrameKind::roster_ack, make_ack(kUuid, 2, roster_digest(good_roster()))));
    Session session = h.session(0, good_join(0), good_config(0));
    check(!session.start(), "formed");

    session.abort("the request was cancelled at a safe boundary");
    check(session.failure().code == FailureCode::aborted, "the session is now aborted");
    check(session.failure().detail.find("safe boundary") != std::string::npos, "with the reason it was given");
    check(h.channel.sends_of(FrameKind::abort) >= 1, "and the peers were told");
    session.abort("a second reason");
    check(session.failure().detail.find("second reason") == std::string::npos,
          "the first reason is the one that stands");
    std::string left_over;
    check(session.bootstrap_id(1, left_over).code == FailureCode::not_ready && left_over.empty(),
          "an aborted session hands out no roster");
}

void credential_rules() {
    std::printf("the credential comes from outside the source\n");
    const char* name = "STRATA_TP3_TEST_CREDENTIAL";
    unsetenv(name);
    Credential credential;
    Failure f = Credential::from_env(name, credential);
    check(f.code == FailureCode::credential, "an unset variable is not an empty credential");

    setenv(name, "short", 1);
    f = Credential::from_env(name, credential);
    check(f.code == FailureCode::credential, "a credential anyone could guess by accident is refused");

    setenv(name, "a-credential-that-is-long-enough-to-identify-a-run", 1);
    f = Credential::from_env(name, credential);
    check(!f && credential.secret() == "a-credential-that-is-long-enough-to-identify-a-run",
          "a real value is read from the environment");
    check(credential.redacted().find("a-credential") == std::string::npos,
          "and the way it is printed does not contain it");
    check(credential.redacted().find("len=") != std::string::npos, "it says how long it was, which is enough");
    unsetenv(name);

    check(Credential::from_env(nullptr, credential).code == FailureCode::invalid_config,
          "and a session that names no variable at all is a configuration mistake");
}

// ---------------------------------------------------------------- the frame codec, byte by byte

void frame_codec() {
    std::printf("the frame codec, byte by byte\n");
    std::string frame;
    check(!encode_frame(FrameKind::join, "hello", frame) && frame.size() == kFrameHeaderBytes + 5,
          "a frame is its header plus its payload");
    FrameDecoder decoder;
    std::vector<DecodedFrame> out;
    check(!decoder.feed(frame.data(), frame.size(), out) && out.size() == 1 && out[0].payload == "hello",
          "and it decodes back to what went in");

    out.clear();
    FrameDecoder split;
    check(!split.feed(frame.data(), 3, out) && out.empty(), "three bytes are not a frame yet");
    check(!split.feed(frame.data() + 3, frame.size() - 3, out) && out.size() == 1,
          "and the rest of it completes the frame");

    {
        std::string too_long = frame;
        too_long[6] = 0x01;  // a payload length of 0x01000000: sixteen megabytes in a control frame
        too_long[7] = 0x00;
        FrameDecoder d;
        std::vector<DecodedFrame> frames;
        const Failure f = d.feed(too_long.data(), too_long.size(), frames);
        check(f.code == FailureCode::oversized_frame && frames.empty() && d.buffered() == too_long.size(),
              "an absurd length is refused from the header, before the payload is read");
    }
    {
        FrameDecoder partial;
        std::vector<DecodedFrame> nothing;
        check(!partial.feed("ZZZZZZZZ", 8, nothing) && nothing.empty() && partial.buffered() == 8,
              "eight bytes are not a header yet, so nothing is judged on them");
    }
    {
        FrameDecoder d;
        std::vector<DecodedFrame> frames;
        // A full header's worth of nonsense: fewer bytes than the header and the decoder is still waiting, which
        // is the right answer but not the one this case is about.
        check(d.feed("ZZZZZZZZZZZZ", 12, frames).code == FailureCode::bad_frame,
              "a stream that is not our framing is refused at the magic");
        check(frames.empty(), "and nothing was handed on from a stream that is not ours");
    }
    {
        std::string other = frame;
        other[3] = 0x02;  // protocol version 2
        FrameDecoder d;
        std::vector<DecodedFrame> frames;
        check(d.feed(other.data(), other.size(), frames).code == FailureCode::protocol_version,
              "a peer from another control protocol is refused, not negotiated");
    }
    {
        std::string unknown = frame;
        unknown[5] = 0x0b;  // a kind this build has no name for
        FrameDecoder d;
        std::vector<DecodedFrame> frames;
        check(d.feed(unknown.data(), unknown.size(), frames).code == FailureCode::unknown_frame,
              "a frame kind this build does not know is refused");
    }
    {
        std::string bad_abort;
        check(!encode_abort(FailureCode::comm_failed, "a reason", bad_abort),
              "an abort carries its code and its words");
        FailureCode code = FailureCode::none;
        std::string reason;
        check(!decode_abort(bad_abort, code, reason) && code == FailureCode::comm_failed &&
                  reason == "a reason",
              "and they come back out");
        check(encode_abort(FailureCode::none, "nothing", bad_abort).code == FailureCode::bad_frame,
              "but an abort of nothing is not a thing");
        const uint32_t beyond = static_cast<uint32_t>(FailureCode::closed) + 1;
        std::string raw;  // four bytes of code and one empty field
        auto put32 = [&raw](uint32_t v) {
            raw.push_back(static_cast<char>((v >> 24) & 0xff));
            raw.push_back(static_cast<char>((v >> 16) & 0xff));
            raw.push_back(static_cast<char>((v >> 8) & 0xff));
            raw.push_back(static_cast<char>(v & 0xff));
        };
        put32(beyond);
        put32(0);
        check(decode_abort(raw, code, reason).code == FailureCode::bad_frame,
              "and a code from a build that means something else by the number is refused");
    }
    {
        std::string joined;
        check(!encode_join(good_join(1), joined), "a join encodes");
        JoinInfo back;
        check(!decode_join(joined, back) && canonical_join(back) == canonical_join(good_join(1)) &&
                  back.nccl_bootstrap_id == bootstrap_of(1),
              "and it comes back field for field, bootstrap id included");
        std::string extra = joined + "x";
        check(decode_join(extra, back).code == FailureCode::bad_frame,
              "a payload with one byte more than the format defines is not swallowed quietly");
        check(decode_join(joined.substr(0, joined.size() - 6), back).code == FailureCode::truncated_frame,
              "and a payload that stops inside a field is a truncated frame, not a default");
        check(!is_canonical_uuid("not-a-uuid") && is_canonical_uuid(kUuid) &&
                  !is_canonical_uuid(std::string(kUuid) + "0"),
              "a session id is a fresh canonical UUID or it is not accepted");
    }
}

// ---------------------------------------------------------------- the codec, on real sockets

uint16_t pick_free_port() {
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return 0;
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = 0;
    if (::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof addr) != 0) {
        ::close(fd);
        return 0;
    }
    socklen_t len = sizeof addr;
    uint16_t port = 0;
    if (::getsockname(fd, reinterpret_cast<sockaddr*>(&addr), &len) == 0) port = ntohs(addr.sin_port);
    ::close(fd);
    return port;
}

// A peer made of raw bytes: what a rank that does not speak this protocol at all looks like from the other side.
class RawPeer {
 public:
    bool connect(uint16_t port) {
        fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
        if (fd_ < 0) return false;
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(port);
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        if (::connect(fd_, reinterpret_cast<sockaddr*>(&addr), sizeof addr) != 0) {
            drop();
            return false;
        }
        timeval tv{1, 0};
        ::setsockopt(fd_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
        return true;
    }

    bool send_raw(const std::string& bytes) {
        size_t sent = 0;
        while (sent < bytes.size()) {
            const ssize_t n = ::send(fd_, bytes.data() + sent, bytes.size() - sent, MSG_NOSIGNAL);
            if (n <= 0) return false;
            sent += static_cast<size_t>(n);
        }
        return true;
    }

    bool send_hello(const std::string& credential) {
        std::string frame;
        if (encode_frame(FrameKind::hello, credential, frame)) return false;
        return send_raw(frame);
    }

    void finish_writes() { if (fd_ >= 0) ::shutdown(fd_, SHUT_WR); }

    void drop() {
        if (fd_ >= 0) {
            ::close(fd_);
            fd_ = -1;
        }
    }

    ~RawPeer() { drop(); }

 private:
    int fd_ = -1;
};

struct TcpOutcome {
    FailureCode code = FailureCode::none;
    bool ready = false;
    std::string digest;
    std::string failure_detail;
};

TcpOutcome run_rank(uint32_t rank, uint16_t port, const std::string& digest_expected) {
    SystemClock clock;
    TcpChannelConfig channel_config;
    channel_config.rank = rank;
    channel_config.world = kTpWorld;
    channel_config.self = TcpEndpoint{"127.0.0.1", port};
    channel_config.rank0 = TcpEndpoint{"127.0.0.1", port};
    channel_config.credential = kCredential;
    std::string err;
    std::unique_ptr<ControlChannel> channel = make_tcp_control_channel(channel_config, clock, err);
    TcpOutcome outcome;
    if (channel == nullptr) {
        outcome.code = FailureCode::listen_failed;
        outcome.failure_detail = err;
        return outcome;
    }
    SessionConfig config = good_config(rank);
    config.budgets = PhaseBudgets{4000, 6000, 6000, 6000};
    Session session(config, good_join(rank), *channel, clock, [](std::string_view) {});
    const Failure f = session.start();
    outcome.code = f.code;
    outcome.failure_detail = f.detail;
    outcome.ready = session.ready();
    if (session.ready()) outcome.digest = session.roster().digest;
    (void)digest_expected;
    session.close();
    return outcome;
}

void a_follower_starts_before_its_coordinator() {
    std::printf("a rank that starts before rank 0 keeps trying, up to its deadline\n");
    const uint16_t port = pick_free_port();
    check(port != 0, "a port for the launch race");
    if (port == 0) return;

    TcpOutcome outcomes[kTpWorld];
    std::thread first_follower([&]() { outcomes[1] = run_rank(1, port, std::string()); });
    std::this_thread::sleep_for(std::chrono::milliseconds(300));  // long enough that the refusal really happened
    std::thread second_follower([&]() { outcomes[2] = run_rank(2, port, std::string()); });
    outcomes[0] = run_rank(0, port, std::string());
    first_follower.join();
    second_follower.join();

    check(outcomes[0].ready && outcomes[1].ready && outcomes[2].ready,
          "a coordinator that comes up after its peers is a start order, not a disagreement");
    for (uint32_t rank = 0; rank < kTpWorld; ++rank) {
        if (!outcomes[rank].ready) {
            std::printf("    rank %u: %s %s\n", rank, failure_code_name(outcomes[rank].code),
                        outcomes[rank].failure_detail.c_str());
        }
    }
}

void three_ranks_over_real_tcp() {
    std::printf("three ranks over real loopback TCP\n");
    bool done = false;
    for (int attempt = 0; attempt < 3 && !done; ++attempt) {
        const uint16_t port = pick_free_port();
        if (port == 0) continue;
        std::atomic<int> ready{0};
        TcpOutcome outcomes[kTpWorld];
        std::vector<std::thread> threads;
        for (uint32_t rank = 0; rank < kTpWorld; ++rank) {
            threads.emplace_back([&, rank]() {
                outcomes[rank] = run_rank(rank, port, std::string());
                if (outcomes[rank].ready) ++ready;
            });
        }
        for (auto& t : threads) t.join();
        done = ready == kTpWorld;
        check(done, "all three ranks reach READY over real sockets");
        if (!done) {
            for (uint32_t rank = 0; rank < kTpWorld; ++rank) {
                if (!outcomes[rank].ready) {
                    std::printf("    rank %u: %s %s\n", rank, failure_code_name(outcomes[rank].code),
                                outcomes[rank].failure_detail.c_str());
                }
            }
        } else {
            check(outcomes[0].digest == outcomes[1].digest && outcomes[1].digest == outcomes[2].digest &&
                      !outcomes[0].digest.empty(),
                  "and every rank holds the same verified roster");
        }
    }
}

void collective_agreement_over_tcp() {
    for (int mismatch = -1; mismatch < 3; ++mismatch) {
        const uint16_t port = pick_free_port();
        check(port != 0, "a TCP port for collective agreement");
        if (!port) return;
        std::atomic<int> finished{0};
        FailureCode results[kTpWorld]{};
        std::string details[kTpWorld];
        std::vector<std::thread> threads;
        for (uint32_t rank = 0; rank < kTpWorld; ++rank) {
            threads.emplace_back([&, rank] {
                SystemClock clock;
                TcpChannelConfig wire;
                wire.rank = rank;
                wire.self = wire.rank0 = TcpEndpoint{"127.0.0.1", port};
                wire.credential = kCredential;
                std::string err;
                auto channel = make_tcp_control_channel(wire, clock, err);
                auto config = good_config(rank);
                config.budgets = PhaseBudgets{3000, 3000, 3000, 3000};
                Session session(config, good_join(rank), *channel, clock, [](std::string_view) {});
                Failure f = session.start();
                for (uint64_t sequence = 1; !f && sequence <= 3; ++sequence) {
                    CollectiveDescriptor d{{sequence, 2, 3}, CollectiveKind::all_gather_padded,
                                            ScalarType::fp32, 256, {256, 192, 192}};
                    if (rank == 2 && mismatch == 0) ++d.operation.sequence;
                    if (rank == 2 && mismatch == 1) --d.valid_counts[2];
                    if (rank == 2 && mismatch == 2) { ++d.elements; ++d.valid_counts[0]; }
                    f = session.agree_operation(d, Deadline{clock.now() + std::chrono::seconds(3)});
                    if (!f) f = session.complete_operation(d.operation);
                }
                results[rank] = f.code;
                details[rank] = f.detail;
                ++finished;
                const auto end = clock.now() + std::chrono::seconds(3);
                while (finished != 3 && clock.now() < end) std::this_thread::yield();
                session.close();
            });
        }
        for (auto& thread : threads) thread.join();
        for (uint32_t rank = 0; rank < kTpWorld; ++rank) {
            const auto code = results[rank];
            check(code == (mismatch < 0 ? FailureCode::none : FailureCode::bad_operation),
                  mismatch < 0 ? "three TCP ranks agree repeated collective shapes" : "TCP metadata mismatch aborts every rank before enqueue");
            if (code != (mismatch < 0 ? FailureCode::none : FailureCode::bad_operation))
                std::printf("    mismatch %d rank %u: %s: %s\n", mismatch, rank,
                            failure_code_name(code), details[rank].c_str());
        }
    }
}

void control_payload_agreement_over_tcp() {
    // The control protocol carries bytes, never floats or a lossy numerical digest.
    for (int mismatch = -1; mismatch < 6; ++mismatch) {
        const uint16_t port = pick_free_port();
        check(port != 0, "a TCP port for exact control agreement");
        if (!port) return;
        std::atomic<int> finished{0};
        FailureCode results[kTpWorld]{};
        std::string details[kTpWorld];
        uint32_t committed[kTpWorld]{};
        std::vector<std::thread> threads;
        for (uint32_t rank = 0; rank < kTpWorld; ++rank) {
            threads.emplace_back([&, rank] {
                SystemClock clock;
                TcpChannelConfig wire;
                wire.rank = rank;
                wire.self = wire.rank0 = TcpEndpoint{"127.0.0.1", port};
                wire.credential = kCredential;
                std::string err;
                auto channel = make_tcp_control_channel(wire, clock, err);
                auto config = good_config(rank);
                config.budgets = PhaseBudgets{3000, 3000, 3000, 3000};
                Session session(config, good_join(rank), *channel, clock, [](std::string_view) {});
                Failure f = session.start();
                for (uint64_t sequence = 1; !f && sequence <= 4; ++sequence) {
                    std::string label("route\0order", 11);
                    // Exercise the full envelope cap, including fields much larger than the join field cap.
                    const size_t overhead = 4 + std::strlen(kUuid) + 8 + 4 + label.size() + 4;
                    std::string payload(kControlMaxPayload - overhead, '\0');
                    payload.front() = static_cast<char>(sequence);
                    payload.back() = static_cast<char>(0xff);
                    if (mismatch >= 0 && sequence == 2) {
                        if ((mismatch == 0 && rank == 1) || (mismatch == 1 && rank == 2) ||
                            (mismatch == 2 && rank == 0)) payload[payload.size() / 2] ^= 1;
                        if (mismatch == 3 && rank == 2) label[2] ^= 1;
                        if (mismatch == 4 && rank == 2) {
                            payload.insert(payload.begin(), label.back());
                            label.pop_back();
                        }
                        if (mismatch == 5 && rank == 2) {
                            session.close();
                            f = Failure::fail(FailureCode::peer_closed, "deliberate test disconnect");
                            break;
                        }
                    }
                    f = session.agree_control_payload(label, payload,
                                                     {clock.now() + std::chrono::seconds(3)});
                    if (!f) ++committed[rank];
                    if (!f) {
                        CollectiveDescriptor d{{sequence, 2, 3}, CollectiveKind::all_reduce_sum,
                                                ScalarType::fp32, 1, {}};
                        f = session.agree_operation(d, {clock.now() + std::chrono::seconds(3)});
                        if (!f) f = session.complete_operation(d.operation);
                    }
                }
                results[rank] = f.code;
                details[rank] = f.detail;
                ++finished;
                const auto end = clock.now() + std::chrono::seconds(3);
                while (finished != 3 && clock.now() < end) std::this_thread::yield();
                session.close();
            });
        }
        for (auto& thread : threads) thread.join();
        for (uint32_t rank = 0; rank < kTpWorld; ++rank) {
            const FailureCode expected = mismatch < 0 ? FailureCode::none :
                mismatch == 5 ? FailureCode::peer_closed : FailureCode::bad_operation;
            check(results[rank] == expected && committed[rank] == (mismatch < 0 ? 4u : 1u),
                  mismatch < 0 ? "three TCP ranks repeatedly agree exact maximum binary payload before compute" :
                                 "TCP byte/label mismatch or disconnect aborts every rank before next compute");
            if (results[rank] != expected)
                std::printf("    mismatch %d rank %u: %s: %s\n", mismatch, rank,
                            failure_code_name(results[rank]), details[rank].c_str());
        }
    }
}

void control_payload_broadcast_over_tcp() {
    for (int failure = -1; failure < 3; ++failure) {
        const uint16_t port = pick_free_port();
        check(port != 0, "a TCP port for rank-zero control broadcast");
        if (!port) return;
        std::atomic<int> finished{0};
        FailureCode results[kTpWorld]{};
        std::string details[kTpWorld];
        uint32_t committed[kTpWorld]{};
        bool exact[kTpWorld]{true, true, true};
        std::vector<std::thread> threads;
        for (uint32_t rank = 0; rank < kTpWorld; ++rank) {
            threads.emplace_back([&, rank] {
                SystemClock clock;
                TcpChannelConfig wire;
                wire.rank = rank;
                wire.self = wire.rank0 = TcpEndpoint{"127.0.0.1", port};
                wire.credential = kCredential;
                std::string err;
                auto channel = make_tcp_control_channel(wire, clock, err);
                auto config = good_config(rank);
                config.budgets = PhaseBudgets{3000, 3000, 3000, 3000};
                Session session(config, good_join(rank), *channel, clock, [](std::string_view) {});
                Failure f = session.start();
                for (uint64_t sequence = 1; !f && sequence <= 6; ++sequence) {
                    std::string label("action\0step", 11);
                    const size_t overhead = 4 + 4 + std::strlen(kUuid) + 8 + 4 + label.size() + 4;
                    std::string expected(kControlMaxPayload - overhead, '\0');
                    expected.front() = static_cast<char>(sequence);
                    expected.back() = static_cast<char>(0xff);
                    if (sequence == 4) expected.clear();
                    std::string output = rank == 0 ? expected : std::string{};
                    if (sequence == 2 && rank == 2) {
                        if (failure == 0) label[1] ^= 1;
                        if (failure == 1) output = "forbidden-follower-input";
                        if (failure == 2) {
                            session.close();
                            f = Failure::fail(FailureCode::peer_closed, "deliberate test disconnect");
                            break;
                        }
                    }
                    if (sequence % 2 == 0) {
                        const auto before = output;
                        f = session.broadcast_control_payload(label, output,
                                                              {clock.now() + std::chrono::seconds(3)});
                        exact[rank] &= output == (f ? before : expected);
                    } else {
                        f = session.agree_control_payload(label, expected,
                                                          {clock.now() + std::chrono::seconds(3)});
                    }
                    if (!f) ++committed[rank];
                    if (!f) {
                        CollectiveDescriptor d{{sequence, 2, 3}, CollectiveKind::all_reduce_sum,
                                                ScalarType::fp32, 1, {}};
                        f = session.agree_operation(d, {clock.now() + std::chrono::seconds(3)});
                        if (!f) f = session.complete_operation(d.operation);
                    }
                }
                results[rank] = f.code;
                details[rank] = f.detail;
                ++finished;
                const auto end = clock.now() + std::chrono::seconds(3);
                while (finished != 3 && clock.now() < end) std::this_thread::yield();
                session.close();
            });
        }
        for (auto& thread : threads) thread.join();
        for (uint32_t rank = 0; rank < kTpWorld; ++rank) {
            const auto expected = failure < 0 ? FailureCode::none :
                failure == 2 ? FailureCode::peer_closed : FailureCode::bad_operation;
            check(results[rank] == expected && exact[rank] && committed[rank] == (failure < 0 ? 6u : 1u),
                  failure < 0 ? "three TCP ranks mix agreement and max-size/empty binary broadcasts before collectives" :
                                "TCP broadcast disagreement or disconnect fails all ranks without publishing output");
            if (results[rank] != expected)
                std::printf("    failure %d rank %u: %s: %s\n", failure, rank,
                            failure_code_name(results[rank]), details[rank].c_str());
        }
    }
}

void queued_abort_survives_tcp_reset() {
    const int listener = ::socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    socklen_t length = sizeof address;
    const bool listening = listener >= 0 &&
        ::bind(listener, reinterpret_cast<sockaddr*>(&address), sizeof address) == 0 &&
        ::getsockname(listener, reinterpret_cast<sockaddr*>(&address), &length) == 0 &&
        ::listen(listener, 1) == 0;
    check(listening, "a raw coordinator for the queued ABORT/reset regression");
    if (!listening) { if (listener >= 0) ::close(listener); return; }
    std::atomic<bool> reset_sent{false};
    std::thread server([&] {
        const int peer = ::accept(listener, nullptr, nullptr);
        if (peer >= 0) {
            std::string hello, abort_payload, abort_frame;
            encode_frame(FrameKind::hello, kCredential, hello);
            encode_abort(FailureCode::bad_operation, "metadata mismatch", abort_payload);
            encode_frame(FrameKind::abort, abort_payload, abort_frame);
            // Let open() finish its hello write before resetting the connection.
            std::string client_hello(hello.size(), '\0');
            size_t received = 0;
            while (received < client_hello.size()) {
                const ssize_t count = ::recv(peer, client_hello.data() + received,
                                              client_hello.size() - received, 0);
                if (count <= 0) break;
                received += static_cast<size_t>(count);
            }
            const std::string frames = hello + abort_frame;
            ::send(peer, frames.data(), frames.size(), MSG_NOSIGNAL);
            // Force RST with readable frames queued at the client. No sleep or scheduler race is needed:
            // recv below starts only once the reset has been sent.
            const linger reset{1, 0};
            ::setsockopt(peer, SOL_SOCKET, SO_LINGER, &reset, sizeof reset);
            ::close(peer);
        }
        reset_sent = true;
    });
    SystemClock clock;
    TcpChannelConfig config;
    config.rank = 1;
    config.rank0 = {"127.0.0.1", ntohs(address.sin_port)};
    config.credential = kCredential;
    std::string err;
    auto channel = make_tcp_control_channel(config, clock, err);
    const Deadline at{clock.now() + std::chrono::seconds(3)};
    const Failure opened = channel->open(at);
    while (!reset_sent) std::this_thread::yield();
    server.join();
    ::close(listener);
    check(!opened, "the follower control link opened before the raw coordinator reset it");
    if (opened) return;
    ReceivedFrame received;
    const Failure result = channel->recv(received, at);
    check(!result && received.frame.kind == FrameKind::abort,
          "a queued ABORT is decoded before a simultaneous POLLERR/reset");
    channel->close();
}

void a_rank0_session_with_one_raw_peer(FailureCode expected, const std::string& what,
                                      const std::function<void(RawPeer&)>& script,
                                      std::vector<std::string>* lines = nullptr) {
    const uint16_t port = pick_free_port();
    check(port != 0, "a port for: " + what);
    if (port == 0) return;

    std::atomic<FailureCode> code{FailureCode::none};
    std::string detail;
    std::vector<std::string> captured;
    std::thread worker([&]() {
        SystemClock clock;
        TcpChannelConfig channel_config;
        channel_config.rank = 0;
        channel_config.world = kTpWorld;
        channel_config.self = TcpEndpoint{"127.0.0.1", port};
        channel_config.credential = kCredential;
        std::string err;
        std::unique_ptr<ControlChannel> channel = make_tcp_control_channel(channel_config, clock, err);
        if (channel == nullptr) {
            code = FailureCode::listen_failed;
            detail = err;
            return;
        }
        SessionConfig config = good_config(0);
        config.budgets = PhaseBudgets{4000, 4000, 4000, 4000};
        Session session(config, good_join(0), *channel, clock, [&captured](std::string_view line) {
            captured.emplace_back(line);
        });
        const Failure f = session.start();
        code = f.code;
        detail = f.detail;
        session.close();
    });

    RawPeer peer;
    bool up = false;
    for (int i = 0; i < 100 && !up; ++i) {
        up = peer.connect(port);
        if (!up) std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    if (!up) {
        worker.join();
        check(false, "the raw peer could not reach the listener: " + what);
        return;
    }
    script(peer);
    worker.join();
    check(code == expected, what);
    if (code != expected) std::printf("    got %s: %s\n", failure_code_name(code), detail.c_str());
    if (lines != nullptr) *lines = captured;
}

struct DeadlineProbe {
    FailureCode code = FailureCode::none;
    long long elapsed_ms = 0;
};

// Rank 0 with nobody to join it: the only thing that can end the session is its own deadline, which is exactly
// what makes this a measurement of the deadline.
DeadlineProbe rank0_with_no_peers(uint16_t port, uint32_t join_ms) {
    SystemClock clock;
    TcpChannelConfig channel_config;
    channel_config.rank = 0;
    channel_config.world = kTpWorld;
    channel_config.self = TcpEndpoint{"127.0.0.1", port};
    channel_config.credential = kCredential;
    std::string err;
    std::unique_ptr<ControlChannel> channel = make_tcp_control_channel(channel_config, clock, err);
    DeadlineProbe probe;
    if (channel == nullptr) {
        probe.code = FailureCode::listen_failed;
        return probe;
    }
    SessionConfig config = good_config(0);
    config.budgets = PhaseBudgets{2000, join_ms, 2000, 2000};
    const auto started = std::chrono::steady_clock::now();
    Session session(config, good_join(0), *channel, clock, [](std::string_view) {});
    const Failure f = session.start();
    probe.elapsed_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - started).count();
    probe.code = f.code;
    session.close();
    return probe;
}

void a_phase_waits_its_own_length() {
    std::printf("a phase waits its own length, not the length of one wait on the socket\n");
    const uint16_t port = pick_free_port();
    check(port != 0, "a port for the deadline length");
    if (port == 0) return;
    const DeadlineProbe probe = rank0_with_no_peers(port, 6000);
    check(probe.code == FailureCode::missing_rank, "a session with no peers ends on the join deadline");
    std::printf("    waited %lld ms for a 6000 ms budget\n", probe.elapsed_ms);
    check(probe.elapsed_ms >= 5900 && probe.elapsed_ms < 7500,
          "and it waits the six seconds it was given, not the five seconds of one poll slice");
}

void raw_peers_over_real_tcp() {
    std::printf("raw peers over real loopback TCP\n");
    a_rank0_session_with_one_raw_peer(
        FailureCode::truncated_frame, "EOF inside a frame is reported as a truncated frame", [](RawPeer& peer) {
            peer.send_hello(kCredential);
            std::string frame;
            encode_frame(FrameKind::join, join_payload(good_join(1)), frame);
            frame.resize(frame.size() - 1);
            peer.send_raw(frame);
            peer.finish_writes();
        });
    a_rank0_session_with_one_raw_peer(
        FailureCode::model_mismatch, "a complete frame larger than one recv survives immediate EOF", [](RawPeer& peer) {
            peer.send_hello(kCredential);
            JoinInfo info = good_join(1);
            info.model_hash.assign(4096, 'x');
            std::string frame;
            encode_frame(FrameKind::join, join_payload(info), frame);
            peer.send_raw(frame);
            peer.finish_writes();
        });
    a_rank0_session_with_one_raw_peer(
        FailureCode::oversized_frame, "a frame whose length is absurd is refused on the socket",
        [](RawPeer& peer) {
            peer.send_hello(kCredential);
            std::string frame;
            encode_frame(FrameKind::join, std::string(4, 'x'), frame);
            frame[6] = 0x01;  // 0x0100 = 256 bytes... claim far more than the cap instead
            frame[7] = 0x00;
            frame += std::string(20, 'y');  // some bytes behind it: the decision must precede reading them
            peer.send_raw(frame);
        });
    a_rank0_session_with_one_raw_peer(
        FailureCode::credential, "a peer with someone else's credential is refused", [](RawPeer& peer) {
            peer.send_hello("not-this-runs-credential-at-all");
        });
    a_rank0_session_with_one_raw_peer(
        FailureCode::bad_frame, "a stream that is not our framing at all is refused", [](RawPeer& peer) {
            peer.send_raw(std::string(12, 'Z'));
        });
    a_rank0_session_with_one_raw_peer(
        FailureCode::protocol_version, "a peer from another control protocol is refused", [](RawPeer& peer) {
            std::string frame;
            encode_frame(FrameKind::hello, kCredential, frame);
            frame[3] = 0x02;
            peer.send_raw(frame);
        });
    a_rank0_session_with_one_raw_peer(
        FailureCode::peer_closed, "a peer that connects, helloes and then dies is a peer that died",
        [](RawPeer& peer) {
            peer.send_hello(kCredential);
            peer.drop();
        });
    a_rank0_session_with_one_raw_peer(
        FailureCode::missing_rank, "a join deadline that passes names the rank that never came",
        [](RawPeer&) {});
    {
        // A peer can write its join and die in the same breath.  The join arrived, so the loss report has to
        // come after it: a session that throws away what it already read tells the reader the wrong story.  The
        // pause is what makes this a test of ordering rather than of scheduling - the peer stays long enough to
        // be authenticated (an unauthenticated link's frames are not usable at all), and then it goes.
        std::vector<std::string> lines;
        a_rank0_session_with_one_raw_peer(
            FailureCode::peer_closed,
            "a peer that sends its join and then closes is read before it is reported as lost",
            [](RawPeer& peer) {
                peer.send_hello(kCredential);
                std::string frame;
                encode_frame(FrameKind::join, join_payload(good_join(1)), frame);
                peer.send_raw(frame);
                std::this_thread::sleep_for(std::chrono::milliseconds(300));
                peer.drop();
            },
            &lines);
        bool accepted = false;
        for (const auto& line : lines) {
            if (line.find("join accepted from rank 1") != std::string::npos) accepted = true;
        }
        check(accepted && !lines.empty(), "and the session had recorded that join first");
    }
}

void tcp_factory_rules() {
    std::printf("the TCP channel refuses to be configured wrongly\n");
    SystemClock clock;
    std::string err;

    TcpChannelConfig no_port;
    no_port.rank = 0;
    no_port.self = TcpEndpoint{"127.0.0.1", 0};
    no_port.credential = kCredential;
    auto channel = make_tcp_control_channel(no_port, clock, err);
    check(channel != nullptr, "the channel itself is built lazily");
    check(channel->open(Deadline{std::chrono::steady_clock::now()}).code == FailureCode::invalid_config,
          "rank 0 with no port cannot open");

    TcpChannelConfig not_ipv4;
    not_ipv4.rank = 0;
    not_ipv4.self = TcpEndpoint{"spark-triangle", 12345};
    not_ipv4.credential = kCredential;
    channel = make_tcp_control_channel(not_ipv4, clock, err);
    check(channel != nullptr &&
              channel->open(Deadline{std::chrono::steady_clock::now()}).code == FailureCode::invalid_config,
          "an interface name is not an address: it is refused with the name in the message");

    TcpChannelConfig no_credential;
    no_credential.rank = 0;
    no_credential.self = TcpEndpoint{"127.0.0.1", 12345};
    channel = make_tcp_control_channel(no_credential, clock, err);
    check(channel != nullptr &&
              channel->open(Deadline{std::chrono::steady_clock::now()}).code == FailureCode::credential,
          "and a channel with no credential says so before it binds");

    TcpChannelConfig wrong_world;
    wrong_world.rank = 0;
    wrong_world.world = 4;
    check(make_tcp_control_channel(wrong_world, clock, err) == nullptr &&
              err.find("three") != std::string::npos,
          "a world other than three is refused by the factory");

    TcpChannelConfig bad_rank;
    bad_rank.rank = 3;
    check(make_tcp_control_channel(bad_rank, clock, err) == nullptr,
          "and so is a rank outside it");
}

}  // namespace

int main() {
    // No case here waits for a human.  If one does, this says which run stopped rather than leaving the suite
    // hanging with no answer.
    alarm(180);
    std::printf("session formation (CPU, real and scripted control channels, no GPU)\n");
    coordinator_reaches_ready();
    collective_agreement();
    control_payload_agreement();
    control_payload_broadcast();
    follower_reaches_ready();
    follower_adopts_the_smaller_budget();
    coordinator_refuses_a_group_that_does_not_agree();
    coordinator_refuses_a_wrong_group_shape();
    stale_and_dead_peers();
    a_roster_must_be_the_one_this_rank_said();
    a_rank_checks_itself_first();
    session_lifecycle_is_final();
    abort_from_a_healthy_session();
    credential_rules();
    frame_codec();
    tcp_factory_rules();
    three_ranks_over_real_tcp();
    collective_agreement_over_tcp();
    control_payload_agreement_over_tcp();
    control_payload_broadcast_over_tcp();
    queued_abort_survives_tcp_reset();
    a_follower_starts_before_its_coordinator();
    raw_peers_over_real_tcp();
    a_phase_waits_its_own_length();
    std::printf("%d checks, %d failed\n", g_checks, g_fail);
    return g_fail == 0 ? 0 : 1;
}
