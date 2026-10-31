// include/strata/distributed/session.hpp - how three ranks agree that they are the same run before any tensor
// byte moves: one bounded control conversation over TCP, then READY or nothing.
//
// TCP carries coordination only.  Tensor traffic is the transport's business (transport.hpp) and goes over
// whatever the device library picks; the control channel carries only bounded coordination metadata.
//
// What the join settles, and what happens when it does not:
//   * The same fresh session UUID on all three ranks.  A rank that kept yesterday's UUID is a rank from
//     another run, and a stale join must never be mistaken for a live peer.
//   * The same model (a content hash over manifest and tensor identity, not a path), the same configuration
//     fingerprint (family, quantization, geometry, shard map, dtype policy, execution protocol - host interface
//     names excluded, since each host has its own), the same backend identity, the same expected NCCL version
//     and the same library provenance.  A matching version number alone does not say the triangle patches are
//     in the library, so provenance is a separate field and both are compared against an approved value.
//   * Rank 0 coordinates: it collects the joins, and only when three verified ranks have acknowledged the same
//     roster does anyone see READY.  Every rank re-checks the roster it is told about instead of trusting
//     rank 0's word, because a rank that accepted a bad roster would sit in a collective the others never
//     entered.
//   * Every phase has a deadline from a monotonic clock, and only durations cross the wire - never a machine's
//     clock value.  A rank that is missing by its deadline ends its own session: a peer that is gone cannot
//     acknowledge an ABORT, so waiting for the broadcast is not enough.
//   * The session credential is read from the environment (never a literal in the source) and is redacted from
//     every log line the session writes, as are the NCCL bootstrap identifiers.
//
// Failure is terminal.  A session that failed to form is not retried: a rank that retried alone would be
// waiting in a join while its peers are somewhere else.  `close()` is idempotent and safe from any state.
//
// CPU-only surface: no CUDA, no NCCL, and no GPU.  The socket layer is POSIX (the Spark hosts and the NAS are
// Linux); on any other platform the TCP factory reports `not_supported` instead of pretending.
#pragma once

#include "strata/distributed/transport.hpp"

#include <cstdint>
#include <chrono>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

namespace strata::distributed {

// The control protocol this build speaks.  A peer that says anything else is refused, not negotiated down:
// two ranks running different control versions cannot agree on what READY means.
inline constexpr uint32_t kControlProtocolVersion = 1;

// The largest payload the control channel accepts.  Coordination frames are a few hundred bytes, so the cap is
// far above any honest frame and low enough that a wrong length field cannot make a rank allocate gigabytes.
inline constexpr uint32_t kControlMaxPayload = 64u * 1024u;

// The largest field inside a join.  Hashes and fingerprints are tens of bytes.
inline constexpr uint32_t kControlMaxField = 4096u;
// The exact roster keeps complete join records and an unhashed agreement token. These two
// protocol containers may be larger than a single identity field, but remain frame bounded.
inline constexpr uint32_t kControlMaxRosterDigest = 52u * 1024u;
inline constexpr uint32_t kControlMaxJoinRecord = 16u * 1024u;

// The largest phase budget a session may run.  A phase with no bound is not a phase, so a configured deadline
// above an hour is refused rather than accepted as "effectively forever".
inline constexpr uint32_t kMaxPhaseMs = 3'600'000u;

// Frames on the control channel.  The wire carries the numeric value; an unknown value is `unknown_frame`.
enum class FrameKind : uint16_t {
    hello = 1,       // the session credential, first frame on every link
    join = 2,        // a rank's identity and what it loaded
    roster = 3,      // rank 0 -> peers: the three verified joins, the roster digest, the phase durations
    roster_ack = 4,  // peer -> rank 0: this rank accepts that roster
    ready = 5,       // rank 0 -> peers: every rank accepted the same roster
    abort = 6,       // any -> peers: this session is over, and why
    operation = 7,
    operation_ack = 8,
    control = 9,
    control_ack = 10,
};

const char* frame_kind_name(FrameKind kind);

// A frame after the header was checked and the payload read.
struct DecodedFrame {
    FrameKind kind = FrameKind::join;
    std::string payload;
};

// The frame header, fixed at 10 bytes and in explicit byte order, so a little-endian and a big-endian build
// cannot disagree by accident: 'S','T' at 0..1, the protocol version as a big-endian u16 at 2..3, the frame kind
// as a big-endian u16 at 4..5, and the payload length as a big-endian u32 at 6..9.  The decoder's "is the whole
// frame here yet" test is `header + length`, so this number is part of the wire format, not a label: a header
// constant that disagrees with the encoder cuts two bytes off every payload and the first frame a rank reads
// already fails authentication.
inline constexpr uint8_t kFrameMagic0 = 'S';
inline constexpr uint8_t kFrameMagic1 = 'T';
inline constexpr size_t kFrameHeaderBytes = 10;

// Serialize one frame (header + payload).  Refuses a payload over `kControlMaxPayload` and an unknown kind.
Failure encode_frame(FrameKind kind, std::string_view payload, std::string& out);

// Byte-stream to frames.  Feed it whatever a read returned; it keeps the remainder and emits complete frames.
// A malformed frame is a failure and the stream is finished: after a bad header there is no trustworthy next
// frame, and guessing where it starts again is how two ranks end up reading each other's leftovers.
class FrameDecoder {
 public:
    Failure feed(const char* data, size_t n, std::vector<DecodedFrame>& out);

    // Bytes buffered but not yet a complete frame.  Bounded: a frame longer than header + payload cap fails.
    size_t buffered() const { return buffer_.size(); }

 private:
    std::string buffer_;
};

// One rank's claim about what it is and what it loaded.  Every string is the caller's to compute; the session
// only compares them, so this header invents no hash of its own.
struct JoinInfo {
    std::string session_uuid;         // canonical 8-4-4-4-12, fresh per run, identical on all three ranks
    uint32_t rank = 0;
    uint32_t world = kTpWorld;
    uint32_t protocol_version = kControlProtocolVersion;
    std::string model_hash;           // content identity of the manifest and tensor set, not a file path
    std::string config_fingerprint;   // family, quantization, geometry, shard map, dtype policy, exec protocol
    std::string backend_identity;     // e.g. "nccl"
    std::string nccl_version_expected;  // the version this rank's loader was built and approved against
    std::string library_provenance;   // which library build and patch set this rank resolved
    std::string nccl_bootstrap_id;    // opaque bootstrap id for the communicator; never logged
};

// The equality token for one join: every field in a fixed order, including its per-rank bootstrap id.
// A digest, in the sense of "the same string on every rank iff the joins
// agree", not a cryptographic hash - the hashes that feed it are computed by whoever read the model.
std::string canonical_join(const JoinInfo& info);

// The same idea over the whole group: the joins in rank order, so a roster that is complete but reordered,
// duplicated or missing a rank reads as a different string.  Every rank recomputes it from what it received and
// compares, rather than taking rank 0's word for what the group looks like.
std::string roster_digest(const std::vector<JoinInfo>& ranks);

// True for the canonical UUID form (lowercase or uppercase hex, 8-4-4-4-12).
bool is_canonical_uuid(std::string_view uuid);

// The phase durations rank 0 advertises.  Durations, not deadlines: each rank starts its own monotonic clock
// for a phase and applies the agreed length.
struct PhaseBudgets {
    uint32_t connect_ms = 10'000u;
    uint32_t join_ms = 15'000u;
    uint32_t initialize_ms = 30'000u;
    uint32_t operation_ms = 60'000u;
};

// A session credential.  Supplied from outside the source (an environment variable, or a caller that read a
// file) and never printed as-is: `redacted()` is what belongs in a log line.
class Credential {
 public:
    Credential() = default;
    explicit Credential(std::string secret) : secret_(std::move(secret)) {}

    // Read the credential from the named environment variable.  Refuses an unset, empty or trivially short
    // value: an unset variable must not read as "the credential is the empty string", which every peer would
    // happily match.
    static Failure from_env(const char* name, Credential& out);

    const std::string& secret() const { return secret_; }
    bool empty() const { return secret_.empty(); }
    std::string redacted() const;

 private:
    std::string secret_;
};

// A frame the channel received, and which peer sent it.
struct ReceivedFrame {
    uint32_t from_rank = 0;
    DecodedFrame frame;
};

// The control transport the session talks to.  One implementation is the TCP channel in control_tcp.cpp; the
// contract tests drive a scripted fake, so the protocol is tested without a network and the socket layer is
// tested with one.
//
// The channel owns framing and the credential handshake: the first frame of every link must be `hello` with a
// matching credential, and a link that fails it is closed before the session sees anything from it.  That
// keeps authentication out of the protocol, so a JOIN cannot arrive from a peer that never proved it belongs
// to this run.
class ControlChannel {
 public:
    virtual ~ControlChannel() = default;

    // Bring the channel up.  Rank 0 binds and listens; the other ranks connect to rank 0 and hello it.
    virtual Failure open(Deadline at) = 0;

    // One frame to one rank, or the reason it could not go out before `at`.
    virtual Failure send(uint32_t to_rank, FrameKind kind, std::string_view payload, Deadline at) = 0;

    // The next frame from any peer, or: `deadline_expired` (nothing in time), `peer_closed` (a link ended),
    // `oversized_frame` / `truncated_frame` / `protocol_version` / `unknown_frame` / `credential` (a peer that
    // spoke badly, which is a disagreement about the protocol and ends the session).
    virtual Failure recv(ReceivedFrame& out, Deadline at) = 0;

    // Idempotent; safe to call twice and from a destructor.
    virtual void close() noexcept = 0;
};

// A peer address for the TCP channel.  IPv4 only for this experiment: the Spark triangle's control links are
// IPv4, and a second address family nobody has run is worse than a clear refusal.
struct TcpEndpoint {
    std::string address;
    uint16_t port = 0;
};

struct TcpChannelConfig {
    uint32_t rank = 0;
    uint32_t world = kTpWorld;
    TcpEndpoint self;                        // rank 0: what to bind.  Other ranks: unused.
    TcpEndpoint rank0;                       // the coordinator to connect to.  Unused by rank 0.
    std::string credential;                  // the shared session credential (from Credential::from_env)
    std::string log_prefix = "strata-tp3";
};

// The TCP control channel.  Non-blocking by construction: every wait inside takes the deadline it is given.
// Returns nullptr with the reason in `err` when the channel could not be created.
std::unique_ptr<ControlChannel> make_tcp_control_channel(const TcpChannelConfig& config, const Clock& clock,
                                                        std::string& err);

// The verified view of the group.  `ranks[i]` is rank i's join, and `digest` is the digest everyone
// acknowledged.
struct Roster {
    std::vector<JoinInfo> ranks;
    std::string digest;
    PhaseBudgets budgets{};                  // what rank 0 advertised and every rank adopted
};

// Payload codecs.  Each field is a big-endian u32 length plus bytes, in a fixed order, and a decoder refuses
// trailing bytes: a frame that does not decode exactly is not a frame we understand.
Failure encode_join(const JoinInfo& info, std::string& out);
Failure decode_join(std::string_view payload, JoinInfo& out);
Failure encode_budgets(const PhaseBudgets& budgets, std::string& out);
Failure decode_budgets(std::string_view payload, PhaseBudgets& out);
Failure encode_roster(const Roster& roster, std::string& out);
Failure decode_roster(std::string_view payload, Roster& out);
Failure encode_abort(FailureCode code, std::string_view reason, std::string& out);
Failure decode_abort(std::string_view payload, FailureCode& code, std::string& reason);

// How far a session got.  Only `ready` may issue a collective.
enum class SessionState : uint32_t { fresh = 0, joining = 1, ready = 2, failed = 3, closed = 4 };

const char* session_state_name(SessionState state);

struct SessionConfig {
    uint32_t rank = 0;
    uint32_t world = kTpWorld;
    std::string session_uuid;                  // fresh, canonical, identical on all three ranks
    Credential credential;                     // from Credential::from_env, never a source literal
    std::string approved_nccl_version;         // what this build was validated against
    std::string approved_library_provenance;   // which library build and patch set is approved
    PhaseBudgets budgets{};                    // this rank's own ceilings; the smaller of ours and rank 0's wins
    uint32_t notify_ms = 250u;                 // how long an ABORT may try to reach the others
};

enum class CollectiveKind : uint32_t { all_reduce_sum = 1, all_gather_padded = 2 };

struct CollectiveDescriptor {
    Operation operation;
    CollectiveKind kind = CollectiveKind::all_reduce_sum;
    ScalarType type = ScalarType::fp32;
    uint64_t elements = 0;  // reduction extent, or one padded gather slot
    std::vector<uint64_t> valid_counts;  // empty for reduction, world entries for gather
};

Failure canonical_collective(const CollectiveDescriptor& descriptor, std::string& out);

// A line the session wants recorded.  Every string reaching this already went through `Session::redact()`.
using LogFn = std::function<void(std::string_view)>;

struct CollectiveScheduleEntry {
    CollectiveDescriptor descriptor;
    std::string stage;
};

// The rank coordinator.  Single-threaded: one rank, one thread, one conversation at a time.
class Session {
 public:
    Session(SessionConfig config, JoinInfo self, ControlChannel& channel, const Clock& clock, LogFn log = {});

    // Run the handshake to READY.  Blocks at most the configured phase budgets, never longer.  On any failure
    // the session is `failed`, an ABORT was attempted, and the channel is closed.
    Failure start();

    SessionState state() const { return state_; }
    Failure failure() const { return failure_; }
    bool ready() const { return state_ == SessionState::ready; }

    // The verified roster, valid once `ready()`.  Empty before that.
    const Roster& roster() const { return roster_; }

    // A peer's NCCL bootstrap id, for the communicator.  Fails if the rank is not in the verified roster or
    // the session is not ready; the value is never written to a log line.
    Failure bootstrap_id(uint32_t rank, std::string& out) const;

    // The deadline for one operation, from the agreed duration and this host's monotonic clock.  The smaller of
    // this rank's own budget and what rank 0 advertised wins, so one rank cannot stretch a group-wide wait.
    Failure operation_deadline(Deadline& out) const;

    // Legacy local sequence check. Device callers use agree_operation instead, which includes all ranks.
    Failure expect_operation(const Operation& operation);

    // All ranks agree the entire wire shape before enqueue. Complete only after transport.wait succeeds.
    Failure agree_operation(const CollectiveDescriptor& descriptor, Deadline at);
    Failure complete_operation(const Operation& operation);
    // Opt-in: agree exact descriptors AND stage labels once, before the command's compute.
    // Bounded to 16384 entries/one MiB; chunked agreements share one deadline. Only one
    // schedule may be active. Local admission still checks every field and sequence before enqueue.
    Failure agree_operation_schedule(std::span<const CollectiveScheduleEntry>, std::string_view label, Deadline at);
    Failure agree_preagreed_operation(const CollectiveDescriptor&, std::string_view stage, Deadline at);
    // Retire only after every entry completed. Does not rewind the collective sequence.
    Failure finish_operation_schedule();

    // Agree exact label and payload bytes (including embedded NULs) before compute or commit. The complete
    // envelope must fit kControlMaxPayload. Independent of collective sequence; no collective may be in flight.
    // All ranks acknowledge the same metadata before this returns. Any disagreement ends the session.
    Failure agree_control_payload(std::string_view label, std::string_view payload, Deadline at);

    // Rank 0 supplies the bytes; followers supply an empty string and receive them only after every rank
    // acknowledges the same bounded binary envelope and rank 0 commits it. Shares the control agreement
    // sequence, requires READY and no collective in flight. Any failure ends the session; output is unchanged.
    Failure broadcast_control_payload(std::string_view label, std::string& payload, Deadline at);

    // End the session and try to tell the others.  Idempotent; the first reason wins.
    void abort(std::string_view reason);

    // Release the channel.  Idempotent, safe after `abort()`, safe in a destructor.
    Failure close();

    // Replace the credential and every known bootstrap id in `line` with a marker.  Applied to every log line.
    std::string redact(std::string_view line) const;

    // What the session logged, for tests and for a rank whose log is the only evidence after a failure.
    const std::vector<std::string>& log_lines() const { return log_lines_; }

 private:
    Failure start_coordinator();
    Failure start_follower();
    // Everything the join may refuse before it spends a wire byte.
    Failure self_check() const;
    // Rank 0: take one join, or say why this session cannot form.
    Failure collect_one_join(Deadline at);
    uint32_t joined_count() const;
    // Rank 0: check the three joins agree, send the roster, take the three acks, send READY.
    Failure agree_roster();
    // A rank other than 0: re-check a roster it was told about instead of trusting rank 0's word.
    Failure verify_roster(const Roster& roster) const;
    // The first disagreement between two joins, naming the field that differs.
    static Failure compare_joins(const JoinInfo& a, const JoinInfo& b, uint32_t b_rank);
    // A peer's ABORT, turned into this rank's reason.
    Failure peer_abort(const ReceivedFrame& got);
    // Record a failure, tell the peers that can still hear it, release the channel.
    Failure fail(Failure f, bool notify);
    void log(std::string_view line);
    Deadline deadline_after(uint32_t ms) const;
    void notify_peers(const Failure& f);

    SessionConfig config_;
    JoinInfo self_;
    ControlChannel& channel_;
    const Clock& clock_;
    LogFn log_fn_;
    std::vector<std::string> log_lines_;

    SessionState state_ = SessionState::fresh;
    Failure failure_{};
    std::vector<JoinInfo> joins_{};            // rank 0: the joins collected, indexed by rank
    Roster roster_{};
    uint64_t next_sequence_ = 1;               // rank 0's ordering check on the model's own sequence numbers
    uint64_t next_control_sequence_ = 1;
    Operation last_operation_{};
    bool operation_in_flight_ = false;
    bool schedule_active_ = false;
    uint64_t schedule_generation_ = 0;
    size_t schedule_cursor_ = 0;
    std::vector<std::string> schedule_signatures_;
    bool notified_ = false;
};

}  // namespace strata::distributed
