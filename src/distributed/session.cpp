// src/distributed/session.cpp - the join: what a rank says about itself, what a frame has to decode to, and the
// one rule that makes three ranks a session instead of three processes that happen to be running.
//
// The order of operations is not negotiable, and neither is the wording of a refusal:
//   * A rank checks itself before it spends a wire byte.  A rank whose own library is not the approved build has
//     no business asking two good ranks to wait for it.
//   * Rank 0 collects joins, agrees them, and only then sends the roster.  Acks come back before READY goes out,
//     so READY means "three ranks have each seen the same roster and said so", not "rank 0 finished its loop".
//   * Every other rank re-derives the group from the roster it was sent and compares.  Trusting rank 0's
//     conclusion would let a confused coordinator put a rank into a collective its peers never entered.
//   * A frame is decoded exactly or not at all: no trailing bytes, no defaults for a field that is missing.
//
// Nothing here waits without a deadline, and a deadline that passes is a dead session: a peer that is gone
// cannot acknowledge an ABORT, so each rank ends itself rather than holding two good ranks in the dark.
#include "strata/distributed/session.hpp"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <string_view>
#include <utility>

namespace strata::distributed {
namespace {

constexpr size_t kMaxLogLines = 256;

bool is_hex(char c) {
    return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
}

void put_u32(std::string& out, uint32_t v) {
    out.push_back(static_cast<char>((v >> 24) & 0xff));
    out.push_back(static_cast<char>((v >> 16) & 0xff));
    out.push_back(static_cast<char>((v >> 8) & 0xff));
    out.push_back(static_cast<char>(v & 0xff));
}

uint32_t get_u32(const char* p) {
    const auto b = reinterpret_cast<const unsigned char*>(p);
    return (static_cast<uint32_t>(b[0]) << 24) | (static_cast<uint32_t>(b[1]) << 16) |
           (static_cast<uint32_t>(b[2]) << 8) | b[3];
}

Failure put_field(std::string& out, std::string_view name, std::string_view value) {
    if (value.size() > kControlMaxField) {
        return Failure::fail(FailureCode::oversized_frame,
                            "field '" + std::string(name) + "' is too long for the control protocol");
    }
    put_u32(out, static_cast<uint32_t>(value.size()));
    out.append(value);
    return ok_failure();
}

// Reads a payload as exactly the fields its frame kind promises.  A field that does not fit is a truncated
// frame; bytes left over are a payload this build does not understand.  Both end the session: a half-read join
// is a rank whose identity is a guess.
class FieldReader {
 public:
    explicit FieldReader(std::string_view payload) : p_(payload) {}

    Failure u32(uint32_t& out) {
        if (p_.size() - off_ < 4) return short_frame();
        out = get_u32(p_.data() + off_);
        off_ += 4;
        return ok_failure();
    }

    Failure field(std::string& out, std::string_view name) {
        uint32_t n = 0;
        if (auto f = u32(n)) return f;
        if (n > kControlMaxField) {
            return Failure::fail(FailureCode::oversized_frame,
                                 "field '" + std::string(name) + "' claims more bytes than the cap allows");
        }
        if (p_.size() - off_ < n) return short_frame();
        out.assign(p_.data() + off_, n);
        off_ += n;
        return ok_failure();
    }

    Failure finish() const {
        if (off_ != p_.size()) {
            return Failure::fail(FailureCode::bad_frame,
                                 "the payload carries bytes this control protocol does not define");
        }
        return ok_failure();
    }

 private:
    Failure short_frame() const {
        return Failure::fail(FailureCode::truncated_frame, "the payload ended in the middle of a field");
    }

    std::string_view p_;
    size_t off_ = 0;
};

Failure require_non_empty(std::string_view value, std::string_view name, FailureCode code) {
    if (value.empty()) return Failure::fail(code, "'" + std::string(name) + "' is empty");
    return ok_failure();
}

}  // namespace

bool is_canonical_uuid(std::string_view uuid) {
    if (uuid.size() != 36) return false;
    for (size_t i = 0; i < uuid.size(); ++i) {
        const bool dash = i == 8 || i == 13 || i == 18 || i == 23;
        const char c = uuid[i];
        if (dash) {
            if (c != '-') return false;
        } else if (!is_hex(c)) {
            return false;
        }
    }
    return true;
}

std::string canonical_join(const JoinInfo& info) {
    std::string out;
    out.reserve(256);
    out += "uuid=" + info.session_uuid + "\n";
    out += "rank=" + std::to_string(info.rank) + "\n";
    out += "world=" + std::to_string(info.world) + "\n";
    out += "protocol=" + std::to_string(info.protocol_version) + "\n";
    out += "model=" + info.model_hash + "\n";
    out += "config=" + info.config_fingerprint + "\n";
    out += "backend=" + info.backend_identity + "\n";
    out += "nccl=" + info.nccl_version_expected + "\n";
    out += "provenance=" + info.library_provenance + "\n";
    // Bootstrap bytes are part of roster agreement even though each rank may supply a distinct id.
    out += "bootstrap=" + info.nccl_bootstrap_id + "\n";
    return out;
}

std::string roster_digest(const std::vector<JoinInfo>& ranks) {
    std::string out;
    for (size_t i = 0; i < ranks.size(); ++i) {
        out += "[" + std::to_string(i) + "]";  // the slot is part of the identity, so a reordered roster reads
        out += canonical_join(ranks[i]);       // differently and never as the same group in another order
    }
    return out;
}

Failure encode_join(const JoinInfo& info, std::string& out) {
    if (info.world != kTpWorld) {
        return Failure::fail(FailureCode::rank_range,
                            "this rank claims a world of " + std::to_string(info.world) +
                                "; the transport speaks to three");
    }
    out.clear();
    if (auto f = put_field(out, "session_uuid", info.session_uuid)) return f;
    put_u32(out, info.rank);
    put_u32(out, info.world);
    put_u32(out, info.protocol_version);
    if (auto f = put_field(out, "model_hash", info.model_hash)) return f;
    if (auto f = put_field(out, "config_fingerprint", info.config_fingerprint)) return f;
    if (auto f = put_field(out, "backend_identity", info.backend_identity)) return f;
    if (auto f = put_field(out, "nccl_version_expected", info.nccl_version_expected)) return f;
    if (auto f = put_field(out, "library_provenance", info.library_provenance)) return f;
    if (auto f = put_field(out, "nccl_bootstrap_id", info.nccl_bootstrap_id)) return f;
    return ok_failure();
}

Failure decode_join(std::string_view payload, JoinInfo& out) {
    FieldReader r(payload);
    JoinInfo info;
    if (auto f = r.field(info.session_uuid, "session_uuid")) return f;
    if (auto f = r.u32(info.rank)) return f;
    if (auto f = r.u32(info.world)) return f;
    if (auto f = r.u32(info.protocol_version)) return f;
    if (auto f = r.field(info.model_hash, "model_hash")) return f;
    if (auto f = r.field(info.config_fingerprint, "config_fingerprint")) return f;
    if (auto f = r.field(info.backend_identity, "backend_identity")) return f;
    if (auto f = r.field(info.nccl_version_expected, "nccl_version_expected")) return f;
    if (auto f = r.field(info.library_provenance, "library_provenance")) return f;
    if (auto f = r.field(info.nccl_bootstrap_id, "nccl_bootstrap_id")) return f;
    if (auto f = r.finish()) return f;

    // Structural rules: these are protocol, not policy, so they belong to the decoder and every reader of a
    // join gets them, including the socket layer that reads a join to learn which rank sent it.
    if (!is_canonical_uuid(info.session_uuid)) {
        return Failure::fail(FailureCode::uuid_mismatch, "a join's session id is not a canonical UUID");
    }
    if (info.world != kTpWorld) {
        return Failure::fail(FailureCode::rank_range,
                            "a join claims a world of " + std::to_string(info.world) +
                                "; this transport is TP3 and does not negotiate a different size");
    }
    if (info.rank >= info.world) {
        return Failure::fail(FailureCode::rank_range,
                            "a join claims rank " + std::to_string(info.rank) + " of " +
                                std::to_string(info.world));
    }
    if (info.protocol_version != kControlProtocolVersion) {
        return Failure::fail(FailureCode::protocol_version,
                            "a join speaks control protocol " + std::to_string(info.protocol_version) +
                                ", this rank is " + std::to_string(kControlProtocolVersion));
    }
    for (auto [name, value] : {std::pair<std::string_view, std::string_view>{"model_hash", info.model_hash},
                              {"config_fingerprint", info.config_fingerprint},
                              {"backend_identity", info.backend_identity},
                              {"nccl_version_expected", info.nccl_version_expected},
                              {"library_provenance", info.library_provenance},
                              {"nccl_bootstrap_id", info.nccl_bootstrap_id}}) {
        if (auto f = require_non_empty(value, name, FailureCode::bad_frame)) return f;
    }
    out = std::move(info);
    return ok_failure();
}

Failure encode_budgets(const PhaseBudgets& budgets, std::string& out) {
    const uint32_t values[4] = {budgets.connect_ms, budgets.join_ms, budgets.initialize_ms,
                               budgets.operation_ms};
    for (const uint32_t v : values) {
        if (v == 0 || v > kMaxPhaseMs) {
            return Failure::fail(FailureCode::invalid_config,
                                 "a phase budget of " + std::to_string(v) + " ms is neither bounded nor usable");
        }
    }
    out.clear();
    for (const uint32_t v : values) put_u32(out, v);
    return ok_failure();
}

Failure decode_budgets(std::string_view payload, PhaseBudgets& out) {
    FieldReader r(payload);
    PhaseBudgets b;
    uint32_t* names[4] = {&b.connect_ms, &b.join_ms, &b.initialize_ms, &b.operation_ms};
    const char* label[4] = {"connect_ms", "join_ms", "initialize_ms", "operation_ms"};
    for (int i = 0; i < 4; ++i) {
        if (auto f = r.u32(*names[i])) return f;
        if (*names[i] == 0 || *names[i] > kMaxPhaseMs) {
            return Failure::fail(FailureCode::invalid_config,
                                 std::string("rank 0 advertised ") + label[i] + " = " +
                                     std::to_string(*names[i]) + " ms, which is not a bounded phase");
        }
    }
    if (auto f = r.finish()) return f;
    out = b;
    return ok_failure();
}

Failure encode_roster(const Roster& roster, std::string& out) {
    std::string budgets;
    if (auto f = encode_budgets(roster.budgets, budgets)) return f;
    out.clear();
    out.append(budgets);
    if (auto f = put_field(out, "digest", roster.digest)) return f;
    if (roster.ranks.size() != kTpWorld) {
        return Failure::fail(FailureCode::bad_frame, "a roster is offered for exactly three ranks or not at all");
    }
    put_u32(out, static_cast<uint32_t>(roster.ranks.size()));
    for (const auto& join : roster.ranks) {
        std::string payload;
        if (auto f = encode_join(join, payload)) return f;
        if (auto f = put_field(out, "join", payload)) return f;
    }
    return ok_failure();
}

Failure decode_roster(std::string_view payload, Roster& out) {
    Roster roster;
    if (auto f = decode_budgets(payload.substr(0, 16), roster.budgets)) return f;
    // The budgets are the first 16 bytes; read the rest with the same cursor.
    FieldReader rest(payload.substr(16));
    if (auto f = rest.field(roster.digest, "digest")) return f;
    uint32_t count = 0;
    if (auto f = rest.u32(count)) return f;
    if (count != kTpWorld) {
        return Failure::fail(FailureCode::bad_frame,
                            "a roster carries " + std::to_string(count) + " ranks; three was agreed");
    }
    for (uint32_t i = 0; i < count; ++i) {
        std::string join_payload;
        if (auto f = rest.field(join_payload, "join")) return f;
        JoinInfo info;
        if (auto f = decode_join(join_payload, info)) return f;
        if (info.rank != i) {
            return Failure::fail(FailureCode::bad_frame, "the roster is not in rank order at slot " +
                                                             std::to_string(i));
        }
        roster.ranks.push_back(std::move(info));
    }
    if (auto f = rest.finish()) return f;
    out = std::move(roster);
    return ok_failure();
}

Failure encode_abort(FailureCode code, std::string_view reason, std::string& out) {
    if (code == FailureCode::none || code > FailureCode::closed) {
        return Failure::fail(FailureCode::bad_frame, "an abort names the failure it carries");
    }
    out.clear();
    put_u32(out, static_cast<uint32_t>(code));
    return put_field(out, "reason", reason);
}

Failure decode_abort(std::string_view payload, FailureCode& code, std::string& reason) {
    FieldReader r(payload);
    uint32_t raw = 0;
    if (auto f = r.u32(raw)) return f;
    if (auto f = r.field(reason, "reason")) return f;
    if (auto f = r.finish()) return f;
    // A code past the last one this build knows came from another protocol build: the number is not
    // trustworthy, so the frame is refused rather than guessed at.
    if (raw == 0 || raw > static_cast<uint32_t>(FailureCode::closed)) {
        return Failure::fail(FailureCode::bad_frame,
                            "an abort carried code " + std::to_string(raw) + ", which this build has no name for");
    }
    code = static_cast<FailureCode>(raw);
    return ok_failure();
}

Failure Credential::from_env(const char* name, Credential& out) {
    if (name == nullptr || *name == '\0') {
        return Failure::fail(FailureCode::invalid_config, "no environment variable was named for the credential");
    }
    const char* raw = std::getenv(name);
    if (raw == nullptr) {
        return Failure::fail(FailureCode::credential,
                            std::string("the session credential is not set in ") + name);
    }
    const std::string secret(raw);
    // An unset variable and an empty one must not read the same way: "empty" is a credential every peer can
    // match by accident, which is the opposite of authentication.
    if (secret.size() < 16) {
        return Failure::fail(FailureCode::credential,
                            std::string("the session credential in ") + name +
                                " is too short to identify a run");
    }
    out = Credential(secret);
    return ok_failure();
}

std::string Credential::redacted() const {
    if (secret_.empty()) return "credential(empty)";
    return "credential(len=" + std::to_string(secret_.size()) + ")";
}

const char* session_state_name(SessionState state) {
    switch (state) {
        case SessionState::fresh: return "fresh";
        case SessionState::joining: return "joining";
        case SessionState::ready: return "ready";
        case SessionState::failed: return "failed";
        case SessionState::closed: return "closed";
    }
    return "unknown";
}

Session::Session(SessionConfig config, JoinInfo self, ControlChannel& channel, const Clock& clock, LogFn log)
    : config_(std::move(config)),
      self_(std::move(self)),
      channel_(channel),
      clock_(clock),
      log_fn_(std::move(log)) {}

Deadline Session::deadline_after(uint32_t ms) const {
    const uint32_t bounded = std::min(ms, kMaxPhaseMs);
    return Deadline{clock_.now() + std::chrono::milliseconds(bounded)};
}

void Session::log(std::string_view line) {
    std::string full = "rank " + std::to_string(config_.rank) + ": " + redact(line);
    if (log_lines_.size() < kMaxLogLines) log_lines_.push_back(full);
    if (log_fn_) {
        log_fn_(full);
    } else {
        std::fprintf(stderr, "strata-tp3 control %s\n", full.c_str());
    }
}

std::string Session::redact(std::string_view line) const {
    std::string out(line);
    auto mask = [&out](const std::string& secret) {
        if (secret.empty()) return;
        size_t pos = 0;
        while ((pos = out.find(secret, pos)) != std::string::npos) {
            out.replace(pos, secret.size(), "[redacted]");
            pos += 10;
        }
    };
    mask(config_.credential.secret());
    mask(self_.nccl_bootstrap_id);
    for (const auto& join : joins_) mask(join.nccl_bootstrap_id);
    for (const auto& join : roster_.ranks) mask(join.nccl_bootstrap_id);
    return out;
}

Failure Session::self_check() const {
    if (config_.world != kTpWorld) {
        return Failure::fail(FailureCode::rank_range,
                            "this session is configured for " + std::to_string(config_.world) +
                                " ranks; the transport is TP3");
    }
    if (config_.rank >= config_.world) {
        return Failure::fail(FailureCode::rank_range,
                            "this rank is " + std::to_string(config_.rank) + " of " +
                                std::to_string(config_.world));
    }
    if (!is_canonical_uuid(config_.session_uuid)) {
        return Failure::fail(FailureCode::uuid_mismatch,
                            "this session's id is not a canonical UUID; a fresh one is required per run");
    }
    if (config_.credential.empty()) {
        return Failure::fail(FailureCode::credential,
                            "no session credential was supplied (Credential::from_env)");
    }
    if (config_.approved_nccl_version.empty() || config_.approved_library_provenance.empty()) {
        return Failure::fail(FailureCode::invalid_config,
                            "this build names no approved NCCL version or library provenance");
    }
    if (self_.session_uuid != config_.session_uuid) {
        return Failure::fail(FailureCode::uuid_mismatch, "this rank's join names another session than its config");
    }
    if (self_.rank != config_.rank || self_.world != config_.world) {
        return Failure::fail(FailureCode::rank_range, "this rank's join disagrees with its own configuration");
    }
    if (self_.protocol_version != kControlProtocolVersion) {
        return Failure::fail(FailureCode::protocol_version, "this rank's join names another control version");
    }
    for (auto [name, value] : {std::pair<std::string_view, std::string_view>{"model_hash", self_.model_hash},
                              {"config_fingerprint", self_.config_fingerprint},
                              {"backend_identity", self_.backend_identity},
                              {"nccl_version_expected", self_.nccl_version_expected},
                              {"library_provenance", self_.library_provenance},
                              {"nccl_bootstrap_id", self_.nccl_bootstrap_id}}) {
        if (auto f = require_non_empty(value, name, FailureCode::invalid_config)) return f;
    }
    // A rank whose own library is not the approved one is a rank that would put three ranks in a collective
    // nobody validated.  Refuse before the wire, so the other two find out from a message, not a timeout.
    if (self_.nccl_version_expected != config_.approved_nccl_version) {
        return Failure::fail(FailureCode::nccl_version_mismatch,
                            "this rank resolved NCCL " + self_.nccl_version_expected +
                                "; this build was approved against " + config_.approved_nccl_version);
    }
    if (self_.library_provenance != config_.approved_library_provenance) {
        return Failure::fail(FailureCode::provenance_mismatch,
                            "this rank's library provenance is '" + self_.library_provenance +
                                "'; the approved build is '" + config_.approved_library_provenance + "'");
    }
    const uint32_t phases[4] = {config_.budgets.connect_ms, config_.budgets.join_ms,
                               config_.budgets.initialize_ms, config_.budgets.operation_ms};
    for (const uint32_t ms : phases) {
        if (ms == 0 || ms > kMaxPhaseMs) {
            return Failure::fail(FailureCode::invalid_config,
                                 "a phase budget of " + std::to_string(ms) + " ms is not a bounded phase");
        }
    }
    if (config_.notify_ms == 0 || config_.notify_ms > kMaxPhaseMs) {
        return Failure::fail(FailureCode::invalid_config, "the ABORT notice needs a bounded budget of its own");
    }
    return ok_failure();
}

Failure Session::start() {
    if (state_ == SessionState::closed) {
        return Failure::fail(FailureCode::closed, "this session is closed");
    }
    if (state_ != SessionState::fresh) {
        return Failure::fail(FailureCode::stale_join,
                             "a session forms once: " + std::string(session_state_name(state_)));
    }
    if (auto f = self_check()) return fail(f, false);

    state_ = SessionState::joining;
    log("forming a TP3 session as rank " + std::to_string(config_.rank) + ", credential " +
        config_.credential.redacted());

    if (auto f = channel_.open(deadline_after(config_.budgets.connect_ms))) {
        return fail(Failure::fail(f.code, "the control channel did not open: " + f.detail), false);
    }

    if (auto f = (config_.rank == 0 ? start_coordinator() : start_follower())) return fail(f, true);
    return ok_failure();
}

uint32_t Session::joined_count() const {
    uint32_t n = 0;
    for (const auto& join : joins_) {
        if (!join.session_uuid.empty()) ++n;
    }
    return n;
}

Failure Session::compare_joins(const JoinInfo& a, const JoinInfo& b, uint32_t b_rank) {
    const std::string who = "rank " + std::to_string(b_rank);
    if (a.session_uuid != b.session_uuid) {
        return Failure::fail(FailureCode::uuid_mismatch, who + " is in session " + b.session_uuid +
                                                            ", not this run");
    }
    if (a.model_hash != b.model_hash) {
        return Failure::fail(FailureCode::model_mismatch,
                            who + " loaded a different model: " + b.model_hash + " against " + a.model_hash);
    }
    if (a.config_fingerprint != b.config_fingerprint) {
        return Failure::fail(FailureCode::config_mismatch,
                            who + " runs a different configuration: " + b.config_fingerprint + " against " +
                                a.config_fingerprint);
    }
    if (a.backend_identity != b.backend_identity) {
        return Failure::fail(FailureCode::backend_mismatch,
                            who + " speaks backend '" + b.backend_identity + "', this group uses '" +
                                a.backend_identity + "'");
    }
    if (a.nccl_version_expected != b.nccl_version_expected) {
        return Failure::fail(FailureCode::nccl_version_mismatch,
                            who + " expects NCCL " + b.nccl_version_expected + ", this group has " +
                                a.nccl_version_expected);
    }
    if (a.library_provenance != b.library_provenance) {
        return Failure::fail(FailureCode::provenance_mismatch,
                            who + " resolved library '" + b.library_provenance + "', this group has '" +
                                a.library_provenance + "'");
    }
    return ok_failure();
}

Failure Session::peer_abort(const ReceivedFrame& got) {
    FailureCode code = FailureCode::aborted;
    std::string reason;
    if (auto f = decode_abort(got.frame.payload, code, reason)) return f;
    return Failure::fail(code, "rank " + std::to_string(got.from_rank) + " ended the session: " + reason);
}

Failure Session::collect_one_join(Deadline at) {
    ReceivedFrame got;
    if (auto f = channel_.recv(got, at)) {
        if (f.code == FailureCode::deadline_expired) {
            std::string missing;
            for (uint32_t i = 0; i < config_.world; ++i) {
                if (joins_[i].session_uuid.empty()) {
                    if (!missing.empty()) missing += ", ";
                    missing += std::to_string(i);
                }
            }
            return Failure::fail(FailureCode::missing_rank,
                                 "the join deadline passed and rank " + missing + " never joined");
        }
        return f;
    }
    if (got.frame.kind == FrameKind::abort) return peer_abort(got);
    if (got.frame.kind != FrameKind::join) {
        return Failure::fail(FailureCode::bad_frame,
                             std::string("a join was expected, got ") + frame_kind_name(got.frame.kind) +
                                 " from rank " + std::to_string(got.from_rank));
    }

    JoinInfo info;
    if (auto f = decode_join(got.frame.payload, info)) return f;
    if (info.rank != got.from_rank) {
        return Failure::fail(FailureCode::bad_frame,
                            "a join names rank " + std::to_string(info.rank) + " but arrived on rank " +
                                std::to_string(got.from_rank) + "'s link");
    }
    if (info.session_uuid != config_.session_uuid) {
        return Failure::fail(FailureCode::uuid_mismatch,
                            "rank " + std::to_string(info.rank) + " is in session " + info.session_uuid +
                                ", not this run's " + config_.session_uuid);
    }
    if (info.rank >= config_.world) {
        return Failure::fail(FailureCode::rank_range,
                             "a join claims rank " + std::to_string(info.rank) + " of " +
                                 std::to_string(config_.world));
    }
    if (!joins_[info.rank].session_uuid.empty()) {
        return Failure::fail(FailureCode::duplicate_rank,
                             "rank " + std::to_string(info.rank) + " joined twice; the first claim stands");
    }
    if (auto f = compare_joins(self_, info, info.rank)) return f;
    if (info.nccl_version_expected != config_.approved_nccl_version) {
        return Failure::fail(FailureCode::nccl_version_mismatch,
                             "rank " + std::to_string(info.rank) + " resolved NCCL " +
                                 info.nccl_version_expected + "; this build was approved against " +
                                 config_.approved_nccl_version);
    }
    if (info.library_provenance != config_.approved_library_provenance) {
        return Failure::fail(FailureCode::provenance_mismatch,
                             "rank " + std::to_string(info.rank) + " resolved library '" +
                                 info.library_provenance + "'; the approved build is '" +
                                 config_.approved_library_provenance + "'");
    }

    const uint32_t accepted = info.rank;
    joins_[info.rank] = std::move(info);
    log("join accepted from rank " + std::to_string(accepted));
    return ok_failure();
}

Failure Session::agree_roster() {
    roster_.ranks = joins_;
    roster_.digest = roster_digest(roster_.ranks);
    roster_.budgets = config_.budgets;  // rank 0's ceilings are the group's; every rank keeps the smaller of
                                        // its own and these

    std::string payload;
    if (auto f = encode_roster(roster_, payload)) return f;
    const Deadline roster_at = deadline_after(config_.budgets.join_ms);
    for (uint32_t i = 1; i < config_.world; ++i) {
        if (auto f = channel_.send(i, FrameKind::roster, payload, roster_at)) return f;
    }
    log("roster offered to the group");

    std::vector<bool> acked(config_.world, false);
    acked[0] = true;  // rank 0's own ack is the roster it built and just checked
    const Deadline ack_at = deadline_after(config_.budgets.initialize_ms);
    for (;;) {
        bool waiting = false;
        for (size_t i = 1; i < acked.size(); ++i) {
            if (!acked[i]) waiting = true;
        }
        if (!waiting) break;

        ReceivedFrame got;
        if (auto f = channel_.recv(got, ack_at)) {
            if (f.code == FailureCode::deadline_expired) {
                std::string missing;
                for (uint32_t i = 1; i < config_.world; ++i) {
                    if (!acked[i]) {
                        if (!missing.empty()) missing += ", ";
                        missing += std::to_string(i);
                    }
                }
                return Failure::fail(FailureCode::deadline_expired,
                                     "rank " + missing + " never acknowledged the roster");
            }
            return f;
        }
        if (got.frame.kind == FrameKind::abort) return peer_abort(got);
        if (got.frame.kind == FrameKind::join) {
            return Failure::fail(FailureCode::stale_join,
                                 "a join arrived after the roster went out; that rank is from another attempt");
        }
        if (got.frame.kind != FrameKind::roster_ack) {
            return Failure::fail(FailureCode::bad_frame,
                                 std::string("a roster ack was expected, got ") +
                                     frame_kind_name(got.frame.kind));
        }

        FieldReader r(got.frame.payload);
        std::string uuid;
        uint32_t rank = 0;
        std::string digest;
        if (auto f = r.field(uuid, "session_uuid")) return f;
        if (auto f = r.u32(rank)) return f;
        if (auto f = r.field(digest, "digest")) return f;
        if (auto f = r.finish()) return f;

        if (uuid != config_.session_uuid) {
            return Failure::fail(FailureCode::uuid_mismatch, "a roster ack names another session");
        }
        if (rank >= config_.world) {
            return Failure::fail(FailureCode::rank_range, "a roster ack claims rank " + std::to_string(rank));
        }
        if (rank != got.from_rank) {
            return Failure::fail(FailureCode::bad_frame, "a roster ack names another rank's control link");
        }
        if (rank == 0 || acked[rank]) {
            return Failure::fail(FailureCode::bad_frame,
                                 "rank " + std::to_string(rank) + " acknowledged the roster twice");
        }
        if (joins_[rank].session_uuid.empty()) {
            return Failure::fail(FailureCode::missing_rank,
                                 "rank " + std::to_string(rank) + " acknowledged a roster it never joined");
        }
        if (digest != roster_.digest) {
            return Failure::fail(FailureCode::roster_mismatch,
                                 "rank " + std::to_string(rank) +
                                     " acknowledged a different roster than the one offered");
        }
        acked[rank] = true;
        log("roster acknowledged by rank " + std::to_string(rank));
    }

    std::string ready_payload;
    if (auto f = put_field(ready_payload, "session_uuid", config_.session_uuid)) return f;
    put_u32(ready_payload, config_.rank);
    if (auto f = put_field(ready_payload, "digest", roster_.digest)) return f;
    const Deadline ready_at = deadline_after(config_.budgets.initialize_ms);
    for (uint32_t i = 1; i < config_.world; ++i) {
        if (auto f = channel_.send(i, FrameKind::ready, ready_payload, ready_at)) return f;
    }
    state_ = SessionState::ready;
    log("READY: three ranks hold the same roster");
    return ok_failure();
}

Failure Session::start_coordinator() {
    joins_.assign(config_.world, JoinInfo{});
    joins_[0] = self_;  // rank 0's own join is local and was checked by self_check; it never needs the wire

    const Deadline join_at = deadline_after(config_.budgets.join_ms);
    while (joined_count() < config_.world) {
        if (auto f = collect_one_join(join_at)) return f;
    }
    log("all three joins accepted");
    return agree_roster();
}

Failure Session::verify_roster(const Roster& roster) const {
    if (roster.ranks.size() != config_.world) {
        return Failure::fail(FailureCode::bad_frame,
                             "the roster carries " + std::to_string(roster.ranks.size()) +
                                 " ranks; three was agreed");
    }
    for (size_t i = 0; i < roster.ranks.size(); ++i) {
        if (roster.ranks[i].rank != i) {
            return Failure::fail(FailureCode::bad_frame,
                                 "the roster is not in rank order at slot " + std::to_string(i));
        }
        if (roster.ranks[i].session_uuid != config_.session_uuid) {
            return Failure::fail(FailureCode::uuid_mismatch,
                                 "slot " + std::to_string(i) + " of the roster is in another session");
        }
    }
    // Re-check the agreement rather than trusting that rank 0 checked it.
    for (uint32_t i = 1; i < config_.world; ++i) {
        if (auto f = compare_joins(roster.ranks[0], roster.ranks[i], i)) return f;
        if (roster.ranks[i].nccl_version_expected != config_.approved_nccl_version) {
            return Failure::fail(FailureCode::nccl_version_mismatch,
                                 "rank " + std::to_string(i) + " expects NCCL " +
                                     roster.ranks[i].nccl_version_expected + "; this build was approved against " +
                                     config_.approved_nccl_version);
        }
        if (roster.ranks[i].library_provenance != config_.approved_library_provenance) {
            return Failure::fail(FailureCode::provenance_mismatch,
                                 "rank " + std::to_string(i) + " resolved library '" +
                                     roster.ranks[i].library_provenance + "'; the approved build is '" +
                                     config_.approved_library_provenance + "'");
        }
    }
    // My own entry has to be what I said it was: a coordinator that rewrote it is a coordinator to distrust.
    if (canonical_join(roster.ranks[config_.rank]) != canonical_join(self_)) {
        return Failure::fail(FailureCode::roster_mismatch,
                             "the roster does not carry this rank's own join");
    }
    if (roster_digest(roster.ranks) != roster.digest) {
        return Failure::fail(FailureCode::roster_mismatch,
                             "the roster's digest does not describe the roster it came with");
    }
    return ok_failure();
}

Failure Session::start_follower() {
    const Deadline join_at = deadline_after(config_.budgets.join_ms);

    std::string payload;
    if (auto f = encode_join(self_, payload)) return f;
    if (auto f = channel_.send(0, FrameKind::join, payload, join_at)) return f;
    log("join sent to the coordinator");

    ReceivedFrame got;
    if (auto f = channel_.recv(got, join_at)) {
        if (f.code == FailureCode::deadline_expired) {
            return Failure::fail(FailureCode::deadline_expired,
                                 "rank 0 never answered this rank's join");
        }
        return f;
    }
    if (got.frame.kind == FrameKind::abort) return peer_abort(got);
    if (got.frame.kind == FrameKind::join) {
        return Failure::fail(FailureCode::stale_join, "another rank's join reached a rank that does not route");
    }
    if (got.frame.kind != FrameKind::roster) {
        return Failure::fail(FailureCode::bad_frame,
                             std::string("a roster was expected, got ") + frame_kind_name(got.frame.kind));
    }

    Roster roster;
    if (auto f = decode_roster(got.frame.payload, roster)) return f;
    if (auto f = verify_roster(roster)) return f;
    roster_ = roster;
    log("roster verified: " + roster_.digest);

    std::string ack;
    if (auto f = put_field(ack, "session_uuid", config_.session_uuid)) return f;
    put_u32(ack, config_.rank);
    if (auto f = put_field(ack, "digest", roster_.digest)) return f;
    const Deadline ack_at = deadline_after(config_.budgets.initialize_ms);
    if (auto f = channel_.send(0, FrameKind::roster_ack, ack, ack_at)) return f;
    log("roster acknowledged");

    const Deadline ready_at = deadline_after(config_.budgets.initialize_ms);
    ReceivedFrame ready;
    if (auto f = channel_.recv(ready, ready_at)) {
        if (f.code == FailureCode::deadline_expired) {
            return Failure::fail(FailureCode::deadline_expired,
                                 "rank 0 never said READY after this rank acknowledged the roster");
        }
        return f;
    }
    if (ready.frame.kind == FrameKind::abort) return peer_abort(ready);
    if (ready.frame.kind == FrameKind::join) {
        return Failure::fail(FailureCode::stale_join, "a join arrived after this rank acknowledged the roster");
    }
    if (ready.frame.kind != FrameKind::ready) {
        return Failure::fail(FailureCode::bad_frame,
                             std::string("READY was expected, got ") + frame_kind_name(ready.frame.kind));
    }

    FieldReader r(ready.frame.payload);
    std::string uuid;
    uint32_t rank = 0;
    std::string digest;
    if (auto f = r.field(uuid, "session_uuid")) return f;
    if (auto f = r.u32(rank)) return f;
    if (auto f = r.field(digest, "digest")) return f;
    if (auto f = r.finish()) return f;
    if (uuid != config_.session_uuid || rank != 0 || digest != roster_.digest) {
        return Failure::fail(FailureCode::roster_mismatch,
                             "READY names a roster this rank did not acknowledge");
    }
    state_ = SessionState::ready;
    log("READY: three ranks hold the same roster");
    return ok_failure();
}

Failure Session::fail(Failure f, bool notify) {
    failure_ = f;
    state_ = SessionState::failed;
    log(std::string("session failed: ") + failure_code_name(f.code) + ": " + f.detail);
    if (notify) notify_peers(f);
    channel_.close();
    return failure_;
}

void Session::notify_peers(const Failure& f) {
    if (notified_) return;
    notified_ = true;
    std::string payload;
    if (auto enc = encode_abort(f.code, f.detail, payload)) return;  // no room to say it: say nothing
    const Deadline at = deadline_after(config_.notify_ms);
    if (config_.rank == 0) {
        // Rank 0 relays: a rank that died locally has told rank 0, and the third rank must hear it too.
        for (uint32_t i = 1; i < config_.world; ++i) {
            if (auto f2 = channel_.send(i, FrameKind::abort, payload, at)) (void)f2;
        }
    } else {
        if (auto f2 = channel_.send(0, FrameKind::abort, payload, at)) (void)f2;
    }
}

Failure Session::bootstrap_id(uint32_t rank, std::string& out) const {
    if (state_ != SessionState::ready) {
        return Failure::fail(FailureCode::not_ready,
                             std::string("the roster is not verified: the session is ") +
                                 session_state_name(state_));
    }
    if (rank >= roster_.ranks.size()) {
        return Failure::fail(FailureCode::rank_range, "rank " + std::to_string(rank) + " is not in the roster");
    }
    out = roster_.ranks[rank].nccl_bootstrap_id;
    return ok_failure();
}

Failure Session::operation_deadline(Deadline& out) const {
    if (state_ != SessionState::ready) {
        return Failure::fail(FailureCode::not_ready,
                             std::string("no operation deadline before READY: the session is ") +
                                 session_state_name(state_));
    }
    // Durations crossed the wire, never a machine's clock value: the group agrees on how long a step may take,
    // and each rank counts it on its own monotonic clock.  The smaller ceiling wins, so no rank can stretch a
    // wait the others did not agree to.
    const uint32_t ms = std::min(config_.budgets.operation_ms, roster_.budgets.operation_ms);
    out.end = clock_.now() + std::chrono::milliseconds(ms);
    return ok_failure();
}

Failure Session::expect_operation(const Operation& operation) {
    if (state_ != SessionState::ready) {
        return Failure::fail(FailureCode::not_ready, "no collective order before READY");
    }
    if (operation.sequence != next_sequence_) {
        return Failure::fail(FailureCode::bad_operation,
                            "the step issued collective " + std::to_string(operation.sequence) +
                                ", expected " + std::to_string(next_sequence_) + " (last was layer " +
                                std::to_string(last_operation_.layer) + ", step " +
                                std::to_string(last_operation_.step) + ")");
    }
    next_sequence_ = operation.sequence + 1;
    last_operation_ = operation;
    return ok_failure();
}

Failure canonical_collective(const CollectiveDescriptor& d, std::string& out) {
    if (scalar_type_bytes(d.type) == 0) return Failure::fail(FailureCode::bad_dtype, "collective dtype is invalid");
    if (d.elements == 0 || d.elements > UINT64_MAX / (scalar_type_bytes(d.type) * kTpWorld)) {
        return Failure::fail(FailureCode::bad_extent, "collective extent is empty or overflows");
    }
    if (d.kind == CollectiveKind::all_reduce_sum) {
        if (!d.valid_counts.empty()) return Failure::fail(FailureCode::bad_counts, "reduction has gather counts");
    } else if (d.kind == CollectiveKind::all_gather_padded) {
        if (d.valid_counts.size() != kTpWorld ||
            *std::max_element(d.valid_counts.begin(), d.valid_counts.end()) != d.elements) {
            return Failure::fail(FailureCode::bad_counts, "gather counts do not describe the padded slot");
        }
    } else {
        return Failure::fail(FailureCode::bad_operation, "unknown collective kind");
    }
    // Decimal unsigned fields and fixed separators are unambiguous; no rank-local pointer/clock crosses wire.
    out = std::to_string(d.operation.sequence) + ":" + std::to_string(d.operation.layer) + ":" +
          std::to_string(d.operation.step) + ":" + std::to_string(static_cast<uint32_t>(d.kind)) + ":" +
          std::to_string(static_cast<uint32_t>(d.type)) + ":" + std::to_string(d.elements);
    for (const uint64_t count : d.valid_counts) out += ":" + std::to_string(count);
    return ok_failure();
}

Failure Session::agree_operation(const CollectiveDescriptor& descriptor, Deadline at) {
    if (state_ != SessionState::ready) return Failure::fail(FailureCode::not_ready, "no collective agreement before READY");
    if (operation_in_flight_) return fail(Failure::fail(FailureCode::in_flight, "a collective is already in flight"), true);
    if (descriptor.operation.sequence != next_sequence_) {
        return fail(Failure::fail(FailureCode::bad_operation, "collective sequence differs from the session"), true);
    }
    std::string signature;
    if (auto f = canonical_collective(descriptor, signature)) return fail(f, true);
    std::string payload;
    if (auto f = put_field(payload, "session_uuid", config_.session_uuid)) return fail(f, true);
    if (auto f = put_field(payload, "signature", signature)) return fail(f, true);
    auto receive = [&](FrameKind kind, std::vector<bool>& seen) -> Failure {
        ReceivedFrame got;
        if (auto f = channel_.recv(got, at)) return f;
        if (got.frame.kind == FrameKind::abort) return peer_abort(got);
        if (got.frame.kind != kind || got.from_rank >= config_.world || seen[got.from_rank]) {
            return Failure::fail(FailureCode::bad_operation, "unexpected or duplicate collective agreement frame");
        }
        FieldReader reader(got.frame.payload);
        std::string uuid, peer_signature;
        if (auto f = reader.field(uuid, "session_uuid")) return f;
        if (auto f = reader.field(peer_signature, "signature")) return f;
        if (auto f = reader.finish()) return f;
        if (uuid != config_.session_uuid || peer_signature != signature) {
            return Failure::fail(FailureCode::bad_operation, "ranks disagree on collective sequence, kind, dtype or layout");
        }
        seen[got.from_rank] = true;
        return ok_failure();
    };
    if (config_.rank == 0) {
        std::vector<bool> seen(config_.world, false);
        seen[0] = true;
        for (uint32_t i = 1; i < config_.world; ++i) {
            if (auto f = receive(FrameKind::operation, seen)) return fail(f, true);
        }
        for (uint32_t i = 1; i < config_.world; ++i) {
            if (auto f = channel_.send(i, FrameKind::operation_ack, payload, at)) return fail(f, true);
        }
    } else {
        if (auto f = channel_.send(0, FrameKind::operation, payload, at)) return fail(f, true);
        std::vector<bool> seen(config_.world, true);
        seen[0] = false;
        if (auto f = receive(FrameKind::operation_ack, seen)) return fail(f, true);
    }
    if (deadline_passed(clock_, at)) return fail(Failure::fail(FailureCode::deadline_expired, "collective agreement deadline passed"), true);
    operation_in_flight_ = true;
    last_operation_ = descriptor.operation;
    return ok_failure();
}

Failure Session::complete_operation(const Operation& operation) {
    if (state_ != SessionState::ready) return Failure::fail(FailureCode::not_ready, "the session is not ready");
    if (!operation_in_flight_ || operation.sequence != last_operation_.sequence ||
        operation.layer != last_operation_.layer || operation.step != last_operation_.step) {
        return fail(Failure::fail(FailureCode::bad_operation, "completion does not name the agreed collective"), true);
    }
    operation_in_flight_ = false;
    ++next_sequence_;
    return ok_failure();
}

void Session::abort(std::string_view reason) {
    if (state_ == SessionState::closed) return;
    if (state_ != SessionState::failed) {
        Failure f = Failure::fail(FailureCode::aborted,
                                  reason.empty() ? std::string("the session was aborted")
                                                 : std::string(reason));
        failure_ = f;
        state_ = SessionState::failed;
        log(std::string("session failed: ") + failure_code_name(f.code) + ": " + f.detail);
        notify_peers(f);
    }
    channel_.close();
}

Failure Session::close() {
    channel_.close();
    if (state_ != SessionState::failed) state_ = SessionState::closed;
    return ok_failure();
}

}  // namespace strata::distributed
