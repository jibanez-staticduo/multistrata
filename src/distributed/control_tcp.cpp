// src/distributed/control_tcp.cpp - the control channel's bytes: the frame codec and the POSIX socket channel.
//
// Framing is fixed-header, big-endian, and checked before anything is allocated: a length field is refused the
// moment the header is read, so a peer that claims a gigabyte costs eight bytes of attention, not a gigabyte of
// memory.  A malformed frame ends the stream rather than resynchronising it, because "where the next frame
// starts" is exactly what two ranks must not guess about.
//
// The channel also owns authentication, so the session's protocol never carries a secret: the first frame on
// every link is `hello` with the session credential, and a link that fails it is closed before the session sees
// a frame from it.  The credential and the join's bootstrap id are never written to a stream here; this file has
// no logging at all, which is the cheapest way to guarantee it.
//
// Every wait inside takes the deadline it was handed and never sleeps past it: connect, send and receive are all
// non-blocking under poll(), so a peer that dies leaves the socket, not the rank.
//
// Sockets are POSIX here (the Spark hosts and the NAS are Linux).  Anywhere else the factory says so plainly.
#include "strata/distributed/session.hpp"

#include <cerrno>
#include <cstring>

namespace strata::distributed {

const char* frame_kind_name(FrameKind kind) {
    switch (kind) {
        case FrameKind::hello: return "hello";
        case FrameKind::join: return "join";
        case FrameKind::roster: return "roster";
        case FrameKind::roster_ack: return "roster_ack";
        case FrameKind::ready: return "ready";
        case FrameKind::abort: return "abort";
        case FrameKind::operation: return "operation";
        case FrameKind::operation_ack: return "operation_ack";
    }
    return "unknown";
}

namespace {

void put_u16(std::string& out, uint16_t v) {
    out.push_back(static_cast<char>((v >> 8) & 0xff));
    out.push_back(static_cast<char>(v & 0xff));
}

void put_u32(std::string& out, uint32_t v) {
    out.push_back(static_cast<char>((v >> 24) & 0xff));
    out.push_back(static_cast<char>((v >> 16) & 0xff));
    out.push_back(static_cast<char>((v >> 8) & 0xff));
    out.push_back(static_cast<char>(v & 0xff));
}

uint16_t get_u16(const char* p) {
    const auto b = reinterpret_cast<const unsigned char*>(p);
    return static_cast<uint16_t>((static_cast<uint16_t>(b[0]) << 8) | b[1]);
}

uint32_t get_u32(const char* p) {
    const auto b = reinterpret_cast<const unsigned char*>(p);
    return (static_cast<uint32_t>(b[0]) << 24) | (static_cast<uint32_t>(b[1]) << 16) |
           (static_cast<uint32_t>(b[2]) << 8) | b[3];
}

}  // namespace

Failure encode_frame(FrameKind kind, std::string_view payload, std::string& out) {
    const uint16_t raw_kind = static_cast<uint16_t>(kind);
    const bool known = raw_kind >= static_cast<uint16_t>(FrameKind::hello) &&
                       raw_kind <= static_cast<uint16_t>(FrameKind::operation_ack);
    if (!known) {
        return Failure::fail(FailureCode::unknown_frame, "cannot encode a frame kind this build does not know");
    }
    if (payload.size() > kControlMaxPayload) {
        return Failure::fail(FailureCode::oversized_frame, "payload is larger than the control channel allows");
    }
    out.clear();
    out.reserve(kFrameHeaderBytes + payload.size());
    out.push_back(static_cast<char>(kFrameMagic0));
    out.push_back(static_cast<char>(kFrameMagic1));
    put_u16(out, static_cast<uint16_t>(kControlProtocolVersion));
    put_u16(out, raw_kind);
    put_u32(out, static_cast<uint32_t>(payload.size()));
    out.append(payload);
    // The header layout above and `kFrameHeaderBytes` are the same rule written twice.  If they drift, every
    // payload is cut by the difference and the first frame a rank reads fails authentication, so check it here
    // where a CPU test sees it rather than in a handshake three ranks deadlock in.
    if (out.size() != payload.size() + kFrameHeaderBytes) {
        return Failure::fail(FailureCode::bad_frame, "the frame encoder and the frame header size disagree");
    }
    return ok_failure();
}

Failure FrameDecoder::feed(const char* data, size_t n, std::vector<DecodedFrame>& out) {
    if (data == nullptr && n != 0) {
        return Failure::fail(FailureCode::bad_frame, "decoder fed a null buffer");
    }
    buffer_.append(data, n);

    for (;;) {
        if (buffer_.size() < kFrameHeaderBytes) return ok_failure();

        const unsigned char* h = reinterpret_cast<const unsigned char*>(buffer_.data());
        if (h[0] != kFrameMagic0 || h[1] != kFrameMagic1) {
            return Failure::fail(FailureCode::bad_frame, "frame magic does not match this control protocol");
        }
        const uint16_t version = get_u16(buffer_.data() + 2);
        if (version != static_cast<uint16_t>(kControlProtocolVersion)) {
            return Failure::fail(FailureCode::protocol_version,
                                 "peer speaks control protocol " + std::to_string(version) + ", this rank is " +
                                     std::to_string(kControlProtocolVersion));
        }
        const uint16_t raw_kind = get_u16(buffer_.data() + 4);
        if (raw_kind < static_cast<uint16_t>(FrameKind::hello) ||
            raw_kind > static_cast<uint16_t>(FrameKind::operation_ack)) {
            return Failure::fail(FailureCode::unknown_frame,
                                 "unknown control frame kind " + std::to_string(raw_kind));
        }
        const uint32_t length = get_u32(buffer_.data() + 6);
        // Refuse the length before reading the body: this is the check that keeps a wrong length field from
        // becoming a huge allocation.
        if (length > kControlMaxPayload) {
            return Failure::fail(FailureCode::oversized_frame,
                                 "peer announced a " + std::to_string(length) + " byte control payload; the cap is " +
                                     std::to_string(kControlMaxPayload));
        }
        if (buffer_.size() < kFrameHeaderBytes + length) return ok_failure();  // need more bytes

        DecodedFrame frame;
        frame.kind = static_cast<FrameKind>(raw_kind);
        frame.payload = buffer_.substr(kFrameHeaderBytes, length);
        buffer_.erase(0, kFrameHeaderBytes + length);
        out.push_back(std::move(frame));
    }
}

}  // namespace strata::distributed

#if defined(__unix__) || defined(__APPLE__)

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <deque>
#include <map>
#include <memory>
#include <vector>
#include <thread>

namespace strata::distributed {
namespace {

// One remaining-time slice, clamped so a bogus deadline cannot turn into an oversized poll() timeout.
int remaining_ms(const Clock& clock, Deadline at) {
    const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(at.end - clock.now()).count();
    if (left <= 0) return 0;
    return static_cast<int>(std::min<long long>(left, 5000LL));
}

// A poll slice ending is not the phase deadline: `remaining_ms` caps one wait at five seconds so a bogus
// deadline cannot become one huge timeout, which means a phase longer than that is several waits.  Only the
// clock may say a deadline has passed, or a 15-second join budget would quietly become five.
bool out_of_time(const Clock& clock, Deadline at) { return deadline_passed(clock, at); }

std::string errno_text() { return std::string(strerror(errno)); }

bool set_nonblocking(int fd, std::string& err) {
    const int flags = fcntl(fd, F_GETFL, 0);
    if (flags < 0) {
        err = "fcntl F_GETFL: " + errno_text();
        return false;
    }
    if (fcntl(fd, F_SETFL, flags | O_NONBLOCK) < 0) {
        err = "fcntl F_SETFL: " + errno_text();
        return false;
    }
    return true;
}

bool parse_ipv4(const std::string& address, sockaddr_in& out, std::string& err) {
    memset(&out, 0, sizeof out);
    out.sin_family = AF_INET;
    if (address.empty() || address == "0.0.0.0") {
        out.sin_addr.s_addr = INADDR_ANY;  // bind any; a peer still has to be reached by a real address
        if (address.empty()) {
            err = "an empty address is not a private address to bind";
            return false;
        }
        return true;
    }
    if (inet_pton(AF_INET, address.c_str(), &out.sin_addr) != 1) {
        err = "control address '" + address + "' is not IPv4 dotted-quad";
        return false;
    }
    return true;
}

// What a launch race looks like, as opposed to an address that will never answer.  Three ranks are started by
// a launcher and rank 0 may come up last: a refused connect at that moment is not a disagreement about the
// session, and refusing it would make the start order decide whether TP3 works.
bool retryable_connect(int error_code) {
    switch (error_code) {
        case ECONNREFUSED:
        case EINTR:
        case EAGAIN:
        case ETIMEDOUT:
            return true;
        // EWOULDBLOCK is EAGAIN on every platform this builds on, so it gets no separate case.
        default:
            return false;
    }
}

// One socket to the coordinator, or the reason there is not one.  Retries while the phase has time, and never
// past its deadline.
bool connect_within_deadline(int& fd, const sockaddr_in& addr, const std::string& where, const Clock& clock,
                            Deadline at, std::string& err) {
    for (;;) {
        if (remaining_ms(clock, at) <= 0) {
            err = "connect " + where + " never completed inside the connect deadline";
            return false;
        }
        const int s = ::socket(AF_INET, SOCK_STREAM, 0);
        if (s < 0) {
            err = "socket: " + errno_text();
            return false;
        }
        if (!set_nonblocking(s, err)) {
            ::close(s);
            return false;
        }
        int failure = 0;
        if (::connect(s, reinterpret_cast<const sockaddr*>(&addr), sizeof addr) != 0) {
            if (errno == EINPROGRESS || errno == EINTR) {
                // Still going: the answer arrives as writability plus SO_ERROR, and only inside the deadline.
                pollfd p{s, POLLOUT, 0};
                const int pr = ::poll(&p, 1, remaining_ms(clock, at));
                if (pr > 0) {
                    int so_error = 0;
                    socklen_t len = sizeof so_error;
                    ::getsockopt(s, SOL_SOCKET, SO_ERROR, &so_error, &len);
                    failure = so_error;
                } else if (pr == 0) {
                    failure = ETIMEDOUT;
                } else {
                    failure = (errno == EINTR) ? EINTR : errno;
                }
            } else {
                failure = errno;
            }
        }
        if (failure == 0) {
            fd = s;  // the socket that connected is the socket handed over
            return true;
        }
        ::close(s);  // a socket whose connect failed cannot be re-connected: the retry opens a new one
        if (!retryable_connect(failure)) {
            err = "connect " + where + ": " + strerror(failure);
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(25));
    }
}

// A link to one peer, or the listener.  `needs_hello` is what makes authentication happen before the protocol:
// no frame from a link is handed to the session until the credential arrived on that link.
struct Link {
    int fd = -1;
    bool listening = false;
    bool needs_hello = false;
    bool has_rank = false;
    uint32_t rank = 0;
    // A dropped link keeps its slot until the end of the poll pass: erasing in the middle would move the
    // links the same pass still has to look at.
    bool dead = false;
    FrameDecoder decoder;
};

class TcpChannel final : public ControlChannel {
 public:
    TcpChannel(TcpChannelConfig config, const Clock& clock) : config_(std::move(config)), clock_(clock) {}

    ~TcpChannel() override { close(); }

    Failure open(Deadline at) override {
        std::string err;
        sockaddr_in addr{};
        if (!parse_ipv4(config_.rank == 0 ? config_.self.address : config_.rank0.address, addr, err)) {
            return Failure::fail(FailureCode::invalid_config, err);
        }
        addr.sin_port = htons(config_.rank == 0 ? config_.self.port : config_.rank0.port);
        if (config_.rank == 0 && config_.self.port == 0) {
            return Failure::fail(FailureCode::invalid_config, "rank 0 needs a port to listen on");
        }
        if (config_.credential.empty()) {
            return Failure::fail(FailureCode::credential, "the session credential was never supplied");
        }

        if (config_.rank == 0) {
            const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
            if (fd < 0) {
                return Failure::fail(FailureCode::listen_failed, "socket: " + errno_text());
            }
            if (!set_nonblocking(fd, err)) {
                ::close(fd);
                return Failure::fail(FailureCode::listen_failed, err);
            }
            const int one = 1;
            ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
            if (::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof addr) != 0) {
                ::close(fd);
                return Failure::fail(FailureCode::listen_failed,
                                     "bind " + config_.self.address + ":" + std::to_string(config_.self.port) +
                                         ": " + errno_text());
            }
            if (::listen(fd, 8) != 0) {
                ::close(fd);
                return Failure::fail(FailureCode::listen_failed, "listen: " + errno_text());
            }
            Link link;
            link.fd = fd;
            link.listening = true;
            links_.push_back(std::make_unique<Link>(std::move(link)));
            return ok_failure();
        }

        // The coordinator may come up after its peers, so a refused connect is retried until the connect
        // deadline says otherwise.  Whatever comes out of this is a socket that is already connected.
        int fd = -1;
        const std::string where =
            config_.rank0.address + ":" + std::to_string(config_.rank0.port);
        std::string connect_err;
        if (!connect_within_deadline(fd, addr, where, clock_, at, connect_err)) {
            return Failure::fail(FailureCode::connect_failed, connect_err);
        }

        Link link;
        link.fd = fd;
        link.needs_hello = true;
        link.has_rank = true;
        link.rank = 0;  // the only peer a follower talks to
        links_.push_back(std::make_unique<Link>(std::move(link)));
        if (auto f = send_hello(*links_.back(), at)) return f;
        return ok_failure();
    }

    Failure send(uint32_t to_rank, FrameKind kind, std::string_view payload, Deadline at) override {
        if (closed_) {
            return Failure::fail(FailureCode::closed, "the control channel is closed");
        }
        if (config_.rank != 0 && to_rank != 0) {
            return Failure::fail(FailureCode::not_supported,
                                 "rank " + std::to_string(config_.rank) + " speaks to the coordinator only");
        }
        Link* link = link_for(to_rank);
        if (link == nullptr) {
            return Failure::fail(FailureCode::connect_failed,
                                 "rank " + std::to_string(to_rank) + " has no control link on this rank");
        }
        std::string frame;
        if (auto f = encode_frame(kind, payload, frame)) return f;
        return write_all(*link, frame, at);
    }

    Failure recv(ReceivedFrame& out, Deadline at) override {
        for (;;) {
            if (!pending_.empty()) {
                out = pending_.front();
                pending_.pop_front();
                return ok_failure();
            }
            // A stream that ended is reported only after what it managed to send has been handed over: a peer
            // can write its join and die in the same breath, and the join is still the more useful fact.
            if (auto f = stream_failure_) return f;
            if (closed_) {
                return Failure::fail(FailureCode::closed, "the control channel is closed");
            }
            const int left = remaining_ms(clock_, at);
            if (left <= 0) {
                return Failure::fail(FailureCode::deadline_expired, "no control frame arrived in time");
            }

            std::vector<pollfd> fds;
            std::vector<Link*> watched;
            fds.reserve(links_.size());
            watched.reserve(links_.size());
            for (auto& link : links_) {
                if (link->dead) continue;
                fds.push_back(pollfd{link->fd, POLLIN, 0});
                watched.push_back(link.get());
            }
            if (fds.empty()) {
                return Failure::fail(FailureCode::peer_closed, "this rank has no control links left");
            }

            const int pr = ::poll(fds.data(), static_cast<nfds_t>(fds.size()), left);
            if (pr < 0) {
                if (errno == EINTR) continue;
                return Failure::fail(FailureCode::peer_closed, "poll: " + errno_text());
            }
            if (pr == 0) {
                if (out_of_time(clock_, at)) {
                    return Failure::fail(FailureCode::deadline_expired,
                                          "no control frame arrived before the phase deadline");
                }
                continue;  // the slice ended, the phase has time left: wait again
            }
            for (size_t i = 0; i < watched.size(); ++i) {
                if (watched[i]->dead) continue;
                if (fds[i].revents == 0) continue;
                if (auto f = service(*watched[i], fds[i].revents)) {
                    compact_links();
                    return f;
                }
            }
            compact_links();
        }
    }

    void close() noexcept override {
        if (closed_) return;
        closed_ = true;
        for (auto& link : links_) {
            if (link->fd >= 0) ::close(link->fd);
            link->fd = -1;
        }
    }

 private:
    Link* link_for(uint32_t rank) {
        for (auto& link : links_) {
            if (!link->listening && link->has_rank && link->rank == rank) return link.get();
        }
        return nullptr;
    }

    Failure send_hello(Link& link, Deadline at) {
        std::string frame;
        if (auto f = encode_frame(FrameKind::hello, config_.credential, frame)) return f;
        return write_all(link, frame, at);
    }

    // Write the whole frame or say why not.  A short write is a wait, not a lost byte.
    Failure write_all(Link& link, std::string_view frame, Deadline at) {
        size_t sent = 0;
        for (;;) {
            const ssize_t n = ::send(link.fd, frame.data() + sent, frame.size() - sent, MSG_NOSIGNAL);
            if (n > 0) {
                sent += static_cast<size_t>(n);
                if (sent == frame.size()) return ok_failure();
                continue;
            }
            if (n < 0 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)) {
                const int left = remaining_ms(clock_, at);
                if (left <= 0) {
                    return Failure::fail(FailureCode::deadline_expired,
                                         "the control socket stayed full; the frame did not go out in time");
                }
                pollfd p{link.fd, POLLOUT, 0};
                const int pr = ::poll(&p, 1, left);
                if (pr < 0 && errno == EINTR) continue;
                if (pr == 0 && !out_of_time(clock_, at)) continue;  // slice over, phase still open
                if (pr <= 0) {
                    return Failure::fail(FailureCode::deadline_expired,
                                         "the control socket stayed full; the frame did not go out in time");
                }
                continue;
            }
            return Failure::fail(FailureCode::peer_closed, "send: " + errno_text());
        }
    }

    // One readable event on one link: accept, read, frame, authenticate, queue.
    Failure service(Link& link, short revents) {
        if (link.listening) {
            for (;;) {
                const int fd = ::accept(link.fd, nullptr, nullptr);
                if (fd < 0) {
                    if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) break;
                    return Failure::fail(FailureCode::peer_closed, "accept: " + errno_text());
                }
                std::string err;
                if (!set_nonblocking(fd, err)) {
                    ::close(fd);
                    return Failure::fail(FailureCode::peer_closed, err);
                }
                Link peer;
                peer.fd = fd;
                peer.needs_hello = true;
                links_.push_back(std::make_unique<Link>(std::move(peer)));
                if (auto f = send_hello(*links_.back(), deadline_now())) return f;
            }
            return ok_failure();
        }

        if (revents & POLLNVAL) {
            report_closed(link);
            return ok_failure();
        }

        // A reset can set POLLERR together with POLLIN while an already-delivered ABORT is still queued.
        // Read that data before observing the reset, just as we drain pending bytes before EOF.
        char scratch[4096];
        const ssize_t n = ::recv(link.fd, scratch, sizeof scratch, 0);
        if (n > 0) {
            std::vector<DecodedFrame> frames;
            if (auto f = link.decoder.feed(scratch, static_cast<size_t>(n), frames)) {
                drop(link);
                return f;
            }
            for (auto& frame : frames) {
                if (link.needs_hello) {
                    if (frame.kind != FrameKind::hello || frame.payload != config_.credential) {
                        stream_failure_ = Failure::fail(
                            FailureCode::credential,
                            "a peer's first control frame was not this session's credential");
                        drop(link);
                        return ok_failure();  // the reason names no secret: neither credential is in it
                    }
                    link.needs_hello = false;
                    continue;  // authentication is not a protocol event the session has to handle
                }
                if (frame.kind == FrameKind::join && !link.has_rank) {
                    JoinInfo info;
                    if (auto f = decode_join(frame.payload, info)) {
                        stream_failure_ = f;
                        drop(link);
                        return ok_failure();
                    }
                    link.has_rank = true;
                    link.rank = info.rank;
                }
                if (!link.has_rank) {
                    stream_failure_ = Failure::fail(FailureCode::bad_frame,
                                                    "a peer sent a control frame before identifying itself");
                    drop(link);
                    return ok_failure();
                }
                ReceivedFrame got;
                got.from_rank = link.rank;
                got.frame = std::move(frame);
                pending_.push_back(std::move(got));
            }
            // POLLHUP can accompany more than one read of queued bytes. Only recv()==0 is EOF.
            return ok_failure();
        }
        if (n == 0) {
            report_closed(link);
            return ok_failure();
        }
        if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) return ok_failure();
        stream_failure_ = Failure::fail(FailureCode::peer_closed, "recv: " + errno_text());
        drop(link);
        return ok_failure();
    }

    // A short look at the clock for a write that should not wait long (a fresh socket takes the frame at once).
    Deadline deadline_now() { return Deadline{clock_.now() + std::chrono::milliseconds(1000)}; }

    void drop(Link& link) {
        if (link.fd >= 0) ::close(link.fd);
        link.fd = -1;
        link.dead = true;
    }

    // A peer that is gone is a fact worth keeping, but not worth throwing a received frame away for: it is
    // reported once everything the peer did send has been handed to the session.
    void report_closed(Link& link) {
        if (!stream_failure_) {
            stream_failure_ = Failure::fail(
                link.decoder.buffered() ? FailureCode::truncated_frame : FailureCode::peer_closed,
                link.decoder.buffered() ? "a peer closed its control connection inside a frame"
                : link.has_rank ? "rank " + std::to_string(link.rank) + " closed its control connection"
                                : "a peer closed its control connection before it said which rank it was");
        }
        drop(link);
    }

    // Remove the links this pass closed, so a rank that lost every peer stops polling dead descriptors.
    void compact_links() {
        links_.erase(std::remove_if(links_.begin(), links_.end(),
                                    [](const std::unique_ptr<Link>& l) { return l->dead; }),
                     links_.end());
    }

    TcpChannelConfig config_;
    const Clock& clock_;
    std::vector<std::unique_ptr<Link>> links_;
    std::deque<ReceivedFrame> pending_;
    Failure stream_failure_{};
    bool closed_ = false;
};

}  // namespace

std::unique_ptr<ControlChannel> make_tcp_control_channel(const TcpChannelConfig& config, const Clock& clock,
                                                        std::string& err) {
    if (config.rank >= config.world) {
        err = "rank " + std::to_string(config.rank) + " is outside a world of " +
              std::to_string(config.world);
        return nullptr;
    }
    if (config.world != kTpWorld) {
        err = "this transport is TP3: three ranks, not " + std::to_string(config.world);
        return nullptr;
    }
    return std::make_unique<TcpChannel>(config, clock);
}

}  // namespace strata::distributed

#else  // no POSIX sockets: the codec above still builds, the channel says what it cannot do.

namespace strata::distributed {

std::unique_ptr<ControlChannel> make_tcp_control_channel(const TcpChannelConfig&, const Clock&,
                                                        std::string& err) {
    err = "the control channel needs a POSIX socket layer; this platform has no TCP control channel";
    return nullptr;
}

}  // namespace strata::distributed

#endif
