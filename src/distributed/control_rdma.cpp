// RDMA-CM RC control channel. It uses SEND/RECV messages only; the existing
// authenticated frame protocol remains responsible for rank identity and payloads.
#include "strata/distributed/control_rdma.hpp"

#if defined(STRATA_BUILD_ROCE_EXPERIMENTAL) && defined(__linux__)

#include <arpa/inet.h>
#include <fcntl.h>
#include <poll.h>
#include <rdma/rdma_cma.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace strata::distributed {
namespace {

constexpr size_t kMessageBytes = 4096;
constexpr size_t kSendDepth = 16;
constexpr size_t kRecvDepth = 64;
constexpr size_t kMaxLinks = kTpWorld - 1;
constexpr auto kConnectRetryDelay = std::chrono::milliseconds(25);

int remaining_ms(const Clock& clock, Deadline at) {
    const auto remaining = at.end - clock.now();
    if (remaining <= std::chrono::steady_clock::duration::zero()) return 0;
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(remaining);
    const auto rounded = ms + (ms < remaining ? std::chrono::milliseconds(1) : std::chrono::milliseconds(0));
    return static_cast<int>(std::min<int64_t>(rounded.count(), INT32_MAX));
}

Failure system_failure(FailureCode code, std::string_view what, int status = errno) {
    std::string detail(what);
    detail += ": ";
    detail += std::strerror(status);
    return Failure::fail(code, detail);
}

const char* cm_event_name(rdma_cm_event_type type) {
    switch (type) {
        case RDMA_CM_EVENT_ADDR_RESOLVED: return "ADDR_RESOLVED";
        case RDMA_CM_EVENT_ADDR_ERROR: return "ADDR_ERROR";
        case RDMA_CM_EVENT_ROUTE_RESOLVED: return "ROUTE_RESOLVED";
        case RDMA_CM_EVENT_ROUTE_ERROR: return "ROUTE_ERROR";
        case RDMA_CM_EVENT_CONNECT_REQUEST: return "CONNECT_REQUEST";
        case RDMA_CM_EVENT_CONNECT_RESPONSE: return "CONNECT_RESPONSE";
        case RDMA_CM_EVENT_CONNECT_ERROR: return "CONNECT_ERROR";
        case RDMA_CM_EVENT_UNREACHABLE: return "UNREACHABLE";
        case RDMA_CM_EVENT_REJECTED: return "REJECTED";
        case RDMA_CM_EVENT_ESTABLISHED: return "ESTABLISHED";
        case RDMA_CM_EVENT_DISCONNECTED: return "DISCONNECTED";
        case RDMA_CM_EVENT_DEVICE_REMOVAL: return "DEVICE_REMOVAL";
        case RDMA_CM_EVENT_MULTICAST_JOIN: return "MULTICAST_JOIN";
        case RDMA_CM_EVENT_MULTICAST_ERROR: return "MULTICAST_ERROR";
        case RDMA_CM_EVENT_ADDR_CHANGE: return "ADDR_CHANGE";
        case RDMA_CM_EVENT_TIMEWAIT_EXIT: return "TIMEWAIT_EXIT";
    }
    return "UNKNOWN";
}

bool ipv4(std::string_view text, sockaddr_in& out) {
    out = {};
    out.sin_family = AF_INET;
    const std::string address(text);
    return ::inet_pton(AF_INET, address.c_str(), &out.sin_addr) == 1;
}

struct Buffer {
    void* data = nullptr;
    ibv_mr* mr = nullptr;
    ibv_sge sge{};
    ibv_recv_wr recv_wr{};
    ibv_send_wr send_wr{};
    bool recv_posted = false;
    bool send_busy = false;
};

struct Link {
    rdma_cm_id* id = nullptr;
    uint32_t rank = UINT32_MAX;
    std::vector<std::unique_ptr<Buffer>> sends;
    std::vector<std::unique_ptr<Buffer>> receives;
    size_t next_send = 0;
    bool established = false;
    bool authenticated = false;
    bool dead = false;
    FrameDecoder decoder;
};

class RdmaControlChannel final : public ControlChannel {
 public:
    RdmaControlChannel(TcpChannelConfig config, const Clock& clock)
        : config_(std::move(config)), clock_(clock) {}
    ~RdmaControlChannel() override { close(); }

    Failure open(Deadline at) override {
        if (opened_) return Failure::fail(FailureCode::invalid_config, "RDMA control channel opened twice");
        opened_ = true;
        if (config_.world != kTpWorld || config_.rank >= config_.world || config_.self.port == 0 ||
            config_.credential.empty())
            return Failure::fail(FailureCode::invalid_config, "invalid TP3 RDMA control configuration");
        event_channel_ = rdma_create_event_channel();
        if (event_channel_ == nullptr) return system_failure(FailureCode::connect_failed, "rdma_create_event_channel");
        const int flags = ::fcntl(event_channel_->fd, F_GETFL, 0);
        if (flags < 0 || ::fcntl(event_channel_->fd, F_SETFL, flags | O_NONBLOCK) < 0)
            return system_failure(FailureCode::connect_failed, "configure RDMA-CM event channel");
        if (config_.rank == 0) return open_listener();
        return connect_peer(at);
    }

    Failure send(uint32_t to_rank, FrameKind kind, std::string_view payload, Deadline at) override {
        if (closed_) return Failure::fail(FailureCode::closed, "the RDMA control channel is closed");
        if (config_.rank != 0 && to_rank != 0)
            return Failure::fail(FailureCode::not_supported, "an RDMA follower communicates with rank 0 only");
        Link* link = find_link(to_rank);
        if (link == nullptr || !link->authenticated)
            return Failure::fail(FailureCode::connect_failed, "RDMA control peer is not authenticated");
        return send_frame(*link, kind, payload, at);
    }

    Failure recv(ReceivedFrame& out, Deadline at) override {
        for (;;) {
            if (!pending_.empty()) {
                out = std::move(pending_.front());
                pending_.pop_front();
                return ok_failure();
            }
            if (stream_failure_) return stream_failure_;
            if (closed_) return Failure::fail(FailureCode::closed, "the RDMA control channel is closed");
            if (auto f = progress_any(at)) return f;
        }
    }

    void close() noexcept override {
        if (closed_) return;
        closed_ = true;
        for (auto& link : links_) destroy_link(*link);
        links_.clear();
        if (listener_ != nullptr) {
            rdma_destroy_id(listener_);
            listener_ = nullptr;
        }
        if (event_channel_ != nullptr) {
            rdma_destroy_event_channel(event_channel_);
            event_channel_ = nullptr;
        }
    }

 private:
    Failure open_listener() {
        sockaddr_in address{};
        if (!ipv4(config_.self.address, address))
            return Failure::fail(FailureCode::invalid_config, "RDMA bind address must be numeric IPv4");
        address.sin_port = htons(config_.self.port);
        if (rdma_create_id(event_channel_, &listener_, nullptr, RDMA_PS_TCP) != 0)
            return system_failure(FailureCode::listen_failed, "rdma_create_id");
        if (rdma_bind_addr(listener_, reinterpret_cast<sockaddr*>(&address)) != 0)
            return system_failure(FailureCode::listen_failed, "rdma_bind_addr");
        if (rdma_listen(listener_, static_cast<int>(kMaxLinks)) != 0)
            return system_failure(FailureCode::listen_failed, "rdma_listen");
        return ok_failure();
    }

    Failure connect_peer(Deadline at) {
        sockaddr_in source{}, destination{};
        if (!ipv4(config_.self.address, source) || !ipv4(config_.rank0.address, destination))
            return Failure::fail(FailureCode::invalid_config, "RDMA peer addresses must be numeric IPv4");
        destination.sin_port = htons(config_.rank0.port);
        for (;;) {
            if (remaining_ms(clock_, at) <= 0)
                return Failure::fail(FailureCode::deadline_expired, "RDMA connect deadline passed");
            rdma_cm_id* id = nullptr;
            if (rdma_create_id(event_channel_, &id, nullptr, RDMA_PS_TCP) != 0)
                return system_failure(FailureCode::connect_failed, "rdma_create_id");
            auto link = std::make_unique<Link>();
            link->id = id;
            link->rank = 0;
            id->context = link.get();
            links_.push_back(std::move(link));
            bool retryable = false;
            Failure failure;
            if (rdma_resolve_addr(id, reinterpret_cast<sockaddr*>(&source),
                                  reinterpret_cast<sockaddr*>(&destination), 2000) != 0) {
                failure = system_failure(FailureCode::connect_failed, "rdma_resolve_addr");
            } else if ((failure = expect_cm_event(RDMA_CM_EVENT_ADDR_RESOLVED, at, &retryable))) {
            } else if (rdma_resolve_route(id, 2000) != 0) {
                failure = system_failure(FailureCode::connect_failed, "rdma_resolve_route");
            } else if ((failure = expect_cm_event(RDMA_CM_EVENT_ROUTE_RESOLVED, at, &retryable))) {
            } else if ((failure = create_resources(*links_.back()))) {
            } else {
                rdma_conn_param parameters{};
                parameters.initiator_depth = 1;
                parameters.responder_resources = 1;
                parameters.retry_count = 5;
                parameters.rnr_retry_count = 5;
                if (rdma_connect(id, &parameters) != 0)
                    failure = system_failure(FailureCode::connect_failed, "rdma_connect");
                else
                    failure = expect_cm_event(RDMA_CM_EVENT_ESTABLISHED, at, &retryable);
            }
            if (!failure) return establish(*links_.back(), at);
            destroy_link(*links_.back());
            links_.pop_back();
            if (!retryable) return failure;
            const int left = remaining_ms(clock_, at);
            if (left <= 0) return Failure::fail(FailureCode::deadline_expired, "RDMA connect deadline passed");
            pollfd delay{};
            const int wait = static_cast<int>(std::min<int64_t>(kConnectRetryDelay.count(), left));
            const int result = ::poll(&delay, 0, wait);
            if (result < 0 && errno != EINTR)
                return system_failure(FailureCode::connect_failed, "RDMA connect retry wait");
        }
    }

    Failure expect_cm_event(rdma_cm_event_type expected, Deadline at, bool* retryable = nullptr) {
        for (;;) {
            rdma_cm_event* event = nullptr;
            if (auto f = get_cm_event(event, at)) return f;
            const auto actual = event->event;
            const int status = event->status;
            rdma_ack_cm_event(event);
            if (actual == expected) return ok_failure();
            if (actual == RDMA_CM_EVENT_ADDR_ERROR || actual == RDMA_CM_EVENT_ROUTE_ERROR ||
                actual == RDMA_CM_EVENT_CONNECT_ERROR || actual == RDMA_CM_EVENT_UNREACHABLE ||
                actual == RDMA_CM_EVENT_REJECTED) {
                if (retryable != nullptr) {
                    *retryable = actual == RDMA_CM_EVENT_UNREACHABLE ||
                                 (actual == RDMA_CM_EVENT_REJECTED && status == 8);
                }
                return Failure::fail(FailureCode::connect_failed,
                    std::string("RDMA-CM ") + cm_event_name(actual) + " (event " +
                    std::to_string(static_cast<int>(actual)) + ", status " + std::to_string(status) +
                    "), expected " + cm_event_name(expected));
            }
            return Failure::fail(FailureCode::connect_failed,
                std::string("unexpected RDMA-CM ") + cm_event_name(actual) + " (event " +
                std::to_string(static_cast<int>(actual)) + ", status " + std::to_string(status) +
                "), expected " + cm_event_name(expected));
        }
    }

    Failure get_cm_event(rdma_cm_event*& event, Deadline at) {
        for (;;) {
            if (rdma_get_cm_event(event_channel_, &event) == 0) return ok_failure();
            if (errno != EAGAIN && errno != EWOULDBLOCK)
                return system_failure(FailureCode::peer_closed, "rdma_get_cm_event");
            if (auto f = wait_fd(event_channel_->fd, POLLIN, at, "RDMA-CM event")) return f;
        }
    }

    Failure wait_fd(int fd, short events, Deadline at, std::string_view what) const {
        for (;;) {
            const int left = remaining_ms(clock_, at);
            if (left <= 0) return Failure::fail(FailureCode::deadline_expired, std::string(what) + " deadline passed");
            pollfd descriptor{fd, events, 0};
            const int result = ::poll(&descriptor, 1, left);
            if (result > 0) return ok_failure();
            if (result < 0 && errno == EINTR) continue;
            if (result == 0) continue;
            return system_failure(FailureCode::peer_closed, what);
        }
    }

    Failure create_resources(Link& link) {
        link.id->pd = ibv_alloc_pd(link.id->verbs);
        if (link.id->pd == nullptr) return system_failure(FailureCode::connect_failed, "ibv_alloc_pd");
        ibv_comp_channel* completion_channel = ibv_create_comp_channel(link.id->verbs);
        if (completion_channel == nullptr)
            return system_failure(FailureCode::connect_failed, "ibv_create_comp_channel");
        link.id->send_cq_channel = completion_channel;
        link.id->recv_cq_channel = completion_channel;
        const int flags = ::fcntl(completion_channel->fd, F_GETFL, 0);
        if (flags < 0 || ::fcntl(completion_channel->fd, F_SETFL, flags | O_NONBLOCK) < 0)
            return system_failure(FailureCode::connect_failed, "configure RDMA completion channel");
        ibv_cq* cq = ibv_create_cq(link.id->verbs, static_cast<int>(kSendDepth + kRecvDepth), &link,
                                   completion_channel, 0);
        if (cq == nullptr) return system_failure(FailureCode::connect_failed, "ibv_create_cq");
        link.id->send_cq = cq;
        link.id->recv_cq = cq;
        ibv_qp_init_attr qp{};
        qp.send_cq = cq;
        qp.recv_cq = cq;
        qp.qp_type = IBV_QPT_RC;
        qp.cap.max_send_wr = static_cast<uint32_t>(kSendDepth);
        qp.cap.max_recv_wr = static_cast<uint32_t>(kRecvDepth);
        qp.cap.max_send_sge = 1;
        qp.cap.max_recv_sge = 1;
        if (rdma_create_qp(link.id, link.id->pd, &qp) != 0)
            return system_failure(FailureCode::connect_failed, "rdma_create_qp");
        for (size_t index = 0; index < kRecvDepth; ++index) {
            auto buffer = std::make_unique<Buffer>();
            buffer->data = std::malloc(kMessageBytes);
            if (buffer->data == nullptr) return Failure::fail(FailureCode::connect_failed, "allocate RDMA receive buffer");
            buffer->mr = ibv_reg_mr(link.id->pd, buffer->data, kMessageBytes, IBV_ACCESS_LOCAL_WRITE);
            if (buffer->mr == nullptr) {
                std::free(buffer->data);
                buffer->data = nullptr;
                return system_failure(FailureCode::connect_failed, "ibv_reg_mr receive");
            }
            buffer->sge = {reinterpret_cast<uint64_t>(buffer->data), static_cast<uint32_t>(kMessageBytes),
                           buffer->mr->lkey};
            buffer->recv_wr.wr_id = reinterpret_cast<uint64_t>(buffer.get());
            buffer->recv_wr.sg_list = &buffer->sge;
            buffer->recv_wr.num_sge = 1;
            link.receives.push_back(std::move(buffer));
        }
        for (size_t index = 0; index < kSendDepth; ++index) {
            auto buffer = std::make_unique<Buffer>();
            buffer->data = std::malloc(kMessageBytes);
            if (buffer->data == nullptr) return Failure::fail(FailureCode::connect_failed, "allocate RDMA send buffer");
            buffer->mr = ibv_reg_mr(link.id->pd, buffer->data, kMessageBytes, 0);
            if (buffer->mr == nullptr) {
                std::free(buffer->data);
                buffer->data = nullptr;
                return system_failure(FailureCode::connect_failed, "ibv_reg_mr send");
            }
            buffer->sge = {reinterpret_cast<uint64_t>(buffer->data), 0, buffer->mr->lkey};
            buffer->send_wr.wr_id = reinterpret_cast<uint64_t>(buffer.get());
            buffer->send_wr.sg_list = &buffer->sge;
            buffer->send_wr.num_sge = 1;
            link.sends.push_back(std::move(buffer));
        }
        return ok_failure();
    }

    Failure establish(Link& link, Deadline at) {
        link.established = true;
        if (auto f = post_receives(link)) return f;
        if (ibv_req_notify_cq(link.id->recv_cq, 0) != 0)
            return system_failure(FailureCode::comm_failed, "ibv_req_notify_cq");
        std::string hello;
        if (auto f = encode_frame(FrameKind::hello, config_.credential, hello)) return f;
        if (auto f = send_bytes(link, hello, at)) return f;
        while (!link.authenticated) {
            if (auto f = progress_link(link, at)) return f;
            if (stream_failure_) return stream_failure_;
            if (link.dead)
                return Failure::fail(FailureCode::peer_closed, "RDMA control peer closed during authentication");
        }
        return ok_failure();
    }

    Failure post_receives(Link& link) {
        for (auto& buffer : link.receives) {
            if (buffer->recv_posted) continue;
            buffer->recv_wr.next = nullptr;
            ibv_recv_wr* bad = nullptr;
            if (ibv_post_recv(link.id->qp, &buffer->recv_wr, &bad) != 0)
                return system_failure(FailureCode::comm_failed, "ibv_post_recv");
            buffer->recv_posted = true;
        }
        return ok_failure();
    }

    Failure send_frame(Link& link, FrameKind kind, std::string_view payload, Deadline at) {
        std::string frame;
        if (auto f = encode_frame(kind, payload, frame)) return f;
        return send_bytes(link, frame, at);
    }

    Failure send_bytes(Link& link, std::string_view frame, Deadline at) {
        size_t offset = 0;
        while (offset < frame.size()) {
            Buffer* buffer = nullptr;
            while (buffer == nullptr) {
                for (auto& candidate : link.sends) {
                    if (!candidate->send_busy) {
                        buffer = candidate.get();
                        break;
                    }
                }
                if (buffer == nullptr) {
                    if (auto f = progress_link(link, at)) return f;
                }
            }
            const size_t amount = std::min(kMessageBytes, frame.size() - offset);
            std::memcpy(buffer->data, frame.data() + offset, amount);
            buffer->sge.length = static_cast<uint32_t>(amount);
            buffer->send_wr.next = nullptr;
            buffer->send_wr.opcode = IBV_WR_SEND;
            buffer->send_wr.send_flags = IBV_SEND_SIGNALED;
            buffer->send_busy = true;
            ibv_send_wr* bad = nullptr;
            if (ibv_post_send(link.id->qp, &buffer->send_wr, &bad) != 0) {
                buffer->send_busy = false;
                return system_failure(FailureCode::comm_failed, "ibv_post_send");
            }
            offset += amount;
            while (buffer->send_busy) {
                if (auto f = progress_link(link, at)) return f;
            }
        }
        return ok_failure();
    }

    Failure progress_link(Link& link, Deadline at) {
        ibv_cq* cq = link.id->send_cq;
        if (ibv_req_notify_cq(cq, 0) != 0) return system_failure(FailureCode::comm_failed, "ibv_req_notify_cq");
        ibv_wc completions[16]{};
        int count = ibv_poll_cq(cq, 16, completions);
        if (count < 0) return Failure::fail(FailureCode::comm_failed, "ibv_poll_cq failed");
        if (count > 0) return dispatch_completions(link, completions, count);
        if (auto f = wait_fd(link.id->send_cq_channel->fd, POLLIN, at, "RDMA completion")) return f;
        ibv_cq* event_cq = nullptr;
        void* context = nullptr;
        if (ibv_get_cq_event(link.id->send_cq_channel, &event_cq, &context) != 0)
            return system_failure(FailureCode::comm_failed, "ibv_get_cq_event");
        ibv_ack_cq_events(event_cq, 1);
        count = ibv_poll_cq(cq, 16, completions);
        if (count < 0) return Failure::fail(FailureCode::comm_failed, "ibv_poll_cq failed");
        if (count == 0) return ok_failure();
        return dispatch_completions(link, completions, count);
    }

    Failure dispatch_completions(Link& link, ibv_wc* completions, int count) {
        for (int index = 0; index < count; ++index) {
            const ibv_wc& completion = completions[index];
            auto* buffer = reinterpret_cast<Buffer*>(completion.wr_id);
            if (completion.status != IBV_WC_SUCCESS) {
                link.dead = true;
                return Failure::fail(FailureCode::comm_failed, "RDMA work completion failed");
            }
            if (completion.opcode == IBV_WC_SEND) {
                buffer->send_busy = false;
                continue;
            }
            if (completion.opcode != IBV_WC_RECV || completion.byte_len > kMessageBytes) {
                link.dead = true;
                return Failure::fail(FailureCode::bad_frame, "unexpected or oversized RDMA receive completion");
            }
            buffer->recv_posted = false;
            std::vector<DecodedFrame> frames;
            if (auto f = link.decoder.feed(static_cast<const char*>(buffer->data), completion.byte_len, frames)) {
                link.dead = true;
                return f;
            }
            for (auto& frame : frames) {
                if (!link.authenticated) {
                    if (frame.kind != FrameKind::hello || frame.payload != config_.credential) {
                        link.dead = true;
                        stream_failure_ = Failure::fail(FailureCode::credential,
                                                        "a peer's first RDMA control frame was not authenticated");
                        break;
                    }
                    link.authenticated = true;
                    continue;
                }
                if (frame.kind == FrameKind::hello) {
                    link.dead = true;
                    stream_failure_ = Failure::fail(FailureCode::credential, "an RDMA peer repeated its hello frame");
                    break;
                }
                if (link.rank == UINT32_MAX) {
                    if (frame.kind != FrameKind::join) {
                        link.dead = true;
                        stream_failure_ = Failure::fail(FailureCode::bad_frame,
                                                        "an RDMA peer sent a frame before identifying its rank");
                        break;
                    }
                    JoinInfo info;
                    if (auto f = decode_join(frame.payload, info)) {
                        link.dead = true;
                        stream_failure_ = f;
                        break;
                    }
                    if (info.rank == 0 || info.rank >= config_.world || find_link(info.rank) != nullptr) {
                        link.dead = true;
                        stream_failure_ = Failure::fail(FailureCode::duplicate_rank,
                                                        "an RDMA link claimed an invalid or duplicate rank");
                        break;
                    }
                    link.rank = info.rank;
                }
                ReceivedFrame received;
                received.from_rank = link.rank;
                received.frame = std::move(frame);
                pending_.push_back(std::move(received));
            }
            if (!link.dead) {
                buffer->recv_wr.wr_id = reinterpret_cast<uint64_t>(buffer);
                if (auto f = post_receives(link)) return f;
            }
        }
        return ok_failure();
    }

    Failure progress_any(Deadline at) {
        // CQ events may already have been consumed by the send-side progress path.
        // Always inspect completions before sleeping on event descriptors.
        std::vector<Link*> active;
        for (const auto& link : links_) {
            if (!link->established || link->dead) continue;
            active.push_back(link.get());
            ibv_wc completions[16]{};
            const int count = ibv_poll_cq(link->id->recv_cq, 16, completions);
            if (count < 0) return Failure::fail(FailureCode::comm_failed, "ibv_poll_cq failed");
            if (count > 0) {
                if (auto f = dispatch_completions(*link, completions, count)) return f;
                continue;
            }
            if (ibv_req_notify_cq(link->id->recv_cq, 0) != 0)
                return system_failure(FailureCode::comm_failed, "ibv_req_notify_cq");
            const int raced_count = ibv_poll_cq(link->id->recv_cq, 16, completions);
            if (raced_count < 0) return Failure::fail(FailureCode::comm_failed, "ibv_poll_cq failed");
            if (raced_count > 0) {
                if (auto f = dispatch_completions(*link, completions, raced_count)) return f;
            }
        }
        if (!pending_.empty()) return ok_failure();
        std::vector<pollfd> descriptors;
        descriptors.push_back({event_channel_->fd, POLLIN, 0});
        for (const Link* link : active)
            descriptors.push_back({link->id->recv_cq_channel->fd, POLLIN, 0});
        const int left = remaining_ms(clock_, at);
        if (left <= 0) return Failure::fail(FailureCode::deadline_expired, "RDMA receive deadline passed");
        const int result = ::poll(descriptors.data(), static_cast<nfds_t>(descriptors.size()), left);
        if (result < 0 && errno == EINTR) return ok_failure();
        if (result < 0) return system_failure(FailureCode::peer_closed, "poll RDMA control");
        if (result == 0) return Failure::fail(FailureCode::deadline_expired, "no RDMA control frame arrived in time");
        if (descriptors[0].revents & (POLLERR | POLLHUP | POLLNVAL))
            return Failure::fail(FailureCode::peer_closed, "RDMA-CM event channel closed");
        if (descriptors[0].revents & POLLIN) {
            if (auto f = process_cm_events(at)) return f;
        }
        for (size_t index = 0; index < active.size(); ++index) {
            Link& link = *active[index];
            if (descriptors[index + 1].revents & (POLLERR | POLLHUP | POLLNVAL)) {
                link.dead = true;
                return Failure::fail(FailureCode::peer_closed, "RDMA completion channel closed");
            }
            if ((descriptors[index + 1].revents & POLLIN) == 0) continue;
            if (auto f = drain_ready_cq(link)) return f;
        }
        return ok_failure();
    }

    Failure process_cm_events(Deadline at) {
        for (;;) {
            rdma_cm_event* event = nullptr;
            if (rdma_get_cm_event(event_channel_, &event) != 0) {
                if (errno == EAGAIN || errno == EWOULDBLOCK) return ok_failure();
                return system_failure(FailureCode::peer_closed, "rdma_get_cm_event");
            }
            const auto type = event->event;
            const int status = event->status;
            rdma_cm_id* id = event->id;
            rdma_cm_id* listen_id = event->listen_id;
            if (type == RDMA_CM_EVENT_CONNECT_REQUEST && config_.rank == 0 && listen_id == listener_) {
                rdma_ack_cm_event(event);
                if (links_.size() >= kMaxLinks)
                    return Failure::fail(FailureCode::duplicate_rank, "too many RDMA control connections");
                auto link = std::make_unique<Link>();
                link->id = id;
                id->context = link.get();
                links_.push_back(std::move(link));
                if (auto f = create_resources(*links_.back())) return f;
                if (auto f = post_receives(*links_.back())) return f;
                if (ibv_req_notify_cq(id->recv_cq, 0) != 0)
                    return system_failure(FailureCode::comm_failed, "ibv_req_notify_cq");
                rdma_conn_param parameters{};
                parameters.initiator_depth = 1;
                parameters.responder_resources = 1;
                parameters.rnr_retry_count = 5;
                if (rdma_accept(id, &parameters) != 0)
                    return system_failure(FailureCode::connect_failed, "rdma_accept");
                continue;
            }
            rdma_ack_cm_event(event);
            if (type == RDMA_CM_EVENT_ESTABLISHED && config_.rank == 0) {
                Link* link = find_link(id);
                if (link == nullptr) return Failure::fail(FailureCode::bad_frame, "unknown RDMA connection established");
                if (auto f = establish(*link, at)) return f;
            } else if (type == RDMA_CM_EVENT_DISCONNECTED || type == RDMA_CM_EVENT_DEVICE_REMOVAL) {
                if (auto* link = find_link(id)) {
                    link->dead = true;
                    if (!stream_failure_)
                        stream_failure_ = Failure::fail(FailureCode::peer_closed, "an RDMA control peer disconnected");
                }
            } else if (status != 0) {
                return system_failure(FailureCode::connect_failed, "RDMA-CM connection event failed", status);
            }
        }
    }

    Failure drain_ready_cq(Link& link) {
        ibv_cq* cq = link.id->recv_cq;
        ibv_cq* event_cq = nullptr;
        void* context = nullptr;
        if (ibv_get_cq_event(link.id->recv_cq_channel, &event_cq, &context) != 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) return ok_failure();
            return system_failure(FailureCode::comm_failed, "ibv_get_cq_event");
        }
        ibv_ack_cq_events(event_cq, 1);
        for (;;) {
            if (ibv_req_notify_cq(cq, 0) != 0) return system_failure(FailureCode::comm_failed, "ibv_req_notify_cq");
            ibv_wc completions[16]{};
            const int count = ibv_poll_cq(cq, 16, completions);
            if (count < 0) return Failure::fail(FailureCode::comm_failed, "ibv_poll_cq failed");
            if (count == 0) return ok_failure();
            if (auto f = dispatch_completions(link, completions, count)) return f;
        }
    }

    Link* find_link(uint32_t rank) {
        for (auto& link : links_)
            if (!link->dead && link->rank == rank) return link.get();
        return nullptr;
    }

    Link* find_link(rdma_cm_id* id) {
        for (auto& link : links_)
            if (link->id == id && !link->dead) return link.get();
        return nullptr;
    }

    void destroy_link(Link& link) noexcept {
        if (link.id == nullptr) return;
        if (link.id->qp != nullptr) rdma_destroy_qp(link.id);
        for (auto& buffer : link.receives) {
            if (buffer->mr != nullptr) ibv_dereg_mr(buffer->mr);
            std::free(buffer->data);
        }
        for (auto& buffer : link.sends) {
            if (buffer->mr != nullptr) ibv_dereg_mr(buffer->mr);
            std::free(buffer->data);
        }
        if (link.id->recv_cq_channel != nullptr) {
            ibv_cq* cq = link.id->recv_cq;
            for (;;) {
                ibv_cq* event_cq = nullptr;
                void* context = nullptr;
                if (ibv_get_cq_event(link.id->recv_cq_channel, &event_cq, &context) != 0) {
                    if (errno == EAGAIN || errno == EWOULDBLOCK) break;
                    break;
                }
                ibv_ack_cq_events(event_cq ? event_cq : cq, 1);
            }
        }
        if (link.id->recv_cq != nullptr) ibv_destroy_cq(link.id->recv_cq);
        if (link.id->recv_cq_channel != nullptr) ibv_destroy_comp_channel(link.id->recv_cq_channel);
        if (link.id->pd != nullptr) ibv_dealloc_pd(link.id->pd);
        rdma_destroy_id(link.id);
        link.id = nullptr;
    }

    TcpChannelConfig config_;
    const Clock& clock_;
    rdma_event_channel* event_channel_ = nullptr;
    rdma_cm_id* listener_ = nullptr;
    std::vector<std::unique_ptr<Link>> links_;
    std::deque<ReceivedFrame> pending_;
    Failure stream_failure_{};
    bool opened_ = false;
    bool closed_ = false;
};

}  // namespace

std::unique_ptr<ControlChannel> make_rdma_control_channel(const TcpChannelConfig& config, const Clock& clock,
                                                          std::string& err) {
    if (config.rank >= config.world || config.world != kTpWorld) {
        err = "RDMA control supports exactly three ranks";
        return nullptr;
    }
    if (config.self.port == 0 || config.credential.empty()) {
        err = "RDMA control needs a configured port and session credential";
        return nullptr;
    }
    return std::make_unique<RdmaControlChannel>(config, clock);
}

}  // namespace strata::distributed

#else

namespace strata::distributed {

std::unique_ptr<ControlChannel> make_rdma_control_channel(const TcpChannelConfig&, const Clock&,
                                                          std::string& err) {
    err = "the RDMA-CM control channel is available only in the Linux RoCE experimental build";
    return nullptr;
}

}  // namespace strata::distributed

#endif
