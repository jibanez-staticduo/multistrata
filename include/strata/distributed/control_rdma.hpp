#pragma once

#include "strata/distributed/session.hpp"

namespace strata::distributed {

// RDMA-CM RC/SEND-RECV control channel. TCP channel settings are reused for
// rank/world/credential and the IPv4 bind/peer endpoints, not for TCP sockets.
std::unique_ptr<ControlChannel> make_rdma_control_channel(const TcpChannelConfig& config, const Clock& clock,
                                                          std::string& err);

}  // namespace strata::distributed
