#pragma once

#include "strata/kernels/sampler.hpp"

#include <cstdint>
#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace strata::distributed {

struct ServeRequest {
    enum class Kind { generate, cancel, shutdown };
    Kind kind = Kind::shutdown;
    std::string request_id;
    std::vector<int32_t> ids;
    uint32_t max_new = 0;
    kernels::SamplerParams sampling;
    std::map<int32_t, float> logit_bias;
};

// One JSON value, at most 1 MiB and 16 nested containers. Failure leaves out unchanged.
// generate requires type, request_id, ids, max_new and sampling; cancel requires
// type and request_id; shutdown requires only type. The context is limited to 65536.
bool parse_serve_request(std::string_view json, int32_t n_vocab, uint32_t context,
                         ServeRequest& out, std::string& err);

// Returns one JSON object without a trailing newline. Unknown/invalid events throw
// std::invalid_argument. All events identify the complete three-rank roster.
std::string serve_event(std::string_view type, std::string_view request_id = {},
                        int32_t token = -1, uint32_t max_context = 0,
                        std::string_view state = "ready");

}  // namespace strata::distributed
