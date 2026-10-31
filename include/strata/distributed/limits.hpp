#pragma once

#include <cstdint>

namespace strata::distributed {

// Bounded capacity for a 64K prompt plus 256 output tokens. GPU qualification is separate.
inline constexpr uint32_t kDistributedMaxContext = 65792;

} // namespace strata::distributed
