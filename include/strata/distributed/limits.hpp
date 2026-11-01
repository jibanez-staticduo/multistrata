#pragma once

#include <cstdint>
#include <cstddef>

namespace strata::distributed {

// Protocol capacity. Actual GPU capacity and resident-page admission are
// separately qualified by the chosen executor and model profile.
inline constexpr uint32_t kDistributedMaxContext = 262144;
inline constexpr size_t kDistributedMaxLineBytes = 4u << 20;
// Retained qualification limit of the diagnostic IQ/exact-reference runners.
inline constexpr uint32_t kDistributedReferenceMaxContext = 65792;

} // namespace strata::distributed
