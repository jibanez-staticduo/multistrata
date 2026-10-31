#pragma once

#include "strata/core/native_upload.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace strata::distributed {

enum class ExpertResidency { SelectedStaging, AllResident };

struct ReferenceResidencyPlan {
    std::vector<size_t> layer_bytes;
    size_t owned_weight_bytes = 0;

    static size_t add(size_t a, size_t b) {
        if (b > std::numeric_limits<size_t>::max() - a)
            throw std::runtime_error("TP1 reference: expert storage size overflow");
        return a + b;
    }
    static size_t multiply(size_t a, size_t b) {
        if (a && b > std::numeric_limits<size_t>::max() / a)
            throw std::runtime_error("TP1 reference: expert storage size overflow");
        return a * b;
    }
    static ReferenceResidencyPlan all(const std::vector<size_t>& slots, size_t experts) {
        if (slots.empty() || !experts)
            throw std::runtime_error("TP1 reference: empty resident expert plan");
        ReferenceResidencyPlan plan;
        for (const auto slot : slots) {
            if (!slot) throw std::runtime_error("TP1 reference: empty original expert blob");
            const auto bytes = multiply(slot, experts);
            plan.layer_bytes.push_back(bytes);
            plan.owned_weight_bytes = add(plan.owned_weight_bytes, bytes);
        }
        return plan;
    }
    size_t required_bytes(size_t buffers) const { return add(owned_weight_bytes, buffers); }
};

// Three GGUF tensors store each role's experts consecutively. The native kernel
// consumes one contiguous gate/up/down blob per expert, with no codec changes.
template<class Backend>
typename Backend::Error upload_reference_experts(core::NativeUpload<Backend>& upload, uint8_t* destination,
        size_t experts, size_t slot_bytes, const uint8_t* const (&roles)[3],
        const size_t (&role_bytes)[3]) {
    if (!destination || !experts || !roles[0] || !roles[1] || !roles[2] ||
        ReferenceResidencyPlan::add(ReferenceResidencyPlan::add(role_bytes[0], role_bytes[1]),
                                   role_bytes[2]) != slot_bytes)
        throw std::runtime_error("TP1 reference: invalid original expert role spans");
    (void)ReferenceResidencyPlan::multiply(slot_bytes, experts);
    for (size_t expert = 0; expert < experts; ++expert) {
        size_t offset = 0;
        for (size_t role = 0; role < 3; ++role) {
            const auto status = upload.copy(destination + expert * slot_bytes + offset,
                roles[role] + expert * role_bytes[role], role_bytes[role]);
            if (status != Backend::success) return status;
            offset += role_bytes[role];
        }
    }
    return Backend::success;
}

} // namespace strata::distributed
