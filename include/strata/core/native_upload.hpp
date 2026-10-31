#pragma once

#include <algorithm>
#include <cstddef>
#include <cstring>

namespace strata::core {

// Keep file-backed pageable pointers out of the CUDA transfer path. The backend's
// copy must finish reading the pinned stage before returning so it can be reused.
template<class Backend>
class NativeUpload {
public:
    using Error = typename Backend::Error;
    static constexpr std::size_t capacity = 8u << 20;

    explicit NativeUpload(Backend& backend) : backend_(backend) {}
    ~NativeUpload() { if (stage_) backend_.release(stage_); }
    NativeUpload(const NativeUpload&) = delete;
    NativeUpload& operator=(const NativeUpload&) = delete;

    Error copy(void* destination, const void* source, std::size_t bytes) {
        if (!bytes) return Backend::success;
        if (!stage_) {
            const auto status = backend_.allocate(&stage_, capacity);
            if (status != Backend::success) return status;
        }
        auto* dst = static_cast<unsigned char*>(destination);
        const auto* src = static_cast<const unsigned char*>(source);
        for (std::size_t offset = 0; offset < bytes;) {
            const auto count = (std::min)(capacity, bytes - offset);
            std::memcpy(stage_, src + offset, count);
            const auto status = backend_.copy(dst + offset, stage_, count);
            if (status != Backend::success) return status;
            offset += count;
        }
        return Backend::success;
    }

private:
    Backend& backend_;
    void* stage_ = nullptr;
};

}  // namespace strata::core
