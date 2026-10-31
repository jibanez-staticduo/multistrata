#pragma once

#include "strata/distributed/transport.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace strata::distributed {

inline constexpr uint64_t kHybridRoceGatherMaxBytes = 4u * 1024u * 1024u;
inline constexpr uint32_t kHybridBootstrapVersion = 1;
inline constexpr size_t kHybridBootstrapFieldMax = 4096;
inline constexpr std::string_view kHybridBackendIdentity = "nccl+roce-v1";

struct HybridBootstrap {
    std::string nccl_id_hex;
    std::vector<uint8_t> roce_blob;
};

namespace hybrid_detail {

inline constexpr std::string_view kHybridBundlePrefix = "strata-hybrid-v1:";
inline constexpr std::string_view kBase64Alphabet =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
inline constexpr size_t kNcclUniqueIdBytes = 128;

inline int base64_value(char c) {
    const auto at = kBase64Alphabet.find(c);
    return at == std::string_view::npos ? -1 : static_cast<int>(at);
}

inline Failure decode_base64url(std::string_view encoded, std::vector<uint8_t>& out) {
    out.clear();
    if (encoded.empty() || encoded.size() % 4 == 1) {
        return Failure::fail(FailureCode::bad_frame, "hybrid bootstrap has invalid base64url length");
    }
    uint32_t accumulator = 0;
    unsigned bits = 0;
    for (const char c : encoded) {
        const int value = base64_value(c);
        if (value < 0) return Failure::fail(FailureCode::bad_frame, "hybrid bootstrap has invalid base64url data");
        accumulator = (accumulator << 6) | static_cast<uint32_t>(value);
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back(static_cast<uint8_t>((accumulator >> bits) & 0xffu));
        }
    }
    if (bits && (accumulator & ((1u << bits) - 1u)) != 0) {
        out.clear();
        return Failure::fail(FailureCode::bad_frame, "hybrid bootstrap has non-canonical base64url data");
    }
    return ok_failure();
}

inline void encode_base64url(std::span<const uint8_t> bytes, std::string& out) {
    out.clear();
    out.reserve((bytes.size() * 4 + 2) / 3);
    uint32_t accumulator = 0;
    unsigned bits = 0;
    for (const uint8_t byte : bytes) {
        accumulator = (accumulator << 8) | byte;
        bits += 8;
        while (bits >= 6) {
            bits -= 6;
            out.push_back(kBase64Alphabet[(accumulator >> bits) & 0x3fu]);
        }
    }
    if (bits) out.push_back(kBase64Alphabet[(accumulator << (6 - bits)) & 0x3fu]);
}

inline int hex_value(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    return -1;
}

inline Failure decode_nccl_hex(std::string_view hex, std::array<uint8_t, kNcclUniqueIdBytes>& out) {
    if (hex.size() != kNcclUniqueIdBytes * 2) {
        return Failure::fail(FailureCode::bad_extent, "hybrid bootstrap NCCL id must contain 128 bytes");
    }
    for (size_t i = 0; i < out.size(); ++i) {
        const int hi = hex_value(hex[i * 2]);
        const int lo = hex_value(hex[i * 2 + 1]);
        if (hi < 0 || lo < 0) {
            return Failure::fail(FailureCode::bad_frame, "hybrid bootstrap NCCL id is not lowercase hex");
        }
        out[i] = static_cast<uint8_t>((hi << 4) | lo);
    }
    return ok_failure();
}

inline std::string encode_nccl_hex(std::span<const uint8_t> bytes) {
    constexpr char digits[] = "0123456789abcdef";
    std::string out;
    out.reserve(bytes.size() * 2);
    for (const uint8_t byte : bytes) {
        out.push_back(digits[byte >> 4]);
        out.push_back(digits[byte & 0x0f]);
    }
    return out;
}

inline void append_u16(std::vector<uint8_t>& out, uint16_t value) {
    out.push_back(static_cast<uint8_t>(value >> 8));
    out.push_back(static_cast<uint8_t>(value & 0xffu));
}

inline uint16_t read_u16(const uint8_t* bytes) {
    return static_cast<uint16_t>((static_cast<uint16_t>(bytes[0]) << 8) | bytes[1]);
}

}  // namespace hybrid_detail

// Encodes the legacy join field without changing the control protocol schema.
inline Failure encode_hybrid_bootstrap(const HybridBootstrap& bundle, std::string& out) {
    out.clear();
    if (!bundle.nccl_id_hex.empty() &&
        bundle.nccl_id_hex.size() != hybrid_detail::kNcclUniqueIdBytes * 2) {
        return Failure::fail(FailureCode::bad_extent, "hybrid bootstrap NCCL id has the wrong size");
    }
    std::array<uint8_t, hybrid_detail::kNcclUniqueIdBytes> nccl{};
    if (!bundle.nccl_id_hex.empty()) {
        if (auto f = hybrid_detail::decode_nccl_hex(bundle.nccl_id_hex, nccl)) return f;
    }
    if (bundle.roce_blob.empty() || bundle.roce_blob.size() > UINT16_MAX) {
        return Failure::fail(FailureCode::bad_extent, "hybrid bootstrap RoCE blob has an invalid size");
    }
    std::vector<uint8_t> payload;
    payload.reserve(1 + nccl.size() + 2 + bundle.roce_blob.size());
    payload.push_back(bundle.nccl_id_hex.empty() ? 0 : 1);
    payload.insert(payload.end(), nccl.begin(), nccl.end());
    hybrid_detail::append_u16(payload, static_cast<uint16_t>(bundle.roce_blob.size()));
    payload.insert(payload.end(), bundle.roce_blob.begin(), bundle.roce_blob.end());
    std::string encoded;
    hybrid_detail::encode_base64url(payload, encoded);
    if (hybrid_detail::kHybridBundlePrefix.size() + encoded.size() > kHybridBootstrapFieldMax) {
        return Failure::fail(FailureCode::oversized_frame, "hybrid bootstrap bundle exceeds the 4096-byte join limit");
    }
    out.assign(hybrid_detail::kHybridBundlePrefix);
    out += encoded;
    return ok_failure();
}

inline Failure decode_hybrid_bootstrap(std::string_view encoded, HybridBootstrap& out) {
    out = {};
    if (encoded.size() > kHybridBootstrapFieldMax) {
        return Failure::fail(FailureCode::oversized_frame, "hybrid bootstrap bundle exceeds the 4096-byte join limit");
    }
    if (!encoded.starts_with(hybrid_detail::kHybridBundlePrefix)) {
        return Failure::fail(FailureCode::protocol_version, "hybrid bootstrap bundle version is unsupported");
    }
    std::vector<uint8_t> payload;
    if (auto f = hybrid_detail::decode_base64url(encoded.substr(hybrid_detail::kHybridBundlePrefix.size()), payload)) return f;
    if (payload.size() < 1 + hybrid_detail::kNcclUniqueIdBytes + 2) {
        return Failure::fail(FailureCode::truncated_frame, "hybrid bootstrap bundle is truncated");
    }
    if (payload[0] > 1) return Failure::fail(FailureCode::bad_frame, "hybrid bootstrap NCCL presence flag is invalid");
    const uint16_t blob_bytes = hybrid_detail::read_u16(payload.data() + 1 + hybrid_detail::kNcclUniqueIdBytes);
    if (blob_bytes == 0 || payload.size() != 1 + hybrid_detail::kNcclUniqueIdBytes + 2u + blob_bytes) {
        return Failure::fail(FailureCode::bad_extent, "hybrid bootstrap RoCE blob size does not match its payload");
    }
    if (payload[0]) out.nccl_id_hex = hybrid_detail::encode_nccl_hex(
        std::span<const uint8_t>(payload.data() + 1, hybrid_detail::kNcclUniqueIdBytes));
    else if (std::any_of(payload.begin() + 1, payload.begin() + 1 + hybrid_detail::kNcclUniqueIdBytes,
                         [](uint8_t byte) { return byte != 0; })) {
        return Failure::fail(FailureCode::bad_frame, "hybrid bootstrap has NCCL bytes without the presence flag");
    }
    const auto blob_begin = payload.begin() + 1 + hybrid_detail::kNcclUniqueIdBytes + 2;
    out.roce_blob.assign(blob_begin, payload.end());
    return ok_failure();
}

// All ranks carry their own RoCE blob. Only rank 0 can create the NCCL unique id before joining.
inline Failure decode_hybrid_bootstrap_roster(std::span<const std::string> rank_fields,
                                              std::string& nccl_id_hex,
                                              std::vector<uint8_t>& rank_ordered_roce_blobs) {
    nccl_id_hex.clear();
    rank_ordered_roce_blobs.clear();
    if (rank_fields.size() != kTpWorld) {
        return Failure::fail(FailureCode::bad_extent, "hybrid bootstrap roster must contain exactly three ranks");
    }
    std::array<HybridBootstrap, kTpWorld> decoded;
    for (size_t rank = 0; rank < rank_fields.size(); ++rank) {
        if (auto f = decode_hybrid_bootstrap(rank_fields[rank], decoded[rank])) return f;
    }
    if (decoded[0].nccl_id_hex.empty()) {
        return Failure::fail(FailureCode::bad_frame, "hybrid bootstrap rank zero did not provide the NCCL id");
    }
    nccl_id_hex = decoded[0].nccl_id_hex;
    const size_t blob_bytes = decoded[0].roce_blob.size();
    if (blob_bytes == 0) return Failure::fail(FailureCode::bad_extent, "hybrid bootstrap RoCE blob is empty");
    rank_ordered_roce_blobs.reserve(blob_bytes * kTpWorld);
    for (size_t rank = 0; rank < decoded.size(); ++rank) {
        if (rank != 0 && !decoded[rank].nccl_id_hex.empty()) {
            rank_ordered_roce_blobs.clear();
            return Failure::fail(FailureCode::bad_frame, "only rank zero may provide the NCCL bootstrap id");
        }
        if (decoded[rank].roce_blob.size() != blob_bytes) {
            rank_ordered_roce_blobs.clear();
            return Failure::fail(FailureCode::bad_extent, "hybrid bootstrap RoCE blob sizes do not match");
        }
        rank_ordered_roce_blobs.insert(rank_ordered_roce_blobs.end(), decoded[rank].roce_blob.begin(),
                                       decoded[rank].roce_blob.end());
    }
    return ok_failure();
}

inline bool hybrid_roce_gather_eligible(ScalarType type, uint64_t slot_elements,
                                        uint64_t maximum_bytes = kHybridRoceGatherMaxBytes) {
    if (type != ScalarType::fp32 && type != ScalarType::bf16) return false;
    uint64_t bytes = 0;
    return element_bytes(slot_elements, type, bytes) && bytes != 0 && bytes <= maximum_bytes && bytes % 16 == 0;
}

// The shape decides the route identically on every rank. An eligible but misaligned call is rejected;
// changing only the local pointer must never send ranks through different collectives.
class HybridBackend final : public Backend {
public:
    HybridBackend(Backend& nccl, Backend& roce) : nccl_(nccl), roce_(roce) {}

    uint32_t world_size() const override { return nccl_.world_size(); }
    uint32_t local_rank() const override { return nccl_.local_rank(); }

    Failure raw_all_reduce_sum(const DeviceBuffer& input, const DeviceBuffer& output,
                               uint64_t elements, cudaStream_t stream) override {
        return enqueue(nccl_, input, output, elements, stream, false);
    }

    Failure raw_all_gather(const DeviceBuffer& input, const DeviceBuffer& output,
                           uint64_t slot_elements, cudaStream_t stream) override {
        if (terminal_) return terminal_;
        if (in_flight_) return fail_terminal(Failure::fail(FailureCode::in_flight,
                                                            "hybrid transport already has an operation in flight"));
        if (!hybrid_roce_gather_eligible(input.type, slot_elements)) {
            return enqueue(nccl_, input, output, slot_elements, stream, true);
        }
        if (!aligned16(input.data) || !aligned16(output.data)) {
            return fail_terminal(Failure::fail(FailureCode::bad_pointer,
                                               "RoCE-eligible gather buffers must be 16-byte aligned"));
        }
        return enqueue(roce_, input, output, slot_elements, stream, true);
    }

    Failure raw_poll(Deadline deadline) override {
        if (terminal_) return terminal_;
        if (!in_flight_ || active_ == nullptr) {
            return fail_terminal(Failure::fail(FailureCode::not_ready, "hybrid transport has no in-flight operation"));
        }
        const Failure result = active_->raw_poll(deadline);
        if (result.code == FailureCode::deadline_expired) return result;
        in_flight_ = false;
        active_ = nullptr;
        return result ? fail_terminal(result) : ok_failure();
    }

    bool raw_async_failure(Failure& out) const override {
        if (terminal_) {
            out = terminal_;
            return true;
        }
        Failure nccl_failure;
        Failure roce_failure;
        const bool nccl_failed = nccl_.raw_async_failure(nccl_failure);
        const bool roce_failed = roce_.raw_async_failure(roce_failure);
        if (nccl_failed || roce_failed) {
            out = nccl_failed ? std::move(nccl_failure) : std::move(roce_failure);
            if (!out) out = Failure::fail(FailureCode::async_error, "a hybrid transport backend failed asynchronously");
            const_cast<HybridBackend*>(this)->fail_terminal(out);
            return true;
        }
        return false;
    }

    void raw_abort() noexcept override {
        if (aborted_) return;
        aborted_ = true;
        in_flight_ = false;
        active_ = nullptr;
        nccl_.raw_abort();
        roce_.raw_abort();
    }

private:
    static bool aligned16(const void* pointer) {
        return pointer != nullptr && reinterpret_cast<uintptr_t>(pointer) % 16 == 0;
    }

    Failure enqueue(Backend& backend, const DeviceBuffer& input, const DeviceBuffer& output,
                    uint64_t elements, cudaStream_t stream, bool gather) {
        if (terminal_) return terminal_;
        if (in_flight_) return fail_terminal(Failure::fail(FailureCode::in_flight,
                                                            "hybrid transport already has an operation in flight"));
        in_flight_ = true;
        active_ = &backend;
        const Failure result = gather ? backend.raw_all_gather(input, output, elements, stream)
                                      : backend.raw_all_reduce_sum(input, output, elements, stream);
        if (result) {
            in_flight_ = false;
            active_ = nullptr;
            return fail_terminal(result);
        }
        return ok_failure();
    }

    Failure fail_terminal(Failure failure) {
        if (!terminal_) terminal_ = std::move(failure);
        raw_abort();
        return terminal_;
    }

    Backend& nccl_;
    Backend& roce_;
    Backend* active_ = nullptr;
    Failure terminal_{};
    bool in_flight_ = false;
    bool aborted_ = false;
};

}  // namespace strata::distributed
