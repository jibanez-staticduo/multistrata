#pragma once

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>

namespace strata::distributed {

struct TokenTiming {
    uint64_t row_wall_ns = 0;
    uint64_t model_forward_ns = 0;
    uint64_t host_readback_ns = 0;
    uint64_t artifact_write_ns = 0;
};

// Marks bound completed forward, host readback/validation, and artifact writing, in that order.
// A disabled timer never queries the clock. CSV/stdout publication is outside the measured row.
// Only artifacts captured inside that row count; reset and terminal state snapshots are excluded.
class TokenRowTimer final {
public:
    explicit TokenRowTimer(bool enabled) : enabled_(enabled) {
        if (enabled_) start_ = last_ = Clock::now();
    }
    void forward_done() { if (enabled_) timing_.model_forward_ns = elapsed(); }
    void readback_done() { if (enabled_) timing_.host_readback_ns = elapsed(); }
    TokenTiming finish() {
        if (enabled_) {
            timing_.artifact_write_ns = elapsed();
            timing_.row_wall_ns = nanoseconds(last_ - start_);
        }
        return timing_;
    }
private:
    using Clock = std::chrono::steady_clock;
    static uint64_t nanoseconds(Clock::duration value) {
        return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(value).count());
    }
    uint64_t elapsed() {
        const auto now = Clock::now();
        const auto value = nanoseconds(now - last_);
        last_ = now;
        return value;
    }
    bool enabled_;
    Clock::time_point start_{}, last_{};
    TokenTiming timing_;
};

class TokenTimingWriter final {
public:
    explicit TokenTimingWriter(const std::filesystem::path& path) {
        if (!path.is_absolute()) throw std::runtime_error("--timing-file must be an absolute fresh path");
        std::filesystem::create_directories(path.parent_path());
        // Exclusive creation also rejects existing symlinks and prevents truncating earlier evidence.
        file_ = std::fopen(path.string().c_str(), "wx");
        if (!file_) throw std::runtime_error("cannot create fresh timing CSV");
        if (std::fputs("request_id,position,input_id,phase,row_wall_ns,model_forward_ns,host_readback_ns,artifact_write_ns\n", file_) < 0 ||
            std::fflush(file_) != 0) {
            std::fclose(file_); file_ = nullptr;
            throw std::runtime_error("cannot write timing CSV header");
        }
    }
    ~TokenTimingWriter() { if (file_) std::fclose(file_); }
    TokenTimingWriter(const TokenTimingWriter&) = delete;
    TokenTimingWriter& operator=(const TokenTimingWriter&) = delete;

    void write(std::string_view request, uint64_t position, int32_t input, std::string_view phase,
               const TokenTiming& timing) {
        auto safe = [](std::string_view value) {
            return !value.empty() && value.size() <= 128 && std::all_of(value.begin(), value.end(), [](char c) {
                return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' || c == '-';
            });
        };
        if (!file_ || !safe(request) || !safe(phase) || input < 0)
            throw std::runtime_error("invalid timing CSV row");
        const auto row = std::string(request) + ',' + std::to_string(position) + ',' + std::to_string(input) + ',' +
            std::string(phase) + ',' + std::to_string(timing.row_wall_ns) + ',' + std::to_string(timing.model_forward_ns) + ',' +
            std::to_string(timing.host_readback_ns) + ',' + std::to_string(timing.artifact_write_ns) + '\n';
        if (std::fwrite(row.data(), 1, row.size(), file_) != row.size())
            throw std::runtime_error("cannot write timing CSV row");
        if (++pending_ == 256) flush();
    }
    void flush() {
        if (!file_ || std::fflush(file_) != 0) throw std::runtime_error("cannot flush timing CSV");
        pending_ = 0;
    }
    void close() {
        if (!file_) return;
        auto* file = file_; file_ = nullptr;
        if (std::fclose(file) != 0) throw std::runtime_error("cannot close timing CSV");
    }
private:
    std::FILE* file_ = nullptr;
    uint32_t pending_ = 0;
};

} // namespace strata::distributed
