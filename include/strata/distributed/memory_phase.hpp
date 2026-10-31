// Small CPU-only JSONL writer for explicitly timed per-rank memory snapshots.
#pragma once

#include <charconv>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <set>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

#include <fcntl.h>
#include <unistd.h>

namespace strata::distributed {

enum class MemoryPhase : uint8_t {
    allocation,
    warmup,
    prefill_start,
    prefill_complete,
    first_decode,
    decode,
    reset,
    graph_capture,
};

inline const char* memory_phase_name(MemoryPhase phase) {
    switch (phase) {
        case MemoryPhase::allocation: return "allocation";
        case MemoryPhase::warmup: return "warmup";
        case MemoryPhase::prefill_start: return "prefill_start";
        case MemoryPhase::prefill_complete: return "prefill_complete";
        case MemoryPhase::first_decode: return "first_decode";
        case MemoryPhase::decode: return "decode";
        case MemoryPhase::reset: return "reset";
        case MemoryPhase::graph_capture: return "graph_capture";
    }
    return nullptr;
}

struct MemoryPhaseSnapshot {
    std::string timestamp_utc;  // ISO-8601 UTC with a trailing Z, supplied by the caller
    uint32_t rank = 0;
    std::string request;        // opaque, non-secret identifier; never pass prompts or credentials
    uint64_t step = 0;
    MemoryPhase phase = MemoryPhase::allocation;
    uint64_t position = 0;
    uint64_t cuda_free_bytes = 0;
    uint64_t cuda_total_bytes = 0;
};

class MemoryPhaseWriter {
 public:
    static constexpr size_t kMaxRecordBytes = 1024;
    static constexpr size_t kMaxRequestBytes = 128;
    static constexpr size_t kMaxSnapshots = 4096;

    MemoryPhaseWriter() = default;
    MemoryPhaseWriter(const MemoryPhaseWriter&) = delete;
    MemoryPhaseWriter& operator=(const MemoryPhaseWriter&) = delete;

    ~MemoryPhaseWriter() { close(); }

    // The path must not exist. This prevents a new run from silently appending
    // to stale evidence or truncating a prior run.
    bool open_new(std::string_view path, std::string& error) {
        error.clear();
        if (fd_ >= 0) return set_error(error, "memory phase writer is already open");
        if (path.empty() || path.size() > 4096 || path.find('\0') != std::string_view::npos)
            return set_error(error, "memory phase output path is invalid");
        const std::string owned_path(path);
        fd_ = ::open(owned_path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_APPEND | O_CLOEXEC, 0600);
        if (fd_ < 0) return set_errno_error(error, "cannot create fresh memory phase output");
        return true;
    }

    // One writer owns one per-rank file. Duplicate request/phase pairs and
    // excessive snapshots are refused before a line reaches disk.
    bool append(const MemoryPhaseSnapshot& snapshot, std::string& error) {
        error.clear();
        if (fd_ < 0) return set_error(error, "memory phase writer is not open");
        if (!valid(snapshot, error)) return false;

        const auto key = std::make_pair(snapshot.request, snapshot.phase);
        if (seen_.contains(key)) return set_error(error, "memory phase already recorded for this request");
        if (seen_.size() >= kMaxSnapshots) return set_error(error, "memory phase snapshot limit reached");

        std::string record;
        record.reserve(384);
        record += "{\"schema\":1,\"timestamp_utc\":\"";
        record += snapshot.timestamp_utc;
        record += "\",\"rank\":";
        append_number(record, snapshot.rank);
        record += ",\"request\":\"";
        append_json_string(record, snapshot.request);
        record += "\",\"step\":";
        append_number(record, snapshot.step);
        record += ",\"phase\":\"";
        record += memory_phase_name(snapshot.phase);
        record += "\",\"position\":";
        append_number(record, snapshot.position);
        record += ",\"cuda_free_bytes\":";
        append_number(record, snapshot.cuda_free_bytes);
        record += ",\"cuda_total_bytes\":";
        append_number(record, snapshot.cuda_total_bytes);
        record += ",\"memory_source\":\"cudaMemGetInfo\",";
        record += "\"memory_scope\":\"device_allocator_availability_not_process_allocation\",";
        record += "\"unified_memory_note\":\"may_overlap_host_accounting\"}\n";
        if (record.size() > kMaxRecordBytes) return set_error(error, "memory phase JSON record exceeds its bound");

        // The per-rank file has one writer. Roll back a partial append so a
        // failed event cannot leave a truncated JSONL row.
        const off_t original_size = ::lseek(fd_, 0, SEEK_END);
        if (original_size < 0) return set_errno_error(error, "cannot locate end of memory phase output");
        size_t offset = 0;
        while (offset < record.size()) {
            const ssize_t written = ::write(fd_, record.data() + offset, record.size() - offset);
            if (written < 0 && errno == EINTR) continue;
            if (written <= 0) {
                const int saved_errno = written < 0 ? errno : EIO;
                if (::ftruncate(fd_, original_size) != 0) close();
                return set_errno_error(error, "cannot append memory phase record", saved_errno);
            }
            offset += static_cast<size_t>(written);
        }
        seen_.insert(key);
        return true;
    }

    void close() noexcept {
        if (fd_ >= 0) {
            ::close(fd_);
            fd_ = -1;
        }
    }

 private:
    int fd_ = -1;
    std::set<std::pair<std::string, MemoryPhase>> seen_;

    static bool set_error(std::string& error, const char* message) {
        error = message;
        return false;
    }

    static bool set_errno_error(std::string& error, const char* prefix, int saved_errno = errno) {
        error = prefix;
        error += ": ";
        error += std::strerror(saved_errno);
        return false;
    }

    static bool valid_timestamp(std::string_view value) {
        // YYYY-MM-DDTHH:MM:SS[.fraction]Z; timezone offsets are rejected.
        if (value.size() < 20 || value[4] != '-' || value[7] != '-' || value[10] != 'T' ||
            value[13] != ':' || value[16] != ':')
            return false;
        size_t suffix = 19;
        if (value[suffix] == '.') {
            ++suffix;
            const size_t digits_start = suffix;
            while (suffix < value.size() && value[suffix] >= '0' && value[suffix] <= '9') ++suffix;
            const size_t digits = suffix - digits_start;
            if (digits == 0 || digits > 9) return false;
        }
        if (suffix + 1 != value.size() || value[suffix] != 'Z') return false;

        auto number = [&](size_t start, size_t count, int& out) {
            out = 0;
            for (size_t i = 0; i < count; ++i) {
                const char c = value[start + i];
                if (c < '0' || c > '9') return false;
                out = out * 10 + (c - '0');
            }
            return true;
        };
        int year_value = 0, month_value = 0, day_value = 0;
        int hour = 0, minute = 0, second = 0;
        if (!number(0, 4, year_value) || !number(5, 2, month_value) || !number(8, 2, day_value) ||
            !number(11, 2, hour) || !number(14, 2, minute) || !number(17, 2, second))
            return false;
        using namespace std::chrono;
        const year_month_day date{year{year_value}, month{static_cast<unsigned>(month_value)},
                                  day{static_cast<unsigned>(day_value)}};
        return date.ok() && hour < 24 && minute < 60 && second < 60;
    }

    static bool valid_utf8(std::string_view value) {
        size_t i = 0;
        while (i < value.size()) {
            const auto first = static_cast<unsigned char>(value[i]);
            if (first < 0x80) { ++i; continue; }
            size_t count = 0;
            uint32_t code = 0;
            if ((first & 0xe0) == 0xc0) { count = 2; code = first & 0x1f; }
            else if ((first & 0xf0) == 0xe0) { count = 3; code = first & 0x0f; }
            else if ((first & 0xf8) == 0xf0) { count = 4; code = first & 0x07; }
            else return false;
            if (i + count > value.size()) return false;
            for (size_t j = 1; j < count; ++j) {
                const auto next = static_cast<unsigned char>(value[i + j]);
                if ((next & 0xc0) != 0x80) return false;
                code = (code << 6) | (next & 0x3f);
            }
            if ((count == 2 && code < 0x80) || (count == 3 && code < 0x800) ||
                (count == 4 && code < 0x10000) || code > 0x10ffff ||
                (code >= 0xd800 && code <= 0xdfff))
                return false;
            i += count;
        }
        return true;
    }

    static void append_number(std::string& out, uint64_t value) {
        char buffer[32];
        const auto result = std::to_chars(buffer, buffer + sizeof(buffer), value);
        out.append(buffer, result.ptr);
    }

    static void append_json_string(std::string& out, std::string_view value) {
        static constexpr char hex[] = "0123456789abcdef";
        for (const unsigned char c : value) {
            switch (c) {
                case '"': out += "\\\""; break;
                case '\\': out += "\\\\"; break;
                case '\b': out += "\\b"; break;
                case '\f': out += "\\f"; break;
                case '\n': out += "\\n"; break;
                case '\r': out += "\\r"; break;
                case '\t': out += "\\t"; break;
                default:
                    if (c < 0x20) {
                        out += "\\u00";
                        out += hex[c >> 4];
                        out += hex[c & 0x0f];
                    } else {
                        out += static_cast<char>(c);
                    }
            }
        }
    }

    static bool valid(const MemoryPhaseSnapshot& snapshot, std::string& error) {
        if (snapshot.rank > 2) return set_error(error, "memory phase rank must be 0, 1 or 2");
        if (!memory_phase_name(snapshot.phase)) return set_error(error, "memory phase is unknown");
        if (!valid_timestamp(snapshot.timestamp_utc))
            return set_error(error, "memory phase timestamp must be a valid ISO-8601 UTC value ending in Z");
        if (snapshot.request.empty() || snapshot.request.size() > kMaxRequestBytes ||
            !valid_utf8(snapshot.request))
            return set_error(error, "memory phase request identifier is empty, too long or invalid UTF-8");
        if (snapshot.cuda_total_bytes == 0 || snapshot.cuda_free_bytes > snapshot.cuda_total_bytes)
            return set_error(error, "CUDA free/total memory values are invalid");
        return true;
    }
};

}  // namespace strata::distributed
