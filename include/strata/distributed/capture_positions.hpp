#pragma once

#include <charconv>
#include <cstddef>
#include <cstdint>
#include <set>
#include <stdexcept>
#include <string_view>

namespace strata::distributed {

// The default captures every row. Explicit positions are zero-based within a
// request; "last" captures the final token of each GEN command, before RESET.
class CapturePositions final {
public:
    static constexpr size_t max_entries = 4096;

    static CapturePositions parse(std::string_view text, uint64_t context) {
        CapturePositions result;
        result.all_ = false;
        size_t entries = 0;
        for (;;) {
            const auto comma = text.find(',');
            const auto word = text.substr(0, comma);
            if (word.empty() || ++entries > max_entries) invalid();
            if (word == "last") {
                if (result.last_) invalid();
                result.last_ = true;
            } else {
                uint64_t value = 0;
                const auto [end, status] = std::from_chars(word.data(), word.data() + word.size(), value);
                if (status != std::errc{} || end != word.data() + word.size() || value >= context ||
                    !result.positions_.insert(value).second) invalid();
            }
            if (comma == std::string_view::npos) break;
            text.remove_prefix(comma + 1);
        }
        return result;
    }

    bool contains(uint64_t position, bool is_segment_last = false) const {
        return all_ || positions_.contains(position) || (last_ && is_segment_last);
    }

private:
    [[noreturn]] static void invalid() {
        throw std::runtime_error("invalid --logits-positions: expected unique positions within context or last (at most 4096 entries)");
    }
    bool all_ = true;
    bool last_ = false;
    std::set<uint64_t> positions_;
};

} // namespace strata::distributed
