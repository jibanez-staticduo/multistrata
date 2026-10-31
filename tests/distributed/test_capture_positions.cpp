#include "strata/distributed/capture_positions.hpp"

#include <cstdio>
#include <limits>
#include <string>

namespace {
using strata::distributed::CapturePositions;
bool rejects(std::string_view text, uint64_t context = 65536) {
    try { (void)CapturePositions::parse(text, context); }
    catch (const std::runtime_error&) { return true; }
    return false;
}
bool check(bool ok, const char* name) {
    if (!ok) std::fprintf(stderr, "capture positions failed: %s\n", name);
    return ok;
}
}

int main() {
    bool ok = true;
    const CapturePositions all;
    ok &= check(all.contains(0) && all.contains(65535) && all.contains(1, true), "default all");
    const auto sparse = CapturePositions::parse("0,8191,65535", 65536);
    ok &= check(sparse.contains(0) && sparse.contains(8191) && sparse.contains(65535) &&
                !sparse.contains(1) && !sparse.contains(2, true), "sparse selection");
    const auto last = CapturePositions::parse("last", 65536);
    ok &= check(!last.contains(0) && last.contains(0, true) && !last.contains(8191) &&
                last.contains(8191, true), "last per GEN segment");
    const auto mixed = CapturePositions::parse("3,last,0", 4);
    ok &= check(mixed.contains(3) && mixed.contains(0) && !mixed.contains(1) && mixed.contains(1, true),
                "mixed selection");
    for (const uint64_t length : {8192, 65536}) {
        const auto endpoints = CapturePositions::parse("0,last", length);
        uint64_t consumed = 0, captured = 0;
        for (uint64_t position = 0; position < length; ++position) {
            ++consumed;
            captured += endpoints.contains(position, position + 1 == length);
        }
        ok &= check(consumed == length && captured == 2, "long prompt consumes all rows and captures endpoints");
        ok &= check(captured * 248320 * sizeof(float) == 1986560, "bounded full-vocabulary artifact bytes");
    }
    for (const auto text : {"", "all", ",", ",0", "0,", "0,,1", "-1", "+1", " 1", "1 ", "1x",
                            "0,0", "0,00", "last,last", "LAST", "65536", "18446744073709551616"})
        ok &= check(rejects(text), text);
    ok &= check(rejects("0", 0), "empty context");
    ok &= check(rejects("18446744073709551615", std::numeric_limits<uint64_t>::max()),
                "position must be below context");
    const auto large = CapturePositions::parse("18446744073709551614", std::numeric_limits<uint64_t>::max());
    ok &= check(large.contains(std::numeric_limits<uint64_t>::max() - 1), "uint64 positions");
    std::string many;
    for (size_t i = 0; i < CapturePositions::max_entries; ++i) {
        if (i) many += ',';
        many += std::to_string(i);
    }
    ok &= check(CapturePositions::parse(many, 65536).contains(4095), "maximum entry count");
    ok &= check(rejects(many + ",last"), "too many entries");
    if (!ok) return 1;
    std::puts("capture positions: default all, sparse, segment-last and bounded strict parsing passed");
}
