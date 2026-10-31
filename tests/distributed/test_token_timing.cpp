#include "strata/distributed/token_timing.hpp"

#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>

int main() {
    using namespace strata::distributed;
    int failures = 0;
    auto check = [&](bool result, const char* label) {
        if (!result) { ++failures; std::cerr << "FAIL: " << label << '\n'; }
    };
    const auto root = std::filesystem::temp_directory_path() /
        ("strata-token-timing-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directory(root);
    const auto path = root / "nested" / "timing.csv";
    {
        TokenTimingWriter writer(path);
        TokenRowTimer off(false);
        off.forward_done(); off.readback_done();
        const auto zero = off.finish();
        check(!zero.row_wall_ns && !zero.model_forward_ns && !zero.host_readback_ns && !zero.artifact_write_ns,
              "disabled timer returns zeros");
        TokenRowTimer on(true);
        on.forward_done(); on.readback_done();
        const auto measured = on.finish();
        check(measured.row_wall_ns >= measured.model_forward_ns && measured.row_wall_ns >= measured.host_readback_ns &&
              measured.row_wall_ns >= measured.artifact_write_ns, "row total bounds each monotonic interval");
        const auto maximum = std::numeric_limits<uint64_t>::max();
        writer.write("request-0", maximum, 42, "teacher-forced", {maximum, maximum, maximum, maximum});
        writer.flush();
        std::ifstream file(path);
        std::ostringstream text; text << file.rdbuf();
        check(text.str() ==
              "request_id,position,input_id,phase,row_wall_ns,model_forward_ns,host_readback_ns,artifact_write_ns\n"
              "request-0,18446744073709551615,42,teacher-forced,18446744073709551615,18446744073709551615,18446744073709551615,18446744073709551615\n",
              "CSV header and uint64 counters round trip without narrowing");
        bool rejected = false;
        try { writer.write("comma,injection", 0, 1, "teacher-forced", {}); }
        catch (const std::runtime_error&) { rejected = true; }
        check(rejected, "invalid CSV identity rejected");
        writer.close();
        rejected = false;
        try { writer.write("request-0", 0, 1, "teacher-forced", {}); }
        catch (const std::runtime_error&) { rejected = true; }
        check(rejected, "write after close rejected");
    }
    bool rejected = false;
    try { TokenTimingWriter existing(path); }
    catch (const std::runtime_error&) { rejected = true; }
    check(rejected, "existing evidence is not overwritten");
    rejected = false;
    try { TokenTimingWriter relative("relative-timing.csv"); }
    catch (const std::runtime_error&) { rejected = true; }
    check(rejected, "relative path rejected");
    const auto link = root / "link.csv";
    std::filesystem::create_symlink(path, link);
    rejected = false;
    try { TokenTimingWriter symlink(link); }
    catch (const std::runtime_error&) { rejected = true; }
    check(rejected, "symlink is not followed for writing");
    std::filesystem::remove_all(root);
    std::cout << "token_timing: " << failures << " failures\n";
    return failures ? 1 : 0;
}
