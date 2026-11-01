#include "strata/distributed/fast_config.hpp"
#include <charconv>
#include <map>
#include <set>
#include <stdexcept>

namespace strata::distributed {
FastLaunchConfig parse_fast_launch(int argc, char** argv) {
    const std::set<std::string> allowed{"mode", "rank", "rank0-address", "port", "session-uuid",
        "library-path", "nccl-version", "library-provenance", "model-hash", "config-fingerprint",
        "pack", "native", "embd", "ple", "context", "timeout-ms", "parallel", "prefill-chunk", "mtp-path", "mtp-drafts"};
    std::map<std::string, std::string> args;
    for (int i = 1; i < argc; i += 2) {
        if (i + 1 >= argc || std::string_view(argv[i]).substr(0, 2) != "--") throw std::invalid_argument("each fast option needs a value");
        const std::string key = argv[i] + 2;
        if (!allowed.contains(key) || !args.emplace(key, argv[i + 1]).second) throw std::invalid_argument("unknown or duplicate fast option");
    }
    auto get = [&](const char* key) -> const std::string& {
        auto it = args.find(key);
        if (it == args.end() || it->second.empty() || it->second.find_first_of("\r\n") != std::string::npos)
            throw std::invalid_argument(std::string("missing/invalid fast option: ") + key);
        return it->second;
    };
    auto number = [&](const char* key, uint32_t minimum, uint32_t maximum) {
        const auto& value = get(key);
        uint32_t output = 0;
        const auto parsed = std::from_chars(value.data(), value.data() + value.size(), output);
        if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() || output < minimum || output > maximum)
            throw std::invalid_argument(std::string("invalid fast numeric option: ") + key);
        return output;
    };
    FastLaunchConfig out;
    if (get("mode") != "multislot") throw std::invalid_argument("production NVFP4 uses the multislot protocol");
    auto& r = out.runtime;
    r.rank = number("rank", 0, 2); r.rank0_address = get("rank0-address");
    r.port = number("port", 1024, 65535); r.timeout_ms = number("timeout-ms", 100, 600000);
    r.context = number("context", 4, 262144); r.slots = number("parallel", 1, 8);
    if (r.slots != 1 && r.slots != 4 && r.slots != 8) throw std::invalid_argument("parallel must be 1, 4 or 8");
    r.prompt_chunk = number("prefill-chunk", 64, 8192);
    r.session_uuid = get("session-uuid"); r.model_hash = get("model-hash"); r.config_hash = get("config-fingerprint");
    r.nccl_library = get("library-path"); r.nccl_version = get("nccl-version"); r.nccl_provenance = get("library-provenance");
    out.pack = get("pack"); out.dense = get("native"); out.embedding = get("embd"); out.ple = get("ple");
    if (args.contains("mtp-path") != args.contains("mtp-drafts")) throw std::invalid_argument("MTP path and drafts must be configured together");
    if (args.contains("mtp-path")) { out.mtp = get("mtp-path"); out.drafts = number("mtp-drafts", 1, 4); }
    r.drafts=out.drafts;r.mtp_path=out.mtp;
    for (const auto* path : {&out.pack, &out.dense, &out.embedding, &out.ple, &r.nccl_library})
        if (path->empty() || path->front() != '/') throw std::invalid_argument("fast model paths must be absolute");
    // The execution policy is agreed in the roster, not inferred independently
    // on each host from what happens to be available at launch.
    r.config_hash += ":context=" + std::to_string(r.context) + ":slots=" + std::to_string(r.slots) +
                     ":chunk=" + std::to_string(r.prompt_chunk) + ":drafts=" + std::to_string(out.drafts);
    return out;
}
}
