#include "strata/distributed/limits.hpp"
#include "strata/distributed/serve_protocol.hpp"

#include <charconv>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace strata::distributed {
namespace {

constexpr size_t kMaxLine = 1u << 20;

struct Value {
    enum class Kind { object, array, string, number, literal } kind = Kind::literal;
    std::string text;
    std::map<std::string, Value> object;
    std::vector<Value> array;
};

class Reader {
 public:
    explicit Reader(std::string_view input) : input_(input) {}
    bool parse(Value& value) {
        if (input_.size() > kMaxLine) return fail("JSON exceeds 1 MiB");
        if (!read(value, 0)) return false;
        ws();
        return pos_ == input_.size() || fail("trailing JSON data");
    }
    std::string error;

 private:
    bool fail(const char* why) { error = why; return false; }
    void ws() {
        while (pos_ < input_.size() && (input_[pos_] == ' ' || input_[pos_] == '\t' ||
               input_[pos_] == '\r' || input_[pos_] == '\n')) ++pos_;
    }
    bool take(char c) {
        ws();
        if (pos_ == input_.size() || input_[pos_] != c) return false;
        ++pos_;
        return true;
    }
    bool hex4(uint32_t& code) {
        code = 0;
        for (int i = 0; i < 4; ++i) {
            if (pos_ == input_.size()) return fail("truncated Unicode escape");
            const char c = input_[pos_++];
            const int n = c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 :
                          c >= 'A' && c <= 'F' ? c - 'A' + 10 : -1;
            if (n < 0) return fail("invalid Unicode escape");
            code = code * 16 + static_cast<uint32_t>(n);
        }
        return true;
    }
    static void utf8(std::string& out, uint32_t c) {
        if (c <= 0x7f) out += static_cast<char>(c);
        else if (c <= 0x7ff) {
            out += static_cast<char>(0xc0 | (c >> 6));
            out += static_cast<char>(0x80 | (c & 63));
        } else if (c <= 0xffff) {
            out += static_cast<char>(0xe0 | (c >> 12));
            out += static_cast<char>(0x80 | ((c >> 6) & 63));
            out += static_cast<char>(0x80 | (c & 63));
        } else {
            out += static_cast<char>(0xf0 | (c >> 18));
            out += static_cast<char>(0x80 | ((c >> 12) & 63));
            out += static_cast<char>(0x80 | ((c >> 6) & 63));
            out += static_cast<char>(0x80 | (c & 63));
        }
    }
    bool string(std::string& out) {
        if (!take('"')) return fail("expected JSON string");
        while (pos_ < input_.size()) {
            const auto c = static_cast<unsigned char>(input_[pos_++]);
            if (c == '"') return true;
            if (c < 0x20) return fail("unescaped control character");
            if (c == '\\') {
                if (pos_ == input_.size()) return fail("truncated escape");
                const char escape = input_[pos_++];
                switch (escape) {
                    case '"': case '\\': case '/': out += escape; break;
                    case 'b': out += '\b'; break;
                    case 'f': out += '\f'; break;
                    case 'n': out += '\n'; break;
                    case 'r': out += '\r'; break;
                    case 't': out += '\t'; break;
                    case 'u': {
                        uint32_t code = 0;
                        if (!hex4(code)) return false;
                        if (code >= 0xd800 && code <= 0xdbff) {
                            if (input_.substr(pos_, 2) != "\\u") return fail("unpaired Unicode surrogate");
                            pos_ += 2;
                            uint32_t low = 0;
                            if (!hex4(low)) return false;
                            if (low < 0xdc00 || low > 0xdfff) return fail("unpaired Unicode surrogate");
                            code = 0x10000 + ((code - 0xd800) << 10) + low - 0xdc00;
                        } else if (code >= 0xdc00 && code <= 0xdfff) return fail("unpaired Unicode surrogate");
                        utf8(out, code);
                        break;
                    }
                    default: return fail("invalid JSON escape");
                }
            } else if (c < 0x80) out += static_cast<char>(c);
            else {
                // Validate raw UTF-8 too: ignored formatting fields still must be valid JSON.
                const unsigned count = c >= 0xc2 && c <= 0xdf ? 1 : c >= 0xe0 && c <= 0xef ? 2 :
                                       c >= 0xf0 && c <= 0xf4 ? 3 : 0;
                if (!count || input_.size() - pos_ < count) return fail("invalid UTF-8");
                uint32_t code = c & ((1u << (6 - count)) - 1);
                for (unsigned i = 0; i < count; ++i) {
                    const auto next = static_cast<unsigned char>(input_[pos_++]);
                    if ((next & 0xc0) != 0x80) return fail("invalid UTF-8");
                    code = (code << 6) | (next & 63);
                }
                if ((count == 1 && code < 0x80) || (count == 2 && code < 0x800) ||
                    (count == 3 && code < 0x10000) || code > 0x10ffff ||
                    (code >= 0xd800 && code <= 0xdfff)) return fail("invalid UTF-8");
                utf8(out, code);
            }
        }
        return fail("unterminated JSON string");
    }
    bool digit() const { return pos_ < input_.size() && input_[pos_] >= '0' && input_[pos_] <= '9'; }
    bool number(Value& out) {
        const size_t begin = pos_;
        if (input_[pos_] == '-') ++pos_;
        if (!digit()) return fail("invalid JSON number");
        if (input_[pos_] == '0') ++pos_;
        else while (digit()) ++pos_;
        if (pos_ < input_.size() && input_[pos_] == '.') {
            ++pos_;
            if (!digit()) return fail("invalid JSON fraction");
            while (digit()) ++pos_;
        }
        if (pos_ < input_.size() && (input_[pos_] == 'e' || input_[pos_] == 'E')) {
            ++pos_;
            if (pos_ < input_.size() && (input_[pos_] == '+' || input_[pos_] == '-')) ++pos_;
            if (!digit()) return fail("invalid JSON exponent");
            while (digit()) ++pos_;
        }
        out.kind = Value::Kind::number;
        out.text = input_.substr(begin, pos_ - begin);
        double finite = 0;
        const auto parsed = std::from_chars(out.text.data(), out.text.data() + out.text.size(), finite);
        return (parsed.ec == std::errc{} && std::isfinite(finite)) || fail("JSON number out of range");
    }
    bool read(Value& out, unsigned depth) {
        ws();
        if (pos_ == input_.size()) return fail("truncated JSON");
        const char c = input_[pos_];
        if (c == '{' || c == '[') {
            if (depth >= 16) return fail("JSON nesting exceeds 16");
            ++pos_;
            out.kind = c == '{' ? Value::Kind::object : Value::Kind::array;
            if (take(c == '{' ? '}' : ']')) return true;
            do {
                std::string key;
                if (c == '{' && (!string(key) || !take(':'))) return fail("invalid JSON object member");
                Value child;
                if (!read(child, depth + 1)) return false;
                if (c == '{') {
                    if (!out.object.emplace(std::move(key), std::move(child)).second) return fail("duplicate JSON key");
                } else out.array.push_back(std::move(child));
                if (take(c == '{' ? '}' : ']')) return true;
            } while (take(','));
            return fail("invalid JSON container");
        }
        if (c == '"') { out.kind = Value::Kind::string; return string(out.text); }
        if (c == '-' || digit()) return number(out);
        for (const auto literal : {std::string_view("true"), std::string_view("false"), std::string_view("null")}) {
            if (input_.substr(pos_, literal.size()) == literal) {
                pos_ += literal.size(); out.text = literal; return true;
            }
        }
        return fail("invalid JSON value");
    }
    std::string_view input_;
    size_t pos_ = 0;
};

bool safe_id(std::string_view id) {
    if (id.empty() || id.size() > 128) return false;
    for (char c : id)
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
              (c >= '0' && c <= '9') || c == '_' || c == '-')) return false;
    return true;
}

bool integer(const Value& value, uint64_t max, uint64_t& out) {
    if (value.kind != Value::Kind::number || value.text.empty() || value.text[0] == '-') return false;
    const auto result = std::from_chars(value.text.data(), value.text.data() + value.text.size(), out);
    return result.ec == std::errc{} && result.ptr == value.text.data() + value.text.size() && out <= max;
}

bool real(const Value& value, float& out) {
    if (value.kind != Value::Kind::number) return false;
    const auto result = std::from_chars(value.text.data(), value.text.data() + value.text.size(), out);
    return result.ec == std::errc{} && result.ptr == value.text.data() + value.text.size() && std::isfinite(out);
}

bool contains(std::string_view key, std::initializer_list<std::string_view> keys) {
    for (auto candidate : keys) if (key == candidate) return true;
    return false;
}

bool sampling(const Value& value, int32_t n_vocab, ServeRequest& out, std::string& err) {
    if (value.kind != Value::Kind::object) { err = "sampling must be an object"; return false; }
    bool window = false;
    for (const auto& [key, field] : value.object) {
        float number = 0;
        uint64_t whole = 0;
        bool valid = true;
        if (key == "seed") {
            valid = integer(field, std::numeric_limits<uint64_t>::max(), whole);
            out.sampling.seed = whole;
        } else if (key == "top_k" || key == "penalty_last_n") {
            valid = integer(field, std::numeric_limits<int>::max(), whole);
            if (key == "top_k") out.sampling.top_k = whole >= 1 && whole <= 64 ? static_cast<int>(whole) : 64;
            else { out.sampling.penalty_last_n = static_cast<int>(whole); window = true; }
        } else if (contains(key, {"temperature", "top_p", "min_p", "repetition_penalty", "frequency_penalty", "presence_penalty"})) {
            valid = real(field, number);
            if (key == "temperature") { out.sampling.temperature = number; out.sampling.greedy = number <= 0; }
            else if (key == "top_p") { valid = valid && number > 0 && number <= 1; out.sampling.top_p = number; }
            else if (key == "min_p") { valid = valid && number >= 0 && number <= 1; out.sampling.min_p = number; }
            else if (key == "repetition_penalty") { valid = valid && number > 0; out.sampling.penalty_repeat = number; }
            else if (key == "frequency_penalty") out.sampling.penalty_freq = number;
            else out.sampling.penalty_present = number;
        } else if (key == "logit_bias") {
            valid = field.kind == Value::Kind::object;
            if (valid) for (const auto& [id, bias] : field.object) {
                int32_t token = -1;
                const auto parsed = std::from_chars(id.data(), id.data() + id.size(), token);
                if (parsed.ec != std::errc{} || parsed.ptr != id.data() + id.size() || token < 0 ||
                    token >= n_vocab || !real(bias, number) || !out.logit_bias.emplace(token, number).second) {
                    valid = false; break;
                }
            }
        } else if (contains(key, {"strata_tune", "strata_checkpoint", "strata_img_keys", "strata_prefix",
                                 "experimental_speed_projection", "logprobs", "top_logprobs", "allowed_token_ids"})) {
            err = "unsupported distributed engine feature: " + key;
            return false;
        } else {
            // Service owns these request/formatting fields; the native sampler never consumes them.
            valid = contains(key, {"stop", "max_tokens", "max_completion_tokens", "max_output_tokens", "stream",
                "stream_options", "model", "messages", "prompt", "tools", "tool_choice", "parallel_tool_calls",
                "response_format", "reasoning_effort", "thinking", "thinking_budget", "reasoning_budget",
                "reasoning_budget_tokens", "thinking_budget_tokens", "budget_tokens", "min_tokens", "user",
                "metadata", "store", "n", "echo", "suffix", "best_of", "service_tier", "verbosity",
                "reasoning", "text", "stop_sequences", "system", "input", "instructions", "include",
                "chat_template_kwargs", "output_config", "return_progress", "strata_mcp", "safety_identifier",
                "prompt_cache_key", "previous_response_id", "conversation", "truncation", "background",
                "repeat_penalty", "repeat_last_n"});
        }
        if (!valid) { err = "invalid or unsupported sampling field: " + key; return false; }
    }
    if (!window && (out.sampling.penalty_repeat != 1 || out.sampling.penalty_freq != 0 || out.sampling.penalty_present != 0))
        out.sampling.penalty_last_n = 64;
    return true;
}

std::string quote(std::string_view text) {
    std::string result = "\"";
    constexpr char hex[] = "0123456789abcdef";
    for (unsigned char c : text) {
        if (c == '"' || c == '\\') { result += '\\'; result += static_cast<char>(c); }
        else if (c < 0x20 || c >= 0x7f) {
            result += "\\u00"; result += hex[c >> 4]; result += hex[c & 15];
        } else result += static_cast<char>(c);
    }
    return result + '"';
}

}  // namespace

bool parse_serve_request(std::string_view json, int32_t n_vocab, uint32_t context,
                         ServeRequest& out, std::string& err) {
    err.clear();
    Value root;
    Reader reader(json);
    if (!reader.parse(root)) { err = reader.error; return false; }
    if (root.kind != Value::Kind::object) { err = "request must be an object"; return false; }
    const auto type = root.object.find("type");
    if (type == root.object.end() || type->second.kind != Value::Kind::string) { err = "missing request type"; return false; }
    ServeRequest candidate;
    candidate.sampling.temperature = 0;
    candidate.sampling.greedy = true;
    candidate.sampling.top_p = 1;
    const auto& kind = type->second.text;
    if (kind == "shutdown") candidate.kind = ServeRequest::Kind::shutdown;
    else if (kind == "cancel") candidate.kind = ServeRequest::Kind::cancel;
    else if (kind == "generate") candidate.kind = ServeRequest::Kind::generate;
    else { err = "unknown request type"; return false; }
    for (const auto& [key, unused] : root.object) {
        (void)unused;
        if (key != "type" && !(kind != "shutdown" && key == "request_id") &&
            !(kind == "generate" && contains(key, {"ids", "max_new", "sampling"}))) {
            err = "unexpected request field: " + key; return false;
        }
    }
    if (kind != "shutdown") {
        const auto id = root.object.find("request_id");
        if (id == root.object.end() || id->second.kind != Value::Kind::string || !safe_id(id->second.text)) {
            err = "request_id must contain 1..128 ASCII letters, digits, underscores or hyphens"; return false;
        }
        candidate.request_id = id->second.text;
    }
    if (kind == "generate") {
        if (n_vocab <= 0 || context == 0 || context > kDistributedMaxContext) { err = "invalid vocabulary or context capacity"; return false; }
        const auto ids = root.object.find("ids"), count = root.object.find("max_new"), params = root.object.find("sampling");
        uint64_t max_new = 0;
        if (ids == root.object.end() || ids->second.kind != Value::Kind::array || ids->second.array.empty() ||
            count == root.object.end() || !integer(count->second, context, max_new) || max_new == 0 ||
            ids->second.array.size() > context - max_new || params == root.object.end()) {
            err = "generate requires nonempty ids, positive max_new and sampling within context capacity"; return false;
        }
        candidate.max_new = static_cast<uint32_t>(max_new);
        for (const auto& token : ids->second.array) {
            uint64_t id = 0;
            if (!integer(token, static_cast<uint64_t>(n_vocab - 1), id)) { err = "invalid token id"; return false; }
            candidate.ids.push_back(static_cast<int32_t>(id));
        }
        if (!sampling(params->second, n_vocab, candidate, err)) return false;
    }
    out = std::move(candidate);
    return true;
}

std::string serve_event(std::string_view type, std::string_view request_id, int32_t token,
                        uint32_t max_context, std::string_view state) {
    if (!contains(type, {"ready", "token", "done", "cancelled", "health", "error"}))
        throw std::invalid_argument("unknown serve event");
    std::string event = "{\"type\":" + quote(type) + ",\"alive_ranks\":[0,1,2]";
    if (type == "ready") {
        if (!max_context || max_context > kDistributedMaxContext) throw std::invalid_argument("invalid event context");
        event += ",\"protocol\":\"strata-distributed-v1\",\"mode\":\"autoregressive\",\"world\":3,\"max_context\":" + std::to_string(max_context);
    } else if (type == "token" || type == "done" || type == "cancelled") {
        if (!safe_id(request_id)) throw std::invalid_argument("invalid event request_id");
        event += ",\"request_id\":" + quote(request_id);
        if (type == "token") {
            if (token < 0) throw std::invalid_argument("invalid event token");
            event += ",\"committed\":true,\"id\":" + std::to_string(token);
        } else event += ",\"state\":\"ready\"";
    } else if (type == "health") {
        if (state != "ready" && state != "busy") throw std::invalid_argument("invalid health state");
        event += ",\"state\":" + quote(state);
    } else event += ",\"state\":" + quote(state);
    return event + '}';
}

}  // namespace strata::distributed
