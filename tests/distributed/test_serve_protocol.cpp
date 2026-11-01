#include "strata/distributed/serve_protocol.hpp"
#include "strata/distributed/limits.hpp"

#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>

using namespace strata::distributed;

namespace {
int checks = 0;
int failures = 0;
void check(bool ok, const char* name) {
    ++checks;
    if (!ok) { ++failures; std::fprintf(stderr, "FAIL: %s\n", name); }
}
std::string generate(std::string sampling = "{}", std::string ids = "[1,2]", std::string max_new = "3") {
    return "{\"type\":\"generate\",\"request_id\":\"r_1-A\",\"ids\":" + ids +
           ",\"max_new\":" + max_new + ",\"sampling\":" + sampling + '}';
}
bool parse(std::string_view json, ServeRequest& out, uint32_t context = 10, int32_t vocab = 100) {
    std::string error;
    return parse_serve_request(json, vocab, context, out, error);
}
void reject(std::string_view json, const char* name) {
    ServeRequest out;
    out.request_id = "unchanged";
    std::string error;
    check(!parse_serve_request(json, 100, 10, out, error) && !error.empty() && out.request_id == "unchanged", name);
}
template <typename F> void throws(F f, const char* name) {
    bool caught = false;
    try { f(); } catch (const std::invalid_argument&) { caught = true; }
    check(caught, name);
}
}  // namespace

int main() {
    ServeRequest out;
    std::string error;
    check(parse(generate(), out), "generate parses");
    check(out.kind == ServeRequest::Kind::generate && out.request_id == "r_1-A" &&
          out.ids == std::vector<int32_t>({1, 2}) && out.max_new == 3, "generate fields preserved");
    check(out.sampling.greedy && out.sampling.temperature == 0 && out.sampling.top_p == 1 &&
          out.sampling.top_k == 20 && out.sampling.penalty_last_n == 0 && out.sampling.seed == 0,
          "default sampler matches native greedy path");
    check(parse("{\"type\":\"cancel\",\"request_id\":\"r\"}", out) && out.kind == ServeRequest::Kind::cancel &&
          out.ids.empty() && out.max_new == 0, "cancel replaces prior generate");
    check(parse("{\"type\":\"shutdown\"}", out) && out.kind == ServeRequest::Kind::shutdown, "shutdown parses");
    check(parse(" \r\n\t" + generate() + "\n", out), "JSON whitespace accepted");
    check(parse(generate(), out, 5), "context equality accepted");
    check(!parse(generate(), out, 4), "context overflow refused");
    check(!parse(generate(), out, kDistributedMaxContext + 1), "context above hard limit refused");
    check(!parse(generate(), out, 0), "zero context refused");
    check(!parse(generate(), out, 10, 0), "invalid vocabulary refused");
    std::string prompt_ids = "[1";
    for (size_t i = 1; i < 65536; ++i) prompt_ids += ",1";
    prompt_ids += ']';
    check(parse(generate("{}", prompt_ids, "256"), out, kDistributedMaxContext) &&
          out.ids.size() == 65536 && out.max_new == 256,
          "64K prompt plus 256 outputs fits bounded context without truncation");
    out.request_id = "unchanged";
    check(!parse(generate("{}", prompt_ids, "257"), out, 65792) &&
          out.request_id == "unchanged", "one extra output is rejected without changing request");
    std::string maximum_ids = "[2147483646";
    maximum_ids.reserve(262143 * 11 + 2);
    for (size_t i = 1; i < 262143; ++i) maximum_ids += ",2147483646";
    maximum_ids += ']';
    check(parse_serve_request(generate("{}", maximum_ids, "1"), std::numeric_limits<int32_t>::max(),
                              262144, out, error) && out.ids.size() == 262143,
          "256K prompt with maximum-width token IDs and one output parses without truncation");
    check(!parse_serve_request(generate("{}", maximum_ids, "2"), std::numeric_limits<int32_t>::max(),
                               262144, out, error),
          "256K prompt still rejects a one-token context overflow");

    for (auto malformed : {"", "{", "[]", "null", "true", "{}", "{\"type\":true}",
        "{\"type\":\"other\"}", "{\"type\":\"shutdown\",}", "{\"type\":\"shutdown\"}{}",
        "{\"type\":\"shutdown\",\"request_id\":\"r\"}", "{\"type\":\"cancel\"}",
        "{\"type\":\"cancel\",\"request_id\":\"r\",\"ids\":[1]}",
        "{\"type\":\"shutdown\",\"type\":\"shutdown\"}",
        "{\"type\":\"shutdown\",\"ty\\u0070e\":\"shutdown\"}"}) reject(malformed, "malformed/duplicate/unexpected request");
    const std::string base = generate();
    for (size_t i = 0; i < base.size(); ++i) reject(std::string_view(base).substr(0, i), "every truncated prefix refused");
    for (auto ids : {"[]", "[true]", "[null]", "[1.0]", "[1e0]", "[-1]", "[-0]", "[100]",
                     "[2147483648]", "[18446744073709551616]", "[01]", "[+1]", "[1,]"})
        reject(generate("{}", ids), "invalid token ids refused");
    for (auto count : {"0", "-1", "true", "null", "1.0", "1e0", "4294967296", "18446744073709551616"})
        reject(generate("{}", "[1]", count), "invalid max_new refused");
    for (auto value : {"NaN", "Infinity", "-Infinity", "1e999", "1e-999", "1.", ".1", "1e", "--1", "00"})
        reject(generate(std::string("{\"temperature\":") + value + '}'), "invalid number refused");
    for (auto sampling : {"null", "[]", "{\"temperature\":true}", "{\"top_p\":0}", "{\"top_p\":1.1}",
        "{\"min_p\":-0.1}", "{\"min_p\":1.1}", "{\"repetition_penalty\":0}", "{\"top_k\":1.1}",
        "{\"top_k\":-1}", "{\"top_k\":2147483648}", "{\"penalty_last_n\":-1}",
        "{\"temperature\":1e39}", "{\"seed\":-1}", "{\"seed\":18446744073709551616}",
        "{\"seed\":1.0}", "{\"seed\":1e2}", "{\"seed\":true}", "{\"bogus\":0}",
        "{\"strata_tune\":{}}", "{\"strata_checkpoint\":false}", "{\"strata_prefix\":{}}",
        "{\"strata_img_keys\":[]}", "{\"experimental_speed_projection\":false}", "{\"logprobs\":true}",
        "{\"temperature\":0,\"temperature\":1}", "{\"stop\":{\"x\":1,\"x\":2}}"})
        reject(generate(sampling), "invalid/unsupported sampling refused");

    check(parse(generate("{\"temperature\":0.75,\"top_p\":0.9,\"top_k\":0,\"min_p\":0.1,"
                         "\"repetition_penalty\":1.2,\"frequency_penalty\":-0.4,\"presence_penalty\":0.5,"
                         "\"seed\":18446744073709551615}"), out), "full sampler parses");
    check(!out.sampling.greedy && out.sampling.temperature == 0.75f && out.sampling.top_p == 0.9f &&
          out.sampling.top_k == 64 && out.sampling.min_p == 0.1f && out.sampling.penalty_repeat == 1.2f &&
          out.sampling.penalty_freq == -0.4f && out.sampling.penalty_present == 0.5f &&
          out.sampling.penalty_last_n == 64 && out.sampling.seed == std::numeric_limits<uint64_t>::max(),
          "full sampler preserved with exact uint64 max seed");
    check(parse(generate("{\"seed\":9007199254740993}"), out) && out.sampling.seed == 9007199254740993ULL,
          "seed beyond double exact range preserved");
    check(parse(generate("{\"temperature\":-1,\"top_k\":65,\"penalty_last_n\":0,\"frequency_penalty\":1}"), out) &&
          out.sampling.greedy && out.sampling.top_k == 64 && out.sampling.penalty_last_n == 0,
          "greedy and explicit disabled penalty window preserved");
    check(parse(generate("{\"logit_bias\":{\"0\":1.5,\"99\":-100}}"), out) &&
          out.logit_bias.size() == 2 && out.logit_bias.at(0) == 1.5f && out.logit_bias.at(99) == -100,
          "logit bias parsed");
    for (auto bias : {"[]", "{\"-1\":1}", "{\"100\":1}", "{\"1.0\":1}", "{\"1\":true}",
                      "{\"1\":1e39}", "{\"1\":1,\"01\":2}"})
        reject(generate(std::string("{\"logit_bias\":") + bias + '}'), "invalid bias refused");

    check(parse(generate("{\"stop\":[\"a\\n\\t\\b\\f\\r\\\\\\\"\\/\",\"\\uD834\\uDD1E\"],\"max_tokens\":3,"
                         "\"response_format\":{\"type\":\"json_object\"},\"thinking_budget\":200,"
                         "\"reasoning_budget_tokens\":200,\"thinking_budget_tokens\":100,\"system\":\"test\","
                         "\"input\":\"test\",\"chat_template_kwargs\":{},\"stop_sequences\":[\"x\"]}"), out),
          "service formatting fields and valid Unicode ignored");
    for (auto string : {"\"\\uD800\"", "\"\\uDC00\"", "\"\\uD800\\u0000\"", "\"\\u12xx\"", "\"\\x01\"", "\"\n\""})
        reject(generate(std::string("{\"stop\":") + string + '}'), "invalid escapes refused even in ignored field");
    reject(generate(std::string("{\"stop\":\"") + char(0xc0) + char(0x80) + "\"}"), "overlong UTF8 refused");
    reject(generate(std::string("{\"stop\":\"") + char(0xed) + char(0xa0) + char(0x80) + "\"}"), "UTF8 surrogate refused");
    reject(generate(std::string("{\"stop\":\"") + char(0xf4) + char(0x90) + char(0x80) + char(0x80) + "\"}"), "out of range UTF8 refused");
    for (const auto& id : {std::string(), std::string("a b"), std::string("a/b"), std::string("a\\n"), std::string(129, 'a')})
        reject("{\"type\":\"cancel\",\"request_id\":\"" + id + "\"}", "unsafe request id refused");
    check(parse("{\"type\":\"cancel\",\"request_id\":\"\\u0061-0_\"}", out) && out.request_id == "a-0_",
          "escaped safe request id decoded");
    check(parse("{\"type\":\"cancel\",\"request_id\":\"" + std::string(128, 'a') + "\"}", out), "128 byte request id accepted");
    std::string nested = "null";
    for (int i = 0; i < 14; ++i) nested = '[' + nested + ']';
    check(parse(generate("{\"stop\":" + nested + '}'), out), "16 container depth accepted");
    nested = '[' + nested + ']';
    reject(generate("{\"stop\":" + nested + '}'), "17 container depth refused");
    reject(std::string(kDistributedMaxLineBytes + 1, ' '), "line above 4 MiB refused");
    check(parse(std::string(kDistributedMaxLineBytes - base.size(), ' ') + base, out),
          "4 MiB line accepted");

    check(serve_event("ready", {}, -1, 65536) ==
          "{\"type\":\"ready\",\"alive_ranks\":[0,1,2],\"protocol\":\"strata-distributed-v1\",\"mode\":\"autoregressive\",\"world\":3,\"max_context\":65536}",
          "ready JSON matches contract");
    check(serve_event("ready", {}, -1, 65792) ==
          "{\"type\":\"ready\",\"alive_ranks\":[0,1,2],\"protocol\":\"strata-distributed-v1\",\"mode\":\"autoregressive\",\"world\":3,\"max_context\":65792}",
          "expanded ready capacity preserves protocol and roster");
    check(serve_event("token", "r", 42) ==
          "{\"type\":\"token\",\"alive_ranks\":[0,1,2],\"request_id\":\"r\",\"committed\":true,\"id\":42}",
          "committed token JSON matches contract");
    for (auto kind : {"done", "cancelled"})
        check(serve_event(kind, "r", -1, 0, "busy") == std::string("{\"type\":\"") + kind +
              "\",\"alive_ranks\":[0,1,2],\"request_id\":\"r\",\"state\":\"ready\"}", "completion state always ready");
    check(serve_event("health", {}, -1, 0, "busy") ==
          "{\"type\":\"health\",\"alive_ranks\":[0,1,2],\"state\":\"busy\"}", "health JSON matches contract");
    check(serve_event("error", {}, -1, 0, "bad\n\"\\") ==
          "{\"type\":\"error\",\"alive_ranks\":[0,1,2],\"state\":\"bad\\u000a\\\"\\\\\"}", "event string safely escaped");
    throws([] { serve_event("unknown"); }, "unknown event refused");
    throws([] { serve_event("ready"); }, "invalid ready context refused");
    throws([] { serve_event("ready", {}, -1, kDistributedMaxContext + 1); }, "oversized ready context refused");
    throws([] { serve_event("token", "r", -1); }, "negative event token refused");
    throws([] { serve_event("token", "bad\n", 1); }, "unsafe event request id refused");
    throws([] { serve_event("health", {}, -1, 0, "failed"); }, "invalid health state refused");
    std::printf("serve_protocol: %d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
