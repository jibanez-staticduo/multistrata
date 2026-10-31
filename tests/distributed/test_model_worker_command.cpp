// CPU checks for exact command agreement and real model's declared stage schedule. No CUDA execution.
#include "strata/distributed/model_worker.hpp"

#include <cstdio>
#include <functional>
#include <limits>

namespace {
using namespace strata::distributed;
int checks = 0;
int failures = 0;
void check(bool yes, const char* label) {
    ++checks;
    if (!yes) { ++failures; std::printf("FAIL: %s\n", label); }
}
void canonical_fields() {
    StepCommand baseline{"session", "request", 0, 123, StepPhase::decode, {100}, {17}, "manifest", {}};
    ScheduledCollective collective;
    collective.descriptor.operation = {19, 3, 123};
    collective.descriptor.kind = CollectiveKind::all_gather_padded;
    collective.descriptor.elements = 256;
    collective.descriptor.valid_counts = {256, 192, 192};
    collective.route = "expert_hidden_0";
    baseline.expected_collective_schedule = {collective};
    std::string original;
    check(!canonical_step_command(baseline, original), "canonical command encoded");
    const std::vector<std::function<void(StepCommand&)>> mutate = {
        [](auto& c) { c.session += 'x'; }, [](auto& c) { c.request += 'x'; },
        [](auto& c) { ++c.slot; }, [](auto& c) { ++c.step; },
        [](auto& c) { c.phase = StepPhase::prefill; }, [](auto& c) { ++c.tokens[0]; },
        [](auto& c) { c.tokens.push_back(1); }, [](auto& c) { ++c.positions[0]; },
        [](auto& c) { c.positions.push_back(1); }, [](auto& c) { c.manifest_hash += 'x'; },
        [](auto& c) { ++c.expected_collective_schedule[0].descriptor.operation.sequence; },
        [](auto& c) { ++c.expected_collective_schedule[0].descriptor.operation.layer; },
        [](auto& c) { ++c.expected_collective_schedule[0].descriptor.operation.step; },
        [](auto& c) { c.expected_collective_schedule[0].descriptor.kind = CollectiveKind::all_reduce_sum; },
        [](auto& c) { c.expected_collective_schedule[0].descriptor.type = ScalarType::bf16; },
        [](auto& c) { ++c.expected_collective_schedule[0].descriptor.elements; },
        [](auto& c) { ++c.expected_collective_schedule[0].descriptor.valid_counts[1]; },
        [](auto& c) { c.expected_collective_schedule[0].route += 'x'; },
        [](auto& c) { c.expected_collective_schedule.push_back(c.expected_collective_schedule[0]); }
    };
    for (const auto& change : mutate) {
        auto command = baseline; change(command); std::string bytes;
        check(!canonical_step_command(command, bytes) && original != bytes, "every command field participates in agreement");
    }
    auto a = baseline; a.session = "a"; a.request = "bc";
    auto b = baseline; b.session = "ab"; b.request = "c";
    std::string left, right;
    check(!canonical_step_command(a, left) && !canonical_step_command(b, right) && left != right,
          "field boundaries cannot collide");
    a.session.assign(1024 * 1024 + 1, 'x');
    check(canonical_step_command(a, left).code == FailureCode::oversized_frame && left.empty(),
          "oversized command rejected without partial payload");
    a = baseline; a.phase = static_cast<StepPhase>(uint32_t(StepPhase::rollback) + 1);
    check(bool(canonical_step_command(a, left)) && left.empty(), "phase above appended rollback rejected");
    a = baseline; a.phase = StepPhase::checkpoint;
    b = baseline; b.phase = StepPhase::rollback;
    check(!canonical_step_command(a, left) && !canonical_step_command(b, right) && left != right,
          "checkpoint and rollback participate distinctly in exact agreement");
    check(uint32_t(StepPhase::abort) == 6 && uint32_t(StepPhase::checkpoint) == 7 &&
          uint32_t(StepPhase::rollback) == 8, "new phases append without changing existing wire values");
    a = baseline; a.slot = kWorkerMaxSlots - 1;
    check(!canonical_step_command(a, left) && left != original,
          "configured final slot identity participates in complete command agreement");
}
void model_schedule() {
    StepCommand command{"session", "request", 0, 3, StepPhase::prefill, {4}, {0}, "manifest", {}};
    std::vector<ScheduledCollective> out;
    const strata::core::ModelGeometry g;
    check(!model_forward_schedule(command, g, 100, out) && out.size() == 48 * 22,
          "all 48 layers have mixer and all ten expert collective pairs");
    check(out.front().descriptor.operation.sequence == 100 && out.back().descriptor.operation.sequence == 1155 &&
          out.back().descriptor.operation.step == 3, "entire declared sequence is contiguous");
    check(out[0].route == "gdn_values" && out[0].descriptor.valid_counts == std::vector<uint64_t>{2304, 1920, 1920} &&
          out[1].route == "mixer" && out[1].descriptor.elements == 2560, "GDN original global stage shapes");
    check(out[3 * 22].route == "qsa_gated_reference" &&
          out[3 * 22].descriptor.valid_counts == std::vector<uint64_t>{2048, 2048, 2048} &&
          out[3 * 22 + 1].route == "qsa_output_rank_zero_reference", "QSA every fourth layer shapes");
    check(out[2].route == "expert_hidden_0" && out[2].descriptor.valid_counts == std::vector<uint64_t>{256, 192, 192} &&
          out[21].route == "expert_lane_terms_9" &&
          out[21].descriptor.valid_counts == std::vector<uint64_t>{327680,327680,327680}, "all expert route slots in order");
    std::vector<ScheduledCollective> batched;
    check(!model_forward_schedule(command, g, 100, batched, true) && batched.size() == 48 * 4,
          "batched schedule has two expert collectives per layer");
    check(batched[2].route == "expert_hidden_batch" && batched[2].descriptor.elements == 2560 &&
          batched[2].descriptor.valid_counts == std::vector<uint64_t>{2560,2560,2560},
          "batched hidden transport includes every padded expert slot");
    check(batched[3].route == "expert_lane_terms_batch" && batched[3].descriptor.elements == 3276800 &&
          batched[3].descriptor.valid_counts == std::vector<uint64_t>{3276800,3276800,3276800},
          "batched exact operand transport retains all expert lanes");
    check(batched.front().descriptor.operation.sequence == 100 &&
          batched.back().descriptor.operation.sequence == 291 &&
          batched[3 * 4].route == "qsa_gated_reference", "batched layer order and sequences");
    auto sequential_command = command; sequential_command.expected_collective_schedule = out;
    auto batched_command = command; batched_command.expected_collective_schedule = batched;
    std::string sequential_bytes, batched_bytes;
    check(!canonical_step_command(sequential_command, sequential_bytes) &&
          !canonical_step_command(batched_command, batched_bytes) && sequential_bytes != batched_bytes,
          "collective policy participates in exact command agreement");
    command.tokens.push_back(5); command.positions.push_back(1);
    check(!model_forward_schedule(command, g, 1, out) && out.size() == 2 * 48 * 22 &&
          out[48 * 22].descriptor.operation.layer == 0, "multi token prefill restarts all layers without reset sequence");
    check(!model_forward_schedule(command, g, 1, batched, true) && batched.size() == 2 * 48 * 4 &&
          batched[48 * 4].descriptor.operation.layer == 0 &&
          batched[48 * 4].descriptor.operation.sequence == 193, "batched prefill preserves token boundaries");
    command.phase = StepPhase::commit;
    check(!model_forward_schedule(command, g, 1, out) && out.empty(), "commit replays no model collective");
    for (auto phase : {StepPhase::checkpoint, StepPhase::rollback}) {
        command.phase = phase;
        check(!model_forward_schedule(command, g, 1, out) && out.empty() &&
              !model_forward_schedule(command, g, 1, batched, true) && batched.empty(),
              "checkpoint and rollback have zero collectives under either expert policy");
    }
    command.phase = static_cast<StepPhase>(99);
    check(bool(model_forward_schedule(command, g, 1, out)), "unknown schedule phase rejected");
    command.phase = StepPhase::prefill;
    auto bad_geometry = g; bad_geometry.n_layers = 47;
    check(bool(model_forward_schedule(command, bad_geometry, 1, out)), "unsupported geometry rejected");
    check(bool(model_forward_schedule(command, g, std::numeric_limits<uint64_t>::max(), out)), "sequence overflow rejected");
    command.tokens.resize(9);
    check(bool(model_forward_schedule(command, g, 1, out)), "command workload bounded before allocation");
}
}
int main() {
    canonical_fields(); model_schedule();
    std::printf("model worker command CPU: %d checks, %d failures (no model/GPU execution)\n", checks, failures);
    return failures ? 1 : 0;
}
