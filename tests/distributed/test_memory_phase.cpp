#include "strata/distributed/memory_phase.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <unistd.h>

namespace {

using namespace strata::distributed;

int failures = 0;
int checks = 0;

void check(bool condition, const char* what) {
    ++checks;
    std::printf("  %-76s %s\n", what, condition ? "ok" : "FAIL");
    if (!condition) ++failures;
}

std::string fresh_path(const char* suffix) {
    return std::string("/tmp/strata-memory-phase-") + std::to_string(::getpid()) + suffix;
}

MemoryPhaseSnapshot valid_snapshot() {
    MemoryPhaseSnapshot snapshot;
    snapshot.timestamp_utc = "2026-10-31T12:34:56.123Z";
    snapshot.rank = 1;
    snapshot.request = "request-7";
    snapshot.step = 12;
    snapshot.phase = MemoryPhase::first_decode;
    snapshot.position = 41;
    snapshot.cuda_free_bytes = 900;
    snapshot.cuda_total_bytes = 1000;
    return snapshot;
}

void fresh_file_and_json_record() {
    const auto path = fresh_path("-ok.jsonl");
    ::unlink(path.c_str());
    std::string error;
    MemoryPhaseWriter writer;
    check(writer.open_new(path, error), "fresh JSONL opens exclusively");
    auto snapshot = valid_snapshot();
    snapshot.request = "request\"\\\n";
    check(writer.append(snapshot, error), "valid snapshot appends");
    check(!writer.append(snapshot, error), "a request/phase pair cannot be duplicated");
    writer.close();

    std::ifstream input(path);
    std::string line, extra;
    const bool one_line = bool(std::getline(input, line)) && !std::getline(input, extra);
    check(one_line, "writer emits exactly one bounded JSONL line");
    check(line.find("\"request\":\"request\\\"\\\\\\n\"") != std::string::npos,
          "request text is escaped as valid JSON");
    check(line.find("\"rank\":1") != std::string::npos &&
              line.find("\"phase\":\"first_decode\"") != std::string::npos &&
              line.find("\"cuda_free_bytes\":900") != std::string::npos,
          "rank, phase and allocator availability are serialized");
    check(line.find("device_allocator_availability_not_process_allocation") != std::string::npos &&
              line.find("unified_memory_may_overlap") == std::string::npos &&
              line.find("may_overlap_host_accounting") != std::string::npos,
          "record identifies allocator scope and unified-memory overlap");
    ::unlink(path.c_str());
}

void refuses_existing_file_and_invalid_values() {
    std::string error;
    MemoryPhaseWriter unopened;
    check(!unopened.append(valid_snapshot(), error), "append before open is refused");

    const auto path = fresh_path("-existing.jsonl");
    { std::ofstream existing(path); existing << "keep\n"; }
    MemoryPhaseWriter writer;
    check(!writer.open_new(path, error), "existing output file is never truncated or appended");
    std::ifstream existing(path);
    std::string content;
    std::getline(existing, content);
    check(content == "keep", "refused existing file retains its original contents");
    ::unlink(path.c_str());

    const auto invalid_path = fresh_path("-invalid.jsonl");
    MemoryPhaseWriter invalid_writer;
    check(invalid_writer.open_new(invalid_path, error), "invalid-value fixture opens a fresh file");
    auto snapshot = valid_snapshot();
    snapshot.rank = 3;
    check(!invalid_writer.append(snapshot, error), "rank outside the three-rank group is rejected");
    snapshot = valid_snapshot();
    snapshot.timestamp_utc = "2026-02-30T12:34:56Z";
    check(!invalid_writer.append(snapshot, error), "impossible UTC calendar date is rejected");
    snapshot = valid_snapshot();
    snapshot.timestamp_utc = "2026-10-31T12:34:56+00:00";
    check(!invalid_writer.append(snapshot, error), "timezone offset is rejected in favor of explicit UTC Z");
    snapshot = valid_snapshot();
    snapshot.phase = static_cast<MemoryPhase>(255);
    check(!invalid_writer.append(snapshot, error), "unknown phase enum is rejected");
    snapshot = valid_snapshot();
    snapshot.cuda_free_bytes = 1001;
    check(!invalid_writer.append(snapshot, error), "free memory greater than total is rejected");
    snapshot = valid_snapshot();
    snapshot.request.assign(MemoryPhaseWriter::kMaxRequestBytes + 1, 'x');
    check(!invalid_writer.append(snapshot, error), "oversized request identifiers are rejected");
    invalid_writer.close();
    std::ifstream empty(invalid_path);
    check(empty.peek() == std::ifstream::traits_type::eof(), "invalid snapshots write no partial record");
    ::unlink(invalid_path.c_str());
}

void distinguishes_request_and_phase() {
    const auto path = fresh_path("-distinct.jsonl");
    ::unlink(path.c_str());
    std::string error;
    MemoryPhaseWriter writer;
    check(writer.open_new(path, error), "distinct-events fixture opens a fresh file");
    const auto first = valid_snapshot();
    check(writer.append(first, error), "first phase for a request is accepted");
    auto next = first;
    next.phase = MemoryPhase::decode;
    next.step = 13;
    next.position = 42;
    check(writer.append(next, error), "another phase for that request is accepted");
    next.request = "request-8";
    next.phase = MemoryPhase::first_decode;
    check(writer.append(next, error), "same phase for a different request is accepted");
    writer.close();
    ::unlink(path.c_str());
}

}  // namespace

int main() {
    std::printf("CPU memory phase evidence writer\n");
    fresh_file_and_json_record();
    refuses_existing_file_and_invalid_values();
    distinguishes_request_and_phase();
    std::printf("%d checks, %d failed\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
