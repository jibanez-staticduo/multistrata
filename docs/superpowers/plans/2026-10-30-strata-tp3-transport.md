# Strata TP3 Transport Implementation Plan

> **For agentic workers:** Use superpowers:subagent-driven-development or superpowers:executing-plans task by task. This companion is scoped by the main implementation plan.

**Goal:** Provide the opt-in native transport for three Strata CUDA workers.

**Architecture:** Private control coordination with independent NCCL tensor traffic; bounded asynchronous operations and group abort on failure.

**Tech Stack:** C++20, CUDA, NCCL C API, private TCP coordination.

**Spec:** [TP3 design](../specs/2026-10-30-strata-tp3-design.md).

**Parent:** [Complete implementation plan](2026-10-30-strata-tp3-implementation.md).

Date: 2026-10-30. Status: proposed work; no implementation or GPU validation.

## Outcome and acceptance

Provide a native CUDA/ARM64 transport for one model using TP inside each layer
across three Spark hosts. Start with NCCL's C API. Keep production serving.
This document covers transport; model partitioning remains a separate workstream.
Transport acceptance does not establish model correctness or speed.

- A separate opt-in build exchanges FP32 and BF16 buffers correctly on three ranks.
- Every rank agrees on session, model, library, configuration and collective order.
- Startup disagreement, peer loss and collective timeout end the whole session.
- The existing build has no distributed dependency or changed execution path.
- GPU evidence is recorded only after an agreed isolated hardware window.

## Verified reference versus proposed files

Verified on DG1: `/home/staticduo/recipes/triple/qwen3.8-flash-next-staticduo21-tp3.yaml`.
Its running configuration uses TP3, PP1 and a patched NCCL library at
`/home/staticduo/recipes/patches/staticduo14-tp3-nccl/libnccl.so.2`.
The runtime mod is `/home/staticduo/recipes/mods/qwen38-staticduo18-handover-runtime`.
Its `network.py:23` installs a B12X C proxy and `network.py:34` makes failure fatal.
Its `roce_proxy.c:113` selects IPv4 RoCEv2 GIDs per HCA; `:380` matches direct /24 links.
The installed B12X proxy matched this mod by SHA-256 during read-only inspection.
Its `tp3.py:10` masks logical vocabulary in an already padded checkpoint; it does
not convert weights or establish that Strata can load that checkpoint.

The local Sparknet snapshot is a separate reference at
`/home/staticduo/tmp/deepseek-work/nvfp4-migration-20261023/transport-diff/sparknet`.
It contains a later TileLang collective and C proxy; the inspected running B12X
collective imports CuTe instead. Do not equate those implementations.
All repository paths in the tasks below are proposed, unless explicitly noted.

## Task 1: define the transport contract

Create `include/strata/distributed/transport.hpp` with these proposed types:

```cpp
enum class ScalarType { fp32, bf16 };
struct DeviceBuffer { void* data; size_t elements; ScalarType type; };
struct Operation { uint64_t sequence; uint64_t layer; uint64_t step; };
struct Deadline { std::chrono::steady_clock::time_point end; };
class Transport {
 public:
  virtual ~Transport() = default;
  virtual void all_reduce_sum(DeviceBuffer input, DeviceBuffer output,
                              Operation operation, cudaStream_t stream) = 0;
  virtual void wait(Deadline deadline) = 0;
  virtual void check_health() = 0;
  virtual void abort(std::string_view reason) noexcept = 0;
};
```

Keep CUDA types inside the distributed build surface. Define ownership explicitly:
the caller owns buffers and stream until completion; transport owns communicator,
control sockets and completion events. Initially support one ordered stream and
one in-flight operation. Validate dtype, extent, overflow and aliasing before enqueue.
Allow exact in-place reduction or disjoint input/output only; reject partial
overlap. The model plan also requires an ordered all-gather with equal padded
wire slots and explicit per-rank valid counts. Agree shape/layout/counts before
initialization and test the contract. Reduce-scatter is not required initially.

## Task 2: add a private rank coordinator

Create `include/strata/distributed/session.hpp`, `src/distributed/session.cpp`
and `src/distributed/control_tcp.cpp`. Use bounded length-prefixed messages,
explicit protocol version and byte order; reject malformed or excessive lengths.
TCP carries coordination only, not tensor payload. Bind a selected private address;
require a session credential supplied outside source and redact it in logs.

Rank 0 coordinates a three-rank join with a fresh session UUID. Each join carries
UUID, rank, world size, model hash, configuration fingerprint, protocol version,
backend identity, expected NCCL version and library provenance fingerprint.
Model hash covers manifest and tensor metadata/content identity, not just a path.
Configuration fingerprint includes model family, quantization, geometry, shard map,
dtype policy and execution protocol. Exclude host-specific interface names.
Reject duplicate/out-of-range ranks, wrong UUID, stale joins and any disagreement.
Do not issue READY until all three verified ranks have acknowledged the same roster.
Exchange the NCCL bootstrap identifier through this channel without logging it.

Give connect, join, initialization and operation phases explicit deadlines.
Use monotonic clocks locally; exchange durations, not machine clock timestamps.
Carry sequence/layer/step metadata in debug validation to detect ordering mistakes.
On disagreement or disconnect, broadcast ABORT and abort locally. A missing peer
cannot acknowledge ABORT, so each rank must also enforce its own deadline.
Never retry a failed collective within the same communicator/session.

## Task 3: implement the NCCL backend and isolated build

Create `src/distributed/nccl_transport.cpp` and a private NCCL API boundary header.
Resolve exact NCCL signatures and lifecycle against the selected official version
before writing calls. Document initialization, async error polling, stream/event
completion and communicator abort behavior, including blocked initialization.
Do not rely on blocking synchronization without a watchdog and abort path.
Use NCCL Sum for FP32/BF16; distinguish enqueue success from GPU completion.

Add an opt-in `STRATA_BUILD_DISTRIBUTED` CMake option, default OFF, in a separate
build directory and a standalone target `strata-tp3-transport-smoke`.
Build CPU protocol tests independently with `STRATA_BUILD_DISTRIBUTED_TESTS=ON`;
they must not require CUDA/NCCL or enable the device backend implicitly.
Pin supported headers/runtime and compare runtime version with the approved build.
Reject mismatched versions and unknown library provenance explicitly; a matching
version number alone does not prove the local triangle patches are present.
Keep the current patched triangle library intact. Build an isolated library from
documented source/patch hashes when needed; never replace production's library.
Review ARM64 ordering fixes and per-port GID selection in the selected NCCL build.
Inspect active topology and actual GID tables during the hardware window.
Generate rank-local network settings; do not copy recipe environment values blindly.
Require IB/RoCE for this experiment and reject socket data fallback with evidence.
Control TCP remains allowed. Do not introduce public Terminal/app flags yet.

## Task 4: CPU contract tests at the external boundary

Create `tests/distributed/test_session.cpp` and `test_transport_contract.cpp`.
Use fake sockets, clock and NCCL API boundary; do not fake internal algorithm steps.
Cover duplicate/missing rank, UUID/model/config mismatch, oversized/truncated frame,
library mismatch, join timeout, peer disconnect, abort propagation and sequence error.
Cover enqueue errors, pending async errors, deadline expiry and idempotent teardown.
Configure with `cmake -S . -B build-tp3-cpu -DSTRATA_BUILD_DISTRIBUTED_TESTS=ON`,
build the new test targets and run `ctest --test-dir build-tp3-cpu -R distributed --output-on-failure`.
Expected: contract cases pass and no indefinite wait. These tests prove no GPU,
RDMA, NCCL kernel, graph or model behavior.

## Task 5: standalone three-rank GPU smoke

Create `tests/distributed/nccl_transport_smoke.cu` and
`tools/distributed/run_transport_smoke.py` using private test-only parameters.
Plan build: `cmake -S . -B build-tp3 -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_DISTRIBUTED=ON` then
`cmake --build build-tp3 --target strata-tp3-transport-smoke`.
Finalize architecture/toolchain flags from the selected Spark CUDA installation.
The runner launches only during an explicitly scheduled idle hardware window.
Do not stop, install into, modify or benchmark against the running production service.

Test FP32 and BF16 with deterministic rank-specific inputs and analytic sums.
Before MoE integration, run `all_gather_padded` on GPU with valid lengths
256/192/192 and rank-specific payloads. Reconstruct the logical 640 values in
global order, excluding wire padding. Compare global Q8_K quantization for a
block crossing offset448 against the independent monolithic quantizer; include
nonuniform maxima so accidentally quantizing each local shard fails the test.
Include model-shaped payloads from about 10 KiB through 2 MiB, boundary element
counts, and larger prefill payloads at 4/8/16 MiB. Derive exact dimensions from the
model plan; measure allocation limits before increasing sizes. Check every rank.
Use exactly representable test values first, then declared dtype-specific tolerances
for varied values; report max/mean errors and NaN/Inf checks. Do not demand bitwise
equality with a different reduction order without evidence.
Repeat across operations, streams only if supported, and later CUDA graph replay.
Inject missing rank, killed peer and deliberately mismatched collective metadata.
Expected: correct sums, bounded failure and no output accepted after failure.
Record commit, library hash, headers/runtime, network paths, commands and raw results.
No GPU test has run: all three Sparks currently serve production.

## Task 6: optional RoCEnante phase after NCCL acceptance

Review licenses/provenance of B12X, Sparknet and extracted helpers before copying.
Evaluate the plain-C proxy ABI independently from Python/torch bootstrap code.
The C proxy posts RDMA; it is not the GPU collective. A native CUDA port also needs
staging, doorbell publication, system-scope fences, peer flags, fixed slots/epochs,
reduction, timeout/error propagation, graph ordering and buffer lifetime management.
Preserve per-HCA GID discovery and direct triangle routes in the chosen source.
Add protocol tests plus three-host payload/timeout/graph tests before enabling it.
Keep NCCL for ineligible payloads, with identical dispatch decisions on every rank;
backend failure aborts the session instead of silently switching collectives.

## Delivery gate

Review this plan with the model partition owner before implementation. Review the
integrated transport once against the acceptance criteria. Publish CPU and GPU
results separately, with failed and not-run checks explicit. A transport smoke pass
permits model integration work; it does not authorize production rollout or claims
about TP3 throughput. Leave runtime parameters internal until that integration is ready.
