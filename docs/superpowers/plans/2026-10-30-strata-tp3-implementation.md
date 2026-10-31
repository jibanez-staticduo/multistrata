# Strata Native TP3 Implementation Plan

> **For agentic workers:** Use superpowers:subagent-driven-development or superpowers:executing-plans task by task. Checkboxes describe future work, not completed implementation.

**Goal (revised by Jordi, 2026-10-31):** Accelerate native Strata TP3/PP1 on DG1/DG2/DG3 with fast GPU kernels, batched prefill, CUDA Graph replay and fewer host/network synchronization points. Deliver measured improvements over the current TP3 reference while preserving the original weights and qualified numerical behavior.

## Current Execution Priority

This section supersedes the original Task 10 execution order. Jordi explicitly
requests GPU acceleration and fast TP3 kernels as the main work. The exhaustive
4K/16K/64K C1/C4/C8 baseline matrix is stopped; retain completed measurements and
mark unfinished cases as not run or interrupted. Do not spend days measuring
the already demonstrated slow reference path.

Existing acceptance evidence remains valid within its recorded scope: original
64K prompt plus 256 continuation steps, state parity, short/extended sampled
MTP, C4/C8 isolation, and C1 NCCL model graphs. Do not repeat those suites unless
a changed path creates a concrete correctness risk.

1. Measure a short TP3 workload with GPU/collective profiling. Separate active
   GPU work, host routing/control, communication and launch overhead.
2. Put the qualified fast paths into an explicit accelerated execution profile:
   device-side expert routing, full-model graph decode and grouped prompt work.
   Remove their current unnecessary mutual exclusions where correctness allows.
3. Implement the dominant missing fast kernel or fusion identified by that
   profile: grouped expert gate/up/down, exact-native multi-row projection or
   quantized GEMM, GDN/QSA projections, or fused residual/normalization work.
   Use original Strata fast kernels when compatible with the TP shard contract.
4. Reduce per-layer host waits and redundant collective/control exchanges.
   Use the existing NCCL/RoCE primitives with a fixed, agreed execution schedule;
   retain bounded failure and no-output-before-commit behavior.
5. For each change run only its component parity and a short model/state check,
   then an A/B timing with the same pack, prompt and output budget. Keep a change
   only when its measured benefit justifies it. A failed numerical gate is a bug
   to diagnose, never a reason to widen the frozen policy automatically.
6. Finish with an opt-in usable TP3 accelerated profile and a concise record of
   prefill/decode gains, remaining limits and actual failures. Run a larger final
   check once the fast candidate is stable, rather than restarting the full
   matrix after every edit. Restore Staticduo21 once at the end and verify it.

Acceptance: an actual three-host GPU execution uses the accelerated kernels,
preserves the qualified model/state behavior on changed seams, and shows a
repeatable prefill or decode improvement over the current TP3 reference. Report
the comparison with TP1 separately; an improvement over slow TP3 alone is not a
claim that TP3 beats TP1. No speed claim from compilation or flags alone.

## Acceleration Execution Record

Implemented and tested on 31 October 2026:

- Fused routed gate/up, SwiGLU and padded output; original six native codecs.
- GPU top-k Q8_1 expansion and device-routed batched prefill without CPU expert
  grouping; batched hidden quantization and direct final output.
- Decode-only full-model CUDA Graph mode with eager grouped prefill and graph
  retirement/recapture across requests and reset.
- ARM/GB10 build, GPU component parity, 126 launcher tests and 63 CPU worker
  checks pass. Independent scoped reviews report no blocking findings.
- Two short service requests pass TP1 comparison: two full vocabulary rows,
  342 logical states, original zero-error policy; four requests exercise reset.
- Short A/B with three warm requests per profile and diagnostics disabled:
  prefill 1.43x and decode 1.49x faster than reference TP3. This is not a claim
  against TP1/vLLM. Exact measurements and limits are in
  [the acceptance record](../../TP3_EXPERIMENTAL_STATUS.md#short-gpu-ab).

The final accelerated execution passes with 923 prompt tokens and 128 output
tokens, reset and all three ranks ready. Staticduo21 is restored once at the end;
health 200, the expected model ID and a chat response `OK` pass. The revised GPU
acceleration goal is complete within the measured C1 scope. The exhaustive old
matrix remains stopped; combined MTP/multislot graphs and long-context graph
qualification remain outside this delivered profile.

**Architecture:** One CUDA process per host, a private control coordinator and NCCL tensor collectives. Preserve original GGUF geometry through explicit unequal tensor ownership; add RoCEnante only after a correct profiled baseline.

**Tech Stack:** C++20, CUDA/GB10 sm_121, NCCL C API, existing Strata quantized formats, Python preparation tools, SparkRun.

**Spec:** [TP3 design](../specs/2026-10-30-strata-tp3-design.md).

**Companion:** [Native transport plan](2026-10-30-strata-tp3-transport.md).

## Global Constraints

- The original document was planning only. Later implementation and hardware
  evidence are recorded in `docs/TP3_EXPERIMENTAL_STATUS.md` and the execution ledger.
- Start from Strata `870aedd9e1ebe1c91f94da93b59af17729f6c5c0` in an isolated branch/worktree.
- One model, TP3/PP1, Qwen/CUDA/ARM64 first. DeepSeek, HIP, SYCL and app UI additions are outside the first feature.
- Keep default execution/build off-path byte-identical; all distributed builds and options are opt-in.
- No production replacement, host dependency/driver/network changes or automatic rollout.
- Keep original weights immutable. No requantization or model geometry transformation in the MVP.
- Prefer compatible ISTA-DASLab IQ3_S; inspect actual projection types before downloading full files. NVFP4 is a documented alternative.
- Count unified memory once, reserve >=50 GiB disk per Spark, use NAS for originals/backups and `/home/staticduo/tmp` for evidence.
- GPU experiments need an idle cluster window and SparkRun lifecycle. C8 maximum; preserve the existing 2200 MHz policy and firmware protections.
- The existing Staticduo21 weights are optional: safetensors is not a direct loader contract for Strata.
- No upstream PR without maintainer scope agreement, required backend builds and honest validation status.

## Review Focus

- Misaligned quantized activation blocks: identical weight bytes must not hide changed activation scales.
- Rank-dependent routing or cancellation: never let one worker omit a collective its peers execute.
- Unequal heads/FFN: correct global offsets, GQA mapping, scale lookup and original row strides.
- Stateful inference: no accepted output or reusable state after partial commit, reset or worker failure.
- Capacity/performance claims: replication, cache residency and queueing must be counted and compared fairly.

## Ownership and dependency map

| Workstream | Owns proposed files | Depends on |
| --- | --- | --- |
| Artifact/geometry | `tools/distributed/`, `include/strata/distributed/shard_plan.hpp` | Actual selected source headers |
| Transport | `src/distributed/`, `transport.hpp`, control protocol, transport tests | Frozen manifest/operation fingerprints |
| Kernels | `src/kernels/cuda/tp/`, `include/strata/kernels/tp.hpp` | Shard/quantization contracts |
| Runtime/server | `include/strata/distributed/worker.hpp`, `src/program/distributed.cpp`, distributed server adapter | Component parity and transport |
| Integration/QA | experimental recipe, benchmark tooling and reports | Correct whole-model C1 run |

Do not assign simultaneous edits to the same existing CMake/server/core file.
The parent integrator owns shared build wiring and verifies the combined result.
Artifact planning and transport can run in parallel; model state integration follows kernels.

## Proposed file map

These paths are planned additions; existing source anchors are listed separately.

- `tools/distributed/model_inventory.py`: bounded local/remote header inspection; no tensor materialization.
- `tools/distributed/prepare_tp_pack.py`: immutable source views, shard manifest and optional byte-preserving materialization.
- `include/strata/distributed/shard_plan.hpp`: global/local shapes, ranges, quantized blocks and activation halos.
- `include/strata/distributed/{transport,session}.hpp`, `src/distributed/`: see companion transport plan.
- `include/strata/kernels/tp.hpp`, `src/kernels/cuda/tp/{linear,moe,gdn,qsa}.cu`: shard-aware primitive execution.
- `include/strata/distributed/worker.hpp`, `src/distributed/worker.cpp`, `src/program/distributed.cpp`: scheduled worker model.
- `serve/distributed_backend.py`: experimental server-to-coordinator adapter using existing API contracts.
- `tests/distributed/`: mathematical, artifact, transport, component and state tests.
- `tools/distributed/{compare_components,compare_model,benchmark_tp}.py`: reference evidence and matched comparisons.
- `packaging/experimental/strata-tp3/`: image recipe, launch wrapper and sample SparkRun recipe; not a release feed change.

Existing anchors: `tools/iq_pack.py`, `tools/quantscope.py`, `include/strata/core/geometry.hpp`,
`src/kernels/cpu/native_expert.cpp`, `include/strata/core/layer.hpp`,
`src/core/native_dense.cpp`, `src/prefill/`, `src/program/generate.cpp`,
`serve/`, `strata_cli/launch/`, `CMakeLists.txt`, `spark/build_engine.sh`.
Extend explicit interfaces; do not refactor the monolithic generator incidentally.

## Task 1: pin source and record inventory

**Produces:** `inspect_model(source: str) -> ModelInventory` in `model_inventory.py`;
JSON with source revision/hash, every tensor's shape/type/block/scale identity,
geometry, tokenizer/draft/PLE assets and estimated bytes. No compatibility-success
verdict without verifying the tensors required by the chosen path.

- [ ] Add fixtures/tests `tests/distributed/test_model_inventory.py`: mixed gate/up/down types, missing scale, truncated header, foreign geometry and mismatched shards.
- [ ] Run `python -m pytest tests/distributed/test_model_inventory.py -q`; observe failures before implementing the new reader.
- [ ] Reuse existing header readers and add bounded validation; record weight-block AND activation-block requirements per consumer.
- [ ] Inspect ISTA IQ3_S headers pinned to a concrete revision; stop if any required tensor layout is unsupported. Evaluate NVFP4 only with a documented reason.
- [ ] Freeze `source-lock.json`, `inventory.json` and disk estimate under the experiment evidence directory. No full download until the candidate passes inventory.
- [ ] Run the fixture tests and manually cross-check representative tensor shapes/types using existing `tools/strata_inspect.py`.

## Task 2: define ownership and quantization semantics

**Produces:** `make_tp_plan(const ModelInventory&, int world) -> ShardPlan` and
`validate_plan(const ShardPlan&) -> ValidationResult` in `shard_plan.hpp`.
`TensorSlice` records tensor ID, global/local shapes, output/input ranges,
original byte row stride, codec/block size, scales, valid lanes and halos.
`ActivationContract` records norm/transform/quantizer scope and gather layout.
The manifest mirrors these fields and includes a plan hash/version.

- [ ] Add `tests/distributed/shard_plan_test.cpp`: world3 only initially; complete nonoverlapping 6/5/5 GDN key groups and FFN256/192/192; invalid ranges/codec boundaries fail.
- [ ] Test QSA queries at global GQA boundaries using the original 24Q/2KV mapping; replicate both KV and indexer4, never derive GQA from local counts.
- [ ] Test global quantization/transform contracts for a block crossing offset448, including a Q8_K or Hadamard counterexample to naive local processing.
- [ ] Implement the planner without allocating GPU memory. Route unsupported cuts to global-activation reference processing or reject; never silently requantize.
- [ ] Validate `--world 1` as identity and plan a supported TP2 control layout without claiming all world sizes.
- [ ] Build/run `shard_plan_test` through opt-in CMake/CTest. Freeze `tp-plan.json`; memory estimates must include halo/global buffers.

## Task 3: prepare and verify rank artifacts

**Produces:** `prepare_tp_pack(source, plan, output, materialize=False)` and
`verify_tp_pack(source, output) -> VerificationReport` in `prepare_tp_pack.py`.
First use original source views; later files are per-rank, versioned and atomically published.

- [ ] Add `tests/distributed/test_prepare_tp_pack.py`: reconstruct copied blocks/scales exactly; cover mixed quant types, missing files, wrong source hash, padding masks and interrupted output.
- [ ] Implement no-write inventory/plan mode before any copying; validate lengths/offsets against source file sizes.
- [ ] Preserve native expert scale tails and original logical vocabulary; no per-rank scale recomputation.
- [ ] Download only the selected full pack to NAS staging, verify sizes/hashes and promote the immutable source directory.
- [ ] Build source views and reconstruction receipt. Materialized rank files require a separate disk-space check and byte-equality receipt.
- [ ] Run new tests plus `python -m pytest tools/test_iq_pack.py -q`; record pass/fail/not-run explicitly.

## Task 4: native transport and three-process smoke

**Consumes:** manifest/plan hash and operation descriptors. **Produces:** `Transport`,
private session protocol and standalone smoke target from
`2026-10-30-strata-tp3-transport.md`.

- [ ] Implement companion tasks 1-4: private control join, provenance-checked NCCL, bounded async completion and failure abort.
- [ ] Extend its initial all-reduce contract before model integration: `all_gather_padded(input, output, valid_counts, operation, stream)` uses uniform max slots and explicit valid counts.
- [ ] Test ordered all-reduce/all-gather sequences and differing shapes/counts on CPU external boundaries; wrong metadata aborts before launching.
- [ ] Compile CUDA/ARM64 using a pinned isolated image/toolchain. Keep NCCL headers/runtime/library patches coherent.
- [ ] During the first agreed idle window run the three-rank smoke across all pairs and the group; retain NCCL network evidence and raw errors.
- [ ] Gate MoE integration on a GPU padded-gather256/192/192 reconstruction and global Q8_K block-crossing448 parity test, not only correct all-reduce sums.
- [ ] Stop model integration if transport is incorrect or hangs. Socket tensor fallback does not pass the RoCE criterion.

## Task 5: distributed projection and MoE parity

**Produces:** shard-aware projection/expert calls in `kernels/tp.hpp` consuming
`TensorSlice`, original `ActivationContract`, caller buffers and CUDA stream.
No fixed local FFN/head assumptions hidden in existing global geometry.

- [ ] Add CPU scalar tiny fixtures independent of the planner; test column-sharded gate/up, original SwiGLU, input-sharded down and one global scale/bias application.
- [ ] Add `tests/distributed/tp_expert_parity.cu` against the existing monolithic same-pack expert path, both synthetic and sampled real experts/layers.
- [ ] Implement original-block views first. Quantize global activation once after any required gather; reuse that representation for owned contributions.
- [ ] Compare rank-sum projections and full MoE, including top-k10 routing, shared expert, expert scales and zero/unselected routes.
- [ ] Exercise prefill and decode shapes separately; no claim that decode-only kernels qualify GEMM prefill.
- [ ] Freeze measured component tolerances and accumulator policy before whole-model work; document the single-rank arithmetic control and exact-byte checks.

## Task 6: GDN, QSA, PLE and replicated operations

**Produces:** full TP layer forward with local recurrent/KV state and globally
correct residual. Existing monolithic kernels remain the independent reference.

- [ ] GDN parity: whole key/value group ownership, conv history, 1/4/many token steps and prefill chunk-boundary continuation; compare local slices and reconstructed outputs.
- [ ] Implement local GDN geometry 6/5/5 keys and18/15/15 values in generic kernels; reduce output-projection partials before HC update.
- [ ] QSA parity: query offsets, replicated KV/indexer selection, compressed/raw caches, causal masks and queries crossing the global GQA boundary.
- [ ] Implement QSA output reduction and compare selected indices/scores/cache state, not text alone.
- [ ] Replicate router/norm/HC/PLE/head initially, with one correct global logical result and explicit memory accounting.
- [ ] If memory gate fails, implement PLE/embedding row sharding with hash/global-ID tests and sum-before-downstream semantics; implement head gather/masks only when required.
- [ ] Run component checks on one GPU emulating ranks sequentially, then real three-rank execution. Neither substitutes for the other.

## Task 7: coordinator and C1 whole-model inference

**Produces:** `Worker::execute(const StepCommand&) -> StepResult`; commands carry
session, request/slot, monotonic step, phase, positions, tokens, manifest hash
and expected collective schedule. Results carry status/state digest; tensors
remain on the data channel. The coordinator alone owns sampling/output.

- [ ] Add `test_worker_protocol.cpp`: init/prefill/decode/reset/cancel/abort order, duplicate/stale commands, differing route schedules and dropped peer.
- [ ] Add a separate `strata-distributed` target. Choose a supported non-speculative uncaptured execution path; do not bypass monolithic assertions to disable features.
- [ ] Integrate all48 layers per worker and one stream/C1. Audit every process-shared pointer/callback/CPU expert lane and keep nonlocal access explicit.
- [ ] Test teacher-forced logits and state over multiple decode steps, two consecutive requests, reset and cancellation; reject outputs after any failed collective.
- [ ] Run `compare_model.py` at8K then64K context against same-pack TP1 with identical settings; record logits, near ties, max/mean error and state differences.
- [ ] If numerical-policy limits fail, diagnose component-first; no automatic tolerance widening or production trial.

## Task 8: memory capacity, MTP, graphs and C8

**Produces:** qualified feature combinations and resource budgets, not automatic defaults.

- [ ] Record actual per-rank memory after allocation/warmup/prefill/decode and compare with the budget; bound expert cache and library pools.
- [ ] Prove one useful additional capacity case or state explicitly that only distribution was achieved.
- [ ] Add MTP only after C1 correctness: common drafts, probabilities, acceptance prefix and rollback/reset for GDN/QSA/PLE; preserve statistical sampling semantics.
- [ ] Add fixed-buffer CUDA graph capture/replay and test long repeated operation sequences, timeout poisoning and graph replay after reset.
- [ ] Add deterministic C4/C8 scheduling with mixed prompt lengths, prefill/decode interleave, cancellation and per-request isolation tests.
- [ ] Recompute budgets for MTP/graph pools/C8. Do not infer64K/C8 capacity from8K/C1 or claim256K support without qualification.

## Task 9: experimental server and SparkRun lifecycle

**Produces:** isolated recipe/image/cache, one endpoint and a documented restoration procedure.

- [ ] Inspect actual SparkRun runtime/plugin interfaces before implementing a launch adapter; do not label a Strata binary `vllm-distributed` unless the runtime contract supports it.
- [ ] Add an experimental launch wrapper with rank/world/device0 mapping and private coordinator config; pin image/weights/manifest/library hashes.
- [ ] Implement `serve/distributed_backend.py` to existing OpenAI/Anthropic server semantics with streaming, cancellation and health reflecting all workers.
- [ ] Add endpoint tests for streaming content, error propagation, tool parsing and isolation. Refusal behavior is not inherited from Staticduo21 hooks.
- [ ] Validate recipe/dry-run and stage payloads without touching active21. Preserve original recipes/mods, backups and sufficient disk.
- [ ] During the coordinated window, use SparkRun to stop/launch/restore exact recipes; verify restored production health. No auto-promotion.

## Task 10: performance decision and optional optimization

- [ ] Freeze matrix and `numerical-policy.json`; run matched TP1/TP2/TP3 where supported:4K/16K/64K prompts,256 output tokens,C1/C4/C8,three paired interleaved rounds.
- [ ] Separate queue-inclusive request latency, first content TTFT, prefill, per-request decode and aggregate throughput; record memory, clocks, temperature and per-layer collective time.
- [ ] Report Staticduo21 separately when source quantization differs; do not claim the GGUF matches its numerical quality.
- [ ] Compare against proposed continuation criteria: >=10% C1 decode gain OR >=20% C8 throughput gain, or an accepted capacity case. Record variability and regressions.
- [ ] If collective time dominates, implement companion RoCEnante task6 with license/protocol review and independent parity/timeout/graph tests; otherwise optimize the measured compute bottleneck.
- [ ] Review the integrated milestone once, fix actual blockers and rerun affected checks. Finish with passed/failed/not-run and remaining hardware limits.

## Validation commands and delivery records

Commands naming new targets/scripts above are acceptance commands to implement,
not tools that already exist. Each task must document its final command/options.
Use CTest for the new C++ suite and pytest for Python artifact tools. Keep raw
CPU, compile, local-GPU and three-host results separate. Build all existing
backends that an eventual shared-code change touches before upstream review.

For every milestone retain: exact commit, dirty diff, model/source manifest,
toolchain/library identity, settings, commands, test output and failures in
`/home/staticduo/tmp/strata-tp3-<date>/<milestone>/`. No secrets or private image
credentials in committed documentation or shared artifacts.

Full-model acceptance requires Tasks1-9 plus the Task10 measurement report.
Transport-only or component-only completion must be named as such. A failed
speed criterion is a valid experimental result, not permission to change the
model/quality/defaults silently.
