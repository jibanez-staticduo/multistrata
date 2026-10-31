# Strata NVFP4 TP3 Final Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans or subagent-driven-development. Implement the final architecture; components are parallel work units, not user-facing intermediate variants.

**Goal:** Native NVFP4 Strata TP3 reaches approximately Staticduo21 TP3 throughput with comparable context and concurrency, and remains serving with DG1 `strata monitor` working.

**Architecture:** Resident NVFP4 row/column shards execute through Strata's fast Prefill and Verifier kernels. GPU route planning, local expert/mixer outputs and two output reductions per layer replace exact-lane reconstruction; large prompt chunks and real slot-row batches share weight reads.

**Tech Stack:** C++20, CUDA13/GB10 sm121a, Strata MMQ and native NVFP4 kernels, cuBLAS dense GEMM, NCCL/RoCE, SparkRun, existing Strata server/Terminal.

**Design:** [Final performance design](../specs/2026-10-31-strata-nvfp4-tp3-design.md).

## Acceptance And Stop Rule

The target is >=95% of same-window Staticduo21 prefill and decode/aggregate
throughput for selected C1/C4/C8 PP2048/TG128 at depths 0/65535, with max context
262144 and one actual maximum-context capacity check. Preserve model bytes and
report checkpoint/quality differences. Freeze numerical/functional budgets before
testing. No default changes, no claimed success from the IQ3 oracle baseline, no
performance acceptance based on compile flags or a single fast request. If a cell
fails, profile that bottleneck and correct it; do not repeat a large slow matrix.

## Ownership And File Map

| Unit | Responsibility / files |
| --- | --- |
| Parent | Shared hook contracts, CMake, final driver/server wiring, transport schedule, integration and measured acceptance |
| NVFP4 kernel worker | New `include/strata/distributed/nvfp4_experts.hpp`, `src/distributed/nvfp4_experts.cpp`, `include/strata/kernels/tp_nvfp4.hpp`, `src/kernels/cuda/tp/nvfp4.cu`, focused CUDA tests |
| Fast prompt worker | `include/strata/prefill/prefill.hpp`, `src/prefill/prefill.cpp`: distributed callback integration that retains large-chunk GEMM/GDN/QSA and bypasses local full-expert loading |
| Fast decode/batch worker | `include/strata/core/verify.hpp`, `src/core/verify.cpp`: distributed hooks before state/residual consumption; slot-row batches and graphs use the same path |
| Mixer worker | Dedicated `include/strata/distributed/fast_mixer.hpp`, `src/distributed/fast_mixer.cpp`, CUDA helpers for local head projections and output reduction; no exact lane reconstruction |
| Storage worker | IQ3 archive and checksums in NAS only; no deletion until verified and dependencies reviewed |

Before editing shared files the parent freezes callback signatures. Workers do
not edit another unit's files; communicate integration needs. Parent validates
all reports and exact tested source hashes.

## Work Units

### 1. Assets, Target Baseline And Host CLI

- [x] Create active Goal with actual Staticduo21 parity target and final Strata service requirement.
- [x] Capture immutable Staticduo21 recipe and historical per-cell anchors, max context and MTP settings.
- [x] Select/pin compatible NVFP4 GPTQ pack; start NAS download and preserve upstream SHA256SUMS.
- [ ] Verify complete NVFP4 assets and stage rank-local files without changing codes/scales.
- [ ] Archive IQ3 GGUFs/pack with SHA checks, then remove only verified retired local assets.
- [x] Install owner-local DG1 CLI/monitor dependencies and `strata` on PATH; help works.
- [ ] Verify monitor against final real TP3 service, including context and active ranks.

### 2. Freeze Final Execution Contracts

- [ ] Define GPU-only expert callback with inputs, route IDs/weights, rows and output accumulation semantics.
- [ ] Define prompt and verify hooks, slot-state ownership and mixer output reduction semantics.
- [ ] Make compute policy and numerical tolerance explicit; keep exact-reference code for diagnostics only.
- [ ] Define graph batch keys, prefill chunk capacity, memory pages and bounded error/commit semantics.
- [ ] Validate contract tests with synthetic independent references before full-model GPU use.

### 3. NVFP4 Resident Shards And GPU Experts

- [ ] Load experts.bin according to v3 metadata, slice gate/up output rows and down input blocks, preserve tail scales and hashes.
- [ ] Reuse tuned native grouped NVFP4 for decode; GPU route grouping and local SwiGLU/down without hidden/lane gathers.
- [ ] Apply route weights and local expert sum, one uniform AllReduce of final rows per layer.
- [ ] Prefill uses expert-major MMQ FP4 tensor cores on sm121a; distinguish fallback and prove actual dispatch.
- [ ] Targeted original-scale component tests: distinct rows/IDs, tails, unequal cuts, partial sums and invalid ID failure; numerical budgets fixed in advance.

### 4. Fast Dense/Mixer And Prompt Integration

- [ ] Integrate TP into existing large-chunk Prefill; dense GEMM, HC fusion, chunked GDN, batched sparse QSA remain enabled where compatible.
- [ ] Shard mixer output input columns and compute local partials; reduce outputs directly, with no full rank-zero output projection.
- [ ] Preserve GDN head/state mapping, QSA GQA/global mapping, KV and PLE history; no silent context clipping.
- [ ] Check short/long continuation and reset at changed state seams using same-pack reference, not bitexact across different algorithms.

### 5. Decode Graphs, Continuous Batch And MTP

- [ ] Integrate fast TP hooks into Verifier's real slot-row batched execution, avoiding full per-slot loops.
- [ ] Capture fixed-address compute+NCCL per batch shape; one rank process each, one step-level agreement, GPU route validity.
- [ ] Chunk prefill admission and decode scheduling, fair active rows, explicit max-context/page budget and queueing.
- [ ] Connect MTP target verification and rollback to fast executor with measured draft acceptance/cost; preserve correct target distribution.
- [ ] Directed isolation/cancel/reset/recapture/failure checks, one bounded check per new behavior.

### 6. Final Candidate And Delivery

- [ ] Independent scoped adversarial review; fix concrete blocking findings and re-run relevant checks.
- [ ] ARM CUDA build and actual 3-host component/model execution with exact source/asset pins.
- [ ] Short same-window Staticduo21 comparison PP2048/TG128 C1/C4/C8 at selected depths; measure client TTFT, prefill, visible decode and aggregate throughput separately.
- [ ] Profile only failing target cells; correct dominant kernel/transport/scheduler costs until acceptance or an evidenced external blocker.
- [ ] Actual max-context capacity check, numerical/quality suite and functional APIs.
- [ ] Leave Strata NVFP4 TP3 serving with explicit runnable profile, DG1 CLI and working monitor; Staticduo21 remains stopped.
- [ ] Report per-cell values, limits, failed attempts and quality differences. Mark Goal complete only if the acceptance contract is met.

## Evidence And Recovery

Evidence root: `/home/staticduo/tmp/strata-nvfp4-tp3`.
NAS pack: `/volume1/models/huggingface/local-models/Strata-Qwen3.8-Flash-Next-NVFP4-GPTQ`.
Do not alter the original Staticduo21 recipe or checkpoints. SparkRun owns container
lifecycle. The user authorizes stopping it; normal final state is Strata, not
restoration. Preserve >=50GiB free storage per host and existing clock protections.
