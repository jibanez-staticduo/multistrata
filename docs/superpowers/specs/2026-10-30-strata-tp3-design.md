# Strata TP3 on three DGX Sparks

Date: 2026-10-30. Status: design for an experiment, not implemented.

## Outcome

Run one Qwen3.8-Flash-Next model with native Strata kernels across DG1, DG2 and
DG3 over their RoCE triangle. All ranks participate within each layer: TP3,
PP1. Establish correctness, useful memory capacity and measured performance
as three separate results. A successful communication test is not a working
distributed model, and a working distributed model is not evidence of speedup.

Jordi requests a complete plan before implementation. Downloading a compatible
GGUF is acceptable; retaining the Staticduo21 checkpoint is optional. This
delivery contains documents only. No production restart, model download,
conversion, dependency installation or GPU experiment is part of this delivery.

## Evidence and limits

Strata base: `870aedd9e1ebe1c91f94da93b59af17729f6c5c0` (0.2.8, engine 0.1.54).
The inspected vLLM checkout is `5423fb8b93a539b24e7f80075d132d4f4dcab95b`.
The running vLLM fork reports `20260926.dev28+g884b4ff68.d20261002`; it is not
the newly cloned upstream tree.

Read-only inspection of DG1 confirmed SparkRun 0.4.1, cluster `cx7`, three
running containers, TP3/PP1, and this active recipe:
`/home/staticduo/recipes/triple/qwen3.8-flash-next-staticduo21-tp3.yaml`.
That recipe uses tensor-sharded experts, not expert parallelism. Each rank
holds part of all 512 experts. The checkpoint has QSA 24/6 heads, GDN 18/54
heads and FFN width 768. Its B12X/RoCEnante path handles eligible collectives;
NCCL handles others. A pinned patched NCCL fixes peer interface advertisement.
No network benchmark, GPU smoke or new inference was performed for this plan.

Current Strata has Qwen pipeline stages inside one process, backed by mapped
host buffers (`include/strata/core/verify.hpp:179`). DeepSeek has local TP
through mapped host memory (`src/dsv41/ds4_model.hpp:56`); that is not Qwen TP
or cross-host communication. No native multihost Strata runtime was found in
the inspected engine, CLI and server tree. Public PR searches found proposals
and local GPU work, but no reviewed implementation of this target.

## Approaches and decision

| Approach | Benefit | Cost / decision |
| --- | --- | --- |
| Native Strata TP3, original model geometry | Keeps Strata packs/kernels and addresses single-request parallel compute | Recommended; new shard-aware kernels and distributed state |
| Native Strata PP3 by layer | Reuses existing stage boundary | Capacity/throughput alternative; not the requested TP3 MVP |
| Staticduo21 safetensors as Strata input | Reuses prepared TP3 dimensions | Different quantization/layout; requires new loader and kernel geometry |
| Strata front end with vLLM backend | Reuses existing distributed serving | Changes the execution engine; not native Strata TP3 |

Decision proposed by the planner: preserve the original model geometry and
weight codes, even when partitions are unequal. Use NCCL first to establish
correctness. Port the small-collective RoCEnante optimization only after
profiling proves its benefit. No Ray dependency is needed for the native
worker design; SparkRun manages processes and NCCL moves tensor payloads.

## Model and artifact policy

First candidate: ISTA-DASLab original IQ3_S, provided its actual per-tensor
types admit the shard layout. Inspect headers before downloading full weights.
Use `tools/quantscope.py`, `tools/strata_inspect.py` and `tools/iq_pack.py` as
existing references; a pack name alone does not define every projection's type.
Pin repository revision, file names, sizes and content hashes in the run record.

If IQ3_S requires excessive cross-rank quantization work, evaluate a supported
NVFP4 pack. Honor ISTA-DASLab -> Unsloth -> Gyro -> others where compatible;
explain a format-driven exception. Gyro is not the default fallback for ARM.
Do not download every candidate. Do not treat different packs as a matched
speed or quality comparison.

Qwen currently consumes a Strata pack, commonly GGUF plus expert/draft files;
it does not directly load arbitrary safetensors. Existing ModelOpt conversion
tools can preserve NVFP4 expert codes/scales, but convert some dense weights
to BF16/Q8_0 and do not solve the Staticduo21 geometry difference. Reusing
those weights is a later input adapter, not a prerequisite for this experiment.

Keep originals immutable. New per-rank artifacts contain a versioned manifest,
source hashes, global geometry, local ownership, tensor formats, byte ranges,
strides, scale references and logical vocabulary. Prefer original file views
for the first implementation; materialize rank-local files only after parity.
No weight requantization or 16->18 / 640->768 model transformation in the MVP.

## Partition contract for the original Qwen geometry

Validate every dimension against the selected file; these values come from
`include/strata/core/geometry.hpp` and are not permission to ignore metadata.

| Component | Global | Proposed rank 0 / 1 / 2 ownership |
| --- | --- | --- |
| Layers | 48 | All 48 on every rank; never consecutive thirds |
| GDN key heads | 16 | 6 / 5 / 5, starts 0 / 6 / 11 |
| GDN value heads | 48 | 18 / 15 / 15, starts 0 / 18 / 33 |
| QSA query heads | 24 | 8 / 8 / 8, starts 0 / 8 / 16 |
| QSA KV heads | 2 | Replicate both on each rank initially |
| QSA indexer heads | 4 | Replicate initially; preserve global selection |
| Routed and shared FFN width | 640 | 256 / 192 / 192, starts 0 / 256 / 448 |
| Routed experts | 512 | All expert IDs; matrices sharded along FFN |
| Router, norms, HC | Original | Replicate, same route and expert weights |
| PLE, embedding, output head | Original | Replicate first if budget fits; see memory gate |

Keep each GDN key head with its three value heads, including convolution,
recurrent state and rollback. In QSA, an eight-query range can intersect a
global GQA group boundary; use original global head mapping into the replicated
KV heads, not a local `query_count / kv_count` formula.

Gate/up ownership slices output rows; their quantized input remains width
2560. Down/output ownership slices input columns and produces a full-width
partial result, reduced once before the next replicated operation. Preserve
original bias, nonlinearities, global scales and shared-expert weighting;
never add a replicated bias once per rank before a sum.

### Quantization boundaries

Per-tensor planning must distinguish weight blocks from activation blocks.
IQ3 gate/up rows can be copied intact when their output rows are partitioned;
the IQ3 block of 256 along input width 2560 is not itself a reason to reject
TP3. Down projections can have other types, including blocks of 32 or 64.
Offsets 256 and 448 preserve such weight blocks. Record the exact types first.

Activation quantization may span a rank boundary even when weight blocks do
not. Norms, activation maxima/scales, Hadamard transforms and nonlinearities
cannot be recomputed independently on arbitrary local slices. The conservative
reference path all-gathers the original logical activation, runs the original
global quantizer/transform, then computes only each rank's owned contribution.
Keep whole overlapping quant blocks as read-only halos and mask unowned lanes
where necessary. Copy original codes and scales; do not requantize a partial
block to make it fit. Optimize this exchange only after parity and profiling.

This reference path needs all-gather in addition to all-reduce. NCCL counts
must match: unequal shards use an explicit max-sized padded wire slot and
valid counts, never unequal counts passed to a uniform collective. Padding is
transport/workspace only and must not change the logical model or stored codes.

## Native runtime

Use one CUDA process per host, local device 0, ranks 0/1/2. Rank 0 owns the
single HTTP endpoint, tokenization, request scheduling and sampling. Every
worker executes the same step/slot/layer order and holds its own local state.
All ranks validate model manifest, shard map, protocol and compute policy.

Separate the private control protocol from tensor traffic. Start with one
request, one stream, one operation in flight, greedy sampling, no speculative
window and no CUDA Graph capture in the new path. Keep existing single-host
engines and defaults untouched. Use a separate experimental binary/build.

Support initialization, prefill chunks, decode, commit, reset, cancel at safe
boundaries and group abort. Tokens and routing decisions must not diverge
between ranks. Cancellation is a scheduled command; rank 0 cannot return early
while peers still expect a collective. A failed step never publishes output.
On worker loss/timeout, abort the communicator, reject the request and discard
all session state. No transparent retry using partly committed KV/GDN state.

MTP is later work: rank 0 chooses drafts, every rank verifies the same window,
and all commit the same accepted prefix. Checkpoint/restore must include GDN,
QSA/indexer caches and PLE history. Graph replay requires fixed buffers and
identical collective sequence; prepare allocations and communicators first.

## Transport

The companion transport plan defines a separately built NCCL C API backend,
bounded initialization, completion/error polling and a three-rank smoke test.
Require evidence of RoCE/IB tensor traffic; bootstrap TCP is expected. Do not
accept a silent TCP tensor fallback as a successful RoCE test.

Preserve the active NCCL library and firmware/network configuration. Build or
reuse a separate provenance-checked library; the active patched build's version
alone does not establish patch equivalence. Discover HCAs and GIDs per host/port.
The inspected RoCEnante proxy is native C but its GPU collective is CuTe; a
port needs both protocol and GPU reduction, not only the proxy. Check licenses
before copying any implementation. The Sparknet snapshot is not the same build.

## Memory, disk and availability

Produce a per-rank budget before any full-model load: expert and dense shards,
replicated tensors/PLE, logical KV and indexer state, recurrent state, prefill
scratch, global-activation gathers, communication buffers, graph pools, draft
weights and process/library overhead. Count GB10 unified memory once. Include
page cache/mmap residency and temporary conversion disk separately.

Start at 8K context/C1 for correctness, then 64K. Replicated PLE/head may be
too expensive; if so, implement vocabulary/PLE row ownership before progressing
to larger contexts. Row sharding contributes zeros for absent IDs then sums;
preserve hash functions, global IDs and logical-vocabulary masks. Do not silently
substitute another table or shorten the declared test to make a load succeed.

Maintain at least 50 GiB free disk on each Spark. Store immutable source/backups
on the NAS and only selected rank artifacts locally. Stage downloads to a
temporary sibling and validate before publishing a manifest. Preserve all
production recipes/mods/models and avoid large temporary files in `/tmp`.

GPU/network trials need a coordinated idle window. Use SparkRun for lifecycle,
an isolated recipe/image/cache/port and a tested restoration command for the
current production recipe. Planning and CPU tests can proceed without that
window. No automatic production promotion is authorized.

## Acceptance and measurement

1. Artifact checks: source blocks/scales reconstruct byte-identically; tensor
   ownership covers the original without gaps/double contributions; reject
   wrong shape, codec, missing scale, incompatible shard or source hash.
2. Component checks: compare sharded projections, expert SwiGLU, GDN recurrence,
   QSA selected keys and layer outputs against the monolithic same-pack path.
   Include scalar/tiny synthetic calculations independent of the new planner.
3. Numerical gates: FP32 reduction of exact integer probes must be exact.
   Use FP32 accumulation for the baseline model reductions. Before integration,
   freeze per-component/logit absolute, relative and normalized error limits
   from an independently computed reference and local emulated-rank runs.
   Record actual limits in `numerical-policy.json`; empty limits fail the gate.
   Do not relax a limit automatically after a failure. No NaN/Inf is accepted.
4. Whole model: teacher-forced logits plus long multi-step state comparisons,
   greedy outputs, reset/cancel/rollback, isolation and tool parser smoke.
   Greedy text alone is insufficient; near-tie divergences need measured logits.
5. Capacity: document a useful context/pack/concurrency configuration that fits
   TP3 but not the same-pack single-node budget. Otherwise report only correct
   distribution, without claiming additional useful capacity.
6. Performance: same revision/pack/settings on 1/2/3 nodes where supported;
   prompts 4K/16K/64K, output cap 256, C1/C4/C8. Freeze thinking, MTP, residency,
   KV type, clocks and cache policy. Three interleaved paired rounds; retain raw
   results, failures, queue time and thermal/memory telemetry. Respect C8 ceiling
   and existing 2200 MHz policy without disabling firmware protections.
7. Report first actual streamed content TTFT, prefill, per-request decode,
   aggregate throughput and p50/p95 separately; never time the empty SSE role
   event as TTFT. Include per-layer compute/collective wait. Staticduo21 is a
   separate product comparison if weights/quantization differ.

A suggested performance continuation threshold is a reproducible >=10% C1
decode gain OR >=20% C8 aggregate gain against matched TP1, or the accepted
capacity result. These are proposed go/no-go thresholds, not measured claims.
If none holds, report results and choose whether to continue optimization;
do not call the project successful at speed solely because TP3 runs.

## Stages, effort and review

| Stage | Deliverable | Indicative senior engineer effort |
| --- | --- | --- |
| A | Header inventory, shard math, memory budget, transport smoke | 1-2 weeks |
| B | Native quantized component TP and local emulated-rank parity | 3-5 additional weeks |
| C | Three-worker model, 8K/64K C1 correctness | 3-5 additional weeks |
| D | Memory capacity, MTP/graphs, C4/C8, lifecycle/failure handling | 3-6 additional weeks |
| E | Optional RoCEnante and measured tuning | 2-4 additional weeks |

Total planning band: roughly 3-6 months, potentially longer if original quant
layouts or model state need substantial kernel work. Parallel ownership can
overlap transport and artifact work; sequential numerical gates remain. No
guaranteed speedup or delivery date. DeepSeek and other backends are follow-up
projects, not hidden requirements of the first Qwen/CUDA/GB10 implementation.

Use one adversarial review of the design and one proportional review per
completed implementation milestone. Fix blocking defects and rerun relevant
checks; avoid indefinite audit loops. Upstream contribution needs maintainer
scope agreement before this large feature, separate from a private experiment.
Build every touched backend before requesting upstream review; missing hardware
validation must remain explicit. Do not open issues/PRs from this plan alone.

## Source map

- `docs/DGX_SPARK_LINUX.md`: one-Spark support, memory and qualification limits.
- `docs/MULTI_GPU.md`, `docs/BATCHING.md`: current local stage/state behavior.
- `include/strata/core/geometry.hpp`, `src/kernels/cpu/native_expert.cpp`:
  fixed geometry, weight and activation quantization contracts.
- `tools/iq_pack.py`, `tools/nvfp4_convert.py`, `tools/nvfp4_local_pack.py`:
  artifact layouts and conversion limits.
- `https://docs.vllm.ai/en/stable/serving/parallelism_scaling/` and inspected
  vLLM `linear.py`, `parallel_state.py`, `pynccl.py`: architectural precedent.
- `https://build.nvidia.com/spark/nccl` and NCCL user guide: external transport.
- DG1 active recipe and installed Qwen/B12X code: deployment reference only;
  sanitized read-only model extracts under
  `/home/staticduo/tmp/staticduo21-model-readonly`.
