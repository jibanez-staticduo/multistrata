# Native TP3 experimental acceptance record

**Recorded 31 October 2026, including actual 8K/64K gates. Performance qualification is in progress.**
Native Strata runs all 48 layers of the original Qwen pack across DG1, DG2 and
DG3. Short C1 checks match the independent TP1 reference bit for bit. The full
plan is still open: useful extra capacity, long feature combinations and
matched performance rounds remain unqualified. The actual 64K prompt and
256-token continuation pass the full model comparison. Short and extended sampled
MTP state and C1 full-model graphs pass the scoped checks recorded below. There is no speed claim
against TP1 or vLLM, and no production recommendation.

The [implementation plan](superpowers/plans/2026-10-30-strata-tp3-implementation.md),
[design](superpowers/specs/2026-10-30-strata-tp3-design.md) and
[transport plan](superpowers/plans/2026-10-30-strata-tp3-transport.md) are historical
planning records. Their original checkboxes are preserved. This document records
later evidence without closing those plans. The latest native milestone is
`077f8a83`; HTTP phase propagation is `697ff515`. Each
window's rank configuration pins its actual binary, source and library hashes.
This record describes completed experiments, not current service health.

## Model, hardware and execution scope

The source is `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`, IQ3_S, revision
`ed59f92082b1e93c0e96d60a8b11aab089b52f09`. IQ3_S is the pack name; individual
expert tensors use mixed native codecs. Original weight codes and geometry are
preserved. The [source lock][lock] records both GGUF sizes and SHA-256 hashes;
the [inventory][inventory] records 48 layers, hidden width 2560, FFN width 640,
512 experts and top-k 10.

Runs use three DGX Sparks (GB10, CUDA/ARM64), one CUDA process per host, TP3/PP1.
The [window 7 NCCL log][nccl] records NCCL 2.30.7 with CUDA 13.0 and `NET/IB`
RoCE devices. This is native Strata execution, not a vLLM model worker.

| Geometry or policy | Recorded implementation |
| --- | --- |
| GDN | Original 16 key / 48 value heads; key groups 6/5/5 and value groups 18/15/15 |
| Routed FFN | Original 640 channels, owned as 256/192/192 |
| QSA | Original 24 query / 2 KV heads; global GQA mapping; both KV heads and indexer4 replicated |
| PLE | Replicated assets: 26.883 GiB per rank; count unified memory once |
| Expert activation | Native CUDA global Q8_1, blocks of 32; distinct from CPU Q8_K contracts |
| Exact down projection | `ExactLaneReference` reconstructs native FMA order from 128 operands per output |

The exact down path adds about 629 MB per rank per token across 48 layers and
10 routes as a summed logical operand-buffer extent. This is neither simultaneous
resident allocation nor measured network traffic. Replication, graph pools and
state buffers must be included in capacity budgets; estimates do not prove fit.
See [exact-lane rationale][lane] and the [expert runtime](../src/distributed/expert_model.cpp).
Linux PSS, CUDA allocations and driver reports overlap on unified memory and
must not be added as independent physical memory pools.

All distributed paths are experimental and opt-in. In
[the executable](../src/program/distributed.cpp), expert collectives default to
`sequential`; `--expert-collectives batched` is explicit. `--graph-finish` defaults
to 0. MTP is off without `--mtp-path`. C8 uses the separate `multislot` mode.
No host driver, firmware or original weight change is part of this record.

## Acceptance by implementation task

“Passed scope” means only the evidence named here. “Partial” leaves the task's
remaining acceptance checks open.

| Task | Status | Evidence and remaining scope |
| --- | --- | --- |
| 1. Source and inventory | Passed scope | [Pinned source lock][lock] and tensor inventory; no general model compatibility claim. |
| 2. Ownership and quantization | Passed scope | [Real TP3 plan][plan], original geometry and global activation contracts; TP1/TP2 plans are controls, not hardware acceptance of every world size. |
| 3. Rank artifacts | Passed scope | [Materialized byte check][bytes] and source hashes; immutable original blocks/scales retained. |
| 4. Native transport | Passed scope | [Window 1 report][transport], mismatch/missing/killed-peer receipts and later `NET/IB` logs; includes unequal padded gathers. |
| 5. Projection and MoE | Partial | Exact expert gate and short whole-model parity pass after arithmetic fixes; short decode evidence does not qualify GEMM prefill. |
| 6. GDN, QSA and PLE | Passed scope | All 48 layers represented in C1 state/trace checks; short 2/4/8 chunk-boundary and RESET parity now pass. The later actual 64K continuation passes as recorded below. |
| 7. Coordinator and C1 | Passed scope | Windows 4/6/7 exact logits, state, RESET and traces; 8192 is configured capacity, not an 8K-token prompt. The later actual 8K and 64K gates are recorded below. |
| 8. Capacity, MTP, graphs, C8 | Partial | Short MTP/C4/C8 full-state replay and cancellation checks pass. Full C1 NCCL graphs pass 255 replays/RESET and bounded peer loss. Phase snapshots exist; useful extra capacity and longer feature qualification remain open. |
| 9. Server and lifecycle | Partial | OpenAI/Anthropic streaming, tools, cancellation and failure-health artifacts; recorded restoration checks through window 7. No automatic rollout. |
| 10. Performance decision | Partial | Three matched 4K C1 pairs show a -61.9% median decode-rate change for TP3. C4/C8 and 16K/64K rounds remain open. |

## Matched Performance Progress

The frozen matrix is running with the same original IQ3_S pack, context 65792,
resident FP16 KV, thinking disabled, MTP off and graphs off. Three rotating
paired rounds of the 4096-token C1 case have finished. Every request produces
256 output tokens and passes the HTTP/usage checks.

| Native phase | TP1, three rounds | TP3, three rounds |
| --- | --- | --- |
| Prefill seconds | 300.5406 / 305.3279 / 296.9001 | 870.5851 / 870.8654 / 869.6152 |
| Decode seconds | 21.6196 / 21.7765 / 21.7972 | 56.8171 / 56.8002 / 57.2105 |

The median paired change in native decode rate is **-61.9%** for TP3. This case
does not meet the proposed 10% C1 improvement threshold. Median request wall
time is 322.261 seconds for TP1 and 939.525 seconds for TP3. Correct state
reconstruction does not imply a performance gain.

The first TP3 C4 4096-token round also passes: four requests each produce 256
output tokens. The concurrent round takes 4030.110 seconds and returns an
aggregate 0.2541 output tokens/s. Per-request first-content TTFT is
3771.771-3780.399 seconds. Native per-request prefill is 873.922-876.716 seconds
and decode is 56.201-58.845 seconds. These native active times exclude waiting
between serialized slots; client wall time includes it. The small reported
admission queue time is not total scheduler waiting. One C4 round does not
establish its variability or a capacity advantage.

The remaining C4/C8 cases and the 16K/64K rounds are in progress. TP2 and
TP1 C4/C8 are unsupported by this candidate and are explicitly not run. No
aggregate C8 gain against TP1 can be claimed without that control. Raw partial
records and the analysis are in
`/home/staticduo/tmp/strata-tp3-20261031/benchmark-qualified/execution-checkpoint.json`
and
`/home/staticduo/tmp/strata-tp3-20261031/benchmark-qualified/PARTIAL.md`.
The first campaign stops before measurement because its lifecycle hook reads
the wrong ID field for a solo SparkRun workload. That attempt is retained in
`failed-campaign-solo-id`; the corrected campaign starts all three rounds anew.
These are partial results, not completion of Task 10 or a production recommendation.

## Passed measurements

Windows [4][q4], [6][q6] and [7][q7] independently compare C1 against the same-pack
monolithic TP1 reference. Each uses context capacity 8192 and four teacher-forced
positions, repeated after RESET: **eight logit rows total**. All 248320 logits
per row agree bit for bit, with no widened tolerance. Each report records 2070
state comparisons, 1728 trace comparisons and 1964 RESET comparisons, all with
zero differing bits. Logical TP1 state coverage is 978 arrays; extra comparisons
check replicas and reset identity. States are saved at reset and positions 0/3;
traces cover six stages of all 48 layers at position 0. Three process receipts
pass in each report. Window 6 adds batched expert collectives; window 7 adds
partial finish graphs.

| Experiment | Measured result and limit |
| --- | --- |
| Window 5 API | OpenAI plain/stream and Anthropic native streaming return HTTP 200. After client disconnect, health is ready when checked after 5 seconds; this is not a precise cancellation-latency measurement. [API artifacts][api5] |
| Window 6 service | Three chat requests and OpenAI stream receipts are retained; the tool request returns HTTP 200 after the window 5 failure. [Service artifacts][api6] |
| Window 7 MTP, four drafts | Greedy response: 14 output tokens in 21.399 s. Fixed-seed sampled 16-token text repeats and matches the MTP-off control; greedy text also matches. This qualifies HTTP text/usage only, not full internal state or the sampling distribution. [HTTP parity][mtp] |
| Window 7 C8 | Eight mixed short requests return 200 in 148.798 s wall time; repeated prompts give identical text. Four concurrent requests with one client cancelled leave the other three successful under the **eight-slot pool**. This is not a separate configured four-slot test or state-isolation proof. [C8 receipts][c8] |
| Window 7 graph component | Two finish codecs, 100 replays per codec, exact bits on one GB10. Whole-model graph capture is not qualified. [Graph summary][graphs] |
| Window 8 resident TP1 | All routed experts resident: 50,292,326,400 owned weight bytes. Retry 2 processes eight token rows and 978 state arrays with zero differences from the earlier independent selected-expert TP1. This establishes a candidate control, not a fair speed comparison. [Parity][resident] |

Window 8 rank 0 profiling covers eight token rows: 1536/1536 completed
collectives and 384 route calls. Summing the raw nanosecond CSV gives 3.675 s
in metadata stages (staging, operation agreement and route agreement), 0.157 s
enqueue, 1.712 s wait and 0.015 s route readback. These are host timing regions,
not GPU kernel duration, request latency or wire bandwidth. The JSON summary's
`*_ns` fields contain seconds; use the [raw CSV][profile] for units. Input/output
byte counters describe collective buffer extents, not measured link traffic.

CPU records on 31 October: [c8-final-python.log][python] reports **372 passed,
2 skipped**; [resident-cpu-test.log][ctest] reports **18/18 CTest passed**. These
are named historical runs, not a claim that all later edits have been retested.

## Later 8K prompt qualification

The frozen zero-error policy was written before the long run. The resident TP1
reference consumed all 8192 native input IDs and exited 0. TP3 then consumed the
same 8192 IDs on all three hosts; all three receipts report exit 0. Full-vocabulary
logits at positions 0 and 8191 and 342 logical state arrays compare with zero
absolute/relative error. `compare_model.py` reports `qualified: true`,
`full_model_pass: true` and `state_validation: passed` in
[the long-run receipt](/home/staticduo/tmp/strata-tp3-20261031/long-context/8192/comparison.json).
This is an actual 8K prompt check, rather than allocated context alone. It does
not qualify 64K or the performance matrix. TP3 process wall time was about
2385 seconds including loading and evidence capture; this is not a decode rate.

## Later 64K reference and compact model checks

The all-resident TP1 GPU reference consumes **65536 actual prompt IDs plus 256
teacher-forced continuation IDs**, at capacity 65792, and exits 0. Its log contains
all consecutive positions 0 through 65791. Full-vocabulary logits and 171
logical state arrays are retained at positions 0, 8191, 65535 and 65791 in
[the reference receipt][longreference]. This proves the reference executed that
input; TP3 qualification of the same long input is still pending.

The compact TP3 short whole-model run exits 0 on all three hosts. Against the
unchanged original reference, eight logit rows, 2070 state comparisons, 1728
traces and 1964 reset comparisons have zero differing bits. The
[independent compact receipt][compactmodel] qualifies this short C1 case.
It does not establish long-input correctness or a performance gain.

## Later component and reference service checks

The optional `--expert-operands compact` path keeps the original FMA operands
and their order, while omitting unowned zero operands from the wire buffers.
For FF640 its per-output payload is 32 floats for IQ4_NL and 16 for Q2_0,
compared with 128 in the default `full` path. These are logical buffer sizes,
not measured link bandwidth. On DG3, 75 component cases include 375 compact
comparisons with zero differing bits, including ten-expert batches and a real
GGUF expert. [The component receipt][compact] records the original scalar
tolerance and the first stricter scalar attempt separately. Whole-model
compact qualification is still pending.

Both codecs and both representations pass 100 partial finish-graph replays
each on DG3, including reset, terminal checks and compact-versus-full bit
comparisons. [The final graph log][compactgraphs] does not qualify a graph of
the full model or collective transport.

The all-resident world-one reference service runs through SparkRun on DG2.
Health reports `loaded: true`, `world: 1`, `alive_ranks: [0]` and capacity 65792.
A plain request returns `OK`; a streamed request reports 18 prompt and 14
completion tokens, with first actual content at 1.210 seconds. These are
[functional checks][referencehttp], not matched performance measurements.
The first launch stalled inside the head's pageable `cudaMemcpy`; a distributed
build now uses bounded pinned staging. An isolated copy of the same tensor
succeeds with both methods, so the driver's underlying cause remains unknown.
The successful launch also advises away clean verified model file-cache pages
after hashing; CUDA's resident-expert budget check remains enforced.

## Exact native prefill chunks

The opt-in layer-major prefill path groups GDN projections through the native
multi-column MMVQ kernels while advancing recurrence, QSA and PLE causally.
The first size-eight whole-model test processes variable chunks including
8, 1, 7 and a final 8+2 remainder, then repeats the first chunk after RESET.
All three ranks exit 0. [The size-eight receipt][prefill8] records nine exact
full-vocabulary boundary logit rows and 3915 exact state comparisons against
TP1. This is a short correctness gate, not a GEMM or speed result. Size two also passes twelve exact logit rows and 5022 state comparisons;
size four passes ten rows and 4284 state comparisons. All three-rank receipts
exit 0. The C1 service also runs the opt-in chunk path and returns measured host-phase
prefill/decode timings. Its same-ID full-state comparison also passes: 24 consumed IDs, all 171 logical
state arrays and the final full-vocabulary logits agree exactly with TP1.
[The service chunk receipt][servicebatch8] qualifies this short request only.

## Routed prefill rows

The separate `--prefill-expert-batch 1` option groups selected expert work across
prompt rows, preserving each row's router weights and the exact compact down
operands. It requires native multi-token prefill, batched expert collectives and
compact operands. Extra provider and model buffers are budgeted before allocation;
this option remains off by default.

Its first size-eight test processes the same variable chunks and RESET replay
as the native prefill control. All three ranks exit 0. [The expert-row receipt][expertrows]
reports nine exact full-vocabulary boundary rows and 3915 state comparisons with
zero differing bits. This short gate does not establish a performance improvement.

## Native RoCEnante transport checks

The separate opt-in native CUDA/C proxy transport passes a three-host smoke:
each rank checks 18 FP32/BF16 reductions, two padded gathers and 100 graph
replays with three operation sizes, then exits 0. This uses the existing
private authenticated roster and direct RoCE links with an IPv4 RoCEv2 GID
selected independently for each HCA. [The smoke receipt][rocesmoke] records
its limits: payload bytes must be divisible by 16, and these runs overlap a
correctness run, so their timings do not establish performance.

A missing-rank injection ends both joining ranks within 10.4 seconds. Killing
a peer after operation agreement ends the other processes within 14.3 seconds,
with no passing result. [Fault receipts][rocefaults] cover eager standalone
failure; a graph peer-loss injection also ends the surviving ranks within 12.4 seconds
from process start, without accepting output. This is a standalone graph fault
check, not a long serving soak.
The first attempts failed before payload execution because of GID selection,
an old staged binary, or the follower coordinator address. They are retained.
NCCL remains the default model transport. The hybrid model path is opt-in and
keeps reductions on NCCL. Its short full-model comparison now passes: all three
ranks exit 0, with eight exact logit rows, 2070 state comparisons, 1728 traces
and 1964 reset comparisons. [The hybrid receipt][hybridmodel] qualifies this
short case only; long prompts and performance remain pending.

## Full-model graph progress

The experimental `--model-graph 1` captures the complete single-token GPU
forward, including 48 layers, the output head and NCCL collectives. Token
embedding and PLE rows are prepared before replay; QSA reads persistent position
buffers. The graph fixes the stream, context and position base. This first mode
requires C1, compact device-routed experts and NCCL, and excludes MTP,
multi-row prefill, trace observers and profiling. Defaults remain graph-off.

The first candidate fails before replay because the NCCL capture session was
not transitioned into capture before its first enqueue. That failed attempt is
retained. After the explicit capture-start check, all three hosts exit zero:
4703 graph nodes, 240 NCCL collectives and seven replays including RESET.
Eight full-vocabulary rows, 2070 state comparisons and 1964 reset comparisons
match TP1 bitwise under the unchanged numerical policy. Evidence:
`/home/staticduo/tmp/strata-tp3-20261031/full-model-graph-retry/independent-qualification.json`.
A subsequent pair of 128-token sequences performs 255 replays across RESET.
All ranks exit zero; fourteen vocabulary rows and 5760 state comparisons are
bitwise equal to TP1. Evidence is in
`/home/staticduo/tmp/strata-tp3-20261031/full-model-graph-repeat2/independent-qualification.json`.
The readback-lifetime review fix also passes the same 255-replay test with
fourteen logits and 5760 state comparisons bitwise equal. The first model
peer-loss injection exposes a real hang in ncclCommAbort, confirmed by GDB;
the external 300-second watchdog kills the survivors. An independent host
deadline and immediate process exit before failed-graph cleanup address that
blocking call. The repeated fault test exits the surviving ranks with code 86
approximately 15 seconds after the injected SIGKILL, using the agreed
15000 ms operation deadline. Its receipt is
`/home/staticduo/tmp/strata-tp3-20261031/full-model-graph-peerloss2/summary.json`.
The watchdog candidate also passes 255 normal replays with RESET, fourteen
logit rows and 5760 state comparisons bitwise equal to TP1:
`/home/staticduo/tmp/strata-tp3-20261031/full-model-graph-watchdog-normal/independent-qualification.json`.
No performance claim is made.

## Device-side expert routing

The opt-in `--device-expert-routing 1` keeps expert IDs on the GPU inside the
expert provider. It requires batched collectives and compact exact operands.
The eager layer protocol still reads IDs for all-rank agreement. This mode
preserves the original weight codes and accumulation order.

On DG3 GB10, the component fixture passes changing expert IDs at stable
addresses, invalid-ID rejection, 100 alternating repetitions and the 80-row
unpack limit. The original scalar component limits remain 1e-5 absolute and
relative; compact operands are compared bitwise.

A three-host short full-model run exits zero on every rank and matches the
independent TP1 reference bitwise: eight full-vocabulary rows, 2070 state
comparisons, 1728 trace comparisons and 1964 reset comparisons. Evidence is in
`/home/staticduo/tmp/strata-tp3-20261031/device-routing-short/independent-qualification.json`.
This validates the eager provider, not full-model CUDA Graph replay.

## NCCL graph sessions

The explicit NCCL graph-session API passes a three-host CUDA smoke with the
approved NCCL 2.30.7 library: three captured collectives, 100 replays, changing
input contents at fixed addresses and exact analytic results. All ranks exit 0.
A later-replay peer-loss injection ends the survivors through their deadlines
without publishing a passing result. [The graph-session receipt][ncclgraphs]
records exits and pending-DMA diagnostics. This establishes transport capture
and failure behavior; it does not qualify a graph of the complete model.

## Service state checks

A normal C1 service request captures 18 consumed input IDs on all three ranks.
The same IDs replayed through TP1 produce exact final logits and all 171 logical
state arrays. With MTP4, a separate request captures 22 consumed IDs and passes
that same full-state check, including the state left after speculative
rollback/replay. [Normal service][plainstate] and [MTP service][mtpstate]
receipts use the unchanged zero-error policy. These are short greedy cases;
they do not establish long MTP correctness or the sampled distribution.
A client disconnect in the MTP case leaves all three ranks ready when checked.

A separately configured C4 pool completes four mixed requests. Replaying their
canonical consumed IDs through TP1 matches 684 logical state arrays and four
full-vocabulary final logit rows with zero error. [The C4 state receipt][c4state]
qualifies this short state-isolation case. A second four-request run cancels one
client while the other three finish and all workers return ready. It does not
qualify 64K/C4. A separately configured C8 pool also completes eight mixed
requests and matches 1368 logical state arrays and eight final full-vocabulary
logit rows against TP1 with zero error. [The C8 receipt][c8state] covers those
short requests. A second eight-request batch cancels one client; the other seven
finish and the group returns ready. Long-context C8 remains unqualified.

The later complete-and-cancel batches also pass full-state replay: MTP covers
two snapshots, C4 covers eight and C8 covers sixteen, including their cancelled
requests. Each captured canonical input matches on all ranks. The
[C8 combined receipt][c8allstate] compares 2736 logical state arrays and sixteen
final logit rows with zero error. These checks concern short requests and state
isolation; their overlapping run times are not benchmark results.

## Actual 64K prompt and continuation

The original 65536-token prompt plus 256 teacher-forced continuation tokens
finishes with exit code zero on all three ranks. The independent same-pack TP1
reference also finishes. With the original frozen zero-error policy, the model
comparator passes all four full-vocabulary rows at positions 0, 8191, 65535 and
65791 and 684 logical state arrays, including reconstructed GDN state and
replicated KV/indexer/PLE. No NaN/Inf or numerical difference is accepted.
Evidence:
`/home/staticduo/tmp/strata-tp3-20261031/long-context/65792/comparison.json`.
A separate bitwise check also passes 1773 state/replica comparisons across
5.67 billion elements, with no changed bits.
The candidate is native revision `29209221` and the reference `d6055ec5`, recorded
as an explicit revision pair. The TP3 process time includes loading, artifact
work and overlapping correctness trials. It is not a matched performance rate.

## Extended sampled MTP state

Two sampled MTP4 requests each consume a 747-token prompt and return 128 output
tokens, using temperature 0.7, top-p 0.9 and seeds 17/29. All ranks record the
same 874 consumed IDs per request. Independent TP1 teacher-forced replay of those
exact IDs matches both final full-vocabulary logits and all 342 logical state
arrays under the unchanged zero-error policy. The group returns ready between
requests. Evidence:
`/home/staticduo/tmp/strata-tp3-20261031/mtp-extended-service/both-check/result/comparison.json`.
This covers longer sampled state and reset continuity. It does not establish
64K MTP capacity, a real-model output distribution or a performance gain. The
CPU categorical verifier and short cancellation evidence are separate checks.

## Explicit memory phase records

`--memory-phase-file` writes a fresh, bounded per-rank JSONL record. It samples
`cudaMemGetInfo` after allocation, after the first prefill chunk, at the first
decode boundary and before/after reset. The UTC labels describe those exact
points; they do not claim a process allocation total or a high-water mark.
C4/C8 slots can use the same optional callback, with separate request/slot IDs.

A C1 HTTP request with context 8192 and expert-row prefill chunks of eight
records all seven configured phases on all three ranks and returns the group
to ready. Evidence:
`/home/staticduo/tmp/strata-tp3-20261031/memory-phase-service/memory-summary.json`.
This run overlaps long-context correctness work. Its allocator availability
cannot establish isolated capacity or performance. The offline telemetry report
keeps host RSS/PSS separate and retains UNKNOWN for missing historical phases:
`/home/staticduo/tmp/strata-tp3-20261031/capacity-accounting/phase-report.md`.

## Sampled verifier control

The distributed MTP correctness path samples the target directly and admits
proposals only when those target draws match. It does not use the upstream
p/q rejection-sampling algorithm or claim batched verification acceleration.
A CPU qualification uses 10000 seeds, 12 outputs per seed and history-dependent
categorical probabilities. Every output and mock rollback state matches ordinary
seeded sampling. It exercises 35739 accepted drafts and 74091 rejected windows;
the first-token counts are 2060/3003/4937 against the frozen 0.2/0.3/0.5 law
and a fixed 0.025 frequency band. All 95355 checks pass. This checks the verifier,
not real-model sampling frequencies. Evidence:
`/home/staticduo/tmp/strata-tp3-20261031/mtp-categorical-summary.json`.

## Failed attempts retained

Early component and whole-model runs failed numerical gates; their original
[window 2 failure record][failure] is retained. Later exact-lane and finish-order
fixes were checked by the exact comparisons above, without relaxing tolerances.
Window 5's tool request returned HTTP 503 during the intentional peer-failure
test when the coordinator ended; it is not a tool-quality result. Its Anthropic
nonstream timeout is retained separately from successful streaming.
Window 8's first resident TP1 launch omitted interactive stdin,
so it reached READY but produced no token outputs; [retry 2][residentlog] supplies
the qualifying outputs. These failed attempts are not counted as passes.

## Not run or not yet qualified

- Long reset/failure cases beyond the separately qualified 64K continuation.
- A useful additional capacity case and complete per-rank warmup/prefill/decode accounting.
- MTP acceptance probabilities, long rollback/reset and failure state against TP1; sampled distribution checks.
- Longer graph contexts and combined MTP/C4/C8 graph modes; C1 NCCL 255 replay/reset and bounded peer-loss gates pass as recorded.
- Matched TP1/TP2/TP3 performance matrix and continuation thresholds; a separate Staticduo21/vLLM comparison when quantization differs.

No upstream PR is implied. Maintainer scope agreement and the required backend
builds remain gates for upstream review. This record does not mark the overall
goal complete.

## Artifact index

Evidence links below point to local experiment files outside Git. Preserve those
files with this report when sharing it; the links are not public downloads.

[lock]: /home/staticduo/tmp/strata-tp3-20261030/evidence/task1/source-lock.json
[inventory]: /home/staticduo/tmp/strata-tp3-20261030/evidence/task1/inventory.json
[plan]: /home/staticduo/tmp/strata-tp3-20261030/evidence/task2/tp-plan-real-w3.json
[bytes]: /home/staticduo/tmp/strata-tp3-20261030/evidence/task3/11-real-materialized-bytes.txt
[transport]: /home/staticduo/tmp/strata-tp3-20261031/operations/window1/REPORT.md
[lane]: /home/staticduo/tmp/strata-tp3-20261031/evidence/task5cuda/exact-lane-reference.md
[nccl]: /home/staticduo/tmp/strata-tp3-20261031/operations/window7/rank1/nccl.log
[q4]: /home/staticduo/tmp/strata-tp3-20261031/operations/window4/independent-qualification.json
[q6]: /home/staticduo/tmp/strata-tp3-20261031/operations/window6/independent-qualification.json
[q7]: /home/staticduo/tmp/strata-tp3-20261031/operations/window7/independent-qualification.json
[api5]: /home/staticduo/tmp/strata-tp3-20261031/operations/window5
[api6]: /home/staticduo/tmp/strata-tp3-20261031/operations/window6/service-batch
[mtp]: /home/staticduo/tmp/strata-tp3-20261031/operations/window7/mtp/http-parity.json
[c8]: /home/staticduo/tmp/strata-tp3-20261031/operations/window7/c8
[graphs]: /home/staticduo/tmp/strata-tp3-20261031/operations/window7/graph-component-summary.json
[resident]: /home/staticduo/tmp/strata-tp3-20261031/operations/window8/reference-attempt2/parity.json
[residentlog]: /home/staticduo/tmp/strata-tp3-20261031/operations/window8/reference-attempt2/run.log
[profile]: /home/staticduo/tmp/strata-tp3-20261031/operations/window8/profile/rank0.csv
[python]: /home/staticduo/tmp/strata-tp3-20261031/c8-final-python.log
[ctest]: /home/staticduo/tmp/strata-tp3-20261031/resident-cpu-test.log
[compact]: /home/staticduo/tmp/strata-tp3-20261031/compact-component-summary.json
[compactgraphs]: /home/staticduo/tmp/strata-tp3-20261031/compact-graphs-gpu-final.log
[referencehttp]: /home/staticduo/tmp/strata-tp3-20261031/head-upload-probes/reference-http-qualification.json
[failure]: /home/staticduo/tmp/strata-tp3-20261031/operations/window2/NUMERICAL-FAILURE.md

[longreference]: /home/staticduo/tmp/strata-tp3-20261031/long-context/65792/reference-summary.json
[compactmodel]: /home/staticduo/tmp/strata-tp3-20261031/compact-short/independent-qualification.json

[rocesmoke]: /home/staticduo/tmp/strata-tp3-20261031/roce-qualified/retry4-autogid/summary.json
[rocefaults]: /home/staticduo/tmp/strata-tp3-20261031/roce-qualified/fault-summary.json

[hybridmodel]: /home/staticduo/tmp/strata-tp3-20261031/hybrid-short2/independent-qualification.json

[plainstate]: /home/staticduo/tmp/strata-tp3-20261031/service-state-validation/plain-check/result/comparison.json
[mtpstate]: /home/staticduo/tmp/strata-tp3-20261031/service-state-validation/mtp-check/result/comparison.json

[c4state]: /home/staticduo/tmp/strata-tp3-20261031/service-state-validation/c4-check/result/comparison.json

[c8state]: /home/staticduo/tmp/strata-tp3-20261031/service-state-validation/c8-check/result/comparison.json

[c8allstate]: /home/staticduo/tmp/strata-tp3-20261031/service-state-validation/c8-all-check/result/comparison.json

[prefill8]: /home/staticduo/tmp/strata-tp3-20261031/prefill-batch8/independent-qualification.json

[servicebatch8]: /home/staticduo/tmp/strata-tp3-20261031/service-state-validation/batch8-check/result/comparison.json

[expertrows]: /home/staticduo/tmp/strata-tp3-20261031/expert-prefill8/independent-qualification.json

[ncclgraphs]: /home/staticduo/tmp/strata-tp3-20261031/nccl-graph-qualified/summary.json
