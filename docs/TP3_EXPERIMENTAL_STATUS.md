# Native TP3 experimental acceptance record

**Recorded 31 October 2026, including the later 8K gate. This is a partial acceptance record.**
Native Strata runs all 48 layers of the original Qwen pack across DG1, DG2 and
DG3. Short C1 checks match the independent TP1 reference bit for bit. The full
plan is still open: 64K prompts, useful extra capacity, full MTP state checks
and matched performance rounds remain unqualified. There is no speed claim
against TP1 or vLLM, and no production recommendation.

The [implementation plan](superpowers/plans/2026-10-30-strata-tp3-implementation.md),
[design](superpowers/specs/2026-10-30-strata-tp3-design.md) and
[transport plan](superpowers/plans/2026-10-30-strata-tp3-transport.md) are historical
planning records. Their original checkboxes are preserved. This document records
later evidence without closing those plans. Source checkout: `3cc3d77d`; each
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
| 6. GDN, QSA and PLE | Partial | All 48 layers represented in C1 state/trace checks; long continuation and chunk-boundary cases remain open. |
| 7. Coordinator and C1 | Partial | Windows 4/6/7 exact logits, state, RESET and traces; 8192 is configured capacity, not an 8K-token prompt. The later 8K gate is recorded below; 64K prompt qualification remains open. |
| 8. Capacity, MTP, graphs, C8 | Partial | Short MTP HTTP and mixed C8 checks, partial graphs and memory snapshots exist; useful extra capacity, full state/failure qualification and full-model graphs remain open. |
| 9. Server and lifecycle | Partial | OpenAI/Anthropic streaming, tools, cancellation and failure-health artifacts; recorded restoration checks through window 7. No automatic rollout. |
| 10. Performance decision | Partial | Profiling and resident TP1 control established; matched 4K/16K/64K, 256-output, C1/C4/C8, three paired rounds remain open. |

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
failure; graph fault injection and model integration remain unqualified.
The first attempts failed before payload execution because of GID selection,
an old staged binary, or the follower coordinator address. They are retained.
NCCL remains the default model transport. The hybrid model path is opt-in and
keeps reductions on NCCL. Its short full-model comparison now passes: all three
ranks exit 0, with eight exact logit rows, 2070 state comparisons, 1728 traces
and 1964 reset comparisons. [The hybrid receipt][hybridmodel] qualifies this
short case only; long prompts and performance remain pending.

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

- Full 64K prompt C1 state/logit continuation, long resets and failure recovery.
- A useful additional capacity case and complete per-rank warmup/prefill/decode accounting.
- MTP acceptance probabilities, rollback/reset and failure state against TP1; sampled distribution checks.
- Full-model CUDA graphs, long replay/reset/timeout behavior and C4/C8 internal state isolation.
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
