# Strata NVFP4 TP3 Final Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans or subagent-driven-development. Implement the final architecture; components are parallel work units, not user-facing intermediate variants.

**Goal:** Native NVFP4 Strata TP3 reaches approximately Staticduo21 TP3 throughput with comparable context and concurrency, and remains serving with DG1 `strata monitor` working.

**Architecture:** Resident NVFP4 row/column shards execute through Strata's fast Prefill and Verifier kernels. GPU route planning, local expert/mixer outputs and two output reductions per layer replace exact-lane reconstruction; large prompt chunks and real slot-row batches share weight reads.

**Tech Stack:** C++20, CUDA13/GB10 sm121a, Strata MMQ and native NVFP4 kernels, cuBLAS dense GEMM, NCCL/RoCE, SparkRun, existing Strata server/Terminal.

**Design:** [Final performance design](../specs/2026-10-31-strata-nvfp4-tp3-design.md).

## Acceptance And Stop Rule

Measurement audit: the existing Staticduo21 and Strata recipe runs use the same
client script and end-to-end formula (reported output tokens divided by batch
wall time, including prefill). These rates are not pure decode rates. All saved
baseline/candidate cells report2048prompt tokens,128output tokens and zero cache
hits. However, checkpoints/quantization, KV(FP8vsINT8), MTP(4vs2), maximum slots
(24vs8), and generated token paths differ. The fixture input-ID hash is copied
metadata, not proof of identical backend token IDs/templates. Staticduo21 has
one round, so tiny candidate gains are not statistically established. Preserve
these as recipe comparison anchors; do not label them a controlled engine A/B.
For future acceptance separate client TTFT, end-to-end throughput and actual
token-timestamp decode metrics, verify templates/token counts/settings, and use
the required paired warm repetitions. Existing post-first-content estimates
cannot substitute for token timestamps because one delta may carry many tokens.
Audit evidence: `/home/staticduo/tmp/strata-nvfp4-tp3/measurement-audit/report.json`.

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
- [x] Verify complete NVFP4 assets: 17/17 upstream SHA256 pass.
- [x] Finish staging assets: 17/17 SHA256 pass on DG1, DG2 and DG3.
- [x] Archive IQ3 GGUFs/pack on NAS: 11/11 source/destination hashes match.
- [x] Remove verified retired IQ3 GGUFs/pack from all three DGX after independent NVFP4 staging; NAS archive/evidence retained.
- [x] Install owner-local DG1 CLI/monitor dependencies and `strata` on PATH; help works.
- [x] Verify monitor against real TP3 HTTP service: loaded model, context262144,8slots, distributedTP3; /health proves ranks0/1/2.

### 2. Freeze Final Execution Contracts

- [x] Define GPU-only expert callback with inputs, route IDs/weights, rows and output accumulation semantics.
- [x] Define prompt and verify hooks, slot-state ownership and mixer output reduction semantics.
- [x] Make compute policy and numerical tolerance explicit; keep exact-reference code for diagnostics only.
- [x] Define graph batch keys, prefill chunk capacity, memory pages and bounded error/commit semantics.
- [ ] Validate contract tests with synthetic independent references before full-model GPU use.

### 3. NVFP4 Resident Shards And GPU Experts

- [x] Load experts.bin according to v3 metadata, slice gate/up output rows and down input blocks, preserve tail scales and hashes.
- [ ] Reuse tuned native grouped NVFP4 for decode; GPU route grouping and local SwiGLU/down without hidden/lane gathers.
- [x] Apply route weights and local expert sum, one uniform AllReduce of final rows per layer.
- [x] Prefill uses expert-major MMQ FP4 tensor cores on sm121a; actual dispatch and component parity verified.
- [ ] Targeted original-scale component tests: distinct rows/IDs, tails, unequal cuts, partial sums and invalid ID failure; numerical budgets fixed in advance.

### 4. Fast Dense/Mixer And Prompt Integration

- [x] Integrate TP into existing large-chunk Prefill; dense GEMM, HC fusion, chunked GDN, batched sparse QSA enabled.
- [x] Shard mixer output input columns and compute local partials; reduce outputs directly, with no full rank-zero output projection.
- [ ] Preserve GDN head/state mapping, QSA GQA/global mapping, KV and PLE history; no silent context clipping.
- [ ] Check short/long continuation and reset at changed state seams using same-pack reference, not bitexact across different algorithms.

### 5. Decode Graphs, Continuous Batch And MTP

- [x] Integrate fast TP hooks into Verifier's real slot-row batched execution; some state kernels still loop per row.
- [x] Capture fixed-address compute+NCCL per batch shape; one rank process each, one step-level agreement, GPU route validity.
- [ ] Chunk prefill admission and decode scheduling, fair active rows, explicit max-context/page budget and queueing.
- [x] Connect MTP target verification and rollback to fast executor with measured draft acceptance/cost; preserve correct target distribution.
- [ ] Directed isolation/cancel/reset/recapture/failure checks, one bounded check per new behavior.

### 6. Final Candidate And Delivery

- [ ] Independent scoped adversarial review; fix concrete blocking findings and re-run relevant checks.
- [x] ARM CUDA build and actual 3-host component/model execution with exact source/asset pins.
- [ ] Short same-window Staticduo21 comparison PP2048/TG128 C1/C4/C8 at selected depths; measure client TTFT, prefill, visible decode and aggregate throughput separately.
- [ ] Profile only failing target cells; correct dominant kernel/transport/scheduler costs until acceptance or an evidenced external blocker.
- [x] Actual max-context capacity check, numerical/quality suite and functional APIs (earlier candidate; final changed seams still require qualification).
- [ ] Leave Strata NVFP4 TP3 serving with explicit runnable profile, DG1 CLI and working monitor; Staticduo21 remains stopped.
- [ ] Report per-cell values, limits, failed attempts and quality differences. Mark Goal complete only if the acceptance contract is met.

## Evidence And Recovery

Evidence root: `/home/staticduo/tmp/strata-nvfp4-tp3`.
NAS pack: `/volume1/models/huggingface/local-models/Strata-Qwen3.8-Flash-Next-NVFP4-GPTQ`.
Do not alter the original Staticduo21 recipe or checkpoints. SparkRun owns container
lifecycle. The user authorizes stopping it; normal final state is Strata, not
restoration. Preserve >=50GiB free storage per host and existing clock protections.

## Execution Ledger: 2026-11-01

- Parent assumed Prefill, Verifier, expert kernel and mixer implementation after bounded workers did not deliver the missing files. Existing header and task changes are preserved.
- ARM CUDA engine and Prefill builds pass. The native executable links. A source manifest checks 43 changed/new C++/CUDA/header files against DG1 staging with zero disagreements.
- NVFP4 GPU component passes: decode relative L2 0.00763957; W4A8 prefill 0.00828264 against independent original-weight double products. W4A4 same-activation reference 0.00000008. W4A4 versus unquantized activation reference 0.19068749 is reported separately. No budget was widened.
- GB10 has 101376 bytes opt-in shared memory. The original GDN TC scan needs 124944 bytes. The dedicated compact HK5/HK6 instantiations use one staging buffer and preserve the original equations. Actual 97-row GPU tests pass with output/state relative L2 below 8e-7.
- Dedicated G4 QSA split-K and tensor-core instances pass the FP64 component harness at context 4096/1500/2100, zero failures. Padding-head output and intermediate writes were corrected after the first failed run.
- First integrated candidate refused configuration due an old 65792 header in staging; fixed and verified sources. Next candidate reaches READY on all three ranks at context 262144, then fails in the layer-9 expert MMQ operation before any token is published. Boundary diagnostics pass mixer/experts layers 0..8 and mixer layer 9. Rank-zero memcheck is running to locate the original invalid access.
- Whole-model correctness, MTP, throughput acceptance and persistent HTTP/monitor remain pending. Component success is not final delivery.

### Measured Integration Update

The native full model now produces tokens with MTP and exits cleanly on all three
ranks. A short 64/8 comparison produces the same eight IDs with MTP disabled and
enabled. This does not qualify long rollback or multi-slot quality. Warm C1
without MTP measures PP2048 in 1.5758 s and 127 decode steps in 8.3099 s. MTP
currently measures slower; the performance acceptance is unmet.

A node-level Nsight profile identifies gate/up and down expert decode kernels
as about 77% of the target graph's GPU time. The next candidate replaces scalar
nibble lookup with packed SIMD decode and specializes warp/subwarp widths. The
component parity test passes unchanged; the full-model speed measurement is
running. The plan remains active until measured target, quality, concurrency
and final service requirements are fulfilled.

### Current Profile And Correction

The packed expert candidate reaches warm C1 generation around 43 tokens/s. A new
node profile shows that experts now account for about 20% of the target graph
(GU154ms + down73ms out of1124ms accumulated nodes). Dense Q8 products account
for259ms, HC reads201ms and NCCL188ms. Subsequent work targets those concrete
costs. Exact expanded-code decode and small-window MMQ were slower in component
measurements and are not retained.

A scoped independent review found lost MTP catch-up pairs after consecutive C8
rounds skip proposal generation. The executor now catches up each pending target
window before overwriting it. Static review accepts the correction; directed GPU
C8-to-C1 validation is still required.

Prompt QSA now batches key/index RoPE calls, uses the original tensor-core scorer
and bounds selection by live blocks. Actual warm PP2048 improves from2.010s to
1.744s (1174tokens/s); decode remains about42tokens/s. This is not acceptance.
Fast service NCCL channel validation passes11tests; targeted Python total71PASS.

### HTTP Service And Long-Request Framing

Actual HTTP service ran on DG1 port18080 with all3ranks ready, context262144
and8slots. DG1 `strata monitor --once` works against it and reports loadedmodel,
context and distributedTP3. Its24functional prompts completed withoutHTTPerrors;
manual outputs include coherent multilingual/code responses and validJSON. This
is not a score against a monolithic model or performance acceptance.

Vocab-sharded output head now computes disjoint exhaustive ranges perrank and
reduces full logits. An independent scoped review finds no blocker. ActualC1
all3exit0 produces same128tokens; firstfullvocablogitcomparison top1agrees and
KL0.000414 passes frozenbudget. PP2048=1.7204s;TG127steps=2.9205s.

A max-context interface defect was found:1MiBJSONframes cannot hold262KIDs.
SharedC++/Pythonlimit is now4MiB;24KiBcontrolfragmentation unchanged. CPUprotocol,
scheduler,service,autoregressive checks and145Pythonbackendtests pass; parent
reruns162selectedPythonchecksPASS. MaxcontextrequestGPUqualification remains.

ConsecutiveC8MTPcatchupfix accepted by scopedreview. Producer now limitsdrafts
to actualbatchrowroom andremainingoutput/contextbudget, avoidingC4discardedwork.
Newcandidate sha73ecbed6 stagedpersistentservice, startupvalidation running.

Rejected experiments retained only in evidence: expandedFP4integercodes, compact
activeexpertW4A8MMQ, HCtwoBF16componenttensorcorepath. They failedspeedselection;
lasttwo pass component/shortlogitbudget after fixes but are not deliverycode.
Goal remains ACTIVE: targetthroughput,samewindowvLLMbaseline,longquality,C8state
andactualmaxcontextstillrequired.

### 64K Correction Verified

The first64KHTTPrequest failed with a CUDA illegal access. Inspection found the
main KV grew perrequest while MTP's elastic KV retained only2048mappedcells.
The executor now grows both, reserves future draft cells before proposal and
shrinks draft pages only after its stream is idle. A scoped review also caught
firstproposalT not following the limiteddraftbudget; it now useslimit+1.
Actual corrected37a83candidate retries the same65536tokenretrieval successfully:
replyCOBALT-84721, usage65536input, all3ranksREADYafterreset. Wall64.46s,
prefill62.73s (1044.8tokens/s). The262128+16capacityrequest is running.
The attempted memcheck wrapper received no generate command, so it supplies no
fault-location evidence. The correction is supported by code inspection and
the successful actualrequest retry, not a fabricated sanitizer result.

### Maximum Context Verified

ActualHTTP886763candidate plus distributed exactbudget server change processes
262128prompttokens with16allowedoutputs, returnsCOBALT-84721 exactly and stops.
Wall211.92s; prefill207.57s (1262.9tokens/s). All3ranksareREADYafterrequestreset.
The firsttrial had silentlyclampedmax16to8 duelegacyCTX_SLACK; servernowkeeps
slackonlyforlegacyengines.5contexttests includingall3API dialectsexactboundary
pass. Thischeckprovesone requestnearmaxcontextandreset, not8simultaneous262K
requests, longtaskqualityorperformanceparity.

### Concurrency Performance Work

SelectedHTTPPP2048TG128measurements on886763candidateallpassC1/C4/C8requests.
C1nativeprefill1627.7tokens/s, generation43.9tokens/s,clientTTFT1.332s.
C4wholebatch13.534s (37.83outputtokens/sincludingallprefills), C8wholebatch26.270s
(38.98includingallprefills). Perrequestnativeoverlappingdecode rates are22.5-23
and14.5-14.8respectively; do notsum them as a measuredphasecriticalpath.
Targetperformance remains unmet.

FP4direct-registerMMA and compactFP4MMQ passedsameactivationcomponentparity
afterlayoutfixes, butwere slowerthanDP4A. Theyare removedfromdeliverycode
and retainedonlyinexperiment evidence. HCWMMAalso passedcomponentparity
butwas4-10xslowerandisremoved. No unmeasuredkernelpromoted.

Currentcandidatebuildselects the existing16rowwindowcompileoption for this
experimentalexecutor.8chat slots remain theceiling. It enlarges rowmetadata,
outputheadscratchandQ8scratchconsistently, and givesC8oneproposalperrequest
where8rowwindowhadnoroomforMTP. Scope review andactual3hostcandidatevalidation
arepending;thisisnotcurrentserviceperformance.

### MTP Cost Policy

The16rowcandidate12b559passes tooland8slotidentifierchecksandshortC1/C4/C8.
C1nativegeneration43.3tokens/s; C4wholebatch14.407s; C8wholebatch25.048s.
Wideningbyitselfdoesnotclosetheperformancegap. ExistingDraftPolicyisnow
integratedperSlot: itusesdraftprobabilities, observedacceptedprefixandroundcost
to shortenunprofitablegreedydraftchains. Non-greedyrequestskeepthefixedbudget,
so timingfeedbackcannotchoosewindowsbasedonfuturestochasticdraws.
Its25existingCPUchecksPASS,scopedreviewfindsnoblockerandARMbuild44/44PASS.
Candidatec9882edcstartupisrunning;actualquality/performancepending.

### Mixer Batch Integration

QSA decode nowpreparesallrows then scores/topk/attentionin chunksupto8using
existingrowpoolkernels. Eachquerycapturesits ownidx_deadafterappend, preserving
thepartialblockwhilelaterrowsupdate theslot. Staticreviewfindsnoblocker;
actualtool/8identifier/C1/C4/C8checksPASSonfdd3fd87. C8wholebatch24.503s
versus25.242scostpolicycandidate, C1still43.8tokens/s. Thisisnotacceptance.

NewGDNbatchcomponentuses3kernelsforallslotgroups(convolution,recurrence,
historycommit), ratherthanperrowlaunchesandD2Dprefixcopies. Parentcorrected
thefirsttestto use independentreferenceprefixbuffersandproper eventtiming.
ActualKH5/KH6state/history/allprefixesarebitexact; convolutionisbitexactand
outputrelativeL2<4e-8 (checkedunder1e-6,belowfrozencomponentbudget).
KH6indicativecomponent0.104msoldto0.035msbatch; coldKH5numbersare notgeneral
speedclaims. Fullmodelintegrationisbuilding; scopedreviewandruntimepending.

### Grouped GDN Candidate And Remaining Performance Work

Candidate `ee4870a6` integrates grouped GDN and passes the selected HTTP tool,
eight identifier isolation and C1/C4/C8 checks. C1 native generation is49.1tokens/s;
C4 wholebatch13.909s and C8 wholebatch24.018s. Historical Staticduo21 C1 is108.15,
so generation acceptance remains unmet; these runs are not a paired comparison.

The registered `distributed_tp_fast_gdn_batch` CUDA CTest passes on DG1 in0.35s.
The profile completed and persistent Strata was restored afterward. Actual
`/health` reports loaded/ready and alive ranks0/1/2; DG1 `strata monitor --once`
reports ready with context262144 and8slots. Staticduo21 remains stopped.

Next work uses the captured GPU timeline to separate host gaps, target graph,
draft and prefix commit costs. Dense Q8 and HC projections now account for a
large share of target compute; optimize measured duplication or launch costs
before further kernel variants. Whole-model logits for the grouped GDN candidate
and a valid paired Staticduo21 comparison remain open. Do not mark the Goal
complete on the strength of service readiness or the improvement over43.8tokens/s.


### Shared Projections And Control Cost

Candidate `0e5315cb` shares Q8 input quantization across mixer projections and
uses existing pair MMVQ where compatible. Plain greedy requests reuse the picks
already returned by Verifier; penalty/bias/stochastic requests retain sampling.
Actual tool/eight-identifier and selected C1/C4/C8 requests pass. C1 native PP is
1612.9tokens/s and generation50.7tokens/s; visible generation from first content
is37.0tokens/s. C4 wholebatch13.550s and C8wholebatch23.668s. Output hash C1
matches the preceding candidate. Concurrent output hashes vary with batch layout;
selected isolation checks pass, but this is not full numerical qualification.

Worker phase timing measures about2ms per control agreement. Scheduler small
agreements currently repeat length/chunk/checked broadcasts, and speculative
publication repeats them for every slot. The next integration uses the existing
Session exact-payload agreement once per small action and one complete verdict
per batch. All ranks still validate complete payloads and acknowledge before
publication. Legacy adapters retain their existing protocol. CPU and actual
three-host checks must pass before selecting the candidate.


### Dense Q8 Tensor Core Selection

A six-shape graph-replay component comparison on DG1 passes finite/error budgets.
For five rows, existing MMQ Q8 is1.41x faster at K2560/N3840,1.20x at K2304/N2560,
1.11x at K2048/N2560, and0.67x at N512. The16row result used an artificial
split8 native baseline and does not qualify a speed claim against native16.
The opt-in mixer dispatch therefore selects MMQ only for4..16rows, K divisible
by256 and N>=2048; small matrices retain native MMVQ. Original stored Q8 weights
are unchanged. Context scratch is primed before capture, bounds are immutable
per row count, and warmup is bounded by the actual allocated max_rows.

Candidate `ebab6bbf` combines this dispatch with compact scheduler agreements.
ARM build passes. The new scheduler test first segfaulted because its replay
pointer captured a constructor parameter by reference; the test capture is fixed.
Local O3/ASan13776checks and actual DG1 scheduler/service CTest2/2 pass. Actual
candidate HTTP checks are running. Numerical changes in MMQ require whole-model
logit qualification before final delivery; component parity is not sufficient.


### Measured Dense/Control Candidate

Candidate `ebab6bbf` passes tool/eight identifiers and all selected short requests.
C1 native generation falls to43.8tokens/s (PP1592.5), while C4 wholebatch falls
to12.375s and C8 to19.653s. C1 output hash changes; do not attribute that to the
control optimization or call the combined candidate a uniform improvement.
An actual native diagnostic completes on all three ranks with exit0. Four
same-pack TP1 first-logit positions pass the frozen policy: meanKL0.00050862,
maxKL0.00200844, top1agreement100%. This covers selected prefixes, not full task
quality. The next service run retains compact control with dense MMQ disabled to
separate the causes of C1 regression and the concurrency improvement.


### Compact Control Selected, HC BF16 Candidate

With dense MMQ disabled, compact control retains C1 native50.2tokens/s and
improves client-visible generation to42.0tokens/s; C4wholebatch12.250s and
C8wholebatch19.910s. All selected requests/tool/identifiers pass. The profile
therefore keeps dense MMQ disabled pending a better selection policy.

A separate HC probe calls `fused_gr_check()` and confirms actual staged variant3.
One BF16 activation image with existing cuBLAS/kernel APIs passes frozen component
error checks and reduces captured HC times from51.1/75.4/114.7us to32.9/38.0/43.1us
at3/5/8rows. The fixture has no pending residual write; whole-model gates must
cover that path. The previous hi+lo attempt had no end-to-end win; it is not the
same arithmetic/launch schedule as this one-image path.

The new opt-in `STRATA_TP_HC_BF16=1` replaces only the Verifier layer HC read,
including pending residual writes, original BF16 weights and FP32 sums. Prefill,
final head and defaults retain their paths. ARM build14/14 and scoped review pass.
A fresh native diagnostic is running four reference prefixes plus a128token
continuation before service selection. Performance acceptance remains unmet.


### HC Whole-Model Result And MTP Policy

HC one-image BF16 passes four same-pack reference logits (meanKL0.0001676,
maxKL0.0006063, top1allagree) and the128token continuation. HTTP tool/eight-slot
identifiers and C1/C4/C8 pass. The service does not show a throughput win:
C1native47.9tokens/s; C4wholebatch12.774s; C8wholebatch19.811s. The selected
profile therefore disables HC BF16 as well as dense MMQ; both remain opt-in.
Component speed alone did not predict full-model performance.

Adaptive draft costs had been enabled unconditionally for greedy requests.
The executor now honors the existing explicit `STRATA_SPEC_COST=1` opt-in;
otherwise it uses the configured fixed MTP budget. A new candidate builds and
is starting the selected three short cells. Defaults of ordinary engine paths
are unchanged. The native Goal remains active until actual target acceptance.


### Fixed Draft Budget And GB10 Draft Norm

Fixed MTP passes selected HTTP tool/isolation/C1/C4/C8 checks. C1native48.5tokens/s,
PP1643.1, visible41.6; C4wholebatch12.123s; C8wholebatch19.861s. This is not
acceptance against the historical108tokens/s reference.

The existing GB10 setup recommends per-stream MTP hidden normalization; source
inspection confirms pooled is vLLM's reference equation, so stream remains an
explicit draft-only opt-in (`STRATA_TP_MTP_HNORM=stream`). It preserves target
weights and target sampling. A candidate using this documented GB10 setting is
starting the same selected short cells. MTP Q4 remains disabled to preserve the
requested stored weights. The full Goal remains active.


### Draft Stream Norm Result And Fresh Baseline

Fixed MTP with draft stream norm passes selected tools/isolation/C1/C4/C8.
C1native50.0tokens/s, PP1615.3 and visible42.65; C4wholebatch11.930s;
C8wholebatch19.743s. The target throughput gap remains large.
The fresh Staticduo21 baseline retry uses a copy of its exact recipe with only
host127.0.0.1 and port18081 changed, to isolate external traffic. The original
recipe/checkpoints remain unchanged. Stop that baseline after the brief selected
cells and restore Strata; service readiness alone does not complete the Goal.


### Packed Prompt Implementation

Fresh isolated Staticduo21 passes all three depth0 selected cells: C1visible85.94,
C4wholebatch6.056s and C8wholebatch9.665s. The corresponding Strata stream-MTP
candidate is42.65visible,11.930s and19.743s. These single-round measurements use
different pinned checkpoints/KV formats and do not qualify parity. Staticduo21
was stopped and persistent Strata restored; actual monitor/health pass again.

The next opt-in `STRATA_TP_PACKED_PREFILL=1` packs contiguous prompt segments
across slots within one aggregate token budget. Embedding/HC/router/experts and
mixer projections execute over concatenated rows. GDN/QSA and PLE state kernels
operate on each segment's own state/positions, followed by one output reduction
for all rows. Draft callbacks receive the slot identity explicitly. Scheduler
quotas are deterministic and bounded; the default path retains one prompt.
ARM build23/23 and scheduler/service CTest2/2 pass; GPU numerical/isolation and
performance qualification plus scoped independent review are pending.


### Packed Prompt GPU Qualification

Candidate `ec807017` passes the actual three-host packed diagnostic (all ranks
exit0). Four concurrently submitted prefixes1/2/16/64 match the same-pack TP1
logit policy: meanKL0.00006494, maxKL0.00024811 and top1agreement100%. Three
concurrent16token continuations with differing prefix lengths complete. This
covers selected segment/state boundaries; it does not prove arbitrary long C8
quality. The persistent service is starting with packed-prefill enabled and an
8192token aggregate prompt budget, fixed MTP4 and the selected draft stream norm.
HTTP functional and performance cells remain pending for that candidate.


### Packed Service Result And Admission Correction

Actual packed HTTP tool/eight identifiers and selected C1/C4/C8 pass. C1native49.5,
wholebatch4.294s; C4wholebatch11.883s; C8wholebatch19.558s. Improvement over the
preceding profile is small, and fresh Staticduo21 remains faster.

C4 timing showed only three prompts sharing the packed prefill; the fourth was
read separately. The native protocol reader returned idle after one8KiB read even
when a complete submitted JSON frame needed another read. It now drains available
nonblocking bytes until a complete line or EAGAIN before scheduling work. Frame
limits/EOF errors remain bounded; no sleep is introduced. The rebuilt candidate
is starting the same selected HTTP cells to verify actual batch formation.


### Queued Frame Drain Verified

Candidate `6b3bbc8e` passes selected HTTP tools/eight identifiers and C1/C4/C8.
C1native50.1tokens/s, wholebatch4.243s; C4wholebatch11.947s;
C8wholebatch18.867s (54.27outputtokens/s including all prefills). C4 requests now
share the same5194.8ms prefill and C8 requests the same9424.2ms prefill, confirming
batch formation. It changes the latency distribution; earlier requests wait for
the packed prefill. It does not reach Staticduo21's9.665s C8 wholebatch.
Actual service and DG1 monitor are READY after the selected checks. Further work
is restricted to the concrete remaining expert-kernel bottleneck.


### Direct RoCE Integration

The fast executor now has an opt-in direct RoCE adapter, inspired by the actual
Staticduo21 handover transport. FP32 reductions up to2MiB use the existing
RoCEnante backend; larger reductions retain NCCL. Authenticated hybrid bootstrap
pins both backends' provenance and rank-specific direct triangle routes. Stream
steps enqueue multiple operations and captured graphs on one stream, then check
the device/control/proxy completion sequence at one boundary. Legacy per-operation
interfaces remain available. Exact arithmetic-affecting settings join the roster.

ARM build61/61 plus final runtime build2/2 pass. Extended RoCE backend state tests
cover graph replay without host enqueues, empty steps and eager/capture mixtures.
Scoped independent review finds no blocker. Actual three-host native prefixes and
concurrent continuations all complete with exit0; logits pass with meanKL0.00006146,
maxKL0.00023648 and top1agreement100%. Service candidate `c7bf95e8` is starting
selected HTTP functional/performance checks. This is not a measured speed win yet.

### Active Tile Grid For Resident FP4 Prefill

The current29f790 candidate passes selected HTTP checks, but C1/C4/C8 whole
batches remain4.047/12.062/19.340s against Staticduo21's2.972/6.056/9.665s.
A fresh node profile attributes2916ms accumulated GPU time to prompt MMQ;
the large down products launch655360CTAs each, many exiting on empty tiles.

The opt-in `STRATA_TP_PREFILL_TILES=1` builds expert tile prefixes on GPU and
maps a bounded compact grid to the same J128/I128/full-K MMQ helper. Original
bounds, codes, scales, route IDs and activation layout remain in use. At81920
routes the grid bound is1151expert tiles, rather than512*64. This is a launch
count bound, not a throughput claim. Decode remains on its selected DP4A path.

ARM build passes after fixing the W4A4 target's include path. Actual GPU products
at K256/K2560 with balanced/skewed routes, empty experts, expert511 and partial
J tiles match native MMQ bit for bit. Existing independent component checks pass
with worst relativeL2.01326. Scoped independent static review finds no blocker.
The persistent candidate db16ff66 is starting; actual model/API/performance
qualification is pending. Goal remains active and throughput acceptance unmet.

### Tile Candidate Result And Shared Expert Partition

Actual selected C1/C4/C8 requests pass on db16ff66. Whole batches measure
3.865/11.169/17.829s; C1native PP1921.3 and TG52.7tokens/s. C8 native prefill
is charged in multiple packed chunks, so a5.01s per-request timing is not the
wall time to process all eight prompts (first content is about8.56s). Goal
acceptance remains unmet against Staticduo21.

The next structural change partitions the shared expert's original Q8 gate/up
rows and down input columns across ranks256/192/192. Its gated local output is
added to the routed partial before the existing expert output AllReduce. The
executor then consumes that reduced sum without computing or adding a replicated
shared expert. Opt-in `STRATA_TP_SHARED_SHARD=1`; default hooks keep the original
path. No extra communication is introduced. The small gate retains FP32-input
decode and BF16-input prefill semantics; native Q8 block boundaries remain32-wide.

The existing prompt HC upmix also passes its actual21-shape GPU component on
GB10, with bitexact outputs on T>=33 and roughly2x component speed at8192rows.
It will be qualified with the shared candidate, not claimed as an end-to-end
win from the component test. Parent owns shared contracts/integration; the Local
worker owns FastMixer slices and local GPU helpers. CPU contract validates add
before reduction and no reduction after a provider failure. Final GPU/logit,
functional, performance and long changed-seam checks remain required.

### Shared Expert Qualified, Remaining Gap

Shared candidate f3393fcc builds on ARM CUDA and passes the three-host native
diagnostic (all ranks exit0), four TP1 first-logit positions (meanKL.00028473,
maxKL.00109413,top1allagree) and three concurrent continuations. The selected
HTTP tool/eight-identifier and C1/C4/C8 requests pass; service and monitor are
READY at context262144/8slots. C1wholebatch3.731s, nativeTG54.0; C4wholebatch
10.913s. This is a useful small gain; it does not meet the Staticduo21 target.
Original full shared weights remain loaded alongside slices; this change saves
compute, and no resident-memory saving is claimed.

### Resident Dense Tensor Core Candidate

The shared candidate C8wholebatch is17.148s (59.72outputtokens/s including all
prefills), still below the Staticduo21 target. A focused GPU graph comparison
of original Q8_0 MMVQ and resident FP16 expansion passes its independent CPU
double reference (eight selected output rows). K2560/N3840 at T5 improves
24.66to18.44us; T16 improves65.50to20.57us. K2304/N2560 improves18.44to14.33us
at T5 and45.05to16.39us at T16. Conversion of activations is included; upfront
weight preparation is separate. No general whole-model gain is inferred.

STRATA_TP_DENSE_F16=1 expands only large Q8 mixer projections once before
capture and selects existing cuBLAS FP16 products for4+rows. Original Q8 data
remain resident and handle smaller batches. Prefill reuses that expanded copy.
The opt-in is included in the authenticated roster. ARM build3/3 and25selected
Python checks pass. Actual three-host logits/continuations and performance are
pending; the persistent service will be restored after the diagnostic.

### Fixed MTP Chain Enqueue Correction

Code inspection found that disabling cost adaptation still left a chain callback
installed. MtpDrafter therefore selected its host-wait-per-draft loop rather than
its existing full-chain enqueue branch. The fast executor now leaves that callback
empty for fixed budgets and installs it only for the explicit adaptive policy at
request admission. Target verification and draft budgets are unchanged. ARM build
passes; selected native continuation/performance qualification is pending. This
change affects the opt-in fast executor only.

### Concurrent Draft Chains

The opt-in fast executor can use Strata's existing per-slot chain_launch and
chain_poll API to launch every independent draft chain before collecting any.
Each drafter owns its stream/KV/workspace; weights remain shared. A single target
stream completion orders the accepted residual copies before these launches.
Preparation captures chain graphs before serving. Fixed-budget requests use the
new batch route; adaptive requests retain their serial early-stop callback.
STRATA_TP_ASYNC_DRAFTS=1 is fingerprinted and disabled by default. ARM build4/4
and selected Python checks pass; actual GPU/model validation is pending.

Correction: the HTTP c6c4 deployment script did not set STRATA_TP_DENSE_F16 in
engine_env, although its native numerical diagnostic did. Therefore its HTTP
numbers do not measure the resident-F16 path and the previous statement of no
whole-model gain was unsupported. The next deployment must set and read back
both resident-F16 and async-draft settings explicitly before measuring them.

### Balanced Dense Shards And Large HC Batches

The actual combined async/F16 HTTP candidate passes selected checks and measures
C1wholebatch3.855s,C4wholebatch10.821s,C8wholebatch16.765s. Target remains unmet.
The initial F16 threshold excluded DG2/DG3's1920-wide GDN projections; eligibility
now includes these1536+wide matrices to avoid leaving slower peers on MMVQ.

The bespoke inline-MMA HC component failed its independent reference twice and
is rejected outside production. The selected implementation instead fuses the
existing cuBLAS HC path's per-row pending writes, norm/pack, SiLU/pack and inject
publication. All8actual component cases pass independent F32 reference, finite
checks and canaries. At T16 with apply/inject, graph205.97us becomes57.33us;
this is a component result. STRATA_TP_HC_BF16=large selects it only for9+rows,
retaining GRV3 for C1. Actual native model qualification is running.

### Large HC Result And PLE Residency

C756 passes selected HTTP tool/isolation/C1/C4/C8; whole batches measure
3.836/10.300/15.829s (C8aggregate64.69outputtokens/s). Generation acceptance
remains unmet. Scoped HC review finds no critical issue; its private `lo`
scratch is not copied to the external scratch, matching the current GRV3
consumer contract (that field is not consumed afterward).

The opt-in STRATA_TP_PLE_IO=ram reuses the existing mapped-and-locked PLE reader
for the original51.2GB FP8 table. TP3 has measured memory headroom that TP1 lacks.
The reader must confirm the complete table locked before READY; otherwise the
profile fails. Locked gather skips per-row page-prefetch hints, which cannot
help pages already resident. Direct I/O remains the default. ARM build3/3 passes;
actual memory/capacity and performance qualification are pending. Preserve full
context262144 and eight slots; RAM residency must not hide a capacity regression.

### RAM Reader Selection Result

The actual RAM reader locks50,000,256KiB on rank0 and reaches READY on all ranks,
with about21GiB available on DG1. Selected HTTP tools/isolation/C1/C4/C8 pass.
C1wholebatch3.734s,C4wholebatch10.016s,C8wholebatch15.837s; C8 is effectively
the same as Direct I/O15.829s. The selected final profile therefore returns to
Direct I/O, preserving memory for long requests. RAM remains an explicit option
and is not claimed faster. A brief node profile of the current complete Direct
candidate is running; restore persistent Strata after export.

### BF16 Reduction Candidate

The current node profile still charges about1403ms accumulated GPU time to
large prompt NCCL reductions. The opt-in STRATA_TP_REDUCE_DTYPE=bf16 packs only
mixer/expert partial outputs, uses existing NCCL/RoCE BF16 Sum and unpacks to
FP32 before residual consumers. Output logits remain FP32. One bounded scratch
is allocated before capture and shared only on the ordered execution stream.
Original weights do not change. ARM build9/9 passes; actual logits/continuations
are running with the fused HC path selected at small and large batches. The
frozen numerical budget is unchanged; reject if it fails. No throughput claim.

### Reduction Dispatch Selection

The all-BF16-reduction/all-BF16-HC candidate passes the frozen logits/continuation
policy (meanKL.00048435,maxKL.00174007,top1allagree) and selected HTTP checks.
C8wholebatch15.115s improves, but C1native43.8tokens/s regresses. The next profile
therefore selects BF16 reductions only for prefill and retains FP32 reductions
and GRV3 at small decode batches; large HC batches retain the measured fused
cuBLAS path. This is explicit prefill-bf16 dispatch, not a default change.

### Reuse The Exact B12X Expert Kernel

Current scalar-DP4A decode remains below the target. CUTLASS DSL4.7 and B12X1.3
in the pinned DG1 container support AOT export to C headers/objects; actual M5
K2560/N256/E512/top10 export has succeeded. Inference can call this artifact from
C++ without a Python kernel runner. A small isolated source patch gives gate and
up their separate original FP32 GGUF tails; all original E4M3 scale bytes remain.

The CUDA repacker converts GGUF's split nibble ordering to adjacent packed FP4,
places up/gate in the B12X source layout and swizzles scale bytes. CPU synthetic
roundtrip/swizzle checks pass; ARM CUDA component compiles/links. Parent will
measure the exact fused kernel through its C ABI before any model integration.
This is pending GPU qualification and is not a throughput claim.

### AOT Component Result

The B12X AOT C++ component launches and captures successfully after correcting
host barrier descriptors and using40cooperative CTAs (80was too large on GB10).
The W4A16 component preserves original codes/scales/tails and gives relativeL2
.00943 against DP4A, but its M5 layer takes.285ms against.273ms DP4A. The A4
component requires activation global scaling to avoid underflow and has not
qualified quality. Neither component is selected for the service. Preserve
these artifacts as reproduction evidence; do not claim an inference win.

Further selection uses FP32 reductions and the now-fused HC path with a bounded
MTP budget; no failed bespoke MMA or AOT kernel is integrated into serving.

### Current Cost Selection

MTP2/fused-HC/FP32 passes selected HTTP checks (C1native55.2tokens/s,
C4wholebatch9.961s,C8wholebatch15.663s). A live verified placement change confines
the RoCE proxy to GB10's big cores; C1native59.5 and C4wholebatch8.688s improve,
while C8wholebatch15.860s does not. The placement is now explicit/fingerprinted.

Resident dense F16 now covers2+rows, matching the MTP2 C1window instead of falling
back to MMVQ at3rows. STRATA_TP_CONTROL_COMPACT=1 validates slot/request/position/
draw/feedback in the proposal broadcast and seals commit plus full publication
payload in the final exact agreement. This removes redundant spec-batch and
commit-only round trips without publishing before all ranks agree. ARM build3/3
and selected Python checks pass; actual native qualification is running.

### Compact Candidate Numerical Qualification

Candidate73412c12 passes actual three-host native execution (all ranks exit0),
four reference logits (meanKL.00037406,maxKL.00148795,top1allagree) and concurrent
continuations. Selected Python suite170PASS, ARM scheduler13779checks and
service11069checks pass, plus fast-execution/RoCE state contracts. All84changed
and new source files match DG1 staging. Persistent service is starting with the
explicit selected profile. Actual HTTP performance and changed long-context
qualification remain pending; Goal is not complete.

### Long Context Regression And Exact Reference

The compact candidate passes short HTTP C1/C4/C8 at whole-request times
3.765/9.986/15.854 seconds. Staticduo21's same-day isolated reference remains
2.972/6.056/9.665 seconds. The performance acceptance is unmet.

Its actual262128+16 request fails with HTTP503 and unloads the native group.
The runner's fixed30second health lease was shorter than the native120second
operation budget. The lease now follows max(30,native_seconds+5);75runner tests
pass, including new regression cases. A fresh-process long request still fails
after12.182seconds with coordinator-ended, so this lease correction alone does
not fix the regression. Capture native stderr through the launcher's existing
bounded/redacted private log before changing GPU code. The earlier256K pass
does not qualify this candidate. Strata must be restored after diagnosis.

Read the original recipe directly on DG1:
`/home/staticduo/recipes/triple/qwen3.8-flash-next-staticduo21-tp3.yaml`.
It pins image1346035488a5c8a0f405ade8e74febf68c66b8df1e026b7e3a6fb83cb9f4ce13,
vLLM884b4ff6 and the18refusal/18handover/21HC-BF16 mods. NAS`~/git/vllm`
HEAD5423fb8b93a539b24e7f80075d132d4f4dcab95b is newer; inspect pinned source
and mods when drawing implementation comparisons.

Raw Nsight evidence is on DG1 at
`integrated/optimized-node-profile/timeline.sqlite` under the evidence root.
Correct cumulative graph203 CUTLASS Kernel2 time is192.14ms, not2192ms;
graph215 F16WMMA is173.59ms with approximately106.4ms BF16WMMA.
RoCE/GU/down are165.76/131.26/67.38ms and235.67/171.50/95.47ms respectively.
These are kernel sums across launches, not wall latency. That profile predates
the final HC-all/MTP2 selection. Costs are distributed; do not infer a single
dominant dense bottleneck or promise parity from an isolated kernel change.

Native diagnostics now identify the actual long-request failure on all ranks:
`native GDN preprocessing requires a stream and count in [1,65535]`.
An8191token prompt chunk produces122865/147438 independent head rows.
The fast GDN provider now splits only the output normalization into calls of
at most65535head rows, preserving the tensor-core scan and prefill chunk size.
Actual GPU checks at97and8191tokens with5/6key heads pass with output and state
relativeL2 below8e-7. Candidate de50a4e1 is restarting for the actual256K retry.

The actual de50a4e1 retry now passes: HTTP200,262128input tokens,16allowed
outputs, exact answer`COBALT-84721`, wall165.037s and nativeprefill163.562s
(1602.6tokens/s). All3ranks remainREADY after cancellation/reset. DG1
`strata monitor --once --output json` confirms loaded:true,context262144,
8slots and distributedTP3. This qualifies one near-max request, not8concurrent
near-max requests or performance parity. Staticduo21 remains stopped.

A bounded fused-QKV/Z resident-F16 experiment passes component parity but gives
only1.01x on rank0M3 and regressesM16. It is removed from production source;
no performance claim or profile option is retained for it.

### Parallel RoCE Copy And Reduction

The exact B12X transport uses multiple GPU blocks for staging and reduction;
the previous Strata transport used one. New explicit`STRATA_TP_ROCE_BLOCKS=8`
uses8cooperative CTAs of512threads. Grid barriers order complete staging before
the proxy doorbell and complete consumption before epoch publication. FP32
copies/reductions use16byte packs and preserve fixed rank summation order;
BF16 retains FP32 accumulation. The default single-block kernel is unchanged.

ARM build passes. Actual3host smoke passes18FP32/BF16reductions,2gathers and
100CUDA Graph replays per rank; all processes exit0. The finite smoke ends
before SparkRun's5second process check and SparkRun reports launch failed;
native receipts and numerical results establish its actual pass.
Scoped independent review finds no blocker. Full-model candidate7e917971
passes4same-pack TP1logit comparisons (meanKL.00037406,max.00148795,top1allagree),
3concurrent continuations and all3processes exit0, including first collective
capture. Persistent service is restarting for actual whole-model measurements.
Fault propagation and the final changed transport's long request remain pending;
do not claim performance acceptance from component/graph checks.

Actual persistent7e917971HTTP tools/eight-slot isolation and C1/C4/C8 pass.
Whole-request times are3.763/9.661/15.187s; C8aggregate67.43outputtokens/s.
C1native generation55.7tokens/s. The previous de50C8takes16.232s, so the
single-round C8improvement is modest; Staticduo21 still takes9.665s.
The actual changed-transport262128+16request is now running. The parity Goal
remains unmet and must not be closed from this improvement.

The changed transport also completes the actual262128+16HTTP request:
exact`COBALT-84721`,wall150.493s,nativeprefill149.063s(1758.5tokens/s),
all3ranksREADY after reset. This is one request, not full concurrent capacity.
The remaining directed transport failure check is graph-peer exit; the
persistent Strata service must be restored after that isolated check.

### Full C8 Speculative Window

The graph-peer-exit check now terminates ranks0/1with exit2within12.44/8.68s
after rank2's injected exit86. Persistent7e917Strata is restored and its monitor
reports a loaded model. No Staticduo21 restoration is required.

The next structural change is an explicit CUDA TP build option,
`STRATA_TP_WIDE_WINDOW=ON`:40logical verification rows permit8slots with4drafts
each. Native matvec instantiations remain at16rows, with row-local wrappers
tiling projections, HC argument packs and shared-expert operations. GDN keeps
8independent groups of at most5rows. Every layer processes the complete logical
batch before its existing two reductions; no full-model pass per slot is added.

Host metadata, scratch and nonfinite-row masks must cover40rows (64bit mask).
The resident plan's compile-time tables grow to512entries in this opt-in build;
defaults8/16stay unchanged. All linked engine objects must share the same window
macro. Build separately in`build-wide40` to preserve the qualified16row binary.
Required checks:17/32/40row projection/HC parity,8x5GDN state/snapshots, actual
C8MTP4logits/continuations and a brief whole-model performance comparison.
This implementation is in progress; speed and capacity are not yet qualified.

The40row build passes ARM compilation and actual GPU components: Q8quantization
and MMVQ40against one-row calls, HC17/32/40and BF16projections against CPU
products, plus8x5GDN states/history/snapshots. Initial full-model loading exposed
an auxiliary interleaved-MMVQ allocation requesting40rows; its native maximum
remains16, so that unused wide auxiliary is bounded to16while plain Q8covers40.

Corrected8d7cf371passes actual3host inference: four TP1logit comparisons have
meanKL.00037406/max.00148795/top1allagree;8concurrent16token continuations finish,
including38-40row graph layouts. All ranks exit0. Persistent service restarts
with MTP4 for C1/C4/C8measurement. Speed and final maximum-context checks remain.

Actual40row/MTP4 C1/C4/C8whole-request times3.564/10.599/18.031s pass requests
but regress concurrency. WithMTP2they take3.749/9.717/15.451s, still no gain
over the16row/MTP2reference15.187s atC8. Logical capacity alone is not acceptance.

### Compact Greedy Output

An explicit`STRATA_TP_COMPACT_HEAD=1` removes full-vocabulary communication for
plain greedy requests. Each rank scans its local logits, emits maximum/value,
global token ID and finite flag in a padded4float record, and reduces all3records
(48bytes per row). Global selection preserves the lowest-ID tie rule. Every
local logit participates in the finite check. Sampling, history transformations,
logit bias, mixed batches and diagnostic logits retain the full head path.
Graph keys include compact/full mode before capture.

ARM build and actual40row component tests pass (negative maxima, ties and
row39finite flags). Candidate76931506 passes actual3host execution, four first
picks and eight16token continuations exactly match the full-head reference;
all ranks exit0. Scoped independent review finds no blocker. Persistent service
is starting for actual API fallback checks and performance. No speed claim yet.

Actual76931506HTTP tools/isolation and C1/C4/C8 pass at3.789/9.546/15.163s.
The C8result is essentially equal to7e917's15.187s and below the target.
Actual262128+16HTTP request returns exact`COBALT-84721` in149.580s with
1768.8nativeprefill tokens/s. Mixed sampling/penalty/greedy requests and a
subsequent pure-greedy request complete; all ranks stayREADY. Token biases
receive the existing HTTP400unsupported response, which is unchanged.
DG1`strata monitor` confirms loaded:true. The parity Goal remains active.

### Current Profile And Balanced Expert Work

The current769profile completes on all3ranks and exports its actual SQLite
timeline on DG1 at`integrated/compact-head-node-profile/timeline.sqlite`.
C8graph215has11replays: GU189.13ms/down93.11ms are its largest compute costs,
RoCE122.54ms, GDN57.04ms. These are cumulative kernel times, not request latency.
The original-weight B12X A16M24component is numerically qualified at relativeL2
.01359but only improves layer time1.267to1.201ms; it is not integrated.

New explicit`STRATA_TP_BALANCED_EXPERTS=1` partitions FF640as224/208/208instead
of256/192/192. Cuts are16aligned, preserving each original16value scale and8code
bytes when rebuilding the padded down rows. Decode omits padded-row dots;
prefill retains zero-padded storage. Defaults are unchanged.

ARM build and scoped independent review pass. Candidateb6bf3e09passes actual
3host inference, four TP1logit comparisons(meanKL.00004646,max.00017225,top1allagree)
and8concurrent continuations; all ranks exit0. Persistent service is starting
for directed HTTP checks and C1/C4/C8measurement. Speed selection is pending.

### Small J8 FP4 Decode Candidate

The current wrapper is templated over J: prefill retains J128 and the optional
`STRATA_TP_DECODE_FP4=1` selects J8 for decode windows above eight rows. Original
weight codes/scales and expert tails are preserved; activations use the existing
W4A4 quantizer. Smaller windows retain DP4A. The setting participates in the
rank fingerprint and service allowlist. It is disabled by default.

Actual DG1 GPU checks: J8 products equal canonical MMQ bit for bit for K256 and
K2560, skewed and dispersed routing, including reversed output IDs. Full T24
expert component agrees with the FP4 activation oracle (relative L2 about 6e-8).
Synthetic skewed routes take 0.202 ms versus DP4A 0.356 ms; dispersed routes take
1.361 ms versus 1.400 ms. These component times do not prove service performance.
A4 versus A8 output L2 is about 0.275 for this synthetic cancellation-heavy input;
that comparison must not be presented as same-precision parity.

ARM build passes. Updated integrated GPU parity passes (decode16 FP4 L2 7e-8,
decode8 DP4A L2 0.0133); invalid routes are detected. The first test invocation
used a stale test object; an explicit rebuild and retry passes. Python profile
and runner tests: 97 passed. Scoped independent review reports no blocker.

Native three-host candidate `0a29393b` exits zero on all ranks. Four TP1 prefix
logit comparisons retain mean KL 0.0000464636 and max 0.0001722456 with all top1
choices agreeing. All eight 16-token continuations equal the previous balanced
candidate. This limited fixture does not establish long-context decode quality.
Persistent service speed and functional checks are pending; the goal is unmet.

Actual J8 service check: tools and eight isolated identifiers pass. C1/C4/C8
whole-request times are 3.698/9.710/15.676 s, versus balanced DP4A
3.739/9.220/15.304 s. J8 does not qualify as a speed improvement in this run.
C4 and most C8 generated texts differ from A8, while C1 is unchanged; functional
checks pass but this is not a complete quality evaluation. Mean eight-slot
verification time is 95.59 ms versus 98.82 ms, with admission/prefill and changing
graph layouts also contributing to whole-request latency.

A concrete excess-work correction bounds the compact grid by total routes:
each nonempty tile consumes at least one route. For 240 routes the old bound
launched more than 500 column positions. The bounded component retains the same
FP4 oracle agreement and measures 0.186 ms for skewed routes and 1.327 ms for
dispersed routes. Build `001c2a40` and integrated GPU parity pass. Its persistent
service measurement is pending. No claim of parity with Staticduo21 is made.

Graph reuse investigation: 25 captures in the small-J8 HTTP run correspond to
25 distinct slot layouts, with no repeated layout in that run. FastMixer binds
GDN state/prefix pointers and QSA pools by host slot ID during capture. Removing
slot IDs from the graph key would corrupt request state. Runtime slot indirection
in GDN/QSA would be required before sharing graphs between slot identities; no
unsafe key-only shortcut is implemented.

Bounded J8 actual HTTP measurement: C1 3.671 s, C4 9.617 s, C8 15.602 s
(65.63 aggregate output tokens/s including prefill). Tools and eight request
identifiers pass. This still fails speed selection against balanced DP4A. The
service is restored to qualified binary `b6bf3e09` with decode-FP4 disabled;
J8 is an unselected experiment, not the delivered performance profile.
The overall throughput goal remains active and unmet.

### Batched QSA Preparation

The measured graph fixes each slot layout, so sharing graphs by removing slot
IDs is unsafe. Its 80 instantiations total 420.66 ms (median 0.468 ms); that does
not justify claiming geometric reuse alone closes the throughput gap.

Two further expert components are not selected: static Q8 grouped work retains
relative L2 zero but yields only 4% in synthetic three-token reuse and no win for
dispersed routes; compact INT8 J8 is slower in that reuse case. Existing prefill
HC upmix is retained: T2048/T8192 GPU parity is bitexact and fused timings are
0.743/2.747 ms versus cuBLAS-plus-mix 1.371/5.607 ms on DG1.

New opt-in `STRATA_TP_QSA_PREPARE_BATCH=1` groups decode metadata and the three
existing QSA RMS/RoPE operations across the full row window. This changes five
preparation launches per row to five per batch; KV append, snapshots, indexer
updates and attention preserve their row/slot ordering. The runtime fingerprint
includes the setting. Default remains disabled.

Actual GPU checks cover metadata for 1/9/24/40 rows against per-row calls,
position changes during graph replay, host slot metadata lifetime and sticky
slot/context errors. Q/K/index RoPE calls are bitexact against per-row calls for
all four batch lengths. ARM build passes, profile/runner tests: 98 passed.
Scoped independent review finds no blocker. Three-host full-model quality and
service performance are pending; component success does not meet the goal.

Candidate `e2020f63` completes the native three-host test with all ranks exit0.
Four TP1 first-token logits retain mean KL 0.00004646 and max 0.00017225;
top1 agrees on all four. Four initial picks and eight 16-token concurrent
continuations match the balanced candidate exactly. The persistent candidate
is loading for HTTP checks and C1/C4/C8 timing; no throughput claim yet.
Additional selected backend tests: 150 passed. Source manifest covers 95 changed
or new code files with zero mismatch on DG1.

Actual e2020f63 HTTP tools and eight isolated identifiers pass; C1/C4/C8 times
are 3.725/9.152/15.480 s. Balanced baseline was 3.739/9.220/15.304 s.
Mean eight-slot verification phase drops from 98.82 to 94.51 ms, but admission,
prefill and batch evolution leave whole C8 latency slightly worse in this one
run. Thus this is an arithmetic-preserving reduction in launch work, not proven
overall speed selection or parity. Current e2020f63 service remains READY with
all three ranks and working DG1 monitor. Decode FP4 is disabled.

Next directed change: batch QSA KV/indexer appends across rows while retaining
each slot's sequential prefix snapshots and rollback. Host pointer tables must
stay tied to the captured layout. Avoid a graph-key shortcut or changing Q8/FP4
precision solely to obtain a faster passing fixture. Validate the component
against the existing row loop, then compare the complete candidate once.

### Grouped QSA Appends

Opt-in `STRATA_TP_QSA_APPEND_BATCH=1` adds one grouped Q8 KV append and one
indexer launch for the decode window. It requires batched preparation. Each
indexer CTA owns a request's contiguous group and calls the existing append body
one token at a time. It saves tail/dead/block position before each speculative
row, raw keys after append and per-row dead keys for attention. The existing
commit reads the same 513-float snapshot stride and accepted-prefix index.
Default kernels retain their original launch signature and behavior.

Actual DG1 CUDA component compares eight groups of 1/3/5 tokens against the old
row loop, including block crossings and replay with changed positions. KV codes,
scales, full pooled/tail/dead state, block position, all snapshots and per-row dead
keys are bitexact. This test validates snapshot contents; it does not directly
call FastMixer::commit_slots. Full-model MTP continuations are a separate gate.

ARM build passes (`fbc4a02d`). Selected profile/runner/logit tests:105 passed.
Scoped independent review reports no concrete blocker. One initial component
build lacked the staged header; corrected sync and retry pass. The full-model
three-host check is running; no throughput claim is made yet.

Candidate fbc4a02d passes native full-model validation: all three ranks exit0,
four TP1 prefix logits retain mean KL 0.00004646/max 0.00017225 with all top1
choices agreeing. All eight 16-token concurrent MTP continuations equal the
previous candidate. Source manifest now covers 100 modified/new code files with
zero mismatch on DG1. Persistent service speed validation is running; final
throughput acceptance is still unmet.

Actual fbc4a02d HTTP tools and eight isolated identifiers pass. C1/C4/C8 wall
is 3.682/9.054/14.989 s; C8 aggregate is 68.32 output tokens/s including prefill.
The previous preparation-only profile was 3.725/9.152/15.480 s. Mean eight-slot
verification drops 94.51 to 90.70 ms. These are one-pass measurements, not final
same-window parity qualification. The original balanced profile's C8 was15.304s.
The new candidate remains below Staticduo21; maximum-context retrieval is running.

Maximum-context fbc4a02d HTTP retrieval passes:262128 input tokens, zero cached,
16 output tokens allowed, exact marker COBALT-84721, HTTP200. Wall148.856s,
native prefill147.563s (1776.4tokens/s). The answer ends naturally at10 visible
tokens; native timing reports14 committed generated tokens. These are distinct
accounting values. All three ranks return READY and DG1 monitor succeeds with
context262144 and8slots. Current selected profile retains both grouped QSA
preparation and grouped appends. Overall Staticduo21 parity remains unmet.

### Distributed MTP Proposal Ownership

Measured C8 spends18.09ms/round proposing on rank0 while other ranks wait,
versus90.70ms for target verification. New opt-in
`STRATA_TP_DISTRIBUTED_DRAFTS=1` assigns stable owner `slot % 3`; each owner loads
its local draft slots and receives the already replicated target residuals.
Target TP3 verification, root sampling and committed verdict remain unchanged.
The owner updates its draft policy and private draft state. The profile validates
the MTP file manifest identity and includes it in the agreed rank fingerprint.
Startup aborts on local draft initialization failure and waits for all owners.

A bounded control exchange gathers one binary fragment per rank and publishes
rank-ordered fragments only after committed broadcast. Each follower verifies
its contribution before ACK. Existing broadcast wire behavior is preserved.
The initial implementation had an early-request race; CPU TCP tests found it.
The correction treats contributions as broadcast entry, avoiding another request
phase. Tests cover malformed/duplicate/oversized contributions, altered aggregate,
output preservation, missing commit, out-of-order ACKs and25 repeated binary/empty
three-rank exchanges:662checks,0failed. A label-mismatch test was corrected to
use equal-length labels rather than expecting a mismatch from a truncated frame.

ARM candidate dbeda5d8 builds. Profile/runner tests100passed. Full-model owner
validation is running; the distributed draft latency gain is not yet measured.

Candidate dbeda5d8 completes native three-host validation with all ranks exit0.
Logs confirm rank0 owns slots0/3/6, rank1 slots1/4/7 and rank2 slots2/5. Four
TP1 logits retain meanKL0.00004646/max0.00017225 and all top1 picks agree;
eight concurrent16-token continuations equal fbc4a02d. Service timing is pending.

Actual distributed-drafter HTTP tools/8identifier isolation pass. C1/C4/C8 wall
is3.703/8.672/14.525s; C8 aggregate70.50tokens/s including prefill. The previous
fbc4 profile was3.682/9.054/14.989s. Mean8slot proposal generation drops18.09 to
7.14ms, exchange takes1.47ms and target verification91.25ms. This supports a
proposal-phase gain; the whole-service measurement remains a single pass and
below Staticduo21. Maximum-context retrieval is running. Decode FP4 stays off.

Distributed-drafter maximum-context retrieval passes:262128 input, zero cached,
16 output allowed, exact COBALT-84721, HTTP200, wall150.298s, native prefill
148.967s (1759.6tokens/s). It returns10visible output tokens and records14native
committed tokens, reported separately. All three ranks return READY; DG1 monitor
works at context262144/8slots. Selected profile remains dbeda5d8 with distributed
proposals and both QSA batch switches. Overall throughput acceptance is unmet.

### Packed HC Down And Inject

Two further components are rejected before service integration: fused expert
down+combine is exact but M24 takes1.332ms versus1.296ms; the installed B12X BF16
AOT dense kernel passes FP32 parity but long-K down/inject are much slower than
cuBLAS. Its up projection's small gain does not justify selecting the full path.

A direct HC fusion stacks original BF16 down[320,10240] and inject[4,10240]
weights, pads output width to336 and makes one cuBLAS BF16 GEMM. A fused helper
extracts SiLU/4 to the existing320-wide BF16 image and publishes inject columns
320..323. The weights are copied once at load without requantization, adding
about630MiB per rank. `STRATA_TP_HC_PACKED_DOWN=1` is opt-in and fingerprinted;
old HC behavior remains the default.

Actual componentM3/M24 outputs equal the CPU reference and separate products;
combined12.57/16.40us versus17.12/21.12us. Integrated GPU helper checks3/17/24/40
rows against the old helper, including hostile padding, pass exactly. ARM build
candidate dbf54cec passes and107selected profile/runner/logit tests pass. Scoped
review finds no stride/alias issue. Full-model validation is running.

Packed-HC full-model dbf54cec passes:all3ranks exit0, meanKL0.00004081 and
max0.00015355 over4TP1prefixes, all top1 agree,8concurrent16-token continuations
match dbeda5d8. Persistent service is loading for the actual timing comparison.

Packed-HC service timing fails selection: C1/C4/C8 wall3.704/9.126/14.717s
versus distributed-draft baseline3.703/8.672/14.525s. Mean8slotverify90.87ms
versus91.25ms is nearly unchanged despite the component saving. Tools and8IDs
pass, but no overall gain is demonstrated. The option stays disabled and the
service is restored to previously qualified dbeda5d8. The new code is an
unselected opt-in experiment. Overall throughput goal remains unmet.

### Selected-Profile Timeline

A fresh Nsight Systems capture of selected dbeda5d8 completes with all ranks
exit0. Evidence:`integrated/selected-drafts-node-profile/timeline.sqlite` and
`critical-path.json`. C8 target graph92 accumulates GU146.88ms, down93.61ms,
RoCE76.33ms, denseF16 62.67ms, GDN56.22ms over11replays. C1 graph83 is dominated
by denseF16 products234.80ms, RoCE141.29ms and HC BF16 products101.24+88.69ms.
These cumulative kernel values are not wall latency and must not be summed
across phases or overlapping streams.

Marked C1 phase is2.941s with2.000s kernel union; C8 is9.355s with8.626s kernel
union, including prefill. Nsight adds substantial launch instrumentation cost;
API launch totals under profiling are not unprofiled runtime overhead. The
capture has36graph instantiations totaling253ms, so capture cost alone does not
explain the full performance gap. Prefill still spends substantial GPU time in
NCCL, FP4 expert products, HC writes/upmix and attention.

A bounded external RoCE block-count benchmark does not produce valid latency:
timing events captured internally first fail cudaErrorInvalidValue; the corrected
external-event repeated-graph attempt times out without timing output. No
transport geometry is selected from this failed harness, and it is not evidence
of a failure in the selected model graph, which completed. The service is
restored with the previously qualified8-block geometry. The throughput goal
remains unmet; next changes must target measured complete-model costs.

### Segmented FP32 Prefill Reductions

New opt-in `STRATA_TP_ROCE_SEGMENT_PREFILL=1` queues large FP32 reductions as
aligned chunks of at most2MiB on the existing stream-step transport. Each chunk
uses the original rank-order FP32 sum; all chunks finish before one host wait.
Ineligible calls stay entirely on NCCL and an enqueue failure never falls back
partway. Small reductions and the default path are unchanged.

Host tests cover whole-span coverage, offsets, FP32/BF16 sizing, in-place spans,
overflow, invalid counts and early enqueue failure. ARM candidate bd29f838 builds;
102profile/runner tests pass. Independent scoped review finds no blocker in
stream-step sequencing or send-slot reuse. Actual native validation completes on
all3ranks, including a2048-token prefill exercising segmentation and8concurrent
continuations. Four short TP1 logits meanKL0.00082384/max0.00328131 pass the frozen
policy with all top1 agreeing; continuations equal dbeda5d8. HTTP speed is pending.

Segmented prefill fails speed selection despite functional success: C1/C4/C8
wall4.084/9.443/15.966s versus selected3.703/8.672/14.525s. C1 prefill1271ms and
C8 packed prefill8315ms exceed the selected profile. Tools/8IDs pass, but many
serialized RoCE chunks lose to NCCL on these large messages. The option remains
disabled and dbeda5d8 is restored. No maximum-context qualification is claimed
for the rejected candidate. Overall goal remains unmet.

### Full-Warp Expert Gate/Up

The selected profile's gate/up remains the largest C8 target compute kernel.
Opt-in `STRATA_TP_GU_WARP32=1` uses32 lanes instead of16 per gate/up pair,
reducing each lane's160-subblock loop from10 to5 iterations. The block computes
4 output rows rather than8; all other projections, quantization and tails retain
the existing path. Default is16. Current padded256-row shards and valid224/208
rows preserve full-warp participation. The switch is allowlisted/fingerprinted.

Actual DG1 componentM24 takes1.163ms versus1.204ms, M3 takes0.123 versus0.132ms;
synthetic outputs are exact. Integrated expert GPU parity and invalid-ID checks
pass;103profile/runner tests pass. ARM36c47fb5 builds. Scoped review finds no
shuffle-mask or padding blocker. Full-model validation is running; component
timing does not establish service throughput.

Full-warp candidate36c47fb5 native3ranks exit0;4TP1 logits meanKL0.00005693,
max0.00021984, all top1 agree;8concurrent16-token continuations equal dbeda5d8.
Service timing is pending. The selected comparison remains C1/C4/C8 wall
3.703/8.672/14.525s versus Staticduo21 2.972/6.056/9.665s, with checkpoint/KV
differences explicitly retained. No parity claim is made.

Full-warp GU fails service selection: C1/C4/C8 wall3.710/9.002/14.763s versus selected3.703/8.672/14.525s. Tools and8identifiers pass. The component gain does not translate into overall throughput; option stays disabled and qualified dbeda5d8 is restored. No long-context test is attributed to this rejected candidate.

### Quantize Prompt Inputs Once Per Token

Large W4A4 expert prefill previously quantized every token once per routed
expert (ten identical quantizations). Opt-in
`STRATA_TP_PREFILL_QUANT_SCATTER=1` uses the pinned upstream FP4 scatter
quantizer: quantize once, then publish identical codes and row scales to the
packed destinations in the existing inverse route map. Down activations and
all products retain their existing layout and arithmetic. The option is
allowlisted and included in the rank configuration fingerprint.

Actual DG1 GPU tests compare every byte and scale for widths256/2560 and
17/2048tokens, including graph replay after changing the route permutation.
All are bit-identical. At2048tokens the input quantizer takes0.233ms versus
0.310ms for gather. Integrated expert parity and invalid-route tests pass;
158profile/runner tests pass. Independent static review finds no mapping,
scale or alignment blocker. ARM candidate60f46872 builds. Three-host full-model
qualification is running; the component gain does not establish service speed.

Candidate60f46872 completes native qualification on all3ranks (exit0), including
2048-token prefill and8concurrent16-token continuations. Four TP1 prefixes have
meanKL0.00004646/max0.00017225, all top1 agree. The eight continuation ID lists
equal selected dbeda5d8. All164profile/runner/logit comparison tests pass.
The persistent candidate is starting for the C1/C4/C8 speed check; the first
launch rejected the new option on followers with the old profile validator.
Synchronizing that file corrects the deployment mismatch. No speed acceptance
is claimed before HTTP measurement.

Actual candidate HTTP checks pass tools and8identifier isolation. C1/C4/C8
wall3.574/8.852/14.157s versus selected3.703/8.672/14.525s: small C1/C8 gains,
C4 regression in this single pass. Aggregate including prompt is
35.81/57.84/72.33tokens/s. This is mixed evidence, not Staticduo21 parity.
The candidate remains serving for its actual262128-input capacity check.

Actual maximum-context check passes:262128input tokens, zero cache hits,
HTTP200, exact answer `COBALT-84721`, wall146.500s, native prompt145.120s
(1806.3tokens/s). Visible answer10tokens and native committed14 are different
accounting scopes. This is faster than dbeda5d8's150.298s in the previous check;
it does not establish Staticduo21 long-context parity. Candidate60f46872 stays
serving, all3ranks READY, context262144,8slots, DG1 monitor exit0/oktrue.
The overall performance goal remains unmet.

### Packed Gate And Up For Prompt Experts

Opt-in `STRATA_TP_PREFILL_PACKED_GU=1` stores each expert's original gate/up
rows in one plane, without changing codes or scales. Large prefill issues one
512-output MMQ product instead of two256-output products. Activation reads the
gate/up halves and applies their separate original tail scales. Decode retains
its existing kernels and offset pointers. Load-time packing releases the two
old buffers; grouped expert storage is explicitly incompatible. The zero guard
and rank fingerprint are preserved.

Actual DG1 tests compare every gate/up output against the separate products.
At2048tokens, diverse routes take2.433ms versus2.580ms; highly reused routes
take0.574ms versus0.629ms. Expert parity and invalid-route checks pass;
160profile/runner tests pass. ARM candidatef047fd6c builds. Three-host native
qualification is running; no full-service speed claim yet.

Candidatef047fd6c native qualification completes with all3ranks exit0. Four TP1
prefixes have unchanged meanKL0.00004646/max0.00017225, all top1 agree. All13
token-ID groups match60f46872, including2048-token prefill and8concurrent
continuations. Independent scoped review finds no lifetime/stride/workspace
blocker. The profile requires W4A4 and rejects grouped storage;167selected
Python tests pass. The persistent candidate is starting for HTTP speed checks.

Packed gate/up fails service speed selection: C1/C4/C8 wall
3.669/9.041/14.360s versus60f46872's3.574/8.852/14.157s. HTTP tools and8identifier
isolation pass. No maximum-context check is run for this rejected candidate.
The option remains disabled and qualified60f46872 is restored. The small
component gain does not establish an end-to-end improvement.

### Vector Loads For Decode Gate And Up

Nsight Compute on actual DG1 GU16/M24 shows76.3% achieved occupancy but only
0.19eligible warps per scheduler per cycle. L1TEX scoreboard dependencies account
for91.9% of inter-issue cycles; memory bandwidth is not saturated. External
scale-conversion and shared-activation alternatives preserve arithmetic but
do not improve complete expert time. A down block-size change gives only1%.

Opt-in `STRATA_TP_DECODE_GU_VECTOR=1` instead copies the original gate/up code
pairs into aligned16-byte vectors and their two original scale bytes into a
16-bit plane. Decode reads both weight fragments with one vector access and
keeps the original Q8 products, scales, reductions and SwiGLU order. Original
planes remain for prefill; extra resident storage is about18GiB per rank.
Only batches above8rows use the vector kernel because M3 does not improve.

Actual DG1 M24 expert layer takes1.182ms versus1.296ms, relative L2zero. M3
is0.109ms for both. Integrated expert parity and invalid-route checks pass;
168profile/runner/logit tests pass. ARM185244c7 builds. Three-host native
qualification is next; service and max-context memory acceptance remain unproven.

Candidate185244c7 completes native qualification: all3ranks exit0, four TP1
prefixes retain meanKL0.00004646/max0.00017225 with all top1 agreeing, and all13
token-ID groups equal60f46872. This includes2048-token prefill and8concurrent
continuations. The actual test profile has Decode FP4 and Native Grouped off;
their incompatible combinations with vector GU are now rejected explicitly.
170selected Python tests pass. Independent scoped review finds no indexing,
padding or lifetime blocker. Persistent candidate is starting for HTTP timing.

Vector GU fails overall service selection: C1/C4/C8 wall3.721/8.536/14.301s
versus60f46872's3.574/8.852/14.157s. C4 improves, C1/C8 regress, and the extra
18GiB per rank is not justified by this mixed result. HTTP tools and8identifier
isolation pass. The option stays disabled and60f46872 is restored. No
maximum-context qualification is attributed to this rejected candidate.

### Selected Profile On All Three Ranks

Actual Nsight node captures run the selected60f46872 binary on all3ranks;
all exit0. Workload: warm64/8, C1PP2048/TG64 and C8PP2048/TG32. Each rank has
its own SQLite trace; timestamps are not assumed aligned across hosts.

For the same11replays of the24-row target window, cumulative kernel work on
ranks0/1/2 is: gate/up144.42/132.16/129.05ms, down92.89/92.54/91.30ms,
GDN recurrence56.74/44.97/44.60ms, and RoCE84.66/170.43/166.38ms. Counts match
for each listed kernel. These sums are not request wall time. RoCE median
kernel durations are53.7/98.6/116.2us. The faster local providers on followers
coincide with more collective wait; collective spans cannot be treated as pure
network transfer cost. Rank0's6GDN key heads and224local routed FF channels
remain a measured source of asymmetric work. Further seam analysis is pending.

After capture, selected60f46872 is restored and actual health reports all3ranks
READY; DG1 monitor exits0 with oktrue, context262144 and8slots. No new speed
acceptance is claimed from the instrumented run.

### Gate/Up Weight Prefetch

External cuBLAS NN pretransposition is slower on almost every representative
resident F16 projection and is not integrated. The RoCE proxy already spins
across short gaps. Actual GU memory-latency evidence instead motivates
`STRATA_TP_GU_PREFETCH=1`: native warp16 decode hints the next gate/up F4 block
into L1 one iteration before use. The guard keeps addresses inside the original
40-block row. No codes, scales, arithmetic or buffers change. Only windows above
8rows use it; incompatible decode routes are rejected by the profile.

DG1 M24 expert layer is exact and takes1.165ms versus1.281ms. With valid224
channels it takes1.115ms versus1.171ms. Integrated expert parity/invalid routes
and175selected Python tests pass. ARM e0aca424 builds. Native three-host
qualification is running; component results do not establish service speed.

Candidatee0aca424 native qualification completes with all3ranks exit0. Four TP1
prefixes retain meanKL0.00004646/max0.00017225 and all top1 agree. All13token-ID
groups equal60f46872, including2048-token prefill and8concurrent continuations.
Independent review finds no pointer/default-path blocker. The persistent
candidate is starting for C1/C4/C8 HTTP measurement.

GU prefetch fails overall speed selection: C1/C4/C8 wall3.658/8.696/14.340s
versus60f46872's3.574/8.852/14.157s. C4 improves slightly but C1/C8 regress.
HTTP tools and8identifier isolation pass. The option stays disabled and
qualified60f46872 is restored. No maximum-context test is run for the rejected
candidate. The Staticduo21 performance goal remains unmet.

### QSA Prompt Launch Aggregation

Native QSA prompt processing used32queries per score/top-k/attention call even
though the kernels accept larger batches. Opt-in `STRATA_TP_QSA_PROMPT_BATCH=256`
reduces these host launch groups up to8x. Scores scratch grows from32to256rows
(about56MiB extra/rank). The IDs/query buffers already cover the prompt chunk.
If tensor-core prompt attention declines, decode attention still runs in32-row
pieces to preserve its scratch bounds. Per-query causal limits remain active.

Actual DG1 TP QSA tests compare whole256-query attention with32-query calls:
bit-identical, plus existing FP64 checks at contexts4096/1500/2100, zero failures.
180selected Python tests pass; independent static review finds no shape/scratch
blocker. ARM1ae7cbf2 builds. Full-model qualification is next.

Candidate1ae7cbf2 completes native qualification with all3ranks exit0. Four TP1
prefixes retain meanKL0.00004646/max0.00017225 with all top1 agreeing. All13
token-ID groups equal60f46872, including2048-token prefill and8concurrent
continuations. The persistent candidate is starting for HTTP measurement.

Actual HTTP tools and8identifier isolation pass. C1/C4/C8 wall
3.624/8.393/14.186s versus60f46872's3.574/8.852/14.157s: C4 improves about5.5%,
C8 is effectively unchanged in this single pass and C1 regresses about1.4%.
Aggregate including prompt is35.32/61.00/72.18tokens/s. The actual262128-input
capacity check is running before selecting the new grouping profile.

Actual maximum-context check passes:262128input tokens, zero cache hits,
HTTP200 and exact `COBALT-84721`. Wall134.807s versus60f46872's146.500s;
native prompt133.424s (1964.6tokens/s). Visible answer10tokens and native
committed14 are distinct accounting scopes. The grouping profile1ae7cbf2 is
selected for the measured C4 and long-prefill improvement; C8 remains effectively
unchanged and C1 has a small regression in the single-pass comparison.
All3ranks finish READY, context262144,8slots, DG1 monitor exit0/oktrue. The
Staticduo21 parity goal remains unmet.

### QSA Identity Selection And INT8 MMA Prompt Attention

Opt-in `STRATA_TP_QSA_PROMPT_FAST=1` enables the existing tested local INT8-MMA
prompt attention and skips block-score computation when every query in a
contiguous prompt sub-batch has at most2051KV cells. All top-k implementations
already emit identity IDs before reading scores in that case. Indexer/KV append
still runs for later positions. Decode rows keep their existing path.

TP QSA256-query component tests already validate INT8 MMA against FP64 at
contexts4096/1500/2100 with errors of a few1e-6 and roughly1.5x component speed.
Independent static review finds no namespace, causal-bound or identity-read
blocker. ARM195b3d06 builds. Full-model numerical/functional qualification is next.

Native195b3d06 completes all3ranks exit0. Four TP1 prefixes retain
meanKL0.00004646/max0.00017225 with all top1 agreeing; all13token-ID groups equal
the selected1ae7cbf2 profile, including2048-token prefill and8concurrent
continuations.181selected Python tests pass. Persistent HTTP measurement is next.

HTTP tools and8identifier isolation pass. C1/C4/C8 wall3.617/8.481/13.873s
versus1ae7cbf2's3.624/8.393/14.186s: C8 improves about2.3%, C1 is similar,
C4 regresses about1%. Aggregate including prompt35.39/60.37/73.81tokens/s.
Actual262128-input capacity check is running; Staticduo21 parity remains unmet.

Actual maximum context passes:262128input, zero cached tokens, HTTP200, exact
`COBALT-84721`, wall135.254s and native prompt133.903s (1957.6tokens/s), similar
to1ae7cbf2's134.807s.195b3d06 stays selected for the measured C8 improvement;
C4 has a small regression in this single-pass comparison. All3ranks READY,
context262144,8slots; DG1 monitor exits0/oktrue. Overall parity remains unmet.

### Resident F16 Output Head For Large Windows

Opt-in `STRATA_TP_HEAD_F16=1` expands only the rank-local Q8 output-head shard
at load and uses cuBLAS F16 GEMM for windows above8rows. The verifier supplies
its original FP32 head activation through a new default-disabled hook. Smaller
windows retain Q8 MMVQ. Both compact greedy candidates and full logits retain
their existing reduction/publication protocol. Model source bytes stay intact;
F16 activations/expanded weights change rounding and require numerical checks.

DG1 synthetic rank shapeN82773/K2560/M24 takes1.923ms versus native2.705ms;
selected CPU-double relative L2 is0.000341, pair difference0.00575. M3 is slower
and keeps the existing path. ARM46ef3deb builds14/14, CPU fast-execution contract
passes,182selected Python tests pass. Full-model concurrent-window logits are
required because short prefixes do not exercise this large-window path.

Initial46ef3deb native3ranks exit0 and all13token-ID groups equal195b3d06.
The existing diagnostic wrote only initial sample logits, so an opt-in-only
capture in FastModel::sample now records accepted MTP rows too. Same-bin
b9a86241 Q8 and F16 runs each finish all3ranks exit0 with140logit vectors.
All140 positions pass the frozen gate: meanKL0.000006208/max0.000132799,
top1agreement100%. No API speed claim follows from diagnostic timing.

Resident head fails overall selection: HTTP C1/C4/C8 wall3.690/8.690/13.727s
versus195b3d06's3.617/8.481/13.873s. C8 improves about1%, while C1/C4 regress
about2%; a single pass does not justify selecting it. Tools/8identifier checks
pass. HEAD_F16 remains disabled; selected195b3d06 is restored. No maximum-context
qualification is claimed for the rejected head candidate.

### Exact INT8 MMA Down Projection Probe

External decode probes use the original NVFP4 integer codes with Q8 activations
and INT8 m16n8k16 tensor-core products per16-value scale group. Single-route and
up-to8-route grouped forms preserve the mathematical integer products and expert
scales. Actual DG1 M24 complete-expert outputs have relative L2zero in both
cases. Single-route takes1.387ms versus DP4A1.221ms; grouped takes2.249ms versus
1.225ms including grouping. Both are rejected before service integration.
Selected195b3d06 remains serving; health and monitor confirm all3ranks READY.
No performance claim is made from the use of tensor-core instructions alone.

### Reproduce Real Expert Windows

Synthetic expert component wins have repeatedly failed overall selection.
Diagnostic-only `STRATA_TP_EXPERT_FIXTURE_DIR` captures the first layer0 native
decode window of16-40rows into preallocated device buffers, then writes original
rank0 expert planes, scales, activations, route IDs/weights and local output after
bounded step completion. No copy/readback belongs inside a captured graph.
The diagnostic refuses combined/grouped storage and leaves serving defaults
unchanged. The next component harness must reproduce this actual output before
measuring candidate expert kernels; synthetic timings alone are insufficient.

Actual capture run finishes all3ranks exit0. The captured window has24rows,
240routes but32distinct experts, largest expert group15. Original weights,
inputs and outputs are saved with a SHA256 manifest. Replay on DG1 reproduces
every output exactly and takes418.255us per complete expert graph, versus about
1.2ms for the earlier diverse synthetic fixture. On this real window, prefetch
is403.878us (3.4% gain), warp32 is412.569us (1.4% gain); these do not qualify a
service change. Selected195b3d06 is being restored after the capture.

### Decode Route Ordering For Cache Locality

Opt-in `STRATA_TP_ROUTE_ORDER=1` stably sorts the GPU launch order by expert ID
for native decode windows above8rows. Each GU/down block still reads and writes
the original route index; quantization, weighted combine and summation order do
not change. A bounded O(R^2) GPU permutation uses at most400routes and the
existing route-sized identity buffer. No weights or activations are moved.
Conflicting alternate decode kernels are rejected by the profile.

Actual captured24-row window reproduces output exactly and takes364.098us,
versus418.255us before ordering. Integrated GPU tests pass parity, changed-ID
graph replay and invalid IDs -1/512.188selected Python tests pass. Independent
review finds no permutation/indexing blocker. ARM b8c55542 builds. Full-model
qualification is running; the captured-window gain is not service acceptance.

Native b8c55542 completes all3ranks exit0. All140 full-vocabulary logits are
identical to the same-arithmetic Q8 reference (KLmean/max0, top1agreement100%).
The graph changed-ID and invalid-ID tests establish that the permutation is
recomputed on replay. The next gate is actual HTTP C1/C4/C8 speed.

HTTP tools and8identifier isolation pass. C1/C4/C8 wall3.645/8.556/13.748s
versus195b3d06's3.617/8.481/13.873s. A second C8 pass on the same service is
13.421s (76.30aggregate tokens/s), versus13.748s (74.49) in the first pass.
The small gain has run-to-run variation; an actual262128-input capacity check
is running. These numbers do not establish Staticduo21 parity.

Actual maximum context passes:262128input, zero cached tokens, HTTP200 and
exact `COBALT-84721`, wall135.527s, native prompt134.098s (1954.8tokens/s),
similar to195b3d06's135.254s. Route orderingb8c55542 stays selected with the
small C8 improvement and unchanged mathematical outputs; the measured speedup
has run-to-run variation. All3ranks READY, context262144,8slots; DG1 monitor
exits0/oktrue. Staticduo21 parity remains unmet.

### Reuse Weight Fragments Across Expert Routes

Opt-in `STRATA_TP_GROUP_REUSE=4` groups native decode visits to each expert in
sets of at most4. Each GU/down fragment is loaded/decoded once and applied to
the grouped activation rows. Outputs stay in compact route order and the
existing inverse map restores the original weighted sum. Per-route DP4A,
scale application and warp reductions retain their original arithmetic order;
padding is zero and valid224/208channels are preserved. Groups of1 use their
own specialization. Only windows above8rows use this path; C1 is unchanged.

Actual captured24-row window takes308.793us versus ordered364.098us and
reproduces every output exactly. GPU parity, changed-ID graph replay and invalid
IDs pass;189selected Python tests pass. Independent static review finds no
scale/padding/permutation blocker. ARM9835594a builds. Full-model qualification
is next; component gains do not establish service acceptance.

Native9835594a completes all3ranks exit0. All140full-vocabulary logits are
identical to the original Q8 reference (KLmean/max0, top1agreement100%).
Persistent candidate is starting for actual HTTP C1/C4/C8 speed measurement.

Group reuse is not selected: HTTP tools/8identifier checks pass, but C1/C4/C8
wall3.578/8.610/13.581s and a second C8 pass13.908s do not improve reliably over
route ordering13.748/13.421s. Eight-slot mean verify time88.64ms versus87.71ms
also shows no gain. The layer0 fixture gain does not generalize to the model's
other layers. No maximum-context test is run for this rejected candidate.
GROUP_REUSE stays disabled and selectedb8c55542 is restored.

### Triangle Route And Latency Audit

Actual three-host port states are ACTIVE/LinkUp at200Gb/s. A production27/64
generation increments only the native P2p HCA counters, with matching byte
counts on DG1f1-DG2f1, DG1f0-DG3f0 and DG2f0-DG3f1. All observed error/discard
counter deltas are zero. Current NCCL logs on all3ranks report2.30.7 and NET/IB,
without NET/Socket data fallback. Bootstrap/control Ethernet is separate.

An independent native RoCE benchmark uses the same HCA/routes and8cooperative
CTAs/512threads,32ordered reductions per graph, one warm and3timed replays.
All3ranks exit0 and exact sums pass. Per-operation event durations across ranks
and repeats:48B20.6-39.8us,30KiB28.0-49.2us,240KiB70.8-94.3us,1MiB203.1-210.0us.
These are complete collective durations, not physical wire latency.96serial
240KiB reductions imply roughly7-9ms per target window before computation.
Network latency remains a relevant optimization target; correct routes/200Gb
link state do not prove parity with Staticduo21's native transport. A matching
Staticduo21 transport benchmark is still absent. Evidence and reproducible
harness: `/home/staticduo/tmp/strata-nvfp4-tp3/network-audit/report.json`.

### Adaptive RoCE Launch Geometry

An isolated transport reconstruction applies the Staticduo21 recipe's exact
proxy/runtime patches over helper modules from the configured base image. Exact
historical CuTe helper hashes are absent, so this is a recipe reconstruction,
not a proven snapshot of the running historical service. All3ranks pass sums
and512operations. Median collective times:48B16.9us,30KiB23.5us,240KiB67.6us,
1MiB200.0us, versus Strata fixed8CTA29.3/37.6/77.1/204.0us. Both use the same
triangle; there is measurable small-message launch/synchronization overhead.

Opt-in `STRATA_TP_ROCE_ADAPTIVE=1` maps the validated maximum8CTA setting to
1/2/4/8blocks by payload (two16-byte packs per thread), keeping large messages
at8. Device epoch and arithmetic stay unchanged. Fixed1CTA tests improve tiny
messages but regress240KiB/1MiB, supporting size-dependent selection.

Actual3rank adaptive benchmark medians20.1/31.2/79.7/203.9us; all exact sums
pass. A mixed1/2/4/8CTA graph passes200reductions/rank across replays.190selected
Python tests pass; scoped review finds no epoch/barrier blocker. ARM748cd4a4
builds after correcting an ambiguous constant name. Full-model qualification is
next. No overall service improvement is claimed from these microbenchmarks.

Native748cd4a4 completes all3ranks exit0; all140full-vocabulary logits are
identical to the Q8 reference (KLmean/max0, top1agreement100%). The persistent
candidate is starting for actual HTTP timing before speed selection.

Actual HTTP tools/8identifier checks pass; C1/C4/C8 wall3.584/8.611/13.595s,
aggregate35.72/59.46/75.32tokens/s. C1 is slightly faster than fixed8CTA3.645s,
C4 is slightly slower, and C8 remains inside prior13.421-13.748s variation.
This does not establish an overall throughput gain. Actual262128-input capacity
qualification is running; do not attribute the entire recipe gap to networking.

Actual maximum context passes:262128input, zero cached tokens, HTTP200 and
exact `COBALT-84721`, wall135.025s and native prompt133.625s (1961.7tokens/s),
similar to fixed geometry. Adaptive748cd4a4 remains selected for the measured
small-message latency reduction and slight C1 improvement; an overall HTTP
throughput gain is not established. All3ranks READY, context262144,8slots;
DG1 monitor exits0/oktrue. Staticduo21 parity remains unmet.

An earlier nearest-scale activation quantizer is retained opt-in but not selected:
it removes the five-candidate error search and preserves original model weights.
Component gather/scatter/layout and numerical tests pass, but only0.03-0.04ms
is saved per intermediate quantizer call; no full-model quality/performance
qualification is claimed and the new candidate keeps that option disabled.

An independent three-host NCCL2.30.7 microbenchmark with production network
settings shows only a small Simple protocol gain: rank0 20MiB1.657ms versus
Auto1.714ms, 80MiB6.277ms versus6.384ms. All ranks return exact sums. No protocol
change is selected. Production NCCL spans can include peer-arrival wait;
the existing one-rank profile does not prove how much comes from rank imbalance.

### Control-channel latency audit

The data triangle is verified by actual request counters, not just configuration.
All three direct edges carry matching peer TX/RX with no reported error/discard
increments. 200Gb/s is the negotiated link rate; sustained payload bandwidth
at that rate has not been established. Current selected NCCL logs confirm
NET/IB without NET/Socket data fallback.

The separate TCP control protocol costs about1.4-1.6ms per operation in a short
three-host test, including its four logical protocol flights. Direct-link TCP,
bounded2ms userspace spin and CPU5 affinity give mixed or worse results across
agreement/broadcast/exchange, so no control configuration change is selected.
Source review finds TCP_NODELAY already enabled, no steady-state1ms sleep,
and preserves the final publication barrier. A millisecond poll timeout does
not force a millisecond wait when a socket becomes readable.

Evidence: `network-audit/control-comparison.json` and `network-audit/CONCLUSION.md`
under `/home/staticduo/tmp/strata-nvfp4-tp3/`. All three ranks complete with
verified payloads. The selected service stays READY, context262144 and8slots;
DG1 monitor exits0/oktrue. Transport/control latency remains part of the gap,
but there is no evidence of GPU data taking the management TCP route. The
full performance acceptance target remains unmet.

### Rank Placement Candidate

The three-rank trace shows that rank0 owns both larger local workloads:
FF224 versus208/208 and GDN6 versus5/5 key heads. In the24-row target graphs,
rank0 GU/down/GDN accumulated144/93/57ms, versus132/93/45 and129/91/45ms on
the followers. These are kernel work sums, not end-to-end wall times.

Opt-in `STRATA_TP_BALANCED_RANKS=1` requires balanced experts and permutes
existing shards: FF physical ranges224..431,0..223,432..639 and GDN key ranges
6..10,11..15,0..5. The larger FF workload goes to rank1 and larger GDN workload
to rank2. Shared FF stays256/192/192. No weight codes, scales, mathematical
operations or model geometry are removed. The option is part of the join
fingerprint, and invalid profile combinations fail before execution.

CPU coverage tests verify that legacy and permuted layouts cover each FF
channel and GDN head exactly once.61 profile/logit comparison tests and the
ARM build pass; independent static review finds no blocker. Three-host
full-vocabulary quality qualification rejected the combined permutation:
all3ranks exit0, but140logits exceed the frozen quality limits (meanKL1.80,
max29.55, top1agreement90%). Some later rows have diverged histories, so their
KL cannot isolate arithmetic error; same-history rows also fail. The combined
permutation is not eligible for service deployment.

The corrective isolation keeps FF shards in their original physical rank order
and permutes GDN only. CPU coverage and the incremental ARM build pass, but
the3rank retry also fails the frozen limits: meanKL0.580, max25.54 and top1
agreement96.43%. Some later rows have diverged histories; this does not prove
a specific kernel defect. Neither placement is selected or speed-benchmarked.
The experimental source changes are archived outside the repository and removed;
the validated748cd4a4 service is restored. Further investigation would require
fixed-history layer/state comparisons, not widening the frozen policy. The full
Goal remains unmet.

### Reuse The Staticduo21 B12X Backend

Jordi explicitly requests the same backend as vLLM. Prioritize a native C++
adapter to the installed B12X1.3/CuTe4.7 implementation and its actual size
selection. The installed dispatcher selects dynamic execution above8tokens;
the Staticduo21 compile cache contains NVFP4 dynamic M16xN128 artifacts for
E512,K2560,I256,top10. Earlier A16 probes do not qualify this path.

Export the original dynamic launch through CuTe's C interface, retaining its
routing, native NVFP4 MMA, activation, intermediate quantization and reduction.
Keep load/compile/allocations outside CUDA capture. Replicate its scratch sizing
and initialization contract, not just its GEMM body. Reuse the original codes
and scale bytes through lossless source-layout repacking. All24576original
expert tails have bit-identical gate/up global scales, so the original single
FC1 alpha is sufficient without patching B12X arithmetic.

Acceptance: verify repack roundtrip, original B12X reference versus exported C
launch on real routed inputs, graph replay, then the frozen full-model quality
and capacity policy and representative service timing. No kernel-selection or
performance equivalence claim follows from using the same backend name. The
selected748cd4a4 service remains available while artifacts are prepared.

The original B12X dynamic reference runs on the real24-row fixture in201us;
the exported C ABI captures/replays in208us versus364us for selected DP4A.
Same-activation B12X reference relativeL2 is0.00619 and exported-C versus
Python is0.00884, both below0.02. Repack/alpha/invalid-route GPU tests pass,
as do62 selected Python tests. These component results do not qualify the model.

Full-model A4 qualification completes on all3ranks but fails the frozen logit
gate (meanKL1.236,max29.70,top1agreement93.57%). Some sampled histories diverge;
same-history rows also exceed limits. Do not deploy this A4 candidate. Restore
selected748cd4a4, verified READY with262144context,8slots and working DG1 monitor.

The corrective attempt uses the same B12X dynamic backend with MXFP8 activations
and original NVFP4 weights (`w4a8_nvfp4`). Its real fixture graph takes263-275us,
relativeL2 to DP4A0.04075. Original gate/up scale decomposition into UE8M0 plus
E4M3 residual reconstructs every tested scale exactly. The native adapter rejects
any lossy decomposition; GPU coverage passes. The same-activation B12X A8 oracle
comparison is0.015596relativeL2, below the frozen0.02component limit. A8 export
and whole-model quality qualification are pending. No performance-parity claim
is made.

Native A8 C ABI also passes the real-fixture Python comparison (relativeL2
0.00598) and captures/replays in216us. The first full-model load correctly
rejects three rare original scale pairs that B12X's usual decomposition would
round: layer2up,35gate,38up. A complete CPU scale audit finds that reducing the
shared exponent by one bit for those pairs reconstructs every original scale
exactly. The adapter applies this fallback only after an exact reconstruction
failure, and retains the rejection for pairs that still lose precision. GPU
tests cover ordinary, recoverable and unrecoverable pairs. The corrected native
A8 model qualification is running; no full-model acceptance is claimed yet.

Corrected A8 completes all3ranks and preserves every selected token across
140logit positions. MeanKL0.002981 passes, but maxKL0.170736 exceeds the frozen
0.05limit (two rows exceed0.05). Do not deploy or speed-qualify the A8 model.
Restore selected748cd4a4 with a fresh evidence directory; do not reuse an
existing directory, which the service runner intentionally refuses after asset
validation. Next investigate the actual B12X W4A16FusedMoeKernel/GemmKernel
Tensor Core backend. The previous A16 probes used micro.py's software path and
do not qualify this separate implementation. Keep the policy unchanged.

### Strict RDMA Interhost Contract

Jordi explicitly requires zero TCP between Strata processes on DG1/DG2/DG3,
including startup and errors. The earlier bootstrap-TCP exception is withdrawn.
NCCL NET/IB is insufficient proof because its connection/bootstrap paths can
open TCP sockets. The strict profile avoids NCCL communicator creation, uses
RDMA-CM SEND/RECV for authenticated Session bootstrap and errors, and native
RoCE collectives for every tensor size. Larger tensors are chunked with explicit
rank-major all-gather placement. The local HTTP API is not an interhost link.

Steady-state RDMA control passes a real three-host test at161/160/169us median
for agreement/broadcast/exchange, versus TCP1603/1452/1396us. All ranks return
exact payloads; CPU protocol and error tests pass. This test still used TCP
bootstrap and does not satisfy the strict contract. RDMA-CM bootstrap and the
RoCE-only tensor API are being implemented; final acceptance requires process
socket/traffic evidence across startup, generation, cancellation and failure.

The actual fused B12X W4A16 native model completes all3ranks,140top1selections
agree and meanKL0.00314 passes, but maxKL0.232 fails the frozen0.05limit.
Do not select its arithmetic or claim service speed from its173us component.
Restore selected748cd4a4 while the strict RDMA transport is qualified.

Strict RDMA-CM bootstrap now passes a simultaneous3host run, including exact
large reductions/gathers. The initial listen race is fixed with a bounded
retry for transient rejected-service events. Wrong credentials fail without
READY; detecting ranks return credential failure in about0.2s. The complete
native model also exits0 on all3hosts and its112syscall trace files show no
AF_INET/SOCK_STREAM socket attempts during startup/generation/shutdown.

Same-binary hybrid-versus-strict comparison still fails the frozen quality
gate (meanKL0.5685,max25.55,top1agreement96.43%). This isolates the difference
to the transport switch; the cause is not yet confirmed. Investigate the
large-prefill reduction association and chunk mapping with cancellation-sensitive
values before another model variant. The four short prompt logits agree exactly.
Keep the strict transport optional until the numerical discrepancy is corrected.

The first ring-order implementation reproduced standalone SIMPLE reductions at
5/10/20/80 MiB exactly, but full-model quality still failed (meanKL0.5543,
max23.8626). Actual model TRACE now identifies the prompt reduction as
5240320 floats (2047 rows), RING SIMPLE, eight channels. NCCL partitions
SIMPLE operations in16KiB cells, not eight equal element spans. Correct this
mapping and capture the first six prompt reduction inputs/outputs on all ranks
to establish parity at the first divergence before another full qualification.
The earlier LL Nsight observation concerns other message sizes; it does not
establish the protocol of this large prompt operation.

The cell partition correction reproduces all six captured prompt reductions
bit-for-bit on all three ranks, including their inputs. The dedicated GPU
regression passes. Final candidate5541e0e9 passes all140full-logit positions
with mean/maxKL0 and100%top1agreement. All three ranks exit0;111syscall traces
contain zero Internet stream socket attempts across startup/generation/shutdown.
The strict service profile is staged with a fresh UUID and port30586. HTTP
performance and final loaded/monitor checks remain required before delivery.

The strict service is loaded on all3ranks and DG1 monitor passes. Its first
PP2048/TG128 C1/C4/C8 pass measures37.56/56.23/64.97aggregate output tokens/s
including prefill and queue time. C1 native decode is63.0tokens/s, but prefill
falls from roughly0.956s to1.286s. Large reductions currently gather into
scratch and launch a second sum kernel per2MiB chunk. Fuse the identical
SIMPLE cell-aware sum into the RoCE collective, preserving staged input and
graph ordering; validate parity before comparing performance. The overall
Staticduo21 performance criterion remains unmet. Do not restore Staticduo21.

The fused native all-reduce candidate1963e3b5 passes the same140logits with
mean/maxKL0 and100%top1agreement. Its111socket traces again show zero Internet
stream sockets. The CPU backend contract test also passes, including range
rejections and forwarding global offsets through the collective launch. The
qualified service is starting with port30588 and fresh UUID; measure its HTTP
performance after loading. The preceding strict5541 service passed actual
262128input+16max-output retrieval (COBALT-84721),178.809s,1473.1prefill tokens/s.

The fused1963 persistent service is loaded on all3ranks. Its same-prompt
PP2048/TG128 C1/C4/C8 checks pass at38.07/57.95/67.24aggregate output tokens/s
including prefill and queue time. C1 native prefill1682.6tokens/s and decode62.3.
This improves the first strict transport but does not meet the Staticduo21
target (historical all-phase43.07/84.55/105.95); retain the comparison caveats
about tokenizer IDs, KV format and MTP configuration. Independent static review
finds no blocker in staged in-place input or float4 partition boundaries.

Final1963 capacity check passes actual262128input+16requested output with
the exact COBALT-84721 retrieval response:170.460s wall,1544.9prefill tokens/s.
All3ranks return READY and DG1 `strata monitor --once --output json` reports
ok with the loaded262144context model and8slots. Leave this strict service
running. The zero-TCP requirement is qualified; overall speed parity is not.

### Paused For Public Collaboration

On2026-11-01 Jordi explicitly paused development, requested saving all work and
restoring Staticduo21. This supersedes earlier instructions to leave Strata
serving. The last qualified Strata binary is1963e3b5 with2MiBslots. A new
opt-in slot-size selector supports2/8/16/32MiB;71profiletests and the runtime
build pass. A three-host cancellation-sensitive component test reports exact
results at all four sizes. At8MiB the roughly20MiB/80MiB reductions measured
4.32/24.47ms versus8.39/34.27ms at2MiB. These are component timings only.

The8MiB full-model run was stopped during the pause and has no qualification
receipt. A peer RDMA retry error during coordinated stop is not evidence of
an intrinsic slot-size defect. Retain the2MiB qualified profile until a fresh
full-model check passes. Jordi requested publishing the current code in the
public `multistrata` fork and inviting Oppo through Marlin to help optimize it.
