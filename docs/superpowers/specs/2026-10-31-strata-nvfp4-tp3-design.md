# Strata NVFP4 TP3 Performance Design

## Goal And Delivery Contract

Build the final native Strata execution path for three GB10 hosts over the existing
RoCE fabric. The reference product is Staticduo21 TP3, not the slow IQ3 TP3 oracle.
The user authorizes stopping Staticduo21 and requires Strata to remain serving at
completion. DG1 must provide `strata` on PATH and a working `strata monitor`.

The maximum context is 262144, matching Staticduo21. Compare the same prompt IDs
where tokenizers agree, output lengths, context depth, concurrency, clock policy,
thinking mode, cache state and visible output accounting. NVFP4 and Staticduo21's
checkpoint differ; identify both and report quality separately. Preserve original
model geometry and every stored NVFP4 code/scale. No expert dropping, approximate
prefix skipping or reduced context may stand in for reaching the target.

Approximate parity means at least 95% of the same-window Staticduo21 throughput
in prefill and generation for each selected C1/C4/C8 case, with no failed requests,
state leakage or unreported cache reuse. Use PP2048/TG128 at depth 0 and 65535,
three warm measurements, as the existing Staticduo21 report does. Confirm one
actual 262144-context capacity workload separately. C8 is the test ceiling;
configured queue/slot limits must be distinguished from simultaneously resident
full-length contexts. Historical numbers are planning anchors, not a final A/B.
If the target is missed, the Goal stays active and the measured bottleneck directs
the correction; compilation or a gain over the old oracle is not completion.

## Model And Storage

Use the ready Strata NVFP4 GPTQ pack
`Maximilian228/Qwen3.8-Flash-Next-NVFP4-GPTQ-Strata`, pinned revision
`9b70ad3f640b2546449bd05387407a64113acf8b`. It has original 48-layer geometry,
512 experts, hidden 2560, FFN 640, NVFP4 type 40 gate/up/down and per-expert scale
tails. Runtime files are experts.bin, dense pack, Q8_0/BF16 dense GGUF, FP8 PLE,
BF16 embedding and MTP. Pin SHA256SUMS and verify downloaded files.

Archive the two IQ3 GGUFs and small pack on NAS, verify against source hashes and
preserve qualification evidence before removing only the retired DGX model files.
Do not remove PLE/tokenizer assets until the independent NVFP4 equivalents work.
Keep at least 50 GiB free per Spark. Do not change drivers, firmware or networking.

## Architecture

One CUDA process per host, TP3/PP1. Reuse Strata's fast `Prefill`, `Verifier`,
NVFP4 grouped expert kernels, MMQ, tensor-core GDN and batched QSA. Add explicit
transport hooks to those paths. The old exact `ModelForward` remains a diagnostic
reference; it is not the production executor for this profile.

### Tensor Layout And Communication

- Gate/up are column parallel, down is row parallel. The balanced profile uses
  FFN cuts 224/208/208, padded to 256 for storage. Down repacking copies each
  original 16-value code/scale group without requantization. The initial
  256/192/192 layout remains available. Original scales apply to their projection.
- Each rank keeps its hidden channels, quantizes locally for the chosen production
  arithmetic and computes its local down contribution. Never gather hidden vectors
  or original FMA operands merely to reproduce single-host bit patterns.
- Apply route weights and sum local routed contributions before one AllReduce of
  `[rows,2560]`; apply replicated/shared terms once. Fuse shared contributions into
  that reduction only when they are actually sharded.
- Attention/GDN use local heads and local input columns of their output projection,
  followed by one output AllReduce. No rank-zero full output projection and no
  gathered full head activation in the final fast path.
- Replicate small routers/norms/HC and both QSA KV heads initially. Routes are selected
  deterministically on GPU from replicated input. Validate at step boundaries or
  sampled diagnostics, not a host readback and control handshake per layer.
- Target two mathematical tensor reductions per layer (mixer and FFN). Any extra
  communication must have an explicit purpose and measured cost.

### Kernels And Batching

Decode uses resident rank-local NVFP4 weights, fixed buffers and capture-compatible
GPU route/group plans. Reuse tuned native kernels for small rows and tensor-core
kernels when their shape wins. Preserve s_gate/s_up before SwiGLU and s_down after
down; never fold tiny scales into an underflowing activation representation.

Prefill uses 1024-8192 token chunks constrained by measured memory, dense GEMM,
chunked tensor-core GDN, batched sparse attention and expert-major NVFP4 MMQ.
Compile the architecture-specific FP4 unit for sm_121a and prove the dispatch;
no silent W4A8 fallback may be reported as FP4 tensor-core execution.

Continuous decode batches contain independent active slot rows and their own GDN,
QSA, PLE and position state. Reuse Verifier slot-row kernels and graph caches keyed
by batch shape. A shared GPU launch reads weights once for the active rows; a loop
over complete per-slot forwards is not batching. Prefill admission is chunked so
it cannot starve decode. MTP uses the actual draft pack and target verification;
report acceptance and visible tokens/s, including its cost, as Staticduo21 does.

### State, Graphs And Failure

Allocate weights/workspaces before capture. Use stable pointers and explicit active
row metadata; capture NCCL uniformly on all ranks. Keep one step-level agreement,
error propagation, bounded timeout and no published output before successful group
commit. Reset/cancel retires or safely reuses graphs without stale slot state.
Keep the existing SparkRun lifecycle and original Staticduo21 recipe for recovery,
but do not restore Staticduo21 at normal completion of this Goal.

### Memory And Monitoring

Context capacity is budgeted from resident sharded weights, replicated dense/PLE,
GDN state, dynamically allocated KV pages, MTP and graph workspaces. Use Strata's
supported compact KV representation with a documented quality check; count unified
memory once. Admit requests against real free-page budgets and queue excess work.

The normal Strata server supplies OpenAI/Anthropic APIs and versioned telemetry.
Expose true TP ranks, context, active/queued rows, phase timings and kernel dispatch.
DG1 CLI defaults to the actual local endpoint; verify monitor once and the TUI.

## Numerical And Quality Contract

Separate storage fidelity from arithmetic equivalence. Stored NVFP4 codes/scales
must remain byte-identical. Fast distributed sums need not reproduce TP1 FMA order.
Before measurements, freeze a same-pack fast TP1 reference and quality policy.
Evaluate finite outputs, relative projection error, recurrent drift, logits KL,
margin-aware top-token agreement, and deterministic same-setting replay. Include
long retrieval, extraction, JSON, code and tool/isolation cases. Use upstream
NVFP4 W4A4 quality evidence to choose initial budgets, then record the actual
reference noise floor; do not widen thresholds after a failing candidate simply
to make it pass. Quantization quality versus Staticduo21 is a separate report.

## Implementation Boundaries

New transport hook contracts live beside core/prefill interfaces. NVFP4 shard
loading and expert execution live in dedicated distributed files and CUDA units.
Fast mixer code is separate from the IQ exact-reference mixer. Shared scheduler,
launcher and server wiring are integrated by the parent. File ownership is explicit
before parallel edits. A bounded independent review and targeted GPU checks precede
final service measurements; no exhaustive intermediate benchmark campaign.
