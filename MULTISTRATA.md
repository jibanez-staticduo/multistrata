# MultiStrata: Native NVFP4 TP3 Over RoCE

Experimental fork of [Niko1221/Strata](https://github.com/Niko1221/Strata).
The aim is to run one model with tensor parallelism across three DGX Sparks,
and approach Staticduo21/vLLM throughput while keeping Strata's native engine.
Performance parity has **not** been reached. Development is paused as of
2026-11-01; the original Staticduo21 service is being restored at the owner's
request. Existing Strata code and experimental paths are preserved here.

## Implemented

- Native NVFP4 TP3/PP1 across three NVIDIA GB10 hosts and a 200 Gb/s RoCE triangle.
- GPU expert, mixer, attention, prefill and speculative verification paths;
  CUDA Graphs and batching for up to eight concurrent requests.
- A strict RDMA profile for bootstrap, authentication, control, errors and
  tensors. It does not initialize NCCL or use TCP between Strata ranks.
- RDMA-CM SEND/RECV bootstrap, authenticated session agreement, native RoCE
  collectives, and fused FP32 reductions with the measured SIMPLE sum order.
- An OpenAI-compatible service, experimental SparkRun lifecycle and monitor.

These are opt-in experimental paths, not general multi-host support in the
upstream release. Weights and machine credentials are not included.

## Measured Status

The last qualified binary was
`1963e3b593b55ff3320558761479e2a25c4d9c77ef0042f62f3acd8548a8a967`.
Hardware: three DGX Sparks, one GB10 per host, direct RoCE links; native
Qwen3.8-Flash-Next NVFP4, context 262144, eight slots, two MTP drafts.

| Check | Observed result |
| --- | --- |
| Full-logit comparison against selected same-pack reference | 140 positions, mean/max KL 0, all top-1 choices agree |
| Socket audit during startup, generation and shutdown | 111 trace files, zero Internet stream socket attempts |
| Actual long-context retrieval | 262128 input tokens, 16 requested output; correct identifier returned |
| Long-context wall time / prefill | 170.460 s / 1544.9 tokens/s |
| C1 native prefill / decode, PP2048/TG128 | 1682.6 / 62.3 tokens/s |
| C1/C4/C8 aggregate output, including prefill and queue time | 38.07 / 57.95 / 67.24 tokens/s |

The historical Staticduo21 comparison is 43.07 / 84.55 / 105.95 aggregate
tokens/s. This is a directional comparison: identical HTTP prompts and reported
token counts do not prove identical input IDs, KV formats or MTP settings.
The Staticduo21 baseline has one round per cell; small differences are not
confidence-qualified. These aggregate rates are not pure decoder speed.

## Work At The Pause

The latest source includes configurable RoCE slots of 2/8/16/32 MiB. The
qualified binary used 2 MiB. A three-host component test found exact reductions
at each size; 8 MiB reduced the measured roughly-20-MiB collective from
8.39 to 4.32 ms. Full-model 8 MiB qualification was interrupted for the pause,
so that option is **not qualified** for serving or model throughput.

B12X A4/A8/A16 experiments remain optional. Component speedups did not pass
the frozen full-model maximum-KL gate; they are not the selected arithmetic.
The frozen policy must not be widened to turn a failed candidate into a pass.

Useful contributions include faster NVFP4 expert kernels, large-message RoCE
collectives, prefill scheduling, and removing verified GPU/host overhead while
preserving request state, quantization and numerical quality. CUDA/GB10 was
tested; HIP, SYCL and other platforms are not qualified for these new paths.

## Code And Plans

- [Design](docs/superpowers/specs/2026-10-31-strata-nvfp4-tp3-design.md)
- [Implementation plan and evidence ledger](docs/superpowers/plans/2026-10-31-strata-nvfp4-tp3-final.md)
- [Experimental launcher](packaging/experimental/strata-tp3/README.md)
- Runtime: `src/distributed/fast_*.cpp`, `control_rdma.cpp`, `roce_backend.cpp`
- GPU kernels: `src/kernels/cuda/tp/`
- Tests: `tests/distributed/`

Some historical plans contain machine-specific paths and earlier lifecycle
decisions. They are experiment records, not portable installation instructions.
The latest operational decision is to pause Strata and restore Staticduo21.

Before proposing a change, inspect the current source and distinguish component
results from full-model qualification. Build and test commands and exact
experimental switches are documented in the launcher and implementation plan.
