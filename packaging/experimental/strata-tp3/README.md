# Experimental native Strata TP3 runtime

This package uses the public SparkRun 0.4.1 plugin API. It launches three
Strata ranks through SparkRun's native container lifecycle. The pinned image
provides CUDA and Python; no vLLM process is launched. The transport smoke,
teacher-forced model and autoregressive service have separate validation gates.
The acceptance record distinguishes the slow reference measurements from the
ongoing GPU acceleration candidate; it establishes no advantage over TP1 or vLLM.

## Isolated setup on DG1

Run from this directory. These commands create a separate config and Python
environment. They do not modify the installed SparkRun uv tool. The config copy
may contain private settings; do not print it or add it to git.

```sh
experiment_dir=/home/staticduo/tmp/strata-tp3-20261031
cp -a "$HOME/.config/sparkrun" "$experiment_dir/sparkrun-config"
host_python=/home/staticduo/.local/share/uv/tools/sparkrun/bin/python
"$host_python" -m venv "$experiment_dir/sparkrun-venv"
host_site=$("$host_python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
"$experiment_dir/sparkrun-venv/bin/python" - "$host_site" <<'PY'
from pathlib import Path
import site
import sys
(Path(site.getsitepackages()[0]) / 'sparkrun-host.pth').write_text(sys.argv[1] + '\n')
PY
uv pip install --python "$experiment_dir/sparkrun-venv/bin/python" --no-deps -e .
"$experiment_dir/sparkrun-venv/bin/python" - "$experiment_dir" <<'PY'
from pathlib import Path
import sys
import yaml
path = Path(sys.argv[1]) / 'sparkrun-config/config.yaml'
data = yaml.safe_load(path.read_text()) or {}
data.setdefault('integrations', {})['strata-native'] = True
path.write_text(yaml.safe_dump(data, sort_keys=False))
path.chmod(0o600)
PY
```

Do not overwrite an existing isolated config. On the prepared DG1 experiment,
the package is in `$experiment_dir/sparkrun-plugin` and setup is already done.
Use `sparkrun-strata` for validate, run, status, logs and stop so discovery and
the configuration stay consistent.

## Stage and check

Each host needs the existing image, `/experiment/run_transport_smoke.py`, the
compiled smoke binary selected by its config, and the read-only NCCL library
mount in `smoke.yaml`. Stage the experiment at the same host path on all nodes.
The recipe disables model downloads and image distribution.
The recipe pins `/usr/bin/python3`, confirmed in the experiment image. A dry
run does not check executables inside that image. If changing the image, check
the interpreter before launch and override `-o python=/absolute/interpreter`
when needed. The runtime's fallback for recipes without that setting is `python3`.

Create `smoke-config/rank0.json`, `rank1.json`, and `rank2.json` with matching
session UUID, model/config/build/library identities, `world: 3`, and rank 0, 1
or 2 respectively. Use `rank0_address: 10.71.14.15` and `port: 30123`. Point
`library_path` to `/experiment-nccl/libnccl.so.2`. Point `credential_file` to
the staged credential file inside `/experiment`; it must be owned by UID 1000
and mode 0600. Do not place the credential in the recipe, argv or env. The
wrapper validates these constraints. Use fresh session/evidence directories
for each run; all three receipts must be checked after completion. The staged
binary in this experiment is `/experiment/strata-tp3-transport-smoke`.
Override `-o evidence_dir=/experiment/smoke-evidence/CASE_ID` on every run and
use a fresh `CASE_ID`; the wrapper refuses to overwrite existing evidence.
For fault cases, also select a fresh staged config directory with
`-o config_dir=/experiment/smoke-config-CASE_ID`.

The authenticated bootstrap retries connections. `native_rendezvous_port`
returns `None` so SparkRun does not make an unauthenticated probe against it.
The smoke has no HTTP readiness endpoint; always use `--no-ready-wait`.

```sh
./sparkrun-strata recipe validate ./smoke.yaml
./sparkrun-strata -v run ./smoke.yaml --cluster cx7 --no-auto-detect \
  --no-sync-tuning --no-ready-wait --no-follow --trust --dry-run
```

After the coordinator has arranged the GPU window and the restoration plan:

```sh
./sparkrun-strata run ./smoke.yaml --cluster cx7 --no-auto-detect \
  --no-sync-tuning --no-ready-wait --no-follow --trust \
  -o evidence_dir=/experiment/smoke-evidence/CASE_ID
./sparkrun-strata logs CLUSTER_ID
./sparkrun-strata stop CLUSTER_ID
```

Use the returned cluster ID. SparkRun owns the ranked containers, logs and
cleanup. This plugin does not stop other jobs or restore the prior workload;
the coordinator must restore and verify that workload after the experiment.
`non-portable-mount` is expected for this site-specific recipe.

## Tests and future mode

In an environment with this package installed and SparkRun 0.4.1:

```sh
python -m pytest tests/distributed/test_sparkrun_runtime.py -q
```

The tests use real plugin discovery and check all rank argv, quoting, topology,
the absence of the bootstrap probe, and delegation to native start/stop.
No GPU or Docker execution is needed.

`mode: distributed` launches a separately built experimental engine. It
requires an explicit absolute `distributed_executable` and a
`distributed_args` argv list. Only `{rank}` and `{world_size}` placeholders are
supported.

## Experimental C1 service

`service.yaml` runs `run_service_rank.py` on each host. Rank zero starts the
existing Python API with `--engine distributed`; ranks one and two run the
native `strata-distributed --mode autoregressive` worker. The endpoint binds
to `127.0.0.1:18080`. The normal launch path is unchanged.

Each rank needs a JSON config with binary, driver source, NCCL and tokenizer
hashes, the staged Python source manifest and a fresh evidence directory.
The wrapper reads its credential from a user-owned mode-0600 file, validates
the artifacts and starts only with `--execute`. It bounds process lifetime and
log size and stops the complete child process group on exit. See its module
docstring for the config fields. The image must provide `regex` and `jinja2`.

The native coordinator owns sampling and sends only tokens whose source
logits have committed on all three ranks. Cancellation finishes the current
step and resets the group before accepting another request. Periodic health
requires a round trip through every worker. A worker failure ends the group.

CPU protocol and API tests use simulated model execution. The short hardware
trial on 2026-10-31 compared four teacher-forced tokens, repeated after RESET,
against an independent TP1 implementation: logits and captured logical state
were bit-identical on DG1/DG2/DG3 with the ISTA IQ3_S pack. That trial does not
qualify autoregressive HTTP serving, long prompts, 64K, MTP, graphs or C8.

`tools/distributed/benchmark_tp.py` prepares a matched TP1/TP2/TP3 matrix.
It defaults to a dry run and records unsupported cases as `NOT_RUN`.
The TP1 numerical oracle's default stages experts and is not a matched speed
baseline. `reference-service.yaml` instead uses the separate `strata-reference`
runtime and `run_reference_rank.py`, with all experts resident, world one and
one active request on `127.0.0.1:18081`. Its config pins every model file as
well as the native binary, server sources and tokenizer. The optional boolean
`release_verified_cache` advises away clean model file-cache pages after those
hash checks. It defaults to false; dry runs never change cache residency.
This helps the Spark's CUDA memory budget reflect available unified memory
after hashing. It does not bypass the budget or change model bytes.

The model and service runners accept explicit `--expert-operands full|compact`
and `--collective-agreement per-op|step`; defaults remain `full` and `per-op`.
`--profile-dir` requires an absolute path separate from trace evidence.
Only the service runner accepts `--service-state-dir`: an absolute, fresh
directory separate from logs and profiles. The driver creates that directory
when it saves canonical consumed IDs and internal state before reset/cancel.
Capturing these files is diagnostic work, not a performance benchmark.
See [the current acceptance record](../../../docs/TP3_EXPERIMENTAL_STATUS.md)
for the exact hardware scope qualified so far.

## Accelerated C1 Candidate

The experimental candidate combines these explicit native flags:

```text
--expert-collectives batched --expert-operands compact
--collective-agreement step --graph-finish 0
--device-expert-routing 1 --prefill-batch-size 8 --prefill-expert-batch 1
--model-graph 1 --model-graph-mode decode
```

Use `mode: autoregressive`, C1 and no MTP. Prompt chunks run eagerly; decode
uses a full-model CUDA Graph. Reset and cancellation retire that graph before
the next request captures a new one. The existing graph mode `all` remains the
default when `--model-graph-mode` is omitted, and graphs remain off unless
`--model-graph 1` is explicit. The runners reject unsupported combinations.

Device routing uses the fused native gate/up, SwiGLU and padded-output kernel.
It keeps original weight formats and Q8_1 activation arithmetic. This is an
integration candidate: consult the acceptance record before treating a flag
combination as qualified or faster. Keep state capture separate from speed
measurements because writing diagnostics adds work.

## NVFP4 Production Profile

The final NVFP4 implementation is tracked by
[`2026-10-31-strata-nvfp4-tp3-final.md`](../../../docs/superpowers/plans/2026-10-31-strata-nvfp4-tp3-final.md).
It has a separate runner profile, `"profile": "nvfp4_tp3"`, because its
arithmetic, model assets, 262144 protocol capacity and execution flags differ
from the IQ exact-reference path. It requires the original NVFP4 expert pack,
dense GGUF, BF16 embedding and FP8 PLE, all pinned in `asset_manifest`.
The runner checks complete source and tokenizer manifests too.

`--prefill-chunk` selects the large prompt chunk (64..8192), and `--parallel`
selects 1, 4 or 8 active slots. A `process_timeout_s` of 0 keeps this opt-in
service running until SparkRun stops it; individual requests still have bounded
deadlines. `engine_env` accepts only the named performance settings validated by
`fast_profile.py`. This profile is under implementation: protocol capacity and
flags do not establish GPU fit, model correctness or performance acceptance.

The NVFP4 executor can select `STRATA_TP_DENSE_MMQ=1` in `engine_env` for
Q8 mixer projections with at least four rows, 2048 outputs and a reduction
width divisible by 256. Smaller projections retain native MMVQ. Weights stay
in their original Q8 format; the tensor-core reduction can change rounding.
`STRATA_TP_PHASE_TIMING=1` writes host phase durations to native stderr for
diagnostics. The managed service enables `inherit_stderr` so its existing
bounded, redacted private log also captures the root engine's diagnostics.
Other backend launchers discard stderr unless they explicitly enable this option.
The service health lease covers the native operation budget plus five seconds,
with a thirty-second minimum; a long prefill step does not expire an earlier lease.

`STRATA_TP_ROCE_BLOCKS=8` selects eight cooperative CUDA blocks with 512
threads for the direct RoCE copy and reduction. The device must support a
cooperative launch with all eight blocks resident. Grid barriers protect the
doorbell and slot reuse; the single-block default keeps its existing kernel.
Use a qualified profile: transport smoke and graph replay alone do not
establish model throughput or maximum-context correctness.

The CUDA distributed build can opt into `-DSTRATA_TP_WIDE_WINDOW=ON` for
40 logical target-verification rows. This build is for `strata-distributed-fast`;
native matvec kernels still process at most 16 rows per launch. Eight slots
can then verify up to four drafts each. A larger window is not necessarily
faster; select the draft budget using whole-model measurements.

`STRATA_TP_COMPACT_HEAD=1` exchanges only candidate maxima for plain greedy
requests. Tie-breaking keeps the lowest global token ID, and every local logit
is checked for nonfinite values. Sampling, history penalties and mixed batches
use full logits. `STRATA_TP_LOGITS_DIR` also forces full logits for diagnostics.
Graph caches distinguish these modes. The HTTP API's existing token-bias
restriction remains in effect.

`STRATA_TP_HC_BF16=1` selects a single BF16 activation image for decode HC
projections with FP32 accumulation and original BF16 weights. It changes rounding;
whole-model numerical and performance qualification is separate from component
checks. `STRATA_SPEC_COST=1` enables adaptive greedy draft-window costs; without
it the executor uses its configured fixed draft budget, bounded by active rows.

`STRATA_TP_MTP_HNORM=stream` selects the existing GB10 draft hidden-norm option.
It normalizes each HC stream separately, while the default pooled norm follows
vLLM's reference equation. This affects proposals only; target verification and
request sampling remain authoritative. Measure acceptance and throughput before
selecting it for a serving profile.

## Strict RDMA Between Hosts

`STRATA_TP_RDMA_ONLY=1` requires the RoCE transport. It uses authenticated
RDMA-CM SEND/RECV for bootstrap and errors, and native RoCE for control and
tensors. It never initializes an NCCL communicator. IPv4 endpoint addresses
select RDMA routes; they do not imply TCP sockets. The local HTTP API remains
available for clients. Set `STRATA_TP_CONTROL_RDMA=1` and rank-specific
`STRATA_TP_RDMA_BIND` / `STRATA_TP_RDMA_ROOT` addresses in the checked profile.

`STRATA_TP_ROCE_RING_ORDER=1` preserves the selected NCCL SIMPLE eight-channel
FP32 sum association for large messages, including16KiB cell boundaries and
the shorter final channel. This is a compatibility profile for the measured
three-Spark setup, not a general NCCL algorithm emulator. Requalify numerical
results when changing the model or transport configuration.
