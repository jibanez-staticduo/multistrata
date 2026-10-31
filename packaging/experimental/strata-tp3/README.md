# Experimental native Strata TP3 runtime

This package uses the public SparkRun 0.4.1 plugin API. It launches three
Strata ranks through SparkRun's native container lifecycle. The pinned image
provides CUDA and Python; no vLLM process is launched. The transport smoke,
teacher-forced model and autoregressive service have separate validation gates.
No performance improvement has been established.

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
