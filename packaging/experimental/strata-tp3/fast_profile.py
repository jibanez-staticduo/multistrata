"""Pinned production NVFP4 TP3 profile validation, separate from the IQ oracle."""
from pathlib import Path
import hashlib
import math
import os
import re

SHA = re.compile(r"sha256:[0-9a-f]{64}\Z")
FLAGS = frozenset({'rank', 'rank0-address', 'port', 'session-uuid', 'library-path',
                   'nccl-version', 'library-provenance', 'model-hash', 'config-fingerprint',
                   'pack', 'native', 'embd', 'ple', 'context', 'timeout-ms', 'parallel',
                   'prefill-chunk', 'mtp-path', 'mtp-drafts'})
REQUIRED = FLAGS - {'mtp-path', 'mtp-drafts'}
PERF_ENV = {'STRATA_PREFILL_NVFP4': {'w4a4', 'w4a8', 'fp16'},
            'STRATA_HC_UPMIX': {'0', '1'}, 'STRATA_SPEC_COST': {'0', '1'},
            'STRATA_DEC_BATCH': {'0', '1'}, 'STRATA_QSA_SLOTS': {'0', '1'},
            'STRATA_BATCH_SELF_COMMIT': {'0', '1'}, 'STRATA_GDN_TC': {'0', '1'},
            'STRATA_TP_PHASE_TIMING': {'1'}, 'STRATA_TP_DENSE_MMQ': {'1'},
            'STRATA_TP_HC_BF16': {'1', 'large'}, 'STRATA_TP_MTP_HNORM': {'stream'},
            'STRATA_TP_PACKED_PREFILL': {'1'}, 'STRATA_TP_TRANSPORT': {'roce'},
            'STRATA_TP_ROCE_HCAS': None, 'STRATA_TP_ROCE_ROUTES': None,
            'STRATA_TP_DRAFT_VOCAB': None, 'STRATA_TP_MMVQ_LAYOUT': {'fast'},
            'STRATA_GR_V3': {'1'}, 'STRATA_TP_PREFILL_TILES': {'1'},
            'STRATA_TP_SHARED_SHARD': {'1'}, 'STRATA_TP_DENSE_F16': {'1'},
            'STRATA_TP_ASYNC_DRAFTS': {'1'}, 'STRATA_TP_PLE_IO': {'ram'},
            'STRATA_TP_REDUCE_DTYPE': {'bf16', 'prefill-bf16'},
            'STRATA_TP_NATIVE_GROUPED': {'1'}, 'SPARKNET_ROCE_PROXY_CPU': {'big'},
            'STRATA_TP_CONTROL_COMPACT': {'1'}, 'STRATA_TP_ROCE_BLOCKS': {'8'},
            'STRATA_TP_COMPACT_HEAD': {'1'}, 'STRATA_TP_BALANCED_EXPERTS': {'1'},
            'STRATA_TP_DECODE_FP4': {'1'}, 'STRATA_TP_QSA_PREPARE_BATCH': {'1'},
            'STRATA_TP_QSA_APPEND_BATCH': {'1'}, 'STRATA_TP_DISTRIBUTED_DRAFTS': {'1'},
            'STRATA_TP_DRAFT_ASSETS': None, 'STRATA_TP_HC_PACKED_DOWN': {'1'},
            'STRATA_TP_ROCE_SEGMENT_PREFILL': {'1'}, 'STRATA_TP_GU_WARP32': {'1'},
            'STRATA_TP_PREFILL_QUANT_SCATTER': {'1'}, 'STRATA_TP_PREFILL_PACKED_GU': {'1'},
            'STRATA_TP_DECODE_GU_VECTOR': {'1'}, 'STRATA_TP_GU_PREFETCH': {'1'},
            'STRATA_TP_NVFP4_NEAREST': {'1'}, 'STRATA_TP_QSA_PROMPT_BATCH': {'256'},
            'STRATA_TP_QSA_PROMPT_FAST': {'1'}, 'STRATA_TP_HEAD_F16': {'1'},
            'STRATA_TP_ROUTE_ORDER': {'1'}, 'STRATA_TP_GROUP_REUSE': {'4'},
            'STRATA_TP_ROCE_ADAPTIVE': {'1'},
            'STRATA_TP_B12X_LIBRARY': None, 'STRATA_TP_B12X_SHA256': None,
            'STRATA_TP_CONTROL_RDMA': {'1'}, 'STRATA_TP_RDMA_ONLY': {'1'},
            'STRATA_TP_RDMA_BIND': None, 'STRATA_TP_RDMA_ROOT': None,
            'STRATA_TP_ROCE_RING_ORDER': {'1'},
            'STRATA_TP_ROCE_SLOT_MIB': {'2', '8', '16', '32'}}


def valid_engine_setting(name, value):
    if not isinstance(value, str) or name not in PERF_ENV:
        return False
    if PERF_ENV[name] is not None:
        return value in PERF_ENV[name]
    if name in ('STRATA_TP_DRAFT_ASSETS','STRATA_TP_B12X_SHA256'):
        return SHA.fullmatch(value) is not None
    if name in ('STRATA_TP_DRAFT_VOCAB','STRATA_TP_B12X_LIBRARY'):
        return Path(value).is_absolute() and not any(c in value for c in '\0\r\n')
    if name in ('STRATA_TP_RDMA_BIND','STRATA_TP_RDMA_ROOT'):
        import ipaddress
        try:return isinstance(ipaddress.ip_address(value),ipaddress.IPv4Address)
        except ValueError:return False
    if name == 'STRATA_TP_ROCE_HCAS':
        return bool(re.fullmatch(r'[A-Za-z0-9_]+(?:,[A-Za-z0-9_]+){0,3}', value))
    if not re.fullmatch(r'-?\d+,-?\d+,-?\d+', value):
        return False
    return all(-1 <= int(part) <= 3 for part in value.split(','))


def validated(supplied, common):
    error = common.RunnerError
    config = dict(supplied)
    def require(value, message):
        if not value:
            raise error(message)
    def absolute(value):
        return isinstance(value, str) and Path(value).is_absolute() and not any(c in value for c in '\0\r\n')
    args = config.get('args')
    require(isinstance(args, list) and len(args) % 2 == 0 and
            all(isinstance(x, str) and x and not any(c in x for c in '\0\n\r') for x in args), 'invalid fast profile argv')
    options = {}
    for flag, value in zip(args[::2], args[1::2]):
        require(flag.startswith('--') and flag[2:] in FLAGS and flag[2:] not in options,
                'unknown or duplicate fast profile flag')
        options[flag[2:]] = value
    require(REQUIRED <= options.keys(), 'fast profile omits required flags')
    for name, lo, hi in [('rank', 0, 2), ('port', 1024, 65535), ('context', 4, 262144),
                         ('parallel', 1, 8), ('prefill-chunk', 64, 8192), ('timeout-ms', 100, 600000)]:
        require(options[name].isdigit() and lo <= int(options[name]) <= hi, 'invalid fast profile ' + name)
    require(options['parallel'] in ('1', '4', '8'), 'fast parallel must be 1, 4 or 8')
    require(config.get('mode', 'multislot') == 'multislot', 'fast profile requires multislot protocol')
    require(config.get('parallel', int(options['parallel'])) == int(options['parallel']), 'fast parallel settings disagree')
    for name in ['pack', 'native', 'embd', 'ple', 'library-path']:
        require(absolute(options[name]), 'fast asset paths must be absolute')
    require(all(SHA.fullmatch(options[k]) for k in ['library-provenance', 'model-hash', 'config-fingerprint']),
            'fast identities must be SHA256 fingerprints')
    require(re.fullmatch(r'[0-9a-fA-F-]{36}', options['session-uuid']) is not None, 'invalid fast session UUID')
    require(re.fullmatch(r'[0-9.]+', options['nccl-version']) is not None, 'invalid NCCL version')
    require(bool(options['rank0-address']) and not options['rank0-address'].startswith('-'), 'invalid rank zero address')
    mtp = 'mtp-path' in options or 'mtp-drafts' in options
    require(not mtp or ('mtp-path' in options and 'mtp-drafts' in options and absolute(options['mtp-path']) and
                        options['mtp-drafts'] in ('1', '2', '3', '4')), 'incomplete fast MTP configuration')
    for name in ['server_source_root', 'tokenizer_dir', 'evidence_dir', 'binary', 'source_file', 'credential_file']:
        require(absolute(config.get(name)), 'fast runtime paths must be absolute')
    common.credential(config['credential_file'])
    for path, digest in [('binary', 'binary_sha256'), ('source_file', 'source_sha256')]:
        require(SHA.fullmatch(str(config.get(digest, ''))) and common.digest(config[path]) == config[digest],
                'fast binary/source fingerprint disagreement')
    require(os.access(config['binary'], os.X_OK), 'fast binary is not executable')
    require(common.digest(options['library-path']) == options['library-provenance'], 'fast NCCL fingerprint disagreement')
    require(re.fullmatch('[0-9a-f]{40}', str(config.get('source_commit', ''))) is not None, 'invalid fast source commit')
    network = config.get('env', {})
    channels = {'NCCL_MIN_NCHANNELS', 'NCCL_MAX_NCHANNELS'}
    require(isinstance(network, dict) and set(network) <= common.NETWORK_ENV | channels and
            all(isinstance(v, str) and not any(c in v for c in '\0\n\r') for v in network.values()), 'invalid fast NCCL environment')
    require(all(network[name].isdigit() and 1 <= int(network[name]) <= 32
                for name in channels & network.keys()), 'invalid fast NCCL channel count')
    require(int(network.get('NCCL_MIN_NCHANNELS', '1')) <= int(network.get('NCCL_MAX_NCHANNELS', '32')),
            'fast NCCL channel bounds disagree')
    engine = config.get('engine_env', {})
    require(isinstance(engine, dict) and all(valid_engine_setting(k,v) for k,v in engine.items()),
            'invalid fast engine environment')
    require('STRATA_TP_CONTROL_RDMA' not in engine or engine.get('STRATA_TP_TRANSPORT') == 'roce',
            'RDMA control requires the RoCE tensor transport')
    require('STRATA_TP_ROCE_SLOT_MIB' not in engine or engine.get('STRATA_TP_TRANSPORT') == 'roce',
            'RoCE slot size requires the RoCE tensor transport')
    if 'STRATA_TP_RDMA_ONLY' in engine:
        require(engine.get('STRATA_TP_TRANSPORT') == 'roce' and
                {'STRATA_TP_RDMA_BIND','STRATA_TP_RDMA_ROOT'} <= engine.keys(),
                'RDMA-only execution requires explicit RDMA endpoints and RoCE')
    require('STRATA_TP_ROCE_RING_ORDER' not in engine or engine.get('STRATA_TP_RDMA_ONLY') == '1',
            'ring-ordered RoCE sums require RDMA-only execution')
    b12x = engine.get('STRATA_TP_B12X_LIBRARY')
    require(bool(b12x) == bool(engine.get('STRATA_TP_B12X_SHA256')),
            'B12X requires both library path and fingerprint')
    if b12x:
        require(Path(b12x).is_file() and common.digest(b12x) == engine['STRATA_TP_B12X_SHA256'],
                'B12X library fingerprint disagreement')
        require(not ({'STRATA_TP_NATIVE_GROUPED','STRATA_TP_PREFILL_PACKED_GU',
                      'STRATA_TP_GROUP_REUSE','STRATA_TP_DECODE_FP4'} & engine.keys()),
                'B12X requires original separate NVFP4 planes and its own decode route')
    require('STRATA_TP_NVFP4_NEAREST' not in engine or engine.get('STRATA_PREFILL_NVFP4') == 'w4a4',
            'nearest FP4 scales require W4A4 activations')
    require(not ('STRATA_TP_PREFILL_PACKED_GU' in engine and 'STRATA_TP_NATIVE_GROUPED' in engine),
            'packed prefill gate/up cannot use grouped expert storage')
    require('STRATA_TP_PREFILL_PACKED_GU' not in engine or engine.get('STRATA_PREFILL_NVFP4') == 'w4a4',
            'packed prefill gate/up requires the qualified W4A4 profile')
    require('STRATA_TP_DECODE_GU_VECTOR' not in engine or
            not ({'STRATA_TP_DECODE_FP4','STRATA_TP_NATIVE_GROUPED'} & engine.keys()),
            'vector GU requires the native Q8 decode path')
    require('STRATA_TP_GU_PREFETCH' not in engine or
            not ({'STRATA_TP_DECODE_FP4','STRATA_TP_NATIVE_GROUPED','STRATA_TP_DECODE_GU_VECTOR',
                  'STRATA_TP_GU_WARP32'} & engine.keys()), 'GU prefetch requires the native warp16 decode path')
    require('STRATA_TP_ROUTE_ORDER' not in engine or
            not ({'STRATA_TP_DECODE_FP4','STRATA_TP_NATIVE_GROUPED','STRATA_TP_DECODE_GU_VECTOR',
                  'STRATA_TP_GU_WARP32','STRATA_TP_GU_PREFETCH'} & engine.keys()),
            'route order requires the native warp16 decode path')
    require('STRATA_TP_GROUP_REUSE' not in engine or
            not ({'STRATA_TP_DECODE_FP4','STRATA_TP_NATIVE_GROUPED','STRATA_TP_DECODE_GU_VECTOR',
                  'STRATA_TP_GU_WARP32','STRATA_TP_GU_PREFETCH'} & engine.keys()),
            'group reuse requires the native Q8 decode path')
    # Files are pinned relative to the supplied model root. Source/header and
    # tokenizer manifests are checked by the service runner, using its own rules.
    root = Path(options['pack']).parent
    for key, name in [('pack', 'pack'), ('native', 'qwen-nvfp4-gptq-dense.gguf'),
                      ('embd', 'token-embd-bf16.gguf'), ('ple', 'ple-fp8.gguf')]:
        require(Path(options[key]).resolve() == (root / name).resolve(), 'fast asset path differs from pinned layout')
    manifest = config.get('asset_manifest')
    required = {'pack/experts.bin', 'pack/native_experts.txt', 'pack/dense.bin', 'pack/index.txt',
                'qwen-nvfp4-gptq-dense.gguf', 'token-embd-bf16.gguf', 'ple-fp8.gguf'}
    if mtp:
        require(Path(options['mtp-path']).resolve() == (root / 'mtp').resolve(), 'fast MTP path differs from pinned layout')
        required |= {'mtp/dense.bin', 'mtp/dense.txt', 'mtp/experts.bin'}
    if 'STRATA_TP_DRAFT_VOCAB' in engine:
        vocab = Path(engine['STRATA_TP_DRAFT_VOCAB']).resolve()
        require(vocab.is_relative_to(root.resolve()), 'draft vocabulary must be in the pinned model root')
        required.add(str(vocab.relative_to(root.resolve())))
    require(isinstance(manifest, dict) and required <= manifest.keys(), 'fast asset manifest is incomplete')
    for name, expected in manifest.items():
        path = Path(name)
        require(isinstance(name, str) and not path.is_absolute() and '..' not in path.parts and SHA.fullmatch(str(expected)),
                'invalid fast asset manifest entry')
        target = (root / path).resolve()
        require(target.is_relative_to(root.resolve()) and target.is_file() and common.digest(target) == expected,
                'fast model asset fingerprint disagreement')
    if 'STRATA_TP_DISTRIBUTED_DRAFTS' in engine:
        require(mtp and {'mtp/dense.txt', 'mtp/dense.bin', 'mtp/experts.bin', 'mtp/draft_vocab.bin'} <= manifest.keys(),
                'distributed drafts require pinned MTP assets')
        draft_manifest = ''.join(name + ':' + manifest[name] + '\n' for name in sorted(manifest) if name.startswith('mtp/'))
        expected = 'sha256:' + hashlib.sha256(draft_manifest.encode('ascii')).hexdigest()
        require(engine.get('STRATA_TP_DRAFT_ASSETS') == expected, 'distributed draft asset identity disagreement')
    for name, default, lo, hi in [('ready_timeout', 600, 300, 3600), ('request_timeout', 900, 1, 86400),
                                  ('cancel_timeout', 60, 1, 120), ('shutdown_timeout', 10, 1, 30),
                                  ('terminate_timeout_s', 10, 0.1, 30), ('process_timeout_s', 0, 0, 86400)]:
        value = config.get(name, default)
        require(type(value) in (int, float) and math.isfinite(value) and lo <= value <= hi,
                'invalid fast duration ' + name)
        config[name] = value
    config.update(options=options, mode='multislot', parallel=int(options['parallel']))
    return config
