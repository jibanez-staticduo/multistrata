"""Production profile uses its own asset and arithmetic contract, not IQ bitexact."""
import importlib.util
from pathlib import Path
import pytest

SPEC = importlib.util.spec_from_file_location('service', Path(__file__).parents[2] /
    'packaging/experimental/strata-tp3/run_service_rank.py')
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def config(tmp_path):
    root = tmp_path / 'model'
    names = ['pack/experts.bin', 'pack/native_experts.txt', 'pack/dense.bin', 'pack/index.txt',
             'qwen-nvfp4-gptq-dense.gguf', 'token-embd-bf16.gguf', 'ple-fp8.gguf']
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
    source = tmp_path / 'source'
    sources = ['tools/strata_tokenizer.py', 'serve/server.py', 'serve/distributed_backend.py', 'serve/chat_template.jinja']
    for name in sources:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
    tokenizer = root / 'pack/tokenizer'
    tokenizer.mkdir()
    for name in ['vocab.json', 'merges.txt', 'token_type.json', 'tokenizer.json']:
        (tokenizer / name).write_text('{}')
    binary = tmp_path / 'engine'
    binary.write_text('#!/bin/sh\nexit 0\n'); binary.chmod(0o755)
    credential = tmp_path / 'credential'
    credential.write_text('private-test-credential'); credential.chmod(0o600)
    library = tmp_path / 'libnccl.so'
    library.write_text('pinned library')
    options = {'rank': '0', 'rank0-address': '10.71.14.15', 'port': '30420',
               'session-uuid': '11111111-2222-3333-4444-555555555555', 'library-path': str(library),
               'nccl-version': '2.30.7', 'library-provenance': runner.digest(library),
               'model-hash': 'sha256:' + 'a' * 64, 'config-fingerprint': 'sha256:' + 'b' * 64,
               'pack': str(root / 'pack'), 'native': str(root / 'qwen-nvfp4-gptq-dense.gguf'),
               'embd': str(root / 'token-embd-bf16.gguf'), 'ple': str(root / 'ple-fp8.gguf'),
               'context': '262144', 'timeout-ms': '120000', 'parallel': '8', 'prefill-chunk': '2048'}
    return {'profile': 'nvfp4_tp3', 'binary': str(binary), 'binary_sha256': runner.digest(binary),
            'source_file': str(source / 'serve/server.py'), 'source_sha256': runner.digest(source / 'serve/server.py'),
            'credential_file': str(credential), 'source_commit': 'c' * 40,
            'args': [x for pair in options.items() for x in ('--' + pair[0], pair[1])],
            'server_source_root': str(source), 'source_manifest': {n: runner.digest(source / n) for n in sources},
            'tokenizer_dir': str(tokenizer), 'tokenizer_manifest': {p.name: runner.digest(p) for p in tokenizer.iterdir()},
            'asset_manifest': {n: runner.digest(root / n) for n in names},
            'evidence_dir': str(tmp_path / 'evidence'), 'mode': 'multislot', 'parallel': 8,
            'process_timeout_s': 0, 'engine_env': {'STRATA_PREFILL_NVFP4': 'w4a4'}}


def test_final_profile_256k_and_persistent_service(tmp_path):
    c = runner.validated(config(tmp_path))
    assert c['options']['context'] == '262144' and c['parallel'] == 8
    assert c['process_timeout_s'] == 0
    assert runner.server_config(c)['distributed']['parallel'] == 8


@pytest.mark.parametrize('name', ['STRATA_TP_PREFILL_TILES', 'STRATA_TP_SHARED_SHARD', 'STRATA_TP_DENSE_F16',
                                 'STRATA_TP_DECODE_FP4', 'STRATA_TP_QSA_PREPARE_BATCH',
                                 'STRATA_TP_QSA_APPEND_BATCH', 'STRATA_TP_HC_PACKED_DOWN',
                                 'STRATA_TP_ROCE_SEGMENT_PREFILL', 'STRATA_TP_GU_WARP32',
                                 'STRATA_TP_PREFILL_QUANT_SCATTER', 'STRATA_TP_PREFILL_PACKED_GU',
                                 'STRATA_TP_DECODE_GU_VECTOR', 'STRATA_TP_GU_PREFETCH',
                                 'STRATA_TP_NVFP4_NEAREST','STRATA_TP_QSA_PROMPT_FAST','STRATA_TP_HEAD_F16',
                                 'STRATA_TP_ROUTE_ORDER','STRATA_TP_ROCE_ADAPTIVE'])
def test_explicit_native_execution_options(tmp_path, name):
    c=config(tmp_path)
    c['engine_env'][name]='1'
    assert runner.validated(c)['engine_env'][name]=='1'
    c['engine_env'][name]='unrecognized'
    with pytest.raises(runner.RunnerError):
        runner.validated(c)


def test_distributed_drafts_pin_the_same_assets(tmp_path):
    import hashlib
    c=config(tmp_path)
    root=Path(c['args'][c['args'].index('--pack')+1]).parent
    names=['mtp/dense.txt','mtp/dense.bin','mtp/experts.bin','mtp/draft_vocab.bin']
    for name in names:
        p=root/name;p.parent.mkdir(exist_ok=True);p.write_text(name)
        c['asset_manifest'][name]=runner.digest(p)
    c['args']+=['--mtp-path',str(root/'mtp'),'--mtp-drafts','2']
    text=''.join(name+':'+c['asset_manifest'][name]+'\n' for name in sorted(names))
    c['engine_env'].update(STRATA_TP_DISTRIBUTED_DRAFTS='1',
        STRATA_TP_DRAFT_ASSETS='sha256:'+hashlib.sha256(text.encode('ascii')).hexdigest())
    assert runner.validated(c)['engine_env']['STRATA_TP_DISTRIBUTED_DRAFTS']=='1'
    c['engine_env']['STRATA_TP_DRAFT_ASSETS']='sha256:'+'0'*64
    with pytest.raises(runner.RunnerError):runner.validated(c)


def test_b12x_library_is_pinned(tmp_path):
    c=config(tmp_path)
    library=tmp_path/'libb12x.so'
    library.write_bytes(b'qualified AOT artifact')
    c['engine_env']['STRATA_TP_B12X_LIBRARY']=str(library)
    with pytest.raises(runner.RunnerError):runner.validated(c)
    c['engine_env']['STRATA_TP_B12X_SHA256']=runner.digest(library)
    assert runner.validated(c)['engine_env']['STRATA_TP_B12X_LIBRARY']==str(library)
    library.write_bytes(b'changed artifact')
    with pytest.raises(runner.RunnerError):runner.validated(c)


def test_b12x_requires_compatible_expert_storage(tmp_path):
    c=config(tmp_path)
    library=tmp_path/'libb12x.so';library.write_bytes(b'AOT')
    c['engine_env'].update(STRATA_TP_B12X_LIBRARY=str(library),
                           STRATA_TP_B12X_SHA256=runner.digest(library),
                           STRATA_TP_NATIVE_GROUPED='1')
    with pytest.raises(runner.RunnerError):runner.validated(c)


def test_rdma_control_requires_roce(tmp_path):
    c=config(tmp_path);c['engine_env']['STRATA_TP_CONTROL_RDMA']='1'
    with pytest.raises(runner.RunnerError):runner.validated(c)
    c['engine_env']['STRATA_TP_TRANSPORT']='roce'
    assert runner.validated(c)['engine_env']['STRATA_TP_CONTROL_RDMA']=='1'

@pytest.mark.parametrize('mib', ['2', '8', '16', '32'])
def test_roce_slot_size_requires_roce_transport(tmp_path, mib):
    c=config(tmp_path)
    c['engine_env'].update(STRATA_TP_TRANSPORT='roce', STRATA_TP_ROCE_SLOT_MIB=mib)
    assert runner.validated(c)['engine_env']['STRATA_TP_ROCE_SLOT_MIB']==mib

@pytest.mark.parametrize('mib', ['', '0', '1', '4', '64', '2MiB', '08'])
def test_roce_slot_size_rejects_invalid_values(tmp_path, mib):
    c=config(tmp_path)
    c['engine_env'].update(STRATA_TP_TRANSPORT='roce', STRATA_TP_ROCE_SLOT_MIB=mib)
    with pytest.raises(runner.RunnerError):runner.validated(c)

def test_roce_slot_size_requires_roce(tmp_path):
    c=config(tmp_path)
    c['engine_env']['STRATA_TP_ROCE_SLOT_MIB']='8'
    with pytest.raises(runner.RunnerError):runner.validated(c)


def test_rdma_only_requires_endpoints(tmp_path):
    c=config(tmp_path);c['engine_env'].update(STRATA_TP_RDMA_ONLY='1',STRATA_TP_TRANSPORT='roce')
    with pytest.raises(runner.RunnerError):runner.validated(c)
    c['engine_env'].update(STRATA_TP_RDMA_BIND='0.0.0.0',STRATA_TP_RDMA_ROOT='192.168.201.15')
    assert runner.validated(c)['engine_env']['STRATA_TP_RDMA_ONLY']=='1'
    c['engine_env']['STRATA_TP_RDMA_ROOT']='192.168.201.999'
    with pytest.raises(runner.RunnerError):runner.validated(c)


def test_ring_order_requires_rdma_only(tmp_path):
    c=config(tmp_path);c['engine_env']['STRATA_TP_ROCE_RING_ORDER']='1'
    with pytest.raises(runner.RunnerError):runner.validated(c)
    c['engine_env'].update(STRATA_TP_RDMA_ONLY='1',STRATA_TP_TRANSPORT='roce',
        STRATA_TP_RDMA_BIND='0.0.0.0',STRATA_TP_RDMA_ROOT='192.168.178.15')
    assert runner.validated(c)['engine_env']['STRATA_TP_ROCE_RING_ORDER']=='1'


def test_packed_gate_up_rejects_grouped_storage(tmp_path):
    c=config(tmp_path)
    c['engine_env'].update(STRATA_TP_PREFILL_PACKED_GU='1',STRATA_TP_NATIVE_GROUPED='1')
    with pytest.raises(runner.RunnerError):runner.validated(c)


def test_packed_gate_up_requires_w4a4(tmp_path):
    c=config(tmp_path)
    c['engine_env'].update(STRATA_TP_PREFILL_PACKED_GU='1',STRATA_PREFILL_NVFP4='w4a8')
    with pytest.raises(runner.RunnerError):runner.validated(c)


@pytest.mark.parametrize('other',['STRATA_TP_DECODE_FP4','STRATA_TP_NATIVE_GROUPED'])
def test_vector_gu_rejects_inactive_decode_routes(tmp_path,other):
    c=config(tmp_path)
    c['engine_env'].update(STRATA_TP_DECODE_GU_VECTOR='1')
    c['engine_env'][other]='1'
    with pytest.raises(runner.RunnerError):runner.validated(c)


@pytest.mark.parametrize('other',['STRATA_TP_DECODE_FP4','STRATA_TP_NATIVE_GROUPED',
                                'STRATA_TP_DECODE_GU_VECTOR','STRATA_TP_GU_WARP32'])
def test_gu_prefetch_rejects_inactive_decode_routes(tmp_path,other):
    c=config(tmp_path)
    c['engine_env'].update(STRATA_TP_GU_PREFETCH='1')
    c['engine_env'][other]='1'
    with pytest.raises(runner.RunnerError):runner.validated(c)


@pytest.mark.parametrize('other',['STRATA_TP_DECODE_FP4','STRATA_TP_NATIVE_GROUPED',
                                'STRATA_TP_DECODE_GU_VECTOR','STRATA_TP_GU_WARP32','STRATA_TP_GU_PREFETCH'])
def test_route_order_rejects_inactive_decode_routes(tmp_path,other):
    c=config(tmp_path);c['engine_env']['STRATA_TP_ROUTE_ORDER']='1';c['engine_env'][other]='1'
    with pytest.raises(runner.RunnerError):runner.validated(c)


@pytest.mark.parametrize('batch',['256','32','257','0'])
def test_qsa_prompt_batch_is_bounded(tmp_path,batch):
    c=config(tmp_path);c['engine_env']['STRATA_TP_QSA_PROMPT_BATCH']=batch
    if batch=='256':assert runner.validated(c)['engine_env']['STRATA_TP_QSA_PROMPT_BATCH']=='256'
    else:
        with pytest.raises(runner.RunnerError):runner.validated(c)


def test_group_reuse_policy_is_explicit(tmp_path):
    c=config(tmp_path);c['engine_env']['STRATA_TP_GROUP_REUSE']='4'
    assert runner.validated(c)['engine_env']['STRATA_TP_GROUP_REUSE']=='4'
    c['engine_env']['STRATA_TP_GROUP_REUSE']='8'
    with pytest.raises(runner.RunnerError):runner.validated(c)
    c['engine_env']['STRATA_TP_GROUP_REUSE']='4';c['engine_env']['STRATA_TP_DECODE_FP4']='1'
    with pytest.raises(runner.RunnerError):runner.validated(c)


def test_cooperative_roce_geometry_is_explicit(tmp_path):
    c=config(tmp_path)
    c['engine_env']['STRATA_TP_ROCE_BLOCKS']='8'
    assert runner.validated(c)['engine_env']['STRATA_TP_ROCE_BLOCKS']=='8'
    c['engine_env']['STRATA_TP_ROCE_BLOCKS']='2'
    with pytest.raises(runner.RunnerError):
        runner.validated(c)


@pytest.mark.parametrize('minimum,maximum', [('4', '8'), ('0', '8'), ('8', '4'), ('1', '33'), ('4x', '8')])
def test_fast_nccl_channel_bounds(tmp_path, minimum, maximum):
    c = config(tmp_path)
    c['env'] = {'NCCL_MIN_NCHANNELS': minimum, 'NCCL_MAX_NCHANNELS': maximum}
    if (minimum, maximum) == ('4', '8'):
        assert runner.validated(c)['env'] == c['env']
    else:
        with pytest.raises(runner.RunnerError):
            runner.validated(c)


@pytest.mark.parametrize('routes,accepted', [('-1,1,0', True), ('0,-1,1', True),
    ('0,1,9', False), ('0,1', False), ('0,1,2;false', False)])
def test_fast_roce_routes_are_bounded(tmp_path, routes, accepted):
    c = config(tmp_path)
    c['engine_env'].update(STRATA_TP_TRANSPORT='roce',
        STRATA_TP_ROCE_HCAS='roceP2p1s0f0,roceP2p1s0f1', STRATA_TP_ROCE_ROUTES=routes)
    if accepted:
        assert runner.validated(c)['engine_env']['STRATA_TP_ROCE_ROUTES'] == routes
    else:
        with pytest.raises(runner.RunnerError):
            runner.validated(c)


@pytest.mark.parametrize('mutation', ['changed_asset', 'external_embedding', 'unknown_engine_env', 'overflow', 'mtp_unpinned'])
def test_unpinned_or_invalid_profile_refused(tmp_path, mutation):
    c = config(tmp_path)
    if mutation == 'changed_asset':
        (tmp_path / 'model/pack/experts.bin').write_text('changed')
    elif mutation == 'external_embedding':
        c['args'][c['args'].index('--embd') + 1] = str(tmp_path / 'external.gguf')
    elif mutation == 'unknown_engine_env':
        c['engine_env']['LD_PRELOAD'] = 'unexpected'
    elif mutation == 'overflow':
        c['args'][c['args'].index('--context') + 1] = '262145'
    else:
        c['args'] += ['--mtp-path', str(tmp_path / 'model/mtp'), '--mtp-drafts', '4']
    with pytest.raises(runner.RunnerError):
        runner.validated(c)
