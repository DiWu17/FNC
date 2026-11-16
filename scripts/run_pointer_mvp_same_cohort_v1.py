"""Post-V2, public-only same-cohort MVP and Qwen3-VL-8B inference.

This is a retrospective comparison on the fixed PointerBench Pro500/Sheets500
cohorts. V2 labels have already been seen by the project, so no blind-test
claim is possible. This file never reads labels or scores predictions. The
separate analysis program may open labels only after both method bundles are
SHA sealed in public_completion.json.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

RUN = ROOT / 'runs/pointer_mvp_same_cohort_v1'
PRED = RUN / 'predictions'
PROTOCOL = RUN / 'protocol.json'
STATUS = RUN / 'status.json'
COMPLETION = RUN / 'public_completion.json'
V2 = ROOT / 'runs/paper_scale_ablation_v2'
DATA = V2 / 'pointerbench'
PUBLIC = DATA / 'public.jsonl'
AUDIT = DATA / 'audit.json'
V2_POINTER_PROTOCOL = DATA / 'evaluation_protocol_v2.json'
V2_POINTER_RECEIPT = DATA / 'public_completion.json'
V2_ANALYSIS_SUMMARY = V2 / 'analysis/summary.json'
MODEL = ROOT / 'models/Qwen3-VL-8B-Instruct'
ADAPTER = ROOT / 'src/clickagent/advantage_runtimes.py'
MVP = ROOT / 'third_party/MVP'
METHODS = ('mvp', 'qwen')
N = 1000
PIXELS_PER_VIEW = 262144
MODEL_SHARDS = tuple(f'model-{i:05d}-of-00004.safetensors' for i in range(1, 5))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def public_rows():
    with PUBLIC.open('r', encoding='utf-8-sig') as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    require(len(rows) == N and
            [row['suite'] for row in rows] == ['pro'] * 500 + ['sheets'] * 500 and
            len({row['sample_id'] for row in rows}) == N,
            'PointerBench public cohort size/order/ID drift')
    expected = {'application', 'application_group', 'dataset', 'dhash256',
                'image', 'image_sha256', 'image_size', 'instruction',
                'pixel_sha256', 'sample_id', 'suite', 'ui_type'}
    require(all(set(row) == expected and row['dataset'] == 'PointerBench' and
                row['image_size'] == [1024, 768] for row in rows),
            'Public input schema, coordinate frame, or label boundary changed')
    return rows


def atomic_json(path, value, *, immutable=False):
    path = Path(path)
    if immutable and path.exists():
        require(read(path) == value, 'Immutable output changed: ' + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f'.tmp.{os.getpid()}.{time.time_ns()}')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n',
                    encoding='utf-8')
    os.replace(temp, path)


def status(phase, **fields):
    record = dict(phase=phase, updated_at_utc=now(), pid=os.getpid(), **fields)
    atomic_json(STATUS, record)
    print(json.dumps(record, ensure_ascii=False), flush=True)


def source_files():
    files = [Path(__file__), ADAPTER, PUBLIC, AUDIT, V2_POINTER_PROTOCOL,
             V2_POINTER_RECEIPT, V2_ANALYSIS_SUMMARY,
             V2 / 'eval/pro/predictions.json',
             V2 / 'eval/sheets/predictions.json',
             DATA / 'preseal_label_access_addendum_v2.json',
             MVP / 'mvp_sspro_qwen3vl.py', MVP / 'qwen3_vl/__init__.py',
             *sorted((MVP / 'qwen3_vl').glob('*.py')),
             MODEL / 'clickagent_download.json', MODEL / 'config.json',
             MODEL / 'generation_config.json', MODEL / 'model.safetensors.index.json',
             MODEL / 'preprocessor_config.json', MODEL / 'tokenizer_config.json',
             MODEL / 'chat_template.json']
    return sorted(set(files), key=str)


def validate_parent():
    audit = read(AUDIT)
    require(audit['complete'] is True and
            audit['independent_history_audit_passed'] is True and
            audit['counts'] == {'pro': 500, 'sheets': 500} and
            audit['public_sha256'] == sha(PUBLIC),
            'V2 source audit drift')
    pointer = read(V2_POINTER_PROTOCOL)
    require(pointer['complete'] is True and
            pointer['public_source_sha256'] == sha(PUBLIC) and
            pointer['labels_source_sha256_unopened'] == audit['labels_sha256'] and
            pointer['strict_blind_confirmation'] is False,
            'V2 Pointer protocol drift')
    receipt = read(V2_POINTER_RECEIPT)
    require(receipt['complete'] is True and receipt['labels_opened'] is False and
            receipt['pointer_protocol_sha256'] == sha(V2_POINTER_PROTOCOL),
            'V2 public prediction receipt drift')
    analysis = read(V2_ANALYSIS_SUMMARY)
    require(analysis['complete'] is True and
            analysis['input_prediction_sha256']['pro'] ==
            receipt['prediction_sha256']['pro'] and
            analysis['input_prediction_sha256']['sheets'] ==
            receipt['prediction_sha256']['sheets'],
            'V2 completed analysis/prediction identity drift')
    for suite in ('pro', 'sheets'):
        require(sha(V2 / f'eval/{suite}/predictions.json') ==
                receipt['prediction_sha256'][suite],
                'V2 sealed prediction bundle drift: ' + suite)
    return public_rows()


def register():
    """Freeze public data, adapter/code and actual 8B shard contents."""
    require(not PROTOCOL.exists(), 'Protocol already registered; use verify-gate')
    require(not PRED.exists() and not COMPLETION.exists(),
            'Cannot register after public predictions exist')
    rows = validate_parent()
    status('registering', checked_public=len(rows), labels_opened=False)
    files = source_files()
    require(all(path.is_file() for path in files), 'A frozen source file is missing')
    hashes = {str(path.relative_to(ROOT)): sha(path) for path in files}
    shard_hashes = {}
    for name in MODEL_SHARDS:
        path = MODEL / name
        require(path.is_file(), 'Model shard missing: ' + name)
        shard_hashes[name] = dict(size_bytes=path.stat().st_size, sha256=sha(path))
        print(json.dumps({'stage': 'hash_model_shard', 'name': name},
                         ensure_ascii=False), flush=True)
    expected_bytes = read(MODEL / 'clickagent_download.json')['weight_bytes']
    require(sum(v['size_bytes'] for v in shard_hashes.values()) == expected_bytes,
            'Qwen3-VL-8B model shard byte count drift')
    record = dict(
        complete=True, run_id='pointer_mvp_same_cohort_v1', registered_at_utc=now(),
        design='Retrospective post-V2 same-sample comparison; no new blind claim',
        methods=list(METHODS), total_n=N, primary='PointerBench Pro500',
        secondary='PointerBench Sheets500', public_order='Pro500 then Sheets500',
        primary_contrast='full_aligned_three_seed_mean_vs_mvp_on_Pro500',
        public_source_sha256=sha(PUBLIC), public_image_sha256_in_manifest=True,
        source_hashes=hashes, model_shards=shard_hashes,
        model='Qwen/Qwen3-VL-8B-Instruct, local ModelScope copy, frozen FP16',
        mvp=dict(adapter='clickagent.advantage_runtimes.MVPAdapter',
                 max_pixels_per_view=PIXELS_PER_VIEW, target_layer_idx=24,
                 target_token=',', max_subimages=2, consistency_threshold=14,
                 prompt='Unchanged official mvp_sspro_qwen3vl.SYSTEM_PROMPT',
                 decoding='deterministic max_new_tokens=32'),
        qwen=dict(adapter='clickagent.advantage_runtimes.QwenCoordinateAdapter',
                  max_pixels=PIXELS_PER_VIEW,
                  prompt='Unchanged COORDINATE_PROMPT; normalized 0..1000 point',
                  decoding='deterministic max_new_tokens=64'),
        common_runtime=dict(dtype='float16', attention='sdpa',
                            device_map='balanced', max_memory_per_gpu='12GiB',
                            batch_size=1, model_frozen=True,
                            no_test_tuned_configuration=True),
        resource_caveat='MVP has up to three views and higher total inference cost; per-view pixel cap is not a matched total-compute budget. MVP minimum image pixels 3136, native Qwen minimum 16384. This local 262144-pixel cap is not the publisher full-resolution setting.',
        infer='Public image and instruction only; no label read, test scoring or model selection',
        invalid_output='Null or out-of-bounds point retained and scored wrong; technical failure stops run',
        metric='Inclusive native 1024x768 point-in-bbox, full fixed denominator',
        analysis='Separate post-seal program; Pro primary RGB-image-cluster paired bootstrap; Sheets secondary',
        bootstrap_unit='RGB image pixel_sha256',
        bootstrap_replicates=10000,
        bootstrap_seed=20261112,
        prior_gt_visibility='V2 GT and result already opened; plus at least seven pre-V2-seal previews; retrospective',
        model_forward_calls_before_registration=0,
        labels_opened_by_this_runner=False,
    )
    atomic_json(PROTOCOL, record, immutable=True)
    status('registered', protocol_sha256=sha(PROTOCOL), labels_opened=False)


def protocol(*, deep_model=False):
    require(PROTOCOL.is_file(), 'Register before inference')
    p = read(PROTOCOL)
    require(p['complete'] is True and p['methods'] == list(METHODS) and
            p['total_n'] == N and p['public_source_sha256'] == sha(PUBLIC) and
            p['labels_opened_by_this_runner'] is False,
            'Registered protocol drift')
    require(set(p['source_hashes']) ==
            {str(path.relative_to(ROOT)) for path in source_files()},
            'Registered source set drift')
    for rel, expected in p['source_hashes'].items():
        require(sha(ROOT / rel) == expected, 'Registered source SHA drift: ' + rel)
    require(set(p['model_shards']) == set(MODEL_SHARDS),
            'Registered model shard set drift')
    for name, frozen in p['model_shards'].items():
        path = MODEL / name
        require(path.stat().st_size == frozen['size_bytes'],
                'Registered model shard size drift: ' + name)
        if deep_model:
            require(sha(path) == frozen['sha256'],
                    'Registered model shard SHA drift: ' + name)
    validate_parent()
    return p


def case_key(sample_id):
    return hashlib.sha256(sample_id.encode('utf-8')).hexdigest()[:24]


def public_identity(row):
    return hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode('utf-8')).hexdigest()


def valid_point(point, image_size):
    return (point is None or
            (isinstance(point, list) and len(point) == 2 and
             all(type(x) is int and 0 <= x < side
                 for x, side in zip(point, image_size))))


def check_case(record, public, method, protocol_sha):
    require(record['sample_id'] == public['sample_id'] and
            record['suite'] == public['suite'] and
            record['method'] == method and
            record['image_sha256'] == public['image_sha256'] and
            record['pixel_sha256'] == public['pixel_sha256'] and
            record['public_identity'] == public_identity(public) and
            record['protocol_sha256'] == protocol_sha and
            valid_point(record['point'], public['image_size']) and
            record['labels_opened'] is False,
            'Per-case public-only prediction identity drift: ' + public['sample_id'])
    require(isinstance(record['elapsed_seconds'], (int, float)) and
            math.isfinite(record['elapsed_seconds']) and
            record['elapsed_seconds'] >= 0 and
            type(record['backbone_forward_calls']) is int and
            type(record['processed_pixels']) is int and
            record['backbone_forward_calls'] >= 0 and
            record['processed_pixels'] >= 0,
            'Per-case runtime counter drift: ' + public['sample_id'])


def infer(method):
    p = protocol(deep_model=True)
    require(method in METHODS, 'Unknown fixed method')
    rows = public_rows()
    bundle_path = PRED / f'{method}.json'
    if bundle_path.exists():
        verify_bundle(method)
        status('already_complete', method=method, completed=N, total=N)
        return
    if all((PRED / method / 'cases' / (case_key(row['sample_id']) + '.json')).exists()
           for row in rows):
        seal_bundle(method)
        return
    # Imports and model loading happen only after the frozen public protocol is checked.
    from PIL import Image
    import torch
    from clickagent.advantage_runtimes import configure_paths, create_runtime
    torch.set_num_threads(4)
    configure_paths(method)
    status('loading_model', method=method, labels_opened=False)
    load_start = time.perf_counter()
    runtime = create_runtime(method, RUN / f'runtime/{method}', PIXELS_PER_VIEW)
    load_seconds = time.perf_counter() - load_start
    metadata_path = RUN / f'runtime/{method}/metadata.json'
    if metadata_path.exists():
        old = read(metadata_path)
        require({k: v for k, v in old.items() if k not in
                 ('load_seconds', 'torch_version', 'transformers_version', 'cuda_version')}
                == runtime.metadata,
                'Runtime metadata changed on resume: ' + method)
    else:
        atomic_json(metadata_path,
                    dict(runtime.metadata, load_seconds=load_seconds,
                         torch_version=torch.__version__,
                         transformers_version=__import__('transformers').__version__,
                         cuda_version=torch.version.cuda), immutable=True)
    protocol_sha = sha(PROTOCOL)
    for index, public in enumerate(rows, 1):
        path = PRED / method / 'cases' / (case_key(public['sample_id']) + '.json')
        if path.exists():
            check_case(read(path), public, method, protocol_sha)
            continue
        image_path = Path(public['image'])
        require(sha(image_path) == public['image_sha256'],
                'Public image SHA drift: ' + public['sample_id'])
        with Image.open(image_path) as opened:
            require(list(opened.size) == public['image_size'],
                    'Public image dimension drift: ' + public['sample_id'])
            image = opened.convert('RGB')
        try:
            for device in range(torch.cuda.device_count()):
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            before = runtime.counter.snapshot()
            started = time.perf_counter()
            with torch.inference_mode():
                output = runtime.infer(image, public['instruction'])
            for device in range(torch.cuda.device_count()):
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            after = runtime.counter.snapshot()
            require(output.get('ground_truth_in_model_input') is False,
                    'Adapter label-free output assurance failed')
            point = output.get('point')
            if point is not None:
                point = [int(x) for x in point]
                if not valid_point(point, public['image_size']):
                    point = None
            record = dict(sample_id=public['sample_id'], suite=public['suite'],
                          method=method, point=point,
                          image_sha256=public['image_sha256'],
                          pixel_sha256=public['pixel_sha256'],
                          public_identity=public_identity(public),
                          protocol_sha256=protocol_sha,
                          elapsed_seconds=elapsed,
                          backbone_forward_calls=after['backbone_forward_calls'] -
                          before['backbone_forward_calls'],
                          processed_pixels=after['processed_pixels'] -
                          before['processed_pixels'],
                          peak_allocated_bytes=[torch.cuda.max_memory_allocated(device)
                                                for device in range(torch.cuda.device_count())],
                          output_failure=output.get('output_failure'),
                          generated_text=output.get('generated_text') if method == 'qwen'
                          else None, labels_opened=False)
            check_case(record, public, method, protocol_sha)
            atomic_json(path, record, immutable=True)
        finally:
            image.close()
        if index % 5 == 0 or index == N:
            status('inference', method=method, completed=index, total=N,
                   last_sample_id=public['sample_id'], labels_opened=False)
    seal_bundle(method)


def seal_bundle(method):
    p = protocol()
    rows = public_rows()
    protocol_sha = sha(PROTOCOL)
    case_hashes = {}
    collected = []
    for public in rows:
        path = PRED / method / 'cases' / (case_key(public['sample_id']) + '.json')
        require(path.is_file(), 'Missing public-only prediction: ' + public['sample_id'])
        record = read(path)
        check_case(record, public, method, protocol_sha)
        case_hashes[public['sample_id']] = sha(path)
        collected.append(record)
    value = dict(complete=True, method=method, n=N, rows=collected,
                 case_hashes=case_hashes, protocol_sha256=protocol_sha,
                 public_source_sha256=p['public_source_sha256'],
                 labels_opened=False)
    atomic_json(PRED / f'{method}.json', value, immutable=True)
    status('method_complete', method=method, completed=N, total=N,
           prediction_sha256=sha(PRED / f'{method}.json'), labels_opened=False)


def verify_bundle(method):
    p = protocol()
    path = PRED / f'{method}.json'
    require(path.is_file(), 'Prediction bundle not yet sealed: ' + method)
    bundle = read(path)
    rows = public_rows()
    require(bundle['complete'] is True and bundle['method'] == method and
            bundle['n'] == N and bundle['labels_opened'] is False and
            bundle['protocol_sha256'] == sha(PROTOCOL) and
            bundle['public_source_sha256'] == p['public_source_sha256'] and
            len(bundle['rows']) == N and len(bundle['case_hashes']) == N and
            set(bundle['case_hashes']) == {row['sample_id'] for row in rows},
            'Prediction aggregate identity drift: ' + method)
    for public, record in zip(rows, bundle['rows']):
        check_case(record, public, method, sha(PROTOCOL))
        case_path = PRED / method / 'cases' / (case_key(public['sample_id']) + '.json')
        require(sha(case_path) == bundle['case_hashes'][public['sample_id']] and
                read(case_path) == record,
                'Per-case SHA drift: ' + public['sample_id'])
    return sha(path)


def verify_gate():
    hashes = {method: verify_bundle(method) for method in METHODS}
    if COMPLETION.exists():
        receipt = read(COMPLETION)
        require(receipt['complete'] is True and receipt['n'] == N and
                receipt['prediction_sha256'] == hashes and
                receipt['protocol_sha256'] == sha(PROTOCOL) and
                receipt['public_source_sha256'] == sha(PUBLIC) and
                receipt['labels_opened'] is False,
                'Public completion receipt drift')
    return hashes


def seal_completion():
    hashes = verify_gate()
    record = dict(complete=True, n=N, prediction_sha256=hashes,
                  protocol_sha256=sha(PROTOCOL),
                  public_source_sha256=sha(PUBLIC), labels_opened=False)
    atomic_json(COMPLETION, record, immutable=True)
    status('complete_public', n=N, prediction_sha256=hashes, labels_opened=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('register', 'mvp', 'qwen',
                                            'all-public', 'verify-gate'), required=True)
    args = parser.parse_args()
    if args.stage == 'register':
        register()
    elif args.stage == 'verify-gate':
        print(json.dumps({'gate': 'passed', 'prediction_sha256': verify_gate(),
                          'labels_opened': False}), flush=True)
    else:
        for method in (METHODS if args.stage == 'all-public' else (args.stage,)):
            infer(method)
        if all((PRED / f'{method}.json').exists() for method in METHODS):
            seal_completion()


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        status('failed', error=traceback.format_exc(), labels_opened=False)
        raise
