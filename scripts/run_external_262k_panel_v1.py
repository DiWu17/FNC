"""Resumable, label-free external baselines on fixed paper cohorts.

This local panel is retrospective. Predictions are sealed before this run's
analysis opens any labels. The published systems have different view counts,
backbones and training data; 262144 is a per-view cap, not equal compute.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
RUN = ROOT / 'runs/external_262k_panel_v1'
PROTOCOL = RUN / 'protocol.json'
STATUS = RUN / 'status.json'
SOURCES = {
    'pro': ROOT / 'runs/paper_scale_ablation_v2/pointerbench/public.jsonl',
    'sheets': ROOT / 'runs/paper_scale_ablation_v2/pointerbench/public.jsonl',
    'mmb': ROOT / 'runs/mmbench_gui_new_source_v1/public.json',
    'sspro': ROOT / 'runs/paper_scale_ablation_v2/eval/full/public.json',
}
COUNTS = {'pro': 500, 'sheets': 500, 'mmb': 1000, 'sspro': 1581}
PIXELS = 262144
METHODS = {'qwen': ('mmb', 'sspro'), 'mvp': ('mmb', 'sspro'),
           'tag': ('pro', 'sheets', 'mmb', 'sspro')}


def require(ok, why):
    if not ok:
        raise ValueError(why)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value, immutable=False):
    path = Path(path)
    if immutable and path.exists():
        require(read(path) == value, 'Immutable record changed: ' + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp.' + str(os.getpid()))
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(temp, path)


def public(suite):
    path = SOURCES[suite]
    if path.suffix == '.jsonl':
        with path.open('r', encoding='utf-8-sig') as f:
            rows = [json.loads(line) for line in f if line.strip()]
        rows = [r for r in rows if r['suite'] == suite]
    else:
        rows = read(path)
    require(len(rows) == COUNTS[suite], 'Cohort count changed: ' + suite)
    require(len({r['sample_id'] for r in rows}) == len(rows), 'Duplicate sample ID')
    for r in rows:
        require({'sample_id', 'image', 'instruction', 'image_size', 'image_sha256',
                 'pixel_sha256'} <= set(r), 'Public-only schema changed')
        require(not any(k in r for k in ('bbox', 'target', 'answer', 'label')),
                'Ground truth entered public input')
    return rows


def sources():
    paths = [Path(__file__), ROOT / 'src/clickagent/advantage_runtimes.py',
             ROOT / 'third_party/MVP/mvp_sspro_qwen3vl.py',
             ROOT / 'third_party/TAG/eval_mm/screenspot/prompt_tag.py',
             ROOT / 'third_party/TAG/eval_mm/screenspot/eval_TAG.py',
             ROOT / 'models/Qwen3-VL-8B-Instruct/clickagent_download.json',
             ROOT / 'models/MiniCPM-Llama3-V-2_5/clickagent_download_verified_v17.json',
             *set(SOURCES.values())]
    return sorted(set(paths), key=str)


def register():
    require(not PROTOCOL.exists(), 'Protocol already registered')
    require(not (RUN / 'predictions').exists(), 'Predictions exist before registration')
    for suite in COUNTS:
        public(suite)
    inherited = ROOT / 'runs/pointer_mvp_same_cohort_v1/public_completion.json'
    require(read(inherited)['complete'], 'Pointer Qwen/MVP prior seal incomplete')
    paths = sources() + [inherited]
    require(all(p.is_file() for p in paths), 'Source file missing')
    record = dict(complete=True, run_id=RUN.name,
                  registered_at_utc=__import__('datetime').datetime.now(
                      __import__('datetime').timezone.utc).isoformat(),
                  design='Retrospective same-cohort external baseline panel',
                  methods={k: list(v) for k, v in METHODS.items()},
                  cohort_n=COUNTS, max_pixels_per_view=PIXELS,
                  source_sha256={str(p.relative_to(ROOT)): sha(p) for p in paths},
                  model='Qwen/MVP share frozen Qwen3-VL-8B; TAG uses original MiniCPM-Llama3-V-2.5',
                  adapters='Frozen clickagent.advantage_runtimes v17 adapters',
                  decoding='Original existing Qwen/MVP/TAG adapter prompts and deterministic decoding',
                  invalid_output='Missing or out-of-bounds point counts incorrect',
                  metric='Native point in ground-truth bbox, inclusive, fixed denominator',
                  reused_pointer_qwen_mvp='Prior sealed 1000-query comparison, identical public cohort',
                  labels_opened_by_runner=False, no_test_tuning=True,
                  caveat='Per-view pixel cap does not equal total compute or publisher full-resolution settings; TAG native slicing can increase processed pixels beyond its capped public image')
    write(PROTOCOL, record, immutable=True)
    print('REGISTERED', sha(PROTOCOL), flush=True)


def verify_protocol():
    p = read(PROTOCOL)
    require(p['complete'] and p['cohort_n'] == COUNTS and
            p['methods'] == {k: list(v) for k, v in METHODS.items()} and
            p['max_pixels_per_view'] == PIXELS, 'Protocol design drift')
    for rel, digest in p['source_sha256'].items():
        require(sha(ROOT / rel) == digest, 'Source changed: ' + rel)
    return p


def key(sample_id):
    return hashlib.sha256(sample_id.encode('utf-8')).hexdigest()[:24]


def identity(row):
    return hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode('utf-8')).hexdigest()


def case_path(method, suite, row):
    return RUN / 'predictions' / method / suite / 'cases' / (key(row['sample_id']) + '.json')


def check_case(r, row, method, suite):
    require(r['sample_id'] == row['sample_id'] and r['method'] == method and
            r['suite'] == suite and r['image_sha256'] == row['image_sha256'] and
            r['public_identity'] == identity(row) and
            r['protocol_sha256'] == sha(PROTOCOL) and r['labels_opened'] is False,
            'Prediction identity drift: ' + row['sample_id'])
    point = r['point']
    require(point is None or (type(point) is list and len(point) == 2 and
            all(type(x) is int and 0 <= x < side
                for x, side in zip(point, row['image_size']))),
            'Invalid point: ' + row['sample_id'])


def status(**fields):
    record = dict(updated_at_utc=__import__('datetime').datetime.now(
        __import__('datetime').timezone.utc).isoformat(), pid=os.getpid(), **fields)
    write(STATUS, record)
    print(json.dumps(record, ensure_ascii=False), flush=True)


def run(method, suite):
    require(method in METHODS and suite in METHODS[method], 'Unregistered method/cohort')
    verify_protocol()
    rows = public(suite)
    bundle = RUN / 'predictions' / method / (suite + '.json')
    if bundle.exists():
        verify_bundle(method, suite)
        status(phase='already_complete', method=method, suite=suite)
        return
    from PIL import Image
    import torch
    from clickagent.advantage_runtimes import configure_paths, create_runtime
    torch.set_num_threads(4)
    configure_paths(method)
    status(phase='loading', method=method, suite=suite)
    runtime = create_runtime(method, RUN / f'runtime/{method}', PIXELS)
    meta = RUN / 'runtime' / method / 'metadata.json'
    if meta.exists():
        require(read(meta) == runtime.metadata, 'Runtime metadata drift')
    else:
        write(meta, runtime.metadata, immutable=True)
    for i, row in enumerate(rows, 1):
        path = case_path(method, suite, row)
        if path.exists():
            check_case(read(path), row, method, suite)
            continue
        image_path = Path(row['image'])
        require(sha(image_path) == row['image_sha256'], 'Image SHA drift')
        with Image.open(image_path) as im:
            require(list(im.size) == row['image_size'], 'Image size drift')
            image = im.convert('RGB')
        try:
            for d in range(torch.cuda.device_count()):
                torch.cuda.synchronize(d)
                torch.cuda.reset_peak_memory_stats(d)
            before = runtime.counter.snapshot()
            t0 = time.perf_counter()
            with torch.inference_mode():
                output = runtime.infer(image, row['instruction'])
            for d in range(torch.cuda.device_count()):
                torch.cuda.synchronize(d)
            elapsed = time.perf_counter() - t0
            after = runtime.counter.snapshot()
            require(output.get('ground_truth_in_model_input') is False,
                    'Adapter label boundary failure')
            point = output.get('point')
            if point is not None:
                point = [int(x) for x in point]
                if not (len(point) == 2 and
                        all(0 <= x < side for x, side in zip(point, row['image_size']))):
                    point = None
            result = dict(sample_id=row['sample_id'], suite=suite, method=method,
                          point=point, image_sha256=row['image_sha256'],
                          public_identity=identity(row), protocol_sha256=sha(PROTOCOL),
                          elapsed_seconds=elapsed,
                          backbone_forward_calls=after['backbone_forward_calls'] - before['backbone_forward_calls'],
                          processed_pixels=after['processed_pixels'] - before['processed_pixels'],
                          peak_allocated_bytes=[torch.cuda.max_memory_allocated(d)
                                                for d in range(torch.cuda.device_count())],
                          output_failure=output.get('output_failure'), labels_opened=False)
            check_case(result, row, method, suite)
            write(path, result, immutable=True)
        finally:
            image.close()
        if i % 5 == 0 or i == len(rows):
            status(phase='inference', method=method, suite=suite,
                   completed=i, total=len(rows), labels_opened=False)
    seal(method, suite)


def seal(method, suite):
    rows = public(suite)
    hashes = {}
    outputs = []
    for row in rows:
        path = case_path(method, suite, row)
        require(path.is_file(), 'Missing prediction: ' + row['sample_id'])
        result = read(path)
        check_case(result, row, method, suite)
        hashes[row['sample_id']] = sha(path)
        outputs.append(result)
    value = dict(complete=True, method=method, suite=suite, n=len(rows),
                 rows=outputs, case_hashes=hashes, protocol_sha256=sha(PROTOCOL),
                 public_sha256=sha(SOURCES[suite]), labels_opened=False)
    bundle = RUN / 'predictions' / method / (suite + '.json')
    write(bundle, value, immutable=True)
    status(phase='sealed', method=method, suite=suite, completed=len(rows),
           prediction_sha256=sha(bundle), labels_opened=False)


def verify_bundle(method, suite):
    verify_protocol()
    bundle = RUN / 'predictions' / method / (suite + '.json')
    value = read(bundle)
    rows = public(suite)
    require(value['complete'] and value['method'] == method and
            value['suite'] == suite and value['n'] == len(rows) and
            value['labels_opened'] is False and
            value['protocol_sha256'] == sha(PROTOCOL) and
            value['public_sha256'] == sha(SOURCES[suite]), 'Bundle drift')
    require(len(value['rows']) == len(rows) and
            set(value['case_hashes']) == {r['sample_id'] for r in rows}, 'Case set drift')
    for row, output in zip(rows, value['rows']):
        check_case(output, row, method, suite)
        path = case_path(method, suite, row)
        require(sha(path) == value['case_hashes'][row['sample_id']] and
                read(path) == output, 'Case SHA drift')
    return sha(bundle)


def gate():
    hashes = {m: {s: verify_bundle(m, s) for s in suites}
              for m, suites in METHODS.items()}
    receipt = dict(complete=True, protocol_sha256=sha(PROTOCOL),
                   prediction_sha256=hashes, labels_opened=False)
    write(RUN / 'public_completion.json', receipt, immutable=True)
    status(phase='complete_public', labels_opened=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('register', 'run', 'verify-gate'), required=True)
    parser.add_argument('--method', choices=METHODS)
    parser.add_argument('--suite', choices=COUNTS)
    args = parser.parse_args()
    try:
        if args.stage == 'register':
            register()
        elif args.stage == 'run':
            require(args.method and args.suite, 'Method and suite required')
            run(args.method, args.suite)
        else:
            gate()
    except BaseException:
        status(phase='failed', error=traceback.format_exc(), labels_opened=False)
        raise
