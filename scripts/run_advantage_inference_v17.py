"""Fresh end-to-end latency, fixed-cohort accuracy and visual perturbations."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
MASTER = ROOT / 'runs/advantage_validation_v17'
sys.path.insert(0, str(ROOT / 'src'))
os.environ.setdefault('HF_HOME', str(MASTER / 'cache/hf'))
os.environ.setdefault('TRANSFORMERS_CACHE', str(MASTER / 'cache/hf/hub'))
os.environ.setdefault('MPLCONFIGDIR', str(MASTER / 'cache/matplotlib'))
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')

import numpy as np
from PIL import Image
from clickagent.advantage_validation import map_to_original, perturb, stable_key
from clickagent.advantage_runtimes import configure_paths


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=json_default) + '\n', encoding='utf-8')
    os.replace(temp, path)


def status(folder, phase, **fields):
    value = dict(phase=phase, complete=phase == 'complete', pid=os.getpid(),
                 updated_at_utc=datetime.now(timezone.utc).isoformat(), **fields)
    write(folder / 'status.json', value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def sync(torch):
    for device in range(torch.cuda.device_count()):
        torch.cuda.synchronize(device)


def infer_once(runtime, payload, instruction, torch):
    sync(torch)
    for d in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(d)
    before = runtime.counter.snapshot()
    started = time.perf_counter()
    with Image.open(BytesIO(payload)) as opened:
        image = opened.convert('RGB')
    try:
        result = runtime.infer(image, instruction)
    finally:
        image.close()
    sync(torch)
    elapsed = time.perf_counter() - started
    after = runtime.counter.snapshot()
    assert result.get('ground_truth_in_model_input') is False
    return result, dict(elapsed_seconds=elapsed,
        counter={k: after[k] - before[k] for k in before},
        peak_allocated_bytes=[torch.cuda.max_memory_allocated(d) for d in range(torch.cuda.device_count())],
        peak_reserved_bytes=[torch.cuda.max_memory_reserved(d) for d in range(torch.cuda.device_count())])


def public_call(runtime, payload, instruction, torch):
    return infer_once(runtime, payload, instruction, torch)


def score_output(case, output, affine, metrics, condition, repetition=None):
    from clickagent.public_evaluation import point_in_geometry
    points = output.get('seed_points') or [output.get('point')]
    mapped = [map_to_original(point, affine, case['image_size']) for point in points]
    hits = [bool(p is not None and point_in_geometry([p[0] / case['image_size'][0], p[1] / case['image_size'][1]], case['target_geometry'])) for p in mapped]
    return dict(sample_id=case['sample_id'], source=case['dataset'], rgb_group=case['pixel_sha256'],
                source_group=case['source_group'], condition=condition, repetition=repetition,
                point=mapped[0], seed_points=mapped, seed_hits=hits, hit=float(np.mean(hits)),
                measured=metrics, output=output, input_image_sha256=case['image_sha256'])


def register(folder, method, pixels, stage):
    p = folder / 'protocol.json'
    code = [Path(__file__), ROOT / 'src/clickagent/advantage_runtimes.py',
            ROOT / 'src/clickagent/advantage_validation.py', MASTER / 'plan.json',
            MASTER / 'data/cohorts.json', ROOT / 'CURRENT_RESEARCH.json']
    # Operational CURRENT_RESEARCH may change; seal only immutable science inputs.
    code = [q for q in code if q.name != 'CURRENT_RESEARCH.json']
    repo = {'tag': 'TAG', 'aima': 'GUI-AIMA', 'aima_zoom': 'GUI-AIMA',
            'mvp': 'MVP', 'mvp_single': 'MVP'}.get(method)
    if repo:
        code += sorted((ROOT / 'third_party' / repo).rglob('*.py'))
    if method in ['ours', 'ours_s1', 'ours_gui']:
        code += [ROOT / 'runs/quality_readout_v15_3k/models/frozen.json',
                 ROOT / 'runs/crossfit_candidates_v12/runtime_identity.json']
    if method == 'ui_tars':
        code += [ROOT / 'src/clickagent/model_baselines.py',
                 ROOT / 'models/UI-TARS-1.5-7B/clickagent_download_verified.json']
    if method in ['aima', 'aima_zoom']:
        code += [ROOT / 'models/GUI-AIMA-3B/clickagent_download_verified_v17.json']
    if method == 'tag':
        candidates = [ROOT / 'models' / name / 'clickagent_download_verified_v17.json'
                      for name in ['MiniCPM-Llama3-V-2_5', 'MiniCPM-Llama3-V-2_5_modelscope_staging']]
        code += [next(p for p in candidates if p.exists())]
    if p.exists():
        spec = read(p)
        for name, expected in spec['hashes'].items():
            assert sha(name) == expected, name
        assert spec['method'] == method and spec['max_pixels'] == pixels
        return spec
    assert not (folder / 'accuracy').exists() and not (folder / 'speed').exists()
    spec = dict(registered_at_utc=datetime.now(timezone.utc).isoformat(), method=method,
        max_pixels=pixels, stage=stage, hashes={str(q): sha(q) for q in code},
        plan_sha256=sha(MASTER / 'plan.json'), label_access='Only score_output after public-only runtime.infer.',
        failed_outputs='Missing/invalid model outputs are misses. Technical adapter/resource failures stop this arm; not accuracy wins.',
        speed='No feature/prediction cache, batch1,2 warmups,2 repetitions on60 fixed screenshots. Model loading and report writes excluded.',
        comparisons='Exploratory reused-cohort extensions; do not retune or reselect mainline.',
        method_source='Pinned official code with declared runtime adapter; FP16 and per-view budget recorded.')
    write(p, spec)
    return spec


def payload_for(case, condition):
    payload = Path(case['image']).read_bytes()
    assert hashlib.sha256(payload).hexdigest() == case['image_sha256']
    if condition == 'clean':
        return payload, (1., 1., 0., 0.)
    with Image.open(BytesIO(payload)) as im:
        changed, affine = perturb(im, condition)
    stream = BytesIO()
    changed.save(stream, format='PNG')
    changed.close()
    return stream.getvalue(), affine


def reuse_accuracy(folder, method, cases, pixels):
    if pixels != 262144 or method not in ['ours', 'ours_gui', 'ui_tars']:
        return None
    if method == 'ui_tars':
        parent = ROOT / 'runs/external_baselines_v16'
        completion = read(parent / 'completion.json')
        for path, expected in completion['artifact_sha256'].items():
            assert sha(path) == expected
        raw = [json.loads(s) for s in (parent / 'ui_tars/predictions.jsonl').read_text(encoding='utf-8').splitlines()]
        lookup = {r['sample_id']: r for r in raw}
        outputs = {c['sample_id']: dict(point=lookup[c['sample_id']]['point'], ground_truth_in_model_input=False) for c in cases}
    else:
        parent = ROOT / 'runs/contribution_v16'
        completion = read(parent / 'completion.json')
        for path, expected in completion['artifact_sha256'].items():
            assert sha(path) == expected
        lookup = {r['sample_id']: r for r in read(parent / 'evaluation/independent/details.json')}
        outputs = {}
        for c in cases:
            methods = lookup[c['sample_id']]['methods']
            if method == 'ours':
                seed_points = [methods[f'aligned_s{i}']['point'] for i in [1, 2, 3]]
                output = dict(point=seed_points[0], seed_points=seed_points, ground_truth_in_model_input=False)
            else:
                output = dict(point=methods['gui_verifier_k16']['point'], ground_truth_in_model_input=False)
            outputs[c['sample_id']] = output
    rows = [score_output(c, outputs[c['sample_id']], (1., 1., 0., 0.), None, 'clean') for c in cases]
    write(folder / 'accuracy/results.json', dict(complete=True, n=len(rows), rows=rows,
          reused_authenticated_accuracy=True, no_latency_reuse=True, parent_completion_sha256=sha(parent / 'completion.json')))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', required=True, choices=['ours', 'ours_s1', 'ours_gui', 'ui_tars', 'qwen', 'aima', 'aima_zoom', 'tag', 'mvp', 'mvp_single'])
    parser.add_argument('--pixels', type=int, default=262144)
    parser.add_argument('--stage', choices=['smoke', 'all', 'speed', 'stronger'], default='all')
    args = parser.parse_args()
    method = args.method
    configure_paths(method)
    import torch
    from clickagent.advantage_runtimes import create_runtime
    torch.set_num_threads(4)
    adapter_sha = sha(ROOT / 'src/clickagent/advantage_runtimes.py')
    worker_sha = sha(Path(__file__))
    name = f'{method}_p{args.pixels}_{args.stage}'
    if args.stage == 'smoke':
        name += '_' + adapter_sha[:12]
    folder = MASTER / ('setup/smoke' if args.stage == 'smoke' else 'inference') / name
    folder.mkdir(parents=True, exist_ok=True)
    if args.stage != 'smoke':
        register(folder, method, args.pixels, args.stage)
    cohorts = read(MASTER / 'data/cohorts.json')
    selected = cohorts['speed_robustness']
    try:
        status(folder, 'loading_model')
        started = time.perf_counter()
        runtime = create_runtime(method, folder, args.pixels)
        sync(torch)
        metadata = dict(runtime.metadata, load_seconds=time.perf_counter() - started,
                        torch_version=torch.__version__, transformers_version=__import__('transformers').__version__,
                        cuda_version=torch.version.cuda, gpus=[torch.cuda.get_device_name(d) for d in range(torch.cuda.device_count())])
        write(folder / 'runtime.json', metadata)
        warm_ids = [e['sample_id'] for e in cohorts['warmup']]
        warm_pool = [json.loads(s) for s in (ROOT / 'runs/quality_readout_v15_3k/data/new_reserve.jsonl').read_text(encoding='utf-8').splitlines()]
        warm = {c['sample_id']: c for c in warm_pool}
        warm_outputs = []
        for sample_id in warm_ids:
            case = warm[sample_id]
            payload, affine = payload_for(case, 'clean')
            output, timing = public_call(runtime, payload, case['instruction'], torch)
            warm_outputs.append(dict(sample_id=sample_id, output=output, metrics=timing))
        write(folder / 'warmup.json', dict(complete=True, outputs=warm_outputs, not_test_or_published_timing=True))
        if args.stage == 'smoke':
            assert sha(ROOT / 'src/clickagent/advantage_runtimes.py') == adapter_sha, 'Adapter changed during development check'
            assert sha(Path(__file__)) == worker_sha, 'Worker changed during development check'
            write(folder / 'smoke_receipt.json', dict(complete=True, cases=2,
                  adapter_sha256=adapter_sha,
                  worker_sha256=worker_sha, method=method, max_pixels=args.pixels,
                  development_inputs_only=True))
            status(folder, 'complete', smoke_only=True, cases=2)
            return
        cases = selected if args.stage == 'stronger' else cohorts['full_test']
        rows = None
        if args.stage == 'all':
            rows = reuse_accuracy(folder, method, cases, args.pixels)
        if args.stage in ['all', 'stronger'] and rows is None:
            rows = []
            for i, case in enumerate(cases, 1):
                path = folder / f'accuracy/cases/{stable_key(case["sample_id"])}.json'
                if path.exists():
                    row = read(path)
                else:
                    payload, affine = payload_for(case, 'clean')
                    output, timing = public_call(runtime, payload, case['instruction'], torch)
                    row = score_output(case, output, affine, timing, 'clean')
                    write(path, row)
                rows.append(row)
                if i % 5 == 0 or i == len(cases):
                    status(folder, 'accuracy', completed=i, total=len(cases))
            write(folder / 'accuracy/results.json', dict(complete=True, n=len(rows), rows=rows, fresh_inference=True))
        if args.stage == 'speed' or (args.stage == 'all' and method != 'ours'):
            speed = []
            rng = np.random.default_rng(20261705)
            for repeat in range(2):
                for j, index in enumerate(rng.permutation(len(selected)), 1):
                    case = selected[int(index)]
                    path = folder / f'speed/cases/{repeat}_{stable_key(case["sample_id"])}.json'
                    if path.exists():
                        row = read(path)
                    else:
                        payload, affine = payload_for(case, 'clean')
                        output, timing = public_call(runtime, payload, case['instruction'], torch)
                        row = score_output(case, output, affine, timing, 'clean', repeat)
                        write(path, row)
                    speed.append(row)
                    if j % 5 == 0 or j == len(selected):
                        status(folder, 'speed', completed=repeat * len(selected) + j, total=2 * len(selected))
            write(folder / 'speed/results.json', dict(complete=True, unique_screenshots=len(selected), repetitions=2,
                  n=len(speed), rows=speed, no_feature_cache=True))
        if args.stage == 'all':
            robust = []
            clean = {r['sample_id']: r for r in rows}
            for condition in ['jpeg70', 'border24']:
                for i, case in enumerate(selected, 1):
                    path = folder / f'robustness/{condition}/{stable_key(case["sample_id"])}.json'
                    if path.exists():
                        row = read(path)
                    else:
                        payload, affine = payload_for(case, condition)
                        output, timing = public_call(runtime, payload, case['instruction'], torch)
                        row = score_output(case, output, affine, timing, condition)
                        write(path, row)
                    robust.append(row)
                    if i % 5 == 0 or i == len(selected):
                        status(folder, 'robustness_' + condition, completed=i, total=len(selected))
            write(folder / 'robustness/results.json', dict(complete=True, unique_screenshots=60, rows=robust,
                  clean_rows=[clean[c['sample_id']] for c in selected], conditions=['jpeg70', 'border24']))
        register(folder, method, args.pixels, args.stage)
        artifacts = [p for p in folder.glob('*/results.json')]
        write(folder / 'completion.json', dict(complete=True, hashes={str(p): sha(p) for p in [folder / 'protocol.json', folder / 'runtime.json'] + artifacts}))
        status(folder, 'complete')
    except Exception:
        status(folder, 'failed', error=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
