"""Sealed V13 predictions and shared GUI point-score references."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import time
import uuid

import numpy as np
from PIL import Image
import torch

from clickagent.action_benchmark import sha256, write_json
from clickagent.candidate_budget import coordinate_key, choose_point
from clickagent.multiregion_scorer import load_multiregion_scorer
from crossfit_protocol import key
from train_candidate_scorer import load_cache, subset
from multiregion_v13_protocol import OUT, BUDGETS, read, cases, assert_sealed


def heads(split, device):
    assert_sealed()
    selections = read(OUT/'models/selection.json')['selections']
    models = {}; indices = {}
    for condition in ('c1', 'c3'):
        index = read(OUT/'cache'/f'{split}_{condition}_index.json'); assert index['complete']
        indices[condition] = {r['sample_id']: r for r in index['cases']}
    for name, selected in selections.items():
        identity = selected['source_checkpoint_path']
        if identity not in models:
            assert sha256(Path(selected['checkpoint_path'])) == selected['sha256']
            models[identity] = load_multiregion_scorer(selected['checkpoint_path'], device)[0]
    for i, case in enumerate(cases(split), 1):
        sid = key(case['sample_id']); dest = OUT/'predictions'/split/(sid+'.json')
        pipeline_path = OUT/'pipeline'/split/(sid+'.json'); pipeline = read(pipeline_path)
        native_path = OUT/'native'/split/(sid+'.json')
        input_identity = dict(frozen_sha256=sha256(OUT/'frozen.json'), pipeline_sha256=sha256(pipeline_path),
            selection_sha256=sha256(OUT/'models/selection.json'), native_sha256=sha256(native_path),
            source_sha256=sha256(Path(__file__)), cache_sha256={c: indices[c][case['sample_id']]['sha256'] for c in ('c1', 'c3')})
        for c in ('c1', 'c3'):
            entry = indices[c][case['sample_id']]
            assert sha256(Path(entry['path'])) == entry['sha256']
        if dest.exists():
            previous = read(dest)
            assert previous['sample_id'] == case['sample_id'] and previous['input_identity'] == input_identity
            continue
        records = {}
        for condition in ('c1', 'c3'):
            entry = indices[condition][case['sample_id']]
            assert sha256(Path(entry['path'])) == entry['sha256']
            records[condition] = load_cache(entry['path']) if entry['available'] else None
            assert records[condition] is None or records[condition]['labels'] is None
        predictions = []; cached = {}
        for name, selected in selections.items():
            condition = selected['condition']; record = records[condition]
            for budget in BUDGETS:
                pool = pipeline['pools'][condition][str(budget)]
                identity = (selected['source_checkpoint_path'], condition, budget)
                if identity not in cached:
                    if pool:
                        ix = record['pool_indices'][str(budget)]
                        assert record['points'][ix].tolist() == [p['point'] for p in pool]
                        if device.startswith('cuda'):
                            torch.cuda.synchronize()
                        start = time.perf_counter()
                        with torch.inference_mode():
                            out = models[identity[0]](**subset(record, ix, False, device))
                        if device.startswith('cuda'):
                            torch.cuda.synchronize()
                        elapsed = time.perf_counter()-start
                        scores = out['scores'].detach().float().cpu().numpy()
                        assert np.isfinite(scores).all()
                        pick = int(scores.argmax()); null = float(out['null_scores'].reshape(-1)[0])
                        cached[identity] = dict(point=pool[pick]['point'], scores=scores.tolist(), selected_index=pick,
                            null_score=null, null_would_reject=bool(null > scores[pick]),
                            elapsed_cached_head_seconds=elapsed, scoring_device=device)
                    else:
                        cached[identity] = dict(point=None, scores=[], failure='no_candidates')
                predictions.append(dict(sample_id=case['sample_id'], method=name, condition=condition,
                    budget=budget, **cached[identity]))
        predictions.append(dict(sample_id=case['sample_id'], method='global_top1', budget=None,
                                point=pipeline['global_result']['point']))
        native = read(native_path)
        predictions.append(dict(sample_id=case['sample_id'], method='qwen_native_completed_lax', budget=None,
                                point=native['completed_lax_point']))
        write_json(dest, dict(sample_id=case['sample_id'], predictions=predictions,
            pipeline_sha256=sha256(pipeline_path), input_identity=input_identity, ground_truth_in_model_inputs=False))
        print(json.dumps(dict(stage='heads', split=split, i=i, n=len(cases(split)))), flush=True)


def partition_positions(n, part, parts):
    if parts not in (1, 2) or not 0 <= part < parts:
        raise ValueError('Expected one or two disjoint case partitions')
    return range(part, n, parts)


def verifier_runtime_contract(metadata):
    if metadata is None:
        return None
    result = json.loads(json.dumps(metadata))
    # HF may report a set of nonpersistent rotary-buffer keys in different
    # iteration order across processes. Values and key multiplicities stay.
    if 'loading' in result and 'unexpected_keys' in result['loading']:
        result['loading']['unexpected_keys'] = sorted(result['loading']['unexpected_keys'])
    return result


def reference_workers(split, phase, execution_dir):
    processes = []
    try:
        for part in range(2):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(part), PYTHONUTF8='1')
            command = [sys.executable, '-X', 'utf8', '-u', str(Path(__file__).resolve()),
                       '--stage', 'reference', '--split', split, '--parts', '2', '--part', str(part),
                       '--reference-phase', phase, '--execution-dir', str(execution_dir)]
            processes.append(subprocess.Popen(command, env=env))
        while any(p.poll() is None for p in processes):
            assert all(p.poll() in (None, 0) for p in processes), 'A verifier partition failed'
            time.sleep(1)
        assert all(p.returncode == 0 for p in processes), 'A verifier partition failed'
    except BaseException:
        for p in processes:
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    p.kill(); p.wait(timeout=10)
        raise


def validate_parallel_acceptance(path, split):
    path = Path(path).resolve()
    assert path.is_relative_to((OUT/'gui_reference'/split/'executions').resolve())
    receipt = read(path)
    assert receipt['complete'] and receipt['split'] == split
    assert receipt['source_sha256'] == sha256(Path(__file__))
    assert receipt['frozen_sha256'] == sha256(OUT/'frozen.json')
    assert receipt['verifier_source_sha256'] == sha256(Path('scripts/run_gui_verifier.py'))
    assert receipt['verifier_asset_identity_sha256'] == sha256(Path('models/GUI-Actor-Verifier-2B/clickagent_download_verified.json'))
    assert receipt['accepted_before_case_workers'] and receipt['anchor_absolute_tolerance'] == 1e-5
    assert np.isfinite(receipt['anchor_max_abs_log_odds_difference']) and receipt['anchor_max_abs_log_odds_difference'] <= 1e-5
    for anchor_path, digest in receipt['anchor_file_sha256'].items():
        assert Path(anchor_path).resolve().is_relative_to(path.parent)
        assert sha256(Path(anchor_path)) == digest
    return receipt


def reference_parallel(split):
    """Approve fixed anchor batches before launching any actual-case worker."""
    folder = OUT/'gui_reference'/split
    execution_dir = (folder/'executions'/uuid.uuid4().hex).resolve()
    execution_dir.mkdir(parents=True)
    reference_workers(split, 'anchors', execution_dir)
    anchors = [read(execution_dir/f'anchor_part{part}.json') for part in range(2)]
    assert anchors[0]['specification'] == anchors[1]['specification']
    a, b = anchors[0]['scores'], anchors[1]['scores']
    assert len(a) == len(b)
    differences = []
    for left, right in zip(a, b):
        assert left['sample_id'] == right['sample_id'] and left['point'] == right['point']
        differences.append(abs(left['log_odds']-right['log_odds']))
    assert np.isfinite(differences).all(), 'Nonfinite verifier anchor scores'
    assert not differences or max(differences) <= 1e-5, 'GPU anchor score disagreement; do not silently accept a device shift'
    metadata = [read(execution_dir/f'runtime_anchors_part{part}.json') if (execution_dir/f'runtime_anchors_part{part}.json').exists() else None for part in range(2)]
    assert verifier_runtime_contract(metadata[0]) == verifier_runtime_contract(metadata[1])
    receipt_path = execution_dir/'acceptance.json'
    receipt = dict(complete=True, split=split, frozen_sha256=sha256(OUT/'frozen.json'),
        verifier_source_sha256=sha256(Path('scripts/run_gui_verifier.py')),
        verifier_asset_identity_sha256=sha256(Path('models/GUI-Actor-Verifier-2B/clickagent_download_verified.json')),
        physical_devices=[0, 1], runtime_contracts_equal=True, loading_key_order_canonicalized=True,
        runtime_contract=verifier_runtime_contract(metadata[0]),
        fixed_label_free_anchor_points=len(a),
        anchor_max_abs_log_odds_difference=max(differences, default=0.), anchor_absolute_tolerance=1e-5,
        anchor_file_sha256={str(execution_dir/f'anchor_part{part}.json'): sha256(execution_dir/f'anchor_part{part}.json') for part in range(2)},
        anchor_specification=anchors[0]['specification'], source_sha256=sha256(Path(__file__)),
        accepted_before_case_workers=True, changes_point_sets=False, changes_verifier_budget=False)
    write_json(receipt_path, receipt)
    reference_workers(split, 'cases', execution_dir)
    for part in range(2):
        runtime_path = execution_dir/f'runtime_cases_part{part}.json'
        if runtime_path.exists():
            assert verifier_runtime_contract(read(runtime_path)) == receipt['runtime_contract']
    if metadata[0] is not None:
        write_json(folder/'runtime.json', verifier_runtime_contract(metadata[0]))
    write_json(folder/'parallel_execution.json', dict(**receipt, case_partition='manifest index modulo2',
        acceptance_path=str(receipt_path), acceptance_sha256=sha256(receipt_path)))


def reference(split, part=0, parts=1, phase='cases', execution_dir=None):
    from run_gui_verifier import VerifierRuntime
    assert_sealed(); runtime = None
    if parts == 1 and torch.cuda.device_count() >= 2:
        reference_parallel(split)
        return
    cohort = cases(split)
    positions = set(partition_positions(len(cohort), part, parts))
    if parts == 2:
        assert execution_dir is not None, 'Partition workers require a parent execution directory'
        execution_dir = Path(execution_dir).resolve()
        assert execution_dir.is_relative_to((OUT/'gui_reference'/split/'executions').resolve())
        assert phase in ('anchors', 'cases')
        if phase == 'cases':
            validate_parallel_acceptance(execution_dir/'acceptance.json', split)
    def get_runtime():
        nonlocal runtime
        if runtime is None:
            runtime = VerifierRuntime()
            if parts == 2 and phase == 'cases':
                receipt = validate_parallel_acceptance(execution_dir/'acceptance.json', split)
                assert verifier_runtime_contract(runtime.metadata) == receipt['runtime_contract'], 'Case runtime differs from accepted anchor contract'
            path = (execution_dir/f'runtime_{phase}_part{part}.json') if parts == 2 else OUT/'gui_reference'/split/'runtime.json'
            write_json(path, runtime.metadata)
        return runtime
    if parts == 2 and phase == 'anchors':
        # Up to three first-batch anchors, selected solely from saved public
        # proposals. Both devices receive the identical actual batch shape.
        specification = []; anchor_scores = []
        for anchor_case in sorted(cohort, key=lambda c: key(c['sample_id'])):
            anchor_pipe = read(OUT/'pipeline'/split/(key(anchor_case['sample_id'])+'.json'))
            pts = [p['point'] for p in anchor_pipe['pools']['c1']['16'][:6]]
            if not pts:
                continue
            specification.append(dict(sample_id=anchor_case['sample_id'], points=pts,
                pipeline_sha256=sha256(OUT/'pipeline'/split/(key(anchor_case['sample_id'])+'.json'))))
            assert sha256(Path(anchor_case['image'])) == anchor_case['image_sha256']
            with Image.open(anchor_case['image']) as im:
                scored = get_runtime().score(im.convert('RGB'), anchor_case['instruction'], pts, 'crop')
            anchor_scores.extend(dict(sample_id=anchor_case['sample_id'], point=r['point'], log_odds=r['log_odds']) for r in scored)
            if len(specification) == 3:
                break
        write_json(execution_dir/f'anchor_part{part}.json', dict(specification=specification, scores=anchor_scores,
            ground_truth_used=False, selection='first three hash-ordered nonempty c1 pools, first up to six points'))
        return
    for position, case in enumerate(cohort):
        if position not in positions:
            continue
        i = position+1
        sid = key(case['sample_id']); dest = OUT/'gui_reference'/split/(sid+'.json')
        pipeline_path = OUT/'pipeline'/split/(sid+'.json'); pipeline = read(pipeline_path)
        head_path = OUT/'predictions'/split/(sid+'.json')
        input_identity = dict(frozen_sha256=sha256(OUT/'frozen.json'), pipeline_sha256=sha256(pipeline_path),
            heads_sha256=sha256(head_path), source_sha256=sha256(Path(__file__)),
            verifier_source_sha256=sha256(Path('scripts/run_gui_verifier.py')),
            verifier_asset_identity_sha256=sha256(Path('models/GUI-Actor-Verifier-2B/clickagent_download_verified.json')))
        if dest.exists():
            previous = read(dest)
            assert previous['sample_id'] == case['sample_id'] and previous['input_identity'] == input_identity
            if previous['execution_mode'] == 'parallel':
                acceptance_path = Path(previous['parallel_acceptance_path'])
                assert sha256(acceptance_path) == previous['parallel_acceptance_sha256']
                validate_parallel_acceptance(acceptance_path, split)
            else:
                assert previous['execution_mode'] == 'single_device'
            continue
        seen = set(); points = []
        for condition in ('c1', 'c3'):
            for p in pipeline['pools'][condition]['16']:
                point = tuple(p['point'])
                if point not in seen:
                    seen.add(point); points.append(list(point))
        records = {}; batches = []
        if points:
            active_runtime = get_runtime()
            assert sha256(Path(case['image'])) == case['image_sha256']
            with Image.open(case['image']) as im:
                image = im.convert('RGB')
            for start in range(0, len(points), 6):
                scores = active_runtime.score(image, case['instruction'], points[start:start+6], 'crop')
                records.update({coordinate_key(r['point']): r for r in scores})
                batches.append(dict(points=points[start:start+6], seconds=scores[0]['batch_elapsed_seconds']))
        predictions = []
        for condition in ('c1', 'c3'):
            for budget in (4, 16):
                predictions.append(dict(sample_id=case['sample_id'], method='gui_'+condition, condition=condition,
                    budget=budget, **choose_point(pipeline['pools'][condition][str(budget)], records)))
        head = read(head_path)
        for pred in head['predictions']:
            if pred['budget'] != 16:
                continue
            condition = pred['condition']; pool = pipeline['pools'][condition]['16']
            ix = np.argsort(-np.asarray(pred['scores']), kind='stable')[:4]
            shortlist = [pool[int(j)] for j in ix]
            predictions.append(dict(sample_id=case['sample_id'], method=pred['method']+'_top4_gui', condition=condition,
                budget=16, retained_points=[r['point'] for r in shortlist], n_verified=len(shortlist),
                **choose_point(shortlist, records)))
        execution = dict(execution_mode='parallel' if parts == 2 else 'single_device')
        if parts == 2:
            execution.update(parallel_acceptance_path=str(execution_dir/'acceptance.json'),
                             parallel_acceptance_sha256=sha256(execution_dir/'acceptance.json'))
        write_json(dest, dict(sample_id=case['sample_id'], records=records, predictions=predictions, batches=batches,
            **execution,
            input_identity=input_identity, ground_truth_in_model_inputs=False, cached_top4_counterfactual=True,
            timing_scope='actual union scoring; Top4 methods select cached scores, no fresh end-to-end Top4 timing'))
        print(json.dumps(dict(stage='gui', split=split, i=i, n=len(cases(split)), points=len(points))), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--stage', required=True, choices=('heads', 'reference'))
    p.add_argument('--split', choices=('test', 'supplement'), default='test')
    p.add_argument('--device', default='cpu')
    p.add_argument('--parts', type=int, choices=(1, 2), default=1)
    p.add_argument('--part', type=int, choices=(0, 1), default=0)
    p.add_argument('--reference-phase', choices=('anchors', 'cases'), default='cases')
    p.add_argument('--execution-dir', type=Path)
    args = p.parse_args(); torch.set_num_threads(4)
    heads(args.split, args.device) if args.stage == 'heads' else reference(args.split, args.part, args.parts, args.reference_phase, args.execution_dir)
