"""Sealed V11 inference. No target annotation is read by a scoring function."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from PIL import Image

from clickagent.action_benchmark import sha256, write_json
from clickagent.candidate_budget import choose_point, coordinate_key
from clickagent.internal_candidate_scorer import load_internal_scorer
from clickagent.point_critic import FrozenPointEncoder, load_point_critic
from clickagent.visual_candidate_verifier import load_visual_verifier
from candidate_scorer_protocol import OUT, VARIANTS
from cache_candidate_scorer import raw_arrays
from extract_candidate_scorer import cases, root_for, assert_test_sealed
from point_aligned_data import read
from point_aligned_predict import load_predictors
from run_gui_verifier import VerifierRuntime
from run_target_token_experiment import key
from train_candidate_scorer import load_cache, subset


def load_heads():
    heads = {}
    for variant in VARIANTS:
        for suffix in ('s1', 's2'):
            name = variant + '_' + suffix
            loader = load_visual_verifier if variant == 'visual' else load_internal_scorer
            model, _ = loader(OUT / 'models' / (name + '.pt'), device='cpu')
            heads[name] = model.eval()
    return heads


def heads(split):
    assert_test_sealed(); models = load_heads()
    index = read(OUT / 'cache' / (split + '_index.json')); assert index['complete'] and index['visual_complete']
    decoder = load_predictors(primary_only=True)['joint_connected_s1'][0]
    old_encoder = FrozenPointEncoder(decoder)
    old_critic, _ = load_point_critic('runs/context_verifier_v9/models/point_critic_s1.pt')
    folder = OUT / 'predictions' / split; folder.mkdir(parents=True, exist_ok=True)
    rows = {r['sample_id']: r for r in index['cases']}
    for i, original in enumerate(cases(split), 1):
        # Labels are not passed to data builders or any model.
        case = {k: original[k] for k in ('sample_id', 'image_size', 'dataset')}
        sid = key(case['sample_id']); dest = folder / (sid + '.json')
        if dest.exists():
            continue
        pipeline = read(root_for(split) / 'pipeline' / (sid + '.json')); row = rows[case['sample_id']]
        result = dict(sample_id=case['sample_id'], methods={}, ground_truth_in_model_inputs=False)
        if row['available']:
            assert sha256(Path(row['path'])) == row['sha256']
            assert sha256(Path(row['visual_path'])) == row['visual_sha256']
            data = {False: load_cache(Path(row['path']), False), True: load_cache(Path(row['visual_path']), True)}
        else:
            data = {}
        for name, model in models.items():
            visual = name.startswith('visual_')
            for budget in (4, 8, 16, 32):
                method = name + '_K' + str(budget)
                pool = pipeline['pools'][str(budget)]
                if not pool:
                    result['methods'][method] = dict(point=None, scores=[], null_score=None, n_points=0,
                                                       elapsed_cached_head_seconds=0., failure='no_candidates')
                    continue
                record = data[visual]; indices = record['pool_indices'][str(budget)]
                assert record['points'][indices].tolist() == [v['point'] for v in pool]
                started = time.perf_counter()
                with torch.inference_mode():
                    output = model(**subset(record, indices, visual, 'cpu'))
                elapsed = time.perf_counter()-started
                scores = output['scores'].detach().float().cpu().numpy()
                assert scores.shape == (len(pool),) and np.isfinite(scores).all()
                choice = int(scores.argmax()); null = float(output['null_scores'].reshape(-1)[0])
                result['methods'][method] = dict(point=pool[choice]['point'], selected_index=choice,
                    scores=scores.tolist(), null_score=null, null_would_reject=bool(null > scores[choice]),
                    n_points=len(pool), elapsed_cached_head_seconds=elapsed,
                    score_device='cpu', score_scope='tensor conversion and frozen small head; excludes cached Qwen/extra vision extraction')
        pool = pipeline['pools']['16']
        if pool:
            raw, meta = raw_arrays(pipeline['global_feature_path'])
            state = old_encoder.prepare(raw, meta['grid_shape'], case['image_size'])
            features = old_encoder.features(state, [v['point'] for v in pool]).astype(np.float16).astype(np.float32)
            with torch.inference_mode():
                scores = old_critic(torch.from_numpy(features)).numpy()
            result['methods']['old_critic_K16'] = dict(point=pool[int(scores.argmax())]['point'], scores=scores.tolist())
            del raw, state, features
        else:
            result['methods']['old_critic_K16'] = dict(point=None, scores=[])
        result['methods']['global_top1'] = dict(point=pipeline['global_result']['point'])
        native_root = Path('runs/context_verifier_v9') if split == 'diagnostic' else root_for(split)
        native = read(native_root / 'native' / (sid + '.json'))
        result['methods']['qwen_native_completed_lax'] = dict(point=native['completed_lax_point'])
        write_json(dest, result)
        print(json.dumps(dict(stage='head_inference', split=split, i=i, n=len(rows), sample_id=case['sample_id'])), flush=True)
    write_json(folder / 'completion.json', dict(complete=all((folder / (key(c['sample_id']) + '.json')).exists() for c in cases(split)),
               n=len(rows), frozen_sha256=sha256(OUT / 'frozen.json')))


def reference(split):
    assert_test_sealed(); folder = OUT / 'gui_reference' / split; folder.mkdir(parents=True, exist_ok=True)
    runtime = None
    for i, original in enumerate(cases(split), 1):
        case = {k: original[k] for k in ('sample_id', 'image', 'image_sha256', 'instruction')}
        sid = key(case['sample_id']); dest = folder / (sid + '.json')
        if dest.exists():
            continue
        pipeline = read(root_for(split) / 'pipeline' / (sid + '.json')); pool = pipeline['pools']['16']
        if split == 'diagnostic':
            cache = read(Path('runs/candidate_budget_v10/verifier') / (sid + '.json'))
            assert cache['complete']; records = cache['records']; batches = []
        else:
            records = {}; batches = []
            if pool:
                if runtime is None:
                    runtime = VerifierRuntime()
                    write_json(folder / 'runtime.json', runtime.metadata)
                assert sha256(Path(case['image'])) == case['image_sha256']
                with Image.open(case['image']) as opened:
                    image = opened.convert('RGB')
                for start in range(0, len(pool), 6):
                    points = [v['point'] for v in pool[start:start+6]]
                    scores = runtime.score(image, case['instruction'], points, 'crop')
                    records.update({coordinate_key(r['point']): r for r in scores})
                    batches.append(dict(points=points, seconds=scores[0]['batch_elapsed_seconds']))
        choices = {str(k): choose_point(pipeline['pools'][str(k)], records) for k in (4, 16)}
        write_json(dest, dict(sample_id=case['sample_id'], records=records, choices=choices, batches=batches,
                   reused_v10_scores=split == 'diagnostic', ground_truth_in_model_inputs=False))
        print(json.dumps(dict(stage='gui_reference', split=split, i=i, candidates=len(pool))), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--stage', choices=['heads', 'reference'], required=True)
    p.add_argument('--split', choices=['test', 'diagnostic'], required=True)
    args = p.parse_args(); torch.set_num_threads(4)
    heads(args.split) if args.stage == 'heads' else reference(args.split)
