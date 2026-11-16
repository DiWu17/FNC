"""Frozen V12 heads and same-pool references; scoring receives public inputs."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from clickagent.action_benchmark import sha256, write_json
from clickagent.candidate_budget import coordinate_key, choose_point
from clickagent.internal_candidate_scorer import load_internal_scorer
from train_candidate_scorer import load_cache, subset
from run_gui_verifier import VerifierRuntime
from extract_crossfit_candidates import folder
from crossfit_protocol import OUT, ARMS, BUDGETS, cases, key, read, rows, assert_sealed


def heads(split):
    assert_sealed()
    models = {a + '_' + s: load_internal_scorer(OUT / 'models' / (a + '_' + s + '.pt'), device='cpu')[0].eval()
              for a in ARMS for s in ('s1', 's2')}
    index = read(OUT / 'cache' / (split + '_index.json')); assert index['complete']
    rows = {r['sample_id']: r for r in index['cases']}
    for i, original in enumerate(cases(split), 1):
        sid = key(original['sample_id']); dest = OUT / 'predictions' / split / (sid + '.json')
        if dest.exists():
            continue
        # The remaining annotation fields never enter a scoring call.
        sample_id = original['sample_id']; row = rows[sample_id]
        pipeline = read(folder(split) / 'pipeline' / (sid + '.json'))
        record = None
        if row['available']:
            assert sha256(Path(row['path'])) == row['sha256']
            record = load_cache(Path(row['path']))
        result = []
        for name, model in models.items():
            for budget in BUDGETS:
                pool = pipeline['pools'][str(budget)]
                if pool:
                    ix = record['pool_indices'][str(budget)]
                    assert record['points'][ix].tolist() == [v['point'] for v in pool]
                    started = time.perf_counter()
                    with torch.inference_mode():
                        output = model(**subset(record, ix, False, 'cpu'))
                    elapsed = time.perf_counter()-started
                    scores = output['scores'].detach().float().numpy()
                    assert len(scores) == len(pool) and np.isfinite(scores).all()
                    selected = int(scores.argmax())
                    null = float(output['null_scores'].reshape(-1)[0])
                    values = dict(point=pool[selected]['point'], scores=scores.tolist(), selected_index=selected,
                        null_score=null, null_would_reject=bool(null > scores[selected]),
                        elapsed_cached_head_seconds=elapsed)
                else:
                    values = dict(point=None, scores=[], failure='no_candidates')
                result.append(dict(sample_id=sample_id, method=name, budget=budget, **values))
        result.append(dict(sample_id=sample_id, method='global_top1', budget=None, point=pipeline['global_result']['point']))
        if split == 'test':
            native = read(folder(split) / 'native' / (sid + '.json'))
            result.append(dict(sample_id=sample_id, method='qwen_native_completed_lax', budget=None, point=native['completed_lax_point']))
        write_json(dest, dict(sample_id=sample_id, predictions=result, ground_truth_in_model_inputs=False))
        print(json.dumps(dict(stage='heads', i=i, n=len(rows))), flush=True)


def reference(split):
    assert_sealed(); runtime = None
    for i, original in enumerate(cases(split), 1):
        public = {k: original[k] for k in ('sample_id', 'image', 'image_sha256', 'instruction')}
        sid = key(public['sample_id']); dest = OUT / 'gui_reference' / split / (sid + '.json')
        if dest.exists():
            continue
        pipeline = read(folder(split) / 'pipeline' / (sid + '.json')); pool = pipeline['pools']['16']
        records = {}; batches = []
        if pool:
            if runtime is None:
                runtime = VerifierRuntime()
                write_json(OUT / 'gui_reference' / split / 'runtime.json', runtime.metadata)
            assert sha256(Path(public['image'])) == public['image_sha256']
            with Image.open(public['image']) as im:
                image = im.convert('RGB')
            for start in range(0, len(pool), 6):
                points = [v['point'] for v in pool[start:start+6]]
                scores = runtime.score(image, public['instruction'], points, 'crop')
                records.update({coordinate_key(r['point']): r for r in scores})
                batches.append(dict(points=points, seconds=scores[0]['batch_elapsed_seconds']))
        predictions = [dict(sample_id=public['sample_id'], method='gui_actor', budget=k,
                            **choose_point(pipeline['pools'][str(k)], records)) for k in (4, 16)]
        head = read(OUT / 'predictions' / split / (sid + '.json'))
        for p in head['predictions']:
            if p['budget'] != 16 or p['method'] not in [a + '_' + s for a in ARMS for s in ('s1', 's2')]:
                continue
            order = np.argsort(-np.asarray(p['scores']), kind='stable')[:4]
            shortlist = [pool[int(j)] for j in order]
            predictions.append(dict(sample_id=public['sample_id'], method=p['method'] + '_top4_gui', budget=16,
                retained_points=[v['point'] for v in shortlist], n_verified=len(shortlist),
                **choose_point(shortlist, records)))
        write_json(dest, dict(sample_id=public['sample_id'], records=records, predictions=predictions,
            batches=batches, ground_truth_in_model_inputs=False, cached_top4_counterfactual=True))
        print(json.dumps(dict(stage='gui', i=i, n=len(cases(split)))), flush=True)


def export(split):
    predictions = []; pools = []
    for case in cases(split):
        sid = key(case['sample_id'])
        predictions.extend(read(OUT / 'predictions' / split / (sid + '.json'))['predictions'])
        gui = OUT / 'gui_reference' / split / (sid + '.json')
        if gui.exists():
            predictions.extend(read(gui)['predictions'])
        pipe = read(folder(split) / 'pipeline' / (sid + '.json'))
        pools.append(dict(sample_id=case['sample_id'], pools=pipe['pools']))
    for name, values in (('predictions', predictions), ('pools', pools)):
        path = OUT / 'evaluation' / split / (name + '.jsonl'); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(''.join(json.dumps(v, ensure_ascii=False) + '\n' for v in values), encoding='utf-8')
    print(json.dumps(dict(stage='export', split=split, predictions=len(predictions), pools=len(pools))), flush=True)


def migration():
    """Already-analyzed V11 test on its exact old V8 pools: secondary only."""
    assert_sealed()
    old = Path('runs/candidate_scorer_v11')
    cohort = [c for c in rows(old / 'manifest.jsonl') if c['experiment_split'] == 'test']
    assert len(cohort) == 192
    models = {a + '_' + s: load_internal_scorer(OUT / 'models' / (a + '_' + s + '.pt'), device='cpu')[0].eval()
              for a in ARMS for s in ('s1', 's2')}
    index = {r['sample_id']: r for r in read(old / 'cache/test_index.json')['cases']}
    all_predictions = []; all_pools = []; evidence = {}
    for i, original in enumerate(cohort, 1):
        sample_id = original['sample_id']; sid = key(sample_id)
        pp = old / 'test/pipeline' / (sid + '.json'); pipeline = read(pp)
        cp = Path(index[sample_id]['path']); assert sha256(cp) == index[sample_id]['sha256']
        record = load_cache(cp)
        assert record['labels'] is None
        predictions = []
        for name, model in models.items():
            for budget in BUDGETS:
                pool = pipeline['pools'][str(budget)]; scores = []
                if pool:
                    ix = record['pool_indices'][str(budget)]
                    assert record['points'][ix].tolist() == [v['point'] for v in pool]
                    with torch.inference_mode():
                        output = model(**subset(record, ix, False, 'cpu'))
                    scores = output['scores'].detach().float().numpy().tolist()
                predictions.append(dict(sample_id=sample_id, method=name, budget=budget,
                    point=pool[int(np.argmax(scores))]['point'] if pool else None, scores=scores))
        gui_path = old / 'gui_reference/test' / (sid + '.json'); gui = read(gui_path)
        for budget in (4, 16):
            predictions.append(dict(sample_id=sample_id, method='gui_actor', budget=budget,
                **choose_point(pipeline['pools'][str(budget)], gui['records'])))
        native_path = old / 'test/native' / (sid + '.json')
        predictions.append(dict(sample_id=sample_id, method='qwen_native_completed_lax', budget=None,
            point=read(native_path)['completed_lax_point']))
        for p in list(predictions):
            if p['budget'] == 16 and p['method'] in models:
                pool = pipeline['pools']['16']
                keep = [pool[int(j)] for j in np.argsort(-np.asarray(p['scores']), kind='stable')[:4]]
                predictions.append(dict(sample_id=sample_id, method=p['method'] + '_top4_gui', budget=16,
                    retained_points=[v['point'] for v in keep], n_verified=len(keep), **choose_point(keep, gui['records'])))
        all_predictions.extend(predictions); all_pools.append(dict(sample_id=sample_id, pools=pipeline['pools']))
        for path in (pp, cp, gui_path, native_path):
            evidence[str(path)] = sha256(path)
        print(json.dumps(dict(stage='reused_migration', i=i, n=192)), flush=True)
    dest = OUT / 'evaluation/migration'; dest.mkdir(parents=True, exist_ok=True)
    for name, values in [('cases',cohort), ('predictions',all_predictions), ('pools',all_pools)]:
        (dest / (name + '.jsonl')).write_text(''.join(json.dumps(v, ensure_ascii=False)+'\n' for v in values), encoding='utf-8')
    write_json(dest / 'identity.json', dict(complete=True, reused_already_analyzed=True, primary=False,
        cohort='old V11 192, old V8 pools and raw feature inputs shared by both new scorer arms',
        evidence=evidence, ground_truth_in_model_inputs=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--stage', choices=['heads', 'reference', 'export', 'migration'], required=True)
    p.add_argument('--split', choices=['val', 'test'], default='test')
    args = p.parse_args(); torch.set_num_threads(4)
    if args.stage == 'migration':
        migration()
    else:
        {'heads': heads, 'reference': reference, 'export': export}[args.stage](args.split)
