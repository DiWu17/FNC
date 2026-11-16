"""Post-inference V11 scoring; annotations are used only for final metrics."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path

import numpy as np

from clickagent.action_benchmark import sha256, write_json
from clickagent.candidate_budget import BUDGETS, build_pools, coordinate_key
from clickagent.public_evaluation import point_in_geometry
from candidate_scorer_protocol import OUT, VARIANTS
from point_aligned_data import read
from run_target_token_experiment import key, read_jsonl

EXPECTED_CASES = 192
PRIMARY = 'internal_full_s1_K16'
ROBUSTNESS = 'internal_full_s2_K16'
BOOTSTRAP_SEED = 20261131
BOOTSTRAP_REPS = 5000
OLD = Path('runs/context_verifier_v9')


def cases(split):
    path = OUT / ('diagnostic_manifest.jsonl' if split == 'diagnostic' else 'manifest.jsonl')
    rows = read_jsonl(path)
    if split == 'test': rows = [c for c in rows if c['experiment_split'] == 'test']
    assert len(rows) == EXPECTED_CASES and len({c['sample_id'] for c in rows}) == EXPECTED_CASES
    if split == 'diagnostic':
        assert all(c['diagnostic_reuse'] and c['experiment_split'] == 'diagnostic' for c in rows)
    return rows


def point_hit(point, case):
    if point is None: return False
    if not isinstance(point, (list, tuple)) or len(point) != 2: return False
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or int(v) != v for v in point):
        return False
    w, h = case['image_size']
    if not (0 <= point[0] < w and 0 <= point[1] < h): return False
    return bool(point_in_geometry([point[0] / w, point[1] / h], case['target_geometry']))


def expected_head_methods():
    return [f'{v}_{s}_K{k}' for v in VARIANTS for s in ('s1', 's2') for k in BUDGETS]


def _scores(value, n):
    scores = np.asarray(value, np.float64)
    assert scores.shape == (n,) and np.isfinite(scores).all()
    return scores


def validate_head_choice(output, pool):
    scores = _scores(output['scores'], len(pool))
    assert output['n_points'] == len(pool)
    if not pool:
        assert output['point'] is None and output['null_score'] is None
        return
    selected = int(scores.argmax())
    assert output['selected_index'] == selected and output['point'] == pool[selected]['point']
    null = output['null_score']
    assert isinstance(null, (int, float)) and math.isfinite(null)
    assert output['null_would_reject'] == bool(null > scores[selected])


def validate_gui_choice(choice, pool, records):
    if not pool:
        assert choice['point'] is None and choice['score'] is None
        return
    values = _scores([records[coordinate_key(v['point'])]['log_odds'] for v in pool], len(pool))
    selected = int(values.argmax()); item = pool[selected]
    assert choice['point'] == item['point'] and choice['score'] == float(values[selected])
    assert choice['source'] == item['source'] and choice['source_rank'] == item['source_rank']
    assert choice['exact_top_ties'] == int((values == values[selected]).sum())
    margin = float(np.sort(values)[-1] - np.sort(values)[-2]) if len(values) > 1 else None
    assert choice['margin'] == margin


def ui_predictions(split, cohort, identities):
    folder = OUT / 'ui_tars' if split == 'test' else OLD / 'ui_tars'
    paths = [folder / (key(c['sample_id']) + '.json') for c in cohort]
    if all(p.is_file() for p in paths):
        result = {}
        for case, path in zip(cohort, paths):
            value = read(path)
            assert value.get('sample_id', case['sample_id']) == case['sample_id']
            assert value.get('ground_truth_in_model_input') is False
            result[case['sample_id']] = value['point']; identities[str(path)] = sha256(path)
        return result
    assert split == 'diagnostic', 'All fresh UI-TARS predictions are required'
    # A whole-cohort legacy export is allowed, never a per-case silent fallback.
    path = OLD / 'evaluation/predictions.jsonl'
    rows = [r for r in read_jsonl(path) if r['method'] == 'ui_tars']
    assert len(rows) == len(cohort) and {r['sample_id'] for r in rows} == {c['sample_id'] for c in cohort}
    identities[str(path)] = sha256(path)
    return {r['sample_id']: r['point'] for r in rows}


def _duration(values):
    values = np.asarray(values, np.float64)
    assert np.isfinite(values).all() and (values >= 0).all()
    return dict(n=len(values), total_seconds=float(values.sum()), mean_seconds=float(values.mean()) if len(values) else None,
                median_seconds=float(np.median(values)) if len(values) else None,
                p95_seconds=float(np.quantile(values, .95)) if len(values) else None)


def collect(split, cohort):
    identities = {}; ui = ui_predictions(split, cohort, identities)
    details, predictions = [], []
    durations = {m: [] for m in expected_head_methods()}
    visual_seconds, global_seconds, local_seconds, native_seconds, gui_seconds = [], [], [], [], []
    gui_batches = 0
    for case in cohort:
        sid = case['sample_id']; stem = key(sid)
        pipeline_path = OUT / split / 'pipeline' / (stem + '.json')
        prediction_path = OUT / 'predictions' / split / (stem + '.json')
        gui_path = OUT / 'gui_reference' / split / (stem + '.json')
        native_path = (OLD if split == 'diagnostic' else OUT / split) / 'native' / (stem + '.json')
        for path in (pipeline_path, prediction_path, gui_path, native_path): identities[str(path)] = sha256(path)
        pipeline, saved, gui, native = map(read, (pipeline_path, prediction_path, gui_path, native_path))
        assert pipeline['sample_id'] == saved['sample_id'] == gui['sample_id'] == sid
        assert saved['ground_truth_in_model_inputs'] is False and gui['ground_truth_in_model_inputs'] is False
        assert pipeline['pools'] == build_pools(pipeline)
        expected = set(expected_head_methods()) | {'old_critic_K16', 'global_top1', 'qwen_native_completed_lax'}
        assert set(saved['methods']) == expected, ('Missing/extra head methods', sid)
        pools = {}
        for budget in BUDGETS:
            pool = pipeline['pools'][str(budget)]; points = [v['point'] for v in pool]
            assert len({tuple(p) for p in points}) == len(points)
            labels = [point_hit(p, case) for p in points]
            pools[str(budget)] = dict(points=points, point_hits=labels, point_oracle=any(labels), n_points=len(points))
        methods = {}
        for method, output in saved['methods'].items():
            budget = int(method.rsplit('_K', 1)[1]) if '_K' in method else None
            if method in durations:
                validate_head_choice(output, pipeline['pools'][str(budget)])
                seconds = output['elapsed_cached_head_seconds']; assert math.isfinite(seconds) and seconds >= 0
                durations[method].append(seconds)
            elif method == 'old_critic_K16':
                pool = pipeline['pools']['16']; values = _scores(output['scores'], len(pool))
                assert output['point'] == (pool[int(values.argmax())]['point'] if pool else None)
            elif method == 'global_top1': assert output['point'] == pipeline['global_result']['point']
            elif method == 'qwen_native_completed_lax': assert output['point'] == native['completed_lax_point']
            row = dict(point=output['point'], hit=point_hit(output['point'], case), pool_budget=budget)
            for field in ('scores', 'selected_index', 'null_score', 'elapsed_cached_head_seconds'):
                if field in output: row[field] = output[field]
            if budget is not None:
                row.update(pool_oracle=pools[str(budget)]['point_oracle'], n_points=pools[str(budget)]['n_points'])
                assert not row['hit'] or row['pool_oracle']
            if 'null_score' in output:
                rejected = bool(output.get('null_would_reject', False))
                optional = None if rejected else output['point']
                row.update(null_would_reject=rejected, optional_null_point=optional, optional_null_hit=point_hit(optional, case))
            methods[method] = row
        assert set(gui['choices']) == {'4', '16'}
        for budget in (4, 16):
            choice = gui['choices'][str(budget)]; validate_gui_choice(choice, pipeline['pools'][str(budget)], gui['records'])
            methods[f'gui_actor_K{budget}'] = dict(point=choice['point'], hit=point_hit(choice['point'], case), pool_budget=budget,
                pool_oracle=pools[str(budget)]['point_oracle'], n_points=pools[str(budget)]['n_points'])
        methods['ui_tars'] = dict(point=ui[sid], hit=point_hit(ui[sid], case), pool_budget=None)
        item = dict(sample_id=sid, dataset=case['dataset'], target_name=pipeline['global_result'].get('target_name', ''), pools=pools, methods=methods)
        details.append(item)
        predictions.extend(dict(sample_id=sid, dataset=case['dataset'], method=m, **v) for m, v in methods.items())
        for batch in gui['batches']:
            gui_seconds.append(float(batch['seconds'])); gui_batches += 1
        native_seconds.append(float(native['elapsed_seconds']))
        for name, dest in (('global_feature_path', global_seconds), ('local_feature_path', local_seconds)):
            if pipeline[name]:
                with np.load(pipeline[name], allow_pickle=False) as archive: metadata = json.loads(str(archive['metadata']))
                dest.append(float(metadata['elapsed_seconds']))
        encoding = OUT / 'vision_encodings' / (case['image_sha256'] + '.npz')
        if encoding.exists():
            with np.load(encoding, allow_pickle=False) as archive: metadata = json.loads(str(archive['metadata']))
            visual_seconds.append(float(metadata['timing_seconds']['total']))
    timing = dict(cached_heads={m: _duration(v) for m, v in durations.items()},
        gui_reference=dict(reused_scores=split == 'diagnostic', batch_calls=gui_batches,
                           measured_batches=_duration(gui_seconds), scope='fresh K16 common reference scoring; diagnostic reuses old K32 scores and has no new timing'),
        extra_visual_encoding=_duration(visual_seconds),
        frozen_qwen_extraction=dict(global_view=_duration(global_seconds), local_view=_duration(local_seconds), reused=split == 'diagnostic'),
        native=_duration(native_seconds),
        warning='Different cached stages and devices; these are not summed into measured end-to-end latency or a speedup claim.')
    return predictions, details, timing, identities


def bootstrap_accuracies(matrix, cohort, reps=BOOTSTRAP_REPS, seed=BOOTSTRAP_SEED):
    """Paired screenshot-cluster bootstrap, stratified by dataset."""
    matrix = np.asarray(matrix, np.float64)
    assert matrix.ndim == 2 and matrix.shape[0] == len(cohort) and reps > 0
    rng = np.random.default_rng(seed); numerator = np.zeros((reps, matrix.shape[1])); denominator = np.zeros(reps)
    for dataset in sorted({c['dataset'] for c in cohort}):
        groups = {}
        for i, case in enumerate(cohort):
            if case['dataset'] == dataset:
                image = case.get('pixel_sha256') or case.get('group_id') or case.get('image_sha256') or case['sample_id']
                groups.setdefault(str(image), []).append(i)
        ordered = [groups[k] for k in sorted(groups)]
        hits = np.stack([matrix[ix].sum(0) for ix in ordered]); sizes = np.asarray([len(ix) for ix in ordered])
        draws = rng.integers(0, len(ordered), (reps, len(ordered)))
        numerator += hits[draws].sum(1); denominator += sizes[draws].sum(1)
    return numerator / denominator[:, None]


def summarize(split, cohort, details, timing, identities, reps=BOOTSTRAP_REPS):
    names = sorted(details[0]['methods']); assert all(sorted(d['methods']) == names for d in details)
    matrix = np.asarray([[d['methods'][m]['hit'] for m in names] for d in details], bool)
    bootstrap = bootstrap_accuracies(matrix, cohort, reps); methods = {}
    for i, method in enumerate(names):
        rows = [d['methods'][method] for d in details]; budget = rows[0]['pool_budget']; hits = int(matrix[:, i].sum())
        available = sum(r['pool_oracle'] for r in rows) if budget else None
        value = dict(n=len(cohort), hits=hits, accuracy=hits / len(cohort), ci95=np.quantile(bootstrap[:, i], [.025, .975]).tolist(),
                     points=sum(r['point'] is not None for r in rows), pool_budget=budget, pool_available=available,
                     conditional_selection_accuracy=hits / available if available else None,
                     empty_pools=sum(r['n_points'] == 0 for r in rows) if budget else None,
                     nonempty_pool_without_correct_point=sum(r['n_points'] > 0 and not r['pool_oracle'] for r in rows) if budget else None,
                     selector_missed_correct_point=sum(r['pool_oracle'] and not r['hit'] for r in rows) if budget else None,
                     datasets={dataset: dict(n=sum(d['dataset'] == dataset for d in details),
                         hits=sum(d['methods'][method]['hit'] for d in details if d['dataset'] == dataset),
                         accuracy=float(np.mean([d['methods'][method]['hit'] for d in details if d['dataset'] == dataset])))
                         for dataset in sorted({c['dataset'] for c in cohort})})
        if 'optional_null_hit' in rows[0]:
            optional_hits = sum(r['optional_null_hit'] for r in rows)
            value['optional_null_policy'] = dict(hits=optional_hits, accuracy=optional_hits / len(cohort),
                learned_rejections=sum(r['null_would_reject'] for r in rows), points=sum(r['optional_null_point'] is not None for r in rows))
        if budget: assert hits + value['empty_pools'] + value['nonempty_pool_without_correct_point'] + value['selector_missed_correct_point'] == len(cohort)
        methods[method] = value
    pools = {}
    for budget in BUDGETS:
        rows = [d['pools'][str(budget)] for d in details]
        pools[str(budget)] = dict(n=len(cohort), oracle=sum(r['point_oracle'] for r in rows),
             empty=sum(r['n_points'] == 0 for r in rows), total_points=sum(r['n_points'] for r in rows),
             candidate_size_counts=dict(Counter(r['n_points'] for r in rows)))
    comparisons = []
    # Primary comparisons and prespecified budget/seed sensitivity; never select a winner for deployment.
    pairs = [(PRIMARY, reference) for reference in names if reference != PRIMARY]
    pairs += [(f'{v}_{s}_K{k}', f'{v}_{s}_K16') for v in VARIANTS for s in ('s1', 's2') for k in (4, 8, 32)]
    for method, reference in dict.fromkeys(pairs):
        a, b = names.index(method), names.index(reference); diff = matrix[:, a].astype(int) - matrix[:, b].astype(int)
        comparisons.append(dict(method=method, reference=reference, n=len(cohort), wins=int((diff > 0).sum()),
            losses=int((diff < 0).sum()), ties=int((diff == 0).sum()), delta=float(diff.mean()),
            ci95=np.quantile(bootstrap[:, a] - bootstrap[:, b], [.025, .975]).tolist()))
    return dict(complete=True, version=11, split=split, n=len(cohort), primary=PRIMARY, robustness=ROBUSTNESS,
        exploratory_reused_images=split == 'diagnostic', selection_used_this_cohort=False,
        methods=methods, pools=pools, paired_comparisons=comparisons, timing=timing,
        bootstrap=dict(reps=reps, seed=BOOTSTRAP_SEED, unit='RGB screenshot cluster', strata='dataset', interval='paired percentile 95%'),
        metric='Integer point inside any inclusive valid bbox/polygon; all192 denominator including missing names/empty pools; positive static localization, not measured UI execution or permission accuracy.',
        input_sha256=identities, protocol_sha256=sha256(OUT / 'protocol.json'), frozen_sha256=sha256(OUT / 'frozen.json'))


def evaluate_split(split, write=True, reps=BOOTSTRAP_REPS):
    cohort = cases(split); predictions, details, timing, identities = collect(split, cohort)
    summary = summarize(split, cohort, details, timing, identities, reps)
    if write:
        dest = OUT / 'evaluation' / split; dest.mkdir(parents=True, exist_ok=True)
        (dest / 'predictions.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in predictions), encoding='utf-8')
        write_json(dest / 'details.json', details); write_json(dest / 'summary.json', summary)
    return predictions, details, summary


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--split', choices=['test', 'diagnostic', 'both'], default='both')
    args = parser.parse_args()
    for split in ('test', 'diagnostic') if args.split == 'both' else (args.split,):
        _, _, summary = evaluate_split(split)
        print(json.dumps(dict(split=split, n=summary['n'], primary=summary['methods'][PRIMARY])), flush=True)


if __name__ == '__main__': main()
