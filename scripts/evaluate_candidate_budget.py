"""Common-cache budget curves; GT is used only after the sealed model inference."""
from collections import Counter, defaultdict
import json
import math

import numpy as np

from clickagent.action_benchmark import write_json
from clickagent.candidate_budget import BUDGETS, choose_point, coordinate_key, ranked_points
from clickagent.public_evaluation import point_in_geometry
from evaluate_phrase_readouts import paired_interval
from point_aligned_data import read
from run_candidate_budget import BASE, OUT, assert_seal, cases
from run_target_token_experiment import key, read_jsonl


def hit(point, case):
    return bool(point is not None and point_in_geometry(
        [point[0] / case['image_size'][0], point[1] / case['image_size'][1]], case['target_geometry']))


def evaluate():
    rows = cases()
    sealed = {r['sample_id']: r for r in read(OUT / 'sealed_pools.json')}
    old_predictions = defaultdict(dict)
    for r in read_jsonl(BASE / 'evaluation/predictions.jsonl'):
        old_predictions[r['sample_id']][r['method']] = r
    predictions = [];details = [];actual_batches = [];drift = []
    full_oracle = 0
    for c in rows:
        sid = c['sample_id'];saved = read(OUT / 'verifier' / f'{key(sid)}.json')
        assert saved['complete']
        records = saved['records'];pools = sealed[sid]['pools'];case_details = {}
        old = read(BASE / 'gui_verifier' / f'{key(sid)}.json')
        old_records = {coordinate_key(v['point']): v for v in old['scores']['crop']}
        for item in pools['4']:
            k = coordinate_key(item['point'])
            drift.append(abs(records[k]['log_odds'] - old_records[k]['log_odds']))
        for budget in BUDGETS:
            pool = pools[str(budget)];chosen = choose_point(pool, records)
            labels = [hit(v['point'], c) for v in pool]
            pixels = sum(int(np.prod(records[coordinate_key(v['point'])]['image_grid_thw'])) * 14 * 14 for v in pool)
            result = dict(sample_id=sid, dataset=c['dataset'], method=f'K{budget}',
                hit=hit(chosen['point'], c), oracle=any(labels), n_points=len(pool),
                unpadded_processed_pixels=pixels, independent_batch_call_budget=math.ceil(len(pool) / 6), **chosen)
            predictions.append(result)
            case_details[str(budget)] = dict(**result, pool=pool, point_hits=labels,
                candidate_log_odds=[records[coordinate_key(v['point'])]['log_odds'] for v in pool])
        legacy = choose_point(pools['4'], old_records)
        assert legacy['point'] == old_predictions[sid]['fixed_pool__gui_actor_crop']['point']
        predictions.append(dict(sample_id=sid, dataset=c['dataset'], method='legacy_K4',
            point=legacy['point'], hit=hit(legacy['point'], c), oracle=case_details['4']['oracle']))
        for method in ['qwen_native_completed_lax', 'ui_tars']:
            old_row = old_predictions[sid][method]
            predictions.append(dict(sample_id=sid, dataset=c['dataset'], method=method,
                point=old_row['point'], hit=hit(old_row['point'], c), oracle=None))
        pipeline = read(BASE / 'pipeline' / f'{key(sid)}.json')
        all_points = ranked_points(pipeline['global_result'], 'global') + ranked_points(
            pipeline.get('old_crop_result'), 'old_crop', pipeline.get('old_crop_transform'))
        full_available = any(hit(v['point'], c) for v in all_points)
        full_oracle += full_available
        case_details['full_decoded_point_oracle'] = full_available
        details.append(dict(sample_id=sid, dataset=c['dataset'], budgets=case_details))
        actual_batches.extend(saved['batches'])
    methods = {}
    for method in [f'K{k}' for k in BUDGETS] + ['legacy_K4', 'qwen_native_completed_lax', 'ui_tars']:
        group = [r for r in predictions if r['method'] == method]
        available = sum(r['oracle'] for r in group) if group[0]['oracle'] is not None else None
        hits = sum(r['hit'] for r in group)
        result = dict(n=len(group), hits=hits, accuracy=hits / len(group), points=sum(r['point'] is not None for r in group),
            available=available, conditional_selection_accuracy=hits / available if available else None,
            datasets={d:dict(n=sum(r['dataset'] == d for r in group), hits=sum(r['hit'] for r in group if r['dataset'] == d))
                      for d in sorted({r['dataset'] for r in group})})
        if method.startswith('K'):
            result.update(empty_pools=sum(r['n_points'] == 0 for r in group),
                nonempty_pool_without_correct_point=sum(r['n_points'] > 0 and not r['oracle'] for r in group),
                selector_missed_correct_point=sum(r['oracle'] and not r['hit'] for r in group),
                mean_candidate_views=float(np.mean([r['n_points'] for r in group])),
                total_candidate_views=sum(r['n_points'] for r in group),
                mean_independent_batch_call_budget=float(np.mean([r['independent_batch_call_budget'] for r in group])),
                mean_unpadded_processed_pixels=float(np.mean([r['unpadded_processed_pixels'] for r in group])),
                candidate_size_counts=dict(Counter(r['n_points'] for r in group)),
                exact_top_tie_cases=sum(r['exact_top_ties'] > 1 for r in group))
        methods[method] = result
    assert methods['legacy_K4']['hits'] == read(BASE / 'evaluation/summary.json')['methods']['fixed_pool__gui_actor_crop']['hits'] == 73
    assert full_oracle == 125
    transitions = {}
    for a, b in [(4, 8), (8, 16), (16, 32), (4, 32)]:
        counts = Counter()
        for d in details:
            before = d['budgets'][str(a)];after = d['budgets'][str(b)]
            assert not before['oracle'] or after['oracle']
            counts['newly_covered'] += after['oracle'] and not before['oracle']
            counts['rescues'] += after['hit'] and not before['hit']
            counts['harms'] += before['hit'] and not after['hit']
            counts['rescues_from_new_coverage'] += after['hit'] and not before['oracle']
            counts['newly_covered_still_missed'] += after['oracle'] and not before['oracle'] and not after['hit']
            counts['harm_new_winner_not_in_old_pool'] += before['hit'] and not after['hit'] and after['point'] not in [v['point'] for v in before['pool']]
        transitions[f'{a}_to_{b}'] = dict(counts)
    legacy_disagreements = []
    for d in details:
        sid = d['sample_id'];fresh = d['budgets']['4'];old = old_predictions[sid]['fixed_pool__gui_actor_crop']
        if fresh['point'] != old['point']:
            legacy_disagreements.append(dict(sample_id=sid, old_point=old['point'], fresh_point=fresh['point'],
                old_hit=old['hit'], fresh_hit=fresh['hit']))
    pairs = [paired_interval(predictions, f'K{k}', 'K4', rows) for k in BUDGETS if k != 4]
    for k in BUDGETS:
        for baseline in ['qwen_native_completed_lax', 'ui_tars']:
            pairs.append(paired_interval(predictions, f'K{k}', baseline, rows))
    summary = dict(complete=True, n=192, exploratory_reused_images=True, methods=methods,
        full_decoded_point_oracle=full_oracle, transitions=transitions, paired_comparisons=pairs,
        legacy_score_drift=dict(points_checked=len(drift), max_abs_log_odds=float(max(drift)) if drift else 0,
            mean_abs_log_odds=float(np.mean(drift)) if drift else 0, point_disagreements=legacy_disagreements,
            explanation='Fresh common-cache scores may differ numerically from v9 batches. K comparisons all share identical fresh per-point scores.'),
        actual_common_cache_compute=dict(batch_calls=len(actual_batches),
            image_views=sum(len(v['points']) for v in actual_batches),
            sum_synchronized_model_batch_seconds=sum(v['seconds'] for v in actual_batches),
            median_batch_seconds=float(np.median([v['seconds'] for v in actual_batches])) if actual_batches else None,
            note='One shared K32 score cache; per-K budgets are workload accounting, not independent measured end-to-end latency.'),
        metric='point inside any valid region; all192 in denominator; no calibrated abstention or negative-state benchmark',
        warning='Already analyzed v9 screenshots. No training or new K chosen from test; larger-pool accuracy need not rise with oracle coverage.')
    return predictions, details, summary


def main():
    assert_seal()
    assert read(OUT / 'completion.json')['complete']
    predictions, details, summary = evaluate()
    dest = OUT / 'evaluation';dest.mkdir(exist_ok=True)
    (dest / 'predictions.jsonl').write_text(''.join(json.dumps(v, ensure_ascii=False) + '\n' for v in predictions), encoding='utf-8')
    write_json(dest / 'details.json', details);write_json(dest / 'summary.json', summary)
    print(json.dumps({k:v['hits'] for k,v in summary['methods'].items()}), flush=True)


if __name__ == '__main__':main()
