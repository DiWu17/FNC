"""Post-test inspection of cached name-token attention; never fit a readout.

Oracle counts use ground truth to choose a token/head and are NOT achievable
accuracy. All counts concern positive-area coarse-cell overlap, not clicks.
The random-set expectation preserves the number of distinct peak cells but
randomizes their spatial assignment; it is not a realistic GUI baseline.
"""
import json
import math
from pathlib import Path

import numpy as np
from shapely.geometry import box

from clickagent.action_benchmark import sha256, write_json
from clickagent.spatial_geometry import shape_from_geometry
from run_target_token_experiment import key, read_jsonl


def random_set_overlap_probability(p, m, n):
    if n > p-m:
        return 1.0
    log_miss = sum(math.log1p(-m/(p-i)) for i in range(n))
    return -math.expm1(log_miss)


def summarize(rows):
    available = [r for r in rows if r['available']]
    result = {'n': len(rows), 'named': len(available)}
    if not available:
        return result
    for name in available[0]['peak_sets']:
        result[name] = {
            'coarse_overlap': sum(r['peak_sets'][name]['hit'] for r in available),
            'random_assignment_expected': sum(r['peak_sets'][name]['random_expected'] for r in available),
            'median_distinct_peak_cells': float(np.median([r['peak_sets'][name]['n_unique'] for r in available])),
        }
    multiple = [r for r in available if r['n_tokens'] > 1]
    result['pooling'] = {
        'n_multiple_tokens': len(multiple),
        'first_has_highest_weight_all': sum(r['first_weight_is_largest'] for r in available),
        'first_has_highest_weight_multiple': sum(r['first_weight_is_largest'] for r in multiple),
        'median_first_weight_multiple': float(np.median([r['first_weight'] for r in multiple])) if multiple else None,
        'mean_first_weight_multiple': float(np.mean([r['first_weight'] for r in multiple])) if multiple else None,
        'mean_last_weight_multiple': float(np.mean([r['last_weight'] for r in multiple])) if multiple else None,
        'median_n_tokens': float(np.median([r['n_tokens'] for r in available])),
    }
    result['last_l15_mean_rescues_first'] = sum(r['peak_sets']['last_l15_mean']['hit'] and not r['peak_sets']['first_l15_mean']['hit'] for r in multiple)
    result['first_l15_mean_rescues_last'] = sum(r['peak_sets']['first_l15_mean']['hit'] and not r['peak_sets']['last_l15_mean']['hit'] for r in multiple)
    result['full_coarse_overlap'] = sum(r['full_coarse_overlap'] for r in available)
    return result


def main():
    out = Path('runs/conditioned_v3')
    cases = [c for c in read_jsonl(out/'manifest.jsonl') if c['experiment_split'] == 'test']
    predictions = read_jsonl(out/'evaluation/all_predictions.jsonl')
    full = {r['sample_id']: r for r in predictions if r['method'] == 'full'}
    native = {r['sample_id']: r for r in predictions if r['method'] == 'qwen_native_completed_lax'}
    rows = []
    for c in cases:
        sid = c['sample_id']; f = full[sid]; n = native[sid]
        with np.load(out/'features'/f'{key(sid)}.npz', allow_pickle=False) as z:
            meta = json.loads(str(z['metadata']))
            a = z['target_attention'].astype(np.float64)
        row = {'sample_id': sid, 'dataset': c['dataset'], 'instruction': c['instruction'],
               'target_name': meta['target_text'], 'available': meta['target_available'],
               'full_hit': f['hit'], 'native_hit': n['hit'], 'full_point': f['point'], 'native_point': n['point']}
        if not row['available']:
            rows.append(row)
            continue
        t, h, p = a.shape; gh, gw = meta['grid_shape']
        assert h == 96 and gh*gw == p and len(f['target_token_weights']) == t
        target = shape_from_geometry(c['target_geometry'])
        overlap = np.array([target.intersection(box(col/gw, rr/gh, (col+1)/gw, (rr+1)/gh)).area > 0
                            for rr in range(gh) for col in range(gw)])
        weights = np.asarray(f['target_token_weights'], dtype=np.float64)
        pooled = np.einsum('t,thp->hp', weights, a)
        peaks = {
            'first_l15_mean': [a[0, :32].mean(0).argmax()],
            'last_l15_mean': [a[-1, :32].mean(0).argmax()],
            'pooled_l15_mean': [pooled[:32].mean(0).argmax()],
            'any_token_l15_mean_oracle': a[:, :32].mean(1).argmax(-1),
            'any_head_first_oracle': a[0].argmax(-1),
            'any_head_last_oracle': a[-1].argmax(-1),
            'any_head_pooled_oracle': pooled.argmax(-1),
            'any_token_any_head_oracle': a.argmax(-1).ravel(),
        }
        peak_sets = {}
        for name, values in peaks.items():
            indices = np.unique(values)
            peak_sets[name] = {'hit': bool(overlap[indices].any()), 'n_unique': len(indices),
                              'random_expected': random_set_overlap_probability(p, int(overlap.sum()), len(indices))}
        prefixes = meta['generated_prefixes']
        pieces = [prefixes[i-1][len(prefixes[i-2]) if i > 1 else 0:] for i in meta['target_raw_indices']]
        token_peaks = a[:, :32].mean(1).argmax(-1)
        row.update(n_tokens=t, pieces=pieces, first_weight=float(weights[0]), last_weight=float(weights[-1]),
                   first_weight_is_largest=int(weights.argmax()) == 0,
                   full_coarse_overlap=f['selected_coarse_cell_overlap'], peak_sets=peak_sets,
                   l15_mean_peak_by_token=[{'piece': piece, 'peak_cell': int(index), 'coarse_overlap': bool(overlap[index])}
                                           for piece, index in zip(pieces, token_peaks)])
        rows.append(row)
    groups = {'all': rows,
              'native_correct_full_wrong': [r for r in rows if r['native_hit'] and not r['full_hit']],
              'native_correct_full_wrong_coarse': [r for r in rows if r['native_hit'] and not r['full_hit'] and not r.get('full_coarse_overlap', False)]}
    result = {'groups': {k: summarize(v) for k, v in groups.items()},
              'source_sha256': sha256(Path(__file__)), 'prediction_sha256': sha256(out/'evaluation/all_predictions.jsonl'),
              'posthoc_diagnostic_only': True, 'no_weights_or_primary_predictions_modified': True,
              'warning': 'Ground-truth token/head selection is an oracle, not model accuracy. Coarse-cell overlap is not a correct click. Random expectations randomize spatial assignments with the same unique-peak count; they do not model GUI priors.'}
    write_json(out/'evaluation/target_information_diagnostics.json', result)
    (out/'evaluation/target_information_cases.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
