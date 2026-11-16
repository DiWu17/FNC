"""Descriptive training-pool shift; never selects or modifies a model."""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from clickagent.action_benchmark import write_json
from clickagent.public_evaluation import point_in_geometry
from crossfit_protocol import OUT, ARMS, cases, key, read
from extract_crossfit_candidates import folder


def summary(rows):
    n = len(rows)
    covered = [r for r in rows if r['oracle16']]
    return dict(n=n, nonempty=sum(r['n16'] > 0 for r in rows), oracle16=len(covered),
        original_quota4_covers=sum(r['oracle4'] for r in rows), global_top1=sum(r['global_top1'] for r in rows),
        local_named=sum(r['local_named'] for r in rows),
        quota4_retention=sum(r['oracle4'] for r in covered)/len(covered) if covered else None,
        conditional_random_top1=float(np.mean([r['positive16']/r['n16'] for r in covered])) if covered else None,
        all_query_random_top1=float(np.mean([r['positive16']/r['n16'] if r['n16'] else 0 for r in rows])) if rows else None,
        mean_actual_points16=float(np.mean([r['n16'] for r in rows])) if rows else None)


def main():
    records = {a: [] for a in ARMS}
    for case in cases('train'):
        for arm in ARMS:
            p = read(folder(arm) / 'pipeline' / (key(case['sample_id']) + '.json'))
            def hit(point):
                return point is not None and point_in_geometry((np.asarray(point)/case['image_size']).tolist(), case['target_geometry'])
            pool = p['pools']['16']
            positives = sum(hit(r['point']) for r in pool)
            records[arm].append(dict(sample_id=case['sample_id'], source=case['dataset'], fold=case['image_fold'],
                proposer_role=p['proposer_role'], n16=len(pool), positive16=positives, oracle16=bool(positives),
                oracle4=any(hit(r['point']) for r in p['pools']['4']), global_top1=hit(p['global_result']['point']),
                local_named=bool(p.get('old_crop_result') and p['old_crop_result'].get('candidates'))))
    result = dict(complete=True, purpose='descriptive mechanism, no model selection', arms={})
    for arm, rows in records.items():
        by_source = defaultdict(list); by_fold = defaultdict(list); contributions = defaultdict(int)
        for r in rows:
            by_source[r['source']].append(r); by_fold[str(r['fold'])].append(r); contributions[r['proposer_role']] += 1
        result['arms'][arm] = dict(total=summary(rows), by_source={k: summary(v) for k,v in by_source.items()},
            by_image_fold={k: summary(v) for k,v in by_fold.items()}, proposer_contributions=dict(contributions))
    result['paired_coverage'] = dict(
        both=sum(a['oracle16'] and b['oracle16'] for a,b in zip(records['in_sample'],records['oof'])),
        in_sample_only=sum(a['oracle16'] and not b['oracle16'] for a,b in zip(records['in_sample'],records['oof'])),
        oof_only=sum(not a['oracle16'] and b['oracle16'] for a,b in zip(records['in_sample'],records['oof'])))
    models = {}
    for arm in ARMS:
        for suffix in ('s1','s2'):
            name = arm + '_' + suffix; meta = read(OUT / 'models' / (name + '.json'))
            models[name] = {k: meta[k] for k in ('selected','selected_training_evaluation','selected_validation_replay','initial_state_sha256')}
    result['models'] = models
    write_json(OUT / 'evaluation/training_pool_shift.json', result)
    write_json(OUT / 'evaluation/training_pool_details.json', records)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
