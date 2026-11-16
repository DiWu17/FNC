"""Separate target-region evidence coverage from exact point success.

A region is found when one actual per-token map's maximum coarse cell has
positive-area overlap with GT. Nonzero attention anywhere is NOT evidence.
This is a post-hoc candidate-coverage diagnostic, without fitting or decoding.
"""
import json
from pathlib import Path

import numpy as np

from clickagent.action_benchmark import sha256, write_json
from diagnose_phrase_token_oracle import overlap_mask
from diagnose_target_information import random_set_overlap_probability
from clickagent.spatial_geometry import shape_from_geometry
from run_target_token_experiment import read_jsonl


def main():
    out = Path('runs/phrase_v4'); dest = out/'evaluation/token_oracle'
    rows = read_jsonl(dest/'cases.jsonl')
    cases = {c['sample_id']: c for c in read_jsonl(out/'manifest.jsonl') if c['experiment_split'] == 'test'}
    families = ['raw_l15_token_forced_route', 'learned_token_forced_route', 'raw_head_any_token']
    result = dict(n=len(rows), named=sum(r['available'] for r in rows),
                  no_name=sum(not r['available'] for r in rows), methods={})
    native_rows = []; details = []
    for row in rows:
        case = cases[row['sample_id']]
        # Missing name still has a cached grid in the feature metadata; it
        # should not prevent scoring an otherwise available native point.
        if row['available']:
            grid = row['grid_shape']
        else:
            from run_target_token_experiment import key
            with np.load(out/'features'/f'{key(row["sample_id"])}.npz', allow_pickle=False) as z:
                grid = json.loads(str(z['metadata']))['grid_shape']
        gh, gw = grid; mask = overlap_mask(shape_from_geometry(case['target_geometry']), grid)
        point = row['native_point']; native_region = False
        if point is not None:
            w, h = case['image_size']
            col = min(gw-1, max(0, int(point[0]*gw/w)))
            rr = min(gh-1, max(0, int(point[1]*gh/h)))
            native_region = bool(mask[rr*gw+col])
        native_rows.append(native_region)
        detail = dict(sample_id=row['sample_id'], dataset=row['dataset'], native_point_hit=row['native_hit'],
                      native_parent_overlap=native_region, named=row['available'], methods={})
        for name in families:
            present = False; chance = 0.; count = 0
            if row['available']:
                parents = np.unique(row['families'][name]['parent_indices'])
                count = len(parents); present = bool(mask[parents].any())
                assert present == row['families'][name]['coarse_overlap']
                chance = random_set_overlap_probability(gh*gw, int(mask.sum()), count)
            detail['methods'][name] = dict(present=present, distinct_peak_cells=count,
                                           random_assignment_expected=chance)
        details.append(detail)
    native_hits = sum(r['native_hit'] for r in rows)
    result['native'] = dict(exact_point_hits=native_hits, exact_point_fraction=native_hits/len(rows),
        same_grid_parent_overlap=sum(native_rows), same_grid_parent_fraction=sum(native_rows)/len(rows))
    for name in families:
        present = [d['methods'][name]['present'] for d in details]
        cross = dict(both_region_present_and_native_point_correct=sum(p and r['native_hit'] for p, r in zip(present, rows)),
            region_present_native_point_wrong=sum(p and not r['native_hit'] for p, r in zip(present, rows)),
            region_not_found_native_point_correct=sum(not p and r['native_hit'] for p, r in zip(present, rows)),
            region_not_found_native_point_wrong=sum(not p and not r['native_hit'] for p, r in zip(present, rows)))
        assert sum(cross.values()) == len(rows)
        found = sum(present)
        result['methods'][name] = dict(region_present=found, region_fraction=found/len(rows),
            region_not_found=len(rows)-found, region_not_found_named=sum(r['available'] and not p for r, p in zip(rows, present)),
            random_assignment_expected=sum(d['methods'][name]['random_assignment_expected'] for d in details),
            difference_from_native_point_pp=100*(found-native_hits)/len(rows), cross_with_native_point=cross)
    result.update(definition='At least one actual name-token map peak coarse cell overlaps the annotated region with positive area; raw_head_any_token retains each of 96 heads from layers 15/28/36 separately.',
        limitations=[
            'Not found means no maximum cell in this saved family, not proof of absent information in all hidden states or in the full VLM.',
            'All-token/all-head coverage is GT-assisted candidate recall, whereas native point accuracy is one final decision.',
            'Positive-area coarse overlap is weaker than a correct pixel click. Native points are also scored at the same coarse grid as a separate comparison.',
            'Random assignment holds the number of distinct peak cells fixed and chooses their spatial cells uniformly without replacement; it is not a realistic GUI prior.',
            'Same 192 already examined test images; exploratory diagnosis only, no training or model selection.'],
        source_sha256=sha256(Path(__file__)), cases_sha256=sha256(dest/'cases.jsonl'),
        predictions_sha256=sha256(out/'evaluation/predictions.jsonl'))
    write_json(dest/'region_presence.json', result)
    (dest/'region_presence_cases.jsonl').write_text(''.join(json.dumps(d, ensure_ascii=False)+'\n' for d in details), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
