"""Post-hoc geometric diagnosis; never fits, selects or alters a predictor."""
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from shapely import affinity
from shapely.geometry import Point

from clickagent.spatial_geometry import shape_from_geometry


ROOT = Path('runs/semantic_head_v7')
PRIMARY = 'semantic_hard_s1__quality__combined'
REPLICA = 'semantic_hard_s2__quality__combined'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stats(values):
    a = np.asarray(values, float)
    return dict(n=len(a), median=float(np.median(a)), mean=float(a.mean()),
                p25=float(np.quantile(a, .25)), p75=float(np.quantile(a, .75))) if len(a) else dict(n=0)


def main():
    pred_path = ROOT/'evaluation/test/predictions.jsonl'
    methods = {PRIMARY, REPLICA, 'qwen_native_completed_lax',
               'semantic_hard_s1__point__rank', 'semantic_hard_s1__point__combined',
               'semantic_hard_s2__point__rank', 'semantic_hard_s2__point__combined'}
    preds = {name: {} for name in methods}
    for line in pred_path.open(encoding='utf-8'):
        r = json.loads(line)
        if r['method'] in preds:
            preds[r['method']][r['sample_id']] = r
    manifest = [json.loads(line) for line in (ROOT/'manifest.jsonl').open(encoding='utf-8')]
    cases = [r for r in manifest if r['experiment_split'] == 'test']
    assert len(cases) == 192 and all(len(r) == 192 for r in preds.values())
    records = []; archive_hashes = {}
    for c in cases:
        sid = c['sample_id']; p = preds[PRIMARY][sid]
        r = dict(sample_id=sid, dataset=c['dataset'], instruction=c['instruction'],
                 target_name=p.get('target_name'), stage=p['failure_stage'], hit=p['hit'],
                 native_hit=preds['qwen_native_completed_lax'][sid]['hit'],
                 parent_oracle=bool(p.get('candidate_region_oracle')),
                 point_oracle=bool(p.get('frozen_fine_point_oracle')))
        if p['point'] is None:
            r['representation_failure'] = 'no_name'
            records.append(r); continue
        name = hashlib.sha256(sid.encode()).hexdigest()[:24]+'.npz'
        cp = ROOT/'candidate_cache'/name; fp = ROOT/'point_cache'/name
        archive_hashes[str(cp)] = sha(cp); archive_hashes[str(fp)] = sha(fp)
        with np.load(cp, allow_pickle=False) as z:
            meta = json.loads(str(z['metadata']))
            area = z['label_area'].astype(float)
            parents = z['parent_indices'].astype(int)
        with np.load(fp, allow_pickle=False) as z:
            points = z['frozen_points']; features = z['frozen_point_features'].astype(float)
            point_hit = z['label_fixed_point_hit'].astype(bool)
        assert bool(point_hit.any()) == r['point_oracle']
        gh, gw = meta['grid_shape']; w, h = c['image_size']
        target = shape_from_geometry(c['target_geometry'])
        target_px = affinity.scale(target, xfact=w, yfact=h, origin=(0, 0))
        x0, y0, x1, y1 = target.bounds
        r['target_short_side_in_parent_cells'] = min((x1-x0)*gw, (y1-y0)*gh)
        r['target_short_side_after_resize_approx_px'] = 32*r['target_short_side_in_parent_cells']
        r['target_components'] = len(getattr(target, 'geoms', [target]))
        r['parent_cell_original_px'] = [w/gw, h/gh]
        r['n_candidates'] = len(parents)
        parent_positive = area.sum(-1) > 0
        assert bool(parent_positive.any()) == r['parent_oracle']
        # Quantized probability ties do not establish the FP32 argmax. Resolve
        # ties only when the stored integer point and offset make one child
        # compatible, allowing full one-FP16-step offset uncertainty + rounding.
        possible = features[:, :4] == features[:, :4].max(-1, keepdims=True)
        ties = int((possible.sum(-1) > 1).sum())
        rr, cc = np.divmod(parents, gw)
        for i in np.flatnonzero(possible.sum(-1) > 1):
            options = []
            for ch in np.flatnonzero(possible[i]):
                predicted = np.asarray([(2*cc[i]+ch%2+features[i, 4])*w/(2*gw),
                                        (2*rr[i]+ch//2+features[i, 5])*h/(2*gh)])
                errors = np.asarray([w/(2*gw), h/(2*gh)])*.001 + .501
                if np.all(np.abs(predicted-points[i]) <= errors):
                    options.append(ch)
            assert options, ('No compatible child for FP16 tie', sid, int(i))
            possible[i] = False; possible[i, options] = True
        child_positive = area > 0
        definite = np.all(child_positive | ~possible, axis=1)
        possible_positive = np.any(child_positive & possible, axis=1)
        r['child_probability_fp16_ties'] = ties
        r['child_ambiguities_after_point_check'] = int((possible.sum(-1) > 1).sum())
        r['selected_child_any_target_overlap_definite'] = bool(definite.any())
        r['selected_child_any_target_overlap_possible'] = bool(possible_positive.any())
        if not r['parent_oracle']:
            r['representation_failure'] = 'no_target_parent'
        elif r['point_oracle']:
            r['representation_failure'] = 'has_hittable_frozen_point'
        elif not possible_positive.any():
            r['representation_failure'] = 'all_chosen_children_miss'
        elif definite.any():
            r['representation_failure'] = 'chosen_child_available_but_offsets_miss'
        else:
            r['representation_failure'] = 'ambiguous_child_due_to_fp16'
        ci = int(p['selected_candidate_index'])
        r['selected_child_overlaps_target'] = bool(definite[ci]) if definite[ci] == possible_positive[ci] else None
        r['point_distance_to_target_original_px'] = float(target_px.distance(Point(p['point'])))
        r['point_distance_to_target_parent_diagonals'] = r['point_distance_to_target_original_px']/np.hypot(w/gw, h/gh)
        r['selected_quality'] = p.get('predicted_point_quality')
        r['selected_offsets'] = features[ci, 4:6].tolist()
        if parent_positive.any():
            weights = area.sum(-1); weights /= weights.sum()
            r['region_loss_target_mass_on_miss_points'] = float(weights[~point_hit].sum())
        top = p['candidates']
        r['frozen_point_hit_in_rank_top5'] = bool(any(point_hit[x['candidate_index']] for x in top))
        r['frozen_point_hit_in_rank_top3'] = bool(any(point_hit[x['candidate_index']] for x in top[:3]))
        records.append(r)
    summary = {}
    for ds in ['ALL']+sorted({r['dataset'] for r in records}):
        rows = [r for r in records if ds == 'ALL' or r['dataset'] == ds]
        named = [r for r in rows if 'selected_offsets' in r]
        covered = [r for r in named if r['parent_oracle']]
        reachable = [r for r in named if r['point_oracle']]
        fine = [r for r in named if r['stage'] == 'fixed_fine_miss_after_intersecting_parent']
        native_wins = [r for r in rows if r['native_hit'] and not r['hit']]
        summary[ds] = dict(n=len(rows), hits=sum(r['hit'] for r in rows),
            representation=Counter(r['representation_failure'] for r in rows),
            target_short_side_after_resize_approx_px=stats([r['target_short_side_after_resize_approx_px'] for r in named]),
            candidate_count=stats([r['n_candidates'] for r in named]),
            regional_supervision_mass_on_miss_points=stats([r['region_loss_target_mass_on_miss_points'] for r in covered]),
            reachable_regional_supervision_mass_on_miss_points=stats([r['region_loss_target_mass_on_miss_points'] for r in reachable]),
            regional_supervision_over_50pct_on_miss_points=sum(r['region_loss_target_mass_on_miss_points']>.5 for r in covered),
            fine_miss_distance_original_px=stats([r['point_distance_to_target_original_px'] for r in fine]),
            fine_miss_within_2px=sum(r['point_distance_to_target_original_px']<=2 for r in fine),
            fine_miss_within_5px=sum(r['point_distance_to_target_original_px']<=5 for r in fine),
            fine_miss_child_status=Counter(str(r['selected_child_overlaps_target']) for r in fine),
            top3_point_oracle=sum(r['frozen_point_hit_in_rank_top3'] for r in named),
            top5_point_oracle=sum(r['frozen_point_hit_in_rank_top5'] for r in named),
            native_wins=len(native_wins), native_wins_representation=Counter(r['representation_failure'] for r in native_wins),
            unresolved_child_cases=sum(r['child_ambiguities_after_point_check']>0 for r in named))
    quality_changes={}
    for seed in ['s1','s2']:
        rank=preds[f'semantic_hard_{seed}__point__rank']; combined=preds[f'semantic_hard_{seed}__point__combined']
        quality_changes[seed]=dict(rescued=sum(not rank[s]['hit'] and combined[s]['hit'] for s in rank),
            damaged=sum(rank[s]['hit'] and not combined[s]['hit'] for s in rank),
            selection_changed=sum(rank[s].get('selected_candidate_index')!=combined[s].get('selected_candidate_index') for s in rank))
    training = Counter(); train_mass = []; train_reachable_mass = []
    for c in manifest:
        if c['experiment_split'] != 'train':
            continue
        training['n'] += 1
        name = hashlib.sha256(c['sample_id'].encode()).hexdigest()[:24]+'.npz'
        cp = (ROOT if c.get('training_extension') else Path('runs/candidate_ranker_v5'))/'candidate_cache'/name
        fp = ROOT/'point_cache'/name
        archive_hashes[str(cp)] = sha(cp); archive_hashes[str(fp)] = sha(fp)
        with np.load(cp, allow_pickle=False) as z:
            meta = json.loads(str(z['metadata']))
            if not meta['target_available']:
                continue
            weights = z['label_area'].astype(float).sum(-1)
        with np.load(fp, allow_pickle=False) as z:
            hits = z['label_fixed_point_hit'].astype(bool)
        training['named'] += 1
        training['parent_oracle'] += bool(weights.sum() > 0)
        training['point_oracle'] += bool(hits.any())
        training['positive_parent_candidates'] += int((weights > 0).sum())
        training['positive_parent_candidates_with_miss_points'] += int(((weights > 0) & ~hits).sum())
        if weights.sum() > 0:
            weights /= weights.sum(); miss_mass = float(weights[~hits].sum())
            train_mass.append(miss_mass)
            training['parent_positive_but_no_hittable_point'] += not bool(hits.any())
            if hits.any(): train_reachable_mass.append(miss_mass)
    training = dict(training,scope='Full expanded training pool before stochastic candidate removal',
        regional_supervision_mass_on_miss_points=stats(train_mass),
        reachable_regional_supervision_mass_on_miss_points=stats(train_reachable_mass))
    assert sum(summary['ALL']['representation'].values()) == 192
    assert summary['ALL']['top5_point_oracle'] <= 94
    dest=ROOT/'evaluation/localization_diagnosis';dest.mkdir(exist_ok=True)
    result=dict(scope='Post-hoc existing fresh-test geometry and cache analysis; no fitting or new predictions',
        source_sha256=sha(__file__), manifest_sha256=sha(ROOT/'manifest.jsonl'), predictions_sha256=sha(pred_path),
        frozen_sha256=sha(ROOT/'frozen_readouts.json'), cache_sha256=archive_hashes,
        limitations=['Descriptive reuse of the already evaluated fresh test; not an independent new test.',
            'Child FP16 probability ties resolved only if point/offset compatibility is unique; ambiguities retained.',
            'Target short side uses the geometry envelope; 32 pixels per merged cell estimates resized scale.',
            'Distance outside GT is diagnostic, never changed into a relaxed success metric.',
            'Region-loss mass uses the frozen v5 point cache, not the newly trained auxiliary fine outputs.'],
        summary=summary, quality_changes=quality_changes, training_loss_conflict=training, records=records)
    (dest/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(summary=summary,quality_changes=quality_changes,training_loss_conflict=training),ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
