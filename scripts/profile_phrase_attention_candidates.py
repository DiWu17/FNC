"""Describe where GT-overlapping name-token/head peaks live.

Train-only head priors are frozen before validation/test are read. Everything
on the already-examined public test is exploratory diagnosis, not a new score.
No VLM execution, checkpoint fitting, or primary prediction mutation occurs.
"""
import csv
from collections import Counter, defaultdict
import json
from pathlib import Path
import re

import numpy as np
import shapely

from clickagent.action_benchmark import sha256, write_json
from clickagent.spatial_geometry import shape_from_geometry
from diagnose_target_information import random_set_overlap_probability
from run_target_token_experiment import key, read_jsonl


LAYERS = (15, 28, 36)
FUNCTION_WORDS = set('a an the of for in on at to from with and or by as is are be this that these those its your my their it into under over within next above below left right'.split())
FEATURE_NAMES = ['image_mass', 'relative_peak', 'relative_top2_gap', 'normalized_entropy',
                 'absolute_peak', 'token_relative_position', 'is_first', 'is_last',
                 'peak_x', 'peak_y', 'same_token_head_agreement', 'all_map_agreement',
                 'distinct_head_agreement', 'distinct_token_agreement']


def token_tags(pieces):
    """Lexical/boundary heuristics, NOT a linguistic POS tagger."""
    phrase = ''.join(pieces)
    words = [(m.start(), m.end(), m.group()) for m in re.finditer(r"\w+(?:['’]\w+)?", phrase, re.UNICODE)]
    result = []; cursor = 0
    for i, piece in enumerate(pieces):
        end = cursor+len(piece)
        overlapping = [(a, b, word) for a, b, word in words if a < end and b > cursor]
        names = [word.casefold() for _, _, word in overlapping]
        if not names:
            role = 'punctuation_or_space'
        elif all(word in FUNCTION_WORDS for word in names):
            role = 'function_or_relation_word'
        elif all(word.isdigit() for word in names):
            role = 'number'
        elif re.search(r'[\u3400-\u9fff]', piece):
            role = 'cjk_piece'
        elif any(a < cursor for a, _, _ in overlapping):
            role = 'content_word_continuation'
        else:
            role = 'content_word_start'
        result.append(dict(piece=piece, word=' / '.join(names) or piece.strip() or '<space>', role=role,
                           position='single' if len(pieces) == 1 else ('first' if i == 0 else 'last' if i == len(pieces)-1 else 'middle')))
        cursor = end
    return result


def head_label(index):
    return f'L{LAYERS[index//32]}/H{index % 32}'


def read_case(case, root):
    path = root/'features'/f'{key(case["sample_id"])}.npz'
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z['metadata'])); attention = z['target_attention'].astype(np.float32)
    assert meta['sample_id'] == case['sample_id'] and meta['own_token_forward'] and not meta['ground_truth_in_model_input']
    r = dict(case=case, metadata=meta, available=meta['target_available'])
    if not r['available']:
        assert len(attention) == 0
        return r
    t, heads, p = attention.shape; assert heads == 96
    gh, gw = meta['grid_shape']; assert gh*gw == p
    rr, cc = np.divmod(np.arange(p), gw)
    target = shape_from_geometry(case['target_geometry'])
    cells = shapely.box(cc/gw, rr/gh, (cc+1)/gw, (rr+1)/gh)
    mask = shapely.area(shapely.intersection(target, cells)) > 0
    w, h = case['image_size']
    centers = np.stack((np.clip(np.rint((cc+.5)*w/gw), 0, w-1), np.clip(np.rint((rr+.5)*h/gh), 0, h-1)), -1).astype(int)
    center_mask = shapely.intersects_xy(target, centers[:, 0]/w, centers[:, 1]/h)
    peaks = attention.argmax(-1); hit = mask[peaks]
    mass = attention.sum(-1); relative = attention/np.maximum(mass[..., None], 1e-12)
    peak = relative.max(-1)
    top2 = np.partition(relative, -2, axis=-1)[..., -2:]
    gap = top2.max(-1)-top2.min(-1)
    entropy = -(relative*np.log(np.maximum(relative, 1e-30))).sum(-1)/np.log(p)
    counts = np.bincount(peaks.ravel(), minlength=p)
    head_support = np.stack([np.bincount(peaks[:, hi], minlength=p) > 0 for hi in range(96)]).sum(0)
    token_support = np.stack([np.bincount(peaks[ti], minlength=p) > 0 for ti in range(t)]).sum(0)
    same_token = np.stack([np.bincount(pk, minlength=p)[pk]/96 for pk in peaks])
    relative_position = np.broadcast_to(np.linspace(0, 1, t)[:, None] if t > 1 else np.zeros((1, 1)), (t, 96))
    index = np.broadcast_to(np.arange(t)[:, None], (t, 96))
    x = ((peaks % gw)+.5)/gw; y = ((peaks//gw)+.5)/gh
    features = np.stack((mass, peak, gap, entropy, attention.max(-1), relative_position,
        index == 0, index == t-1, x, y, same_token, counts[peaks]/(t*96),
        head_support[peaks]/96, token_support[peaks]/t), -1).astype(np.float32)
    prefixes = meta['generated_prefixes']
    pieces = [prefixes[i-1][len(prefixes[i-2]) if i > 1 else 0:] for i in meta['target_raw_indices']]
    assert len(pieces) == t
    chance = np.array([random_set_overlap_probability(p, int(mask.sum()), len(np.unique(peaks[:, hi]))) for hi in range(96)])
    r.update(peaks=peaks, hit=hit, center_hit=center_mask[peaks], features=features,
        target_mask=mask, target_fraction=float(mask.mean()), head_chance=chance,
        centers=centers, center_mask=center_mask, tokens=token_tags(pieces), grid=[gh, gw], n_tokens=t)
    return r


def head_table(records):
    available = [r for r in records if r['available']]
    n = len(records); result = []
    for hi in range(96):
        coverage = sum(r['hit'][:, hi].any() for r in available)
        expected = sum(r['head_chance'][hi] for r in available)
        per_dataset = {}
        for ds in sorted({r['case']['dataset'] for r in records}):
            group = [r for r in available if r['case']['dataset'] == ds]
            total = sum(r['case']['dataset'] == ds for r in records)
            per_dataset[ds] = dict(n=total, region_hits=int(sum(r['hit'][:, hi].any() for r in group)),
                mean_map_excess=float(np.mean([r['hit'][:, hi].mean()-r['target_fraction'] for r in group])) if group else 0.)
        result.append(dict(head_index=hi, label=head_label(hi), layer=LAYERS[hi//32], head_zero_based=hi % 32,
            n=n, named=len(available), region_hits=int(coverage), region_recall=coverage/n,
            center_point_hits=int(sum(r['center_hit'][:, hi].any() for r in available)),
            map_trials=sum(r['n_tokens'] for r in available), map_hits=int(sum(r['hit'][:, hi].sum() for r in available)),
            case_balanced_map_hit_rate=float(np.mean([r['hit'][:, hi].mean() for r in available])),
            random_assignment_expected=float(expected), coverage_excess=float(coverage-expected),
            border_peak_fraction=float(np.mean([np.mean((r['peaks'][:, hi]//r['grid'][1] == 0) |
                (r['peaks'][:, hi]//r['grid'][1] == r['grid'][0]-1) | (r['peaks'][:, hi] % r['grid'][1] == 0) |
                (r['peaks'][:, hi] % r['grid'][1] == r['grid'][1]-1)) for r in available])),
            macro_map_excess=float(np.mean([v['mean_map_excess'] for v in per_dataset.values()])), datasets=per_dataset))
    return result


def selections(record, head_order, head_weights):
    if not record['available']:
        return {}
    f = record['features']; peaks = record['peaks']; t = record['n_tokens']; p = len(record['target_mask'])
    methods = {}
    for name, channel, descending in [('max_image_mass', 0, True), ('max_relative_peak', 1, True), ('min_entropy', 3, False), ('max_absolute_peak', 4, True)]:
        index = int(f[..., channel].argmax() if descending else f[..., channel].argmin())
        methods[name] = np.array([peaks.ravel()[index]])
    methods['train_best_head_last_token'] = np.array([peaks[-1, head_order[0]]])
    votes = np.bincount(peaks.ravel(), minlength=p).astype(float)
    weighted = np.zeros(p)
    distinct = np.zeros(p)
    for hi in range(96):
        weighted += np.bincount(peaks[:, hi], weights=np.full(t, head_weights[hi]/t), minlength=p)
        distinct[np.unique(peaks[:, hi])] += 1
    for name, score in [('uniform_peak_votes', votes), ('distinct_head_votes', distinct), ('train_weighted_peak_votes', weighted)]:
        # Only actual peak candidates are ranked. A zero-vote background cell
        # cannot become a proposal. Ties prefer lower raster index, without GT.
        candidates = np.unique(peaks)
        methods[name] = candidates[np.argsort(-score[candidates], kind='stable')[:5]]
    return methods


def summarize(records, order, head_weights):
    n = len(records); valid = [r for r in records if r['available']]
    heads = head_table(records); layers = {}
    for li, layer in enumerate(LAYERS):
        layer_heads = slice(li*32, (li+1)*32)
        layers[str(layer)] = dict(region_hits=int(sum(r['hit'][:, layer_heads].any() for r in valid)),
            center_point_hits=int(sum(r['center_hit'][:, layer_heads].any() for r in valid)),
            region_hits_first_token=int(sum(r['hit'][0, layer_heads].any() for r in valid)),
            map_hit_rate=float(sum(r['hit'][:, layer_heads].sum() for r in valid)/(32*sum(r['n_tokens'] for r in valid))))
    position_groups = defaultdict(list); role_groups = defaultdict(list); word_groups = defaultdict(list)
    for ri, r in enumerate(valid):
        for ti, tok in enumerate(r['tokens']):
            value = (ri, ti, r)
            position_groups[tok['position']].append(value)
            role_groups[tok['role']].append(value)
            word_groups[tok['word']].append(value)
    def group_stats(groups):
        result = {}
        for name, values in groups.items():
            by_image = defaultdict(list)
            for ri, ti, r in values:
                by_image[ri].append(ti)
            result[name] = dict(token_instances=len(values), images_available=len(by_image),
                map_trials=96*len(values), map_hits=int(sum(r['hit'][ti].sum() for _, ti, r in values)),
                map_hit_rate=float(np.mean([r['hit'][ti].mean() for _, ti, r in values])),
                images_with_any_hit=int(sum(valid[ri]['hit'][indices].any() for ri, indices in by_image.items())),
                image_recall_when_available=float(np.mean([valid[ri]['hit'][indices].any() for ri, indices in by_image.items()])))
        return result
    selected = defaultdict(lambda: dict(region_hits=Counter(), center_hits=Counter(), first_parent_border=0, first_parent_top_left=0))
    selector_rows = []
    for r in valid:
        for method, parents in selections(r, order, head_weights).items():
            row = dict(sample_id=r['case']['sample_id'], method=method, parents=parents.tolist(),
                points=r['centers'][parents].tolist(), region_hits={}, center_hits={},
                candidate_features_use_gt=False, head_weights_from_training_only=True)
            for k in (1, 3, 5):
                row['region_hits'][str(k)] = bool(r['target_mask'][parents[:k]].any())
                row['center_hits'][str(k)] = bool(r['center_mask'][parents[:k]].any())
                selected[method]['region_hits'][str(k)] += int(row['region_hits'][str(k)])
                selected[method]['center_hits'][str(k)] += int(row['center_hits'][str(k)])
            gh, gw = r['grid']; rr, cc = divmod(int(parents[0]), gw)
            row['first_parent_border'] = rr in (0, gh-1) or cc in (0, gw-1)
            row['first_parent_top_left'] = int(parents[0]) == 0
            selected[method]['first_parent_border'] += int(row['first_parent_border'])
            selected[method]['first_parent_top_left'] += int(row['first_parent_top_left'])
            selector_rows.append(row)
    for values in selected.values():
        values['n'] = n
    correct_heads = [int(r['hit'].any(0).sum()) for r in valid if r['hit'].any()]
    correct_tokens = [int(r['hit'].any(1).sum()) for r in valid if r['hit'].any()]
    features = np.concatenate([r['features'].reshape(-1, len(FEATURE_NAMES)) for r in valid])
    hit = np.concatenate([r['hit'].ravel() for r in valid])
    feature_stats = {name: dict(correct_peak_median=float(np.median(features[hit, i])),
                               incorrect_peak_median=float(np.median(features[~hit, i]))) for i, name in enumerate(FEATURE_NAMES)}
    coverage = {str(k): int(sum(r['hit'][:, order[:k]].any() for r in valid)) for k in (1, 4, 8, 16, 32, 96)}
    spatial = {}
    for label, mask in [('all', np.ones_like(hit)), ('correct', hit), ('incorrect', ~hit)]:
        xy = features[mask, 8:10]
        hist, _, _ = np.histogram2d(xy[:, 1], xy[:, 0], bins=[np.linspace(0, 1, 4)]*2)
        spatial[label] = (hist/hist.sum()).tolist()
    result = dict(n=n, named=len(valid), token_instances=sum(r['n_tokens'] for r in valid),
        head_token_trials=int(len(hit)), correct_peak_trials=int(hit.sum()), map_hit_rate=float(hit.mean()),
        zero_image_mass_maps=int((features[:, 0] <= 0).sum()),
        region_hits=int(sum(r['hit'].any() for r in valid)), layers=layers, heads=heads,
        positions=group_stats(position_groups), lexical_roles=group_stats(role_groups),
        recurring_words={k: v for k, v in group_stats(word_groups).items() if v['images_available'] >= 4},
        fixed_train_head_candidate_recall=coverage, label_free_selectors=dict(selected),
        correct_heads_per_found_case=dict(median=float(np.median(correct_heads)), q25_q75=np.quantile(correct_heads, [.25, .75]).tolist(),
            one_head_only=sum(x == 1 for x in correct_heads), at_most_four_heads=sum(x <= 4 for x in correct_heads)),
        correct_tokens_per_found_case=dict(median=float(np.median(correct_tokens)), one_token_only=sum(x == 1 for x in correct_tokens)),
        feature_medians=feature_stats, peak_spatial_thirds=spatial)
    return result, selector_rows


def csv_file(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def main():
    out = Path('runs/phrase_v4'); dest = out/'evaluation/head_token_profile'; dest.mkdir(exist_ok=True)
    cases = read_jsonl(out/'manifest.jsonl'); before = sha256(out/'evaluation/predictions.jsonl')
    records = {}; summaries = {}; all_features = []; all_labels = []; all_center_labels = []
    ids = []; splits = []; token_indices = []; head_indices = []; parents = []; labels_only = []
    order = None; head_weights = None
    for split in ('train', 'val', 'test'):
        group = [c for c in cases if c['experiment_split'] == split]
        root = out if split == 'test' else Path('runs/conditioned_v3')
        records[split] = []
        for i, case in enumerate(group):
            r = read_case(case, root); records[split].append(r)
            if r['available']:
                t = r['n_tokens']; all_features.append(r['features'].reshape(-1, len(FEATURE_NAMES)))
                all_labels.append(r['hit'].ravel()); all_center_labels.append(r['center_hit'].ravel())
                ids.extend([case['sample_id']]*(t*96)); splits.extend([split]*(t*96))
                token_indices.extend(np.repeat(np.arange(t), 96)); head_indices.extend(np.tile(np.arange(96), t)); parents.extend(r['peaks'].ravel())
                labels_only.append(dict(sample_id=case['sample_id'], split=split, pieces=r['tokens'],
                    correct_head_indices_by_token=[np.flatnonzero(x).tolist() for x in r['hit']],
                    parent_indices=r['peaks'].tolist()))
            if (i+1) % 100 == 0:
                print(f'{split}: {i+1}/{len(group)}', flush=True)
        if split == 'train':
            heads = head_table(records[split])
            order = sorted(range(96), key=lambda hi: (-heads[hi]['macro_map_excess'], hi))
            head_weights = np.array([max(0., h['macro_map_excess']) for h in heads])
            assert head_weights.sum() > 0
            write_json(dest/'train_only_head_prior.json', dict(ranking=order, labels=[head_label(h) for h in order],
                weights=head_weights.tolist(), criterion='Training-only macro dataset mean of per-image fraction of token peaks overlapping GT minus fraction of GT-overlapping grid cells.',
                candidate_budget_fixed=[1, 4, 8, 16, 32, 96], validation_or_test_used=False,
                frozen_before_loading_validation_and_test=True, source_sha256=sha256(Path(__file__))))
        summaries[split], selector_rows = summarize(records[split], order, head_weights)
        (dest/f'{split}_selector_predictions.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in selector_rows), encoding='utf-8')
        print(f'{split} complete: {summaries[split]["region_hits"]}/{len(group)} region coverage', flush=True)
    np.savez_compressed(dest/'candidate_features.npz', features=np.concatenate(all_features),
        feature_names=np.array(FEATURE_NAMES), labels_peak_parent_overlap=np.concatenate(all_labels),
        labels_peak_center_hit=np.concatenate(all_center_labels), sample_id=np.array(ids), split=np.array(splits),
        token_index=np.array(token_indices, np.int16), head_index=np.array(head_indices, np.int16), parent_index=np.array(parents, np.int16))
    (dest/'candidate_cases.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in labels_only), encoding='utf-8')
    assert before == sha256(out/'evaluation/predictions.jsonl')
    result = dict(splits=summaries, train_head_ranking=[head_label(h) for h in order],
        warnings=['Already examined test set: exploratory analysis, not independent performance evidence.',
            'All head/token union counts are GT-assisted candidate recall. Candidate selectors choose without test GT, but use an already examined benchmark.',
            'Heads are zero-based within one-based language layers 15,28,36. Only these cached layers are studied.',
            'Token role tags are lexical/boundary heuristics, not manually validated POS tags. Subwords share their reconstructed lexical word.',
            'Per-map trials are correlated within images, heads, and adjacent tokens. Descriptive rates are not independent-sample significance tests.',
            'Uniform random spatial assignments do not model realistic GUI priors.',
            'Saved label arrays are supervision/diagnosis only; all selector input features are computed without GT.'],
        feature_names=FEATURE_NAMES, source_sha256=sha256(Path(__file__)),
        primary_predictions_sha256=before, primary_predictions_unchanged=True)
    write_json(dest/'summary.json', result)
    csv_file(dest/'head_stats.csv', [dict(split=split, **{k: v for k, v in row.items() if k != 'datasets'})
        for split, summ in summaries.items() for row in summ['heads']])
    csv_file(dest/'token_role_stats.csv', [dict(split=split, grouping=kind, role=name, **value)
        for split, summ in summaries.items() for kind in ('positions', 'lexical_roles') for name, value in summ[kind].items()])
    print(json.dumps(dict(test_top_heads=sorted(summaries['test']['heads'], key=lambda x: -x['coverage_excess'])[:12],
        test_layers=summaries['test']['layers'], test_roles=summaries['test']['lexical_roles'],
        test_selectors=summaries['test']['label_free_selectors'], test_head_budget=summaries['test']['fixed_train_head_candidate_recall']), ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
