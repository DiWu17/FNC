"""Post-hoc token/head existence audit, with exact clicks and spatial nulls.

Nothing is fitted. Ground truth chooses candidates only AFTER every candidate
has been decoded. These are diagnostic oracles, never deployment accuracies.
The published model, features, and primary predictions are left untouched.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import shapely
import torch
import torch.nn.functional as F

from clickagent.action_benchmark import sha256, write_json
from clickagent.conditioned_grounder import make_batch
from clickagent.phrase_grounder import decode_phrase, load_phrase, split_children
from clickagent.spatial_geometry import shape_from_geometry
from run_target_token_experiment import read_jsonl
from train_conditioned_readouts import load_records


def point_field(children, offsets, image_size):
    """One decoded integer-pixel point for every possible parent, per token."""
    t, gh, gw, _ = children.shape
    child = children.argmax(-1).reshape(t, -1)
    rr, cc = np.divmod(np.arange(gh*gw), gw)
    row = 2*rr[None]+child//2; col = 2*cc[None]+child % 2
    ii = np.arange(t)[:, None]
    dx = offsets[ii, 0, row, col]; dy = offsets[ii, 1, row, col]
    w, h = image_size
    return np.stack((np.clip(np.rint((col+dx)*w/(2*gw)), 0, w-1),
                     np.clip(np.rint((row+dy)*h/(2*gh)), 0, h-1)), -1).astype(int)


def inside(points, target, image_size):
    w, h = image_size
    return shapely.intersects_xy(target, points[..., 0]/w, points[..., 1]/h)


def overlap_mask(target, grid):
    gh, gw = grid; rr, cc = np.divmod(np.arange(gh*gw), gw)
    cells = shapely.box(cc/gw, rr/gh, (cc+1)/gw, (rr+1)/gh)
    return shapely.area(shapely.intersection(target, cells)) > 0


def shifted_indices(indices, shifts, grid):
    """Translate the WHOLE candidate set together on a toroidal coarse grid."""
    gh, gw = grid; rr, cc = np.divmod(indices, gw)
    return ((rr[None]+shifts[:, 0, None]) % gh)*gw+(cc[None]+shifts[:, 1, None]) % gw


def summarize(rows, family_names):
    result = dict(n=len(rows), named=sum(r['available'] for r in rows),
                  current_hits=sum(r['current_hit'] for r in rows),
                  native_hits=sum(r['native_hit'] for r in rows), families={})
    for name in family_names:
        valid = [r['families'][name] for r in rows if r['available']]
        null = np.asarray([r['null_hits'] for r in valid], float).sum(0) if valid else np.zeros(256)
        observed = sum(r['hit'] for r in valid)
        result['families'][name] = dict(hits=observed, accuracy=observed/len(rows) if rows else None,
            coarse_overlap=sum(r['coarse_overlap'] for r in valid),
            current_misses_rescued=sum(not r['current_hit'] and r['families'][name]['hit'] for r in rows if r['available']),
            current_union_hits=sum(r['current_hit'] or (r['available'] and r['families'][name]['hit']) for r in rows),
            mean_candidates=float(np.mean([r['n_candidates'] for r in valid])) if valid else None,
            median_distinct_points=float(np.median([r['n_unique_points'] for r in valid])) if valid else None,
            shift_null_mean_hits=float(null.mean()), shift_null_95pct_hits=np.quantile(null, [.025, .975]).tolist(),
            shift_null_p_ge_observed=float((1+(null >= observed).sum())/(len(null)+1)))
    return result


@torch.inference_mode()
def diagnose(record, model, current, native, rng):
    c = record['case']; meta = record['metadata']; grid = meta['grid_shape']; gh, gw = grid; p = gh*gw
    row = dict(sample_id=c['sample_id'], dataset=c['dataset'], instruction=c['instruction'],
               target_name=meta['target_text'], available=meta['target_available'],
               current_hit=current['hit'], native_hit=native['hit'], current_point=current['point'],
               native_point=native['point'], current_failure_stage=current.get('failure_stage'))
    if not meta['target_available']:
        return row
    inputs, _ = make_batch([record], next(model.parameters()).device, labels=False)
    output = model(**inputs)
    decoded = decode_phrase(output, 0, c['image_size'], top_k=1)[0]
    assert decoded['point'] == current['point'], (c['sample_id'], decoded, current)
    a = inputs['attention'].float(); _, t, heads, _ = a.shape
    maps = torch.log1p(p*a/a.sum(-1, keepdim=True).clamp_min(1e-12))
    evidence = torch.einsum('bth,bthp->btp', output['head_weights'], maps)[0].cpu().numpy()
    q = F.layer_norm(inputs['query_hidden'].float().reshape(1, t, 3, -1), [4096]).reshape(1, t, -1)
    q = model.query(q)[0]
    # Keep the final complete-phrase state fixed. Select token t globally for
    # both its spatial evidence and its contribution to fine decoding.
    query = q+q[-1:]
    fine_query = query[:, None].expand(t, 4*p, -1)
    vision = model.vision(F.layer_norm(inputs['vision_hidden'].float(), [model.config.vision_dim]))
    vision = vision.expand(t, -1, -1)
    fine = model.fine(torch.cat((vision, fine_query, vision*fine_query), -1))
    forced_children = split_children(fine[..., 0].reshape(t, gh*2, gw*2)).cpu().numpy()
    forced_offsets = fine[..., 1:].sigmoid().transpose(1, 2).reshape(t, 2, gh*2, gw*2).cpu().numpy()
    forced_field = point_field(forced_children, forced_offsets, c['image_size'])
    original_field = point_field(output['child_log_prob'].cpu().numpy(), output['offsets'].cpu().numpy(), c['image_size'])
    original_field = np.repeat(original_field, t, axis=0)
    assert original_field[0, decoded['coarse_index']].tolist() == decoded['point']
    # Stronger control: each token is the ONLY supplied name state, including
    # replacing complete-phrase context. This intervention is out of training
    # distribution; failure here cannot establish absence of information.
    single_inputs = dict(inputs, attention=inputs['attention'][0, :, None],
        query_hidden=inputs['query_hidden'][0, :, None], token_mask=torch.ones((t, 1), device=a.device, dtype=torch.bool),
        image_hidden=inputs['image_hidden'].expand(t, -1, -1),
        vision_hidden=inputs['vision_hidden'].expand(t, -1, -1))
    single = model(**single_inputs)
    single_scores = single['coarse_logits'].reshape(t, p).cpu().numpy()
    single_field = point_field(single['child_log_prob'].cpu().numpy(), single['offsets'].cpu().numpy(), c['image_size'])
    if t == 1:
        torch.testing.assert_close(single['coarse_logits'], output['coarse_logits'])
        np.testing.assert_array_equal(forced_field, original_field)
    raw = a[0].cpu().numpy()
    peaks = evidence.argmax(-1); raw_peaks = raw.argmax(-1)
    l15 = raw[:, :32].mean(1); l15_peaks = l15.argmax(-1)
    rr, cc = np.divmod(np.arange(p), gw); w, h = c['image_size']
    centers = np.stack((np.clip(np.rint((cc+.5)*w/gw), 0, w-1),
                        np.clip(np.rint((rr+.5)*h/gh), 0, h-1)), -1).astype(int)
    center_field = np.repeat(centers[None], t, axis=0)
    tokens = np.arange(t); token_heads = np.repeat(tokens, heads)
    families = {
        'learned_token_original_fine': (peaks, tokens, original_field),
        'learned_token_forced_route': (peaks, tokens, forced_field),
        'single_token_entire_state': (single_scores.argmax(-1), tokens, single_field),
        'learned_token_coarse_center': (peaks, tokens, center_field),
        'raw_l15_token_forced_route': (l15_peaks, tokens, forced_field),
        'raw_l15_token_coarse_center': (l15_peaks, tokens, center_field),
        'raw_head_first_token': (raw_peaks[0], np.zeros(heads, int), forced_field),
        'raw_head_last_token': (raw_peaks[-1], np.full(heads, t-1), forced_field),
        'raw_head_any_token': (raw_peaks.ravel(), token_heads, forced_field),
        'raw_head_any_token_coarse_center': (raw_peaks.ravel(), token_heads, center_field),
    }
    target = shape_from_geometry(c['target_geometry']); mask = overlap_mask(target, grid)
    possible = np.arange(1, p)  # exclude identity; common whole-set translation
    shifts_flat = rng.choice(possible, size=256, replace=len(possible) < 256)
    shifts = np.stack(np.divmod(shifts_flat, gw), -1)
    row['families'] = {}
    for name, (parents, ti, field) in families.items():
        points = field[ti, parents]; good = inside(points, target, c['image_size'])
        field_hits = inside(field, target, c['image_size'])
        translated = shifted_indices(parents, shifts, grid)
        null_hits = field_hits[ti[None], translated].any(-1)
        chosen = np.flatnonzero(good)
        row['families'][name] = dict(hit=bool(good.any()), coarse_overlap=bool(mask[parents].any()),
            n_candidates=len(parents), n_unique_points=len(np.unique(points, axis=0)),
            n_unique_parents=len(np.unique(parents)), null_hits=null_hits.tolist(),
            hit_candidate_indices=chosen.tolist(), points=points.tolist(), parent_indices=parents.tolist(),
            token_indices=ti.tolist())
    prefixes = meta['generated_prefixes']
    pieces = [prefixes[i-1][len(prefixes[i-2]) if i > 1 else 0:] for i in meta['target_raw_indices']]
    weights = output['token_weights'][0].cpu().numpy()
    # Point correctness and cell overlap are deliberately separate fields.
    token_rows = []
    target_indices = np.flatnonzero(mask)
    for i in range(t):
        scores = evidence[i]
        best_target_rank = 1+int((scores > scores[target_indices].max()).sum())
        token_rows.append(dict(index=i, piece=pieces[i], learned_peak_parent=int(peaks[i]),
            learned_peak_overlap=bool(mask[peaks[i]]), best_target_cell_rank=best_target_rank,
            learned_forced_point=forced_field[i, peaks[i]].tolist(),
            learned_forced_hit=bool(inside(forced_field[i, peaks[i]], target, c['image_size'])),
            raw_l15_point=forced_field[i, l15_peaks[i]].tolist(),
            raw_l15_hit=bool(inside(forced_field[i, l15_peaks[i]], target, c['image_size'])),
            weight_at_final_parent=float(weights[i, decoded['coarse_index']]),
            image_attention_mass_mean=float(raw[i].sum(-1).mean())))
    row.update(n_tokens=t, tokens=token_rows, grid_shape=grid,
               final_parent=decoded['coarse_index'], target_parent_count=int(mask.sum()),
               target_area=float(target.area), best_target_cell_rank_over_tokens=min(x['best_target_cell_rank'] for x in token_rows))
    # A disjoint, explicitly limited error decomposition of the existing output.
    if row['current_hit']:
        stage = 'already_correct'
    elif row['families']['learned_token_forced_route']['hit']:
        stage = 'correct_point_exists_with_token_route'
    elif row['families']['learned_token_forced_route']['coarse_overlap']:
        stage = 'token_has_target_parent_but_fine_decode_misses'
    elif row['families']['raw_head_any_token']['hit']:
        stage = 'only_raw_head_peak_decodes_correctly'
    elif row['families']['raw_head_any_token']['coarse_overlap']:
        stage = 'only_raw_head_target_parent_not_exact_point'
    else:
        stage = 'no_target_peak_in_saved_token_head_family'
    row['diagnostic_stage'] = stage
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, default=Path('runs/phrase_v4'))
    ap.add_argument('--device', default='cuda:0'); args = ap.parse_args(); out = args.output
    dest = out/'evaluation/token_oracle'; dest.mkdir(exist_ok=True)
    predictions = out/'evaluation/predictions.jsonl'
    checkpoint = out/'models/local.pt'; source = Path('src/clickagent/phrase_grounder.py')
    before = {str(x): sha256(x) for x in (predictions, checkpoint, source)}
    cases = [c for c in read_jsonl(out/'manifest.jsonl') if c['experiment_split'] == 'test']
    by = {(r['method'], r['sample_id']): r for r in read_jsonl(predictions)}
    records, _ = load_records(out, cases, with_labels=False)
    model, _ = load_phrase(checkpoint, args.device); torch.set_num_threads(4)
    rng = np.random.default_rng(20261031); rows = []
    for i, record in enumerate(records):
        sid = record['case']['sample_id']
        rows.append(diagnose(record, model, by['local', sid], by['qwen_native_completed_lax', sid], rng))
        if (i+1) % 24 == 0:
            print(f'Diagnosed {i+1}/{len(records)}', flush=True)
    names = list(next(r for r in rows if r['available'])['families'])
    groups = {'all': rows, 'current_wrong': [r for r in rows if not r['current_hit']],
        'native_correct_current_wrong': [r for r in rows if r['native_hit'] and not r['current_hit']],
        'native_correct_current_wrong_coarse': [r for r in rows if r['native_hit'] and not r['current_hit'] and r['current_failure_stage'] == 'wrong_coarse_region'],
        'multi_token': [r for r in rows if r.get('n_tokens', 0) > 1]}
    for ds in sorted({r['dataset'] for r in rows}):
        groups[ds] = [r for r in rows if r['dataset'] == ds]
    summary = dict(groups={k: summarize(v, names) for k, v in groups.items()},
        error_partition=dict(Counter(r.get('diagnostic_stage', 'no_name') for r in rows)),
        n_tokens=dict(Counter(r.get('n_tokens', 0) for r in rows)),
        rank_thresholds={str(k): sum(r.get('best_target_cell_rank_over_tokens', 10**9) <= k for r in rows) for k in (1, 3, 5, 10, 20)},
        caveats=[
            'Every any-token/head result uses ground truth to choose among multiple decoded candidates. It is not achievable model accuracy.',
            'Exact click is integer-pixel point inside any annotated target box/polygon, with inclusive boundaries; not equality with a single reference coordinate.',
            'Forced-route evidence weights still use final complete-phrase state. Single-token-entire-state removes that context but is an out-of-distribution intervention.',
            'Spatial null shifts all parent peaks by one common random toroidal offset and re-decodes with original destination image fine features. It controls candidate count and relative peak geometry, but is not a realistic GUI prior or a deployable method.',
            'Only cached attention and hidden states from layers 15,28,36 are inspected. No failure here proves the full VLM lacks position information.',
            'Same previously scored 192 test images: post-hoc diagnosis, no new independent performance claim. No fitting, threshold selection, or checkpoint selection.'],
        integrity_before=before, source_sha256=sha256(Path(__file__)))
    after = {str(x): sha256(x) for x in (predictions, checkpoint, source)}
    assert before == after
    summary.update(integrity_after=after, primary_replay_matches=sum(r['available'] for r in rows))
    write_json(dest/'summary.json', summary)
    (dest/'cases.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows), encoding='utf-8')
    print(json.dumps({k: v for k, v in summary.items() if k != 'groups'}, ensure_ascii=False, indent=2), flush=True)
    print(json.dumps(summary['groups']['all'], ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
