"""Score all frozen phrase variants and unchanged baselines on one fresh test."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch

from clickagent.action_benchmark import sha256, write_json
from clickagent.conditioned_grounder import load_spatial_checkpoint, make_batch
from clickagent.phrase_grounder import decode_phrase, load_phrase
from clickagent.spatial_geometry import decode_candidates
from clickagent.spatial_geometry import shape_from_geometry
from evaluate_conditioned_readouts import describe_candidates, score_point
from run_target_token_experiment import key, read_json, read_jsonl
from train_conditioned_readouts import batches, load_records


def attention_oracles(record, head_weights):
    """GT-assisted diagnostic only; no results are fed back into decoding."""
    import shapely
    from diagnose_target_information import random_set_overlap_probability
    c = record['case']; gh, gw = record['metadata']['grid_shape']; p = gh*gw
    rr, cc = np.divmod(np.arange(p), gw)
    cells = shapely.box(cc/gw, rr/gh, (cc+1)/gw, (rr+1)/gh)
    mask = shapely.area(shapely.intersection(shape_from_geometry(c['target_geometry']), cells)) > 0
    a = record['target_attention'].astype(np.float64)
    relative = a/np.maximum(a.sum(-1, keepdims=True), 1e-12)
    evidence = np.einsum('th,thp->tp', head_weights, np.log1p(p*relative))
    raw_peaks = np.unique(a.argmax(-1))
    return dict(any_token_learned_head_peak_overlap=bool(mask[evidence.argmax(-1)].any()),
                any_token_l15_mean_peak_overlap=bool(mask[a[:, :32].mean(1).argmax(-1)].any()),
                any_token_any_head_peak_overlap=bool(mask[raw_peaks].any()),
                raw_unique_peak_cells=int(len(raw_peaks)),
                raw_peak_random_assignment_expected=random_set_overlap_probability(p, int(mask.sum()), len(raw_peaks)),
                note='GT token/head selection oracle; coarse-cell overlap, not deployable click accuracy.')


def paired_interval(rows, first, second, cases):
    by = {(r['method'], r['sample_id']): int(r['hit']) for r in rows}
    datasets = sorted({c['dataset'] for c in cases})
    groups = [np.array([by[first, c['sample_id']]-by[second, c['sample_id']]
                        for c in cases if c['dataset'] == ds], dtype=float) for ds in datasets]
    rng = np.random.default_rng(20261102)
    draws = np.zeros(5000)
    for group in groups:
        draws += group[rng.integers(0, len(group), (5000, len(group)))].sum(1)
    draws *= 100/len(cases)
    deltas = np.concatenate(groups)
    return dict(first=first, second=second, delta_pp=float(deltas.mean()*100),
                ci95_pp=np.quantile(draws, [.025, .975]).tolist(), wins=int((deltas == 1).sum()), losses=int((deltas == -1).sum()),
                note='Exploratory paired stratified bootstrap, 5000 draws; no multiplicity correction.')


def row_for(case, method, point, proposals=None, meta=None):
    row = dict(sample_id=case['sample_id'], dataset=case['dataset'], method=method, point=point, hit=score_point(point, case))
    if proposals is not None:
        row.update(candidates=proposals, **describe_candidates(proposals, case, meta['grid_shape'], meta['fine_grid_shape']))
        if row['hit']:
            stage = 'hit'
        elif not proposals:
            stage = 'no_name'
        elif not row['selected_coarse_cell_overlap']:
            stage = 'wrong_coarse_region'
        elif not row['selected_fine_cell_overlap']:
            stage = 'correct_coarse_wrong_fine'
        else:
            stage = 'fine_overlap_offset_miss'
        row['failure_stage'] = stage
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output', type=Path, default=Path('runs/phrase_v4'))
    p.add_argument('--device', default='cuda:0'); args = p.parse_args(); out = args.output
    frozen = read_json(out/'frozen_readouts.json'); protocol = read_json(out/'protocol.json')
    assert frozen['test_features_at_freeze'] == 0 and read_json(out/'features_test_completion.json')['complete']
    for source, digest in frozen['head_sources'].items():
        assert sha256(Path(source)) == digest, source
    cases = [c for c in read_jsonl(out/'manifest.jsonl') if c['experiment_split'] == 'test']
    assert len(cases) == protocol['test_queries']
    records, _ = load_records(out, cases, with_labels=False); rows = []; torch.set_num_threads(4)
    for r in records:
        c = r['case']; n = read_json(out/'native'/f'{key(c["sample_id"])}.json')
        for method, point in [('qwen_native_strict', n['point']), ('qwen_native_completed_lax', n['completed_lax_point'])]:
            rows.append(row_for(c, method, point))
        u = read_json(out/'ui_tars'/f'{key(c["sample_id"])}.json')
        rows.append(row_for(c, 'ui_tars', u['point']))
    for name, entry in frozen['models'].items():
        path = out/'models'/entry['file']; assert sha256(path) == entry['sha256']
        model, _ = load_phrase(path, args.device); done = set()
        with torch.inference_mode():
            for group in batches(records, 8):
                inputs, _ = make_batch(group, args.device, labels=False); result = model(**inputs)
                cpu = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in result.items()}
                diagnostic_results = {}
                if name == 'local':
                    # Keep the complete final phrase state unchanged. Rotating
                    # earlier token pairs should preserve this set-based model;
                    # rotating query states alone breaks word/map alignment.
                    q_rotated = inputs['query_hidden'].clone(); a_rotated = inputs['attention'].clone()
                    for j, record in enumerate(group):
                        count = len(record['target_attention'])-1
                        if count > 1:
                            order = torch.roll(torch.arange(count, device=args.device), 1)
                            q_rotated[j, :count] = inputs['query_hidden'][j, order]
                            a_rotated[j, :count] = inputs['attention'][j, order]
                    paired = model(**dict(inputs, query_hidden=q_rotated, attention=a_rotated))
                    torch.testing.assert_close(paired['coarse_logits'], result['coarse_logits'], atol=2e-5, rtol=2e-5)
                    misaligned = model(**dict(inputs, query_hidden=q_rotated))
                    for method, value in [('local_pair_order_control', paired), ('local_query_map_misaligned', misaligned)]:
                        diagnostic_results[method] = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in value.items()}
                for i, r in enumerate(group):
                    c = r['case']; meta = r['metadata']; done.add(c['sample_id'])
                    proposals = decode_phrase(cpu, i, c['image_size'])
                    row = row_for(c, name, proposals[0]['point'], proposals, meta)
                    cell = proposals[0]['coarse_index']
                    weights = cpu['token_weights'][i, :len(r['target_attention']), cell].tolist()
                    row.update(target_name=meta['target_text'], token_weights_at_selected_cell=weights,
                               selected_token_index=int(np.argmax(weights)), generated_action=meta['action'])
                    rows.append(row)
                    if name == 'local':
                        row['attention_oracles'] = attention_oracles(r, cpu['head_weights'][i, :len(r['target_attention'])].numpy())
                        for method, value in diagnostic_results.items():
                            dp = decode_phrase(value, i, c['image_size'], top_k=1)[0]['point']
                            dr = row_for(c, method, dp); dr['eligible_three_or_more_tokens'] = len(r['target_attention']) >= 3
                            rows.append(dr)
                        w, h = c['image_size']; gh, gw = meta['grid_shape']; rr, cc = divmod(cell, gw)
                        center = [min(w-1, round((cc+.5)*w/gw)), min(h-1, round((rr+.5)*h/gh))]
                        rows.append(row_for(c, 'local_coarse_centers', center))
                        fine = decode_phrase(cpu, i, c['image_size'], top_k=1, center_only=True)[0]['point']
                        rows.append(row_for(c, 'local_fine_centers', fine))
                        rows.append(row_for(c, 'local_generated_action_policy', proposals[0]['point'] if meta['action'] == 'CLICK' else None))
            for r in records:
                c = r['case']
                if c['sample_id'] in done:
                    continue
                rows.append(row_for(c, name, None, [], r['metadata']))
                if name == 'local':
                    for method in ('local_coarse_centers', 'local_fine_centers', 'local_generated_action_policy', 'local_pair_order_control', 'local_query_map_misaligned'):
                        rows.append(row_for(c, method, None))
        print('SCORED '+name+': '+str(sum(r['hit'] for r in rows if r['method'] == name)), flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    for name, entry in frozen['baselines'].items():
        path = Path(entry['file']); assert sha256(path) == entry['sha256']
        model, _ = load_spatial_checkpoint(path, args.device); done = set()
        with torch.inference_mode():
            for group in batches(records, 8):
                inputs, _ = make_batch(group, args.device, labels=False); result = model(**inputs)
                values = result['logits'].cpu().numpy(); offsets = result['offsets'].cpu().numpy()
                for i, r in enumerate(group):
                    c = r['case']; meta = r['metadata']; done.add(c['sample_id'])
                    proposals = decode_candidates(values[i], offsets[i], result['grid_shape'], c['image_size'])
                    # static_96 emits coarse grid indices, unlike all other heads.
                    if result.get('static_coarse_control', False):
                        rows.append(row_for(c, name, proposals[0]['point']))
                    else:
                        rows.append(row_for(c, name, proposals[0]['point'], proposals, meta))
            for r in records:
                if r['case']['sample_id'] not in done:
                    rows.append(row_for(r['case'], name, None, [], r['metadata']))
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    methods = {}
    for name in sorted({r['method'] for r in rows}):
        group = [r for r in rows if r['method'] == name]
        assert len(group) == len(cases) and {r['sample_id'] for r in group} == {c['sample_id'] for c in cases}
        datasets = {ds: dict(n=sum(r['dataset'] == ds for r in group), hits=sum(r['hit'] for r in group if r['dataset'] == ds))
                    for ds in sorted({c['dataset'] for c in cases})}
        for ds in datasets.values():
            ds['accuracy'] = ds['hits']/ds['n']
        methods[name] = dict(n=len(cases), hits=sum(r['hit'] for r in group), accuracy=sum(r['hit'] for r in group)/len(cases),
                             macro_accuracy=float(np.mean([d['accuracy'] for d in datasets.values()])), datasets=datasets,
                             points=sum(r['point'] is not None for r in group),
                             failure_stages=dict(Counter(r['failure_stage'] for r in group if 'failure_stage' in r)),
                             selected_coarse_overlap=sum(r.get('selected_coarse_cell_overlap', False) for r in group),
                             candidate_point_oracle=sum(r.get('candidate_point_oracle_hit', False) for r in group))
    primary = {r['sample_id']: r for r in rows if r['method'] == 'local'}
    native = {r['sample_id']: r for r in rows if r['method'] == 'qwen_native_completed_lax'}
    named = [r for r in rows if r['method'] == 'local' and 'token_weights_at_selected_cell' in r]
    primary_diagnostics = dict(named=len(named), first_largest=sum(r['selected_token_index'] == 0 for r in named),
                               mean_first_weight=float(np.mean([r['token_weights_at_selected_cell'][0] for r in named])),
                               native_correct_local_wrong=sum(native[s]['hit'] and not primary[s]['hit'] for s in primary),
                               local_correct_native_wrong=sum(primary[s]['hit'] and not native[s]['hit'] for s in primary))
    diagnostic_rows = [r for r in rows if r['method'] == 'local_query_map_misaligned' and r.get('eligible_three_or_more_tokens')]
    primary_diagnostics['query_map_alignment_intervention'] = dict(eligible=len(diagnostic_rows),
        original_hits=sum(primary[r['sample_id']]['hit'] for r in diagnostic_rows),
        misaligned_hits=sum(r['hit'] for r in diagnostic_rows),
        note='Rotate earlier query states, retain last complete-phrase state and all maps. Paired rotations are an order-invariance control. Diagnostic only.')
    primary_diagnostics['spatial_evidence_oracles'] = dict(
        any_token_learned_head_peak_overlap=sum(r['attention_oracles']['any_token_learned_head_peak_overlap'] for r in named),
        any_token_l15_mean_peak_overlap=sum(r['attention_oracles']['any_token_l15_mean_peak_overlap'] for r in named),
        any_token_any_head_peak_overlap=sum(r['attention_oracles']['any_token_any_head_peak_overlap'] for r in named),
        raw_peak_random_assignment_expected=sum(r['attention_oracles']['raw_peak_random_assignment_expected'] for r in named),
        wrong_coarse_but_some_token_learned_peak_overlaps=sum(not r['selected_coarse_cell_overlap'] and r['attention_oracles']['any_token_learned_head_peak_overlap'] for r in named),
        note='Ground-truth selection upper diagnostics, not achievable accuracy; raw peak coverage benefits from many guesses.')
    length_subsets = {}
    by_prediction = {(r['method'], r['sample_id']): r for r in rows}
    for subset, lower, upper in [('single_token', 1, 1), ('multiple_tokens', 2, 10000), ('no_name', 0, 0)]:
        ids = [r['case']['sample_id'] for r in records if lower <= len(r['target_attention']) <= upper]
        length_subsets[subset] = dict(n=len(ids), methods={
            method: dict(hits=sum(by_prediction[method, sid]['hit'] for sid in ids), n=len(ids))
            for method in ['first', 'uniform', 'global', 'local', 'first_replica', 'local_replica']})
    comparisons = [paired_interval(rows, a, b, cases) for a, b in [
        ('local', 'first'), ('local_replica', 'first_replica'), ('local', 'uniform'), ('local', 'global'),
        ('global', 'first'), ('uniform', 'first'),
        ('local', 'previous_full'), ('local', 'previous_attention_only'), ('local', 'qwen_native_completed_lax'), ('local', 'ui_tars')]]
    summary = dict(primary='local', n=len(cases), methods=methods, paired_comparisons=comparisons,
                   primary_diagnostics=primary_diagnostics, name_length_subsets=length_subsets, all_failures_retained=True,
                   test_used_for_model_selection=False, manifest_sha256=sha256(out/'manifest.jsonl'),
                   evaluator_sha256=sha256(Path(__file__)), frozen_readouts_sha256=sha256(out/'frozen_readouts.json'))
    dest = out/'evaluation'; dest.mkdir(exist_ok=True)
    (dest/'predictions.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows), encoding='utf-8')
    write_json(dest/'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
