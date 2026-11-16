"""Score frozen semantic rankers, old readouts and native VLMs on fresh images."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch

from clickagent.action_benchmark import sha256, write_json
from clickagent.candidate_ranker import decode_ranker, load_ranker, make_ranker_batch
from clickagent.conditioned_grounder import make_batch
from clickagent.phrase_grounder import decode_phrase, load_phrase
from evaluate_conditioned_readouts import score_point
from evaluate_phrase_readouts import paired_interval
from ranker_data import load_records
from run_target_token_experiment import key, read_jsonl
from train_conditioned_readouts import batches, load_records as load_raw_records


def parent_points(children, offsets, grid, image_size):
    gh, gw = grid; w, h = image_size; result = []
    for parent in range(gh*gw):
        rr, cc = divmod(parent, gw); child = int(children[rr, cc].argmax())
        row, col = rr*2+child//2, cc*2+child % 2; dx, dy = offsets[:, row, col]
        result.append([int(np.clip(np.rint((col+float(dx))*w/(2*gw)), 0, w-1)),
                       int(np.clip(np.rint((row+float(dy))*h/(2*gh)), 0, h-1))])
    return result


def prediction(case, method, point, **extra):
    return dict(sample_id=case['sample_id'], dataset=case['dataset'], method=method,
                point=point, hit=score_point(point, case), **extra)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, default=Path('runs/candidate_ranker_v5')); p.add_argument('--device', default='cuda:0')
    args = p.parse_args(); out = args.output; torch.set_num_threads(4)
    frozen = json.loads((out/'frozen_readouts.json').read_text(encoding='utf-8'))
    for source, digest in frozen['head_sources'].items(): assert sha256(Path(source)) == digest, source
    for entry in frozen['models'].values(): assert sha256(out/'models'/entry['file']) == entry['sha256']
    assert json.loads((out/'features_test_completion.json').read_text(encoding='utf-8'))['complete']
    assert json.loads((out/'ui_tars/summary.json').read_text(encoding='utf-8'))['complete']
    records = load_records(out, 'test'); cases = [r['case'] for r in records]; rows = []; fine_lookup = {}
    raw, _ = load_raw_records(out, cases, with_labels=False)
    old_frozen = json.loads(Path('runs/phrase_v4/frozen_readouts.json').read_text(encoding='utf-8'))
    for name in ('local', 'global'):
        entry = old_frozen['models'][name]; path = Path('runs/phrase_v4/models')/entry['file']; assert sha256(path) == entry['sha256']
        model, _ = load_phrase(path, args.device); done = set()
        with torch.inference_mode():
            for group in batches(raw, 8):
                inputs, _ = make_batch(group, args.device, labels=False); output = model(**inputs)
                cpu = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in output.items()}
                for i, r in enumerate(group):
                    c = r['case']; sid = c['sample_id']; done.add(sid)
                    proposals = decode_phrase(cpu, i, c['image_size'])
                    rows.append(prediction(c, 'phrase_'+name, proposals[0]['point'], candidates=proposals))
                    if name == 'local':
                        fine_lookup[sid] = parent_points(cpu['child_log_prob'][i].numpy(), cpu['offsets'][i].numpy(), r['metadata']['grid_shape'], c['image_size'])
        for r in records:
            if r['case']['sample_id'] not in done: rows.append(prediction(r['case'], 'phrase_'+name, None))
        del model; torch.cuda.empty_cache()
    del raw
    for r in records:
        c = r['case']; sid = c['sample_id']
        n = json.loads((out/'native'/f'{key(sid)}.json').read_text(encoding='utf-8'))
        rows.append(prediction(c, 'qwen_native_completed_lax', n['completed_lax_point']))
        rows.append(prediction(c, 'qwen_native_strict', n['point']))
        u = json.loads((out/'ui_tars'/f'{key(sid)}.json').read_text(encoding='utf-8'))
        rows.append(prediction(c, 'ui_tars', u['point']))
    for name, entry in frozen['models'].items():
        model, _ = load_ranker(out/'models'/entry['file'], args.device)
        named = [r for r in records if r['metadata']['target_available']]
        with torch.inference_mode():
            for start in range(0, len(named), 8):
                group = named[start:start+8]; inputs, _ = make_ranker_batch(group, args.device, labels=False); output = model(**inputs)
                cpu = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in output.items()}; cursor = 0; source_cursor = 0
                for i, r in enumerate(group):
                    c = r['case']; sid = c['sample_id']; count = len(r['parent_indices'])
                    proposals = decode_ranker(cpu, r, i); first = proposals[0]; selected = first['candidate_index']
                    area = r['label_area'].sum(-1); covered = bool((area > 0).any()); selected_hit = bool(area[selected] > 0)
                    center = decode_ranker(cpu, r, i, mode='center', top_k=1)[0]['point']
                    frozen_fine = fine_lookup[sid][first['parent_index']]
                    rejected = float(cpu['null_scores'][i]) >= float(cpu['scores'][cursor:cursor+count].max())
                    all_fine = decode_ranker(cpu, r, i, top_k=count)
                    all_center = decode_ranker(cpu, r, i, mode='center', top_k=count)
                    fine_oracle = any(score_point(p['point'], c) for p in all_fine)
                    center_oracle = any(score_point(p['point'], c) for p in all_center)
                    hit = score_point(first['point'], c)
                    source_count = len(r['source_head'])
                    source_weights = cpu['source_weights'][source_cursor:source_cursor+source_count].numpy()
                    support = np.flatnonzero(r['source_candidate'] == selected)
                    support = support[np.argsort(-source_weights[support], kind='stable')[:8]]
                    source_details = [dict(token_index=int(r['source_token'][s]), layer=[15, 28, 36][int(r['source_head'][s])//32],
                        head_zero_based=int(r['source_head'][s]) % 32, within_candidate_weight=float(source_weights[s])) for s in support]
                    stage = 'hit' if hit else ('fine_miss_after_correct_region' if selected_hit else 'ranking_miss_with_target_candidate' if covered else 'no_target_peak_candidate')
                    extras = dict(candidates=proposals, target_name=r['metadata']['target_text'], n_candidates=count,
                        selected_region_overlap=selected_hit, candidate_region_oracle=covered,
                        candidate_new_fine_point_oracle=fine_oracle, candidate_center_point_oracle=center_oracle,
                        ranked_region_hits={str(k): bool(any(area[p['candidate_index']] > 0 for p in proposals[:k])) for k in (1, 3, 5)},
                        null_rejected=rejected, failure_stage=stage, selected_sources=source_details)
                    rows.append(prediction(c, name, first['point'], **extras))
                    rows.append(prediction(c, name+'_center', center))
                    rows.append(prediction(c, name+'_frozen_fine', frozen_fine))
                    rows.append(prediction(c, name+'_null_policy', None if rejected else first['point']))
                    cursor += count; source_cursor += source_count
        for r in records:
            if not r['metadata']['target_available']:
                for suffix in ('', '_center', '_frozen_fine', '_null_policy'):
                    rows.append(prediction(r['case'], name+suffix, None, failure_stage='no_name' if not suffix else 'no_name'))
        print(f'Scored {name}', flush=True); del model; torch.cuda.empty_cache()
    prior = json.loads(Path('runs/phrase_v4/evaluation/head_token_profile/train_only_head_prior.json').read_text(encoding='utf-8'))
    weights = np.array(prior['weights']); best_head = prior['ranking'][0]
    for r in records:
        c = r['case']; point = None; best_point = None
        if r['metadata']['target_available']:
            n = len(r['parent_indices']); source = r['source_candidate']; t = len(r['query_hidden'])
            votes = np.bincount(source, weights=weights[r['source_head']]/t, minlength=n)
            selected = int(votes.argmax()); gh, gw = r['metadata']['grid_shape']; w, h = c['image_size']
            def center_for(index):
                rr, cc = divmod(int(r['parent_indices'][index]), gw)
                return [int(np.clip(round((cc+.5)*w/gw), 0, w-1)), int(np.clip(round((rr+.5)*h/gh), 0, h-1))]
            point = center_for(selected)
            best_source = np.flatnonzero((r['source_head'] == best_head) & (r['source_token'] == t-1))[0]
            best_point = center_for(source[best_source])
        rows.append(prediction(c, 'train_weighted_peak_votes', point)); rows.append(prediction(c, 'train_best_head_last', best_point))
    methods = {}
    for name in sorted({r['method'] for r in rows}):
        group = [r for r in rows if r['method'] == name]
        assert len(group) == len(cases) and len({r['sample_id'] for r in group}) == len(cases)
        result = dict(n=len(cases), hits=sum(r['hit'] for r in group), points=sum(r['point'] is not None for r in group),
            accuracy=sum(r['hit'] for r in group)/len(cases), datasets={ds: dict(n=sum(r['dataset'] == ds for r in group),
            hits=sum(r['hit'] for r in group if r['dataset'] == ds)) for ds in sorted({c['dataset'] for c in cases})})
        if name in frozen['models']:
            covered = sum(r.get('candidate_region_oracle', False) for r in group)
            selected = sum(r.get('selected_region_overlap', False) for r in group)
            result.update(candidate_region_coverage=covered, region_top1=selected,
                region_top3=sum(r.get('ranked_region_hits', {}).get('3', False) for r in group),
                region_top5=sum(r.get('ranked_region_hits', {}).get('5', False) for r in group),
                selection_accuracy_given_covered=selected/covered if covered else None,
                fine_accuracy_given_correct_region=sum(r['hit'] and r.get('selected_region_overlap', False) for r in group)/selected if selected else None,
                candidate_new_fine_point_oracle=sum(r.get('candidate_new_fine_point_oracle', False) for r in group),
                candidate_center_point_oracle=sum(r.get('candidate_center_point_oracle', False) for r in group),
                null_rejections=sum(r.get('null_rejected', False) for r in group),
                failure_stages=dict(Counter(r.get('failure_stage') for r in group)))
        methods[name] = result
    comparisons = [paired_interval(rows, 'full', name, cases) for name in ('phrase_local', 'phrase_global',
        'qwen_native_completed_lax', 'ui_tars', 'stats_only', 'no_visual', 'no_query')]
    summary = dict(primary='full', n=len(cases), methods=methods, paired_comparisons=comparisons,
        frozen_readouts_sha256=sha256(out/'frozen_readouts.json'), manifest_sha256=sha256(out/'manifest.jsonl'),
        evaluator_sha256=sha256(Path(__file__)), test_used_for_selection=False, all_missing_outputs_retained=True,
        warning='Positive static grounding, not negative false-click rate or real execution. Different new images from phrase_v4; compare only same-cohort baselines.')
    dest = out/'evaluation'; dest.mkdir(exist_ok=True)
    (dest/'predictions.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows), encoding='utf-8')
    write_json(dest/'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__': main()
