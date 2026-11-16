"""Frozen v6 evaluation, same-candidate ranking and common-coordinate controls."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch

from clickagent.action_benchmark import sha256, write_json
from clickagent.attention_field_ranker import load_field_ranker, make_field_batch
from clickagent.candidate_ranker import decode_ranker, load_ranker, make_ranker_batch
from attention_field_data import load_records
from evaluate_conditioned_readouts import score_point
from evaluate_phrase_readouts import paired_interval
from run_target_token_experiment import read_jsonl


def row(case, method, point, **extra):
    return dict(sample_id=case['sample_id'], dataset=case['dataset'], method=method,
        point=point, hit=bool(score_point(point, case)), **extra)


@torch.inference_mode()
def common_points(records, base, device):
    frozen = json.loads((base/'frozen_readouts.json').read_text()); entry = frozen['models']['full']
    path = base/'models'/entry['file']; assert sha256(path) == entry['sha256']
    model, _ = load_ranker(path, device); points = {}
    named = [r for r in records if r['metadata']['target_available']]
    for start in range(0, len(named), 8):
        group = named[start:start+8]; inputs, _ = make_ranker_batch(group, device, labels=False)
        output = model(**inputs); cpu = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in output.items()}
        for i, r in enumerate(group):
            points[r['case']['sample_id']] = {p['candidate_index']: p['point'] for p in
                decode_ranker(cpu, r, i, top_k=len(r['parent_indices']))}
    del model; torch.cuda.empty_cache(); return points


@torch.inference_mode()
def evaluate_all(out, device):
    torch.set_num_threads(4)
    protocol = json.loads((out/'protocol.json').read_text()); base = Path(protocol['base_run'])
    frozen = json.loads((out/'frozen_readouts.json').read_text())
    for source, digest in frozen['sources'].items(): assert sha256(Path(source)) == digest, source
    assert sha256(base/'evaluation/predictions.jsonl') == frozen['v5_prediction_sha256']
    records = load_records(out, 'test'); cases = [r['case'] for r in records]
    common = common_points(records, base, device); rows = []
    baseline_names = ['full', 'full_center', 'full_frozen_fine', 'stats_only', 'phrase_local',
        'phrase_global', 'qwen_native_completed_lax', 'ui_tars']
    for r in read_jsonl(base/'evaluation/predictions.jsonl'):
        if r['method'] in baseline_names:
            r = dict(r); r['method'] = 'v5_'+r['method'] if r['method'].startswith(('full', 'stats')) else r['method']
            rows.append(r)
    named = [r for r in records if r['metadata']['target_available']]
    for name, entry in frozen['models'].items():
        path = out/'models'/entry['file']; assert sha256(path) == entry['sha256']
        model, _ = load_field_ranker(path, device)
        for start in range(0, len(named), protocol['batch_size']):
            group = named[start:start+protocol['batch_size']]
            inputs, _ = make_field_batch(group, device, labels=False); output = model(**inputs)
            cpu = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in output.items()}
            for i, r in enumerate(group):
                c = r['case']; sid = c['sample_id']; n = len(r['parent_indices'])
                proposals = decode_ranker(cpu, r, i); first = proposals[0]; ci = first['candidate_index']
                positive = r['label_area'].sum(-1) > 0; covered = bool(positive.any()); selected = bool(positive[ci])
                center = decode_ranker(cpu, r, i, mode='center', top_k=1)[0]['point']
                fixed = common[sid][ci]; hit = bool(score_point(first['point'], c))
                stage = 'hit' if hit else ('fine_miss_after_intersecting_parent' if selected else
                    'ranking_miss_with_target_candidate' if covered else 'no_target_peak_candidate')
                cursor = sum(cpu['candidate_lengths'][:i])
                extras = dict(candidates=proposals, target_name=r['metadata']['target_text'], n_candidates=n,
                    selected_region_overlap=selected, candidate_region_oracle=covered,
                    ranked_region_hits={str(k): bool(any(positive[p['candidate_index']] for p in proposals[:k])) for k in (1, 3, 5)},
                    selected_parent_index=first['parent_index'], selected_candidate_index=ci,
                    residual_score=float(cpu['field_residual'][cursor+ci]), failure_stage=stage)
                rows.append(row(c, name, first['point'], **extras))
                rows.append(row(c, name+'_center', center))
                rows.append(row(c, name+'_fixed_fine', fixed))
        for r in records:
            if not r['metadata']['target_available']:
                for suffix in ('', '_center', '_fixed_fine'):
                    rows.append(row(r['case'], name+suffix, None, failure_stage='no_name'))
        print('SCORED '+name, flush=True); del model; torch.cuda.empty_cache()
    methods = {}
    for name in sorted({r['method'] for r in rows}):
        group = [r for r in rows if r['method'] == name]
        assert len(group) == len(cases) == len({r['sample_id'] for r in group})
        result = dict(n=len(cases), hits=sum(r['hit'] for r in group), accuracy=sum(r['hit'] for r in group)/len(cases),
            points=sum(r['point'] is not None for r in group), datasets={ds: dict(n=sum(r['dataset'] == ds for r in group),
                hits=sum(r['hit'] for r in group if r['dataset'] == ds)) for ds in sorted({c['dataset'] for c in cases})})
        if name in frozen['models'] or name == 'v5_full':
            coverage = sum(r.get('candidate_region_oracle', False) for r in group)
            top1 = sum(r.get('selected_region_overlap', False) for r in group)
            result.update(candidate_region_coverage=coverage, region_top1=top1,
                region_top3=sum(r.get('ranked_region_hits', {}).get('3', False) for r in group),
                region_top5=sum(r.get('ranked_region_hits', {}).get('5', False) for r in group),
                selection_given_covered=top1/coverage if coverage else None,
                failure_stages=dict(Counter(r.get('failure_stage') for r in group)))
        methods[name] = result
    region_rows = [dict(r, method=r['method']+'_parent', hit=r.get('selected_region_overlap', False))
                   for r in rows if r['method'] in list(frozen['models'])+['v5_full']]
    pairs = [('all', 'peaks_only'), ('all_replica', 'peaks_only_replica'), ('all', 'v5_full'),
        ('all', 'qwen_native_completed_lax'), ('all_fixed_fine', 'peaks_only_fixed_fine'),
        ('all_replica_fixed_fine', 'peaks_only_replica_fixed_fine'), ('all_fixed_fine', 'v5_full'),
        ('all_parent', 'peaks_only_parent'), ('all_replica_parent', 'peaks_only_replica_parent'), ('all_parent', 'v5_full_parent')]
    comparisons = [paired_interval(rows+region_rows, a, b, cases) for a, b in pairs]
    summary = dict(primary='all', n=len(cases), methods=methods, paired_comparisons=comparisons,
        manifest_sha256=protocol['manifest_sha256'], frozen_sha256=sha256(out/'frozen_readouts.json'),
        evaluator_sha256=sha256(Path(__file__)), test_used_for_checkpoint_selection=False,
        diagnostic_reuse=True, test_is_fresh=False, warning=protocol['interpretation_limits'])
    dest = out/'evaluation'; dest.mkdir(exist_ok=True)
    (dest/'predictions.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows), encoding='utf-8')
    write_json(dest/'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, default=Path('runs/attention_field_v6'))
    p.add_argument('--device', default='cuda:0'); args = p.parse_args(); evaluate_all(args.output, args.device)


if __name__ == '__main__': main()
