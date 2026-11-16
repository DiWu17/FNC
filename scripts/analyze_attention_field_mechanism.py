"""Post-hoc frozen-weight spatial erasures, alignment shuffles and margin attribution.

This never trains, selects checkpoints or replaces the official v6 predictions.
GT is used only for scoring and choosing diagnostic rescue/loss cohorts.
"""
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
import torch

from clickagent.action_benchmark import sha256, write_json
from clickagent.attention_field_diagnostics import perturb_field, token_pieces
from clickagent.attention_field_ranker import load_field_ranker, make_field_batch
from clickagent.candidate_ranker import decode_ranker
from clickagent.spatial_geometry import shape_from_geometry
from attention_field_data import load_records
from evaluate_conditioned_readouts import score_point
from run_target_token_experiment import read_jsonl


OUT = Path('runs/attention_field_v6')
DEST = OUT/'evaluation/mechanism_analysis'


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def interventions():
    return [('full', 'full', 0), ('nonpeak_all_zero', 'nonpeak_all_zero', 0),
        ('nonpeak_spatial_zero', 'nonpeak_spatial_zero', 0),
        ('nonpeak_context_zero', 'nonpeak_context_zero', 0), ('no_field_residual', None, 0)]+[
        (f'spatial_shuffle_{seed}', 'nonpeak_spatial_shuffle', seed) for seed in range(5)]


@torch.inference_mode()
def replay(model, name, records, stored, paired, device):
    named = [r for r in records if r['metadata']['target_available']]; rows = []
    for start in range(0, len(named), 8):
        group = named[start:start+8]; inputs, _ = make_field_batch(group, device, labels=False)
        original = model(**inputs)
        for label, mode, seed in interventions():
            if mode == 'full': output = original
            elif mode is None: output = dict(original, scores=original['base_scores'])
            else:
                fields = [perturb_field(r['attention_field'], mode, r['case']['sample_id'], seed).reshape(-1, 96, 6) for r in group]
                altered = dict(inputs, attention_field=torch.from_numpy(np.concatenate(fields)).to(device))
                output = model(**altered)
            cpu = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in output.items()}
            cursor = 0
            for i, r in enumerate(group):
                sid = r['case']['sample_id']; c = r['case']; pred = decode_ranker(cpu, r, i, top_k=1)[0]
                ci = pred['candidate_index']; overlap = bool(r['label_area'][ci].sum() > 0)
                full_ci = stored[name, sid]['selected_candidate_index']; other_ci = stored[paired, sid]['selected_candidate_index']
                scores = cpu['scores'][cursor:cursor+len(r['parent_indices'])]
                if label == 'full':
                    assert pred['point'] == stored[name, sid]['point'] and pred['parent_index'] == stored[name, sid]['selected_parent_index']
                rows.append(dict(model=name, intervention=label, sample_id=sid, dataset=c['dataset'],
                    point=pred['point'], hit=bool(score_point(pred['point'], c)), region_hit=overlap,
                    selected_candidate=ci, selected_parent=pred['parent_index'],
                    full_chosen_score=float(scores[full_ci]), control_chosen_score=float(scores[other_ci]),
                    full_vs_control_margin=float(scores[full_ci]-scores[other_ci])))
                cursor += len(r['parent_indices'])
        print(f'REPLAY {name}: {min(start+8, len(named))}/{len(named)}', flush=True)
    for r in records:
        if not r['metadata']['target_available']:
            for label, _, _ in interventions():
                rows.append(dict(model=name, intervention=label, sample_id=r['case']['sample_id'],
                    dataset=r['case']['dataset'], point=None, hit=False, region_hit=False))
    return rows


def attribution(model, r, ci, other, device, steps=32):
    """IG of the correct-versus-control score margin, spatial nonpeaks only."""
    inputs, _ = make_field_batch([r], device, labels=False)
    x = inputs['attention_field'].float().detach(); base = x.clone()
    base[..., :3] *= base[..., 5:6]; delta = x-base
    def margin(y):
        out = model(**dict(inputs, attention_field=y))
        return out['scores'][ci]-out['scores'][other]
    with torch.no_grad(): actual_delta = float(margin(x)-margin(base))
    while True:
        total = torch.zeros_like(x)
        for j in range(steps+1):
            y = (base+delta*(j/steps)).detach().requires_grad_(True)
            grad, = torch.autograd.grad(margin(y), y)
            total += grad*(.5 if j in (0, steps) else 1.)
        attr = delta*total/steps
        residual = float(attr.sum())-actual_delta
        if abs(residual) <= max(.02, .05*abs(actual_delta)) or steps >= 128: break
        steps *= 2
    head = attr.sum((0, 2)).detach().cpu().numpy()
    token = torch.zeros(len(r['query_hidden']), device=device).index_add(0, inputs['field_token'], attr.sum((1, 2))).cpu().numpy()
    top_heads = np.argsort(-head)[:5]
    erase_heads = x.clone()
    for h in top_heads[:3]: erase_heads[:, h, :3] *= erase_heads[:, h, 5:6]
    with torch.no_grad():
        original_margin = float(margin(x)); top3_margin = float(margin(erase_heads))
    return dict(steps=steps, margin_gain_from_spatial_nonpeaks=actual_delta,
        integrated_gradient_sum=float(attr.sum()), completeness_absolute_error=abs(residual),
        full_margin=original_margin, top3_head_erased_margin=top3_margin,
        per_head_margin_contribution=head.tolist(), per_token_margin_contribution=token.tolist(),
        top_heads=[dict(index=int(h), head=r['metadata']['head_order'][int(h)], contribution=float(head[h])) for h in top_heads])


def stats(records, cohort):
    selected = [r for r in records if r['case']['sample_id'] in cohort]
    if not selected: return dict(n=0)
    return dict(n=len(selected), datasets=dict(Counter(r['case']['dataset'] for r in selected)),
        domains=dict(Counter(r['case'].get('domain', 'unknown') for r in selected)),
        data_types=dict(Counter(r['case'].get('metadata', {}).get('data_type', 'unknown') for r in selected)),
        median_tokens=float(np.median([len(r['query_hidden']) for r in selected])),
        median_target_area=float(np.median([shape_from_geometry(r['case']['target_geometry']).area for r in selected])))


def main():
    started = time.perf_counter(); DEST.mkdir(exist_ok=True); torch.set_num_threads(4); device = 'cuda:0'
    frozen = read(OUT/'frozen_readouts.json')
    for source, digest in frozen['sources'].items(): assert sha256(Path(source)) == digest
    original_sha = sha256(OUT/'evaluation/predictions.jsonl')
    identity = dict(official_predictions_sha256=original_sha, frozen_readouts_sha256=sha256(OUT/'frozen_readouts.json'),
        source_sha256={p: sha256(Path(p)) for p in ['scripts/analyze_attention_field_mechanism.py', 'src/clickagent/attention_field_diagnostics.py']},
        checkpoints={n: frozen['models'][n]['sha256'] for n in ['all', 'all_replica']},
        no_training=True, test_is_fresh=False, post_hoc_gt_used_for_cohort_and_contrast_only=True,
        spatial_channels=[0, 1, 2], context_channels=[3, 4], shuffle_seeds=list(range(5)), ig_initial_steps=32,
        warning='Erasure can cause distribution shift; candidate shuffle preserves each token/head spatial-channel joint marginals. IG explains this learned readout, not causal internals of Qwen.')
    write_json(DEST/'identity.json', identity)
    official = read_jsonl(OUT/'evaluation/predictions.jsonl'); stored = {(r['method'], r['sample_id']): r for r in official}
    records = load_records(OUT, 'test'); all_rows = []; details = []; cohorts = {}; summaries = {}
    for name, paired in [('all', 'peaks_only'), ('all_replica', 'peaks_only_replica')]:
        path = OUT/'models'/f'{name}.pt'; assert sha256(path) == frozen['models'][name]['sha256']
        model, _ = load_field_ranker(path, device); model.requires_grad_(False)
        wins = {r['case']['sample_id'] for r in records if stored[name, r['case']['sample_id']].get('selected_region_overlap')
            and not stored[paired, r['case']['sample_id']].get('selected_region_overlap')}
        losses = {r['case']['sample_id'] for r in records if not stored[name, r['case']['sample_id']].get('selected_region_overlap')
            and stored[paired, r['case']['sample_id']].get('selected_region_overlap')}
        cohorts[name] = dict(region_wins=sorted(wins), region_losses=sorted(losses),
            win_characteristics=stats(records, wins), loss_characteristics=stats(records, losses))
        replay_rows = replay(model, name, records, stored, paired, device); all_rows.extend(replay_rows)
        by = {(r['intervention'], r['sample_id']): r for r in replay_rows}
        for label, _, _ in interventions():
            group = [r for r in replay_rows if r['intervention'] == label]
            summaries[name, label] = dict(n=192, hits=sum(r['hit'] for r in group), region_hits=sum(r['region_hit'] for r in group),
                rescue_cohort_region_hits=sum(by[label, sid]['region_hit'] for sid in wins),
                rescue_margin_flips=sum(by[label, sid]['full_vs_control_margin'] <= 0 for sid in wins),
                rescue_margin_gain_median=float(np.median([by['full', sid]['full_vs_control_margin']-by[label, sid]['full_vs_control_margin'] for sid in wins])))
        for i, r in enumerate([r for r in records if r['case']['sample_id'] in wins]):
            sid = r['case']['sample_id']; a = stored[name, sid]; b = stored[paired, sid]
            ci, other = a['selected_candidate_index'], b['selected_candidate_index']; field = r['attention_field']
            ig = attribution(model, r, ci, other, device)
            pieces = token_pieces(r['metadata']); assert len(pieces) == len(r['query_hidden'])
            def support(c):
                f = field[c]; mask = f[..., 5] == 0
                return dict(peak_votes=int((~mask).sum()), nonpeak_ratio_ge_half=int(((f[..., 1] >= .5) & mask).sum()),
                    nonpeak_ratio_ge_quarter=int(((f[..., 1] >= .25) & mask).sum()))
            details.append(dict(model=name, sample_id=sid, instruction=r['case']['instruction'],
                target_name=r['metadata']['target_text'], token_pieces=pieces,
                selected_candidate=ci, control_candidate=other, correct_support=support(ci), wrong_support=support(other),
                spatial_erasure_region_hit=by['nonpeak_spatial_zero', sid]['region_hit'],
                spatial_erasure_margin=by['nonpeak_spatial_zero', sid]['full_vs_control_margin'], **ig))
            print(f'ATTRIBUTION {name}: {i+1}/{len(wins)}', flush=True)
        del model; torch.cuda.empty_cache()
    # Re-score every intervention point; preserve all original results byte-for-byte.
    cases = {r['case']['sample_id']: r['case'] for r in records}
    for row in all_rows: assert row['hit'] == bool(score_point(row['point'], cases[row['sample_id']]))
    assert sha256(OUT/'evaluation/predictions.jsonl') == original_sha
    (DEST/'predictions.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in all_rows), encoding='utf-8')
    write_json(DEST/'rescue_attributions.json', details)
    grouped = {name: {label: summaries[name, label] for label, _, _ in interventions()} for name in ['all', 'all_replica']}
    result = dict(complete=True, methods=grouped, cohorts=cohorts, attribution_n=len(details),
        attribution_completeness_max_error=max(d['completeness_absolute_error'] for d in details),
        original_predictions_unchanged=True, diagnostic_prediction_count=len(all_rows), elapsed_seconds=time.perf_counter()-started,
        identity_sha256=sha256(DEST/'identity.json'), predictions_sha256=sha256(DEST/'predictions.jsonl'))
    write_json(DEST/'summary.json', result)
    print(json.dumps(dict(methods=grouped, attribution_n=len(details), elapsed_seconds=result['elapsed_seconds'])), flush=True)


if __name__ == '__main__': main()
