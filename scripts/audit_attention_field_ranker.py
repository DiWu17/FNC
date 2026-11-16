"""Check frozen inputs, chosen epochs, baseline preservation and every v6 score."""
from collections import Counter
import json
from pathlib import Path

import numpy as np
from shapely.geometry import box
import torch

from clickagent.action_benchmark import sha256, write_json
from clickagent.attention_field_ranker import build_attention_field, load_field_ranker, make_field_batch
from clickagent.candidate_ranker import decode_ranker
from clickagent.spatial_geometry import shape_from_geometry
from attention_field_data import load_records
from evaluate_attention_field_ranker import common_points
from evaluate_conditioned_readouts import score_point
from run_target_token_experiment import key, read_jsonl


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


@torch.inference_mode()
def main():
    out = Path('runs/attention_field_v6'); protocol = read(out/'protocol.json'); base = Path(protocol['base_run'])
    frozen = read(out/'frozen_readouts.json'); summary = read(out/'evaluation/summary.json')
    torch.set_num_threads(4)
    assert protocol['diagnostic_reuse'] and not protocol['test_is_fresh'] and summary['diagnostic_reuse']
    assert sha256(out/'manifest.jsonl') == sha256(base/'manifest.jsonl') == protocol['manifest_sha256']
    assert sha256(base/'evaluation/predictions.jsonl') == protocol['v5_prediction_sha256'] == frozen['v5_prediction_sha256']
    for source, digest in frozen['sources'].items(): assert sha256(Path(source)) == digest, source
    for source, digest in protocol['unchanged_v5_sources'].items(): assert sha256(Path(source)) == digest, source
    assert read(base/'final_audit.json')['complete']
    assert sha256(Path(protocol['head_prior'])) == read(base/'training_identity.json')['prior_sha256']
    prior = np.array(read(protocol['head_prior'])['weights'], np.float32)
    expected_prior = np.log1p(96*prior/max(float(prior.sum()), 1e-9))
    epochs = {}; counts = []
    for name, entry in frozen['models'].items():
        assert sha256(out/'models'/entry['file']) == entry['sha256']
        h = read(out/'models'/f'{name}_history.json')
        best = max(h['history'], key=lambda r: (r['val']['region_hits']['1'], r['val']['hits'], -r['val']['loss'], -r['epoch']))
        assert best['epoch'] == h['best_epoch'] == entry['best_epoch']
        assert entry['parameters'] == h['parameters']; counts.append(entry['parameters']); epochs[name] = best['epoch']
    assert len(set(counts)) == 1
    records = load_records(out, 'test'); by_id = {r['case']['sample_id']: r for r in records}
    common = common_points(records, base, 'cpu')
    reconstructed = 0; nonpeak = 0; total = 0
    for r in records:
        assert sha256(Path(r['case']['image'])) == r['case']['image_sha256']
        if not r['metadata']['target_available']: continue
        path = base/'features'/f'{key(r["case"]["sample_id"])}.npz'
        with np.load(path, allow_pickle=False) as z:
            a = z['target_attention']; parents = np.unique(a.argmax(-1))
            np.testing.assert_array_equal(parents, r['parent_indices'])
            np.testing.assert_array_equal(build_attention_field(a, parents), r['attention_field'])
        f = r['attention_field']; nonpeak += int((f[..., 5] == 0).sum()); total += int(f[..., 5].size)
        reconstructed += 1
    rows = read_jsonl(out/'evaluation/predictions.jsonl'); by = {(r['method'], r['sample_id']): r for r in rows}
    assert len(by) == len(rows)
    for r in rows:
        rec = by_id[r['sample_id']]; case = rec['case']
        assert bool(r['hit']) == bool(score_point(r['point'], case))
        if r['method'] in frozen['models'] and r['point'] is not None:
            ci = r['selected_candidate_index']; parent = int(rec['parent_indices'][ci])
            assert parent == r['selected_parent_index']
            gh, gw = rec['metadata']['grid_shape']; rr, cc = divmod(parent, gw)
            intersection = shape_from_geometry(case['target_geometry']).intersection(box(cc/gw, rr/gh, (cc+1)/gw, (rr+1)/gh)).area > 0
            assert bool(r['selected_region_overlap']) == intersection
            assert by[r['method']+'_fixed_fine', r['sample_id']]['point'] == common[r['sample_id']][ci]
            w, h = case['image_size']
            center = [int(np.clip(np.rint((cc+.5)*w/gw), 0, w-1)), int(np.clip(np.rint((rr+.5)*h/gh), 0, h-1))]
            assert by[r['method']+'_center', r['sample_id']]['point'] == center
    for name, metric in summary['methods'].items():
        group = [r for r in rows if r['method'] == name]
        assert len(group) == metric['n'] == 192 and sum(r['hit'] for r in group) == metric['hits']
    # Exact original point/hit replay of copied baselines, not another parser.
    for r in read_jsonl(base/'evaluation/predictions.jsonl'):
        name = 'v5_'+r['method'] if r['method'].startswith(('full', 'stats')) else r['method']
        if (name, r['sample_id']) in by and name in summary['methods']:
            actual = by[name, r['sample_id']]
            assert actual['point'] == r['point'] and actual['hit'] == r['hit']
    # Independent CPU replay of every new own-decoder point and selected parent.
    replay = {}
    named = [r for r in records if r['metadata']['target_available']]
    for name, entry in frozen['models'].items():
        model, _ = load_field_ranker(out/'models'/entry['file'], 'cpu'); hits = 0; parent_hits = 0
        np.testing.assert_array_equal(model.head_prior.numpy(), expected_prior)
        for start in range(0, len(named), 8):
            group = named[start:start+8]; inputs, _ = make_field_batch(group, 'cpu', labels=False); output = model(**inputs)
            for i, r in enumerate(group):
                sid = r['case']['sample_id']; actual = decode_ranker(output, r, i, top_k=1)[0]; stored = by[name, sid]
                assert actual['point'] == stored['point'] and actual['parent_index'] == stored['selected_parent_index']
                hits += bool(score_point(actual['point'], r['case'])); parent_hits += stored['selected_region_overlap']
        assert hits == summary['methods'][name]['hits'] and parent_hits == summary['methods'][name]['region_top1']
        replay[name] = dict(hits=hits, region_hits=parent_hits); del model
        print('CPU REPLAY '+name+' '+json.dumps(replay[name]), flush=True)
    result = dict(complete=True, same_manifest=True, diagnostic_reuse=True, test_is_fresh=False,
        unchanged_v5_predictions=True, identical_parameter_counts=counts[0], selected_epochs=epochs,
        raw_test_fields_reconstructed=reconstructed, nonpeak_pair_entries=nonpeak, total_pair_entries=total,
        nonpeak_fraction=nonpeak/total, all_point_scores_replayed=len(rows), cpu_gpu_exact_points_agree=True,
        cpu_replay=replay, predictions_sha256=sha256(out/'evaluation/predictions.jsonl'),
        protocol_sha256=sha256(out/'protocol.json'), auditor_sha256=sha256(Path(__file__)))
    write_json(out/'final_audit.json', result); print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__': main()
