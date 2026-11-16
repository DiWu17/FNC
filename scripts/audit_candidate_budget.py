"""Replay every budget choice, quota, marker, common score and metric without VLM reruns."""
from pathlib import Path
import json
import os

import numpy as np
from PIL import Image

from clickagent.action_benchmark import sha256, write_json
from clickagent.candidate_budget import BUDGETS, build_pools, coordinate_key
from clickagent.point_aligned_inference import marked_crop
from evaluate_candidate_budget import evaluate, hit
from point_aligned_data import read
from run_candidate_budget import OUT, BASE, assert_seal, cases
from run_target_token_experiment import key, read_jsonl


def main():
    source_sha = sha256(Path(__file__))
    assert_seal()
    frozen = read(OUT / 'frozen.json');assert frozen['scored_cases_at_freeze'] == 0
    assert frozen['identity_sha256'] == sha256(OUT / 'identity.json')
    sealed = {r['sample_id']: r for r in read(OUT / 'sealed_pools.json')}
    record_count = 0;batch_count = 0;mass = [];cache_hashes = {}
    rows = cases();assert len(rows) == len({c['pixel_sha256'] for c in rows}) == 192
    for c in rows:
        sid = key(c['sample_id']);pipeline = read(BASE / 'pipeline' / f'{sid}.json')
        pools = build_pools(pipeline)
        assert pools == sealed[c['sample_id']]['pools']
        assert [v['point'] for v in pools['4']] == [v['point'] for v in pipeline['pools']['fixed_pool']]
        assert sha256(Path(c['image'])) == c['image_sha256']
        with Image.open(c['image']) as image:
            image = image.convert('RGB');assert list(image.size) == c['image_size']
            path = OUT / 'verifier' / f'{sid}.json';saved = read(path)
            assert os.path.getmtime(path) >= frozen['frozen_at']
            assert saved['complete'] and not saved['ground_truth_in_model_inputs']
            assert saved['original_image_sha256'] == c['image_sha256']
            points = [v['point'] for v in pools['32']]
            assert set(saved['records']) == {coordinate_key(p) for p in points}
            assert [v['start'] for v in saved['batches']] == list(range(0, len(points), 6))
            for batch in saved['batches']:
                expected = points[batch['start']:batch['start'] + 6]
                assert batch['points'] == expected and np.isfinite(batch['seconds']) and batch['seconds'] >= 0
                for point in expected:
                    r = saved['records'][coordinate_key(point)]
                    assert r['point'] == point and r['mode'] == 'crop' and not r['ground_truth_in_model_input']
                    assert r['transform'] == marked_crop(image, point)[1]
                    assert r['batch_size'] == len(expected) and r['batch_elapsed_seconds'] == batch['seconds']
                    assert r['elapsed_seconds_per_point'] == batch['seconds'] / len(expected)
                    assert abs(r['log_odds'] - (r['true_logit'] - r['false_logit'])) <= 1e-5
                    assert np.isfinite([r['true_logit'], r['false_logit'], r['log_odds'], r['label_mass']]).all()
                    assert 0 <= r['label_mass'] <= 1 + 1e-4
                    grid = np.asarray(r['image_grid_thw']);assert grid.shape == (3,) and grid[0] == 1
                    assert np.prod(grid) * 14 * 14 <= 262144
                    mass.append(r['label_mass']);record_count += 1
                batch_count += 1
            previous = set()
            for budget in BUDGETS:
                point_set = {tuple(v['point']) for v in pools[str(budget)]}
                assert previous <= point_set and len(point_set) <= budget
                previous = point_set
            cache_hashes[str(path)] = sha256(path)
    runtime = read(OUT / 'runtime.json')
    assert runtime['trainable_parameters'] == 0 and runtime['max_pixels'] == 262144 and runtime['fixed_max_batch_size'] == 6
    archived = read_jsonl(OUT / 'evaluation/predictions.jsonl')
    replay, details, summary = evaluate()
    assert archived == replay
    assert read(OUT / 'evaluation/details.json') == details
    # JSON object keys are strings: candidate_size_counts originates from an
    # integer-key Counter. Canonicalize keys, preserving all numerical values.
    assert read(OUT / 'evaluation/summary.json') == json.loads(json.dumps(summary))
    by_id = {c['sample_id']: c for c in rows}
    assert len(archived) == 7 * 192 == 1344
    for row in archived:assert row['hit'] == hit(row['point'], by_id[row['sample_id']])
    assert record_count == summary['actual_common_cache_compute']['image_views']
    assert batch_count == summary['actual_common_cache_compute']['batch_calls']
    assert source_sha == sha256(Path(__file__))
    assert_seal()
    result = dict(passed=True, n=192, prediction_rows_replayed=len(replay), score_records=record_count,
        actual_batches=batch_count, all_K_share_identical_point_scores=True, pools_nested=True,
        K4_coordinates_match_v9=True, legacy_K4_hits=summary['methods']['legacy_K4']['hits'],
        summary_replay='exact JSON values, including canonicalized count-dictionary keys',
        min_true_false_label_mass=min(mass), median_true_false_label_mass=float(np.median(mass)),
        full_vlm_rerun=False, source_sha256=source_sha, score_cache_sha256=cache_hashes,
        limitation='Archived VLM logits and image provenance checked; no independent VLM forward replay. Reused screenshots, exploratory only.')
    write_json(OUT / 'evaluation/audit.json', result)
    print(json.dumps({k:v for k,v in result.items() if k != 'score_cache_sha256'}), flush=True)


if __name__ == '__main__':main()
