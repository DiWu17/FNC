"""Verify completed development outputs without fitting or changing any policy."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'src/clickagent').is_dir())
OUT = ROOT / 'runs/evidence_retention_v14'
POLICIES = ('baseline', 'rank_s1', 'rank_s2', 'aligned_s1', 'aligned_s2')
HEADS = ('c3_top1_s1_loso', 'c3_top1_s2_loso')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    assert read(OUT / 'work_state.json')['complete'], 'Run the main summary first'
    model_hashes = {}
    for i in (1, 2, 3):
        folder = OUT / f'experiment{i}'
        completed = read(folder / 'completion.json')
        assert completed['complete']
        for field, filename in (('protocol_sha256', 'protocol.json'),
                                ('results_sha256', 'results.json'),
                                ('pools_sha256', 'pools.jsonl'),
                                ('details_sha256', 'details.json')):
            if field in completed:
                assert sha(folder / filename) == completed[field], (i, field)
        for name, expected in completed.get('models', {}).items():
            assert sha(OUT / f'models/{name}.pt') == expected, name
            assert model_hashes.setdefault(name, expected) == expected

    folder = OUT / 'experiment3'
    protocol_sha = sha(folder / 'protocol.json')
    protocol = read(folder / 'protocol.json')
    compat = read(folder / 'execution_io_compat.json')
    assert protocol_sha == compat['original_protocol_sha256']
    assert protocol['source_identity'] == compat['original_source_identity']
    for path, expected in compat['corrected_source_identity'].items():
        assert sha(ROOT / path) == expected, path
    assert compat['changes_models_candidates_metrics_or_protocol'] is False
    assert read(folder / 'head_completion.json')['legacy_input_exact_reconstructions'] == 12
    assert read(folder / 'gui/completion.json')['complete']
    assert read(OUT / 'checks/gui_restart_consistency.json')['protocol_unchanged']

    bases = {e['sample_id']: e for e in read(OUT / 'base/index.json')['cases'] if e['split'] == 'val'}
    pools = {r['sample_id']: r for r in map(json.loads, (OUT / 'experiment2/pools.jsonl').read_text(encoding='utf-8').splitlines()) if r['split'] == 'val'}
    jobs = read(folder / 'gui_jobs.json')
    assert jobs['contains_gt_geometry_or_labels'] is False
    assert len(jobs['cases']) == len(bases) == len(pools) == 172
    assert len({j['sample_id'] for j in jobs['cases']}) == 172
    details = {r['sample_id']: r for r in read(folder / 'details.json')}
    summary = read(folder / 'results.json')['summary']
    totals = Counter()
    point_budgets = Counter()
    pixel_budgets = Counter()
    shortlist_budgets = Counter()
    shortlist_pixels = Counter()
    coverage = Counter()
    grid_shapes = Counter()
    total_unique = 0
    for job in jobs['cases']:
        assert set(job) == {'sample_id', 'image', 'image_sha256', 'instruction', 'points', 'split'}
        assert job['split'] == 'val'
        sid = job['sample_id']
        key = hashlib.sha256(sid.encode()).hexdigest()[:24]
        row = pools[sid]
        base = bases[sid]
        assert sha(base['path']) == base['sha256']
        assert sha(base['labels_path']) == base['labels_sha256']
        assert sha(job['image']) == job['image_sha256']
        labels = np.load(base['labels_path'], allow_pickle=False)
        with np.load(base['path'], allow_pickle=False) as z:
            assert z['points'].tolist() == row['points']
            assert set(z['baseline'].tolist()) == set(row['pools']['baseline'])
        universe = row['points']
        y_by_point = {tuple(p): bool(y) for p, y in zip(universe, labels)}
        expected_points = [universe[i] for i in sorted({i for p in POLICIES for i in row['pools'][p]})]
        assert expected_points == job['points']
        case = read(folder / f'gui/cases/{key}.json')
        assert case['complete'] and case['sample_id'] == sid
        assert case['ground_truth_in_model_inputs'] is False
        assert case['protocol_sha256'] == protocol_sha
        identity = hashlib.sha256(json.dumps(job, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        assert case['input_sha256'] == identity
        assert read(folder / f'gui/done/{key}.json')['complete']
        scores = {tuple(s['point']): s for s in case['scores']}
        assert len(scores) == len(case['scores']) == len(job['points'])
        assert set(scores) == {tuple(p) for p in job['points']}
        pixel_by_point = {}
        for point, score in scores.items():
            assert np.isfinite(score['log_odds'])
            assert score['ground_truth_in_model_input'] is False and score['mode'] == 'crop'
            assert score['point'] == score['transform']['point']
            grid = tuple(score['image_grid_thw'])
            assert len(grid) == 3 and all(x > 0 for x in grid)
            # Qwen2-VL patch size is 14; this counts processed image pixels,
            # not merged visual tokens or screenshot-area coordinates.
            pixel_by_point[point] = int(np.prod(grid)) * 14 * 14
            grid_shapes[str(grid)] += 1
        assert len(set(pixel_by_point.values())) <= 1, 'Crop processing budget differs within a query'
        total_unique += len(scores)
        predictions = read(folder / f'heads/{key}.json')
        assert predictions['ground_truth_in_model_inputs'] is False
        assert predictions['protocol_sha256'] == protocol_sha
        for policy in POLICIES:
            indices = sorted(row['pools'][policy])
            assert len(indices) == len(set(indices)) == base['matched_count']
            points = [universe[i] for i in indices]
            point_budgets[policy] += len(points)
            pixel_budgets[policy] += sum(pixel_by_point[tuple(p)] for p in points)
            coverage[policy] += any(y_by_point[tuple(p)] for p in points)
            gui_choice = max(points, key=lambda p: scores[tuple(p)]['log_odds']) if points else None
            for name, choice in [(policy + '__gui', gui_choice)]:
                value = details[sid]['methods'][name]
                correct = bool(choice is not None and y_by_point[tuple(choice)])
                assert value['point'] == choice and value['hit'] == correct
                totals[name] += correct
            for head in HEADS:
                prediction = predictions['predictions'][policy][head]
                assert prediction['points'] == points
                assert np.isfinite(prediction['scores']).all()
                assert len(points) == len(prediction['scores'])
                top1 = points[int(np.argmax(prediction['scores']))] if points else None
                assert top1 == prediction['chosen_point']
                order = sorted(np.argsort(-np.asarray(prediction['scores']), kind='stable')[:4].tolist())
                shortlist = [points[j] for j in order]
                shortname = policy + '__' + head
                shortlist_budgets[shortname] += len(shortlist)
                shortlist_pixels[shortname] += sum(pixel_by_point[tuple(p)] for p in shortlist)
                top4 = max(shortlist, key=lambda p: scores[tuple(p)]['log_odds']) if shortlist else None
                for ending, choice in [('__top1', top1), ('__top4_gui', top4)]:
                    name = shortname + ending
                    value = details[sid]['methods'][name]
                    correct = bool(choice is not None and y_by_point[tuple(choice)])
                    assert value['point'] == choice and value['hit'] == correct
                    if ending == '__top4_gui':
                        assert value['top4_coverage'] == any(y_by_point[tuple(p)] for p in shortlist)
                    totals[name] += correct
    assert total_unique == jobs['point_count'] == 4624
    assert set(point_budgets.values()) == {2607}
    assert len(set(pixel_budgets.values())) == len(set(shortlist_budgets.values())) == len(set(shortlist_pixels.values())) == 1
    assert set(summary) == set(totals)
    assert all(summary[n]['hits'] == hits and summary[n]['n'] == 172 for n, hits in totals.items())
    expected_coverage = read(OUT / 'experiment2/results.json')['val']['matched_coverage']
    assert all(coverage[p] == expected_coverage[p] for p in POLICIES)
    result = dict(complete=True, checked_at_utc=datetime.now(timezone.utc).isoformat(),
                  scope='Audit only. No fitting, policy selection, new points or GUI forwards.',
                  model_sha256=model_hashes, n=172, unique_gui_case_points=total_unique,
                  per_policy_point_slots=dict(point_budgets), processed_pixel_budgets=dict(pixel_budgets),
                  old_head_top4_point_slots=dict(shortlist_budgets), old_head_top4_processed_pixels=dict(shortlist_pixels),
                  image_grid_counts=dict(grid_shapes), reconstructed_hits=dict(totals),
                  coverage=dict(coverage), all_public_input_identities_match=True,
                  labels_separate_and_gt_absent_from_model_inputs=True,
                  primary_protocol_unchanged_after_io_repair=True,
                  legacy_rehash_receipt='experiment3/completion.json',
                  caveat='Shared counterfactual scores; actual lab time includes failed I/O attempt and repeated anchors. No per-policy speed claim.')
    dest = OUT / 'checks/final_output_audit.json'
    dest.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    archive = OUT / 'analysis/audit_outputs_source.py'
    archive.write_bytes(Path(__file__).read_bytes())
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
