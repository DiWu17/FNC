"""V13 raw, label-free multi-view scorer inputs; train/val labels stored separately."""
from pathlib import Path
import argparse
import hashlib
import json
import time
import numpy as np

from clickagent.action_benchmark import sha256, write_json
from clickagent.multiregion_scorer import build_multiregion_inputs
from clickagent.public_evaluation import point_in_geometry
from cache_candidate_scorer import raw_arrays, save_npz
from crossfit_protocol import key
from multiregion_v13_protocol import OUT, BUDGETS, cases, read, assert_sealed


def cache(split, only_complete=False):
    if split not in ('train', 'val'):
        assert_sealed()
    entries = {'c1': [], 'c3': []}; pending = 0; started = time.perf_counter()
    for i, case in enumerate(cases(split), 1):
        sid = key(case['sample_id']); pipeline_path = OUT/'pipeline'/split/(sid+'.json')
        if not pipeline_path.exists():
            if only_complete:
                pending += 1; continue
            raise FileNotFoundError(pipeline_path)
        pipeline = read(pipeline_path)
        assert sha256(Path(pipeline['global_feature_path'])) == pipeline['global_feature_sha256']
        for view in pipeline['local_views']:
            assert sha256(Path(view['feature_path'])) == view['feature_sha256']
        ga, gm = raw_arrays(pipeline['global_feature_path'])
        all_locals = [(raw_arrays(v['feature_path']), v['transform']) for v in pipeline['local_views']]
        for condition in entries:
            path = OUT/'cache'/split/condition/(sid+'.npz')
            pools = pipeline['pools'][condition]
            points = np.asarray([r['point'] for r in pools['32']], np.int64).reshape(-1, 2)
            local_views = all_locals[:1] if condition == 'c1' else all_locals
            files = [pipeline['global_feature_path']]+[v['feature_path'] for v in pipeline['local_views'][:1 if condition == 'c1' else 3]]
            evidence = {str(pipeline_path): sha256(pipeline_path)}
            evidence.update({str(p): sha256(Path(p)) for p in files})
            lookup = {tuple(p): j for j, p in enumerate(points.tolist())}
            expected_indices = {str(k): np.asarray([lookup[tuple(p['point'])] for p in pools[str(k)]], np.int64) for k in BUDGETS}
            labels = (np.asarray([point_in_geometry((p/np.asarray(case['image_size'])).tolist(), case['target_geometry'])
                                 for p in points], np.float32) if split in ('train', 'val') else None)
            fingerprint = hashlib.sha256(json.dumps(dict(sample_id=case['sample_id'], condition=condition,
                image_size=case['image_size'], image_sha256=case['image_sha256'], instruction=case['instruction'],
                builder_sources={str(p): sha256(p) for p in (Path(__file__), Path('src/clickagent/multiregion_scorer.py'),
                    Path('src/clickagent/internal_candidate_scorer.py'))},
                evidence=evidence, geometry=case['target_geometry'] if labels is not None else None),
                sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            if not path.exists():
                arrays = dict(sample_id=np.asarray(case['sample_id']), available=np.asarray(bool(len(points))),
                    points=points, pipeline_sha256=np.asarray(sha256(pipeline_path)),
                    cache_fingerprint=np.asarray(fingerprint), annotation_derived_coordinates=np.asarray(False))
                for budget in BUDGETS:
                    arrays['pool_indices_'+str(budget)] = expected_indices[str(budget)]
                if labels is not None:
                    arrays['labels'] = labels
                if len(points):
                    inputs = build_multiregion_inputs(ga, gm, [(a, m, tr) for (a, m), tr in local_views], points, case['image_size'])
                    arrays.update({'i__'+name: value for name, value in inputs.items()})
                save_npz(path, arrays)
            with np.load(path, allow_pickle=False) as z:
                assert str(z['sample_id']) == case['sample_id']
                assert str(z['pipeline_sha256']) == sha256(pipeline_path)
                assert str(z['cache_fingerprint']) == fingerprint, 'Cached feature input identity changed'
                assert ('labels' in z.files) == (split in ('train', 'val'))
                assert np.array_equal(z['points'], points)
                assert bool(z['available']) == bool(len(points)) and not bool(z['annotation_derived_coordinates'])
                for budget in BUDGETS:
                    assert np.array_equal(z['pool_indices_'+str(budget)], expected_indices[str(budget)])
                if labels is not None:
                    assert np.array_equal(z['labels'], labels)
            entries[condition].append(dict(sample_id=case['sample_id'], path=str(path), sha256=sha256(path),
                available=bool(len(points)), n_points=len(points), evidence=evidence))
        if i % 25 == 0:
            print(json.dumps(dict(stage='cache', split=split, i=i, n=len(cases(split)), seconds=time.perf_counter()-started)), flush=True)
    for condition, records in entries.items():
        write_json(OUT/'cache'/f'{split}_{condition}_index.json', dict(complete=pending == 0, pending=pending,
            split=split, condition=condition, cases=records, labels_present=split in ('train', 'val'),
            annotation_derived_coordinates=False, elapsed_seconds=time.perf_counter()-started))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--split', required=True)
    parser.add_argument('--only-complete', action='store_true'); args = parser.parse_args()
    cache(args.split, args.only_complete)
