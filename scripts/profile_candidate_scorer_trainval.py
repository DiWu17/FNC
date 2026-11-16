"""CPU-only descriptive V11 train/validation snapshot; no head/test execution.

The V11 fresh manifest is never opened. Old train/val rows come exclusively
from the hash-bound retained v7 source. Attention/query NPY headers supply
actual token counts without decompressing full feature arrays. A fixed list
of available local archives/pipelines is captured before profiling begins.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import zipfile

import numpy as np
from numpy.lib import format as npformat

from clickagent.public_evaluation import point_in_geometry


ROOT = Path(__file__).resolve().parents[1]
BUDGETS = (4, 8, 16, 32)
SPLIT_PATTERN = re.compile(r'(?<!\\)"experiment_split"\s*:\s*"(train|val)"')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def key(value):
    return hashlib.sha256(value.encode()).hexdigest()[:24]


def utc():
    return datetime.now(timezone.utc).isoformat()


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT/path


def stat_identity(path):
    s = Path(path).stat()
    return dict(path=str(Path(path).resolve()), bytes=s.st_size, mtime_ns=s.st_mtime_ns)


def snapshot(folder, suffix):
    return {p.stem: stat_identity(p) for p in sorted(folder.glob('*'+suffix)) if p.is_file()}


def header(archive, name):
    with archive.open(name+'.npy') as stream:
        version = npformat.read_magic(stream)
        require(version in ((1, 0), (2, 0)), f'Unexpected NPY header version: {version}')
        shape, _, dtype = (npformat.read_array_header_1_0(stream) if version == (1, 0)
                           else npformat.read_array_header_2_0(stream))
    return shape, str(dtype)


def token_information(identity, case):
    path = Path(identity['path'])
    require(stat_identity(path) == identity, 'Archive changed since snapshot')
    with zipfile.ZipFile(path) as archive:
        attention, adtype = header(archive, 'target_attention')
        query, qdtype = header(archive, 'target_hidden')
        require(len(attention) == 3 and attention[1] == 96, 'Expected full T x96 xP archive')
        require(query == (attention[0], 12288), 'Query token count differs from attention')
        with archive.open('metadata.npy') as stream:
            meta = json.loads(str(npformat.read_array(stream, allow_pickle=False)))
    require(meta['sample_id'] == case['sample_id'] and meta['image_sha256'] == case['image_sha256'], 'Raw identity mismatch')
    require(meta['own_token_forward'] and meta['ground_truth_in_model_input'] is False, 'Unexpected feature source')
    require(bool(meta['target_available']) == bool(attention[0]), 'Name availability/token count mismatch')
    require(stat_identity(path) == identity, 'Archive changed during read')
    t = int(attention[0])
    return dict(tokens=t, category='no_name' if t == 0 else 'single_token' if t == 1 else 'multiple_tokens',
                target_available=bool(meta['target_available']), grid_shape=meta['grid_shape'],
                attention_shape=list(attention), attention_dtype=adtype, query_dtype=qdtype,
                identity=identity)


def distribution(values):
    values = list(map(int, values)); a = np.asarray(values, np.float64)
    return dict(n=len(values), counts={str(k):v for k, v in sorted(Counter(values).items())},
                mean=float(a.mean()) if len(a) else None, std=float(a.std()) if len(a) else None,
                min=int(a.min()) if len(a) else None, median=float(np.median(a)) if len(a) else None,
                p95=float(np.quantile(a, .95)) if len(a) else None, max=int(a.max()) if len(a) else None)


def length_summary(records, key_name):
    values = [r[key_name] for r in records if r[key_name] is not None]
    lengths = [v['tokens'] for v in values]; categories = Counter(v['category'] for v in values)
    return dict(n_observed=len(values), n_unobserved=len(records)-len(values),
        categories={k: categories[k] for k in ('no_name', 'single_token', 'multiple_tokens')},
        fractions_over_observed={k: categories[k]/len(values) if values else None
                                for k in ('no_name', 'single_token', 'multiple_tokens')},
        lengths_including_no_name=distribution(lengths), named_lengths=distribution(t for t in lengths if t > 0))


def pool_statistics(case, pipeline, global_info, local_info):
    output = {}; previous = set()
    size = np.asarray(case['image_size'])
    transform = pipeline.get('old_crop_transform')
    for budget in BUDGETS:
        pool = pipeline['pools'][str(budget)]
        points = np.asarray([v['point'] for v in pool], np.float64).reshape(-1, 2)
        require(len(points) <= budget and np.isfinite(points).all(), 'Invalid pool size/point')
        require(bool(((points == np.rint(points)) & (points >= 0) & (points < size)).all()), 'Invalid integer point')
        current = {tuple(p) for p in points.tolist()}
        require(len(current) == len(points) and previous <= current, 'Pools must be nested after exact dedup')
        previous = current
        hits = [bool(point_in_geometry((point/size).tolist(), case['target_geometry'])) for point in points]
        inside = np.zeros(len(points), bool)
        if transform is not None and local_info is not None and local_info['tokens'] > 0:
            box = np.asarray(transform['box'])
            inside = ((points >= box[:2]) & (points < box[2:])).all(-1)
        eligible = (np.ones(len(points), bool) if global_info['tokens'] > 1 else
                    inside if local_info is not None and local_info['tokens'] > 1 else np.zeros(len(points), bool))
        output[str(budget)] = dict(n_points=len(points), n_positive_points=sum(hits), point_oracle=any(hits),
            n_points_with_available_local_state=int(inside.sum()),
            n_points_with_multitoken_state=int(eligible.sum()),
            has_multitoken_candidate=bool(eligible.any()),
            positive_fraction=sum(hits)/len(points) if len(points) else None)
    return output


def split_summary(records):
    completed = [r for r in records if r['pipeline_completed']]
    pools = {}
    for budget in BUDGETS:
        values = [r['pools'][str(budget)] for r in completed]
        n = len(values); nonempty = [v for v in values if v['n_points'] > 0]
        covered = [v for v in values if v['point_oracle']]
        pools[str(budget)] = dict(n_observed_pipelines=n, n_unobserved_pipelines=len(records)-n,
            n_empty=sum(v['n_points'] == 0 for v in values), n_nonempty=len(nonempty),
            point_oracle=sum(v['point_oracle'] for v in values),
            point_oracle_fraction_over_observed=sum(v['point_oracle'] for v in values)/n if n else None,
            total_points=sum(v['n_points'] for v in values),
            total_positive_points=sum(v['n_positive_points'] for v in values),
            n_pools_with_multiple_positive_points=sum(v['n_positive_points'] > 1 for v in values),
            actual_point_count=distribution(v['n_points'] for v in values),
            positive_point_count_including_empty=distribution(v['n_positive_points'] for v in values),
            positive_point_count_given_covered=distribution(v['n_positive_points'] for v in covered),
            total_points_with_local_state=sum(v['n_points_with_available_local_state'] for v in values),
            total_points_with_multitoken_state=sum(v['n_points_with_multitoken_state'] for v in values),
            n_pipelines_with_any_multitoken_candidate=sum(v['has_multitoken_candidate'] for v in values),
            n_pipelines_without_multitoken_candidates=sum(not v['has_multitoken_candidate'] for v in values))
    joint = Counter(('global_'+r['global']['category']+'__local_'+(r['local']['category'] if r['local'] else 'not_observed')) for r in completed)
    return dict(n_manifest=len(records), n_completed_pipelines=len(completed),
        pipeline_completion_fraction=len(completed)/len(records),
        global_target_tokens=length_summary(records, 'global'), local_target_tokens=length_summary(records, 'local'),
        local_expected_if_global_named=sum(r['global']['tokens'] > 0 for r in records),
        local_failure_reasons=dict(Counter(r['local_unobserved_reason'] for r in records if r['local'] is None)),
        completed_pipeline_global_local_categories=dict(sorted(joint.items())), pools=pools)


def source_identity():
    protocol = ROOT/'scripts/candidate_scorer_protocol.py'
    nodes = ast.parse(protocol.read_text(encoding='utf-8')).body
    sources = next(ast.literal_eval(node.value) for node in nodes if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == 'SOURCES' for target in node.targets))
    return {str(resolve(path).resolve()): sha(resolve(path)) for path in sources}


def profile(out):
    started = utc(); sources = source_identity()
    selection_path = out/'data_selection.json'; selection = read(selection_path)
    require(selection['complete'] and selection['retained_train_val_exact_unchanged'], 'Expected unchanged retained old train/val')
    retained = resolve(selection['retained_train_val_source'])
    require(sha(retained) == selection['retained_source_sha256'], 'Retained old manifest identity changed')
    # Do not open V11 manifest, test features, test predictions or new test GT.
    cases = []
    for line in retained.open(encoding='utf-8-sig'):
        if SPLIT_PATTERN.search(line):
            case = json.loads(line)
            require(case['experiment_split'] in ('train', 'val'), 'Scope violation')
            cases.append(case)
    require(Counter(c['experiment_split'] for c in cases) ==
            {s:selection['counts'][s] for s in ('train', 'val')}, 'Retained split counts differ')
    local_snapshot = snapshot(out/'trainval/old_crop_features', '.npz')
    pipeline_snapshot = snapshot(out/'trainval/pipeline', '.json')
    records = []; errors = []
    for i, case in enumerate(cases, 1):
        stem = key(case['sample_id'])
        raw = ROOT/'runs/semantic_head_v7/features'/(stem+'.npz')
        if not raw.exists():
            raw = ROOT/'runs/conditioned_v3/features'/(stem+'.npz')
        info = token_information(stat_identity(raw), case)
        row = dict(sample_id=case['sample_id'], split=case['experiment_split'], dataset=case['dataset'],
            global_=info, local=None, local_unobserved_reason=None, pipeline_completed=False, pools=None)
        row['global'] = row.pop('global_')
        if stem in local_snapshot:
            try:
                row['local'] = token_information(local_snapshot[stem], case)
            except (OSError, EOFError, zipfile.BadZipFile, ValueError) as error:
                row['local_unobserved_reason'] = 'archive_unfinished_or_changed_at_snapshot'
                errors.append(dict(sample_id=case['sample_id'], stage='local_archive', reason=str(error)))
        else:
            row['local_unobserved_reason'] = ('not_expected_without_global_name' if info['tokens'] == 0
                                              else 'not_yet_available_at_snapshot')
        if stem in pipeline_snapshot:
            identity = pipeline_snapshot[stem]; path = Path(identity['path'])
            try:
                require(stat_identity(path) == identity, 'Pipeline changed since snapshot')
                pipeline = read(path)
                require(pipeline['sample_id'] == case['sample_id'], 'Pipeline sample mismatch')
                require(set(pipeline['pools']) == {str(k) for k in BUDGETS}, 'Pipeline pool keys incomplete')
                require(stat_identity(path) == identity, 'Pipeline changed during read')
                # A completed pipeline references a completed local archive.
                # If the fixed initial archive snapshot preceded that file,
                # classify this pipeline as pending rather than mixing times.
                if pipeline.get('local_feature_path') and row['local'] is None:
                    raise ValueError('Local archive not observed in fixed initial snapshot')
                row['pools'] = pool_statistics(case, pipeline, info, row['local'])
                row['pipeline_completed'] = True
                row['pipeline_identity'] = dict(identity, sha256=sha(path))
            except (OSError, ValueError, KeyError) as error:
                errors.append(dict(sample_id=case['sample_id'], stage='pipeline', reason=str(error)))
        records.append(row)
        if i % 200 == 0:
            print(f'PROFILE OLD TRAINVAL {i}/{len(cases)} CPU headers only', flush=True)
    require(sources == source_identity(), 'SOURCES changed during profiling; rerun snapshot')
    summaries = {s:split_summary([r for r in records if r['split'] == s]) for s in ('train', 'val')}
    return dict(complete=True, scope='old_train_validation_descriptive_snapshot_only', started_utc=started,
        finished_utc=utc(), collection_complete=all(v['n_completed_pipelines'] == v['n_manifest'] for v in summaries.values()),
        sources_unchanged=True, sources_sha256=sources, script_sha256=sha(Path(__file__)),
        retained_manifest=str(retained.resolve()), retained_manifest_sha256=sha(retained),
        data_selection_sha256=sha(selection_path),
        read_scope=dict(v11_fresh_manifest_opened=False, test_labels_read=False, test_features_read=False,
                        trained_models_loaded=False, gpu_used=False, scorer_executed=False,
                        labels_used_only='old train/val actual proposal point coverage; no fitting/selection'),
        snapshot=dict(local_archive_paths_at_start=len(local_snapshot), pipeline_paths_at_start=len(pipeline_snapshot),
                      archive_identity_level='NPY headers + model metadata + size/mtime; full weight/feature payloads not rehashed',
                      fixed_file_lists=True, extraction_inflight_files_not_assumed_complete=True),
        interpretation=[
            'Global length is the actual generated-name own-token sequence saved in target_attention/target_hidden, not word/character count or all reply tokens.',
            'Full vs pooled directly changes token evidence only for views with T>1. A pool can still contain locally multi-token candidates when its global name is one token.',
            'With identical weights/eval inputs, all contributing views T<=1 leave no sequence-content difference for early token pooling; separately trained variants can still differ due to training elsewhere.',
            'Token count, pooling eligibility and candidate oracle are descriptive limits, not evidence that full will outperform pooled.',
            'Current pipeline/local subsets follow acquisition progress, not a random sample; observed fractions use completed pipelines only, never label unobserved samples as failures.',
            'Positive count describes points, not independent UI targets; several decoded points may fall inside the same target.',
            'No source/model/candidate/optimization/selection policy was changed; rerun this script after collection for final denominators.',
        ], splits=summaries, inflight_or_invalid_records=errors, cases=records)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'runs/candidate_scorer_v11')
    parser.add_argument('--json', type=Path)
    args = parser.parse_args(argv); out = args.output.resolve()
    result = profile(out)
    dest = args.json.resolve() if args.json else out/'trainval_profile.json'
    dest.parent.mkdir(parents=True, exist_ok=True); temporary = dest.with_suffix(dest.suffix+'.tmp')
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(dest)
    print('PROFILE '+str(dest), flush=True)
    print(json.dumps(dict(collection_complete=result['collection_complete'], splits=result['splits'],
                          inflight_exclusions=len(result['inflight_or_invalid_records'])), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
