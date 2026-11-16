"""Post-hoc CPU review of V11 generalization explanations; no fitting/inference.

Reads completed profiles/predictions, calculates exact expectations under
uniform random candidate selection, and a worst-case empirical source-mix
reweighting bound. These are exploratory controls added after test inspection.
Does not change frozen sources, predictions, checkpoints or primary reports.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/candidate_scorer_v11'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def choose(n, k):
    return math.comb(n, k) if n >= 0 and 0 <= k <= n else 0


def distribution(rows):
    covered = [v for v in rows if v['positives'] > 0]
    return dict(n=len(rows), covered=len(covered),
        quota4_retention_given_k16_covered=sum(v['quota4_covered'] for v in covered)/len(covered),
        positive_count_given_covered=dict(Counter(v['positives'] for v in covered)),
        mean_positive_count_given_covered=statistics.mean(v['positives'] for v in covered),
        uniform_top1_accuracy_given_covered=statistics.mean(v['positives']/v['n_points'] for v in covered),
        covered_by_source=dict(Counter(v['dataset'] for v in covered)))


def main():
    inputs = [OUT / p for p in ['trainval_profile.json', 'manifest.jsonl',
        'evaluation/test/details.json', 'evaluation/test/summary.json',
        'evaluation/compression/analysis.json', 'models/internal_full_s1.json',
        'trainval/runtime.json', 'test/runtime.json']]
    identities = {str(p): sha(p) for p in inputs}
    profile, details = read(inputs[0]), read(inputs[2])
    primary, compression = read(inputs[5]), read(inputs[4])
    rows = {}
    for split in ('train', 'val'):
        rows[split] = [dict(sample_id=c['sample_id'], dataset=c['dataset'],
            n_points=c['pools']['16']['n_points'],
            quota4_covered=c['pools']['4']['point_oracle'],
            positives=c['pools']['16']['n_positive_points'])
            for c in profile['cases'] if c['split'] == split]
    rows['test'] = [dict(sample_id=c['sample_id'], dataset=c['dataset'],
        quota4_covered=c['pools']['4']['point_oracle'],
        n_points=c['pools']['16']['n_points'], positives=sum(c['pools']['16']['point_hits'])) for c in details]
    distributions = {split: distribution(values) for split, values in rows.items()}
    # Maximize reweighted empirical training error over every source allocation
    # consistent with the known aggregate 39 training mistakes. This requires
    # no invented per-case training predictions and is not a confidence bound.
    train_counts = distributions['train']['covered_by_source']
    val_counts = distributions['val']['covered_by_source']
    assert set(train_counts) == set(val_counts)
    errors = primary['selected_training_evaluation']['oracle'] - primary['selected_training_evaluation']['hits']
    remaining, weighted_error, allocation = errors, 0., []
    ordered = sorted(train_counts, key=lambda s: val_counts[s]/sum(val_counts.values())/train_counts[s], reverse=True)
    for source in ordered:
        weight = val_counts[source]/sum(val_counts.values())/train_counts[source]
        take = min(remaining, train_counts[source]); remaining -= take
        weighted_error += take*weight
        allocation.append(dict(source=source, train_covered=train_counts[source], val_covered=val_counts[source],
            allocated_training_errors=take, weight_per_error=weight))
    assert remaining == 0
    random_top1 = random_four_oracle = random_four_gui = 0.
    for case in details:
        pool = case['pools']['16']; n, positives = pool['n_points'], sum(pool['point_hits'])
        if not n:
            continue
        k = min(4, n); denominator = choose(n, k)
        random_top1 += positives/n
        random_four_oracle += 1-choose(n-positives, k)/denominator
        stem = hashlib.sha256(case['sample_id'].encode()).hexdigest()[:24]
        path = OUT / 'gui_reference/test' / (stem+'.json')
        identities[str(path)] = sha(path); gui = read(path)
        order = sorted(range(n), key=lambda i: (-gui['records'][','.join(map(str, pool['points'][i]))]['log_odds'], i))
        random_four_gui += sum(pool['point_hits'][index]*choose(n-rank, k-1)/denominator
            for rank, index in enumerate(order, 1))
    runtime_equal = read(inputs[6]) == read(inputs[7])
    manifest = [json.loads(line) for line in inputs[1].read_text(encoding='utf-8').splitlines() if line.strip()]
    meta_keys = ('head_order', 'image_content_layer', 'image_content_source',
                 'vision_content_source', 'own_token_forward', 'system_prompt', 'all_backbone_parameters_frozen')
    sampled, first = [], None
    # One deterministic first manifest query per split/source, both views.
    groups = {}
    for c in manifest:
        groups.setdefault((c['experiment_split'], c['dataset']), c)
    for (split, dataset), c in sorted(groups.items()):
        stem = hashlib.sha256(c['sample_id'].encode()).hexdigest()[:24]
        pipeline_path = OUT / ('test' if split == 'test' else 'trainval') / 'pipeline' / (stem+'.json')
        identities[str(pipeline_path)] = sha(pipeline_path); pipeline = read(pipeline_path)
        for view in ('global', 'local'):
            path = pipeline[view+'_feature_path']
            if path is None:
                continue
            path = ROOT / path if not Path(path).is_absolute() else Path(path)
            with np.load(path, allow_pickle=False) as archive:
                meta = json.loads(str(archive['metadata']))
            signature = {name: meta[name] for name in meta_keys}
            if first is None:
                first = signature
            assert signature == first and meta['ground_truth_in_model_input'] is False
            sampled.append(dict(split=split, dataset=dataset, view=view, path=str(path)))
    actual = compression['cohorts']['test']
    result = dict(complete=True, classification='exploratory controls added after test inspection',
        computed_at_utc=datetime.now(timezone.utc).isoformat(),
        no_training_or_model_inference=True, no_primary_artifact_changes=True,
        distributions=distributions,
        source_reweighting=dict(training_errors=errors, minimum_reweighted_empirical_train_accuracy=1-weighted_error,
            actual_validation_accuracy=primary['selected_validation_replay']['hits']/primary['selected_validation_replay']['oracle'],
            allocation=allocation, interpretation='Broad source mixture alone cannot produce the observed empirical gap; does not control within-source difficulty, target type or proposal shift. Not a statistical confidence bound.'),
        random_selection=dict(n=192, rule='Uniform subset of min(4,n) distinct original K16 points; same original GUI logits and original-order ties.',
            uniform_top1_expected_hits=random_top1, uniform_top4_expected_oracle=random_four_oracle,
            uniform_top4_then_gui_expected_hits=random_four_gui,
            internal_s1_top4_oracle=actual['methods']['internal_full_s1_K16_top4_gui']['point_oracle'],
            internal_s1_top4_gui_hits=actual['methods']['internal_full_s1_K16_top4_gui']['hits'],
            interpretation='Exact randomization expectations, not another sampled model run, significance test or latency measurement.'),
        feature_contract_check=dict(trainval_local_runtime_equals_test_runtime=runtime_equal,
            sampled_archives=len(sampled), sampled=sampled, checked_signature=first,
            interpretation='Metadata-contract sampling only; does not establish numerical equality across different screenshots or exclude all extraction bugs.'),
        input_sha256=identities, script_sha256=sha(Path(__file__)))
    destination = OUT / 'evaluation/generalization_review'; destination.mkdir(exist_ok=True)
    (destination/'diagnostics.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('distributions', 'source_reweighting', 'random_selection')}, ensure_ascii=False))
    print('FEATURE CONTRACTS', len(sampled), 'MATCH; RUNTIME EXACT MATCH', runtime_equal)


if __name__ == '__main__':
    main()
