"""Pre-registered, three-cohort analysis of sealed paper-scale predictions.

Registration and gate checks use public inputs, model identities and prediction
receipts only. The official ScreenSpot-Pro and PointerBench label files are
opened exclusively after all 2,581 per-case predictions have passed the gate.
No checkpoint selection, fitting, feature extraction or inference occurs here.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'src')]

import numpy as np

import run_paper_scale_eval_v1 as engine
from clickagent.public_evaluation import point_in_geometry

RUN = ROOT / 'runs/paper_scale_ablation_v2'
EVAL = RUN / 'eval'
POINTER = RUN / 'pointerbench'
OUT = RUN / 'analysis'
REGISTRATION = RUN / 'statistical_analysis_protocol.json'
FULL_PROTOCOL = RUN / 'protocol.json'
POINTER_PROTOCOL = POINTER / 'evaluation_protocol_v2.json'
SUPERSEDED_POINTER_PROTOCOL = POINTER / 'evaluation_protocol.json'
SUPERSEDED_INCIDENT = POINTER / 'preseal_label_access_incident.json'
INCIDENT = POINTER / 'preseal_label_access_addendum_v2.json'
FULL_RECEIPT = RUN / 'public_completion.json'
POINTER_RECEIPT = POINTER / 'public_completion.json'
FULL_LABELS = ROOT / 'data/public_v1/manifests/screenspot_pro.jsonl'
POINTER_LABELS = POINTER / 'labels.jsonl'
HISTORY = ROOT / 'runs/paper_scale_ablation_v1/data/full_history_overlap_corrected_v2.json'
CORRECTION = ROOT / 'runs/paper_scale_ablation_v1/data/exposure_correction_v2.json'
POST_V32 = ROOT / 'runs/paper_scale_ablation_v1/data/post_v32_exposure_addendum_v1.json'
ERRATUM = ROOT / 'research/paper/SCALE_ABLATION_ERRATUM_V2.md'
MODEL_REGISTRY = ROOT / 'runs/paper_scale_ablation_v1/models/frozen.json'
COHORTS = ('pro', 'sheets', 'full')
SIZES = {'pro': 500, 'sheets': 500, 'full': 1581}
VARIANTS = ('aligned', 'no_evidence', 'spatial_shuffle', 'attention_only',
            'hidden_only', 'token_mean')
SEEDS = (1, 2, 3)
PRIMARY_CONTRASTS = tuple(f'{scale}_aligned_vs_{variant}'
                          for scale in ('3k', 'full')
                          for variant in ('no_evidence', 'spatial_shuffle'))
HISTORY_COUNTS = {
    'pre_v32_historical_exact': 1063,
    'pre_v32_historical_near_le4': 344,
    'v18_confirmation_observed_exact_rgb': 140,
    'v18_confirmation_observed_near_le4': 1,
    'v32_diagnostic_observed': 24,
    'v32_reserve_previously_unforwarded': 9,
}
REPLICATES = 10000
BOOTSTRAP_SEED = 20261111


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def jsonl(path):
    with Path(path).open('r', encoding='utf-8-sig') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_once(path, value):
    path = Path(path)
    if path.exists():
        require(read(path) == value, 'Existing immutable output differs: ' + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    require(not temporary.exists(), 'Unresolved temporary output: ' + str(temporary))
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n',
                         encoding='utf-8')
    temporary.replace(path)


def text_once(path, content):
    path = Path(path)
    if path.exists():
        require(path.read_text(encoding='utf-8') == content,
                'Existing immutable readout differs: ' + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    require(not temporary.exists(), 'Unresolved temporary output: ' + str(temporary))
    temporary.write_text(content, encoding='utf-8')
    temporary.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def model_hashes():
    registry = engine.models(False)
    names = {f'{scale}_{variant}_s{seed}'
             for scale in ('3k', 'full') for variant in VARIANTS for seed in SEEDS}
    names.update(f'matched3k_aligned_s{seed}' for seed in SEEDS)
    require(set(registry) == names and len(registry) == 39,
            'Frozen model table differs from 3k/full six arms plus matched3k')
    return {name: entry['sha256'] for name, entry in registry.items()}


def verify_incident(incident, pointer_protocol=None):
    require(incident['complete'] is True and
            incident['supersedes_incident_sha256'] == sha(SUPERSEDED_INCIDENT) and
            incident['supersedes_pointer_protocol_sha256'] ==
            sha(SUPERSEDED_POINTER_PROTOCOL) and
            incident['identified_distinct_examples_at_least'] >= 7 and
            len(set(incident['identified_previewed_examples_minimum'])) >= 7 and
            incident['exhaustive_count_known'] is False and
            incident['evaluation_pipeline_label_access_before_prediction_seal'] is False and
            incident['reported_pointer_model_forward_before_addendum'] is False and
            incident['reported_test_scoring_before_addendum'] is False,
            'Preseal human/agent label access addendum changed')
    if pointer_protocol is not None:
        require(pointer_protocol['preseal_label_access_incident_sha256'] == sha(INCIDENT) and
                pointer_protocol['superseded_incident_sha256'] == sha(SUPERSEDED_INCIDENT) and
                pointer_protocol['superseded_pointer_protocol_sha256'] ==
                sha(SUPERSEDED_POINTER_PROTOCOL) and
                pointer_protocol['at_least_seven_ground_truth_rows_previewed_before_prediction_seal'] is True and
                pointer_protocol['exhaustive_preseal_preview_count_known'] is False and
                pointer_protocol['strict_blind_confirmation'] is False and
                pointer_protocol['evaluation_pipeline_labels_opened'] is False,
                'Pointer protocol does not preserve the GT preview disclosure')


def register():
    """Freeze analysis choices while prediction labels remain locked."""
    for path in (FULL_PROTOCOL, POINTER_PROTOCOL, SUPERSEDED_POINTER_PROTOCOL,
                 SUPERSEDED_INCIDENT, INCIDENT, HISTORY, CORRECTION,
                 POST_V32, ERRATUM, MODEL_REGISTRY):
        require(path.exists(), 'Registration input missing: ' + str(path))
    full, pointer, incident = read(FULL_PROTOCOL), read(POINTER_PROTOCOL), read(INCIDENT)
    verify_incident(incident, pointer)
    require(full['complete'] is True and full['full_labels_may_open'] is False and
            full['history_sha256'] == sha(HISTORY) and
            full['correction_sha256'] == sha(CORRECTION) and
            full['post_v32_exposure_sha256'] == sha(POST_V32) and
            full['prose_erratum_sha256'] == sha(ERRATUM) and
            full['engine_script_sha256'] == sha(engine.__file__) and
            full['prior_model_registry_sha256'] == sha(MODEL_REGISTRY),
            'Full V2 prospective protocol/source changed')
    require(pointer['complete'] is True and
            pointer['full_protocol_sha256'] == sha(FULL_PROTOCOL) and
            pointer['model_registry_sha256'] == sha(MODEL_REGISTRY) and
            pointer['engine_script_sha256'] == sha(engine.__file__) and
            pointer['source_audit_sha256'] == sha(POINTER / 'audit.json') and
            pointer['labels_source_sha256_unopened'] ==
            read(POINTER / 'audit.json')['labels_sha256'],
            'Pointer prospective protocol/source changed')
    history = read(HISTORY)
    require(history['complete'] is True and history['n'] == 1581 and
            history['history_status_counts'] == HISTORY_COUNTS,
            'Corrected six-stratum history audit changed')
    hashes = model_hashes()
    record = dict(
        complete=True,
        purpose='Pre-specified statistics for 39 frozen models and three sealed cohorts',
        analysis_script_sha256=sha(__file__),
        full_protocol_sha256=sha(FULL_PROTOCOL),
        pointer_protocol_sha256=sha(POINTER_PROTOCOL),
        superseded_pointer_protocol_sha256=sha(SUPERSEDED_POINTER_PROTOCOL),
        superseded_incident_sha256=sha(SUPERSEDED_INCIDENT),
        incident_sha256=sha(INCIDENT),
        full_engine_script_sha256=sha(engine.__file__),
        pointer_script_sha256=pointer['pointer_script_sha256'],
        model_registry_sha256=sha(MODEL_REGISTRY),
        model_hashes=hashes,
        corrected_history_sha256=sha(HISTORY),
        corrected_exposure_sha256=sha(CORRECTION),
        post_v32_exposure_sha256=sha(POST_V32),
        prose_erratum_sha256=sha(ERRATUM),
        full_labels_sha256_unopened=full['official_manifest_sha256_from_preparation_audit'],
        pointer_labels_sha256_unopened=pointer['labels_source_sha256_unopened'],
        cohort_sizes=SIZES,
        primary_cohort='PointerBench Pro500',
        secondary_cohort='PointerBench Sheets500',
        descriptive_cohort='ScreenSpot-Pro full 1581, six corrected historical exposure strata',
        metric='Original K16 forced Top1 normalized point-in-target-geometry',
        variants=list(VARIANTS), scales=['3k', 'full'], seeds=list(SEEDS),
        matched3k_control='Equal 13,376 update aligned control on original 2,973 examples',
        primary_family=list(PRIMARY_CONTRASTS),
        familywise_alpha=0.05,
        simultaneous_confidence=0.9875,
        other_contrasts='descriptive 95% image-cluster intervals',
        bootstrap_unit='RGB image pixel_sha256',
        bootstrap_replicates=REPLICATES,
        bootstrap_seed=BOOTSTRAP_SEED,
        prediction_gate='all full1581, Pro500 and Sheets500 per-case SHA-sealed before any analysis GT read',
        preseal_human_label_access_disclosed=True,
        at_least_seven_previewed_cases_kept_in_denominators=True,
        strict_blind_confirmation=False,
        no_model_selection_from_evaluation=True,
    )
    write_once(REGISTRATION, record)
    return record


def identity(row):
    fields = {name: row[name] for name in engine.PUBLIC_KEYS}
    return hashlib.sha256(json.dumps(fields, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode()).hexdigest()


def case_key(sample_id):
    return hashlib.sha256(sample_id.encode()).hexdigest()[:24]


def verify_prediction_bundle(folder, cohort, expected_n, hashes,
                             pointer_protocol_sha=None, full_protocol_sha=None):
    """Public-only verifier; accepts synthetic folders for unit fixtures."""
    folder = Path(folder)
    spec_path, public_path, seal_path = (folder / 'protocol.json', folder / 'public.json',
                                        folder / 'predictions.json')
    require(all(p.exists() for p in (spec_path, public_path, seal_path)),
            'Missing sealed public/prediction bundle: ' + cohort)
    spec, rows, seal = read(spec_path), read(public_path), read(seal_path)
    require(spec['complete'] is True and spec['cohort'] == cohort and
            spec['n'] == expected_n and spec['public_only'] is True and
            spec['labels_opened'] is False and spec['model_hashes'] == hashes and
            spec['public_sha256'] == sha(public_path),
            'Public protocol mismatch: ' + cohort)
    if pointer_protocol_sha is not None:
        require(spec['pointer_protocol_sha256'] == pointer_protocol_sha,
                'Pointer public protocol mismatch: ' + cohort)
    if full_protocol_sha is not None:
        require(spec['erratum_sha256'] == full_protocol_sha,
                'Full public protocol mismatch')
    require(len(rows) == expected_n and
            all(set(row) == set(engine.CASE_KEYS) for row in rows) and
            len({row['sample_id'] for row in rows}) == expected_n,
            'Public case whitelist or IDs mismatch: ' + cohort)
    ids = {row['sample_id'] for row in rows}
    require(seal['complete'] is True and seal['cohort'] == cohort and
            seal['n'] == expected_n and seal['labels_opened'] is False and
            seal['public_sha256'] == sha(public_path) and
            seal['model_hashes'] == hashes and
            set(seal['case_hashes']) == ids,
            'Prediction seal incomplete or mismatched: ' + cohort)
    cases = {}
    for row in rows:
        sid = row['sample_id']
        path = folder / 'predictions/cases' / (case_key(sid) + '.json')
        require(path.exists() and sha(path) == seal['case_hashes'][sid],
                'Prediction case SHA changed: ' + sid)
        case = read(path)
        count = case['candidate_count']
        k16 = case['K16_count']
        require(case['complete'] is True and case['sample_id'] == sid and
                case['public_identity'] == identity(row) and
                case['model_hashes'] == hashes and case['labels_opened'] is False and
                type(count) is int and count >= 0 and
                type(k16) is int and 0 <= k16 <= min(16, count) and
                set(case['methods']) == set(hashes),
                'Prediction case metadata/39 methods mismatch: ' + sid)
        for name, output in case['methods'].items():
            point, index = output['point'], output['index']
            if point is None:
                require(index is None, 'Null point has non-null candidate index: ' + sid)
            else:
                require(isinstance(point, list) and len(point) == 2 and
                        all(type(x) in (int, float) and math.isfinite(x) for x in point) and
                        type(index) is int and 0 <= index < count and k16 > 0,
                        'Invalid point/index for ' + name + ': ' + sid)
        cases[sid] = case
    return rows, cases, sha(seal_path)


def gate():
    """Run every non-GT check before even hashing a label file."""
    require(REGISTRATION.exists(), 'Register statistical analysis before unsealing GT')
    protocol = read(REGISTRATION)
    require(protocol['complete'] is True and
            protocol['analysis_script_sha256'] == sha(__file__) and
            protocol['full_protocol_sha256'] == sha(FULL_PROTOCOL) and
            protocol['pointer_protocol_sha256'] == sha(POINTER_PROTOCOL) and
            protocol['superseded_pointer_protocol_sha256'] ==
            sha(SUPERSEDED_POINTER_PROTOCOL) and
            protocol['superseded_incident_sha256'] == sha(SUPERSEDED_INCIDENT) and
            protocol['incident_sha256'] == sha(INCIDENT) and
            protocol['corrected_history_sha256'] == sha(HISTORY) and
            protocol['corrected_exposure_sha256'] == sha(CORRECTION) and
            protocol['post_v32_exposure_sha256'] == sha(POST_V32) and
            protocol['prose_erratum_sha256'] == sha(ERRATUM) and
            protocol['model_registry_sha256'] == sha(MODEL_REGISTRY) and
            protocol['full_engine_script_sha256'] == sha(engine.__file__),
            'Statistical analysis registration or source drift')
    full, pointer, incident = read(FULL_PROTOCOL), read(POINTER_PROTOCOL), read(INCIDENT)
    verify_incident(incident, pointer)
    require(full['complete'] is True and full['full_labels_may_open'] is False and
            pointer['complete'] is True and pointer['full_protocol_sha256'] == sha(FULL_PROTOCOL) and
            pointer['pointer_script_sha256'] ==
            sha(ROOT / 'scripts/run_paper_scale_pointer_v2_resealed.py') and
            protocol['pointer_script_sha256'] == pointer['pointer_script_sha256'] and
            protocol['model_hashes'] == model_hashes(),
            'Model/protocol identity changed before label opening')
    require(FULL_RECEIPT.exists() and POINTER_RECEIPT.exists(),
            'All three public-only prediction receipts must exist before any GT read')
    full_receipt, pointer_receipt = read(FULL_RECEIPT), read(POINTER_RECEIPT)
    require(full_receipt['complete'] is True and full_receipt['labels_opened'] is False and
            full_receipt['erratum_sha256'] == sha(FULL_PROTOCOL) and
            full_receipt['corrected_history_sha256'] == sha(HISTORY) and
            pointer_receipt['complete'] is True and pointer_receipt['labels_opened'] is False and
            pointer_receipt['pointer_protocol_sha256'] == sha(POINTER_PROTOCOL),
            'Public completion receipt changed or labeled inference used')
    bundles = {}
    for cohort in COHORTS:
        rows, cases, prediction_sha = verify_prediction_bundle(
            EVAL / cohort, cohort, SIZES[cohort], protocol['model_hashes'],
            pointer_protocol_sha=sha(POINTER_PROTOCOL) if cohort != 'full' else None,
            full_protocol_sha=sha(FULL_PROTOCOL) if cohort == 'full' else None)
        bundles[cohort] = (rows, cases, prediction_sha)
    require(full_receipt['full_predictions_sha256'] == bundles['full'][2] and
            all(pointer_receipt['prediction_sha256'][cohort] == bundles[cohort][2]
                for cohort in ('pro', 'sheets')),
            'Public completion receipt/prediction SHA mismatch')
    history = read(HISTORY)
    require(history['complete'] is True and history['n'] == 1581 and
            history['history_status_counts'] == HISTORY_COUNTS and
            {row['sample_id'] for row in history['rows']} ==
            {row['sample_id'] for row in bundles['full'][0]} and
            all(row['pixel_sha256'] == {q['sample_id']: q for q in bundles['full'][0]}
                [row['sample_id']]['pixel_sha256'] for row in history['rows']),
            'Corrected full historical strata drift')
    return protocol, bundles, history


def open_labels_after_gate(protocol, bundles):
    """Call only after gate() returns all three verified prediction bundles."""
    require(sha(FULL_LABELS) == protocol['full_labels_sha256_unopened'] and
            sha(POINTER_LABELS) == protocol['pointer_labels_sha256_unopened'],
            'GT source digest changed since prospective registration')
    full_rows, pointer_rows = jsonl(FULL_LABELS), jsonl(POINTER_LABELS)
    require(len(full_rows) == 1581 and len(pointer_rows) == 1000,
            'Official GT counts changed')
    full = {row['sample_id']: row['target_geometry'] for row in full_rows}
    pointer = {row['sample_id']: row['target_geometry'] for row in pointer_rows}
    require(len(full) == 1581 and len(pointer) == 1000 and
            set(full) == {row['sample_id'] for row in bundles['full'][0]} and
            set(pointer) == {row['sample_id'] for cohort in ('pro', 'sheets')
                             for row in bundles[cohort][0]},
            'Public/GT ID mismatch')
    return {'full': full, 'pro': pointer, 'sheets': pointer}


def hit_vectors(rows, cases, labels, hashes):
    hits = {name: [] for name in hashes}
    details = []
    for row in rows:
        sid = row['sample_id']
        width, height = row['image_size']
        outcomes = {}
        for name in hashes:
            point = cases[sid]['methods'][name]['point']
            correct = bool(point is not None and point_in_geometry(
                (point[0] / width, point[1] / height), labels[sid]))
            hits[name].append(correct)
            outcomes[name] = correct
        details.append(dict(sample_id=sid, pixel_sha256=row['pixel_sha256'], hits=outcomes))
    return hits, details


def paired_intervals(hits, rows, replicates=REPLICATES, seed=BOOTSTRAP_SEED):
    """Query-weighted paired deltas; resample RGB image clusters, not queries."""
    clusters_by_rgb = defaultdict(list)
    for ix, row in enumerate(rows):
        clusters_by_rgb[row['pixel_sha256']].append(ix)
    clusters = list(clusters_by_rgb.values())
    n_clusters = len(clusters)
    require(n_clusters > 0, 'No RGB image clusters')
    draws = np.random.default_rng(seed).multinomial(
        n_clusters, np.full(n_clusters, 1 / n_clusters), size=replicates).astype(np.float32)
    denominator = draws @ np.asarray([len(cluster) for cluster in clusters], np.float32)

    def compare(left, right):
        left_seed = np.asarray([hits[f'{left}_s{s}'] for s in SEEDS], np.float32)
        right_seed = np.asarray([hits[f'{right}_s{s}'] for s in SEEDS], np.float32)
        delta_seed = left_seed - right_seed
        per_seed = delta_seed.mean(axis=1).astype(float).tolist()
        query_delta = delta_seed.mean(axis=0)
        numerator = np.asarray([query_delta[ix].sum() for ix in clusters], np.float32)
        sampled = (draws @ numerator) / denominator
        return dict(delta=float(query_delta.mean()), per_seed_delta=per_seed,
                    ci95=np.quantile(sampled, [0.025, 0.975]).tolist(),
                    ci98_75=np.quantile(sampled, [0.00625, 0.99375]).tolist(),
                    bootstrap_unit='RGB image', bootstrap_replicates=replicates,
                    conditional_on_fixed_three_seeds=True)

    contrasts = {}
    for variant in VARIANTS:
        contrasts[f'full_vs_3k/{variant}'] = compare(f'full_{variant}', f'3k_{variant}')
    for scale in ('3k', 'full'):
        for variant in VARIANTS[1:]:
            contrasts[f'{scale}_aligned_vs_{variant}'] = compare(
                f'{scale}_aligned', f'{scale}_{variant}')
    contrasts['full_aligned_vs_matched3k'] = compare('full_aligned', 'matched3k_aligned')
    contrasts['3k_aligned_vs_matched3k'] = compare('3k_aligned', 'matched3k_aligned')
    return contrasts, n_clusters


def summarize(hits, rows):
    n = len(rows)
    methods = {name: dict(hits=int(sum(values)), accuracy=float(np.mean(values)))
               for name, values in hits.items()}
    arms = {}
    for scale in ('3k', 'full'):
        arms[scale] = {}
        for variant in VARIANTS:
            values = [methods[f'{scale}_{variant}_s{s}']['accuracy'] for s in SEEDS]
            arms[scale][variant] = dict(seed_accuracies=values,
                                        mean_accuracy=float(np.mean(values)))
    values = [methods[f'matched3k_aligned_s{s}']['accuracy'] for s in SEEDS]
    arms['matched3k_aligned'] = dict(seed_accuracies=values,
                                     mean_accuracy=float(np.mean(values)))
    contrasts, n_rgb = paired_intervals(hits, rows)
    return methods, arms, contrasts, n_rgb


def history_strata(history, details, hits):
    audit = {row['sample_id']: row for row in history['rows']}
    require(set(audit) == {row['sample_id'] for row in details},
            'Corrected full history IDs differ from analysis')
    answer = {}
    for field in ('history_status', 'trained_3k_exposure', 'trained_full_exposure'):
        strata = {}
        for value in sorted({row[field] for row in audit.values()}):
            indices = [i for i, row in enumerate(details)
                       if audit[row['sample_id']][field] == value]
            aligned = {}
            for scale in ('3k', 'full'):
                seed_hits = [int(sum(hits[f'{scale}_aligned_s{s}'][i] for i in indices))
                             for s in SEEDS]
                aligned[scale] = dict(seed_hits=seed_hits,
                                      mean_accuracy=float(np.mean(seed_hits) / len(indices)))
            strata[value] = dict(n=len(indices), aligned=aligned,
                                 full_minus_3k_pp=100 * (
                                     aligned['full']['mean_accuracy'] -
                                     aligned['3k']['mean_accuracy']))
        answer[field] = strata
    require({key: value['n'] for key, value in answer['history_status'].items()} ==
            HISTORY_COUNTS, 'Six corrected historical strata counts changed')
    return answer


def analyze():
    protocol, bundles, history = gate()
    labels = open_labels_after_gate(protocol, bundles)
    outputs = {}
    for cohort in COHORTS:
        rows, cases, prediction_sha = bundles[cohort]
        hits, details = hit_vectors(rows, cases, labels[cohort], protocol['model_hashes'])
        methods, arms, contrasts, n_rgb = summarize(hits, rows)
        result = dict(complete=True, cohort=cohort,
                      role=('primary independent' if cohort == 'pro' else
                            'secondary independent' if cohort == 'sheets' else
                            'descriptive historically exposed'),
                      n=len(rows), unique_rgb_images=n_rgb,
                      metric=protocol['metric'], predictions_sha256=prediction_sha,
                      labels_sha256=(sha(FULL_LABELS) if cohort == 'full' else sha(POINTER_LABELS)),
                      analysis_protocol_sha256=sha(REGISTRATION), methods=methods,
                      arms=arms, contrasts=contrasts,
                      primary_family=list(PRIMARY_CONTRASTS),
                      primary_ci='98.75% simultaneous Bonferroni' if cohort == 'pro' else None,
                      historical_exposure=history_strata(history, details, hits)
                      if cohort == 'full' else None,
                      preseal_gt_preview_addendum_sha256=sha(INCIDENT),
                      strict_blind_confirmation=False)
        write_once(OUT / f'{cohort}_details.json', details)
        write_once(OUT / f'{cohort}.json', result)
        outputs[cohort] = result
    readout = render_readout(outputs)
    text_once(OUT / 'readout.txt', readout)
    summary = dict(complete=True, analysis_protocol_sha256=sha(REGISTRATION),
                   input_prediction_sha256={cohort: bundles[cohort][2] for cohort in COHORTS},
                   result_sha256={cohort: sha(OUT / f'{cohort}.json') for cohort in COHORTS},
                   readout_sha256=sha(OUT / 'readout.txt'),
                   preseal_gt_preview_addendum_sha256=sha(INCIDENT),
                   strict_blind_confirmation=False,
                   analyzed_at_utc=now())
    if (OUT / 'summary.json').exists():
        old = read(OUT / 'summary.json')
        require({k: v for k, v in old.items() if k != 'analyzed_at_utc'} ==
                {k: v for k, v in summary.items() if k != 'analyzed_at_utc'},
                'Existing analysis summary differs from sealed inputs/results')
    else:
        write_once(OUT / 'summary.json', summary)
    return summary


def render_readout(results):
    lines = [
        'Paper-scale V2: frozen 39-model ablation and independent PointerBench confirmation',
        'Original K16 forced Top1 point-in-target-geometry; all three fixed seeds reported.',
        'PointerBench Pro500 is the predeclared primary cohort; Sheets500 is secondary.',
        'ScreenSpot-Pro 1581 is descriptive because of six corrected historical exposure strata.',
        'RGB-image-cluster paired bootstrap: 10,000 draws, seed 20261111.',
        'Four primary Pro mechanism comparisons use 98.75% Bonferroni intervals; other intervals are descriptive 95%.',
        'Protocol deviation: before prediction sealing, at least seven published GT examples were visible to researchers/agents (Pro pbp_0000, pbp_0001, pbp_0005; Sheets pbs_0000 through pbs_0003). The exhaustive count is unknown. They remain in the fixed 500-case denominators. No reported model fitting, selection, scoring or evaluation-pipeline label loading followed from those previews. These are independent of local training/history by exact/near image audit, but are not strictly blind confirmation.',
        '',
    ]
    for cohort in COHORTS:
        report = results[cohort]
        n = report['n']
        lines += [f'{cohort.upper()} | {report["role"]} | n={n} | RGB clusters={report["unique_rgb_images"]}',
                  'scale | variant | seed1 hits | seed2 hits | seed3 hits | three-seed mean accuracy']
        for scale in ('3k', 'full'):
            for variant in VARIANTS:
                counts = [report['methods'][f'{scale}_{variant}_s{s}']['hits'] for s in SEEDS]
                mean = report['arms'][scale][variant]['mean_accuracy']
                lines.append(f'{scale} | {variant} | {counts[0]}/{n} | {counts[1]}/{n} | '
                             f'{counts[2]}/{n} | {100*mean:.2f}%')
        counts = [report['methods'][f'matched3k_aligned_s{s}']['hits'] for s in SEEDS]
        mean = report['arms']['matched3k_aligned']['mean_accuracy']
        lines.append(f'matched3k | aligned | {counts[0]}/{n} | {counts[1]}/{n} | '
                     f'{counts[2]}/{n} | {100*mean:.2f}%')
        lines.append('Four registered mechanism contrasts: delta percentage points [98.75% cluster CI]')
        for name in PRIMARY_CONTRASTS:
            item = report['contrasts'][name]
            lo, hi = item['ci98_75']
            lines.append(f'{name}: {100*item["delta"]:+.2f} [{100*lo:+.2f}, {100*hi:+.2f}]')
        lines.append('Scale, equal-update control and remaining ablations: delta pp [descriptive 95% cluster CI]')
        for name, item in report['contrasts'].items():
            if name in PRIMARY_CONTRASTS:
                continue
            lo, hi = item['ci95']
            lines.append(f'{name}: {100*item["delta"]:+.2f} [{100*lo:+.2f}, {100*hi:+.2f}]')
        if cohort == 'full':
            lines.append('Corrected prior exposure strata; aligned-only descriptive comparison:')
            for name, item in report['historical_exposure']['history_status'].items():
                lines.append(f'{name}: n={item["n"]}, '
                             f'3k={100*item["aligned"]["3k"]["mean_accuracy"]:.2f}%, '
                             f'full={100*item["aligned"]["full"]["mean_accuracy"]:.2f}%, '
                             f'delta={item["full_minus_3k_pp"]:+.2f} pp')
        lines.append('')
    lines += [
        'CIs resample screenshot RGB-image clusters and condition on these three fixed training seeds.',
        'Full-vs-3k also changes source mixture, so the contrast is not a pure data-quantity effect.',
        'PointerBench is synthetic GUI grounding; results do not measure live UI action execution or establish a direct MVP superiority claim.',
        '',
    ]
    return '\n'.join(lines)


def self_test():
    """Synthetic only: no project GT, checkpoint or test prediction reads."""
    import tempfile

    hashes = {'3k_aligned_s1': 'a', '3k_aligned_s2': 'b'}
    fixture_root = ROOT / '.tmp'
    fixture_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=fixture_root) as location:
        folder = Path(location)
        row = {name: 'synthetic' for name in engine.CASE_KEYS}
        row.update(sample_id='synthetic:1', image_size=[100, 100],
                   pixel_sha256='rgb-synthetic')
        public = folder / 'public.json'
        public.write_text(json.dumps([row]), encoding='utf-8')
        spec = dict(complete=True, cohort='synthetic', n=1, public_only=True,
                    labels_opened=False, model_hashes=hashes, public_sha256=sha(public))
        (folder / 'protocol.json').write_text(json.dumps(spec), encoding='utf-8')
        case = dict(complete=True, sample_id='synthetic:1', public_identity=identity(row),
                    model_hashes=hashes, feature_sha256='synthetic', candidate_count=1,
                    K16_count=1, labels_opened=False,
                    methods={name: dict(point=[50, 50], index=0) for name in hashes})
        case_path = folder / 'predictions/cases' / (case_key(row['sample_id']) + '.json')
        case_path.parent.mkdir(parents=True)
        case_path.write_text(json.dumps(case), encoding='utf-8')
        seal = dict(complete=True, cohort='synthetic', n=1, labels_opened=False,
                    public_sha256=sha(public), model_hashes=hashes,
                    case_hashes={'synthetic:1': sha(case_path)})
        (folder / 'predictions.json').write_text(json.dumps(seal), encoding='utf-8')
        verified_rows, verified_cases, _ = verify_prediction_bundle(
            folder, 'synthetic', 1, hashes)
        require(verified_rows == [row] and verified_cases['synthetic:1'] == case,
                'Synthetic prediction bundle failed')
        case_path.write_text(json.dumps({**case, 'candidate_count': 2}), encoding='utf-8')
        try:
            verify_prediction_bundle(folder, 'synthetic', 1, hashes)
        except ValueError as error:
            require('SHA changed' in str(error), 'Synthetic tamper gave wrong error')
        else:
            raise AssertionError('Synthetic prediction tamper bypassed SHA gate')

    rows = [dict(pixel_sha256='A'), dict(pixel_sha256='A'), dict(pixel_sha256='B')]
    hits = {}
    for scale in ('3k', 'full'):
        for variant in VARIANTS:
            for seed in SEEDS:
                hits[f'{scale}_{variant}_s{seed}'] = [True, True, False]
    for seed in SEEDS:
        hits[f'matched3k_aligned_s{seed}'] = [False, False, False]
        hits[f'full_aligned_s{seed}'] = [True, True, True]
    contrasts, n_rgb = paired_intervals(hits, rows, replicates=100, seed=3)
    require(n_rgb == 2 and
            math.isclose(contrasts['full_vs_3k/aligned']['delta'], 1 / 3, rel_tol=1e-6) and
            math.isclose(contrasts['full_aligned_vs_matched3k']['delta'], 1.0, rel_tol=1e-6),
            'Synthetic paired interval/cluster calculation failed')
    require(point_in_geometry((0.5, 0.5), {'type': 'bbox', 'boxes': [[0.4, 0.4, 0.6, 0.6]]}),
            'Synthetic geometry evaluator failed')
    print(json.dumps({'synthetic_self_test': 'passed', 'real_labels_opened': False,
                      'model_forward_calls': 0}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True,
                        choices=('register', 'verify-gate', 'analyze', 'self-test'))
    args = parser.parse_args()
    if args.stage == 'self-test':
        self_test()
    elif args.stage == 'register':
        record = register()
        print(json.dumps({'registered': True, 'protocol_sha256': sha(REGISTRATION),
                          'gt_opened_by_analysis': False,
                          'preseal_gt_preview_disclosed': True,
                          'primary_family': record['primary_family']}))
    elif args.stage == 'verify-gate':
        _, bundles, _ = gate()
        print(json.dumps({'gate': 'passed', 'cohort_n': {name: len(bundles[name][0])
                                                       for name in COHORTS},
                          'gt_opened_by_analysis': False}))
    else:
        print(json.dumps(analyze(), ensure_ascii=False))


if __name__ == '__main__':
    main()
