"""Pre-registered, label-gated analysis of the cached W2 token-source control."""
from __future__ import annotations

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '2'

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts'), str(Path(__file__).parent)]

import numpy as np
import torch
from clickagent.evidence_retention import RetentionModel, require
from clickagent.contribution_ablation import transform_inputs
from clickagent.public_evaluation import point_in_geometry

import w2_name_token_control as w2
import analyze_pointer_mvp_same_cohort_v1 as pointer_analysis
import run_mmbench_gui_l2_v1 as mmbench_analysis

require(not torch.cuda.is_available(), 'W2 analysis must stay CPU only')

OUT = w2.OUT
OLD_REG = OUT / 'analysis_registration.json'
PREVIOUS_REG = OUT / 'analysis_registration_v2.json'
REG = OUT / 'analysis_registration_v3.json'
PRED = OUT / 'predictions'
MMB_SUPPLEMENT = ROOT / 'runs/mmbench_full_ablation_supplement_v1'
MMB_SUPPLEMENT_AUDIT = OUT / 'supplement_reference_audit.json'
POINTER_LABELS = w2.V2 / 'pointerbench/labels.jsonl'
MMB_LABELS = ROOT / 'data/mmbench_gui_l2_v1/source/L2_annotations.json'
REPLICATES = 10000
BOOT_SEED = 20261114
COHORTS = w2.COHORTS
ARMS = w2.ARMS
W2_MODELS = tuple(f'{arm}_s{s}' for arm in ARMS for s in (1, 2, 3))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return w2.sha(path)


def write(path, value):
    w2.write(path, value)


def case_key(sample_id):
    return hashlib.sha256(sample_id.encode()).hexdigest()[:24]


def reference_dir(cohort):
    return (w2.MMB / 'mmbench') if cohort == 'mmbench' else (w2.V2 / 'eval' / cohort)


def reference_models(cohort):
    names = tuple(f'full_aligned_s{s}' for s in (1, 2, 3))
    return names + tuple(f'full_token_mean_s{s}' for s in (1, 2, 3))


def reference_method(cohort, case, name):
    if cohort == 'mmbench' and 'token_mean' in name:
        return case['supplement']['methods'][name]
    return case['reference']['methods'][name]


def registration():
    require((OUT / 'protocol.json').exists(), 'W2 protocol absent')
    require(OLD_REG.exists() and PREVIOUS_REG.exists() and MMB_SUPPLEMENT_AUDIT.exists(),
            'Original registration/supplement audit absent')
    audit = read(MMB_SUPPLEMENT_AUDIT)
    require(audit['complete'] and audit['no_labels_opened'] and
            audit['all_case_sha_feature_K16_point_model_checks_passed'] and audit['n'] == 1000,
            'MMBench supplement was not pre-audited')
    require(all((PRED / c / 'completion.json').exists() for c in COHORTS) and
            (OUT / 'verify_gate.json').exists(), 'V3 implementation correction requires prior sealed gate')
    mapping_audit = OUT / 'analysis_mapping_correction_audit.json'
    require(read(mapping_audit)['complete'] and read(mapping_audit)['no_labels_read'],
            'Public/index field-source audit missing')
    source_receipts = {c: sha(reference_dir(c) / 'predictions.json') for c in COHORTS}
    official = {c: sha(w2.V2 / 'pointerbench/source' / c / 'eval.py') for c in ('pro', 'sheets')}
    official['mmbench'] = sha(mmbench_analysis.SCORER)
    plan = dict(
        registered_at_utc=datetime.now(timezone.utc).isoformat(),
        correction_reason='Pure implementation fix after all 2000 prediction/gate seals: cluster pixel_sha256 and MMBench source_index belong to the sealed public row, not the feature-index entry. V1/V2 scientific contrasts, fixed cohorts, models, predictions and scoring rules are unchanged. The first analyze attempt opened labels only after the gate, then raised KeyError before writing accuracy or interval results.',
        previous_registration_sha256=sha(PREVIOUS_REG),
        initial_registration_sha256=sha(OLD_REG),
        public_mapping_audit_sha256=sha(mapping_audit),
        registered_after_prediction_seal=True,
        registered_after_label_gate=True,
        mmbench_supplement_audit_sha256=sha(MMB_SUPPLEMENT_AUDIT),
        mmbench_supplement_prediction_receipt_sha256=sha(MMB_SUPPLEMENT / 'predictions.json'),
        analysis_script_sha256=sha(__file__), w2_protocol_sha256=sha(OUT / 'protocol.json'),
        reference_prediction_receipt_sha256=source_receipts,
        official_scorer_sha256=official,
        pointer_labels_sha256_unopened=read(w2.V2 / 'pointerbench/evaluation_protocol_v2.json')['labels_source_sha256_unopened'],
        mmbench_labels_sha256_unopened=read(w2.MMB / 'protocol.json')['source_annotation_sha256'],
        fixed_denominators={'pro': 500, 'sheets': 500, 'mmbench': 1000},
        w2_models=list(W2_MODELS), original_models={c: list(reference_models(c)) for c in COHORTS},
        primary_contrast='last_name minus last_prefill, three fixed seeds; each of Pro500, Sheets500, MMB1000 reported',
        primary_familywise_ci='98.333333% two-sided screenshot-cluster interval for three primary cohort contrasts (Bonferroni)',
        secondary_contrasts=[
            'full_aligned minus last_name', 'full_aligned minus last_prefill',
            'full_token_mean minus last_name (all three cohorts)',
            'full_token_mean minus last_prefill (all three cohorts)',
        ],
        bootstrap=dict(unit='RGB screenshot pixel_sha256', replicates=REPLICATES, seed=BOOT_SEED,
                       seed_aggregation='mean of three fixed seed-wise paired outcomes',
                       conditional_on_fitted_seeds=True),
        prediction_gate='Before any label read, verify all 2000 W2 case SHA receipts, fixed K16 point membership, derived feature receipts, six model checkpoints, all original prediction receipts, full CPU score replay of W2 and original reference models, official scorer file SHA and synthetic scorer cases.',
        labels_after_gate_only=True,
        interpretation='The last-prefill token is a fixed prompt context token, NOT an original instruction token. Even a positive last-name contrast cannot establish name superiority over instruction tokens. Upstream proposer/crops remain name dependent.',
        retrospective=True, strict_blind=False,
    )
    previous = read(PREVIOUS_REG)
    for field in ('w2_protocol_sha256', 'reference_prediction_receipt_sha256',
                  'official_scorer_sha256', 'pointer_labels_sha256_unopened',
                  'mmbench_labels_sha256_unopened', 'fixed_denominators', 'w2_models',
                  'original_models', 'primary_contrast', 'primary_familywise_ci',
                  'secondary_contrasts', 'bootstrap', 'prediction_gate',
                  'labels_after_gate_only', 'interpretation', 'retrospective', 'strict_blind'):
        require(plan[field] == previous[field], 'Scientific analysis field changed in v3: ' + field)
    if REG.exists():
        old = read(REG)
        plan['registered_at_utc'] = old['registered_at_utc']
        require(plan == old, 'W2 analysis registration changed')
    else:
        write(REG, plan)
    print(json.dumps(dict(registered=True, registration_sha256=sha(REG)), ensure_ascii=False), flush=True)


def verify_registration():
    r = read(REG)
    require(r['previous_registration_sha256'] == sha(PREVIOUS_REG) and
            r['initial_registration_sha256'] == sha(OLD_REG) and
            r['public_mapping_audit_sha256'] == sha(OUT / 'analysis_mapping_correction_audit.json') and
            r['mmbench_supplement_audit_sha256'] == sha(MMB_SUPPLEMENT_AUDIT) and
            r['mmbench_supplement_prediction_receipt_sha256'] == sha(MMB_SUPPLEMENT / 'predictions.json'),
            'Transparent v2 correction/reference changed')
    require(r['analysis_script_sha256'] == sha(__file__) and r['w2_protocol_sha256'] == sha(OUT / 'protocol.json'),
            'W2 analysis code/protocol changed')
    require(r['fixed_denominators'] == {'pro': 500, 'sheets': 500, 'mmbench': 1000}, 'Denominators changed')
    for cohort in COHORTS:
        require(r['reference_prediction_receipt_sha256'][cohort] == sha(reference_dir(cohort) / 'predictions.json'),
                'Original prediction seal changed')
    for cohort in ('pro', 'sheets'):
        require(r['official_scorer_sha256'][cohort] == sha(w2.V2 / 'pointerbench/source' / cohort / 'eval.py'),
                'Pointer official scorer changed')
    require(r['official_scorer_sha256']['mmbench'] == sha(mmbench_analysis.SCORER),
            'MMBench official scorer changed')
    return r


def load_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    model = RetentionModel()
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model, checkpoint['mean'].numpy(), checkpoint['std'].numpy()


def choose(model_tuple, base, active, state, mask, baseline, sample_id, variant='aligned'):
    model, mean, std = model_tuple
    args = dict(base=torch.as_tensor((base - mean) / std),
                active=torch.as_tensor(active), state=torch.as_tensor(state), mask=torch.as_tensor(mask))
    if variant == 'token_mean':
        args = transform_inputs(args, [len(base)], [sample_id], 'token_mean')
    with torch.inference_mode():
        scores = model(**args).numpy()
    require(np.isfinite(scores).all(), 'Nonfinite CPU replay score')
    return int(baseline[np.argmax(scores[baseline])]) if len(baseline) else None


def model_bundle(cohort):
    models = {}
    for arm in ARMS:
        for seed in (1, 2, 3):
            tag = f'{arm}_s{seed}'
            receipt = read(OUT / f'models/{tag}.json')
            require(receipt['complete'] and receipt['steps'] == w2.STEPS and
                    receipt['parameter_count'] == 22050 and sha(receipt['checkpoint_path']) == receipt['checkpoint_sha256'],
                    'W2 checkpoint missing/drifted: ' + tag)
            models[tag] = load_model(receipt['checkpoint_path'])
    for seed in (1, 2, 3):
        receipt = read(w2.V18 / f'models/full_s{seed}.json')
        require(sha(receipt['checkpoint_path']) == receipt['checkpoint_sha256'], 'Original aligned checkpoint drift')
        models[f'full_aligned_s{seed}'] = load_model(receipt['checkpoint_path'])
        receipt = read(w2.PAPER_V1 / f'models/token_mean_s{seed}.json')
        require(sha(receipt['checkpoint_path']) == receipt['checkpoint_sha256'], 'Original token_mean checkpoint drift')
        models[f'full_token_mean_s{seed}'] = load_model(receipt['checkpoint_path'])
    return models


def synthetic_scorer_checks():
    for cohort in ('pro', 'sheets'):
        scorer, _ = pointer_analysis.load_official_scorer(cohort)
        for p, box, expected in (((5, 5), (0, 0, 10, 10), True),
                                 ((11, 5), (0, 0, 10, 10), False),
                                 ((0, 0), (0, 0, 10, 10), True)):
            require(bool(scorer(p, box)) == expected, 'Pointer scorer synthetic parity')
    reference = mmbench_analysis.source_reference_hit()
    for point, bbox, size in (((50, 50), [0.4, 0.4, 0.6, 0.6], (100, 100)),
                              ((100, 100), [0.4, 0.4, 0.6, 0.6], (100, 100))):
        require(bool(reference(bbox.copy(), size, point)) == mmbench_analysis.official_hit(point, bbox, list(size)),
                'MMBench synthetic scorer parity')


def gate():
    registration = verify_registration()
    synthetic_scorer_checks()
    bundles = {}
    for cohort in COHORTS:
        rows = w2.entries(cohort)
        source = reference_dir(cohort)
        reference = read(source / 'predictions.json')
        supplement = read(MMB_SUPPLEMENT / 'predictions.json') if cohort == 'mmbench' else None
        derived = read(OUT / 'features' / cohort / 'index.json')
        completion = read(PRED / cohort / 'completion.json')
        n = registration['fixed_denominators'][cohort]
        require(len(rows) == len(derived['cases']) == completion['n'] == reference['n'] == n,
                'Prediction denominator changed: ' + cohort)
        require(derived['complete'] and derived['protocol_sha256'] == sha(OUT / 'protocol.json') and
                derived['source_index_sha256'] == sha(w2.SOURCES[cohort]), 'Derived feature seal changed')
        require(completion['complete'] and completion['labels_opened'] is False and
                completion['protocol_sha256'] == sha(OUT / 'protocol.json') and
                set(completion['model_names']) == set(W2_MODELS), 'W2 prediction completion invalid')
        require(reference['complete'] and reference['labels_opened'] is False and
                set(reference['case_hashes']) == {r['sample_id'] for r in rows}, 'Original prediction receipt invalid')
        require(set(completion['case_hashes']) == set(reference['case_hashes']), 'W2 prediction IDs changed')
        if supplement is not None:
            require(supplement['complete'] and supplement['labels_opened'] is False and supplement['n'] == n and
                    set(supplement['case_hashes']) == set(reference['case_hashes']),
                    'MMBench supplement prediction IDs changed')
        models = model_bundle(cohort)
        verified = {}
        for ix, (row, d) in enumerate(zip(rows, derived['cases']), 1):
            sid = row['sample_id']
            require(sid == d['sample_id'] and d['source_state_maxdiff'] == 0.0 and
                    d['source_entry_sha256'] == w2.base.digest(row), 'Derived source parity missing')
            path = PRED / cohort / 'cases' / (case_key(sid) + '.json')
            rpath = source / 'predictions/cases' / (case_key(sid) + '.json')
            spath = MMB_SUPPLEMENT / 'predictions/cases' / (case_key(sid) + '.json') if supplement is not None else None
            require(completion['case_hashes'][sid] == sha(path) and reference['case_hashes'][sid] == sha(rpath),
                    'Per-case prediction SHA changed')
            new, old = read(path), read(rpath)
            extra = None
            if spath is not None:
                require(supplement['case_hashes'][sid] == sha(spath), 'MMBench supplement case SHA changed')
                extra = read(spath)
                require(extra['complete'] and extra['labels_opened'] is False and extra['sample_id'] == sid and
                        extra['feature_sha256'] == row['sha256'] and extra['candidate_count'] == row['n'] and
                        extra['K16_count'] == old['K16_count'] and
                        all(f'full_token_mean_s{s}' in extra['methods'] for s in (1, 2, 3)),
                        'MMBench supplement feature/K16/model mismatch')
            require(new['complete'] and old['complete'] and new['sample_id'] == old['sample_id'] == sid and
                    new['labels_opened'] is False and old['labels_opened'] is False and
                    set(new['methods']) == set(W2_MODELS) and
                    set(f'full_aligned_s{s}' for s in (1, 2, 3)) <= set(old['methods']) and
                    (cohort == 'mmbench' or set(reference_models(cohort)) <= set(old['methods'])),
                    'Per-case prediction methods changed')
            require(sha(row['path']) == row['sha256'] and sha(row['state_path']) == row['state_sha256'] and
                    sha(row['mask_path']) == row['mask_sha256'], 'Original feature SHA changed')
            with np.load(row['path'], allow_pickle=False) as z:
                points, base, active, baseline = (z[k].copy() for k in ('points', 'base', 'active', 'baseline'))
            require(len(baseline) == new['K16_count'] and len(baseline) <= 16, 'Original K16 changed')
            original_state = np.load(row['state_path'], allow_pickle=False)
            original_mask = np.load(row['mask_path'], allow_pickle=False)
            require(sha(d['mask_path']) == d['mask_sha256'], 'Derived mask SHA changed')
            mask = np.load(d['mask_path'], allow_pickle=False)
            require(np.array_equal(mask[..., 0], original_mask.any(-1)), 'Fixed view mask changed')
            states = {}
            for arm in ARMS:
                require(sha(d['states'][arm]['path']) == d['states'][arm]['sha256'], 'Derived state SHA changed')
                states[arm] = np.load(d['states'][arm]['path'], allow_pickle=False)
            for name in W2_MODELS + reference_models(cohort):
                target = new['methods'][name] if name in W2_MODELS else reference_method(
                    cohort, dict(reference=old, supplement=extra), name)
                chosen = target['index']
                require((chosen is None and len(baseline) == 0) or
                        (chosen is not None and chosen in baseline and target['point'] == points[chosen].tolist()),
                        'Prediction is not an original K16 point')
                if name in W2_MODELS:
                    arm = name.rsplit('_s', 1)[0]
                    replay = choose(models[name], base, active, states[arm], mask, baseline, sid)
                else:
                    variant = 'token_mean' if 'token_mean' in name else 'aligned'
                    replay = choose(models[name], base, active, original_state, original_mask,
                                    baseline, sid, variant)
                require(chosen == replay, 'Frozen CPU checkpoint replay differs: ' + sid + '/' + name)
            verified[sid] = dict(new=new, reference=old, supplement=extra)
        public_path = (w2.MMB / 'public.json') if cohort == 'mmbench' else (source / 'public.json')
        require(sha(public_path) == reference['public_sha256'], 'Public metadata SHA changed')
        public_rows = read(public_path)
        require(len(public_rows) == n and
                all(a['sample_id'] == b['sample_id'] for a, b in zip(public_rows, rows)) and
                all('pixel_sha256' in row and 'image_size' in row for row in public_rows) and
                (cohort != 'mmbench' or all('source_index' in row for row in public_rows)),
                'Sealed public row mapping failed')
        bundles[cohort] = dict(rows=public_rows, cases=verified)
        print(json.dumps(dict(stage='verify_gate_cohort', cohort=cohort, verified=n), ensure_ascii=False), flush=True)
    result = dict(complete=True, all_public_predictions_sealed=True, no_labels_opened=True,
                  denominators=registration['fixed_denominators'],
                  w2_completion_sha256={c: sha(PRED / c / 'completion.json') for c in COHORTS},
                  registration_sha256=sha(REG), at_utc=datetime.now(timezone.utc).isoformat())
    write(OUT / 'verify_gate.json', result)
    return registration, bundles


def open_labels(registration, bundles):
    # This function must be called only after gate() returns all three bundles.
    require(set(bundles) == set(COHORTS) and read(OUT / 'verify_gate.json')['all_public_predictions_sealed'],
            'All-cohort prediction gate incomplete')
    require(sha(POINTER_LABELS) == registration['pointer_labels_sha256_unopened'] and
            sha(MMB_LABELS) == registration['mmbench_labels_sha256_unopened'], 'Label source changed')
    pointer = [json.loads(line) for line in POINTER_LABELS.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    mmb = read(MMB_LABELS)
    require(len(pointer) == 1000 and len(mmb) == 3594, 'Official label source size changed')
    return {r['sample_id']: r for r in pointer}, {r['index']: r for r in mmb}


def hit_vectors(cohort, bundle, pointer_labels, mmb_labels):
    rows = bundle['rows']
    names = W2_MODELS + reference_models(cohort)
    hits = {name: np.zeros(len(rows), dtype=np.float32) for name in names}
    if cohort == 'mmbench':
        source_reference = mmbench_analysis.source_reference_hit()
    else:
        scorer, _ = pointer_analysis.load_official_scorer(cohort)
    for ix, row in enumerate(rows):
        sid = row['sample_id']
        case = bundle['cases'][sid]
        if cohort == 'mmbench':
            label = mmb_labels[row['source_index']]
            require(all(label[k] == row[k] for k in ('instruction', 'image_size', 'data_type', 'platform', 'grounding_type')),
                    'MMBench public/label row changed')
        else:
            label = pointer_labels[sid]
        for name in names:
            method = case['new']['methods'][name] if name in W2_MODELS else reference_method(cohort, case, name)
            point = method['point']
            if cohort == 'mmbench':
                outcome = mmbench_analysis.official_hit(point, label['bbox'], row['image_size'])
                if point is not None:
                    require(outcome == bool(source_reference(label['bbox'].copy(), tuple(row['image_size']), tuple(point))),
                            'Real MMBench official scorer parity failed')
            else:
                outcome = pointer_analysis.hit_with_parity(point, label, sid, scorer)
            hits[name][ix] = float(outcome)
    return hits


def paired_contrast(left, right, rows, hits):
    delta = np.stack([hits[f'{left}_s{s}'] - hits[f'{right}_s{s}'] for s in (1, 2, 3)])
    clusters = defaultdict(list)
    for ix, row in enumerate(rows):
        clusters[row['pixel_sha256']].append(ix)
    groups = list(clusters.values())
    rng = np.random.default_rng(BOOT_SEED)
    draws = rng.multinomial(len(groups), np.full(len(groups), 1 / len(groups)), size=REPLICATES).astype(np.float32)
    per_group = np.asarray([delta.mean(0)[ids].sum() for ids in groups], np.float32)
    counts = np.asarray([len(ids) for ids in groups], np.float32)
    sampled = (draws @ per_group) / (draws @ counts)
    alpha = .05 / 3
    return dict(delta=float(delta.mean()), per_seed_delta=delta.mean(1).astype(float).tolist(),
                ci95=np.quantile(sampled, [.025, .975]).tolist(),
                ci_primary_familywise=np.quantile(sampled, [alpha / 2, 1 - alpha / 2]).tolist(),
                screenshot_clusters=len(groups), replicates=REPLICATES)


def analyze():
    registration, bundles = gate()
    pointer_labels, mmb_labels = open_labels(registration, bundles)
    reports = {}
    for cohort, bundle in bundles.items():
        rows = bundle['rows']
        hits = hit_vectors(cohort, bundle, pointer_labels, mmb_labels)
        rates = {}
        for arm in ARMS + ('full_aligned', 'full_token_mean'):
            vectors = [hits[f'{arm}_s{s}'] for s in (1, 2, 3)]
            rates[arm] = dict(seed_hits=[int(v.sum()) for v in vectors],
                              mean_accuracy=float(np.mean([v.mean() for v in vectors])))
        pairs = [('last_name', 'last_prefill'), ('full_aligned', 'last_name'),
                 ('full_aligned', 'last_prefill')]
        pairs += [('full_token_mean', 'last_name'), ('full_token_mean', 'last_prefill')]
        contrasts = {f'{a}_minus_{b}': paired_contrast(a, b, rows, hits) for a, b in pairs}
        strata = {}
        if cohort == 'mmbench':
            for kind in ('basic', 'advanced'):
                indices = [ix for ix, r in enumerate(rows) if r['grounding_type'] == kind]
                require(indices, 'Missing MMBench stratum')
                strata[kind] = dict(n=len(indices), name_mean=float(np.mean([
                    hits[f'last_name_s{s}'][indices].mean() for s in (1, 2, 3)])),
                    prefill_mean=float(np.mean([hits[f'last_prefill_s{s}'][indices].mean() for s in (1, 2, 3)])))
        reports[cohort] = dict(n=len(rows), rates=rates, contrasts=contrasts, strata=strata,
                               real_scorer_parity_passed=True)
        write(OUT / 'analysis' / f'{cohort}.json', reports[cohort])
    result = dict(complete=True, registration_sha256=sha(REG), gate_sha256=sha(OUT / 'verify_gate.json'),
                  reports=reports,
                  interpretation='Last-prefill is a fixed prompt context token, NOT an original instruction token. Any last-name advantage here cannot establish generated name superiority over original instruction tokens. Upstream candidate proposer and crops remain name dependent.',
                  retrospective=True, strict_blind=False)
    write(OUT / 'analysis/summary.json', result)
    lines = [
        'W2 cached token-source control — retrospective fixed-candidate comparison',
        'CRITICAL: last-prefill is a fixed prompt context token, NOT an original instruction token.',
        'Even if last-name wins, this does NOT establish that generated names outperform original instruction tokens.',
        'The upstream proposer and crop plan still depend on generated names.',
        'All three cohorts and all three seeds are reported; no test-based checkpoint selection.',
    ]
    for cohort in COHORTS:
        r = reports[cohort]
        lines += ['', f'{cohort.upper()} n={r["n"]}', 'arm | seed hits | mean accuracy']
        for name, item in r['rates'].items():
            lines.append(f'{name} | {item["seed_hits"]} | {100*item["mean_accuracy"]:.2f}%')
        lines.append('paired contrast | delta pp | 95% screenshot-cluster CI pp | 98.33% familywise CI pp (primary only)')
        for name, item in r['contrasts'].items():
            family = item['ci_primary_familywise'] if name == 'last_name_minus_last_prefill' else None
            lines.append(f'{name} | {100*item["delta"]:+.2f} | '
                         f'[{100*item["ci95"][0]:+.2f},{100*item["ci95"][1]:+.2f}] | '
                         f'{"" if family is None else f"[{100*family[0]:+.2f},{100*family[1]:+.2f}]"}')
        if r['strata']:
            for kind, item in r['strata'].items():
                lines.append(f'MMBench {kind}: n={item["n"]}, last-name={100*item["name_mean"]:.2f}%, '
                             f'last-prefill={100*item["prefill_mean"]:.2f}%')
    lines += ['', 'Intervals condition on the three fixed trained seeds; screenshot RGBs are resampled as clusters.',
              'PointerBench had preseal answer previews and V2 labels had been analyzed before this retrospective control.',
              'MMBench is a project-history image-disjoint selected subset, not the full official leaderboard denominator.', '']
    path = OUT / 'analysis/readout.txt'
    path.write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(dict(analysis_complete=True, summary=str(OUT / 'analysis/summary.json'), readout=str(path)),
                     ensure_ascii=False), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', required=True, choices=('register', 'verify-gate', 'analyze'))
    args = p.parse_args()
    if args.stage == 'register': registration()
    elif args.stage == 'verify-gate': gate()
    else: analyze()


if __name__ == '__main__':
    main()
