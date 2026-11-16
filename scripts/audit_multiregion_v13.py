"""Final V13 CPU provenance, fixed-input and 120-forward reproduction audit.

Run only after both fresh cohorts have been evaluated. No models are trained,
no GPU is used, and no score tolerance or sample selection is adapted to the
observations. All sealed files/training evidence are rehashed in full.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from clickagent.action_benchmark import write_json
from clickagent.internal_candidate_scorer import INPUT_KEYS
from clickagent.multiregion_proposals import build_c1_c3_pools, plan_multiregion_crops
from clickagent.multiregion_scorer import build_multiregion_inputs
from clickagent.public_evaluation import point_in_geometry
from crossfit_protocol import digest_object, key
from multiregion_v13_protocol import ARMS, BUDGETS, OUT, SCHEDULE_SEED, SEEDS, sources


REPLAY_PER_SOURCE = 6
REPLAY_ATOL = 1e-4
REPLAY_RTOL = 1e-5
PARAMETERS = 1230338
RAW_KEYS = {'target_attention', 'target_hidden', 'image_hidden', 'vision_hidden'}


def check(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def canonical(path):
    return str(Path(path).resolve()).casefold()


def inventory(mapping):
    result = {}
    for path, digest in mapping.items():
        p = canonical(path)
        check(p not in result or result[p] == digest, 'Conflicting path identity: ' + str(path))
        result[p] = digest
    return result


class Ledger:
    """Full streaming SHA with immutable-file memoization, not a stat-only audit."""
    def __init__(self):
        self.files = {}; self.states = {}; self.bytes_read = 0

    def need(self, path, expected=None):
        p = Path(path).resolve(); name = str(p); stat = p.stat(); state = (stat.st_size, stat.st_mtime_ns)
        if name in self.states:
            check(self.states[name] == state, 'Input changed during audit: ' + name)
            digest = self.files[name]
        else:
            h = hashlib.sha256()
            with p.open('rb') as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    h.update(block); self.bytes_read += len(block)
            digest = h.hexdigest()
            check((p.stat().st_size, p.stat().st_mtime_ns) == state, 'Input changed while hashing: ' + name)
            self.states[name] = state; self.files[name] = digest
        check(expected is None or digest == expected, 'SHA mismatch: ' + name)
        return digest

    def unchanged(self):
        for path, state in self.states.items():
            stat = Path(path).stat()
            check((stat.st_size, stat.st_mtime_ns) == state, 'Input changed before commit: ' + path)


def require_sealed(path, seal, ledger, expected=None):
    known = inventory(seal['files']); target = canonical(path)
    check(target in known, 'Required causal evidence was not sealed: ' + str(path))
    if expected is not None:
        check(known[target] == expected, 'Sealed evidence disagrees with sidecar: ' + str(path))
    return ledger.need(path, known[target])


def cv_partition(train, val, held_source):
    """Remove held-source gradients AND its transitive near-image components."""
    held_components = {c['image_fold_group'] for c in train if c['dataset'] == held_source}
    allowed = [c['sample_id'] for c in train if c['dataset'] != held_source and c['image_fold_group'] not in held_components]
    excluded = [c['sample_id'] for c in train if c['dataset'] != held_source and c['image_fold_group'] in held_components]
    validation = [c['sample_id'] for c in val if c['dataset'] == held_source]
    check(allowed and validation, 'Empty scorer CV partition')
    return allowed, validation, excluded


def validate_components(train):
    unique = {}
    for case in train:
        signature = (case['pixel_sha256'], int(case['dhash256'], 16))
        check(signature not in unique or unique[signature] == case['image_fold_group'], 'Same RGB split between near groups')
        unique[signature] = case['image_fold_group']
    values = list(unique.items()); near_pairs = 0
    for i, ((pixel, dhash), group) in enumerate(values):
        for (other_pixel, other_hash), other_group in values[i + 1:]:
            if pixel == other_pixel or (dhash ^ other_hash).bit_count() <= 4:
                near_pairs += 1
                check(group == other_group, 'Near image split between scorer exclusion components')
    return dict(unique_rgb=len({v[0] for v in unique}), near_pairs_checked=near_pairs)


def expected_schedule(ids, eligible, allowed_ids, epoch):
    from train_crossfit_scorer import epoch_plan
    if epoch == 0:
        return []
    allowed = set(allowed_ids)
    return [[ids[i], budget] for i, budget in epoch_plan(ids, SCHEDULE_SEED, epoch)
            if ids[i] in allowed and eligible[ids[i]][str(budget)] > 0]


def validate_history(history, *, epochs, n_validation, schedules, accumulation):
    check([r['epoch'] for r in history] == list(range(epochs + 1)), 'Missing/duplicate/unordered epochs')
    for row, schedule in zip(history, schedules):
        check(row['queries'] == len(schedule), 'Consumed-query count differs from fixed actual eligible schedule')
        check(row['optimizer_steps'] == math.ceil(len(schedule) / accumulation), 'Optimizer update count differs')
        check(row['schedule_sha256'] == digest_object(schedule), 'Actual query/budget schedule differs')
        v = row['validation']
        check(v['n'] == n_validation and 0 <= v['hits'] <= v['top4_hits'] <= v['oracle'] <= v['named'] <= v['n'],
              'Invalid validation denominator/hit/oracle counts')
        check(math.isfinite(v['loss']) and (not schedule or math.isfinite(row['train_loss'])),
              'Nonfinite training/validation loss')


def replay_sample(cases, per_source=REPLAY_PER_SOURCE):
    """Fixed SHA256(sample_id) order, with no availability or correctness filter."""
    source_names = sorted({c.get('source') or c['dataset'] for c in cases})
    selected = []
    for source in source_names:
        group = [c for c in cases if (c.get('source') or c['dataset']) == source]
        check(len(group) >= per_source, 'Insufficient cases for fixed per-source replay')
        selected.extend(sorted(group, key=lambda c: (hashlib.sha256(c['sample_id'].encode()).hexdigest(), c['sample_id']))[:per_source])
    return selected


def _raw_metadata(path, case, transform, ledger, expected, require_instruction=True, load_arrays=False):
    ledger.need(path, expected)
    with np.load(path, allow_pickle=False) as z:
        check(set(z.files) == RAW_KEYS | {'metadata'}, 'Raw archive has unexpected arrays/teacher/supervision fields')
        meta = json.loads(str(z['metadata']))
        arrays = {name: z[name].copy() for name in RAW_KEYS} if load_arrays else None
    size = transform['crop_size'] if transform else case['image_size']
    check(meta['sample_id'] == case['sample_id'] and meta['image_sha256'] == case['image_sha256'] and
          list(meta['image_size']) == list(size), 'Raw feature belongs to another screenshot/view')
    check(meta['own_token_forward'] and meta['all_backbone_parameters_frozen'] and
          meta['ground_truth_in_model_input'] is False, 'Raw feature is not frozen own-token/GT-free')
    check(not ({'target_geometry', 'labels', 'label', 'annotation', 'target_point', 'ground_truth_point'} & set(meta)),
          'Raw metadata contains supervision fields')
    if require_instruction or 'instruction_sha256' in meta:
        check(meta.get('instruction_sha256') == hashlib.sha256(case['instruction'].encode()).hexdigest(),
              'Raw feature instruction mismatch')
    if transform:
        check(meta['transform'] == transform, 'Raw crop transform mismatch')
    return arrays, meta


@lru_cache(maxsize=1)
def _builder_sources():
    import cache_multiregion_v13 as builder
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (
        Path(builder.__file__), Path('src/clickagent/multiregion_scorer.py'),
        Path('src/clickagent/internal_candidate_scorer.py'))}


def cache_fingerprint(case, condition, evidence, has_labels):
    return digest_object(dict(sample_id=case['sample_id'], condition=condition, image_size=case['image_size'],
        image_sha256=case['image_sha256'], instruction=case['instruction'], evidence=evidence,
        builder_sources=_builder_sources(),
        geometry=case['target_geometry'] if has_labels else None))


def check_cache(path, case, condition, pipeline, pipeline_path, evidence, split, ledger, expected_sha=None):
    """Check point-only supervision and exact independent per-K index sets."""
    ledger.need(path, expected_sha); labels_present = split in ('train', 'val')
    points = np.asarray([r['point'] for r in pipeline['pools'][condition]['32']], np.int64).reshape(-1, 2)
    lookup = {tuple(p): i for i, p in enumerate(points.tolist())}
    check(len(lookup) == len(points), 'Repeated K32 point')
    allowed = {'sample_id', 'available', 'points', 'pipeline_sha256', 'cache_fingerprint', 'annotation_derived_coordinates'} | {
        'pool_indices_' + str(k) for k in BUDGETS}
    if labels_present:
        allowed.add('labels')
    if len(points):
        allowed |= {'i__' + name for name in INPUT_KEYS}
    sizes = {}
    with np.load(path, allow_pickle=False) as z:
        check(set(z.files) == allowed, 'Unexpected/missing cache arrays; GT must stay outside forward inputs')
        check(str(z['sample_id']) == case['sample_id'] and bool(z['available']) == bool(len(points)) and
              not bool(z['annotation_derived_coordinates']), 'Cache sample/availability/coordinate provenance differs')
        check(str(z['pipeline_sha256']) == ledger.need(pipeline_path), 'Cache pipeline identity differs')
        check(str(z['cache_fingerprint']) == cache_fingerprint(case, condition, evidence, labels_present),
              'Cache input fingerprint differs')
        check(z['points'].dtype == np.int64 and np.array_equal(z['points'], points), 'Cache moved proposed coordinates')
        for budget in BUDGETS:
            expected = np.asarray([lookup[tuple(r['point'])] for r in pipeline['pools'][condition][str(budget)]], np.int64)
            value = z['pool_indices_' + str(budget)]
            check(value.dtype == np.int64 and np.array_equal(value, expected), 'Per-budget candidate subset changed')
            sizes[str(budget)] = len(expected)
        if labels_present:
            expected = np.asarray([point_in_geometry((p / np.asarray(case['image_size'])).tolist(), case['target_geometry'])
                                   for p in points], np.float32)
            check(z['labels'].dtype == np.float32 and np.array_equal(z['labels'], expected), 'Point labels do not match actual region hits')
    return sizes


def audit_caches(root, cohorts, seal, ledger):
    eligible = {}; evidence_union = {}; counts = {}; seen_raw = set()
    for split, cases in cohorts.items():
        by_condition = {}; pipeline_map = {}
        for condition in ('c1', 'c3'):
            ip = root / 'cache' / f'{split}_{condition}_index.json'; index = read(ip); ledger.need(ip)
            if split in ('train', 'val'):
                require_sealed(ip, seal, ledger)
            check(index['complete'] and index['pending'] == 0 and index['split'] == split and index['condition'] == condition,
                  'Incomplete/wrong cache index')
            check(index['labels_present'] == (split in ('train', 'val')) and index['annotation_derived_coordinates'] is False,
                  'Wrong cache supervision provenance')
            check([r['sample_id'] for r in index['cases']] == [c['sample_id'] for c in cases], 'Cache index omitted/reordered cases')
            available = {}; nonempty = 0
            for i, (case, entry) in enumerate(zip(cases, index['cases'])):
                sid = case['sample_id']; pp = root / 'pipeline' / split / (key(sid) + '.json')
                if sid not in pipeline_map:
                    ledger.need(pp); p = read(pp); pipeline_map[sid] = p
                    check(p['complete'] and p['sample_id'] == sid and p['ground_truth_in_model_inputs'] is False, 'Invalid public pipeline')
                    check(p['pools'] == build_c1_c3_pools(p['global_result'], p['local_views']), 'Pools differ from actual decoded traces')
                    plans = plan_multiregion_crops(p['global_result'], case['image_size'])
                    check([v['transform'] for v in p['local_views']] == [v['transform'] for v in plans], 'Search crop plan changed/used another policy')
                    raw_views = [(p['global_feature_path'], p['global_feature_sha256'], None)] + [
                        (v['feature_path'], v['feature_sha256'], v['transform']) for v in p['local_views']]
                    for raw, digest, transform in raw_views:
                        marker = (canonical(raw), sid, digest_object(transform))
                        if marker not in seen_raw:
                            _raw_metadata(raw, case, transform, ledger, digest,
                                require_instruction=split not in ('train', 'val'))
                            seen_raw.add(marker)
                        if split not in ('train', 'val'):
                            check(Path(raw).stat().st_mtime >= seal['created_at'], 'Fresh raw features predate scorer freeze')
                    if split not in ('train', 'val'):
                        check(p['proposer_role'] == 'full', 'Fresh deployment does not use shared full-D proposer')
                p = pipeline_map[sid]
                expected_evidence = {str(pp): ledger.need(pp), p['global_feature_path']: p['global_feature_sha256']}
                expected_evidence.update({v['feature_path']: v['feature_sha256'] for v in p['local_views'][:1 if condition == 'c1' else 3]})
                check(inventory(entry['evidence']) == inventory(expected_evidence), 'Cache evidence omitted or changed raw/pipeline input')
                for path, digest in entry['evidence'].items():
                    ledger.need(path, digest)
                path = Path(entry['path'])
                available[sid] = check_cache(path, case, condition, p, pp, entry['evidence'], split, ledger, entry['sha256'])
                check(entry['available'] == bool(available[sid]['32']) and entry['n_points'] == available[sid]['32'],
                      'Cache index count differs from actual pool')
                nonempty += int(entry['available'])
                if split in ('train', 'val'):
                    evidence_union[str(path)] = entry['sha256']; evidence_union.update(entry['evidence'])
                else:
                    check(path.stat().st_mtime >= seal['created_at'], 'Fresh cache predates scorer freeze')
                if (i + 1) % 100 == 0:
                    print(f'AUDIT cache {split}/{condition} {i+1}/{len(cases)}', flush=True)
            by_condition[condition] = available; counts[f'{split}_{condition}'] = dict(cases=len(cases), nonempty=nonempty)
        eligible[split] = by_condition
    check(inventory(evidence_union) == inventory(seal['training_evidence']), 'Frozen training_evidence inventory is not complete/exact')
    return eligible, counts


def audit_training(root, cohorts, eligible, seal, ledger, epochs, accumulation):
    import torch
    from dataclasses import asdict
    from clickagent.internal_candidate_scorer import InternalCandidateScorerConfig
    from clickagent.multiregion_scorer import MultiRegionScorer
    from train_crossfit_scorer import state_identity
    source_identity = {canonical(p): ledger.need(p) for p in sources()}
    train, val = cohorts['train'], cohorts['val']; ids = [c['sample_id'] for c in train]
    expected_init = {}
    for suffix, seed in SEEDS.items():
        torch.manual_seed(seed); model = MultiRegionScorer(InternalCandidateScorerConfig(head_mode='full'))
        expected_init[suffix] = state_identity(model)
        check(sum(p.numel() for p in model.parameters()) == PARAMETERS, 'Scorer parameter contract changed')
    full = {}; cv = {}; paired = []
    for held in (None, *sorted({c['dataset'] for c in train})):
        arms = ARMS if held is None else ('c3_top1', 'c3_top4')
        for arm in arms:
            condition = arm.split('_')[0]
            for suffix in SEEDS if held is None else ('s1',):
                index = sorted({c['dataset'] for c in train}).index(held) if held else None
                name = f'{arm}_{suffix}' if held is None else f'{arm}_held{index}_{suffix}'
                path = root / ('training' if held is None else 'cross_validation') / (name + '.json')
                meta = read(path); require_sealed(path, seal, ledger)
                allowed, validation, excluded = (ids, [c['sample_id'] for c in val], []) if held is None else cv_partition(train, val, held)
                check(meta['complete'] and meta['name'] == name and meta['arm'] == arm and meta['seed'] == SEEDS[suffix] and
                      meta['schedule_seed'] == SCHEDULE_SEED and meta['held_source'] == held, 'Wrong training/CV identity')
                check(meta['train_sample_ids'] == allowed and meta['validation_sample_ids'] == validation and
                      meta['excluded_cross_source_component_ids'] == excluded, 'CV gradient IDs include held source/near component or wrong val')
                check(meta['held_source_labels_used_for_gradient'] is False and meta['upstream_seen_held_source'] == bool(held),
                      'Wrong source-heldout causal scope declaration')
                check(meta['parameters'] == PARAMETERS and meta['initial_state_sha256'] == expected_init[suffix], 'Wrong fresh seeded scorer initialization')
                check(inventory(meta['source_identity']) == source_identity, 'Training source identity changed or omitted source')
                expected_inputs = {str(root / 'cache' / f'{split}_{condition}_index.json'):
                                   ledger.need(root / 'cache' / f'{split}_{condition}_index.json') for split in ('train', 'val')}
                check(inventory(meta['input_identity']) == inventory(expected_inputs) and
                      meta['protocol_sha256'] == ledger.need(root / 'protocol.json'), 'Training input/protocol identity differs')
                schedules = [expected_schedule(ids, eligible['train'][condition], allowed, epoch) for epoch in range(epochs + 1)]
                validate_history(meta['history'], epochs=epochs, n_validation=len(validation), schedules=schedules, accumulation=accumulation)
                if held is None:
                    full[name] = meta
                    for row in meta['history']:
                        require_sealed(row['checkpoint_path'], seal, ledger, row['checkpoint_sha256'])
                else:
                    cv[name] = meta
    reference = full['c1_top1_s1']['history']
    for name, meta in full.items():
        check([r['schedule_sha256'] for r in meta['history']] == [r['schedule_sha256'] for r in reference] and
              [r['queries'] for r in meta['history']] == [r['queries'] for r in reference] and
              [r['optimizer_steps'] for r in meta['history']] == [r['optimizer_steps'] for r in reference],
              'Six full trajectories used different query/budget schedules or update counts')
        paired.append(dict(model=name, initial_state_sha256=meta['initial_state_sha256'],
            total_queries=sum(r['queries'] for r in meta['history']),
            total_optimizer_steps=sum(r['optimizer_steps'] for r in meta['history'])))
    selection_path = root / 'models/selection.json'; selection = read(selection_path); require_sealed(selection_path, seal, ledger)
    from evaluate_multiregion_v13 import HEAD_NAMES
    check(selection['complete'] and selection['fresh_test_read'] is False and selection['seed_selected_by_test'] is False and
          set(selection['selections']) == set(HEAD_NAMES), 'Invalid/incomplete pretest selection')
    for objective in ('top1', 'top4'):
        folds = [cv[f'c3_{objective}_held{f}_s1'] for f in range(4)]; field = 'hits' if objective == 'top1' else 'top4_hits'
        curves = []
        for epoch in range(epochs + 1):
            values = [f['history'][epoch]['validation'] for f in folds]
            curves.append(dict(epoch=epoch, macro_rate=float(np.mean([v[field] / v['n'] for v in values])),
                mean_objective_loss=float(np.mean([v['loss'] for v in values])), folds=values))
        chosen = max(curves, key=lambda r: (r['macro_rate'], -r['mean_objective_loss'], -r['epoch']))
        check(selection['cross_validation'][objective]['curve'] == curves and
              selection['cross_validation'][objective]['selected_epoch'] == chosen['epoch'], 'LOSO macro public-epoch selector differs')
    for name, selected in selection['selections'].items():
        condition, objective, suffix, policy = name.split('_'); arm = condition + '_' + objective
        meta = full[arm + '_' + suffix]; field = 'hits' if objective == 'top1' else 'top4_hits'
        expected_epoch = (max(meta['history'], key=lambda r: (r['validation'][field], -r['validation']['loss'], -r['epoch']))['epoch']
            if policy == 'val' else selection['cross_validation'][objective]['selected_epoch'])
        row = meta['history'][expected_epoch]
        check(selected['name'] == name and selected['condition'] == condition and selected['objective'] == objective and
              selected['policy'] == policy and selected['seed'] == SEEDS[suffix] and selected['epoch'] == expected_epoch,
              'Selected head metadata/epoch differs from fixed selector')
        check(canonical(selected['source_checkpoint_path']) == canonical(row['checkpoint_path']) and
              selected['sha256'] == row['checkpoint_sha256'] and selected['validation'] == row['validation'] and
              selected['training_identity'] == ledger.need(root / 'training' / (arm + '_' + suffix + '.json')),
              'Selected checkpoint differs from actual training epoch sidecar')
        require_sealed(selected['checkpoint_path'], seal, ledger, selected['sha256'])
        payload = torch.load(selected['checkpoint_path'], map_location='cpu', weights_only=True)
        check(payload['schema_version'] == 13 and payload['config'] == asdict(InternalCandidateScorerConfig(head_mode='full')) and
              payload['metadata'] == dict(arm=arm, seed=SEEDS[suffix], epoch=expected_epoch,
            initial_state_sha256=expected_init[suffix], protocol_sha256=ledger.need(root / 'protocol.json')),
            'Actual checkpoint metadata differs from selected epoch')
    return selection, dict(full_models=6, cv_models=8, selected_aliases=10, initialized_state_sha256=expected_init,
        paired_full_trajectories=paired, schedule_seed=SCHEDULE_SEED, shared_epoch_selection_recomputed=True,
        cv_gradient_source_and_near_components_excluded=True,
        limitation='Scorer source-heldout CV tuning only; frozen upstream saw sources and original validation labels')


def rebuild_inputs(case, pipeline, condition, cache_path, ledger):
    """Only frozen raw arrays + public view geometry enter the input builder."""
    ga, gm = _raw_metadata(pipeline['global_feature_path'], case, None, ledger,
        pipeline['global_feature_sha256'], load_arrays=True)
    local = []
    for view in pipeline['local_views'][:1 if condition == 'c1' else 3]:
        a, m = _raw_metadata(view['feature_path'], case, view['transform'], ledger,
            view['feature_sha256'], load_arrays=True)
        local.append((a, m, view['transform']))
    points = np.asarray([p['point'] for p in pipeline['pools'][condition]['32']], np.int64).reshape(-1, 2)
    inputs = build_multiregion_inputs(ga, gm, local, points, case['image_size'])
    check(set(inputs) == INPUT_KEYS, 'GT/unknown model input key')
    compared = []
    with np.load(cache_path, allow_pickle=False) as z:
        check('labels' not in z.files, 'Fresh replay cache contains labels')
        cached_names = {n[3:] for n in z.files if n.startswith('i__')}
        check(cached_names == (set(INPUT_KEYS) if len(points) else set()), 'Wrong cached forward input fields')
        for name in sorted(cached_names):
            stored = z['i__' + name]
            check(stored.dtype == inputs[name].dtype and np.array_equal(stored, inputs[name]),
                  'Raw→cache exact input reconstruction failed: ' + name)
            compared.append(name)
        ix = z['pool_indices_16'].astype(np.int64)
    return inputs, points, ix, dict(condition=condition, exact_input_arrays=compared,
        empty_cache_has_no_i_arrays=len(points) == 0, proposed_points_exact=True)


def compare_replay(actual, saved, points, *, atol=REPLAY_ATOL, rtol=REPLAY_RTOL):
    """Fixed numeric allowance never excuses a changed selected coordinate."""
    actual = np.asarray(actual, dtype=np.float32); expected = np.asarray(saved['scores'], dtype=np.float64)
    check(actual.shape == expected.shape == (len(points),) and np.isfinite(actual).all() and np.isfinite(expected).all(),
          'CPU/saved score shape or finite check failed')
    difference = actual.astype(np.float64) - expected
    check(np.allclose(actual, expected, atol=atol, rtol=rtol), 'CPU score replay exceeds fixed registered tolerance')
    selected = int(actual.argmax()) if len(points) else None
    point = points[selected].tolist() if selected is not None else None
    check(point == saved['point'], 'CPU argmax point differs from saved prediction')
    if len(points):
        check(saved['selected_index'] == selected, 'CPU selected candidate index differs')
    cpu_order = np.argsort(-actual, kind='stable')[:4]
    saved_order = np.argsort(-expected, kind='stable')[:4]
    return dict(point=point, selected_point_exact=True, logits_bitwise_equal=bool(np.array_equal(actual, expected)),
        max_abs_difference=float(np.max(np.abs(difference))) if len(difference) else 0.,
        score_differences=difference.tolist(), cpu_scores=actual.tolist(), saved_scores=expected.tolist(),
        cpu_top4_indices=cpu_order.tolist(), saved_top4_indices=saved_order.tolist(),
        top4_order_exact=bool(np.array_equal(cpu_order, saved_order)), atol=atol, rtol=rtol,
        saved_scoring_device=saved.get('scoring_device', 'unspecified'))


def audit_replay(root, primary, selection, ledger):
    import torch
    from clickagent.internal_candidate_scorer import internal_inputs_to_torch, subset_internal_inputs
    from clickagent.multiregion_scorer import load_multiregion_scorer
    chosen = replay_sample(primary); check(len(chosen) == 12, 'Expected fixed two-source 12-case replay')
    models = {name: load_multiregion_scorer(row['checkpoint_path'], 'cpu')[0].eval()
              for name, row in selection['selections'].items()}
    indices = {condition: {r['sample_id']: r for r in read(root / 'cache' / f'test_{condition}_index.json')['cases']}
               for condition in ('c1', 'c3')}
    results = []; rebuilt = []
    for case in chosen:
        sid = case['sample_id']; pipeline = read(root / 'pipeline/test' / (key(sid) + '.json'))
        saved = read(root / 'predictions/test' / (key(sid) + '.json'))
        by_method = {(r['method'], r.get('budget')): r for r in saved['predictions']}
        for condition in ('c1', 'c3'):
            entry = indices[condition][sid]; ledger.need(entry['path'], entry['sha256'])
            inputs, points, ix, proof = rebuild_inputs(case, pipeline, condition, entry['path'], ledger)
            proof['sample_id'] = sid; rebuilt.append(proof)
            batch = internal_inputs_to_torch(subset_internal_inputs(inputs, ix), 'cpu')
            for name, model in models.items():
                if selection['selections'][name]['condition'] != condition:
                    continue
                with torch.inference_mode():
                    out = model(**batch)
                result = compare_replay(out['scores'].detach().float().numpy(), by_method[(name, 16)], points[ix])
                result.update(sample_id=sid, source=case.get('source') or case['dataset'], method=name, budget=16)
                results.append(result)
        print('AUDIT CPU replay ' + sid, flush=True)
    check(len(results) == 120, 'Actual CPU forward count is not 120')
    return dict(sample_rule='Per source first six sorted by (sha256(sample_id),sample_id); no outcome/availability filtering',
        sample_ids=[c['sample_id'] for c in chosen], actual_cpu_forwards=len(results), cpu_threads=4,
        atol=REPLAY_ATOL, rtol=REPLAY_RTOL, input_rebuilds=rebuilt, results=results,
        top4_note='Saved GPU/CPU producer rankings remain canonical; any near-tie replay order differences are disclosed, never substituted')


def audit_evaluation(root, cohorts, ledger):
    from evaluate_multiregion_v13 import evaluate_records, load_records
    result = {}
    for split in ('test', 'supplement'):
        dest = root / 'evaluation' / split; summary = read(dest / 'summary.json'); proof = read(dest / 'audit.json')
        for path in (dest / 'summary.json', dest / 'details.json', dest / 'scored_predictions.jsonl', dest / 'audit.json', dest / 'report.html'):
            ledger.need(path)
        check(summary['complete'] and proof['complete'] and summary['n'] == proof['n'] == len(cohorts[split]) and
              proof['methods'] == 56 and proof['predictions'] == 56 * len(cohorts[split]), 'Incomplete saved evaluation')
        check(proof['registered_method_completeness_checked'] and proof['decoded_pool_trace_rebuilt'] and
              proof['native_and_global_points_unchanged'] and proof['sealed_evaluator_sources_verified'],
              'Saved evaluator record-audit checks were not executed')
        for field, filename in (('summary_sha256', 'summary.json'), ('details_sha256', 'details.json'),
                                 ('scored_predictions_sha256', 'scored_predictions.jsonl')):
            check(proof[field] == ledger.need(dest / filename), 'Evaluator audit bound to another output')
        for evidence in (summary, proof):
            check(evidence['evaluator_sha256'] == ledger.need('scripts/evaluate_multiregion_v13.py'), 'Evaluator source changed')
            for path, digest in {**evidence['input_sha256'], **evidence['evaluator_dependencies']}.items():
                ledger.need(path, digest)
        cases, predictions, pipelines, gui, inputs = load_records(root, split)
        for path in inputs:
            ledger.need(path)
        expected_inputs = {str(p): ledger.need(p) for p in [*inputs, root / 'frozen.json', root / 'models/selection.json']}
        check(inventory(summary['input_sha256']) == inventory(expected_inputs) and
              inventory(proof['input_sha256']) == inventory(expected_inputs), 'Evaluation omitted/changed actual input identities')
        indices = {c: {r['sample_id']: r for r in read(root / 'cache' / f'{split}_{c}_index.json')['cases']}
                   for c in ('c1', 'c3')}
        for case in cases:
            stem = key(case['sample_id']); pp = root / 'pipeline' / split / (stem + '.json')
            hp = root / 'predictions' / split / (stem + '.json'); gp = root / 'gui_reference' / split / (stem + '.json')
            native = root / 'native' / split / (stem + '.json')
            head, reference = read(hp), read(gp)
            expected_head = dict(frozen_sha256=ledger.need(root / 'frozen.json'), pipeline_sha256=ledger.need(pp),
                selection_sha256=ledger.need(root / 'models/selection.json'), native_sha256=ledger.need(native),
                source_sha256=ledger.need('scripts/run_multiregion_v13.py'),
                cache_sha256={c: indices[c][case['sample_id']]['sha256'] for c in ('c1', 'c3')})
            expected_gui = dict(frozen_sha256=ledger.need(root / 'frozen.json'), pipeline_sha256=ledger.need(pp),
                heads_sha256=ledger.need(hp), source_sha256=ledger.need('scripts/run_multiregion_v13.py'),
                verifier_source_sha256=ledger.need('scripts/run_gui_verifier.py'),
                verifier_asset_identity_sha256=ledger.need('models/GUI-Actor-Verifier-2B/clickagent_download_verified.json'))
            check(head['input_identity'] == expected_head and reference['input_identity'] == expected_gui,
                  'Saved inference was resumed using another model/pipeline/cache/verifier identity')
        recomputed = evaluate_records(cases, predictions, pipelines, gui_records=gui, cohort=split)
        check(all(summary[k] == value for k, value in recomputed['summary'].items()), 'Saved summary metric differs from full replay')
        check(read(dest / 'details.json') == recomputed['details'] and rows(dest / 'scored_predictions.jsonl') == recomputed['scored_predictions'],
              'Saved per-case metrics differ from full replay')
        check(proof['independent_geometry'] == recomputed['audit']['independent_geometry'], 'Independent geometry replay differs')
        result[split] = dict(cases=len(cases), methods=56, summary_sha256=ledger.need(dest / 'summary.json'),
            independent_geometry=recomputed['audit']['independent_geometry'])
    return result


def run_audit(root):
    root = Path(root); ledger = Ledger(); print('AUDIT sealed files and full training evidence', flush=True)
    seal_path = root / 'frozen.json'; seal = read(seal_path); ledger.need(seal_path)
    check(seal['complete'] and seal['fresh_test_features_at_freeze'] == 0 and seal['all_training_evidence_verified'], 'Invalid pretest freeze')
    for section in ('files', 'training_evidence'):
        for i, (path, digest) in enumerate(seal[section].items()):
            ledger.need(path, digest)
            if (i + 1) % 100 == 0:
                print(f'AUDIT SHA {section} {i+1}/{len(seal[section])} bytes={ledger.bytes_read}', flush=True)
    require_sealed(__file__, seal, ledger)
    for path in sources():
        require_sealed(path, seal, ledger)
    cohorts = {split: rows(root / 'data' / (split + '.jsonl')) for split in ('train', 'val', 'test', 'supplement')}
    check({s: len(c) for s, c in cohorts.items()} == dict(train=873, val=172, test=192, supplement=6), 'Cohort incomplete')
    for split in cohorts:
        require_sealed(root / 'data' / (split + '.jsonl'), seal, ledger)
    protocol = read(root / 'protocol.json'); require_sealed(root / 'protocol.json', seal, ledger)
    check(protocol['optimization']['epochs'] == 8 and protocol['optimization']['accumulation'] == 8, 'Unexpected fixed training protocol')
    component_proof = validate_components(cohorts['train'])
    print('AUDIT all fixed points/labels/public raw metadata', flush=True)
    eligible, cache_counts = audit_caches(root, cohorts, seal, ledger)
    print('AUDIT full/CV training and selected checkpoint identities', flush=True)
    selection, training = audit_training(root, cohorts, eligible, seal, ledger, epochs=8, accumulation=8)
    print('AUDIT saved decision/metric completeness', flush=True)
    evaluation = audit_evaluation(root, cohorts, ledger)
    print('AUDIT actual image bytes for both fresh cohorts', flush=True)
    for case in cohorts['test'] + cohorts['supplement']:
        ledger.need(case['image'], case['image_sha256'])
    print('AUDIT fixed 12 cases / 120 CPU forwards', flush=True)
    replay = audit_replay(root, cohorts['test'], selection, ledger)
    ledger.unchanged()
    return dict(complete=True, version=13, created_at=time.time(), auditor_sha256=ledger.need(__file__),
        frozen_sha256=ledger.need(seal_path), sealed_files_fully_rehashed=len(seal['files']),
        training_evidence_fully_rehashed=len(seal['training_evidence']), bytes_read=ledger.bytes_read,
        near_components=component_proof, cache_counts=cache_counts, training=training, evaluation=evaluation,
        fresh_image_sha256_checks=198, cpu_replay=replay, input_sha256=ledger.files,
        gpu_used=False, model_training=False,
        scope='Full declared seal/training-evidence bytes, scorer/CV provenance, all saved metrics and fixed 120 CPU forwards. Runtime weight files referenced only transitively are not additionally claimed as rehashed unless present in the declared seal.')


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--root', type=Path, default=OUT)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(4)
    result = run_audit(args.root)
    write_json(args.root / 'final_audit.json', result)
    print(json.dumps(dict(complete=True, actual_cpu_forwards=result['cpu_replay']['actual_cpu_forwards'],
                         fresh_images=198)), flush=True)


if __name__ == '__main__':
    main()
