"""Independent CPU checks before V12 test extraction and after final scoring.

Pretest checks training provenance, complete histories, matched randomness,
actual candidate coordinates/labels, and the whole assigned-proposer chain.
Final additionally verifies the frozen pretest proof and every test prediction.
Neither stage trains a model, loads a GPU runtime, or changes frozen inputs.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import traceback
import time

import numpy as np


OUT = Path('runs/crossfit_candidates_v12')
BUDGETS = (4, 8, 16, 32)
ROLES = tuple(f'fold_{i}' for i in range(5)) + ('full',)
ARMS = ('in_sample', 'oof')


def check(condition, message):
    if not condition:
        raise ValueError(message)


@lru_cache(maxsize=256)
def _read_cached(path, size, modified):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def read(path):
    p = Path(path).resolve(); s = p.stat()
    return _read_cached(str(p), s.st_size, s.st_mtime_ns)


def rows(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8-sig').splitlines() if s.strip()]


@lru_cache(maxsize=16)
def _instruction_reference(path, size, modified):
    return {r['sample_id']: r['instruction'] for r in rows(path)}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def key(sid):
    return hashlib.sha256(sid.encode()).hexdigest()[:24]


class Ledger:
    """Hash each immutable file once per unchanged stat identity, streaming bytes."""
    def __init__(self):
        self.files = {}; self.stats = {}

    def need(self, path, expected=None):
        p = Path(path).resolve(); name = str(p)
        stat = p.stat(); identity = (stat.st_size, stat.st_mtime_ns)
        if self.stats.get(name) != identity:
            if stat.st_size > 512 * 1024**2:
                print(f'AUDIT hash {p.name} ({stat.st_size} bytes)', flush=True)
            h = hashlib.sha256()
            with p.open('rb') as stream:
                while block := stream.read(1024 * 1024): h.update(block)
            self.files[name] = h.hexdigest(); self.stats[name] = identity
        actual = self.files[name]
        check(expected is None or actual == expected, f'SHA mismatch: {p}')
        return actual

    def unchanged(self):
        for name, old in self.stats.items():
            s = Path(name).stat()
            check((s.st_size, s.st_mtime_ns) == old, f'Input changed during audit: {name}')


def validate_folds(train, val, folds):
    """Pure metadata check, also usable with small synthetic fold fixtures."""
    train_ids = {c['sample_id'] for c in train}; val_ids = {c['sample_id'] for c in val}
    check(len(train_ids) == len(train) and not train_ids & val_ids, 'Repeated train IDs or train/val overlap')
    check(set(folds['sample_to_fold']) == train_ids and folds['n_folds'] == 5, 'Incorrect fold membership')
    pixels = {}; hashes = {}
    for c in train:
        sid = c['sample_id']; f = int(folds['sample_to_fold'][sid])
        check(0 <= f < 5 and c['image_fold'] == f, 'Row fold differs from registered map')
        check(c['oof_provider'] == f'fold_{f}' and c['in_sample_provider'] == f'fold_{(f+1)%5}', 'Wrong seen/OOF mapping')
        old = pixels.setdefault(c['pixel_sha256'], f)
        check(old == f, 'Identical screenshot split between folds')
        h = int(c['dhash256'], 16)
        check(c['pixel_sha256'] not in hashes or hashes[c['pixel_sha256']] == h, 'Same RGB has conflicting dHash')
        hashes[c['pixel_sha256']] = h
    items = sorted(pixels)
    near_pairs = 0
    for i, a in enumerate(items):
        for b in items[:i]:
            if (hashes[a] ^ hashes[b]).bit_count() <= 4:
                check(pixels[a] == pixels[b], 'Near screenshots split between folds'); near_pairs += 1
    for role in ROLES:
        spec = folds['roles'][role]; allowed, heldout = set(spec['allowed_train_ids']), set(spec['heldout_ids'])
        expected = set() if role == 'full' else {sid for sid in train_ids if folds['sample_to_fold'][sid] == int(role[-1])}
        check(heldout == expected and allowed == train_ids - expected, f'Incorrect allowlist: {role}')
        check(not allowed & heldout, f'Allowlist overlaps holdout: {role}')
        check(spec['allowed_train_ids_sha256'] == digest(sorted(allowed)) and
              spec['heldout_ids_sha256'] == digest(sorted(heldout)), 'Allowlist digest mismatch')
        selected_pixels = {c['pixel_sha256'] for c in train if c['sample_id'] in allowed}
        heldout_pixels = {c['pixel_sha256'] for c in train if c['sample_id'] in heldout}
        check(not selected_pixels & heldout_pixels, f'Image exposure leak: {role}')
        check(set(spec['allowed_train_groups']) == selected_pixels and set(spec['heldout_groups']) == heldout_pixels,
              'Declared fold image groups differ')
    val_pixels = {c['pixel_sha256'] for c in val}
    check(not val_pixels & set(pixels), 'Train/validation screenshot overlap')
    for c in val:
        h = int(c['dhash256'], 16)
        check(all((h ^ v).bit_count() > 4 for v in hashes.values()), 'Near train/validation overlap')
    return dict(train_queries=len(train), train_rgb_groups=len(pixels), train_near_pairs=near_pairs,
                fold_queries=dict(Counter(c['image_fold'] for c in train)))


def selected_row(history, epochs, n_validation):
    check([r['epoch'] for r in history] == list(range(epochs + 1)), 'Incomplete or unordered epoch history')
    best = None; selected = None
    for row in history:
        v = row['validation']; check(v['n'] == n_validation and math.isfinite(v['loss']), 'Invalid validation record')
        oracle = v.get('point_oracle', v.get('oracle'))
        check(0 <= v['hits'] <= oracle <= v['named'] <= n_validation, 'Impossible validation counts')
        score = (v['hits'], -v['loss'])
        if best is None or score > best: best, selected = score, row
    return selected


def validate_validation_replay(expected, actual):
    check(set(expected) == set(actual), 'Validation replay fields differ')
    for field in ('n', 'named', 'hits', 'oracle', 'refused'):
        check(actual[field] == expected[field], f'Validation replay count differs: {field}')
    check(np.isclose(actual['loss'], expected['loss'], rtol=1e-5, atol=1e-6), 'Validation replay loss differs materially')
    return dict(counts_exact=True, selected_loss=expected['loss'], replay_loss=actual['loss'],
                loss_absolute_difference=abs(actual['loss'] - expected['loss']), loss_rtol=1e-5, loss_atol=1e-6,
                selection_unchanged=True)


def audit_data(root, ledger):
    protocol = read(root / 'protocol.json'); folds = read(root / 'folds.json'); selection = read(root / 'data_selection.json')
    manifests = {s: rows(root / 'data' / (s + '.jsonl')) for s in ('train', 'val', 'test')}
    check(protocol['counts'] == {s: len(v) for s, v in manifests.items()} == dict(train=873, val=172, test=192),
          'Unexpected experimental cohort counts')
    check(selection['complete'] and selection['historical_unresolved'] == [] and selection['selection_used_predictions'] is False,
          'Incomplete/unresolved historical exclusion')
    for name, expected in [('manifest.jsonl', protocol['manifest_sha256']), ('folds.json', protocol['folds_sha256']),
                           ('data_selection.json', protocol['data_selection_sha256'])]: ledger.need(root / name, expected)
    for s in manifests: ledger.need(root / 'data' / (s + '.jsonl'), selection['split_manifest_sha256'][s])
    for path, expected in selection['source_identity'].items(): ledger.need(path, expected)
    for path, expected in selection['historical_manifest_sha256'].items(): ledger.need(path, expected)
    all_rows = rows(root / 'manifest.jsonl')
    check(len({r['sample_id'] for r in all_rows}) == len(all_rows), 'Repeated manifest case')
    for split in manifests:
        check([r for r in all_rows if r['experiment_split'] == split] == manifests[split], 'Split files differ from manifest')
    evidence = validate_folds(manifests['train'], manifests['val'], folds)
    test = manifests['test']; check(len({c['pixel_sha256'] for c in test}) == 192, 'Repeated fresh test screenshot')
    check(Counter(c['source'] for c in test) == dict(screenspot_v2=64, screenspot_pro=64, venusbench_gd=64), 'Wrong source quotas')
    from prepare_candidate_scorer import resolve_history
    _, historical_pixels, historical_hashes, _, _ = resolve_history([Path(p) for p in selection['historical_manifest_sha256']])
    from prepare_semantic_head_data import HammingIndex
    tree = HammingIndex(historical_hashes)
    for c in test:
        h = int(c['dhash256'], 16)
        check(c['pixel_sha256'] not in historical_pixels and not tree.near(h, 4), 'Historical/within-test near screenshot leak')
        tree.add(h)
        check(c['provider'] == c['oof_provider'] == c['in_sample_provider'] == 'full' and c['image_fold'] is None,
              'Test is not common full-D deployment')
    check(all(c['provider'] == c['oof_provider'] == c['in_sample_provider'] == 'full' for c in manifests['val']),
          'Validation does not use common full-D deployment')
    return protocol, folds, manifests, evidence


def audit_proposers(root, protocol, folds, cohorts, ledger):
    import torch
    from clickagent.crossfit_proposer import make_fresh_proposer, CrossfitProposerConfig, state_sha256
    from train_crossfit_proposer import object_sha, SOURCE_PATHS
    fresh = make_fresh_proposer(CrossfitProposerConfig()); expected_init = state_sha256(fresh)
    expected_config = asdict(CrossfitProposerConfig()); train = {c['sample_id']: c for c in cohorts['train']}
    val = {c['sample_id']: c for c in cohorts['val']}; models = {}; cpu_validation = {}; raw_metadata = {}
    archive_path = root / 'upstream_source_archive/manifest.json'; archive = read(archive_path); ledger.need(archive_path)
    race_path = root / 'checks/source_race_resolution.json'; race = read(race_path); ledger.need(race_path)
    check(archive['complete'] and race['complete'] and race['expected_and_restored_sha256'] == archive['restored_training_source_sha256'],
          'Unresolved upstream source restoration')
    check(set(archive['files']) == set(SOURCE_PATHS), 'Upstream source archive is incomplete')
    for source, expected in archive['files'].items():
        ledger.need(source, expected); ledger.need(root / 'upstream_source_archive' / source, expected)
    ledger.need(root / 'checks/old_train_crossfit_proposer.py', race['expected_and_restored_sha256'])
    ledger.need(root / race['diff'])
    smoke_path = root / 'checks/proposer_smoke_frozen_source.json'; smoke = read(smoke_path); ledger.need(smoke_path)
    check(smoke['complete'] and smoke['gpu_used'] is False and smoke['source_unchanged'] and
          smoke['source_sha256'] == archive['files'] and smoke['frozen_train_source_sha256'] == archive['restored_training_source_sha256'] and
          smoke['initial_state_sha256'] == expected_init and smoke['constant_zero_prior'] and
          smoke['finite_loss_and_backward'] and smoke['strict_save_load_predictions_equal'], 'Restored-source CPU smoke proof differs')
    for role in ROLES:
        path = root / 'proposers' / (role + '.json'); meta = read(path); identity = meta['identity']
        ledger.need(path); ledger.need(path.with_suffix('.pt'), meta['checkpoint_sha256'])
        check(meta['complete'] and meta['role'] == identity['role'] == role, 'Incomplete/wrong proposer role')
        check(meta['initial_state_sha256'] == expected_init and identity['config'] == expected_config, 'Nonfresh proposer initialization/config')
        check(meta['parameters'] == protocol['upstream']['parameters'], 'Proposer architecture changed')
        check(meta['n_train'] == len(folds['roles'][role]['allowed_train_ids']) and meta['n_validation'] == len(val) and
              0 < meta['named_train'] <= meta['n_train'], 'Wrong proposer train/validation counts')
        check(identity['heldout_labels_used'] is False and identity['old_learned_weights_used'] is False and
              identity['teacher_points_used'] is False and identity['head_prior'] == 'all-zero constant96', 'Legacy supervision in fresh proposer')
        spec = folds['roles'][role]
        check(meta['allowed_train_ids'] == identity['allowed_train_ids'] == spec['allowed_train_ids'] and
              meta['heldout_ids'] == identity['heldout_ids'] == spec['heldout_ids'], 'Proposer allowlist differs from fold')
        check(identity['validation_ids'] == sorted(val), 'Wrong upstream validation cohort')
        check(identity['protocol_sha256'] == meta['protocol_sha256'] == ledger.need(root / 'protocol.json'), 'Proposer protocol differs')
        for key_name, p in [('train_cases_sha256', root / 'data/train.jsonl'), ('val_cases_sha256', root / 'data/val.jsonl'),
                            ('folds_sha256', root / 'folds.json')]: ledger.need(p, identity[key_name])
        check(set(identity['source_sha256']) == set(SOURCE_PATHS) and meta['source_sha256'] == identity['source_sha256'], 'Incomplete proposer source identity')
        check(identity['source_sha256'] == archive['files'], 'A proposer used a different running upstream source')
        for p, expected in identity['source_sha256'].items(): ledger.need(p, expected)
        allowed = set(spec['allowed_train_ids']); raw = identity['raw_sources']
        check(set(raw) == allowed | set(val), 'Proposer raw sources include heldout or omit allowed input')
        for sid, value in raw.items():
            case = train.get(sid, val.get(sid)); check(Path(value['path']).resolve() == Path(case['raw_path']).resolve(), 'Wrong proposer raw file')
            ledger.need(value['path'], value['sha256']); check(Path(value['path']).stat().st_size == value['size'], 'Raw size changed')
            resolved = str(Path(value['path']).resolve())
            if resolved not in raw_metadata:
                with np.load(value['path'], allow_pickle=False) as z: raw_metadata[resolved] = json.loads(str(z['metadata']))
            raw_meta = raw_metadata[resolved]
            check(raw_meta['sample_id'] == sid and raw_meta['image_sha256'] == case['image_sha256'] and
                  raw_meta['own_token_forward'] and raw_meta['all_backbone_parameters_frozen'] and
                  raw_meta['ground_truth_in_model_input'] is False, 'Unsafe upstream raw source')
        named_train = sum(raw_metadata[str(Path(raw[sid]['path']).resolve())]['target_available'] for sid in allowed)
        named_val = sum(raw_metadata[str(Path(raw[sid]['path']).resolve())]['target_available'] for sid in val)
        check(meta['named_train'] == named_train and all(r['validation']['named'] == named_val for r in meta['history']),
              'Upstream named-query counts differ from actual frozen inputs')
        check(identity['input_raw_identity_sha256'] == meta['input_raw_identity_sha256'] == object_sha(raw), 'Raw identity digest differs')
        check(identity['allowed_train_ids_sha256'] == object_sha(sorted(allowed)), 'Allowed-ID digest differs')
        before_raw = {k: v for k, v in identity.items() if k not in ('raw_sources', 'input_raw_identity_sha256', 'allowed_train_ids_sha256')}
        # The actually executed, byte-restored source A seals configuration
        # before appending raw identities. Raw identities are authenticated
        # separately above; never pretend source A emitted source B's fields.
        check('configuration_fingerprint' not in meta and meta['fingerprint'] == object_sha(before_raw),
              'Proposer fingerprint does not match the archived running source A')
        best = selected_row(meta['history'], expected_config['epochs'], len(val))
        check(meta['selected'] == dict(epoch=best['epoch'], validation=best['validation']), 'Wrong upstream epoch selection')
        for row in meta['history']:
            steps = expected_config['steps_per_epoch'] if row['epoch'] else 0
            check(row['optimizer_steps'] == steps and row['consumed_queries'] == steps * expected_config['batch_size'], 'Upstream budget differs')
        check(meta['optimizer_steps_total'] == expected_config['epochs'] * expected_config['steps_per_epoch'] and
              meta['consumed_queries_total'] == meta['optimizer_steps_total'] * expected_config['batch_size'], 'Upstream totals differ')
        payload = torch.load(path.with_suffix('.pt'), map_location='cpu', weights_only=True)
        check(payload['initial_state_sha256'] == expected_init and payload['metadata']['identity'] == identity and
              payload['metadata']['selected'] == meta['selected'] and payload['metadata']['fingerprint'] == meta['fingerprint'],
              'Checkpoint/sidecar proposer identity differs')
        check(torch.equal(payload['state_dict']['head_prior'], torch.zeros(96)), 'Learned proposer head prior')
        check(all(torch.isfinite(v).all().item() for v in payload['state_dict'].values()), 'Nonfinite proposer checkpoint')
        cpu = meta['selected_cpu_canonical_validation']; check(cpu['selection_unchanged'] is True and
              cpu['selected_epoch'] == best['epoch'] and cpu['integer_decode_device'] == 'cpu', 'Upstream epoch silently reselected on CPU')
        check(cpu['delta_hits'] == cpu['metrics']['hits'] - best['validation']['hits'] and
              cpu['delta_point_oracle'] == cpu['metrics']['point_oracle'] - best['validation']['point_oracle'], 'Incorrect CPU validation delta')
        cpu_validation[role] = cpu; models[role] = meta
    return models, dict(initial_state_sha256=expected_init, roles=len(models), cpu_selected_validation=cpu_validation,
                        archived_training_source_sha256=archive['restored_training_source_sha256'],
                        source_race_resolution_sha256=ledger.need(race_path),
                        fingerprint_scope='running source A configuration; appended full raw identities independently SHA-verified')


def _expected_transform(size, candidate, grid):
    """Independent geometry-only replay; does not open or resize screenshot pixels."""
    gh, gw = grid; parent = int(candidate['parent_index']); row, col = divmod(parent, gw)
    axes = []
    for full, center, span in zip(size, ((col+.5)*size[0]/gw, (row+.5)*size[1]/gh),
                                  (max(256, 4*size[0]/gw), max(256, 4*size[1]/gh))):
        width = min(full, max(1, int(round(span)))); start = min(full-width, max(0, int(round(center-width/2))))
        axes.append((start, start + width))
    a, b = axes; return dict(box=[a[0], b[0], a[1], b[1]], scale=2, original_size=list(size),
        crop_size=[2*(a[1]-a[0]), 2*(b[1]-b[0])], parent_index=parent, grid_shape=list(grid), ground_truth_used=False)


def _validate_raw(path, case, ledger, transform=None):
    ledger.need(path)
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z['metadata']))
        check(meta['sample_id'] == case['sample_id'] and meta['image_sha256'] == case['image_sha256'], 'Raw case/image differs')
        expected_size = transform['crop_size'] if transform else case['image_size']
        check(meta['image_size'] == expected_size and meta['own_token_forward'] and
              meta['all_backbone_parameters_frozen'] and meta['ground_truth_in_model_input'] is False, 'Unsafe/raw contract mismatch')
        if 'instruction_sha256' in meta:
            check(meta['instruction_sha256'] == hashlib.sha256(case['instruction'].encode()).hexdigest(), 'Raw instruction differs')
        else:
            source = Path(path).parent.parent / 'manifest.jsonl'; stat = source.stat()
            original = _instruction_reference(str(source.resolve()), stat.st_size, stat.st_mtime_ns)
            check(original[case['sample_id']] == case['instruction'], 'Legacy raw instruction differs')
            ledger.need(source)
        if transform: check(meta['transform'] == transform, 'Raw crop transform differs')
        check(not any(k.startswith('label') or k in ('target_geometry', 'frozen_points', 'frozen_point_features') for k in z.files),
              'Raw feature archive contains teacher/GT arrays')
        for name in ('target_attention', 'target_hidden', 'image_hidden', 'vision_hidden'):
            check(name in z.files, 'Missing frozen Qwen raw array')
    return meta


def _legacy_reuse(root, pipeline, case, runtime, ledger):
    proof = pipeline['legacy_raw_reuse_identity']; old_path = Path(proof['old_pipeline_path'])
    ledger.need(old_path, proof['old_pipeline_sha256']); old = read(old_path)
    seal_path = Path('runs/candidate_scorer_v11/frozen.json'); ledger.need(seal_path, proof['legacy_frozen_sha256'])
    seal = read(seal_path); evidence = seal['training_input_evidence']['files']
    check(old['sample_id'] == case['sample_id'] and old['old_crop_transform'] == pipeline['old_crop_transform'], 'Legacy crop was not exactly identical')
    check(Path(proof['raw_path']).resolve() == Path(old['local_feature_path']).resolve() == Path(pipeline['local_feature_path']).resolve(),
          'Legacy reuse points at a different raw archive')
    for path, expected in [(old_path, proof['old_pipeline_sha256']), (Path(proof['raw_path']), proof['raw_sha256'])]:
        name = os.path.normcase(os.path.abspath(path)); check(evidence[name] == expected, 'Legacy file was not authenticated by V11 seal')
        ledger.need(path, expected)
    check(proof['raw_sha256'] == old['local_feature_sha256'] == pipeline['local_feature_sha256'], 'Legacy raw SHA differs')
    check(proof['runtime_identity_sha256'] == ledger.need(root / 'runtime_identity.json') and
          proof['runtime_contract_sha256'] == runtime['runtime_contract_sha256'] and
          proof['legacy_frozen_sha256'] == runtime['legacy_frozen_sha256'], 'Legacy runtime proof differs')


def audit_cache_group(root, group, cases, proposer_meta, runtime, ledger, replay_limit=2):
    from clickagent.candidate_budget import build_pools
    from clickagent.internal_candidate_scorer import INPUT_KEYS
    from clickagent.public_evaluation import point_in_geometry
    from clickagent.crossfit_proposer import load_crossfit_proposer, predict_raw
    from cache_candidate_scorer import raw_arrays
    index_path = root / 'cache' / (group + '_index.json'); index = read(index_path); ledger.need(index_path)
    check(index['complete'] and [v['sample_id'] for v in index['cases']] == [c['sample_id'] for c in cases], 'Cache case identity/order differs')
    folder = root / ('test' if group == 'test' else 'trainval') / group
    models = {}; replayed = Counter(); examples = []; reused = 0; label_count = 0; available = []
    for case_number, (c, row) in enumerate(zip(cases, index['cases']), 1):
        sid = c['sample_id']; pp = folder / 'pipeline' / (key(sid) + '.json'); pipeline = read(pp); ledger.need(pp)
        expected_role = 'full' if group in ('val', 'test') else f"fold_{c['image_fold'] if group == 'oof' else (c['image_fold']+1)%5}"
        check(pipeline['sample_id'] == sid and pipeline['proposer_role'] == expected_role and
              pipeline['proposer_sha256'] == proposer_meta[expected_role]['checkpoint_sha256'], 'Wrong assigned proposer in whole chain')
        check(pipeline['ground_truth_in_model_inputs'] is False and pipeline['pools'] == build_pools(pipeline), 'Pool was modified or used GT')
        for p, expected in row['evidence'].items(): ledger.need(p, expected)
        pp_evidence = [v for p, v in row['evidence'].items() if Path(p).resolve() == pp.resolve()]
        check(pp_evidence == [ledger.need(pp)], 'Cache evidence omits pipeline')
        gp = pipeline['global_result']; check(gp['proposer_role'] == expected_role and gp['proposer_sha256'] == pipeline['proposer_sha256'], 'Global proposer differs')
        global_path = folder / 'global' / (key(sid) + '.json'); ledger.need(global_path)
        check(read(global_path) == gp and gp['ground_truth_in_readout_inputs'] is False and
              Path(gp['raw_feature_path']).resolve() == Path(pipeline['global_feature_path']).resolve() and
              gp['raw_feature_sha256'] == pipeline['global_feature_sha256'], 'Global learned output/raw identity differs')
        ledger.need(pipeline['global_feature_path'], pipeline['global_feature_sha256'])
        gm = _validate_raw(pipeline['global_feature_path'], c, ledger)
        transform = pipeline['old_crop_transform']
        if gp['candidates']:
            check(transform == _expected_transform(c['image_size'], gp['candidates'][0], gp['grid_shape']), 'Crop not chosen by assigned new global output')
            ledger.need(pipeline['local_feature_path'], pipeline['local_feature_sha256'])
            lm = _validate_raw(pipeline['local_feature_path'], c, ledger, transform)
            if pipeline.get('identical_legacy_raw_reused'):
                check(group != 'test', 'Fresh test reused a historical local raw feature'); _legacy_reuse(root, pipeline, c, runtime, ledger); reused += 1
            else:
                check('legacy_raw_reuse_identity' not in pipeline, 'Unexpected legacy reuse proof')
                check(Path(pipeline['local_feature_path']).resolve().is_relative_to(root.resolve()), 'Uncertified external local raw archive')
            check(pipeline['old_crop_result']['ground_truth_in_readout_inputs'] is False, 'GT in local proposer')
        else:
            check(transform is None and pipeline['old_crop_result'] is None and pipeline['local_feature_path'] is None, 'No-name case has learned local output')
        path = Path(row['path']); ledger.need(path, row['sha256'])
        with np.load(path, allow_pickle=False) as z:
            points = z['points']; check(str(z['sample_id']) == sid and bool(z['available']) == bool(len(points)) == row['available'], 'Cache availability/case differs')
            check(z['annotation_derived_coordinates'].item() is False and str(z['pipeline_sha256']) == ledger.need(pp), 'GT-derived/cached coordinate identity mismatch')
            check(points.tolist() == [v['point'] for v in pipeline['pools']['32']], 'Cache coordinates differ from real pool')
            for k in BUDGETS:
                ix = z['pool_indices_' + str(k)]
                check(np.issubdtype(ix.dtype, np.integer) and len(set(ix.tolist())) == len(ix) and
                      ((ix >= 0) & (ix < len(points))).all(), 'Invalid cache subset indices')
                check(points[ix].tolist() == [v['point'] for v in pipeline['pools'][str(k)]], 'K subset differs from independent real pool')
            inputs = {k.removeprefix('i__') for k in z.files if k.startswith('i__')}
            check(inputs == (set(INPUT_KEYS) if len(points) else set()), 'Scorer input key contract changed or includes GT')
            if len(points):
                check(np.array_equal(z['i__points'], (points / np.asarray(c['image_size'])).astype(np.float32)), 'Input normalized points changed')
            check(('labels' in z.files) == (group != 'test'), 'Test labels leaked or training labels missing')
            if group != 'test':
                labels = np.asarray([point_in_geometry((point / np.asarray(c['image_size'])).tolist(), c['target_geometry']) for point in points], np.float32)
                check(np.array_equal(z['labels'], labels), 'Candidate labels do not describe actual proposed points'); label_count += len(labels)
            check(not any(k.startswith('label') and k != 'labels' or k in ('target_geometry', 'frozen_points', 'frozen_point_features') for k in z.files),
                  'Unexpected supervision/teacher cache fields')
        available.append(row['available'])
        if replayed[expected_role] < replay_limit:
            if expected_role not in models: models[expected_role] = load_crossfit_proposer(root / 'proposers' / (expected_role + '.pt'), 'cpu')[0]
            for kind, raw, size, expected in [('global', pipeline['global_feature_path'], c['image_size'], gp)] + (
                [('local', pipeline['local_feature_path'], transform['crop_size'], pipeline['old_crop_result'])] if transform else []):
                arrays, meta = raw_arrays(raw); actual = predict_raw(models[expected_role], arrays, meta, size, topk=64)
                for field in ('point', 'all_points', 'parent_indices', 'candidates', 'target_name', 'grid_shape', 'decoder'):
                    check(actual[field] == expected[field], f'CPU new-proposer {kind} replay differs: {sid}/{field}')
                examples.append(dict(sample_id=sid, proposer_role=expected_role, view=kind, exact_points=True))
            # Spot-check that stored scorer features are the actual raw-state
            # transformation, rather than only trusting names and NPZ hashes.
            if row['available']:
                from clickagent.internal_candidate_scorer import build_internal_inputs
                ga, global_meta = raw_arrays(pipeline['global_feature_path'])
                la, local_meta = raw_arrays(pipeline['local_feature_path']) if transform else (None, None)
                with np.load(path, allow_pickle=False) as z:
                    rebuilt = build_internal_inputs(ga, global_meta, la, local_meta, z['points'], c['image_size'], transform)
                    for field, expected in rebuilt.items():
                        check(np.array_equal(z['i__' + field], expected), f'Raw-to-scorer feature rebuild differs: {sid}/{field}')
                examples.append(dict(sample_id=sid, proposer_role=expected_role, view='scorer_cache', exact_raw_rebuild=True))
            replayed[expected_role] += 1
        if case_number % 100 == 0 or case_number == len(cases):
            print(f'AUDIT {group} cases {case_number}/{len(cases)}', flush=True)
    return dict(cases=len(cases), available=sum(available), availability=available, point_labels_checked=label_count,
                identical_legacy_raw_reused=reused, cpu_replay=examples)


def audit_scorers(root, protocol, cohorts, cache_evidence, ledger):
    import torch
    from clickagent.internal_candidate_scorer import InternalCandidateScorer, InternalCandidateScorerConfig
    from train_crossfit_scorer import state_identity, epoch_plan
    from crossfit_protocol import source_files
    ids = [c['sample_id'] for c in cohorts['train']]; eligible = cache_evidence['in_sample']['availability']
    check(eligible == cache_evidence['oof']['availability'], 'Training query eligibility differs between arms')
    outputs = {}; count = sum(eligible); epochs = protocol['optimization']['epochs']; accumulation = protocol['optimization']['accumulation']
    for suffix, seed in zip(('s1', 's2'), protocol['optimization']['seeds']):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed); initial = InternalCandidateScorer(InternalCandidateScorerConfig(head_mode='full'))
        expected_init = state_identity(initial); pair = []
        for arm in ARMS:
            path = root / 'models' / f'{arm}_{suffix}.json'; meta = read(path); ledger.need(path)
            check(meta['complete'] and meta['arm'] == arm and meta['seed'] == seed, 'Incomplete/wrong scorer identity')
            check(meta['initial_state_sha256'] == expected_init and meta['parameters'] == sum(p.numel() for p in initial.parameters()),
                  'Scorer initialization/architecture differs')
            check(meta['protocol_sha256'] == ledger.need(root / 'protocol.json'), 'Scorer protocol mismatch')
            check(set(meta['source_identity']) == {str(p) for p in source_files()}, 'Incomplete scorer source identity')
            for p, expected in meta['source_identity'].items(): ledger.need(p, expected)
            check(meta['input_identity'] == {g: ledger.need(root / 'cache' / (g + '_index.json')) for g in (arm, 'val')}, 'Scorer train/common-D val identity differs')
            best = selected_row(meta['history'], epochs, len(cohorts['val']))
            check(meta['selected'] == best, 'Scorer epoch was not selected by common validation')
            validate_validation_replay(best['validation'], meta['selected_validation_replay'])
            for row in meta['history']:
                schedule = [[ids[i], budget] for i, budget in epoch_plan(ids, seed, row['epoch']) if eligible[i]] if row['epoch'] else []
                check(row['schedule_sha256'] == digest(schedule), 'Scorer query/budget schedule differs')
                check(row['queries'] == (count if row['epoch'] else 0) and row['optimizer_steps'] == (math.ceil(count/accumulation) if row['epoch'] else 0),
                      'Scorer optimizer/query count differs')
            ledger.need(path.with_suffix('.pt'), meta['checkpoint_sha256'])
            saved = torch.load(path.with_suffix('.pt'), map_location='cpu', weights_only=True)
            check(saved['config'] == asdict(initial.config) and all(torch.isfinite(v).all().item() for v in saved['state_dict'].values()),
                  'Scorer configuration/nonfinite checkpoint differs')
            for field in ('arm', 'seed', 'selected', 'initial_state_sha256', 'protocol_sha256', 'source_identity', 'input_identity'):
                check(saved['metadata'][field] == meta[field], f'Scorer checkpoint metadata differs: {field}')
            check(saved['metadata']['architecture_unchanged'] and saved['metadata']['loss_unchanged'] and
                  saved['metadata']['annotation_derived_points'] is False and saved['metadata']['training_uses_gui_actor'] is False,
                  'Scorer structure/loss/data treatment changed')
            check(saved['metadata']['train_n'] == len(ids) and saved['metadata']['validation_n'] == len(cohorts['val']),
                  'Wrong scorer train/validation cohort size')
            pair.append(meta); outputs[f'{arm}_{suffix}'] = meta
        check(pair[0]['initial_state_sha256'] == pair[1]['initial_state_sha256'], 'Paired scorer initial states differ')
        check([(r['schedule_sha256'], r['optimizer_steps'], r['queries']) for r in pair[0]['history']] ==
              [(r['schedule_sha256'], r['optimizer_steps'], r['queries']) for r in pair[1]['history']], 'Paired scorer schedules differ')
    return outputs


def audit_runtime(root, ledger):
    evidence = read(root / 'runtime_identity.json'); ledger.need(root / 'runtime_identity.json')
    check(evidence['complete'] and evidence['no_model_execution'] and evidence['no_predictions_read'], 'Incomplete runtime evidence')
    seal = read(evidence['legacy_frozen_path']); ledger.need(evidence['legacy_frozen_path'], evidence['legacy_frozen_sha256'])
    def old_expected(path):
        canonical = os.path.normcase(os.path.abspath(path)); values = []
        for inventory in (seal['files'], seal.get('external_input_evidence', {}).get('files', {})):
            values += [v for p, v in inventory.items() if os.path.normcase(os.path.abspath(p)) == canonical]
        check(values and len(set(values)) == 1, f'Runtime input not authenticated by V11: {path}')
        return values[0]
    for prefix in ('qwen', 'gui'):
        path = evidence[prefix + '_identity_path']; expected = evidence[prefix + '_identity_sha256']
        check(expected == old_expected(path), 'Model identity is not the original frozen identity'); ledger.need(path, expected)
    # Both runtime getters read this shared file; comparing getters alone would
    # otherwise allow the two sides to drift together.
    ledger.need('runs/candidate_v2/identity.json', old_expected('runs/candidate_v2/identity.json'))
    for _, entry in evidence['core_source_identity'].items():
        check(entry['matches'] and entry['sha256'] == entry['expected_sha256'] == old_expected(entry['path']), 'Core runtime code changed')
        ledger.need(entry['path'], entry['sha256'])
    for entry in evidence['actual_model_files']:
        check(entry['matches'] and entry['full_bytes_read'] and entry['sha256'] == entry['expected_sha256'] and
              entry['bytes'] == entry['expected_bytes'] == Path(entry['path']).stat().st_size, 'Model byte identity differs')
        ledger.need(entry['path'], entry['sha256'])
    check(evidence['actual_weight_files'] == [r for r in evidence['actual_model_files'] if r['kind'] == 'weight'], 'Weight evidence subset differs')
    check(evidence['full_model_bytes_read'] == sum(r['bytes'] for r in evidence['actual_model_files']), 'Model byte total differs')
    qwen = read(evidence['qwen_identity_path']); gui = read(evidence['gui_identity_path'])
    inventories = {'Qwen3-VL-8B-Instruct': (Path(qwen['model_path']), qwen['files']),
                   'GUI-Actor-Verifier-2B': (Path(evidence['gui_identity_path']).parent, {r['file']: r for r in gui['files']})}
    for model_name, (folder, inventory) in inventories.items():
        actual = {r['relative_path']: r for r in evidence['actual_model_files'] if r['model'] == model_name}
        check(set(actual) == set(inventory), 'Missing/uninventoried model assets')
        for relative, expected in inventory.items():
            entry = actual[relative]
            check(Path(entry['path']).resolve() == (folder / relative).resolve() and entry['sha256'] == expected['sha256'] and
                  entry['bytes'] == expected.get('bytes', expected.get('size')), 'Model inventory differs')
        present = {str(p.relative_to(folder)).replace('\\', '/') for p in folder.rglob('*') if p.is_file() and p.suffix in ('.safetensors', '.bin')}
        check(present == {p for p in inventory if p.endswith(('.safetensors', '.bin'))}, 'Uninventoried model weights')
    from extract_candidate_scorer import runtime_config as old_config
    from extract_crossfit_candidates import runtime_config as new_config
    a, b = asdict(old_config()), asdict(new_config()); a.pop('offload_folder'); b.pop('offload_folder')
    check(a == b, 'Frozen-Qwen config differs beyond storage path')
    normalized = json.loads(json.dumps(b))
    check(normalized == evidence['runtime_contract'] and digest(normalized) == evidence['runtime_contract_sha256'] and
          evidence['runtime_config_exactly_equal_except'] == ['offload_folder'], 'Runtime contract digest/config differs')
    for p, expected in evidence['extractor_source_sha256'].items(): ledger.need(p, expected)
    ledger.need(evidence['verifier_source_path'], evidence['verifier_source_sha256'])
    import importlib.metadata
    versions = {n: importlib.metadata.version(n) for n in ('torch', 'transformers', 'accelerate', 'numpy', 'Pillow')}
    check(evidence['library_versions_equal'] is True and evidence['library_versions'] == evidence['legacy_library_versions'] == versions,
          'Extraction libraries differ from authenticated runtime reference')
    reference = evidence['legacy_library_versions_reference']
    ledger.need(reference['path'], reference['sha256'])
    check(read(reference['path'])['library_versions'] == versions, 'Library reference content differs')
    check(reference['listed_in_legacy_seal'] is False,
          'Unpinned historical library reference was incorrectly described as sealed')
    config_reference = evidence['runtime_config_input_identity']
    ledger.need(config_reference['path'], config_reference['sha256'])
    check(config_reference['sha256'] == old_expected(config_reference['path']), 'Shared runtime config is not V11-authenticated')
    return evidence


def audit_final_predictions(root, cohorts, ledger):
    import torch
    from clickagent.candidate_budget import coordinate_key
    from clickagent.internal_candidate_scorer import load_internal_scorer
    from train_candidate_scorer import load_cache, subset
    from evaluate_crossfit_candidates import evaluate_records, method_key
    from clickagent.spatial_geometry import shape_from_geometry
    from shapely.geometry import Point
    from clickagent.public_evaluation import point_in_geometry
    evaluation = root / 'evaluation/test'; cohort = cohorts['test']; merged = []; pool_rows = []; saved_by_id = {}
    names = [a + '_' + s for a in ARMS for s in ('s1', 's2')]
    for case in cohort:
        sid = case['sample_id']; stem = key(sid)
        path = root / 'predictions/test' / (stem + '.json'); ledger.need(path); head = read(path)
        check(head['sample_id'] == sid and head['ground_truth_in_model_inputs'] is False, 'GT in head prediction')
        pp = root / 'test/test/pipeline' / (stem + '.json'); pipeline = read(pp); ledger.need(pp)
        head_map = {(p['method'], p['budget']): p for p in head['predictions']}
        check(len(head_map) == len(head['predictions']) == 18, 'Missing/repeated frozen head/native prediction')
        for name in names:
            for budget in BUDGETS:
                row = head_map[(name, budget)]; pool = pipeline['pools'][str(budget)]; scores = np.asarray(row['scores'])
                check(scores.shape == (len(pool),) and np.isfinite(scores).all(), 'Invalid head scores')
                expected = pool[int(scores.argmax())]['point'] if pool else None
                check(row['sample_id'] == sid and row['point'] == expected, 'Head changed a point or wrong argmax')
                if pool:
                    check(row['selected_index'] == int(scores.argmax()) and math.isfinite(row['null_score']) and
                          row['null_would_reject'] == bool(row['null_score'] > scores.max()), 'Head selected index/null differs')
        native_path = root / 'test/test/native' / (stem + '.json'); ledger.need(native_path); native = read(native_path)
        check(native['ground_truth_in_model_inputs'] is False and head_map[('qwen_native_completed_lax', None)]['point'] == native['completed_lax_point'], 'Native coordinate changed/used GT')
        check(head_map[('global_top1', None)]['point'] == pipeline['global_result']['point'], 'Global Top1 changed')
        gp = root / 'gui_reference/test' / (stem + '.json'); ledger.need(gp); gui = read(gp)
        check(gui['sample_id'] == sid and gui['ground_truth_in_model_inputs'] is False, 'GT in GUI reference')
        records = gui['records']; common = pipeline['pools']['16']
        check(set(records) == {coordinate_key(v['point']) for v in common}, 'GUI scores do not cover exactly common K16')
        def gui_point(pool):
            values = [records[coordinate_key(v['point'])]['log_odds'] for v in pool]
            check(np.isfinite(values).all(), 'Nonfinite GUI scores')
            return pool[int(np.argmax(values))]['point'] if pool else None
        gui_map = {(p['method'], p['budget']): p for p in gui['predictions']}
        check(len(gui_map) == len(gui['predictions']) == 6, 'Missing/repeated GUI/shortlist prediction')
        for budget in (4, 16):
            check(gui_map[('gui_actor', budget)]['point'] == gui_point(pipeline['pools'][str(budget)]), 'Wrong same-pool GUI argmax')
        for name in names:
            p = gui_map[(name + '_top4_gui', 16)]
            scores = head_map[(name, 16)]['scores']; order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:4]
            retained = [common[i] for i in order]
            check(p['retained_points'] == [v['point'] for v in retained] and p['n_verified'] == len(retained) and p['point'] == gui_point(retained),
                  'Secondary shortlist/tie order/GUI choice differs')
        merged.extend(head['predictions']); merged.extend(gui['predictions'])
        pool_rows.append(dict(sample_id=sid, pools=pipeline['pools'])); saved_by_id[sid] = head_map
    predictions_path, pools_path = evaluation / 'predictions.jsonl', evaluation / 'pools.jsonl'
    ledger.need(predictions_path); ledger.need(pools_path)
    check(rows(predictions_path) == merged and rows(pools_path) == pool_rows, 'Flat export differs from original frozen predictions/pools')
    result = evaluate_records(cohort, merged, pool_rows)
    expected_names = {method_key(name, k) for name in names for k in BUDGETS} | {
        method_key('gui_actor', 4), method_key('gui_actor', 16), 'global_top1', 'qwen_native_completed_lax'} | {
        method_key(name + '_top4_gui', 16) for name in names}
    check(set(result['summary']['methods']) == expected_names, 'Incomplete final method cohort')
    summary_path = evaluation / 'summary.json'; details_path = evaluation / 'details.json'; scored_path = evaluation / 'scored_predictions.jsonl'
    summary = read(summary_path)
    for field, value in result['summary'].items(): check(summary[field] == value, 'Recomputed summary differs: ' + field)
    for p, expected in summary['input_sha256'].items(): ledger.need(p, expected)
    ledger.need('scripts/evaluate_crossfit_candidates.py', summary['evaluator_sha256'])
    check(read(details_path) == result['details'] and rows(scored_path) == result['scored_predictions'], 'Recomputed per-case scores differ')
    for path in (summary_path, details_path, scored_path, evaluation / 'report.html'): ledger.need(path)
    differences = []
    for case, detail in zip(cohort, result['details']):
        shape = shape_from_geometry(case['target_geometry']); size = np.asarray(case['image_size'])
        points = {tuple(p) for pool in detail['pools'].values() for p in pool['points']}
        points.update(tuple(p['point']) for p in detail['methods'].values() if p['point'] is not None)
        for point in sorted(points):
            normalized = np.asarray(point) / size
            canonical = bool(point_in_geometry(normalized.tolist(), case['target_geometry']))
            independent = bool(shape.covers(Point(*normalized)))
            if canonical != independent:
                differences.append(dict(sample_id=case['sample_id'], point=list(point), point_in_geometry=canonical,
                                        shapely_covers=independent, numerical_boundary_tolerance=1e-12))
    # Exact selected points, with logit arithmetic differences explicitly recorded.
    replay = []; models = {n: load_internal_scorer(root / 'models' / (n + '.pt'), 'cpu')[0].eval() for n in names}
    chosen = []
    for source in sorted({c['source'] for c in cohort}):
        candidates = sorted([c for c in cohort if c['source'] == source and pool_rows[cohort.index(c)]['pools']['16']], key=lambda c: c['sample_id'])
        if candidates: chosen.append(candidates[0])
    index = {r['sample_id']: r for r in read(root / 'cache/test_index.json')['cases']}
    for case in chosen:
        record = load_cache(index[case['sample_id']]['path']); check(record['labels'] is None, 'Test labels reached CPU replay')
        ix = record['pool_indices']['16']
        for name, model in models.items():
            with torch.inference_mode(): actual = model(**subset(record, ix, False, 'cpu'))['scores'].detach().numpy()
            saved = saved_by_id[case['sample_id']][(name, 16)]; expected = np.asarray(saved['scores'])
            check(np.allclose(actual, expected, rtol=1e-6, atol=1e-6), 'CPU scorer logit replay differs materially')
            check(record['points'][ix[int(actual.argmax())]].tolist() == saved['point'], 'CPU scorer selected point differs')
            replay.append(dict(sample_id=case['sample_id'], method=name, budget=16, selected_point_exact=True,
                               logits_bitwise_equal=bool(np.array_equal(actual, expected)), max_abs_logit_difference=float(np.max(np.abs(actual-expected)))))
    return dict(cases=len(cohort), methods=len(expected_names), scored_predictions=len(merged),
                summary_sha256=ledger.need(summary_path), independent_shapely_differences=differences, cpu_scorer_replay=replay)


def run_audit(root, stage):
    root = Path(root); ledger = Ledger(); print('AUDIT data/folds', flush=True)
    if stage == 'pretest':
        check(not (root / 'frozen.json').exists(), 'Cannot rewrite a sealed pretest proof')
        for folder in (root / 'test', root / 'predictions/test', root / 'gui_reference/test'):
            check(not any(folder.rglob('*.*')), 'Fresh test artifact existed before pretest audit')
    protocol, folds, cohorts, data = audit_data(root, ledger)
    if stage == 'final':
        seal = read(root / 'frozen.json'); check(seal['complete'] and seal['fresh_test_features_at_freeze'] == 0, 'Missing/incomplete scorer freeze')
        for p, expected in seal['files'].items(): ledger.need(p, expected)
        pretest_path = root / 'pretest_audit.json'; pretest = read(pretest_path)
        check(pretest['complete'] and pretest['stage'] == 'pretest' and pretest['auditor_sha256'] == ledger.need(__file__), 'Missing/changed pretest causal proof')
        sealed_pretest = [v for p, v in seal['files'].items() if Path(p).resolve() == pretest_path.resolve()]
        check(sealed_pretest == [ledger.need(pretest_path)] and pretest['created_at'] <= seal['created_at'], 'Pretest was not sealed before testing')
        for p, expected in pretest['input_sha256'].items(): ledger.need(p, expected)
    print('AUDIT runtime identities', flush=True); runtime = audit_runtime(root, ledger)
    print('AUDIT fresh proposers', flush=True); proposers, upstream = audit_proposers(root, protocol, folds, cohorts, ledger)
    caches = {}
    for group in ('in_sample', 'oof', 'val'):
        print('AUDIT cache/whole chain ' + group, flush=True)
        caches[group] = audit_cache_group(root, group, cohorts['train' if group in ARMS else 'val'], proposers, runtime, ledger)
    print('AUDIT paired scorers', flush=True)
    scorers = audit_scorers(root, protocol, cohorts, caches, ledger)
    final = None
    if stage == 'final':
        print('AUDIT fresh test cache/whole chain', flush=True)
        caches['test'] = audit_cache_group(root, 'test', cohorts['test'], proposers, runtime, ledger)
        for p in (root / 'test').rglob('*.npz'):
            check(p.stat().st_mtime >= seal['created_at'], 'Fresh features predate scorer freeze')
        print('AUDIT complete final predictions/statistics', flush=True)
        final = audit_final_predictions(root, cohorts, ledger)
    ledger.unchanged()
    for value in caches.values(): value.pop('availability')
    return dict(complete=True, version=12, stage=stage, created_at=time.time(), auditor_sha256=ledger.need(__file__),
                protocol_sha256=ledger.need(root / 'protocol.json'), data=data, upstream=upstream,
                cache_checks=caches, scorers=list(scorers),
                scorer_validation_replay={name: validate_validation_replay(meta['selected']['validation'], meta['selected_validation_replay'])
                                          for name, meta in scorers.items()}, input_sha256=ledger.files,
                fresh_test_features=0 if stage == 'pretest' else len(cohorts['test']),
                final=final, ground_truth_in_model_inputs=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('pretest', 'final'), required=True)
    parser.add_argument('--root', type=Path, default=OUT)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(4)
    result = run_audit(args.root, args.stage)
    path = args.root / ('pretest_audit.json' if args.stage == 'pretest' else 'final_audit.json')
    temporary = path.with_suffix('.tmp'); temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)
    print(json.dumps(dict(complete=True, stage=args.stage, path=str(path)), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
