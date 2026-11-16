import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import numpy as np


SPEC = importlib.util.spec_from_file_location('crossfit_audit_test_module', Path(__file__).parents[1] / 'scripts/audit_crossfit_candidates.py')
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def data():
    train = []
    for i in range(5):
        rgb = hashlib.sha256(f'image{i}'.encode()).hexdigest()
        for j in range(2 if i == 0 else 1):
            train.append(dict(sample_id=f'q{i}_{j}', pixel_sha256=rgb, dhash256=rgb, image_fold=i,
                              oof_provider=f'fold_{i}', in_sample_provider=f'fold_{(i+1)%5}'))
    v = hashlib.sha256(b'validation').hexdigest()
    val = [dict(sample_id='val', pixel_sha256=v, dhash256=v)]
    roles = {}
    for role in audit.ROLES:
        allowed = [c for c in train if role == 'full' or c['image_fold'] != int(role[-1])]
        heldout = [c for c in train if c not in allowed]
        a, h = sorted(c['sample_id'] for c in allowed), sorted(c['sample_id'] for c in heldout)
        roles[role] = dict(allowed_train_ids=a, heldout_ids=h,
            allowed_train_ids_sha256=audit.digest(a), heldout_ids_sha256=audit.digest(h),
            allowed_train_groups=sorted({c['pixel_sha256'] for c in allowed}), heldout_groups=sorted({c['pixel_sha256'] for c in heldout}))
    folds = dict(n_folds=5, sample_to_fold={c['sample_id']: c['image_fold'] for c in train}, roles=roles)
    return train, val, folds


def test_group_allowlist_and_seen_mapping():
    train, val, folds = data()
    assert audit.validate_folds(train, val, folds)['train_rgb_groups'] == 5
    train[0]['in_sample_provider'] = 'fold_0'
    with pytest.raises(ValueError, match='mapping'): audit.validate_folds(train, val, folds)


def test_same_image_and_near_image_cannot_cross_folds():
    train, val, folds = data()
    train[1]['image_fold'] = 1; folds['sample_to_fold'][train[1]['sample_id']] = 1
    train[1]['oof_provider'] = 'fold_1'; train[1]['in_sample_provider'] = 'fold_2'
    with pytest.raises(ValueError, match='Identical screenshot'): audit.validate_folds(train, val, folds)
    train, val, folds = data()
    train[-1]['dhash256'] = train[0]['dhash256']
    with pytest.raises(ValueError, match='Near screenshots'): audit.validate_folds(train, val, folds)


def test_heldout_exposure_rejected_even_with_forged_digest():
    train, val, folds = data()
    folds['roles']['fold_0']['allowed_train_ids'].append(train[0]['sample_id'])
    folds['roles']['fold_0']['allowed_train_ids_sha256'] = audit.digest(sorted(folds['roles']['fold_0']['allowed_train_ids']))
    with pytest.raises(ValueError, match='allowlist'): audit.validate_folds(train, val, folds)


def test_epoch_selection_keeps_earlier_exact_tie():
    history = [dict(epoch=i, validation=dict(n=4, named=4, hits=hits, oracle=4, loss=loss))
               for i, (hits, loss) in enumerate([(1, 1.), (2, .9), (2, .9)])]
    assert audit.selected_row(history, 2, 4)['epoch'] == 1
    with pytest.raises(ValueError, match='Incomplete'): audit.selected_row(history[:2], 2, 4)
    bad = copy.deepcopy(history); bad[-1]['validation']['hits'] = 5
    with pytest.raises(ValueError, match='Impossible'): audit.selected_row(bad, 2, 4)


def test_streamed_identity_detects_replacement(tmp_path):
    path = tmp_path / 'evidence'; path.write_bytes(b'original'); ledger = audit.Ledger()
    sha = ledger.need(path); path.write_bytes(b'changed with another length')
    with pytest.raises(ValueError, match='SHA mismatch'): ledger.need(path, sha)


def test_replay_loss_roundoff_never_relaxes_point_counts():
    value = dict(n=4, named=4, hits=2, oracle=3, refused=1, loss=1.)
    actual = dict(value, loss=1.000001)
    assert audit.validate_validation_replay(value, actual)['counts_exact']
    with pytest.raises(ValueError, match='count differs'): audit.validate_validation_replay(value, dict(actual, hits=3))
    with pytest.raises(ValueError, match='loss differs'): audit.validate_validation_replay(value, dict(actual, loss=1.1))


@pytest.mark.parametrize('change', [None, 'transform', 'runtime_contract'])
def test_legacy_raw_reuse_requires_exact_authenticated_identity(tmp_path, monkeypatch, change):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / 'new'; root.mkdir()
    old = tmp_path / 'runs/candidate_scorer_v11'; old.mkdir(parents=True)
    raw = old / 'raw.npz'; raw.write_bytes(b'opaque raw arrays authenticated elsewhere')
    raw_sha = hashlib.sha256(raw.read_bytes()).hexdigest()
    transform = dict(box=[0, 0, 10, 10], crop_size=[20, 20], scale=2)
    pp = old / 'pipeline.json'; pp.write_text(json.dumps(dict(sample_id='toy', old_crop_transform=transform,
                         local_feature_path=str(raw), local_feature_sha256=raw_sha)), encoding='utf-8')
    pp_sha = hashlib.sha256(pp.read_bytes()).hexdigest()
    inventory = {audit.os.path.normcase(str(p.resolve())): h for p, h in [(raw, raw_sha), (pp, pp_sha)]}
    seal = old / 'frozen.json'; seal.write_text(json.dumps(dict(training_input_evidence={'files': inventory})), encoding='utf-8')
    seal_sha = hashlib.sha256(seal.read_bytes()).hexdigest()
    runtime = dict(runtime_contract_sha256='exact-contract', legacy_frozen_sha256=seal_sha)
    runtime_path = root / 'runtime_identity.json'; runtime_path.write_text(json.dumps(runtime), encoding='utf-8')
    proof = dict(old_pipeline_path=str(pp), old_pipeline_sha256=pp_sha, raw_path=str(raw), raw_sha256=raw_sha,
                 legacy_frozen_sha256=seal_sha, runtime_contract_sha256='exact-contract',
                 runtime_identity_sha256=hashlib.sha256(runtime_path.read_bytes()).hexdigest())
    pipeline = dict(old_crop_transform=copy.deepcopy(transform), local_feature_path=str(raw),
                    local_feature_sha256=raw_sha, legacy_raw_reuse_identity=proof)
    if change == 'transform': pipeline['old_crop_transform']['box'] = [1, 0, 11, 10]
    elif change == 'runtime_contract': proof['runtime_contract_sha256'] = 'different-contract'
    if change:
        with pytest.raises(ValueError): audit._legacy_reuse(root, pipeline, {'sample_id': 'toy'}, runtime, audit.Ledger())
    else: audit._legacy_reuse(root, pipeline, {'sample_id': 'toy'}, runtime, audit.Ledger())


def cache_fixture(tmp_path, monkeypatch, group):
    scripts = Path(__file__).parents[1] / 'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    from clickagent.crossfit_proposer import make_fresh_proposer, save_crossfit_proposer, predict_raw
    from clickagent.candidate_budget import build_pools
    from clickagent.internal_candidate_scorer import build_internal_inputs
    from cache_candidate_scorer import base_cache, save_npz
    import torch
    torch.set_num_threads(2)
    root = tmp_path / 'experiment'; root.mkdir()
    case = dict(sample_id='toy', image_sha256='fake-image-hash', instruction='button',
                image_size=[100, 80], target_geometry={'type': 'bbox', 'boxes': [[0, 0, 1, 1]]})
    model = make_fresh_proposer().eval(); cp = root / 'proposers/full.pt'
    save_crossfit_proposer(cp, model, {})
    ledger = audit.Ledger(); model_sha = ledger.need(cp)
    rng = np.random.default_rng(5)
    arrays = dict(target_attention=rng.random((1, 96, 4), dtype=np.float32) / 100,
                  target_hidden=rng.normal(size=(1, 12288)).astype(np.float16),
                  image_hidden=rng.normal(size=(4, 4096)).astype(np.float16),
                  vision_hidden=rng.normal(size=(16, 1152)).astype(np.float16))
    meta = dict(sample_id='toy', image_sha256=case['image_sha256'], image_size=case['image_size'],
                grid_shape=[2, 2], target_available=True, target_text='button', own_token_forward=True,
                all_backbone_parameters_frozen=True, ground_truth_in_model_input=False,
                instruction_sha256=hashlib.sha256(case['instruction'].encode()).hexdigest())
    folder = root / ('test' if group == 'test' else 'trainval') / group
    gp = predict_raw(model, arrays, meta, case['image_size']); raw = root / 'global.npz'
    save_npz(raw, dict(arrays, metadata=np.asarray(json.dumps(meta))))
    gp.update(sample_id='toy', proposer_role='full', proposer_sha256=model_sha,
              raw_feature_path=str(raw), raw_feature_sha256=ledger.need(raw))
    global_json = folder / 'global' / (audit.key('toy') + '.json'); global_json.parent.mkdir(parents=True)
    global_json.write_text(json.dumps(gp), encoding='utf-8')
    transform = audit._expected_transform(case['image_size'], gp['candidates'][0], gp['grid_shape'])
    local_meta = dict(meta, image_size=transform['crop_size'], transform=transform)
    local = root / 'local.npz'; save_npz(local, dict(arrays, metadata=np.asarray(json.dumps(local_meta))))
    lp = predict_raw(model, arrays, local_meta, transform['crop_size'])
    pipeline = dict(sample_id='toy', proposer_role='full', proposer_sha256=model_sha, global_result=gp,
                    old_crop_result=lp, old_crop_transform=transform, local_feature_path=str(local),
                    local_feature_sha256=ledger.need(local), global_feature_path=str(raw),
                    global_feature_sha256=ledger.need(raw), ground_truth_in_model_inputs=False,
                    identical_legacy_raw_reused=False)
    pipeline['pools'] = build_pools(pipeline)
    pp = folder / 'pipeline' / (audit.key('toy') + '.json'); pp.parent.mkdir(parents=True)
    pp.write_text(json.dumps(pipeline), encoding='utf-8')
    cache = base_cache(case, pipeline, group)
    inputs = build_internal_inputs(arrays, meta, arrays, local_meta, cache['points'], case['image_size'], transform)
    cache.update({'i__' + k: v for k, v in inputs.items()}); cache['pipeline_sha256'] = np.asarray(ledger.need(pp))
    cached = root / 'cache' / group / (audit.key('toy') + '.npz'); save_npz(cached, cache)
    index = dict(complete=True, cases=[dict(sample_id='toy', path=str(cached), sha256=ledger.need(cached),
                 available=True, evidence={str(pp): ledger.need(pp), str(raw): ledger.need(raw), str(local): ledger.need(local)})])
    ip = root / 'cache' / (group + '_index.json'); ip.write_text(json.dumps(index), encoding='utf-8')
    return root, case, cache, cached, index, ip, {'full': {'checkpoint_sha256': model_sha}}


def test_actual_cpu_cache_chain_and_legal_geometry_input(tmp_path, monkeypatch):
    root, case, _, _, _, _, proposers = cache_fixture(tmp_path, monkeypatch, 'val')
    result = audit.audit_cache_group(root, 'val', [case], proposers, {}, audit.Ledger(), replay_limit=1)
    assert result['cases'] == 1 and result['point_labels_checked'] > 0
    assert len(result['cpu_replay']) == 3
    assert result['cpu_replay'][-1]['exact_raw_rebuild']


@pytest.mark.parametrize('corruption', ['label', 'coordinate', 'feature', 'test_label_leak'])
def test_consistently_rehashed_bad_cache_still_rejected(tmp_path, monkeypatch, corruption):
    from importlib import import_module
    group = 'test' if corruption == 'test_label_leak' else 'val'
    root, case, cache, cached, index, ip, proposers = cache_fixture(tmp_path, monkeypatch, group)
    if corruption == 'label': cache['labels'] = 1 - cache['labels']
    elif corruption == 'coordinate': cache['points'][0, 0] = (int(cache['points'][0, 0]) + 1) % 100
    elif corruption == 'feature': cache['i__query_hidden'] = cache['i__query_hidden'] + np.float16(.25)
    else: cache['labels'] = np.ones(len(cache['points']), np.float32)
    import_module('cache_candidate_scorer').save_npz(cached, cache)
    index['cases'][0]['sha256'] = audit.Ledger().need(cached); ip.write_text(json.dumps(index), encoding='utf-8')
    with pytest.raises(ValueError):
        audit.audit_cache_group(root, group, [case], proposers, {}, audit.Ledger(), replay_limit=1 if corruption == 'feature' else 0)
