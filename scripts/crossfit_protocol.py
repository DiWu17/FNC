"""V12 prespecification and immutable, staged experimental identities."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time

from clickagent.action_benchmark import sha256, write_json

OUT = Path('runs/crossfit_candidates_v12')
SEEDS = (20261131, 20261132)
ARMS = ('in_sample', 'oof')
BUDGETS = (4, 8, 16, 32)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def rows(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s.strip()]


def key(value):
    return hashlib.sha256(value.encode()).hexdigest()[:24]


def digest_object(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def raw_path(case):
    name = key(case['sample_id']) + '.npz'
    first = Path('runs/semantic_head_v7/features') / name
    return first if first.exists() else Path('runs/conditioned_v3/features') / name


def cases(split):
    return rows(OUT / 'data' / (split + '.jsonl'))


def initialize():
    selection = read(OUT / 'data_selection.json')
    assert selection['complete']
    protocol = dict(version=12, created_at=time.time(),
        purpose='Matched in-sample versus out-of-fold training candidate experiment',
        primary='oof_s1 vs in_sample_s1, identical full-D K16, fresh history-excluded 192',
        counts={s: len(cases(s)) for s in ('train', 'val', 'test')},
        folds='5 screenshot/near-duplicate groups, source and query balanced; OOF=f, IS=(f+1)%5',
        deployment='sixth fresh full-training proposer D shared by both arms on original validation and fresh test',
        upstream=dict(architecture='PointAlignedRanker semantic32/16/4 connect_fine',
            parameters=611919, initial_seed=20261201, head_prior='constant zero96',
            initialization='same fresh CPU state for 5fold and full-D; no supervised old weight or teacher cache',
            epochs=24, steps_per_epoch=75, batch_size=8, learning_rate=.0003, weight_decay=.01,
            schedule='cosine end0.1lr', loss='aligned_loss(point_objective=True,region_weight=.25)',
            selection='GPU forward plus canonical CPU integer decode on original172val hits, tie lower val loss; never heldout-fold labels',
            sampling='600 named allowed-train queries per epoch, shuffled cycling if needed; no positive filtering'),
        candidate_chain='same assigned proposer for global decoding, crop selection, actual-crop frozen local Qwen states and fine-point decoding',
        frozen_feature_memoization='only raw four Qwen arrays may be reused if exact sample/image/instruction/crop transform and frozen feature contracts match; all learned local outputs recomputed',
        candidate_policy='unchanged V11 globalTopK/2+localTopK/2, integer dedup after quota, no refill',
        scorer='unchanged InternalCandidateScorer(full), 1230338 parameters',
        loss='exact train_candidate_scorer.ranking_loss: multi-positive listwise/null +0.25 balanced point BCE',
        optimization=dict(epochs=16, learning_rate=.0002, weight_decay=.02, accumulation=8,
                          gradient_clip=1., seeds=list(SEEDS)),
        matched_randomness='same keyed query order, per-query budget, candidate permutation and dropout seed; same eligible queries/updates',
        selection='per-arm per-seed common-D original172val K16 hits, tie lower loss; S1 primary, S2 robustness',
        budgets=list(BUDGETS), primary_budget=16,
        primary_inference='forced best nonempty candidate; null is auxiliary, not permission gate',
        secondary='K4/8/32; scorer Top4 on K16 followed by shared GUI scores; legacy V11 test reused migration diagnostic only',
        references=['common-pool GUI-Actor-Verifier K16 and quotaK4', 'Qwen native completed lax'],
        metric='all192 denominator, any integer point inside valid target bbox/polygon is hit; empty pool fails',
        uncertainty='source-stratified screenshot cluster paired bootstrap2000, frozen model conditional uncertainty',
        causal_limit='compares whole cross-fitting protocol; cyclic arms differ in excluded fold, not a pure single-image exposure intervention',
        freeze_order='protocol -> six upstream models -> two train arm caches/common validation -> four scorers -> fresh test extraction',
        manifest_sha256=sha256(OUT / 'manifest.jsonl'), folds_sha256=sha256(OUT / 'folds.json'),
        data_selection_sha256=sha256(OUT / 'data_selection.json'))
    path = OUT / 'protocol.json'
    if path.exists():
        old = read(path); protocol['created_at'] = old['created_at']; assert old == protocol
    else:
        write_json(path, protocol)
    print('V12 PROTOCOL FIXED', flush=True)


def source_files():
    names = ['scripts/crossfit_protocol.py', 'scripts/prepare_crossfit_candidates.py',
             'scripts/train_crossfit_proposer.py', 'scripts/extract_crossfit_candidates.py',
             'scripts/train_crossfit_scorer.py', 'scripts/run_crossfit_scorer.py', 'scripts/verify_crossfit_runtime.py',
             'scripts/train_candidate_scorer.py', 'scripts/cache_candidate_scorer.py',
             'scripts/run_point_aligned_inference.py', 'scripts/run_gui_verifier.py']
    names += ['src/clickagent/' + x + '.py' for x in (
        'crossfit_proposer', 'candidate_ranker', 'semantic_head_ranker', 'attention_field_ranker',
        'point_aligned_readout', 'point_aligned_inference', 'candidate_budget', 'internal_candidate_scorer',
        'context_verifier', 'conditioned_features', 'target_token', 'model_features', 'public_evaluation',
        'spatial_geometry', 'point_quality_readout', 'semantic_head_experiment', 'action_model',
        'public_model', 'public_diagnostics')]
    return [Path(n) for n in names]


def freeze():
    assert not list((OUT / 'test').rglob('*.npz')), 'Fresh test already extracted'
    pretest = read(OUT / 'pretest_audit.json')
    assert pretest['complete'] and pretest['stage'] == 'pretest'
    assert pretest['protocol_sha256'] == sha256(OUT / 'protocol.json')
    assert pretest['auditor_sha256'] == sha256(Path('scripts/audit_crossfit_candidates.py'))
    files = {str(p): sha256(p) for p in source_files()}
    files[str(OUT / 'pretest_audit.json')] = sha256(OUT / 'pretest_audit.json')
    files['scripts/audit_crossfit_candidates.py'] = pretest['auditor_sha256']
    for name in ('protocol.json', 'manifest.jsonl', 'folds.json', 'data_selection.json'):
        p = OUT / name; files[str(p)] = sha256(p)
    runtime_identity = read(OUT / 'runtime_identity.json')
    assert runtime_identity['complete']
    files[str(OUT / 'runtime_identity.json')] = sha256(OUT / 'runtime_identity.json')
    evidence = {}
    for path, expected in pretest['input_sha256'].items():
        assert sha256(Path(path)) == expected, path
        evidence[path] = expected
    for role in [f'fold_{i}' for i in range(5)] + ['full']:
        sidecar = OUT / 'proposers' / (role + '.json'); meta = read(sidecar)
        assert meta['complete'], role
        for p in (sidecar, sidecar.with_suffix('.pt')):
            files[str(p)] = sha256(p)
    for name in [a + '_' + s for a in ARMS for s in ('s1', 's2')]:
        p = OUT / 'models' / (name + '.json'); meta = read(p)
        assert meta['complete'] and meta['protocol_sha256'] == sha256(OUT / 'protocol.json')
        assert meta['checkpoint_sha256'] == sha256(p.with_suffix('.pt'))
        files[str(p)] = sha256(p); files[str(p.with_suffix('.pt'))] = sha256(p.with_suffix('.pt'))
    for group in ('in_sample', 'oof', 'val'):
        p = OUT / 'cache' / (group + '_index.json'); index = read(p)
        assert index['complete']; files[str(p)] = sha256(p)
        for row in index['cases']:
            for path, expected in row['evidence'].items():
                if path not in evidence:
                    assert sha256(Path(path)) == expected, path
                    evidence[path] = expected
            assert sha256(Path(row['path'])) == row['sha256']
            evidence[row['path']] = row['sha256']
    write_json(OUT / 'frozen.json', dict(complete=True, created_at=time.time(),
        fresh_test_features_at_freeze=0, files=files, training_evidence=evidence))
    print('FOUR SCORERS AND INPUTS FROZEN', flush=True)


def assert_sealed():
    seal = read(OUT / 'frozen.json')
    assert seal['complete'] and seal['fresh_test_features_at_freeze'] == 0
    for path, expected in seal['files'].items():
        assert sha256(Path(path)) == expected, path


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=['init', 'freeze'])
    args = p.parse_args(); initialize() if args.stage == 'init' else freeze()
