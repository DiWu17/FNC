"""Predeclared V13 controls; no prior experiment source is modified."""
from pathlib import Path
import argparse
import json
import time

from clickagent.action_benchmark import sha256, write_json

OUT = Path('runs/multiregion_v13')
SEEDS = {'s1': 20261311, 's2': 20261312}
SCHEDULE_SEED = 20261321
ARMS = ('c1_top1', 'c3_top1', 'c3_top4')
BUDGETS = (4, 8, 16, 32)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def cases(split):
    return rows(OUT/'data'/f'{split}.jsonl')


def sources():
    names = ['scripts/multiregion_v13_protocol.py', 'scripts/prepare_multiregion_v13.py',
        'scripts/extract_multiregion_v13.py', 'scripts/cache_multiregion_v13.py',
        'scripts/train_multiregion_v13.py', 'scripts/run_multiregion_v13.py',
        'scripts/evaluate_multiregion_v13.py', 'scripts/audit_multiregion_v13.py',
        'scripts/evaluate_crossfit_candidates.py',
        'src/clickagent/multiregion_scorer.py',
        'src/clickagent/multiregion_proposals.py']
    from crossfit_protocol import source_files
    return sorted(set([Path(p) for p in names]+source_files()))


def initialize():
    protocol = dict(version=13, created_at=time.time(), seeds=SEEDS,
        purpose='Spatially diverse search, task-specific readout objectives and scorer-source-heldout epoch selection',
        primary='c3_top1_s1_loso vs c1_top1_s1_val on fresh Pro96+Venus96; report all ablations, S2 fixed robustness',
        diagnostic_supplement='all six strictly unused ScreenSpot-v2 RGB components; not pooled into primary denominator',
        upstream='reuse sealed V12 five OOF proposers on train, full D on val/test; backbone frozen',
        crops=dict(max_regions=3, max_pairwise_box_iou=.25, first='identical to sealed V12 highest coarse candidate',
                   subsequent='descending global coarse score with stable ties; no labels', scale=2),
        candidates='nested K=4/8/16/32; global K/2 plus local round-robin K/2; slice before integer dedup, no refill',
        representation='all own generated target-name tokens, 96 heads, query/image/premerge visual states; no coordinate generation',
        fusion='shared V11 local encoder; max candidate-conditioned embeddings over containing local views; mean local null context; unchanged parameter count',
        c1_exact='two-view path delegates unchanged V11 module; input, scores and gradients regression-tested',
        objectives=dict(top1='unchanged multi-positive listwise/null +.25 balanced point BCE',
            top4='relu(1+fourth highest negative-best positive) +.25 balanced BCE; <4 negatives zero rank; no positive uses original null loss'),
        top4_limit='optimizes correct-candidate retention, not which distractors GUI finds convincing; no GUI teacher used in training',
        optimization=dict(epochs=8, learning_rate=.0002, weight_decay=.02, accumulation=8, gradient_clip=1.),
        schedule='same keyed query/budget/permutation/dropout plan across full arms and seeds; mixed K4/8/16/32; unreachable queries retained',
        selection=dict(val='full172 val Top1 hits or Top4 retention; tie lower objective loss, earliest epoch',
            loso='S1 c3 objective-specific four scorer-source-heldout curves; train non-source, remove crossing near-image components; evaluate held-source val; select shared epoch by source macro rate, then mean loss, earliest',
            transfer='apply S1 LOSO-selected objective epoch to both full-data seeds without test-based seed/epoch selection',
            limitation='upstream supervised proposer trained on these source domains and selected using all172val; scorer-only CV, not end-to-end unseen-domain evaluation'),
        references='same actual c1/c3 K16 GUI-Actor scores, fixed quotaK4, same original-image native Qwen completed-lax coordinates',
        primary_inference='forced best candidate, no learned rejection threshold; Top4 GUI counterfactual from same fixed scores',
        metric='any valid annotated bbox/polygon region hit, all primary192 denominator, no exact-center requirement; not permission false-click rate',
        uncertainty='source-stratified RGB cluster paired bootstrap2000; fixed trained-model conditional interval',
        cost='multicrop increases Qwen forwards and pixels despite fixed final candidate/verifier budgets; cached head/GUI subset times are not end-to-end speed measurements',
        freeze_order='fixed data/protocol -> trainval crops/cache -> full and LOSO scorers -> selected checkpoints/source+input seal -> fresh test forward',
        manifest_sha256=sha256(OUT/'manifest.jsonl'), data_selection_sha256=sha256(OUT/'data_selection.json'))
    path = OUT/'protocol.json'
    if path.exists():
        old = read(path); protocol['created_at'] = old['created_at']; assert old == protocol
    else:
        write_json(path, protocol)


def assert_sealed():
    frozen = read(OUT/'frozen.json'); assert frozen['complete']
    for p, digest in frozen['files'].items():
        assert sha256(Path(p)) == digest, p
    return frozen


def freeze():
    assert not list((OUT/'raw/global').glob('*.npz')), 'Test extracted before training freeze'
    selection = read(OUT/'models/selection.json'); assert selection['complete']
    identities = {str(p): sha256(p) for p in sources()}
    for p in [OUT/'protocol.json', OUT/'manifest.jsonl', OUT/'data_selection.json', OUT/'models/selection.json']:
        identities[str(p)] = sha256(p)
    for p in [OUT/'runtime_identity.json', OUT/'data_selection_audit.json', OUT/'data/leave_source_validation.json']:
        identities[str(p)] = sha256(p)
    for group in ('data', 'training', 'cross_validation'):
        for p in (OUT/group).glob('*'):
            if p.is_file() and p.suffix in ('.json', '.jsonl', '.pt'):
                identities[str(p)] = sha256(p)
    for p in (OUT/'models').glob('*.pt'):
        identities[str(p)] = sha256(p)
    evidence = {}
    for split in ('train', 'val'):
        for condition in ('c1', 'c3'):
            p = OUT/'cache'/f'{split}_{condition}_index.json'
            index = read(p); assert index['complete']; identities[str(p)] = sha256(p)
            for record in index['cases']:
                evidence[record['path']] = record['sha256']
                evidence.update(record['evidence'])
    for p, digest in evidence.items():
        assert sha256(Path(p)) == digest, p
    for p in (Path('runs/crossfit_candidates_v12/proposers')).glob('*.pt'):
        identities[str(p)] = sha256(p)
    write_json(OUT/'frozen.json', dict(complete=True, created_at=time.time(), fresh_test_features_at_freeze=0,
        files=identities, training_evidence=evidence, all_training_evidence_verified=True))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=('init', 'freeze'))
    args = p.parse_args(); initialize() if args.stage == 'init' else freeze()
