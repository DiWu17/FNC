"""Seal candidate-ranking training and three fresh screenshot test cohorts."""
import json
from pathlib import Path

from clickagent.action_benchmark import sha256, write_json
from run_target_token_experiment import key, read_jsonl


OUT = Path('runs/candidate_ranker_v5')
SEED = 20261104


def main():
    OUT.mkdir(exist_ok=True)
    if (OUT/'protocol.json').exists():
        print('Already sealed'); return
    trainval = [c for c in read_jsonl(Path('runs/phrase_v4/manifest.jsonl')) if c['experiment_split'] in ('train', 'val')]
    index = json.loads(Path('data/conditioned_v3/public_image_index.json').read_text(encoding='utf-8'))
    by_path = {r['image']: r for r in index['images']}
    manifests = list(Path('runs/public_v1').glob('*/selected_manifest.jsonl'))+list(Path('runs').glob('*/manifest.jsonl'))
    used_pixels = {c['pixel_sha256'] for c in trainval}; used_dhash = {int(c['dhash256'], 16) for c in trainval}
    for path in manifests:
        for c in read_jsonl(path):
            info = by_path.get(c.get('image'))
            pixel = c.get('pixel_sha256') or (info['pixel_sha256'] if info else None)
            dhash = c.get('dhash256') or (info['dhash256'] if info else None)
            if pixel: used_pixels.add(pixel)
            if dhash: used_dhash.add(int(dhash, 16))
    tests = []; available = {}; near_rejected = {}; seen = set(used_pixels)
    for ds in ('screenspot_v2', 'screenspot_pro', 'venusbench_gd'):
        pool = sorted(read_jsonl(Path(f'data/public_v1/manifests/{ds}.jsonl')), key=lambda c: key(f'{SEED}:ranker:{c["sample_id"]}'))
        eligible = []; unique = set(); near_count = 0
        for c in pool:
            info = by_path[c['image']]; pixel = info['pixel_sha256']
            if not c['expected_actionable'] or not c.get('target_geometry') or pixel in seen or pixel in unique:
                continue
            unique.add(pixel); value = int(info['dhash256'], 16)
            if any((value ^ old).bit_count() <= 2 for old in used_dhash):
                near_count += 1; continue
            eligible.append(c)
        available[ds] = len(eligible); near_rejected[ds] = near_count
        if len(eligible) < 64:
            raise ValueError(f'Not enough fresh images after exact/near exclusions: {ds}={len(eligible)}')
        for c in eligible[:64]:
            info = by_path[c['image']]; seen.add(info['pixel_sha256'])
            tests.append(dict(c, experiment_split='test', image_sha256=info['file_sha256'],
                pixel_sha256=info['pixel_sha256'], dhash256=info['dhash256']))
    rows = trainval+tests
    assert len(tests) == len({c['pixel_sha256'] for c in tests}) == 192
    (OUT/'manifest.jsonl').write_text(''.join(json.dumps(c, ensure_ascii=False)+'\n' for c in rows), encoding='utf-8')
    protocol = dict(question='Can source-preserving semantic candidate ranking recover target regions from generated name-token/head peaks?',
        primary='full', variants=['full', 'no_visual', 'no_query', 'stats_only'], replica_variants=['full'],
        primary_seed=SEED, replica_seed=SEED+1, train_queries=618, val_queries=172, test_queries=192,
        fresh_test_quotas={ds: 64 for ds in available}, fresh_available=available, near_images_excluded=near_rejected,
        prior_pixel_images_excluded=len(used_pixels), near_hash_threshold=2, prior_manifest_files=[str(p) for p in manifests],
        osworld_g='No untouched screenshots remain; excluded from this new test, not reused.',
        manifest_sha256=sha256(OUT/'manifest.jsonl'), max_pixels=262144, backbone_trainable_parameters=0,
        raw_feature_root_trainval='runs/conditioned_v3', width=32, learning_rate=.0003, weight_decay=.01,
        batch_size=8, max_epochs=24, min_epochs=12, patience=6, candidate_removal_probability=.25,
        checkpoint_selection='validation forced Top1 parent-region hits, then pixel-point hits, then lower loss, then earlier epoch',
        candidates='All distinct maximum cells from every actually generated name token and each of 96 heads; no head pruning, no GT proposal injection.',
        source_pool='Learned candidate-conditioned softmax and max over source token/head evidence; shared projection for layer15 image/query semantics.',
        supervision='GT area over proposed cells; null if no proposal overlaps. Training only: 25% candidate-list counterfactual removing all positive proposals.',
        point_decoders=['new conditional four-child and within-cell offsets', 'parent-cell centers', 'previous frozen phrase_v4 fine decoder at selected parent'],
        baseline_models=['phrase_v4/local', 'phrase_v4/global', 'train_weighted_peak_votes', 'qwen_native_completed_lax', 'UI-TARS-1.5-7B'],
        main_evaluation='forced pointing on positive static tasks; null policy separately, all missing names retained; no real execution or negative false-click claim',
        test_features_required_absent_until_freeze=True, test_used_for_training_or_selection=False)
    write_json(OUT/'protocol.json', protocol)
    print(json.dumps(protocol, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__': main()
