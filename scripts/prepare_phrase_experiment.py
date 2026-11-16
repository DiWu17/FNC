"""Seal a new test cohort for late token fusion, reusing unchanged train/val."""
import json
from pathlib import Path

from clickagent.action_benchmark import sha256, write_json
from run_target_token_experiment import key, read_json, read_jsonl


OUT = Path('runs/phrase_v4')
SEED = 20261102


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/'protocol.json').exists():
        print('Already sealed: '+str(OUT/'protocol.json'))
        return
    previous = read_jsonl(Path('runs/conditioned_v3/manifest.jsonl'))
    trainval = [c for c in previous if c['experiment_split'] in ('train', 'val')]
    index = read_json(Path('data/conditioned_v3/public_image_index.json'))
    by_path = {r['image']: r for r in index['images']}
    # Include every earlier public/semantic/conditioned selected cohort, not
    # merely the most recent benchmark. No predictions enter this selection.
    manifests = list(Path('runs/public_v1').glob('*/selected_manifest.jsonl'))
    manifests += [Path(f'runs/{n}/manifest.jsonl') for n in ('candidate_v2', 'target_token_v1', 'semantic_target_v2', 'conditioned_v3')]
    used = {c['pixel_sha256'] for c in trainval}
    for path in manifests:
        if not path.exists():
            continue
        for c in read_jsonl(path):
            if c['image'] in by_path:
                used.add(by_path[c['image']]['pixel_sha256'])
    seen = set(used); tests = []; available = {}; pools = {}
    for ds in ('screenspot_v2', 'screenspot_pro', 'osworld_g', 'venusbench_gd'):
        candidates = sorted(read_jsonl(Path(f'data/public_v1/manifests/{ds}.jsonl')),
                            key=lambda c: key(f'{SEED}:phrase-test:{c["sample_id"]}'))
        eligible = []; unique = set()
        for c in candidates:
            info = by_path[c['image']]
            if not c['expected_actionable'] or not c.get('target_geometry') or info['pixel_sha256'] in seen or info['pixel_sha256'] in unique:
                continue
            unique.add(info['pixel_sha256']); eligible.append(c)
        available[ds] = len(eligible)
        pools[ds] = eligible
    # Only 24 OSWorld-G screenshots remain untouched. Keep the total at 192
    # and report per-dataset as well as macro accuracy; do not reuse old tests.
    quotas = dict(screenspot_v2=56, screenspot_pro=56, osworld_g=24, venusbench_gd=56)
    for ds, quota in quotas.items():
        if available[ds] < quota:
            raise ValueError(f'Insufficient untouched images: {available}')
        for c in pools[ds][:quota]:
            info = by_path[c['image']]; seen.add(info['pixel_sha256'])
            tests.append(dict(c, experiment_split='test', image_sha256=info['file_sha256'],
                              pixel_sha256=info['pixel_sha256'], dhash256=info['dhash256']))
    assert len({c['pixel_sha256'] for c in tests}) == 192
    rows = trainval+tests
    (OUT/'manifest.jsonl').write_text(''.join(json.dumps(c, ensure_ascii=False)+'\n' for c in rows), encoding='utf-8')
    protocol = {
        'question': 'Does retaining separate name-token spatial evidence until position-specific fusion improve frozen-VLM pointing?',
        'manifest_sha256': sha256(OUT/'manifest.jsonl'), 'selection_seed': SEED,
        'trainval_feature_root': 'runs/conditioned_v3', 'train_queries': 618, 'val_queries': 172, 'test_queries': 192,
        'test_per_dataset': quotas, 'available_untouched_images': available, 'earlier_manifest_files': [str(p) for p in manifests],
        'prior_images_excluded': len(used), 'prediction_based_test_selection': False,
        'variants': ['first', 'uniform', 'global', 'local'],
        'primary_seed': SEED, 'replication_seed': SEED+1, 'replication_variants': ['first', 'local'],
        'primary_model': 'local', 'checkpoint_selection': 'validation point hits, then lower validation loss, then earlier epoch',
        'optimizer': 'AdamW', 'learning_rate': 0.0003, 'weight_decay': 0.01, 'batch_size': 8,
        'max_epochs': 32, 'min_epochs': 12, 'early_stop_patience': 8, 'width': 32,
        'first_token_position_bias': False, 'absolute_xy_features': False, 'global_spatial_cnn': False,
        'name_features': 'all actually generated own-token rows, unchanged Qwen layers 15/28/36',
        'first_control': 'first token attention only; complete phrase hidden state shared with all variants',
        'coarse_loss': 'direct positive-area cross entropy on coarse cells',
        'fine_loss': 'conditional four-child cell cross entropy plus twice weighted SmoothL1 offsets',
        'decode': 'coarse argmax, then best child within that coarse cell; fine head cannot move the selected parent',
        'baselines': ['previous_full', 'previous_attention_only', 'previous_static_96', 'qwen_native_strict', 'qwen_native_completed_lax', 'ui_tars'],
        'metric': 'point inside any valid target box/polygon, inclusive; all missing names/WAIT/format failures retained',
        'max_pixels': 262144, 'backbone_trainable_parameters': 0,
        'limitations': 'positive static pointing only, no negative false-click rate or real computer execution; same training data isolates readout changes',
    }
    write_json(OUT/'protocol.json', protocol)
    print(json.dumps(protocol, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
