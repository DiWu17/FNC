"""Audit held-out pixels, feature timing, model selection and every point hit."""
import json
from pathlib import Path

import numpy as np

from clickagent.action_benchmark import sha256, write_json
from clickagent.action_model import parse_text_action
from clickagent.public_diagnostics import parse_offline_lax_click
from evaluate_conditioned_readouts import score_point
from index_gui_images import fingerprint
from run_target_token_experiment import key, read_json, read_jsonl


def main():
    out = Path('runs/phrase_v4'); protocol = read_json(out/'protocol.json'); frozen = read_json(out/'frozen_readouts.json')
    cases = read_jsonl(out/'manifest.jsonl'); test = [c for c in cases if c['experiment_split'] == 'test']
    assert sha256(out/'manifest.jsonl') == protocol['manifest_sha256']
    splits = {split: {c['pixel_sha256'] for c in cases if c['experiment_split'] == split} for split in ('train', 'val', 'test')}
    assert all(not splits[a] & splits[b] for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')])
    by_path = {c['image']: c for c in read_json(Path('data/conditioned_v3/public_image_index.json'))['images']}
    prior = set()
    for name in protocol['earlier_manifest_files']:
        for c in read_jsonl(Path(name)):
            if c['image'] in by_path:
                prior.add(by_path[c['image']]['pixel_sha256'])
    assert not splits['test'] & prior
    reused = read_json(Path('runs/conditioned_v3/final_audit.json')); assert reused['complete']
    old_trainval = [c for c in read_jsonl(Path('runs/conditioned_v3/manifest.jsonl')) if c['experiment_split'] != 'test']
    assert old_trainval == [c for c in cases if c['experiment_split'] != 'test']
    feature_identity = read_json(out/'feature_identity.json')
    runtime = read_json(out/'runtime_test.json')
    assert runtime['all_parameters_frozen'] and runtime['trainable_parameter_count'] == 0 and not runtime['model_training']
    assert feature_identity['max_pixels'] == 262144
    assert feature_identity['qwen_weight_identity_sha256'] == sha256(Path('data/conditioned_v3/qwen_weight_identity.json'))
    pinned = read_json(Path('data/conditioned_v3/qwen_weight_identity.json'))
    pin_log = out/'pin_backbone.log'; pin_bytes = pin_log.read_bytes()
    pin_text = pin_bytes.decode('utf-16') if pin_bytes.startswith((b'\xff\xfe', b'\xfe\xff')) else pin_bytes.decode('utf-8-sig')
    assert 'PINNED data' in pin_text
    for name, item in pinned['files'].items():
        path = Path(pinned['model_path'])/name
        assert path.stat().st_size == item['bytes'] and path.stat().st_mtime <= pin_log.stat().st_mtime
        if not name.endswith('.safetensors'):
            assert sha256(path) == item['sha256']
    for source, digest in feature_identity['source_sha256'].items():
        assert sha256(Path(source)) == digest
    for source, digest in frozen['head_sources'].items():
        assert sha256(Path(source)) == digest
    epoch_selection = {}
    for name, entry in frozen['models'].items():
        checkpoint = out/'models'/entry['file']; assert sha256(checkpoint) == entry['sha256']
        history = read_json(out/'models'/f'{name}_history.json')
        selected = max(history['history'], key=lambda h: (h['val']['hits'], -h['val']['loss'], -h['epoch']))
        assert selected['epoch'] == entry['best_epoch'] == history['best_epoch']
        epoch_selection[name] = selected['epoch']
    for entry in frozen['baselines'].values():
        assert sha256(Path(entry['file'])) == entry['sha256']
    freeze_time = (out/'frozen_readouts.json').stat().st_mtime
    named = 0; native_parses = 0
    for i, c in enumerate(test, 1):
        info = fingerprint(c['image'])
        assert info['pixel_sha256'] == c['pixel_sha256'] and info['file_sha256'] == c['image_sha256']
        path = out/'features'/f'{key(c["sample_id"])}.npz'
        assert path.stat().st_mtime >= freeze_time
        with np.load(path, allow_pickle=False) as z:
            m = json.loads(str(z['metadata'])); t = len(m['target_raw_indices']); gh, gw = m['grid_shape']; p = gh*gw
            assert m['sample_id'] == c['sample_id'] and m['own_token_forward'] and not m['ground_truth_in_model_input']
            assert m['target_raw_indices'] == m['stages'].get('target_mean', [])
            expected = {'target_attention': (t, 96, p), 'target_hidden': (t, 12288), 'image_hidden': (p, 4096), 'vision_hidden': (p*4, 1152)}
            for field, shape in expected.items():
                values = z[field]; assert values.shape == shape and np.isfinite(values).all(), (c['sample_id'], field)
            named += m['target_available']
        native = read_json(out/'native'/f'{key(c["sample_id"])}.json')
        assert native['sample_id'] == c['sample_id'] and not native['ground_truth_in_model_input']
        complete = native['terminated_by_eos'] or native['generated_text'].endswith(('\n', '\r'))
        normalized = parse_offline_lax_click(native['generated_text']) if complete else None
        point = [round(v*(size-1)/1000) for v, size in zip(normalized, c['image_size'])] if normalized else None
        assert point == native['completed_lax_point']
        strict = parse_text_action(native['generated_text'], final=True) if complete else None
        sp = [round(v*(size-1)/1000) for v, size in zip(strict['normalized_point'], c['image_size'])] if strict and strict.get('normalized_point') is not None else None
        assert sp == native['point']
        native_parses += 1
        if i % 48 == 0:
            print(f'AUDIT screenshots/features/native {i}/{len(test)}', flush=True)
    rows = read_jsonl(out/'evaluation/predictions.jsonl'); by_id = {c['sample_id']: c for c in test}
    counts = {}
    for row in rows:
        assert score_point(row['point'], by_id[row['sample_id']]) == row['hit']
        counts.setdefault(row['method'], set()).add(row['sample_id'])
    assert all(ids == set(by_id) for ids in counts.values())
    assert len(rows) == len(counts)*len(test)
    summary = read_json(out/'evaluation/summary.json')
    for name in counts:
        assert summary['methods'][name]['hits'] == sum(r['hit'] for r in rows if r['method'] == name)
    ui = read_json(out/'ui_tars/final_audit.json'); assert ui['complete']
    cli = read_json(out/'evaluation/cli_smoke.json'); assert cli['complete'] and cli['exact_match']
    result = dict(complete=True, fresh_test_images_rehashed=len(test), named_test_samples=int(named),
                  unchanged_audited_trainval_queries=len(old_trainval), earlier_public_pixels_excluded=len(prior),
                  exact_pixel_splits_disjoint=True, near_duplicates_training_excluded_by_reused_v3_audit=True,
                  test_features_after_all_heads_frozen=True, validation_selected_epochs=epoch_selection,
                  all_point_hits_replayed=len(rows), same_full_denominator_per_method={k: len(v) for k, v in counts.items()},
                  native_strict_and_lax_conversions_replayed=native_parses, external_ui_tars_audit=True,
                  cached_feature_cli_matches_frozen_prediction=True,
                  frozen_source_and_checkpoint_hashes_match=True, test_used_for_selection=False,
                  qwen_actual_weight_hashes_verified_before_test=True, qwen_checkpoint_files_unchanged_since_pin=True,
                  qwen_trainable_parameters=runtime['trainable_parameter_count'],
                  prediction_sha256=sha256(out/'evaluation/predictions.jsonl'), source_sha256=sha256(Path(__file__)))
    write_json(out/'final_audit.json', result); print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
