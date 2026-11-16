"""Audit fresh pixels, source-preserving inputs, epoch choice and every score."""
from collections import Counter
import json
from pathlib import Path

import numpy as np

from clickagent.action_benchmark import sha256, write_json
from clickagent.action_model import parse_text_action
from clickagent.candidate_ranker import build_candidate_inputs
from clickagent.public_diagnostics import parse_offline_lax_click
from evaluate_conditioned_readouts import score_point
from index_gui_images import fingerprint
from run_target_token_experiment import key, read_jsonl


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    out = Path('runs/candidate_ranker_v5'); protocol = read(out/'protocol.json'); frozen = read(out/'frozen_readouts.json')
    cases = read_jsonl(out/'manifest.jsonl'); tests = [c for c in cases if c['experiment_split'] == 'test']
    assert protocol['manifest_sha256'] == sha256(out/'manifest.jsonl')
    previous = [c for c in read_jsonl(Path('runs/phrase_v4/manifest.jsonl')) if c['experiment_split'] in ('train', 'val')]
    assert previous == [c for c in cases if c['experiment_split'] != 'test']
    index = {r['image']: r for r in read('data/conditioned_v3/public_image_index.json')['images']}
    used_pixels = {c['pixel_sha256'] for c in previous}; used_dhash = {int(c['dhash256'], 16) for c in previous}
    for path in protocol['prior_manifest_files']:
        for c in read_jsonl(Path(path)):
            info = index.get(c.get('image'))
            pixel = c.get('pixel_sha256') or (info['pixel_sha256'] if info else None)
            dhash = c.get('dhash256') or (info['dhash256'] if info else None)
            if pixel: used_pixels.add(pixel)
            if dhash: used_dhash.add(int(dhash, 16))
    assert len({c['pixel_sha256'] for c in tests}) == len(tests) == 192
    assert not {c['pixel_sha256'] for c in tests} & used_pixels
    assert all(min((int(c['dhash256'], 16) ^ h).bit_count() for h in used_dhash) > 2 for c in tests)
    for source, digest in frozen['head_sources'].items(): assert sha256(Path(source)) == digest, source
    epochs = {}
    for name, entry in frozen['models'].items():
        assert sha256(out/'models'/entry['file']) == entry['sha256']
        history = read(out/'models'/f'{name}_history.json')
        best = max(history['history'], key=lambda h: (h['val']['region_hits']['1'], h['val']['hits'], -h['val']['loss'], -h['epoch']))
        assert best['epoch'] == entry['best_epoch'] == history['best_epoch']
        epochs[name] = best['epoch']
    runtime = read(out/'runtime_test.json'); identity = read(out/'feature_identity.json')
    assert runtime['all_parameters_frozen'] and runtime['trainable_parameter_count'] == 0 and not runtime['model_training']
    assert identity['max_pixels'] == 262144
    assert identity['qwen_weight_identity_sha256'] == sha256(Path('data/conditioned_v3/qwen_weight_identity.json'))
    for source, digest in identity['source_sha256'].items(): assert sha256(Path(source)) == digest
    log = out/'pin_backbone.log'; raw_log = log.read_bytes()
    pin_text = raw_log.decode('utf-16') if raw_log.startswith((b'\xff\xfe', b'\xfe\xff')) else raw_log.decode('utf-8-sig')
    assert 'PINNED data' in pin_text
    pin = read('data/conditioned_v3/qwen_weight_identity.json')
    for name, entry in pin['files'].items():
        path = Path(pin['model_path'])/name
        assert path.stat().st_size == entry['bytes'] and path.stat().st_mtime <= log.stat().st_mtime
    freeze_time = (out/'frozen_readouts.json').stat().st_mtime; named = 0; candidate_counts = []
    for i, c in enumerate(tests):
        info = fingerprint(c['image'])
        assert info['file_sha256'] == c['image_sha256'] and info['pixel_sha256'] == c['pixel_sha256']
        source = out/'features'/f'{key(c["sample_id"])}.npz'
        assert source.stat().st_mtime >= freeze_time
        with np.load(source, allow_pickle=False) as z:
            meta = json.loads(str(z['metadata']))
            assert meta['sample_id'] == c['sample_id'] and meta['own_token_forward'] and not meta['ground_truth_in_model_input']
            assert meta['all_backbone_parameters_frozen']
            if meta['target_available']:
                built = build_candidate_inputs(z['target_attention'], z['target_hidden'], z['image_hidden'], z['vision_hidden'], meta['grid_shape'])
                with np.load(out/'candidate_cache'/f'{key(c["sample_id"])}.npz', allow_pickle=False) as cached:
                    for name, value in built.items(): np.testing.assert_array_equal(value, cached[name])
                    cached_meta = json.loads(str(cached['metadata']))
                    assert cached_meta['source_archive_sha256'] == sha256(source)
                named += 1; candidate_counts.append(len(built['parent_indices']))
        native = read(out/'native'/f'{key(c["sample_id"])}.json')
        assert not native['ground_truth_in_model_input'] and native['sample_id'] == c['sample_id']
        complete = native['terminated_by_eos'] or native['generated_text'].endswith(('\n', '\r'))
        normalized = parse_offline_lax_click(native['generated_text']) if complete else None
        point = [round(v*(size-1)/1000) for v, size in zip(normalized, c['image_size'])] if normalized else None
        assert point == native['completed_lax_point']
        strict = parse_text_action(native['generated_text'], final=True) if complete else None
        sp = [round(v*(size-1)/1000) for v, size in zip(strict['normalized_point'], c['image_size'])] if strict and strict.get('normalized_point') is not None else None
        assert sp == native['point']
        if (i+1) % 48 == 0: print(f'AUDIT fresh pixels / candidate arrays / native {i+1}/192', flush=True)
    rows = read_jsonl(out/'evaluation/predictions.jsonl'); by_id = {c['sample_id']: c for c in tests}; summary = read(out/'evaluation/summary.json')
    seen = set(); counts = Counter()
    for row in rows:
        pair = (row['method'], row['sample_id']); assert pair not in seen; seen.add(pair)
        assert row['hit'] == score_point(row['point'], by_id[row['sample_id']])
        counts[row['method']] += 1
    assert all(v == 192 for v in counts.values())
    for name in counts: assert sum(r['hit'] for r in rows if r['method'] == name) == summary['methods'][name]['hits']
    assert read(out/'ui_tars/final_audit.json')['complete']
    cli = read(out/'evaluation/cli_smoke.json')
    assert cli['complete'] and cli['raw_input_and_candidate_cache_same_point']
    assert cli['checkpoint_sha256'] == frozen['models']['full']['sha256']
    test_bytes = (out/'tests.log').read_bytes()
    test_text = test_bytes.decode('utf-16') if test_bytes.startswith((b'\xff\xfe', b'\xfe\xff')) else test_bytes.decode('utf-8-sig')
    assert '6 passed' in test_text
    old = read('runs/phrase_v4/frozen_readouts.json')
    for name in ('local', 'global'):
        entry = old['models'][name]; assert sha256(Path('runs/phrase_v4/models')/entry['file']) == entry['sha256']
    result = dict(complete=True, same_fresh_images=192, named=named,
        all_prior_exact_pixels_excluded=len(used_pixels), prior_near_hash_min_distance=3,
        candidate_inputs_reconstructed_without_geometry=named, candidate_count_median=float(np.median(candidate_counts)),
        test_features_after_freeze=True, validation_selected_epochs=epochs, full_denominator_per_method=dict(counts),
        points_replayed=len(rows), native_strict_lax_replayed=192, official_ui_tars_audit=True,
        backbone_trainable_parameters=0, checkpoint_and_source_hashes_match=True, fresh_test_used_for_selection=False,
        candidate_ranker_tests_passed=6, raw_cache_cli_smoke_matches=True,
        training_queries=618, validation_queries=172, prediction_sha256=sha256(out/'evaluation/predictions.jsonl'),
        source_sha256=sha256(Path(__file__)))
    write_json(out/'final_audit.json', result); print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__': main()
