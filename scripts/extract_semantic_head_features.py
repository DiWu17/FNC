"""Collect only added training queries or sealed fresh test, never use GT inputs."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from clickagent.action_benchmark import sha256, write_json
from run_target_token_experiment import key, read_jsonl


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, default=Path('runs/semantic_head_v7'))
    p.add_argument('--stage', choices=['extra_train', 'test'], required=True); p.add_argument('--limit', type=int)
    args = p.parse_args(); out = args.output; protocol = read(out/'protocol.json')
    assert sha256(out/'manifest.jsonl') == protocol['manifest_sha256']
    cases = read_jsonl(out/'manifest.jsonl')
    selected = [c for c in cases if c['experiment_split'] == 'test'] if args.stage == 'test' else [c for c in cases if c.get('training_extension')]
    if args.stage == 'test':
        frozen = read(out/'frozen_readouts.json')
        for source, digest in frozen['sources'].items(): assert sha256(Path(source)) == digest, source
        for item in frozen['models'].values(): assert sha256(out/'models'/item['file']) == item['sha256']
        assert frozen['fresh_test_features_at_freeze'] == 0
    cfg = read('runs/candidate_v2/identity.json')['config']
    cfg.update(max_new_tokens=64, seed=20261101, conditions=['prefill'], offload_folder=str((out/'offload').resolve()))
    sources = ['src/clickagent/'+n for n in ['conditioned_features.py','target_token.py','action_model.py',
        'model_features.py','public_model.py','public_diagnostics.py']]+[str(Path(__file__))]
    identity = dict(manifest_sha256=protocol['manifest_sha256'], config=cfg,
        source_sha256={s: sha256(Path(s)) for s in sources},
        qwen_weight_identity_sha256=sha256(Path('data/conditioned_v3/qwen_weight_identity.json')),
        image_content_layer=15, max_pixels=262144, gt_in_inputs=False)
    identity_path = out/'feature_identity.json'
    if identity_path.exists(): assert read(identity_path) == identity
    write_json(identity_path, identity)
    snapshot = out/'feature_sources'; snapshot.mkdir(exist_ok=True)
    for s in sources:
        dest = snapshot/Path(s).name
        if not dest.exists(): shutil.copyfile(s, dest)
    folder = out/'features'; folder.mkdir(exist_ok=True); native = out/'native'; native.mkdir(exist_ok=True)
    pending = [c for c in selected if not (folder/f'{key(c["sample_id"])}.npz').exists() or
        (args.stage == 'test' and not (native/f'{key(c["sample_id"])}.json').exists())]
    if args.limit: pending = pending[:args.limit]
    if pending:
        from PIL import Image
        from clickagent.model_features import ExtractionConfig
        from clickagent.conditioned_features import ConditionedFeatureRuntime
        from clickagent.public_diagnostics import parse_offline_lax_click
        cfg.update(layers=tuple(cfg['layers']), conditions=tuple(cfg['conditions']),
            max_memory={int(k) if str(k).isdigit() else k: v for k, v in cfg['max_memory'].items()})
        runtime = ConditionedFeatureRuntime(ExtractionConfig(**cfg))
        write_json(out/f'runtime_{args.stage}.json', runtime.runtime_metadata)
        for i, c in enumerate(pending, 1):
            assert sha256(Path(c['image'])) == c['image_sha256']
            with Image.open(c['image']) as opened: image = opened.convert('RGB')
            path = folder/f'{key(c["sample_id"])}.npz'
            if not path.exists():
                arrays, meta = runtime.infer_spatial_features(image, c['instruction'])
                assert meta['all_backbone_parameters_frozen'] and meta['own_token_forward']
                meta.update(sample_id=c['sample_id'], experiment_split=c['experiment_split'],
                    image_sha256=c['image_sha256'], pixel_sha256=c['pixel_sha256'])
                arrays['metadata'] = np.asarray(json.dumps(meta, ensure_ascii=False))
                temp = path.with_suffix('.tmp')
                with temp.open('wb') as stream: np.savez_compressed(stream, **arrays)
                temp.replace(path)
            else:
                with np.load(path, allow_pickle=False) as z: meta = json.loads(str(z['metadata']))
            npth = native/f'{key(c["sample_id"])}.json'
            if args.stage == 'test' and not npth.exists():
                result = runtime.infer_text_action_public(image, c['instruction'], max_new_tokens=32)
                complete = result['terminated_by_eos'] or result['generated_text'].endswith(('\n', '\r'))
                point = parse_offline_lax_click(result['generated_text']) if complete else None
                result['completed_lax_point'] = [round(v*(size-1)/1000) for v, size in zip(point, image.size)] if point else None
                result.update(sample_id=c['sample_id'], ground_truth_in_model_input=False)
                write_json(npth, result)
            print(f'{args.stage} [{i}/{len(pending)}] {c["dataset"]} name={meta["target_text"]!r}', flush=True)
    done = all((folder/f'{key(c["sample_id"])}.npz').exists() for c in selected)
    if args.stage == 'test': done &= all((native/f'{key(c["sample_id"])}.json').exists() for c in selected)
    write_json(out/f'features_{args.stage}_completion.json', dict(complete=done, n=len(selected), manifest_sha256=protocol['manifest_sha256']))


if __name__ == '__main__': main()
