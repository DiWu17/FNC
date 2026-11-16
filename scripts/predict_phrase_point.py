"""Frozen name-trajectory pointing; JSON proposals only, no mouse actions."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from clickagent.conditioned_grounder import make_batch
from clickagent.phrase_grounder import decode_phrase, load_phrase
from run_target_token_experiment import read_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--features', type=Path); source.add_argument('--image', type=Path)
    p.add_argument('--instruction'); p.add_argument('--checkpoint', type=Path, default=Path('runs/phrase_v4/models/local.pt'))
    p.add_argument('--device', default='cpu'); p.add_argument('--respect-action', action='store_true')
    args = p.parse_args(); torch.set_num_threads(4)
    if args.features:
        with np.load(args.features, allow_pickle=False) as z:
            record = {k: z[k].copy() for k in ('target_attention', 'target_hidden', 'image_hidden', 'vision_hidden')}
            meta = json.loads(str(z['metadata']))
    else:
        if not args.instruction:
            p.error('--instruction is required with --image')
        from PIL import Image
        from clickagent.conditioned_features import ConditionedFeatureRuntime
        from clickagent.model_features import ExtractionConfig
        cfg = read_json(args.checkpoint.parent.parent/'feature_identity.json')['config']
        cfg.update(layers=tuple(cfg['layers']), conditions=tuple(cfg['conditions']),
                   max_memory={int(k) if str(k).isdigit() else k: v for k, v in cfg['max_memory'].items()})
        runtime = ConditionedFeatureRuntime(ExtractionConfig(**cfg))
        with Image.open(args.image) as im:
            record, meta = runtime.infer_spatial_features(im.convert('RGB'), args.instruction)
    record['metadata'] = meta
    result = dict(target_name=meta['target_text'], generated_action=meta['action'], point=None, candidates=[],
                  backbone_frozen=True, note='Spatial proposal scores are not calibrated click-execution confidence. No mouse event is executed.')
    if meta['target_available']:
        model, _ = load_phrase(args.checkpoint, args.device)
        with torch.inference_mode():
            inputs, _ = make_batch([record], args.device, labels=False); output = model(**inputs)
            proposals = decode_phrase(output, 0, meta['image_size'])
            weights = output['token_weights'][0, :, proposals[0]['coarse_index']].cpu().tolist()
        prefixes = meta['generated_prefixes']
        pieces = [prefixes[i-1][len(prefixes[i-2]) if i > 1 else 0:] for i in meta['target_raw_indices']]
        result.update(point=proposals[0]['point'], candidates=proposals,
                      token_evidence_at_selected_cell=[dict(piece=piece, weight=weight) for piece, weight in zip(pieces, weights)])
        if args.respect_action and meta['action'] != 'CLICK':
            result['point'] = None
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
