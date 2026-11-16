"""Decode one cached raw name-token trajectory through the frozen ranker."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from clickagent.candidate_ranker import build_candidate_inputs, decode_ranker, load_ranker, make_ranker_batch


def main():
    p = argparse.ArgumentParser(); p.add_argument('--features', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, default=Path('runs/candidate_ranker_v5/models/full.pt'))
    p.add_argument('--device', default='cpu'); args = p.parse_args()
    with np.load(args.features, allow_pickle=False) as z:
        meta = json.loads(str(z['metadata']))
        if not meta['target_available']:
            print(json.dumps(dict(action='WAIT', reason='no_generated_target_name'))); return
        r = build_candidate_inputs(z['target_attention'], z['target_hidden'], z['image_hidden'], z['vision_hidden'], meta['grid_shape'])
    r.update(metadata=meta, case=dict(image_size=meta['image_size']))
    model, _ = load_ranker(args.checkpoint, args.device)
    with torch.inference_mode():
        inputs, _ = make_ranker_batch([r], args.device, labels=False); output = model(**inputs)
        proposals = decode_ranker(output, r)
        rejected = float(output['null_scores'][0]) >= float(output['scores'].max())
    print(json.dumps(dict(action='WAIT' if rejected else 'CLICK', proposed_point=proposals[0]['point'],
        target_name=meta['target_text'], n_candidates=len(r['parent_indices']), proposals=proposals,
        null_policy_rejected=rejected, scores_are_uncalibrated=True), ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
