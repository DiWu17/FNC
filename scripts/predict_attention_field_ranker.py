"""Label-free raw trajectory -> v6 proposals; does not execute mouse actions."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from clickagent.attention_field_ranker import build_attention_field, load_field_ranker, make_field_batch
from clickagent.candidate_ranker import build_candidate_inputs, decode_ranker


def main():
    p = argparse.ArgumentParser(); p.add_argument('--features', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, default=Path('runs/attention_field_v6/models/all.pt'))
    p.add_argument('--device', default='cpu'); args = p.parse_args(); torch.set_num_threads(4)
    with np.load(args.features, allow_pickle=False) as z:
        meta = json.loads(str(z['metadata']))
        if not meta['target_available']:
            print(json.dumps(dict(proposed_point=None, reason='no_generated_target_name'))); return
        attention = z['target_attention']
        r = build_candidate_inputs(attention, z['target_hidden'], z['image_hidden'], z['vision_hidden'], meta['grid_shape'])
        r['attention_field'] = build_attention_field(attention, r['parent_indices'])
    r.update(metadata=meta, case=dict(image_size=meta['image_size']))
    model, checkpoint = load_field_ranker(args.checkpoint, args.device)
    with torch.inference_mode():
        inputs, _ = make_field_batch([r], args.device, labels=False); output = model(**inputs)
        proposals = decode_ranker(output, r)
    print(json.dumps(dict(proposed_point=proposals[0]['point'], proposals=proposals,
        target_name=meta['target_text'], field_mode=checkpoint['field_mode'],
        n_candidates=len(r['parent_indices']), scores_are_uncalibrated=True), ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
