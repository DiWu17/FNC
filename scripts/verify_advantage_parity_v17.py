"""Replay cached development features through the in-memory deployment path."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
import numpy as np
from PIL import Image
import torch
import run_quality_readout_v15_3k as prior
import run_contribution_v16 as contribution
from clickagent.advantage_runtimes import OurRuntime
from clickagent.crossfit_proposer import load_crossfit_proposer
from clickagent.evidence_retention import RetentionModel


def main():
    torch.set_num_threads(4)
    cases = [json.loads(s) for s in (ROOT / 'runs/quality_readout_v15_3k/data/new_reserve.jsonl').read_text(encoding='utf-8').splitlines()]
    by_case = {c['sample_id']: c for c in cases}
    entries = prior.read(ROOT / 'runs/quality_readout_v15_3k/features/new_reserve/index.json')['cases']
    # Exercise a nonempty multiview case and a zero-candidate case, development only.
    nonempty = sorted((e for e in entries if e['n']), key=lambda e: e['sample_id'])[:2]
    empty = sorted((e for e in entries if not e['n']), key=lambda e: e['sample_id'])[:1]
    records = prior.base.load_records(nonempty + empty)
    expected = {r['sample_id']: r for r in prior.read(ROOT / 'runs/contribution_v16/evaluation/new_development/details.json')}
    obj = OurRuntime.__new__(OurRuntime)
    obj.base, obj.contribution = prior.base, contribution
    obj.proposer = load_crossfit_proposer(ROOT / 'runs/crossfit_candidates_v12/proposers/full.pt', 'cpu')[0].eval()
    obj.models = []
    for i in [1, 2, 3]:
        saved = torch.load(ROOT / f'runs/quality_readout_v15_3k/models/expanded_s{i}.pt', map_location='cpu', weights_only=True)
        model = RetentionModel().to('cuda:0').eval()
        model.load_state_dict(saved['state_dict'])
        obj.models.append((model, saved))
    checked = []
    for record in records:
        case = by_case[record['sample_id']]
        pipe = prior.read(record['pipeline_path'])
        paths = [pipe['global_feature_path']] + [v['feature_path'] for v in pipe['local_views']]
        class FrozenFeatures:
            position = 0
            def infer_spatial_features(self, image, instruction):
                assert instruction == case['instruction']
                path = paths[self.position]
                self.position += 1
                with np.load(path, allow_pickle=False) as z:
                    metadata = json.loads(str(z['metadata']))
                    arrays = {k: z[k].copy() for k in ['target_attention', 'target_hidden', 'image_hidden', 'vision_hidden']}
                assert list(image.size) == metadata['image_size']
                return arrays, metadata
        obj.runtime = FrozenFeatures()
        with Image.open(case['image']) as opened:
            result = obj.infer(opened.convert('RGB'), case['instruction'])
        assert obj.runtime.position == len(paths)
        for key in ['points', 'base', 'active', 'baseline', 'state', 'mask']:
            assert np.array_equal(obj.last_record[key], record[key]), (record['sample_id'], key)
        expected_points = [expected[record['sample_id']]['methods'][f'aligned_s{i}']['point'] for i in [1, 2, 3]]
        assert result['seed_points'] == expected_points, (record['sample_id'], result['seed_points'], expected_points)
        checked.append(dict(sample_id=record['sample_id'], candidates=record['n'], views=len(paths),
                            byte_exact_feature_replay=True, all3_seed_points_equal=True))
    out = ROOT / 'runs/advantage_validation_v17/setup/in_memory_parity.json'
    out.write_text(json.dumps(dict(complete=True, development_only=True, checked=checked,
        adapter_sha256=hashlib.sha256((ROOT / 'src/clickagent/advantage_runtimes.py').read_bytes()).hexdigest()),
        ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(out.read_text(encoding='utf-8'))


if __name__ == '__main__':
    main()
