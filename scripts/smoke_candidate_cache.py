"""Exercise real V11 cache/label/provenance validation on two OLD train/val cases."""
import copy
from pathlib import Path
import shutil

import torch

from clickagent.action_benchmark import sha256, write_json
from clickagent.candidate_budget import build_pools
import cache_candidate_scorer as cache
import visual_candidate_cache as visual
import candidate_scorer_protocol as protocol
import train_candidate_scorer as training
from point_aligned_data import read
from run_target_token_experiment import read_jsonl, key
from train_context_point_critic import raw_path


if __name__ == '__main__':
    torch.set_num_threads(2)
    original = Path('runs/candidate_scorer_v11')
    dest = original / 'real_cache_smoke'
    cases = read_jsonl(original / 'manifest.jsonl')
    selected = []; pipelines = []
    for split in ('train', 'val'):
        for case in cases:
            if case['experiment_split'] != split:
                continue
            sid = key(case['sample_id'])
            encoding = original / 'vision_encodings' / (case['image_sha256'] + '.npz')
            if not encoding.exists():
                continue
            if split == 'train':
                source = original / 'trainval/pipeline' / (sid + '.json')
                if not source.exists():
                    continue
                pipeline = read(source)
            else:
                source = Path('runs/context_verifier_v9/validation_smoke/pipeline') / (sid + '.json')
                if not source.exists():
                    continue
                old = read(source)
                pipeline = {k: old[k] for k in ('global_result', 'old_crop_result', 'old_crop_transform')}
                global_path = raw_path(case)
                local_path = Path('runs/context_verifier_v9/validation_smoke/old_crop_features') / (sid + '.npz')
                pipeline.update(sample_id=case['sample_id'], ground_truth_in_model_inputs=False,
                    global_feature_path=str(global_path), global_feature_sha256=sha256(global_path),
                    local_feature_path=str(local_path) if pipeline['old_crop_result'] is not None else None,
                    local_feature_sha256=sha256(local_path) if pipeline['old_crop_result'] is not None else None)
                pipeline['pools'] = build_pools(pipeline)
            if not pipeline['pools']['32']:
                continue
            selected.append(case); pipelines.append(pipeline)
            (dest / 'vision_encodings').mkdir(parents=True, exist_ok=True)
            shutil.copyfile(encoding, dest / 'vision_encodings' / encoding.name)
            write_json(dest / 'pipeline' / (sid + '.json'), pipeline)
            break
        else:
            raise ValueError('No already available validation-only interface case for ' + split)
    cache.OUT = visual.OUT = protocol.OUT = training.OUT = dest
    cache.cases = visual.cases = lambda _: selected
    cache.root_for = visual.root_for = lambda _: dest
    for split in ('train', 'val'):
        cache.internal(split)
        visual.cache_visual_split(split, 'cpu')
    files = {}
    evidence = protocol.validate_training_inputs(selected, files)
    assert evidence['identities']['internal'] == training.input_identity(False)
    assert evidence['identities']['visual'] == training.input_identity(True)
    for variant in ('internal_full', 'internal_pooled', 'internal_global', 'visual'):
        use_visual = variant == 'visual'
        record = training.dataset('train', use_visual)[0]
        model = training.make_model(variant)
        indices = record['pool_indices']['16']
        output = model(**training.subset(record, indices, use_visual, 'cpu'))
        labels = torch.as_tensor(record['labels'][indices])
        loss = training.ranking_loss(output, labels)
        loss.backward()
        assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        print(dict(variant=variant, finite_loss=float(loss.detach())), flush=True)
    write_json(dest / 'result.json', dict(complete=True, old_train_validation_only=True,
        sample_ids=[c['sample_id'] for c in selected], checked_training_input_evidence=evidence,
        official_checkpoints_written=False))
