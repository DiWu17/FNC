"""Train full-data readout ablations without changing sealed V16/V18 artifacts.

The V18 full readout is the aligned reference. Each ablated arm uses exactly the
same 13,373 cached queries, 13,376 optimizer steps, three seeds, architecture,
loss and base-feature scaler; only the registered state-input transform changes.
Evaluation must apply each checkpoint's feature_ablation transform as well.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

import numpy as np
import torch

import run_quality_readout_v15_3k as prior
from clickagent.contribution_ablation import VARIANTS, transform_inputs
from clickagent.evidence_retention import RetentionModel, require

OUT = ROOT / 'runs/paper_scale_ablation_v1'
V18 = ROOT / 'runs/full_scaling_v18'
V16 = ROOT / 'runs/contribution_v16'
MAIN = ROOT / 'runs/quality_readout_v15_3k'
SEEDS = (20261311, 20261312, 20261313)
STEPS = 13376
TRAIN_QUERIES = 13373
ABLATIONS = tuple(v for v in VARIANTS if v != 'aligned')
ORIGINAL_TENSORS = prior.tensors


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def status(phase, *, run_dir=OUT, **extra):
    value = dict(phase=phase, pid=os.getpid(), updated_at_utc=datetime.now(timezone.utc).isoformat(), **extra)
    write(run_dir / 'status.json', value)
    events = run_dir / 'logs/events.jsonl'
    events.parent.mkdir(parents=True, exist_ok=True)
    with events.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + '\n')
    print(json.dumps(value, ensure_ascii=False), flush=True)


def source_identity():
    v16 = read(V16 / 'protocol.json')['source_identity']
    v18 = read(V18 / 'protocol.json')['hashes']
    expected = {
        'src/clickagent/contribution_ablation.py': v16['src/clickagent/contribution_ablation.py'],
        'scripts/run_quality_readout_v15_3k.py': v18[str(ROOT / 'scripts/run_quality_readout_v15_3k.py')],
        'scripts/run_quality_readout_v15.py': v18[str(ROOT / 'scripts/run_quality_readout_v15.py')],
        'src/clickagent/evidence_retention.py': v18[str(ROOT / 'src/clickagent/evidence_retention.py')],
    }
    for name, digest in expected.items():
        require(sha(ROOT / name) == digest, 'Sealed scientific source drift: ' + name)
    return dict(expected, **{'scripts/run_paper_scale_ablation_train_v1.py': sha(__file__)})


def input_identity():
    completion = read(V18 / 'completion.json')
    require(completion['complete'], 'V18 is not complete')
    paths = [
        ROOT / 'research/paper/method_freeze_v1.json',
        V18 / 'protocol.json',
        V18 / 'models/frozen.json',
        V18 / 'features/new_train/index.json',
        V18 / 'data/old_train_index.json',
        MAIN / 'data/old_train_index.json',
        V16 / 'protocol.json',
        V16 / 'models/frozen.json',
    ]
    for path in paths[1:4]:
        require(sha(path) == completion['artifact_sha256'][str(path)], 'V18 artifact drift: ' + str(path))
    frozen = read(V18 / 'models/frozen.json')
    require(frozen['complete'], 'V18 checkpoints are not frozen')
    for seed_index in (1, 2, 3):
        entry = frozen['models'][f'full_s{seed_index}']
        require(entry['queries'] == TRAIN_QUERIES and entry['steps'] == STEPS, 'V18 reference schedule changed')
        require(sha(entry['path']) == entry['sha256'], 'V18 full checkpoint drift')
        paths.append(Path(entry['path']))
    return {str(path): sha(path) for path in paths}


def register():
    sources = source_identity()
    inputs = input_identity()
    plan = dict(
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        purpose='Full-data feature ablations paired to sealed V18 aligned readout; original V15_3k/V16 and V18 remain read-only.',
        source_identity=sources,
        input_sha256=inputs,
        training_queries=TRAIN_QUERIES,
        steps=STEPS,
        seeds=list(SEEDS),
        ablations=list(ABLATIONS),
        new_fits=len(ABLATIONS) * len(SEEDS),
        full_reference='runs/full_scaling_v18/models/full_s1..s3.pt',
        matched3k_control='runs/full_scaling_v18/models/matched3k_s1..s3.pt: equal 13,376-step control for the effect of training-set size; not an ablation arm in this run.',
        feature_recipe='Unmodified V15_3k aligned tensors, then V16 transform_inputs on state only; keep base, active, mask and candidate order.',
        optimizer='Unmodified V15_3k train_one: batch8, AdamW lr2e-4 wd0.02, clip1, listwise/null +0.25 balanced BCE, fixed final step.',
        evaluation_rule='Use the same variant transform at evaluation; original K16 Top1 with stable canonical union-index tie order is the cross-scale primary metric.',
        evaluation_order=['development and already-used V16 300-case diagnostics',
                          'one-time 58 untouched ScreenSpot-Pro reserve confirmation after all checkpoints are frozen',
                          'full ScreenSpot-Pro benchmark descriptive/comparative evaluation, with historical exposure marked'],
        interpretation='This ablates the readout conditional on the unchanged V12 proposer, which itself uses internal evidence. Dataset scale also changes source mixture; full vs 3k is not quantity-only causality.',
        controls='No checkpoint/seed selection on evaluation. Report all three fixed seeds and their mean. No test-time-only masking presented as trained ablation.',
        output_scope=str(OUT),
        no_html=True,
    )
    path = OUT / 'protocol.json'
    if path.exists():
        existing = read(path)
        plan['created_at_utc'] = existing['created_at_utc']
        require(existing == plan, 'Registered full-data ablation protocol drift')
    else:
        write(path, plan)
        status('registered', variants=len(ABLATIONS), seeds=len(SEEDS), new_fits=len(ABLATIONS) * len(SEEDS))
    return plan


def records_index():
    entries = read(V18 / 'data/old_train_index.json') + read(V18 / 'features/new_train/index.json')['cases']
    require(len(entries) == TRAIN_QUERIES, 'Training cohort changed')
    return entries


def build_inputs(records, mean, std, variant, device):
    args, lengths = ORIGINAL_TENSORS(records, mean, std, 'aligned', device)
    transformed = transform_inputs(args, lengths, [r['sample_id'] for r in records], variant)
    return transformed, lengths


@contextmanager
def adapters(variant, run_dir=OUT):
    saved = prior.OUT, prior.status, prior.tensors
    prior.OUT = run_dir
    prior.status = lambda phase, **extra: status(phase, run_dir=run_dir, **extra)
    prior.tensors = lambda records, mean, std, ignored, device: build_inputs(records, mean, std, variant, device)
    try:
        yield
    finally:
        prior.OUT, prior.status, prior.tensors = saved


def model_state_digest(weights):
    return prior.digest({name: hashlib.sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest()
                         for name, tensor in weights.items()})


def seal_variant(variant, seed_index, *, run_dir=OUT, reference=V18):
    tag = f'{variant}_s{seed_index}'
    receipt_path = run_dir / f'models/{tag}.json'
    receipt = read(receipt_path)
    require(receipt['complete'] and receipt['name'] == tag, 'Incomplete ablation receipt')
    expected_steps = STEPS if run_dir == OUT else 2
    require(receipt['steps'] == expected_steps and receipt['parameter_count'] == 22050, 'Ablation schedule or architecture changed')
    aligned = read(reference / f'models/full_s{seed_index}.json')
    require(receipt['initial_state_sha256'] == aligned['initial_state_sha256'], 'Seed initialization differs from V18 aligned')
    checkpoint_path = Path(receipt['checkpoint_path'])
    require(sha(checkpoint_path) == receipt['checkpoint_sha256'], 'Ablation checkpoint bytes changed')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    full_checkpoint = torch.load(aligned['checkpoint_path'], map_location='cpu', weights_only=True)
    if run_dir == OUT:
        require(torch.equal(checkpoint['mean'], full_checkpoint['mean']) and
                torch.equal(checkpoint['std'], full_checkpoint['std']),
                'Ablation scaler differs from aligned reference')
    before = model_state_digest(checkpoint['state_dict'])
    if receipt.get('feature_ablation') != variant:
        checkpoint.update(variant=variant, feature_ablation=variant)
        temporary = checkpoint_path.with_suffix('.pt.recipe.tmp')
        torch.save(checkpoint, temporary)
        os.replace(temporary, checkpoint_path)
        reread = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        require(model_state_digest(reread['state_dict']) == before, 'Metadata update changed model tensors')
        receipt.update(feature_ablation=variant, model_state_sha256=before, checkpoint_sha256=sha(checkpoint_path))
        write(receipt_path, receipt)
    else:
        require(receipt['model_state_sha256'] == before and sha(checkpoint_path) == receipt['checkpoint_sha256'],
                'Already sealed ablation checkpoint drift')
    return receipt


def train_all(device):
    register()
    entries = records_index()
    status('loading_train', queries=len(entries))
    records = prior.base.load_records(entries)
    require(len(records) == TRAIN_QUERIES, 'Loaded cohort changed')
    frozen = read(V18 / 'models/frozen.json')['models']
    for variant in ABLATIONS:
        for seed_index, seed in enumerate(SEEDS, 1):
            require(sha(frozen[f'full_s{seed_index}']['path']) == frozen[f'full_s{seed_index}']['sha256'],
                    'Aligned V18 checkpoint changed')
            with adapters(variant):
                prior.train_one(variant, records, seed, STEPS, device)
            receipt = seal_variant(variant, seed_index)
            require(receipt['queries'] == TRAIN_QUERIES and receipt['fitting_queries'] == 12886 and
                    receipt['empty_skipped'] == 487, 'Training cohort or skip count changed')
            status('model_complete', model=f'{variant}_s{seed_index}', models_completed=
                   (ABLATIONS.index(variant) * len(SEEDS) + seed_index), models_total=len(ABLATIONS) * len(SEEDS))
    del records
    gc.collect()
    if device.startswith('cuda') and torch.cuda.is_available():
        torch.cuda.empty_cache()


def freeze():
    plan = register()
    references = read(V18 / 'models/frozen.json')['models']
    models = {}
    for variant in VARIANTS:
        for seed_index, seed in enumerate(SEEDS, 1):
            name = f'full_{variant}_s{seed_index}'
            if variant == 'aligned':
                source = references[f'full_s{seed_index}']
                info = dict(path=source['path'], sha256=source['sha256'], variant=variant,
                            seed=seed, queries=TRAIN_QUERIES, steps=STEPS, reused_v18=True)
            else:
                receipt = seal_variant(variant, seed_index)
                info = dict(path=receipt['checkpoint_path'], sha256=receipt['checkpoint_sha256'],
                            variant=variant, seed=seed, queries=receipt['queries'], steps=receipt['steps'])
            require(sha(info['path']) == info['sha256'], 'Frozen model byte drift')
            models[name] = info
    value = dict(complete=True, models=models, protocol_sha256=sha(OUT / 'protocol.json'),
                 source_identity=plan['source_identity'], training_queries=TRAIN_QUERIES,
                 new_fits=len(ABLATIONS) * len(SEEDS), independent_predictions_at_freeze=0)
    path = OUT / 'models/frozen.json'
    if path.exists():
        require(read(path) == value, 'Full-data ablation model freeze drift')
    else:
        write(path, value)
    status('models_frozen', models=len(models), new_fits=value['new_fits'], evaluation_complete=False)


def smoke():
    """Exercise transformed 2-step CPU fitting in a dedicated, non-paper directory."""
    plan = register()
    smoke_dir = OUT / 'smoke'
    smoke_protocol = smoke_dir / 'protocol.json'
    sample_entries = [entry for entry in records_index() if entry['n'] > 0][:16]
    require(len(sample_entries) == 16, 'Insufficient nonempty smoke rows')
    smoke_spec = dict(paper_protocol_sha256=sha(OUT / 'protocol.json'),
                      case_refs=[dict(sample_id=e['sample_id'], sha256=e['sha256']) for e in sample_entries],
                      steps=2, device='cpu', variants=['no_evidence', 'spatial_shuffle'])
    if smoke_protocol.exists():
        require(read(smoke_protocol) == smoke_spec, 'Smoke protocol drift')
    else:
        write(smoke_protocol, smoke_spec)
    records = prior.base.load_records(sample_entries)
    for variant in smoke_spec['variants']:
        with adapters(variant, smoke_dir):
            prior.train_one('smoke_' + variant, records, SEEDS[0], 2, 'cpu')
        receipt = read(smoke_dir / f'models/smoke_{variant}_s1.json')
        require(receipt['complete'] and receipt['queries'] == 16 and receipt['steps'] == 2,
                'CPU smoke training incomplete')
        checkpoint = torch.load(receipt['checkpoint_path'], map_location='cpu', weights_only=True)
        model = RetentionModel().cpu().eval()
        model.load_state_dict(checkpoint['state_dict'])
        with torch.inference_mode():
            args, lengths = build_inputs(records[:1], checkpoint['mean'].numpy(),
                                          checkpoint['std'].numpy(), variant, 'cpu')
            scores = model(**args)
            require(scores.shape == (lengths[0],) and bool(torch.isfinite(scores).all()),
                    'CPU smoke evaluation failed')
    status('smoke_complete', run_dir=smoke_dir, variants=smoke_spec['variants'],
           steps=2, records=16, paper_protocol_sha256=sha(OUT / 'protocol.json'))
    status('smoke_verified', variants=smoke_spec['variants'], full_training_started=False)
    require(plan['source_identity'] == source_identity(), 'Source changed during smoke')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('register', 'smoke', 'train', 'freeze', 'all'), default='register')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    os.chdir(ROOT)
    torch.set_num_threads(4)
    if args.stage == 'register':
        register()
    elif args.stage == 'smoke':
        smoke()
    elif args.stage == 'train':
        train_all(args.device)
    elif args.stage == 'freeze':
        freeze()
    else:
        train_all(args.device)
        freeze()


if __name__ == '__main__':
    main()
