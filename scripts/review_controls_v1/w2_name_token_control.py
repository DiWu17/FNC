"""Retrospective W2 token-source control; no model forward or CUDA required.

Stages: register, smoke, convert-train, convert-eval, train, score.
The evaluation labels are deliberately never opened by this script.
"""
from __future__ import annotations

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '2'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]

import numpy as np
import torch
from clickagent.evidence_retention import RetentionModel, STATE_DIM, require, state_features
import run_quality_readout_v15 as base
import run_quality_readout_v15_3k as prior

torch.set_num_threads(2)
torch.set_num_interop_threads(1)
require(not torch.cuda.is_available(), 'This W2 control must remain CPU only')

OUT = ROOT / 'runs/review_controls_v1/w2'
V18 = ROOT / 'runs/full_scaling_v18'
V2 = ROOT / 'runs/paper_scale_ablation_v2'
MMB = ROOT / 'runs/mmbench_gui_new_source_v1'
PAPER_V1 = ROOT / 'runs/paper_scale_ablation_v1'
ARMS = ('last_prefill', 'last_name')
SEEDS = prior.SEEDS
STEPS = 13376
COHORTS = ('pro', 'sheets', 'mmbench')
SOURCES = {
    'old_train': V18 / 'data/old_train_index.json',
    'new_train': V18 / 'features/new_train/index.json',
    'pro': V2 / 'eval/pro/features/index.json',
    'sheets': V2 / 'eval/sheets/features/index.json',
    'mmbench': MMB / 'mmbench/features/index.json',
}


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def priority_low():
    if os.name == 'nt':
        import ctypes
        kernel = ctypes.windll.kernel32
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        kernel.SetPriorityClass.argtypes = (ctypes.c_void_p, ctypes.c_uint)
        kernel.SetPriorityClass.restype = ctypes.c_int
        require(bool(kernel.SetPriorityClass(kernel.GetCurrentProcess(), 0x00004000)), 'Could not set below-normal process priority')


def log(stage, **fields):
    value = dict(stage=stage, at_utc=now(), pid=os.getpid(), **fields)
    write(OUT / 'status.json', value)
    p = OUT / 'events.jsonl'
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('a', encoding='utf-8') as f:
        f.write(json.dumps(value, ensure_ascii=False) + '\n')
    print(json.dumps(value, ensure_ascii=False), flush=True)


def entries(cohort):
    v = read(SOURCES[cohort])
    e = v if isinstance(v, list) else v['cases']
    expected = 873 if cohort == 'old_train' else 12500 if cohort == 'new_train' else 1000 if cohort == 'mmbench' else 500
    require(len(e) == expected, f'{cohort} count changed: {len(e)}')
    return e


def register():
    inputs = {str(p): sha(p) for p in SOURCES.values()}
    for seed_ix in (1, 2, 3):
        for arm in ('full',):
            receipt = V18 / f'models/{arm}_s{seed_ix}.json'
            inputs[str(receipt)] = sha(receipt)
        receipt = PAPER_V1 / f'models/token_mean_s{seed_ix}.json'
        inputs[str(receipt)] = sha(receipt)
    source = {str(p): sha(p) for p in (
        Path(__file__), ROOT / 'src/clickagent/evidence_retention.py',
        ROOT / 'scripts/run_quality_readout_v15.py',
        ROOT / 'scripts/run_quality_readout_v15_3k.py',
        ROOT / 'scripts/run_evidence_retention_v14.py',
    )}
    plan = dict(
        purpose='W2 retrospective token-source control on frozen full-data ClickAgent pipeline',
        registered_at_utc=now(), source_sha256=source, input_sha256=inputs,
        training_queries=13373, steps=STEPS, batch_size=8, seeds=list(SEEDS),
        arms=list(ARMS), cohorts={k: (1000 if k == 'mmbench' else 500) for k in COHORTS},
        comparison=['last_prefill vs last_name', 'each single-token arm vs full aligned multi-token',
                    'each single-token arm vs full token_mean'],
        recipe={
            'last_prefill': 'Use cached last prefill query attention[96,P] and hidden[12288], before any generated ACTION/TARGET tokens.',
            'last_name': 'Use final cached TARGET-name token attention[96,P] and hidden[12288].',
            'feature': 'Re-run unchanged state_features over identical original candidate coordinates and global/local view transforms; 294 dimensions.',
            'mask': 'One token per view at candidate iff the original name-token mask had any valid token there; preserve original view availability.',
            'candidate': 'Unchanged original K16 baseline indices, base[32], candidate order, global/local views, and forced Top1.',
            'training': 'Unchanged 22,050-parameter RetentionModel, 13,376 steps, AdamW 2e-4 wd .02, clip1, listwise/null + .25 balanced BCE, three fixed seeds and same training labels.',
            'evaluation': 'Seal all three-seed unlabeled predictions on fixed Pro500, Sheets500, MMBench-GUI L2 1000 before any outcome analysis.',
        },
        limitations=[
            'Last prefill token is a fixed prompt context query and is not an aggregate of original instruction tokens; the latter were not cached.',
            'The upstream candidate proposer and crops remain generated-name dependent, so this isolates readout token-source effect conditional on the original pipeline only.',
            'Pro/Sheets and MMBench cohorts are retrospective comparisons with previously analyzed labels, not new blind tests.',
            'The prefill hidden state and generated-name hidden state occur at different autoregressive positions; position and semantics are not separately identified.',
        ],
        no_gpu=True, cpu_threads_max=2, process_priority='below_normal',
    )
    path = OUT / 'protocol.json'
    if path.exists():
        old = read(path)
        plan['registered_at_utc'] = old['registered_at_utc']
        require(plan == old, 'Registered W2 protocol drift')
    else:
        write(path, plan)
    log('registered', protocol_sha256=sha(path))


def verify_protocol():
    spec = read(OUT / 'protocol.json')
    require(spec['no_gpu'] and spec['arms'] == list(ARMS), 'Wrong protocol')
    for path, expected in spec['input_sha256'].items():
        require(sha(path) == expected, 'Registered input drift: ' + path)
    for path, expected in spec['source_sha256'].items():
        require(sha(path) == expected, 'Registered code drift: ' + path)


def raw_views(entry):
    require(sha(entry['pipeline_path']) == entry['pipeline_sha256'], 'Pipeline changed')
    pipe = read(entry['pipeline_path'])
    vs = [dict(feature_path=pipe['global_feature_path'], feature_sha256=pipe['global_feature_sha256'], transform=None)] + pipe['local_views']
    require(len(vs) <= 4, 'More than four views')
    return vs


def make_states(entry, check_target=False):
    require(sha(entry['path']) == entry['sha256'], 'Candidate base changed')
    require(sha(entry['mask_path']) == entry['mask_sha256'], 'Original mask changed')
    with np.load(entry['path'], allow_pickle=False) as z:
        points = z['points'].copy()
        baseline = z['baseline'].copy()
    old_mask = np.load(entry['mask_path'], allow_pickle=False)
    n = len(points)
    require(old_mask.shape[:2] == (n, 4), 'Original mask shape changed')
    present = old_mask.any(2)
    output = {arm: np.zeros((n, 4, 1, STATE_DIM), np.float16) for arm in ARMS}
    source_state = np.load(entry['state_path'], mmap_mode='r', allow_pickle=False) if check_target else None
    source_maxdiff = 0.0
    for view_ix, view in enumerate(raw_views(entry)):
        require(sha(view['feature_path']) == view['feature_sha256'], 'Raw NPZ changed')
        with np.load(view['feature_path'], allow_pickle=False) as z:
            metadata = json.loads(str(z['metadata']))
            image = z['image_hidden'].copy()
            target_a = z['target_attention'].copy()
            target_h = z['target_hidden'].copy()
            prefill_a = z['prefill_attention'].copy()
            prefill_h = z['prefill_hidden'].copy()
        image_size = metadata['image_size'] if view_ix == 0 else view['transform']['original_size']
        require(target_a.ndim == 3 and target_h.shape == (len(target_a), 12288), 'Name token raw dimensions')
        require(prefill_a.shape == target_a.shape[1:] and prefill_h.shape == (12288,), 'Prefill raw dimensions')
        for arm, attention, hidden in (
            ('last_prefill', prefill_a[None], prefill_h[None]),
            ('last_name', target_a[-1:], target_h[-1:]),
        ):
            if len(attention):
                feat, valid = state_features(dict(target_attention=attention, target_hidden=hidden,
                                                  image_hidden=image.copy()), metadata, points, image_size, view['transform'])
                require(np.array_equal(valid, present[:, view_ix]) or not present[:, view_ix].any(),
                        'View candidate validity changed')
                output[arm][:, view_ix, 0] = np.where(present[:, view_ix, None], feat[:, 0], 0)
            else:
                require(not present[:, view_ix].any(), 'Name token missing despite original valid mask')
        if check_target and len(target_a):
            feat, valid = state_features(dict(target_attention=target_a, target_hidden=target_h,
                                              image_hidden=image.copy()), metadata, points, image_size, view['transform'])
            require(np.array_equal(valid, present[:, view_ix]), 'Original view mask parity failed')
            original = np.asarray(source_state[:, view_ix, :len(target_a)], np.float16)
            require(original.shape == feat.shape, 'Original state shape parity failed')
            source_maxdiff = max(source_maxdiff, float(np.max(np.abs(original.astype(np.float32) - feat.astype(np.float32)), initial=0)))
    mask = present[:, :, None].copy()
    for arm in ARMS:
        require(output[arm].shape == (n, 4, 1, 294) and np.isfinite(output[arm]).all(), 'Invalid derived state')
        require(not np.any(output[arm][~mask]), 'Masked position contains evidence')
    require(len(np.unique(baseline)) == len(baseline), 'K16 baseline changed')
    return output, mask, source_maxdiff


def smoke():
    verify_protocol()
    priority_low()
    results = []
    for cohort in ('old_train', 'new_train', *COHORTS):
        entry = entries(cohort)[0]
        started = time.perf_counter()
        output, mask, diff = make_states(entry, check_target=True)
        require(diff == 0.0, 'Raw name-token reconstruction disagrees with frozen state')
        results.append(dict(cohort=cohort, sample_id=entry['sample_id'], candidates=len(mask),
                            valid_candidate_views=int(mask.sum()), max_source_state_difference=diff,
                            prefill_nonzero=int(np.count_nonzero(output['last_prefill'])),
                            name_nonzero=int(np.count_nonzero(output['last_name'])),
                            seconds=round(time.perf_counter() - started, 4)))
        log('smoke_progress', cohort=cohort, seconds=results[-1]['seconds'])
    write(OUT / 'smoke.json', dict(complete=True, samples=results, protocol_sha256=sha(OUT / 'protocol.json')))
    log('smoke_complete', samples=results)


def convert(cohort):
    verify_protocol()
    priority_low()
    records = entries(cohort)
    destination = OUT / 'features' / cohort
    dest_index = destination / 'index.json'
    if dest_index.exists():
        saved = read(dest_index)
        require(saved['complete'] and saved['source_index_sha256'] == sha(SOURCES[cohort]), 'Derived index drift')
        return
    derived = []
    started = time.perf_counter()
    for ix, entry in enumerate(records, 1):
        key = hashlib.sha256(entry['sample_id'].encode()).hexdigest()[:24]
        ep = destination / 'cases' / f'{key}.json'
        if ep.exists():
            item = read(ep)
            require(item['source_entry_sha256'] == base.digest(entry), 'Source entry changed')
            require(item['source_state_maxdiff'] == 0.0, 'Original name-state parity receipt missing')
            for arm in ARMS:
                require(sha(item['states'][arm]['path']) == item['states'][arm]['sha256'], 'Derived state changed')
        else:
            arrays, mask, parity = make_states(entry, check_target=True)
            require(parity == 0.0, 'Original name-token state differs from frozen source')
            mp = destination / 'arrays' / f'{key}.mask.npy'
            mp.parent.mkdir(parents=True, exist_ok=True)
            tmp = mp.with_suffix('.npy.tmp')
            with tmp.open('wb') as f: np.save(f, mask, allow_pickle=False)
            os.replace(tmp, mp)
            states = {}
            for arm in ARMS:
                sp = destination / 'arrays' / f'{key}.{arm}.state.npy'
                tmp = sp.with_suffix('.npy.tmp')
                with tmp.open('wb') as f: np.save(f, arrays[arm], allow_pickle=False)
                os.replace(tmp, sp)
                states[arm] = dict(path=str(sp), sha256=sha(sp))
            item = dict(sample_id=entry['sample_id'], source_entry_sha256=base.digest(entry),
                        source_path=entry['path'], source_sha256=entry['sha256'],
                        source_state_maxdiff=parity,
                        original_mask_sha256=entry['mask_sha256'],
                        mask_path=str(mp), mask_sha256=sha(mp), states=states)
            write(ep, item)
        derived.append(item)
        if ix % 25 == 0 or ix == len(records):
            log('converting', cohort=cohort, completed=ix, total=len(records),
                elapsed_seconds=round(time.perf_counter() - started, 1))
    write(dest_index, dict(complete=True, cohort=cohort, n=len(derived), cases=derived,
                           source_index_sha256=sha(SOURCES[cohort]), protocol_sha256=sha(OUT / 'protocol.json')))
    log('converted', cohort=cohort, count=len(records), index_sha256=sha(dest_index))


def records_for_arm(arm, cohort):
    source = entries(cohort)
    derived = read(OUT / 'features' / cohort / 'index.json')
    require(derived['complete'] and len(derived['cases']) == len(source), 'Derived index absent or partial')
    output = []
    for e, d in zip(source, derived['cases']):
        require(e['sample_id'] == d['sample_id'] and base.digest(e) == d['source_entry_sha256'], 'Derived order changed')
        r = dict(e)
        r['state_path'], r['state_sha256'] = d['states'][arm]['path'], d['states'][arm]['sha256']
        r['mask_path'], r['mask_sha256'] = d['mask_path'], d['mask_sha256']
        output.append(r)
    return output


@contextmanager
def training_adapter():
    original = prior.OUT, prior.status
    prior.OUT = OUT
    prior.status = lambda phase, **fields: log(phase, **fields)
    try:
        yield
    finally:
        prior.OUT, prior.status = original


def train(arm):
    verify_protocol()
    priority_low()
    require(arm in ARMS, 'Unregistered arm')
    source = records_for_arm(arm, 'old_train') + records_for_arm(arm, 'new_train')
    require(len(source) == 13373, 'Training data changed')
    records = base.load_records(source)
    with training_adapter():
        for seed in SEEDS:
            prior.train_one(arm, records, seed, STEPS, 'cpu')
    log('trained', arm=arm)


def score(cohort):
    verify_protocol()
    priority_low()
    require(cohort in COHORTS, 'Unknown evaluation cohort')
    for arm in ARMS:
        for seed_ix in (1, 2, 3):
            receipt = read(OUT / f'models/{arm}_s{seed_ix}.json')
            require(receipt['complete'] and receipt['steps'] == STEPS and
                    sha(receipt['checkpoint_path']) == receipt['checkpoint_sha256'], 'W2 model not sealed')
    rows = entries(cohort)
    derived = read(OUT / 'features' / cohort / 'index.json')
    require(derived['complete'] and len(derived['cases']) == len(rows), 'Evaluation features incomplete')
    models = {}
    for arm in ARMS:
        for seed_ix in (1, 2, 3):
            tag = f'{arm}_s{seed_ix}'
            receipt = read(OUT / f'models/{tag}.json')
            checkpoint = torch.load(receipt['checkpoint_path'], map_location='cpu', weights_only=True)
            model = RetentionModel()
            model.load_state_dict(checkpoint['state_dict'])
            model.eval()
            models[tag] = (model, checkpoint['mean'].numpy(), checkpoint['std'].numpy())
    started = time.perf_counter()
    hashes = {}
    for ix, (e, d) in enumerate(zip(rows, derived['cases']), 1):
        require(e['sample_id'] == d['sample_id'], 'Evaluation order changed')
        key = hashlib.sha256(e['sample_id'].encode()).hexdigest()[:24]
        dest = OUT / 'predictions' / cohort / 'cases' / f'{key}.json'
        if dest.exists():
            case = read(dest)
            require(case['complete'] and case['sample_id'] == e['sample_id'] and len(case['methods']) == 6,
                    'Existing prediction incomplete')
        else:
            with np.load(e['path'], allow_pickle=False) as z:
                points, candidate_base, active, baseline = (z[k].copy() for k in ('points', 'base', 'active', 'baseline'))
            mask = np.load(d['mask_path'], allow_pickle=False)
            states = {arm: np.load(d['states'][arm]['path'], allow_pickle=False) for arm in ARMS}
            methods = {}
            for tag, (model, mean, std) in models.items():
                arm = tag.rsplit('_s', 1)[0]
                inputs = dict(base=torch.as_tensor((candidate_base - mean) / std),
                              active=torch.as_tensor(active), state=torch.as_tensor(states[arm]),
                              mask=torch.as_tensor(mask))
                with torch.inference_mode():
                    scores = model(**inputs).numpy()
                require(np.isfinite(scores).all(), 'Nonfinite score')
                chosen = int(baseline[np.argmax(scores[baseline])]) if len(baseline) else None
                methods[tag] = dict(index=chosen, point=points[chosen].tolist() if chosen is not None else None)
            case = dict(complete=True, sample_id=e['sample_id'], source_entry_sha256=base.digest(e),
                        K16_count=len(baseline), methods=methods, labels_opened=False)
            write(dest, case)
        hashes[e['sample_id']] = sha(dest)
        if ix % 25 == 0 or ix == len(rows):
            log('scoring', cohort=cohort, completed=ix, total=len(rows),
                elapsed_seconds=round(time.perf_counter() - started, 1))
    write(OUT / 'predictions' / cohort / 'completion.json', dict(complete=True, cohort=cohort,
          n=len(rows), model_names=list(models), case_hashes=hashes, labels_opened=False,
          protocol_sha256=sha(OUT / 'protocol.json')))
    log('scored', cohort=cohort, n=len(rows))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('register', 'smoke', 'convert-train', 'convert-eval', 'train', 'score'), required=True)
    parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--cohort', choices=COHORTS)
    args = parser.parse_args()
    if args.stage == 'register': register()
    elif args.stage == 'smoke': smoke()
    elif args.stage == 'convert-train':
        for cohort in ('old_train', 'new_train'): convert(cohort)
    elif args.stage == 'convert-eval':
        for cohort in COHORTS: convert(cohort)
    elif args.stage == 'train':
        require(args.arm is not None, '--arm is required for training')
        train(args.arm)
    elif args.stage == 'score':
        if args.cohort is None:
            for cohort in COHORTS: score(cohort)
        else: score(args.cohort)


if __name__ == '__main__':
    main()
