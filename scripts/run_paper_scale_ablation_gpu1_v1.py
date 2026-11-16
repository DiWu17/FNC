"""GPU1 worker for the last six full-data ablation fits.

This is orchestration only: the sealed trainer, feature transform, optimizer and
training protocol are imported unchanged. Its per-worker status is isolated
from the existing GPU0 run. It stops *before* GPU0 reaches an overlapping arm.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

import torch
import run_paper_scale_ablation_train_v1 as paper
import run_quality_readout_v15_3k as prior

WORKER = paper.OUT / 'gpu1_worker'
ARMS = ('hidden_only', 'token_mean')
ORDER = tuple((variant, seed_index) for variant in paper.ABLATIONS for seed_index in (1, 2, 3))
RANK = {f'{variant}_s{seed_index}': position for position, (variant, seed_index) in enumerate(ORDER, 1)}
STOP_BEFORE = {'hidden_only': 8, 'token_mean': 11}


def worker_status(phase, **extra):
    value = dict(phase=phase, pid=os.getpid(), device='cuda:1',
                 updated_at_utc=datetime.now(timezone.utc).isoformat(), **extra)
    paper.write(WORKER / 'status.json', value)
    log = WORKER / 'logs/events.jsonl'
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + '\n')
    print(json.dumps(value, ensure_ascii=False), flush=True)


def gpu0_position():
    status_file = paper.OUT / 'status.json'
    if not status_file.exists():
        raise RuntimeError('GPU0 main status absent; verify the primary run first')
    state = paper.read(status_file)
    phase = state.get('phase')
    if phase in ('models_frozen', 'complete'):
        return len(ORDER), state
    if phase == 'model_complete':
        return int(state.get('models_completed', 0)), state
    if phase == 'training':
        return RANK.get(state.get('model', ''), 0), state
    if phase == 'loading_train':
        return 0, state
    if phase in ('failed', 'stopped'):
        raise RuntimeError('GPU0 reports ' + str(phase) + '; inspect primary run before GPU1 training')
    return 0, state


def guard_overlap(variant, seed_index):
    tag = f'{variant}_s{seed_index}'
    progress, state = gpu0_position()
    require_before = STOP_BEFORE[variant]
    if progress >= require_before:
        raise RuntimeError(f'GPU0 is too close to {tag}: position {progress} >= safety barrier {require_before}; '
                           f'GPU0 phase={state.get("phase")}, model={state.get("model")}. '
                           'GPU1 stopped before any shared-model write; use the primary process to finish.')
    # GPU0 never writes worker files. Existing target artifacts are acceptable
    # only when they are a complete prior fit or a resume from this worker.
    receipt = paper.OUT / f'models/{tag}.json'
    if receipt.exists():
        row = paper.read(receipt)
        if not row.get('complete'):
            raise RuntimeError('Incomplete target receipt: ' + str(receipt))
        if paper.sha(row['checkpoint_path']) != row['checkpoint_sha256']:
            raise RuntimeError('Target checkpoint identity drift: ' + tag)
    return state


@contextmanager
def isolated_adapters(variant, seed_index):
    saved = prior.OUT, prior.status, prior.tensors
    prior.OUT = paper.OUT

    def monitored_status(phase, **extra):
        # train_one calls this before the first step and at each 250-step
        # checkpoint. Aborting here leaves its .resume.pt intact.
        guard_overlap(variant, seed_index)
        worker_status(phase, worker_model=f'{variant}_s{seed_index}', **extra)

    prior.status = monitored_status
    prior.tensors = lambda records, mean, std, ignored, device: paper.build_inputs(
        records, mean, std, variant, device)
    try:
        yield
    finally:
        prior.OUT, prior.status, prior.tensors = saved


@contextmanager
def exclusive_worker():
    WORKER.mkdir(parents=True, exist_ok=True)
    lock = WORKER / 'active.lock'
    try:
        descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError as exc:
        raise RuntimeError('Another GPU1 worker or stale lock exists: ' + str(lock)) from exc
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
            stream.write(json.dumps(dict(pid=os.getpid(), started_at_utc=datetime.now(timezone.utc).isoformat(),
                                         wrapper_sha256=paper.sha(__file__)), ensure_ascii=False) + '\n')
        yield
    finally:
        # Normal exits, including a clean barrier abort, release the lock.
        # A hard kill leaves it as an explicit signal to inspect before resume.
        lock.unlink(missing_ok=True)


def check(*, device='cuda:1'):
    if device != 'cuda:1':
        raise ValueError('GPU1 worker is fixed to cuda:1')
    paper.register()  # verifies unchanged sealed trainer/protocol/input SHA
    if torch.cuda.device_count() < 2:
        raise RuntimeError('Two CUDA devices are not visible')
    if (WORKER / 'active.lock').exists():
        raise RuntimeError('GPU1 worker lock already exists; inspect before restarting')
    progress, primary = gpu0_position()
    for variant in ARMS:
        if progress >= STOP_BEFORE[variant]:
            raise RuntimeError(f'GPU0 already reached safety barrier for {variant}; skip parallelization')
    for variant in ARMS:
        for seed_index in (1, 2, 3):
            guard_overlap(variant, seed_index)
    result = dict(stage='check', device=device, gpu_name=torch.cuda.get_device_name(1),
                  gpu0_phase=primary.get('phase'), gpu0_model=primary.get('model'),
                  gpu0_position=progress, worker_arms=list(ARMS), worker_fits=6,
                  training_queries=paper.TRAIN_QUERIES, steps=paper.STEPS,
                  paper_protocol_sha256=paper.sha(paper.OUT / 'protocol.json'),
                  trainer_sha256=paper.sha(paper.ROOT / 'scripts/run_paper_scale_ablation_train_v1.py'),
                  wrapper_sha256=paper.sha(__file__),
                  status_path=str(WORKER / 'status.json'), model_output_path=str(paper.OUT / 'models'))
    paper.write(WORKER / 'check.json', result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


def train():
    check()
    with exclusive_worker():
        torch.cuda.set_device(1)
        entries = paper.records_index()
        worker_status('loading_train', queries=len(entries), worker_arms=list(ARMS))
        records = prior.base.load_records(entries)
        if len(records) != paper.TRAIN_QUERIES:
            raise RuntimeError('GPU1 loaded cohort count changed')
        worker_status('loaded_train', queries=len(records))
        for variant in ARMS:
            for seed_index, seed in enumerate(paper.SEEDS, 1):
                guard_overlap(variant, seed_index)
                with isolated_adapters(variant, seed_index):
                    prior.train_one(variant, records, seed, paper.STEPS, 'cuda:1')
                receipt = paper.seal_variant(variant, seed_index)
                if receipt['queries'] != paper.TRAIN_QUERIES or receipt['steps'] != paper.STEPS:
                    raise RuntimeError('GPU1 trained cohort/schedule changed')
                worker_status('model_complete', worker_model=f'{variant}_s{seed_index}',
                              worker_models_completed=ARMS.index(variant) * 3 + seed_index,
                              worker_models_total=6)
        worker_status('worker_complete', worker_models_completed=6,
                      note='Primary GPU0 process may skip these sealed receipts and freeze all 18 models.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('check', 'train'), default='check')
    args = parser.parse_args()
    os.chdir(ROOT)
    torch.set_num_threads(4)
    if args.stage == 'check':
        check()
    else:
        try:
            train()
        except BaseException as exc:
            worker_status('worker_stopped', error=repr(exc),
                          resume_command='.\\.venv\\python.exe -B -X utf8 -u '
                                         'scripts/run_paper_scale_ablation_gpu1_v1.py --stage train')
            raise


if __name__ == '__main__':
    main()
