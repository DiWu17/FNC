"""Resumable V12 orchestration; subprocess logs stay inside this run."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/crossfit_candidates_v12'


def run(label, args):
    log = OUT / 'logs' / (label + '.log'); log.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, '-X', 'utf8', '-u', *args]
    print(json.dumps(dict(start=label, command=command)), flush=True)
    with log.open('ab') as stream:
        process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        code = process.wait()
    if code:
        raise RuntimeError(f'{label} exit {code}: {log.read_text(encoding="utf-8", errors="replace")[-5000:]}')
    print(json.dumps(dict(complete=label)), flush=True)


def proposers():
    def worker(device, roles):
        for role in roles:
            run('proposer_' + role, ['scripts/train_crossfit_proposer.py',
                '--train-cases', str(OUT / 'data/train.jsonl'), '--val-cases', str(OUT / 'data/val.jsonl'),
                '--fold-map', str(OUT / 'folds.json'), '--role', role, '--output', str(OUT), '--device', device])
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(worker, 'cuda:0', ['fold_0', 'fold_2', 'fold_4']),
                pool.submit(worker, 'cuda:1', ['fold_1', 'fold_3', 'full'])]
        for job in jobs:
            job.result()


def features():
    run('runtime_identity', ['scripts/verify_crossfit_runtime.py'])
    for group in ('in_sample', 'oof', 'val'):
        for stage in ('global', 'local', 'cache'):
            run(group + '_' + stage, ['scripts/extract_crossfit_candidates.py', '--stage', stage, '--group', group])


def scorers():
    def worker(device, suffix):
        for arm in ('in_sample', 'oof'):
            run(arm + '_' + suffix, ['scripts/train_crossfit_scorer.py', '--arm', arm, '--seed', suffix, '--device', device])
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(worker, 'cuda:0', 's1'), pool.submit(worker, 'cuda:1', 's2')]
        for job in jobs:
            job.result()


def test():
    # Sealed runs retain the original pretest proof and training inputs.
    if not (OUT / 'frozen.json').exists():
        run('training_pool_shift', ['scripts/analyze_crossfit_training.py'])
        run('pretest_audit', ['scripts/audit_crossfit_candidates.py', '--stage', 'pretest'])
        run('freeze', ['scripts/crossfit_protocol.py', 'freeze'])
    run('test_features', ['scripts/extract_crossfit_candidates.py', '--stage', 'local', '--group', 'test'])
    run('test_cache', ['scripts/extract_crossfit_candidates.py', '--stage', 'cache', '--group', 'test'])
    for split in ('val', 'test'):
        run(split + '_heads', ['scripts/run_crossfit_scorer.py', '--stage', 'heads', '--split', split])
    run('test_gui', ['scripts/run_crossfit_scorer.py', '--stage', 'reference', '--split', 'test'])
    for split in ('val', 'test'):
        run(split + '_export', ['scripts/run_crossfit_scorer.py', '--stage', 'export', '--split', split])
        run(split + '_evaluate', ['scripts/evaluate_crossfit_candidates.py',
            '--cases', str(OUT / 'data' / (split + '.jsonl')),
            '--predictions', str(OUT / 'evaluation' / split / 'predictions.jsonl'),
            '--pools', str(OUT / 'evaluation' / split / 'pools.jsonl'),
            '--out', str(OUT / 'evaluation' / split)])
    run('serialization_diagnostic', ['scripts/diagnose_crossfit_serialization.py'])
    if not (OUT / 'checks/audit_thread_replay_diagnostic.json').exists():
        run('audit_thread_probe', ['scripts/probe_crossfit_audit_threads.py'])
    run('final_audit', ['scripts/audit_crossfit_serialization.py'])


def remaining():
    roles = ['fold_' + str(i) for i in range(5)] + ['full']
    while True:
        complete = []
        for role in roles:
            path = OUT / 'proposers' / (role + '.json')
            if path.exists() and json.loads(path.read_text(encoding='utf-8')).get('complete'):
                complete.append(role)
        if len(complete) == len(roles):
            break
        time.sleep(10)
    features(); scorers(); test()


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--stage', choices=['proposers', 'features', 'scorers', 'test', 'all', 'remaining'], required=True)
    args = p.parse_args()
    if args.stage == 'remaining':
        remaining(); raise SystemExit(0)
    for name, func in [('proposers', proposers), ('features', features), ('scorers', scorers), ('test', test)]:
        if args.stage in (name, 'all'):
            func()
