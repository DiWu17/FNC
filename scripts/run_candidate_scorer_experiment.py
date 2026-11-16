"""Resumable V11 orchestration with hidden Windows helpers and explicit GPU ownership."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/candidate_scorer_v11'
CHILDREN = []


def stage(name, status='running'):
    (OUT / 'experiment_progress.json').write_text(json.dumps(dict(stage=name, status=status,
        updated_at=time.time(), pid=os.getpid()), indent=2), encoding='utf-8')


def start(name, *args):
    streams = [(OUT / (name + suffix)).open('wb') for suffix in ('.log', '.err')]
    child = subprocess.Popen([sys.executable, '-X', 'utf8', '-u', *args], cwd=ROOT, stdin=subprocess.DEVNULL,
        stdout=streams[0], stderr=streams[1], creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    CHILDREN.append((name, child, streams))
    with (OUT / 'processes.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(dict(name=name, pid=child.pid, args=args, started_at=time.time())) + '\n')
    return child


def finish(child):
    code = child.wait()
    if code:
        raise RuntimeError(f'Child {child.pid} failed with exit {code}; see processes.jsonl and its named log/err')


def run(name, *args):
    stage(name); finish(start(name, *args))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if not (OUT / 'data_selection.json').exists():
        run('prepare', 'scripts/prepare_candidate_scorer.py')
    if not (OUT / 'protocol.json').exists():
        run('protocol', 'scripts/candidate_scorer_protocol.py', '--stage', 'init')
    if not (OUT / 'frozen.json').exists():
        run('local_trainval', 'scripts/extract_candidate_scorer.py', '--stage', 'local_trainval')
        for split in ('train', 'val'):
            run('cache_internal_' + split, 'scripts/cache_candidate_scorer.py', '--stage', 'internal', '--split', split)
        stage('training_and_extra_visual_cache')
        s1 = start('train_s1', 'scripts/train_candidate_scorer_worker.py', '--seed', 's1', '--device', 'cuda:0')
        s2 = start('visual_cache_and_train_s2', 'scripts/cache_visual_trainval.py')
        finish(s2); finish(s1)
        run('freeze', 'scripts/candidate_scorer_protocol.py', '--stage', 'freeze')
    run('fresh_test_features', 'scripts/extract_candidate_scorer.py', '--stage', 'test')
    run('cache_internal_test', 'scripts/cache_candidate_scorer.py', '--stage', 'internal', '--split', 'test')
    stage('fresh_test_visual_and_reference')
    visual = start('cache_visual_test', 'scripts/cache_candidate_scorer.py', '--stage', 'visual', '--split', 'test', '--device', 'cuda:1')
    reference = start('reference_test', 'scripts/run_candidate_scorer.py', '--stage', 'reference', '--split', 'test')
    finish(visual); finish(reference)
    stage('fresh_test_heads_and_ui_tars')
    heads = start('heads_test', 'scripts/run_candidate_scorer.py', '--stage', 'heads', '--split', 'test')
    uitars = start('ui_tars_test', 'scripts/run_ui_tars_baseline.py', '--output', 'runs/candidate_scorer_v11')
    finish(heads); finish(uitars)
    run('diagnostic_pools', 'scripts/extract_candidate_scorer.py', '--stage', 'diagnostic')
    for kind in ('internal', 'visual'):
        run('cache_' + kind + '_diagnostic', 'scripts/cache_candidate_scorer.py', '--stage', kind, '--split', 'diagnostic', '--device', 'cuda:1')
    for kind in ('heads', 'reference'):
        run(kind + '_diagnostic', 'scripts/run_candidate_scorer.py', '--stage', kind, '--split', 'diagnostic')
    run('evaluate', 'scripts/evaluate_candidate_scorer.py')
    run('audit', 'scripts/audit_candidate_scorer.py')
    run('report', 'scripts/report_candidate_scorer.py')
    stage('complete', 'complete')


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        for _, child, _ in CHILDREN:
            if child.poll() is None:
                child.terminate()
        stage('failed', 'failed')
        (OUT / 'experiment_failure.txt').write_text(repr(error), encoding='utf-8')
        raise
    finally:
        for _, _, streams in CHILDREN:
            for stream in streams:
                stream.close()
