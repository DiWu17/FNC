"""Resumable V13 orchestration. Logs persist, no prior source/artifact edits."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'runs/multiregion_v13'


def run_job(name, arguments):
    log = OUT/'logs'/(name+'.log'); log.parent.mkdir(parents=True, exist_ok=True)
    stream = log.open('a', encoding='utf-8')
    process = subprocess.Popen([sys.executable, '-X', 'utf8', '-u']+arguments, cwd=ROOT,
        stdout=stream, stderr=subprocess.STDOUT, env=dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4'))
    return process, stream, log


def run(jobs):
    handles = [run_job(name, args) for name, args in jobs]
    print(json.dumps(dict(stage='started', jobs=[dict(pid=h[0].pid, log=str(h[2])) for h in handles])), flush=True)
    latest = None
    try:
        while True:
            statuses = [h[0].poll() for h in handles]
            if any(status is not None and status != 0 for status in statuses):
                for h in handles:
                    if h[0].poll() is None:
                        h[0].terminate()
                raise RuntimeError('Stage failed: '+str([(str(h[2]), h[0].poll()) for h in handles]))
            if all(status is not None for status in statuses):
                break
            state = []
            for process, _, path in handles:
                with path.open(encoding='utf-8', errors='replace') as stream:
                    lines = stream.readlines()
                tail = lines[-1].strip()[-800:] if lines else ''
                state.append(dict(log=path.name, status=process.poll(), tail=tail))
            value = json.dumps(state, ensure_ascii=False)
            if value != latest:
                print(value, flush=True); latest = value
            time.sleep(15)
    finally:
        for _, stream, _ in handles:
            stream.close()


def one(name, script, *args):
    run([(name, ['scripts/'+script]+list(args))])


def main(stage, wait_trainval=False):
    if wait_trainval and stage in ('train', 'all'):
        latest = None
        while True:
            ready = []
            counts = {}
            for split, n in [('train', 873), ('val', 172)]:
                p = OUT/'pipeline'/(split+'_index.json')
                ready.append(p.exists() and json.loads(p.read_text(encoding='utf-8'))['complete'])
                counts[split] = len(list((OUT/'pipeline'/split).glob('*.json')))
            if all(ready):
                break
            coarse = tuple(n//25 for n in counts.values())
            if coarse != latest:
                print(json.dumps(dict(stage='wait_trainval', completed=counts)), flush=True)
                latest = coarse
            time.sleep(15)
    if stage in ('train', 'all'):
        assert not (OUT/'frozen.json').exists()
        one('cache_train', 'cache_multiregion_v13.py', '--split', 'train')
        one('cache_val', 'cache_multiregion_v13.py', '--split', 'val')
        run([(f'full_{seed}', ['scripts/train_multiregion_v13.py', '--stage', 'full', '--seed', seed, '--device', device])
             for seed, device in [('s1', 'cuda:0'), ('s2', 'cuda:1')]])
        run([(f'cv_{objective}', ['scripts/train_multiregion_v13.py', '--stage', 'cv', '--objective', objective, '--device', device])
             for objective, device in [('top1', 'cuda:0'), ('top4', 'cuda:1')]])
        one('select', 'train_multiregion_v13.py', '--stage', 'select')
        one('freeze', 'multiregion_v13_protocol.py', 'freeze')
    if stage in ('test', 'all'):
        for split in ('test', 'supplement'):
            one('extract_'+split, 'extract_multiregion_v13.py', '--stage', 'extract', '--split', split)
            one('cache_'+split, 'cache_multiregion_v13.py', '--split', split)
            # GPU1 readout and GPU0 verifier do not overlap the two-GPU backbone.
            one('heads_'+split, 'run_multiregion_v13.py', '--stage', 'heads', '--split', split, '--device', 'cuda:1')
            one('gui_'+split, 'run_multiregion_v13.py', '--stage', 'reference', '--split', split)
        one('evaluate', 'evaluate_multiregion_v13.py')
    print(json.dumps(dict(complete=True, stage=stage)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--stage', choices=('train', 'test', 'all'), required=True)
    parser.add_argument('--wait-trainval', action='store_true')
    args = parser.parse_args(); main(args.stage, args.wait_trainval)
