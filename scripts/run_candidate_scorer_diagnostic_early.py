"""Run the old, secondary diagnostic on CPU after all V11 models are frozen.

This can overlap the master job's fresh GPU feature extraction. It never fits
or selects a model, touches new-test labels, or changes the primary pipeline.
"""
import json
from pathlib import Path
import subprocess
import sys
import time

from clickagent.action_benchmark import write_json


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/candidate_scorer_v11'


def status(stage, **values):
    write_json(OUT / 'early_diagnostic_progress.json', dict(stage=stage, **values))


def run(name, *args):
    status(name, state='running')
    with (OUT / ('early_' + name + '.log')).open('wb') as stdout, (OUT / ('early_' + name + '.err')).open('wb') as stderr:
        subprocess.run([sys.executable, '-u', *args], cwd=ROOT, stdin=subprocess.DEVNULL,
            stdout=stdout, stderr=stderr, check=True,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))


if __name__ == '__main__':
    try:
        status('waiting_for_all_models_freeze', state='waiting')
        while not (OUT / 'frozen.json').exists():
            progress = OUT / 'experiment_progress.json'
            if progress.exists() and json.loads(progress.read_text(encoding='utf-8')).get('status') == 'failed':
                raise RuntimeError('Master experiment failed before freeze')
            time.sleep(10)
        frozen = json.loads((OUT / 'frozen.json').read_text(encoding='utf-8'))
        assert frozen['complete'] and frozen['fresh_test_features_at_freeze'] == 0
        # The image-only diagnostic worker owns these files and uses no labels.
        while not (OUT / 'visual_diagnostic_precache_completion.json').exists():
            time.sleep(10)
        rows = [json.loads(line) for line in (OUT / 'diagnostic_manifest.jsonl').read_text(encoding='utf-8').splitlines()]
        assert len(rows) == 192
        assert all((OUT / 'vision_encodings' / (row['image_sha256'] + '.npz')).exists() for row in rows)
        run('diagnostic_pools', 'scripts/extract_candidate_scorer.py', '--stage', 'diagnostic')
        for kind in ('internal', 'visual'):
            run('cache_' + kind + '_diagnostic', 'scripts/cache_candidate_scorer.py', '--stage', kind,
                '--split', 'diagnostic', '--device', 'cpu')
        for kind in ('heads', 'reference'):
            run(kind + '_diagnostic', 'scripts/run_candidate_scorer.py', '--stage', kind, '--split', 'diagnostic')
        run('evaluate_diagnostic', 'scripts/evaluate_candidate_scorer.py', '--split', 'diagnostic')
        run('audit_diagnostic', 'scripts/audit_candidate_scorer.py', '--split', 'diagnostic')
        status('complete', state='complete', secondary_reused_cohort=True, model_selection=False)
    except BaseException as error:
        status('failed', state='failed', error=repr(error))
        raise
