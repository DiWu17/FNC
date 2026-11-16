"""Resume the registered 262k train-free baseline panel, one model at a time.

Run with the Python environment used for model inference. It never opens labels.
Each stage writes immutable per-query predictions and resumes from cached cases.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/external_262k_panel_v2'
RUNNER = ROOT / 'scripts/run_external_262k_panel_v2.py'
STAGES = (
    ('qwen', 'mmb'), ('qwen', 'sspro'),
    ('mvp', 'mmb'), ('mvp', 'sspro'),
    ('tag', 'pro'), ('tag', 'sheets'), ('tag', 'mmb'), ('tag', 'sspro'),
)


def note(phase, **fields):
    event = dict(time_utc=datetime.now(timezone.utc).isoformat(), phase=phase, **fields)
    with (RUN / 'queue_events.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(event, ensure_ascii=False) + '\n')
    print(json.dumps(event, ensure_ascii=False), flush=True)


def call(args, log_path):
    cmd = [sys.executable, '-B', '-X', 'utf8', '-u', str(RUNNER), *args]
    with log_path.open('a', encoding='utf-8') as stream:
        result = subprocess.run(cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise RuntimeError(f'Failed with code {result.returncode}; see {log_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.dry_run:
        for method, suite in STAGES:
            print(method, suite)
        print('verify-gate')
        return
    if not (RUN / 'protocol.json').is_file():
        raise FileNotFoundError('Frozen protocol missing: ' + str(RUN / 'protocol.json'))
    for method, suite in STAGES:
        note('start', method=method, suite=suite)
        try:
            call(['--stage', 'run', '--method', method, '--suite', suite],
                 RUN / f'queue_{method}_{suite}.log')
        except Exception as exc:
            note('failed', method=method, suite=suite, error=str(exc))
            raise
        note('complete', method=method, suite=suite)
    call(['--stage', 'verify-gate'], RUN / 'queue_gate.log')
    note('public_predictions_sealed')


if __name__ == '__main__':
    main()
