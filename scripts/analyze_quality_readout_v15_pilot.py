"""Descriptive diagnostics only; never changes pilot samples, models or protocol."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/quality_readout_v15_pilot'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def cohorts():
    output = {}
    for name, filename in [('old', 'old_train'), ('new45', 'add45'), ('new90', 'add90')]:
        entries = read(RUN / f'data/{filename}_index.json')
        output[name] = {}
        for source in sorted({e['source'] for e in entries}):
            rows = [e for e in entries if e['source'] == source]
            union = k16 = zero = 0
            for e in rows:
                labels = np.load(e['labels_path'], allow_pickle=False)
                zero += int(e['n'] == 0)
                union += int(labels.any())
                with np.load(e['path'], allow_pickle=False) as z:
                    k16 += int(labels[z['baseline']].any())
            output[name][source] = dict(queries=len(rows), images=len({e['group'] for e in rows}),
                zero_candidates=zero, union_has_correct_point=union,
                original_k16_has_correct_point=k16)
    write(RUN / 'analysis/cohort_reachability.json', output)
    return output


def results():
    completion = read(RUN / 'completion.json')
    assert completion['complete']
    for filename, expected in completion['artifact_sha256'].items():
        assert hashlib.sha256(Path(filename).read_bytes()).hexdigest() == expected, filename
    for model in read(RUN / 'models/frozen.json')['models'].values():
        assert hashlib.sha256(Path(model['path']).read_bytes()).hexdigest() == model['sha256']
    result = read(RUN / 'evaluation/results.json')
    overview = {}
    old = [result['methods'][f'old_s{s}']['top1_hits'] for s in (1, 2)]
    for name in ('old', 'add45', 'add90'):
        hits = [result['methods'][f'{name}_s{s}']['top1_hits'] for s in (1, 2)]
        overview[name] = dict(hits=hits, mean_hits=float(np.mean(hits)),
            mean_accuracy_percent=100 * float(np.mean(hits)) / result['n'],
            delta_hits_vs_old=[a - b for a, b in zip(hits, old)],
            mean_delta_percentage_points=100 * float(np.mean(np.subtract(hits, old))) / result['n'])
    output = dict(recorded_at_utc=datetime.now(timezone.utc).isoformat(),
        analysis_only=True, completion_hashes_verified=True, frozen_checkpoints_verified=True,
        analysis_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        primary='Top1 on the same original K16 pool, denominator172', overview=overview,
        comparisons=result['comparisons'],
        limits=['Two trained seeds are not344 independent validation examples.',
            'Cluster intervals are conditional on each fixed pair of trained models.',
            'Historically reused old validation; no fresh cross-domain robustness claim.',
            'This pilot changes source diversity and query count together, so it does not isolate quantity alone.',
            'Only45/90 added queries; do not extrapolate to the full12500-query addition.',
            'Only localization scoring; no behavior execution training.'])
    write(RUN / 'analysis/readout.json', output)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cohorts-only', action='store_true')
    args = parser.parse_args()
    value = cohorts()
    if not args.cohorts_only:
        value = results()
    print(json.dumps(value, ensure_ascii=False, indent=2))
