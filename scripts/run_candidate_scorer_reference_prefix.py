"""Schedule a fixed early reference prefix; delegate every score to sealed code.

This changes scheduling only. It waits for the first64 completed pipelines,
then invokes the original reference() with that fixed prefix. It never reads
annotations, selects cases by outcomes, or modifies sealed predictor sources.
"""
from __future__ import annotations

import json
from pathlib import Path
import time

import torch

from clickagent.action_benchmark import sha256, write_json
import run_candidate_scorer as inference
from run_target_token_experiment import key


OUT = Path('runs/candidate_scorer_v11')
PREFIX = 64


if __name__ == '__main__':
    torch.set_num_threads(4)
    inference.assert_test_sealed()
    original_cases = inference.cases
    # No outcome/geometry field is inspected here. The sealed reference()
    # separately selects its public model-input fields, unchanged.
    all_cases = original_cases('test')
    assert len(all_cases) == 192 and PREFIX < len(all_cases)
    selected = all_cases[:PREFIX]
    progress = OUT / 'reference_prefix_progress.json'
    started = time.time()
    while True:
        ready = sum((OUT / 'test/pipeline' / (key(c['sample_id'])+'.json')).exists() for c in selected)
        write_json(progress, dict(stage='waiting_for_prefix', n=PREFIX, ready=ready,
                                 elapsed_seconds=time.time()-started))
        if ready == PREFIX:
            break
        master = json.loads((OUT / 'experiment_progress.json').read_text(encoding='utf-8'))
        assert master['stage'] == 'fresh_test_features', 'Stop before competing with the official reference stage'
        time.sleep(5)
    pipeline_hashes = {str(OUT / 'test/pipeline' / (key(c['sample_id'])+'.json')):
                       sha256(OUT / 'test/pipeline' / (key(c['sample_id'])+'.json')) for c in selected}
    write_json(progress, dict(stage='scoring', n=PREFIX, ready=PREFIX,
                             elapsed_seconds=time.time()-started,
                             schedule_only=True, original_reference_function=True, gpu='physical cuda:0',
                             concurrent_with_frozen_qwen=True, pipeline_sha256=pipeline_hashes,
                             script_sha256=sha256(Path(__file__))))
    def scheduled_cases(split):
        if split != 'test':
            yield from original_cases(split)
            return
        for case in selected:
            phase = json.loads((OUT / 'experiment_progress.json').read_text(encoding='utf-8'))['stage']
            if phase != 'fresh_test_features':
                return  # Release GPU before the official post-extraction jobs.
            yield case
    inference.cases = scheduled_cases
    try:
        inference.reference('test')
    finally:
        inference.cases = original_cases
    assert all(sha256(Path(p)) == digest for p, digest in pipeline_hashes.items())
    inference.assert_test_sealed()
    completed = sum((OUT / 'gui_reference/test' / (key(c['sample_id'])+'.json')).exists() for c in selected)
    write_json(progress, dict(stage='complete' if completed == PREFIX else 'yielded_to_master',
                             complete=completed == PREFIX, n=PREFIX, scored=completed,
                             schedule_only=True, original_reference_function=True,
                             concurrent_with_frozen_qwen=True,
                             elapsed_seconds=time.time()-started, pipeline_sha256=pipeline_hashes,
                             script_sha256=sha256(Path(__file__))))
