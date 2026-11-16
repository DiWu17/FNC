"""Disjoint tail prefetch of the fixed GUI reference on an idle identical GPU.

No prediction/training source changes. Main reference retains all first 128 cases.
Tail outputs publish atomically only while main is still before case 120.
"""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import json
from pathlib import Path
import time

import torch
import run_crossfit_scorer as runner
from clickagent.action_benchmark import sha256, write_json
from crossfit_protocol import OUT, cases, key, read, assert_sealed
from PIL import Image


def main():
    torch.set_num_threads(4)
    assert_sealed()
    cohort = cases('test')
    assert len(cohort) == 192
    tail = cohort[128:]
    target = OUT / 'gui_reference/test'
    prefetch = OUT / 'gui_prefetch/test'
    audit_path = OUT / 'checks/gui_parallel_execution.json'
    evidence = dict(complete=False, created_at=time.time(), purpose='execution-only disjoint tail prefetch',
        physical_gpu=1, cuda_visible_devices='1', selection='fixed manifest positions 129 through 192, no outcome selection',
        sample_ids=[c['sample_id'] for c in tail], source_sha256=sha256(Path(__file__)),
        original_runner_sha256=sha256(Path('scripts/run_crossfit_scorer.py')),
        original_runtime_sha256=sha256(Path('scripts/run_gui_verifier.py')),
        batch_size=6, architecture=torch.cuda.get_device_name(0))
    write_json(audit_path, evidence)
    runtime = runner.VerifierRuntime()
    # Verify identical numerical execution on a main-worker case already complete.
    anchor = next(c for c in cohort[:128] if (target / (key(c['sample_id'])+'.json')).exists()
                  and read(target / (key(c['sample_id'])+'.json'))['records'])
    sid = key(anchor['sample_id'])
    pipe = read(runner.folder('test')/'pipeline'/(sid+'.json'))
    points = [v['point'] for v in pipe['pools']['16'][:6]]
    with Image.open(anchor['image']) as im:
        result = runtime.score(im.convert('RGB'), anchor['instruction'], points, 'crop')
    expected = read(target/(sid+'.json'))['records']
    differences = []
    for value in result:
        original = expected[runner.coordinate_key(value['point'])]
        for field in ('log_odds','true_logit','false_logit','label_mass'):
            differences.append(abs(value[field]-original[field]))
        assert value['preferred_token'] == original['preferred_token']
        assert value['transform'] == original['transform']
    assert max(differences, default=0.) <= 1e-6, differences
    evidence['anchor'] = dict(sample_id=anchor['sample_id'], n_points=len(points),
        max_absolute_numeric_difference=max(differences, default=0.), tolerance=1e-6)
    assert runtime.metadata == read(target/'runtime.json')
    original_write = runner.write_json
    def prefetch_write(path, value):
        path = Path(path)
        return original_write(prefetch/path.relative_to(target), value)
    runner.write_json = prefetch_write
    runner.cases = lambda split: tail if split == 'test' else cases(split)
    runner.VerifierRuntime = lambda: runtime
    runner.reference('test')
    lines=(OUT/'logs/test_gui.log').read_text(encoding='utf-8',errors='replace').splitlines()
    progress=[]
    for line in lines:
        try:
            item=json.loads(line)
            if item.get('stage')=='gui': progress.append(item['i'])
        except (ValueError,AttributeError): pass
    evidence['main_progress_before_publish']=max(progress,default=0)
    # Leave source outputs intact and avoid overlapping work if the main has caught up.
    assert evidence['main_progress_before_publish'] < 120, evidence
    published=[]
    for case in tail:
        name=key(case['sample_id'])+'.json'
        source=prefetch/name; dest=target/name
        assert source.exists() and not dest.exists(), name
        os.link(source,dest)  # Atomic complete-file publication; fails if target exists.
        published.append(dict(sample_id=case['sample_id'],sha256=sha256(dest)))
    evidence.update(complete=True, published=published, elapsed_seconds=time.time()-evidence['created_at'])
    write_json(audit_path,evidence)
    print(json.dumps(dict(complete=True,published=len(published),anchor=evidence['anchor'])),flush=True)


if __name__=='__main__':
    main()
