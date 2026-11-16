"""Parallel verifier scheduling must preserve point sets and saved decisions."""
import json
import os
from pathlib import Path
import sys

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import run_multiregion_v13 as run
import run_gui_verifier as gui
from clickagent.action_benchmark import sha256, write_json
from crossfit_protocol import key


def fixture_tree(tmp_path, name):
    root = tmp_path/name; root.mkdir()
    write_json(root/'frozen.json', {'complete': True})
    image = tmp_path/'shared.png'
    if not image.exists():
        Image.new('RGB', (60, 40), 'white').save(image)
    cases = []
    for i in range(5):
        sid = 'sample'+str(i)
        cases.append(dict(sample_id=sid, image=str(image), image_sha256=sha256(image), instruction='target'))
        pools = {condition: {str(k): [dict(point=[x, i+3], source='global', source_rank=x) for x in
            ([2, 8] if k == 4 else [2, 8, 15, 22] if condition == 'c1' else [2, 8, 32, 45])]
            for k in (4, 16)} for condition in ('c1', 'c3')}
        write_json(root/'pipeline/test'/(key(sid)+'.json'), dict(sample_id=sid, pools=pools))
        write_json(root/'predictions/test'/(key(sid)+'.json'), dict(predictions=[
            dict(method=condition+'_top1_s1_val', budget=16, condition=condition, scores=[0., 1, 2, 3])
            for condition in ('c1', 'c3')]))
    return root, cases


@pytest.mark.parametrize('failure_mode', ['none', 'score', 'contract'])
def test_partitioned_reference_preserves_all_point_scores_and_decisions_or_rejects_drift(tmp_path, monkeypatch, failure_mode):
    drift_state = [failure_mode]
    serial, cases = fixture_tree(tmp_path, 'serial')
    parallel, _ = fixture_tree(tmp_path, 'parallel')
    monkeypatch.setattr(run, 'assert_sealed', lambda: None)
    monkeypatch.setattr(run, 'cases', lambda split: cases)
    class FakeRuntime:
        def __init__(self):
            keys = ['b_inv_freq', 'a_inv_freq'] if os.environ.get('CUDA_VISIBLE_DEVICES') == '0' else ['a_inv_freq', 'b_inv_freq']
            revision = 'mismatched' if drift_state[0] == 'contract' and os.environ.get('V13_SYNTHETIC_PHASE') == 'cases' else 'synthetic'
            self.metadata = {'frozen': True, 'revision': revision, 'loading': {'unexpected_keys': keys}}
        def score(self, image, instruction, points, mode):
            assert instruction == 'target' and mode == 'crop'
            offset = 1e-4 if drift_state[0] == 'score' and os.environ.get('CUDA_VISIBLE_DEVICES') == '1' else 0.
            return [dict(point=p, log_odds=p[0]*.1+p[1]*.03+offset,
                         batch_elapsed_seconds=.1) for p in points]
    monkeypatch.setattr(gui, 'VerifierRuntime', FakeRuntime)
    monkeypatch.setattr(run.torch.cuda, 'device_count', lambda: 0)
    monkeypatch.setattr(run, 'OUT', serial)
    run.reference('test')
    devices = []
    class FakeProcess:
        returncode = 0
        def __init__(self, command, env):
            devices.append(env['CUDA_VISIBLE_DEVICES'])
            part = int(command[command.index('--part')+1])
            phase = command[command.index('--reference-phase')+1]
            execution_dir = Path(command[command.index('--execution-dir')+1])
            with monkeypatch.context() as ctx:
                ctx.setenv('CUDA_VISIBLE_DEVICES', env['CUDA_VISIBLE_DEVICES'])
                ctx.setenv('V13_SYNTHETIC_PHASE', phase)
                run.reference('test', part=part, parts=2, phase=phase, execution_dir=execution_dir)
        def poll(self):
            return self.returncode
        def terminate(self):
            self.returncode = -15
        def wait(self, timeout):
            return self.returncode
        def kill(self):
            self.returncode = -9
    monkeypatch.setattr(run.subprocess, 'Popen', FakeProcess)
    monkeypatch.setattr(run.torch.cuda, 'device_count', lambda: 2)
    monkeypatch.setattr(run, 'OUT', parallel)
    if failure_mode != 'none':
        message = 'GPU anchor score disagreement' if failure_mode == 'score' else 'Case runtime differs from accepted anchor contract'
        with pytest.raises(AssertionError, match=message):
            run.reference('test')
        assert not (parallel/'gui_reference/test/parallel_execution.json').exists()
        assert all(not (parallel/'gui_reference/test'/(key(c['sample_id'])+'.json')).exists() for c in cases)
        drift_state[0] = 'none'
    run.reference('test')
    before_resume = {}
    for case in cases:
        name = key(case['sample_id'])+'.json'
        a = json.loads((serial/'gui_reference/test'/name).read_text(encoding='utf8'))
        b = json.loads((parallel/'gui_reference/test'/name).read_text(encoding='utf8'))
        for field in ('records', 'predictions', 'batches', 'input_identity', 'timing_scope'):
            assert a[field] == b[field]
        assert b['execution_mode'] == 'parallel'
        assert sha256(Path(b['parallel_acceptance_path'])) == b['parallel_acceptance_sha256']
        before_resume[name] = sha256(parallel/'gui_reference/test'/name)
    report = json.loads((parallel/'gui_reference/test/parallel_execution.json').read_text(encoding='utf8'))
    assert report['complete'] and report['anchor_max_abs_log_odds_difference'] == 0
    assert report['fixed_label_free_anchor_points'] == 12 and report['accepted_before_case_workers']
    run.reference('test')
    assert all(sha256(parallel/'gui_reference/test'/name) == digest for name, digest in before_resume.items())
    failed_devices = [] if failure_mode == 'none' else ['0', '1'] if failure_mode == 'score' else ['0', '1', '0']
    assert devices == failed_devices+['0', '1', '0', '1']*2


def test_partition_has_no_omitted_or_duplicate_case_for_odd_count():
    partitions = [set(run.partition_positions(5, part, 2)) for part in (0, 1)]
    assert not partitions[0] & partitions[1]
    assert set.union(*partitions) == set(range(5))
    with pytest.raises(ValueError):
        run.partition_positions(5, 2, 2)
