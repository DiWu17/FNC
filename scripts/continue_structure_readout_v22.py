"""Fresh-process evaluation of the already frozen V22 models."""
from pathlib import Path
import os,sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import run_structure_readout_v22 as r

if __name__=='__main__':
    r.torch.set_num_threads(2);r.verify();frozen=r.read(r.OUT/'models/frozen.json');assert frozen['complete'] and len(frozen['models'])==12
    assert all(r.read(r.OUT/f'workers/{i}.status.json')['phase']=='complete' for i in [0,1])
    assert not list((r.OUT/'evaluation').glob('*.json')),'Preserve/evaluate existing outputs deliberately before resuming'
    r.write(r.OUT/'corrected_evaluation.execution.json',dict(pid=os.getpid(),started_at_utc=r.now(),fresh_import=True,
        runner_sha256=r.sha(__file__),scientific_source_sha256=r.sha(ROOT/'scripts/run_structure_readout_v22.py'),
        model_freeze_sha256=r.sha(r.OUT/'models/frozen.json'),protocol_sha256=r.sha(r.OUT/'protocol.json'),
        training_unchanged=True,prior_parent_stopped_before_any_predictions=True))
    r.evaluate();r.analyze()
