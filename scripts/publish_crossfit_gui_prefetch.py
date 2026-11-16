"""Recover atomic ordinary-file publication when Windows denies hard links."""
from pathlib import Path
import json
import time
from clickagent.action_benchmark import sha256, write_json
from crossfit_protocol import OUT, cases, key, read, assert_sealed


def main():
    assert_sealed()
    audit_path=OUT/'checks/gui_parallel_execution.json'
    evidence=read(audit_path)
    assert not evidence['complete']
    assert evidence['source_sha256']==sha256(Path('scripts/prefetch_crossfit_gui.py'))
    tail=cases('test')[128:]
    assert evidence['sample_ids']==[c['sample_id'] for c in tail] and len(tail)==64
    source=OUT/'gui_prefetch/test';target=OUT/'gui_reference/test'
    assert read(source/'runtime.json')==read(target/'runtime.json')
    log=(OUT/'logs/gui_prefetch.log').read_text(encoding='utf-8',errors='replace')
    assert 'WinError 5' in log
    progress=[]
    for line in log.splitlines():
        try:
            d=json.loads(line)
            if d.get('stage')=='gui':progress.append(d)
        except (ValueError,AttributeError):pass
    assert any(d['i']==64 and d['n']==64 for d in progress)
    main_progress=[]
    for line in (OUT/'logs/test_gui.log').read_text(encoding='utf-8',errors='replace').splitlines():
        try:
            d=json.loads(line)
            if d.get('stage')=='gui':main_progress.append(d['i'])
        except (ValueError,AttributeError):pass
    assert max(main_progress,default=0)<120
    for c in tail:
        name=key(c['sample_id'])+'.json'
        data=read(source/name)
        assert data['sample_id']==c['sample_id'] and not data['ground_truth_in_model_inputs']
        assert len(data['predictions'])==6 and not (target/name).exists()
    published=[]
    for c in tail:
        name=key(c['sample_id'])+'.json'
        write_json(target/name,read(source/name))
        assert sha256(target/name)==sha256(source/name)
        published.append(dict(sample_id=c['sample_id'],sha256=sha256(target/name)))
    evidence.update(complete=True,published=published,
        anchor=dict(checked_before_tail_generation=True,tolerance=1e-6,
            max_absolute_numeric_difference=None,
            exact_value_not_persisted='Worker asserted <=1e-6 before all64 forwards; exception at publication lost its in-memory exact maximum.'),
        publication='atomic write_json ordinary files, identical source/destination SHA256',
        recovered_error='Windows denied os.link, WinError5; no prediction or training changes',
        recovery_script_sha256=sha256(Path(__file__)),
        main_progress_before_publish=max(main_progress,default=0),
        elapsed_seconds=time.time()-evidence['created_at'])
    write_json(audit_path,evidence)
    print(json.dumps(dict(complete=True,published=len(published),main_progress=evidence['main_progress_before_publish'])))


if __name__=='__main__':main()
