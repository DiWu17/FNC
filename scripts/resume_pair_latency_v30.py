"""I/O retry and completed-measurement resume for the sealed V30 pilot.

The original runtime, prompts, models, timing boundaries and analyzer are reused
unchanged. The first failed attempt and its five completed measurements remain.
"""
import argparse
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import run_pair_latency_v30 as run

OUT=run.OUT

def install_retry():
    original=os.replace
    def replace(src,dst,*args,**kwargs):
        for attempt in range(100):
            try:return original(src,dst,*args,**kwargs)
            except PermissionError as error:
                if (getattr(error,'winerror',None) not in (5,32,33)
                    or not Path(dst).resolve().is_relative_to(OUT)
                    or attempt==99):raise
                with (OUT/'io_retry_events.jsonl').open('a',encoding='utf-8') as stream:
                    stream.write(json.dumps(dict(at_utc=run.now(),path=str(dst),attempt=attempt+1))+'\n')
                time.sleep(.1)
    os.replace=replace

def amendment():
    run.verify()
    p=OUT/'execution_amendment_01.json'
    if p.exists():
        value=run.read(p);assert run.sha(__file__)==value['resume_source_sha256'];return value
    completed=list((OUT/'ours/cases').glob('*.json'))
    assert len(completed)==5
    value=dict(registered_at_utc=run.now(),reason='Windows WinError5 atomic status-file replace after fifth valid measurement',
        original_source_unchanged=True, original_protocol_sha256=run.sha(OUT/'protocol.json'),
        resume_source=str(Path(__file__).resolve()),resume_source_sha256=run.sha(__file__),
        change='Retry only in-output atomic writes; resume existing fully saved measurements. Reload and warm model before remaining measurements. No inference, scoring, prompt, sampling, or timing-boundary changes.',
        completed_original_measurement_sha256={str(p):run.sha(p) for p in completed},
        failure_log_sha256=run.sha(OUT/'ours/stderr.log'))
    run.write(p,value)
    return value

def worker(method):
    amendment()
    from clickagent.advantage_runtimes import configure_paths
    configure_paths('mvp' if method=='mvp' else 'coordinate')
    import torch
    torch.set_num_threads(4);folder=OUT/method;folder.mkdir(parents=True,exist_ok=True)
    runtime_old=run.read(folder/'runtime.json') if (folder/'runtime.json').exists() else None
    run.status(method,'loading_model_after_resume' if runtime_old else 'loading_model')
    started=time.perf_counter()
    if method=='ours':runtime=run.CurrentRuntime(folder)
    else:
        from clickagent.advantage_runtimes import MVPAdapter
        runtime=MVPAdapter(folder,262144);run.instrument_mvp(runtime,torch)
    run.sync(torch);load_seconds=time.perf_counter()-started
    attempts=([runtime_old['load_seconds']] if runtime_old else [])+[load_seconds]
    metadata=dict(load_seconds=load_seconds,load_attempt_seconds=attempts,metadata=runtime.metadata,
        torch_version=torch.__version__,transformers_version=__import__('transformers').__version__,
        gpus=[torch.cuda.get_device_name(d) for d in range(torch.cuda.device_count())],cuda_version=torch.version.cuda)
    run.write(folder/f'runtime_attempt{len(attempts):02d}.json',metadata)
    if runtime_old:run.write(folder/'runtime_attempt01.json',runtime_old)
    run.write(folder/'runtime.json',metadata)
    run.status(method,'model_loaded',load_seconds=load_seconds,attempt=len(attempts))
    cases=run.read(OUT/'data/public.json');rows=[]
    def once(c,rep,warm=False):
        payload=Path(c['image']).read_bytes();assert hashlib.sha256(payload).hexdigest()==c['image_sha256']
        run.sync(torch)
        for d in range(torch.cuda.device_count()):torch.cuda.reset_peak_memory_stats(d)
        before=runtime.counter.snapshot()
        if method=='mvp':runtime.generation_records=[]
        started=time.perf_counter()
        with run.Image.open(BytesIO(payload)) as source:image=source.convert('RGB')
        assert list(image.size)==c['image_size']
        try:output=runtime.infer(image,c['instruction'])
        finally:image.close()
        run.sync(torch);seconds=time.perf_counter()-started;after=runtime.counter.snapshot()
        assert output['ground_truth_in_model_input'] is False
        if method=='mvp':
            output['stages']=runtime.generation_records.copy();output['generation_calls']=len(output['stages'])
        row=dict(sample_id=c['sample_id'],method=method,repetition=rep,warmup=warm,process_attempt=len(attempts),
            application=c['application'],ui_type=c['ui_type'],image_size=c['image_size'],elapsed_seconds=seconds,
            counter={k:after[k]-before[k] for k in before},
            peak_allocated_bytes=[torch.cuda.max_memory_allocated(d) for d in range(torch.cuda.device_count())],output=output)
        part='warmup' if warm else 'cases'
        if warm and runtime_old:part='warmup_after_resume'
        run.write(folder/part/f'{run.key(c["sample_id"])}_r{rep}.json',row)
        return row
    for c in run.read(OUT/'data/warmup.json'):
        row=once(c,0,True);run.status(method,'warmup_complete',seconds=row['elapsed_seconds'],attempt=len(attempts))
    reused=0
    for rep in [1,2]:
        for c in cases if rep==1 else list(reversed(cases)):
            path=folder/'cases'/f'{run.key(c["sample_id"])}_r{rep}.json'
            if path.exists():
                row=run.read(path)
                assert row['sample_id']==c['sample_id'] and row['method']==method and row['repetition']==rep and not row['warmup']
                reused+=1
            else:
                row=once(c,rep)
            rows.append(row)
            run.status(method,'measuring',completed=len(rows),total=12,retained_completed_measurements=reused,
                sample_id=c['sample_id'],repetition=rep,seconds=row['elapsed_seconds'],
                generation_calls=row['output']['generation_calls'],counter=row['counter'])
    run.write(folder/'results.json',dict(complete=True,model_load_seconds=attempts[0],
        model_load_attempt_seconds=attempts,retained_completed_measurements=reused,rows=rows))
    run.status(method,'complete',measured_queries=len(rows),retained_completed_measurements=reused,
        mean_seconds=float(run.np.mean([r['elapsed_seconds'] for r in rows])))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['all','ours','mvp']);args=parser.parse_args()
    install_retry();amendment()
    if args.stage!='all':worker(args.stage);return
    for method in ['ours','mvp']:
        result=OUT/method/'results.json'
        if result.exists() and run.read(result)['complete']:continue
        run.status('controller','running_'+method)
        folder=OUT/method;folder.mkdir(parents=True,exist_ok=True)
        with (folder/'stdout.log').open('a',encoding='utf-8') as stdout,(folder/'stderr.log').open('a',encoding='utf-8') as stderr:
            process=subprocess.run([sys.executable,'-B','-X','utf8','-u',str(Path(__file__).resolve()),method],
                cwd=run.ROOT,stdout=stdout,stderr=stderr)
        assert process.returncode==0,f'{method} failed; see {folder}/stderr.log'
    run.analyze()
    run.write(OUT/'completion.json',dict(complete=True,at_utc=run.now(),protocol_sha256=run.sha(OUT/'protocol.json'),
        execution_amendment_sha256=run.sha(OUT/'execution_amendment_01.json'),
        original_completed_rows_preserved=all(run.sha(p)==h for p,h in amendment()['completed_original_measurement_sha256'].items()),
        analysis_sha256=run.sha(OUT/'analysis.json'),methods=['ours','mvp'],queries_per_method=12))

if __name__=='__main__':
    try:main()
    except BaseException:
        run.status('controller','failed',traceback=traceback.format_exc());raise
