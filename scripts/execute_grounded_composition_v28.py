"""Non-interactive, sequential executor; progress is polled by the caller."""
import os,sys,json,time,subprocess,hashlib
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/grounded_composition_v28'
def now():return datetime.now(timezone.utc).isoformat()
def save(name,value):
    path=OUT/name;tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');os.replace(tmp,path)
plan=dict(started_at_utc=now(),driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),phases=['collect_train','fit','collect_dev','finish'],wait_for='predictions_stage1.json',automatic=True)
save('execution_plan.json',plan);rows=[]
while not (OUT/'predictions_stage1.json').exists():
    if (OUT/'status.json').exists():
        current=json.loads((OUT/'status.json').read_text(encoding='utf-8'))
        if current['phase']=='failed':raise RuntimeError('Stage1 failed: '+str(current))
    time.sleep(10)
for phase in plan['phases']:
    started=now();print('START',phase,started,flush=True)
    with (OUT/(phase+'.stdout.log')).open('wb') as stdout,(OUT/(phase+'.stderr.log')).open('wb') as stderr:
        result=subprocess.run([sys.executable,'-B','-X','utf8','-u',str(ROOT/'scripts/run_grounded_composition_v28.py'),phase],cwd=ROOT,stdout=stdout,stderr=stderr)
    rows.append(dict(phase=phase,started_at_utc=started,finished_at_utc=now(),exit_code=result.returncode))
    save('execution.json',dict(complete=False,phases=rows));print('EXIT',phase,result.returncode,flush=True)
    if result.returncode:sys.exit(result.returncode)
save('execution.json',dict(complete=True,finished_at_utc=now(),phases=rows))
print('ALL_STAGES_COMPLETE',now(),flush=True)
