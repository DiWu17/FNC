"""Training diagnostics overlap VLM inference; interpretation follows completion."""
import os,sys,json,time,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/grounded_composition_v28';RECEIPT=ROOT/'research/executions/v28_followups.json'
rows=[]
def wait_for(path):
    while not path.exists():
        if (OUT/'status.json').exists():
            status=json.loads((OUT/'status.json').read_text(encoding='utf-8'))
            if status['phase']=='failed':raise RuntimeError(str(status))
        time.sleep(10)
def execute(script,part):
    print('START',part,flush=True)
    with (OUT/(part+'.stdout.log')).open('wb') as stdout,(OUT/(part+'.stderr.log')).open('wb') as stderr:
        result=subprocess.run([sys.executable,'-B','-X','utf8','-u',str(ROOT/'scripts'/script)],cwd=ROOT,stdout=stdout,stderr=stderr)
    rows.append(dict(part=part,exit_code=result.returncode));RECEIPT.parent.mkdir(parents=True,exist_ok=True);RECEIPT.write_text(json.dumps(dict(complete=False,phases=rows),indent=2)+'\n',encoding='utf-8')
    print('EXIT',part,result.returncode,flush=True)
    if result.returncode:sys.exit(result.returncode)
wait_for(OUT/'models_stage2/frozen.json');execute('analyze_grounded_composition_v28_transfer.py','training_transfer')
wait_for(OUT/'completion.json');execute('analyze_grounded_composition_v28_mechanisms.py','mechanisms')
while not (OUT/'execution.json').exists() or not json.loads((OUT/'execution.json').read_text(encoding='utf-8')).get('complete'):time.sleep(5)
execute('finalize_grounded_composition_v28.py','finalize')
RECEIPT.write_text(json.dumps(dict(complete=True,phases=rows),indent=2)+'\n',encoding='utf-8');print('FOLLOWUPS_COMPLETE',flush=True)
