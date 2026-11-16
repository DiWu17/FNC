"""Active-turn plain Python polling; no scheduler or HTML."""
from datetime import datetime
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/full_scaling_v18'


def read(path):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (FileNotFoundError, PermissionError, json.JSONDecodeError):
        return {}


while True:
    s = read(RUN / 'status.json')
    fields = [datetime.now().astimezone().isoformat(timespec='seconds'), s.get('phase', 'starting')]
    if s.get('total'):
        fields.append(f'{s.get("completed", 0)}/{s["total"]}, reused={s.get("reused", 0)}, new={s.get("computed", 0)}')
    if s.get('total_steps'):
        fields.append(f'{s.get("model")}: {s.get("step", 0)}/{s["total_steps"]}')
    if s.get('models_total'):
        fields.append(f'models={s.get("models_completed", 0)}/{s["models_total"]}')
    print(' | '.join(fields), flush=True)
    with (RUN / 'monitoring.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(dict(observed_at=fields[0], full=s), ensure_ascii=False)+'\n')
    if s.get('phase') == 'failed':
        raise RuntimeError('Full run failed; inspect preserved status and logs')
    if s.get('complete'):
        print('Full run and development analysis complete: '+str(RUN/'results.txt'), flush=True)
        break
    time.sleep(50)
