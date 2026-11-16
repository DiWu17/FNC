"""Observe the active V18 chain in plain text without changing experiment inputs."""
from datetime import datetime, timezone
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/full_scaling_v18'
STAGES = {
    'full': RUN,
    'fresh_mvp': ROOT / 'runs/mvp_confirmation_v18',
    'coordinate_control': ROOT / 'runs/mvp_confirmation_v18/coordinate_addendum',
}


def read(path):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (FileNotFoundError, PermissionError, json.JSONDecodeError):
        return {}


while True:
    states = {name: read(folder / 'status.json') for name, folder in STAGES.items()}
    parts = [datetime.now().astimezone().isoformat(timespec='seconds')]
    for name, state in states.items():
        detail = state.get('phase', 'starting')
        if state.get('total'):
            detail += f' {state.get("completed", 0)}/{state["total"]}'
        if name == 'full' and 'computed' in state:
            detail += f', new={state["computed"]}, reused={state.get("reused", 0)}'
        if state.get('model'):
            detail += f' {state["model"]}: {state.get("step", 0)}/{state.get("total_steps", "?")}'
        if state.get('method'):
            detail += f' method={state["method"]}'
        parts.append(f'{name}: {detail}')
    print(' | '.join(parts), flush=True)
    with (RUN / 'chain_monitoring.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(dict(observed_at_utc=datetime.now(timezone.utc).isoformat(), stages=states), ensure_ascii=False) + '\n')
    failed = [name for name, state in states.items() if state.get('phase') == 'failed']
    if failed:
        raise RuntimeError('Experiment failure; inspect preserved logs: ' + ', '.join(failed))
    if all(state.get('complete') for state in states.values()):
        print('Full training, fresh MVP confirmation, and joint analysis complete.', flush=True)
        break
    time.sleep(50)
