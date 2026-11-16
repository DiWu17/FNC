"""Observe the current Python experiment; no HTML or scheduled automation."""
from datetime import datetime
import json
from pathlib import Path
import time

OUT = Path(__file__).resolve().parents[1] / 'runs/advantage_validation_v17'


def read(path):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (FileNotFoundError, PermissionError, json.JSONDecodeError):
        return {}


def snapshot():
    suite = read(OUT / 'status.json')
    training = read(OUT / 'data_efficiency/status.json')
    fits = sum(path.name != 'frozen.json' for path in (OUT / 'data_efficiency/models').glob('*.json'))
    if suite.get('phase') in ['preflight', 'inference']:
        message = f'{suite.get("job")}: {suite.get("child_phase")}'
        if suite.get('total'):
            message += f' {suite.get("completed")}/{suite["total"]}'
    elif training.get('phase') == 'training':
        message = f'新增训练 {fits}/36；当前 {training.get("step", 0)}/{training.get("total_steps", 0)} 步'
    elif training.get('phase', '').startswith('evaluating_'):
        message = f'新增训练 {fits}/36 已冻结；{training["phase"]} {training.get("models_completed", 0)}/{training.get("models_total", 48)}'
    else:
        message = f'队列={suite.get("phase")}；数据效率={training.get("phase")}；已训练{fits}/36'
    return message, suite, training


def main():
    while True:
        message, suite, training = snapshot()
        now = datetime.now().astimezone().isoformat(timespec='seconds')
        print(f'{now} {message}', flush=True)
        with (OUT / 'monitoring.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(dict(time=now, message=message, suite=suite, training=training), ensure_ascii=False) + '\n')
        if suite.get('complete'):
            print('队列完成，Python分析结果：' + str(OUT / 'analysis/results.txt'), flush=True)
            return
        if suite.get('phase') == 'failed' or training.get('phase') == 'failed':
            raise RuntimeError('Experiment failed; inspect preserved status and logs before continuing')
        time.sleep(50)


if __name__ == '__main__':
    main()
