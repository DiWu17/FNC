"""Exercise the public cached-feature CLI against a frozen test prediction."""
import json
from pathlib import Path
import subprocess
import sys

from clickagent.action_benchmark import write_json
from run_target_token_experiment import key, read_jsonl


def main():
    out = Path('runs/phrase_v4')
    rows = [r for r in read_jsonl(out/'evaluation/predictions.jsonl') if r['method'] == 'local' and r['point'] is not None]
    selected = min(rows, key=lambda r: key(r['sample_id']))
    feature = out/'features'/f'{key(selected["sample_id"])}.npz'
    command = [sys.executable, '-X', 'utf8', 'scripts/predict_phrase_point.py', '--features', str(feature), '--device', 'cpu']
    result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', check=True)
    response = json.loads(result.stdout)
    assert response['point'] == selected['point']
    write_json(out/'evaluation/cli_smoke.json', dict(complete=True, sample_id=selected['sample_id'],
               cli_point=response['point'], cached_test_point=selected['point'], exact_match=True,
               generated_target_name=response['target_name'], raw_response=response))
    print(json.dumps({'complete': True, 'cli_point': response['point'], 'exact_match': True}))


if __name__ == '__main__':
    main()
