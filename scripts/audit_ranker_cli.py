"""One validation-only raw-cache CLI replay; no fresh test is inspected."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from clickagent.action_benchmark import sha256, write_json
from clickagent.candidate_ranker import decode_ranker, load_ranker, make_ranker_batch
from run_target_token_experiment import key, read_jsonl


def main():
    out = Path('runs/candidate_ranker_v5'); candidates = [c for c in read_jsonl(out/'manifest.jsonl') if c['experiment_split'] == 'val']
    for c in candidates:
        with np.load(out/'candidate_cache'/f'{key(c["sample_id"])}.npz', allow_pickle=False) as z:
            meta = json.loads(str(z['metadata']))
            if not meta['target_available']: continue
            record = {k: z[k].copy() for k in z.files if k != 'metadata'}
        record.update(case=c, metadata=meta); break
    model, _ = load_ranker(out/'models/full.pt', 'cpu'); torch.set_num_threads(4)
    with torch.inference_mode():
        inputs, _ = make_ranker_batch([record], 'cpu', labels=False); output = model(**inputs)
        expected = decode_ranker(output, record, top_k=1)[0]['point']
    raw_path = Path('runs/conditioned_v3/features')/f'{key(c["sample_id"])}.npz'
    result = subprocess.run([sys.executable, '-X', 'utf8', 'scripts/predict_candidate_ranker.py',
        '--features', str(raw_path), '--device', 'cpu'], capture_output=True, encoding='utf-8', check=True)
    actual = json.loads(result.stdout)
    assert actual['proposed_point'] == expected
    assert sha256(raw_path) == meta['source_archive_sha256']
    dest = out/'evaluation'; dest.mkdir(exist_ok=True)
    audit = dict(complete=True, split='validation_only', sample_id=c['sample_id'],
        raw_input_and_candidate_cache_same_point=True, proposed_point=expected,
        no_geometry_in_cli_arguments=True, checkpoint_sha256=sha256(out/'models/full.pt'),
        cli_source_sha256=sha256(Path('scripts/predict_candidate_ranker.py')), source_sha256=sha256(Path(__file__)))
    write_json(dest/'cli_smoke.json', audit); print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__': main()
