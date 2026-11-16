"""CPU interim replay of frozen readouts while external baseline runs."""
import json
from pathlib import Path

import torch

from clickagent.action_benchmark import sha256, write_json
from clickagent.candidate_ranker import load_ranker
from ranker_data import load_records
from train_candidate_ranker import evaluate


def main():
    out = Path('runs/candidate_ranker_v5'); torch.set_num_threads(4)
    frozen = json.loads((out/'frozen_readouts.json').read_text(encoding='utf-8'))
    for source, digest in frozen['head_sources'].items(): assert sha256(Path(source)) == digest
    assert json.loads((out/'candidates_test_completion.json').read_text(encoding='utf-8'))['complete']
    records = load_records(out, 'test'); result = {}
    for name, entry in frozen['models'].items():
        path = out/'models'/entry['file']; assert sha256(path) == entry['sha256']
        model, _ = load_ranker(path, 'cpu'); result[name] = evaluate(model, records, 'cpu')
        del model
        print(name+' '+json.dumps(result[name]), flush=True)
    write_json(out/'evaluation/readout_only_results.json', dict(device='cpu', interim=True, methods=result,
        warning='Final canonical GPU evaluation and matched baseline table are produced by evaluate_candidate_ranker.py.',
        source_sha256=sha256(Path(__file__)), frozen_readouts_sha256=sha256(out/'frozen_readouts.json')))


if __name__ == '__main__': main()
