"""Interim CPU replay of old pointing and shared-fine decoder controls."""
import json
from pathlib import Path

import torch

from clickagent.action_benchmark import write_json
from clickagent.candidate_ranker import decode_ranker, load_ranker, make_ranker_batch
from clickagent.conditioned_grounder import make_batch
from clickagent.phrase_grounder import decode_phrase, load_phrase
from evaluate_candidate_ranker import parent_points
from evaluate_conditioned_readouts import score_point
from ranker_data import load_records
from run_target_token_experiment import read_jsonl
from train_conditioned_readouts import batches, load_records as load_raw


@torch.inference_mode()
def main():
    out = Path('runs/candidate_ranker_v5'); torch.set_num_threads(4)
    records = load_records(out, 'test'); by = {r['case']['sample_id']: r for r in records}
    raw, _ = load_raw(out, [r['case'] for r in records], with_labels=False)
    ranker, _ = load_ranker(out/'models/full.pt', 'cpu'); result = {}; rows = []
    for name in ('local', 'global'):
        model, _ = load_phrase(Path('runs/phrase_v4/models')/(name+'.pt'), 'cpu')
        hits = 0; regions = 0; frozen_fine_hits = 0
        for group in batches(raw, 8):
            inputs, _ = make_batch(group, 'cpu', labels=False); old = model(**inputs)
            if name == 'local':
                ranked_group = [by[r['case']['sample_id']] for r in group]
                ri, _ = make_ranker_batch(ranked_group, 'cpu', labels=False); ranked = ranker(**ri)
            for i, r in enumerate(group):
                c = r['case']; point = decode_phrase(old, i, c['image_size'], top_k=1)[0]['point']
                hits += score_point(point, c)
                row = dict(sample_id=c['sample_id'], method='phrase_'+name, point=point)
                if name == 'local':
                    rr = ranked_group[i]; proposal = decode_ranker(ranked, rr, i, top_k=1)[0]
                    table = parent_points(old['child_log_prob'][i].numpy(), old['offsets'][i].numpy(), r['metadata']['grid_shape'], c['image_size'])
                    fp = table[proposal['parent_index']]; frozen_fine_hits += score_point(fp, c)
                    row['new_ranker_frozen_fine_point'] = fp
                rows.append(row)
        result['phrase_'+name] = dict(hits=int(hits), n=len(records))
        if name == 'local': result['full_frozen_fine'] = dict(hits=int(frozen_fine_hits), n=len(records))
        print(name+' '+json.dumps(result), flush=True)
    write_json(out/'evaluation/interim_cpu_controls.json', dict(interim=True, methods=result))
    (out/'evaluation/interim_cpu_controls_points.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows), encoding='utf-8')


if __name__ == '__main__': main()
