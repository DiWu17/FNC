"""Validation-only audit of token-score shift invariance after frozen fitting.

No checkpoint or test prediction is changed. Additive constants do not change
an individual token's spatial distribution. Global token weights preserve
that invariance; position-dependent weights over raw logits need not do so.
Normalization variants here are post-fit interventions, not newly fit models.
"""
import json
from pathlib import Path

import numpy as np
import torch

from clickagent.action_benchmark import sha256, write_json
from clickagent.conditioned_grounder import make_batch
from clickagent.phrase_grounder import decode_phrase, load_phrase
from evaluate_conditioned_readouts import score_point
from run_target_token_experiment import read_jsonl
from train_conditioned_readouts import batches, load_records


def main():
    out = Path('runs/phrase_v4'); torch.set_num_threads(4)
    cases = [c for c in read_jsonl(out/'manifest.jsonl') if c['experiment_split'] == 'val']
    records, _ = load_records(Path('runs/conditioned_v3'), cases, with_labels=False)
    model, _ = load_phrase(out/'models/local.pt', 'cuda:0')
    counts = {name: 0 for name in ('original', 'mean_centered', 'log_spatial_probability')}
    changed = {name: 0 for name in counts}; bias_parent_changes = 0; global_parent_changes = 0; eligible = 0
    rows = []
    with torch.inference_mode():
        for group in batches(records, 8):
            inputs, _ = make_batch(group, 'cuda:0', labels=False); result = model(**inputs)
            a = inputs['attention'].float(); p = a.shape[-1]
            maps = torch.log1p(p*a/a.sum(-1, keepdim=True).clamp_min(1e-12))
            evidence = torch.einsum('bth,bthp->btp', result['head_weights'], maps)
            weights = result['token_weights']; gh, gw = inputs['grid_shape']
            variants = {'original': evidence, 'mean_centered': evidence-evidence.mean(-1, keepdim=True),
                        'log_spatial_probability': evidence-evidence.logsumexp(-1, keepdim=True)}
            points = {}
            for name, scores in variants.items():
                value = dict(result, coarse_logits=(weights*scores).sum(1).reshape(-1, gh, gw))
                cpu = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in value.items()}
                points[name] = [decode_phrase(cpu, i, r['case']['image_size'], top_k=1)[0]['point'] for i, r in enumerate(group)]
            # Same per-token spatial peaks/distributions, arbitrary constants.
            # +/-1 logit-unit constants are independent of GT and patch index.
            lengths = inputs['token_mask'].sum(-1)
            offsets = torch.where(torch.arange(a.shape[1], device=a.device) % 2 == 0, 1., -1.)[None, :, None]
            shifted = evidence+offsets
            torch.testing.assert_close(shifted.argmax(-1), evidence.argmax(-1))
            local_before = (weights*evidence).sum(1).argmax(-1)
            local_after = (weights*shifted).sum(1).argmax(-1)
            global_weights = weights.mean(-1, keepdim=True)
            global_before = (global_weights*evidence).sum(1).argmax(-1)
            global_after = (global_weights*shifted).sum(1).argmax(-1)
            for i, r in enumerate(group):
                c = r['case']; row = dict(sample_id=c['sample_id'], n_tokens=len(r['target_attention']))
                for name in variants:
                    hit = score_point(points[name][i], c); counts[name] += hit
                    changed[name] += points[name][i] != points['original'][i]
                    row[name] = dict(point=points[name][i], hit=hit)
                if int(lengths[i]) > 1:
                    eligible += 1
                    bias_parent_changes += int(local_before[i] != local_after[i])
                    global_parent_changes += int(global_before[i] != global_after[i])
                rows.append(row)
    result = dict(split='validation_only', n=len(cases), named=len(rows), point_hits=counts, point_changes=changed,
                  multi_token_cases=eligible, local_parent_changes_under_token_constant_offsets=bias_parent_changes,
                  global_weights_control_parent_changes=global_parent_changes,
                  intervention='Alternate +1/-1 constant across tokens, constant over every spatial position; individual token peaks unchanged.',
                  no_training=True, no_test_outputs_changed=True, source_sha256=sha256(Path(__file__)),
                  checkpoint_sha256=sha256(out/'models/local.pt'),
                  warning='Normalization interventions use weights fitted on unnormalized scores. They are exploratory validation diagnostics; no fresh independent test claim.')
    write_json(out/'evaluation/score_reference_validation.json', result)
    (out/'evaluation/score_reference_validation_cases.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
