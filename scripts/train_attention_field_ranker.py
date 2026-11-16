"""Matched v6 field/masked-field experiment. Test is a previously seen diagnostic set."""
import argparse
import gc
import json
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch

from clickagent.action_benchmark import sha256, write_json
from clickagent.attention_field_ranker import (AttentionFieldRanker, FieldRankerConfig,
    load_field_ranker, make_field_batch, save_field_ranker, subset_field_candidates)
from clickagent.candidate_ranker import decode_ranker, ranker_loss
from attention_field_data import load_records
from evaluate_conditioned_readouts import score_point


@torch.inference_mode()
def evaluate(model, records, device, batch_size=8):
    model.eval(); named = [r for r in records if r['metadata']['target_available']]
    hits = 0; centers = 0; covered = 0; region = {str(k): 0 for k in (1, 3, 5)}; losses = []
    for start in range(0, len(named), batch_size):
        group = named[start:start+batch_size]; inputs, labels = make_field_batch(group, device)
        output = model(**inputs); loss, _ = ranker_loss(output, **labels); losses.extend([float(loss)]*len(group))
        cpu = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in output.items()}
        for i, r in enumerate(group):
            proposals = decode_ranker(cpu, r, i); center = decode_ranker(cpu, r, i, mode='center', top_k=1)[0]['point']
            hits += score_point(proposals[0]['point'], r['case']); centers += score_point(center, r['case'])
            positive = r['label_area'].sum(-1) > 0; covered += bool(positive.any())
            for k in region: region[k] += any(positive[p['candidate_index']] for p in proposals[:int(k)])
    return dict(n=len(records), named=len(named), hits=int(hits), accuracy=hits/len(records), center_hits=int(centers),
        region_hits={k: int(v) for k, v in region.items()}, candidate_region_coverage=int(covered),
        selection_accuracy_given_covered=region['1']/covered if covered else None, loss=float(np.mean(losses)))


def train_one(out, name, mode, seed, train_all, val, protocol, prior, device):
    path = out/'models'/f'{name}.pt'; history_path = out/'models'/f'{name}_history.json'
    if history_path.exists():
        saved = json.loads(history_path.read_text())
        assert saved['complete'] and saved['checkpoint_sha256'] == sha256(path)
        return saved
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); np.random.seed(seed); random.seed(seed)
    rng = np.random.default_rng(seed)
    model = AttentionFieldRanker(FieldRankerConfig(protocol['width'], mode), prior).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=protocol['learning_rate'], weight_decay=protocol['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, protocol['max_epochs'], eta_min=protocol['learning_rate']*.1)
    train = [r for r in train_all if r['metadata']['target_available']]
    history = []; best_rank = None; best_epoch = 0; started = time.perf_counter()
    for epoch in range(protocol['max_epochs']+1):
        losses = []; removed = 0
        if epoch:
            model.train(); order = rng.permutation(len(train))
            for start in range(0, len(order), protocol['batch_size']):
                group = []
                for index in order[start:start+protocol['batch_size']]:
                    r = train[index]; positive = r['label_area'].sum(-1) > 0
                    if positive.any() and not positive.all() and rng.random() < protocol['candidate_removal_probability']:
                        r = subset_field_candidates(r, ~positive); removed += 1
                    group.append(r)
                inputs, labels = make_field_batch(group, device); optimizer.zero_grad(set_to_none=True)
                loss, _ = ranker_loss(model(**inputs), **labels)
                if not torch.isfinite(loss): raise ValueError('Nonfinite field-ranker loss')
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); optimizer.step()
                losses.append(float(loss.detach()))
            scheduler.step()
        metrics = evaluate(model, val, device, protocol['batch_size'])
        row = dict(epoch=epoch, val=metrics, elapsed_seconds=time.perf_counter()-started,
            train_counterfactual_lists=removed, train_loss=float(np.mean(losses)) if losses else None)
        history.append(row); rank = (metrics['region_hits']['1'], metrics['hits'], -metrics['loss'])
        if best_rank is None or rank > best_rank:
            best_rank = rank; best_epoch = epoch
            save_field_ranker(path, model, dict(name=name, field_mode=mode, epoch=epoch, seed=seed,
                validation=metrics, train_named=len(train), backbone_trainable_parameters=0))
        write_json(out/'models'/f'{name}_progress.json', dict(history=history, best_epoch=best_epoch))
        print(name+' '+json.dumps(row), flush=True)
        if epoch >= protocol['min_epochs'] and epoch-best_epoch >= protocol['patience']: break
    selected, _ = load_field_ranker(path, device)
    result = dict(complete=True, name=name, field_mode=mode, best_epoch=best_epoch, seed=seed,
        validation=history[best_epoch]['val'], training=evaluate(selected, train_all, device), history=history,
        checkpoint_sha256=sha256(path), parameters=sum(p.numel() for p in model.parameters()))
    write_json(history_path, result)
    print('SELECTED '+name+' '+json.dumps({k: v for k, v in result.items() if k != 'history'}), flush=True)
    del model, selected, optimizer; gc.collect(); torch.cuda.empty_cache()
    return result


def main():
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, default=Path('runs/attention_field_v6'))
    p.add_argument('--device', default='cuda:0'); args = p.parse_args(); out = args.output
    if (out/'frozen_readouts.json').exists(): raise ValueError('Already frozen; no refitting')
    protocol = json.loads((out/'protocol.json').read_text()); base = Path(protocol['base_run'])
    assert sha256(base/'manifest.jsonl') == protocol['manifest_sha256']
    assert not (out/'fields_test_completion.json').exists()
    assert not (out/'evaluation/predictions.jsonl').exists()
    for path, digest in protocol['unchanged_v5_sources'].items(): assert sha256(Path(path)) == digest
    sources = ['src/clickagent/attention_field_ranker.py', 'scripts/attention_field_data.py',
        'scripts/train_attention_field_ranker.py', 'scripts/evaluate_attention_field_ranker.py',
        'src/clickagent/candidate_ranker.py', 'scripts/ranker_data.py', 'src/clickagent/spatial_geometry.py',
        'scripts/evaluate_conditioned_readouts.py', str(out/'protocol.json')]
    identity = dict(source_sha256={s: sha256(Path(s)) for s in sources},
        v5_prediction_sha256=sha256(base/'evaluation/predictions.jsonl'))
    if (out/'training_identity.json').exists(): assert json.loads((out/'training_identity.json').read_text()) == identity
    write_json(out/'training_identity.json', identity)
    (out/'models').mkdir(exist_ok=True); (out/'source_snapshot').mkdir(exist_ok=True)
    for source in sources: shutil.copyfile(source, out/'source_snapshot'/Path(source).name)
    torch.set_num_threads(4); torch.backends.cudnn.benchmark = False
    train = load_records(out, 'train'); val = load_records(out, 'val')
    prior = np.array(json.loads(Path(protocol['head_prior']).read_text())['weights'], np.float32)
    prior = np.log1p(96*prior/max(float(prior.sum()), 1e-9))
    results = {j['name']: train_one(out, j['name'], j['mode'], j['seed'], train, val, protocol, prior, args.device)
               for j in protocol['jobs']}
    assert not (out/'fields_test_completion.json').exists()
    frozen = dict(primary_model='all', models={name: dict(file=name+'.pt', sha256=r['checkpoint_sha256'],
        validation=r['validation'], best_epoch=r['best_epoch'], parameters=r['parameters']) for name, r in results.items()},
        sources=identity['source_sha256'], test_labels_used_for_fitting=0,
        test_description='Same previously evaluated v5 test; diagnostic reuse, not fresh independent confirmation',
        protocol_sha256=sha256(out/'protocol.json'), v5_prediction_sha256=identity['v5_prediction_sha256'])
    write_json(out/'frozen_readouts.json', frozen); print('FROZEN '+json.dumps(frozen), flush=True)


if __name__ == '__main__': main()
