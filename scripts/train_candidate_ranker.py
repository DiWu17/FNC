"""Fit candidate ranking on old train/val, then freeze before fresh features."""
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
from clickagent.candidate_ranker import (RankerConfig, SemanticCandidateRanker, decode_ranker,
    load_ranker, make_ranker_batch, ranker_loss, save_ranker, subset_candidates)
from evaluate_conditioned_readouts import score_point
from ranker_data import load_records
from run_target_token_experiment import key, read_jsonl


@torch.inference_mode()
def evaluate(model, records, device, batch_size=8):
    model.eval(); named = [r for r in records if r['metadata']['target_available']]
    hits = 0; centers = 0; nulls = 0; approved_hits = 0; covered = 0; region = {str(k): 0 for k in (1, 3, 5)}; losses = []
    for start in range(0, len(named), batch_size):
        group = named[start:start+batch_size]; inputs, labels = make_ranker_batch(group, device)
        output = model(**inputs); loss, _ = ranker_loss(output, **labels); losses.extend([float(loss)]*len(group))
        cpu = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in output.items()}
        cursor = 0
        for i, r in enumerate(group):
            proposals = decode_ranker(cpu, r, i); center = decode_ranker(cpu, r, i, mode='center', top_k=1)[0]['point']
            hit = score_point(proposals[0]['point'], r['case']); hits += hit; centers += score_point(center, r['case'])
            positive = r['label_area'].sum(-1) > 0; covered += bool(positive.any())
            for k in region: region[k] += any(positive[p['candidate_index']] for p in proposals[:int(k)])
            n = len(r['parent_indices']); rejected = float(cpu['null_scores'][i]) >= float(cpu['scores'][cursor:cursor+n].max())
            nulls += rejected; approved_hits += hit and not rejected; cursor += n
    return dict(n=len(records), named=len(named), hits=int(hits), accuracy=hits/len(records), center_hits=int(centers),
        region_hits={k: int(v) for k, v in region.items()}, candidate_region_coverage=int(covered),
        selection_accuracy_given_covered=region['1']/covered if covered else None,
        null_rejections=int(nulls), null_policy_hits=int(approved_hits), loss=float(np.mean(losses)))


def train_one(out, name, variant, seed, train_all, val, protocol, prior, device):
    path = out/'models'/f'{name}.pt'; history_path = out/'models'/f'{name}_history.json'
    if history_path.exists():
        saved = json.loads(history_path.read_text(encoding='utf-8'))
        assert saved['complete'] and saved['checkpoint_sha256'] == sha256(path)
        return saved
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); np.random.seed(seed); random.seed(seed)
    rng = np.random.default_rng(seed)
    model = SemanticCandidateRanker(RankerConfig(protocol['width'], variant), prior).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=protocol['learning_rate'], weight_decay=protocol['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, protocol['max_epochs'], eta_min=protocol['learning_rate']*.1)
    train = [r for r in train_all if r['metadata']['target_available']]
    history = []; best_rank = None; best_epoch = 0; started = time.perf_counter()
    for epoch in range(protocol['max_epochs']+1):
        train_loss = []; removed = 0
        if epoch:
            model.train(); order = rng.permutation(len(train))
            for start in range(0, len(order), protocol['batch_size']):
                group = []
                for index in order[start:start+protocol['batch_size']]:
                    r = train[index]; positive = r['label_area'].sum(-1) > 0
                    if positive.any() and not positive.all() and rng.random() < protocol['candidate_removal_probability']:
                        r = subset_candidates(r, ~positive); removed += 1
                    group.append(r)
                inputs, labels = make_ranker_batch(group, device); optimizer.zero_grad(set_to_none=True)
                output = model(**inputs); loss, _ = ranker_loss(output, **labels)
                if not torch.isfinite(loss): raise ValueError('nonfinite candidate loss')
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); optimizer.step()
                train_loss.append(float(loss.detach()))
            scheduler.step()
        metrics = evaluate(model, val, device, protocol['batch_size'])
        row = dict(epoch=epoch, val=metrics, elapsed_seconds=time.perf_counter()-started,
                   train_counterfactual_lists=removed, train_loss=float(np.mean(train_loss)) if train_loss else None)
        history.append(row); rank = (metrics['region_hits']['1'], metrics['hits'], -metrics['loss'])
        if best_rank is None or rank > best_rank:
            best_rank = rank; best_epoch = epoch
            save_ranker(path, model, dict(name=name, variant=variant, epoch=epoch, seed=seed, validation=metrics,
                                          train_named=len(train), backbone_trainable_parameters=0))
        write_json(out/'models'/f'{name}_progress.json', dict(history=history, best_epoch=best_epoch))
        print(name+' '+json.dumps(row), flush=True)
        if epoch >= protocol['min_epochs'] and epoch-best_epoch >= protocol['patience']: break
    selected, _ = load_ranker(path, device)
    result = dict(complete=True, name=name, variant=variant, best_epoch=best_epoch, seed=seed,
        validation=history[best_epoch]['val'], training=evaluate(selected, train_all, device), history=history,
        checkpoint_sha256=sha256(path), parameters=sum(p.numel() for p in model.parameters()))
    write_json(history_path, result)
    print('SELECTED '+name+' '+json.dumps({k: v for k, v in result.items() if k != 'history'}), flush=True)
    del model, selected, optimizer; gc.collect(); torch.cuda.empty_cache()
    return result


def main():
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, default=Path('runs/candidate_ranker_v5')); p.add_argument('--device', default='cuda:0')
    args = p.parse_args(); out = args.output
    if (out/'frozen_readouts.json').exists(): raise ValueError('Already frozen, no refitting')
    protocol = json.loads((out/'protocol.json').read_text(encoding='utf-8'))
    assert protocol['manifest_sha256'] == sha256(out/'manifest.jsonl')
    test = [c for c in read_jsonl(out/'manifest.jsonl') if c['experiment_split'] == 'test']
    assert not any((out/'features'/f'{key(c["sample_id"])}.npz').exists() for c in test)
    sources = ['src/clickagent/candidate_ranker.py', 'scripts/ranker_data.py', 'scripts/prepare_ranker_experiment.py', str(Path(__file__)),
               'scripts/evaluate_candidate_ranker.py', 'src/clickagent/spatial_geometry.py',
               'src/clickagent/public_evaluation.py', 'scripts/evaluate_conditioned_readouts.py',
               'scripts/evaluate_phrase_readouts.py', 'src/clickagent/phrase_grounder.py', str(out/'protocol.json')]
    identity = dict(source_sha256={s: sha256(Path(s)) for s in sources}, manifest_sha256=sha256(out/'manifest.jsonl'),
        prior_sha256=sha256(Path('runs/phrase_v4/evaluation/head_token_profile/train_only_head_prior.json')))
    if (out/'training_identity.json').exists(): assert json.loads((out/'training_identity.json').read_text(encoding='utf-8')) == identity
    write_json(out/'training_identity.json', identity)
    (out/'models').mkdir(exist_ok=True); (out/'head_sources').mkdir(exist_ok=True)
    for source in sources: shutil.copyfile(source, out/'head_sources'/Path(source).name)
    torch.set_num_threads(4); torch.backends.cudnn.benchmark = False
    train = load_records(out, 'train'); val = load_records(out, 'val')
    prior = np.array(json.loads(Path('runs/phrase_v4/evaluation/head_token_profile/train_only_head_prior.json').read_text(encoding='utf-8'))['weights'], np.float32)
    prior = np.log1p(96*prior/max(float(prior.sum()), 1e-9))
    jobs = [(v, v, protocol['primary_seed']) for v in protocol['variants']]+[('full_replica', 'full', protocol['replica_seed'])]
    results = {name: train_one(out, name, variant, seed, train, val, protocol, prior, args.device) for name, variant, seed in jobs}
    assert not any((out/'features'/f'{key(c["sample_id"])}.npz').exists() for c in test)
    frozen = dict(primary_model='full', models={name: dict(file=name+'.pt', sha256=r['checkpoint_sha256'],
        validation=r['validation'], best_epoch=r['best_epoch'], parameters=r['parameters']) for name, r in results.items()},
        head_sources=identity['source_sha256'], test_features_at_freeze=0, test_labels_used_for_training=0,
        prior_source='Training-only prior from unchanged train split', test_n=len(test),
        ablation_scope='no_visual/no_query remove explicit readout inputs; the VLM still saw the image and generated target-conditioned attention')
    write_json(out/'frozen_readouts.json', frozen)
    print('FROZEN '+json.dumps(frozen), flush=True)


if __name__ == '__main__': main()
