"""Controlled early routing, point selection and same-image hard-example training."""
import argparse
from collections import defaultdict
import gc
import json
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
import torch.nn.functional as F

from clickagent.action_benchmark import sha256, write_json
from clickagent.candidate_ranker import decode_ranker, ranker_loss
from clickagent.point_quality_readout import quality_loss
from clickagent.semantic_head_experiment import build_readout, load_readout, save_readout, score_output
from evaluate_conditioned_readouts import score_point
from semantic_head_data import load_records, make_batch, subset_record
from run_target_token_experiment import key, read_jsonl


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def hard_negative_loss(output, area, hard_mask):
    cursor = 0; losses = []
    for n in output['candidate_lengths']:
        pos = area[cursor:cursor+n].sum(-1) > 0; neg = hard_mask[cursor:cursor+n] > 0
        scores = output['scores'][cursor:cursor+n]
        if bool(pos.any()) and bool(neg.any()):
            p = torch.logsumexp(scores[pos], 0)-pos.sum().float().log()
            q = torch.logsumexp(scores[neg], 0)-neg.sum().float().log()
            losses.append(F.softplus(.5+q-p))
        cursor += n
    return torch.stack(losses).mean() if losses else output['scores'].sum()*0


def epoch_order(train, rng, recipe, budget):
    chosen = rng.permutation(len(train))[:budget]
    if recipe == 'random': return chosen
    if recipe != 'grouped_hard': raise ValueError('Unknown data recipe')
    groups = defaultdict(list)
    for i in chosen: groups[train[int(i)]['case']['pixel_sha256']].append(int(i))
    keys = list(groups); order = []
    for gi in rng.permutation(len(keys)):
        items = groups[keys[int(gi)]]
        order.extend(np.asarray(items)[rng.permutation(len(items))].tolist())
    return np.asarray(order, np.int64)


@torch.inference_mode()
def evaluate(model, records, device, batch_size=8):
    model.eval(); named = [r for r in records if r['metadata']['target_available']]
    totals = {mode: dict(hits=0, region=0, own_hits=0, center_hits=0) for mode in ['rank','combined','quality']}
    covered = 0; point_covered = 0; rank_losses = []; quality_losses = []
    for start in range(0, len(named), batch_size):
        group = named[start:start+batch_size]; inputs, target = make_batch(group, device)
        output = model(**inputs)
        rank_loss, _ = ranker_loss(output, target['label_area'], target['label_offsets'])
        qloss = quality_loss(output['point_quality_logits'], target['label_fixed_point_hit'], output['candidate_lengths'])
        rank_losses.extend([float(rank_loss)]*len(group)); quality_losses.extend([float(qloss)]*len(group))
        cpu = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in output.items()}
        for i, r in enumerate(group):
            positive = r['label_area'].sum(-1) > 0; covered += bool(positive.any()); point_covered += bool(r['label_fixed_point_hit'].any())
            for mode in totals:
                result = decode_ranker(score_output(cpu, mode), r, i, top_k=1)[0]; ci = result['candidate_index']
                totals[mode]['hits'] += int(r['label_fixed_point_hit'][ci] > 0)
                totals[mode]['region'] += int(positive[ci])
                totals[mode]['own_hits'] += score_point(result['point'], r['case'])
                center = decode_ranker(score_output(cpu, mode), r, i, mode='center', top_k=1)[0]['point']
                totals[mode]['center_hits'] += score_point(center, r['case'])
    return dict(n=len(records), named=len(named), modes=totals, candidate_region_coverage=covered,
        frozen_fine_point_oracle=point_covered, rank_loss=float(np.mean(rank_losses)), quality_loss=float(np.mean(quality_losses)))


def selectors(metrics):
    r, q = metrics['modes']['rank'], metrics['modes']['combined']
    return dict(region=(r['region'], r['hits'], -metrics['rank_loss']),
        point=(r['hits'], r['region'], -metrics['rank_loss']),
        quality=(q['hits'], q['region'], -metrics['quality_loss']))


def train_one(out, job, train_all, val, protocol, prior, device):
    name = job['name']; history_path = out/'models'/f'{name}_history.json'
    if history_path.exists():
        result = read(history_path); assert result['complete']
        for entry in result['checkpoints'].values(): assert sha256(out/'models'/entry['file']) == entry['sha256']
        return result
    seed = job['seed']; torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); np.random.seed(seed); random.seed(seed)
    model = build_readout(job['architecture'], prior, protocol['width']).to(device)
    core_params = list(model.core.parameters()); quality_params = list(model.quality.parameters())
    assert sum(p.numel() for p in core_params+quality_params) == sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW([dict(params=core_params), dict(params=quality_params)],
        lr=protocol['learning_rate'], weight_decay=protocol['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, protocol['epochs'], eta_min=protocol['learning_rate']*.1)
    train = [r for r in train_all if r['metadata']['target_available']]
    budget = protocol['training_queries_per_epoch']; assert len(train) >= budget
    history = []; best = {}; chosen = {}; started = time.perf_counter()
    for epoch in range(protocol['epochs']+1):
        losses = []; hard_used = 0; removed = 0; consumed_queries = 0; optimizer_steps = 0
        if epoch:
            model.train()
            epoch_rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, 0]))
            augmentation = np.random.default_rng(np.random.SeedSequence([seed, epoch, 1])).random(len(train))
            order = epoch_order(train, epoch_rng, job['recipe'], budget)
            for start in range(0, len(order), protocol['batch_size']):
                group = []
                for index in order[start:start+protocol['batch_size']]:
                    r = train[index]; positive = r['label_area'].sum(-1) > 0
                    if positive.any() and not positive.all() and augmentation[int(index)] < protocol['candidate_removal_probability']:
                        r = subset_record(r, ~positive); removed += 1
                    group.append(r)
                inputs, target = make_batch(group, device); optimizer.zero_grad(set_to_none=True)
                output = model(**inputs)
                rloss, _ = ranker_loss(output, target['label_area'], target['label_offsets'])
                qloss = quality_loss(output['point_quality_logits'], target['label_fixed_point_hit'], output['candidate_lengths'])
                hloss = hard_negative_loss(output, target['label_area'], target['label_hard_negative'])
                if job['recipe'] == 'grouped_hard':
                    rloss = rloss+protocol['hard_negative_weight']*hloss; hard_used += int(float(hloss.detach()) > 0)
                loss = rloss+qloss
                if not torch.isfinite(loss): raise ValueError('Nonfinite readout training loss')
                loss.backward()
                # Separate clipping preserves core training independently of BCE.
                torch.nn.utils.clip_grad_norm_(core_params, 1.); torch.nn.utils.clip_grad_norm_(quality_params, 1.)
                optimizer.step(); losses.append(float(loss.detach()))
                consumed_queries += len(group); optimizer_steps += 1
            scheduler.step()
            assert consumed_queries == budget and optimizer_steps == protocol['updates_per_epoch']
        metrics = evaluate(model, val, device, protocol['batch_size'])
        row = dict(epoch=epoch, val=metrics, train_loss=float(np.mean(losses)) if losses else None,
            counterfactual_lists=removed, hard_negative_batches=hard_used, consumed_queries=consumed_queries,
            optimizer_steps=optimizer_steps, elapsed_seconds=time.perf_counter()-started)
        history.append(row)
        for rule, rank in selectors(metrics).items():
            if rule not in best or rank > best[rule]:
                best[rule] = rank; chosen[rule] = epoch
                save_readout(out/'models'/f'{name}__{rule}.pt', model, job['architecture'],
                    dict(job=job, epoch=epoch, selector=rule, validation=metrics, backbone_trainable_parameters=0))
        write_json(out/'models'/f'{name}_progress.json', dict(best_epochs=chosen, history=history))
        print(name+' '+json.dumps(row), flush=True)
    checkpoints = {}
    for rule in ['region','point','quality']:
        path = out/'models'/f'{name}__{rule}.pt'; selected, _ = load_readout(path, device)
        checkpoints[rule] = dict(file=path.name, sha256=sha256(path), epoch=chosen[rule],
            validation=history[chosen[rule]]['val'], training=evaluate(selected, train_all, device))
        del selected
    result = dict(complete=True, job=job, history=history, checkpoints=checkpoints,
        parameters=sum(p.numel() for p in model.parameters()), core_parameters=sum(p.numel() for p in core_params),
        quality_parameters=sum(p.numel() for p in quality_params), fit_named_queries=len(train), updates_per_epoch=budget//protocol['batch_size'])
    write_json(history_path, result)
    print('SELECTED '+name+' '+json.dumps(dict(epochs=chosen, seconds=time.perf_counter()-started)), flush=True)
    del model, optimizer; gc.collect(); torch.cuda.empty_cache(); return result


def source_identity(out):
    sources = ['src/clickagent/semantic_head_ranker.py','src/clickagent/semantic_head_experiment.py',
        'src/clickagent/point_quality_readout.py','src/clickagent/attention_field_ranker.py',
        'src/clickagent/candidate_ranker.py','scripts/semantic_head_data.py','scripts/train_semantic_head_experiment.py',
        'scripts/evaluate_semantic_head_experiment.py','scripts/evaluate_conditioned_readouts.py',
        'scripts/ranker_data.py','scripts/attention_field_data.py',str(out/'protocol.json')]
    return {p:sha256(Path(p)) for p in sources}


def cache_inventory(out):
    inventory = {}
    for case in read_jsonl(out/'manifest.jsonl'):
        if case['experiment_split'] not in ['train', 'val']: continue
        path = out/'point_cache'/f'{key(case["sample_id"])}.npz'
        with np.load(path, allow_pickle=False) as z: meta = json.loads(str(z['metadata']))
        assert meta['sample_id'] == case['sample_id']
        inventory[case['sample_id']] = dict(point_cache_sha256=sha256(path), input_identity=meta['input_identity'])
    return inventory


def freeze(out):
    protocol = read(out/'protocol.json'); sources = source_identity(out)
    identity = read(out/'training_identity.json')
    assert identity['sources'] == sources, 'Training sources changed before freeze'
    assert sha256(out/'manifest.jsonl') == identity['manifest_sha256'] == protocol['manifest_sha256']
    assert identity['train_val_cache_inventory'] == cache_inventory(out)
    for path, digest in protocol['unchanged_artifact_sha256'].items(): assert sha256(Path(path)) == digest, path
    for path, digest in protocol['baseline_checkpoint_sha256'].items(): assert sha256(Path(path)) == digest, path
    models = {}
    for job in protocol['jobs']:
        result = read(out/'models'/f'{job["name"]}_history.json'); assert result['complete']
        assert result['job'] == job
        assert [r['epoch'] for r in result['history']] == list(range(protocol['epochs']+1))
        for row in result['history']:
            assert row['consumed_queries'] == (protocol['training_queries_per_epoch'] if row['epoch'] else 0)
            assert row['optimizer_steps'] == (protocol['updates_per_epoch'] if row['epoch'] else 0)
        assert set(result['checkpoints']) == set(protocol['checkpoint_selectors'])
        for rule, entry in result['checkpoints'].items():
            assert sha256(out/'models'/entry['file']) == entry['sha256']
            ranks = [selectors(r['val'])[rule] for r in result['history']]
            assert entry['epoch'] == max(range(len(ranks)), key=lambda i: ranks[i])
            assert entry['validation'] == result['history'][entry['epoch']]['val']
            models[job['name']+'__'+rule] = dict(entry, job=job, selector=rule)
    assert not any((out/'features'/f'{key(c["sample_id"])}.npz').exists() for c in read_jsonl(out/'manifest.jsonl') if c['experiment_split']=='test')
    write_json(out/'frozen_readouts.json', dict(primary_model=protocol['primary'], models=models, sources=sources,
        fresh_test_features_at_freeze=0, fresh_test_used_for_training_or_selection=False,
        manifest_sha256=protocol['manifest_sha256'], fixed_fine_sha256=protocol['fixed_fine_sha256']))


def main():
    p = argparse.ArgumentParser(); p.add_argument('--output',type=Path,default=Path('runs/semantic_head_v7'))
    p.add_argument('--device',default='cuda:0'); p.add_argument('--seed',type=int); p.add_argument('--freeze-only',action='store_true')
    args=p.parse_args(); out=args.output; torch.set_num_threads(4)
    if (out/'frozen_readouts.json').exists(): raise ValueError('Readouts already frozen')
    if args.freeze_only: freeze(out); print('FROZEN'); return
    protocol=read(out/'protocol.json'); assert sha256(out/'manifest.jsonl')==protocol['manifest_sha256']
    assert not (out/'features_test_completion.json').exists()
    sources=source_identity(out); identity_path=out/'training_identity.json'
    identity=dict(sources=sources,manifest_sha256=protocol['manifest_sha256'],train_val_cache_inventory=cache_inventory(out))
    if identity_path.exists(): assert read(identity_path)==identity
    else:
        write_json(identity_path,identity)
        snapshot=out/'source_snapshot';snapshot.mkdir(exist_ok=True)
        for source in sources: shutil.copyfile(source,snapshot/Path(source).name)
    (out/'models').mkdir(exist_ok=True)
    val=load_records(out,'val'); expanded=load_records(out,'train','expanded')
    base=[r for r in expanded if not r['case'].get('training_extension')]
    assert len(base)==protocol['train_queries_base'] and len(expanded)==protocol['train_queries_expanded']
    prior=np.array(read(protocol['head_prior'])['weights'],np.float32);prior=np.log1p(96*prior/max(float(prior.sum()),1e-9))
    for job in protocol['jobs']:
        if args.seed is None or job['seed']==args.seed:
            train_one(out,job,base if job['data']=='base' else expanded,val,protocol,prior,args.device)


if __name__=='__main__':
    main()
