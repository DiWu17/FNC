"""V13 matched full-data training and scorer-only source-heldout CV."""
from functools import lru_cache
from pathlib import Path
import argparse
import copy
import json
import shutil
import time

import numpy as np
import torch

from clickagent.action_benchmark import sha256, write_json
from clickagent.internal_candidate_scorer import InternalCandidateScorerConfig
from clickagent.multiregion_scorer import MultiRegionScorer, top4_retention_loss, save_multiregion_scorer
from train_candidate_scorer import load_cache, subset, ranking_loss
from train_crossfit_scorer import epoch_plan, keyed_seed, state_identity
from crossfit_protocol import digest_object
from multiregion_v13_protocol import OUT, SEEDS, SCHEDULE_SEED, ARMS, BUDGETS, read, cases, sources


@lru_cache(maxsize=2)
def dataset(split, condition):
    index_path = OUT/'cache'/f'{split}_{condition}_index.json'
    index = read(index_path); assert index['complete']
    data = []
    for entry in index['cases']:
        path = Path(entry['path']); assert sha256(path) == entry['sha256'], str(path)
        data.append(load_cache(path) if entry['available'] else None)
    assert [r['sample_id'] for r in index['cases']] == [r['sample_id'] for r in cases(split)]
    return data


def loss_for(objective, output, labels):
    return ranking_loss(output, labels) if objective == 'top1' else top4_retention_loss(output, labels)


@torch.inference_mode()
def evaluate(model, data, positions, objective, device):
    model.eval(); hits = top4 = oracle = named = 0; losses = []
    for pos in positions:
        record = data[pos]
        if record is None:
            continue
        ix = record['pool_indices']['16']
        if not len(ix):
            continue
        out = model(**subset(record, ix, False, device))
        y = torch.as_tensor(record['labels'][ix], device=device)
        order = torch.argsort(out['scores'], descending=True, stable=True)
        hits += int(y[order[0]].item() > .5)
        top4 += int(bool((y[order[:4]] > .5).any()))
        oracle += int(bool((y > .5).any())); named += 1
        losses.append(float(loss_for(objective, out, y)))
    return dict(n=len(positions), named=named, hits=hits, top4_hits=top4, oracle=oracle,
                loss=float(np.mean(losses)) if losses else 0.)


def train_one(arm, suffix, device, held_source=None):
    assert not (OUT/'frozen.json').exists(), 'Already frozen'
    assert not list((OUT/'raw/global').glob('*.npz')), 'No fitting after fresh test forward'
    protocol = read(OUT/'protocol.json'); opt = protocol['optimization']
    condition, objective = arm.split('_'); seed = SEEDS[suffix]
    train_rows, val_rows = cases('train'), cases('val')
    source_names = sorted({r['dataset'] for r in train_rows})
    fold_index = source_names.index(held_source) if held_source else None
    name = f'{arm}_{suffix}' if held_source is None else f'{arm}_held{fold_index}_{suffix}'
    folder = OUT/('training' if held_source is None else 'cross_validation')
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder/(name+'.json')
    identity = {str(p): sha256(p) for p in sources()}
    input_identity = {str(OUT/'cache'/f'{split}_{condition}_index.json'):
        sha256(OUT/'cache'/f'{split}_{condition}_index.json') for split in ('train', 'val')}
    if dest.exists():
        old = read(dest)
        assert old['complete'] and old['source_identity'] == identity and old['input_identity'] == input_identity
        assert old['protocol_sha256'] == sha256(OUT/'protocol.json')
        return
    if held_source is None:
        train_ix = set(range(len(train_rows))); val_ix = list(range(len(val_rows))); excluded_cross_component = []
    else:
        held_components = {r['image_fold_group'] for r in train_rows if r['dataset'] == held_source}
        train_ix = {i for i, r in enumerate(train_rows) if r['dataset'] != held_source and r['image_fold_group'] not in held_components}
        excluded_cross_component = [r['sample_id'] for r in train_rows if r['dataset'] != held_source and r['image_fold_group'] in held_components]
        val_ix = [i for i, r in enumerate(val_rows) if r['dataset'] == held_source]
        assert train_ix and val_ix
    data, validation = dataset('train', condition), dataset('val', condition)
    torch.manual_seed(seed); np.random.seed(seed % 2**32)
    model = MultiRegionScorer(InternalCandidateScorerConfig(head_mode='full')).to(device)
    initial_sha = state_identity(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt['learning_rate'], weight_decay=opt['weight_decay'])
    ids = [r['sample_id'] for r in train_rows]
    history = []; started = time.perf_counter()
    for epoch in range(opt['epochs']+1):
        losses = []; schedule = []; steps = accumulated = 0
        if epoch:
            model.train(); optimizer.zero_grad(set_to_none=True)
            for pos, budget in epoch_plan(ids, SCHEDULE_SEED, epoch):
                if pos not in train_ix or data[pos] is None:
                    continue
                record = data[pos]; sid = ids[pos]
                indices = record['pool_indices'][str(budget)].copy()
                if not len(indices):
                    continue
                schedule.append([sid, budget])
                rng = np.random.default_rng(keyed_seed(SCHEDULE_SEED, epoch, sid, 'candidate_permutation'))
                rng.shuffle(indices)
                torch.manual_seed(keyed_seed(SCHEDULE_SEED, epoch, sid, 'dropout'))
                output = model(**subset(record, indices, False, device))
                labels = torch.as_tensor(record['labels'][indices], device=device)
                loss = loss_for(objective, output, labels)
                assert torch.isfinite(loss), name
                (loss/opt['accumulation']).backward(); accumulated += 1
                losses.append(float(loss.detach()))
                if accumulated == opt['accumulation']:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), opt['gradient_clip'])
                    optimizer.step(); optimizer.zero_grad(set_to_none=True); steps += 1; accumulated = 0
            if accumulated:
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.mul_(opt['accumulation']/accumulated)
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt['gradient_clip'])
                optimizer.step(); optimizer.zero_grad(set_to_none=True); steps += 1
        metrics = evaluate(model, validation, val_ix, objective, device)
        row = dict(epoch=epoch, validation=metrics, train_loss=float(np.mean(losses)) if losses else None,
            queries=len(losses), optimizer_steps=steps, schedule_sha256=digest_object(schedule),
            elapsed_seconds=time.perf_counter()-started)
        history.append(row)
        if held_source is None:
            ckpt = folder/(name+f'_epoch_{epoch:02}.pt')
            save_multiregion_scorer(ckpt, model, dict(arm=arm, seed=seed, epoch=epoch,
                initial_state_sha256=initial_sha, protocol_sha256=sha256(OUT/'protocol.json')))
            row['checkpoint_path'] = str(ckpt); row['checkpoint_sha256'] = sha256(ckpt)
        write_json(folder/(name+'_progress.json'), dict(name=name, latest=row, history=history))
        print(json.dumps(dict(name=name, **row)), flush=True)
    assert identity == {str(p): sha256(p) for p in sources()}, 'Source changed during fitting'
    write_json(dest, dict(complete=True, name=name, arm=arm, seed=seed, schedule_seed=SCHEDULE_SEED, held_source=held_source,
        train_sample_ids=[ids[i] for i in sorted(train_ix)],
        validation_sample_ids=[val_rows[i]['sample_id'] for i in val_ix],
        excluded_cross_source_component_ids=excluded_cross_component,
        upstream_seen_held_source=bool(held_source), held_source_labels_used_for_gradient=False,
        history=history, parameters=sum(p.numel() for p in model.parameters()),
        initial_state_sha256=initial_sha, source_identity=identity, input_identity=input_identity,
        protocol_sha256=sha256(OUT/'protocol.json'), elapsed_seconds=time.perf_counter()-started))


def select():
    assert not (OUT/'frozen.json').exists()
    folder = OUT/'models'; folder.mkdir(exist_ok=True)
    selections = {}; cv_summary = {}
    for objective in ('top1', 'top4'):
        folds = [read(OUT/'cross_validation'/f'c3_{objective}_held{f}_s1.json') for f in range(4)]
        assert all(r['complete'] and not r['held_source_labels_used_for_gradient'] for r in folds)
        curves = []
        for epoch in range(read(OUT/'protocol.json')['optimization']['epochs']+1):
            values = [f['history'][epoch]['validation'] for f in folds]
            field = 'hits' if objective == 'top1' else 'top4_hits'
            curves.append(dict(epoch=epoch, macro_rate=float(np.mean([v[field]/v['n'] for v in values])),
                mean_objective_loss=float(np.mean([v['loss'] for v in values])), folds=values))
        selected = max(curves, key=lambda r: (r['macro_rate'], -r['mean_objective_loss'], -r['epoch']))
        cv_summary[objective] = dict(selected_epoch=selected['epoch'], curve=curves,
            interpretation='source-heldout CV tuning only; upstream saw domains and original validation; not independent test')
    for arm in ARMS:
        objective = arm.split('_')[1]; field = 'hits' if objective == 'top1' else 'top4_hits'
        for suffix in SEEDS:
            training = read(OUT/'training'/f'{arm}_{suffix}.json'); assert training['complete']
            best = max(training['history'], key=lambda r: (r['validation'][field], -r['validation']['loss'], -r['epoch']))
            policies = {'val': best['epoch']}
            if arm.startswith('c3'):
                policies['loso'] = cv_summary[objective]['selected_epoch']
            for policy, epoch in policies.items():
                row = training['history'][epoch]; source = Path(row['checkpoint_path'])
                assert sha256(source) == row['checkpoint_sha256']
                name = f'{arm}_{suffix}_{policy}'; target = folder/(name+'.pt')
                shutil.copyfile(source, target)
                selections[name] = dict(name=name, condition=arm[:2], objective=objective, seed=SEEDS[suffix],
                    policy=policy, epoch=epoch, checkpoint_path=str(target), sha256=sha256(target),
                    source_checkpoint_path=str(source), validation=row['validation'],
                    training_identity=sha256(OUT/'training'/f'{arm}_{suffix}.json'))
    # Every control starts with the same state within each seed.
    for suffix in SEEDS:
        assert len({read(OUT/'training'/f'{arm}_{suffix}.json')['initial_state_sha256'] for arm in ARMS}) == 1
    write_json(folder/'selection.json', dict(complete=True, selections=selections, cross_validation=cv_summary,
        fresh_test_read=False, seed_selected_by_test=False, protocol_sha256=sha256(OUT/'protocol.json')))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--stage', choices=('full', 'cv', 'select'), required=True)
    p.add_argument('--seed', choices=tuple(SEEDS), default='s1'); p.add_argument('--device', default='cuda:0')
    p.add_argument('--objective', choices=('top1', 'top4')); p.add_argument('--arm', choices=ARMS)
    args = p.parse_args(); torch.set_num_threads(4)
    if args.stage == 'select':
        select()
    elif args.stage == 'cv':
        assert args.objective
        for source in sorted({r['dataset'] for r in cases('train')}):
            train_one('c3_'+args.objective, 's1', args.device, source)
    else:
        for arm in [args.arm] if args.arm else ARMS:
            train_one(arm, args.seed, args.device)
