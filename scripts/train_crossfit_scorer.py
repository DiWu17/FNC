"""Matched V11 scorer training on in-sample or strict cross-fit candidate pools."""
from __future__ import annotations
import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from clickagent.action_benchmark import sha256, write_json
from clickagent.internal_candidate_scorer import InternalCandidateScorer, InternalCandidateScorerConfig, save_internal_scorer, load_internal_scorer
from train_candidate_scorer import load_cache, subset, ranking_loss, evaluate
from crossfit_protocol import OUT, ARMS, SEEDS, BUDGETS, read, digest_object, source_files


def keyed_seed(seed, epoch, sample_id, purpose):
    value = f'{seed}|{epoch}|{sample_id}|{purpose}'
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], 'little') % (2**63 - 1)


def epoch_plan(sample_ids, seed, epoch):
    rng = np.random.default_rng(keyed_seed(seed, epoch, '', 'query_order'))
    return [(int(i), int(np.random.default_rng(keyed_seed(seed, epoch, sample_ids[int(i)], 'budget')).choice(BUDGETS)))
            for i in rng.permutation(len(sample_ids))]


def state_identity(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


@lru_cache(maxsize=2)
def dataset(group):
    index = read(OUT / 'cache' / (group + '_index.json')); assert index['complete']
    result = []
    for row in index['cases']:
        p = Path(row['path']); assert sha256(p) == row['sha256'], str(p)
        result.append(load_cache(p) if row['available'] else None)
    return result


def eligibility():
    indices = {a: read(OUT / 'cache' / (a + '_index.json')) for a in ARMS}
    left, right = [indices[a]['cases'] for a in ARMS]
    assert [r['sample_id'] for r in left] == [r['sample_id'] for r in right]
    assert [r['available'] for r in left] == [r['available'] for r in right], 'Training query eligibility must match'
    return [r['sample_id'] for r in left], [r['available'] for r in left]


def train(arm, suffix, device):
    assert not (OUT / 'frozen.json').exists()
    assert not list((OUT / 'test').rglob('*.npz'))
    protocol = read(OUT / 'protocol.json'); opt = protocol['optimization']
    seed = SEEDS[0 if suffix == 's1' else 1]; name = arm + '_' + suffix
    dest = OUT / 'models' / (name + '.pt'); dest.parent.mkdir(parents=True, exist_ok=True)
    source_identity = {str(p): sha256(p) for p in source_files()}
    identities = {g: sha256(OUT / 'cache' / (g + '_index.json')) for g in (arm, 'val')}
    if dest.with_suffix('.json').exists():
        old = read(dest.with_suffix('.json'))
        assert old['complete'] and old['source_identity'] == source_identity and old['input_identity'] == identities
        return
    ids, eligible = eligibility()
    train_data = dataset(arm); val_data = dataset('val')
    for flag, record in zip(eligible, train_data):
        assert flag == (record is not None)
        if record is not None:
            assert all(len(record['pool_indices'][str(k)]) > 0 for k in BUDGETS)
    torch.manual_seed(seed); np.random.seed(seed % 2**32)
    model = InternalCandidateScorer(InternalCandidateScorerConfig(head_mode='full')).to(device)
    init_sha = state_identity(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt['learning_rate'], weight_decay=opt['weight_decay'])
    best = None; history = []; started = time.perf_counter(); selected = None
    for epoch in range(opt['epochs'] + 1):
        losses = []; steps = 0; schedule = []; accumulated = 0
        if epoch:
            model.train(); optimizer.zero_grad(set_to_none=True)
            for position, budget in epoch_plan(ids, seed, epoch):
                record = train_data[position]
                if record is None:
                    continue
                sid = ids[position]; schedule.append([sid, budget])
                indices = record['pool_indices'][str(budget)].copy()
                rng = np.random.default_rng(keyed_seed(seed, epoch, sid, 'candidate_permutation'))
                rng.shuffle(indices)
                torch.manual_seed(keyed_seed(seed, epoch, sid, 'dropout'))
                output = model(**subset(record, indices, False, device))
                labels = torch.as_tensor(record['labels'][indices], device=device)
                loss = ranking_loss(output, labels)
                assert torch.isfinite(loss), name
                (loss / opt['accumulation']).backward(); accumulated += 1
                losses.append(float(loss.detach()))
                if accumulated == opt['accumulation']:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), opt['gradient_clip'])
                    optimizer.step(); optimizer.zero_grad(set_to_none=True); steps += 1; accumulated = 0
            if accumulated:
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.mul_(opt['accumulation'] / accumulated)
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt['gradient_clip'])
                optimizer.step(); optimizer.zero_grad(set_to_none=True); steps += 1
        validation = evaluate(model, val_data, False, device)
        row = dict(epoch=epoch, validation=validation, train_loss=float(np.mean(losses)) if losses else None,
                   optimizer_steps=steps, queries=len(losses), schedule_sha256=digest_object(schedule),
                   elapsed_seconds=time.perf_counter()-started)
        history.append(row); selector = (validation['hits'], -validation['loss'])
        if best is None or selector > best:
            best = selector; selected = row
            metadata = dict(arm=arm, seed=seed, selected=selected, initial_state_sha256=init_sha,
                protocol_sha256=sha256(OUT / 'protocol.json'), source_identity=source_identity, input_identity=identities,
                train_n=len(train_data), validation_n=len(val_data), architecture_unchanged=True, loss_unchanged=True,
                training_uses_gui_actor=False, annotation_derived_points=False)
            save_internal_scorer(dest, model, metadata)
        write_json(dest.with_name(name + '_progress.json'), dict(name=name, selected=selected, latest=row))
        print(json.dumps(dict(name=name, **row)), flush=True)
    assert source_identity == {str(p): sha256(p) for p in source_files()}, 'Sources changed during training'
    saved, _ = load_internal_scorer(dest, device=device)
    training = evaluate(saved, train_data, False, device); validation = evaluate(saved, val_data, False, device)
    assert all(validation[k] == selected['validation'][k] for k in ('n', 'named', 'hits', 'oracle', 'refused'))
    assert np.isclose(validation['loss'], selected['validation']['loss'], rtol=1e-5, atol=1e-6)
    write_json(dest.with_suffix('.json'), dict(complete=True, arm=arm, seed=seed, selected=selected, history=history,
        checkpoint_sha256=sha256(dest), parameters=sum(p.numel() for p in model.parameters()), initial_state_sha256=init_sha,
        protocol_sha256=sha256(OUT / 'protocol.json'), source_identity=source_identity, input_identity=identities,
        selected_training_evaluation=training, selected_validation_replay=validation,
        training_seconds=time.perf_counter()-started))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--arm', choices=ARMS, required=True)
    p.add_argument('--seed', choices=['s1', 's2'], required=True); p.add_argument('--device', default='cuda:0')
    args = p.parse_args(); torch.set_num_threads(4); train(args.arm, args.seed, args.device)
