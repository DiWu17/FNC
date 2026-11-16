"""Train V11 heads on real candidate sets; original validation selects epochs."""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from clickagent.action_benchmark import sha256, write_json
from clickagent.internal_candidate_scorer import (
    InternalCandidateScorer, InternalCandidateScorerConfig, internal_inputs_to_torch,
    subset_internal_inputs, save_internal_scorer, load_internal_scorer,
)
from clickagent.visual_candidate_verifier import (
    VisualCandidateVerifier, VisualVerifierConfig, visual_inputs_to_torch,
    subset_visual_inputs, save_visual_verifier, load_visual_verifier,
)
from candidate_scorer_protocol import OUT, SEEDS, VARIANTS, SOURCES
from point_aligned_data import read


def load_cache(path, visual=False):
    prefix = 'v__' if visual else 'i__'
    with np.load(path, allow_pickle=False) as z:
        inputs = {k[len(prefix):]: z[k].copy() for k in z.files if k.startswith(prefix)}
        return dict(inputs=inputs, labels=z['labels'].astype(np.float32) if 'labels' in z.files else None,
                    pool_indices={str(k): z['pool_indices_' + str(k)].astype(np.int64) for k in (4, 8, 16, 32)},
                    sample_id=str(z['sample_id']), points=z['points'].copy())


@lru_cache(maxsize=2)
def dataset(split, visual=False):
    # Train/val inputs are read-only. Reuse them across the three internal
    # variants within one seed's worker, then evict them for the visual route.
    # Cache files are SHA-checked on load and scanned again at model freeze.
    index = read(OUT / 'cache' / (split + '_index.json'))
    assert index['complete']
    if visual:
        assert index.get('visual_complete'), 'Dedicated visual features must be complete before training'
    data = []
    for row in index['cases']:
        if not row['available']:
            data.append(None)
        else:
            path = Path(row['visual_path'] if visual else row['path'])
            assert sha256(path) == row['visual_sha256' if visual else 'sha256'], str(path)
            data.append(load_cache(path, visual))
    return data


def input_identity(visual):
    result = {}
    for split in ('train', 'val'):
        index = read(OUT / 'cache' / (split + '_index.json'))
        rows = [(r['sample_id'], r['available'], r['visual_sha256' if visual else 'sha256']) for r in index['cases']]
        result[split] = hashlib.sha256(json.dumps(rows, separators=(',', ':')).encode()).hexdigest()
    return result


def subset(record, indices, visual, device):
    func = subset_visual_inputs if visual else subset_internal_inputs
    convert = visual_inputs_to_torch if visual else internal_inputs_to_torch
    return convert(func(record['inputs'], indices), device=device)


def ranking_loss(output, labels):
    """All labels describe fixed proposed points; no annotation-derived points."""
    scores = output['scores'].float(); null = output['null_scores'].float().reshape(1)
    positive = labels > .5
    values = torch.cat((scores, null))
    targets = torch.cat((positive, (~positive.any()).reshape(1)))
    rank = torch.logsumexp(values, 0) - torch.logsumexp(values[targets], 0)
    element = F.binary_cross_entropy_with_logits(scores, labels, reduction='none')
    if positive.any() and (~positive).any():
        bce = .5 * element[positive].mean() + .5 * element[~positive].mean()
    else:
        bce = element.mean()
    return rank + .25 * bce


@torch.inference_mode()
def evaluate(model, data, visual, device):
    model.eval(); hits = oracle = named = refused = 0; losses = []
    for record in data:
        if record is None:
            continue
        ix = record['pool_indices']['16']
        if not len(ix):
            continue
        inputs = subset(record, ix, visual, device)
        output = model(**inputs)
        y = torch.as_tensor(record['labels'][ix], device=device)
        named += 1; oracle += int(bool(y.any()))
        hits += int(float(y[int(output['scores'].argmax())]) > 0)
        refused += int(float(output['null_scores'].reshape(-1)[0]) > float(output['scores'].max()))
        losses.append(float(ranking_loss(output, y)))
    return dict(n=len(data), named=named, hits=hits, oracle=oracle, refused=refused,
                loss=float(np.mean(losses)) if losses else 0.)


def make_model(variant):
    if variant == 'visual':
        return VisualCandidateVerifier(VisualVerifierConfig())
    mode = {'internal_full': 'full', 'internal_pooled': 'pooled_control', 'internal_global': 'no_local'}[variant]
    return InternalCandidateScorer(InternalCandidateScorerConfig(head_mode=mode))


def train(variant, suffix, device, epochs=None):
    assert not (OUT / 'frozen.json').exists(), 'No training after model freeze'
    assert not list((OUT / 'test/features').glob('*.npz')), 'Do not tune after independent test extraction'
    visual = variant == 'visual'; seed = SEEDS[0 if suffix == 's1' else 1]
    name = variant + '_' + suffix; dest = OUT / 'models' / (name + '.pt')
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.with_suffix('.json').exists():
        previous = read(dest.with_suffix('.json'))
        assert previous['protocol_sha256'] == sha256(OUT / 'protocol.json')
        assert previous['input_identity'] == input_identity(visual)
        assert previous['source_identity'] == {p: sha256(Path(p)) for p in SOURCES}
        print('ALREADY TRAINED ' + name, flush=True); return
    protocol = read(OUT / 'protocol.json'); opt = protocol['optimization']
    epochs = opt['epochs'] if epochs is None else epochs
    torch.manual_seed(seed); np.random.seed(seed % (2**32))
    model = make_model(variant).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt['learning_rate'], weight_decay=opt['weight_decay'])
    train_data = dataset('train', visual); val_data = dataset('val', visual)
    inputs_identity = input_identity(visual)
    sources_identity = {p: sha256(Path(p)) for p in SOURCES}
    best = None; history = []; started = time.perf_counter()
    for epoch in range(epochs + 1):
        losses = []
        if epoch:
            model.train(); rng = np.random.default_rng(seed + epoch)
            order = rng.permutation(len(train_data)); optimizer.zero_grad(set_to_none=True); accumulated = 0
            for position in order:
                record = train_data[int(position)]
                if record is None:
                    continue
                budget = int(rng.choice((4, 8, 16, 32)))
                indices = record['pool_indices'][str(budget)].copy()
                if not len(indices):
                    continue
                rng.shuffle(indices)
                output = model(**subset(record, indices, visual, device))
                labels = torch.as_tensor(record['labels'][indices], device=device)
                loss = ranking_loss(output, labels)
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite training loss ' + name)
                (loss / opt['accumulation']).backward(); accumulated += 1
                losses.append(float(loss.detach()))
                if accumulated == opt['accumulation']:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), opt['gradient_clip'])
                    optimizer.step(); optimizer.zero_grad(set_to_none=True); accumulated = 0
            if accumulated:
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.mul_(opt['accumulation'] / accumulated)
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt['gradient_clip'])
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
        validation = evaluate(model, val_data, visual, device)
        row = dict(epoch=epoch, validation=validation,
                   train_loss=float(np.mean(losses)) if losses else None,
                   elapsed_seconds=time.perf_counter()-started)
        history.append(row); selector = (validation['hits'], -validation['loss'])
        if best is None or selector > best:
            best = selector; selected = row
            metadata = dict(variant=variant, seed=seed, selected=selected,
                            train_n=len(train_data), validation_n=len(val_data), primary_budget=16,
                            pretrained_backbones_frozen=True, training_uses_gui_actor=False,
                            real_candidate_pools=True, annotation_derived_points=False,
                            input_identity=inputs_identity, source_identity=sources_identity,
                            protocol_sha256=sha256(OUT / 'protocol.json'))
            cpu_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if visual:
                save_visual_verifier(dest, model, metadata)
            else:
                save_internal_scorer(dest, model, metadata)
            # Save helpers must preserve the model's device and training state.
            assert all(torch.equal(v.detach().cpu(), cpu_state[k]) for k, v in model.state_dict().items())
        write_json(OUT / 'models' / (name + '_progress.json'), dict(name=name, selected=selected, latest=row))
        print(json.dumps(dict(name=name, **row)), flush=True)
    assert sources_identity == {p: sha256(Path(p)) for p in SOURCES}, 'Training sources changed mid-run'
    loader = load_visual_verifier if visual else load_internal_scorer
    selected_model, _ = loader(dest, device=device)
    selected_train = evaluate(selected_model, train_data, visual, device)
    selected_validation = evaluate(selected_model, val_data, visual, device)
    assert selected_validation['hits'] == selected['validation']['hits']
    write_json(dest.with_suffix('.json'), dict(complete=True, selected=selected, history=history,
        checkpoint_sha256=sha256(dest), parameters=sum(p.numel() for p in model.parameters()),
        variant=variant, seed=seed, training_seconds=time.perf_counter()-started,
        input_identity=inputs_identity, source_identity=sources_identity,
        selected_training_evaluation=selected_train, selected_validation_replay=selected_validation,
        protocol_sha256=sha256(OUT / 'protocol.json')))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--variant', choices=VARIANTS, required=True)
    p.add_argument('--seed', choices=['s1', 's2'], required=True); p.add_argument('--device', default='cuda:0')
    p.add_argument('--epochs', type=int)
    args = p.parse_args(); torch.set_num_threads(4)
    train(args.variant, args.seed, args.device, args.epochs)
