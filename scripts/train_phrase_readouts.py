"""Train the preregistered late-fusion ablations; freeze before new test."""
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
from clickagent.action_grounding import AttentionPointGrounder
from clickagent.conditioned_grounder import make_batch
from clickagent.phrase_grounder import (PhraseConfig, PhraseSpatialGrounder, decode_phrase,
                                       load_phrase, phrase_loss, save_phrase, split_children)
from clickagent.public_evaluation import point_in_geometry
from run_target_token_experiment import key, read_json, read_jsonl
from train_conditioned_readouts import batches, load_records


@torch.inference_mode()
def evaluate(model, records, device, batch_size):
    model.eval(); hits = 0; coarse_hits = 0; fine_hits = 0; loss_sum = 0.; first_largest = 0; first_weight = 0.
    for group in batches(records, batch_size):
        inputs, labels = make_batch(group, device); result = model(**inputs)
        loss, _ = phrase_loss(result, **labels); loss_sum += float(loss)*len(group)
        cpu = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in result.items()}
        for i, r in enumerate(group):
            c = r['case']; w, height = c['image_size']
            proposal = decode_phrase(cpu, i, c['image_size'], top_k=1)[0]
            point = proposal['point']; hits += point_in_geometry([point[0]/w, point[1]/height], c['target_geometry'])
            gh, gw = r['metadata']['grid_shape']; rr, cc = divmod(proposal['coarse_index'], gw)
            coarse_weights = r['label_weights'].reshape(gh, 2, gw, 2).sum((1, 3))
            coarse_hits += coarse_weights[rr, cc] > 0
            fr, fc = divmod(proposal['patch_index'], gw*2); fine_hits += r['label_weights'][fr, fc] > 0
            tw = cpu['token_weights'][i, :len(r['target_attention']), proposal['coarse_index']]
            first_largest += int(tw.argmax()) == 0; first_weight += float(tw[0])
    named = sum(r['metadata']['target_available'] for r in records)
    return dict(n=len(records), hits=int(hits), accuracy=float(hits/len(records)), named=int(named),
                coarse_overlap=int(coarse_hits), fine_overlap=int(fine_hits), loss=loss_sum/named,
                first_largest_at_selected_cell=int(first_largest), mean_first_weight_at_selected_cell=first_weight/named)


def train_one(out, name, variant, seed, records, protocol, initial, device):
    checkpoint = out/'models'/f'{name}.pt'; history_path = out/'models'/f'{name}_history.json'
    if history_path.exists():
        saved = read_json(history_path)
        assert saved['complete'] and saved['checkpoint_sha256'] == sha256(checkpoint)
        return saved
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    c = PhraseConfig(width=protocol['width'], variant=variant)
    model = PhraseSpatialGrounder(c, initial).to(device)
    train_all = [r for r in records if r['case']['experiment_split'] == 'train']
    train = [r for r in train_all if r['metadata']['target_available']]
    val = [r for r in records if r['case']['experiment_split'] == 'val']
    optimizer = torch.optim.AdamW(model.parameters(), lr=protocol['learning_rate'], weight_decay=protocol['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=protocol['max_epochs'], eta_min=protocol['learning_rate']*.1)
    rng = random.Random(seed); started = time.perf_counter(); history = []; best_rank = None; best_epoch = 0
    for epoch in range(protocol['max_epochs']+1):
        if epoch:
            model.train(); train_loss = []
            for group in batches(train, protocol['batch_size'], rng):
                inputs, labels = make_batch(group, device); optimizer.zero_grad(set_to_none=True)
                loss, _ = phrase_loss(model(**inputs), **labels)
                if not torch.isfinite(loss):
                    raise ValueError('nonfinite phrase loss')
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); optimizer.step()
                train_loss.extend([float(loss.detach())]*len(group))
            scheduler.step()
        metrics = evaluate(model, val, device, protocol['batch_size'])
        record = dict(epoch=epoch, val=metrics, elapsed_seconds=time.perf_counter()-started)
        if epoch:
            record['train_loss'] = float(np.mean(train_loss))
        history.append(record)
        rank = (metrics['hits'], -metrics['loss'])
        if best_rank is None or rank > best_rank:
            best_rank = rank; best_epoch = epoch
            save_phrase(checkpoint, model, dict(name=name, variant=variant, seed=seed, epoch=epoch, validation=metrics,
                                               train_named=len(train), backbone_trainable_parameters=0))
        write_json(out/'models'/f'{name}_progress.json', dict(history=history, best_epoch=best_epoch))
        print(name+' '+json.dumps(record), flush=True)
        if epoch >= protocol['min_epochs'] and epoch-best_epoch >= protocol['early_stop_patience']:
            break
    selected, metadata = load_phrase(checkpoint, device)
    training = evaluate(selected, train_all, device, protocol['batch_size'])
    result = dict(complete=True, name=name, variant=variant, seed=seed, best_epoch=best_epoch,
                  validation=history[best_epoch]['val'], train_metrics=training, history=history,
                  checkpoint_sha256=sha256(checkpoint), head_parameters=sum(p.numel() for p in model.parameters()))
    write_json(history_path, result)
    print('SELECTED '+name+' '+json.dumps({k: v for k, v in result.items() if k != 'history'}), flush=True)
    del selected, model, optimizer; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output', type=Path, default=Path('runs/phrase_v4'))
    p.add_argument('--device', default='cuda:0'); args = p.parse_args(); out = args.output
    if (out/'frozen_readouts.json').exists():
        raise ValueError('Already frozen: do not refit after collecting test')
    protocol = read_json(out/'protocol.json'); rows = read_jsonl(out/'manifest.jsonl')
    assert protocol['manifest_sha256'] == sha256(out/'manifest.jsonl')
    test = [c for c in rows if c['experiment_split'] == 'test']
    assert not any((out/'features'/f'{key(c["sample_id"])}.npz').exists() for c in test)
    sources = ['src/clickagent/phrase_grounder.py', 'src/clickagent/conditioned_grounder.py',
               'src/clickagent/spatial_geometry.py', 'src/clickagent/public_evaluation.py',
               'scripts/train_conditioned_readouts.py', str(Path(__file__)), str(out/'protocol.json')]
    identity = dict(manifest_sha256=sha256(out/'manifest.jsonl'), protocol_sha256=sha256(out/'protocol.json'),
                    source_sha256={s: sha256(Path(s)) for s in sources},
                    reused_feature_identity_sha256=sha256(Path('runs/conditioned_v3/feature_identity.json')))
    if (out/'training_identity.json').exists() and read_json(out/'training_identity.json') != identity:
        raise ValueError('Training inputs changed during resume')
    write_json(out/'training_identity.json', identity)
    (out/'models').mkdir(exist_ok=True); (out/'head_sources').mkdir(exist_ok=True)
    for source in sources:
        shutil.copyfile(source, out/'head_sources'/Path(source).name)
    torch.set_num_threads(4); torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    records, missing = load_records(Path(protocol['trainval_feature_root']),
                                    [c for c in rows if c['experiment_split'] in ('train', 'val')])
    old = AttentionPointGrounder.load('runs/action_v3/grounder/grounder.joblib').pipeline
    initial = (old.named_steps['classifier'].coef_[0]/old.named_steps['scale'].scale_*.25).astype(np.float32)
    jobs = [(v, v, protocol['primary_seed']) for v in protocol['variants']]
    jobs += [(v+'_replica', v, protocol['replication_seed']) for v in protocol['replication_variants']]
    results = {name: train_one(out, name, variant, seed, records, protocol, initial, args.device) for name, variant, seed in jobs}
    assert not any((out/'features'/f'{key(c["sample_id"])}.npz').exists() for c in test)
    baselines = {name: {'file': str(Path('runs/conditioned_v3/models')/f'{variant}.pt'),
                        'sha256': sha256(Path('runs/conditioned_v3/models')/f'{variant}.pt')}
                 for name, variant in [('previous_full', 'full'), ('previous_attention_only', 'attention_only'), ('previous_static_96', 'static_96')]}
    frozen = dict(primary_model='local', models={name: dict(file=name+'.pt', sha256=r['checkpoint_sha256'],
                  validation=r['validation'], best_epoch=r['best_epoch'], head_parameters=r['head_parameters']) for name, r in results.items()},
                  baselines=baselines, head_sources=identity['source_sha256'], test_features_at_freeze=0,
                  test_labels_used_for_fitting=0, test_n=len(test), missing_trainval_names=missing,
                  selection='All variants predefined; epochs selected solely on the unchanged validation set. Local is fixed primary.')
    write_json(out/'frozen_readouts.json', frozen)
    print('FROZEN '+json.dumps(frozen), flush=True)


if __name__ == '__main__':
    main()
