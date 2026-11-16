"""CPU-only infrastructure check in a new isolated folder, not an accuracy run."""
from datetime import datetime, timezone
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'src')]
import torch
import run_full_scaling_v18 as full


def main():
    spec = full.verify_protocol()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    folder = ROOT / '.tmp' / f'training_resume_v18_{stamp}'
    assert not folder.exists()
    folder.mkdir(parents=True)
    prior = full.prior
    torch.set_num_threads(2)
    prior.OUT = folder
    prior.status = lambda phase, **fields: print(json.dumps(dict(phase=phase, **fields)), flush=True)
    entries = full.read(full.OUT / 'data/old_train_index.json')[:3]
    records = prior.base.load_records(entries)
    assert all(r['n'] > 0 for r in records)
    full.write(folder / 'protocol.json', dict(purpose='CPU checkpoint/resume infrastructure check only; not scientific training or evaluation',
                                             source_full_protocol_sha256=full.sha(full.OUT / 'protocol.json'),
                                             steps=2, queries=len(records), device='cpu'))
    seed = spec['seeds'][0]
    prior.train_one('direct', records, seed, 2, 'cpu')
    prior.train_one('resumed', records, seed, 1, 'cpu')
    # Preserve the first completed outputs so the genuine resume checkpoint is exercised.
    for suffix in ['json', 'pt']:
        p = folder / f'models/resumed_s1.{suffix}'
        p.rename(folder / f'models/resumed_s1.first.{suffix}')
    prior.train_one('resumed', records, seed, 2, 'cpu')
    direct = torch.load(folder / 'models/direct_s1.pt', map_location='cpu', weights_only=True)
    resumed = torch.load(folder / 'models/resumed_s1.pt', map_location='cpu', weights_only=True)
    assert direct.keys() == resumed.keys()
    for name in direct['state_dict']:
        assert torch.equal(direct['state_dict'][name], resumed['state_dict'][name]), name
    for name in ['mean', 'std']:
        assert torch.equal(direct[name], resumed[name]), name
    # Re-running an already completed fit must leave its checkpoint bytes unchanged.
    checkpoint = folder / 'models/resumed_s1.pt'
    before = full.sha(checkpoint)
    prior.train_one('resumed', records, seed, 2, 'cpu')
    assert full.sha(checkpoint) == before
    for tag in ['direct_s1', 'resumed_s1']:
        receipt = full.read(folder / f'models/{tag}.json')
        assert receipt['complete'] and receipt['steps'] == 2 and receipt['parameter_count'] == 22050
        assert full.sha(receipt['checkpoint_path']) == receipt['checkpoint_sha256']
    full.verify_protocol()
    report = dict(complete=True, checked_at_utc=datetime.now(timezone.utc).isoformat(),
                  exact_cpu_two_step_resume_parity=True, completed_fit_idempotent=True,
                  source_protocol_and_adopted_checkpoints_unchanged=True,
                  scope='Infrastructure smoke test on3 archived training rows; no heldout/development inference; not accuracy evidence',
                  isolated_folder=str(folder), check_source_sha256=full.sha(__file__))
    for model in full.read(full.MAIN / 'models/frozen.json')['models'].values():
        assert full.sha(model['path']) == model['sha256']
    full.write(full.OUT / 'training_infrastructure_check.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
