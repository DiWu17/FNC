"""Compact read-only progress snapshot; no model imports or evaluation."""
import json
from pathlib import Path
import time

root = Path('runs/crossfit_candidates_v12')
proposers = []
for p in sorted((root/'proposers').glob('*.json')):
    d = json.loads(p.read_text(encoding='utf-8'))
    proposers.append(dict(role=p.stem, epoch=d['history'][-1]['epoch'], complete=d['complete']))
progress = sorted(list((root/'trainval').glob('*/progress_*.json')) + list((root/'test').glob('*/progress_*.json')),
                  key=lambda p: p.stat().st_mtime, reverse=True)
models = []
for p in sorted((root/'models').glob('*_progress.json')):
    d = json.loads(p.read_text(encoding='utf-8'))
    models.append(dict(name=d['name'], epoch=d['latest']['epoch'], val_hits=d['latest']['validation']['hits']))
logs = sorted((root/'logs').glob('*.log'), key=lambda p: p.stat().st_mtime, reverse=True)
latest = []
for p in logs[:2]:
    lines = p.read_text(encoding='utf-8', errors='replace').splitlines()
    latest.append(dict(log=p.name, last=lines[-1][-600:] if lines else '', idle_seconds=round(time.time()-p.stat().st_mtime)))
print(json.dumps(dict(proposers_complete=sum(d['complete'] for d in proposers),
    proposers_running=[d for d in proposers if not d['complete']],
    progress=[json.loads(p.read_text(encoding='utf-8')) for p in progress[:2]],
    scorer_models=models, latest_logs=latest,
    frozen=(root/'frozen.json').exists(), final_audit=(root/'final_audit.json').exists(),
    finish_complete=(root/'finish_complete.json').exists()), ensure_ascii=False))
