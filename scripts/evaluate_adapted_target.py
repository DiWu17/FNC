"""Evaluate already-frozen dev-only readouts on the common test screenshots."""
import argparse
import json
from pathlib import Path

from clickagent.action_benchmark import sha256,write_json
from clickagent.action_grounding import AttentionPointGrounder
from clickagent.semantic_target import temporal_map
from run_semantic_target import load_feature,score_map
from run_target_token_experiment import read_json,read_jsonl


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('runs/semantic_target_v2'))
    args=p.parse_args();output=args.output;dest=output/'adapted_readouts'
    freeze=read_json(dest/'frozen_weights.json');choice=freeze['temporal_selection']
    assert freeze['test_features_loaded']==0 and not freeze['test_predictions_exist_at_freeze']
    assert sha256(Path('scripts/fit_target_attention_readouts.py'))==freeze['fit_source_sha256']
    assert (dest/'frozen_weights.json').stat().st_mtime_ns < (output/'predictions.jsonl').stat().st_mtime_ns
    cases=[c for c in read_jsonl(output/'manifest.jsonl') if c['experiment_split']=='test']
    assert not set(freeze['fit_sample_ids'])&{c['sample_id'] for c in cases}
    rows=[]
    for stage in ('prefill','target'):
        assert sha256(dest/f'{stage}.joblib')==freeze['heads'][stage]['sha256']
        model=AttentionPointGrounder.load(dest/f'{stage}.joblib')
        for c in cases:
            raw,meta=load_feature(output,c,choice['variant'])
            attention=raw[0] if stage=='prefill' else temporal_map(raw,meta,choice['policy'])
            rows.append(dict(sample_id=c['sample_id'],dataset=c['dataset'],method='adapted_'+stage,
                **score_map(attention,meta,c,model)))
    (dest/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
    summary={stage:{'n':len(cases),'hits':sum(r['hit'] for r in rows if r['method']=='adapted_'+stage),
        'datasets':{ds:{'n':sum(c['dataset']==ds for c in cases),'hits':sum(r['hit'] for r in rows if r['method']=='adapted_'+stage and r['dataset']==ds)} for ds in sorted({c['dataset'] for c in cases})}} for stage in ('prefill','target')}
    write_json(dest/'summary.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
