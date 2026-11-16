"""Replay frozen-policy scoring and verify timing, source and image separation."""
import argparse
import json
from pathlib import Path

import numpy as np

from clickagent.action_benchmark import sha256,write_json
from clickagent.action_grounding import AttentionPointGrounder
from clickagent.semantic_target import PROMPTS,POLICIES,temporal_map
from run_semantic_target import load_feature,score_map
from run_target_token_experiment import read_json,read_jsonl,key


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('runs/semantic_target_v2'))
    args=p.parse_args();output=args.output
    identity=read_json(output/'identity.json');frozen=read_json(output/'frozen_selection.json')
    cases=read_jsonl(output/'manifest.jsonl')
    assert len({c['image_sha256'] for c in cases})==len(cases)==192
    old_paths={}
    for prior in read_json(output/'selection.json')['prior_manifests']:
        for c in read_jsonl(prior):
            old_paths[c['image']]=c.get('image_sha256') or old_paths.get(c['image'])
    old_hashes={digest or sha256(Path(image)) for image,digest in old_paths.items()}
    assert not ({c['image_sha256'] for c in cases}&old_hashes)
    assert sha256(output/'manifest.jsonl')==identity['manifest_sha256']
    assert sha256(output/'identity.json')==frozen['identity_sha256']
    assert frozen['test_features_at_freeze']==0
    assert sha256(Path('scripts/run_semantic_target.py'))==identity['driver_sha256']
    for name,digest in identity['source_sha256'].items():assert sha256(Path('src/clickagent')/name)==digest,name
    assert sha256(Path('runs/action_v3/grounder/grounder.joblib'))==identity['grounder_sha256']
    grounder=AttentionPointGrounder.load('runs/action_v3/grounder/grounder.joblib')
    timestamps_ok=True;cache_count=0
    for c in cases:
        assert sha256(Path(c['image']))==c['image_sha256']
        for variant in PROMPTS:
            path=output/'features'/variant/f"{key(c['sample_id'])}.npz"
            raw,meta=load_feature(output,c,variant)
            assert meta['system_prompt']==PROMPTS[variant]
            assert meta['ground_truth_in_model_input'] is False and meta['own_token_forward'] is True
            assert raw.shape==(len(meta['generated_token_ids'])+1,96,int(np.prod(meta['grid_shape'])))
            assert np.isfinite(raw).all() and (raw>=0).all() and (raw.sum(-1)<1.001).all()
            assert meta['query_token_indices']==list(range(meta['input_tokens']-1,meta['input_tokens']+len(meta['generated_token_ids'])))
            if c['experiment_split']=='test':timestamps_ok &= path.stat().st_mtime_ns>= (output/'frozen_selection.json').stat().st_mtime_ns
            cache_count+=1
    assert timestamps_ok
    options=[]
    for variant in PROMPTS:
        for policy in POLICIES:
            hits=0
            for c in cases:
                if c['experiment_split']!='dev':continue
                raw,meta=load_feature(output,c,variant)
                hits+=score_map(temporal_map(raw,meta,policy),meta,c,grounder)['hit']
            options.append(dict(variant=variant,policy=policy,dev_hits=int(hits),n_dev=64))
    assert options==frozen['all_development_options']
    assert max(options,key=lambda r:r['dev_hits'])==frozen['selected']
    lookup={c['sample_id']:c for c in cases}
    predictions=read_jsonl(output/'predictions.jsonl')
    optimized=[r for r in predictions if r['method']=='optimized']
    for row in optimized:
        c=lookup[row['sample_id']];choice=frozen['selected']
        raw,meta=load_feature(output,c,choice['variant'])
        replay=score_map(temporal_map(raw,meta,choice['policy']),meta,c,grounder)
        assert all(row[k]==replay[k] for k in ('point','hit','cell_overlap'))
    result={'complete':True,'source_hashes_match':True,'image_content_unique':True,
        'prior_experiment_image_overlap':0,
        'all_test_features_after_policy_freeze':bool(timestamps_ok),'raw_caches_verified':cache_count,
        'development_choice_replayed':True,'optimized_test_predictions_replayed':len(optimized),
        'backbone_trainable_parameters':read_json(output/'runtime.json')['trainable_parameter_count']}
    assert result['backbone_trainable_parameters']==0
    adapted=output/'adapted_readouts'
    if (adapted/'predictions.jsonl').exists():
        weights=read_json(adapted/'frozen_weights.json')
        assert weights['test_features_loaded']==0 and not weights['test_predictions_exist_at_freeze']
        assert sha256(Path('scripts/fit_target_attention_readouts.py'))==weights['fit_source_sha256']
        assert (adapted/'frozen_weights.json').stat().st_mtime_ns < (output/'predictions.jsonl').stat().st_mtime_ns
        assert set(weights['fit_sample_ids'])=={c['sample_id'] for c in cases if c['experiment_split']=='dev'}
        adapted_rows=read_jsonl(adapted/'predictions.jsonl')
        assert len(adapted_rows)==256
        for stage in ('prefill','target'):
            assert sha256(adapted/f'{stage}.joblib')==weights['heads'][stage]['sha256']
            head=AttentionPointGrounder.load(adapted/f'{stage}.joblib')
            assert head.num_heads==96 and weights['heads'][stage]['coefficient_count']==97
            for row in adapted_rows:
                if row['method']!='adapted_'+stage:continue
                c=lookup[row['sample_id']]
                assert c['experiment_split']=='test'
                choice=weights['temporal_selection']
                raw,meta=load_feature(output,c,choice['variant'])
                attention=raw[0] if stage=='prefill' else temporal_map(raw,meta,choice['policy'])
                replay=score_map(attention,meta,c,head)
                assert all(row[k]==replay[k] for k in ('point','hit','cell_overlap'))
        result['adapted_readout_predictions_replayed']=len(adapted_rows)
        result['adapted_weights_frozen_before_any_test_scores']=True
        result['adapted_readout_test_features_used_for_fit']=0
    write_json(output/'final_audit.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
