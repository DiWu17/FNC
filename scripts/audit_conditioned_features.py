"""Replay target-token alignment and validate frozen spatial feature caches."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from clickagent.action_benchmark import sha256,write_json
from clickagent.target_token import select_stages
from run_target_token_experiment import read_json,read_jsonl,key


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('runs/conditioned_v3'))
    p.add_argument('--manifest',type=Path,default=Path('runs/conditioned_v3/manifest.jsonl'))
    p.add_argument('--require-complete',choices=('trainval','test','all'))
    args=p.parse_args();out=args.output
    identity=read_json(out/'feature_identity.json')
    assert identity['manifest_sha256']==sha256(args.manifest)
    assert identity['max_pixels']==262144
    assert identity['qwen_weight_identity_sha256']==sha256(Path('data/conditioned_v3/qwen_weight_identity.json'))
    for name,digest in identity['source_sha256'].items():assert sha256(Path(name))==digest,name
    cases=read_jsonl(args.manifest);by_key={key(c['sample_id']):c for c in cases}
    counts=Counter();names=Counter();grids=Counter();seconds=[];checked=[]
    for path in sorted((out/'features').glob('*.npz')):
        case=by_key[path.stem]
        with np.load(path,allow_pickle=False) as z:
            meta=json.loads(str(z['metadata']));gh,gw=meta['grid_shape'];n=gh*gw
            assert meta['sample_id']==case['sample_id']
            assert meta['image_sha256']==case['image_sha256']==sha256(Path(case['image']))
            assert meta['pixel_sha256']==case['pixel_sha256']
            assert meta['image_size']==case['image_size']
            assert meta['all_backbone_parameters_frozen'] and meta['own_token_forward']
            assert meta['ground_truth_in_model_input'] is False
            assert meta['fine_grid_shape']==[2*gh,2*gw]
            assert meta['image_content_layer']==15
            positions=meta['image_content_positions']
            assert len(positions)==n and len(set(positions))==n
            assert positions==list(range(positions[0],positions[0]+n))
            assert 0<=positions[0]<=positions[-1]<meta['input_tokens']
            stages,parsed=select_stages(meta['generated_text'],meta['generated_prefixes'],
                ended_by_eos=meta['terminated_by_eos'],token_cap_reached=meta['token_cap_reached'])
            indices=stages.get('target_mean',[])
            assert stages==meta['stages'] and indices==meta['target_raw_indices']
            assert parsed['target_available']==meta['target_available']
            assert parsed['target_text']==meta['target_text']
            assert meta['target_query_absolute_positions']==[meta['input_tokens']+i-1 for i in indices]
            assert all(1<=i<=len(meta['generated_token_ids']) for i in indices)
            t=len(indices)
            shapes={'target_attention':(t,96,n),'target_hidden':(t,12288),
                'image_hidden':(n,4096),'vision_hidden':(4*n,1152),
                'prefill_attention':(96,n),'prefill_hidden':(12288,)}
            arrays={name:z[name] for name in shapes}
            for name,shape in shapes.items():
                assert arrays[name].shape==shape,(path,name,arrays[name].shape,shape)
                assert np.isfinite(arrays[name]).all(),(path,name)
            for name in ('target_attention','prefill_attention'):
                assert arrays[name].min(initial=0)>=0
                assert arrays[name].sum(-1).max(initial=0)<=1.002
            assert bool(t)==bool(meta['target_available'])
            assert len(meta['head_order'])==96
            assert meta['head_order']==[f'layer_{l}.head_{h}' for l in (15,28,36) for h in range(32)]
        counts[case['experiment_split']]+=1;names[case['experiment_split']]+=bool(t)
        grids[f'{gh}x{gw}']+=1;seconds.append(meta['elapsed_seconds']);checked.append(case['sample_id'])
    assert checked,'no feature caches to audit'
    if args.require_complete:
        required={'trainval':('train','val'),'test':('test',),'all':('train','val','test')}[args.require_complete]
        for split in required:assert counts[split]==sum(c['experiment_split']==split for c in cases)
    runtimes={}
    for path in out.glob('runtime_*.json'):
        runtime=read_json(path)
        assert runtime['trainable_parameter_count']==0 and runtime['all_parameters_frozen']
        assert not runtime['model_training']
        runtimes[path.stem]={'parameter_count':runtime['parameter_count'],'trainable_parameters':0}
    result={'complete':True,'audited_samples':len(checked),'counts':dict(counts),'named':dict(names),
        'grid_counts':dict(grids),'own_target_token_positions_replayed':True,'all_arrays_finite':True,
        'coarse_image_positions_aligned':True,'fine_vision_token_count_is_4P':True,'source_sha_matches':True,
        'median_collection_seconds':float(np.median(seconds)),'runtime':runtimes,
        'manifest_sha256':sha256(args.manifest),'feature_identity_sha256':sha256(out/'feature_identity.json')}
    write_json(out/'feature_audit.json',result)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
