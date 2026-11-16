"""Collect frozen name-token/query/image features under a pinned manifest."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from clickagent.action_benchmark import sha256,write_json
from run_target_token_experiment import read_json,read_jsonl,key


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('runs/conditioned_v3'))
    p.add_argument('--manifest',type=Path,default=Path('runs/conditioned_v3/manifest.jsonl'))
    p.add_argument('--split',choices=('trainval','test'),default='trainval')
    p.add_argument('--limit',type=int);args=p.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    rows=read_jsonl(args.manifest)
    selected=[r for r in rows if r['experiment_split'] in (('train','val') if args.split=='trainval' else ('test',))]
    if args.split=='test':
        freeze=read_json(out/'frozen_readouts.json')
        assert freeze['test_features_at_freeze']==0
        for name,r in freeze['models'].items():assert sha256(out/'models'/r['file'])==r['sha256']
        for name,digest in freeze['head_sources'].items():assert sha256(Path(name))==digest
    cfg=read_json(Path('runs/candidate_v2/identity.json'))['config']
    cfg.update(max_new_tokens=64,seed=20261101,conditions=['prefill'],offload_folder=str((out/'offload').resolve()))
    sources=['src/clickagent/'+name for name in ('conditioned_features.py','target_token.py','action_model.py','model_features.py','public_model.py','public_diagnostics.py')]
    sources.append(str(Path(__file__)))
    identity={'manifest_sha256':sha256(args.manifest),'config':cfg,'source_sha256':{name:sha256(Path(name)) for name in sources},
        'qwen_weight_identity_sha256':sha256(Path('data/conditioned_v3/qwen_weight_identity.json')),
        'image_content_layer':15,'model_inputs':'screenshot and instruction only','max_pixels':262144}
    if (out/'feature_identity.json').exists() and read_json(out/'feature_identity.json')!=identity:
        raise ValueError('Feature collector/config changed; use a new output')
    write_json(out/'feature_identity.json',identity)
    frozen=out/'feature_sources';frozen.mkdir(exist_ok=True)
    for name in sources:
        dest=frozen/Path(name).name
        if not dest.exists():shutil.copyfile(name,dest)
    folder=out/'features';folder.mkdir(exist_ok=True);native=out/'native';native.mkdir(exist_ok=True)
    pending=[c for c in selected if not (folder/f"{key(c['sample_id'])}.npz").exists() or
        (args.split=='test' and not (native/f"{key(c['sample_id'])}.json").exists())]
    if args.limit:pending=pending[:args.limit]
    if pending:
        from PIL import Image
        from clickagent.model_features import ExtractionConfig
        from clickagent.conditioned_features import ConditionedFeatureRuntime
        from clickagent.public_diagnostics import parse_offline_lax_click
        cfg.update(layers=tuple(cfg['layers']),conditions=tuple(cfg['conditions']),
            max_memory={int(k) if str(k).isdigit() else k:v for k,v in cfg['max_memory'].items()})
        runtime=ConditionedFeatureRuntime(ExtractionConfig(**cfg))
        write_json(out/f'runtime_{args.split}.json',runtime.runtime_metadata)
        for i,c in enumerate(pending,1):
            if sha256(Path(c['image']))!=c['image_sha256']:raise ValueError('Screenshot changed')
            with Image.open(c['image']) as opened:image=opened.convert('RGB')
            path=folder/f"{key(c['sample_id'])}.npz"
            if not path.exists():
                arrays,meta=runtime.infer_spatial_features(image,c['instruction'])
                meta.update(sample_id=c['sample_id'],experiment_split=c['experiment_split'],image_sha256=c['image_sha256'],pixel_sha256=c['pixel_sha256'])
                arrays['metadata']=np.asarray(json.dumps(meta,ensure_ascii=False))
                temp=path.with_suffix('.tmp')
                with temp.open('wb') as f:np.savez_compressed(f,**arrays)
                temp.replace(path)
            else:
                with np.load(path,allow_pickle=False) as z:meta=json.loads(str(z['metadata']))
            native_path=native/f"{key(c['sample_id'])}.json"
            if args.split=='test' and not native_path.exists():
                result=runtime.infer_text_action_public(image,c['instruction'],max_new_tokens=32)
                complete=result['terminated_by_eos'] or result['generated_text'].endswith(('\n','\r'))
                point=parse_offline_lax_click(result['generated_text']) if complete else None
                result['completed_lax_point']=[round(v*(s-1)/1000) for v,s in zip(point,image.size)] if point else None
                result.update(sample_id=c['sample_id'],ground_truth_in_model_input=False)
                write_json(native_path,result)
            print(f"{args.split} [{i}/{len(pending)}] {c['dataset']} name={meta['target_text']!r} grid={meta['grid_shape']}",flush=True)
    done=all((folder/f"{key(c['sample_id'])}.npz").exists() for c in selected)
    if args.split=='test':done &= all((native/f"{key(c['sample_id'])}.json").exists() for c in selected)
    write_json(out/f'features_{args.split}_completion.json',{'complete':done,'n':len(selected),
        'manifest_sha256':sha256(args.manifest)})


if __name__=='__main__':main()
