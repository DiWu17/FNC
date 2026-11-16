"""Predict candidate click points from frozen Qwen target-name states."""
import argparse
import json
from pathlib import Path
import time

from PIL import Image

from clickagent.conditioned_features import ConditionedFeatureRuntime
from clickagent.conditioned_grounder import load_spatial_checkpoint,predict_spatial
from clickagent.model_features import ExtractionConfig,_sync_cuda
from run_target_token_experiment import read_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image',type=Path,required=True);p.add_argument('--instruction',required=True)
    p.add_argument('--checkpoint',type=Path,default=Path('runs/conditioned_v3/models/full.pt'))
    p.add_argument('--device',default='cuda:0');p.add_argument('--output',type=Path)
    args=p.parse_args()
    cfg=read_json(args.checkpoint.parent.parent/'feature_identity.json')['config']
    cfg.update(max_new_tokens=64,seed=20261101,conditions=['prefill'],offload_folder='runs/conditioned_predict_offload')
    cfg.update(layers=tuple(cfg['layers']),conditions=tuple(cfg['conditions']),
        max_memory={int(k) if str(k).isdigit() else k:v for k,v in cfg['max_memory'].items()})
    runtime=ConditionedFeatureRuntime(ExtractionConfig(**cfg))
    head,checkpoint_metadata=load_spatial_checkpoint(args.checkpoint,args.device)
    with Image.open(args.image) as opened:image=opened.convert('RGB')
    arrays,meta=runtime.infer_spatial_features(image,args.instruction)
    _sync_cuda();started=time.perf_counter()
    result=predict_spatial(head,dict(arrays,metadata=meta),args.device)
    _sync_cuda();head_seconds=time.perf_counter()-started
    result.update(target_name=meta['target_text'],generated_text=meta['generated_text'],
        generated_action=meta['action'],feature_collection_seconds=meta['elapsed_seconds'],
        action_by_generated_word='CLICK' if meta['action']=='CLICK' and result['point'] is not None else 'WAIT',
        spatial_readout_seconds=head_seconds,checkpoint_metadata=checkpoint_metadata,
        os_action_executed=False,ground_truth_in_model_input=False)
    # This is a point proposal interface; WAIT and negative click safety require
    # their own independently validated action policy before connecting a mouse.
    text=json.dumps(result,ensure_ascii=False,indent=2)
    if args.output:args.output.write_text(text+'\n',encoding='utf-8')
    print(text,flush=True)


if __name__=='__main__':main()
