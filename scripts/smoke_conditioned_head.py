"""Check real cached feature dimensions and CUDA training before a long run."""
from pathlib import Path
import json

import torch

from clickagent.action_benchmark import write_json
from clickagent.conditioned_grounder import ConditionedSpatialGrounder,SpatialConfig,make_batch,spatial_loss,predict_spatial
from run_target_token_experiment import read_jsonl
from train_conditioned_readouts import load_records


def main():
    torch.set_num_threads(4);torch.manual_seed(20261101)
    cases=[c for c in read_jsonl(Path('runs/conditioned_v3/manifest.jsonl')) if c['experiment_split']=='train'][:1]
    records,_=load_records(Path('runs/conditioned_v3_pilot'),cases)
    inputs,labels=make_batch(records,'cuda:0');results={}
    for variant in ('full','static_96'):
        model=ConditionedSpatialGrounder(SpatialConfig(variant=variant)).to('cuda:0')
        optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4);losses=[]
        for _ in range(2):
            optimizer.zero_grad();output=model(**inputs);loss,_=spatial_loss(output,**labels)
            assert torch.isfinite(loss);loss.backward()
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            optimizer.step();losses.append(float(loss.detach()))
        model.eval();proposal=predict_spatial(model,records[0],'cuda:0')
        assert proposal['point'] is not None
        results[variant]={'losses':losses,'point':proposal['point'],'parameters':sum(p.numel() for p in model.parameters()),
            'proposal_grid_shape':proposal['proposal_grid_shape']}
        del model,optimizer
    result={'complete':True,'real_training_sample_only':True,'smoke_weights_discarded':True,
        'input_shapes':{k:list(v.shape) for k,v in inputs.items() if hasattr(v,'shape')},'variants':results}
    write_json(Path('runs/conditioned_v3_pilot/head_cuda_smoke.json'),result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
