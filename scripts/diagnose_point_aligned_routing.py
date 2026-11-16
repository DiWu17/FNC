"""Post-hoc validation-only checks of the frozen spatial adapter and router."""
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from clickagent.action_benchmark import sha256,write_json
from clickagent.attention_field_ranker import make_field_batch
from clickagent.point_aligned_readout import load_aligned,all_candidate_points
from evaluate_conditioned_readouts import score_point
from diagnose_semantic_head_routing import select_validation,routing_observations,zero_new_query_condition,describe
from point_aligned_data import OUT,BASE,read


@contextmanager
def adapter_off(model):
    handle=model.spatial_fine.register_forward_hook(lambda module,args,out:torch.zeros_like(out))
    try:yield
    finally:handle.remove()


@torch.inference_mode()
def main():
    torch.set_num_threads(4);frozen=read(OUT/'frozen_readouts.json');records,skipped=select_validation(BASE,64)
    result={}
    for suffix in ('s1','s2'):
        for family in ('joint_detached','joint_connected'):
            name=family+'_'+suffix;entry=frozen['models'][name];path=OUT/'models'/entry['file']
            assert sha256(path)==entry['sha256'];model,_=load_aligned(path,'cpu');stats=defaultdict(list);outcomes=defaultdict(int)
            cases=[]
            for r in records:
                inputs,_=make_field_batch([r],'cpu',labels=False);out=model(**inputs,return_routing=True)
                ci=int(out['scores'].argmax());points=all_candidate_points(out,[r])[0];point=points[ci].tolist()
                observations,_=routing_observations(out['head_routing_weights'].numpy(),out['group_token_weights'].numpy(),
                    len(r['parent_indices']),len(r['query_hidden']))
                for k,v in observations.items():stats[k].append(float(np.mean(v)))
                variants={}
                for intervention,context in [('adapter_off',adapter_off(model)),('condition_off',zero_new_query_condition(model))]:
                    with context:changed=model(**inputs)
                    other_ci=int(changed['scores'].argmax());other_points=all_candidate_points(changed,[r])[0];other=other_points[other_ci].tolist()
                    variants[intervention]=dict(point=other,hit=bool(score_point(other,r['case'])),selected_candidate=other_ci)
                    outcomes[intervention+'_selected_parent_changed']+=other_ci!=ci
                    outcomes[intervention+'_selected_point_changed']+=other!=point
                    outcomes[intervention+'_hits']+=variants[intervention]['hit']
                    stats[intervention+'_all_candidate_point_shift_original_px'].append(float(np.linalg.norm(other_points-points,axis=-1).mean()))
                hit=bool(score_point(point,r['case']));outcomes['hits']+=hit
                cases.append(dict(sample_id=r['case']['sample_id'],point=point,hit=hit,variants=variants))
            result[name]=dict(n=64,checkpoint_sha256=entry['sha256'],selected_epoch=entry['selected']['epoch'],
                outcomes=dict(outcomes),case_equal_statistics={k:describe(v) for k,v in stats.items()},cases=cases)
            print(name,{k:v for k,v in outcomes.items()},flush=True)
    write_json(OUT/'evaluation'/'routing_diagnostic.json',dict(complete=True,split='original_validation_only',
        cases=64,skipped=skipped,models=result,source_sha256=sha256(Path(__file__)),
        warning='Post-hoc mechanism diagnosis, no fresh-test features or checkpoint/policy selection. Adapter-off changes only the new fine residual at fixed final weights; it is not a matched retraining result.'))


if __name__=='__main__':main()
