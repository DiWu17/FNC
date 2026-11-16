"""Read-only reproduction of the audit entry-point thread configuration bug."""
import json
import torch
from clickagent.crossfit_proposer import load_crossfit_proposer,predict_raw
from cache_candidate_scorer import raw_arrays
from crossfit_protocol import OUT,cases,key,read
from clickagent.action_benchmark import write_json

torch_defaults=torch.get_num_threads()
torch.set_num_threads(4)
sid='salesforce:183c98dc-0d79-4c25-9b69-0e7d19e2ee3d'
p=read(OUT/'trainval/in_sample/pipeline'/(key(sid)+'.json'))
c=next(c for c in cases('train') if c['sample_id']==sid)
expected=p['global_result']
model,_=load_crossfit_proposer(OUT/'proposers'/(expected['proposer_role']+'.pt'),'cpu')
arrays,meta=raw_arrays(p['global_feature_path'])
actual=predict_raw(model,arrays,meta,c['image_size'],topk=64)
fields=['point','all_points','parent_indices','candidates','target_name','grid_shape','decoder']
matched={f:actual[f]==expected[f] for f in fields}
assert all(matched.values()),matched
result=dict(sample_id=sid,original_default_threads=torch_defaults,canonical_threads=4,
            canonical_all_fields_bitwise_equal=matched)
if torch_defaults!=4:
    torch.set_num_threads(torch_defaults)
    other=predict_raw(model,arrays,meta,c['image_size'],topk=64)
    result['default_thread_fields_equal']={f:other[f]==expected[f] for f in fields}
    result['default_candidate_numeric_difference_example']=[
        dict(index=i,field=f,default=value,canonical=actual['candidates'][i][f])
        for i,row in enumerate(other['candidates']) for f,value in row.items()
        if isinstance(value,(int,float)) and value!=actual['candidates'][i].get(f)][:5]
write_json(OUT/'checks/audit_thread_replay_diagnostic.json',result)
print(json.dumps(result,ensure_ascii=False))
