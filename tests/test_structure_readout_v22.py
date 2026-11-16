import numpy as np
import torch
from clickagent.structure_readout_v22 import StructureReadout,spatial_sequence,pooled_inputs,projection
from clickagent.bottleneck_v19 import BottleneckModel,probe_loss

torch.set_num_threads(2)

def args():
    return dict(base=torch.randn(5,32),state=torch.randn(5,4,3,294),mask=torch.ones(5,4,3,dtype=torch.bool),
        text=torch.randn(2,7,64),text_mask=torch.tensor([[1]*7,[1]*3+[0]*4],dtype=torch.bool),
        spatial=torch.randn(5,4,10,64),spatial_mask=torch.ones(5,4,10,dtype=torch.bool),candidate_case=torch.tensor([0,0,1,1,1]))

def test_zero_residual_replays_width_matched_original_and_exact_pair_init():
    for width in [32,64]:
        torch.manual_seed(71);ref=BottleneckModel(width)
        torch.manual_seed(71);a=StructureReadout(width)
        torch.manual_seed(71);b=StructureReadout(width)
        assert all(torch.equal(x,b.state_dict()[k]) for k,x in a.state_dict().items())
        inp=args();expected=ref(**{k:inp[k] for k in ['base','state','mask']})
        for condition in ['ordered','pooled']:torch.testing.assert_close(a(**inp,condition=condition),expected,rtol=0,atol=0)

def test_pooling_preserves_masks_coarse_token_and_valid_means():
    inp=args();inp['spatial_mask'][0,1,:3]=False
    t,v=pooled_inputs(inp['text'],inp['text_mask'],inp['spatial'],inp['spatial_mask'],True,True)
    torch.testing.assert_close(v[:,:,9],inp['spatial'][:,:,9])
    for i in range(2):
        valid=inp['text_mask'][i];torch.testing.assert_close(t[i,valid],inp['text'][i,valid].mean(0).expand(valid.sum(),-1))
    assert torch.count_nonzero(t[~inp['text_mask']])==0

def test_spatial_cells_keep_order_and_transform_without_target_geometry():
    rng=np.random.default_rng(8)
    arrays=dict(vision_hidden=rng.normal(size=(64,1152)).astype(np.float16),image_hidden=rng.normal(size=(16,4096)).astype(np.float16))
    meta=dict(grid_shape=[4,4],fine_grid_shape=[8,8])
    a,m=spatial_sequence(arrays,meta,[[35,35],[0,0],[90,2]],[80,80])
    assert a.shape==(3,10,64) and m[0].all() and m[1].sum()==5 and not m[2].any()
    assert not np.array_equal(a[0,0],a[0,1])
    b,n=spatial_sequence(arrays,meta,[[45,55]],[100,120],dict(box=[10,20,90,100]))
    np.testing.assert_array_equal(a[0],b[0]);np.testing.assert_array_equal(m[0],n[0])

def test_masked_padding_and_absent_views_are_finite_and_irrelevant():
    torch.manual_seed(3);model=StructureReadout();inp=args()
    inp['mask'][0]=False;inp['spatial_mask'][0]=False
    torch.nn.init.normal_(model.readout[-1].weight)
    a=model(**inp);copy={**inp,'text':inp['text'].clone(),'spatial':inp['spatial'].clone()}
    copy['text'][~inp['text_mask']]=999;copy['spatial'][~inp['spatial_mask']]=-999
    b=model(**copy);torch.testing.assert_close(a,b,rtol=0,atol=0)
    loss=probe_loss(a,torch.tensor([0.,1.,0.,1.,0.]),model.null);loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())

def test_joint_pooling_is_invariant_to_content_permutations_and_ordered_can_use_order():
    torch.manual_seed(5);m=StructureReadout();torch.nn.init.normal_(m.readout[-1].weight);inp=args()
    swapped={**inp,'text':inp['text'].clone(),'spatial':inp['spatial'].clone()}
    swapped['text'][:,[0,1]]=swapped['text'][:,[1,0]];swapped['spatial'][:,:,[0,1]]=swapped['spatial'][:,:,[1,0]]
    torch.testing.assert_close(m(**inp,condition='pooled'),m(**swapped,condition='pooled'),rtol=1e-5,atol=1e-6)
    assert not torch.allclose(m(**inp),m(**swapped),rtol=1e-5,atol=1e-6)

def test_optimizer_resume_matches_uninterrupted_training(tmp_path):
    torch.manual_seed(12);inp=args();labels=torch.tensor([0.,1.,0.,1.,0.])
    def create():
        torch.manual_seed(91);m=StructureReadout();return m,torch.optim.AdamW(m.parameters(),lr=2e-4,weight_decay=.02)
    def step(m,o):
        o.zero_grad(set_to_none=True);loss=probe_loss(m(**inp),labels,m.null);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);o.step()
    a,oa=create();step(a,oa);step(a,oa)
    b,ob=create();step(b,ob);path=tmp_path/'checkpoint.pt';torch.save(dict(state=b.state_dict(),opt=ob.state_dict()),path)
    c,oc=create();v=torch.load(path,weights_only=True);c.load_state_dict(v['state']);oc.load_state_dict(v['opt']);step(c,oc)
    for k,v in a.state_dict().items():torch.testing.assert_close(v,c.state_dict()[k],rtol=0,atol=0)

def test_empty_query_cannot_poison_shared_instruction_cache(tmp_path,monkeypatch):
    import sys,hashlib
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    import run_structure_readout_v22 as run
    instruction='Click save';h=hashlib.sha256(instruction.encode()).hexdigest();records=[];spatial=[]
    for n in [0,2]:
        sid=f'case{n}';records.append(dict(sample_id=sid,n=n,base=np.zeros((n,32),np.float32),state=np.zeros((n,4,1,294),np.float16),mask=np.ones((n,4,1),bool),labels=np.zeros(n,np.float32),sha256=sid))
        p=tmp_path/f'{sid}.npy';q=tmp_path/f'{sid}.mask.npy';np.save(p,np.zeros((n,4,10,64),np.float16));np.save(q,np.ones((n,4,10),bool))
        spatial.append(dict(sample_id=sid,base_sha256=sid,features_path=str(p),features_sha256=run.sha(p),mask_path=str(q),mask_sha256=run.sha(q)))
    t=tmp_path/'text.npy';np.save(t,np.ones((3,64),np.float16));text={h:dict(path=str(t),sha256=run.sha(t))}
    def reader(p):
        p=str(p).replace('\\','/')
        if p.endswith('data/indices.json'):return {'train':[]}
        if p.endswith('spatial/index.json'):return dict(cases=spatial)
        if p.endswith('text/index.json'):return dict(records=text)
        if p.endswith('data/public.json'):return {r['sample_id']:dict(instruction=instruction) for r in records}
        raise AssertionError(p)
    monkeypatch.setattr(run,'read',reader);monkeypatch.setattr(run.prior.legacy,'load_records',lambda _:records)
    monkeypatch.setattr(torch,'load',lambda *args,**kwargs:dict(mean=torch.zeros(32),std=torch.ones(32)))
    cache=run.Cache('train','cpu')
    assert cache.data[1]['text'].shape==(3,64) and bool((cache.data[1]['text']==1).all())
