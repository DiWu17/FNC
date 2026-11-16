import sys
from pathlib import Path
import torch
import pytest
from clickagent.transfer_alignment_v20 import TransferMatcher, admissible_pairs, pair_loss, shared_channel_dropout

torch.set_num_threads(2)


def fixture_args():
    return dict(roi=torch.randn(7,4,128),roi_mask=torch.ones(7,4,dtype=torch.bool),
                query_ids=torch.tensor([0,0,0,1,1,1,1]),queries=2)


def test_zero_initial_residual_and_frozen_logits_preserved():
    torch.manual_seed(7);model=TransferMatcher();args=fixture_args();baseline=torch.randn(7)
    residual,_,_=model(**args)
    assert sum(p.numel() for p in model.parameters())==4097
    torch.testing.assert_close(baseline+residual,baseline,rtol=0,atol=0)
    assert model.anchor_penalty().item()==0
    assert not model.region_anchor.requires_grad and not model.query_anchor.requires_grad


def test_permutation_equivariance_invalid_and_empty_are_finite():
    torch.manual_seed(11);model=TransferMatcher().eval();model.alpha.data.fill_(.3);args=fixture_args()
    args['roi_mask'][0]=False;z=model(**args)[0];assert z[0]==0
    ix=torch.tensor([3,1,6,0,5,2,4]);changed={k:v[ix] if torch.is_tensor(v) and v.ndim else v for k,v in args.items()}
    torch.testing.assert_close(model(**changed)[0],z[ix],rtol=0,atol=0)
    assert model(args['roi'][:0],args['roi_mask'][:0])[0].numel()==0
    z.sum().backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_dropout_is_shared_per_query_not_per_candidate():
    x=torch.ones(7,4,128);ids=torch.tensor([0,0,0,1,1,1,1])
    torch.manual_seed(13);d=shared_channel_dropout(x,ids,2,.2,True)
    torch.testing.assert_close(d[0],d[1],rtol=0,atol=0)
    torch.testing.assert_close(d[3],d[6],rtol=0,atol=0)
    assert not torch.equal(d[0],d[3])
    torch.testing.assert_close(shared_channel_dropout(x,ids,2,.2,False),x,rtol=0,atol=0)


def test_pairing_exclusions_and_matching_supervision():
    allowed=admissible_pairs(['a','a','b'],['save','open','save'],'cpu')
    assert allowed.tolist()==[[True,False,False],[False,True,True],[False,True,True]]
    # Three independent mutually orthogonal matched query/region pairs.
    r=torch.eye(3).view(3,1,3).expand(3,4,3).clone().requires_grad_();q=r.detach().clone().requires_grad_()
    labels=[torch.ones(1,dtype=torch.bool)]*3;groups=['a','b','c'];instructions=['a','b','c']
    good,n=pair_loss(r,q,[1,1,1],labels,groups,instructions)
    bad,_=pair_loss(r,q.roll(1,0),[1,1,1],labels,groups,instructions)
    assert n==3 and good<.001 and bad>1
    good.backward();assert torch.isfinite(r.grad).all() and torch.isfinite(q.grad).all()
    zero,n=pair_loss(r,q,[1,1,1],labels,['same']*3,instructions)
    assert n==0 and zero==0
    zero,n=pair_loss(r,q,[1,1,1],[torch.zeros(1,dtype=torch.bool)]*3,groups,instructions)
    assert n==0 and zero==0


def test_no_label_or_metadata_in_forward_and_reg_does_not_change_eval():
    torch.manual_seed(17);a=TransferMatcher(False).eval()
    torch.manual_seed(17);b=TransferMatcher(True).eval()
    args=fixture_args();a.alpha.data.fill_(.2);b.load_state_dict(a.state_dict())
    for x,y in zip(a(**args),b(**args)):torch.testing.assert_close(x,y,rtol=0,atol=0)
    with pytest.raises(TypeError):a(**args,labels=torch.zeros(7))
    with pytest.raises(TypeError):a(**args,source='source')


def test_dropout_optimizer_resume_including_rng(tmp_path):
    torch.manual_seed(19);args=fixture_args()
    def create():
        torch.manual_seed(23);m=TransferMatcher(True);return m,torch.optim.AdamW(m.parameters(),lr=2e-4)
    def step(m,opt):
        opt.zero_grad(set_to_none=True);residual,r,q=m(**args)
        loss=(residual-torch.arange(7)/7).square().mean()+.1*m.anchor_penalty()+.01*(r-q).square().mean()
        loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
    a,oa=create();step(a,oa);step(a,oa)
    b,ob=create();step(b,ob)
    p=tmp_path/'resume.pt';torch.save(dict(state=b.state_dict(),optimizer=ob.state_dict(),rng=torch.get_rng_state()),p)
    c,oc=create();saved=torch.load(p,weights_only=True);c.load_state_dict(saved['state']);oc.load_state_dict(saved['optimizer']);torch.set_rng_state(saved['rng']);step(c,oc)
    for k,v in a.state_dict().items():torch.testing.assert_close(v,c.state_dict()[k],rtol=0,atol=0)


def test_batch_does_not_forward_labels_or_original_logits():
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    from run_transfer_alignment_v20 import batch
    rows=[dict(n=n,roi=torch.randn(n,4,128),roi_mask=torch.ones(n,4,dtype=torch.bool),scores=torch.randn(3,n),labels=torch.ones(n,dtype=torch.bool)) for n in [2,0,3]]
    args,lengths,selected,baseline=batch(rows,[0,1,2],1)
    assert set(args)=={'roi','roi_mask','query_ids','queries'} and lengths==[2,0,3]
    assert args['query_ids'].tolist()==[0,0,2,2,2] and baseline.shape==(5,)
    assert len(selected)==3
