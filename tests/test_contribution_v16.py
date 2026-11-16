import sys
from pathlib import Path
import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from clickagent.contribution_ablation import VARIANTS,transform_inputs
from clickagent.evidence_retention import RetentionModel,permute_evidence
from run_quality_readout_v15 import digest


def features():
    state=torch.arange(5*4*3*294,dtype=torch.float32).reshape(5,4,3,294)
    mask=torch.ones((5,4,3),dtype=torch.bool);mask[2:,2]=False;mask[:,:,2]=False
    return dict(base=torch.randn(5,32),active=torch.ones(5,4,dtype=torch.bool),state=state,mask=mask)


@pytest.mark.parametrize('variant',VARIANTS)
def test_shape_coordinates_metadata_and_masks_are_preserved(variant):
    inputs=features();original=inputs['state'].clone();got=transform_inputs(inputs,[3,2],['a','b'],variant)
    assert got['state'].shape==inputs['state'].shape
    for field in ('base','active','mask'):assert got[field] is inputs[field]
    torch.testing.assert_close(inputs['state'],original)
    assert sum(p.numel() for p in RetentionModel().parameters())==22050


def test_channel_controls_remove_only_the_registered_channels():
    x=features();a=transform_inputs(x,[5],['a'],'attention_only')['state'];h=transform_inputs(x,[5],['a'],'hidden_only')['state']
    assert torch.equal(a[...,:288],x['state'][...,:288]) and torch.count_nonzero(a[...,288:])==0
    assert torch.equal(h[...,288:],x['state'][...,288:]) and torch.count_nonzero(h[...,:288])==0
    assert torch.count_nonzero(transform_inputs(x,[5],['a'],'no_evidence')['state'])==0


def test_token_average_excludes_padding_and_preserves_view_presence():
    x=features();out=transform_inputs(x,[5],['a'],'token_mean')['state']
    expected=(x['state'][:,:,0]+x['state'][:,:,1])/2
    present=x['mask'][:,:,0]
    torch.testing.assert_close(out[:,:,0][present],expected[present]);torch.testing.assert_close(out[:,:,1][present],expected[present])
    assert torch.count_nonzero(out[:,:,2])==0 and torch.count_nonzero(out[2:,2])==0


def test_spatial_shuffle_matches_frozen_v14_control_and_is_query_local():
    x=features();out=transform_inputs(x,[3,2],['a','b'],'spatial_shuffle')['state'].numpy();start=0
    for sid,n in zip(['a','b'],[3,2]):
        rng=np.random.default_rng(int(digest([sid,'spatial_permutation'])[:8],16))
        expected=permute_evidence(x['state'][start:start+n].numpy(),x['mask'][start:start+n].numpy(),rng)
        np.testing.assert_array_equal(out[start:start+n],expected);start+=n
    got=transform_inputs(x,[3,2],['a','b'],'spatial_shuffle')['state']
    assert torch.equal(got,torch.as_tensor(out))


def test_unknown_variant_and_bad_query_boundaries_rejected():
    with pytest.raises(ValueError,match='Unknown'):transform_inputs(features(),[5],['a'],'best_test_seed')
    with pytest.raises(ValueError,match='boundaries'):transform_inputs(features(),[4],['a'],'no_evidence')


def test_runtime_adapters_restore_registered_parent_even_after_failure():
    import run_contribution_v16 as run
    original=(run.prior.OUT,run.prior.status,run.prior.tensors)
    with pytest.raises(RuntimeError,match='interrupt'):
        with run.adapters('no_evidence'):
            assert run.prior.OUT==run.OUT and run.prior.status is run.status
            raise RuntimeError('interrupt')
    assert (run.prior.OUT,run.prior.status,run.prior.tensors)==original


def test_no_independent_input_before_all18_checkpoints_frozen(tmp_path,monkeypatch):
    import run_contribution_v16 as run
    monkeypatch.setattr(run,'OUT',tmp_path)
    with pytest.raises(ValueError,match='Freeze18'):run.extract_independent()
    run.write(tmp_path/'models/frozen.json',dict(complete=False,models={}))
    with pytest.raises(ValueError,match='Incomplete contribution'):run.require_frozen()


def test_primary_family_bootstrap_retains_fractional_seed_average():
    import run_contribution_v16 as run
    value=run.bootstrap_mean([[1,0],[0,0],[0,0]],[[0,0]]*3,[dict(source='s',group='a'),dict(source='s',group='b')],primary=True)
    assert value['delta']==pytest.approx(1/6)
    assert value['primary_family_size']==3 and value['bonferroni_ci98_333'][1]==pytest.approx(1/3)


def test_completion_accepts_list_details_and_rejects_missing_rows():
    import run_contribution_v16 as run
    result=dict(complete=True,n=1,methods={'a':dict(top1_hits=1)})
    rows=[dict(sample_id='s',methods={'a':dict(top1=True)})]
    run.validate_completion_results(result,rows,1)
    with pytest.raises(ValueError,match='schema'):run.validate_completion_results(result,{},1)
    with pytest.raises(ValueError,match='schema'):run.validate_completion_results(result,[],1)
