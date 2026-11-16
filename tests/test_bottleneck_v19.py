import numpy as np
import torch
import sys
from pathlib import Path
import hashlib
import pytest

torch.set_num_threads(2)

from clickagent.bottleneck_v19 import BottleneckModel, local_view_features, shuffled, probe_loss
from clickagent.evidence_retention import RetentionModel, retention_loss


def test_width32_is_an_exact_original_forward_replay():
    torch.manual_seed(17)
    original = RetentionModel().eval()
    probe = BottleneckModel(width=32).eval()
    probe.load_state_dict(original.state_dict())
    mask = torch.ones(5, 4, 3, dtype=torch.bool)
    mask[0] = False; mask[2, 1] = False
    args = dict(base=torch.randn(5, 32), state=torch.randn(5, 4, 3, 294), mask=mask)
    assert sum(p.numel() for p in probe.parameters()) == 22050
    torch.testing.assert_close(original(**args), probe(**args), rtol=0, atol=0)


def test_loss_original_parity_and_positive_null_cases():
    for labels in [torch.tensor([0., 1., 0.]), torch.zeros(3), torch.ones(3)]:
        scores = torch.tensor([.3, -.7, .9], requires_grad=True)
        null = torch.tensor(.2, requires_grad=True)
        a = retention_loss(scores, labels, null)
        b = probe_loss(scores, labels, null)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        for x, y in zip(torch.autograd.grad(a, [scores, null], retain_graph=True), torch.autograd.grad(b, [scores, null])):
            torch.testing.assert_close(x, y, rtol=0, atol=0)
        assert torch.isfinite(probe_loss(scores, labels, null, 0))


def test_fine_visual_mapping_and_public_geometry_mask():
    vision = np.zeros((16, 1152), np.float16)
    for i in range(16): vision[i, i] = 1
    arrays = dict(vision_hidden=vision, image_hidden=np.ones((4, 4096), np.float16),
                  target_hidden=np.ones((2, 12288), np.float16))
    meta = dict(grid_shape=[2, 2], fine_grid_shape=[4, 4],
                ground_truth_in_model_input=False, all_backbone_parameters_frozen=True)
    feature, valid = local_view_features(arrays, meta, [[5, 5], [35, 35], [50, 5]], [40, 40])
    assert feature.shape == (3, 160) and valid.tolist() == [True, True, False]
    assert not np.array_equal(feature[0, :32], feature[1, :32])
    assert np.count_nonzero(feature[2]) == 0
    shifted, mask = local_view_features(arrays, meta, [[15, 15]], [80, 80], dict(box=[10, 10, 50, 50]))
    np.testing.assert_array_equal(shifted[0], feature[0]); assert mask.all()


def test_shuffle_preserves_view_multisets_and_validity():
    features = np.arange(12*4*160).reshape(12, 4, 160)
    valid = np.ones((12, 4), bool); valid[:3, 1] = False
    a = shuffled(features, valid, 'sample'); b = shuffled(features, valid, 'sample')
    np.testing.assert_array_equal(a, b)
    for v in range(4):
        np.testing.assert_array_equal(np.sort(a[valid[:, v], v, 0]), np.sort(features[valid[:, v], v, 0]))
        np.testing.assert_array_equal(a[~valid[:, v], v], features[~valid[:, v], v])
    assert not np.array_equal(a, features)


def test_aligned_and_shuffled_controls_have_identical_initial_models():
    torch.manual_seed(29); a = BottleneckModel(visual=True)
    torch.manual_seed(29); b = BottleneckModel(visual=True)
    assert all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items())
    args = dict(base=torch.zeros(2, 32), state=torch.zeros(2, 4, 1, 294), mask=torch.zeros(2, 4, 1, dtype=torch.bool),
                roi=torch.zeros(2, 4, 160), roi_mask=torch.zeros(2, 4, dtype=torch.bool))
    loss = a(**args).sum(); assert torch.isfinite(loss)
    loss.backward(); assert all(p.grad is None or torch.isfinite(p.grad).all() for p in a.parameters())


def test_cached_batch_matches_original_padding_and_inputs():
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    from run_bottleneck_v19 import BatchCache
    from run_evidence_retention_v14 import tensors
    rng=np.random.default_rng(31);records=[];data=[]
    for n,t in [(3,2),(5,4)]:
        r=dict(n=n,base=rng.normal(size=(n,32)).astype(np.float32),
               state=rng.normal(size=(n,4,t,294)).astype(np.float16),
               mask=rng.integers(0,2,size=(n,4,t)).astype(bool),active=np.ones((n,4),bool))
        records.append(r)
        data.append({k:torch.tensor(r[k]) for k in ['base','state','mask']})
        data[-1].update(roi=torch.zeros(n,4,160),roi_mask=torch.ones(n,4,dtype=torch.bool),
                        permutation=torch.arange(n)[:,None].expand(n,4),labels=torch.zeros(n))
    cache=BatchCache.__new__(BatchCache);cache.data=data;cache.device=cache.storage='cpu'
    args,lengths,_=cache.batch([0,1]);reference,expected=tensors(records,np.zeros(32,np.float32),np.ones(32,np.float32),'aligned','cpu')
    assert lengths==expected
    for name in ['base','state','mask']:torch.testing.assert_close(args[name],reference[name],rtol=0,atol=0)


def test_optimizer_checkpoint_resume_matches_continuous_fit(tmp_path):
    torch.manual_seed(43)
    args=dict(base=torch.randn(4,32),state=torch.randn(4,4,2,294),mask=torch.ones(4,4,2,dtype=torch.bool),
              roi=torch.randn(4,4,160),roi_mask=torch.ones(4,4,dtype=torch.bool))
    labels=torch.tensor([0.,1.,0.,1.])
    def create():
        torch.manual_seed(47);m=BottleneckModel(64,True)
        return m,torch.optim.AdamW(m.parameters(),lr=2e-4,weight_decay=.02)
    def step(m,opt):
        opt.zero_grad(set_to_none=True);loss=probe_loss(m(**args),labels,m.null)
        loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
    a,oa=create();step(a,oa);step(a,oa)
    b,ob=create();step(b,ob)
    path=tmp_path/'resume.pt';torch.save(dict(state=b.state_dict(),optimizer=ob.state_dict()),path)
    c,oc=create();saved=torch.load(path,weights_only=True);c.load_state_dict(saved['state']);oc.load_state_dict(saved['optimizer']);step(c,oc)
    for name,v in a.state_dict().items():torch.testing.assert_close(v,c.state_dict()[name],rtol=0,atol=0)


def test_legacy_metadata_exception_is_narrow_and_still_checks_identity():
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    from run_bottleneck_v19 import authenticate_meta
    public=dict(sample_id='legacy:1',image_sha256='image',instruction='click Save')
    meta=dict(sample_id='legacy:1',image_sha256='image')
    authenticate_meta(meta,public,legacy_global=True)
    with pytest.raises(AssertionError):authenticate_meta(meta,public)
    with pytest.raises(AssertionError):authenticate_meta(dict(meta,image_sha256='wrong'),public,True)
    modern=dict(meta,instruction_sha256=hashlib.sha256(public['instruction'].encode()).hexdigest())
    authenticate_meta(modern,public)
    with pytest.raises(AssertionError):authenticate_meta(dict(modern,instruction_sha256='wrong'),public,True)
