import numpy as np
import torch
import pytest

from clickagent.internal_candidate_scorer import InternalCandidateScorer, InternalCandidateScorerConfig, build_internal_inputs, internal_inputs_to_torch
from clickagent.multiregion_scorer import MultiRegionScorer, build_multiregion_inputs, top4_retention_loss


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def raw(seed, t=2, grid=(2, 3)):
    rng = np.random.default_rng(seed); p = int(np.prod(grid))
    return dict(target_attention=rng.random((t, 96, p)).astype(np.float32)*.01,
        target_hidden=rng.normal(size=(t, 12288)).astype(np.float16),
        image_hidden=rng.normal(size=(p, 4096)).astype(np.float16),
        vision_hidden=rng.normal(size=(p*4, 1152)).astype(np.float16)), dict(grid_shape=grid)


def sample(nlocal):
    ga, gm = raw(1)
    locals_ = [(*raw(i+3, 2+i, (3, 2)), dict(box=[i*200, 0, i*200+400, 400])) for i in range(nlocal)]
    pts = np.array([[100, 100], [300, 200], [700, 300], [950, 550]])
    return ga, gm, locals_, pts


def test_c1_exact_same_inputs_parameters_scores_and_gradients():
    ga, gm, ls, pts = sample(1)
    args = build_multiregion_inputs(ga, gm, ls, pts, (1000, 600))
    original = build_internal_inputs(ga, gm, ls[0][0], ls[0][1], pts, (1000, 600), ls[0][2])
    for key in args:
        np.testing.assert_array_equal(args[key], original[key])
    cfg = InternalCandidateScorerConfig(width=32, head_width=4, dropout=0)
    torch.manual_seed(8); base = InternalCandidateScorer(cfg).eval()
    torch.manual_seed(8); multi = MultiRegionScorer(cfg).eval()
    assert sum(p.numel() for p in base.parameters()) == sum(p.numel() for p in multi.parameters())
    data = internal_inputs_to_torch(args)
    out1, out2 = base(**data), multi(**data)
    torch.testing.assert_close(out1['scores'], out2['scores'], rtol=0, atol=0)
    out1['scores'].sum().backward(); out2['scores'].sum().backward()
    for (name, p), (nm, q) in zip(base.named_parameters(), multi.named_parameters()):
        assert name == nm
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)


def test_c3_local_order_invariant_outside_masked_and_finite_gradients():
    ga, gm, ls, pts = sample(3)
    cfg = InternalCandidateScorerConfig(width=32, head_width=4, dropout=0)
    torch.manual_seed(1); m = MultiRegionScorer(cfg).eval()
    a = internal_inputs_to_torch(build_multiregion_inputs(ga, gm, ls, pts, (1000, 600)))
    b = internal_inputs_to_torch(build_multiregion_inputs(ga, gm, ls[::-1], pts, (1000, 600)))
    out = m(**a, return_diagnostics=True)
    torch.testing.assert_close(out['scores'], m(**b)['scores'], rtol=1e-5, atol=1e-6)
    assert out['valid_views'][-1].tolist() == [True, False, False, False]
    out['scores'].sum().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())


def test_top4_objective_allows_three_stronger_negatives_but_not_four():
    y = torch.tensor([1., 0, 0, 0, 0, 0])
    # Identical scores except fourth negative: margin rank term changes by2.
    s1 = torch.tensor([1., 4, 3, 2, 0, -1], requires_grad=True)
    s2 = torch.tensor([1., 4, 3, 2, 2, -1], requires_grad=True)
    def rank(s):
        loss = top4_retention_loss(dict(scores=s, null_scores=torch.tensor([99.])), y)
        b = torch.nn.functional.binary_cross_entropy_with_logits(s, y, reduction='none')
        return loss-.25*(.5*b[y.bool()].mean()+.5*b[~y.bool()].mean())
    assert rank(s1).item() == pytest.approx(0)
    assert rank(s2).item() == pytest.approx(2)
    rank(s2).backward(); assert s2.grad[0] < 0


def test_no_positive_loss_equals_original_and_no_gt_builder_argument():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
    from train_candidate_scorer import ranking_loss
    scores = torch.randn(9, requires_grad=True)
    out = dict(scores=scores, null_scores=torch.tensor([.3], requires_grad=True))
    torch.testing.assert_close(top4_retention_loss(out, torch.zeros(9)), ranking_loss(out, torch.zeros(9)))
    ga, gm, ls, pts = sample(3)
    with pytest.raises(TypeError):
        build_multiregion_inputs(ga, gm, ls, pts, (1000, 600), labels='poison')
