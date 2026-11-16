import numpy as np
import pytest

from clickagent.candidate_data import annotation_candidates, candidate_labels
from clickagent.candidate_model import pool_candidate_features, diverse_points
from clickagent.candidate_probes import calibrated_threshold, policy_metrics, make_network, set_loss, fit_head, predict_checkpoint, evaluation_test_indices, evaluate


def example(**updates):
    case = {"sample_id": "x", "split": "train", "image_size": [100, 100], "intent": "click",
            "expected_actionable": True, "target_geometry": {"type": "bbox", "boxes": [[.1,.2,.3,.4]]}}
    return {**case, **updates}


def test_candidate_labels_use_regions_and_respect_intent():
    assert candidate_labels(example(), [[10,20], [29,39], [31,40]]) == [True, True, False]
    assert candidate_labels(example(intent="forbid"), [[20,30]]) == [False]
    assert candidate_labels(example(intent="describe"), [[20,30]]) == [False]
    assert candidate_labels(example(expected_actionable=False, target_geometry=None), [[20,30]]) == [False]


def test_annotation_sampling_is_forbidden_for_calibration_and_test():
    for split in ("calibration", "test"):
        with pytest.raises(ValueError):
            annotation_candidates(example(split=split), [])


def test_training_candidates_include_target_and_real_distractor():
    other = example(sample_id="y", target_geometry={"type":"bbox", "boxes":[[.7,.7,.9,.9]]})
    points = annotation_candidates(example(), [example(), other])
    assert len(points) == len(set(map(tuple, points))) == 4
    assert [20,30] in points and [80,80] in points
    assert points == annotation_candidates(example(), [example(), other])


def test_pooling_keeps_absolute_image_attention_mass():
    raw = np.array([[.01,.02,.03,.04], [.1,.2,.3,.4]], np.float32)
    visual = np.arange(24, dtype=np.float32).reshape(4,6)
    att, hidden, geometry = pool_candidate_features(raw, visual, [2,2], [[25,25],[75,75]], [100,100])
    assert att.shape == (2,8)
    np.testing.assert_allclose(att[:, -2:], [[.1,1.],[.1,1.]])
    np.testing.assert_allclose(hidden[0], .5*visual[0]+.5*visual.mean(0))
    np.testing.assert_allclose(geometry[1], [.75,.75,.5,.5])


def test_calibration_never_splits_ties_and_reject_all_is_representable():
    s = np.array([.8,.8,.9], np.float32).astype(np.float64)
    t = calibrated_threshold(s, [False,True,True], 0)
    assert t == s[2]
    t = calibrated_threshold(s, [False,False,False], 0)
    assert t > max(s)
    assert not np.any(s >= t)


def test_null_veto_cannot_be_overridden_by_calibration():
    scores = [.9,.8,.2]
    eligible = [True,True,False]
    threshold = calibrated_threshold(scores, [True,True,False], .1, eligible)
    cases = [{"case_kind":"click"}] * 3
    labels = np.array([[1,0,0,0], [1,0,0,0], [0,0,0,0]], bool)
    r = policy_metrics(cases, labels, np.zeros(3,int), scores, threshold, eligible)
    assert r["executed"] == 2 and r["wrong_clicks"] == 0


def test_risk_includes_wrong_point_on_click_request():
    cases = [{"case_kind":"click"}, {"case_kind":"forbid"}, {"case_kind":"click"}]
    labels = np.array([[1,0,0,0], [0,0,0,0], [0,1,0,0]], bool)
    r = policy_metrics(cases, labels, np.array([0,0,0]), [.9,.9,.9], .5)
    assert r["correct_clicks"] == 1
    assert r["wrong_clicks"] == 2
    assert r["wrong_location_clicks"] == 1
    assert r["nonactionable_false_clicks"] == 1


@pytest.mark.parametrize("kind", ["attention", "hidden", "combined"])
@pytest.mark.parametrize("candidate", [False, True])
def test_heads_handle_null_and_have_finite_training_gradients(kind, candidate):
    import torch
    dims = {"query_hidden":12,"candidate_visual":12,"global_attention":9,"candidate_attention":8,"geometry":8}
    model = make_network(kind, candidate, dims)
    data = {"query_hidden":torch.randn(3,12), "candidate_visual":torch.randn(3,4,12),
            "global_attention":torch.randn(3,9),"candidate_attention":torch.randn(3,4,8),"geometry":torch.randn(3,4,8)}
    labels = torch.tensor([[1.,0,0,0],[0,0,0,0],[1,1,0,0]])
    out = model(data)
    assert out.shape == ((3,5) if candidate else (3,))
    if candidate:
        loss = set_loss(out, labels)
        loss.backward()
        assert torch.isfinite(loss)
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize("query_selector",[False,True])
def test_test_labels_cannot_change_training_or_scores(tmp_path,query_selector):
    rng = np.random.default_rng(18)
    arrays = {"query_hidden":rng.normal(size=(24,12)).astype(np.float32),
              "candidate_visual":rng.normal(size=(24,4,12)).astype(np.float32),
              "global_attention":rng.normal(size=(24,9)).astype(np.float32),
              "candidate_attention":rng.normal(size=(24,4,8)).astype(np.float32),
              "geometry":rng.normal(size=(24,4,8)).astype(np.float32),
              "labels":rng.integers(0,2,size=(24,4)).astype(np.float32),
              "global_labels":rng.integers(0,2,size=24).astype(np.float32)}
    kind = "hidden" if query_selector else "combined"
    scores,info = fit_head(kind,not query_selector,arrays,np.arange(8),np.arange(8,12),tmp_path/"first.pt",query_selector=query_selector)
    arrays["labels"][12:] = 1-arrays["labels"][12:]
    arrays["global_labels"][12:] = 1-arrays["global_labels"][12:]
    changed,other = fit_head(kind,not query_selector,arrays,np.arange(8),np.arange(8,12),tmp_path/"second.pt",query_selector=query_selector)
    assert info["best_epoch"] == other["best_epoch"]
    np.testing.assert_array_equal(scores,changed)
    np.testing.assert_array_equal(scores,predict_checkpoint(tmp_path/"first.pt",arrays))


def test_smoke_exclusion_is_fixed_and_can_only_exclude_test(tmp_path):
    import json
    cases = [{"sample_id":"train","split":"train"},{"sample_id":"seen","split":"test"},{"sample_id":"fresh","split":"test"}]
    path = tmp_path/"evaluation_exclusions.json"
    path.write_text(json.dumps({"sample_ids":["seen"]}))
    indices,excluded = evaluation_test_indices(cases,tmp_path)
    assert indices.tolist()==[2] and excluded==["seen"]
    path.write_text(json.dumps({"sample_ids":["train"]}))
    with pytest.raises(ValueError):
        evaluation_test_indices(cases,tmp_path)


def test_evaluation_serializes_fixed_candidate_and_selection_metrics(tmp_path):
    import json
    (tmp_path/"augmentation").mkdir()
    for name in ("identity.json","manifest.jsonl","training_source.py","augmentation/manifest.jsonl"):
        (tmp_path/name).write_text("{}")
    cases=[{"sample_id":str(i),"split":split,"case_kind":kind} for i,(split,kind) in enumerate([
        ("train","click"),("dev","click"),("calibration","click"),("calibration","forbid"),
        ("test","click"),("test","click"),("test","forbid"),("test","refusal")])]
    labels=np.zeros((8,4),np.float32)
    labels[[0,1,2,4],0]=1
    labels[5,1]=1
    p=np.tile([.7,.1,.05,.05,.1],(8,1))
    p[3]=[.01,.01,.01,.01,.96]
    metadata=[{"points":[[1,1],[2,2],[3,3],[4,4]]} for _ in cases]
    r=evaluate(cases,metadata,{"labels":labels,"native_probabilities":p},{"native_choice_logits":p},tmp_path,{})
    assert r["n_test"]==4
    assert len((tmp_path/"predictions.jsonl").read_text().splitlines())==24
    assert r["methods"]["native_choice_logits"]["policies"]["select_four"]["0.05"]["test"]["wrong_clicks"]==3
