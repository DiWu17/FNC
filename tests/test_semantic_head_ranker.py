import copy

import numpy as np
import pytest
import torch

from clickagent.attention_field_ranker import AttentionFieldRanker, FieldRankerConfig, build_attention_field
from clickagent.candidate_ranker import build_candidate_inputs, candidate_supervision, ranker_loss
from clickagent.semantic_head_ranker import (
    SemanticHeadRanker, SemanticHeadRankerConfig, load_semantic_head_ranker,
    make_field_batch, save_semantic_head_ranker,
)


@pytest.fixture(autouse=True)
def limited_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def field_record(seed=5, tokens=3):
    rng = np.random.default_rng(seed)
    grid, patches = (2, 3), 6
    attention = rng.random((tokens, 96, patches), dtype=np.float32) * .01
    result = build_candidate_inputs(attention,
        rng.normal(size=(tokens, 12288)).astype(np.float16),
        rng.normal(size=(patches, 4096)).astype(np.float16),
        rng.normal(size=(patches * 4, 1152)).astype(np.float16), grid)
    result["attention_field"] = build_attention_field(attention, result["parent_indices"])
    result.update(candidate_supervision({"type": "bbox", "boxes": [[.12, .15, .3, .45]]}, grid, result["parent_indices"]))
    result.update(metadata={"grid_shape": grid, "target_available": True},
                  case={"image_size": [813, 517]}, instruction="Open the example target")
    return result


def model(routing="semantic", activate=True):
    result = SemanticHeadRanker(SemanticHeadRankerConfig(width=8, head_width=8, groups=4, routing=routing),
                                np.linspace(0, .3, 96)).eval()
    if activate:
        torch.nn.init.normal_(result.field_rank[-1].weight, std=.2)
    return result


def test_same_seed_base_initialization_and_zero_residual_preserve_old_outputs():
    torch.manual_seed(91)
    old = AttentionFieldRanker(FieldRankerConfig(width=8), np.linspace(0, .3, 96)).eval()
    torch.manual_seed(91)
    new = model(activate=False)
    for name in new._base_state_keys:
        torch.testing.assert_close(new.state_dict()[name], old.state_dict()[name], rtol=0, atol=0)
    inputs, _ = make_field_batch([field_record()], "cpu", labels=False)
    before, after = old(**inputs), new(**inputs)
    for name in ("scores", "null_scores", "child_logits", "offsets", "source_weights"):
        torch.testing.assert_close(before[name], after[name], rtol=0, atol=0)
    assert torch.count_nonzero(after["field_residual"]) == 0
    assert new.query[0].out_features == new.config.width == 8
    assert "head_routing_weights" not in after


def test_matched_static_zeros_only_new_explicit_target_condition():
    torch.manual_seed(8)
    semantic, static = model(), model("static")
    static.load_state_dict(semantic.state_dict(), strict=True)
    assert {k: v.shape for k, v in semantic.state_dict().items()} == {k: v.shape for k, v in static.state_dict().items()}
    assert sum(p.numel() for p in semantic.parameters()) == sum(p.numel() for p in static.parameters())
    inputs, _ = make_field_batch([field_record()], "cpu", labels=False)
    changed = dict(inputs, query_hidden=torch.randn_like(inputs["query_hidden"]))
    a, b = static(**inputs, return_routing=True), static(**changed, return_routing=True)
    for name in ("head_routing_weights", "group_token_weights", "field_residual", "field_token_weights"):
        torch.testing.assert_close(a[name], b[name], rtol=0, atol=0)
    assert not torch.allclose(a["base_scores"], b["base_scores"])
    a, b = semantic(**inputs, return_routing=True), semantic(**changed, return_routing=True)
    assert not torch.allclose(a["head_routing_weights"], b["head_routing_weights"])
    assert not torch.allclose(a["field_residual"], b["field_residual"])
    assert semantic.route_query.bias is None and semantic.route_phrase.bias is None


def test_all_six_channels_heads_and_tokens_survive_until_conditioned_soft_pooling():
    torch.manual_seed(10)
    ranker = model()
    record = field_record(tokens=4)
    inputs, _ = make_field_batch([record], "cpu", labels=False)
    inputs["attention_field"] = inputs["attention_field"].float().requires_grad_(True)
    observed = []
    handle = ranker.head_field.register_forward_pre_hook(lambda module, args: observed.append(args[0].detach().clone()))
    output = ranker(**inputs, return_routing=True)
    handle.remove()
    count = len(record["parent_indices"])
    assert observed[0].shape == (count * 4, 96, 6)
    torch.testing.assert_close(observed[0], inputs["attention_field"], rtol=0, atol=0)
    weights = output["head_routing_weights"]
    assert weights.shape == (count * 4, 96, 4) and (weights > 0).all()
    torch.testing.assert_close(weights.sum(1), torch.ones(count * 4, 4))
    token = output["group_token_weights"]
    sums = token.new_zeros(count, 4).index_add(0, inputs["field_candidate"], token)
    torch.testing.assert_close(sums, torch.ones(count, 4))
    averaged = output["field_token_weights"].new_zeros(count).index_add(
        0, inputs["field_candidate"], output["field_token_weights"])
    torch.testing.assert_close(averaged, torch.ones(count))
    output["field_residual"].sum().backward()
    # Every candidate/token/head gets a differentiable route, including heads
    # that did not propose this parent. No token or head is hard discarded.
    gradient = inputs["attention_field"].grad.abs().sum(-1)
    assert gradient.shape == (count * 4, 96) and (gradient > 0).all()
    assert (ranker.head_identity.weight.grad.abs().sum(-1) > 0).all()


def test_candidate_evidence_is_independent_and_never_changes_fine_predictions():
    torch.manual_seed(19)
    ranker = model()
    inputs, _ = make_field_batch([field_record()], "cpu", labels=False)
    changed = dict(inputs, attention_field=inputs["attention_field"].clone())
    mask = inputs["field_candidate"] == 0
    changed["attention_field"][mask, :, 0] += 3
    before, after = ranker(**inputs), ranker(**changed)
    assert not torch.allclose(before["field_residual"][:1], after["field_residual"][:1])
    torch.testing.assert_close(before["field_residual"][1:], after["field_residual"][1:], rtol=0, atol=0)
    for name in ("base_scores", "null_scores", "child_logits", "offsets"):
        torch.testing.assert_close(before[name], after[name], rtol=0, atol=0)


def test_batched_cases_and_permuted_field_rows_do_not_mix_evidence():
    torch.manual_seed(21)
    ranker = model()
    records = [field_record(3, tokens=2), field_record(7, tokens=4)]
    inputs, _ = make_field_batch(records, "cpu", labels=False)
    together = ranker(**inputs)
    cursor = 0
    for case_index, record in enumerate(records):
        batch, _ = make_field_batch([record], "cpu", labels=False)
        separate = ranker(**batch)
        n = len(record["parent_indices"])
        for name in ("scores", "field_residual", "child_logits", "offsets"):
            torch.testing.assert_close(together[name][cursor:cursor+n], separate[name], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(together["null_scores"][case_index], separate["null_scores"][0], atol=1e-5, rtol=1e-5)
        cursor += n
    permutation = torch.randperm(len(inputs["field_token"]))
    permuted = dict(inputs)
    for name in ("attention_field", "field_token", "field_candidate"):
        permuted[name] = inputs[name][permutation]
    result = ranker(**permuted)
    torch.testing.assert_close(together["scores"], result["scores"], atol=1e-5, rtol=1e-5)


def test_labels_case_geometry_and_metadata_do_not_enter_inputs():
    record = field_record()
    changed = copy.deepcopy(record)
    changed.update(label_area=np.ones_like(record["label_area"]), label_offsets=np.ones_like(record["label_offsets"]),
                   case={"target_geometry": {"type": "bbox", "boxes": [[0, 0, 1, 1]]}},
                   metadata={"label": "forbid", "dataset": "different"}, instruction="Unrelated ground truth text")
    before, labels = make_field_batch([record], "cpu", labels=False)
    after, other_labels = make_field_batch([changed], "cpu", labels=False)
    assert labels is other_labels is None
    assert not any("label" in name or "geometry" in name or "instruction" in name for name in before)
    for name in before:
        if isinstance(before[name], torch.Tensor):
            torch.testing.assert_close(before[name], after[name], rtol=0, atol=0)
        else:
            assert before[name] == after[name]


@pytest.mark.parametrize("routing", ["semantic", "static"])
def test_original_loss_and_router_have_finite_nonzero_gradients(routing):
    torch.manual_seed(27)
    ranker = model(routing)
    inputs, labels = make_field_batch([field_record()], "cpu", labels=True)
    loss, _ = ranker_loss(ranker(**inputs), **labels)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in ranker.parameters() if p.grad is not None)
    assert ranker.head_field[0].weight.grad.abs().sum() > 0
    assert ranker.head_group_score.weight.grad.abs().sum() > 0
    if routing == "static":
        assert ranker.route_query.weight.grad.abs().sum() == 0
        assert ranker.route_phrase.weight.grad.abs().sum() == 0
    else:
        assert ranker.route_query.weight.grad.abs().sum() > 0
        assert ranker.route_phrase.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("frozen", [False, True])
def test_save_load_and_optional_strict_base_freeze(tmp_path, frozen):
    torch.manual_seed(28)
    ranker = model()
    if frozen:
        base_state = {name: ranker.state_dict()[name].clone() for name in ranker._base_state_keys}
        ranker.load_base_ranker_state(base_state, freeze=True).train()
        assert not ranker.rank[2].training and ranker.field_rank[2].training
        assert all(not p.requires_grad for name, p in ranker.named_parameters() if name in ranker._base_parameter_names)
        with pytest.raises(ValueError, match="exactly"):
            ranker.load_base_ranker_state({**base_state, "unexpected": torch.ones(1)})
    ranker.eval()
    inputs, _ = make_field_batch([field_record()], "cpu", labels=False)
    before = ranker(**inputs, return_routing=True)
    path = tmp_path / "head.pt"
    save_semantic_head_ranker(path, ranker, {"purpose": "CPU roundtrip", "public_labels_used": False})
    loaded, metadata = load_semantic_head_ranker(path)
    assert loaded.base_frozen is frozen and loaded.training is False
    assert metadata == {"purpose": "CPU roundtrip", "public_labels_used": False}
    after = loaded(**inputs, return_routing=True)
    for name in ("scores", "null_scores", "child_logits", "offsets", "head_routing_weights", "field_token_weights"):
        torch.testing.assert_close(before[name], after[name], rtol=0, atol=0)


def test_default_controls_have_identical_parameter_budgets_and_small_head_width():
    semantic = SemanticHeadRanker(SemanticHeadRankerConfig(), np.zeros(96))
    static = SemanticHeadRanker(SemanticHeadRankerConfig(routing="static"), np.zeros(96))
    assert sum(p.numel() for p in semantic.parameters()) == sum(p.numel() for p in static.parameters())
    assert semantic.semantic_config.head_width == 16 and semantic.semantic_config.groups == 4
    assert semantic.head_field[0].in_features == 6 and semantic.head_group_score.out_features == 4


@pytest.mark.parametrize("kwargs", [{"width": 0}, {"head_width": 0}, {"groups": 0}, {"routing": "flat"}])
def test_invalid_config_rejected(kwargs):
    with pytest.raises(ValueError):
        SemanticHeadRankerConfig(**kwargs)
