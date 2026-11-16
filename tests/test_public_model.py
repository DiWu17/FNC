import copy
import inspect
import json
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

import clickagent.action_model as action
import clickagent.public_model as public
from clickagent.model_features import ExtractionConfig


class Processor:
    def __init__(self, torch, pieces):
        self.torch = torch
        self.pieces = pieces
        self.messages = []
        self.tokenizer = SimpleNamespace(decode=self.decode)

    def decode(self, ids, **kwargs):
        return "".join(self.pieces[token] for token in ids)

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {"tokenize": False, "add_generation_prompt": True}
        self.messages.append(copy.deepcopy(messages))
        return json.dumps(messages, ensure_ascii=False)

    def __call__(self, **kwargs):
        torch = self.torch
        assert kwargs["padding"] is False and kwargs["return_tensors"] == "pt"
        assert len(kwargs["images"]) == 1 and kwargs["images"][0].mode == "RGB"

        class Batch(dict):
            def to(self, device):
                return Batch({key: value.to(device) for key, value in self.items()})

        ids = torch.tensor([[3, 97, 95, 98, 4]])
        return Batch(input_ids=ids, attention_mask=torch.ones_like(ids), image_grid_thw=torch.tensor([[1, 2, 2]]))


class SequenceModel:
    """Known logits permit checking that no category or coordinate is forced."""

    config = SimpleNamespace(image_token_id=95, vision_config=SimpleNamespace(spatial_merge_size=2))

    def __init__(self, torch, sequence):
        self.torch, self.sequence = torch, sequence
        self.calls = []
        self.recording = False

    def __call__(self, **kwargs):
        torch = self.torch
        assert kwargs["output_attentions"] is False and kwargs["output_hidden_states"] is False
        assert kwargs["use_cache"] is True and kwargs["logits_to_keep"] == 1
        token = self.sequence[len(self.calls)] if len(self.calls) < len(self.sequence) else 99
        self.calls.append({key: value.clone() if torch.is_tensor(value) else value for key, value in kwargs.items()})
        logits = torch.full((1, 1, 100), -100.0)
        logits[0, 0, token] = 1
        return SimpleNamespace(logits=logits, past_key_values=object())


class QueryRecorder:
    """Features encode the current query, independently of the predicted token."""

    def __init__(self, model, layers):
        self.model = model

    def __enter__(self):
        assert not self.model.recording
        self.model.recording = True
        return self

    def __exit__(self, *exc):
        self.model.recording = False

    def begin(self, positions, grid):
        np.testing.assert_array_equal(positions, [2])
        assert grid == (1, 1)

    def snapshot(self):
        query = self.model.calls[-1]["input_ids"][0, -1].item()
        return {"attention": np.array([query], np.float32), "hidden": np.array([query], np.float32),
                "raw_image_attention": np.array([[query]], np.float32),
                "attention_names": ["test_attention"], "hidden_names": ["test_hidden"],
                "head_order": ["layer_1.head_0"]}


def fake_runtime(monkeypatch, pieces, *, cap=24):
    torch = pytest.importorskip("torch")
    sequence = list(range(10, 10 + len(pieces)))
    model = SequenceModel(torch, sequence)
    processor = Processor(torch, dict(zip(sequence, pieces)))
    loaded = SimpleNamespace(model=model, processor=processor, device=torch.device("cpu"), eos_ids={99},
                             runtime_metadata={"all_parameters_frozen": True})
    loads = []
    monkeypatch.setattr(action, "FrozenQwenExtractor", lambda config: loads.append(config) or loaded)
    monkeypatch.setattr(public, "_sync_cuda", lambda: None)
    monkeypatch.setattr(public, "RawSelectedLayerRecorder", QueryRecorder)
    runtime = public.PublicQwenRuntime(ExtractionConfig("unused", task="visual_state", layers=(1,), max_new_tokens=cap))
    return runtime, model, processor, loads


def test_joint_true_generated_word_and_own_query_forward_with_generic_instruction(monkeypatch):
    runtime, model, processor, loads = fake_runtime(monkeypatch, ["DIS", "ABLED"])
    instruction = "Open the network settings panel"
    result = runtime.infer_joint(Image.new("RGB", (100, 80)), instruction)
    assert len(loads) == 1
    assert [call["input_ids"].tolist() for call in model.calls] == [[[3, 97, 95, 98, 4]], [[10]], [[11]]]
    assert [call["attention_mask"].shape[-1] for call in model.calls] == [5, 6, 7]
    assert [call["cache_position"].tolist() for call in model.calls[1:]] == [[5], [6]]
    assert result["prefill"]["hidden"].tolist() == [4]  # Last prompt query, not its predicted DIS token.
    assert result["post_decision"]["hidden"].tolist() == [11]  # Complete generated word's own query.
    assert result["prefill"]["query_token_index"] == 4
    assert result["post_decision"]["query_token_index"] == 6
    assert result["post_decision"]["generated_tokens_at_readout"] == 2
    assert result["generated_decision"] == "disabled" and result["parse_valid"] is True
    assert result["generated_text"] == "DISABLED" and result["generated_token_ids"] == [10, 11]
    assert result["decision_completion_token_index"] == 1
    assert result["prefill_raw_map"].tolist() == [[4]]
    assert result["post_decision"]["raw_image_attention"].tolist() == [[11]]
    assert result["shared_trajectory"] is True and result["independent_method_latency_comparison"] is False
    assert result["protocol"] == "generalized_prompt_zero_retrain_transfer"
    assert result["gate_or_grounder_retrained"] is False and result["model_weights_retrained"] is False
    assert result["ground_truth_in_model_inputs"] is False and result["max_pixels"] == 262144
    times = result["timing_seconds"]
    assert 0 <= times["prepare"] <= times["prefill_ready"] <= times["first_character"]
    assert times["first_character"] <= times["word_ready"] <= times["post_decision_ready"] <= times["total"]
    assert processor.messages == [[{"role": "system", "content": public.PUBLIC_DECISION_SYSTEM_PROMPT},
                                   {"role": "user", "content": [{"type": "image"},
                                                                  {"type": "text", "text": instruction}]}]]
    assert not model.recording


@pytest.mark.parametrize("pieces,cap,decision,eos,truncated", [
    (["UNKNOWN"], 4, "unknown", False, False),
    (["ABSENT"], 4, "absent", False, False),
    (["CLICK"], 4, "click", False, False),
    (["Maybe", " later"], 2, None, False, True),
    (["Maybe"], 4, None, True, False),
])
def test_joint_abstention_and_unparsed_coverage(monkeypatch, pieces, cap, decision, eos, truncated):
    runtime, model, _, _ = fake_runtime(monkeypatch, pieces, cap=cap)
    result = runtime.infer_joint(Image.new("RGB", (60, 50)), "Choose the red triangle")
    assert result["generated_decision"] == decision
    assert result["prefill_available"] is True
    assert result["post_decision_available"] is (decision is not None)
    assert (result["post_decision"] is not None) is (decision is not None)
    assert result["decision_abstained"] is (decision == "unknown")
    assert result["parse_valid"] is (decision is not None)
    assert result["terminated_by_eos"] is eos and result["truncated"] is truncated
    assert not model.recording


@pytest.mark.parametrize("pieces,expected_point,eos,cap", [
    (["CLICK ", "5", "00 ", "5", "00", "\n"], [50, 40], False, 32),
    (["CLICK ", "500 ", "500"], [50, 40], True, 32),
    (["WAIT"], None, True, 32),
    (["UNKNOWN", "\n"], None, False, 32),
    (["CLICK ", "500 ", "5"], None, False, 3),
    (["CLICK ", "1001 ", "0", "\n"], None, True, 32),
])
def test_generic_coordinates_have_no_hooks_and_wait_for_boundary(monkeypatch, pieces, expected_point, eos, cap):
    runtime, model, processor, _ = fake_runtime(monkeypatch, pieces)
    monkeypatch.setattr(public, "RawSelectedLayerRecorder", lambda *args: pytest.fail("coordinate baseline installed hooks"))
    result = runtime.infer_text_action_public(Image.new("RGB", (100, 80)), "点击右侧的搜索图标", max_new_tokens=cap)
    assert result["point"] == expected_point
    assert result["hooks_used"] is False and result["shared_trajectory"] is False
    assert result["generated_tokens"] == len(pieces)
    assert result["terminated_by_eos"] is eos
    assert result["protocol"] == public.PUBLIC_PROTOCOL
    assert processor.messages[0][0]["content"] == public.PUBLIC_TEXT_ACTION_SYSTEM_PROMPT
    assert processor.messages[0][1]["content"][-1]["text"] == "点击右侧的搜索图标"
    assert not model.recording
    if cap == 3:
        assert result["truncated"] and not result["parse_valid"]
        assert result["unexecuted_candidate_at_token_limit"]["normalized_point"] == [500, 5]
        assert result["t_ready"] is None
    elif "1001 " in pieces:
        assert not result["parse_valid"] and result["action"] == "wait"
    else:
        assert result["parse_valid"] and result["t_ready"] >= result["t_first"] >= 0


def test_public_pixel_budget_rejected_before_loading(monkeypatch):
    monkeypatch.setattr(action, "FrozenQwenExtractor", lambda *args: pytest.fail("invalid budget loaded a model"))
    with pytest.raises(ValueError, match="max_pixels=262144"):
        public.PublicQwenRuntime(ExtractionConfig("unused", task="visual_state", max_pixels=524288))


@pytest.mark.parametrize("instruction", ["", "   ", None, {"instruction": "Open", "target_bbox": [0, 0, 1, 1]}])
def test_no_truth_or_manifest_payload_in_instructions(monkeypatch, instruction):
    runtime, model, processor, _ = fake_runtime(monkeypatch, ["CLICK"])
    with pytest.raises(ValueError, match="nonempty string"):
        runtime.infer_joint(Image.new("RGB", (60, 50)), instruction)
    assert not model.calls and not processor.messages


def test_public_api_and_prompts_have_no_fixed_target_or_truth_parameters():
    assert list(inspect.signature(public.PublicQwenRuntime.infer_joint).parameters) == ["self", "image", "instruction"]
    assert list(inspect.signature(public.PublicQwenRuntime.infer_text_action_public).parameters) == [
        "self", "image", "instruction", "max_new_tokens"]
    for prompt in (public.PUBLIC_DECISION_SYSTEM_PROMPT, public.PUBLIC_TEXT_ACTION_SYSTEM_PROMPT):
        assert "保存" not in prompt and "save" not in prompt.lower()


def test_actual_tiny_qwen_cpu_own_word_readout_and_no_leaked_hooks(monkeypatch):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    config = transformers.Qwen3VLConfig(
        text_config={"vocab_size": 100, "hidden_size": 32, "intermediate_size": 64,
                     "num_hidden_layers": 3, "num_attention_heads": 4, "num_key_value_heads": 2,
                     "head_dim": 8, "rope_scaling": {"rope_type": "default", "mrope_section": [1, 1, 2]}},
        vision_config={"hidden_size": 32, "out_hidden_size": 32, "intermediate_size": 64,
                       "depth": 3, "num_heads": 4, "deepstack_visual_indexes": [0, 1, 2]},
        image_token_id=95, video_token_id=96, vision_start_token_id=97, vision_end_token_id=98)
    config._attn_implementation = "sdpa"
    model = transformers.Qwen3VLForConditionalGeneration(config).cpu().eval().requires_grad_(False)
    processor = Processor(torch, {})
    # Token identity is deliberately irrelevant to the tiny random vocabulary;
    # cumulative decode needs two actual greedy token IDs to complete the word.
    processor.tokenizer.decode = lambda ids, **kwargs: "DIS" if len(ids) == 1 else "DISABLED"
    loaded = SimpleNamespace(model=model, processor=processor, device=torch.device("cpu"), eos_ids={1000},
                             runtime_metadata={"all_parameters_frozen": True})
    loads = []
    monkeypatch.setattr(action, "FrozenQwenExtractor", lambda config: loads.append(config) or loaded)
    monkeypatch.setattr(public, "_sync_cuda", lambda: None)
    runtime = public.PublicQwenRuntime(ExtractionConfig("unused", task="visual_state", layers=(1, 3), max_new_tokens=3))
    observed = []
    handle = model.register_forward_pre_hook(lambda module, args, kwargs: observed.append(kwargs["input_ids"].clone()),
                                              with_kwargs=True)
    result = runtime.infer_joint(Image.new("RGB", (100, 80)), "Select the sound output dropdown")
    handle.remove()
    assert len(loads) == 1 and len(observed) == 3
    assert observed[1].item() == result["generated_token_ids"][0]
    assert observed[2].item() == result["generated_token_ids"][1]
    assert result["prefill"]["query_token_index"] == 4
    assert result["post_decision"]["query_token_index"] == 6
    assert result["generated_decision"] == "disabled"
    assert result["prefill"]["hidden"].shape == result["post_decision"]["hidden"].shape == (64,)
    assert result["prefill"]["attention"].shape == (144,)
    assert result["prefill_raw_map"].shape == (8, 1)
    assert result["head_order"] == [f"layer_{layer}.head_{head}" for layer in (1, 3) for head in range(4)]
    assert result["timing_seconds"]["post_decision_ready"] >= result["timing_seconds"]["word_ready"]
    for layer in model.model.language_model.layers:
        assert not layer._forward_hooks and not layer.self_attn._forward_hooks
    assert not any(parameter.requires_grad for parameter in model.parameters())

