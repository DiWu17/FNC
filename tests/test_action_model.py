import copy
import json
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

import clickagent.action_model as action
from clickagent.model_features import ExtractionConfig, SelectedLayerRecorder, system_prompt_for_task


@pytest.mark.parametrize("text,point,abstained", [
    ("CLICK 0 1000\n", [0, 1000], False), ("CLICK 500 501", [500, 501], False),
    ("WAIT\n", None, False), ("UNKNOWN", None, True),
])
def test_text_action_parser_valid(text, point, abstained):
    parsed = action.parse_text_action(text)
    assert parsed["normalized_point"] == point and parsed["abstained"] is abstained


@pytest.mark.parametrize("text", ["CLICK 1001 0", "CLICK -1 2", "CLICK 1.2 2", "CLICK 1 2 3", "CLICK 1",
                                  "CLICK 1 2 explanation", "```CLICK 1 2```", "WAITING", "click 1 2", "CLICK +1 2"])
def test_text_action_parser_rejects_malformed(text):
    assert action.parse_text_action(text) is None


def test_streaming_coordinate_prefix_requires_boundary():
    assert action.parse_text_action("CLICK 500 5", final=False) is None
    assert action.parse_text_action("CLICK 500 500", final=False) is None
    assert action.parse_text_action("CLICK 500 500\n", final=False)["normalized_point"] == [500, 500]
    assert action.parse_text_action("WAIT", final=False) is None


class Processor:
    def __init__(self, torch, pieces=None):
        self.torch = torch
        self.messages = []
        self.images = []
        self.pieces = pieces
        self.tokenizer = SimpleNamespace(decode=self.decode)

    def decode(self, ids, **kwargs):
        return "UNKNOWN" if self.pieces is None else "".join(self.pieces[token] for token in ids)

    def apply_chat_template(self, messages, **kwargs):
        self.messages.append(copy.deepcopy(messages))
        return json.dumps(messages, ensure_ascii=False)

    def __call__(self, **kwargs):
        torch = self.torch
        self.images.extend(kwargs["images"])

        class Batch(dict):
            def to(self, device):
                return Batch({key: value.to(device) for key, value in self.items()})

        ids = torch.tensor([[3, 97, 95, 98, 4]])
        return Batch(input_ids=ids, attention_mask=torch.ones_like(ids), image_grid_thw=torch.tensor([[1, 2, 2]]))


def tiny_model(torch):
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration
    config = Qwen3VLConfig(
        text_config={"vocab_size": 100, "hidden_size": 32, "intermediate_size": 64,
                     "num_hidden_layers": 3, "num_attention_heads": 4, "num_key_value_heads": 2,
                     "head_dim": 8, "rope_scaling": {"rope_type": "default", "mrope_section": [1, 1, 2]}},
        vision_config={"hidden_size": 32, "out_hidden_size": 32, "intermediate_size": 64,
                       "depth": 3, "num_heads": 4, "deepstack_visual_indexes": [0, 1, 2]},
        image_token_id=95, video_token_id=96, vision_start_token_id=97, vision_end_token_id=98)
    config._attn_implementation = "sdpa"
    return Qwen3VLForConditionalGeneration(config).cpu().eval().requires_grad_(False)


def test_actual_tiny_raw_readout_keeps_original_features_and_shared_prefill_maps(monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    model = tiny_model(torch)
    processor = Processor(torch)
    loaded = SimpleNamespace(model=model, processor=processor, device=torch.device("cpu"), eos_ids={1000}, runtime_metadata={})
    loads = []
    monkeypatch.setattr(action, "FrozenQwenExtractor", lambda config: loads.append(config) or loaded)
    monkeypatch.setattr(action, "_sync_cuda", lambda: None)
    config = ExtractionConfig("unused", task="visual_state", layers=(1, 3), max_new_tokens=3)
    runtime = action.ActionQwenRuntime(config)
    image = Image.new("RGB", (100, 80), "white")
    calls = []
    counter = model.register_forward_pre_hook(lambda *args: calls.append(1))
    pre = runtime.infer_readouts(image, stage="prefill")
    assert len(calls) == 1
    post = runtime.infer_readouts(image, stage="post_decision")
    assert len(calls) == 3  # One prefill plus the generated category token's own forward.
    direct = runtime.infer_readouts(image, stage="decision_text")
    assert len(calls) == 4  # The word baseline stops without forwarding that word token.
    counter.remove()
    assert len(loads) == 1
    assert pre["hidden"].shape == (64,) and pre["attention"].shape == (144,)
    assert pre["raw_image_attention"].shape == (8, 1)
    assert pre["generated_tokens"] == 0 and pre["t_first"] is None
    assert post["generated_decision"] == "unknown" and post["generated_tokens"] == 1
    assert post["readout_available"] is True and post["decision_abstained"] is True
    assert post["t_ready"] >= post["t_first"] >= 0
    np.testing.assert_array_equal(pre["prefill_raw_map"], post["prefill_raw_map"])
    np.testing.assert_array_equal(pre["prefill_raw_map"], direct["prefill_raw_map"])
    assert direct["generated_decision"] == "unknown" and direct["readout_available"] is True
    assert direct["hidden"] is None and direct["attention"] is None
    assert pre["head_order"] == [f"layer_{layer}.head_{head}" for layer in (1, 3) for head in range(4)]
    assert all(messages[0]["content"] == system_prompt_for_task("visual_state") for messages in processor.messages)
    assert all(messages[1]["content"] == [{"type": "image"}, {"type": "text", "text": "点击保存"}]
               for messages in processor.messages)
    ids = torch.tensor([[3, 97, 95, 98, 4]])
    with torch.inference_mode(), SelectedLayerRecorder(model, (1, 3)) as recorder:
        recorder.begin(np.array([2]), (1, 1))
        model(input_ids=ids, attention_mask=torch.ones_like(ids), image_grid_thw=torch.tensor([[1, 2, 2]]),
              use_cache=True, logits_to_keep=1)
        old = recorder.snapshot()
    np.testing.assert_array_equal(old["attention"], pre["attention"])
    np.testing.assert_array_equal(old["hidden"], pre["hidden"])
    assert not model.model.language_model.layers[0].self_attn._forward_hooks


@pytest.mark.parametrize("pieces,expected_point,eos,cap", [
    (["CLICK ", "5", "00 ", "5", "00", "\n"], [50, 40], False, 32),
    (["CLICK ", "500 ", "500"], [50, 40], True, 32),
    (["WAIT"], None, True, 32),
    (["UNKNOWN", "\n"], None, False, 32),
    (["CLICK ", "500 ", "5"], None, False, 3),
])
def test_text_baseline_waits_for_full_action_and_uses_no_hooks(monkeypatch, pieces, expected_point, eos, cap):
    torch = pytest.importorskip("torch")
    sequence = list(range(10, 10 + len(pieces)))

    class Model:
        config = SimpleNamespace(image_token_id=95, vision_config=SimpleNamespace(spatial_merge_size=2))

        def __init__(self):
            self.calls = 0

        def __call__(self, **kwargs):
            assert kwargs["output_attentions"] is False and kwargs["output_hidden_states"] is False
            token = sequence[self.calls] if self.calls < len(sequence) else 99
            self.calls += 1
            logits = torch.full((1, 1, 100), -100.0)
            logits[0, 0, token] = 1
            return SimpleNamespace(logits=logits, past_key_values=object())

    model = Model()
    processor = Processor(torch, dict(zip(sequence, pieces)))
    loaded = SimpleNamespace(model=model, processor=processor, device=torch.device("cpu"), eos_ids={99}, runtime_metadata={})
    monkeypatch.setattr(action, "FrozenQwenExtractor", lambda config: loaded)
    monkeypatch.setattr(action, "_sync_cuda", lambda: None)
    monkeypatch.setattr(action, "RawSelectedLayerRecorder", lambda *args: pytest.fail("text baseline must not install feature hooks"))
    runtime = action.ActionQwenRuntime(ExtractionConfig("unused", task="visual_state", layers=(1,)))
    result = runtime.infer_text_action(Image.new("RGB", (100, 80)), max_new_tokens=cap)
    assert result["point"] == expected_point
    assert result["hooks_used"] is False
    assert result["generated_tokens"] == len(pieces)
    assert result["terminated_by_eos"] is eos
    assert result["t_first"] >= 0
    if cap == 3:
        assert result["truncated"] is True and result["valid"] is False
        assert result["unexecuted_candidate_at_token_limit"]["normalized_point"] == [500, 5]
        assert result["t_ready"] is None
    else:
        assert result["valid"] is True and result["t_ready"] >= result["t_first"]
    assert processor.messages[0][0]["content"] == action.TEXT_ACTION_SYSTEM_PROMPT

