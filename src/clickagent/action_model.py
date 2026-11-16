"""Live frozen-Qwen action readouts and an independent text-coordinate baseline.

Gate readouts import the original visual_state system prompt verbatim, preserving
the protocol of cached gate.joblib models. The coordinate baseline deliberately
uses a different action-format prompt and no feature hooks; it is not the same
supervision budget as a trained hidden-state gate plus trained point localizer.
All times start after GPU synchronization and include preprocessing/model/readout
work but no screenshots, file dumps, OS actions, or post-action verification.
"""

from __future__ import annotations

import re
import threading
import time
from typing import Any

import numpy as np

from .attention_features import summarize_image_attention
from .model_features import (
    ExtractionConfig, FrozenQwenExtractor, SelectedLayerRecorder, _sync_cuda,
    detect_generated_decision, reconstruct_attention_row, system_prompt_for_task,
    VISUAL_STATE_INSTRUCTION,
)


TEXT_ACTION_SYSTEM_PROMPT = (
    "You control a computer by choosing one action from the screenshot and the user's instruction. "
    "The requested target has the exact visible label 保存. 另存为, Save as, and other synonyms are not this target. "
    "Click only if that target is present and enabled. A grayed-out, low-contrast disabled control must not be clicked. "
    "If it is disabled, absent, or you cannot determine a valid action, output WAIT. "
    "Otherwise output CLICK x y, where x and y are integers from 0 to 1000 indicating a point inside that target, "
    "normalized to the full screenshot width and height. Output only CLICK x y or WAIT, then a newline; "
    "do not explain or claim that the action has already been performed."
)


def parse_text_action(text: str, *, final: bool = True) -> dict[str, Any] | None:
    """Strict whole-action parser; streaming reads require newline termination.

    ``final=True`` is appropriate only after EOS or an explicitly recorded token
    cap. Runtime accepts delimiter-free CLICK only after EOS, conservatively
    abstaining at a token cap because a final digit may still be incomplete.
    """
    if not final and not text.endswith(("\n", "\r")):
        return None
    match = re.fullmatch(r"\s*(CLICK[ \t]+([0-9]+)[ \t]+([0-9]+)|WAIT|UNKNOWN)\s*", text)
    if match is None:
        return None
    word = match.group(1)
    if word in {"WAIT", "UNKNOWN"}:
        return {"action": "wait", "normalized_point": None, "generated_decision": word.lower(),
                "abstained": word == "UNKNOWN", "valid": True}
    x, y = int(match.group(2)), int(match.group(3))
    if not (0 <= x <= 1000 and 0 <= y <= 1000):
        return None
    return {"action": "click", "normalized_point": [x, y], "generated_decision": "click",
            "abstained": False, "valid": True}


class RawSelectedLayerRecorder(SelectedLayerRecorder):
    """One QK reconstruction per selected layer, feeding statistics and raw maps."""

    def __init__(self, model: Any, layers: tuple[int, ...]) -> None:
        super().__init__(model, layers)
        self.raw_attention: dict[int, np.ndarray] = {}

    def begin(self, image_positions: np.ndarray, grid_shape: tuple[int, int]) -> None:
        super().begin(image_positions, grid_shape)
        self.raw_attention.clear()

    def _attention_hook(self, layer: int) -> Any:
        def hook(module: Any, args: Any, kwargs: dict[str, Any], output: Any) -> None:
            import torch

            hidden = kwargs.get("hidden_states", args[0] if args else None)
            position = kwargs.get("position_embeddings", args[1] if len(args) > 1 else None)
            cache = kwargs.get("past_key_values", kwargs.get("past_key_value"))
            if cache is None or not hasattr(cache, "layers"):
                raise RuntimeError("raw action readouts require a DynamicCache")
            cache_layer = cache.layers[module.layer_idx]
            if getattr(cache_layer, "is_sliding", False) or getattr(cache_layer, "is_compileable", False):
                raise RuntimeError("raw action readouts require a full non-static cache")
            if position is None or cache_layer.keys is None:
                raise RuntimeError("missing Qwen3-VL rotary positions or cached keys")
            row = reconstruct_attention_row(module, hidden, position, cache_layer.keys, kwargs.get("attention_mask"))
            if row.shape[0] != 1:
                raise RuntimeError("action inference supports exactly one screenshot")
            indices = torch.as_tensor(self.image_positions, device=row.device, dtype=torch.long)
            if indices.numel() and int(indices.max()) >= row.shape[-1]:
                raise RuntimeError("image positions exceed the KV cache")
            raw = row[0].index_select(-1, indices).detach().cpu().numpy().copy()
            self.raw_attention[layer] = raw
            self.attention[layer], self.names[layer] = summarize_image_attention(raw, self.grid_shape, layer=layer)
        return hook

    def snapshot(self) -> dict[str, Any]:
        result = super().snapshot()
        result["raw_image_attention"] = np.concatenate([self.raw_attention[layer] for layer in self.layers], axis=0).copy()
        result["head_order"] = [f"layer_{layer}.head_{head}" for layer in self.layers
                                for head in range(self.raw_attention[layer].shape[0])]
        return result


class ActionQwenRuntime:
    """Load exactly one frozen model for repeated independent screenshot inferences."""

    def __init__(self, config: ExtractionConfig) -> None:
        if config.task != "visual_state" or config.no_image:
            raise ValueError("ActionQwenRuntime requires task='visual_state' and no_image=False")
        self.config = config
        self.extractor = FrozenQwenExtractor(config)
        self.model = self.extractor.model
        self.processor = self.extractor.processor
        self.device = self.extractor.device
        self.eos_ids = self.extractor.eos_ids
        self.runtime_metadata = getattr(self.extractor, "runtime_metadata", None)
        self._lock = threading.Lock()

    def _prepare_image(self, image: Any, instruction: str, system_prompt: str) -> tuple[Any, np.ndarray, tuple[int, int]]:
        if instruction != VISUAL_STATE_INSTRUCTION:
            raise ValueError("this gate protocol requires the exact instruction '点击保存'")
        if image is None or not hasattr(image, "convert"):
            raise ValueError("image must be a PIL screenshot")
        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": instruction}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image.convert("RGB")], return_tensors="pt", padding=False)
        positions = np.flatnonzero(inputs["input_ids"][0].detach().cpu().numpy() == self.model.config.image_token_id)
        grid = inputs["image_grid_thw"]
        if grid.shape[0] != 1 or int(grid[0, 0]) != 1:
            raise ValueError("exactly one still screenshot is required")
        merge = self.model.config.vision_config.spatial_merge_size
        shape = (int(grid[0, 1]) // merge, int(grid[0, 2]) // merge)
        if not positions.size or int(np.prod(shape)) != positions.size:
            raise RuntimeError("image token count does not match the native post-merger raster grid")
        return inputs.to(self.device), positions, shape

    def _forward_token(self, token_id: int, outputs: Any, attention_mask: Any,
                       input_length: int, step: int) -> tuple[Any, Any]:
        import torch

        if attention_mask is not None:
            attention_mask = torch.cat((attention_mask, attention_mask.new_ones((1, 1))), dim=-1)
        token = torch.tensor([[token_id]], device=self.device, dtype=torch.long)
        cache_position = torch.tensor([input_length + step], device=self.device, dtype=torch.long)
        result = self.model(input_ids=token, attention_mask=attention_mask,
                            past_key_values=outputs.past_key_values, cache_position=cache_position,
                            use_cache=True, logits_to_keep=1, output_attentions=False,
                            output_hidden_states=False, return_dict=True)
        return result, attention_mask

    def infer_readouts(self, image: Any, instruction: str = VISUAL_STATE_INSTRUCTION,
                       stage: str = "prefill") -> dict[str, Any]:
        """Read the requested gate stage and always retain the same prefill maps.

        ``decision_text`` removes feature hooks after the shared prefill map,
        and stops as soon as a category word is generated, before forwarding
        that completed word token. It has no stage hidden/attention readout.
        Failed decision formatting yields readout_available=False rather than
        silently labeling a different token as post_decision. UNKNOWN is a valid
        observed decision/abstention and does have a post_decision readout.
        """
        import torch

        if stage not in {"prefill", "post_decision", "decision_text"}:
            raise ValueError("stage must be prefill, post_decision, or decision_text")
        with self._lock, torch.inference_mode():
            _sync_cuda()
            started = time.perf_counter()
            inputs, image_positions, grid_shape = self._prepare_image(image, instruction, system_prompt_for_task("visual_state"))
            prepared_at = time.perf_counter()
            generated: list[int] = []
            text = ""
            decision = None
            first_at = decision_generated_at = ready_at = None
            terminated_by_eos = False
            with RawSelectedLayerRecorder(self.model, self.config.layers) as recorder:
                recorder.begin(image_positions, grid_shape)
                outputs = self.model(**inputs, use_cache=True, logits_to_keep=1, output_attentions=False,
                                     output_hidden_states=False, return_dict=True)
                prefill = recorder.snapshot()
                _sync_cuda()
                prefill_at = time.perf_counter()
                selected = prefill if stage == "prefill" else None
                if stage == "prefill":
                    ready_at = prefill_at
                else:
                    if stage == "decision_text":
                        # Idempotent removal also runs on the outer context exit.
                        # Decoding partial words needs no further feature hooks.
                        recorder.__exit__(None, None, None)
                    attention_mask = inputs.get("attention_mask")
                    input_length = inputs["input_ids"].shape[-1]
                    for step in range(self.config.max_new_tokens):
                        token_id = int(outputs.logits[0, -1].argmax().item())
                        if token_id in self.eos_ids:
                            terminated_by_eos = True
                            break
                        generated.append(token_id)
                        text = self.processor.tokenizer.decode(generated, skip_special_tokens=True)
                        now = time.perf_counter()
                        if first_at is None and text:
                            first_at = now
                        found = detect_generated_decision(text, task="visual_state")
                        if found is not None:
                            decision_generated_at = now
                            if stage == "decision_text":
                                decision = found
                                _sync_cuda()
                                ready_at = time.perf_counter()
                                break
                        # The generated token is forwarded before reading its
                        # own query row and hidden state, as in the original gate.
                        outputs, attention_mask = self._forward_token(token_id, outputs, attention_mask,
                                                                       input_length, step)
                        if found is not None:
                            decision = found
                            selected = recorder.snapshot()
                            _sync_cuda()
                            ready_at = time.perf_counter()
                            break
                del outputs
            _sync_cuda()
            finished = time.perf_counter()
            return {
                "stage": stage, "readout_available": selected is not None or (stage == "decision_text" and decision is not None),
                "attention": selected["attention"] if selected else None,
                "hidden": selected["hidden"] if selected else None,
                "raw_image_attention": selected["raw_image_attention"] if selected else None,
                "prefill_raw_map": prefill["raw_image_attention"], "head_order": prefill["head_order"],
                "grid_shape": list(grid_shape), "image_size": list(image.size),
                "generated_decision": decision, "generated_text": text, "generated_token_ids": generated,
                "generated_tokens": len(generated), "decision_abstained": decision == "unknown",
                "generation_enabled": stage != "prefill", "terminated_by_eos": terminated_by_eos,
                "truncated": stage != "prefill" and decision is None and not terminated_by_eos,
                "t_first": first_at - started if first_at else None,
                "t_ready": ready_at - started if ready_at else None,
                "readout_elapsed_seconds": ready_at - started if ready_at else None,
                "elapsed_seconds": finished - started, "prefill_elapsed_seconds": prefill_at - started,
                "timing_seconds": {"prepare": prepared_at - started, "prefill_ready": prefill_at - started,
                                   "first_character": first_at - started if first_at else None,
                                   "decision_word_generated": decision_generated_at - started if decision_generated_at else None,
                                   "readout_ready": ready_at - started if ready_at else None, "total": finished - started},
                "system_prompt": system_prompt_for_task("visual_state"),
            }

    def infer_text_action(self, image: Any, instruction: str = VISUAL_STATE_INSTRUCTION,
                          max_new_tokens: int = 32) -> dict[str, Any]:
        """Greedy coordinate-text baseline with no attention/hidden hooks.

        A streaming integer is accepted only after newline or EOS. At a token
        limit a delimiter-free candidate is exposed for audit but not executed.
        """
        import torch

        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer")
        with self._lock, torch.inference_mode():
            _sync_cuda()
            started = time.perf_counter()
            inputs, _, grid_shape = self._prepare_image(image, instruction, TEXT_ACTION_SYSTEM_PROMPT)
            outputs = self.model(**inputs, use_cache=True, logits_to_keep=1, output_attentions=False,
                                 output_hidden_states=False, return_dict=True)
            generated: list[int] = []
            text = ""
            first_at = ready_at = None
            parsed = None
            ended_by_eos = False
            attention_mask = inputs.get("attention_mask")
            input_length = inputs["input_ids"].shape[-1]
            for step in range(max_new_tokens):
                token_id = int(outputs.logits[0, -1].argmax().item())
                if token_id in self.eos_ids:
                    ended_by_eos = True
                    parsed = parse_text_action(text, final=True)
                    if parsed is not None:
                        _sync_cuda()
                        ready_at = time.perf_counter()
                    break
                generated.append(token_id)
                text = self.processor.tokenizer.decode(generated, skip_special_tokens=True)
                if first_at is None and text:
                    first_at = time.perf_counter()
                parsed = parse_text_action(text, final=False)
                if parsed is not None:
                    _sync_cuda()
                    ready_at = time.perf_counter()
                    break
                if step + 1 < max_new_tokens:
                    outputs, attention_mask = self._forward_token(token_id, outputs, attention_mask, input_length, step)
            del outputs
            _sync_cuda()
            finished = time.perf_counter()
            truncated = parsed is None and not ended_by_eos and len(generated) == max_new_tokens
            candidate_at_limit = parse_text_action(text, final=True) if truncated else None
            action = parsed if parsed is not None else {"action": "wait", "normalized_point": None,
                                                        "generated_decision": None, "abstained": True, "valid": False}
            point = None
            if action["action"] == "click":
                x, y = action["normalized_point"]
                point = [int(round(x * (image.width - 1) / 1000)), int(round(y * (image.height - 1) / 1000))]
            return {
                **action, "point": point, "image_size": list(image.size), "grid_shape": list(grid_shape),
                "generated_text": text, "generated_token_ids": generated, "generated_tokens": len(generated),
                "terminated_by_eos": ended_by_eos, "truncated": truncated,
                "unexecuted_candidate_at_token_limit": candidate_at_limit, "hooks_used": False,
                "t_first": first_at - started if first_at else None,
                "t_ready": ready_at - started if ready_at else None,
                "elapsed_seconds": finished - started,
                "parse_failure": None if parsed is not None else "malformed, out-of-range, incomplete, or token-limited action",
                "system_prompt": TEXT_ACTION_SYSTEM_PROMPT,
            }
