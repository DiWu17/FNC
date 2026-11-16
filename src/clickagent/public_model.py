"""Generic-instruction, zero-retraining transfer of the frozen action readouts.

Only this adapter's prompts generalize the original Save-specific experiment.
Original model/actor sources and the old gate/localizer weights are unchanged.
The changed prompt is a distribution shift: this is explicitly recorded as
``generalized_prompt_zero_retrain_transfer``, not the original-task protocol.

``infer_joint`` shares one genuinely generated trajectory among three readouts.
Its stage timings contain shared recorder overhead and must never substitute
for fresh, independently measured policy-method latency comparisons.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from .action_model import ActionQwenRuntime, RawSelectedLayerRecorder, parse_text_action
from .model_features import ExtractionConfig, _sync_cuda, detect_generated_decision


PUBLIC_PROTOCOL = "generalized_prompt_zero_retrain_transfer"
PUBLIC_PROTOCOL_VERSION = 1
PUBLIC_DECISION_SYSTEM_PROMPT = (
    "The user's GUI instruction identifies a target in one screenshot. Your job is to determine whether "
    "that target is visible and supports an immediate click, not to carry out a whole multi-step workflow. "
    "Resolve the target from the instruction and the visible screenshot; do not assume a fixed control name. "
    "First output exactly one ordinary word, CLICK, DISABLED, ABSENT, or UNKNOWN, followed by a newline. "
    "Output CLICK when the intended target is present and enabled for clicking. Output DISABLED when "
    "it is present but visibly disabled or unavailable. Output ABSENT when the intended target is not "
    "present. Output UNKNOWN if the target or its state cannot be determined, including ambiguous targets. "
    "Then briefly explain the visible evidence. Do not invent hidden UI state, actually operate the "
    "computer, or claim that an action was performed."
)
PUBLIC_TEXT_ACTION_SYSTEM_PROMPT = (
    "Identify the target described by the user's GUI instruction in the screenshot and propose at most one click. "
    "Do not assume a fixed target name. Click only when the intended target is visible and enabled. "
    "If it is disabled, absent, ambiguous, or cannot be determined from the screenshot, output WAIT. "
    "Otherwise output CLICK x y, where x and y are integers from 0 to 1000 for a point inside the target, "
    "normalized to the full screenshot width and height. Output only CLICK x y or WAIT, then a newline. "
    "Do not explain, execute an action, or claim that an action was already performed."
)


class PublicQwenRuntime(ActionQwenRuntime):
    """One frozen model load, generic user instructions, and no GT input fields."""

    def __init__(self, config: ExtractionConfig) -> None:
        if config.max_pixels != 262144:
            raise ValueError("public transfer keeps the original max_pixels=262144 budget")
        started = time.perf_counter()
        super().__init__(config)
        self.load_seconds = time.perf_counter() - started

    def _prepare_image(self, image: Any, instruction: str, system_prompt: str) -> tuple[Any, np.ndarray, tuple[int, int]]:
        """Same official image processor/native grid, without a target-name rule."""
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("public instruction must be a nonempty string")
        if image is None or not hasattr(image, "convert"):
            raise ValueError("image must be a PIL screenshot")
        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": instruction}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image.convert("RGB")], return_tensors="pt", padding=False)
        ids = inputs["input_ids"][0].detach().cpu().numpy()
        positions = np.flatnonzero(ids == self.model.config.image_token_id)
        grid = inputs["image_grid_thw"]
        if grid.shape[0] != 1 or int(grid[0, 0]) != 1:
            raise ValueError("public inference requires exactly one still screenshot")
        merge = self.model.config.vision_config.spatial_merge_size
        shape = (int(grid[0, 1]) // merge, int(grid[0, 2]) // merge)
        if not positions.size or positions.size != int(np.prod(shape)):
            raise RuntimeError("image tokens differ from the native post-merger raster grid")
        return inputs.to(self.device), positions, shape

    @staticmethod
    def _protocol_metadata() -> dict[str, Any]:
        return {"protocol": PUBLIC_PROTOCOL, "protocol_version": PUBLIC_PROTOCOL_VERSION,
                "model_weights_retrained": False, "gate_or_grounder_retrained": False,
                "prompt_changed_from_original_task": True, "ground_truth_in_model_inputs": False,
                "max_pixels": 262144}

    def infer_joint(self, image: Any, instruction: str) -> dict[str, Any]:
        """One real prefill, then free greedy decision and its own-token forward.

        ``prefill`` and ``post_decision`` contain the original feature vectors and
        native raw maps. The latter is None on malformed/truncated output;
        UNKNOWN is a valid observed abstention and retains its own readout.
        ``prefill_raw_map`` is the single shared point-grounding input.
        """
        import torch

        with self._lock, torch.inference_mode():
            _sync_cuda()
            started = time.perf_counter()
            inputs, positions, grid_shape = self._prepare_image(image, instruction, PUBLIC_DECISION_SYSTEM_PROMPT)
            prepared_at = time.perf_counter()
            input_length = int(inputs["input_ids"].shape[-1])
            generated: list[int] = []
            text = ""
            decision = None
            decision_index = None
            first_at = word_at = post_at = None
            ended_by_eos = False
            post = None
            with RawSelectedLayerRecorder(self.model, self.config.layers) as recorder:
                recorder.begin(positions, grid_shape)
                outputs = self.model(**inputs, use_cache=True, logits_to_keep=1,
                                     output_attentions=False, output_hidden_states=False, return_dict=True)
                prefill = recorder.snapshot()
                _sync_cuda()
                prefill_at = time.perf_counter()
                prefill.update({"query_token_index": input_length - 1, "generated_tokens_at_readout": 0,
                                "readout_elapsed_seconds": prefill_at - started})
                attention_mask = inputs.get("attention_mask")
                for step in range(self.config.max_new_tokens):
                    token_id = int(outputs.logits[0, -1].argmax().item())
                    if token_id in self.eos_ids:
                        ended_by_eos = True
                        break
                    generated.append(token_id)
                    text = self.processor.tokenizer.decode(generated, skip_special_tokens=True)
                    if first_at is None and text:
                        first_at = time.perf_counter()
                    found = detect_generated_decision(text, task="visual_state")
                    if found is not None:
                        _sync_cuda()
                        word_at = time.perf_counter()
                    # Never substitute a gold word/prefix: every token comes
                    # from the preceding forward's actual argmax logits.
                    outputs, attention_mask = self._forward_token(token_id, outputs, attention_mask, input_length, step)
                    if found is not None:
                        decision, decision_index = found, step
                        post = recorder.snapshot()
                        _sync_cuda()
                        post_at = time.perf_counter()
                        post.update({"query_token_index": input_length + step,
                                     "generated_tokens_at_readout": len(generated),
                                     "readout_elapsed_seconds": post_at - started})
                        break
                del outputs
            _sync_cuda()
            finished = time.perf_counter()
            return {
                **self._protocol_metadata(), "instruction": instruction,
                "prefill": prefill, "post_decision": post,
                "prefill_available": True, "post_decision_available": post is not None,
                "prefill_raw_map": prefill["raw_image_attention"], "head_order": prefill["head_order"],
                "grid_shape": list(grid_shape), "image_size": list(image.size),
                "input_tokens": input_length, "image_tokens": len(positions),
                "generated_decision": decision, "generated_text": text, "generated_token_ids": generated,
                "generated_tokens": len(generated), "decision_completion_token_index": decision_index,
                "decision_abstained": decision == "unknown", "parse_valid": decision is not None,
                "generation_enabled": True, "terminated_by_eos": ended_by_eos,
                "truncated": decision is None and not ended_by_eos and len(generated) == self.config.max_new_tokens,
                "shared_trajectory": True, "independent_method_latency_comparison": False,
                "system_prompt": PUBLIC_DECISION_SYSTEM_PROMPT,
                "t_first": first_at - started if first_at is not None else None,
                "t_ready": post_at - started if post_at is not None else None,
                "elapsed_seconds": finished - started,
                "timing_seconds": {"prepare": prepared_at - started, "prefill_ready": prefill_at - started,
                                   "first_character": first_at - started if first_at is not None else None,
                                   "word_ready": word_at - started if word_at is not None else None,
                                   "post_decision_ready": post_at - started if post_at is not None else None,
                                   "total": finished - started},
                "latency_note": ("One shared generation trajectory, including feature hooks and copies. "
                                 "These stage times do not constitute independent policy-method latency measurements."),
            }

    def infer_text_action_public(self, image: Any, instruction: str, max_new_tokens: int = 32) -> dict[str, Any]:
        """Independent generic coordinate baseline, no feature hooks or GT inputs."""
        import torch

        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer")
        with self._lock, torch.inference_mode():
            _sync_cuda()
            started = time.perf_counter()
            inputs, _, grid_shape = self._prepare_image(image, instruction, PUBLIC_TEXT_ACTION_SYSTEM_PROMPT)
            outputs = self.model(**inputs, use_cache=True, logits_to_keep=1,
                                 output_attentions=False, output_hidden_states=False, return_dict=True)
            generated: list[int] = []
            text = ""
            first_at = ready_at = None
            parsed = None
            ended_by_eos = False
            attention_mask = inputs.get("attention_mask")
            input_length = int(inputs["input_ids"].shape[-1])
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
            candidate = parse_text_action(text, final=True) if truncated else None
            action = parsed if parsed is not None else {"action": "wait", "normalized_point": None,
                                                        "generated_decision": None, "abstained": True, "valid": False}
            point = None
            if action["action"] == "click":
                x, y = action["normalized_point"]
                point = [int(round(x * (image.width - 1) / 1000)), int(round(y * (image.height - 1) / 1000))]
            return {
                **self._protocol_metadata(), **action, "instruction": instruction, "point": point,
                "image_size": list(image.size), "grid_shape": list(grid_shape),
                "generated_text": text, "generated_token_ids": generated, "generated_tokens": len(generated),
                "terminated_by_eos": ended_by_eos, "truncated": truncated, "parse_valid": parsed is not None,
                "unexecuted_candidate_at_token_limit": candidate, "hooks_used": False,
                "system_prompt": PUBLIC_TEXT_ACTION_SYSTEM_PROMPT, "shared_trajectory": False,
                "t_first": first_at - started if first_at is not None else None,
                "t_ready": ready_at - started if ready_at is not None else None,
                "elapsed_seconds": finished - started,
                "parse_failure": None if parsed is not None else "malformed, out-of-range, incomplete, or token-limited action",
            }
