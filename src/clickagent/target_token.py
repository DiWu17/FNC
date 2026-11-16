"""Own-token attention along a freely generated GUI target-name phrase.

This module changes readout timing, not the existing spatial decoder. No target
annotation, oracle name, or teacher-forced answer enters the model call.
"""
from __future__ import annotations

import re
import time

import numpy as np

from .action_model import RawSelectedLayerRecorder
from .model_features import _sync_cuda
from .public_model import PublicQwenRuntime


TARGET_NAME_PROMPT = (
    "Inspect the screenshot and resolve the specific GUI target referred to by the user's instruction. "
    "Reply in exactly two lines:\n"
    "ACTION: CLICK or WAIT\n"
    "TARGET: the target's visible label or a short concrete name\n"
    "On the ACTION line choose CLICK when the requested target is identifiable, visible and enabled; "
    "otherwise choose WAIT. On the TARGET line give only the label/name of that specific target, "
    "in at most twelve words, preferably using its visible text. For an unlabeled icon describe the icon. "
    "If you cannot identify the target, write NONE. Do not repeat the full instruction. "
    "Do not output coordinates, extra lines, explanations, or a claim that you performed the action."
)


def _content_indices(prefixes, start, end):
    """Map a character span to actual generated-token own-forward indices.

    Incremental decoding may revise a trailing replacement character for a
    partial UTF-8 token. Spans use prefix rewrite boundaries, not a retokenized
    answer (which could assign different token IDs).
    """
    indices = []
    previous = ""
    for i, current in enumerate(prefixes):
        common = 0
        for a, b in zip(previous, current):
            if a != b:
                break
            common += 1
        if common < end and len(current) > start:
            indices.append(i)
        previous = current
    return indices


def select_stages(text, prefixes, *, ended_by_eos, token_cap_reached):
    """Choose readouts from generated syntax only; no labels or heatmap scores."""
    stages = {"prefill": [0]}
    action = re.search(r"\AACTION:[ \t]*(CLICK|WAIT)(?=\s|$)", text)
    if action:
        ids = _content_indices(prefixes, *action.span(1))
        if ids:
            stages["action_word"] = [ids[-1] + 1]
    # A target must be completed by newline or genuine EOS, never just a cap.
    match = re.search(r"(?:\A|\n)TARGET:[ \t]*([^\r\n]+)(?:\r?\n|$)", text)
    complete = bool(match and (match.end() < len(text) or text.endswith(("\n", "\r")) or ended_by_eos))
    if token_cap_reached and not text.endswith(("\n", "\r")):
        complete = False
    target = match.group(1).strip() if match else None
    if complete and target and target.upper() not in {"NONE", "UNKNOWN", "N/A"}:
        start, end = match.span(1)
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        ids = _content_indices(prefixes, start, end)
        if ids:
            stages["target_first"] = [ids[0] + 1]
            stages["target_last"] = [ids[-1] + 1]
            stages["target_mean"] = [i + 1 for i in ids]
        marker_start = text.rfind("TARGET:", 0, start)
        marker_ids = _content_indices(prefixes, marker_start, marker_start + len("TARGET:"))
        # A token crossing into the name is not a clean punctuation control.
        if marker_ids and len(prefixes[marker_ids[-1]]) <= start:
            stages["target_marker"] = [marker_ids[-1] + 1]
    if prefixes:
        stages["all_generated_mean"] = list(range(1, len(prefixes) + 1))
    return stages, {"action": action.group(1) if action else None, "target_text": target,
                    "target_complete": complete, "target_available": "target_mean" in stages}


class TargetTokenRuntime(PublicQwenRuntime):
    def infer_target_name(self, image, instruction, max_new_tokens=64):
        """Generate a name, then read each generated token's own query row."""
        import torch

        with self._lock, torch.inference_mode():
            _sync_cuda()
            started = time.perf_counter()
            inputs, positions, grid = self._prepare_image(image, instruction, TARGET_NAME_PROMPT)
            input_length = int(inputs["input_ids"].shape[-1])
            generated, prefixes, maps, hidden, elapsed = [], [], [], [], []
            ended_by_eos = False
            with RawSelectedLayerRecorder(self.model, self.config.layers) as recorder:
                recorder.begin(positions, grid)
                outputs = self.model(**inputs, use_cache=True, logits_to_keep=1,
                    output_attentions=False, output_hidden_states=False, return_dict=True)
                snapshot = recorder.snapshot()
                maps.append(snapshot["raw_image_attention"])
                hidden.append(snapshot["hidden"])
                elapsed.append(time.perf_counter() - started)
                attention_mask = inputs.get("attention_mask")
                text = ""
                for step in range(max_new_tokens):
                    token_id = int(outputs.logits[0, -1].argmax().item())
                    if token_id in self.eos_ids:
                        ended_by_eos = True
                        break
                    generated.append(token_id)
                    text = self.processor.tokenizer.decode(generated, skip_special_tokens=True)
                    prefixes.append(text)
                    # Crucial alignment: forward the sampled token before its
                    # attention is stored. The same call predicts the next one.
                    outputs, attention_mask = self._forward_token(token_id, outputs, attention_mask, input_length, step)
                    snapshot = recorder.snapshot()
                    maps.append(snapshot["raw_image_attention"])
                    hidden.append(snapshot["hidden"])
                    elapsed.append(time.perf_counter() - started)
                    if re.search(r"(?:\A|\n)TARGET:[^\r\n]*\r?\n", text):
                        break
                del outputs
            cap_reached = len(generated) == max_new_tokens and not ended_by_eos
            stages, parsed = select_stages(text, prefixes, ended_by_eos=ended_by_eos,
                                           token_cap_reached=cap_reached)
            assert len(maps) == len(prefixes) + 1
            _sync_cuda()
            metadata = {"generated_text": text, "generated_token_ids": generated,
                "generated_prefixes": prefixes, "stages": stages, **parsed,
                "terminated_by_eos": ended_by_eos, "token_cap_reached": cap_reached,
                "grid_shape": list(grid), "image_size": list(image.size),
                "input_tokens": input_length, "query_token_indices": list(range(input_length - 1, input_length + len(generated))),
                "head_order": snapshot["head_order"], "own_token_forward": True,
                "ground_truth_in_model_input": False, "system_prompt": TARGET_NAME_PROMPT,
                "elapsed_seconds": time.perf_counter() - started,
                "latency_note": "Shared instrumented trajectory; not an independent method-latency comparison."}
            arrays = {"raw_attention": np.asarray(maps, np.float32),
                      "query_hidden": np.asarray(hidden, np.float16),
                      "readout_elapsed_seconds": np.asarray(elapsed, np.float64)}
            return arrays, metadata


def spatial_scores(raw, readout, grounder, grid, size):
    """Keep all spatial decoders fixed across temporal readout stages."""
    if readout == "legacy_96":
        return grounder.predict_point(raw, grid, size)["point"]
    if readout == "layer15_mean":
        score = raw[:32].mean(0)
    elif readout == "all_layers_mean":
        score = raw.mean(0)
    else:
        raise ValueError(readout)
    patch = int(score.argmax())
    gh, gw = grid
    w, h = size
    row, col = divmod(patch, gw)
    return [min(w - 1, round((col + .5) * w / gw)), min(h - 1, round((row + .5) * h / gh))]
