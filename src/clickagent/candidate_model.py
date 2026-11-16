"""Frozen VLM candidate proposals, candidate-visible readouts and native logits.

The runtime accepts screenshot, instruction and proposed points only. No labels,
target boxes, intended answer, application identity or split enter model inputs.
"""
from __future__ import annotations

import time

import numpy as np
from PIL import ImageDraw, ImageFont

from .action_grounding import patch_features, _pixel_centers
from .action_model import RawSelectedLayerRecorder
from .model_features import _sync_cuda
from .public_diagnostics import parse_offline_lax_click
from .public_model import PublicQwenRuntime, PUBLIC_TEXT_ACTION_SYSTEM_PROMPT


CHOICES = ("A", "B", "C", "D", "N")
VERIFY_PROMPT = (
    "Evaluate proposed GUI clicks against the user's instruction and the screenshot. "
    "Four red hollow rings mark candidate click positions A, B, C and D; labels are placed beside the rings. "
    "The ring center, not its label, is the proposed click position. Select a candidate only if its center "
    "is inside the requested target and the user actually requests performing the operation now. "
    "For a description-only request, a prohibition, absent target, visibly disabled target, ambiguous target, "
    "or if none of the proposed centers is correct, select N. Do not reinterpret a description request as permission to click. "
    "Output exactly one letter A, B, C, D or N. Do not output coordinates or explanations."
)


def mark_candidates(image, points):
    image = image.convert("RGB").copy()
    draw = ImageDraw.Draw(image)
    radius = max(4, round(max(image.size) / 180))
    font_size = max(14, round(max(image.size) / 90))
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()
    for label, (x, y) in zip(CHOICES, points):
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline="#ff3030", width=max(1, radius // 4))
        tx, ty = min(x + radius + 2, image.width - font_size), max(0, min(y - radius - font_size, image.height - font_size))
        draw.text((tx, ty), label, font=font, fill="#ff3030", stroke_width=1, stroke_fill="white")
    return image


def pool_candidate_features(raw, image_hidden, grid_shape, points, image_size):
    """Pool centers and 3x3 neighborhoods, retaining absolute image mass."""
    gh, gw = map(int, grid_shape)
    w, h = image_size
    mass = raw.sum(-1)
    relative = np.divide(raw, mass[:, None], out=np.zeros_like(raw), where=mass[:, None] > 0)
    attention, hidden, geometry = [], [], []
    for x, y in points:
        col, row = min(gw - 1, int(x * gw / w)), min(gh - 1, int(y * gh / h))
        patch = row * gw + col
        yy, xx = np.mgrid[max(0, row - 1):min(gh, row + 2), max(0, col - 1):min(gw, col + 2)]
        neighbors = (yy * gw + xx).ravel()
        attention.append(np.concatenate((np.log1p(relative[:, patch] * gh * gw),
            np.log1p(relative[:, neighbors].mean(-1) * gh * gw),
            (relative <= relative[:, patch, None]).mean(-1), mass)))
        # image_hidden is [patches, layers * hidden_dim]. The center receives
        # half the weight, its spatial neighborhood receives the other half.
        hidden.append(.5 * image_hidden[patch].astype(np.float32) + .5 * image_hidden[neighbors].astype(np.float32).mean(0))
        geometry.append([x / w, y / h, (x * gw / w) % 1, (y * gh / h) % 1])
    return np.asarray(attention, np.float32), np.asarray(hidden, np.float32), np.asarray(geometry, np.float32)


class CandidateRecorder(RawSelectedLayerRecorder):
    def __init__(self, model, layers):
        super().__init__(model, layers)
        self.visual = {}

    def _hidden_hook(self, layer):
        original = super()._hidden_hook(layer)
        def hook(module, args, output):
            import torch
            original(module, args, output)
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            indices = torch.as_tensor(self.image_positions, device=hidden.device, dtype=torch.long)
            self.visual[layer] = hidden[0].index_select(0, indices).detach().half().cpu().numpy().copy()
        return hook


class FirstPrefillRecorder(RawSelectedLayerRecorder):
    """Keep proposal-prefill maps even while native coordinates are decoded."""
    def _attention_hook(self, layer):
        base = super()._attention_hook(layer)
        def hook(module, args, kwargs, output):
            if layer not in self.raw_attention:
                base(module, args, kwargs, output)
        return hook

    def _hidden_hook(self, layer):
        base = super()._hidden_hook(layer)
        def hook(module, args, output):
            if layer not in self.hidden:
                base(module, args, output)
        return hook


def diverse_points(proposed, raw, grid_shape, image_size, grounder):
    """Fixed, label-free native/old-grounder/attention-consensus proposal pool."""
    gh, gw = map(int, grid_shape)
    w, h = image_size
    points, sources = [], []
    def add(point, source):
        if point is None:
            return
        point = [int(np.clip(point[0], 0, w - 1)), int(np.clip(point[1], 0, h - 1))]
        if all(np.linalg.norm((np.asarray(point) - p) / [w, h]) > .012 for p in points):
            points.append(point)
            sources.append(source)
    add(proposed, "native_coordinates")
    add(grounder.predict_point(raw, grid_shape, image_size)["point"], "legacy_attention_grounder")
    features = patch_features(raw, grid_shape)
    # The mean log relative attention is independent of every target annotation.
    ranking = np.argsort(-features.mean(-1), kind="stable")
    xs, ys = _pixel_centers(grid_shape, image_size)
    for index in ranking:
        add([xs[index], ys[index]], "attention_consensus")
        if len(points) == 4:
            break
    if len(points) != 4:
        raise RuntimeError("could not obtain four distinct model proposals")
    return points, sources


class CandidateRuntime(PublicQwenRuntime):
    def propose(self, image, instruction, grounder):
        # Public inference returns no labels; the base instruction is used for
        # describe/forbid cases only to obtain plausible targets for verification.
        _, positions, grid = self._prepare_image(image, instruction, PUBLIC_TEXT_ACTION_SYSTEM_PROMPT)
        with FirstPrefillRecorder(self.model, self.config.layers) as recorder:
            recorder.begin(positions, grid)
            result = self.infer_text_action_public(image, instruction, max_new_tokens=32)
            raw = recorder.snapshot()["raw_image_attention"]
        completed = result["terminated_by_eos"] or result["generated_text"].endswith(("\n", "\r"))
        parsed = parse_offline_lax_click(result["generated_text"]) if completed else None
        point = result["point"]
        if point is None and parsed:
            x, y = parsed
            point = [int(round(x * (image.width - 1) / 1000)), int(round(y * (image.height - 1) / 1000))]
        points, sources = diverse_points(point, raw, grid, image.size, grounder)
        return {"points": points, "sources": sources, "native_text": result["generated_text"],
                "native_proposal_point": point, "native_proposal_seconds": result["elapsed_seconds"],
                "grid_shape": list(grid)}

    def verify_features(self, image, instruction, points):
        import torch
        if len(points) != 4 or len({tuple(p) for p in points}) != 4:
            raise ValueError("exactly four distinct candidates required")
        w, h = image.size
        if any(not (0 <= p[0] < w and 0 <= p[1] < h) for p in points):
            raise ValueError("candidate outside screenshot")
        coordinates = "; ".join(f"{label}=({round(x / w * 1000)}, {round(y / h * 1000)})"
                                 for label, (x, y) in zip(CHOICES, points))
        user = f"User instruction: {instruction}\nCandidate ring centers, normalized 0..1000: {coordinates}"
        with self._lock, torch.inference_mode():
            _sync_cuda()
            started = time.perf_counter()
            marked = mark_candidates(image, points)
            inputs, positions, grid = self._prepare_image(marked, user, VERIFY_PROMPT)
            token_ids = [self.processor.tokenizer.encode(c, add_special_tokens=False) for c in CHOICES]
            if any(len(ids) != 1 for ids in token_ids):
                raise RuntimeError("choice labels must each be exactly one native tokenizer token")
            with CandidateRecorder(self.model, self.config.layers) as recorder:
                recorder.begin(positions, grid)
                outputs = self.model(**inputs, use_cache=True, logits_to_keep=1,
                    output_attentions=False, output_hidden_states=False, return_dict=True)
                snapshot = recorder.snapshot()
                image_hidden = np.concatenate([recorder.visual[layer] for layer in self.config.layers], axis=-1)
                logits = outputs.logits[0, -1].float()
                choice_logits = logits[[ids[0] for ids in token_ids]]
                native = torch.softmax(choice_logits, 0).cpu().numpy()
                top_token = int(logits.argmax())
                del outputs
            att, visual, geom = pool_candidate_features(snapshot["raw_image_attention"], image_hidden,
                                                       grid, points, image.size)
            _sync_cuda()
            metadata = {"grid_shape": list(grid), "image_size": list(image.size), "points": points,
                "native_choice": CHOICES[int(np.argmax(native))], "native_top_vocab_token": top_token,
                "native_top_vocab_text": self.processor.tokenizer.decode([top_token]),
                "native_probabilities_are_choice_conditioned": True,
                "elapsed_seconds": time.perf_counter() - started, "generated_tokens": 0,
                "readout_stage": "assistant_prefill", "visual_pooling": "half center + half 3x3 mean",
                "image_is_candidate_marked": True, "ground_truth_in_model_input": False}
            arrays = {"query_hidden": snapshot["hidden"].astype(np.float16),
                "global_attention": snapshot["attention"].astype(np.float32),
                "candidate_attention": att, "candidate_visual": visual.astype(np.float16),
                "geometry": geom, "native_probabilities": native}
            return arrays, metadata
