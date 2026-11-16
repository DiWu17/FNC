"""Fresh point-aligned proposers; no learned legacy checkpoint is consulted.

Only the architecture, label-free feature builders, and training losses are
reused.  ``predict_raw`` deliberately has no case or geometry argument.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .attention_field_ranker import build_attention_field, make_field_batch
from .candidate_ranker import build_candidate_inputs, candidate_supervision, decode_ranker
from .point_aligned_readout import PointAlignedRanker, all_candidate_points
from .semantic_head_ranker import SemanticHeadRankerConfig


SCHEMA_VERSION = 1
RAW_ARRAY_KEYS = ("target_attention", "target_hidden", "image_hidden", "vision_hidden")


@dataclass(frozen=True)
class CrossfitProposerConfig:
    seed: int = 20261201
    width: int = 32
    head_width: int = 16
    groups: int = 4
    epochs: int = 24
    steps_per_epoch: int = 75
    batch_size: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = .01
    region_weight: float = .25
    grad_clip: float = 1.

    def __post_init__(self):
        if (self.width, self.head_width, self.groups) != (32, 16, 4):
            raise ValueError("The prespecified proposer has width32/head_width16/groups4")
        for name in ("epochs", "steps_per_epoch", "batch_size"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.region_weight < 0 or self.grad_clip <= 0:
            raise ValueError("Invalid optimizer/loss configuration")


def state_sha256(state_or_model) -> str:
    """Hash names, shapes, dtypes, and exact CPU state bytes, including buffers."""
    state = state_or_model.state_dict() if hasattr(state_or_model, "state_dict") else state_or_model
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode() + b"\0")
        digest.update(str(tensor.dtype).encode() + b"\0")
        digest.update(str(tuple(tensor.shape)).encode() + b"\0")
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def make_fresh_proposer(config: CrossfitProposerConfig | None = None) -> PointAlignedRanker:
    config = config or CrossfitProposerConfig()
    # A caller's role, device, previous models, and RNG state cannot affect init.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.seed)
        model = PointAlignedRanker(
            SemanticHeadRankerConfig(config.width, config.head_width, config.groups, "semantic"),
            np.zeros(96, dtype=np.float32), connect_fine=True,
        )
    model.crossfit_config = config
    model.initial_state_sha256 = state_sha256(model)
    return model


def build_raw_record(
    arrays: Mapping[str, np.ndarray], metadata: Mapping[str, Any], image_size,
    *, geometry=None,
) -> dict[str, Any]:
    """Build inputs from frozen arrays; optional GT creates loss targets only.

    The record's ``case`` deliberately contains only image size.  Training code
    may attach its allowed training case after this operation for loss-only hit
    computation.  A no-name case stays present, with no proposed points.
    """
    size = tuple(int(x) for x in image_size)
    if len(size) != 2 or min(size) <= 0:
        raise ValueError("image_size must be positive (width,height)")
    meta = dict(metadata)
    if meta.get("ground_truth_in_model_input", False):
        raise ValueError("Frozen feature archive declares GT in model input")
    if not meta.get("own_token_forward", False):
        raise ValueError("Own-name-token features are required")
    if "all_backbone_parameters_frozen" in meta and not meta["all_backbone_parameters_frozen"]:
        raise ValueError("Qwen backbone must be frozen")
    record = {"metadata": meta, "case": {"image_size": list(size)}}
    if not meta.get("target_available", False):
        return record
    missing = set(RAW_ARRAY_KEYS) - set(arrays)
    if missing:
        raise ValueError(f"Missing raw frozen arrays: {sorted(missing)}")
    attention = np.asarray(arrays["target_attention"])
    if not np.isfinite(attention).all() or (attention < 0).any():
        raise ValueError("Attention must be finite and nonnegative")
    record.update(build_candidate_inputs(
        attention, arrays["target_hidden"], arrays["image_hidden"], arrays["vision_hidden"], meta["grid_shape"],
    ))
    record["attention_field"] = build_attention_field(attention, record["parent_indices"])
    if geometry is not None:
        record.update(candidate_supervision(geometry, meta["grid_shape"], record["parent_indices"]))
    return record


@torch.inference_mode()
def predict_raw(model, arrays, meta, image_size, topk=64) -> dict[str, Any]:
    """Canonical CPU, label-free, own-decoder proposal interface.

    This never loads a v5/v6/v7/v8 model, teacher point, head prior, or label.
    It requires the caller to supply an eval model already on CPU, preventing a
    hidden device/mode mutation when the same model is being trained elsewhere.
    """
    if topk < 1:
        raise ValueError("topk must be positive")
    if model.training:
        raise ValueError("predict_raw requires model.eval()")
    if next(model.parameters()).device.type != "cpu":
        raise ValueError("Canonical predict_raw requires a CPU model")
    record = build_raw_record(arrays, meta, image_size)
    name = str(meta.get("target_text", ""))
    if not meta.get("target_available", False):
        return dict(point=None, candidates=[], all_points=[], all_scores=[], parent_indices=[],
                    target_name=name, grid_shape=list(meta.get("grid_shape", [])), decoder="own",
                    failure="no_name", ground_truth_in_readout_inputs=False)
    inputs, _ = make_field_batch([record], "cpu", labels=False)
    output = model(**inputs)
    points = all_candidate_points(output, [record])[0]
    candidates = decode_ranker(output, record, top_k=topk)
    for item in candidates:
        item["point"] = points[item["candidate_index"]].tolist()
    return dict(point=candidates[0]["point"], candidates=candidates,
                all_points=points.tolist(), all_scores=output["scores"].float().cpu().tolist(),
                parent_indices=record["parent_indices"].tolist(), target_name=name,
                grid_shape=list(meta["grid_shape"]), decoder="own", ground_truth_in_readout_inputs=False)


def save_crossfit_proposer(path, model, metadata, config=None):
    """Clone CPU state without moving the model or changing train/eval mode."""
    config = config or getattr(model, "crossfit_config", CrossfitProposerConfig())
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    if not torch.equal(state["head_prior"], torch.zeros(96)):
        raise ValueError("Crossfit proposer head_prior must remain the fixed all-zero constant")
    payload = dict(schema_version=SCHEMA_VERSION, config=asdict(config), connect_fine=True,
                   initial_state_sha256=getattr(model, "initial_state_sha256", None),
                   state_dict=state, metadata=metadata)
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary); temporary.replace(path)


def load_crossfit_proposer(path, device="cpu"):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("schema_version") != SCHEMA_VERSION or saved.get("connect_fine") is not True:
        raise ValueError("Not a fresh crossfit proposer checkpoint")
    config = CrossfitProposerConfig(**saved["config"])
    model = make_fresh_proposer(config)
    if model.initial_state_sha256 != saved["initial_state_sha256"]:
        raise ValueError("Fresh initialization identity mismatch")
    if not torch.equal(saved["state_dict"]["head_prior"], torch.zeros(96)):
        raise ValueError("A learned/nonconstant head prior is forbidden")
    model.load_state_dict(saved["state_dict"], strict=True)
    return model.to(device).eval(), saved["metadata"]
