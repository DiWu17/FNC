"""Dedicated frozen-Qwen visual re-encoder and a trainable candidate verifier.

This is a NEW supervised visual verifier, not a GUI-Actor derivative. Only
``model.visual.*`` tensors are read from the local Qwen checkpoint. A screenshot
is encoded once; all candidate ROIs share its unmerged detail and already
language-projected merger features. Existing full-image generated-name hidden
tokens provide the instruction condition without another language-model pass.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import struct
import time
from typing import Any, Sequence

import numpy as np
from safetensors import safe_open
import torch
from torch import nn
import torch.nn.functional as F


SCHEMA_VERSION = 1
# Feature geometry/head structure is unchanged, but revision-1 visual caches
# used rounded FP16 rotary frequencies and must not enter the new experiment.
ENCODER_REVISION = 2
VISUAL_PREFIX = "model.visual."
CANDIDATE_KEYS = ("fine_roi", "context_roi", "merged_roi", "fine_mask", "context_mask", "merged_mask",
                  "fine_positions", "context_positions", "merged_positions", "candidate_geometry",
                  "candidate_points", "candidate_lengths")
MASK_KEYS = ("fine_mask", "context_mask", "merged_mask")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _positive_size(value, name):
    if not isinstance(value, (list, tuple)) or len(value) != 2 or any(isinstance(v, bool) or int(v) != v or v < 1 for v in value):
        raise ValueError(f"{name} requires positive integer [width,height] or [height,width]")
    return tuple(map(int, value))


def inspect_visual_checkpoint(model_dir: str | Path) -> dict[str, Any]:
    """Read index/config/safetensor JSON headers only, never weight tensors."""
    root = Path(model_dir).resolve()
    if "gui-actor" in str(root).casefold() or "gui_actor" in str(root).casefold():
        raise ValueError("GUI-Actor checkpoints are prohibited for this independent verifier")
    config_path, index_path = root / "config.json", root / "model.safetensors.index.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("model_type") != "qwen3_vl" or not isinstance(config.get("vision_config"), dict):
        raise ValueError("Expected a local Qwen3-VL checkpoint with vision_config")
    weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
    visual = {key: shard for key, shard in weight_map.items() if key.startswith(VISUAL_PREFIX)}
    if not visual:
        raise ValueError("Checkpoint index contains no model.visual.* weights")
    parameters = stored_bytes = 0
    tensors = []
    for shard in sorted(set(visual.values())):
        path = (root / shard).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Unsafe checkpoint shard path")
        with path.open("rb") as stream:
            length = struct.unpack("<Q", stream.read(8))[0]
            if length > 100_000_000:
                raise ValueError("Unreasonably large safetensors JSON header")
            header = json.loads(stream.read(length))
        for name in sorted(key for key, filename in visual.items() if filename == shard):
            item = header[name]
            count = math.prod(item["shape"])
            size = item["data_offsets"][1] - item["data_offsets"][0]
            parameters += count
            stored_bytes += size
            tensors.append({"name": name, "shard": shard, "shape": item["shape"], "dtype": item["dtype"]})
    return {"model_dir": str(root), "vision_config": config["vision_config"], "visual_weight_map": visual,
            "visual_tensor_count": len(visual), "visual_parameter_count": parameters, "stored_visual_bytes": stored_bytes,
            "config_sha256": _sha(config_path), "index_sha256": _sha(index_path), "tensors": tensors,
            "loaded_weight_prefix": VISUAL_PREFIX, "language_weights_loaded": False, "gui_actor_used": False}


def restore_premerge_raster(block_order_features, grid_shape: Sequence[int], merge_size: int = 2):
    """Invert Qwen's [block-row,block-col,intra-row,intra-col] permutation."""
    h, w = _positive_size(list(grid_shape), "grid_shape")
    if h % merge_size or w % merge_size or block_order_features.ndim != 2 or len(block_order_features) != h * w:
        raise ValueError("Unmerged feature count/grid/merge size mismatch")
    c = block_order_features.shape[-1]
    shaped = block_order_features.reshape(h // merge_size, w // merge_size, merge_size, merge_size, c)
    if torch.is_tensor(shaped):
        return shaped.permute(0, 2, 1, 3, 4).reshape(h, w, c)
    return shaped.transpose(0, 2, 1, 3, 4).reshape(h, w, c)


class FrozenVisualEncoder:
    """Standalone vision encoder; construction never instantiates a text model."""
    def __init__(self, vision, image_processor, provenance, *, device="cpu", dtype=torch.float16,
                 max_pixels=1048576, min_pixels=16384):
        self.device, self.dtype = torch.device(device), dtype
        rotary = vision.rotary_pos_emb.inv_freq
        if rotary.dtype != torch.float32 or rotary.is_meta:
            raise ValueError("Vision rotary inv_freq must be freshly initialized FP32 before dtype conversion; half().float() cannot restore its original precision")
        # PyTorch Module.to(dtype=...) casts floating buffers as well as model
        # parameters. Qwen deliberately constructs its nonpersistent rotary
        # frequencies in FP32, and they are absent from the checkpoint. Clone
        # BEFORE casting: converting an already rounded FP16 value back to FP32
        # would silently retain the numerical error.
        preserved = []
        for module_name, module in vision.named_modules():
            for name in sorted(module._non_persistent_buffers_set):
                buffer = module._buffers.get(name)
                if buffer is not None and buffer.dtype == torch.float32:
                    preserved.append((module_name, name, buffer.detach().clone()))
        self.vision = vision.to(device=self.device, dtype=dtype).eval().requires_grad_(False)
        self.preserved_float32_buffers = []
        for module_name, name, original in preserved:
            module = self.vision.get_submodule(module_name)
            # Replacing the existing buffer retains persistent=False, so it
            # cannot be confused with a loaded or trainable checkpoint tensor.
            module._buffers[name] = original.to(device=self.device)
            self.preserved_float32_buffers.append(f"{module_name}.{name}".lstrip("."))
        self.image_processor, self.provenance = image_processor, provenance
        self.max_pixels, self.min_pixels = int(max_pixels), int(min_pixels)
        self.load_seconds = 0.0

    @classmethod
    def from_local(cls, model_dir: str | Path, device="cpu", dtype=torch.float16,
                   max_pixels=1048576, min_pixels=16384, *, image_processor=None):
        """Load ONLY visual tensors, on CPU first, then the requested device.

        Empty parameters prevent an unnecessary random full-size allocation;
        nonpersistent rotary buffers stay materialized on CPU. No from_pretrained
        call can accidentally load the language model or download anything.
        """
        from accelerate import init_empty_weights
        from transformers import AutoImageProcessor
        from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
        started = time.perf_counter()
        provenance = inspect_visual_checkpoint(model_dir)
        config = Qwen3VLVisionConfig(**provenance["vision_config"])
        config._attn_implementation = "sdpa"
        with init_empty_weights(include_buffers=False):
            vision = Qwen3VLVisionModel(config)
        state = {}
        weights = provenance["visual_weight_map"]
        for shard in sorted(set(weights.values())):
            with safe_open(str(Path(model_dir) / shard), framework="pt", device="cpu") as stream:
                for name in sorted(key for key, filename in weights.items() if filename == shard):
                    state[name[len(VISUAL_PREFIX):]] = stream.get_tensor(name).to(dtype=dtype)
        # Strict validation rejects missing/unexpected visual keys. assign=True
        # reuses the loaded CPU tensors rather than copying into random params.
        vision.load_state_dict(state, strict=True, assign=True)
        del state
        if image_processor is None:
            image_processor = AutoImageProcessor.from_pretrained(str(model_dir), local_files_only=True,
                                                                 min_pixels=min_pixels, max_pixels=max_pixels)
        instance = cls(vision, image_processor, provenance, device=device, dtype=dtype,
                       max_pixels=max_pixels, min_pixels=min_pixels)
        instance.load_seconds = time.perf_counter() - started
        return instance

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def encode(self, image) -> dict[str, Any]:
        """One full screenshot forward; return CPU float16 raster caches."""
        self.vision.eval()
        if any(parameter.requires_grad for parameter in self.vision.parameters()):
            raise RuntimeError("Dedicated visual backbone must remain frozen")
        if self.vision.rotary_pos_emb.inv_freq.dtype != torch.float32:
            raise RuntimeError("Vision rotary inv_freq was cast after loading; reconstruct the encoder to preserve original FP32 values")
        self._sync()
        started = time.perf_counter()
        image = image.convert("RGB")
        w, h = image.size
        pixel_hash = hashlib.sha256(f"{w}x{h}:RGB:".encode() + image.tobytes()).hexdigest()
        inputs = self.image_processor(images=[image], return_tensors="pt", device="cpu")
        grids = inputs["image_grid_thw"]
        if tuple(grids.shape) != (1, 3) or int(grids[0, 0]) != 1:
            raise ValueError("Dedicated verifier requires one still screenshot per encoding")
        gh, gw = int(grids[0, 1]), int(grids[0, 2])
        merge = int(self.vision.config.spatial_merge_size)
        captured = {}
        def before_merger(module, args):
            captured["premerge"] = args[0].detach()
        hook = self.vision.merger.register_forward_pre_hook(before_merger)
        prepared = time.perf_counter()
        try:
            with torch.inference_mode():
                values = inputs["pixel_values"].to(device=self.device, dtype=self.dtype)
                merged, _ = self.vision(values, grid_thw=grids.to(self.device))
                self._sync()
                forwarded = time.perf_counter()
                raw = captured["premerge"]
                detail = restore_premerge_raster(raw, (gh, gw), merge)
                if merged.shape != (gh * gw // (merge * merge), self.vision.config.out_hidden_size):
                    raise RuntimeError("Official vision merger output does not match its raster grid")
                semantic = merged.reshape(gh // merge, gw // merge, -1)
                detail = detail.to(device="cpu", dtype=torch.float16).numpy().copy()
                semantic = semantic.to(device="cpu", dtype=torch.float16).numpy().copy()
                self._sync()
                finished = time.perf_counter()
        finally:
            hook.remove()
        metadata = {"schema_version": SCHEMA_VERSION, "encoder_revision": ENCODER_REVISION,
                    "rotary_precision_policy": "clone_original_fp32_before_parameter_cast_then_restore_on_target_device",
                    "rotary_inv_freq_dtype": str(self.vision.rotary_pos_emb.inv_freq.dtype),
                    "preserved_float32_nonpersistent_buffers": self.preserved_float32_buffers,
                    "pixel_sha256": pixel_hash, "image_size": [w, h],
                    "grid_thw": [1, gh, gw], "premerge_grid_shape": [gh, gw],
                    "merged_grid_shape": [gh // merge, gw // merge], "patch_size": int(self.vision.config.patch_size),
                    "spatial_merge_size": merge, "max_pixels": self.max_pixels, "min_pixels": self.min_pixels,
                    "resized_image_size": [gw * self.vision.config.patch_size, gh * self.vision.config.patch_size],
                    "detail_source": "final vision block output before main merger, 2x2 block order inverted to raster",
                    "semantic_source": "official final visual merger output, native merged raster, language-input projection",
                    "premerge_channel_dim": int(detail.shape[-1]), "merged_channel_dim": int(semantic.shape[-1]),
                    "forward_calls": 1, "all_visual_parameters_frozen": True, "language_model_loaded": False,
                    "gui_actor_used": False, "device": str(self.device), "dtype": str(self.dtype),
                    "timing_seconds": {"prepare": prepared - started, "vision_forward": forwarded - prepared,
                                       "cache_copy": finished - forwarded, "total": finished - started},
                    "model_loading_excluded_from_timing": True,
                    "vision_config_sha256": self.provenance.get("config_sha256"),
                    "vision_index_sha256": self.provenance.get("index_sha256"),
                    "model_dir": self.provenance.get("model_dir"), "loaded_weight_prefix": VISUAL_PREFIX}
        return {"features": detail, "merged_features": semantic, "metadata": metadata}


def save_visual_encoding(path: str | Path, encoding: dict[str, Any]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, features=np.asarray(encoding["features"], dtype=np.float16),
                            merged_features=np.asarray(encoding["merged_features"], dtype=np.float16),
                            metadata=np.asarray(json.dumps(encoding["metadata"], ensure_ascii=False, allow_nan=False)))
    temporary.replace(path)


def load_visual_encoding(path: str | Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        metadata = data["metadata"]
        if metadata.ndim != 0 or metadata.dtype.kind not in "US":
            raise ValueError("Visual cache metadata must be a safe scalar JSON string")
        result = {"features": data["features"].copy(), "merged_features": data["merged_features"].copy(),
                  "metadata": json.loads(str(metadata))}
    if result["metadata"].get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Visual encoding schema mismatch")
    if result["metadata"].get("loaded_weight_prefix") == VISUAL_PREFIX and result["metadata"].get("encoder_revision") != ENCODER_REVISION:
        raise ValueError("Qwen visual cache predates the FP32 rotary fix; preserve it for diagnosis and re-encode into a fresh cache")
    for key in ("features", "merged_features"):
        if result[key].ndim != 3 or not np.isfinite(result[key]).all():
            raise ValueError(f"Invalid {key} raster")
    return result


def _roi(features, points, image_size, side, radius):
    """Sample token-center fields in original-image coordinates, align_corners=False."""
    w, h = image_size
    n = len(points)
    offsets = torch.linspace(-1, 1, side, dtype=torch.float32)
    yy, xx = torch.meshgrid(offsets, offsets, indexing="ij")
    relative = torch.stack((xx.flatten(), yy.flatten()), dim=-1)
    xy = points[:, None] + relative[None] * radius
    uv = xy / torch.tensor([w, h], dtype=torch.float32)
    mask = (xy[..., 0] >= 0) & (xy[..., 0] < w) & (xy[..., 1] >= 0) & (xy[..., 1] < h)
    positions = torch.cat((relative.expand(n, -1, -1), uv), dim=-1)
    if n:
        feature_map = torch.as_tensor(np.asarray(features, dtype=np.float32)).permute(2, 0, 1)[None].expand(n, -1, -1, -1)
        sampled = F.grid_sample(feature_map, (uv * 2 - 1).reshape(n, side, side, 2),
                                mode="bilinear", padding_mode="border", align_corners=False)
        sampled = sampled.permute(0, 2, 3, 1).reshape(n, side * side, -1) * mask[..., None]
    else:
        sampled = torch.empty((0, side * side, features.shape[-1]), dtype=torch.float32)
    return sampled.numpy(), mask.numpy(), positions.numpy()


def build_visual_inputs(encoding: dict[str, Any], points: Sequence[Sequence[float]], target_hidden,
                        *, fine_radius=64.0, context_radius=256.0) -> dict[str, np.ndarray]:
    """Cache-ready candidate arrays; target_hidden stays [T,12288], never averaged.

    Coordinate metadata uses original image pixels, even when visual processing
    resized the screenshot. Candidates outside the screenshot are rejected,
    never clamped or repaired. ROI padding is masked; exact subpatch position
    and point location inside the clipped visible ROI remain explicit.
    """
    metadata = encoding["metadata"]
    w, h = _positive_size(metadata["image_size"], "image_size")
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    target_hidden = np.asarray(target_hidden, dtype=np.float32)
    if not np.isfinite(points).all() or (points[:, 0] < 0).any() or (points[:, 0] >= w).any() or (points[:, 1] < 0).any() or (points[:, 1] >= h).any():
        raise ValueError("Candidate points must be finite, original-image pixel positions inside the screenshot")
    if target_hidden.ndim != 2 or not len(target_hidden) or not np.isfinite(target_hidden).all():
        raise ValueError("target_hidden must contain the actual nonempty generated-name token sequence")
    if fine_radius <= 0 or context_radius <= 0:
        raise ValueError("ROI radii must be positive original-image pixel distances")
    features, merged = np.asarray(encoding["features"]), np.asarray(encoding["merged_features"])
    if features.ndim != 3 or merged.ndim != 3:
        raise ValueError("Both detail and semantic visual caches require HWC raster features")
    pt = torch.from_numpy(points)
    result = {}
    for name, raster, side, radius in (("fine", features, 5, fine_radius), ("context", features, 7, context_radius), ("merged", merged, 3, context_radius)):
        values, mask, positions = _roi(raster, pt, (w, h), side, radius)
        result.update({name + "_roi": values, name + "_mask": mask, name + "_positions": positions})
    gh, gw = features.shape[:2]
    uv = points / np.asarray([w, h], dtype=np.float32)
    geometry = [uv, 1 - uv,
                np.tile([fine_radius / w, fine_radius / h, context_radius / w, context_radius / h], (len(points), 1))]
    for radius in (fine_radius, context_radius):
        lower = np.maximum(points - radius, 0)
        upper = np.minimum(points + radius, np.asarray([w, h], dtype=np.float32))
        geometry.append((points - lower) / np.maximum(upper - lower, 1e-8))
    geometry += [np.stack((result["fine_mask"].mean(1), result["context_mask"].mean(1)), axis=-1),
                 np.mod(uv * [gw, gh] - 0.5, 1),
                 np.tile([1 / gw, 1 / gh], (len(points), 1)),
                 np.tile(np.log1p([w, h]) / math.log1p(8192), (len(points), 1))]
    result.update(candidate_geometry=np.concatenate(geometry, axis=-1).astype(np.float32),
                  candidate_points=points.copy(), target_hidden=target_hidden.copy(),
                  candidate_lengths=sum(result[name].sum(1) for name in MASK_KEYS).astype(np.int64))
    return result


def subset_visual_inputs(inputs: dict[str, np.ndarray], indices: Sequence[int]) -> dict[str, np.ndarray]:
    indices = np.asarray(indices, dtype=np.int64)
    n = len(inputs["candidate_points"])
    if indices.ndim != 1 or (indices < 0).any() or (indices >= n).any():
        raise ValueError("Candidate subset indices are out of bounds")
    return {key: np.asarray(value)[indices].copy() if key in CANDIDATE_KEYS else np.asarray(value).copy() for key, value in inputs.items()}


def visual_inputs_to_torch(inputs: dict[str, np.ndarray], device="cpu") -> dict[str, torch.Tensor]:
    return {key: torch.as_tensor(value, device=device, dtype=torch.bool if key in MASK_KEYS else torch.long if key == "candidate_lengths" else torch.float32)
            for key, value in inputs.items()}


@dataclass
class VisualVerifierConfig:
    vision_dim: int = 1152
    semantic_dim: int = 4096
    target_dim: int = 12288
    width: int = 128
    heads: int = 4
    dropout: float = 0.1

    def __post_init__(self):
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in (self.vision_dim, self.semantic_dim, self.target_dim, self.width, self.heads)):
            raise ValueError("Feature and model dimensions must be positive integers")
        if self.width % self.heads or self.target_dim % self.semantic_dim:
            raise ValueError("width must divide heads and target_dim must contain semantic_dim chunks")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must lie in [0,1)")


class VisualCandidateVerifier(nn.Module):
    """One case, all matched candidates: conditioned ROI readout + set comparison.

    No candidate-index embedding is used: scores are permutation equivariant
    and null is invariant in eval mode. All target tokens are retained through
    candidate-specific cross-attention. Frozen cached inputs are detached.
    """
    def __init__(self, config: VisualVerifierConfig | None = None):
        super().__init__()
        self.config = config or VisualVerifierConfig()
        c, d = self.config, self.config.width
        self.detail = nn.Sequential(nn.LayerNorm(c.vision_dim), nn.Linear(c.vision_dim, d))
        self.shared_semantic = nn.Linear(c.semantic_dim, d)
        self.target_projection = nn.Linear(c.target_dim, d)
        self.position = nn.Sequential(nn.Linear(4, d), nn.GELU(), nn.Linear(d, d))
        self.roi_type = nn.Embedding(3, d)
        self.geometry = nn.Sequential(nn.Linear(20, d), nn.GELU(), nn.LayerNorm(d))
        self.target_attention = nn.MultiheadAttention(d, c.heads, dropout=c.dropout, batch_first=True)
        self.visual_attention = nn.MultiheadAttention(d, c.heads, dropout=c.dropout, batch_first=True)
        self.fusion = nn.Sequential(nn.Linear(d * 4, d * 2), nn.GELU(), nn.Dropout(c.dropout), nn.Linear(d * 2, d), nn.LayerNorm(d))
        self.comparison = nn.MultiheadAttention(d, c.heads, dropout=c.dropout, batch_first=True)
        self.score = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.null_query = nn.Parameter(torch.zeros(1, 1, d))
        self.null_target_attention = nn.MultiheadAttention(d, c.heads, dropout=c.dropout, batch_first=True)
        self.null_candidate_attention = nn.MultiheadAttention(d, c.heads, dropout=c.dropout, batch_first=True)
        self.null_head = nn.Sequential(nn.Linear(d * 3, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, fine_roi, context_roi, merged_roi, fine_mask, context_mask, merged_mask,
                fine_positions, context_positions, merged_positions, candidate_geometry, target_hidden,
                candidate_points=None, candidate_lengths=None):
        c = self.config
        if target_hidden.ndim != 2 or target_hidden.shape[1] != c.target_dim or len(target_hidden) == 0:
            raise ValueError("Expected every actual generated-name token in target_hidden[T,target_dim]")
        qraw = target_hidden.detach().float().reshape(len(target_hidden), -1, c.semantic_dim)
        qraw = F.layer_norm(qraw, [c.semantic_dim])
        target = self.target_projection(qraw.flatten(1)) + self.shared_semantic(qraw[:, 0])
        n = len(fine_roi)
        lengths = fine_mask.long().sum(1) + context_mask.long().sum(1) + merged_mask.long().sum(1)
        if candidate_lengths is not None and not torch.equal(candidate_lengths.to(lengths.device), lengths):
            raise ValueError("candidate_lengths disagrees with ROI masks")
        if candidate_geometry.shape != (n, 20):
            raise ValueError("candidate_geometry must contain 20 exact point/ROI fields")
        tokens, masks = [], []
        for index, (values, mask, positions, dim) in enumerate(((fine_roi, fine_mask, fine_positions, c.vision_dim),
                                                              (context_roi, context_mask, context_positions, c.vision_dim),
                                                              (merged_roi, merged_mask, merged_positions, c.semantic_dim))):
            if values.ndim != 3 or values.shape[0] != n or values.shape[-1] != dim or mask.shape != values.shape[:2] or positions.shape != (*values.shape[:2], 4):
                raise ValueError("Candidate ROI features/masks/position fields are misaligned")
            visual = self.detail(values.detach().float()) if index < 2 else self.shared_semantic(F.layer_norm(values.detach().float(), [c.semantic_dim]))
            tokens.append(visual + self.position(positions.detach().float()) + self.roi_type.weight[index])
            masks.append(mask.bool())
        if n and (lengths == 0).any():
            raise ValueError("Every candidate needs at least one valid visual sample")
        null_target, _ = self.null_target_attention(self.null_query, target[None], target[None], need_weights=False)
        if n:
            visual = torch.cat(tokens, dim=1)
            mask = torch.cat(masks, dim=1)
            # The central fine token is the exact bilinear point sample; it is
            # retained alongside explicit subpatch and clipped-ROI geometry.
            seed = tokens[0][:, fine_roi.shape[1] // 2] + self.geometry(candidate_geometry.detach().float())
            conditioned, target_weights = self.target_attention(seed[:, None], target[None].expand(n, -1, -1), target[None].expand(n, -1, -1))
            conditioned = conditioned[:, 0]
            observed, roi_weights = self.visual_attention(conditioned[:, None], visual, visual, key_padding_mask=~mask)
            observed = observed[:, 0]
            fused = self.fusion(torch.cat((seed, conditioned, observed, conditioned * observed), dim=-1))
            compared, _ = self.comparison(fused[None], fused[None], fused[None], need_weights=False)
            representation = fused + compared[0]
            scores = self.score(representation).squeeze(-1)
            null_candidates, _ = self.null_candidate_attention(null_target, representation[None], representation[None], need_weights=False)
        else:
            scores = target.new_empty((0,))
            null_candidates = torch.zeros_like(null_target)
            target_weights = target.new_empty((0, 1, len(target)))
            roi_weights = target.new_empty((0, 1, sum(token.shape[1] for token in tokens)))
        null_scores = self.null_head(torch.cat((null_target, null_candidates, null_target * null_candidates), dim=-1)).reshape(1)
        return {"scores": scores, "null_scores": null_scores, "candidate_lengths": lengths,
                "target_attention_weights": target_weights[:, 0], "roi_attention_weights": roi_weights[:, 0]}


def save_visual_verifier(path: str | Path, model: VisualCandidateVerifier, metadata: dict[str, Any]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({"schema_version": SCHEMA_VERSION, "config": asdict(model.config), "state_dict": state,
                "metadata": metadata}, temporary)
    temporary.replace(path)


def load_visual_verifier(path: str | Path, device="cpu"):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Visual verifier checkpoint schema mismatch")
    model = VisualCandidateVerifier(VisualVerifierConfig(**payload["config"]))
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.to(device), payload["metadata"]


def estimate_vision_cost(image_size: Sequence[int], candidate_count: int, *, crop_size=(512, 512),
                         max_pixels=1048576, crop_max_pixels=262144, min_pixels=16384):
    """Analytic token/attention-work comparison; never presented as measured latency."""
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
    w, h = _positive_size(list(image_size), "image_size")
    cw, ch = _positive_size(list(crop_size), "crop_size")
    if candidate_count < 0 or int(candidate_count) != candidate_count:
        raise ValueError("candidate_count must be a nonnegative integer")
    fh, fw = smart_resize(h, w, factor=32, min_pixels=min_pixels, max_pixels=max_pixels)
    lh, lw = smart_resize(min(ch, h), min(cw, w), factor=32, min_pixels=min_pixels, max_pixels=crop_max_pixels)
    full, local = fh * fw // (16 * 16), lh * lw // (16 * 16)
    return {"shared_full_image": {"vision_forward_calls": 1, "premerge_tokens": full, "quadratic_attention_proxy": full * full},
            "per_candidate_crops": {"vision_forward_calls": int(candidate_count), "premerge_tokens_total": int(candidate_count * local),
                                    "quadratic_attention_proxy_total": int(candidate_count * local * local)},
            "interpretation": "Work proxies, not timings. Global full-frame attention is quadratic, so shared high-resolution encoding is not automatically faster than small crops. Actual GPU timings and candidate count must be measured; no per-candidate language-model pass is required."}
