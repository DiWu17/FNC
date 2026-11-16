"""Fixed-pool capacity and cached local-content probes; no annotation inputs."""
from functools import lru_cache
import hashlib
import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .evidence_retention import BASE_DIM, STATE_DIM

ROI_DIM = 160
PROJECTION_SEED = 20261901


@lru_cache(maxsize=4)
def projection(dimension):
    rng = np.random.default_rng(PROJECTION_SEED + int(dimension))
    # Unit-normalized source vectors give approximately unit-variance channels.
    return (2*rng.integers(0, 2, (dimension, 32), dtype=np.int8)-1).astype(np.float32)


def projected(values):
    values = np.asarray(values, np.float32)
    values = values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-8)
    return values @ projection(values.shape[-1])


def local_view_features(arrays, metadata, points, image_size, transform=None):
    """Fine cell + local neighborhood + coarse content + query + interaction.

    Content comes only from the original frozen VLM cache. This is a probe of
    existing-resolution representations, not a newly encoded high-res crop.
    """
    gh, gw = map(int, metadata['grid_shape'])
    image = np.asarray(arrays['image_hidden'])
    vision = np.asarray(arrays['vision_hidden'])
    query = np.asarray(arrays['target_hidden'])
    assert image.shape == (gh*gw, 4096)
    assert vision.shape == (4*gh*gw, 1152)
    assert query.ndim == 2 and query.shape[1] == 12288
    assert list(metadata['fine_grid_shape']) == [2*gh, 2*gw]
    assert not metadata['ground_truth_in_model_input'] and metadata['all_backbone_parameters_frozen']
    bounds = np.asarray([0, 0, *image_size] if transform is None else transform['box'], float)
    xy = (np.asarray(points).reshape(-1, 2)-bounds[:2])/(bounds[2:]-bounds[:2])
    valid = ((xy >= 0) & (xy < 1)).all(-1)
    coarse = np.clip(np.floor(xy*[gw, gh]).astype(int), [0, 0], [gw-1, gh-1])
    fine = np.clip(np.floor(xy*[2*gw, 2*gh]).astype(int), [0, 0], [2*gw-1, 2*gh-1])
    vf = projected(vision).reshape(2*gh, 2*gw, 32)
    center = vf[fine[:, 1], fine[:, 0]]
    context = np.zeros_like(center); count = np.zeros(len(points), np.float32)
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            x, y = fine[:, 0]+dx, fine[:, 1]+dy
            inside = (x >= 0) & (x < 2*gw) & (y >= 0) & (y < 2*gh)
            context += vf[np.clip(y, 0, 2*gh-1), np.clip(x, 0, 2*gw-1)]*inside[:, None]
            count += inside
    context /= np.maximum(count[:, None], 1)
    semantic = projected(image)[coarse[:, 1]*gw+coarse[:, 0]]
    # Match decoder image layer15 to the corresponding first query layer.
    q = projected(query.reshape(-1, 3, 4096)[:, 0].mean(0, keepdims=True))[0] if len(query) else np.zeros(32, np.float32)
    q = np.broadcast_to(q, semantic.shape)
    result = np.concatenate([center, context, semantic, q, semantic*q], -1)
    result[~valid] = 0
    assert result.shape == (len(points), ROI_DIM) and np.isfinite(result).all()
    return result.astype(np.float16), valid


def shuffle_indices(valid, sample_id):
    """Preserve each view's feature multiset/mask; break candidate association."""
    valid = np.asarray(valid, bool)
    result = np.broadcast_to(np.arange(len(valid))[:, None], valid.shape).copy()
    for view in range(valid.shape[1]):
        seed = int(hashlib.sha256(f'{sample_id}|v19_visual_shuffle|{view}'.encode()).hexdigest()[:16], 16)
        indices = np.flatnonzero(valid[:, view])
        result[indices, view] = np.random.default_rng(seed).permutation(indices)
    return result


def shuffled(features, valid, sample_id):
    indices = shuffle_indices(valid, sample_id)
    return features[indices, np.arange(valid.shape[1])[None, :]]


class BottleneckModel(nn.Module):
    """Width32/no-visual exactly preserves the original parameter layout."""
    def __init__(self, width=64, visual=False):
        super().__init__()
        self.visual = visual
        self.base_encoder = nn.Sequential(nn.Linear(BASE_DIM, width), nn.GELU(), nn.LayerNorm(width))
        self.state_encoder = nn.Sequential(nn.Linear(STATE_DIM, width), nn.GELU(),
                                           nn.Linear(width, width), nn.LayerNorm(width))
        if visual:
            self.visual_encoder = nn.Sequential(nn.Linear(ROI_DIM, 64), nn.GELU(), nn.Linear(64, 32), nn.LayerNorm(32))
        self.output = nn.Sequential(nn.Linear(5*width+(64 if visual else 0), 2*width), nn.GELU(), nn.Linear(2*width, 1))
        self.null = nn.Parameter(torch.zeros(()))

    def forward(self, base, state, mask, active=None, roi=None, roi_mask=None):
        encoded = self.state_encoder(state.float())
        present = mask.any(2); safe_mask = mask.clone(); safe_mask[:, :, 0] |= ~present
        masked = encoded.masked_fill(~safe_mask[..., None], -torch.inf)
        counts = safe_mask.sum(2).clamp_min(1)
        per_view = torch.cat([masked.amax(2), torch.logsumexp(masked, 2)-counts[..., None].float().log()], -1)
        per_view = torch.where(present[..., None], per_view, torch.zeros_like(per_view))
        vmax = per_view.masked_fill(~present[..., None], -torch.inf).amax(1)
        vmean = per_view.sum(1)/present.sum(1).clamp_min(1)[:, None]
        vmax = torch.where(present.any(1)[:, None], vmax, torch.zeros_like(vmax))
        pieces = [self.base_encoder(base), vmax, vmean]
        if self.visual:
            assert roi is not None and roi_mask is not None
            v = self.visual_encoder(roi.float())
            v = torch.where(roi_mask[..., None], v, torch.zeros_like(v))
            maximum = v.masked_fill(~roi_mask[..., None], -torch.inf).amax(1)
            maximum = torch.where(roi_mask.any(1)[:, None], maximum, torch.zeros_like(maximum))
            mean = v.sum(1)/roi_mask.sum(1).clamp_min(1)[:, None]
            pieces += [maximum, mean]
        return self.output(torch.cat(pieces, -1)).squeeze(-1)


def probe_loss(scores, labels, null, bce_weight=.25):
    positive = labels > .5
    all_scores = torch.cat([scores, null.reshape(1)])
    target = torch.cat([positive, (~positive.any()).reshape(1)])
    rank = torch.logsumexp(all_scores, 0)-torch.logsumexp(all_scores[target], 0)
    if bce_weight == 0:
        return rank
    bce = F.binary_cross_entropy_with_logits(scores, labels.float(), reduction='none')
    balanced = (.5*bce[positive].mean()+.5*bce[~positive].mean()
                if positive.any() and (~positive).any() else bce.mean())
    return rank+bce_weight*balanced
