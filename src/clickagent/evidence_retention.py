"""Independent V14 candidate retention. Inference accepts no target geometry."""
from __future__ import annotations

import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

BASE_DIM = 32
STATE_DIM = 294


def require(condition, message):
    if not condition:
        raise ValueError(message)


def map_point(point, transform):
    if transform is None:
        return list(map(int, point))
    x0, y0, _, _ = transform['box']
    w, h = transform['original_size']
    scale = transform['scale']
    return [int(np.clip(round(point[0]/scale+x0), 0, w-1)),
            int(np.clip(round(point[1]/scale+y0), 0, h-1))]


def view_rows(result, view, transform=None):
    points, scores = result['all_points'], result.get('all_scores', [])
    require(len(points) == len(scores), 'Unaligned archived points and scores')
    require(all(math.isfinite(s) for s in scores), 'Nonfinite score')
    order = sorted(range(len(points)), key=lambda i: -scores[i])
    return [dict(point=map_point(points[i], transform), rank=rank, score=float(scores[i]), view=view)
            for rank, i in enumerate(order, 1)]


def build_base(pipeline, image_size):
    """Only predictions, public view geometry and scores; no GT parameter."""
    local = pipeline['local_views']
    views = [view_rows(pipeline['global_result'], 0)] + [
        view_rows(v['result'], i+1, v['transform']) for i, v in enumerate(local)]
    while len(views) < 4:
        views.append([])
    rr = [v[i] for i in range(max(map(len, views[1:]), default=0)) for v in views[1:] if i < len(v)]
    universe, lookup = [], {}
    for item in views[0]+rr:
        p = tuple(item['point'])
        if p not in lookup:
            lookup[p] = len(universe); universe.append(list(p))
    points = np.asarray(universe, np.int64).reshape(-1, 2)
    n = len(points)
    x = np.zeros((n, BASE_DIM), np.float32)
    source_count = np.zeros(n, np.float32)
    for j, rows in enumerate(views):
        if not rows:
            continue
        scores = np.asarray([r['score'] for r in rows])
        center, spread = np.median(scores), max(float(np.std(scores)), 1e-3)
        seen = set()
        for item in rows:
            p = tuple(item['point'])
            if p in seen:
                continue
            seen.add(p); i = lookup[p]; rank = item['rank']; score = item['score']
            x[i, j*7:(j+1)*7] = [1, np.log1p(rank), 1/rank, score,
                (score-center)/spread, rank/len(rows), np.log1p(len(rows))]
            source_count[i] += 1
    if n:
        x[:, 28:30] = points / np.asarray(image_size)
        x[:, 30] = source_count/4
        x[:, 31] = sum(bool(v) for v in views)/4
    active = np.zeros((n, 4), bool)
    if views[0]:
        active[:, 0] = True
    for j, view in enumerate(local, 1):
        if not views[j]:
            continue
        a = np.asarray(view['transform']['box'])
        active[:, j] = ((points >= a[:2]) & (points < a[2:])).all(-1)
    baseline = np.asarray([lookup[tuple(v['point'])] for v in pipeline['pools']['c3']['16']], np.int64)
    require(len(set(baseline.tolist())) == len(baseline), 'Duplicate baseline coordinate')
    require(np.isfinite(x).all(), 'Nonfinite base feature')
    return dict(points=points, base=x, active=active, baseline=baseline)


def state_features(arrays, metadata, points, image_size, transform=None):
    """Preserve every target token and all 96 heads until learned interaction.

    Returns [N,T,294]. Channels: 96 center relative log-attentions, 96 relative
    to head peak, 96 local 3x3 means, 3 query/image cosines, 3 query log-norms.
    This deliberately tests one explicit subset of internal states, not every
    possible internal representation. No image re-encoding is performed.
    """
    gh, gw = map(int, metadata['grid_shape']); p = gh*gw
    a = np.asarray(arrays['target_attention'], np.float32)
    q = np.asarray(arrays['target_hidden'], np.float32)
    image = np.asarray(arrays['image_hidden'], np.float32)
    require(a.ndim == 3 and a.shape[1:] == (96, p), 'Expected complete own-token attention')
    require(q.shape == (len(a), 12288) and image.shape == (p, 4096), 'Unaligned query/image states')
    require(all(np.isfinite(v).all() for v in (a, q, image)), 'Nonfinite raw state')
    pts = np.asarray(points)
    bounds = np.asarray([0, 0, *image_size] if transform is None else transform['box'])
    xy = (pts-bounds[:2])/(bounds[2:]-bounds[:2])
    valid = ((xy >= 0) & (xy < 1)).all(-1)
    n, t = len(pts), len(a)
    output = np.zeros((n, t, STATE_DIM), np.float16)
    if not n or not t:
        return output, np.zeros(n, bool)
    cell = np.clip(np.floor(xy*[gw, gh]).astype(np.int64), [0, 0], [gw-1, gh-1])
    index = cell[:, 1]*gw+cell[:, 0]
    relative = a/np.maximum(a.sum(-1, keepdims=True), 1e-12)
    center = relative[..., index].transpose(2, 0, 1)
    peak = np.maximum(relative.max(-1), 1e-12)
    summed = np.zeros_like(center); count = np.zeros(n)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            col, row = cell[:, 0]+dx, cell[:, 1]+dy
            inside = (col >= 0) & (col < gw) & (row >= 0) & (row < gh)
            ix = np.clip(row, 0, gh-1)*gw+np.clip(col, 0, gw-1)
            summed += relative[..., ix].transpose(2, 0, 1)*inside[:, None, None]
            count += inside
    semantic = image[index]
    semantic /= np.maximum(np.linalg.norm(semantic, axis=-1, keepdims=True), 1e-8)
    query = q.reshape(t, 3, 4096)
    norms = np.linalg.norm(query, axis=-1)
    query = query/np.maximum(norms[..., None], 1e-8)
    cosine = np.einsum('nd,tld->ntl', semantic, query, optimize=True)
    values = np.concatenate((np.log1p(p*center), center/peak[None],
        np.log1p(p*summed/np.maximum(count[:, None, None], 1)), cosine,
        np.broadcast_to(np.log1p(norms/math.sqrt(4096))[None], (n, t, 3))), -1)
    output[:] = np.where(valid[:, None, None], values, 0).astype(np.float16)
    require(np.isfinite(output).all(), 'Nonfinite extracted state feature')
    return output, valid


def permute_evidence(state, mask, rng):
    """Shuffle spatial evidence within each present view; preserve masks/T."""
    output = state.copy()
    for v in range(state.shape[1]):
        ix = np.flatnonzero(mask[:, v].any(-1))
        if len(ix) > 1:
            order = ix[rng.permutation(len(ix))]
            output[ix, v] = state[order, v]
    return output


def select_indices(scores, budget):
    require(0 <= budget <= len(scores), 'Invalid candidate budget')
    require(np.isfinite(scores).all(), 'Nonfinite retention score')
    # Canonical downstream order removes preselector order as a tie confound.
    return np.sort(np.argsort(-np.asarray(scores), kind='stable')[:budget]).astype(np.int64)


class RetentionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_encoder = nn.Sequential(nn.Linear(BASE_DIM, 32), nn.GELU(), nn.LayerNorm(32))
        self.state_encoder = nn.Sequential(nn.Linear(STATE_DIM, 32), nn.GELU(), nn.Linear(32, 32), nn.LayerNorm(32))
        self.output = nn.Sequential(nn.Linear(32+128, 64), nn.GELU(), nn.Linear(64, 1))
        self.null = nn.Parameter(torch.zeros(()))

    def forward(self, base, state=None, mask=None, active=None):
        n = len(base)
        if state is None:
            state = base.new_zeros((n, 4, 1, STATE_DIM))
            mask = active.bool()[:, :, None]
        require(state.shape[:3] == mask.shape and state.shape[0] == n, 'State/mask alignment')
        encoded = self.state_encoder(state.float())
        present = mask.any(2)
        safe_mask = mask.clone()
        safe_mask[:, :, 0] |= ~present
        masked = encoded.masked_fill(~safe_mask[..., None], -torch.inf)
        counts = safe_mask.sum(2).clamp_min(1)
        maximum = masked.amax(2)
        lme = torch.logsumexp(masked, 2)-counts[..., None].float().log()
        per_view = torch.cat((maximum, lme), -1)
        per_view = torch.where(present[..., None], per_view, torch.zeros_like(per_view))
        vmax = per_view.masked_fill(~present[..., None], -torch.inf).amax(1)
        vmean = per_view.sum(1)/present.sum(1).clamp_min(1)[:, None]
        vmax = torch.where(present.any(1)[:, None], vmax, torch.zeros_like(vmax))
        return self.output(torch.cat((self.base_encoder(base), vmax, vmean), -1)).squeeze(-1)


def retention_loss(scores, labels, null):
    positive = labels > .5
    all_scores = torch.cat((scores, null.reshape(1)))
    target = torch.cat((positive, (~positive.any()).reshape(1)))
    rank = torch.logsumexp(all_scores, 0)-torch.logsumexp(all_scores[target], 0)
    bce = F.binary_cross_entropy_with_logits(scores, labels.float(), reduction='none')
    balanced = (.5*bce[positive].mean()+.5*bce[~positive].mean()
                if positive.any() and (~positive).any() else bce.mean())
    return rank+.25*balanced
