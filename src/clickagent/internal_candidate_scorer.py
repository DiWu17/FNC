"""Candidate-set scoring from frozen, unpooled Qwen3-VL feature archives.

Both views are queried at every proposed ORIGINAL-image point. The old crop is
masked when it does not contain that point. Raw layers 15/28/36 target queries,
layer15 spatial image content and premerge visual tokens get NEW trainable
projections; no old readout projection or verifier teacher is reused.

Target token/head evidence is conditioned on candidate content before token
max/logmeanexp reduction. All 96 head identities survive to the following MLP;
there is no four-group softmax over heads. One permutation-equivariant set
layer finally compares the actual requested candidate set, including a null.
"""
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


SCHEMA_VERSION = 1
FIELD_CHANNELS = 15
CANDIDATE_KEYS = frozenset(('coarse_indices', 'coarse_mask', 'fine_indices', 'fine_mask',
                            'attention_field', 'geometry', 'view_mask', 'points'))
VIEW_KEYS = frozenset(('query_hidden', 'query_mask', 'image_hidden', 'image_mask',
                       'image_xy', 'vision_hidden', 'grid_shapes'))
INPUT_KEYS = CANDIDATE_KEYS | VIEW_KEYS


def _require(condition, message):
    if not condition:
        raise ValueError(message)


@dataclass
class InternalCandidateScorerConfig:
    width: int = 96
    head_width: int = 8
    dropout: float = .15
    head_mode: str = 'full'

    def __post_init__(self):
        _require(isinstance(self.width, int) and not isinstance(self.width, bool) and self.width >= 16 and self.width % 4 == 0,
                 'width must be an integer >=16 divisible by4')
        _require(isinstance(self.head_width, int) and not isinstance(self.head_width, bool) and self.head_width > 0,
                 'head_width must be a positive integer')
        _require(0 <= self.dropout < 1, 'dropout must be in [0,1)')
        _require(self.head_mode in ('full', 'pooled_control', 'no_local'), 'Unknown head_mode')


def _raw_view(arrays, metadata):
    gh, gw = map(int, metadata['grid_shape'])
    _require(min(gh, gw) > 0, 'Invalid grid shape')
    p = gh*gw
    a = np.asarray(arrays['target_attention'], np.float32)
    q = np.asarray(arrays['target_hidden'])
    image = np.asarray(arrays['image_hidden'])
    vision = np.asarray(arrays['vision_hidden'])
    _require(a.ndim == 3 and a.shape[1:] == (96, p), 'Expected complete [T,96,P] attention')
    _require(q.shape == (len(a), 12288), 'All three query layers must be retained')
    _require(image.shape == (p, 4096), 'Spatial image content must align with the full grid')
    _require(vision.shape == (4*p, 1152), 'Vision must be restored full raster [4P,1152]')
    for value in (a, q, image, vision):
        _require(bool(np.isfinite(value).all()), 'Nonfinite raw feature')
    _require(bool((a >= 0).all()), 'Attention must be nonnegative')
    return a, q, image, vision, (gh, gw)


def _neighborhood(center, gh, gw, radius):
    yy, xx = np.meshgrid(np.arange(-radius, radius+1), np.arange(-radius, radius+1), indexing='ij')
    row = center[:, 1:2]+yy.reshape(1, -1)
    col = center[:, 0:1]+xx.reshape(1, -1)
    valid = (row >= 0) & (row < gh) & (col >= 0) & (col < gw)
    index = np.clip(row, 0, gh-1)*gw+np.clip(col, 0, gw-1)
    return index.astype(np.int64), valid


def build_internal_inputs(global_arrays, global_metadata, local_arrays, local_metadata,
                          points, image_size, transform=None):
    """Build one label-free, NumPy-only candidate record for K<=32 (or any K).

    ``points`` are integer [N,2] pixels in the original image, ``image_size`` is
    (width,height), and ``transform['box']`` is the old crop's original-pixel
    [left,top,right,bottom]. Optional local arguments are None when no crop.
    All token/head identities and full spatial arrays are retained. Geometry
    means predicted-point/view geometry only; GT is never accepted or read.

    Arrays: q[2,T,12288], image[2,P,4096], vision[2,4P,1152], field[N,2,T,96,15].
    First six field channels match build_attention_field; the remaining nine
    are ordered 3x3 log(1+P*relative attention) around the point's coarse cell.
    Fine neighborhoods are 5x5 raster subpatches; coarse neighborhoods are3x3.
    """
    size = np.asarray(image_size, np.float64)
    pts = np.asarray(points)
    _require(size.shape == (2,) and bool(np.isfinite(size).all()) and bool((size > 0).all()), 'Invalid image size')
    _require(pts.ndim == 2 and pts.shape[1:] == (2,), 'points must have shape [N,2]')
    _require(bool(np.isfinite(pts).all()) and bool((pts == np.rint(pts)).all()), 'Expected finite integer proposed points')
    _require(bool(((pts >= 0) & (pts < size)).all()), 'Proposed points must lie inside the image')
    _require((local_arrays is None) == (local_metadata is None), 'Local arrays/metadata must both be supplied or absent')
    views = [_raw_view(global_arrays, global_metadata)]
    boxes = [np.array([0., 0., size[0], size[1]])]
    if local_arrays is not None:
        _require(transform is not None and 'box' in transform, 'Local view requires original-pixel crop transform')
        box = np.asarray(transform['box'], np.float64)
        _require(box.shape == (4,) and bool(np.isfinite(box).all()) and bool((box[2:] > box[:2]).all()), 'Invalid crop box')
        _require(bool((box[:2] >= 0).all()) and bool((box[2:] <= size).all()), 'Crop must lie in the original image')
        views.append(_raw_view(local_arrays, local_metadata)); boxes.append(box)
    tmax = max(1, *(len(v[0]) for v in views))
    pmax = max(v[2].shape[0] for v in views)
    n = len(pts)
    output = dict(query_hidden=np.zeros((2, tmax, 12288), np.float16),
        query_mask=np.zeros((2, tmax), bool), image_hidden=np.zeros((2, pmax, 4096), np.float16),
        image_mask=np.zeros((2, pmax), bool), image_xy=np.zeros((2, pmax, 2), np.float32),
        vision_hidden=np.zeros((2, 4*pmax, 1152), np.float16), grid_shapes=np.ones((2, 2), np.int64),
        coarse_indices=np.zeros((n, 2, 9), np.int64), coarse_mask=np.zeros((n, 2, 9), bool),
        fine_indices=np.zeros((n, 2, 25), np.int64), fine_mask=np.zeros((n, 2, 25), bool),
        attention_field=np.zeros((n, 2, tmax, 96, FIELD_CHANNELS), np.float16),
        geometry=np.zeros((n, 2, 12), np.float32), view_mask=np.zeros((n, 2), bool),
        points=(pts/size).astype(np.float32))
    for view, ((a, q, image, vision, (gh, gw)), box) in enumerate(zip(views, boxes)):
        t, p = len(a), gh*gw
        output['query_hidden'][view, :t] = q
        output['query_mask'][view, :t] = True
        output['image_hidden'][view, :p] = image
        output['image_mask'][view, :p] = True
        output['vision_hidden'][view, :4*p] = vision
        output['grid_shapes'][view] = (gh, gw)
        rr, cc = np.divmod(np.arange(p), gw)
        output['image_xy'][view, :p] = np.stack(((cc+.5)/gw, (rr+.5)/gh), -1)
        inside = ((pts >= box[:2]) & (pts < box[2:])).all(-1) & (t > 0)
        local = (pts-box[:2])/(box[2:]-box[:2])
        safe_local = local.clip(0, np.nextafter(1., 0.))
        cell = np.floor(safe_local*np.array([gw, gh])).astype(np.int64)
        fine_cell = np.floor(safe_local*np.array([gw*2, gh*2])).astype(np.int64)
        coarse_index, coarse_valid = _neighborhood(cell, gh, gw, 1)
        fine_index, fine_valid = _neighborhood(fine_cell, gh*2, gw*2, 2)
        output['view_mask'][:, view] = inside
        output['coarse_indices'][:, view] = coarse_index
        output['coarse_mask'][:, view] = coarse_valid & inside[:, None]
        output['fine_indices'][:, view] = fine_index
        output['fine_mask'][:, view] = fine_valid & inside[:, None]
        geometry = np.concatenate((local, local*np.array([gw, gh])-cell,
            local*np.array([gw*2, gh*2])-fine_cell,
            np.broadcast_to([1/gw, 1/gh], (n, 2)),
            np.broadcast_to((box[2:]-box[:2])/size, (n, 2)), pts/size), -1)
        output['geometry'][:, view] = np.where(inside[:, None], geometry, 0).astype(np.float32)
        if not t or not n:
            continue
        mass = a.sum(-1)
        relative = a/np.maximum(mass[..., None], 1e-12)
        peak = relative.max(-1)
        ent = -(relative*np.log(np.maximum(relative, 1e-30))).sum(-1)/np.log(max(p, 2))
        parents = cell[:, 1]*gw+cell[:, 0]
        center = relative[..., parents].transpose(2, 0, 1)
        absolute = a[..., parents].transpose(2, 0, 1)
        shape = center.shape
        first = np.stack((np.log1p(p*center), center/np.maximum(peak[None], 1e-12),
            np.log1p(1e4*absolute), np.broadcast_to(np.log1p(100*mass), shape),
            np.broadcast_to(ent, shape), (parents[:, None, None] == a.argmax(-1)[None]).astype(np.float32)), -1)
        neighborhood = relative[..., coarse_index].transpose(2, 0, 1, 3)
        neighborhood = np.log1p(p*neighborhood)*coarse_valid[:, None, None]
        field = np.concatenate((first, neighborhood), -1)*inside[:, None, None, None]
        output['attention_field'][:, view, :t] = field.astype(np.float16)
    return output


def subset_internal_inputs(inputs, indices):
    """Explicit candidate-axis selection, including a permutation of a real pool."""
    _require(set(inputs) == INPUT_KEYS, 'Unexpected input keys; supervision is not accepted')
    index = np.asarray(indices)
    _require(index.ndim == 1 and np.issubdtype(index.dtype, np.integer), 'indices must be a 1D integer array')
    _require(len(set(index.tolist())) == len(index), 'Repeated candidates are not allowed')
    n = len(inputs['points'])
    _require(bool(((index >= 0) & (index < n)).all()), 'Candidate index out of range')
    return {name: value[index] if name in CANDIDATE_KEYS else value for name, value in inputs.items()}


def internal_inputs_to_torch(inputs, device='cpu'):
    _require(set(inputs) == INPUT_KEYS, 'Unexpected input keys; supervision is not accepted')
    return {name: torch.as_tensor(value, device=device) for name, value in inputs.items()}


def _token_reduce(values, valid):
    """[N,T,D] -> [N,2D], only after token/candidate interaction."""
    count = valid.sum().clamp_min(1)
    masked = values.masked_fill(~valid[None, :, None], -torch.inf)
    return torch.cat((masked.amax(1), torch.logsumexp(masked, 1)-count.float().log()), -1)


class InternalCandidateScorer(nn.Module):
    def __init__(self, config: InternalCandidateScorerConfig):
        super().__init__()
        self.config = config
        d, h = config.width, config.head_width
        sub = max(4, d//3)
        self.query_layers = nn.ModuleList([nn.Linear(4096, sub) for _ in range(3)])
        self.query = nn.Sequential(nn.Linear(sub*3, d), nn.GELU(), nn.LayerNorm(d))
        self.image = nn.Sequential(nn.Linear(4096, d), nn.GELU(), nn.LayerNorm(d))
        self.vision = nn.Sequential(nn.Linear(1152, d), nn.GELU(), nn.LayerNorm(d))
        self.token_position = nn.Linear(2, d, bias=False)
        self.global_position = nn.Linear(2, d, bias=False)
        self.local_position = nn.Linear(4, d, bias=False)
        self.geometry = nn.Sequential(nn.Linear(12, d), nn.GELU(), nn.LayerNorm(d))
        self.scale_embedding = nn.Embedding(2, d)
        self.view_embedding = nn.Embedding(2, d)
        self.local_attention = nn.MultiheadAttention(d, 4, dropout=config.dropout, batch_first=True)
        self.global_attention = nn.MultiheadAttention(d, 4, dropout=config.dropout, batch_first=True)
        self.token_context = nn.Sequential(nn.Linear(d*5, d), nn.GELU(), nn.LayerNorm(d), nn.Dropout(config.dropout))
        self.field = nn.Sequential(nn.Linear(FIELD_CHANNELS, h), nn.GELU(), nn.LayerNorm(h))
        self.head_embedding = nn.Embedding(96, h)
        self.query_condition = nn.Linear(d, h)
        self.visual_condition = nn.Linear(d, h)
        self.head_evidence = nn.Sequential(nn.Linear(h*3, h*2), nn.GELU(), nn.Linear(h*2, 1))
        self.head_readout = nn.Sequential(nn.Linear(96*2, d), nn.GELU(), nn.LayerNorm(d))
        self.token_readout = nn.Sequential(nn.Linear(d*2, d), nn.GELU(), nn.LayerNorm(d))
        self.view_fusion = nn.Sequential(nn.Linear(d*3, d), nn.GELU(), nn.LayerNorm(d))
        self.cross_view = nn.Sequential(nn.Linear(d*4+4, d), nn.GELU(), nn.LayerNorm(d), nn.Dropout(config.dropout))
        self.null_embedding = nn.Parameter(torch.zeros(d))
        self.null_context = nn.Linear(d*2+2, d)
        layer = nn.TransformerEncoderLayer(d, 4, dim_feedforward=d*2, dropout=config.dropout,
                                           activation='gelu', batch_first=True, norm_first=True)
        self.set_encoder = nn.TransformerEncoder(layer, num_layers=1, enable_nested_tensor=False)
        self.output_norm = nn.LayerNorm(d)
        self.score = nn.Linear(d, 1)

    def _query(self, raw, valid):
        layers = F.layer_norm(raw.float().reshape(-1, 3, 4096), (4096,))
        query = self.query(torch.cat([projection(layers[:, i]) for i, projection in enumerate(self.query_layers)], -1))
        count = valid.sum().clamp_min(1)
        position = torch.arange(len(raw), device=raw.device, dtype=torch.float32)/max(int(count)-1, 1)
        last = torch.arange(len(raw), device=raw.device) == count-1
        query = query+self.token_position(torch.stack((position, last.float()), -1))
        if self.config.head_mode == 'pooled_control':
            pooled = (query*valid[:, None]).sum(0, keepdim=True)/count
            query = pooled.expand_as(query)
        return query*valid[:, None]

    def _view(self, view, query, valid, raw_image, image_valid, image_xy, raw_vision, shape,
              coarse_index, coarse_valid, fine_index, fine_valid, field, geometry, active):
        n, t, d = len(geometry), len(query), self.config.width
        if not bool(valid.any()) or not bool(image_valid.any()) or not bool(active.any()):
            return geometry.new_zeros((n, d)), geometry.new_zeros((n, t, 96))
        image = self.image(F.layer_norm(raw_image.float(), (4096,)))
        vision = self.vision(F.layer_norm(raw_vision.float(), (1152,)))
        positional = self.global_position(image_xy.float())
        full = image+positional
        geo = self.geometry(geometry.float())+self.view_embedding.weight[view]
        q = query[None].expand(n, -1, -1)
        conditioned_query = q+geo[:, None]
        global_content = self.global_attention(conditioned_query,
            full[None].expand(n, -1, -1), image[None].expand(n, -1, -1),
            key_padding_mask=~image_valid[None].expand(n, -1), need_weights=False)[0]
        gh, gw = shape[0], shape[1]
        coarse_xy = image_xy[coarse_index]
        fine_row = torch.div(fine_index, gw*2, rounding_mode='floor')
        fine_col = fine_index % (gw*2)
        fine_xy = torch.stack(((fine_col.float()+.5)/(gw*2), (fine_row.float()+.5)/(gh*2)), -1)
        center = geometry[:, :2]
        scale = torch.stack((gw, gh)).float()
        coarse_pos = self.local_position(torch.cat(((coarse_xy-center[:, None])*scale, coarse_xy), -1))
        fine_pos = self.local_position(torch.cat(((fine_xy-center[:, None])*scale, fine_xy), -1))
        local = torch.cat((image[coarse_index]+coarse_pos+self.scale_embedding.weight[0],
                           vision[fine_index]+fine_pos+self.scale_embedding.weight[1]), 1)
        local_valid = torch.cat((coarse_valid, fine_valid), -1)
        # Outside-view candidates have no valid key. Use a temporary finite
        # dummy key, then mask their complete representation back to zero.
        safe_valid = local_valid.clone()
        safe_valid[:, 0] |= ~safe_valid.any(-1)
        local_content = self.local_attention(conditioned_query, local, local,
            key_padding_mask=~safe_valid, need_weights=False)[0]
        context = self.token_context(torch.cat((q, local_content, global_content,
                                                q*local_content, q*global_content), -1))
        raw_field = field.float()
        if self.config.head_mode == 'pooled_control':
            pooled = (raw_field*valid[None, :, None, None]).sum(1, keepdim=True)/valid.sum().clamp_min(1)
            raw_field = pooled.expand_as(raw_field)
        encoded = self.field(raw_field)+self.head_embedding.weight[None, None]
        query_condition = self.query_condition(q)[:, :, None].expand(-1, -1, 96, -1)
        visual_condition = self.visual_condition(context)[:, :, None].expand(-1, -1, 96, -1)
        evidence = self.head_evidence(torch.cat((encoded, encoded*query_condition,
                                                  encoded*visual_condition), -1)).squeeze(-1)
        # 96 head channels remain explicit. Pool only the token axis AFTER
        # candidate-conditioned query/content interactions, not before them.
        head_summary = self.head_readout(_token_reduce(evidence, valid))
        token_summary = self.token_readout(_token_reduce(context, valid))
        result = self.view_fusion(torch.cat((head_summary, token_summary, geo), -1))*active[:, None]
        return result, evidence*valid[None, :, None]*active[:, None, None]

    def forward(self, query_hidden, query_mask, image_hidden, image_mask, image_xy, vision_hidden,
                grid_shapes, coarse_indices, coarse_mask, fine_indices, fine_mask,
                attention_field, geometry, view_mask, points, return_diagnostics=False):
        n = len(points); d = self.config.width
        _require(points.ndim == 2 and points.shape[1] == 2, 'points must be [N,2] normalized original coordinates')
        _require(query_hidden.shape[:2] == query_mask.shape and query_hidden.shape[0] == 2 and query_hidden.shape[-1] == 12288,
                 'Query views/tokens misaligned')
        t = query_hidden.shape[1]
        _require(attention_field.shape == (n, 2, t, 96, FIELD_CHANNELS), 'Complete token/head field is required')
        _require(geometry.shape == (n, 2, 12) and view_mask.shape == (n, 2), 'Candidate-view geometry mismatch')
        _require(coarse_indices.shape == coarse_mask.shape == (n, 2, 9), 'Coarse neighborhood mismatch')
        _require(fine_indices.shape == fine_mask.shape == (n, 2, 25), 'Fine neighborhood mismatch')
        active = view_mask.bool().clone()
        if self.config.head_mode == 'no_local':
            active[:, 1] = False
        views, evidences, last_queries = [], [], []
        for view in range(2):
            valid = query_mask[view].bool()
            query = self._query(query_hidden[view], valid)
            last = query[valid][-1] if bool(valid.any()) else query.new_zeros(d)
            if view == 1 and (self.config.head_mode == 'no_local' or not bool(active[:, 1].any())):
                last = torch.zeros_like(last)
            last_queries.append(last)
            if n:
                encoded, evidence = self._view(view, query, valid,
                    image_hidden[view], image_mask[view].bool(), image_xy[view], vision_hidden[view], grid_shapes[view],
                    coarse_indices[:, view], coarse_mask[:, view].bool(), fine_indices[:, view], fine_mask[:, view].bool(),
                    attention_field[:, view], geometry[:, view], active[:, view])
            else:
                encoded, evidence = query.new_zeros((0, d)), query.new_zeros((0, t, 96))
            views.append(encoded); evidences.append(evidence)
        first, second = views
        candidate = self.cross_view(torch.cat((first, second, first*second, (first-second).abs(),
                                               active.float(), points.float()), -1))
        candidate_valid = active.any(-1)
        view_presence = query_mask.any(-1).float()
        if self.config.head_mode == 'no_local' or not bool(active[:, 1].any()):
            view_presence = view_presence.clone(); view_presence[1] = 0
        null = self.null_embedding+self.null_context(torch.cat((*last_queries, view_presence)))
        sequence = torch.cat((candidate, null[None]), 0)[None]
        invalid = torch.cat((~candidate_valid, torch.zeros(1, dtype=torch.bool, device=points.device)))[None]
        encoded = self.set_encoder(sequence, src_key_padding_mask=invalid)[0]
        encoded = self.output_norm(encoded)
        scores = self.score(encoded).squeeze(-1)
        output = dict(scores=scores[:n].masked_fill(~candidate_valid, -1e4), null_scores=scores[n:],
                      candidate_lengths=[n], candidate_embeddings=encoded[:n])
        if return_diagnostics:
            output.update(head_evidence=torch.stack(evidences, 1),
                          view_embeddings=torch.stack(views, 1), valid_views=active)
        return output


def save_internal_scorer(path: str | Path, model: InternalCandidateScorer, metadata: dict[str, Any]):
    # Saving never moves or mutates the live training model or its mode.
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    torch.save(dict(schema_version=SCHEMA_VERSION, config=asdict(model.config), state_dict=state,
                    metadata=metadata), path)


def load_internal_scorer(path: str | Path, device='cpu'):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    _require(saved.get('schema_version') == SCHEMA_VERSION, 'Unsupported internal scorer schema')
    model = InternalCandidateScorer(InternalCandidateScorerConfig(**saved['config']))
    model.load_state_dict(saved['state_dict'], strict=True)
    return model.to(device).eval(), saved['metadata']
