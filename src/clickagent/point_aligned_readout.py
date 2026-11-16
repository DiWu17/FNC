"""Point-aligned ranking and connected spatial refinement over frozen VLM caches.

The Qwen backbone is never part of this module. Ground truth is loss-only.
All four controls have identical parameters and matching warm starts.
"""
from dataclasses import asdict

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .candidate_ranker import ranker_loss
from .semantic_head_ranker import SemanticHeadRanker, SemanticHeadRankerConfig


class PointAlignedRanker(SemanticHeadRanker):
    def __init__(self, config, head_prior, connect_fine=False):
        super().__init__(config, head_prior)
        self.connect_fine = bool(connect_fine)
        d = config.width
        # Fork RNG so adding the adapter does not perturb other initializations.
        with torch.random.fork_rng(devices=[]):
            self.spatial_fine = nn.Sequential(nn.Linear(4*d, 2*d), nn.GELU(), nn.Linear(2*d, 3))
            nn.init.zeros_(self.spatial_fine[-1].weight)
            nn.init.zeros_(self.spatial_fine[-1].bias)

    def _field_residual(self, attention_field, field_token, field_candidate, base, return_routing):
        # Capture the actual pooled field tensor without reimplementing v7.
        captured = []
        handle = self.field_projection.register_forward_hook(lambda m, args, output: captured.append(output))
        try:
            result = super()._field_residual(attention_field, field_token, field_candidate, base, return_routing)
        finally:
            handle.remove()
        assert len(captured) == 1
        result['routed_spatial_features'] = captured[0]
        return result

    def forward(self, attention_field, field_token, field_candidate, **base):
        out = super().forward(attention_field, field_token, field_candidate, **base)
        field = out['routed_spatial_features']
        if not self.connect_fine:
            field = field.detach()
        qraw = F.layer_norm(base['query_hidden'].float().reshape(-1, 3, 4096), [4096])
        query = self.query(qraw.flatten(1))+self.shared_semantic(qraw[:, 0])
        phrase = query[base['last_query']][base['candidate_case']]
        vision = self.vision(F.layer_norm(base['vision_hidden'].float(), [1152]))
        f = field[:, None].expand(-1, 4, -1)
        q = phrase[:, None].expand(-1, 4, -1)
        delta = self.spatial_fine(torch.cat((vision, f, q, f*vision), -1))
        out['base_child_logits'] = out['child_logits']
        out['base_offsets'] = out['offsets']
        out['child_logits'] = out['child_logits']+delta[..., 0]
        base_logit = torch.logit(out['offsets'].clamp(1e-6, 1-1e-6))
        out['offsets'] = (base_logit+delta[..., 1:]).sigmoid()
        return out


def load_warm_start(path, connect_fine, device='cpu'):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    assert saved['architecture'] == 'semantic'
    state = {k.removeprefix('core.'): v for k, v in saved['state_dict'].items() if k.startswith('core.')}
    config = SemanticHeadRankerConfig(width=saved['width'], head_width=16, groups=4, routing='semantic')
    model = PointAlignedRanker(config, state['head_prior'], connect_fine)
    missing = model.load_state_dict(state, strict=False)
    assert not missing.unexpected_keys and all(k.startswith('spatial_fine.') for k in missing.missing_keys)
    return model.to(device)


def save_aligned(path, model, metadata):
    torch.save(dict(schema_version=1, config=asdict(model.semantic_config), connect_fine=model.connect_fine,
                    state_dict=model.state_dict(), metadata=metadata), path)


def load_aligned(path, device='cpu'):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    assert saved['schema_version'] == 1
    model = PointAlignedRanker(SemanticHeadRankerConfig(**saved['config']), saved['state_dict']['head_prior'], saved['connect_fine'])
    model.load_state_dict(saved['state_dict'], strict=True)
    return model.to(device).eval(), saved['metadata']


def multi_positive_point_loss(output, point_hit):
    """Normalize over candidate points and null, rewarding all valid points.

    Null is the target if this current candidate-point set has no hit. This
    does not mean the screenshot lacks the object; evaluation keeps both apart.
    """
    losses = []; cursor = 0
    for i, n in enumerate(output['candidate_lengths']):
        logits = torch.cat((output['scores'][cursor:cursor+n], output['null_scores'][i:i+1]))
        hit = point_hit[cursor:cursor+n].detach().bool()
        if bool(hit.any()):
            losses.append(torch.logsumexp(logits, 0)-torch.logsumexp(logits[:-1][hit], 0))
        else:
            losses.append(torch.logsumexp(logits, 0)-logits[-1])
        cursor += n
    assert cursor == len(point_hit)
    return torch.stack(losses).mean()


def all_candidate_points(output, records):
    """Canonical CPU integer decoding for every candidate, preserving order."""
    logits = output['child_logits'].detach().float().cpu().numpy()
    offsets = output['offsets'].detach().float().cpu().numpy()
    cursor = 0; results = []
    for r in records:
        n = len(r['parent_indices']); gh, gw = r['metadata']['grid_shape']; w, h = r['case']['image_size']
        child = logits[cursor:cursor+n].argmax(-1)
        delta = offsets[cursor:cursor+n][np.arange(n), child]
        row, col = np.divmod(r['parent_indices'], gw)
        xy = np.stack(((2*col+child%2+delta[:, 0])/(2*gw),
                       (2*row+child//2+delta[:, 1])/(2*gh)), -1)
        points = np.clip(np.rint(xy*np.asarray([w, h])), 0, np.asarray([w-1, h-1])).astype(np.int64)
        results.append(points); cursor += n
    return results


def candidate_point_hits(points, records):
    import shapely
    from .spatial_geometry import shape_from_geometry
    return [np.asarray(shapely.covers(shape_from_geometry(r['case']['target_geometry']),
        shapely.points(p/np.asarray(r['case']['image_size']))), bool) for p, r in zip(points, records)]


def aligned_loss(output, targets, point_hit, point_objective, region_weight=.25):
    old, pieces = ranker_loss(output, targets['label_area'], targets['label_offsets'])
    # Recompute the differentiable region component to separate its weight;
    # the existing child+offset objective is exactly shared by all controls.
    regions = []; cursor = 0
    for i, n in enumerate(output['candidate_lengths']):
        area = targets['label_area'][cursor:cursor+n].sum(-1)
        logp = torch.cat((output['scores'][cursor:cursor+n], output['null_scores'][i:i+1])).log_softmax(0)
        regions.append(-(logp[:-1]*area/area.sum()).sum() if bool(area.sum() > 0) else -logp[-1])
        cursor += n
    region = torch.stack(regions).mean()
    point = multi_positive_point_loss(output, point_hit)
    loss = old if not point_objective else old-region+point+region_weight*region
    return loss, dict(region=float(region.detach()), point=float(point.detach()), **{k:v for k,v in pieces.items() if k!='region'})
