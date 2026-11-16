"""V13: shared V11 readout over one global and up to three local views.

No additional parameters: candidate-conditioned local representations are max
pooled over containing views; null context averages present local queries. The
two-view path delegates to the sealed implementation, enabling exact controls.
"""
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .internal_candidate_scorer import (
    CANDIDATE_KEYS, InternalCandidateScorer, InternalCandidateScorerConfig,
    build_internal_inputs,
)


def build_multiregion_inputs(global_arrays, global_metadata, local_views, points, image_size):
    """local_views is [(four raw arrays, metadata, transform), ...]; no labels."""
    if len(local_views) > 3:
        raise ValueError('At most three local views')
    if len(local_views) <= 1:
        a, m, tr = local_views[0] if local_views else (None, None, None)
        return build_internal_inputs(global_arrays, global_metadata, a, m, points, image_size, tr)
    pairs = [build_internal_inputs(global_arrays, global_metadata, a, m, points, image_size, tr)
             for a, m, tr in local_views]
    views = [(pairs[0], 0)] + [(pair, 1) for pair in pairs]
    result = {}
    for name in pairs[0]:
        if name == 'points':
            result[name] = pairs[0][name]
            continue
        axis = 1 if name in CANDIDATE_KEYS else 0
        slices = [np.take(pair[name], [view], axis=axis) for pair, view in views]
        shape = list(slices[0].shape)
        for value in slices:
            shape = [max(x, y) for x, y in zip(shape, value.shape)]
        shape[axis] = len(views)
        merged = np.zeros(shape, dtype=slices[0].dtype)
        for i, value in enumerate(slices):
            where = [slice(0, s) for s in value.shape]
            where[axis] = slice(i, i+1)
            merged[tuple(where)] = value
        result[name] = merged
    return result


class MultiRegionScorer(InternalCandidateScorer):
    def forward(self, query_hidden, query_mask, image_hidden, image_mask, image_xy, vision_hidden,
                grid_shapes, coarse_indices, coarse_mask, fine_indices, fine_mask,
                attention_field, geometry, view_mask, points, return_diagnostics=False):
        args = dict(query_hidden=query_hidden, query_mask=query_mask, image_hidden=image_hidden,
            image_mask=image_mask, image_xy=image_xy, vision_hidden=vision_hidden,
            grid_shapes=grid_shapes, coarse_indices=coarse_indices, coarse_mask=coarse_mask,
            fine_indices=fine_indices, fine_mask=fine_mask, attention_field=attention_field,
            geometry=geometry, view_mask=view_mask, points=points)
        v = query_hidden.shape[0]
        if v == 2:
            return super().forward(**args, return_diagnostics=return_diagnostics)
        if not 3 <= v <= 4 or query_mask.shape != query_hidden.shape[:2]:
            raise ValueError('Expected 2..4 aligned views')
        n, d = len(points), self.config.width
        t = query_hidden.shape[1]
        if attention_field.shape != (n, v, t, 96, 15) or view_mask.shape != (n, v):
            raise ValueError('Candidate/view/token/head mismatch')
        active = view_mask.bool().clone()
        if self.config.head_mode == 'no_local':
            active[:, 1:] = False
        encoded_views, last_queries, evidence_views = [], [], []
        present = []
        for j in range(v):
            valid = query_mask[j].bool()
            query = self._query(query_hidden[j], valid)
            is_present = bool(valid.any()) and (j == 0 or bool(active[:, j].any()))
            last = query[valid][-1] if is_present else query.new_zeros(d)
            last_queries.append(last); present.append(is_present)
            if n:
                emb, ev = self._view(min(j, 1), query, valid, image_hidden[j], image_mask[j].bool(),
                    image_xy[j], vision_hidden[j], grid_shapes[j], coarse_indices[:, j],
                    coarse_mask[:, j].bool(), fine_indices[:, j], fine_mask[:, j].bool(),
                    attention_field[:, j], geometry[:, j], active[:, j])
            else:
                emb, ev = query.new_zeros((0, d)), query.new_zeros((0, t, 96))
            encoded_views.append(emb); evidence_views.append(ev)
        first = encoded_views[0]
        local_active = active[:, 1:].any(-1)
        locals_ = torch.stack(encoded_views[1:], 1).masked_fill(~active[:, 1:, None], -torch.inf)
        second = torch.where(local_active[:, None], locals_.amax(1), torch.zeros_like(first))
        combined_active = torch.stack((active[:, 0], local_active), -1)
        candidate = self.cross_view(torch.cat((first, second, first*second, (first-second).abs(),
                                               combined_active.float(), points.float()), -1))
        local_count = max(sum(present[1:]), 1)
        last_local = torch.stack(last_queries[1:]).sum(0)/local_count
        presence = points.new_tensor([present[0], any(present[1:])], dtype=torch.float32)
        null = self.null_embedding+self.null_context(torch.cat((last_queries[0], last_local, presence)))
        candidate_valid = combined_active.any(-1)
        sequence = torch.cat((candidate, null[None]), 0)[None]
        invalid = torch.cat((~candidate_valid, torch.zeros(1, dtype=torch.bool, device=points.device)))[None]
        encoded = self.output_norm(self.set_encoder(sequence, src_key_padding_mask=invalid)[0])
        scores = self.score(encoded).squeeze(-1)
        output = dict(scores=scores[:n].masked_fill(~candidate_valid, -1e4), null_scores=scores[n:],
                      candidate_lengths=[n], candidate_embeddings=encoded[:n])
        if return_diagnostics:
            output.update(head_evidence=torch.stack(evidence_views, 1),
                          view_embeddings=torch.stack(encoded_views, 1), valid_views=active)
        return output


def top4_retention_loss(output, labels, margin=1.):
    """Best positive must outrank the fourth negative; no GUI teacher.

    With fewer than four negatives, Top4 necessarily contains a positive.
    Null is outside the deployed shortlist, so it is not a rank competitor.
    Unreachable cases retain the original null/point-BCE objective.
    """
    from torch.nn import functional as functional
    scores = output['scores'].float()
    positive = labels > .5
    element = functional.binary_cross_entropy_with_logits(scores, labels.float(), reduction='none')
    if positive.any() and (~positive).any():
        bce = .5*element[positive].mean()+.5*element[~positive].mean()
    else:
        bce = element.mean()
    if not positive.any():
        null = output['null_scores'].float().reshape(1)
        rank = torch.logsumexp(torch.cat((scores, null)), 0)-null[0]
    elif int((~positive).sum()) >= 4:
        fourth = torch.topk(scores[~positive], 4).values[-1]
        rank = F.relu(scores.new_tensor(margin)+fourth-scores[positive].max())
    else:
        rank = scores.sum()*0
    return rank+.25*bce


def save_multiregion_scorer(path, model, metadata):
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    torch.save(dict(schema_version=13, config=asdict(model.config), state_dict=state, metadata=metadata), path)


def load_multiregion_scorer(path: str | Path, device='cpu'):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved.get('schema_version') != 13:
        raise ValueError('Expected V13 model')
    model = MultiRegionScorer(InternalCandidateScorerConfig(**saved['config']))
    model.load_state_dict(saved['state_dict'], strict=True)
    return model.to(device).eval(), saved['metadata']
