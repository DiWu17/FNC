"""Candidate attention spectra: retain every head's value before learned pooling.

The v5 source-preserving ranker is imported unchanged.  The two v6 conditions
have identical parameters and inputs; only the non-proposer field mask differs.
"""
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .candidate_ranker import (RankerConfig, SemanticCandidateRanker,
    make_ranker_batch, segment_softmax, subset_candidates)


def build_attention_field(attention, parents):
    """Label-free [candidate, token, head, feature] values at fixed parents.

    Head order is never averaged away.  Channels are log relative attention,
    relative-to-maximum attention, log absolute attention, total image mass,
    normalized spatial entropy and an own-argmax flag.  All use image keys only.
    """
    a = np.asarray(attention, np.float32)
    parents = np.asarray(parents, np.int64)
    if a.ndim != 3 or a.shape[1] != 96 or not len(a) or a.shape[2] < 2:
        raise ValueError('Expected nonempty own-token [T,96,P] attention')
    if not np.isfinite(a).all() or (a < 0).any():
        raise ValueError('Attention must be finite and nonnegative')
    if parents.ndim != 1 or not len(parents) or (parents < 0).any() or (parents >= a.shape[2]).any():
        raise ValueError('Invalid candidate parents')
    p = a.shape[2]; mass = a.sum(-1); rel = a/np.maximum(mass[..., None], 1e-12)
    peak = rel.max(-1); peaks = a.argmax(-1)
    values = a[..., parents].transpose(2, 0, 1)
    relative = rel[..., parents].transpose(2, 0, 1)
    entropy = -(rel*np.log(np.maximum(rel, 1e-30))).sum(-1)/np.log(p)
    shape = relative.shape
    channels = (np.log1p(p*relative), relative/np.maximum(peak, 1e-12)[None],
        np.log1p(1e4*values), np.broadcast_to(np.log1p(100*mass), shape),
        np.broadcast_to(entropy, shape), (parents[:, None, None] == peaks[None]).astype(np.float32))
    return np.stack(channels, -1).astype(np.float16)


def subset_field_candidates(record, keep):
    result = subset_candidates(record, keep)
    result['attention_field'] = record['attention_field'][keep]
    return result


def make_field_batch(records, device, labels=True):
    inputs, target = make_ranker_batch(records, device, labels)
    values = []; token = []; candidate = []; qcursor = 0; ccursor = 0
    for r in records:
        n = len(r['parent_indices']); t = len(r['query_hidden'])
        field = r['attention_field']
        if field.shape != (n, t, 96, 6):
            raise ValueError('Misaligned candidate attention field')
        values.append(field.reshape(n*t, 96, 6))
        token.append(np.tile(np.arange(t), n)+qcursor)
        candidate.append(np.repeat(np.arange(n), t)+ccursor)
        qcursor += t; ccursor += n
    for name, pieces in [('attention_field', values), ('field_token', token), ('field_candidate', candidate)]:
        inputs[name] = torch.from_numpy(np.concatenate(pieces)).to(device)
    return inputs, target


@dataclass
class FieldRankerConfig:
    width: int = 32
    field_mode: str = 'all'

    def __post_init__(self):
        if self.field_mode not in ('all', 'peaks_only'):
            raise ValueError('Unknown attention-field condition')


class AttentionFieldRanker(SemanticCandidateRanker):
    def __init__(self, config, head_prior):
        # Same original architecture in both conditions; do not change v5 code.
        super().__init__(RankerConfig(config.width, 'full'), head_prior)
        self.field_config = config; d = config.width
        self.field_projection = nn.Sequential(nn.Linear(96*6, d), nn.GELU(), nn.LayerNorm(d))
        self.field_context = nn.Sequential(nn.Linear(d*6, d), nn.GELU(), nn.LayerNorm(d))
        self.field_token_score = nn.Linear(d, 1)
        self.field_rank = nn.Sequential(nn.Linear(d*4, d*2), nn.GELU(), nn.Dropout(.1), nn.Linear(d*2, 1))
        # Start with exactly the v5 base scores for both matched conditions.
        nn.init.zeros_(self.field_rank[-1].weight); nn.init.zeros_(self.field_rank[-1].bias)

    def forward(self, attention_field, field_token, field_candidate, **base):
        output = super().forward(**base)
        n = len(base['image_hidden']); d = self.field_config.width
        field = attention_field.float()
        if self.field_config.field_mode == 'peaks_only':
            field = field*field[..., 5:6]
        encoded = self.field_projection(field.flatten(1))
        qraw = F.layer_norm(base['query_hidden'].float().reshape(-1, 3, 4096), [4096])
        query = self.query(qraw.flatten(1))+self.shared_semantic(qraw[:, 0])
        vision = self.vision(F.layer_norm(base['vision_hidden'].float(), [1152]))
        image = self.shared_semantic(F.layer_norm(base['image_hidden'].float(), [4096]))+vision.mean(1)
        phrase = query[base['last_query']][base['candidate_case']]
        local_query = query[field_token]; local_image = image[field_candidate]
        context = self.field_context(torch.cat((encoded, local_query, local_image,
            encoded*local_query, encoded*local_image, phrase[field_candidate]), -1))
        weights = segment_softmax(self.field_token_score(context).squeeze(-1), field_candidate, n)
        pooled = context.new_zeros((n, d)).index_add(0, field_candidate, context*weights[:, None])
        maximum = context.new_full((n, d), -torch.inf).scatter_reduce(0,
            field_candidate[:, None].expand(-1, d), context, reduce='amax', include_self=True)
        residual = self.field_rank(torch.cat((pooled, maximum, image, phrase), -1)).squeeze(-1)
        output['base_scores'] = output['scores']
        output['scores'] = output['scores']+residual
        output['field_residual'] = residual
        output['field_token_weights'] = weights
        return output


def save_field_ranker(path, model, metadata):
    torch.save(dict(config=asdict(model.field_config), state_dict=model.state_dict(), metadata=metadata), path)


def load_field_ranker(path, device='cpu'):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    model = AttentionFieldRanker(FieldRankerConfig(**saved['config']), saved['state_dict']['head_prior'])
    model.load_state_dict(saved['state_dict'], strict=True)
    return model.to(device).eval(), saved['metadata']
