"""Source-preserving name-token/head candidates and a semantic region ranker."""
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def build_candidate_inputs(attention, query_hidden, image_hidden, vision_hidden, grid_shape):
    """No case, instruction label, target geometry, or dataset ID is accepted."""
    a = np.asarray(attention, np.float32); gh, gw = map(int, grid_shape)
    if a.ndim != 3 or a.shape[1:] != (96, gh*gw) or not len(a):
        raise ValueError('nonempty, aligned [T,96,H*W] own-token attention required')
    t, heads, p = a.shape
    if query_hidden.shape != (t, 12288) or image_hidden.shape != (p, 4096) or vision_hidden.shape != (4*p, 1152):
        raise ValueError('unaligned cached hidden states')
    mass = a.sum(-1); rel = a/np.maximum(mass[..., None], 1e-12)
    peaks = a.argmax(-1); parents, inverse = np.unique(peaks, return_inverse=True)
    inverse = inverse.reshape(t, heads); n = len(parents)
    top = np.partition(rel, -2, axis=-1)[..., -2:]
    gap = top.max(-1)-top.min(-1)
    entropy = -(rel*np.log(np.maximum(rel, 1e-30))).sum(-1)/np.log(p)
    counts = np.bincount(peaks.ravel(), minlength=p)
    head_count = np.stack([np.bincount(peaks[:, h], minlength=p) > 0 for h in range(96)]).sum(0)
    token_count = np.stack([np.bincount(row, minlength=p) > 0 for row in peaks]).sum(0)
    same_token = np.stack([np.bincount(row, minlength=p)[row]/96 for row in peaks])
    ti = np.broadcast_to(np.arange(t)[:, None], (t, 96))
    yy = ((peaks//gw)+.5)/gh; xx = ((peaks % gw)+.5)/gw
    values = [np.log(np.maximum(mass, 1e-8)), np.log(np.maximum(rel.max(-1), 1e-8)),
        np.log(np.maximum(gap, 1e-8)), entropy, np.log(np.maximum(a.max(-1), 1e-10)),
        ti/max(1, t-1), ti == 0, ti == t-1, xx, yy, same_token,
        counts[peaks]/(t*96), head_count[peaks]/96, token_count[peaks]/t]
    logmap = np.log1p(p*rel)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            row = peaks//gw+dy; col = peaks % gw+dx
            valid = (row >= 0) & (row < gh) & (col >= 0) & (col < gw)
            index = np.clip(row, 0, gh-1)*gw+np.clip(col, 0, gw-1)
            values.append(np.take_along_axis(logmap, index[..., None], -1)[..., 0]*valid)
    source_features = np.stack(values, -1).reshape(t*96, 23).astype(np.float32)
    pr, pc = np.divmod(parents, gw)
    candidate_features = np.stack((np.log1p(counts[parents]), head_count[parents]/96,
        token_count[parents]/t, (pc+.5)/gw, (pr+.5)/gh,
        ((pr == 0) | (pr == gh-1) | (pc == 0) | (pc == gw-1)).astype(float)), -1).astype(np.float32)
    children = np.asarray(vision_hidden).reshape(gh, 2, gw, 2, 1152).transpose(0, 2, 1, 3, 4).reshape(p, 4, 1152)
    return dict(query_hidden=np.asarray(query_hidden, np.float16),
        image_hidden=np.asarray(image_hidden[parents], np.float16),
        vision_hidden=np.asarray(children[parents], np.float16),
        source_features=source_features, candidate_features=candidate_features,
        source_token=np.repeat(np.arange(t), 96).astype(np.int64),
        source_head=np.tile(np.arange(96), t).astype(np.int64),
        source_candidate=inverse.ravel().astype(np.int64), parent_indices=parents.astype(np.int64))


def candidate_supervision(geometry, grid_shape, parents):
    """Geometry is supervision only, created after label-free proposals."""
    from .spatial_geometry import spatial_supervision
    gh, gw = map(int, grid_shape)
    weights, offsets = spatial_supervision(geometry, [gh*2, gw*2])
    children = weights.reshape(gh, 2, gw, 2).transpose(0, 2, 1, 3).reshape(gh*gw, 4)
    delta = offsets.transpose(1, 2, 0).reshape(gh, 2, gw, 2, 2).transpose(0, 2, 1, 3, 4).reshape(gh*gw, 4, 2)
    return dict(label_area=children[parents].astype(np.float32), label_offsets=delta[parents].astype(np.float32))


def subset_candidates(record, keep):
    """Training-only removal counterfactual, retaining real source provenance."""
    keep = np.asarray(keep, bool)
    if keep.shape != (len(record['parent_indices']),) or not keep.any():
        raise ValueError('must retain at least one real candidate')
    mapping = np.cumsum(keep)-1
    source_keep = keep[record['source_candidate']]
    result = dict(record)
    for name in ('image_hidden', 'vision_hidden', 'candidate_features', 'parent_indices', 'label_area', 'label_offsets'):
        result[name] = record[name][keep]
    for name in ('source_token', 'source_head', 'source_features'):
        result[name] = record[name][source_keep]
    result['source_candidate'] = mapping[record['source_candidate'][source_keep]]
    # Counts are measured in the original evidence trajectory; they are not
    # recomputed using labels or changed when a proposal list is filtered.
    return result


def make_ranker_batch(records, device, labels=True):
    arrays = {k: [] for k in ('query_hidden', 'image_hidden', 'vision_hidden', 'source_features',
        'candidate_features', 'source_token', 'source_head', 'source_candidate', 'source_case', 'candidate_case')}
    lengths = []; last = []; qcount = 0; ccount = 0
    for i, r in enumerate(records):
        t = len(r['query_hidden']); n = len(r['parent_indices']); s = len(r['source_head'])
        for name in ('query_hidden', 'image_hidden', 'vision_hidden', 'source_features', 'candidate_features', 'source_head'):
            arrays[name].append(r[name])
        arrays['source_token'].append(r['source_token']+qcount)
        arrays['source_candidate'].append(r['source_candidate']+ccount)
        arrays['source_case'].append(np.full(s, i, np.int64))
        arrays['candidate_case'].append(np.full(n, i, np.int64))
        last.append(qcount+t-1); lengths.append(n); qcount += t; ccount += n
    inputs = {k: torch.from_numpy(np.concatenate(v)).to(device) for k, v in arrays.items()}
    inputs['last_query'] = torch.tensor(last, device=device)
    inputs['candidate_lengths'] = lengths
    target = None
    if labels:
        target = {k: torch.from_numpy(np.concatenate([r[k] for r in records])).to(device) for k in ('label_area', 'label_offsets')}
    return inputs, target


@dataclass
class RankerConfig:
    width: int = 32
    variant: str = 'full'

    def __post_init__(self):
        if self.variant not in ('full', 'no_visual', 'no_query', 'stats_only'):
            raise ValueError('unknown ranker ablation')


def segment_softmax(values, indices, count):
    maximum = values.new_full((count,), -torch.inf).scatter_reduce(0, indices, values, reduce='amax', include_self=True)
    exp = (values-maximum[indices]).exp()
    denominator = values.new_zeros(count).index_add(0, indices, exp)
    return exp/denominator[indices].clamp_min(1e-12)


class SemanticCandidateRanker(nn.Module):
    def __init__(self, config, head_prior):
        super().__init__(); self.config = c = config; d = c.width
        self.register_buffer('head_prior', torch.as_tensor(head_prior, dtype=torch.float32))
        self.query = nn.Sequential(nn.Linear(12288, d), nn.GELU(), nn.LayerNorm(d))
        self.shared_semantic = nn.Linear(4096, d, bias=False)
        self.vision = nn.Sequential(nn.Linear(1152, d), nn.GELU(), nn.LayerNorm(d))
        self.head = nn.Embedding(96, d)
        self.stats = nn.Sequential(nn.Linear(26, d), nn.GELU(), nn.LayerNorm(d))
        self.region_stats = nn.Sequential(nn.Linear(6, d), nn.GELU(), nn.LayerNorm(d))
        self.source_context = nn.Sequential(nn.Linear(d*6, d), nn.GELU(), nn.LayerNorm(d))
        self.source_score = nn.Linear(d, 1)
        self.rank = nn.Sequential(nn.Linear(d*6, d*2), nn.GELU(), nn.Dropout(.1), nn.Linear(d*2, 1))
        self.null = nn.Sequential(nn.Linear(d*3, d), nn.GELU(), nn.Linear(d, 1))
        self.fine = nn.Sequential(nn.Linear(d*4, d), nn.GELU(), nn.Linear(d, 3))

    def forward(self, query_hidden, image_hidden, vision_hidden, source_features, candidate_features,
                source_token, source_head, source_candidate, source_case, candidate_case, last_query, candidate_lengths):
        c = self.config; d = c.width; n = len(image_hidden); b = len(candidate_lengths)
        use_query = c.variant not in ('no_query', 'stats_only')
        use_visual = c.variant not in ('no_visual', 'stats_only')
        qraw = F.layer_norm(query_hidden.float().reshape(-1, 3, 4096), [4096])
        q15 = qraw[:, 0]; iraw = F.layer_norm(image_hidden.float(), [4096])
        if use_query:
            q = self.query(qraw.flatten(1))+self.shared_semantic(q15)
        else:
            q = qraw.new_zeros((len(qraw), d))
        phrase = q[last_query]
        if use_visual:
            visual = self.vision(F.layer_norm(vision_hidden.float(), [1152]))
            image = self.shared_semantic(iraw)+visual.mean(1)
        else:
            visual = iraw.new_zeros((n, 4, d)); image = iraw.new_zeros((n, d))
        local_q = q[source_token]; local_phrase = phrase[source_case]; local_image = image[source_candidate]
        if use_query and use_visual:
            raw_cosine = F.cosine_similarity(q15[source_token], iraw[source_candidate], -1)
            phrase_cosine = F.cosine_similarity(q15[last_query][source_case], iraw[source_candidate], -1)
        else:
            raw_cosine = source_features.new_zeros(len(source_head)); phrase_cosine = raw_cosine
        prior = self.head_prior[source_head]
        stats = self.stats(torch.cat((source_features.float(), prior[:, None], raw_cosine[:, None], phrase_cosine[:, None]), -1))
        source = self.source_context(torch.cat((local_q, local_phrase, self.head(source_head), stats,
                                                local_image, local_q*local_image), -1))
        alpha = segment_softmax(self.source_score(source).squeeze(-1), source_candidate, n)
        pooled = source.new_zeros((n, d)).index_add(0, source_candidate, source*alpha[:, None])
        maximum = source.new_full((n, d), -torch.inf).scatter_reduce(0, source_candidate[:, None].expand(-1, d), source, reduce='amax', include_self=True)
        local_query = q.new_zeros((n, d)).index_add(0, source_candidate, local_q*alpha[:, None])
        cp = phrase[candidate_case]
        features = torch.cat((pooled, maximum, image, cp, image*cp, self.region_stats(candidate_features.float())), -1)
        scores = self.rank(features).squeeze(-1)
        mean_case = source.new_zeros((b, d)).index_add(0, candidate_case, pooled)
        mean_case = mean_case/torch.tensor(candidate_lengths, device=source.device)[:, None]
        max_case = pooled.new_full((b, d), -torch.inf).scatter_reduce(0, candidate_case[:, None].expand(-1, d), pooled, reduce='amax', include_self=True)
        null_scores = self.null(torch.cat((phrase, mean_case, max_case), -1)).squeeze(-1)
        fine_context = torch.cat((visual, (local_query+cp)[:, None].expand(-1, 4, -1),
            pooled[:, None].expand(-1, 4, -1), visual*(local_query+cp)[:, None]), -1)
        fine = self.fine(fine_context)
        return dict(scores=scores, null_scores=null_scores, child_logits=fine[..., 0], offsets=fine[..., 1:].sigmoid(),
                    source_weights=alpha, candidate_lengths=candidate_lengths)


def ranker_loss(output, label_area, label_offsets):
    region = []; cursor = 0
    for i, n in enumerate(output['candidate_lengths']):
        area = label_area[cursor:cursor+n].sum(-1)
        logp = torch.cat((output['scores'][cursor:cursor+n], output['null_scores'][i:i+1])).log_softmax(0)
        if bool(area.sum() > 0):
            region.append(-(logp[:-1]*area/area.sum()).sum())
        else:
            region.append(-logp[-1])
        cursor += n
    region = torch.stack(region).mean()
    mass = label_area.sum()
    if bool(mass > 0):
        child = -(label_area*output['child_logits'].log_softmax(-1)).sum()/mass
        offset = (F.smooth_l1_loss(output['offsets'], label_offsets, reduction='none').sum(-1)*label_area).sum()/mass
    else:
        child = output['child_logits'].sum()*0; offset = output['offsets'].sum()*0
    loss = region+.5*child+offset
    return loss, dict(region=float(region.detach()), child=float(child.detach()), offset=float(offset.detach()))


def decode_ranker(output, record, batch_index=0, mode='fine', top_k=5):
    cursor = sum(output['candidate_lengths'][:batch_index]); n = output['candidate_lengths'][batch_index]
    scores = output['scores'][cursor:cursor+n].detach().cpu().numpy()
    children = output['child_logits'][cursor:cursor+n].detach().cpu().numpy()
    offsets = output['offsets'][cursor:cursor+n].detach().cpu().numpy()
    gh, gw = record['metadata']['grid_shape']; w, h = record['case']['image_size']
    result = []
    for ci in np.argsort(-scores, kind='stable')[:top_k]:
        parent = int(record['parent_indices'][ci]); rr, cc = divmod(parent, gw)
        if mode == 'center':
            xy = [(cc+.5)/gw, (rr+.5)/gh]
        else:
            child = int(children[ci].argmax()); row, col = rr*2+child//2, cc*2+child % 2
            dx, dy = offsets[ci, child]; xy = [(col+float(dx))/(2*gw), (row+float(dy))/(2*gh)]
        point = [int(np.clip(np.rint(xy[0]*w), 0, w-1)), int(np.clip(np.rint(xy[1]*h), 0, h-1))]
        result.append(dict(point=point, candidate_index=int(ci), parent_index=parent, score=float(scores[ci]),
                           region=[cc/gw, rr/gh, (cc+1)/gw, (rr+1)/gh]))
    return result


def save_ranker(path, model, metadata):
    torch.save(dict(config=asdict(model.config), state_dict=model.state_dict(), metadata=metadata), path)


def load_ranker(path, device='cpu'):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    model = SemanticCandidateRanker(RankerConfig(**saved['config']), saved['state_dict']['head_prior'])
    model.load_state_dict(saved['state_dict'], strict=True)
    return model.to(device).eval(), saved['metadata']
