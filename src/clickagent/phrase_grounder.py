"""Late, position-specific name-token fusion with supervised spatial hierarchy.

Unlike conditioned_grounder, no token map is pooled before head scoring.
The local selector may use image content to select token evidence at a cell,
but image features cannot add an independent global spatial score. The fine
decoder is conditional on a parent cell and cannot replace its selection.
"""
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class PhraseConfig:
    image_dim: int = 4096
    vision_dim: int = 1152
    query_dim: int = 12288
    query_layers: int = 3
    heads: int = 96
    width: int = 32
    variant: str = 'local'

    def __post_init__(self):
        if self.variant not in ('first', 'uniform', 'global', 'local'):
            raise ValueError('unknown phrase fusion variant')
        if self.query_dim % self.query_layers:
            raise ValueError('query layer dimensions disagree')


def split_children(values):
    """[B,2H,2W] -> [B,H,W,4], child order TL,TR,BL,BR."""
    b, fh, fw = values.shape
    return values.reshape(b, fh//2, 2, fw//2, 2).permute(0, 1, 3, 2, 4).reshape(b, fh//2, fw//2, 4)


def join_children(values):
    b, gh, gw, four = values.shape
    if four != 4:
        raise ValueError('four children required')
    return values.reshape(b, gh, gw, 2, 2).permute(0, 1, 3, 2, 4).reshape(b, gh*2, gw*2)


class PhraseSpatialGrounder(nn.Module):
    def __init__(self, config, initial_head_weights=None):
        super().__init__(); self.config = c = config; d = c.width
        initial = np.zeros(c.heads, np.float32); initial[:min(32, c.heads)] = 1/min(32, c.heads)
        if initial_head_weights is not None:
            initial = np.asarray(initial_head_weights, np.float32)
            if initial.shape != (c.heads,):
                raise ValueError('head initializer shape mismatch')
        self.base_head_weights = nn.Parameter(torch.from_numpy(initial.copy()))
        self.query = nn.Sequential(nn.Linear(c.query_dim, d), nn.GELU(), nn.LayerNorm(d))
        self.mass = nn.Sequential(nn.Linear(c.heads, d), nn.GELU(), nn.LayerNorm(d))
        self.token_context = nn.Sequential(nn.Linear(d*3, d), nn.GELU(), nn.LayerNorm(d))
        self.head_delta = nn.Linear(d, c.heads)
        self.global_selector = nn.Linear(d, 1)
        self.map_embedding = nn.Linear(c.heads, d, bias=False)
        self.image = nn.Sequential(nn.Linear(c.image_dim, d), nn.GELU(), nn.LayerNorm(d))
        self.local_selector = nn.Linear(d, 1)
        self.vision = nn.Sequential(nn.Linear(c.vision_dim, d), nn.GELU(), nn.LayerNorm(d))
        self.fine = nn.Sequential(nn.Linear(d*3, d), nn.GELU(), nn.Linear(d, 3))
        for layer in (self.head_delta, self.global_selector, self.local_selector, self.fine[-1]):
            nn.init.zeros_(layer.weight); nn.init.zeros_(layer.bias)

    def forward(self, attention, query_hidden, image_hidden, vision_hidden, token_mask, grid_shape):
        c = self.config; gh, gw = map(int, grid_shape); b, t, h, p = attention.shape
        if (h, p) != (c.heads, gh*gw) or query_hidden.shape != (b, t, c.query_dim):
            raise ValueError('unaligned query/attention shapes')
        if image_hidden.shape != (b, p, c.image_dim) or vision_hidden.shape != (b, p*4, c.vision_dim):
            raise ValueError('unaligned spatial features')
        if not bool(token_mask.any(-1).all()):
            raise ValueError('empty target phrase')
        # Padded tokens are always a contiguous suffix produced by make_batch.
        lengths = token_mask.sum(-1)
        expected_mask = torch.arange(t, device=token_mask.device)[None] < lengths[:, None]
        if not torch.equal(token_mask, expected_mask):
            raise ValueError('token padding must be a suffix')
        q = F.layer_norm(query_hidden.float().reshape(b, t, c.query_layers, -1), [c.query_dim//c.query_layers]).reshape(b, t, c.query_dim)
        q = self.query(q)
        phrase = q[torch.arange(b, device=q.device), lengths-1]
        head_mass = attention.float().sum(-1)
        relative = attention.float()/head_mass[..., None].clamp_min(1e-12)
        maps = torch.log1p(p*relative)  # [B,T,H,P], still separate name tokens
        mass = self.mass(torch.log(head_mass.clamp_min(1e-8)))
        context = self.token_context(torch.cat((q, phrase[:, None].expand_as(q), mass), -1))
        head_weights = self.base_head_weights[None, None]+.1*self.head_delta(context).tanh()
        evidence = torch.einsum('bth,bthp->btp', head_weights, maps)
        global_logits = self.global_selector(context).squeeze(-1)
        if c.variant == 'first':
            token_weights = torch.zeros_like(evidence); token_weights[:, 0] = 1
        elif c.variant == 'uniform':
            token_weights = token_mask.to(evidence.dtype)[:, :, None].expand_as(evidence)/lengths[:, None, None]
        else:
            logits = global_logits[:, :, None].expand_as(evidence)
            if c.variant == 'local':
                image = self.image(F.layer_norm(image_hidden.float(), [c.image_dim]))
                local = torch.tanh(self.map_embedding(maps.permute(0, 1, 3, 2))+context[:, :, None]+image[:, None])
                logits = logits+self.local_selector(local).squeeze(-1)
            token_weights = logits.masked_fill(~token_mask[:, :, None], -torch.inf).softmax(1)
        coarse_logits = (token_weights*evidence).sum(1).reshape(b, gh, gw)
        # Complete phrase plus cell-specific token state supports local offsets.
        local_query = torch.einsum('btp,btd->bpd', token_weights, q)+phrase[:, None]
        fine_query = local_query.reshape(b, gh, gw, c.width).repeat_interleave(2, 1).repeat_interleave(2, 2).reshape(b, p*4, c.width)
        vision = self.vision(F.layer_norm(vision_hidden.float(), [c.vision_dim]))
        fine = self.fine(torch.cat((vision, fine_query, vision*fine_query), -1))
        child_logits = split_children(fine[..., 0].reshape(b, gh*2, gw*2))
        child_log_prob = child_logits.log_softmax(-1)
        coarse_log_prob = coarse_logits.flatten(1).log_softmax(-1).reshape(b, gh, gw)
        fine_log_prob = join_children(coarse_log_prob[..., None]+child_log_prob)
        offsets = fine[..., 1:].sigmoid().transpose(1, 2).reshape(b, 2, gh*2, gw*2)
        return {'coarse_logits': coarse_logits, 'child_log_prob': child_log_prob,
                'fine_log_prob': fine_log_prob, 'offsets': offsets, 'grid_shape': [gh*2, gw*2],
                'token_weights': token_weights, 'head_weights': head_weights, 'head_mass': head_mass}


def phrase_loss(output, weights, offsets):
    b, gh, gw = output['coarse_logits'].shape
    children = split_children(weights)
    coarse_weights = children.sum(-1)
    if not torch.allclose(coarse_weights.sum((-1, -2)), torch.ones_like(coarse_weights[:, 0, 0]), atol=1e-5):
        raise ValueError('labels must sum to one')
    coarse_log_prob = output['coarse_logits'].flatten(1).log_softmax(-1).reshape_as(coarse_weights)
    region = -(coarse_weights*coarse_log_prob).sum((-1, -2)).mean()
    child = -(children*output['child_log_prob']).sum((-1, -2, -3)).mean()
    offset = (F.smooth_l1_loss(output['offsets'], offsets, reduction='none').sum(1)*weights).sum((-1, -2)).mean()
    return region+child+2*offset, {'coarse_loss': float(region.detach()), 'child_loss': float(child.detach()), 'offset_loss': float(offset.detach())}


def decode_phrase(output, batch_index, image_size, top_k=5, center_only=False):
    """Choose parent by its marginal, then child within that same parent."""
    coarse = output['coarse_logits'][batch_index].detach().cpu().numpy()
    children = output['child_log_prob'][batch_index].detach().cpu().numpy()
    offsets = output['offsets'][batch_index].detach().cpu().numpy()
    gh, gw = coarse.shape; fh, fw = gh*2, gw*2; w, height = image_size
    probs = np.exp(coarse-coarse.max()); probs /= probs.sum()
    order = np.argsort(-coarse.ravel(), kind='stable'); selected = []; proposals = []
    for index in order:
        rr, cc = divmod(int(index), gw)
        if any(max(abs(rr-r), abs(cc-col)) <= 1 for r, col in selected):
            continue
        selected.append((rr, cc)); child = int(children[rr, cc].argmax())
        row, col = rr*2+child//2, cc*2+child % 2
        delta = np.full(2, .5) if center_only else offsets[:, row, col]
        point = [int(np.clip(np.rint((col+float(delta[0]))*w/fw), 0, w-1)),
                 int(np.clip(np.rint((row+float(delta[1]))*height/fh), 0, height-1))]
        proposals.append({'point': point, 'patch_index': row*fw+col, 'coarse_index': int(index),
                          'cell': [col/fw, row/fh, (col+1)/fw, (row+1)/fh],
                          'region': [cc/gw, rr/gh, (cc+1)/gw, (rr+1)/gh],
                          'offset': delta.tolist(), 'coarse_probability': float(probs[rr, cc])})
        if len(proposals) >= top_k:
            break
    return proposals


def save_phrase(path, model, metadata):
    torch.save({'config': asdict(model.config), 'state_dict': model.state_dict(), 'metadata': metadata}, path)


def load_phrase(path, device='cpu'):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    model = PhraseSpatialGrounder(PhraseConfig(**saved['config']))
    model.load_state_dict(saved['state_dict'], strict=True)
    return model.to(device).eval(), saved['metadata']
