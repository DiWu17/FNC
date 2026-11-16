"""Small instruction-region matching residual with no coordinate or ID input."""
import torch
from torch import nn
import torch.nn.functional as F


def shared_channel_dropout(x, query_ids, queries, probability, training):
    """One channel mask per query and view, shared across all its candidates."""
    if not training or probability == 0:
        return x
    keep = torch.empty((queries, x.shape[1], x.shape[2]), device=x.device).bernoulli_(1-probability)
    return x * keep[query_ids] / (1-probability)


class TransferMatcher(nn.Module):
    def __init__(self, regularized=False):
        super().__init__()
        self.region = nn.Linear(96, 32, bias=False)
        self.query = nn.Linear(32, 32, bias=False)
        # Begin near the existing shared-space semantic correspondence. Fine
        # vision and neighborhood channels can learn a shared linear mapping.
        with torch.no_grad():
            self.region.weight.normal_(0, .001)
            self.region.weight[:, 64:96] += torch.eye(32)
            self.query.weight.normal_(0, .001)
            self.query.weight += torch.eye(32)
        self.register_buffer('region_anchor', self.region.weight.detach().clone())
        self.register_buffer('query_anchor', self.query.weight.detach().clone())
        self.alpha = nn.Parameter(torch.zeros(()))
        self.regularized = regularized

    def encode(self, roi, query_ids=None, queries=None):
        # Raw fixed V19 projections, not its standardized product/query MLP.
        # No base/state features, absolute positions, IDs, labels or source tags.
        region, query = roi[..., :96].float(), roi[..., 96:128].float()
        if self.regularized and self.training:
            assert query_ids is not None and queries is not None
            region = shared_channel_dropout(region, query_ids, queries, .2, True)
            query = shared_channel_dropout(query, query_ids, queries, .2, True)
        return F.normalize(self.region(region), dim=-1), F.normalize(self.query(query), dim=-1)

    def forward(self, roi, roi_mask, query_ids=None, queries=None):
        region, query = self.encode(roi, query_ids, queries)
        match = (region*query).sum(-1)
        match = (match*roi_mask).sum(1)/roi_mask.sum(1).clamp_min(1)
        residual = 4*self.alpha.tanh()*match
        return residual, region, query

    def anchor_penalty(self):
        return ((self.region.weight-self.region_anchor).square().sum()
                +(self.query.weight-self.query_anchor).square().sum())/32


def admissible_pairs(groups, instructions, device):
    """Do not label another instruction on the same RGB, or an exact synonym,
    as a negative. Semantically similar nonidentical instructions can remain.
    """
    n = len(groups)
    return torch.tensor([[i == j or (groups[i] != groups[j] and instructions[i] != instructions[j])
                          for j in range(n)] for i in range(n)], dtype=torch.bool, device=device)


def pair_loss(region, query, lengths, labels, groups, instructions, temperature=.1):
    """Symmetric in-batch correspondence of global query and positive region.

    Only fitting labels form positive prototypes; label-free forward is separate.
    Query and image cached vectors were produced under their native instruction:
    this is partial feature supervision, not a re-encoded counterfactual task.
    """
    rr, qq, gg, ii = [], [], [], []
    offset = 0
    for n, y, group, instruction in zip(lengths, labels, groups, instructions):
        if n and bool(y.any()):
            rr.append(F.normalize(region[offset:offset+n, 0][y.bool()].mean(0), dim=0))
            qq.append(query[offset, 0])
            gg.append(group); ii.append(instruction)
        offset += n
    if len(rr) < 2:
        return (region.sum()+query.sum())*0, 0
    logits = torch.stack(qq) @ torch.stack(rr).T / temperature
    allowed = admissible_pairs(gg, ii, logits.device)
    eligible = allowed.sum(1) > 1
    if not bool(eligible.any()):
        return logits.sum()*0, 0
    logits = logits.masked_fill(~allowed, -torch.inf)
    diagonal = logits.diagonal()
    loss = .5*((torch.logsumexp(logits, 1)-diagonal)[eligible].mean()
               +(torch.logsumexp(logits, 0)-diagonal)[eligible].mean())
    return loss, int(eligible.sum())
