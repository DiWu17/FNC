"""Detached prediction of a frozen decoder's point success for each candidate.

The quality branch is a ranking signal, not a calibrated action-permission gate.
Target geometry and fixed-point hit labels belong only in ``quality_loss``.
The ranker's ordinary forward and gradients are preserved by construction.
"""
from contextlib import nullcontext

import torch
from torch import nn
import torch.nn.functional as F


FROZEN_POINT_FEATURE_DIM = 13
FROZEN_POINT_FEATURE_NAMES = (
    'child_p0', 'child_p1', 'child_p2', 'child_p3',
    'selected_offset_x', 'selected_offset_y', 'child_entropy_normalized',
    'point_x_normalized', 'point_y_normalized',
    'point_x_in_parent', 'point_y_in_parent', 'parent_width', 'parent_height',
)


def _lengths(candidate_lengths, count):
    if isinstance(candidate_lengths, torch.Tensor):
        candidate_lengths = candidate_lengths.detach().cpu().tolist()
    result = []
    for value in candidate_lengths:
        if isinstance(value, bool) or int(value) != value or int(value) <= 0:
            raise ValueError('Each case must have a positive integer candidate count')
        result.append(int(value))
    if not result or sum(result) != count:
        raise ValueError('Candidate lengths must partition every candidate exactly once')
    return result


def _projection_width(core):
    for name in ('config', 'field_config', 'semantic_config'):
        config = getattr(core, name, None)
        if config is not None and hasattr(config, 'width'):
            return int(config.width)
    for layer in core.query.modules():
        if isinstance(layer, nn.Linear):
            return layer.out_features
    raise ValueError('The core must expose its query projection width')


class PointQualityReadout(nn.Module):
    """Add an isolated per-candidate fixed-point quality MLP to an existing core.

    ``frozen_point_features`` has 13 columns in ``FROZEN_POINT_FEATURE_NAMES``
    order. They must be calculated from the frozen fine decoder and screenshot
    grid alone. In particular, they may not include a target box or hit label.

    Initialization preserves the global CPU RNG state. The quality branch has
    no dropout, and its inputs and reused semantic projections are detached, so
    adding its BCE does not alter the core ranker's training trajectory.
    """
    def __init__(self, core, quality_width=32):
        super().__init__()
        if int(quality_width) != quality_width or quality_width <= 0:
            raise ValueError('quality_width must be a positive integer')
        self.core = core
        self.quality_width = int(quality_width)
        self.semantic_width = _projection_width(core)
        # Mean target query, final target query, candidate image and two
        # query-image products, plus frozen geometry and within-case scores.
        feature_width = self.semantic_width*5+FROZEN_POINT_FEATURE_DIM+3
        with torch.random.fork_rng(devices=[]):
            self.quality = nn.Sequential(
                nn.Linear(feature_width, self.quality_width), nn.GELU(),
                nn.LayerNorm(self.quality_width),
                nn.Linear(self.quality_width, self.quality_width), nn.GELU(),
                nn.Linear(self.quality_width, 1),
            )

    def _detached_features(self, inputs, output, frozen_point_features, lengths):
        # These modules are shared only to reuse the representation. Running
        # them under no_grad keeps BCE out of the ranker's computation graph.
        # fork_rng also protects the core's RNG if a future projection adds a
        # stochastic layer. Current query/image projections have no dropout.
        device = inputs['query_hidden'].device
        cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
        rng_context = torch.random.fork_rng(devices=cuda_devices) if device.type in ('cpu', 'cuda') else nullcontext()
        with torch.no_grad(), rng_context:
            qraw = F.layer_norm(inputs['query_hidden'].detach().float().reshape(-1, 3, 4096), [4096])
            query = self.core.query(qraw.flatten(1))+self.core.shared_semantic(qraw[:, 0])
            vision = self.core.vision(F.layer_norm(inputs['vision_hidden'].detach().float(), [1152]))
            image = self.core.shared_semantic(F.layer_norm(inputs['image_hidden'].detach().float(), [4096]))+vision.mean(1)
            last = inputs['last_query'].detach().long()
            if len(last) != len(lengths) or not bool((last >= 0).all()) or int(last[-1]) != len(query)-1:
                raise ValueError('last_query must identify the final query token of every case')
            ends = last.cpu().tolist()
            if any(end <= previous for previous, end in zip([-1]+ends[:-1], ends)):
                raise ValueError('Query token groups must be nonempty and ordered')
            starts = [0]+[end+1 for end in ends[:-1]]
            mean_query = torch.stack([query[start:end+1].mean(0) for start, end in zip(starts, ends)])
            expected_case = torch.repeat_interleave(torch.arange(len(lengths), device=device), torch.tensor(lengths, device=device))
            candidate_case = inputs['candidate_case'].detach().long()
            if not torch.equal(candidate_case, expected_case):
                raise ValueError('candidate_case and candidate_lengths disagree')
            phrase = query[last][candidate_case]
            average = mean_query[candidate_case]
            score_features = []
            for scores in output['scores'].detach().float().split(lengths):
                relative = scores-scores.max()
                probability = scores.softmax(0)
                standardized = (scores-scores.mean())/scores.std(unbiased=False).clamp_min(1e-6)
                score_features.append(torch.stack((relative, probability, standardized), -1))
            return torch.cat((average, phrase, image, average*image, phrase*image,
                frozen_point_features.detach().float(), torch.cat(score_features)), -1).detach()

    def forward(self, frozen_point_features, **core_inputs):
        forbidden = {'labels_fixed_point_hit', 'label_fixed_point_hit', 'label_hard_negative',
                     'label_area', 'label_offsets', 'geometry', 'targets'} & core_inputs.keys()
        if forbidden:
            raise TypeError('Supervision is loss-only and may not enter forward: '+', '.join(sorted(forbidden)))
        n = len(core_inputs['image_hidden'])
        if frozen_point_features.shape != (n, FROZEN_POINT_FEATURE_DIM):
            raise ValueError('frozen_point_features must have shape [candidate,13]')
        if not bool(torch.isfinite(frozen_point_features).all()):
            raise ValueError('Frozen point features must be finite')
        if frozen_point_features.device != core_inputs['image_hidden'].device:
            raise ValueError('Frozen point features and core inputs must share a device')
        lengths = _lengths(core_inputs['candidate_lengths'], n)
        output = dict(self.core(**core_inputs))
        if output['scores'].shape != (n,) or not bool(torch.isfinite(output['scores']).all()):
            raise ValueError('Core scores must be a finite flat candidate vector')
        features = self._detached_features(core_inputs, output, frozen_point_features, lengths)
        logits = self.quality(features).squeeze(-1)
        output['point_quality_logits'] = logits
        output['combined_scores'] = output['scores']+F.logsigmoid(logits)
        output['quality_only_scores'] = logits
        return output


def quality_loss(logits, labels_fixed_point_hit, candidate_lengths):
    """Case-balanced, class-balanced binary fixed-point success loss.

    Each screenshot has equal weight. Within a screenshot containing both
    classes, positive and negative groups each get half its weight. All-negative
    and all-positive screenshots retain their ordinary mean BCE. Labels are
    never inputs to the readout itself; probabilities are not calibrated by
    this class-balanced objective.
    """
    if logits.ndim != 1 or labels_fixed_point_hit.shape != logits.shape:
        raise ValueError('Quality logits and binary hit targets must be aligned flat vectors')
    lengths = _lengths(candidate_lengths, len(logits))
    targets = labels_fixed_point_hit.detach().to(device=logits.device, dtype=logits.dtype)
    if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(targets).all()):
        raise ValueError('Quality logits and targets must be finite')
    if not bool(((targets == 0) | (targets == 1)).all()):
        raise ValueError('Fixed-point hit targets must be binary')
    element = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    cases = []
    for losses, hits in zip(element.split(lengths), targets.split(lengths)):
        positive = hits.bool()
        if bool(positive.any()) and bool((~positive).any()):
            cases.append((losses[positive].mean()+losses[~positive].mean())*.5)
        else:
            cases.append(losses.mean())
    return torch.stack(cases).mean()
