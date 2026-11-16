"""Uniform v7 checkpoint and scoring API; original frozen readouts are untouched."""
from pathlib import Path

import torch

from .attention_field_ranker import AttentionFieldRanker, FieldRankerConfig
from .point_quality_readout import PointQualityReadout
from .semantic_head_ranker import SemanticHeadRanker, SemanticHeadRankerConfig


def build_readout(architecture, head_prior, width=32):
    if architecture == 'flat':
        core = AttentionFieldRanker(FieldRankerConfig(width, 'all'), head_prior)
    elif architecture in ('static', 'semantic'):
        core = SemanticHeadRanker(SemanticHeadRankerConfig(width=width, head_width=16, groups=4,
            routing='static' if architecture == 'static' else 'semantic'), head_prior)
    else:
        raise ValueError('Unknown controlled architecture')
    return PointQualityReadout(core, quality_width=32)


def save_readout(path, model, architecture, metadata):
    torch.save(dict(schema_version=1, architecture=architecture, width=model.semantic_width,
        state_dict=model.state_dict(), metadata=metadata), path)


def load_readout(path, device='cpu'):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    assert saved['schema_version'] == 1
    model = build_readout(saved['architecture'], saved['state_dict']['core.head_prior'], saved['width'])
    model.load_state_dict(saved['state_dict'], strict=True)
    return model.to(device).eval(), saved['metadata']


def score_output(output, mode):
    if mode == 'rank': return output
    if mode == 'combined': return dict(output, scores=output['combined_scores'])
    if mode == 'quality': return dict(output, scores=output['quality_only_scores'])
    raise ValueError('Unknown candidate score mode')
