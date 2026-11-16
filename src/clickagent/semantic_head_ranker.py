"""Early semantic head routing over every candidate/token/head observation.

The unmodified v5 SemanticCandidateRanker remains the base. This module adds
only a candidate score residual: null scores and fine points are untouched.
Unlike the v6 flat projection, all six channels are encoded independently for
each of the 96 native head identities. Query/phrase, candidate image content,
mass and entropy condition per-head scores *before* four soft head pools; soft
token pooling then retains evidence from every name token for each candidate.

The matched static control has identical modules and parameter count. It zeros
only the explicit target query/phrase inputs to this new router. The original
base semantic branch and the already target-conditioned attention field remain.
There is no claim that static inputs contain no target information.
"""
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from .attention_field_ranker import make_field_batch
from .candidate_ranker import RankerConfig, SemanticCandidateRanker, segment_softmax


SEMANTIC_HEAD_SCHEMA_VERSION = 1


@dataclass
class SemanticHeadRankerConfig:
    width: int = 32
    head_width: int = 16
    groups: int = 4
    routing: str = "semantic"

    def __post_init__(self):
        for key in ("width", "head_width", "groups"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        if self.routing not in ("semantic", "static"):
            raise ValueError("routing must be semantic or static")


class SemanticHeadRanker(SemanticCandidateRanker):
    """Drop-in make_field_batch consumer with optional routing diagnostics.

    ``return_routing=True`` additionally returns [candidate-token,96,groups]
    head weights and [candidate-token,groups] token weights. The default omits
    the large head tensor to avoid unnecessary CPU copies in evaluation.
    Group heads and tokens are always softly pooled, never selected by argmax.
    """

    def __init__(self, config: SemanticHeadRankerConfig, head_prior):
        # This order preserves the exact seeded base initialization of v5/v6.
        super().__init__(RankerConfig(config.width, "full"), head_prior)
        self._base_parameter_names = tuple(name for name, _ in self.named_parameters())
        self._base_module_names = tuple(self._modules)
        self._base_state_keys = tuple(self.state_dict())
        self.base_frozen = False
        self.semantic_config = config
        self.field_config = config
        d, h, g = config.width, config.head_width, config.groups
        self.head_field = nn.Sequential(nn.Linear(6, h), nn.GELU(), nn.LayerNorm(h))
        self.head_identity = nn.Embedding(96, h)
        self.head_quality = nn.Linear(2, h, bias=False)  # image mass and entropy
        self.head_image = nn.Linear(d, h, bias=False)
        self.head_value_norm = nn.LayerNorm(h)
        # No biases: a zero query/phrase is exactly an absent condition.
        self.route_query = nn.Linear(d, h, bias=False)
        self.route_phrase = nn.Linear(d, h, bias=False)
        self.head_group_score = nn.Linear(h, g)
        self.group_token_score = nn.Linear(h, 1)
        # Compress groups only after per-head semantic scoring and both pools.
        self.field_projection = nn.Sequential(nn.Linear(g * h, d), nn.GELU(), nn.LayerNorm(d))
        self.field_rank = nn.Sequential(nn.Linear(d * 3, d * 2), nn.GELU(),
                                        nn.Dropout(.1), nn.Linear(d * 2, 1))
        nn.init.zeros_(self.field_rank[-1].weight)
        nn.init.zeros_(self.field_rank[-1].bias)

    def _field_residual(self, attention_field, field_token, field_candidate, base, return_routing):
        field = attention_field.float()
        if field.ndim != 3 or tuple(field.shape[1:]) != (96, 6) or not len(field):
            raise ValueError("attention_field must have nonempty shape [candidate-token,96,6]")
        if field_token.shape != (len(field),) or field_candidate.shape != (len(field),):
            raise ValueError("field_token and field_candidate must align with every candidate-token row")
        n = len(base["image_hidden"])
        qraw = F.layer_norm(base["query_hidden"].float().reshape(-1, 3, 4096), [4096])
        query = self.query(qraw.flatten(1)) + self.shared_semantic(qraw[:, 0])
        vision = self.vision(F.layer_norm(base["vision_hidden"].float(), [1152]))
        image = self.shared_semantic(F.layer_norm(base["image_hidden"].float(), [4096])) + vision.mean(1)
        phrase = query[base["last_query"]][base["candidate_case"]]
        local_query, local_phrase = query[field_token], phrase[field_candidate]
        if self.semantic_config.routing == "static":
            local_query = torch.zeros_like(local_query)
            local_phrase = torch.zeros_like(local_phrase)
        condition = self.route_query(local_query) + self.route_phrase(local_phrase)
        # [M,96,h] throughout: no head flattening, averaging or fixed shortlist.
        value = self.head_field(field)
        value = value + self.head_identity.weight[None]
        value = value + self.head_image(image[field_candidate])[:, None]
        value = value + self.head_quality(field[..., 3:5])
        value = self.head_value_norm(F.gelu(value))
        compatibility = torch.tanh(value + condition[:, None] + value * condition[:, None])
        head_logits = self.head_group_score(compatibility)
        head_weights = head_logits.softmax(dim=1)
        grouped_tokens = torch.bmm(head_weights.transpose(1, 2), value)
        token_logits = self.group_token_score(grouped_tokens).squeeze(-1)
        token_weights = torch.stack([
            segment_softmax(token_logits[:, group], field_candidate, n)
            for group in range(self.semantic_config.groups)], dim=-1)
        grouped_candidates = value.new_zeros((n, self.semantic_config.groups, self.semantic_config.head_width))
        grouped_candidates = grouped_candidates.index_add(
            0, field_candidate, grouped_tokens * token_weights[..., None])
        pooled = self.field_projection(grouped_candidates.flatten(1))
        residual = self.field_rank(torch.cat((pooled, image, pooled * image), dim=-1)).squeeze(-1)
        extras = {"field_residual": residual, "field_token_weights": token_weights.mean(dim=-1)}
        if return_routing:
            extras.update(head_routing_weights=head_weights, group_token_weights=token_weights)
        return extras

    def forward(self, attention_field, field_token, field_candidate, return_routing=False, **base):
        output = super().forward(**base)
        extras = self._field_residual(attention_field, field_token, field_candidate, base, return_routing)
        output["base_scores"] = output["scores"]
        output["scores"] = output["scores"] + extras["field_residual"]
        output.update(extras)
        return output

    def freeze_base(self, frozen=True):
        """Optional warm-start helper; from-scratch experiments need not call it."""
        for name, parameter in self.named_parameters():
            if name in self._base_parameter_names:
                parameter.requires_grad_(not frozen)
        self.base_frozen = bool(frozen)
        self.train(self.training)
        return self

    def train(self, mode=True):
        super().train(mode)
        if getattr(self, "base_frozen", False):
            for name in self._base_module_names:
                self._modules[name].eval()
        return self

    def load_base_ranker_state(self, state_dict, freeze=True):
        """Strictly load an unmodified v5 state; never silently drop extra keys."""
        if set(state_dict) != set(self._base_state_keys):
            raise ValueError("base checkpoint must contain exactly the original v5 state keys")
        result = self.load_state_dict(state_dict, strict=False)
        expected_missing = set(self.state_dict()) - set(self._base_state_keys)
        if set(result.missing_keys) != expected_missing or result.unexpected_keys:
            raise RuntimeError("unexpected semantic router/base checkpoint mismatch")
        return self.freeze_base(freeze)


def save_semantic_head_ranker(path: str | Path, model: SemanticHeadRanker, metadata: dict[str, Any]):
    torch.save(dict(schema_version=SEMANTIC_HEAD_SCHEMA_VERSION, config=asdict(model.semantic_config),
                    state_dict=model.state_dict(), base_frozen=model.base_frozen, metadata=metadata), path)


def load_semantic_head_ranker(path: str | Path, device="cpu"):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("schema_version") != SEMANTIC_HEAD_SCHEMA_VERSION:
        raise ValueError("unsupported semantic-head checkpoint schema")
    model = SemanticHeadRanker(SemanticHeadRankerConfig(**saved["config"]), saved["state_dict"]["head_prior"])
    model.load_state_dict(saved["state_dict"], strict=True)
    model.freeze_base(saved["base_frozen"])
    return model.to(device).eval(), saved["metadata"]
