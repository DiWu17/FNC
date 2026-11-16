"""Semantic target-name and temporal readout policies; spatial decoder unchanged."""
from __future__ import annotations

import re
import numpy as np

from .target_token import TARGET_NAME_PROMPT, TargetTokenRuntime


COMPACT_TARGET_PROMPT = (
    "Inspect the screenshot and identify the exact GUI element requested by the user. "
    "Reply in exactly two lines:\nACTION: CLICK or WAIT\nTARGET: minimal target name\n"
    "Choose CLICK only if the requested element is identifiable and can be clicked; otherwise WAIT. "
    "TARGET must be the shortest distinctive visible label, copied with its original spelling and language. "
    "Use at most six words. Omit leading articles, location phrases and generic suffixes such as button, "
    "menu item or text field unless they are literally part of the visible label. "
    "For an icon without text, use a short concrete name describing its appearance or function. "
    "Do not repeat the user's whole instruction or describe surrounding controls. "
    "If no target can be identified, use NONE. Do not output coordinates, explanations, or additional lines."
)
PROMPTS = {"original": TARGET_NAME_PROMPT, "compact": COMPACT_TARGET_PROMPT}
# This order is the fixed simplicity tie-break, not a ranking learned on test.
POLICIES = ("target_mean", "target_first", "target_first2", "core_mean", "core_first2", "mass_weighted_core")
STOP = frozenset("a an the this that of for at on in to and or with button icon menu item option field tab label".split())


class SemanticTargetRuntime(TargetTokenRuntime):
    """Sequential experiment runtime; reuse the audited own-token collector."""
    prompt_variant = "original"

    def _prepare_image(self, image, instruction, system_prompt):
        if system_prompt == TARGET_NAME_PROMPT:
            system_prompt = PROMPTS[self.prompt_variant]
        return super()._prepare_image(image, instruction, system_prompt)

    def infer_variant(self, image, instruction, variant):
        self.prompt_variant = variant
        arrays, meta = self.infer_target_name(image, instruction, max_new_tokens=64)
        meta.update(prompt_variant=variant, system_prompt=PROMPTS[variant])
        return arrays, meta


def core_indices(meta):
    indices = meta["stages"].get("target_mean", [])
    prefixes = [""] + meta["generated_prefixes"]
    kept = []
    for index in indices:
        before, after = prefixes[index - 1], prefixes[index]
        common = 0
        for a, b in zip(before, after):
            if a != b:
                break
            common += 1
        piece = after[common:]
        words = re.findall(r"[^\W_]+", piece, flags=re.UNICODE)
        if "\ufffd" in piece or any(word.casefold() not in STOP for word in words):
            kept.append(index)
    # A literal label may itself be "Menu" or "Tab". Never discard the
    # complete name just because all its words are on the stop list.
    return kept or list(indices)


def temporal_map(raw, meta, policy):
    indices = list(meta["stages"].get("target_mean", []))
    if not indices:
        return None
    if policy in {"core_mean", "core_first2", "mass_weighted_core"}:
        indices = core_indices(meta)
    if policy == "target_first":
        indices = indices[:1]
    elif policy in {"target_first2", "core_first2"}:
        indices = indices[:2]
    elif policy not in {"target_mean", "core_mean", "mass_weighted_core"}:
        raise ValueError(policy)
    chosen = np.asarray(raw[indices], np.float64)
    if policy == "mass_weighted_core":
        mass = chosen.sum(-1).mean(-1)
        weights = mass / mass.sum() if mass.sum() > 0 else np.full(len(indices), 1 / len(indices))
        return np.einsum("t,thp->hp", weights, chosen).astype(np.float32)
    return chosen.mean(0).astype(np.float32)
