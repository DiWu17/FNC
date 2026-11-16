"""Label-free perturbations for frozen attention-field mechanism diagnostics."""
import hashlib

import numpy as np


def perturb_field(field, mode, sample_id='', seed=0):
    """Preserve candidate order and argmax flags; never accept target geometry."""
    x = np.asarray(field)
    if x.ndim != 4 or x.shape[-2:] != (96, 6):
        raise ValueError('Expected [candidate, token, 96 heads, 6 channels]')
    y = x.copy(); nonpeak = x[..., 5] == 0
    if mode == 'full':
        return y
    if mode == 'nonpeak_all_zero':
        y[..., :5] *= (~nonpeak)[..., None]
    elif mode == 'nonpeak_spatial_zero':
        y[..., :3] *= (~nonpeak)[..., None]
    elif mode == 'nonpeak_context_zero':
        y[..., 3:5] *= (~nonpeak)[..., None]
    elif mode == 'nonpeak_spatial_shuffle':
        digest = hashlib.sha256(f'{sample_id}:{seed}'.encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], 'little'))
        for t in range(x.shape[1]):
            for h in range(96):
                indices = np.flatnonzero(nonpeak[:, t, h])
                # The three spatial channels move together, preserving their
                # per-head joint distribution while breaking location alignment.
                y[indices, t, h, :3] = x[rng.permutation(indices), t, h, :3]
    else:
        raise ValueError('Unknown frozen-field perturbation')
    return y


def token_pieces(metadata):
    prefixes = metadata['generated_prefixes']; result = []
    for raw in metadata['target_raw_indices']:
        # Cached raw generation indices are one-based (prefill is position 0).
        before = prefixes[raw-2] if raw > 1 else ''
        after = prefixes[raw-1]
        result.append(after[len(before):] if after.startswith(before) else after)
    return result
