"""Label-free absolute-location residuals on an immutable candidate set."""
from __future__ import annotations

import hashlib
import numpy as np

LOCATIONS = ('upper_left', 'upper_center', 'upper_right', 'middle_left', 'center',
             'middle_right', 'lower_left', 'lower_center', 'lower_right')


def location_index(metadata):
    fields = metadata.get('ground_fields', {})
    name = fields.get('location', 'unknown').strip().lower()
    valid = (metadata.get('ground_complete', False) and metadata.get('target_available', False)
             and metadata.get('action') == 'CLICK' and name in LOCATIONS)
    return LOCATIONS.index(name) if valid else None


def cell_ids(points, image_size):
    xy = np.asarray(points, np.float64).reshape(-1, 2) / np.asarray(image_size)
    ij = np.clip(np.floor(xy * 3), 0, 2).astype(np.int64)
    return ij[:, 1] * 3 + ij[:, 0]


def location_residual(points, image_size, location, sigma=.35):
    """Zero in the named third-bin, soft negative squared distance outside it.

    The named bin is geometry, not its token attention. Unknown is identity.
    This function accepts no labels and never removes or moves a candidate.
    """
    xy = np.asarray(points, np.float64).reshape(-1, 2) / np.asarray(image_size)
    if location is None:
        return np.zeros(len(xy), np.float64)
    if not 0 <= location < 9:
        raise ValueError('Expected a nine-bin location')
    col, row = location % 3, location // 3
    lo = np.asarray([col, row]) / 3
    hi = lo + 1 / 3
    gap = np.maximum(np.maximum(lo - xy, xy - hi), 0)
    return -(gap * gap).sum(1) / (2 * sigma * sigma)


def normalized_scores(scores):
    scores = np.asarray(scores, np.float64)
    if not np.isfinite(scores).all():
        raise ValueError('Nonfinite original score')
    return (scores - scores.mean()) / max(float(scores.std()), 1e-6) if len(scores) else scores


def choose(scores, residual=None, strength=0.):
    scores = np.asarray(scores)
    if not len(scores):
        return None
    if residual is None or strength == 0:
        return int(scores.argmax())
    if np.shape(residual) != scores.shape:
        raise ValueError('Candidate alignment mismatch')
    return int((normalized_scores(scores) + strength * residual).argmax())


def shuffled_locations(cases, locations, draw):
    """Label-free Sattolo donor permutation within cohort/source, missingness fixed.

    Donor indices are deranged when n>1. Equal location values can still agree.
    Development controls use development public metadata, never labels.
    """
    groups = {}
    for c in cases:
        sid = c['sample_id']
        if locations[sid] is not None:
            groups.setdefault((c['cohort'], c['source']), []).append(sid)
    result = dict(locations)
    donors = {sid: sid for sid in locations}
    for group, ids in sorted(groups.items()):
        ids = sorted(ids)
        seed = int(hashlib.sha256(repr((group, draw, 'v26')).encode()).hexdigest()[:16], 16)
        rng = np.random.default_rng(seed)
        order = list(range(len(ids)))
        for i in range(len(ids) - 1, 0, -1):
            j = int(rng.integers(i))
            order[i], order[j] = order[j], order[i]
        for sid, j in zip(ids, order):
            donors[sid] = ids[j]
            result[sid] = locations[ids[j]]
    return result, donors
