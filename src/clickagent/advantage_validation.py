"""Deterministic, label-blind cohorts and geometry for V17 experiments."""
from __future__ import annotations

from collections import defaultdict
import hashlib
from io import BytesIO

import numpy as np
from PIL import Image, ImageOps


def stable_key(value, namespace='v17'):
    return hashlib.sha256(f'{namespace}:{value}'.encode()).hexdigest()


def nested_groups(entries, fraction):
    """Select whole RGB groups per source; never inspect labels or predictions."""
    if not 0 < fraction <= 1:
        raise ValueError('fraction must be in (0,1]')
    by_group = defaultdict(list)
    for entry in entries:
        by_group[entry['group']].append(entry)
    strata = defaultdict(list)
    for group, rows in by_group.items():
        strata[tuple(sorted({r['source'] for r in rows}))].append(group)
    selected = set()
    for groups in strata.values():
        groups.sort(key=lambda g: stable_key(g, 'v17-training-groups'))
        selected.update(groups[:max(1, int(np.ceil(len(groups) * fraction)))])
    return [e for e in entries if e['group'] in selected]


def balanced_cases(cases, per_source, namespace):
    sources = sorted({c['dataset'] for c in cases})
    rows = []
    for source in sources:
        pool = sorted((c for c in cases if c['dataset'] == source),
                      key=lambda c: stable_key(c['sample_id'], namespace))
        if len(pool) < per_source:
            raise ValueError('Insufficient source examples')
        rows.extend(pool[:per_source])
    return rows


def perturb(image, condition):
    """Return image and affine original-to-perturbed coordinate transform."""
    image = image.convert('RGB')
    if condition == 'clean':
        return image.copy(), (1., 1., 0., 0.)
    if condition == 'jpeg70':
        stream = BytesIO()
        image.save(stream, format='JPEG', quality=70, subsampling=2)
        stream.seek(0)
        with Image.open(stream) as decoded:
            return decoded.convert('RGB'), (1., 1., 0., 0.)
    if condition == 'border24':
        return ImageOps.expand(image, border=24, fill=(0, 0, 0)), (1., 1., 24., 24.)
    raise ValueError('Unknown perturbation')


def map_to_original(point, affine, original_size):
    if point is None:
        return None
    if len(point) != 2 or not np.isfinite(point).all():
        return None
    sx, sy, tx, ty = affine
    x, y = (float(point[0]) - tx) / sx, (float(point[1]) - ty) / sy
    w, h = original_size
    if not (0 <= x < w and 0 <= y < h):
        return None
    return [round(x), round(y)] if round(x) < w and round(y) < h else None


def paired_interval(a, b, sources, seed=20261701, reps=2000):
    """One entry per screenshot; repetitions or seeds never enlarge sample n."""
    delta = np.asarray(a, float) - np.asarray(b, float)
    sources = np.asarray(sources)
    if delta.ndim != 1 or len(delta) != len(sources) or not len(delta):
        raise ValueError('One paired value per screenshot is required')
    rng = np.random.default_rng(seed)
    draws = np.concatenate([rng.choice(np.flatnonzero(sources == s),
                                      size=(reps, int((sources == s).sum())), replace=True)
                            for s in sorted(set(sources))], axis=1)
    return dict(n=len(delta), delta=float(delta.mean()),
                ci95=np.quantile(delta[draws].mean(1), [.025, .975]).tolist(),
                scope='Exploratory; conditional on fixed models and this reused cohort')
