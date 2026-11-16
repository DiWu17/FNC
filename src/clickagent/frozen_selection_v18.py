"""Public-only selection rules over fixed points and immutable scorer outputs."""
import numpy as np

METHODS = ('K16', 'union', 'view_rrf', 'support_025', 'support_050', 'support_100', 'top4_consensus')


def percentile(values):
    values = np.asarray(values, np.float64)
    if not len(values):
        return values
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    return (starts[inverse] + (counts[inverse] - 1) / 2) / max(len(values)-1, 1)


def evidence(points, base, image_size, radius=.015, topk=8):
    """At most one proximity contribution from each view, not one per point."""
    points = np.asarray(points, np.float64).reshape(-1, 2)
    base = np.asarray(base)
    assert base.shape == (len(points), 32)
    assert len(image_size) == 2 and min(image_size) > 0
    scale = float(np.hypot(*image_size)) * radius
    support = np.zeros(len(points)); rrf = np.zeros(len(points))
    for view in range(4):
        ids = np.flatnonzero(base[:, view*7] > .5)
        if not len(ids):
            continue
        ids = ids[np.argsort(-base[ids, view*7+2], kind='stable')[:topk]]
        dist2 = ((points[:, None] - points[ids][None])**2).sum(-1)
        kernel = np.exp(-.5*dist2/(scale*scale))
        support += kernel.max(1)
        ranks = np.expm1(base[ids, view*7+1].astype(np.float64))
        rrf += (kernel/(60+ranks)[None]).max(1)
    return support, rrf


def select(points, base, baseline, scores, image_size):
    """No image annotations, labels, action flags, fitting or model access."""
    points = np.asarray(points).reshape(-1, 2); n = len(points)
    scores = np.asarray(scores, np.float64)
    assert scores.shape == (n,) and np.isfinite(scores).all()
    assert np.isfinite(points).all() and np.isfinite(base).all()
    if not n:
        return {name: None for name in METHODS}
    baseline = np.sort(np.asarray(baseline, np.int64))
    assert not len(baseline) or (baseline.min() >= 0 and baseline.max() < n)
    support, rrf = evidence(points, base, image_size)
    quality = percentile(scores); consensus = support/4
    top = np.argsort(-scores, kind='stable')[:4]
    output = dict(K16=int(baseline[np.argsort(-scores[baseline], kind='stable')[0]]) if len(baseline) else None,
                  union=int(np.argmax(scores)), view_rrf=int(np.argmax(rrf)))
    for label, weight in [('025', .25), ('050', .5), ('100', 1.)]:
        output['support_'+label] = int(np.argmax(quality+weight*consensus))
    output['top4_consensus'] = int(top[np.argsort(-support[top], kind='stable')[0]])
    return output
