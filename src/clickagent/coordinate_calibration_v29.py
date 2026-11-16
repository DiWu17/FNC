"""Train-only prior calibration and exact native-coordinate retention.

Features contain relative geometry and output agreement, never source/image IDs,
absolute screen positions, ground truth, or an imputed old score for new points.
"""
import re
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

MODES = ('fixed', 'temperature', 'source')
ORIGINS = ('A', 'global', 'free', 'locked')
Q_NAMES = ('valid_global', 'valid_free', 'valid_locked', 'name_GF', 'name_GL',
           'name_FL', 'distance_FL', 'distance_GF', 'distance_GL', 'A_margin',
           'A_entropy', 'free_crop_edge', 'locked_crop_edge')
X_NAMES = ([f'origin_{p}' for p in ORIGINS] +
           [f'{p}_{d}' for p in ORIGINS[1:] for d in ('distance', 'near_normalized', 'near_14px')] +
           ['inside_local_crop'] + [f'{p}*{q}' for p in ORIGINS for q in Q_NAMES])
GATE_NAMES = ['rival_minus_free', 'rival_free_distance', *[f'rival_{p}' for p in ORIGINS], *Q_NAMES]


def zscore(a):
    a = np.asarray(a, dtype=float)
    return (a-a.mean())/max(float(a.std()), 1e-6) if len(a) else a.copy()


def name_agreement(a, b):
    aa, bb = (set(re.findall(r'\w+', (s or '').casefold())) for s in (a, b))
    return len(aa & bb)/len(aa | bb) if aa and bb else 0.


def features(points, old_count, old_scores, native, names, crop_box, size):
    """All arguments are inference observations. old_scores includes V28 fill."""
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    old_scores = np.asarray(old_scores, dtype=float)
    n = len(points)
    assert len(old_scores) == n and 0 <= old_count <= n
    size = np.asarray(size, dtype=float)
    diagonal = max(float(np.linalg.norm(size)), 1.)
    origin = np.zeros((n, 4))
    origin[:old_count, 0] = 1
    indices = {}
    for j, p in enumerate(ORIGINS[1:], 1):
        point = native[p]
        ix = np.flatnonzero(np.all(points == np.asarray(point), axis=1)) if point is not None else []
        if point is not None:
            assert len(ix) == 1, (p, point, points)
            origin[ix[0], j] = 1
            indices[p] = int(ix[0])
        else:
            indices[p] = None
    a_z = zscore(old_scores[:old_count])
    source_base = np.r_[a_z, np.zeros(n-old_count)]
    def distance(a, b):
        return float(np.linalg.norm(np.asarray(native[a])-native[b])/diagonal) if native[a] is not None and native[b] is not None else 0.
    def edge(p):
        if native[p] is None:
            return 0.
        x0, y0, x1, y1 = crop_box
        x, y = native[p]
        return float(np.clip(min((x-x0)/max(x1-x0, 1), (x1-x)/max(x1-x0, 1),
                                 (y-y0)/max(y1-y0, 1), (y1-y)/max(y1-y0, 1))*2, 0, 1))
    ordered = np.sort(a_z)[::-1]
    margin = float(np.clip(ordered[0]-ordered[1], 0, 5)/5) if old_count > 1 else 0.
    prob = np.exp(a_z-a_z.max()) if old_count else np.empty(0)
    if old_count:
        prob /= prob.sum()
    entropy = float(-(prob*np.log(np.maximum(prob, 1e-12))).sum()/np.log(old_count)) if old_count > 1 else 0.
    q = np.asarray([*[float(native[p] is not None) for p in ORIGINS[1:]],
                    name_agreement(names['global'], names['free']), name_agreement(names['global'], names['locked']),
                    name_agreement(names['free'], names['locked']), distance('free', 'locked'),
                    distance('global', 'free'), distance('global', 'locked'), margin, entropy, edge('free'), edge('locked')])
    distances = []
    for p in ORIGINS[1:]:
        d = np.linalg.norm(points-np.asarray(native[p]), axis=1) if native[p] is not None else np.zeros(n)
        distances += [d/diagonal, np.exp(-.5*(d/diagonal/.03)**2), np.exp(-.5*(d/14.)**2)] if native[p] is not None else [np.zeros(n)]*3
    x0, y0, x1, y1 = crop_box
    inside = ((points[:, 0] >= x0) & (points[:, 0] < x1) & (points[:, 1] >= y0) & (points[:, 1] < y1)).astype(float)
    geometry = np.column_stack([origin, *distances, inside])
    interactions = (origin[:, :, None]*q[None, None, :]).reshape(n, 4*len(Q_NAMES))
    x = np.column_stack([geometry, interactions])
    assert x.shape == (n, len(X_NAMES)) and np.isfinite(x).all()
    return dict(x=x, q=q, origin=origin, base=zscore(old_scores), source_base=source_base,
                native_index=indices, old_count=old_count, points=points.astype(int), size=size)


def _scale(xx):
    mean, std = xx.mean(0), xx.std(0)
    return mean, np.where(std > 1e-10, np.maximum(std, 1e-4), 1.)


def fit_ranker(records, mode, ridge=.3):
    assert mode in MODES
    dim = len(X_NAMES)
    nonempty = [r for r in records if len(r['x'])]
    mean, std = _scale(np.concatenate([r['x'] for r in nonempty])) if nonempty else (np.zeros(dim), np.ones(dim))
    sources = sorted({r['source'] for r in records})
    counts = {s: sum(r['source'] == s for r in records) for s in sources}
    ds, bs, ws = [], [], []
    rankable = 0
    for r in records:
        pos, neg = np.flatnonzero(r['y']), np.flatnonzero(~r['y'])
        if not len(pos) or not len(neg):
            continue
        # Real supplemental positives get priority; no new coordinate is added.
        pos = np.asarray(sorted(pos, key=lambda j: (j < r['old_count'], -r['base'][j], j))[:4])
        distance = np.linalg.norm((r['points'][neg, None]-r['points'][pos][None])/r['size'], axis=-1).min(1)
        neg = np.unique(np.r_[neg[np.argsort(distance, kind='stable')[:8]], neg[np.argsort(-r['base'][neg], kind='stable')[:8]]])
        pi, ni = np.repeat(pos, len(neg)), np.tile(neg, len(pos))
        ds.append((r['x'][pi]-r['x'][ni])/std)
        prior = r['source_base'] if mode == 'source' else r['base']
        bs.append(prior[pi]-prior[ni])
        ws.append(np.full(len(pi), 1/(len(sources)*counts[r['source']]*len(pi))))
        rankable += 1
    default = dict(weight=np.zeros(dim), mean=mean, std=std, prior_coef=np.asarray(1.), mode=mode,
                   rankable_queries=rankable, success=True, pairs=0, iterations=0)
    if not ds:
        return default
    d, b, w = np.concatenate(ds), np.concatenate(bs), np.concatenate(ws)
    w /= w.sum()
    learn = mode != 'fixed'
    def objective(theta):
        coefficient = theta[-1] if learn else 1.
        weight = theta[:-1] if learn else theta
        margin = d@weight+coefficient*b
        g = w*expit(-margin)
        loss = float(w@np.logaddexp(0, -margin)+ridge*(weight@weight)/2)
        grad = -d.T@g+ridge*weight
        if learn:
            loss += ridge*(coefficient-1.)**2/2
            grad = np.r_[grad, -b@g+ridge*(coefficient-1.)]
        return loss, grad
    start = np.r_[np.zeros(dim), 1.] if learn else np.zeros(dim)
    result = minimize(objective, start, jac=True, method='L-BFGS-B',
                      bounds=[(None, None)]*dim+[(0., 10.)] if learn else None,
                      options=dict(maxiter=200, ftol=1e-10, gtol=1e-6))
    assert np.isfinite(result.x).all() and result.success, result.message
    default.update(weight=result.x[:-1] if learn else result.x, prior_coef=np.asarray(result.x[-1] if learn else 1.),
                   pairs=len(d), iterations=int(result.nit), loss=float(result.fun))
    return default


def rank_scores(model, r):
    base = r['source_base'] if model['mode'] == 'source' else r['base']
    return float(model['prior_coef'])*base+(r['x']-model['mean'])/model['std']@model['weight']


def rival(r, scores):
    f = r['native_index']['free']
    available = [j for j in range(len(scores)) if j != f]
    return max(available, key=lambda j: (scores[j], -j)) if available else None


def gate_features(r, scores, other):
    f = r['native_index']['free']
    assert f is not None and other is not None
    return np.r_[scores[other]-scores[f], np.linalg.norm(r['points'][other]-r['points'][f])/np.linalg.norm(r['size']),
                 r['origin'][other], r['q']]


def fit_gate(records, out_of_fold_scores, ridge=.3):
    sources = sorted({r['source'] for r in records})
    counts = {s: sum(r['source'] == s for r in records) for s in sources}
    xx, yy, ww, used = [], [], [], []
    for r in records:
        scores = out_of_fold_scores[r['sample_id']]
        f, other = r['native_index']['free'], rival(r, scores)
        if f is None or other is None or r['y'][f] == r['y'][other]:
            continue
        xx.append(gate_features(r, scores, other));yy.append(float(r['y'][other]));used.append(r['sample_id'])
        ww.append(1/(len(sources)*counts[r['source']]))
    dim = len(GATE_NAMES)
    default = dict(weight=np.zeros(dim), bias=np.asarray(0.), mean=np.zeros(dim), std=np.ones(dim),
                   discordant_queries=len(xx), rival_better=sum(yy), training_ids=used, success=True)
    if not xx:
        return default
    x = np.asarray(xx);y = np.asarray(yy);w = np.asarray(ww);w /= w.sum()
    mean, std = _scale(x);x = (x-mean)/std
    def objective(theta):
        logits = x@theta[:-1]+theta[-1]
        error = w*(expit(logits)-y)
        return (float(w@(np.logaddexp(0, logits)-y*logits)+ridge*(theta@theta)/2),
                np.r_[x.T@error, error.sum()]+ridge*theta)
    result = minimize(objective, np.zeros(dim+1), jac=True, method='L-BFGS-B',
                      options=dict(maxiter=200, ftol=1e-10, gtol=1e-6))
    assert result.success and np.isfinite(result.x).all(), result.message
    default.update(weight=result.x[:-1], bias=np.asarray(result.x[-1]), mean=mean, std=std, loss=float(result.fun))
    return default


def choose(r, scores, gate=None, threshold=.75):
    f = r['native_index']['free']
    other = rival(r, scores)
    if f is None:
        return (int(np.argmax(scores)) if len(scores) else None), dict(reason='free_missing_ranker_fallback', probability=None)
    if other is None or gate is None:
        return f, dict(reason='keep_exact_free', probability=None)
    x = gate_features(r, scores, other)
    probability = float(expit((x-gate['mean'])/gate['std']@gate['weight']+gate['bias']))
    switch = probability > threshold
    return (other if switch else f), dict(reason='learned_switch' if switch else 'keep_exact_free',
                                          probability=probability, free_index=f, rival_index=other,
                                          gate_probability_meaning='Estimated rival-better probability conditional on exactly one of rival/free being correct; regularized train-OOF model, not a correctness guarantee.')
