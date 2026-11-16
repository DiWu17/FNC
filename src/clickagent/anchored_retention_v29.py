"""Native-free anchored comparative learning; no top-rival training bottleneck."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from clickagent.coordinate_calibration_v29 import X_NAMES


def fit(records, ridge=.3):
    nonempty = [r for r in records if len(r['x'])]
    xx = np.concatenate([r['x'] for r in nonempty]) if nonempty else np.zeros((1, len(X_NAMES)))
    std = xx.std(0);std = np.where(std > 1e-10, np.maximum(std, 1e-4), 1.)
    counts = {s: sum(r['source'] == s for r in records) for s in {r['source'] for r in records}}
    ds, bs, ys, ws, eligible = [], [], [], [], []
    for r in records:
        f = r['native_index']['free']
        if f is None: continue
        rivals = np.flatnonzero(r['y'] != r['y'][f])
        if not len(rivals): continue
        ds.append((r['x'][rivals]-r['x'][f])/std)
        bs.append(r['source_base'][rivals]-r['source_base'][f])
        ys.append(r['y'][rivals].astype(float))
        ws.append(np.full(len(rivals), 1/(len(counts)*counts[r['source']]*len(rivals))))
        eligible.append(dict(sample_id=r['sample_id'], free_correct=bool(r['y'][f]), rival_pairs=len(rivals)))
    dim = len(X_NAMES)
    model = dict(weight=np.zeros(dim), std=std, prior_coef=np.asarray(1.), bias=np.asarray(0.),
                 eligible_queries=len(eligible), pairs=sum(len(a) for a in ds), eligible=eligible, success=True)
    if not ds: return model
    d, b, y, w = np.concatenate(ds), np.concatenate(bs), np.concatenate(ys), np.concatenate(ws);w /= w.sum()
    def objective(theta):
        weight, coefficient, bias = theta[:dim], theta[dim], theta[dim+1]
        logits = d@weight+coefficient*b+bias
        error = w*(expit(logits)-y)
        loss = float(w@(np.logaddexp(0, logits)-y*logits)+ridge*(weight@weight+(coefficient-1)**2+bias**2)/2)
        grad = np.r_[d.T@error+ridge*weight, b@error+ridge*(coefficient-1), error.sum()+ridge*bias]
        return loss, grad
    result = minimize(objective, np.r_[np.zeros(dim), 1., 0.], jac=True, method='L-BFGS-B',
                      bounds=[(None, None)]*dim+[(0., 10.), (None, None)],
                      options=dict(maxiter=200, ftol=1e-10, gtol=1e-6))
    assert result.success and np.isfinite(result.x).all(), result.message
    model.update(weight=result.x[:dim], prior_coef=np.asarray(result.x[dim]), bias=np.asarray(result.x[dim+1]), iterations=int(result.nit), loss=float(result.fun))
    return model


def choose(model, r, fallback_scores, threshold=.75):
    f = r['native_index']['free']
    if f is None:
        j = int(np.argmax(fallback_scores)) if len(fallback_scores) else None
        return j, dict(reason='free_missing_ranker_fallback', probability=None)
    rivals = np.asarray([j for j in range(len(r['points'])) if j != f], int)
    if not len(rivals): return f, dict(reason='keep_exact_free', probability=None)
    logits = ((r['x'][rivals]-r['x'][f])/model['std'])@model['weight']
    logits += float(model['prior_coef'])*(r['source_base'][rivals]-r['source_base'][f])+float(model['bias'])
    k = int(np.argmax(logits));probability = float(expit(logits[k]));switch = probability > threshold
    return (int(rivals[k]) if switch else f), dict(reason='anchored_switch' if switch else 'keep_exact_free', probability=probability,
            rival_index=int(rivals[k]), free_index=f, probability_note='Regularized pairwise comparison conditional on one correct, maximized across candidates; not a calibrated correctness or post-selection risk guarantee.')
