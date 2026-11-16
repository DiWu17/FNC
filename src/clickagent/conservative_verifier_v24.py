"""Candidate-complete views, label-order agreement and conservative replacement.

No target geometry is accepted by the inference helpers. The optional learned
gate is fitted separately with application-group out-of-fold predictions.
"""
from __future__ import annotations

import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from .visual_feedback_v23 import digest, valid_point

K = 9
MARGIN = math.log(3.0)
FEATURE_NAMES = (
    'mean_winner_anchor_margin', 'min_winner_anchor_margin',
    'mean_winner_runnerup_margin', 'order_margin_difference',
    'scorer_anchor_winner_gap', 'orders_disagree',
    'mean_winner_probability', 'mean_anchor_probability',
    'mean_none_probability', 'normalized_point_distance',
)
PROMPT = (
    "Identify the exact GUI target described by the complete user instruction. "
    "Image 1 is the full screenshot overview. Each following image is a local "
    "crop centered on one candidate point. A hollow red ring marks the exact "
    "proposed click; its center must be inside the requested target. Letters "
    "and rings are added annotations, not original GUI content. Compare all "
    "candidate regions using the instruction and overview context. Select the "
    "best candidate letter, or N if none is suitable or the target is ambiguous. "
    "Reply with exactly one available letter or N, without explanation."
)


def make_pool(points, scores, size, k=K):
    points = np.asarray(points).reshape(-1, 2)
    scores = np.asarray(scores)
    assert 1 <= k <= K and scores.shape == (len(points),) and np.isfinite(scores).all()
    assert all(valid_point(p, size) for p in points)
    if not len(points):
        return []
    ranked = [points[i].astype(int).tolist() for i in np.argsort(-scores, kind='stable')]
    selected = [ranked[0]]
    while len(selected) < k:
        distinct = [p for p in ranked if p not in selected]
        if not distinct:
            break
        diverse = [p for p in distinct if all(max(abs(p[d]-q[d]) for d in (0, 1)) > 14 for q in selected)]
        selected.append((diverse or distinct)[0])
    return [selected[0]] + sorted(selected[1:])


def order_for(sample_id, pool, reverse=False):
    seed = int(digest([sample_id, pool, 'v24_order'])[:16], 16)
    order = np.random.default_rng(seed).permutation(len(pool)).tolist()
    return list(reversed(order)) if reverse else order


def square(image, points, labels, side):
    w, h = image.size
    scale = min(side/w, side/h)
    rw, rh = max(1, round(w*scale)), max(1, round(h*scale))
    ox, oy = (side-rw)//2, (side-rh)//2
    canvas = Image.new('RGB', (side, side), (240, 240, 240))
    with image.resize((rw, rh), Image.Resampling.LANCZOS) as resized:
        canvas.paste(resized, (ox, oy))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 15)
    except OSError:
        font = ImageFont.load_default(size=15)
    centers = []
    for j, (p, label) in enumerate(zip(points, labels)):
        x, y = ox+p[0]*rw/w, oy+p[1]*rh/h
        draw.ellipse((x-5, y-5, x+5, y+5), outline=(240, 20, 20), width=2)
        dx, dy = [(7, -20), (7, 7), (-22, 7)][j % 3]
        tx, ty = max(0, min(side-19, x+dx)), max(0, min(side-20, y+dy))
        draw.rectangle((tx, ty, tx+18, ty+19), fill='white')
        draw.text((tx+2, ty), label, font=font, fill=(170, 0, 0))
        centers.append([x, y])
    return canvas, dict(source_size=[w, h], resized_size=[rw, rh], offset=[ox, oy], marked_centers=centers)


def visual_inputs(image, instruction, pool, order):
    assert 1 <= len(pool) <= K and sorted(order) == list(range(len(pool)))
    assert all(valid_point(p, image.size) for p in pool)
    labels = list('ABCDEFGHI')[:len(pool)]
    shown = [pool[i] for i in order]
    overview, overview_meta = square(image, shown, labels, 512)
    images = [overview]
    crops = []
    w, h = image.size
    content = [dict(type='text', text='Full screenshot overview:'), dict(type='image')]
    for point, label in zip(shown, labels):
        cw, ch = min(256, w), min(256, h)
        x, y = max(0, min(w-cw, point[0]-cw//2)), max(0, min(h-ch, point[1]-ch//2))
        with image.crop((x, y, x+cw, y+ch)) as crop:
            view, meta = square(crop, [[point[0]-x, point[1]-y]], [label], 256)
        images.append(view)
        crops.append(dict(box=[x, y, x+cw, y+ch], point=list(point), label=label, render=meta))
        content += [dict(type='text', text=f'Candidate {label} local crop:'), dict(type='image')]
    content.append(dict(type='text', text='User instruction: '+instruction+'\nValid replies: '+', '.join(labels+['N'])+'.'))
    return images, [dict(role='system', content=PROMPT), dict(role='user', content=content)], dict(
        order=order, labels=labels+['N'], pool=pool, crops=crops, overview=overview_meta,
        input_pixels=512**2+len(pool)*256**2, ground_truth_in_model_inputs=False)


def combine(pool, observations):
    assert pool and len(observations) == 2
    k = len(pool)
    aligned = []
    for obs in observations:
        scores = np.asarray(obs['scores'], np.float64)
        order = obs['order']
        assert scores.shape == (k+1,) and np.isfinite(scores).all()
        assert sorted(order) == list(range(k))
        logp = scores - (scores.max()+np.log(np.exp(scores-scores.max()).sum()))
        canonical = np.empty(k+1)
        canonical[order] = logp[:-1]
        canonical[-1] = logp[-1]
        aligned.append(canonical)
    aligned = np.stack(aligned)
    scores = aligned.mean(0)
    winner = int(scores.argmax())
    choices = aligned.argmax(1).tolist()
    candidate = winner if winner < k else 0
    margins = [float(row[candidate]-np.delete(row, candidate).max()) for row in aligned]
    agree = choices[0] == choices[1] == winner
    change = winner < k and winner != 0
    fixed_accept = bool(change and agree and min(margins) >= MARGIN)
    return dict(unconditional_point=pool[candidate], fixed_point=pool[candidate] if fixed_accept else pool[0],
        winner=winner, candidate=candidate, per_order_choice=choices, order_agrees=agree,
        proposed_change=change, fixed_accept=fixed_accept, min_winner_runnerup_margin=min(margins),
        aligned_log_probabilities=aligned.tolist(), mean_log_probabilities=scores.tolist())


def features(combined, pool, points, readout_scores, size):
    a = np.asarray(combined['aligned_log_probabilities'], np.float64)
    c = combined['candidate']
    margin = a[:, c]-a[:, 0]
    runner = np.array([row[c]-np.delete(row, c).max() for row in a])
    all_points = np.asarray(points)
    readout_scores = np.asarray(readout_scores)
    scores = [float(readout_scores[np.flatnonzero(np.all(all_points == p, axis=1))].max()) for p in pool]
    gap = (scores[0]-scores[c])/max(1., float(np.std(scores)))
    prob = np.exp(a)
    scaled = lambda x: float(np.clip(x, -8, 8)/8)
    return [scaled(margin.mean()), scaled(margin.min()), scaled(runner.mean()),
        scaled(abs(margin[0]-margin[1])), scaled(gap), float(not combined['order_agrees']),
        float(prob[:, c].mean()), float(prob[:, 0].mean()), float(prob[:, -1].mean()),
        float(max(abs(pool[c][d]-pool[0][d])/max(1, size[d]) for d in [0, 1]))]


def application_folds(cases, nfolds=5):
    """All screenshots from an application/episode remain in the same fold."""
    by_source = {}
    for c in cases:
        by_source.setdefault(c['source'], set()).add((c['source'], c['source_group']))
    folds = {}
    for source, groups in sorted(by_source.items()):
        for i, group in enumerate(sorted(groups, key=lambda g: digest([g, 'v24_gate_fold']))):
            folds[group] = i % nfolds
    return {c['sample_id']: folds[c['source'], c['source_group']] for c in cases}


def fit_gate(records, ridge=20.0):
    """Fixed-strength ridge estimates expected change utility, not correctness confidence."""
    use = [r for r in records if r['proposed_change']]
    dim = len(FEATURE_NAMES)+1
    if not use:
        return dict(coefficients=[0.]*dim, training_rows=0, training_rgb=0)
    counts = {}
    for r in use:
        counts[r['sample_id']] = counts.get(r['sample_id'], 0)+1
    x = np.asarray([[1.]+r['features'] for r in use], np.float64)
    y = np.asarray([r['utility'] for r in use], np.float64)
    w = np.asarray([1/counts[r['sample_id']] for r in use])
    penalty = np.eye(dim)*ridge
    penalty[0, 0] = 1e-8
    coef = np.linalg.solve(x.T@(w[:, None]*x)+penalty, x.T@(w*y))
    return dict(coefficients=coef.tolist(), training_rows=len(use), training_rgb=len(counts))


def predict_utility(model, x):
    return float(np.dot(model['coefficients'], [1.]+list(x)))
