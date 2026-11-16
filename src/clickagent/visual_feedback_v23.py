"""Label-free post-prediction views and strict coordinate/selection decoding."""
from __future__ import annotations

import hashlib
import json
import re

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ARMS = ('local_relook', 'marked_verify', 'marked_refine')
COMMON = (
    "Locate the visible GUI element identified by the complete user instruction. "
    "Image 1 is a letterboxed overview of the original screenshot. Image 2 is a "
    "local crop around the initial prediction. Candidate coordinates below refer "
    "to the ORIGINAL full screenshot, normalized to 0..1000, not either displayed "
    "image or its padding. Colored hollow rings and letters, when present, are "
    "added annotations, not GUI elements. The center of a ring is the proposed "
    "click. The initial prediction is a hypothesis and may be correct or wrong. "
    "Use both images and the instruction. Never claim an action was executed. "
)
COORDINATE_RULE = (
    "Reply with KEEP followed by one candidate letter to retain that exact point, "
    "or CLICK x y to provide a corrected point anywhere in the ORIGINAL full "
    "screenshot, with integer x,y from 0 to 1000. Reply WAIT if uncertain. "
    "Output only one of these formats, without explanation."
)
VERIFY_RULE = (
    "Select the candidate whose exact point best matches the instruction. "
    "Reply with exactly one available candidate letter, or N if none is suitable. "
    "Do not output coordinates or explanations."
)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':')).encode()).hexdigest()


def valid_point(point, size):
    return (isinstance(point, (list, tuple, np.ndarray)) and len(point) == 2
            and all(not isinstance(v, (bool, np.bool_)) and np.isfinite(v)
                    and int(v) == v and 0 <= v < s for v, s in zip(point, size)))


def make_pool(points, scores, size):
    """Keep the original argmax; prefer two different 14px neighborhoods."""
    points = np.asarray(points).reshape(-1, 2)
    scores = np.asarray(scores)
    assert len(points) == len(scores) and np.isfinite(scores).all()
    if not len(points):
        return []
    order = np.argsort(-scores, kind='stable').tolist()
    assert all(valid_point(p, size) for p in points)
    anchor = points[order[0]].astype(int).tolist()
    others = [points[i].astype(int).tolist() for i in order[1:]]
    selected = [anchor]
    while others and len(selected) < 3:
        distinct = [p for p in others if p not in selected]
        diverse = [p for p in distinct
                   if all(max(abs(p[d]-q[d]) for d in (0, 1)) > 14 for q in selected)]
        if not distinct:
            break
        selected.append((diverse or distinct)[0])
        others = distinct
    return [anchor] + sorted(selected[1:])


def presentation(sample_id, pool):
    seed = int(digest([sample_id, pool, 'v23_order'])[:16], 16)
    return np.random.default_rng(seed).permutation(len(pool)).tolist()


def square(image, points=(), labels=()):
    w, h = image.size
    scale = min(512/w, 512/h)
    rw, rh = max(1, round(w*scale)), max(1, round(h*scale))
    ox, oy = (512-rw)//2, (512-rh)//2
    canvas = Image.new('RGB', (512, 512), (240, 240, 240))
    with image.resize((rw, rh), Image.Resampling.LANCZOS) as resized:
        canvas.paste(resized, (ox, oy))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 16)
    except OSError:
        font = ImageFont.load_default(size=16)
    centers = []
    for i, (p, label) in enumerate(zip(points, labels)):
        x, y = ox+p[0]*rw/w, oy+p[1]*rh/h
        draw.ellipse((x-6, y-6, x+6, y+6), outline=(240, 20, 20), width=2)
        dx, dy = [(8, -21), (8, 8), (-23, 8)][i % 3]
        tx, ty = max(0, min(490, x+dx)), max(0, min(490, y+dy))
        draw.rectangle((tx, ty, tx+18, ty+20), fill='white')
        draw.text((tx+2, ty), label, font=font, fill=(170, 0, 0))
        centers.append([x, y])
    return canvas, dict(source_size=[w, h], resized_size=[rw, rh], offset=[ox, oy],
                        marked_centers=centers)


def visual_inputs(image, instruction, sample_id, pool, arm):
    assert arm in ARMS and 1 <= len(pool) <= 3
    assert all(valid_point(p, image.size) for p in pool)
    order = presentation(sample_id, pool)
    shown = [pool[i] for i in order]
    labels = list('ABC')[:len(pool)]
    marked = arm != 'local_relook'
    overview, og = square(image, shown if marked else [], labels if marked else [])
    w, h = image.size
    cw, ch = min(512, w), min(512, h)
    x = max(0, min(w-cw, pool[0][0]-cw//2))
    y = max(0, min(h-ch, pool[0][1]-ch//2))
    inside = [(p, label) for p, label in zip(shown, labels)
              if x <= p[0] < x+cw and y <= p[1] < y+ch]
    with image.crop((x, y, x+cw, y+ch)) as crop:
        local, lg = square(crop, [[p[0]-x, p[1]-y] for p, _ in inside] if marked else [],
                           [label for _, label in inside] if marked else [])
    normalized = [[round(p[0]*1000/(w-1)) if w > 1 else 0,
                   round(p[1]*1000/(h-1)) if h > 1 else 0] for p in shown]
    initial_label = labels[order.index(0)]
    text = (f'Original screenshot size: {w} by {h}.\nUser instruction: {instruction}\n'
            f'Initial prediction: {initial_label}.\nCandidate points (full-image 0..1000):\n'
            + '\n'.join(f'{label}: ({p[0]}, {p[1]})' for label, p in zip(labels, normalized))
            + f'\nLocal crop original-pixel box: ({x}, {y}, {x+cw}, {y+ch}).')
    prompt = COMMON + (VERIFY_RULE if arm == 'marked_verify' else COORDINATE_RULE)
    content = [dict(type='text', text='Full screenshot overview:'), dict(type='image'),
               dict(type='text', text='Local view around the initial prediction:'), dict(type='image'),
               dict(type='text', text=text)]
    messages = [dict(role='system', content=prompt), dict(role='user', content=content)]
    return [overview, local], messages, dict(order=order, labels=labels, initial_label=initial_label,
        pool=pool, shown=shown, normalized=normalized, original_size=list(image.size),
        crop_box=[x, y, x+cw, y+ch], overview=og, local=lg, input_pixels=2*512*512,
        marked=marked, ground_truth_in_model_inputs=False)


def decode_coordinate(text, completed, geometry):
    anchor = list(geometry['pool'][0])
    result = dict(point=anchor, fallback=True, valid=False, kind='invalid_or_truncated')
    if not completed:
        return result
    text = text.strip()
    if text == 'WAIT':
        return dict(point=anchor, fallback=True, valid=True, kind='wait')
    match = re.fullmatch(r'KEEP\s+([ABC])', text)
    if match and match[1] in geometry['labels']:
        i = geometry['labels'].index(match[1])
        return dict(point=list(geometry['shown'][i]), fallback=False, valid=True, kind='keep',
                    candidate_index=geometry['order'][i])
    match = re.fullmatch(r'CLICK\s+(\d{1,4})\s+(\d{1,4})', text)
    if match:
        xy = [int(match[1]), int(match[2])]
        if all(0 <= v <= 1000 for v in xy):
            p = [round(v*(s-1)/1000) for v, s in zip(xy, geometry['original_size'])]
            return dict(point=p, fallback=False, valid=True, kind='coordinate', normalized_point=xy)
    return result


def decode_selection(scores, geometry):
    scores = np.asarray(scores, dtype=np.float64)
    assert scores.shape == (len(geometry['labels'])+1,) and np.isfinite(scores).all()
    i = int(scores.argmax())
    if i == len(geometry['labels']):
        return dict(point=list(geometry['pool'][0]), fallback=True, valid=True, kind='none')
    return dict(point=list(geometry['shown'][i]), fallback=False, valid=True, kind='selection',
                candidate_index=geometry['order'][i])
