"""Explicit pixel-coordinate adapters and screenshot grouping for V15 data."""
from __future__ import annotations

from collections import Counter
import hashlib
import math
from pathlib import Path

import numpy as np
from PIL import Image


def local_image_path(root, relative):
    """Reversibly encode Windows-invalid path characters, retaining source IDs."""
    from clickagent.public_data import safe_path
    if not isinstance(relative, str) or not relative.startswith('images/'):
        raise ValueError('Expected a source-relative images/ path')
    parts = relative.split('/')
    if any(p in ('', '.', '..') for p in parts):
        raise ValueError('Unsafe source image path')
    reserved = {'CON', 'PRN', 'AUX', 'NUL'} | {f'COM{i}' for i in range(1, 10)} | {f'LPT{i}' for i in range(1, 10)}
    encoded = []
    for part in parts:
        tail = len(part.rstrip(' .'))
        component = ''.join(f'%{ord(ch):02X}' if ch in '<>:"\\|?*%' or ord(ch) < 32 or (i >= tail and ch in ' .') else ch
                            for i, ch in enumerate(part))
        if component.split('.')[0].upper() in reserved:
            component = f'%{ord(component[0]):02X}' + component[1:]
        encoded.append(component)
    return safe_path(root, '/'.join(encoded))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def screenshot_identity(path):
    with Image.open(path) as image:
        rgb = image.convert('RGB')
        w, h = rgb.size
        pixel = hashlib.sha256(f'{w}x{h}:RGB:'.encode() + rgb.tobytes()).hexdigest()
        gray = np.asarray(rgb.convert('L').resize((17, 16), Image.Resampling.LANCZOS))
        dhash = np.packbits((gray[:, 1:] > gray[:, :-1]).ravel()).tobytes().hex()
    return dict(image_size=[w, h], pixel_sha256=pixel, dhash256=dhash, image_sha256=digest(path))


def pixel_box(value, image_size):
    """Return the visible intersection of a documented pixel-xyxy box."""
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in value):
        return None
    box = [float(x) for x in value]
    if not all(math.isfinite(x) for x in box) or box[0] >= box[2] or box[1] >= box[3]:
        return None
    w, h = image_size
    clipped = [max(0., min(float(w), box[0])), max(0., min(float(h), box[1])),
               max(0., min(float(w), box[2])), max(0., min(float(h), box[3]))]
    if clipped[0] >= clipped[2] or clipped[1] >= clipped[3]:
        return None
    return [clipped[0]/w, clipped[1]/h, clipped[2]/w, clipped[3]/h], clipped != box


def merge_annotations(annotations, image_size):
    """Equal referring expressions retain every valid source target region."""
    groups = {}; stats = Counter()
    for annotation in annotations:
        stats['raw_annotations'] += 1
        text = annotation.get('instruction')
        if not isinstance(text, str) or not text.strip():
            stats['empty_instruction'] += 1; continue
        result = pixel_box(annotation.get('bbox'), image_size)
        if result is None:
            stats['invalid_or_invisible_box'] += 1; continue
        box, clipped = result
        stats['visible_box_clipped'] += int(clipped)
        text = text.strip()
        # Preserve case and source wording; no generated target descriptions.
        group = groups.setdefault(text, dict(instruction=text, boxes=[], source_annotations=[]))
        if box not in group['boxes']:
            group['boxes'].append(box)
        ref = {k: v for k, v in annotation.items() if k not in ('instruction', 'bbox')}
        ref['raw_bbox_xyxy_pixels'] = annotation['bbox']
        ref['visible_intersection_clipped'] = clipped
        group['source_annotations'].append(ref)
    rows = list(groups.values())
    for row in rows:
        row['boxes'].sort()
    stats['merged_queries'] = len(rows)
    stats['queries_with_multiple_valid_boxes'] = sum(len(r['boxes']) > 1 for r in rows)
    return rows, dict(stats)


class NearIndex:
    """A BK tree for the same256-bit dHash/radius4 used by prior data splits."""
    def __init__(self):
        self.root = None

    def add(self, value, label):
        node = self.root
        if node is None:
            self.root = [value, label, {}]; return
        while True:
            distance = (value ^ node[0]).bit_count()
            if distance == 0:
                return
            child = node[2].get(distance)
            if child is None:
                node[2][distance] = [value, label, {}]; return
            node = child

    def find(self, value, radius=4):
        pending = [] if self.root is None else [self.root]
        while pending:
            node = pending.pop()
            distance = (value ^ node[0]).bit_count()
            if distance <= radius:
                return node[1]
            pending.extend(child for edge, child in node[2].items()
                           if distance-radius <= edge <= distance+radius)
        return None


def iou(a, b):
    overlap = max(0., min(a[2], b[2])-max(a[0], b[0])) * max(0., min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1])-overlap
    return overlap/union if union else 0.


def select_targets(rows, limit=4):
    chosen = []
    for row in sorted(rows, key=lambda r: hashlib.sha256(('20261103:target:'+r['sample_id']).encode()).hexdigest()):
        if all(all(iou(a, b) < .2 for a in row['target_geometry']['boxes'] for b in old['target_geometry']['boxes']) for old in chosen):
            chosen.append(row)
        if len(chosen) == limit:
            break
    return chosen


def reserve_groups(images):
    """Reserve source groups before training; mobile groups are episode proxies."""
    result = set()
    for source in sorted({r['dataset'] for r in images}):
        groups = sorted({r['source_group'] for r in images if r['dataset'] == source},
                        key=lambda g: hashlib.sha256(f'20261103:reserve:{source}:{g}'.encode()).hexdigest())
        n = min(len(groups)-1, max(1, math.ceil(len(groups)*.1))) if len(groups) > 1 else 0
        result.update((source, group) for group in groups[:n])
    return result
