"""Label-free, nested source quotas over already decoded candidate points."""
from __future__ import annotations

import numpy as np

from .point_aligned_inference import map_crop_point, unique_points

BUDGETS = (4, 8, 16, 32)


def ranked_points(result, source, transform=None):
    if not result or not result.get('all_points'):
        return []
    points = np.asarray(result['all_points'])
    scores = np.asarray(result['all_scores'], dtype=np.float64).reshape(-1)
    parents = result['parent_indices']
    if points.shape != (len(scores), 2) or len(parents) != len(scores):
        raise ValueError('unaligned archived candidate arrays')
    if not np.isfinite(points).all() or not np.isfinite(scores).all() or not np.equal(points, np.rint(points)).all():
        raise ValueError('nonfinite scores or noninteger points')
    rows = []
    for rank, index in enumerate(np.argsort(-scores, kind='stable'), 1):
        point = points[index].astype(int).tolist()
        item = dict(point=map_crop_point(point, transform) if transform else point,
                    source=source, source_rank=rank, candidate_index=int(index),
                    parent_index=int(parents[index]), original_rank_score=float(scores[index]))
        if transform:
            item['local_point'] = point
        rows.append(item)
    return rows


def build_pools(pipeline, budgets=BUDGETS):
    global_points = ranked_points(pipeline['global_result'], 'global')
    local_points = ranked_points(pipeline.get('old_crop_result'), 'old_crop', pipeline.get('old_crop_transform'))
    pools = {}
    previous = set()
    for budget in budgets:
        if budget <= 0 or budget % 2:
            raise ValueError('positive even source quota required')
        # Slice BEFORE deduplication, exactly preserving v9 2+2 policy at K=4.
        pool = unique_points(global_points[:budget // 2], local_points[:budget // 2], limit=budget)
        coordinates = {tuple(item['point']) for item in pool}
        assert previous <= coordinates
        previous = coordinates
        pools[str(budget)] = pool
    return pools


def coordinate_key(point):
    return ','.join(map(str, point))


def choose_point(pool, records):
    """Highest fixed True/False log-odds; ties use the original pool order."""
    if not pool:
        return dict(point=None, score=None, margin=None, exact_top_ties=0, source=None, source_rank=None)
    values = np.asarray([records[coordinate_key(item['point'])]['log_odds'] for item in pool], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError('nonfinite verifier scores')
    index = int(values.argmax())
    sorted_values = np.sort(values)
    return dict(point=pool[index]['point'], score=float(values[index]),
                margin=float(sorted_values[-1] - sorted_values[-2]) if len(values) > 1 else None,
                exact_top_ties=int((values == values[index]).sum()), source=pool[index]['source'],
                source_rank=pool[index]['source_rank'])
