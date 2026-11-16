"""Deterministic public-only changes to decoded points, not learned weights."""
from copy import deepcopy
import numpy as np

MODES = ('original', 'coarse_center', 'half_to_center', 'child_center')


def relocate(result, image_size, mode):
    assert mode in MODES
    r = deepcopy(result)
    if mode == 'original' or not r['all_points']:
        return r
    gh, gw = map(int, r['grid_shape']); w, h = image_size
    points = np.asarray(r['all_points'], np.float64)
    row, col = np.divmod(np.asarray(r['parent_indices'], np.int64), gw)
    center = np.stack(((col+.5)*w/gw, (row+.5)*h/gh), -1)
    if mode == 'coarse_center':
        changed = center
    elif mode == 'half_to_center':
        changed = (points+center)/2
    else:
        # Infer the child containing the archived decoded point; no label lookup.
        fx = np.clip(points[:, 0]*gw/w-col, 0, 1)
        fy = np.clip(points[:, 1]*gh/h-row, 0, 1)
        changed = np.stack(((col+.25+.5*(fx >= .5))*w/gw,
                            (row+.25+.5*(fy >= .5))*h/gh), -1)
    changed = np.clip(np.rint(changed), 0, np.asarray(image_size)-1).astype(np.int64).tolist()
    r['all_points'] = changed
    for candidate in r['candidates']:
        candidate['point'] = changed[candidate['candidate_index']]
    r['point'] = r['candidates'][0]['point'] if r['candidates'] else None
    return r


def pipeline_geometry(pipeline, image_size, mode, pool_builder):
    p = deepcopy(pipeline)
    p['global_result'] = relocate(p['global_result'], image_size, mode)
    for view in p['local_views']:
        view['result'] = relocate(view['result'], view['transform']['crop_size'], mode)
    p['pools'] = pool_builder(p['global_result'], p['local_views'])
    return p
