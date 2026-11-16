"""Fixed, label-free V13 crop diversity and nested point quotas.

The first crop and c1 point coordinates reproduce the V12 single-crop policy.
Additional crops are chosen from the same frozen global coarse ranking, never
from labels or a verifier. Local lists consume their quota before deduplication.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math

from .candidate_budget import ranked_points
from .point_aligned_inference import bounded_box, unique_points


BUDGETS = (4, 8, 16, 32)
MAX_REGIONS = 3
MAX_CROP_IOU = .25


def box_iou(first, second):
    if len(first) != 4 or len(second) != 4:
        raise ValueError("Expected two xyxy boxes")
    a, b = [float(x) for x in first], [float(x) for x in second]
    if not all(math.isfinite(x) for x in a+b) or a[2] <= a[0] or a[3] <= a[1] or b[2] <= b[0] or b[3] <= b[1]:
        raise ValueError("Crop boxes must be finite with positive area")
    intersection = max(0., min(a[2], b[2])-max(a[0], b[0])) * max(0., min(a[3], b[3])-max(a[1], b[1]))
    return intersection / ((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection)


def crop_transform(image_size, parent_index, grid_shape):
    """Exactly the V12 proposal_crop geometry, without touching image pixels."""
    w, h = map(int, image_size); gh, gw = map(int, grid_shape); parent = int(parent_index)
    if min(w, h, gh, gw) <= 0 or not 0 <= parent < gh*gw:
        raise ValueError("Invalid image/grid size or parent")
    row, col = divmod(parent, gw)
    center = ((col+.5)*w/gw, (row+.5)*h/gh)
    box = bounded_box((w, h), center, (max(256, 4*w/gw), max(256, 4*h/gh)))
    return dict(box=box, scale=2, original_size=[w, h], crop_size=[2*(box[2]-box[0]), 2*(box[3]-box[1])],
                parent_index=parent, grid_shape=[gh, gw], ground_truth_used=False)


def plan_multiregion_crops(global_result, image_size, max_regions=MAX_REGIONS, max_iou=MAX_CROP_IOU):
    if max_regions not in (1, 2, 3) or not 0 <= max_iou <= 1:
        raise ValueError("At most three regions and IoU in[0,1] required")
    candidates = ranked_points(global_result, "global")
    selected = []
    for candidate in candidates:
        transform = crop_transform(image_size, candidate["parent_index"], global_result["grid_shape"])
        if any(box_iou(transform["box"], old["transform"]["box"]) > max_iou for old in selected):
            continue
        selected.append(dict(region_index=len(selected), transform=transform,
            global_candidate_index=candidate["candidate_index"], global_source_rank=candidate["source_rank"],
            global_rank_score=candidate["original_rank_score"]))
        if len(selected) == max_regions:
            break
    if selected and global_result.get("candidates"):
        if selected[0]["transform"]["parent_index"] != global_result["candidates"][0]["parent_index"]:
            raise ValueError("Global Top1 disagrees with canonical stable coarse ranking")
    return selected


def round_robin(lists, limit=None):
    """Take view0/rank1, view1/rank1,...; skip exhausted lists, no dedup."""
    if limit is not None and limit < 0:
        raise ValueError("limit cannot be negative")
    result = []
    for rank in range(max((len(values) for values in lists), default=0)):
        for values in lists:
            if rank < len(values) and (limit is None or len(result) < limit):
                result.append(values[rank])
        if limit is not None and len(result) >= limit:
            break
    return result


def build_multiregion_pools(global_result, local_results, transforms, *, condition="c3", budgets=BUDGETS):
    if len(local_results) != len(transforms) or len(local_results) > MAX_REGIONS:
        raise ValueError("Aligned zero-to-three local results/transforms required")
    if condition not in ("c1", "c3"):
        raise ValueError("condition must be c1 or c3")
    budgets = tuple(int(k) for k in budgets)
    if not budgets or budgets != tuple(sorted(set(budgets))) or any(k <= 0 or k % 2 for k in budgets):
        raise ValueError("Increasing distinct positive even budgets required")
    global_points = ranked_points(global_result, "global")
    count = min(1 if condition == "c1" else 3, len(local_results))
    local_lists = [ranked_points(local_results[i], "old_crop" if i == 0 else f"local_{i}", transforms[i]) for i in range(count)]
    local_points = round_robin(local_lists, max(budgets)//2)
    result = {}; previous = set()
    for budget in budgets:
        pool = unique_points(global_points[:budget//2], local_points[:budget//2], limit=budget)
        coordinates = {tuple(item["point"]) for item in pool}
        if not previous <= coordinates:
            raise AssertionError("Candidate pools must be nested")
        previous = coordinates; result[str(budget)] = pool
    return result


def build_c1_c3_pools(global_result, local_views):
    results = [view["result"] for view in local_views]
    transforms = [view["transform"] for view in local_views]
    return {condition: build_multiregion_pools(global_result, results, transforms, condition=condition) for condition in ("c1", "c3")}


def public_raw_identity(sample_id, image_sha256, instruction, transform, runtime_contract_sha256):
    """Identity depends only on the frozen Qwen input and exact crop geometry."""
    value = dict(sample_id=sample_id, image_sha256=image_sha256,
        instruction_sha256=hashlib.sha256(instruction.encode()).hexdigest(), transform=transform,
        runtime_contract_sha256=runtime_contract_sha256)
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
