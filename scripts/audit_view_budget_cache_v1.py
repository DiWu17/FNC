"""Label-free audit for prefix-view reassembly from archived GUI caches.

No Qwen inference and no new proposer coordinates. The only candidate operation
is selection of archived per-view proposals under a shorter view prefix.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from collections import Counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from clickagent.candidate_budget import ranked_points
from clickagent.evidence_retention import build_base
from clickagent.multiregion_proposals import build_multiregion_pools
from clickagent.point_aligned_inference import unique_points

PUBLIC = {
    "pro": ROOT / "runs/paper_scale_ablation_v2/eval/pro/features/index.json",
    "sheets": ROOT / "runs/paper_scale_ablation_v2/eval/sheets/features/index.json",
    "mmbench": ROOT / "runs/mmbench_gui_new_source_v1/mmbench/features/index.json",
}
TRAIN = {
    "old_train": ROOT / "runs/full_scaling_v18/data/old_train_index.json",
    "new_train": ROOT / "runs/full_scaling_v18/features/new_train/index.json",
}
OUT = ROOT / "runs/review_view_budget_cached_v1"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def entries(path):
    value = read(path)
    return value if isinstance(value, list) else value["cases"]


def archived_pool(pipeline):
    return pipeline["pools"]["c3"]["16"]


def pool_prefix(pipeline, m, *, global_top=8):
    require(m in range(4), "m must be 0..3")
    if m == 0:
        return unique_points(ranked_points(pipeline["global_result"], "global")[:global_top], limit=global_top)
    local = pipeline["local_views"][:m]
    return build_multiregion_pools(
        pipeline["global_result"], [v["result"] for v in local],
        [v["transform"] for v in local], condition="c3", budgets=(16,)
    )["16"]


def masked_base(base, pipeline, m):
    value = np.asarray(base, np.float32).copy()
    require(value.ndim == 2 and value.shape[1] == 32, "base32 required")
    for view in range(m + 1, 4):
        value[:, 7 * view:7 * view + 7] = 0
    value[:, 30] = value[:, [0, 7, 14, 21]].sum(axis=1) / 4
    result_views = [pipeline["global_result"]] + [v["result"] for v in pipeline["local_views"][:m]]
    value[:, 31] = sum(bool(v.get("all_points")) for v in result_views) / 4
    return value


def audit_case(entry):
    p = read(entry["pipeline_path"])
    require(p["sample_id"] == entry["sample_id"], "pipeline sample ID drift")
    with np.load(entry["path"], allow_pickle=False) as z:
        points = z["points"]
        base = z["base"]
        active = z["active"]
        baseline = z["baseline"]
    require(points.shape == (entry["n"], 2) and base.shape == (entry["n"], 32), "feature shape drift")
    lookup = {tuple(point): i for i, point in enumerate(points.tolist())}
    require(len(lookup) == len(points), "full union has duplicate coordinate")
    m3 = pool_prefix(p, 3)
    require(m3 == archived_pool(p), "m3 prefix pool differs from archived C3 K16")
    indices = np.asarray([lookup[tuple(v["point"])] for v in m3], np.int64)
    require(np.array_equal(indices, baseline), "m3 K16 index differs from archived")
    require(np.array_equal(masked_base(base, p, 3), base), "m3 base differs from archived")
    size = json.loads(np.load(p["global_feature_path"], allow_pickle=False)["metadata"].item())["image_size"]
    rebuilt = build_base(p, size)
    require(np.array_equal(points, rebuilt["points"]) and np.array_equal(base, rebuilt["base"])
            and np.array_equal(active, rebuilt["active"])
            and np.array_equal(baseline, rebuilt["baseline"]), "full build_base parity failure")
    counts = {}
    for m in range(4):
        pools = {"top8": pool_prefix(p, 0, global_top=8), "top16": pool_prefix(p, 0, global_top=16)} if m == 0 else {"k16": pool_prefix(p, m)}
        for kind, pool in pools.items():
            require(all(tuple(item["point"]) in lookup for item in pool), "prefix coordinate missing from union")
            require(all(item["source"] == "global" or (m >= 1 and
                        (item["source"] == "old_crop" or
                         item["source"] in {f"local_{v}" for v in range(1, m)}))
                        for item in pool), "future source in prefix pool")
            require(len(pool) <= (16 if kind != "top8" else 8), "pool budget overflow")
            counts[f"m{m}_{kind}"] = len(pool)
        # On every prefix coordinate, base32 must match a fresh prefix build,
        # including the strict global-only m=0 case.
        local = p["local_views"][:m]
        partial = {"global_result": p["global_result"], "local_views": local,
                   "pools": {"c3": {"16": pool_prefix(p, m)}}}
        prefix = build_base(partial, size)
        pm = masked_base(base, p, m)
        require(all(tuple(point) in lookup for point in prefix["points"]), "prefix universe outside archived union")
        ix = [lookup[tuple(point)] for point in prefix["points"]]
        require(np.array_equal(pm[ix], prefix["base"]) and
                np.array_equal(active[ix, :m + 1], prefix["active"][:, :m + 1]),
                "prefix base/active parity failure")
    return {"sample_id": entry["sample_id"], "n": len(points), "local_views": len(p["local_views"]), "pool_counts": counts,
            "pipeline_sha256": entry.get("pipeline_sha256"), "feature_sha256": entry.get("sha256")}


def main():
    if os.name == "nt":
        import ctypes
        kernel = ctypes.windll.kernel32
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        kernel.SetPriorityClass.argtypes = (ctypes.c_void_p, ctypes.c_uint32)
        kernel.SetPriorityClass(kernel.GetCurrentProcess(), 0x4000)
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=(*PUBLIC, *TRAIN, "all_test"), default="all_test")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    paths = {**PUBLIC, **TRAIN}
    chosen = list(PUBLIC) if args.cohort == "all_test" else [args.cohort]
    reports = {}
    for name in chosen:
        rows = entries(paths[name])
        if args.limit:
            rows = rows[:args.limit]
        totals = Counter(); local = Counter(); count = 0
        for i, entry in enumerate(rows, 1):
            result = audit_case(entry)
            count += 1; local[result["local_views"]] += 1; totals.update(result["pool_counts"])
            if i % 100 == 0:
                print(json.dumps({"cohort": name, "audited": i, "total": len(rows)}), flush=True)
        reports[name] = {"n": count, "local_view_histogram": dict(local),
                         "mean_pool_counts": {k: v / count for k, v in totals.items()},
                         "index_sha256": sha(paths[name])}
    output = {"complete": True, "label_access": False, "no_qwen_inference": True,
              "script_sha256": sha(__file__), "reports": reports}
    file = OUT / (f"audit_{args.cohort}_{args.limit}.json" if args.limit else f"audit_{args.cohort}.json")
    write(file, output)
    print(json.dumps({"output": str(file), "reports": reports}), flush=True)


if __name__ == "__main__":
    main()
