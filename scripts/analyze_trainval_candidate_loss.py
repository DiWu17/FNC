"""Read-only V13 train/val diagnostics in a new directory; never run a model.

Only train/val data, pipelines, cache scalar members and raw metadata/NPY headers
are consumed. No test predictions, raw tensor values or model weights are read.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import html
import io
import json
import math
import os
from pathlib import Path
import sys
import zipfile

import numpy as np
from shapely.geometry import box
from shapely.ops import unary_union

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "src/clickagent/public_evaluation.py").is_file()
            and (p / "runs/multiregion_v13").is_dir())
sys.path.insert(0, str(ROOT / "src"))
from clickagent.public_evaluation import point_in_geometry
from clickagent.spatial_geometry import shape_from_geometry

BUDGETS = (4, 8, 16, 32)
BRANCHES = ("global_no_name", "no_crop_overlap", "overlap_all_no_name", "overlap_named_no_hit")
LABELS = {
    "global_no_name": "全图无名称，未启动局部搜索",
    "no_crop_overlap": "有全图名称，但所有裁剪均无正面积目标交集",
    "overlap_all_no_name": "有目标交集的裁剪全部无名称",
    "overlap_named_no_hit": "存在带名称的目标交集裁剪，但全部点仍未命中",
    "quota_loss": "已有正确解码点，但 K16 全部丢失",
    "reachable": "K16 保留正确点",
}
CASE_NOTES = {
    "salesforce:a7f5ce23-ac92-43f0-adae-66111ac7c8fd":
        "人工看图：局部1包含完整省略号按钮，但 Microsoft Photos 行名称位于裁剪框外。"
        "该局部输出 NONE，另一处无目标的局部却命名为 Microsoft Photos app tile。"
        "这支持将上下文缺失与错目标命名作为待验证假设，尚不能仅凭此例确定因果。",
    "salesforce:31b7911f-838d-4c1b-a545-9e44b1a9d58c":
        "人工看图：目标是密集日历中的单个日期格。局部3完整覆盖目标，生成 March 3rd contribution square，"
        "两个候选父格与目标有正面积交集，但所有具体点均未命中。名称复述正确不证明日期定位正确；"
        "此例适合进一步检查父格内读出及日期语义。",
    "salesforce:0b9de973-0c3f-48d8-ad40-eac65343cf67":
        "人工看图：页面存在多个 X。GT 位于左下弹窗，搜索框主要落在右上侧栏。"
        "全图第17名点能命中 GT，固定全图8点配额未保留它；K32 的全图16点也仍排除它。"
        "这是已有坐标被排序和名额丢失的案例，同时存在目标指令歧义，不能只归因于几何分辨率。",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(path):
    return os.path.normcase(os.path.abspath(path))


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def dumps(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def case_key(sid):
    return hashlib.sha256(sid.encode()).hexdigest()[:24]


class Reader:
    def __init__(self, expected):
        self.expected = {canonical(p): digest for p, digest in expected.items()}
        self.files = {}
        self.members = {}

    def touch(self, path):
        path = Path(path)
        key = canonical(path)
        stat = path.stat()
        current = (stat.st_size, stat.st_mtime_ns)
        if key in self.files:
            require(self.files[key]["stat"] == current, f"Input changed: {path}")
        else:
            self.files[key] = dict(path=str(path), stat=current,
                                   audited_container_sha256=self.expected.get(key))
        return path, key

    def full(self, path):
        path, key = self.touch(path)
        data = path.read_bytes()
        digest = sha_bytes(data)
        require(key in self.expected, f"Input not bound to V13 audit: {path}")
        require(digest == self.expected[key], f"V13 input hash changed: {path}")
        self.files[key]["verified_full_sha256"] = digest
        self.touch(path)
        return data

    def json(self, path):
        return json.loads(self.full(path).decode("utf-8-sig"))

    def npy(self, archive, member, *, header_only=False):
        path, key = self.touch(archive)
        with zipfile.ZipFile(path) as z, z.open(member + ".npy") as stream:
            if header_only:
                magic = np.lib.format.read_magic(stream)
                require(magic in ((1, 0), (2, 0)), "Unsupported NPY header")
                parser = (np.lib.format.read_array_header_1_0 if magic == (1, 0)
                          else np.lib.format.read_array_header_2_0)
                shape, _, dtype = parser(stream)
                length = stream.tell()
                with z.open(member + ".npy") as again:
                    data = again.read(length)
                value = (shape, dtype)
            else:
                data = stream.read()
                value = np.load(io.BytesIO(data), allow_pickle=False)
        mode = "header" if header_only else "member"
        self.members[f"{key}::{member}::{mode}"] = dict(path=str(path), member=member,
            mode=mode, consumed_bytes=len(data), consumed_sha256=sha_bytes(data))
        self.touch(path)
        return value

    def unchanged(self):
        for item in list(self.files.values()):
            self.touch(item["path"])


def hit(point, case):
    w, h = case["image_size"]
    return bool(point_in_geometry([point[0] / w, point[1] / h], case["target_geometry"]))


def map_point(point, transform):
    if transform is None:
        return list(point)
    x0, y0, _, _ = transform["box"]
    w, h = transform["original_size"]
    scale = transform["scale"]
    return [min(w - 1, max(0, round(point[0] / scale + x0))),
            min(h - 1, max(0, round(point[1] / scale + y0)))]


def ranked(result, source, transform=None):
    points = result["all_points"]
    scores = result.get("all_scores", [])
    parents = result.get("parent_indices", [])
    require(len(points) == len(scores) == len(parents), "Unaligned decoded arrays")
    require(all(math.isfinite(s) for s in scores), "Nonfinite rank score")
    output = []
    for rank, index in enumerate(sorted(range(len(points)), key=lambda i: -scores[i]), 1):
        require(len(points[index]) == 2 and all(math.isfinite(x) and x == round(x) for x in points[index]),
                "Invalid archived integer point")
        output.append(dict(point=map_point(points[index], transform), source=source,
            source_rank=rank, candidate_index=index, parent_index=parents[index],
            original_rank_score=scores[index]))
    return output


def rr(groups):
    return [group[i] for i in range(max(map(len, groups), default=0)) for group in groups if i < len(group)]


def unique(rows):
    seen = set()
    output = []
    for row in rows:
        key = tuple(row["point"])
        if key not in seen:
            output.append(row)
            seen.add(key)
    return output


def upstream_branch(global_tokens, crops):
    if global_tokens == 0:
        return "global_no_name"
    covered = [c for c in crops if c["target_area_fraction"] > 0]
    if not covered:
        return "no_crop_overlap"
    if all(c["tokens"] == 0 for c in covered):
        return "overlap_all_no_name"
    return "overlap_named_no_hit"


def view_info(reader, case, result, path, expected_sha, transform=None):
    _, key = reader.touch(path)
    require(reader.expected.get(key) == expected_sha, "Raw reference differs from final audit")
    meta = json.loads(str(reader.npy(path, "metadata")))
    shape, _ = reader.npy(path, "target_attention", header_only=True)
    require(len(shape) == 3 and shape[1] == 96, "Expected [T,96,P] target attention")
    tokens = int(shape[0])
    require(tokens == len(meta["target_raw_indices"]), "Target token metadata/header mismatch")
    require((tokens == 0) == (result.get("failure") == "no_name"), "Name status mismatch")
    require(tokens > 0 or not result["all_points"], "Points decoded without name tokens")
    require(meta["target_text"] == result["target_name"], "Archived name mismatch")
    require(list(meta["grid_shape"]) == list(result["grid_shape"]), "Grid identity mismatch")
    require(shape[2] == math.prod(meta["grid_shape"]), "Image grid/header mismatch")
    require(meta.get("ground_truth_in_model_input") is False, "GT in raw model input")
    if transform is not None:
        require(meta["transform"] == transform, "Raw crop transform mismatch")
    gh, gw = result["grid_shape"]
    w, h = case["image_size"]
    shape_gt = shape_from_geometry(case["target_geometry"])
    crop = box(0, 0, 1, 1) if transform is None else box(*[
        transform["box"][0] / w, transform["box"][1] / h,
        transform["box"][2] / w, transform["box"][3] / h])
    intersection = shape_gt.intersection(crop)
    # Parent cells are support regions, not predicted UI element boxes.
    cx0, cy0, cx1, cy1 = crop.bounds
    parent_hits = 0
    for parent in set(result.get("parent_indices", [])):
        require(0 <= parent < gh * gw, "Invalid parent index")
        row, col = divmod(parent, gw)
        cell = box(cx0 + col / gw * (cx1 - cx0), cy0 + row / gh * (cy1 - cy0),
                   cx0 + (col + 1) / gw * (cx1 - cx0), cy0 + (row + 1) / gh * (cy1 - cy0))
        parent_hits += shape_gt.intersection(cell).area > 0
    source = "global" if transform is None else "old_crop"
    rows = ranked(result, source, transform)
    good = [r for r in rows if hit(r["point"], case)]
    return dict(tokens=tokens, name=meta["target_text"], action=meta.get("action"),
        no_name=tokens == 0, terminated_by_eos=meta.get("terminated_by_eos"),
        token_cap_reached=meta.get("token_cap_reached"),
        target_area_fraction=float(intersection.area / shape_gt.area),
        target_fully_contained=bool(crop.covers(shape_gt)),
        boundary_contact=bool(intersection.area == 0 and not intersection.is_empty),
        parent_cells_intersecting_target=int(parent_hits), decoded_count=len(rows),
        valid_decoded_count=len(good), first_valid_rank=min((r["source_rank"] for r in good), default=None),
        box=list(transform["box"]) if transform else [0, 0, w, h],
        grid_shape=list(meta["grid_shape"]),
        input_size=list(meta["image_size"]), generated_text=meta["generated_text"],
        feature_path=str(path), transform=transform)


def analyze_case(case, pipe, infos):
    require(pipe["complete"] and pipe["sample_id"] == case["sample_id"], "Wrong/incomplete pipeline")
    require(pipe["ground_truth_in_model_inputs"] is False, "GT in pipeline model input")
    global_rows = ranked(pipe["global_result"], "global")
    locals_ = [ranked(v["result"], "old_crop" if i == 0 else f"local_{i}", v["transform"])
               for i, v in enumerate(pipe["local_views"])]
    pools = {}
    for condition in ("c1", "c3"):
        local_order = rr(locals_[:1 if condition == "c1" else 3])
        pools[condition] = {}
        for k in BUDGETS:
            rebuilt = unique(global_rows[:k // 2] + local_order[:k // 2])
            original = pipe["pools"][condition][str(k)]
            require([r["point"] for r in rebuilt] == [r["point"] for r in original], "Pool coordinates differ")
            for a, b in zip(rebuilt, original):
                require(all(a[key] == b[key] for key in a), "Pool provenance differs")
            pools[condition][str(k)] = any(hit(r["point"], case) for r in rebuilt)
    raw_rows = unique(global_rows + rr(locals_))
    raw_good = [r for r in raw_rows if hit(r["point"], case)]
    pool16 = pipe["pools"]["c3"]["16"]
    stage = ("reachable" if pools["c3"]["16"] else "quota_loss" if raw_good
             else upstream_branch(infos[0]["tokens"], infos[1:]))
    selected_coords = {tuple(r["point"]) for r in pool16}
    dropped = [r for r in raw_good if tuple(r["point"]) not in selected_coords]
    target = shape_from_geometry(case["target_geometry"])
    components = list(target.geoms) if hasattr(target, "geoms") else [target]
    dominant = max(components, key=lambda s: s.area)
    x0, y0, x1, y1 = dominant.bounds
    gh, gw = pipe["global_result"]["grid_shape"]
    short = min((x1 - x0) * gw * 32, (y1 - y0) * gh * 32)
    target_bin = "<4" if short < 4 else "4-8" if short < 8 else "8-16" if short < 16 else ">=16"
    local_order = rr(locals_)
    local_order_by_coord = {}
    for rank, item in enumerate(local_order, 1):
        local_order_by_coord.setdefault(tuple(item["point"]), rank)
    for row in dropped:
        row["local_round_robin_rank"] = local_order_by_coord.get(tuple(row["point"]))
    crop_union = unary_union([box(c["box"][0] / case["image_size"][0], c["box"][1] / case["image_size"][1],
                                 c["box"][2] / case["image_size"][0], c["box"][3] / case["image_size"][1])
                             for c in infos[1:]])
    # A refill bound counts only empty slots; it cannot replace any existing point.
    refill_bound = len(pool16) < 16 and any(tuple(r["point"]) not in selected_coords for r in raw_good)
    return dict(sample_id=case["sample_id"], split=case["split"], source=case["dataset"],
        image=case["image"], image_size=case["image_size"], image_sha256=case["image_sha256"],
        rgb_group=case["pixel_sha256"], instruction=case["instruction"], target_geometry=case["target_geometry"],
        stage=stage, raw_reachable=bool(raw_good), raw_count=len(raw_rows),
        global_reachable=infos[0]["valid_decoded_count"] > 0,
        local_reachable=any(v["valid_decoded_count"] > 0 for v in infos[1:]),
        c1_raw_reachable=any(hit(r["point"], case) for r in global_rows + (locals_[0] if locals_ else [])),
        pools=pools, k16_count=len(pool16), unused_k16_slots=16 - len(pool16),
        duplicate_slots=len(global_rows[:8]) + len(local_order[:8]) - len(pool16),
        quota_input_shortfall=16 - len(global_rows[:8]) - len(local_order[:8]),
        best_possible_refill_rescue=bool(refill_bound and not pools["c3"]["16"]),
        best_possible_refill_adds_correct_point=bool(refill_bound),
        target_area_fraction=float(target.area), global_input_target_short_side_px=short,
        target_size_bin=target_bin,
        crop_union_target_area_fraction=float(target.intersection(crop_union).area / target.area),
        global_view=infos[0], crops=infos[1:], dropped_valid_points=dropped,
        quota_pool=pool16)


def counts(rows):
    stages = Counter(r["stage"] for r in rows)
    quota_losses = [r for r in rows if r["stage"] == "quota_loss"]
    crops = [v for r in rows for v in r["crops"]]
    covered = [v for v in crops if v["target_area_fraction"] > 0]
    full = [v for v in crops if v["target_fully_contained"]]
    absent = [v for v in crops if v["target_area_fraction"] == 0]
    named = [v for v in covered if v["tokens"] > 0]
    global_ranks = [r["global_view"]["first_valid_rank"] for r in quota_losses
                    if r["global_view"]["first_valid_rank"] is not None]
    local_ranks = [min(p["local_round_robin_rank"] for p in r["dropped_valid_points"]
                       if p["local_round_robin_rank"] is not None) for r in quota_losses
                   if any(p["local_round_robin_rank"] is not None for p in r["dropped_valid_points"])]
    rank_summary = lambda values: dict(n=len(values), minimum=min(values) if values else None,
        median=float(np.median(values)) if values else None, maximum=max(values) if values else None)
    return dict(n=len(rows), rgb_groups=len({r["rgb_group"] for r in rows}),
        stages={k:stages[k] for k in (*BRANCHES, "quota_loss", "reachable")},
        decoded_union=sum(r["raw_reachable"] for r in rows),
        c1_decoded_union=sum(r["c1_raw_reachable"] for r in rows),
        global_decoded_union=sum(r["global_reachable"] for r in rows),
        local_decoded_union=sum(r["local_reachable"] for r in rows),
        pool_coverage={c:{str(k):sum(r["pools"][c][str(k)] for r in rows) for k in BUDGETS}
                       for c in ("c1", "c3")},
        c3_k16_gains_over_c1=sum(r["pools"]["c3"]["16"] and not r["pools"]["c1"]["16"] for r in rows),
        c3_k16_losses_to_c1=sum(r["pools"]["c1"]["16"] and not r["pools"]["c3"]["16"] for r in rows),
        k32_rescues_among_k16_losses=sum(r["pools"]["c3"]["32"] for r in quota_losses),
        quota_loss_global_only=sum(r["global_reachable"] and not r["local_reachable"] for r in quota_losses),
        quota_loss_local_only=sum(r["local_reachable"] and not r["global_reachable"] for r in quota_losses),
        quota_loss_both=sum(r["global_reachable"] and r["local_reachable"] for r in quota_losses),
        quota_loss_first_valid_rank=dict(global_list=rank_summary(global_ranks), local_round_robin=rank_summary(local_ranks)),
        pool_underfull=sum(r["k16_count"] < 16 for r in rows),
        unused_slots=sum(r["unused_k16_slots"] for r in rows),
        duplicate_slots=sum(r["duplicate_slots"] for r in rows),
        quota_input_shortfall=sum(r["quota_input_shortfall"] for r in rows),
        best_possible_refill_rescues=sum(r["best_possible_refill_rescue"] for r in rows),
        mean_k16_points=float(np.mean([r["k16_count"] for r in rows])) if rows else None,
        crops=dict(n=len(crops), overlap=len(covered), full=len(full), absent=len(absent),
            no_name=sum(c["no_name"] for c in crops),
            overlap_no_name=sum(c["no_name"] for c in covered), full_no_name=sum(c["no_name"] for c in full),
            absent_no_name=sum(c["no_name"] for c in absent),
            overlap_named=len(named), overlap_named_point_hit=sum(c["valid_decoded_count"] > 0 for c in named),
            named_parent_support_but_point_miss=sum(c["parent_cells_intersecting_target"] > 0 and c["valid_decoded_count"] == 0 for c in named),
            no_name_eos=sum(c["terminated_by_eos"] is True for c in crops if c["no_name"]),
            no_name_token_cap=sum(c["token_cap_reached"] is True for c in crops if c["no_name"])),
        per_region=[dict(region=i + 1, n=len(vs), overlap=sum(c["target_area_fraction"] > 0 for c in vs),
                         no_name=sum(c["no_name"] for c in vs), hit=sum(c["valid_decoded_count"] > 0 for c in vs))
                    for i in range(3) if (vs := [r["crops"][i] for r in rows if len(r["crops"]) > i])])


def verify_completion(run):
    record = read_json(run / "completion.json")
    verified = {}
    for base, group in ((run, "artifact_sha256"), (ROOT, "docs_sha256")):
        for rel, expected in record[group].items():
            path = base / rel
            digest = sha_bytes(path.read_bytes())
            require(digest == expected, f"Completion record mismatch: {path}")
            verified[str(path)] = digest
    return verified


def plot_summary(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = ["Global no name", "Crops miss target", "Covered crops: no name",
              "Covered + named: no point", "Quota loses correct point", "K16 reachable"]
    colors = ["#64748b", "#ef8354", "#d8a145", "#ab6fa4", "#d54e60", "#2e8b74"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained")
    for ax, split in zip(axes, ("train", "val")):
        row = summary[split]
        vals = list(row["stages"].values())
        bars = ax.barh(labels, vals, color=colors)
        ax.bar_label(bars, labels=[f"{n} ({n / row['n']:.1%})" for n in vals], padding=5, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlim(0, max(vals) * 1.32)
        ax.set_title(f"{split}: n={row['n']}, RGB groups={row['rgb_groups']}")
        ax.set_xlabel("Queries (descriptive diagnostics)")
        ax.spines[["top", "right"]].set_visible(False)
    path = output / "loss_breakdown.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)


def example_image(row, path):
    from PIL import Image
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Polygon
    require(sha_bytes(Path(row["image"]).read_bytes()) == row["image_sha256"], "Example image identity changed")
    with Image.open(row["image"]) as image:
        require(list(image.size) == row["image_size"], "Example image size mismatch")
        fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
        ax.imshow(image)
    w, h = row["image_size"]
    geom = row["target_geometry"]
    for bounds in geom.get("boxes", []):
        x0, y0, x1, y1 = bounds
        ax.add_patch(Rectangle((x0 * w, y0 * h), (x1 - x0) * w, (y1 - y0) * h,
                               fill=False, edgecolor="#00ef74", linewidth=2.2, zorder=10))
    for polygon in geom.get("polygons", []):
        ax.add_patch(Polygon([(x * w, y * h) for x, y in polygon], fill=False, edgecolor="#00ef74", linewidth=2.2, zorder=10))
    for i, crop in enumerate(row["crops"]):
        x0, y0, x1, y1 = crop["box"]
        color = ("#ffb300", "#00c7ff", "#e36fff")[i]
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=color, linewidth=1.7))
        ax.text(x0, y0, f"C{i+1}", color="black", bbox=dict(facecolor=color, alpha=.85, edgecolor="none"), fontsize=9)
    points = row["quota_pool"]
    if points:
        ax.scatter(*np.array([r["point"] for r in points]).T, marker="x", color="#ff334f", s=22)
    if row["stage"] == "quota_loss":
        points = row["dropped_valid_points"][:20]
        ax.scatter(*np.array([r["point"] for r in points]).T, facecolors="none", edgecolors="#00ef74", s=85)
    ax.set_axis_off()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def table(headers, rows):
    return "<table><thead><tr>" + "".join(f"<th>{html.escape(str(x))}</th>" for x in headers) + "</tr></thead><tbody>" + "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(x))}</td>" for x in row) + "</tr>" for row in rows) + "</tbody></table>"


def report(summary, all_rows, output):
    train, val = summary["train"], summary["val"]
    chunks = ["<h1>V13 训练 / 验证：候选丢失分析</h1>",
        "<p>2026-11-03 · 873 训练 / 172 验证 · 缓存诊断 · 无模型前向</p>",
        "<h2>当前证据支持的判断</h2>",
        f"<p>训练的全部解码点可达 {train['decoded_union']}/873，K16 为 {train['stages']['reachable']}/873；"
        f"验证分别为 {val['decoded_union']}/172 与 {val['stages']['reachable']}/172。"
        f"验证中尚不可达的 {val['n']-val['stages']['reachable']} 题里，"
        f"{val['n']-val['decoded_union']} 题在当前全部坐标中就没有正确点，另 {val['stages']['quota_loss']} 题已有点却被截掉。"
        "优先检查目标相关搜索，同时把保留已有正确点作为单独、可归因的改动。</p>",
        f"<p>验证局部无名称 {val['crops']['no_name']} 次，其中 {val['crops']['absent_no_name']} 次没有正面积目标交集；"
        f"完整包含目标的 {val['crops']['full']} 个局部中仅 {val['crops']['full_no_name']} 个无名称。"
        "这与大量搜索框落在目标之外相符，不能将全部无名称都当作局部读取器失效。"
        "仍应对完整可见却无名称的少数案例检查字段名称与窗口上下文。</p>",
        f"<p>只补空位的最佳可能新增可达题数：训练 {train['best_possible_refill_rescues']}，验证 {val['best_possible_refill_rescues']}。"
        f"验证去重仅吞掉 {val['duplicate_slots']} 个槽位；大部分空位来自一侧候选不足。"
        "因此下一步 K16 对照需要研究替换满池中的低价值点、全图/局部配额分配，不能把补位视为主要解法。"
        f"K32 仅恢复 {val['k32_rescues_among_k16_losses']}/{val['stages']['quota_loss']} 个验证配额丢失案例，扩大预算也不能自动保留全部已有正确点。</p>",
        "<p>建议的后续对照：先固定本轮全部已解码坐标，比较原配额、仅补位和可替换已有点的 K16 分配，分开报告可达性与同池选择；"
        "另对完整可见却无名称的局部，在相同裁剪框和像素预算下检验上下文读取。"
        "这些是本轮诊断提出的假设，本次尚未实现或选择新策略，也未据测试错例调参。</p>",
        "<p>重点是定位已有坐标在哪一步失去可达性。下表为互斥观测分支，不能把裁剪遗漏、无名称和解码失败解释为已隔离的单一因果。"
        "可达性使用 GT 事后几何判分，是候选集合诊断，不是最终点击准确率。</p>",
        table(["观测分支", "训练 / 873", "验证 / 172"], [[LABELS[k], f"{train['stages'][k]} ({train['stages'][k]/873:.1%})",
              f"{val['stages'][k]} ({val['stages'][k]/172:.1%})"] for k in (*BRANCHES, "quota_loss", "reachable")]),
        '<img class="chart" src="loss_breakdown.png" alt="Candidate loss breakdown">',
        "<h2>搜索范围、候选预算与补位上限</h2>",
        table(["候选集合 / 诊断", "训练", "验证"],
            [["全图全部解码点",train["global_decoded_union"],val["global_decoded_union"]],
             ["单局部全部点并集",train["c1_decoded_union"],val["c1_decoded_union"]],
             ["三局部全部点并集",train["decoded_union"],val["decoded_union"]]] +
            [[f"{c} K{k} 可达",train["pool_coverage"][c][str(k)],val["pool_coverage"][c][str(k)]] for c in ("c1","c3") for k in BUDGETS] +
            [["三局部 K16 对单局部：赢 / 输",f"{train['c3_k16_gains_over_c1']} / {train['c3_k16_losses_to_c1']}",f"{val['c3_k16_gains_over_c1']} / {val['c3_k16_losses_to_c1']}"],
             ["K16 配额丢失中，K32 能保留",train["k32_rescues_among_k16_losses"],val["k32_rescues_among_k16_losses"]],
             ["K16 未满题数",train["pool_underfull"],val["pool_underfull"]],
             ["空余槽位总数",train["unused_slots"],val["unused_slots"]],
             ["其中：截取后精确去重产生",train["duplicate_slots"],val["duplicate_slots"]],
             ["其中：某一侧原始列表不足配额",train["quota_input_shortfall"],val["quota_input_shortfall"]],
             ["只补空位、不换已有点的最佳可能新增可达题数",train["best_possible_refill_rescues"],val["best_possible_refill_rescues"]]]),
        "<p>补位上限允许用 GT 理想地选取任何剩余已解码点，实际无标签策略只能低于或等于该值。K32 只是已存预算的可达性诊断；没有运行新选择器，也不据此选择生产预算。</p>",
        table(["K16 丢失题的正确点来源", "训练", "验证"], [[label,train[key],val[key]] for label,key in
              [("仅全图","quota_loss_global_only"),("仅局部","quota_loss_local_only"),("两者都有","quota_loss_both")]]),
        table(["K16 配额丢失：最早正确点排名", "训练：题数 / 中位数 / 范围", "验证：题数 / 中位数 / 范围"],
              [[label]+[f"{s['quota_loss_first_valid_rank'][key]['n']} / {s['quota_loss_first_valid_rank'][key]['median']} / "
                f"{s['quota_loss_first_valid_rank'][key]['minimum']}–{s['quota_loss_first_valid_rank'][key]['maximum']}" for s in (train,val)]
               for label,key in [("全图列表","global_list"),("局部轮流合并列表","local_round_robin")]]),
        "<h2>局部名称与点解码：按视图计数</h2>",
        table(["视图指标", "训练", "验证"],[[label,train["crops"][key],val["crops"][key]] for label,key in
          [("全部裁剪","n"),("有正面积目标交集","overlap"),("完整包含有效目标集合","full"),("没有正面积目标交集","absent"),
           ("全部无名称","no_name"),("相交裁剪中无名称","overlap_no_name"),("完整包含目标仍无名称","full_no_name"),
           ("不相交裁剪中无名称","absent_no_name"),("相交且有名称","overlap_named"),("相交有名称且解码点命中","overlap_named_point_hit"),
           ("相交有名称，父格有正面积目标交集但所有点未命中","named_parent_support_but_point_miss"),
           ("无名称且正常 EOS","no_name_eos"),("无名称且达到 token 上限","no_name_token_cap")]]),
        "<p>名称出现不代表命名正确；父格相交不代表点一定存在，也不代表仅修改细偏移就能恢复。跨视图并非独立样本，同图多指令也不是独立截图。训练使用 OOF 提议器，验证使用完整训练提议器 D；两者差异包含数据与提议器协议差异。</p>"]
    for split in ("train", "val"):
        chunks.append(f"<h2>{split}：来源与目标大小</h2>")
        chunks.append(table(["来源", "题数", "全部点可达", "K16 可达", "配额丢失", "全部点无命中"],
            [[source,s["n"],s["decoded_union"],s["stages"]["reachable"],s["stages"]["quota_loss"],s["n"]-s["decoded_union"]]
             for source,s in summary[split]["by_source"].items()]))
        chunks.append(table(["全图输入目标短边估计 / px", "题数", "全部点可达", "K16 可达", "配额丢失"],
            [[name,s["n"],s["decoded_union"],s["stages"]["reachable"],s["stages"]["quota_loss"]] for name,s in summary[split]["by_target_size"].items()]))
    chunks.append("<p>目标短边使用最大面积连通目标分量的包络，按全图合并网格 ×32 估算输入尺度；固定分箱仅作描述，不是信息上限。"
        "例如验证中小于4像素仅2题，不能据此稳定估计小目标成功率。OS-Atlas 的验证 K16 可达率为20/37，"
        "RICO 为34/39；来源、任务与目标大小共同变化，不能将差异单独归因于某一种因素。</p>")
    chunks.append("<h2>固定规则抽样的验证案例</h2><p>每个失败分支按 sample_id 排序取首例；额外展示目标完整可见但无名称，以及有父格支持但点未命中的首例。无测试集样例。绿框：GT；彩框：裁剪；红叉：K16；绿圈：已解码但被丢失的正确点。</p>")
    examples = []
    vals = sorted([r for r in all_rows if r["split"] == "val"],key=lambda r:r["sample_id"])
    for stage in (*BRANCHES,"quota_loss"):
        candidate = next((r for r in vals if r["stage"] == stage), None)
        if candidate: examples.append((LABELS[stage],candidate))
    for label,predicate in [
        ("完整可见仍无名称",lambda r:any(c["target_fully_contained"] and c["no_name"] for c in r["crops"])),
        ("有父格支持但未解码命中",lambda r:any(c["target_area_fraction"]>0 and c["tokens"]>0 and c["parent_cells_intersecting_target"]>0 and c["valid_decoded_count"]==0 for c in r["crops"]))]:
        candidate=next((r for r in vals if predicate(r)),None)
        if candidate and not any(r['sample_id']==candidate['sample_id'] for _,r in examples):examples.append((label,candidate))
    for i,(label,row) in enumerate(examples):
        name=f"example_{i+1}.png"
        example_image(row,output/name)
        chunks.extend([f"<h3>{html.escape(label)}</h3><p><code>{html.escape(row['sample_id'])}</code> · {html.escape(row['instruction'])}</p>",
                       f'<img class="example" src="{name}" alt="Validation example">',
                       table(["视图","名称","Token 数","目标面积覆盖","正确点数","最早正确点排名"],
                          [["全图" if j==0 else f"局部{j}",v["name"],v["tokens"],f"{v['target_area_fraction']:.1%}",v["valid_decoded_count"],v["first_valid_rank"]] for j,v in enumerate([row["global_view"]]+row["crops"])]),
                       "<p>"+html.escape(CASE_NOTES.get(row['sample_id'],""))+"</p>",
                       "<p>"+html.escape(dumps(row["dropped_valid_points"][:6]))+"</p>" if row["stage"]=="quota_loss" else ""])
    chunks.append("<h2>复现与证据范围</h2><p>source.py 为本次独立入口；cases.jsonl 为逐题证据；crops.jsonl 为逐视图证据；summary.json 含来源、尺寸、区域序号统计。provenance.json 记录已完整重哈希的小文件与实际读取的 NPZ 成员/头字节哈希。NPZ 容器沿用封存预期 SHA，未完整重哈希大型缓存/状态，也未重做完整终审。完成记录的 13 项结果/文档哈希在分析前后核对一致。没有训练、改模型、读取测试预测或计算新最终点击分数。</p>")
    css="body{max-width:1160px;margin:40px auto;padding:0 24px;font:16px/1.65 'Segoe UI','Microsoft YaHei',sans-serif;color:#203044;background:#f8fafc}h1,h2,h3{line-height:1.3}h2{margin-top:40px}table{border-collapse:collapse;width:100%;margin:22px 0;background:white}th,td{padding:9px 13px;border-bottom:1px solid #dde3eb;text-align:left}th{background:#e9eff5}.chart,.example{max-width:100%;border:1px solid #dbe2e8;background:white}code{overflow-wrap:anywhere}p{overflow-wrap:anywhere}"
    (output/"report.html").write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>V13 训练验证候选丢失</title><style>'+css+'</style><body>'+"\n".join(chunks)+"</body></html>",encoding="utf-8")
    return [dict(label=label,sample_id=r["sample_id"],image=f"example_{i+1}.png") for i,(label,r) in enumerate(examples)]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    os.chdir(ROOT)
    run=ROOT/"runs/multiregion_v13"
    output=args.output.resolve()
    require(not output.exists(), "Use a fresh output directory")
    require(output != run and run not in output.parents, "Cannot write inside sealed V13")
    before=verify_completion(run)
    audit=read_json(run/"final_audit.json")
    require(audit["complete"] is True,"V13 final audit incomplete")
    reader=Reader(audit["input_sha256"])
    all_rows=[]
    for split,expected_n in (("train",873),("val",172)):
        data_path=run/f"data/{split}.jsonl"
        cases=[json.loads(l) for l in reader.full(data_path).decode("utf-8-sig").splitlines() if l.strip()]
        require(len(cases)==expected_n and len({c['sample_id'] for c in cases})==expected_n,"Split size/identity mismatch")
        index=reader.json(run/f"cache/{split}_c3_index.json")
        require(index["complete"] and index["labels_present"] is True,"Incomplete/labelless trainval cache")
        cache_map={e["sample_id"]:e for e in index["cases"]}
        require(set(cache_map)=={c['sample_id'] for c in cases},"Cache/case coverage mismatch")
        for i,case in enumerate(cases,1):
            require(case["split"]==split,"Unexpected split")
            path=run/f"pipeline/{split}/{case_key(case['sample_id'])}.json"
            pipe=reader.json(path)
            entry=cache_map[case["sample_id"]]
            require(entry["evidence"][str(path.relative_to(ROOT))] == reader.expected[canonical(path)],"Index/pipeline SHA mismatch")
            infos=[view_info(reader,case,pipe["global_result"],pipe["global_feature_path"],pipe["global_feature_sha256"])]
            infos += [view_info(reader,case,v["result"],v["feature_path"],v["feature_sha256"],v["transform"]) for v in pipe["local_views"]]
            row=analyze_case(case,pipe,infos)
            cache=entry["path"]
            require(reader.expected.get(canonical(cache))==entry["sha256"],"Cache SHA reference mismatch")
            require(str(reader.npy(cache,"sample_id"))==case["sample_id"],"Cached sample mismatch")
            require(str(reader.npy(cache,"pipeline_sha256"))==reader.expected[canonical(path)],"Cached pipeline mismatch")
            require(not bool(reader.npy(cache,"annotation_derived_coordinates")),"Annotation-derived point")
            points=reader.npy(cache,"points")
            labels=reader.npy(cache,"labels")
            require(points.tolist()==[p['point'] for p in pipe['pools']['c3']['32']],"Cache points mismatch")
            expected_labels=[int(hit(p,case)) for p in points.tolist()]
            require(labels.tolist()==expected_labels,"Independent labels differ from cached labels")
            for k in BUDGETS:
                ix=reader.npy(cache,f"pool_indices_{k}")
                require(points[ix].tolist()==[p['point'] for p in pipe['pools']['c3'][str(k)]],"Cached pool index mismatch")
                require(bool(labels[ix].any())==row['pools']['c3'][str(k)],"Cached reachability mismatch")
            all_rows.append(row)
            if i%100==0 or i==len(cases):print(dumps(dict(split=split,processed=i,n=len(cases))),flush=True)
    summary={}
    for split in ("train","val"):
        rows=[r for r in all_rows if r["split"]==split]
        summary[split]=counts(rows)
        summary[split]["by_source"]={source:counts([r for r in rows if r['source']==source]) for source in sorted({r['source'] for r in rows})}
        summary[split]["by_target_size"]={name:counts([r for r in rows if r['target_size_bin']==name]) for name in ("<4","4-8","8-16",">=16")}
        require(sum(summary[split]['stages'].values())==len(rows),"Nonexhaustive branches")
        require(summary[split]['duplicate_slots']+summary[split]['quota_input_shortfall']==summary[split]['unused_slots'],"Empty slot accounting failed")
    reader.unchanged()
    require(before==verify_completion(run),"Completion files changed during analysis")
    output.mkdir(parents=True,exist_ok=False)
    (output/"source.py").write_bytes(Path(__file__).read_bytes())
    (output/"summary.json").write_text(dumps(summary)+"\n",encoding="utf-8")
    (output/"cases.jsonl").write_text("".join(dumps(r)+"\n" for r in all_rows),encoding="utf-8")
    (output/"crops.jsonl").write_text("".join(dumps(dict(sample_id=r['sample_id'],split=r['split'],source=r['source'],region=i+1,**c))+"\n" for r in all_rows for i,c in enumerate(r['crops'])),encoding="utf-8")
    plot_summary(summary,output)
    examples=report(summary,all_rows,output)
    reader.unchanged()
    require(before==verify_completion(run),"Completion files changed during rendering")
    provenance=dict(complete=True,created_at_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=sha_bytes(Path(__file__).read_bytes()),
        scope="train/val only; no model instantiation/forward or test predictions; no tensor values loaded",
        full_npz_container_rehash=False, completion_13_before_after=before,
        verified_cache_cases=len(all_rows),verified_pools=len(all_rows)*8,
        files=list(reader.files.values()),members=list(reader.members.values()),examples=examples)
    (output/"provenance.json").write_text(dumps(provenance)+"\n",encoding="utf-8")
    (output/"output_sha256.json").write_text(dumps({p.name:sha_bytes(p.read_bytes()) for p in output.iterdir() if p.is_file()})+"\n",encoding="utf-8")
    print(dumps(dict(complete=True,output=str(output),summary={s:{k:v for k,v in summary[s].items() if k not in ('by_source','by_target_size')} for s in summary})),flush=True)


if __name__ == "__main__":
    main()
