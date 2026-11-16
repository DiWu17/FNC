"""TRAIN-only fixed-center crop-scale audit. No model loading or inference."""
from collections import defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[1]
PREVIOUS = ROOT / "runs/observation_regions_v33"
OUT = ROOT / "runs/train_scale_diagnostic_v34"
SIDES = (320, 640, 960, 1280, 1600)
ARMS = ("primary", "nearby", "name_fullpool", "coordinate_mass", "random_matched")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def now():
    return datetime.now(timezone.utc).isoformat()


def resize_region(old_box, size, side):
    """Change size around the saved crop center; labels are not an input."""
    width, height = size
    cx = (old_box[0] + old_box[2]) / 2
    cy = (old_box[1] + old_box[3]) / 2
    cw, ch = min(width, side), min(height, side)
    left = max(0, min(round(cx - cw / 2), width - cw))
    top = max(0, min(round(cy - ch / 2), height - ch))
    return [left, top, left + cw, top + ch]


def fraction(region, size, label):
    values = []
    for x0, y0, x1, y1 in label["boxes"]:
        left, top, right, bottom = x0 * size[0], y0 * size[1], x1 * size[0], y1 * size[1]
        overlap = max(0, min(region[2], right) - max(region[0], left)) * max(0, min(region[3], bottom) - max(region[1], top))
        values.append(overlap / ((right - left) * (bottom - top)))
    return max(values, default=0)


def replay_fraction(region, size, label):
    """Independent corner-clipping expression for the geometry audit."""
    result = 0
    for bbox in label["boxes"]:
        corners = [bbox[i] * size[i % 2] for i in range(4)]
        spans = [min(region[axis + 2], corners[axis + 2]) - max(region[axis], corners[axis]) for axis in (0, 1)]
        overlap = spans[0] * spans[1] if min(spans) > 0 else 0
        result = max(result, overlap / ((corners[2] - corners[0]) * (corners[3] - corners[1])))
    return result


def main():
    assert read(PREVIOUS / "final_completion.json")["complete"]
    assert not OUT.exists(), "Use a new run directory; never overwrite sealed results."
    cases = [c for c in read(PREVIOUS / "data/public.json") if c["cohort"] == "train"]
    baseline = read(PREVIOUS / "data/baseline.json")
    assert len(cases) == len({c["image_sha256"] for c in cases}) == 112
    paths = [Path(__file__), PREVIOUS / "data/public.json", PREVIOUS / "data/baseline.json", PREVIOUS / "data/train_labels.json", PREVIOUS / "training_geometry.json", PREVIOUS / "final_completion.json"]
    prior_plans = {}
    for c in cases:
        path = PREVIOUS / "plans" / (sha256(c["sample_id"].encode()).hexdigest()[:24] + ".json")
        prior_plans[c["sample_id"]] = read(path)
        paths.append(path)
    protocol = {
        "registered_at_utc": now(), "authorization": "摘要先提交，继续刚才的实验；先做训练缓存尺度与覆盖诊断",
        "scope": "112 existing TRAIN RGB only. No DEV labels, reserve inference, model loading, training or new predictions.",
        "sides_original_pixels": list(SIDES), "arms": list(ARMS),
        "intervention": "Keep each saved 640px crop center, resize with image-edge clipping. At 640 every old box must replay exactly. Do not replan centers or use GT in plans.",
        "outcomes": "GT intersection and >=99% coverage, old main-crop miss recovery, source strata, area and linear sampling proxies. Geometry diagnostic only, not click accuracy or proof that larger crops improve inference.",
        "limits": "Larger nested crops mechanically improve geometric coverage; under a fixed processed-pixel budget target detail can decrease. Multiple arm unions use GT and are diagnostic. No best-scale promotion from these counts.",
        "input_sha256": {str(path): digest(path) for path in paths},
    }
    save(OUT / "protocol.json", protocol)
    save(OUT / "status.json", {"phase": "planning_without_labels", "at_utc": now()})
    plans = []
    for c in cases:
        sid = c["sample_id"]
        old = {"primary": baseline[sid]["first_crop"]["box"]}
        old.update({arm: prior_plans[sid]["plans"][arm]["transform"]["box"] for arm in ARMS[1:]})
        regions = {arm: {str(side): resize_region(box, c["image_size"], side) for side in SIDES} for arm, box in old.items()}
        for arm in ARMS:
            assert regions[arm]["640"] == old[arm]
            for smaller, larger in zip(SIDES, SIDES[1:]):
                a, b = regions[arm][str(smaller)], regions[arm][str(larger)]
                assert b[0] <= a[0] and b[1] <= a[1] and b[2] >= a[2] and b[3] >= a[3]
        plans.append({"sample_id": sid, "source": c["source"], "image_size": c["image_size"], "regions": regions, "ground_truth_used": False})
    save(OUT / "plans.json", plans)
    # Labels are first decoded only after every label-free region has been saved.
    labels = read(PREVIOUS / "data/train_labels.json")
    rows = []
    by_group = defaultdict(list)
    replays = 0
    for row in plans:
        label = labels[row["sample_id"]]
        assert label["type"] == "bbox"
        scores = {}
        for arm, scales in row["regions"].items():
            scores[arm] = {}
            reference = scales["640"]
            reference_area = (reference[2] - reference[0]) * (reference[3] - reference[1])
            for side, box in scales.items():
                value = fraction(box, row["image_size"], label)
                assert abs(value - replay_fraction(box, row["image_size"], label)) < 1e-12
                area = (box[2] - box[0]) * (box[3] - box[1])
                scores[arm][side] = {"GT_fraction": value, "area_ratio_vs640": area / reference_area, "fixed_pixel_linear_sampling_proxy_vs640": (reference_area / area) ** .5}
                replays += 1
        value = {"sample_id": row["sample_id"], "source": row["source"], "primary640_miss": scores["primary"]["640"]["GT_fraction"] == 0, "arms": scores}
        rows.append(value)
        by_group[row["source"]].append(value)
        assert fraction([0, 0, *row["image_size"]], row["image_size"], label) >= 1 - 1e-12

    def summarize(group):
        output = {}
        for arm in ARMS:
            output[arm] = {}
            for side in map(str, SIDES):
                values = [row["arms"][arm][side] for row in group]
                output[arm][side] = {
                    "n": len(group), "intersects": sum(v["GT_fraction"] > 0 for v in values),
                    "full_ge99pct": sum(v["GT_fraction"] >= .99 for v in values),
                    "new_intersections_vs_same_arm640": sum(row["arms"][arm]["640"]["GT_fraction"] == 0 and row["arms"][arm][side]["GT_fraction"] > 0 for row in group),
                    "primary640_misses_covered": sum(row["primary640_miss"] and row["arms"][arm][side]["GT_fraction"] > 0 for row in group),
                    "mean_area_ratio_vs640": mean(v["area_ratio_vs640"] for v in values),
                    "median_linear_sampling_proxy_vs640": median(v["fixed_pixel_linear_sampling_proxy_vs640"] for v in values),
                }
        return output

    summary = summarize(rows)
    old_geometry = read(PREVIOUS / "training_geometry.json")
    for arm in ARMS[1:]:
        assert summary[arm]["640"]["intersects"] == old_geometry["arms"][arm]["intersects"]
        assert summary[arm]["640"]["full_ge99pct"] == old_geometry["arms"][arm]["full_ge99pct"]
    result = {"complete": True, "at_utc": now(), "n": 112, "primary640_misses": sum(row["primary640_miss"] for row in rows), "arms": summary, "source_strata": {s: summarize(group) for s, group in by_group.items()}, "new_inference_calls": 0, "model_training": False, "no_DEV_labels": True, "reserve_no_forward": True}
    save(OUT / "geometry_details.json", rows)
    save(OUT / "results.json", result)
    lines = ["V34训练缓存尺度诊断：固定中心，只改变窗口大小", "112张既有训练截图；没有模型加载、训练或新推理，不读取开发标签，58张保留图未前向。", f"原主640裁剪完全漏目标：{result['primary640_misses']}/112。", "区域 | 边长上限 | 相交 | 覆盖>=99% | 原主裁剪漏例中覆盖 | 平均面积倍数", "所有数字都是GT几何诊断，不是定位准确率。"]
    for arm in ARMS:
        for side in map(str, SIDES):
            a = summary[arm][side]
            lines.append(f"{arm} | {side} | {a['intersects']} | {a['full_ge99pct']} | {a['primary640_misses_covered']} | {a['mean_area_ratio_vs640']:.2f}")
    lines += ["所有区域先在不输入GT的函数中生成并保存，随后才读取训练标签。640像素框逐项重放V33；各尺度框嵌套，几何覆盖随面积增加是预期性质，不代表模型精度提升。", "固定处理像素预算下，更大原图区域会降低目标采样密度；所存线性采样比例只是面积代理，非实际模型预处理尺寸或耗时。", "训练集含历史随机及失败定向样本，不能将比例推广为独立测试集表现。下一步真实推理需固定中心/提示/模型/像素预算，比较640与扩大窗口的修复和破坏，再决定是否研究选择器。"]
    (OUT / "readout.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for path, value in protocol["input_sha256"].items():
        assert digest(path) == value
    audit = {"complete": True, "at_utc": now(), "geometry_formula_replays": replays, "640_box_identity_checks": 112 * len(ARMS), "nesting_checks": 112 * len(ARMS) * (len(SIDES) - 1), "V33_geometry_count_checks": 8, "input_hashes_verified": len(paths), "no_model_or_inference": True}
    save(OUT / "audit.json", audit)
    save(OUT / "completion.json", {"complete": True, "at_utc": now(), "artifact_sha256": {name: digest(OUT / name) for name in ["protocol.json", "plans.json", "geometry_details.json", "results.json", "readout.txt", "audit.json"]}})
    save(OUT / "status.json", {"phase": "complete", "at_utc": now(), "complete": True})
    print("\n".join(lines))


if __name__ == "__main__":
    main()
