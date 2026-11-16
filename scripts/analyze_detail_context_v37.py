"""Posthoc TRAIN mechanism diagnostics for the sealed V37 observations."""
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean, median

import run_detail_context_v37 as run


def main():
    assert run.read(run.OUT / "final_completion.json")["complete"]
    cases = run.read(run.OLD / "data/public.json")
    rows = run.read(run.OUT / "predictions.json")["rows"]
    assert len(cases) == len(rows) == 112
    measurements = []
    source = defaultdict(list)
    for case, row in zip(cases, rows):
        assert case["sample_id"] == row["sample_id"]
        values = {}
        for arm, record in row["arms"].items():
            side = arm.split("_")[0]
            x0, y0, x1, y1 = case["transforms"][side]["box"]
            original_area = (x1 - x0) * (y1 - y0)
            values[arm] = (record["processed_pixels"] / original_area) ** 0.5
        out = {"sample_id": row["sample_id"], "source": row["source"],
               "high1280_over_low640_sampling_proxy": values["1280_hi"] / values["640_low"],
               "high640_over_low640_sampling_proxy": values["640_hi"] / values["640_low"],
               "high1280_over_high640_sampling_proxy": values["1280_hi"] / values["640_hi"],
               "low1280_over_low640_sampling_proxy": values["1280_low"] / values["640_low"],
               "old640_crop_miss": row["arms"]["640_low"]["GT_fraction"] == 0,
               "old640_full": row["arms"]["640_low"]["GT_fraction"] >= .99,
               "new1280_full": row["arms"]["1280_hi"]["GT_fraction"] >= .99,
               "hits": {arm: record["hit"] for arm, record in row["arms"].items()}}
        measurements.append(out)
        source[row["source"]].append(out)

    ratio_names = [name for name in measurements[0] if name.endswith("sampling_proxy")]

    def aggregate(group):
        pairs = (("640_low", "1280_hi"), ("640_low", "640_hi"),
                 ("1280_low", "1280_hi"), ("640_hi", "1280_hi"))
        return {"n": len(group), "hits": {arm: sum(r["hits"][arm] for r in group) for arm in group[0]["hits"]},
                "paired_changes": {a + "_to_" + b: {
                    "repairs": sum(not r["hits"][a] and r["hits"][b] for r in group),
                    "harms": sum(r["hits"][a] and not r["hits"][b] for r in group)} for a, b in pairs},
                "median_sampling_ratios": {name: median(r[name] for r in group) for name in ratio_names},
                "mean_sampling_ratios": {name: mean(r[name] for r in group) for name in ratio_names}}

    result = {"complete": True, "n": len(measurements), "sampling_proxy_definition":
              "sqrt(actual processor pixels / original ROI area). Ratios approximate linear input sampling; they are not exact target recognizability, and Qwen patches are discrete.",
              "overall": aggregate(measurements),
              "old640_crop_miss": aggregate([r for r in measurements if r["old640_crop_miss"]]),
              "old640_full": aggregate([r for r in measurements if r["old640_full"]]),
              "new1280_full": aggregate([r for r in measurements if r["new1280_full"]]),
              "source": {name: aggregate(group) for name, group in source.items()},
              "individual": measurements,
              "training_only": True}
    run.write(run.OUT / "mechanism.json", result)
    lines = ["V37训练集视野和采样密度补充诊断", "像素采样代理：sqrt(实际处理像素/原图裁剪面积)，它不是目标可辨识度或模型注意力的测量。"]
    for name in ("overall", "old640_crop_miss", "old640_full", "new1280_full"):
        section = result[name]
        lines.append(f"{name}: n={section['n']}，四臂命中={section['hits']}，配对修复/破坏={section['paired_changes']}，采样比例中位数={section['median_sampling_ratios']}")
    lines.append("各来源：")
    for name, section in sorted(result["source"].items()):
        lines.append(f"{name}: n={section['n']}，四臂命中={section['hits']}")
    (run.OUT / "mechanism.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
