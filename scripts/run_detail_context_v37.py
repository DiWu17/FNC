"""TRAIN-only 2x2 local observation study: crop extent x processed pixel budget."""
import os

for _name, _value in {
    "PYTHONDONTWRITEBYTECODE": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
    "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
}.items():
    os.environ.setdefault(_name, _value)

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import run_paired_scale_v35 as prior
from clickagent.grounded_composition_v28 import GroundedRuntime, LOCAL_PROMPT, pixel
from clickagent.action_model import ActionQwenRuntime
from clickagent.public_evaluation import point_in_geometry

OUT = ROOT / "runs/detail_context_v37"
OLD = prior.OUT
HIGH_PIXELS = 1024 * 1024
ARMS = ("640_hi", "1280_hi")


class ExperimentalHighPixelRuntime(GroundedRuntime):
    """Keep the public image preparation and V28 readout; lift only its 262k guard.

    PublicQwenRuntime.__init__ fixes a public transfer protocol to 262144 pixels.
    This TRAIN-only mechanism experiment varies that budget by invoking its
    upstream frozen-model loader directly. No public benchmark policy is changed.
    """

    def __init__(self, config):
        ActionQwenRuntime.__init__(self, config)


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    h = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    text = json.dumps(value, ensure_ascii=False, indent=2, default=default, allow_nan=False) + "\n"
    for attempt in range(10):
        try:
            temporary.write_text(text, encoding="utf-8")
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(min(0.05 * 2 ** attempt, 1))


def status(phase, **fields):
    event = {"phase": phase, "at_utc": now(), "pid": os.getpid(), **fields}
    write(OUT / "status.json", event)
    print(json.dumps(event, ensure_ascii=False), flush=True)


def key(sid):
    return sha256(sid.encode()).hexdigest()[:24]


def archive(path, arrays, metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        np.savez_compressed(stream, metadata=np.array(json.dumps(metadata, ensure_ascii=False, default=default)), **arrays)


def metadata(path):
    with np.load(path, allow_pickle=False) as item:
        return json.loads(str(item["metadata"]))


def prepare():
    assert not OUT.exists(), "A sealed run is never overwritten."
    assert read(OLD / "final_completion.json")["complete"]
    prior.verify()
    old_completion = read(OLD / "final_completion.json")
    assert sha(OLD / "final_readout.txt") == old_completion["final_readout_sha256"]
    cases = read(OLD / "data/public.json")
    assert len(cases) == 112 and all(case["cohort"] == "train" for case in cases)
    labels = read(OLD / "data/train_labels.json")
    assert len(labels) == 112
    adopt = read(ROOT / "CURRENT_RESEARCH.json")
    assert adopt["run_id"] == "quality_readout_v15_3k"
    for model in adopt["models"].values():
        assert sha(model["path"]) == model["sha256"]
    # Order is fixed before looking at any new outcomes; old 640/1280 remain sealed references.
    orders = {case["sample_id"]: list(ARMS if index % 2 == 0 else ARMS[::-1])
              for index, case in enumerate(cases)}
    inputs = [Path(__file__), OLD / "protocol.json", OLD / "final_completion.json",
              OLD / "data/public.json", OLD / "data/train_labels.json",
              OLD / "predictions.json", OLD / "results.json",
              ROOT / "runs/crossfit_candidates_v12/runtime_identity.json",
              ROOT / "src/clickagent/grounded_composition_v28.py",
              ROOT / "src/clickagent/model_features.py",
              ROOT / "src/clickagent/conditioned_features.py"]
    spec = {
        "registered_at_utc": now(),
        "scope": "112 previously selected TRAIN screenshots only; two fresh high-pixel arms, 224 real local calls. Historical low-pixel arms are immutable V35 archives. No DEV labels, no 58 reserved images, no scorer fitting, no mainline adoption. V36 preflight failed before model loading because the public adapter fixes 262144; its failure is preserved.",
        "factorial": {"crop_sides_original_pixels": [640, 1280], "processor_max_pixels": [262144, HIGH_PIXELS], "fresh_arms": list(ARMS)},
        "intervention": "Use identical saved V35 crop boxes, 2x Lanczos RGB, original instruction, LOCAL_PROMPT and 56-token cap. Frozen Qwen3-VL-8B. Only processor max_pixels changes from V35 for each matching ROI; larger ROI at high budget tests added context with less resolution loss, but exact sampling density is measured per case.",
        "primary": "Across all112TRAIN, compare actual native point 1280_hi vs historical 640_low; report repairs, harms, source-stratified paired 95% bootstrap CI (10000 draws, seed20261136). This is a training mechanism comparison, not generalization or a replacement policy.",
        "secondary": "Compare 640_hi vs 640_low and 1280_hi vs 1280_low for detail effect; 1280_hi vs 640_hi for context-at-higher-budget. These are exploratory and unadjusted. Report actual processed pixels, grid sizes, GT crop coverage, abstentions, and discordances. Two-view GT unions are oracle diagnostics only.",
        "order": "Existing V35 case order; new 640_hi/1280_hi alternating by case index, 56 pairs per order. All112 cases remain denominators, including OOM/invalid if recoverable. No outcome-dependent selection.",
        "resource_rule": "Two synthetic warmups test 1024^2 pixel operation. If it fails, stop this run and report the failure; no silent budget change. A lower budget needs a separately registered experiment.",
        "experimental_loader": "ExperimentalHighPixelRuntime invokes ActionQwenRuntime.__init__ directly, bypassing only PublicQwenRuntime.__init__'s public-transfer 262144 budget guard. Inherited PublicQwenRuntime._prepare_image and GroundedRuntime.infer are unchanged. This is not the published public transfer protocol.",
        "limitations": "Processor pixel budgets are caps, so actual pixels may differ. Changing budget also changes image tokens and runtime. Four arms are on TRAIN and cannot establish independent accuracy; primary comparison changes both context and budget.",
        "adopted_models": adopt["models"],
        "qwen_file_inventory": read(OLD / "protocol.json")["qwen_file_inventory"],
        "input_sha256": {str(path): sha(path) for path in inputs},
        "case_order": orders,
    }
    write(OUT / "protocol.json", spec)
    status("prepared", completed=0, total=112)


def verify():
    spec = read(OUT / "protocol.json")
    for path, digest in spec["input_sha256"].items():
        assert sha(path) == digest, path
    for entry in spec["qwen_file_inventory"]:
        stat = Path(entry["path"]).stat()
        assert stat.st_size == entry["bytes"] and stat.st_mtime_ns == entry["mtime_ns"]
    for model in spec["adopted_models"].values():
        assert sha(model["path"]) == model["sha256"]
    return spec


def collect():
    spec = verify()
    from clickagent.advantage_runtimes import configure_paths, RuntimeCounter
    from clickagent.model_features import ExtractionConfig
    configure_paths("coordinate")
    import torch
    torch.set_num_threads(4)
    contract = dict(read(ROOT / "runs/crossfit_candidates_v12/runtime_identity.json")["runtime_contract"])
    contract["layers"] = tuple(contract["layers"])
    contract["conditions"] = tuple(contract["conditions"])
    contract["max_memory"] = {int(k) if str(k).isdigit() else k: v for k, v in contract["max_memory"].items()}
    contract["max_pixels"] = HIGH_PIXELS
    contract["offload_folder"] = str(OUT / "runtime/offload")
    config = ExtractionConfig(**contract)
    status("loading_model", completed=0, total=112)
    runtime = ExperimentalHighPixelRuntime(config)
    counter = RuntimeCounter()
    counter.attach(runtime.model, 16)
    write(OUT / "runtime.json", {"metadata": runtime.runtime_metadata, "torch": torch.__version__, "config": contract, "no_scorer_or_proposer": True})

    def generate(image, instruction):
        before = counter.snapshot()
        arrays, meta = runtime.infer(image, instruction, LOCAL_PROMPT, 56)
        after = counter.snapshot()
        meta["counter"] = {name: after[name] - before[name] for name in before}
        assert meta["ground_truth_in_model_input"] is False and meta["all_backbone_parameters_frozen"] and meta["own_token_forward"]
        assert meta["counter"]["processed_pixels"] <= HIGH_PIXELS
        return arrays, meta

    with Image.new("RGB", (1800, 1800), "white") as synthetic:
        ImageDraw.Draw(synthetic).rectangle([800, 800, 1000, 1000], fill="blue")
        for arm in ARMS:
            path = OUT / "runtime/warmup" / (arm + ".npz")
            if path.exists():
                continue
            side = int(arm.split("_")[0])
            left = (1800 - side) // 2
            status("synthetic_warmup", arm=arm, completed=0, total=112)
            with synthetic.crop([left, left, left + side, left + side]).resize((2 * side, 2 * side), Image.Resampling.LANCZOS) as view:
                arrays, meta = generate(view, "Click the blue rectangle.")
                archive(path, {}, meta)
                del arrays

    cases = read(OLD / "data/public.json")
    for index, case in enumerate(cases, 1):
        assert sha(case["image"]) == case["image_sha256"]
        with Image.open(case["image"]) as opened:
            image = opened.convert("RGB")
        try:
            for arm in spec["case_order"][case["sample_id"]]:
                path = OUT / "raw" / arm / (key(case["sample_id"]) + ".npz")
                if path.exists():
                    continue
                source = arm.split("_")[0]
                transform = case["transforms"][source]
                status("train_observations", completed=index - 1, total=112, current_case=index, arm=arm)
                started = time.perf_counter()
                with image.crop(transform["box"]).resize(tuple(transform["crop_size"]), Image.Resampling.LANCZOS) as view:
                    rgb_digest = sha256(view.tobytes()).hexdigest()
                    arrays, meta = generate(view, case["instruction"])
                meta.update(sample_id=case["sample_id"], arm=arm, transform=transform,
                            image_sha256=case["image_sha256"], local_RGB_sha256=rgb_digest,
                            instruction=case["instruction"], world_point=pixel(meta["point"], meta["image_size"], transform),
                            crop_and_generation_seconds=time.perf_counter() - started)
                archive(path, arrays, meta)
                del arrays
            status("train_observations", completed=index, total=112, arm="case_complete")
        finally:
            image.close()
    status("observations_complete", completed=112, total=112)


def analyze():
    spec = verify()
    cases = read(OLD / "data/public.json")
    labels = read(OLD / "data/train_labels.json")
    old_rows = {r["sample_id"]: r for r in read(OLD / "predictions.json")["rows"]}
    all_arms = ("640_low", "1280_low", *ARMS)
    rows = []
    for case in cases:
        sid = case["sample_id"]
        arm_rows = {}
        for arm in all_arms:
            side = arm.split("_")[0]
            if arm.endswith("low"):
                old = old_rows[sid]["arms"][side]
                archive_path = OLD / "raw" / side / (key(sid) + ".npz")
            else:
                archive_path = OUT / "raw" / arm / (key(sid) + ".npz")
                assert archive_path.is_file()
                old = None
            meta = metadata(archive_path)
            transform = case["transforms"][side]
            point = pixel(meta["point"], meta["image_size"], transform)
            assert meta["transform"] == transform and meta["instruction"] == case["instruction"]
            assert meta["system_prompt"] == LOCAL_PROMPT and meta["max_new_tokens"] == 56
            assert meta["world_point"] == point and meta["ground_truth_in_model_input"] is False
            label = labels[sid]
            hit = point is not None and point_in_geometry([point[0] / case["image_size"][0], point[1] / case["image_size"][1]], label)
            if old is not None:
                assert old["hit"] == hit and old["point"] == point
            arm_rows[arm] = {"point": point, "hit": bool(hit), "valid": point is not None,
                             "processed_pixels": meta["counter"]["processed_pixels"], "seconds": meta["elapsed_seconds"],
                             "grid_shape": meta["grid_shape"], "tokens": len(meta["generated_token_ids"]),
                             "GT_fraction": old_rows[sid]["arms"][side]["GT_fraction"]}
        rows.append({"sample_id": sid, "source": case["source"], "arms": arm_rows})
    assert len(rows) == len({row["sample_id"] for row in rows}) == 112
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row["source"]].append(i)
    comparisons = [
        ("primary_context_with_detail_restored", "640_low", "1280_hi"),
        ("detail_at_640", "640_low", "640_hi"),
        ("detail_at_1280", "1280_low", "1280_hi"),
        ("context_at_high_budget", "640_hi", "1280_hi"),
        ("historical_scale_at_low_budget", "640_low", "1280_low"),
    ]
    result_comparisons = {}
    for ci, (name, a_name, b_name) in enumerate(comparisons):
        a = np.array([r["arms"][a_name]["hit"] for r in rows], bool)
        b = np.array([r["arms"][b_name]["hit"] for r in rows], bool)
        diff = b.astype(int) - a.astype(int)
        rng = np.random.default_rng(20261136 + ci)
        draws = np.array([100 * diff[np.concatenate([rng.choice(indices, len(indices), replace=True) for indices in groups.values()])].mean() for _ in range(10000)])
        result_comparisons[name] = {"a": a_name, "b": b_name, "a_hits": int(a.sum()), "b_hits": int(b.sum()),
                                    "repairs": int((~a & b).sum()), "harms": int((a & ~b).sum()),
                                    "delta_pp": float(100 * diff.mean()), "paired_CI95_pp": np.quantile(draws, [.025, .975]).tolist(),
                                    "seed": 20261136 + ci}
    arm_stats = {}
    for arm in all_arms:
        entries = [r["arms"][arm] for r in rows]
        arm_stats[arm] = {"hits": sum(e["hit"] for e in entries), "valid": sum(e["valid"] for e in entries),
                          "full_coverage": sum(e["GT_fraction"] >= .99 for e in entries),
                          "mean_pixels": float(np.mean([e["processed_pixels"] for e in entries])),
                          "median_pixels": float(np.median([e["processed_pixels"] for e in entries])),
                          "mean_seconds": float(np.mean([e["seconds"] for e in entries])),
                          "mean_image_tokens": float(np.mean([int(np.prod(e["grid_shape"])) for e in entries]))}
    old_miss = [r for r in rows if r["arms"]["640_low"]["GT_fraction"] == 0]
    full_new = [r for r in rows if r["arms"]["1280_hi"]["GT_fraction"] >= .99]
    result = {"complete": True, "at_utc": now(), "n": len(rows), "fresh_calls": 224,
              "comparisons": result_comparisons, "arms": arm_stats,
              "old640_crop_miss_n": len(old_miss),
              "old640_miss_1280_covered": sum(r["arms"]["1280_hi"]["GT_fraction"] > 0 for r in old_miss),
              "old640_miss_1280_hi_hits": sum(r["arms"]["1280_hi"]["hit"] for r in old_miss),
              "1280_full_n": len(full_new), "1280_full_hi_failures": sum(not r["arms"]["1280_hi"]["hit"] for r in full_new),
              "1280_full_hi_abstentions": sum(not r["arms"]["1280_hi"]["valid"] for r in full_new),
              "oracle_four_arm_union": sum(any(r["arms"][arm]["hit"] for arm in all_arms) for r in rows),
              "training_only": True, "no_model_fitting": True, "no_reserve_forward": True, "no_automatic_adoption": True}
    write(OUT / "predictions.json", {"rows": rows})
    write(OUT / "results.json", result)
    lines = ["V36 TRAIN-only 2×2局部观察：窗口640/1280 × 像素上限262144/1048576。", "112张旧TRAIN；V35低像素两臂原封复用，新增高像素两臂共224次真实局部前向。", "实际点命中/112、修复、破坏及来源分层配对95%区间："]
    for name, comparison in result_comparisons.items():
        lines.append(f"{name}: {comparison['a']} {comparison['a_hits']} → {comparison['b']} {comparison['b_hits']}; 修复{comparison['repairs']}，破坏{comparison['harms']}，差{comparison['delta_pp']:.2f}pp，CI {comparison['paired_CI95_pp']}")
    lines.append("各臂实际处理像素、图像token及局部耗时：" + str(arm_stats))
    lines.append(f"原640裁剪漏目标{len(old_miss)}，1280框几何补入{result['old640_miss_1280_covered']}，1280高预算真实救回{result['old640_miss_1280_hi_hits']}。1280完整覆盖{len(full_new)}中高预算仍失败{result['1280_full_hi_failures']}、其中无效点{result['1280_full_hi_abstentions']}。")
    lines.append(f"四臂GT辅助并集{result['oracle_four_arm_union']}/112，只是可达上限，不是已实现点击准确率。TRAIN机制试验、次比较未校正，不作独立泛化或MVP比较；主线及58保留图不变。")
    (OUT / "readout.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for path, digest in spec["input_sha256"].items():
        assert sha(path) == digest
    write(OUT / "completion.json", {"complete": True, "at_utc": now(), "artifact_sha256": {name: sha(OUT / name) for name in ["protocol.json", "predictions.json", "results.json", "readout.txt"]}})
    status("complete", completed=112, total=112, results_path=str(OUT / "readout.txt"))
    print("\n".join(lines), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "collect", "analyze", "all"])
    stage = parser.parse_args().stage
    if stage == "prepare":
        prepare()
        return
    verify()
    if stage in ("collect", "all"):
        collect()
    if stage in ("analyze", "all"):
        analyze()


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        status("failed", error=repr(error), traceback=traceback.format_exc())
        raise
