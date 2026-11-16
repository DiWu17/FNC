"""TRAIN-only fresh 640/1280 local observation experiment. No head fitting."""
import os
for _name, _value in {"PYTHONDONTWRITEBYTECODE": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"}.items():
    os.environ.setdefault(_name, _value)
import argparse
from collections import defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sys
import time
import traceback
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import run_train_scale_diagnostic_v34 as geometry
from clickagent.grounded_composition_v28 import GroundedRuntime, LOCAL_PROMPT, pixel
from clickagent.public_evaluation import point_in_geometry

OUT = ROOT / "runs/paired_scale_v35"
V33 = ROOT / "runs/observation_regions_v33"
V34 = geometry.OUT
ARMS = ("640", "1280")


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    h = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
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
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=default, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def key(sid):
    return sha256(sid.encode()).hexdigest()[:24]


def load_metadata(path):
    with np.load(path, allow_pickle=False) as archive:
        return json.loads(str(archive["metadata"]))


def archive(path, arrays, metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        np.savez_compressed(stream, metadata=np.array(json.dumps(metadata, ensure_ascii=False, default=default)), **arrays)


def status(phase, **fields):
    value = {"phase": phase, "at_utc": now(), "pid": os.getpid(), "complete": phase == "complete", **fields}
    write(OUT / "status.json", value)
    print(json.dumps(value, ensure_ascii=False, default=default), flush=True)


def verify():
    spec = read(OUT / "protocol.json")
    for path, expected in spec["hashes"].items():
        assert sha(path) == expected, path
    for entry in spec["qwen_file_inventory"]:
        stat = Path(entry["path"]).stat()
        assert stat.st_size == entry["bytes"] and stat.st_mtime_ns == entry["mtime_ns"], entry["path"]
    return spec


def prepare():
    if (OUT / "protocol.json").exists():
        return verify()
    assert not OUT.exists()
    completion = read(V34 / "completion.json")
    for name, expected in completion["artifact_sha256"].items():
        assert sha(V34 / name) == expected
    cases = [c for c in read(V33 / "data/public.json") if c["cohort"] == "train"]
    cases.sort(key=lambda c: key("v35_order|" + c["sample_id"]))
    plans = {row["sample_id"]: row for row in read(V34 / "plans.json")}
    assert len(cases) == len({c["image_sha256"] for c in cases}) == 112
    refs = {}
    inputs = [Path(__file__), Path(geometry.__file__), V34 / "completion.json", V34 / "plans.json", V33 / "data/public.json", V33 / "data/train_labels.json", ROOT / "runs/crossfit_candidates_v12/runtime_identity.json"]
    for index, case in enumerate(cases):
        sid = case["sample_id"]
        case["arm_order"] = list(ARMS if index % 2 == 0 else ARMS[::-1])
        case["transforms"] = {}
        for arm in ARMS:
            box = plans[sid]["regions"]["primary"][arm]
            case["transforms"][arm] = {"box": box, "original_size": case["image_size"], "scale": 2, "crop_size": [2 * (box[2] - box[0]), 2 * (box[3] - box[1])], "ground_truth_used": False}
        old = ROOT / "runs/grounded_composition_v28/raw/free" / (key(sid) + ".npz")
        meta = load_metadata(old)
        assert meta["transform"] == case["transforms"]["640"]
        assert meta["system_prompt"] == LOCAL_PROMPT and meta["max_new_tokens"] == 56
        assert meta["ground_truth_in_model_input"] is False
        refs[sid] = {"generated_text": meta["generated_text"], "target": meta["target"], "point": meta["point"], "world_point": pixel(meta["point"], meta["image_size"], meta["transform"]), "archive": str(old)}
        inputs.extend([old, Path(case["image"])])
    write(OUT / "data/public.json", cases)
    write(OUT / "data/reference640.json", refs)
    shutil.copyfile(V33 / "data/train_labels.json", OUT / "data/train_labels.json")
    inputs.extend([OUT / "data/public.json", OUT / "data/reference640.json", OUT / "data/train_labels.json"])
    for name in ["grounded_composition_v28", "conditioned_features", "model_features", "action_model", "target_token", "advantage_runtimes", "public_evaluation"]:
        inputs.append(ROOT / "src/clickagent" / (name + ".py"))
    previous = read(V33 / "protocol.json")
    inventory = [e for e in previous["model_file_inventory"] if "Qwen3-VL-8B-Instruct" in e["path"]]
    assert inventory
    reg = read(ROOT / "CURRENT_RESEARCH.json")
    assert reg["run_id"] == "quality_readout_v15_3k"
    for item in reg["models"].values():
        assert sha(item["path"]) == item["sha256"]
    spec = {
        "registered_at_utc": now(), "authorization": "接着做训练集640/1280真实局部定位对照；继续正常Python执行与轮询直到分析完成",
        "data": "112 existing TRAIN RGB, historical random and failure-targeted mixture. No DEV or reserve inference. No fitted model or threshold. TRAIN outcomes are not independent generalization results.",
        "intervention": "Same saved main crop center, 640 vs 1280 maximum original-pixel side, image-edge clipping, 2x Lanczos resize, unchanged LOCAL_PROMPT and original instruction,56token cap. Same frozen GroundedRuntime and inherited extraction contract. No new global calls or scorer invocation.",
        "order": "Cases sorted by fixed v35 ID hash. Alternate arm order by case;56 pairs each order. Both arms always execute, even if transforms duplicate.",
        "generation_calls": {"scored": 224, "synthetic_warmups": 2},
        "primary": "1280-minus640 actual native point hit across all112TRAIN. One paired comparison, source-stratified10000draw bootstrap95%CI,seed20261135. Report repairs,harms,all abstentions retained.",
        "secondary": "Historical640text/target/normalized/worldpoint parity;old640crop-miss andoldfullcoverage strata;new1280fullcoverage failures;source strata;per-view generation seconds,tokens,actualpixels,forwardcalls. No whole-pipeline orMVPspeed claim. Two synthetic warmups excluded.",
        "limits": "Larger crop changes field of view and relative target scale together, not a pure context-only intervention. Same max262144pixel cap,actualpixel counts may differ with aspect ratio. Geometry gains do not guarantee prediction gains. No dev-selected arm or automatic adoption.",
        "adopted_models": reg["models"], "qwen_file_inventory": inventory,
        "hashes": {str(path): sha(path) for path in inputs},
    }
    write(OUT / "protocol.json", spec)
    verify()
    status("prepared", completed=0, total=112)
    return spec


def collect():
    verify()
    from clickagent.advantage_runtimes import configure_paths, RuntimeCounter
    configure_paths("coordinate")
    import torch
    from clickagent.model_features import ExtractionConfig
    torch.set_num_threads(4)
    contract = dict(read(ROOT / "runs/crossfit_candidates_v12/runtime_identity.json")["runtime_contract"])
    contract["layers"] = tuple(contract["layers"])
    contract["conditions"] = tuple(contract["conditions"])
    contract["max_memory"] = {int(k) if str(k).isdigit() else k: v for k, v in contract["max_memory"].items()}
    contract["offload_folder"] = str(OUT / "runtime/offload")
    config = ExtractionConfig(**contract)
    assert config.max_pixels == 262144 and config.dtype == "float16"
    status("loading_model", completed=0, total=112)
    runtime = GroundedRuntime(config)
    counter = RuntimeCounter()
    counter.attach(runtime.model, 16)
    write(OUT / "runtime.json", {"metadata": runtime.runtime_metadata, "torch": torch.__version__, "config": contract, "no_proposer_or_scorer_loaded": True})

    def generate(image, instruction):
        before = counter.snapshot()
        arrays, meta = runtime.infer(image, instruction, LOCAL_PROMPT, 56)
        after = counter.snapshot()
        meta["counter"] = {name: after[name] - before[name] for name in before}
        assert meta["ground_truth_in_model_input"] is False and meta["all_backbone_parameters_frozen"] and meta["own_token_forward"]
        return arrays, meta

    with Image.new("RGB", (1800, 1800), "white") as synthetic:
        draw = ImageDraw.Draw(synthetic)
        draw.rectangle([800, 800, 1000, 1000], fill="blue")
        for arm in ARMS:
            if (OUT / "runtime/warmup" / (arm + ".npz")).exists():
                continue
            side = int(arm)
            left = (1800 - side) // 2
            with synthetic.crop([left, left, left + side, left + side]).resize((side * 2, side * 2), Image.Resampling.LANCZOS) as view:
                arrays, meta = generate(view, "Click the blue rectangle.")
                archive(OUT / "runtime/warmup" / (arm + ".npz"), {}, meta)
                del arrays
    cases = read(OUT / "data/public.json")
    for index, case in enumerate(cases, 1):
        assert sha(case["image"]) == case["image_sha256"]
        with Image.open(case["image"]) as original:
            image = original.convert("RGB")
        try:
            assert list(image.size) == case["image_size"]
            for arm in case["arm_order"]:
                path = OUT / "raw" / arm / (key(case["sample_id"]) + ".npz")
                if not path.exists():
                    transform = case["transforms"][arm]
                    status("train_observations", completed=index - 1, total=112, current_case=index, arm=arm)
                    started = time.perf_counter()
                    with image.crop(transform["box"]).resize(tuple(transform["crop_size"]), Image.Resampling.LANCZOS) as view:
                        image_digest = sha256(view.tobytes()).hexdigest()
                        arrays, meta = generate(view, case["instruction"])
                    meta.update(sample_id=case["sample_id"], arm=arm, transform=transform, image_sha256=case["image_sha256"], local_RGB_sha256=image_digest, instruction=case["instruction"], world_point=pixel(meta["point"], meta["image_size"], transform), crop_and_generation_seconds=time.perf_counter() - started)
                    archive(path, arrays, meta)
                    del arrays
            status("train_observations", completed=index, total=112, arm="case_complete")
        finally:
            image.close()
    status("observations_complete", completed=112, total=112)


def analyze():
    spec = verify()
    cases = read(OUT / "data/public.json")
    assert all((OUT / "raw" / arm / (key(c["sample_id"]) + ".npz")).is_file() for c in cases for arm in ARMS)
    labels = read(OUT / "data/train_labels.json")
    refs = read(OUT / "data/reference640.json")
    geometry_rows = {row["sample_id"]: row for row in read(V34 / "geometry_details.json")}
    rows = []
    parity = defaultdict(int)
    for case in cases:
        sid = case["sample_id"]
        arms = {}
        for arm in ARMS:
            meta = load_metadata(OUT / "raw" / arm / (key(sid) + ".npz"))
            assert meta["transform"] == case["transforms"][arm] and meta["instruction"] == case["instruction"] and meta["system_prompt"] == LOCAL_PROMPT and meta["max_new_tokens"] == 56
            point = pixel(meta["point"], meta["image_size"], meta["transform"])
            assert point == meta["world_point"]
            label = labels[sid]
            hit = point is not None and point_in_geometry([point[0] / case["image_size"][0], point[1] / case["image_size"][1]], label)
            # Replay point labels directly against all boxes, inclusive edges.
            direct = point is not None and any(x0 - 1e-9 <= point[0] / case["image_size"][0] <= x1 + 1e-9 and y0 - 1e-9 <= point[1] / case["image_size"][1] <= y1 + 1e-9 for x0, y0, x1, y1 in label["boxes"])
            assert hit == direct
            arms[arm] = {"point": point, "hit": bool(hit), "valid": point is not None, "seconds": meta["elapsed_seconds"], "tokens": len(meta["generated_token_ids"]), "counter": meta["counter"], "GT_fraction": geometry_rows[sid]["arms"]["primary"][arm]["GT_fraction"]}
            if arm == "640":
                for field in ["point", "target", "generated_text", "world_point"]:
                    parity[field] += meta[field] == refs[sid][field]
        rows.append({"sample_id": sid, "source": case["source"], "arms": arms})
    a = np.asarray([row["arms"]["640"]["hit"] for row in rows], bool)
    b = np.asarray([row["arms"]["1280"]["hit"] for row in rows], bool)
    difference = b.astype(int) - a.astype(int)
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row["source"]].append(index)
    rng = np.random.default_rng(20261135)
    draws = np.asarray([100 * difference[np.concatenate([rng.choice(indices, len(indices), replace=True) for indices in groups.values()])].mean() for _ in range(10000)])

    def summarize(selected):
        return {arm: {"n": len(selected), "hits": sum(row["arms"][arm]["hit"] for row in selected), "valid": sum(row["arms"][arm]["valid"] for row in selected), "crop_intersects": sum(row["arms"][arm]["GT_fraction"] > 0 for row in selected), "crop_full_ge99pct": sum(row["arms"][arm]["GT_fraction"] >= .99 for row in selected), "mean_seconds": float(np.mean([row["arms"][arm]["seconds"] for row in selected])) if selected else None, "mean_tokens": float(np.mean([row["arms"][arm]["tokens"] for row in selected])) if selected else None, "mean_processed_pixels": float(np.mean([row["arms"][arm]["counter"]["processed_pixels"] for row in selected])) if selected else None} for arm in ARMS}

    strata = {
        "original640_miss": [row for row in rows if row["arms"]["640"]["GT_fraction"] == 0],
        "original640_full_ge99pct": [row for row in rows if row["arms"]["640"]["GT_fraction"] >= .99],
        "1280_full_ge99pct": [row for row in rows if row["arms"]["1280"]["GT_fraction"] >= .99],
    }
    result = {"complete": True, "at_utc": now(), "n": 112, "arms": summarize(rows), "repairs": int((~a & b).sum()), "harms": int((a & ~b).sum()), "delta_pp": float(100 * difference.mean()), "paired_CI95_pp": np.quantile(draws, [.025, .975]).tolist(), "bootstrap_seed": 20261135, "historical640_parity": dict(parity), "geometry_strata": {name: summarize(group) for name, group in strata.items()}, "source_strata": {source: summarize([rows[index] for index in indices]) for source, indices in groups.items()}, "scored_new_calls": 224, "synthetic_warmups": 2, "training_only_not_independent_test": True, "no_model_fitting": True, "reserve_no_forward": True, "no_automatic_adoption": True}
    write(OUT / "predictions.json", {"rows": rows})
    write(OUT / "results.json", result)
    lines = ["V35：训练集真实640/1280窗口对照", "112既有训练截图；同原主观察中心、原指令/LOCAL_PROMPT56、冻结8B、每视图262144像素上限。所有题均计入分母，两个窗口各实际运行一次。", f"640命中{int(a.sum())}/112；1280命中{int(b.sum())}/112；修复{result['repairs']}、破坏{result['harms']}；差{result['delta_pp']:.2f}个百分点，来源分层配对95%区间{result['paired_CI95_pp']}。", "实际局部代价：" + str(result["arms"]), "与V28旧640输出一致数：" + str(dict(parity)), "几何条件分层：" + str(result["geometry_strata"]), "这是原训练图上的机制实验，不是独立性能结果。扩大窗口同时增加视野并减小相对目标尺度，不能单独归因为上下文。像素上限相同不保证实际像素严格相同。没有训练评分器、使用开发标签、运行58保留图或自动替换主线。"]
    (OUT / "readout.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for item in spec["adopted_models"].values():
        assert sha(item["path"]) == item["sha256"]
    verify()
    write(OUT / "audit.json", {"complete": True, "at_utc": now(), "local_coordinate_replays": 224, "independent_point_label_replays": 224, "hashes_verified": len(spec["hashes"]), "warmup_archives": 2, "full_observation_archives": 224, "adopted_models_unchanged": True})
    write(OUT / "completion.json", {"complete": True, "at_utc": now(), "artifact_sha256": {name: sha(OUT / name) for name in ["protocol.json", "predictions.json", "results.json", "readout.txt", "audit.json"]}})
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
    if stage in ["collect", "all"]:
        collect()
    if stage in ["analyze", "all"]:
        analyze()


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        status("failed", error=repr(error), traceback=traceback.format_exc())
        raise
