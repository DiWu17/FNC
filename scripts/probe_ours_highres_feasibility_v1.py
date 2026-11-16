"""Label-free one-case resource preflight for a separate high-pixel K16 variant.

This does not alter the frozen 262144-pixel OurRuntime or claim accuracy.
Use CUDA_VISIBLE_DEVICES to choose the available card(s) before launching.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def measure(name: str, function, torch, output: dict):
    for device in range(torch.cuda.device_count()):
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    before = time.perf_counter()
    result = function()
    for device in range(torch.cuda.device_count()):
        torch.cuda.synchronize(device)
    output["stages"][name] = {
        "seconds": time.perf_counter() - before,
        "gpu": [
            {
                "logical_device": device,
                "peak_allocated_mib": round(torch.cuda.max_memory_allocated(device) / 2**20, 1),
                "peak_reserved_mib": round(torch.cuda.max_memory_reserved(device) / 2**20, 1),
                "end_allocated_mib": round(torch.cuda.memory_allocated(device) / 2**20, 1),
            }
            for device in range(torch.cuda.device_count())
        ],
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-pixels", type=int, default=1024 * 1024)
    parser.add_argument("--case-index", type=int, default=0)
    args = parser.parse_args()
    if args.max_pixels != 1024 * 1024 or args.case_index != 0:
        raise ValueError("This prespecified preflight uses TRAIN public case 0 and a 1M cap only")

    import numpy as np
    from PIL import Image
    import torch

    import run_contribution_v16 as contribution
    import run_quality_readout_v15 as base
    from clickagent.action_model import ActionQwenRuntime
    from clickagent.advantage_runtimes import RuntimeCounter
    from clickagent.conditioned_features import ConditionedFeatureRuntime
    from clickagent.evidence_retention import RetentionModel, STATE_DIM

    class ExperimentalHighPixelRuntime(ConditionedFeatureRuntime):
        def __init__(self, config):
            # V37 used this same loader bypass. The frozen public runtime is untouched.
            ActionQwenRuntime.__init__(self, config)

    visible = torch.cuda.device_count()
    if visible not in (1, 2):
        raise RuntimeError(f"Expected one or two explicitly visible GPUs, got {visible}")
    run = ROOT / "runs/high_pixel_feasibility_v1"
    public = json.loads((ROOT / "runs/paired_scale_v35/data/public.json").read_text(encoding="utf-8"))
    case = public[args.case_index]
    assert case["cohort"] == "train" and all(
        key not in case for key in ("target_geometry", "target_bbox", "label", "labels", "ground_truth")
    )
    output = {
        "purpose": "Resource preflight, not accuracy evaluation",
        "at_utc": datetime.now(timezone.utc).isoformat(),
        "label_files_opened": False,
        "frozen_our_runtime_modified": False,
        "new_variant": "same frozen weights and K16 mathematics; processor max_pixels=1048576",
        "max_pixels_per_view": args.max_pixels,
        "case_source": "previously used TRAIN public manifest, first case; no labels",
        "sample_id": case["sample_id"],
        "image_size": case["image_size"],
        "visible_devices": [torch.cuda.get_device_name(i) for i in range(visible)],
        "stages": {},
        "views": [],
        "completed": False,
    }
    out = run / "ours_k16_1m_single_case.json"
    try:
        legacy = base.Extractor()  # Loads the exact frozen V12 proposer on CPU.
        config = replace(
            legacy.config,
            max_pixels=args.max_pixels,
            device_map="balanced",
            max_memory={**{i: "13GiB" for i in range(visible)}, "cpu": "24GiB"},
            offload_folder=str(run / "offload"),
        )
        runtime = measure("model_load", lambda: ExperimentalHighPixelRuntime(config), torch, output)
        output["effective_device_map_counts"] = {}
        for device in runtime.runtime_metadata["effective_device_map"].values():
            key = str(device)
            output["effective_device_map_counts"][key] = output["effective_device_map_counts"].get(key, 0) + 1
        counter = RuntimeCounter()
        counter.attach(runtime.model, 16)
        from clickagent.multiregion_proposals import plan_multiregion_crops, build_c1_c3_pools
        from clickagent.evidence_retention import build_base, state_features

        image = measure("image_open", lambda: Image.open(case["image"]).convert("RGB"), torch, output)
        assert list(image.size) == case["image_size"]
        try:
            view_data = []

            def extract(view, transform):
                before = counter.snapshot()
                arrays, meta = runtime.infer_spatial_features(view, case["instruction"])
                after = counter.snapshot()
                output["views"].append({
                    "view_index": len(output["views"]),
                    "input_size": list(view.size),
                    "grid_shape": meta["grid_shape"],
                    "processed_pixels": after["processed_pixels"] - before["processed_pixels"],
                    "target_available": meta["target_available"],
                    "target_tokens": len(meta["target_raw_indices"]),
                    "transform": transform,
                })
                view_data.append((arrays, meta, transform))
                return arrays, meta

            arrays, meta = measure("global_qwen", lambda: extract(image, None), torch, output)
            global_result = measure(
                "global_proposer",
                lambda: base.predict_raw(legacy.proposer, arrays, meta, list(image.size), topk=64),
                torch, output,
            )
            local = []
            planned = plan_multiregion_crops(global_result, list(image.size))
            for index, crop in enumerate(planned):
                transform = crop["transform"]
                with image.crop(transform["box"]).resize(
                    tuple(transform["crop_size"]), Image.Resampling.LANCZOS
                ) as view:
                    arrays, meta = measure(
                        f"local_{index + 1}_qwen",
                        lambda view=view, transform=transform: extract(view, transform),
                        torch, output,
                    )
                result = measure(
                    f"local_{index + 1}_proposer",
                    lambda arrays=arrays, meta=meta, transform=transform: base.predict_raw(
                        legacy.proposer, arrays, meta, transform["crop_size"], topk=64
                    ),
                    torch, output,
                )
                local.append({"region_index": crop["region_index"], "result": result, "transform": transform})

            pipeline = {"global_result": global_result, "local_views": local,
                        "pools": build_c1_c3_pools(global_result, local)}
            base_record = measure("build_k16", lambda: build_base(pipeline, list(image.size)), torch, output)
            output["all_candidates"] = len(base_record["points"])
            output["k16_candidates"] = len(base_record["baseline"])

            def score_k16():
                features, masks = [], []
                for arrays, meta, transform in view_data:
                    f, m = state_features(arrays, meta, base_record["points"], list(image.size), transform)
                    features.append(f)
                    masks.append(m)
                tmax = max(1, *(f.shape[1] for f in features))
                state = np.zeros((len(base_record["points"]), 4, tmax, STATE_DIM), np.float16)
                mask = np.zeros((len(base_record["points"]), 4, tmax), bool)
                for j, (f, m) in enumerate(zip(features, masks)):
                    state[:, j, : f.shape[1]] = f
                    mask[:, j, : f.shape[1]] = m[:, None]
                ckpt = torch.load(
                    ROOT / "runs/quality_readout_v15_3k/models/expanded_s1.pt",
                    map_location="cpu", weights_only=True,
                )
                model = RetentionModel().eval().requires_grad_(False)
                model.load_state_dict(ckpt["state_dict"])
                record = dict(base_record, sample_id=case["sample_id"],
                              n=len(base_record["points"]), state=state, mask=mask)
                if not len(base_record["baseline"]):
                    return None
                inputs, _ = contribution.build_inputs(
                    [record], ckpt["mean"].numpy(), ckpt["std"].numpy(), "aligned", "cpu"
                )
                with torch.inference_mode():
                    scores = model(**inputs).float().numpy()
                pool = np.sort(base_record["baseline"])
                order = pool[np.argsort(-scores[pool], kind="stable")]
                return base_record["points"][order[0]].tolist()

            output["point_for_pipeline_sanity_only"] = measure("k16_ranker_cpu", score_k16, torch, output)
            output["completed"] = True
            output["full_k16_path_exercised"] = bool(base_record["baseline"].size and planned)
        finally:
            image.close()
    except BaseException as exc:
        output["error"] = type(exc).__name__ + ": " + str(exc)
        output["traceback"] = traceback.format_exc()[-6000:]
    finally:
        write(out, output)
        print(json.dumps({k: output.get(k) for k in (
            "completed", "full_k16_path_exercised", "error", "visible_devices", "all_candidates", "k16_candidates"
        )}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
