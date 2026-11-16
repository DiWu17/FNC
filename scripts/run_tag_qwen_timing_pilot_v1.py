"""Small, label-free TAG/Qwen timing prototype; not a scored baseline run.

Preserves TAG's description prompt, image-mass top-10 head selection, token
averaging and threshold/connected-region postprocessing. Qwen's native visual
grid replaces MiniCPM's resampler projection. No training or candidate pools.
Only this isolated run directory is written; frozen panel protocols are intact.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/tag_qwen_timing_pilot_v1"
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "third_party/env_v17/common"),
                str(ROOT / "third_party/TAG")]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def stamp():
    return datetime.now(timezone.utc).isoformat()


def prepare(n):
    sources = {
        "pointer": ROOT / "runs/paper_scale_ablation_v2/pointerbench/public.jsonl",
        "mmb": ROOT / "runs/mmbench_gui_new_source_v1/public.json",
        "sspro": ROOT / "runs/paper_scale_ablation_v2/eval/full/public.json",
    }
    pointer = [json.loads(line) for line in sources["pointer"].read_text(encoding="utf-8-sig").splitlines() if line]
    pools = {"pro": [r for r in pointer if r["suite"] == "pro"],
             "sheets": [r for r in pointer if r["suite"] == "sheets"],
             "mmb": read(sources["mmb"]), "sspro": read(sources["sspro"])}
    chosen = []
    warmup = None
    for suite, rows in pools.items():
        ranked = sorted(rows, key=lambda r: hashlib.sha256(("tag_qwen_timing_v1|" + r["sample_id"]).encode()).hexdigest())
        seen = set()
        eligible = []
        for row in ranked:
            assert not ({"bbox", "target_bbox", "label", "answer", "target_geometry"} & set(row))
            if row["image_sha256"] in seen:
                continue
            seen.add(row["image_sha256"])
            eligible.append({**row, "suite": suite})
        chosen.extend(eligible[:n])
        if suite == "pro":
            warmup = eligible[n]
    protocol = {
        "purpose": "Unscored label-free runtime estimate, not formal accuracy evaluation",
        "created_at_utc": stamp(), "n_per_cohort": n,
        "model": "Qwen3-VL-8B-Instruct", "max_pixels": 262144,
        "dtype": "float16", "device_map": "balanced", "max_memory_GiB": [12, 12],
        "attention_backend": "eager", "views": 1, "all_layers_heads": [36, 32],
        "topk_heads_per_description_token": 10, "threshold": 0.5,
        "max_new_tokens": 1024, "do_sample": False,
        "adaptation": "Native post-merger Qwen raster; description-token own attention. Original TAG prompt and connected-region functions.",
        "fallback": "Original instruction-token prefill attention if no usable generated description tokens",
        "sources": {k: {"path": str(v), "sha256": sha(v)} for k, v in sources.items()},
        "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__), ROOT / "third_party/TAG/core_utils/tag.py", ROOT / "third_party/TAG/eval_mm/screenspot/prompt_tag.py", ROOT / "models/Qwen3-VL-8B-Instruct/config.json"]},
        "samples": chosen, "warmup": warmup, "labels_opened": False,
    }
    if (OUT / "protocol.json").exists():
        old = read(OUT / "protocol.json")
        assert old["n_per_cohort"] == n and old["source_sha256"] == protocol["source_sha256"]
        return old
    write(OUT / "protocol.json", protocol)
    return protocol


def token_span(tokenizer, ids, start, end):
    """Align a decoded character span to actual emitted token IDs, not re-encoding."""
    bounds = [0] + [len(tokenizer.decode(ids[:i], skip_special_tokens=False)) for i in range(1, len(ids) + 1)]
    return [i for i in range(len(ids)) if bounds[i] < end and bounds[i + 1] > start]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-cohort", type=int, default=5)
    args = parser.parse_args()
    protocol = prepare(args.per_cohort)
    import numpy as np
    import re
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from core_utils.tag import AttentionDrivenGrounding
    spec = importlib.util.spec_from_file_location("tag_prompt", ROOT / "third_party/TAG/eval_mm/screenspot/prompt_tag.py")
    prompts = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prompts)

    def sync():
        for device in range(torch.cuda.device_count()):
            torch.cuda.synchronize(device)

    write(OUT / "status.json", {"phase": "loading", "at": stamp(), "labels_opened": False})
    print("Loading frozen Qwen for TAG timing only", flush=True)
    load_started = time.perf_counter()
    path = ROOT / "models/Qwen3-VL-8B-Instruct"
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        path, dtype=torch.float16, device_map="balanced", max_memory={0: "12GiB", 1: "12GiB"},
        attn_implementation="eager", local_files_only=True).eval().requires_grad_(False)
    processor = AutoProcessor.from_pretrained(path, min_pixels=16384, max_pixels=262144, local_files_only=True)
    device = model.get_input_embeddings().weight.device
    tag = AttentionDrivenGrounding(processor.tokenizer, topk_layer_head=10, filter_thresold=0.5, vis_flag=False)
    sync()
    load_s = time.perf_counter() - load_started
    write(OUT / "runtime.json", {"load_seconds": load_s, "device_map": {k: str(v) for k, v in model.hf_device_map.items()},
                               "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]})
    print(f"Model loaded in {load_s:.1f}s", flush=True)

    def infer(row):
        gc.collect()
        sync()
        for d in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(d)
        started = time.perf_counter()
        with Image.open(row["image"]) as im:
            image = im.convert("RGB")
        prompt = prompts.query_prompt.format(query_text=row["instruction"], interested_key=prompts.interested_key)
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt").to(device)
        input_ids = inputs["input_ids"][0].tolist()
        input_length = len(input_ids)
        image_positions = [i for i, token in enumerate(input_ids) if token == model.config.image_token_id]
        grid = inputs["image_grid_thw"][0].tolist()
        gh, gw = grid[1] // 2, grid[2] // 2
        assert grid[0] == 1 and len(image_positions) == gh * gw
        processed_pixels = grid[1] * grid[2] * 16 * 16
        sync()
        prepared = time.perf_counter()
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=1024, do_sample=False, use_cache=True,
                                 output_attentions=True, return_dict_in_generate=True)
        sync()
        generated_at = time.perf_counter()
        ids = out.sequences[0, input_length:].tolist()
        response = processor.tokenizer.decode(ids, skip_special_tokens=False)
        match = re.search(r'"element_content_or_description"\s*:\s*"((?:\\.|[^"\\])*)"', response)
        selected = token_span(processor.tokenizer, ids, *match.span(1)) if match else []
        own_steps = [j + 1 for j in selected if j + 1 < len(out.attentions)]
        fallback = not own_steps
        if fallback:
            decoded = processor.tokenizer.decode(input_ids, skip_special_tokens=False)
            start = decoded.find(row["instruction"])
            if start < 0:
                raise RuntimeError("Cannot align instruction fallback")
            prompt_indices = token_span(processor.tokenizer, input_ids, start, start + len(row["instruction"]))
            rows_to_read = [(out.attentions[0], j) for j in prompt_indices]
        else:
            rows_to_read = [(out.attentions[j], -1) for j in own_steps]
        maps = []
        for layers, query_position in rows_to_read:
            if len(layers) != 36:
                raise RuntimeError(f"Incomplete layer capture: {len(layers)}")
            per_layer = []
            for attention in layers:
                if attention is None or attention.shape[1] != 32:
                    raise RuntimeError("Attention output missing or wrong head count")
                idx = torch.tensor(image_positions, device=attention.device)
                per_layer.append(attention[0, :, query_position, :].index_select(-1, idx).detach().float().cpu().numpy())
            heads = np.concatenate(per_layer, axis=0)
            best = np.argsort(-heads.sum(-1), kind="stable")[:10]
            maps.append(heads[best].mean(0))
        if not maps:
            raise RuntimeError("No description or instruction maps")
        grid_map = np.stack(maps).mean(0).reshape(gh, gw).T
        peak = float(grid_map.max())
        point = None
        if peak > 0:
            expanded = np.kron(grid_map / peak, np.ones((32, 32)))
            mask = expanded * (expanded >= 0.5)
            _, boxes = tag.find_connected_regions_with_confidence(mask)
            boxes = tag.filter_bbox(boxes, 0.5)
            if boxes:
                box, confidence = max(boxes, key=lambda b: (b[1], (b[0][2] - b[0][0]) * (b[0][3] - b[0][1])))
                point = [round((box[0] + box[2]) / 2 * image.width / (gw * 32)),
                         round((box[1] + box[3]) / 2 * image.height / (gh * 32))]
        sync()
        elapsed = time.perf_counter() - started
        eos = model.generation_config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        result = {"sample_id": row["sample_id"], "suite": row["suite"], "elapsed_seconds": elapsed,
                  "preprocess_seconds": prepared - started, "generation_seconds": generated_at - prepared,
                  "postprocess_seconds": time.perf_counter() - generated_at, "generated_tokens": len(ids),
                  "terminated_by_eos": bool(ids and ids[-1] in eos), "selected_description_tokens": len(own_steps),
                  "instruction_fallback": fallback, "processed_pixels": processed_pixels, "point": point,
                  "generated_text": response, "grid_hw": [gh, gw], "labels_opened": False,
                  "peak_allocated_bytes": [torch.cuda.max_memory_allocated(d) for d in range(torch.cuda.device_count())]}
        del out, inputs, maps, rows_to_read
        return result

    rows = []
    try:
        warm = infer(protocol["warmup"])
        write(OUT / "warmup.json", warm)
        print(f"Warmup {warm['elapsed_seconds']:.2f}s, {warm['generated_tokens']} tokens", flush=True)
        for i, row in enumerate(protocol["samples"]):
            case_path = OUT / "cases" / (hashlib.sha256(row["sample_id"].encode()).hexdigest() + ".json")
            result = read(case_path) if case_path.exists() else infer(row)
            write(case_path, result)
            rows.append(result)
            write(OUT / "status.json", {"phase": "timing", "done": len(rows), "total": len(protocol["samples"]), "at": stamp(), "labels_opened": False})
            print(f"{i+1}/{len(protocol['samples'])} {row['suite']}: {result['elapsed_seconds']:.2f}s, {result['generated_tokens']} tokens", flush=True)
        suites = {}
        for suite, n in {"pro": 500, "sheets": 500, "mmb": 1000, "sspro": 1581}.items():
            rr = [r for r in rows if r["suite"] == suite]
            tt = [r["elapsed_seconds"] for r in rr]
            suites[suite] = {"n_timed": len(rr), "mean_seconds": statistics.mean(tt), "median_seconds": statistics.median(tt),
                             "min_seconds": min(tt), "max_seconds": max(tt), "estimated_cohort_hours": statistics.mean(tt) * n / 3600,
                             "mean_generated_tokens": statistics.mean(r["generated_tokens"] for r in rr)}
        summary = {"complete": True, "n": len(rows), "load_seconds": load_s, "suites": suites,
                   "total_estimated_hours": sum(s["estimated_cohort_hours"] for s in suites.values()),
                   "mean_seconds": statistics.mean(r["elapsed_seconds"] for r in rows),
                   "peak_allocated_GiB_by_gpu": [max(r["peak_allocated_bytes"][i] for r in rows) / 2**30 for i in range(torch.cuda.device_count())],
                   "fallback_count": sum(r["instruction_fallback"] for r in rows),
                   "truncated_count": sum(not r["terminated_by_eos"] for r in rows),
                   "labels_opened": False, "accuracy_evaluated": False,
                   "caveat": "Five label-free cases per cohort, one warmup excluded. Prototype adaptation, not a formal baseline score or runtime guarantee.",
                   "protocol_sha256": sha(OUT / "protocol.json"), "at": stamp()}
        write(OUT / "summary.json", summary)
        write(OUT / "status.json", {"phase": "complete", "done": len(rows), "at": stamp(), "labels_opened": False})
        print(json.dumps(summary, indent=2), flush=True)
    except Exception as exc:
        write(OUT / "status.json", {"phase": "failed", "done": len(rows), "error": repr(exc), "at": stamp(), "labels_opened": False})
        raise


if __name__ == "__main__":
    main()
