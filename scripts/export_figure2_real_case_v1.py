"""Export the archived pbp_0082 Figure 2 case without running Qwen inference.

All geometry, names, candidates and internal states come from the same sealed
paper_scale_ablation_v2 query. Full-aligned logits are replayed from the frozen
feature cache and checkpoints because the prediction receipt stored winners only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clickagent.evidence_retention import RetentionModel, state_features  # noqa: E402

STEM = "9e82a4775c7a5f03be44ed7a"
SAMPLE_ID = "pbp_0082"
RUN = ROOT / "runs/paper_scale_ablation_v2"
EVAL = RUN / "eval/pro"
MODEL_DIR = ROOT / "runs/full_scaling_v18/models"
MODEL_NAMES = ["full_aligned_s1", "full_aligned_s2", "full_aligned_s3"]
SEEDS = [20261311, 20261312, 20261313]
MODEL_ROOT = ROOT / "models/Qwen3-VL-8B-Instruct"
HIGHLIGHTS = {130: ("A · selected", "#00b8aa"),
              136: ("B · header distractor", "#f07b38"),
              0: ("C · global top proposal", "#7952d3")}
SOURCE_VIEW = {"global": 0, "old_crop": 1, "local_1": 2, "local_2": 3}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def check(path: Path, expected: str):
    actual = digest(path)
    if actual != expected:
        raise RuntimeError(f"SHA mismatch: {path}: {actual} != {expected}")
    return actual


def find_jsonl(path: Path, sample_id: str):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("sample_id") == sample_id:
                return row
    raise KeyError((path, sample_id))


def token_trace(meta: dict, tokenizer) -> list[dict]:
    ids = meta["generated_token_ids"]
    prefixes = meta["generated_prefixes"]
    steps = meta["target_raw_indices"]
    positions = meta["target_query_absolute_positions"]
    assert len(ids) == len(prefixes) and len(steps) == len(positions)
    trace = []
    for token_offset, (step, absolute_pos) in enumerate(zip(steps, positions)):
        prior = prefixes[step - 2] if step > 1 else ""
        current = prefixes[step - 1]
        if not current.startswith(prior):
            raise RuntimeError("Generated decoded prefixes are not monotonic")
        token_id = int(ids[step - 1])
        trace.append({
            "name_token_offset_0_based": token_offset,
            "generation_step_1_based": int(step),
            "token_id": token_id,
            "tokenizer_token": tokenizer.convert_ids_to_tokens(token_id),
            "single_token_decode": tokenizer.decode([token_id], skip_special_tokens=True),
            "decoded_increment_in_context": current[len(prior):],
            "decoded_prefix_after_step": current,
            "query_absolute_position_0_based": int(absolute_pos),
            "raw_attention_axis_0_index": token_offset,
        })
    return trace


def processor_rgb(image: Image.Image, image_processor, fine_grid: list[int]) -> Image.Image:
    """Invert the actual processor's normalized patch sequence to an RGB preview."""
    batch = image_processor(images=[image.convert("RGB")], return_tensors="pt")
    thw = batch["image_grid_thw"].numpy()[0].tolist()
    gh, gw = map(int, fine_grid)
    if thw != [1, gh, gw]:
        raise RuntimeError(f"Processor grid changed: {thw} != {[1, gh, gw]}")
    patch = int(image_processor.patch_size)
    merge = int(image_processor.merge_size)
    temporal = int(image_processor.temporal_patch_size)
    flat = batch["pixel_values"].numpy()
    order = (0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
    blocked = flat.reshape(1, 1, gh // merge, gw // merge, merge, merge,
                           3, temporal, patch, patch)
    restored = blocked.transpose(np.argsort(order)).reshape(
        1, temporal, 3, gh * patch, gw * patch)[0, 0]
    mean = np.asarray(image_processor.image_mean, np.float32)[:, None, None]
    std = np.asarray(image_processor.image_std, np.float32)[:, None, None]
    rgb = ((restored * std + mean) * 255).transpose(1, 2, 0)
    return Image.fromarray(np.clip(np.rint(rgb), 0, 255).astype(np.uint8), "RGB")


def geometry(point: list[int], view: dict) -> dict:
    x, y = map(int, point)
    x0, y0, x1, y1 = view["original_box_xyxy"]
    gw, gh = view["merged_grid_w"], view["merged_grid_h"]
    valid = x0 <= x < x1 and y0 <= y < y1
    input_w, input_h = view["model_input_size_wh"]
    proc_w, proc_h = view["processor_size_wh"]
    in_x = (x - x0) * input_w / (x1 - x0)
    in_y = (y - y0) * input_h / (y1 - y0)
    px = in_x * proc_w / input_w
    py = in_y * proc_h / input_h
    result = {"valid_coverage": valid, "model_input_xy": [in_x, in_y],
              "processor_xy": [px, py]}
    if valid:
        col = min(gw - 1, int(np.floor((x - x0) * gw / (x1 - x0))))
        row = min(gh - 1, int(np.floor((y - y0) * gh / (y1 - y0))))
        result.update({
            "visual_cell_row_0_based": row,
            "visual_cell_col_0_based": col,
            "visual_cell_flat_0_based": row * gw + col,
            "visual_cell_original_xyxy": [
                x0 + col * (x1 - x0) / gw,
                y0 + row * (y1 - y0) / gh,
                x0 + (col + 1) * (x1 - x0) / gw,
                y0 + (row + 1) * (y1 - y0) / gh,
            ],
            "visual_cell_processor_xyxy": [
                col * proc_w / gw, row * proc_h / gh,
                (col + 1) * proc_w / gw, (row + 1) * proc_h / gh,
            ],
        })
    return result


def replay_logits(base, state, mask, active, baseline, sealed):
    torch.set_num_threads(1)
    scores = {}
    models = {}
    for name, seed in zip(MODEL_NAMES, SEEDS):
        path = MODEL_DIR / f"full_s{MODEL_NAMES.index(name) + 1}.pt"
        ck_sha = check(path, sealed["model_hashes"][name])
        saved = torch.load(path, map_location="cpu", weights_only=True)
        model = RetentionModel().eval()
        model.load_state_dict(saved["state_dict"])
        scaled = (base - saved["mean"].numpy()) / saved["std"].numpy()
        with torch.inference_mode():
            logits = model(
                base=torch.as_tensor(scaled),
                state=torch.as_tensor(state),
                mask=torch.as_tensor(mask),
                active=torch.as_tensor(active),
            ).float().numpy()
        pool = np.sort(baseline)
        order = pool[np.argsort(-logits[pool], kind="stable")]
        winner = int(order[0])
        if winner != sealed["methods"][name]["index"]:
            raise RuntimeError(f"Replay winner differs from sealed prediction: {name}")
        scores[name] = {"all_union_logits": logits,
                        "k16_order_desc_union_indices": order.tolist(),
                        "winner_union_index": winner}
        models[name] = {"seed": seed, "checkpoint_path": str(path),
                        "checkpoint_sha256": ck_sha,
                        "score_source": "posthoc_cpu_replay_of_frozen_checkpoint_and_sealed_features",
                        "sealed_winner_verified": True}
    return scores, models


def save_figures(out: Path, original: Image.Image, processed: list[Image.Image],
                 candidates: list[dict], views: list[dict], scores: dict,
                 gt_box: list[int], raw: list[dict]):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import Rectangle

    color_map = {i: color for i, (_, color) in HIGHLIGHTS.items()}

    def frame(ax, image, title):
        ax.imshow(image)
        ax.set_title(title, fontsize=12, weight="bold")
        ax.set_xlim(0, image.width)
        ax.set_ylim(image.height, 0)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])

    def draw_grid(ax, view, candidates_to_draw):
        gw, gh = view["merged_grid_w"], view["merged_grid_h"]
        pw, ph = view["processor_size_wh"]
        for col in range(gw + 1):
            ax.axvline(col * pw / gw, color="white", lw=.42, alpha=.65)
        for row in range(gh + 1):
            ax.axhline(row * ph / gh, color="white", lw=.42, alpha=.65)
        for cand in candidates_to_draw:
            ident = cand["union_index"]
            geo = cand["views"][view["view_index"]]
            if not geo["valid_coverage"]:
                continue
            c = color_map[ident]
            x0, y0, x1, y1 = geo["visual_cell_processor_xyxy"]
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                                   fill=False, edgecolor=c, linewidth=2.5))
            x, y = geo["processor_xy"]
            ax.plot(x, y, "o", color=c, markersize=9, markeredgecolor="black", markeredgewidth=.8)
            ax.text(x + 7, y - 8, HIGHLIGHTS[ident][0].split(" · ")[0], color="white",
                    fontsize=10, weight="bold", bbox={"facecolor": c, "edgecolor": "none", "pad": 2})

    highlighted = [c for c in candidates if c["union_index"] in HIGHLIGHTS]
    fig = plt.figure(figsize=(21, 11), constrained_layout=True, facecolor="white")
    gs = GridSpec(2, 3, figure=fig, width_ratios=(1.45, 1, .82))
    ax0 = fig.add_subplot(gs[:, 0]); frame(ax0, original, "Original screenshot · all 16 K16 candidates")
    for j, view in enumerate(views[1:], 1):
        x0, y0, x1, y1 = view["original_box_xyxy"]
        ax0.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                                fill=False, edgecolor=("#00a6d6", "#f5aa21", "#c075d9")[j-1],
                                linewidth=1.8, linestyle="--"))
        ax0.text(x0 + 3, y0 + 16, f"crop {j}", color="white", fontsize=9,
                 bbox={"facecolor": ("#00a6d6", "#f5aa21", "#c075d9")[j-1], "pad": 2})
    ax0.add_patch(Rectangle((gt_box[0], gt_box[1]), gt_box[2]-gt_box[0], gt_box[3]-gt_box[1],
                            fill=False, edgecolor="#37bd5b", linewidth=2, linestyle=":"))
    for c in candidates:
        x, y = c["original_xy"]
        idx = c["union_index"]
        if idx in HIGHLIGHTS:
            ax0.plot(x, y, "o", color=color_map[idx], markersize=11,
                     markeredgecolor="black", markeredgewidth=1)
            ax0.text(x + 8, y - 8, HIGHLIGHTS[idx][0].split(" · ")[0],
                     color="white", weight="bold", fontsize=10,
                     bbox={"facecolor": color_map[idx], "pad": 2})
        else:
            ax0.plot(x, y, "o", color="#eeeeee", markersize=5,
                     markeredgecolor="#333333", markeredgewidth=.6)
            ax0.text(x + 4, y - 5, str(c["pool_slot_1_based"]),
                     color="white", fontsize=7,
                     bbox={"facecolor": "#3b4750", "alpha": .85, "pad": 1})
    ax0.text(8, 760, "Green dotted box: ground truth annotation (visualization only)",
             color="white", fontsize=9, bbox={"facecolor": "#29343d", "pad": 4})

    ax1 = fig.add_subplot(gs[0, 1]); frame(ax1, processed[1], "Actual crop 1 · processor grid 16 × 16")
    draw_grid(ax1, views[1], highlighted)
    ax2 = fig.add_subplot(gs[1, 1]); frame(ax2, processed[0], "Global processor raster · grid 13 × 18")
    draw_grid(ax2, views[0], highlighted)

    ax3 = fig.add_subplot(gs[:, 2])
    first = scores[MODEL_NAMES[0]]
    order = first["k16_order_desc_union_indices"]
    vals = [float(first["all_union_logits"][i]) for i in order]
    labels = [f"C{next(c['pool_slot_1_based'] for c in candidates if c['union_index']==i)} / ID {i}" for i in order]
    colors = [color_map.get(i, "#aab3b9") for i in order]
    ax3.barh(np.arange(16), vals, color=colors, edgecolor="#4f5962", linewidth=.35)
    ax3.set_yticks(np.arange(16), labels=labels, fontsize=8)
    ax3.invert_yaxis()
    ax3.axvline(0, color="#444", linewidth=.7)
    ax3.set_xlabel("Raw readout logit $s_i$ · seed 20261311")
    ax3.set_title("Frozen full-aligned readout\nK16 rank from cached-feature replay", fontsize=12, weight="bold")
    ax3.grid(axis="x", alpha=.2)
    ax3.text(.02, -.08, "Proposal score $\\rho_i$ is separate; see candidates.csv.\n"
             "All three seeds select A (ID 130).",
             transform=ax3.transAxes, fontsize=9, va="top")
    fig.savefig(out / "figure2_overview.png", dpi=230, bbox_inches="tight")
    fig.savefig(out / "figure2_overview.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(15, 7), constrained_layout=True)
    for ax, vi, title in zip(axes, (0, 1),
                             ("Global: 13 × 18 merged cells", "Crop 1: 16 × 16 merged cells")):
        frame(ax, processed[vi], title)
        draw_grid(ax, views[vi], highlighted)
    fig.savefig(out / "grid_alignment.png", dpi=250, bbox_inches="tight")
    plt.close(fig)

    for vi in (2, 3):
        fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
        frame(ax, processed[vi], f"Crop {vi}: actual 16 × 16 merged-cell grid")
        gw, gh = views[vi]["merged_grid_w"], views[vi]["merged_grid_h"]
        for k in range(gw + 1): ax.axvline(k*processed[vi].width/gw, color="white", lw=.45, alpha=.65)
        for k in range(gh + 1): ax.axhline(k*processed[vi].height/gh, color="white", lw=.45, alpha=.65)
        fig.savefig(out / f"crop_v{vi}_grid.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    maps = []
    for vi in (0, 1):
        a = raw[vi]["attention"]
        p = a.shape[-1]
        rel = a / np.maximum(a.sum(-1, keepdims=True), 1e-12)
        maps.append(np.log1p(p*rel).mean((0, 1)).reshape(
            views[vi]["merged_grid_h"], views[vi]["merged_grid_w"]))
    high = max(float(np.quantile(m, .99)) for m in maps)
    for ax, vi, m in zip(axes, (0, 1), maps):
        frame(ax, processed[vi], f"View {vi}: mean log(1 + P × image-normalized attention)")
        im = ax.imshow(m, extent=(0, processed[vi].width, processed[vi].height, 0),
                       cmap="magma", alpha=.55, vmin=0, vmax=high, interpolation="nearest")
        draw_grid(ax, views[vi], highlighted)
    fig.colorbar(im, ax=axes.ravel().tolist(), label="Mean over target-name tokens and 96 heads; color scale 0–99th percentile")
    fig.savefig(out / "attention_feature_maps.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=ROOT / "paper/figure2_real_case_pbp_0082")
    args = ap.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)

    pipeline_path = EVAL / f"pipeline/{STEM}.json"
    index_path = EVAL / "features/index.json"
    sealed_path = EVAL / f"predictions/cases/{STEM}.json"
    public_path = RUN / "pointerbench/public.jsonl"
    labels_path = RUN / "pointerbench/labels.jsonl"
    pipeline = read_json(pipeline_path)
    sealed = read_json(sealed_path)
    index = read_json(index_path)
    entry = next(e for e in index["cases"] if e["sample_id"] == SAMPLE_ID)
    public = find_jsonl(public_path, SAMPLE_ID)
    labels = find_jsonl(labels_path, SAMPLE_ID)
    assert pipeline["sample_id"] == sealed["sample_id"] == SAMPLE_ID
    assert public["instruction"] == "Enter 3."
    assert pipeline["input_identity"] == sealed["public_identity"]
    image_path = Path(public["image"])
    check(image_path, public["image_sha256"])
    check(pipeline_path, entry["pipeline_sha256"])
    check(Path(entry["path"]), entry["sha256"])
    check(Path(entry["state_path"]), entry["state_sha256"])
    check(Path(entry["mask_path"]), entry["mask_sha256"])
    assert sealed["feature_sha256"] == entry["sha256"]
    original = Image.open(image_path).convert("RGB")
    assert list(original.size) == public["image_size"] == [1024, 768]
    original.save(out / "original.png")

    with np.load(entry["path"], allow_pickle=False) as z:
        feature = {k: z[k].copy() for k in ("points", "base", "active", "baseline")}
    state = np.load(entry["state_path"], allow_pickle=False)
    mask = np.load(entry["mask_path"], allow_pickle=False)
    assert state.shape == (len(feature["points"]), 4, 9, 294)
    assert mask.shape == state.shape[:-1]
    pool = pipeline["pools"]["c3"]["16"]
    baseline = feature["baseline"].tolist()
    assert len(pool) == len(baseline) == 16
    assert [feature["points"][i].tolist() for i in baseline] == [x["point"] for x in pool]

    from transformers import AutoProcessor  # local tokenizer/image processor only; no Qwen weights
    processor = AutoProcessor.from_pretrained(str(MODEL_ROOT), local_files_only=True)
    processor.image_processor.size = {"shortest_edge": 65536, "longest_edge": 262144}
    views = []
    raw = []
    processed = []
    source_views = [{"feature_path": pipeline["global_feature_path"],
                     "feature_sha256": pipeline["global_feature_sha256"],
                     "transform": None}] + pipeline["local_views"]
    for vi, source in enumerate(source_views):
        raw_path = Path(source["feature_path"])
        check(raw_path, source["feature_sha256"])
        with np.load(raw_path, allow_pickle=False) as z:
            metadata = json.loads(z["metadata"].item())
            arrays = {k: z[k].copy() for k in ("target_attention", "target_hidden", "image_hidden")}
        if vi == 0:
            model_input = original
            box = [0, 0, original.width, original.height]
            transform = None
        else:
            transform = source["transform"]
            box = list(transform["box"])
            original.crop(box).save(out / f"crop_v{vi}_source_256.png")
            model_input = original.crop(box).resize(tuple(transform["crop_size"]), Image.Resampling.LANCZOS)
            assert model_input.size == tuple(transform["crop_size"]) == (512, 512)
            model_input.save(out / f"crop_v{vi}_model_input_512.png")
        assert metadata["image_size"] == list(model_input.size)
        assert metadata["target_complete"] and metadata["target_available"]
        assert metadata["own_token_forward"] and metadata["terminated_by_eos"]
        assert not metadata["token_cap_reached"]
        assert not metadata["ground_truth_in_model_input"]
        display = processor_rgb(model_input, processor.image_processor, metadata["fine_grid_shape"])
        display.save(out / f"processor_v{vi}.png")
        processed.append(display)
        gh, gw = map(int, metadata["grid_shape"])
        fh, fw = map(int, metadata["fine_grid_shape"])
        assert (fh, fw) == (gh * 2, gw * 2)
        assert display.size == (fw * 16, fh * 16)
        v = {
            "view_index": vi,
            "kind": "global" if vi == 0 else "local_crop",
            "target_name": metadata["target_text"],
            "name_token_trace": token_trace(metadata, processor.tokenizer),
            "generated_text": metadata["generated_text"],
            "generated_token_ids": metadata["generated_token_ids"],
            "generated_prefixes": metadata["generated_prefixes"],
            "stages": metadata["stages"],
            "target_raw_indices_1_based": metadata["target_raw_indices"],
            "target_query_absolute_positions_0_based": metadata["target_query_absolute_positions"],
            "validity": {k: metadata[k] for k in (
                "target_complete", "target_available", "own_token_forward",
                "terminated_by_eos", "token_cap_reached", "ground_truth_in_model_input")},
            "original_box_xyxy": box,
            "transform_logged": transform,
            "crop_operation": "PIL crop(xyxy), then PIL LANCZOS resize ×2" if vi else "none",
            "model_input_size_wh": list(model_input.size),
            "model_input_size_hw": [model_input.height, model_input.width],
            "processor_size_wh": list(display.size),
            "processor_size_hw": [display.height, display.width],
            "fine_grid_h": fh, "fine_grid_w": fw,
            "merged_grid_h": gh, "merged_grid_w": gw,
            "visual_token_count": gh * gw,
            "raw_feature_path": str(raw_path),
            "raw_feature_sha256": source["feature_sha256"],
            "raw_attention_shape_token_head_cell": list(arrays["target_attention"].shape),
            "image_content_positions_0_based": metadata["image_content_positions"],
        }
        if vi == 0:
            v["processor_operation"] = "Qwen2VLImageProcessorFast smart_resize to 416×576 H×W, bicubic; spatial padding false"
        else:
            v["processor_operation"] = "512×512 already factor-32 aligned; no spatial resize or padding"
        views.append(v)
        raw.append({"metadata": metadata, "attention": arrays["target_attention"]})
        reconstructed, valid = state_features(arrays, metadata, feature["points"], original.size, transform)
        t = len(metadata["target_raw_indices"])
        if not np.array_equal(reconstructed, state[:, vi, :t, :]):
            raise RuntimeError(f"Recomputed 294-d state differs from sealed cache in view {vi}")
        expected_mask = np.broadcast_to(valid[:, None], (len(valid), t))
        if not np.array_equal(expected_mask, mask[:, vi, :t]):
            raise RuntimeError(f"Recomputed mask differs from sealed cache in view {vi}")
        if mask[:, vi, t:].any():
            raise RuntimeError("Unexpected nonempty mask in padded target tokens")

    scores, models = replay_logits(feature["base"], state, mask, feature["active"],
                                   feature["baseline"], sealed)
    candidates = []
    for slot, (idx, row) in enumerate(zip(baseline, pool)):
        pt = row["point"]
        mapped = [geometry(pt, v) for v in views]
        coverage = [m["valid_coverage"] for m in mapped]
        if coverage != feature["active"][idx].tolist():
            raise RuntimeError(f"Coverage differs from sealed feature, ID {idx}")
        token_counts = mask[idx].sum(-1).astype(int).tolist()
        for vi, m in enumerate(mapped):
            m["valid_name_token_count"] = token_counts[vi]
            assert (token_counts[vi] > 0) == coverage[vi]
        cand = {
            "pool_slot_0_based": slot, "pool_slot_1_based": slot + 1,
            "union_index": int(idx), "original_xy": pt,
            "source_view_index": SOURCE_VIEW[row["source"]],
            "source_label_in_pipeline": row["source"],
            "proposal_rank": row["source_rank"],
            "proposal_candidate_index_in_source_view": row["candidate_index"],
            "proposal_parent_visual_cell": row["parent_index"],
            "proposal_score_rho": row["original_rank_score"],
            "source_local_xy_logged": row.get("local_point"),
            "valid_coverage_mask_v0_to_v3": coverage,
            "valid_name_token_counts_v0_to_v3": token_counts,
            "views": mapped,
            "highlight": HIGHLIGHTS[idx][0] if idx in HIGHLIGHTS else None,
        }
        for name in MODEL_NAMES:
            logits = scores[name]["all_union_logits"]
            order = scores[name]["k16_order_desc_union_indices"]
            cand[f"{name}_raw_logit_s_i"] = float(logits[idx])
            cand[f"{name}_k16_rank_1_based"] = order.index(idx) + 1
        candidates.append(cand)
    assert candidates[8]["union_index"] == 130
    assert candidates[8]["original_xy"] == [674, 374]
    assert [v["target_name"] for v in views] == [
        "the digit '3' on the numeric keypad", "3", "Manager PIN input field", "Amount field"]

    highlight_indices = list(HIGHLIGHTS)
    selected_raw = np.full((len(highlight_indices), 4, 9, 3, 32), np.nan, np.float32)
    selected_normalized = selected_raw.copy()
    for vi, datum in enumerate(raw):
        a = datum["attention"]
        rel = a / np.maximum(a.sum(-1, keepdims=True), 1e-12)
        for hi, idx in enumerate(highlight_indices):
            c = next(c for c in candidates if c["union_index"] == idx)
            geo = c["views"][vi]
            if not geo["valid_coverage"]:
                continue
            flat = geo["visual_cell_flat_0_based"]
            t = len(a)
            selected_raw[hi, vi, :t] = a[..., flat].reshape(t, 3, 32)
            selected_normalized[hi, vi, :t] = rel[..., flat].reshape(t, 3, 32)
    evidence_data = {
        "pool_slot_0_based": np.arange(16, dtype=np.int32),
        "union_index": np.asarray(baseline, np.int32),
        "original_xy": feature["points"][baseline].astype(np.int32),
        "base_32": feature["base"][baseline],
        "active_4": feature["active"][baseline],
        "evidence_294": state[baseline],
        "name_token_mask": mask[baseline],
        "highlight_union_index": np.asarray(highlight_indices, np.int32),
        "highlight_evidence_294": state[highlight_indices],
        "highlight_name_token_mask": mask[highlight_indices],
        "highlight_raw_attention_at_cell_token_view_layer_head": selected_raw,
        "highlight_image_normalized_attention_at_cell_token_view_layer_head": selected_normalized,
        "head_layer_numbers": np.repeat(np.asarray([15, 28, 36], np.int32), 32),
        "head_indices_within_layer": np.tile(np.arange(32, dtype=np.int32), 3),
    }
    for vi, datum in enumerate(raw):
        a = datum["attention"]
        gh, gw = views[vi]["merged_grid_h"], views[vi]["merged_grid_w"]
        evidence_data[f"raw_attention_v{vi}_token_layer_head_row_col"] = a.reshape(len(a), 3, 32, gh, gw)
    np.savez_compressed(out / "evidence.npz", **evidence_data)

    with (out / "candidates.csv").open("w", encoding="utf-8-sig", newline="") as f:
        fields = ["pool_slot_1_based", "pool_slot_0_based", "union_index", "x_original", "y_original",
                  "source_view_index", "source_label_in_pipeline", "proposal_rank",
                  "proposal_candidate_index_in_source_view", "proposal_parent_visual_cell",
                  "proposal_score_rho", "valid_coverage_mask_v0_to_v3"]
        for name in MODEL_NAMES:
            fields += [f"{name}_raw_logit_s_i", f"{name}_k16_rank_1_based"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for c in candidates:
            row = {k: c.get(k) for k in fields}
            row["x_original"], row["y_original"] = c["original_xy"]
            row["valid_coverage_mask_v0_to_v3"] = "".join(map(lambda x: "1" if x else "0", c["valid_coverage_mask_v0_to_v3"]))
            writer.writerow(row)
    (out / "candidates.json").write_text(json.dumps(candidates, ensure_ascii=False, indent=2), encoding="utf-8")

    score_rows = []
    for name in MODEL_NAMES:
        score_rows.append({"model": name, "seed": models[name]["seed"],
                           "winner_union_index": scores[name]["winner_union_index"],
                           "winner_point_original_xy": [674, 374],
                           "ranked_union_indices": scores[name]["k16_order_desc_union_indices"],
                           "raw_logits_in_original_pool_order": [float(scores[name]["all_union_logits"][i]) for i in baseline],
                           "logit_definition": "unnormalized RetentionModel forward output; larger ranks higher",
                           "score_source": models[name]["score_source"]})
    (out / "scores.json").write_text(json.dumps(score_rows, indent=2), encoding="utf-8")

    gt_box = labels["bbox_pixels"]
    save_figures(out, original, processed, candidates, views, scores, gt_box, raw)
    manifest = {
        "case": SAMPLE_ID,
        "selection_reason": "Requested case has complete four-view raw caches, archived C16, sealed three-seed predictions, and matching cached 294-d evidence.",
        "run_id": "paper_scale_ablation_v2/eval/pro",
        "protocol": "original K16 / c3 pool / max_pixels 262144 per view / forced Top1 / full aligned readout",
        "instruction": public["instruction"],
        "original_image": {"path": str(image_path), "export": "original.png", "size_wh": list(original.size),
                           "size_hw": [original.height, original.width],
                           "sha256": public["image_sha256"]},
        "ground_truth_posthoc_visualization_only": {"bbox_original_xyxy": gt_box,
                                                    "point_original_xy": labels["point_pixels"],
                                                    "included_in_model_input": False},
        "global_name_verified": views[0]["target_name"],
        "sealed_selected_point_original_xy": [674, 374],
        "sealed_selected_union_index": 130,
        "selected_pool_slot_0_based": 8,
        "highlight_union_indices": highlight_indices,
        "views": views,
        "candidate_pool": {"size": 16, "pipeline_key": "pools.c3.16", "order": "pipeline order retained",
                           "union_indices_in_pool_order": baseline, "candidates_file": "candidates.json"},
        "models": models,
        "scoring": {"readout_scores_file": "scores.json",
                    "proposal_score_field": "proposal_score_rho",
                    "normalization_for_display": "none: all plotted readout values are raw logits",
                    "scaler": "each checkpoint's training-only saved mean/std applied to base features",
                    "winner_rule": "highest logit among sorted original K16 union indices; stable tie order",
                    "original_prediction_receipt_has_candidate_logits": False,
                    "logits_replayed_from_sealed_features": True},
        "visual_indexing": {"postmerge_row_col_and_flat_start": 0,
                            "flatten_order": "row-major: flat = row * grid_width + col",
                            "cell_rule": "floor(((original_xy - view_original_box_origin) / box_size) * [gw,gh]); half-open bounds",
                            "patch_size_px": 16, "spatial_merge_size": 2,
                            "merged_cell_size_processor_px": 32,
                            "spatial_padding": False,
                            "temporal_patch_size": 2,
                            "static_frame_temporally_repeated": True},
        "evidence": {
            "file": "evidence.npz", "dtype": "float16 for sealed evidence; float32 for raw attention",
            "evidence_shape": [16, 4, 9, 294],
            "mask_shape": [16, 4, 9],
            "channels_0_based": {"0:96": "log1p(P_v * attention_at_cell / image_attention_mass), 96 layer/head order",
                                 "96:192": "image-normalized attention_at_cell / per-head image peak",
                                 "192:288": "log1p(P_v * mean of valid 3x3 neighboring image-normalized attention cells)",
                                 "288:291": "cosine(image_hidden_at_cell, target_hidden_query) for layers 15,28,36",
                                 "291:294": "log1p(norm(target_hidden_query)/sqrt(4096)) for layers 15,28,36"},
            "head_order": raw[0]["metadata"]["head_order"],
            "target_token_alignment": "target_raw_indices are 1-based generated steps; raw_attention target axis follows those steps, after own-token forward",
            "raw_attention_definition": "original full-context attention mass on image tokens, before image-mass renormalization",
            "raw_attention_arrays": "raw_attention_v{0..3}_token_layer_head_row_col",
            "highlight_cell_attention_arrays": ["highlight_raw_attention_at_cell_token_view_layer_head",
                                                "highlight_image_normalized_attention_at_cell_token_view_layer_head"],
            "attention_heatmap_definition": "mean over target-name tokens and all 96 heads of log1p(P_v * (raw image-token attention / sum over image cells)); color 0 to shared 99th percentile",
            "recomputed_294d_evidence_exactly_equals_sealed_cache": True,
            "recomputed_mask_exactly_equals_sealed_cache": True,
        },
        "source_paths_sha256": {
            "public_jsonl": {"path": str(public_path), "sha256": digest(public_path)},
            "posthoc_labels_jsonl": {"path": str(labels_path), "sha256": digest(labels_path)},
            "pipeline": {"path": str(pipeline_path), "sha256": entry["pipeline_sha256"]},
            "feature_npz": {"path": entry["path"], "sha256": entry["sha256"]},
            "feature_state_npy": {"path": entry["state_path"], "sha256": entry["state_sha256"]},
            "feature_mask_npy": {"path": entry["mask_path"], "sha256": entry["mask_sha256"]},
            "sealed_predictions": {"path": str(sealed_path), "sha256": digest(sealed_path)},
            "proposer_checkpoint": {"path": str(ROOT / "runs/crossfit_candidates_v12/proposers/full.pt"),
                                    "sha256": pipeline["proposer_sha256"]},
        },
        "code_provenance": {
            "git_commit": "unavailable: this workspace has no readable .git repository",
            "export_script": str(Path(__file__).resolve()),
            "export_script_sha256": digest(Path(__file__)),
            "evidence_retention_py_sha256": digest(ROOT / "src/clickagent/evidence_retention.py"),
            "point_aligned_inference_py_sha256": digest(ROOT / "src/clickagent/point_aligned_inference.py"),
            "public_model_py_sha256": digest(ROOT / "src/clickagent/public_model.py"),
            "original_scoring_py_sha256": digest(ROOT / "scripts/run_paper_scale_eval_v1.py"),
            "image_preprocessor_config_sha256": digest(MODEL_ROOT / "preprocessor_config.json"),
            "tokenizer_json_sha256": digest(MODEL_ROOT / "tokenizer.json"),
        },
        "missing": [
            {"status": "待补（仅当要求原运行直接写出的逐候选 logit 收据）",
             "record": "原 V2 prediction case JSON 中的 16 个原始 readout logit",
             "minimum_step": "从已封存 state/mask/base 缓存和冻结检查点重放 scorer，并另存独立 SHA 收据；本包的 scores.json 已完成确定性 CPU 重放并核对三个 winner。"},
            {"status": "待补（仅当要求当次推理保存的中间 PNG 文件）",
             "record": "原 V2 运行直接落盘的 crop/processor PNG",
             "minimum_step": "原运行未保存这些 PNG；本包已按归档框、LANCZOS 和本地固定 Qwen 图像处理器重构。若需当次运行落盘字节，只能在未来实验中预先启用中间图保存。"},
        ],
        "archive_limitations": [
            "Per-candidate logits were not stored in the original prediction receipt; exported values are deterministic CPU replay from sealed features and frozen checkpoints, with all three winners verified.",
            "Crop and processor PNGs were not saved by the inference run; they are reconstructed from the same original screenshot and archived crop transforms. Processor PNGs invert the local Qwen image processor's cached-free patch operation and reproduce its processed RGB raster, not an archived PNG.",
        ],
    }
    check(ROOT / "runs/crossfit_candidates_v12/proposers/full.pt", pipeline["proposer_sha256"])
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    readme = f"""# Figure 2 · real pbp_0082 case

This package uses the **same** PointerBench-Pro query for the screenshot, four
name traces, actual crop geometry, original K16 proposals, cached evidence and
three full-aligned readouts. Instruction: **Enter 3.** Original image: 1024×768.

The global name is **{views[0]['target_name']}**. Local names are
**{views[1]['target_name']}**, **{views[2]['target_name']}**, and
**{views[3]['target_name']}**. All four names were complete EOS-terminated
generations, not truncated. All three full-aligned seeds select original point
**(674, 374)**, union ID 130, original K16 slot 9 (1-based).

## Files

- `figure2_overview.png` / `.pdf`: original, crop 1, global grid and raw seed-1 logits.
- `grid_alignment.png`: exact global/crop-1 postmerge visual-cell mapping.
- `original.png`; `crop_v1..v3_source_256.png`: source pixels before 2× resize.
- `crop_v1..v3_model_input_512.png`: LANCZOS crops passed to the processor.
- `processor_v0..v3.png`: RGB raster inverted from the pinned processor's patch
  output at the logged grid (global 576×416; locals 512×512).
- `crop_v2_grid.png`, `crop_v3_grid.png`: additional local view grids.
- `attention_feature_maps.png`: mean **log(1 + P × image-normalized attention)**,
  averaged across the actual target-name tokens and 96 heads. It is not raw attention.
- `manifest.json`: full name-token traces, source hashes, transformations,
  coordinates, visual-cell indices, score definitions and provenance.
- `candidates.csv` / `.json`: all 16 candidates in original pool order, proposal
  score $\\rho_i$ separately from three raw readout logits $s_i$.
- `scores.json`: all K16 raw logits, rankings and replay provenance by seed.
- `evidence.npz`: K16 294-d evidence/masks, four raw head maps and highlighted
  candidate cell values, organized by token/view/layer/head.

## Important provenance

The original sealed prediction receipt stores winners but **not all logits**.
This export replays the three frozen checkpoints over the sealed feature cache
on CPU; the winners match the receipt exactly. The replayed 294-d evidence and
mask match the archived arrays element for element. No Qwen model weights were
loaded and no candidate pool was regenerated.

The PNG crops were not saved by the original inference process. They were
reconstructed from the archived screenshot and exact crop boxes with its
LANCZOS ×2 operation. The processor RGB rasters were obtained by running only
the pinned image processor and inverting its patch layout; no new backbone
inference was performed. The green dashed box in the overview is a **post-hoc
ground-truth annotation for visualization only**.

Coordinate convention: original coordinates and crop boxes are pixel xy / xyxy;
box upper bounds are exclusive. Visual cells use 0-based row and column with
row-major flat index `row * grid_width + col`. The merged cell spans 32×32
processed pixels (16-pixel patches, merge size 2).

Reproduce in the original workspace:

```powershell
& .venv\\python.exe -B -X utf8 scripts/export_figure2_real_case_v1.py
```
"""
    (out / "README.md").write_text(readme, encoding="utf-8")
    checksum_lines = [f"{digest(path)}  {path.name}" for path in sorted(out.iterdir())
                      if path.is_file() and path.name != "SHA256SUMS.txt"]
    (out / "SHA256SUMS.txt").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    zip_path = out.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for path in sorted(out.iterdir()):
            if path.is_file():
                z.write(path, path.name)
        z.write(Path(__file__), "export_figure2_real_case_v1.py")
    print(json.dumps({"sample_id": SAMPLE_ID, "out": str(out), "zip": str(zip_path),
                      "selected": [674, 374], "models": MODEL_NAMES,
                      "zip_sha256": digest(zip_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
