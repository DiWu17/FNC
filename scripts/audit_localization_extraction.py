"""Audit frozen GUI coordinate/attention extraction without fitting anything.

Cached comparisons are paired within each run. Optional live checks compare
hooked/unhooked native generation, official QK attention, and processor layout.
All label use happens in offline scoring after model calls. Test diagnostics
never replace predictions, tune heads, or choose deployment thresholds.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

import numpy as np


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(s) for s in Path(path).read_text(encoding="utf-8").splitlines() if s.strip()]


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def key(sample_id):
    return hashlib.sha256(sample_id.encode()).hexdigest()[:24]


def hit(point, case):
    from clickagent.public_evaluation import point_in_geometry
    if point is None:
        return False
    w, h = case["image_size"]
    return point_in_geometry([float(point[0]) / w, float(point[1]) / h], case["target_geometry"])


def parse_point(text, size):
    match = re.fullmatch(r"\s*CLICK[ \t]+([0-9]+)(?:[ \t]*,[ \t]*|[ \t]+)([0-9]+)\s*", text)
    if match is None or any(int(v) > 1000 for v in match.groups()):
        return None
    return [round(int(v) * (side - 1) / 1000) for v, side in zip(match.groups(), size)]


def centers(grid, size):
    gh, gw = grid
    w, h = size
    return np.asarray([[min(w - 1, round((col + .5) * w / gw)),
                        min(h - 1, round((row + .5) * h / gh))]
                       for row in range(gh) for col in range(gw)])


def cell(point, grid, size):
    gh, gw = grid
    w, h = size
    return min(gh - 1, int(point[1] * gh / h)) * gw + min(gw - 1, int(point[0] * gw / w))


def cached_public(root):
    result, examples = {}, []
    for name in ("screenspot_v2", "screenspot_pro", "osworld_g", "venusbench_gd"):
        folder = root / "runs/public_v1" / name
        counts = Counter()
        for case in read_jsonl(folder / "selected_manifest.jsonl"):
            if not case["expected_actionable"] or case.get("target_geometry") is None:
                continue
            counts["n"] += 1
            stem = key(case["sample_id"])
            native = read_json(folder / "native" / f"{stem}.json")
            row = read_json(folder / "rows" / f"{stem}.json")["rows"][0]
            grid = row["grid_shape"]
            point = row["localization_point"]
            points = centers(grid, case["image_size"])
            valid_centers = np.asarray([hit(p, case) for p in points])
            native_point = None if native["truncated"] else parse_point(native["generated_text"], case["image_size"])
            good_native, good_att = hit(native_point, case), hit(point, case)
            counts["native_lax_hits"] += good_native
            counts["attention_hits"] += good_att
            counts["any_valid_grid_center"] += bool(valid_centers.any())
            counts["native_hit_attention_miss"] += good_native and not good_att
            counts["native_hit_attention_miss_no_valid_center"] += good_native and not good_att and not valid_centers.any()
            counts["native_hit_attention_miss_valid_center_exists"] += good_native and not good_att and valid_centers.any()
            if good_native:
                counts["native_hit_quantized_to_own_cell_still_hits"] += hit(points[cell(native_point, grid, case["image_size"])], case)
            with np.load(folder / "features" / f"{stem}.npz", allow_pickle=False) as z:
                raw = z["prefill_raw_map"]
            head_hits = valid_centers[raw.argmax(-1)]
            counts["any_of_96_head_peaks_hits_oracle"] += bool(head_hits.any())
            counts["native_hit_attention_miss_some_head_peak_hits_oracle"] += good_native and not good_att and head_hits.any()
            idx = row["point_readout"]["patch_index"]
            gh, gw = grid
            rr, cc = divmod(idx, gw)
            w, h = case["image_size"]
            alternatives = {"identity": points[idx],
                "horizontal_flip": [(gw - cc - .5) * w / gw, (rr + .5) * h / gh],
                "vertical_flip": [(cc + .5) * w / gw, (gh - rr - .5) * h / gh],
                "rotate_180": [(gw - cc - .5) * w / gw, (gh - rr - .5) * h / gh],
                "column_major": [(idx // gh + .5) * w / gw, (idx % gh + .5) * h / gh]}
            for label, candidate in alternatives.items():
                counts[f"mapping_diagnostic_{label}_hits"] += hit(candidate, case)
            if good_native and not good_att:
                examples.append({"sample_id": case["sample_id"], "dataset": case["dataset"],
                    "instruction": case["instruction"], "native_text": native["generated_text"],
                    "native_point": native_point, "attention_point": point,
                    "grid_shape": grid, "image_size": case["image_size"],
                    "valid_center_exists": bool(valid_centers.any()),
                    "any_head_peak_correct_oracle": bool(head_hits.any())})
        result[name] = {k: int(v) for k, v in counts.items()}
    return {"datasets": result, "totals": dict(sum((Counter(v) for v in result.values()), Counter())),
            "examples": examples,
            "note": "Paired original public cohort. Native lax accepts completed whole CLICK only. Attention uses a different decision prompt. Any-head and remapping checks are test-label diagnostics, not new policies."}


def cached_candidates(root):
    folder = root / "runs/candidate_v2"
    excluded = set(read_json(folder / "evaluation_exclusions.json")["sample_ids"])
    counts, discrepancies, proposals = Counter(), [], {}
    for case in read_jsonl(folder / "manifest.jsonl"):
        if case["split"] != "test" or case["case_kind"] != "click" or case["sample_id"] in excluded:
            continue
        counts["n_click"] += 1
        proposal = read_json(folder / "proposals" / f"{key(case['sample_id'])}.json")
        proposals[case["sample_id"]] = proposal
        native = proposal["native_proposal_point"]
        parsed = parse_point(proposal["native_text"], case["image_size"])
        counts["native_point_available"] += native is not None
        counts["native_wait_fallback"] += native is None and proposal["native_text"].strip() == "WAIT"
        counts["native_point_hits"] += hit(native, case)
        counts["first_candidate_hits"] += hit(proposal["points"][0], case)
        counts["any_candidate_hits"] += any(hit(p, case) for p in proposal["points"])
        if native is not None:
            counts["native_point_equals_first"] += native == proposal["points"][0]
            counts["native_point_equals_reparsed_text"] += native == parsed
            if native != parsed or native != proposal["points"][0]:
                discrepancies.append({"sample_id": case["sample_id"], "proposal": proposal})
    execution_sources = {}
    for method, policy in (("candidate_combined", "select_four"), ("candidate_hidden", "verify_first")):
        rows = [r for r in read_jsonl(folder / "predictions.jsonl")
                if r["method"] == method and r["policy"] == policy and r["calibration_error_budget"] == .05
                and r["execute"] and not r["correct_candidate"] and r["sample_id"] in proposals]
        sources = Counter(proposals[r["sample_id"]]["sources"][r["candidate_index"]] for r in rows)
        execution_sources[f"{method}/{policy}"] = {"wrong_location_executions_click_requests": len(rows),
            "selected_sources": dict(sources),
            "native_wait_before_wrong_execution": sum(proposals[r["sample_id"]]["native_text"].strip() == "WAIT" for r in rows)}
    return {"counts": dict(counts), "discrepancies": discrepancies, "execution_sources": execution_sources,
            "note": "Native coordinates are upstream of the verifier; downstream refusals do not count as coordinate-extraction failures."}


def processor_layout(processor):
    """Known pixel IDs establish native post-merge raster order independently."""
    from PIL import Image
    gh, gw, merge, patch = 4, 6, 2, 16
    image = np.zeros((gh * merge * patch, gw * merge * patch, 3), np.uint8)
    for row in range(gh):
        for col in range(gw):
            image[row*32:(row+1)*32, col*32:(col+1)*32] = row * gw + col
    output = processor.image_processor(images=Image.fromarray(image), do_resize=False,
        do_normalize=False, do_rescale=False, return_tensors="pt")
    grouped = output["pixel_values"].reshape(gh * gw, merge * merge, -1).float()
    got = grouped.mean((1, 2)).numpy()
    expected = np.arange(gh * gw)
    return {"grid_thw": output["image_grid_thw"].tolist(),
            "expected_merged_cell_ids": expected.tolist(), "observed_merged_cell_ids": got.tolist(),
            "max_abs_error": float(np.abs(got - expected).max()), "raster_order_correct": bool(np.array_equal(got, expected))}


def live(root, destination):
    import torch
    from PIL import Image
    from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb, eager_attention_forward
    from clickagent.action_grounding import AttentionPointGrounder
    from clickagent.action_model import RawSelectedLayerRecorder
    from clickagent.candidate_model import CandidateRuntime
    from clickagent.model_features import ExtractionConfig, reconstruct_attention_row
    from clickagent.public_model import PUBLIC_TEXT_ACTION_SYSTEM_PROMPT

    identity = read_json(root / "runs/candidate_v2/identity.json")
    cfg = identity["config"]
    cfg["layers"] = tuple(cfg["layers"])
    cfg["conditions"] = tuple(cfg["conditions"])
    cfg["max_memory"] = {int(k) if k.isdigit() else k: v for k, v in cfg["max_memory"].items()}
    cfg["offload_folder"] = str((destination / "offload").resolve())
    print("Loading frozen Qwen for extraction checks", flush=True)
    runtime = CandidateRuntime(ExtractionConfig(**cfg))
    grounder = AttentionPointGrounder.load(root / "runs/action_v3/grounder/grounder.joblib")
    layout = processor_layout(runtime.processor)
    save(destination / "processor_layout.json", layout)
    # Fixed illustrative controls: same screenshot, different target, and a
    # high-resolution proposal failure. They are diagnostics, not a new test set.
    public_ids = ("screenspot_v2:000213:desktop:213", "screenspot_v2:000212:desktop:212")
    cases = [c for c in read_jsonl(root / "runs/public_v1/screenspot_v2/selected_manifest.jsonl")
             if c["sample_id"] in public_ids]
    cases += [c for c in read_jsonl(root / "runs/candidate_v2/manifest.jsonl")
              if c["sample_id"] == "cv1:test:click:screenspot_pro:001217:unreal_engine_windows_12"]
    assert len(cases) == 3

    class AuditRecorder(RawSelectedLayerRecorder):
        def __init__(self, model, layers):
            super().__init__(model, layers)
            self.parity = []
            self.traces = []

        def _attention_hook(self, layer):
            base = super()._attention_hook(layer)
            def hook(module, args, kwargs, output):
                base(module, args, kwargs, output)
                hidden = kwargs.get("hidden_states", args[0] if args else None)
                position = kwargs["position_embeddings"]
                cached = kwargs["past_key_values"].layers[module.layer_idx].keys
                mask = kwargs.get("attention_mask")
                got = reconstruct_attention_row(module, hidden, position, cached, mask)
                b, length, _ = hidden.shape
                q = module.q_norm(module.q_proj(hidden).view(b, length, -1, module.head_dim)).transpose(1, 2)
                q, _ = apply_rotary_pos_emb(q, torch.zeros_like(q), *position)
                q = q[:, :, -1:]
                # Official eager kernel on the same real query/cache. FP32
                # reference separates extraction errors from FP16 matmul error.
                row_mask = None
                if mask is not None:
                    row_mask = mask[:, :, -1:, :cached.shape[-2]] if mask.ndim == 4 else mask[:, None, None, :].bool()
                    if row_mask.dtype == torch.bool:
                        row_mask = torch.zeros_like(row_mask, dtype=torch.float32).masked_fill(~row_mask, -torch.inf)
                _, expected32 = eager_attention_forward(module, q.float(), cached.float(),
                    torch.zeros_like(cached, dtype=torch.float32), row_mask, scaling=module.scaling)
                _, expected16 = eager_attention_forward(module, q, cached, torch.zeros_like(cached),
                    row_mask, scaling=module.scaling)
                ref = expected32[:, :, 0]
                err = (got - ref).abs()
                self.parity.append({"layer": layer, "query_length": length,
                    "max_abs_error_float32": float(err.max()),
                    "mean_head_l1_error_float32": float(err.sum(-1).mean()),
                    "argmax_agreement_float32": float((got.argmax(-1) == ref.argmax(-1)).float().mean()),
                    "max_abs_error_native_fp16_eager": float((got - expected16[:, :, 0].float()).abs().max())})
            return hook

        def _hidden_hook(self, layer):
            base = super()._hidden_hook(layer)
            def hook(module, args, output):
                base(module, args, output)
                if layer == self.layers[-1]:
                    self.traces.append(self.snapshot()["raw_image_attention"].copy())
            return hook

    rows = []
    for number, case in enumerate(cases, 1):
        with Image.open(case["image"]) as opened:
            image = opened.convert("RGB")
        instruction = case.get("base_instruction", case["instruction"])
        # No annotations are accepted by either inference call.
        plain = runtime.infer_text_action_public(image, instruction, max_new_tokens=32)
        inputs, positions, grid = runtime._prepare_image(image, instruction, PUBLIC_TEXT_ACTION_SYSTEM_PROMPT)
        with torch.inference_mode():
            reference = runtime.model(**inputs, use_cache=True, logits_to_keep=1, return_dict=True).logits.detach().float().cpu()
            with RawSelectedLayerRecorder(runtime.model, runtime.config.layers) as recorder:
                recorder.begin(positions, grid)
                observed = runtime.model(**inputs, use_cache=True, logits_to_keep=1, return_dict=True).logits.detach().float().cpu()
        prefill_error = float((reference - observed).abs().max())
        with AuditRecorder(runtime.model, runtime.config.layers) as recorder:
            recorder.begin(positions, grid)
            hooked = runtime.infer_text_action_public(image, instruction, max_new_tokens=32)
        traces = []
        for i, raw in enumerate(recorder.traces):
            point = grounder.predict_point(raw, grid, image.size)["point"]
            xs = centers(grid, image.size)
            traces.append({"stage": "prefill" if i == 0 else "after_generated_token",
                "forwarded_generated_tokens": i,
                "generated_prefix": runtime.processor.tokenizer.decode(hooked["generated_token_ids"][:i]),
                "legacy_point": point, "legacy_point_hit": hit(point, case),
                "any_head_center_hit_oracle": any(hit(xs[j], case) for j in raw.argmax(-1)),
                "mean_image_attention_mass": float(raw.sum(-1).mean())})
        proposal = runtime.propose(image, instruction, grounder)
        row = {"sample_id": case["sample_id"], "instruction": instruction,
            "plain_text": plain["generated_text"], "hooked_text": hooked["generated_text"],
            "tokens_equal": plain["generated_token_ids"] == hooked["generated_token_ids"],
            "prefill_hook_logit_max_abs_error": prefill_error,
            "plain_lax_point": parse_point(plain["generated_text"], image.size),
            "proposal_native_point": proposal["native_proposal_point"],
            "proposal_text_equal": proposal["native_text"] == plain["generated_text"],
            "proposal_point_equals_plain_lax": proposal["native_proposal_point"] == parse_point(plain["generated_text"], image.size),
            "native_hit": hit(parse_point(plain["generated_text"], image.size), case),
            "grid_shape": list(grid), "image_size": list(image.size),
            "last_prompt_token": runtime.processor.tokenizer.decode([int(inputs["input_ids"][0, -1])]),
            "attention_parity": recorder.parity, "readout_traces": traces}
        rows.append(row)
        save(destination / "live_checks.partial.json", {"processor_layout": layout, "cases": rows})
        print(f"[{number}/{len(cases)}] {case['sample_id']}: text_equal={row['tokens_equal']} "
              f"native_hit={row['native_hit']} hook_logit_error={prefill_error}", flush=True)
    return {"processor_layout": layout, "cases": rows,
            "note": "Three fixed diagnostic controls; no fitting or aggregate generalization claim. Official eager checks use actual real-model Q/K with matching mask, without changing model backend."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("runs/localization_extraction_audit_v1"))
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    public = cached_public(args.root)
    candidate = cached_candidates(args.root)
    save(args.output / "cached_public.json", public)
    save(args.output / "cached_candidates.json", candidate)
    print(json.dumps({"public_totals": public["totals"], "candidate_counts": candidate["counts"]}, ensure_ascii=False), flush=True)
    if args.live:
        save(args.output / "live_checks.json", live(args.root, args.output))


if __name__ == "__main__":
    main()
