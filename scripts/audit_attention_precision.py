"""Compare real-model Q captured in forward with the optimized row readout.

This diagnostic addresses FP16 full-sequence versus last-row projection kernels.
No saved model, original prediction, readout, or threshold is changed.
"""
import argparse
from pathlib import Path

import numpy as np

from audit_localization_extraction import read_json, read_jsonl, save, hit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/localization_extraction_audit_v1"))
    args = parser.parse_args()
    import torch
    from PIL import Image
    from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb, eager_attention_forward
    from clickagent.action_grounding import AttentionPointGrounder
    from clickagent.action_model import RawSelectedLayerRecorder
    from clickagent.candidate_model import CandidateRuntime
    from clickagent.model_features import ExtractionConfig, reconstruct_attention_row
    from clickagent.public_model import PUBLIC_TEXT_ACTION_SYSTEM_PROMPT, PUBLIC_DECISION_SYSTEM_PROMPT

    cfg = read_json("runs/candidate_v2/identity.json")["config"]
    cfg["layers"] = tuple(cfg["layers"])
    cfg["conditions"] = tuple(cfg["conditions"])
    cfg["max_memory"] = {int(k) if k.isdigit() else k: v for k, v in cfg["max_memory"].items()}
    cfg["offload_folder"] = str((args.output / "offload").resolve())
    print("Loading frozen Qwen for exact-query and hidden-state audit", flush=True)
    runtime = CandidateRuntime(ExtractionConfig(**cfg))
    grounder = AttentionPointGrounder.load("runs/action_v3/grounder/grounder.joblib")
    ids = {"screenspot_v2:000213:desktop:213", "screenspot_v2:000212:desktop:212"}
    cases = [c for c in read_jsonl("runs/public_v1/screenspot_v2/selected_manifest.jsonl") if c["sample_id"] in ids]
    cases += [c for c in read_jsonl("runs/candidate_v2/manifest.jsonl")
              if c["sample_id"] == "cv1:test:click:screenspot_pro:001217:unreal_engine_windows_12"]
    rows = []
    for case in cases:
        with Image.open(case["image"]) as opened:
            image = opened.convert("RGB")
        instruction = case.get("base_instruction", case["instruction"])
        for prompt_name, prompt in (("coordinate", PUBLIC_TEXT_ACTION_SYSTEM_PROMPT), ("decision", PUBLIC_DECISION_SYSTEM_PROMPT)):
            inputs, positions, grid = runtime._prepare_image(image, instruction, prompt)
            official_queries, official_rows, errors = {}, {}, []
            handles = []

            def capture_query(layer):
                def hook(module, inputs, output):
                    official_queries[layer] = output[:, -1:].detach().clone()
                return hook

            def compare_attention(layer):
                def hook(module, args, kwargs, output):
                    hidden = kwargs["hidden_states"]
                    position = kwargs["position_embeddings"]
                    cached = kwargs["past_key_values"].layers[module.layer_idx].keys
                    mask = kwargs.get("attention_mask")
                    approximate = reconstruct_attention_row(module, hidden, position, cached, mask)
                    query = official_queries[layer].transpose(1, 2)
                    query, _ = apply_rotary_pos_emb(query, torch.zeros_like(query),
                        position[0][:, -1:], position[1][:, -1:])
                    row_mask = None
                    if mask is not None:
                        row_mask = mask[:, :, -1:, :cached.shape[-2]] if mask.ndim == 4 else mask[:, None, None, :].bool()
                        if row_mask.dtype == torch.bool:
                            row_mask = torch.zeros_like(row_mask, dtype=torch.float32).masked_fill(~row_mask, -torch.inf)
                    _, expected = eager_attention_forward(module, query.float(), cached.float(),
                        torch.zeros_like(cached, dtype=torch.float32), row_mask, scaling=module.scaling)
                    expected = expected[:, :, 0]
                    idx = torch.as_tensor(positions, dtype=torch.long, device=expected.device)
                    raw_expected = expected[0].index_select(-1, idx)
                    raw_approx = approximate[0].index_select(-1, idx)
                    errors.append({"layer": layer, "all_keys_max_abs_error": float((approximate - expected).abs().max()),
                        "image_keys_max_abs_error": float((raw_expected - raw_approx).abs().max()),
                        "image_peak_agreement": float((raw_expected.argmax(-1) == raw_approx.argmax(-1)).float().mean())})
                    official_rows[layer] = raw_expected.detach().cpu().numpy().copy()
                return hook

            try:
                for layer in runtime.config.layers:
                    attention = runtime.model.model.language_model.layers[layer - 1].self_attn
                    handles.append(attention.q_norm.register_forward_hook(capture_query(layer)))
                    handles.append(attention.register_forward_hook(compare_attention(layer), with_kwargs=True))
                with torch.inference_mode(), RawSelectedLayerRecorder(runtime.model, runtime.config.layers) as recorder:
                    recorder.begin(positions, grid)
                    outputs = runtime.model(**inputs, use_cache=True, logits_to_keep=1, return_dict=True)
                    logits = outputs.logits.detach().float().cpu()
                    approximate = recorder.snapshot()["raw_image_attention"]
                    # Hooks capture pre-final-RMSNorm decoder output. Applying
                    # the pretrained norm and LM head must recover native logits.
                    norm = runtime.model.model.language_model.norm
                    head = runtime.model.lm_head
                    h = torch.from_numpy(recorder.hidden[runtime.config.layers[-1]]).to(
                        device=norm.weight.device, dtype=norm.weight.dtype)[None, None]
                    recovered = head(norm(h).to(head.weight.device)).detach().float().cpu()
                    hidden_error = float((logits - recovered).abs().max())
                    del outputs
            finally:
                for handle in handles:
                    handle.remove()
            reference = np.concatenate([official_rows[layer] for layer in runtime.config.layers])
            point_a = grounder.predict_point(approximate, grid, image.size)
            point_b = grounder.predict_point(reference, grid, image.size)
            rows.append({"sample_id": case["sample_id"], "prompt": prompt_name,
                "attention_errors": errors, "optimized_readout_point": point_a["point"],
                "official_query_readout_point": point_b["point"],
                "points_equal": point_a["point"] == point_b["point"],
                "optimized_point_hit": hit(point_a["point"], case), "official_point_hit": hit(point_b["point"], case),
                "last_hidden_pretrained_norm_head_logit_max_abs_error": hidden_error})
            save(args.output / "precision_checks.partial.json", {"cases": rows})
            print(f"[{len(rows)}/6] {case['sample_id']} {prompt_name}: "
                  f"points_equal={rows[-1]['points_equal']} hidden_logit_error={hidden_error}", flush=True)
    save(args.output / "precision_checks.json", {"cases": rows,
        "note": "Official Q is captured during the actual forward, not reprojected. Scores use FP32 accumulation as in the production reconstruction; native SDPA can have its own rounding. Six prefills on three fixed controls do not establish an aggregate accuracy effect."})


if __name__ == "__main__":
    main()
