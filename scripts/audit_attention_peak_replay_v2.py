"""Independent label-free replay of all frozen attention-peak V2 predictions.

This audit never imports the prediction runner and never reads benchmark labels.
It recomputes each global mean-attention peak and checks the complete SHA chain.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "runs/paper_scale_ablation_v2/eval"
RUN = ROOT / "runs/attention_peak_same_trace_v2"
N = {"pro": 500, "sheets": 500, "full": 1581}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def check(condition, message):
    if not condition:
        raise ValueError(message)


def replay(attention, meta, image_size):
    gh, gw = meta["grid_shape"]
    a = np.asarray(attention, np.float32)
    check(a.ndim == 3 and a.shape[1:] == (96, gh * gw), "Attention shape/grid invalid")
    check(np.isfinite(a).all() and (a >= 0).all(), "Attention is invalid")
    check(meta["head_order"] == [f"layer_{layer}.head_{head}"
                                  for layer in (15, 28, 36)
                                  for head in range(32)], "Head order changed")
    if len(a) == 0 or not np.any(a > 0):
        return None, None, len(a)
    masses = a.sum(axis=2, keepdims=True)
    mean_map = np.divide(a, np.maximum(masses, 1e-12)).mean(axis=(0, 1))
    flat = int(np.argmax(mean_map))
    row, col = divmod(flat, gw)
    width, height = image_size
    x = min(width - 1, max(0, round((col + .5) * width / gw)))
    y = min(height - 1, max(0, round((row + .5) * height / gh)))
    return [x, y], flat, len(a)


def main():
    protocol_path = RUN / "protocol.json"
    protocol = load(protocol_path)
    check(protocol["complete"] is True and protocol["experiment"] ==
          "attention_peak_same_trace_v2", "Wrong frozen experiment")
    check(protocol["runner_sha256"] == digest(
          ROOT / "scripts/run_attention_peak_same_trace_v2.py"),
          "Frozen runner source changed")
    checked = {}
    for cohort, expected_n in N.items():
        folder = V2 / cohort
        public_path = folder / "public.json"
        index_path = folder / "features/index.json"
        bundle_path = RUN / "predictions" / (cohort + ".json")
        check(all(path.exists() for path in (public_path, index_path, bundle_path)),
              "Missing public inputs or completed predictions: " + cohort)
        source_hashes = protocol["source_file_sha256"]
        check(source_hashes[cohort + "_public"] == digest(public_path) and
              source_hashes[cohort + "_feature_index"] == digest(index_path),
              "Frozen public/index source SHA drift: " + cohort)
        public, index, bundle = load(public_path), load(index_path), load(bundle_path)
        check(index["complete"] is True and index["labels_present"] is False and
              index["ground_truth_in_model_input"] is False and
              bundle["complete"] is True and bundle["labels_opened"] is False and
              bundle["protocol_sha256"] == digest(protocol_path) and
              len(public) == len(index["cases"]) == len(bundle["rows"]) == expected_n,
              "Bundle/index/public count or label guard drift: " + cohort)
        for i, (row, entry, prediction) in enumerate(
                zip(public, index["cases"], bundle["rows"]), 1):
            sid = row["sample_id"]
            check(sid == entry["sample_id"] == prediction["sample_id"] and
                  entry["labels_present"] is False and
                  entry["ground_truth_in_model_input"] is False,
                  "Public ID or label guard drift: " + sid)
            case_path = RUN / "predictions" / cohort / "cases" / (
                hashlib.sha256(sid.encode("utf-8")).hexdigest()[:24] + ".json")
            check(case_path.exists() and digest(case_path) == bundle["case_hashes"][sid],
                  "Sealed case SHA mismatch: " + sid)
            case = load(case_path)
            check(case["complete"] is True and case["labels_opened"] is False and
                  case["sample_id"] == sid and case["cohort"] == cohort and
                  case["protocol_sha256"] == digest(protocol_path) and
                  case["case_identity"] == entry["case_identity"] and
                  case["pipeline_sha256"] == entry["pipeline_sha256"],
                  "Case metadata mismatch: " + sid)
            pipeline_path = Path(entry["pipeline_path"])
            check(digest(pipeline_path) == entry["pipeline_sha256"],
                  "Pipeline SHA changed: " + sid)
            pipeline = load(pipeline_path)
            check(pipeline["complete"] is True and pipeline["sample_id"] == sid and
                  pipeline["input_identity"] == entry["case_identity"] and
                  pipeline["ground_truth_in_model_inputs"] is False,
                  "Pipeline public/GT identity changed: " + sid)
            raw_path = Path(pipeline["global_feature_path"])
            check(digest(raw_path) == pipeline["global_feature_sha256"] ==
                  case["global_raw_sha256"], "Raw attention SHA changed: " + sid)
            with np.load(raw_path, allow_pickle=False) as z:
                meta = json.loads(str(z["metadata"]))
                attention = z["target_attention"]
            check(meta["sample_id"] == sid and
                  meta["input_identity"] == raw_path.stem and
                  meta["own_token_forward"] is True and
                  meta["ground_truth_in_model_input"] is False and
                  meta["image_size"] == row["image_size"],
                  "Raw metadata identity changed: " + sid)
            point, flat, tokens = replay(attention, meta, row["image_size"])
            check(case["point"] == prediction["point"] == point and
                  case["patch_index"] == prediction["patch_index"] == flat and
                  case["target_name_token_count"] ==
                  prediction["target_name_token_count"] == tokens and
                  case["grid_shape"] == meta["grid_shape"],
                  "Independent peak replay mismatch: " + sid)
            if i % 100 == 0 or i == expected_n:
                print(json.dumps({"cohort": cohort, "replayed": i,
                                  "total": expected_n}), flush=True)
        checked[cohort] = dict(n=expected_n, prediction_sha256=digest(bundle_path),
                               public_sha256=digest(public_path),
                               feature_index_sha256=digest(index_path))
    receipt = dict(complete=True, labels_opened=False,
                   independent_replay="all 2,581 points, peak indices, token counts and raw SHA links",
                   protocol_sha256=digest(protocol_path),
                   audit_script_sha256=digest(__file__), cohorts=checked,
                   checked_n=sum(N.values()), mismatches=0,
                   at_utc=datetime.now(timezone.utc).isoformat())
    receipt_path = RUN / "replay_receipt.json"
    check(not receipt_path.exists(), "Replay receipt already exists")
    temp = receipt_path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
    temp.replace(receipt_path)
    print(json.dumps({"replay_complete": True, "checked_n": receipt["checked_n"],
                      "mismatches": 0, "labels_opened": False}), flush=True)


if __name__ == "__main__":
    main()
