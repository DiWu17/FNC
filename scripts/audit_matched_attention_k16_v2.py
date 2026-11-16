"""Independent public-only replay of all frozen matched K16 attention picks.

This script does not import the experiment runner or open benchmark labels.
It recomputes each scalar directly from V2 raw attention and view geometry,
and compares it against the SHA-sealed per-case/bundle predictions.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/matched_attention_k16_v2"
SOURCE = ROOT / "runs/paper_scale_ablation_v2/eval"
HEAD_ORDER = [f"layer_{layer}.head_{head}" for layer in (15, 28, 36)
              for head in range(32)]


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def replay():
    protocol = read(RUN / "protocol.json")
    receipt = read(RUN / "public_completion.json")
    require(protocol["complete"] is True and
            receipt["complete"] is True and receipt["labels_opened"] is False and
            receipt["protocol_sha256"] == sha(RUN / "protocol.json") and
            receipt["cohorts"] == {"pro": 500, "sheets": 500},
            "Frozen public receipt mismatch")
    counts = {}
    all_raw = set()
    maximum_score_error = 0.0
    for cohort in ("pro", "sheets"):
        folder = SOURCE / cohort
        rows = read(folder / "public.json")
        entries = read(folder / "features/index.json")["cases"]
        bundle_path = RUN / "predictions" / (cohort + ".json")
        bundle = read(bundle_path)
        require(len(rows) == len(entries) == len(bundle["rows"]) == 500 and
                receipt["prediction_sha256"][cohort] == sha(bundle_path) and
                bundle["protocol_sha256"] == sha(RUN / "protocol.json") and
                bundle["labels_opened"] is False,
                "Frozen cohort bundle mismatch: " + cohort)
        null_count = 0
        for row, entry, output in zip(rows, entries, bundle["rows"]):
            sid = row["sample_id"]
            require(entry["sample_id"] == sid and output["sample_id"] == sid and
                    entry["labels_present"] is False and
                    entry["ground_truth_in_model_input"] is False,
                    "Public row identity mismatch")
            case_path = RUN / "predictions" / cohort / "cases" / (
                hashlib.sha256(sid.encode()).hexdigest()[:24] + ".json")
            require(bundle["case_hashes"][sid] == sha(case_path),
                    "Sealed per-case SHA differs: " + sid)
            saved = read(case_path)
            require(saved["sample_id"] == sid and saved["labels_opened"] is False and
                    saved["protocol_sha256"] == sha(RUN / "protocol.json") and
                    saved["feature_sha256"] == entry["sha256"] and
                    saved["pipeline_sha256"] == entry["pipeline_sha256"] and
                    sha(entry["path"]) == entry["sha256"] and
                    sha(entry["pipeline_path"]) == entry["pipeline_sha256"],
                    "Source/experiment identity mismatch: " + sid)
            pipe = read(entry["pipeline_path"])
            require(pipe["complete"] is True and pipe["sample_id"] == sid and
                    pipe["ground_truth_in_model_inputs"] is False and
                    len(pipe["local_views"]) <= 3,
                    "Pipeline identity or view count mismatch: " + sid)
            with np.load(entry["path"], allow_pickle=False) as z:
                all_points = z["points"].astype(int)
                original_pool = z["baseline"].astype(int)
                k16 = np.sort(original_pool)
            require(len(k16) <= 16 and
                    len(k16) == saved["K16_count"] and
                    all_points.shape == (entry["n"], 2) and
                    k16.tolist() == saved["K16_indices"] and
                    all_points[original_pool].tolist() ==
                    [item["point"] for item in pipe["pools"]["c3"]["16"]],
                    "Original K16 pool drift: " + sid)
            points = all_points[k16]
            width, height = map(int, row["image_size"])
            views = [dict(feature_path=pipe["global_feature_path"],
                          feature_sha256=pipe["global_feature_sha256"],
                          transform=None)] + pipe["local_views"]
            require(len(views) == len(saved["raw_view_sha256"]) ==
                    len(saved["token_counts"]), "View count drift: " + sid)
            scores = [[] for _ in range(len(k16))]
            for j, view in enumerate(views):
                raw_path = Path(view["feature_path"])
                digest = sha(raw_path)
                require(digest == view["feature_sha256"] ==
                        saved["raw_view_sha256"][j],
                        "Raw attention file SHA drift: " + sid)
                all_raw.add(str(raw_path))
                with np.load(raw_path, allow_pickle=False) as z:
                    meta = json.loads(str(z["metadata"]))
                    a = z["target_attention"].astype(np.float32)
                gh, gw = map(int, meta["grid_shape"])
                require(meta["head_order"] == HEAD_ORDER and
                        meta["sample_id"] == sid and
                        meta["transform"] == view["transform"] and
                        meta["own_token_forward"] is True and
                        meta["ground_truth_in_model_input"] is False and
                        a.shape == (len(a), 96, gh * gw) and
                        len(a) == saved["token_counts"][j] and
                        np.isfinite(a).all() and np.all(a >= 0),
                        "Raw attention metadata/tensor drift: " + sid)
                if len(a) == 0 or not np.any(a > 0):
                    continue
                image_mass = a.sum(axis=2, keepdims=True)
                normalized = a / np.maximum(image_mass, 1e-12)
                attention_map = normalized.mean(axis=(0, 1), dtype=np.float64) * (gh * gw)
                box = ([0, 0, width, height] if view["transform"] is None
                       else view["transform"]["box"])
                left, top, right, bottom = box
                for i, (x, y) in enumerate(points):
                    if not (left <= x < right and top <= y < bottom):
                        continue
                    col = min(gw - 1, max(0, math.floor((x - left) / (right - left) * gw)))
                    row_ix = min(gh - 1, max(0, math.floor((y - top) / (bottom - top) * gh)))
                    scores[i].append(float(attention_map[row_ix * gw + col]))
            computed = [sum(vals) / len(vals) if vals else None for vals in scores]
            require([len(vals) for vals in scores] == saved["valid_view_counts"],
                    "Candidate valid-view counts differ: " + sid)
            for a, b in zip(computed, saved["attention_scores"]):
                require((a is None and b is None) or
                        (a is not None and b is not None and
                         math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6)),
                        "Candidate raw attention score differs: " + sid)
                if a is not None:
                    maximum_score_error = max(maximum_score_error, abs(a - b))
            eligible = [(k16[i], score) for i, score in enumerate(computed)
                        if score is not None]
            if eligible:
                winner = sorted(eligible, key=lambda item: (-item[1], item[0]))[0][0]
                point = all_points[winner].tolist()
            else:
                winner, point = None, None
                null_count += 1
            require(saved["index"] == winner and saved["point"] == point and
                    output["index"] == winner and output["point"] == point and
                    output["valid_view_counts"] == saved["valid_view_counts"] and
                    output["token_counts"] == saved["token_counts"],
                    "Sealed attention winner differs on replay: " + sid)
        counts[cohort] = dict(n=500, null=null_count)
        print(f"{cohort}: 500/500 label-free replay and SHA parity passed", flush=True)
    result = dict(complete=True, labels_opened=False, n=1000,
                  cohorts=counts, source_raw_files=len(all_raw),
                  max_candidate_score_abs_error=maximum_score_error,
                  protocol_sha256=sha(RUN / "protocol.json"),
                  public_completion_sha256=sha(RUN / "public_completion.json"),
                  prediction_sha256=receipt["prediction_sha256"],
                  auditor_sha256=sha(__file__),
                  note="Read-only independent replay before registered analysis or any result report; scientific protocol and predictions unchanged.")
    path = RUN / "replay_audit.json"
    data = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if path.exists():
        require(path.read_text(encoding="utf-8") == data,
                "Existing replay audit differs")
    else:
        path.write_text(data, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    replay()
