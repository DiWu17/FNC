"""Retrospective, label-gated K16 raw-attention control on frozen V2 views.

The public stages read V2 public inputs, K16 points and original Qwen raw
attention on the global plus every frozen local view. No fitting or new model
forward occurs. PointerBench labels are opened only after all 1,000 predictions
and their SHA receipt have passed the public gate.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import analyze_paper_scale_v2 as v2
import run_attention_peak_same_trace_v2 as old_peak

V2 = ROOT / "runs/paper_scale_ablation_v2"
EVAL = V2 / "eval"
RUN = ROOT / "runs/matched_attention_k16_v1"
PROTOCOL = RUN / "protocol.json"
COMPLETION = RUN / "public_completion.json"
COHORTS = ("pro", "sheets")
SIZES = {"pro": 500, "sheets": 500}
BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_SEED = 20261113
HEAD_ORDER = [f"layer_{layer}.head_{head}" for layer in (15, 28, 36)
              for head in range(32)]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def case_key(sample_id):
    return hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:24]


def write_once(path, value):
    path = Path(path)
    data = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if path.exists():
        require(path.read_text(encoding="utf-8") == data,
                "Existing frozen output differs: " + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    require(not temporary.exists(), "Unresolved temporary output: " + str(temporary))
    temporary.write_text(data, encoding="utf-8")
    temporary.replace(path)


def status(stage, cohort=None, completed=None, total=None):
    row = dict(stage=stage, cohort=cohort, completed=completed, total=total,
               at_utc=datetime.now(timezone.utc).isoformat())
    RUN.mkdir(parents=True, exist_ok=True)
    path = RUN / "status.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    temporary.replace(path)
    print(json.dumps(row, ensure_ascii=False), flush=True)


def source_files():
    result = {
        "v2_analysis_script": ROOT / "scripts/analyze_paper_scale_v2.py",
        "v2_statistical_registration": V2 / "statistical_analysis_protocol.json",
        "v2_full_completion": V2 / "public_completion.json",
        "v2_pointer_completion": V2 / "pointerbench/public_completion.json",
        "v2_pointer_protocol": V2 / "pointerbench/evaluation_protocol_v2.json",
        "old_peak_script": ROOT / "scripts/run_attention_peak_same_trace_v2.py",
        "pointer_pro_official_scorer": V2 / "pointerbench/source/pro/eval.py",
        "pointer_sheets_official_scorer": V2 / "pointerbench/source/sheets/eval.py",
    }
    for cohort in COHORTS:
        folder = EVAL / cohort
        result[cohort + "_public"] = folder / "public.json"
        result[cohort + "_feature_index"] = folder / "features/index.json"
        result[cohort + "_v2_predictions"] = folder / "predictions.json"
    return result


def expected_protocol():
    reg = read(V2 / "statistical_analysis_protocol.json")
    return dict(
        complete=True, experiment="matched_attention_k16_v1",
        purpose="Untrained raw-attention ranking of V2 K16 using identical global/local views",
        retrospective=True, strict_blind=False,
        primary_cohort="PointerBench Pro500", secondary_cohort="PointerBench Sheets500",
        cohorts=SIZES, fixed_denominators=True,
        original_model="Frozen V2 Qwen3-VL-8B original target-name trajectories",
        candidates="Exact V2 original K16 points and candidate order; no new proposals",
        views="Exact V2 global screenshot and all up to three frozen local crops",
        attention_layers=[15, 28, 36], heads_per_layer=32,
        target_tokens="Every generated target-name own-token forward position; no token/head selection",
        rule=("For each view separately and each token/head, divide raw patch attention by "
              "its total image-patch mass with denominator at least 1e-12. Arithmetic-mean "
              "these 96-head/all-token maps, then multiply by patch count so a uniform map "
              "scores 1 across differing grid sizes. For every K16 candidate whose original "
              "pixel coordinate lies inside the view box [left,top,right,bottom), sample the "
              "floor-mapped grid cell (clipped only for numerical edge safety). The global view "
              "uses [0,0,width,height). Arithmetic-mean the sampled scalar over every valid "
              "view, including all overlapping local views. A candidate outside a local crop "
              "uses only its other valid views; one with no local coverage uses global only. "
              "A view with zero tokens or all-zero image attention is skipped. If all views "
              "provide no scalar, output null and count wrong. Rank only original K16 by "
              "strict floating-point greater-than, no tolerance; exact ties use ascending "
              "original candidate-universe index, matching V2 stable K16 order. No learned "
              "weights, label-tuned rule, or alternative aggregation."),
        public_prediction_gate="Both 500-case bundles and all per-case SHA hashes before labels",
        metric="Official PointerBench inclusive point-in-bbox; verify V2 scorer parity",
        primary_contrast="Pro500 V2 full aligned 3-seed mean minus matched raw attention",
        descriptive_contrasts=["Pro500 V2 trained full attention_only 3-seed mean minus raw attention",
                               "Pro500 V2 3k aligned 3-seed mean minus raw attention",
                               "Sheets500 same contrasts"],
        bootstrap_unit="RGB image pixel_sha256 cluster",
        bootstrap_replicates=BOOTSTRAP_REPLICATES,
        bootstrap_seed=BOOTSTRAP_SEED,
        bootstrap_interval="descriptive two-sided percentile 95%",
        previous_v2_labels_and_aggregate_results_seen=True,
        previous_pointer_answers_visible_at_least=7,
        cannot_establish_nonattention_feature_benefit=True,
        comparison_to_trained_attention_only_is_descriptive=True,
        runner_sha256=sha(__file__),
        source_file_sha256={name: sha(path) for name, path in source_files().items()},
        pointer_labels_sha256_unopened=reg["pointer_labels_sha256_unopened"],
    )


def registered():
    require(PROTOCOL.exists(), "Run --stage register before prediction")
    protocol = read(PROTOCOL)
    require(protocol == expected_protocol(), "Registered rule or source drift")
    return protocol


def register():
    write_once(PROTOCOL, expected_protocol())
    status("registered", completed=0, total=1000)


def public_index(cohort):
    registered()
    require(cohort in COHORTS, "Unknown cohort")
    folder = EVAL / cohort
    rows = read(folder / "public.json")
    index = read(folder / "features/index.json")
    old_bundle = read(folder / "predictions.json")
    require(index["complete"] is True and index["cohort"] == cohort and
            index["n"] == len(rows) == SIZES[cohort] and
            index["labels_present"] is False and
            index["ground_truth_in_model_input"] is False and
            index["public_sha256"] == sha(folder / "public.json") and
            [e["sample_id"] for e in index["cases"]] ==
            [r["sample_id"] for r in rows] and
            old_bundle["complete"] is True and old_bundle["labels_opened"] is False and
            old_bundle["n"] == SIZES[cohort] and
            old_bundle["public_sha256"] == index["public_sha256"],
            "V2 public feature/prediction identity mismatch: " + cohort)
    return rows, index["cases"], old_bundle


def view_score(attention, metadata, points, box):
    """Return [K] patch attention and validity using original-pixel geometry."""
    a = np.asarray(attention, np.float32)
    gh, gw = [int(n) for n in metadata["grid_shape"]]
    require(a.ndim == 3 and a.shape[1:] == (96, gh * gw) and gh > 0 and gw > 0,
            "Unexpected attention/grid shape")
    require(metadata["head_order"] == HEAD_ORDER and
            np.isfinite(a).all() and (a >= 0).all(),
            "Raw attention/head order drift")
    pts = np.asarray(points, np.float64)
    bounds = np.asarray(box, np.float64)
    require(pts.ndim == 2 and pts.shape[1] == 2 and
            bounds.shape == (4,) and np.all(bounds[2:] > bounds[:2]),
            "Invalid candidate/view geometry")
    xy = (pts - bounds[:2]) / (bounds[2:] - bounds[:2])
    valid = np.all((xy >= 0) & (xy < 1), axis=1)
    if len(a) == 0 or not bool(np.any(a > 0)):
        return np.zeros(len(pts), np.float64), np.zeros(len(pts), bool), len(a)
    cells = np.floor(xy * np.asarray([gw, gh])).astype(np.int64)
    cells = np.clip(cells, [0, 0], [gw - 1, gh - 1])
    index = cells[:, 1] * gw + cells[:, 0]
    normalized = a / np.maximum(a.sum(-1, keepdims=True), 1e-12)
    map_ = normalized.mean((0, 1), dtype=np.float64) * (gh * gw)
    values = map_[index]
    require(np.isfinite(values).all(), "Invalid candidate attention scores")
    return values, valid, len(a)


def rank_attention(view_values, points, baseline):
    """Aggregate every valid view, then preserve V2's stable K16 tie order."""
    points = np.asarray(points)
    pool = np.sort(np.asarray(baseline, np.int64))
    require(points.ndim == 2 and points.shape[1] == 2 and
            len(set(pool.tolist())) == len(pool) and
            ((pool >= 0) & (pool < len(points))).all(), "Invalid frozen K16")
    totals = np.zeros(len(pool), np.float64)
    counts = np.zeros(len(pool), np.int64)
    for values, valid in view_values:
        require(len(values) == len(valid) == len(pool), "View/K16 misalignment")
        totals[valid] += values[valid]
        counts[valid] += 1
    if len(pool) == 0 or not np.any(counts):
        return None, None, counts.tolist(), []
    scores = np.full(len(pool), -math.inf, np.float64)
    present = counts > 0
    scores[present] = totals[present] / counts[present]
    # Sorted pool order + strict greater-than gives the same tie convention as
    # V2's stable argsort over candidate-universe indices.
    best = None
    for i, score in enumerate(scores):
        if best is None or score > scores[best]:
            best = i
    require(best is not None and math.isfinite(float(scores[best])),
            "No finite attention winner")
    return int(pool[best]), points[pool[best]].astype(int).tolist(), counts.tolist(), [
        None if not math.isfinite(float(s)) else float(s) for s in scores]


def predict_case(cohort, row, entry, old_bundle):
    sid = row["sample_id"]
    require(entry["labels_present"] is False and
            entry["ground_truth_in_model_input"] is False and
            entry["case_identity"] == v2.identity(row), "Public feature identity drift")
    for key, digest in (("path", "sha256"), ("pipeline_path", "pipeline_sha256")):
        require(sha(entry[key]) == entry[digest], "Frozen feature SHA changed: " + sid)
    pipe = read(entry["pipeline_path"])
    require(pipe["complete"] is True and pipe["sample_id"] == sid and
            pipe["input_identity"] == entry["case_identity"] and
            pipe["ground_truth_in_model_inputs"] is False and
            len(pipe["local_views"]) <= 3, "Frozen pipeline identity/GT drift")
    with np.load(entry["path"], allow_pickle=False) as z:
        points, pool = z["points"].copy(), z["baseline"].copy()
    require(points.shape == (entry["n"], 2) and
            len(pool) == len(pipe["pools"]["c3"]["16"]) and
            points[pool].tolist() ==
            [c["point"] for c in pipe["pools"]["c3"]["16"]],
            "Original K16 points differ between feature and pipeline")
    old_case_path = EVAL / cohort / "predictions/cases" / (case_key(sid) + ".json")
    require(sha(old_case_path) == old_bundle["case_hashes"][sid],
            "V2 per-case prediction SHA drift")
    old_case = read(old_case_path)
    require(old_case["complete"] is True and old_case["labels_opened"] is False and
            old_case["candidate_count"] == len(points) and
            old_case["K16_count"] == len(pool) and
            old_case["feature_sha256"] == entry["sha256"],
            "V2 per-case candidate seal drift")
    for name, pred in old_case["methods"].items():
        ix = pred["index"]
        require((ix is None and pred["point"] is None) or
                (ix in pool and pred["point"] == points[ix].tolist()),
                "V2 model prediction outside sealed K16: " + name)
    width, height = map(int, row["image_size"])
    views = [dict(feature_path=pipe["global_feature_path"],
                  feature_sha256=pipe["global_feature_sha256"],
                  transform=None)] + pipe["local_views"]
    view_values, view_hashes, token_counts = [], [], []
    for view in views:
        path, transform = view["feature_path"], view["transform"]
        require(sha(path) == view["feature_sha256"],
                "Frozen raw attention SHA changed: " + sid)
        with np.load(path, allow_pickle=False) as z:
            metadata = json.loads(str(z["metadata"]))
            a = z["target_attention"]
        require(metadata["own_token_forward"] is True and
                metadata["ground_truth_in_model_input"] is False and
                metadata["sample_id"] == sid and
                metadata["input_identity"] == Path(path).stem and
                metadata["transform"] == transform and
                metadata["image_size"] ==
                ([width, height] if transform is None else transform["crop_size"]),
                "Raw attention view/public identity mismatch")
        box = [0, 0, width, height] if transform is None else transform["box"]
        values, valid, tokens = view_score(a, metadata, points[pool], box)
        view_values.append((values, valid))
        view_hashes.append(view["feature_sha256"])
        token_counts.append(tokens)
    pick, point, counts, scores = rank_attention(view_values, points, pool)
    return dict(complete=True, cohort=cohort, sample_id=sid,
                case_identity=entry["case_identity"],
                protocol_sha256=sha(PROTOCOL), labels_opened=False,
                feature_sha256=entry["sha256"],
                pipeline_sha256=entry["pipeline_sha256"],
                v2_case_sha256=old_bundle["case_hashes"][sid],
                raw_view_sha256=view_hashes, token_counts=token_counts,
                candidate_count=len(points), K16_count=len(pool),
                K16_indices=np.sort(pool).astype(int).tolist(),
                valid_view_counts=counts, attention_scores=scores,
                index=pick, point=point)


def case_path(cohort, sid):
    return RUN / "predictions" / cohort / "cases" / (case_key(sid) + ".json")


def verify_case(saved, cohort, row, entry, old_bundle):
    sid = row["sample_id"]
    require(saved.get("complete") is True and
            saved.get("labels_opened") is False and
            saved.get("cohort") == cohort and saved.get("sample_id") == sid and
            saved.get("case_identity") == entry["case_identity"] and
            saved.get("protocol_sha256") == sha(PROTOCOL) and
            saved.get("feature_sha256") == entry["sha256"] and
            saved.get("pipeline_sha256") == entry["pipeline_sha256"] and
            saved.get("v2_case_sha256") == old_bundle["case_hashes"][sid] and
            saved.get("candidate_count") == entry["n"] and
            len(saved.get("K16_indices", [])) == saved.get("K16_count") and
            len(saved.get("attention_scores", [])) == saved.get("K16_count") and
            len(saved.get("valid_view_counts", [])) == saved.get("K16_count") and
            len(saved.get("raw_view_sha256", [])) ==
            len(saved.get("token_counts", [])) and
            1 <= len(saved.get("token_counts", [])) <= 4,
            "Frozen per-case output identity drift: " + sid)
    index, point = saved["index"], saved["point"]
    require((index is None and point is None) or
            (type(index) is int and index in saved["K16_indices"] and
             isinstance(point, list) and len(point) == 2 and
             all(type(x) is int for x in point) and
             0 <= point[0] < row["image_size"][0] and
             0 <= point[1] < row["image_size"][1]),
            "Invalid K16 point: " + sid)


def predict(cohort):
    rows, entries, old_bundle = public_index(cohort)
    status("predicting", cohort, 0, len(rows))
    started = time.perf_counter()
    hashes, outputs = {}, []
    for i, (row, entry) in enumerate(zip(rows, entries), 1):
        path = case_path(cohort, row["sample_id"])
        if path.exists():
            saved = read(path)
            verify_case(saved, cohort, row, entry, old_bundle)
        else:
            saved = predict_case(cohort, row, entry, old_bundle)
            verify_case(saved, cohort, row, entry, old_bundle)
            write_once(path, saved)
        hashes[row["sample_id"]] = sha(path)
        outputs.append(dict(sample_id=row["sample_id"],
                            index=saved["index"], point=saved["point"],
                            valid_view_counts=saved["valid_view_counts"],
                            token_counts=saved["token_counts"]))
        if i % 25 == 0 or i == len(rows):
            status("predicting", cohort, i, len(rows))
    bundle = dict(complete=True, cohort=cohort, n=len(rows),
                  public_sha256=sha(EVAL / cohort / "public.json"),
                  feature_index_sha256=sha(EVAL / cohort / "features/index.json"),
                  v2_prediction_sha256=sha(EVAL / cohort / "predictions.json"),
                  protocol_sha256=sha(PROTOCOL), labels_opened=False,
                  case_hashes=hashes, rows=outputs)
    write_once(RUN / "predictions" / (cohort + ".json"), bundle)
    status("predicted", cohort, len(rows), len(rows))
    print(json.dumps(dict(cohort=cohort,
                          elapsed_seconds=round(time.perf_counter() - started, 2),
                          prediction_sha256=sha(RUN / "predictions" / (cohort + ".json"))),
                     ensure_ascii=False), flush=True)


def verify_bundle(cohort):
    rows, entries, old_bundle = public_index(cohort)
    path = RUN / "predictions" / (cohort + ".json")
    require(path.exists(), "Missing prediction bundle: " + cohort)
    bundle = read(path)
    require(bundle.get("complete") is True and bundle.get("cohort") == cohort and
            bundle.get("n") == SIZES[cohort] and bundle.get("labels_opened") is False and
            bundle.get("protocol_sha256") == sha(PROTOCOL) and
            bundle.get("public_sha256") == sha(EVAL / cohort / "public.json") and
            bundle.get("feature_index_sha256") ==
            sha(EVAL / cohort / "features/index.json") and
            bundle.get("v2_prediction_sha256") ==
            sha(EVAL / cohort / "predictions.json") and
            len(bundle.get("rows", [])) == len(rows) and
            set(bundle.get("case_hashes", {})) == {r["sample_id"] for r in rows},
            "Prediction bundle identity mismatch: " + cohort)
    for row, entry, output in zip(rows, entries, bundle["rows"]):
        sid = row["sample_id"]
        path = case_path(cohort, sid)
        require(path.exists() and sha(path) == bundle["case_hashes"][sid],
                "Per-case prediction SHA changed: " + sid)
        saved = read(path)
        verify_case(saved, cohort, row, entry, old_bundle)
        require(output == {"sample_id": sid, "index": saved["index"],
                           "point": saved["point"],
                           "valid_view_counts": saved["valid_view_counts"],
                           "token_counts": saved["token_counts"]},
                "Bundle/case point mismatch: " + sid)
    return rows, bundle, sha(RUN / "predictions" / (cohort + ".json"))


def seal():
    registered()
    bundles = {cohort: verify_bundle(cohort) for cohort in COHORTS}
    receipt = dict(complete=True, labels_opened=False,
                   protocol_sha256=sha(PROTOCOL), cohorts=SIZES,
                   prediction_sha256={cohort: bundles[cohort][2] for cohort in COHORTS},
                   all_1000_predictions_present=True)
    write_once(COMPLETION, receipt)
    status("sealed", completed=1000, total=1000)


def gate():
    registered()
    require(COMPLETION.exists(), "Both cohorts must be SHA-sealed first")
    receipt = read(COMPLETION)
    require(receipt.get("complete") is True and
            receipt.get("labels_opened") is False and
            receipt.get("protocol_sha256") == sha(PROTOCOL) and
            receipt.get("cohorts") == SIZES and
            receipt.get("all_1000_predictions_present") is True,
            "Public completion receipt mismatch")
    bundles = {cohort: verify_bundle(cohort) for cohort in COHORTS}
    require(receipt["prediction_sha256"] ==
            {cohort: bundles[cohort][2] for cohort in COHORTS},
            "Prediction receipt SHA drift")
    old_protocol, old_bundles, _ = v2.gate()
    for cohort in COHORTS:
        require([r["sample_id"] for r in bundles[cohort][0]] ==
                [r["sample_id"] for r in old_bundles[cohort][0]],
                "V2/new public cohort order drift: " + cohort)
    return bundles, old_protocol, old_bundles


def analyze():
    bundles, old_protocol, old_bundles = gate()
    # Neither label path is opened, hashed, or imported before gate returns.
    require(sha(v2.POINTER_LABELS) ==
            read(PROTOCOL)["pointer_labels_sha256_unopened"],
            "Pointer labels changed after public sealing")
    geometry = v2.open_labels_after_gate(old_protocol, old_bundles)
    pointer_labels = {r["sample_id"]: r for r in v2.jsonl(v2.POINTER_LABELS)}
    from clickagent.public_evaluation import point_in_geometry

    reports, details = {}, {}
    names = [f"{scale}_{variant}_s{s}" for scale in ("3k", "full")
             for variant in ("aligned", "attention_only") for s in (1, 2, 3)]
    for cohort in COHORTS:
        rows, bundle, _ = bundles[cohort]
        old_cases = old_bundles[cohort][1]
        official = old_peak.official_scorer(cohort)
        vectors = {"matched_raw_attention": [], **{name: [] for name in names}}
        per_case = []
        for row, output in zip(rows, bundle["rows"]):
            sid = row["sample_id"]
            width, height = row["image_size"]
            points = {"matched_raw_attention": output["point"]}
            points.update({name: old_cases[sid]["methods"][name]["point"]
                           for name in names})
            hits = {}
            for name, point in points.items():
                hit = bool(point is not None and point_in_geometry(
                    (point[0] / width, point[1] / height), geometry[cohort][sid]))
                native = bool(point is not None and official(
                    tuple(point), pointer_labels[sid]["bbox_pixels"]))
                require(hit == native, "V2/official scorer parity failed: " + sid)
                vectors[name].append(float(native))
                hits[name] = native
            per_case.append(dict(sample_id=sid, pixel_sha256=row["pixel_sha256"],
                                 point=output["point"], hits=hits))
        for scale in ("3k", "full"):
            for variant in ("aligned", "attention_only"):
                vectors[f"{scale}_{variant}_mean"] = np.mean(
                    [vectors[f"{scale}_{variant}_s{s}"] for s in (1, 2, 3)], axis=0)
        accuracy = {name: float(np.mean(values)) for name, values in vectors.items()}
        contrasts = {}
        for name in ("full_aligned_mean", "full_attention_only_mean",
                     "3k_aligned_mean", "3k_attention_only_mean"):
            contrasts[name + "_minus_raw"] = old_peak.paired_ci(
                vectors[name], vectors["matched_raw_attention"], rows,
                BOOTSTRAP_SEED + (0 if cohort == "pro" else 1))
        reports[cohort] = dict(
            n=SIZES[cohort], rgb_clusters=len({r["pixel_sha256"] for r in rows}),
            accuracy=accuracy, counts={name: int(sum(values)) for name, values in
                                      vectors.items() if not name.endswith("_mean")},
            contrasts=contrasts,
            null_point_count=sum(r["point"] is None for r in bundle["rows"]),
            scorer="official PointerBench inclusive bbox with V2 point parity",
            retrospective=True, strict_blind=False)
        details[cohort] = per_case
    summary = dict(
        complete=True, protocol_sha256=sha(PROTOCOL),
        public_completion_sha256=sha(COMPLETION), results=reports,
        note=("Retrospective same-cohort, same-K16, same-global/local-view control "
              "after V2 labels/results were observed. The untrained scalar raw-attention "
              "ranking has the same upstream observation and candidate pools but does not "
              "establish a causal benefit of non-attention internal channels. The trained "
              "V2 attention_only arm is the stronger learned comparator and is shown separately."))
    write_once(RUN / "analysis/summary.json", summary)
    for cohort in COHORTS:
        write_once(RUN / "analysis" / (cohort + "_details.json"), details[cohort])
    lines = ["V2 matched-view/K16 untrained raw-attention scalar baseline",
             "Retrospective, not blind. Pro500 primary; Sheets500 secondary.",
             "Identical frozen global/local Qwen traces and exact original K16 candidate points; no training or new model forward.",
             "Valid overlapping views averaged, no local coverage uses global only; empty traces skipped; strict score ties use lower K16 universe index.",
             "All 1,000 predictions SHA-sealed before this analysis opened labels.",
             "The trained attention_only V2 arm is the stronger learned comparator; this control does not prove non-attention hidden-state benefit.", ""]
    for cohort in COHORTS:
        report = reports[cohort]
        lines.append(f"{cohort.upper()} n={report['n']}, RGB clusters={report['rgb_clusters']}, null={report['null_point_count']}")
        for name in ("matched_raw_attention", "full_aligned_mean",
                     "full_attention_only_mean", "3k_aligned_mean",
                     "3k_attention_only_mean"):
            lines.append(f"  {name}: {100 * report['accuracy'][name]:.2f}%")
        for name, contrast in report["contrasts"].items():
            lines.append(f"  {name}: {100 * contrast['delta']:+.2f} pp; 95% RGB-cluster paired CI [{100 * contrast['ci95'][0]:+.2f}, {100 * contrast['ci95'][1]:+.2f}] pp; wins/losses/ties {contrast['wins']}/{contrast['losses']}/{contrast['ties']}")
        lines.append("")
    text = "\n".join(lines) + "\n"
    path = RUN / "analysis/readout.txt"
    if path.exists():
        require(path.read_text(encoding="utf-8") == text,
                "Existing readout differs")
    else:
        path.write_text(text, encoding="utf-8")
    status("analyzed", completed=1000, total=1000)
    print(text, flush=True)


def self_test():
    metadata = {"grid_shape": [2, 2], "head_order": HEAD_ORDER}
    a = np.zeros((2, 96, 4), np.float32)
    a[..., 1] = 2
    points = np.asarray([[1, 1], [7, 1], [9, 9]], np.int64)
    values, valid, tokens = view_score(a, metadata, points, [0, 0, 10, 10])
    require(tokens == 2 and valid.tolist() == [True, True, True] and
            values.tolist() == [0.0, 4.0, 0.0], "Patch mapping test failed")
    local_values, local_valid, _ = view_score(a, metadata, points, [5, 0, 10, 10])
    require(local_valid.tolist() == [False, True, True],
            "Local crop boundary test failed")
    pick, point, counts, _ = rank_attention(
        [(values, valid), (local_values, local_valid)], points, [2, 1, 0])
    require(pick == 1 and point == [7, 1] and counts == [1, 2, 2],
            "Overlap/valid-view aggregation test failed")
    zeros, zeros_valid, _ = view_score(a[:0], metadata, points, [0, 0, 10, 10])
    require(not zeros_valid.any() and
            rank_attention([(zeros, zeros_valid)], points, [0, 1, 2])[0] is None,
            "Empty trace/null test failed")
    tie, _, _, _ = rank_attention([(np.ones(3), valid)], points, [2, 1, 0])
    require(tie == 0, "Stable K16 strict-tie test failed")
    print("Synthetic raw-grid, overlap, empty-trace and K16-tie checks passed; no labels accessed", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True,
                        choices=("self-test", "register", "predict-pro", "predict-sheets",
                                 "predict-all", "seal", "verify-gate", "analyze"))
    stage = parser.parse_args().stage
    if stage == "self-test":
        self_test()
    elif stage == "register":
        register()
    elif stage == "predict-all":
        for cohort in COHORTS:
            predict(cohort)
    elif stage.startswith("predict-"):
        predict(stage.removeprefix("predict-"))
    elif stage == "seal":
        seal()
    elif stage == "verify-gate":
        gate()
        print("Both 500-case matched-view/K16 bundles and V2 seals verified; no labels opened", flush=True)
    else:
        analyze()


if __name__ == "__main__":
    main()
