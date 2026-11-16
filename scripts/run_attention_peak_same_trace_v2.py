"""Retrospective, label-gated attention-peak control on frozen V2 Qwen traces.

The public stages read only V2 public inputs, sealed pipeline metadata, and raw
own-token attention. All 2,581 predictions must be SHA-sealed before analysis
can open benchmark labels. No model forward, fitting, or method selection occurs.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
V2 = ROOT / "runs/paper_scale_ablation_v2"
EVAL = V2 / "eval"
RUN = ROOT / "runs/attention_peak_same_trace_v2"
PROTOCOL = RUN / "protocol.json"
COMPLETION = RUN / "public_completion.json"
COHORTS = ("pro", "sheets", "full")
SIZES = {"pro": 500, "sheets": 500, "full": 1581}
BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_SEED = 20261112


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
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(data, encoding="utf-8")
    tmp.replace(path)


def status(stage, cohort=None, completed=None, total=None):
    row = dict(stage=stage, cohort=cohort, completed=completed, total=total,
               at_utc=datetime.now(timezone.utc).isoformat())
    RUN.mkdir(parents=True, exist_ok=True)
    path = RUN / "status.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
    print(json.dumps(row, ensure_ascii=False), flush=True)


def source_files():
    paths = {
        "v25_raw_peak_definition": ROOT / "scripts/run_prompt_readout_v25.py",
        "v2_analysis_code": ROOT / "scripts/analyze_paper_scale_v2.py",
        "v2_statistical_registration": V2 / "statistical_analysis_protocol.json",
        "v2_full_completion": V2 / "public_completion.json",
        "v2_pointer_completion": V2 / "pointerbench/public_completion.json",
        "pointer_pro_official_scorer": V2 / "pointerbench/source/pro/eval.py",
        "pointer_sheets_official_scorer": V2 / "pointerbench/source/sheets/eval.py",
    }
    for cohort in COHORTS:
        paths[f"{cohort}_public"] = EVAL / cohort / "public.json"
        paths[f"{cohort}_feature_index"] = EVAL / cohort / "features/index.json"
        paths[f"{cohort}_v2_predictions"] = EVAL / cohort / "predictions.json"
    return paths


def expected_protocol():
    old_registration = read(V2 / "statistical_analysis_protocol.json")
    return dict(
        complete=True,
        experiment="attention_peak_same_trace_v2",
        purpose="Direct untrained global attention-peak mechanism control",
        retrospective=True,
        strict_blind=False,
        previously_previewed_pointer_answers_at_least=7,
        historical_screen_spot_pro_exposure=True,
        cohorts={"pro": 500, "sheets": 500, "full": 1581},
        primary_cohort="pro", secondary_cohort="sheets",
        descriptive_only_cohort="full",
        original_model="V2 frozen Qwen3-VL-8B original target-name trajectory",
        source_view="global screenshot only; frozen local views are not used by this peak control",
        attention_layers=[15, 28, 36], heads_per_layer=32,
        target_tokens="all generated target-name own-token forward positions; no token selection",
        rule=("For each target token and each of 96 heads, divide its raw image-patch "
              "attention by its image-patch mass (maximum denominator 1e-12). "
              "Arithmetic-mean all normalized token-head maps; take NumPy row-major "
              "first argmax patch, then Python round of that patch center in original "
              "image pixels, clipped to image bounds. T=0 or zero total image attention "
              "produces null and is scored wrong. This is the pre-existing V25 raw-peak "
              "mathematics, without choosing a head, layer, token, or crop by labels."),
        fixed_denominators=True,
        labels_open_only_after_three_complete_prediction_bundles_and_sha_receipt=True,
        metric_pointer="Pinned official inclusive point-in-bbox, with V2 scorer parity",
        metric_full="V2 normalized point-in-geometry",
        primary_contrast="Pro500 V2 full aligned three-seed mean minus global raw peak",
        descriptive_contrasts=["Pro500 V2 3k aligned mean minus global raw peak",
                               "Sheets500 full/3k aligned mean minus global raw peak",
                               "full aligned S1 and 3k aligned S1 versus global raw peak"],
        bootstrap_unit="RGB image pixel_sha256 cluster",
        bootstrap_replicates=BOOTSTRAP_REPLICATES,
        bootstrap_seed=BOOTSTRAP_SEED,
        bootstrap_interval="descriptive two-sided percentile 95%",
        inference_cost_comparable=False,
        causal_internal_evidence_claim=False,
        runner_sha256=sha(__file__),
        source_file_sha256={name: sha(path) for name, path in source_files().items()},
        labels_sha256_unopened={
            "pointer": old_registration["pointer_labels_sha256_unopened"],
            "full": old_registration["full_labels_sha256_unopened"],
        },
    )


def registered():
    require(PROTOCOL.exists(), "Run --stage register before inference")
    protocol = read(PROTOCOL)
    require(protocol == expected_protocol(), "Registered protocol or source drift")
    return protocol


def register():
    write_once(PROTOCOL, expected_protocol())
    status("registered", completed=0, total=sum(SIZES.values()))


def peak_point(attention, metadata, original_size):
    """Use exactly the prior V25 normalized mean-map/argmax/round rule."""
    a = np.asarray(attention, np.float32)
    gh, gw = [int(n) for n in metadata["grid_shape"]]
    require(a.ndim == 3 and a.shape[1:] == (96, gh * gw) and gh > 0 and gw > 0,
            "Unexpected frozen attention tensor/grid")
    require(np.isfinite(a).all() and (a >= 0).all(), "Invalid raw attention")
    require(metadata["head_order"] ==
            [f"layer_{layer}.head_{head}" for layer in (15, 28, 36)
             for head in range(32)], "Attention head/layer order drift")
    t = len(a)
    if t == 0 or not bool(np.any(a > 0)):
        return None, None, t
    relative = a / np.maximum(a.sum(-1, keepdims=True), 1e-12)
    patch = int(relative.mean((0, 1)).argmax())
    row, col = divmod(patch, gw)
    width, height = map(int, original_size)
    point = [min(width - 1, max(0, round((col + .5) * width / gw))),
             min(height - 1, max(0, round((row + .5) * height / gh)))]
    return point, patch, t


def public_index(cohort):
    registered()
    rows = read(EVAL / cohort / "public.json")
    index = read(EVAL / cohort / "features/index.json")
    require(index["complete"] is True and index["cohort"] == cohort and
            index["n"] == SIZES[cohort] and index["labels_present"] is False and
            index["ground_truth_in_model_input"] is False and
            index["public_sha256"] == sha(EVAL / cohort / "public.json") and
            len(index["cases"]) == len(rows) == SIZES[cohort] and
            [e["sample_id"] for e in index["cases"]] ==
            [r["sample_id"] for r in rows], "Public/index identity mismatch: " + cohort)
    return rows, index["cases"]


def predict_case(cohort, row, entry):
    require(entry["labels_present"] is False and
            entry["ground_truth_in_model_input"] is False,
            "Forbidden labeled feature: " + row["sample_id"])
    require(sha(entry["pipeline_path"]) == entry["pipeline_sha256"],
            "Frozen pipeline SHA changed: " + row["sample_id"])
    pipe = read(entry["pipeline_path"])
    require(pipe["complete"] is True and pipe["sample_id"] == row["sample_id"] and
            pipe["ground_truth_in_model_inputs"] is False and
            pipe["input_identity"] == entry["case_identity"],
            "Pipeline public identity or GT guard mismatch: " + row["sample_id"])
    raw_path = pipe["global_feature_path"]
    require(sha(raw_path) == pipe["global_feature_sha256"],
            "Global raw attention SHA changed: " + row["sample_id"])
    with np.load(raw_path, allow_pickle=False) as z:
        metadata = json.loads(str(z["metadata"]))
        a = z["target_attention"]
    require(metadata["own_token_forward"] is True and
            metadata["ground_truth_in_model_input"] is False and
            metadata["input_identity"] == Path(raw_path).stem and
            metadata["sample_id"] == row["sample_id"] and
            metadata["image_size"] == row["image_size"],
            "Original own-token trace/public input mismatch: " + row["sample_id"])
    point, patch, t = peak_point(a, metadata, row["image_size"])
    return dict(complete=True, sample_id=row["sample_id"], cohort=cohort,
                case_identity=entry["case_identity"],
                protocol_sha256=sha(PROTOCOL), labels_opened=False,
                global_raw_sha256=pipe["global_feature_sha256"],
                pipeline_sha256=entry["pipeline_sha256"],
                target_name_token_count=t, grid_shape=metadata["grid_shape"],
                point=point, patch_index=patch)


def case_path(cohort, sample_id):
    return RUN / "predictions" / cohort / "cases" / (case_key(sample_id) + ".json")


def verify_case(case, cohort, row, entry):
    point, patch = case.get("point"), case.get("patch_index")
    width, height = row["image_size"]
    require(case.get("complete") is True and case.get("labels_opened") is False and
            case.get("sample_id") == row["sample_id"] and
            case.get("cohort") == cohort and
            case.get("case_identity") == entry["case_identity"] and
            case.get("protocol_sha256") == sha(PROTOCOL) and
            case.get("pipeline_sha256") == entry["pipeline_sha256"] and
            type(case.get("target_name_token_count")) is int and
            case["target_name_token_count"] >= 0 and
            (point is None and patch is None or
             isinstance(point, list) and len(point) == 2 and
             all(type(v) is int for v in point) and
             0 <= point[0] < width and 0 <= point[1] < height and
             type(patch) is int and
             0 <= patch < math.prod(case["grid_shape"])),
            "Frozen per-case prediction invalid: " + row["sample_id"])


def predict(cohort):
    rows, entries = public_index(cohort)
    status("predicting", cohort, 0, len(rows))
    started = time.perf_counter()
    hashes, outputs = {}, []
    for ix, (row, entry) in enumerate(zip(rows, entries), 1):
        path = case_path(cohort, row["sample_id"])
        if path.exists():
            case = read(path)
            verify_case(case, cohort, row, entry)
        else:
            case = predict_case(cohort, row, entry)
            write_once(path, case)
        hashes[row["sample_id"]] = sha(path)
        outputs.append(dict(sample_id=row["sample_id"], point=case["point"],
                            patch_index=case["patch_index"],
                            target_name_token_count=case["target_name_token_count"]))
        if ix % 25 == 0 or ix == len(rows):
            status("predicting", cohort, ix, len(rows))
    bundle = dict(complete=True, cohort=cohort, n=len(rows),
                  public_sha256=sha(EVAL / cohort / "public.json"),
                  feature_index_sha256=sha(EVAL / cohort / "features/index.json"),
                  protocol_sha256=sha(PROTOCOL), labels_opened=False,
                  case_hashes=hashes, rows=outputs)
    write_once(RUN / "predictions" / (cohort + ".json"), bundle)
    status("predicted", cohort, len(rows), len(rows))
    print(json.dumps({"cohort": cohort, "seconds": round(time.perf_counter() - started, 2),
                      "prediction_sha256": sha(RUN / "predictions" / (cohort + ".json"))}),
          flush=True)


def verify_bundle(cohort):
    rows, entries = public_index(cohort)
    path = RUN / "predictions" / (cohort + ".json")
    require(path.exists(), "Missing prediction bundle: " + cohort)
    bundle = read(path)
    require(bundle.get("complete") is True and bundle.get("cohort") == cohort and
            bundle.get("n") == SIZES[cohort] and bundle.get("labels_opened") is False and
            bundle.get("protocol_sha256") == sha(PROTOCOL) and
            bundle.get("public_sha256") == sha(EVAL / cohort / "public.json") and
            bundle.get("feature_index_sha256") ==
            sha(EVAL / cohort / "features/index.json") and
            len(bundle.get("rows", [])) == len(rows) and
            set(bundle.get("case_hashes", {})) == {r["sample_id"] for r in rows},
            "Prediction bundle identity mismatch: " + cohort)
    for row, entry, output in zip(rows, entries, bundle["rows"]):
        sid = row["sample_id"]
        path = case_path(cohort, sid)
        require(path.exists() and sha(path) == bundle["case_hashes"][sid],
                "Prediction case SHA changed: " + sid)
        case = read(path)
        verify_case(case, cohort, row, entry)
        require(output == {"sample_id": sid, "point": case["point"],
                           "patch_index": case["patch_index"],
                           "target_name_token_count": case["target_name_token_count"]},
                "Aggregate/case disagreement: " + sid)
    return rows, bundle, sha(RUN / "predictions" / (cohort + ".json"))


def seal():
    registered()
    bundles = {cohort: verify_bundle(cohort) for cohort in COHORTS}
    receipt = dict(complete=True, labels_opened=False,
                   protocol_sha256=sha(PROTOCOL),
                   cohorts={cohort: SIZES[cohort] for cohort in COHORTS},
                   prediction_sha256={cohort: bundles[cohort][2] for cohort in COHORTS},
                   all_2581_predictions_present=True)
    write_once(COMPLETION, receipt)
    status("sealed", completed=sum(SIZES.values()), total=sum(SIZES.values()))


def gate():
    registered()
    require(COMPLETION.exists(), "All three predictions must be SHA-sealed first")
    receipt = read(COMPLETION)
    require(receipt.get("complete") is True and receipt.get("labels_opened") is False and
            receipt.get("protocol_sha256") == sha(PROTOCOL) and
            receipt.get("cohorts") == SIZES and
            receipt.get("all_2581_predictions_present") is True,
            "Public completion receipt invalid")
    bundles = {cohort: verify_bundle(cohort) for cohort in COHORTS}
    require(receipt["prediction_sha256"] ==
            {cohort: bundles[cohort][2] for cohort in COHORTS},
            "Prediction SHA receipt drift")
    # The old V2 gate checks all 39 previously frozen per-case predictions and
    # all V2 receipts, without opening labels. It is also required here.
    import analyze_paper_scale_v2 as v2
    old_protocol, old_bundles, _ = v2.gate()
    for cohort in COHORTS:
        old_rows = old_bundles[cohort][0]
        require([r["sample_id"] for r in bundles[cohort][0]] ==
                [r["sample_id"] for r in old_rows],
                "V2/new public cohort order drift: " + cohort)
    return bundles, old_protocol, old_bundles, v2


def official_scorer(cohort):
    path = V2 / "pointerbench/source" / cohort / "eval.py"
    spec = importlib.util.spec_from_file_location("peak_pointer_" + cohort, path)
    require(spec is not None and spec.loader is not None,
            "Official scorer cannot load: " + cohort)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._in_bbox


def paired_ci(left, right, rows, seed):
    delta = np.asarray(left, np.float64) - np.asarray(right, np.float64)
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row["pixel_sha256"]].append(i)
    clusters = list(groups.values())
    n = len(clusters)
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(n, np.full(n, 1 / n),
                            size=BOOTSTRAP_REPLICATES).astype(np.float32)
    numer = np.asarray([delta[ix].sum() for ix in clusters], np.float32)
    denom = np.asarray([len(ix) for ix in clusters], np.float32)
    sample = (draws @ numer) / (draws @ denom)
    return dict(delta=float(delta.mean()),
                ci95=np.quantile(sample, [.025, .975]).astype(float).tolist(),
                rgb_clusters=n, wins=int(np.sum(delta > 0)),
                losses=int(np.sum(delta < 0)), ties=int(np.sum(delta == 0)))


def analyze():
    # No label path is read, hashed, imported, or opened before this gate returns.
    bundles, old_protocol, old_bundles, v2 = gate()
    require(sha(v2.FULL_LABELS) == read(PROTOCOL)["labels_sha256_unopened"]["full"] and
            sha(v2.POINTER_LABELS) == read(PROTOCOL)["labels_sha256_unopened"]["pointer"],
            "Label digest changed after public sealing")
    geometry = v2.open_labels_after_gate(old_protocol, old_bundles)
    pointer_labels = {r["sample_id"]: r for r in v2.jsonl(v2.POINTER_LABELS)}
    from clickagent.public_evaluation import point_in_geometry
    reports, details = {}, {}
    for cohort in COHORTS:
        rows, bundle, _ = bundles[cohort]
        old_cases = old_bundles[cohort][1]
        official = official_scorer(cohort) if cohort != "full" else None
        vectors = {"peak": [], **{f"{scale}_aligned_s{s}": []
                                 for scale in ("3k", "full") for s in (1, 2, 3)}}
        per_case = []
        for row, output in zip(rows, bundle["rows"]):
            sid = row["sample_id"]
            width, height = row["image_size"]
            points = {"peak": output["point"]}
            points.update({name: old_cases[sid]["methods"][name]["point"]
                           for name in vectors if name != "peak"})
            outcomes = {}
            for name, point in points.items():
                hit = bool(point is not None and point_in_geometry(
                    (point[0] / width, point[1] / height), geometry[cohort][sid]))
                if official is not None:
                    native = bool(point is not None and official(
                        tuple(point), pointer_labels[sid]["bbox_pixels"]))
                    require(hit == native, "V2/official scorer parity failed: " + sid)
                    hit = native
                vectors[name].append(float(hit))
                outcomes[name] = hit
            per_case.append(dict(sample_id=sid, pixel_sha256=row["pixel_sha256"],
                                 point=output["point"], hits=outcomes))
        vectors["3k_aligned_mean"] = np.mean([vectors[f"3k_aligned_s{s}"]
                                              for s in (1, 2, 3)], axis=0)
        vectors["full_aligned_mean"] = np.mean([vectors[f"full_aligned_s{s}"]
                                                for s in (1, 2, 3)], axis=0)
        accuracy = {name: float(np.mean(values)) for name, values in vectors.items()}
        contrasts = {}
        if cohort != "full":
            seed = BOOTSTRAP_SEED + (0 if cohort == "pro" else 1)
            for name in ("full_aligned_mean", "3k_aligned_mean", "full_aligned_s1"):
                contrasts[name + "_minus_peak"] = paired_ci(
                    vectors[name], vectors["peak"], rows, seed)
        reports[cohort] = dict(n=SIZES[cohort],
                               rgb_clusters=len({r["pixel_sha256"] for r in rows}),
                               accuracy=accuracy, counts={name: int(sum(values))
                                                          for name, values in vectors.items()
                                                          if name != "3k_aligned_mean" and
                                                          name != "full_aligned_mean"},
                               contrasts=contrasts,
                               no_target_name_count=sum(r["target_name_token_count"] == 0
                                                        for r in bundle["rows"]),
                               null_point_count=sum(r["point"] is None
                                                    for r in bundle["rows"]),
                               scorer="official PointerBench with V2 parity" if official
                                      else "V2 ScreenSpot-Pro normalized geometry",
                               retrospective=True, strict_blind=False)
        details[cohort] = per_case
    summary = dict(complete=True, protocol_sha256=sha(PROTOCOL),
                   public_completion_sha256=sha(COMPLETION), results=reports,
                   note=("Retrospective comparison after V2 labels/results were observed. "
                         "Global one-view attention peak versus a trained K16 multi-view "
                         "pipeline; observation budgets differ, so this is a simple "
                         "mechanism/motivation control, not a causal internal-evidence "
                         "ablation or proof of general superiority."))
    write_once(RUN / "analysis/summary.json", summary)
    for cohort in COHORTS:
        write_once(RUN / "analysis" / (cohort + "_details.json"), details[cohort])
    lines = ["Global raw-attention peak on frozen V2 original Qwen trajectories",
             "Retrospective, not blind. Pro500 primary; Sheets500 secondary; SSP1581 historical descriptive.",
             "All 96 heads (layers 15/28/36) and all generated target-name tokens, image-normalized per map, arithmetic mean, row-major argmax cell center.",
             "Single global view vs trained K16 multi-view ClickAgent; unequal observation budgets. This control cannot establish an internal-evidence causal effect.",
             "All three cohort predictions sealed before any benchmark label opened; null points count wrong.", ""]
    for cohort in COHORTS:
        report = reports[cohort]
        lines.append(f"{cohort.upper()} n={report['n']}, RGB clusters={report['rgb_clusters']}, null={report['null_point_count']}")
        for name in ("peak", "3k_aligned_mean", "full_aligned_mean"):
            lines.append(f"  {name}: {100 * report['accuracy'][name]:.2f}%")
        for name, contrast in report["contrasts"].items():
            lines.append(f"  {name}: {100 * contrast['delta']:+.2f} pp; 95% RGB-cluster paired CI [{100 * contrast['ci95'][0]:+.2f}, {100 * contrast['ci95'][1]:+.2f}] pp; wins/losses/ties {contrast['wins']}/{contrast['losses']}/{contrast['ties']}")
        lines.append("")
    text = "\n".join(lines) + "\n"
    readout = RUN / "analysis/readout.txt"
    if readout.exists():
        require(readout.read_text(encoding="utf-8") == text,
                "Existing readout differs")
    else:
        readout.write_text(text, encoding="utf-8")
    status("analyzed", completed=sum(SIZES.values()), total=sum(SIZES.values()))
    print(text, flush=True)


def self_test():
    order = [f"layer_{layer}.head_{head}" for layer in (15, 28, 36)
             for head in range(32)]
    meta = {"grid_shape": [2, 3], "head_order": order}
    a = np.zeros((2, 96, 6), np.float32)
    a[..., 4] = 2
    point, patch, t = peak_point(a, meta, [600, 400])
    require((point, patch, t) == ([300, 300], 4, 2), "Peak center test failed")
    require(peak_point(a[:0], meta, [600, 400]) == (None, None, 0),
            "No-token failure test failed")
    require(peak_point(np.zeros_like(a), meta, [600, 400]) == (None, None, 2),
            "Zero-mass failure test failed")
    print("synthetic peak and failure-case tests passed; no labels accessed", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True,
                        choices=("self-test", "register", "predict-pro", "predict-sheets",
                                 "predict-full", "predict-all", "seal", "verify-gate", "analyze"))
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
        print("All 2,581 predictions and V2 bundles verified; no labels opened", flush=True)
    else:
        analyze()


if __name__ == "__main__":
    main()
