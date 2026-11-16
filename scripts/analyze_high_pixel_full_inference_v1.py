"""Registered, CPU-only analysis of sealed 1M-pixel ClickAgent predictions.

``register`` and ``verify-gate`` never open benchmark labels. ``analyze`` opens
them only after all 2,581 new predictions and all archived 262k cases pass
SHA and identity checks. No model inference, fitting, selection or GPU work.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from clickagent.public_evaluation import point_in_geometry

RUN = ROOT / "runs/high_pixel_full_inference_v1"
V2 = ROOT / "runs/paper_scale_ablation_v2"
OLD_EVAL = V2 / "eval"
REG = RUN / "analysis_registration.json"
OUT = RUN / "analysis"
MANIFEST = RUN / "public_manifest.json"
PROTOCOL = RUN / "protocol.json"
GLOBAL_SEAL = RUN / "public_completion.json"
OLD_FULL_SEAL = V2 / "public_completion.json"
OLD_POINTER_SEAL = V2 / "pointerbench/public_completion.json"
OLD_STATS = V2 / "statistical_analysis_protocol.json"
HISTORY = ROOT / "runs/paper_scale_ablation_v1/data/full_history_overlap_corrected_v2.json"
SSP_LABELS = ROOT / "data/public_v1/manifests/screenspot_pro.jsonl"
POINTER_LABELS = V2 / "pointerbench/labels.jsonl"
SUITES = ("ssp", "pro", "sheets")
OLD_SUITE = {"ssp": "full", "pro": "pro", "sheets": "sheets"}
COUNTS = {"ssp": 1581, "pro": 500, "sheets": 500}
MODEL_NAMES = tuple(f"full_aligned_s{s}" for s in (1, 2, 3))
OLD_PUBLIC_KEYS = ("sample_id", "dataset", "image", "image_size", "image_sha256", "instruction")
BUDGET = 1048576
OLD_BUDGET = 262144
REPLICATES = 10000
BOOTSTRAP_SEED = 20261115
ARCHIVED_262K_HITS = {"ssp": [312, 308, 298], "pro": [220, 218, 218],
                      "sheets": [117, 116, 122]}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def jsonl(path):
    with Path(path).open("r", encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def case_key(sample_id):
    return hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:24]


def write_once(path, content, *, text=False):
    path = Path(path)
    serial = content if text else json.dumps(content, indent=2, ensure_ascii=False) + "\n"
    if path.exists():
        require(path.read_text(encoding="utf-8") == serial,
                "Immutable output differs: " + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    require(not temporary.exists(), "Unresolved temporary output: " + str(temporary))
    temporary.write_text(serial, encoding="utf-8")
    temporary.replace(path)


def pinned_public_sources():
    paths = {"high_protocol": PROTOCOL, "high_manifest": MANIFEST,
             "old_full_receipt": OLD_FULL_SEAL,
             "old_pointer_receipt": OLD_POINTER_SEAL,
             "old_stats_registration": OLD_STATS,
             "history": HISTORY,
             "local_geometry_rule": ROOT / "src/clickagent/public_evaluation.py"}
    for suite in SUITES:
        old = OLD_SUITE[suite]
        paths[f"old_{suite}_public"] = OLD_EVAL / old / "public.json"
        paths[f"old_{suite}_predictions"] = OLD_EVAL / old / "predictions.json"
    for suite in ("pro", "sheets"):
        paths[f"official_{suite}_scorer"] = (
            V2 / "pointerbench/source" / suite / "eval.py")
    return {name: sha(path) for name, path in paths.items()}


def public_match(manifest):
    require(manifest.get("complete") is True and
            manifest.get("labels_present") is False and
            manifest.get("counts") == COUNTS and
            set(manifest.get("cases", {})) == set(SUITES),
            "High-pixel public manifest changed")
    all_ids = []
    for suite in SUITES:
        current = manifest["cases"][suite]
        old = read(OLD_EVAL / OLD_SUITE[suite] / "public.json")
        require(len(current) == len(old) == COUNTS[suite],
                "Cohort denominator changed: " + suite)
        for new_row, old_row in zip(current, old):
            require(new_row["suite"] == suite and
                    all(new_row[field] == old_row[field] for field in OLD_PUBLIC_KEYS) and
                    new_row["pixel_sha256"] == old_row["pixel_sha256"],
                    "1M/262k public case, order, RGB or frame mismatch: " + suite)
            all_ids.append(new_row["sample_id"])
    require(len(all_ids) == len(set(all_ids)) == sum(COUNTS.values()) and
            len({case_key(sid) for sid in all_ids}) == len(all_ids),
            "Duplicate public ID/case filename")


def registration_record():
    protocol, manifest, old_stats = read(PROTOCOL), read(MANIFEST), read(OLD_STATS)
    public_match(manifest)
    require(protocol.get("complete") is True and
            protocol.get("pixel_cap_per_view") == BUDGET and
            protocol.get("suites") == list(SUITES) and
            protocol.get("denominators") == COUNTS and
            protocol.get("labels_may_open_in_this_worker") is False and
            protocol.get("paper_mainline_262144_unchanged") is True and
            "original V15 K16" in protocol.get("scientific_policy", "") and
            "forced Top1" in protocol.get("scientific_policy", "") and
            set(protocol.get("models", {})) == set(MODEL_NAMES),
            "Frozen 1M scientific protocol mismatch")
    model_hashes = {name: protocol["models"][name]["sha256"] for name in MODEL_NAMES}
    require(all(old_stats["model_hashes"][name] == model_hashes[name]
                for name in MODEL_NAMES), "1M model identity differs from V2")
    return {
        "complete": True,
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Retrospective 1M-vs-262k pixel-cap sensitivity, fixed original K16 and full aligned S1-S3",
        "analysis_script_sha256": sha(Path(__file__)),
        "public_source_sha256": pinned_public_sources(),
        "expected_label_sha256_unopened": {
            "ssp": old_stats["full_labels_sha256_unopened"],
            "pointer": old_stats["pointer_labels_sha256_unopened"]},
        "model_hashes": model_hashes,
        "suites": list(SUITES), "denominators": COUNTS,
        "new_pixel_cap_per_view": BUDGET,
        "archived_pixel_cap_per_view": OLD_BUDGET,
        "main_measure": "point-in-target exact hit; arithmetic mean of the three fixed seed accuracies",
        "registered_comparisons": [
            "ssp 1M full aligned S1-S3 vs same-case 262k full aligned S1-S3",
            "pro 1M full aligned S1-S3 vs same-case 262k full aligned S1-S3",
            "sheets 1M full aligned S1-S3 vs same-case 262k full aligned S1-S3"],
        "new_k16_oracle": "any original K16 point in target; full pool oracle also descriptive",
        "reachable_selection": "mean seed hits divided by number of K16-reachable queries",
        "bootstrap_unit": "RGB image pixel_sha256",
        "bootstrap_replicates": REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "interval": "descriptive paired 95% percentile bootstrap; query-weighted, seed-mean contrast; conditional on trained seeds",
        "pointer_scorer": "released _in_bbox inclusive native-pixel scorer with per-point parity against V2 geometry rule",
        "ssp_scorer": "V2 normalized point_in_geometry convention; no independent pinned official scorer found",
        "prediction_gate": "all 2581 new per-case SHA, three new suite SHA and global public SHA; archived per-case SHA and receipts; before any label read",
        "known_limits": [
            "SSP1581 historically exposed; descriptive only, no new independent confirmation",
            "PointerBench labels already seen in earlier V2 work; at least seven answers previewed; retrospective and not strictly blind",
            "Pixel-cap comparison changes the extracted evidence and candidate pool; it is a system-level sensitivity, not a fixed-candidate causal effect",
            "Latency includes image decoding and local runtime; do not equate with MVP timing or infer equal total compute"],
        "no_seed_selection": True,
        "no_new_training_or_gpu": True,
    }


def register():
    expected = registration_record()
    if REG.exists():
        observed = read(REG)
        expected["registered_at_utc"] = observed.get("registered_at_utc")
    write_once(REG, expected)
    print(json.dumps({"phase": "analysis_registered", "sha256": sha(REG),
                      "labels_opened": False}, ensure_ascii=False))


def valid_point(point, size):
    return (type(point) is list and len(point) == 2 and
            all(type(value) in (int, float) and math.isfinite(value) and
                0 <= value < limit for value, limit in zip(point, size)))


def verify_new_case(row, public, model_hashes):
    sid, suite = public["sample_id"], public["suite"]
    require(row.get("complete") is True and row.get("labels_opened") is False and
            row.get("ground_truth_in_model_input") is False and
            row.get("sample_id") == sid and row.get("suite") == suite and
            row.get("public_identity") == identity(public) and
            row.get("input_image_sha256") == public["image_sha256"] and
            row.get("image_size") == public["image_size"] and
            row.get("pixel_cap_per_view") == BUDGET and
            row.get("model_hashes") == model_hashes,
            "New prediction identity/model/GT flag mismatch: " + sid)
    k16, pool, seeds = row["k16_points"], row["candidate_points"], row["seed_points"]
    require(type(k16) is list and type(pool) is list and type(seeds) is list and
            len(seeds) == len(MODEL_NAMES) and
            row.get("point") == seeds[0] and
            row.get("k16_count") == len(k16) <= 16 and
            row.get("candidate_count") == len(pool) >= len(k16),
            "New K16/pool/seed schema mismatch: " + sid)
    size = public["image_size"]
    require(all(valid_point(p, size) for p in k16 + pool),
            "New candidate outside native image: " + sid)
    require(all((p is None and not k16) or
                (valid_point(p, size) and p in k16) for p in seeds),
            "New seed prediction outside original K16: " + sid)
    require(all(p in pool for p in k16), "New K16 not in generated pool: " + sid)
    measured = row.get("measured")
    require(type(measured) is dict and
            type(measured.get("elapsed_seconds")) in (int, float) and
            math.isfinite(measured["elapsed_seconds"]) and
            measured["elapsed_seconds"] > 0 and
            type(measured.get("counter")) is dict,
            "New per-query timing missing: " + sid)


def verify_old_case(row, public, bundle_hashes, model_hashes):
    sid = public["sample_id"]
    old_identity = identity({k: public[k] for k in OLD_PUBLIC_KEYS})
    require(row.get("complete") is True and row.get("labels_opened") is False and
            row.get("sample_id") == sid and
            row.get("public_identity") == old_identity and
            row.get("model_hashes") == bundle_hashes and
            set(row.get("methods", {})) == set(bundle_hashes),
            "Archived case identity/methods mismatch: " + sid)
    for name in MODEL_NAMES:
        require(bundle_hashes[name] == model_hashes[name],
                "Archived model SHA differs from 1M: " + name)
        output = row["methods"][name]
        point, index = output["point"], output["index"]
        require((point is None and index is None) or
                (valid_point(point, public["image_size"]) and
                 type(index) is int and 0 <= index < row["candidate_count"] and
                 row["K16_count"] > 0),
                "Archived aligned prediction malformed: " + sid)


def verify_gate():
    """Verify every public/SHA prerequisite, with no GT path access at all."""
    require(REG.exists(), "Register analysis before opening any benchmark labels")
    reg = read(REG)
    require(reg.get("complete") is True and reg.get("analysis_script_sha256") == sha(__file__) and
            reg.get("public_source_sha256") == pinned_public_sources() and
            reg.get("suites") == list(SUITES) and reg.get("denominators") == COUNTS and
            reg.get("new_pixel_cap_per_view") == BUDGET and
            reg.get("archived_pixel_cap_per_view") == OLD_BUDGET and
            reg.get("bootstrap_replicates") == REPLICATES and
            reg.get("bootstrap_seed") == BOOTSTRAP_SEED and
            reg.get("no_seed_selection") is True,
            "Prospective analysis registration or public source drift")
    # Check the global seal first: an incomplete run fails before any model hash
    # or per-case scan, and before the label-open code path can be reached.
    require(GLOBAL_SEAL.exists(), "All 2581 unlabeled 1M predictions must be sealed first")
    protocol, manifest, global_receipt = read(PROTOCOL), read(MANIFEST), read(GLOBAL_SEAL)
    public_match(manifest)
    require(protocol.get("complete") is True and
            protocol.get("pixel_cap_per_view") == BUDGET and
            protocol.get("labels_may_open_in_this_worker") is False and
            protocol.get("denominators") == COUNTS and
            protocol.get("suites") == list(SUITES) and
            protocol.get("public_manifest_sha256") == sha(MANIFEST) and
            protocol.get("models") is not None and
            {name: protocol["models"][name]["sha256"] for name in MODEL_NAMES} == reg["model_hashes"],
            "Frozen 1M protocol/model identity changed")
    for name in MODEL_NAMES:
        require(sha(Path(protocol["models"][name]["path"])) == reg["model_hashes"][name],
                "Frozen full-aligned checkpoint changed: " + name)
    require(global_receipt.get("complete") is True and
            global_receipt.get("labels_opened") is False and
            global_receipt.get("n") == sum(COUNTS.values()) and
            global_receipt.get("suite_n") == COUNTS and
            global_receipt.get("protocol_sha256") == sha(PROTOCOL) and
            global_receipt.get("public_manifest_sha256") == sha(MANIFEST) and
            set(global_receipt.get("suite_receipt_sha256", {})) == set(SUITES),
            "Global 1M public prediction seal incomplete")
    old_full, old_pointer = read(OLD_FULL_SEAL), read(OLD_POINTER_SEAL)
    require(old_full.get("complete") is True and old_full.get("labels_opened") is False and
            old_pointer.get("complete") is True and
            old_pointer.get("labels_opened") is False,
            "Archived 262k public prediction receipts invalid")
    new_rows, old_rows, public_rows = {}, {}, {}
    for suite in SUITES:
        current_public = manifest["cases"][suite]
        old_suite = OLD_SUITE[suite]
        old_public = read(OLD_EVAL / old_suite / "public.json")
        public_rows[suite] = current_public
        folder = RUN / suite
        receipt_path, predictions_path = folder / "completion.json", folder / "predictions.json"
        require(global_receipt["suite_receipt_sha256"][suite] == sha(receipt_path),
                "1M suite receipt SHA mismatch: " + suite)
        receipt, predictions = read(receipt_path), read(predictions_path)
        require(receipt.get("complete") is True and receipt.get("labels_opened") is False and
                receipt.get("suite") == suite and receipt.get("n") == COUNTS[suite] and
                receipt.get("protocol_sha256") == sha(PROTOCOL) and
                receipt.get("predictions_sha256") == sha(predictions_path) and
                predictions.get("complete") is True and
                predictions.get("labels_opened") is False and
                predictions.get("suite") == suite and
                predictions.get("n") == COUNTS[suite] and
                predictions.get("pixel_cap_per_view") == BUDGET and
                predictions.get("protocol_sha256") == sha(PROTOCOL) and
                predictions.get("public_manifest_sha256") == sha(MANIFEST) and
                predictions.get("model_hashes") == reg["model_hashes"] and
                len(predictions.get("rows", [])) == COUNTS[suite] and
                set(predictions.get("case_hashes", {})) ==
                {row["sample_id"] for row in current_public},
                "1M suite prediction bundle/receipt mismatch: " + suite)
        old_folder = OLD_EVAL / old_suite
        old_pred = read(old_folder / "predictions.json")
        expected_old = (old_full["full_predictions_sha256"] if suite == "ssp" else
                        old_pointer["prediction_sha256"][suite])
        require(expected_old == sha(old_folder / "predictions.json") and
                old_pred.get("complete") is True and
                old_pred.get("labels_opened") is False and
                old_pred.get("n") == COUNTS[suite] and
                old_pred.get("cohort") == old_suite and
                old_pred.get("public_sha256") == sha(old_folder / "public.json") and
                set(old_pred.get("case_hashes", {})) ==
                {row["sample_id"] for row in old_public},
                "Archived 262k bundle/receipt mismatch: " + suite)
        new_cases, old_cases = {}, {}
        for public, archived_public, embedded in zip(current_public, old_public,
                                                      predictions["rows"]):
            sid = public["sample_id"]
            new_path = folder / "cases" / (case_key(sid) + ".json")
            old_path = old_folder / "predictions/cases" / (case_key(sid) + ".json")
            require(sha(new_path) == predictions["case_hashes"][sid],
                    "1M per-case SHA mismatch: " + sid)
            require(sha(old_path) == old_pred["case_hashes"][sid],
                    "262k per-case SHA mismatch: " + sid)
            new_case, old_case = read(new_path), read(old_path)
            require(new_case == embedded, "1M suite bundle row differs from sealed case: " + sid)
            verify_new_case(new_case, public, reg["model_hashes"])
            verify_old_case(old_case, archived_public, old_pred["model_hashes"],
                            reg["model_hashes"])
            new_cases[sid], old_cases[sid] = new_case, old_case
        new_rows[suite], old_rows[suite] = new_cases, old_cases
    history = read(HISTORY)
    require(history.get("complete") is True and history.get("n") == 1581 and
            len(history.get("rows", [])) == 1581 and
            {r["sample_id"]: r["pixel_sha256"] for r in history["rows"]} ==
            {r["sample_id"]: r["pixel_sha256"] for r in public_rows["ssp"]},
            "SSP historical exposure strata changed")
    return reg, public_rows, new_rows, old_rows, history


def load_labels_after_gate(reg, public_rows):
    """The only label-open path. Caller must first return from verify_gate()."""
    require(sha(SSP_LABELS) == reg["expected_label_sha256_unopened"]["ssp"] and
            sha(POINTER_LABELS) == reg["expected_label_sha256_unopened"]["pointer"],
            "Benchmark label SHA differs from pre-registered digest")
    ssp, pointer = jsonl(SSP_LABELS), jsonl(POINTER_LABELS)
    require(len(ssp) == COUNTS["ssp"] and
            len(pointer) == COUNTS["pro"] + COUNTS["sheets"],
            "Benchmark annotation denominator changed")
    ssp_by = {r["sample_id"]: r for r in ssp}
    pointer_by = {r["sample_id"]: r for r in pointer}
    require(len(ssp_by) == len(ssp) and len(pointer_by) == len(pointer) and
            set(ssp_by) == {r["sample_id"] for r in public_rows["ssp"]} and
            set(pointer_by) == {r["sample_id"] for s in ("pro", "sheets")
                                for r in public_rows[s]},
            "Sealed public cases and annotations disagree")
    return {"ssp": ssp_by, "pro": pointer_by, "sheets": pointer_by}


def load_pointer_scorer(suite, reg):
    path = V2 / "pointerbench/source" / suite / "eval.py"
    require(sha(path) == reg["public_source_sha256"][f"official_{suite}_scorer"],
            "Official PointerBench scorer SHA drift")
    spec = importlib.util.spec_from_file_location(f"pointerbench_{suite}_highpixel_eval", path)
    require(spec is not None and spec.loader is not None,
            "Official PointerBench scorer cannot load")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(callable(getattr(module, "_in_bbox", None)),
            "Official PointerBench _in_bbox is missing")
    return module._in_bbox


def score_point(point, label, size, official=None, parity=None):
    local = bool(point is not None and point_in_geometry(
        (point[0] / size[0], point[1] / size[1]), label["target_geometry"]))
    if official is None:
        return local
    native = bool(point is not None and official(tuple(point), label["bbox_pixels"]))
    require(local == native, "PointerBench official/V2 point scorer parity failed")
    if parity is not None:
        parity["points_checked"] += 1
    return native


def paired_cluster_ci(new_hits, old_hits, rows, *, replicates=REPLICATES,
                      seed=BOOTSTRAP_SEED):
    """Query-weighted, fixed-three-seed paired delta with RGB cluster resampling."""
    left = np.asarray(new_hits, dtype=np.float64)
    right = np.asarray(old_hits, dtype=np.float64)
    require(left.shape == right.shape == (3, len(rows)), "Paired hit matrix changed")
    per_seed = (left - right).mean(axis=1)
    delta = (left - right).mean(axis=0)
    by_rgb = defaultdict(list)
    for index, row in enumerate(rows):
        by_rgb[row["pixel_sha256"]].append(index)
    groups = list(by_rgb.values())
    clusters = len(groups)
    rng = np.random.default_rng(seed)
    draw = rng.multinomial(clusters, np.full(clusters, 1.0 / clusters),
                           size=replicates)
    numer = np.asarray([delta[indices].sum() for indices in groups], dtype=np.float64)
    denom = np.asarray([len(indices) for indices in groups], dtype=np.float64)
    sampled = (draw @ numer) / (draw @ denom)
    return {"delta": float(delta.mean()), "per_seed_delta": per_seed.tolist(),
            "ci95": np.quantile(sampled, [0.025, 0.975]).astype(float).tolist(),
            "clusters": clusters, "replicates": replicates, "seed": seed}


def summarize_cost(rows, new_cases):
    measured = [new_cases[r["sample_id"]]["measured"] for r in rows]
    elapsed = np.asarray([r["elapsed_seconds"] for r in measured], dtype=float)
    calls = np.asarray([r["counter"]["backbone_forward_calls"] for r in measured], dtype=float)
    pixels = np.asarray([r["counter"]["processed_pixels"] for r in measured], dtype=float)
    allocated = np.asarray([max(r["peak_allocated_bytes"]) for r in measured], dtype=float)
    reserved = np.asarray([max(r["peak_reserved_bytes"]) for r in measured], dtype=float)
    return {"mean_seconds": float(elapsed.mean()), "median_seconds": float(np.median(elapsed)),
            "p95_seconds": float(np.quantile(elapsed, 0.95)),
            "mean_backbone_forward_calls": float(calls.mean()),
            "mean_processed_pixels": float(pixels.mean()),
            "peak_allocated_bytes_max": int(allocated.max()),
            "peak_reserved_bytes_max": int(reserved.max()),
            "timing_scope": "local one_call including PNG decode; three scorer seeds share one backbone extraction"}


def analyze_suite(suite, rows, new_cases, old_cases, labels, reg):
    official = load_pointer_scorer(suite, reg) if suite != "ssp" else None
    parity = {"points_checked": 0} if official is not None else None
    new_hits = [[], [], []]
    old_hits = [[], [], []]
    reachable, full_reachable, empty = [], [], 0
    wins = [Counter() for _ in MODEL_NAMES]
    for row in rows:
        sid, size = row["sample_id"], row["image_size"]
        label = labels[sid]
        current, archived = new_cases[sid], old_cases[sid]
        k16 = any(score_point(p, label, size, official, parity)
                  for p in current["k16_points"])
        pool = any(score_point(p, label, size, official, parity)
                   for p in current["candidate_points"])
        reachable.append(k16)
        full_reachable.append(pool)
        empty += (current["k16_count"] == 0)
        for i, name in enumerate(MODEL_NAMES):
            a = score_point(current["seed_points"][i], label, size, official, parity)
            b = score_point(archived["methods"][name]["point"], label, size,
                            official, parity)
            require(not a or k16, "Correct new seed was outside its K16 oracle: " + sid)
            new_hits[i].append(a)
            old_hits[i].append(b)
            wins[i]["new_only" if a and not b else
                    "old_only" if b and not a else
                    "both" if a else "neither"] += 1
    n = len(rows)
    new_counts = [int(sum(v)) for v in new_hits]
    old_counts = [int(sum(v)) for v in old_hits]
    require(old_counts == ARCHIVED_262K_HITS[suite],
            "Archived V2 aligned result did not reproduce under frozen scorer: " + suite)
    oracle = int(sum(reachable))
    pool_oracle = int(sum(full_reachable))
    paired = paired_cluster_ci(new_hits, old_hits, rows)
    return {
        "n": n, "pixel_cap_per_view": BUDGET,
        "seed_names": list(MODEL_NAMES),
        "new_hits_by_seed": new_counts,
        "new_accuracy_by_seed": [count / n for count in new_counts],
        "new_mean_accuracy": sum(new_counts) / (3 * n),
        "archived_262k_hits_by_seed": old_counts,
        "archived_262k_accuracy_by_seed": [count / n for count in old_counts],
        "archived_262k_mean_accuracy": sum(old_counts) / (3 * n),
        "paired_1m_minus_262k": paired,
        "paired_outcomes_by_seed": [dict(w) for w in wins],
        "new_k16_oracle_count": oracle,
        "new_k16_coverage": oracle / n,
        "new_full_candidate_oracle_count": pool_oracle,
        "new_full_candidate_coverage": pool_oracle / n,
        "new_empty_k16_count": empty,
        "new_reachable_selection_by_seed":
            [count / oracle if oracle else None for count in new_counts],
        "new_reachable_selection_seed_mean":
            sum(new_counts) / (3 * oracle) if oracle else None,
        "pointer_official_v2_parity": parity,
        "cost": summarize_cost(rows, new_cases),
    }, new_hits, old_hits


def ssp_history_strata(rows, history, new_hits, old_hits):
    by_id = {r["sample_id"]: r["history_status"] for r in history["rows"]}
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[by_id[row["sample_id"]]].append(index)
    result = {}
    for name, indices in sorted(groups.items()):
        new = [sum(hit[i] for i in indices) for hit in new_hits]
        old = [sum(hit[i] for i in indices) for hit in old_hits]
        result[name] = {"n": len(indices), "new_hits_by_seed": new,
                        "old_hits_by_seed": old,
                        "new_mean_accuracy": sum(new) / (3 * len(indices)),
                        "old_mean_accuracy": sum(old) / (3 * len(indices))}
    return result


def render_readout(result):
    lines = ["1M pixel-cap full-aligned sensitivity: registered retrospective analysis",
             "All 2,581 public predictions and archived same-case 262k predictions passed SHA gate before labels opened.",
             "Same frozen full aligned S1-S3, original K16, forced Top1. Three seeds are fixed; no selection.", ""]
    for suite in SUITES:
        r = result["suites"][suite]
        d = r["paired_1m_minus_262k"]
        lines.extend([
            f"{suite}: n={r['n']}; new hits={r['new_hits_by_seed']}; 1M mean={100*r['new_mean_accuracy']:.2f}%",
            f"  262k hits={r['archived_262k_hits_by_seed']}; mean={100*r['archived_262k_mean_accuracy']:.2f}%",
            f"  paired 1M - 262k={100*d['delta']:+.2f} pp; image-cluster 95% CI [{100*d['ci95'][0]:+.2f}, {100*d['ci95'][1]:+.2f}] pp; clusters={d['clusters']}",
            f"  1M K16 oracle={r['new_k16_oracle_count']}/{r['n']} ({100*r['new_k16_coverage']:.2f}%); seed-mean selection|reachable={100*r['new_reachable_selection_seed_mean']:.2f}%",
            f"  1M full candidate oracle={r['new_full_candidate_oracle_count']}/{r['n']}; empty K16={r['new_empty_k16_count']}",
            f"  1M latency mean/median={r['cost']['mean_seconds']:.3f}/{r['cost']['median_seconds']:.3f} s; mean forwards={r['cost']['mean_backbone_forward_calls']:.2f}; mean processed pixels={r['cost']['mean_processed_pixels']:.0f}",
        ])
        if suite != "ssp":
            lines.append(f"  Pointer official/V2 scorer parity: {r['pointer_official_v2_parity']['points_checked']} point evaluations, 0 mismatches")
        lines.append("")
    lines.extend([
        "SSP1581 is historically exposed and descriptive only. See summary.json for six exposure strata.",
        "PointerBench V2 labels had already been analyzed; at least seven answers were previewed before prediction. This is retrospective, not strictly blind.",
        "SSP uses the V2 point_in_geometry convention; no separate pinned official scorer was available.",
        "The pixel cap can change both candidate coverage and ranking. This estimates a system-level sensitivity, not a fixed-candidate effect.",
        "Local timing includes PNG decode and is not an equal-total-compute or MVP latency comparison.",
    ])
    return "\n".join(lines) + "\n"


def analyze():
    reg, rows, new_cases, old_cases, history = verify_gate()
    labels = load_labels_after_gate(reg, rows)
    result = {"complete": True, "registration_sha256": sha(REG),
              "global_prediction_seal_sha256": sha(GLOBAL_SEAL),
              "label_sha256": reg["expected_label_sha256_unopened"],
              "method": "frozen full aligned S1-S3, original K16, forced Top1",
              "suites": {}}
    for suite in SUITES:
        summary, new_hits, old_hits = analyze_suite(
            suite, rows[suite], new_cases[suite], old_cases[suite], labels[suite], reg)
        result["suites"][suite] = summary
        if suite == "ssp":
            result["ssp_historical_exposure_strata"] = ssp_history_strata(
                rows[suite], history, new_hits, old_hits)
    write_once(OUT / "summary.json", result)
    write_once(OUT / "readout.txt", render_readout(result), text=True)
    receipt = {"complete": True, "registration_sha256": sha(REG),
               "global_prediction_seal_sha256": sha(GLOBAL_SEAL),
               "summary_sha256": sha(OUT / "summary.json"),
               "readout_sha256": sha(OUT / "readout.txt")}
    write_once(OUT / "completion.json", receipt)
    print(render_readout(result), end="")


def selftest():
    """Synthetic geometry/cluster/checkpoint tests. Never touch a GT path."""
    geometry = {"type": "bbox", "boxes": [[0.1, 0.2, 0.3, 0.4]]}
    label = {"target_geometry": geometry, "bbox_pixels": [10, 20, 30, 40]}
    def native(point, bbox):
        return bbox[0] <= point[0] <= bbox[2] and bbox[1] <= point[1] <= bbox[3]
    checks = {"points_checked": 0}
    assert score_point([10, 20], label, (100, 100), native, checks)
    assert score_point([30, 40], label, (100, 100), native, checks)
    assert not score_point([9, 20], label, (100, 100), native, checks)
    assert not score_point(None, label, (100, 100), native, checks)
    assert checks["points_checked"] == 4
    rows = [{"pixel_sha256": "A"}, {"pixel_sha256": "A"},
            {"pixel_sha256": "B"}, {"pixel_sha256": "C"}]
    a = [[1, 0, 1, 0], [1, 0, 1, 0], [1, 0, 1, 0]]
    b = [[0, 1, 1, 0], [0, 1, 1, 0], [0, 1, 1, 0]]
    paired = paired_cluster_ci(a, b, rows, replicates=32, seed=17)
    assert paired["clusters"] == 3 and paired["delta"] == 0 and len(paired["ci95"]) == 2
    public = {"sample_id": "synthetic", "suite": "pro", "image_size": [100, 100],
              "image_sha256": "image"}
    models = {name: str(i) for i, name in enumerate(MODEL_NAMES)}
    row = {"complete": True, "labels_opened": False,
           "ground_truth_in_model_input": False, "sample_id": "synthetic",
           "suite": "pro", "public_identity": identity(public),
           "input_image_sha256": "image", "image_size": [100, 100],
           "pixel_cap_per_view": BUDGET, "model_hashes": models,
           "k16_points": [[10, 20]], "candidate_points": [[10, 20], [50, 50]],
           "seed_points": [[10, 20]] * 3, "point": [10, 20],
           "k16_count": 1, "candidate_count": 2,
           "measured": {"elapsed_seconds": 1.0, "counter": {}}}
    verify_new_case(row, public, models)
    row["seed_points"] = [[50, 50]] * 3
    try:
        verify_new_case(row, public, models)
    except ValueError:
        pass
    else:
        raise AssertionError("Out-of-K16 synthetic point was accepted")
    print(json.dumps({"phase": "synthetic_selftest_passed", "cpu_only": True,
                      "benchmark_labels_opened": False}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True,
                        choices=("selftest", "register", "verify-gate", "analyze"))
    args = parser.parse_args()
    if args.stage == "selftest":
        selftest()
    elif args.stage == "register":
        register()
    elif args.stage == "verify-gate":
        verify_gate()
        print(json.dumps({"phase": "all_2581_public_predictions_verified",
                          "benchmark_labels_opened": False,
                          "global_prediction_seal_sha256": sha(GLOBAL_SEAL)}))
    else:
        analyze()
