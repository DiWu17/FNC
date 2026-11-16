"""Independent source/log audit of the finished controlled-browser action run.

This script does not import ClickAgent, NumPy, Torch, sklearn, or Playwright. It
does not infer, fit, retune, bootstrap, rerun a browser, or hash model shards.
Only audit.json is written. By default it refuses to inspect episodes while the
run manifest is still marked running. Execute after the run owner says it ended.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import struct
import sys
import zipfile


METHODS = ("prefill_hidden", "post_hidden", "text_decision_shared", "text_coordinates")
LABELS = ("click", "disabled", "absent")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_jsonl(path):
    rows, errors = [], []
    if not Path(path).exists():
        return rows, [{"file": str(path), "error": "missing file"}]
    with Path(path).open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("row is not a dictionary")
                rows.append(row)
            except (ValueError, json.JSONDecodeError) as exc:
                errors.append({"file": str(path), "line": line_number, "error": str(exc)})
    return rows, errors


def check(checks, name, failures=None, **detail):
    failures = failures or []
    checks[name] = {"status": "fail" if failures else "pass", "n_failures": len(failures), "failures": failures, **detail}


def finite_nonnegative(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def item_failure(index, row, issue):
    return {"row_index": index, "sample_id": row.get("sample_id"), "method": row.get("method"), "issue": issue}


def _scalar_npz_metadata(path):
    """Read only the metadata.npy scalar using stdlib, never raw feature arrays."""
    with zipfile.ZipFile(path) as archive, archive.open("metadata.npy") as stream:
        if stream.read(6) != b"\x93NUMPY":
            raise ValueError("metadata.npy has no NPY header")
        major, _minor = stream.read(2)
        length = struct.unpack("<H" if major == 1 else "<I", stream.read(2 if major == 1 else 4))[0]
        header = ast.literal_eval(stream.read(length).decode("utf-8" if major >= 3 else "latin-1").strip())
        if header["shape"] != ():
            raise ValueError("metadata must be a scalar")
        dtype = header["descr"]
        payload = stream.read()
        if "U" in dtype:
            endian = "be" if dtype.startswith(">") else "le" if dtype.startswith("<") else ("le" if sys.byteorder == "little" else "be")
            text = payload.decode("utf-32-" + endian).rstrip("\x00")
        elif "S" in dtype:
            text = payload.decode("utf-8").rstrip("\x00")
        else:
            raise ValueError(f"unsupported scalar metadata dtype: {dtype}")
    return json.loads(text)


def audit_sources(run, project_root, checks):
    identity = run.get("identity", {})
    expected = {
        "action_benchmark.py": identity.get("actor_sha256"),
        "action_env.py": identity.get("environment_sha256"),
        **identity.get("feature_identity", {}).get("source_sha256", {}),
    }
    failures, actual = [], {}
    for name, value in expected.items():
        path = project_root / "src/clickagent" / name
        if not path.is_file() or not value:
            failures.append({"source": name, "issue": "source or recorded SHA256 missing"})
            continue
        actual[name] = sha256(path)
        if actual[name] != value:
            failures.append({"source": name, "issue": "source changed since benchmark identity was recorded", "recorded": value, "actual": actual[name]})
    check(checks, "captured_source_hashes", failures, actual_sha256=actual)
    runner_path = project_root / "src/clickagent/action_benchmark.py"
    env_path = project_root / "src/clickagent/action_env.py"
    if not runner_path.is_file() or not env_path.is_file():
        check(checks, "source_policy_boundary", ["runner/environment source unavailable"])
        return
    source, environment = runner_path.read_text(encoding="utf-8"), env_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = {node.name: node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    boundary = []
    episode = functions.get("run_episode")
    calls = [node for node in ast.walk(episode) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "infer_method"] if episode else []
    if len(calls) != 1 or [ast.unparse(arg) for arg in calls[0].args[:3]] != ["runtime", "method", "image"]:
        boundary.append("run_episode must call infer_method(runtime, method, image, ...), not pass sample/GT")
    propose = functions.get("propose_action")
    if propose is None or [arg.arg for arg in propose.args.args] != ["method", "result", "grounder", "gates"]:
        boundary.append("propose_action signature no longer isolates policy from sample/GT/env")
    infer = functions.get("infer_method")
    runtime_calls = [node for node in ast.walk(infer) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in ("infer_readouts", "infer_text_action")] if infer else []
    if len(runtime_calls) != 2 or any(not node.args or ast.unparse(node.args[0]) != "image" for node in runtime_calls):
        boundary.append("runtime inference no longer receives only the screenshot image as its first input")
    episode_text = ast.get_source_segment(source, episode) if episode else ""
    if '"total": (finished - started) * 1000' not in (episode_text or ""):
        boundary.append("cannot verify total_ms is directly recorded from finished-start")
    if "self._page.mouse.click(" not in environment or "trusted:event.isTrusted" not in environment:
        boundary.append("cannot verify real Playwright mouse click and event.isTrusted logging")
    check(checks, "source_policy_boundary", boundary,
          inference_call=ast.unparse(calls[0]) if calls else None,
          scope="source review supports PIL-only policy input; evaluator may inspect GT solely for rendering and scoring")
    guard_failures = []
    if "def checked_episode(" not in source or '"failed_attempts.jsonl"' not in source or "automatic retry" not in source:
        guard_failures.append("failed-attempt ledger/refusal guard missing")
    if "orphan log" not in source:
        guard_failures.append("orphan episode log refusal missing")
    check(checks, "failure_and_resume_guards_in_source", guard_failures)


def audit_files_and_training(run_dir, manifest_path, old_manifest_path, cases, new_rows, old_rows, run, checks, *, image_hashes):
    identity = run.get("identity", {})
    failures = []
    actual_manifest = sha256(manifest_path)
    if identity.get("manifest_sha256") != actual_manifest:
        failures.append("input manifest differs from the captured benchmark SHA256")
    check(checks, "new_manifest_identity", failures, actual_sha256=actual_manifest)
    artifacts = {}
    failures = []
    grounder = run_dir.parent / "grounder/grounder.joblib"
    expected = [("grounder", grounder, identity.get("grounder_sha256"))]
    for stage, info in identity.get("gates", {}).items():
        expected.append((stage + "_gate", Path(info.get("model_path", "")), info.get("model_sha256")))
        if info.get("new_test_used_for_selection") is not False or not math.isclose(info.get("val_false_positive_budget", -1), .01):
            failures.append({"gate": stage, "issue": "gate provenance does not declare frozen old-val 1% selection"})
        source_path = Path(info.get("threshold_source", ""))
        if source_path.is_file():
            metrics = read_json(source_path)
            matches = [row for row in metrics.get("runs", []) if row.get("condition") == stage and row.get("feature_set") == "hidden"]
            gates = [g for row in matches for g in row.get("click_gates", []) if math.isclose(g.get("false_positive_budget", -1), .01)]
            if len(gates) != 1 or gates[0].get("source_split") != "val" or gates[0].get("threshold") != info.get("threshold"):
                failures.append({"gate": stage, "issue": "declared threshold differs from original val calibration"})
        else:
            failures.append({"gate": stage, "issue": "original gate calibration report unavailable"})
    for name, path, recorded in expected:
        if not path.is_file() or not recorded:
            failures.append({"artifact": name, "issue": "artifact or recorded hash missing"})
            continue
        actual = sha256(path)
        artifacts[name] = {"path": str(path), "sha256": actual}
        if actual != recorded:
            failures.append({"artifact": name, "issue": "artifact hash changed"})
    check(checks, "frozen_gate_grounder_artifacts", failures, artifacts=artifacts)
    old_test_ids = {row["sample_id"] for row in old_rows if row.get("split") == "test"}
    old_all_ids = {row["sample_id"] for row in old_rows}
    old_groups = {row["group_id"] for row in old_rows}
    overlap = sorted(set(cases) & old_all_ids)
    group_overlap = sorted({row["group_id"] for row in cases.values()} & old_groups)
    check(checks, "new_test_excludes_all_old_ids_groups", ([{"sample_id_overlap": overlap, "group_id_overlap": group_overlap}] if overlap or group_overlap else []), old_test_cases=len(old_test_ids), old_all_cases=len(old_rows))
    raw_dir = run_dir.parent / "grounding_raw"
    cache_rows, cache_errors = [], []
    allowed = {row["sample_id"] for row in old_rows if row.get("split") in ("train", "val") and row.get("label") == "click"}
    training_cases = {row["sample_id"]: row for row in new_rows if row.get("split") in ("train", "val") and row.get("label") == "click"}
    collection_path = run_dir.parent / "grounding_collection.json"
    collection = read_json(collection_path) if collection_path.is_file() else {}
    if collection.get("feature_identity") != identity.get("feature_identity"):
        cache_errors.append("grounding collection model/config/source identity differs from execution")
    for path in sorted(raw_dir.glob("*.npz")):
        try:
            cache_rows.append(_scalar_npz_metadata(path))
        except Exception as exc:
            cache_errors.append({"file": str(path), "issue": str(exc)})
    cache_ids = [row.get("sample_id") for row in cache_rows]
    for metadata in cache_rows:
        sid = metadata.get("sample_id")
        if metadata.get("feature_fingerprint") != collection.get("feature_fingerprint"):
            cache_errors.append({"sample_id": sid, "issue": "grounding cache model/input fingerprint mismatch"})
        if sid in training_cases:
            image = (manifest_path.parent / training_cases[sid]["image"]).resolve()
            if not image.is_file() or metadata.get("image_sha256") != sha256(image):
                cache_errors.append({"sample_id": sid, "issue": "grounding cache image differs from the declared train/val input"})
    disallowed = sorted({sid for sid in cache_ids if sid not in allowed}, key=str)
    if disallowed:
        cache_errors.append({"issue": "grounding cache contains IDs outside old enabled train/val", "sample_ids": disallowed})
    if len(set(cache_ids)) != len(cache_ids):
        cache_errors.append("duplicate grounding-cache sample IDs")
    if set(cache_ids) != allowed:
        cache_errors.append({"issue": "grounding train/val coverage differs", "missing": sorted(allowed - set(cache_ids)), "extra": sorted(set(cache_ids) - allowed, key=str)})
    check(checks, "grounder_cache_excludes_old_and_new_test", cache_errors, n_cache_records=len(cache_rows), n_expected_old_enabled_train_val=len(allowed))
    if not image_hashes:
        checks["image_hash_holdout"] = {"status": "unverified", "reason": "disabled explicitly; ID/group disjointness does not prove pixel disjointness"}
        return
    new_hashes, old_hashes = defaultdict(list), defaultdict(list)
    hash_errors = []
    for source, records, destination in ((manifest_path, cases.values(), new_hashes), (old_manifest_path, old_rows, old_hashes)):
        for row in records:
            image = (source.parent / row["image"]).resolve()
            if not image.is_file():
                hash_errors.append({"sample_id": row["sample_id"], "issue": "image missing", "path": str(image)})
                continue
            destination[sha256(image)].append(row["sample_id"])
    intersections = [{"sha256": digest, "new_ids": new_hashes[digest], "old_ids": old_hashes[digest]} for digest in sorted(set(new_hashes) & set(old_hashes))]
    if intersections:
        hash_errors.append({"issue": "new test shares exact PNG content with old train/val/test", "overlaps": intersections})
    check(checks, "image_hash_holdout", hash_errors, n_new_hashes=len(new_hashes), n_old_hashes=len(old_hashes),
          duplicate_new_test_hashes={digest: ids for digest, ids in new_hashes.items() if len(ids) > 1})


def audit_rows(rows, cases, old_rows, run, checks, methods, expected_warmups, allowed_warmup_ids):
    primary = [(index, row) for index, row in enumerate(rows) if not row.get("warmup", False)]
    warmup = [(index, row) for index, row in enumerate(rows) if row.get("warmup", False)]
    expected_grid = {(sid, method) for sid in cases for method in methods}
    observed = Counter((row.get("sample_id"), row.get("method")) for _, row in primary)
    missing = sorted(expected_grid - set(observed))
    extra = sorted(set(observed) - expected_grid, key=str)
    duplicates = [{"sample_id": sid, "method": method, "count": count} for (sid, method), count in observed.items() if count != 1]
    failures = []
    if missing or extra or duplicates:
        failures.append({"missing": missing, "extra": extra, "duplicates": duplicates})
    if len(primary) != len(expected_grid):
        failures.append({"primary_rows": len(primary), "expected": len(expected_grid)})
    check(checks, "complete_unique_test_method_grid", failures, n_primary_rows=len(primary), n_expected=len(expected_grid), counts_by_method=dict(Counter(row.get("method") for _, row in primary)))
    warm_fail = []
    if len(warmup) != expected_warmups or Counter(row.get("method") for _, row in warmup) != Counter({method: 1 for method in methods}):
        warm_fail.append("expected exactly one warmup per method")
    if any(row.get("split") != "val" for _, row in warmup):
        warm_fail.append("warmup did not use validation-only cases")
    if any(row.get("sample_id") not in allowed_warmup_ids or row.get("sample_id") in cases for _, row in warmup):
        warm_fail.append("warmup ID is outside declared validation cases or touches the new test")
    check(checks, "validation_only_warmups", warm_fail, n_warmup=len(warmup), cold_start_records=[{"index": index, "sample_id": row.get("sample_id"), "method": row.get("method"), "warmup": row.get("warmup")} for index, row in enumerate(rows) if row.get("cold_start")])
    old_test = {row["sample_id"] for row in old_rows if row.get("split") == "test"}
    facts_fail = []
    for index, row in primary:
        case = cases.get(row.get("sample_id"))
        if case is None:
            continue
        for key in ("label", "group_id", "split"):
            if row.get(key) != case.get(key):
                facts_fail.append(item_failure(index, row, f"{key} differs from expected manifest"))
        expected_domain = case.get("metadata", {}).get("domain", case.get("domain"))
        if row.get("domain") != expected_domain:
            facts_fail.append(item_failure(index, row, "domain differs from expected manifest"))
    if any(row.get("sample_id") in old_test for row in rows):
        facts_fail.append("old test case appears in primary or warmup episode log")
    check(checks, "row_facts_and_old_test_exclusion", facts_fail)
    timing_fail, inference_fail, event_fail, reset_fail, policy_fail = [], [], [], [], []
    if run.get("fresh_inference_for_each_episode") is not True:
        inference_fail.append("run does not declare fresh inference for every episode")
    measured = {"primary": 0, "warmup": 0}
    execution_errors = []
    outcome_counts = {method: Counter() for method in methods}
    for index, row in enumerate(rows):
        timing = row.get("timing_ms", {})
        timing = timing if isinstance(timing, dict) else {}
        stages = [timing.get(key) for key in ("observe", "inference", "policy", "execute", "verify", "total")]
        if not all(finite_nonnegative(value) for value in stages):
            timing_fail.append(item_failure(index, row, "timing stage or actual total missing/nonfinite/negative"))
        elif abs(sum(stages[:-1]) - stages[-1]) > .01:
            timing_fail.append(item_failure(index, row, "stage boundaries do not telescope to actual total"))
        if row.get("latency_measured") is not True or row.get("model_load_included") is not False:
            timing_fail.append(item_failure(index, row, "fresh measured total/model-load exclusion not certified"))
        else:
            measured["warmup" if row.get("warmup") else "primary"] += 1
        model = row.get("model", {})
        model = model if isinstance(model, dict) else {}
        elapsed = model.get("elapsed_seconds")
        if not finite_nonnegative(elapsed) or elapsed == 0:
            inference_fail.append(item_failure(index, row, "per-call model elapsed time absent or zero"))
        elif finite_nonnegative(timing.get("inference")) and timing["inference"] + 2 < elapsed * 1000:
            inference_fail.append(item_failure(index, row, "outer fresh inference duration is shorter than the model's own call"))
        tokens = model.get("generated_token_ids")
        count = row.get("n_generated_tokens")
        if not isinstance(tokens, list) or count != len(tokens) or model.get("generated_tokens") != count:
            inference_fail.append(item_failure(index, row, "token IDs/counts disagree"))
        method = row.get("method")
        expected_stage = {"prefill_hidden": "prefill", "post_hidden": "post_decision", "text_decision_shared": "decision_text"}.get(method)
        if expected_stage is not None and model.get("stage") != expected_stage:
            inference_fail.append(item_failure(index, row, "fresh model readout has wrong stage"))
        if method == "prefill_hidden" and count != 0:
            inference_fail.append(item_failure(index, row, "prefill gate unexpectedly generated answer tokens"))
        if method == "text_coordinates" and model.get("hooks_used") is not False:
            inference_fail.append(item_failure(index, row, "coordinate-text baseline unexpectedly used feature hooks"))
        before, after = row.get("env_before", {}), row.get("env_after", {})
        before = before if isinstance(before, dict) else {}
        after = after if isinstance(after, dict) else {}
        verify = row.get("verification", {})
        verify = verify if isinstance(verify, dict) else {}
        if any(before.get(key) != 0 for key in ("save_count", "other_action_count", "other_attempt_count")) or before.get("attempts") != []:
            reset_fail.append(item_failure(index, row, "environment counters/attempts were not reset"))
        if verify.get("reset_pixel_identity_verified") is not True:
            reset_fail.append(item_failure(index, row, "actual browser observation was not verified pixel-identical to input PNG"))
        changed_count, pixel_changed = verify.get("changed_pixel_count"), verify.get("pixel_changed")
        if not finite_nonnegative(changed_count) or pixel_changed is not (changed_count > 0):
            event_fail.append(item_failure(index, row, "pixel difference boolean/count inconsistent"))
        if row.get("pixel_changed") != pixel_changed or row.get("changed_pixel_count") != changed_count:
            event_fail.append(item_failure(index, row, "top-level pixel evidence differs from verification"))
        if verify.get("ui_state_consistent") is not True:
            event_fail.append(item_failure(index, row, "actual pixels/UI counter feedback inconsistent"))
        if row.get("target_hit") != after.get("target_hit") or row.get("mapped_target") != after.get("mapped_target"):
            event_fail.append(item_failure(index, row, "target mapping differs from actual after-state"))
        attempts = after.get("attempts", [])
        emitted = row.get("action", {}).get("type") == "click"
        error = bool(row.get("execution_error"))
        if error:
            execution_errors.append(item_failure(index, row, row["execution_error"]))
        if not isinstance(attempts, list):
            event_fail.append(item_failure(index, row, "after.attempts not a list"))
            attempts = []
        if emitted and not error and len(attempts) != 1:
            event_fail.append(item_failure(index, row, "non-error click did not dispatch exactly one recorded browser event"))
        if not emitted and attempts:
            event_fail.append(item_failure(index, row, "wait dispatched browser mouse events"))
        for attempt in attempts:
            if attempt.get("trusted") is not True:
                event_fail.append(item_failure(index, row, "mouse event isTrusted is not true"))
            if emitted:
                try:
                    coordinate_mismatch = any(abs(float(attempt.get(axis, math.inf)) - float(row["action"].get(axis, -math.inf))) > 1e-6 for axis in ("x", "y"))
                except (TypeError, ValueError):
                    coordinate_mismatch = True
                if coordinate_mismatch:
                    event_fail.append(item_failure(index, row, "actual mouse coordinate differs from unmodified policy action"))
        if row.get("fallback") is not False:
            policy_fail.append(item_failure(index, row, "fallback was used or not explicitly disabled"))
        action = row.get("action", {})
        if method in ("prefill_hidden", "post_hidden"):
            stage = "prefill" if method == "prefill_hidden" else "post_decision"
            original = run.get("identity", {}).get("gates", {}).get(stage, {})
            if row.get("parse_valid"):
                probability, threshold = row.get("click_probability"), row.get("gate_threshold")
                if threshold != original.get("threshold") or not finite_nonnegative(threshold) or not finite_nonnegative(probability) or emitted != (probability >= threshold):
                    policy_fail.append(item_failure(index, row, "live gate action differs from frozen old-val threshold/probability"))
        if method == "text_decision_shared" and emitted != (model.get("generated_decision") == "click"):
            policy_fail.append(item_failure(index, row, "shared textual decision does not match actual click/wait"))
        point = model.get("point") if method == "text_coordinates" else (row.get("point_readout") or {}).get("point")
        if emitted and (not isinstance(point, list) or len(point) != 2 or any(action.get(axis) != point[j] for j, axis in enumerate(("x", "y")))):
            policy_fail.append(item_failure(index, row, "click coordinate was not the actual unmodified model/learned-grounder point"))
        if row.get("warmup") or method not in outcome_counts:
            continue
        c = outcome_counts[method]
        c["n"] += 1
        feasible = row.get("label") == "click"
        c["feasible"] += int(feasible)
        c["negative"] += int(not feasible)
        c["emitted"] += int(emitted)
        save_delta = after.get("save_count", 0) - before.get("save_count", 0)
        other_delta = after.get("other_action_count", 0) - before.get("other_action_count", 0)
        succeeded = feasible and emitted and save_delta == 1 and other_delta == 0 and pixel_changed is True and not error
        behavior = succeeded if feasible else not emitted and save_delta == 0 and other_delta == 0
        c["feasible_success"] += int(succeeded)
        c["action_correct"] += int(behavior and row.get("parse_valid") is True)
        c["behavioral_correct"] += int(behavior)
        c["negative_attempt"] += int(not feasible and emitted)
        c["negative_attempt_" + row.get("label", "unknown")] += int(not feasible and emitted)
    check(checks, "actual_observe_to_verify_timing", timing_fail, measured_counts=measured,
          note="clock/source consistency; no old cache elapsed time or stage sum is used as the primary recorded total")
    check(checks, "fresh_inference_call_evidence", inference_fail, run_declares_fresh=run.get("fresh_inference_for_each_episode"),
          limitation="Positive per-call timers, outputs and captured source support fresh calls; this is not an independent GPU-kernel attestation")
    check(checks, "fresh_environment_and_pixel_identity", reset_fail)
    check(checks, "trusted_mouse_and_visible_feedback", event_fail, recorded_execution_errors=execution_errors,
          environment_scope="real browser mouse events on a controlled canvas; save handler counters/feedback, not OS file saving")
    check(checks, "fixed_policy_no_fallback_no_coordinate_repair", policy_fail)
    seed = run.get("identity", {}).get("seed", 144)
    ordered_ids = run.get("identity", {}).get("test_sample_ids", [])
    rng = random.Random(seed)
    expected_order = []
    for sid in ordered_ids:
        order = list(methods)
        rng.shuffle(order)
        expected_order.extend((sid, method) for method in order)
    actual_order = [(row.get("sample_id"), row.get("method")) for _, row in primary]
    check(checks, "predetermined_random_method_order", [] if actual_order == expected_order else ["primary method order differs from seeded schedule"], seed=seed)
    return {method: dict(counts) for method, counts in outcome_counts.items()}


def audit_action_run(run_dir, manifest_path, old_manifest_path, *, project_root=None, evaluation_path=None,
                     expected_test_cases=180, expected_warmups=4, methods=METHODS, image_hashes=True,
                     allow_incomplete=False):
    run_dir, manifest_path, old_manifest_path = map(Path, (run_dir, manifest_path, old_manifest_path))
    project_root = Path(project_root) if project_root else Path(__file__).resolve().parents[1]
    run_path = run_dir / "action_run.json"
    run = read_json(run_path)
    result = {"audit_version": 1, "status": "pending", "run_dir": str(run_dir.resolve()),
              "scope": "independent consistency audit of captured source, physical browser-event logs, holdout provenance and measured timers; not independent GPU-kernel attestation",
              "environment_scope": "controlled canvas with real mouse events and visible save-handler feedback; not an OS application or file-save benchmark",
              "limitations": ["Logs and source are not cryptographic proof against fabricated records or deleted historical failures.",
                              "Fresh-call evidence is per-row timers/output plus source/provenance, not an independent GPU trace.",
                              "cold_start is a runner tag; actual device coldness is not independently measured.",
                              "Image holdout hashes compare exact PNG file bytes, not differently encoded files with identical decoded pixels.",
                              "Full model weight shards are not hashed by this audit."], "checks": {}}
    if run.get("status") != "complete" and not allow_incomplete:
        result["status"] = "blocked_run_not_complete"
        result["run_status"] = run.get("status")
        result["note"] = "Episode data was not inspected. Wait for explicit run completion before auditing."
        return result
    checks = result["checks"]
    new_rows, manifest_errors = read_jsonl(manifest_path)
    old_rows, old_errors = read_jsonl(old_manifest_path)
    cases = {row["sample_id"]: row for row in new_rows if row.get("split") == "test"}
    failures = [*manifest_errors, *old_errors]
    if len(cases) != expected_test_cases:
        failures.append({"test_cases": len(cases), "expected": expected_test_cases})
    if run.get("benchmark_version", 0) < 2:
        failures.append("benchmark predates the v2 failure-retention/anti-retry protocol")
    groups = defaultdict(list)
    for row in cases.values():
        groups[row["group_id"]].append(row)
    for group, members in groups.items():
        if len(members) != 3 or {row["label"] for row in members} != set(LABELS):
            failures.append({"group_id": group, "issue": "not a complete state triplet"})
    if set(run.get("identity", {}).get("test_sample_ids", [])) != set(cases):
        failures.append("run identity test list does not match the new held-out manifest")
    if tuple(run.get("identity", {}).get("methods", [])) != tuple(methods):
        failures.append("run method list differs from expected four methods")
    check(checks, "new_test_design", failures, n_cases=len(cases), n_groups=len(groups),
          label_counts=dict(Counter(row["label"] for row in cases.values())),
          domain_group_counts=dict(Counter(members[0].get("metadata", {}).get("domain") for members in groups.values())))
    rows, errors = read_jsonl(run_dir / "episodes.jsonl")
    check(checks, "episode_jsonl_integrity", errors, n_rows=len(rows))
    ledger = run_dir / "failed_attempts.jsonl"
    failed, failed_errors = read_jsonl(ledger) if ledger.exists() else ([], [])
    failure_issues = list(failed_errors)
    if failed:
        failure_issues.append({"issue": "failed attempts were retained; run cannot be certified as an unbroken completed benchmark", "attempts": [{key: row.get(key) for key in ("sample_id", "method", "warmup", "error", "status")} for row in failed]})
    check(checks, "no_hidden_or_retried_failed_attempts", failure_issues, failure_ledger_present=ledger.exists(), n_retained_failed_attempts=len(failed),
          scope="Absence of a ledger entry plus the captured abort/refusal source is evidence, not proof that history was never edited")
    runtime = run.get("model_runtime") or {}
    check(checks, "frozen_model_declared", [] if runtime.get("trainable_parameter_count") == 0 and runtime.get("all_parameters_frozen") is True else ["runtime metadata does not certify zero trainable VLM parameters"],
          trainable_parameter_count=runtime.get("trainable_parameter_count"), load_seconds=run.get("load_seconds"))
    check(checks, "isolated_browser", [] if run.get("browser", {}).get("existing_user_session") is False and run.get("browser", {}).get("headless") is True else ["isolation/headless browser declaration missing"], browser=run.get("browser"))
    audit_sources(run, project_root, checks)
    audit_files_and_training(run_dir, manifest_path, old_manifest_path, cases, new_rows, old_rows, run, checks, image_hashes=image_hashes)
    allowed_warmup_ids = {row["sample_id"] for row in new_rows if row.get("split") == "val"}
    counts = audit_rows(rows, cases, old_rows, run, checks, tuple(methods), expected_warmups, allowed_warmup_ids)
    result["independently_recomputed_counts"] = counts
    evaluation_path = Path(evaluation_path) if evaluation_path else run_dir.parent / "evaluation/metrics.json"
    failures = []
    if evaluation_path.is_file():
        evaluation = read_json(evaluation_path)
        for method in methods:
            metrics = evaluation.get("methods", {}).get(method, {}).get("metrics", {})
            observed = counts[method]
            for metric, raw in (("feasible_save_success", "feasible_success"), ("action_correct", "action_correct"), ("behavioral_correct", "behavioral_correct"), ("negative_false_click_attempt", "negative_attempt"), ("negative_false_click_attempt_disabled", "negative_attempt_disabled"), ("negative_false_click_attempt_absent", "negative_attempt_absent")):
                if metrics.get(metric, {}).get("numerator") != observed.get(raw, 0):
                    failures.append({"method": method, "metric": metric, "issue": "evaluation count differs from raw event audit"})
        check(checks, "evaluation_matches_raw_action_counts", failures, evaluation_path=str(evaluation_path))
    else:
        checks["evaluation_matches_raw_action_counts"] = {"status": "unverified", "reason": "evaluation/metrics.json not yet available"}
    result["status"] = "fail" if any(item["status"] == "fail" for item in checks.values()) else "pass_with_evidence_limitations"
    result["n_failed_checks"] = sum(item["status"] == "fail" for item in checks.values())
    result["n_unverified_checks"] = sum(item["status"] == "unverified" for item in checks.values())
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/action_v3/execution"))
    parser.add_argument("--manifest", type=Path, default=Path("data/action_v3/manifest.jsonl"))
    parser.add_argument("--old-manifest", type=Path, default=Path("data/visual_state_v2_matched/manifest.jsonl"))
    parser.add_argument("--evaluation", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-test-cases", type=int, default=180)
    parser.add_argument("--expected-warmups", type=int, default=4)
    parser.add_argument("--methods", nargs="+", default=list(METHODS))
    parser.add_argument("--no-image-hashes", action="store_true")
    parser.add_argument("--allow-incomplete", action="store_true", help="Diagnostic only; default refuses to inspect a running episode log")
    args = parser.parse_args(argv)
    result = audit_action_run(args.run_dir, args.manifest, args.old_manifest, evaluation_path=args.evaluation,
                              expected_test_cases=args.expected_test_cases, expected_warmups=args.expected_warmups,
                              methods=tuple(args.methods), image_hashes=not args.no_image_hashes,
                              allow_incomplete=args.allow_incomplete)
    output = args.output or args.run_dir / "audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"audit": str(output.resolve()), "status": result["status"], "n_failed_checks": result.get("n_failed_checks")}, ensure_ascii=False))
    return 1 if result["status"] in ("fail", "blocked_run_not_complete") else 0


if __name__ == "__main__":
    raise SystemExit(main())
