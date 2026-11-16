"""Validate the complete formal action run, then write post-hoc diagnostics/figures.

Defaults deliberately use runs/action_v3/execution, never the abandoned episode
log at the run root. This script refuses formal figures before all 720 unique
primary episodes, explicit expected-case coverage, and a complete run manifest
agree. It never fits a gate/localizer, changes a threshold, reruns a model, or
repairs an action. Frozen-localizer test-map predictions are attribution only.

Run only after timed inference ends:
    .venv/python.exe scripts/summarize_actions.py
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
METHODS = ("prefill_hidden", "post_hidden", "text_decision_shared", "text_coordinates")
LABELS = ("click", "disabled", "absent")
EXPECTED_CASES = 180
EXPECTED_PRIMARY_EPISODES = 720
METHOD_NAMES = {"prefill_hidden": "Pre hidden", "post_hidden": "Post hidden",
                "text_decision_shared": "Word + shared point", "text_coordinates": "Text coordinates"}
COLORS = ("#2A6FBB", "#E38D26", "#34966C", "#CA5363")


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError("Formal action analysis refused: " + message)


def close(actual: float | None, expected: float | None, message: str) -> None:
    require((actual is None and expected is None)
            or (actual is not None and expected is not None and math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-6)), message)


def rate(count: int, total: int) -> dict[str, Any]:
    return {"count": int(count), "total": int(total), "rate": count / total if total else None}


def distribution(values: list[float]) -> dict[str, Any]:
    value = np.asarray(values, dtype=float)
    return {"n": len(values), "p50_ms": float(np.median(value)) if len(value) else None,
            "p95_ms": float(np.percentile(value, 95)) if len(value) else None,
            "mean_ms": float(value.mean()) if len(value) else None}


def validate_formal_inputs(run_root: Path, manifest_path: Path) -> tuple[dict, list[dict], dict[str, dict], dict]:
    """Coverage and identity guards run before writing any formal artifact."""
    from clickagent.action_evaluation import _normalize
    from clickagent.dataset import load_manifest

    execution = run_root / "execution"
    paths = {"episodes": execution / "episodes.jsonl", "metrics": run_root / "evaluation" / "metrics.json",
             "run": execution / "action_run.json", "manifest": manifest_path}
    require(all(path.is_file() for path in paths.values()), "formal execution/metrics/run/manifest files must all exist")
    failed = execution / "failed_attempts.jsonl"
    require(not failed.exists() or not failed.read_text(encoding="utf-8").strip(), "the formal execution has retained failed attempts")
    manifest = [row for row in load_manifest(manifest_path, validate_images=False) if row["split"] == "test"]
    require(len(manifest) == EXPECTED_CASES, "new test manifest must contain exactly 180 cases")
    facts = {row["sample_id"]: row for row in manifest}
    require(len(facts) == EXPECTED_CASES, "test sample IDs must be unique")
    require(all(row["metadata"].get("seed") == 144 and row.get("source") == "action_v3_fresh_test"
                for row in manifest), "all test cases must be fresh seed 144, not old/development cases")
    require(all(row["instruction"] == "点击保存" for row in manifest), "test instruction is not constant")
    groups = defaultdict(list)
    for row in manifest:
        groups[row["group_id"]].append(row)
    require(len(groups) == 60 and all(len(rows) == 3 and {row["label"] for row in rows} == set(LABELS)
                                    for rows in groups.values()), "expected 60 complete matched triplets")
    require(Counter(row["label"] for row in manifest) == Counter({label: 60 for label in LABELS}), "test labels are not 60/60/60")
    require(Counter(rows[0]["metadata"]["domain"] for rows in groups.values()) == Counter({"iid": 30, "ood": 30}),
            "test domains are not 30 IID and 30 OOD groups")
    run = read_json(paths["run"])
    require(run.get("status") == "complete" and run.get("episodes") == EXPECTED_PRIMARY_EPISODES,
            "execution/action_run.json is not a complete 720-episode run")
    require(run.get("fresh_inference_for_each_episode") is True, "fresh inference certification is missing")
    identity = run.get("identity", {})
    require(set(identity.get("methods", [])) == set(METHODS), "formal methods differ from the four prespecified methods")
    require(identity.get("manifest_sha256") == sha256(manifest_path), "run identity does not match the input manifest")
    require(len(identity.get("test_sample_ids", [])) == EXPECTED_CASES
            and set(identity.get("test_sample_ids", [])) == set(facts), "run identity does not contain the full expected test universe")
    raw = read_jsonl(paths["episodes"])
    require(all(isinstance(row.get("warmup"), bool) and row.get("method") in METHODS for row in raw),
            "execution log has an unclassified/wrong-method row")
    primary = [row for row in raw if row.get("warmup") is False]
    require(len(primary) == EXPECTED_PRIMARY_EPISODES, "execution log must contain exactly 720 non-warmup records")
    keys = [(row["method"], row["sample_id"]) for row in primary]
    expected_keys = {(method, sample_id) for method in METHODS for sample_id in facts}
    require(len(set(keys)) == EXPECTED_PRIMARY_EPISODES and set(keys) == expected_keys,
            "episode keys contain a missing, duplicate, unexpected, or development record")
    normalized = []
    for index, row in enumerate(primary):
        item = _normalize(dict(row), index, METHODS)
        fact = facts[item["sample_id"]]
        require(all(item[key] == fact[key] for key in ("sample_id", "group_id", "label")), "episode facts differ from manifest")
        require(item["domain"] == fact["metadata"]["domain"], "episode domain differs from manifest")
        require(item["latency_valid"], "all 720 cases must retain actual measured latency, including failures/waits")
        require(item["environment_evidence_known"] and item["pixel_changed"] is not None,
                "an episode has missing environment/pixel evidence")
        normalized.append(item)
    metrics = read_json(paths["metrics"])
    coverage = metrics.get("coverage", {})
    require(metrics.get("experiment_type") == "fresh_browser_action_execution" and metrics.get("seed") == 144,
            "evaluation type/seed does not match the formal action run")
    require(coverage.get("universe_source") == "explicit expected_cases", "evaluation lacks explicit expected-case coverage")
    require(coverage.get("n_expected_cases") == 180 and coverage.get("n_expected_groups") == 60
            and coverage.get("n_complete_expected_triplets") == 60 and not coverage.get("incomplete_expected_group_ids")
            and not coverage.get("duplicates"), "evaluation universe/group/duplicate coverage is incomplete")
    require(coverage.get("expected_label_counts") == {label: 60 for label in LABELS}, "metrics label universe differs")
    require(set(metrics.get("methods", {})) == set(METHODS), "metrics methods differ")
    for method in METHODS:
        method_coverage = coverage.get("methods", {}).get(method, {})
        require(method_coverage.get("complete") is True and method_coverage.get("coverage") == 1
                and method_coverage.get("n_unique_scored") == 180 and method_coverage.get("n_raw_primary_rows") == 180
                and method_coverage.get("n_expected") == 180 and not method_coverage.get("missing_sample_ids")
                and not method_coverage.get("ambiguous_duplicate_sample_ids"), f"{method}: expected coverage is not complete")
        require(metrics["methods"][method].get("status") == "complete", f"{method}: metrics are partial")
    require(metrics.get("measurement_audit", {}).get("primary_latency_eligible") == 720
            and not metrics.get("measurement_audit", {}).get("primary_latency_ineligible"), "metrics discard measured primary cases")
    warmups = Counter(row.get("method") for row in raw if row.get("warmup") is True)
    require(all(warmups[method] >= 1 for method in METHODS), "a method has no separate warmup")
    require(metrics.get("warmup", {}).get("n") == sum(warmups.values())
            and metrics.get("warmup", {}).get("by_method") == dict(warmups), "metrics warmup audit differs from execution log")
    provenance = {name: {"path": str(path.resolve()), "sha256": sha256(path)} for name, path in paths.items()}
    provenance["formal_execution_only"] = True
    provenance["development_root_episodes_used"] = False
    provenance["warmup_count_by_method"] = dict(warmups)
    return metrics, normalized, facts, provenance


def method_diagnostics(rows: list[dict], canonical: dict) -> dict:
    enabled = [row for row in rows if row["feasible"]]
    negative = [row for row in rows if not row["feasible"]]
    success = sum(row["save_success"] for row in enabled)
    enabled_wait = sum(not row["emitted_click"] for row in enabled)
    enabled_wrong = sum(row["wrong_target_click"] for row in enabled)
    enabled_unknown = sum(row["unknown_enabled_click_target"] for row in enabled)
    target_hit_failed = sum(row["emitted_click"] and row["target_hit"] is True and not row["save_success"] for row in enabled)
    require(success + enabled_wait + enabled_wrong + enabled_unknown + target_hit_failed == len(enabled), "enabled outcomes do not partition")
    result = {
        "n": len(rows), "n_groups": len({row["group_id"] for row in rows}),
        "enabled_save_success": rate(success, len(enabled)),
        "all_wait_or_reject": rate(sum(not row["emitted_click"] for row in rows), len(rows)),
        "enabled_rejection_loss": rate(enabled_wait, len(enabled)),
        "enabled_wrong_target_click": rate(enabled_wrong, len(enabled)),
        "enabled_unknown_clicked_target": rate(enabled_unknown, len(enabled)),
        "enabled_clicked_target_but_failed_save": rate(target_hit_failed, len(enabled)),
        "negative_click_attempts": rate(sum(row["negative_false_click_attempt"] for row in negative), len(negative)),
        "strict_correct_negative_refusal": rate(sum(row["action_correct"] for row in negative), len(negative)),
        "disabled_click_attempts": rate(sum(row["emitted_click"] for row in rows if row["label"] == "disabled"), 60),
        "absent_click_attempts": rate(sum(row["emitted_click"] for row in rows if row["label"] == "absent"), 60),
        "wrong_side_effect_events": sum(row["wrong_side_effect_events"] or 0 for row in rows),
        "wrong_side_effect_episodes": sum(bool(row["wrong_side_effect_events"]) for row in rows),
        "other_control_activations": sum(row["other_action_delta"] or 0 for row in rows),
        "other_control_attempts": sum(row["other_attempt_delta"] or 0 for row in rows),
        "invalid_parse": rate(sum(row["parse_valid"] is False for row in rows), len(rows)),
        "execution_error_count": sum(row["execution_error"] for row in rows),
        "actual_total_latency": distribution([row["timing_ms"]["total"] for row in rows]),
        "enabled_outcome_counts": {"saved": success, "wait_or_reject": enabled_wait, "wrong_target": enabled_wrong,
                                   "unknown_target": enabled_unknown, "target_hit_but_save_failed": target_hit_failed},
        "canonical_metrics": canonical["metrics"], "group_bootstrap": canonical["group_bootstrap"],
        "latency_subsets_from_metrics": canonical["latency"], "by_domain_from_metrics": canonical["by_domain"],
    }
    counts = canonical["metrics"]
    checks = {"feasible_save_success": success, "abstention": result["all_wait_or_reject"]["count"],
              "feasible_abstention": enabled_wait, "negative_false_click_attempt": result["negative_click_attempts"]["count"],
              "negative_false_click_attempt_disabled": result["disabled_click_attempts"]["count"],
              "negative_false_click_attempt_absent": result["absent_click_attempts"]["count"]}
    require(all(counts[key]["numerator"] == value for key, value in checks.items()), "metrics outcomes differ from episode log")
    require(counts["wrong_target_clicks_on_feasible"] == enabled_wrong
            and counts["actual_wrong_side_effect_event_count"] == result["wrong_side_effect_events"], "metrics attribution differs")
    source_latency = canonical["latency"]["all_cases"]
    require(source_latency["n_measured_eligible"] == 180 and source_latency["n_excluded"] == 0, "method latency coverage differs")
    for key in ("p50", "p95", "mean"):
        close(source_latency["total_ms"][key], result["actual_total_latency"][key + "_ms"], "metrics latency differs from actual episode total")
    return result


def localization_diagnostics(run_root: Path, facts: dict[str, dict], run: dict) -> dict:
    """Post-hoc frozen predictions only: never alter logged actions or parameters."""
    from clickagent.action_grounding import AttentionPointGrounder, point_hits_box

    model_path = run_root / "grounder" / "grounder.joblib"
    require(model_path.is_file(), "frozen shared grounder is missing")
    require(sha256(model_path) == run["identity"].get("grounder_sha256"), "diagnostic grounder differs from the executed frozen model")
    grounder = AttentionPointGrounder.load(model_path)
    root = run_root / "execution" / "test_maps"
    require(all((root / (sample_id + ".npz")).is_file() for sample_id in facts), "formal execution test-map coverage is incomplete")
    entries = {}
    for sample_id, sample in facts.items():
        path = root / (sample_id + ".npz")
        with np.load(path, allow_pickle=False) as stored:
            raw, grid, size = (stored[key].copy() for key in ("raw_attention", "grid_shape", "image_size"))
        prediction = grounder.predict_point(raw, grid, size)
        hit = point_hits_box(prediction["point"], size, sample["target_bbox"]) if sample["target_bbox"] is not None else None
        entries[sample_id] = {"label": sample["label"], "domain": sample["metadata"]["domain"],
                              "point": prediction["point"], "score": prediction["score"], "target_box_hit": hit,
                              "map_path": str(path.resolve()), "map_sha256": sha256(path)}
    enabled = [row for row in entries.values() if row["label"] == "click"]
    present = [row for row in entries.values() if row["label"] != "absent"]
    return {"post_hoc_attribution_only": True, "model_retrained": False, "test_threshold_selection": False,
            "grounder_path": str(model_path.resolve()), "grounder_sha256": sha256(model_path), "n_maps": len(entries),
            "enabled_shared_point_hit_ceiling": rate(sum(bool(row["target_box_hit"]) for row in enabled), len(enabled)),
            "all_present_target_point_hit": rate(sum(bool(row["target_box_hit"]) for row in present), len(present)),
            "enabled_hit_by_domain": {domain: rate(sum(bool(row["target_box_hit"]) for row in enabled if row["domain"] == domain),
                                                   sum(row["domain"] == domain for row in enabled)) for domain in ("iid", "ood")},
            "interpretation": ("Frozen point predictions on already-recorded prefill maps identify a localization ceiling, "
                               "not executed counterfactual successes. No logged action or latency is changed."),
            "samples": entries}


def post_gate_word_diagnostics(raw_rows: list[dict], normalized: list[dict], localization: dict | None) -> dict:
    raw = {(row["method"], row["sample_id"]): row for row in raw_rows if row.get("warmup") is False}
    observed = {(row["method"], row["sample_id"]): row for row in normalized}
    post = [row for row in normalized if row["method"] == "post_hidden"]
    result = {"condition": "post_hidden's own freely generated word is CLICK; not a teacher-forced label",
              "negative_rejection_by_state": {}, "enabled_rejection_loss": {}}
    for label in LABELS:
        matching = [row for row in post if row["label"] == label
                    and raw[("post_hidden", row["sample_id"])].get("model", {}).get("generated_decision") == "click"]
        rejected = [row for row in matching if not row["emitted_click"]]
        word_clicked = [row for row in rejected if observed[("text_decision_shared", row["sample_id"])]["emitted_click"]]
        item = {"own_word_click_count": len(matching), "post_gate_rejected": rate(len(rejected), len(matching)),
                "rejected_sample_ids": [row["sample_id"] for row in rejected],
                "paired_word_policy_did_click_on_rejected_cases": len(word_clicked),
                "paired_word_policy_click_sample_ids": [row["sample_id"] for row in word_clicked]}
        if label == "click":
            lost = [row for row in rejected if observed[("text_decision_shared", row["sample_id"])]["save_success"]]
            item["paired_word_policy_actually_saved_but_post_rejected"] = len(lost)
            item["paired_word_success_lost_sample_ids"] = [row["sample_id"] for row in lost]
            if localization is not None:
                candidates = localization["samples"]
                item["post_rejections_with_frozen_point_box_hit"] = sum(bool(candidates[row["sample_id"]]["target_box_hit"]) for row in rejected)
            result["enabled_rejection_loss"] = item
        else:
            result["negative_rejection_by_state"][label] = item
    risky = [row for row in post if row["label"] != "click" and row["emitted_click"]
             and raw[("post_hidden", row["sample_id"])].get("model", {}).get("generated_decision") != "click"]
    result["negative_gate_clicks_despite_own_nonclick_or_unrecognized_word"] = {"count": len(risky), "sample_ids": [row["sample_id"] for row in risky]}
    enabled_overrides = [row for row in post if row["label"] == "click" and row["emitted_click"]
                         and raw[("post_hidden", row["sample_id"])].get("model", {}).get("generated_decision") != "click"]
    result["enabled_gate_clicks_despite_own_nonclick_or_unrecognized_word"] = {
        "count": len(enabled_overrides), "actual_save_success_count": sum(row["save_success"] for row in enabled_overrides),
        "paired_word_policy_wait_count": sum(not observed[("text_decision_shared", row["sample_id"])]["emitted_click"] for row in enabled_overrides),
        "sample_ids": [row["sample_id"] for row in enabled_overrides]}
    result["interpretation"] = ("Rejecting a model's CLICK word may correct false releases but can also lose enabled tasks. "
                                 "Word-policy comparisons use separately executed matched episodes; post-hoc point hits are not actual saves.")
    return result


def validate_paired_metrics(metrics: dict, rows: list[dict]) -> list[dict]:
    index = {(row["method"], row["sample_id"]): row for row in rows}
    pairs = metrics.get("latency_comparisons", [])
    require(len(pairs) == 6, "expected all six prespecified method pairs")
    require(len({tuple(sorted((row["method_a"], row["method_b"]))) for row in pairs}) == 6,
            "paired comparisons contain duplicates")
    for pair in pairs:
        a, b = pair["method_a"], pair["method_b"]
        require(a in METHODS and b in METHODS and a != b, "unknown paired comparison")
        all_cases = pair["all_paired_cases"]
        require(all_cases.get("n") == 180, "all-case paired latency coverage is incomplete")
        successful = {sample_id for method, sample_id in index if method == a
                      and index[(a, sample_id)]["save_success"] and index[(b, sample_id)]["save_success"]}
        value = pair["successful_feasible_intersection"]
        require(value.get("n") == len(successful) and set(value.get("sample_ids", [])) == successful,
                "successful-pair intersection differs from actual enabled-save outcomes")
    priorities = {("prefill_hidden", "text_coordinates"): 0, ("post_hidden", "text_decision_shared"): 1,
                  ("prefill_hidden", "text_decision_shared"): 2}
    require(all(any((pair["method_a"], pair["method_b"]) == key for pair in pairs) for key in priorities),
            "a primary interface/coordinate comparison is missing")
    return sorted(pairs, key=lambda row: priorities.get((row["method_a"], row["method_b"]), 3))


def _save_figure(fig: Any, output: Path, name: str) -> None:
    fig.savefig(output / (name + ".png"), dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(output / (name + ".pdf"), bbox_inches="tight", facecolor="white")


def make_figures(diagnostics: dict, output: Path) -> None:
    """Scientific static plots: no fitting, recomputed CIs, or hidden case selection."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    methods = diagnostics["methods"]
    y = np.arange(4)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.9), gridspec_kw={"width_ratios": [1.05, 1.05, 1.2]})
    for axis, key, ci_key, title in ((axes[0], "enabled_save_success", "feasible_save_success", "A  Enabled saves completed"),
                                    (axes[1], "negative_click_attempts", "negative_false_click_attempt", "B  Disabled/absent click attempts")):
        values = [methods[method][key]["rate"] for method in METHODS]
        axis.barh(y, values, color=COLORS, height=.6)
        for index, method in enumerate(METHODS):
            metric = methods[method][key]
            interval = methods[method]["group_bootstrap"].get("metrics", {}).get(ci_key, {}).get("ci95")
            if interval is not None:
                axis.plot(interval, [index, index], color="#20252B", linewidth=1.4)
                axis.plot(interval, [index, index], "|", color="#20252B", markersize=7)
            axis.text(min(metric["rate"] + .025, .91), index - .17,
                      f"{metric['count']}/{metric['total']}", fontsize=9, fontweight="bold")
        axis.set(xlim=(0, 1), yticks=y, title=title)
        axis.xaxis.set_major_formatter(PercentFormatter(1))
        axis.invert_yaxis()
        axis.grid(axis="x", alpha=.16)
        axis.set_axisbelow(True)
    axes[0].set_yticklabels([METHOD_NAMES[method] for method in METHODS])
    axes[1].set_yticklabels([])
    localization = diagnostics.get("localization_attribution")
    if localization:
        ceiling = localization["enabled_shared_point_hit_ceiling"]["rate"]
        axes[0].scatter([ceiling] * 3, y[:3], marker="D", facecolor="none", edgecolor="#252B34", s=48,
                        label="Shared-point box-hit ceiling (post hoc)", zorder=5)
        axes[0].legend(loc="lower left", bbox_to_anchor=(-.03, -.29), fontsize=8, frameon=False)
    medians = [methods[method]["actual_total_latency"]["p50_ms"] / 1000 for method in METHODS]
    tails = [methods[method]["actual_total_latency"]["p95_ms"] / 1000 for method in METHODS]
    for index in range(4):
        axes[2].plot([medians[index], tails[index]], [index, index], color=COLORS[index], linewidth=3, alpha=.65)
    axes[2].scatter(medians, y, c=COLORS, s=55, marker="o", label="p50", zorder=4)
    axes[2].scatter(tails, y, c=COLORS, s=55, marker="|", label="p95", zorder=4)
    axes[2].set(yticks=y, yticklabels=[], title="C  Actual end-to-end total time", xlabel="Seconds (all 180 cases per method)")
    axes[2].invert_yaxis()
    axes[2].set_xlim(left=0)
    axes[2].grid(axis="x", alpha=.16)
    axes[2].legend(frameon=False, loc="upper left", bbox_to_anchor=(0, -.18), ncol=2)
    fig.suptitle("Frozen Qwen3-VL-8B: real single-step browser actions", fontsize=14, y=1.04)
    fig.text(.01, -.10, "720 fresh episodes; 60 matched groups. Rate whiskers: group-bootstrap 95% CI. "
             "Total time includes observe/infer/policy/execute/verify, failures and waits; excludes model loading.\n"
             "Supervision differs: hidden gates and shared point readout use old train/val labels; text coordinates is zero-shot.", fontsize=9)
    fig.tight_layout(w_pad=2.3)
    _save_figure(fig, output, "comparison")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 4.9))
    outcome_names = ("saved", "wait_or_reject", "wrong_target", "unknown_target", "target_hit_but_save_failed")
    outcome_colors = ("#318B68", "#ABB3BE", "#CA5363", "#9675A7", "#E4AC48")
    outcome_labels = ("Saved", "Wait / reject", "Wrong target", "Unknown target", "Target hit; save failed")
    left = np.zeros(4)
    for key, color, label in zip(outcome_names, outcome_colors, outcome_labels):
        counts = [methods[method]["enabled_outcome_counts"][key] for method in METHODS]
        axes[0].barh(y, counts, left=left, color=color, label=label, height=.62)
        left += counts
    axes[0].set(xlim=(0, 60), yticks=y, yticklabels=[METHOD_NAMES[method] for method in METHODS],
                title="Enabled-case outcomes", xlabel="Cases (60 enabled targets)")
    axes[0].invert_yaxis()
    axes[0].legend(loc="upper left", bbox_to_anchor=(0, -.16), frameon=False, ncol=2, fontsize=8)
    for offset, key, color, label in ((-.23, "disabled_click_attempts", "#D48531", "Disabled attempts (/60)"),
                                     (0, "absent_click_attempts", "#B84C5C", "Absent attempts (/60)"),
                                     (.23, "wrong_side_effect_episodes", "#604F7F", "Wrong side-effect episodes (/180)")):
        counts = [methods[method][key]["count"] if isinstance(methods[method][key], dict) else methods[method][key] for method in METHODS]
        axes[1].barh(y + offset, counts, height=.2, color=color, label=label)
    axes[1].set(yticks=y, yticklabels=[], xlabel="Count (denominators shown in legend)", title="False releases versus physical side effects")
    axes[1].invert_yaxis()
    axes[1].grid(axis="x", alpha=.16)
    axes[1].set_axisbelow(True)
    axes[1].legend(loc="upper left", bbox_to_anchor=(0, -.16), frameon=False, fontsize=8)
    fig.text(.01, -.13, "Clicking a disabled/absent target is a false release even when the browser ignores it. "
             "Side effects count observed handler changes, not assumed effects.", fontsize=9)
    fig.tight_layout(w_pad=2)
    _save_figure(fig, output, "outcomes")
    plt.close(fig)

    pairs = diagnostics["paired_latency_from_metrics"]
    fig, axis = plt.subplots(figsize=(12.5, 6.0))
    labels = []
    for index, comparison in enumerate(pairs):
        value = comparison["successful_feasible_intersection"]
        labels.append(f"{METHOD_NAMES[comparison['method_a']]} − {METHOD_NAMES[comparison['method_b']]}\n"
                      f"both completed enabled save: n={value['n']}")
        if value.get("status") == "ok":
            point = value["p50_difference_ms"] / 1000
            interval = value.get("group_bootstrap", {}).get("p50_difference_ms", {}).get("ci95")
            if interval is not None:
                axis.plot(np.asarray(interval) / 1000, [index, index], color="#506785", linewidth=2)
            axis.scatter([point], [index], color="#2A6FBB" if index < 3 else "#8E99A6", s=65, zorder=3)
        else:
            axis.text(0, index, "No common successful saves", va="center", fontsize=9)
    axis.axvline(0, color="#444A52", linestyle="--", linewidth=1)
    axis.set(yticks=np.arange(len(pairs)), yticklabels=labels,
             xlabel="Difference in median actual total time: A − B (seconds); negative favors A",
             title="Paired latency on the intersection of actual successful enabled saves")
    axis.invert_yaxis()
    axis.grid(axis="x", alpha=.16)
    fig.text(.01, -.055, "Points and group-bootstrap 95% CIs are read directly from metrics.json. "
             "First three comparisons are prespecified; success-conditioned subsets do not establish overall speed superiority.", fontsize=9)
    fig.tight_layout()
    _save_figure(fig, output, "paired_success_latency")
    plt.close(fig)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=PROJECT / "runs" / "action_v3")
    parser.add_argument("--manifest", type=Path, default=PROJECT / "data" / "action_v3" / "manifest.jsonl")
    parser.add_argument("--output", type=Path, help="Defaults to RUN_ROOT/analysis")
    parser.add_argument("--skip-localization", action="store_true", help="Skip optional frozen-map attribution; never changes the executed metrics")
    args = parser.parse_args(argv)
    run_root, manifest_path = args.run_root.resolve(), args.manifest.resolve()
    output = args.output.resolve() if args.output else run_root / "analysis"
    metrics, rows, facts, provenance = validate_formal_inputs(run_root, manifest_path)
    by_method = {method: [row for row in rows if row["method"] == method] for method in METHODS}
    methods = {method: method_diagnostics(by_method[method], metrics["methods"][method]) for method in METHODS}
    pairs = validate_paired_metrics(metrics, rows)
    run = read_json(run_root / "execution" / "action_run.json")
    localization = None if args.skip_localization else localization_diagnostics(run_root, facts, run)
    raw = read_jsonl(run_root / "execution" / "episodes.jsonl")
    diagnostics = {
        "schema_version": 1, "formal_complete_run": True, "primary_episodes": 720, "test_cases": 180,
        "test_groups": 60, "test_seed": 144, "provenance": provenance, "methods": methods,
        "analysis_script_sha256": sha256(Path(__file__)),
        "paired_latency_from_metrics": pairs, "localization_attribution": localization,
        "post_gate_vs_own_decision_word": post_gate_word_diagnostics(raw, rows, localization),
        "interpretation_limits": ["All rates describe one controlled single-step browser task, not general desktop-agent success.",
                                  "Supervision budgets differ; hidden gates and the shared localizer use old train/val annotations.",
                                  "Actual total latency includes failed/rejected cases; success-paired results are conditional subsets.",
                                  "Disabled/absent click attempts remain errors even without physical side effects.",
                                  "Frozen localizer predictions on test maps are attribution only; no action or threshold is changed."],
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "diagnostics.json").write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    make_figures(diagnostics, output)
    print(json.dumps({"output": str(output), "formal_primary_episodes": 720,
                      "figures": ["comparison", "outcomes", "paired_success_latency"]}, indent=2))
    return diagnostics


if __name__ == "__main__":
    main()
