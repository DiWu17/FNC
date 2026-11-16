"""Leakage-aware, frozen-feature intent probes for the paired GUI experiment.

This module never invokes or updates the VLM. Only numeric attention/hidden
features and (for the language baseline) the instruction enter the estimators.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


LABELS = ("click", "describe", "forbid")
CONDITIONS = ("prefill", "post_decision", "post_response")
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class TaskSpec:
    name: str
    label_order: tuple[str, ...]


def _task_spec(task: str) -> TaskSpec:
    if task == "intent":
        return TaskSpec(task, LABELS)
    if task == "visual_state":
        return TaskSpec(task, ("click", "disabled", "absent"))
    raise ValueError("task must be intent or visual_state")


@dataclass
class SampleData:
    """Labels and partition facts for a genuinely text-only experiment."""

    sample_ids: np.ndarray
    labels: np.ndarray
    groups: np.ndarray
    splits: np.ndarray


@dataclass
class FeatureData:
    condition: str
    attention: np.ndarray
    hidden: np.ndarray
    sample_ids: np.ndarray
    labels: np.ndarray
    groups: np.ndarray
    splits: np.ndarray

    def subset(self, indices: Iterable[int]) -> "FeatureData":
        indices = np.asarray(list(indices), dtype=int)
        return FeatureData(
            self.condition,
            self.attention[indices],
            self.hidden[indices],
            self.sample_ids[indices],
            self.labels[indices],
            self.groups[indices],
            self.splits[indices],
        )


def _normalize_labels(values: np.ndarray, source: str, label_order: tuple[str, ...] = LABELS) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim != 1:
        raise ValueError(f"{source}: labels must be one-dimensional")
    if values.dtype.kind in "iu":
        if not np.isin(values, range(len(label_order))).all():
            raise ValueError(f"{source}: integer labels must follow {label_order}")
        return np.asarray(label_order)[values.astype(int)]
    labels = values.astype(str)
    if not np.isin(labels, label_order).all():
        raise ValueError(f"{source}: unknown labels {sorted(set(labels) - set(label_order))}")
    return labels


def _string_vector(value: np.ndarray, key: str, source: str) -> np.ndarray:
    value = np.asarray(value)
    if value.ndim != 1 or value.dtype.kind not in "US":
        raise ValueError(f"{source}: {key} must be a one-dimensional string array, not object")
    if value.dtype.kind == "S":
        value = np.char.decode(value, "utf-8")
    value = value.astype(str)
    if np.any(value == ""):
        raise ValueError(f"{source}: {key} contains an empty value")
    return value


def load_feature_file(path: Path, label_order: tuple[str, ...] = LABELS) -> FeatureData:
    """Read only declared feature arrays; reject malformed and unsafe NPZ files."""
    path = Path(path)
    required = {"attention", "hidden", "sample_ids", "labels", "groups", "splits"}
    with np.load(path, allow_pickle=False) as archive:
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"{path}: missing fields {sorted(missing)}")
        ids = _string_vector(archive["sample_ids"], "sample_ids", str(path))
        groups = _string_vector(archive["groups"], "groups", str(path))
        splits = _string_vector(archive["splits"], "splits", str(path))
        labels = _normalize_labels(archive["labels"], str(path), label_order)
        features = []
        for key in ("attention", "hidden"):
            value = np.asarray(archive[key])
            if value.ndim != 2 or value.shape[0] != len(ids) or value.shape[1] == 0:
                raise ValueError(f"{path}: {key} must have shape [N, nonzero feature dimension]")
            if value.dtype.kind not in "fiu" or not np.isfinite(value).all():
                raise ValueError(f"{path}: {key} must contain finite numeric values")
            features.append(value.astype(np.float32))
    if len(ids) == 0 or any(len(v) != len(ids) for v in (labels, groups, splits)):
        raise ValueError(f"{path}: empty data or inconsistent row counts")
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: duplicate sample_ids")
    if not np.isin(splits, SPLITS).all():
        raise ValueError(f"{path}: splits must be exactly train, val, test")
    return FeatureData(path.stem, features[0], features[1], ids, labels, groups, splits)


def _count_labels(labels: np.ndarray, label_order: tuple[str, ...] = LABELS) -> dict[str, int]:
    return {label: int(np.sum(labels == label)) for label in label_order}


def _validate_partitions(reference: FeatureData | SampleData, label_order: tuple[str, ...] = LABELS) -> None:
    if len(set(reference.sample_ids)) != len(reference.sample_ids):
        raise ValueError("Duplicate sample_ids")
    if not np.isin(reference.splits, SPLITS).all():
        raise ValueError("splits must be exactly train, val, test")
    group_splits: dict[str, str] = {}
    for group, split in zip(reference.groups, reference.splits):
        if group in group_splits and group_splits[group] != split:
            raise ValueError(f"Group leakage: group {group!r} occurs in multiple splits")
        group_splits[str(group)] = str(split)
    for split in SPLITS:
        mask = reference.splits == split
        if not mask.any():
            raise ValueError(f"No {split} samples available")
        if split in ("train", "val"):
            missing = set(label_order) - set(reference.labels[mask])
            if missing:
                raise ValueError(
                    f"{split} is missing classes {sorted(missing)}; "
                    "cannot perform reliable three-class model/threshold selection"
                )


def _align_conditions(
    datasets: list[FeatureData], min_common_coverage: float, label_order: tuple[str, ...] = LABELS
) -> tuple[list[FeatureData], dict[str, Any]]:
    """Audit complete files before intersection, so missing rows cannot hide leaks."""
    group_splits: dict[str, str] = {}
    sample_facts: dict[str, tuple[str, str, str]] = {}
    for data in datasets:
        for sid, label, group, split in zip(data.sample_ids, data.labels, data.groups, data.splits):
            if group in group_splits and group_splits[group] != split:
                raise ValueError(f"Group leakage: group {group!r} occurs in multiple splits")
            group_splits[str(group)] = str(split)
            fact = (str(label), str(group), str(split))
            if sid in sample_facts and sample_facts[sid] != fact:
                raise ValueError(f"Condition mismatch for sample {sid!r}: label/group/split differs")
            sample_facts[str(sid)] = fact

    common = set(datasets[0].sample_ids)
    for data in datasets[1:]:
        common &= set(data.sample_ids)
    if not common:
        raise ValueError("No common samples across available conditions")
    # Preserve the first condition's ordering; other files may have arbitrary order.
    ordered_ids = [str(sid) for sid in datasets[0].sample_ids if sid in common]
    union_ids = set(sample_facts)
    coverage: dict[str, Any] = {
        "policy": "paired intersection across all available conditions for train, val and test",
        "n_union": len(union_ids),
        "n_common": len(common),
        "common_over_union": len(common) / len(union_ids),
        "available_conditions": [data.condition for data in datasets],
        "unavailable_conditions": [name for name in CONDITIONS if name not in {d.condition for d in datasets}],
        "conditions": {},
    }
    aligned = []
    for data in datasets:
        index = {str(sid): idx for idx, sid in enumerate(data.sample_ids)}
        subset = data.subset(index[sid] for sid in ordered_ids)
        fraction = len(subset.sample_ids) / len(data.sample_ids)
        if fraction < min_common_coverage:
            raise ValueError(
                f"{data.condition}: common coverage {fraction:.3f} is below required "
                f"{min_common_coverage:.3f}"
            )
        details: dict[str, Any] = {
            "n_available": len(data.sample_ids),
            "n_common": len(subset.sample_ids),
            "common_over_available": fraction,
            "excluded_sample_ids": [str(sid) for sid in data.sample_ids if sid not in common],
            "splits": {},
        }
        for split in SPLITS:
            before = data.splits == split
            after = subset.splits == split
            n_before, n_after = int(before.sum()), int(after.sum())
            details["splits"][split] = {
                "n_available": n_before,
                "n_common": n_after,
                "common_over_available": n_after / n_before if n_before else None,
                "available_label_counts": _count_labels(data.labels[before], label_order),
                "common_label_counts": _count_labels(subset.labels[after], label_order),
            }
        coverage["conditions"][data.condition] = details
        aligned.append(subset)

    _validate_partitions(aligned[0], label_order)
    return aligned, coverage


def _wilson_interval(successes: int, total: int) -> list[float] | None:
    if total == 0:
        return None
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def gate_metrics(labels: np.ndarray, click_probability: np.ndarray, threshold: float, label_order: tuple[str, ...] = LABELS) -> dict[str, Any]:
    """A positive gate authorizes a click; coverage is the fraction authorized."""
    truth = np.asarray(labels) == "click"
    # Calibration uses a float64 nextafter cutoff. NumPy 2 weak Python-scalar
    # promotion would otherwise round that cutoff back to float32 when the
    # estimator returns float32 probabilities, accidentally accepting ties.
    accepted = np.asarray(click_probability, dtype=np.float64) >= np.float64(threshold)
    tp = int(np.sum(truth & accepted))
    fp = int(np.sum(~truth & accepted))
    n_click = int(truth.sum())
    n_nonclick = int((~truth).sum())
    n_accepted = int(accepted.sum())
    return {
        "threshold": float(threshold),
        "n": len(truth),
        "n_click": n_click,
        "n_nonclick": n_nonclick,
        "n_accepted": n_accepted,
        "true_clicks": tp,
        "false_clicks": fp,
        "false_click_rate": fp / n_nonclick if n_nonclick else None,
        "false_click_rate_wilson95": _wilson_interval(fp, n_nonclick),
        "precision": tp / n_accepted if n_accepted else None,
        "recall": tp / n_click if n_click else None,
        "coverage": n_accepted / len(truth) if len(truth) else None,
        "conditional_risk": fp / n_accepted if n_accepted else None,
        "false_click_rate_by_negative_state": {
            label: {
                "n": int(np.sum(np.asarray(labels) == label)),
                "false_clicks": int(np.sum((np.asarray(labels) == label) & accepted)),
                "false_click_rate": float(np.mean(accepted[np.asarray(labels) == label])) if np.any(np.asarray(labels) == label) else None,
            }
            for label in label_order if label != "click"
        },
    }


def calibrate_click_threshold(
    labels: np.ndarray, click_probability: np.ndarray, false_positive_budget: float, label_order: tuple[str, ...] = LABELS
) -> dict[str, Any]:
    """Use validation negatives only; this is an empirical constraint, not a guarantee.

    Set the cutoff just above the next disallowed negative probability. Ties are
    rejected together. The caller must pass validation data, never test data.
    """
    if not 0 <= false_positive_budget <= 1:
        raise ValueError("false_positive_budget must be between zero and one")
    labels, click_probability = np.asarray(labels), np.asarray(click_probability, dtype=np.float64)
    negatives = click_probability[labels != "click"]
    if len(negatives) == 0:
        return {"status": "unavailable", "reason": "validation contains no nonclick examples"}
    allowed = int(math.floor(false_positive_budget * len(negatives) + 1e-12))
    if allowed >= len(negatives):
        threshold = 0.0
    else:
        cutoff = np.sort(negatives)[::-1][allowed]
        threshold = float(np.nextafter(cutoff, np.inf))
    return {
        "status": "ok",
        "source_split": "val",
        "false_positive_budget": float(false_positive_budget),
        "threshold": threshold,
        "n_validation_nonclick": len(negatives),
        "empirical_fpr_resolution": 1 / len(negatives),
        "validation": gate_metrics(labels, click_probability, threshold, label_order),
    }


def _classification_metrics(labels: np.ndarray, probabilities: np.ndarray, label_order: tuple[str, ...] = LABELS) -> dict[str, Any]:
    predicted = np.asarray(label_order)[probabilities.argmax(axis=1)]
    click_column = label_order.index("click")
    click_truth = (labels == "click").astype(int)
    has_both = len(np.unique(click_truth)) == 2
    return {
        "n": len(labels),
        "label_order": list(label_order),
        "label_counts": _count_labels(labels, label_order),
        "all_three_classes_present": set(labels) == set(label_order),
        "accuracy": float(accuracy_score(labels, predicted)),
        "macro_f1": float(f1_score(labels, predicted, labels=list(label_order), average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "confusion_matrix": confusion_matrix(labels, predicted, labels=list(label_order)).tolist(),
        "click_auroc": float(roc_auc_score(click_truth, probabilities[:, click_column])) if has_both else None,
        "click_average_precision": float(average_precision_score(click_truth, probabilities[:, click_column])) if has_both else None,
        "click_brier": float(np.mean((probabilities[:, click_column] - click_truth) ** 2)),
        "click_ranking_status": "ok" if has_both else "unavailable: only one binary class present",
    }


def _probabilities(pipeline: Pipeline, features: Any, label_order: tuple[str, ...] = LABELS) -> np.ndarray:
    original = pipeline.predict_proba(features)
    classes = list(pipeline.classes_)
    return original[:, [classes.index(label) for label in label_order]]


def _new_pipeline(feature_set: str, c_value: float, seed: int) -> Pipeline:
    classifier = LogisticRegression(C=c_value, solver="lbfgs", max_iter=3000, random_state=seed)
    if feature_set == "text_only":
        # Character n-grams support Chinese and mixed Chinese/English instructions.
        return Pipeline([
            ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(1, 3), sublinear_tf=True)),
            ("classifier", classifier),
        ])
    return Pipeline([("scale", StandardScaler()), ("classifier", classifier)])


def _interval(point: float | None, draws: np.ndarray) -> dict[str, Any]:
    finite = np.asarray(draws, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return {"point": point, "ci95": np.percentile(finite, [2.5, 97.5]).tolist() if len(finite) else None, "n_valid_bootstrap": len(finite)}


def _group_bootstrap_statistics(
    labels: np.ndarray, probabilities: np.ndarray, groups: np.ndarray,
    gates: list[dict[str, Any]], label_order: tuple[str, ...], seed: int, iterations: int,
) -> dict[str, Any]:
    """Resample whole screenshot/scenario groups, preserving within-group pairs."""
    unique_groups, group_index = np.unique(groups, return_inverse=True)
    n_groups = len(unique_groups)
    base = {"unit": "group_id", "n_groups": n_groups, "iterations": iterations, "seed": seed, "method": "whole-group percentile bootstrap, shared resampling weights for classification and all gates"}
    if iterations == 0 or n_groups < 2:
        return {**base, "status": "disabled" if iterations == 0 else "unavailable", "reason": "bootstrap disabled or fewer than two independent groups"}
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(n_groups, np.full(n_groups, 1 / n_groups), size=iterations).astype(np.float64)
    counts = np.bincount(group_index, minlength=n_groups).astype(float)
    truth = labels == "click"
    score = np.asarray(probabilities[:, label_order.index("click")], dtype=np.float64)
    correct = np.asarray(label_order)[probabilities.argmax(axis=1)] == labels
    total = draws @ counts
    correct_counts = np.bincount(group_index, weights=correct, minlength=n_groups)
    accuracy_draws = (draws @ correct_counts) / total
    # Weighted Mann-Whitney AUC with half credit for score ties. A bootstrap
    # draw gives every row in a resampled group the same multiplicity.
    _, score_index = np.unique(score, return_inverse=True)
    n_scores = int(score_index.max()) + 1
    positives = np.zeros((n_groups, n_scores), dtype=float)
    negatives = np.zeros_like(positives)
    np.add.at(positives, (group_index, score_index), truth.astype(float))
    np.add.at(negatives, (group_index, score_index), (~truth).astype(float))
    boot_pos, boot_neg = draws @ positives, draws @ negatives
    auc_denominator = boot_pos.sum(axis=1) * boot_neg.sum(axis=1)
    auc_numerator = np.sum(boot_pos * (np.cumsum(boot_neg, axis=1) - 0.5 * boot_neg), axis=1)
    auc_draws = np.divide(auc_numerator, auc_denominator, out=np.full(iterations, np.nan), where=auc_denominator > 0)
    point_auc = float(roc_auc_score(truth, score)) if np.unique(truth).size == 2 else None
    result: dict[str, Any] = {**base, "status": "ok", "accuracy": _interval(float(correct.mean()), accuracy_draws), "click_auroc": _interval(point_auc, auc_draws), "click_gates": []}
    for gate in gates:
        if gate["status"] != "ok":
            continue
        accepted = score >= np.float64(gate["threshold"])
        metrics = gate_metrics(labels, score, gate["threshold"], label_order)
        def ratio_draws(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
            top = draws @ np.bincount(group_index, weights=numerator, minlength=n_groups)
            bottom = draws @ np.bincount(group_index, weights=denominator, minlength=n_groups)
            return np.divide(top, bottom, out=np.full(iterations, np.nan), where=bottom > 0)
        entry = {
            "false_positive_budget": gate["false_positive_budget"], "threshold": gate["threshold"],
            "false_click_rate": _interval(metrics["false_click_rate"], ratio_draws(accepted & ~truth, ~truth)),
            "recall": _interval(metrics["recall"], ratio_draws(accepted & truth, truth)),
            "false_click_rate_by_negative_state": {},
        }
        for label in label_order:
            if label == "click":
                continue
            state = labels == label
            entry["false_click_rate_by_negative_state"][label] = _interval(metrics["false_click_rate_by_negative_state"][label]["false_click_rate"], ratio_draws(accepted & state, state))
        result["click_gates"].append(entry)
    return result


def _paired_visual_statistics(
    labels: np.ndarray, probabilities: np.ndarray, groups: np.ndarray,
    gates: list[dict[str, Any]], label_order: tuple[str, ...],
) -> dict[str, Any]:
    expected = ("click", "disabled", "absent")
    if tuple(label_order) != expected:
        return {"status": "unavailable", "reason": "only defined for visual_state"}
    scores, predictions, incomplete, complete_ids = [], [], [], []
    predicted = np.asarray(label_order)[probabilities.argmax(axis=1)]
    click_column = label_order.index("click")
    for group in np.unique(groups):
        rows = np.flatnonzero(groups == group)
        if len(rows) != 3 or any(np.sum(labels[rows] == label) != 1 for label in expected):
            incomplete.append(str(group))
            continue
        ordered = np.asarray([rows[np.flatnonzero(labels[rows] == label)[0]] for label in expected])
        scores.append(np.asarray(probabilities[ordered, click_column], dtype=np.float64))
        predictions.append(predicted[ordered])
        complete_ids.append(str(group))
    if not scores:
        return {"status": "unavailable", "n_total_groups": len(np.unique(groups)), "n_complete_triplets": 0, "incomplete_group_ids": incomplete}
    scores = np.asarray(scores)
    predicted_matrix = np.asarray(predictions)
    disabled_margin, absent_margin = scores[:, 0] - scores[:, 1], scores[:, 0] - scores[:, 2]
    return {
        "status": "ok", "n_total_groups": len(np.unique(groups)), "n_complete_triplets": len(scores),
        "n_incomplete_groups": len(incomplete), "incomplete_group_ids": incomplete,
        "triplet_all_correct_rate": float(np.mean(np.all(predicted_matrix == np.asarray(expected), axis=1))),
        "paired_flip": {
            "enabled_minus_disabled_positive_rate": float(np.mean(disabled_margin > 0)),
            "enabled_minus_absent_positive_rate": float(np.mean(absent_margin > 0)),
            "both_positive_rate": float(np.mean((disabled_margin > 0) & (absent_margin > 0))),
            "enabled_minus_disabled_mean": float(disabled_margin.mean()),
            "enabled_minus_absent_mean": float(absent_margin.mean()),
            "enabled_disabled_tie_rate": float(np.mean(disabled_margin == 0)),
            "enabled_absent_tie_rate": float(np.mean(absent_margin == 0)),
        },
        "click_gates": [
            {"false_positive_budget": gate["false_positive_budget"], "threshold": gate["threshold"],
             "complete_triplet_correct_gate_rate": float(np.mean(np.all((scores >= np.float64(gate["threshold"])) == np.asarray([True, False, False]), axis=1)))}
            for gate in gates if gate["status"] == "ok"
        ],
        "complete_group_ids": complete_ids,
    }


def _metadata_value(record: dict[str, Any], key: str) -> Any:
    nested = record.get("metadata", {})
    return nested.get(key, record.get(key)) if isinstance(nested, dict) else record.get(key)


def _test_slice_statistics(
    reference: FeatureData | SampleData, probabilities: np.ndarray, gates: list[dict[str, Any]],
    label_order: tuple[str, ...], metadata: dict[str, dict[str, Any]] | None,
    seed: int, bootstrap_iterations: int,
) -> dict[str, Any]:
    if not metadata:
        return {"status": "unavailable", "reason": "manifest metadata not supplied"}
    test = reference.splits == "test"
    result: dict[str, Any] = {"status": "ok", "feature_use": "metadata defines test slices only; never passed to estimators", "axes": {}}
    click_column = label_order.index("click")
    for axis in ("domain", "layout", "style_family"):
        values = np.asarray([str(_metadata_value(metadata.get(str(sid), {}), axis) or "") for sid in reference.sample_ids])
        result[f"n_test_without_{axis}"] = int(np.sum(test & (values == "")))
        slices = {}
        for value in sorted(set(values[test]) - {""}):
            mask = test & (values == value)
            labels, probs, groups = reference.labels[mask], probabilities[mask], reference.groups[mask]
            entry: dict[str, Any] = {"classification": _classification_metrics(labels, probs, label_order), "n_groups": len(set(groups)), "click_gates": [
                {"false_positive_budget": gate["false_positive_budget"], **gate_metrics(labels, probs[:, click_column], gate["threshold"], label_order)}
                for gate in gates if gate["status"] == "ok"
            ]}
            if axis == "domain":
                entry["group_bootstrap"] = _group_bootstrap_statistics(labels, probs, groups, gates, label_order, seed, bootstrap_iterations)
            if label_order == ("click", "disabled", "absent"):
                entry["paired_visual_state"] = _paired_visual_statistics(labels, probs, groups, gates, label_order)
            slices[value] = entry
        result["axes"][axis] = slices
    return result


def _train_and_evaluate(
    condition: str,
    feature_set: str,
    features: Any,
    reference: FeatureData | SampleData,
    output_dir: Path,
    seed: int,
    c_values: tuple[float, ...],
    budgets: tuple[float, ...],
    label_order: tuple[str, ...] = LABELS,
    task: str = "intent",
    sample_metadata: dict[str, dict[str, Any]] | None = None,
    bootstrap_iterations: int = 1000,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    train = reference.splits == "train"
    val = reference.splits == "val"
    test = reference.splits == "test"
    validation_search = []
    best_key = None
    selected_pipeline = None
    chosen_c = None
    for c_value in sorted(set(c_values)):
        pipeline = _new_pipeline(feature_set, c_value, seed)
        pipeline.fit(features[train], reference.labels[train])
        metrics = _classification_metrics(reference.labels[val], _probabilities(pipeline, features[val], label_order), label_order)
        validation_search.append({"C": c_value, "macro_f1": metrics["macro_f1"], "balanced_accuracy": metrics["balanced_accuracy"]})
        # Lowest C wins exact ties. Nothing is fit or selected on the test split.
        key = (metrics["macro_f1"], metrics["balanced_accuracy"], -c_value)
        if best_key is None or key > best_key:
            best_key, selected_pipeline, chosen_c = key, pipeline, c_value
    assert selected_pipeline is not None
    probabilities = _probabilities(selected_pipeline, features, label_order)
    click_column = label_order.index("click")
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{condition}__{feature_set}.joblib"
    joblib.dump(selected_pipeline, model_path)
    result: dict[str, Any] = {
        "condition": condition,
        "task": task,
        "feature_set": feature_set,
        "feature_dimension": None if feature_set == "text_only" else int(features.shape[1]),
        "chosen_C": chosen_c,
        "selection_split": "val",
        "fit_split": "train",
        "validation_search": validation_search,
        "model_path": str(model_path.relative_to(output_dir)),
        "label_order": list(label_order),
        "val": _classification_metrics(reference.labels[val], probabilities[val], label_order),
        "test": _classification_metrics(reference.labels[test], probabilities[test], label_order),
        "click_gates": [],
    }
    for budget in budgets:
        calibration = calibrate_click_threshold(reference.labels[val], probabilities[val, click_column], budget, label_order)
        if calibration["status"] == "ok":
            calibration["test"] = gate_metrics(reference.labels[test], probabilities[test, click_column], calibration["threshold"], label_order)
        result["click_gates"].append(calibration)
    result["group_bootstrap"] = _group_bootstrap_statistics(reference.labels[test], probabilities[test], reference.groups[test], result["click_gates"], label_order, seed, bootstrap_iterations)
    result["test_slices"] = _test_slice_statistics(reference, probabilities, result["click_gates"], label_order, sample_metadata, seed, bootstrap_iterations)
    if task == "visual_state":
        result["paired_visual_state"] = _paired_visual_statistics(reference.labels[test], probabilities[test], reference.groups[test], result["click_gates"], label_order)
    rows = []
    for idx, sid in enumerate(reference.sample_ids):
        rows.append({
            "condition": condition,
            "feature_set": feature_set,
            "split": str(reference.splits[idx]),
            "sample_id": str(sid),
            "group_id": str(reference.groups[idx]),
            "label": str(reference.labels[idx]),
            "prediction": label_order[int(probabilities[idx].argmax())],
            **{f"p_{label}": float(probabilities[idx, column]) for column, label in enumerate(label_order)},
        })
    return result, rows


def _load_manifest_records(path: Path) -> dict[str, dict[str, Any]]:
    records = {}
    with Path(path).open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            sid = str(item.get("sample_id", ""))
            if not sid or sid in records:
                raise ValueError(f"{path}:{line_number}: missing or duplicate sample_id")
            records[sid] = item
    if not records:
        raise ValueError(f"{path}: empty manifest")
    return records


def _read_manifest(path: Path, reference: FeatureData | SampleData, label_order: tuple[str, ...] = LABELS) -> tuple[np.ndarray, dict[str, Any]]:
    records = _load_manifest_records(path)
    texts = []
    template_splits: dict[str, set[str]] = {}
    templates_by_split: dict[str, set[str]] = {split: set() for split in SPLITS}
    n_without_template = 0
    for sid, label, group, split in zip(reference.sample_ids, reference.labels, reference.groups, reference.splits):
        if sid not in records:
            raise ValueError(f"Manifest is missing common sample {sid!r}")
        item = records[sid]
        instruction = item.get("instruction")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError(f"Manifest sample {sid!r}: instruction must be nonempty text")
        # Validate facts if supplied, rather than trusting a possibly stale manifest.
        if "label" in item:
            value = _normalize_labels(np.asarray([item["label"]]), f"manifest {sid}", label_order)[0]
            if value != label:
                raise ValueError(f"Manifest label mismatch for {sid!r}")
        item_group = item.get("group_id", item.get("group"))
        if item_group is not None and str(item_group) != group:
            raise ValueError(f"Manifest group mismatch for {sid!r}")
        if "split" in item and item["split"] != split:
            raise ValueError(f"Manifest split mismatch for {sid!r}")
        metadata = item.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        template = metadata.get("template_family", item.get("template_family", item.get("template_id")))
        if template is not None:
            template_splits.setdefault(str(template), set()).add(str(split))
            templates_by_split[str(split)].add(str(template))
        else:
            n_without_template += 1
        texts.append(instruction)
    shared = {key: sorted(value) for key, value in template_splits.items() if len(value) > 1}
    test_overlap = templates_by_split["test"] & (templates_by_split["train"] | templates_by_split["val"])
    train_val_shared = templates_by_split["train"] & templates_by_split["val"]
    holdout = None if n_without_template or not templates_by_split["test"] else not bool(test_overlap)
    return np.asarray(texts), {
        "status": "included",
        "input": "instruction only; no image, gold label, generated answer, or split identifier",
        "n_unique_instructions": len(set(texts)),
        "template_split_audit": {
            "n_template_families": len(template_splits),
            "n_samples_without_template": n_without_template,
            "families_by_split": {split: sorted(families) for split, families in templates_by_split.items()},
            "shared_between_splits": shared,
            "test_overlap_with_train_val": sorted(test_overlap),
            "train_val_shared_families": sorted(train_val_shared),
            "is_template_holdout": holdout,
            "note": "Holdout means test families are absent from train+val; train-val overlap is allowed. Prefer metadata.template_family, then top-level template_family, then template_id. Null means missing template facts. Template overlap is not screenshot/group leakage.",
        },
    }


def _metadata_records(path: Path, facts: dict[str, tuple[str, str, str]], label_order: tuple[str, ...] = LABELS) -> list[dict[str, Any]]:
    """Load optional extraction audit data; never use it as probe features."""
    records, seen = [], set()
    group_splits = {group: split for _, group, split in facts.values()}
    with path.open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            sid = str(row.get("sample_id", ""))
            if not sid or sid in seen:
                raise ValueError(f"{path}:{line_number}: missing or duplicate sample_id")
            seen.add(sid)
            expected = facts.get(sid)
            default_label, default_group, default_split = expected or (None, None, None)
            raw_label = row.get("label", default_label)
            label = str(_normalize_labels(np.asarray([raw_label]), f"{path}:{sid}", label_order)[0])
            group = row.get("group_id", row.get("group", default_group))
            split = row.get("split", default_split)
            if group is None or split not in SPLITS:
                raise ValueError(f"{path}:{sid}: metadata requires group_id and train/val/test split")
            group, split = str(group), str(split)
            if expected is not None and (label, group, split) != expected:
                raise ValueError(f"{path}:{sid}: extraction metadata label/group/split mismatch")
            if group in group_splits and group_splits[group] != split:
                raise ValueError(f"Group leakage in metadata: {group!r}")
            group_splits[group] = split
            records.append({**row, "sample_id": sid, "label": label, "group_id": group, "split": split})
    return records


def _numeric_summary(values: Iterable[Any], n_total: int) -> dict[str, Any]:
    clean = [float(value) for value in values if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0]
    return {
        "n_total": n_total,
        "n_available": len(clean),
        "p50": float(np.percentile(clean, 50)) if clean else None,
        "p95": float(np.percentile(clean, 95)) if clean else None,
        "minimum": min(clean) if clean else None,
        "maximum": max(clean) if clean else None,
    }


def _generation_statistics(records: list[dict[str, Any]], label_order: tuple[str, ...] = LABELS) -> dict[str, Any]:
    n = len(records)
    # A deliberate prefill-only run did not attempt generation. It is not a
    # malformed model answer and must not be scored as an incorrect prediction.
    attempted = [row for row in records if row.get("generation_enabled") is not False]
    n_evaluated = len(attempted)
    base = {
        "n": n,
        "n_evaluated": n_evaluated,
        "n_generation_enabled": sum(row.get("generation_enabled") is True for row in records),
        "n_generation_disabled": n - n_evaluated,
        "n_generation_flag_unknown": sum(not isinstance(row.get("generation_enabled"), bool) for row in records),
        "generation_run_coverage": n_evaluated / n if n else None,
        "generation_coverage_note": "Explicit generation_enabled=false rows are not run and excluded from accuracy. Legacy missing flags retain the previous attempted-generation interpretation.",
    }
    if not n:
        return {**base, "status": "unavailable", "reason": "empty subset"}
    if not attempted:
        return {**base, "status": "not_run", "reason": "generation_disabled_for_all_samples", "accuracy_including_missing": None, "no_decision_count": 0, "wrong_decision_count": 0}
    if not any("generated_decision" in row or row.get("generation_enabled") is True for row in attempted):
        return {**base, "status": "unavailable", "reason": "no generated_decision audit field or explicit generation-enabled flag"}
    confusion = np.zeros((3, 4), dtype=int)
    recognized, correct, no_decision = 0, 0, 0
    for row in attempted:
        prediction = row.get("generated_decision")
        prediction = prediction.strip().lower() if isinstance(prediction, str) else None
        if prediction in label_order:
            column = label_order.index(prediction)
            recognized += 1
            correct += int(prediction == row["label"])
        else:
            column = 3
            no_decision += 1
        confusion[label_order.index(row["label"]), column] += 1
    truncation_known = [row["truncated"] for row in attempted if isinstance(row.get("truncated"), bool)]
    return {
        **base,
        "status": "ok",
        "recognized_count": recognized,
        "recognized_coverage": recognized / n_evaluated,
        "recognized_over_requested": recognized / n,
        "correct_count": correct,
        "wrong_decision_count": recognized - correct,
        "no_decision_count": no_decision,
        "accuracy_including_missing": correct / n_evaluated,
        "accuracy_on_recognized_only_diagnostic": correct / recognized if recognized else None,
        "confusion_matrix": confusion.tolist(),
        "true_label_order": list(label_order),
        "prediction_order": [*label_order, "unrecognized"],
        "click_gate": gate_metrics(
            np.asarray([row["label"] for row in attempted]),
            np.asarray([str(row.get("generated_decision", "")).strip().lower() == "click" for row in attempted], dtype=float),
            0.5, label_order,
        ),
        "n_truncation_known": len(truncation_known),
        "n_truncated": sum(truncation_known),
        "truncated_fraction": sum(truncation_known) / len(truncation_known) if truncation_known else None,
        "timing_scope": "all requested records, including deliberate generation-disabled samples",
        "timing_seconds": {
            name: _numeric_summary((row.get("timing_seconds", {}).get(name) for row in records), n)
            for name in ("prepare", "prefill", "generation", "total", "decision_elapsed")
        },
    }


def _summarize_extraction(features_dir: Path, originals: list[FeatureData], reference: FeatureData, label_order: tuple[str, ...] = LABELS) -> dict[str, Any]:
    facts = {
        str(sid): (str(label), str(group), str(split))
        for data in originals
        for sid, label, group, split in zip(data.sample_ids, data.labels, data.groups, data.splits)
    }
    common_ids = set(reference.sample_ids)
    total_path = features_dir / "metadata.jsonl"
    direct: dict[str, Any] = {"status": "unavailable", "reason": "metadata.jsonl is absent"}
    if total_path.is_file():
        records = _metadata_records(total_path, facts, label_order)
        paired = [row for row in records if row["sample_id"] in common_ids]
        all_statistics = _generation_statistics(records, label_order)
        direct = {
            "status": all_statistics["status"],
            "source": "metadata.jsonl",
            "denominator_note": "n counts all requested recorded samples. n_evaluated excludes explicit generation_enabled=false. Accuracy includes malformed/missing decisions among attempted runs; generation_run_coverage reports attempted/requested separately. Legacy missing generation flags retain previous behavior.",
            "n_common_without_metadata": len(common_ids - {row["sample_id"] for row in records}),
            "all": all_statistics,
            "test": _generation_statistics([row for row in records if row["split"] == "test"], label_order),
            "paired_all": _generation_statistics(paired, label_order),
            "paired_test": _generation_statistics([row for row in paired if row["split"] == "test"], label_order),
        }
    latency: dict[str, Any] = {}
    for data in originals:
        path = features_dir / f"{data.condition}.metadata.jsonl"
        if not path.is_file():
            latency[data.condition] = {"status": "unavailable", "reason": f"{path.name} is absent"}
            continue
        records = _metadata_records(path, facts, label_order)
        file_ids = set(data.sample_ids)
        if {row["sample_id"] for row in records} - file_ids:
            raise ValueError(f"{path}: condition metadata contains IDs absent from its feature file")
        subsets = {
            "all": records,
            "test": [row for row in records if row["split"] == "test"],
            "paired_all": [row for row in records if row["sample_id"] in common_ids],
            "paired_test": [row for row in records if row["split"] == "test" and row["sample_id"] in common_ids],
        }
        latency[data.condition] = {
            "status": "ok",
            "n_feature_rows_without_metadata": len(file_ids - {row["sample_id"] for row in records}),
            "subsets": {
                name: {
                    "n": len(rows),
                    "readout_elapsed_seconds": _numeric_summary((row.get("readout_elapsed_seconds") for row in rows), len(rows)),
                    "generated_tokens_at_readout": _numeric_summary((row.get("generated_tokens_at_readout") for row in rows), len(rows)),
                }
                for name, rows in subsets.items()
            },
        }
    return {
        "direct_generation": direct,
        "readout_latency": latency,
        "latency_note": "Extraction timings include hooks, synchronization and CPU copies; they are experiment readout cost, not deployment latency. Full-token generation costs and readout costs can differ.",
    }


def _fmt(value: float | None) -> str:
    return "不可评估" if value is None else f"{value:.4f}"


def _fmt_interval(value: dict[str, Any] | None) -> str:
    if not value:
        return "不可评估"
    interval = value.get("ci95")
    return _fmt(value.get("point")) + (f" [{interval[0]:.4f}, {interval[1]:.4f}]" if interval else " [CI不可评估]")


def _append_grouped_report(lines: list[str], summary: dict[str, Any]) -> None:
    """Append domain, state and paired metrics without changing fit selection."""
    lines.extend([
        "", "## 按场景组重采样的测试不确定性", "",
        "下列 95% 区间通过完整 group_id 重采样得到；同组图像或指令一起进入每次 bootstrap。分类和所有阈值共用同一组重采样权重。它衡量当前数据分布下的组级抽样波动，不包含重新训练模型或重新选择阈值的波动。",
        "", "| 条件 | 特征 | 范围 | N | 组数 | Accuracy [95% CI] | Click AUROC [95% CI] |",
        "| --- | --- | --- | ---: | ---: | --- | --- |",
    ])
    for result in summary["runs"]:
        bootstrap = result.get("group_bootstrap", {})
        lines.append(f"| {result['condition']} | {result['feature_set']} | 全部test | {result['test']['n']} | {bootstrap.get('n_groups', '—')} | {_fmt_interval(bootstrap.get('accuracy'))} | {_fmt_interval(bootstrap.get('click_auroc'))} |")
        domains = result.get("test_slices", {}).get("axes", {}).get("domain", {})
        for domain, entry in domains.items():
            interval = entry.get("group_bootstrap", {})
            lines.append(f"| {result['condition']} | {result['feature_set']} | {domain} | {entry['classification']['n']} | {entry['n_groups']} | {_fmt_interval(interval.get('accuracy'))} | {_fmt_interval(interval.get('click_auroc'))} |")
    lines.extend([
        "", "## 测试点击门控的组级区间与分状态误点击", "",
        "| 条件 | 特征 | 范围 | Val预算 | FPR [95% CI] | Recall [95% CI] | 各负状态 FPR |",
        "| --- | --- | --- | ---: | --- | --- | --- |",
    ])
    for result in summary["runs"]:
        scopes = [("全部test", result["click_gates"], result.get("group_bootstrap", {}), True)]
        for domain, entry in result.get("test_slices", {}).get("axes", {}).get("domain", {}).items():
            scopes.append((domain, entry["click_gates"], entry.get("group_bootstrap", {}), False))
        for name, gates, bootstrap, calibration_form in scopes:
            boot_gates = {gate["false_positive_budget"]: gate for gate in bootstrap.get("click_gates", [])}
            for gate in gates:
                if calibration_form and gate.get("status") != "ok":
                    continue
                point = gate["test"] if calibration_form else gate
                intervals = boot_gates.get(gate["false_positive_budget"], {})
                state_text = "; ".join(f"{state}: {_fmt(entry['false_click_rate'])} (n={entry['n']})" for state, entry in point["false_click_rate_by_negative_state"].items())
                lines.append(f"| {result['condition']} | {result['feature_set']} | {name} | {gate['false_positive_budget']:.1%} | {_fmt_interval(intervals.get('false_click_rate'))} | {_fmt_interval(intervals.get('recall'))} | {state_text} |")
    lines.extend(["", "iid/ood 只用于测试切片，所有切片复用全局 val 选择的 C 和阈值。layout/style_family 切片的分类与门控指标见 metrics.json，它们不会进入分类器输入。组数不足或 bootstrap 中缺少正/负样本时，区间标记不可评估并记录有效重采样数。"])
    if summary.get("task") == "visual_state":
        lines.extend([
            "", "## 同场景三联组与视觉状态翻转", "",
            "严格配对 click / disabled / absent 三张图。缺失或重复状态的组单独计数，不伪装成完整三联组。positive 差分要求严格大于零，平分算未成功翻转。",
            "", "| 条件 | 特征 | 范围 | 完整组/总组 | 三张分类全对 | enabled−disabled>0 | enabled−absent>0 | 两差均>0 |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
        ])
        for result in summary["runs"]:
            scopes = [("全部test", result.get("paired_visual_state", {}))]
            scopes.extend((domain, entry.get("paired_visual_state", {})) for domain, entry in result.get("test_slices", {}).get("axes", {}).get("domain", {}).items())
            for name, paired in scopes:
                if paired.get("status") != "ok":
                    continue
                flip = paired["paired_flip"]
                lines.append(f"| {result['condition']} | {result['feature_set']} | {name} | {paired['n_complete_triplets']}/{paired['n_total_groups']} | {_fmt(paired['triplet_all_correct_rate'])} | {_fmt(flip['enabled_minus_disabled_positive_rate'])} | {_fmt(flip['enabled_minus_absent_positive_rate'])} | {_fmt(flip['both_positive_rate'])} |")
        lines.extend(["", "| 条件 | 特征 | 范围 | Val预算 | 完整三联组正确门控率 |", "| --- | --- | --- | ---: | ---: |"])
        for result in summary["runs"]:
            scopes = [("全部test", result.get("paired_visual_state", {}))]
            scopes.extend((domain, entry.get("paired_visual_state", {})) for domain, entry in result.get("test_slices", {}).get("axes", {}).get("domain", {}).items())
            for name, paired in scopes:
                for gate in paired.get("click_gates", []):
                    lines.append(f"| {result['condition']} | {result['feature_set']} | {name} | {gate['false_positive_budget']:.1%} | {_fmt(gate['complete_triplet_correct_gate_rate'])} |")
        lines.extend(["", "完整三联组正确门控要求仅 enabled 点击，同时拒绝 disabled/absent。不完整组 ID 与平均分差保存在 paired_visual_state。该统计与单图准确率互补，不能互相替代。"])
        direct = summary.get("extraction_audit", {}).get("direct_generation", {})
        if direct.get("status") == "ok" and direct.get("paired_test", {}).get("status") == "ok":
            baseline = direct["paired_test"]
            states = baseline["click_gate"]["false_click_rate_by_negative_state"]
            lines.extend(["", "直接生成类别词的配对测试门控：" + "; ".join(f"{state} FPR={_fmt(entry['false_click_rate'])}" for state, entry in states.items()) + f"。未识别类别数 {baseline['no_decision_count']}；UNKNOWN 保留为未识别分类错误，执行层只将明确 CLICK 视为点击。"])


def _write_report(summary: dict[str, Any], path: Path) -> None:
    coverage = summary["coverage"]
    label_text = " / ".join(summary["label_order"])
    visual_state = summary.get("task") == "visual_state"
    description = (
        "本实验使用固定指令，区分同一场景三张图中的 click（按钮可用）、disabled（按钮禁用）、absent（按钮不存在）。这评估视觉状态门控，不评估点击坐标、实际执行或多步闭环能力。"
        if visual_state else
        "本实验评估同一截图下 click（点击）、describe（描述）、forbid（禁止点击）的意图区分；不评估点击坐标、实际执行或多步闭环能力。"
    )
    lines = [
        "# 冻结 Qwen 门控探针评估",
        "",
        description + "数值探针仅拟合缓存特征，基础 VLM 保持冻结。",
        "",
        "prefill、post_decision 和 post_response 是不同的读取时机，生成前和生成后均是有效候选；目标是比较效果与成本，不预设提前判断必须胜出。生成后条件必须接在模型实际生成的内容后提取，不得把真值动作标签作为 teacher forcing 输入。本模块不生成文本，需结合提取元数据审计这一条件。",
        "",
        "标准化和分类器只在 train 拟合；正则强度 C 只按 val 的 macro-F1（平分时 balanced accuracy，再选更强正则）选择。模型不使用 train+val 重拟合。点击阈值只由 val 非点击样本校准，随后原样应用 test。",
        "",
        "## 配对样本与覆盖率",
        "",
        f"所有可用条件取相同样本交集：{coverage['n_common']} / 并集 {coverage['n_union']}（{coverage['common_over_union']:.1%}）。缺失条件：{', '.join(coverage['unavailable_conditions']) or '无'}。",
        "",
        "| 条件 | 可用样本 | 共同样本 | 覆盖率 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, details in coverage["conditions"].items():
        lines.append(f"| {name} | {details['n_available']} | {details['n_common']} | {details['common_over_available']:.1%} |")
    lines.extend([
        "",
        "若生成后条件仅覆盖部分样本，配对分数属于这个可用子集，不能直接推广到完整任务集。metrics.json 中按 split、类别报告筛选前后计数与被排除 ID，需检查缺失是否与模型失败或类别有关。整份原始文件先检查分组泄漏，再取交集。",
        "",
        "## 模型实际生成类别词的直接基线",
        "",
        "post_decision 已包含模型自己生成的动作类别词，probe 高分可能只是读回已有决策。因此应同时比较直接解析 generated_decision 的成绩。启用生成的样本中，未生成或格式无法识别的类别算错误；主动禁用生成的 prefill-only 样本属于未运行，不计成模型错误。",
        "",
    ])
    direct = summary["extraction_audit"]["direct_generation"]
    if direct["status"] == "ok":
        lines.extend([
            "| 范围 | 请求 N | 运行 N | 运行覆盖率 | 含无类别的准确率 | 运行中可识别类别覆盖率 | 错误类别数 | 无类别数 | 截断比例 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        names = {"all": "全部已记录提取样本", "test": "全部已记录测试样本", "paired_all": "配对交集已记录样本", "paired_test": "配对交集已记录测试样本"}
        for key, name in names.items():
            metrics = direct[key]
            if metrics["status"] != "ok":
                state = "未运行" if metrics["status"] == "not_run" else "不可评估"
                lines.append(f"| {name} | {metrics['n']} | {metrics['n_evaluated']} | {_fmt(metrics['generation_run_coverage'])} | {state} | {state} | — | — | — |")
                continue
            lines.append(f"| {name} | {metrics['n']} | {metrics['n_evaluated']} | {_fmt(metrics['generation_run_coverage'])} | {_fmt(metrics['accuracy_including_missing'])} | {_fmt(metrics['recognized_coverage'])} | {metrics['wrong_decision_count']} | {metrics['no_decision_count']} | {_fmt(metrics['truncated_fraction'])} |")
        lines.extend([
            "",
            f"交集中缺少总生成元数据的样本数：{direct['n_common_without_metadata']}。准确率以运行样本为分母，generation_enabled=false 的未运行样本另报覆盖率；无 generation_enabled 字段的旧记录沿用原来的已尝试生成解释。元数据不完整时不能视为完整配对比较。截断比例仅以运行样本中有 truncated 字段的记录为分母，具体计数见 metrics.json。",
        ])
    elif direct["status"] == "not_run":
        lines.append(f"全部 {direct['all']['n']} 条元数据均为 generation_enabled=false，直接生成类别基线未运行（not_run），不是 0% 准确率的模型失败。")
    else:
        lines.append("提取元数据缺失或没有 generated_decision 字段，直接生成类别基线不可评估。")
    lines.extend([
        "",
        "## 读取成本（配对测试子集）",
        "",
        "| 条件 | N | 有效计时数 | 读取耗时 P50 秒 | P95 秒 | 已生成 token P50 | P95 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for condition, timing in summary["extraction_audit"]["readout_latency"].items():
        if timing["status"] != "ok":
            lines.append(f"| {condition} | — | 0 | 不可评估 | 不可评估 | 不可评估 | 不可评估 |")
            continue
        subset = timing["subsets"]["paired_test"]
        elapsed = subset["readout_elapsed_seconds"]
        tokens = subset["generated_tokens_at_readout"]
        lines.append(f"| {condition} | {subset['n']} | {elapsed['n_available']} | {_fmt(elapsed['p50'])} | {_fmt(elapsed['p95'])} | {_fmt(tokens['p50'])} | {_fmt(tokens['p95'])} |")
    lines.extend([
        "",
        "这些耗时包含特征 hook、同步和 CPU 拷贝，表示实验读取成本，不是部署后的纯模型或动作端到端延迟。全部样本、测试样本及配对子集的完整时延分位数见 metrics.json。",
        "",
        "## 独立测试集分类结果",
        "",
        "| 读取时机 | 特征 | C | N | Accuracy | Macro-F1 | Balanced Acc | Click AUROC | Click AP | Brier |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for result in summary["runs"]:
        metrics = result["test"]
        lines.append(
            f"| {result['condition']} | {result['feature_set']} | {result['chosen_C']:g} | {metrics['n']} | "
            + " | ".join(_fmt(metrics[key]) for key in ("accuracy", "macro_f1", "balanced_accuracy", "click_auroc", "click_average_precision", "click_brier"))
            + " |"
        )
    lines.extend([
        "",
        f"混淆矩阵见 metrics.json（行是真值，列是预测，顺序 {label_text}）。测试集缺少某类时明确标记 all_three_classes_present=false；二分类仅有单类时 AUROC/AP 为 null，不能解释为完美表现。",
        "",
        "## 验证集校准后的点击门控（测试结果）",
        "",
        "| 读取时机 | 特征 | Val FPR 预算 | 阈值 | Test 误点击率 | 点击 precision | 点击 recall | 触发比例 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for result in summary["runs"]:
        for gate in result["click_gates"]:
            if gate["status"] != "ok":
                continue
            test = gate["test"]
            lines.append(
                f"| {result['condition']} | {result['feature_set']} | {gate['false_positive_budget']:.1%} | {gate['threshold']:.6g} | "
                + " | ".join(_fmt(test[key]) for key in ("false_click_rate", "precision", "recall", "coverage"))
                + " |"
            )
    lines.extend([
        "",
        "误点击率 = 非点击样本中被门控允许点击的比例；触发比例 = 所有样本中允许点击的比例。没有触发时 precision/risk 为 null，不能记成 100% 安全。阈值可超过 1，表示拒绝所有点击。",
        "",
        "FPR 预算是验证集经验约束，不是对未来误点概率的保证。小验证集可能让 1%、5% 预算产生相同阈值；metrics.json 包含验证集非点击数、FPR 最小分辨率以及测试误点率的 Wilson 95% 区间。Wilson 区间仅为逐样本二项近似；同截图的多条指令相关，因此它不是按截图组泛化的置信区间，可能过窄。",
        "",
        "## 语言捷径检查",
        "",
    ])
    text = summary["text_baseline"]
    if text["status"] == "included":
        lines.extend([
            "已包含 instruction-only 字符 TF-IDF + L2 logistic 基线，使用完全相同的样本和 split。它不读取截图或生成回答。",
            "",
            ("指令全文相同时，平衡三类数据的纯文本 accuracy 应接近 1/3、click AUROC 应为 0.5。高于机会水平需排查措辞或元数据泄漏；只有 attention/hidden 相对文本对照的增益才支持视觉信息有帮助。" if visual_state else "如果文本基线已很强，hidden/attention 的高分只能说明模型内部保留了指令类别，不能证明门控理解了图像可执行性。应增加目标不存在、禁用、等待条件，并做模板族留出，再验证图像是否带来增益。"),
            "",
            f"模板族留出审计：{json.dumps(text['template_split_audit'], ensure_ascii=False)}",
        ])
    else:
        lines.append("未提供 manifest，语言基线缺失；当前结果不完整，不能排除仅由指令措辞区分类别的解释。")
    lines.extend([
        "",
        "## 产物与边界",
        "",
        "- metrics.json：完整指标、覆盖率、C 搜索与阈值校准细节。",
        "- predictions.csv：train/val/test 的逐样本概率，注意仅 test 指标用于最终报告。",
        "- models/：每组探针的 joblib pipeline（只加载可信文件）。",
        "",
        "本报告不会把生成后的模型回答正确率当成探针准确率，也不会把 attention 集中程度自动解释为点击意图。禁止人工按 test 分数挑层、挑头、修改模板或阈值；这些选择应在 train/val 或单独开发集完成。",
    ])
    _append_grouped_report(lines, summary)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_probe_suite(
    features_dir: Path,
    output_dir: Path,
    seed: int = 42,
    *,
    manifest_path: Path | None = None,
    false_positive_budgets: Iterable[float] = (0.01, 0.05, 0.10),
    c_values: Iterable[float] = (0.01, 0.1, 1.0, 10.0),
    min_common_coverage: float = 0.0,
    task: str = "intent",
    bootstrap_iterations: int = 1000,
) -> dict[str, Any]:
    """Train three probes per available condition and an optional text baseline.

    Input: prefill.npz / post_decision.npz / post_response.npz with attention,
    hidden, sample_ids, labels, groups and splits. IDs/group/split arrays must be
    non-object strings; labels are strings or 0/1/2 in the selected task order. The manifest
    is JSONL with sample_id and instruction (optional label/group_id/split facts
    are checked). Every pair of conditions uses the common sample intersection.

    Returns the same JSON-serializable summary saved to metrics.json. Raises
    ValueError for leakage, misalignment or missing training/validation classes.
    """
    features_dir, output_dir = Path(features_dir), Path(output_dir)
    spec = _task_spec(task)
    label_order = spec.label_order
    if not isinstance(bootstrap_iterations, int) or isinstance(bootstrap_iterations, bool) or bootstrap_iterations < 0:
        raise ValueError("bootstrap_iterations must be a nonnegative integer")
    budgets = tuple(float(value) for value in false_positive_budgets)
    regularization = tuple(float(value) for value in c_values)
    if not regularization or any(not math.isfinite(value) or value <= 0 for value in regularization):
        raise ValueError("c_values must contain positive finite values")
    if not budgets or any(not math.isfinite(value) or not 0 <= value <= 1 for value in budgets):
        raise ValueError("false_positive_budgets must be nonempty and within [0,1]")
    if not 0 <= min_common_coverage <= 1:
        raise ValueError("min_common_coverage must be within [0,1]")
    datasets = [load_feature_file(features_dir / f"{name}.npz", label_order) for name in CONDITIONS if (features_dir / f"{name}.npz").is_file()]
    if not datasets:
        raise ValueError(f"No condition NPZ files found in {features_dir}")
    originals = datasets
    datasets, coverage = _align_conditions(datasets, min_common_coverage, label_order)
    reference = datasets[0]
    text_features = None
    text_info: dict[str, Any] = {"status": "missing_manifest", "reason": "Pass manifest_path to include the required language-only baseline"}
    sample_metadata = None
    if manifest_path is not None:
        text_features, text_info = _read_manifest(Path(manifest_path), reference, label_order)
        sample_metadata = _load_manifest_records(Path(manifest_path))
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "schema_version": 1,
        "experiment_type": "frozen_vlm_feature_probes",
        "seed": seed,
        "label_order": list(label_order),
        "task": task,
        "task_description": "paired visual-state discrimination with a constant instruction" if task == "visual_state" else "three-class instruction intent; not coordinate accuracy or closed-loop control",
        "coverage": coverage,
        "text_baseline": text_info,
        "extraction_audit": _summarize_extraction(features_dir, originals, reference, label_order),
        "selection_policy": {"fit": "train only", "C": "val macro_f1, then balanced_accuracy, then lower C", "thresholds": "val only", "refit_on_train_val": False},
        "uncertainty_note": "Primary 95% confidence intervals use whole-group percentile bootstrap. Wilson intervals retained in gate records are row-level binomial approximations, not group-generalization intervals.",
        "runs": [],
    }
    prediction_rows = []
    for data in datasets:
        feature_sets = {
            "attention": data.attention,
            "hidden": data.hidden,
            "combined": np.concatenate([data.attention, data.hidden], axis=1),
        }
        for feature_set, features in feature_sets.items():
            result, rows = _train_and_evaluate(data.condition, feature_set, features, data, output_dir, seed, regularization, budgets, label_order, task, sample_metadata, bootstrap_iterations)
            summary["runs"].append(result)
            prediction_rows.extend(rows)
    if text_features is not None:
        result, rows = _train_and_evaluate("instruction", "text_only", text_features, reference, output_dir, seed, regularization, budgets, label_order, task, sample_metadata, bootstrap_iterations)
        summary["runs"].append(result)
        prediction_rows.extend(rows)
    (output_dir / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (output_dir / "predictions.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(prediction_rows[0]))
        writer.writeheader()
        writer.writerows(prediction_rows)
    _write_report(summary, output_dir / "report.md")
    return summary


def _write_text_baseline_report(summary: dict[str, Any], path: Path) -> None:
    result = summary["runs"][0]
    metrics = result["test"]
    label_order = summary["label_order"]
    label_text = " / ".join(label_order)
    lines = [
        "# 仅指令文本基线",
        "",
        f"本次只运行字符 TF-IDF + L2 logistic 回归，检验指令文本能否区分 {label_text}。没有加载 VLM、没有读取截图、没有提取注意力或隐藏状态，也没有使用模拟特征代替模型输出。此结果不是 Qwen 门控效果或视觉能力结果。",
        "",
        "样本按 manifest 中的截图 group 划分；同 group 不得跨 split。TF-IDF 词表、IDF 和分类器只用 train 拟合，C 只由 val 选择。点击阈值只由 val 校准，test 只用于最终评估，不在 train+val 重拟合。",
        "",
        f"共 {summary['sample_audit']['n_samples']} 条指令，{summary['sample_audit']['n_groups']} 个截图组。",
        "",
        "| Split | 样本数 | 截图组数 | " + " | ".join(label_order) + " |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split, detail in summary["sample_audit"]["splits"].items():
        counts = detail["label_counts"]
        lines.append(f"| {split} | {detail['n']} | {detail['n_groups']} | " + " | ".join(str(counts[label]) for label in label_order) + " |")
    lines.extend([
        "",
        "## 独立测试集结果",
        "",
        "| C | Accuracy | Macro-F1 | Balanced Acc | Click AUROC | Click AP | Click Brier |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| {result['chosen_C']:g} | " + " | ".join(_fmt(metrics[key]) for key in ("accuracy", "macro_f1", "balanced_accuracy", "click_auroc", "click_average_precision", "click_brier")) + " |",
        "",
        f"混淆矩阵见 metrics.json：行是真值、列是预测，顺序 {label_text}。测试缺类时 all_three_classes_present=false，只有一个 click-vs-nonclick 类别时 AUROC/AP 为 null。",
        "",
        "## 验证集校准后的点击门控",
        "",
        "| Val FPR 预算 | 阈值 | Test 误点击率 | 点击 precision | 点击 recall | 触发比例 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for gate in result["click_gates"]:
        if gate["status"] == "ok":
            test = gate["test"]
            lines.append(f"| {gate['false_positive_budget']:.1%} | {gate['threshold']:.6g} | " + " | ".join(_fmt(test[key]) for key in ("false_click_rate", "precision", "recall", "coverage")) + " |")
    lines.extend([
        "",
        "误点击率以所有非点击指令为分母；触发比例以全部指令为分母。没有触发时 precision 为 null。预算只是验证集经验约束；Wilson 区间为逐样本二项近似，同图指令相关，因此不是截图组泛化置信区间。",
        "",
        "## 模板审计与解释边界",
        "",
        json.dumps(summary["text_baseline"]["template_split_audit"], ensure_ascii=False),
        "",
        ("visual_state 的指令应完全相同；平衡三类的文本 accuracy 应为 1/3、click AUROC 为 0.5。若明显超出机会水平，先检查指令是否混入状态信息。" if summary.get("task") == "visual_state" else "高分说明指令语言本身提供了强类别线索。后续 attention、hidden、combined 探针应在相同样本和划分上与这个基线比较；本次文本结果不能代替 VLM 的真实提取实验。目标不存在、禁用、等待等需要图像才能判断的情形尚需另行评估。"),
        "",
        "输出：metrics.json 包含完整指标与选择记录；predictions.csv 包含逐样本概率及 split；models/ 包含仅在 train 拟合的 pipeline。",
    ])
    _append_grouped_report(lines, summary)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_text_baseline(
    manifest_path: Path,
    output_dir: Path,
    seed: int = 42,
    *,
    c_values: Iterable[float] = (0.01, 0.1, 1.0, 10.0),
    false_positive_budgets: Iterable[float] = (0.01, 0.05, 0.10),
    task: str = "intent",
    bootstrap_iterations: int = 1000,
) -> dict[str, Any]:
    """Run the instruction-only baseline without VLM weights or feature files.

    Requires a JSONL manifest with sample_id, instruction, label, group_id (or
    group) and split. Does not open image paths. Uses the same train-only fit,
    validation model selection, and validation click calibration as the paired
    feature suite. Outputs an explicitly text-only report and saved pipeline.
    """
    manifest_path, output_dir = Path(manifest_path), Path(output_dir)
    label_order = _task_spec(task).label_order
    if not isinstance(bootstrap_iterations, int) or isinstance(bootstrap_iterations, bool) or bootstrap_iterations < 0:
        raise ValueError("bootstrap_iterations must be a nonnegative integer")
    records = _load_manifest_records(manifest_path)
    ids, labels, groups, splits = [], [], [], []
    for sid, item in records.items():
        for key in ("label", "split"):
            if key not in item:
                raise ValueError(f"Manifest sample {sid!r} is missing {key}")
        group = item.get("group_id", item.get("group"))
        if group is None or not str(group).strip():
            raise ValueError(f"Manifest sample {sid!r} is missing group_id")
        ids.append(sid)
        labels.append(str(_normalize_labels(np.asarray([item["label"]]), f"manifest {sid}", label_order)[0]))
        groups.append(str(group))
        splits.append(str(item["split"]))
    reference = SampleData(np.asarray(ids), np.asarray(labels), np.asarray(groups), np.asarray(splits))
    _validate_partitions(reference, label_order)
    texts, text_info = _read_manifest(manifest_path, reference, label_order)
    budgets = tuple(float(value) for value in false_positive_budgets)
    regularization = tuple(float(value) for value in c_values)
    if not regularization or any(not math.isfinite(value) or value <= 0 for value in regularization):
        raise ValueError("c_values must contain positive finite values")
    if not budgets or any(not math.isfinite(value) or not 0 <= value <= 1 for value in budgets):
        raise ValueError("false_positive_budgets must be nonempty and within [0,1]")
    output_dir.mkdir(parents=True, exist_ok=True)
    result, rows = _train_and_evaluate("instruction", "text_only", texts, reference, output_dir, seed, regularization, budgets, label_order, task, records, bootstrap_iterations)
    summary: dict[str, Any] = {
        "schema_version": 1,
        "experiment_type": "text_only_baseline",
        "seed": seed,
        "label_order": list(label_order),
        "task": task,
        "task_description": "instruction-only baseline; no VLM features or visual ability evaluation",
        "manifest_path": str(manifest_path.resolve()),
        "text_baseline": text_info,
        "sample_audit": {
            "n_samples": len(reference.sample_ids),
            "n_groups": len(set(reference.groups)),
            "splits": {
                split: {
                    "n": int(np.sum(reference.splits == split)),
                    "n_groups": len(set(reference.groups[reference.splits == split])),
                    "label_counts": _count_labels(reference.labels[reference.splits == split], label_order),
                }
                for split in SPLITS
            },
        },
        "selection_policy": {"fit": "train only", "C": "val macro_f1, then balanced_accuracy, then lower C", "thresholds": "val only", "refit_on_train_val": False},
        "uncertainty_note": "Primary 95% confidence intervals use whole-group percentile bootstrap. Wilson intervals retained in gate records are row-level binomial approximations, not group-generalization intervals.",
        "runs": [result],
    }
    (output_dir / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (output_dir / "predictions.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _write_text_baseline_report(summary, output_dir / "report.md")
    return summary
