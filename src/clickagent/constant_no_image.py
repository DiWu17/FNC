"""Transparent deterministic-input caching for the visual-state no-image control.

This is a numerical equivalence control, not 900 independent model observations:
one unique input is checked with real forwards across all three truth labels,
then its exactly equal frozen-model features are reused. Timing is retained only
for rows on which a forward actually ran.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

from .dataset import load_manifest
from .model_features import (
    EXTRACTION_IMPLEMENTATION_VERSION,
    ExtractionConfig,
    FrozenQwenExtractor,
    TASK_LABELS,
    VISUAL_STATE_INSTRUCTION,
    _atomic_json,
    _jsonable_config,
    _library_versions,
    _local_model_identity,
    _read_manifest,
    system_prompt_for_task,
)


CONSTANT_CONTROL_VERSION = 1
CONTROL_KIND = "constant_input_no_image_prefill"
ARTIFACT_FILES = ("prefill.npz", "prefill.metadata.jsonl", "metadata.jsonl", "schema.json")
LOGGER = logging.getLogger(__name__)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _prepared_input_identity(extractor: Any, sample: dict[str, Any], manifest_dir: Path) -> dict[str, Any]:
    inputs, image_positions, grid_shape = extractor._prepare_inputs(sample, manifest_dir)
    if np.asarray(image_positions).size or tuple(grid_shape) != (0, 0):
        raise ValueError("constant no-image control unexpectedly contains image positions")
    digest = hashlib.sha256()
    fields = {}
    for key in sorted(inputs):
        if any(word in key for word in ("pixel_values", "image_grid", "video")):
            raise ValueError(f"unexpected visual input field in no-image control: {key}")
        value = inputs[key]
        if hasattr(value, "detach"):
            value = value.detach().cpu().contiguous().numpy()
        value = np.ascontiguousarray(np.asarray(value))
        if value.dtype.kind == "O":
            raise ValueError("model inputs must contain numerical tensors, not objects")
        descriptor = {"dtype": str(value.dtype), "shape": list(value.shape)}
        fields[key] = descriptor
        digest.update(json.dumps({key: descriptor}, sort_keys=True).encode("utf-8"))
        digest.update(value.tobytes(order="C"))
    if "input_ids" not in fields:
        raise ValueError("prepared no-image input is missing input_ids")
    return {"sha256": digest.hexdigest(), "fields": fields}


def _verify_snapshot(reference: dict[str, Any] | None, snapshot: dict[str, Any], sample_id: str) -> None:
    for field in ("attention", "hidden"):
        value = np.asarray(snapshot[field])
        if value.ndim != 1 or value.size == 0 or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError(f"{sample_id}: {field} must be a finite, nonempty float32 feature vector")
        if len(snapshot[f"{field}_names"]) != value.size:
            raise ValueError(f"{sample_id}: {field} feature names do not match the vector")
        if reference is not None:
            expected = np.asarray(reference[field])
            if (value.dtype != expected.dtype or value.shape != expected.shape
                    or value.tobytes() != expected.tobytes()):
                raise ValueError(f"constant-input bit-exact verification failed for {field} at {sample_id}")
    if reference is not None:
        for field in ("attention_names", "hidden_names", "query_token_index", "generated_tokens_at_readout"):
            if snapshot[field] != reference[field]:
                raise ValueError(f"constant-input readout/schema differs at {sample_id}: {field}")
    if snapshot["generated_tokens_at_readout"] != 0:
        raise ValueError("no-image prefill control must not generate tokens")


def _existing_export(output_dir: Path, fingerprint: str, resume: bool) -> dict[str, Any] | None:
    if not output_dir.exists():
        return None
    if not output_dir.is_dir():
        raise ValueError("control output path is not a directory")
    if not any(output_dir.iterdir()):
        return None
    if not resume:
        raise ValueError("control output directory is nonempty; choose a new path or enable resume")
    run_path, summary_path = output_dir / "run.json", output_dir / "summary.json"
    if not run_path.is_file() or not summary_path.is_file():
        raise ValueError("control output directory is nonempty but has no complete cached export")
    run = json.loads(run_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (run.get("fingerprint") != fingerprint or summary.get("fingerprint") != fingerprint
            or run.get("control_kind") != CONTROL_KIND or run.get("status") != "complete"):
        raise ValueError("control output directory belongs to different input/config or an incomplete run")
    for name in ARTIFACT_FILES:
        path = output_dir / name
        if not path.is_file() or _sha256_file(path) != run.get("artifact_sha256", {}).get(name):
            raise ValueError(f"cached control artifact is missing or changed: {name}")
    return {**summary, "reused_existing_export": True, "forward_count_this_call": 0}


def _atomic_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def _unexpected_output_during_verification(output_dir: Path, offload_folder: str | None) -> bool:
    """Allow only the configured offload subtree created by our model loader."""
    if not output_dir.exists():
        return False
    allowed = Path(offload_folder).resolve() if offload_folder else None
    if allowed is None or not allowed.is_relative_to(output_dir) or allowed == output_dir:
        return any(output_dir.iterdir())
    for entry in output_dir.rglob("*"):
        resolved = entry.resolve()
        if resolved == allowed or resolved.is_relative_to(allowed) or resolved in allowed.parents:
            continue
        return True
    return False


def export_constant_no_image_control(
    manifest_path: Path,
    output_dir: Path,
    config: ExtractionConfig,
    verification_forwards: int = 3,
) -> dict[str, Any]:
    """Verify real constant-input features, then transparently reuse their values.

    The first verification covers one sample from every truth class; additional
    requested verifications cover distinct samples in manifest order. Label and
    annotation fields select audit rows and populate output metadata only. They
    never enter the model prompt or numerical feature vectors.
    """
    if config.task != "visual_state" or not config.no_image or config.conditions != ("prefill",):
        raise ValueError("constant control requires task='visual_state', no_image=True, conditions=('prefill',)")
    if isinstance(verification_forwards, bool) or not isinstance(verification_forwards, int) or verification_forwards < 3:
        raise ValueError("verification_forwards must be an integer >= 3 to cover all labels")
    manifest_path, output_dir = Path(manifest_path).resolve(), Path(output_dir).resolve()
    load_manifest(manifest_path, validate_images=False)  # Full schema/group validation; never open image files.
    samples = _read_manifest(manifest_path, None, task="visual_state")
    if len(samples) < verification_forwards:
        raise ValueError("verification_forwards exceeds the number of distinct samples")
    if {row["instruction"] for row in samples} != {VISUAL_STATE_INSTRUCTION}:
        raise ValueError("all no-image samples must have exactly the same instruction")
    if any(row.get("task", "visual_state") != "visual_state" for row in samples):
        raise ValueError("manifest task disagrees with the visual_state control")
    selected = []
    for label in TASK_LABELS["visual_state"]:
        match = next((row for row in samples if row["label"] == label), None)
        if match is None:
            raise ValueError(f"no verification sample exists for label {label}")
        selected.append(match)
    selected_ids = {row["sample_id"] for row in selected}
    for row in samples:
        if len(selected) >= verification_forwards:
            break
        if row["sample_id"] not in selected_ids:
            selected.append(row)
            selected_ids.add(row["sample_id"])
    input_definition = {
        "messages": [{"role": "system", "content": system_prompt_for_task("visual_state")},
                     {"role": "user", "content": [{"type": "text", "text": VISUAL_STATE_INSTRUCTION}]}],
        "add_generation_prompt": True, "images": None,
        "model_input_annotation_fields": [], "generation": "disabled; prefill only",
    }
    config_record = _jsonable_config(config)
    identity_config = {key: value for key, value in config_record.items() if key != "resume"}
    provenance = {
        "control_kind": CONTROL_KIND, "control_implementation_version": CONSTANT_CONTROL_VERSION,
        "feature_extraction_implementation_version": EXTRACTION_IMPLEMENTATION_VERSION,
        "task": "visual_state", "label_order": list(TASK_LABELS["visual_state"]),
        "manifest_sha256": _sha256_file(manifest_path), "input_definition": input_definition,
        "model_identity": _local_model_identity(config.model_path), "library_versions": _library_versions(),
        "verification_forwards": verification_forwards,
        "verification_sample_ids": [row["sample_id"] for row in selected],
        "cache_assumption": ("Frozen eval/inference is deterministic for identical processed inputs; "
                             "bit-exact equality is checked on the recorded verification forwards, "
                             "not measured independently on every exported manifest row."),
    }
    fingerprint = hashlib.sha256(json.dumps({**provenance, "config": identity_config}, sort_keys=True).encode()).hexdigest()
    existing = _existing_export(output_dir, fingerprint, config.resume)
    if existing is not None:
        return existing

    # No output artifacts are written until all real forwards and input/feature
    # equality checks pass. A failure cannot leave a plausible partial NPZ.
    extractor = FrozenQwenExtractor(config)
    reference = input_identity = None
    verifications: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for sample in selected:
        LOGGER.info("Verifying constant no-image forward %d/%d: %s", len(verifications) + 1,
                    verification_forwards, sample["sample_id"])
        actual_input = _prepared_input_identity(extractor, sample, manifest_path.parent)
        if input_identity is None:
            input_identity = actual_input
        elif actual_input != input_identity:
            raise ValueError(f"processed no-image inputs differ at {sample['sample_id']}")
        snapshots, metadata = extractor.extract_sample(sample, manifest_path.parent)
        if set(snapshots) != {"prefill"}:
            raise ValueError("actual verification forward did not produce exactly the prefill condition")
        if metadata.get("generation_enabled") is not False or metadata.get("generated_tokens") != 0:
            raise ValueError("actual verification unexpectedly generated tokens")
        snapshot = snapshots["prefill"]
        _verify_snapshot(reference, snapshot, sample["sample_id"])
        if reference is None:
            reference = snapshot
        verifications[sample["sample_id"]] = (snapshot, metadata)
    assert reference is not None and input_identity is not None
    runtime = copy.deepcopy(getattr(extractor, "runtime_metadata", None))
    source_id = selected[0]["sample_id"]
    template_metadata = verifications[source_id][1]
    sample_fields = set().union(*(row.keys() for row in samples))
    constant_metadata = {key: copy.deepcopy(value) for key, value in template_metadata.items() if key not in sample_fields}
    rows = []
    for sample in samples:
        sample_id = sample["sample_id"]
        actual = verifications.get(sample_id)
        if actual is not None:
            snapshot, measured_metadata = actual
            row = {**copy.deepcopy(measured_metadata), **copy.deepcopy(sample)}
            elapsed = snapshot["readout_elapsed_seconds"]
        else:
            row = {**copy.deepcopy(constant_metadata), **copy.deepcopy(sample)}
            row["timing_seconds"] = {"prepare": None, "prefill": None, "generation": None,
                                     "total": None, "decision_elapsed": None}
            elapsed = None
        row.update({
            "task": "visual_state", "label_order": list(TASK_LABELS["visual_state"]),
            "control_kind": CONTROL_KIND, "feature_reused": actual is None,
            "from_verification_forward": actual is not None,
            "feature_source_sample_id": sample_id if actual is not None else source_id,
            "verification_forward_index": ([entry["sample_id"] for entry in selected].index(sample_id) if actual else None),
            "generated_text": "", "generated_token_ids": [], "generated_tokens": 0,
            "generated_decision": None, "decision_completion_token_index": None,
            "decision_matches_label": False, "decision_abstained": False, "decision_in_label_space": False,
            "generation_enabled": False, "generation_stop_reason": "disabled",
            "terminated_by_eos": False, "truncated": False,
            "available_conditions": ["prefill"], "missing_conditions": [],
            "query_token_index": reference["query_token_index"], "generated_tokens_at_readout": 0,
            "readout_elapsed_seconds": elapsed,
        })
        rows.append(row)

    if _unexpected_output_during_verification(output_dir, config.offload_folder):
        raise ValueError("control output directory became nonempty during verification; refusing to overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "prefill.npz"
    temporary = target.with_name(target.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle, attention=np.repeat(reference["attention"][None, :], len(samples), axis=0),
            hidden=np.repeat(reference["hidden"][None, :], len(samples), axis=0),
            sample_ids=np.asarray([row["sample_id"] for row in samples], dtype=str),
            labels=np.asarray([row["label"] for row in samples], dtype=str),
            groups=np.asarray([row["group_id"] for row in samples], dtype=str),
            splits=np.asarray([row["split"] for row in samples], dtype=str),
        )
    os.replace(temporary, target)
    _atomic_jsonl(output_dir / "metadata.jsonl", rows)
    _atomic_jsonl(output_dir / "prefill.metadata.jsonl", [{**row, "condition": "prefill"} for row in rows])
    verification_record = {
        "actual_forward_count": verification_forwards, "feature_reused_rows": len(samples) - verification_forwards,
        "unique_input_count": 1, "bit_exact_input_check_passed": True, "bit_exact_feature_check_passed": True,
        "statistical_unit": "one unique model input; manifest rows retain ground-truth pairing only",
        "verification_sample_ids": [row["sample_id"] for row in selected],
        "verification_labels": [row["label"] for row in selected], "input_tensor_identity": input_identity,
        "feature_sha256": {key: hashlib.sha256(reference[key].tobytes()).hexdigest() for key in ("attention", "hidden")},
    }
    schema = {
        "schema_version": 1, "fingerprint": fingerprint, "control_kind": CONTROL_KIND,
        "task": "visual_state", "label_order": list(TASK_LABELS["visual_state"]),
        "layers_one_based": list(config.layers), "config": config_record,
        "system_prompt": system_prompt_for_task("visual_state"), "model_runtime": runtime,
        "model_identity": provenance["model_identity"], "library_versions": provenance["library_versions"],
        "ground_truth_fields_used_as_features": [], "constant_input_control": verification_record,
        "hidden_state_location": "decoder layer output, final query position, before final model norm",
        "attention_method": "unchanged frozen extractor; last-query QK reconstruction and the original attention statistics",
        "latency_note": ("Only verification-forward rows contain measured latency. Reused rows have null timing. "
                         "This dataset has one unique model input and is not N independent forward measurements."),
        "conditions": {"prefill": {"samples": len(samples), "missing": 0,
                                  "attention_feature_names": reference["attention_names"],
                                  "hidden_feature_names": reference["hidden_names"],
                                  "attention_dimension": len(reference["attention"]),
                                  "hidden_dimension": len(reference["hidden"])}}}
    _atomic_json(output_dir / "schema.json", schema)
    run = {**provenance, "fingerprint": fingerprint, "status": "complete", "config": config_record,
           "manifest_path": str(manifest_path), "system_prompt": system_prompt_for_task("visual_state"),
           "model_runtime": runtime, "constant_input_control": verification_record,
           "artifact_sha256": {name: _sha256_file(output_dir / name) for name in ARTIFACT_FILES}}
    summary = {"fingerprint": fingerprint, "samples": len(samples), "condition_counts": {"prefill": len(samples)},
               "task": "visual_state", "label_order": list(TASK_LABELS["visual_state"]),
               "control_kind": CONTROL_KIND, "constant_input_control": verification_record,
               **verification_record, "decision_generated": 0,
               "decision_correct": 0, "output_dir": str(output_dir), "reused_existing_export": False,
               "forward_count_this_call": verification_forwards}
    _atomic_json(output_dir / "run.json", run)
    _atomic_json(output_dir / "summary.json", summary)
    return summary
