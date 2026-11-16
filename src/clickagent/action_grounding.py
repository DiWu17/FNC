"""A small, auditable attention-to-point readout shared by all gate methods.

Each patch has H features, ``log1p(P * image_attention / head_image_mass)``.
No patch coordinates, target box, label, DOM element, or absolute screen feature
enters the classifier. Boxes supervise training patch labels and select C on
validation point hit only. Predicting a point never accepts a ground-truth box.
This localizer always proposes a point; deciding whether to click is a separate
gate. Its maximum patch probability is not a calibrated action confidence.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


TRANSFORM = "log1p(patch_count * raw_attention / per_head_image_mass)"


def patch_features(raw_attention: np.ndarray, grid_shape: Sequence[int]) -> np.ndarray:
    """Return [patches, heads], with zero-mass heads kept as all-zero features."""
    raw = np.asarray(raw_attention, dtype=np.float64)
    grid = _grid(grid_shape)
    if raw.ndim != 2 or raw.shape[0] < 1 or raw.shape[1] != grid[0] * grid[1]:
        raise ValueError("raw_attention must have shape [heads, grid_height * grid_width]")
    if not np.isfinite(raw).all() or (raw < 0).any() or (raw.sum(axis=1) > 1.001).any():
        raise ValueError("raw_attention must contain raw, finite image softmax probabilities")
    mass = raw.sum(axis=1, keepdims=True)
    spatial = np.divide(raw, mass, out=np.zeros_like(raw), where=mass > 0)
    return np.log1p(spatial * raw.shape[1]).T


def _grid(value: Sequence[int]) -> tuple[int, int]:
    if len(value) != 2 or any(isinstance(item, bool) or int(item) != item or item < 1 for item in value):
        raise ValueError("grid_shape must contain positive integer height and width")
    return int(value[0]), int(value[1])


def _size(value: Sequence[int]) -> tuple[int, int]:
    width, height = _grid(value)
    return width, height


def _pixel_centers(grid_shape: Sequence[int], image_size: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    height, width = _grid(grid_shape)
    image_width, image_height = _size(image_size)
    x = np.clip(np.rint((np.arange(width) + 0.5) * image_width / width), 0, image_width - 1).astype(int)
    y = np.clip(np.rint((np.arange(height) + 0.5) * image_height / height), 0, image_height - 1).astype(int)
    return np.tile(x, height), np.repeat(y, width)


def _box(value: Sequence[float]) -> tuple[float, float, float, float]:
    if value is None or len(value) != 4:
        raise ValueError("grounder training/validation records need a target_bbox")
    result = tuple(float(item) for item in value)
    if not all(np.isfinite(result)) or not (0 <= result[0] < result[2] <= 1 and 0 <= result[1] < result[3] <= 1):
        raise ValueError("target_bbox must be normalized positive-area xyxy")
    return result


def _patch_labels(record: dict[str, Any]) -> np.ndarray:
    x, y = _pixel_centers(record["grid_shape"], record["image_size"])
    image_width, image_height = _size(record["image_size"])
    x0, y0, x1, y1 = _box(record["target_bbox"])
    return ((x / image_width >= x0) & (x / image_width < x1)
            & (y / image_height >= y0) & (y / image_height < y1)).astype(np.int64)


def point_hits_box(point: Sequence[int], image_size: Sequence[int], target_bbox: Sequence[float]) -> bool:
    width, height = _size(image_size)
    x0, y0, x1, y1 = _box(target_bbox)
    return bool(x0 <= point[0] / width < x1 and y0 <= point[1] / height < y1)


def _raw(record: dict[str, Any]) -> np.ndarray:
    value = record.get("raw_attention", record.get("raw_image_attention"))
    if value is None:
        raise ValueError("record needs raw_attention (array or NPZ path)")
    if isinstance(value, (str, Path)):
        with np.load(value, allow_pickle=False) as archive:
            key = "raw_attention" if "raw_attention" in archive else "raw_image_attention"
            return archive[key].copy()
    return np.asarray(value)


@dataclass
class AttentionPointGrounder:
    pipeline: Any
    num_heads: int
    selected_c: float
    head_order: tuple[str, ...]

    def predict_point(self, raw_attention: np.ndarray, grid_shape: Sequence[int],
                      original_size: Sequence[int]) -> dict[str, Any]:
        features = patch_features(raw_attention, grid_shape)
        if features.shape[1] != self.num_heads:
            raise ValueError(f"grounder expects {self.num_heads} heads, received {features.shape[1]}")
        positive = list(self.pipeline.classes_).index(1)
        probabilities = self.pipeline.predict_proba(features)[:, positive]
        patch = int(np.argmax(probabilities))  # Stable native raster order for exact ties.
        x, y = _pixel_centers(grid_shape, original_size)
        point = [int(x[patch]), int(y[patch])]
        return {"point": point, "x": point[0], "y": point[1], "score": float(probabilities[patch]),
                "patch_index": patch, "grid_shape": list(_grid(grid_shape)), "image_size": list(_size(original_size))}

    @classmethod
    def load(cls, path: Path | str) -> "AttentionPointGrounder":
        result = joblib.load(path)
        if not isinstance(result, cls):
            raise ValueError("file does not contain an AttentionPointGrounder")
        return result


def fit_grounder(records: Sequence[dict[str, Any]], output_dir: Path | str,
                 *, c_values: Sequence[float] = (0.01, 0.1, 1.0, 10.0), seed: int = 42) -> AttentionPointGrounder:
    """Fit train patches and choose regularization by held-out val point hit.

    Records require raw_attention/grid_shape/image_size/target_bbox/split and may
    include sample_id/group_id/label/head_order. Only train and val are accepted;
    test records are deliberately rejected. A training image with no decoded
    patch center inside its target is skipped and explicitly counted in audit.
    Validation retains such images as misses and reports the quantization ceiling.
    """
    if not records:
        raise ValueError("no grounder training records")
    choices = sorted(set(float(value) for value in c_values))
    if not choices or any(not np.isfinite(value) or value <= 0 for value in choices):
        raise ValueError("c_values must contain positive finite values")
    groups: dict[str, str] = {}
    seen = set()
    prepared = []
    num_heads = None
    head_order = None
    for index, original in enumerate(records):
        record = dict(original)
        split = record.get("split")
        if split not in {"train", "val"}:
            raise ValueError("fit_grounder accepts train/val only; test may only call predict_point")
        sample_id = str(record.get("sample_id", f"record_{index}"))
        if sample_id in seen:
            raise ValueError(f"duplicate grounder sample_id: {sample_id}")
        seen.add(sample_id)
        if record.get("group_id") is not None:
            group = str(record["group_id"])
            if groups.setdefault(group, split) != split:
                raise ValueError("grounder group leaks across train/val")
        if record.get("label") not in {None, "click", "disabled"}:
            raise ValueError("grounder fitting requires a present target; absent records are not accepted")
        raw = _raw(record)
        features = patch_features(raw, record["grid_shape"])
        if num_heads is None:
            num_heads = features.shape[1]
            fallback = ([f"layer_{layer}.head_{head}" for layer in (15, 28, 36) for head in range(32)]
                        if num_heads == 96 else [f"head_{head}" for head in range(num_heads)])
            head_order = tuple(record.get("head_order", fallback))
            if len(head_order) != num_heads:
                raise ValueError("head_order length differs from raw head count")
        if features.shape[1] != num_heads or tuple(record.get("head_order", head_order)) != head_order:
            raise ValueError("grounder head count/order differs between records")
        labels = _patch_labels(record)
        prepared.append((record, sample_id, features, labels))
    train = [row for row in prepared if row[0]["split"] == "train"]
    val = [row for row in prepared if row[0]["split"] == "val"]
    if not train or not val:
        raise ValueError("grounder requires nonempty train and val records")
    skipped = [sample_id for _, sample_id, _, labels in train if not labels.any()]
    usable = [row for row in train if row[3].any()]
    if not usable:
        raise ValueError("no train target contains a decoded patch center; increase image resolution")
    x_train = np.concatenate([features for _, _, features, _ in usable])
    y_train = np.concatenate([labels for _, _, _, labels in usable])
    if len(np.unique(y_train)) != 2:
        raise ValueError("grounder training patches need both target and nontarget examples")
    best = None
    search = []
    for c_value in choices:
        pipeline = Pipeline([("scale", StandardScaler()),
                             ("classifier", LogisticRegression(C=c_value, class_weight="balanced", solver="lbfgs",
                                                               max_iter=2000, random_state=seed))])
        pipeline.fit(x_train, y_train)
        candidate = AttentionPointGrounder(pipeline, int(num_heads), c_value, head_order)
        hits = [point_hits_box(candidate.predict_point(_raw(record), record["grid_shape"], record["image_size"])["point"],
                               record["image_size"], record["target_bbox"]) for record, _, _, _ in val]
        search.append({"C": c_value, "val_point_hits": int(sum(hits)), "val_samples": len(val),
                       "val_point_hit_rate": float(np.mean(hits))})
        key = (sum(hits), -c_value)
        if best is None or key > best[0]:
            best = key, candidate
    grounder = best[1]
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("grounder output directory must be empty to avoid replacing a fitted readout")
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(grounder, output / "grounder.joblib")
    audit = {
        "method": "patchwise logistic regression", "transform": TRANSFORM, "features": int(num_heads),
        "head_order": list(head_order), "selected_C": grounder.selected_c, "validation_search": search,
        "fit_split": "train only", "selection_split": "val point hit only; lower C wins ties",
        "test_used_for_fit_or_selection": False, "absolute_coordinate_features": False,
        "ground_truth_use": "train patch-center labels and validation point-hit scoring only",
        "n_train_records": len(train), "n_train_records_used": len(usable), "n_train_patches": len(y_train),
        "n_train_positive_patches": int(y_train.sum()), "skipped_train_no_positive_patch": skipped,
        "n_val_records": len(val), "val_quantization_oracle_hit_rate": float(np.mean([row[3].any() for row in val])),
        "sample_ids": {"train": [row[1] for row in train], "val": [row[1] for row in val]},
        "record_label_counts": dict(Counter(str(row[0].get("label", "unspecified")) for row in prepared)),
        "limitation": "Always proposes one patch-center point; score is not a calibrated click gate or real task-success probability.",
    }
    (output / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return grounder
