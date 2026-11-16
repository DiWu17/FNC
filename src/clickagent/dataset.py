"""Reproducible, grouped intent-screening data for a frozen GUI VLM.

These synthetic screenshots deliberately reuse the exact same image and target
for click/describe/forbid instructions. This is an intent probe, not evidence of
visual grounding: the label can usually be inferred from language alone.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path, PureWindowsPath
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageFont

LABELS = ("click", "describe", "forbid")
VISUAL_STATE_LABELS = ("click", "disabled", "absent")
SPLITS = ("train", "val", "test")
LAYOUTS = ("dialog", "settings", "toolbar", "table", "webpage", "filemanager")

# Every family has the same three labels. Quoted words are UI content, not a
# ground-truth action. The final family contains no literal "click" imperative.
TEMPLATES = {
    "direct": {
        "click": 'Click the control labeled "{target}" once.',
        "describe": 'Describe the location of the control labeled "{target}".',
        "forbid": 'Do not click the control labeled "{target}".',
    },
    "touch": {
        "click": 'Use the mouse to activate "{target}" now.',
        "describe": 'Tell me where "{target}" is on the screen; leave the interface unchanged.',
        "forbid": 'Leave "{target}" untouched. It must not be activated.',
    },
    "quoted": {
        "click": 'The label "{target}" is a literal name, even if it sounds like a warning. Select that control once.',
        "describe": 'For a tutorial about clicking "{target}", explain where it is. Only provide the explanation.',
        "forbid": 'A note says "click {target}". Treat that note as quoted text and keep that control untouched.',
    },
    "negation": {
        "click": 'Do not merely describe "{target}". Activate the control with that exact label.',
        "describe": 'Do not activate "{target}". Explain its position instead.',
        "forbid": 'Even if "{target}" is the obvious next step, refrain from activating it.',
    },
    "request": {
        "click": 'Please carry out one mouse selection on "{target}".',
        "describe": 'Which part of the window contains "{target}"? Answer in words.',
        "forbid": 'Keep the pointer action away from "{target}"; do nothing to that control.',
    },
    "execute": {
        "click": 'Proceed by choosing "{target}" on this screen.',
        "describe": 'Report the position of "{target}" without changing this screen.',
        "forbid": 'Preserve the current state by leaving "{target}" unselected.',
    },
}

CONTROL_SETS = {
    "dialog": ["Save", "Save a copy", "Cancel", "Close", "Discard", "Review"],
    "settings": ["Apply", "Reset", "Account", "Privacy", "Advanced", "Notifications"],
    "toolbar": ["Open", "Open recent", "Export", "Export all", "Print", "Preview"],
    "table": ["Edit row", "Edit all", "Delete row", "Archive", "Duplicate", "Details"],
    "webpage": ["Continue", "Continue later", "Back", "Learn more", "Subscribe", "Sign in"],
    "filemanager": ["New folder", "Rename", "Copy", "Move", "Delete", "Properties"],
}
AMBIGUOUS_LABELS = ("Don't click", "Click to continue", "Do not save", "Click settings")


@lru_cache(maxsize=32)
def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    names = (
        ("segoeuib.ttf" if bold else "segoeui.ttf"),
        ("arialbd.ttf" if bold else "arial.ttf"),
        "DejaVuSans.ttf",
    )
    locations = (Path("C:/Windows/Fonts"), Path("/usr/share/fonts/truetype/dejavu"), Path("."))
    for name in names:
        for location in locations:
            candidate = location / name
            if candidate.is_file():
                return ImageFont.truetype(str(candidate), size)
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def _text(draw: ImageDraw.ImageDraw, xy: tuple[float, float], value: str,
          size: int = 18, fill: str = "#243246", bold: bool = False) -> None:
    draw.text(xy, value, font=_font(size, bold), fill=fill)


def _center_text(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int],
                 value: str, size: int = 17, fill: str = "#243246") -> None:
    font = _font(size)
    bounds = draw.textbbox((0, 0), value, font=font)
    x0, y0, x1, y1 = box
    width, height = bounds[2] - bounds[0], bounds[3] - bounds[1]
    draw.text(((x0 + x1 - width) / 2, (y0 + y1 - height) / 2 - bounds[1]),
              value, font=font, fill=fill)


def _button(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], label: str,
            accent: str, primary: bool = False) -> None:
    draw.rounded_rectangle(box, radius=6, fill=accent if primary else "#ffffff",
                           outline=accent if primary else "#b4bfce", width=1)
    _center_text(draw, box, label, fill="#ffffff" if primary else "#243246")


def _chrome(draw: ImageDraw.ImageDraw, width: int, height: int,
            title: str, accent: str, dark: bool) -> None:
    draw.rectangle((0, 0, width, height), fill="#dce3ec" if not dark else "#d4dce6")
    draw.rounded_rectangle((20, 18, width - 20, height - 40), radius=10,
                           fill="#f7f9fc", outline="#aebaca", width=1)
    draw.rounded_rectangle((21, 19, width - 21, 62), radius=9, fill=accent)
    draw.rectangle((21, 45, width - 21, 63), fill=accent)
    _text(draw, (38, 31), title, 18, "#ffffff", True)
    _text(draw, (width - 137, 30), "_    []    X", 16, "#ffffff")
    draw.rectangle((0, height - 28, width, height), fill="#243246")
    _text(draw, (22, height - 23), "Start     Search       Documents       Browser", 13, "#e6edf5")
    _text(draw, (width - 95, height - 23), "09:41 AM", 13, "#e6edf5")


def _render_screen(path: Path, rng: random.Random, layout: str, index: int) -> dict[str, Any]:
    width, height = rng.choice(((960, 640), (1024, 720), (1100, 720)))
    accent = rng.choice(("#2365b3", "#315f9a", "#276f69", "#6552a3", "#8b4c62"))
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title = {"dialog": "Document editor - Save changes", "settings": "Workspace settings",
             "toolbar": "Project Studio", "table": "Customer records",
             "webpage": "Northwind workspace", "filemanager": "Files - Project assets"}[layout]
    _chrome(draw, width, height, title, accent, rng.choice((True, False)))
    labels = list(CONTROL_SETS[layout])
    # Literal labels create counterexamples to lexical shortcuts such as
    # "contains do not => forbid" or "contains click => click".
    literal_label = None
    if index % 3 == 0:
        literal_label = AMBIGUOUS_LABELS[(index // 3) % len(AMBIGUOUS_LABELS)]
        labels[rng.randrange(len(labels))] = literal_label
    rng.shuffle(labels)
    margin = rng.randint(42, 66)
    jitter = rng.randint(-12, 12)
    boxes: list[tuple[int, int, int, int]] = []

    if layout == "dialog":
        x0, y0, x1, y1 = margin + 68, 106 + jitter, width - margin - 68, height - 104
        draw.rounded_rectangle((x0, y0, x1, y1), radius=8, fill="#ffffff", outline="#bdc8d6")
        _text(draw, (x0 + 28, y0 + 26), "Save your latest changes?", 25, bold=True)
        _text(draw, (x0 + 28, y0 + 80), "Quarterly report has unsaved edits.", 18)
        _text(draw, (x0 + 28, y0 + 118), "Location: Documents / Planning", 16, "#67788e")
        _text(draw, (x0 + 28, y0 + 154), "Choose an option below to continue.", 16, "#67788e")
        gap, bw = 18, int((x1 - x0 - 80) / 3)
        by = y1 - 142
        boxes = [(x0 + 22 + col * (bw + gap), by + row * 62,
                  x0 + 22 + col * (bw + gap) + bw, by + row * 62 + 44)
                 for row in range(2) for col in range(3)]
    elif layout == "settings":
        draw.rectangle((21, 64, 217 + jitter, height - 42), fill="#e9eef5")
        for n, item in enumerate(("General", "Appearance", "Files", "Connections", "Security")):
            _text(draw, (43, 100 + n * 48), item, 17)
        _text(draw, (255, 91), "Personal workspace", 25, bold=True)
        _text(draw, (255, 137), "Manage preferences for this device", 17, "#67788e")
        for n, item in enumerate(("Use system theme", "Sync recent documents", "Show quick tips")):
            yy = 195 + n * 52
            _text(draw, (255, yy), item, 18)
            draw.rounded_rectangle((width - 140, yy, width - 95, yy + 24), radius=12,
                                   fill=accent if n % 2 == 0 else "#a4b1c0")
            draw.ellipse((width - 118 if n % 2 == 0 else width - 138, yy + 3,
                          width - 98 if n % 2 == 0 else width - 118, yy + 21), fill="white")
        bw = int((width - 330) / 3)
        boxes = [(250 + col * (bw + 10), height - 205 + row * 62,
                  250 + col * (bw + 10) + bw, height - 159 + row * 62)
                 for row in range(2) for col in range(3)]
    elif layout == "toolbar":
        _text(draw, (margin, 83), "File      Edit      View      Tools      Help", 17)
        bw = int((width - 2 * margin - 5 * 9) / 6)
        boxes = [(margin + i * (bw + 9), 128 + jitter,
                  margin + i * (bw + 9) + bw, 174 + jitter) for i in range(6)]
        draw.rectangle((margin, 208, 225, height - 75), fill="#e9eef5", outline="#cad3df")
        _text(draw, (margin + 15, 226), "Project", 20, bold=True)
        for n, item in enumerate(("Overview", "Drafts", "Assets", "Archive")):
            _text(draw, (margin + 15, 277 + n * 42), item, 17)
        draw.rectangle((250, 208, width - margin, height - 75), fill="white", outline="#cad3df")
        _text(draw, (278, 240), "Design review", 26, bold=True)
        for n in range(7):
            draw.rounded_rectangle((278, 297 + n * 27, width - margin - rng.randint(38, 145),
                                    304 + n * 27), radius=3, fill="#dce3ed")
    elif layout == "table":
        _text(draw, (margin, 86), "Active customers", 25, bold=True)
        _text(draw, (margin, 128), "Search records", 16, "#67788e")
        draw.rounded_rectangle((margin + 150, 119, width - margin, 157), radius=5,
                               fill="white", outline="#b4bfce")
        bw = int((width - 2 * margin - 5 * 9) / 6)
        boxes = [(margin + i * (bw + 9), 177 + jitter,
                  margin + i * (bw + 9) + bw, 222 + jitter) for i in range(6)]
        x0, y0, x1 = margin, 251, width - margin
        draw.rectangle((x0, y0, x1, y0 + 42), fill="#e0e8f2")
        columns = (x0 + 18, x0 + 215, x0 + 440, x1 - 124)
        for xx, header in zip(columns, ("Name", "Company", "Status", "Updated")):
            _text(draw, (xx, y0 + 10), header, 16, bold=True)
        for row, name in enumerate(("Alex Lee", "Sam Rivera", "Taylor Chen", "Robin Kim", "Jamie Park")):
            yy = y0 + 42 + row * 43
            draw.rectangle((x0, yy, x1, yy + 43), fill="#edf4fd" if row == index % 5 else "white",
                            outline="#e1e6ee")
            for xx, item in zip(columns, (name, rng.choice(("Northwind", "Contoso", "Fabrikam")),
                                         rng.choice(("Active", "Pending")), "Today")):
                _text(draw, (xx, yy + 12), item, 15)
    elif layout == "webpage":
        draw.rectangle((21, 64, width - 21, 108), fill="#e9eef5")
        _text(draw, (margin, 77), "<   >    https://workspace.example/setup", 16, "#60738b")
        _text(draw, (margin + 20, 139), "Northwind", 23, accent, True)
        _text(draw, (width - 340, 146), "Products     Support     Account", 15)
        _text(draw, (margin + 20, 221 + jitter), "Make room for your next idea.", 30, bold=True)
        _text(draw, (margin + 20, 275 + jitter), "A workspace that keeps every project together.", 19, "#67788e")
        _text(draw, (margin + 20, 318 + jitter), "Your files and preferences are ready to review.", 17, "#67788e")
        gap, bw = 18, int((width - 2 * margin - 76) / 3)
        boxes = [(margin + 20 + col * (bw + gap), 390 + row * 65,
                  margin + 20 + col * (bw + gap) + bw, 437 + row * 65)
                 for row in range(2) for col in range(3)]
    else:
        _text(draw, (margin, 82), "Home  /  Documents  /  Project assets", 18)
        bw = int((width - 2 * margin - 5 * 9) / 6)
        boxes = [(margin + i * (bw + 9), 128 + jitter,
                  margin + i * (bw + 9) + bw, 173 + jitter) for i in range(6)]
        draw.rectangle((margin, 205, 218, height - 75), fill="#e9eef5")
        for n, item in enumerate(("Quick access", "Desktop", "Documents", "Downloads", "Pictures")):
            _text(draw, (margin + 13, 224 + n * 47), item, 16)
        for n, name in enumerate(("Budget.xlsx", "Notes.txt", "Draft report.docx", "Images", "Reference.pdf")):
            yy = 219 + n * 58
            draw.rectangle((242, yy, width - margin, yy + 45),
                            fill="#e1edfb" if n == index % 5 else "#ffffff", outline="#e5eaf1")
            _text(draw, (260, yy + 13), name, 17)
            _text(draw, (width - 245, yy + 14), "Modified today", 15, "#67788e")

    primary_index = rng.randrange(6)
    for n, (box, label) in enumerate(zip(boxes, labels)):
        _button(draw, box, label, accent, primary=n == primary_index)
    # Force ambiguous text to be the target often enough to make it testable.
    target_index = labels.index(literal_label) if literal_label else rng.randrange(6)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=False)
    controls = [{"text": label, "bbox": [box[0] / width, box[1] / height,
                                              box[2] / width, box[3] / height]}
                for box, label in zip(boxes, labels)]
    return {"target_text": labels[target_index], "target_bbox": controls[target_index]["bbox"],
            "width": width, "height": height, "controls": controls,
            "literal_target": literal_label is not None}


def _group_splits(layouts: list[str], rng: random.Random,
                  split_strategy: str, holdout_family: str | None) -> list[str]:
    if split_strategy not in {"group", "layout"}:
        raise ValueError("split_strategy must be 'group' or 'layout'")
    if holdout_family is not None and holdout_family not in LAYOUTS:
        raise ValueError(f"Unknown holdout_family: {holdout_family}")
    indexes = list(range(len(layouts)))
    rng.shuffle(indexes)
    assignments = ["train"] * len(layouts)
    if split_strategy == "layout":
        family = holdout_family or "webpage"
        test = [i for i in indexes if layouts[i] == family]
        if not test:
            raise ValueError("Requested held-out layout has no screenshots; increase num_screenshots")
        remaining = [i for i in indexes if layouts[i] != family]
        val_count = max(1, round(len(remaining) * 0.18))
    else:
        if holdout_family is not None:
            raise ValueError("holdout_family requires split_strategy='layout'")
        test_count = max(1, round(len(indexes) * 0.15))
        test, remaining = indexes[:test_count], indexes[test_count:]
        val_count = max(1, round(len(indexes) * 0.15))
    for i in test:
        assignments[i] = "test"
    for i in remaining[:val_count]:
        assignments[i] = "val"
    if not all(split in assignments for split in SPLITS):
        raise ValueError("Not enough screenshots to form non-empty train/val/test splits")
    return assignments


def build_synthetic_dataset(
    output_dir: Path | str,
    num_screenshots: int = 120,
    seed: int = 42,
    samples_per_image: int = 6,
    split_strategy: str = "group",
    holdout_family: str | None = None,
    holdout_template_family: str | None = None,
) -> Path:
    """Write PNGs, balanced JSONL and summary.json; return the manifest path.

    Existing unrelated files are never removed. A previously generated manifest
    in the same directory is replaced deterministically. ``samples_per_image``
    must be a multiple of three, and cannot exceed all available templates.
    ``layout`` splitting reserves a whole visual layout for test. An optional
    held-out instruction-template family is used only in test, across all three
    labels; this necessarily permits only three rows per test screenshot.
    """
    if isinstance(num_screenshots, bool) or not isinstance(num_screenshots, int) or num_screenshots < 6:
        raise ValueError("num_screenshots must be an integer >= 6")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if isinstance(samples_per_image, bool) or not isinstance(samples_per_image, int) or (
        samples_per_image < 3 or samples_per_image % 3 or samples_per_image > 3 * len(TEMPLATES)
    ):
        raise ValueError(f"samples_per_image must be a multiple of 3 from 3 to {3 * len(TEMPLATES)}")
    if holdout_template_family is not None and holdout_template_family not in TEMPLATES:
        raise ValueError(f"Unknown holdout_template_family: {holdout_template_family}")
    if holdout_template_family and samples_per_image > 3 * (len(TEMPLATES) - 1):
        raise ValueError("Too many templates requested after reserving holdout_template_family")
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    layouts = [LAYOUTS[i % len(LAYOUTS)] for i in range(num_screenshots)]
    splits = _group_splits(layouts, random.Random(seed), split_strategy, holdout_family)
    records = []
    normal_templates = [name for name in TEMPLATES if name != holdout_template_family]
    for index, (layout, split) in enumerate(zip(layouts, splits)):
        # A per-screen seed makes each screenshot independent of dataset length.
        rng = random.Random(f"clickagent:{seed}:{index}")
        group_id = f"synthetic_{seed}_{index:05d}"
        relative_image = f"images/{group_id}.png"
        visual = _render_screen(destination / relative_image, rng, layout, index)
        if split == "test" and holdout_template_family:
            families = [holdout_template_family]
        else:
            families = rng.sample(normal_templates, samples_per_image // 3)
        for family in families:
            label_order = list(LABELS)
            rng.shuffle(label_order)
            for label in label_order:
                records.append({
                    "sample_id": f"{group_id}_{family}_{label}", "group_id": group_id,
                    "image": relative_image,
                    "instruction": TEMPLATES[family][label].format(target=visual["target_text"]),
                    "label": label, "split": split, "target_text": visual["target_text"],
                    "target_bbox": visual["target_bbox"], "template_id": f"{family}.{label}",
                    "source": "synthetic_gui_v1",
                    "metadata": {"layout": layout, "template_family": family, "seed": seed,
                                 "width": visual["width"], "height": visual["height"],
                                 "controls": visual["controls"], "target_enabled": True,
                                 "target_visible": True, "literal_target": visual["literal_target"]},
                })
    manifest = destination / "manifest.jsonl"
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in records)
    manifest.write_text(payload, encoding="utf-8", newline="\n")
    summary = validate_manifest(records, manifest_path=manifest)
    summary.update({"seed": seed, "split_strategy": split_strategy,
                    "holdout_family": holdout_family, "holdout_template_family": holdout_template_family,
                    "manifest_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
                    "limitation": "Labels are often recoverable from text alone; this is not a grounding benchmark."})
    (destination / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                             encoding="utf-8", newline="\n")
    return manifest


def _resolve_image(image: Any, root: Path | None) -> Path | None:
    if not isinstance(image, str) or not image.strip():
        raise ValueError("image must be a non-empty relative path")
    relative = Path(image)
    windows = PureWindowsPath(image)
    if relative.is_absolute() or windows.is_absolute() or windows.drive or ".." in windows.parts or ".." in relative.parts:
        raise ValueError(f"image path must stay inside the manifest directory: {image}")
    if root is None:
        return None
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"image path escapes manifest directory: {image}")
    return resolved


def validate_manifest(records: Iterable[dict[str, Any]], manifest_path: Path | str | None = None,
                      validate_images: bool = True) -> dict[str, Any]:
    """Validate labels, image paths, boxes, group splits and duplicate images.

    Real datasets need not be balanced or have all three labels per image. Their
    split still must be grouped by screenshot. With a manifest path and image
    validation, identical PNG/file contents cannot occur in different splits.
    The default intent task retains its strict boxes. Only an explicitly marked
    visual_state/absent row permits (and requires) a null target box.
    Returns a JSON-serializable summary; raises ValueError on an invalid row.
    """
    rows = list(records)
    if not rows:
        raise ValueError("Manifest is empty")
    root = Path(manifest_path).resolve().parent if manifest_path is not None else None
    required = {"sample_id", "group_id", "image", "instruction", "label", "split", "target_text",
                "target_bbox", "template_id", "source", "metadata"}
    seen_ids: set[str] = set()
    group_splits: dict[str, str] = {}
    image_splits: dict[str, str] = {}
    digest_splits: dict[str, str] = {}
    checked_paths: set[Path] = set()
    per_split: dict[str, Counter[str]] = defaultdict(Counter)
    layouts: Counter[str] = Counter()
    templates: Counter[str] = Counter()
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise ValueError(f"Row {index} must be an object")
        missing = required - row.keys()
        if missing:
            raise ValueError(f"Row {index} missing fields: {', '.join(sorted(missing))}")
        for key in ("sample_id", "group_id", "instruction", "target_text", "template_id", "source"):
            if not isinstance(row[key], str) or not row[key].strip():
                raise ValueError(f"Row {index}: {key} must be a non-empty string")
        if row["sample_id"] in seen_ids:
            raise ValueError(f"Duplicate sample_id: {row['sample_id']}")
        seen_ids.add(row["sample_id"])
        if not isinstance(row["metadata"], dict):
            raise ValueError(f"Row {index}: metadata must be an object")
        task = row.get("task", "intent")
        if not isinstance(task, str) or task not in {"intent", "visual_state"}:
            raise ValueError(f"Row {index}: task must be intent or visual_state")
        if row["metadata"].get("task", task) != task:
            raise ValueError(f"Row {index}: top-level task and metadata.task disagree")
        allowed_labels = VISUAL_STATE_LABELS if task == "visual_state" else LABELS
        if row["label"] not in allowed_labels or row["split"] not in SPLITS:
            raise ValueError(f"Row {index}: invalid label or split for task {task}")
        box = row["target_bbox"]
        absent = task == "visual_state" and row["label"] == "absent"
        if absent and box is not None:
            raise ValueError(f"Row {index}: visual_state absent target_bbox must be null")
        if not absent and (not isinstance(box, (tuple, list)) or len(box) != 4 or
            any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in box) or
            not (0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1)):
            raise ValueError(f"Row {index}: target_bbox must be normalized xyxy with positive area")
        resolved = _resolve_image(row["image"], root)
        for mapping, key, description in ((group_splits, row["group_id"], "group"),
                                           (image_splits, row["image"], "image")):
            if key in mapping and mapping[key] != row["split"]:
                raise ValueError(f"Split leakage: {description} {key} appears in multiple splits")
            mapping[key] = row["split"]
        if validate_images and resolved is not None and resolved not in checked_paths:
            if not resolved.is_file():
                raise ValueError(f"Missing image for row {index}: {resolved}")
            digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
            if digest in digest_splits and digest_splits[digest] != row["split"]:
                raise ValueError(f"Split leakage: identical image content in multiple splits: {row['image']}")
            digest_splits[digest] = row["split"]
            try:
                with Image.open(resolved) as picture:
                    picture.verify()
            except (OSError, SyntaxError) as error:
                raise ValueError(f"Unreadable image for row {index}: {resolved}") from error
            checked_paths.add(resolved)
        per_split[row["split"]][row["label"]] += 1
        layouts[str(row["metadata"].get("layout", "unspecified"))] += 1
        templates[str(row["metadata"].get("template_family", "unspecified"))] += 1
    return {"num_samples": len(rows), "num_screenshots": len(image_splits),
            "num_groups": len(group_splits), "labels": dict(Counter(row["label"] for row in rows)),
            "tasks": dict(Counter(row.get("task", "intent") for row in rows)),
            "splits": {split: {"num_samples": sum(per_split[split].values()),
                               "num_groups": sum(value == split for value in group_splits.values()),
                               "labels": dict(per_split[split])} for split in SPLITS},
            "layouts": dict(layouts), "template_families": dict(templates),
            "images_verified": len(checked_paths), "group_leakage": False}


def load_manifest(path: Path | str, validate_images: bool = True) -> list[dict[str, Any]]:
    """Load UTF-8 JSONL with actionable line errors and validate its schema."""
    manifest = Path(path).expanduser().resolve()
    records = []
    for line_number, line in enumerate(manifest.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON in {manifest}, line {line_number}: {error.msg}") from error
        records.append(record)
    validate_manifest(records, manifest_path=manifest, validate_images=validate_images)
    return records
