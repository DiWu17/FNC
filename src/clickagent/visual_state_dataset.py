"""Pixel-paired visual executability data with a constant Chinese instruction.

Only a target button's pixels change inside a counterfactual scenario. Labels,
control annotations, and the absent target's counterfactual box are audit data,
never model inputs. There is no dependence on a live desktop or a GPU.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from .dataset import load_manifest, validate_manifest

INSTRUCTION = "点击保存"
TARGET_TEXT = "保存"
STATES = ("enabled", "disabled", "absent")
STATE_LABELS = {"enabled": "click", "disabled": "disabled", "absent": "absent"}
SEEN_LAYOUTS = ("dialog", "settings", "toolbar", "webpage")
OOD_LAYOUTS = ("table", "filemanager")
LAYOUTS = SEEN_LAYOUTS + OOD_LAYOUTS
STYLE_FAMILIES = ("fluent_blue", "fluent_teal", "fluent_purple", "outline", "neutral_gray", "classic")


@lru_cache(maxsize=1)
def _cjk_font_path() -> str:
    candidates = (
        "C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/NotoSansSC-VF.ttf",
        "C:/Windows/Fonts/simhei.ttf", "C:/Windows/Fonts/simsun.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    )
    for value in candidates:
        if Path(value).is_file():
            return value
    raise RuntimeError("A Chinese font is required: install Microsoft YaHei or Noto Sans CJK/SC.")


@lru_cache(maxsize=32)
def _font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(_cjk_font_path(), size)


def _text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], value: str,
          size: int = 22, color: str = "#283445") -> None:
    draw.text(xy, value, font=_font(size), fill=color)


def _theme(style: str, dark: bool) -> dict[str, str | bool | int]:
    accents = {"fluent_blue": "#246fbd", "fluent_teal": "#147975", "fluent_purple": "#7252a6",
               "outline": "#416da0", "neutral_gray": "#667281", "classic": "#32659a"}
    return {"dark": dark, "style": style, "accent": accents[style],
            "desktop": "#111923" if dark else "#dce4ed",
            "surface": "#202a36" if dark else "#f6f8fb",
            "card": "#273342" if dark else "#ffffff",
            "panel": "#192430" if dark else "#e9eef5",
            "line": "#465364" if dark else "#c3cddd",
            "text": "#edf2f8" if dark else "#283445",
            "muted": "#b4c2d3" if dark else "#65758b"}


def _button(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str,
            enabled: bool, theme: dict[str, Any], role: str = "secondary", font_size: int | None = None) -> None:
    """Draw inside an exclusive-end xyxy box, including all border pixels."""
    dark = theme["dark"]
    classic = theme["style"] == "classic"
    if enabled:
        if role == "primary" and theme["style"] not in {"outline", "neutral_gray", "classic"}:
            fill, foreground, border = theme["accent"], "#ffffff", theme["accent"]
        elif role == "neutral" or theme["style"] in {"neutral_gray", "classic"}:
            fill, foreground, border = (("#445163", "#f3f6fa", "#9baabd") if dark
                                        else ("#e1e5eb", "#263142", "#7f8b9c"))
        else:
            fill, foreground, border = (("#2b394a", "#f0f4fa", "#92a4bb") if dark
                                        else ("#ffffff", "#283b54", "#8c9fb8"))
    else:
        # Conventional disabled border/text contrast, not an explicit status cue.
        fill, foreground, border = (("#2d3743", "#738091", "#465261") if dark
                                    else ("#edf0f4", "#a5acb7", "#d5dbe3"))
    x0, y0, x1, y1 = box
    draw.rounded_rectangle((x0, y0, x1 - 1, y1 - 1), radius=1 if classic else 6,
                           fill=fill, outline=border, width=2 if theme["style"] == "outline" else 1)
    if classic:
        draw.line((x0 + 2, y0 + 2, x1 - 3, y0 + 2), fill="#718093" if dark else "#ffffff")
        draw.line((x0 + 2, y0 + 2, x0 + 2, y1 - 3), fill="#718093" if dark else "#ffffff")
    font = _font(font_size if font_size is not None else (25 if text == TARGET_TEXT else 21))
    bounds = draw.textbbox((0, 0), text, font=font)
    text_width, text_height = bounds[2] - bounds[0], bounds[3] - bounds[1]
    draw.text(((x0 + x1 - text_width) / 2 - bounds[0], (y0 + y1 - text_height) / 2 - bounds[1]),
              text, font=font, fill=foreground)


def _scenario_plan(num_scenarios: int, seed: int) -> list[dict[str, str]]:
    if isinstance(num_scenarios, bool) or not isinstance(num_scenarios, int) or num_scenarios < 8:
        raise ValueError("num_scenarios must be an integer >= 8 to keep all splits and both OOD layouts")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    n_ood = max(2, round(num_scenarios * 0.10))
    n_iid = max(1, round(num_scenarios * 0.10))
    n_val = max(1, round(num_scenarios * 0.20))
    n_train = num_scenarios - n_val - n_iid - n_ood
    if n_train < len(SEEN_LAYOUTS):
        n_val -= len(SEEN_LAYOUTS) - n_train
        n_train = len(SEEN_LAYOUTS)
    if n_val < 1:
        raise ValueError("Not enough scenarios for nonempty train/val/test")
    rng = random.Random(seed)
    plan = []
    for split, domain, count, layouts in (("train", "iid", n_train, SEEN_LAYOUTS),
                                          ("val", "iid", n_val, SEEN_LAYOUTS),
                                          ("test", "iid", n_iid, SEEN_LAYOUTS),
                                          ("test", "ood", n_ood, OOD_LAYOUTS)):
        order = list(layouts)
        rng.shuffle(order)
        entries = [{"split": split, "domain": domain, "layout": order[i % len(order)]} for i in range(count)]
        rng.shuffle(entries)
        plan.extend(entries)
    rng.shuffle(plan)
    return plan


def _norm(box: tuple[int, int, int, int], width: int, height: int) -> list[float]:
    return [box[0] / width, box[1] / height, box[2] / width, box[3] / height]


def _base_screen(rng: random.Random, layout: str, scenario: int) -> tuple[Image.Image, list[tuple[int, int, int, int]], dict[str, Any]]:
    width, height = rng.choice(((960, 640), (1024, 704)))
    style = rng.choice(STYLE_FAMILIES)
    dark = bool(rng.getrandbits(1))
    theme = _theme(style, dark)
    image = Image.new("RGB", (width, height), theme["desktop"])
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((18, 16, width - 19, height - 39), radius=10, fill=theme["surface"], outline=theme["line"])
    draw.rounded_rectangle((19, 17, width - 20, 65), radius=9, fill=theme["accent"])
    draw.rectangle((19, 44, width - 20, 65), fill=theme["accent"])
    titles = {"dialog": "文档编辑器", "settings": "工作区设置", "toolbar": "项目工作台",
              "webpage": "团队空间", "table": "项目记录", "filemanager": "文件管理器"}
    _text(draw, (37, 26), titles[layout], 23, "#ffffff")
    _text(draw, (width - 118, 28), "—   □   ×", 19, "#ffffff")
    draw.rectangle((0, height - 27, width, height), fill="#1b2635")
    _text(draw, (24, height - 26), "开始     搜索       文档       浏览器", 15, "#d8e2ef")
    _text(draw, (width - 92, height - 26), "09:41", 15, "#d8e2ef")
    margin = rng.randint(42, 62)
    dy = rng.randint(-8, 8)
    document = f"项目文档 {scenario + 1:04d}"
    boxes: list[tuple[int, int, int, int]]
    if layout == "dialog":
        left, top, right, bottom = margin + 60, 103 + dy, width - margin - 60, height - 92
        draw.rounded_rectangle((left, top, right, bottom), radius=9, fill=theme["card"], outline=theme["line"])
        _text(draw, (left + 26, top + 23), "文档操作", 29, theme["text"])
        _text(draw, (left + 26, top + 79), document, 24, theme["text"])
        _text(draw, (left + 26, top + 124), "位置：工作区 / 文档", 21, theme["muted"])
        _text(draw, (left + 26, top + 164), "选择下方工具继续处理当前文档。", 20, theme["muted"])
        gap, button_width = 15, (right - left - 72) // 3
        boxes = [(left + 20 + c * (button_width + gap), bottom - 133 + r * 65,
                  left + 20 + c * (button_width + gap) + button_width, bottom - 79 + r * 65)
                 for r in range(2) for c in range(3)]
    elif layout == "settings":
        draw.rectangle((19, 66, 211 + dy, height - 40), fill=theme["panel"])
        for n, text in enumerate(("通用", "外观", "通知", "同步", "安全")):
            _text(draw, (44, 100 + n * 47), text, 22, theme["text"])
        _text(draw, (246, 91), "工作区偏好", 29, theme["text"])
        _text(draw, (246, 143), document, 21, theme["muted"])
        for n, text in enumerate(("使用系统主题", "显示最近文档", "显示工具提示")):
            yy = 198 + n * 50
            _text(draw, (246, yy), text, 22, theme["text"])
            on = bool(rng.getrandbits(1))
            xx = width - 148
            draw.rounded_rectangle((xx, yy + 2, xx + 49, yy + 29), radius=13,
                                   fill=theme["accent"] if on else theme["line"])
            knob = xx + 25 if on else xx + 4
            draw.ellipse((knob, yy + 6, knob + 19, yy + 25), fill="#f4f7fb")
        button_width = (width - 324) // 3
        boxes = [(244 + c * (button_width + 12), height - 191 + r * 65,
                  244 + c * (button_width + 12) + button_width, height - 137 + r * 65)
                 for r in range(2) for c in range(3)]
    elif layout == "webpage":
        draw.rectangle((19, 66, width - 20, 106), fill=theme["panel"])
        _text(draw, (margin, 70), "‹   ›   workspace.example / documents", 19, theme["muted"])
        _text(draw, (margin + 13, 135), "团队空间", 26, theme["text"])
        _text(draw, (width - 305, 142), "工作区    支持    账户", 20, theme["muted"])
        _text(draw, (margin + 13, 212 + dy), "让项目井然有序", 34, theme["text"])
        _text(draw, (margin + 13, 272 + dy), document, 25, theme["text"])
        _text(draw, (margin + 13, 318 + dy), "在同一个空间整理资料与协作内容。", 22, theme["muted"])
        button_width = (width - 2 * margin - 57) // 3
        boxes = [(margin + 12 + c * (button_width + 16), 395 + r * 65,
                  margin + 12 + c * (button_width + 16) + button_width, 449 + r * 65)
                 for r in range(2) for c in range(3)]
    else:
        if layout == "toolbar":
            _text(draw, (margin, 83), "文件     编辑     视图     工具     帮助", 22, theme["text"])
        elif layout == "table":
            _text(draw, (margin, 78), "项目清单", 27, theme["text"])
            _text(draw, (width - 292, 84), "搜索记录  ▢", 21, theme["muted"])
        else:
            _text(draw, (margin, 83), "主页  /  工作区  /  资料", 23, theme["text"])
        button_width = (width - 2 * margin - 5 * 10) // 6
        boxes = [(margin + c * (button_width + 10), 132 + dy,
                  margin + c * (button_width + 10) + button_width, 186 + dy) for c in range(6)]
        if layout == "toolbar":
            draw.rectangle((margin, 220, 212, height - 72), fill=theme["panel"])
            for n, text in enumerate(("项目", "概览", "文稿", "附件", "归档")):
                _text(draw, (margin + 14, 239 + n * 48), text, 22, theme["text"])
            draw.rectangle((236, 220, width - margin, height - 72), fill=theme["card"], outline=theme["line"])
            _text(draw, (263, 243), document, 27, theme["text"])
            for n in range(7):
                draw.rounded_rectangle((263, 310 + n * 27, width - margin - rng.randint(35, 137),
                                        318 + n * 27), radius=3, fill=theme["line"])
        elif layout == "table":
            yy = 223
            draw.rectangle((margin, yy, width - margin, yy + 45), fill=theme["panel"])
            columns = (margin + 18, margin + 240, margin + 460, width - margin - 110)
            for xx, text in zip(columns, ("名称", "负责人", "分类", "更新")):
                _text(draw, (xx, yy + 8), text, 22, theme["text"])
            for n in range(5):
                row_y = yy + 45 + n * 48
                draw.rectangle((margin, row_y, width - margin, row_y + 48), fill=theme["card"], outline=theme["line"])
                values = (f"资料 {scenario + 1:03d}-{n + 1}", rng.choice(("小陈", "小李", "小林")),
                          rng.choice(("文稿", "附件", "计划")), "今天")
                for xx, value in zip(columns, values):
                    _text(draw, (xx, row_y + 10), value, 21, theme["text"])
        else:
            draw.rectangle((margin, 220, 214, height - 72), fill=theme["panel"])
            for n, text in enumerate(("快速访问", "桌面", "文档", "下载", "图片")):
                _text(draw, (margin + 14, 239 + n * 48), text, 21, theme["text"])
            for n, text in enumerate((document + ".docx", "预算.xlsx", "会议记录.txt", "图片", "参考资料.pdf")):
                yy = 229 + n * 57
                draw.rectangle((238, yy, width - margin, yy + 46), fill=theme["card"], outline=theme["line"])
                _text(draw, (255, yy + 8), text, 21, theme["text"])
                _text(draw, (width - 209, yy + 9), "今天更新", 19, theme["muted"])
    return image, boxes, {"theme": theme, "style_family": style, "dark": dark,
                          "font_name": Path(_cjk_font_path()).name}


def _render_scenario(output: Path, group_id: str, layout: str, scenario: int,
                     seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(f"visual_state_v2:{seed}:{scenario}")
    base, boxes, style = _base_screen(rng, layout, scenario)
    width, height = base.size
    target_index = rng.randrange(len(boxes))
    target_box = boxes[target_index]
    others = [i for i in range(len(boxes)) if i != target_index]
    rng.shuffle(others)
    texts = ["另存为", "取消", "打开", "导出", "预览"]
    rng.shuffle(texts)
    draw = ImageDraw.Draw(base)
    other_controls = []
    for offset, slot in enumerate(others):
        # Each screenshot has both an enabled gray secondary control and a
        # disabled distractor. Whole-image grayness cannot identify the label.
        enabled = offset == 0 or (offset != 1 and rng.random() > 0.30)
        role = "neutral" if offset == 0 else rng.choice(("primary", "secondary", "neutral"))
        _button(draw, boxes[slot], texts[offset], enabled, style["theme"], role)
        other_controls.append({"text": texts[offset], "bbox": _norm(boxes[slot], width, height),
                               "enabled": enabled, "role": role})
    target_role = rng.choice(("primary", "secondary", "neutral"))
    replacement_text = rng.choice([text for text in ("返回", "撤销", "重试", "关闭") if text not in texts])
    state_order = list(STATES)
    rng.shuffle(state_order)
    rows = []
    for variant, state in enumerate(state_order):
        image = base.copy()
        controls = [dict(control) for control in other_controls]
        replacement = None
        if state != "absent":
            _button(ImageDraw.Draw(image), target_box, TARGET_TEXT, state == "enabled", style["theme"], target_role)
            controls.append({"text": TARGET_TEXT, "bbox": _norm(target_box, width, height),
                             "enabled": state == "enabled", "role": target_role})
        else:
            # Match count, shape, enabled style and font size to the enabled
            # target. Presence recognition must depend on the literal label,
            # rather than a missing rectangle or one fewer control.
            _button(ImageDraw.Draw(image), target_box, replacement_text, True, style["theme"], target_role, font_size=25)
            replacement = {"text": replacement_text, "bbox": _norm(target_box, width, height),
                           "enabled": True, "role": target_role}
            controls.append(dict(replacement))
        relative = f"images/{group_id}_v{variant}.png"
        image.save(output / relative, format="PNG", optimize=False)
        rows.append({"state": state, "image": relative,
                     "target_bbox": None if state == "absent" else _norm(target_box, width, height),
                     "controls": controls, "replacement": replacement})
    metadata = {"width": width, "height": height, "counterfactual_bbox": _norm(target_box, width, height),
                "style_family": style["style_family"], "dark": style["dark"], "font_name": style["font_name"]}
    return rows, metadata


def build_visual_state_dataset(output_dir: Path | str, num_scenarios: int = 300, seed: int = 43) -> Path:
    """Write three pixel-paired states per scenario and return manifest.jsonl.

    At N=300: train=180, val=60, test-iid=30, test-ood=30 scenarios.
    The table/filemanager layouts occur exclusively in test-ood. For smaller N
    integer rounding keeps all splits, both test domains and both OOD layouts.
    The exact instruction is identical for every row: ``点击保存``. Absent
    replaces the target with a differently named enabled button of the same
    style, so every image contains the same total number of controls.
    """
    plan = _scenario_plan(num_scenarios, seed)
    _cjk_font_path()  # Fail before writing any files if Chinese cannot be rendered.
    output = Path(output_dir).expanduser().resolve()
    (output / "images").mkdir(parents=True, exist_ok=True)
    records = []
    for scenario, specification in enumerate(plan):
        group_id = f"visual_state_{seed}_{scenario:05d}"
        variants, shared = _render_scenario(output, group_id, specification["layout"], scenario, seed)
        for variant_index, variant in enumerate(variants):
            state = variant["state"]
            records.append({"sample_id": f"{group_id}_v{variant_index}", "group_id": group_id,
                            "task": "visual_state", "image": variant["image"], "instruction": INSTRUCTION,
                            "label": STATE_LABELS[state], "split": specification["split"],
                            "target_text": TARGET_TEXT, "target_bbox": variant["target_bbox"],
                            "template_id": "visual_state.fixed_click_save", "source": "synthetic_visual_state_v2_matched",
                            "metadata": {"task": "visual_state", **specification, **shared,
                                         "visual_state": state, "controls": variant["controls"],
                                         "replacement": variant["replacement"], "seed": seed}})
    # Break label order and group-order correlations. Group/split metadata remain
    # intact and all three variants are still available for paired analyses.
    random.Random(f"visual_state_manifest:{seed}").shuffle(records)
    manifest = output / "manifest.jsonl"
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in records)
    manifest.write_text(payload, encoding="utf-8", newline="\n")
    summary = validate_manifest(records, manifest_path=manifest)
    group_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for item in plan:
        group_counts[item["split"]][item["domain"]] += 1
    summary.update({"schema_version": 2, "task": "visual_state", "seed": seed, "instruction": INSTRUCTION,
                    "counterfactual_protocol": "matched_control_count_and_enabled_style",
                    "num_scenarios": num_scenarios, "scenario_counts_by_split_domain": dict(group_counts),
                    "scenario_counts_by_layout": dict(Counter(item["layout"] for item in plan)),
                    "sample_counts_by_split_domain_label": {
                        split: {domain: {label: count for label in STATE_LABELS.values()} for domain, count in domains.items()}
                        for split, domains in group_counts.items()},
                    "seen_layouts": list(SEEN_LAYOUTS), "ood_layouts": list(OOD_LAYOUTS),
                    "style_families": dict(Counter(row["metadata"]["style_family"] for row in records)),
                    "manifest_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
                    "limitation": "Procedural button-state recognition, not real-application task success. All annotation metadata are excluded from features."})
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return manifest


def write_contact_sheet(manifest_path: Path | str, output_path: Path | str | None = None) -> Path:
    """Render one enabled/disabled/absent triplet for each layout, for visual QA."""
    manifest = Path(manifest_path).resolve()
    records = load_manifest(manifest, validate_images=False)
    if any(row.get("task") != "visual_state" for row in records):
        raise ValueError("Contact sheets require a visual_state manifest")
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_group[row["group_id"]].append(row)
    selected = {}
    for group in by_group.values():
        selected.setdefault(group[0]["metadata"]["layout"], group)
    cell_width, cell_height = 512, 382
    sheet = Image.new("RGB", (cell_width * 3, cell_height * len(selected)), "#ffffff")
    draw = ImageDraw.Draw(sheet)
    for row_index, layout in enumerate(layout for layout in LAYOUTS if layout in selected):
        group = {item["metadata"]["visual_state"]: item for item in selected[layout]}
        for column, state in enumerate(STATES):
            record = group[state]
            with Image.open(manifest.parent / record["image"]) as source:
                thumbnail = source.convert("RGB")
                thumbnail.thumbnail((cell_width - 8, cell_height - 36))
            x, y = column * cell_width, row_index * cell_height
            _text(draw, (x + 8, y + 3), f"{layout} / {state}", 19)
            sheet.paste(thumbnail, (x + 4, y + 31))
    result = Path(output_path).resolve() if output_path is not None else manifest.parent / "contact_sheet.png"
    result.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(result)
    return result
