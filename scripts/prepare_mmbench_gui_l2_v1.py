"""Build a label-free, history-audited MMBench-GUI L2 cohort.

The source JSON contains bboxes. This preparer projects only public fields and
never examines bbox values. Cohort selection uses image identity, application
name and fixed SHA256 order; no target answer or model prediction is used.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/mmbench_gui_l2_v1/source"
OUT = ROOT / "runs/mmbench_gui_new_source_v1"
ANNOTATIONS = SOURCE / "L2_annotations.json"
ARCHIVE = SOURCE / "MMBench-GUI-OfflineImages.zip"
ANNOTATION_SHA = "49fe9873be87c6eba8a087707a496aa1ddce35885818874cef4b3bdfb91f5999"
ARCHIVE_SHA = "56e087c3b5c2941e2bc51c2f347a3d46f41ca50f1c0370369c284aa02c6458a3"
HISTORY = ROOT / "runs/independent_diagnostic_v32/historical_exclusion.json"
ADDITIONAL = ROOT / "runs/independent_diagnostic_v32/additional_history_hashes.json"
INDEX = ROOT / "data/conditioned_v3/public_image_index.json"
EXPANSION = ROOT / "runs/dataset_expansion_v15/image_index.json"
EXPLICIT = (
    "runs/mvp_confirmation_v18/data/heldout.jsonl",
    "runs/independent_diagnostic_v32/data/public.json",
    "runs/independent_diagnostic_v32/data/reserve_public.json",
    "runs/observation_regions_v33/data/public.json",
    "runs/paired_scale_v35/data/public.json",
    "runs/paper_scale_ablation_v2/pointerbench/public.jsonl",
    "data/conditioned_v3/training_pool_b8ed555f5e93.jsonl",
    "runs/quality_readout_v15_3k/data/new_train.jsonl",
    "runs/full_scaling_v18/data/new_train.jsonl",
)
PUBLIC_FIELDS = (
    "index", "image_path", "instruction", "image_size", "data_type",
    "platform", "app_name", "grounding_type",
)
N_IMAGES = 500


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_once(path: Path, obj: object) -> None:
    data = (json.dumps(obj, sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    if path.exists():
        assert path.read_bytes() == data, f"Frozen file changed: {path}"
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def read(path: Path):
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    return json.loads(path.read_text(encoding="utf-8-sig"))


def image_info(raw: bytes) -> dict:
    with Image.open(io.BytesIO(raw)) as opened:
        image = opened.convert("RGB")
        w, h = image.size
        pixel = hashlib.sha256(f"{w}x{h}:RGB:".encode() + image.tobytes()).hexdigest()
        gray = np.asarray(image.convert("L").resize((17, 16), Image.Resampling.LANCZOS))
        dhash = np.packbits((gray[:, 1:] > gray[:, :-1]).ravel()).tobytes().hex()
    return dict(image_sha256=hashlib.sha256(raw).hexdigest(), pixel_sha256=pixel,
                dhash256=dhash, image_size=[w, h])


def public_projection(source_rows: list[dict]) -> list[dict]:
    rows = []
    for item in source_rows:
        assert set(item) == set(PUBLIC_FIELDS) | {"bbox"}, "Source schema changed"
        row = {name: item[name] for name in PUBLIC_FIELDS}
        assert row["grounding_type"] in {"basic", "advanced"}
        assert row["platform"] in {"os_windows", "os_mac", "os_linux", "os_ios", "os_android", "os_web"}
        assert row["data_type"] in {"icon", "text"}
        assert isinstance(row["instruction"], str) and row["instruction"].strip()
        assert Path(row["image_path"]).name == row["image_path"]
        rows.append(row)
    assert len(rows) == 3594 and len({r["index"] for r in rows}) == len(rows)
    return rows


def add_hashes(rows: list[dict], rgb: set[str], dhash: set[int], file_sha: set[str]) -> None:
    for row in rows:
        if row.get("pixel_sha256"):
            rgb.add(row["pixel_sha256"])
        if row.get("dhash256"):
            dhash.add(int(row["dhash256"], 16))
        for field in ("image_sha256", "file_sha256"):
            if row.get(field):
                file_sha.add(row[field])


def history_universe() -> tuple[set[str], set[int], set[str], dict]:
    source_paths = (HISTORY, ADDITIONAL, INDEX, EXPANSION, *(ROOT / p for p in EXPLICIT))
    assert all(p.exists() for p in source_paths), "Historical image source missing"
    history = read(HISTORY)
    assert history["complete"] and not history["unresolved"]
    rgb, dhash, file_sha = set(history["historical_RGB"]), set(), set()
    dhash.update(int(value, 16) for value in history["historical_dhash256"])
    add_hashes(read(ADDITIONAL), rgb, dhash, file_sha)
    add_hashes(read(INDEX)["images"], rgb, dhash, file_sha)
    add_hashes(read(EXPANSION)["images"], rgb, dhash, file_sha)
    for name in EXPLICIT:
        value = read(ROOT / name)
        assert isinstance(value, list), f"Historical manifest schema changed: {name}"
        add_hashes(value, rgb, dhash, file_sha)
    receipt = dict(source_sha256={str(p.relative_to(ROOT)): sha(p) for p in source_paths},
                   history_manifest_count=len(history["history_manifest_paths"]),
                   historical_rgb_count=len(rgb), historical_dhash_count=len(dhash),
                   historical_file_sha_count=len(file_sha))
    return rgb, dhash, file_sha, receipt


def nearest_distance(value: int, others: set[int]) -> int:
    return min(((value ^ other).bit_count() for other in others), default=256)


def select_key(platform: str, image_path: str) -> str:
    return hashlib.sha256(f"MMBench-GUI-L2-v1|{platform}/{image_path}".encode()).hexdigest()


def main() -> None:
    assert sha(ANNOTATIONS) == ANNOTATION_SHA
    assert ARCHIVE.stat().st_size == 1082948866 and sha(ARCHIVE) == ARCHIVE_SHA
    source_rows = read(ANNOTATIONS)
    public_rows = public_projection(source_rows)
    # No bbox reference is allowed below this line.
    del source_rows
    groups = defaultdict(list)
    for row in public_rows:
        groups[(row["platform"], row["image_path"])].append(row)
    rgb_hist, dhash_hist, file_hist, history_receipt = history_universe()
    zf = zipfile.ZipFile(ARCHIVE)
    names = set(zf.namelist())
    images = []
    for i, ((platform, image_path), cases) in enumerate(sorted(groups.items()), 1):
        archive_key = f"offline_images/{platform}/{image_path}"
        assert archive_key in names, f"Missing image: {archive_key}"
        raw = zf.read(archive_key)
        info = image_info(raw)
        assert all(row["image_size"] == info["image_size"] for row in cases), f"Size mismatch: {archive_key}"
        value = int(info["dhash256"], 16)
        distance = nearest_distance(value, dhash_hist)
        images.append(dict(platform=platform, image_path=image_path, archive_key=archive_key,
                           app_name=cases[0]["app_name"], case_indices=sorted(row["index"] for row in cases),
                           grounding_counts=dict(Counter(row["grounding_type"] for row in cases)),
                           historical_exact=info["pixel_sha256"] in rgb_hist or info["image_sha256"] in file_hist,
                           historical_near_le4=distance <= 4, historical_min_dhash_distance=distance,
                           **info))
        if i % 100 == 0 or i == len(groups):
            print(f"audited {i}/{len(groups)} images", flush=True)
    candidates = [r for r in images if r["grounding_counts"].get("basic", 0) and
                  r["grounding_counts"].get("advanced", 0) and
                  not r["app_name"].lower().startswith("screenspot") and
                  not r["historical_exact"] and not r["historical_near_le4"]]
    candidates.sort(key=lambda r: (select_key(r["platform"], r["image_path"]), r["platform"], r["image_path"]))
    selected = []
    selected_hashes = set()
    selected_rgb = set()
    for row in candidates:
        value = int(row["dhash256"], 16)
        if row["pixel_sha256"] in selected_rgb or nearest_distance(value, selected_hashes) <= 4:
            continue
        selected.append(row)
        selected_hashes.add(value)
        selected_rgb.add(row["pixel_sha256"])
        if len(selected) == N_IMAGES:
            break
    assert len(selected) == N_IMAGES, f"Only {len(selected)} history-clean, internally unique paired images"
    by_image = {(r["platform"], r["image_path"]): r for r in selected}
    cohort = []
    for image in selected:
        options = groups[(image["platform"], image["image_path"])]
        for kind in ("basic", "advanced"):
            # First source index per type; this pairs images, not necessarily target elements.
            query = min((r for r in options if r["grounding_type"] == kind), key=lambda r: r["index"])
            extracted = ROOT / "data/mmbench_gui_l2_v1/images" / image["platform"] / image["image_path"]
            row = dict(sample_id=f"mmbench_gui_l2:{query['index']}", dataset="MMBench-GUI-L2",
                       source_index=query["index"], grounding_type=kind,
                       data_type=query["data_type"], platform=query["platform"],
                       application=query["app_name"], application_group=query["platform"],
                       ui_type=query["data_type"], instruction=query["instruction"],
                       image=str(extracted.resolve()), image_size=image["image_size"],
                       image_sha256=image["image_sha256"], pixel_sha256=image["pixel_sha256"],
                       dhash256=image["dhash256"])
            cohort.append(row)
    assert len(cohort) == 1000 and len({r["sample_id"] for r in cohort}) == 1000
    for image in selected:
        target = ROOT / "data/mmbench_gui_l2_v1/images" / image["platform"] / image["image_path"]
        raw = zf.read(image["archive_key"])
        assert hashlib.sha256(raw).hexdigest() == image["image_sha256"]
        if target.exists():
            assert hashlib.sha256(target.read_bytes()).hexdigest() == image["image_sha256"]
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
    zf.close()
    write_once(OUT / "public.json", cohort)
    audit = dict(complete=True, source="OpenGVLab/MMBench-GUI L2",
                 new_source_scope="Project-history-unseen RGB images in MMBench-GUI L2 after excluding explicitly ScreenSpot-named app sources; not a claim that every remaining image was independently collected by the benchmark authors",
                 history_lineage_note="V36/V37 training images inherit the 112-image V35 TRAIN cohort; the 112-image V35 public manifest is explicitly included in the history hash universe alongside the earlier V32 inventory",
                 source_hf_revision="e27757e3910e0d5995b811f916b509b4e34a4690",
                 image_download_mirror="ModelScope OpenGVLab/MMBench-GUI",
                 source_annotation_sha256=ANNOTATION_SHA, source_archive_sha256=ARCHIVE_SHA,
                 annotation_rows=len(public_rows), unique_source_images=len(images),
                 image_paired_no_bbox_matching=True,
                 sample_rule="SHA256 order of platform/image_path; first source index for each instruction type; exclude screenspot-named source, historical exact or dHash<=4, and internally near-duplicate selected images",
                 selected_images=len(selected), selected_queries=len(cohort),
                 selected_platforms=dict(Counter(r["platform"] for r in selected)),
                 all_image_overlap=dict(exact=sum(r["historical_exact"] for r in images),
                                        near_le4=sum(r["historical_near_le4"] for r in images),
                                        screenspot_named=sum(r["app_name"].lower().startswith("screenspot") for r in images)),
                 one_type_only_images=sum(not (r["grounding_counts"].get("basic", 0) and r["grounding_counts"].get("advanced", 0)) for r in images if not r["historical_exact"] and not r["historical_near_le4"] and not r["app_name"].lower().startswith("screenspot")),
                 candidate_images_before_internal_dedup=len(candidates),
                 selected_history_exact=0, selected_history_near_le4=0,
                 historical_reference=history_receipt,
                 all_images_sha256=hashlib.sha256(json.dumps(images, sort_keys=True).encode()).hexdigest(),
                 selected_image_keys=[r["archive_key"] for r in selected],
                 public_sha256=sha(OUT / "public.json"),
                 image_file_sha256={r["archive_key"]: r["image_sha256"] for r in selected},
                 labels_examined_by_selection=False, model_forward_calls=0)
    write_once(OUT / "audit.json", audit)
    print(json.dumps({k: audit[k] for k in ("annotation_rows", "unique_source_images", "all_image_overlap", "candidate_images_before_internal_dedup", "selected_images", "selected_queries", "selected_platforms")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
