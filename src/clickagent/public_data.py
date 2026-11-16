"""Pinned public GUI benchmark acquisition and lossless annotation adapters.

Native curl is intentional: Windows curl reaches these public endpoints while
Python's TLS stack is reset by this network. Raw files are never executed.
"""
from __future__ import annotations

import concurrent.futures
import copy
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path, PureWindowsPath
import shutil
import stat
import subprocess
import tarfile
import threading
import time
from typing import Any
from urllib.parse import quote, urlsplit
import zipfile

from PIL import Image

DATASET_REPOS = {
    "screenspot_v2": "OS-Copilot/ScreenSpot-v2",
    "screenspot_pro": "likaixin/ScreenSpot-Pro",
    "venusbench_gd": "inclusionAI/VenusBench-GD",
    "conflictgui": "serein356/ConflictGUI",
}
DATASET_NAMES = {"screenspot_v2": "ScreenSpot-v2", "screenspot_pro": "ScreenSpot-Pro",
                 "venusbench_gd": "VenusBench-GD", "osworld_g": "OSWorld-G", "conflictgui": "ConflictGUI"}
EXPECTED_SAMPLE_COUNTS = {"screenspot_v2": 1272, "screenspot_pro": 1581, "osworld_g": 564,
                          "venusbench_gd": 6165, "conflictgui": 4660}
_LOCK = threading.Lock()


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def safe_path(root: Path | str, name: str) -> Path:
    root = Path(root).resolve()
    windows = PureWindowsPath(name)
    relative = Path(name.replace("\\", "/"))
    if relative.is_absolute() or windows.drive or windows.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe relative path: {name}")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"Path escapes destination: {name}")
    return resolved


def file_hash(path: Path, algorithm: str = "sha256", git_blob: bool = False) -> str:
    digest = hashlib.new(algorithm)
    if git_blob:
        digest.update(f"blob {path.stat().st_size}\0".encode("ascii"))
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_extract(archive: Path | str, destination: Path | str) -> list[str]:
    """Extract zip/tar files after rejecting traversal, links and special files."""
    archive, destination = Path(archive), Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    written = []
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as bundle:
            members = bundle.infolist()
            for member in members:
                safe_path(destination, member.filename)
                if stat.S_ISLNK(member.external_attr >> 16):
                    raise ValueError(f"Archive symlink rejected: {member.filename}")
            for member in members:
                target = safe_path(destination, member.filename)
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(member) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output, 1024 * 1024)
                written.append(member.filename)
    else:
        with tarfile.open(archive) as bundle:
            members = bundle.getmembers()
            for member in members:
                safe_path(destination, member.name)
                if not (member.isfile() or member.isdir()):
                    raise ValueError(f"Archive link/special file rejected: {member.name}")
            for member in members:
                target = safe_path(destination, member.name)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.extractfile(member) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output, 1024 * 1024)
                written.append(member.name)
    return written


def download_file(urls: list[str], destination: Path | str, *, size: int | None = None,
                  sha256: str | None = None, git_sha1: str | None = None, retries: int = 3) -> dict[str, Any]:
    """Resume a .part with curl, verify published hashes, and atomically promote."""
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    receipt = destination.with_name(destination.name + ".download.json")
    if destination.is_file() and receipt.is_file():
        prior = _json(receipt)
        info = destination.stat()
        if (prior.get("size") == info.st_size and prior.get("mtime_ns") == info.st_mtime_ns and
            (size is None or size == info.st_size) and (sha256 is None or prior.get("sha256") == sha256) and
            (git_sha1 is None or prior.get("git_sha1") == git_sha1)):
            return {**prior, "cached": True}
    part = destination.with_name(destination.name + ".part")
    errors = []
    # A stop after the final byte (or between rename and receipt creation) must
    # not trigger a range request starting at EOF. Only published hashes can
    # establish that these interrupted local files are complete.
    if sha256 or git_sha1:
        for candidate in (destination, part):
            if not candidate.is_file() or (size is not None and candidate.stat().st_size != size):
                continue
            candidate_sha = file_hash(candidate)
            candidate_git = file_hash(candidate, "sha1", git_blob=True) if git_sha1 else None
            if (sha256 and candidate_sha != sha256) or (git_sha1 and candidate_git != git_sha1):
                if candidate == part:
                    os.replace(part, part.with_name(part.name + f".bad-{time.time_ns()}"))
                continue
            if candidate == part:
                os.replace(part, destination)
            record = {"url": urls[0], "effective_origin": None, "path": str(destination),
                      "size": destination.stat().st_size, "sha256": candidate_sha,
                      "git_sha1": candidate_git, "published_sha256": sha256, "published_git_sha1": git_sha1,
                      "seconds": 0.0, "mtime_ns": destination.stat().st_mtime_ns,
                      "errors_before_success": [], "cached": True,
                      "recovery": "verified_local_file_against_published_hash"}
            _write_json(receipt, record)
            return record
    for url in urls:
        for attempt in range(retries):
            arguments = ["curl.exe" if os.name == "nt" else "curl", "--fail", "--location", "--silent", "--show-error",
                         "--connect-timeout", "15", "--max-time", "1800", "--speed-time", "60", "--speed-limit", "1024",
                         "--output", str(part), "--write-out", "%{url_effective}"]
            if part.is_file() and part.stat().st_size:
                arguments += ["--continue-at", "-"]
            started = time.monotonic()
            try:
                result = subprocess.run(arguments + [url], capture_output=True, text=True,
                                        encoding="utf-8", errors="replace", timeout=1810)
            except (OSError, subprocess.TimeoutExpired) as error:
                errors.append({"url": url, "attempt": attempt + 1, "error": f"{type(error).__name__}: {error}"})
                time.sleep(min(2 ** attempt, 4))
                continue
            if result.returncode:
                errors.append({"url": url, "attempt": attempt + 1, "error": result.stderr[-1200:]})
                if result.returncode == 33 and part.is_file():
                    part.unlink()  # Explicit single .part file under this destination.
                time.sleep(min(2 ** attempt, 4))
                continue
            actual_size = part.stat().st_size
            actual_sha = file_hash(part)
            actual_git = file_hash(part, "sha1", git_blob=True) if git_sha1 else None
            if ((size is not None and actual_size != size) or (sha256 and actual_sha != sha256) or
                (git_sha1 and actual_git != git_sha1)):
                errors.append({"url": url, "attempt": attempt + 1, "error": "Published size/hash mismatch",
                               "actual_size": actual_size, "actual_sha256": actual_sha})
                quarantine = part.with_name(part.name + f".bad-{time.time_ns()}")
                os.replace(part, quarantine)
                continue
            os.replace(part, destination)
            record = {"url": url, "effective_origin": urlsplit(result.stdout.strip()).netloc,
                      "path": str(destination), "size": actual_size, "sha256": actual_sha,
                      "git_sha1": actual_git, "published_sha256": sha256, "published_git_sha1": git_sha1,
                      "seconds": time.monotonic() - started, "mtime_ns": destination.stat().st_mtime_ns,
                      "errors_before_success": errors, "cached": False}
            _write_json(receipt, record)
            return record
    raise RuntimeError(json.dumps({"destination": str(destination), "errors": errors}, ensure_ascii=True))


def _inventory(dataset: str, root: Path) -> dict[str, Any]:
    discovery = root / "discovery"
    if dataset == "osworld_g":
        path = discovery / "osworld_tree.json"
        if not path.is_file():
            download_file(["https://api.github.com/repos/xlang-ai/OSWorld-G/git/trees/main?recursive=1"], path)
    else:
        repo = DATASET_REPOS[dataset]
        path = discovery / (repo.replace("/", "__") + ".json")
        if not path.is_file():
            download_file([f"https://huggingface.co/api/datasets/{repo}?blobs=true",
                           f"https://hf-mirror.com/api/datasets/{repo}?blobs=true"], path)
    return _json(path)


def download_public_dataset(dataset: str, root: Path | str = "data/public_v1", workers: int = 4) -> dict[str, Any]:
    """Download every official repository file, including all raw annotations."""
    if dataset not in DATASET_NAMES:
        raise ValueError(f"Unknown dataset: {dataset}")
    root = Path(root).resolve()
    info = _inventory(dataset, root)
    revision = info["sha"]
    raw = root / "raw" / dataset
    raw.mkdir(parents=True, exist_ok=True)
    provenance = {"dataset": DATASET_NAMES[dataset], "revision": revision, "download_complete": False,
                  "official_repo": "https://github.com/xlang-ai/OSWorld-G" if dataset == "osworld_g"
                  else f"https://huggingface.co/datasets/{DATASET_REPOS[dataset]}"}
    _write_json(raw / "provenance.json", provenance)
    errors = []
    if dataset == "osworld_g":
        archive = raw / f"source-{revision}.zip"
        record = download_file([f"https://codeload.github.com/xlang-ai/OSWorld-G/zip/{revision}"], archive)
        extracted = raw / "repository"
        safe_extract(archive, extracted)
        provenance.update({"archive": record, "files_expected": len([f for f in info["tree"] if f["type"] == "blob"]),
                           "repository_directory": str(extracted / f"OSWorld-G-{revision}")})
    else:
        repo = DATASET_REPOS[dataset]
        files = info["siblings"]
        provenance.update({"files_expected": len(files), "bytes_expected": sum(f.get("size", 0) for f in files)})
        def transfer(item: dict[str, Any]) -> dict[str, Any]:
            name = item["rfilename"]
            target = safe_path(raw, name)
            lfs = item.get("lfs") or {}
            encoded = quote(name, safe="/")
            urls = [f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{encoded}?download=true",
                    f"https://hf-mirror.com/datasets/{repo}/resolve/{revision}/{encoded}?download=true"]
            # Official-org ModelScope mirrors were checked against the HF LFS
            # inventory; every byte downloaded still passes the pinned HF hash.
            if dataset in {"screenspot_v2", "venusbench_gd"}:
                ms = f"https://modelscope.cn/api/v1/datasets/{repo}/repo?Revision=master&FilePath={encoded}"
                urls.insert(0, ms)
            return download_file(urls, target, size=item.get("size"), sha256=lfs.get("sha256"),
                                 git_sha1=None if lfs else item.get("blobId"))
        complete = 0
        started = time.monotonic()
        # Annotations first; the full inventory is still always downloaded.
        files = sorted(files, key=lambda f: (not f["rfilename"].endswith((".json", ".md", ".yaml")), f.get("size", 0)))
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(transfer, item): item for item in files}
            for future in concurrent.futures.as_completed(futures):
                item = futures[future]
                try:
                    future.result()
                    complete += 1
                except Exception as error:
                    errors.append({"path": item["rfilename"], "error": str(error)})
                    print(json.dumps({"dataset": dataset, "failed_file": item["rfilename"],
                                      "error": str(error)}, ensure_ascii=True), flush=True)
                if complete % 100 == 0 or complete + len(errors) == len(files):
                    print(json.dumps({"dataset": dataset, "downloaded": complete, "expected": len(files),
                                      "errors": len(errors), "elapsed": round(time.monotonic() - started, 1)}), flush=True)
        provenance["files_downloaded"] = complete
        if dataset == "screenspot_v2" and not errors:
            safe_extract(raw / "screenspotv2_image.zip", raw / "extracted")
    mirror_audit = root / "audits" / f"{dataset}_mirror_recovery.json"
    if mirror_audit.is_file():
        provenance["verified_mirror_recovery"] = {"audit": str(mirror_audit), "sha256": file_hash(mirror_audit)}
    provenance.update({"download_complete": not errors, "download_errors": errors})
    _write_json(raw / "provenance.json", provenance)
    if errors:
        raise RuntimeError(f"{dataset}: {len(errors)} files failed; see {raw / 'provenance.json'}")
    return provenance


def recover_screenspot_pro_from_modelscope(root: Path | str = "data/public_v1", workers: int = 4,
                                          arrow_path: Path | str | None = None) -> dict[str, Any]:
    """Recover exact original PNG bytes from a differently packaged mirror.

    Call before starting a concurrent raw Pro downloader. The mirror's labels
    are never used: only an image matching the pinned official HF path and LFS
    SHA-256 can be promoted into the raw release. PyArrow is optional and may be
    installed in a separate --target directory rather than the model venv.
    """
    if arrow_path:
        import sys
        sys.path.insert(0, str(Path(arrow_path).resolve()))
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("Optional mirror recovery requires pyarrow (or --arrow-path).") from error
    root = Path(root).resolve()
    official = _inventory("screenspot_pro", root)
    known = {item["rfilename"]: item for item in official["siblings"]}
    discovery = root / "discovery" / "pro_modelscope.response"
    if not discovery.is_file():
        download_file(["https://modelscope.cn/api/v1/datasets/lmms-lab/ScreenSpot-Pro/repo/tree?Revision=master&Recursive=true"], discovery)
    mirror = _json(discovery)
    shards = [item for item in mirror["Data"]["Files"] if item["Type"] == "blob" and item["Path"].endswith(".parquet")]
    if len(shards) != 8:
        raise ValueError(f"Expected 8 explicitly inventoried mirror parquet shards, got {len(shards)}")
    cache = root / "mirror_cache" / "screenspot_pro"
    def transfer(item: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        url = ("https://modelscope.cn/api/v1/datasets/lmms-lab/ScreenSpot-Pro/repo?Revision=" +
               item["Revision"] + "&FilePath=" + quote(item["Path"], safe="/"))
        return item, download_file([url], safe_path(cache, item["Path"]), size=item["Size"], sha256=item["Sha256"])
    containers = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for future in concurrent.futures.as_completed([pool.submit(transfer, item) for item in shards]):
            item, receipt = future.result()
            containers.append((item, receipt))
            print(json.dumps({"dataset": "screenspot_pro", "mirror_shard_ready": item["Path"],
                              "sha256": receipt["sha256"]}), flush=True)
    matched, recovered, rejected = [], [], []
    for container, receipt in sorted(containers, key=lambda pair: pair[0]["Path"]):
        for batch in pq.ParquetFile(receipt["path"]).iter_batches(batch_size=1, columns=["image", "img_filename"]):
            for item in batch.to_pylist():
                key = "images/" + item["img_filename"]
                original = known.get(key)
                payload = item["image"]["bytes"]
                actual_sha = hashlib.sha256(payload).hexdigest()
                if original is None or actual_sha != (original.get("lfs") or {}).get("sha256"):
                    rejected.append({"path": key, "reason": "not_in_pinned_inventory_or_sha256_mismatch",
                                     "actual_sha256": actual_sha})
                    continue
                matched.append(key)
                destination = safe_path(root / "raw" / "screenspot_pro", key)
                if destination.is_file() and file_hash(destination) == actual_sha:
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_name(destination.name + ".mirror-part")
                temporary.write_bytes(payload)
                os.replace(temporary, destination)
                image_receipt = {"url": receipt["url"], "effective_origin": receipt["effective_origin"],
                                 "path": str(destination), "size": len(payload), "sha256": actual_sha,
                                 "git_sha1": None, "published_sha256": original["lfs"]["sha256"],
                                 "published_git_sha1": None, "seconds": 0.0,
                                 "mtime_ns": destination.stat().st_mtime_ns, "errors_before_success": [], "cached": False,
                                 "recovery": "image_bytes_in_hash_verified_mirror_parquet",
                                 "official_hf_revision": official["sha"], "official_hf_path": key,
                                 "mirror_container_sha256": receipt["sha256"], "mirror_container_path": receipt["path"]}
                _write_json(destination.with_name(destination.name + ".download.json"), image_receipt)
                recovered.append(key)
    audit = {"official_repo": "https://huggingface.co/datasets/likaixin/ScreenSpot-Pro",
             "official_hf_revision": official["sha"], "mirror_repo": "https://modelscope.cn/datasets/lmms-lab/ScreenSpot-Pro",
             "policy": "only path-and-published-SHA256-identical image bytes; mirror labels never used",
             "containers": [receipt for _, receipt in containers], "matched_rows": len(matched),
             "unique_matched_images": len(set(matched)), "recovered_images": recovered, "rejected_images": rejected}
    _write_json(root / "audits" / "screenspot_pro_mirror_recovery.json", audit)
    return audit


@lru_cache(maxsize=20000)
def _checked_image_info(path: str, size: int, mtime_ns: int) -> tuple[str, list[int] | None, str | None]:
    try:
        with Image.open(path) as image:
            dimensions = list(image.size)
            image.verify()
        return "ok", dimensions, None
    except Exception as error:
        return "corrupt", None, f"{type(error).__name__}: {error}"


def _image_info(path: Path) -> tuple[str, list[int] | None, str | None]:
    if not path.is_file():
        return "missing", None, "File does not exist"
    info = path.stat()
    return _checked_image_info(str(path), info.st_size, info.st_mtime_ns)


def _bbox(values: list[float], width: int, height: int, fmt: str = "xyxy") -> list[float]:
    x0, y0, x1, y1 = map(float, values)
    if fmt == "xywh":
        x1, y1 = x0 + x1, y0 + y1
    return [x0 / width, y0 / height, x1 / width, y1 / height]


def _record(dataset: str, original_id: Any, index: int, image: Path, instruction: str,
            geometry: dict[str, Any] | None, actionable: bool, original: dict[str, Any],
            *, split: str = "test", domain: str = "default", task_kind: str | None = None,
            image_key: str | None = None) -> dict[str, Any]:
    status, dimensions, error = _image_info(image)
    image_id = f"{DATASET_NAMES[dataset]}:{image_key or image.name}"
    return {"sample_id": f"{dataset}:{index:06d}:{original_id}", "dataset": DATASET_NAMES[dataset],
            "source": dataset, "image": str(image.resolve()), "instruction": instruction, "split": split,
            "domain": domain, "task_kind": task_kind or ("grounding" if actionable else "refusal"),
            "target_geometry": geometry, "expected_actionable": bool(actionable), "group_id": image_id,
            "image_id": image_id, "image_size": dimensions,
            "metadata": {"original_id": original_id, "original": original, "domain": domain,
                         "image_status": status, "image_error": error, "point_supported": geometry is not None}}


def _normalize_osworld(raw: Path) -> list[dict[str, Any]]:
    candidates = list(raw.rglob("OSWorld-G.json"))
    if not candidates:
        raise FileNotFoundError("OSWorld-G.json annotations are not downloaded")
    # Prefer the complete extracted repository when available.
    annotation = sorted(candidates, key=lambda p: "repository" not in p.parts)[0]
    refined_path = annotation.with_name("OSWorld-G_refined.json")
    refined = {r["id"]: r for r in _json(refined_path)} if refined_path.is_file() else {}
    rows = []
    for index, item in enumerate(_json(annotation)):
        width, height = item["image_size"]
        kind = item["box_type"]
        coordinates = item["box_coordinates"]
        geometry = None
        if kind == "bbox":
            geometry = {"type": "bbox", "boxes": [_bbox(coordinates, width, height, "xywh")]}
        elif kind == "polygon":
            points = [[coordinates[i] / width, coordinates[i + 1] / height] for i in range(0, len(coordinates), 2)]
            geometry = {"type": "polygon", "polygons": [points]}
        elif kind != "refusal":
            raise ValueError(f"Unknown OSWorld-G box_type: {kind}")
        image = annotation.parent / "images" / item["image_path"]
        row = _record("osworld_g", item["id"], index, image, item["instruction"], geometry,
                      kind != "refusal", item, domain="osworld_g")
        row["metadata"].update({"official_box_format": "xywh" if kind == "bbox" else kind,
                                 "refined_annotation": refined.get(item["id"])})
        rows.append(row)
    if len(rows) != EXPECTED_SAMPLE_COUNTS["osworld_g"]:
        raise ValueError(f"OSWorld-G annotation count mismatch: {len(rows)}")
    return rows


def _normalize_screenspot_v2(raw: Path) -> list[dict[str, Any]]:
    image_paths = {path.name: path for path in (raw / "extracted").rglob("*")
                   if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}}
    rows = []
    for annotation in sorted(raw.glob("screenspot_*_v2.json")):
        domain = annotation.stem.removeprefix("screenspot_").removesuffix("_v2")
        for original_index, item in enumerate(_json(annotation)):
            image = image_paths.get(item["img_filename"], raw / "extracted" / item["img_filename"])
            status, dimensions, _ = _image_info(image)
            geometry = {"type": "bbox", "boxes": [_bbox(item["bbox"], *dimensions, "xywh")]} if dimensions else None
            original_id = f"{domain}:{original_index}"
            row = _record("screenspot_v2", original_id, len(rows), image, item["instruction"], geometry, True,
                          item, domain=domain)
            row["metadata"].update({"annotation_file": annotation.name, "annotation_index": original_index,
                                     "official_box_format": "xywh", "data_type": item.get("data_type"),
                                     "data_source": item.get("data_source")})
            rows.append(row)
    if len(rows) != EXPECTED_SAMPLE_COUNTS["screenspot_v2"]:
        raise ValueError(f"Pinned ScreenSpot-v2 must have 1272 rows; annotations currently yield {len(rows)}")
    return rows


def _boxes(values: list[Any], width: int, height: int, fmt: str = "xyxy") -> dict[str, Any]:
    boxes = values if values and isinstance(values[0], (list, tuple)) else [values]
    return {"type": "bbox", "boxes": [_bbox(box, width, height, fmt) for box in boxes]}


def _visible_box_geometry(row: dict[str, Any]) -> None:
    """Intersect newly adapted boxes with the image, retaining the exact raw GT.

    This is equivalent for predictions constrained to the visible image. Frozen
    v2/OSWorld canonical manifests are deliberately not passed through this rule.
    """
    geometry = row["target_geometry"]
    if geometry is None or geometry["type"] != "bbox":
        return
    clipped = [[min(1.0, max(0.0, value)) for value in box] for box in geometry["boxes"]]
    if clipped == geometry["boxes"]:
        return
    row["metadata"].update({"raw_normalized_target_geometry": copy.deepcopy(geometry),
                             "geometry_adjustment": "intersection_with_visible_image"})
    clipped = [box for box in clipped if box[0] < box[2] and box[1] < box[3]]
    row["target_geometry"] = {"type": "bbox", "boxes": clipped} if clipped else None
    row["metadata"]["point_supported"] = bool(clipped)
    row["metadata"]["geometry_visible"] = bool(clipped)


def _normalize_screenspot_pro(raw: Path) -> list[dict[str, Any]]:
    rows = []
    annotations = sorted(path for path in (raw / "annotations").glob("*.json") if not path.name.endswith(".download.json"))
    for annotation in annotations:
        for original_index, item in enumerate(_json(annotation)):
            image = safe_path(raw / "images", item["img_filename"])
            _, actual, _ = _image_info(image)
            dimensions = actual or item["img_size"]
            geometry = _boxes(item["bbox"], *dimensions)
            row = _record("screenspot_pro", item.get("id", f"{annotation.stem}:{original_index}"), len(rows),
                          image, item["instruction"], geometry, True, item,
                          domain=item.get("platform", annotation.stem), image_key=item["img_filename"])
            row["metadata"].update({"annotation_file": annotation.name, "annotation_index": original_index,
                                     "official_box_format": "xyxy", "application": item.get("application"),
                                     "platform": item.get("platform"), "ui_type": item.get("ui_type"),
                                     "application_group": item.get("group"), "declared_image_size": item["img_size"],
                                     "declared_size_matches_actual": actual == item["img_size"] if actual else None})
            _visible_box_geometry(row)
            rows.append(row)
    if len(rows) != EXPECTED_SAMPLE_COUNTS["screenspot_pro"]:
        raise ValueError(f"Pinned ScreenSpot-Pro must have 1581 rows; annotations currently yield {len(rows)}")
    return rows


def _normalize_venus(raw: Path) -> list[dict[str, Any]]:
    meta = {item["img_filename"]: item for item in _json(raw / "meta.json")}
    rows = []
    for annotation in sorted((raw / "instruction").glob("*_grounding.json")):
        for original_index, item in enumerate(_json(annotation)):
            image = safe_path(raw / "images", item["img_filename"])
            _, dimensions, _ = _image_info(image)
            actionable = annotation.stem != "refusal_grounding"
            geometry = _boxes(item["bbox"], *dimensions) if actionable and dimensions else None
            row = _record("venusbench_gd", f"{annotation.stem}:{original_index}", len(rows), image,
                          item["instruction"], geometry, actionable, item, domain=annotation.stem,
                          image_key=item["img_filename"])
            image_meta = meta.get(item["img_filename"], {})
            row["metadata"].update({"annotation_file": annotation.name, "annotation_index": original_index,
                                     "official_box_format": "xyxy", "category": item.get("category"),
                                     "image_metadata": image_meta, "platform": image_meta.get("platform"),
                                     "application": image_meta.get("application"), "language": "en",
                                     "additional_instruction_zh": item.get("instruction_zh")})
            _visible_box_geometry(row)
            rows.append(row)
    if len(rows) != EXPECTED_SAMPLE_COUNTS["venusbench_gd"]:
        raise ValueError(f"Pinned VenusBench-GD must have 6165 rows; annotations currently yield {len(rows)}")
    return rows


def _normalize_conflict(raw: Path) -> list[dict[str, Any]]:
    all_rows = _json(raw / "data" / "conflictgui.json")
    split_rows = _json(raw / "data" / "calibration.json") + _json(raw / "data" / "test.json")
    master = {item["sample_id"]: item for item in all_rows}
    partitioned = {item["sample_id"]: item for item in split_rows}
    if len(master) != len(all_rows) or master != partitioned or len(split_rows) != len(all_rows):
        raise ValueError("ConflictGUI master and calibration/test files disagree or contain duplicate IDs")
    rows = []
    for index, item in enumerate(all_rows):
        action = item["action"]
        reference = item["clean_reference_action"]
        reference_type = reference["action_type"].lower()
        actionable = item["data_type"] == "clean"
        image_key = item.get("screenshot") or item["images"][0]
        image = safe_path(raw, image_key)
        point_supported = actionable and reference_type in {"click", "tap"}
        row = _record("conflictgui", item["sample_id"], index, image, item["instruction"], None,
                      actionable, item, split=item["split"], domain=f"{item['dataset']}:{item['data_type']}",
                      task_kind="action_feasibility", image_key=image_key)
        point = None
        if reference_type in {"click", "tap"} and "x" in reference and "y" in reference:
            point = [reference["x"] / item["imgwidth"], reference["y"] / item["imgheight"]]
        row["metadata"].update({"reference_action_type": reference_type,
                                 "current_action_type": action["action_type"],
                                 "click_compatible": reference_type in {"click", "tap"},
                                 "point_supported": point_supported, "unsupported_for_point": not point_supported,
                                 "reference_point_normalized": point,
                                 "reference_point_valid_for_instruction": point_supported,
                                 "reference_point_tolerance": {"metric": "normalized_euclidean", "max_distance": 0.14,
                                     "official_rule_source": "https://github.com/serein356/ConflictGuard#evaluation-rules"},
                                 "conflict_type": item["data_type"], "upstream_dataset": item["dataset"],
                                 "task_id": item["task_id"], "label_meaning": "original_task_feasibility",
                                 "evaluation_scope": "binary feasibility only; not full official action SR"})
        rows.append(row)
    if len(rows) != EXPECTED_SAMPLE_COUNTS["conflictgui"]:
        raise ValueError(f"Pinned ConflictGUI must have 4660 rows; got {len(rows)}")
    return rows


def validate_public_geometry(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Reject malformed canonical geometry before publishing a new manifest.

    Raw annotations remain inside metadata. Visible-domain adjustments must be
    performed explicitly by an adapter before invoking this check.
    """
    counts = {"bbox": 0, "polygon": 0, "none": 0, "boxes": 0, "polygons": 0}
    for row in rows:
        geometry = row["target_geometry"]
        if geometry is None:
            counts["none"] += 1
            continue
        def valid_point(point: list[float]) -> bool:
            return (len(point) == 2 and all(isinstance(v, (int, float)) and math.isfinite(v) and
                                           0.0 <= v <= 1.0 for v in point))
        kind = geometry.get("type")
        if kind == "bbox":
            boxes = geometry.get("boxes", [])
            valid = bool(boxes) and all(len(box) == 4 and valid_point(box[:2]) and valid_point(box[2:]) and
                                       box[0] < box[2] and box[1] < box[3] for box in boxes)
            counts["boxes"] += len(boxes)
        elif kind == "polygon":
            polygons = geometry.get("polygons", [])
            valid = bool(polygons) and all(len(polygon) >= 3 and all(valid_point(p) for p in polygon)
                                          for polygon in polygons)
            counts["polygons"] += len(polygons)
        else:
            valid = False
        if not valid:
            raise ValueError(f"Invalid canonical geometry for {row['sample_id']}: {geometry}")
        counts[kind] += 1
    return counts


def _publish(root: Path, dataset: str, rows: list[dict[str, Any]]) -> Path:
    try:
        geometry_validation = {"valid": True, **validate_public_geometry(rows)}
    except ValueError:
        # Preserve the explicitly documented frozen v1 release. A subsequent
        # version may clip it, but rerunning this downloader must not silently
        # rewrite a benchmark already used for a formal evaluation.
        if dataset != "screenspot_v2":
            raise
        exceptions = []
        for row in rows:
            try:
                validate_public_geometry([row])
            except ValueError as error:
                exceptions.append({"sample_id": row["sample_id"], "error": str(error)})
        if [item["sample_id"] for item in exceptions] != ["screenspot_v2:000783:mobile:449"]:
            raise ValueError(f"Unexpected ScreenSpot-v2 geometry exceptions: {exceptions}")
        geometry_validation = {"valid": False, "policy": "frozen_v1_preserves_official_geometry",
                               "known_official_annotation_exceptions": exceptions}
    directory = root / "manifests"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{dataset}.jsonl"
    temporary = target.with_suffix(".jsonl.tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    os.replace(temporary, target)
    from collections import Counter
    audit = {"dataset": DATASET_NAMES[dataset], "samples": len(rows), "manifest_sha256": file_hash(target),
             "images": dict(Counter(row["metadata"]["image_status"] for row in rows)),
             "expected_actionable": dict(Counter(str(row["expected_actionable"]) for row in rows)),
             "splits": dict(Counter(row["split"] for row in rows)),
             "image_issues": [{"sample_id": row["sample_id"], "image": row["image"],
                               "status": row["metadata"]["image_status"], "error": row["metadata"]["image_error"]}
                              for row in rows if row["metadata"]["image_status"] != "ok"]}
    provenance_path = root / "raw" / dataset / "provenance.json"
    if provenance_path.is_file():
        audit["provenance"] = _json(provenance_path)
    audit["normalization_complete"] = True
    audit["all_images_ready"] = all(row["metadata"]["image_status"] == "ok" for row in rows)
    audit["unique_images"] = len({row["image_id"] for row in rows})
    audit["domains"] = dict(Counter(row["domain"] for row in rows))
    audit["geometry_adjustments"] = [{"sample_id": row["sample_id"], "adjustment": row["metadata"]["geometry_adjustment"],
                                      "original": row["metadata"]["raw_normalized_target_geometry"],
                                      "visible": row["target_geometry"]}
                                     for row in rows if row["metadata"].get("geometry_adjustment")]
    audit["geometry_validation"] = geometry_validation
    _write_json(root / "audits" / f"{dataset}.json", audit)
    with _LOCK:
        combined = root / "manifest.jsonl"
        temporary_combined = combined.with_suffix(".jsonl.tmp")
        with temporary_combined.open("wb") as output:
            for path in sorted(directory.glob("*.jsonl")):
                with path.open("rb") as stream:
                    shutil.copyfileobj(stream, output)
        os.replace(temporary_combined, combined)
    return target


def normalize_public_dataset(dataset: str, root: Path | str = "data/public_v1") -> Path:
    root = Path(root).resolve()
    adapters = {"osworld_g": _normalize_osworld, "screenspot_v2": _normalize_screenspot_v2,
                "screenspot_pro": _normalize_screenspot_pro, "venusbench_gd": _normalize_venus,
                "conflictgui": _normalize_conflict}
    if dataset not in adapters:
        raise NotImplementedError(f"Adapter being prepared: {dataset}")
    return _publish(root, dataset, adapters[dataset](root / "raw" / dataset))


def load_public_manifest(path: Path | str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
