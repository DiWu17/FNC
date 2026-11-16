"""Verify the prospective paper-method V1 snapshot without loading a VLM."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "research/paper/method_freeze_v1.json"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_file(path: Path, expected: str, errors: list[str]) -> None:
    if not path.is_file():
        errors.append(f"missing: {path}")
    elif digest(path) != expected:
        errors.append(f"sha256 drift: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive-only", action="store_true",
        help="Verify archived paper sources even if the active working sources changed.",
    )
    parser.add_argument(
        "--full-backbone", action="store_true",
        help="Also hash every local Qwen file, including large weight shards.",
    )
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["freeze_id"] == "paper_method_v1"
    errors: list[str] = []

    for item in manifest["source_archive"]:
        verify_file(ROOT / item["archived"], item["sha256"], errors)
        if not args.archive_only:
            verify_file(ROOT / item["original"], item["sha256"], errors)

    for item in manifest["artifact_hashes"]:
        verify_file(ROOT / item["path"], item["sha256"], errors)

    for item in manifest["backbone_files"]:
        path = ROOT / item["path"]
        if not path.is_file():
            errors.append(f"missing backbone file: {path}")
            continue
        if path.stat().st_size != item["bytes"]:
            errors.append(f"backbone size drift: {path}")
            continue
        if args.full_backbone or item["bytes"] < 100 * 1024 * 1024:
            verify_file(path, item["sha256"], errors)

    if errors:
        print("METHOD FREEZE VERIFICATION FAILED")
        for message in errors:
            print(" -", message)
        return 1
    print(
        "METHOD FREEZE VERIFIED: "
        f"{len(manifest['source_archive'])} archived sources, "
        f"{len(manifest['artifact_hashes'])} artifacts, "
        f"{len(manifest['backbone_files'])} backbone files "
        f"({'full hashes' if args.full_backbone else 'small-file hashes and weight sizes'})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
