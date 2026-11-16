"""Fetch and verify the fixed official MMBench-GUI L2 source snapshot.

The ModelScope OpenGVLab mirror carries the same archive as the authors'
Hugging Face commit e27757e. This script transports files only; it never
opens annotations or target boxes.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from modelscope.hub.file_download import dataset_file_download


ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data/mmbench_gui_l2_v1/source"
SOURCE = "OpenGVLab/MMBench-GUI"
FILES = {
    "L2_annotations.json": ("49fe9873be87c6eba8a087707a496aa1ddce35885818874cef4b3bdfb91f5999", None),
    "MMBench-GUI-OfflineImages.zip": ("56e087c3b5c2941e2bc51c2f347a3d46f41ca50f1c0370369c284aa02c6458a3", 1082948866),
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    for name, (expected_hash, expected_size) in FILES.items():
        path = DEST / name
        if not path.exists():
            got = Path(dataset_file_download(SOURCE, name, local_dir=str(DEST), revision="master"))
            assert got.resolve() == path.resolve(), f"Unexpected download path: {got}"
        if expected_size is not None:
            assert path.stat().st_size == expected_size, f"Unexpected byte count: {name}"
        assert digest(path) == expected_hash, f"Source SHA256 mismatch: {name}"
        print(f"verified {name}: {path.stat().st_size} bytes, SHA256={expected_hash}", flush=True)


if __name__ == "__main__":
    main()
