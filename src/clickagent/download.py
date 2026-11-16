"""Download model files from ModelScope into the project, then verify shards."""

from __future__ import annotations

import json
import os
from pathlib import Path

DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"


def verify_model(model_dir: Path) -> dict:
    model_dir = Path(model_dir).resolve()
    required = ["config.json", "tokenizer_config.json", "preprocessor_config.json"]
    missing = [name for name in required if not (model_dir / name).is_file()]
    index = model_dir / "model.safetensors.index.json"
    if index.is_file():
        manifest = json.loads(index.read_text(encoding="utf-8"))
        shard_names = sorted(set(manifest["weight_map"].values()))
    elif (model_dir / "model.safetensors").is_file():
        shard_names = ["model.safetensors"]
    else:
        shard_names = []
        missing.append("model.safetensors or model.safetensors.index.json")
    for name in shard_names:
        path = model_dir / name
        if not path.is_file() or path.stat().st_size == 0:
            missing.append(name)
    if missing:
        raise FileNotFoundError(f"Incomplete model in {model_dir}: {missing}")
    return {
        "model_dir": str(model_dir),
        "shards": shard_names,
        "weight_bytes": sum((model_dir / name).stat().st_size for name in shard_names),
        "verification": "required files and shard presence; SDK verifies downloaded files",
    }


def download_model(
    output_dir: Path,
    model_id: str = DEFAULT_MODEL,
    revision: str = "master",
    workers: int = 4,
) -> dict:
    # The SDK otherwise defaults to a single connection per multi-GB shard.
    # Respect explicit user overrides; all streams still use SDK verification.
    os.environ.setdefault("MODELSCOPE_DOWNLOAD_PARALLEL_WORKERS", "8")
    os.environ.setdefault("MODELSCOPE_DOWNLOAD_TOTAL_TIMEOUT", "7200")
    from modelscope import snapshot_download

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        model_id=model_id,
        revision=revision,
        local_dir=str(output_dir),
        cache_dir=str(output_dir.parent / ".modelscope-cache"),
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "*.model", "*.tiktoken"],
        max_workers=workers,
    )
    result = verify_model(output_dir)
    result.update({"model_id": model_id, "revision": revision, "source": "ModelScope"})
    (output_dir / "clickagent_download.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result

