"""Can also run using the project-local ModelScope bootstrap installation."""

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
if (PROJECT / ".bootstrap").is_dir():
    sys.path.insert(0, str(PROJECT / ".bootstrap"))

from clickagent.download import download_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=PROJECT / "models" / "Qwen3-VL-8B-Instruct")
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--revision", default="master")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(download_model(args.output, args.model_id, args.revision, args.workers), indent=2))
