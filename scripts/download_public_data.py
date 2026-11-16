"""Complete public benchmark acquisition; raw files and errors stay on disk."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from clickagent.public_data import (DATASET_NAMES, download_public_dataset, normalize_public_dataset,
                                    recover_screenspot_pro_from_modelscope)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/public_v1"))
    parser.add_argument("--datasets", nargs="+", choices=list(DATASET_NAMES), default=list(DATASET_NAMES))
    parser.add_argument("--workers", type=int, default=4, help="Parallel files per dataset")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--normalize-only", action="store_true")
    parser.add_argument("--recover-pro-from-modelscope", action="store_true",
                        help="Recover only SHA256-identical original images from the differently packaged Pro mirror")
    parser.add_argument("--arrow-path", type=Path, help="Optional separately installed PyArrow directory")
    args = parser.parse_args()
    if args.download_only and args.normalize_only:
        parser.error("Choose at most one stage-only flag")
    if args.recover_pro_from_modelscope and (args.normalize_only or "screenspot_pro" not in args.datasets):
        parser.error("Mirror recovery requires a download stage containing screenspot_pro")
    def run(dataset):
        if not args.normalize_only:
            if dataset == "screenspot_pro" and args.recover_pro_from_modelscope:
                result = recover_screenspot_pro_from_modelscope(args.root, workers=min(args.workers, 4),
                                                                arrow_path=args.arrow_path)
                print(json.dumps({"dataset": dataset, "stage": "mirror_recovery_complete",
                                  "recovered_images": len(result["recovered_images"]),
                                  "rejected_images": len(result["rejected_images"])}), flush=True)
            result = download_public_dataset(dataset, args.root, workers=args.workers)
            print(json.dumps({"dataset": dataset, "stage": "download_complete", "revision": result["revision"]}), flush=True)
        if not args.download_only:
            manifest = normalize_public_dataset(dataset, args.root)
            print(json.dumps({"dataset": dataset, "manifest": str(manifest)}), flush=True)
    errors = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.datasets)) as pool:
        futures = {pool.submit(run, dataset): dataset for dataset in args.datasets}
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as error:
                item = {"dataset": futures[future], "error": str(error)}
                errors.append(item)
                print(json.dumps(item, ensure_ascii=True), flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
