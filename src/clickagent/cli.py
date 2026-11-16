"""Command-line entry points; heavy model libraries load only for extraction."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
from pathlib import Path
import sys

DEFAULT_MODEL_DIR = Path("models/Qwen3-VL-8B-Instruct")


def _print(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def _extraction_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--task", choices=["intent", "visual_state"], default="intent")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--layers", nargs="+", type=int, default=[15, 28, 36], help="One-based language decoder layers")
    parser.add_argument("--dtype", choices=["float16", "float32", "bfloat16"], default="float16")
    parser.add_argument("--device-map", default="balanced")
    parser.add_argument("--gpu-memory", help="Per-GPU weight budget, e.g. 13GiB")
    parser.add_argument("--cpu-memory", default="24GiB")
    parser.add_argument("--max-pixels", type=int, default=262144)
    parser.add_argument("--min-pixels", type=int, default=16384)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--conditions", nargs="+", choices=["prefill", "post_decision", "post_response"],
                        default=["prefill", "post_decision", "post_response"])
    parser.add_argument("--no-image", action="store_true", help="VLM language-only ablation with the same prompt")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--limit", type=int, help="First N manifest rows, for extraction smoke checks only")
    parser.add_argument("--seed", type=int, default=42)


def _extract(args: argparse.Namespace, output: Path) -> dict:
    from .dataset import load_manifest
    from .model_features import extract_manifest

    # Validate the whole manifest before a limit can hide screenshot leakage.
    load_manifest(args.manifest, validate_images=not args.no_image)
    return extract_manifest(args.manifest, output, _extraction_config(args, output), limit=args.limit)


def _extraction_config(args: argparse.Namespace, output: Path):
    from .model_features import ExtractionConfig

    max_memory = None
    if args.gpu_memory:
        import torch
        max_memory = {index: args.gpu_memory for index in range(torch.cuda.device_count())}
        max_memory["cpu"] = args.cpu_memory
    return ExtractionConfig(
        model_path=str(args.model.resolve()), layers=tuple(args.layers), dtype=args.dtype,
        device_map=args.device_map, max_memory=max_memory,
        offload_folder=str((output / "offload").resolve()),
        min_pixels=args.min_pixels, max_pixels=args.max_pixels,
        max_new_tokens=args.max_new_tokens, conditions=tuple(args.conditions),
        no_image=args.no_image, resume=not args.no_resume, seed=args.seed,
        task=args.task,
    )


def doctor(model: Path) -> dict:
    result = {"python": sys.version, "executable": sys.executable, "packages": {}, "gpus": []}
    for name in ("torch", "torchvision", "transformers", "accelerate", "modelscope", "numpy", "scikit-learn", "Pillow"):
        try:
            result["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result["packages"][name] = None
    try:
        import torch
        result["cuda_available"] = torch.cuda.is_available()
        result["cuda_build"] = torch.version.cuda
        result["torch_architectures"] = torch.cuda.get_arch_list()
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            # Also launch a tiny kernel: driver discovery alone does not verify architecture support.
            test = torch.ones(8, device=f"cuda:{index}", dtype=torch.float16).sum().item()
            result["gpus"].append({"index": index, "name": props.name,
                                   "memory_gib": round(props.total_memory / 2**30, 2),
                                   "compute_capability": list(torch.cuda.get_device_capability(index)),
                                   "float16_kernel_ok": test == 8.0})
    except Exception as exc:
        result["torch_error"] = f"{type(exc).__name__}: {exc}"
    try:
        from .download import verify_model
        result["model"] = verify_model(model)
    except (OSError, ValueError, KeyError) as exc:
        result["model"] = {"ready": False, "detail": str(exc)}
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="clickagent", description="Frozen Qwen3-VL intent and visual-state gates")
    parser.add_argument("--verbose", action="store_true")
    commands = parser.add_subparsers(dest="command", required=True)

    data = commands.add_parser("dataset", help="Generate paired synthetic GUI screenshots")
    data.add_argument("--output", type=Path, default=Path("data/synthetic"))
    data.add_argument("--screenshots", type=int, default=120)
    data.add_argument("--samples-per-image", type=int, default=6)
    data.add_argument("--seed", type=int, default=42)
    data.add_argument("--split-strategy", choices=["group", "layout"], default="group")
    data.add_argument("--holdout-family", choices=["dialog", "settings", "toolbar", "table", "webpage", "filemanager"])
    data.add_argument("--holdout-template-family", choices=["direct", "touch", "quoted", "negation", "request", "execute"])

    visual_data = commands.add_parser("visual-state-dataset", help="Generate matched enabled/disabled/absent Save screenshots")
    visual_data.add_argument("--output", type=Path, default=Path("data/visual_state_v2_matched"))
    visual_data.add_argument("--scenarios", type=int, default=300)
    visual_data.add_argument("--seed", type=int, default=43)

    validate = commands.add_parser("validate", help="Audit a synthetic or imported real screenshot manifest")
    validate.add_argument("manifest", type=Path)

    download = commands.add_parser("download", help="Download weights from ModelScope")
    download.add_argument("--output", type=Path, default=DEFAULT_MODEL_DIR)
    download.add_argument("--model-id", default="Qwen/Qwen3-VL-8B-Instruct")
    download.add_argument("--revision", default="master")
    download.add_argument("--workers", type=int, default=4)

    extract = commands.add_parser("extract", help="Save attention and hidden features; never updates the VLM")
    _extraction_arguments(extract)
    extract.add_argument("--output", type=Path, required=True)

    constant = commands.add_parser("constant-no-image", help="Verify identical no-image inputs and reuse their real frozen-model features")
    _extraction_arguments(constant)
    constant.set_defaults(task="visual_state", no_image=True, conditions=["prefill"])
    constant.add_argument("--output", type=Path, required=True)
    constant.add_argument("--verification-forwards", type=int, default=3)

    evaluate = commands.add_parser("evaluate", help="Fit small gates on train; select C/thresholds on val; report test")
    evaluate.add_argument("--features", type=Path, required=True)
    evaluate.add_argument("--manifest", type=Path, required=True)
    evaluate.add_argument("--task", choices=["intent", "visual_state"], default="intent")
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--seed", type=int, default=42)
    evaluate.add_argument("--min-common-coverage", type=float, default=0.8)
    evaluate.add_argument("--bootstrap-iterations", type=int, default=1000)

    text_baseline = commands.add_parser("text-baseline", help="Run the instruction-only control without VLM features or GPUs")
    text_baseline.add_argument("--manifest", type=Path, required=True)
    text_baseline.add_argument("--output", type=Path, required=True)
    text_baseline.add_argument("--seed", type=int, default=42)
    text_baseline.add_argument("--task", choices=["intent", "visual_state"], default="intent")
    text_baseline.add_argument("--bootstrap-iterations", type=int, default=1000)

    run = commands.add_parser("run", help="Run feature extraction and probe evaluation sequentially")
    _extraction_arguments(run)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--min-common-coverage", type=float, default=0.8)
    run.add_argument("--bootstrap-iterations", type=int, default=1000)

    check = commands.add_parser("doctor", help="Check CUDA kernels, packages and model shard presence")
    check.add_argument("--model", type=Path, default=DEFAULT_MODEL_DIR)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    if args.command == "dataset":
        from .dataset import build_synthetic_dataset
        manifest = build_synthetic_dataset(args.output, args.screenshots, args.seed, args.samples_per_image,
                                            args.split_strategy, args.holdout_family, args.holdout_template_family)
        _print({"manifest": manifest, "summary": json.loads((manifest.parent / "summary.json").read_text(encoding="utf-8"))})
    elif args.command == "validate":
        from .dataset import load_manifest, validate_manifest
        records = load_manifest(args.manifest)
        _print(validate_manifest(records, args.manifest))
    elif args.command == "visual-state-dataset":
        from .visual_state_dataset import build_visual_state_dataset
        manifest = build_visual_state_dataset(args.output, args.scenarios, args.seed)
        _print({"manifest": manifest, "summary": json.loads((manifest.parent / "summary.json").read_text(encoding="utf-8"))})
    elif args.command == "download":
        from .download import download_model
        _print(download_model(args.output, args.model_id, args.revision, args.workers))
    elif args.command == "extract":
        _print(_extract(args, args.output))
    elif args.command == "constant-no-image":
        from .constant_no_image import export_constant_no_image_control
        if args.limit is not None:
            parser.error("constant-no-image does not accept --limit; it exports the complete manifest")
        _print(export_constant_no_image_control(args.manifest, args.output,
                                                _extraction_config(args, args.output),
                                                verification_forwards=args.verification_forwards))
    elif args.command == "text-baseline":
        from .probes import run_text_baseline
        summary = run_text_baseline(args.manifest, args.output, seed=args.seed, task=args.task,
                                    bootstrap_iterations=args.bootstrap_iterations)
        _print({"report": args.output / "report.md", "test": summary["runs"][0]["test"]})
    elif args.command in {"run", "evaluate"}:
        from .probes import run_probe_suite
        features = args.output / "features" if args.command == "run" else args.features
        report_dir = args.output / "evaluation" if args.command == "run" else args.output
        if args.command == "run":
            _print(_extract(args, features))
        summary = run_probe_suite(features, report_dir, seed=args.seed,
                                  manifest_path=args.manifest, min_common_coverage=args.min_common_coverage,
                                  task=args.task, bootstrap_iterations=args.bootstrap_iterations)
        _print({"report": report_dir / "report.md", "coverage": summary["coverage"],
                "test": [{"condition": r["condition"], "features": r["feature_set"], **r["test"]}
                         for r in summary["runs"]]})
    elif args.command == "doctor":
        _print(doctor(args.model))


if __name__ == "__main__":
    main()

