"""V13 label-free CPU crop planning and frozen-Qwen multi-view extraction.

Plan train/val first. Only extraction constructs ContextFeatureRuntime. New
test images are read by extraction after the V13 scorer freeze, never for plan
selection. Only the four raw arrays may be memoized; learned outputs are rerun.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from PIL import Image
import torch

from clickagent.crossfit_proposer import RAW_ARRAY_KEYS, load_crossfit_proposer, predict_raw
from clickagent.model_features import ExtractionConfig
from clickagent.multiregion_proposals import (
    BUDGETS, MAX_CROP_IOU, MAX_REGIONS, build_c1_c3_pools,
    plan_multiregion_crops, public_raw_identity,
)


OUT = Path("runs/multiregion_v13")
BASE = Path("runs/crossfit_candidates_v12")
PUBLIC_FIELDS = ("sample_id", "image", "image_size", "image_sha256", "instruction", "dataset")
SOURCES = ("src/clickagent/multiregion_proposals.py", "scripts/extract_multiregion_v13.py")
CORE = tuple("src/clickagent/" + name + ".py" for name in (
    "crossfit_proposer", "candidate_ranker", "semantic_head_ranker", "attention_field_ranker",
    "point_aligned_readout", "point_aligned_inference", "context_verifier", "conditioned_features",
    "target_token", "model_features", "action_model", "public_model", "public_diagnostics"))
FRESH_SPLITS = ("test", "supplement")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    tmp.replace(path)


def save_features(path, arrays, metadata):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, metadata=np.asarray(json.dumps(metadata, ensure_ascii=False)), **arrays)
    temporary.replace(path)


def native(runtime, image, case, path, runtime_identity_sha):
    from clickagent.public_diagnostics import parse_offline_lax_click
    result = runtime.infer_text_action_public(image, case["instruction"], max_new_tokens=32)
    complete = result["terminated_by_eos"] or result["generated_text"].endswith(("\n", "\r"))
    parsed = parse_offline_lax_click(result["generated_text"]) if complete else None
    result["completed_lax_point"] = [round(v*(size-1)/1000) for v,size in zip(parsed,image.size)] if parsed else None
    result.update(sample_id=case["sample_id"], image_sha256=case["image_sha256"],
        instruction_sha256=hashlib.sha256(case["instruction"].encode()).hexdigest(),
        runtime_identity_sha256=runtime_identity_sha, ground_truth_in_model_inputs=False)
    write(path, result)


def key(sample_id):
    return hashlib.sha256(sample_id.encode()).hexdigest()[:24]


def norm(path):
    return os.path.normcase(os.path.abspath(path))


def cases(split):
    result = []
    for line in (OUT / "data" / (split+".jsonl")).read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        original = json.loads(line)
        public = {name: original[name] for name in PUBLIC_FIELDS}
        for name in ("raw_feature_path", "raw_path", "oof_provider", "image_fold"):
            if name in original:
                public[name] = original[name]
        result.append(public)
    return result


def raw_arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in RAW_ARRAY_KEYS}, json.loads(str(archive["metadata"]))


def verify_raw(path, case, transform=None, *, legacy=False):
    with np.load(path, allow_pickle=False) as archive:
        meta = json.loads(str(archive["metadata"]))
    expected_size = transform["crop_size"] if transform is not None else case["image_size"]
    assert meta["sample_id"] == case["sample_id"] and meta["image_sha256"] == case["image_sha256"]
    assert list(meta["image_size"]) == list(expected_size)
    assert meta["own_token_forward"] and meta["all_backbone_parameters_frozen"]
    assert meta["ground_truth_in_model_input"] is False
    if "instruction_sha256" in meta:
        assert meta["instruction_sha256"] == hashlib.sha256(case["instruction"].encode()).hexdigest()
    else:
        assert legacy, "New raw archives require an instruction hash"
    if transform is not None:
        assert meta["transform"] == transform
    return meta


def runtime_config(contract):
    cfg = dict(contract)
    cfg["offload_folder"] = str((OUT / "offload").absolute())
    cfg["layers"] = tuple(cfg["layers"]); cfg["conditions"] = tuple(cfg["conditions"])
    cfg["max_memory"] = {int(k) if str(k).isdigit() else k: v for k, v in cfg["max_memory"].items()}
    result = ExtractionConfig(**cfg)
    normalized = asdict(result); normalized.pop("offload_folder")
    assert json.loads(json.dumps(normalized)) == contract
    return result


class FrozenInputs:
    def __init__(self):
        self.seal_path = BASE / "frozen.json"; self.seal = read(self.seal_path)
        assert self.seal["complete"]
        self.evidence = {norm(path): value for section in ("files", "training_evidence")
                         for path, value in self.seal[section].items()}
        self.verified = {}
        self.runtime_path = BASE / "runtime_identity.json"
        self.check(self.runtime_path)
        self.base_runtime = read(self.runtime_path); assert self.base_runtime["complete"]
        self.contract = self.base_runtime["runtime_contract"]
        for path in CORE:
            self.check(path)
        versions = {name: importlib.metadata.version(name) for name in self.base_runtime["library_versions"]}
        assert versions == self.base_runtime["library_versions"]
        weights = []
        for entry in self.base_runtime["actual_model_files"]:
            stat = Path(entry["path"]).stat()
            assert stat.st_size == entry["bytes"] and stat.st_mtime_ns == entry["mtime_ns"]
            weights.append(dict(path=entry["path"], sha256=entry["sha256"], bytes=stat.st_size, mtime_ns=stat.st_mtime_ns))
        self.models = {}; self.model_sha = {}; self.model_meta = {}
        for role in [f"fold_{i}" for i in range(5)]+["full"]:
            path = BASE / "proposers" / (role+".pt"); sidecar = path.with_suffix(".json")
            self.check(path); self.check(sidecar)
            meta = read(sidecar); assert meta["complete"] and meta["checkpoint_sha256"] == sha(path)
            self.models[role] = load_crossfit_proposer(path, "cpu")[0].eval()
            self.model_sha[role] = meta["checkpoint_sha256"]; self.model_meta[role] = meta
        self.identity = dict(complete=True, base_frozen_path=str(self.seal_path), base_frozen_sha256=sha(self.seal_path),
            base_runtime_identity_path=str(self.runtime_path), base_runtime_identity_sha256=sha(self.runtime_path),
            runtime_contract=self.contract, runtime_contract_sha256=self.base_runtime["runtime_contract_sha256"],
            current_core_source_sha256={path: sha(path) for path in CORE},
            implementation_source_sha256={path: sha(path) for path in SOURCES}, library_versions=versions,
            model_files=weights, model_identity_level="Authenticated V12 full-byte hashes; current sizes and mtimes equal V12 inventory",
            proposer_sha256=self.model_sha, offload_folder=str((OUT/"offload").absolute()),
            ground_truth_in_model_inputs=False)
        identity_path = OUT / "runtime_identity.json"
        if identity_path.exists():
            assert read(identity_path) == self.identity, "V13 runtime/source identity drift"
        else:
            write(identity_path, self.identity)
        self.runtime_identity_sha = sha(identity_path)
        self.pipelines = {}
        self.base_cases = {}
        self.check(BASE / "data_selection.json")
        for split in ("train", "val"):
            path = BASE / "data" / (split+".jsonl")
            # V12 data is transitively bound by the frozen combined manifest;
            # authenticate directly against its sealed selection inventory too.
            selection = read(BASE / "data_selection.json")
            assert sha(path) == selection["split_manifest_sha256"][split]
            for line in path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                self.base_cases[row["sample_id"]] = {name: row[name] for name in PUBLIC_FIELDS}

    def check(self, path):
        expected = self.evidence.get(norm(path))
        assert expected, f"Not authenticated by V12: {path}"
        stat = Path(path).stat(); cache_key = (norm(path), stat.st_size, stat.st_mtime_ns)
        if self.verified.get(cache_key) != expected:
            assert sha(path) == expected, f"Frozen file drift: {path}"
            self.verified[cache_key] = expected
        return expected

    def role(self, case, split):
        if split != "train":
            return "full"
        role = case["oof_provider"]
        assert role == f"fold_{int(case['image_fold'])}"
        assert case["sample_id"] in self.model_meta[role]["heldout_ids"]
        assert case["sample_id"] not in self.model_meta[role]["allowed_train_ids"]
        return role

    def pipeline(self, case, group):
        path = BASE / "trainval" / group / "pipeline" / (key(case["sample_id"])+".json")
        if not path.exists():
            return None
        if str(path) not in self.pipelines:
            self.check(path); self.pipelines[str(path)] = read(path)
        return self.pipelines[str(path)]

    def global_path(self, case, split):
        if split in FRESH_SPLITS:
            return OUT / "raw/global" / (key(case["sample_id"])+".npz")
        prior = self.pipeline(case, "oof" if split == "train" else "val")
        assert prior is not None
        path = Path(case.get("raw_feature_path") or case.get("raw_path") or prior["global_feature_path"])
        assert norm(path) == norm(prior["global_feature_path"])
        self.check(path)
        assert self.base_cases[case["sample_id"]] == {name: case[name] for name in PUBLIC_FIELDS}
        verify_raw(path, case, legacy=True)
        return path

    def local_match(self, case, transform):
        for group in ("oof", "in_sample", "val"):
            prior = self.pipeline(case, group)
            if prior is None or prior.get("old_crop_transform") != transform or not prior.get("local_feature_path"):
                continue
            path = Path(prior["local_feature_path"])
            self.check(path); verify_raw(path, case, transform)
            return dict(feature_path=str(path), feature_sha256=sha(path), source="v12_authenticated_raw", group=group)
        return None


def fingerprint(case, role, global_path, frozen):
    return digest(dict(case={k: case[k] for k in PUBLIC_FIELDS}, proposer_role=role,
        proposer_sha256=frozen.model_sha[role], global_feature_sha256=sha(global_path),
        runtime_identity_sha256=frozen.runtime_identity_sha, max_regions=MAX_REGIONS, max_iou=MAX_CROP_IOU))


def make_plan(case, split, frozen):
    started = time.perf_counter(); role = frozen.role(case, split); path = frozen.global_path(case, split)
    destination = OUT / "plan" / split / (key(case["sample_id"])+".json")
    expected = fingerprint(case, role, path, frozen)
    if destination.exists():
        result = read(destination); assert result["fingerprint"] == expected
        return result
    arrays, meta = raw_arrays(path)
    result = predict_raw(frozen.models[role], arrays, meta, case["image_size"], topk=64)
    if split in ("train", "val"):
        prior = frozen.pipeline(case, "oof" if split == "train" else "val")
        assert result["all_points"] == prior["global_result"]["all_points"], "CPU4 V12 global integer proposal drift"
        assert np.allclose(result["all_scores"], prior["global_result"]["all_scores"], atol=1e-6, rtol=1e-6)
    crops = plan_multiregion_crops(result, case["image_size"])
    for crop in crops:
        transform = crop["transform"]
        identity = public_raw_identity(case["sample_id"], case["image_sha256"], case["instruction"], transform,
                                       frozen.identity["runtime_contract_sha256"])
        match = frozen.local_match(case, transform) if split not in FRESH_SPLITS else None
        crop.update(raw_identity=identity, raw_reuse=match,
                    new_feature_path=str(OUT/"raw/local"/(identity+".npz")))
    plan = dict(complete=True, sample_id=case["sample_id"], split=split, fingerprint=expected,
        proposer_role=role, proposer_sha256=frozen.model_sha[role], global_feature_path=str(path),
        global_feature_sha256=sha(path), global_result=result, crops=crops,
        runtime_identity_sha256=frozen.runtime_identity_sha, ground_truth_in_model_inputs=False,
        global_learned_output_recomputed=True, plan_cpu_seconds=time.perf_counter()-started)
    write(destination, plan); return plan


def assert_test_frozen():
    path = OUT / "frozen.json"; seal = read(path); assert seal["complete"]
    for source, expected in seal["files"].items():
        assert sha(source) == expected, source


def materialize(case, transform):
    assert sha(case["image"]) == case["image_sha256"]
    with Image.open(case["image"]) as opened:
        image = opened.convert("RGB")
    assert list(image.size) == case["image_size"]
    crop = image.crop(transform["box"])
    crop = crop.resize(tuple(transform["crop_size"]), Image.Resampling.LANCZOS)
    return crop


def extract_case(case, split, frozen, get_runtime):
    started = time.perf_counter(); destination = OUT / "pipeline" / split / (key(case["sample_id"])+".json")
    global_forwards = 0; global_forward_seconds = None; native_path = None; native_seconds = None
    if split in FRESH_SPLITS:
        global_path = frozen.global_path(case, split)
        if not global_path.exists():
            assert sha(case["image"]) == case["image_sha256"]
            with Image.open(case["image"]) as opened:
                image = opened.convert("RGB")
            runtime = get_runtime(); before = time.perf_counter()
            arrays, meta = runtime.infer_spatial_features(image, case["instruction"])
            global_forward_seconds = time.perf_counter()-before; global_forwards = 1
            meta.update(sample_id=case["sample_id"], image_sha256=case["image_sha256"],
                        instruction_sha256=hashlib.sha256(case["instruction"].encode()).hexdigest())
            save_features(global_path, arrays, meta)
        verify_raw(global_path, case)
        native_path = OUT/"native"/split/(key(case["sample_id"])+".json")
        if not native_path.exists():
            with Image.open(case["image"]) as opened:
                image = opened.convert("RGB")
            active = get_runtime(); before = time.perf_counter()
            native(active, image, case, native_path, frozen.runtime_identity_sha)
            native_seconds = time.perf_counter()-before
    plan = make_plan(case, split, frozen)
    if destination.exists():
        previous = read(destination); assert previous["plan_sha256"] == sha(OUT/"plan"/split/(key(case["sample_id"])+".json"))
        assert previous["pools"] == build_c1_c3_pools(previous["global_result"], previous["local_views"])
        for view in previous["local_views"]:
            assert sha(view["feature_path"]) == view["feature_sha256"]
            verify_raw(view["feature_path"], case, view["transform"])
        return previous
    views = []; forwards = 0
    for crop_plan in plan["crops"]:
        transform = crop_plan["transform"]; match = crop_plan["raw_reuse"]
        path = Path(match["feature_path"] if match else crop_plan["new_feature_path"])
        raw_reused = path.exists(); elapsed = None
        if not raw_reused:
            crop = materialize(case, transform); active_runtime = get_runtime(); before = time.perf_counter()
            arrays, meta = active_runtime.infer_spatial_features(crop, case["instruction"])
            elapsed = time.perf_counter()-before; crop.close(); forwards += 1
            meta.update(sample_id=case["sample_id"], image_sha256=case["image_sha256"], transform=transform,
                        instruction_sha256=hashlib.sha256(case["instruction"].encode()).hexdigest())
            save_features(path, arrays, meta)
        verify_raw(path, case, transform)
        if match is not None:
            assert sha(path) == match["feature_sha256"]
        arrays, meta = raw_arrays(path); before = time.perf_counter()
        decoded = predict_raw(frozen.models[plan["proposer_role"]], arrays, meta, transform["crop_size"], topk=64)
        decode_seconds = time.perf_counter()-before
        views.append(dict(region_index=crop_plan["region_index"], feature_path=str(path), feature_sha256=sha(path),
            transform=transform, result=decoded, raw_reused=raw_reused, raw_reuse_provenance=match,
            raw_identity=crop_plan["raw_identity"], current_raw_forward_wall_seconds=elapsed,
            current_cpu_decode_seconds=decode_seconds, learned_output_recomputed=True))
    pools = build_c1_c3_pools(plan["global_result"], views)
    if split in ("train", "val"):
        prior = frozen.pipeline(case, "oof" if split == "train" else "val")
        assert {k:[x["point"] for x in v] for k,v in pools["c1"].items()} == {k:[x["point"] for x in v] for k,v in prior["pools"].items()}, "V12 single-crop control point drift"
    result = dict(complete=True, sample_id=case["sample_id"], split=split, global_feature_path=plan["global_feature_path"],
        global_feature_sha256=plan["global_feature_sha256"], global_result=plan["global_result"],
        local_views=views, pools=pools,
        proposer_role=plan["proposer_role"], proposer_sha256=plan["proposer_sha256"],
        plan_sha256=sha(OUT/"plan"/split/(key(case["sample_id"])+".json")),
        runtime_identity_sha256=frozen.runtime_identity_sha, ground_truth_in_model_inputs=False,
        current_local_raw_forward_count=forwards, current_global_raw_forward_count=global_forwards,
        current_global_raw_forward_wall_seconds=global_forward_seconds,
        native_path=str(native_path) if native_path is not None else None,
        native_sha256=sha(native_path) if native_path is not None else None,
        current_native_wall_seconds=native_seconds, current_case_wall_seconds=time.perf_counter()-started)
    write(destination, result); return result


def run(stage, splits, limit=None):
    torch.set_num_threads(4)
    if any(split in FRESH_SPLITS for split in splits):
        assert_test_frozen()
        assert stage == "extract", "Fresh test global raw must be extracted after freeze"
    if stage == "extract":
        assert (OUT/"protocol.json").exists(), "Register the V13 protocol before Qwen extraction"
    frozen = FrozenInputs(); runtime = None
    def get_runtime():
        nonlocal runtime
        if runtime is None:
            from clickagent.context_verifier import ContextFeatureRuntime
            runtime = ContextFeatureRuntime(runtime_config(frozen.contract))
            write(OUT/"current_runtime.json", runtime.runtime_metadata)
        return runtime
    for split in splits:
        cohort = cases(split); selected = cohort[:limit] if limit is not None else cohort
        entries = []; n_crops = reuse = 0; started = time.perf_counter()
        for i, case in enumerate(selected, 1):
            result = make_plan(case, split, frozen) if stage == "plan" else extract_case(case, split, frozen, get_runtime)
            plan = result if stage == "plan" else read(OUT/"plan"/split/(key(case["sample_id"])+".json"))
            n_crops += len(plan["crops"]); reuse += sum(c["raw_reuse"] is not None for c in plan["crops"])
            path = OUT/stage/split/(key(case["sample_id"])+".json") if stage == "plan" else OUT/"pipeline"/split/(key(case["sample_id"])+".json")
            entries.append(dict(sample_id=case["sample_id"], path=str(path), sha256=sha(path),
                crop_count=len(plan["crops"]), raw_reuse_count=sum(c["raw_reuse"] is not None for c in plan["crops"])))
            if i % 25 == 0:
                print(json.dumps(dict(stage=stage, split=split, i=i, n=len(selected), crops=n_crops, raw_reuse=reuse,
                                     elapsed_seconds=time.perf_counter()-started)), flush=True)
        index = dict(complete=len(selected)==len(cohort), stage=stage, split=split, n=len(selected),
            n_cohort=len(cohort), crops=n_crops, authenticated_old_raw_hits=reuse, new_raw_inputs=n_crops-reuse,
            runtime_identity_sha256=frozen.runtime_identity_sha, manifest_sha256=sha(OUT/"data"/(split+".jsonl")),
            cases=entries, elapsed_seconds=time.perf_counter()-started, gpu_model_loaded=runtime is not None)
        write(OUT/("plan" if stage == "plan" else "pipeline")/(split+"_index.json"), index)
        print(json.dumps({k:v for k,v in index.items() if k != "cases"}), flush=True)
    assert frozen.identity["implementation_source_sha256"] == {path:sha(path) for path in SOURCES}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["plan", "extract"], required=True)
    parser.add_argument("--split", choices=["train", "val", "trainval", "test", "supplement"], default="trainval")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    run(args.stage, ("train", "val") if args.split == "trainval" else (args.split,), args.limit)


if __name__ == "__main__":
    main()
