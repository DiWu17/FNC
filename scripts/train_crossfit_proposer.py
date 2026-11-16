"""Train six fresh proposers using explicit fold allowlists and old validation.

No legacy learned checkpoint, teacher point cache, or train-derived head prior
is read.  Held-out training-fold geometry is never used by this process.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from clickagent.attention_field_ranker import make_field_batch
from clickagent.crossfit_proposer import (CrossfitProposerConfig, RAW_ARRAY_KEYS,
    build_raw_record, make_fresh_proposer, save_crossfit_proposer, load_crossfit_proposer, state_sha256)
from clickagent.point_aligned_readout import aligned_loss, all_candidate_points, candidate_point_hits


SOURCE_PATHS = (
    "src/clickagent/crossfit_proposer.py", "scripts/train_crossfit_proposer.py",
    "src/clickagent/point_aligned_readout.py", "src/clickagent/semantic_head_ranker.py",
    "src/clickagent/attention_field_ranker.py", "src/clickagent/candidate_ranker.py",
    "src/clickagent/spatial_geometry.py",
)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_cases(path):
    path = Path(path)
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    value = read_json(path)
    return value if isinstance(value, list) else value["cases"]


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def raw_path(case):
    explicit = next((case.get(name) for name in ("raw_path", "raw_feature_path", "global_feature_path", "feature_path") if case.get(name)), None)
    if explicit:
        return Path(explicit)
    name = hashlib.sha256(case["sample_id"].encode()).hexdigest()[:24] + ".npz"
    first = Path("runs/semantic_head_v7/features") / name
    return first if first.exists() else Path("runs/conditioned_v3/features") / name


def load_record(case):
    path = raw_path(case)
    identity = dict(path=str(path.resolve()), sha256=sha(path), size=path.stat().st_size)
    with np.load(path, allow_pickle=False) as archive:
        meta = json.loads(str(archive["metadata"]))
        if meta["sample_id"] != case["sample_id"] or meta["image_sha256"] != case["image_sha256"]:
            raise ValueError(f"Feature/case identity mismatch: {case['sample_id']}")
        if list(meta["image_size"]) != list(case["image_size"]):
            raise ValueError("Feature/case size mismatch")
        if meta.get("ground_truth_in_model_input") is not False or not meta.get("all_backbone_parameters_frozen"):
            raise ValueError("Raw features must be label-free frozen-backbone outputs")
        arrays = {name: archive[name] for name in RAW_ARRAY_KEYS} if meta["target_available"] else {}
        record = build_raw_record(arrays, meta, case["image_size"], geometry=case["target_geometry"])
    record["case"] = case
    record["raw_identity"] = identity
    return record


def role_cases(train, val, folds, role):
    """Filter before constructing supervision, and independently check images."""
    if role not in {"full", *(f"fold_{i}" for i in range(5))}:
        raise ValueError("Unknown crossfit role")
    ids = [row["sample_id"] for row in train]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate training sample IDs")
    spec = folds["roles"][role]
    allowed = set(spec["allowed_train_ids"])
    heldout = set(spec["heldout_ids"])
    if allowed & heldout or allowed | heldout != set(ids):
        raise ValueError("Fold allowlist must partition the complete training cohort")
    if role == "full" and (heldout or allowed != set(ids)):
        raise ValueError("Full proposer must use exactly all training cases")
    if role != "full":
        fold = int(role.rsplit("_", 1)[-1])
        mapping = folds["sample_to_fold"]
        if heldout != {sid for sid in ids if int(mapping[sid]) == fold}:
            raise ValueError("Heldout IDs disagree with fold map")
    selected = [row for row in train if row["sample_id"] in allowed]
    excluded = [row for row in train if row["sample_id"] in heldout]
    selected_images = {row["pixel_sha256"] for row in selected}
    excluded_images = {row["pixel_sha256"] for row in excluded}
    if selected_images & excluded_images:
        raise ValueError("The same screenshot occurs on both sides of a fold")
    if {row["sample_id"] for row in val} & set(ids):
        raise ValueError("Validation sample IDs overlap train")
    if {row["pixel_sha256"] for row in val} & {row["pixel_sha256"] for row in train}:
        raise ValueError("Validation images overlap train")
    return selected, sorted(allowed), sorted(heldout)


@torch.inference_mode()
def evaluate(model, records, device="cpu"):
    model.eval()
    hits = regions = oracle = named = 0
    weighted_loss = 0.
    for start in range(0, len(records), 8):
        group = [r for r in records[start:start+8] if r["metadata"]["target_available"]]
        if not group:
            continue
        inputs, targets = make_field_batch(group, device, labels=True)
        output = model(**inputs)
        actual = candidate_point_hits(all_candidate_points(output, group), group)
        point_labels = torch.from_numpy(np.concatenate(actual)).to(device)
        loss, _ = aligned_loss(output, targets, point_labels, True, region_weight=.25)
        weighted_loss += float(loss) * len(group)
        scores = output["scores"].float().cpu().numpy(); cursor = 0
        for record, candidate_hits in zip(group, actual):
            n = len(candidate_hits); chosen = int(scores[cursor:cursor+n].argmax()); cursor += n
            hits += int(candidate_hits[chosen]); oracle += int(candidate_hits.any())
            regions += int(record["label_area"][chosen].sum() > 0); named += 1
    return dict(n=len(records), named=named, hits=hits, regions=regions, point_oracle=oracle,
                loss=weighted_loss / named if named else None)


def epoch_order(n, count, seed, epoch):
    """Cycle independently shuffled allowed named cases until the exact budget."""
    if n < 1:
        raise ValueError("No named allowed training case")
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, 12]))
    pieces = []; remaining = count
    while remaining:
        order = rng.permutation(n)[:remaining]
        pieces.append(order); remaining -= len(order)
    return np.concatenate(pieces)


def train(args):
    config = CrossfitProposerConfig()
    torch.set_num_threads(args.cpu_threads)
    train_all, val_cases = read_cases(args.train_cases), read_cases(args.val_cases)
    folds = read_json(args.fold_map)
    selected, allowed_ids, heldout_ids = role_cases(train_all, val_cases, folds, args.role)
    sources = {name: sha(name) for name in SOURCE_PATHS}
    protocol = args.output / "protocol.json"
    if not protocol.exists():
        raise ValueError("Initialize the prespecified protocol.json before upstream training")
    if (args.output / "frozen.json").exists() or list((args.output / "test").rglob("*.npz")):
        raise ValueError("Upstream training is forbidden after scorer freeze/fresh-test extraction")
    registered = read_json(protocol)["upstream"]
    expected = dict(initial_seed=config.seed, epochs=config.epochs,
        steps_per_epoch=config.steps_per_epoch, batch_size=config.batch_size,
        learning_rate=config.learning_rate, weight_decay=config.weight_decay)
    if any(registered.get(k) != v for k, v in expected.items()):
        raise ValueError("CLI implementation and prespecified upstream budget disagree")
    identity = dict(config=asdict(config), role=args.role, source_sha256=sources,
        train_cases_sha256=sha(args.train_cases), val_cases_sha256=sha(args.val_cases),
        folds_sha256=sha(args.fold_map), allowed_train_ids=allowed_ids, heldout_ids=heldout_ids,
        validation_ids=sorted(row["sample_id"] for row in val_cases),
        checkpoint_selection="old172 validation hits, then lower aligned loss, earlier epoch; includes epoch0",
        heldout_labels_used=False, old_learned_weights_used=False, teacher_points_used=False,
        head_prior="all-zero constant96", training_device=args.device)
    identity["selection_forward_device"] = args.device
    identity["selection_integer_decode_device"] = "cpu"
    identity["protocol_sha256"] = sha(protocol)
    fingerprint = object_sha(identity)
    folder = args.output / "proposers"; folder.mkdir(parents=True, exist_ok=True)
    checkpoint = folder / f"{args.role}.pt"; history_path = folder / f"{args.role}.json"
    if history_path.exists():
        old = read_json(history_path)
        if old.get("complete") and old.get("fingerprint") == fingerprint and sha(checkpoint) == old["checkpoint_sha256"]:
            for value in old["identity"]["raw_sources"].values():
                if sha(value["path"]) != value["sha256"]:
                    raise ValueError("Completed proposer raw input was changed")
            print(json.dumps(dict(role=args.role, reused_complete=True, selected=old["selected"])), flush=True)
            return
        raise ValueError(f"Existing/incomplete role output must be preserved or explicitly relocated: {history_path}")
    # Only selected training IDs and the independent old validation reach GT builders.
    records = [load_record(row) for row in selected]
    validation = [load_record(row) for row in val_cases]
    raw_identities = {r["case"]["sample_id"]: r["raw_identity"] for r in records+validation}
    identity["raw_sources"] = raw_identities
    identity["input_raw_identity_sha256"] = object_sha(raw_identities)
    identity["allowed_train_ids_sha256"] = object_sha(allowed_ids)
    named = [r for r in records if r["metadata"]["target_available"]]
    if not named:
        raise ValueError("No named training records")
    torch.manual_seed(config.seed); np.random.seed(config.seed); random.seed(config.seed)
    if str(args.device).startswith("cuda"):
        torch.cuda.manual_seed_all(config.seed)
    model = make_fresh_proposer(config).to(args.device)
    initial_sha = model.initial_state_sha256
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.epochs, eta_min=config.learning_rate*.1)
    history = []; best = None; selected_checkpoint = None; started = time.perf_counter()
    for epoch in range(config.epochs+1):
        losses = []; steps = 0
        if epoch:
            model.train()
            order = epoch_order(len(named), config.steps_per_epoch*config.batch_size, config.seed, epoch)
            for start in range(0, len(order), config.batch_size):
                group = [named[int(i)] for i in order[start:start+config.batch_size]]
                inputs, targets = make_field_batch(group, args.device, labels=True)
                optimizer.zero_grad(set_to_none=True); output = model(**inputs)
                point_hit = torch.from_numpy(np.concatenate(candidate_point_hits(all_candidate_points(output, group), group))).to(args.device)
                loss, _ = aligned_loss(output, targets, point_hit, True, region_weight=config.region_weight)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Nonfinite aligned loss")
                loss.backward(); grad = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                if not bool(torch.isfinite(grad)):
                    raise FloatingPointError("Nonfinite proposer gradients")
                optimizer.step(); losses.append(float(loss.detach())); steps += 1
            scheduler.step()
            if steps != config.steps_per_epoch:
                raise AssertionError("Unexpected optimizer step count")
        # GPU evaluation is deterministic within this run; final proposals use
        # predict_raw's canonical CPU integer decoding after the checkpoint seals.
        metrics = evaluate(model, validation, args.device)
        row = dict(epoch=epoch, validation=metrics, train_loss=float(np.mean(losses)) if losses else None,
                   consumed_queries=steps*config.batch_size, optimizer_steps=steps,
                   learning_rate=optimizer.param_groups[0]["lr"], elapsed_seconds=time.perf_counter()-started)
        history.append(row)
        selector = (metrics["hits"], -metrics["loss"])
        if best is None or selector > best:
            best = selector; selected_checkpoint = dict(epoch=epoch, validation=metrics)
            save_crossfit_proposer(checkpoint, model, dict(identity=identity, selected=selected_checkpoint,
                initial_state_sha256=initial_sha, fingerprint=fingerprint), config)
        write_json(history_path, dict(complete=False, fingerprint=fingerprint, identity=identity,
            protocol_sha256=identity["protocol_sha256"], role=args.role,
            allowed_train_ids=allowed_ids, heldout_ids=heldout_ids,
            source_sha256=sources, input_raw_identity_sha256=identity["input_raw_identity_sha256"],
            initial_state_sha256=initial_sha, parameters=sum(p.numel() for p in model.parameters()),
            named_train=len(named), n_train=len(records), n_validation=len(validation),
            history=history, selected=selected_checkpoint, checkpoint_sha256=sha(checkpoint)))
        print(json.dumps(dict(role=args.role, **row), ensure_ascii=False), flush=True)
    if {name: sha(name) for name in SOURCE_PATHS} != sources:
        raise ValueError("Source changed during training")
    final = read_json(history_path); final["complete"] = True
    final["optimizer_steps_total"] = sum(r["optimizer_steps"] for r in history)
    final["consumed_queries_total"] = sum(r["consumed_queries"] for r in history)
    final["final_state_sha256"] = state_sha256(model)
    # Replay the selected state canonically on CPU, without selecting again.
    cpu_model, _ = load_crossfit_proposer(checkpoint, "cpu")
    canonical = evaluate(cpu_model, validation, "cpu")
    selected_metrics = selected_checkpoint["validation"]
    final["selected_cpu_canonical_validation"] = dict(
        metrics=canonical, selection_unchanged=True, selected_epoch=selected_checkpoint["epoch"],
        reference_forward_device=args.device, integer_decode_device="cpu",
        delta_hits=canonical["hits"]-selected_metrics["hits"],
        delta_point_oracle=canonical["point_oracle"]-selected_metrics["point_oracle"],
        note="CPU replay only; device arithmetic can change boundary decisions. No epoch reselection.")
    write_json(history_path, final)
    del model, optimizer, cpu_model; gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path("runs/crossfit_candidates_v12")
    parser.add_argument("--output", type=Path, default=root)
    parser.add_argument("--train-cases", type=Path, default=root/"data/train.jsonl")
    parser.add_argument("--val-cases", type=Path, default=root/"data/val.jsonl")
    parser.add_argument("--fold-map", type=Path, default=root/"folds.json")
    parser.add_argument("--role", choices=["full", *(f"fold_{i}" for i in range(5))], required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args(); train(args)


if __name__ == "__main__":
    main()
