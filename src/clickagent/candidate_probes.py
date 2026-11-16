"""Train only external readout heads; calibrate without looking at test labels."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from .action_benchmark import sha256, stable_hash, write_json
from .candidate_data import SEED
from .public_benchmark import cache_key, read_jsonl


KINDS = ("attention", "hidden", "combined")
BUDGETS = (.01, .05, .10)


def layer_rms(x):
    x = np.asarray(x, np.float32)
    shape = x.shape
    x = x.reshape(*shape[:-1], 3, -1)
    return (x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-6)).reshape(shape)


def load_features(output, *, training_only=False):
    cases = read_jsonl(output / "manifest.jsonl")
    if training_only:
        cases = [r for r in cases if r["split"] in {"train","dev"}]
    fingerprint = stable_hash(json.loads((output / "identity.json").read_text(encoding="utf-8")))
    if len({r["image_sha256"] for r in cases}) != len(cases):
        raise ValueError("screenshot leakage between records")
    learned_apps = {r["application"] for r in cases if r["split"] != "test"}
    test_apps = {r["application"] for r in cases if r["split"] == "test"}
    if learned_apps & test_apps:
        raise ValueError("test applications overlap learned partitions")
    fingerprints = {r["sample_id"]: fingerprint for r in cases}
    folders = {r["sample_id"]: output for r in cases}
    augmentation = output / "augmentation"
    if (augmentation / "manifest.jsonl").exists():
        added = read_jsonl(augmentation / "manifest.jsonl")
        aug_fingerprint = stable_hash(json.loads((augmentation / "identity.json").read_text(encoding="utf-8")))
        base_training_images = {r["image_sha256"] for r in cases if r["split"] == "train"}
        if any(r["split"] != "train" or r["image_sha256"] not in base_training_images for r in added):
            raise ValueError("augmentation must stay inside existing training screenshots")
        cases.extend(added)
        fingerprints.update({r["sample_id"]:aug_fingerprint for r in added})
        folders.update({r["sample_id"]:augmentation for r in added})
    keys = ("query_hidden", "global_attention", "candidate_attention", "candidate_visual", "geometry", "native_probabilities")
    arrays = {k: [] for k in keys}
    metadata = []
    for case in cases:
        path = folders[case["sample_id"]] / "features" / f"{cache_key(case['sample_id'])}.npz"
        with np.load(path, allow_pickle=False) as saved:
            info = json.loads(str(saved["metadata"]))
            if (info["sample_id"] != case["sample_id"] or info["image_sha256"] != case["image_sha256"]
                    or info["fingerprint"] != fingerprints[case["sample_id"]]):
                raise ValueError("feature identity mismatch")
            metadata.append(info)
            for key in keys:
                arrays[key].append(saved[key].astype(np.float32))
    arrays = {k: np.stack(v) for k, v in arrays.items()}
    arrays["query_hidden"] = layer_rms(arrays["query_hidden"])
    arrays["candidate_visual"] = layer_rms(arrays["candidate_visual"])
    arrays["labels"] = np.asarray([r["candidate_labels"] for r in metadata], np.float32)
    arrays["global_labels"] = np.asarray([r["case_kind"] == "click" for r in cases], np.float32)
    # Candidate identifiers allow q's latent choice information to be read out.
    # Their order is random in training, never derived from the correct answer.
    arrays["geometry"] = np.concatenate((arrays["geometry"], np.broadcast_to(np.eye(4, dtype=np.float32), (len(cases), 4, 4))), -1)
    return cases, metadata, arrays


def make_network(kind, candidate, dimensions):
    import torch
    from torch import nn

    class Verifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.use_a, self.use_h = kind != "hidden", kind != "attention"
            size = 0
            if self.use_h:
                self.q = nn.Sequential(nn.Linear(dimensions["query_hidden"], 64), nn.GELU(), nn.LayerNorm(64))
                if candidate:
                    self.v = nn.Sequential(nn.Linear(dimensions["candidate_visual"], 64), nn.GELU(), nn.LayerNorm(64))
                size += 256 if candidate else 64
            if self.use_a:
                self.a = nn.Sequential(nn.Linear(dimensions["global_attention"], 32), nn.GELU())
                size += 32
                if candidate:
                    self.local_a = nn.Sequential(nn.Linear(dimensions["candidate_attention"], 64), nn.GELU(), nn.LayerNorm(64))
                    size += 64
            if candidate:
                size += dimensions["geometry"]
            self.score = nn.Sequential(nn.Linear(size, 64), nn.GELU(), nn.Dropout(.15), nn.Linear(64, 1))
            if candidate:
                global_size = (64 if self.use_h else 0) + (32 if self.use_a else 0)
                self.null = nn.Sequential(nn.Linear(global_size, 32), nn.GELU(), nn.Linear(32, 1))

        def forward(self, data):
            values, global_values = [], []
            if self.use_h:
                q = self.q(data["query_hidden"])
                global_values.append(q)
                if candidate:
                    q = q[:, None].expand(-1, 4, -1)
                    v = self.v(data["candidate_visual"])
                    values.extend([q, v, q * v, torch.abs(q - v)])
                else:
                    values.append(q)
            if self.use_a:
                a = self.a(data["global_attention"])
                global_values.append(a)
                values.append(a[:, None].expand(-1, 4, -1) if candidate else a)
                if candidate:
                    values.append(self.local_a(data["candidate_attention"]))
            if candidate:
                values.append(data["geometry"])
                score = self.score(torch.cat(values, -1)).squeeze(-1)
                null = self.null(torch.cat(global_values, -1))
                return torch.cat((score, null), -1)
            return self.score(torch.cat(values, -1)).squeeze(-1)
    return Verifier()


def set_loss(logits, labels):
    import torch
    correct = torch.cat((labels.bool(), ~labels.bool().any(-1, keepdim=True)), -1)
    log_probability = torch.log_softmax(logits, -1)
    marginal = -torch.logsumexp(log_probability.masked_fill(~correct, -torch.inf), -1).mean()
    # The auxiliary term discourages high scores for multiple incorrect targets.
    return marginal + .25 * torch.nn.functional.binary_cross_entropy_with_logits(logits[:, :4], labels)


def fit_head(kind, candidate, arrays, train, dev, path, *, query_selector=False):
    import torch
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    dimensions = {k: v.shape[-1] for k, v in arrays.items() if v.ndim >= 2}
    model = make_network(kind, candidate, dimensions)
    if query_selector:
        if candidate or kind != "hidden":
            raise ValueError("query-only selector requires the global hidden architecture")
        model.score[-1] = torch.nn.Linear(64,5)
    is_set = candidate or query_selector
    tensors = {k: torch.from_numpy(v) for k, v in arrays.items() if k != "native_probabilities"}
    # Estimate attention scales on train only; hidden vectors use fixed RMS.
    scaling = {}
    for key in ("global_attention", "candidate_attention"):
        selected = arrays[key][train]
        axes = tuple(range(selected.ndim - 1))
        mean = selected.mean(axis=axes)
        scale = np.maximum(selected.std(axis=axes), .05)
        scaling[key] = (mean, scale)
        tensors[key] = torch.from_numpy(np.clip((arrays[key] - mean) / scale, -15, 15))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.02)
    best, best_epoch, stale = float("inf"), 0, 0
    history = []
    started = time.perf_counter()
    for epoch in range(1, 121):
        model.train()
        for ids in np.array_split(rng.permutation(train), max(1, int(np.ceil(len(train) / 32)))):
            batch = {k: v[ids] for k, v in tensors.items()}
            logits = model(batch)
            loss = set_loss(logits, batch["labels"]) if is_set else torch.nn.functional.binary_cross_entropy_with_logits(logits, batch["global_labels"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            batch = {k: v[dev] for k, v in tensors.items()}
            logits = model(batch)
            dev_loss = float(set_loss(logits, batch["labels"]) if is_set else torch.nn.functional.binary_cross_entropy_with_logits(logits, batch["global_labels"]))
        history.append(dev_loss)
        if dev_loss < best - 1e-5:
            best, best_epoch, stale = dev_loss, epoch, 0
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= 15:
            break
    model.load_state_dict(state)
    model.eval()
    scores = []
    with torch.no_grad():
        for start in range(0, len(arrays["labels"]), 64):
            ids = slice(start, start + 64)
            logits = model({k: v[ids] for k, v in tensors.items()})
            p = torch.softmax(logits, -1) if is_set else torch.sigmoid(logits)[:, None].expand(-1, 4)
            scores.append(p.cpu().numpy())
    checkpoint = {"state_dict": state, "scaling": scaling, "kind": kind, "candidate": candidate,
                  "query_selector":query_selector,
                  "dimensions": dimensions, "seed": SEED, "best_epoch": best_epoch,
                  "dev_loss": best, "train_indices": train.tolist(), "dev_indices": dev.tolist()}
    torch.save(checkpoint, path)
    info = {"best_epoch": best_epoch, "epochs": epoch, "dev_loss": best,
            "parameters": sum(p.numel() for p in model.parameters()),
            "training_seconds": time.perf_counter() - started, "dev_history": history}
    return np.concatenate(scores), info


def predict_checkpoint(path, arrays):
    """CPU inference from a locally trained checkpoint; no label inputs."""
    import torch
    checkpoint = torch.load(path,map_location="cpu",weights_only=False)
    model = make_network(checkpoint["kind"],checkpoint["candidate"],checkpoint["dimensions"])
    if checkpoint.get("query_selector"):
        model.score[-1] = torch.nn.Linear(64,5)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval().requires_grad_(False)
    tensors = {key:torch.from_numpy(arrays[key]) for key in
        ("query_hidden","candidate_visual","global_attention","candidate_attention","geometry")}
    for key,(mean,scale) in checkpoint["scaling"].items():
        tensors[key] = torch.from_numpy(np.clip((arrays[key]-mean)/scale,-15,15))
    scores = []
    is_set = checkpoint["candidate"] or checkpoint.get("query_selector")
    with torch.inference_mode():
        for start in range(0,len(arrays["query_hidden"]),64):
            logits = model({key:value[start:start+64] for key,value in tensors.items()})
            p = torch.softmax(logits,-1) if is_set else torch.sigmoid(logits)[:,None].expand(-1,4)
            scores.append(p.numpy())
    return np.concatenate(scores)


def calibrated_threshold(scores, correct, budget, eligible=None):
    """Maximize executed coverage subject to empirical mistakes / all requests.

    Only calibration records may be supplied by the caller. This is an empirical
    operating point, not a finite-sample guarantee or conformal risk bound.
    Atomic ties are never split; >max is an explicit valid reject-all choice.
    """
    scores, correct = np.asarray(scores), np.asarray(correct, bool)
    eligible = np.ones(len(scores), bool) if eligible is None else np.asarray(eligible, bool)
    if not len(scores) or not np.isfinite(scores).all():
        raise ValueError("nonempty finite calibration scores required")
    best = float(np.nextafter(float(scores.max()), np.inf))
    for threshold in np.unique(scores)[::-1]:
        accept = (scores >= threshold) & eligible
        if np.sum(accept & ~correct) / len(scores) <= budget + 1e-12:
            best = float(threshold)
    return best


def policy_metrics(cases, labels, chosen, scores, threshold, eligible=None):
    eligible = np.ones(len(scores), bool) if eligible is None else np.asarray(eligible, bool)
    accept = (np.asarray(scores, np.float64) >= np.float64(threshold)) & eligible
    correct = labels[np.arange(len(labels)), chosen].astype(bool)
    wants_click = np.asarray([r["case_kind"] == "click" for r in cases])
    good, bad = accept & correct, accept & ~correct
    return {"n": len(cases), "executed": int(accept.sum()), "correct_clicks": int(good.sum()),
        "wrong_clicks": int(bad.sum()), "correct_clicks_per_request": float(good.mean()),
        "wrong_clicks_per_request": float(bad.mean()), "execution_precision": float(good.sum() / accept.sum()) if accept.any() else None,
        "correct_candidate_count":int(correct.sum()),
        "correct_candidate_acceptance":float(good.sum()/correct.sum()) if correct.any() else None,
        "bad_candidate_count":int((~correct).sum()),
        "bad_candidate_false_accept_rate":float(bad.sum()/(~correct).sum()) if (~correct).any() else None,
        "click_request_count": int(wants_click.sum()), "click_request_success": float(good.sum() / wants_click.sum()) if wants_click.any() else None,
        "nonactionable_count": int((~wants_click).sum()), "nonactionable_false_clicks": int((accept & ~wants_click).sum()),
        "nonactionable_false_click_rate": float((accept & ~wants_click).sum() / (~wants_click).sum()) if (~wants_click).any() else None,
        "wrong_location_clicks": int((bad & wants_click).sum()), "threshold": float(threshold),
        "null_vetoed": int((~eligible).sum()),
        "by_kind": {kind: {"n": sum(r["case_kind"] == kind for r in cases),
                          "executed": sum(bool(accept[i]) for i,r in enumerate(cases) if r["case_kind"] == kind),
                          "correct": sum(bool(good[i]) for i,r in enumerate(cases) if r["case_kind"] == kind)}
                    for kind in ("click", "forbid", "describe", "refusal")}}


def _auc(y, p):
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None


def evaluation_test_indices(cases,output):
    path = output / "evaluation_exclusions.json"
    exclusions = set(json.loads(path.read_text(encoding="utf-8"))["sample_ids"]) if path.exists() else set()
    test_ids = {r["sample_id"] for r in cases if r["split"]=="test"}
    if not exclusions <= test_ids:
        raise ValueError("evaluation exclusion contains a non-test ID")
    return np.asarray([i for i,r in enumerate(cases) if r["split"]=="test" and r["sample_id"] not in exclusions]), sorted(exclusions)


def evaluate(cases, metadata, arrays, scores, output, training):
    calibration = np.asarray([i for i,r in enumerate(cases) if r["split"] == "calibration"])
    test, exclusions = evaluation_test_indices(cases,output)
    labels = arrays["labels"].astype(bool)
    test_cases = [cases[i] for i in test]
    results, policies = {}, []
    for method, probability in scores.items():
        p = probability[:, :4]
        click_ids = np.asarray([i for i in test if cases[i]["case_kind"] == "click"])
        eligible = [i for i in click_ids if labels[i].any() and not labels[i].all()]
        pair_wins = []
        for i in eligible:
            difference = p[i, labels[i]][:, None] - p[i, ~labels[i]][None, :]
            pair_wins.extend(((difference > 0) + .5 * (difference == 0)).ravel().tolist())
        entry = {"candidate_auroc_all": _auc(labels[test].ravel(), p[test].ravel()),
            "candidate_auroc_click_requests_only": _auc(labels[click_ids].ravel(), p[click_ids].ravel()),
            "within_screenshot_pair_ordering": float(np.mean(pair_wins)) if pair_wins else None,
            "within_screenshot_pair_count": len(pair_wins), "policies": {}}
        for policy in ("verify_first", "select_four"):
            chosen = np.zeros(len(cases), int) if policy == "verify_first" else p.argmax(-1)
            chosen_scores = p[np.arange(len(cases)), chosen]
            correctness = labels[np.arange(len(cases)), chosen]
            # A learned/native explicit N is a veto, never an invitation to
            # execute a low-scoring candidate via a more permissive threshold.
            eligible_for_execution = (chosen_scores > probability[:, 4]) if probability.shape[1] == 5 else np.ones(len(cases), bool)
            budgets = {}
            for budget in BUDGETS:
                threshold = calibrated_threshold(chosen_scores[calibration].astype(np.float64), correctness[calibration], budget, eligible_for_execution[calibration])
                cal_metrics = policy_metrics([cases[i] for i in calibration], labels[calibration], chosen[calibration], chosen_scores[calibration], threshold, eligible_for_execution[calibration])
                metrics = policy_metrics(test_cases, labels[test], chosen[test], chosen_scores[test], threshold, eligible_for_execution[test])
                budgets[str(budget)] = {"calibration": cal_metrics, "test": metrics}
                for i in test:
                    policies.append({"sample_id": cases[i]["sample_id"], "method": method, "policy": policy,
                        "calibration_error_budget": budget, "threshold": threshold, "candidate_index": int(chosen[i]),
                        "point": metadata[i]["points"][chosen[i]], "score": float(chosen_scores[i]),
                        "null_veto": bool(not eligible_for_execution[i]),
                        "execute": bool(eligible_for_execution[i] and np.float64(chosen_scores[i]) >= np.float64(threshold)),
                        "correct_candidate": bool(correctness[i])})
            entry["policies"][policy] = budgets
        results[method] = entry
    native = arrays["native_probabilities"][test]
    native_choice = native.argmax(-1)
    native_execute = native_choice != 4
    native_correct = labels[test][np.arange(len(test)), np.minimum(native_choice, 3)]
    coverage = labels[test].any(-1)
    click_mask = np.asarray([r["case_kind"] == "click" for r in test_cases])
    summary = {"complete": True, "n_total": len(cases), "n_test": len(test),
        "excluded_test_sample_ids":exclusions,"n_test_collected":sum(r["split"]=="test" for r in cases),
        "split_counts": {s:sum(r["split"]==s for r in cases) for s in ("train","dev","calibration","test")},
        "test_click_requests": int(click_mask.sum()), "test_candidates": int(len(test) * 4),
        "test_positive_candidates": int(labels[test].sum()),
        "model_candidate_coverage_click_requests": float(coverage[click_mask].mean()),
        "native_argmax": {"executed": int(native_execute.sum()),
            "correct_clicks": int((native_execute & native_correct).sum()),
            "wrong_clicks": int((native_execute & ~native_correct).sum())},
        "first_candidate_without_gate": {"correct_clicks": int(labels[test, 0].sum()),
            "wrong_clicks": int((~labels[test, 0]).sum())},
        "training": training, "methods": results,
        "sources": {"manifest_sha256": sha256(output / "manifest.jsonl"), "identity_sha256": sha256(output / "identity.json"),
            "training_recipe_sha256":sha256(output / "training_source.py"),
            "evaluation_source_sha256":sha256(Path(__file__)), "augmentation_manifest_sha256":sha256(output / "augmentation/manifest.jsonl")},
        "protocol": "internal repartitioned public-data pilot; frozen backbone; no test threshold/hyperparameter selection",
        "risk_budget_definition": "wrong executed clicks / all calibration requests; includes incorrect locations and forbidden/description/refusal clicks",
        "limitations": ["not official benchmark scores or live application success", f"primary test has {len(test)} screenshots; risk estimates have sampling uncertainty",
            "train/dev annotation candidates differ from model-generated calibration/test candidates",
            "no dedicated disabled-control labels", "description/forbid variants are templated",
            "attention/hidden features share candidate-marked frozen-VLM prefill; no claim of independent latency advantage"]}
    write_json(output / "results.json", summary)
    write_json(output / "calibration_thresholds.json", {m: r["policies"] for m,r in results.items()})
    (output / "predictions.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in policies), encoding="utf-8")
    lines = ["# Candidate-conditioned verifier pilot", "", "Frozen Qwen3-VL-8B; internal public-data repartition, not official benchmark scores.",
        "", f"Test: {len(test)} screenshots, {int(click_mask.sum())} click requests, {len(test)*4} candidate judgments.",
        f"Model candidate coverage on click requests: {coverage[click_mask].mean():.1%}.", "",
        "Thresholds are chosen only on 80 calibration cases. The 5% budget refers to wrong clicks / all requests, not conditional execution error.", "",
        "| Method | Candidate AUROC | Within-image ordering | Verify first: correct/wrong | Select four: correct/wrong |",
        "|---|---:|---:|---:|---:|"]
    for m,r in results.items():
        first = r["policies"]["verify_first"]["0.05"]["test"]
        selected = r["policies"]["select_four"]["0.05"]["test"]
        fmt = lambda v: "N/A" if v is None else f"{v:.3f}"
        lines.append(f"| {m} | {fmt(r['candidate_auroc_click_requests_only'])} | {fmt(r['within_screenshot_pair_ordering'])} | {first['correct_clicks']}/{first['wrong_clicks']} | {selected['correct_clicks']}/{selected['wrong_clicks']} |")
    lines += ["", "## Limitations", ""] + ["- " + s for s in summary["limitations"]]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/candidate_v2"))
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--train-only",action="store_true")
    modes.add_argument("--evaluate-only",action="store_true")
    args = parser.parse_args(argv)
    import torch
    torch.set_num_threads(4)
    if not args.evaluate_only:
        (args.output / "training_source.py").write_bytes(Path(__file__).read_bytes())
    cases, metadata, arrays = load_features(args.output,training_only=args.train_only)
    train = np.asarray([i for i,r in enumerate(cases) if r["split"] == "train"])
    dev = np.asarray([i for i,r in enumerate(cases) if r["split"] == "dev"])
    scores, training = {"native_choice_logits": arrays["native_probabilities"]}, {}
    (args.output / "models").mkdir(exist_ok=True)
    if args.evaluate_only:
        training = json.loads((args.output / "training.json").read_text(encoding="utf-8"))
        for method in training:
            scores[method] = predict_checkpoint(args.output / "models" / f"{method}.pt",arrays)
        np.savez_compressed(args.output / "scores.npz",**scores)
        evaluate(cases,metadata,arrays,scores,args.output,training)
        print("Frozen-head evaluation complete",flush=True)
        return
    for candidate in (False, True):
        for kind in KINDS:
            method = ("candidate_" if candidate else "global_") + kind
            scores[method], training[method] = fit_head(kind, candidate, arrays, train, dev, args.output / "models" / f"{method}.pt")
            print(method, {k:v for k,v in training[method].items() if k != "dev_history"}, flush=True)
    method = "query_only_selector"
    scores[method], training[method] = fit_head("hidden",False,arrays,train,dev,
        args.output / "models" / f"{method}.pt",query_selector=True)
    print(method,{k:v for k,v in training[method].items() if k!="dev_history"},flush=True)
    write_json(args.output / "training.json",training)
    write_json(args.output / "training_records.json",{"sample_ids":[r["sample_id"] for r in cases if r["split"] in {"train","dev"}],
        "train_count":len(train),"dev_count":len(dev),"training_only_loader":args.train_only,
        "training_source_sha256":sha256(args.output / "training_source.py")})
    if args.train_only:
        np.savez_compressed(args.output / "training_scores.npz",**scores)
        print("Training complete; no calibration/test feature or label loaded",flush=True)
        return
    np.savez_compressed(args.output / "scores.npz", **scores)
    summary = evaluate(cases, metadata, arrays, scores, args.output, training)
    print(json.dumps({"complete": summary["complete"], "n_test": summary["n_test"]}), flush=True)


if __name__ == "__main__":
    main()
