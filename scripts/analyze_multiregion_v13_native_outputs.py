"""Explicitly POSTHOC native-WAIT/format diagnostic, never a new main score.

Normal execution requires a successful V13 final audit. Only this new JSON is
written; no prediction, model, selected epoch or original evaluation is changed.
``--self-test`` uses only in-memory synthetic records and reads no run artifacts.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

from clickagent.public_evaluation import point_in_geometry


WORKSPACE = Path(__file__).resolve().parents[1]
ROOT = WORKSPACE / "runs/multiregion_v13"
METHODS = ("c3_top1_s1_loso", "c3_top1_s2_loso")
NATIVE_METHOD = "qwen_native_completed_lax"
NATIVE_TOKEN_LIMIT = 32
STRICT_PATTERN = r"\s*(CLICK[ \t]+([0-9]+)[ \t]+([0-9]+)|WAIT|UNKNOWN)\s*"
LAX_PATTERN = r"\A[ \t\r\n]*CLICK[ \t]+([0-9]+)(?:[ \t]*,[ \t]*|[ \t]+)([0-9]+)[ \t\r\n]*\Z"
STRICT = re.compile(STRICT_PATTERN)
LAX = re.compile(LAX_PATTERN)
LIMITATIONS = [
    "明确的事后诊断：以native自身是否输出有效坐标划分子集，具有输出依赖选择偏差；子集不是新的独立测试集。",
    "原192题全分母completed-lax结果保持权威；有坐标子集与真实WAIT组只解释差异，不替换主分母、不计算新CI。",
    "native提示词允许不可见、禁用、歧义或无法确定时输出WAIT；内部评分器在非空候选池强制选点，许可/拒答约定不同。",
    "内部流程使用额外crop及经过监督训练的proposer/scorer；native是原图直接文本坐标对照，输入计算和监督预算不同。",
    "字面文本WAIT与因CLICK逗号格式导致严格解析回退到wait分别统计，不能把二者统称语义拒答。",
    "completed-lax已在正式主评分中接纳合法逗号CLICK；此处重算格式恢复，不会再次给主评分增加命中。",
    "坐标按round(value*(size-1)/1000)复核；几何判分仍按像素坐标除以原图width/height及既有区域容差。",
    "本脚本不执行模型、训练、选模或阈值调优，不按子集或测试结果选择种子；S1/S2固定分别报告。",
]


def require(value, message):
    if not value:
        raise ValueError(message)


def resolve(path):
    p = Path(path)
    return (p if p.is_absolute() else WORKSPACE / p).resolve()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with resolve(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class InputSnapshots:
    """Every consumed input is fully hashed before/read and again at commit."""
    def __init__(self):
        self.expected = {}; self.files = {}

    def set_audit_inventory(self, inventory):
        for name, value in inventory.items():
            key = str(resolve(name)).casefold()
            require(key not in self.expected or self.expected[key] == value, "Conflicting prior-audit SHA")
            self.expected[key] = value

    def bytes(self, path, role, *, allow_unlisted=False):
        p = resolve(path); name = str(p); prior = self.expected.get(name.casefold())
        require(prior is not None or allow_unlisted, "Consumed input missing from successful audit: " + name)
        before = file_sha(p); data = p.read_bytes()
        require(digest(data) == before, "Input changed between SHA and read: " + name)
        require(prior is None or prior == before, "Input differs from successful audit: " + name)
        if name in self.files:
            require(self.files[name]["sha256_before"] == before, "Repeated input changed: " + name)
        else:
            scope = ("verified_against_prior_final_audit" if prior is not None else
                     "supplemental_native_not_in_prior_audit" if role == "native_output" else
                     "new_posthoc_diagnostic_source" if role == "diagnostic_source" else "successful_audit_gate_file")
            self.files[name] = dict(role=role, sha256_before=before, sha256_after=None,
                                    prior_audit_sha256=prior, prior_audit_scope=scope, bytes=len(data))
        return data

    def json(self, path, role, **kwargs):
        return json.loads(self.bytes(path, role, **kwargs).decode("utf-8-sig"))

    def finish(self):
        for name, record in self.files.items():
            after = file_sha(name)
            require(after == record["sha256_before"], "Consumed input changed before commit: " + name)
            record["sha256_after"] = after


def strict_parse(text):
    """Independent replay of the frozen strict whole-action parser."""
    match = STRICT.fullmatch(text) if isinstance(text, str) else None
    if match is None:
        return None
    word = match.group(1)
    if word in {"WAIT", "UNKNOWN"}:
        return dict(action="wait", normalized_point=None, generated_decision=word.lower(),
                    abstained=word == "UNKNOWN", valid=True)
    point = [int(match.group(2)), int(match.group(3))]
    if not all(0 <= value <= 1000 for value in point):
        return None
    return dict(action="click", normalized_point=point, generated_decision="click", abstained=False, valid=True)


def lax_parse(text):
    match = LAX.fullmatch(text) if isinstance(text, str) else None
    if match is None:
        return None
    values = [int(v) for v in match.groups()]
    return values if all(0 <= value <= 1000 for value in values) else None


def pixel_point(normalized, size):
    if normalized is None:
        return None
    require(len(size) == 2 and all(type(v) is int and v > 0 for v in size), "Invalid image size")
    require(len(normalized) == 2 and all(type(v) is int and 0 <= v <= 1000 for v in normalized),
            "Normalized CLICK coordinate outside integer0..1000")
    return [int(round(value * (length - 1) / 1000)) for value, length in zip(normalized, size)]


def valid_pixel(point, size):
    return point is None or (isinstance(point, list) and len(point) == 2 and
        all(type(v) is int and 0 <= v < limit for v, limit in zip(point, size)))


def hit(point, case):
    require(valid_pixel(point, case["image_size"]), "Invalid pixel coordinate")
    return bool(point is not None and point_in_geometry(
        [v / size for v, size in zip(point, case["image_size"])], case["target_geometry"]))


def paired(a, b):
    n = len(a); require(n == len(b), "Unaligned paired records")
    return dict(n=n, internal_hits=sum(a), native_hits=sum(b),
        wins=sum(x and not y for x, y in zip(a, b)), losses=sum(y and not x for x, y in zip(a, b)),
        both_hit=sum(x and y for x, y in zip(a, b)), both_miss=sum(not (x or y) for x, y in zip(a, b)),
        internal_accuracy=sum(a) / n if n else None, native_accuracy=sum(b) / n if n else None,
        delta=(sum(a) - sum(b)) / n if n else None, new_ci_computed=False)


def protocol_from_sources(public_source, action_source, lax_source, extractor_source):
    """Inspect audited source AST; never import a VLM/torch runtime module."""
    prompt = None
    for node in ast.parse(public_source).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "PUBLIC_TEXT_ACTION_SYSTEM_PROMPT" for t in node.targets):
            prompt = ast.literal_eval(node.value)
    require(isinstance(prompt, str) and "output WAIT" in prompt, "Native prompt no longer declares WAIT")
    strict_patterns = [ast.literal_eval(node.args[0]) for node in ast.walk(ast.parse(action_source))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "fullmatch"
        and node.args and isinstance(node.args[0], ast.Constant)]
    require(STRICT_PATTERN in strict_patterns, "Independent strict parser differs from audited source")
    lax_patterns = [ast.literal_eval(node.value.args[0]) for node in ast.parse(lax_source).body
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "LAX_CLICK" for t in node.targets)
        and isinstance(node.value, ast.Call)]
    require(lax_patterns == [LAX_PATTERN], "Independent lax parser differs from audited source")
    native_functions = [n for n in ast.parse(extractor_source).body if isinstance(n, ast.FunctionDef) and n.name == "native"]
    require(len(native_functions) == 1, "Missing audited native extractor")
    limits = [ast.literal_eval(kw.value) for call in ast.walk(native_functions[0]) if isinstance(call, ast.Call)
              and isinstance(call.func, ast.Attribute) and call.func.attr == "infer_text_action_public"
              for kw in call.keywords if kw.arg == "max_new_tokens"]
    require(limits == [NATIVE_TOKEN_LIMIT], "Native token-limit protocol changed")
    return dict(system_prompt=prompt, system_prompt_sha256=digest(prompt.encode()), permits_wait=True,
        token_limit=NATIVE_TOKEN_LIMIT, strict_and_lax_regex_match_audited_source=True,
        pixel_mapping="round(value*(original_image_dimension-1)/1000)", geometry_normalization="pixel/original_image_dimension")


def inspect_native(case, native, expected_prompt):
    issues = []; sid = case["sample_id"]
    def check(condition, field):
        if not condition:
            issues.append(dict(sample_id=sid, field=field))
    text = native.get("generated_text")
    require(isinstance(text, str), "Native generated_text must be a string")
    eos = native.get("terminated_by_eos") is True
    newline = text.endswith(("\n", "\r")); completed = eos or newline
    strict = strict_parse(text) if completed else None
    normalized = lax_parse(text); expected_lax = pixel_point(normalized, case["image_size"]) if completed else None
    generated_ids = native.get("generated_token_ids", [])
    require(isinstance(generated_ids, list), "Native generated_token_ids must be a list")
    expected_truncated = strict is None and not eos and len(generated_ids) == NATIVE_TOKEN_LIMIT
    expected_action = strict or dict(action="wait", normalized_point=None, generated_decision=None, abstained=True, valid=False)
    expected_strict_point = pixel_point(expected_action["normalized_point"], case["image_size"])
    check(native.get("sample_id") == sid, "sample_id")
    check(native.get("image_size") == case["image_size"], "image_size")
    check(native.get("image_sha256") == case["image_sha256"], "image_sha256")
    check(native.get("instruction_sha256") == digest(case["instruction"].encode()), "instruction_sha256")
    check(native.get("system_prompt") == expected_prompt, "system_prompt")
    check(type(native.get("terminated_by_eos")) is bool, "terminated_by_eos_boolean")
    check(native.get("generated_tokens") == len(generated_ids), "generated_token_count")
    check(len(generated_ids) <= NATIVE_TOKEN_LIMIT and all(type(v) is int and v >= 0 for v in generated_ids), "generated_token_range")
    check(native.get("truncated") is expected_truncated, "truncated_replay")
    check(native.get("parse_valid") is (strict is not None), "strict_parse_valid_replay")
    for field, value in expected_action.items():
        check(native.get(field) == value, "strict_" + field)
    check(native.get("point") == expected_strict_point, "strict_pixel_mapping")
    check(native.get("completed_lax_point") == expected_lax, "completed_lax_pixel_mapping")
    check(valid_pixel(native.get("point"), case["image_size"]), "strict_pixel_range")
    check(valid_pixel(native.get("completed_lax_point"), case["image_size"]), "completed_lax_pixel_range")
    check((native.get("parse_failure") is None) == (strict is not None), "parse_failure_presence")
    expected_at_limit = strict_parse(text) if expected_truncated else None
    check(native.get("unexecuted_candidate_at_token_limit") == expected_at_limit, "token_limit_candidate_replay")
    check(native.get("ground_truth_in_model_inputs") is False, "native_gt_free")
    check(native.get("hooks_used") is False and native.get("shared_trajectory") is False, "native_independent_no_hooks")
    literal_wait = text.strip() == "WAIT" and strict is not None
    normal_wait = literal_wait and eos and not native.get("truncated") and native.get("parse_valid") is True and native.get("parse_failure") is None
    comma = normalized is not None and "," in text
    strict_wait_click_text = native.get("action") == "wait" and text.lstrip().startswith("CLICK")
    shape = LAX.fullmatch(text)
    range_status = ("integer_pair_in_0_1000" if normalized is not None else
                    "integer_pair_out_of_range" if shape is not None else "not_click_integer_pair")
    category = ("literal_wait" if literal_wait else "literal_unknown" if text.strip() == "UNKNOWN" and strict is not None else
                "strict_click" if expected_lax is not None and strict is not None else
                "comma_click_recovered_by_completed_lax" if expected_lax is not None and comma else
                "other_completed_lax_click" if expected_lax is not None else
                "out_of_range_click" if range_status == "integer_pair_out_of_range" else
                "incomplete_click" if normalized is not None and not completed else
                "truncated_other" if native.get("truncated") else "malformed_or_other")
    return dict(sample_id=sid, generated_text=text, category=category,
        recorded_action=native.get("action"), recorded_parse_valid=native.get("parse_valid"),
        recorded_parse_failure=native.get("parse_failure"), terminated_by_eos=eos, ends_with_newline=newline,
        completed_for_lax=completed, truncated=native.get("truncated"), generated_tokens=len(generated_ids),
        coordinate_range_status=range_status, parsed_normalized_click=normalized,
        strict_pixel_point=native.get("point"), completed_lax_point=native.get("completed_lax_point"),
        replayed_strict_point=expected_strict_point, replayed_completed_lax_point=expected_lax,
        literal_wait=literal_wait, literal_wait_normal_eos=normal_wait,
        strict_recorded_wait_but_click_text=strict_wait_click_text,
        comma_click_recovered=category == "comma_click_recovered_by_completed_lax", checks_passed=not issues), issues


def analyze_records(cases, native_records, head_records, details, summary, expected_prompt):
    ids = [c["sample_id"] for c in cases]
    require(len(ids) == len(set(ids)) and ids, "Repeated/empty primary cases")
    native_map = {r["sample_id"]: r for r in native_records}; heads = {r["sample_id"]: r for r in head_records}
    detail_map = {r["sample_id"]: r for r in details}
    require(len(native_map) == len(native_records) and len(heads) == len(head_records) and
            len(detail_map) == len(details) and set(native_map) == set(heads) == set(detail_map) == set(ids), "Input case coverage mismatch")
    require(summary.get("complete") is True and summary.get("n") == len(cases), "Incomplete official full-denominator summary")
    records = []; violations = []
    for case in cases:
        sid = case["sample_id"]; native = native_map[sid]
        row, issues = inspect_native(case, native, expected_prompt); violations.extend(issues)
        predictions = head_records_map(heads[sid])
        require((NATIVE_METHOD, None) in predictions, "Missing official native head record")
        require(predictions[(NATIVE_METHOD, None)]["point"] == native["completed_lax_point"], "Official native point differs from raw record")
        row["native_completed_lax_hit"] = hit(native["completed_lax_point"], case)
        row["native_strict_runtime_hit"] = hit(native["point"], case)
        row["internal"] = {}
        for method in METHODS:
            require((method, 16) in predictions, "Missing registered internal method")
            point = predictions[(method, 16)]["point"]; correct = hit(point, case)
            row["internal"][method] = dict(point=point, hit=correct)
            scored = detail_map[sid]["methods"][method + "__K16"]
            require(scored["point"] == point and scored["hit"] is correct, "Independent internal point scoring differs from official details")
        scored_native = detail_map[sid]["methods"][NATIVE_METHOD]
        require(scored_native["point"] == native["completed_lax_point"] and
                scored_native["hit"] is row["native_completed_lax_hit"], "Independent native scoring differs from official details")
        records.append(row)
    group_specs = {
        "original_all_cases": lambda row: True,
        "posthoc_completed_lax_coordinate_available": lambda row: row["completed_lax_point"] is not None,
        "posthoc_literal_wait": lambda row: row["literal_wait"],
        "posthoc_literal_wait_normal_eos": lambda row: row["literal_wait_normal_eos"],
        "posthoc_strict_wait_but_click_text": lambda row: row["strict_recorded_wait_but_click_text"],
        "posthoc_comma_click_recovered": lambda row: row["comma_click_recovered"],
    }
    groups = {}
    for name, predicate in group_specs.items():
        selected = [r for r in records if predicate(r)]; n = len(selected)
        groups[name] = dict(n=n, sample_ids=[r["sample_id"] for r in selected],
            native_completed_lax_hits=sum(r["native_completed_lax_hit"] for r in selected),
            native_strict_runtime_hits=sum(r["native_strict_runtime_hit"] for r in selected),
            paired_internal_vs_native={method: paired([r["internal"][method]["hit"] for r in selected],
                [r["native_completed_lax_hit"] for r in selected]) for method in METHODS}, new_ci_computed=False,
            interpretation="Original complete denominator" if name == "original_all_cases" else
                           "Posthoc output-dependent subset only; never replaces original denominator")
    full = groups["original_all_cases"]
    require(summary["methods"][NATIVE_METHOD]["hits"] == full["native_completed_lax_hits"], "Native all-case score differs from official summary")
    for method in METHODS:
        require(summary["methods"][method + "__K16"]["hits"] == full["paired_internal_vs_native"][method]["internal_hits"],
                "Internal all-case score differs from official summary")
    counts = dict(categories=dict(Counter(r["category"] for r in records)),
        recorded_actions=dict(Counter(str(r["recorded_action"]) for r in records)),
        coordinate_range_status=dict(Counter(r["coordinate_range_status"] for r in records)),
        terminated_by_eos=sum(r["terminated_by_eos"] for r in records),
        ends_with_newline=sum(r["ends_with_newline"] for r in records),
        truncated=sum(r["truncated"] is True for r in records),
        strict_parse_valid=sum(r["recorded_parse_valid"] is True for r in records),
        strict_parse_failure=sum(r["recorded_parse_failure"] is not None for r in records),
        completed_lax_coordinate_available=sum(r["completed_lax_point"] is not None for r in records))
    return dict(complete=not violations, explicitly_posthoc=True, primary_cohort="test", primary_denominator=len(cases),
        limitations=LIMITATIONS, full_denominator_original_results=full, posthoc_groups=groups,
        counts=counts, per_case_checks=dict(records_checked=len(records), all_passed=not violations,
                                          violations=violations), per_case=records,
        official_scores_changed=False, new_ci_computed=False, model_execution=False,
        model_training=False, model_selection=False, gpu_used=False)


def head_records_map(head):
    result = {}
    for row in head["predictions"]:
        token = (row["method"], row.get("budget"))
        require(token not in result, "Repeated head prediction")
        result[token] = row
    return result


def run(root):
    root = Path(root).resolve(); snapshots = InputSnapshots()
    audit_path = root / "final_audit.json"
    require(audit_path.is_file(), "Successful final_audit.json required; do not poll/wait from this script")
    audit = snapshots.json(audit_path, "final_audit_gate", allow_unlisted=True)
    require(audit.get("complete") is True and audit.get("version") == 13, "V13 final audit has not passed")
    snapshots.set_audit_inventory(audit["input_sha256"])
    source_bytes = snapshots.bytes(__file__, "diagnostic_source", allow_unlisted=True)
    frozen_path = root / "frozen.json"; frozen = snapshots.json(frozen_path, "pretest_freeze")
    require(frozen.get("complete") is True and frozen.get("fresh_test_features_at_freeze") == 0 and
            file_sha(frozen_path) == audit["frozen_sha256"], "Successful audit does not bind this pretest freeze")
    sources = {name: snapshots.bytes("src/clickagent/" + name + ".py", "audited_parser_prompt_source").decode("utf-8-sig")
               for name in ("public_model", "action_model", "public_diagnostics")}
    snapshots.bytes("src/clickagent/public_evaluation.py", "audited_geometry_source")
    extractor = snapshots.bytes("scripts/extract_multiregion_v13.py", "audited_native_extractor_source").decode("utf-8-sig")
    prompt_protocol = protocol_from_sources(sources["public_model"], sources["action_model"], sources["public_diagnostics"], extractor)
    case_path = root / "data/test.jsonl"
    cases = [json.loads(line) for line in snapshots.bytes(case_path, "primary_case_manifest").decode("utf-8-sig").splitlines() if line.strip()]
    require(len(cases) == 192, "Formal native diagnostic requires original complete192 primary cohort")
    summary_path = root / "evaluation/test/summary.json"
    summary = snapshots.json(summary_path, "official_primary_summary")
    require(audit["evaluation"]["test"]["summary_sha256"] == file_sha(summary_path), "Summary differs from successful audit")
    details = snapshots.json(root / "evaluation/test/details.json", "official_primary_details")
    selection = snapshots.json(root / "models/selection.json", "fixed_pretest_selection")
    require(selection.get("complete") is True and selection.get("fresh_test_read") is False and
            selection.get("seed_selected_by_test") is False, "Missing fixed pretest selection")
    selected_models = {name: selection["selections"][name] for name in METHODS}
    require(all(row["condition"] == "c3" and row["objective"] == "top1" and row["policy"] == "loso"
                for row in selected_models.values()), "Unexpected internal comparison methods")
    protocol = snapshots.json(root / "protocol.json", "fixed_internal_protocol")
    native_records = []; head_records = []
    for case in cases:
        stem = hashlib.sha256(case["sample_id"].encode()).hexdigest()[:24]
        native_records.append(snapshots.json(root / "native/test" / (stem + ".json"), "native_output", allow_unlisted=True))
        head_records.append(snapshots.json(root / "predictions/test" / (stem + ".json"), "audited_internal_and_native_prediction"))
    result = analyze_records(cases, native_records, head_records, details, summary, prompt_protocol["system_prompt"])
    snapshots.finish()
    supplemental = [name for name, row in snapshots.files.items() if row["prior_audit_scope"] == "supplemental_native_not_in_prior_audit"]
    result.update(successful_final_audit_sha256=snapshots.files[str(audit_path)]["sha256_before"],
        successful_audit_reference=dict(path=str(audit_path), complete=True, frozen_sha256=audit["frozen_sha256"]),
        diagnostic_source_sha256=digest(source_bytes), native_prompt_and_parser_protocol=prompt_protocol,
        fixed_internal_selected_models=selected_models,
        internal_protocol_excerpt={k: protocol[k] for k in ("primary_inference", "crops", "upstream", "selection")},
        input_identities=snapshots.files, input_sha256={name: record["sha256_before"] for name, record in snapshots.files.items()},
        all_consumed_inputs_sha256_unchanged_before_after=True,
        native_prior_audit_coverage=dict(native_files=len(native_records),
            previously_audited_native_files=len(native_records)-len(supplemental), supplemental_native_files=len(supplemental),
            supplemental_native_paths=supplemental,
            scope="Unlisted native files are supplementary reads rehashed by this diagnostic; they are not retroactively claimed as verified by the prior final audit."))
    out = root / "evaluation/mechanism_analysis/native_output_diagnostic.json"
    if out.exists():
        require(json.loads(out.read_text(encoding="utf-8-sig")) == result, "Existing posthoc diagnostic differs; do not overwrite it")
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        temporary = out.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        temporary.replace(out)
    return result


def self_test():
    """Small pure checks; no real files, audit polling, models or output writes."""
    assert pixel_point([0, 1000], [101, 81]) == [0, 80]
    assert pixel_point([5, 5], [101, 101]) == [0, 0]  # Python ties-to-even mapping.
    assert strict_parse("CLICK 200 300\n")["action"] == "click"
    assert strict_parse("CLICK 200,300") is None and lax_parse("CLICK 200,300") == [200, 300]
    assert lax_parse("CLICK 1001 0") is None and lax_parse("CLICK -1,0") is None
    assert lax_parse("CLICK (1,2)") is None and lax_parse("CLICK 1.0,2") is None
    assert strict_parse(" WAIT ")["action"] == "wait"
    case = dict(sample_id="synthetic", instruction="click target", image_size=[101, 81], image_sha256="synthetic-image")
    prompt = "synthetic prompt permits output WAIT"
    def native(text, eos=True):
        parsed = strict_parse(text) if eos or text.endswith(("\n", "\r")) else None
        action = parsed or dict(action="wait", normalized_point=None, generated_decision=None, abstained=True, valid=False)
        return dict(**action, sample_id="synthetic", image_size=case["image_size"], image_sha256=case["image_sha256"],
            instruction_sha256=digest(case["instruction"].encode()), system_prompt=prompt, generated_text=text,
            generated_token_ids=[1, 2], generated_tokens=2, terminated_by_eos=eos, truncated=False,
            parse_valid=parsed is not None, point=pixel_point(action["normalized_point"], case["image_size"]),
            completed_lax_point=pixel_point(lax_parse(text), case["image_size"]) if eos or text.endswith(("\n", "\r")) else None,
            parse_failure=None if parsed else "format", unexecuted_candidate_at_token_limit=None,
            ground_truth_in_model_inputs=False, hooks_used=False, shared_trajectory=False)
    row, issues = inspect_native(case, native("WAIT"), prompt)
    assert not issues and row["literal_wait_normal_eos"] and not row["strict_recorded_wait_but_click_text"]
    row, issues = inspect_native(case, native("CLICK 200,300"), prompt)
    assert not issues and row["comma_click_recovered"] and row["strict_recorded_wait_but_click_text"]
    assert row["completed_lax_point"] == [20, 24]
    row, issues = inspect_native(case, native("CLICK 200 300", eos=False), prompt)
    assert not issues and row["category"] == "incomplete_click" and row["completed_lax_point"] is None
    altered = native("CLICK 1000 1000"); altered["completed_lax_point"] = [101, 81]
    _, issues = inspect_native(case, altered, prompt)
    assert {v["field"] for v in issues} >= {"completed_lax_pixel_mapping", "completed_lax_pixel_range"}
    assert paired([True, False, True], [False, True, True])["wins"] == 1
    assert paired([], [])["native_accuracy"] is None
    return dict(complete=True, synthetic_only=True, model_execution=False, real_result_reads=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--self-test", action="store_true", help="Pure synthetic checks; never reads run artifacts")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test())); return
    result = run(args.root)
    print(json.dumps(dict(complete=result["complete"], posthoc=True, n=result["primary_denominator"],
        output=str(args.root / "evaluation/mechanism_analysis/native_output_diagnostic.json")), ensure_ascii=False))
    if not result["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
