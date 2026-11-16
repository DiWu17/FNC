"""Prepare a drop-only QC candidate; never mutate or switch the running V15."""
from __future__ import annotations
from collections import Counter
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from clickagent.training_quality_gate import unusable_instruction_reason, audit_sample

BASE=ROOT/'data/expansion_v15/manifests'
OUT=ROOT/'data/expansion_v15_qc1_candidate'
RUN=ROOT/'runs/quality_readout_v15/analysis/data_suitability/qc1_candidate'


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def rows(p):return [json.loads(x) for x in Path(p).read_text(encoding='utf-8').splitlines() if x.strip()]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,value):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def main():
    assert not (RUN/'preparation.json').exists(), 'Candidate already prepared; inspect its receipt instead of overwriting.'
    registry=read(ROOT/'data/expansion_v15/registry.json')
    original={}
    clean={};removed={}
    for name in ('train.jsonl','validation_reserve.jsonl','combined_train.jsonl'):
        path=BASE/name;assert sha(path)==registry['manifests'][name]['sha256']
        original[name]=rows(path)
        clean[name]=[r for r in original[name] if unusable_instruction_reason(r['instruction']) is None]
        removed[name]=[dict(case=r,reason=unusable_instruction_reason(r['instruction'])) for r in original[name] if unusable_instruction_reason(r['instruction']) is not None]
        dest=OUT/'manifests'/name;dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in clean[name]),encoding='utf-8')
    assert len(removed['train.jsonl'])==24 and len(removed['validation_reserve.jsonl'])==2
    assert len(removed['combined_train.jsonl'])==24
    old_ids={r['sample_id'] for r in original['combined_train.jsonl']}-{r['sample_id'] for r in original['train.jsonl']}
    assert len(old_ids)==873 and old_ids <= {r['sample_id'] for r in clean['combined_train.jsonl']}
    train=clean['train.jsonl'];val=clean['validation_reserve.jsonl']
    assert not {r['pixel_sha256'] for r in train}&{r['pixel_sha256'] for r in val}
    assert not {(r['dataset'],r['source_group']) for r in train}&{(r['dataset'],r['source_group']) for r in val}
    # Every retained row is exactly unchanged; subsets cannot introduce a new
    # screenshot leak into the already-protected split.
    for name in clean:
        by_id={r['sample_id']:r for r in original[name]}
        assert all(r==by_id[r['sample_id']] for r in clean[name])
    samples=audit_sample(train,30)
    write(RUN/'manual_review_90.json',dict(complete=False,selection_fixed_before_model_outcomes=True,
        training_only=True,n=90,by_source=dict(Counter(r['dataset'] for r in samples)),cases=samples,
        rubric=['Can the original instruction meaningfully identify a target?',
            'Does at least one annotated box match the described visible target?',
            'Is a coordinate/visibility error obvious?',
            'Record pass, uncertain, or semantic_error with a reason; do not infer clickability from target presence.'],
        interpretation='Stratified diagnostic audit, not an estimate certifying every source annotation. Wording strata are not rejection rules.'))
    write(RUN/'excluded.json',removed)
    accepted={r['sample_id'] for r in train}
    cached=[read(p) for p in (ROOT/'runs/quality_readout_v15/features/new_train/cases').glob('*.json')]
    eligible=[e for e in cached if e['sample_id'] in accepted]
    for e in eligible:
        record=next(r for r in train if r['sample_id']==e['sample_id'])
        identity=hashlib.sha256(json.dumps(record,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        assert identity==e['case_identity']
    manifest={name:dict(path=str(OUT/'manifests'/name),n=len(clean[name]),removed=len(removed[name]),sha256=sha(OUT/'manifests'/name)) for name in clean}
    result=dict(preparation_complete=True,adopted_by_running_pipeline=False,
        created_at_utc=datetime.now(timezone.utc).isoformat(),candidate_manifests=manifest,
        rules='Reject only non-text, invisible-only, or placeholder-only instructions. Preserve meaningful short labels, icons, pointing/hover wording, source text, GT boxes, split membership and all other retained-row fields. Do not refill removed queries.',
        new_train_images=len({r['pixel_sha256'] for r in train}),reserve_images=len({r['pixel_sha256'] for r in val}),
        train_by_source=dict(Counter(r['dataset'] for r in train)),manual_review_pending=90,
        cache_reuse_eligible_at_preparation=len(eligible),cached_excluded=len(cached)-len(eligible),
        cache_reuse_scope='Unchanged row identity only; existing file hashes must be reauthenticated when a new experiment adopts these references.',
        original_manifest_sha256={name:sha(BASE/name) for name in clean},
        current_pipeline='Original V15 continues unchanged. Switching training/evaluation to this candidate requires an explicit versioned protocol amendment before expanded fitting; no in-place replacement.',
        expected_actionable_note='Preserved for original-row identity; not certified clickability and not consumed by current readout supervision. These samples supervise target location only.',
        recommendation=['Retain three source datasets for grounding.',
            'Complete the fixed90-query manual semantic review before choosing any broader source filter.',
            'Use the same cleaned evaluation denominator for all compared models; preserve original data and both completed13400-step old controls.',
            'Inspect source-level outcomes and retain overall plus source-macro reporting; do not change sampling weights silently.',
            'Keep a distinct final benchmark after method freeze; the current reserve remains development data.'],
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),ROOT/'src/clickagent/training_quality_gate.py')})
    write(RUN/'preparation.json',result)
    for name in clean:assert sha(BASE/name)==registry['manifests'][name]['sha256']
    body='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>V15 数据质量纠偏建议</title><style>body{max-width:1000px;margin:36px auto;padding:0 24px;font:16px/1.75 Segoe UI,Microsoft YaHei,sans-serif;background:#f4f7fb;color:#24364b}table{width:100%;border-collapse:collapse;background:white}td,th{text-align:left;padding:12px;border-bottom:1px solid #ddd}a{color:#176e9e}</style><h1>建议：先修正明确异常，再验证扩容收益</h1><p><strong>这是候选修正版，尚未切换正在运行的 V15。</strong>原清单、原实验协议、已完成的对照和缓存均保留。</p><table><tr><th>清单</th><th>原数量</th><th>候选修正后</th></tr>'''
    for name in clean:body+=f'<tr><td>{html.escape(name)}</td><td>{len(original[name]):,}</td><td>{len(clean[name]):,}</td></tr>'
    body+='</table><p>移除24条训练、2条开发预留中的纯占位符指令；悬停、指向、有意义的短标签和图标不按表面形式删除。保留目标框和训练/预留分组，不补位。</p><p>已固定90条训练样本供人工语义复核，每来源30条；包含短标签、指向/悬停及部分被裁剪框，优先分散截图和来源组。人工结果尚未产生。</p><p>后续应先完成语义复核，再在单独版本协议中采用修正版；不能原地改正在运行实验的输入哈希。已有有效特征和两个原数据等步数对照可在身份核验后复用。</p><p><a href="preparation.json">准备记录</a> · <a href="excluded.json">隔离条目</a> · <a href="manual_review_90.json">固定人工复核清单</a> · <a href="../../../progress.html">原实验进度</a></p></html>'
    (RUN/'report.html').write_text(body,encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('preparation_complete','adopted_by_running_pipeline','candidate_manifests','manual_review_pending','cache_reuse_eligible_at_preparation','cached_excluded')},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
