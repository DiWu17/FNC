"""Post-run claim map under fixed, outcome-independent interpretation rules."""
from datetime import datetime,timezone
import hashlib
import html
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/contribution_v16'
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as stream:
        for b in iter(lambda:stream.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');temp.replace(p)

RULES={
 'no_evidence':{'question':'内部证据是否提供候选分数和几何之外的信息？','primary':True,'positive_claim':'内部状态值对固定候选的质量评分有增量价值。'},
 'spatial_shuffle':{'question':'内部证据与候选位置的正确对应是否重要？','primary':True,'positive_claim':'状态的点位对齐比保留未对齐的边际信息更有效。'},
 'qwen_native_completed_lax':{'question':'固定方法在新截图上的几何命中是否优于原生坐标？','primary':True,'positive_claim':'在明确不同视图预算的条件下，相比原生全图坐标获得定位收益。'},
 'attention_only':{'question':'隐藏状态通道是否在注意力之上提供帮助？','primary':False,'positive_claim':'隐藏状态派生特征提供附加收益（次要诊断）。'},
 'hidden_only':{'question':'注意力通道是否在隐藏状态之上提供帮助？','primary':False,'positive_claim':'注意力派生特征提供附加收益（次要诊断）。'},
 'token_mean':{'question':'编码前保留名称token差异是否优于预先平均？','primary':False,'positive_claim':'保留名称token差异提供收益（次要诊断）。'},
 'gui_verifier_k16':{'question':'与相同候选上的外部训练验证器相比如何？','primary':False,'positive_claim':'固定候选中的质量读出与外部验证器相比有收益（次要参考）。'}}


def plan():
    p=OUT/'analysis/claim_plan.json'
    if p.exists():
        v=read(p);assert v['rules']==RULES and v['analysis_source_sha256']==sha(__file__);return
    assert not (OUT/'evaluation/independent/results.json').exists()
    write(p,dict(created_at_utc=datetime.now(timezone.utc).isoformat(),rules=RULES,analysis_source_sha256=sha(__file__),
        criterion='Primary independent claims require positive lower endpoint of registered3-family Bonferroni98.333% cluster interval. Marginal95% intervals for prespecified secondary diagnostics remain descriptive; inconclusive is not equivalence.',
        no_model_or_variant_selection=True,mainline_remains_adopted=True))


def summarize():
    c=read(OUT/'completion.json');assert c['complete']
    for name,h in c['artifact_sha256'].items():assert sha(name)==h,name
    plan();r=read(OUT/'evaluation/independent/results.json');claims={}
    for variant,rule in RULES.items():
        value=r['comparisons']['aligned vs '+variant]
        ci=value['bonferroni_ci98_333'] if rule['primary'] else value['ci95']
        state='supported_within_registered_scope' if ci[0]>0 else 'full_model_lower_than_control' if ci[1]<0 else 'inconclusive'
        claims[variant]=dict(**rule,status=state,delta_pp=100*value['delta'],interval_pp=[100*x for x in ci],
            confirmation_scope='family-adjusted primary' if rule['primary'] else 'secondary descriptive; not family-adjusted')
    spatial=all(claims[v]['status']=='supported_within_registered_scope' for v in ('no_evidence','spatial_shuffle'))
    conclusion='独立子集支持：固定候选条件下，目标名称内部状态的空间对齐为轻量评分器提供增量信息。' if spatial else '独立子集尚未同时确认内部状态增量和空间对齐两条机制贡献；按各项证据收窄论文表述，保持采用模型和完整结果。'
    full=[r['methods'][f'aligned_s{s}']['top1_hits'] for s in (1,2,3)]
    output=dict(complete=True,created_at_utc=datetime.now(timezone.utc).isoformat(),summary=conclusion,n=r['n'],full_seed_hits=full,
        full_mean_accuracy_percent=100*sum(full)/(3*r['n']),claims=claims,completion_sha256=sha(OUT/'completion.json'),claim_plan_sha256=sha(OUT/'analysis/claim_plan.json'),
        limits=['只针对300张项目内首次使用截图，不是两个完整基准的榜单分数。','候选提议器已使用内部状态，消融隔离的是评分器的增量信息。','像素排重不证明未见应用或预训练数据独立。','三个种子不增加独立测试样本数；区间以固定模型为条件。','多token及通道分项是预设次要诊断，不能据此自动挑选新的主模型。','不由本轮结果推导真实执行成功率、SOTA或端到端速度优势。'])
    write(OUT/'analysis/contribution_map.json',output)
    body=['<h1>贡献证据与独立确认</h1>','<p><strong>'+html.escape(conclusion)+'</strong></p>',f'<p>完整模型三个种子命中：{full} / {r["n"]}；平均准确率 {output["full_mean_accuracy_percent"]:.2f}%。</p>',
        '<table><tr><th>研究问题</th><th>完整模型差值</th><th>预定区间</th><th>证据状态</th></tr>']
    labels={'supported_within_registered_scope':'在约定范围内得到支持','full_model_lower_than_control':'完整模型低于该对照','inconclusive':'证据不足以确认差异'}
    for v in claims.values():
        body.append(f'<tr><td>{html.escape(v["question"])}</td><td>{v["delta_pp"]:+.2f}个百分点</td><td>[{v["interval_pp"][0]:.2f}, {v["interval_pp"][1]:.2f}]<br>{html.escape(v["confirmation_scope"])}</td><td>{labels[v["status"]]}</td></tr>')
    body.append('</table><h2>论文表述边界</h2><ul>'+''.join('<li>'+html.escape(t)+'</li>' for t in output['limits'])+'</ul><p><a href="../report.html">全部消融结果</a> · <a href="../protocol.json">固定协议</a> · <a href="claim_plan.json">预先记录的解读规则</a></p>')
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>贡献证据解读</title><style>body{max-width:1150px;margin:32px auto;padding:0 24px;font:16px/1.7 Segoe UI,Microsoft YaHei,sans-serif;color:#24364b;background:#f4f7fb}table{width:100%;border-collapse:collapse;background:white}td,th{padding:9px;border-bottom:1px solid #ddd;text-align:left}a{color:#176e9e}</style><body>'+'\n'.join(body)+'</body></html>'
    (OUT/'analysis/readout.html').write_text(page,encoding='utf-8')
    p=OUT/'report.html';page=p.read_text(encoding='utf-8');link='<p><strong><a href="analysis/readout.html">打开贡献证据解读与论文表述边界</a></strong></p>'
    if link not in page:
        temp=p.with_suffix('.html.analysis.tmp');temp.write_text(page.replace('</h1>','</h1>'+link,1),encoding='utf-8');temp.replace(p)
    print(json.dumps(output,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    plan()
    if '--plan-only' not in sys.argv:
        while '--watch' in sys.argv and not (OUT/'completion.json').exists():
            s=read(OUT/'status.json')
            if s.get('phase')=='failed':raise RuntimeError('Scientific run failed; preserve all artifacts and repair before interpreting results: '+s.get('error',''))
            time.sleep(30)
        summarize()
