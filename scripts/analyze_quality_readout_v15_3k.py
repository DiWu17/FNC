"""Read-only result interpretation; writes only unsealed analysis/display files."""
from datetime import datetime, timezone
import html
import json
from pathlib import Path

import finalize_quality_readout_v15_3k as audit

run=audit.run
out=run.OUT
completion=run.read(out/'completion.json')
run.require(completion['complete'],'Completion audit not finished')
for filename,expected in completion['artifact_sha256'].items():
    run.require(run.sha(filename)==expected,'Completion artifact changed')

overview={}
sources={}
for cohort in ('old_val','new_reserve'):
    result=run.read(out/f'evaluation/{cohort}/results.json');methods={}
    for arm in ('old8','old_matched','expanded'):
        hits=[result['methods'][f'{arm}_s{s}']['top1_hits'] for s in (1,2,3)]
        methods[arm]=dict(hits=hits,mean_accuracy_percent=100*sum(hits)/(3*result['n']))
    overview[cohort]=dict(n=result['n'],methods=methods,comparisons=result['seed_mean_comparisons'],
        all3_seeds_improve_vs_both_controls=all(result['methods'][f'expanded_s{s}']['top1_hits']>result['methods'][f'{arm}_s{s}']['top1_hits'] for s in (1,2,3) for arm in ('old8','old_matched')))
    sources[cohort]={}
    for name,v in result['by_source'].items():
        rates={arm:100*sum(v['hits'][f'{arm}_s{s}'] for s in (1,2,3))/(3*v['n']) for arm in methods}
        sources[cohort][name]=dict(n=v['n'],mean_accuracy_percent=rates,expanded_minus_old8_pp=rates['expanded']-rates['old8'])

conclusion=dict(recorded_at_utc=datetime.now(timezone.utc).isoformat(),complete=True,analysis_only=True,
    summary='扩容到2973条后，两个开发验证集上均出现三个种子方向一致的总体提升；与同训练步数及旧8轮参考比较均为正。',
    overview=overview,by_source=sources,
    limits=['旧172条验证有历史复用，新300条为本轮开发验证，不是最终独立测试。',
        '平均区间以固定3个已训练模型为条件，不覆盖训练随机性的总体不确定性；种子不增加独立验证样本数。',
        'GroundCUA相比旧8轮参考平均仅提升1个百分点，AMEX收益最大，不能宣称所有来源收益同样大。',
        '新增训练同时改变来源与数量，不能只归因为数量；本轮没有验证架构或内部特征的独立贡献。',
        '没有真实点击、行为分类或端到端任务成功率测量。'],
    finalization='两组评估09:35已写出；列表/字典收尾错误通过独立脚本修复，4248份缓存预测与命中/区间重新核对，原训练、特征、预测和结果JSON未变。')
run.write(out/'analysis/readout.json',conclusion)
body=['<h1>2973条训练：结果解读</h1><p><strong>'+html.escape(conclusion['summary'])+'</strong></p>',
    '<p>下面均为三个种子的平均准确率；在同一实际原K16候选中强制选Top1，以几何命中判分。</p>',
    '<table><tr><th>训练方案</th><th>旧验证172条</th><th>新来源开发验证300条</th></tr>']
for arm,title in [('old8','原873条，8轮'),('old_matched','原873条，2976步'),('expanded','2973条，2976步')]:
    body.append('<tr><td>'+title+'</td>'+''.join(f'<td>{overview[c]["methods"][arm]["mean_accuracy_percent"]:.2f}%</td>' for c in ('old_val','new_reserve'))+'</tr>')
body.append('</table><h2>相对同训练步数对照的提升</h2><ul>')
for cohort,title in [('old_val','旧验证'),('new_reserve','新来源开发验证')]:
    v=overview[cohort]['comparisons']['expanded vs old_matched'];lo,hi=v['ci95']
    body.append(f'<li>{title}：+{100*v["delta"]:.2f}个百分点，固定3模型的配对聚类95%区间 [{100*lo:.2f}, {100*hi:.2f}]。</li>')
body.append('</ul><h2>新来源分项</h2><table><tr><th>来源</th><th>旧8轮参考</th><th>扩容后</th><th>变化</th></tr>')
for name,v in sources['new_reserve'].items():
    rates=v['mean_accuracy_percent'];body.append(f'<tr><td>{html.escape(name)}</td><td>{rates["old8"]:.2f}%</td><td>{rates["expanded"]:.2f}%</td><td>{v["expanded_minus_old8_pp"]:+.2f}个百分点</td></tr>')
body.append('</table><h2>结论边界</h2><ul>'+''.join('<li>'+html.escape(s)+'</li>' for s in conclusion['limits'])+'</ul>')
body.append('<p>'+html.escape(conclusion['finalization'])+'</p><p><a href="../report.html">逐种子结果</a> · <a href="../completion.json">完成审计</a> · <a href="../checks/completion_repair.json">收尾修复记录</a></p>')
page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>2973条实验结果解读</title><style>body{max-width:1080px;margin:32px auto;padding:0 24px;font:16px/1.7 Segoe UI,Microsoft YaHei,sans-serif;color:#24364b;background:#f5f7fb}table{width:100%;border-collapse:collapse;background:white}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:left}a{color:#176e9e}</style><body>'+'\n'.join(body)+'</body></html>'
(out/'analysis/readout.html').write_text(page,encoding='utf-8')
p=out/'report.html';page=p.read_text(encoding='utf-8');link='<p><strong><a href="analysis/readout.html">打开结果解读：平均收益、来源差异与结论边界</a></strong></p>'
if link not in page:
    temp=p.with_suffix('.html.analysis.tmp');temp.write_text(page.replace('</h1>','</h1>'+link,1),encoding='utf-8');run.base.replace(temp,p)
print(json.dumps(overview,ensure_ascii=False,indent=2))
