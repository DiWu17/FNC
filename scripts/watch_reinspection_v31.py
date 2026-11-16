"""Continue the current authorized Python experiment through analysis;no scheduler."""
import json
from pathlib import Path
import subprocess
import sys
import time
import run_reinspection_v31 as run

OUT=run.OUT
def status(phase,**kw):
    value=dict(phase=phase,updated_at_utc=run.now(),complete=phase=='complete',**kw)
    run.write(OUT/'execution_chain.json',value);print(json.dumps(value,ensure_ascii=False),flush=True)

def launch(script,arguments,prefix):
    with (OUT/(prefix+'.stdout.log')).open('a',encoding='utf-8') as stdout,(OUT/(prefix+'.stderr.log')).open('a',encoding='utf-8') as stderr:
        p=subprocess.run([sys.executable,'-B','-X','utf8','-u',str(run.ROOT/'scripts'/script),*arguments],cwd=run.ROOT,stdout=stdout,stderr=stderr)
    assert p.returncode==0,f'{script} failed;see {prefix}.stderr.log'

def integrate():
    parent=run.read(OUT/'results.json');context=run.read(OUT/'context256/results.json')
    m=parent['all']['methods'];cm=context['all']['percent']
    lines=['V31整轮结论：互补重看与真实候选内容判断','',
        '固定112训练、100复用开发截图；主干与原评分器均冻结。128px原实验及训练输入QA后、开发前登记的256px上下文补充均完整报告。',
        f'100图：原长格式自由局部点{m["original_local"]["percent"]:.2f}%；精简单局部含回退{m["short_single"]["percent"]:.2f}%；历史MVP{m["MVP_historical"]["percent"]:.2f}%。',
        f'相同视图数：附近重看共识{m["refine_consensus"]["percent"]:.2f}%，互补重看共识{m["diverse_consensus"]["percent"]:.2f}%。',
        f'原生点并集覆盖：附近{m["refine_union"]["percent"]:.2f}%，互补{m["diverse_union"]["percent"]:.2f}%。覆盖是诊断上限，不是最终准确率。',
        f'128上下文内容+几何{m["combined"]["percent"]:.2f}%，其几何/模糊/打乱对照分别{m["geometry"]["percent"]:.2f}/{m["blurred"]["percent"]:.2f}/{m["shuffled"]["percent"]:.2f}%。',
        f'256上下文内容+几何{cm["combined"]:.2f}%，其几何/模糊/打乱对照分别{cm["geometry"]:.2f}/{cm["blurred"]:.2f}/{cm["shuffled"]:.2f}%。',
        '', '128选择器修复/破坏：'+str(parent['all']['replacements']),
        '256选择器修复/破坏：'+str(context['all']['replacements']),
        '128预登记五比较：'+str(parent['all']['comparisons']),
        '256补充三比较：'+str(context['all']['comparisons']),
        '', '新72和旧28分开：']
    for co in ['new_development','old_val']:
        p=parent[co]['methods'];c=context[co]['percent']
        lines.append(f'{co}:短单局部{p["short_single"]["percent"]:.2f}%，附近/互补共识{p["refine_consensus"]["percent"]:.2f}/{p["diverse_consensus"]["percent"]:.2f}%，内容128/256={p["combined"]["percent"]:.2f}/{c["combined"]:.2f}%，历史MVP={p["MVP_historical"]["percent"]:.2f}%。')
    lines += ['', '6张训练图的配对全图阶段计时：'+str(parent['prompt_timing']['means']),
        '全图阶段计时和共享视图工作量不能替代独立运行完整策略的端到端计时。',
        '本轮是历史开发集上的机制筛查，不是正式独立MVP优势确认。没有按最高开发分替换主线，也没有自动启动1581题完整评测。']
    (OUT/'final_readout.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    reg=run.read(run.ROOT/'CURRENT_RESEARCH.json');assert reg['run_id']=='quality_readout_v15_3k'
    reg['active_experiment'].update(status='complete',phase='primary_and_context_supplement_verified',
        results_path=str(OUT/'final_readout.txt'),context_results_path=str(OUT/'context256/results.json'),mainline_unchanged=True)
    run.write(run.ROOT/'CURRENT_RESEARCH.json',reg)
    run.write(OUT/'final_completion.json',dict(complete=True,at_utc=run.now(),mainline_unchanged=True,
        primary_completion_sha256=run.sha(OUT/'completion.json'),context_completion_sha256=run.sha(OUT/'context256/completion.json'),
        final_readout_sha256=run.sha(OUT/'final_readout.txt'),watch_source_sha256=run.sha(__file__)))
    status('complete',results_path=str(OUT/'final_readout.txt'))

def main():
    status('waiting_for_primary')
    while not (OUT/'completion.json').exists():
        if (OUT/'status.json').exists():
            state=run.read(OUT/'status.json');assert state['phase']!='failed',state
        time.sleep(15)
    assert run.read(OUT/'completion.json')['complete']
    # The primary process releases its backbone on exit;avoid simultaneous models.
    for attempt in range(40):
        p=subprocess.run(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True)
        if p.returncode==0 and max(int(s.strip()) for s in p.stdout.splitlines() if s.strip())<3000:break
        time.sleep(5)
    else:raise RuntimeError('GPU still occupied after primary completion;will not load second backbone')
    status('primary_closeout');launch('closeout_reinspection_v31.py',[],'closeout')
    reg=run.read(run.ROOT/'CURRENT_RESEARCH.json');reg['active_experiment'].update(status='running',phase='context256_supplement',context_status_path=str(OUT/'context256/status.json'));run.write(run.ROOT/'CURRENT_RESEARCH.json',reg)
    status('context256_supplement');launch('run_context256_v31.py',['all'],'context256_execution')
    integrate()

if __name__=='__main__':
    try:main()
    except BaseException as error:status('failed',error=str(error));raise
