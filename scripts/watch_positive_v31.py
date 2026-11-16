"""Finish the registered training-positive control after the original V31 chain."""
from pathlib import Path
import subprocess
import sys
import time
import run_reinspection_v31 as run

OUT=run.OUT;AUG=OUT/'positive_utility'
def status(phase,**kw):
    run.write(OUT/'positive_chain.json',dict(phase=phase,updated_at_utc=run.now(),complete=phase=='complete',**kw))

def finish():
    r=run.read(AUG/'results.json');ctx=run.read(OUT/'context256/results.json');rows=run.read(AUG/'predictions.json')['rows']
    train={c['sample_id'] for c in run.read(AUG/'data/public.json') if c['cohort']=='train'}
    frozen=run.read(AUG/'frozen.json');assert set(frozen['training_ids'])==train
    replayed=0
    for model_row in frozen['models'].values():
        model=run.load_model(model_row)
        assert all(s.removesuffix('|mined_positive') in train for s in model['training_ids'])
    natural=run.records([c for c in run.read(AUG/'data/public.json') if c['cohort']!='train'])
    from clickagent.reinspection_v31 import scores,decide,MODES
    eligible=[q for q in natural if q['eligible']]
    for mode in MODES:
        model=run.load_model(frozen['models'][mode]);prediction={q['sample_id']:p for q,p in zip(eligible,scores(model,eligible))}
        for q,row in zip(natural,rows):
            point,_=decide(q['base'],q['alternative'],prediction.get(q['sample_id'],0.),.1)
            assert point==row['methods'][mode];replayed+=1
    lines=['训练正例补充（注册时仅查看训练统计，原实验保留）',
        f'有效训练RGB={r["training_rgb"]}，比较对={r["training_pairs"]}；新增20对来自已有训练图，新增独立图像数=0。',
        '新增对使用真实预测候选中的正确点；训练标签只用于筛选监督，不进入开发推理。相同RGB所有比较共享总权重。',
        f'全部100图：短单局部{r["all"]["percent"]["short_single"]:.2f}%，256上下文原选择器{ctx["all"]["percent"]["combined"]:.2f}%，补充正例后{r["all"]["percent"]["combined"]:.2f}%，历史MVP{r["all"]["percent"]["MVP_historical"]:.2f}%。',
        '所有补充模型：'+str(r['all']['percent']),
        '修复/破坏：'+str(r['all']['replacements']),
        '补充三比较：'+str(r['all']['comparisons']),
        '训练内按RGB/来源组留出：'+str(r['training_crossfit']),
        '新增正例会改变训练对手分布，因此自然开发候选上的净收益和模糊/几何对照必须一起看，不能用训练正例命中率替代部署准确率。']
    (AUG/'interpretation.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    prior=(OUT/'final_readout.txt').read_text(encoding='utf-8')
    (OUT/'final_readout_augmented.txt').write_text(prior+'\n\n'+'\n'.join(lines)+'\n',encoding='utf-8')
    run.write(AUG/'final_verification.json',dict(complete=True,at_utc=run.now(),decision_replays=replayed,
        only_training_ids_in_fit=True,new_independent_training_rgb=0,primary_gates={k:v['positive'] for k,v in r['all']['comparisons'].items()}))
    reg=run.read(run.ROOT/'CURRENT_RESEARCH.json');assert reg['run_id']=='quality_readout_v15_3k'
    reg['active_experiment'].update(status='complete',phase='all_registered_comparisons_complete',results_path=str(OUT/'final_readout_augmented.txt'),positive_results_path=str(AUG/'results.json'),mainline_unchanged=True)
    run.write(run.ROOT/'CURRENT_RESEARCH.json',reg)
    run.write(OUT/'all_completion.json',dict(complete=True,at_utc=run.now(),primary_and_context_completion_sha256=run.sha(OUT/'final_completion.json'),
        positive_completion_sha256=run.sha(AUG/'completion.json'),final_verification_sha256=run.sha(AUG/'final_verification.json'),
        final_readout_sha256=run.sha(OUT/'final_readout_augmented.txt'),watch_source_sha256=run.sha(__file__),mainline_unchanged=True))
    status('complete',results_path=str(OUT/'final_readout_augmented.txt'))

def main():
    status('waiting_primary_and_context')
    while not (OUT/'final_completion.json').exists():
        if (OUT/'execution_chain.json').exists():assert run.read(OUT/'execution_chain.json')['phase']!='failed'
        time.sleep(15)
    for _ in range(40):
        p=subprocess.run(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True)
        if p.returncode==0 and max(int(s) for s in p.stdout.splitlines() if s.strip())<3000:break
        time.sleep(5)
    else:raise RuntimeError('GPU still busy;will not load duplicate backbone')
    status('positive_training_supplement')
    reg=run.read(run.ROOT/'CURRENT_RESEARCH.json');reg['active_experiment'].update(status='running',phase='training_positive_supplement');run.write(run.ROOT/'CURRENT_RESEARCH.json',reg)
    with (AUG/'stdout.log').open('a',encoding='utf-8') as stdout,(AUG/'stderr.log').open('a',encoding='utf-8') as stderr:
        p=subprocess.run([sys.executable,'-B','-X','utf8','-u',str(run.ROOT/'scripts/run_positive_utility_v31.py'),'all'],cwd=run.ROOT,stdout=stdout,stderr=stderr)
    assert p.returncode==0,'Positive supplement failed;see positive_utility/stderr.log'
    run.OUT=AUG;finish()

if __name__=='__main__':
    try:main()
    except BaseException as error:status('failed',error=str(error));raise
