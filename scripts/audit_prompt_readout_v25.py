"""Post-run V25 replay, descriptive diagnosis and static scientific figures.

This never changes the registered prompt, models, sampling or decision gates.
"""
import os
os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
from collections import Counter
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'runs/prompt_readout_v25/mplconfig'))
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import torch
import run_prompt_readout_v25 as run
from clickagent.prompt_readout_v25 import select_readouts,compose_arrays,ARMS,ALL_ARMS,LOCATION_CENTERS
from clickagent.evidence_retention import RetentionModel
from clickagent.crossfit_proposer import state_sha256
from clickagent.visual_feedback_v23 import digest
from run_visual_feedback_v23 import read,write,sha,hit


def audit():
    out=run.OUT;spec=run.verify();completion=read(out/'completion.json')
    assert completion['complete']
    for path,h in completion['artifact_sha256'].items():assert sha(path)==h,path
    provenance=read(out/'data/provenance.json')
    for i,(path,h) in enumerate(provenance.items(),1):
        assert sha(path)==h,path
        if i%50==0:print(f'Fixed historical input audit {i}/{len(provenance)}',flush=True)
    cases=read(out/'data/public.json');labels=read(out/'data/labels.json')
    train=[c for c in cases if c['cohort']=='train'];dev=[c for c in cases if c['cohort']!='train']
    assert len(train)==56 and len(dev)==36 and len({c['group'] for c in cases})==92
    for c in cases:
        assert not set(c)&{'target_geometry','target','bbox','normalized_point'}
        assert sha(c['image'])==c['image_sha256']
    frozen=read(out/'models/frozen.json');freeze=datetime.fromisoformat(frozen['frozen_at_utc'])
    protocol_sha=sha(out/'protocol.json');tasks=read(out/'data/tasks.json')
    stage_counts=Counter();timing=Counter();raw_index={};composition_checks=0
    spatial_counts=Counter();spatial_mass=[]
    case_index={c['sample_id']:c for c in cases}
    for i,t in enumerate(tasks,1):
        p=out/f'raw/{t["task_id"]}.npz'
        with np.load(p,allow_pickle=False) as z:
            m=json.loads(str(z['metadata']));c=case_index[t['sample_id']]
            assert m['task']==t and m['input_identity']==t['input_identity']
            assert m['protocol_sha256']==protocol_sha and m['image_sha256']==c['image_sha256']
            assert m['transform']==c['views'][t['view_index']]['transform']
            assert m['own_token_forward'] and m['all_backbone_parameters_frozen']
            assert not m['ground_truth_in_model_input']
            n=len(m['generated_token_ids']);assert len(m['generated_prefixes'])==n
            assert (m['generated_prefixes'][-1] if n else '')==m['generated_text']
            assert z['all_attention'].shape[0]==z['all_hidden'].shape[0]==n
            assert z['all_attention'].shape[1:]==(96,int(np.prod(m['grid_shape'])))
            assert z['all_hidden'].shape[1]==12288
            stages,parsed=select_readouts(m['generated_text'],m['generated_prefixes'],
                ended_by_eos=m['terminated_by_eos'],capped=m['token_cap_reached'])
            assert stages==m['readout_stages']
            for k,v in parsed.items():assert m[k]==v,(t['task_id'],k)
            for name,ix in stages.items():
                assert ix==sorted(set(ix)) and all(1<=j<=n for j in ix)
                stage_counts[f'{c["cohort"]}/{t["prompt"]}/{name}_available']+=bool(ix)
            if t['prompt']=='context':
                if not m['ground_complete']:
                    assert not any(stages[k] for k in ['target','container','location','phrase']), 'No fallback to unframed text permitted'
                stage_counts[f'{c["cohort"]}/context/ground_complete']+=m['ground_complete']
                location=m['ground_fields'].get('location','unknown').lower().strip()
                stage_counts[f'{c["cohort"]}/context/recognized_geometric_location']+=location in LOCATION_CENTERS
                stage_counts[f'{c["cohort"]}/context/phrase_without_entity']+=bool(stages['phrase'] and not stages['target'])
            if c['cohort']=='train':assert datetime.fromisoformat(m['capture_completed_at_utc'])<freeze
            else:assert datetime.fromisoformat(m['capture_started_at_utc'])>freeze
            if stages['target'] and stages['coordinate']:
                assert max(stages['target'])<min(stages['coordinate'])
            # Verify all arithmetic operators on each framed trajectory. No
            # target geometry participates in this evidence construction.
            if t['prompt']=='context' and stages['target']:
                a={k:z[k].copy() for k in z.files if k!='metadata'}
                if c['cohort']=='new_development' and location in LOCATION_CENTERS and stages['location']:
                    cx,cy=LOCATION_CENTERS[location];col,row=int(cx*3),int(cy*3)
                    gh,gw=m['grid_shape'];yy,xx=np.meshgrid((np.arange(gh)+.5)/gh,(np.arange(gw)+.5)/gw,indexing='ij')
                    region=(xx>=col/3)&(xx<(col+1)/3)&(yy>=row/3)&(yy<(row+1)/3)
                    loc=a['all_attention'][np.asarray(stages['location'])-1].mean(0)
                    loc=(loc/np.maximum(loc.sum(-1,keepdims=True),1e-12)).mean(0).reshape(gh,gw)
                    spatial_mass.append(float(loc[region].sum()));py,px=np.unravel_index(loc.argmax(),loc.shape)
                    spatial_counts['views_with_entity_and_recognized_location']+=1
                    spatial_counts['location_attention_peak_in_named_bin']+=bool(region[py,px])
                    # Post-hoc annotation diagnosis only. No labels enter the
                    # earlier capture, fusion, candidate or decision code.
                    geometry=labels[c['sample_id']];visible=[];w,h=m['image_size']
                    if geometry.get('type')=='bbox':
                        for box in geometry['boxes']:
                            x0,y0,x1,y1=[v*s for v,s in zip(box,c['image_size']*2)]
                            if m['transform'] is not None:
                                tr=m['transform'];x0=(x0-tr['box'][0])*tr['scale'];x1=(x1-tr['box'][0])*tr['scale']
                                y0=(y0-tr['box'][1])*tr['scale'];y1=(y1-tr['box'][1])*tr['scale']
                            x0,x1=max(0,x0),min(w,x1);y0,y1=max(0,y0),min(h,y1)
                            if x1>x0 and y1>y0:visible.append((x0,y0,x1,y1))
                        spatial_counts['bbox_views_with_recognized_location']+=1
                        spatial_counts['bbox_target_intersects_view']+=bool(visible)
                        compatible=any(min(x1,(col+1)*w/3)>max(x0,col*w/3) and
                            min(y1,(row+1)*h/3)>max(y0,row*h/3) for x0,y0,x1,y1 in visible)
                        spatial_counts['bbox_intersects_named_location_bin']+=compatible
                reference,rm=compose_arrays(a,m,'entity_mean')
                q=reference['target_hidden'];mass=reference['target_attention'].sum(-1)
                for op in ['role_sum','role_product','location_gate']:
                    combined,cm=compose_arrays(a,m,op)
                    assert np.array_equal(q,combined['target_hidden'])
                    assert np.allclose(combined['target_attention'].sum(-1),mass,atol=1e-6,rtol=2e-5)
                    assert np.isfinite(combined['target_attention']).all() and (combined['target_attention']>=0).all()
                    assert cm['source_role_indices']=={k:stages[k] for k in ['target','container','location']}
                    composition_checks+=1
            timing[t['prompt']+'_tokens']+=n
            timing[t['prompt']+'_seconds']+=m['elapsed_seconds']
            timing[t['prompt']+'_capped']+=m['token_cap_reached']
            raw_index[str(p)]=sha(p)
        if i%25==0:print(f'Raw trajectory audit {i}/{len(tasks)}',flush=True)
    assert len(tasks)==spec['actual_new_vlm_trajectories']==len(raw_index)
    def nonempty(c,arm):
        with np.load(run.feature_path(c,arm),allow_pickle=False) as z:return len(z['points'])>0
    common=read(out/'common_training.json');expected=[c['sample_id'] for c in train if all(nonempty(c,a) for a in ARMS)]
    assert common['sample_ids']==expected and common['n']==len(expected)
    original_states={};normalization={}
    for name,info in spec['models'].items():
        saved=torch.load(info['path'],map_location='cpu',weights_only=True);model=RetentionModel();model.load_state_dict(saved['state_dict'])
        original_states[name]=state_sha256(model)
        normalization[name]=digest([saved['mean'].numpy().tolist(),saved['std'].numpy().tolist()])
    for arm in ARMS:
        for name in run.NAMES:
            row=frozen['models'][f'{arm}__{name}'];assert sha(row['checkpoint_path'])==row['checkpoint_sha256']
            assert row['initial_state_sha256']==original_states[name]
            assert row['normalization_sha256']==normalization[name]
            assert row['training_rgb']==len(expected) and not row['development_used']
            assert row['steps']==(run.TRAIN_STEPS if expected else 0)
            rng=np.random.default_rng(20262510+int(name[-1]));batches=[];order=[];cursor=0
            for _ in range(row['steps']):
                selected=[]
                while len(selected)<4:
                    if cursor==len(order):order=rng.permutation(len(expected)).tolist();cursor=0
                    selected.append(expected[order[cursor]]);cursor+=1
                batches.append(selected)
            assert row['batch_sequence_sha256']==digest(batches)
    pred=read(out/'predictions.json');rows=pred['rows'];lookup={(r['sample_id'],r['regime'],r['model'],r['arm']):r for r in rows}
    assert len(rows)==len(lookup)==36*6*(len(ALL_ARMS)+len(ARMS))
    torch.set_num_threads(4);loaded={};feature_index={};replays=0;parity=0
    old={(r['sample_id'],r['model']):r for r in read(run.PRIOR/'predictions.json')['rows'] if r['arm']=='baseline'}
    mvp_cases={c['sample_id']:c for c in read(run.V21/'data/public.json')}
    for c in dev:
        sid=c['sample_id'];g=labels[sid]
        assert all(c[k]==mvp_cases[sid][k] for k in run.PUBLIC),'MVP public input mismatch'
        for arm in ALL_ARMS:
            p=run.feature_path(c,arm);record=run.load_record(c,arm);feature_index[str(p)]=sha(p)
            for regime in ['frozen','adapted']:
                if regime=='adapted' and arm not in ARMS:continue
                for name in run.NAMES:
                    model_key=(regime,name,None if regime=='frozen' else arm)
                    if model_key not in loaded:
                        path=spec['models'][name]['path'] if regime=='frozen' else frozen['models'][f'{arm}__{name}']['checkpoint_path']
                        saved=torch.load(path,map_location='cpu',weights_only=True);model=RetentionModel().eval();model.load_state_dict(saved['state_dict']);loaded[model_key]=(model,saved)
                    model,saved=loaded[model_key];point=None;top3=[];top9=[]
                    if record['n']:
                        inputs,_=run.tensors([record],saved['mean'].numpy(),saved['std'].numpy(),'aligned','cpu')
                        with torch.inference_mode():scores=model(**inputs).numpy()
                        point=record['points'][int(scores.argmax())].tolist()
                        order=np.argsort(-scores,kind='stable');top3=record['points'][order[:3]];top9=record['points'][order[:9]]
                    actual=lookup[sid,regime,name,arm]
                    assert point==actual['point'] and hit(point,c,g)==actual['hit']
                    assert any(hit(p.tolist(),c,g) for p in record['points'])==actual['oracle']
                    assert any(hit(p.tolist(),c,g) for p in top3)==actual['top3_oracle']
                    assert any(hit(p.tolist(),c,g) for p in top9)==actual['top9_oracle']
                    if regime=='frozen' and arm=='name':
                        assert point==old[sid,name]['point'] and actual['hit']==old[sid,name]['hit'];parity+=1
                    replays+=1
        mvp=read(run.V21/f'mvp/{run.key(sid)}.json')
        assert pred['native'][sid]['MVP']==mvp['output']['point']
        print(f'Decision audit {dev.index(c)+1}/{len(dev)}, {replays} replays',flush=True)
    write(out/'cache_audit_index.json',dict(raw_sha256=raw_index,development_feature_sha256=feature_index))
    receipt=dict(complete=True,completed_at_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=sha(__file__),protocol_sha256=protocol_sha,train_rgb=56,development_rgb=36,
        common_training_rgb=common['n'],zero_train_dev_overlap=True,all_adapters_frozen_before_new_dev_capture=True,
        raw_trajectories=len(tasks),stage_counts=dict(stage_counts),composition_checks=composition_checks,
        prediction_replays=replays,original_prediction_parity=parity,unchanged_original_checkpoints=True,
        historical_input_hashes_verified=len(provenance),
        posthoc_new24_spatial_diagnosis=dict(counts=dict(spatial_counts),
            mean_location_attention_mass_in_named_bin=float(np.mean(spatial_mass)) if spatial_mass else None,
            scope='Secondary post-hoc diagnostics, view-level correlated counts; bbox intersection is compatibility, not proof of correct referent binding.'),
        token_timing_totals=dict(timing),cache_index_sha256=sha(out/'cache_audit_index.json'))
    write(out/'final_audit.json',receipt)
    return cases,pred,receipt


def figures_and_diagnosis(cases,pred,audit):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    preferred=['Microsoft YaHei','SimHei','Noto Sans CJK SC']
    available={f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams['font.sans-serif']=[f for f in preferred if f in available]+['DejaVu Sans']
    plt.rcParams['axes.unicode_minus']=False
    out=run.OUT;results=read(out/'results.json');cohort=results['cohorts']['new_development']
    names=['name','context_name','framed_phrase','entity_mean','role_sum','role_product','location_gate','coordinate_name','coordinate_mean']
    labels_zh=['原命名','框定目标','框内全部token','目标均值','角色相加','角色软乘积','位置几何约束','坐标任务目标','坐标token']
    fig,axes=plt.subplots(1,2,figsize=(15,5.8),layout='constrained')
    for ax,family,title in zip(axes,['original3k','full'],['3k评分器家族','全量评分器家族']):
        x=np.arange(len(names));width=.36
        for offset,regime,color,label in [(-width/2,'frozen','#94a3b8','冻结原头'),(width/2,'adapted','#2563eb','同预算适配')]:
            values=[cohort['methods'][regime][family][a]['accuracy_percent'] for a in names]
            ax.bar(x+offset,values,width,color=color,label=label)
        ax.axhline(cohort['external']['MVP_cached'],color='#dc2626',ls='--',label='MVP缓存')
        ax.set_xticks(x,labels_zh,rotation=45,ha='right',fontsize=9);ax.set_ylim(0,100)
        ax.set_ylabel('命中率（%）');ax.set_title(title);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        ax.legend(fontsize=9)
    fig.suptitle('V25：完整描述与视觉证据组合\n新来源24张历史复用开发图，三种子均值；冻结与适配须分别比较',fontsize=13)
    fig.savefig(out/'summary_zh.png',dpi=160);plt.close(fig)
    # Explicitly post-hoc explanatory examples, never used for policy selection.
    from PIL import Image
    from matplotlib.patches import Rectangle
    dev=[c for c in cases if c['cohort']=='new_development']
    lookup={(r['sample_id'],r['regime'],r['model'],r['arm']):r for r in pred['rows']}
    def change(c):
        sid=c['sample_id']
        return sum(int(lookup[sid,'adapted',f'full_s{s}','role_product']['hit'])-
            int(lookup[sid,'adapted',f'full_s{s}','entity_mean']['hit']) for s in [1,2,3])
    ordered=sorted(dev,key=lambda c:(change(c),digest(c['sample_id'])))
    examples=[]
    for suffix,label,c in [('loss','净变化最低例',ordered[0]),('gain','净变化最高例',ordered[-1])]:
        task=run.task_for(c,0,'context');a,m=run.load_npz(out/f'raw/{task["task_id"]}.npz')
        if not m['readout_stages']['target']:continue
        maps=[];titles=[]
        for role,title in [('container','容器字段'),('location','位置字段'),('target','目标字段')]:
            indices=m['readout_stages'][role]
            raw=a['all_attention'][np.asarray(indices)-1].mean(0) if indices else np.zeros_like(a['all_attention'][0])
            raw=raw/np.maximum(raw.sum(-1,keepdims=True),1e-12)
            maps.append(raw.mean(0).reshape(m['grid_shape']));titles.append(title+('（无）' if not indices else ''))
        for op,title in [('role_sum','角色相加'),('role_product','角色软乘积'),('location_gate','位置几何约束')]:
            fused,_=compose_arrays(a,m,op);raw=fused['target_attention'][0]
            maps.append((raw/np.maximum(raw.sum(-1,keepdims=True),1e-12)).mean(0).reshape(m['grid_shape']));titles.append(title)
        with Image.open(c['image']) as im:screenshot=np.asarray(im.convert('RGB'))
        w,h=c['image_size'];fig,axes=plt.subplots(2,3,figsize=(12,9),layout='constrained')
        vmax=max(float(np.quantile(v,.995)) for v in maps);vmax=max(vmax,1e-8)
        geometry=read(out/'data/labels.json')[c['sample_id']]
        for ax,v,title in zip(axes.flat,maps,titles):
            ax.imshow(screenshot);ax.imshow(v,cmap='magma',vmin=0,vmax=vmax,alpha=.55,extent=(0,w,h,0),interpolation='bilinear')
            if geometry.get('type')=='bbox':
                for x0,y0,x1,y1 in geometry['boxes']:ax.add_patch(Rectangle((x0*w,y0*h),(x1-x0)*w,(y1-y0)*h,fill=False,ec='#22c55e',lw=1.5))
            ax.set_title(title);ax.set_xlim(0,w);ax.set_ylim(h,0);ax.axis('off')
        fields=m['ground_fields'];description=' / '.join(f'{k}: {v}' for k,v in fields.items())
        fig.suptitle(f'{label}（按全量适配头三种子净命中变化事后选取：{change(c):+d}/3）\n'+description+
            '\n图仅显示全局证据，净变化来自完整多视图最终决策。每头先空间归一化再平均，共用色阶。\n绿色框仅用于结果解释，不参与生成或合成。',fontsize=10)
        filename='example_'+suffix+'.png'
        fig.savefig(out/filename,dpi=140);plt.close(fig)
        examples.append(dict(file=filename,sample_id=c['sample_id'],posthoc_selection=label,
            seed_net_hit_change=change(c),fields=fields,instruction=c['instruction']))
    write(out/'example_selection.json',dict(posthoc=True,examples=examples,not_independent_evidence=True))
    diagnostic={}
    for family in ['original3k','full']:
        diagnostic[family]={}
        for regime in ['frozen','adapted']:
            diagnostic[family][regime]={}
            for label,(a,b) in run.PAIRS.items():
                pairs=[(lookup[c['sample_id'],regime,f'{family}_s{s}',a],
                    lookup[c['sample_id'],regime,f'{family}_s{s}',b]) for c in dev for s in [1,2,3]]
                diagnostic[family][regime][label]=dict(
                    repairs_per_seed=sum(x['hit'] and not y['hit'] for x,y in pairs)/3,
                    harms_per_seed=sum(y['hit'] and not x['hit'] for x,y in pairs)/3,
                    point_changes_per_seed=sum(x['point']!=y['point'] for x,y in pairs)/3,
                    coverage_gains_per_seed=sum(x['oracle'] and not y['oracle'] for x,y in pairs)/3,
                    coverage_losses_per_seed=sum(y['oracle'] and not x['oracle'] for x,y in pairs)/3)
    cost_report={}
    for prompt in ['context','coordinate']:
        cost_report[prompt]=dict(mean_fixed_view_vlm_capture_seconds=float(np.mean([pred['costs'][c['sample_id']][prompt]['seconds'] for c in dev])),
            mean_generated_tokens=float(np.mean([pred['costs'][c['sample_id']][prompt]['tokens'] for c in dev])),
            views=sum(pred['costs'][c['sample_id']][prompt]['views'] for c in dev),
            available_views=sum(pred['costs'][c['sample_id']][prompt]['available'] for c in dev),
            capped_views=sum(pred['costs'][c['sample_id']][prompt]['caps'] for c in dev),
            timing_scope='Actual fixed-view VLM capture for new24, including preprocessing and own-token readout hooks; excludes proposal/state-feature IO, adaptation and scoring. Not end-to-end latency.')
    write(out/'mechanism_diagnosis.json',dict(new_development_rgb=24,transitions=diagnostic,costs=cost_report,
        spatial_diagnosis=audit['posthoc_new24_spatial_diagnosis'],
        prior_mainline_retained=True,arbitrary_relative_anchor_relations_implemented=False))
    lines=['V25 提示词、完整指代表达与视觉证据组合：开发试验诊断','',
        f'审计完成：{audit["raw_trajectories"]}条新生成轨迹，{audit["prediction_replays"]}条决策重放，原流程{audit["original_prediction_parity"]}条预测一致。',
        f'训练56张，全部九种适配读出共同可用{audit["common_training_rgb"]}张；每头同初始化、归一化、批次与64步更新。',
        '8B和611919参数提议器冻结；原评分器保留。开发图36张（新24、旧12）均复用，未触及新独立测试。','']
    for family in ['original3k','full']:
        lines.append(f'{family}：')
        for regime in ['frozen','adapted']:
            m=cohort['methods'][regime][family]
            lines.append(regime+'新24命中率：'+', '.join(f'{label}{m[a]["accuracy_percent"]:.2f}%' for a,label in zip(names,labels_zh)))
        for pair in run.PAIRS:
            c=cohort['comparisons'][f'adapted/{family}/{pair}']
            interpretation='正向开发信号' if c['ci_pp'][0]>0 else ('负向开发信号' if c['ci_pp'][1]<0 else '区间跨零，尚未确认稳定收益')
            d=diagnostic[family]['adapted'][pair]
            lines.append(f'适配后 {pair}：{c["delta_pp"]:+.2f}个百分点，预登记校正区间[{c["ci_pp"][0]:+.2f}, {c["ci_pp"][1]:+.2f}]，{interpretation}；平均每种子修复{d["repairs_per_seed"]:.2f}、误改{d["harms_per_seed"]:.2f}张。')
        lines.append('')
    lines+=['原生坐标与外部对照（相同24张图，计算预算不同）：'+str(cohort['external']),
        '预登记推进信号：'+str(results['positive_gates']),'',
        '新24每张完整固定视图VLM读出成本：'+str(cost_report),'',
        '事后空间诊断（视图相关，不是额外独立样本）：'+str(audit['posthoc_new24_spatial_diagnosis']),'',
        '解释边界：框内全部token比较测试完整短语读出；实体均值/角色相加/软乘积共享同一实体query及图像attention质量，位置约束另用预测的九宫格几何先验。',
        '这是原始attention证据的运算，不是严格逻辑合取，也未实现“搜索框左侧”等任意锚点关系。小样本适配只能检查读出分布适应，不能替代完整训练。',
        '新提示词最多96输出token，历史命名64；不能宣称等计算。没有自动采用开发集最高分方法，没有新的独立泛化或超过MVP结论。',
        '保留原主线；后续是否扩大实验应根据校正区间、候选覆盖、实际选择与耗时共同判断。']
    (out/'diagnosis.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    cases,pred,receipt=audit();figures_and_diagnosis(cases,pred,receipt)
    out=run.OUT;reg=read(ROOT/'CURRENT_RESEARCH.json')
    reg['active_experiment'].update(phase='inference_adaptation_analysis_and_audit_complete',
        final_audit_path=str(out/'final_audit.json'),analysis_path=str(out/'diagnosis.txt'),mainline_unchanged=True)
    write(ROOT/'CURRENT_RESEARCH.json',reg)
    (out/'SESSION_STATE.txt').write_text('V25 inference, equal-budget adaptation, analysis and replay audit complete. Original mainline unchanged. No HTML.\n',encoding='utf-8')
    artifacts=['protocol.json','results.json','results.txt','predictions.json','completion.json','verification.json',
        'final_audit.json','cache_audit_index.json','diagnosis.txt','summary_zh.png','SESSION_STATE.txt','example_selection.json','mechanism_diagnosis.json','method_note.txt']
    artifacts += [e['file'] for e in read(out/'example_selection.json')['examples']]
    write(out/'delivery_manifest.json',dict(complete=True,artifact_sha256={str(out/n):sha(out/n) for n in artifacts},audit_source_sha256=sha(__file__)))
    print(json.dumps(receipt,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
