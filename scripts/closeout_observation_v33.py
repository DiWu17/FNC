"""Independent ROI, real-token, coordinate and decision replay; no generation."""
from collections import Counter,defaultdict
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_observation_regions_v33 as r
from closeout_independent_v32 import direct_hit
from clickagent.observation_regions_v33 import unique_support,box_mass

def main():
    p=r.OUT;spec=r.verify();assert r.read(p/'completion.json')['complete'];cases=r.read(p/'data/public.json');baseline=r.read(p/'data/baseline.json')
    plans_checked=0;coordinate_replays=0;local_replays=0;decision_replays=0;duplicate_groups=0;duplicate_mismatches=0
    concentration=defaultdict(list)
    for c in cases:
        sid=c['sample_id'];b=baseline[sid];saved=r.read(p/'plans'/(r.key(sid)+'.json'))
        gp=Path(b['global_archive']) if c['cohort']=='train' else p/'raw/global'/(r.key(sid)+'.npz')
        arrays,meta=r.cache.load(gp);prob,cm=r.coordinate_distribution(arrays,meta)
        np.testing.assert_allclose(prob,saved['coordinate_probability'],rtol=1e-12,atol=1e-12);assert cm==saved['coordinate_evidence'];coordinate_replays+=1
        expected=r.plan_regions(c['image_size'],meta['grid_shape'],b['first_crop'],b['point'],saved['proposal'],prob,sid)
        assert expected==saved['plans'];plans_checked+=1
        if cm['available']:
            from clickagent.observation_regions_v33 import grid_points
            points=grid_points(meta['grid_shape'],c['image_size']);positive=prob>0
            concentration[c['cohort']+'_coordinate_entropy'].append(float(-(prob[positive]*np.log(prob[positive])).sum()/np.log(len(prob))))
            concentration[c['cohort']+'_coordinate_mass_in_first_crop'].append(float(box_mass([b['first_crop']['box']],points,prob)[0]))
        points,weights=unique_support(saved['proposal']['all_points'],saved['proposal']['all_scores'])
        if len(points):concentration[c['cohort']+'_name_mass_in_first_crop'].append(float(box_mass([b['first_crop']['box']],points,weights)[0]))
    dev=[c for c in cases if c['cohort']=='development'];labels=r.read(p/'data/development_labels.json');pred={q['sample_id']:q for q in r.read(p/'predictions.json')['rows']}
    mechanisms={arm:Counter() for arm in r.ARMS};availability=defaultdict(lambda:defaultdict(Counter))
    for c in dev:
        sid=c['sample_id'];q=pred[sid];b=baseline[sid];gt=labels[sid];plan=r.read(p/'plans'/(r.key(sid)+'.json'));_,gm=r.cache.load(p/'raw/global'/(r.key(sid)+'.npz'));gp=r.pixel(gm['point'],c['image_size'])
        base_hits=[direct_hit(point,c,gt) for point in b['seed_points']];assert base_hits==q['base_hits'];decision_replays+=6
        first_miss=r.prior.crop_overlap(b['first_crop']['box'],c,gt)==0;duplicates=defaultdict(list)
        for arm in r.ARMS:
            _,m=r.cache.load(p/'raw'/arm/(r.key(sid)+'.npz'));assert m['ground_truth_in_model_input'] is False
            if m['point'] is None:point=None
            else:
                point=np.rint(np.asarray(m['point'],float)/1000*(np.asarray(m['image_size'])-1)/m['transform']['scale']+m['transform']['box'][:2])
                point=np.clip(point,[0,0],np.asarray(c['image_size'])-1).astype(int).tolist()
            assert point==m['world_point']==q['arms'][arm]['point'];local_replays+=1
            extra=direct_hit(point,c,gt);assert extra==q['arms'][arm]['extra_hit'];decision_replays+=1
            chosen=[r.consensus([gp,point0,point],point0,14) for point0 in b['seed_points']]
            assert chosen==q['arms'][arm]['consensus_points']
            hits=[direct_hit(point0,c,gt) for point0 in chosen];assert hits==q['arms'][arm]['consensus_hits'];decision_replays+=6
            crop=r.prior.crop_overlap(m['transform']['box'],c,gt);assert crop==q['arms'][arm]['crop_GT_fraction']
            mechanisms[arm]['baseline_errors']+=not base_hits[0]
            mechanisms[arm]['primary_crop_miss_errors']+=first_miss and not base_hits[0]
            mechanisms[arm]['extra_crop_covers_primary_miss_errors']+=first_miss and not base_hits[0] and crop>0
            mechanisms[arm]['extra_point_repairs_primary_miss_errors']+=first_miss and not base_hits[0] and extra
            mechanisms[arm]['valid_extra_swap_harms']+=base_hits[0] and point is not None and not extra
            label='coordinate_available' if q['coordinate_available'] else 'coordinate_absent'
            availability[label][arm]['n']+=1;availability[label][arm]['extra_hits']+=extra;availability[label][arm]['repairs_available']+=not base_hits[0] and extra
            duplicates[tuple(m['transform']['box'])].append((m['generated_text'],point))
        for values in duplicates.values():
            if len(values)>1:duplicate_groups+=1;duplicate_mismatches+=int(any(v!=values[0] for v in values[1:]))
    reserve_ids={q['sample_id'] for q in r.read(r.V32/'data/reserve_public.json')}
    assert not reserve_ids&set(pred) and len(dev)==115 and len(pred)==115
    result=r.read(p/'results.json');assert result['baseline_primary_hits']==38
    for arm in r.ARMS:
        records=[q['arms'][arm] for q in pred.values()];summary=result['arms'][arm]
        assert sum(q['extra_hit'] for q in records)==summary['extra_hits']
        assert sum(q['union_hits'][0] for q in records)==summary['baseline_plus_extra_oracle_hits']
        assert sum(q['consensus_hits'][0] for q in records)==summary['consensus_head_hits']['original3k_s1']
    extra=dict(complete=True,at_utc=r.prior.base.now(),mechanisms={k:dict(v) for k,v in mechanisms.items()},coordinate_availability_strata={k:{a:dict(v) for a,v in group.items()} for k,group in availability.items()},
        evidence_concentration={k:dict(n=len(v),mean=float(np.mean(v))) for k,v in concentration.items()},duplicate_crop_groups=duplicate_groups,duplicate_crop_output_mismatches=duplicate_mismatches)
    r.write(p/'mechanism_readout.json',extra)
    audit=dict(complete=True,at_utc=r.prior.base.now(),ROI_plan_replays=plans_checked,actual_coordinate_attention_replays=coordinate_replays,local_coordinate_replays=local_replays,final_and_extra_point_judgment_replays=decision_replays,
        frozen_hashes_verified=len(spec['hashes']),reserve_no_forward=True,duplicate_crop_groups=duplicate_groups,duplicate_crop_output_mismatches=duplicate_mismatches,mainline_unchanged=True)
    r.write(p/'independent_final_audit.json',audit)
    lines=['V33最终分析：固定完整提示早停，改变补充观察区域',
        '112既有训练图仅做缓存几何检查；115张V32已观察图作为开发集。58张保留图未运行，无评分器/主干训练。',
        '原生基线主头38/115；下表均在相同基线上额外执行一次同规格局部生成。候选并集是GT诊断，固定共识才是实际决策。',
        '分支 | 新视图命中 | 新增可修复 | 基线+新点可达 | 固定共识主头 | 共识修复/破坏 | 新增视图秒/题']
    for arm in r.ARMS:
        a=result['arms'][arm];lines.append(f'{arm} | {a["extra_hits"]} | {a["extra_repairs_available"]} | {a["baseline_plus_extra_oracle_hits"]} | {a["consensus_head_hits"]["original3k_s1"]} | {a["consensus_primary_repairs"]}/{a["consensus_primary_harms"]} | {a["mean_extra_seconds"]:.2f}')
    lines+=['预登记三比较：'+str(result['primary_comparisons']), '全部六头固定共识：'+str({arm:result['arms'][arm]['consensus_head_hits'] for arm in r.ARMS}),
        '新裁剪几何覆盖：'+str({arm:dict(intersects=result['arms'][arm]['crop_intersects'],full=result['arms'][arm]['crop_full_ge99pct']) for arm in r.ARMS}),
        '主裁剪漏目标错误的补救：'+str(extra['mechanisms']),'坐标证据可用/缺失分层：'+str(extra['coordinate_availability_strata']),
        '证据集中程度（描述性，不是正确概率）：'+str(extra['evidence_concentration']),
        '同框重复输出检查：'+str(dict(groups=duplicate_groups,mismatches=duplicate_mismatches)),
        '同115图历史MVP为39/115；本轮添加视图、已复用开发数据，不能宣称独立超过或速度优于MVP。全局重放增加的是研究采集成本，缓存基线与额外视图耗时相加不等于新系统独立端到端计时。',
        'fullpool只指冻结自有候选提议器的全部输出点，不代表恢复全部视觉信息或取消其上游内部预算。坐标与名称缺失时回退到匹配随机且保留样本。',
        '全部主比较和负结果均保留；不按最高开发分更换分支或评分头，不自动采用新模型。']
    final=p/'final_readout.txt';final.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    reg=r.read(r.ROOT/'CURRENT_RESEARCH.json');assert reg['run_id']=='quality_readout_v15_3k'
    for model in reg['models'].values():assert r.sha(model['path'])==model['sha256']
    reg['active_experiment'].update(status='complete',phase='independent_closeout_verified',results_path=str(final),completion_path=str(p/'final_completion.json'));r.write(r.ROOT/'CURRENT_RESEARCH.json',reg)
    r.write(p/'final_completion.json',dict(complete=True,at_utc=r.prior.base.now(),original_completion_sha256=r.sha(p/'completion.json'),final_readout_sha256=r.sha(final),mechanism_sha256=r.sha(p/'mechanism_readout.json'),independent_audit_sha256=r.sha(p/'independent_final_audit.json')))
    print('\n'.join(lines),flush=True)

if __name__=='__main__':main()
