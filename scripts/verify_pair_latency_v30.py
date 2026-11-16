"""Audit pairing, immutable inputs, actual measurements and timing explanation."""
from collections import Counter
import math
from pathlib import Path
import numpy as np
import run_pair_latency_v30 as run


def main():
    p=run.verify();out=run.OUT;analysis=run.read(out/'analysis.json')
    amendment=run.read(out/'execution_amendment_01.json')
    assert run.sha(amendment['resume_source'])==amendment['resume_source_sha256']
    assert run.sha(out/'protocol.json')==amendment['original_protocol_sha256']
    for path,h in amendment['completed_original_measurement_sha256'].items():assert run.sha(path)==h
    public={c['sample_id']:c for c in run.read(out/'data/public.json')}
    methods={};checks=Counter()
    for method in ['ours','mvp']:
        results=run.read(out/method/'results.json');assert results['complete']
        rows=results['rows'];assert len(rows)==12
        assert Counter(r['sample_id'] for r in rows)==Counter({sid:2 for sid in public})
        assert {(r['sample_id'],r['repetition']) for r in rows}=={(sid,rep) for sid in public for rep in [1,2]}
        valid=0;deterministic=0;tokens=[];global_times=[];free_times=[];stage_total=[]
        for r in rows:
            c=public[r['sample_id']]
            assert r['method']==method and not r['warmup'] and r['elapsed_seconds']>0
            assert math.isfinite(r['elapsed_seconds']) and r['image_size']==c['image_size']
            assert r['output']['ground_truth_in_model_input'] is False
            assert r['output']['generation_calls']==len(r['output']['stages'])
            raw=run.read(out/method/'cases'/f'{run.key(c["sample_id"])}_r{r["repetition"]}.json')
            assert raw==r
            point=r['output']['point']
            if point is not None:
                assert len(point)==2 and all(0<=v<s for v,s in zip(point,c['image_size']));valid+=1
            for stage in r['output']['stages']:
                assert stage['elapsed_seconds']>0 and stage['tokens'] is not None
                if stage['part']=='global':global_times.append(stage['elapsed_seconds'])
                if stage['part']=='free':free_times.append(stage['elapsed_seconds'])
            tokens.append(sum(s['tokens'] for s in r['output']['stages']))
            stage_total.append(sum(s['elapsed_seconds'] for s in r['output']['stages']))
            assert stage_total[-1]<=r['elapsed_seconds']+.001
            if method=='ours' and r['output']['reason']=='keep_exact_free':
                f=next(s for s in r['output']['stages'] if s['part']=='free');tr=r['output']['crop']
                native=np.asarray(f['point'])/1000*(np.asarray(f['image_size'])-1)
                native=np.clip(np.rint(native/tr['scale']+tr['box'][:2]),[0,0],np.asarray(c['image_size'])-1).astype(int).tolist()
                assert point==native;checks['exact_local_coordinate_retained']+=1
            checks['raw_measurement_matches_summary']+=1
        for sid in public:
            rr=[r for r in rows if r['sample_id']==sid]
            if rr[0]['output']['point']==rr[1]['output']['point']:deterministic+=1
        assert np.isclose(np.mean([r['elapsed_seconds'] for r in rows]),analysis['methods'][method]['mean_seconds'])
        methods[method]=dict(valid_coordinate_outputs=valid,measured_queries=12,
            repeated_coordinate_agreement_screenshots=deterministic,
            mean_total_generated_tokens=float(np.mean(tokens)),
            mean_global_seconds=float(np.mean(global_times)) if global_times else None,
            mean_free_seconds=float(np.mean(free_times)) if free_times else None,
            mean_stage_seconds=float(np.mean(stage_total)),
            non_stage_overhead_seconds=float(np.mean([r['elapsed_seconds'] for r in rows])-np.mean(stage_total)),
            load_attempt_seconds=results.get('model_load_attempt_seconds',[results['model_load_seconds']]))
    for c in public.values():assert run.sha(c['image'])==c['image_sha256']
    completion=run.read(out/'completion.json');assert completion['complete'] and completion['original_completed_rows_preserved']
    assert completion['analysis_sha256']==run.sha(out/'analysis.json')
    ours=analysis['methods']['ours'];mvp=analysis['methods']['mvp']
    text=[
        '当前默认策略与MVP的单题计时（V30）',
        '固定6张ScreenSpot-Pro截图，各重复2次；同一双P5000、同一冻结Qwen3-VL-8B、单视图最多262144像素。',
        '当前方法为V29有效自由局部点直接保留、缺失则来源校准回退；没有启用可选学习门控。',
        '按需执行与封存V29默认输出已在100张开发图、6个头上600次坐标比较全部一致。',
        '每题包括图像解码、预处理、生成、状态采集、裁剪和评分；排除模型加载、预热和日志写入。没有复用预测来代替实际计时。',
        f'我们：平均{ours["mean_seconds"]:.2f}秒，中位数{ours["median_seconds"]:.2f}秒，范围{ours["min_seconds"]:.2f}—{ours["max_seconds"]:.2f}秒。',
        f'MVP：平均{mvp["mean_seconds"]:.2f}秒，中位数{mvp["median_seconds"]:.2f}秒，范围{mvp["min_seconds"]:.2f}—{mvp["max_seconds"]:.2f}秒。',
        f'耗时比（我们/MVP）：{analysis["ours_to_mvp_ratio"]:.3f}。六图小样本，不能当完整基准通用速度结论。',
        f'实际生成次数均值：我们{ours["mean_generation_calls"]:.2f}、MVP{mvp["mean_generation_calls"]:.2f}；生成token均值：我们{ours["mean_generated_tokens"]:.2f}、MVP{mvp["mean_generated_tokens"]:.2f}。',
        f'我们全图阶段平均{methods["ours"]["mean_global_seconds"]:.2f}秒，自由局部平均{methods["ours"]["mean_free_seconds"]:.2f}秒。长结构化输出及逐token内部状态采集均包括在计时中；没有单独消融测量二者的因果耗时占比。',
        '两边保留各自的运行实现、提示词和跨卡分配，因此是实际配置的比较，不能归因为某一个算法部件。',
        '之前20—26小时指两种方法各跑完1581题，合计3162次请求；不是单次、不是训练，也不要求时间复杂度呈指数增长。',
        f'按本次样本粗略线性外推：我们{analysis["full_1581_inference_hours_extrapolated"]["ours"]:.2f}小时，MVP{analysis["full_1581_inference_hours_extrapolated"]["mvp"]:.2f}小时，总计{analysis["combined_hours"]:.2f}小时；完整集回退率和输出长度会改变估计。',
        '第一进程因Windows状态文件替换冲突中断，5条已完成计时逐字节保留；增加仅I/O重试和断点恢复，重新加载/预热在单题计时之外。故障与修复记录保留。',
        '本次没有评估准确率，也没有启动1581题完整比较。正式完整比较仍待启动。',
    ]
    run.write(out/'verification.json',dict(complete=True,at_utc=run.now(),source_and_model_hashes_checked=len(p['hashes']),
        original_rows_preserved=5,checks=dict(checks),methods=methods,analysis_sha256=run.sha(out/'analysis.json')))
    (out/'explanation_zh.txt').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print('\n'.join(text))


if __name__=='__main__':main()
