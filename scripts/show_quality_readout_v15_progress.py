"""Read-only progress presentation for the already-frozen V15 experiment.

Only progress.html and progress_view.json are written. Training sources,
protocols, features, checkpoints, and production status files remain untouched.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import html
import json
from pathlib import Path
import statistics
import time

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/'runs/quality_readout_v15'
LOCAL = timezone(timedelta(hours=8))


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8-sig'))
    except (FileNotFoundError, PermissionError, json.JSONDecodeError):
        return {} if default is None else default


def save(path, text):
    temporary = path.with_suffix(path.suffix+'.view.tmp')
    temporary.write_text(text, encoding='utf-8')
    for attempt in range(30):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 29: raise
            time.sleep(.1)


def stamp(value):
    return datetime.fromtimestamp(value, LOCAL).strftime('%m-%d %H:%M:%S') if value else '尚无记录'


def feature_count(cohort):
    files = list((RUN/f'features/{cohort}/cases').glob('*.json'))
    return len(files), max((p.stat().st_mtime for p in files), default=0)


def extraction_rate():
    try:
        lines = (RUN/'logs/pipeline.log').read_text(encoding='utf-8', errors='replace').splitlines()
    except (FileNotFoundError, PermissionError):
        return None
    records = []
    for line in lines:
        try: value = json.loads(line)
        except json.JSONDecodeError: continue
        if value.get('phase') == 'extracting_new_train' and 'newly_computed' in value:
            records.append(value)
    rates = []
    recent = records[-9:]
    for a, b in zip(recent, recent[1:]):
        delta = b['newly_computed']-a['newly_computed']
        elapsed = b['elapsed_seconds']-a['elapsed_seconds']
        if a['pid'] == b['pid'] and delta > 0 and elapsed > 0:
            rates.append(elapsed/delta)
    return statistics.median(rates) if rates else None


def render():
    current = read(RUN/'status.json')
    control = read(RUN/'run_control.json')
    if control.get('state') == 'deferred_for_3k':
        selected = read(ROOT/'runs/quality_readout_v15_3k/status.json')
        target = '../quality_readout_v15_3k/report.html'
        save(RUN/'progress.html', '<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
            f'<meta http-equiv="refresh" content="0;url={target}"><title>转到约3000条实验</title>'
            '<body style="font:18px/1.7 Microsoft YaHei,sans-serif;padding:40px">'
            f'<h1>正在执行约3000条数据实验</h1><p><a href="{target}">打开当前训练与验证进度</a></p>'
            '<p>原1.25万条全量任务继续暂停，缓存保留。</p></body></html>')
        save(RUN/'progress_view.json', json.dumps(dict(updated_at_utc=datetime.now(timezone.utc).isoformat(),
            operational_state='deferred_for_3k',active_run='quality_readout_v15_3k',
            active_phase=selected.get('phase'),active_complete=selected.get('complete',False),
            full_run_active=False,display_only=True),ensure_ascii=False,indent=2)+'\n')
        return False
    diverted = control.get('state') == 'deferred_for_pilot'
    pilot = read(ROOT/'runs/quality_readout_v15_pilot/status.json') if diverted else {}
    pilot_result = read(ROOT/'runs/quality_readout_v15_pilot/evaluation/results.json') if pilot.get('complete') else {}
    train, train_latest = feature_count('new_train')
    reserve, reserve_latest = feature_count('new_reserve')
    done = (RUN/'completion.json').exists() and read(RUN/'completion.json').get('complete', False)
    model_names = [f'old_matched_aligned_s{s}' for s in (1, 2)]
    expanded = [f'expanded_{v}_s{s}' for v in ('rank', 'aligned', 'permuted') for s in (1, 2)]
    receipts = {n:read(RUN/f'models/{n}.json') for n in model_names+expanded}
    old_done = sum(bool(receipts[n].get('complete')) for n in model_names)
    expanded_done = sum(bool(receipts[n].get('complete')) for n in expanded)
    tests_done = sum(read(RUN/f'evaluation/{s}/results.json').get('complete', False) for s in ('old_val', 'new_reserve'))
    phase = current.get('phase', '')
    if phase == 'training': active = 1 if current.get('model', '').startswith('old_matched') else 3
    elif phase == 'extracting_new_train': active = 2
    elif phase in ('all_models_frozen', 'extracting_new_reserve'): active = 4
    elif phase.startswith('evaluating_'): active = 5
    else: active = 0
    if diverted: active = 0
    titles = {1:'原数据等步数对照训练', 2:'新增训练数据的 Qwen 特征提取', 3:'扩容模型训练', 4:'新预留数据特征提取', 5:'评估与结果整理'}
    updated = datetime.fromisoformat(current['updated_at_utc']).timestamp() if current.get('updated_at_utc') else 0
    latest = max(updated, train_latest, reserve_latest)
    if diverted and pilot.get('updated_at_utc'):
        latest = datetime.fromisoformat(pilot['updated_at_utc']).timestamp()
    now = time.time()
    failed = phase == 'failed'
    healthy = not failed and now-latest < 600
    headline = '训练和开发评估已完成' if done else ('任务报告了错误' if failed else ('正常推进' if healthy else '较久未收到新进展，请查看日志'))
    stage_text = '结果已保存' if done else titles.get(active, '等待下一个阶段')
    if active == 2:
        explanation = '截图与标注已经入库；现在由冻结的 Qwen 计算每条指令的全图和局部特征。完成这一阶段后，才会训练使用新增数据的小评分器。'
    elif active == 1:
        explanation = '先让原数据模型训练相同的优化步数，用于区分数据增加与训练时间增加的影响。'
    elif active == 3:
        explanation = '训练数据为原 873 条加新增 12,524 条。两个种子和三个特征分支全部按预定方案运行。'
    elif active == 4:
        explanation = '训练模型已冻结，正在处理新预留数据；这些数据不会参与参数优化或模型挑选。'
    elif active == 5:
        explanation = '所有模型使用相同候选与判分规则，分别评估旧验证集和新开发预留集。'
    else: explanation = '这是 V15 实验的中文进度视图。'
    if diverted:
        headline = '全量采集已让位于小规模实验'
        stage_text = '先验证少量新增数据是否改善定位评分'
        explanation = '按用户的新安排，原全量任务暂时停止，已完成对照与所有缓存保留。小试使用873、918、963条训练数据、相同856步和两个种子；评估旧172条开发验证。'
        if pilot.get('phase') == 'training':
            stage_text = f'小试 {pilot.get("model", "")}：{pilot.get("step", 0)} / 856 步'
        elif pilot.get('complete'):
            headline = '小规模训练与验证已完成；全量采集仍暂停'
            stage_text = '查看下面的三档同预算对照结果'
        elif pilot.get('phase') == 'evaluating':
            stage_text = f'小试正在验证：{pilot.get("models_completed", 0)} / 6 个模型'
        elif pilot.get('phase') == 'failed':
            headline = '小规模实验遇到错误；全量采集仍暂停'
            stage_text = pilot.get('error', '请打开小试报告查看')
    speed = extraction_rate()
    eta = (12524-train)*speed/3600 if speed and train < 12524 else None
    stages = [
        (1, '原数据等步数对照', old_done, 2, old_done == 2, '两个随机种子，每个 13,400 步。'),
        (2, '新增训练数据特征提取', train, 12524, (RUN/'features/new_train/index.json').exists(), '计算冻结 Qwen 的全图和局部内部特征，当前主要耗时。'),
        (3, '扩容模型训练', expanded_done, 6, expanded_done == 6, '13,397 条训练数据；aligned、rank、permuted 各两个种子。'),
        (4, '新预留数据特征提取', reserve, 1439, (RUN/'features/new_reserve/index.json').exists(), '在训练模型冻结后处理，保留开发评估隔离。'),
        (5, '开发评估与报告', tests_done, 2, tests_done == 2, '旧验证 172 条＋新开发预留 1,439 条；评估 14 个固定模型。'),
    ]
    esc = html.escape
    parts = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta http-equiv="refresh" content="30">',
        '<title>ClickAgent V15 中文进度</title><style>',
        '*{box-sizing:border-box}body{max-width:1060px;margin:36px auto;padding:0 24px;background:#f3f6fb;color:#24364b;font:16px/1.65 "Segoe UI","Microsoft YaHei",sans-serif}h1{font-size:29px;margin:5px 0}h2{font-size:21px;margin:0}p{margin:10px 0}.sub,.detail{color:#63758b}.hero,.stage,.note{background:#fff;border:1px solid #dfe6ee;border-radius:14px;padding:22px;margin:16px 0}.hero{border-left:6px solid #16846c}.badge{display:inline-block;font-size:13px;padding:3px 10px;border-radius:20px;background:#e6f5ef;color:#146a55}.stage.live{border-color:#4299ce;box-shadow:0 0 0 1px #4299ce}.line{display:flex;justify-content:space-between;align-items:center;gap:20px}.track{height:12px;border-radius:20px;background:#e9eef5;margin:13px 0;overflow:hidden}.fill{height:100%;background:#16846c;border-radius:20px}.live .fill{background:#278bc2}.stat{font-size:22px;font-weight:650;white-space:nowrap}.detail{font-size:14px}.links{display:flex;gap:18px;flex-wrap:wrap}a{color:#176e9e}.warning{color:#a64624}code{background:#edf2f8;padding:2px 5px;border-radius:4px}</style>',
        '<body><div class="sub">ClickAgent · V15 数据扩容实验</div><h1>训练与评估进度</h1>',
        f'<div class="hero"><span class="badge">{esc(headline)}</span><h2 style="margin-top:12px">当前：{esc(stage_text)}</h2><p>{esc(explanation)}</p>',
        f'<div class="detail">最近实际进展：{stamp(latest)}（北京时间） · 页面每 30 秒自动刷新</div></div>']
    if eta is not None and not diverted:
        parts.append(f'<p class="sub">近期约 {speed:.1f} 秒/条；仅新增训练特征这一阶段粗估还需 {eta:.0f} 小时。样本复杂度会改变速度，后续训练与评估时间另计。</p>')
    if failed: parts.append('<p class="warning">'+esc(current.get('error', '请查看运行日志。'))+'</p>')
    if diverted:
        parts.append('<div class="note"><strong><a href="../quality_readout_v15_pilot/report.html">打开小规模训练与结果看板</a></strong><p>下面的数量是全量任务停下时保留的进度，不表示它仍在采集。</p></div>')
        if pilot_result:
            rows = []
            for name, title in [('old', '原873条'), ('add45', '加45条，共918条'), ('add90', '加90条，共963条')]:
                hits = [pilot_result['methods'][f'{name}_s{s}']['top1_hits'] for s in (1, 2)]
                rows.append(f'<tr><td style="padding:8px 20px 8px 0">{title}</td><td style="padding-right:30px">{hits[0]} / 172</td><td>{hits[1]} / 172</td></tr>')
            parts.append('<div class="note"><h2>小试：同一原K16候选中的Top1命中</h2><table><tr><th>训练量</th><th>种子1</th><th>种子2</th></tr>'+''.join(rows)+'</table><p>旧验证集有历史复用；小试不代表独立泛化测试，也不能外推到全量新增数据。</p></div>')
    for number, title, count, total, finished, detail in stages:
        badge = '已完成' if finished else ('进行中' if active == number else '等待前序完成')
        percent = min(100, 100*count/total)
        parts.append(f'<div class="stage {"live" if active==number and not finished else ""}"><div class="line"><h2>{number}. {esc(title)}</h2><span class="badge">{badge}</span></div><div class="line"><div class="stat">{count:,} / {total:,}</div><div>{percent:.2f}%</div></div><div class="track"><div class="fill" style="width:{percent}%"></div></div><div class="detail">{esc(detail)}</div>')
        if active == number and phase == 'training':
            parts.append(f'<p>当前模型：<code>{esc(current.get("model", ""))}</code> · {current.get("step",0):,} / {current.get("total_steps",13400):,} 步</p>')
        if active == number and phase.startswith('evaluating_'):
            parts.append(f'<p>本组已评估 {current.get("models_completed",0)} / {current.get("models_total",14)} 个模型。</p>')
        parts.append('</div>')
    parts.extend(['<div class="note"><strong>怎样理解这些数字</strong><p>特征提取的进度按样本计数；模型训练的进度按优化步数计数，两者不能直接相加。两个原数据对照训练完成，不代表新增数据模型已训练完成。</p><p>训练损失用于检查优化过程；泛化效果以后面的评估为准。新预留集属于开发评估，本轮尚不构成论文最终独立测试。</p></div>',
        '<div class="links"><a href="report.html">实验报告与结果</a><a href="protocol.json">固定实验方案</a><a href="logs/pipeline.log">运行日志</a><a href="status.json">原始状态（技术信息）</a></div>',
        f'<p class="detail">本页生成于 {stamp(now)}（北京时间）。本展示程序只读取实验状态，不修改训练、候选、检查点或评估。</p></body></html>'])
    save(RUN/'progress.html','\n'.join(parts))
    save(RUN/'progress_view.json',json.dumps(dict(updated_at_utc=datetime.now(timezone.utc).isoformat(),
        completed=done,phase=phase,old_control_models=old_done,new_train_features=train,total_new_train_features=12524,
        expanded_models=expanded_done,new_reserve_features=reserve,evaluation_cohorts=tests_done,
        latest_experiment_progress_utc=datetime.fromtimestamp(latest,timezone.utc).isoformat() if latest else None,
        recent_seconds_per_new_query=speed,current_train_feature_remaining_hours=None if diverted else eta,
        operational_state=control.get('state'),pilot_phase=pilot.get('phase'),pilot_complete=pilot.get('complete',False),
        display_only=True),ensure_ascii=False,indent=2)+'\n')
    return done


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true');args=parser.parse_args()
    while True:
        done=render()
        if done or not args.watch:break
        time.sleep(30)


if __name__=='__main__':main()
