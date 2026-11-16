"""Same full global prompt,causal point-field stop;fresh V29/MVP latency."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import time
import run_reinspection_v31 as common
import run_pair_latency_v30 as old
import resume_pair_latency_v30 as resumed

ROOT=common.ROOT;PARENT=common.OUT;BASE=ROOT/'runs/pair_latency_v30';OUT=PARENT/'point_prefix_latency'
ORIGINAL_RUNTIME=old.CurrentRuntime

def verify():
    p=common.read(OUT/'protocol.json')
    for f,h in p['hashes'].items():assert common.sha(f)==h,f
    return p

def prepare():
    if (OUT/'protocol.json').exists():return verify()
    diagnostic=common.read(PARENT/'point_prefix_cached_diagnostic.json')
    assert diagnostic['point_equal']==diagnostic['all_cases']==212 and diagnostic['target_equal']==212
    p=subprocess.run([sys.executable,'-B','-X','utf8','-m','unittest','discover','-s','tests','-p','test_point_prefix_v31.py','-v'],cwd=ROOT,capture_output=True,text=True,encoding='utf-8')
    OUT.mkdir(parents=True,exist_ok=True);common.write(OUT/'unit_tests.json',dict(exit_code=p.returncode,output=p.stdout+p.stderr));assert p.returncode==0,p.stdout+p.stderr
    for name in ['public','warmup']:
        q=OUT/'data'/f'{name}.json';q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(BASE/'data'/f'{name}.json',q)
    deps=[Path(__file__),ROOT/'src/clickagent/point_prefix_v31.py',ROOT/'tests/test_point_prefix_v31.py',
        ROOT/'scripts/run_pair_latency_v30.py',ROOT/'scripts/resume_pair_latency_v30.py',
        ROOT/'src/clickagent/grounded_composition_v28.py',BASE/'protocol.json',BASE/'ours/results.json',BASE/'mvp/results.json',
        PARENT/'point_prefix_cached_diagnostic.json',PARENT/'baseline_input_index.json',OUT/'data/public.json',OUT/'data/warmup.json',OUT/'unit_tests.json']
    # Authenticate all212historical global trajectories used for point equivalence.
    index=common.read(PARENT/'baseline_input_index.json')
    for name,h in index.items():
        if str(Path(name).parent).replace('\\','/').endswith('/raw/global'):
            assert common.sha(name)==h,name;deps.append(Path(name))
    common.write(OUT/'protocol.json',dict(registered_at_utc=common.now(),
        trigger='Primary short-prompt changed accuracy. Repair output scheduling while keeping original full systemprompt,input and greedy prefix exactly unchanged.',
        method='V29default keep exact local/fallback,all6frozenheads. Only global generation stops after BOTHcomplete t/p fields in actual tokenprefix. Local/legacy observations/scoring unchanged. No fabricated generated token or teacher prefix. Parser-only closing XMLtag explicitly marked;actual generated text/IDs retained. Auxiliary fields after coordinate cannot affect earlier coordinates in a causal decoder.',
        cached_equivalence='212/212existing global point/target outputs equal when replaying first complete pointprefix. Observed replay,not independentaccuracytest. Existing V29 predictions remain unchanged for this cached evidence.',
        fresh_timing='Exact same6ScreenSpot-Pro public screenshots,warmup,2uncachedrepetitions,FP16dualP5000,max262144pixels asV30. Fresh MVP run included;noGTaccuracyaccess. PNGdecode/preprocessing/generation/score included,load/warmup/report excluded.',
        checks='Compare72per-head point decisions from12freshqueries to originalV30;all methods/negative outcomes reported. No automatic adoption or benchmark superiority claim.',
        hashes={str(p):common.sha(p) for p in deps}))

class PrefixRuntime(ORIGINAL_RUNTIME):
    def __init__(self,folder):
        super().__init__(folder)
        from clickagent.grounded_composition_v28 import GLOBAL_PROMPT
        from clickagent.point_prefix_v31 import infer_prefix
        runtime=self.runtime;original=runtime.infer
        def infer(image,instruction,prompt,max_new_tokens):
            return infer_prefix(runtime,image,instruction,prompt,max_new_tokens) if prompt==GLOBAL_PROMPT else original(image,instruction,prompt,max_new_tokens)
        runtime.infer=infer
        self.metadata.update(global_schedule='Stop after actual complete target/point prefix;full original systemprompt unchanged',
            generated_text_contains_only_actual_decoded_tokens=True,parser_only_closing_xml_tag=True)

def worker(method):
    old.OUT=OUT;resumed.OUT=OUT;old.verify=verify;resumed.amendment=verify;old.CurrentRuntime=PrefixRuntime
    resumed.install_retry();resumed.worker(method)

def analyze():
    old.OUT=OUT;old.verify=verify;old.analyze()
    a=common.read(OUT/'analysis.json');reference=common.read(BASE/'ours/results.json')['rows']
    ref={(r['sample_id'],r['repetition']):r for r in reference};parity=0
    for r in common.read(OUT/'ours/results.json')['rows']:
        previous=ref[r['sample_id'],r['repetition']]
        assert r['output']['seed_points']==previous['output']['seed_points'];parity+=len(r['output']['seed_points'])
    assert parity==72
    ours=a['methods']['ours'];mvp=a['methods']['mvp'];prior=common.read(BASE/'analysis.json')['methods']['ours']['mean_seconds']
    lines=['保留完整提示、在坐标字段完成后停止：单题计时',
        '保留原完整全图提示，实际逐token生成到目标/坐标字段结束，不生成其后的参照字段。没有伪造生成token；解析器补闭合标签有明确记录。',
        '212条历史全图轨迹的坐标和目标名称均相同；这是缓存轨迹重放，不是新的独立准确率评估。',
        f'同V30六张图、每张两次的全新计时：我们{ours["mean_seconds"]:.2f}秒/题，MVP{mvp["mean_seconds"]:.2f}秒/题，耗时比{a["ours_to_mvp_ratio"]:.3f}。',
        f'我们原V30计时{prior:.2f}秒/题；本次12次查询的6个评分头，共72个坐标全部一致。',
        '未微调主干、未改变评分器或局部提示，未计算这六张图的准确率；不能据此宣称全面追平MVP。']
    (OUT/'explanation_zh.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    common.write(OUT/'verification.json',dict(complete=True,at_utc=common.now(),cached_global_parity=212,fresh_fixed_head_point_parity=parity))
    common.write(OUT/'completion.json',dict(complete=True,at_utc=common.now(),analysis_sha256=common.sha(OUT/'analysis.json'),verification_sha256=common.sha(OUT/'verification.json')))
    print('\n'.join(lines),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','ours','mvp','all']);args=p.parse_args()
    if args.stage=='prepare':prepare();return
    verify()
    if args.stage in ['ours','mvp']:worker(args.stage);return
    for method in ['ours','mvp']:
        folder=OUT/method;folder.mkdir(parents=True,exist_ok=True)
        if (folder/'results.json').exists() and common.read(folder/'results.json')['complete']:continue
        with (folder/'stdout.log').open('a',encoding='utf-8') as stdout,(folder/'stderr.log').open('a',encoding='utf-8') as stderr:
            child=subprocess.run([sys.executable,'-B','-X','utf8','-u',str(Path(__file__).resolve()),method],cwd=ROOT,stdout=stdout,stderr=stderr)
        assert child.returncode==0,f'{method} prefix timing failed;inspect {folder}/stderr.log'
    analyze()

if __name__=='__main__':main()
