"""Run the causal-prefix timing repair after all registered efficacy controls."""
import subprocess
import sys
import time
import run_reinspection_v31 as run

OUT=run.OUT;PREFIX=OUT/'point_prefix_latency'
def status(phase,**kw):run.write(OUT/'prefix_chain.json',dict(phase=phase,updated_at_utc=run.now(),complete=phase=='complete',**kw))

def main():
    status('waiting_all_efficacy_controls')
    while not (OUT/'all_completion.json').exists():
        if (OUT/'positive_chain.json').exists():assert run.read(OUT/'positive_chain.json')['phase']!='failed'
        time.sleep(15)
    for _ in range(40):
        p=subprocess.run(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True)
        if p.returncode==0 and max(int(s) for s in p.stdout.splitlines() if s.strip())<3000:break
        time.sleep(5)
    else:raise RuntimeError('GPU busy after efficacy experiments')
    status('fresh_prefix_and_mvp_timing')
    reg=run.read(run.ROOT/'CURRENT_RESEARCH.json');reg['active_experiment'].update(status='running',phase='causal_prefix_timing_repair');run.write(run.ROOT/'CURRENT_RESEARCH.json',reg)
    with (PREFIX/'stdout.log').open('a',encoding='utf-8') as stdout,(PREFIX/'stderr.log').open('a',encoding='utf-8') as stderr:
        p=subprocess.run([sys.executable,'-B','-X','utf8','-u',str(run.ROOT/'scripts/run_point_prefix_latency_v31.py'),'all'],cwd=run.ROOT,stdout=stdout,stderr=stderr)
    assert p.returncode==0,'Prefix timing failed;see point_prefix_latency/stderr.log'
    explanation=(PREFIX/'explanation_zh.txt').read_text(encoding='utf-8')
    previous=(OUT/'final_readout_augmented.txt').read_text(encoding='utf-8')
    final=OUT/'complete_readout.txt';final.write_text(previous+'\n\n'+explanation,encoding='utf-8')
    reg=run.read(run.ROOT/'CURRENT_RESEARCH.json');assert reg['run_id']=='quality_readout_v15_3k'
    reg['active_experiment'].update(status='complete',phase='all_efficacy_and_prefix_timing_verified',results_path=str(final),prefix_results_path=str(PREFIX/'analysis.json'),mainline_unchanged=True)
    run.write(run.ROOT/'CURRENT_RESEARCH.json',reg)
    run.write(OUT/'super_all_completion.json',dict(complete=True,at_utc=run.now(),mainline_unchanged=True,
        efficacy_completion_sha256=run.sha(OUT/'all_completion.json'),prefix_completion_sha256=run.sha(PREFIX/'completion.json'),
        final_readout_sha256=run.sha(final),watch_source_sha256=run.sha(__file__)))
    status('complete',results_path=str(final))

if __name__=='__main__':
    try:main()
    except BaseException as error:status('failed',error=str(error));raise
