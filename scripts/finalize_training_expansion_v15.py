"""Seal the completed data-only import after programmatic and visual QA."""
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from normalize_training_expansion_v15 import DATA,RUN,read,write,legacy_files


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    result=read(RUN/'results.json');qa=read(RUN/'qa_checks.json');visual=read(RUN/'manual_visual_qa.json')
    tests=read(RUN/'unit_tests.json');download=read(RUN/'download_complete.json')
    assert result['complete'] and qa['complete'] and visual['complete'] and tests['passed']==11
    assert download['complete'] and download['n']==4000 and not download['errors']
    assert sha(RUN/'qa_samples.png')==visual['figure_sha256']
    compatibility=read(RUN/'windows_path_compatibility.json')
    for name,expected in compatibility['corrected_source_identity'].items():assert sha(ROOT/name)==expected
    assert compatibility['selection_sha256']==result['selection_sha256']==sha(RUN/'selection.json')
    assert legacy_files()==result['legacy_completion_hashes']
    legacy_v14=ROOT/'runs/evidence_retention_v14'
    v14_manifest=read(legacy_v14/'report_completion.json')
    for name,expected in v14_manifest['files'].items():assert sha(legacy_v14/name)==expected
    report=RUN/'report.html';text=report.read_text(encoding='utf-8')
    addition='<section id="final-data-checks"><h2>最终核对</h2><p>4000张原图全部通过发布方哈希校验与解码；11项适配测试通过。排除350张重复或近似图，其中198张AMEX与已有公共评测图匹配。程序核对6份清单、3650张保留图及全部输出坐标；另人工查看6个固定抽样例，未发现明显坐标错位。这不等于所有来源标注均已人工认证。</p><p><a href="qa_checks.json">程序核对</a> · <a href="manual_visual_qa.json">人工抽样记录</a> · <a href="unit_tests.json">11项测试</a> · <a href="windows_path_compatibility.json">Windows目录名映射</a> · <a href="../../data/expansion_v15/registry.json">数据来源登记</a></p><p>V13/V14封存产物核验一致。此次只接入数据，没有开始特征提取、重训或效果评测。</p></section>'
    assert 'id="final-data-checks"' not in text
    report.write_text(text.replace('</body>',addition+'</body>'),encoding='utf-8')
    class Links(HTMLParser):
        def __init__(self):super().__init__();self.paths=[]
        def handle_starttag(self,tag,attrs):
            a=dict(attrs)
            if tag=='a' and 'href' in a and not a['href'].startswith('http'):self.paths.append(a['href'])
            if tag=='img' and 'src' in a:self.paths.append(a['src'])
    parser=Links();parser.feed(report.read_text(encoding='utf-8'))
    assert all((RUN/path).is_file() for path in parser.paths)
    archive=RUN/'source_archive';shutil.copyfile(Path(__file__),archive/Path(__file__).name)
    artifacts=[RUN/name for name in ('results.json','report.html','selection.json','download_complete.json','streaming_index_complete.json','qa_checks.json','qa_samples.json','qa_samples.png','manual_visual_qa.json','unit_tests.json','windows_path_compatibility.json')]
    artifacts += [DATA/'registry.json']+[Path(info['path']) for info in result['manifests'].values()]
    completion=dict(complete=True,scope='First-batch public training-data import only; not full source copies and not a training experiment.',
        images_downloaded=4000,new_training_images=result['new_training_images'],new_reserved_images=result['new_reserved_images'],
        new_training_queries=result['new_training_queries'],combined_training_images=result['combined_training_images'],
        combined_training_queries=result['combined_training_queries'],test_count=11,report_local_links_checked=len(parser.paths),
        legacy_v13_artifacts_unchanged=True,legacy_v14_report_manifest_unchanged=True,
        artifact_sha256={str(path.relative_to(ROOT)):sha(path) for path in artifacts},
        qwen_feature_extraction_started=False,model_training_started=False)
    write(RUN/'completion.json',completion)
    write(RUN/'work_state.json',dict(complete=True,phase='data_import_and_qa_complete',
        completion_sha256=sha(RUN/'completion.json'),qwen_feature_extraction_started=False,model_training_started=False,
        next_training_manifest=str((DATA/'manifests/combined_train.jsonl').resolve()),new_development_reserve=str((DATA/'manifests/validation_reserve.jsonl').resolve())))
    print(json.dumps({key:value for key,value in completion.items() if key!='artifact_sha256'},ensure_ascii=False))


if __name__=='__main__':main()
