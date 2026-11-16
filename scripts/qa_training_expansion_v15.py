"""Check completed data manifests and render fixed-hash label QA samples."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import textwrap

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from clickagent.training_expansion_data import NearIndex,digest
RUN=ROOT/'runs/dataset_expansion_v15';DATA=ROOT/'data/expansion_v15'


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def main():
    result=read(RUN/'results.json');assert result['complete']
    contents={}
    for name,info in result['manifests'].items():
        path=Path(info['path']);assert digest(path)==info['sha256']
        rows=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        assert len(rows)==info['rows']==len({r['sample_id'] for r in rows})
        for row in rows:
            w,h=row['image_size'];assert w>0 and h>0 and Path(row['image']).is_file()
            for a,b,c,d in row['target_geometry']['boxes']:assert 0<=a<c<=1 and 0<=b<d<=1
        contents[name]=rows
    images=read(RUN/'image_index.json')['images'];tree=NearIndex();pixels=set()
    for image in images:
        assert image['pixel_sha256'] not in pixels
        assert tree.find(int(image['dhash256'],16)) is None
        tree.add(int(image['dhash256'],16),image['key']);pixels.add(image['pixel_sha256'])
    assert result['new_unique_images']==len(images)
    train=contents['train.jsonl'];val=contents['validation_reserve.jsonl']
    assert not {r['pixel_sha256'] for r in train}&{r['pixel_sha256'] for r in val}
    overlap={(r['dataset'],r['source_group']) for r in train}&{(r['dataset'],r['source_group']) for r in val}
    assert not overlap
    assert len(contents['combined_train.jsonl'])==873+len(train)
    assert len({r['pixel_sha256'] for r in contents['combined_train.jsonl']})==result['combined_training_images']
    reference=read(ROOT/'data/conditioned_v3/public_image_index.json')['images']
    forbidden={r['pixel_sha256'] for r in reference};protected=NearIndex()
    for row in reference:protected.add(int(row['dhash256'],16),'public')
    for split in ('train','val'):
        for line in (ROOT/f'runs/multiregion_v13/data/{split}.jsonl').read_text(encoding='utf-8').splitlines():
            row=json.loads(line);forbidden.add(row['pixel_sha256']);protected.add(int(row['dhash256'],16),'historical')
    assert all(r['pixel_sha256'] not in forbidden and protected.find(int(r['dhash256'],16)) is None for r in images)
    picked=[]
    for dataset in ('GroundCUA','ShowUI-web','AMEX'):
        candidates=sorted([r for r in train if r['dataset']==dataset],key=lambda r:hashlib.sha256(('qa:'+r['sample_id']).encode()).hexdigest())
        seen=set()
        for row in candidates:
            if row['pixel_sha256'] in seen:continue
            picked.append(row);seen.add(row['pixel_sha256'])
            if len(seen)==2:break
    assert len(picked)==6
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib_expansion_v15'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from PIL import Image
    fig,axes=plt.subplots(3,4,figsize=(20,14),layout='constrained')
    for i,row in enumerate(picked):
        with Image.open(row['image']) as image:im=image.convert('RGB')
        w,h=im.size
        full=axes[i//2,(i%2)*2];zoom=axes[i//2,(i%2)*2+1]
        full.imshow(im)
        for box in row['target_geometry']['boxes']:
            a,b,c,d=box;full.add_patch(Rectangle((a*w,b*h),(c-a)*w,(d-b)*h,fill=False,edgecolor='lime',linewidth=1.5))
        a,b,c,d=row['target_geometry']['boxes'][0];cx=(a+c)*w/2;cy=(b+d)*h/2
        cropw=max(240,(c-a)*w*2);croph=max(180,(d-b)*h*3)
        left=max(0,int(cx-cropw/2));top=max(0,int(cy-croph/2))
        right=min(w,int(cx+cropw/2));bottom=min(h,int(cy+croph/2))
        zoom.imshow(im.crop((left,top,right,bottom)))
        zoom.add_patch(Rectangle((a*w-left,b*h-top),(c-a)*w,(d-b)*h,fill=False,edgecolor='lime',linewidth=2))
        full.set_title(row['dataset']+' / '+row['source_group'],fontsize=10)
        zoom.set_title(textwrap.fill(row['instruction'][:180],40),fontsize=9)
        full.axis('off');zoom.axis('off')
    fig.suptitle('V15 fixed-hash training-data QA: full screen and target context',fontsize=16)
    fig.savefig(RUN/'qa_samples.png',dpi=130);plt.close(fig)
    (RUN/'qa_samples.json').write_text(json.dumps(picked,ensure_ascii=False,indent=2),encoding='utf-8')
    archive=RUN/'source_archive';archive.mkdir(exist_ok=True)
    sources=[ROOT/'scripts/acquire_training_expansion_v15.py',ROOT/'scripts/normalize_training_expansion_v15.py',ROOT/'scripts/index_training_expansion_v15.py',Path(__file__),ROOT/'src/clickagent/training_expansion_data.py',ROOT/'tests/test_training_expansion_data.py']
    for path in sources:shutil.copyfile(path,archive/path.name)
    receipt=dict(complete=True,manifests_checked=len(contents),images_checked=len(images),same_source_group_split_overlap=len(overlap),
        new_import_near_duplicate_pairs_at_radius4=0,label_coordinate_checks_passed=True,qa_sample_count=6,
        protected_public_and_historical_screenshots_excluded=True,
        caveat='Only six visual label examples are manually reviewed; this does not certify all source annotations.',
        source_sha256={str(p.relative_to(ROOT)):digest(p) for p in sources})
    (RUN/'qa_checks.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(receipt))


if __name__=='__main__':main()
