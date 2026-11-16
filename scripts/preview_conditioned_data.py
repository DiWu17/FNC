"""Show deterministic training-only annotation examples before model fitting."""
import argparse
from collections import defaultdict
from pathlib import Path
import textwrap

from PIL import Image,ImageDraw,ImageFont

from run_target_token_experiment import read_jsonl,key


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=Path('runs/conditioned_v3'))
    args=p.parse_args();out=args.output;by_source=defaultdict(list)
    for c in read_jsonl(out/'manifest.jsonl'):
        if c['experiment_split']=='train':by_source[c['dataset']].append(c)
    selected=[]
    for source,group in sorted(by_source.items()):
        selected.extend(sorted(group,key=lambda c:key(c['sample_id']))[:2])
    try:font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16)
    except OSError:font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16)
    canvas=Image.new('RGB',(1480,70+460*((len(selected)+1)//2)),'#edf2f6');draw=ImageDraw.Draw(canvas)
    draw.text((15,10),'Training data QC: green annotation, full screenshot + target crop (labels for inspection only)',font=font,fill='#132c3c')
    for i,c in enumerate(selected):
        with Image.open(c['image']) as opened:image=opened.convert('RGB')
        w,h=image.size;x0,y0,x1,y1=c['target_geometry']['boxes'][0];bbox=[x0*w,y0*h,x1*w,y1*h]
        annotated=image.copy();paint=ImageDraw.Draw(annotated);paint.rectangle(bbox,outline='#10b66b',width=max(2,round(w/300)))
        full=annotated.copy();full.thumbnail((430,330))
        cx,cy=(bbox[0]+bbox[2])/2,(bbox[1]+bbox[3])/2
        cw,ch=max(120,(bbox[2]-bbox[0])*2),max(100,(bbox[3]-bbox[1])*3)
        crop=annotated.crop((max(0,int(cx-cw/2)),max(0,int(cy-ch/2)),min(w,int(cx+cw/2)),min(h,int(cy+ch/2))))
        crop.thumbnail((280,300))
        panel=Image.new('RGB',(730,450),'white');panel.paste(full,((430-full.width)//2,10));panel.paste(crop,(440,10))
        paint=ImageDraw.Draw(panel);paint.text((10,345),f"{c['dataset']} | {w}x{h} | bbox px={tuple(round(v) for v in bbox)}",font=font,fill='#132c3c')
        for j,line in enumerate(textwrap.wrap(c['instruction'],width=78)[:3]):paint.text((10,370+j*22),line,font=font,fill='#132c3c')
        canvas.paste(panel,((i%2)*740,70+(i//2)*460))
    dest=out/'training_data_qc.png';canvas.save(dest)
    print(str(dest),flush=True)


if __name__=='__main__':main()
