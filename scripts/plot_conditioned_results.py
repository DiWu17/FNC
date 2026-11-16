"""Export matched-score, learning-curve and deterministic error examples."""
import argparse
import json
from pathlib import Path
import textwrap
import os

os.environ.setdefault('MPLCONFIGDIR',str(Path('runs/conditioned_v3/matplotlib').resolve()))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image,ImageDraw,ImageFont

from run_target_token_experiment import read_json,read_jsonl,key


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('runs/conditioned_v3'))
    args=parser.parse_args();out=args.output;dest=out/'evaluation';comparison=read_json(dest/'comparison.json')
    methods=comparison['methods']
    names=['legacy_first','static_96','attention_only','no_image','no_mass','no_query','no_attention','full','full_generated_action_policy',
        'full_fine_centers','full_coarse_centers','qwen_native_strict','qwen_native_completed_lax','ui_tars']
    labels=['Old 96 heads: target first','Static 96: same training data','New attention only','No image content','No attention mass','No target hidden','No explicit attention features',
        'Full: point proposal','Full: obey generated CLICK/WAIT','Full: fine centers','Full: coarse centers','Qwen native: strict',
        'Qwen native: completed lax','UI-TARS-1.5-7B']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(10,7.8),layout='constrained')
    values=np.array([methods[n]['accuracy']*100 for n in names])
    intervals=np.array([methods[n]['wilson_95ci'] for n in names])*100
    colors=['#de673c' if n=='full' else '#edaa82' if n=='full_generated_action_policy' else '#356880' for n in names]
    y=np.arange(len(names));ax.barh(y,values,color=colors,height=.65)
    ax.errorbar(values,y,xerr=np.vstack((values-intervals[:,0],intervals[:,1]-values)),fmt='none',ecolor='#182f40',capsize=3,lw=1)
    for i,n in enumerate(names):ax.text(intervals[i,1]+1,i,f"{methods[n]['hits']}/192",va='center',fontsize=9)
    ax.set(yticks=y,yticklabels=labels,xlabel='Point inside annotated target (%)',
        title='Same 192 fresh screenshots | frozen backbones | 262,144-pixel budget',xlim=(0,max(70,float(intervals.max()+12))));ax.invert_yaxis()
    ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    fig.savefig(dest/'matched_scores.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.5),layout='constrained')
    for name in ('full','attention_only','no_image','no_mass','no_query','no_attention','static_96'):
        history=read_json(out/'models'/f'{name}_history.json');items=history['history']
        axes[0].plot([r['epoch'] for r in items],[r['val']['accuracy']*100 for r in items],label=name)
        axes[1].plot([r['epoch'] for r in items[1:]],[r['train_loss'] for r in items[1:]],label=name)
        chosen=items[history['best_epoch']];axes[0].scatter([chosen['epoch']],[chosen['val']['accuracy']*100],s=20)
    axes[0].set(xlabel='Epoch',ylabel='Validation point hits (%)',title='Checkpoint selection uses validation only')
    axes[1].set(xlabel='Epoch',ylabel='Training objective',title='Training loss (static uses coarse CE only)')
    for ax in axes:ax.grid(alpha=.15)
    axes[0].legend(fontsize=8);fig.savefig(dest/'learning_curves.png',dpi=180);plt.close(fig)
    rows=read_jsonl(dest/'all_predictions.jsonl');pred={(r['sample_id'],r['method']):r for r in rows}
    cases=[c for c in read_jsonl(out/'manifest.jsonl') if c['experiment_split']=='test']
    wins=[c for c in cases if pred[c['sample_id'],'full']['hit'] and not pred[c['sample_id'],'legacy_first']['hit']]
    misses=[c for c in cases if not pred[c['sample_id'],'full']['hit']]
    for group_name,group in (('wins_over_old',wins),('remaining_misses',misses)):
        selected=sorted(group,key=lambda c:key(c['sample_id']))[:6]
        if not selected:continue
        panels=[]
        try:font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16)
        except OSError:font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16)
        for c in selected:
            with Image.open(c['image']) as opened:image=opened.convert('RGB')
            image.thumbnail((690,440));panel=Image.new('RGB',(720,600),'#f5f7fa');panel.paste(image,((720-image.width)//2,10))
            draw=ImageDraw.Draw(panel);ox=(720-image.width)//2;oy=10;sx=image.width/c['image_size'][0];sy=image.height/c['image_size'][1]
            geometry=c['target_geometry']
            if geometry['type']=='bbox':
                for box in geometry['boxes']:
                    draw.rectangle([ox+box[0]*image.width,oy+box[1]*image.height,ox+box[2]*image.width,oy+box[3]*image.height],outline='#18a45b',width=3)
            else:
                for poly in geometry['polygons']:draw.line([(ox+x*image.width,oy+y*image.height) for x,y in poly]+[(ox+poly[0][0]*image.width,oy+poly[0][1]*image.height)],fill='#18a45b',width=3)
            for method,color in (('full','#e84835'),('qwen_native_completed_lax','#3270df'),('ui_tars','#e6ac12')):
                point=pred[c['sample_id'],method]['point']
                if point:
                    x,y=ox+point[0]*sx,oy+point[1]*sy
                    draw.ellipse([x-7,y-7,x+7,y+7],outline=color,width=3);draw.line([x-10,y,x+10,y],fill=color,width=2)
            header=f"{c['dataset']} | full={int(pred[c['sample_id'],'full']['hit'])} old={int(pred[c['sample_id'],'legacy_first']['hit'])} Qwen={int(pred[c['sample_id'],'qwen_native_completed_lax']['hit'])} UI={int(pred[c['sample_id'],'ui_tars']['hit'])}"
            draw.text((12,455),header,font=font,fill='#172b3a')
            for i,line in enumerate(textwrap.wrap(c['instruction'],width=75)[:3]):draw.text((12,480+i*22),line,font=font,fill='#172b3a')
            target=str(pred[c['sample_id'],'full'].get('generated_target_name'))
            for i,line in enumerate(textwrap.wrap('Generated target: '+target,width=75)[:2]):draw.text((12,550+i*22),line,font=font,fill='#172b3a')
            panels.append(panel)
        canvas=Image.new('RGB',(1440,60+600*((len(panels)+1)//2)),'white');draw=ImageDraw.Draw(canvas)
        draw.text((15,12),'GT green | Full red | Qwen native blue | UI-TARS yellow | deterministic examples',font=font,fill='#172b3a')
        for i,panel in enumerate(panels):canvas.paste(panel,((i%2)*720,60+(i//2)*600))
        canvas.save(dest/f'{group_name}.png')
    print(json.dumps({'complete':True,'output':str(dest),'test_examples_selection':'first six by sample hash within outcome group'}),flush=True)


if __name__=='__main__':main()
