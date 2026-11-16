"""Chinese presentation of frozen V22 results;no numerical/model changes."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

root=Path(__file__).resolve().parents[1];out=root/'runs/structure_readout_v22'
r=json.loads((out/'results.json').read_text(encoding='utf-8'));assert r['complete']
font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc');plt.rcParams['font.family']=font.get_name();plt.rcParams['axes.unicode_minus']=False
names=['original','capacity','pooled32','ordered32','pooled64','ordered64'];x=np.arange(6)
fig,axes=plt.subplots(1,2,figsize=(13.4,4.8))
for offset,co,color,label in [(-.18,'train','#2878B5','训练集 2973'),(.18,'new_development','#DD8452','新开发集 300')]:
    y=[r['cohorts'][co]['methods'][a]['accuracy_percent'] for a in names];bars=axes[0].bar(x+offset,y,.36,color=color,label=label)
    axes[0].bar_label(bars,fmt='%.2f',padding=2,fontsize=7)
axes[0].set_xticks(x,['原版\n2.2万','旧扩容\n6.7万','池化\n4.1万','保留序列\n4.1万','池化\n13.4万','保留序列\n13.4万'],fontsize=8)
axes[0].set_ylabel('准确率（%）');axes[0].set_ylim(0,100);axes[0].legend(fontsize=8);axes[0].set_title('容量增加能改善训练拟合，开发集收益未建立')
c=r['cohorts']['new_development']['comparisons']
for j,key in enumerate(['structure_small','structure_large','net_small']):
    v=c[key];lo,hi=v['ci_pp'];axes[1].plot([lo,hi],[j,j],color='#2878B5',linewidth=2);axes[1].plot(v['delta_pp'],j,'o',color='#2878B5')
    axes[1].annotate(f'{v["delta_pp"]:+.2f} [{lo:+.2f}, {hi:+.2f}]',(v['delta_pp'],j),xytext=(0,11),textcoords='offset points',ha='center',fontsize=8)
axes[1].set_yticks(range(3),['小容量：保留－池化','大容量：保留－池化','小容量保留－原版'],fontsize=9)
axes[1].axvline(0,color='gray');axes[1].set_ylim(-.45,2.6);axes[1].invert_yaxis();axes[1].set_xlim(-2.5,2.7)
axes[1].set_xlabel('新开发集准确率差（百分点），校正 98.75% 区间');axes[1].set_title('保留结构未达到登记的净收益门槛')
fig.suptitle('V22 自有完整候选：固定数据、坐标和训练步数，三个种子均值；复用开发集诊断',fontsize=12)
fig.tight_layout();fig.savefig(out/'readout_zh.png',dpi=180);plt.close(fig)
print(out/'readout_zh.png')
