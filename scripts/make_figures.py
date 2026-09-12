"""Regenerate README figures: pip install matplotlib; python scripts/make_figures.py."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'assets'
OUT.mkdir(exist_ok=True)
M = json.loads((ROOT / 'results/metrics.json').read_text())['metrics']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                     'svg.fonttype': 'none', 'savefig.facecolor': '#ffffff'})
NAVY, TEAL, BLUE, GRAY = '#142d4e', '#008477', '#426db3', '#64748b'

def save(fig, name):
    for ext in ('svg', 'png'):
        fig.savefig(OUT / f'{name}.{ext}', dpi=180, bbox_inches='tight', pad_inches=.25)
    plt.close(fig)

def box(ax, x, y, w, h, title, body, color=BLUE):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.02,rounding_size=0.13',
                              facecolor='#f4f7fb',edgecolor=color,linewidth=1.5))
    ax.text(x+w/2,y+h*.69,title,ha='center',va='center',weight='bold',color=NAVY,fontsize=12)
    ax.text(x+w/2,y+h*.32,body,ha='center',va='center',color=GRAY,fontsize=10.5,linespacing=1.45)

def arrow(ax, a, b):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=15,color=GRAY,linewidth=1.4))

fig, ax = plt.subplots(figsize=(13,7.3))
ax.set(xlim=(0,13),ylim=(0,7.3)); ax.axis('off')
ax.text(.1,7.0,'Blood-cell VLM adaptation & selective inference',fontsize=21,weight='bold',color=NAVY)
ax.text(.1,6.57,'BloodMNIST · 8 classes · Qwen3.5-4B',fontsize=12,color=GRAY)
ax.text(.1,5.99,'01  DOMAIN ADAPTATION',fontsize=10,weight='bold',color=TEAL)
box(ax,.1,4.55,3.1,1.15,'4,000 training images','Class-balanced subset\nAnswer-only supervision')
box(ax,4.0,4.55,4.2,1.15,'Language-side LoRA','q / v projections · rank 4\n458,752 trainable parameters')
box(ax,9.0,4.55,3.7,1.15,'Adapted Qwen3.5-4B','Vision encoder frozen\nDevelopment-selected checkpoint',TEAL)
arrow(ax,(3.25,5.1),(3.95,5.1)); arrow(ax,(8.25,5.1),(8.95,5.1))
ax.text(.1,3.98,'02  CONFIDENCE ROUTING',fontsize=10,weight='bold',color=TEAL)
box(ax,.1,1.9,1.55,1.15,'Image','224 × 224')
box(ax,2.1,1.9,3.1,1.15,'Frozen WR50','ImageNet features\nTrained linear classifier')
box(ax,5.75,1.9,2.5,1.15,'Confidence gate','Max class probability\nτ ≈ 0.9604')
box(ax,9.2,2.8,3.5,1.0,'Keep CNN prediction','86.79% of test images',GRAY)
box(ax,9.2,1.05,3.5,1.0,'Use adapted Qwen','13.21% of test images',TEAL)
arrow(ax,(1.7,2.48),(2.05,2.48)); arrow(ax,(5.25,2.48),(5.7,2.48))
arrow(ax,(8.3,2.62),(9.15,3.22)); arrow(ax,(8.3,2.25),(9.15,1.62))
ax.text(8.65,3.14,'≥ τ',fontsize=10,color=GRAY,ha='center')
ax.text(8.65,1.65,'< τ',fontsize=10,color=GRAY,ha='center')
ax.text(.1,.52,'Policy selected on development predictions; evaluated by offline test-prediction replay.',color=GRAY,fontsize=10.5)
ax.text(.1,.15,'Each branch returns one of 8 cell classes. Routing call rate does not establish end-to-end speedup.',color=GRAY,fontsize=10.5)
save(fig,'method-overview')

keys=['base_qwen','qwen_lora_sft','wr50_linear','development_frozen_routing']
labels=['Qwen base','Qwen + LoRA','Frozen WR50\n+ linear head','CNN–VLM routing\n(offline replay)']
fig, (ax, ax2) = plt.subplots(1,2,figsize=(13,5.5),gridspec_kw={'width_ratios':[2.3,1]})
fig.subplots_adjust(top=.76,bottom=.24,wspace=.26)
fig.text(.06,.94,'Domain adaptation gain, with selective VLM calls',fontsize=20,weight='bold',color=NAVY)
fig.text(.06,.88,'BloodMNIST official test · n = 3,421 · development-selected checkpoint and routing policy',fontsize=11,color=GRAY)
x=np.arange(4); width=.34
for offset, metric, color, label in [(-width/2,'accuracy',BLUE,'Accuracy'),(width/2,'macro_f1',TEAL,'Macro-F1')]:
    vals=[100*M[k][metric] for k in keys]
    bars=ax.bar(x+offset,vals,width,color=color,label=label)
    ax.bar_label(bars,labels=[f'{v:.2f}' for v in vals],padding=4,fontsize=9)
ax.set_xticks(x,labels,fontsize=10); ax.set_ylim(0,110); ax.set_yticks(range(0,101,20)); ax.set_ylabel('Score (%)',color=GRAY)
ax.legend(loc='upper left',bbox_to_anchor=(0,1.16),ncol=2,frameon=False,fontsize=10)
rates=[100,100,0,100*M['development_frozen_routing']['qwen_call_rate']]
bars=ax2.barh(np.arange(4),rates,color=[BLUE,TEAL,GRAY,TEAL],height=.55)
ax2.set_yticks(np.arange(4),['Base','LoRA','CNN','Routing'],fontsize=10); ax2.invert_yaxis(); ax2.set_xlim(0,122)
ax2.set_xticks([0,50,100]); ax2.set_xlabel('VLM call rate (%)',color=GRAY)
ax2.bar_label(bars,labels=[f'{v:.2f}%' for v in rates],padding=4,fontsize=10)
for a in (ax,ax2):
    a.spines[['top','right']].set_visible(False)
    a.spines[['left','bottom']].set_color('#d2dae4')
    a.tick_params(colors=GRAY,length=0,pad=7)
    a.set_axisbelow(True)
ax.yaxis.grid(True,color='#e9edf2')
fig.text(.06,.105,'SFT: +36.16 pp accuracy vs. base     |     Routing: 452 / 3,421 VLM calls',color=NAVY,weight='bold',fontsize=11)
fig.text(.06,.048,'Routing is offline replay; call rate is not measured latency reduction. Full CNN fine-tuning results: docs/EXPERIMENTS.md.',color=GRAY,fontsize=9.5)
save(fig,'test-results')
