"""Regenerate standalone publication figures from reviewed aggregate evidence."""
import hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
E=ROOT/'docs/forecasting/evidence/final_study'
O=ROOT/'docs/forecasting/figures';O.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
COLORS={'stgcn_reference':'#60656b','stgcn_search_005':'#276a9f'}
LABELS={'stgcn_reference':'Compact reference','stgcn_search_005':'Tuned STGCN'}
sources={}
def read(path):
 sources[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
 return json.loads(path.read_text())
s=read(E/'final_pems/summary.json')
fig,axes=plt.subplots(1,3,figsize=(14,4.5))
for seed,marker in zip([17,42,73],['o','s','^']):
 vals=[next(r['test_mae_mph'] for r in s['rows'] if r['configuration']==t and r['seed']==seed) for t in COLORS]
 axes[0].plot([0,1],vals,marker=marker,color='#276a9f',alpha=.8,label=f'Seed {seed}')
axes[0].set(xticks=[0,1],xticklabels=['Reference','Tuned'],ylabel='Test MAE (mph)',title='Matched seeds; focused y-scale',ylim=(1.59,1.705));axes[0].legend(frameon=False)
for trial,color in COLORS.items():
 reports=[read(E/'pems_test'/f'{trial}_seed{x}.json') for x in [17,42,73]]
 y=np.array([[r['test']['all']['horizons'][str(h)]['mae_mph'] for h in range(5,61,5)] for r in reports])
 axes[1].plot(range(5,61,5),y.mean(0),color=color,label=LABELS[trial],marker='o' if trial=='stgcn_reference' else 's',markersize=3)
 axes[1].fill_between(range(5,61,5),y.mean(0)-y.std(0,ddof=1),y.mean(0)+y.std(0,ddof=1),color=color,alpha=.15)
axes[1].set(xlabel='Forecast horizon (minutes)',ylabel='Test MAE (mph)',title='Mean ± sample SD across 3 seeds',ylim=(0,2.3));axes[1].legend(frameon=False)
labels=['Slow\n<30 mph','Moderate\n30–55 mph','Free flow\n≥55 mph']
for i,(trial,color) in enumerate(COLORS.items()):
 vals=[s['groups'][trial][g]['overall']['mae_mph'] for g in ['slow','moderate','free_flow']]
 axes[2].bar(np.arange(3)+(i-.5)*.36,vals,.36,color=color,label=LABELS[trial])
axes[2].set(xticks=range(3),xticklabels=labels,ylabel='Test MAE (mph)',title='Target-speed groups',ylim=(0,8))
fig.suptitle('PEMS-BAY: locked 12-to-12 speed forecasting evaluation',fontsize=14)
fig.text(.02,.015,'Chronological held-out tail; 325 sensors. Missing-history (>25%) group: no qualifying samples. No test-based retuning.',fontsize=9)
fig.tight_layout(rect=(0,.07,1,.92))
for ext in ['png','pdf']:fig.savefig(O/f'pems_locked_test.{ext}',dpi=180)
plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(12,4.7))
for trial,color in COLORS.items():
 for seed,style in zip([17,42,73],['-','--',':']):
  history=read(E/'pems_training'/trial/str(seed)/'epochs.json')
  axes[0].plot([h['epoch'] for h in history],[h['validation']['all']['overall']['mae_mph'] for h in history],color=color,linestyle=style,label=LABELS[trial]+f' / {seed}')
axes[0].set(xlabel='Epoch',ylabel='Validation MAE (mph)',title='PEMS-BAY validation curves',ylim=(1.5,2.25));axes[0].legend(frameon=False,fontsize=8)
m=read(E/'matched_core_validation/summary.json');g=m['groups']
axes[1].barh(range(len(g)),[x['mean_mae_mph'] for x in g],xerr=[x['sample_sd_mae_mph'] for x in g],color='#60656b',capsize=3)
axes[1].set(yticks=range(len(g)),yticklabels=[x['trial'].replace('stgcn_','').replace('_',' ') for x in g],xlabel='Validation MAE (mph)',title='METR-LA matched cohort: exploratory',xlim=(0,3.35));axes[1].invert_yaxis()
for i,x in enumerate(g):axes[1].text(x['mean_mae_mph']+.04,i,f"{x['mean_mae_mph']:.3f}",va='center',fontsize=9)
fig.text(.02,.012,'METR-LA error bars: sample SD over 3 seeds. Residual heads worsened both STGCN configurations; search budgets differ across model families.',fontsize=9)
fig.tight_layout(rect=(0,.05,1,1))
for ext in ['png','pdf']:fig.savefig(O/f'validation_and_ablation.{ext}',dpi=180)
(O/'final_figure_sources.json').write_text(json.dumps({'source_sha256':sources,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2)+'\n')
print('Generated two source-linked PNG/PDF figures')
