import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt, json
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150})
C='#2a6f97'; C2='#e07a5f'
import os; os.makedirs('results', exist_ok=True)
df=pd.read_csv('train.csv'); te=pd.read_csv('test.csv')
out={}
out['shape_train']=df.shape; out['shape_test']=te.shape
out['missing']=int(df.isna().sum().sum()); out['dups_id']=int(df.id.duplicated().sum())
out['target']=df.accident_risk.describe().round(4).to_dict()
fig,ax=plt.subplots(figsize=(6,3.2)); ax.hist(df.accident_risk,bins=50,color=C)
for b in (0.25,0.5,0.75): ax.axvline(b,color='#888',ls='--',lw=.8)
ax.set_xlabel('accident_risk'); ax.set_ylabel('rows'); ax.set_title('Target distribution (dashed = risk-level boundaries)')
fig.tight_layout(); fig.savefig('results/eda_target_distribution.png'); plt.close()
g=df.groupby(pd.cut(df.curvature,20)).accident_risk.mean()
fig,ax=plt.subplots(figsize=(6,3.2)); ax.plot([i.mid for i in g.index],g.values,'o-',color=C)
ax.set_xlabel('curvature'); ax.set_ylabel('mean accident_risk'); ax.set_title('Risk rises with road curvature')
fig.tight_layout(); fig.savefig('results/eda_curvature.png'); plt.close()
cats=['lighting','weather','road_type','time_of_day','speed_limit','num_lanes','num_reported_accidents','road_signs_present','public_road','holiday','school_season']
fig,axes=plt.subplots(3,4,figsize=(11,7.5)); means={}
for ax,c in zip(axes.ravel(),cats):
    s=df.groupby(c).accident_risk.mean(); means[c]={str(k):round(float(v),4) for k,v in s.items()}
    ax.bar([str(k) for k in s.index],s.values,color=C); ax.set_title(c,fontsize=10); ax.set_ylim(0,0.6); ax.tick_params(axis='x',labelsize=8,rotation=30)
axes.ravel()[-1].axis('off'); fig.suptitle('Mean accident_risk by feature value'); fig.tight_layout(); fig.savefig('results/eda_feature_means.png'); plt.close()
out['means']=means
out['counts']={c:{str(k):int(v) for k,v in df[c].value_counts().sort_index().items()} for c in cats}
num=['num_lanes','curvature','speed_limit','num_reported_accidents']
out['corr']=df[num+['accident_risk']].corr()['accident_risk'].round(4).to_dict()
json.dump(out,open('results/eda.json','w'),indent=1,default=str); print(json.dumps(out,default=str)[:3000])
