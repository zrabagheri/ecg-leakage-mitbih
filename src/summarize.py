"""Collect results, add record-level bootstrap CIs on DS2, write tables (CSV) used in the paper."""
import json, glob, numpy as np, pandas as pd
from pipeline import load, cluster_bootstrap, metrics
_, yte, gte = load('DS2', 'det'); _, yre, gre = load('DS2', 'ref')
rows = []
for f in sorted(glob.glob('../results/res_inter_s*.json')):
    seed = int(f.split('_s')[1].split('_')[0])
    for model, r in json.load(open(f)).items():
        for cond in ['C0', 'L5', 'L4', 'L3', 'SMOTE_clean', 'L2']:
            if cond not in r: continue
            y, g = (yre, gre) if cond == 'L5' else (yte, gte)
            row = dict(model=model, seed=seed, cond=cond, **r[cond])
            if seed == 0 and cond + '_pred' in r:
                ci = cluster_bootstrap(y, np.array(r[cond + '_pred']), g, B=1000, seed=0)
                for k in ('acc', 'macroF1'): row[k + '_lo'], row[k + '_hi'] = ci[k]
            rows.append(row)
for f in sorted(glob.glob('../results/res_intra_s*.json')):
    seed = int(f.split('_s')[1].split('_')[0])
    for model, r in json.load(open(f)).items():
        for cond in ['L1', 'ALL']:
            rows.append(dict(model=model, seed=seed, cond=cond, **r[cond]))
df = pd.DataFrame(rows); df.to_csv('../results/results_long.csv', index=False)
order = ['C0', 'L5', 'L4', 'L3', 'SMOTE_clean', 'L2', 'L1', 'ALL']
piv = df.groupby(['cond', 'model'])[['acc', 'macroF1', 'bacc', 'Se_S', 'PP_S', 'Se_V', 'PP_V']].mean().reset_index()
piv['cond'] = pd.Categorical(piv['cond'], order); piv = piv.sort_values(['cond', 'model'])
piv.to_csv('../results/results_mean.csv', index=False)
pd.set_option('display.width', 200); print(piv.round(3).to_string(index=False))
maj = (yte == 0).mean(); print('majority-class accuracy on DS2 (detected beats): %.4f' % maj)
