"""Paired record-level bootstrap for the effect of each leak (difference from its clean reference)."""
import json, glob, numpy as np, pandas as pd
from pipeline import load
_, yte, gte = load('DS2', 'det'); _, yre, gre = load('DS2', 'ref')
recs = np.unique(gte)

def conf_by_rec(y, yh, g):
    C = np.zeros((len(recs), 4, 4))
    for i, r in enumerate(recs):
        m = g == r; np.add.at(C[i], (y[m], yh[m]), 1)
    return C

def mets(C):
    tp = np.diagonal(C, axis1=-2, axis2=-1); P = C.sum(-2); T = C.sum(-1)
    se = np.divide(tp, T, out=np.zeros_like(tp), where=T > 0); pp = np.divide(tp, P, out=np.zeros_like(tp), where=P > 0)
    f1 = np.divide(2 * se * pp, se + pp, out=np.zeros_like(tp), where=(se + pp) > 0)
    return dict(acc=tp.sum(-1) / C.sum((-1, -2)), macroF1=f1.mean(-1), bacc=se.mean(-1), Se_S=se[..., 1], Se_V=se[..., 2])

ref_of = {'L5': 'C0', 'L4': 'C0', 'L3': 'C0', 'L2': 'SMOTE_clean', 'SMOTE_clean': 'C0'}
rng = np.random.default_rng(0); idx = rng.integers(0, len(recs), (2000, len(recs)))
rows = []
for f in sorted(glob.glob('../results/res_inter_s0_*.json')):
    for model, r in json.load(open(f)).items():
        Cs = {}
        for c in ['C0', 'L5', 'L4', 'L3', 'SMOTE_clean', 'L2']:
            y, g = (yre, gre) if c == 'L5' else (yte, gte)
            Cs[c] = conf_by_rec(y, np.array(r[c + '_pred']), g)
        for c, base in ref_of.items():
            A = mets(Cs[c][idx].sum(1)); B = mets(Cs[base][idx].sum(1)); pa = mets(Cs[c].sum(0)); pb = mets(Cs[base].sum(0))
            for k in ['acc', 'macroF1', 'bacc', 'Se_S', 'Se_V']:
                d = A[k] - B[k]
                rows.append(dict(model=model, cond=c, vs=base, metric=k, delta=pa[k] - pb[k],
                                 lo=np.percentile(d, 2.5), hi=np.percentile(d, 97.5)))
df = pd.DataFrame(rows); df.to_csv('../results/paired_deltas.csv', index=False)
pd.set_option('display.width', 200)
print(df[df.metric.isin(['acc', 'macroF1', 'bacc'])].round(3).to_string(index=False))
