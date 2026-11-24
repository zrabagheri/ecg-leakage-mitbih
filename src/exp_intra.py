"""L1 (intra-patient random beat split) and ALL (every leak combined), pooled DS1+DS2 with reference peaks."""
import numpy as np, json, sys, time
from sklearn.model_selection import StratifiedKFold, train_test_split
from pipeline import *

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
models = sys.argv[2].split(',') if len(sys.argv) > 2 else ['LR', 'RF', 'MLP']
X1, y1, g1 = load('DS1', 'ref'); X2, y2, g2 = load('DS2', 'ref')
X = np.vstack([X1, X2]); y = np.concatenate([y1, y2]); g = np.concatenate([g1, g2])
out = {}
for name in models:
    t0 = time.time(); res = {}
    tr, te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=seed)
    # L1: clean in every respect except the beat-level random split
    cv = []
    fixed = None
    if seed > 0:
        import glob
        for f0 in glob.glob('../results/res_intra_s0_*.json'):
            d0 = json.load(open(f0))
            if name in d0: fixed = d0[name]['L1_chosen']
    for p in (GRIDS[name] if fixed is None else []):
        f = []
        for a, b in StratifiedKFold(4, shuffle=True, random_state=seed).split(X[tr], y[tr]):
            prep = Prep(name).fit(X[tr][a], y[tr][a])
            f.append(metrics(y[tr][b], fit_predict(name, p, seed, X[tr][a], y[tr][a], X[tr][b], prep))['macroF1'])
        cv.append(float(np.mean(f)))
    best = int(np.argmax(cv)) if fixed is None else fixed; p = GRIDS[name][best]
    prep = Prep(name).fit(X[tr], y[tr]); yh = fit_predict(name, p, seed, X[tr], y[tr], X[te], prep)
    res['L1'] = metrics(y[te], yh); res['L1_chosen'] = best; res['L1_pred'] = yh.tolist(); res['L1_te'] = te.tolist()
    res['L1_test_records_in_train'] = float(np.isin(np.unique(g[te]), np.unique(g[tr])).mean())
    # ALL: SMOTE on the pooled set, then random split, preprocessing on everything, tuning on the test set
    sc = StandardScaler().fit(X)
    Sx, Sy, _ = smote(sc.transform(X), y, [1, 2, 3], {1: 10000, 2: 10000, 3: 10000}, seed)
    XA = np.vstack([X, sc.inverse_transform(Sx)]); yA = np.concatenate([y, Sy])
    tr2, te2 = train_test_split(np.arange(len(yA)), test_size=0.2, stratify=yA, random_state=seed)
    prepA = Prep(name).fit(XA, yA); scores = []
    for q in GRIDS[name]:
        scores.append(metrics(yA[te2], fit_predict(name, q, seed, XA[tr2], yA[tr2], XA[te2], prepA)))
    res['ALL'] = max(scores, key=lambda d: d['macroF1'])
    res['secs'] = time.time() - t0; out[name] = res
    print(name, f"{res['secs']:.0f}s", {k: round(v['acc'], 4) for k, v in res.items() if isinstance(v, dict)}, flush=True)
json.dump(out, open(f'../results/res_intra_s{seed}_{"_".join(models)}.json', 'w'))
