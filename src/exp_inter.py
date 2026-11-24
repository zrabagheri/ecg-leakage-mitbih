"""Inter-patient experiments: clean baseline (C0) and leaks L2-L5, all trained on DS1, tested on DS2."""
import numpy as np, json, sys, time
from sklearn.model_selection import GroupKFold
from pipeline import *

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
models = sys.argv[2].split(',') if len(sys.argv) > 2 else ['LR', 'RF', 'MLP']
Xtr, ytr, gtr = load('DS1', 'ref')
Xte, yte, gte = load('DS2', 'det')           # realistic: automatic QRS detection at test time
Xre, yre, gre = load('DS2', 'ref')           # reference annotations at test time (L5)
out = {}
for name in models:
    t0 = time.time(); res = {}
    # 1) hyperparameter selection by record-grouped CV inside DS1 (clean)
    cv = []
    fixed = None
    if seed > 0:   # extra seeds reuse the hyperparameters selected (by grouped CV) with seed 0
        import glob
        for f0 in glob.glob('../results/res_inter_s0_*.json'):
            d0 = json.load(open(f0))
            if name in d0: fixed = d0[name]['chosen']
    for p in (GRIDS[name] if fixed is None else []):
        f = []
        for a, b in GroupKFold(n_splits=4).split(Xtr, ytr, gtr):
            prep = Prep(name).fit(Xtr[a], ytr[a])
            f.append(metrics(ytr[b], fit_predict(name, p, seed, Xtr[a], ytr[a], Xtr[b], prep))['macroF1'])
        cv.append(float(np.mean(f)))
    best = int(np.argmax(cv)) if fixed is None else fixed; res['cv_macroF1'] = cv; res['chosen'] = best
    # 2) every grid point trained on full DS1, evaluated on DS2 (needed for C0, L4, L5)
    prep = Prep(name).fit(Xtr, ytr); grid = []
    for p in GRIDS[name]:
        A, B, C = prep.transform(Xtr), prep.transform(Xte), prep.transform(Xre)
        yA = ytr
        if name == 'MLP': A, yA = oversample(A, ytr, seed)
        m = make_model(name, p, seed).fit(A, yA)
        grid.append(dict(det=m.predict(B), ref=m.predict(C)))
    test_f1 = [metrics(yte, g['det'])['macroF1'] for g in grid]
    res['C0'] = metrics(yte, grid[best]['det']); res['C0_pred'] = grid[best]['det'].tolist()
    res['L4'] = metrics(yte, grid[int(np.argmax(test_f1))]['det']); res['L4_chosen'] = int(np.argmax(test_f1)); res['L4_pred'] = grid[int(np.argmax(test_f1))]['det'].tolist()
    res['L5'] = metrics(yre, grid[best]['ref']); res['L5_pred'] = grid[best]['ref'].tolist()
    p = GRIDS[name][best]
    # 3) L3: scaler + supervised feature selection fitted on DS1 + DS2 (with DS2 labels)
    prepL = Prep(name).fit(np.vstack([Xtr, Xte]), np.concatenate([ytr, yte]))
    yh = fit_predict(name, p, seed, Xtr, ytr, Xte, prepL); res['L3'] = metrics(yte, yh); res['L3_pred'] = yh.tolist()
    # 4) SMOTE: clean (DS1 only) vs leaky (pooled DS1+DS2 before the split); synthetic beats go to training
    sc = StandardScaler().fit(Xtr); n_new = {1: 10000, 2: 10000, 3: 10000}
    for tag, (Xs, ys) in {'SMOTE_clean': (Xtr, ytr), 'L2': (np.vstack([Xtr, Xte]), np.concatenate([ytr, yte]))}.items():
        Sx, Sy, src = smote(sc.transform(Xs), ys, [1, 2, 3], n_new, seed)
        Sx = sc.inverse_transform(Sx)
        XA, yA = np.vstack([Xtr, Sx]), np.concatenate([ytr, Sy])
        pr = Prep(name).fit(XA, yA)
        A = pr.transform(XA); B = pr.transform(Xte)
        if name == 'MLP': A, yA = oversample(A, yA, seed)
        m = make_model(name, p, seed).fit(A, yA)
        yh = m.predict(B); res[tag] = metrics(yte, yh); res[tag + '_pred'] = yh.tolist()
        if tag == 'L2': res['L2_frac_synth_from_test'] = float(((src >= len(Xtr)).any(1)).mean())
    res['secs'] = time.time() - t0; out[name] = res
    print(name, f"{res['secs']:.0f}s", {k: round(v['macroF1'], 3) for k, v in res.items() if isinstance(v, dict)}, flush=True)
json.dump(out, open(f'../results/res_inter_s{seed}_{"_".join(models)}.json', 'w'))
