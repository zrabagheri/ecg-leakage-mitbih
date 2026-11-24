"""Shared pipeline: features, models, metrics, SMOTE, cluster bootstrap."""
import numpy as np, pickle, os
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, balanced_accuracy_score
from sklearn.neighbors import NearestNeighbors

HERE = os.path.dirname(__file__); DATA = os.path.join(HERE, '..')
CLASSES = np.array(['N', 'S', 'V', 'F'])

def load(name, peaks):
    D = pickle.load(open(os.path.join(DATA, f'{name}_{peaks}.pkl'), 'rb'))
    X = np.hstack([D['rr'], D['morph']]); m = {c: i for i, c in enumerate(CLASSES)}; y = np.array([m[v] for v in D['y']])
    assert (CLASSES[y] == D['y']).all()
    return X, y, D['rec']

GRIDS = {
    'LR': [dict(C=c) for c in (0.01, 0.1, 1, 10)],
    'RF': [dict(n_estimators=300, max_depth=d, min_samples_leaf=l) for d in (None, 20) for l in (1, 5)],
    'MLP': [dict(hidden_layer_sizes=h, alpha=a) for h in ((64,), (128, 64)) for a in (1e-4, 1e-2)],
    'CNN': [dict(epochs=e, lr=l) for e in (8, 16) for l in (1e-3, 3e-3)],
}

class CNNWrap:
    def __init__(self, seed, **p): self.seed, self.p = seed, p
    def fit(self, X, y):
        from cnn import CNN
        self.m = CNN(7, seed=self.seed, **self.p).fit(X[:, 7:], X[:, :7], y); return self
    def predict(self, X): return self.m.predict(X[:, 7:], X[:, :7])
K_FEAT = 40   # features kept by supervised selection (of 70)

def make_model(name, p, seed):
    if name == 'LR':
        return LogisticRegression(C=p['C'], class_weight='balanced', max_iter=2000)
    if name == 'CNN':
        return CNNWrap(seed, **p)
    if name == 'RF':
        return RandomForestClassifier(class_weight='balanced_subsample', n_jobs=2, random_state=seed, **p)
    return MLPClassifier(early_stopping=True, max_iter=200, random_state=seed, **p)

def oversample(X, y, seed):
    """Random oversampling of minority classes to the majority count (used for MLP only)."""
    r = np.random.default_rng(seed); n = np.bincount(y).max(); idx = []
    for c in np.unique(y):
        ic = np.where(y == c)[0]; idx.append(ic if len(ic) == n else np.concatenate([ic, r.choice(ic, n - len(ic))]))
    idx = np.concatenate(idx); return X[idx], y[idx]

class Prep:
    """Scaler + supervised feature selection (feature-based models) or scaler only (CNN, which needs
    the full ordered waveform); fitted on whatever data the protocol allows."""
    def __init__(self, name=None):
        self.select = name != 'CNN'
    def fit(self, X, y):
        self.sc = StandardScaler().fit(X)
        if self.select: self.sel = SelectKBest(f_classif, k=K_FEAT).fit(self.sc.transform(X), y)
        return self
    def transform(self, X):
        Z = self.sc.transform(X); return self.sel.transform(Z) if self.select else Z

def fit_predict(name, p, seed, Xtr, ytr, Xte, prep):
    A, B = prep.transform(Xtr), prep.transform(Xte)
    if name == 'MLP': A, ytr = oversample(A, ytr, seed)
    m = make_model(name, p, seed).fit(A, ytr)
    return m.predict(B)

def metrics(y, yhat):
    lab = [0, 1, 2, 3]
    se = recall_score(y, yhat, labels=lab, average=None, zero_division=0)
    pp = precision_score(y, yhat, labels=lab, average=None, zero_division=0)
    return dict(acc=accuracy_score(y, yhat), macroF1=f1_score(y, yhat, labels=lab, average='macro', zero_division=0),
                bacc=balanced_accuracy_score(y, yhat), Se_S=se[1], PP_S=pp[1], Se_V=se[2], PP_V=pp[2])

def smote(X, y, classes, n_new, seed, k=5):
    """SMOTE (Chawla et al., 2002): for each class in `classes`, make n_new[c] synthetic samples by
    interpolating a random sample toward one of its k nearest same-class neighbours."""
    r = np.random.default_rng(seed); outX, outy, src = [], [], []
    for c in classes:
        ic = np.where(y == c)[0]; Xc = X[ic]
        nn = NearestNeighbors(n_neighbors=k + 1).fit(Xc); _, nb = nn.kneighbors(Xc)
        a = r.integers(0, len(ic), n_new[c]); b = nb[a, r.integers(1, k + 1, n_new[c])]
        lam = r.random((n_new[c], 1))
        outX.append(Xc[a] + lam * (Xc[b] - Xc[a])); outy.append(np.full(n_new[c], c)); src.append(np.stack([ic[a], ic[b]], 1))
    return np.vstack(outX), np.concatenate(outy), np.vstack(src)

def cluster_bootstrap(y, yhat, rec, B=1000, seed=0):
    """Percentile CIs by resampling test records with replacement."""
    r = np.random.default_rng(seed); recs = np.unique(rec); groups = [np.where(rec == g)[0] for g in recs]
    out = {k: [] for k in metrics(y, yhat)}
    for _ in range(B):
        idx = np.concatenate([groups[i] for i in r.integers(0, len(recs), len(recs))])
        for k, v in metrics(y[idx], yhat[idx]).items(): out[k].append(v)
    return {k: (np.percentile(v, 2.5), np.percentile(v, 97.5)) for k, v in out.items()}
