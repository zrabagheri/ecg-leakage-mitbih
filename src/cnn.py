"""Small 1-D CNN in NumPy (no deep-learning framework available offline).
Input: beat morphology (63 samples) + RR features. Architecture:
conv(1->16, k=7) -> ReLU -> maxpool(2) -> conv(16->32, k=5) -> ReLU -> global average pool
-> concat RR features -> dense(4). Trained with Adam, softmax cross-entropy, class-balanced weights."""
import numpy as np

def im2col(x, k):            # x [B, C, L] -> [B, L-k+1, C*k]
    B, C, L = x.shape; idx = np.arange(L - k + 1)[:, None] + np.arange(k)[None, :]
    return x[:, :, idx].transpose(0, 2, 1, 3).reshape(B, L - k + 1, C * k)

class CNN:
    def __init__(self, n_rr, seed=0, c1=16, c2=32, k1=7, k2=5, lr=1e-3, epochs=12, batch=256):
        r = np.random.default_rng(seed); self.r = r
        self.k1, self.k2, self.lr, self.epochs, self.batch = k1, k2, lr, epochs, batch
        he = lambda fan, shape: r.normal(0, np.sqrt(2 / fan), shape)
        self.P = dict(W1=he(k1, (k1, c1)), b1=np.zeros(c1), W2=he(c1 * k2, (c1 * k2, c2)), b2=np.zeros(c2),
                      W3=he(c2 + n_rr, (c2 + n_rr, 4)) * 0.5, b3=np.zeros(4))
        self.m = {k: np.zeros_like(v) for k, v in self.P.items()}; self.v = {k: np.zeros_like(v) for k, v in self.P.items()}; self.t = 0

    def _fwd(self, xm, xr):
        P = self.P; c = {}
        c['X1'] = im2col(xm[:, None, :], self.k1)                     # [B, L1, k1]
        z1 = c['X1'] @ P['W1'] + P['b1']; a1 = np.maximum(z1, 0); c['z1'] = z1
        L1 = a1.shape[1] // 2 * 2; a1r = a1[:, :L1].reshape(a1.shape[0], L1 // 2, 2, -1)
        c['arg'] = a1r.argmax(2); p1 = a1r.max(2); c['L1'] = L1; c['a1shape'] = a1.shape
        c['X2'] = im2col(p1.transpose(0, 2, 1), self.k2)              # [B, L2, c1*k2]
        z2 = c['X2'] @ P['W2'] + P['b2']; a2 = np.maximum(z2, 0); c['z2'] = z2
        g = a2.mean(1); h = np.hstack([g, xr]); c['h'] = h; c['L2'] = a2.shape[1]
        return h @ P['W3'] + P['b3'], c

    def _bwd(self, dlog, c, xm):
        P = self.P; G = {}
        G['W3'] = c['h'].T @ dlog; G['b3'] = dlog.sum(0); dh = dlog @ P['W3'].T
        c2 = P['W2'].shape[1]; dg = dh[:, :c2]
        da2 = np.repeat(dg[:, None, :] / c['L2'], c['L2'], 1); dz2 = da2 * (c['z2'] > 0)
        G['W2'] = np.einsum('blk,blc->kc', c['X2'], dz2); G['b2'] = dz2.sum((0, 1))
        dX2 = dz2 @ P['W2'].T                                         # [B, L2, c1*k2]
        B, L2, _ = dX2.shape; c1 = P['W1'].shape[1]; k2 = self.k2
        dX2 = dX2.reshape(B, L2, c1, k2); Lp = L2 + k2 - 1; dp1 = np.zeros((B, c1, Lp))
        for j in range(k2): dp1[:, :, j:j + L2] += dX2[:, :, :, j].transpose(0, 2, 1)
        dp1 = dp1.transpose(0, 2, 1)                                  # [B, Lp, c1]
        da1 = np.zeros(c['a1shape']); L1 = c['L1']
        da1r = np.zeros((B, L1 // 2, 2, c1)); np.put_along_axis(da1r, c['arg'][:, :, None, :], dp1[:, :, None, :], 2)
        da1[:, :L1] = da1r.reshape(B, L1, c1); dz1 = da1 * (c['z1'] > 0)
        G['W1'] = np.einsum('blk,blc->kc', c['X1'], dz1); G['b1'] = dz1.sum((0, 1))
        return G

    def fit(self, Xm, Xr, y):
        w = len(y) / (4 * np.maximum(np.bincount(y, minlength=4), 1)); w = w / w[y].mean()
        n = len(y)
        for ep in range(self.epochs):
            perm = self.r.permutation(n)
            for s in range(0, n, self.batch):
                b = perm[s:s + self.batch]; logit, c = self._fwd(Xm[b], Xr[b])
                p = np.exp(logit - logit.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
                d = p.copy(); d[np.arange(len(b)), y[b]] -= 1; d *= w[y[b]][:, None] / len(b)
                G = self._bwd(d, c, Xm[b]); self.t += 1
                for k in self.P:
                    self.m[k] = 0.9 * self.m[k] + 0.1 * G[k]; self.v[k] = 0.999 * self.v[k] + 0.001 * G[k] ** 2
                    mh = self.m[k] / (1 - 0.9 ** self.t); vh = self.v[k] / (1 - 0.999 ** self.t)
                    self.P[k] -= self.lr * mh / (np.sqrt(vh) + 1e-8)
        return self

    def predict(self, Xm, Xr):
        out = []
        for s in range(0, len(Xm), 4096): out.append(self._fwd(Xm[s:s + 4096], Xr[s:s + 4096])[0].argmax(1))
        return np.concatenate(out)

def gradcheck():
    r = np.random.default_rng(1); m = CNN(3, seed=0); xm = r.normal(size=(5, 63)); xr = r.normal(size=(5, 3)); y = r.integers(0, 4, 5)
    def loss():
        lg, _ = m._fwd(xm, xr); p = np.exp(lg - lg.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
        return -np.log(p[np.arange(5), y]).mean()
    lg, c = m._fwd(xm, xr); p = np.exp(lg - lg.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    d = p.copy(); d[np.arange(5), y] -= 1; d /= 5; G = m._bwd(d, c, xm); worst = 0
    for k in m.P:
        flat = m.P[k].reshape(-1)
        for i in r.choice(flat.size, min(5, flat.size), replace=False):
            o = flat[i]; flat[i] = o + 1e-5; lp = loss(); flat[i] = o - 1e-5; lm = loss(); flat[i] = o
            num = (lp - lm) / 2e-5; worst = max(worst, abs(num - G[k].reshape(-1)[i]) / (abs(num) + abs(G[k].reshape(-1)[i]) + 1e-12))
    return worst
if __name__ == '__main__':
    print('max relative gradient error', gradcheck())
