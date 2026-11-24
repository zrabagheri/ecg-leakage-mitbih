"""Beat extraction for MIT-BIH following AAMI EC57 conventions (de Chazal et al., 2004)."""
import numpy as np
from scipy.signal import butter, filtfilt, medfilt
from wfdb_io import read_signal, read_annotations

DS1 = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
DS2 = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]
# MIT annotation codes -> AAMI classes (paced records 102, 104, 107, 217 excluded)
AAMI = {1: 'N', 2: 'N', 3: 'N', 34: 'N', 11: 'N',   # N L R e j
        8: 'S', 4: 'S', 7: 'S', 9: 'S',             # A a J S
        5: 'V', 10: 'V',                            # V E
        6: 'F',                                     # F
        12: 'Q', 38: 'Q', 13: 'Q'}                  # / f Q
CLASSES = ['N', 'S', 'V', 'F']                      # Q (15 beats) excluded, as is standard
PRE, POST = 0.25, 0.45                              # window (s) around R peak

def preprocess(sig, fs):
    """Baseline removal with 200/600 ms median filters, then 35 Hz low-pass (zero phase)."""
    b1 = medfilt(sig, int(0.2 * fs) | 1); b2 = medfilt(b1, int(0.6 * fs) | 1)
    x = sig - b2
    b, a = butter(4, 35 / (fs / 2), 'low')
    return filtfilt(b, a, x)

def detect_qrs(x, fs):
    """Simple Pan-Tompkins-style detector: band-pass 5-15 Hz, derivative, square, 150 ms
    moving integration, adaptive threshold, 200 ms refractory, peak refined on |signal|."""
    b, a = butter(2, [5 / (fs / 2), 15 / (fs / 2)], 'band')
    y = filtfilt(b, a, x)
    d = np.diff(y, prepend=y[0]) ** 2
    w = int(0.15 * fs); m = np.convolve(d, np.ones(w) / w, 'same')
    refr = int(0.2 * fs); peaks = []
    spk = np.percentile(m[: 2 * fs * 10], 98) * 0.5; npk = np.median(m[: 2 * fs * 10])
    i = 1
    from scipy.signal import find_peaks
    cand, _ = find_peaks(m, distance=refr)
    for c in cand:
        thr = npk + 0.25 * (spk - npk)
        if m[c] > thr:
            lo, hi = max(0, c - int(0.1 * fs)), min(len(x), c + int(0.1 * fs))
            r = lo + int(np.argmax(np.abs(x[lo:hi])))
            if peaks and r - peaks[-1] < refr: continue
            peaks.append(r); spk = 0.125 * m[c] + 0.875 * spk
        else:
            npk = 0.125 * m[c] + 0.875 * npk
    return np.array(peaks)

def beats_from_record(rec, peaks='ref', tol=0.15):
    sig, fs, names = read_signal(str(rec))
    ch = names.index('MLII'); x = preprocess(sig[:, ch], fs)
    samp, code = read_annotations(str(rec))
    isbeat = np.array([c in AAMI for c in code])
    rs, rl = samp[isbeat], np.array([AAMI[c] for c in code[isbeat]])
    if peaks == 'ref':
        pos, lab = rs, rl
    else:  # detector positions; label = matched reference beat within tol, else dropped
        det = detect_qrs(x, fs); t = int(tol * fs)
        j = np.searchsorted(rs, det); pos, lab = [], []
        for p, k in zip(det, j):
            best = None
            for kk in (k - 1, k):
                if 0 <= kk < len(rs) and abs(rs[kk] - p) <= t and (best is None or abs(rs[kk] - p) < abs(rs[best] - p)):
                    best = kk
            pos.append(p); lab.append(rl[best] if best is not None else None)
        pos = np.array(pos); lab = np.array(lab, dtype=object)
    rr = np.diff(pos) / fs
    pre_w, post_w = int(PRE * fs), int(POST * fs)
    feats, morph, labels, idx = [], [], [], []
    for i in range(1, len(pos) - 1):
        if lab[i] is None or lab[i] not in CLASSES: continue
        p = pos[i]
        if p - pre_w < 0 or p + post_w >= len(x): continue
        pre, post = rr[i - 1], rr[i]
        lo, hi = max(0, i - 10), min(len(rr), i + 10); local = rr[lo:hi].mean()
        t0 = max(0, i - int(300 / max(rr.mean(), 1e-3))); avg = rr[t0:i].mean() if i > t0 else pre
        feats.append([pre, post, local, avg, pre / local, post / local, pre / avg])
        morph.append(x[p - pre_w: p + post_w: 4])
        labels.append(lab[i]); idx.append(p)
    return dict(rr=np.array(feats), morph=np.array(morph), y=np.array(labels),
                pos=np.array(idx), rec=np.full(len(labels), rec))

def build(records, peaks='ref'):
    parts = [beats_from_record(r, peaks) for r in records]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
