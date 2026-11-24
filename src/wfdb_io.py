"""Minimal readers for MIT-BIH (WFDB format 212 signals, MIT-format annotations)."""
import numpy as np, os

RAW = os.path.join(os.path.dirname(__file__), '..', 'raw', 'mit-bih-arrhythmia-database-1.0.0')

def read_header(rec, d=RAW):
    lines = [l for l in open(os.path.join(d, f'{rec}.hea')).read().splitlines() if l and not l.startswith('#')]
    _, nsig, fs, n = lines[0].split()[:4]
    sigs = []
    for l in lines[1:1 + int(nsig)]:
        p = l.split()
        gain = float(p[2].split('(')[0].split('/')[0]); base = int(p[4]) if len(p) > 4 else 0
        bl = int(p[2].split('(')[1].split(')')[0]) if '(' in p[2] else base
        sigs.append(dict(fmt=p[1], gain=gain, adczero=int(p[4]), baseline=bl, name=p[-1]))
    return int(fs), int(n), sigs

def read_signal(rec, d=RAW):
    """Return (signal in mV [n, 2], fs, channel names)."""
    fs, n, sigs = read_header(rec, d)
    raw = np.fromfile(os.path.join(d, f'{rec}.dat'), dtype=np.uint8)
    raw = raw[: (len(raw) // 3) * 3].reshape(-1, 3).astype(np.int32)
    s0 = raw[:, 0] | ((raw[:, 1] & 0x0F) << 8)
    s1 = raw[:, 2] | ((raw[:, 1] & 0xF0) << 4)
    s0 = np.where(s0 > 2047, s0 - 4096, s0); s1 = np.where(s1 > 2047, s1 - 4096, s1)
    x = np.stack([s0, s1], 1)[:n].astype(float)
    for c, s in enumerate(sigs):
        x[:, c] = (x[:, c] - s['baseline']) / s['gain']
    return x, fs, [s['name'] for s in sigs]

def read_annotations(rec, ext='atr', d=RAW):
    """Return (sample indices, annotation codes) for MIT-format annotation file."""
    b = np.fromfile(os.path.join(d, f'{rec}.{ext}'), dtype='<u2')
    samples, codes = [], []
    t = 0; i = 0
    while i < len(b):
        w = int(b[i]); code = w >> 10; val = w & 0x3FF
        if code == 0 and val == 0: break
        if code == 59:  # SKIP: next two words form a 32-bit interval (PDP-11 order)
            t += (int(b[i + 1]) << 16) | int(b[i + 2]); i += 3; continue
        if code == 63:  # AUX: val bytes follow, padded to even
            i += 1 + (val + 1) // 2; continue
        if code in (60, 61, 62):  # NUM, SUB, CHN
            i += 1; continue
        t += val; samples.append(t); codes.append(code); i += 1
    return np.array(samples), np.array(codes)
