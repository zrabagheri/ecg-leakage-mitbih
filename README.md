# Beyond the inter-patient split: five sources of data leakage in MIT-BIH heartbeat classification

Code and results for the paper *"Beyond the inter-patient split: quantifying five sources of data leakage in MIT-BIH heartbeat classification"* (Z. Bagheri, submitted).

Starting from a leakage-free inter-patient pipeline (train DS1, test DS2, automatic beat detection), the code introduces one leakage source at a time and measures its effect with four model families (logistic regression, random forest, MLP, 1-D CNN):

| Code | Leakage source |
|---|---|
| L1 | Intra-patient (random beat-level) split |
| L2 | SMOTE oversampling before the train/test split |
| L3 | Standardisation and supervised feature selection fitted on all data |
| L4 | Hyperparameter / model selection on the test set |
| L5 | Reference R-peak annotations used to locate test beats |
| ALL | L1–L4 combined |

## Requirements

Python 3.10+ with `numpy`, `scipy`, `scikit-learn`, `pandas`, `matplotlib`. No deep-learning framework or `wfdb` package is needed: the WFDB readers and the CNN are implemented in NumPy.

## Data

Download the MIT-BIH Arrhythmia Database (v1.0.0) from PhysioNet: https://physionet.org/content/mitdb/1.0.0/ and unzip it into `raw/` so that `raw/mit-bih-arrhythmia-database-1.0.0/100.dat` exists. The data are not redistributed here.

## Reproducing the results

```
cd src
python -c "from dataset import *; import pickle
for n,r in [('DS1',DS1),('DS2',DS2)]:
    for p in ('ref','det'): pickle.dump(build(r,p), open(f'../{n}_{p}.pkl','wb'))"
python cnn.py                         # gradient check of the NumPy CNN
python exp_inter.py 0                 # C0 and L2-L5, seed 0 (hyperparameters by grouped CV)
python exp_intra.py 0                 # L1 and ALL, seed 0
python exp_inter.py 1 RF,MLP,CNN      # extra seeds reuse the seed-0 hyperparameters
python exp_inter.py 2 RF,MLP,CNN
python exp_intra.py 1; python exp_intra.py 2
python summarize.py; python paired.py; python figures.py
```

## Files

- `src/wfdb_io.py` – readers for WFDB format-212 signals and MIT annotations
- `src/dataset.py` – AAMI class mapping, DS1/DS2, preprocessing, features, QRS detector
- `src/pipeline.py` – models, preprocessing, SMOTE, metrics, record-level bootstrap
- `src/cnn.py` – 1-D CNN in NumPy with gradient check
- `src/exp_inter.py`, `src/exp_intra.py` – experiments
- `src/summarize.py`, `src/paired.py`, `src/figures.py` – tables and figures
- `results/` – raw result files (JSON) and summary tables (CSV)
- `tables/` – Tables 1–3 of the paper (`table1_results.csv`, `table2_deltas.csv`, `table3_audit.csv`); `figures/` – Figs. 1 and 2
- `audit/audit.csv` – coding of the 19 audited articles

## License

MIT
