"""Paper tables (CSV) from results_long.csv, paired_deltas.csv and audit.csv."""
import pandas as pd, numpy as np
df = pd.read_csv('../results/results_long.csv'); P = "../tables/"
order = ['C0', 'L5', 'L4', 'L3', 'SMOTE_clean', 'L2', 'L1', 'ALL']
label = {'C0': 'C0 clean', 'L5': 'L5 reference peaks', 'L4': 'L4 test-set selection', 'L3': 'L3 global preprocessing',
         'SMOTE_clean': 'SMOTE on training set (control)', 'L2': 'L2 SMOTE before split', 'L1': 'L1 intra-patient split', 'ALL': 'All leaks (L1–L4)'}
models = ['LR', 'RF', 'MLP', 'CNN']
def fmt(s):
    mu, sd, n = s.mean(), s.std(ddof=1), len(s)
    return f'{100*mu:.1f}' if n < 2 or np.isnan(sd) or sd < 5e-4 else f'{100*mu:.1f} ± {100*sd:.1f}'
rows = []
for c in order:
    for mo in models:
        d = df[(df.cond == c) & (df.model == mo)]
        if d.empty: continue
        rows.append({'Condition': label[c], 'Model': mo, 'Acc (%)': fmt(d.acc), 'Macro-F1 (%)': fmt(d.macroF1), 'Bal. acc (%)': fmt(d.bacc),
                     'Se S (%)': fmt(d.Se_S), '+P S (%)': fmt(d.PP_S), 'Se V (%)': fmt(d.Se_V), '+P V (%)': fmt(d.PP_V)})
pd.DataFrame(rows).to_csv(P + 'table1_results.csv', index=False)
# Table 3: paired deltas (seed 0) in percentage points
pdl = pd.read_csv('../results/paired_deltas.csv'); rows = []
for c in ['L5', 'L4', 'L3', 'L2']:
    for mo in models:
        r = {'Leak (vs. reference)': f"{label[c]} (vs. {'SMOTE control' if c == 'L2' else 'C0'})", 'Model': mo}
        for k, nm in [('acc', 'Δ Acc'), ('macroF1', 'Δ Macro-F1'), ('bacc', 'Δ Bal. acc')]:
            x = pdl[(pdl.cond == c) & (pdl.model == mo) & (pdl.metric == k)].iloc[0]
            r[nm + ' (pp, 95% CI)'] = f"{100*x.delta:+.1f} ({100*x.lo:+.1f} to {100*x.hi:+.1f})"
        rows.append(r)
pd.DataFrame(rows).to_csv(P + 'table2_deltas.csv', index=False)
a = pd.read_csv('../audit/audit.csv')
t = a[['first_author', 'year', 'venue', 'split', 'oversampling_aug', 'model_selection', 'r_peaks', 'headline_acc']].copy()
t.columns = ['First author', 'Year', 'Venue', 'Split', 'Oversampling / augmentation', 'Model selection', 'R peaks', 'Accuracy (%)']
t.to_csv(P + 'table3_audit.csv', index=False); print('tables written')
