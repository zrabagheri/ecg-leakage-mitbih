"""Figures for the paper."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 8, 'font.family': 'DejaVu Sans', 'axes.spines.top': False, 'axes.spines.right': False})
df = pd.read_csv('../results/results_long.csv')
m = df.groupby(['cond', 'model'])[['acc', 'macroF1', 'bacc', 'Se_S', 'Se_V']].agg(['mean', 'std'])
models = ['LR', 'RF', 'MLP', 'CNN']; cols = {'LR': '#4C72B0', 'RF': '#55A868', 'MLP': '#C44E52', 'CNN': '#8172B2'}
conds = ['C0', 'L5', 'L4', 'L3', 'L2', 'L1', 'ALL']
names = {'C0': 'Clean\n(C0)', 'L5': 'L5\nref. peaks', 'L4': 'L4\ntest tuning', 'L3': 'L3\nglobal prep.', 'L2': 'L2\nSMOTE first',
         'L1': 'L1\nintra-patient', 'ALL': 'All leaks'}
# Figure 1: accuracy vs macro-F1 per condition
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.9), sharex=True)
for ax, (met, lab) in zip(axs, [('acc', 'Accuracy'), ('macroF1', 'Macro-F1'), ('bacc', 'Balanced accuracy')]):
    x = np.arange(len(conds)); w = 0.2
    for i, mo in enumerate(models):
        mu = [m.loc[(c, mo), (met, 'mean')] if (c, mo) in m.index else np.nan for c in conds]
        sd = [m.loc[(c, mo), (met, 'std')] if (c, mo) in m.index else np.nan for c in conds]
        ax.bar(x + (i - 1.5) * w, mu, w, yerr=np.nan_to_num(sd), color=cols[mo], label=mo, error_kw=dict(lw=0.6, capsize=1))
    ax.set_xticks(x); ax.set_xticklabels([names[c] for c in conds], rotation=90, fontsize=7); ax.set_ylim(0, 1); ax.set_title(lab)
    if met == 'acc': ax.axhline(42530/47791, ls='--', lw=0.8, c='k')
h, l = axs[0].get_legend_handles_labels()
h.append(plt.Line2D([], [], ls='--', c='k', lw=0.8)); l.append('all-normal accuracy')
fig.legend(h, l, ncol=5, fontsize=7, loc='upper center', frameon=False, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig('../figures/fig1_conditions.png', dpi=300); plt.close(fig)
# Figure 2: S and V sensitivity per condition
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
for ax, (met, lab) in zip(axs, [('Se_S', 'Supraventricular (S) sensitivity'), ('Se_V', 'Ventricular (V) sensitivity')]):
    x = np.arange(len(conds)); w = 0.2
    for i, mo in enumerate(models):
        mu = [m.loc[(c, mo), (met, 'mean')] if (c, mo) in m.index else np.nan for c in conds]
        ax.bar(x + (i - 1.5) * w, mu, w, color=cols[mo], label=mo)
    ax.set_xticks(x); ax.set_xticklabels([names[c] for c in conds], rotation=90, fontsize=7); ax.set_ylim(0, 1.02); ax.set_title(lab)
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, ncol=4, fontsize=7, loc='upper center', frameon=False, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig('../figures/fig2_sensitivity.png', dpi=300); plt.close(fig)
print('figures written')
