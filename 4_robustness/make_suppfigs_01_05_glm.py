"""Supplementary Figs. 1-5: GLM task regression.

  SuppFig01_glm_regressors_NF        NF block design and HRF-convolved task regressors
  SuppFig02_glm_regressors_DC        DC block design (reconstructed templates) and regressors
  SuppFig03_glm_comparison_summary   raw vs task-regressed graphs, with run-to-run reference
  SuppFig04_glm_sample_graphs_A0     example contemporaneous graphs, raw / residual / difference
  SuppFig05_glm_sample_graphs_A1     the same for the lagged graphs

Inputs: Results/glm_revision/figures/glm9_comparison_summary.csv (make_glm_comparison_9subj.py)
and the raw and residual CaLLTiF graphs.
Output folder: Results/figures_supplementary/
"""
import os
import glob
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.gridspec import GridSpec
import glm_task_regression as glm

# ------------------------------------------------------------------ config
OUT = 'Results/figures_supplementary'
CSV = 'Results/glm_revision/figures/glm9_comparison_summary.csv'
RESID = 'Results/glm_revision/glm_results_9subj'
RAW_NF = 'Results/glm_revision/raw/calltif_decnef_study{s}_nf_two_runs_one_graph.npz'
RAW_DC = 'Results/calltif_decnef_study{s}_dec_two_runs_one_graph.npz'
MAP = 'Results/glm_revision/parcel_mapping.npz'
EV_DIR_DC = 'DecNef_events_DC'
STUDIES = (1, 2, 3, 4, 5)
SN = {1: 'S1  facial preference', 2: 'S2  orientation-colour', 3: 'S3  fear counter-cond.',
      4: 'S4  confidence', 5: 'S5  fear reprogramming'}
SN_SHORT = {1: 'S1 facial pref.', 2: 'S2 orient-colour', 3: 'S3 fear-cc',
            4: 'S4 confidence', 5: 'S5 fear-reprog.'}
# colour scheme (paper convention): NF = dark greens, DC = light greens; two shades per family
# for the binary vs HRF variants.
COL = {('nf', 'binary'): '#006400', ('nf', 'hrf'): '#2ca02c',
       ('dc', 'binary'): '#74c476', ('dc', 'hrf'): '#addd8e'}
REF = '#8a8a8a'

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 13, 'axes.titlesize': 15, 'axes.labelsize': 14,
    'legend.fontsize': 11, 'xtick.labelsize': 12, 'ytick.labelsize': 12,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.dpi': 130, 'savefig.bbox': 'tight', 'pdf.fonttype': 42,
})

def panel_tag(ax, text, pad, fontsize=None):
    """Panel label placed BELOW the panel (offset in points below the axes)."""
    ax.annotate(text, xy=(0.5, 0), xycoords='axes fraction', xytext=(0, -pad),
                textcoords='offset points', ha='center', va='top',
                fontsize=fontsize or plt.rcParams['axes.titlesize'])


_rawcache = {}


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(OUT, f'{name}.{ext}'), dpi=200)
    plt.close(fig)
    print('saved', os.path.join(OUT, name + '.pdf'))


def raw_lm(cond, study):
    key = (cond, study)
    if key not in _rawcache:
        path = (RAW_NF if cond == 'nf' else RAW_DC).format(s=study)
        d = np.load(path, allow_pickle=True)
        _rawcache[key] = d[[k for k in d.files if 'link_matrix' in k][0]]
    return _rawcache[key]


def paper_order():
    m = np.load(MAP, allow_pickle=True)
    idx = [np.asarray(m[k]) for k in m['net_labels']]
    order = np.concatenate(idx)
    bounds = list(np.cumsum([len(k) for k in idx])[:-1])
    centers = [(a + b) / 2 for a, b in zip([0] + bounds, bounds + [len(order)])]
    labels = [n.replace('7Networks_', '').replace('_', '-') for n in m['net_labels']]
    return order, bounds, centers, labels


# ------------------------------------------------------------------ reference
def run_to_run_reference(cond='nf', lag=0):
    """Within-subject raw-vs-raw edge-weight correlation across all graphs of a
    subject (the natural run-to-run variability the regression must beat)."""
    vals = []
    for s in STUDIES:
        lm = raw_lm(cond, s)
        for subj in range(len(lm)):
            graphs = []
            try:
                sub = lm[subj]
            except Exception:
                continue
            for ses in range(len(sub)):
                for r in range(len(sub[ses])):
                    e = sub[ses][r]
                    if isinstance(e, dict):
                        graphs.append(np.array(e['val_matrix'])[:, :, lag].ravel())
            for a, b in itertools.combinations(graphs, 2):
                ok = np.isfinite(a) & np.isfinite(b)
                if ok.sum() > 2 and a[ok].std() and b[ok].std():
                    vals.append(np.corrcoef(a[ok], b[ok])[0, 1])
    return np.array(vals)


# ------------------------------------------------------------------ fig 1: NF regressors
def fig_regressors_nf():
    evs = ['induction', 'feedback', 'baseline']
    ecol = {'induction': '#d95f02', 'feedback': '#1b9e77', 'baseline': '#7570b3'}
    fig, axes = plt.subplots(5, 2, figsize=(13, 12.5),
                             gridspec_kw={'hspace': 0.62, 'wspace': 0.16,
                                          'width_ratios': [3, 1.25]})
    for i, s in enumerate(STUDIES):
        sub = glm.SUBJECT_IDS[s][0]
        disk = 0 + glm.SESSION_OFFSET[s]
        data, _ = glm.load_nf_npz(s)
        ts = np.array(data[0][0][0])
        n = ts.shape[0]
        ev = glm.load_events(glm.events_path(s, sub, disk, 1))
        X = glm.build_design_matrix(ev, n, hrf=True, event_types=evs)
        t = np.arange(n) * glm.TR
        fb = sorted({d for nm, o, d in ev if nm == 'feedback'})[0]
        for k, e in enumerate(evs):
            axes[i, 0].plot(t, X[:, k], color=ecol[e], lw=1.4, label=e if i == 0 else None)
        axes[i, 0].set_ylabel('HRF regressor')
        axes[i, 0].set_title(f'{SN[s]}   (NF run {n * glm.TR:.0f} s; feedback block {fb:.0f} s)',
                             loc='left')
        axes[i, 0].margins(x=0.01)
        # zoom on ~2 trials -> shows the feedback-width difference across studies
        ind_on = [o for nm, o, d in ev if nm == 'induction']
        z0 = ind_on[0] - 2
        period = (ind_on[1] - ind_on[0]) if len(ind_on) > 1 else 24
        z1 = z0 + 2 * period + 4
        for nm, o, d in ev:
            if nm in ('induction', 'feedback') and o + d > z0 and o < z1:
                axes[i, 1].axvspan(o, o + d, color=ecol[nm], alpha=0.85)
        axes[i, 1].set_xlim(z0, z1)
        axes[i, 1].set_ylim(0, 1)
        axes[i, 1].set_yticks([])
        axes[i, 1].set_title('one trial (zoom)')
        fo = [o for nm, o, d in ev if nm == 'feedback' and o > z0][0]
        axes[i, 1].annotate(f'{fb:.0f} s', xy=(fo + fb / 2, 0.5), ha='center', va='center',
                            fontsize=10, color='white', weight='bold')
    axes[-1, 0].set_xlabel('time (s)')
    axes[-1, 1].set_xlabel('time (s)')
    axes[0, 0].legend(ncol=3, loc='upper right')
    axes[0, 1].legend(handles=[Patch(color=ecol['induction'], label='induction (6 s)'),
                               Patch(color=ecol['feedback'], label='feedback (study-specific)')],
                      loc='upper center', bbox_to_anchor=(0.5, 1.55), fontsize=10)
    save(fig, 'SuppFig01_glm_regressors_NF')


# ------------------------------------------------------------------ fig 2: DC regressors
def fig_regressors_dc():
    N_TRS = {1: 126, 2: 152, 3: 152, 4: 154, 5: 307}
    EVENT_TYPES = {1: ['face', 'rating', 'report'], 2: ['grating', 'blank'],
                   3: ['grating', 'fixation'], 4: ['noise', 'stimulus', 'delay', 'response'],
                   5: ['image']}
    ZOOM = {1: 46, 2: 46, 3: 50, 4: 62, 5: 16}
    NAME = {1: 'S1  facial preference (faces + rating)',
            2: 'S2  orientation-colour (gratings)',
            3: 'S3  fear counter-cond. (gratings)',
            4: 'S4  confidence (dot-motion + decision)',
            5: 'S5  fear reprogramming (rapid images, 1-back)'}
    ccol = {'face': '#d95f02', 'rating': '#1b9e77', 'report': '#7570b3',
            'grating': '#d95f02', 'blank': '#999999', 'fixation': '#999999',
            'noise': '#a6761d', 'stimulus': '#d95f02', 'delay': '#66a61e',
            'response': '#7570b3', 'image': '#d95f02'}
    fig, ax = plt.subplots(5, 2, figsize=(13, 12.5),
                           gridspec_kw={'width_ratios': [3, 1.25], 'hspace': 0.62, 'wspace': 0.16})
    for r, s in enumerate(STUDIES):
        evs = glm.load_events(os.path.join(EV_DIR_DC, f'Study{s}', f'study{s}_task-decoder_events.csv'))
        n_trs = N_TRS[s]
        ets = EVENT_TYPES[s]
        X = glm.build_design_matrix(evs, n_trs, hrf=True, event_types=ets, add_constant=False)
        t = np.arange(n_trs) * glm.TR
        for col, ax_ in enumerate((ax[r, 0], ax[r, 1])):
            for name, onset, dur in evs:
                if name in ets and dur > 0:
                    ax_.axvspan(onset, onset + dur, color=ccol.get(name, '#ccc'), alpha=0.10)
            for i, ev in enumerate(ets):
                ax_.plot(t, X[:, i], color=ccol.get(ev, '#333'), lw=1.4, label=ev)
            ax_.set_xlim(0, t[-1] if col == 0 else ZOOM[s])
            ax_.margins(y=0.1)
        ax[r, 0].set_ylabel('HRF regressor')
        ax[r, 0].set_title(f'{NAME[s]}   ({n_trs} TR / {n_trs * glm.TR:.0f} s)', loc='left')
        ax[r, 1].set_title('zoom (first trials)')
        ax[r, 0].legend(fontsize=10, ncol=len(ets), loc='upper right')
    ax[4, 0].set_xlabel('time (s)')
    ax[4, 1].set_xlabel('time (s)')
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    save(fig, 'SuppFig02_glm_regressors_DC')


# ------------------------------------------------------------------ fig 3: summary a-e
def fig_summary():
    df = pd.read_csv(CSV)
    ref0 = run_to_run_reference('nf', 0); ref0_med = np.nanmedian(ref0)
    ref1 = run_to_run_reference('nf', 1); ref1_med = np.nanmedian(ref1)

    fig = plt.figure(figsize=(13, 9))
    gs = GridSpec(2, 2, figure=fig, hspace=0.34, wspace=0.24)
    axa, axb = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    axc, axd = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])

    def hist_panel(ax, col, xr, title, xlabel, ref=None, ref_med=None):
        if ref is not None:
            ax.hist(ref, bins=40, range=xr, density=True, color=REF, alpha=0.35,
                    label=f'run-to-run (no GLM), med {ref_med:.2f}')
            ax.axvline(ref_med, color=REF, ls='--', lw=1.2)
        for cond in ('nf', 'dc'):
            for v in ('binary', 'hrf'):
                x = df[(df.cond == cond) & (df.variant == v)][col].dropna().values
                m = np.median(x)
                ax.hist(x, bins=40, range=xr, density=True, histtype='step', lw=1.8,
                        color=COL[(cond, v)], label=f'{cond.upper()} {v}, med {m:.3f}')
                ax.axvline(m, color=COL[(cond, v)], ls='--', lw=0.9, alpha=0.7)
        ax.set_xlim(*xr)
        panel_tag(ax, title, 46)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('density')
        ax.legend(fontsize=10, loc='upper left')

    hist_panel(axa, 'r0', (0.70, 1.0), r'(a)  $A_0$ edge-weight correlation',
               'Pearson r (raw vs GLM-residual)', ref=ref0, ref_med=ref0_med)
    hist_panel(axb, 'r1', (-0.2, 1.0), r'(b)  $A_1$ edge-weight correlation',
               'Pearson r (raw vs GLM-residual)', ref=ref1, ref_med=ref1_med)

    # per-study boxplots (HRF variant; NF and DC)
    def per_study(ax, col, title, ylabel, ref_line=None):
        pos = 0
        xt, xl = [], []
        for s in STUDIES:
            for j, cond in enumerate(('nf', 'dc')):
                vals = df[(df.cond == cond) & (df.variant == 'hrf') & (df.study == s)][col].dropna().values
                p = pos + j * 0.85
                bp = ax.boxplot(vals, positions=[p], widths=0.62, patch_artist=True,
                                showfliers=False, medianprops=dict(color='k', lw=1.2))
                bp['boxes'][0].set(facecolor=COL[(cond, 'hrf')], alpha=0.75)
            xt.append(pos + 0.42)
            xl.append(SN_SHORT[s])
            pos += 2.5
        if ref_line is not None:
            ax.axhline(ref_line, color=REF, ls='--', lw=1.3, zorder=0)
        ax.set_xticks(xt)
        ax.set_xticklabels(xl, rotation=18, ha='right')
        panel_tag(ax, title, 52)
        ax.set_ylabel(ylabel)

    per_study(axc, 'r0', r'(c)  $A_0$ correlation per study (NF & DC)', 'Pearson r', ref_line=ref0_med)
    per_study(axd, 'r1', r'(d)  $A_1$ correlation per study (NF & DC)', 'Pearson r', ref_line=ref1_med)
    axc.legend(handles=[Patch(facecolor=COL[('nf', 'hrf')], alpha=.75, label='NF (HRF)'),
                        Patch(facecolor=COL[('dc', 'hrf')], alpha=.75, label='DC (HRF)'),
                        plt.Line2D([0], [0], color=REF, ls='--', label='run-to-run ref')],
               fontsize=10, loc='lower right')

    save(fig, 'SuppFig03_glm_comparison_summary')


# ------------------------------------------------------------------ fig 4/5: sample graphs
def fig_sample_graphs(lag, variant='hrf'):
    """One figure per lag, showing BOTH NF and DC: 5 studies x 6 columns
    (NF raw | NF residual | NF diff | DC raw | DC residual | DC diff)."""
    order, bounds, centers, netlab = paper_order()
    col_titles = ['NF: without GLM', 'NF: GLM-residual', 'NF: difference',
                  'DC: without GLM', 'DC: GLM-residual', 'DC: difference']
    fig, axes = plt.subplots(5, 6, figsize=(19, 15.5),
                             gridspec_kw={'hspace': 0.16, 'wspace': 0.40})
    for i, s in enumerate(STUDIES):
        for ci, cond in enumerate(('nf', 'dc')):
            e = raw_lm(cond, s)[0][0][0]
            raw = np.array(e['val_matrix'])
            rf = os.path.join(RESID, f'{cond}_{variant}_9subj_study{s}_subj0_ses0_pair0.npz')
            resid = np.load(rf)['val_matrix']
            R = raw[:, :, lag][np.ix_(order, order)]
            S = resid[:, :, lag][np.ix_(order, order)]
            D = R - S
            vlim = max(np.nanpercentile(np.abs(R), 99), 1e-6)
            for k, M in enumerate((R, S, D)):
                j = ci * 3 + k
                a = axes[i, j]
                im = a.imshow(M, cmap='RdBu_r', vmin=-vlim, vmax=vlim)
                for b in bounds:
                    a.axhline(b - .5, color='k', lw=.25, alpha=.3)
                    a.axvline(b - .5, color='k', lw=.25, alpha=.3)
                a.set_xticks([]); a.set_yticks([])
                if i == 0:
                    a.set_title(col_titles[j], fontsize=12)
                if j == 0:
                    a.set_yticks(centers)
                    a.set_yticklabels(netlab, fontsize=6.5)
                    a.set_ylabel(SN_SHORT[s], fontsize=12)
                cb = fig.colorbar(im, ax=a, fraction=0.046, pad=0.03)
                cb.ax.tick_params(labelsize=5.5)
                cb.set_ticks([-vlim, 0, vlim])
    save(fig, f'SuppFig{4 + lag:02d}_glm_sample_graphs_A{lag}')


if __name__ == '__main__':
    fig_regressors_nf()
    fig_regressors_dc()
    fig_summary()
    fig_sample_graphs(0)
    fig_sample_graphs(1)
    print('\nfigures written to', OUT)
