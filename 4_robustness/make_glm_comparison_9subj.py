"""Compare CaLLTiF graphs with and without GLM task regression, for NF and DC sessions.

Computes, per graph, the correlation between raw and residual edge weights and the
overlap of significant edges, and writes the summary table used by make_suppfigs_01_05_glm.py.
Inputs:  Results/glm_revision/glm_results_9subj/ and Results/calltif_decnef_study{N}_{nf,dec}_two_runs_one_graph.npz
Outputs: Results/glm_revision/figures/glm9_comparison_summary.csv
"""
import os
import re
import glob
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESID = 'Results/glm_revision/glm_results_9subj'
RAWFMT = {'nf': 'Results/glm_revision/raw/calltif_decnef_study{s}_nf_two_runs_one_graph.npz',
          'dc': 'Results/calltif_decnef_study{s}_dec_two_runs_one_graph.npz'}
MAP = 'Results/glm_revision/parcel_mapping.npz'
FIG = 'Results/glm_revision/figures'
ALPHA = {1: 1e-2 / 12, 2: 1e-2 / 4, 3: 1e-2 / 4, 4: 1e-2 / 4, 5: 1e-2 / 4}
STUDY_NAME = {1: 'S1 facial pref', 2: 'S2 orient-color', 3: 'S3 fear-cc',
              4: 'S4 confidence', 5: 'S5 fear-reprog'}
CV = {'binary': '#2c7fb8', 'hrf': '#d95f02'}
_rawcache = {}


def raw_lm(cond, study):
    key = (cond, study)
    if key not in _rawcache:
        d = np.load(RAWFMT[cond].format(s=study), allow_pickle=True)
        _rawcache[key] = d[[k for k in d.files if 'link_matrix' in k][0]]
    return _rawcache[key]


def parse(b):
    m = re.search(r'(nf|dc)_(binary|hrf)_9subj_study(\d+)_subj(\d+)_ses(\d+)_pair(\d+)', b)
    return m.group(1), m.group(2), int(m.group(3)), int(m.group(4)), int(m.group(5)), int(m.group(6))


def paper_order():
    m = np.load(MAP, allow_pickle=True)
    idx = [np.asarray(m[k]) for k in m['net_labels']]
    order = np.concatenate(idx)
    return order, list(np.cumsum([len(k) for k in idx])[:-1])


def collect():
    """Per residual graph: match raw, compute lag0/lag1 edge corr + Jaccard."""
    rows = []
    for f in sorted(glob.glob(os.path.join(RESID, '*_9subj_*.npz'))):
        cond, variant, study, subj, ses, pair = parse(os.path.basename(f))
        rv = np.load(f)
        lm = raw_lm(cond, study)
        e = lm[subj][ses][pair]
        if not isinstance(e, dict):
            continue
        rawv = np.array(e['val_matrix']); rawp = np.array(e['p_matrix'])
        rv_val = rv['val_matrix']; rv_p = rv['p_matrix']
        a = ALPHA[study]
        out = {'cond': cond, 'variant': variant, 'study': study, 'subj': subj, 'ses': ses, 'pair': pair}
        for lag in (0, 1):
            R, S = rawv[:, :, lag].ravel(), rv_val[:, :, lag].ravel()
            ok = np.isfinite(R) & np.isfinite(S)
            out[f'r{lag}'] = np.corrcoef(R[ok], S[ok])[0, 1] if ok.sum() > 2 else np.nan
            rb, sb = (rawp[:, :, lag] <= a), (rv_p[:, :, lag] <= a)
            inter = (rb & sb).sum(); uni = (rb | sb).sum()
            out[f'jac{lag}'] = inter / uni if uni > 0 else 1.0
        rows.append(out)
    return rows


def fig_summary(rows):
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    for ci, cond in enumerate(('nf', 'dc')):
        # edge-weight correlation (lag0) distribution by variant
        a = ax[ci, 0]
        for v in ('binary', 'hrf'):
            vals = [r['r0'] for r in rows if r['cond'] == cond and r['variant'] == v]
            a.hist(vals, bins=40, alpha=0.6, color=CV[v], label=f'{v} (med {np.median(vals):.3f})')
        a.set_title(f'{cond.upper()}: edge-weight corr raw-vs-residual (lag0)'); a.legend(fontsize=8)
        a.set_xlabel('Pearson r'); a.set_xlim(0.7, 1.0)
        # Jaccard (lag0) of significant edges
        a = ax[ci, 1]
        for v in ('binary', 'hrf'):
            vals = [r['jac0'] for r in rows if r['cond'] == cond and r['variant'] == v]
            a.hist(vals, bins=40, alpha=0.6, color=CV[v], label=f'{v} (med {np.median(vals):.3f})')
        a.set_title(f'{cond.upper()}: Jaccard of significant edges (lag0)'); a.legend(fontsize=8)
        a.set_xlabel('Jaccard')
    fig.suptitle('Raw vs GLM task-regressed graphs (9 subjects per study), NF and DC', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(FIG, f'glm9_summary.{ext}'), dpi=130, bbox_inches='tight')
    print('saved glm9_summary')


def fig_examples(rows, cond, variant='hrf'):
    order, bounds = paper_order()
    picks = [r for r in rows if r['cond'] == cond and r['variant'] == variant and r['subj'] == 0 and r['pair'] == 0]
    picks = sorted(picks, key=lambda r: r['study'])[:5]
    fig, ax = plt.subplots(len(picks), 6, figsize=(20, 3.3 * len(picks)), squeeze=False)
    for row, pk in enumerate(picks):
        s = pk['study']
        f = os.path.join(RESID, f"{cond}_{variant}_9subj_study{s}_subj0_ses0_pair0.npz")
        resid = np.load(f)['val_matrix']
        e = raw_lm(cond, s)[0][0][0]; raw = np.array(e['val_matrix'])
        ro = raw[np.ix_(order, order)]; rn = resid[np.ix_(order, order)]
        for lag in (0, 1):
            R, S = ro[:, :, lag], rn[:, :, lag]; D = R - S
            vlim = max(np.nanpercentile(np.abs(R), 99), 1e-6)        # SAME scale for all three
            for c, (M, ttl) in enumerate([(R, 'raw'), (S, f'{variant}-resid'), (D, 'diff (same scale)')]):
                aa = ax[row, lag * 3 + c]
                im = aa.imshow(M, cmap='RdBu_r', vmin=-vlim, vmax=vlim)
                for bd in bounds:
                    aa.axhline(bd - .5, color='k', lw=.2, alpha=.3); aa.axvline(bd - .5, color='k', lw=.2, alpha=.3)
                aa.set_xticks([]); aa.set_yticks([])
                cb = fig.colorbar(im, ax=aa, fraction=.046, pad=.03); cb.ax.tick_params(labelsize=5); cb.set_ticks([-vlim, 0, vlim])
                if row == 0:
                    aa.set_title(f'lag{lag} {ttl}', fontsize=9)
        ax[row, 0].set_ylabel(f'{STUDY_NAME[s]}\nr0={pk["r0"]:.3f}', fontsize=7)
    fig.suptitle(f'{cond.upper()} example graphs (9 subj, {variant}): raw | residual | diff (diff on same scale)', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(FIG, f'glm9_examples_{cond}_{variant}.{ext}'), dpi=130, bbox_inches='tight')
    print(f'saved glm9_examples_{cond}_{variant}')


def main():
    os.makedirs(FIG, exist_ok=True)
    rows = collect()
    print(f'matched {len(rows)} residual-vs-raw graph pairs')
    with open(os.path.join(FIG, 'glm9_comparison_summary.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    # headline medians
    for cond in ('nf', 'dc'):
        for v in ('binary', 'hrf'):
            sub = [r for r in rows if r['cond'] == cond and r['variant'] == v]
            print(f'  {cond} {v}: n={len(sub)} | r0 med={np.median([r["r0"] for r in sub]):.4f} '
                  f'r1 med={np.median([r["r1"] for r in sub]):.4f} '
                  f'jac0 med={np.median([r["jac0"] for r in sub]):.3f}')
    fig_summary(rows)
    for cond in ('nf', 'dc'):
        fig_examples(rows, cond, 'hrf')


if __name__ == '__main__':
    main()
