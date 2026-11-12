"""Supplementary Figs. 19-20: two-stage discovery-test analysis, brain and parcel view.

Discovery on NF session 1 (candidates ranked by NF > baseline), test on sessions 2-3;
the top 10 subnetwork edges and all within-control parcel candidates are tested, with FDR
within each family.
  mode='graph' -> SuppFig19_twostage_brain                non-overlapping graphs
  mode='mean'  -> SuppFig20_twostage_subject_mean_brain   graphs averaged per subject
Output folder: Results/figures_supplementary/
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import networkx as nx
from scipy.stats import ranksums, spearmanr, wilcoxon
from statsmodels.stats.multitest import fdrcorrection
from nilearn import plotting
from make_fig3_alpha import per_graph_scores
import make_suppfigs_23_24_alpha_brainmaps as BM
import warnings
warnings.filterwarnings('ignore')

SWEEP = 'Results/glm_revision/alpha_sweep/subnet_abar_0.05.npz'
FINAL = 'Results/figures_supplementary'
CB = list(range(38, 51))
K = 10


def load(mode):
    d = np.load(SWEEP)
    nf16, dc16 = d['nf_weighted'].astype(float), d['dc_weighted'].astype(float)
    nfP, dcP = d['nf_parcel_w'].astype(float), d['dc_parcel_w'].astype(float)
    nfidx, dcidx = d['nf_idx'], d['dc_idx']
    sc = np.array([per_graph_scores().get((int(s), int(su), int(se), int(r)), np.nan) for s, su, se, r in nfidx])
    ses = nfidx[:, 2]
    nsub = np.array([f'{int(s)}_{int(su)}' for s, su, _, _ in nfidx])
    dsub = np.array([f'{int(s)}_{int(su)}' for s, su, _, _ in dcidx])
    if mode == 'graph':
        nf_ok, dc_ok = np.isin(nfidx[:, 3], [0, 2, 4, 6]), np.isin(dcidx[:, 3], [0, 2, 4])
    else:
        nf_ok, dc_ok = np.ones(len(nfidx), bool), np.ones(len(dcidx), bool)
    disc = (ses == 0) & nf_ok; okd = np.isfinite(sc[disc])
    test = np.isin(ses, [1, 2]) & nf_ok & np.isfinite(sc)
    D = dict(scd=sc[disc], okd=okd, nfd16=nf16[disc], DCd16=dc16[dc_ok], nfdP=nfP[disc], DCdP=dcP[dc_ok], mode=mode)
    if mode == 'graph':
        D.update(NFt16=nf16[test], DCt16=dc16[dc_ok], NFtP=nfP[test], DCtP=dcP[dc_ok], Y=sc[test])
    else:
        units = sorted(set(nsub[test]))
        av = lambda X: np.array([X[test & (nsub == u)].mean(0) for u in units])
        avd = lambda X: np.array([X[dc_ok & (dsub == u)].mean(0) for u in units])
        D.update(NFt16=av(nf16), DCt16=avd(dc16), NFtP=av(nfP), DCtP=avd(dcP),
                 Y=np.array([np.nanmean(sc[test & (nsub == u)]) for u in units]))
    return D


def two_step(NFd, DCd, scd, okd, NFt, DCt, Y, edges, mode, K=None):
    cand = []
    for e in edges:
        ms = ranksums(NFd[:, e[0], e[1]], DCd[:, e[0], e[1]]).statistic
        col = NFd[okd, e[0], e[1]]
        rho = spearmanr(col, scd[okd]).statistic if col.std() > 0 else 0.0
        if ms > 0 and rho > 0:
            cand.append((ms, e))
    cand.sort(reverse=True, key=lambda x: x[0])
    sel = cand[:K] if K else cand
    pmod, psc = [], []
    for ms, e in sel:
        a, b = NFt[:, e[0], e[1]], DCt[:, e[0], e[1]]
        if mode == 'mean':
            try:
                pm = wilcoxon(a, b).pvalue if (a - b).std() > 0 else 1.0
            except Exception:
                pm = 1.0
        else:
            m = ranksums(a, b); pm = m.pvalue if m.statistic > 0 else 1.0
        pmod.append(pm); psc.append(spearmanr(a, Y)[1])
    smod = fdrcorrection(pmod)[0] if pmod else np.array([])
    ssc = fdrcorrection(psc)[0] if psc else np.array([])
    red = [e for (ms, e), x, y in zip(sel, smod, ssc) if x and y]
    gray = [e for (ms, e), x, y in zip(sel, smod, ssc) if x and not y]
    return red, gray


def build(mode, fname):
    D = load(mode)
    sred, sgray = two_step(D['nfd16'], D['DCd16'], D['scd'], D['okd'], D['NFt16'], D['DCt16'], D['Y'],
                           [(i, j) for i in range(16) for j in range(16)], mode, K=K)
    zred, zgray = two_step(D['nfdP'], D['DCdP'], D['scd'], D['okd'], D['NFtP'], D['DCtP'], D['Y'],
                           [(i, j) for i in CB for j in CB if i != j], mode)
    off = {c: k for k, c in enumerate(CB)}
    Rs = np.zeros((16, 16), int); Gs = np.zeros((16, 16), int)
    for i, j in sred:
        Rs[i, j] = 1
    for i, j in sgray:
        Gs[i, j] = 1
    Rz = np.zeros((13, 13), int); Gz = np.zeros((13, 13), int)
    for i, j in zred:
        Rz[off[i], off[j]] = 1
    for i, j in zgray:
        Gz[off[i], off[j]] = 1
    from make_suppfigs_22_25_alpha import VIS
    print(f'[{mode}] subnet HUB={[f"{VIS[i]}->{VIS[j]}" for i, j in sred]}  zoom hub edges={len(zred)}')

    fig = plt.figure(figsize=(20, 9))
    # ---- (a) glass brain
    axb = fig.add_subplot(1, 2, 1)
    disp = plotting.plot_glass_brain(None, display_mode='z', figure=fig, axes=axb, annotate=False, alpha=0.12)
    ax = disp.axes['z'].ax; ns = 1200
    for M, col in [(Gs, 'gray'), (Rs, 'red')]:
        cross = M.copy(); np.fill_diagonal(cross, 0)
        G = nx.from_numpy_array(cross, create_using=nx.DiGraph)
        if G.number_of_edges():
            nx.draw_networkx_edges(G, pos=BM.POS, ax=ax, edge_color=col, width=3.2, node_size=ns, arrows=True, arrowsize=10)
    nx.draw_networkx_nodes(nx.empty_graph(16), pos=BM.POS, ax=ax, node_color='lightgray', node_size=ns, nodelist=list(range(16)))
    for M, col in [(Gs, 'gray'), (Rs, 'red')]:
        for n in np.where(np.diag(M) > 0)[0]:
            BM.draw_self_loop(ax, *BM.POS[n], col, r=9.0, nr=8.0)
    nx.draw_networkx_labels(nx.empty_graph(16), pos=BM.POS, ax=ax, labels={i: BM.SHORT[i] for i in range(16)}, font_size=10, font_weight='bold')
    ax.add_patch(Rectangle((-20, 33), 40, 30, fill=False, ls='--', lw=2, color='brown'))
    ax.text(-15, 72, 'L', fontsize=15, ha='center', va='center', fontweight='bold')
    ax.text(15, 72, 'R', fontsize=15, ha='center', va='center', fontweight='bold')
    axb.set_title('(a)', loc='left', fontsize=20, fontweight='bold')
    # ---- (b) control zoom
    axc = fig.add_subplot(1, 2, 2)
    Gred = nx.from_numpy_array(Rz, create_using=nx.DiGraph)
    Ggray = nx.from_numpy_array(Gz, create_using=nx.DiGraph)
    nsz = 1500
    nx.draw_networkx_edges(Ggray, pos=BM.ZOOM_POS, ax=axc, edge_color='gray', width=2, node_size=nsz, arrows=True, arrowsize=12)
    nx.draw_networkx_edges(Gred, pos=BM.ZOOM_POS, ax=axc, edge_color='red', width=4, node_size=nsz, arrows=True, arrowsize=12)
    nx.draw_networkx_nodes(nx.empty_graph(13), pos=BM.ZOOM_POS, ax=axc, node_color='lightgray', node_size=nsz, nodelist=list(range(13)), linewidths=3)
    nx.draw_networkx_labels(nx.empty_graph(13), pos=BM.ZOOM_POS, ax=axc, labels={i: BM.ZOOM_LBL[i] for i in range(13)}, font_size=11, font_weight='bold')
    axc.margins(0.16); axc.set_axis_off()
    axc.set_title('(b)', loc='left', fontsize=20, fontweight='bold')
    fig.legend(handles=[Line2D([0], [0], color='red', lw=4, label='stronger in NF AND\n+ score correlation'),
                        Line2D([0], [0], color='gray', lw=4, label='stronger in NF,\nno significant score correlation')],
               loc='lower center', ncol=2, fontsize=16, frameon=False,
               handlelength=2.2, columnspacing=2.5, borderaxespad=0.5)
    fig.subplots_adjust(bottom=0.14, wspace=0.02)
    os.makedirs(FINAL, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(FINAL, f'{fname}.{ext}'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('saved', fname)


if __name__ == '__main__':
    build('graph', 'SuppFig19_twostage_brain')
    build('mean', 'SuppFig20_twostage_subject_mean_brain')
