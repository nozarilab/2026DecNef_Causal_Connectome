"""Supplementary Fig. 16: control-hub analysis on non-overlapping graphs, brain and parcel view.

(a) Subnetwork edges stronger in NF than baseline on a glass brain, coloured by score
correlation; (b) parcel-level control-network zoom.
Output: Results/figures_supplementary/SuppFig16_nonoverlap_brain.{pdf,png}
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import networkx as nx
from nilearn import plotting
from make_suppfigs_22_25_alpha import two_step, CTRL
from make_fig3_alpha import per_graph_scores
import make_suppfigs_23_24_alpha_brainmaps as BM

SWEEP = 'Results/glm_revision/alpha_sweep/subnet_abar_0.05.npz'
FINAL = 'Results/figures_supplementary'


def main():
    d = np.load(SWEEP)
    nf16, dc16 = d['nf_weighted'].astype(float), d['dc_weighted'].astype(float)
    nfP, dcP = d['nf_parcel_w'].astype(float), d['dc_parcel_w'].astype(float)
    nfidx, dcidx = d['nf_idx'], d['dc_idx']
    scores = per_graph_scores()
    sc = np.array([scores.get((int(s), int(su), int(se), int(r)), np.nan) for s, su, se, r in nfidx])

    nfk = np.isin(nfidx[:, 3], [0, 2, 4, 6]); dck = np.isin(dcidx[:, 3], [0, 2, 4])
    nf16, dc16, nfP, dcP, sc = nf16[nfk], dc16[dck], nfP[nfk], dcP[dck], sc[nfk]
    ok = np.isfinite(sc)

    stronger, _, _, _, crm = two_step(nf16, dc16, nf16[ok], sc[ok])
    red = (stronger & (crm > 0)).astype(int)
    blue = (stronger & (crm < 0)).astype(int)
    gray = (stronger & (crm == 0)).astype(int)

    strongerP, _, _, _, crmP = two_step(nfP, dcP, nfP[ok], sc[ok])
    redP = (strongerP & (crmP > 0))[CTRL, CTRL].astype(int)
    grayP = (strongerP & (crmP == 0))[CTRL, CTRL].astype(int)

    fig = plt.figure(figsize=(20, 9))
    # ---- panel b: glass brain
    axb = fig.add_subplot(1, 2, 1)
    disp = plotting.plot_glass_brain(None, display_mode='z', figure=fig, axes=axb, annotate=False, alpha=0.12)
    ax = disp.axes['z'].ax
    ns = 1200
    for M, col in [(gray, 'gray'), (red, 'red'), (blue, 'blue')]:
        cross = M.copy(); np.fill_diagonal(cross, 0)
        G = nx.from_numpy_array(cross, create_using=nx.DiGraph)
        if G.number_of_edges():
            nx.draw_networkx_edges(G, pos=BM.POS, ax=ax, edge_color=col, width=3.2,
                                   node_size=ns, arrows=True, arrowsize=10)
    Gall = nx.from_numpy_array(((red + blue + gray) > 0).astype(int), create_using=nx.DiGraph)
    nx.draw_networkx_nodes(Gall, pos=BM.POS, ax=ax, node_color='lightgray', node_size=ns, nodelist=list(range(16)))
    for M, col in [(gray, 'gray'), (red, 'red'), (blue, 'blue')]:
        for n in np.where(np.diag(M) > 0)[0]:
            BM.draw_self_loop(ax, *BM.POS[n], col, r=9.0, nr=8.0)
    nx.draw_networkx_labels(Gall, pos=BM.POS, ax=ax, labels={i: BM.SHORT[i] for i in range(16)},
                            font_size=10, font_color='black', font_weight='bold')
    ax.add_patch(Rectangle((-20, 33), 40, 30, fill=False, ls='--', lw=2, color='brown'))
    ax.text(-15, 72, 'L', fontsize=15, ha='center', va='center', fontweight='bold')
    ax.text(15, 72, 'R', fontsize=15, ha='center', va='center', fontweight='bold')
    axb.set_title('(a)', loc='left', fontsize=20, fontweight='bold')

    # ---- panel c: control-network zoom
    axc = fig.add_subplot(1, 2, 2)
    Gred = nx.from_numpy_array(redP, create_using=nx.DiGraph)
    Ggray = nx.from_numpy_array(grayP, create_using=nx.DiGraph)
    nsz = 1500
    nx.draw_networkx_edges(Ggray, pos=BM.ZOOM_POS, ax=axc, edge_color='gray', width=2, node_size=nsz, arrows=True, arrowsize=12)
    nx.draw_networkx_edges(Gred, pos=BM.ZOOM_POS, ax=axc, edge_color='red', width=4, node_size=nsz, arrows=True, arrowsize=12)
    nx.draw_networkx_nodes(nx.DiGraph(), pos=BM.ZOOM_POS, ax=axc, node_color='lightgray', node_size=nsz,
                           nodelist=list(range(13)), linewidths=3)
    nx.draw_networkx_labels(nx.empty_graph(13), pos=BM.ZOOM_POS, ax=axc,
                            labels={i: BM.ZOOM_LBL[i] for i in range(13)}, font_size=11,
                            font_color='black', font_weight='bold')
    axc.margins(0.16); axc.set_axis_off()
    axc.set_title('(b)', loc='left', fontsize=20, fontweight='bold')

    fig.legend(handles=[Line2D([0], [0], color='red', lw=4, label='stronger in NF AND\n+ score correlation'),
                        Line2D([0], [0], color='blue', lw=4, label='stronger in NF AND\n$-$ score correlation'),
                        Line2D([0], [0], color='gray', lw=4, label='stronger in NF,\nno significant score correlation')],
               loc='lower center', ncol=3, fontsize=16, frameon=False,
               handlelength=2.2, columnspacing=2.5, borderaxespad=0.5)
    fig.subplots_adjust(bottom=0.14, wspace=0.02)
    os.makedirs(FINAL, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(FINAL, f'SuppFig16_nonoverlap_brain.{ext}'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'subnetwork: red(+score)={int(red.sum())} blue(-score)={int(blue.sum())} gray={int(gray.sum())}')
    print(f'parcel control hub: red={int(redP.sum())} gray={int(grayP.sum())}')
    print('saved', os.path.join(FINAL, 'SuppFig16_nonoverlap_brain.png'))


if __name__ == '__main__':
    main()
