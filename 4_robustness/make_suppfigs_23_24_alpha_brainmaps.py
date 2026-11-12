"""Supplementary Figs. 23-24: brain-network and parcel views of the control hub across alpha-bar.

  SuppFig23_alpha_hub_brainmaps   edges stronger in NF than baseline on a glass brain, coloured
                                  by their score correlation (red +, blue -, gray n.s.)
  SuppFig24_alpha_hub_zoom        parcel-level control-network hub
Output folder: Results/figures_supplementary/
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.path import Path
import networkx as nx
from nilearn import plotting
from make_suppfigs_22_25_alpha import load, two_step, ABAR
from make_fig3_alpha import per_graph_scores

FINAL = 'Results/figures_supplementary'
# 16 network node coordinates (as in Figure 3b), same L/R-interleaved order as the graphs
NODE_LOCS = np.array([[-15, -40], [15, -40], [-22, -90], [22, -90], [-35, -60], [35, -60],
                      [-10, 50], [10, 50], [-30, 30], [30, 30], [-40, 0], [40, 0],
                      [-10, 0], [10, 0], [-35, -20], [35, -20]], dtype=float)
SHORT = ['Sub', 'Sub', 'Vis', 'Vis', 'Lim', 'Lim', 'Cont', 'Cont', 'DMN', 'DMN',
         'Vent-Att', 'Vent-Att', 'Dors-Att', 'Dors-Att', 'Som-Mot', 'Som-Mot']
POS = {i: (NODE_LOCS[i, 0], NODE_LOCS[i, 1]) for i in range(16)}


def draw_self_loop(ax, x, y, col, r=7.0, nr=6.0):
    """Directed self-loop drawn as a half circle (semicircle dome) with an arrowhead into
    the node, matching Figure 3b. networkx self-loops size in points and vanish
    against the ~150 mm glass brain, so we draw the arc explicitly in MNI-mm coordinates."""
    th = np.linspace(0, np.pi, 60)                       # upper half circle only
    pts = np.c_[x + r * np.cos(th), (y + nr) + r * np.sin(th)]
    ax.add_patch(FancyArrowPatch(path=Path(pts), arrowstyle='-|>', mutation_scale=15,
                                 color=col, lw=3.0, zorder=6))


def edge_sets(a, scores):
    d = load(a); nf16, dc16 = d['nf_weighted'], d['dc_weighted']; idx = d['nf_idx']
    sc = np.array([scores.get((int(s), int(su), int(se), int(r)), np.nan) for s, su, se, r in idx])
    ok = np.isfinite(sc)
    stronger, _, _, _, crm = two_step(nf16, dc16, nf16[ok], sc[ok])   # crm = spearman where score-significant
    red = stronger & (crm > 0)        # stronger in NF AND + score
    blue = stronger & (crm < 0)       # stronger in NF AND - score
    gray = stronger & (crm == 0)      # stronger in NF, score not significant
    return red.astype(int), blue.astype(int), gray.astype(int)


# ---- zoomed parcel-level control-network graph (as in Figure 3c) ----
CTRL = slice(38, 51)
# fixed node layout and labels, identical to Figure 3c
ZOOM_POS = {0: (-0.5, -1.5), 1: (-0.5, 2.2), 2: (0.0, -0.3), 3: (0.1, 0.6), 4: (1.5, -1.5),
            5: (1.5, -2.0), 6: (1.3, 2.2), 7: (1.26, 1.15), 8: (1.06, 1.2), 9: (1.22, 0.75),
            10: (0.7, 0.6), 11: (0.7, 1.7), 12: (0.8, -0.3)}
ZOOM_LBL = ['Par', 'PFCl', 'pCun', 'Cing', 'Par_1', 'Par_2', 'PFCl_1', 'PFCl_2',
            'PFCl_3', 'PFCl_4', 'Cing', 'PFCmp', 'pCun']


def parcel_edge_sets(a, scores):
    d = load(a); nfP, dcP = d['nf_parcel_w'].astype(float), d['dc_parcel_w'].astype(float)
    idx = d['nf_idx']
    sc = np.array([scores.get((int(s), int(su), int(se), int(r)), np.nan) for s, su, se, r in idx])
    ok = np.isfinite(sc)
    stronger, _, _, _, crm = two_step(nfP, dcP, nfP[ok], sc[ok])
    red = (stronger & (crm > 0))[CTRL, CTRL]     # stronger in NF AND + score (help)
    gray = (stronger & (crm == 0))[CTRL, CTRL]   # stronger in NF, no score effect
    return red.astype(int), gray.astype(int)


def zoom_fig():
    scores = per_graph_scores()
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    axes = axes.ravel()
    for k, a in enumerate(ABAR):
        red, gray = parcel_edge_sets(a, scores)
        ax = axes[k]
        Gred = nx.from_numpy_array(red, create_using=nx.DiGraph)
        Ggray = nx.from_numpy_array(gray, create_using=nx.DiGraph)
        # as in Figure 3c: gray thin, red thick, arrows; nodes lightgray; fixed positions
        nsz = 1500
        nx.draw_networkx_edges(Ggray, pos=ZOOM_POS, ax=ax, edge_color='gray', width=2,
                               node_size=nsz, arrows=True, arrowsize=12)
        nx.draw_networkx_edges(Gred, pos=ZOOM_POS, ax=ax, edge_color='red', width=4,
                               node_size=nsz, arrows=True, arrowsize=12)
        nx.draw_networkx_nodes(nx.DiGraph(), pos=ZOOM_POS, ax=ax, node_color='lightgray',
                               node_size=nsz, nodelist=list(range(13)), linewidths=3)
        nx.draw_networkx_labels(nx.empty_graph(13), pos=ZOOM_POS, ax=ax,
                                labels={i: ZOOM_LBL[i] for i in range(13)},
                                font_size=11, font_color='black', font_weight='bold')
        ax.margins(0.16)                                 # padding so node markers are not clipped
        ax.set_title(f'$\\bar\\alpha$ = {a}', fontsize=20, pad=12)
        ax.set_axis_off()
        print(f'abar={a}: control-parcel zoom  red(+score)={int(red.sum())}  gray(ns)={int(gray.sum())}')
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([0], [0], color='red', lw=4, label='stronger in NF AND + score correlation'),
                        Line2D([0], [0], color='gray', lw=2, label='stronger in NF, no significant score correlation')],
               loc='lower center', ncol=2, fontsize=13, frameon=False)
    fig.subplots_adjust(bottom=0.07, wspace=0.03, hspace=0.12)
    os.makedirs(FINAL, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(FINAL, f'SuppFig24_alpha_hub_zoom.{ext}'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('saved', os.path.join(FINAL, 'SuppFig24_alpha_hub_zoom.png'))


def main():
    scores = per_graph_scores()
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    for k, a in enumerate(ABAR):
        red, blue, gray = edge_sets(a, scores)
        # draw everything into nilearn's cut-axes (MNI mm), as for Figure 3b,
        # so node coordinates land on the brain rather than in a plain (0,1) axes
        disp = plotting.plot_glass_brain(None, display_mode='z', figure=fig, axes=axes[k],
                                         annotate=False, alpha=0.12)
        ax = disp.axes['z'].ax
        ns = 700
        # draw cross-node edges (self-loops handled separately as rings below)
        for M, col in [(gray, 'gray'), (red, 'red'), (blue, 'blue')]:
            cross = M.copy(); np.fill_diagonal(cross, 0)
            G = nx.from_numpy_array(cross, create_using=nx.DiGraph)
            if G.number_of_edges():
                nx.draw_networkx_edges(G, pos=POS, ax=ax, edge_color=col, width=3.2,
                                       node_size=ns, arrows=True, arrowsize=10)
        Gall = nx.from_numpy_array(((red + blue + gray) > 0).astype(int), create_using=nx.DiGraph)
        nx.draw_networkx_nodes(Gall, pos=POS, ax=ax, node_color='lightgray', node_size=ns,
                               nodelist=list(range(16)))
        # directed self-loops on top of the node (gray under red under blue)
        for M, col in [(gray, 'gray'), (red, 'red'), (blue, 'blue')]:
            for n in np.where(np.diag(M) > 0)[0]:
                draw_self_loop(ax, *POS[n], col)
        nx.draw_networkx_labels(Gall, pos=POS, ax=ax,
                                labels={i: SHORT[i] for i in range(16)}, font_size=7,
                                font_color='black', font_weight='bold')
        # dashed box around the control network (Cont nodes at y~50)
        ax.add_patch(Rectangle((-20, 33), 40, 30, fill=False, ls='--', lw=2, color='brown'))
        ax.text(-15, 72, 'L', fontsize=15, ha='center', va='center', fontweight='bold')
        ax.text(15, 72, 'R', fontsize=15, ha='center', va='center', fontweight='bold')
        ax.set_title(f'$\\bar\\alpha$ = {a}', fontsize=20)
        nr = int(red.sum()); nb = int(blue.sum()); ng = int(gray.sum())
        print(f'abar={a}: stronger-in-NF edges  red(+score)={nr}  blue(-score)={nb}  gray(ns)={ng}')
    # legend
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([0], [0], color='red', lw=3, label='+ score correlation'),
                        Line2D([0], [0], color='blue', lw=3, label='$-$ score correlation'),
                        Line2D([0], [0], color='gray', lw=3, label='no significant score correlation')],
               loc='lower center', ncol=3, fontsize=13, frameon=False)
    fig.subplots_adjust(bottom=0.08, wspace=0.02, hspace=0.12)
    os.makedirs(FINAL, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(FINAL, f'SuppFig23_alpha_hub_brainmaps.{ext}'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('saved', os.path.join(FINAL, 'SuppFig23_alpha_hub_brainmaps.png'))


if __name__ == '__main__':
    main()
    zoom_fig()
