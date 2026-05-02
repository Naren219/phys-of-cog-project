
from connectivity import lagged_corr_2d
import networkx as nx
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import statsmodels.stats.correlation_tools as correlation_tools
import statsmodels.tsa.stattools as stattools
#from scikit_rmt.ensemble.spectral_law import MarchenkoPasturDistribution

from utils import load_eeg_data
from connectivity import (
    select_p2_500ms,
    sliding_corr_from_epochs,
    summary_metrics,
    lagged_corr_2d,
    sliding_lagged_diff_corr_from_epochs
)

SUBJECT = 5
TMIN, TMAX = None, None       # native epoch window (must match across files)
WINDOW_MS = 200
STEP_MS = 50
FREQ_BAND = None              # e.g. 'alpha' / 'beta' for Cell G

blt = load_eeg_data('BLT', SUBJECT, tmin=TMIN, tmax=TMAX, freq_band=FREQ_BAND, normalize=True)
p1_first_10  = load_eeg_data('P1',  SUBJECT, trials = list(range(10)) , tmin=TMIN, tmax=TMAX, freq_band=FREQ_BAND, normalize=True)
p1_last_10 = load_eeg_data('P1',  SUBJECT, trials = list(range(-10,0)) , tmin=TMIN, tmax=TMAX, freq_band=FREQ_BAND, normalize=True)
p1 = load_eeg_data('P1',  SUBJECT, tmin=TMIN, tmax=TMAX, freq_band=FREQ_BAND, normalize=True)
p2_all = load_eeg_data('P2', SUBJECT, tmin=TMIN, tmax=TMAX, freq_band=FREQ_BAND, normalize=True)
p2 = select_p2_500ms(p2_all)
p2_first_10 = load_eeg_data('P2',  SUBJECT, trials = list(range(10)) , tmin=TMIN, tmax=TMAX, freq_band=FREQ_BAND, normalize=True)
p2_last_10 = load_eeg_data('P2',  SUBJECT, trials = list(range(-10,0)) , tmin=TMIN, tmax=TMAX, freq_band=FREQ_BAND, normalize=True)

assert np.allclose(blt.times, p1.times), 'BLT and P1 time axes differ'
assert np.allclose(p1.times, p2.times), 'P1 and P2 time axes differ'

# Channel-set alignment: intersect across files so the NxN matrices line up.
common_ch = [c for c in blt.ch_names if c in p1.ch_names and c in p2.ch_names]
for ep in (blt, p1, p2):
    ep.pick_channels(common_ch, ordered=True)
    
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output

LAGS_SEC = np.linspace(-0.05, 0.05, 81)

conds = {'P1': p1, "P1 First 10": p1_first_10, "P1 Last 10": p1_last_10,
         'P2 First 10': p2_first_10, 'P2 Last 10': p2_last_10}
results = {}
for name, ep in conds.items():
    diff_corr, raw_corr, centers, ch_names = sliding_lagged_diff_corr_from_epochs(
        ep, WINDOW_MS, STEP_MS, LAGS_SEC
    )
    results[name] = {
        'corr':     diff_corr,
        'raw_corr': raw_corr,
        'centers':  centers,
        'ch_names': ch_names,
    }
    print(f'{name}: corr {diff_corr.shape}  centers [{centers[0]:.3f}s .. {centers[-1]:.3f}s]')

cond_names = list(results.keys())

all_data = np.concatenate([results[n]['corr'] for n in cond_names])
clim     = np.nanpercentile(np.abs(all_data), 99)

frontal = ['FPZ', 'AF7', 'AF8', 'AF3', 'AF4', 'F3', 'FZ', 'F4']
frontocentral = ['FC5', 'FC1', 'FC2', 'FC6']
somatosensory = ['C3', 'Cz', 'C4']
centroparietal = ['CP3', 'CP1', 'CPZ', 'CP2', 'CP4']
temporal = ['T7', 'T8', 'TP7', 'TP8']
parietal_occipital = ['P5', 'P1', 'P2', 'P6', 'POz', 'O1', 'Oz', 'O2']

groups = {"frontal": frontal, "frontocentral": frontocentral, "somatosensory": somatosensory,
          "centroparietal": centroparietal, "temporal": temporal, "parietal_occipital": parietal_occipital}
colordict = {"frontal": "red", "frontocentral": "orange", "somatosensory": "yellow",
             "centroparietal": "green", "temporal": "blue", "parietal_occipital": "purple"}

CH_TO_GROUP = {ch: g for g, chs in groups.items() for ch in chs}


def initialize_graph(epochs):
    graph = nx.Graph()
    for i, ch in enumerate(epochs.ch_names):
        group = CH_TO_GROUP.get(ch, "")
        graph.add_node(i, color=colordict[group], name=ch, group=group)
    return graph


def build_corr_graph(epochs, mask, weights, threshold=0.2, use_abs=True):
    g = initialize_graph(epochs)
    mask = np.array(mask)
    weights = np.array(weights)
    n = len(epochs.ch_names)
    iu, ju = np.triu_indices(n, k=1)
    keep = mask[iu, ju] > threshold
    w = weights[iu[keep], ju[keep]]
    if use_abs:
        w = np.abs(w)
    g.add_weighted_edges_from(zip(iu[keep].tolist(), ju[keep].tolist(), w.tolist()))
    return g

lag_slider = widgets.SelectionSlider(
    options=[(f'{l:+.3f}s', i) for i, l in enumerate(LAGS_SEC)],
    description='Lag:',
    layout=widgets.Layout(width='700px'),
)

out = widgets.Output()


def redraw(epoch: str, change=None):
    diff_corr = results[epoch]['corr']      # (n_windows, n_lags, n_ch, n_ch)
    raw_corr  = results[epoch]['raw_corr']  # (n_windows, n_lags, n_ch, n_ch)
    ch_names  = results[epoch]['ch_names']

    lag_idx = lag_slider.value
    lag     = LAGS_SEC[lag_idx]

    # ── Global optimal lag ────────────────────────────────────────────────────
    temporal_variance   = raw_corr.var(axis=0)            # (n_lags, n_ch, n_ch)
    global_variance     = temporal_variance.mean(axis=(1, 2))  # (n_lags,)
    best_lag_global_idx = global_variance.argmin()
    best_lag_global_sec = LAGS_SEC[best_lag_global_idx]

    # ── Per-pair optimal lag ──────────────────────────────────────────────────
    best_lag_idx_pp  = temporal_variance.argmin(axis=0)   # (n_ch, n_ch)
    best_lag_sec_pp  = LAGS_SEC[best_lag_idx_pp]

    raw_corr_mean    = raw_corr.mean(axis=0)              # (n_lags, n_ch, n_ch)
    best_lag_corr_pp = np.take_along_axis(
        raw_corr_mean, best_lag_idx_pp[np.newaxis], axis=0
    ).squeeze(0)
    best_lag_var_pp  = np.take_along_axis(
        temporal_variance, best_lag_idx_pp[np.newaxis], axis=0
    ).squeeze(0)

    threshold     = np.percentile(best_lag_var_pp, 50)
    graph_weights = np.where(best_lag_var_pp < threshold, best_lag_corr_pp, 0)

    # Time-averaged correlation at global optimal lag
    corr_at_global = raw_corr_mean[best_lag_global_idx]   # (n_ch, n_ch)

    # ── Nearest correlation matrix (Higham) ───────────────────────────────────
    def symmetrize_matrix(A):
        return (A + A.T) / 2

    def project_to_positive_semidefinite(A):
        eigenvalues, eigenvectors = np.linalg.eigh(A)
        A_psd = (eigenvectors * np.maximum(eigenvalues, 0)).dot(eigenvectors.T)
        return symmetrize_matrix(A_psd)

    def nearest_correlation_matrix(A, tol=1e-8, max_iterations=100):
        X = symmetrize_matrix(A)
        correction_matrix = np.zeros_like(X)
        for _ in range(max_iterations):
            X_old = X.copy()
            residual = X - correction_matrix
            X = project_to_positive_semidefinite(residual)
            correction_matrix = X - residual
            np.fill_diagonal(X, 1)
            if np.linalg.norm(X - X_old, 'fro') / np.linalg.norm(X, 'fro') < tol:
                break
        return X

    higham_corr = nearest_correlation_matrix(best_lag_corr_pp)

    lambda_max = 4
    x          = np.linspace(-3, 8, 500)
    def mpdist(lambd):
        return np.sqrt((lambda_max - lambd) * lambd) / (2 * np.pi * lambd)

    raw_eig           = np.linalg.eig(corr_at_global)
    normal_eig        = np.linalg.eig(best_lag_corr_pp)
    lag_corrected_eig = np.linalg.eig(higham_corr)
    corrected_corr    = correlation_tools.corr_clipped(corr_at_global, threshold=lambda_max)

    # ── Plot ──────────────────────────────────────────────────────────────────
    with out:
        clear_output(wait=True)
        fig, axes = plt.subplots(3, 3, figsize=(18, 12), constrained_layout=True)

        # [0,0] Δ Corr averaged over all time at selected lag
        mean_diff_at_lag = results['P1']['corr'][:, lag_idx].mean(axis=0)
        im0 = axes[0, 0].imshow(mean_diff_at_lag, vmin=-clim, vmax=clim,
                                 cmap='RdBu_r', aspect='equal')
        axes[0, 0].set_title(f'Δ Corr (time-avg)  |  lag={lag:+.3f}s')
        fig.colorbar(im0, ax=axes[0, 0], label='Δ Pearson r', shrink=0.8)

        # [0,1] Variance curve across lags
        axes[0, 1].plot(LAGS_SEC * 1000, global_variance, color='steelblue')
        axes[0, 1].axvline(best_lag_global_sec * 1000, color='r', linestyle='--',
                           label=f'optimal = {best_lag_global_sec*1000:+.1f}ms')
        axes[0, 1].axvline(lag * 1000, color='orange', linestyle=':',
                           label=f'selected = {lag*1000:+.1f}ms')
        axes[0, 1].set_xlabel('Lag (ms)')
        axes[0, 1].set_ylabel('Mean temporal variance')
        axes[0, 1].set_title('Stability across lags (global)')
        axes[0, 1].legend(fontsize=9)

        # [0,2] Correlation at global optimal lag (time-averaged)
        vmax_g = np.abs(corr_at_global).max()
        im2 = axes[0, 2].imshow(corr_at_global, cmap='RdBu_r', aspect='equal',
                                  vmin=-vmax_g, vmax=vmax_g)
        axes[0, 2].set_title(f'Corr at global optimal lag ({best_lag_global_sec*1000:+.1f}ms, time-avg)')
        fig.colorbar(im2, ax=axes[0, 2], label='Pearson r', shrink=0.8)

        # [1,0] Per-pair optimal lag map
        im3 = axes[1, 0].imshow(best_lag_sec_pp, cmap='RdBu_r', aspect='equal',
                                  vmin=-LAGS_SEC.max(), vmax=LAGS_SEC.max())
        axes[1, 0].set_title('Per-pair optimal lag (s)')
        fig.colorbar(im3, ax=axes[1, 0], label='lag (s)', shrink=0.8)

        # [1,1] Correlation at per-pair optimal lag (time-averaged)
        vmax_pp = np.abs(best_lag_corr_pp).max()
        im4 = axes[1, 1].imshow(best_lag_corr_pp, cmap='RdBu_r', aspect='equal',
                                  vmin=-vmax_pp, vmax=vmax_pp)
        axes[1, 1].set_title('Corr at per-pair optimal lag (time-avg)')
        fig.colorbar(im4, ax=axes[1, 1], label='Pearson r', shrink=0.8)

        # [1,2] Graph weights (thresholded)
        vmax_w = np.abs(graph_weights).max() or 1
        im5 = axes[1, 2].imshow(graph_weights, cmap='RdBu_r', aspect='equal',
                                  vmin=-vmax_w, vmax=vmax_w)
        axes[1, 2].set_title('Graph weights (thresholded to 50th percentile)')
        fig.colorbar(im5, ax=axes[1, 2], label='Pearson r', shrink=0.8)

        # [2,0] Eigenvalue distributions
        axes[2, 0].hist(normal_eig[0],        bins=25, density=True, label='Lag-corrected')
        axes[2, 0].hist(lag_corrected_eig[0], bins=25, density=True, label='Higham')
        axes[2, 0].hist(raw_eig[0],           bins=25, density=True, label='Raw')
        axes[2, 0].plot(x, mpdist(x), label='MP distribution')
        axes[2, 0].legend()
        axes[2, 0].set_title('Eigenvalue distributions')

        # [2,1] Corrected graph weights
        vmax_w = np.abs(corrected_corr).max() or 1
        im6 = axes[2, 1].imshow(corrected_corr, cmap='RdBu_r', aspect='equal',
                                  vmin=-vmax_w, vmax=vmax_w)
        axes[2, 1].set_title('Corrected Graph Weights(without Higham)')
        fig.colorbar(im6, ax=axes[2, 1], label='Pearson r', shrink=0.8)

        # [2,2] Higham correlation matrix
        vmax_h = np.abs(higham_corr).max()
        im7 = axes[2, 2].imshow(higham_corr, cmap='RdBu_r', aspect='equal',
                                  vmin=-vmax_h, vmax=vmax_h)
        axes[2, 2].set_title("Correlation matrix (Higham's Algorithm)")
        fig.colorbar(im7, ax=axes[2, 2], label='Pearson r', shrink=0.8)

        for ax in [axes[0, 0], axes[0, 2], axes[1, 0], axes[1, 1], axes[1, 2]]:
            ax.set_xticks(range(len(ch_names)))
            ax.set_yticks(range(len(ch_names)))
            ax.set_xticklabels(ch_names, rotation=90, fontsize=6)
            ax.set_yticklabels(ch_names, fontsize=6)

        fig.suptitle(f'Time-averaged  |  lag = {lag:+.3f}s  |  epoch = {epoch}', fontsize=12)
      #  plt.show()

    return corrected_corr, higham_corr, corr_at_global, best_lag_sec_pp, best_lag_corr_pp


#lag_slider.observe(lambda change: redraw('P1', change), names='value')
#display(lag_slider, out)
#corrected, higham = redraw('P1')

epfjq,wepof, global_corr, opt_lag, lag_applied = redraw("P2 Last 10")
p2_avg = np.mean(p2_last_10.get_data(), axis = 0)


new_corr = []
signal_noise_ratio = []
time = len(p2_last_10.times)
for i in range(len(opt_lag)):
    x_corr = []
    x_ratio = []
    for j in range(len(opt_lag)):
        if i == j:
            x_corr.append(1.0)
            x_ratio.append(1.0)
            continue
        lag = opt_lag[i][j]
        lag_samples = int(np.ceil(np.round(lag*512,1)))
        lagged_graph = lagged_corr_2d(p2_avg,lag_samples)

        eig = np.linalg.eigh(lagged_graph)
        eigenvalues = eig[0]
        eigenvectors = eig[1]
#        for k in eigenvalues:
#            if k < 0:
#                print("negative eigen",k, i,j, lag, lag_samples, eigenvalues.max())
#                break
        lag_var = np.mean(eigenvalues)
       #print(lag_var)
        

        lambda_plus = lag_var * 4
        noise_mean = eigenvalues[eigenvalues <= lambda_plus].mean()
        clipped = np.where(eigenvalues > lambda_plus, eigenvalues, 0)
        clean_lagged = eigenvectors @ np.diag(clipped) @ eigenvectors.T
        
        clean_corr = clean_lagged[i][j]
        ratio = np.sign(clean_corr)*clean_corr/lag_applied[i][j]
        x_ratio.append(ratio)
        x_corr.append(clean_corr)
    signal_noise_ratio.append(x_ratio)
    new_corr.append(x_corr)
    
#initialize_graph(p2)
#g_signal_noise = build_corr_graph(p1,mask = signal_noise_ratio,weights = new_corr, threshold = .2)
#pos = nx.forceatlas2_layout(g_signal_noise)
#nx.draw(g_signal_noise, pos = nx.forceatlas2_layout(g_signal_noise),node_color = [g_signal_noise.nodes[n]['color'] for n in g_signal_noise.nodes], with_labels = True)

def plot_3d_graph(g):
    pos = nx.forceatlas2_layout(g, dim=3, seed=42)

    # edges
    edge_x, edge_y, edge_z = [], [], []
    for u, v in g.edges():
        x0, y0, z0 = pos[u]
        x1, y1, z1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_z += [z0, z1, None]

    edge_trace = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(color='grey', width=1),
        hoverinfo='none'
    )

    # one trace per group so legend works
    node_traces = []
    for group, color in colordict.items():
        nodes_in_group = [n for n in g.nodes() if g.nodes[n]['group'] == group]
        if not nodes_in_group:
            continue
        node_traces.append(go.Scatter3d(
            x=[pos[n][0] for n in nodes_in_group],
            y=[pos[n][1] for n in nodes_in_group],
            z=[pos[n][2] for n in nodes_in_group],
            mode='markers+text',
            name=group,
            text=[g.nodes[n]['name'] for n in nodes_in_group],
            textposition='top center',
            marker=dict(size=6, color=color),
            hoverinfo='text'
        ))

    fig = go.Figure(data=[edge_trace, *node_traces])
    fig.update_layout(
        showlegend=True,
        scene=dict(
            xaxis=dict(showbackground=False),
            yaxis=dict(showbackground=False),
            zaxis=dict(showbackground=False)
        ),
        margin=dict(l=0, r=0, t=0, b=0)
    )
    fig.show()

# clean versions of your existing functions
def initialize_graph(epochs):
    graph = nx.DiGraph()
    for i, ch in enumerate(epochs.ch_names):
        group = CH_TO_GROUP.get(ch, "")
        graph.add_node(i, color=colordict[group], name=ch, group=group)
    return graph

def build_corr_graph(epochs, mask, weights, threshold=0.2, use_abs=True):
    g = initialize_graph(epochs)
    mask = np.array(mask)
    weights = np.array(weights)
    n = len(epochs.ch_names)
    iu, ju = np.triu_indices(n, k=1)
    keep = mask[iu, ju] > threshold
    w = weights[iu[keep], ju[keep]]
    if use_abs:
        w = np.abs(w)
    g.add_weighted_edges_from(zip(iu[keep].tolist(), ju[keep].tolist(), w.tolist()))
    return g

g_signal_noise = build_corr_graph(p2_last_10, mask=signal_noise_ratio, weights=new_corr, threshold=0.2)
plot_3d_graph(g_signal_noise)

#nodes = np.array([pos[v] for v in g_signal_noise])
#edges = np.array([(pos[u], pos[v]) for u, v in g_signal_noise.edges()])
#fig = plt.figure()
#ax = fig.add_subplot(111, projection="3d")
#ax.clear()
#ax.scatter(*nodes.T, alpha=0.2, s=100, color="blue")
#for vizedge in edges:
#    ax.plot(*vizedge.T, color="gray")
#ax.grid(False)
#ax.set_axis_off()
#
#fig.tight_layout()
#plt.show()