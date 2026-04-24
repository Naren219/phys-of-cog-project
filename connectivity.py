import warnings

import numpy as np


def select_by_style(epochs, condition, target_label):
    """Filter Epochs to positions whose original file index maps to target_label
    under get_epoch_style(condition). Preserves _original_trial_indices.
    """
    from utils import get_epoch_style

    orig_indices = getattr(epochs, "_original_trial_indices", list(range(len(epochs))))
    keep_positions = [
        pos
        for pos, orig in enumerate(orig_indices)
        if get_epoch_style(orig, condition)[1] == target_label
    ]
    if not keep_positions:
        raise ValueError(
            f"No trials with label '{target_label}' for condition '{condition}'."
        )
    out = epochs[keep_positions]
    out._original_trial_indices = [orig_indices[p] for p in keep_positions]
    if getattr(epochs, "_normalized", False):
        out._normalized = True
    return out


def select_p2_500ms(epochs):
    """Filter a P2 Epochs object to only its 500 ms-gap trials (P2_500ms in utils)."""
    from utils import P2_500ms

    orig_indices = getattr(epochs, "_original_trial_indices", list(range(len(epochs))))
    p2_500_set = set(P2_500ms)
    keep_positions = [pos for pos, orig in enumerate(orig_indices) if orig in p2_500_set]
    if not keep_positions:
        raise ValueError("No P2 500 ms trials found in these epochs.")
    out = epochs[keep_positions]
    out._original_trial_indices = [orig_indices[p] for p in keep_positions]
    if getattr(epochs, "_normalized", False):
        out._normalized = True
    return out


def sliding_corr(signal, sfreq, window_ms, step_ms):
    """Sliding-window zero-lag Pearson correlation matrices.

    Parameters
    ----------
    signal : ndarray, shape (n_ch, n_times)
    sfreq : float
    window_ms, step_ms : float

    Returns
    -------
    corr_tensor : ndarray, shape (n_windows, n_ch, n_ch)
    window_centers_s : ndarray, shape (n_windows,)
        Window-center times in seconds, relative to sample 0 of `signal`.
    """
    signal = np.asarray(signal)
    if signal.ndim != 2:
        raise ValueError(f"signal must be 2D (n_ch, n_times); got shape {signal.shape}")
    n_ch, n_times = signal.shape

    win = max(2, int(round(window_ms * 1e-3 * sfreq)))
    step = max(1, int(round(step_ms * 1e-3 * sfreq)))
    if win > n_times:
        raise ValueError(
            f"window_ms={window_ms} ({win} samples) exceeds signal length {n_times}."
        )

    starts = np.arange(0, n_times - win + 1, step)
    n_windows = starts.size
    corr_tensor = np.empty((n_windows, n_ch, n_ch), dtype=np.float64)
    zero_var_windows = 0

    for w, s in enumerate(starts):
        seg = signal[:, s : s + win]
        stds = seg.std(axis=1)
        if np.any(stds == 0):
            corr_tensor[w] = np.nan
            zero_var_windows += 1
        else:
            corr_tensor[w] = np.corrcoef(seg)

    if zero_var_windows:
        warnings.warn(
            f"{zero_var_windows}/{n_windows} windows had a zero-variance channel; "
            "those matrices are NaN."
        )

    window_centers_samples = starts + (win - 1) / 2.0
    window_centers_s = window_centers_samples / sfreq
    return corr_tensor, window_centers_s


def sliding_corr_from_epochs(epochs, window_ms, step_ms, use="evoked"):
    """Run sliding_corr on a signal derived from Epochs.

    Parameters
    ----------
    epochs : mne.Epochs
    window_ms, step_ms : float
    use : {'evoked'}
        Currently only 'evoked' is implemented: the trial-averaged signal.

    Returns
    -------
    corr_tensor : ndarray, shape (n_windows, n_ch, n_ch)
    window_centers_s : ndarray, shape (n_windows,)
        Times in the epoch frame (same units as epochs.times).
    ch_names : list[str]
    """
    if use != "evoked":
        raise NotImplementedError(f"use={use!r} not implemented yet.")

    evoked = epochs.average()
    signal = evoked.data  # (n_ch, n_times)
    sfreq = epochs.info["sfreq"]

    corr_tensor, centers_rel = sliding_corr(signal, sfreq, window_ms, step_ms)
    # Shift centers into the epoch's own time frame (epochs.times[0] may be negative).
    window_centers_s = centers_rel + epochs.times[0]
    return corr_tensor, window_centers_s, list(epochs.ch_names)


ROI_DISPLAY_ORDER = [
    "Frontal",
    "Frontocentral",
    "Somatosensory",
    "Centroparietal",
    "Temporal",
    "Parietal/Occipital",
]

# Longer prefixes first so FC/CP/TP/PO beat F/C/T/P.
_ROI_PREFIX_CHECKS = [
    ("Frontocentral",       ("FC", "FT")),
    ("Centroparietal",      ("CP",)),
    ("Temporal",            ("TP", "T")),
    ("Somatosensory",       ("C",)),
    ("Frontal",             ("FP", "AF", "F")),
    ("Parietal/Occipital",  ("PO", "P", "O", "I")),
]


def _assign_roi(ch_name):
    n = ch_name.upper()
    for roi, prefixes in _ROI_PREFIX_CHECKS:
        for p in prefixes:
            if n.startswith(p):
                return roi
    return "Other"


def roi_order(ch_names):
    """Permutation that reorders channels by scalp ROI for block-diagonal readability.

    Returns
    -------
    perm : list[int]
        Indices into ch_names; [ch_names[i] for i in perm] is the ROI-ordered list.
    ordered_names : list[str]
    boundaries : list[int]
        Start index (in the reordered list) of each ROI that has >=1 channel.
    roi_labels : list[str]
        ROI name per boundary, in display order.
    """
    buckets = {roi: [] for roi in ROI_DISPLAY_ORDER}
    buckets["Other"] = []
    for i, c in enumerate(ch_names):
        buckets.setdefault(_assign_roi(c), []).append(i)

    perm, boundaries, labels = [], [], []
    for roi in ROI_DISPLAY_ORDER + ["Other"]:
        idxs = buckets.get(roi, [])
        if not idxs:
            continue
        boundaries.append(len(perm))
        labels.append(roi)
        perm.extend(idxs)
    ordered_names = [ch_names[i] for i in perm]
    return perm, ordered_names, boundaries, labels


def reorder_corr(corr_tensor, perm):
    """Apply a channel permutation to the last two axes of a (..., n_ch, n_ch) tensor."""
    perm = np.asarray(perm)
    return corr_tensor[..., perm, :][..., :, perm]


def draw_roi_heatmap(ax, matrix, boundaries, labels, vmin=-1, vmax=1,
                     cmap="RdBu_r", line_color="k", line_width=0.6):
    """Imshow a correlation matrix with ROI dividers and group tick labels."""
    import matplotlib.pyplot as plt  # local import keeps module import light

    im = ax.imshow(matrix, vmin=vmin, vmax=vmax, cmap=cmap, aspect="equal")
    n = matrix.shape[0]
    for b in boundaries[1:]:
        ax.axhline(b - 0.5, color=line_color, lw=line_width)
        ax.axvline(b - 0.5, color=line_color, lw=line_width)

    ends = boundaries + [n]
    centers = [(a + b - 1) / 2.0 for a, b in zip(ends, ends[1:])]
    ax.set_xticks(centers)
    ax.set_yticks(centers)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    return im


def roi_average_anchor_r(r_tensor, ch_names, anchor):
    """Mean anchor-vs-other peak r within each scalp ROI, per window.

    Parameters
    ----------
    r_tensor : ndarray, shape (n_windows, n_ch, n_ch)
        Signed peak-r from a lagged xcorr run (full or anchor variant).
    ch_names : list[str]
    anchor : str
        Channel name; excluded from its own ROI mean.

    Returns
    -------
    rois : list[str]
        ROIs (in ROI_DISPLAY_ORDER) with >=1 non-anchor channel.
    mean_r : ndarray, shape (n_rois, n_windows)
    mean_abs_r : ndarray, shape (n_rois, n_windows)
    counts : list[int]
    """
    anchor_idx = ch_names.index(anchor)
    roi_per_ch = [_assign_roi(c) for c in ch_names]

    rois, mean_r, mean_abs_r, counts = [], [], [], []
    for roi in ROI_DISPLAY_ORDER:
        idxs = [i for i, r in enumerate(roi_per_ch)
                if r == roi and i != anchor_idx]
        if not idxs:
            continue
        vals = r_tensor[:, anchor_idx, idxs]
        rois.append(roi)
        mean_r.append(np.nanmean(vals, axis=1))
        mean_abs_r.append(np.nanmean(np.abs(vals), axis=1))
        counts.append(len(idxs))
    return rois, np.asarray(mean_r), np.asarray(mean_abs_r), counts


def draw_roi_anchor_timeseries(r_tensor, centers_s, ch_names, anchor, ncols=3):
    """One-panel-per-ROI time series of anchor's ROI-averaged peak r.

    Solid line = signed mean r across ROI channels; dashed = mean |r|.
    Reduces single-electrode noise to group-level coupling.
    """
    import matplotlib.pyplot as plt

    rois, mean_r, mean_abs_r, counts = roi_average_anchor_r(
        r_tensor, ch_names, anchor
    )
    n = len(rois)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4 * ncols, 2.6 * nrows),
        sharex=True, sharey=True,
    )
    axes = np.atleast_2d(axes).ravel()
    t_ms = np.asarray(centers_s) * 1000.0

    for k, roi in enumerate(rois):
        ax = axes[k]
        ax.axhline(0, color="k", lw=0.5)
        ax.plot(t_ms, mean_r[k], color="C0", lw=1.4, label="mean r")
        ax.plot(t_ms, mean_abs_r[k], color="C3", lw=1.1, ls="--", label="mean |r|")
        ax.set_title(f"{roi}  (n={counts[k]})", fontsize=9)
        ax.set_ylim(-1, 1)
        ax.grid(True, alpha=0.25)

    for ax in axes[n:]:
        ax.set_visible(False)
    for ax in axes[-ncols:]:
        ax.set_xlabel("window center (ms)")
    for ax in axes[::ncols]:
        ax.set_ylabel("peak r")
    axes[0].legend(loc="upper right", fontsize=7)

    fig.suptitle(f"ROI-averaged lagged correlation - anchor {anchor}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def summary_metrics(corr_tensor, density_thresh=0.5):
    """Per-window scalar summaries over the upper triangle of each corr matrix.

    Returns dict of 1-D arrays (length n_windows):
      - mean_abs_r
      - global_mean_r
      - density   (fraction of upper-tri pairs with |r| > density_thresh)
    """
    corr_tensor = np.asarray(corr_tensor)
    n_windows, n_ch, _ = corr_tensor.shape
    iu = np.triu_indices(n_ch, k=1)

    mean_abs_r = np.empty(n_windows)
    global_mean_r = np.empty(n_windows)
    density = np.empty(n_windows)

    for w in range(n_windows):
        vals = corr_tensor[w][iu]
        mean_abs_r[w] = np.nanmean(np.abs(vals))
        global_mean_r[w] = np.nanmean(vals)
        density[w] = np.nanmean(np.abs(vals) > density_thresh)

    return {
        "mean_abs_r": mean_abs_r,
        "global_mean_r": global_mean_r,
        "density": density,
    }
