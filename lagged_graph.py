"""Directed lagged-correlation graphs for sliding windows.

Companion to connectivity.py. Where sliding_corr() computes zero-lag Pearson
correlation per window, this module sweeps lags and records, for each ordered
pair (i, j), the signed correlation at the lag of peak |r| and the lag itself.
Positive lag[i, j] means channel i leads channel j → arrow i → j.
"""

from __future__ import annotations

import warnings

import numpy as np


def lagged_xcorr(signal, sfreq, max_lag_ms):
    """Cross-correlate every pair of channels at lags in [-L, +L] samples.

    Parameters
    ----------
    signal : ndarray, shape (n_ch, n_times)
    sfreq : float
    max_lag_ms : float
        Maximum absolute lag swept, in milliseconds.

    Returns
    -------
    best_r : ndarray, shape (n_ch, n_ch)
        Signed Pearson r at the lag of peak |r|. Diagonal is 1 (or NaN if the
        channel has zero variance).
    best_lag_samples : ndarray, shape (n_ch, n_ch), int
        Signed lag at which best_r occurs. best_lag_samples[i, j] > 0 means i
        leads j; rows of length < lag are dropped from the overlap.
    r_at_zero : ndarray, shape (n_ch, n_ch)
        Signed Pearson r at lag = 0 (zero-lag baseline). Used to score how
        much of `best_r` is above-and-beyond instantaneous coupling.
    """
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 2:
        raise ValueError(f"signal must be 2D (n_ch, n_times); got {signal.shape}")
    n_ch, n_times = signal.shape

    max_lag = int(round(max_lag_ms * 1e-3 * sfreq))
    if max_lag < 0:
        raise ValueError("max_lag_ms must be non-negative")
    if max_lag >= n_times:
        raise ValueError(
            f"max_lag ({max_lag} samples) must be < n_times ({n_times})."
        )

    stds = signal.std(axis=1)
    zero_var = stds == 0

    lags = np.arange(-max_lag, max_lag + 1)
    zero_idx = int(np.where(lags == 0)[0][0])
    best_r = np.zeros((n_ch, n_ch), dtype=np.float64)
    best_lag = np.zeros((n_ch, n_ch), dtype=np.int64)
    r_at_zero = np.zeros((n_ch, n_ch), dtype=np.float64)

    for i in range(n_ch):
        for j in range(n_ch):
            if zero_var[i] or zero_var[j]:
                best_r[i, j] = np.nan
                best_lag[i, j] = 0
                r_at_zero[i, j] = np.nan
                continue
            if i == j:
                best_r[i, j] = 1.0
                best_lag[i, j] = 0
                r_at_zero[i, j] = 1.0
                continue

            r_at_lag = np.empty(lags.size, dtype=np.float64)
            for k, tau in enumerate(lags):
                if tau >= 0:
                    a = signal[i, : n_times - tau]
                    b = signal[j, tau:]
                else:
                    a = signal[i, -tau:]
                    b = signal[j, : n_times + tau]
                # Pearson r on the overlap, centered on each overlap slice.
                a_c = a - a.mean()
                b_c = b - b.mean()
                na = float(np.sqrt((a_c * a_c).sum()))
                nb = float(np.sqrt((b_c * b_c).sum()))
                if na == 0 or nb == 0:
                    r_at_lag[k] = np.nan
                    continue
                r_at_lag[k] = float((a_c * b_c).sum() / (na * nb))

            r_at_zero[i, j] = r_at_lag[zero_idx]

            if np.all(np.isnan(r_at_lag)):
                best_r[i, j] = np.nan
                best_lag[i, j] = 0
                continue
            k_best = int(np.nanargmax(np.abs(r_at_lag)))
            best_r[i, j] = r_at_lag[k_best]
            best_lag[i, j] = int(lags[k_best])

    return best_r, best_lag, r_at_zero


def sliding_lagged_xcorr(signal, sfreq, window_ms=40, step_ms=5, max_lag_ms=10):
    """Sliding-window lagged cross-correlation.

    Returns
    -------
    r_tensor : ndarray, shape (n_windows, n_ch, n_ch)
    lag_tensor : ndarray, shape (n_windows, n_ch, n_ch), int
        Lag (in samples) of the peak |r| per pair per window.
    r0_tensor : ndarray, shape (n_windows, n_ch, n_ch)
        Zero-lag Pearson r per window (the baseline against which peak |r|
        is scored).
    centers_s : ndarray, shape (n_windows,)
        Window-center times in seconds, relative to sample 0 of `signal`.
    """
    signal = np.asarray(signal)
    if signal.ndim != 2:
        raise ValueError(f"signal must be 2D (n_ch, n_times); got {signal.shape}")
    n_ch, n_times = signal.shape

    win = max(2, int(round(window_ms * 1e-3 * sfreq)))
    step = max(1, int(round(step_ms * 1e-3 * sfreq)))
    if win > n_times:
        raise ValueError(
            f"window_ms={window_ms} ({win} samples) exceeds signal length {n_times}."
        )

    starts = np.arange(0, n_times - win + 1, step)
    n_windows = starts.size
    r_tensor = np.empty((n_windows, n_ch, n_ch), dtype=np.float64)
    lag_tensor = np.empty((n_windows, n_ch, n_ch), dtype=np.int64)
    r0_tensor = np.empty((n_windows, n_ch, n_ch), dtype=np.float64)
    nan_windows = 0

    for w, s in enumerate(starts):
        seg = signal[:, s : s + win]
        try:
            r, lag, r0 = lagged_xcorr(seg, sfreq, max_lag_ms)
        except ValueError:
            r_tensor[w] = np.nan
            lag_tensor[w] = 0
            r0_tensor[w] = np.nan
            nan_windows += 1
            continue
        r_tensor[w] = r
        lag_tensor[w] = lag
        r0_tensor[w] = r0
        if np.isnan(r).any():
            nan_windows += 1

    if nan_windows:
        warnings.warn(
            f"{nan_windows}/{n_windows} windows had NaNs (zero-variance channels)."
        )

    centers_samples = starts + (win - 1) / 2.0
    centers_s = centers_samples / sfreq
    return r_tensor, lag_tensor, r0_tensor, centers_s


def anchor_lagged_xcorr(signal, sfreq, max_lag_ms, anchor_indices):
    """Cross-correlate each anchor channel against every other channel.

    Same outputs as `lagged_xcorr`, but only entries (i, j) where i or j is an
    anchor are computed. Non-anchor pairs are returned as 0 for r, lag, and
    r_at_zero (so `build_directed_graph` will naturally filter them out via
    its lag / r thresholds).
    """
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 2:
        raise ValueError(f"signal must be 2D (n_ch, n_times); got {signal.shape}")
    n_ch, n_times = signal.shape
    anchor_indices = list(anchor_indices)
    for a in anchor_indices:
        if not 0 <= a < n_ch:
            raise ValueError(f"anchor index {a} out of range for {n_ch} channels")

    max_lag = int(round(max_lag_ms * 1e-3 * sfreq))
    if max_lag < 0:
        raise ValueError("max_lag_ms must be non-negative")
    if max_lag >= n_times:
        raise ValueError(
            f"max_lag ({max_lag} samples) must be < n_times ({n_times})."
        )

    stds = signal.std(axis=1)
    zero_var = stds == 0

    lags = np.arange(-max_lag, max_lag + 1)
    zero_idx = int(np.where(lags == 0)[0][0])
    best_r = np.zeros((n_ch, n_ch), dtype=np.float64)
    best_lag = np.zeros((n_ch, n_ch), dtype=np.int64)
    r_at_zero = np.zeros((n_ch, n_ch), dtype=np.float64)

    for a in anchor_indices:
        if zero_var[a]:
            best_r[a, a] = np.nan
            r_at_zero[a, a] = np.nan
        else:
            best_r[a, a] = 1.0
            r_at_zero[a, a] = 1.0

    pair_set = set()
    for a in anchor_indices:
        for k in range(n_ch):
            if k == a:
                continue
            pair_set.add((a, k))
            pair_set.add((k, a))

    for i, j in pair_set:
        if zero_var[i] or zero_var[j]:
            best_r[i, j] = np.nan
            best_lag[i, j] = 0
            r_at_zero[i, j] = np.nan
            continue

        r_at_lag = np.empty(lags.size, dtype=np.float64)
        for k, tau in enumerate(lags):
            if tau >= 0:
                a_sig = signal[i, : n_times - tau]
                b_sig = signal[j, tau:]
            else:
                a_sig = signal[i, -tau:]
                b_sig = signal[j, : n_times + tau]
            a_c = a_sig - a_sig.mean()
            b_c = b_sig - b_sig.mean()
            na = float(np.sqrt((a_c * a_c).sum()))
            nb = float(np.sqrt((b_c * b_c).sum()))
            if na == 0 or nb == 0:
                r_at_lag[k] = np.nan
                continue
            r_at_lag[k] = float((a_c * b_c).sum() / (na * nb))

        r_at_zero[i, j] = r_at_lag[zero_idx]

        if np.all(np.isnan(r_at_lag)):
            best_r[i, j] = np.nan
            best_lag[i, j] = 0
            continue
        k_best = int(np.nanargmax(np.abs(r_at_lag)))
        best_r[i, j] = r_at_lag[k_best]
        best_lag[i, j] = int(lags[k_best])

    return best_r, best_lag, r_at_zero


def anchor_sliding_lagged_xcorr(
    signal, sfreq, anchor_indices, window_ms=40, step_ms=5, max_lag_ms=10,
):
    """Sliding-window anchor-based lagged cross-correlation.

    Mirrors `sliding_lagged_xcorr` but delegates to `anchor_lagged_xcorr` so
    only anchor rows/columns are populated per window.
    """
    signal = np.asarray(signal)
    if signal.ndim != 2:
        raise ValueError(f"signal must be 2D (n_ch, n_times); got {signal.shape}")
    n_ch, n_times = signal.shape

    win = max(2, int(round(window_ms * 1e-3 * sfreq)))
    step = max(1, int(round(step_ms * 1e-3 * sfreq)))
    if win > n_times:
        raise ValueError(
            f"window_ms={window_ms} ({win} samples) exceeds signal length {n_times}."
        )

    starts = np.arange(0, n_times - win + 1, step)
    n_windows = starts.size
    r_tensor = np.empty((n_windows, n_ch, n_ch), dtype=np.float64)
    lag_tensor = np.empty((n_windows, n_ch, n_ch), dtype=np.int64)
    r0_tensor = np.empty((n_windows, n_ch, n_ch), dtype=np.float64)
    nan_windows = 0

    for w, s in enumerate(starts):
        seg = signal[:, s : s + win]
        try:
            r, lag, r0 = anchor_lagged_xcorr(seg, sfreq, max_lag_ms, anchor_indices)
        except ValueError:
            r_tensor[w] = np.nan
            lag_tensor[w] = 0
            r0_tensor[w] = np.nan
            nan_windows += 1
            continue
        r_tensor[w] = r
        lag_tensor[w] = lag
        r0_tensor[w] = r0
        if np.isnan(r).any():
            nan_windows += 1

    if nan_windows:
        warnings.warn(
            f"{nan_windows}/{n_windows} windows had NaNs (zero-variance channels)."
        )

    centers_samples = starts + (win - 1) / 2.0
    centers_s = centers_samples / sfreq
    return r_tensor, lag_tensor, r0_tensor, centers_s


def anchor_sliding_lagged_from_epochs(
    epochs,
    anchors,
    window_ms=40,
    step_ms=5,
    max_lag_ms=10,
    use="single_trial",
    trial_idx=0,
):
    """Anchor-restricted version of `sliding_lagged_from_epochs`.

    Parameters
    ----------
    anchors : sequence of str
        Channel names to use as anchors. Only pairs (anchor, other) and
        (other, anchor) are computed.

    Returns
    -------
    Same as `sliding_lagged_from_epochs`. In each returned (n_ch, n_ch) slice,
    only anchor rows/columns are populated; other entries are zero.
    """
    sfreq = epochs.info["sfreq"]
    ch_names = list(epochs.ch_names)
    anchor_indices = []
    for a in anchors:
        if a not in ch_names:
            raise ValueError(f"anchor {a!r} not in channel list: {ch_names}")
        anchor_indices.append(ch_names.index(a))

    if use == "evoked":
        signal = epochs.average().data
    elif use == "single_trial":
        data = epochs.get_data()
        if not 0 <= trial_idx < data.shape[0]:
            raise IndexError(
                f"trial_idx={trial_idx} out of range for {data.shape[0]} trials."
            )
        signal = data[trial_idx]
    else:
        raise ValueError(
            f"use={use!r} not recognized; expected 'single_trial' or 'evoked'."
        )

    r_tensor, lag_tensor, r0_tensor, centers_rel = anchor_sliding_lagged_xcorr(
        signal, sfreq, anchor_indices,
        window_ms=window_ms, step_ms=step_ms, max_lag_ms=max_lag_ms,
    )
    centers_s = centers_rel + epochs.times[0]
    return r_tensor, lag_tensor, r0_tensor, centers_s, ch_names


def anchor_pair_sliding_curves_from_epochs(
    epochs,
    anchor,
    others,
    window_ms=40,
    step_ms=5,
    max_lag_ms=10,
    use="single_trial",
    trial_idx=0,
):
    """Sliding-window r(tau) curves for each (anchor, other) pair.

    Uses the same window / step / lag schedule as `sliding_lagged_from_epochs`,
    so the output is a drop-in view of the exact correlations that feed
    `build_directed_graph`: each window's full r-vs-lag profile is kept
    instead of being collapsed to (peak r, peak lag).

    Parameters
    ----------
    epochs : mne.Epochs
    anchor : str
    others : sequence of str
    window_ms, step_ms, max_lag_ms : float
        Match whatever values were used when building the graph.
    use, trial_idx : see `sliding_lagged_from_epochs`.

    Returns
    -------
    centers_s : ndarray, shape (n_windows,)
        Window-center times in the epoch frame.
    lags_ms : ndarray, shape (2L+1,)
        Lag axis in milliseconds. tau > 0 means `anchor` leads `other`.
    curves : dict[str, ndarray]
        Maps each name in `others` to an (n_windows, 2L+1) Pearson-r tensor.
    """
    sfreq = epochs.info["sfreq"]
    ch_names = list(epochs.ch_names)
    if anchor not in ch_names:
        raise ValueError(f"anchor {anchor!r} not in channel list.")
    for o in others:
        if o not in ch_names:
            raise ValueError(f"other {o!r} not in channel list.")
    a_idx = ch_names.index(anchor)
    other_idxs = [ch_names.index(o) for o in others]

    if use == "evoked":
        signal = epochs.average().data
    elif use == "single_trial":
        data = epochs.get_data()
        if not 0 <= trial_idx < data.shape[0]:
            raise IndexError(
                f"trial_idx={trial_idx} out of range for {data.shape[0]} trials."
            )
        signal = data[trial_idx]
    else:
        raise ValueError(
            f"use={use!r} not recognized; expected 'single_trial' or 'evoked'."
        )

    signal = np.asarray(signal, dtype=np.float64)
    _, n_times = signal.shape
    win = max(2, int(round(window_ms * 1e-3 * sfreq)))
    step = max(1, int(round(step_ms * 1e-3 * sfreq)))
    max_lag = int(round(max_lag_ms * 1e-3 * sfreq))
    if max_lag < 0:
        raise ValueError("max_lag_ms must be non-negative")
    if max_lag >= win:
        raise ValueError(
            f"max_lag ({max_lag} samples) must be < window ({win} samples)."
        )
    if win > n_times:
        raise ValueError(
            f"window ({win} samples) exceeds signal length {n_times}."
        )

    starts = np.arange(0, n_times - win + 1, step)
    n_windows = starts.size
    lags = np.arange(-max_lag, max_lag + 1)
    lags_ms = lags * 1000.0 / sfreq
    curves = {o: np.full((n_windows, lags.size), np.nan) for o in others}

    for w, s in enumerate(starts):
        seg = signal[:, s : s + win]
        x = seg[a_idx]
        for o_name, j in zip(others, other_idxs):
            y = seg[j]
            for k, tau in enumerate(lags):
                if tau >= 0:
                    a = x[: win - tau]
                    b = y[tau:]
                else:
                    a = x[-tau:]
                    b = y[: win + tau]
                a_c = a - a.mean()
                b_c = b - b.mean()
                na = float(np.sqrt((a_c * a_c).sum()))
                nb = float(np.sqrt((b_c * b_c).sum()))
                if na == 0 or nb == 0:
                    continue
                curves[o_name][w, k] = float((a_c * b_c).sum() / (na * nb))

    centers_samples = starts + (win - 1) / 2.0
    centers_s = centers_samples / sfreq + epochs.times[0]
    return centers_s, lags_ms, curves


def draw_sliding_lag_heatmaps(centers_s, lags_ms, curves, anchor, ncols=3, vmax=None):
    """One heatmap per pair: x=window-center time, y=lag, color=r.

    Each panel is the raw r(tau, t) tensor that the graph layer collapses to
    (peak r, peak lag). A consistent ridge above tau=0 means anchor leads;
    below, anchor lags; multiple parallel bands indicate oscillatory coupling.
    """
    import matplotlib.pyplot as plt

    others = list(curves.keys())
    n = len(others)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4 * ncols, 2.8 * nrows),
        sharex=True, sharey=True,
    )
    axes = np.atleast_2d(axes).ravel()

    t_ms = np.asarray(centers_s) * 1000.0
    if vmax is None:
        vmax = 0.1
        for c in curves.values():
            m = np.nanmax(np.abs(c)) if np.any(np.isfinite(c)) else 0.0
            vmax = max(vmax, float(m))

    im = None
    for k, other in enumerate(others):
        ax = axes[k]
        im = ax.imshow(
            curves[other].T, aspect="auto", origin="lower",
            extent=[t_ms[0], t_ms[-1], lags_ms[0], lags_ms[-1]],
            cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest",
        )
        ax.axhline(0, color="k", lw=0.5, ls=":")
        ax.set_title(f"{anchor} vs {other}", fontsize=9)

    for ax in axes[n:]:
        ax.set_visible(False)
    for ax in axes[-ncols:]:
        ax.set_xlabel("window center (ms)")
    for ax in axes[::ncols]:
        ax.set_ylabel("lag (ms)  (+ = anchor leads)")

    fig.suptitle(
        f"Sliding lagged correlation r(tau, t) - anchor {anchor}", fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 0.92, 0.96])
    if im is not None:
        cax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
        fig.colorbar(im, cax=cax, label="r")
    return fig


def sliding_lagged_from_epochs(
    epochs,
    window_ms=40,
    step_ms=5,
    max_lag_ms=10,
    use="single_trial",
    trial_idx=0,
):
    """Sliding lagged cross-correlation driven from an MNE Epochs object.

    Parameters
    ----------
    epochs : mne.Epochs
    use : {'single_trial', 'evoked'}
        'single_trial' selects one trial by `trial_idx`.
        'evoked' uses the trial-averaged signal.
    trial_idx : int
        Required when use='single_trial'.

    Returns
    -------
    r_tensor, lag_tensor, r0_tensor : see sliding_lagged_xcorr.
    centers_s : ndarray
        Window-center times in the epoch's own time frame.
    ch_names : list[str]
    """
    sfreq = epochs.info["sfreq"]

    if use == "evoked":
        signal = epochs.average().data
    elif use == "single_trial":
        data = epochs.get_data()
        if not (0 <= trial_idx < data.shape[0]):
            raise IndexError(
                f"trial_idx={trial_idx} out of range for {data.shape[0]} trials."
            )
        signal = data[trial_idx]
    else:
        raise ValueError(f"use={use!r} not recognized; expected 'single_trial' or 'evoked'.")

    r_tensor, lag_tensor, r0_tensor, centers_rel = sliding_lagged_xcorr(
        signal, sfreq, window_ms=window_ms, step_ms=step_ms, max_lag_ms=max_lag_ms
    )
    centers_s = centers_rel + epochs.times[0]
    return r_tensor, lag_tensor, r0_tensor, centers_s, list(epochs.ch_names)


def build_directed_graph(
    r_matrix,
    lag_matrix,
    ch_names,
    r_at_zero_matrix=None,
    r_threshold=0.2,
    min_abs_lag_samples=2,
    lag_gain_threshold=0.08,
    top_k=None,
    sfreq=None,
):
    """Build a DiGraph from a single window's (r, lag[, r0]) matrices.

    Filter cascade, applied in order to each ordered pair (i, j):
      1. drop non-finite r
      2. keep only lag >= min_abs_lag_samples (one direction per pair, no
         near-zero-lag pairs)
      3. drop |r| < r_threshold  (soft floor — rejects pure noise)
      4. if r_at_zero_matrix is given: drop
         |r| - |r_at_zero| < lag_gain_threshold  (the "lag-gain" criterion —
         keeps only edges whose correlation is genuinely lag-dependent)
      5. if top_k is given: rank the survivors by lag-gain (or |r| if no
         r_at_zero given) and keep the top K.

    Parameters
    ----------
    r_matrix, lag_matrix, r_at_zero_matrix : (n_ch, n_ch) arrays
    ch_names : list[str], length n_ch
    r_threshold : float
        Minimum |r|; mainly a noise floor now that lag-gain does the work.
    min_abs_lag_samples : int
        Minimum |lag| in samples. Pairs whose peak sits at ±(min-1) or closer
        to zero are dropped (their direction is unreliable).
    lag_gain_threshold : float
        Minimum |r_peak| - |r_at_zero|. Only applied when r_at_zero_matrix
        is provided.
    top_k : int | None
        If given, keep only the K strongest surviving edges by ranking metric
        (lag-gain if r_at_zero given, else |r|).
    sfreq : float | None
        If provided, edges carry a `lag_ms` attribute.

    Returns
    -------
    G : networkx.DiGraph
        Node attrs: `name`. Edge attrs: `weight` (|r|), `sign` (+/-1),
        `lag_samples`, `lag_gain` (if r0 given), and `lag_ms` (if sfreq given).
    """
    import networkx as nx

    n_ch = len(ch_names)
    if r_matrix.shape != (n_ch, n_ch) or lag_matrix.shape != (n_ch, n_ch):
        raise ValueError("r_matrix / lag_matrix must be square matching ch_names.")
    if r_at_zero_matrix is not None and r_at_zero_matrix.shape != (n_ch, n_ch):
        raise ValueError("r_at_zero_matrix shape mismatch.")

    candidates = []  # (rank_score, i, j, r, tau, lag_gain_or_None)
    for i in range(n_ch):
        for j in range(n_ch):
            if i == j:
                continue
            r = r_matrix[i, j]
            tau = int(lag_matrix[i, j])
            if not np.isfinite(r):
                continue
            if tau < min_abs_lag_samples:
                continue  # covers zero, negative, and near-zero lags in one shot
            if abs(r) < r_threshold:
                continue
            gain = None
            if r_at_zero_matrix is not None:
                r0 = r_at_zero_matrix[i, j]
                if not np.isfinite(r0):
                    continue
                gain = abs(r) - abs(r0)
                if gain < lag_gain_threshold:
                    continue
            rank_score = gain if gain is not None else abs(r)
            candidates.append((rank_score, i, j, float(r), tau, gain))

    if top_k is not None and len(candidates) > top_k:
        candidates.sort(key=lambda t: t[0], reverse=True)
        candidates = candidates[:top_k]

    G = nx.DiGraph()
    for name in ch_names:
        G.add_node(name, name=name)

    for _score, i, j, r, tau, gain in candidates:
        attrs = {
            "weight": float(abs(r)),
            "sign": int(np.sign(r)) or 1,
            "lag_samples": int(tau),
        }
        if gain is not None:
            attrs["lag_gain"] = float(gain)
        if sfreq is not None:
            attrs["lag_ms"] = float(tau) * 1e3 / sfreq
        G.add_edge(ch_names[i], ch_names[j], **attrs)
    return G


def scalp_positions(ch_names, montage_name="standard_1020"):
    """2D scalp positions keyed by channel name, derived from an MNE montage.

    Channels missing from the montage fall back to circular positions.
    """
    import mne
    import networkx as nx

    montage = mne.channels.make_standard_montage(montage_name)
    pos3d = montage.get_positions()["ch_pos"]

    pos = {}
    missing = []
    for name in ch_names:
        key = name if name in pos3d else name.upper() if name.upper() in pos3d else None
        if key is None:
            # Montages sometimes use different case; try a case-insensitive match.
            for k in pos3d:
                if k.lower() == name.lower():
                    key = k
                    break
        if key is None:
            missing.append(name)
            continue
        x, y, _ = pos3d[key]
        pos[name] = (float(x), float(y))

    if missing:
        warnings.warn(
            f"{len(missing)} channels missing from montage '{montage_name}'; "
            f"placing on a fallback circle: {missing}"
        )
        fallback = nx.circular_layout(missing, scale=0.1)
        for name, (x, y) in fallback.items():
            pos[name] = (x, y - 0.15)  # below the scalp cloud so they're visible

    return pos


def _assign_roi_color(ch_name):
    from connectivity import _assign_roi

    palette = {
        "Frontal": "#6baed6",
        "Frontocentral": "#74c476",
        "Somatosensory": "#fd8d3c",
        "Centroparietal": "#9e9ac8",
        "Temporal": "#e377c2",
        "Parietal/Occipital": "#bcbd22",
        "Other": "#999999",
    }
    return palette.get(_assign_roi(ch_name), "#cccccc")


def draw_lagged_graph(
    G,
    ax,
    pos=None,
    title=None,
    node_size=450,
    edge_width_scale=4.0,
    arrow_size_scale=18.0,
    fontsize=8,
):
    """Draw a directed lagged-correlation graph on a matplotlib Axes.

    Edge color encodes sign of r (red = positive, blue = negative). Edge width
    is proportional to |r|. Arrowhead size is proportional to |lag_ms| if
    present, else constant. Nodes are colored by ROI via `_assign_roi`.
    """
    import networkx as nx

    ch_names = list(G.nodes())
    if pos is None:
        pos = scalp_positions(ch_names)

    node_colors = [_assign_roi_color(n) for n in ch_names]
    nx.draw_networkx_nodes(
        G, pos, ax=ax, node_color=node_colors, node_size=node_size,
        edgecolors="black", linewidths=0.5,
    )
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=fontsize)

    pos_edges = [(u, v) for u, v, d in G.edges(data=True) if d["sign"] > 0]
    neg_edges = [(u, v) for u, v, d in G.edges(data=True) if d["sign"] < 0]

    def _widths(edges):
        return [edge_width_scale * G.edges[e]["weight"] for e in edges]

    def _arrow_sizes(edges):
        sizes = []
        for e in edges:
            d = G.edges[e]
            base = d.get("lag_ms")
            if base is None:
                sizes.append(arrow_size_scale)
            else:
                sizes.append(arrow_size_scale * max(0.25, abs(base) / 10.0))
        return sizes

    if pos_edges:
        nx.draw_networkx_edges(
            G, pos, ax=ax, edgelist=pos_edges,
            width=_widths(pos_edges),
            edge_color="#d62728",
            arrows=True, arrowstyle="-|>",
            arrowsize=max(_arrow_sizes(pos_edges)) if pos_edges else arrow_size_scale,
            connectionstyle="arc3,rad=0.08",
            alpha=0.8,
        )
    if neg_edges:
        nx.draw_networkx_edges(
            G, pos, ax=ax, edgelist=neg_edges,
            width=_widths(neg_edges),
            edge_color="#1f77b4",
            arrows=True, arrowstyle="-|>",
            arrowsize=max(_arrow_sizes(neg_edges)) if neg_edges else arrow_size_scale,
            connectionstyle="arc3,rad=0.08",
            alpha=0.8,
        )

    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10)
    return ax


def consistency_across_windows(r_tensor, lag_tensor, r_threshold=0.5):
    """Per-pair stability metrics across sliding windows within one trial.

    Returns dict of (n_ch, n_ch) arrays:
      - edge_persistence: fraction of windows with |r| >= threshold
      - lag_sign_consistency: among those windows, fraction with sign(lag) ==
        sign of the most-common lag sign (0 if no windows cross threshold)
      - mean_r: mean of r across all windows (ignoring NaN)
      - mean_lag_samples: mean of lag across windows where the edge exists
    """
    r_tensor = np.asarray(r_tensor)
    lag_tensor = np.asarray(lag_tensor)
    n_windows, n_ch, _ = r_tensor.shape

    above = np.abs(r_tensor) >= r_threshold
    finite = np.isfinite(r_tensor)
    exists = above & finite  # (n_windows, n_ch, n_ch)

    edge_persistence = exists.mean(axis=0)
    mean_r = np.nanmean(r_tensor, axis=0)

    lag_sign = np.sign(lag_tensor)
    # For each pair, among windows where the edge exists, what fraction agree
    # with the dominant sign? (dominant = whichever of +/- has more votes)
    pos_votes = ((lag_sign > 0) & exists).sum(axis=0)
    neg_votes = ((lag_sign < 0) & exists).sum(axis=0)
    total_votes = pos_votes + neg_votes
    dominant = np.maximum(pos_votes, neg_votes)
    with np.errstate(invalid="ignore", divide="ignore"):
        lag_sign_consistency = np.where(
            total_votes > 0, dominant / np.maximum(total_votes, 1), 0.0
        )

    lag_sum = np.where(exists, lag_tensor, 0).sum(axis=0)
    lag_count = exists.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_lag_samples = np.where(lag_count > 0, lag_sum / np.maximum(lag_count, 1), 0.0)

    return {
        "edge_persistence": edge_persistence,
        "lag_sign_consistency": lag_sign_consistency,
        "mean_r": mean_r,
        "mean_lag_samples": mean_lag_samples,
    }


if __name__ == "__main__":
    # Sanity check: construct base[t] and then ch1 = base[:N], ch0 = base[shift:shift+N].
    # Then ch0[t] = ch1[t + shift] — ch0 *sees ch1's future*, so ch0 leads ch1.
    # lagged_xcorr should recover lag[0, 1] = +shift with r ≈ 1.
    rng = np.random.default_rng(0)
    sfreq = 500.0
    n_times = 400
    shift = 3
    base = rng.standard_normal(n_times + shift)
    ch1 = base[:n_times]
    ch0 = base[shift : shift + n_times]
    sig = np.stack([ch0, ch1], axis=0)

    r, lag, r0 = lagged_xcorr(sig, sfreq, max_lag_ms=20.0)
    print("r matrix:\n", r)
    print("lag matrix (samples):\n", lag)
    print("r_at_zero matrix:\n", r0)
    assert abs(r[0, 1] - 1.0) < 1e-10, r[0, 1]
    assert lag[0, 1] == shift, lag[0, 1]
    assert lag[1, 0] == -shift, lag[1, 0]
    # Lag-gain on a clean shift should be substantial (much larger than r0).
    assert abs(r[0, 1]) - abs(r0[0, 1]) > 0.2, (r[0, 1], r0[0, 1])
    print("Sanity check passed: ch0 leads ch1 by", shift, "samples.")
    print(f"  lag-gain[0,1] = {abs(r[0, 1]) - abs(r0[0, 1]):.3f}")
