import mne
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# EEG Channel Groups as requested in your provided info
CHANNEL_GROUPS = {
    "BLT": ["C3", "C4", "CP1", "CP2"],
    "BLA": ["T7", "T8", "TP7", "TP8"],
    # Multisensory Integration
    "CISI": ["T7", "TP7", "T8", "TP8", "C3", "CP1", "C4", "CP2"],
    # Temporal Uncertainty
    "VISI": ["C3", "CP1", "C4", "CP2", "F3", "FC1", "FC5", "F4", "FC2", "FC6"],
    # Stimulus Uncertainty
    "RS": ["C3", "CP1", "C4", "CP2", "P1", "P5", "P2", "P6"],

}

FREQ_BANDS = {
    "below_delta": (0, 0.5),
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45),  # Usually 30-45Hz to avoid 50/60Hz line noise
    "above_gamma": (45, 100)
}

# Convert provided indices to 0-based
P2_500ms = [i-1 for i in [
    3,5,13,17,19,21,25,31,37,39,41,47,49,51,
    55,61,65,71,77,79,81,87,89,91,95,
    103,107,109,113,115
]]

P2_2000ms = [i-1 for i in [
    1,7,9,11,15,23,27,29,33,35,43,45,53,57,59,
    63,67,69,73,75,83,85,93,97,99,101,
    105,111,117,119
]]

P3_500ms = [i-1 for i in [
    1,5,11,17,19,21,27,29,31,35,43,47,49,53,
    55,63,65,73,77,79,81,85,91,97,
    99,101,107,109,111,115
]]

P3_missing = [i-1 for i in [
    3,7,9,13,15,23,25,33,37,39,41,45,51,57,59,
    61,67,69,71,75,83,87,89,93,95,
    103,105,113,117,119
]]


def load_eeg_data(
    condition,
    subject_id,
    trials='all',
    channels='all',
    freq_band=None,
    tmin=None,
    tmax=None,
    normalize=False,
):
    """
    Load EEG data as Epochs and optionally apply time cropping, z-score normalization,
    and channel selection (including union of channel groups).

    Parameters
    ----------
    condition : str
        Experiment condition; must match the filename. Examples: "BLA", "BLT", "P1", "P2", "P3".
    subject_id : int
        Subject identifier. Expected file:
        ``dataset/binepochs filtered ICArej {condition}AvgBOS{subject_id}.set``.
    trials : str | list | slice | int
        Which trials to return from the already-filtered set (excluding "no auditory").
        - ``'all'``: all meaningful trials.
        - List of ints: e.g. ``[0, 1, 2]`` for the first three.
        - Slice: e.g. ``slice(0, 10)`` or ``:10`` for the first 10.
        - Int: a single trial.
    channels : str | list
        Channels to load.
        - ``'all'``: all channels in the file.
        - CHANNEL_GROUPS key: e.g. ``"CISI"``, ``"VISI"``, ``"RS"``.
        - List of group names: e.g. ``["CISI", "VISI"]`` → **union** of channels
          from both groups, no duplicates.
        - Single channel name: e.g. ``"Oz"``.
        - List of channel names: e.g. ``["C3", "Oz"]`` (duplicates removed).
    freq_band : str | tuple | None
        Frequency band for filtering. If str, looked up in FREQ_BANDS
        (e.g. ``"alpha"``, ``"delta"``). If tuple ``(low, high)`` in Hz, used as-is.
        ``None`` = no filter.
    tmin : float | None
        Start of time segment in seconds (relative to event). ``None`` = use
        epoch start from file.
    tmax : float | None
        End of time segment in seconds. ``None`` = use epoch end from file.
    normalize : bool
        If ``True``, apply z-score normalization to **all signals** (all trials, all
        channels, all time points). For each channel, that channel's mean and standard
        deviation are computed over **all trials and all times** (meaningful trials
        only); then every value in that channel is replaced by (x - mean) / std.
        So every trial and every time point is normalized using the statistics of
        the channel it belongs to. Default ``False``.

    Returns
    -------
    epochs : mne.Epochs
        Epochs object with only meaningful trials (no "no auditory"). Original file
        indices are stored in ``epochs._original_trial_indices``.

    Raises
    ------
    FileNotFoundError
        If the .set file is not found in ``dataset/``.
    ValueError
        If no meaningful trials remain after filtering.
    """
    data_dir = Path('dataset')
    file_name = f"binepochs filtered ICArej {condition}AvgBOS{subject_id}.set"
    file_path = data_dir / file_name

    if not file_path.exists():
        raise FileNotFoundError(f"File not found. Check path: {file_path.absolute()}")

    epochs = mne.io.read_epochs_eeglab(file_path, verbose=False)

    # Channel Selection (incl. list of groups → union, no duplicates)
    if channels == 'all':
        selected_channels = epochs.ch_names
    elif isinstance(channels, (list, tuple)):
        if all(c in CHANNEL_GROUPS for c in channels):
            # List of groups: union of channels, no duplicates
            selected_channels = []
            seen = set()
            for g in channels:
                for ch in CHANNEL_GROUPS[g]:
                    if ch not in seen:
                        seen.add(ch)
                        selected_channels.append(ch)
        else:
            # List of channel names: unique only
            selected_channels = list(dict.fromkeys(channels))
    elif isinstance(channels, str) and channels in CHANNEL_GROUPS:
        selected_channels = CHANNEL_GROUPS[channels]
    else:
        selected_channels = [channels]
    epochs.pick_channels(selected_channels)

    # Time range (crop before dropping trials)
    if tmin is not None or tmax is not None:
        epochs.crop(tmin=tmin, tmax=tmax)

    # Frequency Selection
    if freq_band is not None:
        band = FREQ_BANDS.get(freq_band, freq_band)
        epochs.filter(l_freq=band[0], h_freq=band[1], verbose=False)

    # Drop "no auditory" (grey) trials so indices in returned epochs are 0, 1, 2, ...
    # and store original file indices for styling/labels.
    keep_indices = []
    for i in range(len(epochs)):
        _color, _label = get_epoch_style(i, condition)
        if _label == "No auditory" or _color == "lightgray":
            continue
        keep_indices.append(i)
    if not keep_indices:
        raise ValueError(
            f"No meaningful trials (all were 'no auditory') for condition '{condition}'."
        )
    epochs = epochs[keep_indices]
    epochs._original_trial_indices = keep_indices

    # Z-score normalization: all trials, all channels, all time points.
    # For each channel, use that channel's mean and std (over all trials and times);
    # then normalize every value in the dataset that belongs to that channel.
    if normalize:
        data = epochs.get_data()  # (n_epochs, n_chs, n_times)
        for c in range(data.shape[1]):
            mean_c = np.mean(data[:, c, :])
            std_c = np.std(data[:, c, :])
            if std_c > 0:
                data[:, c, :] = (data[:, c, :] - mean_c) / std_c
            else:
                data[:, c, :] = data[:, c, :] - mean_c
        epochs._data = data
        epochs._normalized = True  # so plot functions use z-score-friendly scaling

    # Trial selection (applies to the already-filtered set)
    # If trials=[0,1,2] or trials=slice(0,10), we select positions within the
    # filtered set. _original_trial_indices must be updated so each result
    # position still maps to its original file index.
    if trials != 'all':
        orig_before = getattr(epochs, "_original_trial_indices", list(range(len(epochs))))
        epochs = epochs[trials]
        # Which positions in the filtered set were selected?
        if isinstance(trials, slice):
            selected_positions = list(range(*trials.indices(len(orig_before))))
        elif np.isscalar(trials):
            selected_positions = [int(trials)]
        else:
            selected_positions = list(trials)
        # For each selected position, the original file index is orig_before[that_position]
        epochs._original_trial_indices = [orig_before[p] for p in selected_positions]

    return epochs


def plot_all_channels_by_trial(epochs, trial_idx):
    """
    Plot all channels for a single trial in an MNE-style browser window.

    Parameters
    ----------
    epochs : mne.Epochs
        Epochs object (typically from load_eeg_data). trial_idx is the position
        in this object (0 = first trial in the loaded set).
    trial_idx : int
        Index of the trial to plot (0-based within epochs).

    Returns
    -------
    fig
        MNE browser figure (matplotlib or Qt depending on backend). X-axis = time (s).
    """

    # Extract single epoch data
    data = epochs.get_data()[trial_idx]  # shape (n_channels, n_times)

    # Create RawArray
    info = epochs.info.copy()
    raw_trial = mne.io.RawArray(data, info)

    # If data are z-score normalized, use scaling so the plot is not in µV and traces don't overlap.
    # Larger scaling = shorter trace height = less overlap between channels.
    scalings = None
    if getattr(epochs, "_normalized", False):
        scalings = {"eeg": 2.5, "eog": 2.5, "ecg": 2.5, "emg": 2.5, "misc": 2.5}

    fig = raw_trial.plot(
        n_channels=10,
        title=f"Trial {trial_idx} - Channels View (Time in seconds)",
        show_scrollbars=True,
        show_scalebars=False,
        block=True,
        scalings=scalings if scalings is not None else None,
    )

    return fig

def get_epoch_style(epoch_idx, condition):
    """
    Return color and label for a trial given its index in the file and the condition.

    Used to color/label trials in plot_all_trials_by_channel and to decide which
    trials are "no auditory" (grey) in load_eeg_data. epoch_idx must be the
    **original index in the file** (0-based), not the position in filtered epochs.

    Parameters
    ----------
    epoch_idx : int
        Index of the trial in the .set file (0-based).
    condition : str
        Experiment condition: "BLA", "BLT", "P1", "P2", "P3".

    Returns
    -------
    color : str
        Color name (e.g. "blue", "red", "lightgray") for matplotlib.
    label : str
        Human-readable label (e.g. "Auditory onset", "Audio → Tactile (500 ms)",
        "Audio → Missing tactile", "No auditory").
    """

    # ---------- BLA ----------
    if condition == "BLA":
        if epoch_idx % 2 == 0:
            return "blue", "Auditory onset"
        else:
            return "lightgray", "No auditory"

    # ---------- BLT ----------
    if condition == "BLT":
        if epoch_idx % 2 == 0:
            return "blue", "Tactile onset"
        else:
            return "lightgray", "No auditory"

    # ---------- P1 ----------
    if condition == "P1":
        if epoch_idx % 2 == 0:
            return "blue", "Audio → Tactile (500 ms)"
        else:
            return "lightgray", "No auditory"

    # ---------- P2 ----------
    if condition == "P2":
        if epoch_idx in P2_500ms:
            return "blue", "Audio → Tactile (500 ms)"
        elif epoch_idx in P2_2000ms:
            return "red", "Audio → Tactile (2000 ms)"
        else:
            return "lightgray", "No auditory"

    # ---------- P3 ----------
    if condition == "P3":
        if epoch_idx in P3_500ms:
            return "blue", "Audio → Tactile (500 ms)"
        elif epoch_idx in P3_missing:
            return "red", "Audio → Missing tactile"
        else:
            return "lightgray", "No auditory"

    return "black", "Unknown"

def plot_all_trials_by_channel(epochs, channel_name, window_size=10, condition=None, block=True):
    """
    Plot a single channel with one trial per row (MNE browser style), colored by trial type.

    Builds a RawArray where each "channel" is the time series of one trial for the
    given channel, so MNE displays one row per trial. Row colors come from
    get_epoch_style (e.g. blue = Audio→Tactile 500 ms, red = Missing tactile).
    Requires epochs from load_eeg_data (meaningful trials only and _original_trial_indices).
    If epochs were loaded with normalize=True, the same normalized data is plotted with
    scaling tuned for z-score values to avoid overlap and barcode-like display.

    Parameters
    ----------
    epochs : mne.Epochs
        Epochs with only meaningful trials and _original_trial_indices attribute.
    channel_name : str
        Name of the channel to plot (must be in epochs.ch_names).
    window_size : int
        Number of trials (rows) visible at once in the browser. Default 10.
    condition : str | None
        Experiment condition for labels/colors ("BLA", "BLT", "P1", "P2", "P3").
        If None, inferred from epochs.filename.
    block : bool
        If True, execution blocks until the window is closed (for scripts).
        If False, window is shown and execution continues (for notebooks).

    Returns
    -------
    fig
        Browser figure (MNEQtBrowser or MNEBrowseFigure depending on backend).
    """

    if channel_name not in epochs.ch_names:
        raise ValueError(f"Channel {channel_name} not found.")

    if condition is None:
        fname = Path(epochs.filename).name if epochs.filename is not None else ""
        for cond in ["BLA", "BLT", "P1", "P2", "P3"]:
            if cond in fname:
                condition = cond
                break
        if condition is None:
            raise ValueError(
                "Could not infer condition from epochs filename. "
                "Please pass condition='BLA'|'BLT'|'P1'|'P2'|'P3'."
            )

    data_all = epochs.get_data(picks=[channel_name])[:, 0, :]  # (n_trials, n_times)
    n_trials = data_all.shape[0]
    orig_indices = getattr(epochs, "_original_trial_indices", None)

    # All trials are meaningful (load_eeg_data already dropped grey). Use original indices for styling.
    selected_data = []
    selected_labels = []
    selected_colors = []
    selected_orig_indices = []
    for pos in range(n_trials):
        orig_idx = orig_indices[pos] if orig_indices is not None else pos
        color, label = get_epoch_style(orig_idx, condition)
        selected_data.append(data_all[pos])
        selected_labels.append(label)
        selected_colors.append(color)
        selected_orig_indices.append(orig_idx)

    data = np.asarray(selected_data)

    # Map trial condition labels to MNE channel types so Raw.plot can color rows.
    type_pool = ["eeg", "eog", "ecg", "emg", "misc", "resp", "stim"]
    label_to_type = {}
    type_to_color = {}
    ch_types = []

    for trial_pos in range(n_trials):
        color = selected_colors[trial_pos]
        label = selected_labels[trial_pos]
        if label not in label_to_type:
            if len(label_to_type) >= len(type_pool):
                raise ValueError("Too many condition labels to map colors in Raw.plot.")
            next_type = type_pool[len(label_to_type)]
            label_to_type[label] = next_type
            type_to_color[next_type] = color
        ch_types.append(label_to_type[label])

    legend_items = {}
    for label, color in zip(selected_labels, selected_colors):
        if label not in legend_items:
            legend_items[label] = color

    # Include original trial index and category in row name (visible in all backends).
    trial_names = [
        f"Trial {orig} | {label}"
        for orig, label in zip(selected_orig_indices, selected_labels)
    ]
    info = mne.create_info(
        ch_names=trial_names,
        sfreq=epochs.info["sfreq"],
        ch_types=ch_types
    )
    raw_trials = mne.io.RawArray(data, info, verbose=False)

    # Scaling so rows are visually comparable. When normalized, use same value as
    # plot_all_channels_by_trial (2.5) for consistent display and less overlap.
    if getattr(epochs, "_normalized", False):
        amp = 2.5
    else:
        amp = float(np.percentile(np.abs(data), 99.5))
        if amp <= 0:
            amp = 1.0
    scalings = {ch_type: amp for ch_type in set(ch_types)}

    duration = float(epochs.times[-1] - epochs.times[0])
    if duration <= 0:
        duration = 1.0

    # Use block=False first so we can add legend (matplotlib backend); then we block below.
    # With Qt backend, blocking must be done by the browser itself so the window stays open.
    fig = raw_trials.plot(
        n_channels=max(1, min(window_size, n_trials)),
        duration=duration,
        color=type_to_color,
        scalings=scalings,
        group_by="original",
        show_scrollbars=True,
        show_scalebars=False,
        title=f"{channel_name} | {condition} | One trial per row",
        block=False,
    )

    # Custom outside-plot legend overlay is only available with matplotlib backend.
    if hasattr(fig, "canvas") and hasattr(fig, "text"):
        def _apply_legend(_event=None):
            ax_main = fig.mne.ax_main if hasattr(fig, "mne") else None
            if ax_main is None:
                return

            prev_texts = getattr(fig, "_condition_legend_texts", [])
            for txt in prev_texts:
                txt.remove()

            # Reserve a right-side margin once, then re-apply from original geometry.
            refs = getattr(fig, "_legend_layout_refs", None)
            if refs is None:
                ax_hscroll = fig.mne.ax_hscroll if hasattr(fig.mne, "ax_hscroll") else None
                ax_vscroll = fig.mne.ax_vscroll if hasattr(fig.mne, "ax_vscroll") else None
                fig._legend_layout_refs = {
                    "main": ax_main.get_position().bounds,
                    "hscroll": ax_hscroll.get_position().bounds if ax_hscroll is not None else None,
                    "vscroll": ax_vscroll.get_position().bounds if ax_vscroll is not None else None,
                }
                refs = fig._legend_layout_refs

            x0, y0, _, h = refs["main"]
            vscroll_bounds = refs["vscroll"]
            new_right = 0.86
            if vscroll_bounds is not None:
                new_right = min(new_right, vscroll_bounds[0] - 0.015)
            new_right = max(new_right, x0 + 0.25)
            ax_main.set_position([x0, y0, new_right - x0, h])

            hscroll_bounds = refs["hscroll"]
            if hscroll_bounds is not None:
                hx0, hy0, _, hh = hscroll_bounds
                fig.mne.ax_hscroll.set_position([hx0, hy0, new_right - hx0, hh])

            pos = ax_main.get_position()
            x = min(pos.x1 + 0.015, 0.985)
            y = min(pos.y1 - 0.01, 0.98)

            texts = []
            texts.append(
                fig.text(
                    x, y, "Conditions:",
                    va="top", ha="left",
                    fontsize=9,
                    bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=2)
                )
            )
            y -= 0.05
            for label, color in legend_items.items():
                texts.append(
                    fig.text(
                        x, y, f"- {label}",
                        color=color,
                        va="top", ha="left",
                        fontsize=9,
                        bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1)
                    )
                )
                y -= 0.045

            fig._condition_legend_texts = texts

        fig.canvas.mpl_connect("draw_event", _apply_legend)
        fig.canvas.draw()

        # MNE can adjust axes once more right after opening; enforce legend/layout
        # again on the first event-loop tick so it appears outside immediately.
        def _deferred_apply():
            _apply_legend()
            fig.canvas.draw_idle()

        timer = fig.canvas.new_timer(interval=120)
        if hasattr(timer, "single_shot"):
            timer.single_shot = True
        timer.add_callback(_deferred_apply)
        timer.start()
        fig._legend_deferred_timer = timer

    print("Color legend:")
    for label, color in legend_items.items():
        print(f"  {label}: {color}")

    if block:
        if hasattr(fig, "canvas"):
            plt.show(block=True)
        else:
            # Qt backend: block until the browser window is closed so the script doesn't exit
            try:
                from PyQt5.QtWidgets import QApplication
                app = QApplication.instance()
                if app is not None:
                    app.exec()
            except Exception:
                try:
                    from PySide6.QtWidgets import QApplication
                    app = QApplication.instance()
                    if app is not None:
                        app.exec()
                except Exception:
                    pass

    return fig

if __name__ == "__main__":
    # Optional quick demo when running this file directly.
    # Keep it non-fatal if the dataset file is not available locally.
    try:
        demo_epochs = load_eeg_data("P3", 5, "all", "all", normalize=True)
        plot_all_channels_by_trial(demo_epochs, 5)
        plot_all_trials_by_channel(demo_epochs, "Oz")
    except FileNotFoundError as err:
        print(err)
        print("Place your EEGLAB .set files under the 'dataset' folder to run this demo.")
