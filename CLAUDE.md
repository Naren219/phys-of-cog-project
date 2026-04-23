# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Runtime environment

- This is a notebook-driven EEG analysis repo. The Python modules (`utils.py`, `connectivity.py`, `lagged_graph.py`) are the library; the `.ipynb` files drive analysis and are the primary deliverable.
- Dataset files live on an HPC share, not in the repo: `load_eeg_data` reads from `/storage/ice-shared/psyc4745/dataset/` (hard-coded in [utils.py:120](utils.py#L120)). Cells that load data therefore only run on the cluster; local machines typically don't even have `mne`/`numpy` installed.
- There is no build system, lint config, or test suite. Each module has an `if __name__ == "__main__"` sanity block — run e.g. `python3 lagged_graph.py` to execute it. The `lagged_graph.py` block asserts direction recovery on a synthetic shifted signal and is the closest thing to a unit test.
- Notebooks use `%load_ext autoreload` + `%autoreload 2`, so editing the `.py` modules is reflected live.

## High-level architecture

The pipeline is three layers, each in its own module:

**Layer 1 — data loading ([utils.py](utils.py)).** `load_eeg_data(condition, subject_id, ...)` is the single entry point. It reads an EEGLAB `.set` file, applies channel picking (supports named `CHANNEL_GROUPS` or raw channel lists), time cropping, band-pass filtering (`FREQ_BANDS`), per-channel z-score normalization, then **drops "no auditory" grey trials**. Two non-obvious invariants the rest of the code relies on:

- After loading, epoch positions are 0-indexed over *meaningful trials only*. The original file indices are preserved on `epochs._original_trial_indices` so downstream code can still map a position back to its original index for styling/filtering.
- `get_epoch_style(orig_idx, condition)` is the ground truth for trial category (Audio→Tactile 500 ms, 2000 ms, Missing tactile, No auditory). It keys off the **original** index — hand-curated lists `P2_500ms`, `P2_2000ms`, `P3_500ms`, `P3_missing` (converted from 1-based to 0-based at the top of `utils.py`). If you need "all P2 500 ms trials," use `select_p2_500ms` / `select_by_style` in `connectivity.py` — they operate on `_original_trial_indices`.

**Layer 2 — sliding correlation ([connectivity.py](connectivity.py)).** `sliding_corr` computes `(n_windows, n_ch, n_ch)` zero-lag Pearson matrices. ROI utilities (`roi_order`, `reorder_corr`, `draw_roi_heatmap`) block-diagonalize the matrix by scalp region using prefix-based `_assign_roi`; longer prefixes are checked first so `FC`/`CP`/`TP`/`PO` are not shadowed by `F`/`C`/`T`/`P`.

**Layer 3 — lagged directed graphs ([lagged_graph.py](lagged_graph.py)).** `sliding_lagged_xcorr` sweeps lags in `[-L, +L]` samples per pair and records three `(n_windows, n_ch, n_ch)` tensors: `best_r` (signed r at peak |r|), `best_lag` (direction in samples — positive means i leads j), and `r_at_zero` (zero-lag baseline). `build_directed_graph` converts one window into a `networkx.DiGraph` via a five-stage filter cascade; the critical step is the **lag-gain** filter `|r_peak| - |r_at_zero| >= lag_gain_threshold`, which rejects edges that are only coupled at zero lag (i.e., common-mode / volume-conduction artifacts). Narrowband filtering (default `FREQ_BAND = 'beta'`) is used in the notebooks to suppress slow common-mode drift that would otherwise drive every pair toward r≈1.

Anchor variants (`anchor_lagged_xcorr`, `anchor_sliding_lagged_from_epochs`) compute only `(anchor, other)` and `(other, anchor)` pairs. They return the same `(n_ch, n_ch)` shape as the full versions, with zeros for non-anchor entries — this is intentional, so `build_directed_graph`'s `min_abs_lag_samples` and `r_threshold` filters automatically discard the unpopulated entries without a separate code path.

## Conventions worth knowing

- Direction of `lag[i, j]`: positive means channel `i` leads channel `j`, i.e. arrow `i → j` in the graph. The sanity block at the bottom of [lagged_graph.py](lagged_graph.py) pins this convention down with an assertion.
- NaN policy: zero-variance channels produce NaN entries in the output tensors; downstream filters (`np.isfinite` in `build_directed_graph`, `np.nanmean` in `summary_metrics`) handle them silently. A warning is emitted with the count of affected windows.
- Plot helpers in `utils.py` (`plot_all_channels_by_trial`, `plot_all_trials_by_channel`) are interactive MNE browser wrappers with a custom matplotlib legend overlay — they are for exploratory use and block the event loop.
