# CLAUDE.md

This file is guidance for Claude Code working in this EEG analysis repository.

## Quick Reference

| Task | Entry point |
|---|---|
| Load EEG data | `utils.load_eeg_data(condition, subject_id)` |
| Sliding zero-lag correlation | `connectivity.sliding_corr(...)` |
| Lagged directed graph (full) | `lagged_graph.sliding_lagged_xcorr(...)` → `build_directed_graph(...)` |
| Lagged directed graph (anchor) | `lagged_graph.anchor_sliding_lagged_from_epochs(...)` |
| Trial category lookup | `utils.get_epoch_style(orig_idx, condition)` |
| Select trials by style | `connectivity.select_p2_500ms(...)` / `select_by_style(...)` |

## Repository Layout

This is a **notebook-driven** EEG analysis repo:

- **`.py` modules** (`utils.py`, `connectivity.py`, `lagged_graph.py`) are the library.
- **`.ipynb` files** are the primary deliverable — they drive analysis and display results.

### Notebook rules (strictly enforced)

Notebooks contain **only**:
- Imports from `.py` modules
- Logging / config setup
- Top-level calls (`run()`, `train()`, `evaluate()`, etc.)
- Result inspection (prints, plots)

**Never put reusable logic in a notebook.** Before writing any code in a notebook:
1. Check if a `.py` module already owns this logic.
2. If yes, add the function there and import it.
3. If no, create a new `.py` file for it.

As per notebook markdown text: keep them short and concise. no need to write a story out of it. 

## Runtime Environment

- **Data lives on HPC only.** `load_eeg_data` reads from `/storage/ice-shared/psyc4745/dataset/` (hard-coded at `utils.py:120`). Cells that load data will not run on local machines, which typically lack `mne`/`numpy`.
- **No build system, linter, or test suite.** Each module has an `if __name__ == "__main__"` block. Run `python3 lagged_graph.py` to execute the closest thing to a unit test — it asserts direction recovery on a synthetic shifted signal.
- **Live reloading.** Notebooks use `%load_ext autoreload` + `%autoreload 2`, so edits to `.py` files are reflected immediately.

## Architecture: Three-Layer Pipeline

### Layer 1 — Data Loading (`utils.py`)

`load_eeg_data(condition, subject_id, ...)` is the **sole entry point** for data. It:
1. Reads an EEGLAB `.set` file
2. Picks channels (named `CHANNEL_GROUPS` or raw lists)
3. Crops time, band-pass filters (`FREQ_BANDS`), z-score normalizes per channel
4. **Drops "no auditory" grey trials**

**Critical invariants downstream code depends on:**

- Epoch positions are **0-indexed over meaningful trials only** after loading. Original file indices are preserved on `epochs._original_trial_indices` for mapping back.
- `get_epoch_style(orig_idx, condition)` is the **ground truth for trial category** and keys off the *original* index — not the post-drop position. The hand-curated lists `P2_500ms`, `P2_2000ms`, `P3_500ms`, `P3_missing` are converted from 1-based to 0-based at the top of `utils.py`.
- To select trials (e.g., "all P2 500 ms"), use `select_p2_500ms` / `select_by_style` in `connectivity.py` — they operate on `_original_trial_indices`.

### Layer 2 — Sliding Correlation (`connectivity.py`)

`sliding_corr` returns `(n_windows, n_ch, n_ch)` zero-lag Pearson matrices.

ROI utilities (`roi_order`, `reorder_corr`, `draw_roi_heatmap`) block-diagonalize by scalp region using prefix-based `_assign_roi`. **Longer prefixes are checked first** so `FC`/`CP`/`TP`/`PO` are not mis-matched by `F`/`C`/`T`/`P`.

### Layer 3 — Lagged Directed Graphs (`lagged_graph.py`)

`sliding_lagged_xcorr` sweeps lags in `[-L, +L]` samples per channel pair and returns three `(n_windows, n_ch, n_ch)` tensors:

| Tensor | Meaning |
|---|---|
| `best_r` | Signed Pearson r at peak \(\|r\|\) |
| `best_lag` | Lag in samples — **positive = channel i leads j** |
| `r_at_zero` | Zero-lag baseline |

`build_directed_graph` converts one window into a `networkx.DiGraph` via a five-stage filter cascade. The key filter is the **lag-gain** threshold:

$|r_{\text{peak}}| - |r_{\text{at zero}}| \geq \text{lag\_gain\_threshold}$

This rejects edges that are only coupled at zero lag (common-mode / volume-conduction artifacts). Use narrowband filtering (default `FREQ_BAND = 'beta'`) in notebooks to suppress slow common-mode drift that would otherwise push every pair toward \(r \approx 1\).

**Anchor variants** (`anchor_lagged_xcorr`, `anchor_sliding_lagged_from_epochs`) compute only `(anchor, other)` and `(other, anchor)` pairs. They return the same `(n_ch, n_ch)` shape with zeros for non-anchor entries — intentional, so `build_directed_graph`'s filters discard unpopulated entries automatically.

## Conventions and Gotchas

- **Lag direction:** positive `lag[i, j]` means channel `i` leads `j`, i.e., arrow `i → j`. The `lagged_graph.py` sanity block asserts this with a synthetic signal.
- **NaN policy:** zero-variance channels produce NaN entries. `np.isfinite` in `build_directed_graph` and `np.nanmean` in `summary_metrics` handle them silently; a warning is emitted with the affected window count.
- **Plot helpers** (`plot_all_channels_by_trial`, `plot_all_trials_by_channel` in `utils.py`) are interactive MNE browser wrappers for exploratory use — they block the event loop.
- **Don't hardcode new data paths.** If you need a new dataset location, extend `load_eeg_data` rather than adding a path elsewhere.
- **Don't add trial filtering logic outside `connectivity.py`.** Keep `select_by_style` and friends as the single source of truth.