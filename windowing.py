"""
windowing.py
------------
Time-varying DFA-alpha1: slides a 2-minute window over the RR series and
recomputes DFA-alpha1 (and mean HR) every 5 seconds.

Because RR data is irregularly spaced in time (each sample is a beat, not
a fixed time step), we work off the cumulative time axis built from the
RR intervals themselves, and select beats whose timestamps fall inside
each rolling window.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from dfa import dfa_alpha1


def time_varying_dfa(
    rr_ms_for_hr: np.ndarray,
    rr_ms_for_dfa: np.ndarray,
    window_sec: float = 120.0,
    step_sec: float = 5.0,
    lower_scale: int = 4,
    upper_scale: int = 16,
) -> pd.DataFrame:
    """
    Parameters
    ----------
    rr_ms_for_hr : 1D array of artifact-corrected (but NOT detrended) RR
        intervals in ms. Used for mean HR per window, since detrending
        deliberately removes the slow trends that real HR changes over a
        ramp/session are made of - HR should reflect the true recording.
    rr_ms_for_dfa : 1D array of artifact-corrected AND detrended RR
        intervals, same length and beat-alignment as rr_ms_for_hr. Used
        for the DFA-alpha1 calculation, which needs the detrended series.
    window_sec : rolling window width in seconds (default 120s = 2 min).
    step_sec : grid step in seconds (default 5s).

    Returns
    -------
    DataFrame with columns:
        window_start_s, window_end_s, window_center_s,
        n_beats, mean_hr_bpm, alpha1
    Rows where the window did not contain enough beats for a valid
    DFA-alpha1 estimate have alpha1 = NaN.
    """
    rr_ms_for_hr = np.asarray(rr_ms_for_hr, dtype=float)
    rr_ms_for_dfa = np.asarray(rr_ms_for_dfa, dtype=float)
    if len(rr_ms_for_hr) != len(rr_ms_for_dfa):
        raise ValueError("rr_ms_for_hr and rr_ms_for_dfa must be the same length "
                          "(beat-for-beat aligned).")

    # Timestamps are built from the real (non-detrended) RR series, since
    # detrending only adjusts interval magnitudes for the DFA math, not the
    # actual timing of beats.
    t_end = np.cumsum(rr_ms_for_hr) / 1000.0  # beat timestamps (s), end-of-beat
    t_start_axis = np.concatenate(([0.0], t_end[:-1]))  # start-of-beat times

    total_duration = t_end[-1]
    if total_duration < window_sec:
        raise ValueError(
            f"Recording ({total_duration:.1f}s) shorter than the "
            f"{window_sec:.0f}s analysis window."
        )

    grid_starts = np.arange(0.0, total_duration - window_sec + 1e-9, step_sec)

    rows = []
    for w_start in grid_starts:
        w_end = w_start + window_sec
        idx = np.where((t_start_axis >= w_start) & (t_end <= w_end))[0]

        n_beats = len(idx)
        if n_beats == 0:
            continue

        window_rr_hr = rr_ms_for_hr[idx]
        window_rr_dfa = rr_ms_for_dfa[idx]
        mean_hr = 60000.0 / np.mean(window_rr_hr)

        alpha1 = np.nan
        try:
            alpha1 = dfa_alpha1(window_rr_dfa, lower_scale=lower_scale, upper_scale=upper_scale)
        except ValueError:
            pass  # not enough beats in this window yet - leave as NaN

        rows.append({
            "window_start_s": w_start,
            "window_end_s": w_end,
            "window_center_s": w_start + window_sec / 2.0,
            "n_beats": n_beats,
            "mean_hr_bpm": mean_hr,
            "alpha1": alpha1,
        })

    return pd.DataFrame(rows)
