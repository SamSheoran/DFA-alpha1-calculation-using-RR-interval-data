"""
dfa.py
------
Detrended Fluctuation Analysis, alpha1 component (Peng et al., 1995).

Window width fixed at 4 <= n <= 16 beats.
This follows the standard DFA procedure:

  1. Integrate the (mean-subtracted) RR series -> "profile" y(k).
  2. Divide the profile into non-overlapping boxes of length n.
  3. Fit and remove a local linear trend within each box.
  4. Compute the RMS residual (fluctuation) F(n) for each box size.
  5. alpha1 = slope of log F(n) vs log n, over n in [4, 16].
"""

from __future__ import annotations
import numpy as np


def dfa_alpha1(rr: np.ndarray, lower_scale: int = 4, upper_scale: int = 16,
               n_scales: int = 12) -> float:
    """
    Compute DFA-alpha1 for a beat-to-beat RR (or detrended RR) series.

    Parameters
    ----------
    rr : 1D array of RR intervals (ms), ideally already artifact-corrected
         and detrended.
    lower_scale, upper_scale : box sizes in beats (4 and 16 for alpha1).
    n_scales : number of log-spaced scales to sample between lower and
        upper (Kubios/most implementations use ~10-15 points across the
        alpha1 range; more points = smoother log-log fit).

    Returns
    -------
    alpha1 : float, the DFA scaling exponent.
    """
    rr = np.asarray(rr, dtype=float)
    n = len(rr)

    min_required = upper_scale * 4  # need at least a few boxes at the largest scale
    if n < min_required:
        raise ValueError(
            f"Series too short for DFA-alpha1 window (need >= ~{min_required} "
            f"beats for upper_scale={upper_scale}, got {n})."
        )

    scales = np.unique(
        np.floor(
            np.logspace(np.log10(lower_scale), np.log10(upper_scale), n_scales)
        ).astype(int)
    )
    scales = scales[(scales >= lower_scale) & (scales <= upper_scale)]
    if len(scales) < 2:
        scales = np.arange(lower_scale, upper_scale + 1)

    # Step 1: integrated profile
    y = np.cumsum(rr - np.mean(rr))

    F = np.zeros(len(scales))
    for idx, s in enumerate(scales):
        s = int(s)
        n_boxes = n // s
        if n_boxes < 2:
            F[idx] = np.nan
            continue

        rms_vals = []
        # forward segmentation
        for seg in _segment(y, s, n_boxes, from_end=False):
            rms_vals.append(_box_rms(seg))
        # backward segmentation (standard DFA uses both directions to use
        # all samples, matching the reference implementation)
        for seg in _segment(y, s, n_boxes, from_end=True):
            rms_vals.append(_box_rms(seg))

        F[idx] = np.sqrt(np.mean(np.square(rms_vals)))

    valid = ~np.isnan(F) & (F > 0)
    if valid.sum() < 2:
        raise ValueError("Not enough valid scales to fit DFA slope.")

    log_scales = np.log2(scales[valid])
    log_F = np.log2(F[valid])
    slope, _intercept = np.polyfit(log_scales, log_F, 1)
    return float(slope)


def _segment(y: np.ndarray, s: int, n_boxes: int, from_end: bool):
    """Yield successive non-overlapping boxes of length s from y."""
    n = len(y)
    if from_end:
        start = n - s * n_boxes
        y = y[start:]
    for b in range(n_boxes):
        yield y[b * s: (b + 1) * s]


def _box_rms(segment: np.ndarray) -> float:
    x = np.arange(len(segment))
    coeffs = np.polyfit(x, segment, 1)  # linear local trend (m=1)
    trend = np.polyval(coeffs, x)
    resid = segment - trend
    return np.sqrt(np.mean(resid ** 2))
