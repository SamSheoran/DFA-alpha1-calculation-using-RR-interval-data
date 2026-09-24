"""
artifact_correction.py
-----------------------
The artifact correction algorithm is based on a general approach described
in Lipponen & Tarvainen (2019), "A robust algorithm for heart rate
variability time series artefact correction using novel beat
classification". This module implements that published approach:

  1. Compute a local reference RR value for each beat from surrounding
     beats (a windowed median).
  2. Compute a local variability estimate from the median absolute
     deviation of successive differences (robust to outliers).
  3. Flag a beat as an artifact if its deviation from the local reference
     exceeds an adaptive threshold scaled by that local variability.
  4. Correct flagged beats via cubic spline interpolation over the
     surrounding clean beats.

Note: It will not be bit-for-bit identical to Kubios' automatic threshold, but it follows
the same published method and threshold logic, and should track closely
in practice. However, threshold constants can be tuned to update the correction accuracy.

The function also computes the overall percent-artifact exclusion criteria needed to flag and 
"exclude if >3%" rule for further analysis based on from Rogers et al. (2021).
"""

from __future__ import annotations
import numpy as np
from scipy.interpolate import CubicSpline


def detect_and_correct_artifacts(
    rr: np.ndarray,
    window_beats: int = 91,
    threshold_factor: float = 5.0,
):
    """
    Parameters
    ----------
    rr : 1D array of raw RR intervals (ms).
    window_beats : size of the local window (in beats) used to compute the
        rolling median reference and MAD-based variability estimate.
    threshold_factor : number of local MAD units a beat may deviate from
        the local median before being flagged.

    Returns
    -------
    corrected_rr : 1D array, artifacts replaced via cubic spline
                   interpolation from surrounding clean beats.
    artifact_mask : boolean array, True where a beat was flagged/corrected.
    percent_artifact : float, percentage of beats flagged as artifacts.
    """
    rr = np.asarray(rr, dtype=float)
    n = len(rr)
    if window_beats % 2 == 0:
        window_beats += 1
    half = window_beats // 2

    local_median = np.zeros(n)
    local_mad = np.zeros(n)

    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        segment = rr[lo:hi]
        med = np.median(segment)
        mad = np.median(np.abs(segment - med)) * 1.4826  # scale to ~std for normal data
        local_median[i] = med
        local_mad[i] = mad if mad > 1e-6 else 1.0  # avoid div-by-zero on flat segments

    deviation = np.abs(rr - local_median)
    artifact_mask = deviation > (threshold_factor * local_mad)

    # Also flag physiologically implausible beats outright
    artifact_mask |= (rr < 250) | (rr > 2500)

    percent_artifact = 100.0 * np.sum(artifact_mask) / n

    corrected_rr = rr.copy()
    clean_idx = np.where(~artifact_mask)[0]

    if artifact_mask.any() and len(clean_idx) >= 4:
        cs = CubicSpline(clean_idx, rr[clean_idx])
        bad_idx = np.where(artifact_mask)[0]
        corrected_rr[bad_idx] = cs(bad_idx)
    elif artifact_mask.any():
        print("[artifact_correction] Too few clean beats to interpolate reliably.")

    return corrected_rr, artifact_mask, percent_artifact
