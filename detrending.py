"""
detrending.py
-------------
Smoothness-priors detrending (Tarvainen, Ranta-aho & Karjalainen, 2002),
the same method Kubios uses under the "Smoothness priors" setting.

The method estimates a smooth, slowly-varying trend z_trend in the RR
series and subtracts it, acting as a time-varying high-pass filter. The
regularization parameter lambda controls the cutoff: larger lambda ->
smoother trend removed -> lower cutoff frequency. Kubios' default is
lambda = 500, which is also the value used in literature.

Math:
    Minimize  || z - z_trend ||^2 + lambda^2 || D2 @ z_trend ||^2
    =>  z_trend = (I + lambda^2 * D2.T @ D2)^(-1) @ z
    detrended   = z - z_trend

where D2 is the discrete second-order difference operator.

We use a sparse solve since D2 is banded and N can be large for a full
recording.
"""

from __future__ import annotations
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


def _second_diff_matrix(n: int) -> sparse.csc_matrix:
    """Build the (n-2) x n second-order difference operator D2."""
    e = np.ones(n)
    D2 = sparse.diags([e[:-2], -2 * e[:-1], e], offsets=[0, 1, 2], shape=(n - 2, n))
    return D2.tocsc()


def smoothness_priors_detrend(rr: np.ndarray, lam: float = 500.0) -> np.ndarray:
    """
    Apply smoothness-priors detrending to an RR-interval series.

    Parameters
    ----------
    rr : 1D array of RR intervals (ms), assumed already artifact-corrected.
    lam : smoothing parameter (Kubios default = 500).

    Returns
    -------
    detrended : 1D array, same length as rr, with the slow trend removed.
                Mean of the original series is added back so the output
                stays in physiologically meaningful RR units (Kubios does
                the same - it doesn't zero-center the output).
    """
    rr = np.asarray(rr, dtype=float)
    n = len(rr)
    if n < 5:
        raise ValueError("Series too short to detrend (need at least 5 samples).")

    D2 = _second_diff_matrix(n)
    I = sparse.identity(n, format="csc")
    A = (I + (lam ** 2) * (D2.T @ D2)).tocsc()

    z_trend = spsolve(A, rr)
    detrended = rr - z_trend + np.mean(rr)
    return detrended
