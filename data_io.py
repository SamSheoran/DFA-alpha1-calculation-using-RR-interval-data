"""
data_io.py
----------
Loads the raw RR-interval CSV.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def load_rr_csv(path: str) -> pd.DataFrame:
    """
    Load an RR-interval CSV and return a DataFrame with columns:
        rr_ms   : float, RR interval in milliseconds
        flag    : the raw second-column value (bool/str/int, unmodified)

    Handles files with or without a header row.
    """
    # Peek at the first row to guess whether there's a header
    preview = pd.read_csv(path, nrows=5, header=None)

    first_cell = str(preview.iloc[0, 0]).strip()
    has_header = not _looks_numeric(first_cell)

    if has_header:
        df = pd.read_csv(path)
        df = df.iloc[:, :2]
        df.columns = ["rr_ms", "flag"]
    else:
        df = pd.read_csv(path, header=None)
        df = df.iloc[:, :2]
        df.columns = ["rr_ms", "flag"]

    df["rr_ms"] = pd.to_numeric(df["rr_ms"], errors="coerce")

    # Normalize the flag column to a real boolean where possible, but keep
    # original values around too in case they're not simply true/false.
    df["flag_raw"] = df["flag"]
    df["flag"] = df["flag"].apply(_coerce_bool)

    n_before = len(df)
    df = df.dropna(subset=["rr_ms"]).reset_index(drop=True)
    n_after = len(df)
    if n_after < n_before:
        print(f"[data_io] Dropped {n_before - n_after} rows with non-numeric RR values.")

    if (df["rr_ms"] < 200).any() or (df["rr_ms"] > 3000).any():
        n_out = ((df["rr_ms"] < 200) | (df["rr_ms"] > 3000)).sum()
        print(f"[data_io] Warning: {n_out} RR values fall outside a plausible "
              f"200-3000 ms physiological range. They are NOT removed here; "
              f"artifact correction downstream should handle them.")

    return df


def remove_offline_dropouts(df: pd.DataFrame) -> tuple:
    """
    Remove beats flagged True in the 'flag' column, which (confirmed from
    real data) marks sensor connection dropouts: the recorded 'duration'
    for these rows is dead/disconnected time (typically several seconds
    to tens of seconds), NOT a genuine interbeat interval. These rows must
    be dropped from the beat sequence rather than artifact-corrected,
    since there is no real single heartbeat to reconstruct - correcting
    them via interpolation like a normal artifact would fabricate beats
    that never happened.

    Returns
    -------
    (clean_df, n_dropped, total_gap_seconds)
    """
    is_dropout = df["flag"] == True  # noqa: E712 (explicit for clarity with nullable bool)
    n_dropped = int(is_dropout.sum())
    total_gap_seconds = float(df.loc[is_dropout, "rr_ms"].sum() / 1000.0)

    clean_df = df.loc[~is_dropout].reset_index(drop=True)
    return clean_df, n_dropped, total_gap_seconds


def _looks_numeric(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def _coerce_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)) and not pd.isna(v):
        return bool(v)
    if isinstance(v, str):
        vs = v.strip().lower()
        if vs in ("true", "t", "1", "yes"):
            return True
        if vs in ("false", "f", "0", "no"):
            return False
    return np.nan
