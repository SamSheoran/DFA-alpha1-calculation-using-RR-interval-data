"""
main.py
-------
End-to-end pipeline:

  1. Load raw RR CSV.
  2. Artifact detect/correct
  3. Smoothness-priors detrend
  4. Time-varying DFA-alpha1: 2-min rolling window, 5s grid step.
  5. Save results to CSV and plot alpha1 + HR over time.

Usage:
    python main.py path/to/your_data.csv
"""

from __future__ import annotations
import sys
import numpy as np
import matplotlib.pyplot as plt

from data_io import load_rr_csv, remove_offline_dropouts
from artifact_correction import detect_and_correct_artifacts
from detrending import smoothness_priors_detrend
from windowing import time_varying_dfa

ARTIFACT_EXCLUSION_THRESHOLD_PCT = 3.0
DETREND_LAMBDA = 500.0
DFA_LOWER_SCALE = 4
DFA_UPPER_SCALE = 16
WINDOW_SEC = 120.0
STEP_SEC = 5.0


def run(csv_path: str):
    print(f"Loading {csv_path} ...")
    df = load_rr_csv(csv_path)
    print(f"  Loaded {len(df)} beats "
          f"({df['rr_ms'].sum() / 1000 / 60:.1f} minutes of recording).")

    print("Removing sensor-dropout rows (flag column = connection offline) ...")
    df, n_dropped, gap_seconds = remove_offline_dropouts(df)
    print(f"  Removed {n_dropped} dropout rows ({gap_seconds:.1f}s of dead/disconnected "
          f"time). {len(df)} genuine beats remain.")
    rr_raw = df["rr_ms"].to_numpy()

    print("Running artifact detection/correction ...")
    rr_corrected, artifact_mask, pct_artifact = detect_and_correct_artifacts(rr_raw)
    print(f"  Flagged {artifact_mask.sum()} beats as artifacts "
          f"({pct_artifact:.2f}% of total).")

    if pct_artifact > ARTIFACT_EXCLUSION_THRESHOLD_PCT:
        print(f"  *** EXCLUDE: artifact level {pct_artifact:.2f}% exceeds the "
              f"{ARTIFACT_EXCLUSION_THRESHOLD_PCT}% threshold (Rogers et al. 2021b). "
              f"Stopping analysis for this file. ***")
        return None

    print(f"Detrending (smoothness priors, lambda={DETREND_LAMBDA}) ...")
    rr_detrended = smoothness_priors_detrend(rr_corrected, lam=DETREND_LAMBDA)

    print(f"Computing time-varying DFA-alpha1 "
          f"({WINDOW_SEC:.0f}s window, {STEP_SEC:.0f}s step) ...")
    results = time_varying_dfa(
        rr_ms_for_hr=rr_corrected,
        rr_ms_for_dfa=rr_detrended,
        window_sec=WINDOW_SEC,
        step_sec=STEP_SEC,
        lower_scale=DFA_LOWER_SCALE,
        upper_scale=DFA_UPPER_SCALE,
    )

    out_csv = csv_path.rsplit(".", 1)[0] + "_dfa_alpha1_results.csv"
    results.to_csv(out_csv, index=False)
    print(f"Saved results to {out_csv}")

    _plot_results(results, csv_path)
    return results


def _plot_results(results, csv_path: str):
    fig, ax1 = plt.subplots(figsize=(11, 5))

    ax1.plot(results["window_center_s"] / 60.0, results["alpha1"],
              color="tab:blue", label="DFA-alpha1")
    ax1.set_xlabel("Time (min)")
    ax1.set_ylabel("DFA-alpha1", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.axhline(0.75, color="gray", linestyle="--", linewidth=1,
                label="alpha1 = 0.75 (aerobic threshold ref.)")

    ax2 = ax1.twinx()
    ax2.plot(results["window_center_s"] / 60.0, results["mean_hr_bpm"],
              color="tab:red", alpha=0.6, label="HR (bpm)")
    ax2.set_ylabel("Heart rate (bpm)", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    fig.suptitle("Time-varying DFA-alpha1 and Heart Rate")
    fig.tight_layout()

    out_png = csv_path.rsplit(".", 1)[0] + "_dfa_alpha1_plot.png"
    fig.savefig(out_png, dpi=150)
    print(f"Saved plot to {out_png}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py path/to/your_data.csv")
        sys.exit(1)
    run(sys.argv[1])
