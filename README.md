# DFA-alpha1-calculation-using-RR-interval-data
A pipeline to calculate DFA-alpha1 from raw RR data (includes the artifact correction prior to detrending) and generates a time-varying DFA-α1 analysis for a rolling 2-minute HR window.

There are 6 main functions that this script performs as part of DFA-α1 workflow:

1. **`data_io.py`** — loads the RR CSV (col 1 = RR in ms, col 2 = flag).   #This RR datafile is from Polar sensor. Will update this as we get more datafile types.
2. **`artifact_correction.py`** — detects/corrects artifacts (approximation of
   Kubios' "automatic" method, based on Lipponen & Tarvainen 2019), reports
   percent-artifact.
3. **`detrending.py`** — smoothness-priors detrending, λ = 500 (Tarvainen et al. 2002).
4. **`dfa.py`** — core DFA-α1 calculation, 4 ≤ n ≤ 16 beat window (Peng et al. 1995).
5. **`windowing.py`** — time-varying analysis: 2-minute rolling window, 5-second
   grid step, HR computed from the un-detrended corrected series.
6. **`main.py`** — orchestrates the above, applies the >3% artifact exclusion
   rule (Rogers et al. 2021b), writes a results CSV + plot.

**Artifact correction is an approximation.** Kubios' exact "automatic"
  algorithm internals are not fully public. This implementation follows the
  published Lipponen & Tarvainen (2019) approach (rolling median + MAD-based
  adaptive threshold + cubic spline correction), but the exact % artifact and
  which beats get flagged may differ slightly from Kubios.
- **DFA scale sampling**: uses 12 log-spaced scales between 4 and 16 beats
  by default (adjustable via `n_scales` in `dfa.py`). Kubios' internal scale
  density isn't published either — this is a standard choice from the
  literature, but if your alpha1 values are consistently off from Kubios by
  a small constant amount, try adjusting `n_scales`.
- **Flag column (col 2)**: currently loaded but unused, since its meaning
  wasn't confirmed. If it's the device's own artifact flag, it could be
  used as an additional check/cross-validation against the artifact
  correction implemented here, or blended into detection.
- **Minimum window length**: DFA-α1 needs a decent number of beats to be
  stable at scale 16; very short recordings, or the first/last partial
  windows, will show `NaN` in the results if a window is too sparse —
  this is expected, not a bug.
   
