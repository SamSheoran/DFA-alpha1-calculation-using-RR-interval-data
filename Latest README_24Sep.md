# DFA-α1 Pipeline

Processes the DFA-α1 using RR interval data:

1. **`data_io.py`** — loads the RR CSV (col 1 = RR in ms, col 2 = flag) (Presently based on Polar H10 derived raw RR CSV file (will be adding capacity to process other format files such as Garmin).
2. **`artifact_correction.py`** — detects/corrects artifacts (approximation of
   Kubios' "automatic" method, based on Lipponen & Tarvainen 2019), reports
   percent-artifact.
3. **`detrending.py`** — smoothness-priors detrending, λ = 500 (Tarvainen et al. 2002).
4. **`dfa.py`** — core DFA-α1 calculation, 4 ≤ n ≤ 16 beat window (Peng et al. 1995).
5. **`windowing.py`** — time-varying analysis: 2-minute rolling window, 5-second
   grid step, HR computed from the un-detrended corrected series.
6. **`main.py`** — compiles all the above steps together to generate the csv with DFA-α1 for every 5 seconds along with a plot.
7. **`dfa_alpha1_pipeline.ipynb`** - **imports and calls** the functions in the `.py` files sitting in this same folder (`data_io.py`, `artifact_correction.py`,
`detrending.py`, `dfa.py`, `windowing.py`).

**Before running:** make sure this notebook is saved in the same folder as those `.py` files, and that you've run `pip install -r requirements.txt`
once (Cell 2 will also do this for you if you have not already).
