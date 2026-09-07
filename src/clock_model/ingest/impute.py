"""Training-only imputation of self-reported features that NHANES leaves partly blank.

The shipped model gains two self-reportable cohort predictors — BMI and current-smoking dose
(cigarettes/day) — validated out-of-sample in clock_dev EXP-14/14b. In the cohort these carry blanks
(BMI ~5%; cigarettes only asked of *current* smokers), so to train on them without dropping rows we fill
the blanks by iterative ridge regression on the always-present predictors (MICE-lite). This is a
TRAINING-ONLY step: at prediction time the questionnaire supplies every value, so the Rust runtime never
imputes and the bundle ships only coefficients + standardizer. (Alcohol is handled separately, by the
literature monotonic lever in config/literature.py, not fitted here — its cohort signal is confounded by
sick-quitter reverse causation and would read as protective.)

`cigs_day` is a current-smoker dose: never/former smokers smoke 0/day now, so it is set to 0 for them
(an observed 0, not a value to impute) — mirroring the questionnaire, which only asks CIGS of smokers.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

# Always-present (low-missing) columns used to predict the blanks.
PREDICTORS = ["age", "sex", "waist", "smoke", "pa_min", "sleep", "educ_hi", "diab", "hbp_told"]
RIDGE = 1.0
N_ITER = 5


def _zero_cigs_for_nonsmokers(df: pd.DataFrame) -> pd.Series:
    """cigs_day = reported dose for current smokers (smoke==2), 0 for never/former (smoke in {0,1})."""
    cig = pd.to_numeric(df["cigs_day"], errors="coerce")
    return cig.where(df["smoke"] == 2, 0.0)


def impute_cohort(df: pd.DataFrame, targets=("bmi", "cigs_day", "sbp")) -> pd.DataFrame:
    """Return a copy of df with `targets` filled by iterative ridge regression on PREDICTORS.

    Rows still missing a PREDICTOR are left to the caller's dropna (the essential columns); imputation
    only touches the target columns. Deterministic, numpy-only.
    """
    out = df.copy()
    out["cigs_day"] = _zero_cigs_for_nonsmokers(out)

    targets = [t for t in targets if t in out.columns]
    # Predictor design (mean-filled so a blank predictor never blocks a regression); constant bias added.
    P = np.column_stack([pd.to_numeric(out[c], errors="coerce").astype(float) for c in PREDICTORS])
    P = np.where(np.isnan(P), np.nanmean(P, axis=0), P)
    P = np.column_stack([np.ones(len(P)), P])  # bias

    # Initialise targets at their observed mean; remember which cells were blank.
    cols = {t: pd.to_numeric(out[t], errors="coerce").astype(float).to_numpy() for t in targets}
    miss = {t: np.isnan(cols[t]) for t in targets}
    filled = {t: np.where(miss[t], np.nanmean(cols[t]), cols[t]) for t in targets}

    for _ in range(N_ITER):
        for t in targets:
            if not miss[t].any():
                continue
            # Predict t from PREDICTORS + the other (currently-filled) targets.
            extra = [filled[o] for o in targets if o != t]
            A = np.column_stack([P] + extra) if extra else P
            obs = ~miss[t]
            Ao, yo = A[obs], filled[t][obs]
            G = Ao.T @ Ao + RIDGE * np.eye(A.shape[1])
            w = np.linalg.solve(G, Ao.T @ yo)
            pred = A @ w
            filled[t] = np.where(miss[t], pred, filled[t])

    # Clip imputed draws to physically sensible ranges (regression can overshoot slightly).
    bounds = {"bmi": (12.0, 70.0), "alc_day": (0.0, 30.0), "cigs_day": (0.0, 80.0), "sbp": (70.0, 240.0)}
    for t in targets:
        col = filled[t]
        if t in bounds:
            lo, hi = bounds[t]
            col = np.clip(col, lo, hi)
        out[t] = col
    return out
