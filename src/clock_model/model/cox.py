"""Fit the survival model with lifelines.

Two models (per EXP-07/12): a PREDICTION model (all cohort predictors, age-stratified) for the Life-Clock
number, and a TOTAL-EFFECT ATTRIBUTION model (modifiable levers only — no mediator conditioning) for
"Why?"/What-If, so waist/sleep read honestly.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

from clock_model.config import features as F
from clock_model.ingest import impute

NEEDS = ["smoke", "pa_min", "sleep", "waist", "diab", "hbp_told", "copd", "bronchitis",
         "pfq_diff", "mi", "stroke", "chf", "cancer", "educ_hi", "income", "age", "sex", "pm", "dead"]
# Self-reportable extras filled by training-only imputation (EXP-14); not part of the complete-case filter.
# (alcohol is handled by the literature monotonic lever, not fitted here — see config/literature.py.)
IMPUTED = ["bmi", "cigs_day", "sbp"]

# Levers-only design for the total-effect attribution model.
LEVER_COLS = ["smk_former", "smk_current", "activity", "sleep_long", "waist",
              "smk_current_x_young", "activity_x_young", "waist_x_young"]

PENALIZER = 1e-4   # tiny ridge, matches the robust-fit stabilisation used in the experiments


def complete_cohort(df: pd.DataFrame) -> pd.DataFrame:
    df = impute.impute_cohort(df)                 # fill bmi/alc_day/cigs_day (training-only, EXP-14)
    d = df[NEEDS + IMPUTED].copy()
    d = d.dropna(subset=NEEDS)                     # essential predictors must be observed
    return d.reset_index(drop=True)


def _fit(design: pd.DataFrame, T, E) -> dict:
    d = design.copy()
    d["pm"] = np.asarray(T, dtype=float)
    d["dead"] = np.asarray(E, dtype=int)
    cph = CoxPHFitter(penalizer=PENALIZER)
    cph.fit(d, duration_col="pm", event_col="dead", robust=False)
    return cph


def fit_models(df: pd.DataFrame) -> dict:
    """Returns the fitted artefacts needed for the bundle + evaluation."""
    coh = complete_cohort(df)
    standardizer = F.fit_standardizer(F._raw_columns(coh))
    X = F.build_design(coh, standardizer)
    T = coh["pm"].to_numpy(float)
    E = coh["dead"].to_numpy(int)

    prediction = _fit(X, T, E)
    attribution = _fit(X[LEVER_COLS], T, E)

    return {
        "cohort": coh,
        "standardizer": standardizer,
        "design": X,
        "T": T, "E": E,
        "prediction_coefs": {k: float(v) for k, v in prediction.params_.items()},
        "attribution_coefs": {k: float(v) for k, v in attribution.params_.items()},
        "prediction_cph": prediction,
        "n": len(coh), "deaths": int(E.sum()),
    }


def linear_predictor(design: pd.DataFrame, coefs: dict) -> np.ndarray:
    cols = list(coefs.keys())
    beta = np.array([coefs[c] for c in cols])
    return design[cols].to_numpy(float) @ beta
