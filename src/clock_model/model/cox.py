"""Fit the survival model with lifelines.

Two models (per EXP-07/12): a PREDICTION model (all cohort predictors, with age-*interaction* terms —
NOT age-stratified, and age itself is absent from the fit: see M11) for the Life-Clock
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
IMPUTED = ["cigs_day", "sbp"]   # bmi dropped in REFIT-01 (collinear with waist — see features.py)

# Levers-only design for the total-effect attribution model.
LEVER_COLS = ["smk_former", "smk_current", "activity", "sleep_long", "waist",
              "smk_current_x_young", "activity_x_young", "waist_x_young"]

PENALIZER = 1e-4   # tiny ridge, matches the robust-fit stabilisation used in the experiments


def complete_cohort(df: pd.DataFrame) -> pd.DataFrame:
    df = impute.impute_cohort(df)                 # fill cigs_day/sbp (training-only, EXP-14)
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



def fit_models(df):
    """Removed in ONT-02 — use `clock_model.model.fit.fit_models`.

    The old fit had no age/sex strata and no sign constraints, so it reported smoking as protective
    for the under-55s and adiposity as protective for everyone. Kept only as this refusal because
    `train.py` went on calling it silently for one commit after the rewrite — which is exactly what
    a dead function that still returns something invites.
    """
    raise RuntimeError(
        "cox.fit_models was replaced by clock_model.model.fit.fit_models (ONT-02): the old fit was "
        "unstratified and unconstrained, and reported smoking as protective for the young."
    )

def linear_predictor(design: pd.DataFrame, coefs: dict) -> np.ndarray:
    cols = list(coefs.keys())
    beta = np.array([coefs[c] for c in cols])
    return design[cols].to_numpy(float) @ beta
