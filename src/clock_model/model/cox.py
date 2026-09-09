"""Fit the survival model with lifelines.

Cohort assembly and the shared design helpers. The FITTING itself moved to `model/fit.py` in ONT-02
(stratified by age band x sex, sign-constrained from the ontology); what remains here is the
complete-case filter, the imputation entry point and the linear predictor
number, and a TOTAL-EFFECT ATTRIBUTION model (modifiable levers only — no mediator conditioning) for
"Why?"/What-If, so waist/sleep read honestly.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from clock_model.config import features as F
from clock_model.ingest import impute

NEEDS = ["smoke", "pa_min", "sleep", "waist", "diab", "hbp_told", "copd", "bronchitis",
         "pfq_diff", "mi", "stroke", "chf", "cancer", "educ_hi", "income", "age", "sex", "pm", "dead"]
# Self-reportable extras filled by training-only imputation (EXP-14); not part of the complete-case filter.
# (alcohol is handled by the literature monotonic lever, not fitted here — see config/literature.py.)
IMPUTED = ["cigs_day", "sbp"]   # bmi dropped in REFIT-01 (collinear with waist — see features.py)

# Levers-only design for the total-effect attribution model.



def complete_cohort(df: pd.DataFrame) -> pd.DataFrame:
    df = impute.impute_cohort(df)                 # fill cigs_day/sbp (training-only, EXP-14)
    d = df[NEEDS + IMPUTED].copy()
    d = d.dropna(subset=NEEDS)                     # essential predictors must be observed
    return d.reset_index(drop=True)


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
