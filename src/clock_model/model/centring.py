"""Per-country centring reference (generalized from experiments/exp13).

Relative risk is centred on the target country's *average person*, so an average national resident reads
RR ≈ 1.0 and remaining years ≈ the national life expectancy. Smoking & weight come from the country's
prevalence (Eurostat/EHIS); every other covariate uses the cohort mean (owner-approved fallback).
"""
from __future__ import annotations
import math
import numpy as np
from scipy.stats import norm

from clock_model.config import features as F


def cohort_rates(coh) -> dict:
    """Population rates/means for the fallback covariates, from the training cohort."""
    r = F._raw_columns(coh)
    return {
        "smk_former": float((coh["smoke"] == 1).mean()),
        "sleep_long": float((coh["sleep"].astype(float) >= 8.5).mean()),
        "diabetes": float(coh["diab"].mean()),
        "high_bp": float(coh["hbp_told"].mean()),
        "respiratory": float(((coh["copd"] == 1) | (coh["bronchitis"] == 1)).mean()),
        "mobility": float(coh["pfq_diff"].mean()),
        "cvd_hx": float(((coh["mi"] == 1) | (coh["stroke"] == 1) | (coh["chf"] == 1)).mean()),
        "cancer_hx": float(coh["cancer"].mean()),
        "education": float(coh["educ_hi"].mean()),
        # central-overweight prevalence in the cohort (IDF waist thresholds), for the weight z-shift
        "cohort_overweight": float(((coh["waist"] > np.where(coh["sex"] == 1, 94, 80))).mean()),
    }


def reference_vector(coefs: dict, rates: dict, prevalence: dict, age: int) -> float:
    """LP_reference for a country at a given age (young interactions scaled by the age flag).

    prevalence: {'current_smoking': fraction, 'overweight_plus': fraction} for the country (or None → fallback).
    """
    young = 1.0 if age < F.YOUNG_CUTOFF else 0.0
    smk_current = prevalence.get("current_smoking") if prevalence else None
    ow_plus = prevalence.get("overweight_plus") if prevalence else None

    # weight z-shift: match the country's overweight prevalence to a standardized waist mean shift
    if ow_plus is not None:
        pc = min(max(rates["cohort_overweight"], 1e-3), 1 - 1e-3)
        waist_z = float(norm.ppf(min(max(ow_plus, 1e-3), 1 - 1e-3)) - norm.ppf(pc))
    else:
        waist_z = 0.0

    E = {
        "smk_former": rates["smk_former"],
        "smk_current": smk_current if smk_current is not None else 0.0,
        "activity": 0.0,                      # standardized cohort mean
        "sleep_long": rates["sleep_long"],
        "waist": waist_z,
        "diabetes": rates["diabetes"], "high_bp": rates["high_bp"], "respiratory": rates["respiratory"],
        "mobility": rates["mobility"], "cvd_hx": rates["cvd_hx"], "cancer_hx": rates["cancer_hx"],
        "education": rates["education"], "income": 0.0,
        "smk_current_x_young": (smk_current if smk_current is not None else 0.0) * young,
        "activity_x_young": 0.0,
        "waist_x_young": waist_z * young,
    }
    return sum(coefs[k] * E.get(k, 0.0) for k in coefs)
