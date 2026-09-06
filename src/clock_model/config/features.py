"""The confirmed feature set: roles, evidence, and the cohort encodings.

Mirrors clock_dev/FEATURES_AND_QUESTIONS.md (the design deliverable) and the witnessed encodings from
experiments/exp01/exp12. Cohort features are *fitted* from NHANES; literature features (config/literature)
are *appended* with citations. Age/sex are NOT here — they go to the life-table baseline.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

# Age split for stratified lever effects (EXP-08): under 55 vs 55+.
YOUNG_CUTOFF = 55

# Cohort-fitted features, in a stable order. role: lever|manage|context. grade from RES-02.
COHORT_FEATURES = [
    ("smk_former",   "lever",   "strong",   "Cox 1972; smoking cohorts"),
    ("smk_current",  "lever",   "strong",   "smoking cohorts"),
    ("activity",     "lever",   "strong",   "WHO PA guidelines; dose-response meta-analyses"),
    ("sleep_long",   "lever",   "moderate", "Cappuccio 2010 sleep U-shape"),
    ("waist",        "lever",   "strong",   "central-adiposity mortality cohorts"),
    ("diabetes",     "manage",  "strong",   "diabetes mortality"),
    ("high_bp",      "manage",  "strong",   "hypertension mortality"),
    ("respiratory",  "manage",  "strong",   "COPD mortality"),
    ("mobility",     "context", "strong",   "functional-status mortality"),
    ("cvd_hx",       "context", "strong",   "prior CVD event"),
    ("cancer_hx",    "context", "strong",   "prior cancer"),
    ("education",    "context", "moderate", "SES gradient"),
    ("income",       "context", "moderate", "SES gradient"),
]
# Age-stratified interaction terms (young = age < YOUNG_CUTOFF), per EXP-08.
INTERACTIONS = ["smk_current", "activity", "waist"]

# Continuous features that are standardized; μ/σ are learned from training and shipped in the bundle.
STANDARDIZED = ["activity", "waist", "income"]


def _raw_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Build the raw (pre-standardization) design columns from the harmonized cohort dataframe."""
    out = pd.DataFrame(index=df.index)
    out["smk_former"]  = (df["smoke"] == 1).astype(float)
    out["smk_current"] = (df["smoke"] == 2).astype(float)
    out["activity"]    = np.log(df["pa_min"].astype(float) + 1.0)          # log MET-minutes
    out["sleep_long"]  = (df["sleep"].astype(float) >= 8.5).astype(float)  # long-sleep flag (short dropped, EXP-12)
    out["waist"]       = df["waist"].astype(float)
    out["diabetes"]    = df["diab"].astype(float)
    out["high_bp"]     = df["hbp_told"].astype(float)
    out["respiratory"] = ((df["copd"] == 1) | (df["bronchitis"] == 1)).astype(float)
    out["mobility"]    = df["pfq_diff"].astype(float)
    out["cvd_hx"]      = ((df["mi"] == 1) | (df["stroke"] == 1) | (df["chf"] == 1)).astype(float)
    out["cancer_hx"]   = df["cancer"].astype(float)
    out["education"]   = df["educ_hi"].astype(float)
    out["income"]      = df["income"].astype(float)
    return out


def fit_standardizer(raw: pd.DataFrame) -> dict:
    """Learn μ/σ for the standardized continuous features (stored in the bundle)."""
    return {c: {"mean": float(raw[c].mean()), "sd": float(raw[c].std(ddof=0) or 1.0)} for c in STANDARDIZED}


def build_design(df: pd.DataFrame, standardizer: dict) -> pd.DataFrame:
    """Full design matrix (standardized + age-stratified interactions). Rows align to df."""
    raw = _raw_columns(df)
    X = raw.copy()
    for c in STANDARDIZED:
        m, s = standardizer[c]["mean"], standardizer[c]["sd"]
        X[c] = (raw[c] - m) / s
    young = (df["age"].astype(float) < YOUNG_CUTOFF).astype(float)
    for c in INTERACTIONS:
        X[f"{c}_x_young"] = X[c] * young
    return X


def design_columns() -> list[str]:
    base = [k for k, *_ in COHORT_FEATURES]
    return base + [f"{c}_x_young" for c in INTERACTIONS]


def evidence_table() -> dict:
    """feature -> {role, grade, citation} for the cohort features (literature features add their own)."""
    return {k: {"role": r, "grade": g, "citation": c} for k, r, g, c in COHORT_FEATURES}
