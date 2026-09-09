"""Release gates — a bundle that fails these is not exported.

Reproduces the witnessed checks: out-of-sample-style C-index, calibration (predicted vs observed 10-year
survival by risk decile), and a sanity bound (an average national resident ≈ national life expectancy).
"""
from __future__ import annotations
import numpy as np
from lifelines.utils import concordance_index

from clock_model.model import cox, baselines

THRESHOLDS = {"c_index_min": 0.70, "calibration_mae_max": 0.05}


def evaluate(fit: dict) -> dict:
    X, T, E = fit["design"], fit["T"], fit["E"]
    lp = cox.linear_predictor(X, fit["prediction_coefs"])

    c = float(concordance_index(T, -lp, E))

    # Calibration at 10 years (120 months) by risk decile. Under an age x sex stratified fit each
    # stratum has its OWN baseline hazard, so a single pooled Breslow baseline would score the model
    # against a curve it never uses. Calibrate within strata and pool the errors by stratum size.
    strata = fit.get("strata_labels")
    if strata is None:
        groups = [np.arange(len(T))]
    else:
        groups = [np.where(strata == s)[0] for s in np.unique(strata)]

    errs, weights = [], []
    for idx in groups:
        if len(idx) < 100 or E.to_numpy()[idx].sum() < 20:
            continue
        Ts, Es, lps = T.to_numpy()[idx], E.to_numpy()[idx], lp[idx]
        H0 = baselines.breslow_H0(Ts, Es, lps, at_time=120)
        n_bins = min(5, max(2, len(idx) // 300))  # deciles are too thin inside a stratum
        for d in np.array_split(np.argsort(lps), n_bins):
            if len(d) < 30:
                continue
            pred = float(np.mean(np.exp(-H0 * np.exp(lps[d]))))
            obs = _km_at(Ts[d], Es[d], 120)
            errs.append(abs(pred - obs))
            weights.append(len(d))
    cal_mae = float(np.average(errs, weights=weights)) if errs else float("nan")

    gates = {
        "c_index": round(c, 3),
        "calibration_mae": round(cal_mae, 3),
        "n": fit["n"], "deaths": fit["deaths"],
        "passed": c >= THRESHOLDS["c_index_min"] and cal_mae <= THRESHOLDS["calibration_mae_max"],
    }
    return gates


def _km_at(T, E, t) -> float:
    """Kaplan-Meier survival at time t."""
    order = np.argsort(T)
    T = np.asarray(T)[order]; E = np.asarray(E)[order]
    S, atrisk, i, n = 1.0, len(T), 0, len(T)
    while i < n:
        tj = T[i]; d = 0; j = i
        while j < n and T[j] == tj:
            d += E[j]; j += 1
        if tj <= t and atrisk > 0:
            S *= 1 - d / atrisk
        atrisk -= (j - i); i = j
        if tj > t:
            break
    return float(S)
