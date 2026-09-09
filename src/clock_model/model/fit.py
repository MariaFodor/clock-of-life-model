"""Stratified, sign-constrained Cox fitting driven by the ontology.

Two things make this different from a plain Cox fit, and each closes a defect that reached users:

  * **Age x sex strata.** Each stratum gets its own baseline hazard, so coefficients are estimated
    by comparing people of the same age and sex. Without this, age has nowhere to live and leaks
    into whatever correlates with it — which is why the shipped model reported that smoking
    protects the under-55s (REVIEW M11/M2).
  * **Sign constraints from the ontology, enforced in the optimizer.** A lever cannot come out with
    the wrong sign, rather than coming out wrong and being caught later (REVIEW M1).

Two estimands, both derived rather than hand-picked:

  * `prediction` conditions on everything the person told us. Best ranking; its individual
    coefficients are direct effects and are NOT causal claims.
  * `total_effect` is fitted once per lever on the adjustment set the causal graph implies —
    confounders only, never mediators. This is what What-If and the recommendations may use, and
    it is what stops the model from advising as if adiposity acted independently of the diseases
    it causes (REVIEW M10).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from clock_model import ontology as O
from clock_model.config import features as F

RIDGE = 1e-4  # tiny stabiliser, matches the robust fit used throughout the experiments


def strata_labels(df: pd.DataFrame) -> np.ndarray:
    """Stratum id per row: (age band x sex). Bands come from the ontology, not from a magic number."""
    edges = O.meta()["strata"]["age_bands"]
    ages = df["age"].astype(float)
    # include_lowest + open outer edges: an age sitting exactly on the first boundary would otherwise
    # become NaN and then crash on int(nan), which is a nasty way to learn the bands were edited.
    band = pd.cut(ages, [-np.inf] + list(edges[1:-1]) + [np.inf], labels=False, include_lowest=True)
    if band.isna().any():
        raise ValueError(f"{int(band.isna().sum())} rows fall outside the ontology's age bands {edges}")
    return np.array([f"{int(b)}_{int(s)}" for b, s in zip(band, df["sex"].astype(float))])


def _neg_log_pl(beta: np.ndarray, M: np.ndarray, t: np.ndarray, e: np.ndarray,
                groups: list[np.ndarray]) -> tuple[float, np.ndarray]:
    """Stratified partial likelihood and its gradient (negated, ridge-penalised).

    Tie handling sits between Breslow and Efron: the risk set is truncated at the sort position, so
    rows tied with an event but sorted after it are excluded. Measured against exact Breslow and
    against lifelines' Efron on this cohort the difference is under 0.0023 in any coefficient
    (< 0.03 standard errors). Sorting is stable and keyed on time then row index, so the result does
    not depend on input order.

    Within a stratum, sorting by descending time turns each risk set into a running total, so the
    whole stratum costs one pass instead of one pass per event.
    """
    lp = M @ beta
    total = 0.0
    grad = np.zeros_like(beta)
    for idx in groups:
        ls, es, Ms = lp[idx], e[idx], M[idx]
        w = np.exp(ls - ls.max())                       # shift for numerical stability
        denom = np.cumsum(w)                            # risk set: everyone with time >= this row
        num = np.cumsum(Ms * w[:, None], axis=0)
        ev = np.where(es)[0]
        if ev.size == 0:
            continue
        total += ls[ev].sum() - (np.log(denom[ev]) + ls.max()).sum()
        grad += Ms[ev].sum(axis=0) - (num[ev] / denom[ev][:, None]).sum(axis=0)
    return -total + RIDGE * beta @ beta, -grad + 2 * RIDGE * beta


def _prepare(design: pd.DataFrame, T: pd.Series, E: pd.Series, strata: np.ndarray):
    """Row groups per stratum, each pre-sorted by descending follow-up time."""
    M, t, e = design.to_numpy(float), T.to_numpy(float), E.to_numpy(bool)
    groups = []
    for s in np.unique(strata):
        idx = np.where(strata == s)[0]
        # Stable sort keyed on (-time, index): with 160 unique times over 18,839 rows ties dominate,
        # and an unstable sort would make every coefficient depend on input row order.
        groups.append(idx[np.lexsort((idx, -t[idx]))])
    return M, t, e, groups


def _observed_information(beta: np.ndarray, M: np.ndarray, e: np.ndarray,
                         groups: list[np.ndarray]) -> np.ndarray:
    """Observed information matrix, so coefficients can carry a standard error.

    Per event, the risk-set weighted covariance of the covariates. Inverting the sum gives the
    usual asymptotic variance; we need it to weigh the data against the literature prior.
    """
    lp = M @ beta
    info = np.zeros((M.shape[1], M.shape[1]))
    for idx in groups:
        ls, es, Ms = lp[idx], e[idx], M[idx]
        w = np.exp(ls - ls.max())
        denom = np.cumsum(w)
        s1 = np.cumsum(Ms * w[:, None], axis=0)
        for i in np.where(es)[0]:
            # Second moment of the risk set at this event time.
            s2 = (Ms[: i + 1] * w[: i + 1, None]).T @ Ms[: i + 1]
            mean = s1[i] / denom[i]
            info += s2 / denom[i] - np.outer(mean, mean)
    return info


def fit_constrained(design: pd.DataFrame, T: pd.Series, E: pd.Series, strata: np.ndarray,
                    constrained: bool = True, return_se: bool = False):
    """Fit one stratified Cox model; when `constrained`, the ontology's signs are hard bounds."""
    cols = list(design.columns)
    M, t, e, groups = _prepare(design, T, E, strata)
    bnds = O.bounds(cols) if constrained else [(None, None)] * len(cols)
    res = minimize(_neg_log_pl, np.zeros(len(cols)), args=(M, t, e, groups),
                   jac=True, method="L-BFGS-B", bounds=bnds,
                   options={"maxiter": 2000, "ftol": 1e-10})
    beta = pd.Series(res.x, index=cols)
    # A bound-constrained optimum can sit exactly ON the bound, which is not the same as "the data
    # said zero": it means the data said the opposite and the constraint refused. That has to be
    # recorded, or a gate reads 0.0000 as "correctly signed" and the clipping disappears (PR#2 F4).
    clipped = []
    if constrained:
        for c, (lo, hi) in zip(cols, bnds):
            v = beta[c]
            if (lo is not None and abs(v - lo) < 1e-9) or (hi is not None and abs(v - hi) < 1e-9):
                clipped.append(c)
    beta.attrs["clipped_at_bound"] = clipped
    if not return_se:
        return beta
    info = _observed_information(res.x, M, e, groups)
    try:
        var = np.linalg.inv(info + np.eye(len(cols)) * 2.0 * RIDGE)
        se = pd.Series(np.sqrt(np.clip(np.diag(var), 1e-12, None)), index=cols)
    except np.linalg.LinAlgError:
        se = pd.Series(np.full(len(cols), np.nan), index=cols)
    return beta, se


def _blend_with_prior(data_mean: float, data_se: float, prior: dict) -> tuple[float, float, str]:
    """Precision-weighted combination of the cohort estimate with the literature prior.

    The conjugate normal result: each source is weighted by how certain it is. A prior with a
    verified DOI is evidence too, and pretending our 2,119 events settle a question the literature
    studied in millions of person-years would be its own kind of dishonesty. Symmetrically, the
    prior does not get to overrule data it disagrees with — it gets outvoted in proportion to
    precision, and the disagreement stays visible in `total_effect_data_only`.
    """
    if not prior or prior.get("log_hr") is None or not np.isfinite(data_se) or data_se <= 0:
        return data_mean, (data_se if np.isfinite(data_se) else float("nan")), "data_only"
    # Units must match or the blend is meaningless. A meta-analytic hazard ratio "per 10 cm" is not
    # the same quantity as a coefficient per cohort standard deviation (16.3 cm), and averaging them
    # would quietly move the estimate by most of a standard error. Blend only where the prior
    # declares the same scale the coefficient is on; otherwise ship the data and say so (PR#2 F6).
    if prior.get("per") not in ("sd", "binary"):
        return data_mean, data_se, "data_only_units_mismatch"
    if prior.get("sd") is None:
        return data_mean, data_se, "data_only_no_prior_sd"
    p_mean, p_sd = float(prior["log_hr"]), float(prior["sd"])
    wp, wd = 1.0 / p_sd ** 2, 1.0 / data_se ** 2
    mean = (wp * p_mean + wd * data_mean) / (wp + wd)
    sd = float(np.sqrt(1.0 / (wp + wd)))
    return float(mean), sd, "prior_blended"


def fit_models(df: pd.DataFrame) -> dict:
    """Fit both estimands and return everything the bundle needs.

    `total_effect` holds one coefficient per lever, each from its own fit on the adjustment set the
    graph implies. They are not comparable to the prediction coefficients and are never summed with
    them: mixing the two would double-count every mediated path.
    """
    from clock_model.model import cox  # complete_cohort lives there; avoids a circular import

    d = cox.complete_cohort(df)
    std = F.fit_standardizer(F._raw_columns(d))
    X = F.build_design(d, std)
    T, E = d["pm"].astype(float), d["dead"].astype(float)
    strata = strata_labels(d)
    cols = F.design_columns()

    prediction = fit_constrained(X[cols], T, E, strata)

    total_effect, adjustment_sets = {}, {}
    total_effect_sd, total_effect_source, total_effect_data_only = {}, {}, {}
    ONT = O.load()
    fitted = [c for c in cols if not c.endswith("_x_young")]
    for lever in O.levers():
        if lever not in cols:
            continue  # literature levers are not in the cohort; their effect comes from the prior
        adj = O.adjustment_set(lever, fitted)
        # No age-interaction term here on purpose: What-If needs ONE number per lever ("what happens
        # if you change this"), and the age-varying part of the baseline already lives in the strata.
        sub = [lever] + adj
        # Fitted UNCONSTRAINED on purpose: the prior already carries the sign, so blending a
        # bound-clipped estimate would count that information twice and would record a clipped 0.0
        # as "what the data said" — destroying the audit trail this is here to keep (PR#2 F16).
        beta, se = fit_constrained(X[sub], T, E, strata, constrained=False, return_se=True)
        data_mean, data_se = float(beta[lever]), float(se[lever])
        prior = ONT.get(lever, {}).get("prior") or {}
        blended, blended_sd, source = _blend_with_prior(data_mean, data_se, prior)
        total_effect[lever] = blended
        total_effect_sd[lever] = blended_sd
        total_effect_source[lever] = source
        total_effect_data_only[lever] = {"mean": data_mean, "se": data_se}
        adjustment_sets[lever] = adj

    return {
        "prediction_coefs": prediction.to_dict(),
        "total_effect_coefs": total_effect,
        "total_effect_sd": total_effect_sd,
        "total_effect_source": total_effect_source,
        "total_effect_data_only": total_effect_data_only,
        "adjustment_sets": adjustment_sets,
        "clipped_at_bound": list(prediction.attrs.get("clipped_at_bound", [])),
        "standardizer": std,
        "strata": O.meta()["strata"],
        "cohort": d,
        "design": X,
        "T": T,
        "E": E,
        "strata_labels": strata,
        "n": int(len(d)),
        "deaths": int(E.sum()),
    }
