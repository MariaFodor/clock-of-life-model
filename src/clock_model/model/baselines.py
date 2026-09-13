"""Baseline hazards and the remaining-years machinery (ported from experiments/exp01/exp13).

Two pieces:
- Breslow baseline cumulative hazard H0(t) — used for calibration and short-horizon survival.
- National life-table qx by age & sex — the app's remaining-years engine: apply the person's relative
  risk to the national baseline and integrate the survival curve.
"""
from __future__ import annotations
import math
import numpy as np


def breslow_H0(T, E, lp, at_time):
    """Breslow baseline cumulative hazard at `at_time`."""
    order = np.argsort(T)
    T = np.asarray(T)[order]; E = np.asarray(E)[order]; w = np.exp(np.asarray(lp)[order])
    # risk-set sum of exp(lp) for times >= t, computed from the end
    rev = np.cumsum(w[::-1])[::-1]
    H0 = 0.0
    for i in range(len(T)):
        if E[i] and T[i] <= at_time and rev[i] > 0:
            H0 += 1.0 / rev[i]
    return H0


def qx_from_eurostat_lifetable(probdeath: dict[str, float]) -> dict[int, float]:
    """Eurostat/RO life table gives probability of death by age. Keys like 'Y_LT1','Y1',...,'Y40'."""
    out = {}
    for k, v in probdeath.items():
        if k == "Y_LT1":
            out[0] = v
        elif k.startswith("Y"):
            try:
                out[int(k[1:])] = v
            except ValueError:
                pass
    return out


def _fill_gaps(qx: dict[int, float]) -> dict[int, float]:
    """Linearly interpolate any missing ages between the youngest and oldest known age.

    Eurostat can return fewer values than age categories (a suppressed cell), leaving holes in qx.
    Interpolating is correct; falling back to the oldest-age hazard would spike mortality at the hole.
    """
    ages = sorted(qx)
    lo, hi = ages[0], ages[-1]
    filled = {}
    for a in range(lo, hi + 1):
        if a in qx:
            filled[a] = qx[a]
        else:
            prev = max(x for x in ages if x < a)
            nxt = min(x for x in ages if x > a)
            t = (a - prev) / (nxt - prev)
            filled[a] = qx[prev] + t * (qx[nxt] - qx[prev])
    return filled


def remaining_le(qx: dict[int, float], start_age: int, rr: float,
                 ax_last: float | None = None) -> float:
    """Remaining life expectancy from start_age given a relative-risk multiplier on the baseline hazard.

    `ax_last` is the average years still lived by someone who has reached the table's final, OPEN age
    interval — WPP publishes it per country and sex, and for a terminal open interval it IS the
    remaining life expectancy there, because everyone in it dies in it.

    WITHOUT IT THIS FUNCTION IS WRONG AT THE TOP OF THE TABLE, and not subtly. WPP closes every table
    at 100 with qx = 1.0, so the loop below charged the standard half-year for that interval and then
    drove survivorship to zero: every age from 100 to 110 came back as EXACTLY 0.5 years, for every
    country, both sexes, every risk profile — while the questionnaire accepts ages to 110. The real
    figure for a Romanian man of 100 is 1.90; across the world `ax_last` runs 0.79 to 4.61, so half a
    year was wrong everywhere and wrong by a factor of nine at the far end.

    It was also quietly costing accuracy at age 0, where the error is small but real: the median gap
    between this integrator and WPP's own published e(0) falls from 0.0112 to 0.0040 years when
    `ax_last` is used, and the worst country from 0.43 to 0.018. The half-year default was the cause
    of that residual, not the floor of what this method can do.

    `ax_last / rr` rather than `ax_last` under a relative risk: an open interval with constant force of
    mortality mu has e = 1/mu, and a proportional hazard multiplies mu by rr, so the expectation scales
    by 1/rr. That exponential-tail assumption is the same one the interval's own `ax` is built on.

    Falls back to the historic half-year when no `ax_last` is supplied, so a pre-v4.2.0 bundle keeps
    scoring rather than failing — but the release gates refuse to SHIP one.
    """
    if not qx:
        raise ValueError("empty life table")
    qx = _fill_gaps(qx)
    maxa = max(qx)
    S = 1.0
    le = 0.0
    for age in range(start_age, 111):
        q = qx.get(min(age, maxa), qx[maxa])   # ages beyond the table use the oldest known hazard
        qa = 1.0 - (1.0 - q) ** rr             # proportional hazards on the yearly interval
        if qa >= 1.0:
            # The open terminal interval: nobody leaves it alive, so what remains is the time lived
            # inside it. Returning here rather than breaking makes it explicit that no later age can
            # contribute — which is true today only because S becomes 0, an accident this relies on.
            return le + S * ((ax_last / rr) if ax_last else 0.5)
        le += S * (1.0 - qa / 2.0)
        S *= (1.0 - qa)
    return le
