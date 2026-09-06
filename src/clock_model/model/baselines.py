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


def remaining_le(qx: dict[int, float], start_age: int, rr: float) -> float:
    """Remaining life expectancy from start_age given a relative-risk multiplier on the baseline hazard."""
    if not qx:
        raise ValueError("empty life table")
    qx = _fill_gaps(qx)
    maxa = max(qx)
    S = 1.0
    le = 0.0
    for age in range(start_age, 111):
        q = qx.get(min(age, maxa), qx[maxa])   # ages beyond the table use the oldest known hazard
        qa = 1.0 - (1.0 - q) ** rr             # proportional hazards on the yearly interval
        le += S * (1.0 - qa / 2.0)
        S *= (1.0 - qa)
    return le
