"""Harmonize raw NHANES components + mortality into the wide cohort (reproduces data/wide.json).

Handles cross-cycle variable renames via `_first` (pick the first column that exists). Validated against
the vendored wide.json oracle by tests/test_harmonize.py. Physical-activity MET-minutes use the standard
GPAQ weights (vigorous 8, moderate 4, walk 4); other columns map directly.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from clock_model.config.cycles import TRAINING_CYCLES
from clock_model.fetch import nhanes, mortality


def _first(df, *cands):
    for c in cands:
        if df is not None and c in df.columns:
            return df[c]
    return pd.Series(np.nan, index=(df.index if df is not None else None))


def _yn(series):  # NHANES 1=yes 2=no → 1/0
    return series.map({1: 1, 2: 0})


def _met_minutes(paq):
    """GPAQ-style weekly MET-minutes from the PAQ block (2007+): work + recreation + transport."""
    if paq is None:
        return pd.Series(np.nan)
    def block(days, mins, met):
        d = _first(paq, days); m = _first(paq, mins)
        return (pd.to_numeric(d, errors="coerce").where(lambda x: x <= 7) *
                pd.to_numeric(m, errors="coerce").where(lambda x: x <= 1000) * met)
    vig_work = block("PAQ610", "PAD615", 8.0)
    mod_work = block("PAQ625", "PAD630", 4.0)
    walk     = block("PAQ640", "PAD645", 4.0)
    vig_rec  = block("PAQ655", "PAD660", 8.0)
    mod_rec  = block("PAQ670", "PAD675", 4.0)
    total = pd.concat([vig_work, mod_work, walk, vig_rec, mod_rec], axis=1).sum(axis=1, min_count=1)
    return total.fillna(0.0)


def _cohort_one_cycle(cycle: str) -> pd.DataFrame:
    comp = {c: nhanes.component(cycle, c) for c in
            ["DEMO", "SMQ", "ALQ", "PAQ", "SLQ", "BMX", "BPX", "BPQ", "DIQ", "MCQ", "HUQ", "PFQ", "DPQ"]}
    demo = comp["DEMO"]
    if demo is None:                      # DEMO carries SEQN/age/sex — the cohort cannot be built without it
        raise RuntimeError(f"NHANES DEMO component unavailable for cycle {cycle} (download failed)")
    out = pd.DataFrame({"seqn": demo["SEQN"].astype(float)})
    out["cycle"] = cycle.replace("-", "_")
    out["age"] = pd.to_numeric(demo["RIDAGEYR"], errors="coerce")
    out["sex"] = (pd.to_numeric(demo["RIAGENDR"], errors="coerce") == 1).astype(float)
    out["educ_hi"] = (pd.to_numeric(_first(demo, "DMDEDUC2"), errors="coerce") >= 4).astype(float)
    out["income"] = pd.to_numeric(_first(demo, "INDFMPIR"), errors="coerce")

    def by_seqn(df, col):  # align a component column to demo SEQN
        if df is None or col not in df.columns:
            return pd.Series(np.nan, index=out.index)
        s = df.set_index("SEQN")[col]
        return out["seqn"].map(s)

    smq = comp["SMQ"]
    smoked100 = by_seqn(smq, "SMQ020"); nowsmk = by_seqn(smq, "SMQ040")
    smoke = pd.Series(0.0, index=out.index)                 # never
    smoke = smoke.where(~((smoked100 == 1) & (nowsmk == 3)), 1.0)   # former
    smoke = smoke.where(~(nowsmk.isin([1, 2])), 2.0)                # current
    out["smoke"] = smoke
    out["cigs_day"] = by_seqn(smq, "SMD650")

    out["alc_day"] = by_seqn(comp["ALQ"], "ALQ130")
    met = _met_minutes(comp["PAQ"]);  out["pa_min"] = out["seqn"].map(
        pd.Series(met.values, index=comp["PAQ"]["SEQN"].astype(float))) if comp["PAQ"] is not None else np.nan
    slq = comp["SLQ"]
    out["sleep"] = by_seqn(slq, "SLD010H") if (slq is not None and "SLD010H" in slq.columns) else by_seqn(slq, "SLD012")
    out["bmi"] = by_seqn(comp["BMX"], "BMXBMI")
    out["waist"] = by_seqn(comp["BMX"], "BMXWAIST")

    bpx = comp["BPX"]
    if bpx is not None:
        sy = [c for c in ["BPXSY1", "BPXSY2", "BPXSY3", "BPXSY4"] if c in bpx.columns]
        bpx_sbp = bpx.set_index("SEQN")[sy].mean(axis=1) if sy else None
        out["sbp"] = out["seqn"].map(bpx_sbp) if bpx_sbp is not None else np.nan
    else:
        out["sbp"] = np.nan
    out["hbp_told"] = _yn(by_seqn(comp["BPQ"], "BPQ020"))
    out["chol_told"] = _yn(by_seqn(comp["BPQ"], "BPQ080"))
    out["diab"] = _yn(by_seqn(comp["DIQ"], "DIQ010"))

    mcq = comp["MCQ"]
    out["mi"] = _yn(by_seqn(mcq, "MCQ160E"))
    out["stroke"] = _yn(by_seqn(mcq, "MCQ160F"))
    out["chf"] = _yn(by_seqn(mcq, "MCQ160B"))
    out["cancer"] = _yn(by_seqn(mcq, "MCQ220"))
    out["copd"] = _yn(pd.Series(np.where(by_seqn(mcq, "MCQ160O") == 1, 1, np.nan), index=out.index)) if (mcq is not None and "MCQ160O" in mcq.columns) else _yn(by_seqn(mcq, "MCQ160G"))
    out["bronchitis"] = _yn(by_seqn(mcq, "MCQ160K"))

    out["srh"] = pd.to_numeric(by_seqn(comp["HUQ"], "HUQ010"), errors="coerce").where(lambda x: x <= 5)
    pfq = comp["PFQ"]
    out["pfq_diff"] = (pd.to_numeric(by_seqn(pfq, "PFQ061B"), errors="coerce") > 1).astype(float) if (pfq is not None and "PFQ061B" in pfq.columns) else np.nan

    dpq = comp["DPQ"]
    if dpq is not None:
        items = [c for c in dpq.columns if c.startswith("DPQ0")]
        dep = dpq.set_index("SEQN")[items].where(lambda d: d <= 3).sum(axis=1, min_count=len(items))
        out["depression"] = out["seqn"].map(dep)
    else:
        out["depression"] = np.nan

    mort = mortality.mortality(cycle).rename(columns={"SEQN": "seqn"})
    out = out.merge(mort, on="seqn", how="inner")
    return out


def build_cohort() -> pd.DataFrame:
    frames = [_cohort_one_cycle(c) for c in TRAINING_CYCLES]
    return pd.concat(frames, ignore_index=True)
