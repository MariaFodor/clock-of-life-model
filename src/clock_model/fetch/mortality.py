"""Download + parse the NCHS public-use Linked Mortality File (fixed-width) for a cycle.

Layout (2019 public-use, 0-indexed end-exclusive columns), reclen 48:
  SEQN 0-6 | ELIGSTAT 14-15 | MORTSTAT 15-16 | UCOD_LEADING 16-19 | DIABETES 19-20 | HYPERTEN 20-21 |
  PERMTH_INT 42-45 | PERMTH_EXM 45-48
"""
from __future__ import annotations
import os
import urllib.request

import pandas as pd

from clock_model.config import cycles

CACHE = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "cache", "mort")
_UA = {"User-Agent": "Mozilla/5.0 (clock-of-life-model)"}
_COLSPECS = [(0, 6), (14, 15), (15, 16), (16, 19), (19, 20), (20, 21), (42, 45), (45, 48)]
_NAMES = ["SEQN", "ELIGSTAT", "MORTSTAT", "UCOD_LEADING", "DIABETES", "HYPERTEN", "PERMTH_INT", "PERMTH_EXM"]


def mortality(cycle: str) -> pd.DataFrame:
    url = cycles.mortality_url(cycle)
    dest = os.path.join(CACHE, os.path.basename(url))
    os.makedirs(CACHE, exist_ok=True)
    if not os.path.exists(dest):
        req = urllib.request.Request(url, headers=_UA)
        with urllib.request.urlopen(req) as r:
            open(dest, "wb").write(r.read())
    df = pd.read_fwf(dest, colspecs=_COLSPECS, names=_NAMES, na_values=["."], dtype=str)
    for c in ["SEQN", "ELIGSTAT", "MORTSTAT", "PERMTH_INT", "PERMTH_EXM"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[df["ELIGSTAT"] == 1]                     # eligible for linkage only
    df["dead"] = (df["MORTSTAT"] == 1).astype(int)
    df["pm"] = df["PERMTH_EXM"].fillna(df["PERMTH_INT"])
    return df[["SEQN", "dead", "pm"]].dropna(subset=["pm"])
