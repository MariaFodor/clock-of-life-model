"""Download NHANES component .xpt files from CDC and read them as DataFrames. Cached under data/cache."""
from __future__ import annotations
import io
import os
import urllib.request

import pandas as pd

from clock_model.config import cycles

CACHE = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "cache", "nhanes")
_UA = {"User-Agent": "Mozilla/5.0 (clock-of-life-model)"}


def _download(url: str, dest: str) -> str:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if not os.path.exists(dest):
        req = urllib.request.Request(url, headers=_UA)
        with urllib.request.urlopen(req) as r:
            data = r.read()
        if data[:20].find(b"HEADER RECORD") == -1:
            raise ValueError(f"not an XPORT file: {url}")
        open(dest, "wb").write(data)
    return dest


def cycle_available(cycle: str, suffix: str) -> bool:
    """Does NHANES publish this cycle (probe its DEMO file)? Used by --check-updates."""
    year = cycle.split("-")[0]
    url = f"{cycles.CDC_BASE}/{year}/DataFiles/DEMO_{suffix}.xpt"
    try:
        req = urllib.request.Request(url, headers=_UA, method="HEAD")
        with urllib.request.urlopen(req) as r:
            return r.status == 200
    except Exception:
        return False


def component(cycle: str, comp: str) -> pd.DataFrame | None:
    """Download + read one NHANES component; None if the file doesn't exist for this cycle."""
    url = cycles.component_url(cycle, comp)
    dest = os.path.join(CACHE, f"{comp}_{cycles.TRAINING_CYCLES[cycle]}.xpt")
    try:
        path = _download(url, dest)
    except Exception:
        return None
    return pd.read_sas(path, format="xport")
