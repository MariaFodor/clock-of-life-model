"""Fetch Eurostat data: national life tables (all countries) + centring prevalence.

Life tables come from `demo_mlifetable` (PROBDEATH by age/sex) — the per-country baseline. Prevalence
(current smoking, overweight+) comes from EHIS where available; on any failure we return None so the
pipeline falls back to the cohort mean (owner-approved). Responses are cached under data/cache/.
"""
from __future__ import annotations
import hashlib
import json
import os
import socket
import urllib.request

socket.setdefaulttimeout(30)
BASE = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"
CACHE = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "cache")


def _get(url: str) -> dict:
    os.makedirs(CACHE, exist_ok=True)
    key = os.path.join(CACHE, hashlib.sha256(url.encode()).hexdigest() + ".json")
    if os.path.exists(key):
        return json.load(open(key))
    with urllib.request.urlopen(url) as r:
        d = json.load(r)
    json.dump(d, open(key, "w"))
    return d


def _index_to_age(dim: dict) -> dict[int, str]:
    idx = dim["dimension"]["age"]["category"]["index"]
    return {v: k for k, v in idx.items()}   # position -> age key


def _age_key_to_int(k: str):
    if k == "Y_LT1":
        return 0
    if k.startswith("Y_GE"):
        return int(k[4:])
    if k.startswith("Y"):
        try:
            return int(k[1:])
        except ValueError:
            return None
    return None


def fetch_lifetable(geo: str, year: int) -> dict:
    """Return {'M': {age:qx}, 'F': {age:qx}} for a country from demo_mlifetable PROBDEATH."""
    out = {}
    for sex in ("M", "F"):
        url = (f"{BASE}/demo_mlifetable?format=JSON&lang=EN&indic_de=PROBDEATH"
               f"&geo={geo}&sex={sex}&time={year}")
        d = _get(url)
        pos2age = _index_to_age(d)
        qx = {}
        for pos_str, val in d["value"].items():
            age = _age_key_to_int(pos2age[int(pos_str)])
            if age is not None:
                qx[age] = val
        out[sex] = qx
    return out


_VENDORED_RO = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "RO_prevalence.json")


def latest_lifetable_year(geo: str = "RO") -> int | None:
    """Latest year for which demo_mlifetable has data (reads the dataset's time dimension)."""
    url = f"{BASE}/demo_mlifetable?format=JSON&lang=EN&indic_de=LIFEXP&geo={geo}&sex=T&age=Y1"
    try:
        d = _get(url)
        years = [int(y) for y in d["dimension"]["time"]["category"]["index"] if y.isdigit()]
        return max(years) if years else None
    except Exception:
        return None


def fetch_prevalence(geo: str) -> dict | None:
    """{current_smoking, overweight_plus} in [0,1], or None → cohort-mean fallback.

    RO uses the authoritative vendored EHIS figures; other countries pull the EHIS TOTAL cells
    (daily-smoker proxy + BMI>=25), with sanity ranges; anything implausible/missing → None.
    """
    if geo == "RO" and os.path.exists(_VENDORED_RO):
        v = json.load(open(_VENDORED_RO))
        return {"current_smoking": v["current_smoking_pct"][0] / 100.0,
                "overweight_plus": v["overweight_plus_pct"][0] / 100.0}
    smoke = _ehis_total("hlth_ehis_sk3e", geo, "smoking=TOTAL", lo=5, hi=60)      # daily-smoker proxy
    over = _ehis_total("hlth_ehis_bm1e", geo, "bmi=BMI_GE25", lo=30, hi=80)       # overweight+obese
    res = {}
    if smoke is not None:
        res["current_smoking"] = smoke / 100.0
    if over is not None:
        res["overweight_plus"] = over / 100.0
    return res or None


def _ehis_total(dataset: str, geo: str, indicator: str, lo: float, hi: float) -> float | None:
    """Pull the TOTAL-sex/TOTAL-age/TOTAL-education percentage, latest year, within a sanity range."""
    url = (f"{BASE}/{dataset}?format=JSON&lang=EN&geo={geo}"
           f"&unit=PC&{indicator}&isced11=TOTAL&sex=T&age=TOTAL")
    try:
        d = _get(url)
    except Exception:
        return None
    times = d["dimension"]["time"]["category"]["index"]      # {'2014':0,'2019':1}
    latest_pos = max(times.values())
    val = d.get("value", {}).get(str(latest_pos))
    if val is None:                                          # fall back to any available value
        vals = list(d.get("value", {}).values())
        val = vals[-1] if vals else None
    if val is None or not (lo <= float(val) <= hi):
        return None
    return float(val)
