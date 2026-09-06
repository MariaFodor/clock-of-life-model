"""Harmonization oracle: the raw-NHANES harmonizer must reproduce the vendored wide.json.
Run: PYTHONPATH=src .venv/bin/python tests/test_harmonize.py  (downloads NHANES on first run; then cached)

Known approximate columns: pa_min (MET formula differs from the original derivation) and pfq_diff
(multi-item combination) — excluded from the exact-match gate, documented in the README.
"""
import json, os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from clock_model.ingest import harmonize

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
EXACT_COLS = ["age", "sex", "educ_hi", "income", "smoke", "alc_day", "sleep", "bmi", "waist", "sbp",
              "hbp_told", "chol_told", "diab", "srh", "mi", "stroke", "chf", "cancer", "bronchitis", "depression"]


def main():
    cycle = "2013-2014"
    h = harmonize._cohort_one_cycle(cycle)
    w = pd.DataFrame(json.load(open(os.path.join(DATA, "wide.json"))))
    w = w[w.cycle == cycle.replace("-", "_")].copy(); w["seqn"] = w["seqn"].astype(float)
    j = h.merge(w, on="seqn", suffixes=("_h", "_w"))

    checks = [(f"row count {len(h)} == {len(w)}", len(h) == len(w))]
    for c in EXACT_COLS:
        a, b = j[f"{c}_h"], j[f"{c}_w"]
        both = a.notna() & b.notna()
        if both.sum() == 0:
            checks.append((f"{c}: has data", False)); continue
        if c in ("income", "bmi", "waist", "sbp", "alc_day", "depression"):
            agree = (np.abs(a[both] - b[both]) < 0.5).mean()
        else:
            agree = (a[both] == b[both]).mean()
        checks.append((f"{c}: {agree:.2f} exact", agree >= 0.98))

    ok = all(p for _, p in checks)
    for name, p in checks:
        print(f"  [{'PASS' if p else 'FAIL'}] {name}")
    print(f"\nHARMONIZE: {'PASS' if ok else 'FAIL'}  ({len(EXACT_COLS)} exact-match columns; pa_min/pfq_diff approximate)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
