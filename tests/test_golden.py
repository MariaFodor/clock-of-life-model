"""Golden-number regression: the lifelines pipeline must reproduce the witnessed experiment results.
Run: PYTHONPATH=src .venv/bin/python tests/test_golden.py  (no network needed — uses vendored cohort)
"""
import json, os, sys
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from clock_model.model import cox, centring, baselines
from clock_model.evaluate import gates
from clock_model.fetch import eurostat

DATA = os.path.join(os.path.dirname(__file__), "..", "data")


def main():
    df = pd.DataFrame(json.load(open(os.path.join(DATA, "wide.json"))))
    fit = cox.fit_models(df)
    g = gates.evaluate(fit)
    checks = []
    checks.append(("n == 18839", fit["n"] == 18839))
    checks.append(("deaths == 2119", fit["deaths"] == 2119))
    # Bumped from [0.785, 0.80] after adding the cohort BMI + current-smoking-dose + systolic-BP features
    # (clock_dev EXP-14: +0.027 out-of-sample C-index). Alcohol stays a literature monotonic lever.
    checks.append(("C-index in [0.812, 0.828]", 0.812 <= g["c_index"] <= 0.828))
    checks.append(("calibration MAE <= 0.02", g["calibration_mae"] <= 0.02))

    # centring: average Romanian ≈ national life expectancy at 40
    prev = json.load(open(os.path.join(DATA, "RO_prevalence.json")))
    ro = {"current_smoking": prev["current_smoking_pct"][0] / 100, "overweight_plus": prev["overweight_plus_pct"][0] / 100}
    rates = centring.cohort_rates(fit["cohort"])
    lt = json.load(open(os.path.join(DATA, "RO_2024.json")))
    qxM = baselines.qx_from_eurostat_lifetable(lt["PROBDEATH_M"])
    le_avg = baselines.remaining_le(qxM, 40, 1.0)     # RR=1 by construction for the average
    checks.append(("avg-RO male LE@40 ≈ national 34.5", abs(le_avg - lt["LIFEXP_M"]["Y40"]) < 0.6))

    ok = all(p for _, p in checks)
    for name, p in checks:
        print(f"  [{'PASS' if p else 'FAIL'}] {name}")
    print(f"\nGOLDEN: {'PASS' if ok else 'FAIL'}  (C-index={g['c_index']}, calibration={g['calibration_mae']})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
