# clock-of-life-model

The Clock of Life **model module** — an offline, retrainable Python pipeline that fits the survival
model and emits a **versioned artifact bundle** for the Rust scoring service. Intended as a standalone
repo (developed inside `clock_dev/`, which is a dev context, not committed).

## What it does
`fetch → ingest → features → fit → baselines → evaluate → export`, driven by one command:

```bash
python -m clock_model.train --countries all --version 3.0.0     # full run → artifacts/model-v3.0.0/
python -m clock_model.train --countries RO,DE,FR                # a subset
python -m clock_model.train --check-updates                     # report newer upstream data
python -m clock_model.train --from-raw                          # rebuild the cohort from raw NHANES
```

- **Engine:** stratified Cox (scipy L-BFGS-B on the partial likelihood) (interpretable). Two models: a **prediction** model and a separate
  **total-effect attribution** model (levers only) for Why?/What-If. Age/sex go to the life-table
  baseline; lever effects are **age-interacted (age and sex are strata since v3.0.0, so coefficients compare like with like)** (young/old).
- **Scrapers (real):** NHANES `.xpt` (CDC) + NCHS Linked Mortality `.dat` → harmonized cohort;
  Eurostat life tables + EHIS prevalence for **all 30 EU/EEA countries**. All cached under `data/cache/`.
- **Multi-country baselines:** the Cox relative-risk model is country-agnostic; each country gets its own
  life-table baseline + centring reference (smoking/weight from EHIS, cohort-mean fallback). An average
  national resident reads RR ≈ 1.0 = national life expectancy.
- **Output:** `artifacts/model-v<ver>/` with `manifest.json`, `coefficients.json` (prediction +
  attribution + standardizer + literature), `evidence.json`, `model-card.md`, and `baselines/<ISO>.json`.

## Training window
NHANES **2007–2014** (4 cycles, n≈18.8k complete-case, ~2.1k deaths), linked mortality **through 2019**.
`config/cycles.py` holds the window; `config/countries.py` the country list.

## Dataset discovery
`--check-updates` reports newer NCHS mortality linkage (the binding gate for adding training cycles),
newer NHANES cycles (and why each is/isn't trainable under 2019 follow-up), and newer Eurostat years
(baselines refresh freely without changing the fitted coefficients).

## Validation (committed probes)
- `tests/test_golden.py` — the lifelines pipeline reproduces the witnessed experiment numbers:
  **C-index 0.76, within-stratum 0.685, calibration MAE 0.013**, avg-Romanian = national life
  expectancy. (No network, but it needs a built bundle — the test says how.) The figures here read
  0.805 / 0.019 from before the age-and-sex-stratified refit, which is the same defect the test's own
  precondition had: documentation left a revision behind the thing it documents.
- `tests/test_ontology_gates.py` — a malformed citation declaration is REPORTED, naming the field,
  rather than raising mid-gate. (No network, no bundle.)
- `tests/test_harmonize.py` — the raw-NHANES harmonizer reproduces the vendored `data/wide.json`:
  **20 columns exact**. Known approximate: `pa_min` (MET formula differs from the original derivation)
  and `pfq_diff` (multi-item combination) — they wash out under the log/threshold encodings; refining the
  MET derivation to bit-exact is a follow-up.

## Setup
```bash
python3 -m venv .venv --without-pip && .venv/bin/python get-pip.py   # (Python 3.14: pip bootstrap)
.venv/bin/pip install numpy pandas lifelines
```

## Status
v1 = **Cox only**. The ML (gradient-boosted survival) implementation + ONNX export is a later addition
(ADR-007). The shipping bundle defaults to the vendored (exact, witnessed) cohort; `--from-raw` uses the
scraper path.
