"""Assemble the immutable, versioned artifact bundle the Rust service loads."""
from __future__ import annotations
import hashlib
import json
import os

from clock_model.config import features as F
from clock_model.config.literature import LITERATURE
from clock_model.config.cycles import MORT_FOLLOWUP_THROUGH
from clock_model.config.countries import LIFETABLE_YEAR


def _write(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(obj, open(path, "w"), indent=2)


def _checksum(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


def _merged_standardizer(fitted: dict) -> dict:
    lit = {k: {"mean": v["standardizer"]["mean"], "sd": v["standardizer"]["sd"]}
           for k, v in LITERATURE.items() if "standardizer" in v}
    overlap = set(fitted) & set(lit)
    if overlap:
        raise SystemExit(f"[bundle] standardizer key(s) {sorted(overlap)} are both fitted and "
                         "literature-declared — resolve which one ships before exporting")
    return {**fitted, **lit}


def assemble(out_dir: str, version: str, fit: dict, gates: dict, countries: dict) -> str:
    """countries: {ISO: {qx:{M,F}, reference_lp:{young,old}, national_le_40:{M,F}, prevalence}}."""
    root = os.path.join(out_dir, f"model-v{version}")
    # Bundles are immutable (ADR-006): a change means a NEW version, never an in-place edit.
    if os.path.exists(root):
        raise SystemExit(f"[bundle] {root} already exists — bump the version instead of overwriting")

    coefficients = {
        "prediction": fit["prediction_coefs"],
        "attribution": fit["attribution_coefs"],
        # Fitted standardizers + the literature features' published/declared ones, in one map, so
        # the service z-scores every continuous input the same way (and can validate coverage).
        # A key can only come from one side: if a future fit gains a column the literature also
        # declares, that's a modelling decision, not a silent override.
        "standardizer": _merged_standardizer(fit["standardizer"]),
        "literature": LITERATURE,
        "young_cutoff": F.YOUNG_CUTOFF,
    }
    _write(os.path.join(root, "coefficients.json"), coefficients)

    evidence = F.evidence_table()
    for k, v in LITERATURE.items():
        # ENV is CONTEXT for the personal clock (you don't "recommend" moving) and a lever only
        # inside "Where Should I Live?" (THE_QUESTIONNAIRE.md S10); the rest are true levers.
        role = "context" if k == "env" else "lever"
        evidence[k] = {"role": role, "grade": v["grade"], "citation": v["citation"]}
    _write(os.path.join(root, "evidence.json"), evidence)

    for iso, b in countries.items():
        _write(os.path.join(root, "baselines", f"{iso}.json"), b)

    card = f"""# Model card — The Clock of Life v{version}

**Algorithm:** Cox proportional hazards (interpretable), lifestyle + pathology predictors; age & sex to
the national life-table baseline. **Attribution/What-If** uses a separate total-effect model.

**Training:** NHANES 2007-2014 linked to NCHS mortality (through {MORT_FOLLOWUP_THROUGH}); n={fit['n']},
deaths={fit['deaths']}. **Discrimination** C-index {gates['c_index']}; **calibration** MAE
{gates['calibration_mae']}.

**Baselines:** {len(countries)} countries (Eurostat life tables); relative risk centred on each country's
average person (smoking & weight from EHIS, cohort-mean fallback otherwise).

**Literature features** (diet, alcohol, sedentary, stress, environment) are appended from the evidence
base at stated confidence, not fitted on the cohort.

**Intended use:** wellness/education only — a statistical estimate, never a prediction or diagnosis
(ADR-001). Recommendations target modifiable/manageable factors only.
"""
    with open(os.path.join(root, "model-card.md"), "w") as f:
        f.write(card)

    # checksum every file in the bundle (except the manifest itself), then write the manifest last
    checksums = {}
    for dirpath, _, names in os.walk(root):
        for name in names:
            if name == "manifest.json":
                continue
            p = os.path.join(dirpath, name)
            checksums[os.path.relpath(p, root)] = _checksum(p)
    manifest = {
        "version": version,
        "algorithm": "cox_ph",
        "reference_population": "per-country (Eurostat)",
        "training_cohort": "NHANES 2007-2014 + NCHS Linked Mortality",
        "mortality_followup_through": MORT_FOLLOWUP_THROUGH,
        # The DATA vintage, not the build date. Kept a bare ISO date — the service parses this
        # into a DATE column (seed.rs %Y-%m-%d); the prose context goes in data_vintage_note.
        "data_as_of": MORT_FOLLOWUP_THROUGH,
        "data_vintage_note": f"mortality linkage through {MORT_FOLLOWUP_THROUGH}; "
                             f"Eurostat life tables {LIFETABLE_YEAR}",
        "n": fit["n"], "deaths": fit["deaths"],
        "gates": gates,
        "countries": sorted(countries.keys()),
        "checksums": checksums,
    }
    _write(os.path.join(root, "manifest.json"), manifest)
    return root
