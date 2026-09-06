"""Assemble the immutable, versioned artifact bundle the Rust service loads."""
from __future__ import annotations
import hashlib
import json
import os
from datetime import date

from clock_model.config import features as F
from clock_model.config.literature import LITERATURE
from clock_model.config.cycles import MORT_FOLLOWUP_THROUGH


def _write(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(obj, open(path, "w"), indent=2)


def _checksum(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


def assemble(out_dir: str, version: str, fit: dict, gates: dict, countries: dict) -> str:
    """countries: {ISO: {qx:{M,F}, reference_lp:{young,old}, national_le_40:{M,F}, prevalence}}."""
    root = os.path.join(out_dir, f"model-v{version}")

    coefficients = {
        "prediction": fit["prediction_coefs"],
        "attribution": fit["attribution_coefs"],
        "standardizer": fit["standardizer"],
        "literature": LITERATURE,
        "young_cutoff": F.YOUNG_CUTOFF,
    }
    _write(os.path.join(root, "coefficients.json"), coefficients)

    evidence = F.evidence_table()
    for k, v in LITERATURE.items():
        evidence[k] = {"role": "lever", "grade": v["grade"], "citation": v["citation"]}
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
        "data_as_of": date.today().isoformat(),
        "n": fit["n"], "deaths": fit["deaths"],
        "gates": gates,
        "countries": sorted(countries.keys()),
        "checksums": checksums,
    }
    _write(os.path.join(root, "manifest.json"), manifest)
    return root
