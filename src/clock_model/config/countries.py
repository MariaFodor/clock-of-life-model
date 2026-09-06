"""Eurostat geographies for the per-country baselines.

The Cox relative-risk model is country-agnostic; only the life-table baseline and the centring prevalence
are per-country. Life tables come from Eurostat `demo_mlifetable`; prevalence (smoking, overweight) from
EHIS where available, cohort-mean fallback otherwise.
"""

# EU-27 + EFTA/EEA, Eurostat geo codes. RO is the validated reference.
EUROSTAT_COUNTRIES = [
    "RO",  # reference (validated)
    "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR", "DE", "EL", "HU",
    "IE", "IT", "LV", "LT", "LU", "MT", "NL", "PL", "PT", "SK", "SI", "ES", "SE",
    "IS", "NO", "CH",  # EFTA/EEA
]

LIFETABLE_YEAR = 2024   # latest good Eurostat year (avoid 2020-21 pandemic distortion)
PREVALENCE_SOURCE = "EHIS 2019"
