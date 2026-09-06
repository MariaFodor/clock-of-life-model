"""NHANES training window + component/mortality file naming.

Training window v1 = 2007-2014 (physical activity harmonizable from 2007; adequate mortality follow-up
by the 2019 linkage). One-line change to extend once a newer NCHS linkage lands.
"""

# cycle -> CDC file suffix
TRAINING_CYCLES = {
    "2007-2008": "E",
    "2009-2010": "F",
    "2011-2012": "G",
    "2013-2014": "H",
}

# NHANES components we read (base name -> the harmonized columns they feed)
COMPONENTS = ["DEMO", "SMQ", "ALQ", "PAQ", "SLQ", "BMX", "BPX", "BPQ", "DIQ", "MCQ", "HUQ", "PFQ", "DPQ"]

CDC_BASE = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public"   # /{first_year}/DataFiles/{COMP}_{suffix}.xpt
MORT_BASE = "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/datalinkage/linked_mortality"
MORT_FILE = "NHANES_{a}_{b}_MORT_2019_PUBLIC.dat"       # a_b = cycle years, e.g. 2007_2008
MORT_FOLLOWUP_THROUGH = "2019-12-31"


def component_url(cycle: str, comp: str) -> str:
    suffix = TRAINING_CYCLES[cycle]
    year = cycle.split("-")[0]
    return f"{CDC_BASE}/{year}/DataFiles/{comp}_{suffix}.xpt"


def mortality_url(cycle: str) -> str:
    a, b = cycle.split("-")
    return f"{MORT_BASE}/{MORT_FILE.format(a=a, b=b)}"
