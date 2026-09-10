"""EXP-15 (W-A1) — UN WPP 2024 against Eurostat 2024 as the baseline life-table source.

The map and the clock cannot come from the same place today. The estimate runs on Eurostat
`demo_mlifetable` 2024, which covers 30 EU/EEA countries and can never draw a world map; every global
source we might draw one from disagrees with it (WHO's 2021 estimates put Romanian men at 69.2 where the
shipped clock says 72.8 — the same product contradicting itself by 3.6 years on the same screen).

UN World Population Prospects 2024 is the only candidate that covers both: 237 countries or areas at the
same granularity the model needs — single year of age, by sex. This measures what switching would cost,
before anything is switched.

Three questions, all answered in numbers:
  1. how far does each country's estimate move?
  2. do the two agencies rank countries the same way, or is one of them telling a different story?
  3. what happens in the tail — where Eurostat's open interval at 95 currently gives every 95-year-old
     in every country exactly half a year to live?

Sources. UN World Population Prospects 2024 — "Copyright © 2024 by United Nations, made available
under a Creative Commons license CC BY 3.0 IGO: http://creativecommons.org/licenses/by/3.0/igo/",
cited as: United Nations, Department of Economic and Social Affairs, Population Division (2024). World
Population Prospects 2024, Online Edition. Eurostat `demo_mlifetable`, reused under the Commission's
reuse policy. The WHO comparison quoted above is GHE 2021, indicator WHOSIS_000001
(https://ghoapi.azureedge.net/api/WHOSIS_000001).

Run: PYTHONPATH=src .venv/bin/python experiments/exp15_wpp_vs_eurostat.py
"""
from __future__ import annotations

import csv
import gzip
import io
import json
import os
import math
import statistics
import sys
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from scipy.stats import spearmanr  # noqa: E402

from clock_model.config.countries import EUROSTAT_COUNTRIES, LIFETABLE_YEAR  # noqa: E402
from clock_model.fetch import eurostat  # noqa: E402
from clock_model.model.baselines import remaining_le  # noqa: E402

# Per-operation read timeout, passed to urlopen rather than set globally: socket.setdefaulttimeout
# would also relax the 30 s that fetch/eurostat.py sets for itself.
TIMEOUT = 180
# A 200 MB gzip of repeated bytes with no newline in it expands unboundedly inside readline(), long
# before csv.reader gets a chance to object. The real files are ~4 GB decompressed.
MAX_DECOMPRESSED = 8 * 1024 ** 3

WPP_YEAR = 2023          # the last ESTIMATE year of the 2024 revision; everything after is a projection
WPP_BASE = "https://population.un.org/wpp/assets/Excel%20Files/1_Indicator%20(Standard)/CSV_FILES"
WPP_FILES = {
    "M": "WPP2024_Life_Table_Complete_Medium_Male_1950-2023.csv.gz",
    "F": "WPP2024_Life_Table_Complete_Medium_Female_1950-2023.csv.gz",
}
# Each file holds one sex, so the file decides the sex — which is trusting someone else's filenames
# for the one dimension whose transposition would produce a fully sex-swapped comparison that passes
# every other check. The Variant filter below refuses to trust the filename; so does this.
EXPECTED_SEX = {"M": "Male", "F": "Female"}
# The whole point of the switch is that WPP covers the world. A revision that returned 35 well-formed
# countries would otherwise exit 0 under a green verdict reading "coverage goes 30 -> 35".
MIN_COUNTRIES = 200
CACHE = os.path.join(os.path.dirname(__file__), "..", "data", "cache")

# Eurostat calls Greece EL. ISO 3166-1 — and therefore WPP — calls it GR. Nothing else in the 30 differs.
ALIASES = {"EL": "GR"}

WPP_AGES = range(0, 101)


class _Budgeted(io.RawIOBase):
    """Refuse to keep decompressing past a sane ceiling.

    `readline()` assembles a whole line before csv.reader can object to anything, so a stream with no
    newline in it expands without limit. The real files are ~4 GB decompressed; anything far past that
    is not the file we asked for.
    """

    def __init__(self, raw, limit: int = MAX_DECOMPRESSED):
        self._raw, self._limit, self._seen = raw, limit, 0

    def readable(self) -> bool:
        return True

    def readinto(self, buf) -> int:
        chunk = self._raw.read(len(buf))
        if not chunk:
            return 0
        self._seen += len(chunk)
        if self._seen > self._limit:
            raise ValueError(f"decompressed past {self._limit} bytes — refusing to continue")
        buf[:len(chunk)] = chunk
        return len(chunk)


def _validate(tables: dict[str, dict[str, dict[int, float]]]) -> None:
    """Refuse a life table that is merely plausible.

    A truncated download does not fail: a table cut at 90 still integrates, because `remaining_le`
    carries the oldest known hazard forward, and across these 30 countries it comes back a median
    0.37 years high (up to 1.47) — several times the difference this experiment exists to measure. So the shape is checked rather than assumed, on the
    way in AND on the way back out of the cache.

    Deliberately not shared with `fetch/wpp.py`: this script is the evidence for adopting that adapter,
    and evidence that trusts the thing it is judging is not evidence.
    """
    if len(tables) < MIN_COUNTRIES:
        raise ValueError(f"only {len(tables)} countries survived the filter; expected at least "
                         f"{MIN_COUNTRIES} — the coverage this comparison exists to justify")
    for iso, sexes in sorted(tables.items()):
        if set(sexes) != set(WPP_FILES):
            raise ValueError(f"{iso}: has {sorted(sexes)}, expected {sorted(WPP_FILES)}")
        for sex, qx in sexes.items():
            if set(qx) != set(WPP_AGES):
                missing = sorted(set(WPP_AGES) - set(qx))[:5]
                raise ValueError(f"{iso}/{sex}: not a complete {WPP_AGES.start}-{WPP_AGES.stop - 1} "
                                 f"table (missing {missing})")
            bad = [(a, q) for a, q in qx.items() if not (math.isfinite(q) and 0.0 <= q <= 1.0)]
            if bad:
                raise ValueError(f"{iso}/{sex}: qx not a probability at {bad[:3]}")


def wpp_lifetables() -> dict[str, dict[str, dict[int, float]]]:
    """{iso2: {'M'|'F': {age: qx}}} for WPP_YEAR, streamed out of the two per-sex files.

    Each file is ~200 MB gzipped and carries 1950-2023 for every location and every single year of age —
    roughly ten million rows, of which we want one year of one column. So they are filtered as they
    arrive and never written to disk whole; the filtered extract is cached under data/cache/ (gitignored)
    so a rerun costs nothing.
    """
    # The key carries a schema number as well as the year: changing the filter below has to miss the
    # cache, or a rerun quietly re-reads an extract built by different rules.
    path = os.path.join(CACHE, f"wpp2024_qx_{WPP_YEAR}_v2.json")
    if os.path.exists(path):
        with open(path) as fh:
            raw = json.load(fh)
        cached = {iso: {s: {int(a): q for a, q in ages.items()} for s, ages in sexes.items()}
                  for iso, sexes in raw.items()}
        _validate(cached)          # a cache is a file someone else can have truncated too
        return cached

    os.makedirs(CACHE, exist_ok=True)
    out: dict[str, dict[str, dict[int, float]]] = {}
    for sex, fname in WPP_FILES.items():
        print(f"  streaming {fname} …", flush=True)
        with urllib.request.urlopen(f"{WPP_BASE}/{fname}", timeout=TIMEOUT) as resp:
            if resp.url.split(":", 1)[0] != "https":
                raise ValueError(f"refusing a redirect off https: {resp.url}")
            # gzip.GzipFile over the socket, not a download-then-read: the decompressed file is several
            # gigabytes. csv.reader with column indices rather than DictReader — building ten million
            # dicts to throw all but 24,000 of them away is most of the runtime.
            raw = _Budgeted(gzip.GzipFile(fileobj=resp))
            with io.TextIOWrapper(io.BufferedReader(raw), encoding="utf-8-sig") as text:
                reader = csv.reader(text)
                head = next(reader)
                i_iso2, i_loctype = head.index("ISO2_code"), head.index("LocTypeName")
                i_time, i_variant = head.index("Time"), head.index("Variant")
                i_age, i_qx = head.index("AgeGrpStart"), head.index("qx")
                i_sex = head.index("Sex")
                want = str(WPP_YEAR)
                for row in reader:
                    if row[i_time] != want:
                        continue
                    # The file opens with regional aggregates ("ADB region: Central and West Asia") that
                    # carry no ISO code; a country baseline built from one of those would be nonsense.
                    if row[i_loctype] != "Country/Area" or not row[i_iso2]:
                        continue
                    # A projection variant would look exactly like an estimate by the time it reached a
                    # life table. Refuse it at the door rather than trusting the filename.
                    if row[i_variant] != "Medium":
                        continue
                    if row[i_sex] != EXPECTED_SEX[sex]:
                        raise ValueError(f"{fname} holds {row[i_sex]!r} rows, not "
                                         f"{EXPECTED_SEX[sex]!r} — refusing to file them under {sex!r}")
                    out.setdefault(row[i_iso2], {}).setdefault(sex, {})[int(row[i_age])] = float(row[i_qx])

    _validate(out)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump({iso: {s: {str(a): q for a, q in ages.items()} for s, ages in sexes.items()}
                   for iso, sexes in out.items()}, fh)
    os.replace(tmp, path)          # an interrupted write must not leave a file the next run trusts
    return out


def _closed_at(qx: dict[int, float], last: int) -> dict[int, float]:
    """The same table re-closed at an earlier open interval, the way Eurostat closes at 95.

    Needed to separate two things the raw difference blends: how much the DISPLAYED clock moves (which
    is what a reader would see) and how much the two AGENCIES actually disagree (which is what "switch
    source" is a decision about). Roughly 60% of the raw movement is the closeout, and it is
    one-directional — Eurostat's `Y_GE95` carries PROBDEATH = 1.0, so `remaining_le` returns exactly
    half a year at 95 no matter what the country's old-age mortality is.
    """
    return {a: (1.0 if a == last else q) for a, q in qx.items() if a <= last}


def _ranks(values: dict[str, float]) -> dict[str, int]:
    """1 = longest lives. Ties take the position they sort into; no country here ties to 1e-9."""
    return {iso: i + 1 for i, iso in enumerate(sorted(values, key=lambda k: -values[k]))}


def _rank_moves(a: dict[str, float], b: dict[str, float]) -> list[tuple[int, str, int, int]]:
    """Every country's (displacement, iso, rank_before, rank_after), biggest move first.

    A rank correlation answers "is the ordering broadly the same"; it cannot answer "does any country
    move somewhere its neighbours would notice". Cyprus moving 3rd to 20th is invisible in ρ = 0.885
    and is the single most decision-relevant thing in this comparison.
    """
    ra, rb = _ranks(a), _ranks(b)
    return sorted(((abs(ra[k] - rb[k]), k, ra[k], rb[k]) for k in ra), reverse=True)


# Decision rule, fixed before the numbers were looked at (the gbm_benchmark.py precedent). These are
# not gates on the bundle — W-A3 has its own — they are the thresholds at which this experiment stops
# saying "cheap" and starts saying "look at this first".
MAX_MOVE_YR = 2.0        # a country whose displayed clock moves more than this
MAX_RANK_MOVE = 10       # a country that changes places with ten of its neighbours


def main() -> int:
    print(f"EXP-15 — Eurostat demo_mlifetable {LIFETABLE_YEAR} vs UN WPP 2024 (estimates, {WPP_YEAR})\n")

    try:
        wpp = wpp_lifetables()
    except Exception as e:                                   # noqa: BLE001 — the verdict is the product
        print(f"FAILED: could not read WPP ({e})")
        return 1

    rows, missing = [], []
    for iso in sorted(EUROSTAT_COUNTRIES):
        try:
            euro = eurostat.fetch_lifetable(iso, LIFETABLE_YEAR)
        except Exception as e:                               # noqa: BLE001
            missing.append(f"{iso} (eurostat: {e})")
            continue
        w = wpp.get(ALIASES.get(iso, iso))
        if not w or not euro.get("M") or not euro.get("F"):
            missing.append(f"{iso} (no table on one side)")
            continue
        rec = {"iso": iso}
        for sex in ("M", "F"):
            euro_last = max(euro[sex])
            for age in (0, 40, 95):
                rec[f"e{age}{sex}_eu"] = remaining_le(euro[sex], age, 1.0)
                rec[f"e{age}{sex}_un"] = remaining_le(w[sex], age, 1.0)
                # …and the same WPP table re-closed where Eurostat closes, so the closeout artifact
                # can be subtracted out instead of being reported as disagreement.
                rec[f"e{age}{sex}_unc"] = remaining_le(_closed_at(w[sex], euro_last), age, 1.0)
            rec[f"last{sex}_eu"] = euro_last
            rec[f"last{sex}_un"] = max(w[sex])
        rows.append(rec)

    if missing:
        # A partial table is worse than no table: it would be read as "the switch is cheap" when the
        # expensive countries are the ones that dropped out.
        print("FAILED: incomplete comparison — " + "; ".join(missing))
        return 1

    def delta(rec, age, sex, key="un"):
        return rec[f"e{age}{sex}_eu"] - rec[f"e{age}{sex}_{key}"]

    def worst_of(rec, age):
        return max(abs(delta(rec, age, "M")), abs(delta(rec, age, "F")))

    print(f"{'':4} {'e0 men':>17} {'e0 women':>17} {'e40 men':>17} {'e40 women':>17}")
    print(f"{'ISO':4} {'euro':>6}{'un':>6}{'Δ':>5} {'euro':>6}{'un':>6}{'Δ':>5}"
          f" {'euro':>6}{'un':>6}{'Δ':>5} {'euro':>6}{'un':>6}{'Δ':>5}")
    # Sorted by the same statistic the summary calls "worst", so the table's top row and the named
    # country cannot disagree.
    for r in sorted(rows, key=lambda r: -max(worst_of(r, 0), worst_of(r, 40))):
        cells = [f"{r[f'e{age}{s}_eu']:6.1f}{r[f'e{age}{s}_un']:6.1f}{delta(r, age, s):5.1f}"
                 for age in (0, 40) for s in ("M", "F")]
        print(f"{r['iso']:4} " + " ".join(cells))

    print(f"\n{len(rows)} countries compared; WPP covers {len(wpp)} countries or areas in total.\n")

    for age, what in ((0, "at birth"), (40, "at 40 — the quantity the bundle ships as national_le_40")):
        raw = [abs(delta(r, age, s)) for r in rows for s in ("M", "F")]
        matched = [abs(delta(r, age, s, "unc")) for r in rows for s in ("M", "F")]
        worst = max(rows, key=lambda r: worst_of(r, age))
        print(f"e{age} ({what})")
        print(f"  displayed movement      median {statistics.median(raw):.2f} yr, "
              f"max {max(raw):.2f} ({worst['iso']})")
        # Same tables, same closeout: what is left is the agencies actually disagreeing.
        print(f"  agency disagreement     median {statistics.median(matched):.2f} yr, "
              f"max {max(matched):.2f}   (WPP re-closed at 95, as Eurostat closes)")
        for s, label in (("M", "men  "), ("F", "women")):
            signed = [delta(r, age, s) for r in rows]
            higher = sum(1 for d in signed if d > 0)
            # Absolute values hide that the switch moves the sexes in opposite directions.
            print(f"  {label} signed mean {statistics.mean(signed):+.3f} yr "
                  f"(Eurostat higher in {higher}/{len(rows)})")
        for s, label in (("M", "men"), ("F", "women")):
            eu = {r["iso"]: r[f"e{age}{s}_eu"] for r in rows}
            un = {r["iso"]: r[f"e{age}{s}_un"] for r in rows}
            rho = spearmanr(list(eu.values()), [un[k] for k in eu]).statistic
            moves = _rank_moves(eu, un)
            top = ", ".join(f"{iso} {a}→{b}" for _, iso, a, b in moves[:3])
            print(f"  {label:5} Spearman {rho:.3f}; biggest rank moves: {top}")
        print()

    big_move = [(r["iso"], max(worst_of(r, 0), worst_of(r, 40))) for r in rows
                if max(worst_of(r, 0), worst_of(r, 40)) > MAX_MOVE_YR]
    big_rank = []
    for age in (0, 40):
        for s in ("M", "F"):
            eu = {r["iso"]: r[f"e{age}{s}_eu"] for r in rows}
            un = {r["iso"]: r[f"e{age}{s}_un"] for r in rows}
            for d, iso, a, b in _rank_moves(eu, un):
                if d > MAX_RANK_MOVE:
                    big_rank.append(f"{iso} {a}→{b} (e{age} {'men' if s == 'M' else 'women'})")

    e95_eu = [r[f"e95{s}_eu"] for r in rows for s in ("M", "F")]
    e95_un = [r[f"e95{s}_un"] for r in rows for s in ("M", "F")]
    print(f"oldest age in the table: Eurostat {min(r[f'last{s}_eu'] for r in rows for s in 'MF')}"
          f"-{max(r[f'last{s}_eu'] for r in rows for s in 'MF')}, "
          f"WPP {min(r[f'last{s}_un'] for r in rows for s in 'MF')}"
          f"-{max(r[f'last{s}_un'] for r in rows for s in 'MF')}")
    print(f"remaining years at 95: Eurostat {min(e95_eu):.2f}-{max(e95_eu):.2f}, "
          f"WPP {min(e95_un):.2f}-{max(e95_un):.2f} (median {statistics.median(e95_un):.2f})")
    # Both figures are artifacts of the same bug, at different sizes. Eurostat's Y_GE95 carries
    # PROBDEATH = 1.0, so 0.50 is arithmetic, not a claim about 95-year-olds; WPP closes at 100 the
    # same way, which is why its 95 figure still lands ~0.07 below WPP's own published ex. The defect
    # is `baselines.remaining_le` treating an open interval as a one-year interval — switching source
    # shrinks it, and does not fix it.
    print("   (both are closeout artifacts of remaining_le treating an open interval as one year;\n"
          "    WPP moves the truncation from 95 to 100, it does not remove it)")

    ok = not big_move and not big_rank
    print(f"\nVERDICT: switching the baseline to the UN WPP 2024 revision ({WPP_YEAR} estimates) moves "
          f"the displayed clock at birth by {statistics.median([abs(delta(r, 0, s)) for r in rows for s in 'MF']):.2f} "
          f"years at the median; with the closeout matched, the two agencies disagree by "
          f"{statistics.median([abs(delta(r, 0, s, 'unc')) for r in rows for s in 'MF']):.2f}. "
          f"Coverage goes {len(rows)} → {len(wpp)} countries.")
    if big_move:
        print(f"  ATTENTION: {len(big_move)} country-sex clocks move more than {MAX_MOVE_YR} yr: "
              + ", ".join(f"{iso} {d:.2f}" for iso, d in big_move))
    if big_rank:
        print(f"  ATTENTION: {len(big_rank)} rank moves exceed {MAX_RANK_MOVE} places: "
              + "; ".join(big_rank[:6]))
    if ok:
        print(f"  No country moves more than {MAX_MOVE_YR} yr and none changes more than "
              f"{MAX_RANK_MOVE} places.")
    print("  This is a measurement, not an approval: the thresholds above are what makes it "
          "reportable, and the decision is the owner's.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
