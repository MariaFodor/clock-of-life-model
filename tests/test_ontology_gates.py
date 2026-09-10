"""The citation gate's shape handling, pinned.

Every one of these was verified once by hand and reported in a commit message, which is not a
place: the same argument this repo made against registering a defect in prose applies to
registering a proof in prose. A malformed declaration must REPORT — naming the field — rather than
raise, because an AttributeError mid-gate refuses the release for the wrong reason and tells the
author nothing about which factor is wrong.

Run: PYTHONPATH=src python3 tests/test_ontology_gates.py   (no network, no bundle needed)
"""
import copy
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from clock_model.ontology import ONT, citation_problems

FACTOR = "cigs_day"  # the only factor carrying intervention_evidence today


def mutate(**changes):
    o = copy.deepcopy(ONT)
    for path, value in changes.items():
        spec = o[FACTOR]
        if value is KeyError:
            spec.pop(path, None)
        else:
            spec[path] = value
    return citation_problems(o)


def main():
    checks = []

    def check(name, ok, detail=""):
        checks.append((name, bool(ok), detail))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))

    def reports(problems, *must_contain):
        """A problem must NAME the field. Asserting only that the list is non-empty let both
        `must be an object` messages be deleted with the suite still green — and a gate that
        refuses without saying which factor is wrong is barely better than the AttributeError it
        replaced."""
        return problems and all(any(frag in p for p in problems) for frag in must_contain)

    check("the shipped ontology has no citation problems", not citation_problems())

    # A block that is absent is not a defect: most factors have no intervention advice to qualify.
    check("an absent intervention_evidence block is fine",
          not mutate(intervention_evidence=KeyError))

    # A block that is present but says nothing is worse than none: it reads as "the model has
    # sources for this" while declaring zero.
    for label, value, frags in [
            ("null", None, (f"{FACTOR}.intervention_evidence", "declared with no sources")),
            ("empty object", {}, (f"{FACTOR}.intervention_evidence", "declared with no sources")),
            ("a list", [], (f"{FACTOR}.intervention_evidence", "declared with no sources")),
            ("a string", "Godtfredsen 2002", (f"{FACTOR}.intervention_evidence", "must be an object")),
            ("a number", 7, (f"{FACTOR}.intervention_evidence", "must be an object"))]:
        problems = mutate(intervention_evidence=value)
        check(f"intervention_evidence as {label} is reported, naming the field",
              reports(problems, *frags), "; ".join(problems)[:70])

    good = copy.deepcopy(ONT[FACTOR]["intervention_evidence"])
    for label, value, frag in [("a dict", {"a": 1}, "sources: must be a list"),
                               ("a string", "x", "sources: must be a list"),
                               ("a list of strings", ["10.1093/aje/kwf150"], "sources[0]: not an object"),
                               ("empty", [], "declared with no sources")]:
        problems = mutate(intervention_evidence={**good, "sources": value})
        check(f"sources as {label} is reported, naming the field",
              reports(problems, f"{FACTOR}.intervention_evidence", frag),
              "; ".join(problems)[:70])

    # Each source is held to the prior's bar, plus a statement of what it supports.
    for field in ("doi", "verified", "supports"):
        srcs = copy.deepcopy(good["sources"])
        srcs[0].pop(field)
        problems = mutate(intervention_evidence={**good, "sources": srcs})
        check(f"a source missing `{field}` is reported", problems, "; ".join(problems)[:70])

    # A url is as openable as a doi.
    srcs = copy.deepcopy(good["sources"])
    srcs[0].pop("doi")
    srcs[0]["url"] = "https://doi.org/10.1093/aje/kwf150"
    check("a source with a url instead of a doi is accepted",
          not mutate(intervention_evidence={**good, "sources": srcs}))

    check("a block with no claim is reported",
          mutate(intervention_evidence={k: v for k, v in good.items() if k != "claim"}))

    # priors get the same treatment — this was asymmetric, and the string case raised.
    for label, value, frags in [
            ("null", None, (f"{FACTOR}.prior", "no doi/url")),
            ("a bare doi string", "10.1136/bmj.m3324", (f"{FACTOR}.prior", "must be an object")),
            ("a list", ["10.1136/bmj.m3324"], (f"{FACTOR}.prior", "must be an object"))]:
        problems = mutate(prior=value)
        check(f"prior as {label} is reported, not raised", reports(problems, *frags),
              "; ".join(problems)[:70])

    failed = [n for n, ok, _ in checks if not ok]
    print(f"\nONTOLOGY GATES: {'PASS' if not failed else 'FAIL'}  ({len(checks) - len(failed)}/{len(checks)})")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
