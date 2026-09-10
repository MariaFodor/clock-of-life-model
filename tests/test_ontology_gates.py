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

    check("the shipped ontology has no citation problems", not citation_problems())

    # A block that is absent is not a defect: most factors have no intervention advice to qualify.
    check("an absent intervention_evidence block is fine",
          not mutate(intervention_evidence=KeyError))

    # A block that is present but says nothing is worse than none: it reads as "the model has
    # sources for this" while declaring zero.
    for label, value in [("null", None), ("empty object", {}), ("a list", []),
                         ("a string", "Godtfredsen 2002"), ("a number", 7)]:
        problems = mutate(intervention_evidence=value)
        check(f"intervention_evidence as {label} is reported", problems,
              "; ".join(problems)[:70])

    good = copy.deepcopy(ONT[FACTOR]["intervention_evidence"])
    for label, value in [("a dict", {"a": 1}), ("a string", "x"),
                         ("a list of strings", ["10.1093/aje/kwf150"]), ("empty", [])]:
        problems = mutate(intervention_evidence={**good, "sources": value})
        check(f"sources as {label} is reported", problems, "; ".join(problems)[:70])

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
    for label, value in [("null", None), ("a bare doi string", "10.1136/bmj.m3324"),
                         ("a list", ["10.1136/bmj.m3324"])]:
        problems = mutate(prior=value)
        check(f"prior as {label} is reported, not raised", problems, "; ".join(problems)[:70])

    failed = [n for n, ok, _ in checks if not ok]
    print(f"\nONTOLOGY GATES: {'PASS' if not failed else 'FAIL'}  ({len(checks) - len(failed)}/{len(checks)})")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
