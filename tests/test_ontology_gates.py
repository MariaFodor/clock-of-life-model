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
        """ONE problem must carry every fragment — the field path AND what is wrong with it.

        Asserting only that the list is non-empty let both `must be an object` messages be deleted
        with the suite still green. Testing each fragment against the list SEPARATELY was barely
        better: a sibling problem supplies the field name, so dropping the name from a message — or
        naming the wrong factor entirely — still passed. A gate that refuses without saying which
        factor is wrong is barely better than the AttributeError it replaced.
        """
        return any(all(frag in p for frag in must_contain) for p in problems)

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
    for field, frag in (("doi", "no doi/url"), ("verified", "never verified"),
                        ("supports", "does not say WHICH claim it supports")):
        srcs = copy.deepcopy(good["sources"])
        srcs[0].pop(field)
        problems = mutate(intervention_evidence={**good, "sources": srcs})
        check(f"a source missing `{field}` is reported, naming the source",
              reports(problems, f"{FACTOR}.intervention_evidence.sources[0]", frag),
              "; ".join(problems)[:70])

    # A url is as openable as a doi.
    srcs = copy.deepcopy(good["sources"])
    srcs[0].pop("doi")
    srcs[0]["url"] = "https://doi.org/10.1093/aje/kwf150"
    check("a source with a url instead of a doi is accepted",
          not mutate(intervention_evidence={**good, "sources": srcs}))

    check("a block with no claim is reported, naming the block",
          reports(mutate(intervention_evidence={x: v for x, v in good.items() if x != "claim"}),
                  f"{FACTOR}.intervention_evidence", "does not say what it is qualifying"))

    # priors get the same treatment — this was asymmetric, and the string case raised.
    for label, value, frags in [
            ("null", None, (f"{FACTOR}.prior", "no doi/url")),
            ("a bare doi string", "10.1136/bmj.m3324", (f"{FACTOR}.prior", "must be an object")),
            ("a list", ["10.1136/bmj.m3324"], (f"{FACTOR}.prior", "must be an object"))]:
        problems = mutate(prior=value)
        check(f"prior as {label} is reported, not raised", reports(problems, *frags),
              "; ".join(problems)[:70])

    # The priors' verification branch was the ONLY enforcement of "verified" in the repo, covering
    # 20 of 22 factors, and nothing pinned it: deleting it left this suite green, 50 gates passing
    # and a bundle shipping. The sources' twin WAS pinned, so the suite tested one arm of a promise
    # and not the other — the asymmetry two commits on this branch each set out to end.
    for field in ("prior", "prior_secondary"):
        donor = ONT[FACTOR]["prior"]
        unverified = {k: v for k, v in donor.items() if k != "verified"}
        problems = mutate(**{field: unverified})
        check(f"an unverified {field} is reported, naming the field",
              reports(problems, f"{FACTOR}.{field}", "never verified"), "; ".join(problems)[:70])

    # `reports()` is only as strong as "one problem per defect". Join the list into a single string
    # — a plausible tidy-up, since ontology_gates already does exactly that for display — and every
    # fragment check goes on matching while the messages degrade freely. So: one defect, one
    # problem; and separate defects stay separate rather than arriving as one blob.
    one_defect = mutate(prior={k: v for k, v in ONT[FACTOR]["prior"].items() if k != "verified"})
    check("one defect produces one problem, naming the field", len(one_defect) == 1
          and reports(one_defect, f"{FACTOR}.prior", "never verified"),
          f"{len(one_defect)}: {one_defect}")

    srcs = copy.deepcopy(good["sources"])
    srcs[0].pop("verified")
    srcs[1].pop("doi")
    two = mutate(intervention_evidence={**good, "sources": srcs})
    # `> 1` and not `== 2`: an exact count fails when the gate gets STRONGER — adding a block-level
    # summary problem, or a second message explaining a doi-less source, both break `== 2` while
    # announcing "not one merged blob". What this needs to pin is that separate defects stay
    # separate AND are attributed to the right source; the anti-duplication half is already pinned
    # by the one-defect case above, on a cleaner fixture.
    check("two defects arrive as two problems, not one merged blob",
          len(two) > 1
          and reports(two, f"{FACTOR}.intervention_evidence.sources[0]", "never verified")
          and reports(two, f"{FACTOR}.intervention_evidence.sources[1]", "no doi/url"),
          f"{len(two)}: {'; '.join(two)[:60]}")

    # The index must name WHICH source is wrong. `sources[{i}]` -> `sources[0]` is one token, and no
    # case reached the second source before this one. BOTH sites that format an index are covered:
    # pinning one left the other free to hardcode 0.
    srcs = copy.deepcopy(good["sources"])
    srcs[1].pop("verified")
    check("the second source is named as the second",
          reports(mutate(intervention_evidence={**good, "sources": srcs}),
                  f"{FACTOR}.intervention_evidence.sources[1]", "never verified"))
    srcs = copy.deepcopy(good["sources"])
    srcs[1] = "10.1136/tc.2005.011932"
    check("a non-object SECOND source is named as the second",
          reports(mutate(intervention_evidence={**good, "sources": srcs}),
                  f"{FACTOR}.intervention_evidence.sources[1]", "not an object"))

    # All three sites that format `{field}` must say WHICH of prior/prior_secondary is wrong. One
    # was pinned; the other two could collapse to the literal `prior` and report an env.prior_secondary
    # defect as env.prior — the mislabelling this branch already claimed to have ended once.
    check("a prior_secondary of the wrong shape is named as prior_secondary",
          reports(mutate(prior_secondary="10.1136/bmj.m3324"),
                  f"{FACTOR}.prior_secondary", "must be an object"))
    check("a prior_secondary with no doi/url is named as prior_secondary",
          reports(mutate(prior_secondary={"title": "x"}),
                  f"{FACTOR}.prior_secondary", "no doi/url"))

    # `elif` -> `if` is one token, and it makes a citation with NEITHER doi nor verified report
    # "doi present but never verified" about a doi that is not present — a message that sends the
    # author looking for the wrong thing, and a second problem for a single defect.
    no_doi_no_verified = mutate(prior={"title": "x"})
    check("a citation missing both doi and verification says so once, correctly",
          len(no_doi_no_verified) == 1
          and reports(no_doi_no_verified, f"{FACTOR}.prior", "no doi/url"),
          "; ".join(no_doi_no_verified)[:70])

    # The SOURCE arm of the same `elif`. Pinning the prior's and not this one is the third time this
    # branch has fixed an asymmetry on one side only: last round the source was pinned and the prior
    # was not, so the fix inverted it rather than ending it.
    srcs = copy.deepcopy(good["sources"])
    srcs[0].pop("doi")
    srcs[0].pop("verified")
    src_both = mutate(intervention_evidence={**good, "sources": srcs})
    check("a SOURCE missing both doi and verification says so once, correctly",
          len(src_both) == 1
          and reports(src_both, f"{FACTOR}.intervention_evidence.sources[0]", "no doi/url"),
          "; ".join(src_both)[:70])

    # The prior's url fallback: the source's is pinned above, the prior's was not — same family.
    check("a prior with a url and no doi is accepted",
          not mutate(prior={"url": "https://doi.org/10.1136/bmj.m3324", "verified": "2026-09-10"}))

    # `continue` -> `break` is one token, and it stops the scan at the first malformed source, so
    # everything after it goes unreported and the author fixes one thing at a time.
    srcs = copy.deepcopy(good["sources"])
    srcs[0] = "10.1093/aje/kwf150"
    srcs[1].pop("verified")
    both = mutate(intervention_evidence={**good, "sources": srcs})
    check("a malformed source does not hide the ones after it",
          reports(both, f"{FACTOR}.intervention_evidence.sources[0]", "not an object")
          and reports(both, f"{FACTOR}.intervention_evidence.sources[1]", "never verified"),
          "; ".join(both)[:70])

    # Present-but-empty is not declared. `not X.get(k)` -> `"k" not in X` is one token and it lets
    # a citation ship with `"doi": ""` or `"verified": null` — an unopenable citation, which is the
    # thing this gate exists to say is not a citation. Both `raw is not None` guards are the same
    # shape: drop them and a null block stops being reported as null.
    for label, bad in (
            ("doi and url both empty", {**ONT[FACTOR]["prior"], "doi": "", "url": ""}),
            ("verified null", {**ONT[FACTOR]["prior"], "verified": None}),
            ("verified empty", {**ONT[FACTOR]["prior"], "verified": ""})):
        check(f"a prior with {label} is reported",
              reports(mutate(prior=bad), f"{FACTOR}.prior"), label)
    # An empty `url` alongside a real doi is NOT a defect: the doi is what makes it openable, and a
    # gate that rejected it would be demanding a field the format does not require.
    check("an empty url alongside a good doi is accepted",
          not mutate(prior={**ONT[FACTOR]["prior"], "url": ""}))
    for field, empty in (("doi", ""), ("verified", None), ("supports", "")):
        srcs = copy.deepcopy(good["sources"])
        srcs[0][field] = empty
        srcs[0].pop("url", None)
        check(f"a source whose `{field}` is present but empty is reported",
              reports(mutate(intervention_evidence={**good, "sources": srcs}),
                      f"{FACTOR}.intervention_evidence.sources[0]"),
              f"{field}={empty!r}")
    # A null is a declared-nothing, not a wrong TYPE. Dropping the `is not None` guards makes the
    # gate say "must be an object" about a null — and emit two problems for one defect, since the
    # coerced {} then also reports no doi/url. Same class as the elif above: the message sends the
    # author after a shape problem when the real one is an empty declaration.
    for field in ("prior", "prior_secondary"):
        null_one = mutate(**{field: None})
        check(f"a null {field} is reported as empty, once, not as a wrong type",
              len(null_one) == 1 and reports(null_one, f"{FACTOR}.{field}", "no doi/url")
              and not any("must be an object" in p for p in null_one),
              "; ".join(null_one)[:70])
    null_ie = mutate(intervention_evidence=None)
    check("a null intervention_evidence is reported as empty, not as a wrong type",
          reports(null_ie, f"{FACTOR}.intervention_evidence", "declared with no sources")
          and not any("must be an object" in p for p in null_ie),
          "; ".join(null_ie)[:70])

    check("a block whose `claim` is present but empty is reported",
          reports(mutate(intervention_evidence={**good, "claim": ""}),
                  f"{FACTOR}.intervention_evidence", "does not say what it is qualifying"))

    failed = [n for n, ok, _ in checks if not ok]
    print(f"\nONTOLOGY GATES: {'PASS' if not failed else 'FAIL'}  ({len(checks) - len(failed)}/{len(checks)})")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
