#!/usr/bin/env python3
"""Recompute every number in reports/abstain-reconciliation.md from row dumps already on disk.

Runs no model, submits no job, touches nothing outside reports/ (it only writes
reports/abstain-reconciliation.json). Reads:

  reports/rows-kev-27b-probe.jsonl     the census's 2,510 panel_choice records, Kev-27B
  reports/jev-full/rows.jsonl          the census's 2,510 panel_choice records, Jev
  reports/census-suite-ids.jsonl       the 7,203-row census-matched suite population
  reports/census-suite-ids-exact.jsonl the 3,594-row like-for-like (same-prompt) subset
  reports/rows-full-*.jsonl            our arms (suite-level: id = test/custom/<n>)

Usage:  source scripts/wsl/env.sh && uv run --no-sync python reports/abstain-reconciliation.py
"""
from __future__ import annotations

import hashlib
import json
import os
from math import comb, lgamma, log

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "reports", "abstain-reconciliation.json")


def read_jsonl(path: str) -> list[dict]:
    out = []
    with open(os.path.join(ROOT, path)) as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def argmax(xs):
    best = 0
    for i in range(1, len(xs)):
        if xs[i] > xs[best]:
            best = i
    return best


def forced(r):
    """argmax over p with the abstain option removed (abstention not offered)."""
    a, p = r["abstain"], r["p"]
    best, bv = None, None
    for i in range(len(p)):
        if i == a:
            continue
        if bv is None or p[i] > bv:
            bv, best = p[i], i
    return best


def served(r):
    return argmax(r["p"])


def answer_field(r):
    try:
        return r["keys"].index(r["answer"])
    except Exception:
        return None


def binom_two_sided_log10(b: int, c: int) -> float:
    """``log10`` of the same two-sided exact binomial -- the readable form when ``p`` underflows.

    :func:`binom_two_sided` returns a float, so a p-value below ``5e-324`` comes back as ``0.0``:
    correctly rounded for the type, and unreadable in a table. A lopsided pair on 63,851 questions
    reaches that, so the magnitude is carried here instead: ``math.lgamma`` for ``log C(n, k)``, the
    exact descending ratio recurrence ``C(n, j-1) / C(n, j) = j / (n - j + 1)`` for the rest of the
    tail (every ratio is <= 1 for ``j <= k <= n/2``, so the sum accumulates without cancellation), and
    **no power of two as a float** anywhere. ``0.0`` exactly where the value is capped at ``1.0``.

    This is the one name :func:`binom_two_sided` reaches for, and it is a name both live consumers ask
    for (``scripts/noul-accuracy.py``, whose populations are the only ones that can reach that range).
    """
    n = b + c
    if n == 0:
        return 0.0                                      # p = 1.0 by construction
    k = min(b, c)
    ratio = acc = 1.0                                   # acc = sum_j C(n, j) / C(n, k)
    for j in range(k, 0, -1):
        ratio *= j / (n - j + 1)
        acc += ratio
    log_p = (lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1) + log(acc)
             + log(2.0) - n * log(2.0))
    return 0.0 if log_p >= 0.0 else log_p / log(10.0)


def binom_two_sided(b: int, c: int) -> float:
    """Exact two-sided binomial on discordant pairs -- the test reports/PAIRED-TESTS.md uses.

    ``min(1.0, 2 * sum_{i <= min(b, c)} C(n, i) / 2 ** n)`` with ``n = b + c``: the *same* number the
    settled expression returns, computed so that it survives an ``n`` in the tens of thousands (the v4
    noul family's discordant-pair counts, 63,851 questions per arm), where ``2.0 ** n`` -- the float
    power the original expression divided by -- raised ``OverflowError``.

    Three ranges, each the cheapest that is exact there. The cutoff and the tail recurrence below are
    written **here, as literals**, on purpose: this definition is lifted out of the file's *source* by
    ``scripts/noul-accuracy.py``, ``scripts/arms-forced.py`` and ``scripts/9b-finetune-score.py``
    (imports and function bodies, not module-level state -- importing this file would rewrite
    ``reports/abstain-reconciliation.json``), so a module constant or a helper function it reached for
    by name would be a ``NameError`` inside a live scoring run. One name is unavoidable and is guarded
    below.

    * ``n <= 1023`` (``2.0 ** n`` finite): the settled float expression, **unchanged**, including its
      ``min(1.0, ...)`` cap -- the published p-values come from exactly this arithmetic;
    * ``n >= 1024`` and ``min(b, c) <= 4096``: the tail over **exact integers**, one rounding at the
      end. This is the same double the settled expression produces wherever that one runs -- scaling by
      a power of two commutes with rounding to nearest, so ``round(2S) / 2**n == round(2S / 2**n)``,
      and ``int / int`` is correctly rounded at any size -- and no power of two becomes a float;
    * ``min(b, c) > 4096``: **log space** (:func:`binom_two_sided_log10`), the only range that needs
      another name, and the only one no choice-lane population can reach (1,819 and 2,510 questions a
      side cap ``min(b, c)`` at 1,255). At the noul lane's scale it is the everyday branch -- and a
      caller that lifted this function *alone* is told so by name instead of by ``NameError``.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    if n <= 1023:                                       # 2.0 ** 1024 is the OverflowError
        return min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / (2.0 ** n))
    if k <= 4096:
        term = total = 1                                # C(n, i) by the recurrence: integers only
        for i in range(1, k + 1):
            term = term * (n - i + 1) // i
            total += term
        return min(1.0, 2 * total / (2 ** n))
    if "binom_two_sided_log10" not in globals():
        raise RuntimeError(
            f"n={n}, min(b, c)={k} needs binom_two_sided_log10, and this caller lifted "
            f"binom_two_sided alone; scripts/noul-accuracy.py lifts both names")
    log10_p = binom_two_sided_log10(b, c)
    return 1.0 if log10_p >= 0.0 else min(1.0, 10.0 ** log10_p)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(os.path.join(ROOT, path), "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------- load
KEV = {r["id"]: r for r in read_jsonl("reports/rows-kev-27b-probe.jsonl")
       if r.get("family") == "panel_choice"}
JEV = {r["id"]: r for r in read_jsonl("reports/jev-full/rows.jsonl")
       if r.get("family") == "panel_choice"}
CENSUS_IDS = sorted(KEV)
assert set(KEV) == set(JEV), "the two census dumps do not carry the same panel_choice ids"

SUITE = read_jsonl("reports/census-suite-ids.jsonl")
PC_SUITE = [r for r in SUITE if r["family"] == "panel_choice"]
EXACT = read_jsonl("reports/census-suite-ids-exact.jsonl")
PC_EXACT = [r for r in EXACT if r["family"] == "panel_choice"]

# census verdicts, attributed to suite rows by (dataset, cell, family)
ID1819 = [f"test/{r['dataset']}/{r['cell']}/{r['family']}" for r in PC_SUITE]
ID909 = [f"test/{r['dataset']}/{r['cell']}/{r['family']}" for r in PC_EXACT]

RULES = {"served_argmax": served, "served_answer_field": answer_field, "forced": forced}
KOK = {k: {i: (f(KEV[i]) == KEV[i]["label"]) for i in CENSUS_IDS} for k, f in RULES.items()}
JOK = {k: {i: (f(JEV[i]) == JEV[i]["label"]) for i in CENSUS_IDS} for k, f in RULES.items()}


def pair(ka: str, kb: str, ids: list[str], label: str) -> dict:
    a, b = KOK[ka], JOK[kb]
    n = len(ids)
    ra = sum(1 for i in ids if a[i])
    rb = sum(1 for i in ids if b[i])
    x = sum(1 for i in ids if a[i] and not b[i])
    y = sum(1 for i in ids if b[i] and not a[i])
    return {"comparison": label, "n": n, "a_correct": ra, "b_correct": rb,
            "a_acc": ra / n, "b_acc": rb / n,
            "a_right_b_wrong": x, "b_right_a_wrong": y, "p": binom_two_sided(x, y)}


out: dict = {"what_this_is":
             "Common-footing recomputation for reports/abstain-reconciliation.md; "
             "reads only row dumps already on disk.", "populations": {}}

out["populations"]["census_2510_panel_choice"] = {
    "source": "reports/rows-kev-27b-probe.jsonl + reports/jev-full/rows.jsonl",
    "definition": "the 2,510 census records whose family is panel_choice; both dumps carry "
                  "exactly the same 2,510 ids, so no question is missing on either side",
    "n": 2510,
    "pairs": [
        pair("forced", "served_answer_field", CENSUS_IDS,
             "27B forced vs Jev as served (answer field)"),
        pair("forced", "served_argmax", CENSUS_IDS,
             "27B forced vs Jev as served (argmax p)"),
        pair("forced", "forced", CENSUS_IDS, "27B forced vs Jev forced"),
        pair("served_argmax", "served_answer_field", CENSUS_IDS,
             "27B as served vs Jev as served"),
        pair("served_argmax", "forced", CENSUS_IDS, "27B as served vs Jev forced"),
    ]}

out["populations"]["census_matched_1819"] = {
    "source": "reports/census-suite-ids.jsonl x the census dumps, verdict attributed by "
              "(dataset, cell, family)",
    "definition": "the 1,819 suite rows in family panel_choice; 322 of the 1,497 distinct "
                  "triples carry two suite rows, so those census verdicts are counted twice",
    "n": 1819,
    "pairs": [
        pair("forced", "forced", ID1819, "27B forced vs Jev forced"),
        pair("forced", "served_argmax", ID1819, "27B forced vs Jev as served (argmax p)"),
        pair("forced", "served_answer_field", ID1819,
             "27B forced vs Jev as served (answer field)"),
        pair("served_argmax", "served_answer_field", ID1819, "27B as served vs Jev as served"),
    ]}

out["populations"]["like_for_like_909"] = {
    "source": "reports/census-suite-ids-exact.jsonl",
    "definition": "the 909 panel_choice suite rows whose prompt is the census's own record",
    "n": 909,
    "pairs": [
        pair("forced", "forced", ID909, "27B forced vs Jev forced"),
        pair("forced", "served_argmax", ID909, "27B forced vs Jev as served"),
    ]}

per_atlas = {}
for ds in sorted({KEV[i]["dataset"] for i in CENSUS_IDS}):
    ii = [i for i in CENSUS_IDS if KEV[i]["dataset"] == ds]
    per_atlas[ds] = {"n": len(ii),
                     "27B forced vs Jev forced": pair("forced", "forced", ii, "forced/forced"),
                     "27B forced vs Jev as served": pair("forced", "served_answer_field", ii,
                                                         "forced/served")}
out["populations"]["census_2510_per_atlas"] = per_atlas

# --------------------------------------------------------------------- split
H = {i: int(hashlib.sha256((KEV[i]["dataset"] + "|" + KEV[i]["cell"]).encode()).hexdigest(), 16)
     for i in CENSUS_IDS}
ODD = [i for i in CENSUS_IDS if H[i] % 2 == 1]
EVEN = [i for i in CENSUS_IDS if H[i] % 2 == 0]
out["split"] = {
    "rule": "sha256('dataset|cell') % 2  (1,277 odd / 1,233 even)",
    "note": "the fitted margin is 'no abstention' in both directions, i.e. the policy has no "
            "free parameter, so the held-out number equals the full-sample number",
    "odd": {"n": len(ODD),
            "27B_forced_correct": sum(1 for i in ODD if KOK["forced"][i]),
            "27B_forced_acc": sum(1 for i in ODD if KOK["forced"][i]) / len(ODD),
            "Jev_served_correct": sum(1 for i in ODD if JOK["served_answer_field"][i]),
            "Jev_served_acc": sum(1 for i in ODD if JOK["served_answer_field"][i]) / len(ODD)},
    "even": {"n": len(EVEN),
             "27B_forced_correct": sum(1 for i in EVEN if KOK["forced"][i]),
             "27B_forced_acc": sum(1 for i in EVEN if KOK["forced"][i]) / len(EVEN),
             "Jev_served_correct": sum(1 for i in EVEN if JOK["served_answer_field"][i]),
             "Jev_served_acc": sum(1 for i in EVEN if JOK["served_answer_field"][i]) / len(EVEN)},
}

# --------------------------------------------------------------------- arms
ARM_PATHS = {"frozen": "reports/rows-full-frozen.jsonl",
             "armB": "reports/rows-full-armb.jsonl",
             "delta-0.8b-s1": "reports/rows-full-delta-kev-0.8b-s1.jsonl",
             "delta-0.8b-s2": "reports/rows-full-delta-kev-0.8b-s2.jsonl",
             "delta-0.8b-s3": "reports/rows-full-delta-kev-0.8b-s3.jsonl",
             "delta-0.8b-s4": "reports/rows-full-delta-kev-0.8b-s4.jsonl",
             "delta-0.8b-s5": "reports/rows-full-delta-kev-0.8b-s5.jsonl",
             "delta-4b-s1": "reports/rows-full-delta-kev-4b-s1.jsonl",
             "seed1": "reports/rows-full-seed1.jsonl",
             "seed2": "reports/rows-full-seed2.jsonl",
             "seed3": "reports/rows-full-seed3.jsonl"}
ARMS = {k: {r["id"]: r for r in read_jsonl(p)} for k, p in ARM_PATHS.items()}

cells = {}
for name in ("frozen", "armB"):
    for fam in ("panel_choice", "panel_miss", "evidence_diluted"):
        rows = [r for r in SUITE if r["family"] == fam]
        ok = n = ab = 0
        for r in rows:
            a = ARMS[name].get(r["id"])
            if a is None:
                continue
            n += 1
            if a.get("top1") == a.get("label"):
                ok += 1
            if a.get("top1") == a.get("abstain"):
                ab += 1
        cells[f"{name}|{fam}"] = {"correct": ok, "n": n, "accuracy": ok / n,
                                  "abstain_pick": ab / n}
out["cells"] = cells

seed_values = {}
for name in ("delta-0.8b-s1", "delta-0.8b-s2", "delta-0.8b-s3", "delta-0.8b-s4",
             "delta-0.8b-s5"):
    ok = n = 0
    for r in PC_SUITE:
        a = ARMS[name].get(r["id"])
        if a is None:
            continue
        n += 1
        if a.get("top1") == a.get("label"):
            ok += 1
    seed_values[name] = {"correct": ok, "n": n, "accuracy": ok / n}
out["seed_values"] = {
    "manuscript_says": [0.119, 0.044, 0.058, 0.103, 0.059],
    "measured_on_census_matched_1819": seed_values,
    "measured_mean": sum(v["accuracy"] for v in seed_values.values()) / 5,
    "measured_spread_max_over_min": (max(v["accuracy"] for v in seed_values.values())
                                     / min(v["accuracy"] for v in seed_values.values())),
}

out["sha256"] = {p: sha256_file(p) for p in
                 ["reports/rows-kev-27b-probe.jsonl", "reports/jev-full/rows.jsonl",
                  "reports/census-suite-ids.jsonl", "reports/census-suite-ids-exact.jsonl"]
                 + list(ARM_PATHS.values())}

# ------------------------------------------------- proof: A's Jev column is "as served"
def policy(r, margin):
    """Answer with the best non-abstain option unless the abstain option wins by more than margin."""
    a, p = r["abstain"], r["p"]
    if p[a] > margin:
        return a
    return forced(r)


breakpoints = sorted({JEV[i]["p"][JEV[i]["abstain"]] for i in CENSUS_IDS})
margins = [0.0] + [(a + b) / 2 for a, b in zip(breakpoints, breakpoints[1:])] + [1.0, float("inf")]
reach = []
for m in margins:
    j = {i: (policy(JEV[i], m) == JEV[i]["label"]) for i in CENSUS_IDS}
    x = sum(1 for i in CENSUS_IDS if KOK["forced"][i] and not j[i])
    y = sum(1 for i in CENSUS_IDS if j[i] and not KOK["forced"][i])
    if sum(j.values()) == 1624 or (x, y) == (218, 77):
        reach.append({"margin": m, "jev_correct": sum(j.values()), "discordant": [x, y]})
out["margin_sweep"] = {
    "what": "every margin in the one-parameter policy family, swept over all breakpoints of "
            "p[abstain], on the 2,510 census panel_choice records, against 27B forced",
    "n_margins_tested": len(margins),
    "margins_reaching_Jev_1624_or_discordance_218_77": reach,
    "conclusion": "no margin reaches Jev = 1624/2510 or the discordance (218, 77); those values are "
                  "Jev's served `answer` field, untouched by the policy",
}

with open(OUT, "w") as fh:
    json.dump(out, fh, indent=1)

# --------------------------------------------------------------------- print
for pname, pv in out["populations"].items():
    if pname == "census_2510_per_atlas":
        print("\n[per atlas, census 2,510]")
        for d, v in pv.items():
            f, s = v["27B forced vs Jev forced"], v["27B forced vs Jev as served"]
            print(f"  {d:20s} n={v['n']:4d}  forced/forced {f['a_right_b_wrong']:4d} vs "
                  f"{f['b_right_a_wrong']:4d} p={f['p']:.3g}   |  forced/served "
                  f"{s['a_right_b_wrong']:4d} vs {s['b_right_a_wrong']:4d} p={s['p']:.3g}")
        continue
    print(f"\n[{pname}]  n={pv['n']}")
    for c in pv["pairs"]:
        print(f"  {c['comparison']:46s} A={c['a_correct']:4d}/{c['n']} ({c['a_acc']:.4f})  "
              f"B={c['b_correct']:4d}/{c['n']} ({c['b_acc']:.4f})  A-only={c['a_right_b_wrong']:4d}  "
              f"B-only={c['b_right_a_wrong']:4d}  p={c['p']:.4g}")

print("\n[split]")
for h in ("odd", "even"):
    v = out["split"][h]
    print(f"  {h:4s} n={v['n']:4d}  27B forced {v['27B_forced_correct']:4d}/{v['n']} "
          f"= {v['27B_forced_acc']:.4f}   Jev served {v['Jev_served_correct']:4d}/{v['n']} "
          f"= {v['Jev_served_acc']:.4f}")

print("\n[cells]")
for k, v in cells.items():
    print(f"  {k:26s} {v['correct']:5d}/{v['n']:5d} = {v['accuracy']:.6f}   "
          f"abstain-pick {v['abstain_pick']:.6f}")

print("\n[seed values, census-matched 1,819]")
for k, v in seed_values.items():
    print(f"  {k:16s} {v['correct']:4d}/{v['n']} = {v['accuracy']:.4f}")
print(f"  mean {out['seed_values']['measured_mean']:.4f}   "
      f"spread {out['seed_values']['measured_spread_max_over_min']:.3f}x")

print("\n[margin sweep]")
print(f"  margins tested: {len(margins)} (every breakpoint of p[abstain])")
print(f"  margins reaching Jev = 1624/2510 or discordance (218, 77): "
      f"{reach if reach else 'NONE'}")
print(f"\nwrote {os.path.relpath(OUT, ROOT)}")
