#!/usr/bin/env python3
"""Score the Kev-initialized fine-tunes (and their comparators) on the census-matched population.

This is the measurement behind ``reports/kev-init-finetunes.md``. It runs no model and submits no
job: it re-aggregates row dumps already on disk under the population and the rule
``reports/abstain-reconciliation.py`` settled — *that* script is the definition, and this one
imports nothing from it only because it lives beside it as a report rather than a module. The two
rules are copied, not re-derived:

* **the population** is the 1,819 ``panel_choice`` rows of ``reports/census-suite-ids.jsonl`` —
  the census-matched set every engine in ``reports/TABLE-census-matched.md`` is scored on. Each
  suite row carries ``(dataset, cell, family)``, so the census key of a suite row is
  ``test/<dataset>/<cell>/<family>``; 322 of the 1,497 distinct triples carry two suite rows and
  are counted twice, exactly as the reconciliation script counts them.
* **naming accuracy** is the fraction of those rows whose *served* prediction is the label — an
  abstention is scored as incorrect, which is the manuscript's Table 2 definition.
* **abstain-pick rate** is the fraction whose served prediction is the abstain option.
* **forced** (abstain removed) is ``abstain-reconciliation.py``'s ``forced()``: the argmax of the
  probability vector with the abstain option skipped. It is computable only where a full ``p``
  vector is on disk — the released rungs' dumps carry one, our arms' dumps do not (they carry
  ``top1``, ``top1_p`` and ``none_p``), so it is reported for the rungs and marked absent for the
  arms rather than approximated.

Our arms and the rungs are joined differently because their dumps are keyed differently, and that
is a property of the dumps rather than a choice made here: a suite dump is keyed by the suite row's
own id (``test/custom/<n>``), a rung dump by ``test/<dataset>/<cell>/<family>``.

Usage:  source scripts/wsl/env.sh && uv run --no-sync python scripts/kev-init-score.py
"""

from __future__ import annotations

import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "reports", "kev-init-finetunes.json")

#: name -> (dump, rule). ``ours`` reads the dump's own served ``top1``; ``rung`` reads the rung
#: dump's ``p`` vector through ``argmax`` (the rungs' dumps carry no ``top1`` — the reconciliation
#: script's §6.1 note is the same fact).
DUMPS: dict[str, tuple[str, str]] = {
    # the released rungs, zero-shot
    "kev-0.5b": ("reports/kev-rungs/05b-rows.jsonl", "rung"),
    "kev-0.6b": ("reports/kev-rungs/06b-rows.jsonl", "rung"),
    "kev-0.8b": ("reports/kev-rungs/08b-rows.jsonl", "rung"),
    "kev-4b": ("reports/kev-rungs/4b-rows.jsonl", "rung"),
    "kev-9b": ("reports/kev-rungs/9b-rows.jsonl", "rung"),
    # our arms, initialized from the BASE checkpoint
    "frozen": ("reports/rows-full-frozen.jsonl", "ours"),
    "armB": ("reports/rows-full-armb.jsonl", "ours"),
    "from-base-s1": ("reports/rows-full-seed1.jsonl", "ours"),
    "from-base-s2": ("reports/rows-full-seed2.jsonl", "ours"),
    "from-base-s3": ("reports/rows-full-seed3.jsonl", "ours"),
    # our arms, initialized from a RELEASED KEV CHECKPOINT (--init_from)
    "delta-kev-0.8b-s1": ("reports/rows-full-delta-kev-0.8b-s1.jsonl", "ours"),
    "delta-kev-0.8b-s2": ("reports/rows-full-delta-kev-0.8b-s2.jsonl", "ours"),
    "delta-kev-0.8b-s3": ("reports/rows-full-delta-kev-0.8b-s3.jsonl", "ours"),
    "delta-kev-0.8b-s4": ("reports/rows-full-delta-kev-0.8b-s4.jsonl", "ours"),
    "delta-kev-0.8b-s5": ("reports/rows-full-delta-kev-0.8b-s5.jsonl", "ours"),
    "delta-kev-4b-s1": ("reports/rows-full-delta-kev-4b-s1.jsonl", "ours"),
}

FIVE_SEEDS = [f"delta-kev-0.8b-s{i}" for i in range(1, 6)]
THREE_BASE_SEEDS = ["from-base-s1", "from-base-s2", "from-base-s3"]


def read_jsonl(path: str) -> list[dict]:
    out = []
    with open(os.path.join(ROOT, path), encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def argmax(xs) -> int:
    best = 0
    for i in range(1, len(xs)):
        if xs[i] > xs[best]:
            best = i
    return best


def forced(r: dict) -> int:
    """argmax over p with the abstain option removed (abstention not offered) — the settled rule."""
    a, p = r["abstain"], r["p"]
    best, bv = None, None
    for i in range(len(p)):
        if i == a:
            continue
        if bv is None or p[i] > bv:
            bv, best = p[i], i
    assert best is not None, "a choice row with no non-abstain option: the rule does not apply"
    return best


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(os.path.join(ROOT, path), "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- load
SUITE = read_jsonl("reports/census-suite-ids.jsonl")
PC = [r for r in SUITE if r["family"] == "panel_choice"]
assert len(PC) == 1819, f"the census-matched panel_choice population is {len(PC)}, not 1,819"
assert len({r["id"] for r in PC}) == 1819, "the suite ids are not unique"

#: suite row id -> the census key its (dataset, cell, family) forms
CENSUS_KEY = {r["id"]: f"test/{r['dataset']}/{r['cell']}/{r['family']}" for r in PC}
assert len(set(CENSUS_KEY.values())) == 1497, (
    f"{len(set(CENSUS_KEY.values()))} distinct triples, expected 1,497 (322 triples carry two rows)")

results: dict[str, dict] = {}
missing_report: dict[str, int] = {}
for name, (path, rule) in DUMPS.items():
    rows = read_jsonl(path)
    if rule == "rung":
        by_key = {r["id"]: r for r in rows}
        keys = [CENSUS_KEY[r["id"]] for r in PC]
    else:
        by_key = {r["id"]: r for r in rows}
        keys = [r["id"] for r in PC]
    absent = [k for k in keys if k not in by_key]
    missing_report[name] = len(absent)
    assert not absent, (
        f"{name}: {len(absent)} of the 1,819 census-matched rows are not in {path} "
        f"(e.g. {absent[:3]}) -- scoring it would report a number for a population it was not run on")

    ok = ab = 0
    fok = fn = 0
    for key in keys:
        r = by_key[key]
        pred = argmax(r["p"]) if rule == "rung" else r["top1"]
        if pred == r["label"]:
            ok += 1
        if pred == r["abstain"]:
            ab += 1
        if rule == "rung":
            fn += 1
            if forced(r) == r["label"]:
                fok += 1
    entry = {"n": len(keys), "correct": ok, "accuracy": ok / len(keys),
             "abstain_pick": ab / len(keys), "rule": rule,
             "dump": path, "dump_sha256": sha256_file(path)}
    if rule == "rung":
        entry["forced_correct"] = fok
        entry["forced_accuracy"] = fok / fn
    results[name] = entry


def mean_of(names: list[str], field: str) -> float:
    return sum(results[n][field] for n in names) / len(names)


# ------------------------------------------------------------- the forced column's calibration
# A `forced` number is only readable against chance: these panels average 17.5 options, so a uniform
# pick among the non-abstain options is the null the rung's 0.3178 has to clear. Computed from the
# 0.8B rung's own dump, which is where the `p` vectors are.
_rung_rows = read_jsonl(DUMPS["kev-0.8b"][0])
_rung_pc = {r["id"]: r for r in _rung_rows if r["family"] == "panel_choice"}
_rows = [_rung_pc[CENSUS_KEY[r["id"]]] for r in PC]
_sizes = [len(r["p"]) for r in _rows]
_tops = [max(p for i, p in enumerate(r["p"]) if i != r["abstain"]) for r in _rows]
calibration = {
    "why": "a forced (abstain-removed) accuracy is only interpretable against the chance rate of the "
           "same rule on the same panels",
    "options_per_row_mean": sum(_sizes) / len(_sizes),
    "options_per_row_min": min(_sizes),
    "options_per_row_max": max(_sizes),
    "chance_uniform_over_non_abstain": sum(1.0 / (s - 1) for s in _sizes) / len(_sizes),
    "kev_0.8b_forced_over_chance": results["kev-0.8b"]["forced_accuracy"] /
                                   (sum(1.0 / (s - 1) for s in _sizes) / len(_sizes)),
    "mean_top_non_abstain_p": sum(_tops) / len(_tops),
    "mean_p_abstain": sum(r["p"][r["abstain"]] for r in _rows) / len(_rows),
}


summary = {
    "what_this_is": "Scoring for reports/kev-init-finetunes.md. Reads only row dumps already on disk; "
                    "runs no model, submits no job, writes only reports/kev-init-finetunes.json.",
    "population": {
        "source": "reports/census-suite-ids.jsonl",
        "definition": "the 1,819 census-matched suite rows in family panel_choice; 1,497 distinct "
                      "(dataset, cell, family) triples, 322 of which carry two suite rows",
        "n": 1819,
        "census_key_rule": "test/<dataset>/<cell>/<family> (the rungs' dumps are keyed this way; "
                           "our arms' dumps are keyed by the suite row's own id)",
    },
    "rules": {
        "ours": "the dump's own served top1; abstain scored as incorrect",
        "rung": "argmax(p) over the rung dump's probability vector (the rungs' dumps carry no top1)",
        "forced": "argmax over p with the abstain option removed -- abstain-reconciliation.py's rule; "
                  "computable only where a full p vector is on disk, i.e. for the rungs",
    },
    "calibration": calibration,
    "arms": results,
    "groups": {
        "kev_0.8b_zero_shot": results["kev-0.8b"],
        "kev_4b_zero_shot": results["kev-4b"],
        "from_kev_0.8b_5_seeds": {
            "arms": FIVE_SEEDS,
            "naming_accuracy_mean": mean_of(FIVE_SEEDS, "accuracy"),
            "naming_accuracy_min": min(results[n]["accuracy"] for n in FIVE_SEEDS),
            "naming_accuracy_max": max(results[n]["accuracy"] for n in FIVE_SEEDS),
            "abstain_pick_mean": mean_of(FIVE_SEEDS, "abstain_pick"),
            "abstain_pick_min": min(results[n]["abstain_pick"] for n in FIVE_SEEDS),
            "abstain_pick_max": max(results[n]["abstain_pick"] for n in FIVE_SEEDS),
        },
        "from_base_3_seeds": {
            "arms": THREE_BASE_SEEDS,
            "naming_accuracy_mean": mean_of(THREE_BASE_SEEDS, "accuracy"),
            "naming_accuracy_min": min(results[n]["accuracy"] for n in THREE_BASE_SEEDS),
            "naming_accuracy_max": max(results[n]["accuracy"] for n in THREE_BASE_SEEDS),
            "abstain_pick_mean": mean_of(THREE_BASE_SEEDS, "abstain_pick"),
        },
    },
}
with open(OUT, "w", encoding="utf-8") as fh:
    json.dump(summary, fh, indent=1)

# --------------------------------------------------------------------------- print
print(f"\n[population] {len(PC)} census-matched panel_choice rows, "
      f"{len(set(CENSUS_KEY.values()))} distinct triples\n")
print(f"{'arm':22s} {'n':>5s} {'naming':>9s} {'abstain':>9s} {'forced':>9s}   dump")
for name, (path, rule) in DUMPS.items():
    e = results[name]
    forced_cell = f"{e['forced_accuracy']:.4f}" if rule == "rung" else "—"
    print(f"{name:22s} {e['n']:5d} {e['accuracy']:9.4f} {e['abstain_pick']:9.4f} {forced_cell:>9s}   {path}")

g = summary["groups"]
print("\n[the comparison the question asks for]")
print(f"  Kev-0.8B zero-shot            naming {g['kev_0.8b_zero_shot']['accuracy']:.4f}  "
      f"abstain-pick {g['kev_0.8b_zero_shot']['abstain_pick']:.4f}  "
      f"forced {g['kev_0.8b_zero_shot']['forced_accuracy']:.4f}")
print(f"  from Kev-0.8B, 5 seeds (mean) naming {g['from_kev_0.8b_5_seeds']['naming_accuracy_mean']:.4f}  "
      f"abstain-pick {g['from_kev_0.8b_5_seeds']['abstain_pick_mean']:.4f}")
print(f"  from the base, 3 seeds (mean) naming {g['from_base_3_seeds']['naming_accuracy_mean']:.4f}  "
      f"abstain-pick {g['from_base_3_seeds']['abstain_pick_mean']:.4f}")
print(f"\nwrote {os.path.relpath(OUT, ROOT)}")
sys.exit(0)
