#!/usr/bin/env python3
"""Recompute axis 1 from a scored row dump — and check the new definition against the old numbers.

`reports/eval-real-v2-eval.json` carries no per-row scores, so the step-12 diagnosis could not be
recomputed from it and had to dump rows to the cluster (`reports/axis1-rows-{transfer,pair,bypass}
.jsonl`, 5,439 + 3,395 + 1,824 rows). This script reads those dumps, runs the **library** metric
over them, and prints:

* the stratified axis-1 report the metric now produces (primary statistic, per-family table, the
  two serving-convention arms side by side);
* the seven numbers quoted in `reports/step12.md`, recomputed through the same code path and marked
  `ok` or `DIFFERS`. A `DIFFERS` is a finding about the new code — it is printed, not reconciled.

The dumps predate the metric's provenance fields, so :func:`as_served` reconstructs what the metric
reads from what they do record, and the reconstruction is checked rather than trusted:

    uv run python scripts/axis1-recompute.py                       # every dump present
    uv run python scripts/axis1-recompute.py --dump transfer       # one of them
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sckevtype.eval import metrics as M

DUMP_DIR = Path("reports")
#: The three step-12 probes, plus ``eval`` — the dump the re-scoring pass writes, which needs no
#: reconstruction because the CLI records the provenance and the index itself — plus ``jevfull``,
#: the Jev batched census (`reports/jev-full/axis1-rows.jsonl`, copied here under that name), which
#: is the same reader on a different model and carries the same provenance fields, plus ``armb``
#: (arm B's scoring pass, ``reports/axis1-rows-armb.jsonl``), which is the same reader again, plus
#: ``seed1``/``seed2``/``seed3`` — the seed-variance draws (``reports/axis1-rows-seed<S>.jsonl``,
#: written by <cluster-job>), which are the same reader on the frozen
#: configuration at a different seed and carry the same provenance fields as ``eval``.
DUMPS = ("transfer", "pair", "bypass", "eval", "jevfull", "kev-0.5b", "kev-0.6b", "kev-0.8b", "kev-4b", "kev-9b",
         "armb", "seed1", "seed2", "seed3")

#: The numbers `reports/step12.md` reports, each with the selection that produced it and the dump
#: it was measured on. ``expected`` is at the precision step 12 quotes, which is 4 decimals.
CLAIMS: tuple[tuple[str, str, str, float], ...] = (
    ("transfer", "hao_cite, pooled over families and conventions", "dataset == hao_cite", 0.8174),
    ("pair", "pooled over every row of the dump", "all", 0.6248),
    ("pair", "detection@FPR0.05, pooled", "all_detection", 0.1301),
    ("pair", "factory-given rows only, pooled", "factory", 0.6599),
    ("pair", "detection@FPR0.05, factory-given rows only", "factory_detection", 0.2131),
    ("pair", "mean P(abstain), injected rows of panel_choice", "injected_choice", 0.9772),
    ("bypass", "pooled, lora_scale=0.0", "all", 0.4908),
)


def load(path: Path) -> list[dict]:
    with open(path) as handle:
        return [json.loads(line) for line in handle]


def as_served(row: dict) -> dict:
    """A dump row as a row :func:`sckevtype.eval.metrics.panel_adequacy` reads.

    A dump written by the CLI since the axis-1 change carries everything the metric reads and is
    passed through untouched. A **step-12** dump is missing two fields. ``abstain_source`` is
    recovered exactly: the probe called ``eval.run.with_abstain``, so ``factory_abstain`` false
    *is* the injected convention. The abstain option's **index** is not in the dump. ``size``
    counts the options of the question *as the factory asked it* — the probe measured the dump's
    ``positive`` against the factory's keys — so an injected option sits at index ``size`` of the
    served question, one past every index the row's label can hold. For a factory-given negative
    the position is unrecorded and irrelevant to the label: any index other than the row's own
    answer reads as a negative, so it is reconstructed the same way. The dump's ``positive`` column
    is therefore a *check* on the reconstruction, not an input to it (see :func:`verify`).
    """
    if "abstain_source" in row and "abstain" in row:
        return row
    size, label = int(row["size"]), int(row["label"])
    factory = bool(row["factory_abstain"])
    return {
        **row,
        "abstain": label if (factory and row["positive"]) else size,
        "abstain_source": M.FACTORY if factory else M.INJECTED,
    }


def verify(raw: list[dict], served: list[dict]) -> list[str]:
    """The two things the reconstruction can get wrong, checked against the dump's own columns."""
    problems = []
    for before, after in zip(raw, served):
        if M.axis1_label(after) != bool(before["positive"]):
            problems.append(f"    row {before['id']}: label check disagrees with the dump's `positive`")
        # A question the factory left without an abstain option cannot have that option as its
        # answer, so an injected row is a negative by construction. A dump row that violates this
        # means the injection, the projection or this script is wrong — not that the model acted up.
        if after["abstain_source"] == M.INJECTED and before["positive"]:
            problems.append(f"    row {before['id']}: injected row claims a positive")
    return problems


def value(rows: list[dict], selection: str) -> tuple[float, int]:
    """The claimed quantity, as ``(score, n)``. Every branch goes through the library metric."""
    if selection == "injected_choice":
        group = [row for row in rows
                 if row["family"] == "panel_choice" and row["abstain_source"] == M.INJECTED]
        if not group:
            raise ValueError("no injected panel_choice rows in this dump")
        return sum(float(row["none_p"]) for row in group) / len(group), len(group)
    if selection == "dataset == hao_cite":
        rows = [row for row in rows if row["dataset"] == "hao_cite"]
    adequacy = M.panel_adequacy(rows)
    if selection.startswith("factory"):
        arm = adequacy["by_serving_convention"][M.FACTORY]
        return arm["detection_rate" if selection.endswith("_detection") else "auroc"], arm["n"]
    return adequacy["detection_rate" if selection == "all_detection" else "auroc"], adequacy["n"]


def show(name: str, rows: list[dict]) -> None:
    print(f"\n{'=' * 96}\n{name}: {len(rows)} rows\n{'=' * 96}")
    adequacy = M.panel_adequacy(rows)
    primary = adequacy["primary"]
    if primary is None:
        print(f"  PRIMARY   : n/a — {adequacy['primary_note']}")
    else:
        print(f"  PRIMARY   : auroc={primary['auroc']:.4f}  n={primary['n']}  pos={primary['n_positive']}  "
              f"neg={primary['n_negative']}  detection@FPR0.05={primary['detection_rate']:.4f}")
        print(f"              families {', '.join(primary['families'])}"
              + (f"; excluded one-class families: {', '.join(primary['excluded_families'])}"
                 if primary["excluded_families"] else ""))
    print(f"  POOLED    : n={adequacy['n']}  pos={adequacy['n_positive']}  neg={adequacy['n_negative']}  "
          f"auroc={adequacy['auroc']:.4f}")
    print(f"              {adequacy['pooled_qualifier']}")
    for arm in (M.FACTORY, M.INJECTED):
        block = adequacy["by_serving_convention"][arm]
        if block["n"] == 0:
            print(f"  {arm.upper():<10}: no rows")
        else:
            print(f"  {arm.upper():<10}: n={block['n']:5d}  pos={block['n_positive']:5d}  auroc="
                  f"{block['auroc'] if block['auroc'] is None else round(block['auroc'], 4)}  "
                  f"mean P(abstain)={block['mean_score']:.4f}")
    print("  PER FAMILY:")
    for family, block in adequacy["per_family"].items():
        if block["auroc"] is None:
            print(f"    {family:20s} n={block['n']:5d}  {block['note']}")
        else:
            print(f"    {family:20s} n={block['n']:5d}  pos={block['n_positive']:5d}  neg={block['n_negative']:5d}"
                  f"  auroc={block['auroc']:.4f}" + ("  THIN" if block["thin"] else ""))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dump", action="append", choices=DUMPS, help="default: every dump that is present")
    args = parser.parse_args()

    dumps = args.dump or list(DUMPS)
    loaded: dict[str, list[dict]] = {}
    disagreements = 0
    for name in dumps:
        path = DUMP_DIR / f"axis1-rows-{name}.jsonl"
        if not path.exists():
            print(f"no {path} — skipping")
            continue
        raw = load(path)
        served = [as_served(row) for row in raw]
        problems = verify(raw, served)
        loaded[name] = served
        print(f"{path}: {len(raw)} rows, {len(problems)} reconstruction problem(s)"
              + ("".join(["\n" + line for line in problems]) if problems else ""))
        show(f"{name} ({path.name})", served)

    print(f"\n{'=' * 96}\nSTEP-12 NUMBERS, RECOMPUTED THROUGH THE NEW CODE PATH\n{'=' * 96}")
    print(f"  {'dump':9s} {'claim':52s} {'n':>6s} {'got':>9s} {'step 12':>9s}")
    for name, claim, selection, expected in CLAIMS:
        if name not in loaded:
            continue
        got, n = value(loaded[name], selection)
        verdict = "ok" if round(got, 4) == expected else "DIFFERS"
        disagreements += verdict == "DIFFERS"
        print(f"  {name:9s} {claim:52s} {n:6d} {got:9.4f} {expected:9.4f}  {verdict}")
    if disagreements:
        print(f"\n  {disagreements} disagreement(s). Each is a finding about the new code — the number it "
              f"produces is printed above, not adjusted to match.")
    return 1 if disagreements else 0


if __name__ == "__main__":
    raise SystemExit(main())
