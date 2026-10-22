#!/usr/bin/env python3
"""Draw the **v4 census population** as a probe sample — the population the Jev ask is taken on.

Why this exists instead of ``sckevtype.jev.probe sample``. That command's population is defined by an
*eligibility filter* (a cell qualifies when it carries all four served families), which is the right
rule for a within-cell probe and the wrong one for the v4 census: v4's test partition holds 4,035
``panel_choice`` cells and only 1,043 ``panel_miss`` records, so an all-four filter would drop 2,992 of
the census's own ``panel_choice`` cells. The v4 census population is already written down —
``reports/census-suite-ids-v4.jsonl``: one row per ``(dataset, cell, family)`` triple of the four census
families in the test split, and exactly what every v4 arm is scored on. This script turns that id list
into a ``sample.json`` in the probe's own schema, so ``ask``/``ask-batched``/``evaluate`` read it
unchanged.

The two artifacts are cross-checked rather than assumed to agree, in both directions:

* every id of the id list must be the id the partition's own line index gives that record
  (``<split>/custom/<0-based line>``, the rule ``<cluster-job>`` derives and
  which ``reports/census-suite-ids-v4.jsonl`` already carries) — a mismatch means the two are about
  different records and the run refuses;
* every triple the id list names must be present in the partition, and the count of records differenced
  must be zero — otherwise the population would silently shrink, which is the one failure a probe run
  cannot notice afterwards.

    python scripts/jev/census-sample-v4.py \\
        --ids reports/census-suite-ids-v4.jsonl \\
        --partition artifacts/corpus/real-v4/test.jsonl \\
        --out reports/jev-v4

``--limit-cells N`` writes a sample of the first N cells (a pilot: a handful of cells through the real
``ask`` CLI, to price the population before paying for it). Its output must never be read as the census.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:                      # the package lives in the repo, not in the venv
    sys.path.insert(0, str(ROOT))

from sckevtype.contracts.freeze import file_digest, write_json

FAMILIES = ("panel_choice", "panel_miss", "evidence_diluted", "marker_noul")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ids", required=True, help="the census id list (reports/census-suite-ids-v4.jsonl)")
    ap.add_argument("--partition", required=True, help="the corpus partition the population is drawn from")
    ap.add_argument("--out", required=True, help="directory to write sample.json into")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit-cells", type=int, default=0, help="write only the first N cells (a pilot)")
    ap.add_argument("--seed", type=int, default=0,
                    help="recorded for provenance; this draw is not random (the population is an id list)")
    args = ap.parse_args()

    ids_path, partition = Path(args.ids), Path(args.partition)
    for path in (ids_path, partition):
        if not path.exists():
            raise SystemExit(f"{path} does not exist")
    rows = read_jsonl(ids_path)
    if not rows:
        raise SystemExit(f"{ids_path}: no ids -- refusing to write an empty population")

    wanted_families = {row["family"] for row in rows}
    unknown = sorted(wanted_families - set(FAMILIES))
    if unknown:
        raise SystemExit(f"{ids_path}: family/families {unknown} are outside the probe's served set "
                         f"{list(FAMILIES)}; the ask path has no request builder for them")

    # The census's own triples, and the id it says each one carries.
    expected: dict[tuple[str, str, str], str] = {}
    for row in rows:
        key = (row["dataset"], row["cell"], row["family"])
        if key in expected:
            raise SystemExit(f"{ids_path}: two rows name {key} -- the population would be counted twice")
        expected[key] = row["id"]

    # One pass over the partition: line number -> (dataset, cell, family), for the families asked.
    index: dict[int, tuple[str, str, str]] = {}
    id_mismatch: list[str] = []
    per_family_partition: Counter = Counter()
    with partition.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle):
            if not line.strip():
                continue
            meta = json.loads(line)["_meta"]
            family = meta["family"]
            if family not in wanted_families:
                continue
            key = (str(meta["dataset_id"]), str(meta["cell_id"]), family)
            index[number] = key
            per_family_partition[family] += 1
            if key in expected:
                given = f"{args.split}/custom/{number}"
                if expected[key] != given:
                    id_mismatch.append(f"{key}: id list says {expected[key]}, line {number} is {given}")

    by_key = {key: number for number, key in index.items()}
    absent = sorted(key for key in expected if key not in by_key)
    if id_mismatch:
        print(f"REFUSING: {len(id_mismatch)} id(s) disagree with the partition's own line index, e.g.")
        for line in id_mismatch[:5]:
            print(f"  {line}")
        return 2
    if absent:
        print(f"REFUSING: {len(absent)} of the id list's {len(expected)} triples are not in {partition}, e.g.")
        for key in absent[:5]:
            print(f"  {key}")
        return 3

    records: dict[str, dict[str, int]] = {}
    for (dataset, cell, family), _id in sorted(expected.items()):
        records.setdefault(f"{dataset}/{cell}", {})[family] = by_key[(dataset, cell, family)]
    chosen = sorted(records)
    if args.limit_cells:
        chosen = chosen[: args.limit_cells]
        records = {group: records[group] for group in chosen}

    families_per_cell = Counter(len(records[group]) for group in chosen)
    counts = Counter(group.split("/", 1)[0] for group in chosen)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sample = {
        "probe": "sckevtype.jev", "step": "sample", "corpus": str(partition),
        "corpus_sha256": file_digest(partition), "split": args.split,
        "families": sorted(wanted_families), "seed": args.seed, "cells_per_dataset": None,
        "cells": len(chosen), "records_total": sum(len(records[group]) for group in chosen),
        "why_these_cells": (
            f"the census population: every (dataset, cell, family) triple of {ids_path} that the "
            f"{args.split} partition carries, one record each. Every v4 arm is scored on exactly this "
            f"population (reports/census-suite-ids-v4.jsonl), so the ask and the arms describe the same "
            f"questions. Not drawn: the id list IS the population"),
        "population_label": (
            f"the v4 census population ({len(chosen)} cells, {sum(len(records[g]) for g in chosen)} "
            f"records: the test-split records of the four census families, one per (dataset, cell, family) "
            f"triple of reports/census-suite-ids-v4.jsonl)"),
        "population_reason": (
            "the population is reports/census-suite-ids-v4.jsonl — a census of the four census families' "
            "test-split records, not a sample of the partition and not the all-four-families eligible set "
            "(the partition carries 4,035 panel_choice cells against 1,043 panel_miss records, so the two "
            "are not the same thing on v4)"),
        "indexed_cells": len({key[0] + "/" + key[1] for key in index.values()}),
        "eligible_cells": len(expected),
        "eligible_by_dataset": dict(sorted(Counter(key[0] for key in expected).items())),
        "by_dataset": dict(sorted(counts.items())),
        "families_per_cell": {str(k): v for k, v in sorted(families_per_cell.items())},
        "id_list": {"path": str(ids_path), "rows": len(rows), "sha256": file_digest(ids_path)},
        "chosen": chosen,
        "records": records,
    }
    write_json(out / "sample.json", sample)

    print(f"id list        : {ids_path}  {len(rows)} rows  {file_digest(ids_path)[:16]}")
    print(f"partition      : {partition}  {file_digest(partition)[:16]}")
    print(f"cells          : {len(chosen)}  {dict(sorted(counts.items()))}")
    print(f"records        : {sum(len(records[g]) for g in chosen)}  families {sorted(wanted_families)}")
    print(f"families/cell  : {dict(sorted(families_per_cell.items()))}")
    print(f"partition fams : {dict(sorted(per_family_partition.items()))}")
    print("id check       : 0 mismatches against <split>/custom/<line>; 0 triples absent from the partition")
    print(f"wrote {out / 'sample.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
