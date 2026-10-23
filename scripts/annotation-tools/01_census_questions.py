"""Step 1 of the standard-annotator baseline: the census `panel_choice` question table.

WHAT THIS BUILDS
----------------
One row per census-matched `panel_choice` question -- by default the 1,819 questions named by
``reports/census-suite-ids.jsonl`` (``family == "panel_choice"``), which is the population the
manuscript's engines are compared on. The population is a **parameter**: ``--population``
points at any census id list of the same shape, and the v4 run passes
``reports/census-suite-ids-v4.jsonl`` (4,035 questions over six atlases).

For every question the table carries everything a *cell-level* annotator has to be scored
against:

* ``target``  -- the option name the question's labelled answer points at (the corpus's own
  ``label`` index into its own ``keys``), i.e. the cell type the question is looking for;
* ``options`` -- the candidate panel, as the corpus rendered it;
* ``cl_id``   -- the cell's own register (CL) id, from the record's ``_meta``;
* ``target_cl_id`` -- the CL id the target option names, resolved through the corpus's own
  register (``artifacts/v3/definitions.json``), which is what lets a tool's CL-space label be
  compared with the question's answer without any string heuristics.

WHERE THE QUESTION COMES FROM
-----------------------------
Two sources, cross-checked against each other:

* the census dump (``reports/jev-full/rows.jsonl``; ``reports/jev-v4/rows.jsonl`` for the v4
  population) is the authoritative rendering of each question: it carries ``keys`` (the
  panel), ``label`` (the index of the correct option) and ``answer`` (that option's name)
  *for the variant the census asked*;
* the record the population's own id names. For the v2 population that is a capped suite
  (``artifacts/suites/real-v2/test.jsonl``, ids ``test/custom/<n>`` at arbitrary lines, read
  through a byte-offset index); for the v4 population it is the corpus test partition itself
  (``artifacts/corpus/real-v4/test.jsonl``), whose line *n* is ``test/custom/<n>`` and which
  carries no ``id`` field at all. The layout is detected from the first line and can be
  forced with ``--suite-layout``; in the corpus layout the record's ``_meta`` triple is
  asserted equal to the population row's, so a wrong detection cannot pass silently.

For 910 of the 1,819 v2 rows the suite holds the *other* rendering variant of the same
(cell, family) -- ``census-suite-ids.jsonl`` marks those ``variant_mismatch``. The variant is a
property of the rendered state, not of the question, so this script **checks** that the two
renditions agree on the panel and on the answer rather than assuming it; a disagreement is
reported and the row is dropped, never silently scored.

OUTPUT
------
``<dir>/questions.jsonl`` -- one JSON object per question:

    {"id", "dataset", "cell", "variant", "variant_mismatch", "panel_id",
     "target", "target_cl_id", "options": [...], "n_options", "abstain_option",
     "cl_id", "target_is_cells_own_type": bool}

plus ``<dir>/questions.summary.json`` with the counts and every check.

Run (WSL, from the repo root):

    source scripts/wsl/env.sh
    uv run --no-sync python scripts/annotation-tools/01_census_questions.py

    # the v4 population, into its own working directory
    uv run --no-sync python scripts/annotation-tools/01_census_questions.py \
        --population reports/census-suite-ids-v4.jsonl \
        --census-rows reports/jev-v4/rows.jsonl \
        --suite artifacts/corpus/real-v4/test.jsonl \
        --dir reports/annotation-tools-v4
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(os.environ.get("SCKEVTYPE_ROOT", "/mnt/f/E/hermes-workspace/scKevType"))
DEFAULT_POPULATION = ROOT / "reports" / "census-suite-ids.jsonl"
DEFAULT_CENSUS_ROWS = ROOT / "reports" / "jev-full" / "rows.jsonl"
DEFAULT_SUITE = ROOT / "artifacts" / "suites" / "real-v2" / "test.jsonl"
DEFAULT_DEFINITIONS = ROOT / "artifacts" / "v3" / "definitions.json"
DEFAULT_DIR = ROOT / "reports" / "annotation-tools"

SUITE_ID_MARKER = b'"id": "test/custom/'


def suite_offsets(path: Path) -> dict[str, int]:
    """id -> byte offset of its line, so the 745 MB suite can be read by id."""
    offsets: dict[str, int] = {}
    with open(path, "rb") as fh:
        pos = 0
        while True:
            line = fh.readline()
            if not line:
                break
            i = line.find(SUITE_ID_MARKER)
            if i >= 0:
                j = line.find(b'"', i + 7)
                offsets[line[i + 7 : j].decode()] = pos
            pos += len(line)
    if not offsets:
        raise SystemExit(f"{path}: no suite ids found -- refusing an empty index")
    return offsets


def read_suite_row(path: Path, offsets: dict[str, int], rid: str) -> dict:
    with open(path, "rb") as fh:
        fh.seek(offsets[rid])
        return json.loads(fh.readline().decode())


def detect_layout(path: Path) -> str:
    """``suite`` when the records carry an ``id`` field, ``corpus`` when the line number is the id."""
    with open(path, "rb") as fh:
        first = fh.readline()
    if not first:
        raise SystemExit(f"{path}: empty -- nothing to read a question out of")
    return "suite" if SUITE_ID_MARKER in first else "corpus"


def read_records(path: Path, layout: str, ids: list[str]) -> dict[str, dict]:
    """id -> record, for exactly the ids asked for. A missing id is a hard error."""
    if layout == "suite":
        offsets = suite_offsets(path)
        missing = [rid for rid in ids if rid not in offsets]
        if missing:
            raise SystemExit(f"{path}: {len(missing)} population ids are not in the suite, e.g. {missing[:5]}")
        return {rid: read_suite_row(path, offsets, rid) for rid in ids}
    if layout != "corpus":
        raise SystemExit(f"unknown layout {layout!r}")
    # corpus layout: the id is `test/custom/<0-based line index>` of the partition itself
    want: dict[int, list[str]] = {}
    for rid in ids:
        head, _, tail = rid.rpartition("/")
        if head != "test/custom" or not tail.isdigit():
            raise SystemExit(f"{path}: id {rid!r} is not of the form test/custom/<line index>")
        want.setdefault(int(tail), []).append(rid)
    records: dict[str, dict] = {}
    with open(path, encoding="utf-8") as fh:
        for n, line in enumerate(fh):
            if n not in want:
                continue
            if not line.strip():
                raise SystemExit(f"{path}: line {n} is blank but a population id names it")
            record = json.loads(line)
            for rid in want[n]:
                records[rid] = record
    missing = [rid for rid in ids if rid not in records]
    if missing:
        raise SystemExit(
            f"{path}: {len(missing)} population ids name a line past the end of the partition, "
            f"e.g. {missing[:5]} -- is this the partition the population was taken from?"
        )
    return records


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def show(path: Path) -> str:
    """The path as the reports quote it: relative to the repo root when it is inside it."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the census panel_choice question table.")
    parser.add_argument("--population", default=str(DEFAULT_POPULATION),
                        help="the census id list (JSONL with id/dataset/cell/family), filtered to "
                             "family == panel_choice")
    parser.add_argument("--census-rows", default=str(DEFAULT_CENSUS_ROWS),
                        help="the census row dump the population was joined against")
    parser.add_argument("--suite", default=str(DEFAULT_SUITE),
                        help="the partition the population's ids index: a capped suite (each record "
                             "carries its id) or a corpus test partition (line n is test/custom/n)")
    parser.add_argument("--suite-layout", choices=("auto", "suite", "corpus"), default="auto",
                        help="how --suite maps an id to a record; auto detects it from the first line")
    parser.add_argument("--definitions", default=str(DEFAULT_DEFINITIONS),
                        help="the register the option names resolve through")
    parser.add_argument("--dir", default=str(DEFAULT_DIR), help="where questions.jsonl is written")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    population_path = Path(args.population)
    census_rows_path = Path(args.census_rows)
    suite_path = Path(args.suite)
    definitions_path = Path(args.definitions)
    out_dir = Path(args.dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in (population_path, census_rows_path, suite_path, definitions_path):
        if not path.exists():
            raise SystemExit(f"{path}: not found")

    layout = detect_layout(suite_path) if args.suite_layout == "auto" else args.suite_layout
    print(f"population                         : {show(population_path)}")
    print(f"suite partition                    : {show(suite_path)} (layout {layout})")

    # --- the population: the census ids, panel_choice only -------------------------------
    wanted: dict[str, dict] = {}
    with open(population_path, encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if row["family"] == "panel_choice":
                wanted[row["id"]] = row
    if not wanted:
        raise SystemExit(f"{population_path}: no panel_choice ids -- refusing to score nothing")
    print(f"census panel_choice ids            : {len(wanted):,}")

    # --- the question as the census asked it ---------------------------------------------
    # keyed by (dataset, cell, family, variant) exactly as census-suite-ids.py keyed its join.
    census_rows: dict[tuple[str, str, str, str], dict] = {}
    with open(census_rows_path, encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if row["family"] != "panel_choice":
                continue
            census_rows[(row["dataset"], row["cell"], row["family"], row["variant"])] = row
    print(f"census panel_choice rows on disk   : {len(census_rows):,}")

    # --- the register, so an option name resolves to a CL id ------------------------------
    defs = json.loads(definitions_path.read_text(encoding="utf-8"))["definitions"]
    name_to_cl: dict[str, str] = {}
    for d in defs:
        name_to_cl.setdefault(d["name"], d["cl_id"])
    cl_to_name = {d["cl_id"]: d["name"] for d in defs}
    print(f"register terms                     : {len(defs):,}")

    records = read_records(suite_path, layout, sorted(wanted))
    print(f"records read                       : {len(records):,}")

    out_rows: list[dict] = []
    checks: Counter = Counter()
    mismatch_examples: list[dict] = []
    per_dataset = Counter()

    for rid, cid in sorted(wanted.items(), key=lambda kv: kv[0]):
        # The record the population's own id names. In the corpus layout the id is a line
        # index, so this triple is the only thing tying the record to the population row; it
        # is asserted here, and a mismatch is a hard error rather than a dropped row.
        srow = records[rid]
        meta = srow["_meta"]
        triple_ok = (meta.get("dataset_id") == cid["dataset"] and meta.get("cell_id") == cid["cell"]
                     and meta.get("family") == "panel_choice")
        if not triple_ok:
            checks["record_triple_disagrees_with_population"] += 1
            mismatch_examples.append(
                {"id": rid, "why": "the record at this id is not the cell/family the population row names",
                 "population": [cid["dataset"], cid["cell"], "panel_choice"],
                 "record": [meta.get("dataset_id"), meta.get("cell_id"), meta.get("family")]}
            )
            continue
        checks["record_triple_matches_population"] += 1

        # the census's own rendition
        ckey = (cid["dataset"], cid["cell"], "panel_choice", cid.get("variant"))
        crow = census_rows.get(ckey)
        if crow is None:
            # the census key has a variant the dump does not carry under that name: the
            # variant_mismatch rows were written for exactly this case, so fall back to the
            # (cell, family) pair and report it rather than dropping the question.
            alts = [v for k, v in census_rows.items() if k[:3] == ckey[:3]]  # type: ignore[arg-type]
            if len(alts) != 1:
                checks["census_row_not_found"] += 1
                mismatch_examples.append({"id": rid, "why": "no census row", "key": ckey})
                continue
            crow = alts[0]
            checks["census_row_found_by_cell_family"] += 1

        slab = srow["questions"]["label"]

        census_keys = list(crow["keys"])
        suite_keys = list(slab["criteria"].keys())
        # The census dump's ``answer`` is the *engine's* answer, not the key: the ground truth
        # is ``keys[label]`` (``label`` is the index of the correct option). Reading ``answer``
        # as the truth silently scores 618 of these questions against the engine's own choice.
        if crow.get("label") is None or not (0 <= int(crow["label"]) < len(census_keys)):
            checks["census_label_index_out_of_range"] += 1
            mismatch_examples.append({"id": rid, "why": "census label index out of range",
                                      "label": crow.get("label"), "n_keys": len(census_keys)})
            continue
        census_answer = census_keys[int(crow["label"])]
        suite_answer = slab["label"]

        if set(census_keys) != set(suite_keys):
            checks["panel_set_disagrees"] += 1
            mismatch_examples.append(
                {"id": rid, "why": "panel option set differs",
                 "only_census": sorted(set(census_keys) - set(suite_keys))[:5],
                 "only_suite": sorted(set(suite_keys) - set(census_keys))[:5]}
            )
            continue
        if len(census_keys) != len(suite_keys):
            checks["panel_size_disagrees"] += 1
        if census_answer != suite_answer:
            checks["answer_disagrees"] += 1
            mismatch_examples.append(
                {"id": rid, "why": "target differs", "census": census_answer, "suite": suite_answer}
            )
            continue
        checks["question_agrees_across_variants"] += 1

        target = suite_answer
        target_cl = name_to_cl.get(target)
        cl_id = meta.get("cl_id")
        abstain = crow.get("abstain_key")
        own_type = bool(target_cl and cl_id and target_cl == cl_id)
        if not own_type:
            checks["target_is_not_cells_own_register_type"] += 1
        if target_cl is None:
            checks["target_has_no_register_term"] += 1

        out_rows.append(
            {
                "id": rid,
                "dataset": cid["dataset"],
                "cell": cid["cell"],
                "variant": cid.get("variant"),
                "variant_mismatch": bool(cid.get("variant_mismatch")),
                "suite_variant": meta.get("variant"),
                "panel_id": meta.get("panel_id"),
                "panel_scope": meta.get("panel_scope"),
                "target": target,
                "target_cl_id": target_cl,
                "target_cl_name": cl_to_name.get(target_cl) if target_cl else None,
                "options": census_keys,
                "n_options": len(census_keys),
                "abstain_option": abstain,
                "cl_id": cl_id,
                "target_is_cells_own_type": own_type,
            }
        )
        per_dataset[cid["dataset"]] += 1

    if not out_rows:
        raise SystemExit("no question rows survived the checks -- refusing to write an empty table")

    with open(out_dir / "questions.jsonl", "w", encoding="utf-8") as fh:
        for row in out_rows:
            fh.write(json.dumps(row) + "\n")

    summary = {
        "population": f"{show(population_path)}, family == panel_choice",
        "population_sha256": sha256_file(population_path),
        "questions_written": len(out_rows),
        "questions_per_dataset": dict(sorted(per_dataset.items())),
        "distinct_cells": len({(r["dataset"], r["cell"]) for r in out_rows}),
        "variant_mismatch_rows": sum(1 for r in out_rows if r["variant_mismatch"]),
        "checks": dict(checks),
        "mismatch_examples": mismatch_examples[:10],
        "inputs": {
            "population": show(population_path),
            "census_rows": show(census_rows_path),
            "suite": show(suite_path),
            "suite_layout": layout,
            "definitions": show(definitions_path),
        },
    }
    (out_dir / "questions.summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "mismatch_examples"}, indent=2))
    print(f"wrote {out_dir / 'questions.jsonl'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
