"""Step 1 of the standard-annotator baseline: the census `panel_choice` question table.

WHAT THIS BUILDS
----------------
One row per census-matched `panel_choice` question -- the 1,819 questions named by
``reports/census-suite-ids.jsonl`` (``family == "panel_choice"``), which is the population the
manuscript's engines are compared on.

For every question the table carries everything a *cell-level* annotator has to be scored
against:

* ``target``  -- the option name the question's labelled answer points at (the corpus's own
  ``label`` index into its own ``keys``), i.e. the cell type the question is looking for;
* ``options`` -- the candidate panel, as the corpus rendered it;
* ``cl_id``   -- the cell's own register (CL) id, from the suite record's ``_meta``;
* ``target_cl_id`` -- the CL id the target option names, resolved through the corpus's own
  register (``artifacts/v3/definitions.json``), which is what lets a tool's CL-space label be
  compared with the question's answer without any string heuristics.

WHERE THE QUESTION COMES FROM
-----------------------------
The census dump (``reports/jev-full/rows.jsonl``) is the authoritative rendering of each
question: it carries ``keys`` (the panel), ``label`` (the index of the correct option) and
``answer`` (that option's name) *for the variant the census asked*. The capped suite
(``artifacts/suites/real-v2/test.jsonl``) carries the same question under a ``test/custom/<n>``
id, and for 910 of the 1,819 rows the suite holds the *other* rendering variant of the same
(cell, family) -- ``census-suite-ids.jsonl`` marks those ``variant_mismatch``. The variant is a
property of the rendered state, not of the question, so this script **checks** that the two
renditions agree on the panel and on the answer rather than assuming it; a disagreement is
reported and the row is dropped, never silently scored.

OUTPUT
------
``reports/annotation-tools/questions.jsonl`` -- one JSON object per question:

    {"id", "dataset", "cell", "variant", "variant_mismatch", "panel_id",
     "target", "target_cl_id", "options": [...], "n_options", "abstain_option",
     "cl_id", "target_is_cells_own_type": bool}

plus ``reports/annotation-tools/questions.summary.json`` with the counts and every check.

Run (WSL, from the repo root):

    source scripts/wsl/env.sh
    uv run --no-sync python scripts/annotation-tools/01_census_questions.py
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(os.environ.get("SCKEVTYPE_ROOT", "/mnt/f/E/hermes-workspace/scKevType"))
CENSUS_IDS = ROOT / "reports" / "census-suite-ids.jsonl"
CENSUS_ROWS = ROOT / "reports" / "jev-full" / "rows.jsonl"
SUITE_TEST = ROOT / "artifacts" / "suites" / "real-v2" / "test.jsonl"
DEFINITIONS = ROOT / "artifacts" / "v3" / "definitions.json"
OUT_DIR = ROOT / "reports" / "annotation-tools"


def suite_offsets(path: Path) -> dict[str, int]:
    """id -> byte offset of its line, so the 745 MB suite can be read by id."""
    offsets: dict[str, int] = {}
    with open(path, "rb") as fh:
        pos = 0
        while True:
            line = fh.readline()
            if not line:
                break
            i = line.find(b'"id": "test/custom/')
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


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- the population: the census ids, panel_choice only -------------------------------
    wanted: dict[str, dict] = {}
    with open(CENSUS_IDS, encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if row["family"] == "panel_choice":
                wanted[row["id"]] = row
    if not wanted:
        raise SystemExit(f"{CENSUS_IDS}: no panel_choice ids -- refusing to score nothing")
    print(f"census panel_choice ids            : {len(wanted):,}")

    # --- the question as the census asked it ---------------------------------------------
    # keyed by (dataset, cell, family, variant) exactly as census-suite-ids.py keyed its join.
    census_rows: dict[tuple[str, str, str, str], dict] = {}
    with open(CENSUS_ROWS, encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if row["family"] != "panel_choice":
                continue
            census_rows[(row["dataset"], row["cell"], row["family"], row["variant"])] = row
    print(f"census panel_choice rows on disk   : {len(census_rows):,}")

    # --- the register, so an option name resolves to a CL id ------------------------------
    defs = json.loads(DEFINITIONS.read_text(encoding="utf-8"))["definitions"]
    name_to_cl: dict[str, str] = {}
    for d in defs:
        name_to_cl.setdefault(d["name"], d["cl_id"])
    cl_to_name = {d["cl_id"]: d["name"] for d in defs}
    print(f"register terms (definitions.json)  : {len(defs):,}")

    offsets = suite_offsets(SUITE_TEST)
    print(f"suite rows indexed                 : {len(offsets):,}")

    out_rows: list[dict] = []
    checks: Counter = Counter()
    mismatch_examples: list[dict] = []
    per_dataset = Counter()

    for rid, cid in sorted(wanted.items(), key=lambda kv: kv[0]):
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

        # the suite's rendition (the one whose _meta.cl_id the population is defined on)
        srow = read_suite_row(SUITE_TEST, offsets, rid)
        meta = srow["_meta"]
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

    with open(OUT_DIR / "questions.jsonl", "w", encoding="utf-8") as fh:
        for row in out_rows:
            fh.write(json.dumps(row) + "\n")

    summary = {
        "population": "census-suite-ids.jsonl, family == panel_choice",
        "questions_written": len(out_rows),
        "questions_per_dataset": dict(sorted(per_dataset.items())),
        "distinct_cells": len({(r["dataset"], r["cell"]) for r in out_rows}),
        "variant_mismatch_rows": sum(1 for r in out_rows if r["variant_mismatch"]),
        "checks": dict(checks),
        "mismatch_examples": mismatch_examples[:10],
        "inputs": {
            "census_ids": str(CENSUS_IDS.relative_to(ROOT)),
            "census_rows": str(CENSUS_ROWS.relative_to(ROOT)),
            "suite": str(SUITE_TEST.relative_to(ROOT)),
            "definitions": str(DEFINITIONS.relative_to(ROOT)),
        },
    }
    (OUT_DIR / "questions.summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "mismatch_examples"}, indent=2))
    print(f"wrote {OUT_DIR / 'questions.jsonl'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
