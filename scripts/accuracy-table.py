#!/usr/bin/env python3
"""The cross-engine accuracy table: one generated table, every cell's source attached.

Which engine is right, on which family of inputs, is the manuscript's accuracy pivot, and those
numbers currently live in five kinds of artifact with their own populations, units and caveats. A
number that travels without its population has already cost this project time. This script
assembles the numbers into one table -- markdown for the reader, JSON for the record -- where every
value carries the population it was measured on, the unit it was counted in, the artifact, the
field path, and that artifact's sha256.

It reads JSON only; it never re-derives a model output. Two inputs are read indirectly:

* the corpus manifest (``runs/real-v2/manifest.json``) supplies the cell count of the test
  partition that the eval reports' ``suite.corpus`` block names but does not itself count, and is
  checked against the digest that block pins;
* the floors block (``reports/figures/headline-table.json``) supplies the already-computed
  always-abstain floor for the full-partition rows.

A missing input is a hard error naming the path and what it would have contributed: an empty cell
would read as "not measured" and a skipped engine as "not run".

With ``--verify`` the script re-reads the JSON it just wrote and re-checks every recorded value
against its source artifact, prints one line per check, and writes the markdown only when every
check passes -- a generator that refuses to write when its verifier fails needs no trust.

    uv run --no-sync python scripts/accuracy-table.py \
      --ours reports/eval-real-v2-eval.json reports/eval-armb-v1-eval.json \
             reports/eval-seed1-v1-eval.json reports/eval-seed2-v1-eval.json \
             reports/eval-seed3-v1-eval.json \
      --jev reports/jev-full/report.json \
      --kev 0.5b=reports/kev-rungs/05b-dump.json 0.6b=reports/kev-rungs/06b-dump.json \
            0.8b=reports/kev-rungs/08b-dump.json 9b=reports/kev-rungs/9b-dump.json \
      --laya scratch/runs/laya-pilot-score/report.json \
      --out reports/accuracy-table.json --md reports/accuracy-table.md --verify
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

#: Fixed family vocabulary, one column each, in every table this script emits.
FAMILIES = [
    "evidence_diluted",
    "evidence_removed",
    "marker_noul",
    "out_of_register",
    "panel_choice",
    "panel_miss",
]

#: The families the census engines report as choices; ``marker_noul`` is their noul rows.
CENSUS_CHOICE_FAMILIES = ["evidence_diluted", "panel_choice", "panel_miss"]

FULL_PARTITION = "full-test-partition"
CENSUS = "eligible-census"

FLOOR_NOT_COMPUTED = "not computed here"

FLOOR_NOT_COMPUTED_NOTE = (
    "would be computed by an always-abstain pass over the census rows "
    "(reports/jev-full/axis1-rows.jsonl: 7,530 choice rows + 41,414 marker_noul rows), scoring each "
    "row's own abstain option as the answer -- this step assembles, it does not compute floors"
)

KNOWN_OURS_CHECKPOINTS = {
    "train-real-v1": ("ours-frozen", "ours (frozen)", 0),
    "train-armb-v1": ("ours-armb", "ours (arm B)", 0),
}


class ShapeError(Exception):
    """An input is missing, or does not have the shape this table reads."""


# -- reading -------------------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: str, contributes: str) -> tuple[dict, str]:
    """Read one input. Missing or unreadable is a hard error, never an empty cell."""
    source = Path(path)
    if not source.is_file():
        raise ShapeError(f"missing input: {path} (would have contributed: {contributes})")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ShapeError(f"unreadable input: {path} ({exc})") from exc
    return payload, sha256_file(source)


def dig(payload: dict, field: str, where: str):
    current = payload
    for part in field.split("."):
        if not isinstance(current, dict) or part not in current:
            raise ShapeError(f"{where}: field {field!r} missing at {part!r}")
        current = current[part]
    if current is None:
        raise ShapeError(f"{where}: field {field!r} is null")
    return current


def family_block(payload: dict, base: str, where: str) -> dict:
    return {
        "n": dig(payload, base + ".n", where),
        "accuracy": dig(payload, base + ".accuracy", where),
        "ece": dig(payload, base + ".ece", where),
        "mean_confidence": dig(payload, base + ".mean_confidence", where),
    }


def axis1_block(payload: dict, where: str, primary_fields: list[str], pooled_field: str) -> dict:
    """The evidence_diluted AUROC if the artifact reports one, else the pooled number, flagged."""
    pooled = dig(payload, pooled_field, where)
    for field in primary_fields:
        try:
            value = dig(payload, field, where)
        except ShapeError:
            continue
        return {"primary_auroc": value, "primary_field": field, "pooled_auroc": pooled,
                "pooled_field": pooled_field, "pooled_fallback": False}
    return {"primary_auroc": pooled, "primary_field": pooled_field, "pooled_auroc": pooled,
            "pooled_field": pooled_field, "pooled_fallback": True}


#: metric name -> how to reach the value inside a row, for the verified-values list.
def value_for(row: dict, metric: str):
    head, _, rest = metric.partition(".")
    if head == "family":
        family, _, leaf = rest.partition(".")
        return row["families"][family][leaf]
    if head in ("overall", "noul", "axis1"):
        return row[head][rest]
    if metric in ("ece", "training_seed", "gate_verdict", "corpus_manifest_sha256"):
        return row[metric]
    raise ShapeError(f"row {row['key']}: no value for metric {metric!r}")


def finish_row(row: dict) -> dict:
    row["values"] = [
        {"metric": metric, "field": field, "value": value_for(row, metric)}
        for metric, field in row.pop("fields").items()
    ]
    return row


# -- rows ----------------------------------------------------------------------------------------


def ours_identity(checkpoint_path: str, where: str) -> tuple[str, str, int]:
    name = Path(checkpoint_path).name
    if name in KNOWN_OURS_CHECKPOINTS:
        return KNOWN_OURS_CHECKPOINTS[name]
    match = re.fullmatch(r"train-seed(\d+)-v1", name)
    if match:
        seed = int(match.group(1))
        return f"ours-seed{seed}", f"ours (seed {seed})", seed
    raise ShapeError(
        f"{where}: checkpoint.path {checkpoint_path!r} is not an arm this table knows "
        "(expected runs/train-real-v1, runs/train-armb-v1 or runs/train-seed<N>-v1)")


def build_ours_row(path: str, payload: dict, sha: str) -> dict:
    key, label, seed = ours_identity(dig(payload, "checkpoint.path", path), path)
    if dig(payload, "checkpoint.training_seed", path) != seed:
        raise ShapeError(f"{path}: checkpoint.training_seed disagrees with {key}")
    if dig(payload, "partitions.reported_on", path) != "test":
        raise ShapeError(f"{path}: partitions.reported_on is not 'test'")
    overall = {"n": dig(payload, "accuracy.n", path),
               "accuracy": dig(payload, "accuracy.accuracy", path)}
    scored = dig(payload, "partitions.rows.test.test", path)
    if scored != overall["n"]:
        raise ShapeError(f"{path}: partitions.rows.test.test ({scored}) != accuracy.n ({overall['n']})")
    fields = {"overall.n": "accuracy.n", "overall.accuracy": "accuracy.accuracy",
              "ece": "temperature.after.ece", "noul.n": "breakdowns.type.noul.n",
              "noul.accuracy": "breakdowns.type.noul.accuracy",
              "training_seed": "checkpoint.training_seed"}
    families = {}
    for family in FAMILIES:
        base = f"breakdowns.family.{family}"
        families[family] = family_block(payload, base, path)
        for leaf in ("n", "accuracy", "ece", "mean_confidence"):
            fields[f"family.{family}.{leaf}"] = f"{base}.{leaf}"
    if sum(block["n"] for block in families.values()) != overall["n"]:
        raise ShapeError(f"{path}: family n's do not sum to accuracy.n ({overall['n']})")
    axis1 = axis1_block(
        payload, path,
        primary_fields=["axis1_panel_adequacy.per_family.evidence_diluted.auroc"],
        pooled_field="axis1_panel_adequacy.auroc")
    fields["axis1.primary_auroc"] = axis1["primary_field"]
    fields["axis1.pooled_auroc"] = axis1["pooled_field"]
    row = {
        "key": key, "label": label, "engine": "ours",
        "source": {"path": path, "sha256": sha},
        "population": FULL_PARTITION, "unit": "question-row",
        "overall": overall, "families": families,
        "ece": dig(payload, "temperature.after.ece", path),
        "noul": {"n": dig(payload, "breakdowns.type.noul.n", path),
                 "accuracy": dig(payload, "breakdowns.type.noul.accuracy", path)},
        "axis1": axis1, "training_seed": seed, "fields": fields,
    }
    if key == "ours-frozen":  # the digest the corpus manifest for this table is pinned by
        row["corpus_manifest_sha256"] = dig(payload, "suite.corpus.manifest_sha256", path)
        fields["corpus_manifest_sha256"] = "suite.corpus.manifest_sha256"
    return finish_row(row)


def build_laya_row(path: str, payload: dict, sha: str) -> dict:
    if dig(payload, "partitions.reported_on", path) != "test":
        raise ShapeError(f"{path}: partitions.reported_on is not 'test'")
    overall = {"n": dig(payload, "partitions.test_rows", path),
               "accuracy": dig(payload, "accuracy", path)}
    fields = {"overall.n": "partitions.test_rows", "overall.accuracy": "accuracy",
              "ece": "ece_after_temperature", "noul.n": "breakdowns.type.noul.n",
              "noul.accuracy": "breakdowns.type.noul.accuracy", "gate_verdict": "gate.verdict"}
    families = {}
    for family in FAMILIES:
        base = f"breakdowns.family.{family}"
        families[family] = family_block(payload, base, path)
        for leaf in ("n", "accuracy", "ece", "mean_confidence"):
            fields[f"family.{family}.{leaf}"] = f"{base}.{leaf}"
    if sum(block["n"] for block in families.values()) != overall["n"]:
        raise ShapeError(f"{path}: family n's do not sum to partitions.test_rows")
    axis1 = axis1_block(
        payload, path,
        primary_fields=["axis1.per_family.evidence_diluted.auroc"],
        pooled_field="axis1_auroc")
    fields["axis1.primary_auroc"] = axis1["primary_field"]
    fields["axis1.pooled_auroc"] = axis1["pooled_field"]
    return finish_row({
        "key": "laya-pilot", "label": "Laya pilot (fine-tuned)", "engine": "laya",
        "source": {"path": path, "sha256": sha},
        "population": FULL_PARTITION, "unit": "question-row",
        "overall": overall, "families": families,
        "ece": dig(payload, "ece_after_temperature", path),
        "noul": {"n": dig(payload, "breakdowns.type.noul.n", path),
                 "accuracy": dig(payload, "breakdowns.type.noul.accuracy", path)},
        "axis1": axis1, "gate_verdict": dig(payload, "gate.verdict", path), "fields": fields,
    })


def build_census_row(key: str, label: str, path: str, payload: dict, sha: str) -> dict:
    """Jev and the Kev rungs: the eligible census, one row per (cell, family) choice question."""
    where, fields = path, {}
    noul_rows = dig(payload, "noul_rows", where)
    if "population" in payload:  # the reports carry the population block; the dumps do not
        if dig(payload, "population.split", where) != "test":
            raise ShapeError(f"{where}: population.split is not 'test'")
        choice_rows = dig(payload, "population.choice_rows", where)
        questions = dig(payload, "population.questions", where)
    else:  # the kev dumps: the same counts on the dump itself
        choice_rows = dig(payload, "choice_rows", where)
        questions = dig(payload, "rows", where)
    if choice_rows + noul_rows != questions:
        raise ShapeError(f"{where}: choice_rows + noul_rows != questions ({questions})")
    overall = {"n": dig(payload, "accuracy.n", where),
               "accuracy": dig(payload, "accuracy.accuracy", where)}
    if overall["n"] != choice_rows:
        raise ShapeError(f"{where}: accuracy.n ({overall['n']}) != choice_rows ({choice_rows})")
    fields.update({"overall.n": "accuracy.n", "overall.accuracy": "accuracy.accuracy",
                   "ece": "ece.ece", "noul.n": "noul_accuracy.n",
                   "noul.accuracy": "noul_accuracy.accuracy"})
    families = {}
    for family in CENSUS_CHOICE_FAMILIES:
        base = f"accuracy_by_family.{family}"
        families[family] = family_block(payload, base, where)
        for leaf in ("n", "accuracy", "ece", "mean_confidence"):
            fields[f"family.{family}.{leaf}"] = f"{base}.{leaf}"
    if sum(block["n"] for block in families.values()) != overall["n"]:
        raise ShapeError(f"{where}: census family n's do not sum to accuracy.n")
    noul = {"n": dig(payload, "noul_accuracy.n", where),
            "accuracy": dig(payload, "noul_accuracy.accuracy", where)}
    if noul["n"] != noul_rows:
        raise ShapeError(f"{where}: noul_accuracy.n ({noul['n']}) != noul_rows ({noul_rows})")
    axis1 = axis1_block(
        payload, where,
        primary_fields=["axis1_panel_adequacy.primary.auroc",
                        "axis1_panel_adequacy.per_family.evidence_diluted.auroc"],
        pooled_field="axis1_panel_adequacy.auroc")
    fields["axis1.primary_auroc"] = axis1["primary_field"]
    fields["axis1.pooled_auroc"] = axis1["pooled_field"]
    return finish_row({
        "key": key, "label": label, "engine": "kev" if key.startswith("kev-") else "jev",
        "source": {"path": path, "sha256": sha},
        "population": CENSUS, "unit": "per-cell-family-question-row",
        "overall": overall, "families": families,
        "ece": dig(payload, "ece.ece", where), "noul": noul, "axis1": axis1, "fields": fields,
    })


# -- populations, floors, seed band ----------------------------------------------------------------


def build_populations(frozen: dict, ours_rows: list[dict], laya: dict, jev: dict, corpus_path: str,
                      corpus_payload: dict, corpus_sha: str) -> dict:
    pinned = next(value["value"] for value in frozen["values"]
                  if value["metric"] == "corpus_manifest_sha256")
    if corpus_sha != pinned:
        raise ShapeError(
            f"corpus manifest digest {corpus_sha[:12]}... != the digest pinned by "
            f"{frozen['source']['path']} (suite.corpus.manifest_sha256 = {pinned[:12]}...)")
    cells = dig(corpus_payload, "splits.splits.test.cells", corpus_path)
    full_rows = {row["overall"]["n"] for row in ours_rows + [laya]}
    if len(full_rows) != 1:
        raise ShapeError(f"the full-partition engines disagree on n: {sorted(full_rows)}")
    family_ns = {tuple(sorted(block["n"] for block in row["families"].values()))
                 for row in ours_rows + [laya]}
    if len(family_ns) != 1:
        raise ShapeError("the full-partition engines disagree on family n's")
    jev_population = "reports/jev-full/report.json"
    census_cells = dig(jev, "population.cells", jev_population)
    return {
        FULL_PARTITION: {
            "definition": (
                "the whole real-v2 test partition as drawn for the frozen eval suite (capped: "
                "31,997 of 89,802 test records); ours and Laya's rows are scored on the same rows"),
            "cells": cells,
            "cells_source": {"path": corpus_path, "field": "splits.splits.test.cells",
                             "sha256": corpus_sha, "pinned_by": frozen["source"]["path"]},
            # cells_source.sha256 is pinned by the frozen report's suite.corpus.manifest_sha256

            "rows": full_rows.pop(),
        },
        CENSUS: {
            "definition": (
                "the eligible census: every test-partition cell carrying all four census families, "
                "asked one request per (cell, state group) -- not a benchmark"),
            "cells": census_cells,
            "cells_source": {"path": jev_population, "field": "population.cells"},
            "choice_rows": dig(jev, "population.choice_rows", jev_population),
            "noul_rows": dig(jev, "noul_rows", jev_population),
            "questions": dig(jev, "population.questions", jev_population),
        },
    }


def build_floors(path: str, payload: dict, sha: str, populations: dict) -> dict:
    root = "the_floor_our_model_has_to_beat"
    rows = dig(payload, root + ".rows", path)
    if rows != populations[FULL_PARTITION]["rows"]:
        raise ShapeError(
            f"{path}: the floor is recorded for {rows} rows but the full partition carries "
            f"{populations[FULL_PARTITION]['rows']} -- the floor does not belong to these rows")
    per_family = {}
    for family in FAMILIES:
        per_family[family] = {
            "value": dig(payload, f"{root}.per_family.{family}.floor", path),
            "field": f"{root}.per_family.{family}.floor",
        }
    return {
        "kind": "always-abstain",
        "source": {"path": path, "sha256": sha},
        "applies_to": FULL_PARTITION,
        "note": (
            "the floor is the accuracy of always answering the row's own abstain option; it is a "
            "property of the scored rows and the family, not of a model. This figure is on the "
            "frozen eval suite's own capped draw (31,997 of 89,802 test records); it is not a "
            "statement about the uncapped partition"),
        "overall": {"value": dig(payload, root + ".floor", path), "n": rows,
                    "field": root + ".floor", "n_field": root + ".rows"},
        "choice_rows": {"value": dig(payload, root + ".choice_rows.floor", path),
                        "n": dig(payload, root + ".choice_rows.n", path),
                        "field": root + ".choice_rows.floor"},
        "noul_rows": {"value": dig(payload, root + ".noul_rows.floor", path),
                      "n": dig(payload, root + ".noul_rows.n", path),
                      "field": root + ".noul_rows.floor"},
        "per_family": per_family,
        CENSUS: {"status": FLOOR_NOT_COMPUTED, "note": FLOOR_NOT_COMPUTED_NOTE},
    }


def build_seed_band(rows: list[dict], frozen: dict) -> dict:
    seeds = sorted((row for row in rows if row["key"].startswith("ours-seed")),
                   key=lambda row: row["training_seed"])
    if len(seeds) != 3:
        raise ShapeError(f"the seed band needs exactly 3 redraws, found {len(seeds)}")
    metrics = []
    for metric in ["overall"] + FAMILIES:
        if metric == "overall":
            values = [row["overall"]["accuracy"] for row in seeds]
            frozen_value = frozen["overall"]["accuracy"]
        else:
            values = [row["families"][metric]["accuracy"] for row in seeds]
            frozen_value = frozen["families"][metric]["accuracy"]
        low, high = min(values), max(values)
        position = "inside" if low <= frozen_value <= high else (
            "above" if frozen_value > high else "below")
        metrics.append({
            "metric": metric,
            "seeds": values,
            "min": low, "max": high, "width": high - low,
            "mean": sum(values) / len(values),
            "frozen": frozen_value,
            "position": position,
            "distance_outside": 0.0 if position == "inside" else (
                frozen_value - high if position == "above" else low - frozen_value),
        })
    return {
        "frozen": frozen["key"],
        "seeds": [{"key": row["key"], "training_seed": row["training_seed"]} for row in seeds],
        "metrics": metrics,
    }


# -- verification ----------------------------------------------------------------------------------


def verify(payload: dict) -> list[str]:
    """Re-read every source and re-check every recorded value. Returns one line per check."""
    lines = []
    by_key = {row["key"]: row for row in payload["engines"]}
    band = payload["seed_band"]

    def check(ok: bool, line: str) -> None:
        lines.append(("OK   " if ok else "FAIL ") + line)

    for population in payload["populations"].values():
        source = population["cells_source"]
        try:
            fresh, sha = load_json(source["path"], "a population cell count")
        except ShapeError as exc:
            check(False, f"population cells: {exc}")
            continue
        if "sha256" in source:
            check(sha == source["sha256"], f"cells source {source['path']} sha256 {sha[:12]}...")
            pinned = next(value["value"] for value in by_key[band["frozen"]]["values"]
                          if value["metric"] == "corpus_manifest_sha256")
            check(sha == pinned,
                  f"corpus manifest sha256 = the digest pinned by {source['pinned_by']}")
        cells = dig(fresh, source["field"], source["path"])
        check(cells == population["cells"],
              f"population cells = {cells} [{source['path']}:{source['field']}]")
    floors = payload.get("floors")
    if floors:
        source = floors["source"]
        fresh, sha = load_json(source["path"], "the always-abstain floor")
        check(sha == source["sha256"], f"floors {source['path']} sha256 {sha[:12]}...")
        for label, block in (("overall", floors["overall"]), ("choice_rows", floors["choice_rows"]),
                             ("noul_rows", floors["noul_rows"]),
                             *[(f"family {fam}", block) for fam, block in floors["per_family"].items()]):
            value = dig(fresh, block["field"], source["path"])
            check(value == block["value"],
                  f"floor {label} = {value} [{block['field']}]")
        rows = dig(fresh, floors["overall"]["n_field"], source["path"])
        check(rows == floors["overall"]["n"],
              f"floor rows = {rows:,} [{floors['overall']['n_field']}]")
    for row in payload["engines"]:
        source = row["source"]
        fresh, sha = load_json(source["path"], f"engine {row['label']}")
        check(sha == source["sha256"], f"{row['key']} source sha256 {sha[:12]}... ({source['path']})")
        for recorded in row["values"]:
            value = dig(fresh, recorded["field"], source["path"])
            check(value == recorded["value"],
                  f"{row['key']} {recorded['metric']} = {value!r} [{recorded['field']}]")
    seed_rows = [by_key[seed["key"]] for seed in band["seeds"]]
    frozen = by_key[band["frozen"]]
    for metric in band["metrics"]:
        if metric["metric"] == "overall":
            values = [row["overall"]["accuracy"] for row in seed_rows]
            frozen_value = frozen["overall"]["accuracy"]
        else:
            values = [row["families"][metric["metric"]]["accuracy"] for row in seed_rows]
            frozen_value = frozen["families"][metric["metric"]]["accuracy"]
        recomputed = (min(values), max(values), sum(values) / len(values))
        recorded = (metric["min"], metric["max"], metric["mean"])
        check(recomputed == recorded and frozen_value == metric["frozen"],
              f"seed band {metric['metric']}: min/max/mean {recomputed[0]:.4f}/{recomputed[1]:.4f}/"
              f"{recomputed[2]:.4f}, frozen {metric['frozen']:.4f} ({metric['position']})")
    return lines


def verify_payload_or_none(payload: dict) -> tuple[list[str], bool]:
    try:
        lines = verify(payload)
    except ShapeError as exc:
        return [f"FAIL verification could not complete: {exc}"], False
    return lines, not any(line.startswith("FAIL") for line in lines)


# -- markdown --------------------------------------------------------------------------------------


def pct(value: float, digits: int = 1) -> str:
    return f"{100 * value:.{digits}f}%"


def band_note(metric: dict, digits: int = 1) -> str:
    """The band beside a frozen number, marked when the frozen value is outside it."""
    band = f"[{100 * metric['min']:.{digits}f}-{100 * metric['max']:.{digits}f}]"
    if metric["position"] != "inside":
        band += f" outside by {100 * metric['distance_outside']:.{digits}f} pp"
    return band


def render_markdown(payload: dict) -> str:
    populations, band = payload["populations"], payload["seed_band"]
    frozen_key = band["frozen"]
    band_of = {metric["metric"]: metric for metric in band["metrics"]}
    lines = [
        "<!-- GENERATED by scripts/accuracy-table.py -- do not edit by hand -->",
        "",
        "# The cross-engine accuracy table",
        "",
        "Every number below was measured on the population and unit its row names, and is recorded",
        "in `reports/accuracy-table.json` with its source artifact, sha256 and field path.",
        "",
        "## The table",
        "",
    ]
    header = ["engine", "overall", "n", *FAMILIES, "ECE", "noul accuracy", "axis-1 primary AUROC",
              "floor (always-abstain)", "population", "unit", "source"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for row in payload["engines"]:
        frozen = row["key"] == frozen_key
        cells = []
        for family in FAMILIES:
            block = row["families"].get(family)
            if block is None:
                cells.append("—" if family != "marker_noul" else "— (yes/no rows: see noul acc.)")
            else:
                text = pct(block["accuracy"])
                if frozen:
                    text += " " + band_note(band_of[family])
                cells.append(text)
        overall = pct(row["overall"]["accuracy"]) + (" " + band_note(band_of["overall"]) if frozen else "")
        axis1 = f"{row['axis1']['primary_auroc']:.4f}"
        if row["axis1"]["pooled_fallback"]:
            axis1 += " (pooled)"
        floor = row["floor"]
        floor_text = (pct(floor["value"]) if floor["status"] == "computed"
                      else f"floor: {floor['status']}")
        unit = ("one scored question row (marker_noul = 111,661 of the rows)"
                if row["unit"] == "question-row" else
                "one row per (cell, family) choice question; marker_noul as yes/no rows")
        population = (f"full test partition ({populations[FULL_PARTITION]['cells']:,} cells / "
                      f"{populations[FULL_PARTITION]['rows']:,} rows)"
                      if row["population"] == FULL_PARTITION else
                      f"eligible census ({populations[CENSUS]['cells']:,} cells / "
                      f"{populations[CENSUS]['choice_rows']:,} + {populations[CENSUS]['noul_rows']:,} rows)")
        cells = [row["label"], overall, f"{row['overall']['n']:,}", *cells, f"{row['ece']:.3f}",
                 f"{pct(row['noul']['accuracy'])} ({row['noul']['n']:,})", axis1, floor_text,
                 population, unit, f"`{row['source']['path']}`"]
        lines.append("| " + " | ".join(cells) + " |")

    full, census = populations[FULL_PARTITION], populations[CENSUS]
    lines += [
        "",
        "## How to read this",
        "",
        "- **Population is a column, not a footnote.** `ours` (all five arms) and `Laya` were measured",
        f"  on the {full['definition']}: {full['cells']:,} cells, {full['rows']:,} scored rows.",
        f"  `Jev` and the `Kev` rungs were measured on the {census['definition']}: {census['cells']:,}",
        f"  cells, {census['choice_rows']:,} choice rows + {census['noul_rows']:,} `marker_noul` rows.",
        "- **Unit is a column, not a footnote.** Our arms and Laya answer every yes/no question as",
        "  its own row, so `marker_noul` is 111,661 of our 136,606 rows; the census engines answer",
        f"  one row per (cell, family) choice question, with `marker_noul` as {census['noul_rows']:,}",
        "  yes/no rows. The two `marker_noul` numbers have different denominators (111,661 vs",
        f"  {census['noul_rows']:,} rows) and different populations — say so when quoting either.",
        "- **Jev's and the Kev rungs' numbers must never be subtracted from ours** (or Laya's):",
        f"  different cells ({census['cells']:,} census cells vs {full['cells']:,} partition cells),",
        "  different families (the census reports three choice families plus `marker_noul`; ours",
        "  reports six), and different question mixes",
        f"  ({census['choice_rows']:,} + {census['noul_rows']:,} census rows vs {full['rows']:,} ours).",
        "",
        "## Seed band: three redraws of the frozen configuration",
        "",
        "For every family and for the overall number: min / max / mean over `ours-seed1..3`, and where",
        "the frozen value sits relative to that band. Values are accuracies (fractions).",
        "",
    ]
    seed_header = ["metric", "seed 1", "seed 2", "seed 3", "min", "max", "width (pp)", "mean",
                   "frozen", "position"]
    lines.append("| " + " | ".join(seed_header) + " |")
    lines.append("|" + "---|" * len(seed_header))
    for metric in band["metrics"]:
        seeds = " | ".join(f"{value:.4f}" for value in metric["seeds"])
        position = metric["position"] if metric["position"] == "inside" else (
            f"{metric['position']} by {100 * metric['distance_outside']:.1f} pp")
        lines.append(f"| `{metric['metric']}` | {seeds} | {metric['min']:.4f} | {metric['max']:.4f} "
                     f"| {100 * metric['width']:.1f} | {metric['mean']:.4f} | {metric['frozen']:.4f} "
                     f"| {position} |")
    outside = [metric for metric in band["metrics"] if metric["position"] != "inside"]
    for metric in outside:
        lines.append(
            f"\nThe frozen `{metric['metric']}` value sits outside the band, "
            f"{100 * metric['distance_outside']:.1f} pp {metric['position']} its "
            f"{'max' if metric['position'] == 'above' else 'min'} of {metric['max' if metric['position'] == 'above' else 'min']:.4f}."
            " The frozen row of the table above carries each band beside its value for this reason.")
    marker = next(metric for metric in band["metrics"] if metric["metric"] == "marker_noul")
    lines += [
        "",
        (f"The `marker_noul` band is {100 * marker['width']:.1f} points wide "
         f"({marker['min']:.4f}-{marker['max']:.4f}) against a frozen {marker['frozen']:.4f}: the"),
        "frozen point is a draw from the band's own distribution, not a stable quantity at the",
        "4th decimal a table prints.",
        "",
        "## Floors",
        "",
        "A floor is a property of the scored population and the family. In an abstention task the",
        "honest baseline is always abstaining, not the majority class.",
        "",
    ]
    floors = payload.get("floors")
    if floors:
        census_floor = floors[CENSUS]
        no_floor = [row["label"] for row in payload["engines"] if row["population"] == CENSUS]
        lines += [
            f"- On the full-partition rows ({floors['overall']['n']:,}), a rule that always answers",
            f"  the abstain option scores **{pct(floors['overall']['value'])}**",
            f"  (`{floors['source']['path']}` → `{floors['overall']['field']}`; the file's own",
            f"  `rows` = {floors['overall']['n']:,}, checked against the partitions in this table).",
            f"  {floors['note']}.",
            f"- Choice rows only ({floors['choice_rows']['n']:,}): floor {pct(floors['choice_rows']['value'])}.",
            f"  `marker_noul` rows only ({floors['noul_rows']['n']:,}): floor {pct(floors['noul_rows']['value'])}.",
            "- Per family on the full-partition rows:",
            "",
            "| family | floor (always-abstain) | field |",
            "|---|---:|---|",
        ]
        for family in FAMILIES:
            block = floors["per_family"][family]
            lines.append(f"| `{family}` | {pct(block['value'])} | `{block['field']}` |")
        lines += [
            "",
            f"- For {' and '.join(no_floor)} ({census['cells']:,}-cell census): the always-abstain",
            f"  floor is {census_floor['status']}; it",
            f"  {census_floor['note']}.",
            "",
        ]
    lines += ["## Provenance", ""]
    for row in payload["engines"]:
        lines.append(f"- **{row['label']}** (`{row['key']}`) — `{row['source']['path']}` "
                     f"sha256 `{row['source']['sha256']}`")
        for value in row["values"]:
            lines.append(f"    - {value['metric']} → `{value['field']}`")
    for name, source in (("corpus manifest", populations[FULL_PARTITION]["cells_source"]),
                         ("floors", floors["source"] if floors else None)):
        if source is None:
            continue
        sha = source.get("sha256", "—")
        lines.append(f"- **{name}** — `{source['path']}` sha256 `{sha}`")
    lines += [
        "",
        "Populations: " + "; ".join(
            f"`{key}` = {value['definition']} (cells from `{value['cells_source']['path']}` → "
            f"`{value['cells_source']['field']}`)"
            for key, value in populations.items()) + ".",
        "",
        f"Command: `{payload['command']}`",
        "",
    ]
    return "\n".join(lines)


# -- main -------------------------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ours", nargs="+", required=True, metavar="PATH",
                        help="the five ours arms: frozen, arm B, seed1..3")
    parser.add_argument("--jev", required=True, metavar="PATH")
    parser.add_argument("--kev", nargs="+", required=True, metavar="TAG=PATH")
    parser.add_argument("--laya", required=True, metavar="PATH")
    parser.add_argument("--out", required=True, metavar="PATH", help="the JSON to write")
    parser.add_argument("--md", required=True, metavar="PATH", help="the markdown to write")
    parser.add_argument("--verify", action="store_true",
                        help="re-read the written JSON and re-check every value before writing the md")
    parser.add_argument("--floors", default="reports/figures/headline-table.json", metavar="PATH",
                        help="an already-computed floors block (default: %(default)s)")
    parser.add_argument("--corpus-manifest", default="runs/real-v2/manifest.json", metavar="PATH",
                        help="the corpus manifest the eval reports pin by digest (default: %(default)s)")
    return parser.parse_args(argv)


def build_table(args: argparse.Namespace, invocation: list[str]) -> dict:
    ours_rows = []
    for path in args.ours:
        payload, sha = load_json(path, "an ours arm row")
        ours_rows.append(build_ours_row(path, payload, sha))
    frozen_rows = [row for row in ours_rows if row["key"] == "ours-frozen"]
    if len(frozen_rows) != 1:
        raise ShapeError(f"--ours needs exactly one frozen arm (runs/train-real-v1), found {len(frozen_rows)}")
    laya_path = args.laya
    laya_payload, laya_sha = load_json(laya_path, "the Laya pilot row")
    laya = build_laya_row(laya_path, laya_payload, laya_sha)

    jev_path = args.jev
    jev_payload, jev_sha = load_json(jev_path, "the Jev row")
    rows = [build_census_row("jev", "Jev (census)", jev_path, jev_payload, jev_sha)]
    for spec in args.kev:
        tag, _, path = spec.partition("=")
        if not path:
            raise ShapeError(f"--kev expects TAG=PATH, got {spec!r}")
        payload, sha = load_json(path, f"the Kev {tag.upper()} row")
        rows.append(build_census_row(f"kev-{tag}", f"Kev {tag.upper()}", path, payload, sha))
    census_rows = rows
    if len({row["overall"]["n"] for row in census_rows}) != 1 or \
            len({row["noul"]["n"] for row in census_rows}) != 1:
        raise ShapeError("the census engines disagree on the census population's row counts")

    corpus_path = args.corpus_manifest
    corpus_payload, corpus_sha = load_json(
        corpus_path, "the test-partition cell count for the full-partition population")
    populations = build_populations(frozen_rows[0], ours_rows, laya, jev_payload, corpus_path,
                                    corpus_payload, corpus_sha)
    floors_path = args.floors
    floors_payload, floors_sha = load_json(floors_path, "the always-abstain floor for the full-partition rows")
    floors = build_floors(floors_path, floors_payload, floors_sha, populations)
    floor_value = floors["overall"]["value"]

    for row in ours_rows + [laya]:
        row["floor"] = {"status": "computed", "kind": "always-abstain", "value": floor_value,
                        "n": populations[FULL_PARTITION]["rows"],
                        "source_path": floors["source"]["path"],
                        "source_field": floors["overall"]["field"]}
    for row in census_rows:
        row["floor"] = {"status": floors[CENSUS]["status"], "kind": "always-abstain"}
    if laya["gate_verdict"]:
        laya["gate_note"] = f"pilot gate verdict: {laya['gate_verdict']}"

    engines = sorted(rows + ours_rows + [laya], key=lambda row: row["key"])
    band = build_seed_band(ours_rows, frozen_rows[0])
    return {
        "tool": "scripts/accuracy-table.py",
        "command": " ".join(["uv run --no-sync python scripts/accuracy-table.py", *invocation]),
        "families": FAMILIES,
        "family_order": "fixed, alphabetical; identical in this JSON and in the markdown",
        "populations": populations,
        "floors": floors,
        "seed_band": band,
        "engines": engines,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    invocation = sys.argv[1:] if argv is None else argv
    try:
        payload = build_table(args, invocation)
    except ShapeError as exc:
        print(f"accuracy-table: {exc}", file=sys.stderr)
        return 1
    out = Path(args.out)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out} ({len(payload['engines'])} engines)")
    if args.verify:
        written = json.loads(out.read_text(encoding="utf-8"))
        lines, ok = verify_payload_or_none(written)
        for line in lines:
            print(line)
        failed = sum(line.startswith("FAIL") for line in lines)
        print(f"verified: {len(lines)} checks, {failed} failed")
        if not ok:
            print(f"verification failed: {args.md} not written", file=sys.stderr)
            return 1
    Path(args.md).write_text(render_markdown(payload), encoding="utf-8")
    print(f"wrote {args.md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
