"""The cross-engine accuracy table, on synthetic JSON only.

``scripts/accuracy-table.py`` writes the table the manuscript quotes, and gates the
markdown on a re-read of the JSON it just wrote. The load-bearing risks pinned here:

* order -- the six families are a fixed vocabulary and engines are emitted sorted, so
  two runs (and the JSON and the markdown) agree on the shape of the table;
* fail-closed reads -- a missing input is exit 1 naming the path, nothing written;
* the seed band -- min/max/mean over the three redraws, and the frozen value's
  position, marked in the markdown with the distance when it is outside;
* the verifier -- a JSON whose number disagrees with its source fails the check and
  the markdown is not written (a generator that refuses to write needs no trust);
* population and unit travel with every row as columns, never blank cells;
* a floor the sources do not carry stays ``floor: not computed here``, never a number.

Every input is a small synthetic JSON written under ``tmp_path``; no network, no GPU,
no model runs, nothing read from ``reports/`` or ``runs/``.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "accuracy-table.py"


def _load():
    spec = importlib.util.spec_from_file_location("accuracy_table", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["accuracy_table"] = module
    spec.loader.exec_module(module)
    return module


acc = _load()

# The fixed order the module declares, restated as a literal so the assertion is not
# the module compared against itself.
FAMILIES = ["evidence_diluted", "evidence_removed", "marker_noul",
            "out_of_register", "panel_choice", "panel_miss"]
CENSUS_FAMILIES = ["evidence_diluted", "panel_choice", "panel_miss"]

POP_FULL = "full-test-partition"
POP_CENSUS = "eligible-census"
FLOOR_PLACEHOLDER = "not computed here"

FULL_N = {"evidence_diluted": 2, "evidence_removed": 1, "marker_noul": 5,
          "out_of_register": 1, "panel_choice": 2, "panel_miss": 1}
FULL_ROWS = 12  # == sum(FULL_N.values())
NOUL_N = FULL_N["marker_noul"]
CENSUS_N = {"evidence_diluted": 2, "panel_choice": 2, "panel_miss": 2}
CHOICE_ROWS = 6
NOUL_ROWS = 3
QUESTIONS = 9  # == CHOICE_ROWS + NOUL_ROWS
CENSUS_CELLS = 4
PARTITION_CELLS = 7
FLOOR_OVERALL = 0.60

SEED_OVERALL = {1: 0.44, 2: 0.50, 3: 0.56}
SEED_ACCS = {
    1: {"evidence_diluted": 0.30, "evidence_removed": 0.90, "marker_noul": 0.50,
        "out_of_register": 0.80, "panel_choice": 0.20, "panel_miss": 0.70},
    2: {"evidence_diluted": 0.32, "evidence_removed": 0.92, "marker_noul": 0.52,
        "out_of_register": 0.82, "panel_choice": 0.24, "panel_miss": 0.72},
    3: {"evidence_diluted": 0.34, "evidence_removed": 0.94, "marker_noul": 0.54,
        "out_of_register": 0.84, "panel_choice": 0.28, "panel_miss": 0.74},
}
FROZEN_OVERALL = 0.50
# Everything but panel_choice sits inside the seed band by construction; panel_choice's
# frozen value is the parameter, so one fixture can hold the baseline inside the band
# and another outside it.
FROZEN_ACCS = {"evidence_diluted": 0.33, "evidence_removed": 0.91, "marker_noul": 0.51,
               "out_of_register": 0.83, "panel_miss": 0.71}
ARMB_ACCS = {"evidence_diluted": 0.31, "evidence_removed": 0.95, "marker_noul": 0.60,
             "out_of_register": 0.86, "panel_choice": 0.30, "panel_miss": 0.75}
LAYA_ACCS = {"evidence_diluted": 0.35, "evidence_removed": 0.97, "marker_noul": 0.80,
             "out_of_register": 0.90, "panel_choice": 0.15, "panel_miss": 0.85}
CENSUS_ACCS = {"evidence_diluted": 0.60, "panel_choice": 0.45, "panel_miss": 0.55}


def _family_blocks(order, n_of, accs):
    return {fam: {"n": n_of[fam], "accuracy": accs[fam], "ece": 0.10,
                  "mean_confidence": 0.60} for fam in order}


def _write_ours(path, checkpoint, seed, accs, overall, manifest_sha=None):
    payload = {
        "checkpoint": {"path": checkpoint, "training_seed": seed},
        "partitions": {"reported_on": "test", "rows": {"test": {"test": FULL_ROWS}}},
        "accuracy": {"n": FULL_ROWS, "accuracy": overall},
        "temperature": {"after": {"ece": 0.05}},
        "breakdowns": {
            "type": {"noul": {"n": NOUL_N, "accuracy": accs["marker_noul"]}},
            "family": _family_blocks(FAMILIES, FULL_N, accs),
        },
        "axis1_panel_adequacy": {"per_family": {"evidence_diluted": {"auroc": 0.61}},
                                 "auroc": 0.55},
    }
    if manifest_sha is not None:  # only the frozen arm pins the corpus digest
        payload["suite"] = {"corpus": {"manifest_sha256": manifest_sha}}
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _write_laya(path):
    path.write_text(json.dumps({
        "partitions": {"reported_on": "test", "test_rows": FULL_ROWS},
        "accuracy": 0.70,
        "ece_after_temperature": 0.12,
        "breakdowns": {"type": {"noul": {"n": NOUL_N, "accuracy": LAYA_ACCS["marker_noul"]}},
                       "family": _family_blocks(FAMILIES, FULL_N, LAYA_ACCS)},
        "axis1": {},  # no per-family AUROC: the pooled fallback applies
        "axis1_auroc": 0.48,
        "gate": {"verdict": "pass"},
    }, indent=2) + "\n")


def _write_census(path, with_population):
    payload = {
        "noul_rows": NOUL_ROWS,
        "accuracy": {"n": CHOICE_ROWS, "accuracy": 0.50},
        "ece": {"ece": 0.20},
        "noul_accuracy": {"n": NOUL_ROWS, "accuracy": 0.90},
        "accuracy_by_family": _family_blocks(CENSUS_FAMILIES, CENSUS_N, CENSUS_ACCS),
        "axis1_panel_adequacy": {"primary": {"auroc": 0.70}, "auroc": 0.60},
    }
    if with_population:  # the reports carry the block; the kev dumps carry the counts
        payload["population"] = {"split": "test", "cells": CENSUS_CELLS,
                                 "choice_rows": CHOICE_ROWS, "questions": QUESTIONS}
    else:
        payload["choice_rows"] = CHOICE_ROWS
        payload["rows"] = QUESTIONS
    path.write_text(json.dumps(payload, indent=2) + "\n")


def write_fixture_tree(root: Path, frozen_choice: float = 0.42) -> dict[str, Path]:
    """Every source ``scripts/accuracy-table.py`` reads, as small synthetic JSON."""
    paths = {name: root / f"{name}.json" for name in
             ("frozen", "armb", "seed1", "seed2", "seed3", "jev", "kev05", "kev9",
              "laya", "floors", "manifest")}
    paths["out"] = root / "accuracy-table.json"
    paths["md"] = root / "accuracy-table.md"

    manifest = paths["manifest"]
    manifest.write_text(json.dumps(
        {"splits": {"splits": {"test": {"cells": PARTITION_CELLS}}}}))
    # the digest is computed here, independently of the script's sha256_file
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()

    frozen_accs = dict(FROZEN_ACCS, panel_choice=frozen_choice)
    _write_ours(paths["frozen"], "runs/train-real-v1", 0, frozen_accs, FROZEN_OVERALL,
                manifest_sha=manifest_sha)
    _write_ours(paths["armb"], "runs/train-armb-v1", 0, ARMB_ACCS, 0.62)
    for seed in (1, 2, 3):
        _write_ours(paths[f"seed{seed}"], f"runs/train-seed{seed}-v1", seed,
                    SEED_ACCS[seed], SEED_OVERALL[seed])
    _write_laya(paths["laya"])
    _write_census(paths["jev"], with_population=True)
    # scripts/accuracy-table.py hard-codes the census cells_source path as
    # "reports/jev-full/report.json" (build_populations), ignoring --jev, and the
    # verifier re-reads it relative to the CWD. So a synthetic copy must sit at that
    # relative path for verification to read the same cells. (Reported, not fixed.)
    jev_lookalike = root / "reports" / "jev-full" / "report.json"
    jev_lookalike.parent.mkdir(parents=True, exist_ok=True)
    jev_lookalike.write_bytes(paths["jev"].read_bytes())
    _write_census(paths["kev05"], with_population=False)
    _write_census(paths["kev9"], with_population=False)
    paths["floors"].write_text(json.dumps({"the_floor_our_model_has_to_beat": {
        "rows": FULL_ROWS,
        "floor": FLOOR_OVERALL,
        "choice_rows": {"floor": 0.40, "n": 8},
        "noul_rows": {"floor": 0.80, "n": 4},
        "per_family": {fam: {"floor": 0.50} for fam in FAMILIES},
    }}))
    return paths


def argv_for(paths: dict[str, Path], verify: bool = True) -> list[str]:
    argv = [
        "--ours", str(paths["frozen"]), str(paths["armb"]),
        str(paths["seed1"]), str(paths["seed2"]), str(paths["seed3"]),
        "--jev", str(paths["jev"]),
        "--kev", f"0.5b={paths['kev05']}", f"9b={paths['kev9']}",
        "--laya", str(paths["laya"]),
        "--out", str(paths["out"]), "--md", str(paths["md"]),
        "--floors", str(paths["floors"]),
        "--corpus-manifest", str(paths["manifest"]),
    ]
    return argv + (["--verify"] if verify else [])


def build_table(tmp_path: Path, monkeypatch, frozen_choice: float = 0.42):
    """Write the fixture tree, run the real CLI with --verify, return (payload, md, paths)."""
    paths = write_fixture_tree(tmp_path, frozen_choice)
    monkeypatch.chdir(tmp_path)  # the census cells source is read by a relative path
    argv = argv_for(paths)
    rc = acc.main(argv)
    assert rc == 0, f"the synthetic table failed to build (rc={rc})"
    return (json.loads(paths["out"].read_text()), paths["md"].read_text(), paths)


def _row_line(md: str, label: str) -> str:
    return next(line for line in md.splitlines() if line.startswith(f"| {label} |"))


# ----------------------------------------------------------------------------------
# Order
# ----------------------------------------------------------------------------------


def test_family_order_is_fixed_and_the_engines_are_sorted(tmp_path, monkeypatch):
    payload, md, _ = build_table(tmp_path, monkeypatch)
    assert payload["families"] == [
        "evidence_diluted", "evidence_removed", "marker_noul",
        "out_of_register", "panel_choice", "panel_miss"]
    keys = [row["key"] for row in payload["engines"]]
    assert keys == sorted(keys)
    assert keys == ["jev", "kev-0.5b", "kev-9b", "laya-pilot", "ours-armb",
                    "ours-frozen", "ours-seed1", "ours-seed2", "ours-seed3"]
    for row in payload["engines"]:
        # Classify by the row's own population, not by a hard-coded key list: every census
        # engine (jev and each kev rung) carries the choice families only, while the
        # full-partition engines carry all six. A key list missed kev-9b, which is census.
        expected = CENSUS_FAMILIES if row["population"] == POP_CENSUS else FAMILIES
        assert list(row["families"]) == expected
    assert [m["metric"] for m in payload["seed_band"]["metrics"]] == ["overall", *FAMILIES]
    header = next(line for line in md.splitlines() if line.startswith("| engine |"))
    assert header == (
        "| engine | overall | n | evidence_diluted | evidence_removed | marker_noul "
        "| out_of_register | panel_choice | panel_miss | ECE | noul accuracy "
        "| axis-1 primary AUROC | floor (always-abstain) | population | unit | source |")
    # a second run over the same sources emits the same order
    payload2, _, _ = build_table(tmp_path, monkeypatch)
    assert [row["key"] for row in payload2["engines"]] == keys


# ----------------------------------------------------------------------------------
# Missing input: exit non-zero, name the path, write nothing
# ----------------------------------------------------------------------------------


def test_a_missing_input_is_a_hard_error_and_nothing_is_written(tmp_path, capsys):
    paths = write_fixture_tree(tmp_path)
    missing = tmp_path / "does-not-exist-laya.json"
    paths["laya"] = missing
    rc = acc.main(argv_for(paths, verify=True))
    err = capsys.readouterr().err
    assert rc == 1  # the negative control fires
    assert "missing input" in err and str(missing) in err
    assert not paths["out"].exists()
    assert not paths["md"].exists()


# ----------------------------------------------------------------------------------
# The seed band
# ----------------------------------------------------------------------------------


@pytest.mark.parametrize(("frozen_choice", "position"), [(0.24, "inside"), (0.42, "above")])
def test_seed_band_min_max_mean_and_the_frozen_position(tmp_path, monkeypatch, frozen_choice, position):
    payload, md, _ = build_table(tmp_path, monkeypatch, frozen_choice=frozen_choice)
    band = payload["seed_band"]
    assert band["frozen"] == "ours-frozen"
    assert [seed["training_seed"] for seed in band["seeds"]] == [1, 2, 3]
    metrics = {metric["metric"]: metric for metric in band["metrics"]}

    choice = metrics["panel_choice"]
    assert choice["seeds"] == [0.20, 0.24, 0.28]
    assert choice["min"] == 0.20 and choice["max"] == 0.28
    assert choice["mean"] == pytest.approx(0.24)
    assert choice["width"] == pytest.approx(0.08)
    assert choice["frozen"] == frozen_choice
    assert choice["position"] == position
    # the frozen overall value sits inside the band in both fixtures
    assert metrics["overall"]["position"] == "inside"
    assert metrics["overall"]["min"] == 0.44 and metrics["overall"]["max"] == 0.56
    assert metrics["overall"]["mean"] == pytest.approx(0.50)
    assert metrics["overall"]["distance_outside"] == 0.0

    frozen_line = _row_line(md, "ours (frozen)")
    band_line = next(line for line in md.splitlines()
                     if line.startswith("| `panel_choice` |"))
    if position == "inside":
        assert choice["distance_outside"] == 0.0
        assert "24.0% [20.0-28.0]" in frozen_line
        assert "outside" not in frozen_line and "outside" not in md
        assert band_line.endswith("| inside |")
    else:
        assert choice["distance_outside"] == pytest.approx(0.14)
        assert "42.0% [20.0-28.0] outside by 14.0 pp" in frozen_line
        assert band_line.endswith("| above by 14.0 pp |")
        assert ("The frozen `panel_choice` value sits outside the band, 14.0 pp above "
                "its max of 0.2800.") in md


# ----------------------------------------------------------------------------------
# The verifier
# ----------------------------------------------------------------------------------


def test_a_verified_run_reports_zero_failed_checks(tmp_path, monkeypatch, capsys):
    build_table(tmp_path, monkeypatch)
    out = capsys.readouterr().out
    assert "verified:" in out and "0 failed" in out
    assert "FAIL" not in out


def test_the_verifier_rejects_a_changed_number_and_the_markdown_is_not_written(
        tmp_path, monkeypatch, capsys):
    paths = write_fixture_tree(tmp_path)
    monkeypatch.chdir(tmp_path)
    argv = argv_for(paths, verify=True)
    good = acc.build_table(acc.parse_args(argv), argv)  # a real, verifiable payload
    lines, ok = acc.verify_payload_or_none(good)
    assert ok, [line for line in lines if line.startswith("FAIL")]

    # change one number in the generated JSON: the frozen arm's overall accuracy
    frozen = next(row for row in good["engines"] if row["key"] == "ours-frozen")
    record = next(value for value in frozen["values"] if value["metric"] == "overall.accuracy")
    record["value"] += 0.01
    lines, ok = acc.verify_payload_or_none(good)
    assert not ok
    assert any(line.startswith("FAIL") and "ours-frozen overall.accuracy" in line
               for line in lines)

    # the same tampered payload, through main(): exit 1 and the markdown left alone
    monkeypatch.setattr(acc, "build_table", lambda args, invocation: good)
    md_path = paths["md"]
    md_path.write_text("previous run\n")
    rc = acc.main(argv)
    captured = capsys.readouterr()
    assert rc == 1
    assert "FAIL ours-frozen overall.accuracy" in captured.out
    assert "1 failed" in captured.out
    assert "not written" in captured.err
    assert md_path.read_text() == "previous run\n"
    written = json.loads(paths["out"].read_text())
    row = next(row for row in written["engines"] if row["key"] == "ours-frozen")
    value = next(v for v in row["values"] if v["metric"] == "overall.accuracy")
    assert value["value"] == pytest.approx(0.51)  # what the verifier read and rejected


# ----------------------------------------------------------------------------------
# Population and unit travel with every row
# ----------------------------------------------------------------------------------


def test_every_row_carries_its_population_and_unit(tmp_path, monkeypatch):
    payload, md, _ = build_table(tmp_path, monkeypatch)
    for row in payload["engines"]:
        assert row["population"] in (POP_FULL, POP_CENSUS)
        assert isinstance(row["unit"], str) and row["unit"]
    # and the markdown renders both as columns, filled, for every row
    for row in payload["engines"]:
        line = _row_line(md, row["label"])
        if row["population"] == POP_FULL:
            assert f"full test partition ({PARTITION_CELLS} cells / {FULL_ROWS} rows)" in line
            assert "one scored question row" in line
        else:
            assert f"eligible census ({CENSUS_CELLS} cells / {CHOICE_ROWS} + {NOUL_ROWS} rows)" in line
            assert "one row per (cell, family) choice question" in line


@pytest.mark.parametrize(("mutation", "message"), [
    ("no-population-block", "choice_rows"),
    ("split-not-test", "population.split is not 'test'"),
])
def test_a_source_without_a_usable_population_is_refused(tmp_path, capsys, mutation, message):
    paths = write_fixture_tree(tmp_path)
    jev = json.loads(paths["jev"].read_text())
    if mutation == "no-population-block":
        del jev["population"]  # neither the block nor the dump-style counts remain
    else:
        jev["population"]["split"] = "train"
    paths["jev"].write_text(json.dumps(jev))
    rc = acc.main(argv_for(paths, verify=False))
    err = capsys.readouterr().err
    assert rc == 1
    assert message in err
    assert not paths["out"].exists() and not paths["md"].exists()


# ----------------------------------------------------------------------------------
# The floor column
# ----------------------------------------------------------------------------------


def test_the_floor_column_keeps_the_honest_placeholder(tmp_path, monkeypatch):
    payload, md, _ = build_table(tmp_path, monkeypatch)
    by_key = {row["key"]: row for row in payload["engines"]}
    for key in ("jev", "kev-0.5b", "kev-9b"):
        floor = by_key[key]["floor"]
        assert floor["status"] == FLOOR_PLACEHOLDER
        assert "value" not in floor  # a placeholder, not a number
        assert f"floor: {FLOOR_PLACEHOLDER}" in _row_line(md, by_key[key]["label"])
    for key in ("ours-frozen", "laya-pilot"):
        floor = by_key[key]["floor"]
        assert floor["status"] == "computed"
        assert floor["value"] == FLOOR_OVERALL
        assert "60.0%" in _row_line(md, by_key[key]["label"])
    assert payload["floors"][POP_CENSUS]["status"] == FLOOR_PLACEHOLDER
    assert "would be computed by an always-abstain pass" in md
    assert "- On the full-partition rows (12), a rule that always answers" in md


# ----------------------------------------------------------------------------------
# Axis-1 AUROC fallback
# ----------------------------------------------------------------------------------


def test_a_pooled_axis1_auroc_is_flagged_as_pooled(tmp_path, monkeypatch):
    payload, md, _ = build_table(tmp_path, monkeypatch)
    laya = next(row for row in payload["engines"] if row["key"] == "laya-pilot")
    assert laya["axis1"]["pooled_fallback"] is True
    assert laya["axis1"]["primary_field"] == "axis1_auroc"
    assert "0.4800 (pooled)" in _row_line(md, laya["label"])
    frozen = next(row for row in payload["engines"] if row["key"] == "ours-frozen")
    assert frozen["axis1"]["pooled_fallback"] is False
    assert frozen["axis1"]["primary_field"] == \
        "axis1_panel_adequacy.per_family.evidence_diluted.auroc"
    assert "0.6100" in _row_line(md, frozen["label"])
