"""The Baron reconciliation table: the readings it borrows, the denominators it quotes, and its gates.

`reports/baron-reconciliation.py` is a **re-aggregation**, so what can go wrong is not a model run but a
reader: a rule re-implemented instead of imported, a denominator swapped, a family scored as if its
reference label were a class rather than the abstain option, a prefix of a run in flight read as a
result, or an external baseline given a column it cannot have. Each of those has a test here, and every
one of them can fail:

* **the rules are imported, not copied** — `cell()` comes from `reports/abstain-reconciliation-v4.py` by
  path, so the module must expose it and the tests assert the identity that ties the three readings
  together on hand-built rows;
* **both denominators are reported** — the same rows scored with and without the abstain option are
  *different numbers*, and a table that quotes one and calls it the other is wrong in both directions;
* **`panel_miss` is a decline rate** — the reference label is the abstain option on every row, so the
  served accuracy and the abstain-pick rate are the same number by construction and `forced` is 0;
* **a short `marker_noul` dump is not scored** — the 4B rung's top-up was still appending when this table
  was first built, and a prefix scored as an accuracy would have been a fabricated result;
* **the v4 controls are the gate** — the six numbers this table asserts are the ones that already existed
  elsewhere, and a run that misses one writes nothing at all.

No GPU, no cluster, no dumps on disk: everything below is hand-built rows and the committed artefacts.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "analysis" / "baron-reconciliation.py"
COMMITTED = ROOT / "reports" / "baron-reconciliation.json"


@pytest.fixture(scope="module")
def baron():
    spec = importlib.util.spec_from_file_location("baron_reconciliation", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def settled():
    """``cell()`` from the v4 table the Baron one borrows -- executed, never re-written here."""
    spec = importlib.util.spec_from_file_location("abstain_reconciliation_v4_for_tests",
                                                  ROOT / "analysis" / "abstain-reconciliation-v4.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def choice_row(record: str, p, label: int, abstain=None, family: str = "panel_choice") -> dict:
    """An option-probability row in the dump's shape: ``p``, ``label``, and the abstain position."""
    return {"id": record, "family": family, "label": label, "abstain": abstain,
            "keys": [f"opt{index}" for index in range(len(p))], "p": list(p)}


def noul_row(record: str, question: str, p, label: int) -> dict:
    return {"id": record, "question": question, "keys": ["false", "true"], "label": label,
            "p": list(p)}


def dump(path: Path, rows) -> Path:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


# ------------------------------------------------------------------ the rules are imported, not written
def test_the_readings_are_imported_from_the_v4_table(baron):
    """The module refuses to load if the settled rule has moved, rather than re-deriving it."""
    mods = baron.load_settled()
    for name in ("cell", "read_jsonl", "sha256_file", "fmt", "KEV_ARMS", "LAYA_ARMS"):
        assert hasattr(mods["settled"], name), f"the settled table no longer defines {name}"
    assert mods["settled"].__file__.replace("\\", "/").endswith("analysis/abstain-reconciliation-v4.py")
    # The arm list is the settled script's own, so an arm cannot exist in one corpus and not the other.
    assert len(mods["settled"].KEV_ARMS) == 12 and len(mods["settled"].LAYA_ARMS) == 2
    assert mods["laya_noul"].EXPECTED_QUESTIONS["baron"] == 39197
    assert mods["laya_noul"].EXPECTED_QUESTIONS["v4"] == 63851


def test_a_moved_rule_is_refused_rather_than_replaced(baron):
    with pytest.raises(baron.Refused, match="no longer defines"):
        baron.module_from(SCRIPT, "baron_self", ("cell",))


def test_the_three_readings_satisfy_the_identity_that_ties_them_together(baron, settled):
    """`forced_correct - served_correct == #{declined and forced-correct}`, on rows a decline can lose."""
    rows = [choice_row("r0", [0.6, 0.3, 0.1], 0, abstain=2),      # served right, decline not taken
            choice_row("r1", [0.2, 0.7, 0.1], 0, abstain=2),      # served wrong, forced wrong
            choice_row("r2", [0.1, 0.4, 0.5], 1, abstain=2),      # declines; forced would be right
            choice_row("r3", [0.1, 0.6, 0.3], 0, abstain=2)]      # served wrong, forced wrong
    block = baron.choice_arm_block(settled.cell, rows, lambda cell_fn, r: settled.cell(
        [row for row in r if row["label"] != row.get("abstain")]))
    assert (block["served"]["correct"], block["abstain_pick"]["picks"],
            block["forced"]["correct"]) == (1, 1, 2)
    assert block["hidden_correct"] == 1, "the one decline that was actually right"
    assert block["identity"]["holds"] is True
    assert block["identity"]["forced_correct_minus_served_correct"] == 1
    assert block["delta_points"] == pytest.approx(100 * 1 / 4)


def test_forced_is_retaken_without_removing_the_position_from_the_label(baron):
    """Forcing drops the abstain *position*; a winner's original index is compared to the label."""
    row = choice_row("r", [0.1, 0.2, 0.7], 2, abstain=0)
    assert settled_forced_index(baron, row) == 2, "the label is an original index, not a shifted one"
    row = choice_row("r", [0.7, 0.2, 0.1], 1, abstain=0)
    assert settled_forced_index(baron, row) == 1, "removing index 0 must not shift index 1 onto 0"


def settled_forced_index(baron, row) -> int:
    mods = baron.load_settled()
    return mods["settled"].forced_index(row)


# ------------------------------------------------------------------ the two denominators
def test_a_population_of_rows_without_an_abstain_option_reports_both_denominators(baron, settled):
    """On Baron `panel_choice` 75% of rows carry no abstain option: the two forced numbers differ."""
    rows = [choice_row("a", [0.9, 0.1], 0, abstain=None),     # no option: forced == served
            choice_row("b", [0.1, 0.9], 1, abstain=None),     # no option: right either way
            choice_row("c", [0.1, 0.2, 0.7], 0, abstain=2),   # declines; forced picks 1 -> wrong
            choice_row("d", [0.1, 0.6, 0.3], 1, abstain=2)]   # right, no decline
    block = baron.choice_arm_block(settled.cell, rows, lambda cell_fn, r: settled.cell(
        [row for row in r if row["label"] != row.get("abstain")]))
    assert block["n"] == 4 and block["abstain_pick"]["rows_offering_an_abstain_option"] == 2
    assert block["served"]["correct"] == 3
    assert block["forced"]["correct"] == 3, "over the population: the no-option rows are their own top1"
    assert block["over_the_rows_offering_an_abstain_option"]["n"] == 2
    assert block["over_the_rows_offering_an_abstain_option"]["forced"]["correct"] == 1, \
        "over the offering rows alone -- the denominator Jev's own scorer uses"
    assert block["abstain_pick"]["rate"] == pytest.approx(1 / 4)
    assert block["abstain_pick"]["rate_among_rows_offering_it"] == pytest.approx(1 / 2)


def test_when_every_row_offers_an_option_the_two_denominators_coincide(baron, settled):
    rows = [choice_row("a", [0.2, 0.7, 0.1], 1, abstain=2),
            choice_row("b", [0.1, 0.2, 0.7], 2, abstain=2)]
    block = baron.choice_arm_block(settled.cell, rows, lambda cell_fn, r: settled.cell(r))
    assert block["forced"]["correct"] == block["over_the_rows_offering_an_abstain_option"]["forced"]["correct"]
    assert block["abstain_pick"]["rate"] == block["abstain_pick"]["rate_among_rows_offering_it"]


# ------------------------------------------------------------------ panel_miss is a decline rate
def test_a_family_whose_label_is_the_abstain_option_is_reported_as_a_decline_rate(baron, settled):
    """Every row's label is the abstain option: served ``accuracy`` and the decline rate are one number."""
    rows = [choice_row("a", [0.1, 0.2, 0.7], 2, abstain=2, family="panel_miss"),   # declines
            choice_row("b", [0.2, 0.1, 0.7], 2, abstain=2, family="panel_miss"),   # declines
            choice_row("c", [0.7, 0.2, 0.1], 2, abstain=2, family="panel_miss")]   # names a cell
    block = baron.choice_arm_block(settled.cell, rows, lambda cell_fn, r: settled.cell(
        [row for row in r if row["label"] != row.get("abstain")]))
    assert block["rows_whose_label_is_the_abstain_option"] == 3
    assert block["share_whose_label_is_the_abstain_option"] == 1.0
    assert block["served"]["accuracy"] == block["abstain_pick"]["rate"] == pytest.approx(2 / 3), \
        "the same number, by construction -- quoting it as an accuracy is the mistake"
    assert block["forced"]["accuracy"] == 0.0, "removing the abstain option removes the only right answer"
    assert block["identity"]["holds"] is False, \
        "the raw identity cannot hold here, which is why the table reports it restricted as well"
    assert block["population_where_the_label_is_not_the_abstain_option"]["n"] == 0


# ------------------------------------------------------------------ marker_noul: the floor and the gate
def test_the_floor_is_computed_from_the_reference_and_the_margin_is_reported(baron, tmp_path):
    """Three questions of four are *no*: always-no scores 0.75 reading nothing."""
    rows = [noul_row("r0", "q0", [0.9, 0.1], 0),     # says no, reference no  -> right
            noul_row("r1", "q1", [0.2, 0.8], 0),     # says yes, reference no -> wrong
            noul_row("r2", "q2", [0.9, 0.1], 1),     # says no, reference yes -> wrong
            noul_row("r0", "q3", [0.9, 0.1], 0)]     # says no, reference no  -> right
    path = dump(tmp_path / "forced-rows-arm-marker_noul.jsonl", rows)
    mods = baron.load_settled()
    block = baron.noul_arm_block(mods["noul"], "arm", path, {"r0", "r1", "r2"}, 4)
    assert block["status"] == "complete"
    assert (block["questions"], block["correct"]) == (4, 2)
    assert block["majority_floor"] == 0.75 and block["reference"]["yes_rate"] == 0.25
    assert block["margin_over_floor"] == pytest.approx(0.5 - 0.75)
    assert block["says_yes"]["rate"] == pytest.approx(0.25), "answered yes once"
    assert block["says_yes_minus_reference_yes_rate"] == pytest.approx(0.0)
    assert block["forced_reading"].startswith("none"), "a yes/no question has no hidden reading"


def test_a_dump_short_of_the_question_count_is_not_scored(baron, tmp_path):
    """The 4B rung's top-up: a prefix is reported, never scored, and the complete dump beside it is."""
    rows = [noul_row(f"r{i}", f"q{i}", [0.9, 0.1], 0) for i in range(3)]
    short = dump(tmp_path / "short.jsonl", rows)
    mods = baron.load_settled()
    block = baron.noul_arm_block(mods["noul"], "short", short, {f"r{i}" for i in range(3)}, 10)
    assert block["status"] == "short"
    assert block["questions"] == 3 and block["questions_expected"] == 10
    assert "accuracy" not in block, "a prefix carries no accuracy in the cell"
    assert block["if_scored_as_a_prefix"]["correct"] == 3, "the arithmetic is kept, labelled"
    assert "NOT this family's number" in block["why"]

    complete = dump(tmp_path / "complete.jsonl", [noul_row(f"r{i}", f"q{i}", [0.9, 0.1], 0)
                                                  for i in range(10)])
    assert baron.noul_arm_block(mods["noul"], "complete", complete,
                                {f"r{i}" for i in range(10)}, 10)["status"] == "complete"


def test_a_partial_last_line_of_a_live_append_is_a_status_not_a_crash(baron, tmp_path):
    """A dump a running job is appending to can end mid-object; the table says so and moves on."""
    path = tmp_path / "appending.jsonl"
    path.write_text(json.dumps(noul_row("r0", "q0", [0.9, 0.1], 0)) + '\n{"id": "r1", "ques',
                    encoding="utf-8")
    mods = baron.load_settled()
    block = baron.noul_arm_block(mods["noul"], "appending", path, {"r0", "r1"}, 2)
    assert block["status"] == "unreadable" and "why" in block


def test_rows_outside_the_population_are_counted_rather_than_dropped(baron, tmp_path):
    rows = [noul_row("in", "q0", [0.9, 0.1], 0), noul_row("out", "q0", [0.9, 0.1], 0)]
    path = dump(tmp_path / "mixed.jsonl", rows)
    mods = baron.load_settled()
    block = baron.noul_arm_block(mods["noul"], "mixed", path, {"in"}, 2)
    assert block["rows_outside_the_population"] == 1 and block["questions"] == 1


# ------------------------------------------------------------------ the naming rules
def test_the_dump_name_carries_the_corpus_infix_and_the_family_tag(baron):
    """Baron ``forced-rows-<slug>-baron[-<family>].jsonl``; v4 has no infix. Never renamed across."""
    panel = "reports/forced-rows-delta-kev-0.8b-v4-s1.jsonl"
    assert baron.choice_dump_path(panel, "baron", "panel_choice") == \
        "reports/forced-rows-delta-kev-0.8b-v4-s1-baron.jsonl"
    assert baron.choice_dump_path(panel, "baron", "panel_miss") == \
        "reports/forced-rows-delta-kev-0.8b-v4-s1-baron-panel_miss.jsonl"
    assert baron.choice_dump_path(panel, "v4", "panel_choice") == panel
    assert baron.choice_dump_path(panel, "v4", "evidence_diluted") == \
        "reports/forced-rows-delta-kev-0.8b-v4-s1-evidence_diluted.jsonl"
    with pytest.raises(baron.Refused, match="not a .jsonl dump path"):
        baron.choice_dump_path("reports/forced-rows-arm.json", "baron", "panel_choice")


def test_the_laya_half_is_read_out_of_the_settled_path(baron):
    assert baron.laya_half("runs/laya-v4-census-spec/reconcile.json") == "spec"
    assert baron.laya_half("runs/laya-baron-census-base-evidence_diluted/reconcile.json") == "base"
    with pytest.raises(baron.Refused, match="cannot read the Laya half"):
        baron.laya_half("runs/other/reconcile.json")


def test_the_panel_choice_run_directory_is_a_search_not_a_guess(baron):
    """v4 reconciles it into ``…-census-<half>/``, Baron into ``…-census-<half>-panel_choice/``."""
    assert baron.laya_run_path("baron", "spec", "panel_choice")[0] == \
        "runs/laya-baron-census-spec/reconcile.json"
    assert baron.laya_run_path("baron", "spec", "panel_choice")[1] == \
        "runs/laya-baron-census-spec-panel_choice/reconcile.json"
    assert baron.laya_run_path("v4", "spec", "panel_choice")[1] == \
        "runs/laya-v4-census-spec-panel_choice/reconcile.json"
    assert baron.laya_run_path("baron", "base", "panel_miss") == \
        ["runs/laya-baron-census-base-panel_miss/reconcile.json"]


# ------------------------------------------------------------------ the controls are the gate
def _result_with(arm_blocks: dict, noul_arms: dict | None = None) -> dict:
    """A minimal result shaped as `apply_controls` reads it, so the gate can be fired on purpose."""
    return {
        "families": {
            "panel_choice": {"arms": {}, "baselines": {
                "jev": {"status": "complete", "served": {"accuracy": 0.6123}},
                "llm": {"variants": {"varA": {"status": "complete", "accuracy": 0.4146},
                                     "varB": {"status": "complete", "accuracy": 0.5523}}}}},
            "evidence_diluted": {"arms": {}, "baselines": {"jev": {"status": "absent"},
                                                           "llm": {"variants": {}}}},
            "panel_miss": {"arms": {}, "baselines": {"jev": {"status": "absent"},
                                                     "llm": {"variants": {}}}},
            "marker_noul": {"arms": noul_arms or {},
                            "baselines": {"jev": {"status": "complete", "accuracy": 0.8879}}},
        },
    }


def test_every_control_reproduces_when_the_numbers_are_the_published_ones(baron):
    l1, l2, l3 = "laya-v4-ft-spec", "laya-v4-base", "x"
    result = _result_with({}, {l1: {"status": "complete", "accuracy": 0.9064},
                               l2: {"status": "complete", "accuracy": 0.1362},
                               l3: {"status": "absent"}})
    baron.apply_controls(result)
    assert result["controls"]["n_checks"] == 6
    assert result["controls"]["n_reproduced"] == 6
    assert all(check["reproduced"] for check in result["controls"]["checks"])


@pytest.mark.parametrize("control_id, wrong", [
    ("jev.panel_choice.served", 0.65),
    ("laya.base.marker_noul.accuracy", 0.9064),
    ("llm.panel_choice.varB", 0.5524),
])
def test_a_control_that_moved_is_reported_as_a_mismatch(baron, control_id, wrong):
    """The gate can fail: a reader that shifted one number must not report the control as reproduced."""
    l1, l2 = "laya-v4-ft-spec", "laya-v4-base"
    result = _result_with({}, {l1: {"status": "complete", "accuracy": 0.9064},
                               l2: {"status": "complete", "accuracy": wrong}})
    if control_id.startswith("jev"):
        result["families"]["panel_choice"]["baselines"]["jev"]["served"]["accuracy"] = wrong
    if control_id.startswith("llm"):
        result["families"]["panel_choice"]["baselines"]["llm"]["variants"]["varB"]["accuracy"] = wrong
    baron.apply_controls(result)
    check = next(entry for entry in result["controls"]["checks"] if entry["id"] == control_id)
    assert check["reproduced"] is False
    assert result["controls"]["n_reproduced"] < result["controls"]["n_checks"]


# ------------------------------------------------------------------ the committed table cannot rot
def test_the_committed_table_carries_every_arm_family_and_its_controls():
    assert COMMITTED.is_file(), "reports/baron-reconciliation.json is the deliverable"
    report = json.loads(COMMITTED.read_text(encoding="utf-8"))
    assert report["corpus"]["selector"] == "baron"
    assert report["population"]["rows"] == 8450
    assert report["population"]["families"] == {"evidence_diluted": 2600, "marker_noul": 2600,
                                                "panel_choice": 2600, "panel_miss": 650}
    assert report["controls"]["n_reproduced"] == report["controls"]["n_checks"] == 6, \
        "the committed table ships only when every control reproduced"
    arms = set(report["families"]["panel_choice"]["arms"])
    assert len(arms) == 14, "six fine-tunes, six released rungs, two Laya halves"
    for family in ("panel_choice", "evidence_diluted", "panel_miss", "marker_noul"):
        assert set(report["families"][family]["arms"]) == arms, f"{family} is missing an arm"
    assert report["status"]["short"] == [], "the 4B rung's top-up is complete in the shipped table"
    assert len(report["status"]["complete"]) == 56


def test_the_committed_table_states_the_things_a_reader_would_otherwise_infer():
    report = json.loads(COMMITTED.read_text(encoding="utf-8"))
    noul = report["families"]["marker_noul"]
    assert round(noul["floor"], 4) == 0.8708 and noul["floor"] != 0.8680, \
        "Baron's floor is not v4's, and the two must never be interchanged"
    assert noul["unit"] == "questions" and noul["questions"] == 39197
    for arm, cell in noul["arms"].items():
        assert cell["questions"] == 39197 and "forced_reading" in cell
    miss = report["families"]["panel_miss"]
    assert "DECLINE RATE" in miss["checks"]["reading"]
    for arm, cell in miss["arms"].items():
        assert cell["served"]["accuracy"] == cell["abstain_pick"]["rate"], \
            f"{arm}: on panel_miss the served number IS the decline rate"
        assert cell["forced"]["accuracy"] == 0.0
    choice = report["families"]["panel_choice"]
    for arm, cell in choice["arms"].items():
        assert cell["abstain_pick"]["rows_offering_an_abstain_option"] == 650
        assert cell["over_the_rows_offering_an_abstain_option"]["n"] == 650, \
            f"{arm}: the eligible-restricted reading must be present beside the population one"
    llm = choice["baselines"]["llm"]["variants"]
    for cell in llm.values():
        assert cell["forced_reading"].startswith("none"), "a text answer has no probability vector"
        assert "forced" not in cell, "and no forced column may be invented for it"


def test_the_committed_table_carries_the_v4_proof_that_it_was_not_disturbed():
    report = json.loads(COMMITTED.read_text(encoding="utf-8"))
    identity = report.get("v4_byte_identity")
    assert identity is not None, "the shipped table records that v4 was re-run and left alone"
    same = [entry for entry in identity["checks"] if entry["identical"]]
    assert any(entry["frozen"] == "reports/abstain-reconciliation-v4.json" for entry in same), \
        "the settled v4 table reproduces byte for byte"
    for entry in identity["checks"]:
        if entry["identical"]:
            continue
        diff = entry["leaf_diff"]
        assert diff["changed"], "a byte difference must be explained at leaf level, not just flagged"
        assert set(diff["changed"]) <= {"best_worst.order", "rules_from.sha256", "checks.identity.n_arms",
                                        "checks.identity.n_violating",
                                        "checks.identity_where_a_decline_cannot_be_right.n_arms",
                                        ("checks.identity_where_a_decline_cannot_be_right."
                                         "n_arms_with_an_empty_subpopulation")}, \
            f"{entry['frozen']}: a cell moved, which is drift and not a stale snapshot"
