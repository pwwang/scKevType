"""The forced-naming aggregation: the rule taken from the file that settled it, and its refusals.

`scripts/arms-forced.py` reports our fine-tuned arms' served / abstain-pick / forced numbers on the
1,819 census-matched `panel_choice` questions. Two of its properties are load-bearing and are checked
here on synthetic rows, with no GPU and no cluster:

* the **rule is not re-derived** — `forced`, `argmax` and `binom_two_sided` are executed from the
  parsed source of `reports/abstain-reconciliation.py`, and that file's module body (which would
  rewrite `reports/abstain-reconciliation.json`) is *not* run;
* the **population and the joins** are the settled ones, and a re-score that does not reproduce the
  archived row is reported as such rather than averaged in.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"


@pytest.fixture(scope="module")
def arms_forced():
    spec = importlib.util.spec_from_file_location("arms_forced", SCRIPTS / "arms-forced.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------- the rule is taken, not re-written
def test_the_rule_comes_from_the_file_that_settled_it(arms_forced):
    rule = arms_forced.load_rule()
    assert set(rule) == {"argmax", "forced", "binom_two_sided"}

    # argmax: the lowest index among equal maxima (the tie rule every row dump uses)
    assert rule["argmax"]([0.5, 0.5, 0.1]) == 0
    assert rule["argmax"]([0.1, 0.4, 0.4]) == 1

    # forced: the argmax over p with the abstain option removed
    assert rule["forced"]({"abstain": 2, "p": [0.1, 0.2, 0.7]}) == 1
    assert rule["forced"]({"abstain": 0, "p": [0.9, 0.2, 0.3]}) == 2
    # ...and a row that offers no abstain option has nothing to remove
    assert rule["forced"]({"abstain": None, "p": [0.1, 0.9]}) == 1

    # the exact two-sided binomial on discordant pairs, hand-computed
    assert rule["binom_two_sided"](0, 0) == 1.0
    assert rule["binom_two_sided"](5, 0) == pytest.approx(2 * 1 / 2 ** 5)
    assert rule["binom_two_sided"](6, 0) == pytest.approx(2 / 2 ** 6)
    assert rule["binom_two_sided"](3, 3) == 1.0


def test_loading_the_rule_does_not_execute_the_report_script(tmp_path, arms_forced):
    """The whole reason for parsing instead of importing: the module body rewrites a report."""
    path = tmp_path / "rule.py"
    side_effect = tmp_path / "side-effect"
    path.write_text(f"import pathlib\npathlib.Path({str(side_effect)!r}).write_text('boom')\n"
                    "def forced(row):\n    return 7\n", encoding="utf-8")

    rule = arms_forced.load_rule(path, ("forced",))
    assert rule["forced"]({}) == 7
    assert not side_effect.exists(), "the module body must not run"


def test_a_missing_definition_is_refused(tmp_path, arms_forced):
    path = tmp_path / "rule.py"
    path.write_text("def argmax(values):\n    return 0\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="forced"):
        arms_forced.load_rule(path, ("argmax", "forced"))


# --------------------------------------------------------------------- the population is the settled one
@pytest.mark.skipif(not (ROOT / "reports" / "census-suite-ids.jsonl").exists(),
                    reason="the census-matched suite ids are not on disk")
def test_the_population_is_the_settled_1819(arms_forced):
    ids, rows = arms_forced.population()
    assert len(ids) == 1819 and len(set(ids)) == 1819
    assert all(row["family"] == "panel_choice" for row in rows)
    assert all(row["id"].startswith("test/") for row in rows)
    assert all(isinstance(i, str) and i.startswith("test/") for i in ids)


# --------------------------------------------------------------------- the arithmetic, on synthetic dumps
def _row(index, p, label, *, abstain=2, top1=None, keys=("T cell", "B cell", "other"), **extra):
    top1 = max(range(len(p)), key=lambda i: p[i]) if top1 is None else top1
    row = {"id": f"test/custom/{index}", "keys": list(keys), "label": label, "p": list(p),
           "abstain": abstain, "top1": top1, "top1_p": p[top1]}
    row.update(extra)
    return row


def _write(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


ARM = {"arm": "frozen", "key": "ours-frozen", "label": "0.8B from-base — frozen",
       "class": "ours, from-base", "initialization": "base checkpoint"}


def test_the_five_columns_are_the_counts_beside_them(tmp_path, arms_forced):
    """Three rows, hand-computed: served 1/3, abstain 1/3, forced 2/3, one hidden correct."""
    archived = [_row(0, [0.5, 0.2, 0.3], 0),          # served right, nothing hidden
                _row(1, [0.1, 0.2, 0.7], 1),          # declines; forcing recovers it
                _row(2, [0.2, 0.6, 0.2], 0)]          # wrong either way
    rescored = archived
    _write(tmp_path / "archived.jsonl", archived)
    _write(tmp_path / "rescored.jsonl", rescored)

    row = arms_forced.measure(ARM, [r["id"] for r in archived], arms_forced.load_rule(),
                              archived_path=str(tmp_path / "archived.jsonl"),
                              rescored_path=str(tmp_path / "rescored.jsonl"))
    cell = row["census_matched_1819"]
    assert cell["n"] == 3
    assert cell["served"]["correct"] == 1 and cell["served"]["archived_top1_correct"] == 1
    assert cell["abstain_pick"]["picks"] == 1
    assert cell["forced"]["correct"] == 2
    assert cell["questions_gained"] == 1
    assert cell["delta_points"] == pytest.approx((2 - 1) / 3 * 100)
    assert cell["hidden_correct_share_of_abstentions"] == pytest.approx(1.0)
    assert row["fidelity"]["verdict"] == "identical rows"


def test_a_rescore_that_is_not_the_archived_row_is_reported(tmp_path, arms_forced):
    """The negative control: one perturbed `top1` must surface, not be averaged in."""
    archived = [_row(0, [0.5, 0.2, 0.3], 0), _row(1, [0.1, 0.2, 0.7], 1)]
    rescored = [_row(0, [0.5, 0.2, 0.3], 0), _row(1, [0.1, 0.2, 0.7], 1, top1=0)]
    _write(tmp_path / "archived.jsonl", archived)
    _write(tmp_path / "rescored.jsonl", rescored)

    row = arms_forced.measure(ARM, [r["id"] for r in archived], arms_forced.load_rule(),
                              archived_path=str(tmp_path / "archived.jsonl"),
                              rescored_path=str(tmp_path / "rescored.jsonl"))
    assert row["fidelity"]["top1_mismatch"]["count"] == 1
    assert row["fidelity"]["top1_mismatch"]["ids"] == ["test/custom/1"]
    assert row["fidelity"]["verdict"] != "identical rows"


def test_a_short_population_is_refused_not_averaged(tmp_path, arms_forced):
    """A re-scored dump missing a row cannot report a number on a smaller population."""
    archived = [_row(0, [0.5, 0.2, 0.3], 0), _row(1, [0.1, 0.2, 0.7], 1)]
    _write(tmp_path / "archived.jsonl", archived)
    _write(tmp_path / "rescored.jsonl", archived[:1])

    row = arms_forced.measure(ARM, [r["id"] for r in archived], arms_forced.load_rule(),
                              archived_path=str(tmp_path / "archived.jsonl"),
                              rescored_path=str(tmp_path / "rescored.jsonl"))
    assert "not_measured" in row["census_matched_1819"]
    assert "not covered" in row["census_matched_1819"]["not_measured"]


def test_the_abstain_index_must_agree_with_the_row(tmp_path, arms_forced):
    """`abstain` is a property of the question; a dump that disagrees with it is not measured on."""
    archived = [_row(0, [0.5, 0.2, 0.3], 0)]
    rescored = [_row(0, [0.5, 0.2, 0.3], 0, abstain=1)]
    _write(tmp_path / "archived.jsonl", archived)
    _write(tmp_path / "rescored.jsonl", rescored)

    row = arms_forced.measure(ARM, [r["id"] for r in archived], arms_forced.load_rule(),
                              archived_path=str(tmp_path / "archived.jsonl"),
                              rescored_path=str(tmp_path / "rescored.jsonl"))
    assert row["fidelity"]["abstain_index_disagreement"]["count"] == 1
    assert row["fidelity"]["verdict"] != "identical rows"


# --------------------------------------------------------------------- the whole script, end to end
@pytest.mark.skipif(not (ROOT / "reports" / "census-suite-ids.jsonl").exists(),
                    reason="the census-matched suite ids are not on disk")
def test_the_script_runs_end_to_end(tmp_path, arms_forced):
    """Runs on whatever dumps are on disk (a missing re-score is reported, not crashed on)."""
    out = tmp_path / "arms-forced.json"
    md = tmp_path / "arms-forced.md"
    assert arms_forced.main(["--json", str(out), "--md", str(md), "--job-ids", "1,2"]) == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["populations"]["census_matched_1819"]["n"] == 1819
    assert len(payload["engines"]) == len(arms_forced.ARMS) == 11
    assert payload["provenance"]["job_ids"] == ["1", "2"]
    text = md.read_text(encoding="utf-8")
    assert "forced accuracy" in text and "§2.5" in text
