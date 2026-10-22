"""The paired tests' arithmetic, their refusals, and the report they produce.

`reports/paired-tests-v4.py` adds what `reports/abstain-reconciliation-v4.md` left out: an exact
paired test under every `panel_choice` comparison the paper makes. Six properties are load-bearing,
and each is checked here on hand-built data — no GPU, no cluster, no dump on the network share:

* **the rule is taken, not re-derived** — the exact two-sided binomial comes from the parsed source
  of `reports/abstain-reconciliation.py`, the file that settled it, whose module body would rewrite
  `reports/abstain-reconciliation.json` and is therefore never executed;
* **the arm list has one home** — `KEV_ARMS`/`LAYA_ARMS` are lifted out of
  `reports/abstain-reconciliation-v4.py` by `ast`, so a table built here cannot drift from the table
  it extends, and a renamed name is refused by name instead of yielding an empty roster;
* **the cells are the discordant pairs** — `both / A only / B only / both wrong` come from the two
  arms' verdicts record by record, and two arms that do not cover the same records are refused
  rather than compared;
* **the forced reading really removes the abstain option** — a record where the model declined is
  wrong served and can be right forced; a record served with no abstain option has one verdict;
* **the correction is Holm in log space** — a family of p-values is adjusted step-up, no adjusted
  value is below its raw value, and the family size travels with each row;
* **the published table re-derives from its own cells** — every p-value in
  `reports/paired-tests-v4.json` is `binom_two_sided(b, c)` on the counts printed beside it, so a
  hand-edit or a stale run cannot survive.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
REPORT_JSON = REPORTS / "paired-tests-v4.json"
REPORT_MD = REPORTS / "paired-tests-v4.md"


@pytest.fixture(scope="module")
def paired():
    spec = importlib.util.spec_from_file_location("paired_tests_v4", ROOT / "analysis" / "paired-tests-v4.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def rule(paired):
    return paired.rule_from(paired.RULE_SOURCE, ("binom_two_sided", "binom_two_sided_log10"))


def _row(record: str, correct: bool, label: int = 0, keys=("alpha", "beta", "unknown"),
         abstain: int | None = 2, weight: float = 0.9) -> dict:
    """A choice dump row whose verdict is ``correct`` — one option held at ``weight``."""
    size = len(keys)
    probabilities = [(1.0 - weight) / (size - 1)] * size
    probabilities[label if correct else (label + 1) % (size - 1)] = weight
    return {"id": record, "family": "panel_choice", "type": "choice", "keys": list(keys),
            "label": label, "abstain": abstain, "p": probabilities}


def _dump(path: Path, rows) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def _verdicts(*flags, label: int = 0) -> dict:
    """``{id: {"label", "served"}}`` from a sequence of served verdicts."""
    return {f"r{index}": {"label": label, "served": bool(flag)} for index, flag in enumerate(flags)}


# --------------------------------------------------------------- the rule is taken, not re-written
def test_the_paired_test_comes_from_the_file_that_settled_it(rule):
    """``binom_two_sided`` is executed from that file's source; the module body never runs."""
    assert rule["binom_two_sided"](0, 0) == 1.0, "no discordant pairs, no evidence"
    assert rule["binom_two_sided"](5, 5) == 1.0, "a symmetrical split is the null itself"
    assert rule["binom_two_sided"](10, 0) == pytest.approx(2 * 0.5 ** 10)
    assert rule["binom_two_sided"](0, 10) == rule["binom_two_sided"](10, 0), "two-sided by construction"
    assert rule["binom_two_sided"](3, 20) == rule["binom_two_sided"](20, 3), "the pair's order is not a variable"
    assert rule["binom_two_sided_log10"](10, 0) == pytest.approx(math.log10(rule["binom_two_sided"](10, 0)))


def test_a_moved_rule_is_refused_rather_than_replaced(paired):
    """If the definitions this report applies disappear, it must fail — not silently apply a copy."""
    with pytest.raises(ValueError, match="no longer defines"):
        paired.rule_from(ROOT / "analysis" / "paired-tests-v4.py", ("binom_two_sided",))


# --------------------------------------------------------------- the arm list has one home
def test_the_arm_list_is_lifted_from_the_settled_table(paired):
    """The roster is read out of the file it extends, so the two tables cannot list different arms."""
    constants = paired.lift_constants(paired.ARM_SOURCE, ("KEV_ARMS", "LAYA_ARMS"))
    kev, laya = constants["KEV_ARMS"], constants["LAYA_ARMS"]
    assert len(kev) == 12 and len(laya) == 2, "6 fine-tunes, 6 rungs, and the two Laya halves"
    assert all(len(item) == 3 for item in kev + laya), "every arm is (slug, shown_as, path)"
    slugs = {item[0] for item in kev}
    assert {"delta-kev-0.5b-v4-s1", "kev-0.5b-v4", "kev-27b-v4"} <= slugs
    assert {item[0] for item in laya} == {"laya-v4-ft-spec", "laya-v4-base"}


def test_a_renamed_arm_list_is_refused_by_name(paired):
    """A name that moved must fail loudly: an empty roster would print an empty table, not an error."""
    with pytest.raises(ValueError, match="no longer declares"):
        paired.lift_constants(paired.ARM_SOURCE, ("ARMS_THAT_DO_NOT_EXIST",))

# --------------------------------------------------------------- McNemar's cells, and its refusals
def test_mcnemar_counts_the_four_cells_and_tests_the_discordant_pairs(rule, paired):
    """The 2x2 is built record by record; the test is on `b` and `c` only, never on two accuracies."""
    left = _verdicts(True, True, True, True, True, True, True, False, False, False)
    right = _verdicts(True, True, True, True, False, False, False, True, False, False)
    block = paired.mcnemar(left, right, "served", rule)

    assert block["contingency"] == {"both_correct": 4, "left_only": 3,
                                    "right_only": 1, "both_wrong": 2}
    assert block["discordant"] == 4, "only the records the two arms disagree on enter the test"
    assert block["accuracy_left"] == pytest.approx(0.7)
    assert block["accuracy_right"] == pytest.approx(0.5)
    assert block["difference_points"] == pytest.approx(20.0)
    assert block["p_value"] == rule["binom_two_sided"](3, 1)
    assert block["p_value_log10"] == rule["binom_two_sided_log10"](3, 1)


def test_two_arms_that_do_not_cover_the_same_records_are_refused(rule, paired):
    """A paired test over different populations is not a conservative test — it is a different test."""
    left = _verdicts(True, False, True, False)
    right = {"r0": {"label": 0, "served": True}, "r1": {"label": 0, "served": True}}
    with pytest.raises(SystemExit, match="not on the same records"):
        paired.mcnemar(left, right, "served", rule)


def test_a_reading_an_arm_does_not_have_is_not_silently_substituted(rule, paired):
    """The LLM's rows carry served verdicts only; asking for forced must fail, not return served."""
    left = {"r0": {"label": 0, "served": True}}
    right = {"r0": {"label": 0, "served": True}}
    with pytest.raises(KeyError):
        paired.mcnemar(left, right, "forced", rule)


# --------------------------------------------------------------- the readings
def test_forced_removes_the_abstain_option_and_served_does_not(paired):
    """The decline: the model named the abstain option, so served is wrong and forced can be right."""
    row = {"keys": ["alpha", "beta", "unknown"], "label": 0, "abstain": 2,
           "p": [0.3, 0.1, 0.6]}
    assert paired.verdicts(row, "served") is False, "argmax is the abstain option"
    assert paired.verdicts(row, "forced") is True, "drop it and the reference's option wins"


def test_a_record_served_without_an_abstain_option_has_one_verdict(paired):
    """Nothing to remove: forced *is* served, on all 2,992 such records of the real population."""
    row = {"keys": ["alpha", "beta"], "label": 0, "abstain": None, "p": [0.9, 0.1]}
    assert paired.verdicts(row, "served") is True
    assert paired.verdicts(row, "forced") is True


def test_the_abstain_option_is_found_by_its_key_not_by_the_row_field(paired):
    """`sckevtype.eval.abstain_index` is the rule the Laya lane uses; a row that names no `abstain`
    field but carries the key must still have the option removed."""
    row = {"keys": ["alpha", "beta", "unknown"], "label": 1, "p": [0.1, 0.3, 0.6]}
    assert paired.verdicts(row, "served") is False, "the abstain option wins (0.6)"
    assert paired.verdicts(row, "forced") is True, "removed by its key, the reference's option wins"


# --------------------------------------------------------------- Holm
def test_holm_is_the_step_up_of_the_bonferroni_penalty(paired):
    """m = 3, ranks 1..3: adjusted = max over j <= rank of (m - j + 1) * p_j."""
    entries = [{"p_value": 0.001, "p_value_log10": math.log10(0.001)},
               {"p_value": 0.02, "p_value_log10": math.log10(0.02)},
               {"p_value": 0.5, "p_value_log10": math.log10(0.5)}]
    paired.holm(entries)

    assert [entry["holm"]["rank"] for entry in entries] == [1, 2, 3]
    assert entries[0]["holm"]["p_value"] == pytest.approx(0.003)
    assert entries[1]["holm"]["p_value"] == pytest.approx(0.04)
    assert entries[2]["holm"]["p_value"] == pytest.approx(0.5), "the largest is unchanged"
    assert {entry["holm"]["family_size"] for entry in entries} == {3}


def test_holm_never_lowers_a_p_value_and_survives_the_underflow(paired):
    """A p-value below `5e-324` is `0.0` as a float; the adjustment is taken on its log10 instead."""
    entries = [{"p_value": 0.0, "p_value_log10": -400.0},
               {"p_value": 0.01, "p_value_log10": math.log10(0.01)}]
    paired.holm(entries)

    for entry in entries:
        assert entry["holm"]["p_value"] >= entry["p_value"]
        assert entry["holm"]["p_value_log10"] <= entry["p_value_log10"] + math.log10(2) + 1e-12
    assert entries[0]["holm"]["p_value"] == 0.0, "still below what a float holds, and still reported"
    assert entries[0]["holm"]["p_value_log10"] == pytest.approx(-400.0 + math.log10(2))
    assert entries[1]["holm"]["p_value"] == pytest.approx(0.01)
    assert math.isfinite(entries[1]["holm"]["p_value_log10"])


# --------------------------------------------------------------- the roster's derivations
def _arm(slug: str, kind: str = "kev") -> dict:
    return {"slug": slug, "shown_as": slug, "kind": kind, "declared": slug,
            "canonical": slug, "read": Path(slug), "readings": ("served", "forced")}


def test_the_ladder_is_derived_from_the_arm_list_and_ordered_by_size(paired):
    arms = {slug: _arm(slug) for slug in ("delta-kev-9b-v4-s1", "kev-9b-v4",
                                          "delta-kev-0.5b-v4-s1", "kev-0.5b-v4")}
    assert [(size, left, right) for size, left, right in paired.ladder(arms)] == [
        ("0.5b", "delta-kev-0.5b-v4-s1", "kev-0.5b-v4"),
        ("9b", "delta-kev-9b-v4-s1", "kev-9b-v4"),
    ], "ordered by parameter count, and the rung is derived from the fine-tune's own slug"


def test_a_fine_tune_without_its_rung_is_refused_rather_than_dropped(paired):
    """A missing rung must stop the run: a silently absent pair reads like a pair that was tested."""
    arms = {"delta-kev-4b-v4-s1": _arm("delta-kev-4b-v4-s1")}
    with pytest.raises(SystemExit, match="no rung"):
        paired.ladder(arms)


def test_the_laya_halves_are_the_fine_tune_and_the_base_by_name(paired):
    arms = {"laya-v4-base": _arm("laya-v4-base", "laya"),
            "laya-v4-ft-spec": _arm("laya-v4-ft-spec", "laya")}
    assert paired.laya_pair(arms) == ("laya-v4-ft-spec", "laya-v4-base")
    with pytest.raises(SystemExit, match="one fine-tune and one base"):
        paired.laya_pair({"laya-a-base": _arm("laya-a-base", "laya")})

# --------------------------------------------------------------- end to end, on a hand-built population
def _population(tmp_path: Path, records: int = 4) -> tuple[Path, dict]:
    """A census file and the ids it names, in the shapes the real dumps use."""
    rows = [{"id": f"test/custom/{index}", "dataset": "synthetic", "cell": f"c{index}",
             "family": "panel_choice", "variant": "clean"} for index in range(records)]
    rows.append({"id": "test/custom/999", "dataset": "synthetic", "cell": "c0",
                 "family": "marker_noul", "variant": "clean"})
    census = tmp_path / "census.jsonl"
    _dump(census, rows)
    triples = {(row["dataset"], row["cell"], row["variant"]): row["id"]
               for row in rows if row["family"] == "panel_choice"}
    return census, triples


def _synthetic_tree(tmp_path: Path, paired) -> dict:
    """Every arm the roster names, on four records, with a known discordance for each comparison."""
    census, triples = _population(tmp_path)
    ids = sorted(triples.values())
    truth = lambda index: index < 3          # the fine-tunes are right on 0, 1, 2 and wrong on 3
    dumps = tmp_path / "dumps"
    runs = tmp_path / "runs"
    baselines = tmp_path / "baselines"

    for slug, _shown, _path in paired.lift_constants(paired.ARM_SOURCE, ("KEV_ARMS",))["KEV_ARMS"]:
        right = truth if slug.startswith("delta-kev-") else (lambda index: False)
        _dump(dumps / f"forced-rows-{slug}.jsonl",
              [_row(record, right(index)) for index, record in enumerate(ids)])

    spec = paired.lift_constants(paired.ARM_SOURCE, ("LAYA_ARMS",))["LAYA_ARMS"]
    for slug, _shown, _path in spec:
        folder = runs / ("laya-v4-census-spec" if "ft" in slug else "laya-v4-census-base")
        right = (lambda index: index == 0) if "ft" in slug else (lambda index: False)
        rows = [dict(_row(record, right(index)), ask="base") for index, record in enumerate(ids)]
        rows.append(dict(_row(ids[0], True), ask="abstain"))     # the second ask, ignored by the join
        _dump(folder / "rows" / "census-000000.jsonl", rows)

    def _baseline(path: Path, correct: int, with_p: bool) -> None:
        rows = []
        for index, (triple, record) in enumerate(triples.items()):
            dataset, cell, variant = triple
            row = {"family": "panel_choice", "task": "panel_choice", "dataset": dataset,
                   "cell": cell, "variant": variant, "label": 0,
                   "hit": int(index < correct)}
            if with_p:
                row.update({"keys": ["alpha", "beta", "unknown"], "abstain": 2,
                            "p": [0.7, 0.2, 0.1]})
            rows.append(row)
        _dump(path, rows)

    _baseline(baselines / "jev-v4" / "rows.jsonl", 2, True)
    _baseline(baselines / "llm-baseline-v4-rows-census-varA.jsonl", 1, False)
    _baseline(baselines / "llm-baseline-v4-rows-census-varB.jsonl", 1, False)

    return {"census": census, "dumps": dumps, "runs": runs, "baselines": baselines}


def test_the_whole_table_is_produced_on_hand_built_dumps(tmp_path, paired):
    """End to end: the table, its cells, its correction and its refusals, on four records.

    The ladder row is the one to read: the fine-tune is right on three records the rung is wrong on,
    so `b = 3, c = 0` and the exact two-sided binomial is `2 * 0.5 ** 3` — not a difference of two
    accuracies.
    """
    tree = _synthetic_tree(tmp_path, paired)
    out, markdown = tmp_path / "paired.json", tmp_path / "paired.md"

    rc = paired.main(["--dumps", str(tree["dumps"]), "--runs", str(tree["runs"]),
                      "--baselines", str(tree["baselines"]), "--census", str(tree["census"]),
                      "--out", str(out), "--markdown", str(markdown)])
    report = json.loads(out.read_text())
    text = markdown.read_text()

    assert rc == 0
    assert report["population"]["rows"] == 4, "the marker_noul row of the same file is not the population"
    assert len(report["comparisons"]) == 46, "12 ladder + 2 sibling + 14 Jev + 14 LLM + 4 exploratory"

    ladder = next(entry for entry in report["comparisons"]
                  if entry["group"] == "ladder" and entry["reading"] == "served"
                  and entry["left"]["arm"] == "delta-kev-0.5b-v4-s1")
    assert ladder["contingency"] == {"both_correct": 0, "left_only": 3,
                                     "right_only": 0, "both_wrong": 1}
    assert ladder["p_value"] == pytest.approx(2 * 0.5 ** 3)
    assert ladder["holm"] is not None and ladder["holm"]["family_size"] == 28

    exploratory = [entry for entry in report["comparisons"] if entry["exploratory"]]
    assert len(exploratory) == 4 and all(entry["holm"] is None for entry in exploratory), \
        "an exploratory row is reported raw, and says so instead of borrowing a family"

    assert report["checks"]["reproduces_the_settled_table"]["checked"] is False, \
        "a synthetic population is not the census, and the check says so rather than passing quietly"
    assert "not run" in text

    assert [block["reading"] for block in report["not_testable"]] == ["forced"]
    assert set(report["not_testable"][0]["arms"]) == {"llm-a", "llm-b"}
    assert "| fine-tuned 0.5B | Kev-0.5B zero-shot | served |" in text
    assert "## Multiplicity" in text and "Holm" in text


# --------------------------------------------------------------- the published table does not rot
@pytest.mark.skipif(not REPORT_JSON.exists(), reason="the report has not been generated on this checkout")
def test_the_published_table_rederives_from_its_own_cells(rule):
    """Every p-value in the report is `binom_two_sided(b, c)` on the counts printed beside it.

    This is the anti-rot check on the artifact: a hand-edited p, a stale table after an arm moved, or
    a comparison dropped from the markdown all fail here, because the report has to agree with
    itself and with the markdown that ships beside it.
    """
    report = json.loads(REPORT_JSON.read_text())
    text = REPORT_MD.read_text() if REPORT_MD.exists() else ""
    comparisons = report["comparisons"]
    arms = {arm["arm"] for arm in report["arms"]}

    assert report["population"]["rows"] == 4035 and report["population"]["family"] == "panel_choice"
    assert len(comparisons) == 46, "a comparison leaving the table is a regression, not a tidy-up"

    for entry in comparisons:
        cells = entry["contingency"]
        total = sum(cells.values())
        assert total == entry["n"], f"{entry['comparison']}: the 2x2 does not cover the population"
        assert entry["discordant"] == cells["left_only"] + cells["right_only"]
        assert entry["accuracy_left"] == pytest.approx((cells["both_correct"] + cells["left_only"]) / total)
        assert entry["accuracy_right"] == pytest.approx((cells["both_correct"] + cells["right_only"]) / total)
        assert entry["p_value"] == rule["binom_two_sided"](cells["left_only"], cells["right_only"]), \
            f"{entry['comparison']} ({entry['reading']}): the p-value is not the test on its own cells"
        assert entry["p_value_log10"] == rule["binom_two_sided_log10"](
            cells["left_only"], cells["right_only"])
        assert entry["left"]["arm"] in arms and entry["right"]["arm"] in arms
        if entry["holm"] is not None:
            assert entry["holm"]["p_value_log10"] >= entry["p_value_log10"] - 1e-9, \
                "the correction may not lower a p-value"
        assert entry["left"]["shown_as"] in text and entry["right"]["shown_as"] in text, \
            "every comparison in the JSON is a row of the markdown"

    ladder = [entry for entry in comparisons if entry["group"] == "ladder"]
    assert {entry["left"]["arm"] for entry in ladder} == {
        f"delta-kev-{size}-v4-s1" for size in ("0.5b", "0.6b", "0.8b", "4b", "9b", "27b")}
    assert {entry["reading"] for entry in comparisons} == {"served", "forced"}, \
        "the forced reading the paper reports is tested, not only the served one"
    assert report["checks"]["reproduces_the_settled_table"]["checked"] is True
    assert report["checks"]["reproduces_the_settled_table"]["mismatches"] == 0


