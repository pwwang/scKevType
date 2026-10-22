"""The probe's own logic: which cells are drawn, what a row looks like, and whether the dump is readable.

The last one is the test that matters for E3.4: the axis-1 dump is only useful if
``scripts/axis1-recompute.py`` reads it, so this file imports *that script* (by path — it is a
script, not a module) and runs its own reconstruction and label checks over a dump this package
wrote. A dump that failed this test would still look fine in a JSON viewer.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from sckevtype.eval import abstain_key
from sckevtype.eval import metrics as M
from sckevtype.jev import probe as P
from sckevtype.jev import request as R

REPO = Path(__file__).resolve().parents[2]
ABSTAIN = "other"


def load_recompute():
    """``scripts/axis1-recompute.py`` as a module — the reader the dump has to satisfy."""
    path = REPO / "scripts" / "axis1-recompute.py"
    spec = importlib.util.spec_from_file_location("axis1_recompute", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- the sample


def test_cells_are_drawn_only_when_they_carry_every_family_the_probe_serves():
    index = {"d/one": {family: 1 for family in P.FAMILIES},
             "d/two": {"panel_choice": 2, "panel_miss": 3},
             "e/three": {family: 4 for family in P.FAMILIES}}
    datasets = {group: P.dataset_of(group) for group in index}
    assert P.choose_cells(index, 50, 1, datasets) == ["d/one", "e/three"]
    # and the count the report quotes is the eligible one, not every cell that carried some family
    assert P.eligible_by_dataset(index, datasets) == {"d": ["d/one"], "e": ["e/three"]}


def test_the_draw_is_deterministic_in_the_seed_and_not_in_a_dicts_order():
    index = {f"d/c{number:03d}": {family: 1 for family in P.FAMILIES} for number in range(200)}
    datasets = {group: P.dataset_of(group) for group in index}
    first = P.choose_cells(index, 50, 7, datasets)
    shuffled = dict(reversed(list(index.items())))
    assert first == P.choose_cells(shuffled, 50, 7, datasets) == P.choose_cells(index, 50, 7, datasets)
    assert first != P.choose_cells(index, 50, 8, datasets)
    assert len(first) == 50


def test_a_stratum_with_fewer_cells_than_asked_contributes_what_it_has_rather_than_being_topped_up():
    index = {f"big/c{number}": {family: 1 for family in P.FAMILIES} for number in range(60)}
    index.update({f"small/c{number}": {family: 2 for family in P.FAMILIES} for number in range(3)})
    datasets = {group: P.dataset_of(group) for group in index}
    chosen = P.choose_cells(index, 50, 1, datasets)
    assert sum(group.startswith("big/") for group in chosen) == 50
    assert sum(group.startswith("small/") for group in chosen) == 3


def test_dataset_of_splits_on_the_first_slash_only():
    assert P.dataset_of("tabula_muris_senis/3-M-5/6") == "tabula_muris_senis"


def test_a_cell_id_with_a_slash_in_it_still_names_one_file():
    assert "/" not in P.slug("3-M-5/6") and P.slug("3-M-5/6") == "3-M-5_6"


# --------------------------------------------------------------------------- the rows


def a_record(criteria=None, label="T cell", qtype="choice", noul_label=True):
    criteria = {"T cell": "positive markers: CD3E", "B cell": "positive markers: MS4A1",
                ABSTAIN: "None of the above"} if criteria is None else criteria
    question = {"type": qtype, "src": "panel_choice", "label": noul_label if qtype == "noul" else label,
                "instructions": "Which candidate does the state support?", "criteria": criteria}
    return {"state": "CD3E 1 | MS4A1 2 | ...", "questions": {"label": question},
            "_meta": {"dataset_id": "d", "family": "panel_choice", "cell_id": "c1", "group_id": "d/c1",
                      "donor_id": "n1", "panel_id": "p1", "variant": "clean", "source": "real-v2"}}


def a_prediction(probabilities, answers=None):
    return {"probabilities": {"label": probabilities}, "answers": answers or {"label": {"choice": "", "confidence": 0.5}},
            "served_model": "jev-1.13.0", "latency_ms": 100.0, "input_tokens": 1000, "output_tokens": 5}


def test_a_factory_given_abstain_option_takes_its_statistic_from_the_answer_as_served():
    record = a_record()
    rows = P.rows_of_answer(record, "panel_choice", "test",
                            a_prediction({"T cell": 0.6, "B cell": 0.25, ABSTAIN: 0.15}), variants=None)
    row = rows[0]
    assert row["abstain_source"] == "factory" and row["abstain"] == 2 and row["none_p"] == pytest.approx(0.15)
    assert row["abstain_key"] == ABSTAIN and row["abstain_last"] is True
    assert row["size"] == 3 and row["label"] == 0 and row["p"] == [0.6, 0.25, 0.15]
    assert M.axis1_label(row) is False      # the answer is a real option, so this is a negative


def test_a_question_without_an_abstain_option_takes_its_statistic_from_the_second_ask_and_is_marked_injected():
    record = a_record(criteria={"T cell": "positive markers: CD3E", "B cell": "positive markers: MS4A1"})
    variant = {"criteria": {**record["questions"]["label"]["criteria"], ABSTAIN: "None of the above"},
               "probabilities": {"T cell": 0.1, "B cell": 0.2, ABSTAIN: 0.7}}
    # keyed by the question it re-asks: a record with two such questions must not have the first
    # one's criteria scored against the second one's answer
    row = P.rows_of_answer(record, "panel_choice", "test",
                           a_prediction({"T cell": 0.7, "B cell": 0.3}), variants={"label": variant})[0]
    assert row["abstain"] is None and row["abstain_source"] == "injected"
    assert row["none_p"] == pytest.approx(0.7) and row["abstain_position"] == 2
    assert row["p"] == [0.7, 0.3]        # the scored distribution is the one without the abstain option


def test_a_choice_row_with_no_second_ask_raises_rather_than_being_emitted_without_a_statistic():
    record = a_record(criteria={"T cell": "x", "B cell": "y"})
    with pytest.raises(P.ProbeError, match="no abstain option and no second ask"):
        P.rows_of_answer(record, "panel_choice", "test", a_prediction({"T cell": 0.7, "B cell": 0.3}), variants=None)


def test_a_noul_row_carries_no_abstain_statistic_and_keeps_its_label_as_an_index():
    record = a_record(criteria=None, qtype="noul", noul_label=False)
    row = P.rows_of_answer(record, "marker_noul", "test",
                           a_prediction({"false": 0.9, "true": 0.1}), variants=None)[0]
    assert row["type"] == "noul" and row["label"] == 0 and row["p"] == [0.9, 0.1]
    assert row["none_p"] is None and row["abstain"] is None and row["abstain_source"] is None


def test_rows_carry_the_stratification_keys_the_metrics_read():
    rows = P.rows_of_answer(a_record(), "panel_choice", "test",
                            a_prediction({"T cell": 0.6, "B cell": 0.3, ABSTAIN: 0.1}), variants=None)
    assert set(P.ROW_FIELDS) <= set(rows[0])
    assert rows[0]["id"] == "test/d/c1/panel_choice" and rows[0]["group"] == "d/c1"


def test_the_row_order_matches_the_question_order_the_model_was_served():
    record = a_record()
    keys = list(record["questions"]["label"]["criteria"])
    row = P.rows_of_answer(record, "panel_choice", "test",
                           a_prediction({key: 1 / len(keys) for key in keys}), variants=None)[0]
    assert row["keys"] == keys and len(row["p"]) == len(keys) and abstain_key(row["keys"]) == ABSTAIN


# --------------------------------------------------------------------------- the dump, through its reader


def a_dump_row(abstain_source, none_p, label, family="panel_choice"):
    """A row as :func:`sckevtype.eval.run.write_row_dump` writes it, in a both-class family."""
    keys = ["T cell", "B cell", ABSTAIN]
    return {"id": f"test/d/c1/{family}", "group": "d/c1", "question": "label", "source": "real-v2",
            "task": family, "type": "choice", "variant": "clean", "split": "test", "family": family,
            "dataset": "d", "cell": "c1", "donor": "n1", "panel": "p1",
            "keys": keys, "label": label, "p": [0.6, 0.3, 0.1], "size": len(keys),
            "abstain": 2 if abstain_source == "factory" else None, "none_p": none_p,
            "abstain_source": abstain_source, "abstain_key": ABSTAIN, "abstain_text": "None of the above",
            "abstain_position": 2, "abstain_last": True}


def test_the_axis1_dump_is_read_by_axis1_recomputes_own_reader(tmp_path):
    from sckevtype.eval.run import write_row_dump

    recompute = load_recompute()
    rows = ([a_dump_row("factory", 0.2, 0) for _ in range(6)]
            + [a_dump_row("factory", 0.9, 2) for _ in range(6)]
            + [a_dump_row("injected", 0.3, 0) for _ in range(4)])
    written = write_row_dump(tmp_path / "axis1-rows.jsonl", rows)
    raw = recompute.load(tmp_path / "axis1-rows.jsonl")
    assert written["rows"] == len(raw) == len(rows)

    served = [recompute.as_served(row) for row in raw]
    # the reader passes a row through untouched when it carries the provenance fields; if it had to
    # reconstruct them, the dump is not in the CLI's format and `positive` would not be the label rule
    assert served == raw
    assert recompute.verify(raw, served) == []
    assert [row["positive"] for row in raw] == [0] * 6 + [1] * 6 + [0] * 4

    adequacy = M.panel_adequacy(served)
    assert adequacy["primary"] is not None
    assert adequacy["primary"]["auroc"] == pytest.approx(1.0)
    assert adequacy["primary"]["n"] == 12 and adequacy["primary"]["n_positive"] == 6
    assert adequacy["by_serving_convention"]["injected"]["n"] == 4


def test_the_dump_carries_every_field_the_reader_and_the_metric_read():
    from sckevtype.eval.run import DUMP_FIELDS

    # `positive` is not a row field: write_row_dump derives it from `axis1_label(row)`, and
    # scripts/axis1-recompute.py checks that derivation against the dump rather than trusting it.
    assert set(P.ROW_FIELDS) >= set(DUMP_FIELDS)
    assert "positive" not in P.ROW_FIELDS
    assert {"group", "keys", "p", "type"} <= set(P.ROW_FIELDS)


def test_a_dump_written_by_the_probe_round_trips_through_the_recompute_script(tmp_path):
    from sckevtype.eval.run import write_row_dump

    recompute = load_recompute()
    rows = [P.rows_of_answer(a_record(), "evidence_diluted", "test",
                             a_prediction({"T cell": 0.5, "B cell": 0.3, ABSTAIN: 0.2}), variants=None)[0],
            P.rows_of_answer({**a_record(), "questions": {"label": {**a_record()["questions"]["label"],
                                                                    "label": ABSTAIN}}},
                             "evidence_diluted", "test",
                             a_prediction({"T cell": 0.3, "B cell": 0.1, ABSTAIN: 0.6}), variants=None)[0]]
    write_row_dump(tmp_path / "d.jsonl", rows)
    raw = recompute.load(tmp_path / "d.jsonl")
    served = [recompute.as_served(row) for row in raw]
    assert recompute.verify(raw, served) == []
    assert [bool(row["positive"]) for row in raw] == [False, True]
    assert M.panel_adequacy(served)["primary"]["auroc"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- the report


def test_the_report_says_what_population_its_numbers_came_from(tmp_path):
    rows = ([a_dump_row("factory", 0.2, 0) for _ in range(6)]
            + [a_dump_row("factory", 0.9, 2) for _ in range(6)])
    sample = {"split": "test", "cells": 200, "families": list(P.FAMILIES), "seed": 1,
              "eligible_cells": 1741, "by_dataset": {"d": 200}, "corpus": str(tmp_path / "test.jsonl")}
    manifest = {"splits": {"splits": {"test": {"cells": 15626}}}}
    result = P.report_block(rows, sample, {"calls": 1}, P.FPR, manifest)
    assert result["population"]["cells"] == 200
    # the formatter reads the same keys the report writes — a rename that only one of them followed
    # is a KeyError at the end of a run that already cost money, so it is checked here
    text = "\n".join(P.lines(result))
    assert "sckevtype.jev.probe" in text and "axis 1 PRIMARY" in text
    assert result["plumbing_check"]["is_result"] is False
    reasons = " ".join(result["plumbing_check"]["reasons"])
    assert "200 of the partition's 15626 cells" in reasons
    assert "no calibration partition" in reasons
    assert "cannot be read as a benchmark" in result["population"]["not_a_benchmark"]


def test_the_report_refuses_an_empty_population_rather_than_reporting_a_zero():
    with pytest.raises(P.ProbeError, match="no choice rows"):
        P.report_block([], {"split": "test", "cells": 0, "families": [], "seed": 1, "eligible_cells": 0,
                            "by_dataset": {}}, {}, P.FPR, {})


# --------------------------------------------------------------------------- batching

#: Two states, as the corpus has them: the family trio's rank list and the diluted panel's.
STATE = "CD3E 1 | MS4A1 2 | ..."
DILUTED = "CD3E 1 | ..."
CHOICE = {"T cell": "positive markers: CD3E", "B cell": "positive markers: MS4A1", ABSTAIN: "None of the above"}
NOUL = {"true": "yes, the state satisfies the definition", "false": "no, it does not"}


def a_cell_records():
    """One cell's four families: the trio shares one state, ``evidence_diluted`` has a diluted one.

    ``panel_miss`` is built *without* an abstain option, which is the case that needs the second
    ask — so the comparison below covers the injected path as well as the factory-given one.
    """
    def build(family, criteria, state, qtype="choice"):
        record = a_record(criteria=criteria, qtype=qtype, label="T cell",
                          noul_label=True)
        return {**record, "state": state, "_meta": {**record["_meta"], "family": family}}

    return {"panel_choice": build("panel_choice", CHOICE, STATE),
            "panel_miss": build("panel_miss", {"T cell": "positive markers: CD3E",
                                               "B cell": "positive markers: MS4A1"}, STATE),
            "evidence_diluted": build("evidence_diluted", CHOICE, DILUTED),
            "marker_noul": build("marker_noul", NOUL, STATE, qtype="noul")}


def an_answering_transport(calls=None):
    """A transport that answers from the *question*, not from its id or the batch it arrived in.

    That is the property being tested: if the model's answer depended on the id or on the other
    questions in the request, this fake would not notice, and the comparison below would pass for
    the wrong reason. It is the same assumption the vendor states ("no change in answers") and the
    measurement in ``reports/jev-full`` is what tests it against the real endpoint.
    """
    def transport(url, payload, headers, timeout):
        request = json.loads(payload)
        answers = {}
        for qid, question in request["questions"].items():
            if question["type"] == "noul":
                answers[qid] = {"type": "noul", "noul": 0.25}
            else:
                keys = list(question["criteria"])
                answers[qid] = {"type": "choice", "choice": keys[0], "confidence": 0.5,
                                "probabilities": {key: 1.0 / len(keys) for key in keys}}
        if calls is not None:
            calls.append(json.loads(payload))
        return 200, {"body": {"model": "jev-test", "answers": answers,
                              "usage": {"input_tokens": len(payload), "output_tokens": 0}}, "headers": {}}
    return transport


def a_client(calls=None):
    from sckevtype.jev.client import JevClient

    return JevClient(key="test-key", transport=an_answering_transport(calls), sleep=lambda _s: None)


def test_the_state_groups_are_decided_by_byte_identical_states_not_by_the_family_list():
    records = a_cell_records()
    families = list(records)
    assert P.state_groups(families, records) == [["panel_choice", "panel_miss", "marker_noul"],
                                                 ["evidence_diluted"]]
    # one state per request is the API's shape, so the grouping is the number of calls a cell costs
    assert len(P.state_groups(families, records)) == 2


def test_a_family_missing_from_the_batch_raises_rather_than_being_dropped():
    records = a_cell_records()
    with pytest.raises(R.RequestError, match="no record for family"):
        P.state_groups([*records, "panel_axis"], records)


def test_a_batch_carries_every_question_of_every_family_over_their_shared_state():
    records = a_cell_records()
    request = R.batched_request(records, ["panel_choice", "marker_noul"])
    assert request["state"] == STATE
    assert sorted(request["questions"]) == ["marker_noul/label", "panel_choice/label"]
    assert request["questions"]["panel_choice/label"] == R.question_payload(records["panel_choice"]["questions"]["label"])
    assert request["questions"]["marker_noul/label"]["type"] == "noul"
    assert R.check_request(request)["questions"] == 2
    assert request["questions"]["panel_choice/label"]["criteria"] == CHOICE


def test_a_batch_refuses_families_whose_states_differ_rather_than_picking_one():
    records = a_cell_records()
    with pytest.raises(R.RequestError, match="do not share one state"):
        R.batched_request(records, ["panel_choice", "evidence_diluted"])


def test_a_one_family_batch_is_the_singleton_request_under_a_prefixed_id():
    records = a_cell_records()
    singleton = R.record_request(records["panel_choice"])
    batch = R.batched_request(records, ["panel_choice"])
    assert batch["state"] == singleton["state"] and batch["model"] == singleton["model"]
    assert batch["questions"]["panel_choice/label"] == singleton["questions"]["label"]


def test_the_batched_ask_builds_the_same_rows_as_one_request_per_family(tmp_path):
    """The whole claim of the batching, on a fake endpoint: same rows, fewer requests.

    Two clients over the same answering transport (answers depend on the question only): one asked
    a family at a time, one asked the cell as two requests. The rows must be identical — same
    option order, same labels, same ``none_p``, same abstain provenance — because they are built by
    the same :func:`P.rows_of_answer` from the same answers. ``latency_ms`` is excluded: it is the
    one field that is genuinely per-request, and the batched slice sets it to ``None`` for exactly
    that reason (the row dump does not carry it at all).
    """
    records = a_cell_records()
    singleton, batched = [], []
    one = a_client()
    for family, record in records.items():
        rows, _prediction = P.build_rows(record, family, "test", one, tmp_path / "single", "d", "c1")
        singleton.extend(rows)
    calls = []
    two = a_client(calls)
    rows, meta = P.build_batched_cell(list(records.items()), "test", two, tmp_path / "batched", "d", "c1")
    batched.extend(rows)
    # the singleton ask spends 5 requests: 4 base asks plus the abstain ask that ``panel_miss``
    # needs; the batched ask spends 2 — the trio (with that abstain question folded in) over their
    # shared state, and ``evidence_diluted`` over its own state
    assert meta["requests"] == 2 and two.calls == 2 and one.calls == 5
    assert [row["id"] for row in batched] == [row["id"] for row in singleton]
    def strip(rows):
        return [{key: value for key, value in row.items() if key != "latency_ms"} for row in rows]

    assert strip(batched) == strip(singleton)
    assert any(row["abstain_source"] == "injected" for row in batched)


def test_a_served_batched_cell_is_not_sent_again(tmp_path):
    records = a_cell_records()
    calls = []
    first, _meta = P.build_batched_cell(list(records.items()), "test", a_client(calls),
                                        tmp_path / "raw", "d", "c1")
    sent = len(calls)
    again, _meta = P.build_batched_cell(list(records.items()), "test", a_client(calls),
                                        tmp_path / "raw", "d", "c1")
    assert sent == 2 and len(calls) == sent, "the second pass must not send what is on disk"
    assert [row["id"] for row in again] == [row["id"] for row in first]


def test_the_full_population_draw_takes_every_eligible_cell():
    index = {f"d/c{number:03d}": {family: 1 for family in P.FAMILIES} for number in range(12)}
    index["d/x1"] = {"panel_choice": 1}          # carries only one family: not eligible
    datasets = {group: P.dataset_of(group) for group in index}
    assert len(P.choose_cells(index, None, 7, datasets)) == 12
    assert len(P.choose_cells(index, 5, 7, datasets)) == 5


# --------------------------------------------------------------- spend that survives a re-run


def test_a_cache_served_rerun_keeps_what_the_first_run_spent(tmp_path):
    """A resumable ask that spends nothing must not erase what the first pass spent.

    Both ask paths write ``ask-log.json`` on every run, and a re-run serves the bodies already on
    disk. Overwriting the accounting then records a $0.12 probe as $0.00 — real money disappearing
    from the artifact a reader would check.
    """
    P.write_ask_log(tmp_path, {"step": "ask", "accounting": {"calls": 796, "input_tokens": 2901625,
                                                             "output_tokens": 210855, "estimated_usd": 0.1219}})
    P.write_ask_log(tmp_path, {"step": "ask", "accounting": {"calls": 0, "input_tokens": 0,
                                                             "output_tokens": 0, "estimated_usd": 0.0}})
    log = json.loads((tmp_path / "ask-log.json").read_text())
    assert log["accounting"]["calls"] == 0, "this run still says what this run spent"
    assert [block["calls"] for block in log["superseded_runs"]] == [796]
    assert log["accounting_total"]["calls"] == 796 and log["accounting_total"]["runs"] == 2
    assert log["accounting_total"]["estimated_usd"] == 0.1219

    provider = P.provider_block(log["accounting"], log)
    assert provider["calls"] == 0
    assert provider["cumulative_over_runs"]["estimated_usd"] == 0.1219
    assert "0.1219" in provider["note"], "the total has to be readable without opening ask-log.json"


def test_a_single_run_reports_its_own_spend_and_nothing_else(tmp_path):
    P.write_ask_log(tmp_path, {"step": "ask", "accounting": {"calls": 12, "input_tokens": 100,
                                                             "estimated_usd": 0.0001}})
    log = json.loads((tmp_path / "ask-log.json").read_text())
    assert "superseded_runs" not in log and "accounting_total" not in log
    assert P.provider_block(log["accounting"], log) == log["accounting"]


def test_the_three_ask_logs_on_disk_would_report_their_own_histories():
    """The directories on disk are single runs, so their blocks must stay as they are."""
    for name, calls in (("jev-full", 4980), ("jev-smoke", 0), ("jev-equivalence", None)):
        path = P.root() / "reports" / name
        if not (path / "ask-log.json").exists():
            continue
        log = json.loads((path / "ask-log.json").read_text())
        if calls is not None:
            assert log["accounting"]["calls"] == calls or log.get("accounting_total", {}).get("calls") == calls
