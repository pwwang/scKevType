"""The token budget: what it bounds, and what it must not change.

The OOM this exists for is a *record* that does not fit, not a batch: at ``--batch 1`` one
forward pass is already one record, and the longest records in the training suite cost ~69,000
padded tokens, which is why a 9B died on a 40 GB card with a step-1 peak of 11.8 GiB. So the
budget decides how many forward/backward passes an optimizer step's records take — never which
records, or how many, an optimizer step is worth.

The tests that matter are therefore the two invariants around it: the *effective batch size*
(steps of ``batch x accum`` records, the same records in the same steps as the count-budgeted
path, the same LR-schedule step count), and the *loss a cut record contributes* (its parts carry
exactly its rows, weighted so the step's gradient is the one the uncut record would have made).
Both are checked against kev's own encoder and loss rather than against a copy of their rules.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace

import pytest

from sckevtype.train.train import (
    Forward,
    groups_of,
    oversized_parts,
    parse_args,
    part_cost,
    plan_parts,
    provenance_record,
    row_lengths,
    sub_encoding,
    variant_forwards,
)

# --------------------------------------------------------------------------- the effective batch


def test_the_optimizer_steps_are_batch_times_accum_records_however_the_records_are_batched() -> None:
    """The invariant a budget must not touch: a step is ``batch x accum`` records, the last one short."""
    for n, batch, accum in ((5592, 1, 8), (40, 3, 4), (17, 2, 3), (1, 1, 1), (100, 1, 1), (6, 2, 3)):
        steps = groups_of(n, batch, accum)
        assert sum(stop - start for start, stop in steps) == n
        assert [stop - start for start, stop in steps][:-1] == [batch * accum] * (len(steps) - 1)
        assert 0 < steps[-1][1] - steps[-1][0] <= batch * accum
        assert steps[0][0] == 0 and steps[-1][1] == n
        # the count that sizes the LR schedule, so the budget cannot lengthen or shorten a run
        assert len(steps) == math.ceil(math.ceil(n / batch) / accum)


def test_the_count_budgeted_path_microbatches_tile_the_token_budgeted_paths_steps() -> None:
    """``--accum`` microbatches of ``--batch`` records is the same partition as one step's records.

    This is what makes the two paths' optimizer steps the same object: the count-budgeted loop
    ends a step every ``accum`` microbatches (or at the end of the epoch), and those boundaries
    have to fall exactly where :func:`groups_of` cuts.
    """
    for n, batch, accum in ((5592, 1, 8), (17, 2, 3), (40, 3, 4)):
        micro_per_epoch = math.ceil(n / batch)
        tiled = [(start * batch, min((start + accum) * batch, n))
                 for start in range(0, micro_per_epoch, accum)]
        assert tiled == groups_of(n, batch, accum)


@pytest.fixture
def kev_train():
    """kev's own parser and encoder: the tests that drive a path end to end need the train extra."""
    return pytest.importorskip("kev.train", reason="the flags are kev's; requires the train extra")


@pytest.fixture
def budgeted_suite(tmp_path: Path) -> Path:
    """A suite whose manifest records a budget — the only thing ``kev_namespace`` reads from one."""
    suite = tmp_path / "suite"
    suite.mkdir()
    (suite / "manifest.json").write_text(json.dumps({"name": "smoke", "context": {"max_state": 1664}}), encoding="utf-8")
    return suite


@pytest.fixture(scope="module")
def suite_records() -> list[dict]:
    """The smoke suite's training partition: what a run trains on, ``_meta.id`` included.

    Not the corpus artifact: kev's ``encode_batch`` seeds each record's augmentation with
    ``_meta.id`` and the corpus records do not carry one (the suite build adds it), so a test that
    drives the trainer's own paths has to hand them the suite's records.
    """
    path = Path(__file__).resolve().parents[2] / "artifacts" / "suites" / "smoke" / "train.jsonl"
    if not path.exists():
        pytest.skip(f"no smoke suite at {path}; build one with `python -m sckevtype.train.suite`")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class RecordsModel:
    """What ``kev.train.encode_batch`` and the planner ask a model for, and nothing else.

    ``encode_batch`` encodes through the model (``model.encode``), and the planner asks whether an
    encoding runs in the row form: no transformer, no card, no optimizer is involved in either.
    """

    option_isolation = False

    def encode(self, tokenizer, record, **limits):
        from kev.model import encode

        return encode(tokenizer, record, option_isolation=False, **limits)

    def rows_form(self, encs) -> bool:
        return True


def test_a_token_budget_leaves_the_optimizer_steps_alone(kev_train, budgeted_suite: Path,
                                                         suite_records: list[dict], stand_in_tokenizer) -> None:
    """Strongest form of "the effective batch size is unchanged": the same records in the same steps.

    Both paths are driven over the same suite records with the same seed, and the records each
    optimizer step covers are compared: the budget may only change how many passes carry them.
    """
    from sckevtype.train.train import budgeted_steps, counted_steps, kev_namespace

    requests = suite_records[:12]
    args = parse_args(["--suite", str(budgeted_suite), "--base", "b", "--batch", "2", "--accum", "3", "--out", "o"])
    namespace = kev_namespace(args, "o")
    model = RecordsModel()
    counted = list(counted_steps(model, stand_in_tokenizer, namespace, requests, 0, math.ceil(len(requests) / namespace.batch)))
    budgeted = list(budgeted_steps(model, stand_in_tokenizer, namespace, requests, 0, 60, Counter()))

    def covered(step) -> list[str]:
        """The records a step covers, in order: a cut record's parts are consecutive, so runs collapse."""
        ids = [variant.request_id for forward in step.forwards for variant in forward.batch]
        return [record_id for index, record_id in enumerate(ids) if index == 0 or ids[index - 1] != record_id]

    assert len(counted) == len(budgeted) == math.ceil(len(requests) / (namespace.batch * namespace.accum))
    assert [covered(step) for step in counted] == [covered(step) for step in budgeted]
    assert [step.records for step in counted] == [step.records for step in budgeted] == [6, 6]
    # and the budget did change the passes: a record over 60 padded tokens is carried by several
    assert sum(len(step.forwards) for step in budgeted) > sum(len(step.forwards) for step in counted)


def test_without_a_budget_every_record_is_one_pass(kev_train, budgeted_suite: Path,
                                                   suite_records: list[dict], stand_in_tokenizer) -> None:
    """``--token_budget 0`` is the count-budgeted path: one pass per ``--batch`` records, byte for byte."""
    from sckevtype.train.train import counted_steps, kev_namespace

    requests = suite_records[:8]
    namespace = kev_namespace(parse_args(["--suite", str(budgeted_suite), "--base", "b", "--out", "o"]), "o")
    # the defaults: batch 1 x accum 8, so these eight records are one optimizer step of eight passes
    steps = list(counted_steps(RecordsModel(), stand_in_tokenizer, namespace, requests, 0, 8))
    assert len(steps) == 1
    assert [len(step.forwards) for step in steps] == [8]
    assert [step.records for step in steps] == [8]
    assert all(forward.divisor == 8 for forward in steps[0].forwards)


# --------------------------------------------------------------------------- what the budget bounds


def test_a_token_budget_caps_the_padded_tokens_a_pass_pays() -> None:
    """A pass pays ``rows x longest row`` — the pad is real work, so that product is what is bounded.

    Four equal rows of 1,500 tokens: three rows would cost 4,500 and two cost 3,000, so a 3,200-token
    budget cuts them in pairs. Summing the rows instead (6,000 over four) would have called this one
    pass and kept the memory this exists to bound.
    """
    lengths = [1500, 1500, 1500, 1500]
    parts = plan_parts(lengths, 3200)
    assert parts == [(0, 2), (2, 4)]
    assert all(part_cost(lengths, first, last) <= 3200 for first, last in parts)


def test_the_longest_row_decides_what_fits_beside_it() -> None:
    """One long row pads the whole pass, so it must be priced into what the budget admits."""
    lengths = [100, 1500, 1500]
    assert plan_parts(lengths, 3000) == [(0, 2), (2, 3)]      # 3 rows x 1500 = 4500 does not fit


def test_a_record_inside_the_budget_is_one_pass() -> None:
    """The budget only splits what has to be split: a short record is the pass the count path would run."""
    assert plan_parts([100, 100, 100], 5000) == [(0, 3)]
    assert plan_parts([100, 100, 100], 0) == [(0, 3)]


def test_no_question_is_ever_cut_out_of_a_pass() -> None:
    """Parts are consecutive, complete runs of a record's rows: cutting inside a row would cut the context."""
    lengths = [10, 900, 20, 30, 4000, 40]
    parts = plan_parts(lengths, 500)
    assert parts[0][0] == 0 and parts[-1][1] == len(lengths)
    assert all(first < last for first, last in parts)
    assert all(left[1] == right[0] for left, right in pairwise(parts))
    assert sum(last - first for first, last in parts) == len(lengths)


def test_a_single_row_over_the_budget_is_its_own_pass_and_is_counted() -> None:
    """A row cannot be split, so it goes over and the run says how many times that happened.

    A 2,000-token budget does not fit the two 1,500-token rows either (2 x 1,500 = 3,000), so the
    assertion that separates the two cases is the 3,000 one: the 9,000-token row is a single row,
    which no budget can cut, while the two 1,500s are simply a pair that does not fit.
    """
    lengths = [9000, 1500, 1500]
    assert plan_parts(lengths, 2000) == [(0, 1), (1, 2), (2, 3)]
    assert plan_parts(lengths, 3000) == [(0, 1), (1, 3)]
    assert oversized_parts(lengths, 3000) == 1        # the 9,000-token row, and only that one
    assert oversized_parts([100, 100], 2000) == 0


# --------------------------------------------------------------------------- what a part carries


@pytest.fixture(scope="module")
def wide_record(corpus_dir: Path) -> dict:
    """The corpus's widest record: several questions, since a one-question record cannot be cut."""
    records = [json.loads(line) for line in (corpus_dir / "train.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    return max(records, key=lambda record: len(record["questions"]))


def test_a_part_encodes_exactly_the_records_own_rows(kev_train, wide_record: dict, stand_in_tokenizer) -> None:
    """The state plus the questions it kept — the same tokens, positions and readouts as the whole record.

    This is the property the cut rests on: ``rows_of`` splits a record into state + one row per
    question, so a subset of the questions re-encoded on its own has to reproduce those rows exactly.
    Anything else would serve the model a context it was not trained on.
    """
    from kev.data import materialize
    from kev.model import encode, rows_of

    record = materialize(wide_record)
    enc = encode(stand_in_tokenizer, record, max_state=4096, max_branch=8192)
    state, positions, rows = rows_of(enc)
    assert len(rows) >= 3

    sub_state, sub_positions, sub_rows = rows_of(sub_encoding(enc, 1, 3))
    assert sub_state == state and sub_positions == positions
    for key in ("ids", "pos", "decide", "opts"):
        assert [row[key] for row in sub_rows] == [row[key] for row in rows[1:3]]
    # the question that was left out is really gone, and the state is still whole
    assert len(sub_rows) == 2


def test_the_parts_of_a_cut_record_weigh_exactly_one(kev_train, wide_record: dict, stand_in_tokenizer) -> None:
    """A cut record contributes what an uncut one did: its parts' weights sum to 1 over the record.

    The weight is ``questions in the part / questions in the record``, which is what makes a step's
    gradient the same object whether one of its records was carried by one pass or by five.
    """
    from kev.data import materialize
    from kev.model import encode
    from kev.train import Variant

    record = materialize(wide_record)
    enc = encode(stand_in_tokenizer, record, max_state=4096, max_branch=8192)
    variant = Variant(rec=record, enc=enc, request_id="r", source="s")
    lengths = row_lengths(RecordsModel(), enc)
    assert lengths is not None and len(lengths) >= 3

    budget = 2 * max(lengths) + 1          # two of the widest rows fit, the whole record does not
    stats: Counter = Counter()
    forwards = variant_forwards(RecordsModel(), variant, budget, divisor=8.0, stats=stats)

    assert len(forwards) == len(plan_parts(lengths, budget)) > 1
    assert sum(forward.weight for forward in forwards) == pytest.approx(1.0)
    assert stats["split_records"] == 1 and stats["oversized"] == 0
    assert all(forward.divisor == 8.0 for forward in forwards)
    # each part carries a consecutive, non-empty run of the record's questions, and only those
    questions = [len(forward.batch[0].rec["questions"]) for forward in forwards]
    assert questions == [last - first for first, last in plan_parts(lengths, budget)]
    assert sum(questions) == len(record["questions"])
    assert all(forward.batch[0].request_id == "r" for forward in forwards)


def test_a_cut_record_means_the_same_over_its_questions(kev_train) -> None:
    """The identity the weighting exists for, on kev's own loss: mean over the record = Σ weight x mean over the part."""
    pytest.importorskip("torch")
    import torch
    from kev.train import question_loss

    logits = [torch.tensor([0.7, -0.2]), torch.tensor([-1.0, 0.4]), torch.tensor([0.1, 0.9]),
              torch.tensor([2.0, -0.5]), torch.tensor([0.3, 0.2])]
    questions = [{"label": 0, "qtype": "choice"}, {"label": 1, "qtype": "choice"}, {"label": 1, "qtype": "choice"},
                 {"label": 0, "qtype": "choice"}, {"label": 0, "qtype": "choice"}]
    whole = sum(question_loss(z, q, "cpu", 0.0) for z, q in zip(logits, questions)) / len(questions)
    parts = [(0, 2), (2, 5)]
    cut = sum((last - first) / len(questions)
              * sum(question_loss(z, q, "cpu", 0.0) for z, q in zip(logits[first:last], questions[first:last])) / (last - first)
              for first, last in parts)
    assert torch.allclose(whole, cut, atol=1e-6)


def test_a_variant_with_no_permuted_copy_is_still_cut(kev_train, wide_record: dict, stand_in_tokenizer) -> None:
    """An unpermuted record is the common case (``--perm_kl 0``, every run here) and must cut cleanly."""
    from kev.data import materialize
    from kev.model import encode
    from kev.train import Variant

    record = materialize(wide_record)
    enc = encode(stand_in_tokenizer, record, max_state=4096, max_branch=8192)
    variant = Variant(rec=record, enc=enc, request_id="r", source="s", permuted=None)
    forwards = variant_forwards(RecordsModel(), variant, 60, 8.0, Counter())
    assert len(forwards) > 1
    assert all(forward.batch[0].permuted is None for forward in forwards)


# --------------------------------------------------------------------------- the flag


def test_the_token_budget_is_ours_and_is_never_handed_to_kev(kev_train, budgeted_suite: Path) -> None:
    """kev has no such flag; handing its parser one would fail the run at the last moment."""
    from sckevtype.train.train import kev_namespace

    args = parse_args(["--suite", str(budgeted_suite), "--base", "b", "--token_budget", "16384", "--out", "o"])
    assert args.token_budget == 16384
    assert not hasattr(kev_namespace(args, "o"), "token_budget")


def test_the_default_is_off(kev_train, budgeted_suite: Path) -> None:
    """Every existing lane keeps the count-budgeted path unless it asks for the budget."""
    assert parse_args(["--suite", "s", "--base", "b", "--out", "o"]).token_budget == 0


def test_a_negative_budget_is_refused(kev_train) -> None:
    with pytest.raises(SystemExit):
        parse_args(["--suite", "s", "--base", "b", "--token_budget", "-1", "--out", "o"])


def test_the_provenance_records_the_budget_and_its_absence(tmp_path: Path) -> None:
    """A run that was cut has to say so, and a run that was not is unchanged: the key is absent, not null."""
    suite = tmp_path / "suite"
    suite.mkdir()
    (suite / "manifest.json").write_text('{"name": "smoke"}\n', encoding="utf-8")

    def build(token_budget=None) -> dict:
        return provenance_record(
            namespace=argparse.Namespace(suite=str(suite), seed=1), revision="dc7cdfe2ee41",
            manifest={"name": "smoke"}, suite_dir=suite,
            trainables=SimpleNamespace(count=1, by_kind={}, names=()), base={}, environment={},
            epochs=[], optimizer_steps=0, peak_device_bytes=0, stopped_early=False, token_budget=token_budget,
        )

    assert "token_budget" not in build()
    record = build({"budget": 16384, "passes": 20, "split_records": 2, "oversized": 0})
    assert record["token_budget"]["budget"] == 16384
    assert json.loads(json.dumps(record, allow_nan=False)) == record


def test_a_forward_is_what_the_loop_scales_and_divides() -> None:
    """``Forward`` carries the two numbers the loop needs, and the count path's default is 1.0 (no scaling)."""
    assert Forward([], 8.0).weight == 1.0
    assert Forward([], 8.0, 0.25).weight == 0.25
