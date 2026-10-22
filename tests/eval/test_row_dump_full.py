"""The full row dump: the option the model named, so an error can be told as *which* type.

Synthetic rows only — no GPU, no checkpoint, no network: the rows are built here in the shape the
scoring path produces, so the dump's arithmetic (argmax, the tie rule) and its refusals are checked
against the library rather than against a transcript. ``--rows-out``'s bytes are pinned here too,
because "the new dump does not touch the old one" is a claim about bytes, not about prose.
"""

from __future__ import annotations

import json
from argparse import Namespace
from typing import Any

import pytest

from sckevtype.eval import run

CHOICE, NOUL = "choice", "noul"


def _row(index: int, p=None, *, keys: tuple[str, ...] = ("alpha", "beta"), label: int = 0,
         type: str = CHOICE, split: str = "test", **extra: Any) -> dict:
    """A row shaped like the CLI's: kev's fields plus the metadata ``extended_row`` adds."""
    row = {
        "id": f"row-{index}", "question": "label", "split": split, "type": type,
        "dataset": "synthetic", "donor": "d0", "family": "panel_choice", "variant": "clean",
        "keys": list(keys), "label": label, "size": len(keys),
        "abstain": None, "abstain_source": "factory", "abstain_key": None,
        "abstain_text": None, "abstain_position": None, "abstain_last": False, "none_p": None,
    }
    if p is not None:
        row["p"] = list(p)
    row.update(extra)
    return row


def test_top1_and_top1_p_are_the_argmax_of_the_row_p(tmp_path):
    """The one number a confusion analysis needs: which option the model named, and with what."""
    info = run.write_full_row_dump(tmp_path / "full.jsonl",
                                   [_row(0, [0.05, 0.7, 0.25], keys=("alpha", "beta", "gamma"), label=2)])
    line = json.loads((tmp_path / "full.jsonl").read_text().splitlines()[0])

    assert line["top1"] == 1 and line["top1_p"] == 0.7
    assert line["keys"] == ["alpha", "beta", "gamma"] and line["label"] == 2
    assert line["id"] == "row-0" and line["size"] == 3
    assert set(line) == set(run.FULL_DUMP_FIELDS), "the full dump carries exactly its own fields"
    assert info == {"path": str(tmp_path / "full.jsonl"), "rows": 1, "skipped": 0,
                    "sha256": run.file_digest(tmp_path / "full.jsonl")}


def test_a_tie_names_the_lowest_index(tmp_path):
    """``argmax`` ties break to the lowest index, so the same row names the same option every run."""
    run.write_full_row_dump(tmp_path / "full.jsonl", [_row(0, [0.5, 0.5, 0.0], keys=("alpha", "beta", "gamma"))])
    line = json.loads((tmp_path / "full.jsonl").read_text().splitlines()[0])
    assert line["top1"] == 0 and line["top1_p"] == 0.5


@pytest.mark.parametrize("p", ["absent", None, []], ids=["absent", "none", "empty"])
def test_a_choice_row_without_p_raises_naming_the_id(tmp_path, p):
    """The negative control: a missing prediction fails the run rather than shrinking the dump."""
    row = _row(7) if p == "absent" else _row(7, p)
    with pytest.raises(ValueError, match="row-7"):
        run.write_full_row_dump(tmp_path / "full.jsonl", [_row(0, [0.9, 0.1]), row])
    assert not (tmp_path / "full.jsonl").exists(), "a refused dump is not written partially"


def test_noul_rows_are_absent_from_the_dump_and_counted(tmp_path):
    """Marker questions are out of scope, and the count says how many the dump left behind."""
    rows = [_row(0, [0.9, 0.1]), _row(1, None, type=NOUL, keys=("false", "true")),
            _row(2, [0.2, 0.8])]
    info = run.write_full_row_dump(tmp_path / "full.jsonl", rows)
    dumped = [json.loads(line) for line in (tmp_path / "full.jsonl").read_text().splitlines()]

    assert [row["id"] for row in dumped] == ["row-0", "row-2"]
    assert info["rows"] == 2 and info["skipped"] == 1


def test_rows_out_output_is_byte_identical_across_runs(tmp_path):
    """The new dump must not touch the old one: the same rows give the same bytes, here literal.

    The expected string was produced by ``write_row_dump`` before the full dump existed; a change
    to either the field list or the serialisation shows up as a byte difference, not a whisper.
    """
    expected = ('{"id": "r0", "question": "label", "split": "test", "dataset": "synthetic", '
                '"donor": "d0", "family": "panel_choice", "variant": "clean", "size": 2, '
                '"label": 0, "abstain": null, "abstain_source": "factory", '
                '"abstain_key": "not_listed", "abstain_text": null, "abstain_position": null, '
                '"abstain_last": false, "none_p": 0.25, "positive": 0}\n')
    row = {**_row(0, keys=("alpha", "beta")), "id": "r0", "abstain_key": "not_listed", "none_p": 0.25}

    first, second = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    run.write_row_dump(first, [row])
    run.write_row_dump(second, [row])
    assert first.read_bytes() == second.read_bytes() == expected.encode()
    assert b"top1" not in first.read_bytes(), "the axis-1 dump has not grown the full dump's fields"


def test_the_report_rows_block_carries_the_full_dump(tmp_path):
    """What ``report`` records: the block ``write_dumps`` returns is the report's ``rows`` block."""
    args = Namespace(rows_out=str(tmp_path / "axis1.jsonl"), rows_full=str(tmp_path / "full.jsonl"))
    block = run.write_dumps(args, {
        "calibration": [_row(0, [0.9, 0.1], split="calibration")],
        "test": [_row(1, [0.2, 0.8]), _row(2, None, type=NOUL, keys=("false", "true"))],
    })

    assert block["full_dump"] == {"path": str(tmp_path / "full.jsonl"), "rows": 2, "skipped": 1,
                                  "sha256": run.file_digest(tmp_path / "full.jsonl")}
    written = [json.loads(line) for line in (tmp_path / "full.jsonl").read_text().splitlines()]
    assert [row["id"] for row in written] == ["row-0", "row-1"], "calibration first, then test"
    assert block["axis1_dump"]["rows"] == 1, "the axis-1 dump still covers the test choice rows only"
    assert run.write_dumps(Namespace(rows_out=None, rows_full=None), {}) == {}, "no flags, no block"
