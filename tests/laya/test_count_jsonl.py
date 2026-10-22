"""The resume path's row count.

``infer`` skips any chunk whose rows file already exists and adds that file's row count to ``skipped``.
That count used to be ``len(_read_jsonl(target))`` — ``_read_jsonl`` is a generator, so the first
*re-run* into a populated rows directory died with ``TypeError: object of type 'generator' has no
len()`` before answering anything. The first full pass never met it, which is why it survived.
"""

from pathlib import Path

from sckevtype.laya.score import _count_jsonl


def test_count_jsonl_counts_rows_and_ignores_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text('{"a": 1}\n\n{"a": 2}\n', encoding="utf-8")
    assert _count_jsonl(path) == 2


def test_count_jsonl_on_an_empty_file_is_zero(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")
    assert _count_jsonl(path) == 0
