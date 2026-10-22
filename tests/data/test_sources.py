"""How a corpus dataset resolves to *bytes*: the fetched file, not the fetched directory.

Step 6 §7. ``fetch`` returns the dataset directory (right for a 10x bundle), and the
corpus handed that straight to the h5ad adapter — ``IsADirectoryError:
data/raw/muraro``. The regression is one line of arithmetic about paths, so it needs
no atlas and no network: a directory that contains the file the registry names.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from sckevtype.data.build import (
    CorpusConfig,
    DatasetConfig,
    _primary_file,
    resolve_dataset,
)


def entry(name: str) -> SimpleNamespace:
    """Only ``access[0].name`` is read, so a real registry entry would be 4 fields of noise."""
    return SimpleNamespace(access=(SimpleNamespace(name=name),))


def test_a_fetched_directory_resolves_to_the_file_inside_it(tmp_path: Path):
    """The exact shape of the bug: fetch() returns a directory, the adapter needs a file."""
    fetched = tmp_path / "muraro"
    fetched.mkdir()
    (fetched / "muraro.h5ad").write_bytes(b"not really an h5ad")
    assert _primary_file(fetched, entry("muraro.h5ad")) == fetched / "muraro.h5ad"


def test_a_directory_without_the_named_file_is_left_alone(tmp_path: Path):
    """A 10x bundle has no such file, and the directory *is* the dataset."""
    fetched = tmp_path / "pbmc3k"
    fetched.mkdir()
    (fetched / "matrix.mtx").write_text("1 1 1\n", encoding="utf-8")
    assert _primary_file(fetched, entry("pbmc3k")) == fetched


def test_a_loose_file_is_its_own_answer(tmp_path: Path):
    loose = tmp_path / "already.h5ad"
    loose.write_bytes(b"x")
    assert _primary_file(loose, entry("muraro.h5ad")) == loose
    assert _primary_file(loose, None) == loose


def test_fetch_and_path_are_mutually_exclusive(tmp_path: Path):
    corpus = CorpusConfig(
        name="x",
        seed=1,
        datasets=[DatasetConfig(id="p", registered=False, format="csv_expr", tier="T3", path="/tmp/x", cap=1)],
        state={"max_tokens": 384},
    )
    with pytest.raises(ValueError, match="no path and fetch is false"):
        resolve_dataset(DatasetConfig(id="p", registered=False, format="csv_expr", tier="T3", cap=1), corpus)
