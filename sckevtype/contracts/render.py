"""THE canonical state renderer.

One function, ``render_rank_list``, turns (gene symbols, values) into the exact
string the model reads. Training, evaluation and serving all call this; a second
implementation is a bug, because the paper's claim "train and serve text are
identical" is only true if there is one implementation.

Frozen behaviour, in order (each step is a paper claim — change any of them and
every prior result is void, so a change produces a new artifact version):

1. drop genes matching ``cfg.exclude_patterns`` (prefix-anchored regex) and the
   supplied ubiquitous-gene list, and any symbol that is not printable ASCII;
2. order by value **descending**, ties broken by symbol **ascending**;
3. keep at most ``cfg.max_genes``;
4. render ``SYM1 1 | SYM2 2 | ...`` — rank shown, no counts, no expression values;
5. append the threshold delimiter (``"| ..."``) meaning "everything after this is
   below threshold, not absent".

Duplicate symbols collapse to their highest value, so one symbol is one rank.
The rendered string is count-free: absolute magnitude is deliberately discarded
(design-notes §8, "known loss").
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from .freeze import payload_digest, text_digest

#: Unsupervised, deterministic drop patterns (spec §1). Prefix-anchored: ``MT-``
#: matches ``MT-CO1``, ``RP[LS]\d`` matches ``RPS6``/``RPL13``, ``HB[ABD]``
#: matches ``HBA1``/``HBB``/``HBD``.
DEFAULT_EXCLUDE_PATTERNS: tuple[str, ...] = ("MT-", r"RP[LS]\d", "MTRNR", "HB[ABD]")
DEFAULT_MAX_GENES = 200
DEFAULT_DELIMITER = "| ..."
#: Separator between entries of the rank list (not configurable: it is part of the
#: frozen rendering, and a configurable separator would silently fork the format).
ENTRY_SEPARATOR = " | "


class FilterConfig(BaseModel):
    """The filter/ordering contract. Every field is frozen and every field is digested.

    ``digest()`` covers all four fields, so any state can be traced to the exact
    configuration that produced it — and a changed configuration cannot pass as
    the old one.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    exclude_patterns: tuple[str, ...] = Field(default=DEFAULT_EXCLUDE_PATTERNS)
    #: Gene symbols dropped for being expressed everywhere (the list is supplied
    #: data, from the seed config, never hard-coded here).
    ubiquitous: tuple[str, ...] = ()
    max_genes: int = Field(default=DEFAULT_MAX_GENES, ge=1)
    delimiter: str = DEFAULT_DELIMITER

    @property
    def ubiquitous_set(self) -> frozenset[str]:
        return frozenset(self.ubiquitous)

    def excluded(self, symbol: str) -> bool:
        """True when the symbol is filtered out of the state."""
        if not symbol.isascii() or not symbol.isprintable():
            return True
        if symbol in self.ubiquitous_set:
            return True
        return any(re.match(pattern, symbol) for pattern in self.exclude_patterns)

    def digest(self) -> str:
        """sha256 over every field of this config (key-sorted, so it is order-free)."""
        return payload_digest(self.model_dump(mode="json"))


def render_rank_list(genes: Sequence[str], values: Sequence[float], cfg: FilterConfig) -> str:
    """Render one cell's state. See the module docstring for the frozen behaviour.

    ``values`` are used for ordering only and never appear in the output.
    Raises ValueError on a length mismatch or a NaN value, because either one
    would make the ranking silently arbitrary.
    """
    if len(genes) != len(values):
        raise ValueError(f"genes and values must be the same length: {len(genes)} != {len(values)}")
    best: dict[str, float] = {}
    for symbol, value in zip(genes, values):
        if not isinstance(symbol, str) or not symbol or cfg.excluded(symbol):
            continue
        if math.isnan(value):
            raise ValueError(f"NaN value for gene {symbol!r}")
        if symbol not in best or value > best[symbol]:
            best[symbol] = value
    ordered = sorted(best.items(), key=lambda item: (-item[1], item[0]))[: cfg.max_genes]
    body = ENTRY_SEPARATOR.join(f"{symbol} {rank}" for rank, (symbol, _) in enumerate(ordered, start=1))
    return f"{body} {cfg.delimiter}" if body else cfg.delimiter


def parse_rank_list(rendered: str, cfg: FilterConfig) -> list[tuple[str, int]]:
    """Inverse of :func:`render_rank_list` (symbol, rank pairs), for QA and round-trip checks."""
    body = rendered[: -len(cfg.delimiter)].rstrip() if rendered.endswith(cfg.delimiter) else rendered
    if not body:
        return []
    pairs = []
    for entry in body.split(ENTRY_SEPARATOR):
        symbol, _, rank = entry.rpartition(" ")
        if not symbol or not rank.isdigit():
            raise ValueError(f"malformed rank-list entry: {entry!r}")
        pairs.append((symbol, int(rank)))
    return pairs


def render_digest(rendered: str) -> str:
    """sha256 of the rendered state — the digest a CellState pins."""
    return text_digest(rendered)


def verify_roundtrip(genes: Sequence[str], values: Sequence[float], cfg: FilterConfig) -> str:
    """Render, then parse the result back and check every frozen invariant. Returns the string.

    Used by the tests and by the QA tools: it fails loudly if the renderer ever
    emits a rank list that cannot be read back as the ranking it came from.
    """
    rendered = render_rank_list(genes, values, cfg)
    pairs = parse_rank_list(rendered, cfg)
    expected = [
        symbol
        for symbol, _ in sorted(
            {s: v for s, v in zip(genes, values) if isinstance(s, str) and s and not cfg.excluded(s)}.items(),
            key=lambda item: (-item[1], item[0]),
        )[: cfg.max_genes]
    ]
    assert [symbol for symbol, _ in pairs] == expected, f"round trip changed the ranking: {pairs} != {expected}"
    assert [rank for _, rank in pairs] == list(range(1, len(pairs) + 1)), "ranks must be 1..n in order"
    assert rendered.endswith(cfg.delimiter), "the threshold delimiter must be present"
    assert rendered.isascii(), "the rendered state must be ASCII"
    assert rendered == render_rank_list(genes, values, cfg), "rendering must be deterministic"
    return rendered
