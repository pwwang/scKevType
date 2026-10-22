"""The state budget: what a rendered state costs kev, and what a suite records.

A state is a rendered rank list; its cost is not its character count. ``RPLP1``
is several tokens, and the pinned tokenizer spends ~1,574 tokens on a state a
character heuristic would call ~600. So the budget a suite is trained under has
to be *measured with the tokenizer kev will use*, not guessed — which is the
defect this module exists to remove: the first real training run died on its
first batch with ``state exceeds 384 tokens: 1574``, after a cold GPU boot, on a
suite whose manifest recorded kev's default budget because nothing had ever
compared the data to it.

Everything here is the counting rule, in one place, so the suite builder, the
corpus QC and the trainer all ask the same question of the same bytes:

* :func:`state_tokens` — kev's own count, via ``kev.model.user_tokens``, the
  function ``kev.model.encode`` calls. The ``+1`` is real: the encoder prefixes
  one delimiter token to the state and compares ``len(state) + 1`` to
  ``max_state``, so that is the quantity a budget has to cover;
* :func:`budget_for` — an observed maximum rounded up to a multiple of
  :data:`BUDGET_UNIT`. The rounding is not cosmetic: a budget equal to the
  observed maximum would be exceeded by the next record drawn from the same
  distribution, and a multiple of 64 is what makes two independently measured
  suites record comparable numbers;
* :func:`required_budget` — the number a *suite* records, which is not the state
  maximum. kev derives three limits from ``--max_state`` alone
  (``kev.model.training_context``: the row at ``N + 640``, the packed record at
  ``N + 1664``), so a record has to fit all three and the packed one binds here.
  A suite that records only the state axis is the second version of the same
  first-batch failure;
* :func:`summarize` — the distribution a manifest and a QC report both print.

The tokenizer is a parameter rather than a constant: the suite pins no base
(the base is a run's choice), so the caller resolves the tokenizer — from the
model registry, in the CLIs — and hands it here. Nothing in this module reaches
the network.
"""

from __future__ import annotations

import statistics
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

#: The unit a recorded budget is rounded up to.
BUDGET_UNIT = 64


def pinned_tokenizer(pin: Mapping[str, str]) -> Any:
    """kev's own tokenizer loader for a ``{id, revision}`` pin.

    Both callers that measure (the suite builder and the corpus build) name a
    tokenizer the same way — a registry entry's id and revision — and both must
    load it the way kev does, since the counts are only comparable if the
    tokenizer is.
    """
    from kev.model import load_tokenizer

    return load_tokenizer(pin["id"], revision=pin["revision"])


def state_tokens(tokenizer: Any, state: str) -> int:
    """What ``state`` costs kev: ``len(user_tokens(...)) + 1``, the number ``encode`` compares to ``max_state``.

    ``user_tokens`` is kev's own tokenization (it rewrites ``<|name|>`` so a
    caller cannot forge a delimiter before tokenizing). Calling it rather than
    ``tokenizer(...)`` is the difference between measuring kev's encoder and
    measuring a plausible approximation of it.
    """
    from kev.model import user_tokens

    return len(user_tokens(tokenizer, state)) + 1


def state_lengths(tokenizer: Any, states: Iterable[str]) -> list[int]:
    """The token count of every state, in order."""
    return [state_tokens(tokenizer, state) for state in states]


def truncate_state(tokenizer: Any, state: str, max_state: int) -> str:
    """``state`` cut to the tokens kev would keep under ``max_state`` — its first ``max_state - 1``.

    This is exactly ``kev.model.encode``'s own truncation (``state_tokens[: max_state - 1]``,
    with its delimiter accounted for), applied to the text instead of to the
    encoding, so the suite's bytes are already what the encoder would have used.

    A token cut is not a symbol cut: the tail symbol is usually left partial,
    which is what kev itself does when it encodes a state under a strict budget.
    Callers must re-measure the result — ``decode`` then re-encoding is not
    guaranteed to be idempotent — which is why this returns text and not a
    guarantee.
    """
    from kev.model import user_tokens

    return tokenizer.decode(user_tokens(tokenizer, state)[: max_state - 1])


def round_up(value: int, unit: int = BUDGET_UNIT) -> int:
    """``value`` rounded up to a multiple of ``unit`` (``unit`` divides nothing, so 0 maps to 0)."""
    return -(-value // unit) * unit


def budget_for(maximum: int, unit: int = BUDGET_UNIT) -> int:
    """The budget a measured maximum requires: rounded up to a multiple of ``unit``.

    The state axis alone. A recorded budget has to cover two more
    (:func:`required_budget`), which is the defect this function's callers used to
    have: a suite measuring 2,020 state tokens recorded 2,048 and still died on its
    first batch, because kev derives its row and packed limits from that one number.
    """
    return round_up(maximum, unit)


def budget_terms(max_state: int, max_row: int, max_packed: int) -> dict[str, int]:
    """What each of kev's three limits asks of the budget ``N``.

    kev derives its row and packed limits from ``--max_state`` alone
    (``kev.model.training_context``): a budget of ``N`` admits a row of
    ``N + (MAX_BRANCH - MAX_STATE)`` and a packed record of
    ``N + (MAX_PACKED - MAX_STATE)``. Subtracting those slacks from an observation
    turns it into a lower bound on ``N`` — the three inequalities a recorded budget
    has to satisfy. The slacks are read off kev's constants rather than written down
    here, so this cannot drift from the trainer that will enforce them.
    """
    from kev.model import MAX_BRANCH, MAX_PACKED, MAX_STATE

    return {
        "state": max_state,
        "row": max_row - (MAX_BRANCH - MAX_STATE),
        "packed": max_packed - (MAX_PACKED - MAX_STATE),
    }


def required_budget(max_state: int, max_row: int, max_packed: int, unit: int = BUDGET_UNIT) -> int:
    """The smallest multiple of ``unit`` that satisfies all three of kev's limits at once.

    The floor is kev's own ``MAX_STATE``: below it ``training_context`` refuses the
    value, so a suite measuring nothing still records a trainable number.

    The packed term is the binding one for this corpus — ``640`` tokens of headroom
    over the state is not enough for a record whose questions cost thousands — which
    is why a budget derived from the state alone is a budget that fails on its first
    batch. A budget ``N`` is trainable at all only while ``N <= MAX_TRAIN_STATE``;
    callers check that (see :mod:`sckevtype.train.suite`).
    """
    from kev.model import MAX_STATE

    return round_up(max(MAX_STATE, *budget_terms(max_state, max_row, max_packed).values()), unit)


def observed_limits(observed: Mapping[str, Any]) -> dict[str, int] | None:
    """The three maxima a manifest's ``observed`` block records, or ``None`` when it records none.

    Flat ``max_state`` / ``max_row`` / ``max_packed`` is what this module's writer
    emits. Suites written before the three-limit rule nested the same measurements
    (``max`` / ``row.max`` / ``packed.max``), and those are read too, so an old suite
    is refused by name rather than silently skipped.
    """
    def nested(key: str) -> int | None:
        return (observed.get(key) or {}).get("max")

    limits = {
        "max_state": observed.get("max_state", observed.get("max")),
        "max_row": observed.get("max_row", nested("row")),
        "max_packed": observed.get("max_packed", nested("packed")),
    }
    if any(value is None for value in limits.values()):
        return None
    return {key: int(value) for key, value in limits.items()}


def summarize(lengths: Sequence[int]) -> dict[str, int]:
    """``{n, min, median, p95, max}`` — the distribution both a manifest and a QC report carry.

    ``p95`` is the nearest-rank value at index ``0.95 * (n - 1)``, the same
    definition ``scripts/measure-state-tokens.py`` prints, so a manifest and an
    independent sample of the same suite can be compared by eye.
    """
    if not lengths:
        return {"n": 0, "min": 0, "median": 0, "p95": 0, "max": 0}
    ordered = sorted(lengths)
    return {
        "n": len(ordered),
        "min": ordered[0],
        "median": int(statistics.median(ordered)),
        "p95": ordered[int(0.95 * (len(ordered) - 1))],
        "max": ordered[-1],
    }


def over_budget(lengths: Sequence[int], budget: int) -> int:
    """How many states cost more than ``budget`` tokens."""
    return sum(1 for length in lengths if length > budget)


#: The shape every caller measures with: state text in, token count out.
Measure = Callable[[str], int]


def encoded_lengths(records: Iterable[Any], tokenizer: Any, *, max_state: int) -> tuple[list[int], list[int]]:
    """Per record, ``(row, packed)`` token lengths — the other two limits kev's trainer enforces.

    A record is not admitted on its state alone. ``kev.model.encode`` has a row
    limit (the state plus its longest question, ``max_branch - len(S)``) and
    ``kev.train.encode_batch`` a packed limit (the state plus *every* question,
    ``max_packed``), and both are derived by kev from ``--max_state`` alone. A
    suite whose states fit and whose branches do not therefore still dies on its
    first batch — which is what ``branch too long: 2197 tokens with a 1585-token
    state (row limit 2688)`` was.

    Both counts come from ``kev.model.encode`` itself, at the state budget the
    suite records and with kev's serving branch limit so nothing raises: a
    measurement, not a copy of kev's rules. ``row`` is what the branch check
    compares and ``packed`` is ``len(enc["ids"])``, the quantity
    ``kev.train.encode_batch`` compares to ``max_packed``.

    Each count includes the tokens training's own augmentation can still add
    (:func:`augmentation_headroom`): the trainer encodes *variants* of these
    records — a permuted option order, sometimes one extra option — so a budget
    measured on the stored bytes alone is exceeded by the first variant that
    gains an option.
    """
    from kev.data import materialize
    from kev.model import SERVE_MAX_BRANCH, encode

    widest = augmentation_headroom(tokenizer)
    rows, packed = [], []
    for record in records:
        rec = materialize(record)
        enc = encode(tokenizer, rec, max_state=max_state, max_branch=SERVE_MAX_BRANCH)
        segments = Counter(enc["seg"])
        state = segments[0]
        headroom = len(rec["questions"]) * widest
        rows.append(state + max((count for question, count in segments.items() if question), default=0) + headroom)
        packed.append(len(enc["ids"]) + headroom)
    return rows, packed


def augmentation_headroom(tokenizer: Any) -> int:
    """The most tokens one augmented option can add to a branch, over kev's own option tables.

    ``kev.data.augment`` gives a question at most one extra option — a
    'none of the above', a distractor, or (in ``none_pair``) a none option
    swapped for the true one — drawn from ``kev.data``'s constants. The widest of
    those options is therefore the per-question headroom a budget needs.
    Measuring it beats inheriting ``kev.suite``'s ``ADMISSION_BRANCH_HEADROOM``
    (64), which is sized for kev's synthetic records: a record here can carry
    dozens of questions.

    Two of ``NONE_OPTIONS``' entries carry ``None`` as their description and are
    *not* errors: kev's ``api.option_text`` renders those as the bare key
    (``"none_of_the_above"``), which is why the text is built by that function
    rather than by reading the tables' values. The count is one option span —
    ``[<opt>] + tokens(option) + [</opt>]``, i.e. ``state_tokens`` plus the
    closing delimiter.
    """
    from kev.api import option_text
    from kev.data import DISTRACTORS, NONE_OPTIONS

    options = [option_text(key, text) for key, text in NONE_OPTIONS]
    options += [option_text(key, text) for key, text in DISTRACTORS.items()]
    return max((state_tokens(tokenizer, option) + 1 for option in options), default=0)


def measure(records: Iterable[Any], measure_state: Measure) -> list[int]:
    """Token counts for records, through a caller-supplied measurer.

    Records arrive in either of the two shapes this project uses: the factory's
    ``Record`` (:mod:`sckevtype.contracts.schema`) and the JSONL dict a reader
    hands back. The corpus QC cannot import kev (the ``data`` extra does not
    install it), so it takes the counting function as an argument; this is the
    loop it would otherwise repeat.
    """
    return [measure_state(record["state"] if isinstance(record, dict) else record.state) for record in records]
