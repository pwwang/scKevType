"""QC: the probes the paper reports about its own corpus.

Five checks, all computed from the written records (no re-derivation, no network):

* **dataset identifiability** — can a plain classifier tell which dataset a state
  came from? If it can, a model trained on this corpus can shortcut "which
  dataset is this" into "which type is this", and every cross-dataset number in
  the paper is inflated. Reported with a label-permutation baseline, because an
  accuracy is only meaningful next to what chance looks like for the same
  features.
* **label coverage** — cells per dataset and type that actually reached the corpus.
* **family counts** — how many records of each family, and how balanced their
  targets are (an abstain target that dominates would teach "always abstain").
* **panel-miss pairing** — every miss record must have its correct-panel sibling
  in the same group; an unpaired miss is a record whose answer is guessable from
  the option list alone.
* **state length** — the distribution of rendered lengths against the model's
  token budget, in symbols *and* — when a tokenizer is supplied — in the tokens
  kev's encoder will actually spend. Reported, never truncated: silently dropping
  genes to fit a budget would change the state contract, which is a paper claim.
  The token count is the one that decides whether the corpus can be trained at
  all (a state is ~1,574 tokens, not the ~600 a character heuristic suggests), so
  a corpus whose states cannot be afforded is visible here, at build time, rather
  than on a cluster three minutes into a GPU allocation.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence

import numpy as np

from ..contracts import budget
from ..contracts.budget import Measure, measure
from ..contracts.render import FilterConfig, parse_rank_list
from ..contracts.schema import Record, RecordFamily
from . import seeded_rng


def state_symbols(record: Record, cfg: FilterConfig) -> list[str]:
    return [symbol for symbol, _ in parse_rank_list(record.state, cfg)]


def dataset_identifiability(
    records: Sequence[Record],
    cfg: FilterConfig,
    *,
    seed: int,
    n_permutations: int = 20,
    test_frac: float = 0.3,
    max_features: int | None = None,
) -> dict:
    """Accuracy of a multinomial naive-Bayes dataset classifier, and its null.

    Groups (cells) — not records — are split, so the probe cannot win by seeing
    another record of the same cell. ``n_permutations`` shuffles the dataset
    labels across groups and re-fits, giving the baseline the accuracy has to
    beat: a corpus is only confounded if the observed accuracy is far above what
    the same features achieve with the labels destroyed.
    """
    groups = sorted({record.meta.group_id or record.meta.cell_id or record.state for record in records})
    if len(groups) < 2:
        return {"groups": len(groups), "accuracy": None, "note": "not enough groups to probe"}
    rng = seeded_rng(seed, "identifiability")
    n_test = max(1, round(test_frac * len(groups)))
    test_groups = set(rng.sample(groups, n_test))
    rows = [(group, record) for group, record in ((record.meta.group_id or record.meta.cell_id or record.state, record) for record in records)]
    train_rows = [(group, record) for group, record in rows if group not in test_groups]
    test_rows = [(group, record) for group, record in rows if group in test_groups]
    if not train_rows or not test_rows:
        return {"groups": len(groups), "accuracy": None, "note": "one side of the split is empty"}

    counts = Counter(symbol for _, record in rows for symbol in state_symbols(record, cfg))
    features = [symbol for symbol, _ in counts.most_common(max_features)] if max_features else sorted(counts)
    index = {symbol: position for position, symbol in enumerate(features)}
    labels = sorted({record.meta.dataset_id for record in records})
    label_index = {label: position for position, label in enumerate(labels)}
    x_train = _presence(train_rows, index, cfg)
    x_test = _presence(test_rows, index, cfg)
    y_train = np.array([label_index[record.meta.dataset_id] for _, record in train_rows])
    y_test = np.array([label_index[record.meta.dataset_id] for _, record in test_rows])
    observed = _accuracy(x_train, y_train, x_test, y_test, len(labels))
    baseline = []
    for _ in range(n_permutations):
        # Shuffled with the same seeded RNG the probe already drew its split from, not
        # numpy's global one: the null has to be reproducible from (seed, records).
        shuffled = [int(label) for label in y_train]
        rng.shuffle(shuffled)
        baseline.append(_accuracy(x_train, np.array(shuffled), x_test, y_test, len(labels)))
    above = sum(1 for value in baseline if value >= observed)
    return {
        "groups": len(groups),
        "records": len(rows),
        "features": len(features),
        "datasets": labels,
        "accuracy": round(observed, 6),
        "baseline_mean": round(float(np.mean(baseline)), 6) if baseline else None,
        "baseline_max": round(float(np.max(baseline)), 6) if baseline else None,
        "permutations": n_permutations,
        "permutations_at_or_above": above,
        "permutation_p": round((above + 1) / (n_permutations + 1), 6) if n_permutations else None,
    }


def _presence(rows, index: dict[str, int], cfg: FilterConfig) -> np.ndarray:
    """One row per record: 1 where the symbol is in the state (symbol presence only)."""
    matrix = np.zeros((len(rows), len(index)), dtype=np.float32)
    for row, (_, record) in enumerate(rows):
        for symbol in state_symbols(record, cfg):
            position = index.get(symbol)
            if position is not None:
                matrix[row, position] = 1.0
    return matrix


def _accuracy(x_train, y_train, x_test, y_test, n_classes: int) -> float:
    """Add-one-smoothed multinomial naive Bayes, fitted and scored in one place."""
    log_prior = np.log(np.bincount(y_train, minlength=n_classes) + 1.0)
    log_prior -= np.log(len(y_train) + n_classes)
    counts = np.zeros((n_classes, x_train.shape[1]), dtype=np.float64)
    np.add.at(counts, y_train, x_train)
    log_likelihood = np.log(counts + 1.0)
    log_likelihood -= np.log(counts.sum(axis=1, keepdims=True) + x_train.shape[1])
    scores = x_test @ log_likelihood.T + log_prior
    return float(np.mean(np.argmax(scores, axis=1) == y_test))


def label_coverage(records: Sequence[Record]) -> dict:
    """Cells per dataset and type that reached the corpus."""
    coverage: dict[str, Counter] = defaultdict(Counter)
    for record in records:
        coverage[record.meta.dataset_id][record.meta.cl_id] += 1
    return {
        dataset: {"cells": sum(types.values()), "types": len(types), "by_type": {cl: n for cl, n in sorted(types.items())}}
        for dataset, types in sorted(coverage.items())
    }


def family_counts(records: Sequence[Record]) -> dict:
    """Records per family, with the target balance that keeps a family learnable."""
    families: dict[str, dict] = {}
    for family in RecordFamily:
        subset = [record for record in records if record.meta.family == family]
        entry: dict = {"records": len(subset)}
        if subset and family in (RecordFamily.MARKER_NOUL,):
            answers = Counter(bool(answer) for record in subset for answer in record.answers.values())
            entry["targets"] = {"true": answers[True], "false": answers[False]}
        elif subset:
            answers = Counter(str(answer) for record in subset for answer in record.answers.values())
            entry["targets"] = {answer: n for answer, n in sorted(answers.items(), key=lambda item: (-item[1], item[0]))[:8]}
        families[family.value] = entry
    return families


def panel_miss_pairing(records: Sequence[Record]) -> dict:
    """Every panel-miss record must have its correct-panel sibling in the same group.

    The sibling is identified structurally, without the definitions: same cell,
    exactly one more option, and its answer is that extra option. That is the
    contrast kev's ``none_pair`` describes — same cell, same abstain wording, the
    true type present in one record and absent in the other — so a corpus where it
    fails has a miss record whose answer is guessable from the option list alone.
    """
    by_group: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        if record.meta.group_id:
            by_group[record.meta.group_id].append(record)
    paired, siblings, unpaired = 0, 0, []
    for group_id, group in sorted(by_group.items()):
        for miss in (record for record in group if record.meta.family == RecordFamily.PANEL_MISS):
            miss_keys = {key for question in miss.questions for key in question.criteria}
            sibling = _find_sibling(group, miss, miss_keys)
            if sibling is None:
                unpaired.append(group_id)
            else:
                paired += 1
                siblings += sibling
    return {
        "miss_records": paired + len(unpaired),
        "paired": paired,
        "unpaired": len(unpaired),
        "unpaired_groups": unpaired[:8],
        "siblings": siblings,
    }


def _find_sibling(group: Sequence[Record], miss: Record, miss_keys: set[str]) -> int | None:
    """How many correct-panel siblings of ``miss`` are in its group (None if none)."""
    found = 0
    for record in group:
        if record.meta.family != RecordFamily.PANEL_CHOICE or record.meta.cl_id != miss.meta.cl_id:
            continue
        for question in record.questions:
            keys = set(question.criteria)
            answer = record.answers[question.qid]
            if miss_keys <= keys and len(keys) == len(miss_keys) + 1 and answer not in miss_keys and answer in keys:
                found += 1
                break
    return found or None


def state_lengths(
    records: Sequence[Record],
    cfg: FilterConfig,
    *,
    max_tokens: int,
    tokens_per_symbol: float = 3.0,
    tokenize: Measure | None = None,
) -> dict:
    """The symbol-count histogram, and the exact token count when a tokenizer is supplied.

    ``tokens_per_symbol`` is a measured constant supplied by the corpus config
    (the tokenizer's cost for ``"SYMBOL 12 | "``), and ``max_tokens`` is kev's
    ``MAX_STATE``. The histogram is exact; the token estimate is reported beside
    the counts that produced it so the estimate can be recomputed.

    ``tokenize`` is the corpus's pinned tokenizer, as a ``state -> tokens``
    function (:func:`sckevtype.contracts.budget.state_tokens`). It is a parameter
    rather than an import because the ``data`` extra does not install kev or
    transformers: a corpus built without one reports ``tokens: null`` and the
    estimate, which is a guess, not a measurement. The ``tokens`` block and its
    ``over_budget`` are kev's own count against the same ``max_tokens``, so a
    corpus that cannot be afforded says so in the report instead of failing to
    train later.
    """
    lengths = Counter(len(state_symbols(record, cfg)) for record in records)
    symbols = sorted(lengths)
    over = sum(n for length, n in lengths.items() if length * tokens_per_symbol > max_tokens)
    measured = measure(records, tokenize) if tokenize else []
    return {
        "records": len(records),
        "min": symbols[0] if symbols else 0,
        "max": symbols[-1] if symbols else 0,
        "mean": round(sum(length * n for length, n in lengths.items()) / len(records), 3) if records else 0.0,
        "histogram": {str(length): lengths[length] for length in symbols},
        "max_tokens": max_tokens,
        "tokens_per_symbol": tokens_per_symbol,
        "estimated_max_tokens": round((symbols[-1] if symbols else 0) * tokens_per_symbol, 1),
        "over_budget": over,
        "at_render_cap": lengths.get(cfg.max_genes, 0),
        # Measured, not estimated: the tokens kev's encoder spends on these states,
        # against the same budget. `over_budget` above is the estimate's count; this
        # one is the number that decides whether a run can start.
        "tokens": None if not tokenize else {
            **budget.summarize(measured),
            "max_tokens": max_tokens,
            "over_budget": budget.over_budget(measured, max_tokens),
        },
    }


def report(records: Sequence[Record], cfg: FilterConfig, *, seed: int, **kwargs) -> dict:
    """Every probe in one payload (the corpus's ``qc.json``)."""
    return {
        "counts": {"records": len(records)},
        "checks": {
            "families": family_counts(records),
            "coverage": label_coverage(records),
            "pairing": panel_miss_pairing(records),
        },
        "state_lengths": state_lengths(
            records, cfg, **{k: v for k, v in kwargs.items() if k in ("max_tokens", "tokens_per_symbol", "tokenize")}
        ),
        "identifiability": dataset_identifiability(
            records, cfg, seed=seed, **{k: v for k, v in kwargs.items() if k in ("n_permutations", "test_frac", "max_features")}
        ),
    }
