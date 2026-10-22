"""The suite builder: a Step-2 corpus in, a suite ``kev.train`` can load out.

    python -m sckevtype.train.suite --corpus artifacts/corpus/smoke --out artifacts/suites/smoke

A corpus is already *almost* a kev suite — the same JSONL shape, the same state
and question fields — but it carries the data factory's metadata and not the
trainer's. What this module writes is the trainer's:

* ``train.jsonl`` / ``calibration.jsonl`` / ``development.jsonl`` / ``test.jsonl``
  — the corpus partitions with kev's ``_meta`` contract filled in. The records are
  normalised by ``kev.data.load_records`` itself, not by a copy of its rules, so
  the ``id`` / ``source`` / ``text_sha256`` a run keys its seeds off are kev's;
* ``manifest.json`` — sha256, byte size, record and question counts per file,
  the counts by family and question type, and the digest of the corpus it came
  from. ``kev.suite.load_split`` reads the per-file entries and verifies them.

``--cap-test`` writes a **stratified sample of the test partition** instead of all of it, for an
evaluation that has to fit a wall clock: see :func:`cap_test_partition`. The training suite is never
built with it, and calibration is never capped at any setting — it is the partition every fit reads.

The ``context`` block is **measured, not restated** (:mod:`sckevtype.contracts.budget`).
Every state is tokenized with the pinned tokenizer before anything is written, the
observed maximum becomes the recorded ``max_state`` (rounded up to a multiple of
64), and a record that does not fit is a build failure rather than a training run
that dies on its first batch. ``--truncate`` is the opt-in alternative, and it
truncates the bytes instead of only counting them. The corpus is still read-only:
truncation cuts tokens the way ``kev.model.encode`` does, which is not a second
renderer — the rendered list is not re-generated, only cut.

Two properties, the same two :mod:`sckevtype.data.build` holds itself to:

**Byte-identical rebuilds.** Same corpus bytes ⇒ same suite bytes, into any
output directory. Nothing here records a timestamp, a path or an iteration order
that comes from the filesystem, and the JSON is written with kev's own writers
so the formatting matches what a frozen kev suite looks like.

**Refusal to overwrite.** An existing file with different bytes is an error
unless ``--force``. A suite that has been trained on must not silently become a
different suite under the same name.

The corpus is read-only here: this step converts, it does not re-render, so the
one canonical renderer (:func:`sckevtype.contracts.render.render_rank_list`)
stays the only thing that has ever written a state.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from ..contracts import budget as state_budget
from ..contracts import paths
from ..contracts.freeze import ENCODING, file_digest, git_commit, read_json
from ..contracts.schema import SIZE_BIN_EDGES, size_bin

#: The suite manifest format this writes. kev's own frozen suites are version 2;
#: this is a *different* manifest (it credits a corpus rather than an admission
#: report), so it says so instead of impersonating one.
MANIFEST_VERSION = "sckevtype-1"

#: The default ``--cap-floor``: the smallest support a summarised group may rest on. Records for a
#: sampled ``(atlas, family)`` stratum, question rows for a panel-size bin.
DEFAULT_CAP_FLOOR = 150

#: The partition a cap applies to. Never ``calibration``: every temperature, quantile and threshold
#: in :mod:`sckevtype.eval` is fitted on it, so shrinking it changes *what* is reported rather than
#: how precisely.
_CAP_PARTITION = "test"

#: The frozen panel-size ladder in ladder order — ``1-1``, ``2-4``, ``5-9``, …. The histogram below
#: is keyed on these bins rather than on an exact option count because that is the stratum the
#: plan's F4 binned ECE is defined in, and it is a bin that has to carry its support.
_PANEL_SIZE_LADDER = tuple(size_bin(options) for options in (1, *SIZE_BIN_EDGES))

#: The partition names, in the order they are written. kev.suite.SPLITS is the
#: authority; this is only the fallback when kev is not importable.
_FALLBACK_SPLITS = ("train", "calibration", "development", "test")


def _splits() -> tuple[str, ...]:
    """kev's partition names, so a suite is written in the order kev reads them."""
    try:
        from kev.suite import SPLITS

        return tuple(SPLITS)
    except ImportError:  # pragma: no cover - kev is installed wherever this runs for real
        return _FALLBACK_SPLITS


def _write(writer: Any, value: Any, path: Path, *, force: bool) -> None:
    """Write ``value`` with kev's own writer unless different bytes are already there.

    The bytes are produced by ``kev.suite.write_jsonl`` / ``write_json`` rather
    than by a copy of their formatting, so a suite written here is formatted
    exactly like a frozen kev suite. They are staged in a sibling temp file
    first: a refusal must not leave a half-overwritten partition behind, and an
    identical rewrite (the rebuild case) must be a no-op rather than an error.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(path.name + ".tmp")
    writer(staging, value)
    if path.exists() and not force and path.read_bytes() != staging.read_bytes():
        staging.unlink()
        raise ValueError(f"{path} already exists with different content (pass --force to overwrite)")
    os.replace(staging, path)


def read_corpus(corpus_dir: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """The corpus manifest, with every file it lists re-hashed and re-counted.

    A corpus is the *input* of a suite, so a suite over a corpus that no longer
    has the bytes its manifest records would pin the wrong thing. This is the
    check that makes the ``corpus`` block of the suite manifest mean something.
    """
    corpus_dir = Path(corpus_dir)
    manifest_path = corpus_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"no corpus manifest at {manifest_path}")
    manifest = read_json(manifest_path)
    for name, entry in sorted(manifest.get("files", {}).items()):
        path = corpus_dir / name
        if not path.exists():
            raise ValueError(f"corpus file is missing: {path}")
        digest = file_digest(path)
        if digest != entry["sha256"]:
            raise ValueError(f"corpus digest mismatch: {name} (manifest {entry['sha256'][:12]}…, actual {digest[:12]}…)")
        if "records" in entry:
            lines = sum(1 for line in path.read_text(encoding=ENCODING).splitlines() if line.strip())
            if lines != entry["records"]:
                raise ValueError(f"corpus record count mismatch: {name} (manifest {entry['records']}, actual {lines})")
    return manifest, {"manifest_sha256": file_digest(manifest_path), "manifest": manifest}


def corpus_records(corpus_dir: str | Path, split: str) -> list[dict[str, Any]]:
    """One partition of the corpus, normalised by kev's own record loader.

    ``kev.data.load_records`` fills in the fields a run keys off (``id``,
    ``source``, ``text_sha256``, ``row``, ``variant``) and the corpus's own
    metadata survives on top of them. Its ``id`` is line-based and so repeats
    across partitions, which nothing in training minds but everything in a report
    does — so it is re-prefixed with the split here.
    """
    from kev.data import load_records

    path = Path(corpus_dir) / f"{split}.jsonl"
    try:
        records = load_records(path)
    except Exception as error:
        raise ValueError(f"{path}: kev cannot read this partition: {error}") from error
    for record in records:
        record["_meta"]["id"] = f"{split}/{record['_meta']['id']}"
        record["_meta"]["split"] = split
    return records


def _counts(records: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """Records by family and questions by type — the two axes a run reports on."""
    families = Counter(str(record["_meta"].get("family", "unknown")) for record in records)
    questions = Counter(str(question.get("type", "unknown")) for record in records for question in record["questions"].values())
    return {"per_family": dict(sorted(families.items())), "per_question_type": dict(sorted(questions.items()))}


def _panel_size_histogram(records: list[dict[str, Any]]) -> dict[str, int]:
    """Question rows per frozen panel-size bin, in ladder order.

    Rows, not records: :func:`sckevtype.eval.run.score_partition` emits one row per question, so a
    bin's support in the report is the number of questions whose panel sits in it — which is also
    why the floor below is checked in rows. Bins the records do not reach are absent rather than
    zero, so the caller can tell "no support" from "not a bin here".
    """
    rows = Counter(size_bin(len(question["criteria"]))
                   for record in records for question in record["questions"].values())
    return {name: rows[name] for name in _PANEL_SIZE_LADDER if name in rows}


def cap_test_partition(records: list[dict[str, Any]], *, cap: int, floor: int,
                       seed: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """A stratified sample of the test partition, and the manifest block that records the draw.

    The evaluation does not fit the cluster's wall clock at the full test split, and the sample size
    is the only honest lever: the alternative is to change the precision the report is computed at,
    which would break the property the CLI is for (a report is a deterministic function of its
    inputs). So the rows that are scored are chosen here, before anything is scored, and the choice
    is written down.

    **Stratified by ``(dataset_id, family)``**, the two axes the report breaks down on: atlas, so an
    atlas is not dropped by a draw that lands on another; family, so the families are all represented
    at the rate they occur. **Proportional allocation** to each stratum's share of the test split,
    **raised to ``floor``** for any stratum that proportion would thin below it — a stratum is
    allocated records here, and this is a floor on records, while the panel-size bins below are
    floors on rows.

    **Deterministic**: the strata are walked in sorted key order and each draws from a generator
    seeded on ``(seed, dataset, family)`` alone, so the result depends on none of the iteration
    orders a dict or a filesystem supplies. ``seed`` is the corpus manifest's own seed — recorded in
    the block below, so the draw is re-derivable from the frozen corpus manifest.

    **The panel-size bins are asserted, not hoped for.** Every bin that has rows in the sample must
    hold at least ``floor`` of them, or this raises and the suite is not written: a bin whose ECE
    rests on a handful of rows is not a noisy estimate, it is a number, and the report summarises it
    either way. The block records both histograms, so a bin that the cap emptied is visible as
    ``capped: 0`` rather than silently absent.

    Calibration is not capped by any setting of ``cap``: it is the partition every fit reads.
    """
    if cap < 1:
        raise ValueError(f"cap_test_partition: cap must be at least 1 record, not {cap}")
    if floor < 1:
        raise ValueError(f"cap_test_partition: floor must be at least 1 row, not {floor} — without it a "
                         f"stratum can be thinned until its bin cannot carry a quantile")

    strata: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in records:
        meta = record["_meta"]
        strata.setdefault((str(meta.get("dataset_id", "unknown")), str(meta.get("family", "unknown"))), []).append(record)

    available = len(records)
    sampled: list[dict[str, Any]] = []
    per_stratum: dict[str, dict[str, int]] = {}
    for key in sorted(strata):
        group = strata[key]
        proportional = round(len(group) * cap / available)
        take = min(len(group), max(proportional, floor))
        rng = random.Random(f"cap-test|{seed}|{key[0]}|{key[1]}")
        # Sorted so the sampled partition keeps the corpus's own order within a stratum, and
        # extended stratum by stratum in sorted order, so the written bytes depend on the draw and
        # on nothing else.
        sampled.extend(group[index] for index in sorted(rng.sample(range(len(group)), take)))
        per_stratum[f"{key[0]}/{key[1]}"] = {"available": len(group), "taken": take}

    available_rows = _panel_size_histogram(records)
    capped_rows = _panel_size_histogram(sampled)
    thin = {name: rows for name, rows in capped_rows.items() if rows < floor}
    if thin:
        raise ValueError(
            f"capping test to {cap} records would leave {len(thin)} panel-size bin(s) below the "
            f"{floor}-row floor: " + ", ".join(f"{name} {rows} rows" for name, rows in thin.items())
            + f". The report's binned ECE would be computed on air. Raise --cap-test (currently {cap}) "
              f"or lower --cap-floor; the test partition holds {available} records."
        )

    block = {
        "partition": _CAP_PARTITION,
        "requested": cap,
        # Above `requested` when a thin stratum was raised: the floor is what the draw is for.
        "achieved": len(sampled),
        "available": available,
        "floor": floor,
        "seed": seed,
        "strata_used": len(per_stratum),
        "per_stratum": per_stratum,
        "panel_size_bins": {
            name: {"available": available_rows.get(name, 0), "capped": capped_rows.get(name, 0)}
            for name in _PANEL_SIZE_LADDER if name in available_rows or name in capped_rows
        },
        "rule": "proportional to each (atlas, family) stratum's share of the test split, raised to the "
                "floor; every panel-size bin in the sample holds at least the floor in rows; the test "
                "partition only — calibration is never capped, because every fit reads it",
    }
    return sampled, block


def _validate(records: list[dict[str, Any]], path: Path) -> None:
    """Every record must be a request kev accepts, before it is written into a suite.

    ``kev.data.materialize`` is the serving path training will call on each
    record; a record that fails it is a converter bug, and the suite freeze is
    the last place it is cheap to find out.
    """
    from kev.data import materialize

    for record in records:
        try:
            materialize(record)
        except Exception as error:
            raise ValueError(f"{path}: record {record['_meta'].get('id')!r} is not a kev request: {error}") from error


def load_pinned_tokenizer(key: str = "0.8b", registry: str | Path | None = None) -> tuple[Any, dict[str, str]]:
    """The tokenizer states are measured with: a registry entry's, pinned to its revision.

    ``configs/models.yaml`` is the registry a run draws its base from, so the
    suite is measured with the tokenizer of the model the paper trains rather
    than with whatever ``main`` is today. An entry with no revision is refused
    for the same reason ``sckevtype.train.compute`` refuses one: an unpinned
    measurement describes a moving target. Returns ``(tokenizer, pin)``, and the
    pin is what the manifest records.
    """
    import yaml

    path = Path(registry) if registry else paths.configs_dir() / "models.yaml"
    models = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("models") or {}
    entry = models.get(key)
    if entry is None:
        raise ValueError(f"models.yaml: no {key!r} entry at {path}; known: {sorted(models)}")
    if not entry.get("revision"):
        raise ValueError(f"models.yaml: {key!r} pins no revision — an unpinned tokenizer cannot measure a suite")
    pin = {"id": entry["id"], "revision": entry["revision"]}
    return state_budget.pinned_tokenizer(pin), pin


def measured_context(
    partitions: dict[str, list[dict[str, Any]]],
    tokenizer: Any,
    *,
    max_state: int = 0,
    truncate: bool = False,
    pin: dict[str, str] | None = None,
) -> dict[str, Any]:
    """The ``context`` block for suites of these partitions, measured with ``tokenizer``.

    ``max_state=0`` measures the budget from the data, and the budget is not the
    state maximum. kev derives *three* limits from ``--max_state`` alone
    (``kev.model.training_context``: the row at ``N + 640``, the packed record at
    ``N + 1664``), and a record has to fit all three, so ``max_state`` is the
    smallest multiple of :data:`sckevtype.contracts.budget.BUDGET_UNIT` that
    satisfies all three inequalities at once
    (:func:`sckevtype.contracts.budget.required_budget`) — never below kev's own
    floor. A non-zero ``max_state`` is a *pinned* budget and is enforced against the
    data all the same.

    The three observations it is derived from are recorded flat, so a reader can
    recompute the budget from the manifest alone: ``observed.max_state`` is the
    longest state in the suite, and ``observed.max_row`` / ``observed.max_packed``
    are the longest row (state plus its longest question) and the longest packed
    record (state plus every question) *in the training partition* — the partition
    kev's trainer encodes, unfiltered. Both come from kev's own encoder rather than
    a copy of its packing rule, and include what the trainer's augmentation can
    still add. They are measured at the budget the states were written to fit,
    which cannot truncate them: every state is already within it.

    ``max_branch`` and ``max_packed`` are kev's own limits for the recorded budget
    (``kev.model.training_context``), recorded rather than a second, unenforced set
    of numbers.

    ``truncate`` decides what happens to a record that does not fit: without it
    the build fails (the suite is not written), with it the state is cut to the
    tokens ``kev.model.encode`` would have kept and the count is recorded. The
    point of the two being one function is that the recorded numbers cannot
    disagree with the data they describe — the ``observed`` block is measured
    again after any truncation, and a budget that still cannot hold the records
    raises rather than being written down.

    ``observed`` describes the records as *written*, after any cut, so
    ``over_budget`` is always 0 against the recorded budget and ``truncated`` counts
    how many states had to be cut to get there. Every recorded limit is recomputable
    from it.
    """
    from kev.model import MAX_STATE, MAX_TRAIN_STATE, training_context

    lengths = {split: state_budget.state_lengths(tokenizer, (record["state"] for record in records))
               for split, records in partitions.items()}
    empty = [split for split, values in lengths.items() if not values]
    if empty:
        raise ValueError(f"{empty}: partition(s) with no records — a suite needs states to measure")
    observed_max = max(max(values) for values in lengths.values())
    state_required = state_budget.budget_for(observed_max)
    if state_required > MAX_TRAIN_STATE:
        raise ValueError(
            f"required budget {state_required} exceeds kev's MAX_TRAIN_STATE ({MAX_TRAIN_STATE}): the longest state is "
            f"{observed_max} tokens, which no checkpoint may be trained on and still serve its questions"
        )
    if max_state and not MAX_STATE <= max_state <= MAX_TRAIN_STATE:
        raise ValueError(
            f"pinned max_state {max_state} is outside kev's range [{MAX_STATE}, {MAX_TRAIN_STATE}]: below the floor "
            f"a run cannot hand the encoder a length kev serves, above the ceiling the longest state leaves a "
            f"question no room"
        )
    # The budget the *states* are written to fit: never below kev's floor, which is
    # what a small corpus measures to.
    state_limit = max_state or max(state_required, MAX_STATE)

    truncated = _truncate_states(partitions, tokenizer, state_limit) if truncate else 0
    # Re-measured after any truncation, so ``observed`` describes the bytes on disk
    # rather than the bytes the corpus arrived with.
    lengths = {split: state_budget.state_lengths(tokenizer, (record["state"] for record in records))
               for split, records in partitions.items()}
    per_split = {split: state_budget.summarize(values) for split, values in lengths.items()}
    over = sum(state_budget.over_budget(values, state_limit) for values in lengths.values())
    if over and not truncate:
        raise ValueError(
            f"{over} of {sum(len(values) for values in lengths.values())} records exceed the recorded max_state "
            f"{state_limit}: longest state {observed_max} tokens, required budget {state_required} "
            f"(the observed maximum rounded up to a multiple of {state_budget.BUDGET_UNIT}). "
            f"Build with --max_state {state_required}, or --truncate to cut them to {state_limit}."
        )
    if over:
        raise ValueError(
            f"truncation left {over} records over the {state_limit}-token budget: refusing to record a truncation "
            f"that did not happen"
        )
    observed_max = max(summary["max"] for summary in per_split.values())

    # The state is not the only limit kev enforces. ``kev.model.encode`` refuses a row
    # longer than ``max_branch - len(state)`` and ``kev.train.encode_batch`` a record
    # longer than ``max_packed``, and kev derives *both* from ``--max_state`` alone. A
    # suite whose states fit and whose branches do not therefore still dies on its
    # first batch, which is what both earlier versions of this budget did.
    #
    # Measured over the training partition, because that is the partition kev's
    # trainer encodes: ``kev.suite`` leaves a frozen suite's records unfiltered, so
    # whatever this measures is what the first batch will contain. Each count comes
    # from kev's own encoder, at the budget being recorded, and includes what the
    # trainer's augmentation can still add.
    rows, packed = state_budget.encoded_lengths(partitions["train"], tokenizer, max_state=state_limit)
    max_row, max_packed = max(rows), max(packed)

    # The budget itself: the smallest multiple of 64 that satisfies all three of kev's
    # limits at once. The packed term binds on this corpus, which is why the state
    # axis alone recorded a number the data could not train under.
    required = state_budget.required_budget(observed_max, max_row, max_packed)
    ceiling = training_context(MAX_TRAIN_STATE)
    unaffordable = sum(1 for row, pack in zip(rows, packed, strict=True)
                       if row > ceiling["max_branch"] or pack > ceiling["max_packed"])
    if required > MAX_TRAIN_STATE:
        raise ValueError(
            f"required budget {required} exceeds kev's MAX_TRAIN_STATE ({MAX_TRAIN_STATE}): the longest row is "
            f"{max_row} tokens and the longest packed record {max_packed}, and kev derives both limits from one "
            f"number. {unaffordable} of {len(rows)} training records cannot be trained on at any budget kev serves."
        )
    if max_state and required > state_limit:
        raise ValueError(
            f"pinned max_state {state_limit} cannot afford this data: it needs {required}. The longest row is "
            f"{max_row} tokens and the longest packed record {max_packed}, against the "
            f"{training_context(state_limit)['max_branch']} row and {training_context(state_limit)['max_packed']} "
            f"packed limits kev derives from {state_limit}."
        )

    # A measured budget is the derived one; a pinned budget is the operator's, and is
    # recorded as given. kev's own limits for the recorded budget are what a run hands
    # the encoder, so they are recorded rather than a second, unenforced set.
    recorded = state_limit if max_state else required
    limits = training_context(recorded)
    context: dict[str, Any] = {
        "max_state": recorded,
        "max_branch": limits["max_branch"],
        "max_packed": limits["max_packed"],
        "truncate": bool(truncate),
        "source": "measured" if not max_state else "pinned",
        # The three observations the budget was derived from, flat so a reader can
        # recompute ``max_state`` from the manifest alone.
        "observed": {
            "max_state": observed_max,
            "max_row": max_row,
            "max_packed": max_packed,
            "per_split": per_split,
            "n": sum(len(values) for values in lengths.values()),
            "over_budget": sum(state_budget.over_budget(values, recorded) for values in lengths.values()),
            "truncated": truncated,
        },
    }
    if pin:
        context["tokenizer"] = dict(pin)
    return context


def _truncate_states(partitions: dict[str, list[dict[str, Any]]], tokenizer: Any, max_state: int) -> int:
    """Cut every over-long state to ``max_state`` tokens in place. Returns how many were cut."""
    truncated = 0
    for records in partitions.values():
        for record in records:
            if state_budget.state_tokens(tokenizer, record["state"]) > max_state:
                record["state"] = state_budget.truncate_state(tokenizer, record["state"], max_state)
                truncated += 1
    return truncated


def build_suite(corpus_dir: str | Path, out_dir: str | Path, *, force: bool = False, tokenizer: Any = None,
                max_state: int = 0, truncate: bool = False, registry: str | Path | None = None,
                cap_test: int = 0, cap_floor: int = DEFAULT_CAP_FLOOR) -> dict[str, Any]:
    """Build the suite over ``corpus_dir`` into ``out_dir``. Returns its manifest.

    Nothing is written until every partition has been read, validated and
    measured: a suite that cannot be trained must not exist on disk, which is
    the whole point of measuring before writing.

    ``cap_test`` samples the test partition (:func:`cap_test_partition`) and records the draw in the
    manifest; ``0`` — the default — writes it whole and adds no ``cap`` block, so a suite built
    without it is byte-for-byte the suite this module has always written.
    """
    from kev.suite import write_json, write_jsonl

    corpus_dir, out_dir = Path(corpus_dir), Path(out_dir)
    manifest, corpus = read_corpus(corpus_dir)
    name = manifest.get("name", corpus_dir.name)

    partitions: dict[str, list[dict[str, Any]]] = {}
    cap_block: dict[str, Any] | None = None
    for split in _splits():
        path = corpus_dir / f"{split}.jsonl"
        if not path.exists():
            continue
        records = corpus_records(corpus_dir, split)
        if split == _CAP_PARTITION and cap_test:
            # Before `_validate`, so the records the cap dropped are neither validated nor
            # measured: nothing downstream can disagree with what is written.
            records, cap_block = cap_test_partition(records, cap=cap_test, floor=cap_floor, seed=manifest.get("seed"))
        _validate(records, path)
        partitions[split] = records
    if "train" not in partitions:
        raise ValueError(f"{corpus_dir}: no train.jsonl — a suite needs a training partition")
    if cap_test and _CAP_PARTITION not in partitions:
        raise ValueError(f"a cap of {cap_test} records was asked for, but {corpus_dir} has no "
                         f"{_CAP_PARTITION}.jsonl to draw it from")

    pin = None
    if tokenizer is None:
        tokenizer, pin = load_pinned_tokenizer(registry=registry)
    context = measured_context(partitions, tokenizer, max_state=max_state, truncate=truncate, pin=pin)

    files: dict[str, dict[str, Any]] = {}
    for split, records in partitions.items():
        target = out_dir / f"{split}.jsonl"
        _write(write_jsonl, records, target, force=force)
        files[target.name] = {
            "sha256": file_digest(target),
            "bytes": target.stat().st_size,
            "kind": "data",
            "records": len(records),
            "questions": sum(len(record["questions"]) for record in records),
            **_counts(records),
        }

    # Which sources a run may train on is read off the partitions rather than
    # declared: a source that reaches train.jsonl is trainable, and anything that
    # only ever appears in development/test is eval-only. kev.suite.validate_training
    # enforces both against these lists.
    trained = {str(record["_meta"].get("source", "")) for record in partitions["train"]}
    every = {str(record["_meta"].get("source", "")) for records in partitions.values() for record in records}
    suite_manifest = {
        "name": name,
        "version": MANIFEST_VERSION,
        "seed": manifest.get("seed"),
        "corpus": {
            "path": str(corpus_dir),
            "name": name,
            "version": manifest.get("version"),
            "config_digest": manifest.get("config_digest"),
            "manifest_sha256": corpus["manifest_sha256"],
        },
        "kev": {"commit": git_commit(Path(__file__).resolve().parents[2] / "kev")},
        # The context these records were measured to fit: kev's own limits for the
        # measured max_state, plus what was observed and how the budget was arrived at.
        "context": context,
        # A corpus is text: which base it is trained on is the run's choice, so the
        # suite pins no revision and train.py requires --base_revision instead.
        "base_revisions": {},
        "trainable_sources": sorted(trained),
        "eval_only_sources": sorted(every - trained),
        "holdout_sources": [],
        "files": files,
        "counts": {
            "records": sum(entry["records"] for entry in files.values()),
            "questions": sum(entry["questions"] for entry in files.values()),
            "per_split": {split: len(records) for split, records in partitions.items()},
            **_counts([record for records in partitions.values() for record in records]),
        },
        "splits": list(partitions),
    }
    if cap_block is not None:
        # Added only when a cap was taken: an uncapped manifest is the one the frozen real-v2 was
        # written with, and a key that is always present would change those bytes.
        suite_manifest["cap"] = cap_block
    _write(write_json, suite_manifest, out_dir / "manifest.json", force=force)
    return suite_manifest


def _report(manifest: dict[str, Any]) -> None:
    print(f"{manifest['name']} ({manifest['version']}): {manifest['counts']['records']} records, "
          f"{manifest['counts']['questions']} questions over {manifest['counts']['per_split']}")
    for name, entry in sorted(manifest["files"].items()):
        print(f"  {name:20s} {entry['sha256'][:12]}  {entry['bytes']:>9} bytes  {entry['records']:>5} records  "
              f"{entry['per_family']}")
    print(f"  trainable sources: {manifest['trainable_sources']}  eval-only: {manifest['eval_only_sources']}")
    context = manifest["context"]
    print(f"  context: max_state {context['max_state']} / max_branch {context['max_branch']} / "
          f"max_packed {context['max_packed']} ({context['source']}"
          f"{', ' + str(context['tokenizer']['id']) if context.get('tokenizer') else ''})")
    observed = context["observed"]
    print(f"    observed maxima: state {observed['max_state']}  row {observed['max_row']}  "
          f"packed {observed['max_packed']} -> max_state {context['max_state']}")
    for split, observed in context["observed"]["per_split"].items():
        print(f"    {split:14s} n {observed['n']:>6}  min {observed['min']:>5}  median {observed['median']:>5}  "
              f"p95 {observed['p95']:>5}  max {observed['max']:>5}  over {context['max_state']}: "
              f"{context['observed']['over_budget']}  truncated: {context['observed']['truncated']}")
    if cap := manifest.get("cap"):
        print(f"  cap: test {cap['achieved']} of {cap['available']} records (asked for {cap['requested']}), "
              f"{cap['strata_used']} strata, floor {cap['floor']}, seed {cap['seed']}")
        for name, rows in cap["panel_size_bins"].items():
            print(f"    panel size {name:>6}: {rows['capped']:>7} of {rows['available']:>7} rows")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sckevtype.train.suite", description="Wrap a Step-2 corpus as a kev-loadable suite.")
    parser.add_argument("--corpus", required=True, help="corpus directory written by sckevtype.data.build")
    parser.add_argument("--out", required=True, help="output suite directory")
    parser.add_argument("--force", action="store_true", help="overwrite files whose bytes changed")
    parser.add_argument("--max_state", type=int, default=0,
                        help="pin the state budget (0 = measure it: the longest state, rounded up to a multiple of 64)")
    parser.add_argument("--truncate", action="store_true",
                        help="cut over-long states to the budget and record how many, instead of failing")
    parser.add_argument("--model", default="0.8b", help="registry key whose tokenizer measures the states (configs/models.yaml)")
    parser.add_argument("--models_yaml", default=None, help="registry to read (default configs/models.yaml)")
    parser.add_argument("--cap-test", type=int, default=0, metavar="N",
                        help="score N sampled test records instead of the whole partition (0 = uncapped). "
                             "Test only — calibration is never capped. Pin --max_state as well: a capped "
                             "partition is not what the training suite measured its budget from")
    parser.add_argument("--cap-floor", type=int, default=DEFAULT_CAP_FLOOR, metavar="M",
                        help=f"floor on the sample (default {DEFAULT_CAP_FLOOR}): records per (atlas, family) "
                             f"stratum, and rows per panel-size bin — a build whose bins fall under it is refused")
    args = parser.parse_args(argv)
    if args.cap_test < 0 or (args.cap_test and args.cap_floor < 1):
        parser.error("--cap-test must be >= 0, and --cap-floor >= 1 when --cap-test is set")
    try:
        manifest = build_suite(args.corpus, args.out, force=args.force, max_state=args.max_state,
                               truncate=args.truncate, registry=args.models_yaml,
                               cap_test=args.cap_test, cap_floor=args.cap_floor)
    except (SystemExit, ValueError, KeyError, FileNotFoundError, ImportError) as error:
        print(f"suite build failed: {error}", file=sys.stderr)
        return 1
    _report(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
