"""E3.1–E3.4: ask Jev our own questions, then read its answers with *our* metrics.

    uv run --no-sync python -m sckevtype.jev.probe sample       --out reports/jev-smoke
    uv run --no-sync python -m sckevtype.jev.probe ask          --out reports/jev-smoke
    uv run --no-sync python -m sckevtype.jev.probe ask-batched  --out reports/jev-full
    uv run --no-sync python -m sckevtype.jev.probe evaluate     --out reports/jev-smoke

``--corpus`` is a **top-level** flag: it goes before the subcommand, and after it argparse rejects it
as an unknown argument of the subparser (``unrecognized arguments``). Both forms work; the default is
``<root>/runs/real-v2``, which is why the four lines above need no flag:

    uv run --no-sync python -m sckevtype.jev.probe --corpus runs/real-v2 sample --out reports/jev-smoke

Three steps, because they spend differently and only the last one interprets anything:

``sample``
    Choose the cells. Deterministic, seeded, stratified by dataset, and written to
    ``sample.json`` **before** any call — the selection is a decision that must be visible to a
    reader of the result, not a side effect of how the loop happened to iterate. A cell is eligible
    when it carries all four families; ``--cells-per-dataset 0`` takes every eligible cell.
``ask``
    Send one request per (cell, family) ask. Every response is written to ``raw/`` as it arrives,
    request and response together, and a request whose file already exists is skipped — so a run
    that hits a rate limit, a deadline or the budget resumes where it stopped. **Nothing is
    interpreted here**: the raw bodies are the evidence, and ``evaluate`` can be re-run against them
    without spending another cent.
``ask-batched``
    The same records, asked in as few requests as their **states** allow: every family that agrees
    with another on its state goes into one body with the question id prefixed ``<family>/``, so a
    cell costs one request per distinct state rather than one per family. The grouping is decided by
    byte-identical state, recorded per cell in ``ask-log.json``, and the rows are built by the same
    :func:`rows_of_answer` the singleton path uses — so the two asks produce the same row schema and
    the *only* difference between them is how many requests it took.
``evaluate``
    Rebuild :mod:`sckevtype.eval`'s row schema from the saved bodies and run the same metrics the
    local model is scored with — accuracy, ECE, risk–coverage, and
    :func:`sckevtype.eval.metrics.panel_adequacy` for axis 1 — from the saved bodies only. This step
    makes no network call and needs no key.

What the probe is, and what it is not. It is a **feasibility** measurement: does the frontier model
of this class accept our prompts at all, is it competent on our task, and does its abstention
statistic separate "the panel contains the answer" from "it does not" — the dissociation the paper
claims for the class. It is **not** a benchmark. :func:`report_block` records the population it was
taken on so the numbers cannot be read as more than that, and that population is a flag:
``--cells-per-dataset N`` asks N cells per dataset (the 200-cell probe, a sample) and
``--cells-per-dataset 0`` asks every eligible cell (the census — every cell that carries all four
families, which is a census of the eligible set and still 16% of the 15,626-cell partition).

The cells are drawn from ``runs/real-v2/test.jsonl`` — the partition every reported scKevType number
already comes from — and a cell is only eligible if it carries all four of the families this probe
serves. That restriction is stated rather than hidden: it makes every comparison *within-cell*
(same cell, same panel register, four different questions), which is what makes the reading worth
having, and it does mean the eligible set is not a uniform draw from the partition.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar

from ..contracts.freeze import file_digest, read_json, write_json
from ..contracts.paths import root
from ..eval import abstain_index, abstain_key
from .client import BudgetExceeded, JevClient, JevError
from .request import (
    BATCH_SEPARATOR,
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    abstain_request,
    batched_request,
    option_keys,
    record_request,
    request_digest,
    state_digest,
    state_groups,
)

#: The four families one request each is drawn from, and why each is here:
#:
#: ``panel_choice``      the ordinary annotation question — the label is a real option (axis-1 negative)
#: ``panel_miss``        the same cell with the true type removed (axis-1 positive, and the pair)
#: ``evidence_diluted``  the only family of the four whose target is *sometimes* the abstain option,
#:                       so it is the only one that carries both axis-1 classes **within a family** —
#:                       which is what :func:`sckevtype.eval.metrics.panel_adequacy`'s primary
#:                       statistic is taken on
#: ``marker_noul``      30 per-option "does this definition hold in this state?" questions in one
#:                       request — the API's parallel-questions path, and the cheapest evidence that
#:                       batching works
FAMILIES = ("panel_choice", "panel_miss", "evidence_diluted", "marker_noul")

#: ``(family, ask)`` pairs that may be sent for a cell. ``abstain`` is the second ask
#: :func:`sckevtype.jev.request.abstain_request` builds, and it is only sent when the first ask's
#: choice question offered no abstain option.
ASKS = ("base", "abstain")

#: How many cells per dataset. 200 cells over the four datasets the test partition covers.
CELLS_PER_DATASET = 50

#: The sample's seed. Fixed and recorded, so the 200 cells are reproducible from
#: ``(corpus, seed)`` and a reader can re-derive the population.
SAMPLE_SEED = 20260929

#: ``sckevtype.eval.run``'s target false-positive rate, so the axis-1 detection number is the same
#: operating point the local reports quote.
FPR = 0.05

#: Where the local model's published marginals live, for the comparison block. Read-only.
OURS = "reports/eval-real-v2-eval.json"

#: What a row carries. ``kev.benchmark.prediction_rows`` supplies the first group and
#: ``sckevtype.eval.run.extended_row`` the second; together they are a superset of
#: ``sckevtype.eval.run.DUMP_FIELDS``, so the axis-1 dump is a projection of these rows rather than a
#: second derivation of them.
ROW_FIELDS = ("id", "group", "question", "source", "task", "type", "variant", "split", "family",
              "dataset", "cell", "donor", "panel", "keys", "label", "p", "size", "abstain",
              "abstain_source", "abstain_key", "abstain_text", "abstain_position", "abstain_last",
              "none_p")


class ProbeError(RuntimeError):
    """Something the probe cannot measure as asked, named rather than defaulted."""


# --------------------------------------------------------------------------- the corpus


def corpus_dir(corpus: str | None) -> Path:
    """The corpus directory. Defaults to the in-repo ``runs/real-v2``.

    Not :func:`sckevtype.contracts.paths.runs_dir`: that resolver honours ``SCKEVTYPE_RUNS``, which
    ``scripts/wsl/env.sh`` points at the Linux filesystem for *training* output, while the corpus
    this probe reads lives in the checkout. One of the two would silently be the wrong directory.
    """
    return Path(corpus) if corpus else root() / "runs" / "real-v2"


def read_records(path: Path) -> Iterator[dict[str, Any]]:
    """Every record of a partition file, in file order (the order is the corpus's own)."""
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def index_families(path: Path, wanted: Sequence[str]) -> dict[str, dict[str, int]]:
    """``{group_id: {family: line number}}`` for the records whose family is wanted.

    One streaming pass and no record kept: ``test.jsonl`` is 731 MB, and holding it would cost more
    than the pass. The line number is the handle the ``ask`` step seeks with, so the two steps agree
    on a record without hashing the file twice.
    """
    index: dict[str, dict[str, int]] = {}
    for number, record in enumerate(read_records(path)):
        family = record["_meta"]["family"]
        if family in wanted:
            index.setdefault(record["_meta"]["group_id"], {})[family] = number
    return index


def eligible_by_dataset(index: Mapping[str, Mapping[str, int]],
                        datasets: Mapping[str, str]) -> dict[str, list[str]]:
    """``{dataset: [cell, ...]}`` — the cells that carry **every** family this probe serves.

    Sorted, so the draw below does not depend on the iteration order of a dict. The restriction is
    real and narrows the population: a cell that carries ``panel_choice`` but no ``panel_miss`` has
    no panel-swap counterpart, and comparing two cells instead of one cell twice would put the
    difference between them into every number the probe reports.
    """
    eligible: dict[str, list[str]] = {}
    for group, families in index.items():
        if set(FAMILIES) <= set(families):
            eligible.setdefault(datasets[group], []).append(group)
    return {name: sorted(groups) for name, groups in eligible.items()}


def choose_cells(index: Mapping[str, Mapping[str, int]], per_dataset: int | None, seed: int,
                 datasets: Mapping[str, str]) -> list[str]:
    """``per_dataset`` cells from each dataset, from the cells that carry every wanted family.

    Deterministic in ``(index, per_dataset, seed)``: candidates are sorted before the draw, so the
    sample does not depend on the iteration order of a dict. A dataset with fewer eligible cells
    than asked for contributes what it has — recorded in ``sample.json`` rather than padded from
    another dataset, because a stratum that had to be topped up is a fact about the corpus.
    ``per_dataset=None`` draws **every** eligible cell, which is how the full-population run is
    selected: the same eligibility filter, no sampling on top of it.
    """
    rng = random.Random(seed)
    chosen: list[str] = []
    for name, candidates in sorted(eligible_by_dataset(index, datasets).items()):
        take = len(candidates) if per_dataset is None else min(per_dataset, len(candidates))
        chosen.extend(rng.sample(candidates, take))
    return chosen


def dataset_of(group_id: str) -> str:
    """``"muraro/D28.1_1" -> "muraro"`` — the ``_meta`` split is by ``/`` and the dataset never has one."""
    return group_id.split("/", 1)[0]


# --------------------------------------------------------------------------- sampling


def cmd_sample(args: argparse.Namespace) -> int:
    corpus = corpus_dir(args.corpus)
    partition = corpus / f"{args.split}.jsonl"
    if not partition.exists():
        raise ProbeError(f"no {partition}")
    index = index_families(partition, FAMILIES)
    datasets = {group: dataset_of(group) for group in index}
    eligible = eligible_by_dataset(index, datasets)
    per_dataset = None if args.cells_per_dataset <= 0 else args.cells_per_dataset
    chosen = choose_cells(index, per_dataset, args.seed, datasets)
    if not chosen:
        raise ProbeError(f"no cell in {partition} carries all of {FAMILIES}")

    counts = Counter(dataset_of(group) for group in chosen)
    sample = {
        "probe": "sckevtype.jev", "step": "sample", "corpus": str(partition),
        "corpus_sha256": file_digest(partition), "split": args.split, "families": list(FAMILIES),
        "seed": args.seed, "cells_per_dataset": per_dataset, "cells": len(chosen),
        "why_these_cells": (f"a cell is eligible when it carries all of {list(FAMILIES)}; the draw is "
                            + ("every eligible cell of every dataset (the full-population run)"
                               if per_dataset is None else
                               f"seeded with {args.seed} over the sorted eligible cells of each dataset")),
        "indexed_cells": len(index),
        "eligible_cells": sum(len(groups) for groups in eligible.values()),
        "eligible_by_dataset": {name: len(groups) for name, groups in sorted(eligible.items())},
        "by_dataset": dict(sorted(counts.items())),
        "chosen": sorted(chosen),
        "records": {group: dict(sorted(families.items())) for group, families in
                    ((group, index[group]) for group in sorted(chosen))},
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "sample.json", sample)
    print(f"sampled {len(chosen)} cells ({dict(sorted(counts.items()))}) from "
          f"{sample['eligible_cells']} eligible of {sample['indexed_cells']} indexed")
    print(f"wrote {out / 'sample.json'}")
    return 0


# --------------------------------------------------------------------------- asking


def slug(text: str) -> str:
    """A filename-safe form of a cell id. ``tabula_muris_senis/3-M-5/6`` has two slashes in it."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text)


def ask_path(directory: Path, dataset: str, cell: str, family: str, ask: str) -> Path:
    return directory / f"{slug(dataset)}__{slug(cell)}__{family}__{ask}.json"


def collect_records(partition: Path, sample: Mapping[str, Any]) -> list[tuple[str, str, dict[str, Any]]]:
    """The sampled records, as ``(group, family, record)`` in the sample's own order.

    One streaming pass over the partition, holding only the records the sample names (about 800 of
    89,802 — a few megabytes) rather than seeking to each line: the corpus is 731 MB, and a seek per
    record would re-read its whole prefix every time. The order is the sample's, not the file's, so
    the rows a run produces do not depend on where in the file a record happens to sit.

    The families served for a group are **the sample's own** for that group, not the module's
    four. A sample drawn by :func:`cmd_sample` names exactly those four, so the two agree there and
    the order is unchanged (:data:`FAMILIES` order). A census population assembled from an id list
    may name fewer: v4's test partition carries no ``panel_miss`` record for 2,992 of its 4,035
    ``panel_choice`` cells, and a probe that insisted on all four would either refuse those cells or
    KeyError on them — neither of which is a fact about the corpus. Asking the families the sample
    names is the same census, read for the families that exist.
    """
    wanted = {line: (group, family) for group, families in sample["records"].items()
              for family, line in families.items()}
    found: dict[int, tuple[str, str, dict[str, Any]]] = {}
    for number, record in enumerate(read_records(partition)):
        if number in wanted:
            group, family = wanted[number]
            found[number] = (group, family, record)
    missing = sorted(set(wanted) - set(found))
    if missing:
        raise ProbeError(f"{partition} no longer holds line(s) {missing[:5]}; the sample cannot be served")

    order = {family: index for index, family in enumerate(FAMILIES)}
    tasks = []
    for group in sample["chosen"]:
        families = sample["records"][group]
        for family in sorted(families, key=lambda name: (order.get(name, len(FAMILIES)), name)):
            tasks.append(found[families[family]])
    return tasks


def meta_fields(record: Mapping[str, Any], family: str, split: str) -> dict[str, Any]:
    """The stratification keys, copied off ``_meta`` rather than recomputed (``extended_row``'s rule)."""
    meta = record["_meta"]
    cell = str(meta.get("cell_id") or "unknown")
    dataset = str(meta.get("dataset_id") or "unknown")
    return {"id": f"{split}/{dataset}/{cell}/{family}", "group": str(meta.get("group_id") or "unknown"),
            "split": split, "family": family, "dataset": dataset, "cell": cell,
            "donor": str(meta.get("donor_id") or "unknown"), "panel": str(meta.get("panel_id") or "unknown"),
            "variant": str(meta.get("variant") or "clean"), "source": str(meta.get("source") or "real-v2")}


def served_fields(keys: Sequence[str], criteria: Mapping[str, Any] | None,
                  source: str | None) -> dict[str, Any]:
    """Where the abstain option sat in the prompt as served — ``run.served_row_fields``, restated.

    Read off the keys the *response* came back under, not off the request: an option the model did
    not answer under is not an option it was served, and the two would only differ if the API were
    rewriting the question, which is exactly what a reader needs to be able to see.
    """
    key = abstain_key(keys)
    position = list(keys).index(key) if key is not None else None
    return {"abstain_source": source, "abstain_key": key,
            "abstain_text": (criteria or {}).get(key) if key is not None else None,
            "abstain_position": position,
            "abstain_last": key is not None and position == len(keys) - 1}


def variant_map(questions: Mapping[str, Any], prediction: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """The abstain ask, keyed by the question it re-asks.

    The variant ask is a *second question* about the same state, and it is one question per record
    by construction (:func:`~sckevtype.jev.request.abstain_request` sends only the choice questions
    that lacked an abstain option), so the per-question shape is what a row needs: the option order
    of *that* question, and its own criteria. Keyed rather than merged so a record with two such
    questions cannot silently have the first one's criteria scored against the second one's answer.
    """
    return {qid: {"criteria": question["criteria"], "probabilities": dict(prediction["probabilities"][qid])}
            for qid, question in questions.items()}


def rows_of_answer(record: Mapping[str, Any], family: str, split: str, prediction: Mapping[str, Any],
                   variants: Mapping[str, Mapping[str, Any]] | None) -> list[dict[str, Any]]:
    """One record's prediction as the rows :mod:`sckevtype.eval` scores.

    The abstain handling is :func:`sckevtype.eval.run.score_partition`'s and
    :func:`~sckevtype.eval.run.score_abstain_variant`'s, in that order: a question that already
    offered an abstain option takes its ``none_p`` from the answer it was served under; a choice
    question that offered none takes it from the second ask, keeps ``abstain: None`` (the question
    *as asked* had no such option) and is marked ``injected``. A choice question whose second ask was
    not obtained is a **rejected** row, not an emitted row with ``none_p: None`` — the metric drops
    those silently, and a statistic computed on whichever rows happened to fit is a silent sample.

    ``variants`` is :func:`variant_map`'s output for this record (``None`` when no question needed
    one). A question missing from it raises by name: the second ask is per question, and scoring one
    question's abstain option against another's answer is the failure this shape exists to prevent.
    """
    common = meta_fields(record, family, split)
    rows = []
    for qid, question in record["questions"].items():
        keys = option_keys(question)
        probabilities = prediction["probabilities"][qid]
        total = sum(float(probabilities[key]) for key in keys)
        p = [float(probabilities[key]) / total for key in keys]
        label = keys.index(question["label"]) if question["type"] == "choice" else int(question["label"])
        row = {"question": qid, "task": str(question.get("src") or family), "type": question["type"],
               "keys": keys, "label": label, "p": p, "raw_probability_sum": total,
               "size": len(keys), "answer": (prediction["answers"][qid].get("choice")
                                             if question["type"] == "choice" else prediction["answers"][qid].get("noul")),
               "confidence": prediction["answers"][qid].get("confidence"),
               "served_model": prediction.get("served_model"), "latency_ms": prediction.get("latency_ms")}
        if question["type"] == "choice":
            key = abstain_key(keys)
            row.update(abstain=abstain_index(row),
                       none_p=None if key is None else p[keys.index(key)],
                       **served_fields(keys, question.get("criteria"), "factory" if key is not None else None))
            if row["none_p"] is None:
                entry = (variants or {}).get(qid)
                if entry is None:
                    raise ProbeError(f"{common['id']}/{qid}: no abstain option and no second ask; "
                                     f"axis 1 has no statistic for this row")
                variant_keys = option_keys({**question, "criteria": entry["criteria"]})
                vkey = abstain_key(variant_keys)
                values = entry["probabilities"]
                vtotal = sum(float(values[k]) for k in variant_keys)
                row["none_p"] = float(values[vkey]) / vtotal
                row.update(abstain=None, **served_fields(variant_keys, entry["criteria"], "injected"))
        else:
            row.update(abstain=None, none_p=None, **served_fields(keys, None, None))
        rows.append({**common, **row})
    return rows


def build_rows(record: Mapping[str, Any], family: str, split: str, client: JevClient,
               directory: Path, dataset: str, cell: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Ask one record's questions, save every raw body, and build its rows. Resumable per ask.

    A saved body is reused rather than re-sent, so the second run of an interrupted probe costs
    nothing and produces the bytes the first run would have produced. The saved object holds the
    request beside the response because the request is the thing the claim is about: it is what
    proves the label never left, and it is what a reader would re-send to check the answer.
    """
    directory.mkdir(parents=True, exist_ok=True)
    base_path = ask_path(directory, dataset, cell, family, "base")
    prediction = None
    if base_path.exists():
        saved = read_json(base_path)
        prediction = saved["prediction"]
    else:
        body = record_request(record, client.model)
        prediction = client(record)
        write_json(base_path, {"request": body, "request_sha256": request_digest(body),
                               "response": {"model": prediction["served_model"], "answers": prediction["answers"],
                                            "usage": {"input_tokens": prediction["input_tokens"],
                                                      "output_tokens": prediction["output_tokens"]}},
                               "prediction": prediction})

    variants = None
    variant_request = abstain_request(record, client.model)
    if variant_request is not None:
        variant_path = ask_path(directory, dataset, cell, family, "abstain")
        if variant_path.exists():
            saved_variant = read_json(variant_path)
        else:
            variant = client(record, questions=variant_request["questions"])
            saved_variant = {"request": variant_request, "request_sha256": request_digest(variant_request),
                             "response": {"model": variant["served_model"], "answers": variant["answers"],
                                          "usage": {"input_tokens": variant["input_tokens"],
                                                    "output_tokens": variant["output_tokens"]}},
                             "prediction": variant}
            write_json(variant_path, saved_variant)
        variants = variant_map(saved_variant["request"]["questions"], saved_variant["prediction"])
    return rows_of_answer(record, family, split, prediction, variants), prediction


def cmd_ask(args: argparse.Namespace) -> int:
    """Serve the sample, in one streaming pass, with a bounded pool of in-flight requests.

    Concurrency is threads in **one** process, not processes: the work is waiting on the network, and
    the host this runs on has fork pressure. ``--workers 1`` is the sequential path and produces the
    same rows in the same order — the pool is an ordering-preserving map (results are collected by
    task index), so nothing about the output depends on which request finished first.
    """
    out = Path(args.out)
    sample = read_json(out / "sample.json")
    corpus = corpus_dir(args.corpus)
    partition = corpus / f"{sample['split']}.jsonl"
    if file_digest(partition) != sample["corpus_sha256"]:
        raise ProbeError(f"{partition} changed since the sample was drawn; the sample would name different records")
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    client = JevClient(base_url=args.base_url, model=args.model, budget_usd=args.budget_usd,
                       max_calls=args.max_calls, max_call_usd=args.max_call_usd)
    tasks = collect_records(partition, sample)
    if args.limit:
        tasks = tasks[: args.limit]
    print(f"{len(tasks)} records to serve from {partition.name}; {args.workers} in flight; "
          f"budget ${args.budget_usd}", flush=True)

    reused = 0
    for group, family, _record in tasks:
        directory, cell = dataset_of(group), group.split("/", 1)[1]
        reused += sum(ask_path(raw, directory, cell, family, ask).exists() for ask in ASKS)

    def work(index: int, task: tuple[str, str, dict[str, Any]]) -> tuple[int, list[dict[str, Any]], dict | None]:
        group, family, record = task
        directory, cell = dataset_of(group), group.split("/", 1)[1]
        try:
            rows, _prediction = build_rows(record, family, sample["split"], client, raw, directory, cell)
            return index, rows, None
        except BudgetExceeded:
            raise       # a spend cap is not a bad record: it stops the run, it does not reject a row
        except (JevError, ProbeError, KeyError, ValueError) as error:
            return index, [], {"id": f"{sample['split']}/{directory}/{cell}/{family}", "family": family,
                               "error": f"{type(error).__name__}: {error}"}

    results: list[list[dict[str, Any]]] = [[] for _ in tasks]
    rejected: list[dict[str, Any]] = []
    stopped = None
    try:
        if args.workers > 1:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {pool.submit(work, index, task): index for index, task in enumerate(tasks)}
                for done, future in enumerate(as_completed(futures), start=1):
                    index, rows, failure = future.result()
                    results[index] = rows
                    if failure:
                        rejected.append(failure)
                    if done % args.progress_every == 0 or done == len(tasks):
                        print(f"  [{done:4d}/{len(tasks)}] calls={client.calls:4d} "
                              f"in_tokens={client.input_tokens:8d} ${client.estimated_usd:.4f}", flush=True)
        else:
            for index, task in enumerate(tasks, start=1):
                _, rows, failure = work(0, task)
                results[index - 1] = rows
                if failure:
                    rejected.append(failure)
                if index % args.progress_every == 0 or index == len(tasks):
                    print(f"  [{index:4d}/{len(tasks)}] calls={client.calls:4d} "
                          f"in_tokens={client.input_tokens:8d} ${client.estimated_usd:.4f}", flush=True)
    except (JevError, ProbeError) as error:
        stopped = f"{type(error).__name__}: {error}"
        print(f"stopping: {stopped}", file=sys.stderr)

    rows = [row for group in results for row in group]
    write_rows(out / "rows.jsonl", rows)
    write_ask_log(out, {
        "step": "ask", "records_planned": len(tasks), "records_served": sum(1 for group in results if group),
        "rows": len(rows), "rejected": rejected, "stopped": stopped, "workers": args.workers,
        "accounting": client.accounting(), "asks_reused_from_disk": reused, "model_requested": args.model,
    })
    print(f"\n{len(rows)} rows from {client.calls} calls, {client.input_tokens} input tokens, "
          f"${client.estimated_usd:.4f}; {len(rejected)} rejected, {reused} asks reused from disk")
    print(f"wrote {out / 'rows.jsonl'} and {out / 'ask-log.json'}")
    return 0 if rows else 1


# --------------------------------------------------------------------------- asking, batched


def batch_path(directory: Path, dataset: str, cell: str, index: int) -> Path:
    """Where one batched request for a cell is saved; ``index`` is its state group's position."""
    return directory / f"{slug(dataset)}__{slug(cell)}__batch{index}.json"


def cells_of(tasks: Sequence[tuple[str, str, dict[str, Any]]]) -> list[tuple[str, list[tuple[str, dict[str, Any]]]]]:
    """The sample's records regrouped by cell, families keeping the order the sample lists them in.

    The singleton ask's unit is a record; the batched ask's unit is a cell, because the saving comes
    from asking several families over one state. Regrouping here (rather than seeking the partition
    again) keeps the two asks on the same selected records and the same order.
    """
    cells: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for group, family, record in tasks:
        cells.setdefault(group, []).append((family, record))
    return list(cells.items())


def family_slice(prediction: Mapping[str, Any], family: str, questions: Mapping[str, Any],
                 label: str = "") -> dict[str, Any]:
    """One family's part of a batched answer, in the shape :meth:`JevClient.__call__` returns.

    The batched body's question ids carry a ``<family>/`` prefix; stripping it here is what lets the
    batched ask reuse :func:`rows_of_answer` unchanged, so a batched row and a singleton row are the
    same row built by the same code. ``latency_ms`` is ``None`` on purpose: the batch's latency is a
    property of the whole request, and reporting it as each family's would be four numbers for one
    measurement.
    """
    prefix = BATCH_SEPARATOR.join(part for part in (family, label) if part) + BATCH_SEPARATOR
    return {"probabilities": {qid: prediction["probabilities"][f"{prefix}{qid}"] for qid in questions},
            "answers": {qid: prediction["answers"][f"{prefix}{qid}"] for qid in questions},
            "served_model": prediction.get("served_model"), "latency_ms": None,
            "input_tokens": prediction.get("input_tokens"), "output_tokens": prediction.get("output_tokens")}


def variant_slices(prediction: Mapping[str, Any], family: str,
                   questions: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """The abstain asks carried inside a batched answer, in :func:`variant_map`'s shape."""
    prefix = f"{family}{BATCH_SEPARATOR}abstain{BATCH_SEPARATOR}"
    return {qid: {"criteria": question["criteria"],
                  "probabilities": dict(prediction["probabilities"][f"{prefix}{qid}"])}
            for qid, question in questions.items()}


def build_batched_cell(entries: Sequence[tuple[str, dict[str, Any]]], split: str, client: JevClient,
                       directory: Path, dataset: str, cell: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Serve one cell's families in as few requests as their states allow; save every raw body.

    The families are grouped by **byte-identical state** (:func:`state_groups`), which is the only
    thing that decides how many requests a cell costs — one per group, not one per family and not
    one per cell. Each body is saved with its request, its state digest and the families it carried,
    so the number of calls is read off the artifacts rather than asserted by the runner, and a
    re-run re-serves from disk for free.
    """
    records = dict(entries)
    directory.mkdir(parents=True, exist_ok=True)
    variants = {family: request["questions"] for family, record in records.items()
                if (request := abstain_request(record, client.model)) is not None}
    groups = state_groups(list(records), records)
    served, predictions = [], {}
    for index, families in enumerate(groups):
        path = batch_path(directory, dataset, cell, index)
        if path.exists():
            saved = read_json(path)
        else:
            request = batched_request(records, families, client.model, variants or None)
            prediction = client.respond(request)
            saved = {"families": families, "state_sha256": state_digest(records[families[0]]),
                     "request": request, "request_sha256": request_digest(request),
                     "response": {"model": prediction["served_model"], "answers": prediction["answers"],
                                  "usage": {"input_tokens": prediction["input_tokens"],
                                            "output_tokens": prediction["output_tokens"]}},
                     "prediction": prediction}
            write_json(path, saved)
        served.append({"families": saved["families"], "state_sha256": saved["state_sha256"],
                       "questions": len(saved["request"]["questions"]),
                       "state_chars": len(saved["request"]["state"]),
                       "input_tokens": saved["prediction"].get("input_tokens")})
        for family in saved["families"]:
            predictions[family] = saved["prediction"]

    rows: list[dict[str, Any]] = []
    for family, _record in entries:
        # The caller's order, not the request grouping's: the batched rows then come out in the same
        # order the singleton ask writes them, so two runs over the same cells are comparable line by
        # line. The order the requests were sent in is in ``served``.
        prediction = predictions[family]
        questions = records[family]["questions"]
        rows.extend(rows_of_answer(records[family], family, split,
                                   family_slice(prediction, family, questions),
                                   variant_slices(prediction, family, variants[family])
                                   if family in variants else None))
    return rows, {"requests": len(groups), "served": served}


def cmd_ask_batched(args: argparse.Namespace) -> int:
    """Serve the sample with one request per (cell, state group) — the cost lever, made visible.

    The corpus's four families sit on **two** states per cell (the trio
    ``panel_choice``/``panel_miss``/``marker_noul`` shares one; ``evidence_diluted`` is a diluted
    panel and has its own), so a cell costs two requests where the singleton ask spends four. The
    grouping is computed per cell and written into the ask log, and every raw body records the state
    digest it was served over — a batched run whose saving came from serving the wrong state would
    be visible in these files rather than only in the numbers.
    """
    out = Path(args.out)
    sample = read_json(out / "sample.json")
    corpus = corpus_dir(args.corpus)
    partition = corpus / f"{sample['split']}.jsonl"
    if file_digest(partition) != sample["corpus_sha256"]:
        raise ProbeError(f"{partition} changed since the sample was drawn; the sample would name different records")
    raw = out / (args.raw_dir or "raw-batched")
    raw.mkdir(parents=True, exist_ok=True)
    client = JevClient(base_url=args.base_url, model=args.model, budget_usd=args.budget_usd,
                       max_calls=args.max_calls, max_call_usd=args.max_call_usd)
    cells = cells_of(collect_records(partition, sample))
    if args.limit:
        cells = cells[: args.limit]
    print(f"{len(cells)} cells to serve from {partition.name}; {args.workers} in flight; "
          f"budget ${args.budget_usd}", flush=True)

    def work(index: int, task: tuple[str, list[tuple[str, dict[str, Any]]]]):
        group, entries = task
        directory, cell = dataset_of(group), group.split("/", 1)[1]
        try:
            rows, meta = build_batched_cell(entries, sample["split"], client, raw, directory, cell)
            return index, rows, meta, None
        except BudgetExceeded:
            raise       # a spend cap is not a bad record: it stops the run, it does not reject a row
        except (JevError, ProbeError, KeyError, ValueError) as error:
            return index, [], None, {"id": f"{sample['split']}/{directory}/{cell}", "family": "batch",
                                     "error": f"{type(error).__name__}: {error}"}

    results: list[list[dict[str, Any]]] = [[] for _ in cells]
    metas: list[dict[str, Any] | None] = [None for _ in cells]
    rejected: list[dict[str, Any]] = []
    stopped = None
    try:
        if args.workers > 1:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {pool.submit(work, index, task): index for index, task in enumerate(cells)}
                for done, future in enumerate(as_completed(futures), start=1):
                    index, rows, meta, failure = future.result()
                    results[index], metas[index] = rows, meta
                    if failure:
                        rejected.append(failure)
                    if done % args.progress_every == 0 or done == len(cells):
                        print(f"  [{done:4d}/{len(cells)}] calls={client.calls:4d} "
                              f"in_tokens={client.input_tokens:9d} ${client.estimated_usd:.4f}", flush=True)
        else:
            for index, task in enumerate(cells, start=1):
                _, rows, meta, failure = work(index - 1, task)
                results[index - 1], metas[index - 1] = rows, meta
                if failure:
                    rejected.append(failure)
                if index % args.progress_every == 0 or index == len(cells):
                    print(f"  [{index:4d}/{len(cells)}] calls={client.calls:4d} "
                          f"in_tokens={client.input_tokens:9d} ${client.estimated_usd:.4f}", flush=True)
    except (JevError, ProbeError) as error:
        stopped = f"{type(error).__name__}: {error}"
        print(f"stopping: {stopped}", file=sys.stderr)

    rows = [row for group in results for row in group]
    served = [meta for meta in metas if meta]
    per_cell = Counter(meta["requests"] for meta in served)
    planned_requests = sum(meta["requests"] for meta in served)
    write_rows(out / "rows.jsonl", rows)
    write_ask_log(out, {
        "step": "ask", "batched": True, "records_planned": len(cells), "records_served": len(served),
        "rows": len(rows), "rejected": rejected, "stopped": stopped, "workers": args.workers,
        "accounting": client.accounting(),
        "requests_planned": planned_requests,
        "requests_sent": client.calls,
        "requests_served_from_disk": planned_requests - client.calls,
        "requests_per_cell": dict(sorted(per_cell.items())),
        "state_groups_per_cell": [{"cell": group, "requests": meta["requests"],
                                   "families": [served_["families"] for served_ in meta["served"]],
                                   "state_sha256": [served_["state_sha256"][:12] for served_ in meta["served"]],
                                   "state_chars": [served_["state_chars"] for served_ in meta["served"]],
                                   "questions": [served_["questions"] for served_ in meta["served"]]}
                                  for (group, _entries), meta in zip(cells, metas) if meta],
        "model_requested": args.model,
    })
    print(f"\n{len(rows)} rows from {client.calls} calls, {client.input_tokens} input tokens, "
          f"${client.estimated_usd:.4f}; {len(rejected)} rejected; requests per cell {dict(sorted(per_cell.items()))}")
    print(f"wrote {out / 'rows.jsonl'} and {out / 'ask-log.json'}")
    return 0 if rows else 1


def write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """One JSON object per row, in the order the rows were produced."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps({name: row.get(name) for name in ROW_FIELDS} | {"answer": row.get("answer"),
                                                                        "confidence": row.get("confidence"),
                                                                        "raw_probability_sum": row.get("raw_probability_sum"),
                                                                        "served_model": row.get("served_model")})
             for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"path": str(path), "rows": len(lines), "sha256": file_digest(path)}


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# --------------------------------------------------------------------------- evaluating


def asked_population(sample: Mapping[str, Any]) -> str:
    """How to name what the probe asked, for a caveat: a sample and a census are different claims.

    A sample that carries its own ``population_label`` says how to name it — a population assembled
    from an id list is not the all-four-families eligible set, and naming it as that would be a false
    statement about what was asked.
    """
    if sample.get("population_label"):
        return str(sample["population_label"])
    if sample.get("eligible_cells") and sample["cells"] >= sample["eligible_cells"]:
        return (f"every eligible cell of this partition ({sample['cells']}, the census of the cells that "
                "carry all four families)")
    return f"this {sample['cells']}-cell sample"


def ours_block(path: Path, population: str) -> dict[str, Any] | None:
    """The local model's published marginals, for the comparison, with the caveat attached.

    Two different populations are being put side by side: Jev's numbers are the cells this probe
    asked, the local model's are the whole 15,626-cell partition scored once after model selection.
    A reader has to be told which is which, so the caveat is part of the block rather than a sentence
    in the report that could be dropped — and it names the probe's population, which the caller
    passes in, because "200 cells" and "every eligible cell" are different claims.
    """
    if not path.exists():
        return None
    report = read_json(path)
    axis1 = report.get("axis1_panel_adequacy") or {}
    return {
        "source": str(path), "checkpoint": (report.get("checkpoint") or {}).get("base"),
        "population": "the whole real-v2 test partition, scored once after model selection",
        "accuracy": (report.get("accuracy") or {}).get("accuracy"),
        "accuracy_n": (report.get("accuracy") or {}).get("n"),
        "ece": ((report.get("temperature") or {}).get("before") or {}).get("ece"),
        "axis1_auroc": axis1.get("auroc"), "axis1_n": axis1.get("n"),
        "axis1_selection": "pooled over every choice row of the partition (mixed families and conventions)",
        "caveat": (f"not a paired comparison: Jev's numbers are {population}, these are the whole "
                   "partition; the two are different populations and the axis-1 selections differ"),
    }


def report_block(rows: Sequence[dict[str, Any]], sample: Mapping[str, Any], accounting: Mapping[str, Any],
                 fpr: float, corpus_manifest: Mapping[str, Any] | None = None,
                 ask_log: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Every number the probe produced, and the population each was taken on."""
    from ..eval import metrics, rows_of
    from ..eval.run import plumbing_check

    choice = rows_of(rows, type="choice")
    if not choice:
        raise ProbeError("no choice rows were scored; there is nothing to evaluate")
    adequacy = metrics.panel_adequacy(choice, fpr)
    check = plumbing_check(corpus_manifest or {}, _no_sets(),
                           {sample["split"]: [dict(row) for row in choice]}, fpr, adequacy)
    extra = _probe_reasons(sample, corpus_manifest or {})
    if extra:
        check = {**check, "reasons": check["reasons"] + extra, "is_result": False}
    batched = bool((ask_log or {}).get("batched"))
    asked = ("one request per (cell, state group): every family whose state agrees was asked in one "
             "request" if batched else "one request per ask")
    return {
        "probe": "sckevtype.jev", "step": "evaluate",
        "population": {"split": sample["split"], "cells": sample["cells"], "families": sample["families"],
                       "records": len({(row["group"], row["family"]) for row in rows}),
                       "questions": len(rows), "choice_rows": len(choice),
                       "cells_per_dataset": sample["by_dataset"], "seed": sample["seed"],
                       "eligible_cells": sample["eligible_cells"],
                       "requests_per_cell": (ask_log or {}).get("requests_per_cell"),
                       "how_it_was_asked": asked,
                       "not_a_benchmark": (f"{sample['cells']} cells, {asked}, no selection "
                                           f"discipline beyond the eligibility filter; the population is "
                                           f"recorded so the numbers cannot be read as a benchmark")},
        "accuracy": metrics.top1_accuracy(choice),
        "accuracy_by_family": metrics.breakdown(choice, "family"),
        "accuracy_by_dataset": metrics.breakdown(choice, "dataset"),
        "ece": metrics.ece(choice),
        "risk_coverage": metrics.risk_coverage(choice),
        "axis1_panel_adequacy": adequacy,
        "axis1_by_size": metrics.breakdown(rows_of(choice, family="evidence_diluted"), "size"),
        "noul_rows": len(rows_of(rows, type="noul")),
        "noul_accuracy": (metrics.top1_accuracy(rows_of(rows, type="noul"))
                          if rows_of(rows, type="noul") else None),
        "raw_probability_sums": summarize_sums(rows),
        "provider": provider_block(accounting, ask_log),
        "comparison_to_ours": ours_block(root() / OURS, asked_population(sample)),
        "plumbing_check": check,
        "metric_policy": ("the same sckevtype.eval functions the local model is scored with; there is no "
                          "temperature to fit (a fit needs a calibration partition) and the reported "
                          "probabilities are the model's own, renormalised only to the extent the API's "
                          "0.01 grid requires"),
    }


def _probe_reasons(sample: Mapping[str, Any], corpus_manifest: Mapping[str, Any]) -> list[str]:
    """Why these numbers are a feasibility reading and not a benchmark — computed, not declared.

    The sample's size is set against the partition it was drawn from, which is read from the corpus
    manifest rather than from the sample (a sample that reports its own denominator could be wrong
    about it and nothing would notice).
    """
    reasons = []
    stated = sample.get("population_reason")
    if stated:
        # The sample knows what it is; a population assembled from an id list cannot be described by the
        # all-four-families sentence below, and describing it that way would be a false claim about the ask.
        reasons.append(str(stated))
    partition_cells = (((corpus_manifest.get("splits") or {}).get("splits") or {})
                       .get(sample["split"], {})).get("cells")
    if partition_cells:
        if stated:
            reasons.append(f"{sample['cells']} of the partition's {partition_cells} cells were asked "
                           f"({sample['cells'] / partition_cells:.3%}); the sentence above names that set")
        elif sample.get("eligible_cells") and sample["cells"] >= sample["eligible_cells"]:
            reasons.append(f"{sample['cells']} of the partition's {partition_cells} cells were asked "
                           f"({sample['cells'] / partition_cells:.3%}) — that is every cell carrying all "
                           f"four families, so this is a census of the eligible population, not a sample of "
                           f"the partition; the partition itself is not covered")
        else:
            reasons.append(f"{sample['cells']} of the partition's {partition_cells} cells were drawn "
                           f"({sample['cells'] / partition_cells:.3%}), from the {sample['eligible_cells']} that "
                           f"carry all four families: a sample statistic, not a partition result")
    else:
        reasons.append(f"{sample['cells']} cells were drawn; the corpus manifest states no cell count for "
                       f"{sample['split']!r}, so the sampling fraction is unstated")
    reasons.append("no threshold was fitted: with no calibration partition in this probe the reported "
                   "probabilities are the model's own, and nothing here is a coverage guarantee")
    return reasons


def _no_sets() -> Any:
    """``plumbing_check`` reads ``sets.quantiles``; this probe fits no conformal sets."""
    class _NoSets:
        quantiles: ClassVar[dict[str, Any]] = {}

    return _NoSets()


def _sum_accounting(blocks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Sum several runs' accounting blocks into one, for the directories that were asked twice."""

    def total(key: str) -> float:
        return sum(float(block.get(key) or 0) for block in blocks)

    return {"runs": len(blocks), "calls": int(total("calls")), "input_tokens": int(total("input_tokens")),
            "output_tokens": int(total("output_tokens")),
            "estimated_usd": round(total("estimated_usd"), 6),
            "median_latency_ms": blocks[-1].get("median_latency_ms") if blocks else None}


def write_ask_log(out: Path, payload: Mapping[str, Any]) -> None:
    """Write ``ask-log.json`` without erasing what earlier runs on this directory spent.

    Both ask paths are resumable: a re-run serves the bodies already on disk and spends nothing. Writing
    the accounting straight over the previous file then replaces a real figure with zero — which is how a
    $0.12 probe came to be recorded as $0.00 for anyone reading the artifacts afterwards. So the current
    run's block stays under ``accounting``, earlier runs are kept under ``superseded_runs``, and
    ``accounting_total`` sums the lot. Nothing is rewritten after the fact: a run that spent nothing says
    it spent nothing, and the history is beside it.
    """
    path = out / "ask-log.json"
    log = dict(payload)
    if path.exists():
        previous = read_json(path)
        history = list(previous.get("superseded_runs") or [])
        if previous.get("accounting"):
            history.append(previous["accounting"])
        if history:
            log["superseded_runs"] = history
            log["accounting_total"] = _sum_accounting([*history, log["accounting"]])
    write_json(path, log)


def provider_block(accounting: Mapping[str, Any], ask_log: Mapping[str, Any] | None) -> dict[str, Any]:
    """The provider block for the report: this run's spend, plus the directory's total when they differ.

    A report whose provider block reads ``$0.00 / 0 calls`` because the last run was served from disk is
    worse than no block at all: it reads as "this was free". When the directory has a history the total is
    stated beside the current run.
    """
    block = dict(accounting)
    total = (ask_log or {}).get("accounting_total")
    if total and total.get("calls", 0) > (accounting.get("calls") or 0):
        block["cumulative_over_runs"] = total
        block["note"] = (f"this run served from disk and spent ${float(accounting.get('estimated_usd') or 0):.4f}; "
                         f"the runs on this directory together spent ${total['estimated_usd']:.4f} over "
                         f"{total['calls']} calls (ask-log.json: superseded_runs)")
    return block


def summarize_sums(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """How far the returned distributions sat from 1 — evidence about the 0.01 grid, not a defect list."""
    sums = [float(row["raw_probability_sum"]) for row in rows if row.get("raw_probability_sum") is not None]
    if not sums:
        return {"n": 0}
    return {"n": len(sums), "min": min(sums), "max": max(sums),
            "off_grid_share": sum(1 for value in sums if abs(value - 1.0) > 1e-12) / len(sums),
            "why": "the vendor returns probabilities on a 0.01 grid, so they need not sum to exactly 1"}


def cmd_evaluate(args: argparse.Namespace) -> int:
    from ..eval.run import write_row_dump

    out = Path(args.out)
    sample = read_json(out / "sample.json")
    ask_log = read_json(out / "ask-log.json") if (out / "ask-log.json").exists() else {}
    accounting = ask_log.get("accounting") or {}
    rows = read_rows(out / "rows.jsonl")
    corpus_manifest = Path(sample["corpus"]).parent / "manifest.json"
    result = report_block(rows, sample, accounting, args.fpr,
                          read_json(corpus_manifest) if corpus_manifest.exists() else {}, ask_log)
    dump = write_row_dump(out / "axis1-rows.jsonl", [row for row in rows if row["type"] == "choice"])
    result["rows"] = {"rows": len(rows), "axis1_dump": dump,
                      "dump_reader": "scripts/axis1-recompute.py reads this file unchanged "
                                     "(sckevtype.eval.run.DUMP_FIELDS + positive)"}
    write_json(out / "report.json", result)
    print("\n".join(lines(result)))
    print(f"wrote {out / 'report.json'} and {out / 'axis1-rows.jsonl'}")
    return 0


def lines(result: Mapping[str, Any]) -> list[str]:
    """The report, ordered so the population is read before any number taken on it."""
    population, adequacy = result["population"], result["axis1_panel_adequacy"]
    lines = [
        f"sckevtype.jev.probe — Jev on {population['cells']} cells of real-v2/{population['split']}",
        (f"  {population['questions']} questions over {population['records']} records; "
         f"{population['choice_rows']} choice rows; cells {population['cells_per_dataset']}"),
        (f"  model {result['provider'].get('served_models')}; {result['provider'].get('calls')} calls, "
         f"{result['provider'].get('input_tokens')} input tokens, ${result['provider'].get('estimated_usd', 0):.4f}"),
        (f"  accuracy {result['accuracy']['accuracy']:.4f} (n={result['accuracy']['n']}); "
         f"ece {result['ece']['ece']:.4f} at mean confidence {result['ece']['mean_confidence']:.4f}"),
        (f"  risk-coverage: coverage at 5% error {result['risk_coverage']['coverage_at_5pct_error']:.4f}, "
         f"aurc {result['risk_coverage']['aurc']:.4f}"),
    ]
    for family, entry in sorted(result["accuracy_by_family"].items()):
        lines.append(f"    family {family:<20} n={entry['n']:<5d} accuracy {entry['accuracy']:.4f} "
                     f"ece {entry['ece']:.4f}")
    primary = adequacy.get("primary")
    if primary is None:
        lines.append(f"  axis 1 PRIMARY n/a — {adequacy['primary_note']}")
    else:
        lines.append(f"  axis 1 PRIMARY auroc {primary['auroc']:.4f} [{primary['selection']}; "
                     f"n={primary['n']} pos={primary['n_positive']} neg={primary['n_negative']}]; "
                     f"at FPR {primary['target_fpr']:.2f} detects {primary['detection_rate']:.4f}")
    for name in ("factory", "injected"):
        arm = adequacy["by_serving_convention"][name]
        auroc = "n/a" if arm.get("auroc") is None else f"{arm['auroc']:.4f}"
        lines.append(f"    arm {name:<10} n={arm['n']:<5d} pos={arm['n_positive']:<5d} "
                     f"negs={arm['n_negative']:<5d} auroc {auroc}  mean P(abstain) "
                     f"{arm.get('mean_score') if arm.get('mean_score') is None else round(arm['mean_score'], 4)}")
    for family, entry in sorted(adequacy["per_family"].items()):
        auroc = "n/a (one axis-1 class only)" if entry.get("auroc") is None else f"{entry['auroc']:.4f}"
        lines.append(f"    family {family:<20} n={entry['n']:<5d} pos={entry['n_positive']:<5d} "
                     f"neg={entry['n_negative']:<5d} auroc {auroc}")
    ours = result.get("comparison_to_ours")
    if ours:
        lines.append(f"  ours (whole partition, not a paired comparison): accuracy {ours['accuracy']:.4f} "
                     f"(n={ours['accuracy_n']}), ece {ours['ece']:.4f}, axis-1 auroc {ours['axis1_auroc']:.4f}")
    check = result["plumbing_check"]
    if not check["is_result"]:
        lines.append("  PLUMBING CHECK — this is a feasibility probe, not a benchmark:")
        lines.extend(f"    - {reason}" for reason in check["reasons"])
    return lines


# --------------------------------------------------------------------------- cli


def add_ask_options(parser: argparse.ArgumentParser) -> None:
    """The flags both ask paths take, defined once so the two CLIs cannot drift apart.

    The money flags are the same on both because the guard is the same client: a batched run and a
    singleton run differ in how many requests they send, not in what a request is allowed to cost.
    """
    parser.add_argument("--out", required=True, help="the directory sample.json was written into")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="override the endpoint")
    parser.add_argument("--budget-usd", type=float, default=1.0)
    parser.add_argument("--max-calls", type=int, default=2000)
    parser.add_argument("--max-call-usd", type=float, default=0.10,
                        help="refuse a single request whose worst case exceeds this")
    parser.add_argument("--limit", type=int, default=0,
                        help="stop after N records (0 = all); on the batched path, after N cells")
    parser.add_argument("--workers", type=int, default=4,
                        help="requests in flight: threads in one process, not processes (1 = sequential)")
    parser.add_argument("--progress-every", type=int, default=25, help="print a progress line every N records")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="sckevtype.jev.probe",
                                     description="Ask TypeSafe's Jev our own corpus, then score it with our metrics.")
    parser.add_argument("--corpus", default=None,
                        help="corpus directory, a TOP-LEVEL flag: it must precede the subcommand "
                             "(default: <root>/runs/real-v2)")
    sub = parser.add_subparsers(dest="command", required=True)

    sample = sub.add_parser("sample", help="choose the cells (no network, no key)")
    sample.add_argument("--out", required=True, help="directory to write sample.json into")
    sample.add_argument("--split", default="test", help="partition to draw from")
    sample.add_argument("--cells-per-dataset", type=int, default=CELLS_PER_DATASET,
                        help="0 = every eligible cell (the full-population run); N = N per dataset")
    sample.add_argument("--seed", type=int, default=SAMPLE_SEED)
    sample.set_defaults(func=cmd_sample)

    ask = sub.add_parser("ask", help="send the requests (network; resumable; budgeted)")
    add_ask_options(ask)
    ask.set_defaults(func=cmd_ask)

    batched = sub.add_parser("ask-batched",
                             help="send one request per (cell, state group) — network; resumable; budgeted")
    add_ask_options(batched)
    batched.add_argument("--raw-dir", default="raw-batched",
                         help="where the batched bodies are saved (beside the singleton ask's)")
    batched.set_defaults(func=cmd_ask_batched)

    evaluate = sub.add_parser("evaluate", help="score the saved answers (local; no key needed)")
    evaluate.add_argument("--out", required=True)
    evaluate.add_argument("--fpr", type=float, default=FPR)
    evaluate.set_defaults(func=cmd_evaluate)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return args.func(args)
    except (ProbeError, JevError, ValueError, KeyError, FileNotFoundError, ImportError) as error:
        print(f"jev probe failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
