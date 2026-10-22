"""The evaluation CLI: a suite and a checkpoint in, one report out.

    uv run python -m sckevtype.eval.run --suite artifacts/suites/smoke \\
        --checkpoint runs/smoke-0.8b --alpha 0.10 --out reports/eval-smoke.json

What it does, in order, and the one thing each step is not allowed to do:

1. **Score both partitions** with the checkpoint, at ``T = 1`` — raw logits, because a
   temperature fitted on probabilities is a temperature fitted on someone else's calibration.
   Each choice question is also asked again with the abstain option offered
   (:func:`with_abstain`), which is what gives axis 1 a statistic on every choice row rather
   than only on the rows the factory happened to give one;
2. **Fit on ``calibration``** — the temperature, the conformal quantiles, the triage threshold —
   through the guarded entry points, so a bug that hands a fit a test row raises instead of
   quietly reporting an in-sample number;
3. **Report on ``test``** — coverage per stratum and per family, set sizes, ECE before and after
   the temperature, the axis-1 detection numbers, the triage policy and its risk–coverage sweep.

The report carries the suite's and the checkpoint's digests, the partitions used by name, and a
:func:`plumbing_check` block. On a synthetic fixture corpus that block fires, and its reasons are
in the file: a number from a 32-cell smoke corpus proves the machinery runs end to end and is
self-consistent, and proves nothing about cell-type annotation.

Every number here is a deterministic function of ``(suite digest, checkpoint, alpha, beta)``.
There is no sampling step to seed: the model runs in fp32 with no dropout, the conformal
quantiles and the thresholds are order statistics, and the abstain wording appended in
:func:`with_abstain` is fixed rather than drawn. The one seed in the report is the checkpoint's
own training seed, read from its provenance.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..contracts import paths
from ..contracts.freeze import file_digest, read_json, write_json
from . import (
    CALIBRATION,
    TEST,
    abstain_index,
    abstain_key,
    conformal,
    metrics,
    rows_of,
    triage,
)

#: The partitions this CLI scores. ``development`` is deliberately absent: it belongs to model
#: selection, and every additional partition read at report time is another way to fit by accident.
PARTITIONS = (CALIBRATION, TEST)

#: The default grids the risk–coverage sweep runs over. Stated here, chosen in advance, and never
#: fitted on the partition they are reported against.
SWEEP_ALPHAS = (0.02, 0.05, 0.10, 0.20)
SWEEP_BETAS = (0.02, 0.05, 0.10, 0.20)


# --------------------------------------------------------------------------- arguments


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="sckevtype.eval.run", description="Two-axis uncertainty evaluation on a suite.")
    parser.add_argument("--suite", required=True, help="suite directory written by sckevtype.train.suite")
    parser.add_argument("--checkpoint", required=True, help="run directory holding the LoRA adapter and head.pt")
    parser.add_argument("--alpha", type=float, required=True, help="conformal level: the per-stratum miscoverage budget")
    parser.add_argument("--beta", type=float, default=0.05, help="axis-1 risk-control budget for the triage gate")
    parser.add_argument("--method", choices=conformal.METHODS, default="lac", help="nonconformity score for the reported policy")
    parser.add_argument("--fpr", type=float, default=0.05, help="target false-positive rate for the axis-1 detection rate")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default=None)
    parser.add_argument("--out", required=True, help="report path (.json)")
    parser.add_argument("--rows-out", default=None,
                        help="also write the per-row scores axis 1 is computed from (.jsonl), so the "
                             "statistic can be re-derived without a second GPU pass")
    parser.add_argument("--rows-full", default=None,
                        help="also write one JSON line per scored choice row (.jsonl) with the option "
                             "list as served and the option the model named, so a confusion analysis "
                             "can say which type it chose; noul (yes/no marker) rows are out of scope "
                             "and are not written")
    args = parser.parse_args(argv)
    if not 0.0 < args.alpha < 1.0:
        parser.error("--alpha must be in (0, 1)")
    if not 0.0 < args.beta < 1.0:
        parser.error("--beta must be in (0, 1)")
    return args


# --------------------------------------------------------------------------- scoring


def served_none_option() -> tuple[str, str | None]:
    """The abstain option this CLI serves on a row the factory gave none: ``(key, wording)``.

    It is the factory's **own** first entry — same table, same rendering — appended to the
    question's criteria exactly as the factory appends its own. What it is *not* is the factory's
    whole convention: the factory draws one of the thirteen entries per cell and shuffles it into
    the option order, so a factory-given abstain option is last in only 52.6% of the rows this
    corpus serves and carries no wording at all in 15.4%. That difference is measured and recorded
    per row rather than papered over — ``abstain_source`` says which convention served a row, and
    the axis-1 statistic is taken on the factory-given rows only. See
    ``docs/axis1-metric-runbook.md`` for the measurement and for why reproducing the draw was
    refused rather than approximated.
    """
    from ..data.families import NONE_OPTIONS

    return NONE_OPTIONS[0]


def with_abstain(question: Mapping[str, Any]) -> dict[str, Any] | None:
    """The question with an abstain option offered, or ``None`` when it already offers one.

    The factory decides per record whether a choice question carries an abstain option, and on the
    rows where it did not the question *as written* has no positivity to detect — the answer is a
    real option by construction. Axis 1 needs the same statistic on those rows too, so they are
    re-asked with one option added, and only that option: the rest of the question is the record's
    own. The convention is fixed rather than drawn so the report stays a deterministic function of
    ``(suite digest, checkpoint, alpha, beta)`` (see :func:`served_none_option`).
    """
    from kev.api import question_keys

    if question["type"] != "choice":
        return None
    if abstain_key(question_keys(question["type"], question.get("criteria"))) is not None:
        return None
    key, wording = served_none_option()
    return {**question, "criteria": {**question["criteria"], key: wording}}


def served_row_fields(row: Mapping[str, Any], source: str, keys: Sequence[str],
                      criteria: Mapping[str, Any]) -> dict[str, Any]:
    """How the abstain option reached this row's prompt, recorded per row rather than inferred.

    ``abstain_source`` is :data:`~sckevtype.eval.metrics.FACTORY` or
    :data:`~sckevtype.eval.metrics.INJECTED`; ``abstain_key``/``abstain_text`` are the option
    itself; ``abstain_position``/``abstain_last`` are where it sat in the prompt *as served*, which
    is the one thing the two conventions do not share and the reason they are reported apart. A
    statistic recomputed from a row dump needs the first three; the last two are what makes the
    convention gap visible in the dump instead of only in the report.
    """
    key = abstain_key(keys)
    return {
        "abstain_source": source,
        "abstain_key": key,
        "abstain_text": criteria.get(key) if key is not None else None,
        "abstain_position": list(keys).index(key) if key is not None else None,
        "abstain_last": key is not None and list(keys).index(key) == len(keys) - 1,
    }


def extended_row(row: dict[str, Any], record: Mapping[str, Any], split: str) -> dict[str, Any]:
    """kev's prediction row plus the metadata Step 4 stratifies and reports on.

    kev's row already carries what :mod:`kev.metrics` scores; what it does not carry is the
    partition, the record family, the panel and its size — the three keys the two axes are
    stratified by. They are copied from ``_meta`` rather than recomputed, so a row cannot disagree
    with the record it came from.
    """
    meta = record["_meta"]
    row.update(
        split=split,
        family=str(meta.get("family", "unknown")),
        panel=str(meta.get("panel_id", "unknown")),
        dataset=str(meta.get("dataset_id", "unknown")),
        cell=str(meta.get("cell_id", "unknown")),
        donor=str(meta.get("donor_id", "unknown")),
        size=len(row["keys"]),
        abstain=abstain_index(row),
        none_p=None,
        **served_row_fields(row, metrics.FACTORY, row["keys"], record["questions"][row["question"]].get("criteria") or {}),
    )
    return row


def score_partition(records: Sequence[Mapping[str, Any]], predictor: Any, split: str) -> tuple[list[dict], list[dict]]:
    """Rows for one partition, plus the records the encoder rejected. Never raises on a long record.

    A record that does not fit the serving context is counted and named rather than aborting the
    run: silently dropping it would make the coverage denominators wrong, and aborting would make
    the tool useless on exactly the records worth knowing about. The count goes in the report.
    """
    from kev.benchmark import prediction_rows
    from kev.model import ContextOverflow

    rows: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for record in records:
        try:
            prediction = predictor(record)
            new_rows = [extended_row(row, record, split) for row in prediction_rows(record, prediction)]
        except (ContextOverflow, ValueError) as error:
            rejected.append({"id": record["_meta"]["id"], "split": split, "error": f"{type(error).__name__}: {error}"})
            continue
        for row in new_rows:
            key = abstain_key(row["keys"])
            row["none_p"] = None if key is None else float(prediction["probabilities"][row["question"]][key])
        score_abstain_variant(record, predictor, new_rows)
        rows.extend(new_rows)
    return rows, rejected


def score_abstain_variant(record: Mapping[str, Any], predictor: Any, rows: Sequence[dict]) -> None:
    """Re-ask the choice questions that lack an abstain option and fill in ``none_p`` on their rows.

    Filled in on the *same* row objects rather than emitted as new rows: the abstain variant is
    not a second observation of the label, it is the axis-1 statistic of the row that already
    exists, and composing the two axes requires them to be keyed to one row. A record whose
    variant does not encode is marked rather than dropped — an axis-1 number computed on whichever
    rows happened to fit would be a silent sample.
    """
    from kev.model import ContextOverflow

    variants = {}
    for question_id, question in record["questions"].items():
        variant = with_abstain(question)
        if variant is not None:
            variants[question_id] = variant
    affected = [row for row in rows if row["question"] in variants]
    if not affected:
        return
    try:
        prediction = predictor({**record, "questions": variants})
    except (ContextOverflow, ValueError) as error:
        for row in affected:
            row["none_p_unavailable"] = f"{type(error).__name__}: {error}"
        return
    for row in affected:
        probabilities = prediction["probabilities"][row["question"]]
        key = abstain_key(list(probabilities))
        row["abstain"] = None
        row["none_p"] = float(probabilities[key])
        row.update(served_row_fields(row, metrics.INJECTED, list(probabilities),
                                     variants[row["question"]]["criteria"]))


# --------------------------------------------------------------------------- the row dump

#: The two question types a row dump exists for, named once. ``choice`` is the type whose *named
#: option* an error analysis reads (which type was picked); ``noul`` is the reference's binary marker
#: question — ``keys == ["false","true"]``, one row per question, no abstain option — whose accuracy
#: is a column of the census table in its own right. Both are written with :data:`FULL_DUMP_FIELDS`;
#: ``dump_lines`` is the single place that turns rows of either type into lines.
CHOICE, NOUL = "choice", "noul"

#: The fields a dumped axis-1 row carries. Everything
#: :func:`sckevtype.eval.metrics.panel_adequacy` reads is here — ``label``, the abstain index the
#: scoring path recorded, ``none_p``, ``family`` and ``abstain_source`` — plus the four that make
#: the serving convention checkable from the dump rather than only from the report.
DUMP_FIELDS = ("id", "question", "split", "dataset", "donor", "family", "variant", "size", "label",
               "abstain", "abstain_source", "abstain_key", "abstain_text", "abstain_position",
               "abstain_last", "none_p")


def write_row_dump(path: Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Write the per-row scores axis 1 is computed from, one JSON object per line.

    ``positive`` is written beside ``label`` and ``abstain`` so the dump reads on its own; it is the
    metric's own label rule applied here, and ``scripts/axis1-recompute.py`` checks the two agree
    rather than trusting either. A choice row whose abstain ask produced no statistic is written
    too, with ``none_p: null`` — the dump is the population the statistic was taken from *and* the
    rows it had to drop, so a reader sees the sample instead of inferring it.
    """
    lines = [json.dumps({**{name: row.get(name) for name in DUMP_FIELDS},
                         "positive": int(metrics.axis1_label(row))}) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"path": str(path), "rows": len(lines), "sha256": file_digest(path)}


#: The fields a full-dump row carries: the option list as served and the option the model named, so
#: an error can be told as *which type* was chosen, plus the axis-1 fields the same rows carry, plus
#: ``p`` — the option-probability vector ``top1``/``top1_p`` are read off.
#:
#: ``p`` is retained because ``top1`` alone cannot re-apply a policy that removes the abstain option
#: ("forced naming"): on a row where the model named the abstain option, the option it would have
#: named without one is the argmax over ``p`` with the abstain index dropped, and that runner-up is
#: in neither ``top1``/``top1_p`` nor ``none_p``. The vector is the row's own ``p`` — raw, scored at
#: T = 1 — so the rule can be re-applied later without a second GPU pass. Every field that was here
#: before is unchanged; ``p`` is appended.
FULL_DUMP_FIELDS = ("id", "question", "split", "dataset", "donor", "family", "variant", "size",
                    "keys", "label", "top1", "top1_p", "abstain", "abstain_source", "abstain_key",
                    "none_p", "p")


def dump_lines(rows: Sequence[Mapping[str, Any]], question_type: str) -> tuple[list[str], int]:
    """The dump's JSON lines for the rows of one question type, plus how many rows were left out.

    One place turns rows into dump lines, so the choice dump and the noul dump cannot drift apart in
    field list, argmax rule or serialisation: which rows go in is the *only* difference between them.
    ``p`` is the vector ``top1`` is the argmax of, written as plain floats (aligned with ``keys``), and
    a row whose ``p`` is missing **raises** rather than being dropped — a silently missing prediction
    is how a partial population starts looking like a full one.

    A selection matching **no** row is refused by name, not written as an empty file: an empty dump
    reads exactly like a population that was never scored, and that ambiguity has already cost this
    project a queue slot (``marker_noul``: 4,035 records carrying 63,851 ``noul`` questions and zero
    ``choice`` questions, scored through the choice writer, which wrote nothing and said so only in a
    ``skipped`` count nobody read).
    """
    lines, skipped = [], 0
    for row in rows:
        if row.get("type") != question_type:
            skipped += 1
            continue
        probabilities = row.get("p")
        if probabilities is None or len(probabilities) == 0:
            raise ValueError(f"row {row.get('id')!r} carries no 'p'; refusing to write a dump "
                             f"that would silently drop its prediction")
        top1 = max(range(len(probabilities)), key=lambda index: probabilities[index])
        line = {name: row.get(name) for name in FULL_DUMP_FIELDS}
        line["top1"], line["top1_p"] = top1, float(probabilities[top1])
        line["p"] = [float(value) for value in probabilities]
        lines.append(json.dumps(line))
    if not lines:
        raise ValueError(f"no {question_type!r} rows in a selection of {len(rows)}: refusing to write an "
                         f"empty dump, which cannot be told from a population that was never scored")
    return lines, skipped


def write_full_row_dump(path: Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Write the choice rows with the option the model named, one JSON object per line.

    ``keys`` is the option list as served and ``label`` the index of the correct option, both
    already on the row; ``top1``/``top1_p`` are read off the row's own ``p`` — the raw probabilities
    the eval scores at T = 1 by design — with ``top1`` the lowest index among equal maxima, so a tie
    resolves the same way on every run. ``noul`` rows (yes/no marker questions, whose answer is not
    a type) are out of scope, are not written, and are counted as ``skipped``. The line shape and
    every refusal are :func:`dump_lines`'s.

    ``p`` is written beside ``top1`` for the same reason and from the same object: it is the vector
    ``top1`` is the argmax of, aligned with ``keys``, so a rule that drops an option (the abstain
    option, for forced naming) can be applied to the row without the model. It is written as plain
    floats rather than the row's own container so the line is JSON by construction. On a row whose
    abstain option was injected by the CLI (``abstain_source`` = ``injected``) ``keys`` is the
    question *as served*, which offers no abstain option, and ``p`` is over that list — there is no
    option in it to drop, and the row's ``abstain`` is null.
    """
    lines, skipped = dump_lines(rows, CHOICE)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"path": str(path), "rows": len(lines), "sha256": file_digest(path), "skipped": skipped}


def write_noul_row_dump(path: Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Write the ``noul`` (yes/no marker) rows, one JSON object per line, in the full dump's own shape.

    A ``noul`` answer is a yes/no, and the reference fixes its keys to ``["false","true"]`` in that
    order (``kev.api.question_keys``), so the row's own ``label`` is ``0`` for *no* and ``1`` for
    *yes*, ``top1 == 1`` is the model answering *yes*, and ``p[1]`` is P(yes): the binary accuracy a
    reader wants is ``argmax(p) == label``, which is :func:`sckevtype.eval.metrics.top1_accuracy`'s
    own rule applied to the same field. The **row shape is unchanged** — ``FULL_DUMP_FIELDS``, the
    same 17 fields every choice rung dump carries — so a consumer reads this dump with the reader it
    already has; the abstain fields are null because a yes/no question offers no abstain option
    (``abstain_key(["false","true"])`` is ``None``).

    **One row here is one question**, which is the unit an accuracy over this family is taken on: a
    ``marker_noul`` record asks 4-41 of them, so ``rows`` is the question count and the acceptance
    check for a noul population compares it against the suite's own question count for those records
    (63,851 over the v4 census's 4,035 records) rather than against the record count.
    """
    lines, skipped = dump_lines(rows, NOUL)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"path": str(path), "rows": len(lines), "sha256": file_digest(path), "skipped": skipped}


def write_dumps(args: argparse.Namespace, rows_by_partition: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    """Write whichever row dumps were asked for and return the report's ``rows`` block.

    Both dumps are written once, at the end of the run. The axis-1 dump covers the test choice rows
    it is the population of; the full dump covers every scored choice row in partition order —
    calibration, then test — and each partition's rows are already in record order and each record's
    in question order. Each entry records its path, row count and digest beside the rows skipped.
    """
    block: dict[str, Any] = {}
    if args.rows_out:
        block["axis1_dump"] = write_row_dump(Path(args.rows_out), rows_of(rows_by_partition[TEST], type="choice"))
    if args.rows_full:
        block["full_dump"] = write_full_row_dump(
            Path(args.rows_full), [row for split in PARTITIONS for row in rows_by_partition[split]])
    return block


# --------------------------------------------------------------------------- the report


def plumbing_check(manifest: Mapping[str, Any], sets: conformal.StratifiedSets,
                   rows_by_partition: Mapping[str, Sequence[dict]], alpha: float,
                   adequacy: Mapping[str, Any]) -> dict[str, Any]:
    """Whether these numbers may be read as a result — and when not, the computed reasons why.

    Every reason is computed rather than declared, so this block travels to the real benchmark
    corpus unchanged. A corpus whose datasets are unregistered fixtures is synthetic. A corpus
    whose calibration stratum holds ``n`` rows cannot resolve a coverage target finer than
    ``1/(n+1)``, so at that stratum a single record moves the empirical coverage by more than the
    nominal gap it is being compared against — the number is arithmetic, not evidence. And an
    axis-1 statistic taken on a selection with one class, or on rows that do not say which serving
    convention produced them, is not a measurement of panel adequacy — the two ways that happens
    are read off the rows themselves in :func:`_axis1_reasons`.
    """
    reasons = _axis1_reasons(adequacy)
    corpus = manifest.get("corpus") or {}
    corpus_path = paths.root() / str(corpus.get("path", ""))
    if corpus.get("path") and (corpus_path / "manifest.json").exists():
        datasets = read_json(corpus_path / "manifest.json").get("datasets") or {}
        unregistered = sorted(name for name, entry in datasets.items() if not entry.get("registered"))
        if unregistered:
            reasons.append(f"corpus datasets {unregistered} are unregistered fixtures, not published atlases")

    calibration = rows_by_partition.get(CALIBRATION, [])
    resolution = 1.0 / (len(calibration) + 1) if calibration else None
    if resolution is None or resolution > alpha:
        reasons.append(
            f"the calibration partition holds {len(calibration)} questions, so one record moves empirical "
            f"coverage by {resolution if resolution is None else round(resolution, 4)} against a nominal gap of {alpha}"
        )
    coarse = [stratum for stratum in sets.quantiles if len(stratum) == 1]
    if coarse:
        reasons.append(f"{len(coarse)} quantile(s) had to be fitted at the marginal (type-only) stratum: "
                       f"no panel-size stratum had the calibration support to carry alpha={alpha}")
    return {"is_result": not reasons, "reasons": reasons,
            "calibration_questions": len(calibration), "nominal_target": 1.0 - alpha,
            "rule": "reasons are computed from the corpus manifest, the calibration support and the axis-1 "
                    "rows, never from the suite's name"}


def _axis1_reasons(adequacy: Mapping[str, Any]) -> list[str]:
    """Why the axis-1 number may not be reported as panel adequacy — read off the rows.

    Two ways the statistic stops measuring what it names. A **single-class selection**: no family
    carrying both classes among the factory-given rows, so there is no AUROC to take at all. And an
    **unstated serving convention**: rows whose prompt was not the factory's and not the CLI's, so
    the number cannot be attributed to either. Both are counts, not judgements about the corpus.
    """
    reasons = []
    if adequacy.get("primary") is None:
        reasons.append(f"axis 1's primary statistic is undefined: {adequacy.get('primary_note')} "
                       f"(of {adequacy.get('n')} choice rows carrying the statistic)")
    unstated = int(adequacy.get("n_unstated_convention") or 0)
    if unstated:
        reasons.append(f"{unstated} of {adequacy.get('n')} axis-1 rows state no serving convention "
                       f"(factory-given or eval-injected), so the statistic cannot be attributed to one")
    if adequacy.get("mixed_family_pool"):
        reasons.append(f"the pooled axis-1 AUROC mixes families carrying one class only: "
                       f"{adequacy.get('pooled_qualifier')}")
    return reasons


def report(args: argparse.Namespace) -> dict[str, Any]:
    """Score, fit on calibration, report on test, and return the report."""
    from kev.checkpoint import LoadOptions
    from kev.device import default_device
    from kev.predictors import LocalPredictor
    from kev.suite import load_split, read_manifest

    manifest = read_manifest(args.suite)
    device = args.device or default_device()
    context = manifest.get("context")
    # T = 1: raw logits. The checkpoint's own shipped temperature would be a fit on data this
    # report never sees, and fitting our own on calibration requires the untempered logits.
    predictor = LocalPredictor(args.checkpoint, device, LoadOptions(temperature=1.0), **({"context": context} if context else {}))

    rows_by_partition: dict[str, list[dict]] = {}
    rejected: list[dict] = []
    for split in PARTITIONS:
        records = load_split(args.suite, split, allow_test=True)
        rows_by_partition[split], refused = score_partition(records, predictor, split)
        rejected.extend(refused)

    calibration, test = rows_by_partition[CALIBRATION], rows_by_partition[TEST]

    temperature = metrics.temperature_scaling(calibration, test)
    conformal_by_method = {
        method: conformal.StratifiedSets.fit(calibration, args.alpha, method) for method in conformal.METHODS
    }
    sets = conformal_by_method[args.method]
    policy = triage.fit(calibration, args.alpha, args.beta, args.method)
    triage_rows = {split: rows_of(rows, type="choice") for split, rows in rows_by_partition.items()}

    adequacy = metrics.panel_adequacy(rows_of(test, type="choice"), args.fpr)
    row_dumps = write_dumps(args, rows_by_partition)
    return {
        "tool": {"name": "sckevtype.eval", "alpha": args.alpha, "beta": args.beta, "method": args.method,
                 "fpr": args.fpr, "device": device, "rows_out": args.rows_out},
        "suite": {"path": args.suite, "name": manifest.get("name"), "manifest_sha256": file_digest(Path(args.suite) / "manifest.json"),
                  "partitions": {name: entry["sha256"] for name, entry in sorted(manifest.get("files", {}).items())},
                  "corpus": manifest.get("corpus"), "base_revisions": manifest.get("base_revisions")},
        "checkpoint": checkpoint_summary(args.checkpoint),
        "partitions": {
            "fitted_on": CALIBRATION,
            "reported_on": TEST,
            "never_read": [name for name in ("train", "development") if name not in PARTITIONS],
            "rows": {split: metrics.chosen_partition(rows) for split, rows in rows_by_partition.items()},
            "rejected_records": rejected,
            "guard": "every fit path calls sckevtype.eval.assert_fit_partition and raises on anything but calibration",
        },
        "rows": row_dumps,
        "seeds": {"checkpoint_training_seed": checkpoint_summary(args.checkpoint).get("training_seed"),
                  "eval_is_deterministic": True,
                  "why": "no sampling step: fp32 scoring, order-statistic quantiles, a fixed abstain wording"},
        "temperature": temperature,
        "axis1_panel_adequacy": adequacy,
        "axis2_conformal": {
            "nominal_coverage": 1.0 - args.alpha,
            "by_method": {method: {"fit": fitted_sets.to_dict(), "test": conformal.coverage(test, fitted_sets.sets(test)),
                                   "set_sizes": metrics.set_sizes(fitted_sets.sets(test)),
                                   "strata_used": fitted_sets.strata_used(test)}
                          for method, fitted_sets in conformal_by_method.items()},
            "comparison": _compare_methods(conformal_by_method, test),
        },
        "triage": triage.summary(policy, triage_rows[TEST]),
        "triage_calibration": triage.summary(policy, triage_rows[CALIBRATION]),
        "sweep": triage.sweep(calibration, triage_rows[TEST], alphas=SWEEP_ALPHAS, betas=SWEEP_BETAS, method=args.method),
        "breakdowns": {key: metrics.breakdown(test, key) for key in ("family", "size", "type", "dataset", "donor")},
        "per_stratum": metrics.per_stratum(test),
        "accuracy": metrics.top1_accuracy(test),
        "risk_coverage": metrics.risk_coverage(test),
        "plumbing_check": plumbing_check(manifest, sets, rows_by_partition, args.alpha, adequacy),
    }


def checkpoint_summary(path: str) -> dict[str, Any]:
    """What was scored: the base revision, the adapter and head digests, and the training seed."""
    from kev.checkpoint import Checkpoint

    checkpoint = Checkpoint(path)
    provenance = Path(checkpoint.path) / "provenance.json"
    recorded = read_json(provenance) if provenance.exists() else {}
    return {
        "path": str(path),
        "base": checkpoint.meta.base,
        "base_revision": checkpoint.meta.base_revision,
        "lora": checkpoint.meta.lora,
        "head_dim": checkpoint.meta.head_dim,
        "shipped_temperature": checkpoint.meta.temperature,
        "adapter_sha256": file_digest(checkpoint.file("adapter_model.safetensors")),
        "head_sha256": file_digest(checkpoint.file("head.pt")),
        "suite_sha256_trained_on": (checkpoint.meta.extra or {}).get("suite_sha256"),
        "training_seed": (recorded.get("hyperparameters") or {}).get("seed"),
        "provenance": str(provenance) if provenance.exists() else None,
    }


def _compare_methods(fitted: Mapping[str, conformal.StratifiedSets], rows: Sequence[dict]) -> dict[str, Any]:
    """Set size at equal coverage: the comparison the paper's axis-2 claim rests on.

    Both methods are fitted at the same ``alpha`` on the same calibration rows, so their nominal
    coverage is identical and the only difference left is what a set costs. The gap between each
    method's empirical coverage and ``1 - alpha`` is reported beside it, because a method that is
    badly undercovered wins a set-size comparison it should not be in.
    """
    table = {}
    for method, sets in fitted.items():
        masks = sets.sets(rows)
        table[method] = {"coverage": conformal.coverage(rows, masks)["coverage"], **metrics.set_sizes(masks)}
    sized = {name: value["mean"] for name, value in table.items() if value["mean"] is not None}
    smaller = min(sized, key=lambda name: sized[name]) if sized else None
    return {"per_method": table, "smaller_mean_set": smaller,
            "note": "nominal coverage is identical by construction; compare sizes only against the empirical coverage column"}


def _arm_line(name: str, arm: Mapping[str, Any]) -> str:
    auroc = "n/a (one axis-1 class only)" if arm["auroc"] is None else f"{arm['auroc']:.4f}"
    mean = "n/a" if not arm["n"] else f"{arm['mean_score']:.4f}"
    return (f"  axis 1 arm {name:<13s} n={arm['n']:<6d} pos={arm['n_positive']:<6d} neg={arm['n_negative']:<6d} "
            f"auroc {auroc:<24s} mean P(abstain) {mean}")


def _axis1_lines(adequacy: Mapping[str, Any]) -> list[str]:
    """The axis-1 block, in the one order that cannot be misread.

    The **primary** first, with the n and both class counts it is a function of; then the two
    serving conventions beside each other, because a reader who sees only one of them cannot tell
    the statistic from a measurement of prompt shape; then every family, one-class families
    included, because their absence is the finding; and the pooled number last and only ever with
    its qualifier attached — it is not panel adequacy and must not be printed as though it were.
    """
    primary = adequacy.get("primary")
    lines = [f"  axis 1 PRIMARY auroc {primary['auroc']:.4f}  [{primary['selection']}; "
             f"families {primary['families']}; n={primary['n']} pos={primary['n_positive']} "
             f"neg={primary['n_negative']}]; at FPR {primary['target_fpr']:.2f} detects "
             f"{primary['detection_rate']:.4f} (achieved FPR {primary['achieved_fpr']:.4f})"
             if primary is not None else f"  axis 1 PRIMARY n/a — {adequacy['primary_note']}"]
    if primary is not None and primary.get("note"):
        lines.append(f"    {primary['note']}")
    conventions = adequacy["by_serving_convention"]
    lines.extend(_arm_line(name, conventions[name]) for name in (metrics.FACTORY, metrics.INJECTED))
    gap = conventions["gap"]
    if gap["auc_convention"] is not None:
        lines.append(f"  axis 1 convention gap: AUROC {gap['auc_convention']:.4f} of none_p telling the two "
                     f"conventions apart ({gap['note']})")
    for family, entry in sorted(adequacy["per_family"].items()):
        auroc = "n/a (one axis-1 class only)" if entry["auroc"] is None else f"{entry['auroc']:.4f}"
        lines.append(f"    family {family:<20s} n={entry['n']:<6d} pos={entry['n_positive']:<6d} "
                     f"neg={entry['n_negative']:<6d} auroc {auroc}")
    if adequacy["auroc"] is None:
        lines.append(f"  axis 1 pooled: n/a (one axis-1 class only) — {adequacy['pooled_qualifier']}")
    else:
        pooled = f"  axis 1 pooled auroc {adequacy['auroc']:.4f}"
        lines.append(f"{pooled} — QUALIFIER: {adequacy['pooled_qualifier']}" if adequacy["mixed_family_pool"]
                     else f"{pooled} ({adequacy['pooled_qualifier']})")
    return lines


def _report_lines(result: Mapping[str, Any]) -> list[str]:
    axis2 = result["axis2_conformal"]
    lines = [
        f"sckevtype.eval — suite {result['suite']['name']}, checkpoint {result['checkpoint']['base']}",
        (f"  fitted on {result['partitions']['fitted_on']}, reported on {result['partitions']['reported_on']}; "
         f"{result['plumbing_check']['calibration_questions']} calibration questions, "
         f"{axis2['by_method'][result['tool']['method']]['test']['n']} test questions"),
        (f"  accuracy {result['accuracy']['accuracy']:.3f}; ece {result['temperature']['before']['ece']:.3f} "
         f"-> {result['temperature']['after']['ece']:.3f} at T={result['temperature']['temperature']:.3f} "
         f"(fitted on calibration)"),
    ]
    for method, entry in sorted(axis2["by_method"].items()):
        lines.append(f"  conformal[{method}] nominal {axis2['nominal_coverage']:.2f} "
                     f"empirical {entry['test']['coverage']:.3f} mean set size {entry['set_sizes']['mean']:.2f} "
                     f"({entry['set_sizes']['singleton_share']:.2f} singleton)")
    lines.extend(_axis1_lines(result["axis1_panel_adequacy"]))
    chosen = result["triage"]
    lines.append(f"  triage alpha={chosen['policy']['alpha']} beta={chosen['policy']['beta']}: "
                 f"accept {chosen['coverage']:.3f} at risk {chosen['risk']}; "
                 f"request more markers {chosen['n_request_more_markers']}, abstain {chosen['n_abstain']}")
    if not result["plumbing_check"]["is_result"]:
        lines.append("  PLUMBING CHECK ONLY — not a result:")
        lines.extend(f"    - {reason}" for reason in result["plumbing_check"]["reasons"])
    return lines


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = report(args)
    except (ValueError, KeyError, FileNotFoundError, ImportError) as error:
        print(f"evaluation failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    write_json(Path(args.out), result)
    print("\n".join(_report_lines(result)))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
