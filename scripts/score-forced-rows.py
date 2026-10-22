#!/usr/bin/env python3
"""Re-score one arm on a *named* population, keeping the option-probability vector ``p``.

Why this exists
---------------

``reports/rows-full-<arm>.jsonl`` — the dump ``sckevtype.eval.run --rows-full`` writes — records
``top1``, ``top1_p`` and ``none_p`` per scored choice row but **not** the vector they were read
from. That is enough to say which option the model named and not enough to say which option it
would have named with the abstain option removed: on the rows where ``top1`` *is* the abstain
option (0.86–0.98 of our arms' census-matched rows) the runner-up is on no field of the row. So
our own arms cannot enter the forced-naming table every other engine is in, and no amount of
re-reading the archive recovers them — the numbers have to be re-scored.

``write_full_row_dump`` now carries ``p`` (see ``sckevtype/eval/run.py``). This script produces
that dump for a restricted population, so the re-score costs the population and not the suite: the
same suite, checkpoint, device, temperature (T = 1, raw logits) and row construction the eval CLI
uses, driven through the same :func:`sckevtype.eval.run.score_partition`, so the rows it writes are
the rows the CLI would write for the same records. That claim is checked row by row against the
archived dump by ``scripts/arms-forced.py`` (``fidelity`` block: re-scored ``top1`` must equal the
archived ``top1`` on every row of the population, and re-scored ``top1_p`` must agree to 1e-9).

What it deliberately does **not** do: fit anything, report coverage or accuracy, or write an eval
report. It is a re-score of one population for one purpose, and its output is a row dump in the
full dump's own shape — not a second scorer for the suite. The suite-level eval pass stays the only
thing that produces ``reports/eval-<arm>-rowsfull.json``.

Two question types, one row shape
---------------------------------

``--question-type`` selects which rows the dump keeps, and with it the writer:

* ``choice`` (default) — the type the forced-naming table is taken on, through
  ``write_full_row_dump``;
* ``noul`` — the reference's binary marker question (``keys == ["false","true"]``, ``label`` 0 for
  *no* and 1 for *yes*, ``p[1]`` = P(yes)), through ``write_noul_row_dump``, in the *same*
  ``FULL_DUMP_FIELDS`` shape. ``marker_noul`` is the population here whose questions are all
  ``noul``: 4,035 records carrying **63,851 noul questions** (4–41 per record), which is why its
  acceptance is a count of **questions**, not of records — a dump holding 4,035 records and no
  questions has scored nothing, and that is exactly what the choice writer does with this family
  (it skips every non-choice row and reports the loss only in its ``skipped`` count).

``--expect-questions N`` turns that acceptance into a refusal: the run does not exit 0 unless the
dump holds exactly N questions. ``--append`` makes a wall-clock kill survivable: chunks are written
as they are scored and a re-run continues the partial dump instead of throwing it away. The noul
population costs 1.4–1.8x the packed tokens per record of a choice population on the same records
(measured here: p50 2,807 tokens against 2,066, p95 4,981 against 2,824), so the 9B rung's pass is
long enough that an undersized wall is a real failure mode — measured once already, when a 3 h
default killed the 9B at 3,200/4,035 and discarded every row it had scored.

``--rows-per-pass N`` caps kev's own per-forward-pass row budget (``kev.model.rows_per_pass``), which
is what a scoring pass's memory is sized by: a pass holds ``rows_per_pass`` rows padded to the longest
of them, so the cost is ``rows x longest row`` with an L x L attention term that is quadratic in the
longest row. It is the same lever this repo's trainer applies to *training* passes as
``--token_budget`` — cut between rows, never inside one — and it was needed here: the 9B rung's noul
pass died with ``this process has 39.17 GiB memory in use`` inside ``_rows_hidden`` on a 40 GB card,
on a population whose rows are longer than the choice family the same rung scored on one card. The
rows are independent by construction (``kev.model.rows_of``: state + one question), so the answers do
not depend on the split; ``0`` (the default) leaves kev's own budget in place, and the value used is
recorded in the meta file beside the population.

Usage
-----

On a compute node with the venv and a GPU (``cd $W`` first — the package is not installed into the
venv, it is imported from the repo root, which this script puts on ``sys.path`` itself)::

    $W/venv-train/bin/python scripts/score-forced-rows.py \\
        --suite artifacts/suites/real-v2-eval --checkpoint runs/delta-kev-0.8b-s1 \\
        --ids reports/census-suite-ids.jsonl --family panel_choice \\
        --device cuda --out reports/forced-rows-delta-kev-0.8b-s1.jsonl

    $W/venv-train/bin/python scripts/score-forced-rows.py \\
        --suite artifacts/suites/real-v4-eval --checkpoint runs/delta-kev-9b-v4-s1 \\
        --ids reports/census-suite-ids-v4.jsonl --family marker_noul --question-type noul \\
        --expect-questions 63851 --append \\
        --device cuda --out reports/forced-rows-delta-kev-9b-v4-s1-marker_noul.jsonl

``--limit N`` scores only the first N records of the population and says so in the meta file; it
exists for a smoke probe and its output must never be read as the population.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:          # the package lives in the repo, not in the venv
    sys.path.insert(0, str(ROOT))

from sckevtype.eval import TEST
from sckevtype.eval.run import (
    CHOICE,
    NOUL,
    checkpoint_summary,
    dump_lines,
    score_partition,
    write_full_row_dump,
    write_noul_row_dump,
)

#: One writer per question type, so the type that selects the rows and the type that shapes the dump
#: cannot disagree: ``--question-type`` indexes this map and nothing else branches on the type.
WRITERS = {CHOICE: write_full_row_dump, NOUL: write_noul_row_dump}

#: The serving split this harness may cap, and the one place it is named. kev bounds a scoring pass by
#: the *tokens of rows* it carries, not by a record count: ``kev.model._rows_hidden`` takes
#: ``rows_per_pass(rows, prefix_len)`` rows at a time, ``max(1, budget // (prefix_len + longest row))``
#: for ``budget = SERVE_MAX_PACKED`` (16,384), and a pass's memory is the padded shape that builds —
#: ``rows x longest row`` — with the L x L attention term quadratic in the longest row.
SERVE_MAX_PACKED = None       # filled in from kev at first use; never re-derived here

#: The function ``cap_rows_per_pass`` replaced, so :func:`cap_rows_per_pass` can put kev's own back
#: (``restore=True``). Held here rather than read back off the module: after a cap the module's
#: attribute *is* the cap, so "restore to what is there now" would be a no-op that reads like a reset.
_ORIGINAL_ROWS_PER_PASS = None


def cap_rows_per_pass(budget: int, restore: bool = False) -> Any:
    """Cap kev's own per-forward-pass row budget at ``budget`` tokens, returning what it replaced.

    Measured need, not a precaution: the 9B rung's noul pass (job 4328448) died on a 40 GB card with
    ``this process has 39.17 GiB memory in use`` inside ``_rows_hidden``, on a population whose rows are
    1.36x the packed tokens of the choice family the same rung scored on one card. The lever is the one
    kev documents for exactly this (``rows_per_pass``: *"Memory per pass is bounded by one maximal row
    however many questions a request has, and the rows are independent, so the answers do not depend on
    the split"*) and the one this repo's trainer already applies to training passes as
    ``--token_budget``: cut *between* rows, never inside one, so every question is still scored against
    its own state+definition row with the same weights, the same fp32-exact kernels and the same T = 1.

    The cap only ever makes a pass **smaller** (``min`` against kev's own rule, called with kev's own
    function rather than a copy of its arithmetic), so a budget at or above kev's default is a no-op and
    ``0`` is refused rather than interpreted. ``restore=True`` puts the function this module replaced
    back, which is how a test proves nothing else was touched.
    """
    import kev.model as kev_model

    global SERVE_MAX_PACKED, _ORIGINAL_ROWS_PER_PASS
    SERVE_MAX_PACKED = kev_model.SERVE_MAX_PACKED
    if _ORIGINAL_ROWS_PER_PASS is None:
        _ORIGINAL_ROWS_PER_PASS = kev_model.rows_per_pass
    if restore:
        kev_model.rows_per_pass = _ORIGINAL_ROWS_PER_PASS
        return _ORIGINAL_ROWS_PER_PASS
    if not isinstance(budget, int) or budget <= 0:
        raise ValueError(f"rows-per-pass budget must be a positive int, not {budget!r}: 0 would be "
                         f"'no rows in a pass', and every score comes from a pass")
    original = _ORIGINAL_ROWS_PER_PASS

    def capped(rows, prefix_len=0):
        return min(original(rows, prefix_len), original(rows, prefix_len, budget))

    capped.__doc__ = f"kev.model.rows_per_pass capped at {budget} tokens per pass ({original.__doc__})"
    kev_model.rows_per_pass = capped
    return original


def rows_per_pass_record(budget: int) -> dict:
    """What the meta file says about the split a dump was scored under — null when it is kev's own.

    A dump scored with a smaller per-pass budget carries the same rows in the same order, so nothing in
    the dump itself could say so; the meta is where a reader finds it, beside the population and the
    checkpoint it belongs to.
    """
    return {"budget": budget or None, "kev_default": SERVE_MAX_PACKED,
            "capped": bool(budget), "why": ("kev.model.rows_per_pass, capped for this pass: memory per "
                                            "pass is the padded rows x longest row, so a long-row "
                                            "population can need it (see cap_rows_per_pass)")}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_ids(path: Path, family: str | None) -> list[str]:
    """The ids to score, in the order the file lists them.

    ``family`` filters on the record's own ``family`` field (``reports/census-suite-ids.jsonl``
    carries the family of every suite row), so the population is named by the same field the rest
    of the analysis joins on rather than by a list kept in a second place. Duplicates are refused:
    a repeated id would score one question twice and quietly weight it double.
    """
    ids: list[str] = []
    with open(path, encoding="utf-8") as fh:
        for number, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if family is not None and record.get("family") != family:
                continue
            if "id" not in record:
                raise ValueError(f"{path}:{number} has no 'id' field")
            ids.append(record["id"])
    if len(set(ids)) != len(ids):
        seen: set[str] = set()
        repeated = sorted({i for i in ids if i in seen or seen.add(i)})
        raise ValueError(f"{path}: {len(ids) - len(set(ids))} repeated ids, e.g. {repeated[:5]}")
    if not ids:
        raise ValueError(f"{path}: no ids selected (family filter {family!r})")
    return ids


def select_records(records: list[dict], ids: list[str], split: str) -> list[dict]:
    """The records whose ``_meta.id`` is in ``ids``, in the suite's own order.

    A requested id the partition does not carry **raises**: scoring the ids that happen to be there
    would produce a population that is smaller than the one asked for and identical in shape, which
    is exactly how a partial population starts looking like a full one. The suite's order (not the
    id file's) is kept, so the dump's row order is the order the eval pass would have written.
    """
    wanted = set(ids)
    chosen = [record for record in records if record["_meta"]["id"] in wanted]
    found = {record["_meta"]["id"] for record in chosen}
    missing = wanted - found
    if missing:
        raise ValueError(f"{len(missing)} of the {len(ids)} requested ids are not in the {split} "
                         f"partition of this suite, e.g. {sorted(missing)[:5]} -- refusing to score "
                         f"a population smaller than the one asked for")
    return chosen


def chunks(items: list, size: int):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def resume_state(path: Path, order: Sequence[str]) -> tuple[list[str], int]:
    """The lines of a partially written dump worth keeping, and the record index to resume at.

    A chunked pass can be killed between chunks, so a dump on disk is at best a prefix of the
    population. Two rules make continuing it safe:

    * lines are kept whole and the caller **rewrites** the file with exactly those lines, so the
      resumed dump is byte-for-byte the dump a single-shot pass would have written (one JSON object
      per row, in record order and then question order);
    * the **last** record present is re-scored rather than trusted. A record's rows are the only
      unit whose completeness is not recoverable from the file — every row carries the record's
      ``id`` and never its position — so a kill mid-record leaves a record that looks finished.
      Re-scoring it costs one record; trusting it would publish a record with rows missing.

    A file that does not end in a newline has a truncated last line, which is dropped. An id this
    population does not carry, a line that is not a dump row, or rows out of record order all
    **raise**: continuing a dump written from some other population would put two populations in
    one file.
    """
    if not path.exists() or path.stat().st_size == 0:
        return [], 0
    raw = path.read_text(encoding="utf-8").split("\n")
    if raw and raw[-1] == "":
        raw.pop()                                    # the writer's own trailing newline
    elif raw:
        raw.pop()                                    # a truncated last line: not a row
    if not raw:
        return [], 0
    position = {record_id: index for index, record_id in enumerate(order)}
    ids = []
    for number, line in enumerate(raw, start=1):
        try:
            record_id = json.loads(line)["id"]
        except (ValueError, KeyError, TypeError) as error:
            raise ValueError(f"{path}:{number} is not a dump row ({type(error).__name__}: {error}); "
                             f"refusing to continue a file this script did not write") from error
        if record_id not in position:
            raise ValueError(f"{path}:{number} carries id {record_id!r}, which is not in this "
                             f"population -- refusing to continue a dump of another population")
        ids.append(record_id)
    indexes = [position[record_id] for record_id in ids]
    if indexes != sorted(indexes):
        raise ValueError(f"{path}: its records are not in the suite's order -- refusing to continue "
                         f"a dump whose rows cannot be told apart from a re-ordered one")
    start = indexes[-1]
    return [line for line, index in zip(raw, indexes) if index < start], start


def expected_questions(records: Sequence[dict], suite: Path, split: str) -> int:
    """How many ``noul`` questions the given records ask — the acceptance unit, read from the suite.

    Counted from the suite rather than from the population file because the ids file carries no
    question information at all (``reports/census-suite-ids-v4.jsonl`` holds ``id``/``dataset``/
    ``cell``/``family``/``variant``), and a count taken from anything but the records themselves
    would be a second place for the population to disagree with itself.
    """
    wanted = {record["_meta"]["id"] for record in records}
    total = 0
    with open(suite / f"{split}.jsonl", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            record = json.loads(line)
            if record["_meta"]["id"] in wanted:
                total += sum(1 for question in record["questions"].values() if question["type"] == NOUL)
    return total


def rescore(chosen: Sequence[dict], predictor: Any, split: str, out_path: Path, *,
            question_type: str = CHOICE, chunk: int = 64, append: bool = False,
            say: Callable[[str], None] = print) -> tuple[dict, list[dict], int]:
    """Score ``chosen`` through the eval CLI's own path and write the dump; -> (info, rejected, resumed).

    This is everything below the model: ``score_partition``'s row construction, the writer for
    ``question_type``, the resume rule when ``append`` is set, and the dump's own counts. It takes the
    predictor as an argument precisely so it can be exercised with a fake one and no GPU — the code
    under test is the real code, only the forward pass is a stub.

    With ``append`` the dump is written chunk by chunk, so a wall-clock kill costs the tail instead of
    the whole pass; a resumed pass rewrites the file with the lines it keeps and re-scores the last
    record present (:func:`resume_state`). Raises ``ValueError`` when the pass would leave an **empty**
    dump: a file with no rows cannot be told from a population that was never scored.
    """
    rows: list[dict] = []
    rejected: list[dict] = []
    written = skipped_total = 0
    resumed_from = 0
    todo = chosen
    if append:
        kept, resumed_from = resume_state(out_path, [record["_meta"]["id"] for record in chosen])
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.writelines(line + "\n" for line in kept)
        written = len(kept)
        todo = chosen[resumed_from:]
        say(f"resume     : {resumed_from} of {len(chosen)} records already in {out_path} "
            f"({written} lines kept); re-scoring from record {resumed_from + 1}" if resumed_from
            else f"append     : writing {out_path} as each chunk lands")

    scoring_started = time.time()
    for index, group in enumerate(chunks(list(todo), max(1, chunk)), start=1):
        group_rows, group_rejected = score_partition(group, predictor, split)
        rejected.extend(group_rejected)
        if append:
            lines, skipped = dump_lines(group_rows, question_type)
            with open(out_path, "a", encoding="utf-8") as fh:
                fh.write("\n".join(lines) + "\n")
            written += len(lines)
            skipped_total += skipped
        else:
            rows.extend(group_rows)
        done = resumed_from + min(index * max(1, chunk), len(todo))
        elapsed = time.time() - scoring_started
        say(f"  scored {done}/{len(chosen)}  elapsed_s={elapsed:.0f}  "
            f"s_per_record={elapsed / max(done - resumed_from, 1):.3f}  rejected={len(rejected)}")

    if append:
        if written == 0:
            raise ValueError(f"no {question_type} rows were written for this population -- refusing to "
                             f"leave an empty dump, which cannot be told from a population never scored")
        info = {"path": str(out_path), "rows": written, "sha256": sha256_file(out_path),
                "skipped": skipped_total}
    else:
        info = WRITERS[question_type](out_path, rows)
    return info, rejected, resumed_from


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="score-forced-rows",
                                     description="Re-score a named population of a suite, keeping p.")
    parser.add_argument("--suite", required=True, help="suite directory (the same one the eval pass scored)")
    parser.add_argument("--checkpoint", required=True, help="run directory holding the LoRA adapter and head.pt")
    parser.add_argument("--ids", required=True, help="JSONL of {'id': ...} records naming the population")
    parser.add_argument("--family", default="panel_choice", help="only ids whose record carries this family ('' = no filter)")
    parser.add_argument("--split", default=TEST, help=f"partition to read the records from (default {TEST})")
    parser.add_argument("--question-type", choices=(CHOICE, NOUL), default=CHOICE,
                        help=f"which question type the dump keeps: {CHOICE} (default) writes the rows the "
                             f"forced rule needs, {NOUL} writes the binary marker rows (keys "
                             f"['false','true']), one row per question")
    parser.add_argument("--expect-questions", type=int, default=0,
                        help="refuse to finish unless the dump holds exactly this many rows/questions "
                             "(0 = no check). For a noul population this is the suite's own count over "
                             "the same records, so a dump that lost records cannot read as a result.")
    parser.add_argument("--append", action="store_true",
                        help="write each chunk as it is scored and continue a partially written dump "
                             "instead of accumulating in memory and writing once at the end; the bytes "
                             "are the same, and a wall-clock kill then costs the tail rather than all")
    parser.add_argument("--out", required=True, help="row dump to write (.jsonl), in the full dump's shape")
    parser.add_argument("--meta-out", default=None, help="provenance JSON (default: <out>.meta.json)")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default=None)
    parser.add_argument("--limit", type=int, default=0, help="score only the first N records (smoke probe; 0 = all)")
    parser.add_argument("--chunk", type=int, default=64, help="records per progress line")
    parser.add_argument("--rows-per-pass", type=int, default=0,
                        help="cap kev's per-forward-pass row budget at this many tokens (0 = kev's own "
                             "SERVE_MAX_PACKED, 16384). It cuts BETWEEN a record's question rows and "
                             "never inside one, so every question is scored against the same row with the "
                             "same weights and the same T = 1 -- it exists because a population with long "
                             "rows (marker_noul) can OOM a 9B on one 40 GB card at the default budget")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from kev.checkpoint import LoadOptions
    from kev.device import default_device
    from kev.predictors import LocalPredictor
    from kev.suite import load_split, read_manifest

    started = time.time()
    ids_path = Path(args.ids)
    ids = read_ids(ids_path, args.family or None)
    manifest = read_manifest(args.suite)
    records = load_split(args.suite, args.split, allow_test=True)
    chosen = select_records(records, ids, args.split)
    if args.limit:
        chosen = chosen[:args.limit]

    device = args.device or default_device()
    context = manifest.get("context")
    print(f"suite      : {args.suite}  ({args.split} partition, {len(records)} records, "
          f"manifest {sha256_file(Path(args.suite) / 'manifest.json')[:16]})", flush=True)
    print(f"population : {len(ids)} ids from {ids_path}"
          + (f"  LIMITED to the first {len(chosen)} records (--limit)" if args.limit else ""), flush=True)
    print(f"questions  : {args.question_type} rows of those records"
          + (f"  (acceptance: exactly {args.expect_questions})" if args.expect_questions else ""), flush=True)
    print(f"checkpoint : {args.checkpoint}   device {device}   T = 1 (raw logits)", flush=True)

    predictor = LocalPredictor(args.checkpoint, device, LoadOptions(temperature=1.0),
                               **({"context": context} if context else {}))
    print(f"model      : loaded in {time.time() - started:.1f}s", flush=True)
    if args.rows_per_pass:
        cap_rows_per_pass(args.rows_per_pass)
        print(f"rows/pass  : capped at {args.rows_per_pass} tokens (kev's own default is "
              f"{SERVE_MAX_PACKED}); rows are independent, so this changes the pass shape and not the "
              f"answers -- see --rows-per-pass", flush=True)

    out_path = Path(args.out)
    try:
        info, rejected, resumed_from = rescore(
            chosen, predictor, args.split, out_path, question_type=args.question_type,
            chunk=args.chunk, append=args.append)
    except ValueError as error:
        print(f"REFUSED: {error}", flush=True)
        return 2

    wall = time.time() - started
    meta = {
        "what_this_is": "One arm re-scored on a named population, with the option-probability vector "
                        "retained, so the forced rule can be re-applied to rows whose top1 is the "
                        "abstain option. Same suite, checkpoint, device and T = 1 row construction "
                        "as sckevtype.eval.run's full dump; the fidelity of the re-score against the "
                        "archived dump is checked by scripts/arms-forced.py.",
        "command": " ".join([Path(sys.argv[0]).name] + sys.argv[1:]),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "suite": {"path": args.suite, "split": args.split,
                  "manifest_sha256": sha256_file(Path(args.suite) / "manifest.json"),
                  "records_in_partition": len(records)},
        "ids": {"path": str(ids_path), "sha256": sha256_file(ids_path),
                "family": args.family, "n_ids": len(ids), "limit": args.limit or None,
                "n_scored": len(chosen)},
        "question_type": args.question_type,
        "rows_per_pass": rows_per_pass_record(args.rows_per_pass),
        "checkpoint": {"path": args.checkpoint, "device": device, "temperature": 1.0},
        "rows": {"written": info["rows"], "skipped": info["skipped"], "sha256": info["sha256"],
                 "out": info["path"], "expected_questions": args.expect_questions or None,
                 "resumed_from_records": resumed_from if args.append else None},
        "rejected_records": rejected,
        "wall_s": round(wall, 1),
    }
    try:
        meta["checkpoint"].update(checkpoint_summary(args.checkpoint))
    except (OSError, ValueError, KeyError, ImportError) as error:   # never cost the dump
        meta["checkpoint"]["summary_error"] = f"{type(error).__name__}: {error}"
    meta_path = Path(args.meta_out) if args.meta_out else Path(args.out + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")

    print(f"wrote {info['path']}  {args.question_type}_rows={info['rows']}  skipped={info['skipped']}  "
          f"rejected={len(rejected)}  sha256={info['sha256']}", flush=True)
    print(f"wrote {meta_path}   wall_s={wall:.1f}", flush=True)
    if rejected:
        print("REJECTED RECORDS (the population is short -- see the meta file):", flush=True)
        for entry in rejected[:10]:
            print(f"  {entry}", flush=True)
    if args.expect_questions and info["rows"] != args.expect_questions:
        print(f"REFUSED: the dump holds {info['rows']} {args.question_type} rows, the suite's own count "
              f"for this population is {args.expect_questions} -- a dump that is not the population is "
              f"not a result", flush=True)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
