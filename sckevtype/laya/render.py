"""Re-render a frozen scKevType suite into laya's input format.

`reports/laya-exploration.md` §4 measured what a `real-v2` record costs in laya's own window: every
state fits in 8,192 tokens (max 1,992), the option head is the binding constraint (2 of 11,313
sampled questions need more than 2,048), and at the checkpoint's *shipped* 1024/256 every option
definition is cut to 7 tokens — the marker lists that decide the answer never reach the model. The
pilot's first part is therefore a re-encode at ``max_len=8192 / head_max_len=2048``, and this module
is it.

Two rules make the re-render trustworthy, and both are deliberate:

**One renderer, and it is laya's.** The sequence is built by ``laya.common.build_sequence`` — the
same function laya's own training and serving paths call — rather than by a second implementation of
laya's format living here. A re-implementation would be free to drift, and the drift would look
exactly like a result. What this module supplies is the *mapping*: which scKevType question becomes
which laya question, which option is which, and where the label sits. (The same argument as
``sckevtype.contracts.render``'s "one canonical renderer", applied to the other side of the border.)

**Nothing about the suite changes.** The renderer reads a suite and writes new files; it never
touches the suite, and the states it hands laya are the suite's own ``state`` strings byte for byte.

Determinism: a pure function of ``(suite bytes, render config, laya's build_sequence)``. No
sampling, no shuffling, no dict-order dependence beyond the JSON's own insertion order — which is
part of the bytes.

The counted side (:func:`count`) exists so the re-render can be *checked* rather than asserted: it
reproduces the analytic quantities `reports/laya-exploration.md` §4 reports (state token
percentiles, head tokens needed before any truncation, total sequence length, and how many questions
truncate), and cross-checks that analytic model against real ``build_sequence`` calls on a bounded
sample. A disagreement with the report's numbers is a finding, not a rounding error.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: laya's own per-option cap, read off ``laya.common.build_sequence`` (it tokenizes each option with
#: ``truncation=True, max_length=48``). Restated here because the analytic count below has to mirror
#: it, and a silent change there would move every head-token number in the report.
LAYA_OPTION_CAP = 48

#: The framing tokens ``build_sequence`` wraps around the head and the state: ``[CLS]``, the ``[SEP]``
#: that closes the head, the ``[SEP]`` after the option block, and the trailing ``[SEP]``. Four, and
#: the fourth is the one `fit3.py` in the exploration missed — see the module docstring of
#: :func:`count`.
FRAMING_TOKENS = 4


@dataclass(frozen=True)
class RenderConfig:
    """The window the re-render targets. `SHIPPED` is the checkpoint's own; `WIDE` is the pilot's."""

    max_len: int = 8192
    head_max_len: int = 2048

    def as_dict(self) -> dict[str, int]:
        return {"max_len": self.max_len, "head_max_len": self.head_max_len}


#: What `convaiinnovations/laya-multilingual` ships (`rl_agent_config.json`): the configuration under
#: which §4 measured a median of 56% of every state read and every option cut to 7 tokens.
SHIPPED = RenderConfig(max_len=1024, head_max_len=256)

#: The pilot's configuration: 99.98% of questions render with nothing truncated.
WIDE = RenderConfig(max_len=8192, head_max_len=2048)

#: laya's question types, as ``laya.common.QTYPES``. Restated rather than imported so this module
#: stays importable without torch; :func:`check_laya` asserts the two agree.
QTYPES = {"choice": 0, "score": 1, "noul": 2}

#: The two types `real-v2` actually contains. Anything else is refused rather than guessed at: a
#: question rendered under the wrong type would be scored by the wrong head and look like a result.
SUPPORTED_TYPES = ("choice", "noul")


class RenderError(RuntimeError):
    """A record or question cannot be rendered into laya's format without inventing something."""


# --------------------------------------------------------------------------- the mapping


def option_keys(question: Mapping[str, Any]) -> list[str]:
    """The question's option keys, **in the order laya will render its options**.

    This is the contract the whole pilot rests on: ``markers[i]`` in the built sequence is option
    ``i``'s position, ``target[i]`` is its target, and a probability vector read back off the model
    is indexed the same way. It matches ``kev.api.question_keys`` — for ``choice`` the criteria
    dict's own order, for ``noul`` the fixed semantic order ``[false, true]`` that
    ``laya.common.render_options`` uses regardless of how the criteria dict is written.
    """
    qtype = question["type"]
    if qtype == "choice":
        return list(question["criteria"])
    if qtype == "noul":
        return ["false", "true"]
    raise RenderError(f"no option order is defined for question type {qtype!r}")


def label_index(question: Mapping[str, Any], keys: Sequence[str]) -> int:
    """Where the label sits among ``keys`` — ``kev.benchmark.labels``' rule, restated for one row."""
    if question["type"] == "choice":
        label = question["label"]
        if label not in keys:
            raise RenderError(f"label {label!r} is not among the options {list(keys)!r}")
        return list(keys).index(label)
    return int(question["label"])


def laya_question(question: Mapping[str, Any]) -> dict[str, Any]:
    """The dict ``laya.common.build_sequence`` takes: ``{t, ins, crit}``.

    ``ins`` is passed through untouched — a dict for ``noul``, a string for ``choice`` — because
    ``build_sequence`` stringifies it itself and the counted numbers in the exploration were measured
    through that same path. Reformatting it here would make the re-render disagree with the
    measurement that justifies it.
    """
    if question["type"] not in SUPPORTED_TYPES:
        raise RenderError(f"unsupported question type {question['type']!r}")
    return {"t": question["type"], "ins": question["instructions"], "crit": question["criteria"]}


def target_of(question: Mapping[str, Any], keys: Sequence[str]) -> list[float]:
    """The one-hot target laya's ``proper_reward`` scores against, in option order."""
    index = label_index(question, keys)
    return [1.0 if i == index else 0.0 for i in range(len(keys))]


# --------------------------------------------------------------------------- tokenization


def _mask(tok: Any) -> str:
    return getattr(tok, "mask_token", None) or "[MASK]"


def instruction_ids(tok: Any, question: Mapping[str, Any]) -> list[int]:
    """The head's instruction tokens, tokenized exactly as ``build_sequence`` tokenizes them."""
    text = f"{question['type']} question: {str(question['instructions']).replace(_mask(tok), ' ')}"
    return tok(text, add_special_tokens=False)["input_ids"]


def option_ids(tok: Any, opts: Sequence[str]) -> list[list[int]]:
    """Per-option token ids at laya's 48-token cap, one batched tokenizer call for the whole list."""
    if not opts:
        return []
    return tok(
        [" " + opt.replace(_mask(tok), " ") for opt in opts],
        add_special_tokens=False,
        truncation=True,
        max_length=LAYA_OPTION_CAP,
    )["input_ids"]


def head_tokens(instr: int, opts: Sequence[Sequence[int]]) -> int:
    """Head tokens *needed*: the instructions plus one mask per option plus each option's tokens.

    This is the quantity `reports/laya-exploration.md` §4 reports as "head tokens needed", before
    ``build_sequence`` shortens either half of it — which is the point of reporting it: it is the
    requirement, and the truncation columns below say whether the requirement was met.
    """
    return instr + sum(1 + len(option) for option in opts)


def render_options(tok: Any, question: Mapping[str, Any]) -> list[str]:
    """The option texts ``render_options`` would produce — resolved by calling laya's own.

    Imported lazily and called rather than re-derived: an option's text is part of the prompt, and
    the prompt is what the model's numbers mean.
    """
    from laya.common import render_options as laya_render_options

    return list(laya_render_options(laya_question(question)))


# --------------------------------------------------------------------------- building items


@dataclass
class BuiltItems:
    """What one partition rendered into, plus everything that did not make it."""

    items: list[dict[str, Any]]
    dropped: list[dict[str, Any]]
    questions: int
    records: int


def build_record(record: Mapping[str, Any], tok: Any, cfg: RenderConfig, split: str) -> BuiltItems:
    """One record's questions as laya items.

    A question whose option markers do not all survive ``max_len`` is **dropped, not padded**. laya's
    answer is a distribution over the options whose markers survived, so a question that lost one has
    a different option set from the one the suite labelled — scoring it would enter a row for a
    question nobody asked, and its target would not even have the right length for
    ``laya.common.collate_items``. The count is reported by the caller and belongs in the log.
    """
    from laya.common import build_sequence

    state = record["state"]
    state_ids = tok(state, add_special_tokens=False)["input_ids"]
    meta = record["_meta"]
    items: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for qid, question in record["questions"].items():
        keys = option_keys(question)
        ask = laya_question(question)
        try:
            target = target_of(question, keys)
        except RenderError as error:
            dropped.append({"id": meta["id"], "question": qid, "reason": str(error)})
            continue
        ids, markers = build_sequence(tok, state, ask, cfg.max_len, cfg.head_max_len, state_ids=state_ids)
        if len(markers) != len(keys):
            dropped.append(
                {
                    "id": meta["id"],
                    "question": qid,
                    "reason": f"{len(keys) - len(markers)} of {len(keys)} option markers fall outside max_len={cfg.max_len}",
                }
            )
            continue
        items.append(
            {
                "ids": ids,
                "markers": markers,
                "target": target,
                "qtype": QTYPES[question["type"]],
                "label": label_index(question, keys),
                # Carried through for the scoring pass, which has to write these back onto a row for
                # sckevtype.eval to stratify by. `collate_items` drops them into `meta`, so the
                # training path never sees them.
                "id": meta["id"],
                "group": meta.get("group_id"),
                "question": qid,
                "split": split,
                "family": str(meta.get("family", "unknown")),
                "variant": str(meta.get("variant", "unknown")),
                "source": str(meta.get("source", "unknown")),
                "task": str(question.get("src", "unknown")),
                "dataset": str(meta.get("dataset_id", "unknown")),
                "cell": str(meta.get("cell_id", "unknown")),
                "donor": str(meta.get("donor_id", "unknown")),
                "panel": str(meta.get("panel_id", "unknown")),
                "keys": keys,
                "options": render_options(tok, question),
            }
        )
    return BuiltItems(items=items, dropped=dropped, questions=len(record["questions"]), records=1)


def build_partition(
    records: Iterable[Mapping[str, Any]], tok: Any, cfg: RenderConfig, split: str
) -> BuiltItems:
    """Every record of one partition, in file order, concatenated."""
    items: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    questions = count = 0
    for record in records:
        built = build_record(record, tok, cfg, split)
        items.extend(built.items)
        dropped.extend(built.dropped)
        questions += built.questions
        count += 1
    return BuiltItems(items=items, dropped=dropped, questions=questions, records=count)


# --------------------------------------------------------------------------- counting


def _percentiles(values: Sequence[int]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    import numpy as np

    array = np.asarray(values)
    return {
        "n": int(array.size),
        "min": int(array.min()),
        "p50": int(np.percentile(array, 50)),
        "p90": int(np.percentile(array, 90)),
        "p95": int(np.percentile(array, 95)),
        "p99": int(np.percentile(array, 99)),
        "max": int(array.max()),
        "mean": round(float(array.mean()), 1),
    }


def count_states(records: Iterable[Mapping[str, Any]], tok: Any, cfg: RenderConfig, batch: int = 2000) -> dict[str, Any]:
    """State token counts over a whole partition, tokenized in batches.

    This is the one distribution the exploration measured over the *full* 89,802-record file rather
    than a sample, so the counter has to be able to reach the whole file without tokenizing a single
    option: batching is what makes that a minute instead of an hour.
    """
    lengths: list[int] = []
    records_seen = 0
    pending: list[str] = []
    for record in records:
        records_seen += 1
        pending.append(record["state"])
        if len(pending) >= batch:
            lengths.extend(len(ids) for ids in tok(pending, add_special_tokens=False)["input_ids"])
            pending = []
    if pending:
        lengths.extend(len(ids) for ids in tok(pending, add_special_tokens=False)["input_ids"])
    return {
        "config": cfg.as_dict(),
        "records": records_seen,
        "questions": None,
        "state_tokens": _percentiles(lengths),
        "states_over_max_len": sum(1 for length in lengths if length > cfg.max_len),
    }


def count_record(record: Mapping[str, Any], tok: Any, cfg: RenderConfig) -> dict[str, Any]:
    """The analytic token cost of one record: state tokens, and per question the head and total.

    "Head tokens needed" and "total sequence" are counted the way `fit3.py`/`fit4.py` counted them,
    so the numbers here are comparable to the ones in the report::

        head  = len(instruction tokens) + sum(1 + len(option tokens at the 48-token cap))
        total = head + len(state tokens) + FRAMING_TOKENS

    ``FRAMING_TOKENS`` is 4 and the exploration's `fit3.py` used 3 — the report says so itself
    ("the exact count is +4 — a single token") and `fit4.py` re-derived the fit verdict with it. The
    one-token difference moves no percentile here, and :func:`count` reports both so the reader can
    see that rather than take it on trust.
    """
    state_ids = tok(record["state"], add_special_tokens=False)["input_ids"]
    questions = list(record["questions"].values())
    if not questions:
        return {"state": len(state_ids), "questions": []}
    instr = tok(
        [
            f"{q['type']} question: {str(q['instructions']).replace(_mask(tok), ' ')}"
            for q in questions
        ],
        add_special_tokens=False,
    )["input_ids"]
    flat_opts: list[str] = []
    spans: list[tuple[int, int]] = []
    for question in questions:
        options = render_options(tok, question)
        spans.append((len(flat_opts), len(flat_opts) + len(options)))
        flat_opts.extend(options)
    tokenized = option_ids(tok, flat_opts)
    per_question = []
    for question, instruction_tokens, (start, end) in zip(questions, instr, spans):
        options = tokenized[start:end]
        head = head_tokens(len(instruction_tokens), options)
        per_question.append(
            {
                "type": question["type"],
                "options": len(options),
                "option_tokens_max": max((len(o) for o in options), default=0),
                "option_tokens_total": sum(len(o) for o in options),
                "head": head,
                "total3": head + len(state_ids) + 3,
                "total4": head + len(state_ids) + FRAMING_TOKENS,
            }
        )
    return {"state": len(state_ids), "questions": per_question}


def count(
    records: Iterable[Mapping[str, Any]],
    tok: Any,
    cfg: RenderConfig,
    check: int = 0,
    check_seed: int = 20260926,
) -> dict[str, Any]:
    """The counted output for a stream of records: distributions, truncation verdicts, and a check.

    ``check`` > 0 additionally builds ``check`` real sequences with ``laya.common.build_sequence`` and
    confirms the analytic total equals the real one and that every option marker survived — the
    analytic model is what produces the distributions, so it is the thing that has to be checked
    against the implementation it mirrors. The sample is drawn with a fixed seed, so the check is as
    reproducible as the rest.
    """
    states: list[int] = []
    heads: list[int] = []
    totals3: list[int] = []
    totals4: list[int] = []
    option_max: list[int] = []
    option_counts: list[int] = []
    questions = records_seen = 0
    over_head = over_state = over_max = truncated = at_option_cap = 0
    check_n = check_ok = 0
    check_mismatch: list[dict[str, Any]] = []
    rng = random.Random(check_seed)
    for record in records:
        records_seen += 1
        counted = count_record(record, tok, cfg)
        states.append(counted["state"])
        if counted["state"] > cfg.max_len:
            over_state += 1
        for index, entry in enumerate(counted["questions"]):
            questions += 1
            heads.append(entry["head"])
            totals3.append(entry["total3"])
            totals4.append(entry["total4"])
            option_max.append(entry["option_tokens_max"])
            option_counts.append(entry["options"])
            if entry["option_tokens_max"] >= LAYA_OPTION_CAP:
                at_option_cap += 1
            if entry["head"] > cfg.head_max_len:
                over_head += 1
            if entry["total4"] > cfg.max_len:
                over_max += 1
            # "Truncated" is a question build_sequence would have to shorten: the option block over
            # its own budget, or the whole sequence over the window. Either one alone is enough.
            if entry["head"] > cfg.head_max_len or entry["total4"] > cfg.max_len:
                truncated += 1
            if check and rng.random() < check:
                real, markers = _real_sequence(record, index, tok, cfg)
                ok = len(real) == entry["total4"] and len(markers) == entry["options"]
                check_n += 1
                check_ok += int(ok)
                if not ok and len(check_mismatch) < 10:
                    check_mismatch.append(
                        {
                            "id": record["_meta"]["id"],
                            "question": list(record["questions"])[index],
                            "analytic_total": entry["total4"],
                            "real_length": len(real),
                            "options": entry["options"],
                            "markers": len(markers),
                        }
                    )
    return {
        "config": cfg.as_dict(),
        "records": records_seen,
        "questions": questions,
        "state_tokens": _percentiles(states),
        "head_tokens_needed": _percentiles(heads),
        "total_sequence_plus4": _percentiles(totals4),
        "total_sequence_plus3": _percentiles(totals3),
        # Excluding the [MASK] prefix build_sequence puts in front of every option, so a definition
        # that arrives whole reads 48 here and 49 as a marker-to-marker span — the report's number.
        "longest_option_tokens_per_question": _percentiles(option_max),
        "questions_whose_longest_option_reached_the_48_cap": at_option_cap,
        "options_per_question": _percentiles(option_counts),
        "over_head_budget": over_head,
        "over_max_len": over_max,
        "states_over_max_len": over_state,
        "truncated": truncated,
        "nothing_truncated": questions - truncated,
        "check": {
            "sampled": check_n,
            "analytic_matches_build_sequence": check_ok,
            "mismatches": check_mismatch,
        }
        if check
        else None,
    }


def _real_sequence(record: Mapping[str, Any], index: int, tok: Any, cfg: RenderConfig) -> tuple[list[int], list[int]]:
    from laya.common import build_sequence

    question = list(record["questions"].values())[index]
    return build_sequence(tok, record["state"], laya_question(question), cfg.max_len, cfg.head_max_len)


# --------------------------------------------------------------------------- loading and CLI


def read_records(path: Path) -> Iterator[dict[str, Any]]:
    with open(path, "rb") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def sample_records(path: Path, fraction: float, seed: int) -> Iterator[dict[str, Any]]:
    """Every ``fraction`` of the file's records by one fixed draw per line.

    Exists to reproduce the exploration's own samples, which is how the counted output gets compared
    to its numbers: `fit3.py` drew at 10% with seed 20260926 over `test.jsonl`.
    """
    rng = random.Random(seed)
    for record in read_records(path):
        if rng.random() < fraction:
            yield record


def partition_path(suite: Path, split: str) -> Path:
    path = suite / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"no {split}.jsonl in {suite}")
    return path


def check_laya() -> dict[str, Any]:
    """What the re-render is about to be built by — recorded, so a drift is visible in the log."""
    import hashlib

    import laya
    from laya.common import QTYPES as laya_qtypes

    if dict(laya_qtypes) != QTYPES:
        raise RenderError(f"laya's question types moved: {laya_qtypes} != {QTYPES}")
    source = Path(laya.__file__).resolve().parent / "common.py"
    return {
        "laya_version": getattr(laya, "__version__", "unknown"),
        "laya_common_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "qtypes": dict(laya_qtypes),
    }


def load_tokenizer(path: str | None) -> Any:
    """The tokenizer laya's checkpoint ships. ``None`` means "the checkpoint's own default"."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path or "jhu-clsp/mmBERT-base")


def _cmd_count(args: argparse.Namespace) -> int:
    tok = load_tokenizer(args.tokenizer)
    cfg = RenderConfig(args.max_len, args.head_max_len)
    path = partition_path(Path(args.suite), args.split)
    stream = select(path, sample_frac=args.sample_frac, sample_seed=args.sample_seed, limit=args.limit)
    result = (
        count_states(stream, tok, cfg)
        if args.states_only
        else count(stream, tok, cfg, check=args.check, check_seed=args.sample_seed)
    )
    result["source"] = {"suite": args.suite, "split": args.split, "sample_frac": args.sample_frac,
                        "sample_seed": args.sample_seed, "limit": args.limit, "tokenizer": args.tokenizer,
                        "states_only": args.states_only}
    result["laya"] = check_laya()
    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")
    return 0


def select(path: Path, sample_frac: float = 1.0, sample_seed: int = 20260926,
           limit: int = 0) -> Iterator[dict[str, Any]]:
    """A partition's records, optionally sampled and truncated — one place both commands select in."""
    stream: Iterable[dict[str, Any]] = (
        sample_records(path, sample_frac, sample_seed) if sample_frac < 1.0 else read_records(path)
    )
    for index, record in enumerate(stream):
        if limit and index >= limit:
            return
        yield record


def _cmd_build(args: argparse.Namespace) -> int:
    import hashlib

    import torch

    tok = load_tokenizer(args.tokenizer)
    cfg = RenderConfig(args.max_len, args.head_max_len)
    suite = Path(args.suite)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"suite": str(suite), "config": cfg.as_dict(), "laya": check_laya(),
                                "tokenizer": args.tokenizer, "splits": {}}
    for split in args.splits.split(","):
        split = split.strip()
        records = select(partition_path(suite, split), limit=args.limit)
        built = build_partition(records, tok, cfg, split)
        path = out / f"{split}.pt"
        torch.save(built.items, path)
        manifest["splits"][split] = {
            "records": built.records,
            "questions": built.questions,
            "items": len(built.items),
            "dropped": len(built.dropped),
            "dropped_examples": built.dropped[:5],
            "path": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
        }
        print(f"  {split:<12} records={built.records} questions={built.questions} "
              f"items={len(built.items)} dropped={len(built.dropped)} -> {path.name} "
              f"({path.stat().st_size / 1e6:.1f} MB)")
    (out / "render-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {out / 'render-manifest.json'}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="sckevtype.laya.render",
                                     description="Re-render a frozen suite into laya's input format.")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (("count", "count tokens and truncation without writing items"),
                            ("build", "write laya training items")):
        child = sub.add_parser(name, help=help_text)
        child.add_argument("--suite", required=True, help="frozen suite directory")
        child.add_argument("--max-len", type=int, default=WIDE.max_len)
        child.add_argument("--head-max-len", type=int, default=WIDE.head_max_len)
        child.add_argument("--tokenizer", default=None, help="tokenizer directory (default: mmBERT-base)")
        child.add_argument("--limit", type=int, default=0, help="stop after N records (0 = all)")
        child.set_defaults(func=_cmd_count if name == "count" else _cmd_build)
    counted = sub.choices["count"]
    counted.add_argument("--split", default="test")
    counted.add_argument("--sample-frac", type=float, default=1.0,
                         help="draw this fraction of records (to reproduce the exploration's samples)")
    counted.add_argument("--sample-seed", type=int, default=20260926)
    counted.add_argument("--check", type=float, default=0.0,
                         help="verify the analytic count against real build_sequence on this fraction")
    counted.add_argument("--states-only", action="store_true",
                         help="count state tokens only (batched; the whole file in about a minute)")
    counted.add_argument("--out", default=None)
    sub.choices["build"].add_argument("--splits", default="train,calibration")
    sub.choices["build"].add_argument("--out", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return args.func(args)
    except (RenderError, FileNotFoundError, ImportError) as error:
        print(f"re-render failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
