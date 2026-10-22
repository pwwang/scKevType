"""laya's answers back onto the instrument step 11 used — and the pre-registered gate.

    uv run --extra train python -m sckevtype.laya.score items \\
        --suite artifacts/suites/real-v2-eval --tokenizer <ckpt>/tokenizer --out pilot/
    uv run --extra train python -m sckevtype.laya.score infer \\
        --items pilot/items --checkpoint <ft-ckpt> --out pilot/rows
    uv run --extra train python -m sckevtype.laya.score evaluate \\
        --rows pilot/rows --out reports/eval-laya-pilot.json

Three steps, because they run in two places and only the last one decides anything:

``items``
    Re-render the suite with :mod:`sckevtype.laya.render` — one item per question, plus laya's own
    ask of the abstain-bearing variant of every choice question that lacks one, exactly as
    :func:`sckevtype.eval.run.with_abstain` injects it. Only the questions are read from the suite;
    the **state is never modified**, and neither suite is written to.
``infer``
    Load the fine-tuned checkpoint and answer the items, one file of rows per chunk. A chunk whose
    rows file already exists is skipped, so a job that hits the wall clock resumes where it stopped
    instead of from the beginning. Runs where the GPU and the checkpoint are.
``evaluate``
    Rebuild :mod:`sckevtype.eval`'s row schema from those answers and run the *same* metrics on the
    *same* rows inside *the same* guards — the temperature and every threshold are fitted on
    ``calibration`` through :func:`sckevtype.eval.assert_fit_partition` and reported on ``test``.
    Runs on the box that holds the report, and needs no GPU.

What this does **not** do: it does not import laya (only :mod:`~sckevtype.laya.render` does, and only
inside its functions), and ``infer`` is the only subcommand that needs torch. The gate below is
copied from ``docs/SPEC-laya-finetune.md`` and is compared as written. Nothing in this file derives a
threshold from a number the run produces.
"""

from __future__ import annotations

import argparse
import json
import sys
from array import array
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from ..contracts.freeze import write_json
from .render import (
    QTYPES,
    SUPPORTED_TYPES,
    WIDE,
    RenderConfig,
    RenderError,
    build_record,
    check_laya,
    partition_path,
    read_records,
)

#: The pre-registered gate, verbatim from ``docs/SPEC-laya-finetune.md``. Compared as written: the
#: operator is part of the criterion, and ``ECE ≤ 0.059`` is not ``ECE < 0.059``. Three of three
#: must hold; the family-conditional breakdown below is context and never a substitute for one.
GATE_SENTENCE = "PASS requires all three: accuracy > 0.899 AND ECE ≤ 0.059 AND axis-1 AUROC > 0.611."

#: ``(name, operator, threshold, which reported number)``. The last field is a key into the block
#: :func:`criteria` assembles, so the comparison and the printed value cannot come apart.
GATE: tuple[tuple[str, str, float, str], ...] = (
    ("accuracy", ">", 0.899, "accuracy"),
    ("ECE", "≤", 0.059, "ece_after_temperature"),
    ("axis-1 AUROC", ">", 0.611, "axis1_auroc"),
)

#: Step 11's fitted-on-calibration settings, so the pilot's triage line is the same policy. Changing
#: one would change what the reported policy *is*, which is why they are named here rather than
#: defaulted at the call site.
STEP11 = {"alpha": 0.10, "beta": 0.05, "fpr": 0.05, "method": "lac"}

#: What a row carries from its item. The five ``_meta`` keys are the ones
#: :func:`sckevtype.eval.run.extended_row` copies for stratification; ``type`` is what
#: :func:`sckevtype.eval.rows_of` filters on.
ROW_FIELDS = ("id", "group", "question", "split", "family", "variant", "source", "task", "type",
              "dataset", "cell", "donor", "panel")

#: The question types this pilot can score. laya has a third (``score``) that neither suite asks —
#: ``sckevtype.laya.render`` renders it, but the gate has no statistic for it, and silently
#: dropping a scored row would make the accuracy denominator wrong.
SCORED_TYPES = SUPPORTED_TYPES

TYPE_OF_QTYPE = {value: key for key, value in QTYPES.items()}


class ScoreError(RuntimeError):
    """An input the pilot cannot score, named rather than defaulted."""


# --------------------------------------------------------------------------- asking


def asks_for(record: Mapping[str, Any], tok: Any, cfg: RenderConfig, split: str) -> tuple[list[dict], list[dict]]:
    """One record's items: every question as put, then the abstain variant of the choice ones.

    The abstain ask is a **second item, not a second row**. :func:`sckevtype.eval.run` fills the
    variant's probability onto the row the first ask already produced, keyed by question id, and
    :func:`rows_from_answers` does the same — the two axes have to be keyed to one row to compose.
    """
    from ..eval.run import with_abstain

    built = build_record(record, tok, cfg, split)
    items = _typed(built.items, "base")
    variants = {qid: variant for qid, question in record["questions"].items()
                if (variant := with_abstain(question)) is not None}
    dropped = list(built.dropped)
    if variants:
        # One state tokenisation for the whole record's variants rather than one per variant: the
        # state is the expensive part and it does not change between them.
        extra = build_record({**record, "questions": variants}, tok, cfg, split)
        items.extend(_typed(extra.items, "abstain"))
        dropped.extend(extra.dropped)
    return items, dropped


def _typed(items: Sequence[dict], ask: str) -> list[dict]:
    """Tag each item with its ask and its question type, and pack the ids.

    ``ids`` becomes ``bytes``: a chunk of items is loaded whole, and a list of 8,192 Python ints per
    item costs ~36 bytes each against 4 in an ``array``. The item is the same sequence either way —
    :func:`_tensors` is the one place that unpacking happens.
    """
    packed = []
    for item in items:
        question_type = TYPE_OF_QTYPE.get(item["qtype"])
        if question_type not in SCORED_TYPES:
            raise ScoreError(f"{item['id']}/{item['question']}: question type {question_type!r} has no gate statistic")
        packed.append({**item, "ask": ask, "type": question_type,
                       "ids": array("i", item["ids"]).tobytes()})
    return packed


def _cmd_items(args: argparse.Namespace) -> int:
    from .render import load_tokenizer

    tok = load_tokenizer(args.tokenizer)
    # The geometry comes from the checkpoint that will answer these items, unless the caller overrides
    # it explicitly. `infer` refuses a mismatch ("scoring would ask a different question from the one
    # trained on"), and the two checkpoints here genuinely differ -- the fine-tuned pilot answers at
    # 8192/2048, the shipped laya-base at 1024/256 -- so taking the defaults for granted rendered items
    # for one model and then tried to score them with the other.
    cfg = RenderConfig(args.max_len, args.head_max_len)
    if args.checkpoint and not args.max_len_forced:
        ckpt_cfg = checkpoint_render_config(args.checkpoint)
        if (cfg.max_len, cfg.head_max_len) != (ckpt_cfg.max_len, ckpt_cfg.head_max_len):
            print(f"  geometry: rendering at the checkpoint's own {ckpt_cfg.as_dict()} "
                  f"(was {cfg.as_dict()})")
        cfg = ckpt_cfg
    out = Path(args.out)
    chunk_dir = out / "items"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "suite": args.suite, "config": cfg.as_dict(), "laya": check_laya(),
        "tokenizer": args.tokenizer, "chunk": args.chunk, "splits": {},
    }

    for split in args.splits.split(","):
        split = split.strip()
        path = partition_path(Path(args.suite), split)
        chunk: list[dict] = []
        chunks = records = questions = kept = 0
        dropped: list[dict] = []
        for record in read_records(path):
            if args.limit and records >= args.limit:
                break
            records += 1
            items, lost = asks_for(record, tok, cfg, split)
            questions += sum(item["ask"] == "base" for item in items)
            dropped.extend(lost)
            kept += len(items)
            chunk.extend(items)
            while len(chunk) >= args.chunk:
                _write_chunk(chunk_dir, split, chunks, chunk[: args.chunk], manifest)
                chunk = chunk[args.chunk:]
                chunks += 1
        if chunk:
            _write_chunk(chunk_dir, split, chunks, chunk, manifest)
            chunks += 1
        manifest["splits"][split] = {
            "records": records, "questions": questions, "items": kept, "chunks": chunks,
            "abstain_asks": kept - questions, "dropped": len(dropped), "dropped_examples": dropped[:5],
        }
        print(f"  {split:<12} records={records} questions={questions} abstain_asks={kept - questions} "
              f"items={kept} chunks={chunks} dropped={len(dropped)}")

    text = json.dumps(manifest, indent=2)
    (out / "items-manifest.json").write_text(text + "\n")
    print(f"wrote {out / 'items-manifest.json'}")
    if kept == 0:
        raise ScoreError(f"no items rendered from {args.suite}: nothing to score")
    return 0


def _write_chunk(chunk_dir: Path, split: str, index: int, items: list[dict], manifest: dict) -> None:
    import torch

    path = chunk_dir / f"{split}-{index:06d}.pt"
    torch.save(items, path)
    manifest.setdefault("chunk_files", []).append(
        {"path": path.name, "items": len(items), "bytes": path.stat().st_size}
    )


# --------------------------------------------------------------------------- answering


def load_checkpoint(path: str, device: str) -> tuple[Any, Any, dict]:
    """laya's fine-tuned checkpoint: the config that says how wide, the tokenizer, the weights.

    The config is read from the checkpoint rather than passed in, so the width the model was
    *trained* at is the width it *answers* at. A run that trained at 8192/2048 and scored at the
    shipped 1024/256 would be scored on a different prompt from the one it learned.
    """
    import json as _json

    from laya.common import build_model
    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    root = Path(path)
    for name in ("rl_agent_config.json", "model.safetensors", "encoder", "tokenizer"):
        if not (root / name).exists():
            raise ScoreError(f"{root} is not a laya checkpoint: no {name}")
    cfg = _json.loads((root / "rl_agent_config.json").read_text())
    tok = AutoTokenizer.from_pretrained(str(root / "tokenizer"))
    model = build_model(cfg, encoder_dir=str(root / "encoder"))
    model.load_state_dict(load_file(str(root / "model.safetensors")), strict=True)
    model.to(device).eval()
    return model, tok, cfg


def _tensors(items: Sequence[Mapping[str, Any]], pad_id: int) -> tuple[Any, ...]:
    """The four tensors laya's forward takes, padded to the batch, built without a Python int per token."""
    import numpy as np
    import torch

    n, length = len(items), max(len(it["ids"]) // 4 for it in items)
    width = max(len(it["markers"]) for it in items)
    ids = np.full((n, length), pad_id, dtype=np.int64)
    att = np.zeros((n, length), dtype=np.int64)
    mpos = np.zeros((n, width), dtype=np.int64)
    mmask = np.zeros((n, width), dtype=bool)
    for i, item in enumerate(items):
        row = np.frombuffer(item["ids"], dtype=np.int32)
        ids[i, : row.size] = row
        att[i, : row.size] = 1
        markers = np.asarray(item["markers"], dtype=np.int64)
        mpos[i, : markers.size] = markers
        mmask[i, : markers.size] = True
    return (torch.from_numpy(ids), torch.from_numpy(att), torch.from_numpy(mpos), torch.from_numpy(mmask))


def answer(model: Any, items: Sequence[Mapping[str, Any]], pad_id: int, device: str) -> list[dict]:
    """``p`` and the raw ``logits`` for each item, at T = 1.

    T = 1 because the checkpoint's own fitted temperatures are a fit on data this report never sees
    and, more to the point, a temperature fitted on probabilities is a temperature fitted on
    someone else's calibration — :func:`sckevtype.eval.metrics.temperature_scaling` needs the
    untempered logits to fit its own on ``calibration``. The same reason and the same call as
    :func:`sckevtype.eval.run.report`.
    """
    import torch

    ids, att, mpos, mmask = _tensors(items, pad_id)
    with torch.no_grad():
        logits, _ = model(ids.to(device), att.to(device), mpos.to(device), mmask.to(device),
                          torch.tensor([it["qtype"] for it in items], device=device))
    logits = logits.float().cpu()
    out = []
    for i, item in enumerate(items):
        width = len(item["keys"])
        z = logits[i, :width]
        p = torch.softmax(z, -1)
        out.append({"logits": [float(v) for v in z], "p": [float(v) for v in p]})
    return out


def _cmd_infer(args: argparse.Namespace) -> int:
    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, tok, cfg = load_checkpoint(args.checkpoint, device)
    item_dir, row_dir = Path(args.items), Path(args.out)
    row_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(item_dir.glob("*.pt"))
    if not files:
        raise ScoreError(f"no chunk files in {item_dir}; run the items step first")
    manifest = _read_json(item_dir.parent / "items-manifest.json")
    if manifest and manifest.get("config") != RenderConfig(**cfg_width(cfg)).as_dict():
        raise ScoreError(
            f"the items were rendered at {manifest.get('config')} but the checkpoint answers at "
            f"{cfg_width(cfg)}; scoring would ask a different question from the one trained on"
        )
    print(f"answering {len(files)} chunk(s) into {row_dir} on {device}")

    done = skipped = 0
    for path in files:
        target = row_dir / f"{path.stem}.jsonl"
        if target.exists() and not args.overwrite:
            skipped += _count_jsonl(target)
            continue
        items = torch.load(path, weights_only=False)
        rows = []
        for start in range(0, len(items), args.batch):
            part = items[start: start + args.batch]
            answers = answer(model, part, tok.pad_token_id, device)
            rows.extend(_row(item, given) for item, given in zip(part, answers))
        with target.open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        done += len(rows)
        print(f"  {path.name}: {len(rows)} rows -> {target.name}")
    print(f"answered {done} rows, skipped {skipped} already present")
    return 0


def cfg_width(cfg: Mapping[str, Any]) -> dict[str, int]:
    """The width the checkpoint answers at, as :class:`RenderConfig` keys."""
    return {"max_len": int(cfg["max_len"]), "head_max_len": int(cfg["head_max_len"])}


def checkpoint_render_config(path: str) -> RenderConfig:
    """The geometry a checkpoint answers at, read from its own ``rl_agent_config.json``.

    Read without building the model: this is what the ``items`` step needs, and it must be the same
    source ``load_checkpoint`` uses, or the render and the answering disagree.
    """
    import json as _json

    root = Path(path)
    config_path = root / "rl_agent_config.json"
    if not config_path.exists():
        raise ScoreError(f"{root} is not a laya checkpoint: no rl_agent_config.json (needed for the "
                         "geometry these items must be rendered at)")
    cfg = _json.loads(config_path.read_text())
    width = cfg_width(cfg)
    return RenderConfig(width["max_len"], width["head_max_len"])


def _row(item: Mapping[str, Any], given: Mapping[str, Any]) -> dict[str, Any]:
    """The answer to one item: what was asked, and what came back."""
    return {**{name: item[name] for name in ROW_FIELDS},
            "ask": item["ask"], "keys": list(item["keys"]), "label": int(item["label"]),
            "qtype": int(item["qtype"]), "p": given["p"], "logits": given["logits"]}


# --------------------------------------------------------------------------- scoring


def rows_from_answers(answers: Iterable[Mapping[str, Any]]) -> tuple[list[dict], list[dict]]:
    """laya's per-item answers as the rows :mod:`sckevtype.eval` scores.

    One row per base ask, carrying ``none_p`` — the probability of the abstain option — because
    axis 1 is a property of the question, not a second question. Where the question already offered
    an abstain option, its probability is read off the base answer and ``abstain`` is that option's
    index; where it did not, ``none_p`` comes from the abstain ask and ``abstain`` is ``None``.
    That is :func:`sckevtype.eval.run.score_partition` and
    :func:`~sckevtype.eval.run.score_abstain_variant`, reproduced field for field.

    A choice row whose abstain ask was never scored is **returned in the rejected list**, not
    emitted with ``none_p: None``: :func:`sckevtype.eval.metrics.panel_adequacy` drops such rows
    silently, and a pooled AUROC computed on whichever rows happened to fit is a silent sample.
    """
    from ..eval import abstain_index, abstain_key

    base = [row for row in answers if row["ask"] == "base"]
    variants = {(row["id"], row["question"]): row for row in answers if row["ask"] == "abstain"}
    rows: list[dict] = []
    rejected: list[dict] = []
    for answer in base:
        row = {name: answer[name] for name in ROW_FIELDS}
        row.update(keys=list(answer["keys"]), label=int(answer["label"]), p=list(answer["p"]),
                   logits=list(answer["logits"]), split=answer["split"])
        row["size"] = len(row["keys"])
        row["abstain"] = abstain_index(row)
        key = abstain_key(row["keys"])
        row["none_p"] = None if key is None else float(row["p"][list(row["keys"]).index(key)])
        if row["none_p"] is None and row["type"] == "choice":
            variant = variants.get((row["id"], row["question"]))
            if variant is None:
                rejected.append({"id": row["id"], "question": row["question"], "split": row["split"],
                                 "error": "no abstain variant was scored, so axis 1 has no statistic"})
                continue
            variant_key = abstain_key(variant["keys"])
            row["abstain"] = None
            row["none_p"] = float(variant["p"][list(variant["keys"]).index(variant_key)])
        rows.append(row)
    return rows, rejected


def criteria(metrics: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The three gate criteria, each as ``(name, operator, threshold, observed, passed)``.

    ``observed`` is looked up by key from the assembled metrics, so the number printed beside a
    criterion is the number that criterion compared.
    """
    import operator

    operations = {">": operator.gt, "≤": operator.le}
    out = []
    for name, op, threshold, key in GATE:
        if key not in metrics:
            raise ScoreError(f"the gate names {key!r} but the run produced no such number; the gate cannot be reported")
        out.append({"name": name, "operator": op, "threshold": threshold,
                    "observed": float(metrics[key]), "passed": bool(operations[op](metrics[key], threshold))})
    return out


def verdict(entries: Sequence[Mapping[str, Any]]) -> str:
    """``PASS`` only when every criterion passed. There is no partial credit and no tie-break."""
    return "PASS" if all(entry["passed"] for entry in entries) else "FAIL"


def gate(rows: Sequence[Mapping[str, Any]], fpr: float) -> dict[str, Any]:
    """Every number the gate reads, on the partition the gate is defined on.

    The temperature and the triage policy are fitted on ``calibration``; accuracy, ECE and axis 1
    are reported on ``test``. The fit is inside the guard, so a run that handed the fit a test row
    raises here rather than quietly reporting an in-sample number.
    """
    from ..eval import CALIBRATION, TEST, assert_fit_partition, metrics, rows_of, triage

    calibration = [row for row in rows if row["split"] == CALIBRATION]
    test = [row for row in rows if row["split"] == TEST]
    if not calibration or not test:
        raise ScoreError(f"need both partitions: {len(calibration)} calibration, {len(test)} test rows")
    assert_fit_partition(calibration, "sckevtype.laya.score")

    temperature = metrics.temperature_scaling(calibration, test)
    axis1 = metrics.panel_adequacy(rows_of(test, type="choice"), fpr)
    policy = triage.fit(calibration, STEP11["alpha"], STEP11["beta"], STEP11["method"])
    return {
        "partitions": {"fitted_on": CALIBRATION, "reported_on": TEST,
                       "calibration_rows": len(calibration), "test_rows": len(test)},
        "accuracy": metrics.top1_accuracy(test)["accuracy"],
        "ece_before_temperature": temperature["before"]["ece"],
        "ece_after_temperature": temperature["after"]["ece"],
        "temperature": temperature["temperature"],
        "axis1": axis1,
        "axis1_auroc": axis1["auroc"],
        "triage": triage.summary(policy, rows_of(test, type="choice")),
        "breakdowns": {key: metrics.breakdown(test, key) for key in ("family", "type", "size", "dataset")},
        "per_stratum": metrics.per_stratum(test),
        "settings": dict(STEP11, fpr=fpr),
    }


def _cmd_evaluate(args: argparse.Namespace) -> int:
    answers = list(read_answers(Path(args.rows)))
    if not answers:
        raise ScoreError(f"no answers under {args.rows}")
    rows, rejected = rows_from_answers(answers)
    result = gate(rows, args.fpr)
    result["rows"] = {"answered": len(answers), "scored": len(rows), "rejected": rejected}
    gate_criteria = criteria(result)
    result["gate"] = {"sentence": GATE_SENTENCE, "criteria": gate_criteria, "verdict": verdict(gate_criteria)}
    write_json(Path(args.out), result)
    print("\n".join(_lines(result)))
    print(f"wrote {args.out}")
    return 0 if result["gate"]["verdict"] == "PASS" else 1


def _lines(result: Mapping[str, Any]) -> list[str]:
    """The report, ending in the verbatim sentence and one explicit verdict."""
    partitions = result["partitions"]
    lines = [
        f"sckevtype.laya.score — fitted on {partitions['fitted_on']}, reported on {partitions['reported_on']}",
        (f"  {partitions['test_rows']} test questions, {partitions['calibration_rows']} calibration; "
         f"{result['rows']['scored']} rows scored, {len(result['rows']['rejected'])} rejected"),
        (f"  accuracy {result['accuracy']:.4f}; ece {result['ece_before_temperature']:.4f} -> "
         f"{result['ece_after_temperature']:.4f} at T={result['temperature']:.3f}"),
        (f"  axis-1 auroc {result['axis1_auroc']:.4f} over {result['axis1']['n']} choice rows "
         f"({result['axis1']['n_abstain_targets']} with an abstain target)"),
        (f"  triage alpha={result['triage']['policy']['alpha']} beta={result['triage']['policy']['beta']}: "
         f"accept {result['triage']['coverage']:.3f} at risk {result['triage']['risk']}"),
        "  axis-1 by family (context, not a substitute for the gate):",
    ]
    for family, entry in sorted(result["axis1"]["per_family"].items()):
        auroc = entry.get("auroc")
        # `entry['note']` is present-but-None on a family that carries one axis-1 class only, so
        # `'note' in entry` is not the guard: it printed `'  (' + None + ')'` and killed the
        # evaluator on the base checkpoint's pass (job 4207298) after it had written its report.
        lines.append(f"    {family:<20} n={entry['n']:<7} auroc "
                     f"{'n/a' if auroc is None else f'{auroc:.4f}'}"
                     f"{'  (' + entry['note'] + ')' if entry.get('note') else ''}")
    lines.append("")
    for criterion in result["gate"]["criteria"]:
        verdict = "PASS" if criterion["passed"] else "FAIL"
        lines.append(f"  {criterion['name']:<14} {criterion['observed']:.4f} "
                     f"{criterion['operator']} {criterion['threshold']:<7} {verdict}")
    lines.append(GATE_SENTENCE)
    lines.append(f"laya pilot gate: {result['gate']['verdict']}")
    return lines


# --------------------------------------------------------------------------- reading


def read_answers(path: Path) -> Iterator[dict[str, Any]]:
    """Every answer under ``path``, whether it is one file or a directory of chunk files."""
    if path.is_dir():
        files = sorted(path.glob("*.jsonl"))
        if not files:
            raise ScoreError(f"no *.jsonl under {path}")
        for file in files:
            yield from _read_jsonl(file)
    else:
        yield from _read_jsonl(path)


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _count_jsonl(path: Path) -> int:
    """How many rows a dump file holds — the resume path's ``skipped`` count.

    ``_read_jsonl`` is a generator, so ``len()`` on it raises ``TypeError``; that only ever fired when
    a chunk had already been answered, which is why the first full pass never met it and the first
    *re-run* into a populated rows directory died before answering anything.
    """
    return sum(1 for _ in _read_jsonl(path))


def _read_json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text()) if path.exists() else None


# --------------------------------------------------------------------------- cli


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="sckevtype.laya.score",
                                     description="Score fine-tuned laya on step 11's instrument.")
    sub = parser.add_subparsers(dest="command", required=True)

    items = sub.add_parser("items", help="re-render a suite into laya asks")
    items.add_argument("--suite", required=True, help="frozen suite directory")
    items.add_argument("--out", required=True, help="directory to write items/ and items-manifest.json into")
    items.add_argument("--splits", default="calibration,test")
    items.add_argument("--max-len", type=int, default=WIDE.max_len)
    items.add_argument("--head-max-len", type=int, default=WIDE.head_max_len)
    items.add_argument("--checkpoint", default=None,
                       help="the checkpoint these items are for; its own geometry is used unless "
                            "--max-len/--head-max-len are given explicitly")
    items.add_argument("--max-len-forced", action="store_true",
                       help="use --max-len/--head-max-len even when --checkpoint says otherwise")
    items.add_argument("--tokenizer", default=None, help="tokenizer directory (default: mmBERT-base)")
    items.add_argument("--chunk", type=int, default=1000, help="items per chunk file")
    items.add_argument("--limit", type=int, default=0, help="stop after N records per split (0 = all)")
    items.set_defaults(func=_cmd_items)

    infer = sub.add_parser("infer", help="answer the items with a checkpoint (GPU; resumable)")
    infer.add_argument("--items", required=True, help="the items/ directory the items step wrote")
    infer.add_argument("--checkpoint", required=True, help="fine-tuned laya checkpoint directory")
    infer.add_argument("--out", required=True, help="directory to write one rows file per chunk into")
    infer.add_argument("--batch", type=int, default=4, help="items per forward pass")
    infer.add_argument("--device", choices=["cpu", "cuda", "mps"], default=None)
    infer.add_argument("--overwrite", action="store_true", help="re-answer chunks whose rows exist")
    infer.set_defaults(func=_cmd_infer)

    evaluate = sub.add_parser("evaluate", help="step 11's metrics on those answers, and the gate")
    evaluate.add_argument("--rows", required=True, help="a rows file, or the directory of them")
    evaluate.add_argument("--out", required=True, help="report path (.json)")
    evaluate.add_argument("--fpr", type=float, default=STEP11["fpr"],
                          help="target false-positive rate for the axis-1 detection rate")
    evaluate.set_defaults(func=_cmd_evaluate)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return args.func(args)
    except (ScoreError, RenderError, ValueError, KeyError, FileNotFoundError, ImportError) as error:
        print(f"laya scoring failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
