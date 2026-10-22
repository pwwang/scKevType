"""The training CLI: a suite in, a LoRA adapter + pointer head out, with provenance.

    python -m sckevtype.train.train --suite artifacts/suites/smoke \\
        --base Qwen/Qwen3.5-0.8B-Base --base_revision <sha> \\
        --steps 30 --batch 2 --dtype bf16 --checkpointing 1 --device cuda --out runs/smoke-0.8b

The flags mirror ``kev.train``'s, because the two things this step must not do
are fork the trainer and restate its defaults. So the run's hyperparameters —
the LoRA recipe, the augmentation rates, the loss weights, the context limits —
are ``kev.train``'s own argparse defaults, read from its parser and overridden
only where this CLI exposes a flag. ``--steps`` and ``--seed`` are the additions:
``--steps`` bounds a run to a number of optimizer steps, which is what makes a
smoke run a *bounded* foreground command rather than an epoch-length one.
``--init_from`` is kev's own delta flag and goes through kev's own warm start
(``kev.checkpoint.Checkpoint.warm_start``) rather than a copy of its checks, so a
delta run is the same object whichever trainer produced it.

The state budget is the one context knob this CLI does expose, and it always
hands kev a number: the suite manifest's measured ``context.max_state`` unless
``--max_state`` overrides it, verified against the training partition's own
states before the model is loaded. kev's ``MAX_STATE`` default is never reachable
from here — a suite that records no budget is refused rather than trained.

Multi-GPU
---------
``--gpus N`` shards the *frozen* backbone over N cards (``sckevtype.train.shard``)
and leaves the LoRA adapter, the pointer head and the optimizer's state on
``cuda:0``. A LoRA-only delta has no optimizer state for the backbone, so this is a
placement rather than a data-parallel run: the weights spread, the trainable bytes
do not. The placement is recorded in ``training_config.json`` (the device list and
the per-device parameter bytes) and its per-device peaks in ``provenance.json``'s
epochs; a map that offloads to cpu/disk, splits a decoder layer, or uses fewer
cards than were asked for is refused by name before the first step. ``--gpus 1`` is
today's single-card path, byte for byte.

Token budget
------------
``--token_budget N`` bounds *one forward pass* by the tokens the row form actually pays —
``rows x longest row``, the padded shape ``kev.model._rows_hidden`` builds, which is what the
activations are sized by (the count-budgeted path bounds the pass by records instead, so a step
that draws long records does not fit). It exists because the corpus is variable-length by more
than an order of magnitude per record while ``--batch`` is a *count*: at ``--batch 1`` a pass is
one record, and the longest records in this suite are ~69,000 row tokens, enough to OOM a 9B on a
40 GB card whatever the batch is.

The *optimizer step* is untouched: the epoch is still cut into steps of ``batch x accum`` records
(:func:`groups_of`), the same records land in the same steps for a given seed, and the LR
schedule's step count is the same, so a token-budgeted run stays comparable with a count-budgeted
one. What the budget decides is how many forward/backward passes those records take: a record that
fits is one pass (the count-budgeted path's pass, byte for byte), and one that does not is cut
*between* its questions — never inside one — with each part weighted
(``questions in the part / questions in the record``) so its loss contributes exactly what the
whole record's did. Cutting between rows is not cutting the context: a row is state + one question
and the rows are independent by construction (``kev.model.rows_of``; ``kev.model.rows_per_pass``
splits the same way at serving time and says so).

Checkpoints and resume
-----------------------
A delta epoch is hours long and this fleet reboots nodes, so the trainer writes a *resumable*
checkpoint into the run directory every ``--checkpoint_every`` optimizer steps (default 200, or the
``SCKEVTYPE_CHECKPOINT_EVERY`` environment variable; 0 = never), and ``--resume`` continues from the
newest one instead of starting over. Without ``--resume`` a run neither reads nor needs one, and
nothing about it changes.

A checkpoint is ``checkpoint-step-<n>.pt``: the adapter tensors and the pointer head (the same two
things a released checkpoint carries), the optimizer and OneCycleLR states, the step/epoch/step-in-
epoch cursor, the epoch's record order, the measured peak, the attempt count, the recipe the run was
launched with, and the torch/cuda/python/numpy RNG states *at the step boundary*. The generators are
what make a continuation identical rather than merely similar: the resumed process restores the state
the next step would have been drawn from, so the loss of step ``n+1`` is the loss the killed attempt
would have produced, bit for bit — same recipe, same card count (a different card count is a
different per-layer LoRA-dropout draw, which this lane already reports and does not assert on).
Only the latest two are kept, the write is atomic (tmp file + rename), and the older one is the
fallback for a file the crash itself truncated. The final save at the end of a run is unchanged, and
``run.log.jsonl`` keeps one row per step: a resume drops the rows it is about to re-run, so the log
reads as one run rather than as two attempts interleaved.

Three files are written next to the adapter (plus the checkpoints above, while the run lasts):

* ``training_config.json`` — every hyperparameter the run used, the base revision,
  and the sha256 of the suite manifest;
* ``provenance.json`` — what the run was, so it can be reproduced: the commits of
  this repo and of the pinned ``kev``, the suite and corpus digests, the base
  revision and its shape, the device, the library versions, the *names* of the
  trainable tensors, and the measured tokens/s and peak device bytes per epoch —
  plus, for a delta run, the checkpoint it was warm-started from;
* ``run.log.jsonl`` — one JSON object per optimizer step, so a run can be
  inspected without parsing stdout.

The checkpoint is written with ``kev.checkpoint``'s own writer, so a run started
here is loadable by ``kev``'s serving and evaluation paths without conversion.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import re
import resource
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..contracts import budget as state_budget
from ..contracts import paths
from ..contracts.freeze import file_digest, git_commit, read_json, write_json
from . import model as model_module
from . import shard as shard_module

#: The flags that decide the *model*, as opposed to the optimisation: they are the
#: ones ``build_model`` needs, and the ones ``training_config.json`` records so it
#: describes the architecture the run actually built rather than the default one.
MODEL_SETTINGS: tuple[str, ...] = (
    "lora", "head_dim", "lora_targets", "option_isolation", "special_embeddings", "weights_dtype",
)

#: The modules whose bytes identify this step's code in ``provenance.json``.
_SOURCE_FILES = ("model.py", "suite.py", "train.py", "compute.py")

#: Steps between resumable checkpoints when nothing asks for another number. Chosen against the
#: measured per-step cost of this lane's runs (~20 s/step on the 4B shard): 200 steps is about an
#: hour of work at risk, against a whole epoch (7.6 h) before this existed.
DEFAULT_CHECKPOINT_EVERY = 200

#: How many checkpoints a run keeps. Two, not one: the crash this exists for can truncate the
#: newest file, and the one before it is then still readable — a save that *is* atomic costs the
#: older file only until the next checkpoint prunes it.
RETAINED_CHECKPOINTS = 2

#: The environment variable that overrides :data:`DEFAULT_CHECKPOINT_EVERY`, so a lane can change the
#: interval without editing the command line it submits (``multigpu-delta-task.sh`` passes the same
#: number as ``--checkpoint_every``).
ENV_CHECKPOINT_EVERY = "SCKEVTYPE_CHECKPOINT_EVERY"

#: ``checkpoint-step-<n>.pt``, where ``<n>`` is the number of optimizer steps completed.
CHECKPOINT_PREFIX = "checkpoint-step-"
CHECKPOINT_FORMAT = "sckevtype.train.resume.v1"

#: What a resume reads out of a checkpoint. Checked by name, so a file that is not one of ours (or
#: one written by a later format) is refused rather than half-read.
CHECKPOINT_KEYS = ("format", "step", "epoch", "step_in_epoch", "records_seen", "attempt",
                   "peak_device_bytes", "epoch_records", "epoch_tokens", "passes",
                   "order", "recipe", "adapter", "head", "optimizer", "scheduler", "rng")

#: Recipe fields a resume reports rather than refuses. ``gpus`` decides where each layer's LoRA
#: dropout is drawn from, so a different card count is a different draw of the same recipe — the
#: distinction the sharded lane already reports and does not assert on. ``init_from`` cannot change
#: the trajectory at all: the checkpoint's own adapter supersedes whatever warm start the command
#: line names, and it is reported so a reader sees the difference.
RESUME_NOTED_FIELDS = ("gpus", "init_from")


def checkpoint_every_default(env: Mapping[str, str] | None = None) -> str:
    """The default ``--checkpoint_every``: the environment's, else :data:`DEFAULT_CHECKPOINT_EVERY`.

    Returned as a string on purpose — ``argparse`` applies the flag's ``type`` to a string default,
    so a bad ``SCKEVTYPE_CHECKPOINT_EVERY`` is reported the way any other bad argument is, naming
    the flag, instead of crashing the CLI before its parser has said anything.
    """
    return (env if env is not None else os.environ).get(ENV_CHECKPOINT_EVERY) or str(DEFAULT_CHECKPOINT_EVERY)



# --------------------------------------------------------------------------- arguments


@contextlib.contextmanager
def _argv(argv: list[str]) -> Iterator[None]:
    """Run a block with ``sys.argv`` replaced (``kev.train.parse_args`` reads it directly)."""
    original = sys.argv
    sys.argv = argv
    try:
        yield
    finally:
        sys.argv = original


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """This CLI's flags: the mirrored subset, plus ``--steps``."""

    parser = argparse.ArgumentParser(prog="sckevtype.train.train", description="Train the scKevType decision model on a suite.")
    parser.add_argument("--suite", required=True, help="suite directory written by sckevtype.train.suite")
    parser.add_argument("--base", required=True, help="base model id, e.g. Qwen/Qwen3.5-0.8B-Base")
    parser.add_argument("--base_revision", default="", help="base commit; required unless the suite pins this base")
    parser.add_argument("--init_from", default="",
                        help="delta fine-tune: warm-start the LoRA adapter and pointer head from an existing run "
                             "(a local run directory, or a hub id like jaredpalmer/kev-0.8b optionally @revision), "
                             "instead of starting from the base model alone")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=0,
                        help="run for this many optimizer steps, raising --epochs if one epoch is shorter (0 = run --epochs epochs)")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--head_lr", type=float, default=0.0, help="separate learning rate for the pointer head (0 = --lr)")
    parser.add_argument("--batch", type=int, default=1, help="records per forward pass")
    parser.add_argument("--accum", type=int, default=8, help="micro-batches per optimizer step")
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--weights_dtype", choices=["fp32", "bf16"], default=None,
                        help="the frozen backbone's dtype (default: kev's own, fp32). It decides whether "
                             "a size fits at all: at fp32 a 4B base is 16 GB and a 9B base 36 GB of "
                             "weights on a 40 GB card, before a single activation")
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default=None)
    parser.add_argument("--gpus", type=int, default=1,
                        help="cards to shard the frozen backbone over (1 = the single-card path, unchanged). "
                             "A LoRA-only delta trains the adapter and the head, so >1 is a placement and not "
                             "data parallelism: it needs --device cuda, and it is what makes a 4B/9B delta fit "
                             "a 40 GB card (the single-card runs die at 36.4 GB allocated)")
    parser.add_argument("--checkpointing", type=int, choices=[0, 1], default=0)
    parser.add_argument("--profile_memory", type=int, choices=[0, 1], default=0,
                        help="print the peak device memory after each phase of every microbatch, so a peak has a "
                             "location rather than a total. Off by default: it reads and clears the peak counters "
                             "the epoch's own measurement uses, which costs nothing but is not a training knob")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--p_none_pair", type=float, default=0.0,
                        help="fraction of choice records that also emit kev's none-present/none-absent minimal pair. "
                             "kev's default and arm B's value are both 0.0: the pair was measured on this corpus and "
                             "rejected, because it duplicates the source row on every record that already answers the "
                             "abstain option, and the flag is reachable so that measurement can be repeated. "
                             "kev's own bound is [0, 1]")
    parser.add_argument("--max_state", type=int, default=0,
                        help="state tokens per record (0 = the budget the suite manifest records; never kev's default)")
    parser.add_argument("--token_budget", type=int, default=0,
                        help="padded tokens one forward pass may cost (rows x longest row; 0 = off, one pass per "
                             "--batch records as before). The optimizer step is unchanged -- still batch x accum "
                             "records, accumulated over however many passes that takes -- so a run keeps the "
                             "effective batch size (and the LR schedule) of the count-budgeted path; only the "
                             "grouping into passes changes. A record over the budget is cut between its questions, "
                             "each part weighted so it contributes what the whole record did. It bounds a 9B on 4x "
                             "A100-40G, whose peak is set by the longest record the shuffle draws (~69,000 row "
                             "tokens here) and not by the batch size")
    parser.add_argument("--checkpoint_every", type=int, default=checkpoint_every_default(),
                        help="write a resumable checkpoint into --out every N optimizer steps (default 200, or the "
                             "SCKEVTYPE_CHECKPOINT_EVERY environment variable; 0 = never). Only the latest "
                             f"{RETAINED_CHECKPOINTS} are kept, and a run's own final save is unchanged")
    parser.add_argument("--resume", action="store_true",
                        help="continue the run in --out from its newest checkpoint-step-<n>.pt instead of starting "
                             "over. The directory must hold one (that is the whole precondition), the recipe must "
                             "be the one the checkpoint records, and the step/loss sequence continues rather than "
                             "restarting. Without this flag a populated directory is still refused")
    parser.add_argument("--out", required=True, help="run directory; it must not exist yet (with --resume: it must, "
                                                     "and must hold a checkpoint)")
    args = parser.parse_args(argv)
    if min(args.epochs, args.batch, args.accum) < 1 or args.steps < 0 or args.max_state < 0:
        parser.error("epochs, batch and accum must be positive; steps and max_state must not be negative")
    if args.checkpoint_every < 0:
        parser.error("--checkpoint_every is a step count: 0 turns it off, a negative one is a typo")
    if args.token_budget < 0:
        parser.error("--token_budget is a token ceiling: 0 turns it off, a negative one is a typo")
    if args.gpus < 1:
        parser.error("--gpus must be at least 1 (1 = the single-card path)")
    if args.gpus > 1 and args.device != "cuda":
        parser.error(f"--gpus {args.gpus} needs --device cuda: there is nothing to shard a model over on "
                     f"{args.device or 'the default device'}")
    if not 0.0 <= args.p_none_pair <= 1.0:
        parser.error("p_none_pair is a probability: kev's own bound is [0, 1]")
    if args.dtype == "bf16" and args.device != "cuda":
        parser.error("--dtype bf16 requires --device cuda")
    return args


def suite_budget(suite_dir: str | Path) -> dict[str, Any]:
    """The ``context`` block of a suite's manifest — the budget it was measured to.

    Read from the manifest and never defaulted: kev's own ``MAX_STATE`` (384) is
    what the first real run silently trained against, on a suite whose states
    cost ~1,574 tokens, and it died on its first batch after a cold GPU boot. A
    suite that records no budget is an error here rather than a fallback.
    """
    path = Path(suite_dir) / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"no suite manifest at {path}: a run needs the budget the suite was measured to")
    context = read_json(path).get("context")
    if not context or "max_state" not in context:
        raise ValueError(f"{path} records no context.max_state; refusing to fall back to kev's default budget")
    return context


def recorded_max_state(context: dict[str, Any]) -> int:
    """The one number kev must be handed for the suite recording this ``context``.

    ``max_state``, which under the three-limit rule *is* the number kev's trainer
    needs: the manifest derives it from the state, row and packed observations at
    once, so there is no second value a run could disagree with. Suites written
    before that rule record the state budget in ``max_state`` and the trainer
    budget in ``train_max_state``; those are honoured so an old suite keeps
    meaning what it meant. Every caller that sizes a run or a measurement reads
    the budget through here, so no two of them can disagree.
    """
    return context.get("train_max_state") or context["max_state"]


def verify_recorded_budget(context: dict[str, Any], suite_dir: str | Path) -> int:
    """The recorded budget must satisfy all three of kev's limits. Returns it.

    ``context.observed`` is the measurement the budget was derived from, so the three
    inequalities can be re-checked here without re-tokenizing anything. A suite that
    records only the state axis — or one whose records have moved under the budget it
    recorded — stops in a second, naming the axis that is short, instead of one batch
    into a run that has already booted a GPU.
    """
    from kev.model import training_context

    budget = recorded_max_state(context)
    observed = state_budget.observed_limits(context.get("observed") or {})
    if observed is None:
        return budget  # nothing recorded to re-check; the measurement that follows still runs
    required = state_budget.required_budget(**observed)
    if required > budget:
        terms = state_budget.budget_terms(**observed)
        axis = max(terms, key=terms.__getitem__)
        limits = training_context(budget)
        admitted = {"state": limits["max_state"], "row": limits["max_branch"], "packed": limits["max_packed"]}
        raise ValueError(
            f"{suite_dir} records max_state {budget}, which cannot hold the records it measured: the longest "
            f"{axis} is {observed['max_' + axis]} tokens against the {admitted[axis]} limit kev derives from "
            f"{budget}. It needs max_state {required}. Rebuild the suite."
        )
    return budget


def verify_state_budget(tokenizer: Any, records: list[dict[str, Any]], limits: dict[str, int],
                        suite_dir: str | Path) -> dict[str, int]:
    """Every training record must fit the context this run will hand kev. Returns the state distribution.

    kev's trainer does not filter a frozen suite's records — it encodes them
    strictly — so a suite whose budget is too small fails on the first batch that
    contains an over-long record, which on the cluster was after the GPU had
    booted. This measures the whole training partition first (seconds, on CPU)
    and stops with the counts a rebuild needs instead.

    All three of kev's limits are checked, not just the state: ``kev.model.encode``
    also refuses a branch longer than ``max_branch - len(state)`` and
    ``kev.train.encode_batch`` a record longer than ``max_packed``, and both are
    derived from ``--max_state``. A state that fits while its questions do not is
    the same failure one axis over.

    ``limits`` is the context kev itself will compute (``kev.model.training_context``
    of the value being passed), and the tokenizer is the run's own (``--base``),
    not the one the suite was measured with: whatever the run will encode with is
    what the budget has to hold.
    """
    measured = state_budget.state_lengths(tokenizer, (record["state"] for record in records))
    summary = state_budget.summarize(measured)
    required = state_budget.budget_for(summary["max"])
    over = state_budget.over_budget(measured, limits["max_state"])
    if over:
        raise ValueError(
            f"{over} of {len(records)} training records exceed the {limits['max_state']}-token budget "
            f"{suite_dir} records: longest state {summary['max']} tokens, required budget {required}. "
            f"Rebuild the suite with --max_state {required} (or --truncate), or pass --max_state {required}."
        )
    rows, packed = state_budget.encoded_lengths(records, tokenizer, max_state=limits["max_state"])
    for axis, values, limit in (("row", rows, limits["max_branch"]), ("packed", packed, limits["max_packed"])):
        over = state_budget.over_budget(values, limit)
        if over:
            raise ValueError(
                f"{over} of {len(records)} training records exceed the {limit}-token {axis} limit {suite_dir} "
                f"records: longest {axis} {max(values)} tokens, required {axis} budget "
                f"{state_budget.budget_for(max(values))}. kev derives that limit from --max_state alone, so rebuild "
                f"the suite (its context records the value that covers it) or pass a larger --max_state."
            )
    return summary


def kev_namespace(args: argparse.Namespace, out: str) -> Any:
    """``kev.train``'s own namespace for this run: its defaults, our overrides, its validation.

    The flags this CLI mirrors are handed back to kev's parser, so everything
    this CLI does *not* expose still arrives at the trainer with kev's default
    and kev's checks (bf16-needs-cuda, positive sizes, an ``--out`` that does not
    exist yet) still run.

    ``--max_state`` is always passed explicitly: from ``--max_state`` when the
    operator pinned one, otherwise from the suite manifest. Leaving it off is
    what let kev silently train a 1,574-token suite against its 384-token
    default, so there is no path through this function that does not name a
    budget.

    The number handed over comes from :func:`recorded_max_state`: the budget the
    manifest derived from all three of kev's limits, which is what kev needs,
    since it derives its state, row and packed limits from this one flag.
    """
    import kev.train

    max_state = args.max_state or recorded_max_state(suite_budget(args.suite))
    argv = [
        "sckevtype.train.train",
        "--suite", args.suite,
        "--base", args.base,
        "--epochs", str(args.epochs),
        "--lr", str(args.lr),
        "--head_lr", str(args.head_lr),
        "--batch", str(args.batch),
        "--accum", str(args.accum),
        "--dtype", args.dtype,
        "--checkpointing", str(args.checkpointing),
        "--seed", str(args.seed),
        "--p_none_pair", str(args.p_none_pair),
        "--max_state", str(max_state),
    ]
    if args.base_revision:
        argv += ["--base_revision", args.base_revision]
    if args.device:
        argv += ["--device", args.device]
    if args.weights_dtype:
        argv += ["--weights_dtype", args.weights_dtype]
    # ``--out``: kev refuses one that already exists, and a resume's run directory has to exist —
    # so a resume hands its parser a path that does not exist yet (a fresh temporary directory's own
    # child) and puts this run's directory back into the namespace afterwards. Only that one check is
    # bypassed; every other check kev's parser makes still runs, and the resume's own precondition (a
    # readable checkpoint whose recipe matches) is checked before a single weight is loaded. The value
    # restored below is what `training_config.json` and `provenance.json` record.
    placeholder: Any = contextlib.nullcontext(out)
    if args.resume:
        placeholder = tempfile.TemporaryDirectory(prefix="sckevtype-resume-")
    with placeholder as holder:
        argv += ["--out", str(Path(holder) / "out") if args.resume else out]
        with _argv(argv):
            ns = kev.train.parse_args()
    if args.resume:
        ns.out = str(out)
    return ns


# --------------------------------------------------------------------------- measurement


def forward_tokens(model: Any, enc: dict) -> int:
    """The tokens the backbone actually sees for one encoded record.

    Not ``len(enc["ids"])``: a hybrid backbone (Qwen3.5's Gated DeltaNet) cannot
    honour the packed block-causal mask, so kev runs each question as its own
    causal row continuing from the state and the state is recomputed per row. The
    packed length undercounts that by a factor of the question count, and a
    tokens/s figure built on it would be wrong in the flattering direction.
    """
    if not model.rows_form([enc]):
        return len(enc["ids"])
    from kev.model import rows_of

    state, _, rows = rows_of(enc)
    return sum(len(state) + len(row["ids"]) for row in rows)


def batch_shape(model: Any, batch: list) -> tuple[int, int, int]:
    """``(rows, padded slots, longest row)`` for an encoded batch: what the backbone actually runs.

    ``forward_tokens`` counts ``sum(len(state) + len(row))`` over the questions. The row form
    does not run that sum: ``_rows_hidden`` puts *every* question of the batch into one
    right-padded batch of length ``L_max``, so the kernel's token count is ``rows x L_max`` and
    the padding is real work. The two agree only when every row is the same length — which is
    true of a single-question record and false of most of this corpus. ``L_max`` is reported
    separately because it, and not the token count, is what the attention layers are quadratic in.
    """
    if not model.rows_form([variant.enc for variant in batch]):
        lengths = [len(variant.enc["ids"]) for variant in batch]
        return len(batch), sum(lengths), max(lengths, default=0)
    from kev.model import rows_of

    lengths = [len(state) + len(row["ids"])
               for variant in batch
               for state, _, rows in [rows_of(variant.enc)]
               for row in rows]
    return len(lengths), len(lengths) * max(lengths, default=0), max(lengths, default=0)


class PhaseMemory:
    """Peak device memory per phase, printed where it is read.

    ``kev.device.allocated_bytes`` is ``max_memory_allocated`` — a running peak since the last
    reset — so a per-phase reading only means anything if the counter is cleared between phases.
    Clearing it would hide the epoch's own peak from the loop that records it, so the largest
    reading seen is kept here and folded back into that number.
    """

    def __init__(self, device: str, enabled: bool) -> None:
        self.enabled = bool(enabled) and device == "cuda"
        self.peak = 0

    def reset(self) -> None:
        self.peak = 0

    def __call__(self, phase: str, **detail: Any) -> None:
        if not self.enabled:
            return
        import torch

        allocated = torch.cuda.max_memory_allocated()
        reserved = torch.cuda.max_memory_reserved()
        self.peak = max(self.peak, allocated)
        torch.cuda.reset_peak_memory_stats()
        shown = "  ".join(f"{key}={value}" for key, value in detail.items())
        print(f"    [mem] {phase:<10s} peak_allocated {allocated / 2**30:7.3f} GiB "
              f"peak_reserved {reserved / 2**30:7.3f} GiB  {shown}", flush=True)


# --------------------------------------------------------------------------- microbatches


@dataclass
class Forward:
    """One forward + backward pass, and how its loss enters its optimizer step's sum.

    ``divisor`` is kev's accumulation weight: the source records in the step times the variants this
    pass carries per record — ``kev.train.accumulation_records``' factor, which keeps an optimizer
    step a mean over its *records* rather than over its variants. ``weight`` is this pass's share of
    its parent record's loss: 1 for a whole record (which is every pass on the count-budgeted path)
    and ``questions in the part / questions in the record`` for a part of one, so a cut record
    contributes exactly what an uncut record did.
    """

    batch: list
    divisor: float
    weight: float = 1.0


@dataclass
class Step:
    """One optimizer step: the passes it runs, and the source records they cover (its effective batch)."""

    forwards: list[Forward]
    records: int


def groups_of(n: int, batch: int, accum: int) -> list[tuple[int, int]]:
    """The record ranges an epoch's optimizer steps cover: ``batch x accum`` records each, the last one short.

    This is the invariant a token budget must not touch, so both paths cut the epoch with this one
    function: a step is ``batch`` records per accumulation slot times ``accum`` slots, which is what
    fixes the effective batch size and the number of LR-schedule steps
    (``ceil(ceil(n / batch) / accum)``, the same count by ``ceil(n / (batch * accum))``).
    """
    size = batch * accum
    return [(start, min(start + size, n)) for start in range(0, n, size)]


def part_cost(row_lengths: Sequence[int], first: int, last: int) -> int:
    """What a pass over rows ``[first, last)`` costs: rows x longest row, the shape the row form builds."""
    runs = row_lengths[first:last]
    return len(runs) * max(runs, default=0)


def plan_parts(row_lengths: Sequence[int], budget: int) -> list[tuple[int, int]]:
    """Cut one record's rows into consecutive parts costing at most ``budget`` padded tokens each.

    A pass pays ``rows x longest row`` — every row of the batch is right-padded to the longest one
    (``_rows_hidden``) — so the cost is not the sum of the rows, and the parts are grown greedily
    against that product. The rows of a record are independent causal sequences (``rows_of``: the
    state plus one question each), so a part is exactly the questions it carries; no question is ever
    cut, which is what would change the context the model is served.

    A single row over the budget becomes its own part: a row cannot be split without cutting the
    causal context, and that is a decision about what a run means rather than a memory knob.
    :func:`oversized_parts` counts those, so a run can report how many passes went over.
    """
    if budget <= 0 or len(row_lengths) < 2:
        return [(0, len(row_lengths))]
    parts, first = [], 0
    for index in range(1, len(row_lengths) + 1):
        if index == len(row_lengths) or part_cost(row_lengths, first, index + 1) > budget:
            parts.append((first, index))
            first = index
    return parts


def oversized_parts(row_lengths: Sequence[int], budget: int) -> int:
    """How many of :func:`plan_parts`' parts went over the budget because one row did."""
    return sum(1 for first, last in plan_parts(row_lengths, budget) if part_cost(row_lengths, first, last) > budget)


def row_lengths(model: Any, enc: dict) -> list[int] | None:
    """The per-question token lengths of one encoding, or ``None`` when it does not run in the row form.

    ``None`` is the packed form (an attention-only backbone whose packed record fits one serving
    row): its pass is not shaped by a longest-row pad, so it has no rows to budget and the record
    stays one pass.
    """
    if not model.rows_form([enc]):
        return None
    from kev.model import rows_of

    state, _, rows = rows_of(enc)
    return [len(state) + len(row["ids"]) for row in rows]


def sub_encoding(enc: dict, first: int, last: int) -> dict:
    """Questions ``[first, last)`` of a packed encoding, as a packed encoding of their own.

    kev's encoder is per question — a branch's ids are its own and its positions restart from the
    state — so the corresponding rows of the whole record and of the subset are the same tokens in
    the same positions: this is ``rows_of``'s own layout read from the other end. The state is kept,
    because it is what every row continues from.
    """
    state = enc["seg"].count(0)
    ends = [state] + [index + 1 for index in enc["decide_idx"]]      # exclusive end of each question
    ids, pos, opt, seg = enc["ids"][:state], enc["pos"][:state], enc["opt"][:state], [0] * state
    decide_idx, opt_idx, labels = [], [], []
    for branch, question in enumerate(range(first, last), start=1):
        start, stop = ends[question], ends[question + 1]
        base = len(ids)
        ids += enc["ids"][start:stop]
        pos += enc["pos"][start:stop]
        opt += enc["opt"][start:stop]
        seg += [branch] * (stop - start)
        decide_idx.append(base + enc["decide_idx"][question] - start)
        opt_idx.append([base + index - start for index in enc["opt_idx"][question]])
        labels.append(enc["labels"][question])
    return {**enc, "ids": ids, "pos": pos, "opt": opt, "seg": seg,
            "decide_idx": decide_idx, "opt_idx": opt_idx, "labels": labels}


def split_variant(variant: Any, first: int, last: int) -> Any:
    """The variant of one part of a record: its questions ``[first, last)``, encoded as their own record.

    ``kev.train.batch_loss`` scores a variant against ``variant.rec["questions"]``, so the record and
    the encoding are cut together — and the permuted copy with them, by the same question indices
    (a permutation reorders options *within* a question, so the question indices are the record's own).
    """
    from kev.train import Variant

    permuted = None
    if variant.permuted is not None:
        permuted = (sub_encoding(variant.permuted[0], first, last), variant.permuted[1][first:last])
    return Variant(rec={**variant.rec, "questions": variant.rec["questions"][first:last]},
                   enc=sub_encoding(variant.enc, first, last),
                   request_id=variant.request_id, source=variant.source, permuted=permuted)


def variant_forwards(model: Any, variant: Any, budget: int, divisor: float, stats: Counter) -> list[Forward]:
    """One record's passes: itself when it fits the budget, its rows cut when they do not.

    ``stats`` collects what a run has to report about the cutting: the records it cut, the passes it
    made, and the passes that went over the budget because a single row did.
    """
    lengths = row_lengths(model, variant.enc)
    if lengths is None:
        stats["packed_records"] += 1
        return [Forward([variant], divisor)]
    stats["records"] += 1
    parts = plan_parts(lengths, budget)
    stats["passes"] += len(parts)
    if len(parts) == 1:
        return [Forward([variant], divisor)]
    stats["split_records"] += 1
    stats["oversized"] += oversized_parts(lengths, budget)
    questions = len(variant.rec["questions"])
    return [Forward([split_variant(variant, first, last)], divisor, (last - first) / questions)
            for first, last in parts]


def budgeted_steps(built: Any, tokenizer: Any, namespace: Any, requests: list, epoch: int, budget: int,
                   stats: Counter, skip: int = 0) -> Iterator[Step]:
    """One epoch's optimizer steps, each cut into passes of at most ``budget`` padded tokens.

    The steps are :func:`groups_of`'s — the count-budgeted path's own grouping — so the budget decides
    the *passes* and nothing else. Each record is encoded on its own: kev seeds the augmentation per
    ``(seed, epoch, record id)``, so a record's variants are the ones the count-budgeted path trains,
    whichever records share its step.

    ``skip`` drops the epoch's first ``skip`` steps without encoding them, which is how a resume picks
    the epoch up where the checkpoint left it. Nothing is lost by not encoding them: the RNG state at
    that boundary is restored from the checkpoint itself, so the steps that *are* run are encoded from
    the state they were encoded from the first time.
    """
    import kev.train as kt

    for start, stop in groups_of(len(requests), namespace.batch, namespace.accum)[skip:]:
        group = requests[start:stop]
        forwards: list[Forward] = []
        for request in group:
            variants = kt.encode_batch(built, tokenizer, namespace, [request], epoch)
            for variant in variants:
                forwards += variant_forwards(built, variant, budget, len(group) * len(variants), stats)
        stats["steps"] += 1
        yield Step(forwards=forwards, records=len(group))


def counted_steps(built: Any, tokenizer: Any, namespace: Any, requests: list, epoch: int,
                  micro_per_epoch: int, skip: int = 0) -> Iterator[Step]:
    """Today's path, as steps: ``--batch`` records per pass, ``--accum`` passes per optimizer step.

    Kept as its own function so the token budget cannot change it: the divisor is
    ``kev.train.accumulation_records``' own (the records in this step, times the variants this pass
    carries per record), and the step boundary is every ``--accum`` passes or the end of the epoch.
    ``skip`` is :func:`budgeted_steps`' ``skip``: the steps a resume does not re-run.
    """
    import kev.train as kt

    for start in range(skip * namespace.accum, micro_per_epoch, namespace.accum):
        forwards, records = [], 0
        for microbatch in range(start, min(start + namespace.accum, micro_per_epoch)):
            chunk = requests[microbatch * namespace.batch:(microbatch + 1) * namespace.batch]
            batch = kt.encode_batch(built, tokenizer, namespace, chunk, epoch)
            divisor = kt.accumulation_records(
                len(requests), namespace.batch, namespace.accum, microbatch) * (len(batch) / len(chunk))
            forwards.append(Forward(batch, divisor))
            records += len(chunk)
        yield Step(forwards=forwards, records=records)


# --------------------------------------------------------------------------- delta


def warm_start(init_from: str, model: Any, meta: Any) -> dict[str, Any]:
    """Delta fine-tuning: load an existing run's adapter and pointer head into a freshly built ``model``.

    The rules are ``kev.checkpoint.Checkpoint.warm_start``'s and are not restated here — the
    source's architecture fields are compared against ``meta`` (the Meta this run will save)
    *before* anything is loaded, and the adapter must cover exactly the model's adapter tensors,
    because peft loads matching keys silently and a half-loaded adapter still trains and still
    reports a loss. Calling kev's own check is what keeps a delta run the same object whichever
    trainer produced it; re-implementing the comparison here is how the two would drift.

    What this adds is provenance: kev's record of the load plus the source's own base id and
    revision, so the run is reproducible from ``provenance.json`` alone.
    """
    from kev.checkpoint import Checkpoint

    source = Checkpoint(init_from)
    return {**source.warm_start(model, meta),
            "base": source.meta.base, "base_revision": source.meta.base_revision,
            # The source *repo's* own commit, when it has one. kev records `resolved`, the snapshot
            # directory the id resolved to, and huggingface_hub names that directory after the
            # commit — so the pin is in the path already. It is lifted out here so a reader of
            # `training_config.json` does not have to know that convention to name the revision a
            # run started from. A local run directory is not a commit and records `None`.
            "source_revision": source_revision(source.path)}


#: A hub snapshot directory is named after the commit it holds; a local run directory is not.
_SOURCE_REVISION = re.compile(r"^[0-9a-f]{40}$")


def source_revision(resolved: str) -> str | None:
    """The source checkpoint's commit, read from the snapshot path kev resolved it to."""
    name = Path(resolved).name
    return name if _SOURCE_REVISION.match(name) else None


# --------------------------------------------------------------------------- checkpoints


def checkpoint_files(out_dir: str | Path) -> list[Path]:
    """Every resumable checkpoint in a run directory, oldest step first.

    Read from the names (``checkpoint-step-<n>.pt``) rather than from a listing that has to be
    trusted: a file that does not parse as one of ours is not a checkpoint, so an unrelated
    ``.pt`` in the directory can never be picked up as one.
    """
    found: list[tuple[int, Path]] = []
    for path in Path(out_dir).glob(f"{CHECKPOINT_PREFIX}*.pt"):
        step = path.name[len(CHECKPOINT_PREFIX):-len(".pt")]
        if step.isdigit():
            found.append((int(step), path))
    return [path for _, path in sorted(found)]


def save_checkpoint(out_dir: str | Path, payload: Mapping[str, Any], *, keep: int = RETAINED_CHECKPOINTS) -> Path:
    """Write one checkpoint atomically and prune the older ones. Returns the path written.

    Atomic because the thing being survived is a reboot: the file is written to ``<name>.tmp``,
    flushed to the disk, and renamed over the real name — so the next attempt sees either the
    previous checkpoint or the whole new one, never a half-written file with a valid-looking name.
    """
    import torch

    out_dir = Path(out_dir)
    path = out_dir / f"{CHECKPOINT_PREFIX}{int(payload['step'])}.pt"
    scratch = path.with_name(path.name + ".tmp")
    started = time.time()
    with scratch.open("wb") as handle:
        torch.save(dict(payload), handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(scratch, path)
    removed = prune_checkpoints(out_dir, keep)
    print(f"checkpoint: {path.name} written ({path.stat().st_size / 2**20:.1f} MiB, {time.time() - started:.1f}s; "
          f"step {payload['step']}, attempt {payload['attempt']})" + (f"; removed {', '.join(removed)}" if removed else ""),
          file=sys.stderr, flush=True)
    return path


def prune_checkpoints(out_dir: str | Path, keep: int = RETAINED_CHECKPOINTS) -> list[str]:
    """Keep the newest ``keep`` checkpoints, drop the rest and any half-written temporary. Returns the names removed."""
    out_dir = Path(out_dir)
    files = checkpoint_files(out_dir)
    doomed = files if keep <= 0 else files[:-keep]
    for path in [*doomed, *sorted(out_dir.glob(f"{CHECKPOINT_PREFIX}*.pt.tmp"))]:
        path.unlink(missing_ok=True)
    return [path.name for path in doomed]


def load_resume_checkpoint(out_dir: str | Path) -> tuple[Path, dict[str, Any]]:
    """The newest *readable* checkpoint in a run directory, with its payload.

    Readable rather than merely newest: the file that does not load is the one the crash interrupted,
    which is why a run keeps two — it is skipped with a note and the one before it is used. A
    directory with no checkpoint at all is a refusal that names ``--resume``, because that flag
    without a checkpoint is a typo rather than a run.
    """
    import torch

    out_dir = Path(out_dir)
    candidates = checkpoint_files(out_dir)
    if not candidates:
        raise FileNotFoundError(
            f"--resume: no checkpoint in {out_dir} (looked for {CHECKPOINT_PREFIX}<step>.pt). A resume needs one — "
            f"that is the whole precondition. Run without --resume into a new directory, or resume a directory a "
            f"checkpointed run wrote ('--checkpoint_every' is what makes one).")
    broken: list[str] = []
    for path in reversed(candidates):
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as error:  # noqa: BLE001 - any read failure means "not this one"
            broken.append(f"{path.name} ({type(error).__name__}: {str(error)[:120]})")
        else:
            if isinstance(payload, dict) and payload.get("format") == CHECKPOINT_FORMAT:
                missing = [key for key in CHECKPOINT_KEYS if key not in payload]
                if missing:
                    raise ValueError(f"--resume: {path} is a {CHECKPOINT_FORMAT} checkpoint but records no {missing}; "
                                     f"it was written by a different format — resume needs one from this one")
                return path, payload
            broken.append(f"{path.name} (not a {CHECKPOINT_FORMAT} checkpoint)")
        print(f"resume: {broken[-1]}; trying the checkpoint before it", file=sys.stderr, flush=True)
    raise ValueError(f"--resume: no readable checkpoint in {out_dir}: " + "; ".join(broken))


def rng_snapshot(generator: random.Random) -> dict[str, Any]:
    """The generators a continuation has to restore to be the same continuation.

    ``generator`` is the run's *own* ``random.Random``: the epoch's shuffle comes from an instance,
    not from the module's global generator, so ``random.setstate`` alone would restore the wrong
    object — a resume that did that shuffled its next epoch from the seed instead of from the
    checkpoint's stream and trained a different order. Both are recorded (the global one is what any
    library reached from this loop would draw from) and both are restored.

    Five in total, because five can be consumed between two steps: ``torch`` and ``cuda`` by the LoRA
    dropout the forward passes draw, ``python`` by the epoch's shuffle, ``random_module`` by a
    library's own draw, and ``numpy`` by nothing here today but by anything this loop might grow.
    Only what exists is recorded — ``cuda`` needs a device, ``numpy`` needs the package — and
    :func:`restore_rng` refuses a checkpoint that recorded one this box cannot reproduce rather than
    continuing without it.
    """
    import torch

    state: dict[str, Any] = {"python": generator.getstate(), "random_module": random.getstate(),
                             "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    try:
        import numpy
    except ImportError:
        pass
    else:
        state["numpy"] = numpy.random.get_state()
    return state


def restore_rng(state: Mapping[str, Any], generator: random.Random) -> None:
    """Restore what :func:`rng_snapshot` recorded into ``generator``. Refuses a state this box cannot reproduce."""
    import torch

    generator.setstate(state["python"])
    random.setstate(state["random_module"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state:
        if not torch.cuda.is_available():
            raise ValueError("--resume: the checkpoint records CUDA RNG states and this process has no CUDA device, "
                             "so the continuation could not be the one the checkpoint was taken at")
        torch.cuda.set_rng_state_all(state["cuda"])
    if "numpy" in state:
        try:
            import numpy
        except ImportError as error:
            raise ValueError("--resume: the checkpoint records a numpy RNG state and numpy is not importable "
                             "here, so the continuation could not be the one the checkpoint was taken at") from error
        numpy.random.set_state(state["numpy"])


def run_recipe(args: argparse.Namespace, namespace: Any, *, revision: str | None, planned: int) -> dict[str, Any]:
    """The knobs that decide a run's trajectory — what a resume has to find again to be a continuation.

    Not the whole command line: this is the set whose difference would make the resumed steps a
    different run (the weights, the data, the grouping, the seed, the schedule's length) and whose
    sameness makes them the run the checkpoint was taken from. ``--steps`` is deliberately absent: it
    only decides where a run stops, and its effect on the schedule travels as ``epochs`` and
    ``planned_steps``, which are here — so lowering ``--steps`` inside the same schedule is allowed
    and raising it far enough to change the LR schedule is refused.
    """
    return {
        "base": namespace.base, "base_revision": revision, "suite": str(namespace.suite),
        "suite_sha256": file_digest(Path(namespace.suite) / "manifest.json"),
        "seed": namespace.seed, "batch": namespace.batch, "accum": namespace.accum,
        "epochs": namespace.epochs, "planned_steps": planned,
        "lr": namespace.lr, "head_lr": namespace.head_lr, "weight_decay": namespace.weight_decay,
        "dtype": namespace.dtype, "weights_dtype": namespace.weights_dtype, "max_state": namespace.max_state,
        "checkpointing": namespace.checkpointing,
        "lora": namespace.lora, "head_dim": namespace.head_dim, "lora_targets": namespace.lora_targets,
        "option_isolation": bool(namespace.option_isolation),
        "special_embeddings": bool(namespace.special_embeddings),
        "p_none_pair": namespace.p_none_pair, "token_budget": args.token_budget, "gpus": args.gpus,
        "init_from": args.init_from,
    }


def check_resume_recipe(saved: Mapping[str, Any], expected: Mapping[str, Any], where: Path) -> list[str]:
    """Refuse a resume whose recipe is not the checkpoint's, naming every field that differs.

    Returns the fields that may differ (:data:`RESUME_NOTED_FIELDS`) so the caller can report them:
    a different card count is the same recipe drawn on other cards, and ``--init_from`` cannot reach
    the trajectory at all because the checkpoint's own adapter supersedes the warm start.
    """
    wrong = {key: (saved.get(key), expected[key]) for key in expected
             if key not in RESUME_NOTED_FIELDS and saved.get(key) != expected[key]}
    if wrong:
        shown = "; ".join(f"{key}: checkpoint {was!r} vs this run {now!r}" for key, (was, now) in sorted(wrong.items()))
        raise ValueError(
            f"--resume: {where.name} was written by a different run — {shown}. A resume continues the run its "
            f"checkpoint was taken from; if the recipe really changed, start a fresh run in a new directory.")
    return sorted(key for key in RESUME_NOTED_FIELDS if saved.get(key) != expected[key])


def place_optimizer_state(optimizer: Any) -> None:
    """Put every optimizer moment back on its parameter's device and dtype.

    ``torch.load(map_location="cpu")`` brings the moments home, and this is the cast back made
    explicit: a moment left on the host is a device error on the next step rather than a wrong
    number, but it would land after a 9B has loaded. Scalar state — AdamW's step count — is left
    where a fresh optimizer keeps it, on the host.
    """
    import torch

    for group in optimizer.param_groups:
        for parameter in group["params"]:
            state = optimizer.state.get(parameter) or {}
            for key, value in state.items():
                if isinstance(value, torch.Tensor) and value.dim() > 0:
                    state[key] = value.to(device=parameter.device, dtype=parameter.dtype)


def adapter_state(model: Any) -> dict[str, Any]:
    """A run's adapter tensors, on the host and detached: what a checkpoint carries of the backbone.

    ``get_peft_model_state_dict`` is peft's own answer to "which tensors are the adapter" — the same
    call ``save_pretrained`` makes at the end of a run — so a checkpoint holds exactly the tensors the
    released checkpoint would, and not one byte of the frozen backbone (which is what keeps a 27B
    arm's checkpoint loadable rather than 51 GB).
    """
    from peft import get_peft_model_state_dict

    return {name: tensor.detach().cpu().clone() for name, tensor in get_peft_model_state_dict(model.lm).items()}


def restore_checkpoint(payload: Mapping[str, Any], model: Any, optimizer: Any, schedule: Any, where: Path) -> None:
    """Load a checkpoint's weights, optimizer state and schedule position into a freshly built run.

    The adapter goes through peft's own pair — ``get_peft_model_state_dict`` when it was written,
    ``set_peft_model_state_dict`` here — which is the pair ``kev.checkpoint.Checkpoint.warm_start``
    uses, so a resume and a delta load agree on what "the adapter" is. The key sets are compared in
    both directions the way kev compares them: peft loads matching keys silently, and a half-loaded
    adapter still trains and still reports a loss.

    The optimizer is loaded before the scheduler: the schedule's position is a step count, and the
    learning rate the resumed step runs at is the one the optimizer's parameter group carries.
    """
    from peft import get_peft_model_state_dict, set_peft_model_state_dict

    weights = payload["adapter"]
    have = set(get_peft_model_state_dict(model.lm))
    unexpected, missing = sorted(set(weights) - have), sorted(have - set(weights))
    if unexpected:
        raise ValueError(f"--resume: {where.name} carries {len(unexpected)} adapter tensors this model does not have "
                         f"(e.g. {unexpected[:2]}); check --lora_targets / --lora against the run that wrote it")
    if missing:
        raise ValueError(f"--resume: {where.name} does not cover {len(missing)} of this model's adapter tensors "
                         f"(e.g. {missing[:2]}); check --lora_targets")
    set_peft_model_state_dict(model.lm, dict(weights))
    model.head.load_state_dict(payload["head"])
    optimizer.load_state_dict(payload["optimizer"])
    place_optimizer_state(optimizer)
    schedule.load_state_dict(payload["scheduler"])


def order_of(requests: Sequence[Any], positions: Mapping[int, int]) -> list[int]:
    """The current epoch's training order, as indices into the pre-shuffle list: a resume's cursor.

    Indices rather than record ids, because an id is not unique (``--synthetic_repeat`` duplicates
    records outright) and the order is a permutation of the *list*, which is what has to be restored.
    """
    return [positions[id(request)] for request in requests]


def apply_order(requests: list, order: Sequence[int], where: Path) -> None:
    """Put a restored epoch order back into ``requests`` in place, refusing one that is not a permutation."""
    indices = [int(index) for index in order]
    if sorted(indices) != list(range(len(requests))):
        raise ValueError(f"--resume: {where.name} records an epoch order of {len(indices)} index/indices that is not a "
                         f"permutation of this run's {len(requests)} training records; the suite or its training "
                         f"partition has changed under the checkpoint")
    requests[:] = [requests[index] for index in indices]


def trim_run_log(log_path: str | Path, step: int) -> tuple[int, int]:
    """Drop the rows a resume is about to re-run. Returns ``(rows dropped, unreadable lines dropped)``.

    A run's log has to read as one run: the steps after the newest checkpoint are trained again from
    it, so appending would leave their first attempt's rows in the file beside the second's, and a
    step would appear twice with different wall-clock fields. The kept rows are the original bytes —
    this truncates a JSONL file, it does not re-serialize one.
    """
    log_path = Path(log_path)
    if not log_path.exists():
        return 0, 0
    lines = [line for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    kept, broken = [], 0
    for line in lines:
        try:
            seen = int(json.loads(line)["step"])
        except (TypeError, ValueError, KeyError):
            broken += 1        # a row the kill truncated mid-write; the re-run writes it again
            continue
        if seen <= step:
            kept.append(line)
    dropped = len(lines) - len(kept) - broken
    if dropped or broken:
        scratch = log_path.with_name(log_path.name + ".tmp")
        scratch.write_text("".join(f"{line}\n" for line in kept), encoding="utf-8")
        os.replace(scratch, log_path)
    return dropped, broken


# --------------------------------------------------------------------------- run


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Train one run and write its artifacts. Returns the provenance record."""
    import contextlib as _contextlib

    import kev.train as kt
    import torch
    from kev.checkpoint import Meta, write_meta
    from kev.device import allocated_bytes, default_device, empty_cache
    from kev.model import load_tokenizer, training_context
    from kev.suite import read_manifest

    out_dir = Path(args.out)
    ns = kev_namespace(args, str(out_dir))
    if args.resume:
        # A resume's directory has to exist -- that is where its checkpoint is -- and the
        # checkpoint is looked for before the tokenizer and the training partition are read, so a
        # resume of a directory without one costs a second rather than a suite's tokenization.
        if not out_dir.is_dir():
            raise FileNotFoundError(f"--resume: {out_dir} does not exist, so there is no checkpoint to "
                                    f"resume from (without --resume a run refuses an existing --out instead)")
    else:
        out_dir.mkdir(parents=True)
    resume_path: Path | None = None
    payload: dict[str, Any] | None = None
    if args.resume:
        resume_path, payload = load_resume_checkpoint(out_dir)

    torch.manual_seed(ns.seed)
    rng = random.Random(ns.seed)
    device = ns.device or default_device()
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if ns.dtype == "bf16" else _contextlib.nullcontext()

    manifest = read_manifest(ns.suite)
    revision = kt.pinned_revision(ns, manifest)
    holdout = manifest["holdout_sources"]
    tokenizer = load_tokenizer(ns.base, revision=revision)
    requests = kt.training_requests(ns, tokenizer, manifest, holdout)
    # Before the model is loaded, so a budget that cannot hold the data costs
    # seconds on the CPU rather than a cold GPU boot and one failed batch.
    context = suite_budget(ns.suite)
    verify_recorded_budget(context, ns.suite)
    limits = training_context(ns.max_state)
    measured = verify_state_budget(tokenizer, requests, limits, ns.suite)
    print(f"state budget {context['max_state']} measured over {measured['n']} training records "
          f"(longest state {measured['max']}, median {measured['median']}); kev is handed {ns.max_state} "
          f"(row limit {limits['max_branch']}, packed limit {limits['max_packed']})", file=sys.stderr, flush=True)
    # --steps bounds a run in optimizer steps, which is what makes a smoke run a
    # bounded foreground command. A smoke suite's training partition is far shorter
    # than the bound, so the epoch count is raised until the bound is reachable
    # instead of the run ending after one step and calling itself a smoke test.
    steps_per_epoch = max(math.ceil(math.ceil(len(requests) / ns.batch) / ns.accum), 1)
    if args.steps > ns.epochs * steps_per_epoch:
        ns.epochs = math.ceil(args.steps / steps_per_epoch)
        print(f"--steps {args.steps} over {len(requests)} training records: {ns.epochs} epochs "
              f"of {steps_per_epoch} optimizer step(s)", file=sys.stderr, flush=True)
    micro_per_epoch = math.ceil(len(requests) / ns.batch)
    planned = ns.epochs * math.ceil(micro_per_epoch / ns.accum)

    # The recipe a resume has to find again, and the checkpoint it reads it from. Both are decided
    # here -- before a single weight is loaded -- so a resume that cannot be a continuation costs
    # seconds rather than a cold model load.
    recipe = run_recipe(args, ns, revision=revision, planned=planned)
    # the order `training_requests` built, before any shuffle: a checkpoint records an epoch's shuffle
    # as indices into *this* list, so the permutation can be put back without touching the RNG
    positions = {id(request): index for index, request in enumerate(requests)}
    if payload is not None and resume_path is not None:
        noted = check_resume_recipe(payload["recipe"], recipe, resume_path)
        print(f"resume: {resume_path.name} — step {payload['step']} of epoch {payload['epoch']} "
              f"({payload['step_in_epoch']} step(s) into that epoch), attempt {int(payload['attempt'])}",
              file=sys.stderr, flush=True)
        for key in noted:
            print(f"resume: note — {key} differs from the checkpoint's ({payload['recipe'].get(key)!r} there, "
                  f"{recipe[key]!r} here); the continuation is this recipe's draw, not the killed attempt's",
                  file=sys.stderr, flush=True)
        if args.checkpoint_every == 0:
            print("resume: note — --checkpoint_every 0: this attempt writes no checkpoint, so a second "
                  "interruption resumes from the one it started from", file=sys.stderr, flush=True)

    settings = model_module.lora_config({key: getattr(ns, key) for key in MODEL_SETTINGS})
    built = model_module.build_model(ns.base, revision, settings, device=device, tokenizer=tokenizer)
    meta = Meta(base=ns.base, base_revision=revision, lora=ns.lora, head_dim=ns.head_dim,
                option_isolation=bool(ns.option_isolation), special_embeddings=bool(ns.special_embeddings),
                weights_dtype=ns.weights_dtype, holdout=holdout)
    init_source = None
    if args.init_from:
        # delta mode: start from an already trained adapter + pointer head instead of the base
        # model, before the optimizer exists, so the first step already trains the released model
        init_source = warm_start(args.init_from, built, meta)
        print(f"delta: warm start from {init_source['resolved']}: {init_source['adapter_tensors']} adapter tensors "
              f"and the pointer head loaded", file=sys.stderr, flush=True)
    trainables = model_module.verify_frozen(built)
    placement = None
    if args.gpus > 1:
        # The shard happens after the warm start (so the adapter is attached before anything is placed)
        # and before the optimizer exists (so no state was created on a card the weights then left).
        placement = shard_module.shard_model(built, args.gpus, device)
        print(f"multi-gpu: {shard_module.describe(placement)}", file=sys.stderr, flush=True)
    if ns.checkpointing:
        built.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        # With a frozen backbone no input requires grad, and torch's non-reentrant checkpoint *silently
        # skips* in that case -- it warns and keeps every activation instead of recomputing it. That is
        # what the 4B/9B deltas hit: 8 GB of bf16 weights, 36 GB allocated, OOM on a 40 GB card, with
        # checkpointing nominally on. Marking the embeddings' output as requiring grad is the standard
        # fix and costs one fp32 copy of the embedding activations.
        built.lm.enable_input_require_grads()
    built.lm.config.use_cache = False

    head_ids = {id(parameter) for parameter in built.head.parameters()}
    groups = [
        {"params": [p for p in built.trainable_parameters() if id(p) not in head_ids], "lr": ns.lr},
        {"params": list(built.head.parameters()), "lr": ns.head_lr or ns.lr},
    ]
    optimizer = torch.optim.AdamW(groups, lr=ns.lr, weight_decay=ns.weight_decay)
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[ns.lr, ns.head_lr or ns.lr], total_steps=max(planned, 1), pct_start=0.1
    )

    # The resume, all of it: the checkpoint's weights, optimizer and schedule, its cursor into the
    # epoch's steps, its record order, and the RNG states its step boundary was drawn from. Nothing
    # consumed RNG after the boundary in the killed attempt, so restoring the states *here* -- after
    # the model build's own seeding and LoRA initialisation, all of which the checkpoint's weights
    # supersede -- makes the next step the step the checkpoint was taken before.
    attempt, start_epoch, skip_steps = 1, 0, 0
    step = seen = run_peak = loaded_step = 0
    carried: dict[str, int] = {}          # the resumed epoch's running counters, from the checkpoint
    resumed = payload is not None
    if payload is not None and resume_path is not None:
        attempt = int(payload["attempt"]) + 1
        start_epoch, skip_steps = int(payload["epoch"]), int(payload["step_in_epoch"])
        step = loaded_step = int(payload["step"])
        seen, run_peak = int(payload["records_seen"]), int(payload["peak_device_bytes"])
        if args.steps and step >= args.steps:
            raise ValueError(
                f"--resume: {resume_path.name} records step {step}, at or past this run's --steps {args.steps}: "
                f"nothing is left to train. Drop --resume to start a fresh run in a new directory.")
        carried = {key: int(payload[key]) for key in ("epoch_records", "epoch_tokens", "passes")}
        restore_checkpoint(payload, built, optimizer, schedule, resume_path)
        restore_rng(payload["rng"], rng)
        apply_order(requests, payload["order"], resume_path)
        dropped, broken = trim_run_log(out_dir / "run.log.jsonl", step)
        where = (f"epoch {start_epoch}, {skip_steps} step(s) in" if start_epoch < ns.epochs
                 else f"the end of the {ns.epochs}-epoch schedule")
        print(f"resume: continued from step {step} ({where}); {dropped} row(s) the first attempt wrote past it "
              f"and {broken} unreadable row(s) dropped from run.log.jsonl, to be written again by this attempt",
              file=sys.stderr, flush=True)
    elif args.checkpoint_every:
        print(f"checkpoints: one every {args.checkpoint_every} optimizer step(s) into {out_dir} "
              f"(latest {RETAINED_CHECKPOINTS} kept); --resume continues from one", file=sys.stderr, flush=True)

    write_json(out_dir / "training_config.json", {
        "args": {**vars(ns), "steps": args.steps, "epochs_requested": args.epochs,
                 "profile_memory": args.profile_memory, "gpus": args.gpus,
                 "token_budget": args.token_budget,
                 "checkpoint_every": args.checkpoint_every, "resume": args.resume,
                 # kev's parser has an `init_from` of its own and this CLI never hands it the flag
                 # (the flag must not reach `vars(ns)`, or every ordinary run's config would move),
                 # so `vars(ns)` records kev's *empty default* on every run -- a delta run's own
                 # config said "not a delta run". The flag's real value is what the run used.
                 "init_from": args.init_from},
        "suite_sha256": file_digest(Path(ns.suite) / "manifest.json"),
        # The base the run actually built. `model_settings["base"]` below is kev's own default
        # (`Qwen/Qwen3-0.6B-Base`), which no run in this project trains; this block is the pair a
        # reader should use, and it is the same pair `provenance.json` records.
        "base": {"id": ns.base, "revision": revision},
        "base_revision": revision,
        # kev's `lora_config` fills in *its* defaults for every key the run does not set, and one of
        # them is a base id. Left as kev's default it named a model that was never loaded -- the
        # single field a reader would check to ask "which base?" answered with the wrong one, which
        # is how a series of runs warm-started from the released kev checkpoints came to be read as
        # base-initialized. `base` is the one key where kev's default is not a safe default, so the
        # true value is substituted; every other key is still exactly what was handed to `build_model`.
        "model_settings": {**settings, "base": ns.base},
        "trainable": {"count": trainables.count, "by_kind": trainables.by_kind, "tensors": len(trainables.names)},
        # The placement block exists only for a sharded run: a single-card run's placement is the whole
        # model on cuda:0, and `"gpus": 1` above is what says so. Its absence is that statement, not a
        # missing measurement.
        **({"placement": placement} if placement is not None else {}),
        # The initialization, in the same shape `provenance.json` records it: what was requested, what
        # it resolved to, the source's own adapter and head digests, and the base pair its head.pt
        # carries. Absent on an ordinary run -- its absence is the statement that the run started from
        # the base model alone, and it is a statement rather than a missing measurement because the
        # key is written whenever the flag is given and cannot be written otherwise.
        **({"init_from": init_source} if init_source is not None else {}),
    })

    built.train()
    profile = PhaseMemory(device, args.profile_memory)
    peaks = shard_module.DevicePeaks(placement["devices_used"]) if placement is not None else None

    def device_peak() -> int:
        """The peak allocated bytes this run reports: the worst card when sharded, kev's own device otherwise."""
        return peaks.max() if peaks is not None else allocated_bytes(device)

    log_path = out_dir / "run.log.jsonl"
    epochs: list[dict[str, Any]] = []
    started = time.time()
    stop = False
    checkpoint_every = args.checkpoint_every
    budget_stats: Counter = Counter()
    if args.token_budget:
        per_epoch = math.ceil(micro_per_epoch / ns.accum)
        print(f"token budget {args.token_budget}: a forward pass costs at most that many padded tokens "
              f"(rows x longest row, the shape the row form builds); an optimizer step is still "
              f"{ns.batch} x {ns.accum} = {ns.batch * ns.accum} records over {per_epoch} step(s) per epoch "
              f"and the LR schedule is unchanged", file=sys.stderr, flush=True)
    # `run.log.jsonl` gets one line per optimizer step of the *run*, not per attempt: a resume has
    # dropped the rows it is about to write again (trim_run_log), and the step counter it continues
    # from is the checkpoint's, so the file reads as one run end to end.
    with log_path.open("a" if resumed else "w", encoding="utf-8") as log:
        for epoch in range(start_epoch, ns.epochs):
            # the resumed epoch keeps the order its checkpoint recorded -- its position in the RNG
            # stream is not reconstructible from a seed, and a different order is a different epoch
            resuming_this_epoch = resumed and epoch == start_epoch
            if not resuming_this_epoch:
                rng.shuffle(requests)
            skip = skip_steps if resuming_this_epoch else 0
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()   # so the epoch's peak is the epoch's, not the load's
            if peaks is not None:
                peaks.reset()
            profile.reset()
            epoch_start = time.time()
            epoch_peak = 0
            # the resumed epoch continues its counters (the checkpoint's), so the log's epoch columns
            # read as one epoch rather than as two attempts side by side
            epoch_records, epoch_tokens = (carried["epoch_records"], carried["epoch_tokens"]) if resuming_this_epoch else (0, 0)
            # a Counter, so a term the pass did not report reads 0 rather than raising; its values are
            # floats because a pass's terms are weighted by its share of its record's loss
            window: Any = Counter()
            # forward passes made this epoch: what the log calls `microbatch`
            passes = carried["passes"] if resuming_this_epoch else 0
            epoch_steps = (budgeted_steps(built, tokenizer, ns, requests, epoch, args.token_budget, budget_stats, skip)
                           if args.token_budget else counted_steps(built, tokenizer, ns, requests, epoch, micro_per_epoch, skip))
            for planned_index, planned_step in enumerate(epoch_steps):
                step_slots = step_tokens = tokens = 0
                for forward in planned_step.forwards:
                    batch = forward.batch
                    rows, slots, longest = batch_shape(built, batch)
                    shape = {"records": len(batch), "rows": rows, "slots": slots, "l_max": longest}
                    profile("encode", microbatch=passes, **shape)
                    loss, terms = kt.batch_loss(built, ns, batch, device, {}, None, autocast)
                    profile("forward", microbatch=passes, **shape)
                    # weight by source records in the accumulation group so a none-pair sibling does not
                    # inflate its parent record's share of the update; `forward.weight` is 1 for a whole
                    # record and the part's share of its record's questions on the token-budgeted path
                    (loss * forward.weight / forward.divisor).backward()
                    profile("backward")
                    for key, value in terms.items():
                        window[key] = window[key] + forward.weight * value
                    window["n"] = window["n"] + forward.weight * len(batch)
                    tokens = sum(forward_tokens(built, variant.enc) for variant in batch)
                    epoch_tokens += tokens
                    step_tokens += tokens
                    step_slots = max(step_slots, slots)
                    passes += 1
                # the step's data (its records, and the passes that carry them) counts once per step
                seen += planned_step.records
                epoch_records += planned_step.records
                torch.nn.utils.clip_grad_norm_(built.trainable_parameters(), 1.0)
                optimizer.step()
                schedule.step()
                optimizer.zero_grad()
                profile("optimizer", microbatch=passes - 1)
                # the per-phase readings clear the running peak, so the epoch's number is the larger of
                # what is left in the counter and the largest phase reading taken from it. It is taken
                # after the optimizer step, not before, or the optimizer's own peak never counts.
                epoch_peak = max(epoch_peak, device_peak(), profile.peak)
                run_peak = max(run_peak, epoch_peak)
                step += 1
                if device == "mps":
                    empty_cache(device)
                mean = window["ce"] / max(window["n"], 1)
                log.write(json.dumps({
                    "step": step, "epoch": epoch, "microbatch": passes - 1, "records_seen": seen,
                    "loss": mean, "kl": window["kl"] / max(window["kl_n"], 1),
                    "anchor": window["anchor"] / max(window["anchor_n"], 1), "window_tokens": tokens,
                    "step_tokens": step_tokens, "passes": len(planned_step.forwards), "slots_max": step_slots,
                    "epoch_tokens": epoch_tokens, "elapsed_s": time.time() - started,
                    "peak_device_bytes": epoch_peak, "lr": schedule.get_last_lr()[0],
                }, ensure_ascii=False) + "\n")
                log.flush()
                window = Counter()
                if step % 10 == 0 or step == args.steps:
                    print(f"ep{epoch} step {step} loss {mean:.3f} {epoch_tokens / max(time.time() - epoch_start, 1e-9):.0f} tok/s",
                          flush=True)
                if args.steps and step >= args.steps:
                    stop = True
                    break
                # A resumable checkpoint, at a *step* boundary: the snapshot is taken here because
                # this is the point the next step's encoding starts from, so a resume restores
                # exactly the generators the killed attempt would have continued with. A step the
                # run stops on is never checkpointed -- a completed run leaves no resumable file
                # behind for a resume to run past its own bound.
                if checkpoint_every and step % checkpoint_every == 0:
                    save_checkpoint(out_dir, {
                        "format": CHECKPOINT_FORMAT,
                        "step": step, "epoch": epoch, "step_in_epoch": planned_index + 1,
                        "records_seen": seen, "attempt": attempt, "peak_device_bytes": run_peak,
                        # the epoch's running counters, so the resumed log's `epoch_tokens`/`microbatch`
                        # columns continue across the boundary instead of restarting (the step's loss
                        # does not depend on them; a reader's arithmetic on the column does)
                        "epoch_records": epoch_records, "epoch_tokens": epoch_tokens, "passes": passes,
                        "order": order_of(requests, positions), "recipe": recipe,
                        "adapter": adapter_state(built),
                        "head": {name: tensor.detach().cpu().clone() for name, tensor in built.head.state_dict().items()},
                        "optimizer": optimizer.state_dict(), "scheduler": schedule.state_dict(),
                        "rng": rng_snapshot(rng),
                    })
            epochs.append({
                "epoch": epoch, "seconds": time.time() - epoch_start, "records": epoch_records,
                "forward_tokens": epoch_tokens, "peak_device_bytes": epoch_peak,
                "tokens_per_second": epoch_tokens / max(time.time() - epoch_start, 1e-9),
                # A sharded run's peak is one number per card; the worst card is `peak_device_bytes`,
                # and the vector is what says whether the shard was balanced where it matters.
                **({"peak_bytes_per_device": peaks.read()} if peaks is not None else {}),
                # Which attempt measured this epoch, on a resumed run only: a resumed attempt's first
                # epoch is the tail of an epoch the first attempt started, so its seconds and records
                # are this attempt's. A run that was not resumed keeps today's keys, byte for byte.
                **({"attempt": attempt} if resumed else {}),
            })
            if stop:
                break

    built.lm.save_pretrained(out_dir)
    meta.head, meta.extra = built.head.state_dict(), {"args": {**vars(ns), "steps": args.steps}, "suite_sha256": file_digest(Path(ns.suite) / "manifest.json")}
    write_meta(out_dir, meta)
    tokenizer.save_pretrained(out_dir)

    provenance = provenance_record(
        namespace=ns, revision=revision, manifest=manifest, suite_dir=ns.suite, trainables=trainables,
        base={
            "id": ns.base, "revision": revision, "text_only": True,
            "class": type(built.lm).__name__, "hidden_size": _hidden_size(built),
            "parameters": sum(parameter.numel() for parameter in built.parameters()),
            "vision_parameters": 0,
            "weight_dtype": ns.weights_dtype,
        },
        environment=_environment(device, placement), epochs=epochs, optimizer_steps=step,
        peak_device_bytes=run_peak, stopped_early=stop, init_from=init_source,
        token_budget=({"budget": args.token_budget, **dict(budget_stats)} if args.token_budget else None),
        resume=({"checkpoint": resume_path.name, "attempt": attempt, "from_step": loaded_step,
                 "from_epoch": start_epoch, "step_in_epoch": skip_steps,
                 "steps_this_attempt": step - loaded_step, "wall_seconds_this_attempt": time.time() - started}
                if resumed and resume_path is not None else None),
    )
    write_json(out_dir / "provenance.json", provenance)
    if args.token_budget:
        print(f"token budget {args.token_budget}: {budget_stats.get('passes', 0)} forward pass(es) over "
              f"{budget_stats.get('steps', 0)} optimizer step(s); {budget_stats.get('split_records', 0)} record(s) cut "
              f"between their questions; {budget_stats.get('oversized', 0)} pass(es) over budget (a single row longer "
              f"than it, which cannot be cut)", flush=True)
    print(f"saved {out_dir} after {step} optimizer steps", flush=True)
    return provenance


def provenance_record(*, namespace: Any, revision: str | None, manifest: dict[str, Any], suite_dir: str | Path,
                      trainables: Any, base: dict[str, Any], environment: dict[str, Any],
                      epochs: list[dict[str, Any]], optimizer_steps: int, peak_device_bytes: int,
                      stopped_early: bool, init_from: dict[str, Any] | None = None,
                      token_budget: dict[str, Any] | None = None,
                      resume: dict[str, Any] | None = None) -> dict[str, Any]:
    """The run's provenance, assembled from measurements.

    A pure function of what the run measured rather than of the run itself, so its
    schema can be asserted on a CPU-only box — which is where the tests for it run.
    It carries what reproducing the run needs: which code (this package's digests
    and the pinned ``kev``'s commit), which data (the suite manifest and every
    partition's sha256), which weights (the base revision), which machine, which
    seed, and what the run cost. A delta run additionally records the checkpoint it
    was warm-started from (``init_from``): the key is absent, not null, on a run
    that started from the base model, so an ordinary run's record is unchanged.

    ``token_budget`` is the same kind of key for the token-budgeted path: the ceiling
    the passes were cut to, and what the cutting did (passes, records cut, passes that
    a single row forced over the ceiling). It is absent, not null, on a run that ran
    the count-budgeted path, so an ordinary run's record is unchanged.

    ``resume`` is that key for a resumed run: which checkpoint this attempt continued
    from, which step it was taken at, the attempt number, and what this attempt cost
    (its own steps and wall seconds — the ``epochs`` entries a resumed attempt writes
    carry an ``attempt`` field for the same reason, since its first epoch is the tail
    of an epoch the previous attempt began). ``optimizer_steps`` is the *run's* total,
    so a resumed run reports the steps it has trained, not the ones this process did.
    Absent, not null, on a run that was not resumed.
    """
    suite_dir = Path(suite_dir)
    record = {
        "tool": {"name": "sckevtype.train", "version": _version()},
        "repos": {
            "sckevtype": git_commit(paths.root()),
            "kev": git_commit(paths.root() / "kev"),
            "sckevtype_source": {name: file_digest(Path(__file__).parent / name) for name in _SOURCE_FILES},
        },
        "suite": {
            "path": str(suite_dir),
            "name": manifest.get("name"),
            "manifest_sha256": file_digest(suite_dir / "manifest.json"),
            "corpus": manifest.get("corpus"),
            "partitions": {name: entry["sha256"] for name, entry in sorted(manifest.get("files", {}).items())},
        },
        "base": base,
        "trainable": {"count": trainables.count, "by_kind": trainables.by_kind, "names": list(trainables.names)},
        "hyperparameters": vars(namespace),
        "environment": environment,
        "determinism": {
            "seed": namespace.seed,
            "torch_manual_seed": namespace.seed,
            "batch_order": "random.Random(seed).shuffle(reqs) at each epoch",
            "augmentation": "kev.train.encode_batch, seeded per (seed, epoch, record id)",
        },
        "epochs": epochs,
        "optimizer_steps": optimizer_steps,
        "peak_device_bytes": peak_device_bytes,
        "stopped_early": bool(stopped_early),
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024),
    }
    if init_from is not None:
        record["init_from"] = init_from
    if token_budget is not None:
        record["token_budget"] = token_budget
    if resume is not None:
        record["resume"] = resume
    return record


def _version() -> str:
    from .. import __version__

    return __version__


def _hidden_size(built: Any) -> int | None:
    return getattr(getattr(built.lm, "config", None), "hidden_size", None)


def _environment(device: str, placement: dict[str, Any] | None = None) -> dict[str, Any]:
    """Library and device versions, recorded so a number can be attributed.

    ``placement`` is the sharded run's own record (``None`` on the single-card path): its device list
    is part of the environment a number was measured in, and ``gpu`` alone would name card 0 of a run
    that used four.
    """
    import platform

    import peft
    import torch
    import transformers

    environment = {
        "device": device,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
    }
    if device == "cuda":
        environment |= {
            "gpu": torch.cuda.get_device_name(0),
            "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
            "cuda": torch.version.cuda,
        }
    if placement is not None:
        environment |= {"gpus": placement["requested_gpus"], "devices": placement["devices_used"]}
    return environment


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    # An existing run directory is refused -- unless this is a resume, whose whole precondition is
    # one, and whose own precondition (a readable checkpoint whose recipe matches) is checked by
    # `run` before anything is loaded or written.
    if Path(args.out).exists() and not args.resume:
        print(f"refusing to overwrite an existing run: {args.out}", file=sys.stderr)
        return 1
    try:
        run(args)
    except (SystemExit, ValueError, KeyError, FileNotFoundError, ImportError) as error:
        print(f"training failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
