#!/usr/bin/env python3
"""Assert a seed run's ``training_config.json`` is the FROZEN configuration, field by field.

``runs/train-real-v1`` was trained by a command line that passes **no** ``--seed`` at all (the
trainer's default is 0) and only a handful of flags; every other hyperparameter in its recorded
configuration is a trainer default. A seed-variance run adds exactly one flag, ``--seed S``, and
changes one path, ``--out runs/train-seed<S>-v1``. So "trained the frozen configuration at a new
seed" is a decidable claim about two files, and this script decides it: diff the seed run's
``training_config.json`` against the frozen one and **refuse if anything other than ``seed`` and
``out`` moved**.

Why it matters: the whole point of the seed family is that a difference between a seed draw and the
frozen draw is *noise*. That reading is only valid if the configuration is bit-for-bit the same. A
moved ``lr``, ``p_none``, ``head_dim``, ``steps`` or trainable count would make the draw a different
configuration, and any difference it showed would be an intervention, not noise. This check runs on
the compute node between training and scoring, so a configuration that is not the frozen one costs
the training pass and not the scoring pass as well.

    python3 scripts/seedvar-config-diff.py runs/train-real-v1/training_config.json \\
                                        runs/train-seed1-v1/training_config.json

Exit 0 only when every compared block is equal except the allowed keys **and** those two keys really
do differ (a "new seed" that did not change the seed is the frozen draw again). Stdlib only, so it
runs under the system ``python3`` as well as under ``$W/venv-train/bin/python``.

The blocks compared, and what each one would catch:

  ``args``           the full sckevtype namespace the run recorded -- the flags, the defaults, the
                     budgets. ``seed`` and ``out`` are the only allowed differences.
  ``suite_sha256``   the manifest digest of the suite trained on (a digest, not a path): a run on
                     the capped evaluation suite instead of the training suite shows up here.
  ``base_revision``  the pinned base weights.
  ``trainable``      ``count`` / ``by_kind`` / ``tensors``: the model that was actually built.
                     ``by_kind.backbone`` must stay 0 -- anything else means the base was unfrozen.
  ``model_settings`` the kev settings handed to ``build_model``. Only lora, head_dim, lora_targets,
                     option_isolation, special_embeddings and weights_dtype come from the run
                     (``sckevtype/train/train.py:59``); the rest are kev defaults, so this block is
                     the architecture and must be identical.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

#: Keys whose value is *expected* to differ between the frozen run and a seed draw.
ALLOWED = ("seed", "out")

#: Top-level blocks compared verbatim (scalars or nested dicts).
SCALAR_BLOCKS = ("suite_sha256", "base_revision")
DICT_BLOCKS = ("trainable", "model_settings")


def load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        print(f"MISSING {path} -- cannot compare a configuration that is not there")
        raise SystemExit(2)
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as error:
        print(f"{path} is not JSON: {error}")
        raise SystemExit(2)


def walk(prefix: str, frozen: Any, new: Any, problems: list[str]) -> int:
    """Compare two nested values, recording every leaf that differs. Returns leaves compared."""
    if isinstance(frozen, dict) and isinstance(new, dict):
        compared = 0
        for key in sorted(set(frozen) | set(new)):
            if key not in frozen:
                problems.append(f"{prefix}.{key}: absent from the frozen config, present as {new[key]!r}")
            elif key not in new:
                problems.append(f"{prefix}.{key}: {frozen[key]!r} in the frozen config, absent here")
            else:
                compared += walk(f"{prefix}.{key}", frozen[key], new[key], problems)
        return compared
    if isinstance(frozen, list) and isinstance(new, list):
        if frozen != new:
            problems.append(f"{prefix}: {frozen!r} != {new!r}")
        return max(len(frozen), 1)
    if frozen != new:
        problems.append(f"{prefix}: {frozen!r} != {new!r}")
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("frozen", type=Path, help="the frozen run's training_config.json")
    parser.add_argument("new", type=Path, help="the seed run's training_config.json")
    args = parser.parse_args(argv)

    frozen, new = load(args.frozen), load(args.new)
    problems: list[str] = []
    allowed_moved: list[str] = []
    leaves = 0

    # -- args: the flags, the defaults and the budgets the run recorded.
    fa, na = frozen.get("args") or {}, new.get("args") or {}
    if not fa or not na:
        problems.append(f"an 'args' block is absent or empty (frozen {bool(fa)}, new {bool(na)})")
    for key in sorted(set(fa) | set(na)):
        if key in ALLOWED:
            if fa.get(key) != na.get(key):
                allowed_moved.append(f"args.{key}: {fa.get(key)!r} -> {na.get(key)!r}")
            elif key not in fa or key not in na:
                problems.append(f"args.{key} is missing on one side ({fa.get(key)!r} vs {na.get(key)!r})")
            continue
        if key not in fa:
            problems.append(f"args.{key}: absent from the frozen config, present here as {na[key]!r}")
        elif key not in na:
            problems.append(f"args.{key}: {fa[key]!r} in the frozen config, absent here")
        else:
            leaves += 1
            if fa[key] != na[key]:
                problems.append(f"args.{key}: {fa[key]!r} != {na[key]!r}")

    for block in SCALAR_BLOCKS:
        leaves += 1
        if frozen.get(block) != new.get(block):
            problems.append(f"{block}: {frozen.get(block)!r} != {new.get(block)!r}")

    for block in DICT_BLOCKS:
        if not isinstance(frozen.get(block), dict) or not isinstance(new.get(block), dict):
            problems.append(f"{block}: absent or not a mapping (frozen {type(frozen.get(block)).__name__}, "
                            f"new {type(new.get(block)).__name__})")
            continue
        leaves += walk(block, frozen[block], new[block], problems)

    # A draw that did not move the seed, or that reused the frozen run's directory, is not a new draw.
    if fa.get("seed") == na.get("seed"):
        problems.append(f"args.seed did not change ({fa.get('seed')!r} on both sides): this is the frozen "
                        f"draw, not a new one")
    if fa.get("out") == na.get("out"):
        problems.append(f"args.out did not change ({fa.get('out')!r} on both sides): the run wrote into the "
                        f"frozen run's directory")

    print(f"  frozen : {args.frozen}")
    print(f"  new    : {args.new}")
    print(f"  compared {leaves} leaf value(s) across args, {' and '.join(SCALAR_BLOCKS)}, "
          f"{' and '.join(DICT_BLOCKS)}")
    for line in allowed_moved:
        print(f"  allowed difference -- {line}")
    if problems:
        print()
        print("  NOT THE FROZEN CONFIGURATION:")
        for problem in problems:
            print(f"    - {problem}")
        print("  (a run that differs in anything but seed and out is a different configuration, and a")
        print("   difference it shows against the frozen run is not seed noise)")
        return 1
    print("  PRE-FLIGHT OK -- identical to the frozen configuration except the two allowed keys")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
