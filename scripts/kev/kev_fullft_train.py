#!/usr/bin/env python3
"""LoRA fine-tuning on a **full-weight** kev checkpoint (`jaredpalmer/kev-27b`) — the missing warm start.

Why this file exists
--------------------
Every other rung of the ladder is `base + LoRA adapter + head.pt`, so `sckevtype.train.train --init_from`
warm-starts it through kev's own ``Checkpoint.warm_start`` (the released adapter tensors + the released
pointer head). `jaredpalmer/kev-27b` is a **full finetune**: 11 model shards, no ``adapter_config.json``,
``head.pt`` declaring ``weights: "full"`` and ``lora = 0``. Two things follow, and both are structural
rather than a missing flag:

* ``Checkpoint.warm_start`` cannot load it — ``kev.checkpoint`` reads ``adapter_config.json``
  unconditionally (``kev/kev/checkpoint.py:193``);
* ``DecisionModel.__init__`` ends with ``self.to(device)``, so the trainer's own build puts all 51.2 GB
  of bf16 weights on ONE card before anything can be sharded. On a 40 GB A100 that is fatal before the
  first step, whatever ``--gpus`` says.

What this file is, and what it is not
-------------------------------------
It is **not a second trainer**. It calls ``sckevtype.train.train.run()`` — the same function the
0.8B/4B/9B rungs ran — and replaces exactly one step of that function's pipeline: the model build
(``sckevtype.train.model.build_model``). Everything after it is that function's own code path, unchanged:
``verify_frozen``, ``shard.shard_model`` (the balanced-caps placement over N cards, the same policy the
9B's completed epoch used), the AdamW/OneCycleLR setup, ``budgeted_steps``/``counted_steps`` with
``--token_budget``, ``kev.train.batch_loss``, the per-card ``DevicePeaks``, and the three artifacts
(``training_config.json``, ``provenance.json``, ``run.log.jsonl``). The step count, the records and the
loss are therefore the trainer's, not this file's.

The replacement build, in order
-------------------------------
1. **the released checkpoint's own weights are the backbone** — loaded from its snapshot directory on the
   HOST (transformers 5 materialises one shard at a time, so peak host RAM is the model and not twice it).
   There is no base repo to load and nothing to merge: a full finetune *is* the model.
2. **a fresh LoRA adapter** is attached with ``kev``'s own recipe (``lora``/``lora_targets``/dropout from
   kev's parser defaults, the same ones ``--init_from`` runs use).
3. **the checkpoint's own pointer head** is loaded from ``head.pt`` — the readout the 27B's published
   numbers were produced with, not a fresh one. Its temperature comes with it; ``PointerHead`` skips the
   temperature while training, which is what every other rung does too.
4. ``attn="sdpa"`` is passed **explicitly**, because ``DecisionModel`` derives the attention backend from
   the device string it was *constructed* with and this build is on ``cpu`` (the 27B does not fit one
   card). This is the same reason ``scripts/kev/kev_fullft.py`` forces it on the load path.
5. **the pointer head is moved to ``cuda:0``** before returning: ``shard.shard_model`` dispatches only
   ``model.lm``, and its own guard refuses a head that is not on the card the inputs are built on — the
   plan's main device, which is ``cuda:0`` (the embedding is filled first).

What is deliberately NOT done
-----------------------------
* No new trainer, no new step loop, no new loss, no new memory measurement.
* No change to ``sckevtype/`` — the seam is the module attribute ``model_module.build_model`` that
  ``train.run`` calls, patched for the duration of one call and restored afterwards.
* No silent precision change: the 27B's shards are bf16 on disk and a fp32 load would be ~102 GB, so
  ``--weights_dtype bf16`` is required rather than defaulted, and the checkpoint's own
  ``weights_dtype`` is checked against it.

Usage (on the cluster, from the repo root, under ``venv-train``):

    python scripts/kev/kev_fullft_train.py --full_weight jaredpalmer/kev-27b \\
        --suite artifacts/suites/real-v2 --gpus 4 --checkpointing 1 \\
        --token_budget 8192 --steps 20 --seed 1 --out runs/smoke-27b-tb8192

``PREFLIGHT_ONLY=1`` runs every check below and stops before the model is built (no GPU, seconds).
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:                       # a script under scripts/ does not put the repo on sys.path
    sys.path.insert(0, str(ROOT))

from sckevtype.train import model as M             # noqa: E402
from sckevtype.train import shard as shard_module  # noqa: E402
from sckevtype.train import train as T             # noqa: E402

#: ``scripts/kev/kev_fullft.py`` — the repo's one placement/loader policy (``checkpoint_kind``,
#: ``load_dtype``, ``balanced_caps``, ``verify_placement``). Loaded through ``shard``'s own by-path
#: loader, so this file and a sharded training run share one module object rather than two copies.
F = shard_module.placement()

WARM_START_FILE = "warm_start.json"


class FullWeightWarmStartError(RuntimeError):
    """The released checkpoint cannot be warm-started into a trainable model. Never a silent guess."""


# --------------------------------------------------------------------------- the checkpoint


def resolve(source: str) -> Any:
    """``kev.checkpoint.Checkpoint`` for ``source``, refused unless it really is a full-weight checkpoint."""
    from kev.checkpoint import Checkpoint

    ck = Checkpoint(source)
    kind = F.checkpoint_kind(ck)
    if kind != F.FULL:
        raise FullWeightWarmStartError(
            f"{source} resolved to {ck.path}, which is a {kind!r} checkpoint (it ships "
            f"{F.ADAPTER_FILE}); this path is for a FULL-WEIGHT checkpoint only — a LoRA checkpoint "
            f"warm-starts through `sckevtype.train.train --init_from`, which is kev's own loader and "
            f"already the lane every other rung used")
    return ck


def declared_shape(ck: Any) -> dict[str, Any]:
    """The checkpoint's own statement about the model it is, from ``head.pt``."""
    meta = ck.meta
    return {"base": meta.base, "base_revision": meta.base_revision, "lora": int(meta.lora or 0),
            "head_dim": int(meta.head_dim), "option_isolation": bool(meta.option_isolation),
            "special_embeddings": bool(meta.special_embeddings), "weights_dtype": meta.weights_dtype,
            "temperature": float(meta.temperature), "weights": (meta.extra or {}).get("weights"),
            "head_tensors": len(meta.head or {})}


def check_shape(ck: Any, settings: dict[str, Any], weights_dtype: str) -> dict[str, Any]:
    """The run must build the shape the released checkpoint declares, or refuse by name.

    ``lora`` is *not* comparable and is not compared: the source has no adapter (``lora = 0``) and this
    run attaches a fresh one at kev's own rank — that is the whole point of the path. The three fields
    that are compared decide the pointer head's shape and the encoder's behaviour, and a mismatch would
    load a head into the wrong architecture.
    """
    declared = declared_shape(ck)
    wanted = {"head_dim": int(settings["head_dim"]),
              "option_isolation": bool(settings["option_isolation"]),
              "special_embeddings": bool(settings["special_embeddings"])}
    wrong = {key: (declared[key], value) for key, value in wanted.items() if declared[key] != value}
    if wrong:
        shown = ", ".join(f"{key}: checkpoint {was!r} vs run {now!r}" for key, (was, now) in wrong.items())
        raise FullWeightWarmStartError(
            f"{ck.requested}: the run would build a model the checkpoint's head.pt does not describe "
            f"({shown}); the pointer head would be loaded into a different architecture")
    if weights_dtype != "bf16":
        raise FullWeightWarmStartError(
            f"--weights_dtype {weights_dtype}: this checkpoint's shards are bf16 on disk "
            f"({declared['weights_dtype']!r} declared) and the 27B is 51.2 GB at bf16 against 102 GB at "
            f"fp32, which four 40 GB cards cannot hold. Pass --weights_dtype bf16 (the ladder's own "
            f"precision decision for this arm) rather than loading a model that cannot fit")
    return declared


def warm_start_record(ck: Any, declared: dict[str, Any]) -> dict[str, Any]:
    """What the run started from, in the shape a reader can re-derive it from.

    The snapshot directory's name *is* the revision (huggingface_hub names it after the commit), and it
    is quoted as ``source_revision`` so the pin is not inferred. The 11 weight shards are 51.2 GB: their
    *sizes* and the index's own ``total_size``/``total_parameters`` are recorded rather than a 51 GB
    re-read for a hash, and ``head.pt`` — 10 MB, the file that decides every number — is hashed.
    """
    path = Path(ck.path)
    shards = sorted(path.glob("model-*.safetensors"))
    index = path / "model.safetensors.index.json"
    record: dict[str, Any] = {
        "init_from": ck.requested,
        "resolved": str(path),
        "source_revision": path.name if len(path.name) == 40 else None,
        "declared": declared,
        "shards": {"count": len(shards), "bytes": sum(p.stat().st_size for p in shards)},
        "head_sha256": _sha256(path / "head.pt"),
        "index_sha256": _sha256(index) if index.is_file() else None,
        "adapter_config": (path / F.ADAPTER_FILE).is_file(),
    }
    if index.is_file():
        record["index"] = json.loads(index.read_text(encoding="utf-8")).get("metadata")
    return record


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------- the build


def build_full_weight(ck: Any, settings: dict[str, Any], tokenizer: Any, *, attn: str = "sdpa") -> Any:
    """``kev.model.DecisionModel`` for a full-weight checkpoint, built on the HOST with a fresh adapter.

    ``sckevtype.train.model.build_model``'s own two rules are kept — the text tower is pinned through
    ``_text_tower_loader`` when the config is composite (it is not for this checkpoint: ``qwen3_5_text``
    has no ``vision_config``) and the result is checked by ``verify_text_only``/``verify_frozen`` — and
    the constructor is ``kev``'s, called rather than copied.
    """
    import kev.model as km
    import torch
    from kev.checkpoint import LoadOptions

    declared = declared_shape(ck)
    dtype = F.load_dtype(ck, LoadOptions())
    inner = None
    if settings.get("text_only", True):
        from transformers import AutoConfig

        inner = M.text_config(AutoConfig.from_pretrained(str(ck.path)))

    print(f"build: full-weight backbone from {ck.path} on cpu "
          f"(dtype {F.DTYPE_NAMES.get(dtype, dtype)}, attn {attn}, lora {settings['lora']} "
          f"targets {settings['lora_targets']})", file=sys.stderr, flush=True)
    loader = M._text_tower_loader(inner) if inner is not None else contextlib.nullcontext()
    started = time.time()
    with loader:
        model = km.DecisionModel(str(ck.path), tokenizer, "cpu", lora=settings["lora"], revision=None, attn=attn,
                                 head_dim=settings["head_dim"], lora_targets=settings["lora_targets"],
                                 option_isolation=bool(settings["option_isolation"]),
                                 special_embeddings=bool(settings["special_embeddings"]), dtype=dtype)
    print(f"build: backbone + adapter on cpu in {time.time() - started:.1f} s", file=sys.stderr, flush=True)

    # The released pointer head: `Checkpoint.load`'s own two lines (weights from head.pt, temperature
    # from head.pt), minus `.eval()` — this is a training run and `PointerHead` skips the temperature
    # division while `self.training` is true, exactly as it does for the warm-started LoRA rungs.
    missing, unexpected = model.head.load_state_dict(ck.meta.head, strict=False)
    if missing or unexpected:
        raise FullWeightWarmStartError(
            f"{ck.requested}: the pointer head did not load cleanly — missing {sorted(missing)}, "
            f"unexpected {sorted(unexpected)}")
    model.head.temperature = float(ck.meta.temperature)
    print(f"build: pointer head loaded ({declared['head_tensors']} tensors, temperature "
          f"{model.head.temperature})", file=sys.stderr, flush=True)

    M.verify_text_only(model)
    M.verify_frozen(model)
    # `shard.shard_model` dispatches only `model.lm` and its guard refuses a head that is not on the
    # card the inputs are built on (the plan's main device — the embedding is filled first, so cuda:0).
    model.head.to("cuda:0")
    print(f"build: head on {next(model.head.parameters()).device} (shard_model dispatches only the "
          f"backbone)", file=sys.stderr, flush=True)
    return model


# --------------------------------------------------------------------------- arguments


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """This driver's flags. Everything that decides *what is trained* is ``sckevtype.train.train``'s."""
    parser = argparse.ArgumentParser(prog="kev_fullft_train",
                                     description="LoRA fine-tuning on a full-weight kev checkpoint, through "
                                                 "sckevtype.train.train's own pipeline.")
    parser.add_argument("--full_weight", required=True,
                        help="the full-weight checkpoint to warm-start from (a hub id like "
                             "jaredpalmer/kev-27b, optionally @revision, or a local directory)")
    parser.add_argument("--suite", required=True, help="suite directory written by sckevtype.train.suite")
    parser.add_argument("--out", required=True, help="run directory; it must not exist yet")
    parser.add_argument("--gpus", type=int, default=4, help="cards to shard the frozen backbone over (>= 2)")
    parser.add_argument("--steps", type=int, default=20, help="optimizer steps (0 = one epoch)")
    parser.add_argument("--token_budget", type=int, default=0,
                        help="padded tokens one forward pass may cost (0 = off)")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--weights_dtype", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--checkpointing", type=int, choices=[0, 1], default=1)
    parser.add_argument("--profile_memory", type=int, choices=[0, 1], default=0)
    parser.add_argument("--checkpoint_every", type=int, default=None,
                        help="optimizer steps between resumable checkpoints (forwarded to the trainer only "
                             "when given; its own default is 200, or SCKEVTYPE_CHECKPOINT_EVERY). A 27B "
                             "epoch is the longest run in the fleet, so the checkpoints are on by default")
    parser.add_argument("--resume", action="store_true",
                        help="continue the run in --out from its newest checkpoint-step-<n>.pt instead of "
                             "starting over. A 24-hour epoch that a reboot interrupted costs minutes to "
                             "pick up this way; without the flag a populated --out is refused as before")
    args = parser.parse_args(argv)
    if args.gpus < 2:
        parser.error(f"--gpus {args.gpus}: this path builds the backbone on the host because it does not "
                     f"fit one card, and only `shard.shard_model` puts it back on GPUs — with fewer than "
                     f"two cards the run would train on the CPU")
    if args.device != "cuda":
        parser.error("--device cuda is the only path here (the trainer's bf16 autocast and the shard both "
                     "need it)")
    return args


def train_argv(args: argparse.Namespace, ck: Any, base_revision: str) -> list[str]:
    """The ``sckevtype.train.train`` command line for this run: its own flags, ``--base`` = the snapshot.

    ``--base`` is the checkpoint's *snapshot directory* rather than ``Qwen/Qwen3.8-27B``: the released
    weights ARE that base after its finetune, and the base repo is not in this cluster's HF cache (this
    path must work offline, ``HF_HUB_OFFLINE=1``). ``--base_revision`` carries the base revision the
    checkpoint's own ``head.pt`` names, which is what ``provenance.json`` will record as the base pin.

    No program name: ``train.parse_args`` hands its list straight to argparse (unlike ``kev.train``, which
    reads ``sys.argv``), so a leading token would be parsed as a positional argument.
    """
    argv = ["--suite", args.suite, "--base", str(ck.path), "--base_revision", base_revision,
            "--dtype", args.dtype, "--weights_dtype", args.weights_dtype,
            "--checkpointing", str(args.checkpointing), "--device", args.device,
            "--gpus", str(args.gpus), "--token_budget", str(args.token_budget),
            "--seed", str(args.seed), "--steps", str(args.steps), "--out", args.out]
    if args.profile_memory:
        argv += ["--profile_memory", str(args.profile_memory)]
    # Forwarded only when asked for, so a command line that does not name them produces the argv it
    # always did: the trainer's own checkpoint default (200 steps, or the environment's) applies
    # either way, and --resume is this driver's way to reach the trainer's resume path.
    if args.checkpoint_every is not None:
        argv += ["--checkpoint_every", str(args.checkpoint_every)]
    if args.resume:
        argv += ["--resume"]
    return argv


def preflight(train_args: argparse.Namespace) -> None:
    """Everything ``train.run`` does *before* it builds the model — run here, on the login node, CPU only.

    This is the trainer's own block, not a re-statement of it: the suite manifest and its recorded budget,
    the training partition, and ``verify_state_budget`` over all of it with **this run's** tokenizer. That
    last one is the check most likely to refuse this arm, and it is the one worth paying for before a
    queue slot: the suite was frozen against ``Qwen/Qwen3.5-0.8B-Base``'s tokenizer, and a run whose
    tokenizer encodes these records differently would stop here (by name) instead of on a GPU node.
    """
    import kev.train as kt
    from kev.model import load_tokenizer, training_context
    from kev.suite import read_manifest

    ns = T.kev_namespace(train_args, str(Path(train_args.out)))
    manifest = read_manifest(ns.suite)
    revision = kt.pinned_revision(ns, manifest)
    tokenizer = load_tokenizer(ns.base, revision=revision)
    requests = kt.training_requests(ns, tokenizer, manifest, manifest["holdout_sources"])
    context = T.suite_budget(ns.suite)
    T.verify_recorded_budget(context, ns.suite)
    limits = training_context(ns.max_state)
    measured = T.verify_state_budget(tokenizer, requests, limits, ns.suite)
    print(f"preflight: {len(requests)} training record(s) fit the {limits['max_state']}-token budget "
          f"(longest state {measured['max']}, median {measured['median']}); kev is handed "
          f"max_state {ns.max_state} (row limit {limits['max_branch']}, packed {limits['max_packed']})",
          flush=True)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ck = resolve(args.full_weight)
    declared = declared_shape(ck)
    revision = declared["base_revision"] or ""
    if not revision:
        raise FullWeightWarmStartError(
            f"{ck.requested}: head.pt records no base_revision, so the base pin of this run's weights is "
            f"unknown; refusing rather than writing a run whose provenance cannot be re-derived")

    train_args = T.parse_args(train_argv(args, ck, revision))
    out_dir = Path(train_args.out)
    # A resume's directory has to exist; the trainer checks that it holds a readable checkpoint whose
    # recipe matches this run *before* the 51.2 GB build below, so a resume that cannot be one costs
    # seconds rather than a model load. Without --resume the refusal is the one this path always had.
    if out_dir.exists() and not args.resume:
        print(f"refusing to overwrite an existing run: {out_dir}", file=sys.stderr)
        return 1
    settings = M.lora_config({key: getattr(_kev_defaults(), key) for key in T.MODEL_SETTINGS})
    check_shape(ck, settings, args.weights_dtype)

    record = warm_start_record(ck, declared)
    print("warm start: " + json.dumps({k: record[k] for k in
                                       ("init_from", "resolved", "source_revision", "shards", "head_sha256")}),
          file=sys.stderr, flush=True)

    if os.environ.get("PREFLIGHT_ONLY") == "1":
        preflight(train_args)
        print("PREFLIGHT_ONLY=1: the checkpoint resolved, is full-weight, declares the shape this run "
              "would build, and the suite fits the budget under this run's tokenizer; the model was not "
              "built and nothing was trained.", flush=True)
        print("PREFLIGHT OK", flush=True)
        return 0

    # The one seam: `train.run` calls `model_module.build_model`, so replacing that attribute for the
    # duration of one call is the whole of this driver's difference from the ladder's own lane.
    original = M.build_model
    calls: list[int] = []

    def builder(base: str, rev: str | None, settings: dict[str, Any], *, text_only: bool = True,
                device: str = "cpu", tokenizer: Any = None) -> Any:
        calls.append(1)
        if len(calls) > 1:
            raise FullWeightWarmStartError(f"build_model was called {len(calls)} times; this path replaces "
                                           f"exactly one build and will not guess which one is which")
        return build_full_weight(ck, {**settings, "text_only": text_only}, tokenizer)

    started = time.time()
    M.build_model = builder
    try:
        provenance = T.run(train_args)
    finally:
        M.build_model = original
    wall = time.time() - started
    if not calls:
        raise FullWeightWarmStartError("the trainer never built a model — the seam this driver relies on "
                                       "has moved; nothing was trained")

    record["trainer_wall_s"] = wall
    record["run"] = str(out_dir)
    (out_dir / WARM_START_FILE).write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")

    summary = {
        "run": str(out_dir), "token_budget": args.token_budget, "steps": provenance["optimizer_steps"],
        "stopped_early": provenance["stopped_early"],
        "peak_device_bytes": provenance["peak_device_bytes"],
        "worst_card_gib": round(provenance["peak_device_bytes"] / 2**30, 3),
        "peak_bytes_per_device": {k: round(v / 2**30, 3) for k, v in
                                  (provenance["epochs"][-1].get("peak_bytes_per_device") or {}).items()},
        "token_budget_stats": provenance.get("token_budget"),
        "trainer_wall_s": round(wall, 1),
    }
    print("SUMMARY " + json.dumps(summary), flush=True)
    return 0


def _kev_defaults() -> argparse.Namespace:
    """kev's own parser defaults as a namespace, so the shape check compares against the real recipe."""
    import argparse as _argparse

    return _argparse.Namespace(**M.kev_defaults())


if __name__ == "__main__":
    raise SystemExit(main())
