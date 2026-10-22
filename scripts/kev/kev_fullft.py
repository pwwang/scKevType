#!/usr/bin/env python3
"""A **full-weight** Kev checkpoint (no adapter) through the same encoder and pointer head as the LoRA arms.

Why this module exists
----------------------
``kev.checkpoint.Checkpoint.load`` is a LoRA loader. It reads ``adapter_config.json`` unconditionally
(``kev/kev/checkpoint.py:193``) and then builds a ``PeftModel`` from the checkpoint directory
(``:196``), so a full-weight checkpoint fails before any weight is touched::

    FileNotFoundError: .../adapter_config.json

``jaredpalmer/kev-27b`` is that checkpoint: ``head.pt`` declares ``weights: "full"`` (a field
``kev.checkpoint.Meta`` does not know, so it round-trips through ``Meta.extra``), ``lora = 0``, and the
repo ships 11 ``model-*.safetensors`` shards instead of ``adapter_model.safetensors``. It is the only
arm of the ladder that is not ``base + LoRA adapter + head.pt``.

Why the change lives here and not in ``kev/``
---------------------------------------------
``kev/`` is a pinned upstream reference that this repo reads and never edits — the rule is written in
three places: ``pyproject.toml`` (``[tool.ruff] extend-exclude = ["kev", ...]``, "a pinned upstream
reference (commit 2ba8166) that this repo reads and never edits"), ``kev/AGENTS.md`` ("Any change here
must keep the parity tests in tests/test_model.py ... and tests/test_mlx.py passing"), and the project
skill ("Keep ``kev/`` read-only"). A fork would also put five working arms at risk: ``kev-0.5b``,
``kev-0.6b``, ``kev-4b``, ``kev-8b`` and the two already-run rungs (``0.8b``, ``9b``) all depend on
``Checkpoint.load`` behaving exactly as it does today.

So this is an **additive** path *beside* the vendored package, and nothing here is imported unless a
checkpoint really is a full finetune (``kev_probe.KevPredictor`` imports this module lazily and only
calls into it for ``kind == "full"``). The LoRA arms construct ``kev.predictors.LocalPredictor``
exactly as they did before; ``--gpus != 1`` on a LoRA arm is refused rather than silently ignored.

What is reused, verbatim
------------------------
Only the *loading* differs. The scoring path is the vendor's own code:

* ``kev.model.DecisionModel`` is constructed by its own ``__init__`` — ``head_dim``,
  ``option_isolation``, ``pad_id``, hybrid detection and the pointer head are the same lines the LoRA
  path runs, with ``lora=None`` because there is no adapter to apply;
* :class:`FullFinetunePredictor` subclasses ``kev.predictors.LocalPredictor`` and inherits ``__call__``,
  so ``encode`` / ``forward`` / ``softmax`` / ``question_keys`` are literally the functions the LoRA
  rungs use.

Precision is not a flag you can quietly turn down here. The loaded dtype is resolved by the *same rule*
``kev.checkpoint._load_torch`` applies — ``opts.dtype`` unless the checkpoint declares
``weights_dtype == "bf16"``, in which case bf16 (``kev-27b`` declares bf16, and its shards are bf16 on
disk: 51,279,886,109 bytes over ~25.6e9 parameters). ``LocalPredictor``'s TF32 / fused-SDPA switches
are reproduced line for line and then **asserted**, because a silent precision change would invalidate
the comparison with every other rung.

Multi-GPU
---------
The vendored package has no multi-GPU support at all (``device_map``, ``DataParallel``, ``shard``:
zero hits under ``kev/kev/``; ``torch_device=str(device)`` pins one card), and a 27B in bf16 is ~51.3 GB
against one A100-40G's ~40 GB. The sharding here is ``accelerate`` — already a hard dependency of both
``kev`` (``kev/pyproject.toml``) and this repo (``pyproject.toml`` ``[project.optional-dependencies]
train``), so nothing new is installed:

1. the model is built on CPU (``DecisionModel(..., device="cpu")``), which transformers 5 loads
   shard-by-shard — peak host RAM is the model size, not twice it;
2. :func:`plan_device_map` budgets each visible GPU from ``torch.cuda.mem_get_info`` minus a reserve,
   caps each card at its balanced share when more than one card is asked for (:func:`caps_for`) — a
   free-memory budget alone packs the first card solid, measured: the 27B's 51.2 GB came out as 38.3 GB
   on ``cuda:0`` of four, with ``cuda:2``/``cuda:3`` empty, and the forward then died on that card's
   last gigabyte — and **refuses by name** when the weights do not fit, before ``accelerate`` is asked
   anything (accelerate 1.15's ``infer_auto_device_map`` contains no ``raise`` at all: an over-budget
   model is silently left on CPU, which would turn a 40-minute pass into a day-long one);
3. ``accelerate.dispatch_model`` moves the decoder layers (``no_split_module_classes`` = the model's own
   layer class, so a layer is never split across cards) and :func:`verify_placement` re-reads the result
   off the live parameters and refuses if any parameter is not on one of the requested GPUs.

Numerically, sharding is a placement, not a precision: every matmul still runs in the checkpoint's own
dtype on a single device, so the answers are the same as the single-GPU path (the cluster job records
``hf_device_map`` and the per-device parameter bytes so the placement is auditable).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from kev.checkpoint import Checkpoint, LoadOptions
from kev.model import DecisionModel, load_tokenizer
from kev.predictors import LocalPredictor
from kev.suite import CONTEXT

#: ``head.pt``'s own statement about what kind of checkpoint this is. ``kev-9b`` (current revision)
#: writes ``"lora"``; ``kev-27b`` writes ``"full"``; ``kev-0.8b`` predates the field and writes nothing,
#: which is why the field is a *cross-check* on the structural fact (``adapter_config.json``) and never
#: the sole discriminator.
FULL, LORA = "full", "lora"
DECLARED = (FULL, LORA)

#: ``adapter_config.json`` is what ``kev.checkpoint`` reads unconditionally; its presence *is* the
#: structural definition of a LoRA checkpoint here.
ADAPTER_FILE = "adapter_config.json"

#: The vendor's own spelling for a dtype (``LoadOptions`` and the ``KEV_DTYPE`` variable), so a report
#: says ``bf16`` where the rest of the project says ``bf16`` and not ``torch.bfloat16``.
DTYPE_NAMES = {torch.bfloat16: "bf16", torch.float16: "fp16", torch.float32: "fp32"}

#: Left free on each GPU for the forward pass's activations, the attention masks and the allocator's
#: fragmentation. On one card this is the whole weight budget (``free - reserve``); on more than one the
#: per-card caps :func:`caps_for` computes are the weight budget and this is the floor under them.
RESERVE_BYTES = 2 * 1024**3


class FullWeightError(RuntimeError):
    """A full-weight checkpoint cannot be loaded as this run needs it."""


# --------------------------------------------------------------------------- which loader

def checkpoint_kind(checkpoint: Checkpoint) -> str:
    """``"full"`` or ``"lora"``, from the two facts that must agree.

    ``adapter_config.json`` is the structural fact — it is the file ``Checkpoint.load`` reads
    unconditionally, so a checkpoint without it cannot be loaded by the vendored loader whatever
    ``head.pt`` says. ``head.pt``'s ``weights`` field is the checkpoint's own statement, and it is
    *newer* than the oldest published arm (``kev-0.8b`` carries no such field), so it can only ever
    confirm or contradict — never decide alone. A contradiction is refused rather than resolved: a
    checkpoint that says ``full`` while shipping an adapter is a checkpoint whose provenance is
    unknown, and guessing which half is right is how a ladder ends up comparing a model with itself.
    """
    has_adapter = (Path(checkpoint.path) / ADAPTER_FILE).is_file()
    declared = (checkpoint.meta.extra or {}).get("weights")
    if declared is not None and declared not in DECLARED:
        raise FullWeightError(f"{checkpoint.path}: head.pt says weights={declared!r}, which is neither "
                              f"{FULL!r} nor {LORA!r}; refusing to guess which loader to use")
    if declared == FULL and has_adapter:
        raise FullWeightError(f"{checkpoint.path}: head.pt says weights='full' but {ADAPTER_FILE} is "
                              f"present; one of the two is wrong and the loaders disagree about what "
                              f"would be loaded")
    if declared == LORA and not has_adapter:
        raise FullWeightError(f"{checkpoint.path}: head.pt says weights='lora' but {ADAPTER_FILE} is "
                              f"absent; the adapter the checkpoint claims cannot be loaded")
    return LORA if has_adapter else FULL


def tokenizer_for(checkpoint: Checkpoint):
    """The checkpoint's own tokenizer when it ships one, else the base's — the same choice ``Checkpoint.load``
    makes only because a LoRA repo *is* a directory of deltas and has to borrow the base's.

    A full finetune publishes the tokenizer it was trained with (``kev-27b``: ``tokenizer.json``,
    ``tokenizer_config.json``, ``chat_template.jinja``), so nothing about the arm depends on the base
    repo being on disk. Verified equal to ``Qwen/Qwen3.8-27B``'s own tokenizer for this checkpoint: same
    ``model.vocab`` and the same ``added_tokens`` (see ``scripts/kev/kev_fullft_selftest.py``).
    """
    if (Path(checkpoint.path) / "tokenizer.json").is_file():
        return load_tokenizer(checkpoint.path)
    return load_tokenizer(checkpoint.meta.base, revision=checkpoint.meta.base_revision)


def load_dtype(checkpoint: Checkpoint, opts: LoadOptions) -> torch.dtype:
    """``kev.checkpoint._load_torch``'s dtype rule, applied to a checkpoint with no adapter to merge.

    The rule matters more than it looks: ``kev-27b``'s ``head.pt`` declares ``weights_dtype: "bf16"``,
    so the vendored loader would force bf16 regardless of ``--dtype`` — and the LoRA arms end up bf16
    too, after an fp32 merge that is cast at the end (``m.lm = m.lm.to(dtype)``). Resolving it the same
    way is what makes the 27B's readout comparable with the 0.8B's and the 9B's.
    """
    dtype = opts.dtype or torch.float32
    if checkpoint.meta.weights_dtype == "bf16":
        dtype = torch.bfloat16
    return dtype


# --------------------------------------------------------------------------- where it goes

def parameter_bytes(module) -> int:
    """Bytes of parameters (not buffers): what has to fit on the GPUs."""
    return sum(p.numel() * p.element_size() for p in module.parameters())


def no_split_classes(module) -> list[str]:
    """The decoder-layer class name, read off the model rather than hardcoded.

    ``accelerate`` splits at module boundaries, and a boundary *inside* a decoder layer is legal but
    needless: it turns one layer's forward into several device hops. Deriving the name from the live
    module means the same code works for ``Qwen3_5DecoderLayer`` (this checkpoint's
    ``Qwen3_5TextModel``) and for any other backbone a later rung brings.
    """
    layers = getattr(module, "layers", None)
    if layers is None or not len(layers):
        return []
    return [type(layers[0]).__name__]


def unit_bytes(module) -> int:
    """The largest indivisible chunk ``accelerate``'s fill will reserve room for on every card.

    ``infer_auto_device_map`` keeps one such chunk free on each card it is filling
    (``current_max_size = max_memory[device] - max_layer_size``), so a per-card weight budget has to
    carry it twice: once for the room that is reserved, and once for the room the reservation is meant
    to leave usable. Read off the live module rather than guessed: a leaf is indivisible because it has
    no children to split at, and a no-split class is indivisible because :func:`no_split_classes` asked
    ``accelerate`` to treat it that way.
    """
    classes = set(no_split_classes(module))
    largest = 0
    for _, child in module.named_modules():
        leaf = len(list(child.children())) == 0
        if leaf or type(child).__name__ in classes:
            largest = max(largest, parameter_bytes(child))
    return largest


def largest_layer_bytes(module) -> int:
    """The biggest decoder layer: the unit the greedy fill moves to the next card when it does not fit."""
    layers = getattr(module, "layers", None)
    if layers is None or not len(layers):
        return 0
    return max(parameter_bytes(layer) for layer in layers)


def balanced_caps(need: int, n_devices: int, unit: int, layer: int) -> dict[int, int]:
    """The per-card weight budget for a sharded run: ``max(ceil(need / n) + layer, 2 * unit)`` each.

    One implementation, in the module both placement paths already share, because two would drift:
    ``sckevtype.train.shard`` (a training run) and :func:`load_full_finetune` (a load) both place weights
    with this rule, and a card that got nothing looks exactly like a card that was never needed.

    Why per card rather than one scalar: the fill keeps the largest remaining indivisible unit free on
    the card it fills **first** — the one that has to be able to take back a layer offloaded to the cpu
    (``accelerate`` 1.15: ``main_devices = [gpus[0], "cpu"]``) — and that reservation shrinks as the fill
    proceeds, from the embedding to a decoder layer. A single scalar therefore leaves the later cards a
    bigger effective budget than the first one got, and when it is generous enough the last card gets
    nothing. Measured on this cluster before the caps existed: a free-memory budget put the 4B's
    four-card plan on two cards and the 9B's on three.

    The two terms, both read off the fill's own arithmetic:

    * the fill moves in whole layers, so a card's fill ends within one *layer* of its cap — and the first
      card's cap is effectively one layer lower, because it also carries the reservation — so
      ``ceil(need / n) + layer`` is a card that holds its share;
    * a card must be able to *place* the largest indivisible unit, not merely hold it: the fill requires
      ``used + unit <= cap - max_layer_size``, and while the embedding is unplaced the reservation is
      that same unit — so every card also gets a floor of ``2 * unit``.

    The caps this computes for ``jaredpalmer/kev-27b`` are 13,578,846,400 bytes per card (``share`` is
    12,812,300,032; the largest layer is 766,546,368; the ``2 * unit`` floor is 5,085,593,600), and the
    fill then spreads all 51,249,200,128 bytes of weights over the four cards as 11.59 / 12.05 / 12.05 /
    12.03 GiB — the property ``tests/train/test_fullft_load.py`` pins, because before the caps existed
    the same weights came out as 38.3 GB on ``cuda:0`` with ``cuda:2``/``cuda:3`` empty, and the forward
    died on that card's last gigabyte.
    """
    if n_devices < 1:
        raise FullWeightError(f"a per-card cap needs at least one card; got {n_devices}")
    share = -(-need // n_devices)                      # ceil(need / n)
    return {index: max(share + layer, 2 * unit) for index in range(n_devices)}


def caps_for(module, devices: list[str]) -> dict[int, int]:
    """The caps :func:`load_full_finetune` applies on more than one card: the sizes read off ``module``,
    then :func:`balanced_caps`.

    ``module`` is the bare text model the loader builds (``model.lm``), so the direct reads in
    :func:`unit_bytes` and :func:`largest_layer_bytes` are the right ones. A *training* run places a
    ``PeftModel``, whose layer list sits one or two wrappers down, so ``sckevtype.train.shard`` reads
    the same two sizes through its own tree walk and calls :func:`balanced_caps` with them — one policy,
    two trees.
    """
    return balanced_caps(parameter_bytes(module), len(devices), unit_bytes(module), largest_layer_bytes(module))


def device_specs(gpus: int, device: str) -> list[str]:
    """The CUDA devices this run may use, in order — ``[]`` when the run is not on CUDA."""
    if not str(device).startswith("cuda"):
        return []
    return [f"cuda:{index}" for index in range(int(gpus))]


def free_bytes(spec: str) -> int:
    return int(torch.cuda.mem_get_info(int(str(spec).split(":")[1]))[0])


def plan_device_map(module, devices: list[str], reserve_bytes: int = RESERVE_BYTES,
                    max_bytes_per_device: int | Mapping[int, int] | None = None) -> dict[str, Any]:
    """An ``accelerate`` device_map that puts every parameter of ``module`` on ``devices`` — or a refusal.

    The refusal is the point. ``accelerate.infer_auto_device_map`` (1.15.0) contains no ``raise`` at all:
    a module that fits nowhere is walked past and the returned map simply does not mention it, so
    ``dispatch_model`` leaves it on CPU and the run "works" at CPU speed. That failure is invisible in
    every log — the job runs, the rows are correct, and the wall clock is a day. So the arithmetic is
    done here first, by name, with the numbers in the message.

    ``["cpu"]`` is a legal request (a checkpoint run without a GPU) and yields ``{"": "cpu"}`` — the same
    path ``accelerate`` takes for a single-device map, kept here so the dispatch/verify machinery has a
    locally testable case and is not a code path that only ever runs on the cluster.

    ``max_bytes_per_device`` caps each card's budget below its free memory, which is what a *training*
    run needs and a *load* does not. Free-memory budgeting is right when the question is "does it fit
    at all"; for training the question is "is each card's share of the weights small enough to leave
    room for activations", and the greedy fill below would otherwise put a model that fits on one card
    entirely on ``cuda:0`` — which is exactly the single-card run that OOMs.

    It takes either one cap for every card or a mapping of device index to cap, and the mapping is what
    more than two cards need. ``infer_auto_device_map`` keeps the largest remaining indivisible unit
    free on each card it is filling (``current_max_size = max_memory[device] - max_layer_size``), and
    that reservation *shrinks* as the fill proceeds: once the embedding is placed, a later card's
    effective budget is its cap minus one decoder layer instead of minus the embedding. With a single
    scalar, the later cards are therefore handed more room than the first one and the last card gets
    nothing — measured on this cluster, the 4B's four-card plan used two cards and the 9B's used three
    (``logs/multigpu-plan2-*.out``). A per-device mapping is how the caller gives the first card its
    reservation back and holds every later card at its share.

    The cap is applied to the budget and then the same refusal runs on the capped sum, so an over-capped
    ask is still refused by name here rather than left on CPU by ``accelerate``.
    """
    from accelerate import infer_auto_device_map

    if not devices:
        raise FullWeightError("no device was requested; there is nowhere to put the weights")
    if [str(device) for device in devices] == ["cpu"]:
        return {"": "cpu"}
    if any(str(device).split(":")[0] != "cuda" for device in devices):
        raise FullWeightError(f"devices must all be cuda:N (or exactly ['cpu']); got {devices}")
    budgets = {int(str(spec).split(":")[1]): max(free_bytes(spec) - reserve_bytes, 0) for spec in devices}
    if isinstance(max_bytes_per_device, Mapping):
        unknown = sorted(set(max_bytes_per_device) - set(budgets))
        if unknown:
            raise FullWeightError(f"max_bytes_per_device names device(s) {unknown}, which are not in {sorted(budgets)}")
        budgets = {index: min(budget, int(max_bytes_per_device.get(index, budget)))
                   for index, budget in budgets.items()}
    elif max_bytes_per_device is not None:
        if max_bytes_per_device < 1:
            raise FullWeightError(f"max_bytes_per_device must be positive when given; got {max_bytes_per_device}")
        budgets = {index: min(budget, max_bytes_per_device) for index, budget in budgets.items()}
    need = parameter_bytes(module)
    available = sum(budgets.values())
    if need > available:
        shown = ("one cap each" if isinstance(max_bytes_per_device, Mapping) else
                 f"a {max_bytes_per_device:,}-byte cap each" if max_bytes_per_device is not None else "no cap")
        raise FullWeightError(
            f"the checkpoint's {need:,} bytes of weights do not fit on {len(devices)} device(s) "
            f"({available:,} bytes free after a {reserve_bytes:,}-byte reserve each and {shown}: "
            f"{ {k: v for k, v in sorted(budgets.items())} }); this is the loader's own check, because "
            f"accelerate would silently leave the overflow on CPU rather than fail")
    return infer_auto_device_map(module, max_memory=budgets, no_split_module_classes=no_split_classes(module))


def main_device_of(device_map: dict[str, Any]) -> str:
    """The device the input tensors must be built on: the first mapped device, which is where the
    embedding lives (``infer_auto_device_map`` fills in module order, ``model.embed_tokens`` first).

    ``"cpu"`` is returned only when *every* entry is cpu, which is the single-device case ``accelerate``
    itself treats as ``model.to(device)``. An entry on ``"disk"`` is never acceptable here: a
    disk-offloaded model would read its weights off lustre on every forward pass.

    Normalised through :func:`_as_device` for the same reason the guard is: with ``max_memory={0: …, 1:
    …}`` the map's values are the bare ints ``0``/``1``, and ``torch`` rejects ``"0"`` as a device
    string (``torch.device("0")`` → ``RuntimeError: Invalid device string: '0'``). This value is used
    to build input tensors and to move the pointer head, so returning it raw is a crash one line after
    a guard that passed.
    """
    mapped = [str(device) for device in device_map.values()]
    if "disk" in mapped:
        raise FullWeightError(f"the device map offloads to disk: {device_map}")
    for name in mapped:
        if name.split(":")[0] != "cpu":
            return _as_device(name)
    return "cpu"


def _as_device(value: Any) -> str:
    """An ``hf_device_map`` value as a fully-qualified device string.

    ``accelerate`` fills the map with *hints*, not device strings: with ``max_memory={0: …, 1: …}`` every
    entry comes back as the bare int ``0``/``1``, and it emits ``"cpu"``/``"disk"`` as themselves. The
    placement guard compares against ``["cuda:0", …]``, so comparing the raw value rejects every module
    of a multi-GPU run — which is how the 27B failed to load at all, reporting ``('embed_tokens', '0')``
    as a module "mapped outside the requested devices". Normalise here; never loosen the guard.
    """
    text = str(value)
    if text.isdigit():
        return f"cuda:{text}"
    return text


def verify_placement(module, device_map: dict[str, Any], devices: list[str], where: str) -> dict[str, Any]:
    """Re-read the placement off the live parameters. Raises :class:`FullWeightError` if it is wrong.

    Two independent readings, because either alone can be fooled: the *map* is what accelerate said it
    would do, and the *parameters' own ``.device``* is what actually happened. A map that mentions only
    CUDA while a parameter sits on CPU is exactly the silent-offload failure this guard exists for, and
    it is the one a map-only check cannot see.

    The rule is *"nothing outside the devices this run asked for"* rather than *"no CPU"*: ``cpu`` is a
    legitimate request (a checkpoint run without a GPU) and an illegitimate surprise (a CUDA run that
    quietly landed on the host), and only the requested set can tell the two apart.
    """
    allowed = {str(device) for device in devices}
    offload = {name: _as_device(device)
               for name, device in device_map.items() if _as_device(device) not in allowed}
    if offload:
        raise FullWeightError(f"{where}: {len(offload)} module(s) were mapped outside the requested "
                              f"devices {sorted(allowed)} (e.g. {list(offload.items())[:2]}); a CUDA run "
                              f"that lands on cpu/disk is a CPU pass wearing a GPU's log")
    misplaced = sorted({str(p.device) for p in module.parameters() if str(p.device) not in allowed})
    if misplaced:
        raise FullWeightError(f"{where}: parameters live on {misplaced}, which is not one of the "
                              f"requested devices {sorted(allowed)}")
    per_device: dict[str, int] = {}
    for parameter in module.parameters():
        per_device[str(parameter.device)] = per_device.get(str(parameter.device), 0) + \
            parameter.numel() * parameter.element_size()
    return {"hf_device_map": {name: str(device) for name, device in device_map.items()},
            "devices": sorted(allowed), "weight_bytes_per_device": dict(sorted(per_device.items())),
            "weight_bytes": sum(per_device.values())}


# --------------------------------------------------------------------------- the loader

def load_full_finetune(checkpoint: Checkpoint, device: str, opts: LoadOptions = LoadOptions(),  # noqa: B008 - the vendor's own signature
                       gpus: int = 1, reserve_bytes: int = RESERVE_BYTES) -> tuple[Any, Any, dict[str, Any]]:
    """``(tokenizer, model, placement)`` for a full-weight checkpoint — the vendored ``Checkpoint.load``'s
    contract, minus the adapter.

    The model is built by ``DecisionModel.__init__`` on CPU (transformers 5 materialises a shard at a
    time, so peak host RAM is the model size and not twice it), then ``accelerate`` places its decoder
    layers on ``gpus`` cards. On more than one card those cards are budgeted with the balanced caps
    (:func:`caps_for`), which is what makes the fill spread the weights instead of packing the first
    card solid; a single card — and the ``cpu`` path — is planned from its free memory exactly as it
    was before the caps existed. ``attn`` is forced to ``"sdpa"`` on CUDA because ``DecisionModel``
    derives it from the *device string it was constructed with* — and that string is ``"cpu"`` here,
    which would otherwise silently select eager attention, a different kernel from the one the LoRA
    rungs ran.
    """
    meta = checkpoint.meta
    if meta.option_isolation:
        raise FullWeightError("option_isolation needs the packed mask, which a hybrid backbone cannot "
                              "honour (kev.model.DecisionModel refuses it too)")
    dtype = load_dtype(checkpoint, opts)
    tok = tokenizer_for(checkpoint)
    attn = opts.attn or ("sdpa" if str(device).startswith("cuda") else "eager")
    model = DecisionModel(str(checkpoint.path), tok, "cpu", lora=None, revision=None, attn=attn,
                          head_dim=meta.head_dim, option_isolation=meta.option_isolation, dtype=dtype)
    placement: dict[str, Any] = {"kind": FULL, "dtype": DTYPE_NAMES.get(dtype, str(dtype)),
                                 "attn": model.lm.config._attn_implementation,
                                 # Which forward path this checkpoint runs is decided by the config's
                                 # own `layer_types` (`kev.model.is_hybrid`), not by an argument here --
                                 # a hybrid backbone runs one causal row per question and the packed
                                 # block-causal mask is not built at all. Recorded because a reader
                                 # comparing two rungs' timings needs to know which form each one ran.
                                 "hybrid": bool(model.hybrid),
                                 "weights_dtype_declared": meta.weights_dtype,
                                 "parameters_bytes": parameter_bytes(model.lm)}
    # `Checkpoint.load`'s last two lines, which this loader must reproduce or the readout is a random
    # head: the pointer head's weights come from head.pt, and the temperature is the checkpoint's own
    # calibration (kev/kev/checkpoint.py:171-172). `.eval()` is load-bearing rather than cosmetic --
    # `PointerHead.forward` skips the temperature division entirely while `self.training` is true.
    model.head.load_state_dict(meta.head)
    model.eval()
    model.head.temperature = meta.temperature if opts.temperature is None else opts.temperature
    placement["temperature"] = float(model.head.temperature)
    placement["head_tensors"] = len(meta.head or {})
    # One placement path for every device, including a CPU run: `accelerate.dispatch_model` with a
    # single-entry map is `model.to(device)` plus `hf_device_map`, so routing the CPU case through it too
    # means the guard below is exercised by a locally runnable test rather than only on the cluster.
    from accelerate import dispatch_model

    devices = device_specs(gpus, device) or [str(device)]
    # Free-memory budgeting answers "does it fit at all", and on one card that is the whole question. On
    # more than one it is the wrong question: the fill budgets every card from its own free memory and
    # fills them in order, so the first card is packed solid — measured (job 4207520): the 27B's 51.2 GB
    # came out as {'cuda:0': 38.3 GB, 'cuda:1': 12.9 GB} with cuda:2/cuda:3 empty — and the forward then
    # dies on it. The balanced caps (:func:`caps_for`, the training path's own policy) give every card
    # its share of the weights and leave the reserve for activations. The single-device path is
    # byte-for-byte what it was: there is nothing to balance across one card.
    caps = caps_for(model.lm, devices) if len(devices) > 1 else None
    if caps is None:
        device_map = plan_device_map(model.lm, devices, reserve_bytes=reserve_bytes)
    else:
        device_map = plan_device_map(model.lm, devices, reserve_bytes=reserve_bytes,
                                     max_bytes_per_device=caps)
    dispatch_model(model.lm, device_map)
    placement.update(verify_placement(model.lm, device_map, devices, "load_full_finetune"))
    if caps is not None:
        # The caps are part of the record: a reader comparing each card's weight bytes against what that
        # card was allowed is how the placement is audited (the training path's record carries the same
        # key). Keyed by the device it caps, in the same shape `shard.verify_shard` writes it.
        placement["max_bytes_per_device"] = {f"cuda:{index}": int(cap) for index, cap in sorted(caps.items())}
    main = main_device_of(device_map)
    model.head.to(main)
    model.device = main
    head_device = str(next(model.head.parameters()).device)
    if head_device != main:
        raise FullWeightError(f"the pointer head is on {head_device} while the backbone's inputs are "
                              f"built on {main}; the readout would move every hidden state across devices")
    placement["main_device"] = main
    return tok, model, placement


class FullFinetunePredictor(LocalPredictor):
    """``LocalPredictor`` for a full-weight checkpoint: **the same ``__call__``**, a loader that needs no adapter.

    Everything after the load is inherited. That is the whole argument for this class existing instead of
    a reimplementation: ``encode``, ``forward``, the softmax, the key ordering and the token accounting
    are the vendor's functions, unmodified, so a row from the 27B differs from a row from the 9B in its
    weights and in nothing else.
    """

    def __init__(self, run: str, device: str = "cuda", opts: LoadOptions = LoadOptions(),  # noqa: B008 - the vendor's own signature
                 context: dict = CONTEXT, gpus: int = 1, reserve_bytes: int = RESERVE_BYTES) -> None:
        # The three lines below are LocalPredictor.__init__'s own guard and CUDA switches, reproduced
        # because this __init__ replaces that one and the switches have to be set before any forward
        # pass. They are asserted straight after, so a drift in the vendored default is a load failure
        # here rather than a quiet change of precision in a reported number.
        if opts.temperature is not None and not (math.isfinite(opts.temperature) and opts.temperature > 0):
            raise ValueError("temperature must be finite and positive")
        if device == "cuda":
            # evaluation is fp32-exact: TF32 (10-bit mantissa) moves probabilities by ~1e-3, the
            # isolation gate's tolerance (kev/kev/predictors.py:35-37)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
        self.run = str(run)
        checkpoint = Checkpoint(self.run)
        self.checkpoint_path = checkpoint.path
        self.tok, self.model, self.placement = load_full_finetune(
            checkpoint, device, opts, gpus=gpus, reserve_bytes=reserve_bytes)
        self.temperature = self.model.head.temperature
        self.device = self.placement.get("main_device", device)
        self.context = context
        assert_no_exact_path_drift(device, self.placement)


def assert_no_exact_path_drift(device: str, placement: dict[str, Any]) -> None:
    """The exact-path switches and the attention backend, read back off the live process and the model.

    A check that cannot fail is worse than no check: each of these is false for a real CUDA run if the
    switches were not set or if the model was built with the wrong ``attn`` string, and the failure is a
    silently different number in a reported metric.
    """
    if not str(device).startswith("cuda"):
        return
    off = {"matmul.allow_tf32": torch.backends.cuda.matmul.allow_tf32,
           "cudnn.allow_tf32": torch.backends.cudnn.allow_tf32,
           "flash_sdp": torch.backends.cuda.flash_sdp_enabled(),
           "mem_efficient_sdp": torch.backends.cuda.mem_efficient_sdp_enabled()}
    wrong = {name: value for name, value in off.items() if value}
    if wrong:
        raise FullWeightError(f"the fp32-exact switches are not in the state LocalPredictor sets: "
                              f"{wrong}; every reported number would be measured on a different path")
    if placement.get("attn") != "sdpa":
        raise FullWeightError(f"the backbone was built with attn={placement.get('attn')!r}; the CUDA "
                              f"path every reported number uses is 'sdpa'")


def build(run: str, device: str, opts: LoadOptions, context: dict, gpus: int = 1,
          reserve_bytes: int = RESERVE_BYTES) -> FullFinetunePredictor:
    """Convenience: the predictor for a full-weight checkpoint."""
    return FullFinetunePredictor(run, device, opts, context, gpus=gpus, reserve_bytes=reserve_bytes)
