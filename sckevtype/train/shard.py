"""Multi-GPU delta training: the frozen backbone is sharded, the readout is not.

Why this module exists
----------------------
A delta fine-tune updates a LoRA adapter and the pointer head over a *frozen* backbone, so the only
thing a second card can usefully hold is part of the frozen weights — the optimizer has no state for
the backbone at all. That makes sharding a *placement* problem rather than a distributed-optimizer
one, and it is the only way the 4B and 9B deltas fit here: the single-card runs die with
``torch.OutOfMemoryError`` at 36.4 GB allocated (4B, job 4207304: ``Tried to allocate 2.05 GiB`` on a
40 GB card with 291 MiB reserved-but-unallocated), the peak scales with the window, the suite runs to
8,192-token windows, and this cluster has no 80 GB card (180× A100-40G over two partitions).

What is reused, verbatim
------------------------
The placement rules are ``scripts/kev/kev_fullft.py``'s, imported rather than copied: that module
was written for the 27B's *load* and already does the hard half — budget each visible card from
``torch.cuda.mem_get_info`` minus a reserve, call ``accelerate.infer_auto_device_map`` with the
backbone's own no-split class, then re-read the placement off the live parameters and refuse any map
that offloads to cpu or disk. One implementation, so the load path and the training path cannot
drift; :func:`plan` only adds the one thing a *load* does not need (see :func:`plan`).

Where the trainable parameters end up, stated plainly
-----------------------------------------------------
The pointer head is a sibling of the backbone (``model.head``), so it stays on ``cuda:0`` and the
optimizer's AdamW state for it is created there. The LoRA adapter is *not* a sibling: peft attaches
``lora_A``/``lora_B`` inside each decoder layer's linear modules, so an adapter tensor rides with the
layer it modifies — and it must, because ``LoraLayer.forward`` adds the adapter's output to the base
layer's on the same device. Pinning the adapter to ``cuda:0`` while its layer sits on ``cuda:1``
would need a device transfer inside every LoRA module (two copies of a full activation per layer per
pass, and every layer's saved dropout output retained on one card) and would defeat the point of the
shard. What this module guarantees instead, and checks: every decoder layer is on exactly one card
(:func:`split_layers`), so no adapter is ever separated from its base weight; nothing is on cpu or
disk (:func:`kev_fullft.verify_placement`); and the placement record says where every trainable byte
went, so a row can be traced to how it was run.

Refuse, never silently degrade
------------------------------
Every failure here is a named refusal before any training step: weights that do not fit the requested
cards, a map that offloads to cpu/disk, a layer split across cards, a head that is not on the card the
inputs are built on, or a run that asked for N cards and got fewer. An offloaded run is a CPU pass
wearing a GPU's log, and a one-card run wearing an N-card ask has the memory story of the cards it did
not get.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

#: Bytes left free on every card for the forward/backward activations, the attention masks and the
#: allocator's fragmentation. This is not a weight budget — :func:`plan`'s per-card share is — it is
#: the floor under it, and it is deliberately larger than ``kev_fullft``'s 2 GB *loading* reserve:
#: the 0.8B delta peaked at 21.4 GiB on a 40 GB card against 1.5 GB of weights, i.e. a training step
#: spends more on non-weight memory than a load ever does.
RESERVE_BYTES = 8 * 1024**3

#: The module that owns the placement rules (see the docstring). It sits beside this package rather
#: than inside it because it is also the entry point the 27B *load* uses, and it is loaded from its
#: path so neither the package nor that script has to put ``scripts/`` on ``sys.path``.
PLACEMENT_SOURCE = Path(__file__).resolve().parents[2] / "scripts" / "kev" / "kev_fullft.py"

_placement: Any = None


class ShardError(RuntimeError):
    """A run that asked for N cards cannot have them. Never a silent downgrade."""


def placement() -> Any:
    """``scripts/kev/kev_fullft.py`` as a module object, loaded once per process.

    Lazy, and by path: importing this module must not import torch (the training layer's import guard
    is tested), and the script is not on any package path. A tree that lacks the file is a tree that
    cannot place a sharded run, which is said by name rather than by an ImportError traceback.
    """
    global _placement
    if _placement is None:
        if not PLACEMENT_SOURCE.is_file():
            raise ShardError(
                f"no placement module at {PLACEMENT_SOURCE}: a sharded run reuses "
                "scripts/kev/kev_fullft.py's plan_device_map/verify_placement, so stage that file "
                "before submitting (it is ~25 KB).")
        spec = importlib.util.spec_from_file_location("sckevtype_kev_fullft", PLACEMENT_SOURCE)
        if spec is None or spec.loader is None:  # pragma: no cover - a path that is a directory, etc.
            raise ShardError(f"{PLACEMENT_SOURCE} is not loadable as a python module")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _placement = module
    return _placement


# --------------------------------------------------------------------------- which cards

def devices_for(gpus: int, device: str) -> list[str]:
    """The cards this run may use, in order: ``["cuda:0", ..., "cuda:N-1"]``, or ``[]`` off CUDA."""
    return list(placement().device_specs(gpus, device))


# --------------------------------------------------------------------------- what may not be split

def decoder_layers(module: Any) -> Any:
    """The backbone's repeated layer list, found through whatever wrappers are in the way.

    ``kev_fullft.no_split_classes`` reads ``module.layers``, which is the bare text model the 27B
    loader builds. A *training* model is a ``PeftModel`` around that text model, so the layer list is
    one or two wrappers down; peft forwards unknown attributes to its base model, so the direct read
    usually still works, and the walk below is what makes it work when it does not. Getting this
    wrong is not a slow run: accelerate would then be free to split a decoder layer, and a LoRA module
    on a different card from the base weight it is added to is a device mismatch inside
    ``LoraLayer.forward``.
    """
    import torch

    def usable(value: Any) -> bool:
        return isinstance(value, torch.nn.ModuleList) and len(value) > 0

    direct = getattr(module, "layers", None)
    if usable(direct):
        return direct
    for _, child in module.named_modules():
        found = getattr(child, "layers", None)
        if usable(found):
            return found
    return None


def no_split_classes(module: Any) -> list[str]:
    """The class name accelerate must treat as atomic — ``kev_fullft.no_split_classes``'s rule,
    extended to the wrapped tree by :func:`decoder_layers`."""
    F = placement()
    direct = F.no_split_classes(module)
    if direct:
        return direct
    layers = decoder_layers(module)
    return [type(layers[0]).__name__] if layers is not None else []


def layer_names(module: Any) -> list[str]:
    """The qualified names of the decoder layers, as accelerate's device map would name them."""
    layers = decoder_layers(module)
    if layers is None:
        return []
    wanted = {id(layer) for layer in layers}
    return [name for name, child in module.named_modules() if id(child) in wanted]


def split_layers(module: Any) -> dict[str, list[str]]:
    """Decoder layers whose parameters are not all on one card: ``{layer name: [devices]}``.

    The invariant that makes the shard legal. ``no_split_module_classes`` asks accelerate for it; this
    reads it back off the live parameters, because the ask and the result are two different things and
    only the second one runs.
    """
    offenders: dict[str, list[str]] = {}
    layers = decoder_layers(module)
    if layers is None:
        return offenders
    wanted = {id(layer) for layer in layers}
    for name, child in module.named_modules():
        if id(child) not in wanted:
            continue
        devices = sorted({str(parameter.device) for parameter in child.parameters()})
        if len(devices) > 1:
            offenders[name] = devices
    return offenders


def unit_bytes(module: Any) -> int:
    """The largest indivisible chunk accelerate will reserve room for on every card.

    ``infer_auto_device_map`` keeps one such chunk free on each main device (``current_max_size =
    max_memory[device] - max_layer_size``) so a later module can still be split there, so a per-card
    budget has to carry it twice: once for the room that is reserved, and once for the room the
    reservation is meant to leave usable. Being generous here costs only balance; being short is how
    the overflow ends up on the cpu, which is the failure this module exists to refuse.
    """
    F = placement()
    classes = set(no_split_classes(module))
    largest = 0
    for _, child in module.named_modules():
        leaf = len(list(child.children())) == 0
        if leaf or type(child).__name__ in classes:
            largest = max(largest, F.parameter_bytes(child))
    return largest


# --------------------------------------------------------------------------- where it goes

def largest_layer_bytes(module: Any) -> int:
    """The biggest decoder layer: the unit the greedy fill moves to the next card when it does not fit."""
    layers = decoder_layers(module)
    if layers is None:
        return 0
    F = placement()
    return max(F.parameter_bytes(layer) for layer in layers)


def plan(module: Any, devices: list[str], reserve_bytes: int = RESERVE_BYTES) -> tuple[dict[str, Any], dict[int, int]]:
    """``(device_map, the per-card budget caps)`` for a *training* run over ``devices``.

    ``kev_fullft.plan_device_map`` budgets every card from its free memory, which answers "does this fit
    at all". For training the question is different: a 4B at bf16 is 8 GB and fits a 40 GB card whole,
    so a free-memory plan puts all of it on ``cuda:0`` and the run is the single-card run that dies.
    Each card's budget is therefore capped at its balanced share — the arithmetic is
    ``kev_fullft.balanced_caps``'s, the same function the 27B's *load* path calls, so the policy has one
    implementation and this module adds only the two sizes a wrapped training tree needs — and the caps
    are handed to kev_fullft's own ``plan_device_map`` so the budget arithmetic and the refusal stay in
    one place too.

    The caps are *per card*, not one number for all of them, and that is a measured requirement rather
    than a refinement. ``infer_auto_device_map`` keeps the largest remaining indivisible unit free on
    the card it fills **first** (``main_devices = [gpus[0], "cpu"]`` in accelerate 1.15 — the card that
    has to be able to take back a layer offloaded to the cpu), and that reservation shrinks as the fill
    proceeds, from the embedding to a decoder layer; the later cards carry no reservation at all. One
    scalar cap therefore gives the later cards a bigger effective budget than the first one got, and
    when it is generous enough the last card gets nothing — measured on the cluster, the 4B's four-card
    plan used **two** cards (3.94 + 4.02 GiB, ``cuda:2``/``cuda:3`` empty) and the 9B's used **three**
    (``logs/multigpu-plan2-*.out``), both of which the run's own guard refuses.

    The two facts from the same source set the caps, and both were checked against the fill's own numbers
    (``tests/train/test_shard.py``, ``tests/train/test_fullft_load.py``):

    * the fill moves in whole layers, so a card's fill ends within one *layer* of its cap (the first card
      one layer lower still, since it carries the reservation) — ``ceil(need / n) + layer`` is a card
      that holds its share;
    * a card must be able to *place* the largest indivisible unit, not merely hold it: the fill requires
      ``used + unit <= cap - max_layer_size``, and while the embedding is unplaced the reservation is
      that same unit — so every card gets a floor of ``2 * unit``. Without it the 4B's four-card plan
      cannot put the embedding anywhere and the first card is skipped.
    """
    F = placement()
    if not devices:
        raise ShardError("no device was requested; there is nowhere to put the weights")
    # One arithmetic, in kev_fullft (`balanced_caps`): this module cannot be imported *by* kev_fullft —
    # shard imports it — so the policy lives there and this module supplies only the two sizes, read
    # through the wrapper walk a training tree needs (`decoder_layers`).
    caps = F.balanced_caps(F.parameter_bytes(module), len(devices), unit_bytes(module),
                           largest_layer_bytes(module))
    return F.plan_device_map(module, devices, reserve_bytes=reserve_bytes,
                             max_bytes_per_device=caps), caps


def trainable_bytes_per_device(model: Any) -> dict[str, int]:
    """Bytes of *trainable* parameters (the adapter and the head) per device they live on."""
    per: dict[str, int] = {}
    for parameter in model.parameters():
        if parameter.requires_grad:
            key = str(parameter.device)
            per[key] = per.get(key, 0) + parameter.numel() * parameter.element_size()
    return dict(sorted(per.items()))


def _layers_per_device(module: Any) -> dict[str, int]:
    """How many decoder layers each card got — the balance a reader can check at a glance.

    Read off the live parameters rather than off the map's names: a map entry may be a whole subtree
    (``{"": "cpu"}`` names no layer at all), and what a reader wants to know is where the layers
    actually are. A layer split across two cards is counted under both — :func:`verify_shard` refuses
    that case, and counting it under one would hide it from this record.
    """
    counts: dict[str, int] = {}
    layers = decoder_layers(module)
    if layers is None:
        return counts
    for layer in layers:
        for device in sorted({str(parameter.device) for parameter in layer.parameters()}):
            counts[device] = counts.get(device, 0) + 1
    return dict(sorted(counts.items()))


def verify_shard(model: Any, device_map: dict[str, Any], devices: list[str], where: str,
                 *, reserve_bytes: int = RESERVE_BYTES,
                 max_bytes_per_device: dict[int, int] | int | None = None) -> dict[str, Any]:
    """Re-read the placement off the live model and return the record — or refuse, by name.

    The first half is ``kev_fullft.verify_placement``'s: nothing outside the requested cards, read off
    the parameters rather than off the map. The second half is what a *training* run has to add, and
    each of the three is a failure that would otherwise be silent:

    * a decoder layer split across cards (an adapter separated from its base weight);
    * the pointer head not on the card the inputs are built on (every hidden state would move);
    * fewer cards used than requested (the run's memory story is then not the one it was submitted
      with — the operator asked for four cards and got the headroom of one).
    """
    F = placement()
    record = dict(F.verify_placement(model.lm, device_map, devices, where))
    # The map is recorded the way a reader needs it — as devices, not as accelerate's raw values
    # (`max_memory={0: …, 1: …}` comes back as the bare ints 0/1, and `str(0)` is not a device string).
    # `_as_device` is the same normalisation the guard above uses, so the two readings cannot disagree.
    record["hf_device_map"] = {name: F._as_device(device) for name, device in device_map.items()}
    used = sorted(set(record["hf_device_map"].values()))
    record["requested_gpus"] = len(devices)
    record["devices"] = list(devices)
    record["devices_used"] = used
    record["main_device"] = F.main_device_of(device_map)
    record["head_device"] = str(next(model.head.parameters()).device)
    record["reserve_bytes"] = int(reserve_bytes)
    if max_bytes_per_device is None:
        record["max_bytes_per_device"] = None
    elif isinstance(max_bytes_per_device, dict):
        # Keyed by the device it caps, because that is what a reader compares against the bytes each
        # card actually got (`weight_bytes_per_device`).
        record["max_bytes_per_device"] = {f"cuda:{index}": int(cap) for index, cap in sorted(max_bytes_per_device.items())}
    else:
        record["max_bytes_per_device"] = {device: int(max_bytes_per_device) for device in devices}
    record["layers_per_device"] = _layers_per_device(model.lm)
    record["trainable_bytes_per_device"] = trainable_bytes_per_device(model)
    missing = [device for device in devices if device not in used]
    if missing:
        raise ShardError(
            f"{where}: the map uses {used} of the {len(devices)} requested card(s) — {missing} got "
            f"nothing ({record['weight_bytes_per_device']}). A run that asked for {len(devices)} cards "
            f"and got fewer has the memory story of the cards it did not get; submit fewer, or free "
            f"the cards, or raise the reserve.")
    offenders = split_layers(model.lm)
    if offenders:
        shown = list(offenders.items())[:2]
        raise ShardError(
            f"{where}: {len(offenders)} decoder layer(s) are split across cards (e.g. {shown}); the "
            f"no-split class {no_split_classes(model.lm)} was not honoured, and an adapter on a "
            f"different card from its base weight is a device mismatch, not a slow run")
    if record["head_device"] != record["main_device"]:
        raise ShardError(
            f"{where}: the pointer head is on {record['head_device']} while the backbone's inputs are "
            f"built on {record['main_device']}; every hidden state would move across cards, and the "
            f"head is the readout of every row this run reports")
    return record


def shard_model(model: Any, gpus: int, device: str, reserve_bytes: int = RESERVE_BYTES) -> dict[str, Any]:
    """Place ``model.lm`` over the run's cards with the readout on the first one. Returns the record.

    The adapter and the head are *not* moved: they are where ``build_model`` put them (``cuda:0``, the
    main device the plan starts from), which is what keeps the optimizer's state on one card and the
    head next to the inputs. ``model.device`` is then pinned to that card so the input tensors,
    the attention masks and the position ids are built where the first layer's hook expects them.
    """
    from accelerate import dispatch_model

    devices = devices_for(gpus, device)
    if len(devices) < 2:
        raise ShardError(
            f"--gpus {gpus} on --device {device!r} names {devices or 'no cuda device'}; a sharded run "
            f"needs at least two cuda:N devices")
    device_map, cap = plan(model.lm, devices, reserve_bytes=reserve_bytes)
    dispatch_model(model.lm, device_map)
    record = verify_shard(model, device_map, devices, "shard_model", reserve_bytes=reserve_bytes,
                          max_bytes_per_device=cap)
    model.device = record["main_device"]
    return record


def describe(record: dict[str, Any]) -> str:
    """One line for the run log: where the weights and the trainable bytes went."""
    weights = ", ".join(f"{device} {size / 2**30:.2f} GiB" for device, size in record["weight_bytes_per_device"].items())
    trainable = ", ".join(f"{device} {size / 2**20:.1f} MiB" for device, size in record["trainable_bytes_per_device"].items())
    layers = ", ".join(f"{device} {count}" for device, count in record["layers_per_device"].items())
    cap = record.get("max_bytes_per_device")
    if isinstance(cap, dict):
        caps = ", ".join(f"{device} {size / 2**30:.2f} GiB" for device, size in cap.items())
    elif cap is None:
        caps = "unset"
    else:
        caps = f"{cap:,} B"
    return (f"sharded over {record['devices_used']}: weights {weights} | layers {layers} | "
            f"trainable {trainable} | head {record['head_device']} | caps {caps}")


# --------------------------------------------------------------------------- what it costs

class DevicePeaks:
    """Peak allocated bytes per card: a sharded run has one number per card and the log has one.

    ``kev.device.allocated_bytes`` and ``torch.cuda.reset_peak_memory_stats()`` both read and clear
    device 0 alone, so a sharded run's recorded peak would be ``cuda:0``'s — the one card the run
    exists to relieve. This reads every card the placement used, and is CUDA-only on purpose (a
    non-CUDA run has one device and kev's own helper is right for it).
    """

    def __init__(self, devices: list[str]) -> None:
        if not devices or any(not str(spec).startswith("cuda") for spec in devices):
            raise ShardError(f"DevicePeaks is for the cards a sharded run used; got {devices}")
        self.devices = [str(spec) for spec in devices]

    @staticmethod
    def _index(spec: str) -> int:
        return int(str(spec).split(":")[1])

    def reset(self) -> None:
        import torch

        for spec in self.devices:
            torch.cuda.reset_peak_memory_stats(self._index(spec))

    def read(self) -> dict[str, int]:
        import torch

        return {spec: int(torch.cuda.max_memory_allocated(self._index(spec))) for spec in self.devices}

    def max(self) -> int:
        """The worst card's peak — the number that decides whether a run fits."""
        return max(self.read().values(), default=0)
