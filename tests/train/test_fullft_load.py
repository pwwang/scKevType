"""The load path's half of the balance rule: the caps, and the single-device path that must not change.

``scripts/kev/kev_fullft.py::load_full_finetune`` planned every card from its free memory, which answers
"does it fit at all" and packs the first card solid: on this cluster (job 4207520) the 27B's 51.2 GB of
bf16 weights came out as ``{'cuda:0': '38.3 GB', 'cuda:1': '12.9 GB'}`` with ``cuda:2``/``cuda:3`` empty,
and the forward then died with ``torch.OutOfMemoryError`` on ``cuda:0``'s last gigabyte. The fix is the
training path's per-card caps — ``kev_fullft.balanced_caps``, the one arithmetic both paths call — applied
by the load path whenever more than one card is requested.

Everything here runs without a card: the caps are arithmetic over parameter bytes, and
``accelerate.infer_auto_device_map`` is pure python over sizes, class names and the tree's shape. The
27B's own numbers (51,249,200,128 bytes of weights, 64 layers of 744,510,464/766,546,368 bytes, a
2,542,796,800-byte embedding, read from the checkpoint's ``model.safetensors.index.json``) are pinned in
:func:`test_the_27b_fill_spreads_over_four_cards`, so a change to the caps that would strand part of the
checkpoint shows up here instead of in a 24-hour queue slot.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from sckevtype.train import shard

torch = pytest.importorskip("torch", reason="the placement layer is torch's; requires the train extra")
pytest.importorskip("kev.checkpoint", reason="the loader is kev's; requires the train extra")
pytest.importorskip("accelerate", reason="the fill is accelerate's; requires the train extra")

#: What an idle A100-40G on this cluster reports free — the same number ``multigpu-plan-probe.py`` uses,
#: read off the failed single-card run's own OOM text ("a total capacity of 39.50 GiB").
FREE_40G = int(39.5 * 1024**3)

#: ``jaredpalmer/kev-27b``'s own sizes, from its safetensors index (850 tensors, 51,249,200,128 bytes).
KEV_27B = {"weights": 51_249_200_128, "embed": 2_542_796_800, "norm": 10_240,
           "layer_big": 766_546_368, "layer_small": 744_510_464, "layers": 64}
#: What ``balanced_caps`` must come out at for those sizes over four cards: the share plus one large
#: layer, which is above the ``2 * unit`` floor (that floor is 5,085,593,600).
KEV_27B_CAP = 13_578_846_400


# --------------------------------------------------------------------------- stand-ins

class Layer(torch.nn.Module):
    """A decoder-layer-shaped unit: the class name accelerate must not split."""

    def __init__(self, width: int = 64) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(width, width, bias=False)


class Backbone(torch.nn.Module):
    """The shape ``DecisionModel.lm`` has: ``embed_tokens``, a ``layers`` ModuleList, a ``norm``."""

    def __init__(self, layers: int = 4, width: int = 64, embed: int = 128) -> None:
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(embed, width)
        self.layers = torch.nn.ModuleList(Layer(width) for _ in range(layers))
        self.norm = torch.nn.LayerNorm(width)


class Leaf(torch.nn.Module):
    """A leaf whose single bf16 parameter has exactly the asked-for bytes: an embedding or a norm."""

    def __init__(self, total_bytes: int) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(total_bytes // 2, dtype=torch.bfloat16, device="meta"))


class DecoderLayer(torch.nn.Module):
    """A decoder layer as the fill sees it: one indivisible module of exactly the asked-for bytes."""

    def __init__(self, total_bytes: int) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(total_bytes // 2, dtype=torch.bfloat16, device="meta"))


def kev_27b_tree() -> torch.nn.Module:
    """The 27B's shape and sizes, nothing else: an embedding leaf, 64 layers, a norm.
    The layer sizes are the checkpoint's own (48 are ``layer_big``, every fourth is ``layer_small``),
    so ``parameter_bytes`` of this tree is exactly the checkpoint's 51,249,200,128 bytes.
    """
    tree = torch.nn.Module()
    tree.embed_tokens = Leaf(KEV_27B["embed"])
    tree.layers = torch.nn.ModuleList(
        DecoderLayer(KEV_27B["layer_big"] if index % 4 != 3 else KEV_27B["layer_small"])
        for index in range(KEV_27B["layers"]))
    tree.norm = Leaf(KEV_27B["norm"])
    return tree


class FakeLM(torch.nn.Module):
    """Stands in for the built backbone: a tree the caps can be read off, plus the config record."""

    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(16, 8)
        self.layers = torch.nn.ModuleList(Layer(8) for _ in range(4))
        self.norm = torch.nn.LayerNorm(8)
        self.config = SimpleNamespace(_attn_implementation="sdpa")


class FakeDecisionModel(torch.nn.Module):
    """Stands in for ``kev.model.DecisionModel``: ``.lm``, ``.head``, ``.hybrid``."""

    def __init__(self) -> None:
        super().__init__()
        self.lm = FakeLM()
        self.head = torch.nn.Linear(8, 4)
        self.hybrid = False


def a_fake_checkpoint() -> SimpleNamespace:
    """The parts of ``kev.checkpoint.Checkpoint`` ``load_full_finetune`` reads, with no file behind it."""
    return SimpleNamespace(
        path="no-such-checkpoint",                                   # never opened: tokenizer_for is patched
        meta=SimpleNamespace(option_isolation=False, head_dim=8, weights_dtype="bf16", temperature=1.0,
                             head={"weight": torch.zeros(4, 8), "bias": torch.zeros(4)}))


# --------------------------------------------------------------------------- the arithmetic

def test_the_arithmetic_rounds_the_share_up_and_floors_at_two_units() -> None:
    """``max(ceil(need / n) + layer, 2 * unit)`` per card: both terms, on numbers a reader can check."""
    F = shard.placement()
    assert F.balanced_caps(10, 4, unit=1, layer=1) == {0: 4, 1: 4, 2: 4, 3: 4}       # ceil(10/4)=3, +1
    assert F.balanced_caps(10, 4, unit=100, layer=1) == {0: 200, 1: 200, 2: 200, 3: 200}  # the floor
    assert F.balanced_caps(9, 3, unit=2, layer=1) == {0: 4, 1: 4, 2: 4}               # 3+1 == 2*unit
    assert F.balanced_caps(1, 1, unit=1, layer=1) == {0: 2}                          # one card, still capped
    with pytest.raises(F.FullWeightError, match="at least one card"):
        F.balanced_caps(10, 0, unit=1, layer=1)


def test_the_training_path_calls_the_same_arithmetic(monkeypatch: pytest.MonkeyPatch) -> None:
    """The anti-drift property: ``shard.plan`` must *call* kev_fullft's function, not re-implement it.

    This module cannot be imported by ``kev_fullft`` (the import would be circular), so the one policy
    lives there; if a second copy ever appears in ``shard.py``, this fails.
    """
    F = shard.placement()
    monkeypatch.setattr(F, "free_bytes", lambda spec: FREE_40G)
    backbone = Backbone(layers=8, width=64, embed=64)
    seen: dict[str, object] = {}

    def spy(need: int, n_devices: int, unit: int, layer: int) -> dict[int, int]:
        seen.update(need=need, n_devices=n_devices, unit=unit, layer=layer)
        return {index: 2**40 for index in range(n_devices)}

    monkeypatch.setattr(F, "balanced_caps", spy)
    _, caps = shard.plan(backbone, ["cuda:0", "cuda:1"])
    assert seen == {"need": F.parameter_bytes(backbone), "n_devices": 2,
                    "unit": shard.unit_bytes(backbone), "layer": shard.largest_layer_bytes(backbone)}
    assert caps == {0: 2**40, 1: 2**40}


def test_the_caps_are_read_off_the_module_in_the_fills_own_terms() -> None:
    """``caps_for`` reads the two sizes the fill reserves by: the largest indivisible unit and layer."""
    F = shard.placement()
    tree = kev_27b_tree()
    assert F.parameter_bytes(tree) == KEV_27B["weights"]
    assert F.unit_bytes(tree) == KEV_27B["embed"]                 # the embedding is bigger than a layer
    assert F.largest_layer_bytes(tree) == KEV_27B["layer_big"]
    assert F.caps_for(tree, ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]) == {index: KEV_27B_CAP
                                                                         for index in range(4)}
    # ... and with a small embedding the floor takes over, exactly as shard.unit_bytes' stand-in shows.
    assert F.unit_bytes(Backbone(layers=4, width=64, embed=16)) == 64 * 64 * 4


# --------------------------------------------------------------------------- the fill

def test_a_capped_plan_spreads_what_a_free_memory_plan_packs(monkeypatch: pytest.MonkeyPatch) -> None:
    """The failure in miniature: 40 GiB free puts the whole backbone on ``cuda:0`` unless it is capped."""
    F = shard.placement()
    monkeypatch.setattr(F, "free_bytes", lambda spec: FREE_40G)
    backbone = Backbone(layers=64, width=256, embed=1024)
    devices = ["cuda:0", "cuda:1"]
    packed = F.plan_device_map(backbone, devices)
    assert sorted({F._as_device(device) for device in packed.values()}) == ["cuda:0"], packed
    spread = F.plan_device_map(backbone, devices, max_bytes_per_device=F.caps_for(backbone, devices))
    assert sorted({F._as_device(device) for device in spread.values()}) == devices, spread


def test_the_27b_fill_spreads_over_four_cards(monkeypatch: pytest.MonkeyPatch) -> None:
    """The property the 27B's placement line is checked against: four cards, nothing on cpu/disk.

    The caps are the training path's, the fill is ``accelerate`` 1.15's own, and the sizes are the
    checkpoint's — so this is the same arithmetic the job will run, minus the cards. Measured here
    (before the caps existed): 38.3 GB of weights on ``cuda:0`` and two cards empty.
    """
    F = shard.placement()
    monkeypatch.setattr(F, "free_bytes", lambda spec: FREE_40G)
    tree = kev_27b_tree()
    need = F.parameter_bytes(tree)
    devices = [f"cuda:{index}" for index in range(4)]
    caps = F.caps_for(tree, devices)
    assert set(caps.values()) == {KEV_27B_CAP}
    assert max(caps.values()) < need, "a cap that can hold the whole checkpoint spreads nothing"
    device_map = F.plan_device_map(tree, devices, reserve_bytes=F.RESERVE_BYTES, max_bytes_per_device=caps)
    per: dict[str, int] = {}
    for name, device in device_map.items():
        key = F._as_device(device)
        per[key] = per.get(key, 0) + F.parameter_bytes(tree.get_submodule(name))
    assert sorted(per) == devices, f"the fill put weights on {sorted(per)}; the loader refuses cpu/disk"
    assert sum(per.values()) == need, "the fill must account for every weight byte"
    assert max(per.values()) <= max(caps.values()), f"a card went over its cap: {per} vs {caps}"
    assert min(per.values()) >= need // 5, f"a card got a token share: {per} of {need} bytes"


# --------------------------------------------------------------------------- the loader's wiring

def test_the_load_path_caps_only_when_more_than_one_card_is_asked_for(monkeypatch: pytest.MonkeyPatch) -> None:
    """``load_full_finetune`` must hand the caps to ``plan_device_map`` for >1 card and nothing for 1.

    The single-device call is the one the CPU selftest and every single-GPU rung depend on, so it is
    asserted as the *same call shape* — no ``max_bytes_per_device`` argument at all.
    """
    from kev.checkpoint import LoadOptions

    F = shard.placement()
    calls: list[dict[str, object]] = []

    def fake_plan(module, devices, reserve_bytes=None, **kwargs):
        calls.append({"devices": list(devices), "kwargs": dict(kwargs)})
        return {"": "cpu"}

    monkeypatch.setattr(F, "plan_device_map", fake_plan)
    monkeypatch.setattr(F, "tokenizer_for", lambda checkpoint: None)
    monkeypatch.setattr(F, "DecisionModel", lambda *args, **kwargs: FakeDecisionModel())
    monkeypatch.setattr(F, "verify_placement",
                        lambda module, device_map, devices, where: {
                            "hf_device_map": {"": "cpu"},
                            "weight_bytes_per_device": {"cpu": F.parameter_bytes(module)},
                            "weight_bytes": F.parameter_bytes(module)})
    checkpoint = a_fake_checkpoint()

    _, model, placement = F.load_full_finetune(checkpoint, "cuda", LoadOptions(), gpus=4)
    assert calls[-1]["devices"] == ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
    assert calls[-1]["kwargs"] == {"max_bytes_per_device": F.caps_for(model.lm, calls[-1]["devices"])}
    expected = {f"cuda:{index}": cap for index, cap in sorted(F.caps_for(model.lm, calls[-1]["devices"]).items())}
    assert placement["max_bytes_per_device"] == expected
    assert placement["max_bytes_per_device"] != placement.get("weight_bytes_per_device"), \
        "the caps must be the budget, not a restatement of where the weights landed"

    _, _, single = F.load_full_finetune(checkpoint, "cuda", LoadOptions(), gpus=1)
    assert calls[-1]["devices"] == ["cuda:0"] and calls[-1]["kwargs"] == {}
    assert "max_bytes_per_device" not in single
