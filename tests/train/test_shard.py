"""Multi-GPU placement: the balance rule, the refusals, and the record a row is traced by.

Everything here runs without a card. The placement rules live in
``scripts/kev/kev_fullft.py`` and are pure arithmetic over device strings, a module's parameter bytes
and a ``free_bytes`` reading — so the balance rule, the cpu/disk refusal and the record can all be
exercised on this box, and only ``accelerate``'s own fill and the real cards are cluster-only.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from sckevtype.train import shard

ROOT = Path(__file__).resolve().parents[2]

torch = pytest.importorskip("torch", reason="the placement layer is torch's; requires the train extra")


# --------------------------------------------------------------------------- stand-ins


class Layer(torch.nn.Module):
    """A decoder-layer-shaped unit: one Linear, and the class name accelerate must not split."""

    def __init__(self, width: int = 64) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(width, width, bias=False)


class Backbone(torch.nn.Module):
    """The shape both models share: ``embed``, a ``layers`` ModuleList, and a final ``norm``."""

    def __init__(self, layers: int = 4, width: int = 64, embed: int = 128) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(embed, width)
        self.layers = torch.nn.ModuleList(Layer(width) for _ in range(layers))
        self.norm = torch.nn.LayerNorm(width)


class Wrapped(torch.nn.Module):
    """A peft-shaped wrapper: the layer list sits two levels down, with no ``__getattr__`` forwarding.

    ``PeftModel`` forwards unknown attributes to its base model, so ``kev_fullft.no_split_classes``'s
    direct read happens to work through it — this stand-in is the case where it does not, which is why
    :func:`shard.decoder_layers` walks instead of trusting the direct read.
    """

    def __init__(self, backbone: Backbone) -> None:
        super().__init__()
        self.base_model = torch.nn.Module()
        self.base_model.model = backbone


class Decision(torch.nn.Module):
    """A DecisionModel-shaped stand-in: a backbone and a sibling pointer head."""

    def __init__(self, backbone: Backbone | None = None) -> None:
        super().__init__()
        self.lm = backbone if backbone is not None else Backbone()
        self.head = torch.nn.Linear(64, 8)
        for parameter in self.head.parameters():
            parameter.requires_grad_(True)


class FakeParameter:
    """A parameter-shaped object: ``split_layers`` reads ``.device`` and nothing else."""

    def __init__(self, device: str) -> None:
        self.device = device


class FakeLayer(torch.nn.Module):
    """A layer whose parameters can be *told* to be on two cards, which no real module can be on a CPU box."""

    def __init__(self, *devices: str) -> None:
        super().__init__()
        self._devices = devices

    def parameters(self, recurse: bool = True):
        del recurse  # torch's signature; the stand-in's devices are all it has to report
        return iter(FakeParameter(device) for device in self._devices)


class FakeModel(torch.nn.Module):
    def __init__(self, *layers: FakeLayer) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)


def _free(monkeypatch: pytest.MonkeyPatch, module, bytes_each: int) -> None:
    """Make every card report ``bytes_each`` free: the placement arithmetic, without a card."""
    monkeypatch.setattr(module, "free_bytes", lambda spec: bytes_each)


# --------------------------------------------------------------------------- the module it reuses


def test_the_placement_rules_are_kev_fullfts_own() -> None:
    """One implementation for the load path and the training path: the four names are imported, not copied."""
    F = shard.placement()
    assert Path(F.__file__).resolve() == shard.PLACEMENT_SOURCE
    for name in ("plan_device_map", "verify_placement", "no_split_classes", "device_specs", "parameter_bytes",
                 "_as_device", "main_device_of", "balanced_caps", "caps_for", "unit_bytes", "largest_layer_bytes"):
        assert hasattr(F, name), f"{name} is missing from {shard.PLACEMENT_SOURCE}"


def test_a_tree_without_the_placement_module_is_refused_by_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """A staged tree that lost the file must say so, not raise an ImportError from inside an importlib loader."""
    monkeypatch.setattr(shard, "PLACEMENT_SOURCE", ROOT / "scripts" / "kev" / "not-here.py")
    monkeypatch.setattr(shard, "_placement", None)
    with pytest.raises(shard.ShardError, match="no placement module at"):
        shard.devices_for(2, "cuda")


def test_the_import_guard_holds_for_the_shard_module() -> None:
    """``import sckevtype.train.shard`` must not pull torch: the training layer's import guard covers it too."""
    code = ("import json, sys\n"
            "import sckevtype.train.shard\n"
            "print(json.dumps(sorted({m.split('.')[0] for m in sys.modules})))\n")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT, check=False)
    assert result.returncode == 0, result.stderr
    pulled = sorted({"torch", "transformers", "peft", "kev", "accelerate"} & set(json.loads(result.stdout)))
    assert not pulled, f"import sckevtype.train.shard pulled in {pulled}; move that import inside the function"


# --------------------------------------------------------------------------- what may not be split


def test_the_layer_class_is_found_through_a_wrapper() -> None:
    """A peft-wrapped backbone has no ``.layers`` of its own; the class must still be named."""
    assert shard.no_split_classes(Backbone()) == ["Layer"]
    assert shard.no_split_classes(Wrapped(Backbone())) == ["Layer"]
    assert shard.no_split_classes(torch.nn.Linear(4, 4)) == []


def test_the_layer_names_are_the_ones_a_device_map_uses() -> None:
    assert shard.layer_names(Wrapped(Backbone())) == [f"base_model.model.layers.{index}" for index in range(4)]


def test_a_layer_split_across_cards_is_an_offender() -> None:
    """The invariant that makes the shard legal: an adapter must never leave its base weight's card."""
    offenders = shard.split_layers(FakeModel(FakeLayer("cuda:0"), FakeLayer("cuda:0", "cuda:1")))
    assert offenders == {"layers.1": ["cuda:0", "cuda:1"]}
    assert shard.split_layers(FakeModel(FakeLayer("cuda:1"), FakeLayer("cuda:1"))) == {}


def test_the_unit_size_is_the_largest_indivisible_chunk() -> None:
    """accelerate reserves one of these on every card, so the per-card budget has to carry it twice."""
    backbone = Backbone(layers=4, width=64, embed=512)
    assert shard.unit_bytes(backbone) == 512 * 64 * 4          # the embedding, bigger than a layer
    assert shard.unit_bytes(Backbone(layers=4, width=64, embed=16)) == 64 * 64 * 4


# --------------------------------------------------------------------------- the balance rule


def test_the_plan_caps_each_card_at_its_balanced_share(monkeypatch: pytest.MonkeyPatch) -> None:
    """The reason this module exists: a model that fits one card must still be spread over two.

    Free-memory budgeting puts a 4B at bf16 (8 GB, against 40 GB free) entirely on ``cuda:0``, which is
    the single-card run that OOMs. With the caps, the greedy fill has to move on to the second card.
    """
    backbone = Backbone(layers=64, width=256, embed=1024)
    _free(monkeypatch, shard.placement(), 40 * 1024**3)
    device_map, caps = shard.plan(backbone, ["cuda:0", "cuda:1"])
    need = shard.placement().parameter_bytes(backbone)
    share, unit, layer = -(-need // 2), shard.unit_bytes(backbone), shard.largest_layer_bytes(backbone)
    assert caps == {0: max(share + layer, 2 * unit), 1: max(share + layer, 2 * unit)}
    assert max(caps.values()) < need, "the caps must be below the whole model, or there is nothing to spread"
    assert sorted(set(device_map.values())) == [0, 1], device_map
    sizes = {}
    for name, device in device_map.items():
        sizes[device] = sizes.get(device, 0) + shard.placement().parameter_bytes(backbone.get_submodule(name))
    assert sizes[0] and sizes[1]
    assert max(sizes.values()) <= max(caps.values())
    assert min(sizes.values()) >= need // 3, f"the shard is not balanced: {sizes} of {need} bytes"


def test_a_four_card_plan_uses_all_four_cards(monkeypatch: pytest.MonkeyPatch) -> None:
    """The measured failure this cap scheme exists for: one scalar cap leaves the last cards empty.

    ``infer_auto_device_map`` reserves the largest remaining indivisible unit on each card, and that
    reservation shrinks once the embedding is placed, so with a single scalar the later cards get more
    room than the first and the last one gets nothing — on the cluster the 4B's four-card plan used two
    cards and the 9B's three, which the run's own guard refuses. Per-card caps are what make a four-card
    ask a four-card run.
    """
    backbone = Backbone(layers=64, width=256, embed=1024)
    _free(monkeypatch, shard.placement(), 40 * 1024**3)
    devices = [f"cuda:{index}" for index in range(4)]
    device_map, caps = shard.plan(backbone, devices)
    used = sorted({shard.placement()._as_device(device) for device in device_map.values()})
    assert used == devices, f"only {used} of {devices} were used (caps {caps})"
    sizes = {}
    for name, device in device_map.items():
        sizes[shard.placement()._as_device(device)] = (
            sizes.get(shard.placement()._as_device(device), 0)
            + shard.placement().parameter_bytes(backbone.get_submodule(name)))
    need = shard.placement().parameter_bytes(backbone)
    assert min(sizes.values()) >= need // 6, f"a card got a token share: {sizes} of {need} bytes"


def test_the_cap_cannot_talk_the_refusal_out_of_an_over_budget_ask(monkeypatch: pytest.MonkeyPatch) -> None:
    """The capped budgets are what the refusal is computed from, so a cap below the weights still refuses."""
    backbone = Backbone(layers=4, width=64)
    F = shard.placement()
    _free(monkeypatch, F, 4 * 1024**3)
    with pytest.raises(Exception, match="do not fit on 2 device"):
        shard.plan(backbone, ["cuda:0", "cuda:1"])


def test_a_run_with_no_cuda_device_is_refused() -> None:
    assert shard.devices_for(1, "cuda") == ["cuda:0"]
    assert shard.devices_for(4, "cuda") == [f"cuda:{index}" for index in range(4)]
    assert shard.devices_for(4, "cpu") == []
    with pytest.raises(shard.ShardError, match="needs at least two cuda"):
        shard.shard_model(Decision(), 1, "cpu")


# --------------------------------------------------------------------------- the refusals


def test_the_main_device_is_a_string_torch_accepts() -> None:
    """``infer_auto_device_map`` returns bare ints as map values, and ``torch`` rejects ``"0"``.

    The value is used to build the input tensors and to move the pointer head, so an unnormalised
    ``"0"`` is a crash one line after a guard that passed — which is how the 27B load died on the
    cluster (its own log: the guard fired first, on the same raw values).
    """
    F = shard.placement()
    assert F.main_device_of({"embed_tokens": 0, "layers.0": 0}) == "cuda:0"
    assert F.main_device_of({"embed_tokens": "cuda:1", "layers.0": "cuda:0"}) == "cuda:1"
    assert F.main_device_of({"": "cpu"}) == "cpu"
    assert F.main_device_of({"": torch.device("cpu")}) == "cpu"
    torch.device(F.main_device_of({"embed_tokens": 0}))          # raises on "0"
    with pytest.raises(Exception, match="offloads to disk"):
        F.main_device_of({"embed_tokens": 0, "layers.9": "disk"})


def test_verify_placement_refuses_a_map_that_offloads_to_cpu() -> None:
    """A hand-built map, no card: an offloaded run is a CPU pass wearing a GPU's log."""
    F = shard.placement()
    with pytest.raises(Exception, match="mapped outside the requested devices"):
        F.verify_placement(torch.nn.Linear(4, 4), {"": "cpu"}, ["cuda:0"], "test")


def test_verify_placement_refuses_a_parameter_that_is_not_on_a_requested_device() -> None:
    """The map and the live parameters are two readings; this is the one a map-only check cannot see."""
    F = shard.placement()
    with pytest.raises(Exception, match=r"parameters live on \['cpu'\]"):
        F.verify_placement(torch.nn.Linear(4, 4), {"": "cuda:0"}, ["cuda:0"], "test")


def test_verify_shard_refuses_a_run_that_got_fewer_cards_than_it_asked_for() -> None:
    """A four-card ask that used one card has the memory story of one card; that is not a silent downgrade."""
    model = Decision()
    with pytest.raises(shard.ShardError, match="of the 2 requested card"):
        shard.verify_shard(model, {"": "cpu"}, ["cpu", "cuda:0"], "test")


def test_verify_shard_records_where_every_parameter_went() -> None:
    """The record a row is traced by: the device list, and the bytes of weights and trainables per device."""
    model = Decision()
    record = shard.verify_shard(model, {"": "cpu"}, ["cpu"], "test")
    assert record["requested_gpus"] == 1 and record["devices"] == ["cpu"] and record["devices_used"] == ["cpu"]
    assert record["main_device"] == record["head_device"] == "cpu"
    assert record["weight_bytes_per_device"] == {"cpu": shard.placement().parameter_bytes(model.lm)}
    trainable = sum(parameter.numel() * parameter.element_size()
                    for parameter in model.parameters() if parameter.requires_grad)
    assert record["trainable_bytes_per_device"] == {"cpu": trainable} and trainable > 0
    assert record["layers_per_device"] == {"cpu": 4}
    assert record["hf_device_map"] == {"": "cpu"}
    assert json.loads(json.dumps(record, allow_nan=False)) == record


def test_the_description_names_the_devices_and_the_trainable_bytes() -> None:
    record = shard.verify_shard(Decision(), {"": "cpu"}, ["cpu"], "test")
    described = shard.describe(record)
    assert "sharded over ['cpu']" in described and "trainable" in described and "head cpu" in described


# --------------------------------------------------------------------------- the per-card peaks


def test_device_peaks_is_for_the_cards_a_sharded_run_used() -> None:
    with pytest.raises(shard.ShardError, match="for the cards a sharded run used"):
        shard.DevicePeaks(["cpu"])
    with pytest.raises(shard.ShardError, match="for the cards a sharded run used"):
        shard.DevicePeaks([])


def test_device_peaks_reads_every_card_it_was_given(monkeypatch: pytest.MonkeyPatch) -> None:
    """``kev.device.allocated_bytes`` reads device 0 alone — the one card a shard exists to relieve."""
    seen = []

    def fake_max_memory_allocated(index: int) -> int:
        seen.append(index)
        return (index + 1) * 1024

    monkeypatch.setattr(torch.cuda, "max_memory_allocated", fake_max_memory_allocated)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda index: seen.append(("reset", index)))
    peaks = shard.DevicePeaks(["cuda:0", "cuda:1"])
    assert peaks.read() == {"cuda:0": 1024, "cuda:1": 2048}
    assert peaks.max() == 2048
    peaks.reset()
    assert seen[-2:] == [("reset", 0), ("reset", 1)]


# --------------------------------------------------------------------------- the CLI


@pytest.fixture
def kev_train():
    return pytest.importorskip("kev.train", reason="the flags are kev's; requires the train extra")


def test_the_gpus_flag_needs_cuda_and_a_positive_count(kev_train) -> None:
    from sckevtype.train.train import parse_args

    base = ["--suite", "s", "--base", "b", "--out", "o"]
    assert parse_args([*base, "--gpus", "4", "--device", "cuda"]).gpus == 4
    assert parse_args(base).gpus == 1
    with pytest.raises(SystemExit):
        parse_args([*base, "--gpus", "2", "--device", "cpu"])
    with pytest.raises(SystemExit):
        parse_args([*base, "--gpus", "0", "--device", "cuda"])


def test_the_gpus_flag_never_reaches_kevs_namespace(kev_train, tmp_path: Path) -> None:
    """kev's parser has no ``--gpus``; handing it over would fail the run at its first line."""
    from sckevtype.train.train import kev_namespace, parse_args

    suite = tmp_path / "suite"
    suite.mkdir()
    (suite / "manifest.json").write_text(json.dumps({"context": {"max_state": 1664}}), encoding="utf-8")
    ns = kev_namespace(parse_args(["--suite", str(suite), "--base", "b", "--gpus", "4", "--device", "cuda",
                                   "--out", "o"]), "o")
    assert not hasattr(ns, "gpus")


def test_the_placement_block_carries_the_device_list_and_the_per_device_bytes() -> None:
    """A row is traced by this block: the device list, and the bytes of weights and trainables per device."""
    record = shard.verify_shard(Decision(), {"": "cpu"}, ["cpu"], "test")
    assert {"devices", "devices_used", "weight_bytes_per_device", "trainable_bytes_per_device",
            "layers_per_device", "head_device", "main_device", "reserve_bytes",
            "max_bytes_per_device", "requested_gpus"} <= set(record)
    assert SimpleNamespace(**record).devices_used == ["cpu"]
