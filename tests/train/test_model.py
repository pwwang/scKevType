"""The model guards: what is trainable, what is not, and what refuses to build.

The guards are the point of this module — the memory anomaly the specification
flags is only testable if "the backbone is frozen" is a fact the code proves
rather than a property of ``peft`` that everyone assumes. They are duck-typed, so
they are tested here without torch, a checkpoint or a card.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from sckevtype.train.model import (
    ADAPTER,
    BACKBONE,
    HEAD,
    Trainables,
    lora_config,
    text_config,
    trainable_parameters,
    verify_frozen,
    verify_text_only,
)


class FakeParameter:
    """The two attributes the guards read off a parameter, and nothing else."""

    def __init__(self, count: int, requires_grad: bool = True) -> None:
        self.count = count
        self.requires_grad = requires_grad

    def numel(self) -> int:
        return self.count


class FakeModel:
    def __init__(self, named: list[tuple[str, FakeParameter]]) -> None:
        self._named = named

    def named_parameters(self):
        return iter(self._named)


def frozen_model() -> FakeModel:
    """A built model's shape: adapters and the pointer head trainable, the rest not."""
    return FakeModel([
        ("lm.model.layers.0.self_attn.q_proj.lora_A.default.weight", FakeParameter(64 * 1024)),
        ("lm.model.layers.0.self_attn.q_proj.lora_B.default.weight", FakeParameter(64 * 1024)),
        ("lm.model.layers.1.linear_attn.in_proj_qkv.lora_A.weight", FakeParameter(32)),
        ("head.proj.weight", FakeParameter(500)),
        ("head.bias", FakeParameter(4)),
        ("lm.model.embed_tokens.weight", FakeParameter(248320 * 1024, requires_grad=False)),
        ("lm.model.layers.0.self_attn.q_proj.base_layer.weight", FakeParameter(1024 * 1024, requires_grad=False)),
    ])


# --------------------------------------------------------------------------- accounting


def test_trainable_parameters_counts_each_group_separately() -> None:
    trainables = trainable_parameters(frozen_model())
    assert trainables.by_kind == {ADAPTER: 64 * 1024 * 2 + 32, HEAD: 504, BACKBONE: 0}
    assert trainables.count == trainables.by_kind[ADAPTER] + trainables.by_kind[HEAD]
    assert "head.proj.weight" in trainables.names
    assert "lm.model.layers.0.self_attn.q_proj.lora_A.default.weight" in trainables.names


def test_trainable_parameters_skips_frozen_tensors() -> None:
    assert not any("embed_tokens" in name for name in trainable_parameters(frozen_model()).names)


def test_the_names_are_the_count() -> None:
    """A count with no names behind it cannot be audited, so the two must agree."""
    model = frozen_model()
    trainables = trainable_parameters(model)
    by_name = {name: parameter for name, parameter in model.named_parameters()}
    assert sum(by_name[name].numel() for name in trainables.names) == trainables.count


def test_str_says_what_it_counts() -> None:
    assert str(Trainables(3, ("head.a",), {ADAPTER: 1, HEAD: 2, BACKBONE: 0})) == "3 trainable parameters (adapter 1, head 2)"


# --------------------------------------------------------------------------- the guards


def test_verify_frozen_returns_the_accounting_for_a_frozen_model() -> None:
    assert verify_frozen(frozen_model()).count == 64 * 1024 * 2 + 32 + 504


def test_verify_frozen_raises_when_a_backbone_tensor_is_unfrozen() -> None:
    """The failure this guards is silent: the run trains, just with the whole model's optimizer state."""
    model = FakeModel([
        ("head.proj.weight", FakeParameter(500)),
        ("lm.model.layers.0.self_attn.q_proj.base_layer.weight", FakeParameter(1024 * 1024)),
    ])
    with pytest.raises(ValueError, match="backbone"):
        verify_frozen(model)


def test_verify_frozen_names_the_offending_tensors() -> None:
    model = FakeModel([("lm.model.norm.weight", FakeParameter(4))])
    with pytest.raises(ValueError, match=r"lm\.model\.norm\.weight"):
        verify_frozen(model)


def test_verify_text_only_passes_a_text_tower() -> None:
    verify_text_only(frozen_model())


def test_verify_text_only_raises_on_a_loaded_vision_tower() -> None:
    model = FakeModel([
        ("lm.model.visual.patch_embed.proj.weight", FakeParameter(1024)),
        ("lm.model.visual.blocks.0.attn.qkv.weight", FakeParameter(1024)),
        ("head.proj.weight", FakeParameter(500)),
    ])
    with pytest.raises(ValueError, match="vision"):
        verify_text_only(model)


def test_a_parameter_merely_named_like_a_vision_tower_is_not_one() -> None:
    """The check is per dot-separated part, so a weight *about* vision is not a vision weight."""
    verify_text_only(FakeModel([("lm.model.layers.0.visual_proj.lora_A.weight", FakeParameter(8))]))


# --------------------------------------------------------------------------- the config


def test_text_config_is_extracted_only_from_a_composite_config() -> None:
    composite = SimpleNamespace(text_config=SimpleNamespace(hidden_size=1024), vision_config=SimpleNamespace())
    assert text_config(composite).hidden_size == 1024
    assert text_config(SimpleNamespace(hidden_size=1024)) is None
    assert text_config(SimpleNamespace(text_config=SimpleNamespace(patch_size=16))) is None


def test_lora_config_overrides_one_key_and_keeps_kevs_others() -> None:
    default = lora_config()
    assert default["lora"] > 0
    changed = lora_config({"lora": 8})
    assert changed["lora"] == 8
    assert {key: value for key, value in changed.items() if key != "lora"} == {
        key: value for key, value in default.items() if key != "lora"
    }


def test_lora_config_rejects_a_key_kev_does_not_know() -> None:
    """A typo must not quietly train a different model than the config describes."""
    with pytest.raises(ValueError, match="unknown model settings"):
        lora_config({"loraa": 8})


def test_kev_defaults_cannot_be_mutated_by_a_caller() -> None:
    """``lora_config`` hands out a copy, so a caller cannot poison every later build."""
    lora_config({"lora": 999})
    assert lora_config()["lora"] != 999
