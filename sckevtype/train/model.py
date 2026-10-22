"""The decision model this paper trains: kev's, with the text tower pinned and the backbone frozen.

``kev.model.DecisionModel`` is the reference implementation and stays untouched.
What this module adds is the three things a *run* has to be able to state:

**The backbone is the text tower, not the multimodal model.** The Qwen3.5 bases
are composite configs (``model_type: qwen3_5`` with a ``text_config`` *and* a
``vision_config``). ``AutoModelForCausalLM`` currently resolves those to the
text class, so no vision tensor is instantiated — but that is an accident of the
current transformers mapping, not a guarantee. Here the composite config is
resolved once, the text config is handed to the same ``from_pretrained`` call
kev makes, and :func:`verify_text_only` refuses a model that grew a vision tower.

**The backbone is frozen.** ``peft`` freezes the base weights when it attaches
the adapters, and this module proves it on the built model rather than trusting
it: :func:`verify_frozen` raises if any parameter that is neither an adapter nor
the pointer head requires grad. This is the guard behind docs/compute.md — a
LoRA-only run and a fully trainable one differ by more than an order of
magnitude in optimizer state.

**The trainable set is named, not just counted.** :func:`trainable_parameters`
returns the count *and* the names, so ``training_config.json`` records what was
actually trained.

Nothing here imports torch or transformers at module import time: every heavy
import sits inside the function that needs it, so ``import sckevtype.train.model``
works on a CPU-only box.
"""

from __future__ import annotations

import argparse
import contextlib
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

#: Parameter-name prefixes that mean a vision tower got loaded. Qwen3.5's vision
#: tower lives under ``model.visual``; the others cover the usual multimodal
#: layouts, so the guard is not tuned to one checkpoint.
VISION_PREFIXES: tuple[str, ...] = (
    "visual",
    "vision_tower",
    "vision_model",
    "multi_modal_projector",
    "vision_encoder",
)

#: The trainable parameter groups :func:`trainable_parameters` reports.
ADAPTER, HEAD, BACKBONE = "adapter", "head", "backbone"


# --------------------------------------------------------------------------- kev's defaults


class _Captured(Exception):
    """Raised to stop ``kev.train.parse_args`` once its parser has been built."""


def _kev_parser() -> argparse.ArgumentParser:
    """``kev.train``'s own argparse parser, captured before its validation runs.

    The parser is built inline inside ``kev.train.parse_args`` and never
    returned, so there is no way to ask kev for its defaults without running the
    whole function — which errors on things we have not decided yet (a run
    directory that already exists). Intercepting ``parse_args`` gets the parser
    object itself; the exception unwinds before any of kev's checks see it.
    """
    import kev.train

    captured: list[argparse.ArgumentParser] = []
    original = argparse.ArgumentParser.parse_args

    def spy(self: argparse.ArgumentParser, *args: Any, **kwargs: Any) -> Any:
        captured.append(self)
        raise _Captured

    argparse.ArgumentParser.parse_args = spy
    try:
        with contextlib.suppress(_Captured):
            kev.train.parse_args()
    finally:
        argparse.ArgumentParser.parse_args = original
    if not captured:
        raise RuntimeError("could not capture kev.train's parser — has kev.train.parse_args changed?")
    return captured[0]


_DEFAULTS: dict[str, Any] | None = None


def kev_defaults() -> dict[str, Any]:
    """Every ``kev.train`` flag default, read from kev's parser.

    Rule 4 of this project is that no number is copied into this package. The
    hyperparameters below (LoRA rank, pointer dimension, augmentation rates) are
    kev's, and they are read from the parser kev ships rather than restated here,
    so a change upstream is inherited instead of silently diverging.
    """
    global _DEFAULTS
    if _DEFAULTS is None:
        _DEFAULTS = {action.dest: action.default for action in _kev_parser()._actions if action.dest != "help"}
    return dict(_DEFAULTS)


def lora_config(lora_cfg: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The model-construction knobs: kev's defaults with ``lora_cfg`` applied.

    A key kev does not know is an error, not a silent no-op — a typo must not
    quietly train a different model than the one the config describes.
    """
    settings = kev_defaults()
    unknown = sorted(set(lora_cfg or {}) - set(settings))
    if unknown:
        raise ValueError(f"unknown model settings {unknown}; kev.train knows {sorted(settings)}")
    settings.update(lora_cfg or {})
    return settings


# --------------------------------------------------------------------------- text tower


def text_config(config: Any) -> Any | None:
    """The text tower's config when ``config`` is a composite multimodal one, else ``None``.

    ``None`` means "this is already a text config"; it is not an error, and it is
    what keeps the adapter working for a plain text base (a Qwen3-0.6B, say).
    """
    inner = getattr(config, "text_config", None)
    return inner if inner is not None and hasattr(inner, "hidden_size") else None


class _TextTowerLoader:
    """Stand-in for ``AutoModelForCausalLM`` that builds the text tower of a composite base.

    ``kev.model`` hard-codes its backbone call, so the only way to load the text
    tower *explicitly* rather than by whatever class the checkpoint's
    ``model_type`` happens to map to is to go through that same call with a
    loader that supplies the resolved text config. Passing ``config=`` is the
    documented ``from_pretrained`` argument for exactly this.
    """

    def __init__(self, config: Any) -> None:
        self._config = config

    def from_pretrained(self, name: str, **kwargs: Any) -> Any:
        from transformers import AutoModelForCausalLM

        return AutoModelForCausalLM.from_pretrained(name, config=self._config, **kwargs)


@contextlib.contextmanager
def _text_tower_loader(config: Any) -> Iterator[None]:
    """Make ``kev.model`` build the text tower for the duration of the block."""
    import kev.model as km

    original = km.AutoModelForCausalLM
    km.AutoModelForCausalLM = _TextTowerLoader(config)
    try:
        yield
    finally:
        km.AutoModelForCausalLM = original


# --------------------------------------------------------------------------- build


def build_model(
    base: str,
    revision: str | None,
    lora_cfg: Mapping[str, Any] | None = None,
    *,
    text_only: bool = True,
    device: str = "cpu",
    tokenizer: Any = None,
) -> Any:
    """``kev.model.DecisionModel`` for ``base``, with the text tower pinned as the backbone.

    Returns the model (the tokenizer it was built with is available as
    ``model.tokenizer``). Everything past the backbone load — the pointer head,
    the packing encoder, the LoRA recipe, the option-isolation rule — is kev's,
    reached by calling kev's constructor rather than by copying it.
    """
    import kev.model as km
    import torch

    settings = lora_config(lora_cfg)
    tok = tokenizer if tokenizer is not None else km.load_tokenizer(base, revision=revision)
    dtype = torch.bfloat16 if settings["weights_dtype"] == "bf16" else torch.float32
    kwargs = {
        "lora": settings["lora"],
        "revision": revision,
        "head_dim": settings["head_dim"],
        "lora_targets": settings["lora_targets"],
        "option_isolation": bool(settings["option_isolation"]),
        "special_embeddings": bool(settings["special_embeddings"]),
        "dtype": dtype,
    }

    inner = None
    if text_only:
        from transformers import AutoConfig

        inner = text_config(AutoConfig.from_pretrained(base, revision=revision))
    if inner is None:
        model = km.DecisionModel(base, tok, device, **kwargs)
    else:
        with _text_tower_loader(inner):
            model = km.DecisionModel(base, tok, device, **kwargs)

    verify_text_only(model)
    verify_frozen(model)
    return model


# --------------------------------------------------------------------------- the guards


@dataclass(frozen=True)
class Trainables:
    """What a built model will actually update: a count *and* the names behind it."""

    count: int
    names: tuple[str, ...]
    by_kind: dict[str, int]

    def __str__(self) -> str:
        kinds = ", ".join(f"{kind} {size:,}" for kind, size in sorted(self.by_kind.items()) if size)
        return f"{self.count:,} trainable parameters ({kinds})"


def _kind(name: str) -> str:
    """Which group a parameter belongs to: the adapters, the pointer head, or the backbone."""
    if "lora_" in name:
        return ADAPTER
    if name.startswith("head."):
        return HEAD
    return BACKBONE


def trainable_parameters(model: Any) -> Trainables:
    """The trainable parameters of ``model``, by name.

    Duck-typed over ``named_parameters()`` rather than over ``torch.nn.Module``,
    so the accounting can be tested without a GPU, a checkpoint or torch.
    """
    names, by_kind = [], {ADAPTER: 0, HEAD: 0, BACKBONE: 0}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        names.append(name)
        by_kind[_kind(name)] += parameter.numel()
    return Trainables(count=sum(by_kind.values()), names=tuple(names), by_kind=by_kind)


def verify_frozen(model: Any) -> Trainables:
    """Raise unless every trainable parameter is an adapter or the pointer head.

    The failure this guards is silent: an unfrozen backbone still trains, just
    with gradients and two AdamW moments for every one of its ~750M weights
    instead of for the ~11M adapter ones. The run looks healthy and needs many
    GB more, so it is checked rather than assumed.
    """
    trainables = trainable_parameters(model)
    stray = [name for name in trainables.names if _kind(name) == BACKBONE]
    if stray:
        shown = ", ".join(stray[:5]) + (f" (+{len(stray) - 5} more)" if len(stray) > 5 else "")
        raise ValueError(
            f"{len(stray)} backbone parameter(s) require grad — the backbone is not frozen: {shown}. "
            "Emitting an unfrozen backbone doubles the memory of every step for no gain here; "
            "build the model through sckevtype.train.model.build_model."
        )
    return trainables


def verify_text_only(model: Any) -> None:
    """Raise if a vision tower was loaded: the state is text, so the tower is dead weight."""
    offenders = [name for name, _ in model.named_parameters() if any(part in VISION_PREFIXES for part in name.split("."))]
    if offenders:
        raise ValueError(
            f"{len(offenders)} vision parameter(s) were loaded (e.g. {offenders[0]}); "
            "this model reads a rank list, so the vision tower is dead weight — "
            "build the model through build_model(text_only=True)."
        )
