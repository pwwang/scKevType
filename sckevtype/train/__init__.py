"""The training adapter: a Step-2 corpus in, a trained pointer head + LoRA out.

This layer is a *thin adapter over the pinned foundation method* in ``kev/``. It
re-implements none of it: the pointer readout, the packing/option-isolation
encoder, the augmentation, the losses and the suite loader are all imported from
``kev`` and used through their own entry points.

    model    -> kev's decision model with the text tower pinned, the backbone
                frozen, and the trainable set named and checked
    suite    -> a corpus becomes a kev-loadable suite (JSONL + manifest)
    train    -> the CLI that runs a bounded, seeded, provenance-recording run
    compute  -> the capacity measurement behind docs/compute.md

Dependencies point one way: ``train`` imports ``contracts`` and ``kev`` and
nothing else in the package. ``torch``/``transformers`` are imported inside
functions, so ``import sckevtype.train.suite`` works on a CPU-only box.
"""

from __future__ import annotations

__all__ = ["build_model", "build_suite", "trainable_parameters", "verify_frozen", "verify_text_only"]


def __getattr__(name: str):
    """Lazy re-exports: importing this package must not pull torch or transformers."""
    if name in ("build_model", "trainable_parameters", "verify_frozen", "verify_text_only"):
        from . import model

        return getattr(model, name)
    if name == "build_suite":
        from . import suite

        return suite.build_suite
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
