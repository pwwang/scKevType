#!/usr/bin/env python3
"""Where one training step's activation memory actually goes, on the CPU, with no GPU involved.

The A100 measured 39.4 GB at batch 1 and the probe predicted single-digit GB. A total says the
prediction is wrong but not which term is. This attributes the memory to its producer instead:
every tensor autograd saves for the backward pass is intercepted
(`torch.autograd.graph.saved_tensors_hooks`) and summed **by the op that produced it**
(`type(t.grad_fn).__name__`), per forward / backward / optimizer phase.

What this measures and what it does not:

* it measures **saved activation bytes** — the memory the backward pass requires to exist, which is
  the dominant and the configurable term when gradient checkpointing is off (every OOM recorded in
  `docs/SPEC-step10d-memory.md` was recorded with `--checkpointing 0`);
* it does **not** measure a CUDA peak. Transient per-layer working sets, the caching allocator's
  reserve and kernel-specific scratch are invisible here. It is a driver measurement, not a peak.

The rows are built from a real record's own tokens, cut or repeated to a target length, so the
shape sweep is controlled while the vocabulary and the branch structure stay real.

    python scripts/probe-activation-memory.py [--lens 512,1024,2048] [--rows 1] [--checkpointing 0]
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class Saved:
    """Autograd's saved tensors, summed by the op that produced them, while a block runs."""

    def __init__(self) -> None:
        import torch

        self.totals: dict[str, int] = defaultdict(int)
        self.hooks = torch.autograd.graph.saved_tensors_hooks(self._pack, lambda tensor: tensor)

    def _pack(self, tensor):
        name = type(tensor.grad_fn).__name__ if tensor.grad_fn is not None else "leaf/input"
        self.totals[name] += tensor.numel() * tensor.element_size()
        return tensor

    def reset(self) -> None:
        self.totals = defaultdict(int)

    def total(self) -> int:
        return sum(self.totals.values())


def build_rows(model, tokenizer, record: dict, length: int, limits: dict) -> dict:
    """One real encoding, cut to exactly `length` tokens, laid out as a single causal row.

    The tokens are the record's own — its state, then its branch tokens repeated or cut to fill — so
    the vocabulary is real while the row length is under our control. The layout is the one
    `kev.model.rows_of` requires (a state segment 0, then one branch segment 1), which is what the
    row form feeds the backbone.

    Positions are a plain `range`: only the length matters to the accounting, and a monotone
    sequence is what a causal row has anyway.
    """
    from kev.data import materialize
    from kev.model import rows_of

    enc = model.encode(tokenizer, materialize(record), strict=True, **limits)
    state, _, branches = rows_of(enc)
    body = [token for branch in branches for token in branch["ids"]]
    state_len = min(len(state), max(length - 2, 0))
    need = length - state_len
    body = (body * (need // max(len(body), 1) + 1))[:need]
    ids = list(state[:state_len]) + body
    return {
        "ids": ids,
        "pos": list(range(len(ids))),
        "seg": [0] * state_len + [1] * len(body),
        "decide_idx": [len(ids) - 1],
        "opt_idx": [[len(ids) - 2]],
        "option_isolation": False,
    }


def main() -> int:

    import torch
    from kev.model import load_tokenizer, training_context
    from kev.suite import read_manifest
    from kev.train import load_split

    from sckevtype.train import model as model_module

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--suite", default="artifacts/suites/real-v2")
    parser.add_argument("--lens", default="512,1024,2048,3392",
                        help="row lengths to sweep; 3,392 is real-v2's median training row")
    parser.add_argument("--rows", type=int, default=1, help="questions per record (rows in the padded batch)")
    parser.add_argument("--checkpointing", default="0,1")
    parser.add_argument("--weights_dtype", default="fp32", help="kev.train's default is fp32")
    parser.add_argument("--limit_seconds", type=float, default=600.0)
    args = parser.parse_args()

    manifest = read_manifest(args.suite)
    pin = manifest["context"]["tokenizer"]
    tokenizer = load_tokenizer(pin["id"], revision=pin["revision"])
    record = load_split(args.suite, "train")[0]
    # the same budget the trainer hands kev, from the same manifest — not kev's 384 default
    context = training_context(manifest["context"]["max_state"])
    limits = {"max_state": context["max_state"], "max_branch": context["max_branch"]}
    print(f"suite {args.suite}: max_state {context['max_state']}, row limit {context['max_branch']}")

    built = model_module.build_model(
        pin["id"], pin["revision"],
        {"weights_dtype": args.weights_dtype, "lora_targets": "all"},
        device="cpu", tokenizer=tokenizer,
    )
    built.train()
    built.lm.config.use_cache = False
    print(f"model built on cpu: {type(built.lm).__name__}, hybrid={built.hybrid}, "
          f"weights {next(built.lm.parameters()).dtype}, "
          f"layers {built.lm.config.num_hidden_layers}, "
          f"full-attention every {built.lm.config.full_attention_interval}", flush=True)

    autocast = torch.autocast("cpu", dtype=torch.bfloat16)
    optimizer = torch.optim.AdamW(list(built.trainable_parameters()), lr=1e-6)

    print(f"\n{'ckpt':>4s} {'L':>6s} {'rows':>5s} {'slots':>8s} {'fwd saved':>12s} {'bwd saved':>12s} "
          f"{'total':>12s} {'opt':>10s}  top op")
    for checkpointing in [int(v) for v in args.checkpointing.split(",") if v]:
        if checkpointing:
            built.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        else:
            built.lm.gradient_checkpointing_disable()
        for length in [int(v) for v in args.lens.split(",") if v]:
            enc = build_rows(built, tokenizer, record, length, limits)
            length = len(enc["ids"])
            batch = [enc for _ in range(args.rows)]
            saved = Saved()
            optimizer.zero_grad(set_to_none=True)
            try:
                with saved.hooks:
                    with autocast:
                        logits = built.forward_batch(batch)
                    fwd = dict(saved.totals)
                    saved.reset()
                    loss = sum(z.float().mean() for z in logits[0])
                    loss.backward()
                    bwd = dict(saved.totals)
                    saved.reset()
                    optimizer.step()
                    opt = dict(saved.totals)
            except (RuntimeError, torch.OutOfMemoryError) as error:
                print(f"{checkpointing:>4d} {length:>6d} {args.rows:>5d} {'':>8s}   stopped: "
                      f"{type(error).__name__}: {str(error)[:90]}", flush=True)
                optimizer.zero_grad(set_to_none=True)
                continue
            optimizer.zero_grad(set_to_none=True)
            total = sum(fwd.values()) + sum(bwd.values())
            top = sorted(fwd.items(), key=lambda kv: -kv[1])[:3]
            print(f"{checkpointing:>4d} {length:>6d} {args.rows:>5d} {length * args.rows:>8,d} "
                  f"{sum(fwd.values()) / 2**30:11.3f}G {sum(bwd.values()) / 2**30:11.3f}G "
                  f"{total / 2**30:11.3f}G {sum(opt.values()) / 2**20:9.1f}M  "
                  + ", ".join(f"{name} {value / 2**30:.2f}G" for name, value in top), flush=True)
            print("          forward saved by op (top 8):")
            for name, value in sorted(fwd.items(), key=lambda kv: -kv[1])[:8]:
                print(f"            {name:42s} {value / 2**30:9.3f} G")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
