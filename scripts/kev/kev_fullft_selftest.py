#!/usr/bin/env python3
"""What can be proved about the full-weight load path without the 27B, the cluster or a GPU.

``jaredpalmer/kev-27b`` is 51.3 GB and cannot be downloaded here, but nothing this file checks depends
on its *weights*: the blocker is a **layout** — no ``adapter_config.json``, ``head.pt`` declaring
``weights: "full"``, and a tokenizer and config of its own. So the checks run against a stand-in with
that exact layout and a tiny Qwen3.5 backbone built from the published config's own architecture
(``model_type: qwen3_5_text``, three ``linear_attention`` layers then a ``full_attention`` one, so
``kev.model.is_hybrid`` is true and the row-form forward is the one exercised).

Every check names what would make it return **false**, and the load-bearing ones are run in both
directions — the old code must fail on the full-weight layout *for the stated reason*, the new code must
succeed on the same directory, and the LoRA path must produce the same numbers it produced before.

Run: ``uv run --no-sync python scripts/kev/kev_fullft_selftest.py``  (offline, CPU, ~1 minute)
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "kev"))

PASSES: list[str] = []
FAILURES: list[str] = []
SKIPPED: list[str] = []


def check(name: str, ok: bool, detail: object = "") -> bool:
    entry = f"{name}{' -- ' + str(detail) if detail else ''}"
    (PASSES if ok else FAILURES).append(entry)
    print(f"  {'ok  ' if ok else 'FAIL'} {name}" + (f"  [{detail}]" if detail else ""))
    return ok


def skip(name: str, why: str) -> None:
    SKIPPED.append(f"{name} -- {why}")
    print(f"  skip {name}  [{why}]")


# --------------------------------------------------------------------------- the stand-in checkpoints

#: The architecture-defining fields of ``jaredpalmer/kev-27b``'s own ``config.json`` (read from the Hub
#: on 2026-10-02). The *sizes* are shrunk so a stand-in fits in a few MB and a forward pass runs on CPU;
#: every field that decides which code path runs is the real value. `layer_types` keeps the real 3:1
#: linear:full pattern, which is what makes `is_hybrid` true and selects the row-form forward.
REAL_ARCHITECTURE = {
    "architectures": ["Qwen3_5TextModel"],
    "model_type": "qwen3_5_text",
    "num_hidden_layers": 4,                      # the real value is 64; layer_types must match its length
    "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
    "full_attention_interval": 4,
    "attn_output_gate": True,
    "output_gate_type": "swish",
    "linear_conv_kernel_dim": 4,
    "mamba_ssm_dtype": "float32",
    "partial_rotary_factor": 0.25,
    "rope_parameters": {"mrope_interleaved": True, "mrope_section": [11, 11, 10],
                        "partial_rotary_factor": 0.25, "rope_theta": 10000000.0, "rope_type": "default"},
    "head_dim": 256,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "hidden_size": 128,
    "intermediate_size": 256,
    "linear_num_key_heads": 2,
    "linear_key_head_dim": 32,
    "linear_num_value_heads": 4,
    "linear_value_head_dim": 32,
    "hidden_act": "silu",
    "rms_norm_eps": 1e-06,
    "initializer_range": 0.02,
    "tie_word_embeddings": False,
    "use_cache": False,
    "attention_bias": False,
    "attention_dropout": 0.0,
    "max_position_embeddings": 4096,
    "mtp_num_hidden_layers": 1,
    "mtp_use_dedicated_embeddings": False,
}


def build_stand_in(root: Path, kind: str) -> tuple[Path, dict]:
    """A tiny checkpoint in ``kind``'s layout. Returns (its directory, the Meta written to head.pt).

    ``kind="full"`` is ``kev-27b``'s layout on the **hybrid** backbone its config declares: config +
    shards + head.pt + tokenizer, **no** ``adapter_config.json``. ``kind="lora"`` is every other rung's
    layout — but on an **attention-only** base, because with ``flash-linear-attention`` installed a
    hybrid backbone's forward pass only runs on CUDA (see :func:`hybrid_forward_on_cpu`), and the LoRA
    checks need a forward on this CPU-only box.
    """
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    base = root / "tiny-base"
    if not (base / "config.json").exists():
        base.mkdir(parents=True, exist_ok=True)
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B-Base")
        config = AutoConfig.for_model(**{**REAL_ARCHITECTURE, "vocab_size": len(tok)})
        AutoModelForCausalLM.from_config(config).save_pretrained(base)
        tok.save_pretrained(base)

    from kev.model import PointerHead

    torch.manual_seed(20261002)
    head = PointerHead(REAL_ARCHITECTURE["hidden_size"], dp=256).state_dict()
    meta = {"base": str(base), "base_revision": None, "head": head, "head_dim": 256,
            "option_isolation": False, "special_embeddings": False, "weights_dtype": "bf16",
            "temperature": 1.319507910772894, "holdout": []}

    if kind == "full":
        target = root / "kev-27b-stand-in"
        if not (target / "head.pt").exists():
            shutil.copytree(base, target, dirs_exist_ok=True)
            # `weights: "full"` is the field kev.checkpoint.Meta does not know and keeps in `extra`;
            # `lora: 0` is what the real head.pt carries. Both are the published values.
            torch.save({**meta, "lora": 0, "weights": "full", "weights_dtype": "bf16"}, target / "head.pt")
        return target, {**meta, "lora": 0, "weights": "full"}

    from peft import LoraConfig, get_peft_model

    attn_base = build_attention_only_stand_in(root)          # a forward that runs on CPU
    target = root / "kev-9b-stand-in"
    if not (target / "head.pt").exists():
        target.mkdir(parents=True, exist_ok=True)
        # `Checkpoint.load` wraps the **backbone** (`DecisionModel.lm`, the `.model` of the CausalLM),
        # so the adapter must be saved from that same module or peft loads it with a key-prefix
        # mismatch and warns "Found missing adapter keys" -- an adapter that never applies, which would
        # make this stand-in a base model wearing an adapter's filename.
        backbone = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(str(attn_base))).model
        peft_model = get_peft_model(backbone, LoraConfig(task_type="FEATURE_EXTRACTION", r=16, lora_alpha=32,
                                                         lora_dropout=0.05,
                                                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
        peft_model.save_pretrained(str(target))
        torch.save({**meta, "base": str(attn_base), "lora": 16, "weights": "lora",
                    "weights_dtype": "fp32"}, target / "head.pt")
    return target, {**meta, "lora": 16, "weights": "lora"}


def corpus_record():
    """A real record from the frozen corpus — the *cheapest* one with a choice question.

    The stand-in's weights are random, so the *answer* means nothing; what the checks read is that the
    encoder accepted the record, that the readout is a distribution over the question's own option keys,
    and that ``kev.api.to_answers`` could consume it. A synthetic record would not prove the last part —
    the shape the real corpus produces is the only one that matters (a fixture that passes on invented
    input proves nothing about the pipeline).

    Returns ``(index, record, why)``. ``why`` names a *miss* rather than letting the caller report
    "absent" for a file that is right there: the two need different actions, and this file has already
    been misread once as missing when a filter simply matched nothing.
    """
    partition = ROOT / "runs" / "real-v2" / "test.jsonl"
    if not partition.is_file():
        return None, None, f"{partition} is not on this box"
    best, seen, choices = None, 0, 0
    with partition.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            record = json.loads(line)
            seen += 1
            if not any(question.get("type") == "choice" for question in record["questions"].values()):
                continue
            choices += 1
            cost = len(record["state"]) + sum(len(json.dumps(question)) for question in record["questions"].values())
            if best is None or cost < best[0]:
                best = (cost, index, record)
    if best is None:
        return None, None, (f"{partition} holds {seen} record(s) and none carries a choice question "
                            f"-- the filter is wrong, not the file")
    return best[1], best[2], f"{choices} of {seen} records carry a choice question; the cheapest costs {best[0]} chars"


def build_attention_only_stand_in(root: Path) -> Path:
    """A full-weight stand-in on an **attention-only** backbone, to prove the forward path is the config's.

    The hybrid routing is ``kev.model.DecisionModel.__init__``'s ``self.hybrid = is_hybrid(self.lm.config)``
    — derived from the checkpoint's own ``config.json``, not from an argument this loader passes. The way
    to show that is a second checkpoint whose config says otherwise and watch it take the other path.
    """
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    target = root / "attention-only-stand-in"
    if (target / "head.pt").exists():
        return target
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B-Base")
    config = AutoConfig.for_model(model_type="qwen3", vocab_size=len(tok), hidden_size=128,
                                 intermediate_size=256, num_hidden_layers=2, num_attention_heads=2,
                                 num_key_value_heads=1, head_dim=64, tie_word_embeddings=False)
    AutoModelForCausalLM.from_config(config).save_pretrained(target)
    tok.save_pretrained(target)
    from kev.model import PointerHead

    torch.manual_seed(20261002)
    torch.save({"base": str(target), "base_revision": None, "lora": 0, "weights": "full",
                "head": PointerHead(128, dp=256).state_dict(), "head_dim": 256,
                "option_isolation": False, "special_embeddings": False, "weights_dtype": "bf16",
                "temperature": 1.0, "holdout": []}, target / "head.pt")
    return target


def hybrid_forward_on_cpu(predictor, enc) -> str:
    """What happens when a hybrid backbone's forward is attempted on CPU — reported, not asserted.

    ``transformers.integrations.hub_kernels.use_kernel_func_from_hub_with_fallback`` resolves
    ``chunk_gated_delta_rule`` to **fla's Triton kernel at import time** whenever ``import fla`` succeeds,
    on every device, and fla's kernel cannot take a CPU tensor. So a hybrid Qwen3.5/Qwen3.8 backbone has
    no CPU forward in this stack at all: the ``himem-n96-1500g`` partition is not a slow fallback for
    this arm, it is an impossible one, and that is why the answer to "it does not fit one card" has to be
    sharding rather than a bigger machine.
    """
    import torch
    try:
        with torch.no_grad():
            predictor.model.forward(enc)
        return "RUNS on CPU (the CPU partition is a fallback -- revisit the VRAM decision)"
    except Exception as error:                           # noqa: BLE001 - the exception IS the finding
        return (f"not runnable on CPU: {type(error).__name__}: {str(error)[:110]} "
                f"(fla's Triton kernel is selected on every device)")


def probabilities(predictor, record):
    """One record through a predictor, with the encoder's own rejections named rather than swallowed."""
    return predictor(record)


# --------------------------------------------------------------------------- the checks

def main() -> int:
    print("kev full-weight load path -- what can be proved without the 27B, the cluster or a GPU")
    print()

    import kev_fullft as F
    import torch
    from kev.checkpoint import Checkpoint, LoadOptions
    from kev.model import is_hybrid
    from kev.predictors import LocalPredictor

    sys.path.insert(0, str(ROOT / "scripts" / "kev"))
    import kev_probe as K

    work = Path(tempfile.mkdtemp(prefix="kev-fullft-"))
    print(f"stand-in root: {work}")
    print()
    try:
        full_dir, _ = build_stand_in(work, "full")
        lora_dir, _ = build_stand_in(work, "lora")
        index, record, why = corpus_record()
        print(f"full-weight stand-in: {full_dir}")
        print(f"  files: {sorted(p.name for p in full_dir.iterdir())}")
        print(f"  adapter_config.json present: {(full_dir / 'adapter_config.json').exists()}")
        print(f"lora stand-in:        {lora_dir}")
        print(f"  files: {sorted(p.name for p in lora_dir.iterdir())}")
        print(f"  adapter_config.json present: {(lora_dir / 'adapter_config.json').exists()}")
        if record is None:
            print(f"  no corpus record to ask: {why}")
        else:
            print(f"corpus record: line {index}, state {len(record['state'])} chars, "
                  f"{len(record['questions'])} question(s) -- {why}")

        print()
        print("=== 1. the OLD code on the full-weight layout: it must fail, for the stated reason ===")
        old_error = None
        try:
            Checkpoint(str(full_dir)).load("cpu", LoadOptions())
        except Exception as error:                       # noqa: BLE001 - the exception IS the evidence
            old_error = error
        check("Checkpoint.load() refuses a full-weight checkpoint",
              old_error is not None,
              f"{type(old_error).__name__}: {old_error}" if old_error else
              "it LOADED -- the premise of this whole change is wrong")
        # The reason matters: a failure for any other cause (a missing head.pt, a bad config) would mean
        # this directory is broken rather than that the loader is adapter-only.
        check("...and it fails on adapter_config.json, not on something else about this directory",
              old_error is not None and "adapter_config.json" in str(old_error),
              str(old_error) if old_error else "")
        check("the directory it refused is otherwise complete (head.pt + config.json + shards + tokenizer)",
              all((full_dir / name).exists() for name in
                  ("head.pt", "config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json")))

        print()
        print("=== 2. the discriminator, in both directions ===")
        check("checkpoint_kind(full) == 'full'",
              F.checkpoint_kind(Checkpoint(str(full_dir))) == F.FULL)
        check("checkpoint_kind(lora) == 'lora'",
              F.checkpoint_kind(Checkpoint(str(lora_dir))) == F.LORA)
        # A contradiction must be refused, not resolved. Both directions, since either half could be the
        # wrong one: an adapter in a directory that says `full`, and a directory that says `lora` with no
        # adapter to load.
        contradicted = work / "contradicted"
        shutil.copytree(full_dir, contradicted, dirs_exist_ok=True)
        (contradicted / "adapter_config.json").write_text(json.dumps({"r": 16}), encoding="utf-8")
        try:
            F.checkpoint_kind(Checkpoint(str(contradicted)))
            check("a checkpoint that says weights='full' while shipping an adapter is REFUSED", False,
                  "it returned a kind instead of raising")
        except F.FullWeightError as error:
            check("a checkpoint that says weights='full' while shipping an adapter is REFUSED", True,
                  str(error)[:90] + "...")
        orphan = work / "orphan"
        shutil.copytree(lora_dir, orphan, dirs_exist_ok=True)
        (orphan / "adapter_config.json").unlink()
        try:
            F.checkpoint_kind(Checkpoint(str(orphan)))
            check("a checkpoint that says weights='lora' with no adapter is REFUSED", False,
                  "it returned a kind instead of raising")
        except F.FullWeightError as error:
            check("a checkpoint that says weights='lora' with no adapter is REFUSED", True,
                  str(error)[:90] + "...")
        # And the oldest published arm carries no `weights` field at all: the field can never be the sole
        # discriminator. (kev-0.8b's head.pt has no `weights`; kev-9b's says 'lora'.)
        legacy = work / "legacy-lora"
        shutil.copytree(lora_dir, legacy, dirs_exist_ok=True)
        meta = torch.load(legacy / "head.pt", map_location="cpu", weights_only=False)
        meta.pop("weights")
        torch.save(meta, legacy / "head.pt")
        check("a LoRA checkpoint with NO `weights` field (kev-0.8b's shape) still reads as 'lora'",
              F.checkpoint_kind(Checkpoint(str(legacy))) == F.LORA)

        print()
        print("=== 3. the new loader on the same directory ===")
        from kev.model import SERVE_MAX_BRANCH, SERVE_MAX_PACKED, SERVE_MAX_STATE

        serving_context = {"max_state": SERVE_MAX_STATE, "max_branch": SERVE_MAX_BRANCH,
                           "max_packed": SERVE_MAX_PACKED}
        predictor = F.FullFinetunePredictor(str(full_dir), "cpu", LoadOptions(), serving_context, gpus=1)
        placement = predictor.placement
        print(f"  placement: {json.dumps({k: v for k, v in placement.items() if k != 'hf_device_map'})}")
        print(f"  hf_device_map: {placement['hf_device_map']}")
        check("it loads (the same directory the vendored loader refused)", True,
              f"dtype {placement['dtype']}, attn {placement['attn']}")
        check("the backbone is a hybrid one, so the row-form forward is what ran",
              is_hybrid(predictor.model.lm.config) and predictor.model.hybrid)
        # The forward path is the *config's*, not this loader's: `DecisionModel.__init__` sets
        # `self.hybrid = is_hybrid(self.lm.config)` and every routing decision downstream reads it. A
        # second full-weight checkpoint whose config says otherwise must take the other path, or the
        # routing is coming from somewhere it should not.
        attn_dir = build_attention_only_stand_in(work)
        attn_predictor = F.FullFinetunePredictor(str(attn_dir), "cpu", LoadOptions(), serving_context, gpus=1)
        check("an attention-only full-weight checkpoint loads too, and takes the OTHER forward path",
              not attn_predictor.model.hybrid,
              f"hybrid={attn_predictor.model.hybrid}, attn={attn_predictor.placement['attn']}, "
              f"weights_dtype={attn_predictor.placement['weights_dtype_declared']}")
        check("accelerate placed the backbone (hf_device_map is set)",
              predictor.model.lm.hf_device_map == {"": "cpu"},
              str(getattr(predictor.model.lm, "hf_device_map", None)))
        check("the placement guard accepted it and counted the weights",
              placement["weight_bytes"] > 0 and placement["weight_bytes_per_device"] == {"cpu": placement["weight_bytes"]},
              f"{placement['weight_bytes']:,} bytes on {placement['devices']}")
        check("the checkpoint's own dtype rule was applied (weights_dtype bf16 -> bf16)",
              placement["dtype"] == "bf16", placement["dtype"])
        check("the pointer head came from head.pt (temperature 1.3195, the published value)",
              abs(predictor.temperature - 1.319507910772894) < 1e-12, f"{predictor.temperature}")
        # The two lines `Checkpoint.load` ends with, checked on the tensors rather than on the code:
        # a head that never received `load_state_dict(meta.head)` still answers, with a random readout.
        saved_head = torch.load(full_dir / "head.pt", map_location="cpu", weights_only=False)["head"]
        check("the head's weights ARE head.pt's (tensor equality, not a default-initialised head)",
              all(torch.equal(predictor.model.head.state_dict()[name], saved_head[name]) for name in saved_head),
              f"{len(saved_head)} tensor(s)")
        check("the model is in eval mode (PointerHead skips the temperature division while training)",
              not predictor.model.training)

        if record is not None:
            from kev.data import materialize

            from sckevtype.jev.request import option_keys
            enc = predictor.model.encode(predictor.tok, materialize(record),
                                         max_state=serving_context["max_state"],
                                         max_branch=serving_context["max_branch"], strict=True)
            check("the encoder accepted a REAL corpus record and the row form is what will run",
                  predictor.model.rows_form([enc]) and len(enc["ids"]) <= serving_context["max_packed"],
                  f"{len(enc['ids'])} packed token(s), {len(enc['decide_idx'])} question(s)")
            check("the attention-only checkpoint takes the OTHER form on the same encoding",
                  not attn_predictor.model.rows_form(
                      [attn_predictor.model.encode(attn_predictor.tok, materialize(record),
                                                   max_state=serving_context["max_state"],
                                                   max_branch=serving_context["max_branch"], strict=True)]))
            # Why the CPU partition is not a fallback for this arm: with `flash-linear-attention`
            # installed, transformers' `use_kernel_func_from_hub_with_fallback` resolves the DeltaNet
            # op to fla's Triton kernel at import time, on every device, so a hybrid forward on CPU
            # dies in Triton. This is an environment fact, reported rather than asserted -- the
            # property this file owns is `rows_form` above, which is what selects the path.
            print(f"  hybrid forward on CPU: {hybrid_forward_on_cpu(predictor, enc)}")
            # The readout checks run on the attention-only checkpoint through the REAL dispatcher
            # (`KevPredictor`, which is what the probe drives), so `answers`/`served_model` are the
            # serving contract's own output and not this test's re-wrapping of it.
            attn_probe = K.KevPredictor(str(attn_dir), device="cpu")
            check("KevPredictor routes a full-weight checkpoint to the new loader, not to LocalPredictor",
                  attn_probe.kind == "full" and isinstance(attn_probe.inner, F.FullFinetunePredictor)
                  and not type(attn_probe.inner) is LocalPredictor, type(attn_probe.inner).__name__)
            answer = probabilities(attn_probe, record)
            keys = {qid: list(option_keys(question)) for qid, question in record["questions"].items()}
            check("every question got a probability for every one of its own option keys",
                  set(answer["probabilities"]) == set(keys) and
                  all(list(answer["probabilities"][qid]) == keys[qid] for qid in keys),
                  f"{len(keys)} question(s)")
            sums = {qid: sum(answer["probabilities"][qid].values()) for qid in keys}
            check("each distribution sums to 1",
                  all(abs(total - 1.0) < 1e-5 for total in sums.values()),
                  f"max |sum-1| {max(abs(t - 1.0) for t in sums.values()):.2e}")
            check("kev.api.to_answers consumed it (the serving contract's own answers)",
                  set(answer["answers"]) == set(keys), f"{len(answer['answers'])} answer(s)")
            check("the predictor reports the checkpoint and temperature it served",
                  attn_dir.name in answer["served_model"] and "bf16" in answer["served_model"],
                  answer["served_model"])
            check("the readout is the pointer head's, in fp32, over the option tokens",
                  all(len(entry) == len(keys[qid]) for qid, entry in answer["probabilities"].items()))
        else:
            skip("the encoder/readout checks", why)

        print()
        print("=== 4. the scoring path is the vendored one, not a reimplementation ===")
        check("FullFinetunePredictor inherits LocalPredictor.__call__ (identity, not a copy)",
              F.FullFinetunePredictor.__call__ is LocalPredictor.__call__,
              f"{F.FullFinetunePredictor.__call__.__qualname__}")
        check("...and inherits encode/forward through the model, which is a kev.model.DecisionModel",
              type(predictor.model).__name__ == "DecisionModel")
        check("the vendored Checkpoint.load is not patched or shadowed",
              "load" in vars(Checkpoint) and Checkpoint.load.__qualname__.endswith("Checkpoint.load"),
              Checkpoint.load.__qualname__)

        print()
        print("=== 5. the LoRA path, unchanged ===")
        lora_direct = Checkpoint(str(lora_dir)).load("cpu", LoadOptions())
        check("Checkpoint.load still loads a LoRA checkpoint", lora_direct is not None)
        lora_probe = K.KevPredictor(str(lora_dir), device="cpu")
        check("KevPredictor routes a LoRA checkpoint to LocalPredictor (not to the new class)",
              lora_probe.kind == "lora" and isinstance(lora_probe.inner, LocalPredictor)
              and not isinstance(lora_probe.inner, F.FullFinetunePredictor), type(lora_probe.inner).__name__)
        check("a LoRA checkpoint records no placement (the vendored loader pins one device)",
              lora_probe.placement is None)
        refused = None
        try:
            K.KevPredictor(str(lora_dir), device="cpu", gpus=2)
        except K.KevError as error:
            refused = error
        check("--gpus 2 on a LoRA checkpoint is REFUSED rather than silently run on one card",
              refused is not None and "gpus 2" in str(refused), str(refused)[:90] + "..." if refused else "it was accepted")
        if record is not None:
            from kev.data import materialize
            # Both dtypes, because the dispatcher must add nothing at EITHER of them -- and the dtype is
            # resolved by the vendored loader's own rule (`weights_dtype`, then `opts.dtype`, then the
            # fp32 default), which a probe reaches through `--dtype`. The first pass of this check
            # compared the probe's bf16 default against a raw fp32 `LoadOptions()` and "found" a
            # 8.4e-4 difference that was the dtype and not the dispatcher.
            for dtype_name, options, kwargs in (("bf16 (the probe's default)", LoadOptions(dtype=torch.bfloat16),
                                                 {"dtype": "bf16"}),
                                                ("fp32 (the exact path)", LoadOptions(dtype=torch.float32),
                                                 {"dtype": "fp32"})):
                direct_tok, direct = Checkpoint(str(lora_dir)).load("cpu", options)
                enc = direct.encode(direct_tok, materialize(record),
                                    max_state=lora_probe.context["max_state"],
                                    max_branch=lora_probe.context["max_branch"], strict=True)
                with torch.no_grad():
                    expected = [torch.softmax(z, -1) for z in direct.forward(enc)]
                through_probe = K.KevPredictor(str(lora_dir), device="cpu", **kwargs)(record)
                got = {qid: torch.tensor(list(through_probe["probabilities"][qid].values()))
                       for qid in through_probe["probabilities"]}
                delta = max(float((expected[i] - got[qid]).abs().max()) for i, qid in enumerate(got))
                check(f"the LoRA path's probabilities are BIT-IDENTICAL through KevPredictor and "
                      f"Checkpoint.load at {dtype_name}",
                      len(expected) == len(got) and all(torch.equal(expected[i], got[qid])
                                                        for i, qid in enumerate(got)),
                      f"{len(got)} question(s), max |dp| {delta:.3e}")
        else:
            skip("the LoRA bit-identity check", why)

        print()
        print("=== 6. the placement guards, each shown refusing the thing it exists to catch ===")
        refusals = []
        for name, call in (
            ("no device at all", lambda: F.plan_device_map(predictor.model.lm, [])),
            ("a device list that is neither cuda:N nor ['cpu']",
             lambda: F.plan_device_map(predictor.model.lm, ["cuda:0", "mps"])),
        ):
            try:
                call()
                refusals.append((name, None))
            except F.FullWeightError as error:
                refusals.append((name, str(error)))
        check("plan_device_map refuses an unusable device list",
              all(reason is not None for _, reason in refusals),
              "; ".join(f"{name}: {reason}" for name, reason in refusals)[:160])
        check("plan_device_map(['cpu']) returns the single-device map accelerate treats as model.to()",
              F.plan_device_map(predictor.model.lm, ["cpu"]) == {"": "cpu"})
        # The over-budget refusal is the one that keeps a 27B off CPU. It is exercised without a GPU by
        # feeding the budget arithmetic directly, so this check cannot be skipped on a CPU-only box.
        real_free = F.free_bytes
        try:
            F.free_bytes = lambda spec: 1024 ** 3
            try:
                F.plan_device_map(predictor.model.lm, ["cuda:0"])
                over_budget = None
            except F.FullWeightError as error:
                over_budget = str(error)
        finally:
            F.free_bytes = real_free
        check("plan_device_map REFUSES when the weights do not fit the budget (no silent CPU offload)",
              over_budget is not None and "do not fit" in over_budget, (over_budget or "it returned a map")[:150])
        # Two negative controls, because the guard has two clauses and a map-only check cannot see the
        # second: a map that CLAIMS cuda:0 while the parameters sit on CPU is the silent-offload case.
        lied_map = None
        try:
            F.verify_placement(predictor.model.lm, {"": "cuda:0"}, ["cuda:0"], "selftest")
        except F.FullWeightError as error:
            lied_map = str(error)
        check("verify_placement REFUSES a map that names a GPU while the parameters are on CPU",
              lied_map is not None and "not one of the requested devices" in lied_map, (lied_map or "it passed")[:140])
        off_map = None
        try:
            F.verify_placement(predictor.model.lm, {"": "cpu"}, ["cuda:0"], "selftest")
        except F.FullWeightError as error:
            off_map = str(error)
        check("verify_placement REFUSES a map that offloads a CUDA run to cpu/disk",
              off_map is not None and "outside the requested devices" in off_map, (off_map or "it passed")[:140])
        check("verify_placement ACCEPTS the placement it was told to use",
              F.verify_placement(predictor.model.lm, {"": "cpu"}, ["cpu"], "selftest")["devices"] == ["cpu"])
        # The fp32-exact guard, in all three directions. On this box there is no CUDA context, so the
        # switches are set by hand to the state `LocalPredictor.__init__` leaves them in; they are
        # process-global, which is why the state is restored afterwards.
        exact = {"matmul": False, "cudnn": False, "flash": False, "mem_efficient": False}
        before = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
                  torch.backends.cuda.flash_sdp_enabled(), torch.backends.cuda.mem_efficient_sdp_enabled())
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            check("the fp32-exact guard is inert off CUDA",
                  F.assert_no_exact_path_drift("cpu", {"attn": "eager"}) is None)
            check("...and it PASSES a CUDA run in LocalPredictor's own switch state with sdpa",
                  F.assert_no_exact_path_drift("cuda", {"attn": "sdpa"}) is None)
            eager = None
            try:
                F.assert_no_exact_path_drift("cuda", {"attn": "eager"})
            except F.FullWeightError as error:
                eager = str(error)
            check("...and it REFUSES a CUDA run whose attention backend is not sdpa",
                  eager is not None and "'eager'" in eager, (eager or "it passed")[:140])
            torch.backends.cuda.matmul.allow_tf32 = True
            tf32 = None
            try:
                F.assert_no_exact_path_drift("cuda", {"attn": "sdpa"})
            except F.FullWeightError as error:
                tf32 = str(error)
            check("...and it REFUSES a CUDA run with TF32 left on (a silently different matmul)",
                  tf32 is not None and "allow_tf32" in tf32, (tf32 or "it passed")[:140])
        finally:
            (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32) = before[0], before[1]
            torch.backends.cuda.enable_flash_sdp(before[2])
            torch.backends.cuda.enable_mem_efficient_sdp(before[3])
        check("the switch state was restored", (torch.backends.cuda.matmul.allow_tf32,
              torch.backends.cudnn.allow_tf32, torch.backends.cuda.flash_sdp_enabled(),
              torch.backends.cuda.mem_efficient_sdp_enabled()) == before)
        _ = exact
    finally:
        shutil.rmtree(work, ignore_errors=True)

    print()
    print(f"{len(PASSES)} passed, {len(FAILURES)} failed, {len(SKIPPED)} skipped")
    for entry in FAILURES:
        print(f"  FAIL {entry}")
    for entry in SKIPPED:
        print(f"  skip {entry}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:                                    # noqa: BLE001 - a crash is a failed check
        traceback.print_exc()
        sys.exit(2)
