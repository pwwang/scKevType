"""Hardware capacity probe: can this GPU actually train the scKevType decision model?

Measures LoRA forward+backward throughput and peak VRAM on the *actual* Qwen3.5 base
models, at the sequence length the paper's records use (state 384 + question ~896).
This is a capacity measurement, not a correctness test: random token batches, LM loss.

Usage:  uv run python scripts/bench_train.py [--models 0.8b,4b] [--batch 4] [--seq 1280]
"""

from __future__ import annotations

import argparse
import time

import torch
from transformers import AutoModelForCausalLM

MODELS = {
    # ids + revisions pinned by the foundation method's released recipes
    "0.8b": ("Qwen/Qwen3.5-0.8B-Base", "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"),
    "4b": ("Qwen/Qwen3.5-4B-Base", "1001bb4d826a52d1f399e183466143f4da7b741b"),
    # attention-only previous generation: no FLA/DeltaNet kernel needed
    "4b-qwen3": ("Qwen/Qwen3-4B-Base", None),
}


def load(model_id: str, revision: str | None):
    kwargs = {"device_map": "cuda"}
    for dtype_key in ("dtype", "torch_dtype"):  # transformers renamed this
        try:
            return AutoModelForCausalLM.from_pretrained(
                model_id, revision=revision, **{dtype_key: torch.bfloat16}, **kwargs
            )
        except TypeError:
            continue
    raise RuntimeError("could not load model with either dtype kwarg")


def probe(key: str, batch: int, seq: int, steps: int, grad_ckpt: bool) -> dict:
    from peft import LoraConfig, get_peft_model

    model_id, revision = MODELS[key]
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    model = load(model_id, revision)
    load_s = time.time() - t0

    cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, cfg)
    if grad_ckpt:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model.train()

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    vocab = model.config.vocab_size
    ids = torch.randint(0, vocab, (batch, seq), device="cuda")
    attn = torch.ones_like(ids)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-5)

    # warmup: allocator, kernel autotune, and the clock ramp (the A5000 idles at
    # ~495 MHz / P5 and needs a couple of seconds of load before boosting — a
    # 2-step warmup produced readings 20x slower than the true rate)
    for i in range(steps + 6):
        if i == 6:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
        out = model(input_ids=ids, attention_mask=attn, labels=ids)
        out.loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    dt = time.time() - t0

    tok = batch * seq * steps
    return {
        "model": key,
        "hub_id": model_id,
        "batch": batch,
        "seq": seq,
        "grad_ckpt": grad_ckpt,
        "trainable_params": n_train,
        "tokens_per_s": round(tok / dt, 1),
        "step_s": round(dt / steps, 3),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "load_s": round(load_s, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="0.8b,4b-qwen3,4b")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq", type=int, default=1280)
    ap.add_argument("--steps", type=int, default=8)
    args = ap.parse_args()

    print(f"GPU: {torch.cuda.get_device_name(0)}  torch {torch.__version__}", flush=True)
    try:
        import triton

        print(f"triton {triton.__version__}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"triton unavailable: {e}", flush=True)

    results = []
    for key in [m.strip() for m in args.models.split(",") if m.strip()]:
        for ckpt in (True, False):
            try:
                r = probe(key, args.batch, args.seq, args.steps, ckpt)
            except RuntimeError as e:  # OOM or kernel failure
                print(f"[{key} ckpt={ckpt}] FAILED: {type(e).__name__}: {str(e)[:200]}", flush=True)
                torch.cuda.empty_cache()
                results.append({"model": key, "grad_ckpt": ckpt, "error": str(e)[:200]})
                break  # no point trying the hungrier variant
            results.append(r)
            print(f"[{key} ckpt={ckpt}] {r['tokens_per_s']} tok/s  "
                  f"{r['step_s']}s/step  peak {r['peak_vram_gb']} GB  "
                  f"lora_params {r['trainable_params']:,}  load {r['load_s']}s", flush=True)
            del r
        torch.cuda.empty_cache()

    print("RESULTS_JSON " + __import__("json").dumps(results), flush=True)


if __name__ == "__main__":
    main()
