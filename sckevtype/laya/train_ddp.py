"""The released laya fine-tuning recipe, vendored so the cluster can run it from this repo.

    python -m sckevtype.laya.train_ddp <model_dir> <output_dir> <items.pt> \\
        --max-len 8192 --head-max-len 2048

``<model_dir>`` is the released checkpoint (``convaiinnovations/laya-multilingual``), ``<items.pt>``
is what ``sckevtype.laya.render build`` wrote, and everything it saves lands in ``<output_dir>``.

**This is not our code and is not written as our code.** It is ``train_ddp.py`` from the model
card's Kaggle notebook, kept line for line so the pilot is the published recipe and not a
reinterpretation of it — the reward weights (``w_sph=0.75, w_rps=1.0``), the two learning rates,
the cosine schedule, the GRPO group sampling, and the LBFGS temperature fit are all as released.
The deltas are listed here in full, and are the only differences:

1. **The three paths are arguments**, not ``sys.argv[1]``, ``sys.argv[2]`` and a hard-coded
   ``/kaggle/working/train_items.pt``.
2. **``max_len`` and ``head_max_len`` are arguments.** The released script pins 1024/256, which is
   the shipped serving width; this pilot trains at 8192/2048 because the fit measurements say the
   material does not fit at 1024 (``reports/laya-exploration.md`` §4). The width is written into
   the saved config, so scoring reads the width the model was trained at.
3. **DDP is optional.** The released script calls ``dist.init_process_group("nccl")`` unconditionally
   and only runs under a launcher. Here, with no ``WORLD_SIZE`` in the environment, it runs as one
   process on one GPU with no process group — which is what the cluster allocates.
4. **A rolling checkpoint every ``--save-every`` optimizer updates**, on top of the released
   per-epoch one, because the A100 partition is shared and jobs get preempted.
5. **The micro-batch split moves, the effective batch does not.** Released is ``8 x 4`` on two T4s;
   here it is ``2 x 16`` on one A100: the same items per update against a quarter of the activation.

Nothing else is changed, and any further change should be listed here before it is made.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
from laya.common import build_model, proper_reward
from safetensors.torch import load_file, save_file
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer

#: The released script holds this many items out of training so the temperature fit has data the
#: optimiser has not already made certain about. The seed is the released one and rank-independent.
CALIB_MAX = 400
CALIB_SEED = 20260922


def collate_train_batch(items, pad_id):
    n, L = len(items), max(len(it["ids"]) for it in items)
    kmax = max(len(it["markers"]) for it in items)
    ids = torch.full((n, L), pad_id, dtype=torch.long)
    att = torch.zeros((n, L), dtype=torch.long)
    mpos = torch.zeros((n, kmax), dtype=torch.long)
    mmask = torch.zeros((n, kmax), dtype=torch.bool)
    target = torch.zeros((n, kmax), dtype=torch.float32)
    for i, it in enumerate(items):
        ids[i, : len(it["ids"])] = torch.tensor(it["ids"])
        att[i, : len(it["ids"])] = 1
        k = len(it["markers"])
        mpos[i, :k] = torch.tensor(it["markers"])
        mmask[i, :k] = True
        target[i, : len(it["target"])] = torch.tensor(it["target"], dtype=torch.float32)
    return {
        "input_ids": ids,
        "attention_mask": att,
        "marker_pos": mpos,
        "marker_mask": mmask,
        "target": target,
        "qtype": torch.tensor([it["qtype"] for it in items]),
        "label": torch.tensor([it["label"] for it in items]),
    }


def fit_one_temp(sel):
    if len(sel) < 10:
        return 1.0
    kmax = max(len(z) for z, _ in sel)
    Z = torch.full((len(sel), kmax), -1e4)
    T = torch.zeros((len(sel), kmax))
    for i, (z, t) in enumerate(sel):
        Z[i, :len(z)] = torch.tensor(z)
        T[i, :len(t)] = torch.tensor(t, dtype=torch.float32)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = -(T * torch.log_softmax(Z / log_t.exp(), -1)).sum(-1).mean()
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.clamp(log_t.exp(), 0.1, 10.0).item())


def save_checkpoint(model, tok, out_dir: Path, meta: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file({k: v.half().contiguous().cpu() for k, v in model.state_dict().items()},
              str(out_dir / "model.safetensors"))
    model.encoder.config.save_pretrained(str(out_dir / "encoder"))
    tok.save_pretrained(str(out_dir / "tokenizer"))
    (out_dir / "checkpoint_meta.json").write_text(json.dumps(meta, indent=2) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sckevtype.laya.train_ddp", description=__doc__.splitlines()[0])
    parser.add_argument("model_dir", help="the released laya checkpoint to fine-tune from")
    parser.add_argument("output_dir", help="where the fine-tuned checkpoint is written")
    parser.add_argument("items", help="train_items.pt from sckevtype.laya.render build")
    parser.add_argument("--max-len", type=int, default=8192)
    parser.add_argument("--head-max-len", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=4)
    # The released 8 x 4 x (2 GPUs) is an effective 64 items per update. At 8192 tokens a micro-batch
    # of 8 is too much activation for a 40 GB card, so the split moves: same effective batch at
    # world_size 1, a quarter of the peak. `--grad-accum` is what preserves the group statistics.
    parser.add_argument("--micro-batch", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--lr-encoder", type=float, default=2.5e-5)
    parser.add_argument("--lr-head", type=float, default=1.0e-4)
    parser.add_argument("--sigma-start", type=float, default=0.4)
    parser.add_argument("--sigma-end", type=float, default=0.1)
    parser.add_argument("--save-every", type=int, default=500, help="optimizer updates between rolling checkpoints")
    parser.add_argument("--limit", type=int, default=0, help="stop after N items (0 = all); for a smoke run")
    args = parser.parse_args(argv)

    distributed = "WORLD_SIZE" in os.environ
    if distributed:
        dist.init_process_group("nccl")
        rank, world_size = dist.get_rank(), dist.get_world_size()
    else:
        rank, world_size = 0, 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    model_dir, output_dir, items_path = Path(args.model_dir), Path(args.output_dir), args.items
    for path in (model_dir / "rl_agent_config.json", model_dir / "model.safetensors", Path(items_path)):
        if not path.exists():
            raise SystemExit(f"missing input: {path}")

    cfg = json.loads((model_dir / "rl_agent_config.json").read_text())
    cfg["gradient_checkpointing"] = True
    cfg["max_len"] = args.max_len
    cfg["head_max_len"] = args.head_max_len

    tok = AutoTokenizer.from_pretrained(str(model_dir / "tokenizer"))
    model = build_model(cfg, encoder_dir=str(model_dir / "encoder"))
    model.load_state_dict(load_file(str(model_dir / "model.safetensors")), strict=True)
    model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.head_checkpointing = True
    model.to(device)
    model.train()
    ddp_model = DDP(model, device_ids=[local_rank], find_unused_parameters=True) if distributed else model

    all_items = torch.load(items_path, weights_only=False)
    if not all_items:
        raise SystemExit(f"{path} holds no items; run sckevtype.laya.render build first")
    if args.limit:
        all_items = all_items[: args.limit]

    # Hold the calibration slice out of training before sharding. Temperatures fitted on items the
    # run has already trained on measure the fit rather than the calibration. The seed is fixed and
    # rank-independent, so every rank withholds exactly the same items.
    order = list(range(len(all_items)))
    random.Random(CALIB_SEED).shuffle(order)
    n_calib = min(CALIB_MAX, len(all_items) // 10)
    calib_items = [all_items[i] for i in sorted(order[:n_calib])]
    train_items = [all_items[i] for i in sorted(order[n_calib:])]
    my_items = train_items[rank::world_size]

    if rank == 0:
        print(f"training {len(train_items)} items ({len(calib_items)} held out for calibration) | "
              f"{len(my_items)} per rank | {args.epochs} epochs | max_len={args.max_len} "
              f"head_max_len={args.head_max_len} | world_size={world_size}", flush=True)

    enc_params = [p for n, p in ddp_model.named_parameters() if "encoder." in n]
    head_params = [p for n, p in ddp_model.named_parameters() if "encoder." not in n]
    optimizer = torch.optim.AdamW([{"params": enc_params, "lr": args.lr_encoder},
                                   {"params": head_params, "lr": args.lr_head}], weight_decay=0.01)
    updates_per_epoch = len(my_items) // (args.micro_batch * args.grad_accum)
    total_updates = updates_per_epoch * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_updates), eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    t0 = time.time()
    updates = 0
    for epoch in range(args.epochs):
        random.seed(42 + epoch + rank)
        random.shuffle(my_items)
        epoch_loss, n_batches = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        accum_step = 0
        sigma = args.sigma_start + (args.sigma_end - args.sigma_start) * (epoch / max(1, args.epochs - 1))

        for b_idx in range(0, len(my_items), args.micro_batch):
            chunk = my_items[b_idx:b_idx + args.micro_batch]
            if not chunk:
                continue
            batch = collate_train_batch(chunk, tok.pad_token_id)
            with torch.autocast("cuda", dtype=torch.float16):
                logits, act = ddp_model(batch["input_ids"].to(device), batch["attention_mask"].to(device),
                                        batch["marker_pos"].to(device), batch["marker_mask"].to(device),
                                        batch["qtype"].to(device))
            logits = logits.float()
            mask = batch["marker_mask"].to(device)
            k = mask.sum(-1, keepdim=True).float()
            target = batch["target"].to(device)

            # 1. Sample G noisy logit distributions with zero-mean projection
            eps = torch.randn((args.group_size,) + logits.shape, device=device) * sigma * mask
            eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
            z = logits.detach().unsqueeze(0) + eps
            q = torch.softmax(z.masked_fill(~mask, -1e4), -1)

            # 2. Evaluate proper scoring reward (w_sph=0.75 for soft target matching)
            with torch.no_grad():
                r = proper_reward(q, target.unsqueeze(0), batch["qtype"].to(device), mask, w_sph=0.75, w_rps=1.0)
                adv = r - r.mean(0, keepdim=True)
                adv = adv / (adv.std() + 1e-6)

            # 3. Policy gradient loss + full 1.0 soft cross-entropy guidance
            logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma ** 2)
            loss_rl = -(adv * logp).mean()
            loss_ce = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()
            loss = (loss_rl + 1.0 * loss_ce) / args.grad_accum + 0.0 * act.sum()

            scaler.scale(loss).backward()
            accum_step += 1
            if accum_step % args.grad_accum == 0 or (b_idx + args.micro_batch) >= len(my_items):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                updates += 1
                if rank == 0 and args.save_every and updates % args.save_every == 0:
                    save_checkpoint(model, tok, output_dir / "checkpoint_latest",
                                    {"updates": updates, "total_updates": total_updates, "epoch": epoch,
                                     "partial": True, "elapsed_s": time.time() - t0})
                    print(f"  saved rolling checkpoint at update {updates}/{total_updates} "
                          f"({(time.time() - t0) / 60:.1f} min)", flush=True)
            epoch_loss += loss.item() * args.grad_accum
            n_batches += 1
            if rank == 0 and (n_batches % 50) == 0:
                print(f"  Epoch {epoch + 1}/{args.epochs} | Step {n_batches} | "
                      f"Loss: {loss.item() * args.grad_accum:.4f} | Reward: {r.mean().item():.3f} | "
                      f"LR: {scheduler.get_last_lr()[0]:.2e}", flush=True)

        if rank == 0:
            print(f"=== Epoch {epoch + 1}/{args.epochs} completed in {time.time() - t0:.1f}s | "
                  f"avg loss {epoch_loss / max(1, n_batches):.4f} ===", flush=True)
        if distributed:
            dist.barrier()
        if rank == 0:
            save_checkpoint(model, tok, output_dir / "checkpoint_latest",
                            {"epoch": epoch + 1, "total_epochs": args.epochs,
                             "avg_loss": epoch_loss / max(1, n_batches), "updates": updates,
                             "elapsed_s": time.time() - t0})

    if distributed:
        dist.barrier()

    # Post-training temperature calibration, on rank 0 only. Note for the pilot: the gate's ECE does
    # not read this. It is fitted on the suite's own `calibration` partition by
    # `sckevtype.laya.score evaluate`, which is step 11's instrument. This one is the recipe's, kept
    # because removing it would make this not the released recipe.
    if rank == 0:
        print("\nfitting the recipe's post-training temperatures on the held-out slice...", flush=True)
        del optimizer, scaler, scheduler
        torch.cuda.empty_cache()
        model.eval()
        calib_preds = []
        with torch.no_grad():
            for c_idx in range(0, len(calib_items), 16):
                c_chunk = calib_items[c_idx:c_idx + 16]
                cb = collate_train_batch(c_chunk, tok.pad_token_id)
                with torch.autocast("cuda", dtype=torch.float16):
                    l_sub, _ = model(cb["input_ids"].to(device), cb["attention_mask"].to(device),
                                     cb["marker_pos"].to(device), cb["marker_mask"].to(device),
                                     cb["qtype"].to(device))
                l_np = l_sub.float().cpu().numpy()
                for r_i, it in enumerate(c_chunk):
                    calib_preds.append((it["qtype"], l_np[r_i, : len(it["markers"])], it["target"]))

        fitted_temps = [1.2, 1.2, 1.2]
        try:
            for qt in range(3):
                sel = [(z, t) for q_type, z, t in calib_preds if q_type == qt]
                if sel:
                    fitted_temps[qt] = fit_one_temp(sel)
            print("fitted temperatures (choice, score, noul):", [round(t, 3) for t in fitted_temps], flush=True)
        except Exception as error:  # noqa: BLE001 -- the released script's own fallback, kept verbatim
            print("temperature fitting fallback:", error, flush=True)

        cfg["fine_tuned"] = True
        cfg["model_name"] = "laya-typed-decisions"
        cfg["temperature"] = fitted_temps
        cfg.pop("temperature_by_options", None)
        cfg["training"] = {"updates": updates, "epochs_completed": args.epochs, "world_size": world_size,
                           "items": len(my_items), "base": str(model_dir), "items_sha256": _digest(items_path),
                           "elapsed_s": time.time() - t0}
        save_file({k: v.half().contiguous().cpu() for k, v in model.state_dict().items()},
                  str(output_dir / "model.safetensors"))
        model.encoder.config.save_pretrained(str(output_dir / "encoder"))
        tok.save_pretrained(str(output_dir / "tokenizer"))
        (output_dir / "rl_agent_config.json").write_text(json.dumps(cfg, indent=2) + "\n")
        print(f"saved fine-tuned checkpoint to {output_dir}", flush=True)

    if distributed:
        dist.destroy_process_group()
    return 0


def _digest(path: str) -> str:
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


if __name__ == "__main__":
    sys.exit(main())
