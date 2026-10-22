#!/usr/bin/env python3
"""Ask a general instruct-tuned LLM the decision models' own questions, zero-shot.

One **ask** here is one record's questions answered in one batched generation, which is the unit the
speed/cost table measures a Kev rung in (one forward for a whole record). The analogue is exact enough
to be useful and different enough to have to be said out loud: a Kev forward reads the state once and
every option branch in the same pass, while a chat model is given the state again for each question, so
the LLM's tokens-per-ask is larger by the number of questions. Both counts are recorded --
``input_tokens_processed`` (what was really prefilled) and ``input_tokens_state_once`` (state once plus
every question's own text, the Kev-comparable count) -- so the difference is a measurement, not a caveat.

    python scripts/llm/run_llm.py --asks reports/llm-asks-smoke.jsonl \\
        --out reports/llm-baseline/smoke-varA --variant a --shard 0 --shards 4

Resumable per record: an ask whose file exists is not re-run, so a preemption or a wall-clock kill costs
only the asks in flight and a re-submission continues instead of starting over. Every ask's record holds
the exact prompt that was sent, so the report's claim about the input is checkable against the output
tree rather than against this file's description of it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from render import PROMPT_VARIANTS, messages, parse_answer, user_message  # noqa: E402


#: Family-set aliases, because Slurm's ``--export`` splits its own value on commas and a comma-separated
#: family list therefore arrives as several broken variables. Observed live: `LLM_FAMILIES=a,b,c` made
#: the pass answer family `a` only, silently, with a manifest that looked plausible. Aliases have no
#: commas, so they survive the export; the multi-character separators are accepted for interactive use.
FAMILY_ALIASES: dict[str, tuple[str, ...]] = {
    "all-choice": ("panel_choice", "panel_miss", "evidence_diluted"),
    "choice-families": ("panel_choice", "panel_miss", "evidence_diluted"),
    "all": ("panel_choice", "panel_miss", "evidence_diluted", "marker_noul"),
}


def family_filter(spec: str) -> set[str]:
    """``--families`` -> the set of families to keep. Empty means every family in the ask set."""
    out: set[str] = set()
    for token in spec.replace("|", ",").replace("+", ",").split(","):
        token = token.strip()
        if not token:
            continue
        if token in FAMILY_ALIASES:
            out.update(FAMILY_ALIASES[token])
        else:
            out.add(token)
    return out


def record_key(ask: dict) -> str:
    return f"{ask['dataset']}/{ask['cell']}/{ask['family']}"


def shard_of(key: str, index: int, count: int) -> int:
    """A stable, order-independent shard assignment: sha1 of the record key, mod ``count``."""
    return int(hashlib.sha1(key.encode("utf-8")).hexdigest(), 16) % count


def safe(name: str) -> str:
    return name.replace("/", "_").replace(":", "-")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--asks", required=True)
    ap.add_argument("--out", required=True, help="the output tree; raw/ holds one file per ask")
    ap.add_argument("--variant", required=True, choices=list(PROMPT_VARIANTS))
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--revision", default="")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--thinking", action="store_true",
                    help="leave the model's reasoning mode on (off by default, in BOTH variants)")
    ap.add_argument("--families", default="", help="comma-separated family filter")
    ap.add_argument("--limit-records", type=int, default=0)
    ap.add_argument("--device-map", default="cuda",
                    help="transformers device_map. 'cuda' (the default) pins the whole checkpoint to one "
                         "card, which is what every arm measured before the multi-size pass did. 'auto' "
                         "spreads it over every visible card, which is the only way a bf16 27B (55.6 GB) "
                         "fits this cluster's A100-40G cards; the sharding is data-parallel over records "
                         "in either case, so one ask is still one forward.")
    args = ap.parse_args()


    out_dir = Path(args.out)
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "ask-log.jsonl"

    asks: dict[str, list[dict]] = defaultdict(list)
    wanted = family_filter(args.families)
    with open(args.asks, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            ask = json.loads(line)
            if wanted and ask["family"] not in wanted:
                continue
            asks[record_key(ask)].append(ask)
    keys = sorted(asks)
    mine = [k for k in keys if shard_of(k, args.shard, args.shards) == args.shard]
    if args.limit_records:
        mine = mine[:args.limit_records]

    print(f"host={platform.node()} date={time.strftime('%Y-%m-%dT%H:%M:%S%z')} "
          f"job={os.environ.get('SLURM_JOB_ID', 'none')}")
    print(f"asks file   : {args.asks}")
    print(f"families    : {sorted(wanted) if wanted else 'every family in the ask set'}")
    print(f"records     : {len(keys)} total, {len(mine)} on shard {args.shard}/{args.shards}")
    print(f"questions   : {sum(len(asks[k]) for k in mine)}")
    print(f"variant     : {args.variant}   thinking={args.thinking}   "
          f"max_new_tokens={args.max_new_tokens}")

    todo = [k for k in mine if not (raw_dir / f"{safe(k)}.json").exists()]
    print(f"already done: {len(mine) - len(todo)}")
    if not todo:
        print("LLM_RUN_DONE rc=0 nothing to do (every ask on this shard already has a file)")
        return 0

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision or None)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision or None, dtype=torch.bfloat16, device_map=args.device_map)
    model.eval()
    load_s = time.time() - t0
    print(f"model loaded in {load_s:.1f}s  "
          f"params={sum(p.numel() for p in model.parameters()) / 1e9:.2f}B  "
          f"dtype={next(model.parameters()).dtype}  device_map={args.device_map}")
    if args.device_map != "cuda":
        # A sharded load must be shown to have stayed on the cards: this backbone is hybrid
        # (Gated DeltaNet), and a silent CPU offload is not a fallback here -- it is a stall that
        # reads as a slow model. Name where the first and last blocks live, and refuse if any is cpu.
        placed = sorted({str(p.device) for p in model.parameters()})
        print(f"parameter devices: {placed}")
        if any(d.startswith("cpu") for d in placed):
            raise SystemExit("REFUSING: a parameter landed on CPU -- the hybrid backbone cannot run "
                             "there, so this is a misplaced device_map, not a slow run")

    # The ask clock starts *after* the weights are resident: the speed/cost pass measures what answering
    # costs, not what loading costs, and the two are separated here for the same reason they are there.
    ask_started_at = time.time()
    asked = answered = 0
    for n, key in enumerate(todo, 1):
        group = sorted(asks[key], key=lambda a: a["question"])
        prompts, plain = [], []
        for ask in group:
            msgs = messages(ask["state"], ask, args.variant)
            text = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=bool(args.thinking))
            prompts.append(text)
            plain.append(user_message(ask["state"], ask, args.variant))
        enc = tokenizer(prompts, return_tensors="pt", padding=True,
                        add_special_tokens=False).to("cuda")
        t1 = time.time()
        with torch.inference_mode():
            gen = model.generate(**enc, do_sample=False, max_new_tokens=args.max_new_tokens,
                                 pad_token_id=tokenizer.pad_token_id)
        latency_ms = (time.time() - t1) * 1000.0
        new = gen[:, enc["input_ids"].shape[1]:]

        state_tokens = len(tokenizer(plain[0].split("\n\n", 1)[0], add_special_tokens=False)["input_ids"])
        rows = []
        for ask, prompt, user_text, row in zip(group, prompts, plain, new):
            completion = tokenizer.decode(row, skip_special_tokens=True)
            keys_list = [str(k) for k in ask["criteria"]]
            parsed, rule = parse_answer(completion, keys_list)
            n_in = int(enc["attention_mask"][len(rows)].sum())
            n_out = int((row != tokenizer.pad_token_id).sum())
            rest = user_text.split("\n\n", 1)[1] if "\n\n" in user_text else user_text
            rows.append({
                "key": ask["key"], "dataset": ask["dataset"], "cell": ask["cell"],
                "family": ask["family"], "question": ask["question"], "type": ask["type"],
                "option_keys": keys_list,
                "prompt": prompt, "user_message": user_text,
                "completion": completion, "parsed": parsed, "parse_rule": rule,
                "input_tokens": n_in, "output_tokens": n_out,
                "state_tokens": state_tokens,
                "question_tokens": len(tokenizer(rest, add_special_tokens=False)["input_ids"]),
                "source_file": ask["source_file"], "source_sha256": ask["source_sha256"],
                "request_sha256": ask.get("request_sha256"),
            })
            asked += 1
            answered += int(parsed is not None)
        payload = {
            "ask": key, "dataset": group[0]["dataset"], "cell": group[0]["cell"],
            "family": group[0]["family"], "variant": args.variant, "thinking": bool(args.thinking),
            "model": args.model, "revision": args.revision,
            "n_questions": len(rows),
            "latency_ms": latency_ms,
            "input_tokens_processed": sum(r["input_tokens"] for r in rows),
            "input_tokens_state_once": sum(r["question_tokens"] for r in rows) + rows[0]["state_tokens"],
            "output_tokens": sum(r["output_tokens"] for r in rows),
            "questions": rows,
        }
        (raw_dir / f"{safe(key)}.json").write_text(
            json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({k: v for k, v in payload.items() if k != "questions"}) + "\n")
        if n % 25 == 0 or n == len(todo):
            rate = n / max(time.time() - ask_started_at, 1e-9)
            print(f"  {n}/{len(todo)} asks  parsed={answered}/{asked}  "
                  f"{rate:.2f} asks/s  "
                  f"eta {(len(todo) - n) / max(rate, 1e-9) / 60:.1f} min", flush=True)

    finished_at = time.time()
    # Accumulate rather than overwrite: a resumed run's wall clock is the sum of its invocations' ask
    # clocks, and the pass's own wall clock is the largest of the shards'. Overwriting would let a
    # preempted-and-resumed shard report only its last leg, which reads as a faster engine than it was.
    run_path = out_dir / f"run-shard{args.shard}.json"
    history = json.loads(run_path.read_text(encoding="utf-8")) if run_path.exists() else []
    if isinstance(history, dict):
        history = [history]
    history.append({
        "shard": args.shard, "shards": args.shards, "variant": args.variant,
        "thinking": bool(args.thinking), "model": args.model, "revision": args.revision,
        "max_new_tokens": args.max_new_tokens,
        "asks": asked, "parsed": answered, "unparsed": asked - answered,
        "load_s": load_s, "ask_wall_s": finished_at - ask_started_at,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(t0)),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(finished_at)),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "node": platform.node(),
    })
    run_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")

    print(f"LLM_RUN_DONE rc=0 shard={args.shard}/{args.shards} asks={asked} parsed={answered} "
          f"unparsed={asked - answered} ask_wall_s={finished_at - ask_started_at:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
