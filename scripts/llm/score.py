#!/usr/bin/env python3
"""Score the LLM baseline exactly the way the decision models are scored.

The scoring rule is not re-implemented here; it is *read from the same row dumps every other engine's
number was computed from*. A reference row carries ``keys`` (the option list in the order the raw request
held it), ``label`` (the index of the correct option) and ``abstain`` (the index of the abstain option,
or ``null`` when the ballot has none). So:

* **hit** = the option the LLM named is at index ``label``.
* **abstain pick** = the option the LLM named is at index ``abstain`` (a pick of the abstain option is
  counted as an abstain, never as an error against ``label`` in the abstain-rate column, and never as a
  hit in the accuracy column -- ``panel_miss`` is the family where those two coincide).
* **parse failure** = the model's reply could not be mapped onto any option. The row is **kept** and
  counted as a miss, and its share is reported per family; a model that refuses to answer in-format is a
  result.

The accuracy is also reported restricted to the *strict* parse rules (``exact``/``leader``/``first_line``
in :mod:`scripts.llm.render`), so the size of the substring rescue is visible rather than assumed.

    python scripts/llm/score.py --runs reports/llm-baseline/smoke-varA \\
        --reference reports/jev-smoke/rows.jsonl \\
        --out-rows reports/llm-baseline-rows-smoke-varA.jsonl \\
        --out-summary reports/llm-baseline-smoke-varA.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

STRICT_RULES = ("exact", "leader", "first_line")


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def pct(numerator: int, denominator: int) -> float | None:
    return (numerator / denominator) if denominator else None


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))
    return ordered[idx]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", required=True, help="an ask tree written by run_llm.py")
    ap.add_argument("--reference", required=True, help="the row dump the decision models were scored on")
    ap.add_argument("--out-rows", required=True)
    ap.add_argument("--out-summary", required=True)
    ap.add_argument("--label", default="", help="a name for this run, recorded in the summary")
    ap.add_argument("--subset-ids", default="",
                    help="optional JSONL of {dataset,cell,family[,variant]} to restrict the summary to")
    ap.add_argument("--require-variant", default="",
                    help="with --subset-ids: keep only ids whose variant equals this")
    args = ap.parse_args()

    runs_dir = Path(args.runs)
    files = sorted((runs_dir / "raw").glob("*.json"))
    if not files:
        raise SystemExit(f"{runs_dir}/raw: no ask files -- refusing to score an empty run")

    reference: dict[tuple, dict] = {}
    for row in load_jsonl(Path(args.reference)):
        reference[(row["dataset"], row["cell"], row["family"], row["question"])] = row

    subset = None
    if args.subset_ids:
        ids = load_jsonl(Path(args.subset_ids))
        if args.require_variant:
            ids = [i for i in ids if i.get("variant") == args.require_variant]
        subset = {(i["dataset"], i["cell"], i["family"]) for i in ids}
        print(f"subset ids : {len(ids)} rows over {len(subset)} (dataset, cell, family) triples")

    scored: list[dict] = []
    asks: list[dict] = []
    unmatched: list[str] = []
    model_ids: Counter = Counter()
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        model_ids[f"{payload['model']}@{payload['revision']}"] += 1
        asks.append({k: v for k, v in payload.items() if k != "questions"})
        for row in payload["questions"]:
            key = (row["dataset"], row["cell"], row["family"], row["question"])
            ref = reference.get(key)
            if ref is None:
                unmatched.append(row["key"])
                continue
            keys = [str(k) for k in ref["keys"]]
            label = ref["label"]
            abstain = ref.get("abstain")
            idx = keys.index(row["parsed"]) if row["parsed"] in keys else None
            # The LLM's own option list must be the reference's. For a `choice` question that includes
            # the ORDER: option order is part of the corpus bytes and this class of model is
            # order-sensitive, so a re-ordered ballot is a different question. A `noul` question's
            # reference keys are the API's fixed ["false", "true"] while the raw request stores its
            # criteria true-first, so there the check is on the SET -- the model's answer is mapped by
            # name, and the order it was shown in cannot change which name it named.
            run_keys = [str(k) for k in row["option_keys"]]
            same = run_keys == keys if row["type"] == "choice" else set(run_keys) == set(keys)
            if not same:
                raise SystemExit(f"{row['key']}: the run's option list is not the reference's "
                                 f"({run_keys} vs {keys})")
            if subset is not None and key[:3] not in subset:
                continue
            scored.append({
                "key": row["key"], "dataset": row["dataset"], "cell": row["cell"],
                "family": row["family"], "question": row["question"], "type": row["type"],
                "variant": ref.get("variant"),
                "n_options": len(keys), "keys": keys, "label": label, "abstain": abstain,
                "abstain_key": ref.get("abstain_key"),
                "parsed": row["parsed"], "parsed_index": idx, "parse_rule": row["parse_rule"],
                "completion": row["completion"],
                "hit": int(idx is not None and idx == label),
                "abstain_pick": int(idx is not None and abstain is not None and idx == abstain),
                "parsed_ok": int(idx is not None),
                "strict_rule": int(row["parse_rule"] in STRICT_RULES),
                "input_tokens": row["input_tokens"], "output_tokens": row["output_tokens"],
                "variant_of_prompt": payload["variant"], "thinking": payload["thinking"],
                "model": payload["model"], "revision": payload["revision"],
            })

    if unmatched:
        print(f"WARNING: {len(unmatched)} scored question(s) had no reference row, e.g. {unmatched[:3]}")

    def block(rows: list[dict]) -> dict:
        n = len(rows)
        return {
            "n": n,
            "accuracy": pct(sum(r["hit"] for r in rows), n),
            "accuracy_strict_rules_only": pct(
                sum(r["hit"] for r in rows if r["strict_rule"]), sum(r["strict_rule"] for r in rows)),
            "n_strict_rules": sum(r["strict_rule"] for r in rows),
            "abstain_pick_rate": pct(sum(r["abstain_pick"] for r in rows),
                                     sum(1 for r in rows if r["abstain"] is not None)),
            "abstain_rows": sum(1 for r in rows if r["abstain"] is not None),
            "parse_failures": sum(1 for r in rows if not r["parsed_ok"]),
            "parse_failure_rate": pct(sum(1 for r in rows if not r["parsed_ok"]), n),
        }

    per_family = {f: block([r for r in scored if r["family"] == f])
                  for f in sorted({r["family"] for r in scored})}
    per_dataset = {d: block([r for r in scored if r["dataset"] == d])
                   for d in sorted({r["dataset"] for r in scored})}
    choice = [r for r in scored if r["type"] == "choice"]

    latencies = [a["latency_ms"] for a in asks]
    summary = {
        "what_this_is": "a general instruct-tuned LLM, zero-shot, scored by the decision models' own "
                        "rule on the decision models' own questions",
        "label": args.label,
        "runs_dir": str(runs_dir),
        "reference": args.reference,
        "subset_ids": args.subset_ids or None,
        "require_variant": args.require_variant or None,
        "model": next(iter(model_ids)) if len(model_ids) == 1 else None,
        "models_seen": dict(model_ids),
        "asks": len(asks),
        "questions": len(scored),
        "unmatched_questions": len(unmatched),
        "overall": block(scored),
        "choice_rows_only": block(choice),
        "per_family": per_family,
        "per_dataset": per_dataset,
        "parse_rules": dict(Counter(r["parse_rule"] for r in scored)),
        "serving": {
            "variant": asks[0]["variant"], "thinking": asks[0]["thinking"],
            "max_new_tokens": None,
            "latency_ms_per_ask": {
                "n": len(latencies),
                "median": statistics.median(latencies) if latencies else None,
                "p90": quantile(latencies, 0.90), "p95": quantile(latencies, 0.95),
                "p99": quantile(latencies, 0.99), "max": max(latencies) if latencies else None,
                "count_over_1s": sum(1 for x in latencies if x > 1000),
            },
            "input_tokens_per_ask_processed": statistics.median(
                [a["input_tokens_processed"] for a in asks]) if asks else None,
            "input_tokens_per_ask_state_once": statistics.median(
                [a["input_tokens_state_once"] for a in asks]) if asks else None,
            "output_tokens_per_ask": statistics.median(
                [a["output_tokens"] for a in asks]) if asks else None,
            "questions_per_ask": statistics.median(
                [a["n_questions"] for a in asks]) if asks else None,
        },
    }

    Path(args.out_rows).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.out_rows).open("w", encoding="utf-8") as fh:
        for row in scored:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    Path(args.out_summary).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"asks {len(asks)}  questions scored {len(scored)}")
    print(f"overall accuracy {summary['overall']['accuracy']}  "
          f"parse failures {summary['overall']['parse_failures']}")
    for family, b in per_family.items():
        print(f"  {family:18s} n={b['n']:5d} acc={b['accuracy']} abstain={b['abstain_pick_rate']} "
              f"parsefail={b['parse_failures']}")
    print(f"parse rules: {summary['parse_rules']}")
    print(f"wrote {args.out_rows} and {args.out_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
