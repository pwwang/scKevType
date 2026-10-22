#!/usr/bin/env python3
"""Assemble `reports/llm-baseline-summary.json`: the LLM row, and the same-rows comparison it needs.

Three blocks, and the reason each exists:

``rows``
    The headline numbers, read from the scorer's own summaries (never re-typed). Every value carries the
    path it was read from, so a reader can re-derive it instead of trusting this file.

``same_rows_as_the_census_matched_table``
    The census-matched table scores 1,819 `panel_choice` rows, but only **909** of them are records the
    census actually asked: the other 910 are the *other variant* of the same cell+family, which the
    capped suite holds and no raw ask dump does. This block reduces the census-matched engine row dumps
    to the 909 whose prompt is on disk and scores them with the decision models' own rule, so the LLM
    number has a like-for-like partner rather than a comparison against a population it cannot be run on.
    Control: Jev's accuracy over all 1,819 of its census-matched rows is in the census-matched table, and
    its accuracy over the 909 here is printed beside it -- if the subset were a biased slice these two
    would disagree.

``variant_comparison``
    Both documented prompts on both populations, so the headline can be the better of the two and the
    reader can see that the choice was made for the model rather than against it.

    python scripts/llm/assemble.py --summaries <dir of *.summary.json> --out reports/llm-baseline-summary.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def argmax(values: list[float]) -> int:
    return max(range(len(values)), key=lambda i: values[i])


def score_rows(rows: list[dict], prediction) -> dict:
    """The decision models' own rule, applied to a row dump: argmax == label is a hit."""
    out: dict = {}
    for family in sorted({r["family"] for r in rows}):
        group = [r for r in rows if r["family"] == family]
        hits = abstains = abstain_rows = floor = 0
        for row in group:
            pick = prediction(row)
            hits += int(pick == row["label"])
            if row.get("abstain") is not None:
                abstain_rows += 1
                abstains += int(pick == row["abstain"])
                floor += int(row["abstain"] == row["label"])
        out[family] = {
            "n": len(group),
            "accuracy": hits / len(group) if group else None,
            "abstain_pick_rate": (abstains / abstain_rows) if abstain_rows else None,
            "abstain_rows": abstain_rows,
            "always_abstain_floor": (floor / len(group)) if group else None,
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summaries", required=True,
                    help="a glob for the scorer's summaries, e.g. 'reports/llm-baseline-*.summary.json'")
    ap.add_argument("--census-ids", default="reports/census-suite-ids.jsonl")
    ap.add_argument("--jev-rows", default="reports/jev-full/rows.jsonl")
    ap.add_argument("--frozen-rows", default="reports/rows-full-frozen.jsonl")
    ap.add_argument("--rungs", default="reports/kev-rungs",
                    help="the served Kev rungs' census row dumps, <rung>-rows.jsonl")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    summaries = {}
    for path in sorted(ROOT.glob(args.summaries)):
        summaries[path.name] = json.loads(path.read_text(encoding="utf-8"))
    if not summaries:
        raise SystemExit(f"{args.summaries}: no *.summary.json matched -- nothing to assemble")

    ids = load_jsonl(ROOT / args.census_ids)
    permuted = {(i["dataset"], i["cell"], i["family"]) for i in ids if i.get("variant") == "permuted"}

    jev_all = load_jsonl(ROOT / args.jev_rows)
    jev = [r for r in jev_all if (r["dataset"], r["cell"], r["family"]) in permuted]
    frozen_ids = {i["id"]: i for i in ids}
    frozen_all = load_jsonl(ROOT / args.frozen_rows)
    frozen = [r for r in frozen_all
              if r["id"] in frozen_ids and frozen_ids[r["id"]].get("variant") == "permuted"]

    # The served Kev rungs, from their own census row dumps. Their `id` is corpus-style
    # (`test/<dataset>/<cell>/<family>`) and they carry a real `p` vector, so the same rule applies
    # unchanged -- which is what makes this a comparison rather than a re-scoring of one arm.
    rungs = {}
    for path in sorted((ROOT / args.rungs).glob("*-rows.jsonl")):
        name = path.name[: -len("-rows.jsonl")]
        rows = [r for r in load_jsonl(path)
                if (r["dataset"], r["cell"], r["family"]) in permuted]
        if not rows:
            continue
        rungs[f"kev-{name}"] = {
            "source": str(path.relative_to(ROOT)),
            "rows_before_restriction": len(load_jsonl(path)),
            "by_family": score_rows(rows, lambda r: argmax(r["p"])),
        }

    comparison = {
        "what_this_is": "every engine scored with the decision models' own rule on the SAME census-matched "
                        "rows: the half whose prompt is on disk (the census's own permuted record). The "
                        "other half of the census-matched population (910 panel_choice rows) is the "
                        "capped suite's clean-variant record of the same cell+family, and no raw ask dump "
                        "holds its prompt, so the LLM baseline cannot be run on it.",
        "rows_restricted_to": "census-suite-ids.jsonl entries with variant == 'permuted'",
        "triples": len(permuted),
        "jev_on_the_same_rows": score_rows(jev, lambda r: argmax(r["p"])),
        "served_kev_rungs_on_the_same_rows": rungs,
        "ours_frozen_on_the_same_rows": score_rows(
            frozen, lambda r: r["top1"] if r.get("top1") is not None else -1),
        "source_rows_total": {str(Path(args.jev_rows)): len(jev_all),
                              str(Path(args.frozen_rows)): len(frozen_all)},
    }

    family_counts = Counter(f for _, _, f in permuted)
    comparison["rows_per_family"] = dict(sorted(family_counts.items()))

    # The variant comparison, keyed on the run directory AND on whether the summary was restricted by
    # `--subset-ids`: the census-matched numbers are the *same run* as the census numbers with a filter
    # applied, so the run directory alone does not identify a summary.
    def pick(run: str, subset: bool, family: str = "panel_choice") -> dict | None:
        for summary in summaries.values():
            if Path(summary.get("runs_dir", "")).name != run:
                continue
            if (summary.get("subset_ids") is not None) != subset:
                continue
            block = summary["per_family"].get(family)
            if block is None:
                return None
            return {"n": block["n"], "accuracy": block["accuracy"],
                    "abstain_pick_rate": block["abstain_pick_rate"],
                    "parse_failures": block["parse_failures"],
                    "label": summary.get("label"), "runs_dir": summary.get("runs_dir"),
                    "summary_file": [k for k, v in summaries.items() if v is summary][0],
                    "subset": subset}
        return None

    variant_comparison = {
        "what_this_is": "both documented prompts, run on both populations, so the headline can be the "
                        "better of the two without the choice being invisible.",
        "panel_choice": {
            "smoke_draw_200": {"a_minimal": pick("smoke-varA", False),
                               "b_explicit": pick("smoke-varB", False)},
            "census_2510": {"a_minimal": pick("census-varA", False),
                            "b_explicit": pick("census-varB", False)},
            "census_matched_909": {"a_minimal": pick("census-varA", True),
                                   "b_explicit": pick("census-varB", True)},
        },
        "other_families_census_matched_909": {
            family: {"a_minimal": pick("census-varA", True, family),
                     "b_explicit": pick("census-varB", True, family)}
            for family in ("panel_miss", "evidence_diluted")
        },
        "other_families_census_2510": {
            family: {"a_minimal": pick("census-varA", False, family),
                     "b_explicit": pick("census-varB", False, family)}
            for family in ("panel_miss", "evidence_diluted")
        },
    }

    payload = {
        "what_this_is": "a general instruct-tuned LLM, zero-shot, on the decision models' own questions, "
                        "scored by the decision models' own rule. No fine-tuning, no adapters, no "
                        "task-specific head, no calibration.",
        "model": {
            "hub_id": "Qwen/Qwen3.5-9B",
            "revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
            "why": "the post-trained (instruct) sibling of Qwen/Qwen3.5-9B-Base, which is the base the "
                   "Kev-9B rung was built from: same family, same size class, so the comparison is "
                   "like-for-like. Qwen/Qwen3.5-9B-Instruct does not exist on the Hub.",
            "params_b": 8.95,
            "dtype": "bfloat16",
            "download_bytes": 19329439261,
            "quirk": "the checkpoint is a vision-language `Qwen3_5ForConditionalGeneration`; it is loaded "
                     "with `AutoModelForCausalLM` so only the text tower is instantiated, and the "
                     "tokenizer's own `enable_thinking=False` path is used so no reasoning trace precedes "
                     "the answer. Both prompt variants are served identically.",
        },
        "summaries": {name: {"label": s.get("label"), "reference": s.get("reference"),
                             "runs_dir": s.get("runs_dir"),
                             "asks": s["asks"], "questions": s["questions"],
                             "subset_ids": s.get("subset_ids"), "require_variant": s.get("require_variant"),
                             "overall": s["overall"], "per_family": s["per_family"],
                             "parse_rules": s["parse_rules"], "serving": s["serving"]}
                      for name, s in summaries.items()},
        "same_rows_as_the_census_matched_table": comparison,
        "variant_comparison": variant_comparison,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    print(f"  summaries read: {len(summaries)}")
    print(f"  same-rows comparison: {comparison['rows_per_family']}")
    for engine in ("jev_on_the_same_rows", "ours_frozen_on_the_same_rows"):
        pc = comparison[engine].get("panel_choice", {})
        print(f"  {engine:32s} panel_choice n={pc.get('n')} acc={pc.get('accuracy')}")
    for pop, arms in variant_comparison["panel_choice"].items():
        cells = " | ".join(f"{k}: {v['accuracy'] if v else 'not measured'}" for k, v in arms.items())
        print(f"  {pop:22s} {cells}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
