#!/usr/bin/env python3
"""The extra seeds of the 4B and 9B fine-tunes, on the census-matched population.

Why this exists
---------------

The 0.8B rung of the fine-tuning ladder has **five** seeds, and their forced-naming spread is
0.3793-0.4854 (a 2.6x range). A single run at 4B and a single run at 9B therefore cannot separate a
real effect from seed noise:

* ``runs/delta-kev-4b-s1`` is a **null** against its own rung — 0.6097 against Kev-4B's 0.6168,
  paired p = 0.52;
* ``runs/delta-kev-9b-s1`` is a **loss** — 0.5668 against Kev-9B's 0.6295, paired p = 7.8e-10.

This script measures the extra seeds (4B seeds 2 and 3, 9B seed 2) on the same 1,819 census-matched
``panel_choice`` questions, under the same settled rule every other row in the table was measured
with, and reports each seed against **its own rung** with the paired test — plus the seed mean and
spread at each size. What it answers is whether the verdicts above survive more seeds, not whether
they can be restated.

What was *not* re-derived
-------------------------

The population, the join, the abstain index and the forced rule are not restated here:

* the population is ``reports/census-suite-ids.jsonl``'s ``panel_choice`` rows, joined on the row's
  own id — the same 1,819 rows ``scripts/arms-forced.py`` and ``scripts/9b-finetune-score.py`` use;
* ``argmax``, ``forced`` and ``binom_two_sided`` are taken from ``reports/abstain-reconciliation.py``'s
  own source (its top-level function definitions, via ``ast`` — the module body is *not* executed,
  because running it would rewrite ``reports/abstain-reconciliation.json``);
* the abstain index is ``sckevtype.eval.abstain_index``, the same function the runner uses, and the
  dump's own ``abstain`` field is required to agree with it on every row;
* the rungs' forced verdicts are attributed to the 1,819 suite rows by ``(dataset, cell, family)`` —
  the attribution ``reports/both-forced-table.json`` uses — and their recomputed forced counts are
  required to reproduce the settled 1122/1819 (Kev-4B) and 1145/1819 (Kev-9B).

Every row below is computed from the per-question dump of the arm it names. No number is transcribed
from another report.

Usage
-----

    source scripts/wsl/env.sh && uv run --no-sync python scripts/extra-seeds-score.py \\
        --job-ids delta-kev-4b-s2=...,delta-kev-4b-s3=...,delta-kev-9b-s2=...

Reads only dumps already on disk (the re-scored arm dumps, the two census rung dumps, the population
and the two settled tables); runs no model and submits no job. Writes ``reports/extra-seeds.json``
and ``reports/extra-seeds.md`` — new files; it never writes a report that already exists.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sckevtype.eval import abstain_index

#: The script that settled the forced rule — the same source both existing aggregators take it from.
RULE_SOURCE = ROOT / "analysis" / "abstain-reconciliation.py"
RULE_NAMES = ("argmax", "forced", "binom_two_sided")

#: The population: the census-matched suite's own ``panel_choice`` rows, joined on the row's own id.
SUITE_IDS = "reports/census-suite-ids.jsonl"
POPULATION_FAMILY = "panel_choice"

#: The settled tables the rungs' references are checked against.
SETTLED = "reports/both-forced-table.json"
ARMS_FORCED = "reports/arms-forced.json"

#: Each size's released rung: its census dump, and the forced count the settled tables publish for
#: it on this population (reproduced here, not trusted).
RUNGS = {
    "kev-4b": {"dump": "reports/kev-rungs/4b-rows.jsonl", "settled_forced_correct": 1122},
    "kev-9b": {"dump": "reports/kev-rungs/9b-rows.jsonl", "settled_forced_correct": 1145},
}

#: The seeds per size. ``arm`` is the run directory's own name (``runs/<arm>``), which is also the
#: name of the re-scored dump the cluster pass wrote.
SIZES = {
    "4b": {"label": "4B fine-tune (Kev-4B warm start)", "rung": "kev-4b",
           "base": "Qwen/Qwen3.5-4B-Base", "source": "jaredpalmer/kev-4b",
           "seeds": [{"arm": "delta-kev-4b-s1", "seed": 1},
                     {"arm": "delta-kev-4b-s2", "seed": 2},
                     {"arm": "delta-kev-4b-s3", "seed": 3}]},
    "9b": {"label": "9B fine-tune (Kev-9B warm start)", "rung": "kev-9b",
           "base": "Qwen/Qwen3.5-9B-Base", "source": "jaredpalmer/kev-9b",
           "seeds": [{"arm": "delta-kev-9b-s1", "seed": 1},
                     {"arm": "delta-kev-9b-s2", "seed": 2}]},
}

#: The runs' own facts, generated on the cluster from ``sacct`` and each run's provenance — read, not
#: invented. Absent means the facts are reported as not recorded.
RUN_FACTS = "reports/extra-seeds-run-facts.json"

OUT_JSON = ROOT / "reports" / "extra-seeds.json"
OUT_MD = ROOT / "reports" / "extra-seeds.md"

#: Files this script must never write: the pre-existing reports.
FROZEN = {ROOT / "reports" / "arms-forced.json", ROOT / "reports" / "arms-forced.md",
          ROOT / "reports" / "9b-finetune.json", ROOT / "reports" / "9b-finetune.md",
          ROOT / "reports" / "abstain-reconciliation.json", ROOT / "reports" / "PAIRED-TESTS.md",
          ROOT / "reports" / "TABLE-census-matched.md"}


# --------------------------------------------------------------------------- the settled rule
def load_rule(path: Path = RULE_SOURCE, names: tuple[str, ...] = RULE_NAMES) -> dict:
    """The settled functions, taken from the report script's own source text.

    Identical in intent to ``scripts/arms-forced.py``'s and ``scripts/9b-finetune-score.py``'s
    loaders: importing ``reports/abstain-reconciliation.py`` would recompute and rewrite
    ``reports/abstain-reconciliation.json``, so its *definitions* are parsed out and executed alone.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    definitions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    missing = [name for name in names if name not in definitions]
    if missing:
        raise SystemExit(f"{path}: no top-level def for {missing}; the forced rule cannot be taken "
                         f"from the file that settled it, and re-deriving it here would be a second "
                         f"source of truth")
    imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    module = ast.Module(body=[*imports, *[definitions[name] for name in names]], type_ignores=[])
    namespace: dict = {}
    exec(compile(module, str(path), "exec"), namespace)      # noqa: S102 -- see the docstring
    return {name: namespace[name] for name in names}


# --------------------------------------------------------------------------- loading
def read_jsonl(path: str | Path) -> list[dict]:
    with open(ROOT / path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(ROOT / path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def by_id(rows: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for row in rows:
        if row["id"] in out:
            raise SystemExit(f"row {row['id']!r} appears twice in a dump -- refusing to pick one")
        out[row["id"]] = row
    return out


def population() -> tuple[list[str], list[dict]]:
    rows = [r for r in read_jsonl(SUITE_IDS) if r.get("family") == POPULATION_FAMILY]
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise SystemExit(f"{SUITE_IDS}: repeated ids in the population -- the join would be ambiguous")
    return ids, rows


# --------------------------------------------------------------------------- measurement
def measure_arm(ids: list[str], rule: dict, dump_path: str) -> dict:
    """One seed's row: served, abstain-pick, forced, gain, hidden correct — plus what was checked."""
    if not (ROOT / dump_path).exists():
        return {"not_measured": f"{dump_path} is not on disk: this seed has not been re-scored, and "
                                f"without the option-probability vector `p` the forced choice is not "
                                f"recoverable"}
    dumped_all = read_jsonl(dump_path)
    wanted = set(ids)
    dumped = by_id([r for r in dumped_all if r["id"] in wanted])
    missing = [i for i in ids if i not in dumped]
    extra = sorted({r["id"] for r in dumped_all} - wanted)
    if missing or extra:
        return {"not_measured":
                f"the dump is not the population: {len(missing)} of the {len(ids)} ids are absent "
                f"(e.g. {missing[:5]}) and {len(extra)} ids outside it are present "
                f"(e.g. {extra[:5]}). A partial population is not the population."}

    served_ok = served_top1_ok = abstain_picks = forced_ok = gained = 0
    checks = {"rows_checked": 0, "no_p": [], "abstain_index_disagreement": [],
              "top1_not_argmax_of_its_own_p": [], "missing_abstain_field": []}
    for i in ids:
        r = dumped[i]
        checks["rows_checked"] += 1
        p = r.get("p")
        if not p:
            checks["no_p"].append(i)
            continue
        if r.get("abstain") is None:
            checks["missing_abstain_field"].append(i)
        elif r["abstain"] != abstain_index(r):
            checks["abstain_index_disagreement"].append(i)
        if r.get("top1") != rule["argmax"](p):
            checks["top1_not_argmax_of_its_own_p"].append(i)
        served_hit = rule["argmax"](p) == r["label"]
        served_ok += int(served_hit)
        served_top1_ok += int(r.get("top1") == r["label"])
        abstain_picks += int(r.get("top1") == r.get("abstain"))
        hit = rule["forced"](r) == r["label"]
        forced_ok += int(hit)
        gained += int(hit and not served_hit)

    n = len(ids)
    for key in ("no_p", "abstain_index_disagreement", "top1_not_argmax_of_its_own_p",
                "missing_abstain_field"):
        checks[key] = {"count": len(checks[key]), "ids": checks[key][:5]}
    checks["served_from_p_equals_served_from_top1"] = served_ok == served_top1_ok
    clean = not any(checks[key]["count"] for key in
                    ("no_p", "abstain_index_disagreement", "top1_not_argmax_of_its_own_p",
                     "missing_abstain_field"))
    checks["verdict"] = ("the dump is internally consistent: every row carries `p`, its own `top1` "
                         "is the argmax of that vector, and its `abstain` field agrees with the "
                         "abstain index" if clean
                         else "THE DUMP FAILS ITS OWN INTERNAL CHECKS -- see the counts")
    return {
        "n": n,
        "dump": dump_path,
        "served": {"correct": served_ok, "accuracy": served_ok / n,
                   "dump_top1_correct": served_top1_ok,
                   "rule": "argmax(row['p']) == label; the dump's own `top1` is the same count"},
        "abstain_pick": {"picks": abstain_picks, "rate": abstain_picks / n},
        "forced": {"n": n, "correct": forced_ok, "accuracy": forced_ok / n,
                   "rule": "argmax over p with the abstain index removed == label"},
        "delta_points": (forced_ok - served_ok) / n * 100.0,
        "questions_gained": gained,
        "hidden_correct_share_of_abstentions": (gained / abstain_picks) if abstain_picks else None,
        "checks": checks,
    }


def census_rung(rule: dict, suite_rows: list[dict], path: str) -> tuple[dict, dict[str, dict]]:
    """A released rung's numbers on this same 1,819, attributed by (dataset, cell, family).

    The iteration is over the **1,819 suite rows**, not over the distinct triples: 322 of the 1,497
    triples carry two suite rows, and a census verdict reached by a doubled triple is counted twice
    — that doubling is what makes the denominator 1,819.
    """
    census = {r["id"]: r for r in read_jsonl(path) if r.get("family") == POPULATION_FAMILY}
    served = forced = abstain_picks = gained = 0
    absent: list[str] = []
    by_suite_id: dict[str, dict] = {}
    for suite_row in suite_rows:
        key = f"test/{suite_row['dataset']}/{suite_row['cell']}/{suite_row['family']}"
        row = census.get(key)
        if row is None:
            absent.append(key)
            continue
        served_hit = rule["argmax"](row["p"]) == row["label"]
        hit = rule["forced"](row) == row["label"]
        served += int(served_hit)
        abstain_picks += int(rule["argmax"](row["p"]) == row["abstain"])
        forced += int(hit)
        gained += int(hit and not served_hit)
        by_suite_id[suite_row["id"]] = row
    n = len(suite_rows)
    return {"source": path,
            "attribution": "each of the 1,819 suite rows -> its census record by (dataset, cell, "
                           "family); a doubled triple counts twice",
            "n": n, "census_records_absent": len(absent), "absent_examples": absent[:5],
            "served": {"correct": served, "accuracy": served / n},
            "abstain_pick": {"picks": abstain_picks, "rate": abstain_picks / n},
            "forced": {"n": n, "correct": forced, "accuracy": forced / n},
            "questions_gained": gained}, by_suite_id


def paired_against_rung(rule: dict, arm_rows: dict, rung_by_suite_id: dict, ids: list[str]) -> dict:
    """Discordance between a seed's forced verdict and its rung's, on the same rows, with its p."""
    b = c = skipped = 0
    for i in ids:
        if i not in rung_by_suite_id:
            skipped += 1
            continue
        rung_row = rung_by_suite_id[i]
        arm_hit = rule["forced"](arm_rows[i]) == arm_rows[i]["label"]
        rung_hit = rule["forced"](rung_row) == rung_row["label"]
        b += int(arm_hit and not rung_hit)
        c += int(rung_hit and not arm_hit)
    return {"arm_right_rung_wrong": b, "rung_right_arm_wrong": c,
            "rows_compared": len(ids) - skipped, "rows_skipped_no_census_row": skipped,
            "p_two_sided_exact": rule["binom_two_sided"](b, c)}


def spread(values: list[float]) -> dict:
    """Mean and spread of a small list of seed accuracies — reported as counts as well as rates."""
    n = len(values)
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n if n > 1 else 0.0
    return {"n_seeds": n, "mean": mean, "min": min(values), "max": max(values),
            "range": max(values) - min(values),
            "sd_population": variance ** 0.5,
            "spread_factor": (max(values) / min(values)) if min(values) else None,
            "seeds_above_their_rung": None}      # filled by the caller


# --------------------------------------------------------------------------- writing
def fmt(value: float | None, digits: int = 4) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def write_md(out: dict, path: Path = OUT_MD) -> None:
    lines: list[str] = []
    pop = out["populations"]["census_matched_1819"]
    lines.append("# Extra seeds of the 4B and 9B fine-tunes\n")
    lines.append("*Machine-readable companion: `reports/extra-seeds.json`. Two more seeds of the 4B "
                 "fine-tune and one more seed of the 9B fine-tune, run with the same recipes as "
                 "their seed-1 runs and scored on the manuscript's primary measure — the "
                 f"{pop['n']:,} census-matched `panel_choice` questions — under the settled rule of "
                 "`reports/abstain-reconciliation.py` (taken from that file's own source), with the "
                 "columns of `reports/arms-forced.json` and `reports/9b-finetune.json`.*\n")

    lines.append("## 1. Why more seeds\n")
    lines.append(out["why"][0] + "\n")
    lines.append(out["why"][1] + "\n")

    lines.append("## 2. The runs\n")
    lines.append("| run | seed | Slurm job | state | wall | optimizer steps | peak (worst card) | "
                 "weights dtype | token budget | GPUs |\n|---|---:|---|---|---:|---:|---:|---|---:|---:|")
    for size_key, size in out["sizes"].items():
        for seed in size["seeds"]:
            facts = seed.get("run_facts") or {}

            def shown(key: str) -> str:
                value = facts.get(key)
                return "—" if value is None else str(value)

            lines.append(f"| `runs/{seed['arm']}` | {seed['seed']} | {shown('job_id')} | "
                         f"{shown('state')} | {shown('elapsed')} | "
                         f"{shown('optimizer_steps')} | "
                         f"{fmt_gib(facts.get('peak_device_bytes'))} | "
                         f"{shown('weights_dtype')} | {fmt_budget(facts.get('token_budget'))} | "
                         f"{shown('gpus')} |")
    lines.append("")
    for size_key, size in out["sizes"].items():
        lines.append(f"* **{size['label']}** — warm-started from `{size['source']}` on "
                     f"`{size['base']}`; recipe: one epoch, batch 1 x accum 8 x checkpointing 1, "
                     f"LoRA r=16 on all targets, lr 2e-4, seed per run, frozen backbone sharded over "
                     f"4x A100-40G. {size['recipe_note']}")
    lines.append("")

    lines.append("## 3. The population and the rules\n")
    lines.append(f"**{pop['label']}** — n = **{pop['n']}**. {pop['definition']}\n")
    lines.append("| rule | what was applied |\n|---|---|")
    for name, text in out["rules"].items():
        lines.append(f"| *{name}* | {text} |")
    lines.append("")

    for size_key, size in out["sizes"].items():
        ordinal = {"4b": "4.1", "9b": "4.2"}.get(size_key, f"4.{size_key}")
        lines.append(f"## {ordinal} The {size['label']}: one table per seed\n")
        lines.append("| seed | run | served accuracy (abstain available) | abstain-pick rate | "
                     "forced accuracy | Δ (points) | hidden correct (share of abstentions) | "
                     "dump sha256 |\n|---|---|---:|---:|---:|---:|---:|---|")
        for seed in size["seeds"]:
            cell = seed.get("census_matched_1819") or {}
            if "not_measured" in cell or not cell:
                lines.append(f"| {seed['seed']} | `runs/{seed['arm']}` | — | — | — | — | — | "
                             f"*not measured: {(cell or {}).get('not_measured', 'no dump')[:230]}* |")
                continue
            lines.append(f"| {seed['seed']} | `runs/{seed['arm']}` | "
                         f"{cell['served']['correct']}/{cell['n']} = {fmt(cell['served']['accuracy'])} | "
                         f"{cell['abstain_pick']['picks']}/{cell['n']} = {fmt(cell['abstain_pick']['rate'])} | "
                         f"**{cell['forced']['correct']}/{cell['n']} = {fmt(cell['forced']['accuracy'])}** | "
                         f"{'+' if cell['delta_points'] >= 0 else ''}{cell['delta_points']:.1f} pt "
                         f"({cell['questions_gained']} questions) | "
                         f"{fmt(cell['hidden_correct_share_of_abstentions'])} | "
                         f"`{seed['dump_sha256'][:16]}…` |")
        lines.append("")
        sp = size["spread"]
        if sp["n_seeds"] > 1:
            lines.append(f"**Seed mean and spread (forced accuracy, {sp['n_seeds']} seeds).** "
                         f"mean **{fmt(sp['mean'])}**, min {fmt(sp['min'])} "
                         f"(seed {sp['min_seed']}), max {fmt(sp['max'])} (seed {sp['max_seed']}), "
                         f"range **{sp['range'] * 100:.1f} points**, "
                         f"spread factor {sp['spread_factor']:.2f}x, "
                         f"population sd {sp['sd_population']:.4f}.")
        else:
            lines.append(f"**Seed mean and spread (forced accuracy, {sp['n_seeds']} seed).** "
                         f"mean **{fmt(sp['mean'])}**; a single seed has no spread, which is the "
                         f"point of this report.")
        lines.append("")

    lines.append("## 5. Each seed against its own rung\n")
    lines.append(out["comparison_note"] + "\n")
    lines.append("| size | seed | run | rung | rung forced | seed forced | gap (pt) | paired "
                 "(seed right / rung right) | exact two-sided p | exceeds its own rung |\n"
                 "|---|---:|---|---|---:|---:|---:|---:|---:|---|")
    for size_key, size in out["sizes"].items():
        rung = size["rung_forced"]
        for seed in size["seeds"]:
            cell = seed.get("census_matched_1819") or {}
            if not cell or "not_measured" in cell:
                lines.append(f"| {size_key.upper()} | {seed['seed']} | `runs/{seed['arm']}` | "
                             f"{size['rung']} | {rung['correct']}/{rung['n']} = "
                             f"{fmt(rung['accuracy'])} | — | — | — | — | *not measured* |")
                continue
            paired = seed["paired_vs_rung"]
            lines.append(f"| {size_key.upper()} | {seed['seed']} | `runs/{seed['arm']}` | "
                         f"{size['rung']} | {rung['correct']}/{rung['n']} = {fmt(rung['accuracy'])} | "
                         f"{cell['forced']['correct']}/{cell['n']} = "
                         f"{fmt(cell['forced']['accuracy'])} | "
                         f"{seed['gap_to_rung_points']:+.1f} | "
                         f"{paired['arm_right_rung_wrong']} / {paired['rung_right_arm_wrong']} | "
                         f"{paired['p_two_sided_exact']:.3g} | "
                         f"{'yes' if seed['exceeds_its_own_rung'] else '**no**'} |")
    lines.append("")

    lines.append("## 6. Does more than one seed change the verdicts?\n")
    for para in out["verdict"]:
        lines.append(para + "\n")

    lines.append("## 7. What was checked\n")
    for check in out["checks"]:
        lines.append(f"* {check}")
    lines.append("")

    lines.append("## 8. Reproducing it\n")
    lines.append("```")
    lines.extend(out["provenance"]["commands"])
    lines.append("```")
    lines.append("")
    lines.append("* training job ids: " + (", ".join(out["provenance"]["train_job_ids"]) or "—"))
    lines.append("* re-score job ids: " + (", ".join(out["provenance"]["rescore_job_ids"]) or "—"))
    lines.append(f"* `sckevtype/eval/run.py` sha256 `{out['provenance']['sha256'].get('sckevtype/eval/run.py', '—')}`")
    lines.append(f"* `scripts/score-forced-rows.py` sha256 "
                 f"`{out['provenance']['sha256'].get('scripts/score-forced-rows.py', '—')}`")
    lines.append("")
    lines.append("Every file read, with its digest:\n")
    lines.append("| file | sha256 |\n|---|---|")
    for rel, digest in sorted(out["provenance"]["sha256"].items()):
        lines.append(f"| `{rel}` | `{digest}` |")
    lines.append("")

    lines.append("## 9. Limits\n")
    for limit in out["limits"]:
        lines.append(f"* {limit}")
    lines.append("")
    text = "\n".join(lines) + "\n"
    Path(path).write_text(text, encoding="utf-8")


def fmt_gib(value) -> str:
    return "—" if value is None else f"{value / 2**30:.2f} GiB"


def fmt_budget(value) -> str:
    """The token budget as it reads in the run's own files: a number, or 'off'.

    A run whose ``training_config.json`` carries no ``token_budget`` key ran with the flag off (the
    4B's recipe), and the 9B's ``provenance.json`` records the budget as a block of counts — both
    are reported as what they are rather than as a bare ``None``.
    """
    if value is None:
        return "off (0)"
    if isinstance(value, dict):
        return str(value.get("budget", "—"))
    return "off (0)" if value == 0 else str(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="The extra 4B/9B seeds on the census-matched population.")
    parser.add_argument("--json", default=str(OUT_JSON))
    parser.add_argument("--md", default=str(OUT_MD))
    parser.add_argument("--run-facts", default=RUN_FACTS)
    parser.add_argument("--train-job-ids", default="",
                        help="the training Slurm job ids, comma-separated; recorded verbatim")
    parser.add_argument("--rescore-job-ids", default="",
                        help="the re-score Slurm job ids, comma-separated; recorded verbatim")
    args = parser.parse_args(argv)

    for target in (Path(args.json), Path(args.md)):
        if target in FROZEN:
            raise SystemExit(f"{target} is a pre-existing report; this script writes only new files")

    rule = load_rule()
    ids, suite_rows = population()

    facts = {}
    if (ROOT / args.run_facts).exists():
        facts = json.loads((ROOT / args.run_facts).read_text(encoding="utf-8"))

    rung_cache: dict[str, tuple[dict, dict[str, dict]]] = {}
    for name, rung in RUNGS.items():
        reference, by_suite_id = census_rung(rule, suite_rows, rung["dump"])
        if reference["census_records_absent"]:
            raise SystemExit(f"{rung['dump']}: {reference['census_records_absent']} of the "
                             f"{reference['n']} suite rows have no census record "
                             f"(e.g. {reference['absent_examples']}) -- the rung's reference would "
                             f"be taken on a subset")
        if reference["forced"]["correct"] != rung["settled_forced_correct"]:
            raise SystemExit(f"{rung['dump']}: the rung's forced count recomputed here is "
                             f"{reference['forced']['correct']}, the settled value is "
                             f"{rung['settled_forced_correct']} -- the rule or the population here "
                             f"is not the settled one, and nothing measured on it would be "
                             f"comparable")
        rung_cache[name] = (reference, by_suite_id)

    sizes_out: dict[str, dict] = {}
    for size_key, size in SIZES.items():
        reference, rung_by_suite_id = rung_cache[size["rung"]]
        seeds_out = []
        accuracies: list[float] = []
        for seed in size["seeds"]:
            dump = f"reports/forced-rows-{seed['arm']}.jsonl"
            cell = measure_arm(ids, rule, dump)
            entry = {"arm": seed["arm"], "seed": seed["seed"], "dump": dump,
                     "run_facts": (facts.get("runs") or {}).get(seed["arm"]),
                     "census_matched_1819": cell}
            if "not_measured" in cell:
                rf = entry["run_facts"] or {}
                if rf.get("job_id"):
                    cell["not_measured"] += (f" Its training run is Slurm job {rf['job_id']} "
                                             f"({rf.get('state', 'state unknown')}"
                                             + (f", elapsed {rf['elapsed']}" if rf.get("elapsed") else "")
                                             + "), so the dump cannot exist yet.")
                entry["dump_sha256"] = None
                entry["paired_vs_rung"] = None
                entry["gap_to_rung_points"] = None
                entry["exceeds_its_own_rung"] = None
            else:
                entry["dump_sha256"] = sha256_file(dump)
                arm_rows = by_id([r for r in read_jsonl(dump) if r["id"] in set(ids)])
                paired = paired_against_rung(rule, arm_rows, rung_by_suite_id, ids)
                if paired["rows_compared"] != len(ids):
                    raise SystemExit(f"the paired comparison for {seed['arm']} covered "
                                     f"{paired['rows_compared']} of {len(ids)} rows -- a comparison "
                                     f"on a subset must not be reported as one on the population")
                entry["paired_vs_rung"] = paired
                entry["gap_to_rung_points"] = ((cell["forced"]["accuracy"]
                                                - reference["forced"]["accuracy"]) * 100.0)
                entry["exceeds_its_own_rung"] = cell["forced"]["accuracy"] > reference["forced"]["accuracy"]
                accuracies.append(cell["forced"]["accuracy"])
            seeds_out.append(entry)
        sp = spread(accuracies) if accuracies else {"n_seeds": 0}
        if accuracies:
            sp["min_seed"] = next(s["seed"] for s in seeds_out
                                  if (s.get("census_matched_1819") or {}).get("forced", {}).get("accuracy") == sp["min"])
            sp["max_seed"] = next(s["seed"] for s in seeds_out
                                  if (s.get("census_matched_1819") or {}).get("forced", {}).get("accuracy") == sp["max"])
            sp["seeds_above_their_rung"] = [s["seed"] for s in seeds_out if s["exceeds_its_own_rung"]]
            sp["per_seed_gap_points"] = {str(s["seed"]): round(s["gap_to_rung_points"], 2)
                                         for s in seeds_out if s["gap_to_rung_points"] is not None}
            sp["per_seed_p"] = {str(s["seed"]): s["paired_vs_rung"]["p_two_sided_exact"]
                                for s in seeds_out if s["paired_vs_rung"]}
        sizes_out[size_key] = {
            "label": size["label"], "source": size["source"], "base": size["base"],
            "rung": size["rung"], "seeds": seeds_out, "spread": sp,
            "rung_forced": reference["forced"], "rung_served": reference["served"],
            "rung_abstain_pick": reference["abstain_pick"],
            "rung_dump": RUNGS[size["rung"]]["dump"],
            "recipe_note": "",
        }

    # ---- the prose the numbers support, assembled from them (never hand-typed) ----
    four = sizes_out["4b"]
    nine = sizes_out["9b"]
    four_seeds = [s for s in four["seeds"] if s.get("census_matched_1819", {}).get("forced")]
    nine_seeds = [s for s in nine["seeds"] if s.get("census_matched_1819", {}).get("forced")]
    four_note = (f"Its {four['spread']['n_seeds']} seed(s)' forced accuracy: "
                 + ", ".join(f"{s['seed']}:{s['census_matched_1819']['forced']['accuracy']:.4f}"
                             for s in four_seeds)
                 + f" (mean {four['spread']['mean']:.4f}, range "
                   f"{four['spread']['range'] * 100:.1f} pt).")
    nine_note = (f"Its {nine['spread']['n_seeds']} seed(s)' forced accuracy: "
                 + ", ".join(f"{s['seed']}:{s['census_matched_1819']['forced']['accuracy']:.4f}"
                             for s in nine_seeds)
                 + (f" (mean {nine['spread']['mean']:.4f}, range "
                    f"{nine['spread']['range'] * 100:.1f} pt)." if nine["spread"]["n_seeds"] > 1 else "."))
    four["recipe_note"] = four_note
    nine["recipe_note"] = nine_note

    verdict: list[str] = []
    for size_key, size, note in (("4b", four, four_note), ("9b", nine, nine_note)):
        sp = size["spread"]
        rung = size["rung_forced"]
        seeds = [s for s in size["seeds"] if s.get("census_matched_1819", {}).get("forced")]
        if len(seeds) < 2:
            verdict.append(f"**{size['label']}: not yet decided by more than one seed.** "
                           f"{note} One seed has no spread, so a single run at this size cannot "
                           f"separate a real effect from seed noise — which is the question this "
                           f"report was asked.")
            continue
        above = [s for s in seeds if s["exceeds_its_own_rung"]]
        gaps = [s["gap_to_rung_points"] for s in seeds]
        significant_neg = [s for s in seeds if not s["exceeds_its_own_rung"]
                           and s["paired_vs_rung"]["p_two_sided_exact"] < 0.05]
        significant_pos = [s for s in seeds if s["exceeds_its_own_rung"]
                           and s["paired_vs_rung"]["p_two_sided_exact"] < 0.05]
        nulls = [s for s in seeds if s["paired_vs_rung"]["p_two_sided_exact"] >= 0.05]
        p_text = ", ".join(f"seed {s['seed']}: {s['paired_vs_rung']['p_two_sided_exact']:.3g}"
                           for s in seeds)
        verdict.append(
            f"**{size['label']}.** {note} Rung forced = {rung['correct']}/{rung['n']} = "
            f"{rung['accuracy']:.4f}. Against its own rung, {len(above)} of {len(seeds)} seeds "
            f"exceed it; gaps range {min(gaps):+.1f} to {max(gaps):+.1f} points and the exact paired "
            f"p-values are {p_text}. {len(significant_pos)} seed(s) significantly above the rung, "
            f"{len(significant_neg)} significantly below, {len(nulls)} indistinguishable from it "
            f"(p >= 0.05). The seed mean is {sp['mean']:.4f} (range {sp['range'] * 100:.1f} points, "
            f"spread factor {sp['spread_factor']:.2f}x).")

    out = {
        "what_this_is": "The extra seeds of the 4B and 9B fine-tunes on the 1,819 census-matched "
                        "panel_choice questions: served, abstain-pick, forced, gain and hidden "
                        "correct per seed, each seed against its own rung with the exact paired "
                        "test, and the seed mean and spread per size.",
        "why": [
            "The 0.8B rung of the fine-tuning ladder has five seeds and its forced-naming accuracy "
            "spans 0.3793-0.4854 (a 2.6x range), so a single run at 4B or 9B cannot separate a real "
            "effect from seed noise.",
            "The 4B fine-tune's seed 1 is a null against its own rung (0.6097 against 0.6168, "
            "p = 0.52) and the 9B fine-tune's seed 1 is a loss (0.5668 against 0.6295, p = 7.8e-10). "
            "Extra seeds are what turn both into bounded statements: two more seeds at 4B and one "
            "more at 9B, run with the same recipes as their seed-1 runs.",
        ],
        "populations": {"census_matched_1819": {
            "label": "the 1,819 census-matched panel_choice questions (the manuscript's primary measure)",
            "definition": "the suite rows of reports/census-suite-ids.jsonl with family panel_choice, "
                          "joined for each arm on the row's own id (`test/custom/<n>`, one row each). "
                          "The census-level rungs are attributed to each suite row by (dataset, cell, "
                          "family), as reports/abstain-reconciliation.py and "
                          "reports/both-forced-table.json do; 322 of the 1,497 panel_choice triples "
                          "carry two suite rows, which is what makes the denominator 1,819.",
            "n": len(ids), "source": SUITE_IDS}},
        "rules": {
            "served": "argmax(row['p']) == label on the re-scored row; the dump's own `top1` is "
                      "reported beside it and is the same count (checked per row).",
            "abstain_index": "sckevtype.eval.abstain_index(row) -- the option whose key is in "
                             "sckevtype.data.families.NONE_OPTIONS; checked equal to the row's own "
                             "`abstain` field on all 1,819 rows.",
            "forced": "argmax over the row's p vector with the abstain option removed -- taken from "
                      "reports/abstain-reconciliation.py's own source, not re-implemented.",
            "delta_points": "forced accuracy minus served accuracy, in percentage points.",
            "hidden_correct_share_of_abstentions": "questions_gained / abstain-pick count -- the "
                                                   "share of abstentions that forcing converts into "
                                                   "a correct answer."},
        "sizes": sizes_out,
        "comparison_note":
            "Each seed is compared against the released rung of **its own size** — the checkpoint the "
            "run was warm-started from — not against the manuscript's §2.5 reference (the 0.8B rung). "
            "Both sides of every comparison are recomputed here from the seed's own re-scored dump "
            "and the rung's own census dump, on the same 1,819 rows; the test is the exact two-sided "
            "binomial on discordant pairs (McNemar), the same one reports/PAIRED-TESTS.md uses.",
        "verdict": verdict,
        "checks": [],
        "provenance": {
            "method": "one re-score per seed on the population with `p` retained "
                      "(<cluster-job> -> scripts/score-forced-rows.py), then "
                      "one aggregation pass under the settled rule; no number here is transcribed "
                      "from another file.",
            "train_job_ids": [e.strip() for e in args.train_job_ids.split(",") if e.strip()],
            "rescore_job_ids": [e.strip() for e in args.rescore_job_ids.split(",") if e.strip()],
            "commands": [
                "# 1. the two extra 4B seeds and the extra 9B seed (one Slurm job each, 4x A100-40G):",
                "bash <cluster-job> jaredpalmer/kev-4b 2 bf16 4 0",
                "bash <cluster-job> jaredpalmer/kev-4b 3 bf16 4 0",
                "MG_WALL=16:00:00 bash <cluster-job> jaredpalmer/kev-9b 2 bf16 4 0 32768",
                "# 2. the re-score of every new seed on the population, keeping `p` (one job, one card):",
                "sbatch -p <gpu-partition> --gres=gpu:1 --cpus-per-task=12 -n 1 -N 1 "
                "--mem=64G -t 12:00:00 -J sckev-forced-rows-extra "
                "--wrap \"ARMS='delta-kev-4b-s2 delta-kev-4b-s3 delta-kev-9b-s2' bash "
                "$W/<cluster-job>\"",
                "# 3. pull each dump + its meta (size and sha256 verified on both sides), then aggregate:",
                "bash <cluster-job>",
                "source scripts/wsl/env.sh && uv run --no-sync python scripts/extra-seeds-score.py "
                f"--train-job-ids {args.train_job_ids or '...'} "
                f"--rescore-job-ids {args.rescore_job_ids or '...'}",
            ],
            "sha256": {},
        },
        "limits": [
            "The forced number is the argmax over the row's own probability vector with the abstain "
            "index dropped — the settled rule — not a model re-run under a no-abstain prompt.",
            "The population is the suite-level 1,819 in which 322 of the 1,497 census triples are "
            "counted twice, as in every other table on this measure; the census's 2,510-record "
            "population is not reported here.",
            "A seed's row is its re-scored dump alone: the 4B seeds 2 and 3 and the 9B seed 2 have no "
            "archived reports/rows-full-<arm>.jsonl, so the row-by-row fidelity check the eleven "
            "earlier arms have is not available for them. The internal checks in §7 are what stand in "
            "its place.",
            "The run facts (state, wall, steps, peak) are quoted from the cluster's own sacct records "
            "and the runs' provenance.json, not re-measured here.",
        ],
    }

    # checks
    checks = []
    for size_key, size in out["sizes"].items():
        for seed in size["seeds"]:
            cell = seed.get("census_matched_1819") or {}
            if not cell or "not_measured" in cell:
                checks.append(f"`{seed['arm']}`: NOT MEASURED — {cell.get('not_measured', 'no dump')}")
                continue
            c = cell["checks"]
            checks.append(f"`{seed['arm']}`: {c['rows_checked']} rows checked; rows without `p`: "
                          f"{c['no_p']['count']}; rows whose `top1` is not the argmax of their own "
                          f"vector: {c['top1_not_argmax_of_its_own_p']['count']}; rows whose "
                          f"`abstain` field disagrees with the abstain index: "
                          f"{c['abstain_index_disagreement']['count']}; served counted from `p` "
                          f"equals served counted from `top1`: "
                          f"{c['served_from_p_equals_served_from_top1']} — {c['verdict']}")
    for name, rung in RUNGS.items():
        reference, _ = rung_cache[name]
        checks.append(f"the {name} rung's forced count recomputed here from `{rung['dump']}` on this "
                      f"same {reference['n']}-row population: {reference['forced']['correct']}"
                      f"/{reference['n']} = {reference['forced']['accuracy']:.4f} — the settled value "
                      f"is {rung['settled_forced_correct']}/{reference['n']} = "
                      f"{rung['settled_forced_correct'] / reference['n']:.4f}; "
                      f"{'reproduced to the integer' if reference['forced']['correct'] == rung['settled_forced_correct'] else 'DISAGREES'}")
    out["checks"] = checks

    for rel in ([SUITE_IDS, RUNGS["kev-4b"]["dump"], RUNGS["kev-9b"]["dump"], SETTLED, ARMS_FORCED,
                 RULE_SOURCE.relative_to(ROOT).as_posix()]
                + [f"reports/forced-rows-{s['arm']}.jsonl" for size in SIZES.values() for s in size["seeds"]]
                + [args.run_facts]):
        if (ROOT / rel).exists():
            out["provenance"]["sha256"][rel] = sha256_file(rel)
    for rel in ("sckevtype/eval/run.py", "scripts/score-forced-rows.py",
                "<cluster-job>", "<cluster-job>",
                "<cluster-job>", "<cluster-job>"):
        if (ROOT / rel).exists():
            out["provenance"]["sha256"][rel] = sha256_file(rel)

    Path(args.json).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    write_md(out, Path(args.md))

    print(f"[population] n={len(ids)}  ({SUITE_IDS}, family {POPULATION_FAMILY})")
    for size_key, size in out["sizes"].items():
        print(f"[{size['label']}]  rung {size['rung']} forced "
              f"{size['rung_forced']['correct']}/{size['rung_forced']['n']} = "
              f"{size['rung_forced']['accuracy']:.4f}")
        for seed in size["seeds"]:
            cell = seed.get("census_matched_1819") or {}
            if not cell or "not_measured" in cell:
                print(f"  seed {seed['seed']}  NOT MEASURED -- {cell.get('not_measured', '')[:70]}")
                continue
            print(f"  seed {seed['seed']}  served {cell['served']['accuracy']:.4f}  "
                  f"abstain {cell['abstain_pick']['rate']:.4f}  forced {cell['forced']['accuracy']:.4f} "
                  f"({cell['forced']['correct']}/{cell['n']})  gain {cell['delta_points']:+.1f} pt  "
                  f"hidden {cell['hidden_correct_share_of_abstentions']:.4f}  "
                  f"gap {seed['gap_to_rung_points']:+.1f} pt  p={seed['paired_vs_rung']['p_two_sided_exact']:.3g}")
        sp = size["spread"]
        if sp.get("n_seeds", 0) > 1:
            print(f"  mean {sp['mean']:.4f}  range {sp['range'] * 100:.1f} pt  "
                  f"factor {sp['spread_factor']:.2f}x")
    def shown(p: str | Path) -> str:
        try:
            return str(Path(p).resolve().relative_to(ROOT))
        except ValueError:
            return str(p)

    print(f"wrote {shown(args.json)} and {shown(args.md)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
