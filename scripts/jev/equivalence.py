"""Does batching a cell's question families into one request change the model's answers?

The batched ask (`sckevtype.jev.probe ask-batched`) sends every family whose *state* agrees in one
request instead of one request per family. That saves requests only if it is also a no-op on the
answers, and the answer to that is not "yes" or "no" — **the endpoint is not deterministic**, so the
question is whether batching moves the answers *more than the model moves itself*.

That reframing is not a hedge; it was measured. Re-sending a byte-identical request (same
``request_sha256``, same served model ``jev-1.13.0``) moves individual probabilities by a couple of
points, so a single call is not a reproducible measurement and every comparison needs a same-window
null beside it. The arms, all sent by this one script in one window:

``single1``, ``single2``
    the same singleton request, sent twice. ``single2 - single1`` is the **null**: what a
    byte-identical request does on its own, with the clock held as fixed as it can be held.
``batched``
    one request per (cell, state group) — the arm under test. ``batched - single1`` is the
    **grouping** effect at the same instant, which is the comparison the claim is about.
``cached``
    the earlier smoke run's singleton answers. ``single1 - cached`` is the **drift** (same request,
    minutes apart), and ``batched - cached`` is what a reader comparing the two published runs would
    compute — reported so the reader does not have to.

Each pair is summarised per question: the option-keyed distribution (max/mean/p95 |Δp|, share
unchanged), the argmax, whether the abstention decision moved, the confidence, and — through
``sckevtype.jev.probe.rows_of_answer`` — whether the *row the axis-1 instrument scores* changed.

    uv run --no-sync python scripts/jev/equivalence.py --cached reports/jev-smoke \
        --out reports/jev-equivalence --cells 50 --budget-usd 4.00 --workers 6
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from sckevtype.contracts.freeze import read_json, write_json
from sckevtype.eval import abstain_key, metrics, rows_of
from sckevtype.eval.run import write_row_dump
from sckevtype.jev import probe as P
from sckevtype.jev.client import JevClient, default_transport

#: The pairs reported: ``(name, a_arm, b_arm)`` — b's answers are compared against a's.
PAIRS = (("null", "single1", "single2"), ("drift", "cached", "single1"),
         ("grouping", "single1", "batched"), ("grouping+drift", "cached", "batched"))

#: Which directory each arm's raw bodies are read back from.
DIRECTORIES = {"single1": "raw-single1", "single2": "raw-single2", "batched": "raw-batched",
               "cached": "raw"}


def answers_of(saved: dict[str, Any]) -> dict[str, Any]:
    """The vendor's answers out of a saved body — the one shape every arm is stored in."""
    return saved["response"]["answers"]


def probabilities_of(saved: dict[str, Any]) -> dict[str, Any]:
    """The scored distributions out of a saved body, keyed by the id that was sent."""
    return saved["prediction"]["probabilities"]


def served_models(directory: Path) -> dict[str, int]:
    """Every served model id in a directory of saved bodies, with how many responses carried it."""
    return dict(Counter(read_json(path)["response"]["model"] for path in sorted(directory.glob("*.json"))))


def per_question(a: dict[str, Any], b: dict[str, Any], prefix_a: str, prefix_b: str,
                 cell: str, family: str) -> list[dict[str, Any]]:
    """One question's two answers, as the numbers a summary is taken over. ``a`` is the reference.

    A question the other body does not carry raises (``KeyError``) rather than shrinking the
    denominator: a comparison that quietly drops its missing questions reports the questions that
    happened to match, which is the one failure mode this whole script exists to avoid.
    """
    out = []
    for qid, answer in answers_of(a).items():
        other = answers_of(b)[f"{prefix_b}{qid}"]
        keys = list(probabilities_of(a)[f"{prefix_a}{qid}"])
        ours, theirs = probabilities_of(a)[f"{prefix_a}{qid}"], probabilities_of(b)[f"{prefix_b}{qid}"]
        deltas = {key: abs(float(value) - float(theirs[key])) for key, value in ours.items()}
        abstain = abstain_key(keys)
        top = max(keys, key=lambda key: ours[key])
        other_top = max(keys, key=lambda key: theirs[key])
        out.append({
            "cell": cell, "family": family, "question": qid, "type": answer.get("type"),
            "size": len(keys), "max_abs_delta": max(deltas.values()),
            "mean_abs_delta": statistics.fmean(deltas.values()), "unchanged": max(deltas.values()) == 0.0,
            "over_one_point": max(deltas.values()) > 0.01, "argmax_same": top == other_top,
            "abstain_decision_same": (top == abstain) == (other_top == abstain),
            "abstain_p_delta": (None if abstain is None else abs(float(ours[abstain]) - float(theirs[abstain]))),
            "confidence_delta": (None if answer.get("type") != "choice" else
                                 abs(float(answer.get("confidence") or 0.0)
                                     - float(other.get("confidence") or 0.0))),
            "top_a": top, "top_b": other_top, "delta_p": float(ours[top]) - float(theirs[top]),
        })
    return out


def block(group: list[dict[str, Any]]) -> dict[str, Any]:
    """One population of questions, summarised. Empty populations are reported, not skipped."""
    if not group:
        return {"n": 0}
    maxes = [row["max_abs_delta"] for row in group]
    confidences = [row["confidence_delta"] for row in group if row["confidence_delta"] is not None]
    abstain_p = [row["abstain_p_delta"] for row in group if row["abstain_p_delta"] is not None]
    ordered = sorted(maxes)
    flips = [row for row in group if not row["argmax_same"]]
    return {
        "n": len(group), "unchanged": sum(row["unchanged"] for row in group),
        "unchanged_share": sum(row["unchanged"] for row in group) / len(group),
        "over_one_point": sum(row["over_one_point"] for row in group),
        "max_abs_delta": max(maxes), "mean_abs_delta": statistics.fmean(maxes),
        "median_abs_delta": statistics.median(maxes),
        "p95_abs_delta": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
        "argmax_flips": len(flips), "abstain_decision_flips": sum(not row["abstain_decision_same"] for row in group),
        "abstain_p_max_delta": max(abstain_p) if abstain_p else None,
        "confidence_max_delta": max(confidences) if confidences else None,
        "flips": [{"cell": row["cell"], "family": row["family"], "question": row["question"],
                   "from": row["top_a"], "to": row["top_b"], "max_abs_delta": row["max_abs_delta"]}
                  for row in flips[:10]],
    }


def summarise(deltas: list[dict[str, Any]]) -> dict[str, Any]:
    """A pair's per-question numbers, pooled and split by question type and by family."""
    return {"all": block(deltas),
            "by_type": {qtype: block([row for row in deltas if row["type"] == qtype])
                        for qtype in sorted({row["type"] for row in deltas})},
            "by_family": {family: block([row for row in deltas if row["family"] == family])
                          for family in sorted({row["family"] for row in deltas})}}


def axis1_block(rows: list[dict[str, Any]], fpr: float) -> dict[str, Any]:
    """The axis-1 primary statistic and the headline accuracy/ECE, so a moved number is visible."""
    choice = rows_of(rows, type="choice")
    if not choice:
        return {"n": 0}
    adequacy = metrics.panel_adequacy(choice, fpr)
    primary = adequacy.get("primary") or {}
    return {"choice_rows": len(choice), "auroc": primary.get("auroc"), "n": primary.get("n"),
            "n_positive": primary.get("n_positive"), "n_negative": primary.get("n_negative"),
            "selection": primary.get("selection"), "detection_rate": primary.get("detection_rate"),
            "accuracy": metrics.top1_accuracy(choice)["accuracy"], "ece": metrics.ece(choice)["ece"],
            "per_family": {name: {"n": entry["n"], "auroc": entry["auroc"]}
                           for name, entry in adequacy["per_family"].items()}}


def serve_a_cell(cell: str, entries: list[tuple[str, dict[str, Any]]], split: str, client: JevClient,
                 out: Path) -> tuple[dict[str, dict[tuple[str, str], dict[str, Any]]],
                                     dict[str, dict[str, dict[str, Any]]]]:
    """Every arm's bodies and rows for one cell — the unit of work, so cells can run in parallel.

    Two singleton asks of the same request (``single1``, ``single2``) and one batched ask. The two
    singleton directories are separate on purpose: the probe's cache reuses a saved body, so writing
    both sends into one directory would have the second ask answered from disk and the null would
    measure nothing.
    """
    dataset, name = P.dataset_of(cell), cell.split("/", 1)[1]
    arms: dict[str, dict[tuple[str, str], dict[str, Any]]] = {"single1": {}, "single2": {}, "batched": {}}
    rows_by_arm: dict[str, dict[str, dict[str, Any]]] = {"single1": {}, "single2": {}, "batched": {}}
    for arm in ("single1", "single2"):
        for family, record in entries:
            rows, _prediction = P.build_rows(record, family, split, client, out / DIRECTORIES[arm], dataset, name)
            rows_by_arm[arm].update({row["id"]: row for row in rows})
            arms[arm][(cell, family)] = read_json(P.ask_path(out / DIRECTORIES[arm], dataset, name, family, "base"))
    rows, meta = P.build_batched_cell(entries, split, client, out / DIRECTORIES["batched"], dataset, name)
    rows_by_arm["batched"].update({row["id"]: row for row in rows})
    for index, served in enumerate(meta["served"]):
        saved = read_json(P.batch_path(out / DIRECTORIES["batched"], dataset, name, index))
        for family in served["families"]:
            arms["batched"][(cell, family)] = saved
    return arms, rows_by_arm


def paired_block(rows_by_pair: dict[str, dict[tuple[str, str, str], dict[str, Any]]],
                 a: str, b: str) -> dict[str, Any]:
    """Two pairs on the *same questions*, paired — the comparison the verdict turns on.

    A per-question ``max |Δp|`` under the null (a byte-identical resend) and under the grouping, so
    the grouping effect is read against the model's own variation on the same question rather than
    against a different population of questions. The bootstrap is the interval for the mean
    difference; the sign test is the count of questions where the grouping moved more than the
    resend did, which needs no distributional assumption beyond exchangeability under the null.
    """
    shared = sorted(set(rows_by_pair[a]) & set(rows_by_pair[b]))
    if not shared:
        return {"n": 0}
    import numpy as np

    left = np.array([rows_by_pair[a][key]["max_abs_delta"] for key in shared])
    right = np.array([rows_by_pair[b][key]["max_abs_delta"] for key in shared])
    difference = right - left
    rng = np.random.default_rng(20260930)
    draws = rng.choice(difference, size=(4000, len(difference)), replace=True).mean(axis=1)
    larger = int((difference > 0).sum())

    def discordant(field: str) -> dict[str, Any]:
        """Both arms' flips on the same question: (only a, only b, both) — a McNemar table."""
        a_only = sum(rows_by_pair[a][key][field] and not rows_by_pair[b][key][field] for key in shared)
        b_only = sum(rows_by_pair[b][key][field] and not rows_by_pair[a][key][field] for key in shared)
        both = sum(rows_by_pair[a][key][field] and rows_by_pair[b][key][field] for key in shared)
        return {"flips_only_null": a_only, "flips_only_grouping": b_only, "flips_both": both,
                "mcnemar_exact_p_two_sided": _mcnemar(a_only, b_only)}

    return {
        "n": len(shared),
        "mean_abs_delta_null": float(left.mean()), "mean_abs_delta_grouping": float(right.mean()),
        "mean_difference": float(difference.mean()),
        "median_difference": float(np.median(difference)),
        "bootstrap_ci95_mean_difference": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
        "share_grouping_larger": larger / len(shared),
        "sign_test_p_two_sided": _sign_test(larger, len(shared)),
        "argmax_flips": discordant("argmax_same"),
        "abstain_flips": discordant("abstain_decision_same"),
        "argmax_flips_null": int(sum(not rows_by_pair[a][key]["argmax_same"] for key in shared)),
        "argmax_flips_grouping": int(sum(not rows_by_pair[b][key]["argmax_same"] for key in shared)),
        "abstain_flips_null": int(sum(not rows_by_pair[a][key]["abstain_decision_same"] for key in shared)),
        "abstain_flips_grouping": int(sum(not rows_by_pair[b][key]["abstain_decision_same"] for key in shared)),
    }


def _mcnemar(a_only: int, b_only: int) -> float:
    """Exact McNemar: the paired flips disagree, so the question is whether one arm flips more.

    Only the questions where exactly one arm flipped carry the information — the ones both arms
    flipped (or neither) are concordant and drop out, which is why the counts above are printed
    beside the p-value rather than folded into it.
    """
    from math import comb

    discordant = a_only + b_only
    if discordant == 0:
        return 1.0
    tail = sum(comb(discordant, k) for k in range(min(a_only, b_only) + 1)) / 2 ** discordant
    return float(min(1.0, 2 * tail))


def _sign_test(larger: int, n: int) -> float:
    """Two-sided exact sign-test p-value, from the binomial likelihoods (no scipy needed)."""
    from math import comb

    if n == 0:
        return 1.0
    tail = sum(comb(n, k) for k in range(min(larger, n - larger) + 1)) / 2 ** n
    return float(min(1.0, 2 * tail))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cached", default="reports/jev-smoke", help="the singleton smoke run to compare against")
    parser.add_argument("--out", default="reports/jev-equivalence")
    parser.add_argument("--cells", type=int, default=50, help="cells to re-ask in every arm (0 = the whole sample)")
    parser.add_argument("--budget-usd", type=float, default=4.0)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    cached_dir, out = Path(args.cached), Path(args.out)
    sample = read_json(cached_dir / "sample.json")
    partition = Path(sample["corpus"])
    if not partition.exists():
        raise SystemExit(f"the sample's corpus {partition} is not here; this comparison is on its records")
    cells = P.cells_of(P.collect_records(partition, sample))
    if args.cells:
        cells = cells[: args.cells]
    requests = sum(len(P.state_groups([family for family, _record in entries], dict(entries)))
                   for _group, entries in cells)
    print(f"{len(cells)} cells: {requests} batched requests and "
          f"{sum(len(entries) for _group, entries in cells)} singleton asks, sent twice; "
          f"budget ${args.budget_usd}", flush=True)

    client = JevClient(transport=default_transport, budget_usd=args.budget_usd,
                       max_calls=max(200, 16 * len(cells)), max_call_usd=0.10)
    arms: dict[str, dict[tuple[str, str], dict[str, Any]]] = {"single1": {}, "single2": {}, "batched": {}}
    rows_by_arm: dict[str, dict[str, dict[str, Any]]] = {"single1": {}, "single2": {}, "batched": {}}
    results: list[Any] = [None] * len(cells)
    if args.workers > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(serve_a_cell, group, entries, sample["split"], client, out): index
                       for index, (group, entries) in enumerate(cells)}
            for done, future in enumerate(as_completed(futures), start=1):
                results[futures[future]] = future.result()
                if done % 5 == 0 or done == len(cells):
                    print(f"  [{done:4d}/{len(cells)}] calls={client.calls} "
                          f"in_tokens={client.input_tokens} ${client.estimated_usd:.4f}", flush=True)
    else:
        for index, (group, entries) in enumerate(cells, start=1):
            results[index - 1] = serve_a_cell(group, entries, sample["split"], client, out)
            if index % 5 == 0 or index == len(cells):
                print(f"  [{index:4d}/{len(cells)}] calls={client.calls} "
                      f"in_tokens={client.input_tokens} ${client.estimated_usd:.4f}", flush=True)
    for cell_arms, cell_rows in results:
        for arm, bodies in cell_arms.items():
            arms[arm].update(bodies)
        for arm, rows in cell_rows.items():
            rows_by_arm[arm].update(rows)
    spent = {"calls": client.calls, "input_tokens": client.input_tokens,
             "output_tokens": client.output_tokens, "estimated_usd": client.estimated_usd,
             "median_latency_ms": client.accounting()["median_latency_ms"],
             "budget_usd": args.budget_usd, "cells": len(cells)}
    print(f"spent {spent['calls']} calls, {spent['input_tokens']} input tokens, ${spent['estimated_usd']:.4f}",
          flush=True)

    report: dict[str, Any] = {"pairs": {}, "arms": {}, "rows": {}, "axis1": {}, "spent": spent}
    arms["cached"] = {}
    for group, entries in cells:
        dataset, cell = P.dataset_of(group), group.split("/", 1)[1]
        for family, _record in entries:
            path = P.ask_path(cached_dir / DIRECTORIES["cached"], dataset, cell, family, "base")
            if path.exists():
                arms["cached"][(group, family)] = read_json(path)
    # A hole in an arm is a hole in the comparison, not a row to drop: the cached arm is read through
    # the probe's own path builder (a cell id with a "/" is slugged on disk, so re-deriving the name
    # here would silently miss exactly the multi-level mouse ids) and the count is stated.
    expected = sum(len(entries) for _group, entries in cells)
    report["arm_coverage"] = {"cell_family_bodies_expected_per_arm": expected,
                              "found": {name: len(arm) for name, arm in arms.items()}}

    for name, arm in arms.items():
        directory = cached_dir / DIRECTORIES[name] if name == "cached" else out / DIRECTORIES[name]
        report["arms"][name] = {"cell_family_bodies": len(arm), "directory": str(directory),
                                "served_models": served_models(directory),
                                "distinct_request_digests": len({saved["request_sha256"] for saved in arm.values()})}
    question_rows: dict[str, dict[tuple[str, str, str], dict[str, Any]]] = {}
    for name, a_arm, b_arm in PAIRS:
        deltas = []
        keyed = question_rows.setdefault(name, {})
        for (cell, family), saved in sorted(arms[a_arm].items()):
            if (cell, family) not in arms[b_arm]:
                continue
            for row in per_question(saved, arms[b_arm][(cell, family)],
                                    "" if a_arm != "batched" else f"{family}/",
                                    "" if b_arm != "batched" else f"{family}/", cell, family):
                deltas.append(row)
                keyed[(cell, family, row["question"])] = row
        report["pairs"][name] = {"a": a_arm, "b": b_arm, "cells_compared": len({row["cell"] for row in deltas}),
                                 "cell_family_pairs_compared": len({(row["cell"], row["family"]) for row in deltas}),
                                 "summary": summarise(deltas)}
        summary = report["pairs"][name]["summary"]["all"]
        print(f"{name:15s} {b_arm:8s} vs {a_arm:8s} n={summary['n']:5d} "
              f"unchanged {summary['unchanged_share']:.3f} max|dp| {summary['max_abs_delta']:.4f} "
              f"mean|dp| {summary['mean_abs_delta']:.5f} argmax flips {summary['argmax_flips']} "
              f"abstain flips {summary['abstain_decision_flips']}", flush=True)

    cached_rows = [json.loads(line) for line in (cached_dir / "rows.jsonl").read_text(encoding="utf-8").splitlines()
                   if line.strip()]
    chosen = {group for group, _entries in cells}
    same_cells = [row for row in cached_rows if row["group"] in chosen]
    for name in ("single1", "single2", "batched"):
        rows = rows_by_arm[name]
        kept = [row for row in same_cells if row["id"] in rows]
        differing = [row["id"] for row in kept if rows[row["id"]] != row]
        report["rows"][name] = {
            "cached_rows_on_these_cells": len(same_cells), "rows_compared": len(kept),
            "rows_differing": len(differing), "rows_differing_share": len(differing) / max(1, len(kept)),
            "first_differing": differing[:5],
            "labels_and_keys_identical": all(rows[row["id"]]["label"] == row["label"]
                                             and rows[row["id"]]["keys"] == row["keys"] for row in kept),
        }
    report["axis1"]["cached"] = axis1_block(same_cells, P.FPR)
    for name in ("single1", "batched"):
        report["axis1"][name] = axis1_block([row for row in rows_by_arm[name].values() if row["group"] in chosen],
                                            P.FPR)
    dump = write_row_dump(out / "axis1-rows-batched.jsonl",
                          [row for row in rows_by_arm["batched"].values() if row["type"] == "choice"])
    report["rows"]["axis1_dump"] = dump

    null, grouping = report["pairs"]["null"]["summary"]["all"], report["pairs"]["grouping"]["summary"]["all"]
    report["paired"] = {"grouping_vs_null": paired_block(question_rows, "null", "grouping"),
                        "drift_vs_null": paired_block(question_rows, "null", "drift")}
    report["verdict"] = {
        "question": "does batching move the answers more than the model moves itself?",
        "null": {"max_abs_delta": null["max_abs_delta"], "mean_abs_delta": null["mean_abs_delta"],
                 "unchanged_share": null["unchanged_share"], "argmax_flips": null["argmax_flips"],
                 "abstain_decision_flips": null["abstain_decision_flips"]},
        "grouping": {"max_abs_delta": grouping["max_abs_delta"], "mean_abs_delta": grouping["mean_abs_delta"],
                     "unchanged_share": grouping["unchanged_share"], "argmax_flips": grouping["argmax_flips"],
                     "abstain_decision_flips": grouping["abstain_decision_flips"]},
        "grouping_max_within_null": grouping["max_abs_delta"] <= null["max_abs_delta"],
        "grouping_mean_within_null": grouping["mean_abs_delta"] <= null["mean_abs_delta"],
        "grouping_flips_within_null": grouping["argmax_flips"] <= null["argmax_flips"],
        "paired_bootstrap_ci95_mean_difference": report["paired"]["grouping_vs_null"]["bootstrap_ci95_mean_difference"],
        "paired_mcnemar_argmax": report["paired"]["grouping_vs_null"]["argmax_flips"],
        "paired_mcnemar_abstain": report["paired"]["grouping_vs_null"]["abstain_flips"],
        "note": ("the endpoint is not deterministic: a byte-identical request re-sent in the same window "
                 "moves probabilities by up to the null's max |dp| and flips some argmaxes, so 'batching "
                 "changed nothing' is not a claim one call can support — 'changed no more than a "
                 "byte-identical resend' is, and the paired rows above are that comparison. Read the "
                 "effect size and the bootstrap CI; the sign test's p-value is not a significance claim "
                 "at n in the thousands, where a per-row drift of ~1e-4 is 'significant' while the CI "
                 "still contains zero"),
        "no_seed_or_temperature": ("the request body is exactly {state, model, questions} — the vendor's "
                                   "documented contract, and sckevtype.jev.request.check_request refuses "
                                   "any other top-level key — so there is no seed or temperature knob to "
                                   "fix, and every Jev number is one draw from a nondeterministic readout"),
    }
    write_json(out / "report.json", report)
    write_json(out / "answer-equivalence.json",
               {"arms": report["arms"], "pairs": {name: entry["summary"] for name, entry in report["pairs"].items()},
                "rows": report["rows"], "axis1": report["axis1"], "verdict": report["verdict"], "spent": spent})
    print("\nVERDICT")
    for section in ("null", "grouping"):
        print(f"  {section:9s} {report['verdict'][section]}")
    for key in ("grouping_max_within_null", "grouping_mean_within_null", "grouping_flips_within_null"):
        print(f"  {key}: {report['verdict'][key]}")
    for name, entry in report["paired"].items():
        if not entry.get("n"):
            continue
        print(f"  paired {name}: n={entry['n']} null {entry['mean_abs_delta_null']:.5f} "
              f"other {entry['mean_abs_delta_grouping']:.5f} diff {entry['mean_difference']:+.5f} "
              f"CI95 [{entry['bootstrap_ci95_mean_difference'][0]:+.5f}, "
              f"{entry['bootstrap_ci95_mean_difference'][1]:+.5f}] sign p={entry['sign_test_p_two_sided']:.3g}")
        for field in ("argmax_flips", "abstain_flips"):
            table = entry[field]
            print(f"    {field}: only-null {table['flips_only_null']} only-grouping "
                  f"{table['flips_only_grouping']} both {table['flips_both']} "
                  f"McNemar p={table['mcnemar_exact_p_two_sided']:.3g}")
    for name in ("cached", "single1", "batched"):
        entry = report["axis1"][name]
        auroc = "n/a" if entry.get("auroc") is None else f"{entry['auroc']:.4f}"
        accuracy = entry.get("accuracy")
        print(f"  axis 1 {name:8s} auroc {auroc} n={entry.get('n')} pos={entry.get('n_positive')} "
              f"accuracy {'n/a' if accuracy is None else f'{accuracy:.4f}'}")
    print(f"wrote {out / 'report.json'} and {out / 'answer-equivalence.json'}; "
          f"dump {dump['rows']} rows sha256 {dump['sha256'][:16]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
