#!/usr/bin/env python3
"""Confusion structure of annotation errors: what an engine names when it is wrong.

The question here is not "is the engine unsure?" but "does it name the right cell type?" --
and when it does not, whether the name it reached for is a *coarser* type (a Cell-Ontology
ancestor of the truth: the engine reported the most specific distinction the evidence
supported), a *finer* one, a sibling, a cousin, or something unrelated. Errors that are
systematically coarsenings are a loss of resolution, not a hallucination, and the difference
is the point of the analysis.

Streams the per-question row dumps the engines already wrote (``p`` over the row's ``keys``,
``label`` the index of the correct option), the frozen corpus (each cell's true CL term), the
v3 ontology (``parents`` -> transitive ancestors) and the 82-term register (option name ->
CL id). Stdlib only, deterministic, no network, no GPU.

Usage::

    uv run --no-sync python scripts/confusion-structure.py \\
      --rows reports/jev-full/rows.jsonl --engine jev \\
      --corpus runs/real-v2/test.jsonl \\
      --cl-terms artifacts/v3/cl_terms.json --definitions artifacts/v3/definitions.json \\
      --out reports/confusion-jev.json --md reports/confusion-jev.md

and, to regenerate the cross-engine summary from finished JSONs::

    uv run --no-sync python scripts/confusion-structure.py \\
      --summary reports/confusion-jev.json reports/confusion-kev-05b.json ... \\
      --out reports/confusion-summary.md
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
from collections import Counter

# The six classes partition the errors whose truth and prediction are both named answers.
TAXONOMY = ("coarsening", "narrowing", "sibling", "cousin", "unrelated", "unmappable")
# Families a row dump can carry. ``marker_noul`` rows are ``type == "noul"`` (yes/no marker
# questions, not type choices) and are dropped by the type filter -- the family is still
# listed so its emptiness is visible rather than implied.
FAMILIES = (
    "panel_choice",
    "panel_miss",
    "evidence_diluted",
    "evidence_removed",
    "out_of_register",
    "marker_noul",
)
TOP_PAIRS = 20
TOLERANCE = 1e-6

HOW_TO_READ = [
    ("This measures the **named** answers. A row whose top-1 is the abstain option is counted "
     "as an abstention, not as an error, and never enters the taxonomy."),
    ("The taxonomy is over the errors where the truth and the prediction are both real types: "
     "coarsening / narrowing / sibling / cousin / unrelated when both sides resolve in the CL "
     "ontology, `unmappable` when either side does not. The six classes partition those errors "
     "(a pair that resolves to one CL id under two names is counted under `same_id` instead)."),
    ("Direction is pinned and is the whole point: prediction = *ancestor* of truth is a "
     "**coarsening** (a broader name); prediction = *descendant* of truth is a **narrowing** "
     "(a finer name). A sibling shares a parent -- a sibling is *not* the parent, so "
     "neutrophil named as granulocyte is a coarsening."),
    ("The corpus is a corpus of *cells*: one cell contributes several rows (across families "
     "and questions). `n` counts rows; the distinct-cell count is in the population block."),
    ("Starved-arm rows (the truth is the abstain option: panel_miss, and any other row whose "
     "labelled answer is abstain) are reported separately. What the engine named *instead of* "
     "abstaining is a fallback, classified against the cell's own true type from the corpus."),
    "Nothing here is a claim about another engine: each report is one engine's rows.",
]


# --------------------------------------------------------------------------------------
# Ontology and register
# --------------------------------------------------------------------------------------


class Ontology:
    """``cl_terms.json`` as a parent map, with memoised ancestor sets.

    ``parents`` is a DAG, not a tree (CL terms are multiply parented), so ancestry is the
    transitive closure over every parent edge, and the deepest common ancestor is the common
    ancestor with the largest ancestor set -- not "walk up until the two meet".
    """

    ROOT = "CL:0000000"

    def __init__(self, terms: list[dict]):
        self.parents = {t["cl_id"]: tuple(t.get("parents") or ()) for t in terms}
        self.names: dict[str, str] = {}
        self.by_id: dict[str, str] = {}
        for t in terms:
            self.names.setdefault(t["name"], t["cl_id"])
            self.by_id.setdefault(t["cl_id"], t["name"])
        self._ancestors: dict[str, frozenset[str]] = {self.ROOT: frozenset()}

    def ancestors(self, cl_id: str) -> frozenset[str]:
        """Every strict ancestor of ``cl_id`` (the node itself excluded)."""
        memo = self._ancestors.get(cl_id)
        if memo is not None:
            return memo
        out: set[str] = set()
        stack = list(self.parents.get(cl_id, ()))
        while stack:
            node = stack.pop()
            if node in out:
                continue
            out.add(node)
            stack.extend(self.parents.get(node, ()))
        frozen = frozenset(out)
        self._ancestors[cl_id] = frozen
        return frozen

    def lca(self, a: str, b: str) -> str | None:
        """Deepest common ancestor of two terms -- ``None`` when they share none at all."""
        common = (self.ancestors(a) | {a}) & (self.ancestors(b) | {b})
        if not common:
            return None
        return max(common, key=lambda c: (len(self.ancestors(c)), c))

    def name_of(self, cl_id: str) -> str:
        return self.by_id.get(cl_id, cl_id)


class Register:
    """Option key (a register name) -> CL id.

    Resolution order: the register's names, the ontology's names, then the key taken as a CL
    id. A key that resolves in none of the three is ``unmappable`` -- counted and reported,
    never silently dropped.
    """

    def __init__(self, ontology: Ontology, definitions: list[dict]):
        self.ontology = ontology
        self.by_name = {d["name"]: d["cl_id"] for d in definitions}

    def resolve(self, key: str) -> str | None:
        for cid in (self.by_name.get(key), self.ontology.names.get(key)):
            if cid is not None:
                return cid
        return key if key in self.ontology.parents else None

    def mappable(self, cl_id: str | None) -> bool:
        return cl_id is not None and cl_id in self.ontology.parents


def classify(
    pred: str | None, truth: str | None, ontology: Ontology, register: Register
) -> tuple[str, str | None]:
    """Classify one (prediction, truth) CL-id pair. Returns ``(class, lca_or_None)``.

    ``coarsening`` means the *prediction* is an ancestor of the truth -- the engine named a
    broader type, e.g. truth ``medium spiny neuron`` named ``neuron``. ``narrowing`` is the
    mirror. The order is deliberate: an ancestor/descendant relation is never a sibling, so
    the two set tests come before the parent-equality test.
    """
    if not register.mappable(pred) or not register.mappable(truth):
        return "unmappable", None
    if pred == truth:
        return "same", None
    pred_anc, truth_anc = ontology.ancestors(pred), ontology.ancestors(truth)
    if pred in truth_anc:
        return "coarsening", None
    if truth in pred_anc:
        return "narrowing", None
    pred_parents, truth_parents = set(ontology.parents[pred]), set(ontology.parents[truth])
    if pred_parents and pred_parents == truth_parents:
        return "sibling", None
    lca = ontology.lca(pred, truth)
    if lca is None or lca == Ontology.ROOT:
        return "unrelated", None
    return "cousin", lca


# --------------------------------------------------------------------------------------
# Accumulation
# --------------------------------------------------------------------------------------


class FamilyStats:
    """Counters for one population (a family, a dataset, a starved-arm fallback)."""

    def __init__(self) -> None:
        self.n = 0
        self.abstained = 0
        self.answered = 0
        self.truth_abstain = 0
        self.correct = 0
        self.no_abstain_option = 0
        self.errors: Counter[str] = Counter()
        self.same_id = 0
        self.pairs: Counter[tuple[str, str, str]] = Counter()
        self.pair_lca: dict[tuple[str, str], str] = {}

    def add_error(self, truth_key: str, pred_key: str, klass: str, lca: str | None) -> None:
        if klass == "same":
            self.same_id += 1
            return
        self.errors[klass] += 1
        self.pairs[(truth_key, pred_key, klass)] += 1
        if lca is not None:
            self.pair_lca[(truth_key, pred_key)] = lca

    @property
    def n_errors(self) -> int:
        return sum(self.errors.values()) + self.same_id

    def as_json(self, ontology: Ontology) -> dict:
        errors = self.n_errors

        def share(x: int) -> float:
            return round(x / errors, 6) if errors else 0.0

        top = []
        for (truth, pred, klass), count in sorted(
            self.pairs.items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1])
        )[:TOP_PAIRS]:
            entry = {"truth": truth, "predicted": pred, "count": count, "class": klass}
            lca = self.pair_lca.get((truth, pred))
            if lca is not None:
                entry["lca"] = lca
                entry["lca_name"] = ontology.name_of(lca)
            top.append(entry)
        return {
            "n": self.n,
            "abstained": self.abstained,
            "abstained_share": round(self.abstained / self.n, 6) if self.n else 0.0,
            "answered": self.answered,
            "answered_share": round(self.answered / self.n, 6) if self.n else 0.0,
            "truth_is_abstain": self.truth_abstain,
            "correct": self.correct,
            "accuracy": round(self.correct / self.n, 6) if self.n else 0.0,
            "errors": errors,
            "same_id": self.same_id,
            "taxonomy": {k: {"n": self.errors.get(k, 0), "share": share(self.errors.get(k, 0))}
                         for k in TAXONOMY},
            "top_pairs": top,
        }


def analyse(rows, ctx: dict) -> dict:
    """One streaming pass over the kept rows. Returns the report body (no provenance)."""
    ontology: Ontology = ctx["ontology"]
    register: Register = ctx["register"]
    corpus: dict[tuple[str, str], str] = ctx["corpus"]
    pred_field: Counter[str] = ctx["pred_field"]
    extras: Counter[str] = ctx["extras"]

    # Every known family is present even at n=0, so a family whose rows were all dropped by
    # the type filter (marker_noul) shows up as empty rather than silently missing.
    families: dict[str, FamilyStats] = {f: FamilyStats() for f in FAMILIES}
    by_dataset: dict[str, FamilyStats] = {}
    fallback: dict[str, FamilyStats] = {}
    cells: set[tuple[str, str]] = set()
    datasets: Counter[str] = Counter()

    for row in rows:
        stats = families.setdefault(row.get("family"), FamilyStats())
        ds_stats = by_dataset.setdefault(row.get("dataset"), FamilyStats())
        for s in (stats, ds_stats):
            s.n += 1
        cells.add((row.get("dataset"), row.get("cell")))
        datasets[row.get("dataset")] += 1

        keys = row["keys"]
        key_abstain = ctx["abstain_key"](keys)
        if key_abstain is None:
            extras["rows_without_abstain_option"] += 1
        if "p" in row:
            probs = row["p"]
            pred_idx = max(range(len(probs)), key=lambda i: probs[i])
            pred_field["p"] += 1
        else:
            pred_idx = row["top1"]
            pred_field["top1"] += 1
        truth_key, pred_key = keys[row["label"]], keys[pred_idx]
        truth_is_abstain = key_abstain is not None and truth_key == key_abstain
        abstained = key_abstain is not None and pred_key == key_abstain

        for s in (stats, ds_stats):
            s.truth_abstain += truth_is_abstain
            s.abstained += abstained
            s.answered += not abstained
            s.correct += pred_idx == row["label"]

        if truth_is_abstain and not abstained:
            # The starved arm: the truth was not on the panel (or not decidable), and the
            # engine named a type anyway. A fallback, against the cell's own true type.
            truth_cl = corpus.get((row.get("dataset"), row.get("cell")))
            if truth_cl is None:
                extras["fallback_cells_not_in_corpus"] += 1
            klass, lca = classify(register.resolve(pred_key), truth_cl, ontology, register)
            # Labelled with the cell's own true type, not the row's abstain key: every row
            # here has truth_key == some abstain wording, which says nothing about the cell.
            truth_label = ontology.name_of(truth_cl) if truth_cl else "(cell not in corpus)"
            fallback.setdefault(row.get("family"), FamilyStats()).add_error(
                truth_label, pred_key, klass, lca
            )
        elif not abstained and pred_key != truth_key:
            truth_cl = register.resolve(truth_key)
            pred_cl = register.resolve(pred_key)
            extras["truth_unresolved"] += not register.mappable(truth_cl)
            extras["pred_unresolved"] += not register.mappable(pred_cl)
            klass, lca = classify(pred_cl, truth_cl, ontology, register)
            for s in (stats, ds_stats):
                s.add_error(truth_key, pred_key, klass, lca)

    return {
        "families": families,
        "by_dataset": by_dataset,
        "fallback": fallback,
        "cells": cells,
        "datasets": datasets,
    }


# --------------------------------------------------------------------------------------
# Reading rows
# --------------------------------------------------------------------------------------


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def digest_of(path: str, pattern: str) -> tuple[str, int]:
    """One digest for the input plus the number of files it covers (a directory is a set)."""
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, pattern)))
        lines = [f"{os.path.relpath(f, path)}\t{sha256_file(f)}" for f in files]
        return hashlib.sha256("\n".join(lines).encode()).hexdigest(), len(files)
    return sha256_file(path), 1


def row_paths(path: str, pattern: str) -> list[str]:
    return sorted(glob.glob(os.path.join(path, pattern))) if os.path.isdir(path) else [path]


def read_ids(path: str) -> tuple[set[str], set[tuple[str, str, str]]]:
    """The rows named by a dump -- the population filter behind ``--ids-from``.

    **Two keys, because two dumps of the same rows do not have to spell a row the same way.** The
    census dump names a row ``test/<dataset>/<cell>/<family>``, while ``sckevtype.eval.run``'s
    ``--rows-full`` dump names the very same row ``test/custom/<n>``. Both files carry ``id`` plus
    ``dataset``/``cell``/``family``, so a row is kept when either key matches; matching on ``id``
    alone kept 0 of 24,945 rows and the run refused, which reads as "the filter is broken" rather than
    "the two files spell the row differently".

    A line without an ``id`` is refused rather than skipped: silently filtering on a subset of a file
    is how a filter ends up smaller than its reader believes.
    """
    ids: set[str] = set()
    triples: set[tuple[str, str, str]] = set()
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{lineno}: not JSON: {exc}") from exc
            if not isinstance(row, dict) or "id" not in row:
                raise SystemExit(f"{path}:{lineno}: no 'id' field; refusing to filter on ids this "
                                 "file does not carry")
            ids.add(row["id"])
            triple = (row.get("dataset"), row.get("cell"), row.get("family"))
            if all(isinstance(part, str) and part for part in triple):
                triples.add(triple)
    if not ids:
        raise SystemExit(f"{path}: no ids read -- refusing an empty filter")
    return ids, triples


def read_rows(paths: list[str], filters: dict[str, str], split: str, dropped: Counter,
              ids: set[str] | None = None):
    """Yield kept rows; count what each filter drops, in filter order."""
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SystemExit(f"{path}:{lineno}: not JSON: {exc}") from exc
                dropped["read"] += 1
                if split != "any" and row.get("split") != split:
                    dropped["split"] += 1
                    continue
                if any(str(row.get(k)) != v for k, v in filters.items()):
                    dropped["row_filter"] += 1
                    continue
                row_type = row.get("type")
                if row_type is None:
                    # A choice-only dump (`sckevtype.eval.run --rows-full`) omits `type`: it writes
                    # choice rows and nothing else. Infer it from the row's own shape instead of
                    # dropping the entire population as "not choice" -- which is the failure that
                    # reads as "no rows survived" on a 27,079-row file.
                    row_type = "choice" if ("keys" in row and "label" in row) else "?"
                if row_type != "choice":
                    dropped["type_not_choice"] += 1
                    continue
                if ids is not None:
                    id_set, triples = ids
                    triple = (row.get("dataset"), row.get("cell"), row.get("family"))
                    if row.get("id") not in id_set and triple not in triples:
                        dropped["ids_from_dropped"] += 1
                        continue
                if "p" not in row and "top1" not in row:
                    raise SystemExit(
                        f"{path}:{lineno}: row carries neither 'p' nor 'top1'; refusing to "
                        "score a partial population silently"
                    )
                dropped["kept"] += 1
                yield row


# --------------------------------------------------------------------------------------
# The pipeline
# --------------------------------------------------------------------------------------


def build_ctx(args) -> dict:
    from sckevtype.eval import abstain_key

    with open(args.cl_terms, encoding="utf-8") as f:
        ontology = Ontology(json.load(f)["terms"])
    with open(args.definitions, encoding="utf-8") as f:
        register = Register(ontology, json.load(f)["definitions"])
    corpus = {}
    with open(args.corpus, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                meta = json.loads(line)["_meta"]
                corpus[(meta["dataset_id"], meta["cell_id"])] = meta["cl_id"]
    return {
        "ontology": ontology,
        "register": register,
        "corpus": corpus,
        "abstain_key": abstain_key,
        "pred_field": Counter(),
        "extras": Counter(),
    }


def build_report(args) -> tuple[dict, list[str]]:
    """Run the analysis; return the JSON-able report and the cross-check lines to print."""
    ctx = build_ctx(args)
    paths = row_paths(args.rows, args.glob)
    if not paths:
        raise SystemExit(f"no row files under {args.rows!r} matching {args.glob!r}")
    dropped: Counter[str] = Counter()
    ids = read_ids(args.ids_from) if args.ids_from else None
    rows = read_rows(paths, dict(kv.split("=", 1) for kv in args.row_filter), args.split, dropped, ids)
    body = analyse(rows, ctx)
    if not dropped["kept"]:
        # An empty intersection is how a "result" gets computed on nothing: say which filter
        # emptied it instead of printing a report over zero rows.
        extra = ""
        if ids is not None:
            extra = (f"; --ids-from {args.ids_from} kept no scored row "
                     f"({dropped['ids_from_dropped']:,} rows carried neither its id nor its "
                     f"(dataset, cell, family))")
        raise SystemExit(
            f"no rows survived the filters under {args.rows!r} (read {dropped['read']:,}: "
            f"split={dropped['split']:,}, row_filter={dropped['row_filter']:,}, "
            f"type_not_choice={dropped['type_not_choice']:,}){extra} -- refusing to write an empty report"
        )

    ontology: Ontology = ctx["ontology"]
    digest, n_files = digest_of(args.rows, args.glob)
    report = {
        "tool": {"name": "confusion-structure", "version": 1, "engine": args.engine},
        "provenance": {
            "rows": args.rows,
            "rows_glob": args.glob,
            "rows_files": n_files,
            "rows_sha256": digest,
            "engine": args.engine,
            "name": args.name or args.engine,
            "row_filter": args.row_filter,
            "split": args.split,
            "ids_from": args.ids_from,
            "ids_from_sha256": sha256_file(args.ids_from) if args.ids_from else None,
            "ids_from_names": len(ids[0]) if ids is not None else None,
            "ids_from_kept": dropped["kept"] if ids is not None else None,
            "ids_from_dropped": dropped["ids_from_dropped"] if ids is not None else None,
            "prediction_field": dict(ctx["pred_field"]),
            "corpus": args.corpus,
            "corpus_sha256": sha256_file(args.corpus),
            "cl_terms": args.cl_terms,
            "cl_terms_sha256": sha256_file(args.cl_terms),
            "definitions": args.definitions,
            "definitions_sha256": sha256_file(args.definitions),
            "counts": dict(dropped),
            "rows_without_abstain_option": ctx["extras"]["rows_without_abstain_option"],
            "truth_unresolved": ctx["extras"]["truth_unresolved"],
            "pred_unresolved": ctx["extras"]["pred_unresolved"],
            "fallback_cells_not_in_corpus": ctx["extras"]["fallback_cells_not_in_corpus"],
        },
        "population": {
            "rows": dropped["kept"],
            "distinct_cells": len(body["cells"]),
            "families": {f: s.n for f, s in sorted(body["families"].items()) if s.n},
            "datasets": dict(sorted(body["datasets"].items())),
        },
        "families": {f: s.as_json(ontology) for f, s in sorted(body["families"].items())},
        "starved_fallback": _fallback_json(body["fallback"], ontology),
        "by_dataset": {d: s.as_json(ontology) for d, s in sorted(body["by_dataset"].items())},
        "pooled": _pooled(body["families"]).as_json(ontology),
    }
    lines = crosscheck(report, args.check_report) if args.check_report else []
    report["crosscheck"] = {
        "report": args.check_report,
        "lines": lines,
        "pass": all(not line.startswith("DIFFERS") for line in lines),
    }
    return report, lines


def _fallback_json(fallback: dict[str, FamilyStats], ontology: Ontology) -> dict:
    pooled = _pooled(fallback).as_json(ontology)
    top = [
        {"named": p["predicted"], "true_type": p["truth"], "count": p["count"],
         "class": p["class"], **({"lca_name": p["lca_name"]} if "lca_name" in p else {})}
        for p in pooled["top_pairs"]
    ]
    return {
        "rows": sum(s.n_errors for s in fallback.values()),
        "by_family": {f: s.n_errors for f, s in sorted(fallback.items())},
        "taxonomy": pooled["taxonomy"],
        "top_pairs": top,
    }


def _pooled(stats: dict[str, FamilyStats]) -> FamilyStats:
    """Pool populations by summing raw counters (never by averaging shares)."""
    total = FamilyStats()
    for s in stats.values():
        total.n += s.n
        total.abstained += s.abstained
        total.answered += s.answered
        total.truth_abstain += s.truth_abstain
        total.correct += s.correct
        total.errors.update(s.errors)
        total.same_id += s.same_id
        total.pairs.update(s.pairs)
        total.pair_lca.update(s.pair_lca)
    return total


def crosscheck(report: dict, check_path: str) -> list[str]:
    """Recompute-vs-report: PASS/DIFFERS per family against the engine's own report."""
    with open(check_path, encoding="utf-8") as f:
        doc = json.load(f)
    if "accuracy_by_family" in doc:
        theirs = {f: v["accuracy"] for f, v in doc["accuracy_by_family"].items()}
    elif "breakdowns" in doc:
        theirs = {f: v["accuracy"] for f, v in doc["breakdowns"]["family"].items()}
    else:
        return [f"SKIP {check_path}: no accuracy_by_family or breakdowns.family"]
    overall = doc.get("accuracy")

    lines = [f"cross-check vs {check_path} (tolerance {TOLERANCE:g})"]
    mine = {f: v for f, v in report["families"].items() if v["n"]}
    for family in sorted(set(mine) | set(theirs)):
        if family not in mine:
            lines.append(f"SKIP    {family:<18} not in this population (n=0 here)")
            continue
        if family not in theirs:
            lines.append(f"SKIP    {family:<18} not in the engine report")
            continue
        ours, their = mine[family]["accuracy"], theirs[family]
        verdict = "PASS" if abs(ours - their) <= TOLERANCE else "DIFFERS"
        lines.append(f"{verdict:<7} {family:<18} mine={ours:.6f} report={their:.6f} "
                     f"(n={mine[family]['n']})")
    their_n = overall.get("n") if isinstance(overall, dict) else None
    their_acc = overall.get("accuracy") if isinstance(overall, dict) else overall
    ours, n_here = report["pooled"]["accuracy"], report["pooled"]["n"]
    if their_acc is None:
        lines.append("SKIP    accuracy           not in the engine report")
    elif their_n != n_here:
        # A report whose overall accuracy has no n, or a different one, is over a different
        # population (it includes rows this analysis does not score, e.g. noul rows). Say so
        # rather than calling a real difference a mismatch -- or a mismatch a pass.
        over = f"n={their_n:,}" if their_n is not None else "an unstated n"
        lines.append(
            f"NOTE    accuracy           report={their_acc:.6f} over {over} vs "
            f"mine={ours:.6f} over the {n_here:,} choice rows kept here -- different "
            "denominators, not comparable; compare per family only"
        )
    else:
        verdict = "PASS" if abs(ours - their_acc) <= TOLERANCE else "DIFFERS"
        lines.append(f"{verdict:<7} accuracy           mine={ours:.6f} report={their_acc:.6f}")
    return lines


# --------------------------------------------------------------------------------------
# Markdown
# --------------------------------------------------------------------------------------


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _render_stats_md(lines: list[str], stats: dict) -> None:
    tax = stats["taxonomy"]
    lines.append("")
    lines.append(f"n={stats['n']:,}, abstained {stats['abstained']:,} "
                 f"({pct(stats['abstained_share'])}), answered {stats['answered']:,} "
                 f"({pct(stats['answered_share'])}), accuracy {stats['accuracy']:.4f}")
    lines.append(f"truth is the abstain option in {stats['truth_is_abstain']:,} rows.")
    if not stats["errors"]:
        lines.append("")
        lines.append("No errors against a real-type truth in this population "
                     "(every wrong answer here is a starved-arm fallback -- see below).")
        return
    lines.append("")
    lines.append(f"Errors: {stats['errors']:,} (both sides named types). Shares are of those errors.")
    lines.append("")
    lines.append("| class | n | share of errors |")
    lines.append("|---|---:|---:|")
    for klass in TAXONOMY:
        lines.append(f"| {klass} | {tax[klass]['n']:,} | {pct(tax[klass]['share'])} |")
    if stats["same_id"]:
        lines.append(f"| same_id (two names, one CL id) | {stats['same_id']:,} | -- |")
    if stats["top_pairs"]:
        lines.append("")
        lines.append(f"Top {len(stats['top_pairs'])} confusion pairs (truth -> predicted):")
        lines.append("")
        lines.append("| truth | predicted | count | class |")
        lines.append("|---|---|---:|---|")
        for p in stats["top_pairs"]:
            klass = p["class"]
            if klass == "cousin" and p.get("lca_name"):
                klass = f"cousin (LCA={p['lca_name']})"
            lines.append(f"| {p['truth']} | {p['predicted']} | {p['count']:,} | {klass} |")


def render_md(report: dict) -> str:
    prov, pop = report["provenance"], report["population"]
    out: list[str] = [f"# Confusion structure — {prov['name']}", ""]
    out.append("## How to read this")
    out.append("")
    out.extend(f"* {item}" for item in HOW_TO_READ)

    out += ["", "## Provenance", "", "| field | value |", "|---|---|"]
    for key in ("engine", "rows", "rows_glob", "rows_files", "rows_sha256", "row_filter",
                "split", "prediction_field", "corpus", "corpus_sha256", "cl_terms",
                "cl_terms_sha256", "definitions", "definitions_sha256"):
        out.append(f"| {key} | `{json.dumps(prov[key])}` |")
    counts = prov["counts"]
    out.append(f"| rows read / kept | {counts.get('read', 0):,} / {counts.get('kept', 0):,} |")
    if prov.get("ids_from"):
        out.append(f"| ids kept / dropped by `--ids-from` | {prov['ids_from_kept']:,} / "
                   f"{prov['ids_from_dropped']:,} (of {prov['ids_from_names']:,} ids named by "
                   f"`{prov['ids_from']}`) |")
    out.append(f"| dropped: split filter | {counts.get('split', 0):,} |")
    out.append(f"| dropped: --row-filter | {counts.get('row_filter', 0):,} |")
    out.append(f"| dropped: type != choice (noul) | {counts.get('type_not_choice', 0):,} |")
    out.append(f"| rows with no abstain option | {prov['rows_without_abstain_option']:,} |")
    out.append(f"| errors with an unresolvable truth / prediction | "
               f"{prov['truth_unresolved']:,} / {prov['pred_unresolved']:,} |")
    out.append(f"| fallback rows whose cell is not in the corpus | "
               f"{prov['fallback_cells_not_in_corpus']:,} |")

    out += ["", "## Population", "",
            f"rows {pop['rows']:,} over {pop['distinct_cells']:,} distinct cells.",
            "", f"families: {json.dumps(pop['families'])}",
            "", f"datasets: {json.dumps(pop['datasets'])}"]

    out.append("")
    out.append("## Per-family taxonomy")
    for family, stats in report["families"].items():
        out += ["", f"### {family}"]
        if not stats["n"]:
            out.append("")
            out.append("No choice rows in this population (its rows, if any, are noul-type "
                       "marker questions and were dropped by the type filter).")
            continue
        _render_stats_md(out, stats)

    fb = report["starved_fallback"]
    out += ["", "## Starved arm: what the engine named instead of abstaining", "",
            (f"The truth is the abstain option in these {fb['rows']:,} rows "
             f"(by family: {json.dumps(fb['by_family'])}); the engine named a type anyway. "
             "Each named type is classified against **the cell's own true type from the "
             "corpus**, not against the abstain option."), ""]
    if fb["rows"]:
        out.append("| class | n | share |")
        out.append("|---|---:|---:|")
        for klass in TAXONOMY:
            v = fb["taxonomy"][klass]
            out.append(f"| {klass} | {v['n']:,} | {pct(v['share'])} |")
        out.append("")
        out.append(f"Top {len(fb['top_pairs'])} fallbacks (named -> the cell's true type):")
        out.append("")
        out.append("| named | true type | count | class |")
        out.append("|---|---|---:|---|")
        for p in fb["top_pairs"]:
            klass = p["class"]
            if klass == "cousin" and p.get("lca_name"):
                klass = f"cousin (LCA={p['lca_name']})"
            out.append(f"| {p['named']} | {p['true_type']} | {p['count']:,} | {klass} |")
    else:
        out.append("(No such rows -- the engine always abstained when the truth was abstain.)")

    out += ["", "## Per-dataset (choice rows pooled across families)", "",
            "| dataset | n | accuracy | errors | " + " | ".join(TAXONOMY) + " |",
            "|---|---:|---:|---:|" + "---:|" * len(TAXONOMY)]
    for dataset, stats in report["by_dataset"].items():
        cells = " | ".join(pct(stats["taxonomy"][k]["share"]) for k in TAXONOMY)
        out.append(f"| {dataset} | {stats['n']:,} | {stats['accuracy']:.4f} | "
                   f"{stats['errors']:,} | {cells} |")

    if report["crosscheck"]["lines"]:
        out += ["", "## Cross-check against the engine's own report", "", "```"]
        out.extend(report["crosscheck"]["lines"])
        out.append("```")
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------------------
# Summary mode
# --------------------------------------------------------------------------------------


def _share_of(classes: tuple[str, ...], stats: dict) -> float:
    errors = stats["errors"]
    return sum(stats["taxonomy"][k]["n"] for k in classes) / errors if errors else 0.0


def write_summary(paths: list[str], out_path: str) -> None:
    rows = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
        p = doc["pooled"]
        rows.append({
            "engine": doc["tool"]["engine"],
            "n": p["n"],
            "abstained_share": p["abstained_share"],
            "accuracy": p["accuracy"],
            "coarsening": _share_of(("coarsening",), p),
            "narrowing": _share_of(("narrowing",), p),
            "sibling_cousin": _share_of(("sibling", "cousin"), p),
            "unrelated": _share_of(("unrelated",), p),
            "unmappable": _share_of(("unmappable",), p),
            "errors": p["errors"],
        })
    lines = [
        "# Confusion structure — summary",
        "",
        ("One line per engine, pooled over the choice rows of every family in its report "
         "(marker_noul rows are noul-type marker questions and are never scored here). "
         "The class columns are shares of that engine's errors (the rows where both the truth "
         "and the prediction are named types). Regenerate with:"),
        "",
        "```",
        "uv run --no-sync python scripts/confusion-structure.py --summary "
        + " ".join(paths) + " --out " + out_path,
        "```",
        "",
        ("| engine | rows | abstained | accuracy | errors | coarsening | narrowing | "
         "sibling+cousin | unrelated | unmappable |"),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['engine']} | {r['n']:,} | {pct(r['abstained_share'])} | {r['accuracy']:.4f} "
            f"| {r['errors']:,} | {pct(r['coarsening'])} | {pct(r['narrowing'])} "
            f"| {pct(r['sibling_cousin'])} | {pct(r['unrelated'])} | {pct(r['unmappable'])} |"
        )
    text = "\n".join(lines) + "\n"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)


# --------------------------------------------------------------------------------------
# Selftest
# --------------------------------------------------------------------------------------

# A miniature ontology that mirrors the real v3 edges for the pinned cases (the real
# artifacts are not needed to check the taxonomy's direction). Every node is a real CL term.
SELFTEST_TERMS = [
    {"cl_id": "CL:0000000", "name": "cell", "parents": []},
    {"cl_id": "CL:0000540", "name": "neuron", "parents": ["CL:0000000"]},
    {"cl_id": "CL:0000617", "name": "GABAergic neuron", "parents": ["CL:0000540"]},
    {"cl_id": "CL:1001474", "name": "medium spiny neuron", "parents": ["CL:0000617"]},
    {"cl_id": "CL:0000700", "name": "dopaminergic neuron", "parents": ["CL:0000540"]},
    {"cl_id": "CL:0000094", "name": "granulocyte", "parents": ["CL:0000000"]},
    {"cl_id": "CL:0000775", "name": "neutrophil", "parents": ["CL:0000094"]},
    {"cl_id": "CL:0000771", "name": "eosinophil", "parents": ["CL:0000094"]},
    {"cl_id": "CL:0000066", "name": "epithelial cell", "parents": ["CL:0000000"]},
    {"cl_id": "CL:0000068", "name": "duct epithelial cell", "parents": ["CL:0000066"]},
    {"cl_id": "CL:0000069", "name": "branched duct epithelial cell", "parents": ["CL:0000068"]},
    {"cl_id": "CL:0002079", "name": "pancreatic ductal cell", "parents": ["CL:0000069"]},
    {"cl_id": "CL:0000057", "name": "fibroblast", "parents": ["CL:0000000"]},
]
SELFTEST_ABSTAIN = "other"
# case name -> (truth key, prediction key, expected class)
SELFTEST_CASES = {
    "coarsening": ("medium spiny neuron", "neuron", "coarsening"),
    "narrowing": ("neuron", "medium spiny neuron", "narrowing"),
    "sibling": ("neutrophil", "eosinophil", "sibling"),
    "cousin": ("medium spiny neuron", "dopaminergic neuron", "cousin"),
    "unrelated": ("medium spiny neuron", "fibroblast", "unrelated"),
    "unmappable_pred": ("neuron", "local:siletti:cl_4023008", "unmappable"),
    "unmappable_truth": ("local:siletti:cl_4023008", "neuron", "unmappable"),
    # the pinned traps: a parent is not a sibling, and ductal -> epithelial is a coarsening
    "parent_not_sibling": ("neutrophil", "granulocyte", "coarsening"),
    "pancreatic_ductal": ("pancreatic ductal cell", "epithelial cell", "coarsening"),
}


def selftest() -> int:
    """Run the whole taxonomy on synthetic rows. One line per case; non-zero on any miss."""
    from sckevtype.eval import abstain_key

    ontology = Ontology(SELFTEST_TERMS)
    definitions = [{"name": t["name"], "cl_id": t["cl_id"]} for t in SELFTEST_TERMS]
    ctx = {
        "ontology": ontology,
        "register": Register(ontology, definitions),
        "corpus": {("selftest", "msn"): "CL:1001474", ("selftest", "neut"): "CL:0000775"},
        "abstain_key": abstain_key,
        "pred_field": Counter(),
        "extras": Counter(),
    }
    rows = []
    for case, (truth, pred, _) in SELFTEST_CASES.items():
        rows.append({
            "family": case, "type": "choice", "split": "test", "dataset": "selftest",
            "cell": "msn", "keys": [truth, pred], "label": 0, "p": [0.4, 0.6],
        })
    # starved arm: the labelled truth is the abstain option, the engine named a type anyway
    rows.append({"family": "starved", "type": "choice", "split": "test", "dataset": "selftest",
                 "cell": "msn", "keys": [SELFTEST_ABSTAIN, "neuron"], "label": 0,
                 "p": [0.1, 0.9]})
    # the engine abstained
    rows.append({"family": "abstained", "type": "choice", "split": "test", "dataset": "selftest",
                 "cell": "msn", "keys": ["medium spiny neuron", SELFTEST_ABSTAIN], "label": 0,
                 "p": [0.2, 0.8]})

    body = analyse(rows, ctx)
    failures = 0
    for case, (truth, pred, expected) in SELFTEST_CASES.items():
        got, lca = classify(ctx["register"].resolve(pred), ctx["register"].resolve(truth),
                            ontology, ctx["register"])
        ok = got == expected
        failures += not ok
        detail = f" (LCA={ontology.name_of(lca)})" if lca else ""
        print(f"{case:<18} truth={truth!r} pred={pred!r} -> {got}{detail}  "
              f"{'OK' if ok else 'FAIL: expected ' + expected}")

    def check(name: str, ok: bool, detail: str) -> None:
        nonlocal failures
        failures += not ok
        print(f"{name:<18} {detail}  {'OK' if ok else 'FAIL'}")

    starved = body["families"]["starved"]
    check("starved", starved.truth_abstain == 1 and starved.abstained == 0
          and starved.n_errors == 0, "truth=abstain, engine named a type: no taxonomy error")
    check("starved-fallback", body["fallback"]["starved"].errors["coarsening"] == 1,
          "named 'neuron' for a medium spiny neuron -> fallback coarsening")
    abstained = body["families"]["abstained"]
    check("abstained", abstained.abstained == 1 and abstained.answered == 0,
          "engine named the abstain option: abstention bucket, not an error")
    # A check that cannot fail is not a check: the narrowing input must NOT be coarsening.
    got, _ = classify(ctx["register"].resolve("medium spiny neuron"),
                      ctx["register"].resolve("neuron"), ontology, ctx["register"])
    check("negative-control", got != "coarsening" and got == "narrowing",
          "narrowing input (truth=neuron pred=medium spiny neuron) is NOT coarsening")
    print(f"selftest: {'PASS' if not failures else f'{failures} FAILURES'}")
    return 1 if failures else 0


# --------------------------------------------------------------------------------------


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", help="row dump: a file or a directory")
    parser.add_argument("--glob", default="*.jsonl", help="pattern when --rows is a directory")
    parser.add_argument("--engine", help="engine name (provenance + summary rows)")
    parser.add_argument("--name", help="display name; defaults to --engine")
    parser.add_argument("--corpus", help="runs/real-v2/test.jsonl (cell -> true CL term)")
    parser.add_argument("--cl-terms", help="artifacts/v3/cl_terms.json")
    parser.add_argument("--definitions", help="artifacts/v3/definitions.json")
    parser.add_argument("--split", default="test", help="'test' (default) or 'any'")
    parser.add_argument("--row-filter", action="append", default=[], metavar="KEY=VALUE",
                        help="repeatable; keep rows whose KEY equals VALUE (e.g. ask=base)")
    parser.add_argument("--check-report", help="the engine's own report.json to cross-check")
    parser.add_argument("--ids-from", metavar="ROWS.JSONL",
                        help="restrict the scored population to the row ids named in this file "
                             "(e.g. reports/jev-full/rows.jsonl -- the census dump, so every "
                             "engine's number is taken on the same rows)")
    parser.add_argument("--out", help="JSON out (normal mode) or markdown out (--summary)")
    parser.add_argument("--md", help="markdown out (normal mode)")
    parser.add_argument("--summary", nargs="+", metavar="JSON",
                        help="regenerate the cross-engine summary from report JSONs")
    parser.add_argument("--selftest", action="store_true",
                        help="run the taxonomy on synthetic rows and exit")
    args = parser.parse_args(argv)

    if args.selftest:
        return selftest()
    if args.summary:
        if not args.out:
            parser.error("--summary needs --out <markdown path>")
        write_summary(args.summary, args.out)
        return 0

    for required in ("rows", "engine", "corpus", "cl_terms", "definitions", "out"):
        if not getattr(args, required):
            parser.error(f"--{required.replace('_', '-')} is required")
    report, lines = build_report(args)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1, sort_keys=False)
        f.write("\n")
    if args.md:
        with open(args.md, "w", encoding="utf-8") as f:
            f.write(render_md(report))
    for line in lines:
        print(line)
    print(f"wrote {args.out}" + (f" and {args.md}" if args.md else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
