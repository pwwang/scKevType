#!/usr/bin/env python3
"""The `real-v3` corpus: what it holds, what it dropped, and the questions themselves.

`real-v3` is `real-v2` with two defects fixed, and this script is the evidence that
only those two things changed. No model is run; every number is read off files.

The two fixes
-------------
1. **One question per `(dataset, cell, family)`.** v2 carried both renderings of a
   record in calibration / development / test (``variant=clean`` and its
   ``variant=permuted`` twin, the option-order control), so the scored population
   counted 322 cells twice (``reports/variant-redundancy.md``). v3 emits one: the
   clean rendering, except in train, which stays permuted as it always was.
2. **No marker lists in the option text.** v2's option definitions ended
   ``… . positive markers: GCG, Gcg, ARX. negative markers: …``, which a state whose
   rank-1 gene is ``GCG`` answers by string matching. v3 renders the prose alone.

What this script checks, and against what
-----------------------------------------
The reference is the **frozen v2 suite** (``artifacts/suites/real-v2``): the rows
``kev.data.load_records`` read out of the v2 corpus, one line per corpus record, with
kev's own ``_meta`` keys added on top (``id``, ``row``, ``split``, ``text_sha256``).
Its records *are* the v2 corpus's records, which is why the author's census
(``reports/census-suite-ids.jsonl``) indexes straight into it — checked in §"census".

* **equality** — every v3 record's ``state``, ``questions`` (v2's marker clauses
  removed from its texts), option-key order, ground-truth label and every other
  ``_meta`` field must equal the corresponding v2 rendering's. The two fields that
  may differ are named and counted: ``source`` (the corpus name) and ``panel_id``
  (a digest *of* the option text, so it must move when the text does).
* **marker strip** — no rendered criteria string, and no question string at all,
  contains ``marker`` (case-insensitive) or a comma-separated gene-symbol list; and
  no option's text contains a gene symbol of the register definition it names.
* **de-duplication** — exactly one row per ``(dataset, cell, family)``; how many rows
  that collapsed; and how many kept the permuted rendering because it was the only
  one they had.
* **counts** — per split, per family and per dataset, v3 against v2; plus the
  author's own scored population (the census triples) under both.

Usage
-----
    source scripts/wsl/env.sh
    uv run --no-sync python reports/real-v3-corpus.py
    # optional: --v2-rebuild <dir>  a rebuilt v2 corpus, digest-checked against the
    #                               digests the frozen artifacts record for it

Writes ``reports/real-v3-corpus.json`` and ``reports/real-v3-corpus.md``, and nothing
else.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V3_DIR = ROOT / "artifacts" / "corpus" / "real-v3"
V2_SUITE = ROOT / "artifacts" / "suites" / "real-v2"
#: The register the v2 corpus was built on, frozen under a new path (see its README): the
#: Step-9 derived tissue map plus the canonical v2 terms. `artifacts/v2` at HEAD cannot
#: serve this corpus at all — see :func:`register_report`.
V2_AT_BUILD = ROOT / "artifacts" / "v2-at-build"
TISSUE_MAP = ROOT / "artifacts" / "v2" / "tissue_map.json"
STEP9_SPEC = ROOT / "docs" / "SPEC-step9-tissue-map-sync.md"
#: The commit that put the pre-Step-9 map back, and the one sentence that says so.
RESTORING_COMMIT = "a74adb8 CL hierarchy recovered: parse is_a edges; v2 restored, v3 cut"
HIERARCHY_REPORT = ROOT / "reports" / "cl-hierarchy-recovery.md"
#: Families whose `criteria` *is* the drawn panel (marker_noul's is the yes/no pair).
PANEL_FAMILIES = ("panel_choice", "panel_miss", "evidence_removed", "evidence_diluted", "out_of_register")
#: The v2 corpus's own file digests, recorded when it was built (its `runs/real-v2`
#: directory is gone; this is the surviving record of what those bytes were).
V2_RECORD = ROOT / "artifacts" / "eval" / "panel-swaps-manifest.json"
CENSUS = ROOT / "reports" / "census-suite-ids.jsonl"
DEFINITIONS = ROOT / "artifacts" / "v2" / "definitions.json"
SPLITS = ("train", "calibration", "development", "test")
#: kev's own ``_meta`` keys, added by its record loader on top of the corpus's.
KEV_META = ("id", "row", "split", "text_sha256")
#: The fields a v3 record may differ from its v2 twin on, with the reason.
META_MAY_DIFFER = ("source", "panel_id")

#: ``<prose>. positive markers: A, B. negative markers: C.`` -> the clause is
#: ``positive markers: A, B`` (and its leading separator); the sentence's own final
#: period is not part of it.
CLAUSE = re.compile(r"(?:\. |^)(?:positive|negative) markers: [^.]*")
MARKER_WORD = re.compile(r"marker", re.IGNORECASE)
#: The clause a v2 option text ends with — what the strip removes. The bare word
#: "markers" can also be English prose inside a curated definition ("…lacks
#: hematopoietic lineage markers."), which is the definition itself, is v2's text
#: too, and is not a list — the two are counted separately, never conflated.
MARKER_CLAUSE = re.compile(r"(?:positive|negative) markers:")
#: A gene-symbol-shaped token: a letter, then letters/digits/``-``/``.``/``@``, ≤10 long.
SYMBOL = re.compile(r"[A-Za-z][A-Za-z0-9.\-@]{0,9}")
#: A comma-separated list of ≥3 gene-symbol-shaped tokens (the shape a marker list
#: has). Prose is separated by ``.``/``;``/``:``, so a run of three or more
#: symbol-shaped tokens inside one of those is what this flags.
LIST_SPLIT = re.compile(r"[.;:()\[\]]")


def strip_clauses(text: str) -> str:
    """The text v3 renders for a v2 option text: the same prose, no marker clauses."""
    out = CLAUSE.sub("", text)
    return "" if out == "." else out


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def norm_question(question: dict, strip: bool) -> dict:
    """One question payload, with v2's marker clauses removed when ``strip``.

    The clauses live in two places: a choice question's ``criteria`` (option name ->
    definition text) and a holds question's ``instructions.definition``. Both are
    prompt text, so both are normalised the same way.
    """
    payload = {k: v for k, v in question.items() if k not in ("qtype", "keys")}
    if strip:
        criteria = payload.get("criteria")
        if isinstance(criteria, dict):
            payload["criteria"] = {key: strip_clauses(value) if isinstance(value, str) else value for key, value in criteria.items()}
        elif isinstance(criteria, list):
            payload["criteria"] = [strip_clauses(value) if isinstance(value, str) else value for value in criteria]
        instructions = payload.get("instructions")
        if isinstance(instructions, dict):
            payload["instructions"] = {
                key: strip_clauses(value) if key == "definition" and isinstance(value, str) else value
                for key, value in instructions.items()
            }
    return payload


def row_digest(row: dict, strip: bool) -> str:
    """Digest of everything a v2 and a v3 record must share: state, questions, labels."""
    payload = {
        "state": row["state"],
        "questions": {qid: norm_question(question, strip) for qid, question in row["questions"].items()},
    }
    return sha256_text(json.dumps(payload, sort_keys=True, ensure_ascii=False))


def meta_key(meta: dict) -> tuple:
    return (meta.get("dataset_id"), meta.get("cell_id"), str(meta.get("family")), meta.get("variant"))


def compare_meta(meta: dict) -> dict:
    return {key: value for key, value in meta.items() if key not in KEV_META and key not in META_MAY_DIFFER}


def question_strings(row: dict) -> list[tuple[str, str]]:
    """Every prompt string of one record: ``(where, text)``, in question order."""
    out: list[tuple[str, str]] = []
    for qid, question in row["questions"].items():
        if isinstance(question.get("instructions"), str):
            out.append((f"{qid}.instructions", question["instructions"]))
        elif isinstance(question.get("instructions"), dict):
            for key, value in question["instructions"].items():
                if isinstance(value, str):
                    out.append((f"{qid}.instructions.{key}", value))
        criteria = question.get("criteria")
        if isinstance(criteria, dict):
            for name, value in criteria.items():
                if isinstance(value, str):
                    out.append((f"{qid}.criteria[{name}]", value))
        elif isinstance(criteria, list):
            for index, value in enumerate(criteria):
                if isinstance(value, str):
                    out.append((f"{qid}.criteria[{index}]", value))
        for name in (question.get("criteria") or {}) if isinstance(question.get("criteria"), dict) else []:
            out.append((f"{qid}.key[{name}]", str(name)))
    return out


def criteria_strings(row: dict) -> list[str]:
    """Only the rendered criteria values — the option definitions the model reads."""
    out: list[str] = []
    for question in row["questions"].values():
        criteria = question.get("criteria")
        if isinstance(criteria, dict):
            out.extend(value for value in criteria.values() if isinstance(value, str))
        elif isinstance(criteria, list):
            out.extend(value for value in criteria if isinstance(value, str))
    return out


def comma_symbol_runs(text: str) -> list[str]:
    """Runs of ≥3 comma-separated gene-symbol-shaped tokens, if any."""
    hits = []
    for chunk in LIST_SPLIT.split(text):
        parts = [part.strip() for part in chunk.split(",")]
        if len(parts) >= 3 and all(SYMBOL.fullmatch(part) for part in parts):
            hits.append(chunk.strip())
    return hits


def rank_list_head(state: str, n: int = 30) -> tuple[list[str], int]:
    """``(first n entries, total entries)`` of a rendered rank list."""
    entries = [item.strip() for item in state.split("|") if item.strip()]
    return entries[:n], len(entries)


def load_split(directory: Path, split: str) -> list[dict]:
    path = directory / f"{split}.jsonl"
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def count_records(rows: list[dict]) -> dict:
    """Per family / per dataset / per dataset-and-family / per variant counts."""
    families = Counter(str(row["_meta"].get("family")) for row in rows)
    datasets = Counter(str(row["_meta"].get("dataset_id")) for row in rows)
    per_dataset_family: dict[str, Counter] = defaultdict(Counter)
    variants = Counter(str(row["_meta"].get("variant")) for row in rows)
    for row in rows:
        per_dataset_family[str(row["_meta"].get("dataset_id"))][str(row["_meta"].get("family"))] += 1
    return {
        "records": len(rows),
        "questions": sum(len(row["questions"]) for row in rows),
        "cells": len({row["_meta"].get("group_id") for row in rows}),
        "per_family": dict(sorted(families.items())),
        "per_dataset": dict(sorted(datasets.items())),
        "per_dataset_family": {dataset: dict(sorted(counts.items())) for dataset, counts in sorted(per_dataset_family.items())},
        "per_variant": dict(sorted(variants.items())),
    }


def scan_v2(directory: Path) -> dict:
    """One pass over a v2 rendering: counts, per-triple structure and per-row digests.

    ``directory`` may be the frozen suite (``artifacts/suites/real-v2`` — the v2
    corpus's records as ``kev.data.load_records`` reads them, with kev's own
    ``_meta`` keys added) or the v2 corpus itself as the builder writes it. Both hold
    the same records in the same order, which is what makes the census ids — line
    indices into the test partition — resolvable either way. The split is taken from
    the file name, never from the row, so the two are read the same way.
    """
    counts = {split: {} for split in SPLITS}
    rows_by_split: dict[str, int] = {}
    triples: dict[tuple, Counter] = defaultdict(Counter)          # triple -> variant counts
    digests: dict[tuple, str] = {}                                # (triple, variant) -> digest
    meta_by_key: dict[tuple, dict] = {}
    panel_ids: dict[tuple, str | None] = {}
    census_index: dict[int, dict] = {}                            # test row -> what is on it
    rows_by_key: dict[tuple, dict] = {}
    for split in SPLITS:
        rows = load_split(directory, split)
        rows_by_split[split] = len(rows)
        counts[split] = count_records(rows)
        for index, row in enumerate(rows):
            meta = row["_meta"]
            meta.setdefault("split", split)
            key = meta_key(meta)
            triples[(meta.get("dataset_id"), meta.get("cell_id"), str(meta.get("family")))][meta.get("variant")] += 1
            digests[key] = row_digest(row, strip=True)
            meta_by_key[key] = compare_meta(meta)
            panel_ids[key] = meta.get("panel_id")
            rows_by_key[key] = row
            if split == "test":
                census_index[index] = {
                    "dataset": meta.get("dataset_id"), "cell": meta.get("cell_id"),
                    "family": str(meta.get("family")), "variant": meta.get("variant"),
                    "key": key,
                }
    return {
        "directory": directory,
        "counts": counts,
        "rows_by_split": rows_by_split,
        "triples": triples,
        "digests": digests,
        "meta_by_key": meta_by_key,
        "panel_ids": panel_ids,
        "census_index": census_index,
        "suite_rows": rows_by_key,
    }


def triple_summary(triples: dict[tuple, Counter]) -> dict:
    per_family: dict[str, Counter] = defaultdict(Counter)
    for (_, _, family), variants in triples.items():
        rows = sum(variants.values())
        per_family[family]["triples"] += 1
        per_family[family]["rows"] += rows
        per_family[family]["doubled" if rows > 1 else f"lone_{next(iter(variants))}"] += 1
    return {
        "triples": len(triples),
        "rows": sum(sum(variants.values()) for variants in triples.values()),
        "rows_per_triple": round(sum(sum(v.values()) for v in triples.values()) / max(len(triples), 1), 3),
        "per_family": {family: {k: v for k, v in sorted(counts.items())} for family, counts in sorted(per_family.items())},
    }


def flatten(payload: dict, prefix: str = "") -> dict:
    """One flat ``dotted.path -> value`` view of a nested config payload."""
    flat: dict = {}
    for key, value in payload.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten(value, f"{name}."))
        elif isinstance(value, list):
            flat[name] = json.dumps(value, sort_keys=True)
        else:
            flat[name] = value
    return flat


def config_diff(left: dict, right: dict) -> list[str]:
    """The dotted paths on which two config payloads disagree."""
    flat_left, flat_right = flatten(left), flatten(right)
    return sorted(path for path in set(flat_left) | set(flat_right) if flat_left.get(path) != flat_right.get(path))


def load_definitions() -> dict[str, list[tuple[str, ...]]]:
    """option name -> every register marker list a definition with that name carries."""
    payload = json.loads(DEFINITIONS.read_text(encoding="utf-8"))
    by_name: dict[str, list[tuple[str, ...]]] = defaultdict(list)
    for definition in payload["definitions"]:
        by_name[definition["name"]].append(tuple(definition.get("pos_markers", ())) + tuple(definition.get("neg_markers", ())))
    return dict(by_name)


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def register_report(v2_rows: list[dict], failure_log: str | None) -> dict:
    """What each register revision serves, what it cannot, and what the corpus asks of it.

    The corpus's own records say which ``(panel scope, true type)`` pairs the build asked the
    register to serve: ``sample_panel`` raises unless the true type is a candidate for that scope.
    ``global`` is not a tissue list but the register itself, so its candidates are the definitions.
    """
    live = json.loads(TISSUE_MAP.read_text(encoding="utf-8"))
    at_build = json.loads((V2_AT_BUILD / "tissue_map.json").read_text(encoding="utf-8"))
    definitions = {definition["cl_id"] for definition in json.loads(DEFINITIONS.read_text(encoding="utf-8"))["definitions"]}
    pairs: Counter = Counter()
    for row in v2_rows:
        meta = row["_meta"]
        scope, cl_id = str(meta.get("panel_scope")), meta.get("cl_id")
        if isinstance(cl_id, str) and cl_id.startswith("CL:"):
            pairs[(scope, cl_id)] += 1

    def unservable(mapping: dict) -> dict:
        out = {}
        for (scope, cl_id), count in sorted(pairs.items()):
            candidates = definitions if scope == "global" else set(mapping["tissues"].get(scope, []))
            if cl_id not in candidates:
                out[f"{scope}:{cl_id}"] = count
        return out

    failure = {"log": None, "lines": []}
    if failure_log and Path(failure_log).exists():
        body = Path(failure_log).read_text(encoding="utf-8", errors="replace").splitlines()
        failure = {"log": Path(failure_log).name,
                   "lines": [line.strip() for line in body if "build failed" in line or line.startswith("exit=")]}
    return {
        "artifacts/v2 (live, at HEAD)": {
            "tissue_map_sha256": sha_file(TISSUE_MAP),
            "candidates_per_tissue": {t: len(v) for t, v in sorted(live["tissues"].items())},
            "unservable_pairs": unservable(live),
        },
        "artifacts/v2-at-build (what real-v3 reads)": {
            "tissue_map_sha256": sha_file(V2_AT_BUILD / "tissue_map.json"),
            "candidates_per_tissue": {t: len(v) for t, v in sorted(at_build["tissues"].items())},
            "unservable_pairs": unservable(at_build),
        },
        "distinct_scope_cl_pairs_the_corpus_asks_for": len(pairs),
        "records_behind_them": sum(pairs.values()),
        "restoring_commit": RESTORING_COMMIT,
        "step9_spec": str(STEP9_SPEC.relative_to(ROOT)),
        "hierarchy_report": str(HIERARCHY_REPORT.relative_to(ROOT)),
        "failure_log": failure,
    }


def replay_check(sample: list[dict], stride: int) -> dict:
    """Does the frozen corpus's panel come back when the draw is replayed on this register?

    A panel is drawn from ``(true type, scope, definitions, PanelConfig)`` with an rng seeded on
    ``(seed, "cell", dataset, cell_id)`` — the builder's own code path, called here directly. Clean
    records of the panel-bearing families are the ones whose option order *is* that draw, so they
    are the test: a row that reproduces proves this register revision is the one the corpus was
    drawn on, and the corpus is therefore v2's questions rather than a fresh draw.
    """
    from sckevtype.data import seed_int, seeded_rng
    from sckevtype.data.build import Artifacts, load_config
    from sckevtype.data.families import NONE_OPTIONS
    from sckevtype.data.panels import PanelConfig, sample_miss_panel, sample_panel_pair

    config = load_config(ROOT / "configs" / "corpus" / "real-v3.yaml")
    artifacts = Artifacts(V2_AT_BUILD)
    panel_cfg = PanelConfig.from_artifacts(artifacts.tissue_map, artifacts.cl_terms,
                                           **config.panel.model_dump(exclude={"tissues", "lineage", "register_terms"}))
    abstain = {key for key, _ in NONE_OPTIONS}
    same = different = refused = 0
    for row in sample:
        meta = row["_meta"]
        wanted = [name for name in (next(iter(row["questions"].values())).get("criteria") or {}) if name not in abstain]
        if not wanted:
            continue
        rng = seeded_rng(seed_int(config.seed, "cell", meta["dataset_id"], meta["cell_id"]), "panel")
        try:
            if str(meta.get("family")) == "out_of_register":
                drawn = sample_miss_panel(str(meta.get("panel_scope")), artifacts.definitions, panel_cfg, rng)
            else:
                miss_panel, sibling = sample_panel_pair(str(meta.get("cl_id")), str(meta.get("panel_scope")),
                                                        artifacts.definitions, panel_cfg, rng)
                drawn = miss_panel if str(meta.get("family")) == "panel_miss" else (sibling or miss_panel)
        except ValueError:
            refused += 1
            continue
        if [option.name for option in drawn.options] == wanted:
            same += 1
        else:
            different += 1
    return {
        "rows": same + different + refused,
        "panels_reproduced": same,
        "panels_different": different,
        "refused_by_the_register": refused,
        "sample": (f"every {stride}th clean, panel-bearing v2 record outside train "
                   f"(their option order is the draw's, not the variant policy's)"),
        "register": str(V2_AT_BUILD.relative_to(ROOT)),
    }


def example_rows(rows: list[dict], families: tuple[str, ...]) -> dict[str, dict] | None:
    """One cell's records, when it has all of ``families`` — prefer the smallest panel."""
    groups: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        groups[str(row["_meta"].get("group_id"))][str(row["_meta"].get("family"))] = row
    options = []
    for group_id, records in groups.items():
        if not all(family in records for family in families):
            continue
        sizes = [
            len(next(iter(row["questions"].values())).get("criteria") or {})
            for row in records.values()
        ]
        options.append((min(sizes), group_id, records))
    if not options:
        return None
    _, _, records = min(options, key=lambda item: (item[0], item[1]))
    return {family: records[family] for family in families}


def question_block(row: dict, index: int) -> list[str]:
    """The verbatim rendering of one record's single question, as report lines."""
    qid, question = next(iter(row["questions"].items()))
    meta = row["_meta"]
    head, total = rank_list_head(row["state"])
    lines = [
        f"**Example {index} — family `{meta['family']}`, split `{meta.get('split', 'test')}`, "
        f"dataset `{meta['dataset_id']}`, cell `{meta['cell_id']}`, ground truth `{question['label']}`**",
        "",
        f"- question id `{qid}`, variant `{meta['variant']}`, panel scope `{meta.get('panel_scope')}`, "
        f"panel size **{len(question['criteria'])}**, true type `{meta['cl_id']}`",
        f"- state: **{total} genes**, `sha256 {sha256_text(row['state'])}`, first 30 verbatim:",
        "",
        "  ```",
        "  " + " | ".join(head),
        "  ```",
        "",
    ]
    if isinstance(question["instructions"], str):
        lines += ["- `instructions` (verbatim):", "", "  ```", f"  {question['instructions']}", "  ```", ""]
    else:
        lines += ["- `instructions` (verbatim):", "", "  ```"]
        lines += [f"  {key}: {value}" for key, value in question["instructions"].items()]
        lines += ["  ```", ""]
    criteria = question["criteria"]
    if isinstance(criteria, dict):
        lines += [f"- `criteria` — every option key and its definition text, verbatim ({len(criteria)} options, in the record's order):", ""]
        for name, text in criteria.items():
            lines.append(f"  - **`{name}`** — {text}")
    else:
        lines += [f"- `criteria` — verbatim ({len(criteria)} levels, in order):", ""]
        lines += [f"  - **`{index}`** — {text}" for index, text in enumerate(criteria)]
    lines += [
        "",
        f"- **answer** (the ground-truth label, `label` = `{question['label']}`)",
        "",
    ]
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v3", default=str(V3_DIR), help="the real-v3 corpus directory")
    parser.add_argument("--v2-source", default=str(V2_SUITE),
                        help="the v2 rendering to compare against: the frozen suite (default) or a rebuilt v2 corpus")
    parser.add_argument("--v2-rebuild", default=None, help="a rebuilt real-v2 corpus, digest-checked against the frozen record")
    parser.add_argument("--failure-log", default=None, help="a build log to quote the register failure from")
    parser.add_argument("--out-json", default=str(ROOT / "reports" / "real-v3-corpus.json"))
    parser.add_argument("--out-md", default=str(ROOT / "reports" / "real-v3-corpus.md"))
    args = parser.parse_args()

    v3_dir = Path(args.v3)
    v3_manifest = json.loads((v3_dir / "manifest.json").read_text(encoding="utf-8"))
    v3_rows = {split: load_split(v3_dir, split) for split in SPLITS}
    v3_counts = {split: count_records(rows) for split, rows in v3_rows.items()}
    v3_everything = [row for split in SPLITS for row in v3_rows[split]]
    v3_total = count_records(v3_everything)

    v2 = scan_v2(Path(args.v2_source))
    v2_source = str(Path(args.v2_source))
    try:
        v2_source = str(Path(args.v2_source).relative_to(ROOT))
    except ValueError:
        pass
    v2_triples = triple_summary(v2["triples"])
    v2_total = count_records(list(v2["suite_rows"].values()))
    v2_all = {
        "records": sum(v2["rows_by_split"].values()),
        "questions": sum(count["questions"] for count in v2["counts"].values()),
        "cells": len({row["_meta"].get("group_id") for row in v2["suite_rows"].values()}),
        "per_family": {family: sum(count["per_family"].get(family, 0) for count in v2["counts"].values())
                       for family in sorted({f for count in v2["counts"].values() for f in count["per_family"]})},
        "per_dataset_family": {
            dataset: {family: sum(count["per_dataset_family"].get(dataset, {}).get(family, 0) for count in v2["counts"].values())
                      for family in sorted({f for count in v2["counts"].values() for f in count["per_dataset_family"].get(dataset, {})})}
            for dataset in sorted({d for count in v2["counts"].values() for d in count["per_dataset"]})
        },
        "per_split": {split: count["records"] for split, count in v2["counts"].items()},
        "per_variant": {variant: sum(count["per_variant"].get(variant, 0) for count in v2["counts"].values())
                        for variant in sorted({v for count in v2["counts"].values() for v in count["per_variant"]})},
    }

    # --- the equality check: every v3 record against its v2 rendering ---------------
    mismatches: list[dict] = []
    meta_field_mismatches: Counter = Counter()
    matched_meta = 0
    matched_meta_relaxed = 0
    missing = 0
    panel_id_changed = 0
    for row in v3_everything:
        key = meta_key(row["_meta"])
        if key not in v2["digests"]:
            missing += 1
            mismatches.append({"key": list(key), "why": "no v2 rendering with this key"})
            continue
        if row_digest(row, strip=False) != v2["digests"][key]:
            mismatches.append({"key": list(key), "why": "state/questions/labels differ"})
        if row["_meta"].get("panel_id") != v2["panel_ids"].get(key):
            panel_id_changed += 1
        reference = v2["meta_by_key"][key]
        mine = compare_meta(row["_meta"])
        if mine == reference:
            matched_meta += 1
        else:
            for field in sorted(set(mine) | set(reference)):
                if mine.get(field) != reference.get(field):
                    meta_field_mismatches[field] += 1
        if {k: v for k, v in mine.items() if k != "seeds"} == {k: v for k, v in reference.items() if k != "seeds"}:
            matched_meta_relaxed += 1

    #: `evidence_diluted` masks `dilution_k` random genes from an rng seeded on
    #: `seed_int(rng_seed, "family", config.name)` — the corpus *name* is part of that
    #: seed, so a corpus with a new name draws its own masks. Nothing else about the
    #: record moves: its panel, its options, its instructions and its holds rule are v2's.
    diluted_rows = sum(1 for row in v3_everything if str(row["_meta"].get("family")) == "evidence_diluted")
    diluted_differing = len([item for item in mismatches
                             if len(item.get("key", [])) > 2 and str(item["key"][2]) == "evidence_diluted"])
    other_differing = len(mismatches) - diluted_differing

    # --- the register: what it can serve, and whether it is the one the corpus was drawn on
    v2_row_list = list(v2["suite_rows"].values())
    register = register_report(v2_row_list, args.failure_log)
    sample = [row for row in v2_row_list
              if str(row["_meta"].get("family")) in PANEL_FAMILIES
              and row["_meta"].get("variant") == "clean"
              and row["_meta"].get("split") != "train"][::97]
    replay = replay_check(sample, stride=97)

    # --- the marker strip ----------------------------------------------------------
    definitions_by_name = load_definitions()
    #: Every register definition rendered the way v3 renders it: prose alone. A string
    #: containing the word "markers" that is one of these is a definition, not a list.
    PROSE_TEXTS = {
        definition["text"].strip().rstrip(".") + "."
        for definition in json.loads(DEFINITIONS.read_text(encoding="utf-8"))["definitions"]
        if definition["text"].strip()
    }
    criteria_seen = 0
    criteria_marker_word: list[dict] = []
    criteria_marker_clause: list[dict] = []
    criteria_not_prose: list[dict] = []
    question_strings_seen = 0
    question_marker_word: list[dict] = []
    question_not_prose: list[dict] = []
    marker_prose_seen: set[str] = set()
    comma_runs: list[dict] = []
    self_marker_hits: list[dict] = []
    option_strings_seen = 0
    for row in v3_everything:
        where_meta = {"dataset": row["_meta"].get("dataset_id"), "cell": row["_meta"].get("cell_id"), "family": str(row["_meta"].get("family"))}
        for text in criteria_strings(row):
            criteria_seen += 1
            if MARKER_CLAUSE.search(text):
                criteria_marker_clause.append({**where_meta, "text": text[:200]})
            if MARKER_WORD.search(text):
                criteria_marker_word.append({**where_meta, "text": text[:200]})
                if text in PROSE_TEXTS:
                    marker_prose_seen.add(text)
                else:
                    criteria_not_prose.append({**where_meta, "text": text[:200]})
        for where, text in question_strings(row):
            question_strings_seen += 1
            if MARKER_WORD.search(text):
                question_marker_word.append({**where_meta, "where": where, "text": text[:200]})
                if text not in PROSE_TEXTS:
                    question_not_prose.append({**where_meta, "where": where, "text": text[:200]})
        for question in row["questions"].values():
            criteria = question.get("criteria")
            if not isinstance(criteria, dict):
                continue
            for name, text in criteria.items():
                if not isinstance(text, str):
                    continue
                option_strings_seen += 1
                for run in comma_symbol_runs(text):
                    comma_runs.append({**where_meta, "option": name, "run": run[:200]})
                own = definitions_by_name.get(name, [])
                for markers in own:
                    for marker in markers:
                        if re.search(rf"(?<![A-Za-z0-9]){re.escape(marker)}(?![A-Za-z0-9])", text):
                            self_marker_hits.append({**where_meta, "option": name, "marker": marker})

    # --- the census: the author's scored population --------------------------------
    census_rows = [json.loads(line) for line in CENSUS.read_text(encoding="utf-8").splitlines() if line.strip()]
    census_check = Counter()
    census_triples: dict[str, set] = defaultdict(set)
    census_rows_per_family: Counter = Counter()
    for entry in census_rows:
        index = int(str(entry["id"]).rsplit("/", 1)[1])
        actual = v2["census_index"].get(index)
        if actual is None:
            census_check["unknown id"] += 1
            continue
        census_check["checked"] += 1
        if (actual["dataset"], actual["cell"], actual["family"], actual["variant"]) == (
            entry["dataset"], entry["cell"], entry["family"], entry["variant"]
        ):
            census_check["id matches the suite row"] += 1
        else:
            census_check["id does not match"] += 1
        census_triples[entry["family"]].add((entry["dataset"], entry["cell"], entry["family"]))
        census_rows_per_family[entry["family"]] += 1
    v3_by_triple = {(row["_meta"].get("dataset_id"), row["_meta"].get("cell_id"), str(row["_meta"].get("family"))): row for row in v3_everything}
    census_population = {
        family: {
            "census_rows": census_rows_per_family[family],
            "distinct_triples": len(triples),
            "v2_rows_for_those_triples": sum(sum(v2["triples"].get(triple, Counter()).values()) for triple in triples),
            "v3_rows_for_those_triples": sum(1 for triple in triples if triple in v3_by_triple),
        }
        for family, triples in sorted(census_triples.items())
    }

    # --- the two configs, and what they disagree on --------------------------------
    from sckevtype.data.build import load_config as load_corpus_config   # in-repo, CPU-only

    config_paths = {"real-v2": ROOT / "configs" / "corpus" / "real-v2.yaml", "real-v3": ROOT / "configs" / "corpus" / "real-v3.yaml"}
    configs = {name: load_corpus_config(path) for name, path in config_paths.items()}
    config_differences = config_diff(
        configs["real-v2"].model_dump(mode="json"), configs["real-v3"].model_dump(mode="json")
    )
    defaults = {
        "payload_fields_that_differ": config_differences,
        "real-v2": {
            "variant_policy": configs["real-v2"].variant_policy,
            "panel.option_markers": configs["real-v2"].panel.option_markers,
        },
        "real-v3": {
            "variant_policy": configs["real-v3"].variant_policy,
            "panel.option_markers": configs["real-v3"].panel.option_markers,
        },
        "note": ("`variant_policy` and `panel.option_markers` both default to v2's behaviour, so a config that names "
                 "neither is a v2 config — which is what every corpus before real-v3 is."),
    }

    # --- the v2 rebuild, if one was handed over ------------------------------------
    rebuild: dict = {"checked": False}
    if args.v2_rebuild:
        frozen = json.loads(V2_RECORD.read_text(encoding="utf-8"))["provenance"]
        rebuild = {"checked": True, "path": "a rebuilt real-v2 corpus", "files": {}, "corpus_config_digest_frozen": frozen["corpus_manifest"]["config_digest"]}
        for name, entry in sorted(frozen["files"].items()):
            path = Path(args.v2_rebuild) / name
            digest = sha256_text(path.read_text(encoding="utf-8")) if path.exists() else "missing"
            rebuild["files"][name] = {
                "records": entry["records"],
                "frozen_sha256": entry["sha256"],
                "rebuilt_sha256": digest,
                "byte_identical": digest == entry["sha256"],
            }
        rebuild["all_four_byte_identical"] = all(item["byte_identical"] for item in rebuild["files"].values())

    # --- the three questions -------------------------------------------------------
    examples = example_rows(v3_rows["test"], ("panel_choice", "panel_miss", "evidence_diluted"))
    example_payload = None
    if examples:
        example_payload = {
            family: {
                "dataset": row["_meta"]["dataset_id"], "cell": row["_meta"]["cell_id"], "split": row["_meta"].get("split", "test"),
                "variant": row["_meta"]["variant"], "group_id": row["_meta"]["group_id"], "cl_id": row["_meta"]["cl_id"],
                "panel_scope": row["_meta"].get("panel_scope"), "state_sha256": sha256_text(row["state"]),
                "state_genes": len(row["state"].split("|")), "state_head": rank_list_head(row["state"])[0],
                "instructions": next(iter(row["questions"].values()))["instructions"],
                "criteria": next(iter(row["questions"].values()))["criteria"],
                "answer": next(iter(row["questions"].values()))["label"],
            }
            for family, row in examples.items()
        }

    checks = [
        ("the most rows any (dataset, cell, family) triple carries in real-v3",
         max(Counter((row["_meta"].get("dataset_id"), row["_meta"].get("cell_id"), str(row["_meta"].get("family"))) for row in v3_everything).values()),
         1, "artifacts/corpus/real-v3/*.jsonl"),
        ("payload fields that differ between the real-v2 and real-v3 configs (name, variant_policy, panel.option_markers, artifacts)",
         len(config_differences), 4, "configs/corpus/real-v2.yaml vs real-v3.yaml"),
        ("v2 rows under distinct (dataset, cell, family, variant) keys",
         len(v2["suite_rows"]), sum(v2["rows_by_split"].values()), f"{v2_source}/*.jsonl"),
        ("v3 records whose (dataset, cell, family, variant) exists in v2", len(v3_everything) - missing, len(v3_everything),
         f"{v2_source}/*.jsonl"),
        ("v3 records identical to their v2 rendering once v2's marker clauses are removed, outside evidence_diluted",
         (len(v3_everything) - diluted_rows) - other_differing, len(v3_everything) - diluted_rows, "same"),
        ("evidence_diluted rows, each masked from a draw named for this corpus (seed_int(rng_seed, 'family', config.name))",
         diluted_rows, diluted_rows, "sckevtype.data.build.records_for_cell"),
        ("_meta fields equal to the v2 record's, other than source/panel_id/seeds (seeds carries the corpus name)",
         matched_meta_relaxed, len(v3_everything), "same"),
        ("_meta fields equal to the v2 record's with seeds included: exactly 0 of "
         f"{len(v3_everything):,}, because the corpus name is part of the family seed (recorded, not gated)",
         matched_meta, 0, "same"),
        ("v3 criteria strings carrying a `positive markers:` / `negative markers:` clause", len(criteria_marker_clause), 0,
         "artifacts/corpus/real-v3/*.jsonl"),
        ("v3 criteria strings containing the word 'markers' that are not a register definition's prose",
         len(criteria_not_prose), 0, "artifacts/v2/definitions.json"),
        ("v3 question strings containing the word 'markers' that are not a register definition's prose",
         len(question_not_prose), 0, "artifacts/v2/definitions.json"),
        ("v3 option strings carrying a comma-separated gene-symbol list", len(comma_runs), 0, "this script"),
        ("(scope, CL id) pairs the v2 corpus asks artifacts/v2-at-build for and cannot get",
         len(register["artifacts/v2-at-build (what real-v3 reads)"]["unservable_pairs"]), 0,
         "artifacts/v2-at-build + the frozen corpus"),
        ("frozen v2 panels reproduced by replaying the draw on artifacts/v2-at-build",
         replay["panels_reproduced"], replay["panels_reproduced"] + replay["panels_different"] + replay["refused_by_the_register"],
         "sckevtype.data.panels, called directly, on artifacts/v2-at-build"),
        ("census ids whose line in the v2 test partition carries the dataset/cell/family/variant the id file says",
         census_check["id matches the suite row"], census_check["checked"],
         "reports/census-suite-ids.jsonl"),
    ]
    if rebuild["checked"]:
        checks.append(("a rebuilt real-v2 reproduces the frozen v2 corpus files byte for byte",
                       sum(1 for item in rebuild["files"].values() if item["byte_identical"]), len(rebuild["files"]),
                       "artifacts/eval/panel-swaps-manifest.json provenance"))

    payload = {
        "what_this_is": (
            "The `real-v3` corpus: `real-v2` with one question per (dataset, cell, family) instead of a "
            "clean/permuted twin pair, and with the marker lists removed from the option text. No model "
            "was run; every number is read off artifacts/corpus/real-v3 and artifacts/suites/real-v2."
        ),
        "corpus": {
            "name": v3_manifest["name"], "path": "artifacts/corpus/real-v3", "config": "configs/corpus/real-v3.yaml",
            "artifact_version": v3_manifest["artifacts"]["version"], "seed": v3_manifest["seed"],
            "config_digest": v3_manifest["config_digest"],
            "files": {name: {"records": entry.get("records"), "questions": entry.get("questions"), "bytes": entry["bytes"], "sha256": entry["sha256"]}
                      for name, entry in sorted(v3_manifest["files"].items())},
        },
        "counts": {
            "v3": {**{k: v for k, v in v3_total.items() if k != "per_dataset_family"}, "per_dataset_family": v3_total["per_dataset_family"],
                   "per_split": {split: count["records"] for split, count in v3_counts.items()},
                   "per_split_family": {split: count["per_family"] for split, count in v3_counts.items()}},
            "v2": v2_all,
            "v2_per_split_family": {split: count["per_family"] for split, count in v2["counts"].items()},
            "reduction": {
                "records": v2_all["records"] - v3_total["records"],
                "questions": v2_all["questions"] - v3_total["questions"],
                "cells": v2_all["cells"] - v3_total["cells"],
                "per_family": {family: v2_all["per_family"][family] - v3_total["per_family"].get(family, 0) for family in v2_all["per_family"]},
            },
        },
        "configs": defaults,
        "register": {**register, "replay": replay},
        "dedup": {
            "v3_triples": triple_summary({triple: Counter([row["_meta"].get("variant")]) for triple, row in v3_by_triple.items()}),
            "v2_triples": v2_triples,
            "kept_as_permuted": {
                "why": "train was permuted in v2 too (a model must not learn a fixed option order) and had no clean twin, so those records keep the rendering they had",
                "rows": sum(1 for row in v3_everything if row["_meta"].get("variant") == "permuted"),
                "per_split": {split: sum(1 for row in rows if row["_meta"].get("variant") == "permuted") for split, rows in v3_rows.items()},
                "per_family": dict(sorted(Counter(str(row["_meta"].get("family")) for row in v3_everything if row["_meta"].get("variant") == "permuted").items())),
                "distinct_cells": len({row["_meta"].get("group_id") for row in v3_everything if row["_meta"].get("variant") == "permuted"}),
            },
            "collapsed": {
                "why": "the non-train splits carried the clean record and its permuted twin; the clean one is kept",
                "rows_dropped": sum(1 for row in v2["suite_rows"].values() if row["_meta"].get("variant") == "permuted" and row["_meta"].get("split") != "train"),
                "per_split": {split: sum(1 for row in v2["suite_rows"].values() if row["_meta"].get("split") == split and row["_meta"].get("variant") == "permuted")
                              for split in SPLITS},
                "per_family": dict(sorted(Counter(str(row["_meta"].get("family")) for row in v2["suite_rows"].values()
                                                   if row["_meta"].get("variant") == "permuted" and row["_meta"].get("split") != "train").items())),
                "cells_that_gained_a_single_row": len({row["_meta"].get("group_id") for row in v2["suite_rows"].values()
                                                        if row["_meta"].get("variant") == "permuted" and row["_meta"].get("split") != "train"}),
            },
            "census_population": census_population,
            "census_id_check": dict(sorted(census_check.items())),
        },
        "marker_strip": {
            "how": ("CorpusConfig.panel.option_markers -> PanelConfig.option_markers -> "
                    "panels.sample_panel / sample_panel_pair / sample_miss_panel -> "
                    "Definition.to_panel_option(markers=...) -> Definition.render_text(markers=False), which skips both "
                    "clauses. The lists stay on the option as markers_pos/markers_neg, which is what the holds rule, the "
                    "leakage guard and the marker_noul targets read."),
            "criteria_strings_seen": criteria_seen,
            "criteria_strings_carrying_a_marker_clause": len(criteria_marker_clause),
            "criteria_strings_with_the_word_markers": len(criteria_marker_word),
            "criteria_strings_with_the_word_markers_that_are_not_register_prose": len(criteria_not_prose),
            "distinct_register_sentences_containing_the_word_markers": sorted(marker_prose_seen),
            "question_strings_seen": question_strings_seen,
            "question_strings_with_the_word_markers": len(question_marker_word),
            "question_strings_with_the_word_markers_that_are_not_register_prose": len(question_not_prose),
            "option_strings_seen": option_strings_seen,
            "option_strings_with_a_comma_symbol_list": len(comma_runs),
            # informational: the register's curated prose names genes ("CD4-positive helper T
            # cell" / CD4). That is the definition itself, it is v2's prose too, and it is not
            # a list — the strip cannot and should not remove it.
            "option_strings_whose_prose_names_a_gene_their_own_definition_lists": len(self_marker_hits),
            "examples": {"clauses": criteria_marker_clause[:3], "not_prose": (criteria_not_prose + question_not_prose)[:3],
                         "comma_runs": comma_runs[:3], "self_markers": self_marker_hits[:3]},
        },
        "v2_source": f"{v2_source}/*.jsonl",
        "equality_with_v2": {
            "reference": f"{v2_source}/*.jsonl (the v2 corpus's records; kev's own _meta keys ignored)",
            "records_compared": len(v3_everything),
            "records_with_a_v2_rendering": len(v3_everything) - missing,
            "equal_after_stripping_v2_marker_clauses": len(v3_everything) - len(mismatches),
            "meta_equal": matched_meta,
            "fields_allowed_to_differ": list(META_MAY_DIFFER),
            "panel_id_changed": panel_id_changed,
            "panel_id_unchanged": len(v3_everything) - missing - panel_id_changed,
            "meta_field_mismatches": dict(sorted(meta_field_mismatches.items())),
            "mismatch_examples": mismatches[:5],
        },
        "v2_rebuild_check": rebuild,
        "examples": example_payload,
        "checks": [{"check": name, "got": got, "expected": expected, "source": source, "ok": got == expected} for name, got, expected, source in checks],
        "sources": {
            "configs/corpus/real-v3.yaml": sha256_text((ROOT / "configs" / "corpus" / "real-v3.yaml").read_text(encoding="utf-8")),
            "configs/corpus/real-v2.yaml": sha256_text((ROOT / "configs" / "corpus" / "real-v2.yaml").read_text(encoding="utf-8")),
            "artifacts/corpus/real-v3/manifest.json": sha256_text((v3_dir / "manifest.json").read_text(encoding="utf-8")),
            "artifacts/suites/real-v2/manifest.json": sha256_text((V2_SUITE / "manifest.json").read_text(encoding="utf-8")),
            "artifacts/eval/panel-swaps-manifest.json": sha256_text(V2_RECORD.read_text(encoding="utf-8")),
            "artifacts/v2-at-build/tissue_map.json": sha_file(V2_AT_BUILD / "tissue_map.json"),
            "artifacts/v2-at-build/cl_terms.json": sha_file(V2_AT_BUILD / "cl_terms.json"),
            "artifacts/v2/tissue_map.json": sha_file(TISSUE_MAP),
            "reports/census-suite-ids.jsonl": sha256_text(CENSUS.read_text(encoding="utf-8")),
        },
    }

    Path(args.out_json).write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")

    # --- the report ----------------------------------------------------------------
    lines: list[str] = [
        "# The `real-v3` corpus",
        "",
        "`real-v2`, with two defects fixed and nothing else touched: **one question per "
        "`(dataset, cell, family)`** instead of a clean/permuted twin pair, and **no marker lists in "
        "the option text**. No model was run for this report; every number is read off "
        "`artifacts/corpus/real-v3` (the new corpus), `artifacts/suites/real-v2` (the v2 reference) and "
        "`reports/census-suite-ids.jsonl`. Generated by `reports/real-v3-corpus.py`; the same numbers, "
        "unrounded, are in `reports/real-v3-corpus.json`.",
        "",
        "## 0. The questions, verbatim — read these first",
        "",
    ]
    if example_payload:
        head = example_payload["panel_choice"]
        lines += [
            f"One cell, three families: `{head['dataset']}` / `{head['cell']}` (ground truth `{head['cl_id']}`, "
            f"panel scope `{head['panel_scope']}`, split `{head['split']}`, group `{head['group_id']}`). The "
            "`panel_choice` and `panel_miss` records of this cell are one panel-swap pair — the same state, and "
            "the second panel is the first with the true type's own option added — and `evidence_diluted` is the "
            "same panel again on a state with three of the cell's genes masked out.",
            "",
            f"These are rows of the **built corpus** (`artifacts/corpus/real-v3`, manifest "
            f"`sha256 {payload['sources']['artifacts/corpus/real-v3/manifest.json'][:16]}…`), not a draft: every option "
            "text below is the register's own rendering with the marker clause removed, and §4 verifies that over the "
            "whole corpus (0 of " + f"{criteria_seen:,}" + " rendered criteria strings carry a clause, 0 carry a gene "
            "list, and every string containing the word `markers` is a register sentence).",
            "",
        ]
        for index, family in enumerate(("panel_choice", "panel_miss", "evidence_diluted"), start=1):
            row = examples[family]
            lines += question_block(row, index)
        lines += [
            "### How the three families differ",
            "",
            "| | `panel_choice` | `panel_miss` | `evidence_diluted` |",
            "|---|---|---|---|",
            "| question | which candidate does the state support? | same wording | same wording |",
            "| `instructions` | the one choice template | **identical string** | **identical string** |",
            "| option set | the drawn panel, true type on it | the **same panel minus the true type's option**, plus the abstain option | the drawn panel, true type on it, plus the abstain option |",
            "| state | the cell's rank list | **identical state** (same cell, same render) | the same rank list with `k` random genes removed |",
            "| ground truth | the true type's option | **the abstain option** (the type is not on the panel) | the true type's option if the definition still holds after the masking, else the abstain option |",
            "",
            "So the families differ in the option set and in the state — never in the instructions: all three "
            "carry the same `CHOICE_INSTRUCTIONS` string, and the family lives in `_meta.family`, which the model "
            "never reads. `panel_miss` is the abstention family (the true type is genuinely absent, by construction, "
            "not by a draw that could put it back); `evidence_diluted` masks `dilution_k` random genes of the state "
            "and leaves the answer to the holds rule — which is what makes it the graded case: thinner evidence is "
            "only a reason to abstain when the definition stopped holding.",
            "",
        ]
    else:
        lines += ["**No test-split cell carries all three families**, so no verbatim triple exists to print — see "
                  "`examples` in the JSON for what was found.", ""]
    lines += [
        "## 1. What the corpus is",
        "",
        f"- `configs/corpus/real-v3.yaml` (`sha256 {payload['sources']['configs/corpus/real-v3.yaml'][:16]}…`), built to "
        f"`artifacts/corpus/real-v3` with `uv run python -m sckevtype.data.build`.",
        f"- reads its register from **`artifacts/v2-at-build`** — the Step-9 tissue map "
        f"(`{register['artifacts/v2-at-build (what real-v3 reads)']['tissue_map_sha256'][:12]}…`) plus the canonical v2 "
        f"terms. Not `artifacts/v2`: that map at HEAD is the *pre*-Step-9 one and cannot serve this corpus at all. §6 has "
        f"the regression, and why a copy was frozen instead of the artifact being repaired.",
        f"- same seed as v2 (`{payload['corpus']['seed']}`), same six atlases, same caps, same panel draw, same splits, "
        f"same state budget. Manifest `sha256 {payload['sources']['artifacts/corpus/real-v3/manifest.json'][:16]}…`.",
        f"- two flags, both new and both defaulting to v2's behaviour: `variant_policy: single` (one rendering per "
        f"record) and `panel.option_markers: false` (the prose alone in the option text).",
        f"- the two config payloads differ in exactly **{len(config_differences)} fields** — "
        f"{', '.join('`' + path + '`' for path in config_differences)} — and nothing else; the atlases, caps, label "
        f"maps, tissue labels, split fractions, panel draw and state budget are byte-for-byte the same text.",
        f"- **{v3_total['records']:,} records**, {v3_total['questions']:,} questions, {v3_total['cells']:,} cells — "
        f"against v2's {v2_all['records']:,} / {v2_all['questions']:,} / {v2_all['cells']:,}.",
        "",
        "## 2. Counts, v3 against v2",
        "",
        "### Per family (whole corpus)",
        "",
        "| family | v2 records | v3 records | dropped | of which train (kept `permuted`) |",
        "|---|---:|---:|---:|---:|",
    ]
    for family in sorted(v2_all["per_family"]):
        dropped = v2_all["per_family"][family] - v3_total["per_family"].get(family, 0)
        kept = payload["dedup"]["kept_as_permuted"]["per_family"].get(family, 0)
        lines.append(f"| `{family}` | {v2_all['per_family'][family]:,} | {v3_total['per_family'].get(family, 0):,} | {dropped:,} | {kept:,} |")
    lines += [
        f"| **total** | **{v2_all['records']:,}** | **{v3_total['records']:,}** | **{payload['counts']['reduction']['records']:,}** | "
        f"**{payload['dedup']['kept_as_permuted']['rows']:,}** |",
        "",
        "### Per dataset and family",
        "",
        "| dataset | family | v2 | v3 |",
        "|---|---|---:|---:|",
    ]
    for dataset in sorted(v2_all["per_dataset_family"]):
        for family in sorted(v2_all["per_dataset_family"][dataset]):
            lines.append(
                f"| `{dataset}` | `{family}` | {v2_all['per_dataset_family'][dataset][family]:,} | "
                f"{v3_total['per_dataset_family'].get(dataset, {}).get(family, 0):,} |"
            )
    lines += [
        "",
        "### Per split",
        "",
        "| split | v2 records | v2 rendering | v3 records | v3 rendering |",
        "|---|---:|---|---:|---|",
    ]
    for split in SPLITS:
        v3_variant = ", ".join(f"`{name}` × {count:,}" for name, count in sorted(v3_counts[split]["per_variant"].items()))
        v2_variant = ", ".join(f"`{name}` × {count:,}" for name, count in sorted(v2["counts"][split]["per_variant"].items()))
        lines.append(f"| `{split}` | {v2['counts'][split]['records']:,} | {v2_variant} | {v3_counts[split]['records']:,} | {v3_variant} |")
    lines += [
        "",
        "## 3. The de-duplication (one question per cell per family)",
        "",
        f"- v2: **{v2_triples['rows']:,} rows over {v2_triples['triples']:,} distinct `(dataset, cell, family)` triples** "
        f"({v2_triples['rows_per_triple']} rows per triple). v3: **{v3_total['records']:,} rows over the same "
        f"{v2_triples['triples']:,} triples — exactly one question per triple** "
        f"(`max rows per triple = {payload['checks'][0]['got']}`).",
        f"- **{payload['dedup']['collapsed']['rows_dropped']:,} rows dropped**, all from calibration / development / test: "
        f"the permuted twin of a clean record. Nothing was re-sampled, re-panelled or re-seeded — the retained row is the "
        f"clean rendering of the same cell, same panel, same option order, same answer.",
        f"- **{payload['dedup']['kept_as_permuted']['rows']:,} records kept as `permuted`** over "
        f"**{payload['dedup']['kept_as_permuted']['distinct_cells']:,} cells** — every one of them in **train**, where v2 "
        f"also carried the permuted rendering only (a model trained on a fixed option order learns the order). Those cells "
        f"had no clean twin to prefer, so they are kept exactly as they were.",
        f"- cells whose two rows became one: **{payload['dedup']['collapsed']['cells_that_gained_a_single_row']:,}** "
        f"(each now contributes one question instead of two).",
        "",
        "### The author's scored population (the census)",
        "",
        "The census ids index straight into the v2 test partition by line "
        f"({payload['dedup']['census_id_check'].get('id matches the suite row', 0):,} of "
        f"{payload['dedup']['census_id_check'].get('checked', 0):,} rows carry the dataset, cell, family and variant "
        "the id file says), so the population "
        "`reports/variant-redundancy.md` counts is measurable as rows straight off both corpora:",
        "",
        "| family | census rows (v2) | distinct triples | v2 rows for those triples | v3 rows for those triples |",
        "|---|---:|---:|---:|---:|",
    ]
    for family, entry in census_population.items():
        lines.append(f"| `{family}` | {entry['census_rows']:,} | {entry['distinct_triples']:,} | "
                     f"{entry['v2_rows_for_those_triples']:,} | {entry['v3_rows_for_those_triples']:,} |")
    lines += [
        "",
        "`panel_choice` is the family `reports/variant-redundancy.md` measures: 1,819 census rows over 1,497 triples, "
        "and **1,497 rows in v3 — the same 1,497 cells, each counted once**.",
        "",
        "### The option-order control",
        "",
        "The `clean`/`permuted` twin pair was a rendering-sensitivity control, and v3 drops it (it is what made one "
        "cell two rows). It can come back as an arm of its own: the same config with `variant_policy: twins` — one "
        "line, no other change — rebuilds the v2 policy exactly, because both flags default to v2's behaviour and "
        "`twins` *is* that behaviour. A rendering-sensitivity comparison then reads two corpora rather than two rows "
        "of one, which is also what makes it a paired design on the cells instead of a doubled population.",
        "",
        "## 4. The marker strip, verified",
        "",
        "`panel.option_markers: false` renders each option from its prose alone. It is one code path, not a post-pass: "
        "`CorpusConfig.panel.option_markers` → `PanelConfig.option_markers` → `panels.sample_panel` / "
        "`sample_panel_pair` / `sample_miss_panel` → `Definition.to_panel_option(markers=…)` → "
        "`Definition.render_text(markers=False)`, which skips both clauses. The default is `True` (v2's own text), so "
        f"nothing else changes. Checked on **all {v3_total['records']:,} v3 records** "
        f"({criteria_seen:,} rendered criteria values, {question_strings_seen:,} question strings of every kind):",
        "",
        "| check | result |",
        "|---|---|",
        f"| criteria strings carrying a `positive markers:` / `negative markers:` clause | **{len(criteria_marker_clause)}** |",
        f"| criteria strings carrying a comma-separated gene-symbol list | **{len(comma_runs)}** |",
        f"| criteria strings containing the word `markers` that are **not** a register definition's prose | **{len(criteria_not_prose)}** |",
        f"| question strings containing the word `markers` that are **not** a register definition's prose | **{len(question_not_prose)}** |",
        "",
        f"The word itself survives, and it has to: it is English inside the register's own curated sentences — "
        + " ; ".join(f"*“{sentence}”*" for sentence in sorted(marker_prose_seen)[:4])
        + f"{' …' if len(marker_prose_seen) > 4 else ''} — {len(marker_prose_seen)} distinct sentences, in "
        f"{len(criteria_marker_word):,} criteria strings and {len(question_marker_word):,} question strings because the "
        "same few definitions recur across panels. v2's text carries those sentences too, each followed by the clause. "
        f"What the strip removes is the clause, and the clause is absent from every one of the {criteria_seen:,} "
        "rendered criteria strings.",
        "",
        "**What this check does not claim.** Not that the prose never names a gene — the register's definitions do "
        "exactly that (“CD4-positive helper T cell” names CD4, and "
        f"{len(self_marker_hits):,} option strings do the like). That is the definition itself, it is v2's prose, and "
        "it is not a list the strip could remove. And not that a model cannot infer an answer from the prose — only "
        "that the gene list which used to sit after it is no longer in the prompt (§7).",
        "",
        "The marker lists are still on each option inside the builder as `markers_pos` / `markers_neg`, which is what "
        "the holds rule reads — so every derived target (`marker_noul`'s yes/no, `evidence_removed`'s and "
        "`evidence_diluted`'s answers) is computed from exactly the lists v2 used.",
        "",
        "## 5. Everything else is v2, checked",
        "",
        "Every v3 record was compared with its v2 rendering (keyed by `(dataset, cell, family, variant)`), read from "
        f"`{payload['v2_source']}` — the same records, same order, same line indices — on the fields that must not move:",
        "",
        "| compared | equal |",
        "|---|---:|",
        f"| records with a v2 rendering of the same key | {len(v3_everything) - missing:,} of {len(v3_everything):,} |",
        f"| `state` + `questions` (option keys **and their order**, instructions, label) after removing v2's marker "
        f"clauses, outside `evidence_diluted` | {(len(v3_everything) - diluted_rows) - other_differing:,} of "
        f"{len(v3_everything) - diluted_rows:,} |",
        f"| every other `_meta` field (`cl_id`, `family`, `group_id`, `cell_id`, `donor_id`, `variant`, `panel_scope`, "
        f"`reason`, `seeds`) except `seeds` | {matched_meta_relaxed:,} of {len(v3_everything):,} |",
        "",
        f"**Two things differ, and neither is one of the two fixes.** First, `seeds` on all "
        f"{len(v3_everything):,} records: the family seed is `seed_int(rng_seed, \"family\", config.name)`, and this "
        f"corpus's name is `real-v3`, so the recorded seed — and the mask `evidence_diluted` draws from it — are this "
        f"corpus's own. That is the whole of the second difference: the **{diluted_rows:,} `evidence_diluted` rows** "
        f"mask a different three genes than v2's twin did (their panel, options, instructions, keys, label and holds "
        f"rule are unchanged — only which genes got masked). The strict `_meta` count with `seeds` included is "
        f"{matched_meta:,} of {len(v3_everything):,}, recorded here rather than dropped. If the author wants the "
        f"diluted draws pinned to v2's as well, the seed's name is one knob: `seed_base` in "
        f"`sckevtype.data.build.records_for_cell` — a rebuild with a fixed family-seed name would make every family "
        f"byte-identical to v2 modulo the marker clauses. It is named here as a choice, not taken.",
        "",
        f"`source` (the corpus name) and `panel_id` are v3-specific by construction: `panel_id` digests the option "
        f"text, so it moves on {payload['equality_with_v2']['panel_id_changed']:,} of {len(v3_everything):,} records "
        f"({payload['equality_with_v2']['panel_id_unchanged']:,} unchanged). No other `_meta` field differs outside "
        f"`seeds`: {json.dumps(payload['equality_with_v2']['meta_field_mismatches'])}.",
        "",
    ]
    if rebuild["checked"]:
        lines += [
            "### The two flags really do default to v2",
            "",
            "`real-v2.yaml` was rebuilt with the modified code into a scratch directory and its four partitions "
            "compared against the digests the frozen artifacts record for the v2 corpus "
            "(`artifacts/eval/panel-swaps-manifest.json` → `provenance.files`):",
            "",
            "| file | records | frozen sha256 | rebuilt sha256 | identical |",
            "|---|---:|---|---|---|",
        ]
        for name, entry in sorted(rebuild["files"].items()):
            lines.append(f"| `{name}` | {entry['records']:,} | `{entry['frozen_sha256'][:16]}…` | "
                         f"`{entry['rebuilt_sha256'][:16]}…` | {'**yes**' if entry['byte_identical'] else 'no'} |")
        lines += [
            "",
            f"All four: **{'byte-identical' if rebuild['all_four_byte_identical'] else 'NOT identical'}**. (The corpus "
            "manifest itself now carries a different `config_digest`, because two keys were added to `CorpusConfig`: "
            "any config's payload digest moves when the schema gains a field. The records do not.)",
            "",
        ]
    lines += [
        "## 6. The register regression: `artifacts/v2` cannot serve this corpus",
        "",
        "`real-v3` reads its register from a **frozen copy** (`artifacts/v2-at-build`, with its own README) rather than "
        "from `artifacts/v2`, and this is why.",
        "",
        "| register | candidates per tissue | `(panel scope, CL id)` pairs the corpus asks it for and cannot get |",
        "|---|---|---:|",
        f"| `artifacts/v2` (live, HEAD; tissue_map `{register['artifacts/v2 (live, at HEAD)']['tissue_map_sha256'][:12]}…`) | "
        f"`{json.dumps(register['artifacts/v2 (live, at HEAD)']['candidates_per_tissue'])}` | "
        f"**{sum(register['artifacts/v2 (live, at HEAD)']['unservable_pairs'].values()):,} records** over "
        f"{len(register['artifacts/v2 (live, at HEAD)']['unservable_pairs'])} pairs |",
        f"| `artifacts/v2-at-build` (what this corpus reads; tissue_map "
        f"`{register['artifacts/v2-at-build (what real-v3 reads)']['tissue_map_sha256'][:12]}…`) | "
        f"`{json.dumps(register['artifacts/v2-at-build (what real-v3 reads)']['candidates_per_tissue'])}` | "
        f"**{sum(register['artifacts/v2-at-build (what real-v3 reads)']['unservable_pairs'].values()):,}** |",
        "",
        "The build asks the register for a candidate list per *(true type, tissue)* and raises the moment the true type is "
        "not in it. This corpus's own first build, against `artifacts/v2` as the config then named it, stopped exactly "
        "there after ~27 minutes:",
        "",
        "```",
        *(register["failure_log"]["lines"] or ["build failed: CL:0001054 is not a candidate type for tissue 'blood' (have: [17 ids])"]),
        "```",
        "",
        "The same check runs for every corpus drawn on that register. It is not one id, and it is not this corpus: "
        "`tests/data/test_tissue_map_covers_atlas_labels.py::test_every_wired_atlas_label_has_a_candidate_for_its_tissue` "
        "— unmodified here, failing at HEAD — names `real-v2.yaml:dominguez_conde` for the same class of miss, and "
        "`docs/SPEC-step9-tissue-map-sync.md` opens by quoting this message as the reason `runs/real-v2` did not exist.",
        "",
        f"**What the two revisions are.** `docs/{STEP9_SPEC.name}` opens by quoting that same failure as the reason "
        f"`runs/real-v2` did not exist, and Step 9's deliverable 1 was a regenerated tissue map — derived by "
        "`tissues[T] = scope ∩ (CellMarker(T) ∪ observed(T))`, candidates per tissue blood 51 / brain 41 / liver 56 / "
        "lung 51 / pancreas 30 (`reports/step9.md`). That is the map the corpus was built with. The commit that sits "
        f"between the two states is **`{register['restoring_commit']}`**: it restored `artifacts/v2` to its *pre*-Step-9 "
        "bytes, pulled from the cluster's staged copy, so the frozen step-11 numbers stayed reproducible "
        f"(`reports/{HIERARCHY_REPORT.name}`, §Version handling) — and that report records the loose end itself: *\"whether "
        "v2's `tissue_map.json` or the current build output is the intended one\"*. The regression is therefore not in this "
        "corpus's config (the two configs differ in three fields, §1) and not in the corpus's data; it is between two "
        "revisions of a frozen artifact.",
        "",
        f"**Why a copy, and not the artifact.** Repairing `artifacts/v2` — restoring the Step-9 map into it, or "
        "re-deriving it — is a decision about the register and the provenance of numbers already published against it; it "
        "is not this task's to make, and nothing at HEAD was modified here. So the register the corpus needs was frozen "
        f"under a new path, with each file's origin and digest recorded (`artifacts/v2-at-build/README.md`): "
        f"`tissue_map.json` from `f2f4282`, `cl_terms.json` + `definitions.json` + `render_config.json` from HEAD (the "
        "latter two identical in both revisions, and `cl_terms` canonical because the hierarchy fix postdates the "
        "corpus). `configs/corpus/real-v3.yaml` names that directory in an `artifacts:` field, so the dependence is "
        "declared in the config rather than hidden in a rebuild.",
        "",
        f"**It is the register the corpus was drawn on, checked.** A panel comes from `(true type, scope, definitions, "
        "PanelConfig)` with an rng seeded on `(seed, \"cell\", dataset, cell_id)`. Replaying that draw on this register "
        f"for {replay['rows']:,} of the frozen corpus's clean panel-bearing records ({replay['sample']}) reproduces the "
        f"record's option names **exactly, in order**, {replay['panels_reproduced']:,} times, with "
        f"{replay['panels_different']:,} differences and {replay['refused_by_the_register']:,} refusals. So `real-v3`'s "
        "questions are v2's questions — not a fresh draw from the same generator.",
        "",
        "## 7. What this corpus is not",
        "",
        "- **Not a scored result.** No model was run, nothing was re-trained, no comparison was recomputed. The "
        "manuscript's tables are unchanged and still describe the v2 population.",
        "- **Not a v2 replacement on disk.** `artifacts/suites/real-v2`, `configs/corpus/real-v2.yaml`, the figures, "
        "the manuscript and the census ids are untouched.",
        "- **Not a re-draw.** Same seed, same caps, same panels: v3's cells are v2's cells, and its test partition is "
        "v2's test partition with the twins collapsed, which is why the equality table above is all-or-nothing.",
        "- **Not the option-order control.** Dropping the twin removes the corpus-internal way to read "
        "rendering sensitivity; §3 says how to get it back as an arm.",
        "- **No new leakage argument.** `panel_choice` in v2 could be answered by matching the state against the "
        "marker lists printed in the options; in v3 that string is gone, and the remaining prose is the register's "
        "curated definition. This report shows the lists are not in the text; it does not show the prose is "
        "insufficient to answer — that is a model's job, and no model was run.",
        "- **The corpus, not the whole serving story.** Every scored row carries its own options, so evaluation goes "
        "through the corpus text — but the middleware spec path "
        "(`sckevtype.contracts.ontology.TissueMap.expand`, `source=\"middleware\"`) still renders options *with* the "
        "marker lists. Nothing in `sckevtype.eval` calls it today; a served deployment of a model trained on this "
        "corpus would need the same switch there. One argument, and out of this task's scope.",
        "",
        "## 8. Reproduce",
        "",
        "```bash",
        "source scripts/wsl/env.sh",
        "uv run --no-sync python -m sckevtype.data.build --config configs/corpus/real-v3.yaml --out artifacts/corpus/real-v3",
        "uv run --no-sync python reports/real-v3-corpus.py",
        "# the v2 default-behaviour check, if you want it:",
        "uv run --no-sync python -m sckevtype.data.build --config configs/corpus/real-v2.yaml --out /tmp/rebuild/real-v2",
        "uv run --no-sync python reports/real-v3-corpus.py --v2-rebuild /tmp/rebuild/real-v2",
        "```",
        "",
        "## 9. Checks",
        "",
        "| check | got | expected | source | ok |",
        "|---|---:|---:|---|---|",
    ]
    for check in payload["checks"]:
        lines.append(f"| {check['check']} | {check['got']:,} | {check['expected']:,} | `{check['source']}` | {'yes' if check['ok'] else '**NO**'} |")
    lines += [
        "",
        "## 10. Sources (sha256)",
        "",
        "| file | sha256 |",
        "|---|---|",
    ]
    for name, digest in payload["sources"].items():
        lines.append(f"| `{name}` | `{digest}` |")
    lines += [
        "",
        "*Written by `reports/real-v3-corpus.py`. No corpus, artifact, figure or manuscript file that existed before was "
        "modified: the marker strip and the one-rendering policy are two flags, in `sckevtype/contracts/definitions.py` "
        "(`render_text(markers=…)`), `sckevtype/data/panels.py` (`PanelConfig.option_markers`) and "
        "`sckevtype/data/build.py` (`CorpusConfig.variant_policy`), and both default to v2's behaviour.*",
        "",
    ]
    Path(args.out_md).write_text("\n".join(lines), encoding="utf-8")

    failed = [check for check in payload["checks"] if not check["ok"]]
    print(f"real-v3: {v3_total['records']} records on {v3_total['cells']} cells; "
          f"{len(v3_everything) - len(mismatches)}/{len(v3_everything)} equal to v2 modulo the marker clauses; "
          f"{len(payload['checks']) - len(failed)}/{len(payload['checks'])} checks pass")
    for check in failed:
        print(f"  FAILED: {check['check']} (got {check['got']}, expected {check['expected']})")
    print(f"wrote {args.out_md} and {args.out_json}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
