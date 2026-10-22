#!/usr/bin/env python3
"""Extract the corpus's own panel-swap pairs into one paired artifact (E4.1).

The claim this artifact exists to make checkable: *hold the cell fixed, change only the
input panel, and the right answer changes while the model's confidence does not.* The
frozen corpus already contains that contrast — it does not have to be generated.

**Where the pairing already is.** ``sckevtype.data.build.records_for_cell`` calls
``sample_panel_pair`` per cell. When the draw is not a miss, the cell gets one
``panel_choice`` record and nothing else. When the draw *is* a miss (``panel_miss_p``,
0.25), it gets **two** records from one draw, linked by ``_meta.group_id``:

* ``panel_miss``      — the starved arm: the true type's option is **not** on the panel,
  so the correct answer is the abstain option;
* ``panel_choice``    — the informative arm: the same panel **plus the true type's own
  option**, so the definition naming the true type's markers is on the list and the
  correct answer is that option.

Both arms are built from the same ``CellRecord`` and render the state with the same
:func:`~sckevtype.contracts.render.render_rank_list` call, so the two records differ in
the option list and in nothing else. ``variant_records`` then applies the permutation
policy: train carries the permuted presentation only, the other splits carry the clean
record and its permuted twin. A pair is therefore keyed by ``(split, group_id, variant)``
and both arms always share the variant.

So this script **extracts and formalizes** — it does not sample, perturb or re-render
anything. Every row is traceable to a line of the frozen corpus by ``state_ref`` and
``state_sha256``, and every derived field is recomputed from the frozen definitions and
the state the corpus stores, not asserted.

What the manifest reports about the extraction (all measured, none assumed):

* how many pairs are complete, and which invariants each complete pair was checked
  against (same state, sibling adds exactly the true option, both answers, one abstain
  wording, ...);
* how many rows are unpaired, split by *why*: a ``panel_miss`` row with no sibling
  (structurally impossible — it would be a corpus bug) and the ``panel_choice`` rows
  whose draw was not a miss (structurally unpaired: that cell has no starved arm in the
  corpus at all, by construction);
* per-family / per-split / per-dataset / per-variant counts, cross-checked against the
  corpus and suite manifests' own family tables;
* the strata the paper will read the swap at: whether the informative arm is *answerable*
  (the state actually satisfies the true definition), how much of the true type's
  evidence the starved panel still names elsewhere, and the panel sizes on each side.

Everything is deterministic and carries no timestamp: the same corpus bytes rebuild the
same artifact bytes.

    uv run python scripts/eval/build_panel_swaps.py
    uv run python scripts/eval/build_panel_swaps.py --corpus runs/real-v2 --out artifacts/eval/panel-swaps.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from sckevtype.contracts.definitions import DefinitionSet
from sckevtype.contracts.freeze import file_digest, text_digest
from sckevtype.contracts.render import FilterConfig, parse_rank_list
from sckevtype.data.families import HoldsRule

SPLITS = ("train", "calibration", "development", "test")
#: arm -> the family the corpus stores that arm under. The informative arm is the
#: ``panel_choice`` sibling ``panel_miss_pair`` builds, the starved arm is the miss.
ARM_FAMILY = {"informative": "panel_choice", "starved": "panel_miss"}
ARM_ORDER = {"informative": 0, "starved": 1}
ARM_NOTE = {
    "informative": (
        "the true type's own option is on the panel, so the definition that names its "
        "markers is offered and the correct answer is that option"
    ),
    "starved": (
        "the true type's option is absent from the panel, so no offered definition names "
        "it and the correct answer is the abstain option"
    ),
}
#: Families whose records are the two arms of a swap. Everything else in the corpus is
#: read for the counts (and to prove the scan saw every record) but contributes no row.
PAIRABLE = ("panel_miss", "panel_choice")


def read_jsonl(path: Path):
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def record_summary(record: dict, split: str, *, keep_state: bool) -> dict:
    """One choice record as the fields the pairing needs — never the whole record.

    ``state`` is dropped unless ``keep_state``: both arms of a pair render the same
    state, so only one copy is ever parsed and the other is verified by digest.
    """
    meta = record["_meta"]
    question = record["questions"]["label"]
    criteria = question["criteria"]
    summary = {
        "split": split,
        "dataset": meta["dataset_id"],
        "cell_id": meta["cell_id"],
        "donor": meta["donor_id"],
        "group_id": meta["group_id"],
        "cl_id": meta["cl_id"],
        "panel_id": meta["panel_id"],
        "panel_scope": meta["panel_scope"],
        "variant": meta["variant"],
        "family": meta["family"],
        "record_seed": meta["seeds"].get("record"),
        # served order: the criteria as the corpus stores them, abstain option included
        "names": list(criteria),
        "definitions": criteria,
        "label_key": question["label"],
        "state_sha256": text_digest(record["state"]),
    }
    if keep_state:
        summary["state"] = record["state"]
    return summary


def scan_split(path: Path, split: str, text_by_cl: dict[str, str]) -> tuple[dict, dict, Counter, int, list[int]]:
    """Read one split file once. Returns (miss, choice, per-family counts, n records, text checks).

    The text check walks every option the two arms offer and compares its wording with the
    frozen definition's own rendering: it is what lets a row's marker lists — which come
    from ``definitions.json`` — be claimed as the markers the model actually read. Returns
    ``[options_checked, options_mismatched]``.
    """
    miss: dict[tuple, dict] = {}
    choice: dict[tuple, dict] = {}
    families: Counter = Counter()
    text_checked = 0
    text_mismatched = 0
    n = 0
    for record in read_jsonl(path):
        n += 1
        family = record["_meta"]["family"]
        families[family] += 1
        if family not in PAIRABLE:
            continue
        summary = record_summary(record, split, keep_state=family == "panel_miss")
        for name, text in summary["definitions"].items():
            expected = text_by_cl.get(name)
            if expected is None:
                continue                      # the abstain option: not a cell type, no definition
            text_checked += 1
            if expected != text:
                text_mismatched += 1
        key = (summary["group_id"], summary["variant"])
        target = miss if family == "panel_miss" else choice
        if key in target:
            raise SystemExit(f"{path}: two {family} records for {key} — the pairing key is not unique")
        target[key] = summary
    return miss, choice, families, n, [text_checked, text_mismatched]


def check_pair(miss: dict, present: dict, defs: DefinitionSet, names_by_cl: dict) -> tuple[list[str], dict]:
    """Every invariant the pairing must satisfy, each checked rather than assumed.

    Returns ``(problems, facts)``: ``problems`` empty means the pair is complete, and
    ``facts`` holds the values the later derivation reuses.
    """
    problems: list[str] = []
    true_cl = miss["cl_id"]
    true_def = defs.by_cl_id.get(true_cl)
    if true_def is None:
        return [f"no frozen definition for {true_cl}"], {}

    true_name = true_def.name
    miss_names = miss["names"]
    present_names = present["names"]
    abstain_key = miss["label_key"]
    facts = {
        "true_name": true_name,
        "abstain_key": abstain_key,
        "miss_names": miss_names,
        "present_names": present_names,
    }

    # 1. the cell is held fixed: same state, same cell, same donor, same panel scope
    if miss["state_sha256"] != present["state_sha256"]:
        problems.append("the two arms render different states")
    for field in ("cell_id", "donor", "dataset", "group_id", "panel_scope", "variant", "cl_id", "record_seed"):
        if miss[field] != present[field]:
            problems.append(f"{field} differs across the arms: {miss[field]!r} != {present[field]!r}")

    # 2. only the panel changed: one option is added, nothing else moves
    added = [name for name in present_names if name not in set(miss_names)]
    dropped = [name for name in miss_names if name not in set(present_names)]
    if dropped:
        problems.append(f"the informative panel drops option(s) {dropped}")
    if len(added) != 1:
        problems.append(f"the informative panel adds {len(added)} options, not 1: {added}")
    else:
        facts["added_name"] = added[0]
        if added[0] != true_name:
            problems.append(f"the added option is {added[0]!r}, not the true type {true_name!r}")
    if true_name in set(miss_names):
        problems.append("the starved panel already contains the true type")
    if true_name not in set(present_names):
        problems.append("the informative panel does not contain the true type")
    if miss["panel_id"] == present["panel_id"]:
        problems.append("both arms carry the same panel_id")
    if len(present_names) - len(miss_names) != 1:
        problems.append("panel sizes do not differ by exactly one")

    # 3. the abstain option: one wording on both arms, drawn by the factory, and the
    #    answer each arm is graded against is the one the corpus stores in `label`
    if abstain_key not in miss_names or abstain_key not in present_names:
        problems.append(f"abstain key {abstain_key!r} is not in both option lists")
    if abstain_key in names_by_cl:
        problems.append(f"abstain key {abstain_key!r} collides with a cell type name")
    if miss["label_key"] != abstain_key:
        problems.append(f"the starved arm's answer is {miss['label_key']!r}, not the abstain key")
    if present["label_key"] != true_name:
        problems.append(f"the informative arm's answer is {present['label_key']!r}, not the true type")
    else:
        facts["answer_changed"] = True
    if miss["definitions"][abstain_key] != present["definitions"][abstain_key]:
        problems.append("the abstain option's wording differs across the arms")

    # 4. the added option is the true type's, and every option the swap keeps is the same
    #    *definition* on both sides — the swap must not reword anything it does not change.
    for name in miss_names:
        if name == abstain_key:
            continue
        if name not in present["definitions"] or miss["definitions"][name] != present["definitions"][name]:
            problems.append(f"option {name!r} is worded differently on the two arms")

    # 5. the added option's identity: the name the two arms disagree about must be the one
    #    the frozen definitions give the cell's own CL id, or the "added marker" fields
    #    below would be describing some other type's evidence.
    if "added_name" in facts and names_by_cl.get(true_name) != true_cl:
        problems.append(f"the true definition names {true_name!r} for {true_cl}, not {names_by_cl.get(true_name)!r}")
    return problems, facts


def markers_named_by(names: list[str], defs_by_name: dict, abstain_key: str) -> set[str]:
    """Every marker named by the definitions offered on a panel (abstain option excluded)."""
    out: set[str] = set()
    for name in names:
        if name == abstain_key:
            continue
        definition = defs_by_name.get(name)
        if definition is not None:
            out.update(definition.pos_markers)
            out.update(definition.neg_markers)
    return out


def swap_facts(
    miss: dict,
    present: dict,
    facts: dict,
    defs: DefinitionSet,
    names_by_cl: dict,
    defs_by_name: dict,
    filter_cfg: FilterConfig,
    holds: HoldsRule,
) -> dict:
    """The pair-level quantities, all read off the frozen definitions and the state."""
    true_def = defs.by_cl_id[miss["cl_id"]]
    symbols = [symbol for symbol, _ in parse_rank_list(miss["state"], filter_cfg)]
    present_set = set(symbols)
    deciding = list(true_def.pos_markers) + list(true_def.neg_markers)
    deciding_present = [marker for marker in deciding if marker in present_set]
    named_by_starved = markers_named_by(facts["miss_names"], defs_by_name, facts["abstain_key"])
    named_by_informative = markers_named_by(facts["present_names"], defs_by_name, facts["abstain_key"])
    return {
        "kind": "panel_swap",
        "held_fixed": ["cell", "state", "donor", "abstain_wording", "panel_scope", "variant"],
        "changed": ["panel_id", "option_list"],
        "added_option_cl_id": miss["cl_id"],
        "added_option_name": facts["true_name"],
        "added_option_is_true_type": True,
        "added_markers_pos": list(true_def.pos_markers),
        "added_markers_neg": list(true_def.neg_markers),
        "added_markers": deciding,
        # what each added gene *is* to the true type, read off the frozen definition: a
        # positive marker or a negative one (no marker of this register is both).
        "added_marker_role": {
            **{marker: "pos" for marker in true_def.pos_markers},
            **{marker: "neg" for marker in true_def.neg_markers},
        },
        "added_marker_present": {marker: marker in present_set for marker in deciding},
        "added_markers_present": deciding_present,
        "n_added_markers": len(deciding),
        "n_added_markers_present": len(deciding_present),
        "added_marker_in_state": bool(deciding_present),
        # `HoldsRule` is the corpus's own definition-satisfaction rule (min_pos=1, max_neg=0):
        # true when the state supports the true type's definition at all.
        "true_pos_present": [marker for marker in true_def.pos_markers if marker in present_set],
        "true_neg_present": [marker for marker in true_def.neg_markers if marker in present_set],
        "true_definition_holds": holds.holds(present_set, true_def),
        # how much of the true type's evidence the *starved* panel still names elsewhere:
        # the panel-level measure of how starved it really is.
        "true_markers_named_by_starved_panel": sorted(set(deciding) & named_by_starved),
        "n_true_markers_named_by_starved_panel": len(set(deciding) & named_by_starved),
        "n_true_markers_named_by_informative_panel": len(set(deciding) & named_by_informative),
        "answer_informative": facts["true_name"],
        "answer_starved": facts["abstain_key"],
        "answer_changed": True,
        "state_symbols_total": len(symbols),
    }


def arm_rows(
    miss: dict,
    present: dict,
    facts: dict,
    swap: dict,
    names_by_cl: dict,
) -> dict[str, dict]:
    """The two rows of one pair: same cell, same state, one option between them."""
    true_name = facts["true_name"]
    abstain_key = facts["abstain_key"]
    pair_id = f"{miss['split']}/{miss['group_id']}/{miss['variant']}"
    rows = {}
    for arm in ("informative", "starved"):
        source = present if arm == "informative" else miss
        names = source["names"]
        label = names.index(true_name if arm == "informative" else abstain_key)
        abstain = names.index(abstain_key)
        options = [name for name in names if name != abstain_key]
        rows[arm] = {
            "id": f"{pair_id}/{arm}",
            "pair_id": pair_id,
            "sibling_id": f"{pair_id}/{'starved' if arm == 'informative' else 'informative'}",
            "question": "label",
            "split": miss["split"],
            "dataset": miss["dataset"],
            "donor": miss["donor"],
            "cell_id": miss["cell_id"],
            "group_id": miss["group_id"],
            "true_cl_id": miss["cl_id"],
            "true_type_name": true_name,
            "family": ARM_FAMILY[arm],
            "arm": arm,
            "arm_note": ARM_NOTE[arm],
            "variant": miss["variant"],
            "size": len(names),
            "label": label,
            "label_key": names[label],
            "abstain": abstain,
            "abstain_key": abstain_key,
            "abstain_text": source["definitions"][abstain_key],
            "abstain_position": abstain,
            "abstain_last": abstain == len(names) - 1,
            "abstain_source": "factory",
            # the index space `label` and `abstain` live in: every key of the served question,
            # option order as the corpus stores it (the abstain option among them). `option_names`
            # below is the same list with the abstain removed.
            "option_order": names,
            # the metric's own axis-1 label rule, computed offline: the answer is the abstain
            # option, i.e. the panel cannot decide this cell.
            "axis1_positive": int(label == abstain),
            # model outputs: not producible from the corpus. A scoring pass fills these; until
            # then they are null and must not be read as a measurement.
            "none_p": None,
            "panel_id": source["panel_id"],
            "panel_scope": source["panel_scope"],
            "n_options": len(options),
            "option_cl_ids": [names_by_cl[name] for name in options],
            "option_names": options,
            "true_option_present": true_name in options,
            "true_option_position": options.index(true_name) if true_name in options else None,
            "swap": swap,
            "state_sha256": miss["state_sha256"],
            "state_ref": {
                "corpus": "runs/real-v2",
                "split": miss["split"],
                "dataset": miss["dataset"],
                "cell_id": miss["cell_id"],
                "variant": miss["variant"],
                "family": ARM_FAMILY[arm],
            },
            "record_seed": miss["record_seed"],
        }
    return rows


def scan_and_pair(corpus: Path, defs: DefinitionSet, filter_cfg: FilterConfig, holds: HoldsRule, text_by_cl: dict[str, str]):
    """One pass over every split: collect the arms, join them, check the invariants."""
    names_by_cl = {definition.name: definition.cl_id for definition in defs.definitions}
    defs_by_name = {definition.name: definition for definition in defs.definitions}
    pairs: list[dict] = []
    rows: list[dict] = []
    incomplete: list[dict] = []
    read = {split: 0 for split in SPLITS}
    text_checks = [0, 0]
    state_equal = 0
    per_family: dict[str, Counter] = {split: Counter() for split in SPLITS}
    for split in SPLITS:
        path = corpus / f"{split}.jsonl"
        miss, choice, families, n, checks = scan_split(path, split, text_by_cl)
        read[split] = n
        text_checks[0] += checks[0]
        text_checks[1] += checks[1]
        per_family[split] = families
        unpaired_miss = sorted(set(miss) - set(choice))
        for key in unpaired_miss:
            incomplete.append({"split": split, "group_id": key[0], "variant": key[1], "arm": "starved",
                               "why": "panel_miss with no panel_choice sibling in the same split"})
        for key in sorted(set(miss) & set(choice)):
            if miss[key]["state_sha256"] == choice[key]["state_sha256"]:
                state_equal += 1
            problem, facts = check_pair(miss[key], choice[key], defs, names_by_cl)
            if problem:
                incomplete.append({"split": split, "group_id": key[0], "variant": key[1],
                                   "why": "; ".join(problem)})
                continue
            swap = swap_facts(miss[key], choice[key], facts, defs, names_by_cl, defs_by_name, filter_cfg, holds)
            pair_rows = arm_rows(miss[key], choice[key], facts, swap, names_by_cl)
            rows.extend([pair_rows["informative"], pair_rows["starved"]])
            pairs.append({
                "pair_id": pair_rows["informative"]["pair_id"],
                "split": split,
                "dataset": miss[key]["dataset"],
                "donor": miss[key]["donor"],
                "group_id": key[0],
                "variant": key[1],
                "true_cl_id": miss[key]["cl_id"],
                "true_type_name": facts["true_name"],
                "panel_scope": miss[key]["panel_scope"],
                "panel_id_informative": choice[key]["panel_id"],
                "panel_id_starved": miss[key]["panel_id"],
                "size_informative": len(choice[key]["names"]),
                "size_starved": len(miss[key]["names"]),
                "true_definition_holds": swap["true_definition_holds"],
                "n_added_markers": swap["n_added_markers"],
                "n_added_markers_present": swap["n_added_markers_present"],
                "n_true_markers_named_by_starved_panel": swap["n_true_markers_named_by_starved_panel"],
                "answer_informative": swap["answer_informative"],
                "answer_starved": swap["answer_starved"],
            })
    rows.sort(key=lambda row: (row["split"], row["dataset"], row["cell_id"], row["variant"], ARM_ORDER[row["arm"]]))
    return pairs, rows, incomplete, read, per_family, text_checks, state_equal


def cross_checks(corpus: Path, read: dict, per_family: dict, pairs: list[dict]) -> tuple[dict, list[str]]:
    """Check the scan against the two manifests that already describe this corpus."""
    checks: dict = {}
    problems: list[str] = []
    corpus_manifest = json.loads((corpus / "manifest.json").read_text(encoding="utf-8"))
    checks["corpus_manifest"] = {
        "path": str(corpus / "manifest.json"),
        "sha256": file_digest(corpus / "manifest.json"),
        "config_digest": corpus_manifest["config_digest"],
    }
    total_family = Counter()
    for families in per_family.values():
        total_family.update(families)
    checks["corpus_manifest"]["per_family_expected"] = dict(sorted(corpus_manifest["counts"]["per_family"].items()))
    checks["corpus_manifest"]["per_family_read"] = dict(sorted(total_family.items()))
    for family, expected in corpus_manifest["counts"]["per_family"].items():
        if total_family.get(family, 0) != expected:
            problems.append(f"family {family}: read {total_family.get(family, 0)} records, manifest says {expected}")
    if sum(read.values()) != corpus_manifest["counts"]["records"]:
        problems.append(f"read {sum(read.values())} records, manifest says {corpus_manifest['counts']['records']}")

    suite = Path("artifacts/suites/real-v2/manifest.json")
    if not suite.exists():
        checks["suite_manifest"] = {"path": str(suite), "status": "absent"}
    else:
        suite_manifest = json.loads(suite.read_text(encoding="utf-8"))
        per_split: dict[str, dict] = {}
        for split in SPLITS:
            expected = suite_manifest["files"][f"{split}.jsonl"]["per_family"]
            got = per_family[split]
            per_split[split] = {"expected": dict(sorted(expected.items())), "read": dict(sorted(got.items()))}
            for family, count in expected.items():
                if got.get(family, 0) != count:
                    problems.append(f"{split}/{family}: read {got.get(family, 0)}, suite manifest says {count}")
        checks["suite_manifest"] = {
            "path": str(suite),
            "sha256": file_digest(suite),
            "per_split": per_split,
            "note": "the suite manifest's per-file family table is the reference the scan is checked against",
        }

    # the miss rate is the corpus config's panel_miss_p: a miss is what draws a pair
    n_choice_rows = sum(
        count for split in SPLITS for family, count in per_family[split].items() if family == "panel_choice"
    )
    checks["miss_rate"] = {
        "pairs_over_choice_rows": round(len(pairs) / n_choice_rows, 6) if n_choice_rows else None,
        "n_pairs": len(pairs),
        "n_choice_rows": n_choice_rows,
        "config_note": "configs/corpus/real-v2.yaml panel.panel_miss_p = 0.25",
    }
    # the abstraction the task's audit asks for: every panel_miss row must have a sibling, so the
    # number of pairs equals the number of panel_miss rows the scan read.
    n_miss_rows = sum(count for split in SPLITS for family, count in per_family[split].items() if family == "panel_miss")
    checks["pair_coverage"] = {
        "panel_miss_rows_read": n_miss_rows,
        "pairs_built": len(pairs),
        "starved_rows_without_sibling": n_miss_rows - len(pairs),
    }
    return checks, problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", default="runs/real-v2", help="the frozen corpus directory (default: runs/real-v2)")
    parser.add_argument("--artifacts", default="artifacts/v2", help="the frozen contract artifacts (default: artifacts/v2)")
    parser.add_argument("--out", default="artifacts/eval/panel-swaps.jsonl", help="the row file to write")
    parser.add_argument("--manifest", default="artifacts/eval/panel-swaps-manifest.json", help="the manifest to write")
    parser.add_argument("--pairs-out", default="artifacts/eval/panel-swaps-pairs.jsonl",
                        help="one line per pair: the pair table (the same facts the manifest aggregates)")
    args = parser.parse_args()

    corpus = Path(args.corpus)
    artifacts = Path(args.artifacts)
    defs = DefinitionSet.load(str(artifacts / "definitions.json"))
    filter_cfg = FilterConfig.model_validate(json.loads((artifacts / "render_config.json").read_text(encoding="utf-8")))
    holds = HoldsRule()
    text_by_cl = {definition.name: definition.render_text() for definition in defs.definitions}

    pairs, rows, incomplete, read, per_family, text_checks, state_equal = scan_and_pair(
        corpus, defs, filter_cfg, holds, text_by_cl
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    checks, problems = cross_checks(corpus, read, per_family, pairs)
    per_arm = Counter(row["arm"] for row in rows)
    per_dataset = Counter(row["dataset"] for row in rows)
    per_split = Counter(row["split"] for row in rows)
    per_variant = Counter(row["variant"] for row in rows)
    per_family_rows = Counter(row["family"] for row in rows)
    per_scope = Counter(row["panel_scope"] for row in rows)
    pair_per_split = Counter(pair["split"] for pair in pairs)
    pair_per_variant = Counter(pair["variant"] for pair in pairs)
    holds_stratum = Counter(pair["true_definition_holds"] for pair in pairs)
    leak_stratum = Counter("none" if pair["n_true_markers_named_by_starved_panel"] == 0 else "some" for pair in pairs)
    marker_stratum = Counter(
        "no marker in state" if pair["n_added_markers_present"] == 0 else "some markers in state"
        for pair in pairs
    )
    sizes_inf = sorted(pair["size_informative"] for pair in pairs)
    sizes_star = sorted(pair["size_starved"] for pair in pairs)

    def quantiles(values: list[int]) -> dict:
        if not values:
            return {}
        return {
            "min": values[0],
            "median": values[len(values) // 2],
            "max": values[-1],
            "mean": round(sum(values) / len(values), 3),
        }

    manifest = {
        "name": "panel-swaps",
        "version": "v1",
        "kind": "paired-panel-swap",
        "claim": (
            "hold the cell fixed, change only the input panel: the correct answer changes "
            "(the true type's option, or the abstain option) while the state, the cell and "
            "the abstain wording do not"
        ),
        "paths": {
            "corpus": str(corpus),
            "artifacts": str(artifacts),
            "rows": str(out_path),
            "script": "scripts/eval/build_panel_swaps.py",
            "plan": "docs/PLAN-audit-steps-by-expected-result.md E4.1",
        },
        "command": "uv run python scripts/eval/build_panel_swaps.py",
        "next_step": {
            "for": "E4.2 / E4.3 — ask every model the same question on both arms and read the pair",
            "how": [
                "load a row's state from the frozen corpus by state_ref (split/dataset/cell_id/variant/family) "
                "and check it against state_sha256",
                "rebuild the served question from option_order (definitions from artifacts/v2/definitions.json, "
                "digested in provenance.artifacts) — the row keeps the reports/axis1-rows-*.jsonl field names, so a "
                "scoring harness written against that dump reads these rows unchanged",
                "score both arms; join on pair_id; Δcorrect is a certainty (axis1_positive differs on every pair) "
                "and Δconfidence is the measurement the paper wants",
            ],
            "gate": "confidence moves far less than the answer does (docs/PLAN-audit-steps-by-expected-result.md E4)",
        },
        "provenance": {
            "corpus_manifest": checks["corpus_manifest"],
            "files": {
                f"{split}.jsonl": {
                    "sha256": json.loads((corpus / "manifest.json").read_text(encoding="utf-8"))["files"][f"{split}.jsonl"]["sha256"],
                    "records": read[split],
                }
                for split in SPLITS
            },
            "artifacts": {
                name: file_digest(artifacts / name)
                for name in ("definitions.json", "render_config.json")
            },
            "definitions_digest": defs.digest(),
            "filter_digest": filter_cfg.digest(),
            "holds_rule": holds.model_dump(),
            "script_sha256": file_digest("scripts/eval/build_panel_swaps.py"),
            "output_sha256": file_digest(out_path),
            "byte_identical_rebuild": "no timestamp, no absolute path: the same corpus bytes rebuild the same rows",
        },
        "pairing": {
            "rule": (
                "one panel_miss record plus the panel_choice record sharing (split, group_id, variant); "
                "the pairing is the corpus's own panel_miss_pair, extracted — nothing was re-sampled"
            ),
            "key": ["split", "group_id", "variant"],
            "source_families": {"informative": "panel_choice", "starved": "panel_miss"},
            "arms": {"informative": ARM_NOTE["informative"], "starved": ARM_NOTE["starved"]},
            "invariants_checked": [
                "both arms render the same state (state_sha256 equal)",
                "cell_id, donor, dataset, group_id, panel_scope, variant, cl_id, record seed equal across arms",
                "the informative panel adds exactly one option and it is the true type's",
                "the starved panel does not contain the true type",
                "the panel_id differs between the arms",
                "both arms share one abstain key and one abstain wording",
                "the abstain key is not a cell type name",
                "every option kept on both sides is worded identically",
                "the corpus's stored answer is the abstain option on the starved arm and the true type on the informative arm",
            ],
            "invariant_failures": len(incomplete),
            "state_identical_pairs": state_equal,
            "swap_semantics": (
                "the swap adds no gene to the input and removes none: the state is byte-identical on both arms "
                "(state_sha256, checked on every pair). What changes is the option list — the true type's option, "
                "i.e. the one definition that names it, is on the informative panel and absent from the starved one. "
                "`swap.added_markers` is what that option makes relevant: each gene with its role in the frozen "
                "definition (pos/neg) and whether the state expresses it."
            ),
            "answer_change": {
                "rows_with_changed_answer": sum(1 for pair in pairs if pair["answer_informative"] != pair["answer_starved"]),
                "n_pairs": len(pairs),
            },
        },
        "counts": {
            "pairs_complete": len(pairs),
            "rows": len(rows),
            "rows_per_pair": 2,
            "cells_with_a_pair": len({pair["group_id"] for pair in pairs}),
            "datasets": len({row["dataset"] for row in rows}),
            "per_arm": dict(sorted(per_arm.items())),
            "per_family": dict(sorted(per_family_rows.items())),
            "per_split": dict(sorted(per_split.items())),
            "per_variant": dict(sorted(per_variant.items())),
            "per_dataset": dict(sorted(per_dataset.items())),
            "per_panel_scope": dict(sorted(per_scope.items())),
            "pairs_per_split": dict(sorted(pair_per_split.items())),
            "pairs_per_variant": dict(sorted(pair_per_variant.items())),
            "source_records_read": {split: read[split] for split in SPLITS},
            "source_records_per_family": {
                split: dict(sorted(per_family[split].items())) for split in SPLITS
            },
        },
        "unpaired": {
            "starved_rows_without_informative_sibling": sum(
                1 for item in incomplete if item["arm"] == "starved"
            ),
            "rows_with_a_failed_invariant": sum(1 for item in incomplete if item["arm"] != "starved"),
            "informative_rows_without_starved_sibling": (
                sum(count for split in SPLITS for family, count in per_family[split].items() if family == "panel_choice")
                - len(pairs)
            ),
            "why": (
                "a cell whose panel draw was not a miss gets one panel_choice record and no starved arm at all — "
                "sample_panel_pair returns no sibling there, so the corpus has no pair for it by construction, not by "
                "loss in this extraction. Those rows are the 19,886 abstain-less choice questions the axis-1 runbook "
                "counts."
            ),
            "detail": incomplete[:50],
        },
        "strata": {
            "_note": (
                "the strata the swap has to be read at. `informative_arm_answerable` says whether the "
                "informative arm is answerable from the state at all; `true_markers_in_state_*` counts "
                "how many of the true type's deciding markers the state expresses; "
                "`true_markers_named_by_starved_panel_hist` counts how many of those the *starved* panel "
                "still names through some other option's definition — the starved arm removes the true "
                "type's option, not every mention of its markers."
            ),
            "informative_arm_answerable": {
                "rule": "HoldsRule(min_pos=1, max_neg=0) over the state against the true type's frozen definition",
                **{("holds" if key else "does_not_hold"): value for key, value in holds_stratum.items()},
            },
            "true_markers_in_state": dict(sorted(marker_stratum.items())),
            "true_markers_in_state_hist": dict(sorted(Counter(pair["n_added_markers_present"] for pair in pairs).items())),
            "starved_panel_names_true_markers": dict(sorted(leak_stratum.items())),
            "true_markers_named_by_starved_panel_hist": dict(
                sorted(Counter(pair["n_true_markers_named_by_starved_panel"] for pair in pairs).items())
            ),
            "added_markers_hist": dict(sorted(Counter(pair["n_added_markers"] for pair in pairs).items())),
            "panel_size_informative": quantiles(sizes_inf),
            "panel_size_starved": quantiles(sizes_star),
            "panel_size_pairs": {
                f"{a}->{b}": count for (a, b), count in
                sorted(Counter((p["size_starved"], p["size_informative"]) for p in pairs).items())
            },
        },
        "validation": {
            "cross_checks": checks,
            "option_text_vs_frozen_definitions": {
                "checked": text_checks[0],
                "mismatched": text_checks[1],
                "note": (
                    "every option offered by the two arms, compared with the frozen definition's own "
                    "render_text(): the marker lists a row carries describe the text the model reads"
                ),
            },
            "problems": problems,
            "status": "ok" if not problems and not incomplete else "problems found — read `problems` and `unpaired.detail`",
        },
        "row_schema": {
            "note": (
                "a superset of the reports/axis1-rows-*.jsonl dump shape: id, question, split, dataset, donor, family, "
                "variant, size, label, abstain, abstain_source, abstain_key, abstain_text, abstain_position and "
                "abstain_last all keep that dump's meaning, and `axis1_positive` is that dump's `positive` (the "
                "metric's own label rule). scripts/axis1-recompute.py's row reader passes such rows through "
                "unchanged, but its statistics need `none_p`, which only a scoring pass can produce."
            ),
            "model_outputs_present": False,
            "model_outputs_note": (
                "none_p is null on every row: producing it needs a forward pass over the model, which this extraction "
                "does not do. axis1_positive is the metric's own offline label rule (the answer is the abstain option) "
                "and is complete."
            ),
            "state": "not embedded; state_ref + state_sha256 locate and verify it in the frozen corpus",
        },
    }
    if args.pairs_out:
        pairs_path = Path(args.pairs_out)
        pairs_path.parent.mkdir(parents=True, exist_ok=True)
        with open(pairs_path, "w", encoding="utf-8", newline="\n") as handle:
            for pair in sorted(pairs, key=lambda item: item["pair_id"]):
                handle.write(json.dumps(pair, sort_keys=True) + "\n")
        manifest["paths"]["pairs"] = str(pairs_path)
        manifest["provenance"]["pairs_sha256"] = file_digest(pairs_path)

    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"pairs complete : {len(pairs)}")
    print(f"rows           : {len(rows)}  ({dict(sorted(per_arm.items()))})")
    print(f"per family     : {dict(sorted(per_family_rows.items()))}")
    print(f"per split      : {dict(sorted(per_split.items()))}")
    print(f"per dataset    : {dict(sorted(per_dataset.items()))}")
    print(f"unpaired       : starved w/o sibling {manifest['unpaired']['starved_rows_without_informative_sibling']}, "
          f"failed invariants {manifest['unpaired']['rows_with_a_failed_invariant']}, "
          f"informative w/o starved sibling {manifest['unpaired']['informative_rows_without_starved_sibling']}")
    print(f"answerable     : {dict(sorted(holds_stratum.items()))} (informative arm, state supports the definition)")
    print(f"starved leak   : {dict(sorted(leak_stratum.items()))} (a true marker named elsewhere on the starved panel)")
    print(f"written        : {out_path}")
    print(f"written        : {manifest_path}")
    if problems or incomplete:
        print("PROBLEMS:")
        for problem in problems:
            print("  -", problem)
        for item in incomplete[:20]:
            print("  -", item)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
