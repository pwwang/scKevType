#!/usr/bin/env python3
"""Build the learning-curve suites: subsampled training partitions and the census-matched eval suite.

WHAT THIS IS
------------
The learning-curve experiment trains the same 0.8B delta recipe on three fractions of the *training*
partition (25%, 50%, 100%) and scores each on the settled census-matched ``panel_choice`` population.
This script builds the two things that experiment needs, and nothing else:

  <out>/lc25/train.jsonl + manifest.json     a 25% subsample of real-v2's training partition
  <out>/lc50/train.jsonl + manifest.json     a 50% subsample, a superset of the 25% one
  <out>/lc-census-eval/                      an eval suite whose test partition is exactly the 1,819
                                             census-matched panel_choice records, copied verbatim from
                                             real-v2's test partition, plus real-v2's calibration
                                             partition verbatim (the eval CLI fits on it)

The 100% point is real-v2 itself: it is never rewritten, so the 100% runs are the existing
``runs/delta-kev-0.8b-s*`` runs and their digests are recorded here for the record.

THE SUBSAMPLING RULE (deterministic, recorded)
----------------------------------------------
    rank(record) = int(sha256("<SALT>|<SEED>|<record _meta.id>").hexdigest(), 16)
    keep the round(frac * n) records with the smallest rank (ties broken by id)
    the kept records keep the source file's own order

SALT is ``sckevtype-learning-curve-v1`` and SEED is ``20261006`` by default; both are recorded in every
manifest's ``learning_curve`` block, so the subsets are reproducible from the manifest alone. Ranking by
a hash of the record id — rather than slicing the file, or drawing with an RNG whose stream depends on
the order of a call sequence — makes the subsets nested by construction (25% ⊂ 50% ⊂ 100%), independent
of file order, and identical on any machine that reads the same bytes.

WHAT IS *NOT* DONE HERE
-----------------------
Nothing under artifacts/, runs/ or kev/ is written. The parent suite is read-only: every file this
script writes lives under ``--out``. The parent manifest is copied and its ``files`` block is replaced
with entries for the files actually written, so the parent's digests are never restated for bytes that
are not there.

    python scripts/learning-curve/build-suites.py --root <repo> --out scratch/learning-curve/suites
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

DEFAULT_SALT = "sckevtype-learning-curve-v1"
DEFAULT_SEED = 20261006
DEFAULT_PARENT = "artifacts/suites/real-v2"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_lines(path: Path) -> list[bytes]:
    """Every non-empty line, verbatim (no re-serialisation: the bytes are the artifact)."""
    with open(path, "rb") as handle:
        return [line for line in handle.read().splitlines() if line.strip()]


def records_of(lines: list[bytes]) -> list[dict]:
    return [json.loads(line) for line in lines]


def rank(record: dict, salt: str, seed: int) -> tuple[int, str]:
    rid = record["_meta"]["id"]
    value = int(sha256_bytes(f"{salt}|{seed}|{rid}".encode()), 16)
    return value, rid


def select(records: list[dict], frac: float, salt: str, seed: int) -> list[int]:
    """The source indices the fraction keeps, in source order."""
    n = len(records)
    k = round(frac * n)
    order = sorted(range(n), key=lambda i: rank(records[i], salt, seed))
    keep = set(order[:k])
    return [i for i in range(n) if i in keep]


def file_entry(path: Path, lines: list[bytes]) -> dict:
    """The per-file manifest entry: what kev's load_split verifies, plus the counts a reader wants."""
    records = records_of(lines)
    families = Counter(record["_meta"]["family"] for record in records)
    types = Counter(question["type"] for record in records for question in record["questions"].values())
    return {
        "sha256": file_digest(path),
        "bytes": path.stat().st_size,
        "kind": "data",
        "records": len(records),
        "questions": sum(len(record["questions"]) for record in records),
        "per_family": dict(sorted(families.items())),
        "per_question_type": dict(sorted(types.items())),
    }


def measure_context(tokenizer, records: list[dict], parent_context: dict, subset_note: str) -> dict:
    """The parent's budget, with this subset's own ``observed`` measurement under it.

    ``max_state`` / ``max_branch`` / ``max_packed`` are the parent suite's, carried verbatim so every
    fraction of the curve trains under one identical budget -- that is the whole point of holding
    everything but the training-set size fixed. ``observed`` is re-measured on the bytes actually
    written, so the block never claims a measurement of records that are not in the file.
    """
    from sckevtype.contracts import budget as state_budget

    max_state = parent_context["max_state"]
    lengths = state_budget.state_lengths(tokenizer, (record["state"] for record in records))
    summary = state_budget.summarize(lengths)
    rows, packed = state_budget.encoded_lengths(records, tokenizer, max_state=max_state)
    required = state_budget.required_budget(summary["max"], max(rows), max(packed))
    if required > max_state:
        raise SystemExit(
            f"REFUSED: the subset needs max_state {required} but the parent's budget is {max_state}; "
            f"the subsets must train under the parent's budget for the curve to be comparable"
        )
    context = {
        "max_state": max_state,
        "max_branch": parent_context["max_branch"],
        "max_packed": parent_context["max_packed"],
        "truncate": bool(parent_context.get("truncate", False)),
        "source": parent_context.get("source", "measured"),
        "observed": {
            "max_state": summary["max"],
            "max_row": max(rows),
            "max_packed": max(packed),
            "per_split": {"train": summary},
            "n": len(records),
            "over_budget": state_budget.over_budget(lengths, max_state),
            "truncated": 0,
        },
        "budget_note": subset_note,
    }
    if parent_context.get("tokenizer"):
        context["tokenizer"] = dict(parent_context["tokenizer"])
    return context


def write_jsonl(path: Path, lines: list[bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(b"\n".join(lines) + b"\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the learning-curve suites: subsampled training "
                                                 "partitions and the census-matched eval suite.")
    parser.add_argument("--root", default=".", help="repo root")
    parser.add_argument("--parent", default=DEFAULT_PARENT, help="the suite whose partitions are subsampled")
    parser.add_argument("--out", default="scratch/learning-curve/suites", help="where the built suites go")
    parser.add_argument("--ids-from", default="reports/census-suite-ids.jsonl",
                        help="the census population's ids (family panel_choice)")
    parser.add_argument("--fractions", default="0.25,0.50", help="training fractions to write (100% is the parent, untouched)")
    parser.add_argument("--salt", default=DEFAULT_SALT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--eval-suite-name", default="real-v2-lc-census-eval")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    parent_dir = root / args.parent
    out_dir = root / args.out
    parent_manifest_path = parent_dir / "manifest.json"
    parent_manifest = read_json(parent_manifest_path)

    print(f"root            : {root}")
    print(f"parent suite    : {args.parent}  manifest sha256 {file_digest(parent_manifest_path)[:16]}")
    print(f"out             : {args.out}")

    # ---- the parent's own bytes, verified before anything is derived from them -----------------
    parent_files = {}
    for name in ("train.jsonl", "test.jsonl", "calibration.jsonl"):
        path = parent_dir / name
        digest = file_digest(path)
        want = parent_manifest["files"][name]["sha256"]
        status = "ok" if digest == want else "MISMATCH"
        print(f"  {name:18s} {digest[:16]}  expected {want[:16]}  {status}")
        if digest != want:
            raise SystemExit(f"REFUSED: {path} is not the file the parent manifest records")
        parent_files[name] = path
    if parent_manifest.get("cap"):
        raise SystemExit("REFUSED: the parent suite carries a cap block; that is the eval suite, not the training one")

    # ---- the census-matched population ---------------------------------------------------------
    ids_path = root / args.ids_from
    census = [json.loads(line) for line in ids_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    census_ids = sorted({row["id"] for row in census if row["family"] == "panel_choice"})
    print(f"\ncensus ids      : {args.ids_from}  sha256 {file_digest(ids_path)[:16]}  "
          f"{len(census_ids)} panel_choice ids of {len(census)} rows")

    build_record: dict = {
        "what": "the learning-curve suites: subsampled training partitions and the census-matched eval suite",
        "rule": ("rank every source train record by int(sha256('<salt>|<seed>|<_meta.id>').hexdigest(), 16); "
                 "keep the round(frac * n) records with the smallest rank, ties by id; the kept records keep "
                 "the source file's own order"),
        "salt": args.salt,
        "seed": args.seed,
        "parent": {"path": args.parent, "manifest_sha256": file_digest(parent_manifest_path),
                   "name": parent_manifest.get("name"), "version": parent_manifest.get("version")},
        "parent_files": {name: {"sha256": parent_manifest["files"][name]["sha256"],
                                "records": parent_manifest["files"][name]["records"]} for name in parent_files},
        "census_ids": {"path": args.ids_from, "sha256": file_digest(ids_path), "panel_choice_ids": len(census_ids)},
        "suites": {},
    }

    train_lines = read_lines(parent_files["train.jsonl"])
    train_records = records_of(train_lines)
    n = len(train_records)
    print(f"parent train    : {n} records")

    tokenizer = None
    pin = (parent_manifest.get("context") or {}).get("tokenizer")
    if pin:
        from sckevtype.contracts import budget as state_budget
        tokenizer = state_budget.pinned_tokenizer(pin)
        print(f"tokenizer       : {pin['id']} @ {pin['revision'][:12]}")

    # ---- the training fractions ----------------------------------------------------------------
    for frac_text in args.fractions.split(","):
        frac = float(frac_text)
        if not 0 < frac < 1:
            raise SystemExit(f"REFUSED: fraction {frac_text} is not in (0, 1); 100% is the parent suite, untouched")
        keep = select(train_records, frac, args.salt, args.seed)
        name = f"lc{round(frac * 100)}"
        suite_dir = out_dir / name
        lines = [train_lines[i] for i in keep]
        write_jsonl(suite_dir / "train.jsonl", lines)
        entry = file_entry(suite_dir / "train.jsonl", lines)
        records = records_of(lines)
        if tokenizer is None:
            raise SystemExit("REFUSED: the parent manifest records no context.tokenizer, so the subset's "
                             "budget cannot be measured; refusing to copy a measurement that is not of these bytes")
        context = measure_context(
            tokenizer, records, parent_manifest["context"],
            f"max_state/max_branch/max_packed are {args.parent}'s, carried verbatim so every fraction of the "
            f"learning curve trains under one identical budget; observed is this subset's own measurement",
        )
        manifest = json.loads(json.dumps(parent_manifest))
        manifest["name"] = f"{parent_manifest.get('name')}-{name}"
        manifest["files"] = {"train.jsonl": entry}
        manifest["context"] = context
        manifest["learning_curve"] = {
            "what": "a subsampled training partition of the learning-curve experiment",
            "rule": build_record["rule"], "salt": args.salt, "seed": args.seed,
            "fraction": frac, "records": entry["records"], "of_parent_records": n,
            "parent": build_record["parent"],
            "parent_train_sha256": parent_manifest["files"]["train.jsonl"]["sha256"],
            "budget": "parent context.max_state carried verbatim; context.observed re-measured on this subset",
        }
        (suite_dir / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
        print(f"\n{name:6s} fraction {frac}  {entry['records']} of {n} records  "
              f"({entry['records'] / n:.4f})  questions {entry['questions']}")
        print(f"       train.jsonl sha256 {entry['sha256']}")
        print(f"       observed: state {context['observed']['max_state']}  row {context['observed']['max_row']}  "
              f"packed {context['observed']['max_packed']}  budget {context['max_state']}")
        build_record["suites"][name] = {"path": f"{args.out}/{name}", "fraction": frac,
                                        "records": entry["records"], "train_jsonl": entry,
                                        "manifest_sha256": file_digest(suite_dir / "manifest.json")}

    # ---- the census-matched eval suite ----------------------------------------------------------
    test_lines = read_lines(parent_files["test.jsonl"])
    test_records = records_of(test_lines)
    wanted = set(census_ids)
    keep = [i for i, record in enumerate(test_records) if record["_meta"]["id"] in wanted]
    kept_ids = {test_records[i]["_meta"]["id"] for i in keep}
    missing = sorted(wanted - kept_ids)
    if missing:
        raise SystemExit(f"REFUSED: {len(missing)} census panel_choice ids have no record in {args.parent}/test.jsonl "
                         f"(first: {missing[:3]}); the eval suite would not carry the settled population")
    eval_dir = out_dir / "lc-census-eval"
    eval_lines = [test_lines[i] for i in keep]
    write_jsonl(eval_dir / "test.jsonl", eval_lines)
    shutil.copyfile(parent_files["calibration.jsonl"], eval_dir / "calibration.jsonl")
    cal_digest = file_digest(eval_dir / "calibration.jsonl")
    if cal_digest != parent_manifest["files"]["calibration.jsonl"]["sha256"]:
        raise SystemExit("REFUSED: the copied calibration partition is not byte-identical to the parent's")
    test_entry = file_entry(eval_dir / "test.jsonl", eval_lines)
    cal_entry = dict(parent_manifest["files"]["calibration.jsonl"])
    manifest = json.loads(json.dumps(parent_manifest))
    manifest["name"] = args.eval_suite_name
    manifest["files"] = {"calibration.jsonl": cal_entry, "test.jsonl": test_entry}
    manifest["learning_curve"] = {
        "what": ("an eval suite whose test partition is exactly the census-matched panel_choice population: "
                 "the 1,819 records of reports/census-suite-ids.jsonl with family panel_choice, copied "
                 "verbatim from the parent's test partition; calibration is the parent's, copied verbatim"),
        "census_ids": build_record["census_ids"],
        "parent": build_record["parent"],
        "population_n": len(keep),
        "context": "the parent context block, carried verbatim: the serving budget must be the one every other arm was served under",
    }
    (eval_dir / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print(f"\nlc-census-eval  test {test_entry['records']} records  questions {test_entry['questions']}")
    print(f"       test.jsonl sha256 {test_entry['sha256']}")
    print(f"       calibration.jsonl sha256 {cal_digest} (parent's, byte-identical)")
    build_record["suites"]["lc-census-eval"] = {"path": f"{args.out}/lc-census-eval",
                                                "test_jsonl": test_entry,
                                                "calibration_sha256": cal_digest,
                                                "manifest_sha256": file_digest(eval_dir / "manifest.json")}
    build_record["population"] = {"n": len(keep), "ids_sha256": build_record["census_ids"]["sha256"]}

    (out_dir / "build-record.json").write_text(json.dumps(build_record, indent=1) + "\n", encoding="utf-8")
    print(f"\nwrote {args.out}/build-record.json")
    print("BUILD OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
