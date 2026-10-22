#!/usr/bin/env python3
"""Reproduce the *capped* test population the frozen evaluation was scored on, and prove it.

The frozen headline number (accuracy 0.899) was measured by the step-11 evaluation over
``artifacts/suites/real-v2-eval`` -- a **stratified sample** of the corpus's test split (31,997 of
89,802 records, 24 ``(atlas, family)`` strata, floor 150, seed 20260923). Step 6's baseline numbers
are over the *whole* test split. Those are different populations and must never be put side by side.

That suite directory does not exist on this box (it was built on the cluster), so this script
rebuilds the sampled partition **with the project's own selection code** -- not a second copy of the
rule:

* ``sckevtype.train.suite.corpus_records``   -- kev's own record loader, the same one the suite build uses
* ``sckevtype.train.suite.cap_test_partition`` -- the selection, verbatim; ``seed`` is the corpus
  manifest's own seed, so the draw is re-derivable from the frozen corpus manifest
* ``sckevtype.train.suite._validate``        -- every sampled record is a request kev accepts
* ``kev.suite.write_jsonl``                  -- the *same writer* that formats a frozen partition

The claim that this is the scored population is then **checked against the frozen report** rather
than asserted: ``reports/eval-real-v2-eval.json`` records the sha256 of the ``test.jsonl`` the
evaluation read (``suite.partitions["test.jsonl"]``). If the rebuild's digest differs, the capped
numbers are not the frozen numbers' population and this script fails.

    uv run --no-sync python scripts/baselines/capped_population.py \\
        --out "$TMPDIR/capped-population" [--expect-test-sha256 <hex>]

Writes into ``--out`` (default: the same scratch dir):

* ``test.jsonl``        the capped test partition, byte-identical to the frozen suite's if it matches
* ``cap.json``          the cap block, both digests, and the verdict
* ``record_index.jsonl`` one line per capped record: the key the baseline joins on
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path

ROOT = Path(os.environ.get("SCKEVTYPE_ROOT", "/mnt/f/E/hermes-workspace/scKevType"))

#: The suite the frozen evaluation was scored on, as the frozen report records it.
FROZEN_REPORT = ROOT / "reports" / "eval-real-v2-eval.json"
SUITE_DIR = "artifacts/suites/real-v2-eval"

#: The cap the step-11b build was submitted with (docs/SPEC-step11b-capped-suite.md, and the
#: achieved counts are what reports/step11.md quotes: 31,997 of 89,802 records).
CAP_TEST = 32000
CAP_FLOOR = 150


def frozen_expectations(report_path: Path) -> dict:
    """What the frozen report says it read: suite path, manifest digest, partition digests."""
    report = json.loads(report_path.read_text(encoding="utf-8"))
    suite = report["suite"]
    return {
        "report": str(report_path.relative_to(ROOT)) if report_path.is_relative_to(ROOT) else str(report_path),
        "suite_path_as_recorded": suite["path"],
        "manifest_sha256": suite["manifest_sha256"],
        "test_jsonl_sha256": suite["partitions"]["test.jsonl"],
        "calibration_jsonl_sha256": suite["partitions"]["calibration.jsonl"],
        "reported_rows": report["partitions"]["rows"]["test"]["test"],
        "reported_accuracy": report["accuracy"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", default=str(ROOT / "runs" / "real-v2"))
    parser.add_argument("--out", default=os.environ.get("TMPDIR", "/tmp") + "/capped-population")
    parser.add_argument("--cap-test", type=int, default=CAP_TEST)
    parser.add_argument("--cap-floor", type=int, default=CAP_FLOOR)
    parser.add_argument("--expect-test-sha256", default=None,
                        help="override the frozen digest (default: read it out of reports/eval-real-v2-eval.json)")
    parser.add_argument("--no-validate", action="store_true",
                        help="skip kev.data.materialize on every sampled record (does not change the bytes)")
    args = parser.parse_args()

    # Imported here so --help works without the project venv.
    from kev.suite import write_jsonl

    from sckevtype.contracts.freeze import file_digest
    from sckevtype.train.suite import (
        _validate,
        cap_test_partition,
        corpus_records,
        read_corpus,
    )

    t0 = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"corpus      : {args.corpus}")
    print(f"out         : {out}")
    print(f"cap         : --cap-test {args.cap_test} --cap-floor {args.cap_floor}")
    print()

    print("=== 1. the corpus, re-hashed against its own manifest (read_corpus) ===", flush=True)
    manifest, corpus = read_corpus(args.corpus)
    print(f"  corpus manifest sha256 : {corpus['manifest_sha256']}")
    print(f"  corpus seed            : {manifest.get('seed')}   <- the cap's seed comes from here")
    print(f"  {len(manifest.get('files', {}))} partition(s) re-hashed and re-counted: ok")
    print(f"  [{time.time() - t0:.1f}s]", flush=True)

    print("\n=== 2. the test partition, through kev's own record loader ===", flush=True)
    records = corpus_records(args.corpus, "test")
    print(f"  records available      : {len(records)}")
    print(f"  [{time.time() - t0:.1f}s]", flush=True)

    print("\n=== 3. the cap: sckevtype.train.suite.cap_test_partition (the project's own code) ===", flush=True)
    sampled, cap_block = cap_test_partition(records, cap=args.cap_test, floor=args.cap_floor,
                                            seed=manifest.get("seed"))
    print(f"  achieved               : {cap_block['achieved']} of {cap_block['available']} records")
    print(f"  strata used            : {cap_block['strata_used']}   floor {cap_block['floor']}   seed {cap_block['seed']}")
    print("  panel-size bins (rows capped of rows available):")
    for name, rows in cap_block["panel_size_bins"].items():
        print(f"    {name:>6}  {rows['capped']:>7} of {rows['available']:>7}")

    if not args.no_validate:
        print(f"\n=== 4. every sampled record is a kev request (_validate) ===\n  [{time.time() - t0:.1f}s]", flush=True)
        _validate(sampled, Path(args.corpus) / "test.jsonl")
        print(f"  {len(sampled)} records materialise: ok")

    print(f"\n=== 5. write the partition with kev's own writer ===\n  [{time.time() - t0:.1f}s]", flush=True)
    target = out / "test.jsonl"
    write_jsonl(target, sampled)
    got = file_digest(target)
    print(f"  wrote {target} ({target.stat().st_size} bytes)")

    expectation = frozen_expectations(FROZEN_REPORT)
    expect = args.expect_test_sha256 or expectation["test_jsonl_sha256"]
    matches = got == expect
    print("\n=== 6. is this the population the frozen report was scored on? ===")
    print(f"  frozen report          : {expectation['report']}")
    print(f"  it records the suite   : {expectation['suite_path_as_recorded']}")
    print(f"  its manifest sha256    : {expectation['manifest_sha256']}")
    print(f"  test.jsonl rebuilt     : {got}")
    print(f"  test.jsonl as scored   : {expect}")
    print(f"  VERDICT                : {'IDENTICAL -- this is the scored population' if matches else 'DIFFERS -- NOT the scored population'}")
    if args.expect_test_sha256:
        print("  (digest supplied on the command line, not read from the report)")

    # The joined key the step-6 baseline uses, plus what the cap drew per family/dataset.
    index_path = out / "record_index.jsonl"
    per_dataset, per_family, per_variant = Counter(), Counter(), Counter()
    with index_path.open("w", encoding="utf-8") as handle:
        for record in sampled:
            meta = record["_meta"]
            questions = record["questions"]
            kinds = Counter(str(q.get("type")) for q in questions.values())
            sizes = [len(q.get("criteria") or []) for q in questions.values() if q.get("type") == "choice"]
            handle.write(json.dumps({
                "id": meta["id"],
                "dataset_id": meta.get("dataset_id"),
                "family": meta.get("family"),
                "cell_id": meta.get("cell_id"),
                "variant": meta.get("variant"),
                "donor": meta.get("donor"),
                "question_types": dict(kinds),
                "n_questions": len(questions),
                "choice_sizes": sizes,
            }, ensure_ascii=False) + "\n")
            per_dataset[meta.get("dataset_id")] += 1
            per_family[meta.get("family")] += 1
            per_variant[meta.get("variant")] += 1

    payload = {
        "what_this_is": "the capped test population of artifacts/suites/real-v2-eval, rebuilt with the "
        "project's own selection code and checked against the digest the frozen report recorded",
        "scored_population_selected_by": "sckevtype.train.suite.cap_test_partition(corpus_records('test'), "
        "cap=32000, floor=150, seed=corpus manifest seed)",
        "corpus": {"path": args.corpus, "manifest_sha256": corpus["manifest_sha256"], "seed": manifest.get("seed")},
        "cap": cap_block,
        "frozen_expectation": expectation,
        "rebuilt_test_jsonl": {"path": str(target), "bytes": target.stat().st_size, "sha256": got,
                               "records": len(sampled)},
        "digest_matches_the_frozen_report": matches,
        "counts": {"records": len(sampled), "per_dataset": dict(sorted(per_dataset.items())),
                   "per_family": dict(sorted(per_family.items())), "per_variant": dict(sorted(per_variant.items()))},
        "record_index": str(index_path),
        "seconds": round(time.time() - t0, 1),
    }
    (out / "cap.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n  per dataset : {dict(sorted(per_dataset.items()))}")
    print(f"  per family  : {dict(sorted(per_family.items()))}")
    print(f"  per variant : {dict(sorted(per_variant.items()))}")
    print(f"  index       : {index_path}")
    print(f"  cap block   : {out / 'cap.json'}")
    print(f"\nDONE in {time.time() - t0:.1f}s -- digest_matches_the_frozen_report={matches}")
    return 0 if matches else 1


if __name__ == "__main__":
    raise SystemExit(main())
