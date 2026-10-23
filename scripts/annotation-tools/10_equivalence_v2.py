"""Step 10: the equivalence proof for the parameterised pipeline.

Every step of this pipeline gained a `--dir` (and `01`/`02` a `--population`), which is a change
to the machinery that produced the published v2 numbers. The change is licensed only if the
machinery still reproduces those numbers exactly, so this script compares a fresh run of the v2
population (`reports/annotation-tools-repro/scores.json`) against the frozen
`reports/annotation-tools-baseline.json`, field by field, and exits non-zero on any difference.

The four numbers the report quotes as the acceptance test are checked both as the count of
questions answered correctly and as the rate, because a rate that rounds to the published value
while the count differs is a different measurement.

Run (WSL, from the repo root):

    source scripts/wsl/env.sh
    uv run --no-sync python scripts/annotation-tools/10_equivalence_v2.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path("/mnt/f/E/hermes-workspace/scKevType")
FROZEN = ROOT / "reports" / "annotation-tools-baseline.json"
DEFAULT_REPRO = ROOT / "reports" / "annotation-tools-repro"

#: the four readings the equivalence test is stated on, and the per-dataset arm each one picks
READINGS = ("celltypist_default_model", "celltypist_best_arm", "singler_label_main",
            "singler_best_label_level")
#: fields compared inside every reading, in addition to the four headline readings
FIELDS = ("questions", "named_target", "named_target_rate", "on_ballot", "on_ballot_rate",
          "off_register", "off_register_rate", "unmapped", "unmapped_rate", "pruned_declined",
          "pruned_rate", "target_reachable_by_this_vocabulary", "target_reachable_rate",
          "named_target_given_reachable", "named_target_given_reachable_rate",
          "per_dataset_arm")


def compare(label: str, frozen, repro, field: str) -> bool:
    a = frozen.get(field)
    b = repro.get(field)
    ok = a == b
    print(f"  {'ok  ' if ok else 'DIFF'} {label}.{field}: frozen={a!r} repro={b!r}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Prove the parameterised pipeline reproduces v2.")
    parser.add_argument("--repro", default=str(DEFAULT_REPRO), help="the fresh v2 run's directory")
    parser.add_argument("--frozen", default=str(FROZEN), help="the published baseline JSON")
    args = parser.parse_args()
    repro_path = Path(args.repro)
    frozen_path = Path(args.frozen)
    repro_scores = json.loads((repro_path / "scores.json").read_text(encoding="utf-8"))
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))

    ok = True
    print("POPULATION")
    for field in ("questions", "distinct_cells", "per_dataset"):
        a = frozen["population"][field]
        b = repro_scores["population"][field]
        same = a == b
        ok &= same
        print(f"  {'ok  ' if same else 'DIFF'} population.{field}: frozen={a} repro={b}")

    print("\nTHE FOUR READINGS")
    for reading in READINGS:
        print(f"[{reading}]")
        for field in FIELDS:
            ok &= compare(reading, frozen["headline"][reading], repro_scores["pooled"][reading], field)

    print("\nPER ARM, EVERY DECLARED ARM OF THIS POPULATION")
    frozen_arms = {k: v for k, v in frozen["per_arm"]["celltypist"].items()}
    repro_arms = {k: v for k, v in repro_scores["per_arm"]["celltypist"].items()}
    if set(frozen_arms) != set(repro_arms):
        ok = False
        print(f"  DIFF arm sets: only frozen {sorted(set(frozen_arms) - set(repro_arms))}, "
              f"only repro {sorted(set(repro_arms) - set(frozen_arms))}")
    for key in sorted(set(frozen_arms) & set(repro_arms)):
        for field in ("named_target", "named_target_rate", "on_ballot", "off_register", "unmapped",
                      "panel_restricted_correct", "panel_restricted_rate", "cells_scored"):
            ok &= compare(key, frozen_arms[key], repro_arms[key], field)
    frozen_sr = frozen["per_arm"]["singler"]
    repro_sr = repro_scores["per_arm"]["singler"]
    for key in sorted(set(frozen_sr) & set(repro_sr)):
        for field in ("named_target", "named_target_rate", "on_ballot", "off_register", "unmapped",
                      "pruned_declined", "cells_scored"):
            ok &= compare(key, frozen_sr[key], repro_sr[key], field)

    print("\nHEADLINE, AS THE REPORT QUOTES IT")
    for reading in READINGS:
        f = frozen["headline"][reading]
        r = repro_scores["pooled"][reading]
        print(f"  {reading:28s} frozen {f['named_target']:5d}/{f['questions']} = "
              f"{100 * f['named_target_rate']:.1f}%   repro {r['named_target']:5d}/{r['questions']} = "
              f"{100 * r['named_target_rate']:.1f}%")

    print(f"\nEQUIVALENCE {'PASS' if ok else 'FAIL'}"
          f"  (frozen {frozen_path.name}, fresh run {repro_path.name})")
    (repro_path / "equivalence.json").write_text(json.dumps({
        "frozen": str(frozen_path),
        "repro": str(repro_path),
        "verdict": "PASS" if ok else "FAIL",
        "what": "every field of the four published readings, every per-arm count, and the "
                "population, compared between the fresh v2 run and the frozen baseline JSON",
    }, indent=2), encoding="utf-8")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
