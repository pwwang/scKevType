#!/usr/bin/env python3
"""Turn the frozen report plus the seed-variance draws into a per-metric noise floor.

This is the script the whole seed-variance run exists for. We have exactly one trained model of our
own (``runs/train-real-v1``, seed 0) and one variant (arm B). Arm B's scored deltas are small --
axis-1 PRIMARY 0.6171 against 0.6605 (-0.0434), ``panel_choice`` accuracy 0.2419 against 0.2920
(-0.0501) -- and nothing so far says how much of that is arm B's change and how much is the
run-to-run noise of the pipeline itself. Seeds 1-3 retrain the frozen configuration and re-score it,
so for every metric this script prints

    the frozen value, each seed's value, the seed mean, the spread (min-max and sd) and whether the
    frozen value sits inside that range -- i.e. whether the frozen draw is an ordinary member of its
    own family or an outlier in it,

and, when arm B's report is present, arm B's delta against the frozen value beside that same spread.
"arm B moved 0.0434" becomes "arm B moved 0.0434, and three redraws of the same configuration move
by [x, y]".

    uv run python scripts/seedvar-spread.py                      # seeds 1, 2, 3
    uv run python scripts/seedvar-spread.py --armb reports/eval-armb-v1-eval.json
    uv run python scripts/seedvar-spread.py --json reports/seedvar-spread.json

EXIT CODES, because a spread that was not measured must not read like one that was:

    0  every requested seed report is present, every integrity check passed, and the frozen report
       resolved by sha256 and carries the values this exercise is quoted against.
    1  an integrity check failed -- a report is not this draw on the frozen evaluation, or two
       reports are the same checkpoint. The numbers are printed, but not as a noise floor.
    2  usage error.
    3  partial: fewer seed reports than requested. The spread over what is present is printed and
       labelled PARTIAL; it is not yet the family's noise floor.
    4  the spread is complete and passed, but arm B's report failed its own integrity checks, so its
       column was not read.

The frozen report is located by CONTENT. The two names it travels under are tried in turn, and a file
whose sha256 is not e37e5ad1... is named and skipped, never used. A ``reports/eval-real-v2-eval.json``
that is the older step-11 file -- which carries no axis-1 PRIMARY block at all, so a spread read from
it would silently compare against a different statistic -- is exactly the failure this avoids.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any, Callable, Sequence

FROZEN_SHA = "e37e5ad1d1c64fd5b75a2f8b807c954335c924837c0579494d83c1f4b1da5253"
FROZEN_CANDIDATES = ("reports/eval-real-v2-eval.json", "reports/eval-real-v2-eval-axis1.json")
#: The capped evaluation the frozen numbers were taken on, and the training suite every draw in this
#: family was trained on. A report whose suite digests differ is not a member of the family.
CAP_MANIFEST = "5be659143bd4612d16f359e871f7a5698125c6092461e048a5217b4a8cea4e85"
CAP_CALIBRATION = "3ed8672147380ea20b79aef8dee32209e83b204e53897760429d80f7cbe35e3a"
CAP_TEST = "25cf1df6a110fdffcf411683517b05c7236911626a604953d127cf32e3be32ff"
TRAIN_MANIFEST = "559a4a40ae2a16d941e2c7afcfc8984bb477ad79180c0281e00eb188284ab98c"
#: The operating point every report in this family must carry, named rather than defaulted.
OPERATING_POINT = (0.1, 0.05, "lac", 0.05)
#: The frozen values as they are quoted, at 4 decimals. A resolved frozen report that disagrees with
#: any of these is not the file those numbers came from -- a failure, never reconciled silently.
EXPECTED_FROZEN = {
    "accuracy": 0.8985,
    "axis1_primary_auroc": 0.6605,
    "axis1_primary_detection": 0.2161,
    "axis1_pooled_auroc": 0.6114,
    "triage_coverage": 0.1611,
    "triage_risk": 0.4438,
    "family.marker_noul.accuracy": 0.9625,
    "family.evidence_diluted.accuracy": 0.3413,
    "family.panel_choice.accuracy": 0.2920,
    "family.evidence_removed.accuracy": 0.9976,
    "family.out_of_register.accuracy": 0.9941,
    "family.panel_miss.accuracy": 0.9955,
    "family.marker_noul.mean_confidence": 0.9847,
    "family.evidence_diluted.mean_confidence": 0.9709,
    "family.panel_choice.mean_confidence": 0.8548,
    "family.evidence_removed.mean_confidence": 0.9869,
    "family.out_of_register.mean_confidence": 0.9808,
    "family.panel_miss.mean_confidence": 0.9887,
    "dataset.muraro.accuracy": 0.9108,
    "dataset.tabula_sapiens.accuracy": 0.9167,
    "dataset.tabula_muris_senis.accuracy": 0.9123,
    "dataset.siletti.accuracy": 0.8647,
}
FAMILIES = ("marker_noul", "evidence_diluted", "panel_choice",
            "evidence_removed", "out_of_register", "panel_miss")
ATLASES = ("muraro", "tabula_sapiens", "tabula_muris_senis", "siletti")
#: The frozen row dump's population: a draw that dumped a different number of rows scored a different
#: population, which is a fact about the pass and not a seed effect.
FROZEN_ROWS = 24945

Metric = tuple[str, str, str, Callable[[dict], Any], str]   # (group, key, label, getter, format)


def dig(report: dict, path: Sequence[str]) -> Any:
    """Follow ``path`` and return the leaf, or None if any step is absent."""
    value: Any = report
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def at(*path: str) -> Callable[[dict], Any]:
    return lambda report, path=path: dig(report, path)


def block(name: str, field: str) -> Callable[[dict], Any]:
    """``report['breakdowns'][<family|dataset>][name][field]``."""
    section, key = name.split(".")
    return lambda report, section=section, key=key, field=field: dig(
        report, ("breakdowns", section, key, field))


def metrics() -> list[Metric]:
    head: list[tuple[str, str, Callable[[dict], Any], str]] = [
        ("accuracy", "accuracy (n=136,606 rows)", at("accuracy", "accuracy"), "%.4f"),
        ("ece_before", "ECE before T", at("temperature", "before", "ece"), "%.4f"),
        ("ece_after", "ECE after T", at("temperature", "after", "ece"), "%.4f"),
        ("temperature", "temperature T", at("temperature", "temperature"), "%.4f"),
        ("aps_coverage", "APS coverage (alpha 0.10)",
         at("axis2_conformal", "by_method", "aps", "test", "coverage"), "%.4f"),
        ("lac_coverage", "LAC coverage (alpha 0.10)",
         at("axis2_conformal", "by_method", "lac", "test", "coverage"), "%.4f"),
        ("axis1_primary_auroc", "axis-1 PRIMARY auroc",
         at("axis1_panel_adequacy", "primary", "auroc"), "%.4f"),
        ("axis1_primary_detection", "axis-1 PRIMARY detection@FPR.05",
         at("axis1_panel_adequacy", "primary", "detection_rate"), "%.4f"),
        ("axis1_primary_achieved_fpr", "axis-1 PRIMARY achieved FPR",
         at("axis1_panel_adequacy", "primary", "achieved_fpr"), "%.4f"),
        ("axis1_pooled_auroc", "axis-1 POOLED auroc (qualified)",
         at("axis1_panel_adequacy", "auroc"), "%.4f"),
        ("triage_coverage", "triage coverage (at risk 0.4438)", at("triage", "coverage"), "%.4f"),
        ("triage_risk", "triage risk", at("triage", "risk"), "%.4f"),
        ("aurc", "aurc", at("risk_coverage", "aurc"), "%.4f"),
        ("coverage_at_5pct", "coverage@5%error", at("risk_coverage", "coverage_at_5pct_error"), "%.4f"),
        ("sweep_risk_at_50pct", "sweep risk@50%coverage", at("sweep", "risk_at_50pct_coverage"), "%.4f"),
    ]
    out: list[Metric] = [("headline", key, label, getter, spec)
                         for key, label, getter, spec in head]
    for family in FAMILIES:
        out.append(("family", f"family.{family}.accuracy", f"{family} accuracy",
                    block(f"family.{family}", "accuracy"), "%.4f"))
        out.append(("family", f"family.{family}.mean_confidence", f"{family} mean confidence",
                    block(f"family.{family}", "mean_confidence"), "%.4f"))
    for atlas in ATLASES:
        out.append(("atlas", f"dataset.{atlas}.accuracy", f"{atlas} accuracy",
                    block(f"dataset.{atlas}", "accuracy"), "%.4f"))
        out.append(("atlas", f"dataset.{atlas}.mean_confidence", f"{atlas} mean confidence",
                    block(f"dataset.{atlas}", "mean_confidence"), "%.4f"))
    return out


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fmt(value: Any, spec: str) -> str:
    return "n/a" if value is None else spec % float(value)


def fmt_sd(sd: float, spec: str) -> str:
    if sd != sd:
        return "n/a"
    decimals = int(spec.lstrip("%.").rstrip("f") or 4)
    text = f"{sd:.{decimals}f}"
    if float(text) == 0.0 and sd != 0.0:
        return f"{sd:.2e}"       # a spread smaller than the metric's printed precision
    return text


def fmt_signed(value: Any, spec: str) -> str:
    if value is None:
        return "n/a"
    decimals = int(spec.lstrip("%.").rstrip("f") or 4)
    return f"{float(value):+.{decimals}f}"


def resolve_frozen(explicit: Path | None, problems: list[str]) -> Path | None:
    """The frozen report, by content. A candidate with the wrong digest is named and skipped."""
    candidates = [explicit] if explicit is not None else [Path(name) for name in FROZEN_CANDIDATES]
    for candidate in candidates:
        if not candidate.is_file():
            print(f"  frozen report: {candidate} is not present")
            continue
        got = sha256(candidate)
        if got == FROZEN_SHA:
            print(f"  frozen report: {candidate}  sha256 matches {FROZEN_SHA[:16]}...")
            return candidate
        if explicit is not None:
            problems.append(f"{candidate} hashes to {got}, not the frozen report's {FROZEN_SHA}: "
                            f"refusing to measure a spread against it")
            return None
        print(f"  frozen report: {candidate} hashes to {got[:16]}... -- NOT the frozen report "
              f"({FROZEN_SHA[:16]}...); skipped, never used")
    problems.append("no file under " + " or ".join(FROZEN_CANDIDATES) + f" hashes to {FROZEN_SHA}, so "
                    "the frozen values this spread is measured against could not be resolved. The frozen "
                    "report is the one the step-13 re-score wrote (job 4118422); pull it back first.")
    return None


def check_expected_frozen(frozen: dict, problems: list[str]) -> None:
    """The resolved frozen report must carry the values this exercise is quoted against."""
    paths = {
        "accuracy": ("accuracy", "accuracy"),
        "axis1_primary_auroc": ("axis1_panel_adequacy", "primary", "auroc"),
        "axis1_primary_detection": ("axis1_panel_adequacy", "primary", "detection_rate"),
        "axis1_pooled_auroc": ("axis1_panel_adequacy", "auroc"),
        "triage_coverage": ("triage", "coverage"),
        "triage_risk": ("triage", "risk"),
    }
    bad = 0
    for key, want in EXPECTED_FROZEN.items():
        parts = key.split(".")
        if parts[0] in ("family", "dataset"):
            got = dig(frozen, ("breakdowns", parts[0], parts[1], parts[2]))
        else:
            got = dig(frozen, paths[key])
        if got is None or round(float(got), 4) != want:
            bad += 1
            problems.append(f"the frozen report's {key} is {got!r}, not the quoted {want}: either the "
                            f"definitions drifted or this is not the file those numbers came from")
    if not bad:
        print(f"  the frozen report carries the quoted value for all {len(EXPECTED_FROZEN)} metrics "
              f"(4 decimals), so the spread is measured against the same definitions")


def check_report(name: str, report: dict, seed: int | None, frozen: dict | None,
                 problems: list[str], verbose: bool = True,
                 expect_frozen_training_suite: bool = True) -> None:
    """Everything that makes a report a member of this family rather than a plausible file.

    ``expect_frozen_training_suite`` is True for the seed draws, which must have been trained on the
    frozen training suite, and False for arm B: arm B's intervention *is* a different training suite,
    so a rule that refused it would refuse the very report this spread exists to judge. Arm B is
    still held to the evaluation it was scored on and to the operating point.
    """
    suite = dig(report, ("suite", "manifest_sha256"))
    trained_on = dig(report, ("checkpoint", "suite_sha256_trained_on"))
    op = (dig(report, ("tool", "alpha")), dig(report, ("tool", "beta")),
          dig(report, ("tool", "method")), dig(report, ("tool", "fpr")))
    if suite != CAP_MANIFEST:
        problems.append(f"{name}: scored on suite {suite!r}, not the frozen capped evaluation "
                        f"{CAP_MANIFEST}")
    if expect_frozen_training_suite and trained_on != TRAIN_MANIFEST:
        problems.append(f"{name}: the checkpoint was trained on suite {trained_on!r}, not the frozen "
                        f"training suite {TRAIN_MANIFEST}")
    if not expect_frozen_training_suite:
        print(f"    trained on {trained_on} -- not the frozen training suite, as expected for arm B")
    if op != OPERATING_POINT:
        problems.append(f"{name}: operating point {op} != alpha 0.1 / beta 0.05 / lac / fpr 0.05")
    if seed is not None:
        recorded = dig(report, ("seeds", "checkpoint_training_seed"))
        if recorded != seed:
            problems.append(f"{name}: the report's checkpoint_training_seed is {recorded!r}, not {seed}: "
                            f"another draw's report under this seed's name")
    if verbose:
        print("    " + ", ".join(f"{label}={value}" for label, value in (
            ("accuracy n", dig(report, ("accuracy", "n"))),
            ("PRIMARY n", dig(report, ("axis1_panel_adequacy", "primary", "n"))),
            ("POOLED n", dig(report, ("axis1_panel_adequacy", "n"))),
            ("triage n", dig(report, ("triage", "n"))),
            ("training seed", dig(report, ("seeds", "checkpoint_training_seed"))),
        )))
    if frozen is None:
        return
    # The populations must be the frozen pass's: a pass that dropped rows would move every rate at
    # once and look like a small seed effect.
    for family in FAMILIES:
        got = dig(report, ("breakdowns", "family", family, "n"))
        want = dig(frozen, ("breakdowns", "family", family, "n"))
        if got != want:
            problems.append(f"{name}: family {family} has n={got}, the frozen pass n={want}: not the "
                            f"same population")
    for atlas in ATLASES:
        got = dig(report, ("breakdowns", "dataset", atlas, "n"))
        want = dig(frozen, ("breakdowns", "dataset", atlas, "n"))
        if got != want:
            problems.append(f"{name}: atlas {atlas} has n={got}, the frozen pass n={want}")
    for label, got, want in (
        ("accuracy n", dig(report, ("accuracy", "n")), dig(frozen, ("accuracy", "n"))),
        ("axis-1 PRIMARY n", dig(report, ("axis1_panel_adequacy", "primary", "n")),
         dig(frozen, ("axis1_panel_adequacy", "primary", "n"))),
        ("axis-1 POOLED n", dig(report, ("axis1_panel_adequacy", "n")),
         dig(frozen, ("axis1_panel_adequacy", "n"))),
        ("triage n", dig(report, ("triage", "n")), dig(frozen, ("triage", "n"))),
    ):
        if got != want:
            problems.append(f"{name}: {label} is {got}, the frozen pass {want}")


def spread(values: Sequence[float]) -> tuple[float, float, float, float]:
    """``(mean, sd, low, high)``. Sample sd: the draws are a sample of the pipeline, not a population."""
    mean = statistics.fmean(values)
    sd = statistics.stdev(values) if len(values) > 1 else float("nan")
    return mean, sd, min(values), max(values)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", default="1,2,3", help="the seed draws to read (default 1,2,3)")
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"),
                        help="where the reports live (default: reports/)")
    parser.add_argument("--frozen", type=Path, default=None,
                        help="the frozen report; by default it is resolved by sha256 among the names it "
                             "travels under")
    parser.add_argument("--armb", type=Path, default=None,
                        help="arm B's report, to read its deltas against this noise floor (default: "
                             "reports/eval-armb-v1-eval.json when that file exists)")
    parser.add_argument("--json", type=Path, default=None, help="also write the table as JSON here")
    args = parser.parse_args(argv)

    try:
        seeds = [int(part) for part in str(args.seeds).replace(" ", ",").split(",") if part]
    except ValueError:
        parser.error(f"--seeds takes integers, got {args.seeds!r}")
    problems: list[str] = []
    notes: list[str] = []

    print("=" * 122)
    print("SEED-VARIANCE SPREAD -- the frozen configuration retrained at other seeds, measured against"
          " the frozen report")
    print("=" * 122)
    frozen_path = resolve_frozen(args.frozen, problems)
    frozen = json.loads(frozen_path.read_text()) if frozen_path else None
    if frozen is not None:
        check_expected_frozen(frozen, problems)

    print()
    print(f"--- the {len(seeds)} seed draw(s) asked for ---")
    reports: dict[int, dict] = {}
    for seed in seeds:
        path = args.reports_dir / f"eval-seed{seed}-v1-eval.json"
        if not path.is_file():
            print(f"  seed {seed}: MISSING {path}")
            notes.append(f"seed {seed}: no report at {path} -- pull it from the cluster")
            continue
        report = json.loads(path.read_text())
        print(f"  seed {seed}: {path}   {path.stat().st_size} bytes   sha256 {sha256(path)[:16]}...")
        check_report(f"seed {seed}", report, seed, frozen, problems)
        reports[seed] = report

    print()
    print("--- are these draws distinct? (adapter sha256 per run -- the evidence that a seed job trained"
          " a new model rather than re-scoring the frozen one) ---")
    labelled: list[tuple[str, Any]] = []
    if frozen is not None:
        labelled.append(("frozen (seed 0)", dig(frozen, ("checkpoint", "adapter_sha256"))))
    labelled.extend((f"seed {seed}", dig(report, ("checkpoint", "adapter_sha256")))
                    for seed, report in sorted(reports.items()))
    seen: dict[str, list[str]] = {}
    for name, digest in labelled:
        print(f"  {name:<16} adapter {digest}")
        if digest is not None:
            seen.setdefault(str(digest), []).append(name)
    for digest, names in seen.items():
        if len(names) > 1:
            problems.append(f"{' and '.join(names)} carry the SAME adapter ({str(digest)[:16]}...): one "
                            f"model scored more than once, which is not a draw of the pipeline")

    print()
    print("--- per-draw row dumps (for scripts/axis1-recompute.py --dump seed<S>) ---")
    for seed in seeds:
        path = args.reports_dir / f"axis1-rows-seed{seed}.jsonl"
        if path.is_file():
            rows = sum(1 for line in path.read_text().splitlines() if line.strip())
            print(f"  seed {seed}: {path}  {rows} rows"
                  + ("" if rows == FROZEN_ROWS else f"   WARNING: the frozen pass dumped {FROZEN_ROWS}"))
        else:
            print(f"  seed {seed}: {path} is not present (pull it to re-derive axis 1 with no GPU)")

    if frozen is None:
        print()
        print("VERDICT: FAIL -- the frozen report could not be resolved, so there is nothing to measure"
              " the draws against.")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    table = metrics()
    frozen_values = {key: getter(frozen) for _, key, _, getter, _ in table}
    values = {seed: {key: getter(report) for _, key, _, getter, _ in table}
              for seed, report in reports.items()}
    draws = sorted(values)

    print()
    print("=" * 122)
    print(f"THE SPREAD  ({len(draws)} of {len(seeds)} seed draws + the frozen draw; sd is the sample sd"
          f" over {'the %d draw(s)' % len(draws) if draws else 'no draws'}"
          f"{', %d df' % (len(draws) - 1) if len(draws) > 1 else ''})")
    print("=" * 122)
    header = (f"  {'metric':<40} {'frozen':>9}" + "".join(f" {'seed%d' % s:>9}" for s in seeds)
              + f" {'mean':>9} {'sd':>8} {'min':>9} {'max':>9}  {'frozen in range':>15}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    rows_out: list[dict[str, Any]] = []
    for group, title in (("headline", "headline metrics"), ("family", "per-family breakdown"),
                         ("atlas", "per-atlas breakdown")):
        print(f"  --- {title} ---")
        for g, key, label, getter, spec in table:
            if g != group:
                continue
            frozen_value = frozen_values.get(key)
            numeric = [float(values[s][key]) for s in draws
                       if isinstance(values[s][key], (int, float))]
            if not numeric:
                print(f"  {label:<40} {fmt(frozen_value, spec):>9}   (no draw carries this metric)")
                continue
            mean, sd, low, high = spread(numeric)
            inside = frozen_value is not None and low <= float(frozen_value) <= high
            rows_out.append({
                "metric": key, "label": label, "frozen": frozen_value,
                "draws": {str(s): values[s][key] for s in draws},
                "mean": mean, "sd": None if sd != sd else sd, "min": low, "max": high,
                "frozen_in_range": bool(inside),
                "frozen_minus_mean_over_sd": (None if (sd != sd or sd == 0 or frozen_value is None)
                                              else (float(frozen_value) - mean) / sd),
            })
            print(f"  {label:<40} {fmt(frozen_value, spec):>9}"
                  + "".join(f" {fmt(values[s][key], spec) if s in values else 'n/a':>9}" for s in seeds)
                  + f" {fmt(mean, spec):>9} {fmt_sd(sd, spec):>8} {fmt(low, spec):>9} "
                  + f"{fmt(high, spec):>9}  {'yes' if inside else 'NO':>15}")

    print()
    print("--- the frozen draw's standing in its own family, on the metrics the claim rests on ---")
    for key in ("accuracy", "axis1_primary_auroc", "triage_coverage", "family.panel_choice.accuracy"):
        row = next((r for r in rows_out if r["metric"] == key), None)
        if row is None:
            continue
        sd, z = row["sd"], row["frozen_minus_mean_over_sd"]
        print(f"  {row['label']:<40} frozen {row['frozen']:.4f}  seed mean {row['mean']:.4f}  sd "
              f"{'n/a' if sd is None else format(sd, '.4f')}  ->  "
              + ("z n/a" if z is None else f"z {z:+.2f}")
              + "   " + ("inside the seed range" if row["frozen_in_range"]
                         else "OUTSIDE the seed range -- the frozen draw is not an ordinary redraw here"))

    armb_path = args.armb if args.armb is not None else Path("reports/eval-armb-v1-eval.json")
    if armb_path.is_file():
        armb = json.loads(armb_path.read_text())
        print()
        print("=" * 122)
        print(f"ARM B AGAINST THE NOISE FLOOR   ({armb_path})")
        print("=" * 122)
        armb_problems: list[str] = []
        check_report(f"arm B ({armb_path})", armb, None, frozen, armb_problems, verbose=True,
                     expect_frozen_training_suite=False)
        header = (f"  {'metric':<40} {'armB':>9} {'frozen':>9} {'armB-frozen':>12} {'seed sd':>9}"
                  f" {'delta/sd':>9}  {'verdict':>23}")
        print(header)
        print("  " + "-" * (len(header) - 2))
        beyond = within = 0
        lines: dict[str, str] = {}
        for g, key, label, getter, spec in table:
            got = getter(armb)
            row = next((r for r in rows_out if r["metric"] == key), None)
            if got is None or row is None or row["frozen"] is None:
                print(f"  {label:<40} {'n/a':>9} {'n/a':>9} {'n/a':>12} {'n/a':>9} {'n/a':>9}  "
                      f"{'not comparable':>23}")
                continue
            delta = float(got) - float(row["frozen"])
            largest = max(abs(row["min"] - row["frozen"]), abs(row["max"] - row["frozen"]))
            inside = abs(delta) <= largest
            beyond += not inside
            within += inside
            sd = row["sd"]
            ratio = "n/a" if (not sd or sd != sd) else f"{delta / sd:+.2f}"
            verdict = "within the seed range" if inside else "beyond the seed range"
            print(f"  {label:<40} {fmt(float(got), spec):>9} {fmt(float(row['frozen']), spec):>9} "
                  f"{fmt_signed(delta, spec):>12} {fmt_sd(sd, spec) if sd else 'n/a':>9} "
                  f"{ratio:>9}  {verdict:>23}")
            lines[key] = (f"    {label}: arm B {float(got):.4f}, frozen {float(row['frozen']):.4f}, delta "
                          f"{delta:+.4f}; the redraws span {row['min']:.4f}-{row['max']:.4f} "
                          f"(width {row['max'] - row['min']:.4f}) -> {verdict}")
        print()
        print(f"  {beyond} metric(s) beyond the seed range, {within} within it.")
        print("  the two numbers the arm-B claim rests on:")
        for key in ("axis1_primary_auroc", "family.panel_choice.accuracy"):
            if key in lines:
                print(lines[key])
        if armb_problems:
            print("  arm B's report has integrity problems of its own -- its column above is NOT read:")
            for problem in armb_problems:
                print(f"    - {problem}")
    else:
        armb_problems = []
        print()
        print(f"  (arm B's report was not read: {armb_path} is not present -- pass --armb to point at it)")

    if args.json is not None:
        args.json.write_text(json.dumps({
            "frozen_report": str(frozen_path), "frozen_sha256": FROZEN_SHA,
            "seeds_requested": seeds, "seeds_present": draws, "rows": rows_out,
        }, indent=2))
        print(f"\n  wrote {args.json}")

    print()
    print("=" * 122)
    for note in notes:
        print(f"  note: {note}")
    if problems:
        print("INTEGRITY FAILURES -- the numbers above are NOT a noise floor for this family:")
        for problem in problems:
            print(f"  - {problem}")
        print("VERDICT: FAIL -- see the integrity failures above.")
        return 1
    if armb_problems:
        print("VERDICT: ARM-B COLUMN NOT READABLE -- the spread above passed its own integrity checks,"
              " but arm B's report did not, so its column is not read. Fix arm B's report (or drop"
              " --armb) and read this again.")
        return 4
    if len(draws) != len(seeds):
        print(f"VERDICT: PARTIAL -- {len(draws)} of {len(seeds)} seed draws are on disk"
              f" ({', '.join('seed %d' % s for s in draws) or 'none'}). The spread above is over what is"
              f" present; it is not yet the family's noise floor.")
        return 3
    print(f"VERDICT: OK -- {len(draws)} redraws plus the frozen report, every integrity check passed."
          f" The frozen value is inside the seed range for "
          f"{sum(1 for r in rows_out if r['frozen_in_range'])} of {len(rows_out)} metrics; where it is"
          f" not, the frozen draw is not an ordinary member of this family on that metric.")
    print("  Quote the range (min-max) as the statement and the sd as an indication: three draws give"
          " two degrees of freedom, and a sample sd on two degrees of freedom is itself uncertain by"
          " roughly a factor of two.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
