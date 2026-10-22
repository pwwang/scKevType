"""Every number the depth-layer report quotes, recomputed from ``artifacts/cl``.

Run after ``python -m sckevtype.data.depth --out artifacts/cl --corpus runs/real-v2``:

    python scripts/cl-depth-report.py

It reads only the four files the depth pass wrote — ``spines.json``,
``roots.tsv``, ``depth-histogram.json``, ``depth-targets.jsonl`` — so the report's
tables are reproducible from the row dump rather than from a log line. The one
thing it recomputes from the ontology itself is the per-dataset *ceiling*: the
number of register levels a descent has available, which is what any depth target
is bounded by.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sckevtype.data import depth as D

OUT = ROOT / "artifacts" / "cl"
V3 = ROOT / "artifacts" / "v3"
SEED = ROOT / "configs" / "seed" / "v3" / "atlas_labels.json"

PRIMARY = "discriminant.fold"
#: The populations the gate is quoted on. ``clean`` is the record as a lab would
#: have it; ``permuted`` is the same record with the panel order and option names
#: shuffled, so its depth target is by construction identical.
POPULATIONS = ("clean", "permuted")


def median(values: list[int]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    return float(ordered[middle]) if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2


def rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def read_rows() -> tuple[dict, list[dict]]:
    targets = list(rows(OUT / "depth-targets.jsonl"))
    histogram = json.loads((OUT / "depth-histogram.json").read_text(encoding="utf-8"))
    return histogram, targets


def table_spine() -> None:
    payload = json.loads((OUT / "spines.json").read_text(encoding="utf-8"))
    print("## Spine\n")
    print(f"- terms: {payload['n_terms']}")
    print(f"- terms whose chain reaches {payload['cut']}: {payload['n_passes_cut']}")
    print(f"- terms with no CL parent (the root and the `part_of`-only ones): {len(payload['parentless'])}")
    print(f"  {', '.join(payload['parentless'])}")
    print(f"- branch ties (equal-sized branches, lowest id taken): {payload['n_ties']}")
    print(f"- terms that reach the top without the cut: {len(payload['outside_cut'])}")
    print(f"  {json.dumps(payload['outside_cut'])}")
    print()


def table_roots() -> None:
    lines = (OUT / "roots.tsv").read_text(encoding="utf-8").splitlines()
    header = lines[0].split("\t")
    keep = ["dataset", "root", "root_name", "coverage", "labels_covered", "labels_total",
            "register_labels_covered", "cells_covered", "cells_total", "level_one_options"]
    index = [header.index(name) for name in keep]
    print("## Roots (plan E1.4: the deepest node covering ≥95% of the dataset's cells)\n")
    print("| " + " | ".join(keep) + " |")
    print("|" + "---|" * len(keep))
    for line in lines[1:]:
        cells = line.split("\t")
        print("| " + " | ".join(cells[i] for i in index) + " |")
    print()


def table_ceiling(register: D.RegisterTree, counts: dict[str, dict[str, int]], root_of: dict[str, str]) -> dict:
    """The register levels available per dataset, label-weighted by the atlas's cells."""
    out = {}
    for dataset_id, labels in sorted(counts.items()):
        root = root_of[dataset_id]
        lengths = Counter()
        for label, cells in labels.items():
            if label not in register.members:
                continue
            chain = register.chain_from(label, root)
            if chain:
                lengths[len(chain)] += cells
        out[dataset_id] = lengths
    return out


def print_ceiling(name: str, lengths: Counter, *, note: str = "") -> None:
    values = sorted(depth for depth, n in lengths.items() for _ in range(n))
    if not values:
        print(f"| {name} | 0 | - | - |")
        return
    shares = {f"{depth}": round(n / len(values), 3) for depth, n in sorted(lengths.items())}
    print(f"| {name}{note} | {len(values):,} | {median(values)} | {shares} |")


def table_gate(histogram: dict) -> None:
    print("## The gate, per separability rule\n")
    print("| rule | records | median (all) | root share (all) | median (on chain) | root share (on chain) | passes |")
    print("|---|---|---|---|---|---|---|")
    for rule, entry in sorted(histogram["rules"].items()):
        gate = entry["gate"]
        print(
            f"| `{rule}` | {entry['records']['n']:,} | {gate['median_depth_all_records']} | "
            f"{gate['share_at_root_all_records']:.4f} | {gate['median_depth_on_chain']} | "
            f"{gate['share_at_root_on_chain']:.4f} | {gate['passes_all_records']}/{gate['passes_on_chain']}/{gate['passes_cells']} |"
        )
    print()


def table_by_variant(histogram: dict) -> None:
    print("## The gate, per variant (population matters)\n")
    for rule in (PRIMARY, "holds"):
        entry = histogram["rules"][rule]
        print(f"`{rule}`\n")
        print("| variant | median (all) | root share (all) | median (on chain) | root share (on chain) | passes |")
        print("|---|---|---|---|---|---|")
        for variant, gate in sorted(entry["gate_by_variant"].items()):
            print(
                f"| {variant} | {gate['median_depth_all_records']} | {gate['share_at_root_all_records']:.4f} | "
                f"{gate['median_depth_on_chain']} | {gate['share_at_root_on_chain']:.4f} | "
                f"{gate['passes_all_records']}/{gate['passes_on_chain']} |"
            )
        print()


def table_targets(targets: list[dict], register: D.RegisterTree, root_of: dict[str, str]) -> None:
    print(f"## Depth targets per dataset, `clean` variant, rule `{PRIMARY}`\n")
    print("| dataset | on chain | off register | off root | median | root share | dist | median ceiling | median target |")
    print("|---|---|---|---|---|---|---|---|---|")
    by_dataset = defaultdict(list)
    for row in targets:
        if row["variant"] != "clean":
            continue
        by_dataset[row["dataset"]].append(row)
    for dataset_id in sorted(by_dataset):
        group = by_dataset[dataset_id]
        chain = [row for row in group if row["levels"] not in ("off_register", "off_root")]
        off_register = sum(1 for row in group if row["levels"] == "off_register")
        off_root = sum(1 for row in group if row["levels"] == "off_root")
        dist = Counter(row["depth_target"] for row in chain)
        ceiling = Counter(len(row["chain"]) for row in chain)
        print(
            f"| {dataset_id} | {len(chain):,} | {off_register:,} | {off_root:,} | "
            f"{median([d for d, n in dist.items() for _ in range(n)])} | "
            f"{dist.get(0, 0) / max(len(chain), 1):.4f} | "
            f"{json.dumps({str(d): dist[d] for d in sorted(dist)})} | "
            f"{median([d for d, n in ceiling.items() for _ in range(n)])} | "
            f"{median([row['depth_target'] for row in chain])} |"
        )
    print()


def table_stops(targets: list[dict]) -> None:
    print(f"## Where the descent stops at level 1 (rule `{PRIMARY}`, `clean` variant)\n")
    stops = defaultdict(Counter)
    for row in targets:
        if row["variant"] != "clean" or row["levels"] in ("off_register", "off_root"):
            continue
        parts = row["levels"].split("|")
        if row["depth_target"] == 0:
            stops[row["dataset"]][parts[0].split(":", 1)[1]] += 1
    print("| dataset | level-1 stop reason | records |")
    print("|---|---|---|")
    for dataset_id in sorted(stops):
        for code, n in stops[dataset_id].most_common():
            print(f"| {dataset_id} | `{code}` | {n:,} |")
    print()


def table_blocking(targets: list[dict]) -> None:
    """The panel memo's "claims a lab must decline", counted on level 1."""
    print("## Claims a lab must decline at level 1 (`nodisc` reasons, `clean` variant)\n")
    blocked = Counter()
    for row in targets:
        if row["variant"] != "clean" or row["depth_target"] != 0:
            continue
        if row["levels"].startswith("1:nodisc"):
            blocked[(row["cl_id"], len(row["chain"]))] += 1
    deepest = max((length for _, length in blocked), default=0)
    print(f"- records stopping at level 1 for a missing discriminating marker: {sum(blocked.values()):,}")
    print(f"- over {len(blocked)} (label, chain-length) pairs; the deepest of them is {deepest} levels long")
    print()


def table_comparison() -> None:
    """The gate under its three definitions, one row per configuration (E1.5, fixed).

    Every scoped row is printed directly against the whole-register row it replaces,
    and every extended-register row against the as-built one: the fixes are separable
    and which one buys what is the question this table answers.
    """
    path = OUT / "depth-comparison.json"
    if not path.exists():
        print("## The gate after the three fixes\n")
        print(f"_{path.name} is not built: run the depth layer with `--compare`._\n")
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    configurations = payload["configurations"]
    print("## The gate under its three definitions, by configuration\n")
    print("Clauses: median depth ≥ 3, and under 20% at the root. `all` is every record as written")
    print("(an `off_register`/`off_root` record carries 0), `chain` the records a descent starts on,")
    print("`cells` one reading per (dataset, cell).\n")
    for definition in payload["definitions"]:
        print(f"`{definition}`\n")
        print("| configuration | n | median | at root | passes |")
        print("|---|---|---|---|---|")
        for name, entry in configurations.items():
            reading = entry["gate"][definition]
            print(
                f"| {name} | {reading['n']:,} | {reading['median']:.1f} | "
                f"{reading['share_at_root']:.4f} | {'**yes**' if reading['passes'] else 'no'} |"
            )
        print()


def table_dataset_gate(configuration: str, label: str) -> None:
    """One configuration's per-dataset verdicts, under all three definitions."""
    path = OUT / "depth-comparison.json"
    if not path.exists():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    if configuration not in payload["configurations"]:
        raise SystemExit(f"{configuration!r} is not in {path} (have: {list(payload['configurations'])})")
    entry = payload["configurations"][configuration]
    print(f"### Per dataset — {label}\n")
    print("| dataset | root | on chain | all: med / at root | passes | chain: med / at root | passes | cells: med / at root | passes |")
    print("|---|---|---|---|---|---|---|---|---|")
    for dataset in sorted(entry["dataset_gate"]):
        record = entry["dataset_gate"][dataset]
        cells = entry["dataset_cells_gate"][dataset]
        root = entry["config"]["root_of"][dataset]
        print(
            f"| {dataset} | {root} | {record['on_chain_records']:,} | "
            f"{record['median_depth_all_records']:.1f} / {record['share_at_root_all_records']:.4f} | "
            f"{'yes' if record['passes_all_records'] else 'no'} | "
            f"{record['median_depth_on_chain']:.1f} / {record['share_at_root_on_chain']:.4f} | "
            f"{'yes' if record['passes_on_chain'] else 'no'} | "
            f"{cells['median']:.1f} / {cells['share_at_root']:.4f} | "
            f"{'yes' if cells['passes'] else 'no'} |"
        )
    print()


def main() -> int:
    histogram, targets = read_rows()
    terms = D.load_terms(str(V3 / "cl_terms.json"))
    names = {cl_id: term.name for cl_id, term in terms.items()}
    spine = D.build_spine(terms)
    definitions = D.load_definitions(str(V3 / "definitions.json"))
    register = D.RegisterTree(spine, D.register_terms(definitions))
    counts = D.atlas_label_counts(str(SEED))
    roots = D.dataset_roots(counts, spine, register, names)
    root_of = {row.dataset_id: row.cl_id for row in roots}

    table_spine()
    table_roots()
    table_gate(histogram)
    table_by_variant(histogram)
    table_targets(targets, register, root_of)
    table_comparison()
    table_dataset_gate("as-built / global / plan", "the register as built, the whole-register options, the plan's roots")
    table_dataset_gate("extended / scoped / overrides", "the canonical configuration: all three fixes")

    print("## Ceiling: register levels available per dataset (label-weighted by cells)\n")
    print("| dataset | cells in a descent | median | share by depth |")
    print("|---|---|---|---|")
    for dataset_id, lengths in sorted(table_ceiling(register, counts, root_of).items()):
        print_ceiling(dataset_id, lengths)

    record_lengths = Counter(
        len(row["chain"])
        for row in targets
        if row["variant"] == "clean" and row["levels"] not in ("off_register", "off_root")
    )
    print_ceiling("**the corpus as built (records, clean variant)**", record_lengths)
    print()
    table_stops(targets)
    table_blocking(targets)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
