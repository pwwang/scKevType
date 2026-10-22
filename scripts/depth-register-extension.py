"""The depth layer's register, extended: the interior nodes every descent walks past.

E1.5's gate fails partly because the register can only name the corpus's *labels*, not
the nodes a descent passes *through* on its way down from a dataset's root
(``reports/cl-depth-layer.md`` §5: the median record has 2 register levels available,
so no separability rule can reach the gate's median of 3). The fix is curation, not a
new rule: give the register definitions for the interior nodes that are already on
the corpus's own descent paths.

Nothing here is hand-appended. The extension is selected by rule and built through the
same path the as-built register came from (``scripts/curate_register_v2.py``'s
``build_definitions`` over CellMarker 2.0, with its ScType fallback):

    extension = every CL term that
        * lies strictly below a dataset's root on some seeded label's spine chain,
        * is not already a register term, and
        * a marker resource can ground.

The roots are the ones the layer actually uses: the plan's E1.4 rule with
:data:`sckevtype.data.depth.ROOT_OVERRIDES` applied, because a node is only worth a
definition if some descent can reach it. (Against the plan's roots alone the same rule
selects 83 terms; the two overrides admit one more. Both counts are printed.)

Writes ``artifacts/v3/definitions.depth.json`` — a new artifact. ``definitions.json``
is left untouched: the flat corpus renders from it, and its digest is in the v3
manifest.

Run: uv run --no-sync python scripts/depth-register-extension.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import curate_register_v2 as crv2  # noqa: E402

from sckevtype.contracts.definitions import DefinitionSet  # noqa: E402
from sckevtype.contracts.freeze import file_digest, read_json  # noqa: E402
from sckevtype.data.depth import (  # noqa: E402
    ROOT_OVERRIDES,
    RegisterTree,
    apply_root_overrides,
    atlas_label_counts,
    build_spine,
    dataset_roots,
    load_definitions,
    load_terms,
    register_terms,
)

EXTERNAL = ROOT / "data" / "external"
CELLMARKER_SPECIES = ("Human", "Mouse")


def groundable_by(cl_id: str, terms: dict, cellmarker: dict, sctype: dict) -> bool:
    """Can a marker resource ground this term? — the curator's own test, reused.

    ``build_definitions`` would fall back to ScType's name index, so a term either
    resource can ground is groundable; a term neither can is left out of the scope
    rather than given an empty definition.
    """
    if cellmarker.get(cl_id):
        return True
    name = (terms.get(cl_id, {}).get("name") or "").strip().casefold()
    return bool(name and sctype.get(name))


def interior_nodes(datasets: dict, roots: list, spine, members: set[str]) -> set[str]:
    """The register terms a descent from each dataset's root passes but cannot name.

    Strictly below the root on a seeded label's spine chain: the levels the descent
    asks about, minus the ones the register already names.
    """
    needed: set[str] = set()
    for row in roots:
        for cl_id in datasets[row.dataset_id]:
            chain = spine.chains.get(cl_id)
            if not chain or row.cl_id not in chain:
                continue
            needed.update(node for node in chain[: chain.index(row.cl_id)] if node not in members)
    return needed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extend the register with the interior spine nodes the corpus's descents pass through."
    )
    parser.add_argument("--artifacts", default="artifacts/v3", help="the CL snapshot / register version to read")
    parser.add_argument("--seed", default="configs/seed/v3/atlas_labels.json", help="the atlas label inventory")
    parser.add_argument(
        "--out",
        default=None,
        help="where the extended register is written (default: <artifacts>/definitions.depth.json)",
    )
    parser.add_argument(
        "--no-root-overrides",
        action="store_true",
        help="select the extension against the plan's roots alone, ignoring ROOT_OVERRIDES",
    )
    args = parser.parse_args(argv)

    artifacts = Path(args.artifacts)
    out = Path(args.out) if args.out else artifacts / "definitions.depth.json"
    terms = load_terms(artifacts / "cl_terms.json")
    names = {cl_id: term.name for cl_id, term in terms.items()}
    spine = build_spine(terms, prefer_cut=True)
    as_built = load_definitions(artifacts / "definitions.json")
    members = set(register_terms(as_built))
    datasets = atlas_label_counts(args.seed)
    register = RegisterTree(spine, tuple(sorted(members)))
    plan_roots = dataset_roots(datasets, spine, register, names)
    roots = plan_roots if args.no_root_overrides else apply_root_overrides(plan_roots, datasets, spine, register, names)

    # --- the resources, exactly as the register's own curation reads them -------
    cl_terms = crv2.cl_release(EXTERNAL / "cl-basic.json")
    cm_paths = {species: EXTERNAL / f"Cell_marker_{species}.xlsx" for species in CELLMARKER_SPECIES}
    sctype_path = EXTERNAL / "ScTypeDB_full.xlsx"
    for path in (EXTERNAL / "cl-basic.json", *cm_paths.values(), sctype_path):
        if not path.exists():
            print(f"missing resource: {path} (the extension is derived from the resources, not typed in)", file=sys.stderr)
            return 1
    digests = {species: file_digest(path) for species, path in cm_paths.items()}
    digests["cl"] = file_digest(EXTERNAL / "cl-basic.json")
    digests["sctype"] = file_digest(sctype_path)
    cellmarker = crv2.cellmarker_index(cm_paths)
    sctype = crv2.sctype_index(sctype_path)

    # --- the extension ---------------------------------------------------------
    print("roots (the plan's rule, with ROOT_OVERRIDES applied"
          + (" — NOT applied here)" if args.no_root_overrides else ")"))
    for row in roots:
        print(f"  {row.dataset_id:20s} {row.cl_id} {row.name[:24]:24s} cov={row.coverage:.3f}"
              + (f"  [{row.override}]" if row.override else ""))

    plan_needed = interior_nodes(datasets, plan_roots, spine, members)
    plan_groundable = sorted(
        cl_id for cl_id in plan_needed if cl_id in cl_terms and groundable_by(cl_id, cl_terms, cellmarker, sctype)
    )
    needed = interior_nodes(datasets, roots, spine, members)
    extension = sorted(
        cl_id for cl_id in needed if cl_id in cl_terms and groundable_by(cl_id, cl_terms, cellmarker, sctype)
    )
    ungroundable = sorted(cl_id for cl_id in needed if cl_id not in set(extension))
    print(
        f"\ninterior nodes not in the register: {len(needed)} (plan roots: {len(plan_needed)}); "
        f"groundable: {len(extension)} (plan roots: {len(plan_groundable)}); "
        f"left out for want of a resource: {len(ungroundable)}"
    )
    if ROOT_OVERRIDES and not args.no_root_overrides:
        moved = sorted(set(extension) - set(plan_groundable))
        dropped = sorted(set(plan_groundable) - set(extension))
        print(f"  the overrides add {len(moved)} node(s) and drop {len(dropped)}: "
              + ", ".join(f"+{cl_id} {names.get(cl_id, '')}" for cl_id in moved)
              + (", " + ", ".join(f"-{cl_id} {names.get(cl_id, '')}" for cl_id in dropped) if dropped else ""))
    for cl_id in extension:
        print(f"  + {cl_id} {names.get(cl_id, '')}")

    # --- write -----------------------------------------------------------------
    scope = sorted(members | set(extension))
    built = crv2.build_definitions(scope, cl_terms, cellmarker, sctype, digests)
    extended = DefinitionSet.model_validate(
        {
            "definitions": built["definitions"],
            "synonyms": as_built.synonyms,
            "source": as_built.source
            + " Plus the interior-node extension (scripts/depth-register-extension.py): the spine nodes the"
            " corpus's own descents pass through that a marker resource can ground.",
            "version": f"{as_built.version}+depth",
        }
    )
    digest = extended.save(str(out))
    print(
        f"\n{out}: {len(extended.definitions)} definitions "
        f"({len(members)} as built + {len(extension)} interior), {len(built['fallbacks'])} by the ScType fallback, "
        f"digest {digest[:16]}…"
    )
    assert set(extension) <= set(register_terms(extended)), "the extension did not reach the register"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
