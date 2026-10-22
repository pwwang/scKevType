"""CLI: refresh a local Cell Ontology snapshot. **The only network code in the package.**

    python -m sckevtype.contracts.fetch --what cl [--out configs/seed/cl_terms.seed.yaml]
    python -m sckevtype.contracts.fetch --what cl --from-file data/external/cl-basic.json ...

Library code never fetches (see ``ontology.load_cl_terms``): a build that
reached the network would not be reproducible. This command is run by hand,
rarely, and its output is committed as a seed input; the digest of that file
then pins every artifact built from it.

``--from-file`` parses an already-downloaded release from disk instead of over
the wire, so the snapshot can be rebuilt offline (a refresh and a rebuild share
the parser; only the source of the bytes differs). The default path needs
network access, so it is not covered by the CPU-only test suite — the parser it
delegates to is.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

from .ontology import ClTerm, save_cl_terms

#: OBO Foundry PURL for the CL basic (JSON) release.
CL_BASIC = "http://purl.obolibrary.org/obo/cl/cl-basic.json"
#: Obsolete terms are skipped: they must not become a join key.
CL_PREFIX = "http://purl.obolibrary.org/obo/CL_"


def _cl_id(identifier: str) -> str:
    """The short ``CL:0000084`` form of a CL PURL."""
    return "CL:" + identifier[len(CL_PREFIX) :]


def parse_cl_graph(graph: dict, source: str = CL_BASIC) -> dict[str, ClTerm]:
    """Convert one OBO Graph JSON graph into our snapshot shape.

    The hierarchy is **not** a node field. In OBO Graph JSON a node carries only
    ``id``/``lbl``/``meta``; ancestry lives in ``graph["edges"]``, where
    ``{"sub": child, "pred": "is_a", "obj": parent}`` is the only parent
    relation. Reading a ``parents`` key off a node yields nothing on every term,
    silently — an empty tuple is a perfectly valid ``ClTerm``.

    Only ``is_a`` edges are ancestry, and only parents that are themselves CL
    terms are kept: the snapshot is the cell-type spine, so a UBERON/CARO/BFO
    parent is dropped rather than carried as an id no dataset can map to.
    """
    parents_of: dict[str, set[str]] = {}
    for edge in graph.get("edges", []):
        if edge.get("pred") != "is_a":
            continue
        subject, obj = edge.get("sub", ""), edge.get("obj", "")
        if subject.startswith(CL_PREFIX) and obj.startswith(CL_PREFIX):
            parents_of.setdefault(_cl_id(subject), set()).add(_cl_id(obj))

    terms: dict[str, ClTerm] = {}
    for node in graph.get("nodes", []):
        identifier, label = node.get("id", ""), node.get("lbl")
        if not identifier.startswith(CL_PREFIX) or not label or node.get("meta", {}).get("deprecated"):
            continue
        cl_id = _cl_id(identifier)
        terms[cl_id] = ClTerm(cl_id=cl_id, name=label, parents=tuple(sorted(parents_of.get(cl_id, ()))))
    if not terms:
        raise ValueError(f"{source}: no CL terms parsed — the release format may have changed")
    return terms


def load_cl_graph(path: str | Path) -> dict[str, ClTerm]:
    """Parse an already-downloaded CL basic release (OBO Graph JSON) from disk."""
    graph = json.loads(Path(path).read_text(encoding="utf-8"))["graphs"][0]
    return parse_cl_graph(graph, source=str(path))


def fetch_cl(url: str = CL_BASIC, timeout: int = 120) -> dict[str, ClTerm]:
    """Download the CL basic release and convert it to our snapshot shape."""
    with urllib.request.urlopen(url, timeout=timeout) as response:
        payload = response.read().decode("utf-8")
    return parse_cl_graph(json.loads(payload)["graphs"][0], source=url)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Refresh a local ontology snapshot (needs network unless --from-file).")
    parser.add_argument("--what", choices=("cl",), default="cl", help="what to fetch (only 'cl' for now)")
    parser.add_argument("--out", required=True, help="where to write the snapshot, e.g. configs/seed/cl_terms.seed.yaml")
    origin = parser.add_mutually_exclusive_group()
    origin.add_argument("--url", default=CL_BASIC, help="release to download (default: the CL basic PURL)")
    origin.add_argument("--from-file", metavar="PATH", help="parse an already-downloaded release from disk (no network)")
    parser.add_argument("--source", help="provenance recorded in the snapshot (default: the --url value)")
    args = parser.parse_args(argv)
    try:
        terms = load_cl_graph(args.from_file) if args.from_file else fetch_cl(args.url)
    except (OSError, ValueError) as error:
        print(f"fetch failed: {error}", file=sys.stderr, flush=True)
        return 1
    digest = save_cl_terms(args.out, terms, source=args.source or args.url)
    print(f"wrote {len(terms)} CL terms to {args.out} (digest {digest[:12]}…)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
