"""The CL snapshot's lineage comes from the graph's ``is_a`` edges, never a node field.

Regression (every term parentless, silently): ``cl-basic.json`` is OBO Graph
JSON, so nodes and edges live in separate arrays and a node carries only
``id``/``lbl``/``meta``. A parser reading ``node["parents"]`` got an empty list
for all 3,335 terms and still produced a valid snapshot — an empty tuple is a
legal ``ClTerm``, so nothing failed and nothing downstream could tell "this is
the root" from "we lost the hierarchy".
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sckevtype.contracts.fetch import main, parse_cl_graph
from sckevtype.contracts.ontology import load_cl_terms

CL = "http://purl.obolibrary.org/obo/CL_"
UBERON = "http://purl.obolibrary.org/obo/UBERON_"
DEVELOPS_FROM = "http://purl.obolibrary.org/obo/RO_0002202"
PART_OF = "http://purl.obolibrary.org/obo/BFO_0000050"
RELEASE = Path(__file__).resolve().parents[2] / "data" / "external" / "cl-basic.json"


def graph(nodes: list[tuple[str, str]], edges: list[tuple[str, str, str]], deprecated: tuple[str, ...] = ()) -> dict:
    """A minimal OBO Graph JSON graph: nodes are ``(id, label)``, edges ``(sub, pred, obj)``."""
    return {
        "nodes": [
            {"id": identifier, "lbl": label, "type": "CLASS", **({"meta": {"deprecated": True}} if identifier in deprecated else {})}
            for identifier, label in nodes
        ],
        "edges": [{"sub": subject, "pred": pred, "obj": obj} for subject, pred, obj in edges],
    }


@pytest.fixture(scope="module")
def release_terms():
    """The vendored CL release, parsed once — skipped on a checkout without the 6 MB file."""
    if not RELEASE.exists():
        pytest.skip(f"vendored CL release not present: {RELEASE}")
    from sckevtype.contracts.fetch import load_cl_graph

    return load_cl_graph(RELEASE)


def test_parents_come_from_is_a_edges_not_from_a_node_field():
    """The bug itself: ``T cell is_a lymphocyte`` is an edge, and the node has no parents key."""
    terms = parse_cl_graph(
        graph(
            nodes=[(CL + "0000000", "cell"), (CL + "0000542", "lymphocyte"), (CL + "0000084", "T cell")],
            edges=[(CL + "0000084", "is_a", CL + "0000542"), (CL + "0000542", "is_a", CL + "0000000")],
        )
    )
    assert terms["CL:0000084"].parents == ("CL:0000542",)
    assert terms["CL:0000542"].parents == ("CL:0000000",)
    assert sum(1 for term in terms.values() if term.parents) == 2, "all-empty parents is the bug"


def test_only_is_a_edges_count_and_only_cl_parents_are_kept():
    """Ancestry is ``is_a``; the cell-type spine keeps CL parents and drops every other ontology."""
    terms = parse_cl_graph(
        graph(
            nodes=[(CL + "0000542", "lymphocyte"), (CL + "0000084", "T cell")],
            edges=[
                (CL + "0000084", "is_a", CL + "0000542"),
                (CL + "0000084", "is_a", UBERON + "0002405"),  # immune system: not a cell type
                (CL + "0000084", DEVELOPS_FROM, CL + "0000057"),  # a fibroblast is an origin, not a parent
                (CL + "0000084", PART_OF, CL + "0000084"),
            ],
        )
    )
    assert terms["CL:0000084"].parents == ("CL:0000542",)


def test_multiple_parents_are_sorted_and_deduplicated():
    terms = parse_cl_graph(
        graph(
            nodes=[
                (CL + "0000168", "insulin secreting cell"),
                (CL + "0008024", "pancreatic endocrine cell"),
                (CL + "0000169", "type B pancreatic cell"),
            ],
            edges=[
                (CL + "0000169", "is_a", CL + "0008024"),
                (CL + "0000169", "is_a", CL + "0000168"),
                (CL + "0000169", "is_a", CL + "0000168"),
            ],
        )
    )
    assert terms["CL:0000169"].parents == ("CL:0000168", "CL:0008024")


def test_obsolete_terms_are_still_skipped():
    """The deprecation filter is unchanged: an obsolete term is not a join key."""
    terms = parse_cl_graph(
        graph(
            nodes=[(CL + "0000542", "lymphocyte"), (CL + "0000999", "obsolete lymphocyte")],
            edges=[(CL + "0000999", "is_a", CL + "0000542")],
            deprecated=(CL + "0000999",),
        )
    )
    assert set(terms) == {"CL:0000542"}


def test_the_vendored_release_carries_its_known_hierarchy(release_terms):
    """Two terms whose parents are documented, plus the root — the check the bug failed."""
    assert release_terms["CL:0000084"].name == "T cell"
    assert release_terms["CL:0000084"].parents == ("CL:0000542",)  # lymphocyte
    assert release_terms["CL:0000169"].parents == ("CL:0000168", "CL:0008024")
    assert release_terms["CL:0000000"].parents == (), "the root has no CL parent"
    with_parents = sum(1 for term in release_terms.values() if term.parents)
    assert with_parents >= 3300, f"hierarchy lost again: only {with_parents} of {len(release_terms)} terms carry parents"


def test_the_cli_rebuilds_the_snapshot_offline_from_a_local_file(tmp_path, release_terms):
    """``--from-file`` is how the seed is refreshed on a box with no network."""
    out = tmp_path / "cl_terms.seed.yaml"
    assert main(["--what", "cl", "--from-file", str(RELEASE), "--out", str(out)]) == 0
    terms = load_cl_terms(str(out))
    assert len(terms) == len(release_terms) == 3335  # the vendored release, pinned by artifacts/v2
    assert terms["CL:0000084"].parents == ("CL:0000542",)
    assert terms["CL:0000000"].parents == ()
    assert sum(1 for term in terms.values() if term.parents) >= 3300
