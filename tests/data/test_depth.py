"""The depth layer: one parent per term, per-dataset roots, evidence-driven targets.

The unit-level tests build a seven-term synthetic ontology, so they run on a box with no
artifacts at all. The tests over the real CL snapshot are guarded: they run where
``artifacts/v3/cl_terms.json`` exists, and skip cleanly where it does not.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from sckevtype.contracts.definitions import Definition, DefinitionSet
from sckevtype.data import depth as D
from sckevtype.data.families import HoldsRule

ROOT = Path(__file__).resolve().parents[2]
V3 = ROOT / "artifacts" / "v3"

#: The 10 terms that carry only ``part_of`` links, so no CL parent exists for them
#: (``reports/cl-hierarchy-recovery.md`` §3). ``CL:0000000`` is the eleventh, and is
#: the root the ontology actually has.
PART_OF_ONLY = frozenset(
    {
        "CL:0017500",
        "CL:0017502",
        "CL:0017503",
        "CL:0017504",
        "CL:0017506",
        "CL:0017507",
        "CL:0017508",
        "CL:0017509",
        "CL:4052024",
        "CL:4052025",
    }
)

#: The synthetic ontology: ``0255 -> {0001, 0002}``, ``0001 -> {0003, 0005}``, plus
#: ``0004`` hanging straight off the CL root. The names say the role, not CL's.
CUT = "CL:0000255"
ROOT_CL = "CL:0000000"
ALPHA, BETA, ALPHA_ONE, ALPHA_TWO, GAMMA = "CL:0000001", "CL:0000002", "CL:0000003", "CL:0000005", "CL:0000004"
TREE = {
    CUT: (ROOT_CL,),
    ROOT_CL: (),
    ALPHA: (CUT,),
    BETA: (CUT,),
    ALPHA_ONE: (ALPHA,),
    ALPHA_TWO: (ALPHA,),
    GAMMA: (ROOT_CL,),
}

MARKERS = DefinitionSet(
    definitions=(
        Definition(cl_id=CUT, name="eukaryotic cell", pos_markers=("EEEE",)),
        Definition(cl_id=ALPHA, name="alpha", pos_markers=("AAAA",), neg_markers=("BBBB",)),
        Definition(cl_id=BETA, name="beta", pos_markers=("BBBB",)),
        Definition(cl_id=ALPHA_ONE, name="alpha one", pos_markers=("A1A1",)),
        Definition(cl_id=ALPHA_TWO, name="alpha two", pos_markers=("A2A2",)),
        Definition(cl_id=GAMMA, name="gamma", pos_markers=("CCCC",)),
    )
)
REGISTER = (CUT, ALPHA, BETA, ALPHA_ONE, ALPHA_TWO, GAMMA)


def _terms(pairs: dict[str, tuple[str, ...]]) -> dict[str, D.ClTerm]:
    return {cl_id: D.ClTerm(cl_id=cl_id, name=cl_id, parents=parents) for cl_id, parents in pairs.items()}


@pytest.fixture
def toy_spine() -> D.Spine:
    return D.build_spine(_terms(TREE))


@pytest.fixture
def toy_register(toy_spine) -> D.RegisterTree:
    return D.RegisterTree(toy_spine, REGISTER)


def _target(markers, chain, register, definitions=MARKERS, *, rule="discriminant", root=CUT):
    return D.depth_target(set(markers), chain, register, root, definitions, HoldsRule(), rule=rule)


# --- the spine: one parent per term -------------------------------------------------


def test_every_term_gets_exactly_one_parent_or_none(toy_spine):
    assert set(toy_spine.parents) == set(toy_spine.chains)
    assert toy_spine.parents[ALPHA_ONE] == ALPHA
    assert toy_spine.parents[BETA] == CUT
    assert toy_spine.parents[ROOT_CL] is None


def test_chains_are_acyclic_and_end_at_a_parentless_term(toy_spine):
    for cl_id, chain in toy_spine.chains.items():
        assert chain[0] == cl_id
        assert len(set(chain)) == len(chain), "a cycle would repeat a term"
        assert toy_spine.parents[chain[-1]] is None
        assert toy_spine.top[cl_id] == chain[-1]


def test_a_multi_parent_term_takes_the_larger_branch_not_the_first_parent():
    terms = _terms(
        {
            CUT: (),
            "CL:0000001": (CUT,),  # the bigger branch
            "CL:0000002": (CUT,),
            "CL:0000003": ("CL:0000001",),
            "CL:0000009": ("CL:0000002", "CL:0000001"),
        }
    )
    spine = D.build_spine(terms)
    assert spine.parents["CL:0000009"] == "CL:0000001"
    assert spine.ties == {}, "no tie here: the branches differ in size"


def test_a_tie_is_recorded_rather_than_hidden():
    terms = _terms({CUT: (), "CL:0000001": (CUT,), "CL:0000002": (CUT,), "CL:0000009": ("CL:0000002", "CL:0000001")})
    spine = D.build_spine(terms)
    assert spine.parents["CL:0000009"] == "CL:0000001"
    assert spine.ties["CL:0000009"] == ("CL:0000001", "CL:0000002")


def test_prefer_cut_is_the_only_difference_between_the_two_rules():
    terms = _terms(
        {
            ROOT_CL: (),
            CUT: (ROOT_CL,),
            "CL:0000001": (ROOT_CL,),  # a branch of three
            "CL:0000002": ("CL:0000001",),
            "CL:0000003": ("CL:0000001",),
            "CL:0000009": ("CL:0000001", CUT),
        }
    )
    assert D.build_spine(terms, prefer_cut=True).parents["CL:0000009"] == CUT
    assert D.build_spine(terms, prefer_cut=False).parents["CL:0000009"] == "CL:0000001"


def test_parents_are_terms_of_the_snapshot():
    terms = _terms({CUT: (), "CL:0000001": (CUT,), "CL:0000002": ("CL:9999999", "CL:0000001")})
    spine = D.build_spine(terms)
    assert spine.parents["CL:0000002"] == "CL:0000001", "a parent outside the snapshot cannot be walked"


# --- the register projection: levels only ask what has a definition ------------------


def test_chain_from_is_root_side_first_and_register_only(toy_register):
    assert toy_register.chain_from(ALPHA_ONE, CUT) == (ALPHA, ALPHA_ONE)
    assert toy_register.chain_from(ALPHA, CUT) == (ALPHA,)
    assert toy_register.chain_from(CUT, CUT) == ()
    assert toy_register.chain_from(GAMMA, CUT) == (), "GAMMA is not under the root"


def test_level_one_is_the_roots_register_children(toy_register):
    assert set(toy_register.level_one(CUT)) == {ALPHA, BETA}
    assert set(toy_register.level_one(ALPHA)) == {ALPHA_ONE, ALPHA_TWO}


def test_a_non_register_root_still_starts_a_descent(toy_register):
    # CL:0000000 is not a register term, so GAMMA's level 1 is GAMMA itself.
    assert toy_register.chain_from(GAMMA, ROOT_CL) == (GAMMA,)
    assert set(toy_register.level_one(ROOT_CL)) == {CUT, GAMMA}


def test_options_are_the_sibling_list_at_a_node(toy_register):
    assert set(toy_register.options(ALPHA, CUT)) == {ALPHA_ONE, ALPHA_TWO}
    assert set(toy_register.options(CUT, CUT)) == {ALPHA, BETA}


# --- the depth target is evidence-driven (SPEC §4 / plan E1.6) -----------------------


def test_no_evidence_no_depth_under_every_rule(toy_register):
    for rule in D.RULES:
        target, levels = _target(set(), (ALPHA, ALPHA_ONE), toy_register, rule=rule)
        assert target == 0, rule
        assert levels[0].separable is False


def test_the_descent_climbs_while_the_evidence_lasts(toy_register):
    chain = (ALPHA, ALPHA_ONE)
    assert _target({"AAAA"}, chain, toy_register)[0] == 1  # level 1 only
    assert _target({"AAAA", "A1A1"}, chain, toy_register)[0] == 2  # both levels
    assert _target({"A2A2"}, chain, toy_register)[0] == 0  # ALPHA_TWO's marker does not carry ALPHA


def test_the_target_is_a_prefix_not_the_deepest_separable_level(toy_register):
    """A separated level further down does not rescue a level the evidence failed."""
    target, levels = _target({"A1A1"}, (ALPHA, ALPHA_ONE), toy_register)
    assert target == 0
    assert levels[0].code() == "1:nodisc(1of1)"


def test_a_level_stops_for_the_rule_that_is_running(toy_register):
    _, levels = _target({"AAAA"}, (ALPHA, ALPHA_ONE), toy_register, rule="discriminant")
    assert levels[0].code() == "1:sep"
    assert levels[1].code() == "2:nodisc(1of1)"
    _, held = _target({"CCCC"}, (ALPHA,), toy_register, rule="holds")
    assert held[0].code() == "1:none", "no candidate's positive marker is present"
    _, wrong = _target({"BBBB"}, (ALPHA,), toy_register, rule="holds")
    assert wrong[0].code() == "1:other", "exactly one candidate holds, and it is not the true child"


def test_target_never_exceeds_the_chain_length(toy_register):
    for rule in D.RULES:
        for markers in (set(), {"AAAA"}, {"AAAA", "A1A1"}, {"EEEE", "AAAA", "A1A1", "A2A2"}):
            target, _ = _target(markers, (ALPHA, ALPHA_ONE), toy_register, rule=rule)
            assert 0 <= target <= 2


def test_unknown_rule_is_an_error_not_a_silent_fallback(toy_register):
    with pytest.raises(ValueError, match="unknown rule"):
        _target({"AAAA"}, (ALPHA,), toy_register, rule="vibes")


# --- what a level asks about: the record's own register -------------------------------

#: A panel whose evidence cannot separate ALPHA from BETA (they share their only marker),
#: so the level is unanswerable over the whole register and answerable over a register
#: that does not contain BETA — the tissue-scoping case in miniature.
SHARED_MARKER = DefinitionSet(
    definitions=(
        Definition(cl_id=CUT, name="eukaryotic cell", pos_markers=("EEEE",)),
        Definition(cl_id=ALPHA, name="alpha", pos_markers=("AAAA",)),
        Definition(cl_id=BETA, name="beta", pos_markers=("AAAA",)),
        Definition(cl_id=ALPHA_ONE, name="alpha one", pos_markers=("A1A1",)),
        Definition(cl_id=ALPHA_TWO, name="alpha two", pos_markers=("A2A2",)),
    )
)


def test_a_level_over_the_whole_register_stops_where_the_record_own_register_would_not(toy_register):
    state = {"AAAA", "A1A1"}
    over_everything, _ = _target(state, (ALPHA, ALPHA_ONE), toy_register, SHARED_MARKER)
    scoped, _ = D.depth_target(
        state, (ALPHA, ALPHA_ONE), toy_register, CUT, SHARED_MARKER, HoldsRule(), scope=frozenset({ALPHA, ALPHA_ONE})
    )
    assert over_everything == 0, "BETA shares ALPHA's only marker, so the level cannot be answered"
    assert scoped == 2, "without BETA in the option list the same evidence reaches both levels"


def test_scoping_keeps_the_true_child_even_when_it_is_outside_the_scope(toy_spine):
    tree = {ROOT_CL: (), CUT: (ROOT_CL,), ALPHA: (CUT,), BETA: (CUT,), "CL:0000006": (CUT,)}
    register = D.RegisterTree(D.build_spine(_terms(tree)), (CUT, ALPHA, BETA, "CL:0000006"))
    options = register.options(CUT, CUT)  # level one of a descent from the cut
    assert set(options) == {ALPHA, BETA, "CL:0000006"}
    scoped = D._scoped_options(options, ALPHA, frozenset({"CL:0000006"}))
    assert scoped == (ALPHA, "CL:0000006"), "the question is still asked, minus the sibling the panel never had"


def test_scoping_does_not_change_a_level_that_was_already_narrow(toy_register):
    target, levels = D.depth_target(
        {"AAAA", "A1A1"},
        (ALPHA, ALPHA_ONE),
        toy_register,
        CUT,
        SHARED_MARKER,
        HoldsRule(),
        scope=frozenset(REGISTER),
    )
    assert target == 0, "the whole register as the scope is the as-built question"
    assert len(levels) == 1


# --- the dataset roots: the plan's rule, and the two measured exceptions --------------


def _toy_counts() -> dict[str, dict[str, int]]:
    # ALPHA covers 98%, so the plan's rule (deepest ≥95%) lands on it, not on the cut.
    return {"toy": {ALPHA_ONE: 960, ALPHA_TWO: 20, GAMMA: 20}}


def test_the_plan_rule_takes_the_deepest_node_covering_the_dataset(toy_spine, toy_register):
    roots = D.dataset_roots(_toy_counts(), toy_spine, toy_register, {})
    assert [row.cl_id for row in roots] == [ALPHA]
    assert roots[0].coverage == pytest.approx(0.98)
    assert roots[0].override == ""


def test_an_override_moves_the_root_and_says_what_it_replaced(toy_spine, toy_register):
    names = {ALPHA: "alpha", CUT: "eukaryotic cell"}
    roots = D.dataset_roots(_toy_counts(), toy_spine, toy_register, names)
    moved = D.apply_root_overrides(roots, _toy_counts(), toy_spine, toy_register, names, overrides={"toy": CUT})
    assert [row.cl_id for row in moved] == [CUT]
    assert moved[0].override == f"replaces {ALPHA} alpha"
    assert moved[0].coverage == pytest.approx(0.98), "the override is measured with the rule's own arithmetic"


def test_an_override_that_fails_the_plans_coverage_test_is_an_error():
    tree = {
        ROOT_CL: (),
        CUT: (ROOT_CL,),
        ALPHA: (CUT,),
        BETA: (CUT,),
        "CL:0000006": (BETA,),
        "CL:0000007": (ALPHA,),
    }
    spine = D.build_spine(_terms(tree))
    register = D.RegisterTree(spine, (CUT, ALPHA, BETA, "CL:0000006", "CL:0000007"))
    counts = {"toy": {ALPHA: 98, BETA: 2}}
    roots = D.dataset_roots(counts, spine, register, {})
    assert roots[0].cl_id == ALPHA, "the plan's rule takes the deepest node that still has something to ask"
    with pytest.raises(ValueError, match="below the plan's"):
        D.apply_root_overrides(roots, counts, spine, register, {}, overrides={"toy": BETA})


def test_an_override_root_with_nothing_to_ask_about_is_an_error(toy_spine, toy_register):
    roots = D.dataset_roots(_toy_counts(), toy_spine, toy_register, {})
    only_the_cut = D.RegisterTree(toy_spine, (CUT,))
    with pytest.raises(ValueError, match="no register children"):
        D.apply_root_overrides(roots, _toy_counts(), toy_spine, only_the_cut, {}, overrides={"toy": CUT})


def test_an_override_root_outside_the_snapshot_is_an_error(toy_spine, toy_register):
    roots = D.dataset_roots(_toy_counts(), toy_spine, toy_register, {})
    with pytest.raises(ValueError, match="no CL term"):
        D.apply_root_overrides(roots, _toy_counts(), toy_spine, toy_register, {}, overrides={"toy": "CL:9999999"})


# --- symbol case: a curation property, not a panel property --------------------------


def test_folding_matches_a_mouse_style_marker_to_a_human_style_state(toy_register):
    definitions = DefinitionSet(
        definitions=(
            Definition(cl_id=CUT, name="eukaryotic cell"),
            Definition(cl_id=ALPHA, name="alpha", pos_markers=("Cd3e", "CD8A")),
            Definition(cl_id=BETA, name="beta", pos_markers=("MS4A1",)),
        )
    )
    plain, _ = _target({"CD3E"}, (ALPHA,), toy_register, definitions)
    folded, _ = D.depth_target(
        D.presence({"CD3E"}, "fold"),
        (ALPHA,),
        toy_register,
        CUT,
        D.with_case(definitions, "fold"),
        HoldsRule(),
        rule="discriminant",
    )
    assert plain == 0, "exact matching sees no marker of ALPHA at all"
    assert folded == 1


def test_with_case_does_not_mutate_the_frozen_definition_set():
    before = MARKERS.model_dump(mode="json")
    D.with_case(MARKERS, "fold")
    assert MARKERS.model_dump(mode="json") == before


def test_rule_names_the_exact_case_plainly():
    assert D.rule_name(("holds", "exact")) == "holds"
    assert D.rule_name(("discriminant", "fold")) == "discriminant.fold"


# --- the gate, as arithmetic ---------------------------------------------------------


def test_gate_verdict_is_the_plans_two_conditions():
    passing = Counter({0: 2, 3: 8, 4: 10, 5: 10})
    verdict = D.gate_verdict(passing, 30, 30, 2, Counter(passing), 30)
    assert verdict["median_depth_all_records"] >= 3
    assert verdict["share_at_root_all_records"] < 0.20
    assert verdict["passes_all_records"] is True and verdict["passes_cells"] is True
    failing = D.gate_verdict(Counter({0: 20, 1: 10}), 30, 30, 20, Counter({0: 20, 1: 10}), 30)
    assert failing["passes_all_records"] is False and failing["passes_cells"] is False


def test_gate_counts_root_abstentions_over_the_records_the_descent_starts_on():
    """An off-register record is not a depth of 0; it is a record with no descent."""
    counter = Counter({0: 10, 1: 10, 2: 10, 3: 10, 4: 10, 5: 10})
    verdict = D.gate_verdict(counter, 60, 50, 0, Counter(counter), 60)
    assert verdict["share_at_root_all_records"] == pytest.approx(round(10 / 60, 6))
    assert verdict["share_at_root_on_chain"] == 0.0
    assert verdict["on_chain_records"] == 50


def test_the_three_definitions_share_one_threshold_and_differ_by_population():
    """The gate is read three ways; a different reading is not a different criterion."""
    entry = {
        "records": {"n": 100},
        "cells": {"n": 20},
        "gate": {
            "median_depth_all_records": 2.0,
            "share_at_root_all_records": 0.19,
            "passes_all_records": False,
            "median_depth_on_chain": 3.0,
            "share_at_root_on_chain": 0.05,
            "passes_on_chain": True,
            "on_chain_records": 50,
            "median_depth_cells": 2.0,
            "share_at_root_cells": 0.05,
            "passes_cells": False,
        },
    }
    readings = D.gate_definitions(entry)
    assert tuple(readings) == D.GATE_DEFINITIONS == ("all_records", "on_chain", "cells")
    assert readings["on_chain"] == {"n": 50, "median": 3.0, "share_at_root": 0.05, "passes": True}
    assert readings["all_records"] == {"n": 100, "median": 2.0, "share_at_root": 0.19, "passes": False}
    assert readings["cells"] == {"n": 20, "median": 2.0, "share_at_root": 0.05, "passes": False}
    # every reading is the same conjunction
    for reading in readings.values():
        assert reading["passes"] == D.passes_gate(reading["median"], reading["share_at_root"])
    assert D.passes_gate(3, 0.199) is True and D.passes_gate(3, 0.20) is False
    assert D.passes_gate(2.5, 0.01) is False


# --- over the real snapshot, when it is present --------------------------------------


@pytest.fixture(scope="module")
def real_terms() -> dict[str, D.ClTerm]:
    if not (V3 / "cl_terms.json").exists():
        pytest.skip("artifacts/v3/cl_terms.json is not built in this checkout")
    return D.load_terms(V3 / "cl_terms.json")


def test_real_snapshot_rejects_no_dangling_parent(real_terms):
    spine = D.build_spine(real_terms)
    assert set(spine.parents) <= set(real_terms)
    assert all(parent is None or parent in real_terms for parent in spine.parents.values())


def test_real_parentless_terms_are_the_root_and_the_part_of_only_ten(real_terms):
    assert set(D.build_spine(real_terms).parentless) == PART_OF_ONLY | {ROOT_CL}


def test_real_cut_carries_the_tree(real_terms):
    spine = D.build_spine(real_terms)
    below = [cl_id for cl_id, chain in spine.chains.items() if D.CUT in chain]
    assert len(below) > 3000, "the cut is the root of the usable tree"
    for cl_id in below:
        assert spine.chains[cl_id][-2:] == (D.CUT, ROOT_CL), f"{cl_id}: the cut is not the last step below the root"
    assert D.CUT not in spine.outside_cut
    assert spine.parents[D.CUT] == ROOT_CL


def test_real_register_chains_only_use_register_terms(real_terms):
    if not (V3 / "definitions.json").exists():
        pytest.skip("artifacts/v3/definitions.json is not built in this checkout")
    spine = D.build_spine(real_terms)
    register = D.RegisterTree(spine, D.register_terms(D.load_definitions(V3 / "definitions.json")))
    seen = 0
    for label in register.cl_ids:
        chain = register.chain_from(label, "CL:0000738")
        if not chain:
            continue
        seen += 1
        assert chain[-1] == label
        assert all(node in register.members for node in chain)
    assert seen > 10, "the leukocyte root covers a real register subtree"


def test_real_tissue_register_names_only_terms_of_the_snapshot(real_terms):
    """``tissue_map.json`` is derived, and derived from the same release: nothing dangles."""
    if not (V3 / D.TISSUE_MAP).exists():
        pytest.skip("artifacts/v3/tissue_map.json is not built in this checkout")
    scopes = D.load_tissue_register(V3 / D.TISSUE_MAP)
    assert set(scopes) == {"blood", "brain", "liver", "lung", "pancreas"}
    for tissue, cl_ids in scopes.items():
        assert cl_ids, f"{tissue}: an empty register is not a register"
        assert cl_ids <= set(real_terms), f"{tissue}: a term outside the CL snapshot"


def test_real_root_overrides_still_pass_the_plans_own_coverage_test(real_terms):
    """The two exceptions are claims about datasets; they have to keep holding."""
    if not (V3 / "definitions.json").exists() or not (V3 / D.TISSUE_MAP).exists():
        pytest.skip("the v3 register is not built in this checkout")
    seed = ROOT / "configs" / "seed" / "v3" / "atlas_labels.json"
    if not seed.exists():
        pytest.skip("the v3 atlas inventory is not built in this checkout")
    names = {cl_id: term.name for cl_id, term in real_terms.items()}
    spine = D.build_spine(real_terms)
    register = D.RegisterTree(spine, D.register_terms(D.load_definitions(V3 / "definitions.json")))
    datasets = D.atlas_label_counts(seed)
    roots = D.dataset_roots(datasets, spine, register, names)
    assert set(D.ROOT_OVERRIDES) <= set(datasets), "an override for a dataset this seed does not carry"
    moved = D.apply_root_overrides(roots, datasets, spine, register, names)
    by_dataset = {row.dataset_id: row for row in moved}
    for dataset_id, node in D.ROOT_OVERRIDES.items():
        row = by_dataset[dataset_id]
        assert row.cl_id == node
        assert row.coverage >= 0.95
        assert register.level_one(node), "an override root with nothing to ask about is not a root"
        assert row.override.startswith("replaces")
    kept = [row for row in moved if row.dataset_id not in D.ROOT_OVERRIDES]
    assert all(row.override == "" for row in kept), "only the two named datasets move"
