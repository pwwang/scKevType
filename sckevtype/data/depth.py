"""The ontology depth layer: a CL-only canonical spine, per-dataset roots, depth targets.

Design is `docs/SPEC-step14-ontology-curriculum.md` §3–§4; this module is its
implementation and its audit trail. Four objects, in order:

**The spine** (:func:`build_spine`). CL is a DAG and its upper structure is not a
cell-type hierarchy: `type B pancreatic cell` has 44 ancestors including BFO and
CARO terms, and thirty-two CL terms hang directly off `cell` (CL:0000000), of
which only one is `eukaryotic cell` (CL:0000255). A descent needs *one* parent
per term, so the spine is a choice, and the choice is recorded rather than
implied:

* **the cut** — the spine is rooted at `eukaryotic cell` (CL:0000255), the
  deliberate cut SPEC §3 states. A term that has *any* route into that subtree is
  routed through it, so its chain ends at CL:0000255 and the E1.3 audit check
  holds;
* **the branch** — where a term still has several CL parents (1,148 of 3,324 have
  more than one, and no two of them are ancestor-related, so the choice is real),
  SPEC's rule applies: the parent with the largest number of CL descendants,
  ties broken by the lowest CL id. Every tie is recorded;
* **the exceptions** — the terms that reach `cell` without ever passing through
  `eukaryotic cell` are not cells that fail to be eukaryotic; they are terms CL
  attaches to the whole-ontology root (`blood cell`, `secretory cell`,
  `motile cell`, `hematopoietic cell`, …). No parent choice can route them
  through the cut, so they are listed as the cut's honest exceptions rather than
  forced onto a node, and the 11 parentless terms (the root, plus ten cytoplasm /
  nucleus / junction terms that carry `part_of` links only) are listed with them.

**The register tree** (:class:`RegisterTree`). The descent's questions can only
name types the register has definitions for, so levels are steps between
*register* terms — skipping the intermediate nodes the model could never name.
The parent of a register term is the next register term up its spine chain, so
the register tree is a projection of the spine, not a second structure.

**The dataset roots** (:func:`dataset_roots`). E1.4: the deepest node covering
≥95% of a dataset's labels, counted from the root (`coverage` = the share under
it). Coverage is weighted by cells by default, because the root exists to bound a
*corpus* and cells are what a corpus spends; the label-weighted root is reported
beside it so the two can be compared rather than conflated.

**The depth targets** (:func:`depth_target`). SPEC §4: a record's target depth is
the deepest level ℓ such that the true child at ℓ is separable from its siblings
using only the evidence the state carries — read with the existing
:class:`~sckevtype.data.families.HoldsRule`, not a new heuristic:

    separable(ℓ)  ⟺  the true child's definition holds of the state
                      and no sibling's definition holds

"Exactly one candidate is supported" is what makes the child identifiable. The
target is the deepest ℓ whose *whole prefix* 1…ℓ is separable: the descent walks
one level at a time, so a level the state cannot support stops the walk, and that
is what makes "stop here" a claim about the panel rather than a learned mystery.
A level with a single candidate is separable-for-free and is counted separately,
because a curriculum of one-option questions would be no curriculum at all.

**What a level asks about** is not only a property of the ontology. A level's options
are register terms, and the register is *global*: as built, a pancreatic islet state
is asked to eliminate `hepatocyte`, `keratinocyte`, `neuron` and `kidney collecting
duct principal cell` before it can be called an epithelial cell — types its 200-gene
panel was never drawn to choose between, and which no lab could answer from it. The
corpus's own panels *were* drawn from a tissue register, named on every record
(``_meta.panel_scope``) and derived by rule in ``artifacts/v3/tissue_map.json``
(``tissues[T] = scope ∩ (cellmarker(T) ∪ observed(T))``). :func:`depth_target`
therefore takes a ``scope``: the level's options narrowed to the record's own
register, always keeping the true child so the question stays answerable. No scope
is the as-built behaviour, and every scoped number is reported beside it.

**The dataset roots** (:func:`dataset_roots`, above) are the plan's rule. It
maximizes *coverage*, which says nothing about whether a descent from the chosen
node has levels the record's evidence can carry — and for two datasets it does not:
:data:`ROOT_OVERRIDES` records those two, each with the measurement that says so.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from ..contracts.freeze import ENCODING, file_digest, read_json, text_digest, write_text
from ..contracts.ontology import ClTerm
from ..contracts.render import FilterConfig
from .families import HoldsRule, state_symbols

#: CL's own root — the whole-ontology root, not ours.
CL_ROOT = "CL:0000000"
#: The spine's root: SPEC §3's deliberate cut. Every chain that can reach it does.
CUT = "CL:0000255"
#: The corpus splits, in the corpus's own order.
SPLITS = ("train", "calibration", "development", "test")

#: Where the tissue registers live, inside the artifacts directory the layer reads.
#: ``tissue_map.json`` is *derived*, not hand-written: its own ``source`` states the
#: rule (``tissues[T] = scope ∩ (cellmarker(T) ∪ observed(T))``).
TISSUE_MAP = "tissue_map.json"

#: Per-dataset root overrides: the plan's E1.4 rule, with its two measured exceptions.
#:
#: E1.4 takes the *deepest* node covering ≥95% of a dataset's cells. Coverage is not
#: the only thing a root has to do — a descent from it also needs levels the records'
#: evidence can carry, and a node can be too deep for the register it descends into.
#: Each exception below moves one dataset's root *up* to a node the plan's own ≥95%
#: test still passes, because the deeper root measured worse on the gate
#: (``artifacts/cl/depth-comparison.json``; the frontier behind it is
#: ``scratch/cl-root-frontier-vex.py``):
#:
#: * ``siletti`` → `neural cell` (`CL:0002319`). E1.4 lands on `neuron`, which covers
#:   99.8% of the dataset's cells because the atlas is one brain region — but a large
#:   share of its *records* carry the atlas's non-neuronal labels (T, B, NK, monocyte,
#:   astrocyte, oligodendrocyte), which sit outside the neuron subtree and so never
#:   start a descent. `neural cell` covers the same 99.8% of cells and starts the
#:   descent on 24,392 records rather than 6,868.
#: * ``tabula_muris_senis`` → `cell` (`CL:0000000`). E1.4 lands on the cut
#:   (`eukaryotic cell`), which some of its labels do not pass through — the cut's
#:   documented exceptions (`blood cell`, `secretory cell`, `motile cell`, …). `cell`
#:   covers 100% of them and every gate statistic comes out equal or better.
#:
#: The other four datasets keep the plan's root: for each of them every ≥95%
#: alternative measured worse on the gate's own two clauses, and both the rejected
#: candidates and their numbers are in the comparison artifact.
ROOT_OVERRIDES: dict[str, str] = {"siletti": "CL:0002319", "tabula_muris_senis": CL_ROOT}


@dataclass
class Spine:
    """One parent per CL term, its chain to the top, and every choice it made."""

    parents: dict[str, str | None]
    chains: dict[str, tuple[str, ...]]
    top: dict[str, str]
    descendants: dict[str, frozenset[str]]
    #: Terms whose parents tied on the branch rule — the choice was arbitrary, so it is listed.
    ties: dict[str, tuple[str, ...]]
    #: Terms that reach the top without passing through the cut, and why.
    outside_cut: dict[str, str]
    #: Terms with no CL parent at all (CL's own root, and the `part_of`-only terms).
    parentless: tuple[str, ...]
    prefer_cut: bool

    @property
    def depth_from_top(self) -> dict[str, int]:
        """How far each term sits below the top of its own chain (0 = a top)."""
        return {cl_id: len(chain) - 1 for cl_id, chain in self.chains.items()}

    def passes_cut(self, cl_id: str) -> bool:
        return CUT in self.chains[cl_id]

    def payload(self, names: dict[str, str]) -> dict:
        """The ``spines.json`` payload: every term, its parent, and its chain."""
        rows = []
        for cl_id in sorted(self.chains):
            chain = self.chains[cl_id]
            rows.append(
                {
                    "cl_id": cl_id,
                    "name": names.get(cl_id, ""),
                    "parent": self.parents.get(cl_id),
                    "chain": list(chain),
                    "chain_length": len(chain),
                    "top": self.top[cl_id],
                    "passes_cut": CUT in chain,
                    "n_cl_descendants": len(self.descendants.get(cl_id, ())),
                    "tie": list(self.ties.get(cl_id, ())),
                }
            )
        return {
            "rule": "one CL parent per term: prefer a parent under the cut (CL:0000255); then the "
                    "largest CL descendant count; then the lowest CL id",
            "cut": CUT,
            "cl_root": CL_ROOT,
            "prefer_cut": self.prefer_cut,
            "parentless": list(self.parentless),
            "outside_cut": {cl_id: reason for cl_id, reason in sorted(self.outside_cut.items())},
            "n_terms": len(rows),
            "n_passes_cut": sum(1 for row in rows if row["passes_cut"]),
            "n_ties": sum(1 for row in rows if row["tie"]),
            "terms": rows,
        }


def build_spine(terms: dict[str, ClTerm], *, prefer_cut: bool = True) -> Spine:
    """One parent per term, deterministically, with the branch choice recorded.

    ``prefer_cut`` is the canonical rule: among a term's CL parents, those inside
    the cut's subtree are considered first, so every term that *can* be described
    as a eukaryotic cell is. Turning it off reproduces SPEC §3's literal rule
    (largest descendant count, nothing else), which is kept so the two can be
    compared rather than chosen by assertion.
    """
    children: dict[str, set[str]] = defaultdict(set)
    for cl_id, term in terms.items():
        for parent in term.parents:
            children[parent].add(cl_id)

    descendants = _descendant_sets(terms, children)
    targets = set(descendants.get(CUT, ())) | {CUT}

    parents: dict[str, str | None] = {}
    ties: dict[str, tuple[str, ...]] = {}
    for cl_id in sorted(terms):
        candidates = list(terms[cl_id].parents)
        if not candidates:
            parents[cl_id] = None
            continue
        # ``prefer_cut`` narrows the pool to the parents inside the cut's subtree,
        # falling back to all of them when a term has none there. With the flag off,
        # the pool is every CL parent: SPEC §3's literal rule, and the only difference
        # between the two.
        inside = [parent for parent in candidates if parent in targets] if prefer_cut else []
        pool = inside or candidates
        ranked = sorted(pool, key=lambda parent: (-len(descendants.get(parent, ())), parent))
        parents[cl_id] = ranked[0]
        if len(ranked) > 1 and len(descendants.get(ranked[0], ())) == len(descendants.get(ranked[1], ())):
            ties[cl_id] = tuple(ranked)

    chains = {cl_id: _chain(cl_id, parents, set(terms)) for cl_id in sorted(terms)}
    top = {cl_id: chain[-1] for cl_id, chain in chains.items()}
    outside = {
        cl_id: ("no_cl_parent" if parents[cl_id] is None else "no_path_through_cut")
        for cl_id, chain in chains.items()
        if CUT not in chain and cl_id != CUT
    }
    parentless = tuple(sorted(cl_id for cl_id, parent in parents.items() if parent is None))
    return Spine(
        parents=parents,
        chains=chains,
        top=top,
        descendants=descendants,
        ties=ties,
        outside_cut=outside,
        parentless=parentless,
        prefer_cut=prefer_cut,
    )


def _descendant_sets(terms: dict[str, ClTerm], children: dict[str, set[str]]) -> dict[str, frozenset[str]]:
    """Transitive CL descendants of every term, over the full is_a DAG.

    A property of the ontology, not of the spine: the branch rule is stated in
    terms of how large a branch is *in CL*, so computing it on a spine the rule
    itself produced would make the rule self-referential.
    """
    out: dict[str, frozenset[str]] = {}
    for cl_id in sorted(terms):
        seen: set[str] = set()
        stack = list(children.get(cl_id, ()))
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            stack.extend(children.get(node, ()))
        out[cl_id] = frozenset(seen)
    return out


def _chain(cl_id: str, parents: dict[str, str | None], known: set[str]) -> tuple[str, ...]:
    """``(self, parent, …, top)`` — cycle-safe, and stopping at a term with no CL parent."""
    chain, seen, current = [cl_id], {cl_id}, cl_id
    while True:
        parent = parents.get(current)
        if parent is None or parent not in known or parent in seen:
            break
        chain.append(parent)
        seen.add(parent)
        current = parent
    return tuple(chain)


class RegisterTree:
    """The register as a hierarchy: the descent's levels are steps between register terms.

    A level's option list is the definitions the model can read, and only the
    register has definitions, so a level cannot ask about a type with none.
    Projecting the spine onto the register keeps every level answerable: the
    parent of a register term is the *next register term up its spine chain*, so
    a chain's levels are exactly the register terms its spine path passes
    through.
    """

    def __init__(self, spine: Spine, cl_ids: tuple[str, ...]) -> None:
        self.spine = spine
        self.cl_ids = tuple(sorted(cl_ids))
        self.members = frozenset(self.cl_ids)
        self.parent: dict[str, str | None] = {
            cl_id: next((node for node in spine.chains[cl_id][1:] if node in self.members), None)
            for cl_id in self.cl_ids
        }
        self.children: dict[str, tuple[str, ...]] = {
            cl_id: tuple(child for child in self.cl_ids if self.parent[child] == cl_id)
            for cl_id in self.cl_ids
        }
        #: The spine chain of *every* CL term, not just the register's: a dataset
        #: root has to cover the labels that are out of register too (they are
        #: the corpus's `out_of_register` family), and only the descent is
        #: register-only.
        self._chain_index = {
            cl_id: {node: index for index, node in enumerate(chain)}
            for cl_id, chain in spine.chains.items()
        }

    def under(self, label: str, node: str) -> bool:
        """Is ``node`` on ``label``'s spine chain (itself included)?"""
        return node in self._chain_index.get(label, {})

    def chain_from(self, label: str, root: str) -> tuple[str, ...]:
        """The descent's levels: the register terms from just below ``root`` to ``label``.

        The chain is the sequence of questions a descent from the root asks, so its
        length is the label's depth and its entries are level 1, 2, …. Empty when
        the root is not on the label's chain (the ≤5% a root need not cover) and
        when the label *is* the root (nothing to ask). ``root`` may be any CL node:
        when it is not a register term, the label's lowest register term is level 1,
        because by construction no register term lies between the two.
        """
        if not self.under(label, root):
            return ()
        chain = [node for node in self.spine.chains[label] if node in self.members]
        if root in chain:
            return tuple(reversed(chain[: chain.index(root)]))
        return tuple(reversed(chain))

    def level_one(self, root: str) -> tuple[str, ...]:
        """The register terms a descent from ``root`` chooses between at level 1.

        A register term is level 1 when the root is on its chain and no register
        term sits *strictly* between the two: the root's own register children,
        plus every register term whose register parent is above the root (which is
        what makes a non-register root such as `eukaryotic cell` usable). A
        register root is *at* level 0, so it is not one of its own options.
        """
        out = []
        for cl_id in self.cl_ids:
            if cl_id == root or not self.under(cl_id, root):
                continue
            parent = self.parent[cl_id]
            if parent is None or parent == root or not self.under(parent, root):
                out.append(cl_id)
        return tuple(out)

    def options(self, node: str, root: str) -> tuple[str, ...]:
        """The sibling list at a level whose parent is ``node``."""
        return self.level_one(root) if node == root else self.children[node]

    def covered(self, root: str) -> tuple[str, ...]:
        """Every register term the root covers, in id order."""
        return tuple(cl_id for cl_id in self.cl_ids if self.under(cl_id, root))


@dataclass
class RootChoice:
    """One dataset's root, and what starting the descent there costs."""

    dataset_id: str
    cl_id: str
    name: str
    depth_from_top: int
    coverage: float
    labels_covered: int
    labels_total: int
    cells_covered: int
    cells_total: int
    label_coverage: float
    register_labels_covered: int
    median_depth: float
    median_spine_depth: float
    level_one: int
    label_weighted_root: str | None
    label_weighted_coverage: float
    #: Empty when the plan's own rule chose this root; otherwise what
    #: :data:`ROOT_OVERRIDES` replaced, so the artifact says which roots are the
    #: rule's and which are the two measured exceptions.
    override: str = ""


def _root_choice(
    dataset_id: str,
    root: str,
    counts: dict[str, int],
    spine: Spine,
    register: RegisterTree,
    names: dict[str, str],
    *,
    label_root: str | None = None,
    override: str = "",
) -> RootChoice:
    """One node's numbers as a dataset root: coverage, register labels below it, median descent.

    The builder is shared by :func:`dataset_roots` and :func:`apply_root_overrides`,
    so an override is measured with exactly the arithmetic of the node it replaces.
    """
    cells_total = sum(counts.values())
    labels_total = len(counts)
    covered = [cl_id for cl_id in counts if register.under(cl_id, root)]
    depths, weights, spine_depths = [], [], []
    register_covered = 0
    for cl_id, cells in counts.items():
        if not register.under(cl_id, root):
            continue
        spine_depths.append(spine.chains[cl_id].index(root))
        if cl_id in register.members:
            register_covered += 1
            depths.append(len(register.chain_from(cl_id, root)))
            weights.append(cells)
    return RootChoice(
        dataset_id=dataset_id,
        cl_id=root,
        name=names.get(root, ""),
        depth_from_top=spine.depth_from_top[root],
        coverage=round(sum(counts[cl_id] for cl_id in covered) / cells_total, 6),
        labels_covered=len(covered),
        labels_total=labels_total,
        cells_covered=sum(counts[cl_id] for cl_id in covered),
        cells_total=cells_total,
        label_coverage=round(len(covered) / labels_total, 6),
        register_labels_covered=register_covered,
        median_depth=_weighted_median(depths, weights),
        median_spine_depth=_weighted_median(spine_depths, weights),
        level_one=len(register.level_one(root)),
        label_weighted_root=label_root,
        label_weighted_coverage=(
            round(
                len([cl_id for cl_id in counts if label_root and register.under(cl_id, label_root)]) / labels_total,
                6,
            )
            if label_root
            else 0.0
        ),
        override=override,
    )


def dataset_roots(
    datasets: dict[str, dict[str, int]],
    spine: Spine,
    register: RegisterTree,
    names: dict[str, str],
    *,
    min_coverage: float = 0.95,
) -> list[RootChoice]:
    """One root per dataset: the deepest node covering ≥95% of its cells (E1.4).

    ``datasets`` maps a dataset id to ``cl_id -> cells``. Non-CL labels (an
    atlas's ``unknown``) are out of the denominator by construction — they have no
    position in CL — and appear in the inventory instead. A node is only a
    candidate if it has register children, since a root with none leaves the
    descent with nothing to ask.
    """
    depth_from_top = spine.depth_from_top
    out: list[RootChoice] = []
    for dataset_id in sorted(datasets):
        counts = {cl_id: n for cl_id, n in datasets[dataset_id].items() if cl_id in spine.chains}
        cells_total = sum(counts.values())
        labels_total = len(counts)
        by_cells, by_labels = [], []
        for candidate in sorted(spine.chains):
            if not register.level_one(candidate):
                continue
            covered = [cl_id for cl_id in counts if register.under(cl_id, candidate)]
            if not covered:
                continue
            share_cells = sum(counts[cl_id] for cl_id in covered) / cells_total
            share_labels = len(covered) / labels_total
            if share_cells >= min_coverage:
                by_cells.append((depth_from_top[candidate], share_cells, len(covered), candidate))
            if share_labels >= min_coverage:
                by_labels.append((depth_from_top[candidate], share_labels, len(covered), candidate))
        if not by_cells:
            raise ValueError(f"{dataset_id}: no CL node covers {min_coverage:.0%} of its cells")
        _, _share_cells, _n_covered, root = max(by_cells)
        label_root = max(by_labels)[3] if by_labels else None
        out.append(_root_choice(dataset_id, root, counts, spine, register, names, label_root=label_root))
    return out


def apply_root_overrides(
    roots: list[RootChoice],
    datasets: dict[str, dict[str, int]],
    spine: Spine,
    register: RegisterTree,
    names: dict[str, str],
    *,
    overrides: dict[str, str] | None = None,
    min_coverage: float = 0.95,
) -> list[RootChoice]:
    """The plan's roots with :data:`ROOT_OVERRIDES` applied, and the plan's own tests re-run.

    An override has to pass the test it replaces: the node must cover ≥95% of the
    dataset's cells *and* have register children to ask about, or the override is an
    error rather than a silent fall-back to the plan's root. A claim about a dataset
    that no longer holds is a claim that has to be loud — the register it was made
    against can change under it.
    """
    overrides = ROOT_OVERRIDES if overrides is None else overrides
    out: list[RootChoice] = []
    for row in roots:
        node = overrides.get(row.dataset_id)
        if node is None or node == row.cl_id:
            out.append(row)
            continue
        if node not in spine.chains:
            raise ValueError(f"{row.dataset_id}: override root {node} is no CL term of the snapshot")
        if not register.level_one(node):
            raise ValueError(f"{row.dataset_id}: override root {node} has no register children to ask about")
        counts = {cl_id: n for cl_id, n in datasets[row.dataset_id].items() if cl_id in spine.chains}
        replacement = _root_choice(
            row.dataset_id,
            node,
            counts,
            spine,
            register,
            names,
            label_root=row.label_weighted_root,
            override=f"replaces {row.cl_id} {row.name}".rstrip(),
        )
        if replacement.coverage < min_coverage:
            raise ValueError(
                f"{row.dataset_id}: override root {node} covers {replacement.coverage:.3f} of its cells, "
                f"below the plan's {min_coverage:.0%} test"
            )
        out.append(replacement)
    return out


def _weighted_median(values: list[int], weights: list[int]) -> float:
    """The value at half the weight — the median a corpus-weighted reader expects."""
    pairs = sorted(zip(values, weights))
    total = sum(weights)
    if not total:
        return 0.0
    running = 0
    for index, (value, weight) in enumerate(pairs):
        running += weight
        if running * 2 >= total:
            if running * 2 == total and index + 1 < len(pairs):
                return (value + pairs[index + 1][0]) / 2
            return float(value)
    return float(pairs[-1][0])


@dataclass
class LevelResult:
    """One level of one descent: what was on offer, and whether the evidence chose."""

    level: int
    node: str
    options: tuple[str, ...]
    holds: tuple[str, ...]
    separable: bool
    singleton: bool
    rule: str = "holds"
    blocked: tuple[str, ...] = ()

    def code(self) -> str:
        """A compact per-level code for the row dump: separable / why not.

        The code names the *rule's* reason, not the holds rule's: a
        ``discriminant`` level that stops because four of eleven siblings could
        not be separated says so, rather than reporting which candidates happen
        to hold under a rule the target was not computed with.
        """
        if self.separable:
            return f"{self.level}:sep" + (":1opt" if self.singleton else "")
        if self.rule != "holds":
            return f"{self.level}:nodisc({len(self.blocked)}of{len(self.options) - 1})"
        if not self.holds:
            return f"{self.level}:none"
        if len(self.holds) == 1:
            # Exactly one candidate holds and — since the level was not separable —
            # it is the wrong one. The distinction matters: "none" is a panel that
            # says nothing, "other" is a panel that says the wrong thing.
            return f"{self.level}:other"
        return f"{self.level}:amb({len(self.holds)})"


#: How marker symbols are matched. ``exact`` is what the frozen flat families do:
#: a definition's marker must appear in the state with the same spelling.
#: ``fold`` upper-cases both sides first. The register's markers mix conventions —
#: 99 of 710 are mouse-style (``Cd3e``, ``C1qa``) while a human atlas's state
#: spells the same gene ``CD3E`` — so under ``exact`` those markers are
#: unmeasurable in every human dataset, which is a property of the curation and
#: not of the panel. Both are computed and reported; neither is assumed.
CASES = ("exact", "fold")

#: The separability rules the target can be computed with.
#:
#: ``holds`` is SPEC §4's proposal, verbatim: the true child's definition holds of
#: the state and no sibling's does — the same :class:`HoldsRule` the ``marker_noul``
#: family uses. ``discriminant`` is the fallback SPEC §4 records ("panel-coverage …
#: the fraction of the level's discriminative markers present"), taken at its
#: threshold-free reading: for every sibling there is a marker the true child
#: claims, the sibling does not, and the state carries. That is the check
#: `docs/panel-design-discovery.md` §2c describes as mechanical — "intersect the
#: panel with the pair's discriminating markers" — and it needs no threshold that
#: nobody can defend. ``discriminant_pos`` is the same test restricted to the true
#: child's *positive* markers, i.e. evidence genuinely in favour of the child.
RULES = ("holds", "discriminant", "discriminant_pos")

#: The rule × case combinations the corpus pass computes, and the artifact suffix
#: each one's rows are written under. ``discriminant.fold`` is the primary: it is
#: the recorded fallback at the one case convention the data actually supports,
#: and the report shows every other combination beside it.
RULE_SPECS: tuple[tuple[str, str], ...] = (
    ("holds", "exact"),
    ("holds", "fold"),
    ("discriminant", "exact"),
    ("discriminant", "fold"),
    ("discriminant_pos", "fold"),
)
PRIMARY_SPEC = ("discriminant", "fold")


def rule_name(spec: tuple[str, str]) -> str:
    """The name a rule spec is reported and filed under: ``discriminant.fold``."""
    rule, case = spec
    return rule if case == "exact" else f"{rule}.{case}"


def fold_symbol(symbol: str, case: str) -> str:
    return symbol.upper() if case == "fold" else symbol


def with_case(definitions, case: str):
    """The definition set as it is matched under ``case`` (a copy; nothing frozen changes)."""
    if case == "exact":
        return definitions
    return definitions.model_copy(
        update={
            "definitions": tuple(
                definition.model_copy(
                    update={
                        "pos_markers": tuple(fold_symbol(m, case) for m in definition.pos_markers),
                        "neg_markers": tuple(fold_symbol(m, case) for m in definition.neg_markers),
                    }
                )
                for definition in definitions.definitions
            )
        }
    )


def presence(symbols, case: str) -> set[str]:
    """The state's symbols as the rule sees them."""
    return {fold_symbol(symbol, case) for symbol in symbols}


def _markers(definition, *, positive_only: bool = False) -> set[str]:
    if positive_only:
        return set(definition.pos_markers)
    return set(definition.pos_markers) | set(definition.neg_markers)


def _holding(options, presence: set[str], definitions, holds: HoldsRule) -> tuple[str, ...]:
    return tuple(option for option in options if holds.holds(presence, definitions.get(option)))


def separable(
    node: str, options: tuple[str, ...], presence: set[str], definitions, holds: HoldsRule, rule: str
) -> bool:
    """Can the evidence pick ``node`` out of ``options``? — see :data:`RULES`."""
    if rule == "holds":
        return _holding(options, presence, definitions, holds) == (node,)
    if rule == "discriminant":
        return _discriminates(node, options, presence, definitions, positive_only=False)
    if rule == "discriminant_pos":
        return _discriminates(node, options, presence, definitions, positive_only=True)
    raise ValueError(f"unknown rule {rule!r} (have: {list(RULES)})")


def _blocked(node: str, options: tuple[str, ...], presence: set[str], definitions, rule: str) -> tuple[str, ...]:
    """The siblings the evidence could not separate ``node`` from."""
    positive_only = rule == "discriminant_pos"
    own = _markers(definitions.get(node), positive_only=positive_only)
    return tuple(
        sibling
        for sibling in options
        if sibling != node and not ((own - _markers(definitions.get(sibling))) & presence)
    )


def _discriminates(
    node: str, options: tuple[str, ...], presence: set[str], definitions, *, positive_only: bool
) -> bool:
    """Every sibling is separated from ``node`` by a marker the state carries."""
    own = _markers(definitions.get(node), positive_only=positive_only)
    for sibling in options:
        if sibling == node:
            continue
        if not ((own - _markers(definitions.get(sibling))) & presence):
            return False
    return True


def _scoped_options(options: tuple[str, ...], node: str, scope: frozenset[str]) -> tuple[str, ...]:
    """A level's options as the record's *own* panel register can name them.

    The true child is always kept: the level is a real question about this label, so
    dropping it would make the question unanswerable rather than narrower. What the
    scope removes is the siblings the panel was never built to choose between.
    """
    return tuple(option for option in options if option in scope or option == node)


def depth_target(
    symbols: set[str] | list[str],
    chain: tuple[str, ...],
    register: RegisterTree,
    root: str,
    definitions,
    holds: HoldsRule,
    *,
    rule: str = "holds",
    scope: frozenset[str] | None = None,
) -> tuple[int, list[LevelResult]]:
    """The deepest level whose whole prefix the state can support (SPEC §4).

    Returns ``(target, levels)``. ``target`` is 0 when even the first level cannot
    be supported: the record's honest answer is then "not this one" at the top,
    the same abstention the flat ``panel_miss`` family teaches.

    ``scope`` is the register the record's own panel was drawn from (see
    :data:`TISSUE_MAP`); ``None`` is the whole register, i.e. the as-built question.
    """
    presence = set(symbols)
    levels: list[LevelResult] = []
    target = 0
    for index, node in enumerate(chain, start=1):
        parent = root if index == 1 else chain[index - 2]
        options = register.options(parent, root)
        if scope is not None:
            options = _scoped_options(options, node, scope)
        if node not in options:
            raise ValueError(f"level {index}: {node} is not among the options under {parent}")
        holding = _holding(options, presence, definitions, holds)
        chose = separable(node, options, presence, definitions, holds, rule)
        levels.append(
            LevelResult(
                level=index,
                node=node,
                options=options,
                holds=holding,
                separable=chose,
                singleton=len(options) == 1,
                rule=rule,
                blocked=() if chose or rule == "holds" else _blocked(node, options, presence, definitions, rule),
            )
        )
        if not chose:
            break
        target = index
    return target, levels


def register_terms(definitions) -> tuple[str, ...]:
    """The CL ids the register can name — the only types a level may offer."""
    return tuple(sorted(definition.cl_id for definition in definitions.definitions))


def load_definitions(path: str | Path):
    """The definitions artifact, loaded through the package's own model."""
    from ..contracts.definitions import DefinitionSet

    return DefinitionSet.load(str(path))


def load_terms(path: str | Path) -> dict[str, ClTerm]:
    """The CL snapshot (``{"terms": [...]}``) — v3, the version that carries parents."""
    return {term["cl_id"]: ClTerm.model_validate(term) for term in read_json(Path(path))["terms"]}


def load_tissue_register(path: str | Path) -> dict[str, frozenset[str]]:
    """The tissue registers the corpus's own panels were drawn from: ``tissue -> CL ids``.

    ``tissue_map.json`` is derived, not hand-appended — its own ``source`` states the
    rule (``tissues[T] = scope ∩ (cellmarker(T) ∪ observed(T))``) and the digests of
    the resources it was derived from. Every corpus record names its tissue in
    ``_meta.panel_scope``, so a level can be asked over the same list its panel was
    built from (see :func:`depth_target`'s ``scope``).
    """
    payload = read_json(Path(path))
    return {tissue: frozenset(cl_ids) for tissue, cl_ids in payload["tissues"].items()}


def atlas_label_counts(path: str | Path) -> dict[str, dict[str, int]]:
    """``dataset -> {cl_id: cells}`` from the curated atlas inventory.

    The seed is the inventory of record: it is what the datasets were read
    against, so the roots describe the same cells the corpus does.
    """
    payload = read_json(Path(path))
    return {
        dataset_id: {cl_id: int(n) for cl_id, n in sorted(entry["labels"].items())}
        for dataset_id, entry in sorted(payload.items())
    }


def write_tsv(path: Path, header: tuple[str, ...], rows: list[tuple]) -> dict:
    """A TSV artifact: bytes, digest, no timestamps — the repo's write rule."""
    body = "\t".join(header) + "\n" + "".join("\t".join(str(value) for value in row) + "\n" for row in rows)
    write_text(path, body)
    return {"sha256": file_digest(path), "bytes": len(body.encode(ENCODING)), "rows": len(rows), "kind": "table"}


def write_json(path: Path, payload: object) -> dict:
    body = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    write_text(path, body)
    return {"sha256": text_digest(body), "bytes": len(body.encode(ENCODING)), "kind": "report"}


def inventory_rows(
    datasets: dict[str, dict[str, int]], spine: Spine, names: dict[str, str], register: RegisterTree
) -> list[tuple]:
    """One row per (dataset, label): the inventory E1.1 and E1.4 read from."""
    rows = []
    for dataset_id in sorted(datasets):
        for cl_id, cells in sorted(datasets[dataset_id].items()):
            known = cl_id in spine.chains
            rows.append(
                (
                    dataset_id,
                    cl_id,
                    names.get(cl_id, ""),
                    cells,
                    int(cl_id in register.members),
                    int(known),
                    len(spine.chains.get(cl_id, ())),
                    spine.top.get(cl_id, ""),
                    int(known and spine.passes_cut(cl_id)),
                )
            )
    return rows


def corpus_records(corpus: Path):
    """Stream ``(split, record)`` over a corpus's four partitions, in file order."""
    for split in SPLITS:
        path = corpus / f"{split}.jsonl"
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                yield split, json.loads(line)


def corpus_targets(
    corpus: Path,
    out: Path,
    spine: Spine,
    register: RegisterTree,
    definitions,
    roots: list[RootChoice],
    *,
    filter_config: FilterConfig,
    holds: HoldsRule | None = None,
    rule_specs: tuple[tuple[str, str], ...] = RULE_SPECS,
    primary: tuple[str, str] = PRIMARY_SPEC,
    scoped: bool = False,
    scopes: dict[str, frozenset[str]] | None = None,
    dump: bool = True,
    config: dict | None = None,
) -> dict:
    """Write ``depth-targets.jsonl`` and ``depth-histogram.json`` (E1.5).

    One row per record, because the target is a property of the record's *own*
    state: the two masking families carry a different state from their siblings
    and therefore a different honest depth, and averaging them would hide exactly
    the panel effect the paper is about. The state → target mapping is pure, so
    identical states are computed once and reused (the clean/permuted twins of a
    record share one state).

    ``scoped`` asks every level over the register the record's own panel was drawn
    from (``scopes``, from :func:`load_tissue_register`, keyed by the record's
    ``_meta.panel_scope``); an unknown or absent tissue keeps the whole register.
    ``dump=False`` computes the same numbers without writing the row files, which
    is what a comparison pass over several configurations wants. ``config`` is
    recorded verbatim under the histogram's own ``config`` key, so the artifact
    says which configuration produced it.
    """
    holds = holds or HoldsRule()
    rule_specs = tuple(rule_specs)
    scopes = scopes or {}
    if primary not in rule_specs:
        raise ValueError(f"primary rule {primary!r} is not among {list(rule_specs)}")
    rules = tuple(rule_name(spec) for spec in rule_specs)
    case_of = {rule_name(spec): spec[1] for spec in rule_specs}
    definitions_of = {rule_name(spec): with_case(definitions, spec[1]) for spec in rule_specs}
    primary_name = rule_name(primary)
    root_of = {choice.dataset_id: choice.cl_id for choice in roots}
    targets = {
        rule: out / ("depth-targets.jsonl" if rule == primary_name else f"depth-targets.{rule}.jsonl")
        for rule in rules
    }
    counts = {rule: Counter() for rule in rules}
    per_family = {rule: defaultdict(Counter) for rule in rules}
    per_dataset = {rule: defaultdict(Counter) for rule in rules}
    per_dataset_chain = {rule: defaultdict(int) for rule in rules}
    per_dataset_root = {rule: defaultdict(int) for rule in rules}
    per_split = {rule: defaultdict(Counter) for rule in rules}
    on_chain = {rule: 0 for rule in rules}
    chain_root = {rule: 0 for rule in rules}
    variant_counts: dict[str, dict[str, Counter]] = {rule: defaultdict(Counter) for rule in rules}
    variant_chain = {rule: defaultdict(int) for rule in rules}
    variant_root = {rule: defaultdict(int) for rule in rules}
    off = Counter()
    ambiguous = Counter()
    singleton_levels = Counter()
    by_cell: dict[str, dict[tuple[str, str], tuple[str, int]]] = {rule: {} for rule in rules}
    cache: dict[
        tuple[str, str, str, frozenset[str] | None],
        tuple[tuple[str, ...], str, dict[str, int], dict[str, str], dict[str, int]],
    ] = {}
    rows = 0
    handles = (
        {rule: targets[rule].open("w", encoding=ENCODING, newline="\n") for rule in rules} if dump else {}
    )
    try:
        for split, record in corpus_records(corpus):
            meta = record["_meta"]
            dataset_id, cl_id, family = meta["dataset_id"], meta["cl_id"], meta["family"]
            root = root_of[dataset_id]
            tissue = meta.get("panel_scope") if scoped else None
            scope = scopes.get(tissue) if tissue else None
            key = (cl_id, root, record["state"], scope)
            if key not in cache:
                chain = register.chain_from(cl_id, root) if cl_id in register.members else ()
                if not chain:
                    kind = "off_register" if cl_id not in register.members else "off_root"
                    cache[key] = (chain, kind, dict.fromkeys(rules, 0), dict.fromkeys(rules, kind), {"amb": 0, "one": 0})
                else:
                    symbols = state_symbols(record["state"], filter_config)
                    levels_by_rule, chosen, stats = {}, {}, {"amb": 0, "one": 0}
                    for spec in rule_specs:
                        rule = rule_name(spec)
                        target, levels = depth_target(
                            presence(symbols, case_of[rule]),
                            chain,
                            register,
                            root,
                            definitions_of[rule],
                            holds,
                            rule=spec[0],
                            scope=scope,
                        )
                        levels_by_rule[rule] = target
                        chosen[rule] = "|".join(level.code() for level in levels)
                        stats["amb"] += sum(1 for level in levels if not level.separable and level.holds)
                        stats["one"] += sum(1 for level in levels if level.separable and level.singleton)
                    cache[key] = (chain, "on_chain", levels_by_rule, chosen, stats)
            chain, kind, targets_by_rule, codes, stats = cache[key]
            absent = kind != "on_chain"
            off[kind if absent else "on_chain"] += 1
            ambiguous["records"] += stats["amb"]
            singleton_levels["records"] += stats["one"]
            rows += 1
            cell_key = (dataset_id, meta["cell_id"])
            for rule in rules:
                target = targets_by_rule[rule]
                counts[rule][target] += 1
                per_family[rule][family][target] += 1
                per_dataset[rule][dataset_id][target] += 1
                per_dataset_chain[rule][dataset_id] += not absent
                per_dataset_root[rule][dataset_id] += (not absent) and target == 0
                per_split[rule][split][target] += 1
                on_chain[rule] += not absent
                chain_root[rule] += (not absent) and target == 0
                variant_counts[rule][meta["variant"]][target] += 1
                variant_chain[rule][meta["variant"]] += not absent
                variant_root[rule][meta["variant"]] += (not absent) and target == 0
                if dump:
                    handles[rule].write(
                        json.dumps(
                            {
                                "split": split,
                                "dataset": dataset_id,
                                "cell": meta["cell_id"],
                                "cl_id": cl_id,
                                "family": family,
                                "variant": meta["variant"],
                                "panel_scope": meta.get("panel_scope", ""),
                                "root": root,
                                "chain": list(chain),
                                "depth_target": target,
                                "levels": codes[rule],
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                previous = by_cell[rule].get(cell_key)
                if previous is None or target > previous[1]:
                    by_cell[rule][cell_key] = (cl_id, target)
    finally:
        for handle in handles.values():
            handle.close()
    cell_dist = {rule: Counter(depth for _, depth in by_cell[rule].values()) for rule in rules}

    if dump:
        write_text(
            out / "cell-depth-targets.tsv",
            "dataset\tcell\tcl_id\troot\tdepth_target\tchain\n"
            + "".join(
                "\t".join(
                    (
                        key[0],
                        key[1],
                        by_cell[primary_name][key][0],
                        root_of[key[0]],
                        str(by_cell[primary_name][key][1]),
                        ",".join(register.chain_from(by_cell[primary_name][key][0], root_of[key[0]])),
                    )
                )
                + "\n"
                for key in sorted(by_cell[primary_name])
            ),
        )
    cells_by_dataset: dict[str, Counter] = defaultdict(Counter)
    for (dataset_id, _), (_, depth) in by_cell[primary_name].items():
        cells_by_dataset[dataset_id][depth] += 1
    histograms = {}
    for rule in rules:
        counter = counts[rule]
        distribution = _distribution(counter)
        histograms[rule] = {
            "rule": rule,
            "records": {
                "n": rows,
                "on_chain": on_chain[rule],
                "by_depth": {str(depth): counter[depth] for depth in sorted(counter)},
                "median": _weighted_median(list(counter), list(counter.values())),
                "share_at_root": round(counter[0] / rows, 6),
                "median_on_chain": _median(
                    [depth for depth in counter for _ in range(counter[depth]) if depth or on_chain[rule]]
                ),
                "mean": round(sum(depth * n for depth, n in counter.items()) / rows, 4),
                "distribution": distribution,
            },
            "cells": {
                "n": len(by_cell[rule]),
                "by_depth": {str(depth): cell_dist[rule][depth] for depth in sorted(cell_dist[rule])},
                "median": _median([depth for depth in cell_dist[rule] for _ in range(cell_dist[rule][depth])]),
                "share_at_root": round(cell_dist[rule][0] / max(len(by_cell[rule]), 1), 6),
            },
            "by_family": {
                family: {str(depth): n for depth, n in sorted(entry.items())}
                for family, entry in sorted(per_family[rule].items())
            },
            "by_dataset": {
                dataset: {str(depth): n for depth, n in sorted(entry.items())}
                for dataset, entry in sorted(per_dataset[rule].items())
            },
            "dataset_gate": {
                dataset: gate(
                    per_dataset[rule][dataset],
                    sum(per_dataset[rule][dataset].values()),
                    per_dataset_chain[rule][dataset],
                    per_dataset_root[rule][dataset],
                )
                for dataset in sorted(per_dataset[rule])
            },
            "by_split": {
                split: {str(depth): n for depth, n in sorted(entry.items())}
                for split, entry in sorted(per_split[rule].items())
            },
            "by_variant": {
                variant: {str(depth): n for depth, n in sorted(entry.items())}
                for variant, entry in sorted(variant_counts[rule].items())
            },
            "gate_by_variant": {
                variant: gate(
                    entry,
                    sum(entry.values()),
                    variant_chain[rule][variant],
                    variant_root[rule][variant],
                )
                for variant, entry in sorted(variant_counts[rule].items())
            },
            "cells_by_dataset": {
                dataset: {str(depth): n for depth, n in sorted(entry.items())}
                for dataset, entry in sorted(cells_by_dataset.items())
            },
            #: The gate per dataset on the cell population: one reading per cell, so
            #: a dataset's verdict does not depend on how many masking variants its
            #: cells were rendered under.
            "dataset_cells_gate": {
                dataset: {
                    "n": sum(entry.values()),
                    "median": _weighted_median(list(entry), list(entry.values())),
                    "share_at_root": round(entry[0] / sum(entry.values()), 6) if entry else 1.0,
                    "passes": passes_gate(
                        _weighted_median(list(entry), list(entry.values())),
                        (entry[0] / sum(entry.values())) if entry else 1.0,
                    ),
                }
                for dataset, entry in sorted(cells_by_dataset.items())
            },
            "gate": gate_verdict(counter, rows, on_chain[rule], chain_root[rule], cell_dist[rule], len(by_cell[rule])),
        }
    histogram = {
        "primary_rule": primary_name,
        "rule_specs": {rule_name(spec): {"rule": spec[0], "case": spec[1]} for spec in rule_specs},
        #: The configuration that produced these numbers, as the caller described it
        #: (register, option scoping, roots). Recorded rather than assumed: the gate
        #: is read off this file, and two configurations differ by a lot.
        "config": config or {},
        "rules": histograms,
        "diagnostics": {
            "records_off_register": off["off_register"],
            "records_off_root": off["off_root"],
            "records_on_chain": off["on_chain"],
            "levels_stopped_with_a_contender": ambiguous["records"],
            "singleton_levels_passed": singleton_levels["records"],
            "unique_states": len(cache),
        },
        "sources": {
            "targets": {rule: file_digest(path) for rule, path in sorted(targets.items())} if dump else {},
            "root_of": root_of,
        },
    }
    if dump:
        write_json(out / "depth-histogram.json", histogram)
    return histogram


def _distribution(counter: Counter) -> dict:
    """The share of targets at each depth, over the population that carries one.

    ``off_register`` and ``off_root`` records have no chain at all, so they are not
    a depth of 0 — they are the records the descent never starts on. Mixing them
    into the histogram would make "the task has no resolution" and "the task was
    never asked" the same number.
    """
    total = sum(counter.values())
    return {str(depth): round(n / total, 6) for depth, n in sorted(counter.items())} if total else {}


def gate_verdict(counter: Counter, rows: int, on_chain: int, chain_root: int, cell_dist: Counter, cells: int) -> dict:
    """E1.5's non-degeneracy gate: median depth ≥ 3 and under 20% at the root.

    Reported on both populations: every record (the file as written, where an
    ``off_register`` abstention *is* a target of 0), and the records the descent
    actually starts on. A gate that passes on one and fails on the other is not
    passed, and the report says which is which.
    """
    out = gate(counter, rows, on_chain, chain_root)
    cell_median = _weighted_median(list(cell_dist), list(cell_dist.values()))
    cell_share = cell_dist[0] / cells if cells else 1.0
    out.update(
        {
            "median_depth_cells": cell_median,
            "share_at_root_cells": round(cell_share, 6),
            "passes_cells": passes_gate(cell_median, cell_share),
        }
    )
    return out


def gate(counter: Counter, rows: int, on_chain: int, chain_root: int) -> dict:
    """The record-level half of the gate, computable on any sub-population."""
    median = _weighted_median(list(counter), list(counter.values()))
    share = counter[0] / rows if rows else 1.0
    chain_median = _median([depth for depth in counter for _ in range(counter[depth]) if depth or on_chain])
    chain_share = chain_root / on_chain if on_chain else 1.0
    return {
        "median_depth_all_records": median,
        "share_at_root_all_records": round(share, 6),
        "median_depth_on_chain": chain_median,
        "share_at_root_on_chain": round(chain_share, 6),
        "passes_all_records": passes_gate(median, share),
        "passes_on_chain": passes_gate(chain_median, chain_share),
        "on_chain_records": on_chain,
        "chain_total": sum(n for depth, n in counter.items() if depth or on_chain),
    }


#: The gate's three populations, named: every record as written, the records the
#: descent starts on, and the cells the annotation task is actually asked about.
GATE_DEFINITIONS: tuple[str, ...] = ("all_records", "on_chain", "cells")


def passes_gate(median: float, share_at_root: float) -> bool:
    """E1.5's two clauses, on whichever population a caller is quoting.

    One function so the three readings of the gate cannot drift apart: the clauses
    are the plan's (median ≥ 3, under 20% at the root), and what differs between the
    readings is the population, never the threshold.
    """
    return bool(median >= 3 and share_at_root < 0.20)


def gate_definitions(entry: dict) -> dict:
    """The gate under its three definitions, read off one histogram entry.

    They differ by population and the difference is not a rounding error. 15,777 of
    the corpus's records are ``off_register`` and 19,091 are ``off_root``: they carry
    a target of 0 in the file as written, but they are not abstentions at the root —
    they are records the descent never starts on. ``cells`` is the deepest target a
    cell reached across its variants, i.e. the unit the corpus asks about.
    """
    gate = entry["gate"]
    return {
        "all_records": {
            "n": entry["records"]["n"],
            "median": gate["median_depth_all_records"],
            "share_at_root": gate["share_at_root_all_records"],
            "passes": gate["passes_all_records"],
        },
        "on_chain": {
            "n": gate["on_chain_records"],
            "median": gate["median_depth_on_chain"],
            "share_at_root": gate["share_at_root_on_chain"],
            "passes": gate["passes_on_chain"],
        },
        "cells": {
            "n": entry["cells"]["n"],
            "median": gate["median_depth_cells"],
            "share_at_root": gate["share_at_root_cells"],
            "passes": gate["passes_cells"],
        },
    }


def _median(values: list[int]) -> float:
    """The plain median of a list of depths (0.0 for an empty list)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return (ordered[middle - 1] + ordered[middle]) / 2


def _spine_digest(spine: Spine) -> str:
    body = {
        "prefer_cut": spine.prefer_cut,
        "parents": {cl_id: spine.parents[cl_id] for cl_id in sorted(spine.parents)},
    }
    return text_digest(json.dumps(body, sort_keys=True))


def comparison(
    corpus: Path,
    out: Path,
    seeds: dict[str, dict[str, int]],
    spine: Spine,
    names: dict[str, str],
    registers: dict[str, tuple[RegisterTree, object]],
    *,
    filter_config: FilterConfig,
    scopes: dict[str, frozenset[str]] | None = None,
    holds: HoldsRule | None = None,
    rule_specs: tuple[tuple[str, str], ...] = (PRIMARY_SPEC,),
    separate_overrides: bool = True,
) -> dict:
    """The gate under every configuration the three fixes move, one corpus pass each.

    The fixes are separable and a reader is owed the ability to see which one buys
    what, so each is turned on alone as well as together. Every entry is the same
    pass the canonical run makes, with ``dump=False`` — the numbers, not 30 MB of
    rows per configuration. Keys, and what they mean:

    * ``<register> / global / plan`` — the whole-register option list and the plan's
      roots. The as-built configuration is one of these, and it is the one that has
      to reproduce the published numbers.
    * ``<register> / scoped / plan`` — level options narrowed to each record's own
      tissue register (fix 1).
    * ``<register> / scoped / overrides`` — fixes 1–2 plus :data:`ROOT_OVERRIDES`.
    * ``<register> / scoped / <dataset> root only`` — one override at a time, so the
      root change is attributable on its own rather than as a bundle.

    The register label is the caller's (``V0`` as built, ``Vex`` extended); every
    configuration is reported under the gate's three definitions, and per dataset.
    Writes ``depth-comparison.json`` and returns it.
    """
    holds = holds or HoldsRule()
    scopes = scopes or {}
    configurations: dict[str, dict] = {}
    for label, (register, definitions) in registers.items():
        plan_roots = dataset_roots(seeds, spine, register, names)
        variants: list[tuple[str, list[RootChoice], bool]] = [
            (f"{label} / global / plan", plan_roots, False),
        ]
        if scopes:
            variants += [
                (f"{label} / scoped / plan", plan_roots, True),
                (
                    f"{label} / scoped / overrides",
                    apply_root_overrides(plan_roots, seeds, spine, register, names),
                    True,
                ),
            ]
            if separate_overrides:
                for dataset_id, node in sorted(ROOT_OVERRIDES.items()):
                    variants.append(
                        (
                            f"{label} / scoped / {dataset_id} root only",
                            apply_root_overrides(
                                plan_roots, seeds, spine, register, names, overrides={dataset_id: node}
                            ),
                            True,
                        )
                    )
        for name, roots, scoped in variants:
            entry = corpus_targets(
                corpus,
                out,
                spine,
                register,
                definitions,
                roots,
                filter_config=filter_config,
                holds=holds,
                rule_specs=rule_specs,
                scoped=scoped,
                scopes=scopes,
                dump=False,
                config={
                    "register": label,
                    "register_terms": len(register.members),
                    "scoped": scoped,
                    "root_of": {row.dataset_id: row.cl_id for row in roots},
                    "root_overrides": {
                        row.dataset_id: row.override for row in roots if row.override
                    },
                },
            )
            rule_entry = entry["rules"][entry["primary_rule"]]
            configurations[name] = {
                "config": entry["config"],
                "diagnostics": entry["diagnostics"],
                "records": rule_entry["records"],
                "cells": rule_entry["cells"],
                "gate": gate_definitions(rule_entry),
                "gate_by_variant": rule_entry["gate_by_variant"],
                "dataset_gate": rule_entry["dataset_gate"],
                "dataset_cells_gate": rule_entry["dataset_cells_gate"],
            }
            gate = configurations[name]["gate"]
            print(
                f"  {name:44s} all: med {gate['all_records']['median']:.1f} @root "
                f"{gate['all_records']['share_at_root']:.4f} {'PASS' if gate['all_records']['passes'] else 'fail'} | "
                f"chain: med {gate['on_chain']['median']:.1f} @root {gate['on_chain']['share_at_root']:.4f} "
                f"{'PASS' if gate['on_chain']['passes'] else 'fail'} | "
                f"cells: med {gate['cells']['median']:.1f} @root {gate['cells']['share_at_root']:.4f} "
                f"{'PASS' if gate['cells']['passes'] else 'fail'}",
                flush=True,
            )
    payload = {
        "corpus": str(corpus),
        "rules": [rule_name(spec) for spec in rule_specs],
        "definitions": GATE_DEFINITIONS,
        "configurations": configurations,
    }
    write_json(out / "depth-comparison.json", payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sckevtype.data.depth", description="Build the ontology depth layer.")
    parser.add_argument("--artifacts", default="artifacts/v3", help="the CL snapshot version to read")
    parser.add_argument("--seed", default="configs/seed/v3/atlas_labels.json", help="the atlas label inventory")
    parser.add_argument("--out", default="artifacts/cl", help="where the depth layer's artifacts are written")
    parser.add_argument("--corpus", default=None, help="a corpus directory (runs/<name>) to compute depth targets for")
    parser.add_argument("--no-prefer-cut", action="store_true", help="reproduce SPEC §3's literal branch rule")
    parser.add_argument(
        "--definitions",
        default=None,
        help="the register's definition set (default: <artifacts>/definitions.json). "
        "The extended register built by scripts/depth-register-extension.py is the fix; "
        "the default is the as-built one, so this command alone reproduces the old numbers.",
    )
    parser.add_argument(
        "--tissues",
        default=None,
        help="the tissue registers (default: <artifacts>/tissue_map.json), used by --scoped",
    )
    parser.add_argument(
        "--scoped",
        action="store_true",
        help="ask each level over the record's own tissue register instead of the whole register",
    )
    parser.add_argument(
        "--root-overrides",
        action="store_true",
        help="apply the measured per-dataset root exceptions (depth.ROOT_OVERRIDES)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="also write depth-comparison.json: the gate under each configuration, one corpus pass each",
    )
    parser.add_argument(
        "--compare-rules",
        default=rule_name(PRIMARY_SPEC),
        help="the rule the comparison pass reports (default: the primary rule)",
    )
    args = parser.parse_args(argv)

    artifacts = Path(args.artifacts)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    terms = load_terms(artifacts / "cl_terms.json")
    names = {cl_id: term.name for cl_id, term in terms.items()}
    spine = build_spine(terms, prefer_cut=not args.no_prefer_cut)
    definitions_path = Path(args.definitions) if args.definitions else artifacts / "definitions.json"
    definitions = load_definitions(definitions_path)
    register = RegisterTree(spine, register_terms(definitions))
    datasets = atlas_label_counts(args.seed)
    tissues_path = Path(args.tissues) if args.tissues else artifacts / TISSUE_MAP
    scopes = load_tissue_register(tissues_path) if args.scoped else {}
    if args.scoped and not scopes:
        parser.error(f"--scoped needs the tissue registers, and {tissues_path} has none")
    filter_config = FilterConfig.model_validate(read_json(artifacts / "render_config.json"))

    payload = spine.payload(names)
    files: dict[str, dict] = {}
    files["spines.json"] = write_json(out / "spines.json", {**payload, "digest": _spine_digest(spine)})
    print(
        f"spines.json: {len(spine.chains)} terms, {len(spine.parentless)} parentless, "
        f"{payload['n_passes_cut']} chains reach {CUT}, {payload['n_ties']} ties"
    )
    files["labels.tsv"] = write_tsv(
        out / "labels.tsv",
        ("dataset", "cl_id", "name", "cells", "in_register", "in_cl", "chain_length", "top", "passes_cut"),
        inventory_rows(datasets, spine, names, register),
    )
    roots = dataset_roots(datasets, spine, register, names)
    if args.root_overrides:
        roots = apply_root_overrides(roots, datasets, spine, register, names)
    files["roots.tsv"] = write_tsv(
        out / "roots.tsv",
        ("dataset", "root", "root_name", "depth_from_top", "coverage", "labels_covered", "labels_total",
         "register_labels_covered", "cells_covered", "cells_total", "label_coverage", "median_depth",
         "median_spine_depth", "level_one_options", "label_weighted_root", "label_weighted_coverage",
         "override"),
        [
            (row.dataset_id, row.cl_id, row.name, row.depth_from_top, row.coverage, row.labels_covered,
             row.labels_total, row.register_labels_covered, row.cells_covered, row.cells_total,
             row.label_coverage, row.median_depth, row.median_spine_depth, row.level_one,
             row.label_weighted_root or "", row.label_weighted_coverage, row.override)
            for row in roots
        ],
    )
    for row in roots:
        print(
            f"  {row.dataset_id:20s} root={row.cl_id} {row.name[:26]:26s} cov={row.coverage:.3f} "
            f"median={row.median_depth} level1={row.level_one}"
            + (f"  [{row.override}]" if row.override else "")
        )
    if args.corpus:
        histogram = corpus_targets(
            Path(args.corpus),
            out,
            spine,
            register,
            definitions,
            roots,
            filter_config=filter_config,
            scoped=args.scoped,
            scopes=scopes,
            config={
                "definitions": str(definitions_path),
                "register_terms": len(register.members),
                "tissues": str(tissues_path) if scopes else "",
                "tissue_terms": {tissue: len(cl_ids) for tissue, cl_ids in sorted(scopes.items())},
                "scoped": bool(args.scoped),
                "root_overrides": {row.dataset_id: row.override for row in roots if row.override},
            },
        )
        for rule, digests in sorted(histogram["sources"]["targets"].items()):
            name = "depth-targets.jsonl" if rule == histogram["primary_rule"] else f"depth-targets.{rule}.jsonl"
            files[name] = {"sha256": digests, "kind": "data", "rows": histogram["rules"][rule]["records"]["n"]}
        files["depth-histogram.json"] = {"kind": "report", "rules": list(histogram["rules"])}
        files["cell-depth-targets.tsv"] = {"kind": "table", "rows": histogram["rules"][histogram["primary_rule"]]["cells"]["n"]}
        for rule, entry in sorted(histogram["rules"].items()):
            for definition, reading in gate_definitions(entry).items():
                print(
                    f"  {rule:16s} {definition:11s} n {reading['n']:6d}: median {reading['median']:.1f} "
                    f"root share {reading['share_at_root']:.4f} | passes "
                    f"{'YES' if reading['passes'] else 'no'}"
                )
        if args.compare:
            if args.compare_rules not in {rule_name(spec) for spec in RULE_SPECS}:
                parser.error(f"--compare-rules {args.compare_rules!r} is not one of the computed rules")
            chosen = next(spec for spec in RULE_SPECS if rule_name(spec) == args.compare_rules)
            as_built = load_definitions(artifacts / "definitions.json")
            registers: dict[str, tuple[RegisterTree, object]] = {
                "as-built": (RegisterTree(spine, register_terms(as_built)), as_built),
                "extended": (register, definitions),
            }
            print(f"\ndepth-comparison.json (rule {args.compare_rules}, one corpus pass per configuration)")
            comparison(
                Path(args.corpus),
                out,
                datasets,
                spine,
                names,
                registers,
                filter_config=filter_config,
                scopes=scopes or load_tissue_register(tissues_path),
                rule_specs=(chosen,),
            )
            files["depth-comparison.json"] = {"kind": "report", "configurations": "see the file"}
    write_json(out / "depth-layer.json", {"spine": _spine_digest(spine), "files": files})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
