"""Split registry: dataset tiers, donor-level grouping, sampling caps.

The split names are kev's (``train``/``calibration``/``development``/``test``), so
a corpus is loadable by ``kev.suite.load_split`` without a translation step. What
goes where is decided in two steps, in this order:

1. **the dataset's tier** (``configs/corpus/<name>.yaml``, defaulting to the registry's
   entry) says which splits it may feed — ``train`` tier donors are split three ways
   (train / calibration / development), ``all`` donors are split four ways (the same
   three plus test: the tier that lets a dataset be trained on *and* tested on, on
   donors of its own), ``T2`` donors all go to development (the cross-tissue shift)
   and ``T3`` donors all go to test (the held-out shifts, never seen during model
   selection);
2. **the donor** decides which of the tier's splits a cell lands in, by an
   independent per-donor draw. Grouping by donor is not a nicety: one donor's
   cells are near-duplicates of each other, so a donor that straddles a split
   leaks the test set into training. The donor table is one dict, so a donor
   cannot be in two splits — the invariant holds by construction, and the tests
   re-check it from the written files.

Caps are sampling policy, applied *before* the donor draw so a capped-away donor
never appears half-kept: per dataset (from the corpus config, defaulting to the
registry's ``default_cap``) and per (dataset, type). Both cap by seeded sampling
over sorted cell ids, so a cap changes the corpus only in the way the seed says.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict, Field

from . import seeded_rng

#: kev's split names, in the order the paper uses them. Restated here because kev
#: (the training stack) is an extra and this module must import without it; a test
#: asserts the two lists still agree.
SPLIT_NAMES: tuple[str, ...] = ("train", "calibration", "development", "test")
#: Split a dataset's cells may feed, by tier. A tier's donor draw walks exactly the
#: tuples with more than one split, in ``SPLIT_NAMES`` order, and ``train`` (the
#: remainder) is what is left when every fraction has been claimed.
TIER_SPLITS: dict[str, tuple[str, ...]] = {
    "train": ("train", "calibration", "development"),
    "T2": ("development",),
    "T3": ("test",),
    #: Every split, drawn per donor: the tier a corpus uses when it wants each dataset
    #: to both train the model and test it on its own held-out donors.
    "all": ("train", "calibration", "development", "test"),
}

#: The splits a donor draw can land in, and the ``SplitRegistry`` field that says how
#: many donors each claims, in the order the draw walks them. ``train`` is absent on
#: purpose: it is the remainder, not a bucket, which is why the walk below adds only
#: the buckets the tier actually reaches — an earlier tier's draw is untouched by a
#: later bucket's existence.
_DRAWN_SPLITS: tuple[tuple[str, str], ...] = (
    ("calibration", "calibration_frac"),
    ("development", "development_frac"),
    ("test", "test_frac"),
)


def reads_test_frac(tier: str | None) -> bool:
    """Whether a tier's donor draw reads ``SplitRegistry.test_frac``.

    True only where the draw has a test bucket to reach *and* something else to draw
    against: ``all`` does both, and ``T3`` reaches test without drawing at all (a
    single split is where every donor of that tier goes). It is the question a corpus
    config's fraction check is really asking, and the reason a dataset on ``train`` /
    ``T2`` / ``T3`` cannot be moved by ``test_frac`` however it is set.
    """
    splits = TIER_SPLITS.get(tier or "", ())
    return len(splits) > 1 and "test" in splits


class SplitCaps(BaseModel):
    """How much of each dataset the corpus keeps (sampling policy, not a fact)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: dataset id -> cap; a dataset not listed keeps every cell.
    per_dataset: dict[str, int] = Field(default_factory=dict)
    #: Cap per (dataset, type): keeps a corpus from being one big cell type.
    per_type: int | None = Field(default=None, ge=1)
    #: dataset id -> cap on the cells the register cannot *name*. A separate
    #: budget from ``per_dataset``, drawn separately: the two are different
    #: populations, and a shared budget would let an atlas whose labels are mostly
    #: out of register crowd out the cells the register does cover. A dataset not
    #: listed falls back to its own ``per_dataset`` cap.
    per_unregistered: dict[str, int] = Field(default_factory=dict)


@dataclass
class SplitRegistry:
    """The assignment of cells to splits, built by registering cells then freezing."""

    seed: int
    tiers: Mapping[str, str]
    caps: SplitCaps = field(default_factory=SplitCaps)
    #: Fractions of a tier's donors given to calibration / development / test, in that
    #: draw order. ``test`` is claimed last, so the first two buckets of every tier are
    #: where they always were, and ``train`` keeps the remainder.
    calibration_frac: float = 0.1
    development_frac: float = 0.1
    #: The share of a dataset's donors held back as test *within its own tier*: it is
    #: read only by a tier whose split list reaches ``test`` (``all``), and every dataset
    #: on that tier then trains on roughly ``1 - calibration - development - test`` of
    #: its donors and is tested on the rest. A tier that cannot reach test (``train``,
    #: ``T2``, ``T3``) never reads it, so adding this knob changed no earlier corpus's
    #: assignment. 0.2 is the paper's per-dataset holdout: with the two fractions above,
    #: about 0.6 / 0.1 / 0.1 / 0.2 train / calibration / development / test.
    test_frac: float = 0.2
    #: The CL ids the register can name. ``None`` means every id is in scope, which
    #: is what a fixture or a pre-register corpus wants. Cells outside it are the
    #: out-of-register stratum: capped separately, but *not* split separately — the
    #: donor table is still one dict, so a donor cannot straddle a split by having
    #: cells in both strata.
    registered: frozenset[str] | None = None
    cells: dict[str, list[tuple[str, str | None, str]]] = field(default_factory=dict)
    donors: dict[str, str] = field(default_factory=dict)
    assignments: dict[str, str] = field(default_factory=dict)
    #: split -> dataset -> type -> n, and split -> dataset -> donors (filled by ``build``).
    _counts: dict[str, dict[str, Counter]] = field(default_factory=dict)
    _donors: dict[str, Counter] = field(default_factory=dict)
    _built: bool = False

    def register(self, dataset_id: str, cells: Iterable[tuple[str, str | None, str]]) -> None:
        """Add one dataset's ``(cell_id, donor_id, cl_id)`` triples (expression not needed)."""
        if dataset_id not in self.tiers:
            raise KeyError(f"dataset {dataset_id!r} has no tier (have: {sorted(self.tiers)})")
        bucket = self.cells.setdefault(dataset_id, [])
        for cell_id, donor_id, cl_id in cells:
            bucket.append((str(cell_id), None if donor_id is None else str(donor_id), str(cl_id)))

    def build(self) -> SplitRegistry:
        """Apply caps, draw donor splits, and freeze the cell -> split assignment."""
        for dataset_id in sorted(self.cells):
            kept = _apply_caps(self.cells[dataset_id], self.caps, self.seed, dataset_id, self.registered)
            splits = TIER_SPLITS[self.tiers[dataset_id]]
            for cell_id, donor_id, cl_id in kept:
                donor_key = f"{dataset_id}/{donor_id or 'unspecified'}"
                split = self.donors.get(donor_key)
                if split is None:
                    split = self.donors[donor_key] = _donor_split(donor_key, splits, self)
                    self._donors.setdefault(split, Counter())[dataset_id] += 1
                self.assignments[f"{dataset_id}/{cell_id}"] = split
                self._counts.setdefault(split, {}).setdefault(dataset_id, Counter())[cl_id] += 1
        self._built = True
        return self

    def split_of(self, dataset_id: str, cell_id: str) -> str:
        if not self._built:
            raise RuntimeError("SplitRegistry.build() has not run")
        key = f"{dataset_id}/{cell_id}"
        if key not in self.assignments:
            raise KeyError(f"{key} was not registered (or was capped away)")
        return self.assignments[key]

    def kept(self, dataset_id: str) -> set[str]:
        return {key.split("/", 1)[1] for key in self.assignments if key.startswith(f"{dataset_id}/")}

    def counts(self) -> dict[str, dict[str, dict[str, int]]]:
        """Realised counts: split -> dataset -> type -> n (the report the spec asks for)."""
        return {
            split: {dataset: {cl_id: n for cl_id, n in sorted(types.items())} for dataset, types in sorted(datasets.items())}
            for split, datasets in sorted(self._counts.items())
        }

    def donor_counts(self) -> dict[str, dict[str, int]]:
        return {split: {dataset: n for dataset, n in sorted(datasets.items())} for split, datasets in sorted(self._donors.items())}

    def payload(self) -> dict:
        """Deterministic form for the corpus manifest (no paths, no timestamps)."""
        counts, donors = self.counts(), self.donor_counts()
        return {
            "seed": self.seed,
            "caps": self.caps.model_dump(mode="json"),
            "calibration_frac": self.calibration_frac,
            "development_frac": self.development_frac,
            "test_frac": self.test_frac,
            "splits": {
                split: {
                    "cells": sum(sum(types.values()) for types in counts.get(split, {}).values()),
                    "donors": sum(donors.get(split, {}).values()),
                    "datasets": counts.get(split, {}),
                }
                for split in SPLIT_NAMES
            },
            "donors": {key: self.donors[key] for key in sorted(self.donors)},
        }


def _donor_split(donor_key: str, splits: tuple[str, ...], registry: SplitRegistry) -> str:
    """Which of the tier's splits this donor goes to (independent per-donor draw).

    One uniform number per donor, and the tier's own splits as cumulative buckets
    walked in ``_DRAWN_SPLITS`` order: calibration, then development, then test, and
    ``train`` takes whatever is left. A tier that does not reach one of those buckets
    does not advance the cumulative total past it, which is what keeps ``train``'s
    three-way draw — and so every v1–v3 assignment — identical to what it was before
    ``test_frac`` existed. A tier with a single split (``T2``, ``T3``) has nothing to
    draw: every donor goes there, and the draw is not even made.
    """
    if len(splits) == 1:
        return splits[0]
    draw = seeded_rng(registry.seed, "split", donor_key).random()
    cumulative = 0.0
    for name, share in _DRAWN_SPLITS:
        if name not in splits:
            continue
        cumulative += getattr(registry, share)
        if draw < cumulative:
            return name
    return "train"


def _apply_caps(
    cells: list[tuple[str, str | None, str]],
    caps: SplitCaps,
    seed: int,
    dataset_id: str,
    registered: frozenset[str] | None = None,
) -> list[tuple[str, str | None, str]]:
    """Cap by (dataset, type) then by dataset, always by seeded sampling over sorted ids.

    The dataset cap is applied to the registered and the out-of-register cells
    under separate draws (``per_unregistered``, defaulting to the dataset's own
    cap), so neither can crowd the other out.
    """
    kept = list(cells)
    if caps.per_type is not None:
        by_type: dict[str, list[tuple[str, str | None, str]]] = {}
        for cell in kept:
            by_type.setdefault(cell[2], []).append(cell)
        kept = []
        for cl_id in sorted(by_type):
            kept.extend(_cap(list(by_type[cl_id]), caps.per_type, seed, dataset_id, cl_id))
    cap = caps.per_dataset.get(dataset_id)
    if registered is None or cap is None:
        return _cap(kept, cap, seed, dataset_id) if cap is not None else kept
    in_scope = [cell for cell in kept if cell[2] in registered]
    outside = [cell for cell in kept if cell[2] not in registered]
    budget = caps.per_unregistered.get(dataset_id, cap)
    return _cap(in_scope, cap, seed, dataset_id) + _cap(outside, budget, seed, dataset_id, "unregistered")


def _cap(cells: list[tuple[str, str | None, str]], cap: int, seed: int, *parts: str) -> list[tuple[str, str | None, str]]:
    if len(cells) <= cap:
        return cells
    order = sorted(cells, key=lambda cell: cell[0])
    chosen = set(seeded_rng(seed, "cap", *parts).sample([cell[0] for cell in order], cap))
    return [cell for cell in order if cell[0] in chosen]
