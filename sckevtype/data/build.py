"""The corpus builder: atlases in, kev-loadable suite out.

    python -m sckevtype.data.build --config configs/corpus/<name>.yaml --out artifacts/corpus/<name>/

What it writes:

* ``train.jsonl`` / ``calibration.jsonl`` / ``development.jsonl`` / ``test.jsonl``
  — kev labelled-request JSONL, one record per line, exactly the bytes
  ``kev.data.load_records`` reads;
* ``manifest.json`` — sha256, size and record count per file, the seed, the config
  digest, the digest of every input dataset, the split table and the per-family
  counts. ``kev.suite.load_split`` reads the per-file entries and verifies them,
  which is why they are written in its shape;
* ``qc.json`` — the probes in :mod:`sckevtype.data.qc`.

Two properties are deliberate and load-bearing:

**Byte-identical rebuilds.** Same ``(seed, config, dataset digests)`` ⇒ same
bytes, including into a *different* output directory. Nothing written here
contains a timestamp, an absolute path or an iteration order that depends on a
set or a dict from the filesystem; records are sorted before they are written.

**Refusal to overwrite.** An existing output file with different bytes is an
error unless ``--force``: a corpus that was trained on must not silently become a
different corpus under the same name.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Collection
from functools import partial
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from ..contracts import budget, paths
from ..contracts.budget import Measure
from ..contracts.definitions import DefinitionSet
from ..contracts.freeze import (
    ENCODING,
    file_digest,
    payload_digest,
    read_json,
    text_digest,
    write_text,
)
from ..contracts.ontology import ClTerm, LabelMap, TissueMap
from ..contracts.render import FilterConfig
from ..contracts.schema import Record, RecordFamily
from . import fetch as fetch_module
from . import qc, registry, seed_int, seeded_rng
from .adapters import CellRecord, LabelStats, adapter_for
from .adapters import probe as probe_dataset
from .augment import permute_options
from .families import (
    UNREGISTERED_TISSUE,
    UNREGISTERED_TYPE,
    AxisSpec,
    FamilyContext,
    HoldsRule,
    axis_score,
    choose_none,
    evidence_diluted,
    evidence_removed,
    marker_noul,
    out_of_register,
    panel_choice,
    panel_miss_pair,
)
from .panels import GLOBAL_TISSUE, PanelConfig, sample_miss_panel, sample_panel_pair
from .registry import ObsColumns, PinnedFile
from .splits import SPLIT_NAMES, SplitCaps, SplitRegistry, reads_test_frac

#: The order families are built and written in — fixed, so the files are stable.
FAMILY_ORDER: tuple[RecordFamily, ...] = tuple(RecordFamily)


class TokenizerPin(BaseModel):
    """The tokenizer a corpus counts its states with: an id and the commit it was loaded at.

    Restated from ``configs/models.yaml`` rather than referenced, like every other
    registry value a corpus config carries — the digest of this file is what pins
    the corpus, so a value that lives elsewhere is a value the corpus does not
    pin. ``tests/data/test_real_config.py`` checks the two do not drift.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    revision: str = Field(min_length=1)


class StateBudget(BaseModel):
    """What the model can read: kev's ``MAX_STATE`` and the tokenizer's cost.

    Both numbers are configuration rather than constants here, because the whole
    point of reporting the length distribution is to compare the corpus against a
    budget someone else can check.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_tokens: int = Field(ge=1)
    tokens_per_symbol: float = Field(default=3.0, gt=0.0)
    qc_permutations: int = Field(default=20, ge=0)
    qc_max_features: int | None = Field(default=None, ge=1)
    #: The tokenizer the ``tokens`` block of the report is measured with. Without one the
    #: report carries only the symbol estimate, which is a guess and cannot decide whether
    #: a corpus can be trained — so a corpus meant to be trained should name one.
    tokenizer: TokenizerPin | None = None


class DatasetConfig(BaseModel):
    """One dataset of a corpus: where it is, how to read it, and how much to keep."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    path: str | None = None
    fetch: bool = False
    #: Where this corpus actually fetches the atlas, when the registry's verified path
    #: is a landing page (``configs/datasets.yaml`` records what was verified, not a
    #: download plan). Non-empty *replaces* the registry entry's access paths; every
    #: URL here must be probe-verified (a ranged GET returning 206) before it is written.
    access: tuple[PinnedFile, ...] = ()
    #: Registry id whose entry supplies the defaults for the fields left unset
    #: (format, tier, tissue, licence, default cap). ``None`` means every field
    #: below must be given — the honest choice for a synthetic fixture, which must
    #: never be recorded as the dataset it merely looks like.
    registered: bool = False
    format: Literal["cellxgene_h5ad", "tenx_mtx", "csv_expr", "baron_archive"] | None = None
    #: Which splits this dataset may feed (``sckevtype.data.splits.TIER_SPLITS``). The
    #: registry's three tiers are ``train`` / ``T2`` / ``T3``; ``all`` is this config's
    #: own and has no registry entry, because it is a *corpus* design (every dataset
    #: trains and every dataset keeps donors back to be tested on) rather than a fact
    #: about an atlas.
    tier: Literal["train", "T2", "T3", "all"] | None = None
    tissue: str | None = None
    tissue_labels: dict[str, str] = Field(default_factory=dict)
    #: Artifact name (``label_map.<name>``) or an inline ``label -> CL id`` table.
    label_map: str | dict[str, str] | None = None
    #: Adapter options that name a file inside the dataset rather than a column.
    label_file: str | None = None
    label_column: str | None = None
    #: ``obs`` columns for a dataset the registry does not describe (a fixture, or a
    #: dataset read from a layout the registry entry does not match).
    obs: ObsColumns | None = None
    cell_column: str | None = None
    genes_column: str | None = None
    donor_id: str | None = None
    donor_from_stem: bool = False
    #: Cells this dataset may contribute (overrides the registry's ``default_cap``).
    cap: int | None = Field(default=None, ge=1)
    #: This dataset's own differential-expression hits — used *only* by the leakage
    #: guard, never as a target (design-notes §6).
    de_markers: tuple[str, ...] = ()
    provenance: str = ""


class SplitConfig(BaseModel):
    """How much of a tier's donors go to calibration / development / test, and the caps."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: Fractions of a tier's donors, in the order ``splits._DRAWN_SPLITS`` walks them.
    calibration_frac: float = Field(default=0.1, ge=0.0, le=1.0)
    development_frac: float = Field(default=0.1, ge=0.0, le=1.0)
    #: The share of each dataset's donors held back as *its own* test set, drawn by the
    #: same per-donor, seeded, donor-level rule the other two use (never a cell-level
    #: split: one donor's cells are near-duplicates, so a donor that straddled the
    #: boundary would leak the test set into training). It bites only on a dataset whose
    #: tier reaches ``test`` with more than one split — ``all`` — which is the tier a
    #: v4 corpus puts every atlas on so that "does the model know cell types" and "does
    #: it carry to another lab's protocol" stop being the same question. A dataset on
    #: ``train`` / ``T2`` / ``T3`` never reads it, so this field defaults to 0.2 without
    #: moving a donor in any corpus built before it existed.
    test_frac: float = Field(default=0.2, ge=0.0, le=1.0)
    per_dataset: dict[str, int] = Field(default_factory=dict)
    per_type: int | None = Field(default=None, ge=1)
    #: dataset id -> cap on the cells the register cannot name (the separate budget
    #: `SplitCaps.per_unregistered` describes). Absent: the dataset's own cap.
    unregistered: dict[str, int] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> SplitConfig:
        if self.calibration_frac + self.development_frac >= 1.0:
            raise ValueError("calibration_frac + development_frac must leave something for train")
        return self


class CorpusConfig(BaseModel):
    """The corpus definition: frozen, digested, and the only source of the numbers."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    seed: int
    version: str = "v1"
    #: Where the frozen contract artifacts live (default: ``artifacts/<version>``).
    artifacts: str | None = None
    #: Where fetched datasets are cached (only used by datasets with ``fetch: true``).
    cache: str | None = None
    datasets: tuple[DatasetConfig, ...] = Field(min_length=1)
    panel: PanelConfig = Field(default_factory=PanelConfig)
    holds: HoldsRule = Field(default_factory=HoldsRule)
    leakage_alpha: float = Field(default=0.0, ge=0.0, le=1.0)
    dilution_k: int = Field(default=3, ge=1)
    splits: SplitConfig = Field(default_factory=SplitConfig)
    state: StateBudget
    axis: AxisSpec | None = None
    families: tuple[RecordFamily, ...] = FAMILY_ORDER
    #: How many renderings of each record a split carries (see `variant_records`).
    #: ``twins`` is the policy every corpus up to and including ``real-v2`` was
    #: built with. ``single`` emits exactly one: the permuted rendering in train
    #: (a model trained on one fixed option order learns the order), the clean one
    #: in every other split — so the scored population is one question per
    #: ``(dataset, cell, family)`` triple instead of one per rendering of it.
    variant_policy: Literal["twins", "single"] = "twins"

    @model_validator(mode="after")
    def _check(self) -> CorpusConfig:
        ids = [dataset.id for dataset in self.datasets]
        duplicates = sorted({dataset_id for dataset_id in ids if ids.count(dataset_id) > 1})
        if duplicates:
            raise ValueError(f"duplicate dataset ids in the corpus: {duplicates}")
        if len(set(self.families)) != len(self.families):
            raise ValueError("duplicate families in the corpus config")
        if self.panel.tissues or self.panel.register_terms:
            raise ValueError("panel.tissues and panel.register_terms come from the tissue-map artifact, not the corpus config")
        # `test_frac` is in the sum only where it is read: a corpus on the splits that
        # cannot reach test draws from the two numbers it always did, so a config that
        # was valid before the field existed still is. Where it *is* read (`all`), the
        # three fractions must leave train something, or the draw has no remainder.
        drawn = sorted(dataset.id for dataset in self.datasets if reads_test_frac(dataset.tier))
        if drawn:
            total = self.splits.calibration_frac + self.splits.development_frac + self.splits.test_frac
            if total >= 1.0:
                raise ValueError(
                    f"calibration_frac + development_frac + test_frac = {total} must leave something for train "
                    f"(datasets on a tier whose draw reaches test: {drawn})"
                )
        return self


class Artifacts:
    """The frozen Step-1 resources the factory reads (never re-implements)."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.filter = FilterConfig.model_validate(read_json(directory / "render_config.json"))
        self.definitions = DefinitionSet.load(str(directory / "definitions.json"))
        self.tissue_map = TissueMap.load(str(directory / "tissue_map.json"), self.definitions)
        payload = read_json(directory / "cl_terms.json")
        self.cl_terms = {term["cl_id"]: ClTerm.model_validate(term) for term in payload["terms"]}
        self.digests = {
            name: file_digest(directory / name)
            for name in sorted(("render_config.json", "definitions.json", "tissue_map.json", "cl_terms.json"))
        }

    def label_map(self, dataset: DatasetConfig, *, unregistered: bool = False) -> LabelMap:
        """The dataset's ``label -> CL id`` table, bound to the dataset it is applied to.

        An artifact map is a vocabulary table, so the same artifact can serve two
        datasets; binding it to the corpus dataset id is what makes the drops it
        reports attributable to the right dataset.

        ``unregistered=True`` turns on the map's local-type policy, so a label the
        register cannot name survives as an explicit ``local:<dataset>:<label>`` id
        instead of being dropped. The *cells* are the same ones either way — a
        local id is by construction not a register term — which is what lets the
        out-of-register family be built from them (Step 8 §2) without a second
        pass over 87 GB of atlas.
        """
        table = None
        if isinstance(dataset.label_map, dict):
            table = LabelMap(dataset_id=dataset.id, mapping=dict(dataset.label_map), source="corpus config")
        if isinstance(dataset.label_map, str):
            name = dataset.label_map if dataset.label_map.endswith(".json") else f"{dataset.label_map}.json"
            table = LabelMap.load(str(self.directory / name)).model_copy(update={"dataset_id": dataset.id})
        if table is None:
            raise ValueError(f"{dataset.id}: no label_map — a corpus needs a label -> CL id table")
        return table.model_copy(update={"allow_local_types": True}) if unregistered else table


def _write(path: Path, body: str, *, force: bool) -> None:
    """Write ``body`` unless a *different* body is already there (``--force`` overrides).

    A corpus that was trained on must not silently become a different corpus under
    the same name, so a changed file is an error by default — the same rule the
    contracts build uses for frozen artifacts.
    """
    if path.exists() and not force and path.read_text(encoding=ENCODING) != body:
        raise ValueError(f"{path} already exists with different content (pass --force to overwrite)")
    write_text(path, body)


def _write_json(path: Path, value: object, *, force: bool) -> None:
    """``write_json`` with the same refusal (byte-identical formatting)."""
    _write(path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", force=force)


def load_config(path: str | Path) -> CorpusConfig:
    """Read a corpus YAML (unknown keys are errors, not ignored)."""
    import yaml

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    try:
        return CorpusConfig.model_validate(payload)
    except ValidationError as error:
        raise ValueError(f"{path}: invalid corpus config\n{error}") from None


def resolve_dataset(dataset: DatasetConfig, corpus: CorpusConfig, root: Path | None = None) -> dict:
    """Merge a corpus dataset with its registry entry; return the adapter's arguments.

    The registry is what says how a real dataset is shaped; a corpus may override
    it (the local copy of a file, a narrower tissue map) but never silently ignore
    it — a dataset marked ``registered: true`` whose id is not in the registry is
    an error, not a fallback.
    """
    entry = None
    loaded = registry.load()
    if dataset.access and dataset.registered:
        loaded = registry.with_access(loaded, dataset.id, dataset.access)
    if dataset.registered:
        entry = loaded.get(dataset.id)
    settings = {
        "format": dataset.format or (entry.format if entry else None),
        "tier": dataset.tier or (entry.tier if entry else None),
        "tissue": dataset.tissue or (entry.tissue if entry else None),
        "tissue_labels": dataset.tissue_labels or (dict(entry.tissue_labels) if entry else {}),
        "cap": dataset.cap if dataset.cap is not None else (entry.default_cap if entry else None),
    }
    missing = [name for name in ("format", "tier", "cap") if settings[name] is None]
    if missing:
        raise ValueError(f"{dataset.id}: {missing} must be set (not in the config, not in the registry)")
    settings["provenance"] = dataset.provenance or (entry.provenance if entry else "")
    settings["licence"] = entry.licence if entry else None
    settings["path"] = dataset.path
    if dataset.fetch and dataset.path is None:
        cache = Path(corpus.cache) if corpus.cache else paths.data_dir() / "datasets"
        settings["path"] = str(_primary_file(fetch_module.fetch(dataset.id, cache, registry=loaded), entry))
    if settings["path"] is None:
        raise ValueError(f"{dataset.id}: no path and fetch is false")
    return settings


def _primary_file(fetched: str | Path, entry) -> Path:
    """The file to hand the adapter inside a fetched dataset, not the directory.

    ``fetch`` returns the *dataset* path — a directory, since it is also what a 10x
    bundle resolves to. An h5ad adapter opening that directory is an
    ``IsADirectoryError``, so the registry's first access entry names the file that
    belongs to the format. Falls back to the directory when there is no such file,
    which is what a bundle wants.
    """
    fetched = Path(fetched)
    if not fetched.is_dir() or entry is None or not entry.access:
        return fetched
    candidate = fetched / entry.access[0].name
    return candidate if candidate.exists() else fetched


def dataset_digest(path: str | Path) -> str:
    """Digest of a dataset's bytes, so the corpus manifest pins its inputs."""
    path = Path(path)
    return fetch_module.tree_digest(path) if path.is_dir() else file_digest(path)


def adapter_kwargs(dataset: DatasetConfig, settings: dict, label_map: LabelMap) -> dict:
    """The keyword arguments the concrete adapter takes (each ignores what it has no use for)."""
    kwargs = {
        "tissue_labels": settings["tissue_labels"],
        "default_tissue": settings["tissue"],
    }
    if settings["format"] == "cellxgene_h5ad":
        kwargs |= {"genes_column": dataset.genes_column, "cell_column": dataset.cell_column}
    elif settings["format"] == "tenx_mtx":
        kwargs |= {
            "label_file": dataset.label_file,
            "label_column": dataset.label_column,
            "donor_id": dataset.donor_id,
        }
    elif settings["format"] == "baron_archive":
        # An archive of per-donor tables: the label column is the only thing the
        # config names, because the donor is the member and the cell key is `barcode`.
        kwargs |= {"label_column": dataset.label_column}
    else:
        kwargs |= {
            "cell_column": dataset.cell_column,
            "donor_id": dataset.donor_id,
            "donor_from_stem": dataset.donor_from_stem,
        }
    return kwargs


def cell_records(
    config: CorpusConfig,
    artifacts: Artifacts,
    settings: dict,
    dataset: DatasetConfig,
    *,
    stats: LabelStats | None,
    unregistered: bool = False,
    expression: bool = True,
    only: Collection[str] | None = None,
):
    """A fresh generator over one dataset's labelled cells.

    ``unregistered=True`` also yields the cells the register cannot name, under
    explicit dataset-local ids (see :meth:`Artifacts.label_map`), which is what
    the out-of-register family is built from. ``expression``/``only`` are the
    adapter's: the registration pass needs no expression, and the record pass
    needs it only for the cells a split kept.
    """
    label_map = artifacts.label_map(dataset, unregistered=unregistered)
    return adapter_for(settings["format"]).to_cells(
        Path(settings["path"]),
        label_map,
        columns=_obs_columns(settings, dataset),
        stats=stats,
        expression=expression,
        only=only,
        **adapter_kwargs(dataset, settings, label_map),
    )


def _obs_columns(settings: dict, dataset: DatasetConfig) -> ObsColumns:
    """The ``obs`` column names: the config's, else the registry entry's."""
    if dataset.obs is not None:
        return dataset.obs
    if dataset.registered:
        return registry.load().get(dataset.id).obs
    return ObsColumns(label=dataset.label_column, donor=None, tissue=None)


def dataset_probe(config: CorpusConfig, artifacts: Artifacts, settings: dict, dataset: DatasetConfig) -> dict:
    """What the adapter found in the file, for the corpus report."""
    info = probe_dataset(settings["format"], Path(settings["path"]), columns=_obs_columns(settings, dataset))
    payload = info.payload()
    payload.pop("path", None)      # a corpus report never carries a path that may be absolute
    payload["id"] = dataset.id
    payload["registered"] = dataset.registered
    return payload


def records_for_cell(
    cell: CellRecord,
    panel_cfg: PanelConfig,
    ctx: FamilyContext,
    config: CorpusConfig,
    rng_seed: int,
) -> tuple[list[Record], Counter]:
    """Every record family for one cell, given the panel drawn for it.

    Returns the records and a counter of the families that were not applicable
    (the two evidence families need the cell to express the markers they mask) —
    a skipped family is counted, never silently absent.
    """
    skipped: Counter = Counter()
    seed_base = seed_int(rng_seed, "family", config.name)
    none = choose_none(seeded_rng(rng_seed, "none"))
    # §1: the panel comes from the tissue's list when the register maps the cell's
    # tissue, and from the register itself when it does not. The scope name travels
    # on the panel, so every record says which of the two it was drawn from.
    scope, _ = panel_cfg.resolve(cell.tissue)
    if not ctx.registered(cell.cl_id):
        # §2: the register cannot name this cell, so abstaining is the right answer
        # and the panel is a genuine miss by construction. There is no sibling and
        # no definition of the true type, so this cell gets this family alone.
        if RecordFamily.OUT_OF_REGISTER not in config.families:
            return [], skipped
        panel = sample_miss_panel(scope, ctx.definitions, panel_cfg, seeded_rng(rng_seed, "panel"))
        reason = UNREGISTERED_TYPE if scope != GLOBAL_TISSUE else UNREGISTERED_TISSUE
        return [out_of_register(cell, panel, ctx, seed=seed_base, none=none, reason=reason)], skipped
    panel, sibling = sample_panel_pair(cell.cl_id, scope, ctx.definitions, panel_cfg, seeded_rng(rng_seed, "panel"))
    base_panel = sibling or panel        # the panel that contains the true type
    records: list[Record] = []
    if sibling is None:
        if RecordFamily.PANEL_CHOICE in config.families:
            records.append(panel_choice(cell, base_panel, ctx, seed=seed_base))
    elif RecordFamily.PANEL_MISS in config.families:
        miss, present = panel_miss_pair(cell, panel, sibling, ctx, seed=seed_base, none=none)
        records.append(miss)
        if RecordFamily.PANEL_CHOICE in config.families:
            records.append(present)
    if RecordFamily.MARKER_NOUL in config.families:
        records.append(marker_noul(cell, base_panel, ctx, seed=seed_base))
    for family, builder in ((RecordFamily.EVIDENCE_REMOVED, evidence_removed), (RecordFamily.EVIDENCE_DILUTED, evidence_diluted)):
        if family not in config.families:
            continue
        record = builder(cell, base_panel, ctx, seed=seed_base, none=none)
        if record is None:
            skipped[family.value] += 1
        else:
            records.append(record)
    if config.axis is not None and RecordFamily.AXIS_SCORE in config.families:
        records.append(axis_score(cell, base_panel, ctx, config.axis, seed=seed_base))
    return records, skipped


def variant_records(records: list[Record], split: str, seed: int, *, policy: str = "twins") -> list[Record]:
    """Apply the permutation policy: always in train, clean + permuted twin elsewhere.

    A model trained on one fixed option order learns the order; a model evaluated
    on it cannot be compared with a serving request that shuffles. So training
    sees only permuted records, and development/test carry both presentations of
    the same record, which is what lets the paper report order sensitivity.

    ``policy="single"`` keeps the train half of that rule and drops the twin: every
    split carries exactly one rendering of a record — permuted in train, clean
    everywhere else — so a ``(dataset, cell, family)`` triple is one question,
    counted and scored once. The option-order control then has to be a corpus of
    its own rather than a second row under the same one.
    """
    if policy not in ("twins", "single"):
        raise ValueError(f"unknown variant policy {policy!r} (have: twins, single)")
    out: list[Record] = []
    for record in records:
        permuted = permute_options(record, seed_int(seed, "permute", record.meta.family.value, record.meta.cl_id))
        if split == "train":
            out.append(permuted)
        elif policy == "single":
            out.append(record)
        else:
            out.append(record)
            out.append(permuted)
    return out


def state_tokenizer(config: CorpusConfig) -> Measure | None:
    """The corpus's pinned tokenizer as a ``state -> tokens`` function, or ``None`` when it names none.

    The tokenizer is kev's own, loaded the way the trainer loads it, so the
    report's token counts are the counts a run will hit. kev is not installed by
    the ``data`` extra, so the import is inside the function: a corpus config that
    names no tokenizer builds without it (and reports the estimate instead).
    """
    if config.state.tokenizer is None:
        return None
    tokenizer = budget.pinned_tokenizer(config.state.tokenizer.model_dump())
    return partial(budget.state_tokens, tokenizer)


def build(config_path: str | Path, out: str | Path, *, force: bool = False) -> dict:
    """Build the corpus described by ``config_path`` into ``out``. Returns the manifest."""
    config = load_config(config_path)
    artifacts = Artifacts(Path(config.artifacts) if config.artifacts else paths.artifacts_dir() / config.version)
    panel_cfg = PanelConfig.from_artifacts(
        artifacts.tissue_map,
        artifacts.cl_terms,
        **config.panel.model_dump(exclude={"tissues", "lineage", "register_terms"}),
    )
    settings = {dataset.id: resolve_dataset(dataset, config) for dataset in config.datasets}
    caps = SplitCaps(per_dataset={dataset.id: settings[dataset.id]["cap"] for dataset in config.datasets}, per_type=config.splits.per_type)
    caps = caps.model_copy(update={"per_dataset": {**caps.per_dataset, **config.splits.per_dataset}})
    # A cell the register cannot name is a different population from one it can, so
    # it is capped separately: one budget over both would let an atlas whose labels
    # are mostly out of register (siletti: 79% of its cells) starve the register's.
    caps = caps.model_copy(update={"per_unregistered": {**caps.per_unregistered, **config.splits.unregistered}})
    registered = frozenset(artifacts.definitions.by_cl_id)
    splits = SplitRegistry(
        seed=config.seed,
        tiers={dataset.id: settings[dataset.id]["tier"] for dataset in config.datasets},
        caps=caps,
        calibration_frac=config.splits.calibration_frac,
        development_frac=config.splits.development_frac,
        test_frac=config.splits.test_frac,
        registered=registered,
    )
    probes, stats = {}, {}
    for dataset in config.datasets:
        probes[dataset.id] = dataset_probe(config, artifacts, settings[dataset.id], dataset)
        entry_stats = LabelStats(registered=registered)
        splits.register(
            dataset.id,
            (
                (cell.cell_id, cell.donor_id, cell.cl_id)
                for cell in cell_records(
                    config, artifacts, settings[dataset.id], dataset, stats=entry_stats, unregistered=True, expression=False
                )
            ),
        )
        stats[dataset.id] = entry_stats
    splits.build()

    built: dict[str, list[Record]] = {split: [] for split in SPLIT_NAMES}
    skipped: dict[str, Counter] = {dataset.id: Counter() for dataset in config.datasets}
    # split -> dataset -> family -> n, and dataset -> panel scope -> cells: the
    # per-atlas reports §3 and §1 ask for, counted where the records are made.
    per_split_dataset: dict[str, dict[str, Counter]] = {split: {} for split in SPLIT_NAMES}
    scopes: dict[str, Counter] = {dataset.id: Counter() for dataset in config.datasets}
    for dataset in config.datasets:
        kept = splits.kept(dataset.id)
        ctx = FamilyContext(
            definitions=artifacts.definitions,
            filter=artifacts.filter,
            corpus=config.name,
            holds=config.holds,
            leakage_alpha=config.leakage_alpha,
            de_markers=dataset.de_markers,
            dilution_k=config.dilution_k,
        )
        dataset_records: list[tuple[str, Record]] = []
        # `only=kept` keeps the adapter from reading expression for the cells the caps
        # sampled away: on a 42 GB atlas that is the difference between a build and a
        # night. `kept` is the split registry's own answer, so a cell is read exactly
        # when it is written.
        for cell in cell_records(
            config, artifacts, settings[dataset.id], dataset, stats=None, unregistered=True, only=kept
        ):
            if cell.cell_id not in kept:
                continue
            split = splits.split_of(dataset.id, cell.cell_id)
            cell_records_out, cell_skipped = records_for_cell(
                cell, panel_cfg, ctx, config, seed_int(config.seed, "cell", dataset.id, cell.cell_id)
            )
            skipped[dataset.id] |= cell_skipped
            cell_records_out = variant_records(cell_records_out, split, config.seed, policy=config.variant_policy)
            scopes[dataset.id][panel_cfg.resolve(cell.tissue)[0]] += 1
            for record in cell_records_out:
                per_split_dataset[split].setdefault(dataset.id, Counter())[record.meta.family.value] += 1
            dataset_records.extend((split, record) for record in cell_records_out)
        dataset_records.sort(key=lambda item: (item[1].meta.cell_id or "", FAMILY_ORDER.index(item[1].meta.family), item[1].meta.variant, item[1].meta.cl_id))
        for split, record in dataset_records:
            built[split].append(record)

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    files: dict[str, dict] = {}
    for split in SPLIT_NAMES:
        body = "".join(json.dumps(record.to_kev(), ensure_ascii=False) + "\n" for record in built[split])
        name = f"{split}.jsonl"
        _write(out / name, body, force=force)
        files[name] = {
            "sha256": text_digest(body),
            "bytes": len(body.encode(ENCODING)),
            "kind": "data",
            "records": len(built[split]),
            "questions": sum(len(record.questions) for record in built[split]),
        }

    everything = [record for split in SPLIT_NAMES for record in built[split]]
    report = qc.report(
        everything,
        artifacts.filter,
        seed=config.seed,
        max_tokens=config.state.max_tokens,
        tokens_per_symbol=config.state.tokens_per_symbol,
        tokenize=state_tokenizer(config),
        n_permutations=config.state.qc_permutations,
        max_features=config.state.qc_max_features,
    )
    report["datasets"] = probes
    report["labels"] = {dataset_id: entry.payload() for dataset_id, entry in sorted(stats.items())}
    report["skipped"] = {dataset_id: {family: n for family, n in sorted(counter.items())} for dataset_id, counter in sorted(skipped.items())}
    # §1: cells per panel scope, and the split between the two kinds of scope. A
    # scope the register maps is a tissue list; ``global`` is the register itself.
    report["tissue_resolution"] = {
        dataset_id: {
            "cells_per_scope": {scope: n for scope, n in sorted(counter.items())},
            "tissue_list": sum(n for scope, n in counter.items() if scope != GLOBAL_TISSUE),
            "global": counter.get(GLOBAL_TISSUE, 0),
        }
        for dataset_id, counter in sorted(scopes.items())
    }
    _write_json(out / "qc.json", report, force=force)
    files["qc.json"] = {"sha256": file_digest(out / "qc.json"), "bytes": (out / "qc.json").stat().st_size, "kind": "report"}

    manifest = {
        "name": config.name,
        "version": config.version,
        "seed": config.seed,
        "config_digest": payload_digest(config.model_dump(mode="json")),
        "datasets": {
            dataset.id: {
                "digest": dataset_digest(settings[dataset.id]["path"]),
                "registered": dataset.registered,
                "tier": settings[dataset.id]["tier"],
                "licence": settings[dataset.id]["licence"],
                "path": settings[dataset.id]["path"],
                "cap": settings[dataset.id]["cap"],
            }
            for dataset in config.datasets
        },
        "artifacts": {"version": config.version, "digests": artifacts.digests},
        "files": files,
        "counts": {
            "records": len(everything),
            "questions": sum(len(record.questions) for record in everything),
            "cells": len({record.meta.group_id for record in everything}),
            "per_family": {family.value: sum(1 for record in everything if record.meta.family == family) for family in FAMILY_ORDER},
            "per_split": {split: len(built[split]) for split in SPLIT_NAMES},
            # §3: per atlas, and per atlas *and* family — the table a reader needs
            # to see which atlas fed which family, which one global total cannot say.
            "per_dataset": {
                dataset.id: {
                    "records": sum(sum(per_split_dataset[split].get(dataset.id, Counter()).values()) for split in SPLIT_NAMES),
                    "per_family": {
                        family.value: sum(per_split_dataset[split].get(dataset.id, Counter()).get(family.value, 0) for split in SPLIT_NAMES)
                        for family in FAMILY_ORDER
                    },
                    "cells_read": stats[dataset.id].total,
                    "cells_in_scope": stats[dataset.id].mapped,
                    "cells_dropped": stats[dataset.id].dropped,
                    "in_scope_rate": round(stats[dataset.id].rate, 6),
                }
                for dataset in config.datasets
            },
        },
        "splits": splits.payload(),
    }
    _write_json(out / "manifest.json", manifest, force=force)
    return manifest


def _report(manifest: dict) -> None:
    print(f"{manifest['name']}: seed {manifest['seed']}, {manifest['counts']['records']} records over {manifest['counts']['cells']} cells")
    for name, entry in sorted(manifest["files"].items()):
        print(f"  {name:20s} {entry['sha256'][:12]}  {entry['bytes']:>9} bytes  {entry.get('records', '-'):>6} records")
    for split, counts in manifest["splits"]["splits"].items():
        print(f"  {split:20s} {counts['cells']:>6} cells  {counts['donors']:>4} donors  {sorted(counts['datasets'])}")
    print(f"  families: {manifest['counts']['per_family']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sckevtype.data.build", description="Build a kev-loadable supervision corpus.")
    parser.add_argument("--config", required=True, help="corpus config YAML")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--force", action="store_true", help="overwrite files whose bytes changed")
    args = parser.parse_args(argv)
    try:
        manifest = build(args.config, args.out, force=args.force)
    except (SystemExit, ValueError, KeyError, FileNotFoundError, fetch_module.FetchError) as error:
        print(f"build failed: {error}", file=sys.stderr)
        return 1
    _report(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
