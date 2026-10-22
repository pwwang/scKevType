"""The frozen dataset registry: ``configs/datasets.yaml``, validated and never invented.

Every access path, accession and licence string in that YAML comes verbatim from
``research/lit/08-name-datasets.md`` (the verified-access-path inventory). Nothing
here invents a URL: where the verified path is a landing page rather than a file,
the entry says so and :mod:`sckevtype.data.fetch` refuses to guess.

Loading validates the YAML against the models below with ``extra="forbid"``, so a
typo'd key is an error rather than a silently ignored field — the registry is the
frozen answer to "what is this dataset", and a registry that quietly drops a
field is a registry that lies.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from ..contracts import paths
from ..contracts.freeze import payload_digest

#: What a pinned access path points at:
#:   ``file``       a direct file URL (fetchable)
#:   ``collection`` a CELLxGENE collection API URL (fetch resolves its H5AD asset)
#:   ``page``       a landing page / accession page (fetch refuses: download by hand)
AccessKind = Literal["file", "collection", "page"]
DatasetFormat = Literal["cellxgene_h5ad", "tenx_mtx", "csv_expr"]
Tier = Literal["train", "T2", "T3"]


class PinnedFile(BaseModel):
    """One access path of a dataset, with its digest when known.

    ``sha256`` is ``None`` for a dataset whose bytes have not been pinned yet;
    ``DatasetEntry.verify_on_fetch`` then says whether a fetch may record the
    digest it actually saw (pinning the cache) or must refuse.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1, description="local file name")
    url: str = Field(min_length=1)
    kind: AccessKind = "file"
    #: For ``kind: "collection"``: the substring of the CELLxGENE dataset title to
    #: pick. ``None`` means the collection holds a single H5AD asset.
    selector: str | None = None
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    bytes: int | None = Field(default=None, ge=0)


class ObsColumns(BaseModel):
    """Which ``obs`` column holds what. ``None`` means the dataset does not ship it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    label: str | None = None
    donor: str | None = None
    tissue: str | None = None


class DatasetEntry(BaseModel):
    """One dataset: where it comes from, how to read it, and what it may be used for."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    tier: Tier
    format: DatasetFormat
    access: tuple[PinnedFile, ...] = Field(min_length=1)
    obs: ObsColumns
    #: ``var`` column holding the gene symbol. CELLxGENE files keep Ensembl ids in
    #: ``var/_index`` and the symbol in ``feature_name``; ``None`` lets the adapter
    #: use the symbol column when it is there and the index otherwise.
    var_genes: str | None = None
    #: Digest of the entry's *primary* file (``access[0]``); ``None`` until pinned.
    #: A single dataset-level digest cannot pin a three-file 10x bundle, which is
    #: why every access entry carries its own too.
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    #: When the digest is unknown: record the digest the fetch actually saw
    #: (``True``), or refuse to fetch until the YAML is pinned (``False``).
    verify_on_fetch: bool = True
    #: Label-map artifact name (``artifacts/<version>/<label_map>``), or None when
    #: the map is supplied with the corpus config (datasets shipping no labels).
    label_map: str | None = None
    #: Cells this dataset may contribute by default (a sampling policy, not a fact
    #: about the dataset — the realised counts land in the corpus manifest).
    default_cap: int = Field(ge=1)
    licence: str = Field(min_length=1)
    provenance: str = Field(min_length=1)
    #: Verified cell count from ``research/lit/08-name-datasets.md``; None when the
    #: inventory marks it unverified.
    n_cells: int | None = Field(default=None, ge=1)
    #: Canonical tissue used for panel sampling when the dataset is single-tissue.
    tissue: str | None = None
    #: ``obs`` tissue value -> canonical tissue name (the middleware's key).
    tissue_labels: dict[str, str] = Field(default_factory=dict)
    #: Where labels come from when the dataset ships none (community silver
    #: standard), recorded so a silver-standard label is never mistaken for one.
    silver_standard: str | None = None

    @model_validator(mode="after")
    def _reconcile_primary_digest(self) -> DatasetEntry:
        """The dataset-level ``sha256`` and ``access[0].sha256`` are one fact, not two."""
        primary = self.access[0]
        if self.sha256 and primary.sha256 and self.sha256 != primary.sha256:
            raise ValueError(f"{self.id}: sha256 disagrees with the primary access entry's digest")
        if self.sha256 and not primary.sha256:
            object.__setattr__(self, "access", ({**primary.model_dump(), "sha256": self.sha256}, *self.access[1:]))
        elif primary.sha256 and not self.sha256:
            object.__setattr__(self, "sha256", primary.sha256)
        return self

    @property
    def primary(self) -> PinnedFile:
        return self.access[0]

    @property
    def unpinned(self) -> tuple[str, ...]:
        """Names of access paths with no digest (the cache has to pin them on fetch)."""
        return tuple(file.name for file in self.access if file.sha256 is None)


class DatasetRegistry(BaseModel):
    """The whole registry: a frozen list of datasets plus the inventory it came from."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: str = Field(min_length=1)
    source: str = Field(min_length=1)
    datasets: tuple[DatasetEntry, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_ids(self) -> DatasetRegistry:
        ids = [entry.id for entry in self.datasets]
        duplicates = sorted({entry for entry in ids if ids.count(entry) > 1})
        if duplicates:
            raise ValueError(f"duplicate dataset ids: {duplicates}")
        return self

    @property
    def by_id(self) -> dict[str, DatasetEntry]:
        return {entry.id: entry for entry in self.datasets}

    def get(self, dataset_id: str) -> DatasetEntry:
        try:
            return self.by_id[dataset_id]
        except KeyError:
            raise KeyError(f"no dataset {dataset_id!r} in the registry (have: {sorted(self.by_id)})") from None

    def tiers(self, tier: Tier) -> tuple[DatasetEntry, ...]:
        return tuple(entry for entry in self.datasets if entry.tier == tier)

    def payload(self) -> dict:
        return self.model_dump(mode="json")

    def digest(self) -> str:
        return payload_digest(self.payload())


def with_access(registry: DatasetRegistry, dataset_id: str, access: tuple[PinnedFile, ...]) -> DatasetRegistry:
    """``registry`` with ``dataset_id``'s access paths replaced.

    A corpus may record where an atlas is *actually* fetched from when the registry's
    verified path is a landing page: ``configs/datasets.yaml`` is the inventory of what
    was verified, not a download plan. Nothing else about the entry changes, and the
    replacement is digest-unpinned until the first fetch pins the bytes it saw.
    """
    updated = registry.get(dataset_id).model_copy(update={"access": access, "sha256": None})
    datasets = tuple(updated if entry.id == dataset_id else entry for entry in registry.datasets)
    return registry.model_copy(update={"datasets": datasets})


def load(path: str | Path | None = None) -> DatasetRegistry:
    """Read and validate the registry (default: ``configs/datasets.yaml``)."""
    path = Path(path) if path else paths.configs_dir() / "datasets.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    try:
        return DatasetRegistry.model_validate(payload)
    except ValidationError as error:
        raise ValueError(f"{path}: invalid dataset registry\n{error}") from None
