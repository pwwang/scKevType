"""Cell Ontology joins and the tissue -> types middleware.

Three frozen resources live here (design-notes §5, paper_plan §4.2):

* a **CL snapshot** — local, digest-pinned, never fetched by library code
  (``python -m sckevtype.contracts.fetch --what cl`` refreshes it, and is the
  only thing in the package that touches the network);
* a **label map** per dataset — ``dataset_label -> CL id``, the join key that
  makes multi-dataset training possible without free-text matching;
* a **tissue map** — ``tissue -> [CL id]``, which expands into the PanelSpec the
  middleware hands the model.

Unmappable labels are never coerced to the nearest CL term. They are dropped,
or kept as explicit ``local:<dataset>:<label>`` types when the artifact's policy
allows it, and the count is reported either way.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .definitions import DefinitionSet
from .freeze import payload_digest, read_json, write_json
from .schema import CoverageReport, PanelSpec
from .vocab import coverage

#: Prefix of a dataset-local type id. The dataset id is part of the id so two
#: datasets' local types can never collide.
LOCAL_PREFIX = "local:"


def local_type_id(dataset_id: str, label: str) -> str:
    """The explicit dataset-local CL id for an unmappable label."""
    slug = re.sub(r"[^a-z0-9]+", "_", label.casefold()).strip("_")
    return f"{LOCAL_PREFIX}{dataset_id}:{slug}"


class ClTerm(BaseModel):
    """One Cell Ontology term from the local snapshot."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    cl_id: str = Field(pattern=r"^(CL:\d{7}|local:.+)$")
    name: str = Field(min_length=1)
    parents: tuple[str, ...] = ()


def load_cl_terms(path: str) -> dict[str, ClTerm]:
    """Load the CL snapshot (``{"terms": [...]}``) from a local file.

    Library code never fetches: an offline snapshot is what makes a build
    reproducible, and a build that reached the network would not be.
    """
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    terms = [ClTerm.model_validate(term) for term in payload["terms"]]
    return {term.cl_id: term for term in terms}


def cl_terms_payload(source: str, terms: dict[str, ClTerm]) -> dict:
    """The *artifact* payload for a CL snapshot: terms sorted by id, with its own digest."""
    body = _cl_body(source, terms)
    return {**body, "digest": payload_digest(body)}


def _cl_body(source: str, terms: dict[str, ClTerm]) -> dict:
    return {
        "source": source,
        "terms": [term.model_dump(mode="json") for term in sorted(terms.values(), key=lambda term: term.cl_id)],
    }


def save_cl_terms(path: str, terms: dict[str, ClTerm], source: str) -> str:
    """Write a **seed** snapshot as YAML (human-editable input, no digest of its own).

    Seed inputs are inputs, not artifacts: the build recomputes and pins the
    digest when it writes ``cl_terms.json``.
    """
    body = _cl_body(source, terms)
    Path(path).write_text(yaml.safe_dump(body, sort_keys=True, allow_unicode=True), encoding="utf-8")
    return payload_digest(body)


class LabelMap(BaseModel):
    """A frozen ``dataset_label -> CL id`` table for one dataset."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset_id: str = Field(min_length=1)
    mapping: dict[str, str] = Field(min_length=1)
    source: str = ""
    version: str = "v1"
    #: Artifact-level policy: whether unmappable labels survive as dataset-local types.
    allow_local_types: bool = False

    def payload(self) -> dict:
        return self.model_dump(mode="json")

    def digest(self) -> str:
        return payload_digest(self.payload())

    def resolve(self, label: str) -> str | None:
        """The CL id for a label: the mapped term, a dataset-local id, or None (dropped)."""
        if label in self.mapping:
            return self.mapping[label]
        return local_type_id(self.dataset_id, label) if self.allow_local_types else None

    def coverage_report(self, labels) -> CoverageReport:
        """Canonical-mapping coverage of ``labels`` (local types count as unmapped)."""
        return coverage(labels, set(self.mapping))

    def map_labels(self, labels) -> tuple[dict[str, str], CoverageReport]:
        """Mapped labels plus a coverage report.

        The returned mapping includes dataset-local ids when the policy allows
        them; the report always counts those labels as unmapped, so a build can
        never look fully covered by inventing local types.
        """
        report = self.coverage_report(labels)
        resolved = {label: cl_id for label in labels if (cl_id := self.resolve(label))}
        return resolved, report

    def require_min_coverage(self, labels, min_coverage: float = 0.95) -> CoverageReport:
        """Gate: raise (listing the offenders) when coverage is below ``min_coverage``.

        Called at build time, so a label map that silently drops a quarter of a
        dataset's cells fails the build instead of skewing a later evaluation.
        """
        report = self.coverage_report(labels)
        if report.rate < min_coverage:
            raise ValueError(
                f"{self.dataset_id}: label-map coverage {report.rate:.3f} < {min_coverage:.3f}; "
                f"unmapped labels ({report.unmapped}): {list(report.unmapped_items)}"
            )
        return report

    def save(self, path: str) -> str:
        digest = self.digest()
        write_json(path, {**self.payload(), "digest": digest})
        return digest

    @classmethod
    def load(cls, path: str) -> LabelMap:
        payload = read_json(path)
        digest = payload.pop("digest", None)
        label_map = cls.model_validate(payload)
        if digest != label_map.digest():
            raise ValueError(f"{path}: label-map digest mismatch (recorded {digest}, actual {label_map.digest()})")
        return label_map


class TissueMap(BaseModel):
    """The middleware spec: tissue -> candidate CL ids, with the definitions to render."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tissues: dict[str, tuple[str, ...]] = Field(min_length=1)
    definitions: DefinitionSet
    source: str = ""
    version: str = "v1"

    @model_validator(mode="after")
    def _check(self) -> TissueMap:
        known = set(self.definitions.by_cl_id)
        missing = sorted({cl_id for cl_ids in self.tissues.values() for cl_id in cl_ids} - known)
        if missing:
            raise ValueError(f"tissue map references types with no definition: {missing}")
        return self

    def payload(self) -> dict:
        """Everything but the definitions themselves, which carry their own digest."""
        return {
            "tissues": {tissue: list(cl_ids) for tissue, cl_ids in sorted(self.tissues.items())},
            "source": self.source,
            "version": self.version,
            "definitions_digest": self.definitions.digest(),
        }

    def digest(self) -> str:
        return payload_digest(self.payload())

    def expand(self, tissue: str) -> PanelSpec:
        """The panel a middleware supplies for a tissue (source = ``middleware``).

        Panel size and composition come from the frozen tissue map, so the same
        tissue always yields the same panel; the training-time *sampler* that
        varies panels draws its own seeds elsewhere.
        """
        try:
            cl_ids = self.tissues[tissue]
        except KeyError:
            raise KeyError(f"no tissue {tissue!r} in the middleware spec (have: {sorted(self.tissues)})") from None
        return PanelSpec(
            options=tuple(self.definitions.to_panel_option(cl_id) for cl_id in cl_ids),
            tissue=tissue,
            source="middleware",
        )

    def save(self, path: str) -> str:
        digest = self.digest()
        write_json(path, {**self.payload(), "digest": digest})
        return digest

    @classmethod
    def load(cls, path: str, definitions: DefinitionSet) -> TissueMap:
        """Read a tissue map, re-attaching the definition set it was built against."""
        payload = read_json(path)
        digest = payload.pop("digest", None)
        tissue_map = cls(
            tissues={tissue: tuple(cl_ids) for tissue, cl_ids in payload["tissues"].items()},
            definitions=definitions,
            source=payload.get("source", ""),
            version=payload.get("version", "v1"),
        )
        if digest != tissue_map.digest():
            raise ValueError(f"{path}: tissue-map digest mismatch (recorded {digest}, actual {tissue_map.digest()})")
        return tissue_map
