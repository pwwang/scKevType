"""Every atlas label a corpus wires must be a candidate for that cell's tissue.

The build's own invariant, made testable before the build (``docs/SPEC-step9-tissue-map-sync.md``):
for a dataset wired with tissue ``T`` and a ``label_map``, every cell whose label the
register can name and whose tissue resolves to ``T`` needs ``T``'s candidate list to
contain that label — otherwise ``sckevtype.data.panels.sample_panel`` raises, 69 GB
into the read. `CL:0001054` (CD14+ monocyte, added to the scope in Step 7) was missing
from ``blood`` exactly that way, because ``artifacts/v2/tissue_map.json`` was a
hand-kept list while the scope was machine-derived.

Two tests, because the corpus is 87 GB and may not be there:

* :func:`test_every_wired_atlas_label_has_a_candidate_for_its_tissue` reads each wired
  atlas's labels — two ``obs`` columns for an h5ad, the sidecar for a 10x bundle, never
  the expression matrix — and checks the real labels. It **skips** when no wired atlas
  is on disk, naming what could not be read.
* :func:`test_the_committed_measurement_agrees_with_the_wired_atlases` needs no atlas
  bytes: it checks the measurement the tissue map was derived from
  (``configs/seed/v2/atlas_tissue_labels.json``) and fails when a wired atlas that *is*
  on disk is absent from it, i.e. when the map has gone stale against the wiring. This
  is the half that runs on a CPU-only box, so an absent corpus never means "unchecked".

Neither test needs the network or a GPU.
"""

from __future__ import annotations

import json
import sys
import warnings
from collections import Counter
from pathlib import Path

import h5py
import pytest

from sckevtype.data import registry as registry_module
from sckevtype.data.adapters import resolve_tissue
from sckevtype.data.adapters.cellxgene_h5ad import _column, _obs_keys, resolve_column
from sckevtype.data.adapters.tenx_mtx import LABEL_STEMS, _find, read_labels
from sckevtype.data.build import Artifacts, DatasetConfig, load_config
from sckevtype.data.panels import PanelConfig

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))  # the derivation's own atlas-path helper

from curate_register_v2 import atlas_file

from sckevtype.data.pull import resolve_cache

CORPUS_DIR = ROOT / "configs" / "corpus"
MEASUREMENT = ROOT / "configs" / "seed" / "v2" / "atlas_tissue_labels.json"
CACHE = ROOT / "data" / "raw"
#: Formats whose labels this test can read without touching the expression matrix.
READERS = ("cellxgene_h5ad", "tenx_mtx")

#: A corpus config is a file to parse and an atlas a 2-45 GB read, so both are read once.
_CONFIGS: dict[str, tuple[Path, object] | None] = {}
_ARTIFACTS: dict[str, Artifacts] = {}
_LABELS: dict[tuple, dict[tuple[str | None, str], int]] = {}


def corpus_configs() -> list[tuple[Path, object]]:
    """Every ``configs/corpus/*.yaml`` that is a corpus (the assets file is not)."""
    if not _CONFIGS:
        for path in sorted(CORPUS_DIR.glob("*.yaml")):
            try:
                _CONFIGS[path.name] = (path, load_config(path))
            except ValueError:
                pass  # not a corpus config: no datasets, so nothing is wired
    return [entry for entry in _CONFIGS.values() if entry is not None]


def artifacts_for(version: str) -> Artifacts:
    if version not in _ARTIFACTS:
        _ARTIFACTS[version] = Artifacts(ROOT / "artifacts" / version)
    return _ARTIFACTS[version]


def atlas_path(dataset: DatasetConfig) -> Path | None:
    """Where the corpus reads this atlas from — its own path, else the registry's cache entry.

    Through the derivation's own :func:`atlas_file`, so a config that reaches an atlas by
    its registry id is resolved (cross-source atlas included) exactly as the map's
    measurement resolved it. Re-deriving the path here read ``dominguez_conde`` as absent
    while its bytes were on disk under the CELLxGENE name.
    """
    if dataset.path:
        return ROOT / dataset.path
    registry = registry_module.load()
    entry = registry.get(dataset.id)
    if dataset.access:
        entry = registry_module.with_access(registry, dataset.id, dataset.access).get(dataset.id)
    if not entry.access:
        return None
    return atlas_file(dataset.id, entry, resolve_cache("data/raw"))


def label_pairs(path: Path, dataset: DatasetConfig) -> dict[tuple[str | None, str], int]:
    """``((raw tissue value, raw label), cells)`` — the columns the build's adapter reads."""
    obs = dataset.obs  # a 10x dataset names its label sidecar instead of an obs column
    key = (path, obs.label if obs else None, obs.tissue if obs else None, dataset.label_file)
    if key in _LABELS:
        return _LABELS[key]
    if dataset.format == "tenx_mtx":
        # A 10x bundle carries no tissue column and its labels live in a sidecar; the
        # adapter resolves the tissue from the registry default alone.
        sidecar = Path(dataset.label_file) if dataset.label_file else _find(path, LABEL_STEMS, required=False)
        labels = read_labels(sidecar, label_column=dataset.label_column) if sidecar and sidecar.exists() else {}
        pairs: dict[tuple[str | None, str], int] = {}
        for label in labels.values():
            pairs[(None, label)] = pairs.get((None, label), 0) + 1
    else:
        with h5py.File(path, "r") as handle:
            keys = _obs_keys(handle)
            label_column = resolve_column(keys, obs.label if obs else None, "label")
            tissue_column = resolve_column(keys, obs.tissue if obs else None, "tissue")
            labels = _column(handle, "obs", label_column) if label_column else []
            tissues = _column(handle, "obs", tissue_column) if tissue_column else [None] * len(labels)
        pairs = {}
        for text, label in zip(tissues, labels):
            pairs[(str(text), label)] = pairs.get((str(text), label), 0) + 1
    _LABELS[key] = pairs
    return pairs


def wired_datasets():
    """``(config, dataset, path or None)`` for every dataset a corpus gives a label map."""
    for _config_path, config in corpus_configs():
        for dataset in config.datasets:
            if dataset.label_map is None:
                continue  # no table: no cell of this atlas can name a register type
            yield config, dataset, atlas_path(dataset)


def test_every_wired_atlas_label_has_a_candidate_for_its_tissue():
    """The build's invariant, on the real labels: a label with no candidate is a failure."""
    checked, absent, unreadable, failures = [], [], [], Counter()
    for config, dataset, path in wired_datasets():
        if path is None or not path.exists():
            absent.append(f"{config.name}/{dataset.id}")
            continue
        if dataset.format not in READERS:
            unreadable.append(f"{config.name}/{dataset.id} ({dataset.format})")
            continue
        metadata = artifacts_for(config.version)
        mapping = metadata.label_map(dataset).mapping
        # The build's own rule for a cell's candidates, so the test cannot disagree with
        # it: a tissue the map carries gets its list, anything else the whole register.
        panel = PanelConfig.from_artifacts(metadata.tissue_map, metadata.cl_terms)
        pairs = label_pairs(path, dataset)
        if not pairs:
            unreadable.append(f"{config.name}/{dataset.id} (no label column or sidecar found)")
            continue
        for (text, label), cells in pairs.items():
            cl_id = mapping.get(label) if label is not None else None
            tissue = resolve_tissue(text, dataset.tissue_labels, dataset.tissue)
            if cl_id is None or tissue is None:
                continue  # not a register type, or no register tissue: drawn on the register
            scope, candidates = panel.resolve(tissue)
            if cl_id not in candidates:
                failures[(config.name, dataset.id, scope, cl_id)] += cells
        checked.append(f"{config.name}/{dataset.id}")
    if failures:
        pytest.fail(
            "wired atlas labels with no candidate for their tissue — re-derive the map "
            "(uv run python scripts/derive_tissue_map.py) or fix the config:\n  "
            + "\n  ".join(
                f"{config}/{dataset}: {cells} cells are {cl_id} in tissue {tissue!r}"
                for (config, dataset, tissue, cl_id), cells in sorted(failures.items())
            )
        )
    if unreadable:
        warnings.warn(f"not checked (format this test does not read): {', '.join(unreadable)}", stacklevel=1)
    if not checked:
        pytest.skip(
            "no wired atlas is on disk, so no real label could be read; skipped: "
            + ", ".join(sorted(absent))
            + f" (looked under {CACHE})"
        )
    if absent:
        warnings.warn(f"checked {len(checked)}; not checked (bytes absent): {', '.join(sorted(absent))}", stacklevel=1)


def test_the_committed_measurement_agrees_with_the_wired_atlases():
    """The map's derivation input, checked without atlas bytes — and against the wiring."""
    if not MEASUREMENT.exists():
        pytest.fail(f"{MEASUREMENT} is missing: the tissue map records it as its input, so it must be committed")
    measurement = json.loads(MEASUREMENT.read_text(encoding="utf-8"))
    tissue_map = artifacts_for("v2").tissue_map

    for name, record in sorted(measurement.items()):
        for pair in sorted(record["pairs"]):
            tissue, cl_id = pair.split("|")
            assert tissue in tissue_map.tissues, f"{name}: measured tissue {tissue!r} is not in the tissue map"
            assert cl_id in tissue_map.tissues[tissue], (
                f"{name}: {record['pairs'][pair]} cells are {cl_id} in tissue {tissue!r}, which has no candidate for it"
            )

    # The measurement must cover the wiring it claims to describe. An atlas that is on
    # disk and wired with a tissue but absent from the measurement means the map was
    # derived before that atlas was wired, and the invariant is unchecked for it.
    measured_ids = {name.split(":", 1)[1] for name in measurement}
    stale = sorted(
        f"{config.name}/{dataset.id}"
        for config, dataset, path in wired_datasets()
        if dataset.registered  # a synthetic fixture is not an atlas and is never measured
        and dataset.id not in measured_ids
        and path is not None
        and path.exists()
    )
    assert not stale, (
        f"wired atlases on disk with no record in {MEASUREMENT.relative_to(ROOT)}: {', '.join(stale)} "
        "— re-run uv run python scripts/derive_tissue_map.py"
    )
