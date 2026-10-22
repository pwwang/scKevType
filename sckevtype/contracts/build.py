"""CLI: build the stage-1 frozen artifacts from the in-repo seed inputs.

    python -m sckevtype.contracts.build --out artifacts/v1 [--seed N] [--force]

Inputs are the small, human-editable files in ``configs/seed/`` (a gene list,
and YAML for the filter, the definitions, the label map, the tissue map and the
CL snapshot). The dataset-derived versions of these resources arrive in Step 2
with the dataset adapters; this build proves the artifact formats, the digests
and the QA gates.

Two properties the build guarantees:

* **no constant changes behind your back** — if a rebuild would change the bytes
  of a file in an existing version, the build refuses and tells you to cut a new
  version with ``freeze.new_version`` (``--force`` overrides, for development);
* **byte-stable rebuilds** — the same seeds produce the same files and the same
  manifest, so a verify after a rebuild is meaningful.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from . import paths
from .definitions import DefinitionSet
from .freeze import (
    artifact_bytes,
    file_digest,
    git_commit,
    read_json,
    update_manifest,
    write_artifact,
)
from .ontology import ClTerm, LabelMap, TissueMap, cl_terms_payload
from .render import FilterConfig
from .schema import SuiteManifest
from .vocab import VocabConfig, build_vocabulary

DEFAULT_NAME = "sckevtype-contracts"
#: Contracts modules whose content determines the artifacts, hashed into the
#: manifest the way ``kev.suite`` hashes its converter modules.
CODE_FILES = ("build.py", "definitions.py", "freeze.py", "ontology.py", "paths.py", "render.py", "schema.py", "vocab.py")
SEED_FILES = (
    "genes.seed.txt",
    "filter.seed.yaml",
    "cl_terms.seed.yaml",
    "definitions.seed.yaml",
    "label_map.seed.yaml",
    "tissue_map.seed.yaml",
)


def read_genes(path: str | Path) -> list[str]:
    """Read a one-symbol-per-line gene list (``#`` starts a comment)."""
    lines = [line.split("#", 1)[0].strip() for line in Path(path).read_text(encoding="utf-8").splitlines()]
    return [line for line in lines if line]


def seed_inputs(seed_dir: str | Path) -> dict:
    """Parse every seed input: the raw material of a build, all of it in-repo."""
    seed_dir = Path(seed_dir)

    def load(name: str):
        return yaml.safe_load((seed_dir / name).read_text(encoding="utf-8"))

    return {
        "genes": read_genes(seed_dir / "genes.seed.txt"),
        "filter": load("filter.seed.yaml"),
        "cl_terms": load("cl_terms.seed.yaml"),
        "definitions": load("definitions.seed.yaml"),
        "label_map": load("label_map.seed.yaml"),
        "tissue_map": load("tissue_map.seed.yaml"),
    }


def build(out_dir: str | Path, seed_dir: str | Path | None = None, seed: int = 0, name: str = DEFAULT_NAME, force: bool = False) -> dict:
    """Materialise one artifact version. Returns the manifest it wrote."""
    out_dir = Path(out_dir)
    seed_dir = Path(seed_dir) if seed_dir else paths.seed_dir()
    raw = seed_inputs(seed_dir)
    manifest_path = out_dir / "manifest.json"
    version = out_dir.name

    # --- payloads (built before anything is written, so a failure writes nothing)
    filter_cfg = FilterConfig.model_validate(raw["filter"]["filter"])
    vocabulary = build_vocabulary(raw["genes"], VocabConfig.model_validate(raw["filter"]["vocabulary"]))
    cl_terms: dict[str, ClTerm] = {term.cl_id: term for term in (ClTerm.model_validate(t) for t in raw["cl_terms"]["terms"])}
    definitions = DefinitionSet.model_validate(raw["definitions"])
    # `labels`/`min_coverage` are the seed's own QA settings, not artifact fields.
    label_map = LabelMap.model_validate({key: value for key, value in raw["label_map"].items() if key in LabelMap.model_fields})
    tissue_map = TissueMap.model_validate({**raw["tissue_map"], "definitions": definitions})

    # --- QA gates: a build fails here, not in an evaluation three steps later
    for cl_id in definitions.by_cl_id:
        if cl_id not in cl_terms:
            raise ValueError(f"definition {cl_id} is not in the CL snapshot {seed_dir / 'cl_terms.seed.yaml'}")
    dangling = sorted({parent for term in cl_terms.values() for parent in term.parents} - set(cl_terms))
    if dangling:
        raise ValueError(f"CL snapshot has parents that are not terms: {dangling} (hierarchical metrics would silently drop them)")
    unknown = sorted({mapped for mapped in label_map.mapping.values()} - set(cl_terms))
    if unknown:
        raise ValueError(f"{label_map.dataset_id}: label map points at CL ids that are not in the snapshot: {unknown}")
    coverage = label_map.require_min_coverage(raw["label_map"]["labels"], raw["label_map"].get("min_coverage", 0.95))
    panels = {tissue: tissue_map.expand(tissue) for tissue in tissue_map.tissues}   # expansion must not raise

    payloads = {
        "render_config.json": ("render_config", filter_cfg.model_dump(mode="json"), None),
        "vocabulary.json": ("vocabulary", {**vocabulary.payload(), "digest": vocabulary.digest()}, len(vocabulary)),
        "cl_terms.json": ("ontology", cl_terms_payload(raw["cl_terms"].get("source", ""), cl_terms), len(cl_terms)),
        "definitions.json": ("definitions", {**definitions.payload(), "digest": definitions.digest()}, len(definitions.definitions)),
        f"label_map.{label_map.dataset_id}.json": ("label_map", {**label_map.payload(), "digest": label_map.digest()}, len(label_map.mapping)),
        "tissue_map.json": ("tissue_map", {**tissue_map.payload(), "digest": tissue_map.digest()}, len(tissue_map.tissues)),
    }

    # --- frozen files are never rewritten with different content
    for file_name, (_, payload, _) in payloads.items():
        target = out_dir / file_name
        if target.exists() and target.read_bytes() != artifact_bytes(payload) and not force:
            raise SystemExit(
                f"{target} already exists with different content: artifacts are frozen. "
                f"Cut a new version (python -m sckevtype.contracts.freeze --new-version v2 {out_dir}) or pass --force."
            )

    update_manifest(
        manifest_path,
        name=name,
        version=version,
        seed=seed,
        git_commit=git_commit(paths.root()),
        source_digests={file_name: file_digest(seed_dir / file_name) for file_name in SEED_FILES},
        code_hashes={file_name: file_digest(Path(__file__).parent / file_name) for file_name in CODE_FILES},
    )
    for file_name, (kind, payload, counts) in payloads.items():
        write_artifact(out_dir, file_name, payload, manifest_path=manifest_path, kind=kind, counts=counts)
    update_manifest(manifest_path, counts={file_name: counts for file_name, (_, _, counts) in payloads.items() if counts is not None})

    manifest = SuiteManifest.model_validate(read_json(manifest_path))
    _report(out_dir, manifest, seed_dir, panels, coverage)
    return manifest.model_dump(mode="json")


def _report(out_dir: Path, manifest: SuiteManifest, seed_dir: Path, panels: dict, coverage) -> None:
    print(f"built {out_dir} (version {manifest.version}) from {seed_dir}", flush=True)
    for file_name, entry in sorted(manifest.files.items()):
        counts = f", {entry.counts} entries" if entry.counts is not None else ""
        print(f"  {file_name:<34} {entry.sha256[:12]}…  {entry.bytes:>7} B  [{entry.kind}{counts}]", flush=True)
    print("  panels: " + ", ".join(f"{tissue}({len(spec.options)})" for tissue, spec in sorted(panels.items())), flush=True)
    print(f"  label coverage: {coverage.rate:.3f} ({coverage.mapped}/{coverage.total})", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a frozen artifacts version from configs/seed/.")
    parser.add_argument("--out", required=True, help="artifact version directory, e.g. artifacts/v1")
    parser.add_argument("--seed-dir", help="seed inputs (default: configs/seed)")
    parser.add_argument("--seed", type=int, default=0, help="recorded in the manifest; stage 1 samples nothing")
    parser.add_argument("--name", default=DEFAULT_NAME)
    parser.add_argument("--force", action="store_true", help="rewrite a frozen file in place (development only; breaks the freeze rule)")
    args = parser.parse_args(argv)
    try:
        build(args.out, args.seed_dir, args.seed, args.name, args.force)
    except (SystemExit, ValueError, KeyError, FileNotFoundError) as error:
        print(f"build failed: {error}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
