"""Dump the CellTypist models' vocabularies and file digests.

The mapping from a CellTypist label to a CL id is the one place a baseline needs a
human decision, so it is made here explicitly and written to
``out/celltypist_models.json`` for review: the model's own label list, the gene
list length, the model file's sha256, and -- for each label -- the CL ids whose
name matches it exactly after normalisation, or ``null``. Labels that resolve to
nothing are *not* guessed at: a prediction that cannot be expressed as a CL id is
scored as ``unmapped`` and reported as a rate, never coerced to the nearest type.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib_common import QCOUT, definitions, normalise_label, resolve_celltypist_map, write_json  # noqa: E402

MODELS = [
    "Immune_All_Low.pkl",
    "Immune_All_High.pkl",
    "Human_Lung_Atlas.pkl",
    "Adult_Mouse_Gut.pkl",
    "Adult_Human_PancreaticIslet.pkl",
    "Fetal_Human_Pancreas.pkl",
    "Adult_Human_MTG.pkl",
    "Adult_Human_PrefrontalCortex.pkl",
    "Mouse_Whole_Brain.pkl",
    "Cells_Intestinal_Tract.pkl",
    "Healthy_Human_Liver.pkl",
]


def main() -> None:
    from celltypist import models

    defs = definitions()  # the 82 terms the panels define
    register = sorted(defs)
    norm_defs: dict[str, list[str]] = {}
    for cl_id in register:
        norm_defs.setdefault(normalise_label(defs[cl_id]["name"]), []).append(cl_id)
    payload = {}
    for model in MODELS:
        path = Path(models.models_path) / model
        if not path.exists():
            payload[model] = {"available": False}
            continue
        instance = models.Model.load(str(path))
        labels = list(instance.cell_types)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        table = {}
        for label in labels:
            key = normalise_label(label)
            exact = norm_defs.get(key)
            candidates = list(exact or [])
            table[label] = {
                "normalised": key,
                "matched_cl_ids": candidates,
                "matched_names": [defs[c]["name"] for c in candidates],
            }
        payload[model] = {
            "available": True,
            "sha256": digest,
            "bytes": path.stat().st_size,
            "n_labels": len(labels),
            "n_features": len(instance.features),
            "features_head": [str(f) for f in list(instance.features)[:5]],
            "cell_types": [str(c) for c in labels],
            "labels": table,
        }
        resolved = sum(1 for v in table.values() if v["matched_cl_ids"])
        print(f"{model}: {len(labels)} labels, {resolved} resolve to a register term by name")
    write_json(QCOUT / "celltypist_models.json", payload)
    effective = {}
    for model, info in payload.items():
        if not info.get("available"):
            print(f"{model}: NOT AVAILABLE")
            continue
        mapping = resolve_celltypist_map(model)
        effective[model] = mapping
        # Count coverage over the model's own labels only: the map also holds the curated
        # base labels (e.g. 'Pvalb') that the cluster rule expands, and counting those in
        # the numerator would overstate coverage.
        covered = [label for label in info["cell_types"] if label in mapping]
        unresolved = [label for label in info["cell_types"] if label not in mapping]
        print(f"  {model}: effective map covers {len(covered)}/{len(info['cell_types'])} labels "
              f"({len(mapping)} entries incl. cluster bases)")
        if unresolved:
            print(f"    unmapped labels (scored as 'unmapped', never coerced): {unresolved}")
    write_json(QCOUT / "celltypist_label_map_effective.json", effective)


if __name__ == "__main__":
    main()
