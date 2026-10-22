"""Step 1 -- the evaluation unit, read off the corpus itself.

The corpus's records are (cell, family, variant) rows: every family carries a
``clean`` and a ``permuted`` variant of the *same* cell with the *same* target, so
the record count is not the number of cells and a baseline that predicts one label
per cell would otherwise be scored twice on it. This step collapses the corpus to

  * ``out/cells_{split}.json``  -- dataset -> cell_id -> {cl_id, donor, families},
  * ``out/records_{split}.jsonl`` -- one line per (cell, family), the target and the
    option list that family asked about,

so every later step scores cells once and families once, and never a duplicate
variant.

Every field comes from the corpus; nothing is inferred. Two things it *does* record
rather than assume:

  * ``cl_id`` starting with ``local:`` is a cell whose type is not in the CL
    register (the ``out_of_register`` family, ``_meta.reason == "unregistered_type"``).
    Such a cell has no CL id a CL-trained baseline could ever emit, so it is kept in
    the inventory and *flagged*, never scored as if it were a missing label.
  * a family whose target option is the abstain option (``other``) is recorded with
    ``target_kind = "abstain"``; a binary ``noul`` question with its bool target.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib_common import CORPUS, QCOUT, definitions, write_json  # noqa: E402

ABSTAIN_NAMES = {"other"}


def main() -> None:
    defs = definitions()
    cl_to_name = {c: d["name"] for c, d in defs.items()}
    # Measured unambiguous: 82 definitions carry 82 distinct names, so a name is a key.
    name_to_cl = {d["name"]: c for c, d in defs.items()}
    summary = {}
    for split in ("train", "test", "development", "calibration"):
        path = CORPUS / f"{split}.jsonl"
        if not path.exists():
            continue
        cells: dict[str, dict] = {}
        records: list[dict] = []
        fam_cells: dict[str, Counter] = defaultdict(Counter)
        fam_records = Counter()
        fam_variants = Counter()
        local_labels = Counter()
        qid_kinds = Counter()
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                rec = json.loads(line)
                meta = rec["_meta"]
                dataset, cell_id, family = meta["dataset_id"], meta["cell_id"], meta["family"]
                cl_id = meta["cl_id"]
                cell = cells.setdefault(
                    f"{dataset}/{cell_id}",
                    {
                        "dataset_id": dataset,
                        "cell_id": cell_id,
                        "cl_id": cl_id,
                        "cl_name": cl_to_name.get(cl_id),
                        "registered": not str(cl_id).startswith("local:"),
                        "donor_id": meta.get("donor_id"),
                        "panel_scope": meta.get("panel_scope"),
                        "families": [],
                    },
                )
                # One (cell, family) row per family: the clean/permuted variants are the
                # same question about the same cell, so only the first is kept as the
                # family's record and the variant is counted.
                variant = meta.get("variant")
                fam_variants[(family, variant)] += 1
                entry = None
                for qid, spec in rec["questions"].items():
                    qid_kinds[(family, "holds:CL:*" if qid.startswith("holds:") else qid)] += 1
                    target = spec["label"]
                    criteria = list(spec.get("criteria", {}))
                    if spec["type"] == "noul":
                        # holds:<CL id> -- the candidate is the question id, not an option list.
                        candidate_cl = qid.split(":", 1)[1]
                        entry = {
                            "qid": qid,
                            "kind": "noul",
                            "target": bool(target),
                            "candidate_cl_id": candidate_cl,
                            "candidate_name": cl_to_name.get(candidate_cl),
                            "option_names": [],
                            "options_cl_ids": [candidate_cl],
                        }
                    else:
                        abstain = target in ABSTAIN_NAMES
                        entry = {
                            "qid": qid,
                            "kind": "choice",
                            "target": target,
                            "target_cl_id": None if abstain else name_to_cl.get(target),
                            "target_kind": "abstain" if abstain else "type",
                            "option_names": criteria,
                            "options_cl_ids": [name_to_cl[n] for n in criteria if n in name_to_cl],
                            "n_options": len(criteria),
                            "n_options_cl_mapped": sum(1 for n in criteria if n in name_to_cl),
                        }
                    break  # one question per record in every family this corpus builds
                if entry is None:
                    continue
                if variant in (None, "clean") or family not in cell["families"]:
                    cell["families"].append(family)
                    records.append(
                        {
                            "dataset_id": dataset,
                            "cell_id": cell_id,
                            "family": family,
                            "variant": variant,
                            "cl_id": cl_id,
                            "registered": not str(cl_id).startswith("local:"),
                            "q": entry,
                        }
                    )
                    fam_cells[family][dataset] += 1
                    fam_records[family] += 1
                if str(cl_id).startswith("local:"):
                    local_labels[cl_id] += 1

        summary[split] = {
            "cells": len(cells),
            "cells_registered": sum(1 for c in cells.values() if c["registered"]),
            "cells_local_label": sum(1 for c in cells.values() if not c["registered"]),
            "records_kept": len(records),
            "records_in_file": sum(fam_variants.values()),
            "per_family_records": dict(sorted(fam_records.items())),
            "per_family_cells": {f: dict(sorted(c.items())) for f, c in sorted(fam_cells.items())},
            "variants_seen": {f"{f}/{v}": n for (f, v), n in sorted(fam_variants.items())},
            "qid_kinds": {f"{f}/{q}": n for (f, q), n in sorted(qid_kinds.items())},
            "local_label_ids": dict(sorted(local_labels.items())),
        }
        write_json(QCOUT / f"cells_{split}.json", cells)
        with (QCOUT / f"records_{split}.jsonl").open("w", encoding="utf-8") as out:
            for row in records:
                out.write(json.dumps(row, sort_keys=False) + "\n")
        print(f"[{split}] cells={len(cells)} records_kept={len(records)} file_records={sum(fam_variants.values())}")
        for family, n in sorted(fam_records.items()):
            print(f"    {family:18s} records={n:7d} cells={sum(fam_cells[family].values()):7d}")
    write_json(QCOUT / "step1_summary.json", summary)


if __name__ == "__main__":
    main()
