"""Probe: tissue composition of the census cells, per dataset, for choosing CellTypist arms."""
import json, h5py, numpy as np
from collections import Counter
from pathlib import Path
ROOT = Path("/mnt/f/E/hermes-workspace/scKevType")
cells = {}
with open(ROOT/"reports/annotation-tools/questions.jsonl") as fh:
    for line in fh:
        r = json.loads(line)
        cells.setdefault(r["dataset"], set()).add(r["cell"])
ATLAS = {
 "muraro": (ROOT/"data/raw.9p/muraro/muraro.h5ad", "_index"),
 "siletti": (ROOT/"data/raw.9p/siletti/siletti.h5ad", "CellID"),
 "tabula_muris_senis": (ROOT/"data/raw.9p/tabula_muris_senis/tabula_muris_senis_10x.h5ad", "index"),
 "tabula_sapiens": (ROOT/"data/raw.9p/tabula_sapiens/tabula_sapiens.h5ad", "_index"),
}
def col(fh, group, name):
    node = fh[group][name]
    if isinstance(node, h5py.Group):
        cats=[c.decode() if isinstance(c,bytes) else str(c) for c in node["categories"][:]]
        return [cats[c] if c>=0 else None for c in node["codes"][:]]
    return [x.decode() if isinstance(x,bytes) else str(x) for x in node[:]]
for ds,(path,idcol) in ATLAS.items():
    want = cells.get(ds, set())
    with h5py.File(path,"r") as fh:
        ids = col(fh,"obs",idcol)
        idx = {c:i for i,c in enumerate(ids)}
        rows = np.array([idx[c] for c in want if c in idx])
        keys = list(fh["obs"].keys())
        fields = [k for k in ("tissue","donor_tissue","cell_type","cell_type_ontology_term_id","free_annotation","compartment") if k in keys]
        print("="*70); print(ds, "n=", len(rows), "fields:", fields)
        for f in fields:
            vals = col(fh,"obs",f)
            c = Counter(vals[r] for r in rows)
            print(f"  {f}: {dict(c.most_common(14))}")
