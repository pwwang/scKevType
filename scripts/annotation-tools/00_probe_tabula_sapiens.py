"""Probe: can we read the tabula_sapiens atlas the way the census needs, and is it the right file?

Read-only. Prints the obs/var layout, the raw count matrix key, whether the census cell ids
resolve, and the sha256 sidecar's own value so the report can quote a digest for the file the
extraction actually read.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py

ROOT = Path("/mnt/f/E/hermes-workspace/scKevType")
ATLAS = ROOT / "data" / "raw.9p" / "tabula_sapiens" / "tabula_sapiens.h5ad"
QUESTIONS = ROOT / "reports" / "annotation-tools" / "questions.jsonl"


def main() -> int:
    want = set()
    with open(QUESTIONS, encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if row["dataset"] == "tabula_sapiens":
                want.add(row["cell"])
    print(f"census tabula_sapiens cells: {len(want)}")

    with h5py.File(ATLAS, "r") as fh:
        print("top-level keys:", list(fh.keys()))
        print("obs keys:", list(fh["obs"].keys())[:40])
        print("var keys:", list(fh["var"].keys())[:20])
        print("raw group:", "raw" in fh, list(fh["raw"].keys()) if "raw" in fh else None)
        for key in ("raw/X", "X"):
            if key.split("/")[0] in fh and key.split("/")[-1] in (fh[key.split("/")[0]] if "/" in key else fh):
                grp = fh[key]
                if isinstance(grp, h5py.Group):
                    print(f"{key}: group, keys={list(grp.keys())}, dtype={grp['data'].dtype}, shape={grp['indptr'].shape}")
        obs_index = None
        for col in ("_index", "index"):
            if col in fh["obs"]:
                obs_index = [x.decode() if isinstance(x, bytes) else str(x) for x in fh["obs"][col][:]]
                print(f"obs column used: {col}, n={len(obs_index)}, first={obs_index[:3]}")
                break
        if obs_index is None:
            print("NO cell-id column in obs")
            return 1
        hits = want & set(obs_index)
        print(f"census cells found in obs: {len(hits)} / {len(want)}")
        missing = sorted(want - set(obs_index))[:5]
        print("missing examples:", missing)

    sidecar = ATLAS.with_suffix(".h5ad.sha256")
    print("sidecar:", sidecar.read_text().strip() if sidecar.exists() else "NONE")
    print("bytes:", ATLAS.stat().st_size)
    return 0


if __name__ == "__main__":
    sys.exit(main())
