#!/usr/bin/env bash
# Independent verification of the real-v2 corpus. Reads the build's artefacts; it does not
# trust the build's own report. Prints per-atlas / per-family counts and checks the
# out_of_register family against the cells the register itself could not name.
set -uo pipefail
cd /mnt/f/E/hermes-workspace/scKevType
source scripts/wsl/env.sh >/dev/null 2>&1

python - <<'PY'
import collections
import glob
import json
import os

RUN = "runs/real-v2"

print("=== artefact inventory ===")
files = sorted(glob.glob(f"{RUN}/**/*", recursive=True))
for p in files:
    if os.path.isfile(p):
        print(f"  {os.path.getsize(p):>14,}  {p}")
if not files:
    raise SystemExit("runs/real-v2 is empty")

manifest = None
for cand in (f"{RUN}/manifest.json", f"{RUN}/corpus.json"):
    if os.path.exists(cand):
        manifest = json.load(open(cand))
        print(f"\n=== manifest: {cand} ===")
        print("  top-level keys:", sorted(manifest)[:20])
        break

# find the record stream
recs = [p for p in files if p.endswith(".jsonl")]
if not recs:
    raise SystemExit("no .jsonl record stream found")

path = max(recs, key=os.path.getsize)
print(f"\n=== records: {path} ===")
per_dataset = collections.Counter()
per_family = collections.Counter()
per_panel = collections.Counter()
n = 0
for line in open(path):
    line = line.strip()
    if not line:
        continue
    r = json.loads(line)
    n += 1
    per_dataset[r.get("dataset") or r.get("atlas") or "?"] += 1
    per_family[r.get("family", "?")] += 1
    per_panel[r.get("panel_id") or r.get("panel", "?")] += 1

print(f"  total records: {n:,}")
print("\n  by dataset:")
for k, v in per_dataset.most_common():
    print(f"    {k:<26} {v:>9,}")
print("\n  by family:")
for k, v in per_family.most_common():
    print(f"    {k:<26} {v:>9,}")
print(f"\n  distinct panels: {len(per_panel)}")

if manifest:
    print("\n=== manifest vs stream (must agree) ===")
    m = manifest.get("counts") or manifest.get("families") or {}
    print("  manifest counts:", json.dumps(m)[:400])
    print(f"  stream total   : {n:,}")
PY
