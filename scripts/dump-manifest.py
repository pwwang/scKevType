import json

with open("runs/real-v2/manifest.json") as fh:
    m = json.load(fh)
c = m["counts"]
print("=== TOTALS ===")
print("  records   {:>10,}".format(c["records"]))
print("  questions {:>10,}".format(c["questions"]))
print("  cells     {:>10,}".format(c["cells"]))
print()
print("=== PER DATASET ===")
for ds, d in sorted(c["per_dataset"].items(), key=lambda kv: -kv[1]["records"]):
    print("  {:<24} {:>9,}".format(ds, d["records"]))
print()
print("=== PER FAMILY ===")
for fam, n in sorted(c["per_family"].items(), key=lambda kv: -kv[1]):
    print(f"  {fam:<24} {n:>9,}")
print()
print("=== PER SPLIT ===")
for s, n in sorted(c["per_split"].items(), key=lambda kv: -kv[1]):
    print(f"  {s:<24} {n:>9,}")
print()
print("=== files recorded in the manifest ===")
files = m.get("files", {})
if isinstance(files, dict):
    items = files.items()
else:
    items = [(str(i), f) for i, f in enumerate(files)]
for f, h in sorted(items):
    print(f"  {f:<28} {json.dumps(h)[:110]}")
print()
print("=== artifacts / config ===")
print("  artifacts:", json.dumps(m.get("artifacts", {}))[:280])
print("  seed:", m.get("seed"), "| version:", m.get("version"), "| name:", m.get("name"))
print()
print("=== one sample record (keys only) ===")
with open("runs/real-v2/test.jsonl") as fh:
    r = json.loads(fh.readline())
print("  ", sorted(r))
