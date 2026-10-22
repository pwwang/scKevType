"""Is the census a superset of the published 200-cell probe, and does it reproduce the probe's number?

Two runs of the same Jev probe, asked through different paths: the published 200-cell probe sent one
request per ask (reports/jev-smoke), the census sent one request per (cell, state group)
(reports/jev-full, 2,510 cells). If the census's rows restricted to the probe's 200 cells land near
the probe's published 0.8277, the census measured the same quantity on a superset of the same
population; if they do not, one of the two runs is not what it says it is.

Reads dumps and JSON on disk: no network, no key, no spend.
"""

from __future__ import annotations

import json
from pathlib import Path

from sckevtype.eval import metrics

ROOT = Path(__file__).resolve().parent.parent
census_rows = [json.loads(line) for line in (ROOT / "reports/jev-full/axis1-rows.jsonl").open()]
probe_sample = json.loads((ROOT / "reports/jev-smoke/sample.json").read_text())
probe_report = json.loads((ROOT / "reports/jev-smoke/report.json").read_text())
census_report = json.loads((ROOT / "reports/jev-full/report.json").read_text())

probe_cells = set(probe_sample["chosen"])
census_cells = {f"{row['dataset']}/{row['id'].split('/')[2]}" for row in census_rows}
print(f"probe cells {len(probe_cells)}  census cells {len(census_cells)}  "
      f"probe cells missing from the census: {len(probe_cells - census_cells)}")

subset = [row for row in census_rows
          if f"{row['dataset']}/{row['id'].split('/')[2]}" in probe_cells]
adequacy = metrics.panel_adequacy(subset, 0.05)
primary = adequacy["primary"]
print(f"census rows restricted to the probe's cells: {len(subset)}  "
      f"primary auroc {primary['auroc']:.4f} n={primary['n']} pos={primary['n_positive']} neg={primary['n_negative']}")
p = probe_report["axis1_panel_adequacy"]
c = census_report["axis1_panel_adequacy"]
print(f"published probe  primary auroc {p['primary']['auroc']:.4f} n={p['primary']['n']} "
      f"pos={p['primary']['n_positive']} neg={p['primary']['n_negative']}")
print(f"census (2,510)   primary auroc {c['primary']['auroc']:.4f} n={c['primary']['n']} "
      f"pos={c['primary']['n_positive']} neg={c['primary']['n_negative']}")
