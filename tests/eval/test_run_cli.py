"""The CLI end to end — the one test here that needs the train extra and the smoke checkpoint.

Marked ``slow`` so a CPU-only box skips it; the acceptance command runs it in the foreground with
``--run-slow``. What it checks is the wiring the pure modules cannot: that a suite and a checkpoint
in fact produce rows, that every fit is handed calibration rows, and that the report is a
deterministic function of its inputs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sckevtype.eval import metrics, run

ROOT = Path(__file__).resolve().parents[2]
SUITE, CHECKPOINT = ROOT / "artifacts" / "suites" / "smoke", ROOT / "runs" / "smoke-0.8b"


@pytest.mark.slow
def test_cli_scores_a_suite_and_writes_a_report(tmp_path):
    """Two runs of the same command must produce byte-identical reports."""
    if not (SUITE / "manifest.json").exists():
        pytest.skip(f"no suite at {SUITE}")
    if not (CHECKPOINT / "adapter_model.safetensors").exists():
        pytest.skip(f"no checkpoint at {CHECKPOINT}")
    pytest.importorskip("kev.suite", reason="requires the train extra")

    outputs = []
    for index in range(2):
        out = tmp_path / f"report-{index}.json"
        run.main(["--suite", str(SUITE), "--checkpoint", str(CHECKPOINT), "--alpha", "0.10",
                  "--out", str(out), "--rows-out", str(tmp_path / f"rows-{index}.jsonl"),
                  "--device", "cpu"])
        outputs.append(json.loads(out.read_text()))

    first, second = outputs
    assert first["partitions"]["fitted_on"] == "calibration"
    assert first["partitions"]["reported_on"] == "test"
    assert first["partitions"]["rows"]["test"]["test"] > 0
    assert first["plumbing_check"]["is_result"] is False        # a 32-cell synthetic fixture corpus
    assert first["plumbing_check"]["reasons"]
    assert first["axis2_conformal"]["by_method"]["lac"]["test"]["n"] > 0
    assert first["triage"]["n"] > 0
    assert first == second, "the report is not a deterministic function of its inputs"

    # The dump has to stand on its own: a reader with this file and no GPU must be able to re-derive
    # the axis-1 statistic, which means every row states the label and the serving convention.
    dump = [json.loads(line) for line in (tmp_path / "rows-0.jsonl").read_text().splitlines()]
    assert dump, "no rows dumped"
    assert all(row["abstain_source"] in (metrics.FACTORY, metrics.INJECTED) for row in dump)
    assert all(row["positive"] == int(metrics.axis1_label(row)) for row in dump)
    rederived = metrics.panel_adequacy(dump)
    assert rederived["auroc"] == first["axis1_panel_adequacy"]["auroc"]
    assert rederived["by_serving_convention"] == first["axis1_panel_adequacy"]["by_serving_convention"]
