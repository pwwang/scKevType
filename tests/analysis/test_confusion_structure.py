"""The confusion-structure taxonomy, on synthetic rows, plus one slice of the real Jev dump.

The load-bearing risk in ``scripts/confusion-structure.py`` is direction: prediction =
*ancestor* of truth is a coarsening, and getting it backwards produces a confident, wrong
report. So the six classes are pinned here with explicit pairs (including the
medium-spiny-neuron/neuron case in both directions and the parent-is-not-a-sibling trap),
and the pipeline is then exercised end to end on synthetic rows -- the abstain buckets, the
``top1``-instead-of-``p`` path, ``unmappable``, and the two refusals (a row with no
prediction at all, and a population emptied by the filters).

The integration test runs the same pipeline over the first 200 lines of
``reports/jev-full/rows.jsonl`` and checks it against a count made here without the script;
the full file carries the numbers the parent session measured (n=2,510, abstained 399,
answered 2,111, wrong 488) and is asserted too because that is the reproduction the step
exists for. Both skip when the dump is absent.

No network, no GPU, no train extra: the script is stdlib-only and so is everything here.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from sckevtype.eval import abstain_key  # the one source of truth for the abstain option

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "confusion-structure.py"
JEV_ROWS = ROOT / "reports" / "jev-full" / "rows.jsonl"
JEV_REPORT = ROOT / "reports" / "jev-full" / "report.json"
CORPUS = ROOT / "runs" / "real-v2" / "test.jsonl"
CL_TERMS = ROOT / "artifacts" / "v3" / "cl_terms.json"
DEFINITIONS = ROOT / "artifacts" / "v3" / "definitions.json"


def _load():
    spec = importlib.util.spec_from_file_location("confusion_structure", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["confusion_structure"] = module
    spec.loader.exec_module(module)
    return module


cs = _load()

ONTOLOGY = cs.Ontology(cs.SELFTEST_TERMS)
REGISTER = cs.Register(
    ONTOLOGY, [{"name": t["name"], "cl_id": t["cl_id"]} for t in cs.SELFTEST_TERMS]
)


def cl(key: str) -> str | None:
    return REGISTER.resolve(key)


# --------------------------------------------------------------------------------------
# The taxonomy, and its direction
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("truth", "pred", "expected"),
    [
        # the pinned trap: neuron IS an ancestor of medium spiny neuron, not the reverse
        ("medium spiny neuron", "neuron", "coarsening"),
        ("neuron", "medium spiny neuron", "narrowing"),
        # a parent is not a sibling: granulocyte is neutrophil's direct parent
        ("neutrophil", "granulocyte", "coarsening"),
        ("neutrophil", "eosinophil", "sibling"),
        ("medium spiny neuron", "dopaminergic neuron", "cousin"),
        ("medium spiny neuron", "fibroblast", "unrelated"),
        ("pancreatic ductal cell", "epithelial cell", "coarsening"),
        ("epithelial cell", "pancreatic ductal cell", "narrowing"),
    ],
)
def test_each_class_and_both_directions(truth, pred, expected):
    klass, _ = cs.classify(cl(pred), cl(truth), ONTOLOGY, REGISTER)
    assert klass == expected


def test_cousin_names_its_deepest_common_ancestor():
    klass, lca = cs.classify(cl("dopaminergic neuron"), cl("medium spiny neuron"),
                             ONTOLOGY, REGISTER)
    assert (klass, ONTOLOGY.name_of(lca)) == ("cousin", "neuron")


def test_unrelated_when_the_only_shared_ancestor_is_cell():
    klass, lca = cs.classify(cl("fibroblast"), cl("medium spiny neuron"), ONTOLOGY, REGISTER)
    assert (klass, lca) == ("unrelated", None)
    assert "CL:0000000" in ONTOLOGY.ancestors(cl("fibroblast"))


def test_unmappable_when_a_side_has_no_cl_ancestry():
    assert cs.classify("local:siletti:cl_4023008", cl("neuron"), ONTOLOGY, REGISTER)[0] == \
        "unmappable"
    assert cs.classify(cl("neuron"), "local:siletti:cl_4023008", ONTOLOGY, REGISTER)[0] == \
        "unmappable"


def test_two_names_for_one_cl_id_are_not_an_error_class():
    register = cs.Register(ONTOLOGY, [{"name": "MSN", "cl_id": "CL:1001474"}])
    assert register.resolve("MSN") == register.resolve("medium spiny neuron") == "CL:1001474"
    assert cs.classify("CL:1001474", "CL:1001474", ONTOLOGY, register)[0] == "same"


# --------------------------------------------------------------------------------------
# The pipeline, end to end on synthetic rows
# --------------------------------------------------------------------------------------


def write_inputs(tmp_path: Path, corpus: dict[tuple[str, str], str]) -> list[str]:
    """Synthetic cl_terms, register and corpus: enough to run the real CLI."""
    terms, definitions = [], []
    for term in cs.SELFTEST_TERMS:
        terms.append(term)
        definitions.append({"name": term["name"], "cl_id": term["cl_id"]})
    cl_terms = tmp_path / "cl_terms.json"
    cl_terms.write_text(json.dumps({"terms": terms}))
    defs = tmp_path / "definitions.json"
    defs.write_text(json.dumps({"definitions": definitions}))
    corpus_path = tmp_path / "corpus.jsonl"
    with open(corpus_path, "w") as f:
        f.writelines(json.dumps({"_meta": {"dataset_id": dataset, "cell_id": cell,
                                           "cl_id": cl_id}}) + "\n"
                     for (dataset, cell), cl_id in corpus.items())
    return ["--corpus", str(corpus_path), "--cl-terms", str(cl_terms),
            "--definitions", str(defs)]


def write_rows(tmp_path: Path, rows: list[dict], name: str = "rows.jsonl") -> str:
    path = tmp_path / name
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return str(path)


def row(keys, label, p=None, top1=None, family="panel_choice", dataset="ds", cell="c1",
        split="test", type="choice", **extra):
    out = {"family": family, "type": type, "split": split, "dataset": dataset, "cell": cell,
           "keys": keys, "label": label}
    if p is not None:
        out["p"] = p
    if top1 is not None:
        out["top1"] = top1
    out.update(extra)
    return out


def run(tmp_path: Path, rows: list[dict], corpus=None, extra_args=(), name="rows.jsonl"):
    out = str(tmp_path / ("out-" + name + ".json"))
    args = ["--engine", "synthetic", "--rows", write_rows(tmp_path, rows, name),
            *write_inputs(tmp_path, corpus or {}), "--split", "any", "--out", out,
            *extra_args]
    assert cs.main(args) == 0
    return json.loads(Path(out).read_text())


def test_the_three_buckets_and_the_accuracy(tmp_path):
    report = run(tmp_path, [
        # correct: medium spiny neuron is option 0 and has the mass
        row(["medium spiny neuron", "neuron"], label=0, p=[0.7, 0.3]),
        # abstained: the abstain option wins
        row(["medium spiny neuron", "other"], label=0, p=[0.2, 0.8]),
        # answered and wrong: predicted the ancestor
        row(["medium spiny neuron", "neuron"], label=0, p=[0.3, 0.7]),
    ])
    stats = report["families"]["panel_choice"]
    assert (stats["n"], stats["abstained"], stats["answered"], stats["correct"]) == (3, 1, 2, 1)
    assert stats["accuracy"] == pytest.approx(1 / 3)
    assert stats["errors"] == 1
    assert stats["taxonomy"]["coarsening"]["n"] == 1
    assert stats["taxonomy"]["coarsening"]["share"] == 1.0


def test_truth_is_abstain_is_a_fallback_not_a_taxonomy_error(tmp_path):
    report = run(tmp_path, [
        # the true type is not on the panel: the labelled answer is the abstain option
        row(["other", "neuron"], label=0, p=[0.1, 0.9], family="panel_miss"),
    ], corpus={("ds", "c1"): "CL:1001474"})
    stats = report["families"]["panel_miss"]
    assert (stats["truth_is_abstain"], stats["answered"], stats["errors"]) == (1, 1, 0)
    fallback = report["starved_fallback"]
    assert fallback["rows"] == 1
    assert fallback["top_pairs"][0] == {
        "named": "neuron", "true_type": "medium spiny neuron", "count": 1,
        "class": "coarsening",
    }


def test_a_model_that_abstains_on_a_starved_row_is_not_a_fallback(tmp_path):
    report = run(tmp_path, [row(["other", "neuron"], label=0, p=[0.9, 0.1],
                                family="panel_miss")])
    assert report["families"]["panel_miss"]["abstained"] == 1
    assert report["starved_fallback"]["rows"] == 0


def test_top1_is_used_when_p_is_absent(tmp_path):
    report = run(tmp_path, [row(["medium spiny neuron", "neuron"], label=0, top1=1)])
    assert report["provenance"]["prediction_field"] == {"top1": 1}
    assert report["families"]["panel_choice"]["errors"] == 1


def test_a_row_with_neither_p_nor_top1_refuses_to_be_scored(tmp_path):
    with pytest.raises(SystemExit, match="neither 'p' nor 'top1'"):
        run(tmp_path, [row(["medium spiny neuron", "neuron"], label=0)])


def test_an_empty_population_is_refused(tmp_path):
    with pytest.raises(SystemExit, match="no rows survived the filters"):
        run(tmp_path, [row(["medium spiny neuron", "neuron"], label=0, p=[1.0, 0.0],
                           split="calibration")], extra_args=("--split", "test"))


def test_unmappable_is_counted_not_dropped(tmp_path):
    report = run(tmp_path, [row(["medium spiny neuron", "local:siletti:cl_4023008"], label=0,
                                p=[0.2, 0.8])])
    stats = report["families"]["panel_choice"]
    assert stats["errors"] == 1
    assert stats["taxonomy"]["unmappable"]["n"] == 1
    assert report["provenance"]["pred_unresolved"] == 1


def test_row_filter_keeps_only_the_asked_variant(tmp_path):
    report = run(tmp_path, [
        row(["medium spiny neuron", "neuron"], label=0, p=[0.9, 0.1], ask="base"),
        row(["medium spiny neuron", "neuron"], label=0, p=[0.9, 0.1], ask="abstain"),
    ], extra_args=("--row-filter", "ask=base"))
    assert report["families"]["panel_choice"]["n"] == 1
    assert report["provenance"]["counts"]["row_filter"] == 1


# --------------------------------------------------------------------------------------
# --ids-from: the census subset
# --------------------------------------------------------------------------------------


def write_ids(tmp_path: Path, ids: list[str], name: str = "ids.jsonl") -> str:
    """A row dump whose rows carry nothing but the ``id`` the filter matches."""
    path = tmp_path / name
    with open(path, "w") as f:
        f.writelines(json.dumps({"id": i}) + "\n" for i in ids)
    return str(path)


def test_ids_from_keeps_only_the_named_rows(tmp_path):
    report = run(tmp_path, [
        row(["medium spiny neuron", "neuron"], label=0, p=[0.9, 0.1], id="test/ds/c1/panel_choice"),
        row(["neutrophil", "granulocyte"], label=0, p=[0.3, 0.7], id="test/ds/c2/panel_choice"),
    ], extra_args=("--ids-from", write_ids(tmp_path, ["test/ds/c1/panel_choice"])))
    assert report["population"]["rows"] == 1
    prov = report["provenance"]
    # kept and dropped are both in the provenance block, and the kept row is the correct one --
    # so the filter moved the population, not just the denominator
    assert (prov["ids_from_kept"], prov["ids_from_dropped"], prov["ids_from_names"]) == (1, 1, 1)
    assert prov["counts"]["ids_from_dropped"] == 1
    assert report["families"]["panel_choice"]["accuracy"] == 1.0


def test_ids_from_matches_the_census_key_not_only_the_id(tmp_path):
    """Two dumps of the same row do not have to spell its ``id`` the same way.

    The census dump names a row ``test/<dataset>/<cell>/<family>``; ``sckevtype.eval.run --rows-full``
    names the same row ``test/custom/<n>``. Matching on ``id`` alone kept 0 of 24,945 rows, so the
    filter falls back to the ``(dataset, cell, family)`` triple that both files carry.
    """
    ids = tmp_path / "census-ids.jsonl"
    ids.write_text(json.dumps({"id": "test/muraro/D28.1_1/panel_choice", "dataset": "muraro",
                               "cell": "D28.1_1", "family": "panel_choice"}) + "\n")
    report = run(tmp_path, [
        row(["medium spiny neuron", "neuron"], label=0, p=[0.9, 0.1], id="test/custom/7",
            dataset="muraro", cell="D28.1_1", family="panel_choice"),
        row(["neutrophil", "granulocyte"], label=0, p=[0.3, 0.7], id="test/custom/8",
            dataset="muraro", cell="D28.1_2", family="panel_choice"),
    ], extra_args=("--ids-from", str(ids)))
    assert report["population"]["rows"] == 1
    assert report["provenance"]["counts"]["ids_from_dropped"] == 1


def test_ids_from_with_no_intersection_is_refused(tmp_path):
    with pytest.raises(SystemExit, match="--ids-from .* kept no scored row"):
        run(tmp_path, [row(["medium spiny neuron", "neuron"], label=0, p=[0.9, 0.1], id="a")],
            extra_args=("--ids-from", write_ids(tmp_path, ["b"])))


def test_an_ids_file_with_no_ids_is_refused(tmp_path):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    with pytest.raises(SystemExit, match="no ids read"):
        run(tmp_path, [row(["medium spiny neuron", "neuron"], label=0, p=[0.9, 0.1], id="a")],
            extra_args=("--ids-from", str(empty)))


def test_ids_from_absent_changes_nothing(tmp_path):
    report = run(tmp_path, [row(["medium spiny neuron", "neuron"], label=0, p=[0.9, 0.1], id="a")])
    assert report["provenance"]["ids_from"] is None
    assert "ids_from_dropped" not in report["provenance"]["counts"]


def test_noul_rows_are_dropped_and_the_family_stays_visible(tmp_path):
    report = run(tmp_path, [
        row(["true", "false"], label=0, p=[1.0, 0.0], family="marker_noul", type="noul"),
        row(["medium spiny neuron", "neuron"], label=0, p=[0.9, 0.1]),
    ])
    assert report["provenance"]["counts"]["type_not_choice"] == 1
    assert report["families"]["marker_noul"]["n"] == 0
    assert report["population"]["rows"] == 1


def test_a_choice_only_dump_without_a_type_field_is_kept(tmp_path):
    """`sckevtype.eval.run --rows-full` writes choice rows and omits `type`.

    The dump is choice-only by construction, so a missing `type` has to be inferred from the row's own
    shape. Treating it as "not choice" dropped a whole 27,079-row population, and the run refused with
    "no rows survived the filters" -- a refusal that is correct but points at the wrong thing.
    """
    report = run(tmp_path, [
        row(["medium spiny neuron", "neuron"], label=0, p=[0.9, 0.1], type=None),
        row(["true", "false"], label=0, p=[1.0, 0.0], family="marker_noul", type="noul"),
    ])
    assert report["provenance"]["counts"]["type_not_choice"] == 1
    assert report["population"]["rows"] == 1
    assert report["families"]["panel_choice"]["n"] == 1


def test_the_selftest_passes_and_says_so(capsys):
    assert cs.main(["--selftest"]) == 0
    out = capsys.readouterr().out
    assert "selftest: PASS" in out
    assert "negative-control" in out
    assert "FAIL" not in out


# --------------------------------------------------------------------------------------
# The real Jev dump
# --------------------------------------------------------------------------------------

pytestmark_jev = pytest.mark.skipif(not JEV_ROWS.exists(), reason="Jev row dump not present")


def naive_panel_choice_counts(rows) -> dict:
    """The panel_choice bucket counts, straight from the rows, with no taxonomy."""
    n = abstained = correct = 0
    for r in rows:
        if r["family"] != "panel_choice" or r["type"] != "choice" or r["split"] != "test":
            continue
        n += 1
        pred = max(range(len(r["p"])), key=lambda i: r["p"][i])
        if r["keys"][pred] == r["keys"][r["label"]]:
            correct += 1
        option = abstain_key(r["keys"])
        if option is not None and r["keys"][pred] == option:
            abstained += 1
    return {"n": n, "abstained": abstained, "correct": correct,
            "answered": n - abstained, "wrong": n - abstained - correct}


@pytestmark_jev
def test_a_200_line_slice_of_the_jev_dump(tmp_path):
    """The pipeline on the first 200 lines agrees with a count made here."""
    slice_path = tmp_path / "slice.jsonl"
    with open(JEV_ROWS) as src, open(slice_path, "w") as dst:
        for i, line in enumerate(src):
            if i == 200:
                break
            dst.write(line)
    rows = [json.loads(line) for line in slice_path.read_text().splitlines()]
    got = run(tmp_path, rows, name="slice.jsonl")["families"]["panel_choice"]
    expected = naive_panel_choice_counts(rows)
    assert expected["n"] == 11  # 200 lines of this dump: 11 choices per family + noul rows
    assert (got["n"], got["abstained"], got["answered"]) == (
        expected["n"], expected["abstained"], expected["answered"])
    # every wrong answer is either classified or counted as two names for one CL id
    assert got["errors"] + got["same_id"] == expected["wrong"]


@pytestmark_jev
def test_the_full_jev_dump_reproduces_the_measured_panel_choice_numbers(tmp_path):
    """The numbers the parent session measured, reproduced from ``p`` and ``label``."""
    args = ["--engine", "jev", "--rows", str(JEV_ROWS),
            "--corpus", str(CORPUS), "--cl-terms", str(CL_TERMS),
            "--definitions", str(DEFINITIONS),
            "--check-report", str(JEV_REPORT),
            "--out", str(tmp_path / "jev.json")]
    assert cs.main(args) == 0
    report = json.loads((tmp_path / "jev.json").read_text())
    choice = report["families"]["panel_choice"]
    expected = naive_panel_choice_counts(
        json.loads(line) for line in JEV_ROWS.open() if line.strip()
    )
    # the parent's numbers: n=2,510, abstained 399 (15.9%), answered 2,111, wrong 488
    assert expected["n"] == 2510
    assert expected["abstained"] == 399
    assert (choice["n"], choice["abstained"], choice["answered"], choice["errors"]) == (
        2510, 399, 2111, 488)
    # and the dominant errors are the four measured coarsenings
    top = {(p["truth"], p["predicted"]): (p["count"], p["class"]) for p in choice["top_pairs"]}
    assert top[("medium spiny neuron", "neuron")] == (70, "coarsening")
    assert top[("pancreatic ductal cell", "epithelial cell")] == (43, "coarsening")
    assert top[("neutrophil", "granulocyte")] == (33, "coarsening")
    assert top[("CD4-positive, alpha-beta T cell", "T cell")] == (28, "coarsening")
    # coarsening is the largest class among this family's errors
    shares = {k: v["share"] for k, v in choice["taxonomy"].items()}
    assert shares["coarsening"] == max(shares.values())
    assert report["crosscheck"]["pass"]
