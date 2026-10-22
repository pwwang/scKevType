#!/usr/bin/env python3
"""What can be proved about the Kev probe without the cluster, the weights or a GPU.

Four things, in the order they would embarrass a run that got them wrong:

1. **The formatter.** A synthetic model output — the exact dict shape :class:`KevPredictor` returns —
   is put through the *real* :func:`sckevtype.jev.probe.rows_of_answer` and
   :func:`sckevtype.eval.run.write_row_dump`, and the dump is then read back by
   ``scripts/axis1-recompute.py``'s own loader (:func:`as_served` / :func:`verify`) and scored by
   :func:`sckevtype.eval.metrics.panel_adequacy`. The row schema is asserted to be
   ``DUMP_FIELDS + ('positive',)`` exactly, and the dump's ``positive`` column to agree with
   :func:`sckevtype.eval.metrics.axis1_label`.
2. **The encoder.** ``scripts/kev/kev_probe.py``'s body builder is compared against
   :func:`sckevtype.jev.request.record_request` on real corpus records: the two bodies must be equal
   field for field apart from ``model``. This is the claim that Kev is asked the same question Jev was.
3. **The syntax** of the harness and the two shell scripts, plus a parse of the Slurm header
   (every directive present, and the whole block ahead of the first executable line — Slurm silently
   ignores a directive that comes late, which is a job that runs with the wrong resources).
4. **The scratch-dir honesty check**: the harness writes nothing outside the directories the project
   allows (``runs/kev-probe``, ``reports/``).

Run: ``uv run --no-sync python scripts/kev/selftest.py``   (no network, no torch, no weights)
"""

from __future__ import annotations

import ast
import importlib.util
import json
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "kev"))

import kev_probe as K                                               # noqa: E402
from sckevtype.eval import abstain_key, metrics                     # noqa: E402
from sckevtype.eval.run import DUMP_FIELDS, write_row_dump          # noqa: E402
from sckevtype.jev import probe as P                                # noqa: E402
from sckevtype.jev.request import DEFAULT_MODEL as JEV_MODEL        # noqa: E402
from sckevtype.jev.request import record_request                    # noqa: E402

PASSES: list[str] = []
FAILURES: list[str] = []


def check(name: str, ok: bool, detail: object = "") -> bool:
    entry = f"{name}{' -- ' + str(detail) if detail else ''}"
    (PASSES if ok else FAILURES).append(entry)
    print(f"  {'ok  ' if ok else 'FAIL'} {name}" + (f"  [{detail}]" if detail else ""))
    return ok


# --------------------------------------------------------------------------- 1. the formatter

class FakeKev:
    """A Kev predictor's *interface* with deterministic numbers — no torch, no weights, no GPU.

    It answers every question over that question's own option keys and puts extra mass on the abstain
    option for the records this test declares positive, so the axis-1 statistic comes out as a real,
    perfectly separating number instead of a constant.
    """

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, record, questions=None):
        from kev.api import question_keys

        self.calls += 1
        active = record["questions"] if questions is None else questions
        positive = bool(record.get("_test_positive"))
        probabilities, answers = {}, {}
        for qid, question in active.items():
            keys = question_keys(question["type"], question.get("criteria"))
            raw: list[float] = [float((index + 1) ** 2) for index in range(len(keys))]
            if positive:
                key = abstain_key(keys)
                raw[keys.index(key) if key is not None else -1] += 30.0
            total = float(sum(raw))
            distribution = {key: value / total for key, value in zip(keys, raw)}
            probabilities[qid] = distribution
            if question["type"] == "choice":
                answers[qid] = {"type": "choice", "choice": max(distribution, key=distribution.get),
                                "confidence": 0.5,
                                "probabilities": {k: round(v, 4) for k, v in distribution.items()}}
            else:
                answers[qid] = {"type": "noul", "noul": round(distribution["true"], 4)}
        return {"probabilities": probabilities, "answers": answers,
                "served_model": "fake-kev (selftest)", "latency_ms": 1.0, "input_tokens": 0}


#: The factory's own abstain key for the synthetic records. Read from the factory's vocabulary rather
#: than invented, and the *injected* key is never written as a literal at all: it is read off the variant
#: :func:`sckevtype.eval.run.with_abstain` builds, because that function is the only source of truth for
#: what the second ask adds and where it sits.
FACTORY_ABSTAIN = ("unknown", "Cannot be determined from the options given")


def synthetic_records() -> list[tuple[str, str, dict]]:
    """Records that carry both axis-1 classes *within* a family (the primary statistic's selection), both
    serving conventions, and the noul path.

    Every ``label`` is taken from the question's own option keys — ``list(criteria)[...]`` — never written
    as a literal. ``sckevtype.jev.probe.rows_of_answer`` scores a choice row with
    ``keys.index(question["label"])``, so a fixture whose label is a plausible-looking string that is not
    one of its own options is a fixture that raises in the row builder rather than a fixture that models
    anything. (Learned the hard way: an invented ``"other"`` label crashed this test.)
    """
    instruction = "Which candidate does the state support?"
    panel = {"acinar cell": "An acinar cell.", "alpha cell": "An alpha cell."}
    records = []
    for index in range(6):          # evidence_diluted: factory-given abstain, 3 positives / 3 negatives
        positive = index < 3
        records.append((f"synth/diluted-{index}", "evidence_diluted", {
            "_test_positive": positive, "state": f"GCG 1 | TTR 2 | INS {index}",
            "questions": {"label": {"type": "choice", "instructions": instruction,
                                    "criteria": {**panel, FACTORY_ABSTAIN[0]: FACTORY_ABSTAIN[1]},
                                    "src": "evidence_diluted",
                                    "label": FACTORY_ABSTAIN[0] if positive else list(panel)[0]}},
            "_meta": {"group_id": f"synth/diluted-{index}", "family": "evidence_diluted",
                      "cell_id": f"diluted-{index}", "dataset_id": "synth", "donor_id": "D1",
                      "panel_id": "p", "variant": "diluted", "source": "selftest"}}))
    for index in range(2):          # no factory abstain -> the injected second ask supplies none_p, and the
        records.append((f"synth/miss-{index}", "panel_miss", {       # label is one of the real options
            "_test_positive": False, "state": f"GCG 1 | TTR 2 | INS {index}",
            "questions": {"label": {"type": "choice", "instructions": instruction, "criteria": dict(panel),
                                    "src": "panel_miss", "label": list(panel)[index]}},
            "_meta": {"group_id": f"synth/miss-{index}", "family": "panel_miss", "cell_id": f"miss-{index}",
                      "dataset_id": "synth", "donor_id": "D1", "panel_id": "p", "variant": "permuted",
                      "source": "selftest"}}))
    for index in range(2):          # panel_choice: factory-given abstain, negative
        records.append((f"synth/choice-{index}", "panel_choice", {
            "_test_positive": False, "state": f"GCG 1 | TTR 2 | INS {index}",
            "questions": {"label": {"type": "choice", "instructions": instruction,
                                    "criteria": {**panel, FACTORY_ABSTAIN[0]: FACTORY_ABSTAIN[1]},
                                    "src": "panel_choice", "label": list(panel)[index]}},
            "_meta": {"group_id": f"synth/choice-{index}", "family": "panel_choice",
                      "cell_id": f"choice-{index}", "dataset_id": "synth", "donor_id": "D1",
                      "panel_id": "p", "variant": "clean", "source": "selftest"}}))
    records.append(("synth/noul", "marker_noul", {        # the noul path: two questions in one record
        "_test_positive": False, "state": "GCG 1 | TTR 2",
        "questions": {f"holds:CL:000{i}": {"type": "noul",
                                           "instructions": {"question": "does the definition hold?"},
                                           "criteria": {"true": "yes, it does", "false": "no, it does not"},
                                           "src": "marker_noul", "label": i == 1} for i in range(2)},
        "_meta": {"group_id": "synth/noul", "family": "marker_noul", "cell_id": "noul", "dataset_id": "synth",
                  "donor_id": "D1", "panel_id": "p", "variant": "clean", "source": "selftest"}}))
    return records


def formatter_checks(work: Path) -> None:
    print("\n1. formatter: synthetic model output -> rows -> axis-1 dump -> the scorer's own reader")
    raw = work / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    # The fixture is only trustworthy if its labels are the corpus's own shape. Stated as a check rather
    # than as care taken while writing it: this is the failure that already happened once.
    from sckevtype.jev.request import option_keys

    fixture_violations = [violation for _, _, record in synthetic_records()
                          for violation in K.label_violations(record)]
    check("every synthetic label is one of its own option keys", not fixture_violations,
          f"{fixture_violations[:2]}")
    direct = [(f"{group}/{qid}", question["label"], list(option_keys(question))[:3])
              for group, _, record in synthetic_records() for qid, question in record["questions"].items()
              if question["type"] == "choice" and question["label"] not in option_keys(question)]
    check("...checked directly against sckevtype.jev.request.option_keys, not only via the guard",
          not direct, f"{direct[:2]}")
    raised = []
    for _, _, record in synthetic_records():
        try:
            K.check_labels(record)
        except P.ProbeError as error:
            raised.append(str(error)[:80])
    check("the harness's own guard accepts every fixture record", not raised, f"{raised[:2]}")

    # ... and that the guard actually refuses a record with an invented label (a test of the test).
    broken = json.loads(json.dumps(synthetic_records()[0][2]))
    broken["_meta"]["group_id"] = "test/broken/label"      # the key real records carry
    broken["questions"]["label"]["label"] = "a label that is not an option"
    try:
        K.check_labels(broken)
        check("the guard refuses a label that is not an option key", False, "no exception raised")
    except P.ProbeError as error:
        named = "test/broken/label" in str(error) and "a label that is not an option" in str(error)
        check("the guard refuses a label that is not an option key, naming the record and the label",
              named, str(error)[:160])

    predictor = FakeKev()
    rows = []
    for group, family, record in synthetic_records():
        dataset, cell = group.split("/", 1)
        new_rows, _ = K.build_rows(record, family, "test", predictor, raw, dataset, cell)
        rows.extend(new_rows)
    check("rows were built for every question", len(rows) == 12, f"{len(rows)} rows, {predictor.calls} asks")
    check("the raw asks were saved one file per (record, ask)",
          len(list(raw.glob("*.json"))) == 13, f"{len(list(raw.glob('*.json')))} files")

    rows_path = work / "rows.jsonl"
    P.write_rows(rows_path, rows)
    reloaded = P.read_rows(rows_path)
    check("rows.jsonl round-trips", len(reloaded) == len(rows) and reloaded[0]["id"] == rows[0]["id"],
          f"{len(reloaded)} rows")

    choice = [row for row in reloaded if row["type"] == "choice"]
    dump_path = work / "axis1-rows-kev.jsonl"
    dump = write_row_dump(dump_path, choice)
    written = [json.loads(line) for line in dump_path.read_text(encoding="utf-8").splitlines()]
    keys = set(written[0])
    check("dump schema is DUMP_FIELDS + positive", keys == set(DUMP_FIELDS) | {"positive"},
          f"extra {sorted(keys - (set(DUMP_FIELDS) | {'positive'}))}, missing "
          f"{sorted((set(DUMP_FIELDS) | {'positive'}) - keys)}")
    check("positive column agrees with axis1_label",
          all(bool(row["positive"]) == metrics.axis1_label(row) for row in written))
    check("both serving conventions are present",
          {row["abstain_source"] for row in written} == {metrics.FACTORY, metrics.INJECTED},
          f"{sorted({row['abstain_source'] for row in written})}")
    check("injected rows have no abstain index and a none_p", all(
        row["abstain"] is None and row["none_p"] is not None
        for row in written if row["abstain_source"] == metrics.INJECTED))
    check("factory rows carry the abstain index", all(
        row["abstain"] is not None and row["none_p"] is not None
        for row in written if row["abstain_source"] == metrics.FACTORY))

    scorer = import_scorer()
    loaded = scorer.load(dump_path)
    served = [scorer.as_served(row) for row in loaded]
    problems = scorer.verify(loaded, served)
    check("axis1-recompute.py reads the dump with 0 reconstruction problems", not problems,
          "; ".join(problems[:2]))
    adequacy = metrics.panel_adequacy([dict(row) for row in served])
    check("the library metric returns a primary statistic on this dump",
          adequacy["primary"] is not None, adequacy["primary_note"])
    if adequacy["primary"]:
        auroc, n = scorer.value([dict(row) for row in served], "all")
        check("the scorer's own number path agrees with the metric",
              abs(auroc - adequacy["auroc"]) < 1e-12 and n == adequacy["n"], f"auroc {auroc:.4f}, n {n}")
    for row in served:
        if row["abstain_source"] == metrics.INJECTED and row["positive"]:
            check("no injected row claims a positive", False, f"row {row['id']}")
    check("dump row count equals the choice rows", dump["rows"] == len(choice), f"{dump['rows']}")


def import_scorer():
    """``scripts/axis1-recompute.py`` as a module — its own loader, its own pass-through, its own check."""
    path = ROOT / "scripts" / "axis1-recompute.py"
    spec = importlib.util.spec_from_file_location("axis1_recompute", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- 2. the encoder

def encoder_checks() -> None:
    print("\n2. encoder: the Kev body equals the Jev body, apart from `model`; and the label invariant")
    partition = ROOT / "runs" / "real-v2" / "test.jsonl"
    if not partition.exists():
        check("corpus present for the encoder check", False, str(partition))
        return
    wanted = {"panel_choice", "panel_miss", "evidence_diluted", "marker_noul"}
    from sckevtype.jev.request import option_keys

    found: dict[str, dict] = {}
    injectable: dict[str, dict] = {}
    injectable_records: Counter = Counter()
    scanned = questions = choice_questions = violations = 0
    needs_injection: Counter = Counter()
    already_offering: Counter = Counter()
    with partition.open(encoding="utf-8") as handle:
        for line in handle:
            if scanned >= ENCODER_SCAN_LINES:
                break
            scanned += 1
            record = json.loads(line)
            family = record["_meta"]["family"]
            if family in wanted and family not in found:
                found[family] = record
            if K.variant_questions(record) is not None:
                injectable_records[family] += 1
                injectable.setdefault(family, record)
            for question in record["questions"].values():
                questions += 1
                if question["type"] != "choice":
                    continue
                choice_questions += 1
                if question["label"] not in option_keys(question):
                    violations += 1
                (already_offering if K.variant_questions(record) is None else needs_injection)[family] += 1
            if len(found) == len(wanted) and len(injectable) >= 1 and scanned >= ENCODER_SCAN_LINES // 4:
                break

    # The invariant the row builder's `keys.index(label)` needs, on REAL records: the fixture cannot drift
    # from the contract without this failing, and a corpus that broke it would be caught here rather than
    # on the cluster after the GPU time was spent.
    check(f"every choice label in {choice_questions} real choice questions is one of its own option keys",
          violations == 0, f"{violations} violation(s) over {scanned} records / {questions} questions")
    check("at least one corpus record was read for the encoder check", bool(found),
          f"{len(found)} of {len(wanted)} families in the first {scanned} records: {sorted(found)}")

    mismatches = [family for family, record in found.items()
                  if {**K.body_for(record, K.KEV_MODEL), "model": JEV_MODEL} != record_request(record, JEV_MODEL)]
    check("bodies are equal up to `model`", not mismatches, f"differ: {mismatches}")
    check("the injected convention is needed by real records (the second ask is not hypothetical)",
          sum(injectable_records.values()) > 0,
          f"records needing injection {dict(injectable_records)} / questions "
          f"factory-given {dict(already_offering)} vs injected {dict(needs_injection)}")

    variant_source = next(iter(injectable.values()), None)
    if variant_source is None:
        check("a real question without an abstain option can be injected into", False,
              f"no injectable record among {scanned} read: {sorted(found)}")
    else:
        base_question = next(question for question in variant_source["questions"].values()
                             if question["type"] == "choice")
        variant = K.variant_questions(variant_source)
        injected_key = abstain_key(list(variant["label"]["criteria"]))
        check("the abstain variant adds exactly one option, and its key comes from the factory's vocabulary",
              len(variant["label"]["criteria"]) == len(base_question["criteria"]) + 1 and injected_key is not None,
              f"{len(variant['label']['criteria'])} options, injected key {injected_key!r}")
        check("the injected option is not one of the question's own labels (it is never a positive)",
              injected_key != base_question["label"], f"key {injected_key!r}, label {base_question['label']!r}")
        check("the variant ask is addressed in the Jev probe's shape (one question, the record's state)",
              set(variant) <= set(variant_source["questions"]), f"{sorted(variant)}")
    noul = found["marker_noul"]["questions"]
    check("the noul family's instructions are an object, not a string (the API's other content type)",
          any(isinstance(question["instructions"], dict) for question in noul.values()))


def population_checks() -> None:
    """The plan step's population logic, over REAL corpus records — the check its synthetic fixture cannot be.

    ``collect_records`` is the production call, so this is the same walk the cluster's `plan` step makes;
    it streams the partition once (731 MB). What it proves is the thing the ladder depends on: the sample's
    line numbers name the cells the sample says they do, and each record's own ``_meta.group_id`` agrees.
    """
    print("\n2b. population: the sample's names, checked against the records the corpus hands back")
    sample_path = ROOT / "reports" / "jev-smoke" / "sample.json"
    partition = ROOT / "runs" / "real-v2" / "test.jsonl"
    if not (sample_path.exists() and partition.exists()):
        check("a sample and a corpus are present for the population check", False,
              f"{sample_path.exists()=} {partition.exists()=}")
        return
    sample = json.loads(sample_path.read_text(encoding="utf-8"))
    tasks = P.collect_records(partition, sample)
    expected = len(sample["chosen"]) * len(sample["families"])
    check("collect_records returns every (cell, family) the sample names", len(tasks) == expected,
          f"{len(tasks)} tasks, {expected} expected ({len(sample['chosen'])} cells x "
          f"{len(sample['families'])} families)")
    identities = [K.record_identity(record) for _, _, record in tasks]
    unnamed = [identity for identity in identities if identity.startswith("<no ")]
    check("every real record names itself (no bare _meta['id'] read)", not unnamed,
          f"{len(unnamed)} of {len(tasks)} without _meta.group_id")
    mismatches = [(group, identity) for (group, _, _), identity in zip(tasks, identities) if group != identity]
    check("every real record's own id equals the cell the sample named for it", not mismatches,
          f"{len(mismatches)} mismatch(es), first {mismatches[:2]}")
    named = {(group, family) for group, family, _ in tasks}
    declared = {(group, family) for group, families in sample["records"].items() for family in families}
    check("the (cell, family) set is the sample's own", named == declared,
          f"only in plan {sorted(named - declared)[:2]}, only in sample {sorted(declared - named)[:2]}")


# --------------------------------------------------------------------------- 3. syntax + slurm header

#: What the task script's header has to ask for. A Slurm directive after the first executable line is
#: silently ignored, so the block's *position* is checked as well as its contents.
SLURM_REQUIRED = ("partition", "gres", "time", "mem", "cpus-per-task", "job-name", "output")

#: The shell variables the task script passes on its command lines, with values that satisfy the parser.
#: The check is about argument *order*, so the values only have to be admissible: `--dtype "$DTYPE"` is a
#: shell variable, not the literal string argparse would reject as an invalid choice.
SHELL_VALUES = {"PY": "python", "OUT": "runs/kev-probe", "DUMP": "reports/axis1-rows-kev.jsonl",
                "RUN": "jaredpalmer/kev-9b", "DTYPE": "bf16", "SAMPLE": "reports/jev-full/sample.json",
                "JEV_ROWS": "reports/jev-full/rows.jsonl", "W": "/path/to/x",
                # the full-weight arm's task script only
                "GPUS": "4", "RESERVE_GIB": "2", "PREFLIGHT_ONLY": "0", "PUBLISHED_BYTES": "51279886109"}

#: The task scripts whose header, `bash -n` and kev_probe.py invocations are checked. The 27B's script is
#: a separate file (a full-weight checkpoint is a different resource ask and a different preflight), so it
#: needs the same checks -- an argument-order mistake there costs a queue slot exactly as it did before.
TASK_SCRIPTS = ("kev-probe-task.sh", "kev-probe-task-27b.sh")

#: How far into ``runs/real-v2/test.jsonl`` the encoder check reads looking for one record per family.
#: The file is 731 MB; a bounded scan keeps the check a check rather than a corpus read. The cluster's
#: `plan` step runs the same comparison over all 800 sampled records, which is where the full claim lives.
ENCODER_SCAN_LINES = 20000


def shell_checks() -> None:
    print("\n3. syntax: the harness parses, the shell scripts parse, the Slurm header is well-formed")
    for name in ("scripts/kev/kev_probe.py", "scripts/kev/selftest.py"):
        path = ROOT / name
        try:
            ast.parse(path.read_text(encoding="utf-8"))
            check(f"{name} parses", True)
        except SyntaxError as error:
            check(f"{name} parses", False, f"{error}")

    for name in ("<cluster-job>", "<cluster-job>",
                 "<cluster-job>", "<cluster-job>"):
        path = ROOT / name
        if not path.exists():
            check(f"{name} exists", False)
            continue
        result = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
        check(f"{name} passes bash -n", result.returncode == 0, result.stderr.strip()[:200])

    for task in TASK_SCRIPTS:
        slurm_and_invocations(ROOT / "scripts" / "cluster" / task, task)


def slurm_and_invocations(path: Path, name: str) -> None:
    """One task script's Slurm header, its step coverage and the argument order of its invocations."""
    if not path.exists():
        check(f"{name}: the task script's Slurm header", False, "script absent")
        return
    lines = path.read_text(encoding="utf-8").splitlines()
    directives: dict[str, str] = {}
    first_code = None
    for number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if number > 1 and stripped and not stripped.startswith("#"):
            first_code = number
            break
        if stripped.startswith("#SBATCH"):
            token = stripped.split(None, 1)[1].strip() if len(stripped.split(None, 1)) > 1 else ""
            key = token.split("=", 1)[0].lstrip("-")
            directives[key] = token.split("=", 1)[1] if "=" in token else ""
    missing = sorted(key for key in SLURM_REQUIRED if key not in directives)
    check(f"{name}: every required #SBATCH directive is present", not missing, f"missing {missing}")
    check(f"{name}: the partition is an a100 partition", "a100" in directives.get("partition", ""),
          directives.get("partition"))
    check(f"{name}: the gres asks for a GPU", "gpu" in directives.get("gres", ""), directives.get("gres"))
    check(f"{name}: the whole #SBATCH block precedes the first executable line",
          all(not line.strip().startswith("#SBATCH") for line in lines[(first_code or len(lines) + 1) - 1:]),
          f"first executable line {first_code}")
    check(f"{name}: the job's own log lands under logs/",
          directives.get("output", "").startswith("/path/to/") and "/logs/" in directives.get("output", ""),
          directives.get("output"))
    check(f"{name}: the job is named", bool(directives.get("job-name")), directives.get("job-name"))

    # The two sibling CLIs disagree about where --out lives: the Jev probe's is top-level, this one's is
    # per subcommand. Getting it wrong costs a job that dies in seconds (job 4125076 did exactly that),
    # so the invocations the task script actually runs are parsed here with the harness's own parser.
    invocations = []
    pending = ""
    for line in lines:
        pending += line.rstrip("\\").strip() + " "
        if line.rstrip().endswith("\\"):
            continue
        # Only the harness's own invocations: an interpreter before the path, `--` after it. A quoted file
        # list (`for p in ... "scripts/kev/kev_probe.py" ...`) and a `python3 -c` one-liner that mentions the
        # path are not CLI calls, and extracting them would feed shlex an unbalanced quote.
        head, found, tail = pending.partition("scripts/kev/kev_probe.py")
        if found and '"$PY"' in head and tail.strip().startswith("--"):
            invocations.append(tail)
        pending = ""
    parsed, refused = [], []
    for invocation in invocations:
        try:
            expanded = re.sub(r'"\$([A-Z_]+)"', lambda m: SHELL_VALUES.get(m.group(1), "x"), invocation)
            tokens = shlex.split(expanded)
            parsed.append(K.parse_args(tokens).command)
        except (SystemExit, ValueError) as error:
            refused.append(f"{invocation.strip()[:70]} -> {error}")
    check(f"{name}: every kev_probe.py invocation parses (--out after the subcommand)",
          bool(invocations) and not refused, f"{len(invocations)} found, refused: {refused[:2]}")
    check(f"{name}: the task script runs all three steps", {"plan", "ask", "dump"} <= set(parsed), f"{parsed}")
    # ... and that this check can fail: the order that broke job 4125076 must be refused by the same parser.
    try:
        K.parse_args(shlex.split("--corpus runs/real-v2 --out runs/kev-probe plan --run x"))
        check("the parser refuses --out before the subcommand (the bug that killed job 4125076)", False,
              "accepted a top-level --out")
    except SystemExit:
        check("the parser refuses --out before the subcommand (the bug that killed job 4125076)", True)


# --------------------------------------------------------------------------- 4. no stray writes

def write_scope_checks() -> None:
    print("\n4. write scope: the harness writes only under the run dir and reports/")
    source = (ROOT / "scripts" / "kev" / "kev_probe.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    literals = {node.value for node in ast.walk(tree)
                if isinstance(node, ast.Constant) and isinstance(node.value, str)}
    # A real absolute path has a second separator in it; `.split("/", 1)` and `.replace("/label", "")` do not,
    # and are not paths this check is about.
    suspicious = sorted(text for text in literals
                        if text.startswith("/") and "/" in text[1:] and not text.startswith("/mnt"))
    check("no absolute output paths are hardcoded in the harness", not suspicious, f"{suspicious}")
    check("the run dir defaults under runs/, the dump under reports/",
          any(text.startswith("runs/") for text in literals) and "reports/axis1-rows-kev.jsonl" in literals,
          f"{sorted(t for t in literals if t.startswith('reports/'))}")
    # The Jev probe's artefacts are inputs to compare against: the harness reads them (sample, rows, digest)
    # and must never write one. Stated over the write calls rather than over every mention in the file, so
    # a docstring referring to the old probe's paths is not mistaken for a write.
    writes = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
              and isinstance(node.func, ast.Name)
              and node.func.id in {"write_json", "write_rows", "write_rows_partial"}]
    hardcoded = [ast.unparse(node)[:60] for node in writes
                 if node.args and isinstance(node.args[0], ast.Constant)]
    check("every write target is a path variable, never a hardcoded literal", not hardcoded, f"{hardcoded[:2]}")
    check("no write call names a Jev-probe artefact", not [node for node in writes
                                                          if "jev-smoke" in ast.unparse(node)],
          f"{[ast.unparse(node)[:60] for node in writes if 'jev-smoke' in ast.unparse(node)][:2]}")
    check("plan/evaluate write through the one --out directory",
          "plan.json" in literals and "ask-log.json" in literals and "rows.jsonl" in literals)


def main() -> int:
    work = Path(tempfile.mkdtemp(prefix="kev-selftest-"))
    print(f"kev selftest — work dir {work}")
    try:
        formatter_checks(work)
        encoder_checks()
        population_checks()
        shell_checks()
        write_scope_checks()
    finally:
        shutil.rmtree(work, ignore_errors=True)

    print(f"\n{len(PASSES)} ok, {len(FAILURES)} failed")
    for failure in FAILURES:
        print(f"  FAIL {failure}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
