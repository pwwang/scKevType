"""Step 6 -- the apples-to-apples baseline: a logistic regression on our model's own task shape.

Steps 3 and 4 answer "what does a standard annotator make of these cells". Both of them see
the **full expression matrix**, and both answer a different question from ours: they emit one
label from a fixed global label set. Our model instead reads a cell's **rank list** (the
corpus's ``state``: ~201 gene symbols with ranks, no expression values) plus the **option
definitions** it is asked about, and chooses among the candidate types that record's panel
raises -- or abstains.

This step builds the baseline on that same shape, from the corpus alone:

* rows: one per (train record, option) pair, options = the record's own option list plus the
  abstain option ("other");
* features: only what our model gets -- the panel genes present, weighted by 1/rank, the
  option's identity, and the definition-marker features (how many of the option's positive
  markers appear in the panel, how highly ranked, how many of its negative markers appear);
* target: 1 for the option the record's own label names;
* prediction: argmax over the record's options, so abstention is a first-class answer.

Nothing here reads an expression value, an atlas, or a baseline model file: the whole point
is that the only difference from our model is the model. ``--variant`` selects which corpus
variant to score; the default "clean" matches baselines/step1_cells.py.

**``--capped <dir>`` is the population switch, and it is the reason this step can be put next to
our model at all.** Our model's headline accuracy was measured on the *capped* evaluation suite
(``artifacts/suites/real-v2-eval``: 31,997 of the corpus's 89,802 test records, a stratified
sample). This step's default is the *whole* test split. Two numbers over different populations do
not belong in adjacent columns, so ``--capped`` points the score at exactly the capped draw:

* the directory is the output of ``scripts/baselines/capped_population.py``, which rebuilds the
  sampled partition with the suite builder's own ``cap_test_partition`` and proves it by digest;
* this step re-checks two things rather than trusting the directory: ``test.jsonl`` still hashes
  to the digest the frozen report recorded, and the set of records it could score equals the set
  the cap drew for the variant being scored. Both are printed; either failing aborts the run.

Run: uv run --no-sync python baselines/step6_panel_lr.py
     uv run --no-sync python baselines/step6_panel_lr.py --capped "$TMPDIR/capped-population"
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib_common import (  # noqa: E402
    ASYMMETRY,
    CORPUS,
    QCOUT,
    TAG_LEGEND,
    arm_tags,
    definitions,
    name_to_cl,
    write_json,
)

ABSTAIN = "other"
#: The abstain option's keys, from the factory that writes them. The corpus does NOT always key its
#: abstain option ``"other"``: measured over the clean test split's choice records, 25,421 of 35,005
#: carry one and its key is ``other`` (14,835), ``none`` (4,296), ``not_listed`` (2,111),
#: ``none_of_the_above`` (2,096) or ``unknown`` (2,083); the other 9,584 carry no abstain option at
#: all. Hard-coding ``"other"`` therefore did two things to this arm, both wrong: it appended a 32nd
#: option the corpus never offered to every record whose abstain option is keyed differently, and it
#: undercounted the always-abstain floor by 20.3 points of the population. Imported from the factory
#: rather than restated, for the reason ``sckevtype.eval.metrics`` imports it: a second copy of the
#: vocabulary for the one option the ballot hinges on is a second source of truth.
from sckevtype.data.families import NONE_OPTIONS  # noqa: E402

NONE_KEYS = frozenset(key for key, _ in NONE_OPTIONS)
TEST_SPLIT_RECORDS = QCOUT / "records_test.jsonl"

#: The digest of the capped test partition the frozen evaluation was scored on, as
#: ``reports/eval-real-v2-eval.json`` records it. ``--capped`` refuses a directory whose
#: ``test.jsonl`` is not these bytes: a population that cannot be pointed at the scored one is
#: not the scored one, and scoring it anyway is how two different denominators get compared.
FROZEN_CAPPED_TEST_SHA256 = "25cf1df6a110fdffcf411683517b05c7236911626a604953d127cf32e3be32ff"


class CappedPopulation:
    """The capped evaluation suite's own draw of test records, re-checked on every use.

    Built from the directory ``scripts/baselines/capped_population.py`` writes: ``record_index.jsonl``
    (the key of every sampled record) plus ``cap.json`` and ``test.jsonl`` (the proof). Two checks
    happen here and neither is a formality:

    * **the bytes.** ``test.jsonl`` must hash to :data:`FROZEN_CAPPED_TEST_SHA256`, so the population
      cannot be swapped under the run that claims to be about it;
    * **the coverage.** Every record the cap drew for the variant being scored must actually be
      found in the corpus, and nothing else may be scored. A short match is the failure mode this
      class exists for: it would mean the number is over a subset of the population it names.
    """

    def __init__(self, directory: Path, variant: str | None):
        self.directory = Path(directory)
        index_path = self.directory / "record_index.jsonl"
        if not index_path.exists():
            raise SystemExit(f"--capped {self.directory}: no {index_path.name} -- run "
                             f"scripts/baselines/capped_population.py --out {self.directory} first")
        self.variant = variant
        self.expected: set[tuple] = set()
        self.drawn_by_variant: Counter = Counter()
        read = 0
        with index_path.open(encoding="utf-8") as handle:
            for line in handle:
                entry = json.loads(line)
                read += 1
                self.drawn_by_variant[entry.get("variant")] += 1
                if variant and entry.get("variant") not in (None, variant):
                    continue
                self.expected.add((entry["dataset_id"], entry["family"], entry["cell_id"]))
        if not self.expected:
            raise SystemExit(f"--capped {self.directory}: {read} sampled records, none with "
                             f"variant={variant!r} -- nothing to score")
        self.index_records = read

        test_path = self.directory / "test.jsonl"
        digest = None
        if test_path.exists():
            sha = hashlib.sha256()
            with test_path.open("rb") as handle:
                for block in iter(lambda: handle.read(1 << 20), b""):
                    sha.update(block)
            digest = sha.hexdigest()
        self.test_sha256 = digest
        self.matched = 0
        self.unmatched_expected: set[tuple] = set()
        self.missing_expected: list[tuple] = []

    @staticmethod
    def key(meta: dict) -> tuple:
        return (meta.get("dataset_id"), meta.get("family"), meta.get("cell_id"))

    def keeps(self, meta: dict) -> bool:
        """Is this corpus record one the cap drew for the variant being scored?"""
        key = self.key(meta)
        if key in self.expected:
            self.matched += 1
            return True
        return False

    def check(self) -> list[str]:
        """The two refusals, plus the coverage accounting. Raises rather than reporting."""
        problems = []
        if self.test_sha256 is None:
            problems.append(f"  no {self.directory / 'test.jsonl'} to hash -- the population's bytes "
                            f"cannot be checked against the frozen digest")
        elif self.test_sha256 != FROZEN_CAPPED_TEST_SHA256:
            problems.append(f"  {self.directory / 'test.jsonl'} is sha256 {self.test_sha256}, but the frozen "
                            f"evaluation was scored on {FROZEN_CAPPED_TEST_SHA256} -- this is NOT the scored "
                            f"population")
        if len(self.expected) != self.matched:
            problems.append(f"  the cap drew {len(self.expected)} records for variant={self.variant!r}, but the "
                            f"corpus supplied {self.matched} -- scoring this subset would be a number over a "
                            f"population it does not name")
        if problems:
            raise SystemExit("--capped: refusing to score\n" + "\n".join(problems))
        return problems

    def describe(self, keys: list[tuple]) -> dict:
        """The population block that rides into the payload: what was drawn, and what was scored."""
        return {
            "directory": str(self.directory),
            "what_this_is": "the capped evaluation suite's own draw of test records (artifacts/suites/"
            "real-v2-eval), rebuilt and digest-checked by scripts/baselines/capped_population.py",
            "frozen_test_jsonl_sha256": FROZEN_CAPPED_TEST_SHA256,
            "test_jsonl_sha256_on_disk": self.test_sha256,
            "bytes_match_the_scored_suite": self.test_sha256 == FROZEN_CAPPED_TEST_SHA256,
            "records_in_the_draw": self.index_records,
            "records_in_the_draw_by_variant": dict(sorted(self.drawn_by_variant.items())),
            "variant_scored_here": self.variant,
            "expected_keys_for_this_variant": len(self.expected),
            "keys_scored": self.matched,
            "keys_scored_equals_keys_drawn": len(self.expected) == self.matched,
            "distinct_keys_scored": len(set(keys)),
        }


def read_records(path: Path, variant: str | None, what: str, population=None) -> tuple[list[dict], dict]:
    """Corpus records of one split, keeping at most one variant per (cell, family).

    ``variant`` is applied to the *scored* split only. It has to be optional because the two
    splits do not carry the same variants: measured, ``test.jsonl`` holds the clean and the
    permuted record for every (cell, family) 1:1, while ``train.jsonl`` holds **only** the
    permuted ones (5,592 of 5,592). The permuted variant of a record is the same question of
    the same cell with the same target and the same option *set* -- only the option order (and
    a per-record permutation seed) differ -- which this design is invariant to, since every
    option is scored as its own (record, option) row with the option's identity as a feature.
    Hard-coding "clean" therefore silently produced an EMPTY training split; the counts are
    returned so the caller can refuse that instead of fitting on nothing.

    ``population`` -- a :class:`CappedPopulation` or ``None`` -- restricts the scored records to
    the capped evaluation suite's own draw. It is a *filter with an accounting*, not a silent
    one: the counts it matched, dropped and could not find are in the returned stats, and the
    caller asserts them (see :meth:`CappedPopulation.check_coverage`).
    """
    out, seen, variants = [], set(), Counter()
    keys: list[tuple] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            meta = record["_meta"]
            if variant and meta.get("variant") not in (None, variant):
                continue
            if population is not None and not population.keeps(meta):
                continue
            key = (meta["cell_id"], meta["family"])
            if key in seen:
                continue
            seen.add(key)
            variants[meta.get("variant")] += 1
            out.append(record)
            keys.append(population.key(meta) if population is not None else key)
    stats = {"records": len(out), "variants": dict(variants), "variant_filter": variant, "file": path.name}
    if population is not None:
        stats["population"] = population.describe(keys)
    if not out:
        raise SystemExit(
            f"{what}: {path} yielded no records with variant={variant!r} -- refusing to fit or score on "
            f"nothing. Check which variants that split actually carries (see the counts this function "
            f"records in the payload)."
        )
    return out, stats


def parse_panel(state: str) -> dict[str, float]:
    """'GENE 1 | GENE 2' -> {GENE: 1/rank}. The only expression our model ever sees."""
    panel = {}
    for token in state.split(" | "):
        parts = token.rsplit(" ", 1)
        if len(parts) != 2:
            continue
        gene, rank = parts[0], parts[1]
        try:
            panel[gene] = 1.0 / float(rank)
        except ValueError:
            continue
    return panel


def abstain_option_of(record: dict) -> str | None:
    """The key of the record's **own** abstain option, or ``None`` when the factory wrote none.

    Read off the record's criteria against the factory's own key list, the same rule
    ``sckevtype.eval.metrics.abstain_index`` uses: the abstain keys are not cell-type names and a
    panel option's key is a CL name, so the two cannot collide. More than one match would mean a
    panel had taken an abstain key -- a hard error, because the ballot would then be wrong rather
    than merely narrower. A ``noul`` record has no abstain option in its criteria (it is a yes/no
    question about one candidate), so it returns ``None``; the two-way ballot its answer needs is
    built in :func:`options_of`.
    """
    spec = next(iter(record["questions"].values()))
    if spec["type"] == "noul":
        return None
    found = [key for key in spec["criteria"] if key in NONE_KEYS]
    if len(found) > 1:
        raise SystemExit(f"{record['_meta'].get('id')!r}: {len(found)} options claim to be the abstain "
                         f"option: {found} -- the ballot would be wrong, not merely narrower")
    return found[0] if found else None


def options_of(record: dict, abstain: str | None) -> list[str]:
    """The record's candidate set: its option list plus the abstain option, in order.

    For ``noul`` the corpus asks about a single candidate, so the candidate set is that candidate
    plus abstain -- the same two-way question, expressed once. For ``choice`` the set is the
    record's own criteria, and the abstain option is **the one the record already carries**: an
    option the factory wrote is never replaced, and the literal ``"other"`` is appended only when
    the record offers no abstain option at all (which is the CLI's own injection rule). Duplicates
    are removed keeping first position: an option list that already names the abstain option (or
    repeats a type) must not put the same choice on the ballot twice.
    """
    spec = next(iter(record["questions"].values()))
    if spec["type"] == "noul":
        raw = [spec["instructions"]["candidate"], ABSTAIN]
    else:
        raw = list(spec["criteria"]) + ([] if abstain is not None else [ABSTAIN])
    out: list[str] = []
    for option in raw:
        if option not in out:
            out.append(option)
    return out


def target_of(record: dict) -> str:
    """The option the corpus's own label names (abstain when the target is 'other' or False)."""
    spec = next(iter(record["questions"].values()))
    if spec["type"] == "noul":
        return spec["instructions"]["candidate"] if spec["label"] else ABSTAIN
    return spec["label"]


def main() -> None:
    argv = sys.argv[1:]
    if "--help" in argv or "-h" in argv:
        print(__doc__)
        return
    variant = "clean"
    capped: Path | None = None
    for index, arg in enumerate(argv):
        if arg.startswith("--variant="):
            variant = arg.split("=", 1)[1]
        elif arg == "--variant":
            variant = argv[index + 1]
        elif arg.startswith("--capped="):
            capped = Path(arg.split("=", 1)[1])
        elif arg == "--capped":
            capped = Path(argv[index + 1])

    # The population is resolved BEFORE anything is read, and it either checks out or the run stops.
    population = CappedPopulation(capped, variant) if capped else None
    if population is not None:
        print(f"population: CAPPED -- {population.directory}", flush=True)
        print(f"  {population.index_records} records in the draw ({population.drawn_by_variant}); "
              f"{len(population.expected)} of them are variant={variant!r}", flush=True)
    else:
        print("population: FULL test split (no --capped)", flush=True)

    t0 = time.time()
    train, train_source = read_records(CORPUS / "train.jsonl", None, "train split")
    test, test_source = read_records(CORPUS / "test.jsonl", variant, "test split", population)
    print(f"records: train={len(train)} ({train_source['variants']}) "
          f"test={len(test)} ({test_source['variants']}, variant filter={variant!r})", flush=True)
    if population is not None:
        population.check()
        block = test_source["population"]
        print(f"  population check: bytes {block['test_jsonl_sha256_on_disk']} == frozen "
              f"{block['frozen_test_jsonl_sha256']}: {block['bytes_match_the_scored_suite']}")
        print(f"  population check: keys drawn {block['expected_keys_for_this_variant']} == keys scored "
              f"{block['keys_scored']}: {block['keys_scored_equals_keys_drawn']}")
        print(f"  coverage of the cap's draw: {len(test)} of {population.index_records} sampled records "
              f"({100 * len(test) / population.index_records:.1f}%), the rest being the same "
              f"questions with the option order permuted", flush=True)

    # The panel vocabulary is built from the training split only, so a test gene the model
    # has never seen contributes nothing -- a handicap for this baseline, quantified below.
    genes: dict[str, int] = {}
    for record in train:
        for gene in parse_panel(record["state"]):
            genes.setdefault(gene, len(genes))
    print(f"panel vocabulary from the train split: {len(genes)} gene symbols", flush=True)

    option_names = sorted({option for record in train for option in options_of(record, abstain_option_of(record))})
    option_index = {name: i for i, name in enumerate(option_names)}
    print(f"option vocabulary: {len(option_names)} names (incl. the abstain keys {sorted(NONE_KEYS)})", flush=True)

    definitions_by_name = {}
    for cl_id, entry in definitions().items():
        definitions_by_name[entry["name"]] = entry
    name_lookup = name_to_cl()

    def marker_features(option: str, panel: dict[str, float]) -> list[float]:
        """What the option's own definition says, evaluated against the panel.

        Our model reads the definition text; these three numbers are the same information in
        the form a linear model can use, so the baseline is not starved of the definitions.
        """
        entry = definitions_by_name.get(option)
        if not entry:
            return [0.0, 0.0, 0.0]
        pos = [gene for gene in (entry.get("pos_markers") or []) if gene in panel]
        neg = [gene for gene in (entry.get("neg_markers") or []) if gene in panel]
        top = max((panel[gene] for gene in pos), default=0.0)
        return [float(len(pos)), top, float(len(neg))]

    def build(records: list[dict], fit_targets: bool) -> tuple[csr_matrix, np.ndarray | None, dict, list[tuple]]:
        """Assemble the (record, option) design.

        Vectorised per record rather than per feature: the panel block is built once and
        tiled across the record's options, which is the difference between seconds and the
        better part of an hour on 45k test records.
        """
        row_parts: list[np.ndarray] = []
        col_parts: list[np.ndarray] = []
        val_parts: list[np.ndarray] = []
        y: list[int] = []
        row = 0
        record_index: list[tuple] = []
        unknown_tokens = total_tokens = 0
        unknown_options = 0
        panel_base = len(genes)
        # One column past the option one-hots is reserved for an option name the train split
        # never saw. Without it that option would be folded into the first *marker* column,
        # silently mixing "this option is new to me" with "this option's positive markers are
        # present in the panel". The count is reported in the stats either way.
        unseen_option_column = len(option_names)
        marker_base = panel_base + len(option_names) + 1
        for record in records:
            panel = parse_panel(record["state"])
            total_tokens += len(panel)
            known = [(gene, value) for gene, value in panel.items() if gene in genes]
            unknown_tokens += len(panel) - len(known)
            gene_cols = np.fromiter((genes[g] for g, _ in known), dtype=np.int64, count=len(known))
            gene_vals = np.fromiter((v for _, v in known), dtype=np.float32, count=len(known))
            options = options_of(record, abstain_option_of(record))
            unknown_options += sum(1 for option in options if option not in option_index)
            target = target_of(record)
            k = len(options)
            rows = np.arange(row, row + k, dtype=np.int64)
            meta = record["_meta"]
            record_index.append(
                (row, options, target, meta["dataset_id"], meta["family"], meta["cell_id"],
                 abstain_option_of(record) or ABSTAIN)
            )
            if gene_cols.size:
                row_parts.append(np.repeat(rows, gene_cols.size))
                col_parts.append(np.tile(gene_cols, k))
                val_parts.append(np.tile(gene_vals, k))
            row_parts.append(rows)
            col_parts.append(
                panel_base + np.fromiter(
                    (option_index.get(o, unseen_option_column) for o in options), dtype=np.int64, count=k
                )
            )
            val_parts.append(np.ones(k, dtype=np.float32))
            marker = np.array([marker_features(o, panel) for o in options], dtype=np.float32).reshape(k, 3)
            row_parts.append(np.repeat(rows, 3))
            col_parts.append(marker_base + np.tile(np.arange(3, dtype=np.int64), k))
            val_parts.append(marker.ravel())
            if fit_targets:
                y.extend(1 if option == target else 0 for option in options)
            row += k
        matrix = csr_matrix(
            (
                np.concatenate(val_parts),
                (np.concatenate(row_parts), np.concatenate(col_parts)),
            ),
            shape=(row, marker_base + 3),
        )
        coverage = 1.0 - (unknown_tokens / total_tokens if total_tokens else 0.0)
        stats = {
            "rows": row,
            "positive_share": (float(np.mean(y)) if fit_targets else None),
            "panel_token_coverage_in_vocabulary": round(coverage, 4),
            "distinct_panel_genes_seen": len(genes),
            "options_unseen_in_the_train_vocabulary": unknown_options,
        }
        return matrix, (np.array(y, dtype=np.int8) if fit_targets else None), stats, record_index

    x_train, y_train, train_stats, _ = build(train, fit_targets=True)
    print(f"train design: {x_train.shape[0]} rows x {x_train.shape[1]} columns, "
          f"{round(100 * train_stats['positive_share'], 2)}% positive, "
          f"panel token coverage {train_stats['panel_token_coverage_in_vocabulary']}", flush=True)

    model = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced", random_state=0)
    model.fit(x_train, y_train)
    print(f"fitted in {time.time() - t0:.1f}s", flush=True)

    x_test, _, test_stats, test_index = build(test, fit_targets=False)
    scores = model.predict_proba(x_test)[:, 1]

    # ------------------------------------------------------------------ scoring
    rows_out = []
    per = defaultdict(lambda: {"n": 0, "correct": 0, "picked_abstain": 0, "abstain_rate_trivial": 0,
                               "picked_literal_other": 0, "literal_other_rate_trivial": 0,
                               "first_rate_trivial": 0, "type_correct": 0, "type_n": 0})
    for start, options, target, dataset_id, family, cell_id, abstain_name in test_index:
        block = scores[start : start + len(options)]
        predicted = options[int(np.argmax(block))]
        # The same model, asked the same question with the abstain option off the ballot. The
        # two numbers answer different questions: the headline accuracy mixes "cannot tell the
        # offered types apart" with "would rather abstain", and on a family whose target is
        # always a type (panel_choice) those two failures are indistinguishable in it.
        type_slots = [i for i, option in enumerate(options) if option != abstain_name]
        type_predicted = options[type_slots[int(np.argmax([block[i] for i in type_slots]))]] if type_slots else None
        slot = per[(dataset_id, family)]
        slot["n"] += 1
        slot["correct"] += int(predicted == target)
        # "Picked the abstain option" is a question about THIS record's ballot, so it is read
        # against that record's own abstain key. The literal-'other' reading is counted beside it
        # so the size of the correction is visible rather than asserted.
        slot["picked_abstain"] += int(predicted == abstain_name)
        slot["picked_literal_other"] += int(predicted == ABSTAIN)
        slot["abstain_rate_trivial"] += int(target == abstain_name)
        slot["literal_other_rate_trivial"] += int(target == ABSTAIN)
        slot["first_rate_trivial"] += int(options[0] == target)
        slot["type_n"] += int(bool(type_slots))
        slot["type_correct"] += int(type_predicted == target)
        rows_out.append(
            {
                "cell_id": cell_id,
                "dataset_id": dataset_id,
                "family": family,
                "target": target,
                "predicted": predicted,
                "type_only_predicted": type_predicted,
                "abstain_name": abstain_name,
                "n_options": len(options),
                "top_score": round(float(np.max(block)), 4),
            }
        )

    aggregated = {}
    for (dataset, family), slot in sorted(per.items()):
        aggregated[f"{dataset}/{family}"] = {
            "n": slot["n"],
            "accuracy": round(slot["correct"] / slot["n"], 4),
            "picked_abstain_rate": round(slot["picked_abstain"] / slot["n"], 4),
            "target_is_abstain_rate": round(slot["abstain_rate_trivial"] / slot["n"], 4),
            "always_abstain_accuracy": round(slot["abstain_rate_trivial"] / slot["n"], 4),
            "always_abstain_note": "the score of a rule that always answers THIS record's own abstain option "
            "(one of the factory's keys, not necessarily the string 'other'). The floor a real abstainer has "
            "to beat on this family.",
            "always_abstain_accuracy_literal_other": round(slot["literal_other_rate_trivial"] / slot["n"], 4),
            "literal_other_note": "the same floor read as 'the label is literally the string \"other\"'. This was "
            "this arm's only reading before the abstain vocabulary was read off the corpus, and it UNDERCOUNTS "
            "the floor by every record whose abstain option the factory keyed 'none', 'not_listed', "
            "'none_of_the_above' or 'unknown' -- kept so the size of that correction is visible.",
            "picked_abstain_rate_literal_other": round(slot["picked_literal_other"] / slot["n"], 4),
            "always_first_option_accuracy": round(slot["first_rate_trivial"] / slot["n"], 4),
            "type_only_accuracy": round(slot["type_correct"] / slot["type_n"], 4) if slot["type_n"] else None,
            "type_only_note": "argmax over the record's OWN TYPE options with the abstain option removed from the "
            "ballot -- reads next to the headline accuracy as: how often the model picks the right type when it is "
            "not allowed to abstain. NOT the headline metric (that one has abstain on the ballot and is reported "
            "in `accuracy`), but it is what separates a model that cannot tell the offered types apart from one "
            "that prefers to abstain.",
        }
    by_dataset = defaultdict(lambda: {"n": 0, "correct": 0, "abstain": 0, "abstain_target": 0,
                                      "literal_other_target": 0, "type_correct": 0, "type_n": 0})
    by_family = defaultdict(lambda: {"n": 0, "correct": 0, "abstain": 0, "abstain_target": 0,
                                     "literal_other_target": 0, "type_correct": 0, "type_n": 0})
    for row in rows_out:
        for bucket, name in ((by_dataset, row["dataset_id"]), (by_family, row["family"])):
            slot = bucket[name]
            slot["n"] += 1
            slot["correct"] += int(row["predicted"] == row["target"])
            slot["abstain"] += int(row["predicted"] == row["abstain_name"])
            slot["abstain_target"] += int(row["target"] == row["abstain_name"])
            slot["literal_other_target"] += int(row["target"] == ABSTAIN)
            if row["type_only_predicted"] is not None:
                slot["type_n"] += 1
                slot["type_correct"] += int(row["type_only_predicted"] == row["target"])
    for bucket in (by_dataset, by_family):
        for name, slot in bucket.items():
            bucket[name] = {
                "n": slot["n"],
                "accuracy": round(slot["correct"] / slot["n"], 4),
                "picked_abstain_rate": round(slot["abstain"] / slot["n"], 4),
                "always_abstain_accuracy": round(slot["abstain_target"] / slot["n"], 4),
                "always_abstain_accuracy_literal_other": round(slot["literal_other_target"] / slot["n"], 4),
                "type_only_accuracy": round(slot["type_correct"] / slot["type_n"], 4) if slot["type_n"] else None,
            }

    payload = {
        "what_this_is": "logistic regression on the corpus's own task shape: the cell's rank list plus the "
        "option definitions, choosing among that record's options (abstain is one of them)",
        "tags": arm_tags("panel_lr"),
        "tag_legend": TAG_LEGEND,
        "asymmetry": ASYMMETRY,
        "population": ("capped" if population is not None else "full"),
        "population_note": (
            "SCORED ON THE CAPPED EVALUATION SUITE'S OWN DRAW -- the same records our frozen model's "
            "headline was measured on, checked by digest. Comparable with our model's numbers; NOT "
            "comparable with this step's own --uncapped output."
            if population is not None else
            "SCORED ON THE WHOLE TEST SPLIT. NOT comparable with our model's headline (0.899), which "
            "was measured on the capped suite (31,997 of 89,802 records). Re-run with --capped to put "
            "the two numbers on one population."
        ),
        "what_our_model_sees": "the same rank list and the same option definitions",
        "what_steps_3_and_4_see": "the full expression matrix -- a strictly larger input, answering a different "
        "question (one label from a fixed global label set)",
        "variant": variant,
        "variant_policy": "the --variant filter applies to the SCORED (test) split only, matching step 1/step 5. "
        "The train split is read unfiltered, one record per (cell, family), because train.jsonl carries only "
        "the permuted variant while test.jsonl carries clean and permuted 1:1; the permuted variant changes "
        "only the option ORDER, which this design is invariant to (each option is its own row, identified by "
        "its own feature). The variants actually used are recorded above.",
        "train_source": train_source,
        "test_source": test_source,
        "features": "panel genes present weighted by 1/rank, option one-hot, and 3 definition-marker features "
        "(count of the option's positive markers present in the panel, best rank among them, count of its "
        "negative markers present). No expression values anywhere: our model does not get them either.",
        "model": "sklearn LogisticRegression(C=1.0, max_iter=2000, class_weight='balanced'), one row per "
        "(record, option) pair, positive = the option the record's label names",
        "abstain_reference": "always_abstain_accuracy is the score of a rule that always answers the record's OWN "
        "abstain option -- the floor a real abstainer has to beat on these families. The factory keys that "
        "option 'other', 'none', 'not_listed', 'none_of_the_above' or 'unknown', so the floor has to be read "
        "per record; `always_abstain_accuracy_literal_other` is the narrower 'label == \"other\"' reading this "
        "arm used before the vocabulary was read off the corpus, and it undercounts.",
        "abstain_vocabulary": {
            "keys": sorted(NONE_KEYS),
            "source": "sckevtype.data.families.NONE_OPTIONS -- the factory that writes the option",
            "why_imported": "a second copy of the keys for the one option the ballot hinges on would be a second "
            "source of truth, which is the same reason sckevtype.eval.metrics imports them",
        },
        "train": train_stats,
        "test": test_stats,
        "per_dataset_family": aggregated,
        "per_dataset": by_dataset,
        "per_family": by_family,
        "seconds": round(time.time() - t0, 1),
    }
    out_json = QCOUT / ("panel_lr.json" if population is None else f"panel_lr_capped_{variant}.json")
    out_predictions = QCOUT / ("panel_lr_predictions.jsonl" if population is None
                               else f"panel_lr_predictions_capped_{variant}.jsonl")
    write_json(out_json, payload)
    with out_predictions.open("w", encoding="utf-8") as handle:
        for row in rows_out:
            handle.write(json.dumps(row) + "\n")

    print("\nper dataset x family:")
    for key, slot in aggregated.items():
        print(f"  {key}: n={slot['n']} acc={slot['accuracy']} abstain_picked={slot['picked_abstain_rate']} "
              f"floor(always_abstain)={slot['always_abstain_accuracy']} "
              f"type_only={slot['type_only_accuracy']}")
    print("\nper dataset:")
    for name, slot in by_dataset.items():
        print(f"  {name}: n={slot['n']} acc={slot['accuracy']} floor={slot['always_abstain_accuracy']} "
              f"type_only={slot['type_only_accuracy']}")
    print("\nper family:")
    for name, slot in by_family.items():
        print(f"  {name}: n={slot['n']} acc={slot['accuracy']} floor={slot['always_abstain_accuracy']} "
              f"type_only={slot['type_only_accuracy']}")
    print(f"\nwrote {out_json} and {out_predictions}")


if __name__ == "__main__":
    main()
