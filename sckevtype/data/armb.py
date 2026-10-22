"""Arm B: the evidence target, and the supervision that makes it learnable.

**The target.** For a ``choice`` question the answer is the option that the *input* —
the pair (state, panel) — supports:

    the option naming the cell's own type, if that option is on the panel
    **and** the state has not been deprived of that type's deciding evidence;
    the abstain option otherwise.

That is one rule, and the corpus's five choice families are its five cases:

=========================  ====================  ==================  ===============
family                     true option on panel  state carries it    target
=========================  ====================  ==================  ===============
``panel_choice``           yes                   yes (nothing cut)   the true option
``evidence_diluted``       yes                   tested, not assumed true option / abstain
``evidence_removed``       yes                   no (markers masked) abstain
``panel_miss``             no                    —                   abstain
``out_of_register``        no (no register term) —                   abstain
=========================  ====================  ==================  ===============

:func:`evidence_answer` recomputes that rule from a record alone — the family for the
panel clause, the state for the ``evidence_diluted`` clause — and :func:`build` asserts
it against the label the frozen corpus already carries. **The labels do not change.**
Arm B's finding is that they never needed to: measured on ``runs/real-v2/train.jsonl``
the frozen labels *are* this rule's labels on every one of the 4,434 choice rows.

**Why the frozen model does not learn it anyway** — the supervision, not the target. Measured
over the frozen training partition (``scratch/armB/measure-stream.py`` runs kev's own
augmentation and counts; every number below is counted, not estimated):

* 2,923 of the 3,562 choice rows that offer an abstain option (**82.1%**) answer it, and in
  three of the five families it is the answer on **100%** of rows, so "answer the abstain
  option whenever one is offered" scores **0.6592** over the partition's own choice rows while
  reading nothing;
* the counter-examples — abstain offered, a real option correct — are **639 rows**, 17.9% of
  the rows that offer one;
* **872 ``panel_choice`` rows carry no abstain option at all.** ``panel_choice`` is the family
  whose evaluation floor is 0%, and it is where the frozen model scores 29.2%: it abstains on
  a question that had no option to abstain on. :func:`sckevtype.eval.run.with_abstain`
  *injects* one when those same rows are scored, so the question trained and the question
  served were different questions — the convention gap ``paper_plan.md`` §7 measures at AUROC
  0.4673.

**Arm B is therefore one change: the abstain option is on every training choice record.** The
872 rows change and nothing else does. Its presence stops being a cue, the training question
becomes the served question, and the counter-example share rises from **17.9% to 30.3%** of
the rows that offer an abstain option (both counted after kev's own augmentation, which is what
the loss sees — ``scratch/armB/measure-stream.py``). The other partitions are copied byte for
byte and digest-checked, so the scored population is the frozen one and an arm-B number is
comparable to the frozen number. Everything else is the frozen recipe: same base and revision,
same LoRA recipe, same ``batch 1 x accum 8 x checkpointing 1``, same ``--steps 699`` (one epoch
over the same 5,592 source records), same seed, same suite apart from the 872 inserted options.

**What arm B is not.** The obvious second lever is kev's ``--p_none_pair``, which emits a
minimal pair — one state asked with and without the true type's option on the list — and which
the released finetuning recipe runs at 0.25. It was measured on this corpus rather than
assumed, and it is **not** part of arm B. The pair fires on every choice record, including the
3,090 whose answer is *already* the abstain option, and there its two siblings both answer the
abstain option — a duplicate of the source row rather than a contrast. At rate 1.0 the stream's
counter-example share falls from 30.3% to **21.5%** while the epoch more than doubles (5,592 →
14,460 rows, ≈2.6x the optimizer steps): worse supervision for 2.6x the compute. The flag is
reachable from this project's trainer CLI (default 0.0, kev's own value) so the measurement can
be repeated; ``reports/armB.md`` carries it.

**What this module does not do.** It does not touch ``artifacts/v2/`` or
``configs/seed/v2/`` (frozen for the published numbers), it does not re-render a state,
and it does not re-label a record. It reads a corpus and writes a corpus.

    python -m sckevtype.data.armb --corpus runs/real-v2 --out runs/real-v2-armB
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from ..contracts import paths
from ..contracts.definitions import NEG_LABEL, POS_LABEL, TEXT_JOIN, Definition
from ..contracts.freeze import (ENCODING, file_digest, payload_digest, read_json, text_digest,
                                write_json)
from ..contracts.render import FilterConfig
from . import families as corpus_families
from .families import HoldsRule, NONE_OPTIONS

#: The families whose evidence the corpus removed *before* rendering: their states carry
#: no deciding marker of the true type by construction, so the state clause is false and
#: the target is the abstain option whatever the option list says.
STATE_DEPRIVED = ("evidence_removed",)

#: The families whose target the *state* decides rather than the option list, and which
#: therefore need the rule re-run rather than read off the family.
STATE_TESTED = ("evidence_diluted",)

#: The choice families, in the order the corpus's manifest counts them.
CHOICE_FAMILIES = ("panel_choice", "marker_noul", "panel_miss", "evidence_removed",
                   "evidence_diluted", "axis_score", "out_of_register")

#: One record's arm-B provenance lives here, so a reader can tell at a glance which
#: records arm B inserted an option into and what it decided the target was.
META_KEY = "armb"

#: The keys ``NONE_OPTIONS`` uses: an option under any of them means "none of these".
ABSTAIN_KEYS = frozenset(key for key, _ in NONE_OPTIONS)


@lru_cache(maxsize=4)
def render_config(version: str = "v2") -> FilterConfig:
    """The filter config of one artifact version, read the way the corpus read it.

    The arm-B rule has to read a state's symbols exactly the way the family builders that
    produced the record did; anything else would be a second reader of the same format.
    ``render_config.json`` *is* the ``FilterConfig`` payload (``sckevtype.data.build`` writes
    it that way), not a wrapper around one. The version is a parameter rather than a constant
    so a corpus built against ``artifacts/v1`` is read with v1's filters, not v2's.
    """
    return FilterConfig(**read_json(paths.artifact(version, "render_config.json")))


@dataclass(frozen=True)
class NameIndex:
    """The register's ``name <-> CL id`` table, from the artifact version the corpus names.

    A choice option is stored as ``{name: definition_text}`` and the cell's CL id lives only
    in ``_meta.cl_id``, so the option that *stands for* a record's own type has to be resolved
    through the register. Both directions are bijections on this resource (82 names, 82 CL
    ids, no duplicates), which is what makes the lookup exact rather than a guess — and
    ``out_of_register`` falls out of it for free: a type the register cannot name has no name
    to find, so the panel clause fails and the target is the abstain option.
    """

    version: str
    digest: str
    by_name: Mapping[str, str]
    by_cl: Mapping[str, str]

    @classmethod
    def load(cls, version: str = "v2") -> NameIndex:
        path = paths.artifact(version, "definitions.json")
        payload = read_json(path)
        definitions = payload["definitions"]
        by_name = {str(entry["name"]): str(entry["cl_id"]) for entry in definitions}
        by_cl = {str(entry["cl_id"]): str(entry["name"]) for entry in definitions}
        if len(by_name) != len(definitions) or len(by_cl) != len(definitions):
            raise ValueError(
                f"{path}: {len(definitions)} definitions but {len(by_name)} names and {len(by_cl)} CL ids — a name "
                f"or an id is not unique, so a name cannot resolve a type and the panel clause would be a guess"
            )
        return cls(version=version, digest=str(payload.get("digest", "")), by_name=by_name, by_cl=by_cl)

    def cl_of(self, name: str) -> str | None:
        return self.by_name.get(name)

    def name_of(self, cl_id: str) -> str | None:
        return self.by_cl.get(cl_id)


# --------------------------------------------------------------------------- the rule


def parse_definition(text: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Recover ``(pos_markers, neg_markers)`` from a rendered definition string.

    The inverse of :meth:`sckevtype.contracts.definitions.Definition.render_text`, which
    writes ``<prose>. positive markers: A, B. negative markers: C.`` and never lets a marker
    contain ``", "`` or ``". "``. The parse is checked against the whole frozen resource
    (82/82 definitions round-trip) rather than trusted, because the arm-B target is computed
    from it: a silently truncated marker list would move the rule's answer without moving
    anything a test would notice.

    The two label names and the sentence separator are imported from the renderer, not
    restated, so a change to either cannot desynchronise the parser.
    """
    pos: tuple[str, ...] = ()
    neg: tuple[str, ...] = ()
    for sentence in (text or "").split(TEXT_JOIN):
        sentence = sentence.strip().rstrip(".")
        for label, which in ((POS_LABEL, "pos"), (NEG_LABEL, "neg")):
            prefix = f"{label}: "
            if sentence.startswith(prefix):
                markers = tuple(m.strip() for m in sentence[len(prefix):].split(",") if m.strip())
                if which == "pos":
                    pos = markers
                else:
                    neg = markers
    return pos, neg


def definition_from_text(name: str, text: str) -> Definition:
    """The option's definition as a :class:`Definition`, for the corpus's own rule.

    Built for :class:`~sckevtype.data.families.HoldsRule` to consume, so the evidence test
    applied here *is* the one the corpus applies — not a second implementation of it.
    """
    pos, neg = parse_definition(text)
    return Definition(cl_id="CL:0000000", name=name or "option", text=text or "",
                      pos_markers=pos, neg_markers=neg, provenance="arm-B parse")


def state_symbols(state: str, version: str = "v2") -> tuple[str, ...]:
    """The symbols of a rendered state, in rank order.

    Delegates to the corpus's own reader so the arm-B rule sees exactly the symbol list
    ``marker_noul`` and ``evidence_diluted`` were computed against.
    """
    return tuple(corpus_families.state_symbols(state, render_config(version)))


def carries_evidence(state: str, text: str, rule: HoldsRule | None = None,
                     version: str = "v2") -> bool:
    """Does this state satisfy that definition — the corpus's own rule, unmodified?"""
    return (rule or HoldsRule()).holds(state_symbols(state, version), definition_from_text("option", text))


def choice_question(record: Mapping[str, Any]) -> tuple[str | None, dict[str, Any] | None]:
    """The record's one ``choice`` question, if it has one. Returns ``(qid, question)``."""
    found = [(qid, q) for qid, q in (record.get("questions") or {}).items() if q.get("type") == "choice"]
    if not found:
        return None, None
    if len(found) > 1:
        raise ValueError(f"record {record.get('_meta', {}).get('id')!r} carries {len(found)} choice questions; "
                         f"arm B's target is defined on one question per record")
    return found[0]


def abstain_keys(criteria: Mapping[str, Any]) -> tuple[str, ...]:
    """Every abstain option on this list, in option order — the reporting form.

    Use this, not :func:`abstain_key`, on anything the *trainer* will see. A rendered training
    row can legitimately carry two: ``kev.data.augment``'s ``p_none_distract`` branch adds a
    second "none of these" wording as a *wrong* alternative while keeping the true option
    (``kev/kev/data.py``, the branch between ``p_none`` and ``p_none + p_none_distract``), and
    it chooses a key the list does not already carry. At the frozen rates that is 11.86% of the
    training rows (``scratch/armB/render-shape.py``). A corpus record never has two.
    """
    return tuple(name for name in criteria if name in ABSTAIN_KEYS)


def abstain_key(criteria: Mapping[str, Any]) -> str | None:
    """The one abstain option on a **corpus** record's option list, or ``None``.

    Strict on purpose. The frozen corpus holds at most one abstain key on every choice
    question, in the raw ``criteria`` shape and in the rendered one (4,434 train and 70,010
    test questions, counted in ``scratch/armB/render-shape.py``), so a second one means the
    caller is looking at an augmented training row — where two is expected, and where
    :func:`abstain_keys` is the right reader. Silently taking the first would hide that.
    """
    present = abstain_keys(criteria)
    if len(present) > 1:
        raise ValueError(
            f"option list carries {len(present)} abstain options: {list(present)}. A corpus record carries at most "
            f"one; two means this is an augmented training row (kev's p_none_distract adds a second none wording) "
            f"and abstain_keys() is the reader for it."
        )
    return present[0] if present else None


def real_options(criteria: Mapping[str, Any]) -> list[str]:
    """The option list's non-abstain options, in order."""
    return [name for name in criteria if name not in ABSTAIN_KEYS]


def evidence_answer(record: Mapping[str, Any], index: NameIndex, *,
                    rule: HoldsRule | None = None) -> dict[str, Any]:
    """What the arm-B target is for this record, recomputed from the record alone.

    Returns ``{"target": <label|None>, "side": "supported"|"withheld"|"n/a", "reason": <str>}``.
    The rule's parts are applied in the order they can fail:

    1. **the state clause**, for the family whose state the corpus built by removing the true
       type's deciding markers (``evidence_removed``) and for the one family where the clause
       is a *measurement* rather than a construction (``evidence_diluted``: the corpus's own
       ``HoldsRule`` re-run on the masked state, which is how the corpus labelled it);
    2. **the panel clause** — is there an option naming ``_meta.cl_id``? Its absence is what
       ``panel_miss`` (a draw that left it out) and ``out_of_register`` (a type no panel
       could contain) both mean;
    3. otherwise the state carries the evidence and the panel holds it: the target is the
       option naming the cell's own type.
    """
    rule = rule or HoldsRule()
    meta = record.get("_meta") or {}
    family = str(meta.get("family", ""))
    qid, question = choice_question(record)
    if question is None:
        return {"target": None, "side": "n/a", "reason": "no choice question"}
    criteria = question.get("criteria") or {}
    existing = abstain_key(criteria)

    if family in STATE_DEPRIVED:
        return {"target": existing, "side": "withheld",
                "reason": "state: the true type's deciding markers were removed"}

    true_name = next((name for name in real_options(criteria) if index.cl_of(name) == meta.get("cl_id")), None)

    if family in STATE_TESTED:
        if true_name is None:
            return {"target": existing, "side": "withheld", "reason": "state: the true type is not on this panel"}
        holds = carries_evidence(record["state"], criteria[true_name], rule, index.version)
        return {"target": true_name if holds else existing,
                "side": "supported" if holds else "withheld",
                "reason": f"state: HoldsRule on the masked state says {holds}"}

    if true_name is None:
        reason = ("panel: a type no register panel can name" if family == "out_of_register"
                  else "panel: a draw that left the true type out")
        return {"target": existing, "side": "withheld", "reason": reason}

    return {"target": true_name, "side": "supported", "reason": "panel + state: both carry the evidence"}


# --------------------------------------------------------------------------- B1


def choose_abstain_option(criteria: Mapping[str, Any], rng: random.Random) -> tuple[str, str | None]:
    """One factory abstain option this option list does not already carry.

    ``NONE_OPTIONS`` is the corpus's own list (``sckevtype.data.families``), so an option
    inserted here is a wording the model already meets in the other four families and in
    the evaluation's own injections. Several wordings exist for the same reason: the
    *presence* and the *phrasing* of "none of these" must not become the cue for the
    answer.
    """
    available = [option for option in NONE_OPTIONS if option[0] not in criteria]
    if not available:
        raise ValueError(f"every abstain key is already on this option list: {sorted(criteria)}")
    return available[rng.randrange(len(available))]


def insert_abstain(record: Mapping[str, Any], *, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """B1: give this record's choice question an abstain option. Returns ``(record, note)``.

    A no-op for a record that already has one, and for a record with no choice question
    (``marker_noul``). The label is **never** touched: arm B does not re-label, it makes
    the abstain answer *expressible* on the rows where the frozen corpus made it
    unreachable, which is what the evaluation has been injecting all along.

    The returned record is always a fresh mapping, so the caller's provenance stamp cannot
    reach back into the record it was given. The inserted key is drawn from a generator
    seeded by the record's own id, so the build is reproducible and does not depend on the
    order records are visited in.
    """
    qid, question = choice_question(record)
    out = json.loads(json.dumps(record, ensure_ascii=False))
    if question is None:
        return out, {"inserted": False, "reason": "no choice question"}
    criteria = dict(question.get("criteria") or {})
    existing = abstain_key(criteria)
    if existing is not None:
        return out, {"inserted": False, "abstain_key": existing, "reason": "already offered"}
    rng = random.Random(f"armb|abstain|{seed}|{record.get('_meta', {}).get('id')}")
    key, wording = choose_abstain_option(criteria, rng)
    criteria[key] = wording
    out["questions"][qid] = {**question, "criteria": criteria}
    return out, {"inserted": True, "abstain_key": key, "abstain_text": wording, "reason": "the frozen corpus offered none"}


def arm_b_record(record: Mapping[str, Any], index: NameIndex, *, seed: int,
                 rule: HoldsRule | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """One training record in, one out: B1 applied, then the target checked.

    The check is the part that must never be quietly skipped. The frozen corpus's labels
    are supposed to *be* the evidence rule's answers (:func:`evidence_answer`); where they
    are not, arm B's premise is false and the build has to say so with the record's id
    rather than train on a target it believes in and cannot name.
    """
    out, note = insert_abstain(record, seed=seed)
    answer = evidence_answer(out, index, rule=rule)
    qid, question = choice_question(out)
    family = str((out.get("_meta") or {}).get("family", ""))
    if question is None:
        note |= {"side": "n/a", "label_checked": False}
    else:
        label = question.get("label")
        agrees = label == answer["target"]
        note |= {"side": answer["side"], "target": answer["target"], "reason": answer["reason"],
                 "label_checked": True, "agrees": agrees}
        if not agrees:
            raise ValueError(
                f"{out.get('_meta', {}).get('id')!r} (family {family!r}): the corpus labels this row {label!r} "
                f"but the evidence rule answers {answer['target']!r} ({answer['reason']}). Arm B does not re-label; "
                f"a corpus whose labels are not the evidence rule's is a different corpus."
            )
    out["_meta"][META_KEY] = note
    return out, note


# --------------------------------------------------------------------------- the build


def _read_lines(path: Path) -> list[dict[str, Any]]:
    lines = []
    for line in path.read_text(encoding=ENCODING).splitlines():
        if line.strip():
            lines.append(json.loads(line))
    return lines


def _write_lines(path: Path, records: list[dict[str, Any]], *, force: bool) -> str:
    """Write JSONL and return its digest — the corpus's own writer, restated.

    ``sckevtype.data.build`` writes ``json.dumps(record.to_kev(), ensure_ascii=False)``
    per line and digests the body; this does the same so an arm-B partition is formatted
    exactly like the corpus it came from, and an untouched partition can be shown to be
    byte-identical rather than merely equivalent.
    """
    body = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    staging = path.with_name(path.name + ".tmp")
    staging.write_text(body, encoding=ENCODING)
    if path.exists() and not force and path.read_bytes() != staging.read_bytes():
        staging.unlink()
        raise ValueError(f"{path} already exists with different content (pass --force to overwrite)")
    path.parent.mkdir(parents=True, exist_ok=True)
    staging.replace(path)
    return text_digest(body)


def _family_of(record: dict[str, Any]) -> str:
    return str((record.get("_meta") or {}).get("family", "unknown"))


def build(corpus_dir: str | Path, out_dir: str | Path, *, force: bool = False,
          splits: tuple[str, ...] = ("train",), index: NameIndex | None = None) -> dict[str, Any]:
    """Write the arm-B corpus over ``corpus_dir`` into ``out_dir``. Returns its manifest.

    Only the partitions named in ``splits`` are transformed. Everything else is copied
    **byte for byte** and its digest is compared to the source manifest, because the
    evaluation is scored on those partitions: an arm-B number is only comparable to the
    frozen number if the population scored is the frozen population.
    """
    corpus_dir, out_dir = Path(corpus_dir), Path(out_dir)
    source = read_json(corpus_dir / "manifest.json")
    seed = int(source.get("seed") or 0)
    index = index or NameIndex.load(str((source.get("artifacts") or {}).get("version") or "v2"))
    out_dir.mkdir(parents=True, exist_ok=True)

    files: dict[str, dict[str, Any]] = {}
    strata: dict[str, dict[str, Counter]] = {}
    inserted = Counter()
    wording = Counter()
    checked = 0
    for name, entry in sorted(source.get("files", {}).items()):
        target = out_dir / name
        if not name.endswith(".jsonl") or name.split(".")[0] not in splits:
            body = (corpus_dir / name).read_bytes()
            if target.exists() and not force and target.read_bytes() != body:
                raise ValueError(f"{target} already exists with different content (pass --force to overwrite)")
            target.write_bytes(body)
            digest = file_digest(target)
            if digest != entry["sha256"]:
                raise ValueError(f"{name}: copied partition does not match the source digest "
                                 f"({entry['sha256'][:12]}… vs {digest[:12]}…)")
            files[name] = dict(entry)
            continue

        records = _read_lines(corpus_dir / name)
        built: list[dict[str, Any]] = []
        for record in records:
            out, note = arm_b_record(record, index, seed=seed)
            built.append(out)
            if note.get("label_checked"):
                checked += 1
            if note.get("inserted"):
                inserted[_family_of(out)] += 1
                wording[str(note.get("abstain_key"))] += 1
            cell = strata.setdefault(name.split(".")[0], {}).setdefault(
                f"{out.get('_meta', {}).get('dataset_id')}/{_family_of(out)}/{out.get('_meta', {}).get('variant')}",
                Counter())
            cell["rows"] += 1
            cell["supported"] += note.get("side") == "supported"
            cell["withheld"] += note.get("side") == "withheld"
            cell["abstain_offered"] += abstain_key((choice_question(out)[1] or {}).get("criteria") or {}) is not None
        digest = _write_lines(target, built, force=force)
        files[name] = {"sha256": digest, "bytes": target.stat().st_size, "kind": entry.get("kind", "data"),
                       "records": len(built),
                       "questions": sum(len(record.get("questions") or {}) for record in built)}

    counts = {
        "records": sum(entry.get("records", 0) for entry in files.values()),
        "choice_rows_checked": checked,
        "abstain_options_inserted": sum(inserted.values()),
        "abstain_options_inserted_per_family": dict(sorted(inserted.items())),
        "abstain_wordings_used": dict(sorted(wording.items())),
    }
    manifest = {
        "name": f"{source.get('name', corpus_dir.name)}-armB",
        "version": f"{source.get('version', '')}+armB",
        "seed": seed,
        "config_digest": source.get("config_digest"),
        "datasets": source.get("datasets"),
        "artifacts": source.get("artifacts"),
        "files": files,
        "counts": counts,
        "splits": source.get("splits"),
        "armB": {
            "source": {"path": str(corpus_dir), "name": source.get("name"), "version": source.get("version"),
                       "manifest_sha256": file_digest(corpus_dir / "manifest.json")},
            "transformed_splits": sorted(splits),
            "unchanged_splits": sorted({entry for name in files if name.endswith(".jsonl")
                                        for entry in [name.split(".")[0]]} - set(splits)),
            "register": {"version": index.version, "digest": index.digest, "terms": len(index.by_name)},
            "target": "the option naming the cell's own type when it is on the panel and the state has not been "
                      "deprived of that type's deciding evidence; the abstain option otherwise",
            "change": "one factory abstain option inserted on every training choice record that offered none, so "
                      "its presence is not a cue and the question trained is the question served",
            "rejected": "kev's --p_none_pair 1.0, measured on this corpus and not used: it fires on every choice "
                        "record, including the 3,090 whose answer is already the abstain option, where both "
                        "siblings answer the abstain option, so the counter-example share falls from 30.3% to "
                        "21.5% while the epoch more than doubles (5,592 -> 14,460 rows; "
                        "scratch/armB/measure-stream.py)",
            "labels_changed": 0,
            "per_stratum": {split: {key: dict(sorted(value.items())) for key, value in sorted(cells.items())}
                            for split, cells in sorted(strata.items())},
            "config_digest_of_builder": payload_digest({"B1": True}),
        },
    }
    write_json(out_dir / "manifest.json", manifest)
    return manifest


def report(manifest: dict[str, Any]) -> None:
    """The build's own numbers, printed where they are read."""
    counts = manifest["counts"]
    print(f"{manifest['name']} ({manifest['version']}): {counts['records']} records over "
          f"{[entry.get('records') for entry in manifest['files'].values() if 'records' in entry]}")
    print(f"  choice rows whose label was checked against the evidence rule: {counts['choice_rows_checked']}")
    print(f"  abstain options inserted: {counts['abstain_options_inserted']} "
          f"({counts['abstain_options_inserted_per_family']})")
    print(f"  wordings used: {counts['abstain_wordings_used']}")
    for split, cells in manifest["armB"]["per_stratum"].items():
        print(f"  {split}: abstain-correct share of the choice rows that offer an abstain option, per stratum")
        for key, cell in cells.items():
            offered = max(cell["abstain_offered"], 1)
            print(f"    {key:<52} n {cell['rows']:>5}  offered {cell['abstain_offered']:>5}  "
                  f"withheld {cell['withheld']:>5}  supported {cell['supported']:>5}  "
                  f"abstain share {cell['withheld'] / offered:.4f}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sckevtype.data.armb",
                                     description="Write the arm-B corpus: the evidence target, with the supervision to reach it.")
    parser.add_argument("--corpus", required=True, help="source corpus directory (read-only)")
    parser.add_argument("--out", required=True, help="output corpus directory")
    parser.add_argument("--force", action="store_true", help="overwrite files whose bytes changed")
    parser.add_argument("--splits", default="train",
                        help="comma-separated partitions to transform (default: train — the evaluation is scored "
                             "on the others, so they are copied byte for byte and digest-checked)")
    args = parser.parse_args(argv)
    splits = tuple(part.strip() for part in args.splits.split(",") if part.strip())
    try:
        manifest = build(args.corpus, args.out, force=args.force, splits=splits)
    except (SystemExit, ValueError, KeyError, FileNotFoundError, ImportError) as error:
        print(f"arm-B build failed: {error}", file=sys.stderr)
        return 1
    report(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
