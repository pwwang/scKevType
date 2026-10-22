"""Our corpus records in TypeSafe System One's request shape — the encoder for this model class.

`docs/PLAN-audit-steps-by-expected-result.md` E3.1 asks for a client that renders our state into the
API format. This module is the rendering half; :mod:`sckevtype.jev.client` is the transport half.

The contract it targets is TypeSafe's, read from ``https://docs.typesafe.ai/api`` on 2026-09-29 and
mirrored by `kev` and by :class:`sckevtype.contracts.schema`::

    POST /v1/systemone
    {"state": <str | object | array>, "model": "jev-latest",
     "questions": {<id>: {"type": "choice|noul|score", "instructions": <any>,
                          "criteria": {option: rubric} | [level, ...] | null}}}

Three rules, and each one is a claim the paper would otherwise have to take on trust:

**One renderer, and the state is the corpus's own.** ``state`` is the record's ``state`` string
byte for byte — the output of :func:`sckevtype.contracts.render.render_rank_list`. Nothing here
re-renders, re-orders, truncates or re-filters it; a second renderer would be free to drift from
the frozen one, and a drift in the prompt would look exactly like a result about the model.

**Labels never leave the machine.** The wire payload is built by *allow-list*
(``QUESTION_FIELDS``), so a question's ``label``, ``src``, ``target`` and the record's ``_meta``
cannot reach the API by accident. :func:`check_request` then re-derives that claim from the built
payload and raises if a key it does not know appears anywhere inside a question.

**A long record is cut at one place, and the cut is recorded.** System One accepts a large state
and up to 255 options; our longest question is 31 options and our longest state ~2.3 kB, so nothing
in the corpus needs cutting — :func:`check_request` measures it instead of assuming it, and reports
the state and option counts so a reader can see that no truncation happened. (Kev's serving limits
are 8,192 tokens for state + one question; Jev's are documented as 32,768 per branch and 65,536 per
request. Neither is approached, and this module does not guess which one binds.)

Determinism: a pure function of the record. Option order is the JSON's own insertion order, which is
part of the corpus bytes — the model is order-sensitive (``option order matters`` is in every
write-up of this class) so the order is carried through rather than normalised.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

from ..contracts.freeze import payload_digest

#: The endpoint. ``POST <BASE_URL>`` is the whole API surface this package uses.
DEFAULT_BASE_URL = "https://api.typesafe.ai/v1/systemone"

#: The vendor alias. The response's own ``model`` field reports the versioned id that answered
#: (``jev-1.13.0`` on 2026-09-29), which is what a report should pin.
DEFAULT_MODEL = "jev-latest"

#: The two models ``GET /v1/models`` returned on 2026-09-29. Anything else is refused rather than
#: sent: an unknown model name is a 422 that would be counted as a transport failure.
KNOWN_MODELS = ("jev-latest", "jev-preview")

#: The only question fields that go on the wire. Allow-list, not deny-list — ``kev.data.api_request``
#: does the same thing for the same reason, and it is the one place a label could leak from.
QUESTION_FIELDS = ("type", "instructions", "criteria")

#: The question types this probe serves. ``score`` exists in the contract and in the corpus's
#: schema; no record in ``runs/real-v2`` asks one (the ``axis_score`` family is empty in the
#: manifest), so an ``axis_score`` record reaching this module is a corpus change, not a client
#: limitation, and it raises rather than being silently sent.
SERVED_TYPES = ("choice", "noul")

#: What the vendor documents as the per-request ceiling: 32,768 tokens per branch, 65,536 per
#: request, 255 options per Choice. Recorded here so :func:`check_request` compares against a
#: number that came from a source rather than from this file's author.
MAX_OPTIONS = 255
MAX_BRANCH_TOKENS = 32768


class RequestError(RuntimeError):
    """A record that cannot be put on the wire as written, named rather than repaired."""


# --------------------------------------------------------------------------- questions


def option_keys(question: Mapping[str, Any]) -> list[str]:
    """The keys a question's distribution is reported under, in option order.

    The same three cases as :func:`kev.api.question_keys`, restated here rather than imported so
    this package stays free of the training stack (``sckevtype.laya`` makes the same choice for the
    same reason): ``choice`` -> the criteria names, ``noul`` -> ``["false", "true"]``, ``score`` ->
    the level indices as strings.
    """
    qtype = question["type"]
    criteria = question.get("criteria")
    if qtype == "choice":
        if not isinstance(criteria, Mapping) or not criteria:
            raise RequestError(f"a choice question needs a non-empty criteria map; got {type(criteria).__name__}")
        return list(criteria)
    if qtype == "noul":
        return ["false", "true"]
    if qtype == "score":
        return [str(index) for index in range(len(criteria or []))]
    raise RequestError(f"unknown question type {qtype!r}")


def label_index(question: Mapping[str, Any]) -> int:
    """Where the record's own answer sits among :func:`option_keys`.

    A label that is not one of the question's keys raises: it would otherwise be scored as index
    ``-1`` (``list.index`` does not raise here — it would raise, but a ``bool`` label would not) and
    quietly become the last option. The corpus's labels are the option name for a choice, true/false
    for a noul, and the level index for a score.
    """
    keys = option_keys(question)
    label = question["label"]
    if question["type"] == "choice":
        if label not in keys:
            raise RequestError(f"label {label!r} is not one of this question's options")
        return keys.index(label)
    if question["type"] == "noul":
        if not isinstance(label, bool):
            raise RequestError(f"a noul label must be a bool; got {label!r}")
        return int(label)
    return int(label)


def question_payload(question: Mapping[str, Any]) -> dict[str, Any]:
    """One question as it goes on the wire: the allow-listed fields, nothing else.

    ``criteria`` is passed through as the corpus stores it — an ordered map of option to rubric for
    a choice, ``None`` or a ``{true, false}`` pair for a noul. Rebuilding it would be a second
    implementation of the question format; the contract accepts the same shape.
    """
    qtype = question.get("type")
    if qtype not in SERVED_TYPES:
        raise RequestError(f"question type {qtype!r} is not served by this probe (it serves {SERVED_TYPES})")
    payload = {name: question[name] for name in QUESTION_FIELDS if name in question}
    if "instructions" not in payload:
        raise RequestError("a question needs instructions: the model is given the question in them, not in the key")
    return payload


# --------------------------------------------------------------------------- requests


def state_text(record: Mapping[str, Any]) -> str:
    """The record's state, verbatim, with the one thing this module refuses to do to it.

    The corpus stores the rendered rank list as a string; that string is what the frozen renderer
    produced and what every prior scKevType number was measured on. A non-string state (an object or
    array) is legal in the API but does not occur in ``runs/real-v2``, and rendering it here would
    put a *different* prompt on the wire than the one the corpus froze — so it raises.
    """
    state = record.get("state")
    if not isinstance(state, str) or not state:
        raise RequestError(f"a state must be the corpus's own rendered string; got {type(state).__name__}")
    return state


# --------------------------------------------------------------------------- batching

#: The separator between a family name and a question id in a batched request. A family name has no
#: ``/`` in it and no question id in the corpus starts with one, so the mapping back is unambiguous;
#: ``marker_noul``'s ids carry ``:`` (``holds:CL:0000057``), which is why the separator is not that.
BATCH_SEPARATOR = "/"


def state_digest(record: Mapping[str, Any]) -> str:
    """sha256 of the record's state bytes — the key two records must agree on to share a request."""
    return hashlib.sha256(state_text(record).encode("utf-8")).hexdigest()


def state_groups(families: Sequence[str], records: Mapping[str, Mapping[str, Any]]) -> list[list[str]]:
    """The asked families grouped by **byte-identical** state, in the order first seen.

    A batched request carries one state, so only records whose state agrees to the byte can share
    it. Grouping by digest rather than assuming one group per cell is the point: the corpus's
    ``evidence_diluted`` family is a *diluted panel* of the same cell, so its state is a different
    string from the family trio's, and a request that carried the trio's state for it would be a
    different experiment dressed as the same one.
    """
    order: dict[str, int] = {}
    groups: list[list[str]] = []
    for family in families:
        if family not in records:
            raise RequestError(f"no record for family {family!r}; a batch needs every family it asks")
        digest = state_digest(records[family])
        if digest not in order:
            order[digest] = len(groups)
            groups.append([])
        groups[order[digest]].append(family)
    return groups


def family_of(qid: str) -> str:
    """The family a batched question id was prefixed with (``"panel_miss"`` from ``"panel_miss/label"``)."""
    return qid.split(BATCH_SEPARATOR, 1)[0]


def batched_request(records: Mapping[str, Mapping[str, Any]], families: Sequence[str],
                    model: str = DEFAULT_MODEL,
                    variants: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """One request carrying every listed family's questions over the state they all share.

    The API sends one state per request, so batching is possible exactly when the states agree:
    :func:`state_groups` is what decides, and this function **re-raises** rather than picking one —
    a batch that quietly served the wrong state would be a result about the wrong prompt.

    Question ids are prefixed ``<family>/<qid>`` so two families' ``label`` questions can sit in one
    request without colliding; the prefix is stripped again on the way back
    (:func:`family_of`). ``variants`` **adds** the abstain-bearing second ask for a family to the
    *same* request (as ``<family>/abstain/<qid>``) instead of sending it as a second call — the
    family's own questions are still asked, because the row's distribution and its ``none_p`` come
    from the two different answers (``sckevtype.jev.probe.rows_of_answer``).
    """
    if model not in KNOWN_MODELS:
        raise RequestError(f"model {model!r} is not one of {KNOWN_MODELS}; GET /v1/models is the list")
    if not families:
        raise RequestError("a batch needs at least one family")
    digests = {family: state_digest(records[family]) for family in families}
    if len(set(digests.values())) != 1:
        raise RequestError(f"the batched families do not share one state: "
                           f"{ {family: digest[:12] for family, digest in digests.items()} }")

    questions: dict[str, Any] = {}
    for family in families:
        variant = (variants or {}).get(family)
        sources = [(records[family]["questions"], "")] + ([(variant, "abstain")] if variant else [])
        for source, label in sources:
            if not source:
                raise RequestError(f"family {family!r} contributes no question to the batch")
            for qid, question in source.items():
                key = BATCH_SEPARATOR.join(part for part in (family, label, str(qid)) if part)
                if key in questions:
                    raise RequestError(f"two questions would share the id {key!r} in one batch")
                questions[key] = question_payload(question)
    request = {"state": state_text(records[families[0]]), "model": model, "questions": questions}
    check_request(request)
    return request


def record_request(record: Mapping[str, Any], model: str = DEFAULT_MODEL,
                   questions: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The ``POST /v1/systemone`` body for one labelled record.

    ``questions`` overrides the record's own map (this is how the abstain variant is sent); it
    defaults to the record's questions as they are. Every question of the record goes into *one*
    request — the API is designed for it (``parallel questions``: 13 questions in one call are
    12.2x cheaper and 10.0x faster than 13 calls, with no change in answers), and ``marker_noul``
    records carry 30 of them.
    """
    if model not in KNOWN_MODELS:
        raise RequestError(f"model {model!r} is not one of {KNOWN_MODELS}; GET /v1/models is the list")
    source = record["questions"] if questions is None else questions
    if not source:
        raise RequestError("a request needs at least one question")
    request = {"state": state_text(record), "model": model,
               "questions": {qid: question_payload(question) for qid, question in source.items()}}
    check_request(request)
    return request


def abstain_request(record: Mapping[str, Any], model: str = DEFAULT_MODEL) -> dict[str, Any] | None:
    """The second ask: the same record with the abstain option added to every choice question that lacks one.

    ``None`` when every choice question already offers one. The option itself is produced by
    :func:`sckevtype.eval.run.with_abstain` — the same function the local evaluation CLI calls — so
    the two models are asked the *same second question* and their axis-1 statistics are comparable.
    Re-implementing the injection here would be a second source of truth for the one option the
    whole of axis 1 is about.
    """
    from ..eval.run import with_abstain

    variants = {qid: variant for qid, question in record["questions"].items()
                if (variant := with_abstain(question)) is not None}
    if not variants:
        return None
    return record_request(record, model, questions=variants)


def request_digest(request: Mapping[str, Any]) -> str:
    """sha256 of the request as canonical JSON — the pin a raw response is stored against."""
    return payload_digest(dict(request))


# --------------------------------------------------------------------------- the guard


def check_request(request: Mapping[str, Any]) -> dict[str, Any]:
    """Re-derive from the built payload what :func:`record_request` claims, and refuse the rest.

    Four things are checked, each read off the payload rather than off the code that made it:
    the top-level keys are exactly the contract's; every question carries only allow-listed fields;
    every option count is within the documented 255; and the state is a non-empty string. Returns
    the *measurements* (option counts, character counts) so the report can state that nothing was
    truncated instead of asserting it.
    """
    unknown = set(request) - {"state", "model", "questions"}
    if unknown:
        raise RequestError(f"the request carries keys the contract does not define: {sorted(unknown)}")
    if not isinstance(request.get("state"), str) or not request["state"]:
        raise RequestError("the request's state is not a non-empty string")
    questions = request.get("questions")
    if not isinstance(questions, Mapping) or not questions:
        raise RequestError("the request carries no questions")

    options, characters = {}, len(request["state"])
    for qid, question in questions.items():
        extra = set(question) - set(QUESTION_FIELDS)
        if extra:
            raise RequestError(f"question {qid!r} carries fields that are not on the wire: {sorted(extra)}")
        qtype = question.get("type")
        if qtype not in SERVED_TYPES:
            raise RequestError(f"question {qid!r} has type {qtype!r}, which this probe does not serve")
        criteria = question.get("criteria")
        count = len(criteria) if isinstance(criteria, (Mapping, list)) else 0
        if qtype == "choice" and not 1 <= count <= MAX_OPTIONS:
            raise RequestError(f"question {qid!r} has {count} options; the API takes 1..{MAX_OPTIONS}")
        options[qid] = count
        characters += len(str(question.get("instructions") or "")) + len(repr(criteria))
    return {"questions": len(questions), "options": options, "state_chars": len(request["state"]),
            "question_chars": characters, "state_bytes": len(request["state"].encode("utf-8"))}


def cost_estimate(request: Mapping[str, Any]) -> dict[str, Any]:
    """A deliberately **pessimistic** input-token estimate, for the budget guard.

    One token per character, plus 256: a real request is far cheaper than that (a minimal call came
    back at 440 input tokens for ~470 characters, so the true ratio is about 1.0 — and this repo's
    corpus requests are ~4 kB), but the guard's job is to refuse *before* spending, and an estimate
    that is low in the direction of a bigger bill is not a guard. It never appears in a report: the
    number a report quotes comes from the response's own ``usage.input_tokens``, which is
    authoritative, and :func:`sckevtype.jev.client.JevClient.accounting` accounts from those.

    What the guard can and cannot promise: it refuses to *start* a call whose worst case would cross
    the budget, so the exposure of a run is bounded by the budget plus the cost of one call already
    in flight. Nothing short of a hard quota upstream can do better, and this is why the per-call cap
    (:func:`~sckevtype.jev.client.JevClient.__init__`\\ 's ``max_call_usd``) exists beside it.
    """
    measured = check_request(request)
    characters = measured["state_chars"] + measured["question_chars"]
    return {"characters": characters, "basis": "1 token per character + 256, deliberately pessimistic",
            "estimated_input_tokens": characters + 256}
