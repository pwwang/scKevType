"""Render one decision-model question into a chat prompt, and parse the answer back to an option.

This module is the *only* place the LLM baseline's prompt is built and the only place its reply is read.
Both are quoted verbatim in `reports/llm-baseline.md`, generated from the constants below, so the report
cannot drift from the code that ran.

Two rules the rest of the pass depends on:

**The content is lifted, never re-rendered.** :func:`option_list` and :func:`question_text` read the
``state``, the ``instructions`` and the ``criteria`` *as they were stored in the raw ask file the
decision models were served from*. Nothing here re-orders options, re-words a definition, truncates the
rank list or invents a label. If the raw file did not carry it, this module raises rather than filling
it in.

**Two prompt variants, both documented, both zero-shot.** ``a`` is minimal -- the state, the question,
the option list, one line asking for one option. ``b`` is explicit -- a system message stating the
output contract, a numbered option list, the answer strings spelled out on their own lines, and a line
forbidding explanation. Both are given to the model in the same serving configuration
(``enable_thinking=False``, greedy decoding); the difference between them is phrasing alone, which is
what makes "we tried two prompts and quote the better one" a fair reading of a general LLM rather than
a handicap.

**Parsing is a cascade, and which rule fired is recorded.** A model that refuses to answer in-format is
a result, not a nuisance: :func:`parse_answer` returns ``None`` with a named reason, the scorer keeps
the row, and the report prints the parse-failure rate beside the accuracy.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

#: The two prompt variants. Both are zero-shot; neither carries a worked example.
PROMPT_VARIANTS = ("a", "b")

#: The system message of variant ``b``. Variant ``a`` has no system message at all.
SYSTEM_B = (
    "You answer multiple-choice questions about single-cell RNA-seq data. "
    "Reply with exactly one option, copied character for character from the option list you are given. "
    "Do not explain, do not add punctuation, and do not answer with anything else."
)

#: The last line of variant ``a``'s user message.
ASK_A = (
    "Reply with exactly one of the option names above, copied character for character. "
    "Output the option name and nothing else."
)

#: The last lines of variant ``b``'s user message. ``{keys}`` is the option keys, one per line.
ASK_B = (
    "Answer with exactly one of the following strings, copied character for character:\n"
    "{keys}\n"
    "Output that string alone, with no explanation, no quotes and no trailing punctuation."
)


class RenderError(RuntimeError):
    """A question that cannot be rendered as written, named rather than repaired."""


# --------------------------------------------------------------------------- lifted content


def option_list(question: Mapping[str, Any]) -> list[tuple[str, str]]:
    """The question's options in the order the raw request stored them: ``[(key, rubric), ...]``.

    ``choice`` questions carry an ordered ``criteria`` map of option name to definition; ``noul``
    questions carry a two-entry map over ``true``/``false``. The order is the JSON's own insertion
    order, which is part of the corpus bytes -- this class of model is order-sensitive, and re-sorting
    would silently ask a different question than the one the decision models were scored on.
    """
    criteria = question.get("criteria")
    if not isinstance(criteria, Mapping) or not criteria:
        raise RenderError(f"a {question.get('type')!r} question needs a non-empty criteria map")
    return [(str(key), "" if value is None else str(value)) for key, value in criteria.items()]


def question_text(question: Mapping[str, Any]) -> str:
    """The question's own words, lifted from ``instructions``.

    A ``choice`` question's instructions are a string. A ``noul`` question's are the three-field map
    the corpus stores (``question`` / ``candidate`` / ``definition``); they are laid out here in that
    order and nothing is added or dropped.
    """
    instructions = question.get("instructions")
    if isinstance(instructions, str):
        if not instructions.strip():
            raise RenderError("the instructions are empty")
        return instructions
    if isinstance(instructions, Mapping):
        parts = []
        for field in ("question", "candidate", "definition"):
            if field not in instructions:
                raise RenderError(f"a noul question's instructions are missing {field!r}")
            parts.append(f"{field.capitalize()}: {instructions[field]}")
        return "\n".join(parts)
    raise RenderError(f"unsupported instructions of type {type(instructions).__name__}")


# --------------------------------------------------------------------------- the two prompts


def user_message(state: str, question: Mapping[str, Any], variant: str) -> str:
    """The user turn for one question, in variant ``a`` or ``b``.

    The state is the raw request's own string, byte for byte, and it opens the message in both
    variants: a reader of the report can diff this against the raw ask file and see no edit.
    """
    if variant not in PROMPT_VARIANTS:
        raise RenderError(f"unknown prompt variant {variant!r}; known: {PROMPT_VARIANTS}")
    if not isinstance(state, str) or not state:
        raise RenderError("the state must be the raw request's own non-empty string")

    options = option_list(question)
    text = question_text(question)

    if variant == "a":
        rendered = "\n".join(f"- {key}: {rubric}" for key, rubric in options)
        return f"{state}\n\n{text}\n\nOptions:\n{rendered}\n\n{ASK_A}"

    rendered = "\n".join(f"{i}. {key} - {rubric}" for i, (key, rubric) in enumerate(options, 1))
    keys = "\n".join(key for key, _ in options)
    return (f"{state}\n\n{text}\n\nThe candidate options are:\n{rendered}\n\n"
            + ASK_B.format(keys=keys))


def messages(state: str, question: Mapping[str, Any], variant: str) -> list[dict[str, str]]:
    """The chat messages for one question. Variant ``b`` prepends a system turn; variant ``a`` does not."""
    out: list[dict[str, str]] = []
    if variant == "b":
        out.append({"role": "system", "content": SYSTEM_B})
    out.append({"role": "user", "content": user_message(state, question, variant)})
    return out


# --------------------------------------------------------------------------- parsing


_LEADER = re.compile(r"^\s*(?:[-*•]|\(?[a-z0-9]{1,3}[.)])\s+", re.IGNORECASE)
_QUOTES = " \t\r\n\"'`“”‘’*_.,;:"


def _norm(text: str) -> str:
    """Case-fold and strip the decoration a chat model wraps a short answer in."""
    return text.strip().strip(_QUOTES).strip().casefold()


def _word_boundary_hits(reply: str, keys: Sequence[str]) -> list[str]:
    """Option keys that occur in ``reply`` as a whole word, longest first, each at most once.

    Whole-word matching is what keeps ``T cell`` from firing inside ``regulatory T cell`` -- but the
    *nesting* case is real, so a reply naming both is reported as ambiguous rather than resolved.
    """
    hits = []
    for key in keys:
        pattern = r"(?<![A-Za-z0-9])" + re.escape(key.casefold()) + r"(?![A-Za-z0-9])"
        if re.search(pattern, reply.casefold()):
            hits.append(key)
    hits.sort(key=len, reverse=True)
    return hits


def parse_answer(completion: str, option_keys: Sequence[str]) -> tuple[str | None, str]:
    """Map a completion onto one of ``option_keys``. Returns ``(key_or_None, rule)``.

    The cascade, in order, with the rule name that fired recorded on every row so a reader can see how
    much leniency the headline number rests on:

    ``exact``
        the whole reply, case-folded and stripped of quotes/backticks/terminal punctuation, is an
        option key.
    ``leader``
        the same, after removing one leading list marker (``-``, ``1.``, ``A)``).
    ``first_line``
        the first non-empty line of the reply, under the two rules above. A model that answers and then
        explains is still read as having answered.
    ``substring``
        exactly one option key occurs in the reply as a whole word. Two or more (the nesting case) is
        **not** resolved -- it is reported as ``ambiguous``.
    ``no_option``
        no option key occurs at all: the model answered something that is not on the ballot.
    ``empty``
        the model produced nothing.
    """
    if completion is None or not completion.strip():
        return None, "empty"
    keys = list(option_keys)
    folded = {_norm(k): k for k in keys}

    def try_exact(text: str) -> str | None:
        return folded.get(_norm(text))

    hit = try_exact(completion)
    if hit is not None:
        return hit, "exact"
    stripped = _LEADER.sub("", completion.strip(), count=1)
    hit = try_exact(stripped)
    if hit is not None:
        return hit, "leader"
    first = next((line for line in completion.splitlines() if line.strip()), "")
    if first and first != completion:
        hit = try_exact(first)
        if hit is not None:
            return hit, "first_line"
        hit = try_exact(_LEADER.sub("", first.strip(), count=1))
        if hit is not None:
            return hit, "first_line"
    hits = _word_boundary_hits(completion, keys)
    if len(hits) == 1:
        return hits[0], "substring"
    if len(hits) > 1:
        return None, "ambiguous"
    return None, "no_option"


def parse_rule_is_strict(rule: str) -> bool:
    """Whether a rule is an *exact* reading of the ballot rather than a rescue.

    The report prints the accuracy under ``exact``+``leader``+``first_line`` (strict) beside the
    headline, so the size of the substring rescue is visible instead of assumed.
    """
    return rule in ("exact", "leader", "first_line")
