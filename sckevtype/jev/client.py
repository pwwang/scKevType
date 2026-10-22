"""The transport: one budgeted, retrying client for ``POST /v1/systemone``.

Mirrors :class:`kev.predictors.RemotePredictor` (record in, ``{"probabilities": {qid: {key: p}}}``
out, server-reported model id captured) and :class:`kev.predictors.JevPredictor` (every call counted
against a spend cap, ``accounting()`` in the same key shape so the two reports read alike). What it
adds over both is that the budget is enforced per request as well as in total: the smoke run's
instruction is "if a single call could cost more than $0.10, stop", and a guard that only checks the
running total cannot see the one call that breaks it.

Retry policy, and the line between the two kinds of failure:

* ``429``/``529``/``5xx``/timeouts/connection resets — the *hosted* side's problem. Retried with
  exponential backoff, honouring ``Retry-After`` when the vendor sends one (the docs say the SDKs
  do; this client is not an SDK, so it does it here).
* any other ``4xx`` — the *request's* problem. Raised immediately with the body, because a 422 names
  the field it rejected and retrying it would only spend the budget on a bug.

The API key is read from the environment (``JEV_KEY``), passed to the process that needs it, and
never written into any artifact this package produces: error text is redacted through
:func:`redact` before it can reach a log, a report or an exception message.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from .request import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    check_request,
    cost_estimate,
    option_keys,
    record_request,
    request_digest,
)

#: Where the key lives. ``~/.bashrc`` defines it for interactive shells only, so the runner reads it
#: from the environment it is started with — never from a file this package writes.
KEY_ENV = "JEV_KEY"

#: The vendor's own list price (``docs.typesafe.ai/models``, read 2026-09-29): **$42 per billion
#: input tokens**, i.e. $0.042 per million. Output tokens are free — a System One answer is a
#: readout, not a generation — so only input is charged and only input is accounted. This follows
#: ``kev.predictors.PRICE_PER_MILLION``; the two agree, which is the point of quoting a source.
PRICE_PER_MILLION_INPUT = 0.042

#: Statuses worth retrying. ``529`` is the vendor's "overloaded"; ``408``/``425`` are transport-level.
RETRY_STATUSES = (408, 409, 425, 429, 500, 502, 503, 504, 529)

#: The largest total budget this client will accept. A spend cap is only a cap if a typo in the
#: command line cannot lift it, so the money ask in a flag is bounded here as well: $1 was the
#: smoke run's ceiling, the batched full-population run is authorised at $5, and a *larger* number
#: than the authorised ceiling is a mistake rather than a decision — pass ``budget_usd`` explicitly
#: after changing this constant if a bigger run is ever authorised.
MAX_BUDGET_USD = 5.0


class JevError(RuntimeError):
    """A call that did not produce a usable answer, named rather than swallowed."""


class BudgetExceeded(JevError):
    """The spend guard fired. Raised *before* the call it would have paid for."""


def redact(text: str, key: str | None) -> str:
    """The text with the key removed. Applied to every error path before it can be logged."""
    return text.replace(key, "<JEV_KEY>") if key else text


def _retry_after(headers: Mapping[str, str]) -> float | None:
    """``Retry-After`` in seconds, when the response carries a parseable one."""
    raw = None
    for name, value in (headers or {}).items():
        if str(name).lower() == "retry-after":
            raw = value
            break
    if raw is None:
        return None
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return None


def default_transport(url: str, payload: bytes, headers: Mapping[str, str], timeout: float) -> tuple[int, dict]:
    """One request over ``urllib``; returns ``(status, body)`` with the body parsed when it is JSON."""
    request = urllib.request.Request(url, data=payload, method="POST", headers=dict(headers))
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", "replace")
            return response.status, _parse(raw, response.headers)
    except urllib.error.HTTPError as error:
        raw = error.read().decode("utf-8", "replace")
        return error.code, _parse(raw, error.headers)


def _parse(raw: str, headers: Any) -> dict:
    try:
        body: Any = json.loads(raw)
    except json.JSONDecodeError:
        body = {"detail": raw[:2000]}
    return {"body": body if isinstance(body, dict) else {"body": body}, "headers": dict(headers or {})}


class JevClient:
    """A budgeted client. Call it with a record and get back the predictor-shaped answer.

    ``transport`` is injectable so the tests can drive the retry ladder and the validators without
    a network; the default is :func:`default_transport`.
    """

    def __init__(self, key: str | None = None, base_url: str = DEFAULT_BASE_URL, model: str = DEFAULT_MODEL,
                 budget_usd: float = 1.0, max_calls: int = 2000, max_call_usd: float = 0.10,
                 timeout: float = 180.0, retries: int = 4, backoff: float = 2.0,
                 transport: Any = None, sleep: Any = time.sleep) -> None:
        self.key = key if key is not None else os.environ.get(KEY_ENV, "")
        if not self.key:
            raise JevError(f"no API key: set {KEY_ENV} in the environment this process starts with")
        if not 0 < budget_usd:
            raise JevError(f"budget_usd must be positive; got {budget_usd}")
        if budget_usd > MAX_BUDGET_USD:
            raise JevError(f"budget_usd is capped at ${MAX_BUDGET_USD} by MAX_BUDGET_USD; got {budget_usd}")
        self.base_url, self.model = base_url, model
        self.budget_usd, self.max_calls, self.max_call_usd = budget_usd, max_calls, max_call_usd
        self.timeout, self.retries, self.backoff = timeout, retries, backoff
        self.transport, self.sleep = transport or default_transport, sleep
        self.started_at = datetime.now(UTC).isoformat(timespec="seconds")
        # The counters are read and written from a thread pool, so every mutation of them goes
        # through one lock. The guard is inside it too: two threads must not each be told the last
        # five dollars are still unspent.
        self._lock = threading.Lock()
        self.calls = self.input_tokens = self.output_tokens = self.retries_after_hosted = 0
        self.served_models: dict[str, int] = {}
        self.latencies_ms: list[float] = []

    # --------------------------------------------------------------- accounting

    @property
    def estimated_usd(self) -> float:
        """Spend so far, at the vendor's list price. Input tokens only — output is free."""
        return self.input_tokens * PRICE_PER_MILLION_INPUT / 1e6

    def accounting(self) -> dict[str, Any]:
        """The recap block, in ``kev.predictors.JevPredictor.accounting``'s key shape."""
        with self._lock:
            return {
                "vendor": "TypeSafe System One", "endpoint": self.base_url, "requested_model": self.model,
                "served_models": dict(self.served_models), "started_at": self.started_at,
                "calls": self.calls, "input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
                "retries_after_hosted_failure": self.retries_after_hosted,
                "median_latency_ms": (sorted(self.latencies_ms)[len(self.latencies_ms) // 2]
                                      if self.latencies_ms else None),
                "listed_input_usd_per_million": PRICE_PER_MILLION_INPUT, "estimated_usd": self.estimated_usd,
                "budget_usd": self.budget_usd, "max_call_usd": self.max_call_usd, "max_calls": self.max_calls,
                "price_source": "https://docs.typesafe.ai/models, read 2026-09-29 ($42/Btok input, output free)",
            }

    def _guard(self, request: Mapping[str, Any]) -> dict[str, Any]:
        """Refuse before spending: the caps, then the worst case for this one request."""
        estimate = cost_estimate(request)
        call_usd = estimate["estimated_input_tokens"] * PRICE_PER_MILLION_INPUT / 1e6
        with self._lock:
            if self.calls >= self.max_calls:
                raise BudgetExceeded(f"request cap reached ({self.max_calls} calls); nothing further was sent")
            if call_usd > self.max_call_usd:
                raise BudgetExceeded(f"one request could cost up to ${call_usd:.4f}, over the ${self.max_call_usd} "
                                     f"single-call cap; stopping before sending it")
            if self.estimated_usd + call_usd > self.budget_usd:
                raise BudgetExceeded(f"${self.estimated_usd:.4f} spent and this request could cost ${call_usd:.4f}, "
                                     f"over the ${self.budget_usd} budget; stopping before sending it")
        return estimate

    # --------------------------------------------------------------- asking

    def ask(self, request: Mapping[str, Any]) -> dict[str, Any]:
        """Send one request, retrying hosted-side failures. Returns the parsed response body.

        ``request`` is validated by :func:`sckevtype.jev.request.check_request` before it is sent, so
        an unknown field is a local error rather than a 422 that costs a round trip.
        """
        check_request(request)
        self._guard(request)
        payload = json.dumps(request).encode("utf-8")
        headers = {"content-type": "application/json", "authorization": f"Bearer {self.key}"}
        last = "no attempt was made"
        for attempt in range(self.retries + 1):
            start = time.perf_counter()
            try:
                status, response = self.transport(self.base_url, payload, headers, self.timeout)
            except Exception as error:  # noqa: BLE001 — a transport failure is the retryable kind
                status, response = None, {"body": {}, "headers": {}}
                last = redact(f"{type(error).__name__}: {error}", self.key)
            latency = 1000.0 * (time.perf_counter() - start)
            with self._lock:
                self.latencies_ms.append(latency)

            if status == 200:
                return self._record_usage(response["body"], request)
            if status is not None and status not in RETRY_STATUSES:
                raise JevError(f"HTTP {status}: {redact(json.dumps(response['body'])[:600], self.key)}")
            last = f"HTTP {status}: {redact(json.dumps(response['body'])[:300], self.key)}"
            if attempt == self.retries:
                break
            with self._lock:
                self.retries_after_hosted += 1
            wait = _retry_after(response.get("headers") or {})
            if wait is None:
                wait = self.backoff ** attempt
            self.sleep(min(wait, 60.0))
        raise JevError(f"gave up after {self.retries + 1} attempts; last failure {last}")

    def _record_usage(self, body: Mapping[str, Any], request: Mapping[str, Any]) -> dict[str, Any]:
        """Count the call, pin the served model, and check the response answers what was asked."""
        usage = body.get("usage") or {}
        if usage.get("input_tokens") is None:
            # The same rule as JevPredictor: an unauditable call is refused, because a spend cap
            # that cannot see the spend is not a cap.
            raise JevError("response carried no usage.input_tokens; the spend cannot be accounted")
        model = str(body.get("model") or "unstated")
        validate_answers(body, request)
        with self._lock:
            self.calls += 1
            self.input_tokens += int(usage["input_tokens"])
            self.output_tokens += int(usage.get("output_tokens") or 0)
            self.served_models[model] = self.served_models.get(model, 0) + 1
            if self.calls == 1 or self.calls % 100 == 0:
                print(f"  jev: {self.calls} calls, {self.input_tokens} in-tokens, "
                      f"${self.estimated_usd:.4f}", flush=True)
        return dict(body)

    # --------------------------------------------------------------- predictor shape

    def __call__(self, record: Mapping[str, Any], questions: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """``record -> {"probabilities": {qid: {key: p}}, ...}`` — the shape the rest of the repo reads.

        ``kev.benchmark.prediction_rows`` and ``sckevtype.eval.run.score_partition`` both consume
        exactly this, so a Jev prediction and a checkpoint prediction are scored by one code path.
        A ``noul`` answer is expanded to both keys (``false`` = 1 - ``noul``) as ``RemotePredictor``
        does, even though the API returns only the probability of yes.
        """
        return self.respond(record_request(record, self.model, questions=questions))

    def respond(self, request: Mapping[str, Any]) -> dict[str, Any]:
        """Send a request that is already built and shape its answer like :meth:`__call__`.

        The batched ask is several families' questions in one body, which
        :func:`~sckevtype.jev.request.record_request` cannot build from one record; this is the
        entry point for it. One sender, one guard, one validation and one accounting path for both
        asks — the alternative (a second sender beside this one) is a second spend path that a
        budget guard would not see.
        """
        start = time.perf_counter()
        body = self.ask(request)
        latency = 1000.0 * (time.perf_counter() - start)
        probabilities = {}
        for qid, payload in request["questions"].items():
            answer = body["answers"][qid]
            if payload["type"] == "noul":
                yes = float(answer["noul"])
                probabilities[qid] = {"false": 1.0 - yes, "true": yes}
            else:
                # In the *question's* option order, not the order the API happened to serialize its
                # JSON in: the two are the same distribution, and one of them is reproducible.
                # (validate_answers has already established that the key sets agree.)
                probabilities[qid] = {str(key): float(answer["probabilities"][key])
                                      for key in option_keys(payload)}
        return {"probabilities": probabilities, "answers": body["answers"], "served_model": body.get("model"),
                "request_digest": request_digest(request), "latency_ms": latency,
                "input_tokens": int((body.get("usage") or {}).get("input_tokens") or 0),
                "output_tokens": int((body.get("usage") or {}).get("output_tokens") or 0)}


def validate_answers(body: Mapping[str, Any], request: Mapping[str, Any]) -> dict[str, Any]:
    """The response answers every question asked, under the same ids, with a usable distribution.

    Three ways a wrong answer would otherwise be scored as a real one: an id the request never sent
    (or a missing one), an option set that differs from the question's, and probabilities that are
    not a distribution. The vendor's returned probabilities sit on a 0.01 grid, so they need not sum
    to exactly 1 — the sum is *recorded* (``raw_probability_sum``) rather than assumed, and a
    distribution that is off by more than the grid is refused.
    """
    answers = body.get("answers")
    if not isinstance(answers, Mapping):
        raise JevError("the response carries no answers object")
    if set(answers) != set(request["questions"]):
        raise JevError(f"answer ids differ from request ids: {sorted(set(answers) ^ set(request['questions']))}")
    for qid, payload in request["questions"].items():
        answer = answers[qid]
        if answer.get("type") != payload["type"]:
            raise JevError(f"{qid}: answer type {answer.get('type')!r} does not match the question's {payload['type']!r}")
        keys = option_keys(payload)
        if payload["type"] == "noul":
            value = answer.get("noul")
            if not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
                raise JevError(f"{qid}: noul answer {value!r} is not a probability")
            continue
        probabilities = answer.get("probabilities")
        if not isinstance(probabilities, Mapping) or list(probabilities) == []:
            raise JevError(f"{qid}: a choice answer needs a non-empty probabilities map")
        if set(probabilities) != set(keys):
            raise JevError(f"{qid}: option keys differ from the question's: {sorted(set(probabilities) ^ set(keys))}")
        values = [float(probabilities[key]) for key in keys]
        if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in values):
            raise JevError(f"{qid}: probabilities are not finite numbers in [0, 1]")
        total = sum(values)
        if total <= 0 or abs(total - 1.0) > max(1e-9, len(keys) * 0.005 + 1e-8):
            raise JevError(f"{qid}: probabilities sum to {total}, further from 1 than a {len(keys)}-option "
                           f"0.01 grid can explain")
    return {"questions": len(answers)}


def normalise(probabilities: Mapping[str, Any], keys: Sequence[str]) -> tuple[list[float], float]:
    """``(p, raw sum)`` for one question, renormalised exactly as ``kev.benchmark.validate_distribution`` does.

    The raw sum travels with the row because Jev rounds to a 0.01 grid; renormalising without
    recording what was renormalised would hide how far the grid moved a probability.
    """
    p = [float(probabilities[key]) for key in keys]
    total = sum(p)
    if total <= 0:
        raise JevError(f"a distribution summed to {total}")
    return [value / total for value in p], total


__all__ = [
    "KEY_ENV",
    "PRICE_PER_MILLION_INPUT",
    "RETRY_STATUSES",
    "BudgetExceeded",
    "JevClient",
    "JevError",
    "default_transport",
    "normalise",
    "redact",
    "validate_answers",
]
