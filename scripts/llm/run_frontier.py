#!/usr/bin/env python3
"""Ask frontier chat models the decision models' own questions through OpenRouter.

Two things make this script different from :mod:`scripts.llm.run_llm` and they are both deliberate:

**It is one question per request, not one record per forward.** A Kev rung answers a whole record's
questions in one pass; a hosted chat model is asked them one at a time. So the unit of a row here is a
question ``key``, and the output is an append-only JSONL keyed by that key rather than a tree of
per-record files. A rerun reads the keys already answered and skips them, so a preemption, a rate-limit
storm or a budget stop costs only the questions in flight. The run manifest records ``complete: false``
whenever any planned key is missing -- a partial run is not allowed to look finished.

**The prompt is the baseline's variant B, built by the baseline's own code.** :func:`render.messages`
is imported, not re-implemented, and so is :func:`render.parse_answer` -- the ``exact``-first cascade the
Qwen baselines were scored with. The only thing this file decides about a prompt is which model reads it.

Cost is measured, never estimated from a tokenizer: every response's ``usage`` block
(``prompt_tokens`` / ``completion_tokens``) is priced with the constants in :data:`MODELS`, which are the
per-million-token list prices verified against ``GET /api/v1/models`` at startup. ``--budget-usd`` is a
hard stop on that measured total: once the next request's worst-case reservation would cross the budget,
no further request is dispatched and the rows already in hand are kept.

Reasoning tokens are a first-class part of the measurement, because both target models report
``reasoning.mandatory: true`` in ``/api/v1/models`` -- they cannot be told not to think, their thinking
is billed at the output rate, and on most providers it counts against ``max_tokens``. A small
``--max-tokens`` therefore returns ``finish_reason: "length"`` with empty ``content`` while the
reasoning tokens are still charged. That is recorded per row (``usage.reasoning_tokens``,
``finish_reason``) instead of being hidden inside a mean.

    # realtime, 20 sampled questions, both models, hard stop at $2
    python scripts/llm/run_frontier.py --asks reports/llm-asks-v4-census.jsonl \\
        reports/llm-asks-v4-noul.jsonl --out reports/frontier/dryrun \\
        --mode realtime --limit-per-file 10 --max-tokens 2048 --reasoning low \\
        --budget-usd 2 --max-requests 40

    # batch: builds the exact JSON body that would be POSTed and writes it to disk; it does NOT
    # submit unless --confirm-batch is also given.
    python scripts/llm/run_frontier.py ... --mode batch --plan-only

Batch protocol, read from https://openrouter.ai/docs/batch-quickstart on 2026-10-21:

* submit ``POST https://openrouter.ai/api/v1/batches`` with a JSON object whose fields are
  ``endpoint`` (``/v1/chat/completions``), ``model``, optional ``provider`` / ``completion_window``
  (``24h`` is the only value) and finally ``requests`` -- an array of ``{custom_id, body}``. The
  top-level ``requests`` **must be serialized last** or the API returns 400, because it stream-parses
  the body to accept very large arrays. There is no JSONL upload: the array is inline.
* the 202 reply carries ``id`` and ``status: "validating"``;
* poll ``GET https://openrouter.ai/api/v1/batches/:id``; terminal statuses are ``completed``,
  ``failed``, ``expired``, ``cancelled``, and a completed batch returns its results **inline** in that
  same response (``results[].custom_id``, ``results[].response.body`` or ``results[].error``). There is
  no separate results-download endpoint;
* batch runs on the model's ``:batch`` endpoint variant and is billed at that variant's price; results
  are deleted 30 days after creation (``410 Gone`` afterwards);
* ``GET /api/v1/batches?limit=&after=&status=`` lists batches, ``DELETE /api/v1/batches/:id`` purges one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import threading
import time
import urllib.parse
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render import PROMPT_VARIANTS, messages, parse_answer  # noqa: E402
from score import STRICT_RULES  # noqa: E402  (the baseline's own definition of a strict parse)

API_BASE = "https://openrouter.ai/api/v1"
CHAT_PATH = "/chat/completions"
BATCHES_PATH = "/batches"
MODELS_DOC_URL = "https://openrouter.ai/api/v1/models"
BATCH_DOC_URL = "https://openrouter.ai/docs/batch-quickstart"

#: One entry per model this harness will spend money on. Nothing outside this table is ever contacted:
#: an unknown ``--models`` value is an error, not a substitution. Prices are USD per million tokens and
#: were read from ``GET /api/v1/models`` on 2026-10-21 (``--verify-models`` re-checks them live).
#: ``long_context`` is gpt-6-astra's published step-up: from 272k prompt tokens on, the rate doubles
#: (``min_prompt_tokens: 272000`` in the live pricing object). No question in the v4 census is anywhere
#: near that, but pricing it wrong silently would be a false cheap number, so it is priced.
MODELS: dict[str, dict] = {
    "anthropic/claude-opus-5.5": {
        "label": "Claude Opus 5.5",
        "batch_id": "anthropic/claude-opus-5.5:batch",
        "context_length": 1_000_000,
        "realtime": {"prompt": 4.0, "completion": 20.0},
        "batch": {"prompt": 2.0, "completion": 10.0},
        "long_context": None,
    },
    "openai/gpt-6-astra": {
        "label": "GPT-6 Astra",
        "batch_id": "openai/gpt-6-astra:batch",
        "context_length": 1_050_000,
        "realtime": {"prompt": 10.0, "completion": 50.0},
        "batch": {"prompt": 5.0, "completion": 25.0},
        "long_context": {
            "min_prompt_tokens": 272_000,
            "realtime": {"prompt": 20.0, "completion": 75.0},
            "batch": {"prompt": 10.0, "completion": 37.5},
        },
    },
}

DEFAULT_CONCURRENCY = 8
DEFAULT_MAX_TOKENS = 64
DEFAULT_TIMEOUT = (20.0, 900.0)


class HarnessError(RuntimeError):
    """A condition this script refuses to paper over: a bad key, an unknown model, a rejected config."""


# --------------------------------------------------------------------------- small utilities


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def safe_tag(name: str) -> str:
    """A filesystem-safe tag for a model slug (``a/b:c`` -> ``a_b-c``)."""
    return name.replace("/", "_").replace(":", "-")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_api_key(env_path: Path) -> str:
    """Read ``OPENROUTER_API_KEY`` from the environment, falling back to the repo's ``.env``.

    The value is never printed, logged, echoed into a command line or written to any output file; the
    caller reports only its presence and length.
    """
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if key:
        return key
    if env_path.is_file():
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, _, value = line.partition("=")
            if name.strip() in ("OPENROUTER_API_KEY", "export OPENROUTER_API_KEY"):
                value = value.strip().strip('"').strip("'")
                if value:
                    os.environ["OPENROUTER_API_KEY"] = value
                    return value
    raise HarnessError(
        "OPENROUTER_API_KEY is not set and was not found in the repo .env; "
        "refusing to run without credentials (no model is substituted, nothing is spent)")


def approx_tokens(text: str) -> float:
    """A deliberately pessimistic token estimate, used only to reserve against the budget.

    3.2 characters per token is below every tokenizer's real ratio for this corpus (English prose,
    digit-heavy gene lists), so the reservation over-states cost rather than under-stating it. The
    number that matters -- what was actually charged -- always comes from the response's ``usage``.
    """
    return len(text) / 3.2


# --------------------------------------------------------------------------- the ask files


@dataclass
class Task:
    """One question, resolved to its prompt. ``key`` is unique across the ask files."""

    key: str
    ask: dict
    model: str
    messages: list[dict]
    prompt_chars: int
    prompt_sha256: str
    option_keys: list[str]


def record_key(ask: dict) -> str:
    return f"{ask['dataset']}/{ask['cell']}/{ask['family']}"


def load_asks(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    seen: dict[str, str] = {}
    for path in paths:
        if not path.is_file():
            raise HarnessError(f"ask file not found: {path}")
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                ask = json.loads(line)
                if ask["key"] in seen:
                    raise HarnessError(
                        f"duplicate ask key {ask['key']!r} in {path} and {seen[ask['key']]}: "
                        "the output is keyed by it, so a duplicate would be silently deduplicated")
                seen[ask["key"]] = str(path)
                ask["_ask_file"] = str(path)
                rows.append(ask)
    return rows


def _largest_remainder(weights: dict, n: int, seed: str) -> dict:
    """Proportional integer allocation of ``n`` seats by ``weights``, largest remainder, hash tie-break."""
    total = sum(weights.values())
    if total <= 0:
        return {k: 0 for k in weights}
    exact = {k: n * v / total for k, v in weights.items()}
    quota = {k: int(x) for k, x in exact.items()}
    left = n - sum(quota.values())
    if left > 0:
        order = sorted(exact, key=lambda k: (-(exact[k] - quota[k]), sha256_text(f"{seed}:{k}")))
        for key in order[:left]:
            quota[key] += 1
    return quota


def stratified_pick(rows: list[dict], n: int, seed: str, strata: tuple[str, ...]) -> list[dict]:
    """Pick ``n`` rows, proportional to the strata mix at every level, deterministically.

    Proportional rather than round-robin because prompt length -- the thing that drives cost -- is a
    property of the dataset (state length) and of the family (how many options are printed). The
    allocation is *hierarchical*: ``n`` is split across ``strata[0]`` first, then each group's seats are
    split across ``strata[1]`` inside it. One flat largest-remainder pass over the combined
    (family, dataset) cells starves small cells -- a first version of this function gave ``panel_miss``
    (1,043 of 9,113 choice questions) zero seats in an 18-question sample -- and a family that is
    silently absent from a cost probe is a family whose cost was not measured.

    Ties are broken by ``sha1(seed:key)``, so the same seed always yields the same sample and a reader
    of the report can re-derive it.
    """
    if n <= 0 or n >= len(rows):
        return list(rows)

    def take(subset: list[dict], budget: int, depth: int) -> list[dict]:
        if budget <= 0:
            return []
        ordered = sorted(subset, key=lambda r: sha256_text(f"{seed}:{r['key']}"))
        if budget >= len(ordered) or depth >= len(strata):
            return ordered[:budget]
        groups: dict = defaultdict(list)
        for row in subset:
            groups[row[strata[depth]]].append(row)
        if len(groups) == 1:
            return take(subset, budget, depth + 1)
        quota = _largest_remainder({k: len(v) for k, v in groups.items()}, budget, seed)
        picked: list[dict] = []
        for key in sorted(groups, key=str):
            picked.extend(take(groups[key], quota.get(key, 0), depth + 1))
        return picked

    picked = take(rows, n, 0)
    picked.sort(key=lambda r: r["key"])
    return picked


def build_tasks(asks: list[dict], model: str, variant: str) -> list[Task]:
    tasks: list[Task] = []
    for ask in asks:
        msgs = messages(ask["state"], ask, variant)
        text = "\n\n".join(m["content"] for m in msgs)
        tasks.append(Task(
            key=ask["key"], ask=ask, model=model, messages=msgs,
            prompt_chars=len(text), prompt_sha256=sha256_text(text),
            option_keys=[str(k) for k in ask["criteria"]],
        ))
    return tasks


# --------------------------------------------------------------------------- pricing


def price_for(model: str, mode: str, prompt_tokens: int) -> tuple[dict, str]:
    """The (prompt, completion) rates that apply to this request, and which tier named them."""
    spec = MODELS[model]
    long_ctx = spec.get("long_context")
    if long_ctx and prompt_tokens >= long_ctx["min_prompt_tokens"]:
        return long_ctx[mode], f"long-context (>={long_ctx['min_prompt_tokens']} prompt tokens)"
    return spec[mode], "standard"


def cost_usd(model: str, mode: str, prompt_tokens: int, completion_tokens: int) -> tuple[float, str]:
    rates, tier = price_for(model, mode, prompt_tokens)
    usd = (prompt_tokens * rates["prompt"] + completion_tokens * rates["completion"]) / 1_000_000
    return usd, tier


def verify_model_table(session: requests.Session, models: list[str]) -> None:
    """Check the constants in :data:`MODELS` against the live catalogue before spending anything."""
    try:
        resp = session.get(f"{API_BASE}/models", timeout=60)
        resp.raise_for_status()
        catalogue = {m["id"]: m for m in resp.json()["data"]}
    except Exception as exc:  # noqa: BLE001 -- a network hiccup must not silently skip the check
        raise HarnessError(f"cannot verify models against {MODELS_DOC_URL}: {exc}") from exc
    problems: list[str] = []
    for model in models:
        live = catalogue.get(model)
        if live is None:
            problems.append(f"{model}: not in the live catalogue (ids are checked, not normalised)")
            continue
        for mode, suffix in (("realtime", ""), ("batch", ":batch")):
            variant = catalogue.get(model + suffix)
            if variant is None:
                problems.append(f"{model}{suffix}: no {mode} variant in the live catalogue")
                continue
            live_in = float(variant["pricing"]["prompt"]) * 1_000_000
            live_out = float(variant["pricing"]["completion"]) * 1_000_000
            ours = MODELS[model][mode]
            if abs(live_in - ours["prompt"]) > 1e-6 or abs(live_out - ours["completion"]) > 1e-6:
                problems.append(
                    f"{model}{suffix}: live ${live_in:g}/${live_out:g} per Mtok != "
                    f"table ${ours['prompt']:g}/${ours['completion']:g}")
        fam = (live.get("reasoning") or {})
        if fam.get("mandatory"):
            print(f"  note: {model} reports reasoning.mandatory=true "
                  f"(supported efforts {fam.get('supported_efforts')}); reasoning cannot be switched "
                  f"off, its tokens bill at the output rate and count against max_tokens")
    if problems:
        raise HarnessError("model table does not match the live catalogue:\n  " + "\n  ".join(problems))
    print(f"  live catalogue agrees with the price table for {len(models)} model(s)")


# --------------------------------------------------------------------------- budget and limiting


class Budget:
    """Measured spend plus a hard stop. Nothing is dispatched once ``stopped`` is set."""

    def __init__(self, budget_usd: float, max_requests: int):
        self.budget_usd = float(budget_usd or 0.0)
        self.max_requests = int(max_requests or 0)
        self.spend = 0.0
        self.billed = 0
        self.stopped = False
        self.stop_reason = ""
        self.lock = threading.Lock()

    def try_reserve(self, reserve_usd: float) -> tuple[bool, str]:
        with self.lock:
            if self.stopped:
                return False, self.stop_reason
            if self.max_requests and self.billed >= self.max_requests:
                self.stopped, self.stop_reason = True, f"request cap reached ({self.max_requests})"
                return False, self.stop_reason
            if self.budget_usd and self.spend + reserve_usd > self.budget_usd:
                self.stopped = True
                self.stop_reason = (f"budget guard: measured spend ${self.spend:.4f} + worst-case "
                                    f"reservation ${reserve_usd:.4f} would cross --budget-usd "
                                    f"${self.budget_usd:.4f}")
                return False, self.stop_reason
            return True, ""

    def charge(self, usd: float) -> float:
        with self.lock:
            self.spend += usd
            self.billed += 1
            return self.spend

    def snapshot(self) -> dict:
        with self.lock:
            return {"spend_usd": round(self.spend, 6), "billed_requests": self.billed,
                    "budget_usd": self.budget_usd or None, "max_requests": self.max_requests or None,
                    "stopped": self.stopped, "stop_reason": self.stop_reason or None}


class AdaptiveLimiter:
    """A gate whose width halves on 429/5xx and creeps back up on sustained success."""

    def __init__(self, max_parallel: int):
        self.max_parallel = max(1, max_parallel)
        self.limit = self.max_parallel
        self.active = 0
        self.cond = threading.Condition()
        self.throttles = 0
        self.relaxes = 0

    def acquire(self) -> None:
        with self.cond:
            while self.active >= self.limit:
                self.cond.wait()
            self.active += 1

    def release(self, success: bool = True) -> None:
        with self.cond:
            self.active -= 1
            if not success:
                self.throttles += 1
                new = max(1, self.limit // 2)
                if new != self.limit:
                    self.limit = new
                    print(f"  [limiter] width {self.limit} (after a rate-limit/5xx)", flush=True)
            elif self.limit < self.max_parallel:
                self.relaxes += 1
                if self.relaxes % 10 == 0:
                    self.limit += 1
            self.cond.notify_all()


# --------------------------------------------------------------------------- append-only outputs


class JsonlSink:
    """An append-only JSONL writer: one row per line, flushed and locked, never rewritten."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

    def append(self, row: dict) -> None:
        line = json.dumps(row, ensure_ascii=False)
        with self.lock:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
                fh.flush()


def answered_keys(path: Path) -> dict[str, dict]:
    """Every key already in an answers file, last row winning. The resume set."""
    out: dict[str, dict] = {}
    if not path.is_file():
        return out
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            out[row["key"]] = row
    return out


# --------------------------------------------------------------------------- the client


@dataclass
class Reply:
    ok: bool
    status: int
    body: dict | None
    error: str
    latency_ms: float
    attempts: int


class OpenRouter:
    """The two endpoints this harness uses, with a retry policy that cannot double-count a row.

    A row is written only after a ``200`` whose body parsed as a chat completion; a retry therefore
    replaces a failed attempt rather than adding a second row, and every attempt -- failed or not --
    lands in the attempts ledger with the HTTP status that produced it.
    """

    def __init__(self, api_key: str, retries: int, base_sleep: float, timeout: tuple[float, float]):
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",  # never printed, never serialised
            "Content-Type": "application/json",
            "X-Title": "scKevType frontier census",
        })
        self.retries = max(0, retries)
        self.base_sleep = base_sleep
        self.timeout = timeout
        self.attempts = 0

    def _backoff(self, attempt: int, resp: requests.Response | None) -> float:
        if resp is not None:
            header = resp.headers.get("Retry-After", "")
            if header.strip().isdigit():
                return min(120.0, float(header.strip()))
        return min(120.0, self.base_sleep * (2 ** attempt)) * (0.6 + 0.8 * random.random())

    def chat(self, payload: dict, what: str) -> Reply:
        last = Reply(False, 0, None, "no attempt made", 0.0, 0)
        for attempt in range(self.retries + 1):
            started = time.time()
            try:
                resp = self.session.post(f"{API_BASE}{CHAT_PATH}", data=json.dumps(payload),
                                         timeout=self.timeout)
            except requests.RequestException as exc:
                last = Reply(False, 0, None, f"{type(exc).__name__}: {exc}", 0.0, attempt + 1)
                sleep = self._backoff(attempt, None)
                print(f"  [retry] {what} transport error ({type(exc).__name__}), {sleep:.1f}s", flush=True)
                time.sleep(sleep)
                continue
            latency_ms = (time.time() - started) * 1000.0
            status = resp.status_code
            if status == 200:
                try:
                    body = resp.json()
                except ValueError as exc:
                    last = Reply(False, status, None, f"unparsable 200 body: {exc}", latency_ms, attempt + 1)
                    time.sleep(self._backoff(attempt, resp))
                    continue
                if body.get("error"):
                    last = Reply(False, status, body, str(body["error"])[:2000], latency_ms, attempt + 1)
                    break
                return Reply(True, status, body, "", latency_ms, attempt + 1)
            text = (resp.text or "")[:2000]
            last = Reply(False, status, None, f"HTTP {status}: {text}", latency_ms, attempt + 1)
            retryable = status == 429 or 500 <= status < 600
            if not retryable:
                break
            sleep = self._backoff(attempt, resp)
            print(f"  [retry] {what} HTTP {status}, {sleep:.1f}s", flush=True)
            time.sleep(sleep)
        return last

    def submit_batch(self, payload: dict) -> Reply:
        started = time.time()
        try:
            resp = self.session.post(f"{API_BASE}{BATCHES_PATH}", data=json.dumps(payload).encode("utf-8"),
                                     timeout=self.timeout)
        except requests.RequestException as exc:
            return Reply(False, 0, None, f"{type(exc).__name__}: {exc}", 0.0, 1)
        latency_ms = (time.time() - started) * 1000.0
        if resp.status_code in (200, 201, 202):
            return Reply(True, resp.status_code, resp.json(), "", latency_ms, 1)
        return Reply(False, resp.status_code, None, f"HTTP {resp.status_code}: {(resp.text or '')[:2000]}",
                     latency_ms, 1)

    def get_batch(self, batch_id: str) -> Reply:
        started = time.time()
        try:
            resp = self.session.get(f"{API_BASE}{BATCHES_PATH}/{urllib.parse.quote(batch_id)}",
                                    timeout=self.timeout)
        except requests.RequestException as exc:
            return Reply(False, 0, None, f"{type(exc).__name__}: {exc}", 0.0, 1)
        latency_ms = (time.time() - started) * 1000.0
        if resp.status_code == 200:
            return Reply(True, 200, resp.json(), "", latency_ms, 1)
        return Reply(False, resp.status_code, None, f"HTTP {resp.status_code}: {(resp.text or '')[:2000]}",
                     latency_ms, 1)


# --------------------------------------------------------------------------- response -> row


def message_text(message: dict) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
            elif isinstance(part, str):
                parts.append(part)
        return "".join(parts)
    return ""


def usage_of(body: dict) -> dict:
    usage = body.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    prompt_details = usage.get("prompt_tokens_details") or {}
    return {
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "reasoning_tokens": int(details.get("reasoning_tokens") or 0),
        "cached_tokens": int(prompt_details.get("cached_tokens") or 0),
        "total_tokens": int(usage.get("total_tokens") or 0),
        "cost_reported_usd": usage.get("cost"),
    }


def make_row(task: Task, body: dict, latency_ms: float, mode: str, max_tokens: int,
             reasoning: dict | None, arm: str) -> dict:
    choice = (body.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content = message_text(message)
    finish_reason = choice.get("finish_reason")
    usage = usage_of(body)
    parsed, rule = parse_answer(content, task.option_keys)
    table_usd, tier = cost_usd(task.model, mode, usage["prompt_tokens"], usage["completion_tokens"])
    #: What the account was actually charged, when OpenRouter reports it. The list-price table is kept
    #: beside it rather than instead of it: measured against a live run the two disagree -- gpt-6-astra
    #: returns a cache-write line item that lifts its input rate from $10 to ~$12.5 per Mtok on a fresh
    #: prefix -- and a projection built on the table alone would understate the bill by a fifth.
    reported = usage.get("cost_reported_usd")
    charged = float(reported) if isinstance(reported, (int, float)) else table_usd
    ask = task.ask
    return {
        "key": task.key,
        "dataset": ask["dataset"], "cell": ask["cell"], "family": ask["family"],
        "question": ask["question"], "type": ask["type"],
        "model": task.model, "mode": mode, "arm": arm,
        "prompt_variant": "b",
        "messages": task.messages, "prompt_sha256": task.prompt_sha256,
        "option_keys": task.option_keys,
        "max_tokens": max_tokens, "reasoning": reasoning,
        "completion": content,
        "reasoning_text_present": bool(message.get("reasoning")),
        "finish_reason": finish_reason,
        "parsed": parsed, "parse_rule": rule, "parsed_ok": int(parsed is not None),
        "strict_rule": int(rule in STRICT_RULES),
        "hit_cap": int(bool(finish_reason == "length")
                       or (usage["completion_tokens"] >= max_tokens)),
        "usage": usage,
        "cost_usd": round(charged, 9), "cost_usd_table": round(table_usd, 9),
        "cost_basis": "reported" if isinstance(reported, (int, float)) else "table",
        "price_tier": tier,
        "latency_ms": round(latency_ms, 1),
        "request_id": body.get("id"),
        "provider": body.get("provider"), "service_tier": body.get("service_tier"),
        "created_at": utc_now(),
        "source_file": ask.get("source_file"), "source_sha256": ask.get("source_sha256"),
        "request_sha256": ask.get("request_sha256"),
        "served_model": ask.get("served_model"),
        "response_model": body.get("model"),
    }


# --------------------------------------------------------------------------- realtime


def reserve_usd(task: Task, mode: str, max_tokens: int) -> float:
    """A worst-case cost for one request: every prompt token plus the whole output cap.

    Used only to ask the budget guard whether a request may be dispatched; the charge that actually
    lands in ``budget.spend`` is always the ``usage``-derived one. The token estimate is deliberately
    pessimistic (see :func:`approx_tokens`) so the guard errs toward stopping early.
    """
    estimate = approx_tokens("\n\n".join(m["content"] for m in task.messages))
    rates, _ = price_for(task.model, mode, int(estimate))
    return (estimate * rates["prompt"] + max_tokens * rates["completion"]) / 1_000_000


def run_realtime(args, tasks: list[Task], client: OpenRouter, budget: Budget, sinks: dict,
                 attempt_sinks: dict, limiter: AdaptiveLimiter) -> dict:
    answered = {model: answered_keys(sink.path) for model, sink in sinks.items()}
    pending = [t for t in tasks if t.key not in answered.get(t.model, {})]
    print(f"  {len(tasks)} planned, {len(tasks) - len(pending)} already answered, {len(pending)} to go")
    if not pending:
        return {"dispatched": 0, "answered": 0, "failed": 0, "skipped_present": len(tasks)}

    stopped_for = ""
    lock = threading.Lock()
    counters = Counter()
    warned: set[str] = set()
    started = time.time()

    def work(task: Task) -> None:
        nonlocal stopped_for
        allowed, why = budget.try_reserve(reserve_usd(task, "realtime", args.max_tokens))
        if not allowed:
            with lock:
                stopped_for = why
            return
        payload: dict = {"model": task.model, "messages": task.messages,
                         "max_tokens": args.max_tokens, "stream": False}
        if args.reasoning_config is not None:
            payload["reasoning"] = args.reasoning_config
        limiter.acquire()
        ok = True
        try:
            reply = client.chat(payload, what=f"{safe_tag(task.model)}:{task.key[-40:]}")
            ok = reply.ok
        finally:
            limiter.release(success=ok)
        if reply.ok and reply.body:
            row = make_row(task, reply.body, reply.latency_ms, "realtime", args.max_tokens,
                           args.reasoning_config, args.arm)
            total = budget.charge(row["cost_usd"])
            sinks[task.model].append(row)
            with lock:
                counters["answered"] += 1
                counters["parse_ok"] += row["parsed_ok"]
                counters["cap_hits"] += row["hit_cap"]
                if (row["cost_basis"] == "reported" and row["model"] not in warned
                        and row["cost_usd_table"] > 0
                        and abs(row["cost_usd"] - row["cost_usd_table"]) / row["cost_usd_table"] > 0.02):
                    warned.add(row["model"])
                    print(f"  [price] {row['model']}: charged ${row['cost_usd']:.6f} but the list-price "
                          f"table says ${row['cost_usd_table']:.6f} "
                          f"({row['cost_usd'] / row['cost_usd_table']:.3f}x, provider="
                          f"{row['provider']}); the measured charge is what the guard accumulates",
                          flush=True)
                n = counters["answered"]
                if n % args.log_every == 0 or n == len(pending):
                    print(f"  [{safe_tag(task.model)}] {n}/{len(pending)} answered  "
                          f"parsed={counters['parse_ok']}/{n}  cap_hits={counters['cap_hits']}  "
                          f"measured spend ${total:.4f}", flush=True)
        else:
            with lock:
                counters["failed"] += 1
            print(f"  [fail] {safe_tag(task.model)} {task.key[-60:]} :: {reply.error[:300]}", flush=True)
        attempt_sinks[task.model].append({
            "ts": utc_now(), "key": task.key, "model": task.model, "mode": "realtime",
            "arm": args.arm, "attempts": reply.attempts, "http_status": reply.status,
            "ok": reply.ok, "billed": reply.ok, "latency_ms": round(reply.latency_ms, 1),
            "error": reply.error[:2000] if not reply.ok else "",
        })

    threads = []
    queue = list(pending)
    index = threading.Lock()
    cursor = {"i": 0}

    def pump() -> None:
        while True:
            with index:
                if cursor["i"] >= len(queue) or stopped_for:
                    return
                task = queue[cursor["i"]]
                cursor["i"] += 1
            work(task)

    # one dispatcher thread per allowed concurrency slot; the limiter throttles the HTTP calls
    for _ in range(min(limiter.max_parallel, len(pending))):
        t = threading.Thread(target=pump, daemon=True)
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    elapsed = time.time() - started
    print(f"  realtime finished in {elapsed:.1f}s: answered={counters['answered']} "
          f"failed={counters['failed']} spend=${budget.spend:.4f}")
    return {"dispatched": counters["answered"] + counters["failed"],
            "answered": counters["answered"], "failed": counters["failed"],
            "skipped_present": len(tasks) - len(pending), "wall_s": round(elapsed, 1),
            "stopped_for": stopped_for or None}


# --------------------------------------------------------------------------- batch


def batch_payload(tasks: list[Task], model: str, args) -> tuple[dict, dict[str, str]]:
    """The exact JSON body to POST, with ``requests`` serialised last (the API 400s otherwise).

    ``custom_id`` is a short ASCII index rather than the ask key: the key contains ``/`` and can be
    long, while ``custom_id`` only has to be unique within the batch. The mapping is persisted beside
    the plan so a resumed collection can turn results back into keys without re-deriving anything.
    """
    requests_out: list[dict] = []
    mapping: dict[str, str] = {}
    for i, task in enumerate(tasks):
        custom_id = f"q{i:06d}"
        mapping[custom_id] = task.key
        body: dict = {"messages": task.messages, "max_tokens": args.max_tokens}
        if args.reasoning_config is not None:
            body["reasoning"] = args.reasoning_config
        requests_out.append({"custom_id": custom_id, "body": body})
    payload: dict = {"endpoint": "/v1/chat/completions", "model": args.batch_model_id or model}
    if args.completion_window:
        payload["completion_window"] = args.completion_window
    payload["requests"] = requests_out
    assert list(payload)[-1] == "requests", "the requests array must be serialised last"
    return payload, mapping


def run_batch(args, tasks: list[Task], models: list[str], out_dir: Path, budget: Budget,
              attempt_sinks: dict) -> dict:
    """Plan (and, only with --confirm-batch, submit) one batch per model, then poll and collect."""
    summary: dict = {"per_model": {}}
    for model in models:
        mine = [t for t in tasks if t.model == model]
        answers_path = out_dir / f"answers-{safe_tag(model)}.jsonl"
        have = answered_keys(answers_path)
        pending = [t for t in mine if t.key not in have]
        model_dir = out_dir / "batch" / safe_tag(model)
        model_dir.mkdir(parents=True, exist_ok=True)
        print(f"  [{safe_tag(model)}] {len(mine)} planned, {len(have)} already answered, "
              f"{len(pending)} pending")
        info: dict = {"model": model, "planned": len(mine), "already_answered": len(have),
                      "pending": len(pending), "chunks": []}
        if not pending:
            summary["per_model"][model] = info
            continue
        chunks = [pending[i:i + args.batch_chunk] for i in range(0, len(pending), args.batch_chunk)]
        for index, chunk in enumerate(chunks):
            payload, mapping = batch_payload(chunk, model, args)
            plan_path = model_dir / f"plan-{index:03d}.json"
            plan_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            plan_path.write_bytes(plan_bytes)
            (model_dir / f"plan-{index:03d}.map.json").write_text(
                json.dumps({"_model": model, "_mode": "batch", "_arm": args.arm,
                            "_max_tokens": args.max_tokens, "_reasoning": args.reasoning_config,
                            "_batch_model_id": args.batch_model_id or model,
                            "_plan_sha256": hashlib.sha256(plan_bytes).hexdigest(),
                            "_plan_path": str(plan_path), "_requests": len(chunk),
                            "map": mapping}, indent=2) + "\n", encoding="utf-8")
            state_path = model_dir / f"state-{index:03d}.json"
            state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {
                "model": model, "chunk": index, "requests": len(chunk),
                "plan_sha256": hashlib.sha256(plan_bytes).hexdigest(),
                "submitted_at": None, "batch_id": None, "status": None, "collected": False,
            }
            info["chunks"].append({"chunk": index, "requests": len(chunk),
                                   "plan_path": str(plan_path), "batch_id": state.get("batch_id"),
                                   "status": state.get("status"), "collected": state.get("collected")})
            print(f"    chunk {index}: {len(chunk)} requests, plan {plan_path.name} "
                  f"({len(plan_bytes) / 1e6:.2f} MB), sha256 {state['plan_sha256'][:12]}")
            if not args.confirm_batch:
                print("    not submitted (--confirm-batch absent): plan written, nothing spent")
                continue
            # A batch is a single paid commitment, so the guard has to run *before* the POST: the
            # worst case for every request in the chunk, priced at the batch rate, must fit.
            reserve = sum(reserve_usd(t, "batch", args.max_tokens) for t in chunk)
            allowed, why = budget.try_reserve(reserve)
            if not allowed:
                print(f"    NOT submitting: {why}")
                info["chunks"][-1]["error"] = why
                break
            by_key = {t.key: t for t in chunk}
            key_by_custom = mapping
            # --- submit (only with --confirm-batch)
            if not state.get("batch_id"):
                resp = client_of(args).submit_batch(payload)
                if not resp.ok or not resp.body:
                    raise HarnessError(
                        f"batch submit failed for {model}: {resp.error}\n"
                        f"(no fallback to realtime: that would multiply the price by two)")
                state.update({"batch_id": resp.body.get("id"), "status": resp.body.get("status"),
                              "submitted_at": utc_now(), "submit_response": resp.body})
                state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
                print(f"    submitted: batch {state['batch_id']} status={state['status']}")
            # --- resume: a chunk whose results were already collected is not re-polled
            if state.get("collected"):
                print(f"    chunk {index} already collected at {state.get('collected_at')}")
                continue
            # --- poll until a terminal status (completed/failed/expired/cancelled)
            batch_obj: dict | None = None
            while True:
                resp = client_of(args).get_batch(state["batch_id"])
                if not resp.ok or not resp.body:
                    raise HarnessError(f"batch poll failed for {state['batch_id']}: {resp.error}")
                batch_obj = resp.body
                state["status"] = batch_obj.get("status")
                state["request_counts"] = batch_obj.get("request_counts")
                state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
                print(f"    batch {state['batch_id']} status={state['status']} "
                      f"counts={batch_obj.get('request_counts')}", flush=True)
                if state["status"] in ("completed", "failed", "expired", "cancelled"):
                    break
                time.sleep(max(5.0, args.poll_seconds))
            if state["status"] != "completed":
                info["chunks"][-1]["error"] = f"terminal status {state['status']}"
                break
            results = batch_obj.get("results") or []
            if not results:
                print("    WARNING: a completed batch returned no results array; nothing collected")
                break
            if budget.budget_usd and budget.spend >= budget.budget_usd:
                print("    budget already spent; results left uncollected for a later resume")
                break
            sink = JsonlSink(answers_path)
            unmapped: list[str] = []
            collected = failed = 0
            for result in results:
                custom_id = result.get("custom_id")
                key = key_by_custom.get(custom_id)
                task = by_key.get(key) if key else None
                if task is None:
                    unmapped.append(str(custom_id))
                    continue
                response = result.get("response") or {}
                body = response.get("body") or {}
                if result.get("error") or response.get("status_code") != 200 or not body:
                    failed += 1
                    attempt_sinks[model].append({
                        "ts": utc_now(), "key": key, "model": model, "mode": "batch",
                        "batch_id": state["batch_id"], "ok": False, "billed": False,
                        "http_status": response.get("status_code"),
                        "error": str(result.get("error"))[:2000]})
                    continue
                row = make_row(task, body, 0.0, "batch", args.max_tokens, args.reasoning_config,
                               args.arm)
                budget.charge(row["cost_usd"])
                sink.append(row)
                collected += 1
            print(f"    collected {collected} row(s), {failed} per-request error(s), "
                  f"{len(unmapped)} unmapped; batch usage {batch_obj.get('usage')}")
            if unmapped:
                print(f"    WARNING: {len(unmapped)} result(s) had no custom_id mapping: {unmapped[:3]}")
            state["collected"] = True
            state["collected_at"] = utc_now()
            state["batch_usage"] = batch_obj.get("usage")
            state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
            info["chunks"][-1].update({"status": state["status"], "collected": True,
                                       "collected_rows": collected,
                                       "request_errors": failed,
                                       "batch_usage": state.get("batch_usage")})
        summary["per_model"][model] = info
    return summary


_CLIENT: OpenRouter | None = None


def client_of(args) -> OpenRouter:  # noqa: ARG001 -- kept as the single accessor for the session
    assert _CLIENT is not None, "the API client is initialised in main()"
    return _CLIENT


# --------------------------------------------------------------------------- score-tree export


def export_score_tree(out_dir: Path, model: str, tasks: list[Task],
                      expected_by_record: dict[str, int] | None = None) -> dict:
    """Write a ``run_llm.py``-shaped ask tree so ``score.py`` can score this run unchanged.

    Only a record whose every question *in the ask file* has an answer is written: a half-answered
    record in the tree would be scored as if the missing questions did not exist, which is exactly the
    kind of partial pass this harness is supposed to make impossible. ``expected_by_record`` therefore
    counts the record's questions in the ask file, not in this run's sample -- a sampled subset of a
    16-question noul record is not a complete record.
    """
    answers = answered_keys(out_dir / f"answers-{safe_tag(model)}.jsonl")
    by_record: dict[str, list[dict]] = defaultdict(list)
    records_expected: dict[str, int] = Counter()
    for task in tasks:
        rec = record_key(task.ask)
        records_expected[rec] += 1
        if task.key in answers:
            by_record[rec].append(answers[task.key])
    if expected_by_record:
        records_expected.update(expected_by_record)
    tree = out_dir / "score-tree" / safe_tag(model)
    raw = tree / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    written = skipped = 0
    for rec, rows in sorted(by_record.items()):
        if len(rows) < records_expected[rec]:
            skipped += 1
            continue
        rows.sort(key=lambda r: r["question"])
        questions = [{
            "key": r["key"], "dataset": r["dataset"], "cell": r["cell"], "family": r["family"],
            "question": r["question"], "type": r["type"], "option_keys": r["option_keys"],
            "prompt": json.dumps(r["messages"], ensure_ascii=False), "completion": r["completion"],
            "parsed": r["parsed"], "parse_rule": r["parse_rule"],
            "input_tokens": r["usage"]["prompt_tokens"], "output_tokens": r["usage"]["completion_tokens"],
            "state_tokens": None, "question_tokens": None,
            "source_file": r.get("source_file"), "source_sha256": r.get("source_sha256"),
            "request_sha256": r.get("request_sha256"),
        } for r in rows]
        payload = {"ask": rec, "dataset": rows[0]["dataset"], "cell": rows[0]["cell"],
                   "family": rows[0]["family"], "variant": "b", "thinking": False,
                   "model": model, "revision": "",
                   "n_questions": len(questions),
                   "latency_ms": sum(r["latency_ms"] for r in rows),
                   "input_tokens_processed": sum(r["usage"]["prompt_tokens"] for r in rows),
                   "input_tokens_state_once": None,
                   "output_tokens": sum(r["usage"]["completion_tokens"] for r in rows),
                   "questions": questions}
        (raw / f"{safe_tag(rec)}.json").write_text(json.dumps(payload, ensure_ascii=False) + "\n",
                                                   encoding="utf-8")
        written += 1
    info = {"model": model, "tree": str(tree), "records_written": written,
            "records_skipped_incomplete": skipped,
            "note": "score.py scores raw/*.json; only records with every question answered are here"}
    (tree / "export.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(f"  exported {written} complete record(s) to {tree}; skipped {skipped} incomplete")
    return info


# --------------------------------------------------------------------------- main


def reasoning_config(spec: str, models: list[str], catalogue_reasoning: dict[str, dict]) -> dict | None:
    """``--reasoning`` -> the ``reasoning`` object, refusing a config the live models would 400 on."""
    if spec in ("", "model-default"):
        return None
    if spec == "off":
        for model in models:
            fam = catalogue_reasoning.get(model) or {}
            if fam.get("mandatory"):
                raise HarnessError(
                    f"--reasoning off is not available for {model}: the live catalogue reports "
                    f"reasoning.mandatory=true (supported efforts {fam.get('supported_efforts')}). "
                    "Pass an effort level instead, or drop the flag to take the model's own default.")
        return {"enabled": False}
    if spec not in ("minimal", "low", "medium", "high", "xhigh", "max"):
        raise HarnessError(f"unknown --reasoning {spec!r}; use off, model-default or an effort level")
    return {"effort": spec}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--asks", nargs="+", required=True, help="one or more ask JSONL files")
    ap.add_argument("--out", required=True, help="output directory (created if absent)")
    ap.add_argument("--models", nargs="+", default=list(MODELS),
                    help="model slugs; must be keys of the model table (default: both frontier models)")
    ap.add_argument("--mode", choices=("realtime", "batch"), default="realtime")
    ap.add_argument("--variant", choices=list(PROMPT_VARIANTS), default="b",
                    help="prompt variant; 'b' is what the frontier comparison is defined on")
    ap.add_argument("--limit-per-file", type=int, default=0,
                    help="sample this many questions from each ask file (0 = every question)")
    ap.add_argument("--sample-seed", default="frontier-v1")
    ap.add_argument("--stratify", default="family,dataset",
                    help="comma-separated fields the sample is made proportional to")
    ap.add_argument("--families", default="", help="comma-separated family filter")
    ap.add_argument("--arm", default="default", help="a label recorded on every row (e.g. cap64)")
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    ap.add_argument("--reasoning", default="model-default",
                    help="off | model-default | minimal|low|medium|high|xhigh|max")
    ap.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    ap.add_argument("--retries", type=int, default=6)
    ap.add_argument("--base-sleep", type=float, default=2.0)
    ap.add_argument("--budget-usd", type=float, default=2.0,
                    help="hard stop on measured spend (0 disables the guard -- discouraged)")
    ap.add_argument("--max-requests", type=int, default=0,
                    help="hard cap on billed (HTTP 200) requests; 0 disables")
    ap.add_argument("--log-every", type=int, default=1)
    ap.add_argument("--batch-chunk", type=int, default=2000,
                    help="requests per submitted batch. The docs state no maximum, so this is a "
                         "judgement: one measured plan of 1,000 requests was 4.6 MB, so 2,000 is ~9 MB "
                         "of JSON per POST -- big enough that polling and re-submission are not the "
                         "dominant cost, small enough that a rejected or expired batch loses 2,000 "
                         "answers rather than 10,000")
    ap.add_argument("--batch-model-id", default="",
                    help="override the model field of the batch payload (e.g. the ':batch' slug)")
    ap.add_argument("--completion-window", default="24h",
                    help="the only value the API accepts; recorded in the payload")
    ap.add_argument("--poll-seconds", type=float, default=60.0)
    ap.add_argument("--confirm-batch", action="store_true",
                    help="actually POST the batch; without it the plan is written and nothing is spent")
    ap.add_argument("--plan-only", action="store_true",
                    help="write the batch plan and exit before any submission")
    ap.add_argument("--verify-models", dest="verify_models", action="store_true", default=True,
                    help="check the price table against GET /api/v1/models (default: on)")
    ap.add_argument("--no-verify-models", dest="verify_models", action="store_false")
    ap.add_argument("--export-score-tree", action="store_true",
                    help="after the run, write a score.py-compatible tree for complete records only")
    args = ap.parse_args()

    for model in args.models:
        if model not in MODELS:
            raise SystemExit(f"refusing unknown model {model!r}; known: {sorted(MODELS)} "
                             "(a substitution would silently change the comparison)")
    if args.variant != "b":
        print("WARNING: the frontier comparison is defined on variant b; --variant a is exploratory only")

    repo_root = Path(__file__).resolve().parents[2]
    key = load_api_key(repo_root / ".env")
    print(f"api key     : present={bool(key)} len={len(key)} (value never printed)")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = [Path(p) for p in args.asks]
    rows = load_asks(paths)
    wanted = {f.strip() for f in args.families.replace("|", ",").split(",") if f.strip()}
    if wanted:
        rows = [r for r in rows if r["family"] in wanted]
    print(f"ask files   : {', '.join(str(p) for p in paths)}")
    print(f"questions   : {len(rows)} in scope after the family filter "
          f"({dict(Counter(r['family'] for r in rows))})")

    sampled: list[dict] = []
    for path in paths:
        mine = [r for r in rows if r["_ask_file"] == str(path)]
        if not mine:
            continue
        picked = stratified_pick(mine, args.limit_per_file, args.sample_seed,
                                 tuple(s.strip() for s in args.stratify.split(",") if s.strip()))
        print(f"  {Path(path).name}: {len(picked)}/{len(mine)} sampled  "
              f"{dict(Counter(r['family'] for r in picked))}  "
              f"datasets={dict(Counter(r['dataset'] for r in picked))}")
        sampled.extend(picked)
    sampled.sort(key=lambda r: r["key"])

    session = requests.Session()
    catalogue_reasoning: dict[str, dict] = {}
    if args.verify_models:
        verify_model_table(session, args.models)
        try:
            data = session.get(f"{API_BASE}/models", timeout=60).json()
            catalogue_reasoning = {m["id"]: (m.get("reasoning") or {}) for m in data["data"]}
        except Exception:  # noqa: BLE001 -- the reasoning note is advisory only
            pass
    args.reasoning_config = reasoning_config(args.reasoning, args.models, catalogue_reasoning)

    global _CLIENT
    _CLIENT = OpenRouter(key, retries=args.retries, base_sleep=args.base_sleep,
                         timeout=DEFAULT_TIMEOUT)

    planned = len(sampled) * len(args.models)
    print(f"mode        : {args.mode}   models={args.models}")
    print(f"sample      : {len(sampled)} questions x {len(args.models)} model(s) = {planned} requests")
    print(f"decode      : max_tokens={args.max_tokens}  reasoning={args.reasoning} "
          f"({json.dumps(args.reasoning_config) if args.reasoning_config else 'the model default'})")
    print(f"guards      : budget=${args.budget_usd}  max billed requests={args.max_requests or 'none'}  "
          f"concurrency={args.concurrency}")
    if not args.budget_usd:
        print("WARNING: --budget-usd 0 disables the spend guard")

    budget = Budget(args.budget_usd, args.max_requests)
    sinks = {m: JsonlSink(out_dir / f"answers-{safe_tag(m)}.jsonl") for m in args.models}
    attempt_sinks = {m: JsonlSink(out_dir / f"attempts-{safe_tag(m)}.jsonl") for m in args.models}

    tasks: list[Task] = []
    for model in args.models:
        tasks.extend(build_tasks(sampled, model, args.variant))

    started = time.time()
    stats: dict
    if args.mode == "realtime":
        limiter = AdaptiveLimiter(args.concurrency)
        stats = run_realtime(args, tasks, _CLIENT, budget, sinks, attempt_sinks, limiter)
        stats["limiter"] = {"width": limiter.limit, "throttles": limiter.throttles}
    else:
        stats = run_batch(args, tasks, args.models, out_dir, budget, attempt_sinks)
        if args.plan_only:
            print("--plan-only: no submission was made")

    # ---- completeness: a partial run must not look finished
    per_model: dict[str, dict] = {}
    for model in args.models:
        done = answered_keys(sinks[model].path)
        mine = [t for t in tasks if t.model == model]
        missing = [t.key for t in mine if t.key not in done]
        per_model[model] = {
            "planned": len(mine), "answered": len(mine) - len(missing), "missing": len(missing),
            "complete": not missing, "missing_examples": missing[:5],
        }
    manifest = {
        "what_this_is": "frontier chat models answering the decision models' own questions "
                        "(variant b, the baseline's own prompt and parse rule) through OpenRouter",
        "started_at": utc_now(), "wall_s": round(time.time() - started, 1),
        "args": {k: v for k, v in vars(args).items()
                 if k not in ("reasoning_config",) and not k.startswith("_")},
        "reasoning_config": args.reasoning_config,
        "ask_files": [str(p) for p in paths],
        "sample_seed": args.sample_seed, "stratify": args.stratify,
        "questions_sampled": len(sampled),
        "sample_by_family": dict(Counter(r["family"] for r in sampled)),
        "sample_by_dataset": dict(Counter(r["dataset"] for r in sampled)),
        "budget": budget.snapshot(),
        "stats": stats,
        "per_model": per_model,
        "complete": all(v["complete"] for v in per_model.values()),
        "batch_protocol_doc": BATCH_DOC_URL,
        "model_table": {m: MODELS[m] for m in args.models},
    }
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    manifest_path = out_dir / f"run-{stamp}-{safe_tag(args.arm)}-{args.mode}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    exported = None
    if args.export_score_tree:
        expected = Counter(record_key(r) for r in rows)
        exported = [export_score_tree(out_dir, m, tasks, expected) for m in args.models]

    print(f"measured spend ${budget.spend:.6f} over {budget.billed} billed request(s)")
    for model, info in per_model.items():
        print(f"  {model}: answered {info['answered']}/{info['planned']}  complete={info['complete']}")
    if budget.stop_reason:
        print(f"STOPPED: {budget.stop_reason}")
    print(f"wrote {manifest_path}")
    if exported:
        print(f"score trees: {[e['tree'] for e in exported]}")
    print(f"FRONTIER_RUN_DONE rc=0 complete={manifest['complete']} "
          f"stopped={'yes' if budget.stopped else 'no'} spend=${budget.spend:.4f}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except HarnessError as exc:
        print(f"FRONTIER_RUN_FAILED rc=2 :: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
