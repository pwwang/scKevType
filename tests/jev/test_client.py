"""What the client can get wrong without a network noticing.

The transport, the retry ladder and the budget guard are all driven here with a fake transport, so
the failures that would otherwise only show up as a strange number in a report — a retried 422, an
answer scored against the wrong option order, a spend cap that only checks the total, a key echoed
into an error message — are checked before anything is spent.

The fake speaks the client's own transport protocol: ``(status, {"body": ..., "headers": ...})``,
which is what :func:`sckevtype.jev.client.default_transport` returns.
"""

from __future__ import annotations

import json

import pytest

from sckevtype.jev import client as C
from sckevtype.jev import request as R

KEY = "test-key-not-a-real-one"
REGISTRY = "A lymphocyte. positive markers: CD3E, IL7R."


# --------------------------------------------------------------------------- fixtures


def a_record(**overrides) -> dict:
    """A record in the corpus's own shape, small enough to read."""
    record = {
        "state": "CD3E 1 | IL7R 2 | MS4A1 3 | ...",
        "questions": {
            "label": {"type": "choice", "src": "panel_choice", "label": "T cell",
                      "instructions": "Which candidate cell type does the state support?",
                      "criteria": {"T cell": REGISTRY, "B cell": "positive markers: MS4A1, CD79A.",
                                   "other": "None of the above"}},
        },
        "_meta": {"dataset_id": "d", "family": "panel_choice", "cell_id": "c1", "group_id": "d/c1",
                  "donor_id": "n1", "panel_id": "p1", "variant": "clean", "source": "real-v2"},
    }
    return {**record, **overrides}


def ok(record: dict, probabilities: dict, input_tokens: int = 1234, model: str = "jev-1.13.0"):
    """A 200 carrying one choice answer, in the vendor's own response shape.

    Deliberately does not assert that the keys match the question's: a test below sends a *wrong* key
    set to check that the client refuses it, and a fixture that refused it first would test nothing.
    """
    body = {"model": model,
            "answers": {"label": {"type": "choice", "choice": next(iter(probabilities)),
                                  "confidence": 0.5, "probabilities": probabilities}},
            "usage": {"input_tokens": input_tokens, "output_tokens": 7}}
    return 200, {"body": body, "headers": {}}


def failed(status: int, detail: str = "something went wrong", headers: dict | None = None):
    """A non-200 carrying the vendor's error shape."""
    return status, {"body": {"detail": detail}, "headers": headers or {}}


class Fake:
    """A transport that plays ``script`` in order (a callable is called; a tuple is returned)."""

    def __init__(self, script):
        self.script = script
        self.calls: list[dict] = []

    def __call__(self, url, payload, headers, timeout):
        self.calls.append(json.loads(payload))
        assert headers["authorization"] == f"Bearer {KEY}"
        assert headers["content-type"] == "application/json"
        step = self.script[min(len(self.calls) - 1, len(self.script) - 1)]
        return step(url, self.calls[-1]) if callable(step) else step


def fake(script) -> Fake:
    return Fake(script)


def a_client(script, **kwargs) -> C.JevClient:
    return C.JevClient(key=KEY, transport=fake(script), sleep=lambda _seconds: None, **kwargs)


# --------------------------------------------------------------------------- the request shape


def test_the_wire_payload_carries_the_questions_and_none_of_the_labels():
    record = a_record()
    body = R.record_request(record)
    assert body["state"] == record["state"] and body["model"] == R.DEFAULT_MODEL
    assert body["questions"]["label"] == {"type": "choice",
                                          "instructions": record["questions"]["label"]["instructions"],
                                          "criteria": record["questions"]["label"]["criteria"]}
    # the audit: every question is inside the allow-list, so `label`/`src` cannot have travelled
    assert all(set(question) <= set(R.QUESTION_FIELDS) for question in body["questions"].values())
    assert all("label" not in question and "src" not in question for question in body["questions"].values())
    assert "_meta" not in json.dumps(body)


def test_the_state_is_the_corpus_string_byte_for_byte():
    record = a_record()
    assert R.record_request(record)["state"] == record["state"]


def test_a_non_string_state_is_refused_rather_than_rendered():
    with pytest.raises(R.RequestError, match="rendered string"):
        R.record_request(a_record(state={"genes": ["CD3E"]}))


def test_a_label_that_is_not_an_option_is_refused_rather_than_scored_as_the_last_one():
    with pytest.raises(R.RequestError, match="not one of this question's options"):
        R.label_index({"type": "choice", "label": "T-cell", "criteria": {"T cell": "x", "B cell": "y"}})


def test_an_unknown_model_is_refused_locally_rather_than_spent_on_a_422():
    with pytest.raises(R.RequestError, match="not one of"):
        R.record_request(a_record(), model="jev-experimental")


def test_a_question_type_the_probe_does_not_serve_raises():
    record = a_record()
    record["questions"]["label"]["type"] = "score"
    with pytest.raises(R.RequestError, match="not served"):
        R.record_request(record)


def test_a_multi_question_record_goes_in_one_request():
    record = a_record()
    record["questions"] = {f"holds:{index}": {"type": "noul", "instructions": "Does this hold?",
                                              "label": bool(index % 2), "criteria": None} for index in range(30)}
    body = R.record_request(record)
    assert len(body["questions"]) == 30 and R.check_request(body)["questions"] == 30


def test_option_counts_are_measured_so_silence_about_truncation_is_not_an_assertion():
    state = "CD3E 1 | IL7R 2 | MS4A1 3 | ..."
    measured = R.check_request(R.record_request(a_record(state=state)))
    assert measured["questions"] == 1 and measured["options"] == {"label": 3}
    assert measured["state_chars"] == len(state) == measured["state_bytes"]
    assert measured["question_chars"] > 0


# --------------------------------------------------------------------------- the predictor shape


def test_a_choice_answer_comes_back_keyed_by_option_name_in_the_questions_order():
    record = a_record()
    probabilities = {"other": 0.10, "T cell": 0.62, "B cell": 0.28}   # a different order on the wire
    prediction = a_client([ok(record, probabilities)])(record)
    assert list(prediction["probabilities"]["label"]) == list(record["questions"]["label"]["criteria"])
    assert prediction["probabilities"]["label"]["T cell"] == 0.62
    assert prediction["served_model"] == "jev-1.13.0" and prediction["input_tokens"] == 1234
    assert prediction["answers"]["label"]["confidence"] == 0.5


def test_a_noul_answer_is_expanded_to_both_keys():
    record = a_record()
    record["questions"] = {"label": {"type": "noul", "instructions": "Does this hold?", "label": True,
                                     "criteria": None}}
    body = {"model": "jev-1.13.0", "answers": {"label": {"type": "noul", "noul": 0.8}},
            "usage": {"input_tokens": 10, "output_tokens": 1}}
    prediction = C.JevClient(key=KEY, transport=fake([(200, {"body": body, "headers": {}})]))(record)
    assert prediction["probabilities"]["label"] == {"false": pytest.approx(0.2), "true": 0.8}


def test_an_answer_under_a_different_option_key_raises_rather_than_being_scored():
    record = a_record()
    with pytest.raises(C.JevError, match="option keys differ"):
        a_client([ok(record, {"other": 0.1, "T cell": 0.7, "B-cell": 0.2})])(record)


def test_a_missing_answer_id_raises():
    record = a_record()
    body = {"model": "m", "answers": {}, "usage": {"input_tokens": 1}}
    with pytest.raises(C.JevError, match="answer ids differ"):
        C.JevClient(key=KEY, transport=fake([(200, {"body": body, "headers": {}})]))(record)


def test_an_answer_whose_type_contradicts_the_question_raises():
    record = a_record()
    body = {"model": "m", "answers": {"label": {"type": "noul", "noul": 0.5}},
            "usage": {"input_tokens": 1}}
    with pytest.raises(C.JevError, match="answer type"):
        C.JevClient(key=KEY, transport=fake([(200, {"body": body, "headers": {}})]))(record)


def test_a_response_without_token_usage_is_refused_so_the_cap_can_see_the_spend():
    record = a_record()
    body = {"model": "m", "answers": {"label": {"type": "choice", "choice": "T cell",
                                                "probabilities": {"T cell": 0.8, "B cell": 0.1, "other": 0.1}}},
            "usage": {}}
    with pytest.raises(C.JevError, match="usage.input_tokens"):
        C.JevClient(key=KEY, transport=fake([(200, {"body": body, "headers": {}})]))(record)


def test_a_0_01_grid_distribution_is_renormalised_and_its_raw_sum_recorded():
    p, total = C.normalise({"a": 0.33, "b": 0.33, "c": 0.33}, ["a", "b", "c"])
    assert total == pytest.approx(0.99) and sum(p) == pytest.approx(1.0)


def test_a_distribution_further_from_one_than_the_grid_explains_is_refused():
    record = a_record()
    with pytest.raises(C.JevError, match="sum to"):
        a_client([ok(record, {"T cell": 0.9, "B cell": 0.9, "other": 0.9})])(record)


# --------------------------------------------------------------------------- retries


def test_a_429_is_retried_and_the_answer_is_the_successful_one():
    record = a_record()
    script = [failed(429, "slow down", {"retry-after": "0"}),
              ok(record, {"T cell": 0.7, "B cell": 0.2, "other": 0.1})]
    client = a_client(script)
    assert client(record)["probabilities"]["label"]["T cell"] == 0.7
    assert client.calls == 1 and client.retries_after_hosted == 1


def test_a_529_is_retried_too_because_the_vendor_documents_it_as_temporary():
    record = a_record()
    client = a_client([failed(529, "overloaded"), ok(record, {"T cell": 0.7, "B cell": 0.2, "other": 0.1})])
    client(record)
    assert client.retries_after_hosted == 1


def test_retry_after_is_honoured_over_the_exponential_ladder():
    slept: list[float] = []
    record = a_record()
    script = [failed(429, "slow down", {"retry-after": "7"}),
              ok(record, {"T cell": 0.7, "B cell": 0.2, "other": 0.1})]
    C.JevClient(key=KEY, transport=fake(script), sleep=slept.append)(record)
    assert slept == [7.0]


def test_a_422_is_not_retried_because_it_names_a_bug_not_a_bad_day():
    record = a_record()
    transport = fake([failed(422, "field 'criteria' must be an object")])
    with pytest.raises(C.JevError, match="HTTP 422"):
        C.JevClient(key=KEY, transport=transport, sleep=lambda _s: None)(record)
    assert len(transport.calls) == 1


def test_a_transport_failure_is_retried_and_then_gives_up_with_a_message():
    record = a_record()

    def transport(url, payload, headers, timeout):
        raise TimeoutError("the read operation timed out")

    with pytest.raises(C.JevError, match="gave up after 3 attempts"):
        C.JevClient(key=KEY, transport=transport, retries=2, sleep=lambda _s: None)(record)


def test_the_key_never_appears_in_an_error():
    record = a_record()
    with pytest.raises(C.JevError) as error:
        C.JevClient(key=KEY, transport=fake([failed(401, f"bad key {KEY}")]), sleep=lambda _s: None)(record)
    assert KEY not in str(error.value) and "<JEV_KEY>" in str(error.value)


# --------------------------------------------------------------------------- the budget


def test_a_single_call_over_the_single_call_cap_is_refused_before_it_is_sent():
    record = a_record()
    transport = fake([ok(record, {"T cell": 1.0, "B cell": 0.0, "other": 0.0})])
    client = C.JevClient(key=KEY, transport=transport, max_call_usd=1e-9, sleep=lambda _s: None)
    with pytest.raises(C.BudgetExceeded, match="single-call cap"):
        client(record)
    assert transport.calls == []


def test_a_call_that_crosses_the_budget_stops_the_run_at_the_next_one():
    """The guard prices a call before making it; it cannot price one it has not made.

    So the honest statement is: a run's exposure is the budget plus at most one call in flight. The
    test asserts exactly that — the call that overshot went through, and *nothing after it did*.
    """
    record = a_record()
    client = C.JevClient(key=KEY, transport=fake([ok(record, {"T cell": 1.0, "B cell": 0.0, "other": 0.0},
                                                             input_tokens=10 ** 9)]),
                         sleep=lambda _s: None)
    client(record)
    assert client.estimated_usd > client.budget_usd
    with pytest.raises(C.BudgetExceeded, match="budget"):
        client(record)
    assert client.calls == 1


def test_the_estimate_the_guard_uses_is_pessimistic_not_optimistic():
    request = R.record_request(a_record())
    measured = R.check_request(request)
    estimate = R.cost_estimate(request)
    assert estimate["estimated_input_tokens"] >= measured["state_chars"] + measured["question_chars"]


def test_the_call_cap_stops_the_run_before_the_next_request():
    record = a_record()
    client = a_client([ok(record, {"T cell": 1.0, "B cell": 0.0, "other": 0.0})], max_calls=1)
    client(record)
    with pytest.raises(C.BudgetExceeded, match="request cap"):
        client(record)
    assert client.calls == 1


def test_spend_is_accounted_at_the_vendors_list_price_from_the_responses_own_usage():
    record = a_record()
    client = a_client([ok(record, {"T cell": 1.0, "B cell": 0.0, "other": 0.0}, input_tokens=1_000_000)])
    client(record)
    assert client.estimated_usd == pytest.approx(C.PRICE_PER_MILLION_INPUT)
    assert client.accounting()["served_models"] == {"jev-1.13.0": 1}
    assert client.accounting()["price_source"].startswith("https://docs.typesafe.ai/models")


def test_a_client_needs_a_key_from_the_environment(monkeypatch):
    monkeypatch.delenv(C.KEY_ENV, raising=False)
    with pytest.raises(C.JevError, match=C.KEY_ENV):
        C.JevClient()
    monkeypatch.setenv(C.KEY_ENV, KEY)
    assert C.JevClient().key == KEY


def test_the_smoke_budget_cannot_be_raised_above_the_instruction():
    with pytest.raises(C.JevError, match="capped at"):
        C.JevClient(key=KEY, budget_usd=10.0)
