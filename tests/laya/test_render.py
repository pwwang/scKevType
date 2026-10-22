"""The re-render's one load-bearing contract: option order.

``markers[i]``, ``target[i]`` and a probability read back off the model are all indexed by this
order, so if it disagrees with what ``laya.common.render_options`` writes, every score in the pilot
is silently attributed to the wrong option. The tokenizer and ``build_sequence`` are exercised by the
cluster job -- what is checked here is the ordering rule and the analytic count around it.
"""

from __future__ import annotations

import pytest

from sckevtype.laya.render import (
    FRAMING_TOKENS,
    SHIPPED,
    WIDE,
    RenderConfig,
    RenderError,
    head_tokens,
    label_index,
    option_keys,
    target_of,
)


def _choice(criteria: dict, label: str):
    return {"type": "choice", "criteria": criteria, "label": label, "instructions": "which one?"}


def _noul(label: int):
    return {"type": "noul", "criteria": {}, "label": label, "instructions": {"a": True, "b": False}}


def test_choice_keys_follow_the_criteria_dicts_own_order():
    assert option_keys(_choice({"CL:1": "T cell", "CL:2": "B cell", "CL:3": "NK cell"}, "CL:2")) == \
        ["CL:1", "CL:2", "CL:3"]


def test_noul_keys_are_the_fixed_semantic_order_regardless_of_how_the_dict_is_written():
    assert option_keys(_noul(1)) == ["false", "true"]


def test_a_type_with_no_defined_order_raises_rather_than_guessing_one():
    with pytest.raises(RenderError, match="option order"):
        option_keys({"type": "score", "label": 0.5})


def test_the_label_index_is_the_labels_position_among_the_keys():
    question = _choice({"CL:1": "T cell", "CL:2": "B cell"}, "CL:2")
    assert label_index(question, option_keys(question)) == 1


def test_a_label_that_is_not_an_option_raises_rather_than_defaulting_to_zero():
    question = _choice({"CL:1": "T cell", "CL:2": "B cell"}, "CL:9")
    with pytest.raises(RenderError, match="CL:9"):
        label_index(question, option_keys(question))


def test_noul_labels_are_indices_into_the_semantic_order():
    assert label_index(_noul(0), ["false", "true"]) == 0
    assert label_index(_noul(1), ["false", "true"]) == 1


def test_the_target_is_one_hot_at_the_label_index():
    question = _choice({"a": "A", "b": "B", "c": "C"}, "b")
    assert target_of(question, option_keys(question)) == [0.0, 1.0, 0.0]


def test_head_tokens_counts_a_marker_per_option_plus_the_instruction():
    assert head_tokens(7, [[1, 2, 3], [4], [5, 6]]) == 7 + (1 + 3) + (1 + 1) + (1 + 2)
    assert head_tokens(0, []) == 0


def test_the_two_disclosed_configurations_are_the_ones_the_report_names():
    assert SHIPPED.as_dict() == {"max_len": 1024, "head_max_len": 256}
    assert WIDE.as_dict() == {"max_len": 8192, "head_max_len": 2048}
    assert RenderConfig(8192, 2048) == WIDE
    assert FRAMING_TOKENS == 4  # the [CLS]/[SEP]/[SEP]/[SEP] the count adds around head + state
