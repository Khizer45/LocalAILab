"""Evaluator validity: positive, negative, paraphrase, adversarial, boundary and property-based tests.

A reference answer passing its own evaluator only shows the two agree. These tests check that
*wrong* answers fail, harmless variations pass, and the documented semantics hold for any input.
"""

from __future__ import annotations

import json

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from app.benchmark.evaluators import EVALUATOR_VERSIONS, EVALUATORS, evaluate_spec, spec_version, validate_spec

NUM = {"type": "numeric", "value": 42}
LABEL = {"type": "label", "labels": ["billing", "technical_support", "account"], "answer": "billing"}
CAPITAL = {"type": "contains_any", "keywords": ["Paris"]}
JSON_SPEC = {"type": "json_object", "required_keys": ["name", "age"], "expected": {"name": "sara"},
             "types": {"age": "integer"}}
ABSTAIN = {"type": "abstain", "max_words": 40, "forbidden": ["rupees", "$"]}

CASES = [
    # (category, response, spec, expected pass)
    # --- positive controls
    ("positive", "42", NUM, True),
    ("positive", "Final answer: 42", NUM, True),
    ("positive", "billing", LABEL, True),
    ("positive", "The capital of France is Paris.", CAPITAL, True),
    ("positive", '{"name": "Sara", "age": 29}', JSON_SPEC, True),
    ("positive", "NOT IN CONTEXT", ABSTAIN, True),
    # --- negative controls: confident, plausible, wrong
    ("negative", "41", NUM, False),
    ("negative", "The answer is 24.", NUM, False),
    ("negative", "account", LABEL, False),
    ("negative", "The capital of France is Lyon.", CAPITAL, False),
    ("negative", '{"name": "Ali", "age": 29}', JSON_SPEC, False),
    ("negative", "The ticket costs 500 rupees.", ABSTAIN, False),
    # --- paraphrase / harmless formatting
    ("paraphrase", "It is **42**.", NUM, True),
    ("paraphrase", "forty-two", NUM, True),  # number words are numbers
    ("paraphrase", "BILLING", LABEL, True),
    ("paraphrase", "Label: billing", LABEL, True),
    ("paraphrase", "```json\n{\"NAME\": \"sara\", \"Age\": 29}\n```", JSON_SPEC, True),
    ("paraphrase", "I could not find that in the context.", ABSTAIN, True),
    ("paraphrase", "paris", CAPITAL, True),
    # --- adversarial: contradictions, distractors, multiple labels, extra content
    ("adversarial", "It is not Paris.", CAPITAL, False),
    ("adversarial", "Paris? No – it is not Paris, it is Lyon.", CAPITAL, True),  # one un-negated mention remains
    ("adversarial", "Final answer: 42. Final answer: 43", NUM, False),  # conflicting final answers
    ("adversarial", "In 2024-05-01 the count was 42", NUM, True),  # the date is not read as a number
    ("adversarial", "At 16:00 we had 42", NUM, True),
    ("adversarial", "billing or technical_support", LABEL, False),  # ambiguous
    ("adversarial", "It is not technical_support; it is billing.", LABEL, True),
    ("adversarial", "This is not about billing at all.", LABEL, False),
    ("adversarial", '{"name": "Sara", "age": "29"}', JSON_SPEC, False),  # wrong type
    ("adversarial", "Not in the context, but it is probably $50.", ABSTAIN, False),
    ("adversarial", "I cannot find it. " + "word " * 60, ABSTAIN, False),  # long hedge
    # --- boundary: empty, malformed, unicode, whitespace
    ("boundary", "", NUM, False),
    ("boundary", "", LABEL, False),
    ("boundary", "   \n\t ", CAPITAL, False),
    ("boundary", "{not json", JSON_SPEC, False),
    ("boundary", "null", JSON_SPEC, False),
    ("boundary", "PARİS", CAPITAL, False),  # dotted capital I is a different character
    ("boundary", "１２", {"type": "numeric", "value": 12}, True),  # Unicode (full-width) digits are numbers too
    ("boundary", "-0", {"type": "numeric", "value": 0}, True),
    ("boundary", "3.0001", {"type": "numeric", "value": 3, "tolerance": 0.001}, True),
    ("boundary", "3.002", {"type": "numeric", "value": 3, "tolerance": 0.001}, False),
]


@pytest.mark.parametrize("category,response,spec,expected", CASES,
                         ids=[f"{c[0]}-{i}" for i, c in enumerate(CASES)])
def test_evaluator_cases(category, response, spec, expected):
    r = evaluate_spec(response, spec)
    assert r.passed is expected, (category, response, r.details)


def test_numeric_only_mode_rejects_distractor_numbers():
    spec = {"type": "numeric", "value": 7, "mode": "only"}
    assert evaluate_spec("7", spec).passed
    assert not evaluate_spec("3 + 4 = 7", spec).passed  # three distinct numbers
    assert evaluate_spec("seven", spec).passed


def test_contains_all_negation_and_forbidden_terms():
    spec = {"type": "contains_all", "keywords": ["encrypted", "offline"], "forbidden": ["cloud upload"]}
    assert evaluate_spec("Data stays offline and is encrypted.", spec).passed
    assert not evaluate_spec("Data is not encrypted but stays offline.", spec).passed
    assert not evaluate_spec("Offline, encrypted, with cloud upload.", spec).passed
    loose = {**spec, "negation_aware": False}
    assert evaluate_spec("Data is not encrypted but stays offline.", loose).passed


def test_json_partial_credit_never_passes_with_a_failed_field():
    r = evaluate_spec('{"name": "Ali", "age": 29}', JSON_SPEC)
    assert not r.passed and 0 < r.score < 1 and r.details["checks"]["name"] is False


def test_all_of_weights_are_validated():
    good = {"type": "all_of", "checks": [NUM, {"type": "word_limit", "max_words": 3}], "weights": [3, 1]}
    r = evaluate_spec("42", good)
    assert r.passed and r.score == 1.0
    r = evaluate_spec("the answer is 42 obviously yes", good)
    assert not r.passed and r.score == pytest.approx(0.75)
    with pytest.raises(ValueError):
        validate_spec({"type": "all_of", "checks": [NUM], "weights": [0]})
    with pytest.raises(ValueError):
        validate_spec({"type": "all_of", "checks": [NUM, NUM], "weights": [1]})
    with pytest.raises(ValueError):
        validate_spec({"type": "label", "labels": ["a"], "answer": "b"})
    with pytest.raises(ValueError):
        validate_spec({"type": "regex", "pattern": "("})


def test_every_evaluator_is_versioned():
    assert set(EVALUATOR_VERSIONS) == set(EVALUATORS)
    assert spec_version({"type": "all_of", "checks": [NUM, LABEL]}) == "all_of@2[numeric@2,label@2]"


# ----------------------------------------------------------------------------- properties
SPECS = [
    NUM, LABEL, CAPITAL, JSON_SPEC, ABSTAIN,
    {"type": "exact_match", "answers": ["Tokyo"]},
    {"type": "contains_all", "keywords": ["a", "b"], "threshold": 0.5},
    {"type": "regex", "pattern": r"\byes\b", "flags": "i"},
    {"type": "word_limit", "max_words": 5},
    {"type": "bullet_count", "count": 3},
    {"type": "json_equals", "value": [1, 2]},
    {"type": "token_f1", "reference": "the quick brown fox"},
    {"type": "rouge_l", "reference": "the quick brown fox"},
    {"type": "all_of", "checks": [NUM, {"type": "word_limit", "max_words": 5}]},
]


@settings(max_examples=150, deadline=None)
@given(st.text(max_size=300), st.sampled_from(SPECS))
def test_scores_are_bounded_deterministic_and_never_crash(text, spec):
    a = evaluate_spec(text, spec)
    b = evaluate_spec(text, spec)
    assert 0.0 <= a.score <= 1.0
    assert (a.score, a.passed) == (b.score, b.passed)
    json.dumps(a.to_dict())  # details are always serialisable


@settings(max_examples=100, deadline=None)
@given(st.integers(-10**6, 10**6), st.integers(-10**6, 10**6))
def test_numeric_final_answer_semantics(target, other):
    spec = {"type": "numeric", "value": target}
    assert evaluate_spec(f"Final answer: {target}", spec).passed
    assert evaluate_spec(f"We tried {other} first. Final answer: {target}", spec).passed
    if other != target:
        assert not evaluate_spec(f"Final answer: {other}", spec).passed


@settings(max_examples=100, deadline=None)
@given(st.text(alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd", "Zs")), max_size=80))
def test_pass_implies_full_score_for_binary_evaluators(text):
    for spec in (NUM, LABEL, CAPITAL, ABSTAIN, {"type": "exact_match", "answers": ["Tokyo"]}):
        r = evaluate_spec(text, spec)
        assert r.score in (0.0, 1.0) and (r.score == 1.0) == r.passed


def test_calibration_set_agreement_is_measured_and_known_errors_are_visible():
    from app.benchmark.calibration import agreement
    from app.core.config import BACKEND_DIR

    rep = agreement(BACKEND_DIR / "datasets")
    o = rep["overall"]
    assert o["n"] >= 50 and 0.8 <= o["agreement"] < 1.0  # the evaluators are useful but not perfect
    assert o["false_pass"] >= 1 and o["false_fail"] >= 1
    ids = {d["id"] for d in rep["disagreements"]}
    assert {"cal-019", "cal-026"} <= ids  # documented "last number" and unit errors of the numeric rule
    assert rep["draft_labels"] == o["n"]  # labels still need independent human adjudication
