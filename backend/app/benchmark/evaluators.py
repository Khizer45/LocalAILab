"""Deterministic, reference-based evaluators for benchmark responses.

Every evaluator returns an :class:`EvalResult` with a score in [0, 1], a
pass/fail decision and machine-readable details. No evaluator uses another
LLM as a judge, so results are reproducible and explainable. They measure
*task success against a defined rule*, not general "intelligence" – see
docs/BENCHMARK_METHODOLOGY.md for the limits of each method.

Deterministic is not the same as correct, so every evaluator is versioned
(``EVALUATOR_VERSIONS``; stored with each result) and tested with positive,
negative, paraphrase, adversarial, boundary and property-based cases
(tests/test_evaluators.py). Their agreement with adjudicated judgements is
measured with ``scripts/evaluator_agreement.py`` on
``datasets/evaluator_calibration``.

v2 changes (Local AI Lab 1.2): negated keywords no longer count as found
("it is *not* Paris"); conflicting final answers fail the numeric check; dates
and times are not read as numbers; a label answer that names several labels
without an explicit marker is ambiguous (fails); ``abstain`` can forbid the
invented answer and limit length; JSON checks can require value types; and
``all_of`` validates its weights.
"""

from __future__ import annotations

import json
import re
import string
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_PUNCT = str.maketrans("", "", string.punctuation)
_NUMBER = re.compile(r"(?<![\w.])-?\d+(?:,\d{3})*(?:\.\d+)?(?![\w])|(?<![\w.-])\d+(?:,\d{3})*(?:\.\d+)?")
_DATE_TIME = re.compile(r"\b\d{4}-\d{1,2}-\d{1,2}\b|\b\d{1,2}[/.]\d{1,2}[/.]\d{2,4}\b|\b\d{1,2}:\d{2}(?::\d{2})?\b")
_NEGATION = re.compile(r"\b(?:not|no|never|isn't|isnt|is not|wasn't|aren't|doesn't|don't|didn't|cannot|can't|"
                       r"neither|nor|without)\b(?:\W+\w+){0,2}\W+$", re.IGNORECASE)

EVALUATOR_VERSIONS = {
    "exact_match": "1", "contains_all": "2", "contains_any": "2", "numeric": "2", "regex": "1", "word_limit": "1",
    "bullet_count": "1", "json_object": "2", "json_equals": "1", "label": "2", "token_f1": "1", "rouge_l": "1",
    "abstain": "2", "all_of": "2",
}


@dataclass
class EvalResult:
    score: float
    passed: bool
    method: str
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"score": round(self.score, 4), "passed": self.passed, "method": self.method, "details": self.details}


def normalize(text: str) -> str:
    text = text.lower().translate(_PUNCT)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def _tokens(text: str) -> list[str]:
    return normalize(text).split()


def _first_line(text: str) -> str:
    for line in text.strip().splitlines():
        if line.strip():
            return line.strip()
    return ""


def extract_json(text: str) -> Any:
    """Parse the first JSON value in a response, tolerating ``` fences."""
    t = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", t, re.DOTALL | re.IGNORECASE)
    if fence:
        t = fence.group(1).strip()
    try:
        return json.loads(t)
    except ValueError:
        pass
    for open_c, close_c in (("{", "}"), ("[", "]")):
        start, end = t.find(open_c), t.rfind(close_c)
        if 0 <= start < end:
            try:
                return json.loads(t[start:end + 1])
            except ValueError:
                continue
    raise ValueError("no JSON found")


# --------------------------------------------------------------------------- evaluators
def exact_match(resp: str, spec: dict) -> EvalResult:
    answers = [normalize(a) for a in spec["answers"]]
    got = normalize(_first_line(resp))
    ok = got in answers
    return EvalResult(1.0 if ok else 0.0, ok, "exact_match", {"normalized_response": got[:200], "accepted": answers})


def _mentions(low: str, keyword: str, negation_aware: bool) -> bool:
    """True if ``keyword`` occurs at least once without a negation just before it."""
    kw = keyword.lower()
    start = 0
    while (i := low.find(kw, start)) >= 0:
        if not negation_aware or not _NEGATION.search(low[max(0, i - 40):i]):
            return True
        start = i + 1
    return False


def _forbidden(low: str, spec: dict, negation_aware: bool = True) -> list[str]:
    return [f for f in spec.get("forbidden", []) if _mentions(low, f, negation_aware)]


def contains_all(resp: str, spec: dict) -> EvalResult:
    low = resp.lower()
    kws = spec["keywords"]
    neg = spec.get("negation_aware", True)
    found = [k for k in kws if _mentions(low, k, neg)]
    bad = _forbidden(low, spec, neg)
    score = (len(found) / len(kws) if kws else 0.0) * (0.0 if bad else 1.0)
    threshold = spec.get("threshold", 1.0)
    return EvalResult(score, score >= threshold - 1e-9 and not bad, "contains_all",
                      {"found": found, "missing": [k for k in kws if k not in found], "threshold": threshold,
                       "forbidden_found": bad})


def contains_any(resp: str, spec: dict) -> EvalResult:
    low = resp.lower()
    neg = spec.get("negation_aware", True)
    found = [k for k in spec["keywords"] if _mentions(low, k, neg)]
    bad = _forbidden(low, spec, neg)
    ok = bool(found) and not bad
    return EvalResult(1.0 if ok else 0.0, ok, "contains_any", {"found": found, "forbidden_found": bad})


_UNITS = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen "
                                       "fourteen fifteen sixteen seventeen eighteen nineteen".split())}
_TENS = {w: 10 * i for i, w in enumerate("_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()) if i >= 2}
_NUM_WORD = re.compile(r"\b(?:(?:" + "|".join(sorted(_TENS, key=len, reverse=True)) + r")(?:[- ](?:"
                       + "|".join(sorted(list(_UNITS)[1:10], key=len, reverse=True)) + r"))?|"
                       + "|".join(sorted(_UNITS, key=len, reverse=True)) + r")\b(?:\s+(hundred|thousand))?",
                       re.IGNORECASE)
_FINAL_MARK = re.compile(r"(?:final answer\s*(?:is|:|=)?|####)\s*\**\s*(-?\d[\d,]*(?:\.\d+)?)", re.IGNORECASE)
_ANSWER_MARK = re.compile(r"(?:final answer|the answer is|answer\s*:|answer\s*=|####)\s*\**\s*(-?\d[\d,]*(?:\.\d+)?)",
                          re.IGNORECASE)


def _word_value(m: re.Match) -> float:
    words = m.group(0).lower().replace("-", " ").split()
    total = 0
    for w in words:
        if w in _UNITS:
            total += _UNITS[w]
        elif w in _TENS:
            total += _TENS[w]
        elif w == "hundred":
            total = max(total, 1) * 100
        elif w == "thousand":
            total = max(total, 1) * 1000
    return float(total)


def numbers_in(text: str) -> list[float]:
    """All numbers in reading order – digits ("1,250.5") and English number words up to the thousands.

    Dates (2024-05-01, 1/5/2024) and clock times (16:00) are not numbers here."""
    text = _DATE_TIME.sub(lambda m: " " * len(m.group(0)), text)
    found = [(m.start(), float(m.group(0).replace(",", ""))) for m in _NUMBER.finditer(text)]
    found += [(m.start(), _word_value(m)) for m in _NUM_WORD.finditer(text)]
    return [v for _, v in sorted(found)]


def numeric(resp: str, spec: dict) -> EvalResult:
    """Final numeric answer: the number after an explicit answer marker if present, else the last number.

    Number words count as numbers ("seven continents" → 7)."""
    target, tol = float(spec["value"]), float(spec.get("tolerance", 0.0))
    # a "final answer" marker outranks "the answer is" (models often restate intermediate answers);
    # different values within the winning tier are a contradiction and fail
    marked = [float(m.replace(",", "")) for m in _FINAL_MARK.findall(resp)] or \
        [float(m.replace(",", "")) for m in _ANSWER_MARK.findall(resp)]
    conflict = len({round(v, 9) for v in marked}) > 1
    if spec.get("mode") == "only":  # the response must contain exactly one distinct number
        nums = numbers_in(resp)
        distinct = {round(v, 9) for v in nums}
        final = nums[0] if len(distinct) == 1 else None
        source = "only number" if final is not None else f"{len(distinct)} distinct numbers"
    elif marked:
        final, source = marked[-1], "answer marker"
    else:
        nums = numbers_in(resp)
        final, source = (nums[-1] if nums else None), "last number"
    ok = final is not None and abs(final - target) <= tol + 1e-9 and not conflict
    return EvalResult(1.0 if ok else 0.0, ok, "numeric (final answer)",
                      {"final_number": final, "expected": target, "tolerance": tol, "taken_from": source,
                       "conflicting_answers": sorted({*marked}) if conflict else None})


def regex(resp: str, spec: dict) -> EvalResult:
    flags = 0
    for f in spec.get("flags", ""):
        flags |= {"i": re.IGNORECASE, "m": re.MULTILINE, "s": re.DOTALL}[f]
    ok = re.search(spec["pattern"], resp.strip(), flags) is not None
    if spec.get("negate"):
        ok = not ok
    return EvalResult(1.0 if ok else 0.0, ok, "regex", {"pattern": spec["pattern"], "negate": bool(spec.get("negate"))})


def word_limit(resp: str, spec: dict) -> EvalResult:
    n = len(resp.split())
    lo, hi = spec.get("min_words", 1), spec["max_words"]
    ok = lo <= n <= hi
    return EvalResult(1.0 if ok else 0.0, ok, "word_limit", {"words": n, "min": lo, "max": hi})


def bullet_count(resp: str, spec: dict) -> EvalResult:
    lines = [ln.strip() for ln in resp.strip().splitlines() if ln.strip()]
    bullets = [ln for ln in lines if re.match(r"^([-*•]|\d+[.)])\s+", ln)]
    want = spec["count"]
    extra = len(lines) - len(bullets)
    extra_penalty = extra > 0 and spec.get("only_bullets", True)
    ok = len(bullets) == want and not extra_penalty
    score = max(0.0, 1 - abs(len(bullets) - want) / want) * (0.5 if extra_penalty else 1.0)
    return EvalResult(score, ok, "bullet_count", {"bullets": len(bullets), "expected": want, "other_lines": extra})


def _norm_value(v: Any, spec: dict) -> Any:
    if isinstance(v, str):
        if spec.get("strip_chars"):
            v = v.strip(spec["strip_chars"])
        v = v.strip()
        if spec.get("ignore_case", True):
            v = v.lower()
        return v
    if isinstance(v, float) and v.is_integer():
        return int(v)
    if isinstance(v, list):
        items = [_norm_value(x, spec) for x in v]
        return sorted(items, key=str) if spec.get("ignore_order") else items
    if isinstance(v, dict):
        return {str(k).lower(): _norm_value(x, spec) for k, x in v.items()}
    return v


def json_object(resp: str, spec: dict) -> EvalResult:
    try:
        obj = extract_json(resp)
    except ValueError:
        return EvalResult(0.0, False, "json_object", {"valid_json": False})
    if not isinstance(obj, dict):
        return EvalResult(0.2, False, "json_object", {"valid_json": True, "is_object": False})
    obj_n = _norm_value(obj, spec)
    required = [k.lower() for k in spec.get("required_keys", [])]
    expected = {k.lower(): _norm_value(v, spec) for k, v in spec.get("expected", {}).items()}
    checks = {k: (k in obj_n) for k in required}
    for k, v in expected.items():
        val = obj_n.get(k)
        if isinstance(v, (int, float)) and isinstance(val, str):
            try:
                val = float(val) if "." in val else int(val)
            except ValueError:
                pass
        checks[k] = val == v
    raw_lower = {str(kk).lower(): vv for kk, vv in obj.items()}
    for k, typ in (spec.get("types") or {}).items():
        checks[f"{k.lower()}:{typ}"] = _has_type(raw_lower.get(k.lower(), _MISSING), typ)
    exact_keys = spec.get("exact_keys", False)
    if exact_keys:
        checks["__no_extra_keys__"] = set(obj_n) == set(required)
    frac = sum(checks.values()) / len(checks) if checks else 1.0
    score = 0.4 + 0.6 * frac
    return EvalResult(score, frac == 1.0, "json_object", {"valid_json": True, "checks": checks})


_MISSING = object()
_TYPES = {"string": (str,), "integer": (int,), "number": (int, float), "boolean": (bool,), "array": (list,),
          "object": (dict,), "null": (type(None),)}


def _has_type(value: Any, typ: str) -> bool:
    if value is _MISSING:
        return False
    if typ in ("integer", "number") and isinstance(value, bool):
        return False
    if typ == "integer" and isinstance(value, float):
        return value.is_integer()
    return isinstance(value, _TYPES.get(typ, (object,)))


def json_equals(resp: str, spec: dict) -> EvalResult:
    try:
        val = extract_json(resp)
    except ValueError:
        return EvalResult(0.0, False, "json_equals", {"valid_json": False})
    ok = _norm_value(val, spec) == _norm_value(spec["value"], spec)
    return EvalResult(1.0 if ok else 0.3, ok, "json_equals", {"valid_json": True})


_LABEL_MARK = re.compile(r"(?:label|intent|category|class|answer|classification)\s*[:=]\s*\**\s*([^\n*]+)",
                         re.IGNORECASE)


def _label_in(fragment: str, labels: list[str]) -> str | None:
    frag = re.sub(r"[^a-z0-9_ -]", " ", fragment.lower()).strip()
    for lab in labels:
        if frag == lab.lower() or frag.replace(" ", "_") == lab.lower():
            return lab
    return None


def _mentioned_labels(text: str, labels: list[str]) -> list[tuple[int, str]]:
    out = []
    for lab in labels:
        for variant in {lab.lower(), lab.lower().replace("_", " ")}:
            for m in re.finditer(rf"\b{re.escape(variant)}\b", text):
                if not _NEGATION.search(text[max(0, m.start() - 40):m.start()]):
                    out.append((m.start(), lab))
    return out


def label(resp: str, spec: dict) -> EvalResult:
    """Classification: which allowed label did the response choose?

    Order: an explicit marker ("Label: billing"), then the first line being exactly a label, then a single label
    mentioned anywhere (negated mentions ignored). Several different labels without a marker = ambiguous."""
    labels = spec["labels"]
    text = resp.strip().lower()
    predicted, source = None, None
    marks = [_label_in(m.group(1).split(".")[0], labels) for m in _LABEL_MARK.finditer(text)]
    marks = [m for m in marks if m]
    if marks:
        predicted, source = marks[-1], "marker"
    if predicted is None:
        predicted = _label_in(_first_line(text), labels)
        source = "first line" if predicted else None
    mentioned: list[str] = []
    if predicted is None:
        mentioned = sorted({lab for _, lab in _mentioned_labels(text, labels)})
        if len(mentioned) == 1:
            predicted, source = mentioned[0], "only label mentioned"
        elif len(mentioned) > 1:
            source = "ambiguous"
    ok = predicted == spec["answer"]
    return EvalResult(1.0 if ok else 0.0, ok, "label", {"predicted": predicted, "expected": spec["answer"],
                                                         "taken_from": source,
                                                         **({"mentioned": mentioned} if len(mentioned) > 1 else {})})


def token_f1(resp: str, spec: dict) -> EvalResult:
    pred, ref = _tokens(resp), _tokens(spec["reference"])
    common = Counter(pred) & Counter(ref)
    same = sum(common.values())
    if not pred or not ref or same == 0:
        f1 = 0.0
    else:
        p, r = same / len(pred), same / len(ref)
        f1 = 2 * p * r / (p + r)
    thr = spec.get("threshold", 0.5)
    return EvalResult(f1, f1 >= thr, "token_f1", {"f1": round(f1, 4), "threshold": thr})


def rouge_l(resp: str, spec: dict) -> EvalResult:
    a, b = _tokens(resp), _tokens(spec["reference"])
    if not a or not b:
        return EvalResult(0.0, False, "rouge_l", {"f": 0.0})
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    lcs = prev[-1]
    p, r = lcs / len(a), lcs / len(b)
    f = 0.0 if lcs == 0 else 2 * p * r / (p + r)
    thr = spec.get("threshold", 0.3)
    return EvalResult(f, f >= thr, "rouge_l", {"f": round(f, 4), "threshold": thr})


def abstain(resp: str, spec: dict) -> EvalResult:
    phrases = spec.get("phrases", ["not in context", "not in the context", "cannot find", "could not find",
                                   "can't find", "couldn't find", "not mentioned", "no information", "does not say",
                                   "doesn't say", "not provided", "does not mention", "doesn't mention",
                                   "is not stated", "not stated", "not specified", "does not contain",
                                   "doesn't contain", "isn't mentioned", "not available in the context"])
    low = resp.lower()
    matched = [p for p in phrases if p in low]
    bad = [f for f in spec.get("forbidden", []) if _mentions(low, f, True)]
    words = len(resp.split())
    too_long = spec.get("max_words") is not None and words > int(spec["max_words"])
    ok = bool(matched) and not bad and not too_long
    return EvalResult(1.0 if ok else 0.0, ok, "abstain", {"matched": matched, "forbidden_found": bad, "words": words,
                                                           "too_long": too_long})


def all_of(resp: str, spec: dict) -> EvalResult:
    """Every sub-check must pass; the score is their weighted mean (weights must be positive)."""
    parts = [evaluate_spec(resp, c) for c in spec["checks"]]
    weights = spec.get("weights") or [1.0] * len(parts)
    if len(weights) != len(parts) or any(w <= 0 for w in weights):
        raise ValueError("all_of: weights must be positive and one per check")
    score = sum(w * p.score for w, p in zip(weights, parts, strict=True)) / sum(weights)
    return EvalResult(score, all(p.passed for p in parts), "all_of", {"checks": [p.to_dict() for p in parts]})


def validate_spec(spec: dict) -> None:
    """Static checks used when a suite is loaded or saved (raises ValueError)."""
    t = spec.get("type")
    if t not in EVALUATORS:
        raise ValueError(f"unknown evaluator type {t!r}")
    required = {"exact_match": ["answers"], "contains_all": ["keywords"], "contains_any": ["keywords"],
                "numeric": ["value"], "regex": ["pattern"], "word_limit": ["max_words"], "bullet_count": ["count"],
                "json_equals": ["value"], "label": ["labels", "answer"], "token_f1": ["reference"],
                "rouge_l": ["reference"], "all_of": ["checks"]}.get(t, [])
    missing = [k for k in required if k not in spec]
    if missing:
        raise ValueError(f"{t}: missing {', '.join(missing)}")
    if t == "regex":
        try:
            re.compile(spec["pattern"])
        except re.error as exc:
            raise ValueError(f"regex: invalid pattern ({exc})") from exc
    if t == "label" and spec["answer"] not in spec["labels"]:
        raise ValueError("label: the answer must be one of the labels")
    if t in ("contains_all", "contains_any") and not spec["keywords"]:
        raise ValueError(f"{t}: keywords must not be empty")
    if t == "all_of":
        if not spec["checks"]:
            raise ValueError("all_of: needs at least one check")
        w = spec.get("weights")
        if w is not None and (len(w) != len(spec["checks"]) or any(x <= 0 for x in w)):
            raise ValueError("all_of: weights must be positive and one per check")
        for sub in spec["checks"]:
            validate_spec(sub)


EVALUATORS = {
    "exact_match": exact_match,
    "contains_all": contains_all,
    "contains_any": contains_any,
    "numeric": numeric,
    "regex": regex,
    "word_limit": word_limit,
    "bullet_count": bullet_count,
    "json_object": json_object,
    "json_equals": json_equals,
    "label": label,
    "token_f1": token_f1,
    "rouge_l": rouge_l,
    "abstain": abstain,
    "all_of": all_of,
}


def evaluate_spec(response: str, spec: dict) -> EvalResult:
    fn = EVALUATORS.get(spec.get("type", ""))
    if fn is None:
        raise ValueError(f"Unknown evaluator type: {spec.get('type')}")
    res = fn(response or "", spec)
    res.score = min(1.0, max(0.0, float(res.score)))
    return res


def spec_version(spec: dict) -> str:
    """Version string of an evaluator spec, including nested checks (e.g. "all_of@2[numeric@2,regex@1]")."""
    t = spec.get("type", "")
    v = f"{t}@{EVALUATOR_VERSIONS.get(t, '?')}"
    if t == "all_of":
        v += "[" + ",".join(spec_version(c) for c in spec.get("checks", [])) + "]"
    return v
