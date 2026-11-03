"""Tools that agents can call during an Agent Arena run.

Two kinds of tools:

* **Mock tools** are declared in JSON (name, description, JSON-schema
  parameters, canned responses chosen by argument matchers). They make agent
  benchmarks deterministic and safe: "send_email" or "delete_file" never touch
  anything real. Users describe their *own* tools this way to test models on
  their own workflow before wiring up real APIs.
* **Built-in tools** run real, side-effect-free code: a safe calculator, a
  currency converter with fixed demo rates, the current date/time, and a
  search over the user's own indexed documents.

Matchers (used for mock responses and for expected arguments in tasks)::

    "Lahore"                    equal (case-insensitive for text, numeric tolerance 1e-6)
    {"equals": 45}              same as a plain value
    {"contains": "invoice"}     substring, case-insensitive
    {"any_of": ["USD", "usd"]}  one of several values
    {"regex": "^A-\\d{4}$"}     regular expression (search, case-sensitive unless (?i))
    {"number": 434, "tolerance": 0.5}
    {"exists": true}            argument present with any value
    {"absent": true}            argument not given
"""

from __future__ import annotations

import ast
import json
import math
import operator
import re
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
MISSING = object()


# ----------------------------------------------------------------------------- matchers
def _num(v) -> float | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v.replace(",", "").strip())
        except ValueError:
            return None
    return None


def _eq(expected, value) -> bool:
    if value is MISSING:
        return False
    if isinstance(expected, bool) or isinstance(value, bool):
        return str(expected).lower() == str(value).lower()
    if isinstance(expected, (int, float)):
        n = _num(value)
        return n is not None and abs(n - float(expected)) <= 1e-6 * max(1.0, abs(float(expected)))
    if isinstance(expected, str):
        return isinstance(value, (str, int, float)) and str(value).strip().lower() == expected.strip().lower()
    return expected == value


def match_value(spec: Any, value: Any) -> bool:
    if isinstance(spec, dict) and spec and set(spec) <= {"equals", "contains", "any_of", "regex", "number",
                                                          "tolerance", "exists", "absent"}:
        if spec.get("absent"):
            return value is MISSING
        if value is MISSING:
            return False
        if spec.get("exists"):
            return True
        if "equals" in spec:
            return _eq(spec["equals"], value)
        if "contains" in spec:
            return str(spec["contains"]).lower() in json.dumps(value, ensure_ascii=False).lower() \
                if not isinstance(value, str) else str(spec["contains"]).lower() in value.lower()
        if "any_of" in spec:
            return any(_eq(x, value) for x in spec["any_of"])
        if "regex" in spec:
            return isinstance(value, (str, int, float)) and re.search(spec["regex"], str(value)) is not None
        if "number" in spec:
            n = _num(value)
            return n is not None and abs(n - float(spec["number"])) <= float(spec.get("tolerance", 1e-6))
        return False
    return _eq(spec, value)


def match_args(when: dict[str, Any], args: dict[str, Any]) -> float:
    """Fraction of argument matchers satisfied (1.0 when `when` is empty)."""
    if not when:
        return 1.0
    hits = sum(match_value(spec, args.get(k, MISSING)) for k, spec in when.items())
    return hits / len(when)


def example_value(spec: Any) -> Any:
    """A concrete value satisfying a matcher (used by the simulator and docs)."""
    if isinstance(spec, dict):
        if "equals" in spec:
            return spec["equals"]
        if "contains" in spec:
            return spec["contains"]
        if "any_of" in spec:
            return spec["any_of"][0]
        if "number" in spec:
            return spec["number"]
        return "example"
    return spec


# ----------------------------------------------------------------------------- JSON schema (subset)
_TYPES = {"string": str, "boolean": bool, "object": dict, "array": list}


def validate_args(schema: dict, args: Any) -> list[str]:
    if not isinstance(args, dict):
        return ["arguments must be a JSON object"]
    errors: list[str] = []
    props: dict = schema.get("properties") or {}
    for req in schema.get("required") or []:
        if req not in args:
            errors.append(f"missing required argument '{req}'")
    if schema.get("additionalProperties") is False:
        for k in args:
            if k not in props:
                errors.append(f"unknown argument '{k}'")
    for k, v in args.items():
        p = props.get(k)
        if not isinstance(p, dict):
            continue
        t = p.get("type")
        if t == "number" and (isinstance(v, bool) or not isinstance(v, (int, float))):
            errors.append(f"'{k}' must be a number")
        elif t == "integer" and (isinstance(v, bool) or not (isinstance(v, int) or (isinstance(v, float) and v.is_integer()))):
            errors.append(f"'{k}' must be an integer")
        elif t in _TYPES and not isinstance(v, _TYPES[t]):
            errors.append(f"'{k}' must be of type {t}")
        if "enum" in p and v not in p["enum"]:
            errors.append(f"'{k}' must be one of {p['enum']}")
    return errors


# ----------------------------------------------------------------------------- tool specs
class MockResponse(BaseModel):
    when: dict[str, Any] = Field(default_factory=dict)
    result: Any


BUILTIN_SPECS: dict[str, dict] = {
    "calculator": {
        "description": "Evaluate an arithmetic expression, e.g. '2480 * 0.175' or 'sqrt(2) * 10'. Supports + - * / "
                       "// % ** parentheses and sqrt, round, abs, min, max, pow, log, exp.",
        "parameters": {"type": "object", "properties": {"expression": {"type": "string",
                                                                       "description": "The expression to compute"}},
                       "required": ["expression"]},
    },
    "currency_converter": {
        "description": "Convert an amount between currencies using fixed demo exchange rates. Use ISO codes like "
                       "USD, EUR, PKR, GBP, AED, SAR, INR, CNY.",
        "parameters": {"type": "object", "properties": {
            "amount": {"type": "number", "description": "Amount to convert"},
            "from_currency": {"type": "string", "description": "ISO code of the source currency, e.g. USD"},
            "to_currency": {"type": "string", "description": "ISO code of the target currency, e.g. PKR"}},
            "required": ["amount", "from_currency", "to_currency"]},
    },
    "current_datetime": {
        "description": "Return the current local date and time.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    "search_documents": {
        "description": "Search the user's private documents indexed in Local AI Lab and return the most relevant "
                       "passages with file name and page.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "What to look for"},
            "top_k": {"type": "integer", "description": "Number of passages (1-8)"}}, "required": ["query"]},
    },
}
DEMO_RATES_PER_USD = {"USD": 1.0, "PKR": 280.0, "EUR": 0.92, "GBP": 0.79, "AED": 3.6725, "SAR": 3.75, "INR": 83.0,
                      "CNY": 7.2, "JPY": 150.0, "CAD": 1.36}


class ToolSpec(BaseModel):
    name: str
    description: str = Field("", max_length=2000)
    parameters: dict = Field(default_factory=lambda: {"type": "object", "properties": {}, "required": []})
    builtin: str | None = None
    responses: list[MockResponse] = Field(default_factory=list)
    default_result: Any = Field(default_factory=lambda: {"error": "No result for these arguments."})
    risky: bool = False

    @field_validator("name")
    @classmethod
    def _name(cls, v: str) -> str:
        if not NAME_RE.match(v):
            raise ValueError("tool names may only contain letters, digits and underscores (max 64)")
        return v

    @model_validator(mode="after")
    def _builtin(self) -> ToolSpec:
        if self.builtin is not None:
            if self.builtin not in BUILTIN_SPECS:
                raise ValueError(f"unknown built-in tool '{self.builtin}' (choose {', '.join(BUILTIN_SPECS)})")
            spec = BUILTIN_SPECS[self.builtin]
            if not self.description:
                self.description = spec["description"]
            if not (self.parameters.get("properties")):
                self.parameters = spec["parameters"]
        if self.parameters.get("type", "object") != "object" or not isinstance(self.parameters.get("properties", {}),
                                                                                 dict):
            raise ValueError(f"tool '{self.name}': parameters must be a JSON schema of type 'object'")
        self.parameters.setdefault("type", "object")
        self.parameters.setdefault("properties", {})
        if not self.description:
            raise ValueError(f"tool '{self.name}' needs a description")
        return self

    def ollama_schema(self) -> dict:
        return {"type": "function", "function": {"name": self.name, "description": self.description,
                                                 "parameters": self.parameters}}

    def prompt_line(self) -> str:
        return f"- {self.name}: {self.description}\n  parameters: {json.dumps(self.parameters, ensure_ascii=False)}"


# ----------------------------------------------------------------------------- built-in implementations
_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {"sqrt": math.sqrt, "round": round, "abs": abs, "min": min, "max": max, "pow": pow, "log": math.log,
          "exp": math.exp, "floor": math.floor, "ceil": math.ceil}


def safe_eval(expression: str) -> float:
    expr = expression.replace("×", "*").replace("÷", "/").replace("^", "**").replace(",", "")
    expr = re.sub(r"(\d+(?:\.\d+)?)\s*%", r"(\1/100)", expr)
    if len(expr) > 300:
        raise ValueError("expression too long")
    tree = ast.parse(expr, mode="eval")

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("exponent too large")
            return _BIN[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            return _UNARY[type(node.op)](ev(node.operand))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS and not node.keywords:
            return _FUNCS[node.func.id](*[ev(a) for a in node.args])
        raise ValueError("unsupported expression")

    result = ev(tree)
    if isinstance(result, float) and (math.isnan(result) or math.isinf(result)):
        raise ValueError("result is not a finite number")
    return result


def run_builtin(name: str, args: dict, context: dict | None) -> Any:
    if name == "calculator":
        try:
            value = safe_eval(str(args.get("expression", "")))
        except (ValueError, SyntaxError, TypeError, ZeroDivisionError, OverflowError) as exc:
            return {"error": f"Could not evaluate: {exc}"}
        return {"expression": args.get("expression"), "result": round(value, 10) if isinstance(value, float) else value}
    if name == "currency_converter":
        src = str(args.get("from_currency", "")).strip().upper()
        dst = str(args.get("to_currency", "")).strip().upper()
        amount = _num(args.get("amount"))
        if amount is None:
            return {"error": "amount must be a number"}
        if src not in DEMO_RATES_PER_USD or dst not in DEMO_RATES_PER_USD:
            return {"error": f"Unsupported currency. Supported: {', '.join(DEMO_RATES_PER_USD)}"}
        rate = DEMO_RATES_PER_USD[dst] / DEMO_RATES_PER_USD[src]
        return {"amount": amount, "from": src, "to": dst, "rate": round(rate, 6), "converted": round(amount * rate, 2),
                "note": "fixed demo rates"}
    if name == "current_datetime":
        now = datetime.now()
        return {"date": now.strftime("%Y-%m-%d"), "time": now.strftime("%H:%M"), "weekday": now.strftime("%A")}
    if name == "search_documents":
        search = (context or {}).get("search_documents")
        if search is None:
            return {"error": "Document search is not available here."}
        top_k = int(_num(args.get("top_k")) or 4)
        return search(str(args.get("query", "")), max(1, min(top_k, 8)))
    return {"error": f"Unknown built-in tool {name}"}


# ----------------------------------------------------------------------------- executor
class ToolExecutor:
    def __init__(self, tools: list[ToolSpec], context: dict | None = None):
        self.tools = {t.name: t for t in tools}
        self.context = context or {}

    def call(self, name: str, args: Any) -> tuple[Any, dict]:
        tool = self.tools.get(name)
        if tool is None:
            return ({"error": f"Unknown tool '{name}'. Available tools: {', '.join(self.tools) or 'none'}."},
                    {"valid": False, "errors": ["unknown tool"], "unknown": True, "risky": False})
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except ValueError:
                return ({"error": "Arguments were not valid JSON."},
                        {"valid": False, "errors": ["arguments are not valid JSON"], "unknown": False,
                         "risky": tool.risky})
        if args is None:
            args = {}
        errors = validate_args(tool.parameters, args)
        meta = {"valid": not errors, "errors": errors, "unknown": False, "risky": tool.risky}
        if errors:
            return {"error": "Invalid arguments: " + "; ".join(errors)}, meta
        if tool.builtin:
            return run_builtin(tool.builtin, args, self.context), meta
        for r in tool.responses:  # first match wins; an empty "when" matches anything
            if match_args(r.when, args) == 1.0:
                return r.result, meta
        return tool.default_result, meta
