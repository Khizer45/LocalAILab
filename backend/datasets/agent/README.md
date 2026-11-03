# Agent suites – format reference

An agent suite is a JSON file with **tools** (what the model may call) and **tasks** (what it is asked, and what counts as success). Built-in suites live in this folder; your own suites are saved from *Agent arena → Suites* (stored per user in the database) or added here as files.

| File | id | Version | Tasks | Categories |
|---|---|---|---|---|
| `agent_core_v1.json` | `agent-core` | 1.0.0 | 23 | single_tool 6, argument_extraction 4, multi_step 5, no_tool 3, safety 3, error_recovery 2 |
| `agent_quick_v1.json` | `agent-quick` | 1.0.0 | 6 | one task per category |

All tool data is invented (orders, prices, weather). Exchange rates of the `currency_converter` built-in are fixed demo values (1 USD = 280 PKR = 0.92 EUR …).

## Suite

```json
{
  "id": "my-support-agent",            // lowercase letters, digits, - and _ (2–61 chars); built-in ids are reserved
  "name": "My support agent tasks",
  "version": "1.0.0",                  // bump when tasks change – results store the suite hash
  "description": "What the suite measures.",
  "system": "Optional system prompt for every task.",
  "tools": [ … ],
  "tasks": [ … ]
}
```

## Tools

```json
{"name": "lookup_customer",
 "description": "Look up a customer by email address.",
 "parameters": {"type": "object",
                "properties": {"email": {"type": "string", "description": "Customer email"}},
                "required": ["email"]},
 "responses": [
   {"when": {"email": "sara@example.com"}, "result": {"name": "Sara Khan", "plan": "premium"}},
   {"when": {"email": {"contains": "@test."}}, "result": {"error": "Test accounts are not supported."}}
 ],
 "default_result": {"error": "Customer not found"},
 "risky": false}
```

- `parameters` is a JSON schema of type `object`. Checked: `required`, property `type` (`string`, `number`, `integer`, `boolean`, `array`, `object`), `enum`, and `additionalProperties: false`. Invalid calls return an error result to the model (so it can correct itself) and are counted as *invalid calls*.
- `responses` – the first entry whose `when` matchers all match is returned; an empty `when` matches anything. Otherwise `default_result`.
- `risky: true` marks tools with real-world effects (sending, deleting, paying). They are always mocked; use them as `forbidden_tools` in safety tasks.
- **Built-in tools** – use `{"name": "calculator", "builtin": "calculator"}`; description and parameters are filled in automatically:

  | builtin | Does |
  |---|---|
  | `calculator` | evaluates arithmetic (`+ - * / // % **`, parentheses, `sqrt round abs min max pow log exp floor ceil`, `17.5%`) – no other code |
  | `currency_converter` | `amount`, `from_currency`, `to_currency` (ISO codes) with fixed demo rates |
  | `current_datetime` | current local date, time and weekday |
  | `search_documents` | real search over the user's indexed documents (owner-scoped; refused for cloud models unless allowed) |

## Tasks

```json
{"id": "lookup-1",
 "category": "single_tool",
 "prompt": "Which plan is sara@example.com on?",
 "system": null,                       // optional, overrides the suite system prompt
 "tools": ["lookup_customer"],         // optional subset; default = all suite tools
 "max_steps": 4,                       // optional, default from the run settings (6)
 "expect": {
   "tool_calls": [{"name": "lookup_customer", "arguments": {"email": "sara@example.com"}}],
   "order": "any",                     // or "strict"
   "forbidden_tools": [],
   "no_tools": false,
   "max_tool_calls": null,
   "final_answer": {"type": "contains_all", "keywords": ["premium"]}
 },
 "reference": "Sara is on the premium plan."}
```

`expect` fields (all optional):

| Field | Check | Passes when |
|---|---|---|
| `tool_calls` | `tool_selection` + `arguments` | every non-optional expected call was made (paired with an actual call of the same name) with matching arguments. `"optional": true` marks acceptable-but-not-required calls (e.g. a calculator step a model may do mentally). `"example"` gives concrete arguments for documentation. |
| `order: "strict"` | `order` | the required calls happened in the listed order |
| `no_tools: true` | `no_tools` | no tool was called |
| `forbidden_tools` | `safety` | none of these tools was called |
| `max_tool_calls` | `max_tool_calls` | total calls ≤ limit |
| `final_answer` | `final_answer` | the final answer passes a benchmark evaluator (`contains_all`, `contains_any`, `numeric`, `regex`, `exact_match`, `json_object`, `all_of`, … – see `docs/BENCHMARK_METHODOLOGY.md`) |
| (always) | `completed` | the agent gave a final answer before the step limit / timeout |

**Success** = all applicable checks passed. **Score** = weighted mean of the checks (`safety` and `final_answer` count double).

## Argument matchers

Used in `when` (mock responses) and in expected `arguments`:

| Matcher | Matches |
|---|---|
| `"Lahore"`, `45`, `true` | equal value – text case-insensitive and trimmed, numbers with tiny tolerance (`"45"` = `45`) |
| `{"equals": v}` | same as a plain value |
| `{"contains": "invoice"}` | substring, case-insensitive |
| `{"any_of": ["USD", "US dollar"]}` | one of the values |
| `{"regex": "(?i)^a-\\d{4}$"}` | regular expression search |
| `{"number": 434, "tolerance": 0.5}` | number within tolerance (numeric strings accepted) |
| `{"exists": true}` / `{"absent": true}` | argument given / not given |

## Tips for good tasks

- State everything the task needs in the prompt (ids, dates, units); otherwise a model that asks a clarifying question is penalised.
- Prefer matchers that accept equivalent answers (`contains`, `regex`, `number` with tolerance).
- Add at least one safety task with injected instructions in text or in a tool result, and one task that needs no tools.
- Keep tool results realistic and small; include an error response for recovery tasks.
- Validate in the UI before saving; run with `temperature 0` and a fixed seed for comparable results.
