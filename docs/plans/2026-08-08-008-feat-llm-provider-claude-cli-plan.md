---
title: LLM Provider Seam — Anthropic API or Claude Code CLI - Plan
type: feat
date: 2026-08-08
status: design (not yet implemented)
execution: code
---

# LLM Provider Seam — Anthropic API or Claude Code CLI

## Goal Capsule

- **Objective:** Let every LLM call in the pipeline (Pass 1, Pass 2, and the
  Stage 1.5 forward-growth extraction) run through either the Anthropic API
  (today's path, pay-as-you-go credits) **or** headless Claude Code
  (`claude -p`, billed against the Max subscription's monthly Agent-SDK
  credit pool), selectable per run by a CLI flag with a config default.
- **Why:** `--deep` runs Opus through API credits. The owner holds a Max
  subscription whose included headless-credit pool covers this workload;
  the API path stays for machines without a Claude Code login and as the
  fallback.
- **Non-goals:** No prompt changes. No new JSON-schema statement of any
  pass's output (the prompts remain the single statement). No change to the
  action guard, the sidecar version block, either regime hash, or any score.
  No retry machinery beyond what exists. No third path via the Agent SDK —
  subscription OAuth is API-key-only there and disallowed by ToS; the CLI
  shell-out is the one sanctioned route.

## The one decision that shapes everything

**The provider is a transport, not a contract.** Both providers receive the
identical rendered prompt, and both return text that flows through the same
`_parse_json_response`. Nothing downstream — parsing, validation, grounding,
the sidecar, the action guard — may be able to tell which transport ran
except by reading the usage metadata. This is what keeps the feature additive:
if the two paths could diverge in what they submit or how they parse, every
consumer would inherit a second behaviour to reason about.

Consequences, stated up front:

1. **Prompts stay single-sourced.** The CLI's `--json-schema` flag is
   deliberately NOT used in v1: it would be a second statement of each pass's
   output schema beside the one already in the prompt text, and the two would
   drift invisibly. (Future option once schemas are extracted to shared
   constants — out of scope here.)
2. **The sidecar version block does not gain a `provider` field.** Its
   identity is *source text + schema + prompt + model* — the reading's
   contract, which transport is not part of. A provider switch therefore does
   NOT invalidate cached extractions, which is correct: a grounded reading
   stays a grounded reading. Residual accepted: the two transports could
   produce different-but-both-valid readings; whichever ran last is what the
   cache holds.
3. **Neither regime hash moves.** `registry_hash` covers registry + macro;
   `forward_signal_hash` covers zero-weight definitions + extraction schema.
   `llm.provider` sits in neither, and a test pins that.

## Vocabulary

One vocabulary everywhere, no mapping tables. `config.yaml` already has
`llm.provider: "anthropic"`; it gains a second legal value:

| Value | Meaning |
|---|---|
| `anthropic` | Anthropic API via the `anthropic` SDK (today's behaviour, stays the default) |
| `claude_cli` | Headless Claude Code via `claude -p`, subscription-billed |

The CLI flag takes the same two literals: `--llm-provider anthropic` /
`--llm-provider claude_cli`. Any other value is a startup error naming the
two legal ones.

## Architecture

### The transport seam (new, inside `llm_layer/`)

A small module, `llm_layer/transport.py`, owning both implementations:

```python
@dataclass
class TransportResponse:
    text: str                    # the model's raw text output
    input_tokens: int
    output_tokens: int
    cost_usd: float | None       # actual cost when the transport knows it (CLI)
    tokens_basis: str            # "reported" | "estimated"

class AnthropicAPITransport:
    def __init__(self, config: dict): ...   # raises ValueError if no ANTHROPIC_API_KEY
    def complete(self, model: str, prompt: str, max_tokens: int) -> TransportResponse: ...

class ClaudeCLITransport:
    def __init__(self, config: dict): ...   # raises ValueError if `claude` not on PATH
    def complete(self, model: str, prompt: str, max_tokens: int) -> TransportResponse: ...

def build_transport(config: dict) -> AnthropicAPITransport | ClaudeCLITransport:
    """Resolve llm.provider to a constructed transport. Unknown value → ValueError."""
```

`complete()` raises `TransportError` (one new exception type) on any failure
— API error, subprocess failure, timeout, malformed envelope. The
orchestrator catches it exactly where it catches `anthropic.APIError` today
and returns the same `{"error": ..., "pass": ...}` dict, so every downstream
consumer sees the failure shape it already handles.

### `AnthropicAPITransport`

The current body of `_call_api` lines 331–338, moved verbatim: client
construction (from `__init__`) and `messages.create`. The API-key
precondition check moves from `LLMOrchestrator.__init__` into this
transport's constructor — same `ValueError`, same message, so
`service.py`'s existing `except ValueError` ("LLM not available") catch
covers both providers without change.

### `ClaudeCLITransport`

Invocation (prompt on **stdin**, never as an argv item — avoids quoting and
ARG_MAX concerns on multi-tens-of-KB extraction prompts):

```
claude -p \
  --bare \
  --model <model-id, passed through verbatim> \
  --output-format json \
  --max-turns 1
```

with `input=prompt` and a **scrubbed environment**:

- **Strip `ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_BASE_URL`**
  from the child env. This is the load-bearing line of the whole feature:
  the project's `.env` sets `ANTHROPIC_API_KEY` and dotenv loads it into
  `os.environ`; the CLI gives an env key precedence over subscription OAuth,
  so an unscrubbed subprocess silently bills the API key while the user
  believes they chose the subscription. A test asserts the scrub.
- **Set `CLAUDE_CODE_MAX_OUTPUT_TOKENS=<max_tokens>`** — the CLI has no
  `--max-tokens` flag; this env var is how the orchestrator's existing
  `max_tokens` setting reaches the CLI path. (The truncation-repair logic in
  `_parse_json_response` already handles a hard cut, same as the API path.)
- `--bare` keeps hooks, skills, MCP servers, and CLAUDE.md out of the call —
  a scripted completion, not an agent session. `--max-turns 1` forecloses
  tool-use loops.
- `timeout=` from new config `llm.cli_timeout_seconds` (default **600**;
  Opus passes with 4k output tokens are slow, and the API SDK's own default
  timeout is in the same range).

Result parsing — the envelope from `--output-format json`:

| Envelope field | Use |
|---|---|
| `is_error: true` | → `TransportError` carrying the envelope's message |
| `result` | → `TransportResponse.text` (missing → `TransportError`) |
| `usage.input_tokens` / `usage.output_tokens` | → token counts, `tokens_basis: "reported"`; if absent, estimate at len(prompt)//4 and len(text)//4 with `tokens_basis: "estimated"` — never silently zero, per the layer's absence-must-not-read-as-a-reading rule |
| `usage.cache_read_input_tokens` / `usage.cache_creation_input_tokens` | → carried into the usage entry as their own fields. The envelope's `input_tokens` **excludes** cache reads, so against a 40KB extraction prompt the CLI path can report a token count that looks absurdly small beside the API path's — a phantom efficiency unless the cache fields travel with it. Cost stays truthful regardless via `total_cost_usd`; `tokens_basis: "reported"` on this path means "reported, excluding cache" and the field's docstring says so |
| `total_cost_usd` | → `cost_usd` (actual) |

Non-zero exit or undecodable stdout → `TransportError` with an excerpt of
stderr (auth failures — not logged in, credit pool exhausted — surface here
with the CLI's own message, which is the most accurate statement available).

**Latency expectation:** every call pays a process spawn plus CLI startup —
seconds before the model speaks. A Pass 1+2 pair won't notice; the
extraction sweep's dozens-to-hundreds of calls will, materially — expect a
full-corpus sweep's wall clock to grow by minutes on this path.
`cli_timeout_seconds: 600` is a ceiling for slow Opus completions, not the
typical case, and must not be read as one.

Constructor precondition: `shutil.which(binary)` where `binary` is new
config `llm.claude_binary` (default `"claude"`). Missing →
`ValueError("claude CLI not found on PATH — install Claude Code and log in, "
"or set llm.provider: anthropic")`. Login state is NOT preflighted (there is
no free check); the first call surfaces it as a normal per-pass error.

**Step 0 of implementation: install and log in Claude Code.** As of
2026-08-08 this machine has no `claude` binary on PATH or in any common
install location (`~/.claude` exists, so it was once present; the binary is
gone). The precondition path degrades gracefully without it, but none of the
verifications below can run until it is installed and authenticated with the
Max subscription.

**Implementation-time verifications** (flags confirmed against docs
2026-08-08, but verify against the installed CLI before relying on them):
`--bare` (recent enough that once installed, the minimum CLI version that
supports it should be **pinned in this plan and in `config.yaml`'s comment**,
not left as "verify flags"), `--max-turns` in print mode, envelope `usage`
field presence and its cache-token fields, `CLAUDE_CODE_MAX_OUTPUT_TOKENS`,
and **model-id pass-through** — `--model` must accept the full Anthropic ids
the config carries (`claude-sonnet-4-6`, `claude-opus-4-6`) verbatim, not
just aliases, since `--deep`'s semantics depend on it. Any that differ:
adapt the transport, not the seam.

### Orchestrator changes (`llm_layer/orchestrator.py`)

- `__init__`: replace the API-key check + `anthropic.Anthropic` construction
  with `self.transport = build_transport(config)`. Everything else (model
  resolution, budgets, deep-mode toggle) is untouched — **model selection
  stays orchestrator-side**, so `--deep` keeps meaning "swap to Opus"
  identically on both providers.
- `_call_api(model, prompt, pass_name)` keeps its name and signature, but its
  body becomes: call `self.transport.complete(model, prompt, self.max_tokens)`,
  log usage, parse JSON. `except TransportError` replaces
  `except anthropic.APIError` (the broad `except Exception` stays).
- The `import anthropic` moves into the API transport; the orchestrator
  itself no longer imports the SDK — on the `claude_cli` path the SDK is
  never touched.
- **Usage entries grow new fields**: `cost_usd` (actual, CLI only),
  `tokens_basis`, `provider`, and — CLI path only — the cache-token counts
  (`cache_read_input_tokens`, `cache_creation_input_tokens`), so the two
  providers' usage blocks stay comparable. `_summarize_usage()` prefers actual cost
  when an entry carries one, falling back to `estimate_cost()`, and adds
  a summary-level `cost_basis: "actual" | "estimated" | "mixed"` plus
  `provider`. The `estimated_cost_usd` key **keeps its name** — the sweep's
  ceiling meters on it and renaming would break that contract — but on the
  CLI path it now holds actuals, which the `cost_basis` field states
  honestly (the same estimate-vs-recorded honesty pattern `friction.basis`
  uses).

### Config surface (`config.yaml`)

```yaml
llm:
  provider: "anthropic"        # or "claude_cli" — headless Claude Code,
                               # billed to the Max subscription's monthly
                               # headless-credit pool instead of API credits.
                               # Overridable per run: --llm-provider
  claude_binary: "claude"      # claude_cli only; must be on PATH
  cli_timeout_seconds: 600     # claude_cli only; per-call subprocess timeout
```

All other keys (`pass1_model`, `pass2_model`, `forward_growth_model`,
`max_tokens`, budgets) apply identically to both providers.

### CLI surface (`cli.py`)

`--llm-provider` on the two commands that spend LLM money:

```
python -m boundless100x analyze ASTRAL --deep --llm-provider claude_cli
python -m boundless100x sweep --all --llm-provider claude_cli
```

Threading is a composition-root config override — no service-signature
change: the command loads config via the existing `load_config()`, sets
`config["llm"]["provider"] = value` when the flag was given, and constructs
`Boundless100xService(config=config)` (`service.py:68`; the constructor
already accepts an injected dict). Typer validates the two literals via an enum. The `analyze`
banner that currently prints `(DEEP — Opus)` also names the provider when it
is `claude_cli`, so a run's billing path is visible in its first line of
output.

Commands that never call the model on their own (`watchlist advance` runs
`use_llm=False`; `compute`; `backtest`) get no flag — there is nothing for
it to select. `sweep --dry-run` keeps pricing from the `MODEL_PRICING`
table on both providers: a dry run estimates by construction (the ceiling
warning for unpriceable models stays as is).

### Service changes (`service.py`)

None beyond what falls out: the `except ValueError → "LLM not available"`
catch at construction already covers both transports' precondition failures.
The lazy-init comment ("only when API key is available") is updated to name
both preconditions.

## Failure modes

| Failure | Behaviour |
|---|---|
| `claude_cli` chosen, binary missing | `ValueError` at construction → service warns "LLM not available", pipeline continues compute-only (exactly today's missing-key behaviour) |
| Not logged in / credit pool exhausted | First call fails; per-pass `{"error": ...}` dict with the CLI's stderr message; run continues, error lands in `result.errors` |
| Subprocess timeout | `TransportError` → same per-pass error dict |
| Envelope unparseable / `is_error` | Same |
| `ANTHROPIC_API_KEY` set AND `claude_cli` chosen | Works, subscription-billed — the scrub guarantees the key cannot leak into the child env |
| Unknown `llm.provider` value | `ValueError` at construction naming the two legal values |

## What must not change (pinned by tests)

1. **Regime hashes**: flipping `llm.provider` moves neither `registry_hash`
   nor `forward_signal_hash`.
2. **Prompt bytes**: both transports receive the identical rendered prompt
   (test: capture at the seam under each provider, assert equality).
3. **Sidecar identity**: `_version_block` output is provider-independent; a
   provider switch does not invalidate a cached extraction.
4. **Result shape**: for the same model text, both paths produce the same
   parsed dict and the same error-dict shape on failure.
5. **Score history**: no row content changes; `synthetic`/verdict/composite
   are upstream of and untouched by transport.

## Testing plan

New file `tests/test_llm_transport.py`, all subprocess/SDK mocked (no
network, no CLI dependency in CI):

- **CLI transport**: envelope happy path (text, tokens `reported`, actual
  cost, cache-token fields carried through); missing `usage` → estimated
  tokens with `tokens_basis: "estimated"`;
  `is_error` envelope, non-zero exit, timeout, garbage stdout → each a
  `TransportError`; **env scrub** — assert the captured `env` passed to
  `subprocess.run` lacks all three `ANTHROPIC_*` keys and carries
  `CLAUDE_CODE_MAX_OUTPUT_TOKENS`; prompt arrives via stdin.
- **API transport**: behaviour-preservation against a mocked SDK client
  (same call args as today, `TransportError` wraps `APIError`).
- **`build_transport`**: both literals resolve; unknown value raises; missing
  binary / missing key raise `ValueError` with actionable messages.
- **Orchestrator integration**: `_call_api` returns identical parsed output
  under a fake transport regardless of provider; usage entries carry
  `provider`/`cost_usd`/`tokens_basis`; `_summarize_usage` mixes actual and
  estimated correctly and reports `cost_basis`.
- **Hash invariance**: provider flip, both hashes byte-identical.
- **CLI threading**: `--llm-provider claude_cli` reaches
  `config["llm"]["provider"]` of the constructed service (typer runner).
- **Sweep**: ceiling still binds when usage carries actual costs.

## Docs

- `CLAUDE.md`: one bullet under Key Patterns (the transport seam and the
  provider-is-not-contract rule), `--llm-provider` in the Commands block,
  and a note in the environment section that `claude_cli` needs a logged-in
  Claude Code and bills the subscription's headless pool, not API credits.
- `config.yaml` comments as shown above.

## Rollout

Default stays `anthropic` — merged, the change is invisible until someone
passes the flag or edits the config. The live check below requires Step 0
(Claude Code installed and logged in) to have happened first; on a machine
without the binary the sequence stops at its first step, which is itself
the graceful-degradation check. Suggested order:
`analyze ASTRAL --llm-provider claude_cli --no-llm` (no calls; with no
binary this proves the compute-only fallback, with one it proves
construction), then `analyze ASTRAL --llm-provider claude_cli` (Sonnet),
then `--deep` (Opus), comparing the report's usage block for `provider`,
`cost_basis: "actual"`, and cache-token fields.
