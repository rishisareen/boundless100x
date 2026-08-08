"""How an LLM call leaves this machine — the Anthropic API, or Claude Code.

**The provider is a transport, not a contract.** Both implementations receive
the identical rendered prompt and both return text that flows through the same
`_parse_json_response`. Nothing downstream — parsing, validation, grounding, the
sidecar's version block, the action guard, either regime hash — can tell which
one ran except by reading the usage metadata. That is what keeps the feature
additive: if the two paths could diverge in what they submit or how their output
is read, every consumer would inherit a second behaviour to reason about.

Two consequences worth stating where the code is:

- **The CLI's `--json-schema` is deliberately unused.** It would be a second
  statement of each pass's output schema beside the one already in the prompt
  text, and the two would drift invisibly. The prompts stay single-sourced.
- **Every failure is a `TransportError`.** The orchestrator catches it exactly
  where it used to catch `anthropic.APIError` and returns the same
  `{"error": ..., "pass": ...}` dict, so no consumer learns a new failure shape.

The CLI path bills the Max subscription's headless-credit pool rather than API
credits, which is the whole point of it existing. It is not free and it is not
cheaper per token: Claude Code writes every prompt token at 1-hour-TTL
cache-write rates (2× standard input price), and each fresh invocation pays a
measured ~$0.033–0.065 of harness prefix before the model reads a word of ours.
What changes is which pool the dollars come from.
"""

import json
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

DEFAULT_CLAUDE_BINARY = "claude"

# Sized when `max_tokens` was 4096; it is now 16000 with thinking on by default
# on the Claude 5 family, so a slow Opus completion has materially more room to
# run than when this number was chosen. It is a ceiling for the worst case, not
# a description of the typical call.
DEFAULT_CLI_TIMEOUT_SECONDS = 600

# The load-bearing line of the whole provider feature. The project's `.env` sets
# `ANTHROPIC_API_KEY` and dotenv loads it into `os.environ`; the CLI gives an env
# key precedence over subscription OAuth. An unscrubbed child would silently bill
# API credits while the owner believed they had chosen the subscription — and
# would look exactly like success either way.
SCRUBBED_ENV_KEYS = frozenset(
    {"ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL"}
)

# The two basis vocabularies, named for the same reason `friction.BASIS_*` are:
# both are compared against and defaulted to across four modules, and a typo in
# any one of them ("estimate" for "estimated") breaks a comparison silently
# rather than loudly. They are separate sets that happen to share a word —
# tokens are reported-or-estimated, cost is metered-or-priced-or-both.
TOKENS_BASIS_REPORTED = "reported"
TOKENS_BASIS_ESTIMATED = "estimated"

COST_BASIS_ACTUAL = "actual"
COST_BASIS_MIXED = "mixed"
COST_BASIS_ESTIMATED = "estimated"

# How much of a subprocess's stderr travels in an error message. Auth failures
# and exhausted credit pools surface here in the CLI's own words, which is the
# most accurate statement available; a stack trace's worth of it is not.
_STDERR_EXCERPT = 500


class LLMProvider(str, Enum):
    """The one vocabulary, shared by `config.yaml` and `--llm-provider`.

    A `str` enum so a typer option, a YAML value and a dict key are the same
    literal everywhere — no mapping table to fall out of agreement with itself.
    """

    ANTHROPIC = "anthropic"
    CLAUDE_CLI = "claude_cli"


class TransportError(RuntimeError):
    """Any failure to obtain text from the model, whichever transport ran."""


@dataclass
class TransportResponse:
    """What a completed call yields, in terms both transports can state.

    `tokens_basis` is `"reported"` when the transport was told the counts and
    `"estimated"` when it had to infer them from character length. It is never
    silently zero: a zero token count reads as a free call, and the layer's rule
    is that an absent reading must not look like a reading of zero.

    On the CLI path `"reported"` means **reported, excluding cache reads** — the
    envelope's `input_tokens` counts only what was written fresh, so a call that
    moved 37K tokens can report `input_tokens: 2`. The cache counts below travel
    beside it precisely so that number cannot be mistaken for efficiency.

    `cost_usd` is the actual metered cost when the transport knows it (the CLI
    reports one) and `None` when it does not (the API path, where cost is
    reconstructed from `MODEL_PRICING`).
    """

    text: str
    input_tokens: int
    output_tokens: int
    cost_usd: float | None = None
    tokens_basis: str = TOKENS_BASIS_REPORTED
    cache_read_input_tokens: int | None = None
    cache_creation_input_tokens: int | None = None


class AnthropicAPITransport:
    """Today's path: the `anthropic` SDK against pay-as-you-go API credits."""

    name = LLMProvider.ANTHROPIC.value

    def __init__(self, config: dict):
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            # Word-for-word what `LLMOrchestrator.__init__` used to raise, so
            # `service.py`'s existing `except ValueError` catch keeps printing
            # the message the owner already knows.
            raise ValueError(
                "ANTHROPIC_API_KEY environment variable not set. "
                "Set it or disable LLM with llm.enabled: false in config.yaml"
            )

        # Imported here rather than at module scope so the `claude_cli` path
        # never touches the SDK at all — a machine that has Claude Code and no
        # API credentials should not need the package to be importable.
        import anthropic

        self._anthropic = anthropic
        self.client = anthropic.Anthropic(api_key=api_key)

    def complete(self, model: str, prompt: str, max_tokens: int) -> TransportResponse:
        try:
            response = self.client.messages.create(
                model=model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
        except self._anthropic.APIError as e:
            raise TransportError(str(e)) from e

        return TransportResponse(
            text=response.content[0].text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cost_usd=None,
            tokens_basis=TOKENS_BASIS_REPORTED,
        )


class ClaudeCLITransport:
    """Headless Claude Code (`claude -p`), billed to the Max subscription.

    The invocation below is measured rather than derived from the docs, on CLI
    **2.1.225**, and is version-specific in both directions:

    - **`--bare` is absent on purpose.** It was the obvious flag — the thing
      that would make this "a scripted completion, not an agent session" — and
      on 2.1.225 it breaks subscription auth outright: `is_error: true`,
      `result: "Not logged in · Please run /login"`, `duration_api_ms: 0`, on a
      machine whose keychain holds a valid token and where the identical command
      without it succeeds. `--setting-sources ""` plus an empty strict MCP config
      is the closest reachable substitute for its intent, and authenticates.
    - **`--mcp-config` needs `{"mcpServers":{}}`**, not `{}`.
    - **`--tools ""`** strips the built-in tool schemas: $0.0651 → $0.0370 on a
      two-token probe. With `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1` (which
      removes the harness's auxiliary haiku call) the same probe costs $0.0331.
    - **`--system-prompt` is not passed.** It looks like the way to shrink the
      prefix and does the opposite — overriding it invalidates the server-side
      prompt cache and the same probe went to $0.1711 with `cache_read: 0`.

    The remaining ~$0.033 is a per-call floor across independent sessions: an
    immediately repeated identical call pays it again. It amortizes only inside
    a resumed session (`--resume`, measured at $0.0030 for a follow-up turn),
    which is a session-per-company design and deliberately v2 of this transport.

    Login state is not preflighted — there is no free check — so a logged-out
    machine surfaces as a normal per-call error carrying the CLI's own message.
    """

    name = LLMProvider.CLAUDE_CLI.value

    def __init__(self, config: dict):
        llm_config = config.get("llm") or {}
        self.binary = llm_config.get("claude_binary", DEFAULT_CLAUDE_BINARY)
        self.timeout = llm_config.get(
            "cli_timeout_seconds", DEFAULT_CLI_TIMEOUT_SECONDS
        )
        if not shutil.which(self.binary):
            raise ValueError(
                f"{self.binary} CLI not found on PATH — install Claude Code and "
                "log in, or set llm.provider: anthropic"
            )

    def complete(self, model: str, prompt: str, max_tokens: int) -> TransportResponse:
        argv = [
            self.binary,
            "-p",
            "--model",
            model,
            "--output-format",
            "json",
            "--setting-sources",
            "",
            "--strict-mcp-config",
            "--mcp-config",
            json.dumps({"mcpServers": {}}),
            "--tools",
            "",
        ]

        try:
            completed = subprocess.run(
                argv,
                # On stdin, never as an argv item: extraction prompts run to
                # tens of KB, which is a quoting and ARG_MAX problem waiting to
                # happen.
                input=prompt,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env=self._child_env(max_tokens),
            )
        except subprocess.TimeoutExpired as e:
            raise TransportError(
                f"{self.binary} timed out after {self.timeout}s"
            ) from e
        except OSError as e:
            raise TransportError(f"could not run {self.binary!r}: {e}") from e

        if completed.returncode != 0:
            raise TransportError(
                f"{self.binary} exited {completed.returncode}: "
                f"{(completed.stderr or '').strip()[:_STDERR_EXCERPT]}"
            )

        return self._read_envelope(completed.stdout, model, prompt)

    def _child_env(self, max_tokens: int) -> dict:
        env = {k: v for k, v in os.environ.items() if k not in SCRUBBED_ENV_KEYS}
        # The CLI has no `--max-tokens` flag; this env var is the only route the
        # orchestrator's configured ceiling has to reach it.
        env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = str(max_tokens)
        env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"
        return env

    def _read_envelope(self, stdout: str, model: str, prompt: str) -> TransportResponse:
        try:
            envelope = json.loads(stdout)
        except json.JSONDecodeError as e:
            raise TransportError(
                f"could not decode the {self.binary} JSON envelope: {e}"
            ) from e
        if not isinstance(envelope, dict):
            raise TransportError(
                f"could not decode the {self.binary} JSON envelope: not an object"
            )

        if envelope.get("is_error"):
            detail = envelope.get("result") or envelope.get("terminal_reason") or ""
            raise TransportError(f"{self.binary} reported an error: {detail}")

        text = envelope.get("result")
        if not isinstance(text, str):
            raise TransportError(
                f"{self.binary} envelope carried no result text "
                f"(terminal_reason={envelope.get('terminal_reason')!r})"
            )

        self._warn_on_unexpected_model(envelope, model)

        usage = envelope.get("usage")
        usage = usage if isinstance(usage, dict) else {}
        input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")
        if isinstance(input_tokens, int) and isinstance(output_tokens, int):
            return TransportResponse(
                text=text,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=_as_float(envelope.get("total_cost_usd")),
                tokens_basis=TOKENS_BASIS_REPORTED,
                cache_read_input_tokens=_as_int(usage.get("cache_read_input_tokens")),
                cache_creation_input_tokens=_as_int(
                    usage.get("cache_creation_input_tokens")
                ),
            )

        # Half a reading is not a reading, and a zero would read as free.
        return TransportResponse(
            text=text,
            input_tokens=len(prompt) // 4,
            output_tokens=len(text) // 4,
            cost_usd=_as_float(envelope.get("total_cost_usd")),
            tokens_basis=TOKENS_BASIS_ESTIMATED,
        )

    def _warn_on_unexpected_model(self, envelope: dict, model: str) -> None:
        """`modelUsage` names what actually ran, so it can be checked rather than trusted.

        A warning and not an error: the harness makes auxiliary calls of its own
        (a `claude-haiku-4-5` entry appears beside the requested model), so the
        block is a superset by design and a hard assertion would fail on a
        perfectly good call.
        """
        model_usage = envelope.get("modelUsage")
        if isinstance(model_usage, dict) and model_usage and model not in model_usage:
            logger.warning(
                f"{self.binary} was asked for {model!r} but reports usage for "
                f"{sorted(model_usage)} — the served model may differ"
            )


def _as_int(value) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _as_float(value) -> float | None:
    return (
        float(value)
        if isinstance(value, (int, float)) and not isinstance(value, bool)
        else None
    )


TRANSPORTS = {
    LLMProvider.ANTHROPIC.value: AnthropicAPITransport,
    LLMProvider.CLAUDE_CLI.value: ClaudeCLITransport,
}


def build_transport(config: dict) -> AnthropicAPITransport | ClaudeCLITransport:
    """Resolve `llm.provider` to a constructed transport.

    Both a precondition failure (no API key, no `claude` on PATH) and an unknown
    provider value raise `ValueError` — the same type `service.py` already
    catches to warn "LLM not available" and continue compute-only, so a
    misconfigured provider degrades exactly the way a missing key does.
    """
    provider = (config.get("llm") or {}).get("provider", LLMProvider.ANTHROPIC.value)
    transport_class = TRANSPORTS.get(provider)
    if transport_class is None:
        legal = ", ".join(repr(value) for value in TRANSPORTS)
        raise ValueError(f"unknown llm.provider {provider!r} — legal values are {legal}")
    return transport_class(config)
