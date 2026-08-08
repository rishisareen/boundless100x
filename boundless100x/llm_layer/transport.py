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

# Rule-of-thumb characters per token for English prose, and the **one**
# definition of it in the repo: `llm_layer/sweep.py` prices its dry run with this
# same number, and a second copy living here drifted from it immediately — this
# module's own fallback estimate was written at 4, the exact divisor the sweep's
# constant had already been measured down from.
#
# Deliberately a round number and deliberately named: the estimate is an
# estimate, and tuning it to three decimal places would only make it look like a
# quote. 4 was the Sonnet 4.6 figure. The Claude 5 family tokenizes the same text
# to roughly 30% more tokens, so 4 leaves every estimate about a third *under*
# the bill — wrong in the direction that matters, since the sweep's `--ceiling`
# meters on numbers derived from it. 3 rounds the other way, so an estimate built
# on it reads slightly high.
CHARS_PER_TOKEN = 3

# ── The child environment ─────────────────────────────────────────────────────
#
# **This is the load-bearing control of the whole provider feature**, and its
# *shape* is the control rather than its contents. The project's `.env` sets
# `ANTHROPIC_API_KEY` and dotenv loads it into `os.environ`; the CLI gives an env
# key precedence over subscription OAuth, and a whole family of other variables
# (`CLAUDE_CODE_USE_BEDROCK`, `CLAUDE_CODE_USE_VERTEX`, `AWS_BEARER_TOKEN_BEDROCK`,
# `ANTHROPIC_BEDROCK_BASE_URL`, …) redirects a call to a third-party provider
# that bills its own credentials — the CLI's own help is explicit that
# "3P providers (Bedrock/Vertex/Foundry) use their own credentials". Any of them
# leaking into the child bills the wrong pool while looking exactly like success.
#
# This was a three-name denylist and is now an allowlist, because a denylist is
# the wrong shape for a control whose failure is silent and financial: it has to
# be re-audited against every CLI release, and `config.yaml` pins a *minimum*
# version, not a maximum. The asymmetry settles it — starve the child of
# something it needs and the call fails loudly with the CLI's own message, one
# name to add; miss a routing variable and the bill goes somewhere else quietly.
#
# Every name below is here for a stated reason. Add to it when a real invocation
# proves it necessary, not pre-emptively.
INHERITED_ENV_KEYS = frozenset(
    {
        # Finding and running the binary at all.
        "PATH",
        "SHELL",
        "PWD",
        "TMPDIR",
        # Who the process is, and therefore where its credentials are: Claude
        # Code reads the subscription token from the login keychain and its
        # state from `~/.claude`, both located from HOME.
        "HOME",
        "USER",
        "LOGNAME",
        # …unless the owner has moved `~/.claude`. `--setting-sources ""`
        # already neutralises settings files, so what this contributes is the
        # credential store's location; drop it on a relocated install and a
        # perfectly good login reads as logged out.
        "CLAUDE_CONFIG_DIR",
        # Corporate egress. A machine that reaches the network only through a
        # proxy or a private CA bundle fails at the socket without these, and
        # the resulting error names a refused connection rather than a cause.
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "no_proxy",
        "NODE_EXTRA_CA_CERTS",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        # The CLI is a Node program and `--max-old-space-size` is a real thing
        # owners set on it. It cannot redirect where a call goes or who pays.
        "NODE_OPTIONS",
        # How it renders what it prints back.
        "TERM",
        "LANG",
        "TZ",
    }
)

# Locale, which arrives as an open-ended family (`LC_ALL`, `LC_CTYPE`, …) that
# cannot be enumerated. Nothing under it can route a call.
INHERITED_ENV_PREFIXES = ("LC_",)

# Never inherited, whatever the allowlist says. Strictly redundant today — the
# allowlist above names none of these — and kept precisely for the day someone
# widens that list: a prefix rule covers a routing variable that does not exist
# yet without a code change, which is the property the old denylist could not
# have. `AWS_`/`GOOGLE_` are here because Bedrock and Vertex authenticate from
# the ambient cloud credentials rather than from an `ANTHROPIC_*` name.
BILLING_ROUTING_PREFIXES = (
    "ANTHROPIC_",
    "CLAUDE_CODE_USE_",
    "AWS_",
    "GOOGLE_",
)

# The two basis vocabularies, named for the same reason `friction.BASIS_*` are:
# both are compared against and defaulted to across four modules, and a typo in
# any one of them ("estimate" for "estimated") breaks a comparison silently
# rather than loudly. They are separate sets that happen to share a word —
# tokens are reported-or-estimated-or-unknown, cost is metered-or-priced-or-both.
TOKENS_BASIS_REPORTED = "reported"
TOKENS_BASIS_ESTIMATED = "estimated"
# A call that failed before anyone could count it. Distinct from "estimated",
# which still carries a number: this one says there is no number, which is why
# the usage entry it tags leaves the token counts `None` rather than 0.
TOKENS_BASIS_UNKNOWN = "unknown"

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
    """Any failure to obtain text from the model, whichever transport ran.

    **A failed call is not a free call.** The CLI writes Claude Code's own
    harness prefix before the model reads a word of ours — a measured $0.033 to
    $0.065 — so a call that fails *after* that has already billed the pool. The
    envelope says so in `total_cost_usd`, and that number survives here rather
    than dying with the exception: without it the spend vanishes from
    `usage_summary()` entirely, which means the sweep's `--ceiling` never meters
    it (a run of failing-but-billed calls walks straight through the ceiling
    across the dozens of serial calls a full sweep makes) and `_extract_one`
    reports the ticker at $0.0000 when it was not free.

    `None` means "not known to have been billed" — never "known to be free".
    The API path always leaves it `None`: that transport reports no cost at all,
    successful or otherwise, and cost there is reconstructed from
    `MODEL_PRICING`.
    """

    def __init__(self, message: str, cost_usd: float | None = None):
        super().__init__(message)
        self.cost_usd = cost_usd


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
                f"{(completed.stderr or '').strip()[:_STDERR_EXCERPT]}",
                # A non-zero exit is not evidence that nothing was spent, and
                # the CLI often prints a well-formed envelope on stdout before
                # failing. Decoding it here is the only chance to keep a cost it
                # already reported; the alternative is a call that billed real
                # money and reports nothing at all.
                cost_usd=self._reported_cost(completed.stdout),
            )

        return self._read_envelope(completed.stdout, model, prompt)

    def _child_env(self, max_tokens: int) -> dict:
        """The child's whole environment, built up from the allowlist.

        Built *up* rather than filtered down: the constructive form is what
        makes a routing variable nobody has heard of yet unable to reach the
        child. See `INHERITED_ENV_KEYS` for why the shape matters more than the
        contents, and `BILLING_ROUTING_PREFIXES` for the backstop under it.
        """
        env = {
            key: value
            for key, value in os.environ.items()
            if (key in INHERITED_ENV_KEYS or key.startswith(INHERITED_ENV_PREFIXES))
            and not key.startswith(BILLING_ROUTING_PREFIXES)
        }
        # The CLI has no `--max-tokens` flag; this env var is the only route the
        # orchestrator's configured ceiling has to reach it.
        env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = str(max_tokens)
        env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"
        return env

    def _reported_cost(self, stdout: str) -> float | None:
        """`total_cost_usd` off a stdout that may not be an envelope at all.

        Deliberately silent about failure: this runs on paths that are *already*
        raising, and the error being raised is the accurate statement of what
        went wrong. All this adds is the spend, when the CLI managed to say.
        """
        try:
            envelope = json.loads(stdout or "")
        except (json.JSONDecodeError, TypeError):
            return None
        if not isinstance(envelope, dict):
            return None
        return _as_float(envelope.get("total_cost_usd"))

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

        # Both failure branches below carry the envelope's own cost. An
        # `is_error` envelope is a call that reached the harness, wrote its
        # prefix, and *then* failed — the most expensive kind of failure there
        # is on this path, and the one whose spend used to be discarded.
        billed = _as_float(envelope.get("total_cost_usd"))

        if envelope.get("is_error"):
            detail = envelope.get("result") or envelope.get("terminal_reason") or ""
            raise TransportError(
                f"{self.binary} reported an error: {detail}", cost_usd=billed
            )

        text = envelope.get("result")
        if not isinstance(text, str):
            raise TransportError(
                f"{self.binary} envelope carried no result text "
                f"(terminal_reason={envelope.get('terminal_reason')!r})",
                cost_usd=billed,
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

        # Half a reading is not a reading, and a zero would read as free. This
        # is also the branch where the estimate is the *only* number anyone
        # has — it feeds the displayed totals and, when `total_cost_usd` is
        # unreadable too, `estimate_cost()` and therefore the sweep's ceiling —
        # so it divides by the measured `CHARS_PER_TOKEN` the sweep prices with
        # rather than by a second, more optimistic constant of its own.
        return TransportResponse(
            text=text,
            input_tokens=len(prompt) // CHARS_PER_TOKEN,
            output_tokens=len(text) // CHARS_PER_TOKEN,
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
