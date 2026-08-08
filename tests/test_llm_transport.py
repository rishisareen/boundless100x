"""The provider seam: two transports, one contract.

Every LLM call in the pipeline can run through the Anthropic API (pay-as-you-go
credits) or through headless Claude Code (`claude -p`, billed against the Max
subscription's monthly pool). **The provider is a transport, not a contract**:
both receive the identical rendered prompt and both return text that flows
through the same `_parse_json_response`. Nothing downstream — parsing,
validation, grounding, the sidecar, the action guard, either regime hash — may
be able to tell which one ran except by reading the usage metadata. That is the
whole reason the feature is additive rather than a second behaviour every
consumer has to reason about, and most of this file exists to pin it.

Four things here are not cosmetic:

**The child environment is the load-bearing control of the feature.** The
project's `.env` sets `ANTHROPIC_API_KEY` and dotenv loads it into `os.environ`;
the CLI gives an env key precedence over subscription OAuth, and a whole family
of other variables reroutes a call to a third-party provider billing its own
credentials. Any of them reaching the child bills the wrong pool — a failure
that costs money and looks exactly like success. It is an allowlist for that
reason, and the tests below assert the *property* (nothing that could reroute
the bill gets through, including names that do not exist yet) rather than the
contents of a list, which would pass no matter which variable is discovered
next.

**Absent tokens are estimated, never zeroed.** A zero token count reads as a
free call; the layer's rule is that absence must not read as a reading, so a
missing `usage` block estimates and says `tokens_basis: "estimated"`.

**Cache fields must travel — all the way to a surface.** The CLI envelope's
`input_tokens` *excludes* cache reads — a call that moved ~37K tokens reported
`input_tokens: 2` — so a usage block without the cache counts beside it shows a
phantom efficiency the API path can never match. Carrying them per-call and then
summing only `input_tokens` defeats the defence one layer above where it was
built, so the totals and the rendered lines are pinned too.

**A failed call is not a free call.** The CLI bills its harness prefix before
the model reads a word of ours; a failure after that has already spent. The
cost travels on the exception and into the usage log, or the sweep's `--ceiling`
meters a number that omits it.

No test here touches the network or needs a `claude` binary: the SDK client and
`subprocess.run` are both faked.
"""

import json
import subprocess
from types import SimpleNamespace

import pytest

from boundless100x.llm_layer.transport import (
    BILLING_ROUTING_PREFIXES,
    CHARS_PER_TOKEN,
    COST_BASIS_ACTUAL,
    COST_BASIS_ESTIMATED,
    TOKENS_BASIS_UNKNOWN,
    AnthropicAPITransport,
    ClaudeCLITransport,
    LLMProvider,
    TransportError,
    TransportResponse,
    build_transport,
)
# The sweep's own stub and corpus, reused rather than rebuilt: the ceiling
# behaviour under actual costs is a property of the same code path those
# fixtures already exercise under estimated ones.
from tests.test_extraction_sweep import RecordingLLM, corpus, service  # noqa: F401

CLI_CONFIG = {"llm": {"provider": "claude_cli"}}


# ── Fakes ──────────────────────────────────────────────────────────────────


def envelope(result: str = '{"ok": true}', **overrides) -> dict:
    """The `--output-format json` envelope, shaped as 2.1.225 emits it.

    Field-for-field from a real recorded call, including the `modelUsage` entry
    for the auxiliary haiku call the harness makes on its own — one more reason
    per-call cost on this path is not a clean per-request figure.
    """
    payload = {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "result": result,
        "duration_api_ms": 4120,
        "total_cost_usd": 0.0331,
        "usage": {
            "input_tokens": 2,
            "output_tokens": 7,
            "cache_read_input_tokens": 28083,
            "cache_creation_input_tokens": 5349,
        },
        "modelUsage": {"claude-sonnet-5": {}, "claude-haiku-4-5": {}},
    }
    payload.update(overrides)
    return payload


class FakeRun:
    """Stands in for `subprocess.run`, recording exactly how it was called."""

    def __init__(self, stdout="", stderr="", returncode=0, raises=None):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode
        self.raises = raises
        self.argv = None
        self.kwargs = None

    def __call__(self, argv, **kwargs):
        self.argv = argv
        self.kwargs = kwargs
        if self.raises is not None:
            raise self.raises
        return subprocess.CompletedProcess(
            argv, self.returncode, self.stdout, self.stderr
        )

    def flag(self, name: str) -> str:
        """The value that followed `name` in the recorded argv."""
        return self.argv[self.argv.index(name) + 1]


@pytest.fixture
def cli_run(monkeypatch):
    """Fake `subprocess.run` plus a `claude` that resolves on PATH."""
    from boundless100x.llm_layer import transport as transport_module

    monkeypatch.setattr(
        transport_module.shutil, "which", lambda binary: f"/usr/local/bin/{binary}"
    )

    def install(**kwargs):
        run = FakeRun(**kwargs)
        monkeypatch.setattr(transport_module.subprocess, "run", run)
        return run

    return install


def cli_transport(config: dict | None = None) -> ClaudeCLITransport:
    return ClaudeCLITransport(config or CLI_CONFIG)


# ── The CLI transport ──────────────────────────────────────────────────────


class TestClaudeCLIEnvelope:
    def test_happy_path_carries_text_tokens_and_actual_cost(self, cli_run):
        cli_run(stdout=json.dumps(envelope('{"verdict": "buy"}')))

        response = cli_transport().complete("claude-sonnet-5", "prompt", 16000)

        assert response.text == '{"verdict": "buy"}'
        assert (response.input_tokens, response.output_tokens) == (2, 7)
        assert response.tokens_basis == "reported"
        assert response.cost_usd == pytest.approx(0.0331)

    def test_cache_counts_travel_with_the_token_counts(self, cli_run):
        """Without these the CLI path reports a phantom efficiency.

        `input_tokens: 2` against a 40KB prompt is not a small call — it is a
        call whose prefix was read from cache. Reported alone beside the API
        path's honest count, it would read as the CLI path being twenty
        thousand times cheaper in tokens.
        """
        cli_run(stdout=json.dumps(envelope()))

        response = cli_transport().complete("claude-sonnet-5", "prompt", 16000)

        assert response.cache_read_input_tokens == 28083
        assert response.cache_creation_input_tokens == 5349

    def test_missing_usage_estimates_rather_than_zeroes(self, cli_run):
        """A zero means free. An estimate means unknown-but-roughly-this.

        The divisor is the sweep's measured `CHARS_PER_TOKEN`, not a second
        constant: this branch fires exactly when the estimate is the only number
        anyone has, and when `total_cost_usd` is unreadable too it feeds
        `estimate_cost()` and therefore the `--ceiling`. It was written at 4 —
        the Sonnet 4.6 figure the sweep had already been measured down from,
        which leaves an estimate about a third under the Claude 5 bill.
        """
        cli_run(stdout=json.dumps(envelope("answer text", usage={})))

        response = cli_transport().complete("claude-sonnet-5", "x" * 400, 16000)

        assert response.tokens_basis == "estimated"
        assert response.input_tokens == 400 // CHARS_PER_TOKEN == 133
        assert response.output_tokens == len("answer text") // CHARS_PER_TOKEN
        assert response.cache_read_input_tokens is None

    def test_the_divisor_is_the_one_the_sweep_prices_with(self):
        """One definition, or the two drift in the direction that under-prices."""
        from boundless100x.llm_layer import sweep as sweep_module

        assert sweep_module.CHARS_PER_TOKEN is CHARS_PER_TOKEN

    def test_partial_usage_is_estimated_too(self, cli_run):
        """Half a reading is not a reading — both counts or neither."""
        cli_run(stdout=json.dumps(envelope(usage={"input_tokens": 2})))

        response = cli_transport().complete("claude-sonnet-5", "x" * 40, 16000)

        assert response.tokens_basis == "estimated"

    def test_prompt_arrives_on_stdin_not_in_argv(self, cli_run):
        """Extraction prompts run to tens of KB — argv is not a place for them."""
        run = cli_run(stdout=json.dumps(envelope()))
        prompt = "MD&A text " * 5000

        cli_transport().complete("claude-sonnet-5", prompt, 16000)

        assert run.kwargs["input"] == prompt
        assert prompt not in run.argv

    def test_model_id_passes_through_verbatim(self, cli_run):
        run = cli_run(stdout=json.dumps(envelope()))

        cli_transport().complete("claude-opus-5", "prompt", 16000)

        assert run.flag("--model") == "claude-opus-5"

    def test_the_invocation_is_the_measured_one(self, cli_run):
        """`--bare` breaks subscription auth on 2.1.225; these three replace it.

        The flags are version-specific in both directions — `--mcp-config`
        needs the `{"mcpServers":{}}` shape rather than `{}` — so the argv is
        pinned rather than left to drift.
        """
        run = cli_run(stdout=json.dumps(envelope()))

        cli_transport().complete("claude-sonnet-5", "prompt", 16000)

        assert "--bare" not in run.argv
        assert run.argv[:2] == ["claude", "-p"]
        assert run.flag("--output-format") == "json"
        assert run.flag("--setting-sources") == ""
        assert "--strict-mcp-config" in run.argv
        assert json.loads(run.flag("--mcp-config")) == {"mcpServers": {}}
        assert run.flag("--tools") == ""


class TestChildEnvironment:
    # Every variable the pinned CLI reads that can change where a call goes or
    # who pays for it. Three of these were the whole of the original denylist;
    # the rest were found by reading the 2.1.225 binary, whose own `--bare` help
    # is explicit that "3P providers (Bedrock/Vertex/Foundry) use their own
    # credentials". None is set on this machine, so the gap this list closes was
    # latent rather than live — which is exactly how it survived review.
    #
    # The list is *evidence*, not the specification: the allowlist in
    # `_child_env` is what actually protects the child, and asserting
    # set-equality against a list of literals (as this test used to) passes no
    # matter which routing variable is discovered next.
    ROUTING_VARIABLES = (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_CUSTOM_HEADERS",
        "ANTHROPIC_CONFIG_DIR",
        "ANTHROPIC_AWS_API_KEY",
        "ANTHROPIC_BEDROCK_BASE_URL",
        "ANTHROPIC_VERTEX_BASE_URL",
        "CLAUDE_CODE_USE_BEDROCK",
        "CLAUDE_CODE_USE_VERTEX",
        "CLAUDE_CODE_USE_FOUNDRY",
        "CLAUDE_CODE_USE_GATEWAY",
        "CLAUDE_CODE_USE_ANTHROPIC_AWS",
        "CLAUDE_CODE_USE_ANTHROPIC_GOOGLE_CLOUD",
        "AWS_BEARER_TOKEN_BEDROCK",
        "AWS_PROFILE",
        "GOOGLE_APPLICATION_CREDENTIALS",
    )

    def _env_after_a_call(self, cli_run, monkeypatch, **environment) -> dict:
        run = cli_run(stdout=json.dumps(envelope()))
        for key, value in environment.items():
            monkeypatch.setenv(key, value)

        cli_transport().complete("claude-sonnet-5", "prompt", 16000)

        return run.kwargs["env"]

    def test_nothing_that_could_reroute_the_bill_reaches_the_child(
        self, cli_run, monkeypatch
    ):
        """The property, not a list: an env key outranks subscription OAuth.

        Choosing `claude_cli` and then billing API credits — or a Bedrock or
        Vertex account — is a failure that costs money and reports success. Any
        one of these surviving is that failure.
        """
        env = self._env_after_a_call(
            cli_run, monkeypatch, **{k: "leaked" for k in self.ROUTING_VARIABLES}
        )

        assert not [k for k in self.ROUTING_VARIABLES if k in env]

    def test_a_routing_variable_nobody_has_heard_of_yet_is_covered(
        self, cli_run, monkeypatch
    ):
        """The property an allowlist has and a denylist structurally cannot.

        `config.yaml` pins a *minimum* CLI version, not a maximum, so the next
        release can add a name this repo has never seen. Under the old denylist
        that name reached the child; under an allowlist it cannot, and this is
        the test that would go red if the shape were ever inverted back.
        """
        env = self._env_after_a_call(
            cli_run,
            monkeypatch,
            ANTHROPIC_FUTURE_ROUTER="leaked",
            CLAUDE_CODE_USE_SOMETHING_UNRELEASED="leaked",
            SOME_VENDOR_INFERENCE_ENDPOINT="leaked",
        )

        assert "ANTHROPIC_FUTURE_ROUTER" not in env
        assert "CLAUDE_CODE_USE_SOMETHING_UNRELEASED" not in env
        assert "SOME_VENDOR_INFERENCE_ENDPOINT" not in env

    def test_the_prefix_rule_holds_even_if_the_allowlist_is_widened(self):
        """The backstop, checked directly because nothing else can reach it.

        `BILLING_ROUTING_PREFIXES` is redundant today — the allowlist names none
        of these families — and exists for the day someone adds a
        `CLAUDE_CODE_*` name to that list and catches a `_USE_` variable with
        it. A redundant guard nobody checks is a guard that quietly stops
        working.
        """
        for key in self.ROUTING_VARIABLES:
            assert key.startswith(BILLING_ROUTING_PREFIXES), key

    def test_the_ordinary_environment_still_survives(self, cli_run, monkeypatch):
        """The allowlist's real risk is starving the child, so pin what it keeps.

        PATH and HOME are the two that decide whether the CLI runs at all and
        whether it finds the login it is supposed to bill; the proxy and CA
        variables are what a machine behind corporate egress needs to reach the
        network; `LC_*` arrives as an open family and is covered by prefix.
        """
        env = self._env_after_a_call(
            cli_run,
            monkeypatch,
            PATH="/usr/bin",
            HOME="/Users/owner",
            HTTPS_PROXY="http://proxy.internal:8080",
            NODE_EXTRA_CA_CERTS="/etc/ssl/corp.pem",
            LC_ALL="en_IN.UTF-8",
        )

        assert env["PATH"] == "/usr/bin"
        assert env["HOME"] == "/Users/owner"
        assert env["HTTPS_PROXY"] == "http://proxy.internal:8080"
        assert env["NODE_EXTRA_CA_CERTS"] == "/etc/ssl/corp.pem"
        assert env["LC_ALL"] == "en_IN.UTF-8"

    def test_an_unrecognised_variable_does_not_reach_the_child(
        self, cli_run, monkeypatch
    ):
        """The cost of the allowlist, stated rather than discovered later.

        Anything not named is dropped, including things that are entirely
        harmless. The trade is deliberate: a starved child fails loudly with the
        CLI's own message and one name to add, while a leaked routing variable
        bills the wrong pool in silence.
        """
        env = self._env_after_a_call(
            cli_run, monkeypatch, EDITOR="vim", MY_PROJECT_DEBUG="1"
        )

        assert "EDITOR" not in env
        assert "MY_PROJECT_DEBUG" not in env

    def test_max_tokens_reaches_the_cli_by_env_var(self, cli_run):
        """The CLI has no `--max-tokens`; this env var is the only route."""
        run = cli_run(stdout=json.dumps(envelope()))

        cli_transport().complete("claude-sonnet-5", "prompt", 12345)

        assert run.kwargs["env"]["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == "12345"

    def test_nonessential_traffic_is_disabled(self, cli_run):
        """Measured: removes the auxiliary haiku call, $0.0651 → $0.0331."""
        run = cli_run(stdout=json.dumps(envelope()))

        cli_transport().complete("claude-sonnet-5", "prompt", 16000)

        assert run.kwargs["env"]["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] == "1"


@pytest.mark.network
def test_the_allowlist_still_reaches_the_subscription():
    """The one check a faked subprocess cannot make, and it costs nothing.

    Every other test in this file asserts what goes *into* the child env. None
    of them can say whether that env is still enough for the real binary to find
    its credentials — and an allowlist's failure mode is exactly that: it starves
    the child, `claude` reports "Not logged in", and every faked test stays
    green. `auth status` answers it without a model call, so this is free to run.

    Deselected by default (`pytest.ini`) because it needs the binary installed
    and logged in. Run it with `-m network` after changing `INHERITED_ENV_KEYS`
    or bumping the pinned CLI version — those are the two things that can break
    the property it pins.

    `apiProvider: firstParty` is the assertion that matters most: it is the CLI
    saying the call would go to Anthropic on the subscription rather than to a
    third-party provider billing its own credentials, which is the whole reason
    the routing variables are excluded.
    """
    import shutil
    import subprocess

    if not shutil.which("claude"):
        pytest.skip("claude CLI not on PATH")

    transport = ClaudeCLITransport({"llm": {"provider": "claude_cli"}})
    completed = subprocess.run(
        [transport.binary, "auth", "status"],
        capture_output=True, text=True, timeout=60,
        env=transport._child_env(16000),
    )

    assert completed.returncode == 0, completed.stderr
    status = json.loads(completed.stdout)
    assert status["loggedIn"] is True
    assert status["apiProvider"] == "firstParty"


class TestClaudeCLIFailures:
    """Every failure is a `TransportError`, so the orchestrator has one catch."""

    def test_error_envelope(self, cli_run):
        cli_run(
            stdout=json.dumps(
                envelope(
                    "Not logged in · Please run /login",
                    is_error=True,
                    terminal_reason="api_error",
                )
            )
        )

        with pytest.raises(TransportError, match="Not logged in"):
            cli_transport().complete("claude-sonnet-5", "prompt", 16000)

    def test_an_error_envelope_carries_the_cost_it_already_billed(self, cli_run):
        """The most expensive kind of failure on this path, not a free one.

        The harness prefix is written before the model reads a word of ours, so
        an `is_error` envelope has usually already spent $0.033–0.065. That
        number was read off the same envelope and thrown away with the
        exception, which took the spend out of `usage_summary()` entirely: the
        sweep's `--ceiling` could not see it, and a run of failing-but-billed
        calls walks straight through the ceiling across the dozens of serial
        calls a full sweep makes.
        """
        cli_run(
            stdout=json.dumps(
                envelope("Not logged in", is_error=True, total_cost_usd=0.0331)
            )
        )

        with pytest.raises(TransportError) as excinfo:
            cli_transport().complete("claude-sonnet-5", "prompt", 16000)

        assert excinfo.value.cost_usd == pytest.approx(0.0331)

    def test_non_zero_exit_carries_stderr(self, cli_run):
        cli_run(returncode=1, stderr="credit balance is too low")

        with pytest.raises(TransportError, match="credit balance is too low"):
            cli_transport().complete("claude-sonnet-5", "prompt", 16000)

    def test_a_non_zero_exit_keeps_a_cost_the_envelope_still_reported(self, cli_run):
        """A non-zero exit is not evidence that nothing was spent.

        The CLI can print a well-formed envelope and then exit non-zero, so the
        decode is attempted before the error is raised — otherwise the one place
        that number exists is discarded on the way past.
        """
        cli_run(
            returncode=1,
            stderr="terminated after reporting",
            stdout=json.dumps(envelope(is_error=True, total_cost_usd=0.05)),
        )

        with pytest.raises(TransportError) as excinfo:
            cli_transport().complete("claude-sonnet-5", "prompt", 16000)

        assert excinfo.value.cost_usd == pytest.approx(0.05)

    def test_an_unreadable_failure_reports_no_cost_rather_than_zero(self, cli_run):
        """`None` means "not known to have been billed", never "known free".

        A 0.0 here would total into the summary as a call that cost nothing,
        which is a claim nobody is in a position to make.
        """
        cli_run(returncode=1, stderr="killed", stdout="not an envelope")

        with pytest.raises(TransportError) as excinfo:
            cli_transport().complete("claude-sonnet-5", "prompt", 16000)

        assert excinfo.value.cost_usd is None

    def test_timeout(self, cli_run):
        cli_run(raises=subprocess.TimeoutExpired(cmd="claude", timeout=600))

        with pytest.raises(TransportError, match="timed out"):
            cli_transport().complete("claude-sonnet-5", "prompt", 16000)

    def test_undecodable_stdout(self, cli_run):
        cli_run(stdout="not json at all")

        with pytest.raises(TransportError, match="could not decode"):
            cli_transport().complete("claude-sonnet-5", "prompt", 16000)

    def test_missing_result_field(self, cli_run):
        cli_run(stdout=json.dumps(envelope(result=None)))

        with pytest.raises(TransportError) as excinfo:
            cli_transport().complete("claude-sonnet-5", "prompt", 16000)

        assert "no result" in str(excinfo.value)
        # Same reasoning as the `is_error` branch: an envelope that reached us
        # at all is a call that reached the harness and paid for its prefix.
        assert excinfo.value.cost_usd == pytest.approx(0.0331)

    def test_a_timeout_claims_no_cost_it_cannot_see(self, cli_run):
        """Nothing was printed, so nothing is known — and unknown is not zero."""
        cli_run(raises=subprocess.TimeoutExpired(cmd="claude", timeout=600))

        with pytest.raises(TransportError) as excinfo:
            cli_transport().complete("claude-sonnet-5", "prompt", 16000)

        assert excinfo.value.cost_usd is None

    def test_binary_that_will_not_run(self, cli_run):
        cli_run(raises=OSError("Exec format error"))

        with pytest.raises(TransportError, match="Exec format error"):
            cli_transport().complete("claude-sonnet-5", "prompt", 16000)

    def test_configured_timeout_is_the_one_used(self, cli_run):
        run = cli_run(stdout=json.dumps(envelope()))

        cli_transport(
            {"llm": {"provider": "claude_cli", "cli_timeout_seconds": 42}}
        ).complete("claude-sonnet-5", "prompt", 16000)

        assert run.kwargs["timeout"] == 42


# ── The API transport ──────────────────────────────────────────────────────


class FakeMessages:
    def __init__(self, outer):
        self._outer = outer

    def create(self, **kwargs):
        self._outer.create_kwargs = kwargs
        if self._outer.raises is not None:
            raise self._outer.raises
        return self._outer.response


class FakeClient:
    def __init__(self, response=None, raises=None):
        self.response = response
        self.raises = raises
        self.create_kwargs = None
        self.messages = FakeMessages(self)


def sdk_response(text='{"ok": true}', input_tokens=1200, output_tokens=800):
    """The three attributes the API transport reads off an SDK response."""
    return SimpleNamespace(
        content=[SimpleNamespace(text=text)],
        usage=SimpleNamespace(
            input_tokens=input_tokens, output_tokens=output_tokens
        ),
    )


@pytest.fixture
def api_transport(monkeypatch):
    """An `AnthropicAPITransport` whose SDK client is a fake."""

    def build(response=None, raises=None):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-used")
        transport = AnthropicAPITransport({})
        transport.client = FakeClient(
            response if response is not None else sdk_response(), raises
        )
        return transport

    return build


class TestAnthropicAPITransport:
    def test_call_args_are_unchanged(self, api_transport):
        """Behaviour preservation: the same three arguments as before the seam."""
        transport = api_transport()

        transport.complete("claude-sonnet-5", "the prompt", 16000)

        assert transport.client.create_kwargs == {
            "model": "claude-sonnet-5",
            "max_tokens": 16000,
            "messages": [{"role": "user", "content": "the prompt"}],
        }

    def test_response_is_reported_and_has_no_actual_cost(self, api_transport):
        """The API path knows tokens but not dollars — cost stays an estimate."""
        transport = api_transport(sdk_response("answer", 1200, 800))

        response = transport.complete("claude-sonnet-5", "prompt", 16000)

        assert response.text == "answer"
        assert (response.input_tokens, response.output_tokens) == (1200, 800)
        assert response.tokens_basis == "reported"
        assert response.cost_usd is None

    def test_api_error_becomes_a_transport_error(self, api_transport):
        import anthropic

        error = anthropic.APIError(
            message="overloaded", request=None, body=None
        )
        transport = api_transport(raises=error)

        with pytest.raises(TransportError, match="overloaded"):
            transport.complete("claude-sonnet-5", "prompt", 16000)

    def test_missing_key_raises_the_message_the_service_already_prints(
        self, monkeypatch
    ):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
            AnthropicAPITransport({})


# ── build_transport ────────────────────────────────────────────────────────


class TestBuildTransport:
    def test_default_is_the_api(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-used")

        assert isinstance(build_transport({}), AnthropicAPITransport)

    def test_both_literals_resolve(self, monkeypatch, cli_run):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-used")
        cli_run()

        assert isinstance(
            build_transport({"llm": {"provider": "anthropic"}}), AnthropicAPITransport
        )
        assert isinstance(build_transport(CLI_CONFIG), ClaudeCLITransport)

    def test_unknown_value_names_the_legal_ones(self):
        with pytest.raises(ValueError) as excinfo:
            build_transport({"llm": {"provider": "openai"}})

        message = str(excinfo.value)
        assert "openai" in message
        assert "anthropic" in message and "claude_cli" in message

    def test_missing_binary_is_actionable(self, monkeypatch):
        from boundless100x.llm_layer import transport as transport_module

        monkeypatch.setattr(transport_module.shutil, "which", lambda binary: None)

        with pytest.raises(ValueError) as excinfo:
            build_transport(CLI_CONFIG)

        message = str(excinfo.value)
        assert "claude" in message
        assert "llm.provider: anthropic" in message

    def test_binary_is_configurable(self, monkeypatch):
        from boundless100x.llm_layer import transport as transport_module

        seen = []
        monkeypatch.setattr(
            transport_module.shutil,
            "which",
            lambda binary: seen.append(binary) or "/opt/claude",
        )

        transport = build_transport(
            {"llm": {"provider": "claude_cli", "claude_binary": "claude-2.1"}}
        )

        assert seen == ["claude-2.1"]
        assert transport.binary == "claude-2.1"

    def test_the_two_literals_are_the_enum(self):
        assert [p.value for p in LLMProvider] == ["anthropic", "claude_cli"]


# ── The orchestrator over a transport ──────────────────────────────────────


class RecordingTransport:
    """A transport that records the prompt and returns whatever it was given."""

    def __init__(self, name: str, response: TransportResponse):
        self.name = name
        self.response = response
        self.prompts: list[str] = []

    def complete(self, model, prompt, max_tokens):
        self.prompts.append(prompt)
        return self.response


def orchestrator_over(transport, config=None):
    from boundless100x.llm_layer.orchestrator import LLMOrchestrator

    orchestrator = LLMOrchestrator.__new__(LLMOrchestrator)
    orchestrator._config = config or {}
    orchestrator.max_tokens = 16000
    orchestrator.transport = transport
    orchestrator._usage_log = []
    return orchestrator


API_RESPONSE = TransportResponse(
    text='{"verdict": "buy"}', input_tokens=1200, output_tokens=800
)
CLI_RESPONSE = TransportResponse(
    text='{"verdict": "buy"}',
    input_tokens=2,
    output_tokens=800,
    cost_usd=0.0331,
    cache_read_input_tokens=28083,
    cache_creation_input_tokens=5349,
)


class TestOrchestratorIntegration:
    def test_parsed_output_is_identical_across_providers(self):
        """The seam's whole claim, tested at the one place it could break."""
        api = orchestrator_over(RecordingTransport("anthropic", API_RESPONSE))
        cli = orchestrator_over(RecordingTransport("claude_cli", CLI_RESPONSE))

        assert api._call_api("claude-sonnet-5", "prompt", "pass1") == cli._call_api(
            "claude-sonnet-5", "prompt", "pass1"
        ) == {"verdict": "buy"}

    def test_both_transports_receive_the_identical_prompt(self):
        api = RecordingTransport("anthropic", API_RESPONSE)
        cli = RecordingTransport("claude_cli", CLI_RESPONSE)
        prompt = "Analyse ASTRAL.\n\nMD&A: ...\n"

        orchestrator_over(api)._call_api("claude-sonnet-5", prompt, "pass1")
        orchestrator_over(cli)._call_api("claude-sonnet-5", prompt, "pass1")

        assert api.prompts == cli.prompts == [prompt]

    def test_usage_entry_says_which_provider_paid(self):
        orchestrator = orchestrator_over(RecordingTransport("claude_cli", CLI_RESPONSE))

        orchestrator._call_api("claude-sonnet-5", "prompt", "pass1")

        entry = orchestrator._usage_log[0]
        assert entry["provider"] == "claude_cli"
        assert entry["cost_usd"] == pytest.approx(0.0331)
        assert entry["tokens_basis"] == "reported"
        assert entry["cache_read_input_tokens"] == 28083
        assert entry["cache_creation_input_tokens"] == 5349

    def test_api_usage_entry_carries_no_cache_fields(self):
        """Absent, not zero — the API path has nothing to say about them."""
        orchestrator = orchestrator_over(RecordingTransport("anthropic", API_RESPONSE))

        orchestrator._call_api("claude-sonnet-5", "prompt", "pass1")

        entry = orchestrator._usage_log[0]
        assert entry["cost_usd"] is None
        assert "cache_read_input_tokens" not in entry

    def test_transport_failure_returns_the_error_dict_downstream_handles(self):
        class FailingTransport:
            name = "claude_cli"

            def complete(self, model, prompt, max_tokens):
                raise TransportError("claude CLI exited 1: not logged in")

        result = orchestrator_over(FailingTransport())._call_api(
            "claude-sonnet-5", "prompt", "pass2"
        )

        assert result == {
            "error": "claude CLI exited 1: not logged in",
            "pass": "pass2",
        }


class TestUsageSummary:
    def test_actual_cost_wins_over_the_price_table(self):
        orchestrator = orchestrator_over(RecordingTransport("claude_cli", CLI_RESPONSE))
        orchestrator._call_api("claude-sonnet-5", "prompt", "pass1")

        summary = orchestrator.usage_summary()

        assert summary["cost_basis"] == "actual"
        assert summary["provider"] == "claude_cli"
        assert summary["estimated_cost_usd"] == pytest.approx(0.0331)

    def test_the_api_path_still_estimates(self):
        from boundless100x.llm_layer.orchestrator import estimate_cost

        orchestrator = orchestrator_over(RecordingTransport("anthropic", API_RESPONSE))
        orchestrator._call_api("claude-sonnet-5", "prompt", "pass1")

        summary = orchestrator.usage_summary()

        assert summary["cost_basis"] == "estimated"
        assert summary["estimated_cost_usd"] == pytest.approx(
            estimate_cost("claude-sonnet-5", 1200, 800), abs=1e-4
        )

    def test_a_mixed_log_says_mixed(self):
        """One orchestrator cannot switch providers, but a fallback could.

        `cost_basis` states what the number is rather than asserting a purity
        the accounting does not enforce — the same estimate-versus-recorded
        honesty `friction.basis` uses.
        """
        orchestrator = orchestrator_over(RecordingTransport("claude_cli", CLI_RESPONSE))
        orchestrator._call_api("claude-sonnet-5", "prompt", "pass1")
        orchestrator.transport = RecordingTransport("anthropic", API_RESPONSE)
        orchestrator._call_api("claude-sonnet-5", "prompt", "pass2")

        assert orchestrator.usage_summary()["cost_basis"] == "mixed"

    def test_estimated_cost_usd_keeps_its_name(self):
        """The sweep's ceiling meters on this key; renaming it breaks that."""
        orchestrator = orchestrator_over(RecordingTransport("claude_cli", CLI_RESPONSE))

        assert "estimated_cost_usd" in orchestrator.usage_summary()


class BilledFailureTransport:
    """A CLI call that failed *after* the harness prefix was already billed."""

    name = "claude_cli"

    def __init__(self, cost_usd: float | None = 0.0331):
        self.cost_usd = cost_usd

    def complete(self, model, prompt, max_tokens):
        raise TransportError(
            "claude reported an error: Not logged in", cost_usd=self.cost_usd
        )


class TestAFailedCallIsNotAFreeCall:
    """The spend a failure already made has to reach `usage_summary()`.

    Usage was logged only after `transport.complete()` returned, so a call that
    billed and then failed vanished from the accounting entirely — invisible to
    the sweep's `--ceiling` across the dozens of serial calls a full sweep
    makes, and reported as $0.0000 for the ticker that paid it.
    """

    def test_the_billed_failure_reaches_the_usage_log(self):
        orchestrator = orchestrator_over(BilledFailureTransport())

        result = orchestrator._call_api("claude-sonnet-5", "prompt", "pass1")

        assert result == {
            "error": "claude reported an error: Not logged in",
            "pass": "pass1",
        }
        entry = orchestrator._usage_log[0]
        assert entry["failed"] is True
        assert entry["cost_usd"] == pytest.approx(0.0331)
        assert entry["provider"] == "claude_cli"

    def test_the_entry_says_the_token_counts_are_unknown_not_zero(self):
        """Zero reported tokens would be the lie this layer explicitly forbids.

        The call may have moved tens of thousands of tokens before it failed.
        `None` plus `tokens_basis: "unknown"` is the honest reading, in the same
        vocabulary the successful entries use.
        """
        orchestrator = orchestrator_over(BilledFailureTransport())

        orchestrator._call_api("claude-sonnet-5", "prompt", "pass1")

        entry = orchestrator._usage_log[0]
        assert entry["input_tokens"] is None
        assert entry["output_tokens"] is None
        assert entry["tokens_basis"] == TOKENS_BASIS_UNKNOWN

    def test_the_summary_meters_the_spend_and_flags_the_short_totals(self):
        """The half the `--ceiling` reads, and the caveat a reader needs."""
        orchestrator = orchestrator_over(BilledFailureTransport())
        orchestrator._call_api("claude-sonnet-5", "prompt", "pass1")

        summary = orchestrator.usage_summary()

        assert summary["estimated_cost_usd"] == pytest.approx(0.0331)
        assert summary["cost_basis"] == COST_BASIS_ACTUAL
        assert summary["failed_calls"] == 1
        # 0 here means "nothing known", which `failed_calls` is what says.
        assert summary["total_tokens"] == 0

    def test_a_successful_run_carries_no_failed_calls_key(self):
        """Present only when there is something to correct."""
        orchestrator = orchestrator_over(RecordingTransport("claude_cli", CLI_RESPONSE))
        orchestrator._call_api("claude-sonnet-5", "prompt", "pass1")

        assert "failed_calls" not in orchestrator.usage_summary()

    def test_an_unpriceable_failure_invents_neither_a_cost_nor_a_basis(self):
        """A timeout knows nothing, and nothing must not be priced at zero.

        It adds nothing to the total and does not vote on `cost_basis`: calling
        it "estimated" would claim a $0 estimate nobody made, and flipping an
        otherwise-metered run to "mixed" would misdescribe the number beside it.
        `failed_calls` is the honest statement, and it is still made.
        """
        orchestrator = orchestrator_over(RecordingTransport("claude_cli", CLI_RESPONSE))
        orchestrator._call_api("claude-sonnet-5", "prompt", "pass1")
        orchestrator.transport = BilledFailureTransport(cost_usd=None)
        orchestrator._call_api("claude-sonnet-5", "prompt", "pass2")

        summary = orchestrator.usage_summary()

        assert summary["estimated_cost_usd"] == pytest.approx(0.0331)
        assert summary["cost_basis"] == COST_BASIS_ACTUAL
        assert summary["failed_calls"] == 1

    def test_a_log_of_nothing_but_failures_still_summarizes(self):
        """Regression guard: pricing a `None` token count is a TypeError."""
        orchestrator = orchestrator_over(BilledFailureTransport(cost_usd=None))
        orchestrator._call_api("claude-sonnet-5", "prompt", "pass1")

        summary = orchestrator.usage_summary()

        assert summary["estimated_cost_usd"] == 0.0
        assert summary["cost_basis"] == COST_BASIS_ESTIMATED
        assert summary["total_input_tokens"] == 0

    def test_a_failure_after_the_call_succeeded_is_not_double_counted(self):
        """Only `TransportError` logs a failure — the broad catch must not.

        Everything else reaching that handler either never got as far as the
        transport or failed in parsing, *after* the successful entry was already
        appended. A second entry there would count one call twice.
        """
        class ParseBreaker(RecordingTransport):
            def complete(self, model, prompt, max_tokens):
                super().complete(model, prompt, max_tokens)
                return TransportResponse(text=None, input_tokens=1, output_tokens=1)

        orchestrator = orchestrator_over(ParseBreaker("claude_cli", CLI_RESPONSE))
        orchestrator._call_api("claude-sonnet-5", "prompt", "pass1")

        assert len(orchestrator._usage_log) == 1
        assert "failed" not in orchestrator._usage_log[0]


class TestCacheTotalsReachTheReader:
    """Per-call cache counts that no aggregate sums are a defence in name only.

    The envelope's `input_tokens` excludes everything the cache served, so a
    Pass 1 + Pass 2 run that moved ~35K tokens summed to ~1.6K and every surface
    printed that alone. Beside an API-path report's honest 34,000 the CLI path
    read as forty times more token-efficient at twice the price — the exact
    misreading the fields were added to prevent, defeated one layer above where
    they were added.
    """

    def _cli_summary(self, calls: int = 2) -> dict:
        orchestrator = orchestrator_over(RecordingTransport("claude_cli", CLI_RESPONSE))
        for _ in range(calls):
            orchestrator._call_api("claude-sonnet-5", "prompt", "pass1")
        return orchestrator.usage_summary()

    def test_the_summary_totals_both_cache_counts(self):
        summary = self._cli_summary()

        assert summary["total_cache_read_input_tokens"] == 2 * 28083
        assert summary["total_cache_creation_input_tokens"] == 2 * 5349

    def test_the_derived_total_is_what_the_surfaces_render(self):
        """Derived once here rather than three times in three templates.

        The split stays beside it because the halves price differently — Claude
        Code writes cache at 2× standard input and reads it at 0.1×.
        """
        summary = self._cli_summary()

        assert summary["total_cached_input_tokens"] == 2 * (28083 + 5349)

    def test_the_api_path_emits_no_cache_keys_at_all(self):
        """Absence stays distinguishable from zero.

        A `0` would not read as "not applicable" — it would read as "every
        prompt was written fresh", which is the expensive case, not the absent
        one.
        """
        orchestrator = orchestrator_over(RecordingTransport("anthropic", API_RESPONSE))
        orchestrator._call_api("claude-sonnet-5", "prompt", "pass1")

        summary = orchestrator.usage_summary()

        assert "total_cache_read_input_tokens" not in summary
        assert "total_cached_input_tokens" not in summary

    def test_the_per_pass_breakdown_is_kept(self):
        """Aggregating is additive: the per-call counts still travel."""
        summary = self._cli_summary(calls=1)

        assert summary["passes"][0]["cache_read_input_tokens"] == 28083

    def test_the_console_line_states_the_cached_tokens(self, capsys):
        """A bare cache-excluded number with no corrective beside it is the bug."""
        from boundless100x.cli import _print_llm_summary
        from tests.conftest import make_result

        result = make_result()
        result.llm_analysis = {"usage": self._cli_summary()}

        _print_llm_summary(result)

        printed = capsys.readouterr().out
        assert "cached" in printed
        assert f"{2 * (28083 + 5349):,}" in printed


# ── The CLI flag, and the sweep it meters ──────────────────────────────────


class _ConstructedWith(Exception):
    """Raised out of the fake service once it has recorded its config."""


@pytest.fixture
def constructed_config(monkeypatch):
    """Capture the config `analyze`/`sweep` hand to the service, then stop.

    The commands go on to fetch, score and render, none of which this is about
    — recording the injected dict and raising is the whole assertion, and the
    typer runner catches the exception for us.
    """
    import boundless100x.service as service_module

    seen = {}

    class FakeService:
        def __init__(self, config_path=None, config=None):
            seen["config"] = config
            raise _ConstructedWith()

    monkeypatch.setattr(service_module, "Boundless100xService", FakeService)
    return seen


def invoke(*args):
    from typer.testing import CliRunner

    from boundless100x.cli import app

    return CliRunner().invoke(app, list(args))


class TestProviderFlag:
    def test_the_flag_reaches_the_constructed_service(self, constructed_config):
        invoke("analyze", "ASTRAL", "--llm-provider", "claude_cli")

        assert constructed_config["config"]["llm"]["provider"] == "claude_cli"

    def test_without_the_flag_the_config_decides(self, constructed_config):
        invoke("analyze", "ASTRAL")

        assert constructed_config["config"]["llm"]["provider"] == "anthropic"

    def test_the_sweep_carries_it_too(self, constructed_config):
        invoke("sweep", "--tickers", "ASTRAL", "--llm-provider", "claude_cli")

        assert constructed_config["config"]["llm"]["provider"] == "claude_cli"

    def test_an_illegal_value_is_refused_before_anything_runs(
        self, constructed_config
    ):
        result = invoke("analyze", "ASTRAL", "--llm-provider", "openai")

        assert result.exit_code != 0
        assert "config" not in constructed_config

    def test_the_banner_names_the_billing_path(self, constructed_config):
        """A run's billing path belongs in its first line, not its usage block."""
        chosen = invoke("analyze", "ASTRAL", "--llm-provider", "claude_cli")
        default = invoke("analyze", "ASTRAL")

        assert "claude_cli" in chosen.output
        assert "claude_cli" not in default.output


class TestSurfacesStateTheBasis:
    """A real bill must not be rendered as a guess.

    `estimated_cost_usd` keeps its name on both paths, so every surface that
    prints it has to say which kind of number it is — otherwise the CLI path's
    metered dollars appear under the same `~$` that used to mean "priced from
    MODEL_PRICING, give or take".
    """

    def _render(self, tmp_path, usage: dict) -> str:
        from boundless100x.output.report_generator import ReportGenerator
        from tests.conftest import make_result

        result = make_result()
        result.llm_analysis = {"pass2": {}, "usage": usage}
        report_dir = ReportGenerator(output_dir=str(tmp_path)).generate(
            result, formats=["html"]
        )
        return (report_dir / f"{result.ticker}_dashboard.html").read_text()

    def test_an_actual_cost_is_labelled_actual(self, tmp_path):
        html = self._render(
            tmp_path,
            {
                "total_tokens": 30_000,
                "estimated_cost_usd": 0.0662,
                "total_seconds": 12.0,
                "cost_basis": COST_BASIS_ACTUAL,
                "provider": "claude_cli",
            },
        )

        assert f"{COST_BASIS_ACTUAL} $0.0662" in html
        assert "claude_cli" in html

    def test_an_estimate_still_reads_as_one(self, tmp_path):
        html = self._render(
            tmp_path,
            {
                "total_tokens": 30_000,
                "estimated_cost_usd": 0.0662,
                "total_seconds": 12.0,
                "cost_basis": COST_BASIS_ESTIMATED,
                "provider": "anthropic",
            },
        )

        assert f"{COST_BASIS_ESTIMATED} $0.0662" in html

    def test_a_usage_block_from_before_the_seam_still_renders(self, tmp_path):
        """Reports already on disk carry no basis; they are estimates by history."""
        html = self._render(
            tmp_path,
            {"total_tokens": 30_000, "estimated_cost_usd": 0.0662, "total_seconds": 12.0},
        )

        assert f"{COST_BASIS_ESTIMATED} $0.0662" in html

    def test_the_report_shows_the_cached_tokens_beside_the_count(self, tmp_path):
        """`1,604 tokens` alone is the phantom efficiency, in the report itself."""
        html = self._render(
            tmp_path,
            {
                "total_tokens": 1_604,
                "total_cached_input_tokens": 33_432,
                "estimated_cost_usd": 0.0662,
                "total_seconds": 12.0,
                "cost_basis": COST_BASIS_ACTUAL,
                "provider": "claude_cli",
            },
        )

        assert "1604 tokens" in html
        assert "(+33,432 cached)" in html

    def test_the_report_says_when_the_totals_are_short(self, tmp_path):
        """A failed call's tokens are unknown, so the total beside it is partial."""
        html = self._render(
            tmp_path,
            {
                "total_tokens": 1_604,
                "estimated_cost_usd": 0.0662,
                "total_seconds": 12.0,
                "cost_basis": COST_BASIS_ACTUAL,
                "provider": "claude_cli",
                "failed_calls": 2,
            },
        )

        assert "2 failed calls (tokens unknown)" in html

    def test_an_api_path_report_gains_no_cache_clause(self, tmp_path):
        """Nothing to correct, so nothing is said — absence, not a zero."""
        html = self._render(
            tmp_path,
            {
                "total_tokens": 34_000,
                "estimated_cost_usd": 0.0662,
                "total_seconds": 12.0,
                "cost_basis": COST_BASIS_ESTIMATED,
                "provider": "anthropic",
            },
        )

        assert "cached)" not in html
        assert "failed call" not in html


class ActualCostLLM(RecordingLLM):
    """A CLI-path orchestrator stub: every call reports a real metered cost."""

    PER_CALL_USD = 0.05
    # Cache read + creation on one call, at the order of magnitude a real
    # envelope reports — and the reason the stub bothers: the sweep footer reads
    # its token figures off `usage_summary()` deltas, so a stub with no cache
    # counts could not tell whether they reached it.
    PER_CALL_CACHED_TOKENS = 33_432

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._spent = 0.0
        self._cached = 0

    def run_forward_growth_extraction(self, ticker, company_name, submission):
        # Incremented before the failure branch below, deliberately: the harness
        # prefix is billed whether or not the call goes on to succeed.
        self._spent += self.PER_CALL_USD
        self._cached += self.PER_CALL_CACHED_TOKENS
        return super().run_forward_growth_extraction(ticker, company_name, submission)

    def usage_summary(self):
        summary = super().usage_summary()
        summary["estimated_cost_usd"] = round(self._spent, 6)
        summary["cost_basis"] = "actual"
        summary["provider"] = "claude_cli"
        summary["total_cached_input_tokens"] = self._cached
        return summary


def _extraction_response() -> dict:
    """One grounded guidance entry against the shared MD&A fixture."""
    from boundless100x import forward_growth_schema as schema
    from tests.conftest import make_ar_sections

    sentence = "We expect revenue of Rs 1,500 crore in FY2026."
    assert sentence in make_ar_sections()["2025"]["mdna"]["text"]
    return {"years": {"2025": {"guidance": [{
        "metric": "revenue", "target_value": 1500, "target_period": "FY2026",
        "subject": schema.SUBJECT_COMPANY, "unit": schema.UNIT_INR_CR,
        "source_sentence": sentence, "section": "mdna",
    }]}}}


# ── What must not change ───────────────────────────────────────────────────


def _hash_inputs() -> list[dict]:
    """Every payload the two regime hashes are actually computed over.

    Captured from the run rather than re-listed here. A list of inputs copied
    out of `engine.py` into this file would go stale the moment a fifth one is
    added, and the check reading it would keep passing over an input it had
    never seen — which is the same failure mode as the tautology this replaced.

    Both hashes funnel through `ComputeEngine._digest`, so a subclass that
    records what reaches it sees the real payloads, whatever they grow into.
    Subclassing rather than patching keeps the parent class untouched: nothing
    here can leak into another test through a botched restore.
    """
    from boundless100x.compute_engine.engine import ComputeEngine

    class Capturing(ComputeEngine):
        def __init__(self, **kwargs):
            self.captured = []
            super().__init__(**kwargs)

        def _digest(self, payload):
            # Shadows the parent's staticmethod on this subclass only, and
            # still returns the real digest — the engine is otherwise normal.
            self.captured.append(payload)
            return ComputeEngine._digest(payload)

    engine = Capturing(macro={"inflation": 0.05})

    assert len(engine.captured) == 2, (
        "expected one payload per regime hash; `_digest` is no longer the "
        "single funnel, so this check is blind to whatever now bypasses it"
    )
    return engine.captured


def _every_key(node) -> set[str]:
    """Every dict key anywhere in a nested payload, at any depth."""
    if isinstance(node, dict):
        return set(node) | {k for v in node.values() for k in _every_key(v)}
    if isinstance(node, (list, tuple)):
        return {k for item in node for k in _every_key(item)}
    return set()


class TestNothingDownstreamCanTell:
    def test_neither_regime_hash_moves(self):
        """`llm.provider` is a transport, not a scoring or signal regime.

        If it moved `registry_hash`, choosing a billing route would reset every
        ticker's momentum baseline unrecoverably — history is append-only.

        The provider is varied through `Boundless100xService` because that is
        the only constructor that can see it. `ComputeEngine.__init__` takes
        `registry_dir` and `macro` and structurally cannot receive an `llm` key
        at all, so two engines built directly here would be byte-identical
        arguments compared against themselves — determinism dressed up as
        exclusion, green no matter what the wiring did. The wiring is the whole
        risk surface: `service.py`'s single
        `ComputeEngine(macro=self.config.get("macro", {}))` is what decides
        that the `llm` block never reaches a hash input. Widen it to hand over
        the full config and these assertions go red, which is the point.

        The service constructs without credentials on either path — an absent
        API key and an absent `claude` binary both raise `ValueError`, which
        `__init__` catches and degrades to compute-only — and neither fetchers
        nor the engine touch the network at construction.
        """
        from boundless100x.service import Boundless100xService, load_config

        api = Boundless100xService(
            config={**load_config(), "llm": {"provider": "anthropic"}}
        )
        cli = Boundless100xService(
            config={**load_config(), "llm": {"provider": "claude_cli"}}
        )

        assert api.engine.registry_hash == cli.engine.registry_hash
        assert api.engine.forward_signal_hash == cli.engine.forward_signal_hash

    def test_no_hash_input_carries_the_llm_block(self):
        """Stronger than the pair above, and it outlives today's constructor.

        The test above pins the wiring; this one pins the property the wiring
        exists to preserve — that nothing either hash is computed over is keyed
        on the LLM at all. It stays meaningful if `ComputeEngine` is ever given
        the whole config, where a service-level comparison alone would be the
        only thing standing between a provider flip and a dead momentum
        baseline.

        `model` is in the key set and belongs there: it is the extraction model
        id from `_extraction_regime()`, which `forward_signal_hash` carries so
        a row can still say which extraction regime produced the entries a
        forward signal was read from. It resolves from defaults rather than
        from the run's `llm` block, and it is the same id on both transports.
        That is precisely why `provider` is asserted absent beside `llm`:
        `_extraction_regime()` is the one place a transport label could
        plausibly be added by someone reasoning about extraction provenance,
        and adding it there would fragment forward-signal history on a billing
        choice.
        """
        keys = set()
        for payload in _hash_inputs():
            keys |= _every_key(payload)

        assert "llm" not in keys
        assert "provider" not in keys
        # Floor: an empty or shallow walk would make both assertions above
        # vacuously true. These two are the outermost and innermost things the
        # registry hash reads, so their presence says the walk really ran.
        assert {"macro", "element_weights"} <= keys

    def test_the_ceiling_still_binds_when_the_costs_are_actual(
        self, service, corpus
    ):
        """`estimated_cost_usd` holds actuals on the CLI path — and still meters.

        The key keeps its name precisely so this keeps working; what changes is
        what the number *means*, which is why the report now carries the basis
        and the provider beside it.
        """
        from boundless100x.llm_layer import sweep as sweep_module

        service._llm = ActualCostLLM(response=_extraction_response())

        report = sweep_module.sweep(
            service, all_tickers=True, cost_ceiling_usd=0.04
        )

        assert [r["ticker"] for r in report["results"]] == ["ASTRAL"]
        assert report["not_reached"] == ["VBL"]
        assert report["actual"]["cost_basis"] == "actual"
        assert report["actual"]["provider"] == "claude_cli"

    def test_the_sidecar_version_block_has_no_provider(self):
        """A grounded reading stays a grounded reading; transport is not identity.

        A `provider` field here would invalidate every cached extraction on a
        provider switch and re-spend the corpus for nothing.
        """
        from boundless100x.llm_layer.forward_growth import _version_block

        block = _version_block({"2025": {"mdna": "text"}}, "claude-sonnet-5")

        assert "provider" not in block
        assert set(block) == {
            "schema_version",
            "prompt_digest",
            "model",
            "source_digest",
        }


class TestTheSweepReportsWhatWasReallySpent:
    """The footer is the sweep's whole account of a run that cost real money.

    Two things used to be missing from it: the cache counts, without which the
    printed token figures are the envelope's cache-*excluded* ones and cannot be
    reconciled with the dollar figure beside them; and the spend of any call
    that billed and then failed, which never entered `usage_summary()` at all.
    """

    def test_the_cache_deltas_reach_the_footer_dict(self, service, corpus):
        from boundless100x.llm_layer import sweep as sweep_module

        service._llm = ActualCostLLM(response=_extraction_response())

        report = sweep_module.sweep(service, all_tickers=True)

        assert report["actual"]["cached_input_tokens"] == (
            2 * ActualCostLLM.PER_CALL_CACHED_TOKENS
        )

    def test_the_api_path_footer_gains_no_cache_keys(self, service, corpus):
        """Absent rather than zero: the API path does not cache-report."""
        from boundless100x.llm_layer import sweep as sweep_module

        report = sweep_module.sweep(service, all_tickers=True)

        assert "cached_input_tokens" not in report["actual"]

    def test_a_ticker_whose_extraction_failed_still_reports_what_it_billed(
        self, service, corpus
    ):
        """$0.0000 beside a failure is a claim the sweep is in no position to make.

        On the claude_cli path the harness prefix is paid before the model reads
        a word of ours, so a ticker that failed can have spent as much as one
        that succeeded.
        """
        from boundless100x.llm_layer import sweep as sweep_module

        service._llm = ActualCostLLM(
            response=_extraction_response(), fail_on=("ASTRAL",)
        )

        report = sweep_module.sweep(service, tickers=["ASTRAL"])

        assert report["results"][0]["status"] == "failed"
        assert report["results"][0]["cost_usd"] == pytest.approx(
            ActualCostLLM.PER_CALL_USD
        )


def _flat(text: str) -> str:
    """Rendered console output with its wrapping collapsed.

    Rich wraps to the console width, and a caveat that reads correctly on a
    terminal would otherwise fail a substring assertion at whatever column the
    runner happened to break it — a failure about the width, not the words.
    """
    return " ".join(text.split())


@pytest.fixture
def canned_sweep(monkeypatch):
    """Drive the `sweep` command's console output off a fixed report.

    The footer is the code under test; the service and the extraction it wraps
    are not, and standing either up would be scaffolding rather than signal.
    `cli.sweep` imports the sweep module *inside* the function, so the patch
    has to land on that module's own attribute — there is no `sweep_module`
    name on `cli` to replace.

    Returns an installer so each test states only the fields it is about.
    """
    from rich.console import Console

    import boundless100x.llm_layer.sweep as sweep_module
    import boundless100x.service as service_module
    from boundless100x import cli, cli_common

    class FakeService:
        def __init__(self, config_path=None, config=None):
            self.config = config

    monkeypatch.setattr(service_module, "Boundless100xService", FakeService)
    # Every module that imported the console by name holds its own binding.
    wide = Console(width=240)
    for module in (cli, cli_common):
        monkeypatch.setattr(module, "console", wide)

    def install(**overrides) -> dict:
        report = {
            "dry_run": False,
            "plans": [],
            "skipped": [],
            "deferred": [],
            "estimate": {
                "tickers": 2, "input_tokens": 40_000, "usd": 0.79, "usd_max": 1.2,
            },
            "results": [],
            "not_reached": [],
            # Exactly what `sweep()` initialises `actual` to and — on a dry run
            # — returns untouched: no provider, because nothing ran. Tests of
            # the live footer override it with what a finished run reports.
            "actual": {
                "usd": 0.0, "input_tokens": 0, "output_tokens": 0,
                "provider": None, "cost_basis": COST_BASIS_ESTIMATED,
            },
        }
        report.update(overrides)
        monkeypatch.setattr(sweep_module, "sweep", lambda *a, **k: report)
        return report

    return install


class TestTheSweepFooterSaysWhatCLIDollarsAre:
    """Both of the sweep's cost footers, in both provider directions.

    The dry run is the branch that matters most and was the one that could not
    warn. `sweep()` returns before any transport runs, so a dry-run report's
    `actual["provider"]` is `None` — and the caveat was guarded on exactly that
    key, inside the `else` of `if report["dry_run"]`. The one path whose entire
    job is to inform a spending decision was the one path structurally unable
    to qualify the number it was informing it with, while the header banner
    named `claude_cli` beside it and read as confirmation. The provider is now
    resolved from the config, the same value the banner reads.
    """

    def test_a_dry_run_on_the_cli_path_says_the_estimate_is_not_the_bill(
        self, canned_sweep
    ):
        report = canned_sweep(dry_run=True)
        # The precondition that made the old guard unreachable, asserted rather
        # than assumed: if `sweep()` ever starts naming a provider on a dry run,
        # this test stops covering the case it was written for.
        assert report["actual"]["provider"] is None

        out = _flat(invoke(
            "sweep", "--tickers", "ASTRAL", "--dry-run",
            "--llm-provider", "claude_cli",
        ).output)

        assert "MODEL_PRICING is API pricing" in out
        assert "1.7–1.8x" in out
        # The per-call harness floor, which the estimate has no term for at all
        # — a multiplier alone would still understate a many-ticker sweep.
        assert "$0.033 per call" in out
        assert "--ceiling" in out

    def test_a_dry_run_on_the_api_path_adds_no_caveat(self, canned_sweep):
        canned_sweep(dry_run=True)

        out = _flat(invoke("sweep", "--tickers", "ASTRAL", "--dry-run").output)

        assert "Dry run" in out
        assert "1.7–1.8x" not in out
        assert "claude_cli" not in out

    def test_the_live_footer_states_the_basis_provider_and_cache_delta(
        self, canned_sweep
    ):
        canned_sweep(actual={
            "usd": 0.1324, "input_tokens": 1_600, "output_tokens": 2_700,
            "cached_input_tokens": 33_432,
            "provider": LLMProvider.CLAUDE_CLI.value,
            "cost_basis": COST_BASIS_ACTUAL,
        })

        out = _flat(invoke(
            "sweep", "--tickers", "ASTRAL", "--llm-provider", "claude_cli",
        ).output)

        assert f"Actual: $0.1324 ({COST_BASIS_ACTUAL}, {LLMProvider.CLAUDE_CLI.value})" in out
        assert "1,600 in + 2,700 out (+33,432 cached)" in out
        assert "harness overhead" in out

    def test_the_live_footer_on_the_api_path_carries_no_cli_caveat(
        self, canned_sweep
    ):
        canned_sweep(actual={
            "usd": 0.79, "input_tokens": 40_000, "output_tokens": 4_050,
            "provider": LLMProvider.ANTHROPIC.value,
            "cost_basis": COST_BASIS_ESTIMATED,
        })

        out = _flat(invoke("sweep", "--tickers", "ASTRAL").output)

        assert f"Actual: $0.7900 ({COST_BASIS_ESTIMATED}, {LLMProvider.ANTHROPIC.value})" in out
        assert "harness overhead" not in out
        # No cache keys on this path, so no clause — not `(+0 cached)`.
        assert "cached" not in out
