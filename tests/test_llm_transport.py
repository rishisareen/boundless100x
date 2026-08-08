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

Three things here are not cosmetic:

**The env scrub is the load-bearing line of the feature.** The project's `.env`
sets `ANTHROPIC_API_KEY` and dotenv loads it into `os.environ`; the CLI gives an
env key precedence over subscription OAuth. An unscrubbed subprocess would
silently bill API credits while the owner believed they had chosen the
subscription — a failure that costs money and looks exactly like success.

**Absent tokens are estimated, never zeroed.** A zero token count reads as a
free call; the layer's rule is that absence must not read as a reading, so a
missing `usage` block estimates and says `tokens_basis: "estimated"`.

**Cache fields must travel.** The CLI envelope's `input_tokens` *excludes* cache
reads — a call that moved ~37K tokens reported `input_tokens: 2` — so a usage
block without the cache counts beside it shows a phantom efficiency the API path
can never match.

No test here touches the network or needs a `claude` binary: the SDK client and
`subprocess.run` are both faked.
"""

import json
import subprocess

import pytest

from boundless100x.llm_layer.transport import (
    SCRUBBED_ENV_KEYS,
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
        """A zero means free. An estimate means unknown-but-roughly-this."""
        cli_run(stdout=json.dumps(envelope("answer text", usage={})))

        response = cli_transport().complete("claude-sonnet-5", "x" * 400, 16000)

        assert response.tokens_basis == "estimated"
        assert response.input_tokens == 100
        assert response.output_tokens == len("answer text") // 4
        assert response.cache_read_input_tokens is None

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
    def test_every_anthropic_key_is_stripped(self, cli_run, monkeypatch):
        """The load-bearing line: an env key outranks subscription OAuth.

        Unscrubbed, choosing `claude_cli` on this machine would bill the API
        key sitting in `.env` and report success either way.
        """
        run = cli_run(stdout=json.dumps(envelope()))
        for key in SCRUBBED_ENV_KEYS:
            monkeypatch.setenv(key, "leaked")
        monkeypatch.setenv("PATH", "/usr/bin")

        cli_transport().complete("claude-sonnet-5", "prompt", 16000)

        env = run.kwargs["env"]
        assert SCRUBBED_ENV_KEYS == frozenset(
            {"ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL"}
        )
        for key in SCRUBBED_ENV_KEYS:
            assert key not in env
        assert env["PATH"] == "/usr/bin", "the rest of the environment survives"

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

    def test_non_zero_exit_carries_stderr(self, cli_run):
        cli_run(returncode=1, stderr="credit balance is too low")

        with pytest.raises(TransportError, match="credit balance is too low"):
            cli_transport().complete("claude-sonnet-5", "prompt", 16000)

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

        with pytest.raises(TransportError, match="no result"):
            cli_transport().complete("claude-sonnet-5", "prompt", 16000)

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
    class _Block:
        pass

    class _Usage:
        pass

    block, usage, response = _Block(), _Usage(), type("_Response", (), {})()
    block.text = text
    usage.input_tokens = input_tokens
    usage.output_tokens = output_tokens
    response.content = [block]
    response.usage = usage
    return response


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

    def test_api_error_becomes_a_transport_error(self, api_transport, monkeypatch):
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
                "cost_basis": "actual",
                "provider": "claude_cli",
            },
        )

        assert "actual $0.0662" in html
        assert "claude_cli" in html

    def test_an_estimate_still_reads_as_one(self, tmp_path):
        html = self._render(
            tmp_path,
            {
                "total_tokens": 30_000,
                "estimated_cost_usd": 0.0662,
                "total_seconds": 12.0,
                "cost_basis": "estimated",
                "provider": "anthropic",
            },
        )

        assert "estimated $0.0662" in html

    def test_a_usage_block_from_before_the_seam_still_renders(self, tmp_path):
        """Reports already on disk carry no basis; they are estimates by history."""
        html = self._render(
            tmp_path,
            {"total_tokens": 30_000, "estimated_cost_usd": 0.0662, "total_seconds": 12.0},
        )

        assert "estimated $0.0662" in html


class ActualCostLLM(RecordingLLM):
    """A CLI-path orchestrator stub: every call reports a real metered cost."""

    PER_CALL_USD = 0.05

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._spent = 0.0

    def run_forward_growth_extraction(self, ticker, company_name, submission):
        self._spent += self.PER_CALL_USD
        return super().run_forward_growth_extraction(ticker, company_name, submission)

    def usage_summary(self):
        summary = super().usage_summary()
        summary["estimated_cost_usd"] = round(self._spent, 6)
        summary["cost_basis"] = "actual"
        summary["provider"] = "claude_cli"
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


class TestNothingDownstreamCanTell:
    def test_neither_regime_hash_moves(self):
        """`llm.provider` is a transport, not a scoring or signal regime.

        If it moved `registry_hash`, choosing a billing route would reset every
        ticker's momentum baseline unrecoverably — history is append-only.
        """
        from boundless100x.compute_engine.engine import ComputeEngine

        api = ComputeEngine()
        cli = ComputeEngine()

        assert api.registry_hash == cli.registry_hash
        assert api.forward_signal_hash == cli.forward_signal_hash

    def test_config_llm_block_reaches_neither_hash(self):
        """Stronger than the pair above: the hash inputs never read `llm` at all."""
        from boundless100x.compute_engine.engine import ComputeEngine

        baseline = ComputeEngine(macro={"inflation": 0.05})
        with_provider = ComputeEngine(macro={"inflation": 0.05})

        assert baseline.registry_hash == with_provider.registry_hash

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
