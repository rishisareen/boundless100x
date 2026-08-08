"""What the pipeline pays, and what it asks for.

Two things drift silently and cost real money when they do: the price table,
which was carrying Opus 3 rates long after the pipeline had moved to Opus 4.6,
and the output ceiling, which now has to accommodate thinking as well as the
answer. Neither has a failing call site — a stale price still returns a
number and a low ceiling still returns a response, just a truncated one.
"""

import pytest
import yaml

from boundless100x.llm_layer.orchestrator import (
    DEFAULT_MAX_TOKENS,
    DEEP_MODEL,
    MODEL_PRICING,
    LLMOrchestrator,
    estimate_cost,
)
from boundless100x.service import DEFAULT_CONFIG_PATH

MILLION = 1_000_000


@pytest.fixture
def shipped_llm_config():
    with open(DEFAULT_CONFIG_PATH) as f:
        return yaml.safe_load(f).get("llm", {})


@pytest.fixture
def orchestrator(monkeypatch):
    def build(llm_config=None):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-used")
        return LLMOrchestrator({"llm": llm_config or {}})

    return build


# ── The price table ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "model,per_input,per_output",
    [
        ("claude-opus-5", 5.0, 25.0),
        ("claude-sonnet-5", 3.0, 15.0),
        ("claude-haiku-4-5", 1.0, 5.0),
        ("claude-fable-5", 10.0, 50.0),
    ],
)
def test_published_rates(model, per_input, per_output):
    """Pinned against the published rates, because nothing else checks them.

    The table read $15/$75 for Opus — a retired generation's price — while the
    pipeline ran Opus 4.6 at $5/$25, so the sweep's ceiling metered against a
    threefold overestimate and `--deep` was documented at three times its real
    cost. A wrong price is invisible: it still returns a plausible number.
    """
    assert estimate_cost(model, MILLION, 0) == pytest.approx(per_input)
    assert estimate_cost(model, 0, MILLION) == pytest.approx(per_output)


def test_every_configured_model_is_priced(shipped_llm_config):
    """An unpriced model prints a ceiling and enforces nothing.

    `estimate_cost` returns 0.0 for an id it does not recognise — deliberately,
    so an invented price never reads as a real estimate. The sweep meters its
    cost ceiling on that same number, so a config naming a model absent from
    the table would sweep the corpus unbounded.
    """
    configured = {
        model
        for key, model in shipped_llm_config.items()
        if key.endswith("_model") and isinstance(model, str)
    }
    assert configured, "config names no models — the lookup below proves nothing"

    unpriced = {m for m in configured if estimate_cost(m, MILLION, MILLION) == 0.0}
    assert not unpriced, (
        f"{sorted(unpriced)} priced at zero. Add the family to MODEL_PRICING — "
        "the sweep's --ceiling meters on this number."
    )


def test_an_unknown_model_still_prices_at_zero():
    """The absence has to stay loud rather than become a guess."""
    assert estimate_cost("some-model-nobody-has-priced", MILLION, MILLION) == 0.0
    assert estimate_cost(None, MILLION, MILLION) == 0.0


# ── The output ceiling ─────────────────────────────────────────────────────


def test_the_shipped_ceiling_leaves_room_for_thinking(shipped_llm_config):
    """Thinking is on by default on the Claude 5 family and bills against this.

    A ceiling sized for the JSON alone truncates the object mid-write. The
    parser's repair path would catch that, but a repaired object is a guess at
    what the model meant to say.
    """
    assert shipped_llm_config["max_tokens"] >= DEFAULT_MAX_TOKENS


def test_deep_mode_does_not_shrink_the_ceiling(orchestrator):
    """It used to set 4000 against a configured 4096 — a deeper model with
    less room to answer in than the shallow one it replaced."""
    llm = orchestrator({"max_tokens": 16000})

    llm.use_deep_models()

    assert llm.max_tokens == 16000


def test_the_ceiling_survives_the_round_trip(orchestrator):
    llm = orchestrator({"max_tokens": 16000})

    llm.use_deep_models()
    llm.use_configured_models()

    assert llm.max_tokens == 16000


# ── Which model each mode reaches for ──────────────────────────────────────


def test_deep_mode_uses_the_configured_deep_model(orchestrator):
    llm = orchestrator(
        {
            "pass1_model": "shallow-model",
            "pass2_model": "shallow-model",
            "forward_growth_model": "shallow-model",
            "deep_model": "configured-deep-model",
        }
    )

    llm.use_deep_models()

    assert llm.pass1_model == "configured-deep-model"
    assert llm.pass2_model == "configured-deep-model"
    assert llm.forward_growth_model == "configured-deep-model"


def test_deep_mode_falls_back_to_the_module_default(orchestrator):
    """A config with no `deep_model` must still have somewhere to go."""
    llm = orchestrator({"pass1_model": "shallow-model"})

    llm.use_deep_models()

    assert llm.pass1_model == DEEP_MODEL


def test_configured_models_are_restored(orchestrator):
    """The service is reusable, so one --deep run must not make every later
    run deep as well."""
    llm = orchestrator(
        {
            "pass1_model": "shallow-model",
            "pass2_model": "shallow-model",
            "forward_growth_model": "extraction-model",
        }
    )

    llm.use_deep_models()
    llm.use_configured_models()

    assert llm.pass1_model == "shallow-model"
    assert llm.pass2_model == "shallow-model"
    assert llm.forward_growth_model == "extraction-model"


def test_the_extraction_model_is_pinned_to_its_cached_generation(
    shipped_llm_config,
):
    """Its id is part of the sidecar version block.

    Moving it invalidates every cached extraction in `raw_data/` and the next
    sweep re-extracts the corpus at cost, so it is deliberately left behind
    the passes rather than moved with them. This test is a speed bump, not a
    prohibition: change it in the same commit that prices the re-extraction.
    """
    assert shipped_llm_config["forward_growth_model"] == "claude-sonnet-4-6"


def test_the_families_the_table_prices_are_distinguishable():
    """Matching is a substring test over the model id, so two families that
    could both match one id would resolve by dict order and price by luck."""
    for family in MODEL_PRICING:
        others = [f for f in MODEL_PRICING if f != family]
        assert not any(family in other or other in family for other in others)
