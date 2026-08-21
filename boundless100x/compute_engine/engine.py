"""Auto-discovery compute engine — loads metrics from YAML, runs Python functions."""

import hashlib
import importlib
import json
import logging
from pathlib import Path

import yaml

from boundless100x import forward_growth_schema
from boundless100x.compute_engine.eligibility import effective_gates
from boundless100x.compute_engine.metrics.base import MetricResult
from boundless100x.compute_engine.metrics.validator import validate_registry
from boundless100x.compute_engine.sector import load_sector_applicability

logger = logging.getLogger(__name__)


def _extraction_regime() -> dict:
    """Prompt digest and model id, or a stated unavailability.

    Deliberately tolerant: the hash must stay computable on a machine with no
    API key and no prompt file, because scoring runs offline. An unreadable
    regime hashes as a named constant rather than raising or, worse, silently
    hashing as absent.
    """
    try:
        from boundless100x.llm_layer import forward_growth
        from boundless100x.llm_layer.orchestrator import forward_growth_model

        return {
            "prompt_digest": forward_growth.prompt_digest(),
            "model": forward_growth_model({}),
        }
    except Exception:
        return {"prompt_digest": "unavailable", "model": "unavailable"}


class ComputeEngine:
    """Registry-driven metric computation engine.

    Auto-discovers metric definitions from elements/*.yaml and custom/*.yaml,
    validates them on startup, and runs each registered function against data.
    """

    def __init__(self, registry_dir: str | None = None, macro: dict | None = None):
        if registry_dir is None:
            registry_dir = str(Path(__file__).parent / "metrics")
        self.registry_dir = Path(registry_dir)
        # Shared macro assumptions (inflation, G-Sec yield, discount rate) reach
        # metrics as parameter defaults; a metric's own YAML params still win.
        self.macro = macro or {}

        self.master = self._load_yaml(self.registry_dir / "registry.yaml")
        self.element_weights = self.master["element_weights"]
        self.gates = self.master.get("eligibility_gates", {})
        self.metrics = self._discover_metrics()

        # Validate on startup
        errors = validate_registry(self.metrics)
        if errors:
            for e in errors:
                logger.error(f"  REGISTRY ERROR: {e}")
            raise ValueError(f"Registry validation failed: {len(errors)} errors")

        self._registry_hash = self._compute_registry_hash()
        self._forward_signal_hash = self._compute_forward_signal_hash()

        logger.info(
            f"ComputeEngine loaded: {len(self.metrics)} metrics "
            f"across {len(self.element_weights)} elements "
            f"(registry {self._registry_hash}, "
            f"forward signals {self._forward_signal_hash})"
        )

    @property
    def registry_hash(self) -> str:
        """Fingerprint of every input that can move a score.

        Score-history rows carry this so trajectory diffs never silently
        compare numbers produced under different scoring regimes: a weight
        change, a threshold edit, a new scored metric, or a macro assumption
        would otherwise read as fundamental momentum.

        Zero-weight metrics are deliberately absent (KTD8). The scorer's
        `weight == 0` branch `continue`s before weighted accumulation, so such
        a metric contributes nothing to an element mean, the composite, or the
        coverage denominator — it provably cannot move a score, and so has no
        business in the hash that describes scoring. Keeping it here would
        make Phase 5 circular: it needs trajectory evidence to calibrate the
        forward signals, and calibrating one would reset every ticker's
        baseline — unrecoverably, since history is append-only.
        """
        return self._registry_hash

    @property
    def forward_signal_hash(self) -> str:
        """Fingerprint of the zero-weight forward-signal regime.

        The other half of the split: the definitions of metrics that carry no
        weight, plus the extraction schema they read. Carried on score-history
        rows beside `registry_hash` so a later reader can still tell which
        forward-signal regime produced a row, without that regime being able
        to reset the momentum baseline.
        """
        return self._forward_signal_hash

    def _scored(self, config: dict) -> bool:
        """Whether a metric's weight lets it reach a composite at all."""
        weight = (config.get("scoring") or {}).get("weight", 0) or 0
        return weight > 0

    # Keys a metric may carry that provably cannot move a score, and so must
    # not reach either regime hash. Two different reasons, both ending here:
    #
    #   `_`-prefixed — **provenance, not semantics**. `_source_file` changes
    #   when a metric moves between files without altering what it computes,
    #   and fragmenting history on a file rename would be a false positive.
    #
    #   `presentation` — **score-inert display data**. R11 puts each metric's
    #   unit, direction of goodness and interpretation bands beside its scoring
    #   config, in this same payload. Nothing about how a number is rendered
    #   can change the number, so hashing it would record a regime change that
    #   never happened — and score history is append-only, so the resulting
    #   fragmentation could never be repaired.
    #
    # Adding to this set is a claim that the key cannot affect a computed
    # value. Verify that before you make it: the cost of being wrong is a
    # regime change no row is stamped with.
    HASH_EXEMPT_KEYS = frozenset({"presentation"})

    @classmethod
    def _is_hashed_key(cls, key: str) -> bool:
        return not key.startswith("_") and key not in cls.HASH_EXEMPT_KEYS

    def _metric_definitions(self, scored: bool) -> dict:
        """Metric definitions on one side of the weight split, minus the exempt keys.

        See `HASH_EXEMPT_KEYS` for what is excluded and why.
        """
        return {
            metric_id: {
                key: value
                for key, value in config.items()
                if self._is_hashed_key(key)
            }
            for metric_id, config in self.metrics.items()
            if self._scored(config) is scored
        }

    @staticmethod
    def _digest(payload: dict) -> str:
        canonical = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()[:12]

    def _compute_registry_hash(self) -> str:
        """Hash the loaded registry, not the YAML bytes.

        Hashing the assembled state means custom-metric drop-ins are covered
        and cosmetic YAML reformatting is not. Five inputs, each of which can
        change a score: the whole master file (element weights, declared
        gates, history waiver, anything added later), the *effective* gates
        (so a run governed by the code-level defaults is not recorded as an
        empty config), the definitions of the *scored* metrics, the macro
        assumptions that reach every metric as parameter defaults, and the
        sector applicability table.

        **The applicability table earned its place here by changing job.** It
        was display data while only the report read it, and display data is
        hash-exempt by the same rule that keeps `presentation:` out (see
        `HASH_EXEMPT_KEYS`). Now the scorer consults it, so adding one line to
        it withdraws a metric from a composite — a regime change in every
        sense that matters, and one that would otherwise have been recorded
        under the *unchanged* hash of the regime it replaced. Momentum groups
        on this hash, and score history is append-only: a silent change here
        would have produced diffs across two scoring regimes with nothing
        anywhere able to tell them apart.
        """
        return self._digest({
            "master": self.master,
            "effective_gates": effective_gates(self.gates),
            "metrics": self._metric_definitions(scored=True),
            "macro": self.macro,
            "sector_applicability": load_sector_applicability(),
        })

    def _compute_forward_signal_hash(self) -> str:
        """Hash the regime the zero-weight signals are produced under.

        Element weights and gates are absent because a zero-weight metric
        never consults them. Macro is present in *both* hashes and that is
        correct rather than a leak: it reaches every metric as a parameter
        default, so a discount-rate change genuinely moves a composite and a
        forward signal alike.
        """
        # The **effective extraction regime**, not just the schema. A sidecar
        # invalidates on the prompt digest and the model id as well as the
        # schema, so a row hashed without them cannot say which regime produced
        # the entries a forward signal was read from — two runs either side of a
        # prompt rewrite would carry the same label. Imported lazily: this is
        # `compute_engine`, and a module-level `llm_layer` import would invert
        # the seam KTD2 rests on. Only `forward_signal_hash` moves, so no
        # ticker's momentum baseline is touched.
        return self._digest({
            "metrics": self._metric_definitions(scored=False),
            "extraction_schema": forward_growth_schema.schema_fingerprint(),
            "extraction_prompt": _extraction_regime(),
            "macro": self.macro,
        })

    def _discover_metrics(self) -> dict:
        """Auto-discover all metric definitions from elements/ and custom/ dirs."""
        all_metrics = {}

        for subdir in ["elements", "custom"]:
            scan_dir = self.registry_dir / subdir
            if not scan_dir.exists():
                continue
            for yaml_file in sorted(scan_dir.glob("*.yaml")):
                config = self._load_yaml(yaml_file)
                element_name = config.get("element", "custom")
                for metric_id, metric_def in config.get("metrics", {}).items():
                    if metric_id in all_metrics:
                        # Silently overwriting would drop a scored metric and
                        # shift the element's weight normalisation unnoticed.
                        raise ValueError(
                            f"Duplicate metric id '{metric_id}' in {yaml_file.name}; "
                            f"already defined by {all_metrics[metric_id]['_source_file']}"
                        )
                    metric_def["element"] = element_name
                    metric_def["_source_file"] = yaml_file.name
                    all_metrics[metric_id] = metric_def

        return all_metrics

    def run_all(self, data: dict) -> dict[str, MetricResult]:
        """Run every registered metric against the provided data."""
        results = {}
        for metric_id, config in self.metrics.items():
            results[metric_id] = self._run_metric(metric_id, config, data)
        return results

    def run_element(self, element: str, data: dict) -> dict[str, MetricResult]:
        """Run only metrics belonging to a specific SQGLP element."""
        return {
            mid: self._run_metric(mid, cfg, data)
            for mid, cfg in self.metrics.items()
            if cfg["element"] == element
        }

    def _run_metric(
        self, metric_id: str, config: dict, data: dict
    ) -> MetricResult:
        """Run a single metric function."""
        required = set(config.get("inputs", []))
        available = set(data.keys())

        # Check required inputs are present and non-empty
        missing = set()
        for req in required:
            if req not in available:
                missing.add(req)
            else:
                val = data[req]
                # Allow dicts (metadata, analyst_coverage) even if "empty"
                if hasattr(val, "empty") and val.empty:
                    missing.add(req)

        if missing:
            return MetricResult(
                # Rendered, not repr'd: this string reaches the report as the
                # reason a signal is unknown, and a reader should not have to
                # decode a Python set literal to learn what was not fetched.
                error=f"Missing input(s): {', '.join(sorted(missing))}",
                metadata={"metric_id": metric_id},
            )

        try:
            module_path = f"boundless100x.compute_engine.metrics.{config['module']}"
            module = importlib.import_module(module_path)
            func = getattr(module, config["function"])
            params = {**self.macro, **config.get("params", {})}
            result = func(data, params)

            if not isinstance(result, MetricResult):
                return MetricResult(
                    error=f"Function returned {type(result).__name__}, expected MetricResult"
                )

            result.metadata["metric_id"] = metric_id
            return result

        except Exception as e:
            logger.warning(f"Metric {metric_id} failed: {e}")
            return MetricResult(
                error=str(e),
                metadata={"metric_id": metric_id},
            )

    def get_metrics_by_element(self) -> dict[str, list[str]]:
        """Return metric IDs grouped by element."""
        by_element: dict[str, list[str]] = {}
        for mid, cfg in self.metrics.items():
            el = cfg["element"]
            by_element.setdefault(el, []).append(mid)
        return by_element

    def _load_yaml(self, path: Path) -> dict:
        with open(path) as f:
            return yaml.safe_load(f)
