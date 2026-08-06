"""Auto-discovery compute engine — loads metrics from YAML, runs Python functions."""

import hashlib
import importlib
import json
import logging
from pathlib import Path

import yaml

from boundless100x.compute_engine.eligibility import effective_gates
from boundless100x.compute_engine.metrics.base import MetricResult
from boundless100x.compute_engine.metrics.validator import validate_registry

logger = logging.getLogger(__name__)


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

        logger.info(
            f"ComputeEngine loaded: {len(self.metrics)} metrics "
            f"across {len(self.element_weights)} elements "
            f"(registry {self._registry_hash})"
        )

    @property
    def registry_hash(self) -> str:
        """Fingerprint of every input that can move a score.

        Score-history rows carry this so trajectory diffs never silently
        compare numbers produced under different scoring regimes: a weight
        change, a threshold edit, a new metric, or a macro assumption would
        otherwise read as fundamental momentum.
        """
        return self._registry_hash

    def _compute_registry_hash(self) -> str:
        """Hash the loaded registry, not the YAML bytes.

        Hashing the assembled state means custom-metric drop-ins are covered
        and cosmetic YAML reformatting is not. Four inputs, each of which can
        change a score: the whole master file (element weights, declared
        gates, history waiver, anything added later), the *effective* gates
        (so a run governed by the code-level defaults is not recorded as an
        empty config), the metric definitions, and the macro assumptions that
        reach every metric as parameter defaults.

        Keys prefixed with `_` are provenance, not semantics — `_source_file`
        changes when a metric moves between files without altering what it
        computes, and fragmenting history on a file rename would be a false
        positive.
        """
        payload = {
            "master": self.master,
            "effective_gates": effective_gates(self.gates),
            "metrics": {
                metric_id: {
                    key: value
                    for key, value in config.items()
                    if not key.startswith("_")
                }
                for metric_id, config in self.metrics.items()
            },
            "macro": self.macro,
        }
        canonical = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()[:12]

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
                error=f"Missing inputs: {missing}",
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
