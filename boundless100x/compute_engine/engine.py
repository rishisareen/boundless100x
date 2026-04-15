"""Auto-discovery compute engine — loads metrics from YAML, runs Python functions."""

import importlib
import logging
from pathlib import Path

import yaml

from boundless100x.compute_engine.metrics.base import MetricResult
from boundless100x.compute_engine.metrics.validator import validate_registry

logger = logging.getLogger(__name__)


class ComputeEngine:
    """Registry-driven metric computation engine.

    Auto-discovers metric definitions from elements/*.yaml and custom/*.yaml,
    validates them on startup, and runs each registered function against data.
    """

    def __init__(self, registry_dir: str | None = None):
        if registry_dir is None:
            registry_dir = str(Path(__file__).parent / "metrics")
        self.registry_dir = Path(registry_dir)

        self.master = self._load_yaml(self.registry_dir / "registry.yaml")
        self.element_weights = self.master["element_weights"]
        self.metrics = self._discover_metrics()

        # Validate on startup
        errors = validate_registry(self.metrics)
        if errors:
            for e in errors:
                logger.error(f"  REGISTRY ERROR: {e}")
            raise ValueError(f"Registry validation failed: {len(errors)} errors")

        logger.info(
            f"ComputeEngine loaded: {len(self.metrics)} metrics "
            f"across {len(self.element_weights)} elements"
        )

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
            result = func(data, config.get("params", {}))

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
