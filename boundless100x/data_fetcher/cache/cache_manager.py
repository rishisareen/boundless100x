"""TTL-based local file cache for fetched data."""

import hashlib
import json
import time
from pathlib import Path

import pandas as pd


class CacheManager:
    """File-based cache with per-key TTL.

    Stores DataFrames as CSV and dicts as JSON in a local directory.
    Each entry has a metadata sidecar (.meta) tracking creation time.
    """

    def __init__(self, cache_dir: str = None, ttl_hours: int = 24):
        if cache_dir is None:
            cache_dir = str(Path(__file__).parent / "cached_data")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_hours * 3600

    def _key_to_path(self, key: str) -> Path:
        safe = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / safe

    def _meta_path(self, key: str) -> Path:
        return self._key_to_path(key).with_suffix(".meta")

    def _is_expired(self, key: str) -> bool:
        meta = self._meta_path(key)
        if not meta.exists():
            return True
        with open(meta) as f:
            info = json.load(f)
        return (time.time() - info["created_at"]) > self.ttl_seconds

    def _write_meta(self, key: str, fmt: str) -> None:
        meta = self._meta_path(key)
        with open(meta, "w") as f:
            json.dump({"created_at": time.time(), "format": fmt}, f)

    def _read_meta(self, key: str) -> dict | None:
        meta = self._meta_path(key)
        if not meta.exists():
            return None
        with open(meta) as f:
            return json.load(f)

    def get(self, key: str) -> pd.DataFrame | dict | None:
        """Return cached data or None if missing/expired."""
        if self._is_expired(key):
            return None

        meta = self._read_meta(key)
        if meta is None:
            return None

        base = self._key_to_path(key)
        fmt = meta["format"]

        if fmt == "csv":
            path = base.with_suffix(".csv")
            if path.exists():
                return pd.read_csv(path)
        elif fmt == "json":
            path = base.with_suffix(".json")
            if path.exists():
                with open(path) as f:
                    return json.load(f)

        return None

    def set(self, key: str, data: pd.DataFrame | dict) -> None:
        """Store data in cache."""
        base = self._key_to_path(key)

        if isinstance(data, pd.DataFrame):
            data.to_csv(base.with_suffix(".csv"), index=False)
            self._write_meta(key, "csv")
        elif isinstance(data, dict):
            with open(base.with_suffix(".json"), "w") as f:
                json.dump(data, f, indent=2, default=str)
            self._write_meta(key, "json")
        else:
            raise TypeError(f"Unsupported cache type: {type(data)}")

    def invalidate(self, key: str) -> None:
        """Remove a cached entry."""
        base = self._key_to_path(key)
        for suffix in [".csv", ".json", ".meta"]:
            path = base.with_suffix(suffix)
            if path.exists():
                path.unlink()

    def clear_all(self) -> int:
        """Remove all cached data. Returns count of removed entries."""
        count = 0
        for path in self.cache_dir.iterdir():
            if path.suffix in (".csv", ".json", ".meta"):
                path.unlink()
                count += 1
        return count
