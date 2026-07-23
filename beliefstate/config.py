"""Configuration loading and a light-weight typed accessor.

The whole pipeline is driven by a single ``config.yaml`` so that runs are
reproducible and every experiment shares the same seed, data split, and
thresholds.  We keep the accessor deliberately simple (attribute access over
nested dicts) rather than pulling in a heavy schema library.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict

import yaml

_DEFAULT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")


class _Section:
    """Attribute-style, read-only view over a nested config dict."""

    def __init__(self, data: Dict[str, Any]):
        self._data = data

    def __getattr__(self, key: str) -> Any:
        try:
            value = self._data[key]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(key) from exc
        if isinstance(value, dict):
            return _Section(value)
        return value

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def as_dict(self) -> Dict[str, Any]:
        return dict(self._data)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"_Section({self._data!r})"


@dataclass
class Config:
    """Top-level config object exposing each YAML section as an attribute."""

    raw: Dict[str, Any]

    @property
    def seed(self) -> int:
        return int(self.raw["seed"])

    @property
    def model(self) -> _Section:
        return _Section(self.raw["model"])

    @property
    def data(self) -> _Section:
        return _Section(self.raw["data"])

    @property
    def tasks(self) -> _Section:
        return _Section(self.raw["tasks"])

    @property
    def identify(self) -> _Section:
        return _Section(self.raw["identify"])

    @property
    def intervene(self) -> _Section:
        return _Section(self.raw["intervene"])

    @property
    def output(self) -> _Section:
        return _Section(self.raw["output"])

    def as_dict(self) -> Dict[str, Any]:
        return dict(self.raw)


def load_config(path: str | None = None) -> Config:
    """Load ``config.yaml`` (or a supplied path) into a :class:`Config`."""

    path = path or _DEFAULT_PATH
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return Config(raw=raw)
