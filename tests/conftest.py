"""Shared fixtures: a small, fast config forced onto the mock backend."""

from __future__ import annotations

import numpy as np
import pytest

from beliefstate import load_config
from beliefstate.data import DataModule


@pytest.fixture()
def cfg():
    c = load_config()
    # Force the offline, deterministic backend and shrink the corpus so tests
    # are fast while still exercising every code path.
    c.raw["model"]["backend"] = "mock"
    c.raw["data"]["n_firms"] = 12
    c.raw["data"]["periods_per_firm"] = 2
    c.raw["intervene"]["n_task_instances"] = 12
    c.raw["intervene"]["n_random_controls"] = 6
    return c


@pytest.fixture()
def rng(cfg):
    return np.random.default_rng(cfg.seed)


@pytest.fixture()
def corpus(cfg, rng):
    return DataModule(cfg, rng=rng).build()
