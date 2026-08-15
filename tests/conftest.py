from __future__ import annotations

from pathlib import Path

import pytest

from margin.config import ProjectConfig, load_config
from margin.data_registry.registry import RegistryTables
from margin.fixtures import build_synthetic_registry


@pytest.fixture(scope="session")
def synthetic_config() -> ProjectConfig:
    return load_config(Path("configs/synthetic.yaml"))


@pytest.fixture(scope="session")
def synthetic_registry(synthetic_config: ProjectConfig) -> RegistryTables:
    return build_synthetic_registry(synthetic_config, domain_count=4)
