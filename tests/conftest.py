from pathlib import Path

import pytest

from gdb.synth.dsbm import DSBMConfig

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def small_cfg() -> DSBMConfig:
    """Purity-test preset (d_edge=32)."""
    return DSBMConfig.from_yaml(ROOT / "configs" / "synth" / "small_bipartite.yaml")


@pytest.fixture(scope="session")
def tiny_cfg() -> DSBMConfig:
    """Speed preset (d_edge=4)."""
    return DSBMConfig.from_yaml(ROOT / "configs" / "synth" / "tiny_fast.yaml")
