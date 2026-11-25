from pathlib import Path

import pytest

from medtrace.config import Settings
from medtrace.pipeline import MedTrace

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def settings(tmp_path) -> Settings:
    return Settings(index_dir=str(tmp_path / "index"), kg_path=str(ROOT / "data/kg/triples.json"))


@pytest.fixture()
def engine(settings) -> MedTrace:
    return MedTrace.build(str(ROOT / "data/sample"), settings)
