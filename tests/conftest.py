from pathlib import Path

import pytest


@pytest.fixture(
    name="dataset_path"
)
def make_dataset_path():
    return Path(__file__).parent / "dataset"
