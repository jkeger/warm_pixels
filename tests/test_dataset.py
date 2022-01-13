from pathlib import Path

import pytest

from warm_pixels.hst_data import Dataset


@pytest.fixture(
    name="dataset"
)
def make_dataset():
    return Dataset(
        Path(__file__).parent / "dataset"
    )


def test_dataset_attributes(dataset):
    assert dataset.name == "dataset"
    assert len(dataset.images) == 1


def test_image(dataset):
    image, = dataset.images
    assert image.name == "example"
