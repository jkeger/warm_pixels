from pathlib import Path

import pytest

from warm_pixels.hst_data import Dataset


@pytest.fixture(
    name="dataset_path"
)
def make_dataset_path():
    return Path(__file__).parent / "dataset"


@pytest.fixture(
    name="dataset"
)
def make_dataset(
        dataset_path
):
    return Dataset(
        dataset_path
    )


def test_dataset_attributes(dataset):
    assert dataset.name == "dataset"
    assert len(dataset.images) == 1


def test_image(
        dataset,
        dataset_path
):
    image, = dataset.images
    assert image.name == "example"
    assert image.path == dataset_path / "example_raw.fits"
    assert image.cor_path == dataset_path / "example_raw_cor.fits"
