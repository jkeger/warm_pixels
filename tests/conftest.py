from pathlib import Path

import numpy as np
import pytest
from autoarray.instruments.acs import ImageACS, HeaderACS

from warm_pixels.hst_data import Dataset


@pytest.fixture(
    name="dataset_path"
)
def make_dataset_path():
    return Path(__file__).parent / "dataset"


class MockImage:
    name = "image"

    def __init__(self, array):
        self.array = array

    def load_quadrant(
            self,
            quadrant
    ):
        return self.array


class MockDataset(Dataset):
    # noinspection PyMissingConstructor
    def __init__(
            self,
            images,
            path
    ):
        self.images = images
        self.path = path

    @property
    def _output_path(self):
        return self.path


@pytest.fixture(
    name="image"
)
def make_image():
    image = np.zeros((10, 10))

    image[3, 3] = 10
    image[4, 3] = 20
    image[4, 4] = 30
    image[4, 8] = 40

    return image


@pytest.fixture(
    name="mock_dataset"
)
def make_mock_dataset(image, dataset_path):
    return MockDataset(
        images=[MockImage(
            ImageACS(
                image,
                np.zeros(image.shape),
                header=HeaderACS(
                    header_sci_obj={
                        "DATE-OBS": "2020-01-01",
                        "TIME-OBS": "15:00:00",
                    },
                    header_hdu_obj=None,
                    hdu=None,
                    quadrant_letter="A",
                )
            )
        )],
        path=dataset_path
    )
