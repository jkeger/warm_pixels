from pathlib import Path

import numpy as np
import pytest
from autoarray.instruments.acs import ImageACS, HeaderACS

from warm_pixels.hst_data import Dataset, Image

directory = Path(__file__).parent
output_path = directory / "output"
dataset_path = directory / "dataset"


class MockImage(Image):
    name = "image"

    def __init__(self, array):
        self.array = array
        super().__init__(
            path=dataset_path / self.name,
            output_path=output_path,
        )

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
    return np.load(
        str(dataset_path / "array.npy")
    )


@pytest.fixture(
    name="mock_dataset"
)
def make_mock_dataset(image):
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
