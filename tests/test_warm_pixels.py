import numpy as np
import pytest
from autoarray.instruments.acs import ImageACS, HeaderACS

from warm_pixels.hst_data import Dataset
from warm_pixels.warm_pixels import find_warm_pixels, find_dataset_warm_pixels


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


def test_warm_pixels(
        image
):
    result = find_warm_pixels(
        image=image,
        trail_length=2,
        ignore_bad_columns=False,
    )
    assert len(result) == 1


def test_dataset_warm_pixels(
        dataset_path,
        image
):
    result = find_dataset_warm_pixels(
        MockDataset(
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
        ),
        quadrant="A"
    )

    assert len(result) == 0
